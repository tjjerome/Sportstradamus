"""Unit tests for the Operation Ship 75 strategy sweep (``training.model_strategy.sweep``).

Two layers are covered. The per-corner *primitive* ``_score_corner`` is the honest scorer: it loads
one trained deterministic dump and runs the production :func:`scorecard.gate_row` on its served
(validation-fit-calibrated) predictive — no test re-fit; the heavy dump load + scorecard gate are
monkeypatched so the test pins only the plumbing. The *orchestration* (conditional-TPE search,
board assembly, verdict formatting) is exercised with the heavy per-corner ``meditate`` train+score
monkeypatched out, so no model trains: one budgeted study per cell samples the enrolled families'
corners, scores each by the honest gate, and ranks the board by ship slack.

The search is budgeted, so no test may assume a cell's whole grid is enumerated. What is pinned is
the contract around it: which families enroll, how many corners are reachable, that the seed and
incumbent corners are evaluated first, that an admissible prior row is reused instead of retrained,
and that a legacy row scored on the ship holdout is inert.
"""

import json
import pathlib
import subprocess

import click
import pandas as pd
import pytest

from sportstradamus.training.model_strategy import sweep, tpe_search
from sportstradamus.training.baselines import get_target_normalization
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.model_strategy import (
    MODEL_STRATEGY_MODEL_KEY,
    SPLIT_FINGERPRINT_CSV_COLUMN,
    InactiveStrategyArtifactError,
    artifact_identity_columns,
    build_artifact_identity,
    validate_strategy_artifacts,
)
from sportstradamus.training.model_strategy import validate_strategy_frame
from sportstradamus.training.model_strategy import (
    BASE_STRUCTURAL_STRATEGY,
    CAP_SERVE,
    SWEEP_CAPABILITIES,
    CellContext,
    controls_json,
    corner_fingerprint,
    distribution_class,
    get_strategy,
    registered_strategies,
    strategies_for_cell,
    strategy_controls,
    validate_strategy_selection,
)
from sportstradamus.training.model_strategy import resolve_report_identity
from sportstradamus.training.model_strategy.specs import SEED_CORNERS
from sportstradamus.training.posthoc import CDF_STAGE, POSTHOC_SLUGS, STRUCTURAL_STAGE
from sportstradamus.training.role_specs import role_spec_for
from sportstradamus.training.structural_strategies import (
    AFFINE_STRATEGY as RUSHING,
)
from sportstradamus.training.structural_strategies import (
    TWO_PART_STRATEGY as RECEIVING,
)

_MATRIX_SHA = "matrix-123"
# The role×position two-part strategy is gated by the per-(league, market) role registry;
# the NFL rushing-affine strategy is cell-pinned. A structural spec only *enrolls* for a cell whose
# real matrix carries its grouping columns — the sweep then excludes it anyway, but the registry
# tests still need the enrolled shape. These fixtures give each registered cell those columns.
_STRUCTURAL_CELL_COLUMNS = {
    ("NFL", "receiving yards"): frozenset(role_spec_for("NFL", "receiving yards").all_columns),
    ("NFL", "rushing yards"): frozenset(role_spec_for("NFL", "rushing yards").all_columns),
    ("NBA", "PTS"): frozenset(role_spec_for("NBA", "PTS").all_columns),
}


# A low-mean integer target — the shape that admits every registered base family, so a test that
# doesn't care about admission gets all of them.
_ADMITS_EVERY_FAMILY = (True, 5.0)


@pytest.fixture(autouse=True)
def _fixed_matrix_contract(monkeypatch):
    monkeypatch.setattr(
        sweep,
        "_training_matrix_contract",
        lambda league, market: (
            _STRUCTURAL_CELL_COLUMNS.get((league, market), frozenset()),
            _MATRIX_SHA,
            *_ADMITS_EVERY_FAMILY,
        ),
    )


@pytest.fixture(autouse=True)
def _sandboxed_study_journal(monkeypatch, tmp_path):
    """Point every cell study's Optuna journal at a per-test directory.

    Without this a test that reaches :func:`tpe_search.cell_study` writes into ``research/optuna/``
    and the next run resumes that journal — cross-test state, and a suggestion sequence that depends
    on which tests ran before.

    Patch the module the function lives in, not the one that imports it: ``cell_study`` resolves
    ``_STUDY_ROOT`` as a bare global in ``tpe_search``, so patching it on ``sweep`` would detach
    silently and let real journals through.
    """
    monkeypatch.setattr(tpe_search, "_STUDY_ROOT", tmp_path / "optuna")


@pytest.fixture(autouse=True)
def _sandboxed_nominee_ledger(monkeypatch, tmp_path):
    """Point the confirm ledger at a per-test path so no test reads the real research/ CSV.

    ``search_cell`` now prices its board against the ledger, so without this every orchestration
    test would read whatever ``research/confirm_nominee_gates.csv`` the box happens to carry.
    """
    monkeypatch.setattr(sweep, "NOMINEE_LEDGER_PATH", tmp_path / "confirm_nominee_gates.csv")


def _canned_row(*, ship, g4_pit_ks):
    """A scorecard ship-row where Gate 4 binds the slack (the other gates pass with headroom)."""
    return {
        "ship": ship,
        "g1_brier_diff_ci_hi": -0.01,
        "g1_brier_skill_score": 0.04,
        "g1_pass": True,
        "g2_star_z": 0.1,
        "g2_pass": True,
        "g3_bench_z": 0.2,
        "g3_pass": True,
        "g4_pit_ks": g4_pit_ks,
        "g4_pit_ks_max": 0.05,
        "g4_pass": g4_pit_ks < 0.05,
        "g5_ece_debiased": 0.03,
        "g5_pass": True,
        "g6_pass": True,
        "central50_coverage": 0.49,
        "n_rows": 1500,
    }


def _fake_row(family, corner, slack):
    """A board row for one scored corner — the shape ``_run_and_score`` returns."""
    return {
        **corner,
        "family": family,
        "slack": slack,
        "ships": slack > 0,
        "g1_pass": True,
        "g1_brier_diff_ci_hi": -0.01,
        "g1_brier_skill": 0.03,
        "g2_pass": True,
        "g2_star_z": 0.1,
        "g3_pass": True,
        "g3_bench_z": 0.2,
        "g4_pass": True,
        "g4_pit_ks": 0.04,
        "g4_pit_ks_max": 0.05,
        "g5_pass": True,
        "g5_ece_debiased": 0.03,
        "g6_pass": True,
        "central50_coverage": 0.49,
        "dispersion_cal": 1.0,
        "skew_cal": 0.0,
        "n": 2000,
    }


def _fake_run_and_score(league, market, family, corner):
    """One honest row per corner; slack favors a known best corner per family (no model trains)."""
    # Every family sweeps posthoc, so it must move slack or the grid's best corner is a tie.
    posthoc = 0.03 if corner["posthoc"] == "roe_mean" else 0.0
    if family == "SkewNormal":
        slack = (
            {
                "ratio_meanyr": 0.0,
                "centered_additive_mean10": 0.20,
                "centered_additive_eb_meanyr_k10": -0.10,
                "ratio_projvol": -0.05,
            }[corner["normalization"]]
            + (0.05 if corner["dist_training_loss"] == "nll" else 0.0)
            + (0.01 if corner["sn_param"] == "centered" else 0.0)
            + (0.02 if corner["blending_loss_fn"] == "crps" else 0.0)
            + posthoc
        )
    elif family == "ZINB":
        slack = (
            {"joint": 0.0, "hurdle": 0.15}[corner["zinb_mode"]]
            + (0.05 if corner["count_dispersion_objective"] == "pit_ks" else 0.0)
            + (0.02 if corner["blending_loss_fn"] == "crps" else 0.0)
            + posthoc
        )
    else:  # NegBin — kept below ZINB's best so cross-family ranking has an unambiguous winner
        slack = (
            (0.05 if corner["count_dispersion_objective"] == "pit_ks" else 0.0)
            + (0.02 if corner["blending_loss_fn"] == "crps" else 0.0)
            + posthoc
        )
    spec = get_strategy(family)
    legacy = (
        {"validation_audit": {"split_fingerprint_sha256": "split-123"}}
        if spec.is_structural
        else None
    )
    identity = build_artifact_identity(
        spec.slug, league, market, corner, legacy, matrix_hash=_MATRIX_SHA
    )
    return {
        **_fake_row(spec.family, corner, slack),
        "strategy_slug": identity.strategy_slug,
        "structural_strategy": identity.structural_strategy,
        "strategy_signature": identity.signature,
        "strategy_implementation_version": identity.implementation_version,
        "artifact_schema_version": identity.artifact_schema_version,
        "strategy_status": identity.status,
        "controls_json": controls_json(corner),
        "corner_fingerprint": identity.corner_fingerprint,
        "matrix_hash": _MATRIX_SHA,
        "split_fingerprint": identity.split_fingerprint,
        "eval_split": sweep.EVAL_SPLIT_CROSSFIT,
    }


def _spy_run_and_score(trained):
    """``_run_and_score`` stand-in that records ``(family, corner)`` per trial before scoring it."""

    def run_and_score(league, market, family, corner):
        trained.append((family, dict(corner)))
        return _fake_run_and_score(league, market, family, corner)

    return run_and_score


def _artifact(spec, league, market, controls, *, status="active"):
    legacy = None
    if spec.is_structural:
        legacy = {
            "slug": spec.slug,
            "schema_version": spec.artifact_schema_version,
            "status": status,
            "validation_audit": {"split_fingerprint_sha256": "split-123"},
        }
    identity = build_artifact_identity(
        spec.slug, league, market, controls, legacy, matrix_hash=_MATRIX_SHA
    )
    normalization = controls.get("normalization")
    model = {
        MODEL_STRATEGY_MODEL_KEY: identity.as_model_blob(),
        "dispersion_cal": 1.0,
        "distribution": controls["dist"],
        "target_normalization": normalization or "none",
        "normalized": bool(
            normalization
            and get_target_normalization(normalization).start_mode_flag == "normalized"
        ),
        **{
            control: controls[control]
            for control in ("sn_param", "zinb_mode", "posthoc")
            if control in controls
        },
        **spec.fixed_persist,
    }
    if spec.legacy_model_key:
        model[spec.legacy_model_key] = legacy
    columns = {
        name: [value, value]
        for name, value in artifact_identity_columns(identity.as_model_blob()).items()
    }
    for name in spec.required_columns:
        columns.setdefault(name, [spec.slug, spec.slug])
    return pd.DataFrame(columns), model


# --- family registry -------------------------------------------------------------------------


def test_family_registry_grids_and_persist_maps():
    """Base families and both registered NFL-yards methods expose reproducible contracts."""
    import math

    sn = get_strategy("SkewNormal")
    mix = get_strategy("Mixture")
    zinb = get_strategy("ZINB")
    negbin = get_strategy("NegBin")
    dpo = get_strategy("DPO")
    receiving = get_strategy(RECEIVING)
    rushing = get_strategy(RUSHING)
    # SN: 4 norms × 2 dist-loss × 2 sn-param × 2 blend × 6 posthoc.
    assert math.prod(len(v) for v in sn.axes.values()) == 192
    # Mixture: 1 dist × 4 norms × 2 blend × 5 posthoc — no loss axis (lightgbmlss rejects crps
    # components, so the pinned nll is the only trainable value).
    assert math.prod(len(v) for v in mix.axes.values()) == 40
    assert mix.fixed_controls["dist_training_loss"] == "nll"
    # 1 dist × 2 mode × 2 disp × 2 blend × 5 posthoc.
    assert math.prod(len(v) for v in zinb.axes.values()) == 40
    assert math.prod(len(v) for v in negbin.axes.values()) == 20  # 1 dist × 2 disp × 2 blend × 5
    assert math.prod(len(v) for v in dpo.axes.values()) == 20
    assert sn.persist == {
        "dist": "dist",
        "normalization": "target_normalization",
        "dist_training_loss": "dist_training_loss",
        "sn_param": "sn_param",
        "blending_loss_fn": "blending",
        "posthoc": "posthoc",
    }
    assert mix.axes["dist"] == ("Mixture",)
    assert mix.axes["normalization"] == sn.axes["normalization"]
    assert mix.persist == {
        "dist": "dist",
        "normalization": "target_normalization",
        "dist_training_loss": "dist_training_loss",
        "blending_loss_fn": "blending",
        "posthoc": "posthoc",
    }
    # Count families persist their single-choice dist so a winner's dist writes to stat_meta.
    assert zinb.persist == {
        "dist": "dist",
        "zinb_mode": "zinb_mode",
        "count_dispersion_objective": "count_dispersion_objective",
        "blending_loss_fn": "blending",
        "posthoc": "posthoc",
    }
    assert negbin.persist == {
        "dist": "dist",
        "count_dispersion_objective": "count_dispersion_objective",
        "blending_loss_fn": "blending",
        "posthoc": "posthoc",
    }
    assert dpo.persist == negbin.persist  # same gate-free count-family persist surface
    assert zinb.axes["dist"] == ("ZINB",) and negbin.axes["dist"] == ("NegBin",)
    assert dpo.axes["dist"] == ("DPO",)
    # A count corner can now win on a cell whose stat_meta carries a continuous normalization, so
    # the family pins target_normalization=none or ship_config._validate_cell rejects the ship.
    for count_spec in (zinb, negbin, dpo):
        assert count_spec.fixed_persist == {"target_normalization": "none"}
    # Neither knob belongs to a base corner, because neither is decided by the corner: the cli
    # flips hpo_selection to 'calibrated' after a Gate-4-only failure, and the pipeline silently
    # upgrades stabilization to L2 for centered SkewNormal and Mixture. Signing either would
    # claim a value the run did not train with. A structural spec pins the whole recipe instead.
    for spec in (sn, mix, zinb, negbin, dpo):
        for knob in ("hpo_selection", "stabilization"):
            assert knob not in spec.fixed_controls and knob not in spec.axes
    for method, slug in (
        (receiving, RECEIVING),
        (rushing, RUSHING),
    ):
        assert method.slug == slug and method.axes == {}
        # The method rides the single-valued ``posthoc`` calibration pool: its slug IS the
        # ``posthoc`` fixed control (never "none", a different pool member), and ``posthoc``
        # persists so a confirm writes it to the cell's stat_meta.
        assert method.fixed_controls == {
            "dist": "SkewNormal",
            "normalization": "ratio_meanyr",
            "dist_training_loss": "crps",
            "sn_param": "direct",
            "blending_loss_fn": "nll",
            "hpo_selection": "loss",
            "stabilization": "None",
            "posthoc": slug,
        }
        assert method.persist == {
            "dist": "dist",
            "normalization": "target_normalization",
            "dist_training_loss": "dist_training_loss",
            "sn_param": "sn_param",
            "blending_loss_fn": "blending",
            "hpo_selection": "hpo_selection",
            "posthoc": "posthoc",
        }
        assert method.fixed_persist == {}
        assert method.is_structural
        assert len(method.canonical_signature) == 64
    # Every swept axis persists — a shipping corner is always reproducible from stat_meta alone
    # (S4: dist_training_loss gained a stat_meta field, deleting the ranks-only corner concept).
    for spec in (sn, mix, zinb, negbin, dpo):
        assert set(spec.persist) == set(spec.axes) | set(spec.fixed_controls)


def test_posthoc_pool_is_per_family_and_never_offers_a_structural_slug():
    """Every base family sweeps the light-corrector pool; only SkewNormal also gets the whole-CDF
    slug, because the pipeline gates that stage on ``dist == "SkewNormal"`` — on any other family
    the corner falls through to the scalar path and duplicates ``none`` under a new fingerprint.
    A structural method is its own spec, so its slug is never an axis value.
    """
    sn_pool = set(get_strategy("SkewNormal").axes["posthoc"])
    assert sn_pool <= POSTHOC_SLUGS
    for slug in ("Mixture", "ZINB", "NegBin", "DPO"):
        assert sn_pool - set(get_strategy(slug).axes["posthoc"]) == CDF_STAGE
    for spec in registered_strategies():
        assert not STRUCTURAL_STAGE & set(spec.axes.get("posthoc", ()))


def test_cell_knobs_the_corner_does_not_decide_stay_out_of_every_base_corner():
    """A cell carrying ``hpo_selection: calibrated`` must still resolve to a registered corner.

    ``pipeline.train_market`` projects the cell's runtime knobs onto ``(fixed_controls, axes)`` and
    hands the result to :func:`validate_strategy_selection`, so pinning a knob the corner does not
    decide turns every cell holding a different value into a hard training failure. Two knobs are
    decided elsewhere: ``cli._retry_calibrated_if_g4_only`` persists ``hpo_selection=calibrated``
    after a Gate-4-only miss (shipped cells carry it today), and the pipeline silently upgrades
    ``stabilization`` to L2 for centered SkewNormal and Mixture.
    """
    cell_knobs = {"hpo_selection": "calibrated", "stabilization": "L2"}
    cell = CellContext("WNBA", "AST", "SkewNormal", "continuous", frozenset(), _MATRIX_SHA)
    for slug in ("SkewNormal", "ZINB", "NegBin", "DPO"):
        spec = get_strategy(slug)
        runtime = {**strategy_controls(spec)[0], **cell_knobs}
        corner = {name: runtime[name] for name in (*spec.fixed_controls, *spec.axes)}
        assert not cell_knobs.keys() & corner.keys()
        validate_strategy_selection(cell, spec, corner, required_capabilities=SWEEP_CAPABILITIES)


def test_seed_corners_are_registered_corners_of_their_named_spec():
    """A seeded recipe is one with an independent full-HPO six-gate pass, kept so the board can
    start from it. Drift between the seed and its family's declared grid fails loud at import;
    this pins the same contract so an edit to either side is caught here too.
    """
    assert SEED_CORNERS
    for league, market, slug, controls in SEED_CORNERS:
        assert market in ALL_MARKETS[league]
        assert controls in strategy_controls(get_strategy(slug))


def test_required_matrix_columns_fail_closed_and_strategy_identity_is_spec_specific():
    receiving = get_strategy(RECEIVING)
    no_schema = CellContext("NFL", "receiving yards", "SkewNormal", "continuous")
    assert receiving not in strategies_for_cell(no_schema)
    with_schema = CellContext(
        "NFL",
        "receiving yards",
        "SkewNormal",
        "continuous",
        frozenset(role_spec_for("NFL", "receiving yards").all_columns),
        _MATRIX_SHA,
    )
    assert receiving in strategies_for_cell(with_schema)

    sn_controls = {
        "dist": "SkewNormal",
        "normalization": "ratio_meanyr",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending_loss_fn": "nll",
        "posthoc": "none",
    }
    mixture_controls = {
        "dist": "Mixture",
        "normalization": "ratio_meanyr",
        "dist_training_loss": "nll",
        "blending_loss_fn": "nll",
        "posthoc": "none",
    }
    sn = build_artifact_identity("SkewNormal", "WNBA", "AST", sn_controls, matrix_hash=_MATRIX_SHA)
    mixture = build_artifact_identity(
        "Mixture", "WNBA", "AST", mixture_controls, matrix_hash=_MATRIX_SHA
    )
    assert sn.strategy_slug != mixture.strategy_slug
    assert sn.signature != mixture.signature
    assert sn.structural_strategy == mixture.structural_strategy == "none"

    other_sn_controls = {**sn_controls, "blending_loss_fn": "crps"}
    other_sn = build_artifact_identity(
        "SkewNormal", "WNBA", "AST", other_sn_controls, matrix_hash=_MATRIX_SHA
    )
    assert other_sn.signature == sn.signature  # same implementation
    assert other_sn.controls_json != sn.controls_json
    assert other_sn.corner_fingerprint != sn.corner_fingerprint

    frame, model = _artifact(get_strategy("SkewNormal"), "WNBA", "AST", sn_controls)
    with pytest.raises(ValueError, match="mismatched strategy artifact"):
        validate_strategy_artifacts(
            get_strategy("SkewNormal"),
            other_sn_controls,
            frame,
            model,
            league="WNBA",
            market="AST",
            matrix_hash=_MATRIX_SHA,
        )


@pytest.mark.parametrize("distribution", ["Gamma", "ZAGamma"])
def test_legacy_gamma_compatibility_is_serve_only_and_excluded_from_research_board(distribution):
    spec = get_strategy(distribution)
    context = CellContext("NBA", "legacy", distribution, "continuous")

    assert spec.capabilities == frozenset({CAP_SERVE})
    assert spec in strategies_for_cell(context, required_capabilities=(CAP_SERVE,))
    assert spec not in strategies_for_cell(
        context,
        required_capabilities=SWEEP_CAPABILITIES,
    )

    identity = resolve_report_identity({"distribution": distribution}, "NBA", "legacy")
    assert identity.strategy_slug == distribution
    assert identity.structural_strategy == BASE_STRUCTURAL_STRATEGY
    assert identity.status == "active"
    assert identity.controls_json is None
    assert identity.corner_fingerprint is None
    assert identity.matrix_hash is None
    assert identity.split_fingerprint is None


def test_artifact_validation_binds_optional_split_and_cell_and_malformed_identity_fails():
    spec = get_strategy(RUSHING)
    controls = dict(spec.fixed_controls)
    frame, model = _artifact(spec, "NFL", "rushing yards", controls)
    assert (
        validate_strategy_artifacts(
            spec,
            controls,
            frame,
            model,
            league="NFL",
            market="rushing yards",
            matrix_hash=_MATRIX_SHA,
        ).strategy_slug
        == RUSHING
    )

    tampered_recipe = {**model, "posthoc": "roe_mean"}
    with pytest.raises(ValueError, match="model recipe mismatch: posthoc"):
        resolve_report_identity(tampered_recipe, "NFL", "rushing yards")
    with pytest.raises(ValueError, match="model recipe mismatch: posthoc"):
        validate_strategy_artifacts(
            spec,
            controls,
            frame,
            tampered_recipe,
            league="NFL",
            market="rushing yards",
            matrix_hash=_MATRIX_SHA,
        )

    stale = frame.copy()
    stale[SPLIT_FINGERPRINT_CSV_COLUMN] = "WRONG"
    with pytest.raises(ValueError, match="model/CSV strategy identity mismatch"):
        validate_strategy_artifacts(
            spec,
            controls,
            stale,
            model,
            league="NFL",
            market="rushing yards",
            matrix_hash=_MATRIX_SHA,
        )
    with pytest.raises(ValueError, match="malformed model_strategy"):
        resolve_report_identity({MODEL_STRATEGY_MODEL_KEY: None}, "NFL", "rushing yards")

    legacy_only = dict(model)
    legacy_only.pop(MODEL_STRATEGY_MODEL_KEY)
    assert resolve_report_identity(legacy_only, "NFL", "rushing yards").strategy_slug == RUSHING

    missing_split = {
        "slug": spec.slug,
        "schema_version": spec.artifact_schema_version,
        "status": "active",
    }
    with pytest.raises(ValueError, match="split fingerprint"):
        build_artifact_identity(
            spec.slug,
            "NFL",
            "rushing yards",
            controls,
            missing_split,
            matrix_hash=_MATRIX_SHA,
        )


def test_csv_identity_rejects_partial_adapter_and_nonintegral_contracts():
    assert validate_strategy_frame(pd.DataFrame({"P": [0.4]})) == (None, None)
    with pytest.raises(ValueError, match="partial generic"):
        validate_strategy_frame(pd.DataFrame({"StrategySignature": ["orphan"]}))
    with pytest.raises(ValueError, match="adapter strategy columns"):
        validate_strategy_frame(pd.DataFrame({"StructuralRoute": ["expert"]}))
    assert validate_strategy_frame(pd.DataFrame({"P_PrePool": [0.5]})) == (None, None)
    assert validate_strategy_frame(pd.DataFrame({"PITRecalKnots": ["[]"]})) == (None, None)

    spec = get_strategy("SkewNormal")
    controls = {
        "dist": "SkewNormal",
        "normalization": "ratio_meanyr",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending_loss_fn": "nll",
        "posthoc": "none",
    }
    frame, _ = _artifact(spec, "WNBA", "AST", controls)
    frame["StrategyImplementationVersion"] = 1.5
    with pytest.raises(ValueError, match="requires integral implementation_version"):
        validate_strategy_frame(frame, league="WNBA", market="AST")


def test_dist_axis_flag_and_class_maps():
    """The dist axis forwards --dist; the class maps route count/continuous cells to their families."""
    assert get_strategy("DPO").cli_flags["dist"] == "--dist"
    assert distribution_class("SkewNormal") == "continuous"
    assert distribution_class("DPO") == "count"
    # dist leads the swept-axis columns and rides in the board schema, and every family's
    # calibration-pool axis has a convenience column beside the authoritative controls_json.
    assert sweep._AXIS_COLUMNS[0] == "dist"
    assert set(sweep._AXIS_COLUMNS) >= {"dist", "posthoc"}
    assert set(sweep._AXIS_COLUMNS) <= set(sweep._BOARD_COLUMNS)
    # The frame a corner was scored on rides beside the split it was fingerprinted against.
    assert sweep._BOARD_COLUMNS[sweep._BOARD_COLUMNS.index("split_fingerprint") + 1] == "eval_split"
    assert sweep.EVAL_SPLIT_CROSSFIT == "crossfit_validation"


def test_sn_param_axis_swept_persisted_and_dump_inert():
    """SkewNormal sweeps sn_param (direct vs centered parametrization) and persists it per-cell —
    a ship-deciding knob without a stat_meta field would be pruned by the warm cron retrain. The
    axis forwards --sn-param, rides the board schema, and does NOT key the deterministic dump
    subdir (like the loss axes, corners sharing a subdir train+score sequentially).
    """
    sn = get_strategy("SkewNormal")
    assert sn.axes["sn_param"] == ("direct", "centered")
    assert sn.persist["sn_param"] == "sn_param"
    assert sn.cli_flags["sn_param"] == "--sn-param"
    assert "sn_param" in sweep._AXIS_COLUMNS and "sn_param" in sweep._BOARD_COLUMNS
    corner = {"normalization": "ratio_meanyr", "sn_param": "centered"}
    assert sweep._dump_subdir(corner, sn) == "ratio_meanyr"


def test_dump_subdir_matches_meditate_keying():
    """The dump subdir mirrors pipeline.py: SN uses the normalization; a ZINB corner uses the
    canonical ``none`` count namespace, with a ``_hurdle`` suffix in hurdle mode.
    """
    assert (
        sweep._dump_subdir(
            {"normalization": "centered_additive_mean10"}, get_strategy("SkewNormal")
        )
        == "centered_additive_mean10"
    )
    assert sweep._dump_subdir({"zinb_mode": "joint"}, get_strategy("ZINB")) == "none"
    assert sweep._dump_subdir({"zinb_mode": "hurdle"}, get_strategy("ZINB")) == "none_hurdle"
    # A NegBin/DPO corner carries no zinb_mode key → the non-hurdle fallback subdir, matching what
    # the pipeline.py --dist fix writes (keyed on the trained dist, not raw zinb_mode).
    assert (
        sweep._dump_subdir(
            {"dist": "NegBin", "count_dispersion_objective": "crps"},
            get_strategy("NegBin"),
        )
        == "none"
    )
    assert (
        sweep._dump_subdir(
            {"dist": "DPO", "count_dispersion_objective": "pit_ks"}, get_strategy("DPO")
        )
        == "none"
    )
    assert (
        sweep._dump_subdir({"normalization": "ratio_meanyr"}, get_strategy(RUSHING))
        == f"ratio_meanyr__{RUSHING}"
    )
    assert sweep._decode_strategy({"dist": "NegBin"}) == "none"
    assert sweep._decode_strategy({"normalization": "ratio_meanyr"}) == "ratio_meanyr"


def test_cell_families_admits_on_target_shape_not_the_stat_meta_class(monkeypatch):
    """Family admission reads the cell's own target, not the class its stat_meta ``dist`` names.

    A low-mean integer target offers all five base families whatever the configured dist; a
    non-integer target or a mean above the count ceiling leaves only the continuous pair, because
    the count branch's dispersion cap could then only fail Gate 4. An unregistered family is loud.
    """
    fake = {
        "MLB": {
            "pitcher strikeouts": {"dist": "ZINB"},
            "hits allowed": {"dist": "Gamma"},  # unswept family
        },
        "WNBA": {"AST": {"dist": "SkewNormal"}},
        "NFL": {
            "passing yards": {"dist": "SkewNormal"},
            "receiving yards": {"dist": "SkewNormal"},
        },
    }
    contracts = {
        ("MLB", "pitcher strikeouts"): (frozenset(), _MATRIX_SHA, True, 5.4),
        ("WNBA", "AST"): (frozenset(), _MATRIX_SHA, True, 3.9),
        ("NFL", "passing yards"): (frozenset(), _MATRIX_SHA, True, 230.0),
        ("NFL", "receiving yards"): (
            _STRUCTURAL_CELL_COLUMNS[("NFL", "receiving yards")],
            _MATRIX_SHA,
            False,
            41.2,
        ),
    }
    monkeypatch.setattr(sweep, "load_stat_meta", lambda path: fake)
    monkeypatch.setattr(sweep, "_training_matrix_contract", lambda lg, mkt: contracts[(lg, mkt)])

    every_base = ("SkewNormal", "ZINB", "NegBin", "DPO")
    assert sweep._cell_families("MLB", "pitcher strikeouts") == every_base
    # A SkewNormal-configured cell with an integer low-mean target still gets the count families.
    assert sweep._cell_families("WNBA", "AST") == every_base
    assert sweep._cell_families("NFL", "passing yards") == ("SkewNormal",)
    # A non-integer target keeps the count families out even at a mean well under the ceiling.
    assert sweep._cell_families("NFL", "receiving yards") == ("SkewNormal",)
    with pytest.raises(click.UsageError):
        sweep._cell_families("MLB", "hits allowed")
    # --dist-class filters families, not cells: every eligible cell keeps only that class's pool.
    assert sweep._cell_families("WNBA", "AST", "count") == ("ZINB", "NegBin", "DPO")
    assert sweep._cell_families("MLB", "pitcher strikeouts", "continuous") == ("SkewNormal",)
    assert sweep._cell_families("NFL", "passing yards", "count") == ()


def test_cell_families_excludes_structural_specs_because_holdout_blind_refuses_them(monkeypatch):
    """A structural method is enrolled for these cells yet never enters the sweep's family pool.

    ``train_market`` raises under ``--holdout-blind`` for a structural strategy: its role support is
    derived from validation row counts, so the folds would disagree about whether the method fell
    back and the scored corner would not be the corner that trained. Both NFL yards cells enroll
    both structural specs today, which makes them the cells where the exclusion has to bite.
    """
    fake = {
        "NFL": {
            "receiving yards": {"dist": "SkewNormal"},
            "rushing yards": {"dist": "SkewNormal"},
        }
    }
    monkeypatch.setattr(sweep, "load_stat_meta", lambda path: fake)
    for market in ("receiving yards", "rushing yards"):
        enrolled = {
            spec.slug
            for spec in strategies_for_cell(
                sweep._cell_context("NFL", market), required_capabilities=SWEEP_CAPABILITIES
            )
        }
        assert {RECEIVING, RUSHING} <= enrolled
        assert not {RECEIVING, RUSHING} & set(sweep._cell_families("NFL", market))


def test_structural_strategies_are_market_agnostic_sweep_candidates():
    """Neither structural strategy is cell-pinned; both are applicability-gated so the board can
    give them a good-faith search on any qualifying cell. The role×position two-part strategy needs
    a per-(league, market) role-registry entry plus its role columns; the affine strategy needs only
    a continuous cell whose matrix carries ``Player position``.
    """
    two_part = get_strategy(RECEIVING)
    affine = get_strategy(RUSHING)
    nba_pts_columns = frozenset(role_spec_for("NBA", "PTS").all_columns)
    registered = CellContext("NBA", "PTS", "SkewNormal", "continuous", nba_pts_columns, _MATRIX_SHA)
    missing_columns = CellContext(
        "NBA", "PTS", "SkewNormal", "continuous", frozenset({"unrelated"}), _MATRIX_SHA
    )
    position_only = CellContext(
        "NBA", "STL", "SkewNormal", "continuous", frozenset({"Player position"}), _MATRIX_SHA
    )

    assert two_part in strategies_for_cell(registered, required_capabilities=SWEEP_CAPABILITIES)
    assert two_part not in strategies_for_cell(
        missing_columns, required_capabilities=SWEEP_CAPABILITIES
    )
    # two-part needs the role columns of the cell's own role-registry entry, absent here
    assert two_part not in strategies_for_cell(
        position_only, required_capabilities=SWEEP_CAPABILITIES
    )
    # the affine strategy offers a good-faith corner on any continuous cell carrying Player position
    assert affine in strategies_for_cell(registered, required_capabilities=SWEEP_CAPABILITIES)
    assert affine in strategies_for_cell(position_only, required_capabilities=SWEEP_CAPABILITIES)
    assert affine not in strategies_for_cell(
        missing_columns, required_capabilities=SWEEP_CAPABILITIES
    )


def test_cached_matrix_contract_reads_schema_target_moments_and_resha_when_parquet_changes(
    monkeypatch, tmp_path
):
    """The frozen input yields the schema, the exact-bytes SHA, and the two target facts that gate
    count-family admission — a fractional Result is not an integer lattice; whole floats are.
    """
    monkeypatch.undo()  # exercise the real helper, not this module's fixed-contract fixture
    monkeypatch.setattr(sweep, "_TRAINING_DATA_ROOT", tmp_path)
    sweep._read_training_matrix_contract.cache_clear()
    path = sweep._training_matrix_path("NFL", "receiving yards")
    pd.DataFrame({"Player position": ["WR"], "Result": [10.5], "route_feature": [1.0]}).to_parquet(
        path
    )
    columns_before, sha_before, integer_before, mean_before = sweep._training_matrix_contract(
        "NFL", "receiving yards"
    )

    pd.DataFrame(
        {
            "Player position": ["WR", "RB"],
            "Result": [10.0, 12.0],
            "route_feature": [1.0, 2.0],
        }
    ).to_parquet(path)
    columns_after, sha_after, integer_after, mean_after = sweep._training_matrix_contract(
        "NFL", "receiving yards"
    )

    assert columns_before == columns_after == {"Player position", "Result", "route_feature"}
    assert len(sha_before) == len(sha_after) == 64
    assert sha_before != sha_after
    assert (integer_before, mean_before) == (False, 10.5)
    assert (integer_after, mean_after) == (True, 11.0)


def test_cell_trial_count_is_the_reachable_grid_capped_by_the_budget():
    """A cell contributes its reachable corners, or the per-cell budget when the grid is bigger."""
    assert (
        tpe_search.reachable_corners(sweep._cell_context("MLB", "pitcher strikeouts"), ("NegBin",))
        == 20
    )
    assert sweep._cell_trial_count("MLB", "pitcher strikeouts", ("NegBin",), 48) == 20
    # ZINB + NegBin + DPO = 80 reachable, so the 48-trial budget binds.
    families = ("ZINB", "NegBin", "DPO")
    assert sweep._cell_trial_count("MLB", "pitcher strikeouts", families, 48) == 48
    assert sweep._cell_trial_count("MLB", "pitcher strikeouts", families, 200) == 80


def test_ratio_projvol_is_pruned_per_cell_by_its_denominator_mapping(monkeypatch):
    """``ratio_projvol`` needs a projected-volume denominator, so it is a per-cell axis value.

    Pruning it in :func:`tpe_search.cell_axis_choices` — rather than through the spec's applicability —
    keeps SkewNormal searchable on a cell that simply has no volume feature to divide by. NHL saves
    has no mapping at all; NFL passing yards maps to ``attempts`` and keeps the value.
    """
    monkeypatch.undo()  # real matrices: the denominator resolves against the cached schema
    sn = get_strategy("SkewNormal")
    pruned = tpe_search.cell_axis_choices(sweep._cell_context("NHL", "saves"), sn)
    kept = tpe_search.cell_axis_choices(sweep._cell_context("NFL", "passing yards"), sn)
    assert "ratio_projvol" not in pruned["normalization"]
    assert "ratio_projvol" in kept["normalization"]


def test_reachable_corners_on_the_live_nfl_matrices(monkeypatch):
    """The budget is spent against the corners a cell can actually reach, not the declared grid."""
    monkeypatch.undo()  # real matrices: target lattice + mean decide count-family admission
    # 192 SkewNormal + 40 ZINB + 20 NegBin + 20 DPO — a low-mean integer target. Mixture
    # declares 40 more but has no serve path, so SWEEP_CAPABILITIES keeps it out of the pool.
    tds = sweep._cell_context("NFL", "passing tds")
    assert tpe_search.reachable_corners(tds, sweep._cell_families("NFL", "passing tds")) == 272
    # 192 + 40: the mean-226.7 target is over the count-admission ceiling, and the structural specs
    # this cell enrolls are excluded from the sweep.
    yards = sweep._cell_context("NFL", "passing yards")
    assert tpe_search.reachable_corners(yards, sweep._cell_families("NFL", "passing yards")) == 192


def test_decode_strategy_is_registered_norm_for_sn_and_none_for_count():
    assert sweep._decode_strategy({"normalization": "ratio_meanyr"}) == "ratio_meanyr"
    assert sweep._decode_strategy({"zinb_mode": "hurdle"}) == "none"


# --- honest per-corner scorer ----------------------------------------------------------------


def test_score_corner_runs_production_gate_for_sn(monkeypatch):
    captured = {}
    spec = get_strategy("SkewNormal")
    corner = {
        "dist": "SkewNormal",
        "normalization": "centered_additive_mean10",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending_loss_fn": "nll",
        "posthoc": "none",
    }
    frame, model = _artifact(spec, "WNBA", "AST", corner)

    def fake_gate_row(df, pred_col, **kwargs):
        captured.update(kwargs)
        return _canned_row(ship=True, g4_pit_ks=0.03)

    model.update(dispersion_cal=1.27, skew_cal=2.3)
    monkeypatch.setattr(sweep, "load_test_set", lambda path, col: frame)
    monkeypatch.setattr(sweep.pd, "read_pickle", lambda path: model)
    monkeypatch.setattr(sweep, "gate_row", fake_gate_row)
    monkeypatch.setattr(sweep, "apply_thresholds", lambda row: row)

    row = sweep._score_corner("WNBA", "AST", corner, spec)

    # The honest gate decodes under the trial's normalization — no test re-fit of calibration.
    assert captured["decode_strategy"] == "centered_additive_mean10"
    assert row["normalization"] == "centered_additive_mean10"
    assert row["ships"] is True
    assert row["slack"] == (0.05 - 0.03) / 0.05  # Gate 4 binds
    # All six gates surfaced (g6 included).
    assert row["g1_pass"] is True and row["g6_pass"] is True
    assert row["dispersion_cal"] == 1.27 and row["skew_cal"] == 2.3
    assert row["n"] == 1500


def test_score_corner_decodes_zinb_with_none_and_no_skew(monkeypatch):
    """A ZINB corner scores through the same gate with ``decode_strategy=none``; a count
    dump has no ``skew_cal`` so it surfaces as 0.0.
    """
    captured = {}
    spec = get_strategy("ZINB")
    corner = {
        "dist": "ZINB",
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "crps",
        "blending_loss_fn": "nll",
        "posthoc": "none",
    }
    frame, model = _artifact(spec, "MLB", "pitcher strikeouts", corner)

    def fake_gate_row(df, pred_col, **kwargs):
        captured.update(kwargs)
        return _canned_row(ship=True, g4_pit_ks=0.02)

    monkeypatch.setattr(sweep, "load_test_set", lambda path, col: frame)
    monkeypatch.setattr(sweep.pd, "read_pickle", lambda path: model)  # no skew_cal
    monkeypatch.setattr(sweep, "gate_row", fake_gate_row)
    monkeypatch.setattr(sweep, "apply_thresholds", lambda row: row)

    row = sweep._score_corner("MLB", "pitcher strikeouts", corner, spec)
    assert captured["decode_strategy"] == "none"
    assert row["zinb_mode"] == "hurdle"
    assert row["skew_cal"] == 0.0


def test_score_corner_requires_matching_active_structural_artifacts(monkeypatch):
    slug = RUSHING
    spec = get_strategy(slug)
    corner = dict(spec.fixed_controls)
    frame, model = _artifact(spec, "NFL", "rushing yards", corner)
    monkeypatch.setattr(sweep, "load_test_set", lambda path, col: frame)
    monkeypatch.setattr(sweep.pd, "read_pickle", lambda path: model)
    monkeypatch.setattr(
        sweep,
        "gate_row",
        lambda *args, **kwargs: _canned_row(ship=True, g4_pit_ks=0.02),
    )
    monkeypatch.setattr(sweep, "apply_thresholds", lambda row: row)
    assert sweep._score_corner("NFL", "rushing yards", corner, spec)["ships"] is True

    _, killed = _artifact(spec, "NFL", "rushing yards", corner, status="killed_fallback")
    monkeypatch.setattr(sweep.pd, "read_pickle", lambda path: killed)
    with pytest.raises(InactiveStrategyArtifactError, match="inactive strategy artifact"):
        sweep._score_corner("NFL", "rushing yards", corner, spec)


def test_verdict_and_failed_gates_include_g6():
    assert sweep._verdict({"ships": True}) == "SHIP"
    killed = {"ships": False, "g1_pass": True, "g4_pass": False, "g6_pass": False}
    assert sweep._failed_gates(killed) == ["g4", "g6"]
    assert sweep._verdict(killed) == "KILL: g4 g6"


def test_run_and_score_tags_family_and_uses_honest_gate(monkeypatch):
    monkeypatch.setattr(sweep, "_run_deterministic_meditate", lambda *a, **k: None)
    captured = {}

    def fake_score(league, market, corner, spec):
        captured["corner"] = corner
        return {**corner, "slack": 0.1, "ships": True}

    monkeypatch.setattr(sweep, "_score_corner", fake_score)
    corner = {
        "dist": "SkewNormal",
        "normalization": "ratio_meanyr",
        "dist_training_loss": "nll",
        "sn_param": "direct",
        "blending_loss_fn": "crps",
        "posthoc": "none",
    }
    row = sweep._run_and_score("WNBA", "AST", "SkewNormal", corner)
    assert captured["corner"] == corner
    assert row["family"] == "SkewNormal"
    assert row["slack"] == 0.1


def test_run_and_score_records_failed_corner_non_shipping(monkeypatch):
    """A corner whose meditate errors is caught and returned as a non-shipping row — never scored."""

    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, "meditate")

    monkeypatch.setattr(sweep, "_run_deterministic_meditate", boom)
    monkeypatch.setattr(
        sweep, "_score_corner", lambda *a, **k: pytest.fail("a failed corner must not be scored")
    )
    corner = {
        "dist": "SkewNormal",
        "normalization": "ratio_meanyr",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending_loss_fn": "nll",
        "posthoc": "none",
    }
    row = sweep._run_and_score("NHL", "blocked", "SkewNormal", corner)
    assert row["ships"] is False
    assert row["slack"] == sweep._FAILED_CORNER_SLACK
    assert row["family"] == "SkewNormal"
    assert row["normalization"] == "ratio_meanyr"  # corner axes preserved for the board
    # A failed corner is still a cross-fit verdict, so it caches and never retrains on resume.
    assert row["eval_split"] == sweep.EVAL_SPLIT_CROSSFIT


def test_run_and_score_records_inactive_explicit_strategy_but_not_other_identity_errors(
    monkeypatch,
):
    spec = get_strategy(RUSHING)
    monkeypatch.setattr(sweep, "_run_deterministic_meditate", lambda *args: None)
    monkeypatch.setattr(
        sweep,
        "_score_corner",
        lambda *args: (_ for _ in ()).throw(InactiveStrategyArtifactError("fallback")),
    )
    row = sweep._run_and_score("NFL", "rushing yards", spec.slug, dict(spec.fixed_controls))
    assert row["ships"] is False and row["slack"] == sweep._FAILED_CORNER_SLACK

    monkeypatch.setattr(
        sweep,
        "_score_corner",
        lambda *args: (_ for _ in ()).throw(ValueError("wrong matrix")),
    )
    with pytest.raises(ValueError, match="wrong matrix"):
        sweep._run_and_score("NFL", "rushing yards", spec.slug, dict(spec.fixed_controls))


# --- per-cell budgeted study ------------------------------------------------------------------


def test_search_cell_samples_the_sn_grid_under_budget_and_ranks(monkeypatch):
    """The study ranks a budgeted sample of the family's reachable corners, best slack first.

    ``_known_good_corners`` is stubbed empty so the suggestion sequence depends only on the seeded
    sampler and the fake objective, not on whatever recipe stat_meta carries for the cell today;
    the enqueue path has its own test. With 21 evaluations of a 192-corner grid the sampler is
    expected to find the dominant normalization (worth +0.20 against 0.00 for the runner-up).
    """
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_known_good_corners", lambda context: [])
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    board = sweep.search_cell("WNBA", "AST")

    assert 0 < len(board) <= tpe_search.MAX_TRIALS_PER_CELL
    assert board.iloc[0]["normalization"] == "centered_additive_mean10"
    assert board["slack"].is_monotonic_decreasing
    assert (board["family"] == "SkewNormal").all()
    assert (board["league"] == "WNBA").all() and (board["market"] == "AST").all()
    # The base family is part of the reproducible corner and persists explicitly.
    assert (board["dist"] == "SkewNormal").all()
    # controls_json is authoritative; the convenience columns echo it, never diverge from it.
    assert all(
        json.loads(row["controls_json"])["posthoc"] == row["posthoc"] for _, row in board.iterrows()
    )
    # Every row is a cross-fit verdict, and g6 is surfaced.
    assert (board["eval_split"] == sweep.EVAL_SPLIT_CROSSFIT).all()
    assert "g6_pass" in board.columns


def test_search_cell_evaluates_the_seed_corner_first(monkeypatch):
    """NFL passing tds carries a DPO seed corner, so the budget never risks missing it.

    A 48-trial sample of a 312-corner grid cannot be relied on to rediscover a recipe with an
    independent full-HPO pass, so the seed and the stat_meta incumbent are enqueued ahead of the
    sampler. Assert the seed's position and its uniqueness, not the whole enqueued list: the cell's
    incumbent is whatever it last shipped, so pinning the list would fail on every future ship.
    """
    seed = next(
        controls
        for league, market, _, controls in SEED_CORNERS
        if (league, market) == ("NFL", "passing tds")
    )
    context = sweep._cell_context("NFL", "passing tds")
    known_good = [(spec.slug, corner) for spec, corner in sweep._known_good_corners(context)]
    assert known_good.count(("DPO", seed)) == 1

    trained = []
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("ZINB", "NegBin", "DPO"))
    monkeypatch.setattr(sweep, "_run_and_score", _spy_run_and_score(trained))
    sweep.search_cell("NFL", "passing tds", max_trials=6)
    assert trained[0] == ("DPO", seed)


def test_search_cell_count_cell_sweeps_both_families_and_unions(monkeypatch):
    """A count cell studies ZINB *and* plain NegBin; the boards union, slack-ranked across families.

    One conditional study spans both families, so the closure must read the suggested spec per trial
    — scoring every trial as one family would collapse both onto a single dist. The per-family
    coverage floor (:data:`tpe_search.MIN_FAMILY_TRIALS`) is what keeps a budgeted study from ending
    before the second family is sampled at all.
    """
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("ZINB", "NegBin"))
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    board = sweep.search_cell("MLB", "pitcher strikeouts")

    assert 0 < len(board) <= tpe_search.MAX_TRIALS_PER_CELL
    assert set(board["family"]) == {"ZINB", "NegBin"}
    # Per-trial spec binding: each family's rows carry its own single-choice dist (no bleed).
    assert (board.loc[board["family"] == "ZINB", "dist"] == "ZINB").all()
    assert (board.loc[board["family"] == "NegBin", "dist"] == "NegBin").all()
    # Union is slack-sorted; ZINB's hurdle+pit_ks+crps (0.15+0.05+0.02) tops NegBin's best (0.05+0.02).
    assert board["slack"].is_monotonic_decreasing
    assert board.iloc[0]["family"] == "ZINB" and board.iloc[0]["zinb_mode"] == "hurdle"
    # NegBin corners never carry a zinb_mode key → its column stays blank for them (schema superset).
    assert board.loc[board["family"] == "NegBin", "zinb_mode"].isna().all()
    # SN-only axes are blank on a count board (the schema is a shared superset).
    assert board["normalization"].isna().all()
    assert board["sn_param"].isna().all()


def test_search_cell_survives_a_failing_corner(monkeypatch):
    """One corner erroring does not abort the study: the bad trial lands non-shipping and it goes on.

    Mirrors the NHL `blocked` case — the SkewNormal grid's `dist_training_loss=crps` corners crash
    when the cell trains as ZINB, but the `nll` corners score fine and the board still ranks them.
    """
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))

    def maybe_fail(league, market, corner, spec):
        if corner["dist_training_loss"] == "crps":
            raise subprocess.CalledProcessError(1, "meditate")

    monkeypatch.setattr(sweep, "_run_deterministic_meditate", maybe_fail)
    monkeypatch.setattr(
        sweep,
        "_score_corner",
        lambda lg, mkt, corner, spec: {**corner, "slack": 0.1, "ships": True},
    )
    board = sweep.search_cell("NHL", "blocked")

    failed = board[board["slack"] == sweep._FAILED_CORNER_SLACK]
    scored = board[board["dist_training_loss"] == "nll"]
    assert not failed.empty and not scored.empty
    assert (failed["dist_training_loss"] == "crps").all()
    assert not failed["ships"].astype(bool).any()
    assert (scored["slack"] == 0.1).all()
    # A failed corner is tagged with the search frame like any other, so it caches under its
    # fingerprint and a resume never retrains a reliable crash.
    assert (failed["eval_split"] == sweep.EVAL_SPLIT_CROSSFIT).all()


def test_run_deterministic_meditate_builds_corner_flags(monkeypatch, tmp_path):
    """Each corner axis is forwarded as its flag; a ZINB corner omits --target-normalization and the
    output is captured to a per-corner log under the research log root. Every trial runs
    ``--holdout-blind``, so the search is scored on the cross-fit frame and never adapts against the
    rows the ship gate uses.
    """
    monkeypatch.setattr(sweep, "_DETERMINISTIC_LOG_ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    sn = {
        "normalization": "ratio_meanyr",
        "dist_training_loss": "nll",
        "sn_param": "centered",
        "blending_loss_fn": "crps",
        "posthoc": "isotonic_mean",
    }
    sn_spec = get_strategy("SkewNormal")
    sweep._run_deterministic_meditate("WNBA", "AST", sn, sn_spec)
    last = calls[-1]
    assert {"--deterministic", "--bypass-withholding", "--holdout-blind"} <= set(last)
    assert last[last.index("--target-normalization") + 1] == "ratio_meanyr"
    assert last[last.index("--dist-training-loss") + 1] == "nll"
    assert last[last.index("--sn-param") + 1] == "centered"
    assert last[last.index("--blending-loss-fn") + 1] == "crps"
    # Base families now sweep the calibration pool, so the corner's posthoc overrides the cell's
    # stat_meta value rather than deferring to it.
    assert last[last.index("--posthoc") + 1] == "isotonic_mean"
    assert "--structural-strategy" not in last
    assert sweep._log_path("WNBA", "AST", sn, sn_spec).exists()

    sweep._run_deterministic_meditate("NFL", "receiving yards", sn, sn_spec)
    # The retired --structural-strategy axis emits no selector flag for a base corner.
    assert "--structural-strategy" not in calls[-1]

    zinb = {
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending_loss_fn": "nll",
        "posthoc": "none",
    }
    sweep._run_deterministic_meditate("MLB", "pitcher strikeouts", zinb, get_strategy("ZINB"))
    z = calls[-1]
    assert "--target-normalization" not in z
    assert "--sn-param" not in z
    assert z[z.index("--zinb-mode") + 1] == "hurdle"
    assert z[z.index("--count-dispersion-objective") + 1] == "pit_ks"
    assert "--structural-strategy" not in z
    assert (
        sweep._log_path("MLB", "pitcher strikeouts", zinb, get_strategy("ZINB")).parent.name
        == "none_hurdle"
    )

    for family in ("NegBin", "DPO"):
        count = {
            "dist": family,
            "count_dispersion_objective": "crps",
            "blending_loss_fn": "nll",
        }
        spec = get_strategy(family)
        sweep._run_deterministic_meditate("MLB", "pitcher strikeouts", count, spec)
        command = calls[-1]
        assert command[command.index("--dist") + 1] == family
        assert "--target-normalization" not in command
        assert sweep._log_path("MLB", "pitcher strikeouts", count, spec).parent.name == "none"

    method_spec = get_strategy(RUSHING)
    method = dict(method_spec.fixed_controls)
    sweep._run_deterministic_meditate("NFL", "rushing yards", method, method_spec)
    structural = calls[-1]
    assert structural[structural.index("--dist") + 1] == "SkewNormal"
    assert structural[structural.index("--target-normalization") + 1] == "ratio_meanyr"
    assert structural[structural.index("--dist-training-loss") + 1] == "crps"
    assert structural[structural.index("--blending-loss-fn") + 1] == "nll"
    assert structural[structural.index("--hpo-selection") + 1] == "loss"
    assert structural[structural.index("--stabilization") + 1] == "None"
    # The method rides the ``posthoc`` calibration pool: its slug is the --posthoc value,
    # and the retired --structural-strategy axis emits nothing.
    assert structural[structural.index("--posthoc") + 1] == RUSHING
    assert "--structural-strategy" not in structural


# --- archive-lock retry ----------------------------------------------------------------------


def _fake_meditate_run(*, lock_error, succeed_on_call=None):
    """A ``subprocess.run`` stand-in that writes to the captured log, then succeeds or raises.

    ``lock_error`` writes the DuckDB lock signature so the retry fires; otherwise a generic traceback
    so the run re-raises at once. ``succeed_on_call`` (1-indexed) is the first call that exits clean —
    ``None`` never succeeds. Returns ``(run, calls)`` where ``calls["n"]`` counts invocations.
    """
    calls = {"n": 0}

    def run(cmd, **kw):
        calls["n"] += 1
        log = kw["stdout"]
        if succeed_on_call is not None and calls["n"] >= succeed_on_call:
            log.write("trained ok\n")
            log.flush()
            return subprocess.CompletedProcess(cmd, 0)
        log.write("Could not set lock on file\n" if lock_error else "ValueError: boom\n")
        log.flush()
        raise subprocess.CalledProcessError(1, cmd)

    return run, calls


def test_lock_retry_retries_on_lock_then_succeeds(monkeypatch, tmp_path):
    """A lock-error trial waits per the back-off schedule and re-runs until one attempt exits clean."""
    monkeypatch.setattr(sweep, "_LOCK_RETRY_WAITS_S", (1, 2, 3))
    run, calls = _fake_meditate_run(lock_error=True, succeed_on_call=3)
    monkeypatch.setattr(sweep.subprocess, "run", run)
    slept = []
    monkeypatch.setattr(sweep.time, "sleep", slept.append)

    sweep._run_meditate_with_lock_retry(["meditate"], tmp_path / "x.log", timeout=10)

    assert calls["n"] == 3  # two lock failures, third attempt succeeds
    assert slept == [1, 2]  # waited before each retry


def test_lock_retry_reraises_nonlock_without_retrying(monkeypatch, tmp_path):
    """A non-lock failure is a real error: re-raise on the first try, no wait."""
    monkeypatch.setattr(sweep, "_LOCK_RETRY_WAITS_S", (1, 2, 3))
    run, calls = _fake_meditate_run(lock_error=False)
    monkeypatch.setattr(sweep.subprocess, "run", run)
    slept = []
    monkeypatch.setattr(sweep.time, "sleep", slept.append)

    with pytest.raises(subprocess.CalledProcessError):
        sweep._run_meditate_with_lock_retry(["meditate"], tmp_path / "x.log", timeout=10)

    assert calls["n"] == 1
    assert slept == []


def test_lock_retry_reraises_after_exhausting_waits(monkeypatch, tmp_path):
    """A lock that never clears exhausts the schedule (one final no-wait attempt) then re-raises."""
    monkeypatch.setattr(sweep, "_LOCK_RETRY_WAITS_S", (1, 2, 3))
    run, calls = _fake_meditate_run(lock_error=True)
    monkeypatch.setattr(sweep.subprocess, "run", run)
    slept = []
    monkeypatch.setattr(sweep.time, "sleep", slept.append)

    with pytest.raises(subprocess.CalledProcessError):
        sweep._run_meditate_with_lock_retry(["meditate"], tmp_path / "x.log", timeout=10)

    assert calls["n"] == 4  # three retries + a final attempt
    assert slept == [1, 2, 3]


# --- board derivation from stat_meta ---------------------------------------------------------


def test_select_board_cells_withheld_default_shipped_flag_and_data_filter(monkeypatch):
    """Default board = withheld × registered-family × trainable (`ALL_MARKETS`) × has-cached-data.
    `--include-shipped` adds already-shipped cells; a cell without a cached matrix is set aside as
    missing-data (warned, not swept); unsupported registered-market families fail loud, while
    non-market config stems are ignored.
    """
    fake = {
        "WNBA": {
            "AST": {"dist": "SkewNormal", "shipped": "withheld"},  # default board
            "PTS": {"dist": "SkewNormal", "shipped": "devel"},  # only with --include-shipped
            "STL": {
                "dist": "ZINB",
                "shipped": "withheld",
            },  # withheld but no cached matrix → missing
        },
        "MLB": {
            "pitcher strikeouts": {"dist": "ZINB", "shipped": "withheld"},  # default board
            "hits allowed": {"dist": "Gamma", "shipped": "withheld"},  # excluded: unswept family
            "1st inning hits allowed": {
                "dist": "ZINB",
                "shipped": "withheld",
            },  # excluded: not a market
        },
    }
    fake_markets = {"WNBA": ["AST", "PTS", "STL"], "MLB": ["pitcher strikeouts", "hits allowed"]}
    has_data = {("WNBA", "AST"), ("WNBA", "PTS"), ("MLB", "pitcher strikeouts")}
    monkeypatch.setattr(sweep, "load_stat_meta", lambda path: fake)
    monkeypatch.setattr(sweep, "ALL_MARKETS", fake_markets)
    monkeypatch.setattr(sweep, "_has_training_data", lambda lg, mkt: (lg, mkt) in has_data)

    with pytest.raises(click.UsageError, match="MLB hits allowed"):
        sweep._select_board_cells()
    del fake["MLB"]["hits allowed"]

    sweepable, missing, no_families = sweep._select_board_cells()
    assert set(sweepable) == {("WNBA", "AST"), ("MLB", "pitcher strikeouts")}
    assert missing == [("WNBA", "STL")]  # withheld + registry, but no cached matrix
    assert no_families == []
    # --include-shipped pulls in the devel cell (it has data); STL still missing.
    incl, _, _ = sweep._select_board_cells(include_shipped=True)
    assert ("WNBA", "PTS") in incl
    # League filter narrows to that league's sweepable cells.
    sweepable_wnba, _, _ = sweep._select_board_cells("WNBA")
    assert list(sweepable_wnba) == [("WNBA", "AST")]


def test_select_board_cells_dist_class_filters_families_and_sets_aside_empty_cells(monkeypatch):
    """``--dist-class`` narrows each cell's family pool, not the cohort of cells.

    A count corner can win on a cell whose stat_meta names a continuous family, so eligibility can't
    follow the configured dist. Every cell keeps the classes' families it can support; a cell the
    filter (or the count-admission gates) leaves with nothing is set aside in ``no_families`` and
    warned like a missing matrix rather than silently dropped.
    """
    fake = {
        "WNBA": {"AST": {"dist": "SkewNormal", "shipped": "withheld"}},
        "MLB": {"pitcher strikeouts": {"dist": "ZINB", "shipped": "withheld"}},
        # Mean over the count-admission ceiling → no count family survives --dist-class count.
        "NFL": {"passing yards": {"dist": "SkewNormal", "shipped": "withheld"}},
    }
    contracts = {
        ("WNBA", "AST"): (frozenset(), _MATRIX_SHA, True, 3.9),
        ("MLB", "pitcher strikeouts"): (frozenset(), _MATRIX_SHA, True, 5.4),
        ("NFL", "passing yards"): (frozenset(), _MATRIX_SHA, True, 230.0),
    }
    fake_markets = {"WNBA": ["AST"], "MLB": ["pitcher strikeouts"], "NFL": ["passing yards"]}
    monkeypatch.setattr(sweep, "load_stat_meta", lambda path: fake)
    monkeypatch.setattr(sweep, "_training_matrix_contract", lambda lg, mkt: contracts[(lg, mkt)])
    monkeypatch.setattr(sweep, "ALL_MARKETS", fake_markets)
    monkeypatch.setattr(sweep, "_has_training_data", lambda lg, mkt: True)

    count, _, no_families = sweep._select_board_cells(dist_class="count")
    assert count == {
        ("WNBA", "AST"): ("ZINB", "NegBin", "DPO"),
        ("MLB", "pitcher strikeouts"): ("ZINB", "NegBin", "DPO"),
    }
    assert no_families == [("NFL", "passing yards")]

    continuous, _, none_left = sweep._select_board_cells(dist_class="continuous")
    assert set(continuous) == set(contracts)
    assert all(families == ("SkewNormal",) for families in continuous.values())
    assert none_left == []

    every, _, _ = sweep._select_board_cells(dist_class="all")
    assert every[("MLB", "pitcher strikeouts")] == ("SkewNormal", "ZINB", "NegBin", "DPO")


def test_run_board_mode_warns_skipped_cells_and_passes_filters(monkeypatch, capsys):
    """The board runner warns per skipped cell — missing matrix or empty family pool — and forwards
    its scope filters and per-cell budget.
    """
    captured = {}

    def fake_select(league, include_shipped, dist_class):
        captured["incl"] = include_shipped
        captured["dist_class"] = dist_class
        return {("WNBA", "AST"): ("ZINB",)}, [("WNBA", "STL")], [("NFL", "passing yards")]

    monkeypatch.setattr(sweep, "_select_board_cells", fake_select)
    monkeypatch.setattr(sweep, "_cell_trial_count", lambda lg, mkt, families, max_trials: 12)

    def fake_run_board(cells, **kwargs):
        captured.update(cells=cells, **kwargs)
        return pd.DataFrame({"league": ["WNBA"], "market": ["AST"], "ships": [False]})

    monkeypatch.setattr(sweep, "run_board", fake_run_board)
    monkeypatch.setattr(sweep, "_print_board_rollup", lambda b: None)

    sweep._run_board_mode("WNBA", True, "count", "/tmp/board.csv", False, False, 32)
    assert captured["incl"] is True and captured["dist_class"] == "count"
    assert captured["max_trials"] == 32
    assert captured["families_by_cell"] == {("WNBA", "AST"): ("ZINB",)}
    out = capsys.readouterr().out
    assert "skip WNBA STL: no cached training matrix" in out
    assert "skip NFL passing yards: no applicable family under --dist-class count" in out
    assert "1 cells to sweep · 2 skipped · <=12 deterministic trainings" in out


def _cell_frame(league, market, **kwargs):
    """A one-row board frame for a cell — the shape ``search_cell`` returns for the run_board tests."""
    del kwargs  # budget/families/cache are asserted by the tests that care
    return pd.DataFrame({"league": [league], "market": [market], "slack": [0.1], "ships": [True]})


def _board_row(league, market, slug, controls):
    """One board row for a scored corner, as ``search_cell`` would persist it."""
    return {
        "league": league,
        "market": market,
        **_fake_run_and_score(league, market, slug, controls),
    }


def test_run_board_upserts_per_cell_preserving_other_leagues(monkeypatch, tmp_path):
    """A ``--league``-scoped board run upserts each cell and leaves foreign-league rows intact
    (the pre-fix concat-overwrite wiped them)."""
    out = str(tmp_path / "board.csv")
    pd.DataFrame({"league": ["NFL"], "market": ["sacks"], "slack": [0.3], "ships": [True]}).to_csv(
        out, index=False
    )
    monkeypatch.setattr(sweep, "search_cell", _cell_frame)
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda b: None)

    sweep.run_board([("WNBA", "AST")], out=out)
    board = pd.read_csv(out)
    assert set(zip(board["league"], board["market"], strict=True)) == {
        ("NFL", "sacks"),
        ("WNBA", "AST"),
    }


def test_run_board_resume_reuses_prior_rows_and_still_returns_the_cell(monkeypatch, tmp_path):
    """``--resume`` no longer skips whole cells — a budgeted search never completes one — so a
    resumed cell is swept again with its admissible prior rows handed in as the corner cache.
    """
    out = str(tmp_path / "board.csv")
    cell = ("MLB", "pitcher strikeouts")
    prior = _board_row(*cell, "NegBin", strategy_controls(get_strategy("NegBin"))[0])
    pd.DataFrame([prior]).to_csv(out, index=False)
    handed = {}

    def spy(league, market, **kwargs):
        handed[(league, market)] = kwargs["cached"]
        return _cell_frame(league, market)

    monkeypatch.setattr(sweep, "search_cell", spy)
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda b: None)

    board = sweep.run_board([cell, ("NBA", "FGA")], out=out, resume=True)

    assert list(handed[cell]) == [prior["corner_fingerprint"]]
    assert handed[("NBA", "FGA")] == {}  # no prior rows for that cell
    assert set(zip(board["league"], board["market"], strict=True)) == {cell, ("NBA", "FGA")}


def test_run_board_returns_only_requested_scope_but_preserves_unrelated_csv(monkeypatch, tmp_path):
    """A scoped run cannot leak an unrelated prior winner into ``--confirm`` input."""
    out = str(tmp_path / "board.csv")
    pd.DataFrame(
        {
            "league": ["WNBA", "NFL"],
            "market": ["AST", "sacks"],
            "slack": [0.9, 0.8],
            "ships": [True, True],
        }
    ).to_csv(out, index=False)
    monkeypatch.setattr(sweep, "search_cell", _cell_frame)
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda board: None)

    result = sweep.run_board([("WNBA", "AST"), ("NBA", "FGA")], out=out, resume=True)

    assert set(zip(result["league"], result["market"], strict=True)) == {
        ("WNBA", "AST"),
        ("NBA", "FGA"),
    }
    persisted = pd.read_csv(out)
    assert set(zip(persisted["league"], persisted["market"], strict=True)) == {
        ("WNBA", "AST"),
        ("NBA", "FGA"),
        ("NFL", "sacks"),
    }


def test_row_matches_contract_admits_a_current_corner_and_rejects_drift():
    """Resume admission is per row, not per cell: the row must name a registered corner of a
    registered spec, carry that spec's current identity, and be fingerprinted against the matrix in
    hand. A budgeted search never enumerates a cell, so 'has every corner' is not a usable test.
    """
    cell = ("MLB", "pitcher strikeouts")
    spec = get_strategy("NegBin")
    controls = strategy_controls(spec)[0]
    context = sweep._cell_context(*cell)
    row = _board_row(*cell, spec.slug, controls)
    assert sweep._row_matches_contract(pd.Series(row), context)

    stale_matrix = pd.Series({**row, "matrix_hash": "old-matrix"})
    assert not sweep._row_matches_contract(stale_matrix, context)

    off_grid = pd.Series({**row, "controls_json": controls_json({**controls, "posthoc": "bogus"})})
    assert not sweep._row_matches_contract(off_grid, context)

    stale_signature = pd.Series({**row, "strategy_signature": "old-signature"})
    assert not sweep._row_matches_contract(stale_signature, context)

    unregistered = pd.Series({**row, "strategy_slug": "NoSuchFamily"})
    assert not sweep._row_matches_contract(unregistered, context)

    incomplete = pd.Series({**row, "corner_fingerprint": pd.NA})
    assert not sweep._row_matches_contract(incomplete, context)


def test_legacy_holdout_scored_rows_never_enter_the_corner_cache(tmp_path):
    """``eval_split`` is the marker separating the cross-fit board from the rows scored on the ship
    holdout. A legacy row carries no value for it — whether the column is blank or absent from an
    older CSV entirely — and reusing one would mix two different frames on one board.
    """
    cell = ("MLB", "pitcher strikeouts")
    context = sweep._cell_context(*cell)
    row = _board_row(*cell, "NegBin", strategy_controls(get_strategy("NegBin"))[0])
    out = tmp_path / "board.csv"

    pd.DataFrame([row]).to_csv(out, index=False)
    assert list(sweep._cached_corners(str(out), *cell, context)) == [row["corner_fingerprint"]]
    # _cached_corners is cell-scoped: another cell's rows are not reusable here.
    assert sweep._cached_corners(str(out), "NBA", "FGA", context) == {}

    pd.DataFrame([{**row, "eval_split": ""}]).to_csv(out, index=False)
    assert sweep._cached_corners(str(out), *cell, context) == {}

    legacy = pd.DataFrame([row]).drop(columns=["eval_split"])
    legacy.to_csv(out, index=False)
    assert "eval_split" not in legacy.columns
    assert list(sweep._read_board(out).columns) == sweep._BOARD_COLUMNS  # back-filled, still <NA>
    assert sweep._cached_corners(str(out), *cell, context) == {}


def test_search_cell_reuses_cached_corners_instead_of_retraining(monkeypatch, tmp_path):
    """A corner this matrix already has a verdict for is never handed back to ``_run_and_score``.

    The sampler re-suggests corners freely — that is what makes a resumed multi-hour run cheap —
    so the cache is checked per trial, not just at start-up.
    """
    cell = ("MLB", "pitcher strikeouts")
    out = str(tmp_path / "board.csv")
    trained = []
    monkeypatch.setattr(sweep, "_run_and_score", _spy_run_and_score(trained))

    first = sweep.search_cell(*cell, families=("NegBin",), max_trials=8)
    sweep._upsert_cell(first, out)
    cached = sweep._cached_corners(out, *cell, sweep._cell_context(*cell))
    assert set(cached) == set(first["corner_fingerprint"])

    trained.clear()
    sweep.search_cell(*cell, families=("NegBin",), max_trials=8, cached=cached)
    retrained = {
        corner_fingerprint(get_strategy(family), corner, _MATRIX_SHA) for family, corner in trained
    }
    assert not retrained & set(cached)


def test_a_changed_family_pool_gets_its_own_journal_instead_of_crashing(monkeypatch, tmp_path):
    """Optuna fixes a categorical's choices at first suggestion, so the pool must key the study.

    Resuming a journal written under a different ``family`` arm set raises "CategoricalDistribution
    does not support dynamic value space" and kills the whole run — which is exactly what dropping
    Mixture did to every journal already on disk, and what a --dist-class run would do to a
    full-pool journal.
    """
    monkeypatch.setattr(tpe_search, "_STUDY_ROOT", tmp_path)
    full = tpe_search.cell_study("NBA", "DREB", ("SkewNormal", "ZINB", "NegBin", "DPO"))
    narrowed = tpe_search.cell_study("NBA", "DREB", ("ZINB", "NegBin", "DPO"))

    assert full.study_name != narrowed.study_name
    assert len(list(tmp_path.glob("NBA_DREB.*.log"))) == 2
    # The same pool reopens the same study, so a resume still resumes.
    again = tpe_search.cell_study("NBA", "DREB", ("SkewNormal", "ZINB", "NegBin", "DPO"))
    assert again.study_name == full.study_name


def test_a_cell_killed_mid_search_keeps_the_corners_it_already_trained(monkeypatch, tmp_path):
    """Every scored corner reaches the board CSV immediately, not once the cell finishes.

    The Optuna journal replays which corners were proposed but not what they scored, so a cell that
    dies on its 30th corner would otherwise resume with nothing and retrain all 29.
    """
    cell = ("MLB", "pitcher strikeouts")
    out = tmp_path / "board.csv"
    trained = []
    spy = _spy_run_and_score(trained)

    def die_on_the_third(league, market, family, corner):
        if len(trained) == 2:
            raise KeyboardInterrupt
        return spy(league, market, family, corner)

    monkeypatch.setattr(sweep, "_run_and_score", die_on_the_third)
    with pytest.raises(KeyboardInterrupt):
        sweep.search_cell(*cell, families=("NegBin",), max_trials=8, out=str(out))

    banked = sweep._read_board(out)
    assert len(banked) == 2
    assert set(banked["market"]) == {cell[1]}
    # ... and a resume treats them as already-scored rather than retraining them.
    assert set(sweep._cached_corners(str(out), *cell, sweep._cell_context(*cell))) == set(
        banked["corner_fingerprint"]
    )


def test_budget_counts_retrains_not_trials(monkeypatch, tmp_path):
    """A re-proposed corner costs no training, so it must not spend budget either.

    TPE re-proposes a seen corner often enough to matter — on an observed NBA DREB run roughly a
    quarter of trials came back cached — and charging those to ``--max-trials`` silently shrinks the
    search well below the budget the CLI advertises.
    """
    cell = ("MLB", "pitcher strikeouts")
    out = str(tmp_path / "board.csv")
    trained = []
    monkeypatch.setattr(sweep, "_run_and_score", _spy_run_and_score(trained))

    seeded = sweep.search_cell(*cell, families=("NegBin",), max_trials=4)
    sweep._upsert_cell(seeded, out)
    cached = sweep._cached_corners(out, *cell, sweep._cell_context(*cell))
    assert cached

    trained.clear()
    board = sweep.search_cell(*cell, families=("NegBin",), max_trials=4, cached=cached)
    retrained = {
        corner_fingerprint(get_strategy(family), corner, _MATRIX_SHA) for family, corner in trained
    }

    # NegBin's grid is far larger than the budget, so a run that charged cache hits to --max-trials
    # would stop having trained fewer than 4 corners. It must spend the budget on new ones.
    assert len(retrained) == 4
    assert not retrained & set(cached)
    # The board reports every corner the run touched, cached and fresh alike.
    assert set(board["corner_fingerprint"]) >= retrained


def test_cli_board_dry_run_trains_nothing(monkeypatch, tmp_path):
    """Board-mode ``--dry-run`` (no --market) prints the scope and exits without sweeping a single
    cell; --dist-class threads through to the cell selection.
    """
    from click.testing import CliRunner

    seen = {}

    def fake_select(lg, incl, dist_class):
        seen["dist_class"] = dist_class
        return {("WNBA", "AST"): ("ZINB",)}, [], []

    monkeypatch.setattr(sweep, "_select_board_cells", fake_select)
    monkeypatch.setattr(sweep, "_cell_trial_count", lambda lg, mkt, families, max_trials: 12)

    def boom(*a, **k):
        raise AssertionError("run_board must not be called on a dry run")

    monkeypatch.setattr(sweep, "run_board", boom)
    out = str(tmp_path / "board.csv")
    result = CliRunner().invoke(sweep.main, ["--dry-run", "--dist-class", "count", "--out", out])
    assert result.exit_code == 0, result.output
    assert "[dry-run]" in result.output
    assert seen["dist_class"] == "count"


# --- family-aware actionable summary ---------------------------------------------------------


def test_stat_meta_edit_is_family_aware():
    sn_controls = {
        "dist": "SkewNormal",
        "normalization": "centered_additive_mean10",
        "dist_training_loss": "nll",
        "sn_param": "centered",
        "blending_loss_fn": "crps",
        "posthoc": "prob_recal_platt",
    }
    sn = {
        "league": "WNBA",
        "market": "AST",
        "family": "SkewNormal",
        "strategy_slug": "SkewNormal",
        "structural_strategy": BASE_STRUCTURAL_STRATEGY,
        "controls_json": controls_json(sn_controls),
    }
    assert sweep._stat_meta_edit(sn) == (
        "dist=SkewNormal, target_normalization=centered_additive_mean10, dist_training_loss=nll, "
        "sn_param=centered, blending=crps, posthoc=prob_recal_platt"
    )
    receiving_base = {**sn, "league": "NFL", "market": "receiving yards"}
    assert sweep._stat_meta_edit(receiving_base).endswith("posthoc=prob_recal_platt")
    # Count families persist dist first (pins the winning family, e.g. a ZINB→NegBin flip) and
    # pin target_normalization=none, which a win on a continuous-configured cell would otherwise
    # leave carrying a slug ship_config._validate_cell rejects.
    zinb_controls = {
        "dist": "ZINB",
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending_loss_fn": "nll",
        "posthoc": "none",
    }
    zinb = {
        "league": "MLB",
        "market": "pitcher strikeouts",
        "family": "ZINB",
        "strategy_slug": "ZINB",
        "structural_strategy": BASE_STRUCTURAL_STRATEGY,
        "controls_json": controls_json(zinb_controls),
    }
    assert sweep._stat_meta_edit(zinb) == (
        "dist=ZINB, zinb_mode=hurdle, count_dispersion_objective=pit_ks, blending=nll, "
        "posthoc=none, target_normalization=none"
    )
    negbin_controls = {
        "dist": "NegBin",
        "count_dispersion_objective": "crps",
        "blending_loss_fn": "crps",
        "posthoc": "roe_mean",
    }
    negbin = {
        "league": "MLB",
        "market": "pitcher strikeouts",
        "family": "NegBin",
        "strategy_slug": "NegBin",
        "structural_strategy": BASE_STRUCTURAL_STRATEGY,
        "controls_json": controls_json(negbin_controls),
    }
    assert sweep._stat_meta_edit(negbin) == (
        "dist=NegBin, count_dispersion_objective=crps, blending=crps, posthoc=roe_mean, "
        "target_normalization=none"
    )
    spec = get_strategy(RUSHING)
    structural = {
        "league": "NFL",
        "market": "rushing yards",
        "family": spec.family,
        "strategy_slug": spec.slug,
        "structural_strategy": spec.slug,
        "controls_json": controls_json(spec.fixed_controls),
    }
    # The method rides the ``posthoc`` calibration pool: its slug is the persisted posthoc
    # value, and no separate structural_strategy field is written. Unlike a base family it also
    # persists hpo_selection, because a structural spec pins the whole recipe.
    assert sweep._stat_meta_edit(structural) == (
        "dist=SkewNormal, target_normalization=ratio_meanyr, dist_training_loss=crps, "
        f"sn_param=direct, blending=nll, posthoc={RUSHING}, hpo_selection=loss"
    )


# --- CLI -------------------------------------------------------------------------------------


def test_cli_runs_a_single_cell_under_the_max_trials_budget(monkeypatch, tmp_path):
    """``--league``/``--market`` sweeps one cell; ``--max-trials`` overrides the per-cell budget."""
    from click.testing import CliRunner

    monkeypatch.setattr(
        sweep, "_cell_families", lambda lg, mkt, dist_class=sweep._DIST_CLASS_ALL: ("SkewNormal",)
    )
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    out = str(tmp_path / "board.csv")
    result = CliRunner().invoke(
        sweep.main,
        ["--league", "WNBA", "--market", "AST", "--out", out, "--max-trials", "6"],
    )
    assert result.exit_code == 0, result.output
    assert "WNBA AST" in result.output
    board = pd.read_csv(out)
    assert 0 < len(board) <= 6
    assert (board["league"] == "WNBA").all()


def test_cli_confirm_invokes_run_confirm(monkeypatch, tmp_path):
    """``--confirm`` hands the ranked board to the confirm loop (which is mocked here)."""
    from click.testing import CliRunner

    from sportstradamus.training.model_strategy import confirm as model_strategy_confirm

    monkeypatch.setattr(
        sweep, "_cell_families", lambda lg, mkt, dist_class=sweep._DIST_CLASS_ALL: ("SkewNormal",)
    )
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    seen = {}
    monkeypatch.setattr(
        model_strategy_confirm,
        "run_confirm",
        lambda board, *, yes, max_nominees: seen.update(
            n=len(board), yes=yes, max_nominees=max_nominees
        ),
    )
    out = str(tmp_path / "board.csv")
    result = CliRunner().invoke(
        sweep.main,
        [
            "--league",
            "WNBA",
            "--market",
            "AST",
            "--out",
            out,
            "--max-trials",
            "5",
            "--confirm",
            "--yes",
            "--confirm-nominees",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["yes"] is True and 0 < seen["n"] <= 5
    assert seen["max_nominees"] == 2


def test_cli_scoped_board_confirms_only_the_requested_cells(monkeypatch, tmp_path):
    """A ``--league``-scoped board hands ``--confirm`` its own cells only, while the CSV keeps the
    rows of every cell outside the scope.
    """
    from click.testing import CliRunner

    from sportstradamus.training.model_strategy import confirm as model_strategy_confirm

    out = str(tmp_path / "board.csv")
    pd.DataFrame(
        {
            "league": ["NFL", "WNBA"],
            "market": ["receiving yards", "AST"],
            "slack": [0.9, 0.8],
            "ships": [True, True],
        }
    ).to_csv(out, index=False)
    requested = [("NFL", "receiving yards"), ("NFL", "rushing yards")]
    monkeypatch.setattr(
        sweep,
        "_select_board_cells",
        lambda *args: (dict.fromkeys(requested, ("SkewNormal",)), [], []),
    )
    monkeypatch.setattr(sweep, "_cell_trial_count", lambda lg, mkt, families, max_trials: 8)
    monkeypatch.setattr(sweep, "search_cell", _cell_frame)
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda board: None)
    monkeypatch.setattr(sweep, "_print_board_rollup", lambda board: None)
    confirmed = {}
    monkeypatch.setattr(
        model_strategy_confirm,
        "run_confirm",
        lambda board, *, yes, max_nominees: confirmed.update(
            cells=set(zip(board["league"], board["market"], strict=True)),
            yes=yes,
            max_nominees=max_nominees,
        ),
    )

    result = CliRunner().invoke(
        sweep.main,
        ["--league", "NFL", "--resume", "--confirm", "--yes", "--out", out],
    )

    assert result.exit_code == 0, result.output
    assert confirmed == {"cells": set(requested), "yes": True, "max_nominees": None}
    persisted = pd.read_csv(out)
    assert set(zip(persisted["league"], persisted["market"], strict=True)) == {
        *requested,
        ("WNBA", "AST"),
    }


def test_cli_single_cell_upserts_into_existing_board(monkeypatch, tmp_path):
    """A single-cell run merges into the board file — replacing that cell's prior rows, keeping the
    others — so it refreshes the living board instead of clobbering it.
    """
    from click.testing import CliRunner

    monkeypatch.setattr(
        sweep, "_cell_families", lambda lg, mkt, dist_class=sweep._DIST_CLASS_ALL: ("SkewNormal",)
    )
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    out = str(tmp_path / "board.csv")
    runner = CliRunner()

    def run(league, market):
        return runner.invoke(
            sweep.main,
            ["--league", league, "--market", market, "--out", out, "--max-trials", "6"],
        )

    assert run("WNBA", "AST").exit_code == 0
    assert run("NBA", "FGA").exit_code == 0
    board = pd.read_csv(out)
    assert set(zip(board["league"], board["market"], strict=True)) == {
        ("WNBA", "AST"),
        ("NBA", "FGA"),
    }
    n_other_cell = int((board["league"] == "NBA").sum())

    assert run("WNBA", "AST").exit_code == 0  # re-run replaces the cell's rows, not appends
    reswept = pd.read_csv(out)
    assert set(zip(reswept["league"], reswept["market"], strict=True)) == {
        ("WNBA", "AST"),
        ("NBA", "FGA"),
    }
    assert int((reswept["league"] == "NBA").sum()) == n_other_cell


# --- ledger discounts, confirm risk, calibrated expectations ---------------------------------


def _ledger_row(**overrides):
    """One confirm-ledger row (the model_stats echo confirm.py records per nominee)."""
    row = {
        "recorded_at": "2026-07-28T00:00:00",
        "strategy_slug": "SkewNormal",
        "source": "board slack +0.100",
        "league": "WNBA",
        "market": "AST",
        "distribution": "SkewNormal",
        "strategy_controls_json": '{"corner":0}',
        "dispersion_cal": 1.0,
        "ship": False,
        "g4_pit_ks": 0.06,
        "g2_star_z": 1.5,
        "g3_bench_z": 0.8,
        "g5_ece_debiased": 0.04,
        "g1_brier_diff_ci_hi": 0.01,
    }
    row.update(overrides)
    return row


def _board_csv_row(controls_json):
    """The board-side row a ledger nominee joins against, with known gate values."""
    return {
        "league": "WNBA",
        "market": "AST",
        "strategy_slug": "SkewNormal",
        "controls_json": controls_json,
        "g4_pit_ks": 0.05,
        "g2_star_z": 1.0,
        "g3_bench_z": 1.0,
        "g5_ece_debiased": 0.03,
        "g1_brier_diff_ci_hi": 0.0,
    }


def test_ledger_gate_discounts_joins_board_drops_diverged_and_goes_inert_when_thin(tmp_path):
    out = tmp_path / "board.csv"
    ledger_rows, board_rows = [], []
    for i in range(9):
        controls = f'{{"corner":{i}}}'
        ledger_rows.append(_ledger_row(strategy_controls_json=controls))
        board_rows.append(_board_csv_row(controls))
    # A diverged confirm (dispersion_cal on its 0.1 floor) and a seed row absent from the board.
    ledger_rows.append(
        _ledger_row(strategy_controls_json='{"corner":90}', dispersion_cal=0.1000001, g4_pit_ks=9.0)
    )
    board_rows.append(_board_csv_row('{"corner":90}'))
    ledger_rows.append(_ledger_row(strategy_controls_json='{"corner":91}', source="seed/incumbent"))

    assert sweep._ledger_gate_discounts(str(out)) == {}  # no ledger file yet

    pd.DataFrame(ledger_rows).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    assert sweep._ledger_gate_discounts(None) == {}  # no board CSV to pair against
    pd.DataFrame(board_rows).to_csv(out, index=False)

    discounts = sweep._ledger_gate_discounts(str(out))
    assert set(discounts) == set(sweep._DISCOUNTED_GATES)
    assert discounts["g4_pit_ks"] == pytest.approx(0.01)
    assert discounts["g2_star_z"] == pytest.approx(0.5)
    assert discounts["g3_bench_z"] == pytest.approx(-0.2)  # a gate confirm lands better on
    assert discounts["g5_ece_debiased"] == pytest.approx(0.01)
    assert discounts["g1_brier_diff_ci_hi"] == pytest.approx(0.01)

    # 7 clean pairs + the diverged one: dropping the diverged row is what makes the ledger thin.
    thin = [*ledger_rows[:7], ledger_rows[9]]
    pd.DataFrame(thin).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    assert sweep._ledger_gate_discounts(str(out)) == {}


def test_ledger_gate_discounts_prefers_board_echo_columns():
    """Once confirm echoes ``board_*`` gate values into the ledger, no board CSV join is needed."""
    echo = {f"board_{gate}": 0.0 for gate in sweep._DISCOUNTED_GATES}
    rows = [_ledger_row(strategy_controls_json=f'{{"corner":{i}}}', **echo) for i in range(8)]
    pd.DataFrame(rows).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)

    discounts = sweep._ledger_gate_discounts(None)

    assert discounts["g4_pit_ks"] == pytest.approx(0.06)
    assert discounts["g2_star_z"] == pytest.approx(1.5)
    assert set(discounts) == set(sweep._DISCOUNTED_GATES)


def _gate_scalars(g4, g4_max):
    """Gate values whose g4 term binds min_gate_slack exactly (the others sit at full headroom)."""
    return {
        "g1_brier_diff_ci_hi": None,
        "g2_star_z": 0.0,
        "g3_bench_z": 0.0,
        "g4_pit_ks": g4,
        "g4_pit_ks_max": g4_max,
        "g5_ece_debiased": 0.0,
    }


def _scored_row(marker, family, slack, gates):
    return {
        "family": family,
        "strategy_slug": family,
        "structural_strategy": BASE_STRUCTURAL_STRATEGY,
        "controls_json": marker,
        "slack": slack,
        "ships": slack > 0,
        **gates,
    }


def test_rank_cell_board_discounted_slack_reranks_and_matches_slack_when_inert():
    """The same +0.02 g4 shift costs a thin-scale corner its rank while a wide-scale corner
    absorbs it; inert discounts leave the board byte-identical to the slack ranking; a failed
    corner stays ``-inf``; and discounting never lifts a row above its raw slack (the g6 bind
    the board columns don't carry survives via the cap).
    """
    scored = {
        "thin": _scored_row("thin-headroom", "SkewNormal", 0.10, _gate_scalars(0.045, 0.05)),
        "wide": _scored_row("wide-scale", "NegBin", 0.09, _gate_scalars(0.455, 0.5)),
        "g6": _scored_row("g6-bound", "NegBin", 0.02, _gate_scalars(0.01, 0.05)),
        "dead": {
            "family": "SkewNormal",
            "strategy_slug": "SkewNormal",
            "controls_json": "dead",
            "slack": sweep._FAILED_CORNER_SLACK,
            "ships": False,
        },
    }

    inert = sweep._rank_cell_board("WNBA", "AST", scored)
    assert inert["discounted_slack"].tolist() == inert["slack"].tolist()
    assert inert["controls_json"].tolist() == ["thin-headroom", "wide-scale", "g6-bound", "dead"]

    ranked = sweep._rank_cell_board("WNBA", "AST", scored, {"g4_pit_ks": 0.02})
    assert ranked["controls_json"].tolist() == ["wide-scale", "g6-bound", "thin-headroom", "dead"]
    by_marker = ranked.set_index("controls_json")
    assert by_marker.loc["thin-headroom", "discounted_slack"] == pytest.approx(-0.3)
    assert by_marker.loc["wide-scale", "discounted_slack"] == pytest.approx(0.05)
    # g6-bound: the g1-g5 recompute (0.4) exceeds the stored slack, so the cap holds it at 0.02.
    assert by_marker.loc["g6-bound", "discounted_slack"] == pytest.approx(0.02)
    assert by_marker.loc["dead", "discounted_slack"] == sweep._FAILED_CORNER_SLACK
    # Raw slack and raw gate values stay untouched — only the ranking column shifts.
    assert by_marker.loc["thin-headroom", "slack"] == pytest.approx(0.10)
    assert by_marker.loc["thin-headroom", "g4_pit_ks"] == pytest.approx(0.045)


def test_confirm_risk_flags_continuous_on_integer_target_and_g4_inside_inflation(monkeypatch):
    context = sweep._cell_context("WNBA", "AST")  # integer target via the fixed matrix contract
    sn = {"family": "SkewNormal", **_gate_scalars(0.01, 0.05)}
    assert sweep._confirm_risk(sn, context, {}) == "high"

    count_safe = {"family": "NegBin", **_gate_scalars(0.03, 0.05)}
    assert sweep._confirm_risk(count_safe, context, {"g4_pit_ks": 0.01}) == ""
    count_tight = {"family": "NegBin", **_gate_scalars(0.045, 0.05)}
    assert sweep._confirm_risk(count_tight, context, {"g4_pit_ks": 0.0115}) == "high"

    # On a non-integer target the continuous family alone is not a risk shape.
    monkeypatch.setattr(
        sweep, "_training_matrix_contract", lambda lg, mkt: (frozenset(), _MATRIX_SHA, False, 41.2)
    )
    assert sweep._confirm_risk(sn, sweep._cell_context("WNBA", "AST"), {}) == ""


def test_search_cell_threads_ledger_discounts_into_the_board(monkeypatch, tmp_path):
    seen = {}

    def fake_discounts(out):
        seen["out"] = out
        return {"g4_pit_ks": 0.02}

    monkeypatch.setattr(sweep, "_ledger_gate_discounts", fake_discounts)
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_known_good_corners", lambda context: [])
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    out = str(tmp_path / "board.csv")

    board = sweep.search_cell("WNBA", "AST", max_trials=4, out=out)

    assert seen["out"] == out
    # Every fake row carries g4 0.04/0.05, so the +0.02 shift prices each at (0.05-0.06)/0.05.
    assert board["discounted_slack"].tolist() == pytest.approx([-0.2] * len(board))
    assert (board["g4_pit_ks"] == 0.04).all()
    assert board["slack"].max() > 0  # raw slack survives beside the discounted ranking
    persisted = sweep._read_board(pathlib.Path(out))
    assert persisted["discounted_slack"].astype(float).tolist() == pytest.approx(
        [-0.2] * len(board)
    )


def test_read_board_backfills_the_ranking_columns_on_legacy_csvs(tmp_path):
    out = tmp_path / "board.csv"
    pd.DataFrame([{"league": "WNBA", "market": "AST", "slack": 0.1, "ships": True}]).to_csv(
        out, index=False
    )
    board = sweep._read_board(out)
    assert list(board.columns) == sweep._BOARD_COLUMNS
    assert board["discounted_slack"].isna().all() and board["confirm_risk"].isna().all()
    order = sweep._BOARD_COLUMNS.index
    assert order("ships") < order("discounted_slack") < order("confirm_risk")


def test_cell_summary_shows_discounted_slack_and_risk(capsys):
    board = pd.DataFrame(
        [
            {
                "league": "WNBA",
                "market": "AST",
                "strategy_slug": "SkewNormal",
                "structural_strategy": BASE_STRUCTURAL_STRATEGY,
                "controls_json": "{}",
                "slack": 0.1,
                "discounted_slack": 0.04,
                "confirm_risk": "high",
                "ships": False,
                "g4_pass": False,
            }
        ]
    )
    sweep._print_cell_summary(board)
    out = capsys.readouterr().out
    assert "disc slack" in out and "risk" in out
    corner_line = next(line for line in out.splitlines() if "SkewNormal" in line)
    assert "0.04" in corner_line and "high" in corner_line


def test_rollup_prints_calibrated_expectations_and_skips_thin_ledger(capsys):
    board = pd.DataFrame({"league": ["WNBA"], "market": ["AST"], "ships": [False]})

    sweep._print_board_rollup(board)  # no ledger at all
    assert "P(ship)" not in capsys.readouterr().out

    count_row = {"distribution": "NegBin", "strategy_slug": "NegBin"}
    rows = (
        [_ledger_row(source="board slack +0.030", ship=True)] * 2
        + [_ledger_row(source="board slack +0.030", ship=False)] * 2
        + [_ledger_row(source="board slack +0.100", ship=True, **count_row)] * 3
        + [_ledger_row(source="board slack +0.100", ship=False, **count_row)]
        + [_ledger_row(source="board slack +0.200", ship=False)]
        + [_ledger_row(source="seed/incumbent", ship=True)]  # no board slack → not usable
    )
    pd.DataFrame(rows).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    sweep._print_board_rollup(board)
    out = capsys.readouterr().out
    assert "P(ship)" in out and "(9 ledger nominees)" in out
    assert "50% (4)" in out  # continuous, <0.05
    assert "75% (4)" in out  # count, 0.05-0.15
    top_band = next(line for line in out.splitlines() if ">=0.15" in line)
    assert "0% (1)" in top_band and " - " in top_band  # no count observations in the band

    # A board_slack echo column takes precedence over parsing the source label.
    echoed = pd.DataFrame(rows).assign(
        source="rewritten label", board_slack=[0.03] * 4 + [0.10] * 4 + [0.20, None]
    )
    echoed.to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    sweep._print_board_rollup(board)
    assert "(9 ledger nominees)" in capsys.readouterr().out

    pd.DataFrame(rows[:7]).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    sweep._print_board_rollup(board)
    assert "P(ship)" not in capsys.readouterr().out
