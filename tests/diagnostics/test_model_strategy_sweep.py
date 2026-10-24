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

import dataclasses
import io
import json
import math
import pathlib
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import click
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.training.baselines import get_target_normalization
from sportstradamus.training.calibration import DEFAULT_BLENDING
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.model_strategy import (
    BASE_STRUCTURAL_STRATEGY,
    CAP_SERVE,
    MODEL_STRATEGY_MODEL_KEY,
    SPLIT_FINGERPRINT_CSV_COLUMN,
    SWEEP_CAPABILITIES,
    CellContext,
    InactiveStrategyArtifactError,
    artifact_identity_columns,
    build_artifact_identity,
    controls_json,
    corner_fingerprint,
    distribution_class,
    get_strategy,
    progress,
    registered_strategies,
    resolve_report_identity,
    strategies_for_cell,
    strategy_cli_args,
    strategy_controls,
    strategy_persistence_edits,
    sweep,
    tpe_search,
    validate_strategy_artifacts,
    validate_strategy_frame,
    validate_strategy_selection,
)
from sportstradamus.training.model_strategy.specs import (
    CONFIRM_EVIDENCE_CORNERS,
    INCUMBENT_CONTROL_DEFAULTS,
    MANDATORY_SWEEP_CORNERS,
)
from sportstradamus.training.posthoc import CDF_STAGE, POSTHOC_SLUGS, PROB_STAGE, STRUCTURAL_STAGE
from sportstradamus.training.role_specs import role_spec_for
from sportstradamus.training.structural_strategies import (
    AFFINE_STRATEGY as RUSHING,
)
from sportstradamus.training.structural_strategies import (
    TWO_PART_STRATEGY as RECEIVING,
)

pytestmark = pytest.mark.diagnostics

_MATRIX_SHA = "matrix-123"


def _dpo_controls(dispersion: str, blending: str, posthoc: str) -> dict[str, str]:
    return {
        "dist": "DPO",
        "dist_training_loss": "nll",
        "count_dispersion_objective": dispersion,
        "blending_loss_fn": blending,
        "posthoc": posthoc,
    }


# Stands in for the per-cell trial ceiling wherever a test is not about the ceiling itself.
_TIMEOUT = sweep._MEDITATE_TRIAL_TIMEOUT_S
# The role×position two-part strategy is gated by the per-(league, market) role registry; the
# affine strategy needs only a ``Player position`` column. A structural spec enrolls — and now
# also ranks — only for a cell whose real matrix carries its grouping columns. These fixtures
# give each registered cell those columns.
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


def _fake_run_and_score(league, market, family, corner, timeout_s=None):
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

    def run_and_score(league, market, family, corner, timeout_s=None):
        trained.append((family, dict(corner)))
        return _fake_run_and_score(league, market, family, corner, timeout_s)

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
        "diagnostics": {"n_authentic_validation": 0},
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
    # SN: 4 norms × 2 dist-loss × 2 sn-param × 3 blend × 7 posthoc.
    assert math.prod(len(v) for v in sn.axes.values()) == 336
    # Mixture: 1 dist × 4 norms × 3 blend × 6 posthoc — no loss axis (lightgbmlss rejects crps
    # components, so the pinned nll is the only trainable value).
    assert math.prod(len(v) for v in mix.axes.values()) == 72
    assert mix.fixed_controls["dist_training_loss"] == "nll"
    # 1 dist × 2 mode × 2 disp × 3 blend × 6 posthoc.
    assert math.prod(len(v) for v in zinb.axes.values()) == 72
    assert math.prod(len(v) for v in negbin.axes.values()) == 36  # 1 dist × 2 disp × 3 blend × 6
    assert math.prod(len(v) for v in dpo.axes.values()) == 36
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
        "dist_training_loss": "dist_training_loss",
        "zinb_mode": "zinb_mode",
        "count_dispersion_objective": "count_dispersion_objective",
        "blending_loss_fn": "blending",
        "posthoc": "posthoc",
    }
    assert negbin.persist == {
        "dist": "dist",
        "dist_training_loss": "dist_training_loss",
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
    decided elsewhere: the confirm walk's calibrated fallback persists ``hpo_selection=calibrated``
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
    """Both seed registers name real cells and real corners of their declared family.

    Drift between a seed and its family's declared grid fails loud at import; this pins the same
    contract for both registers so an edit to either side is caught here too.
    """
    assert MANDATORY_SWEEP_CORNERS and CONFIRM_EVIDENCE_CORNERS
    for league, market, slug, controls in (*MANDATORY_SWEEP_CORNERS, *CONFIRM_EVIDENCE_CORNERS):
        assert market in ALL_MARKETS[league]
        assert controls in strategy_controls(get_strategy(slug))


def test_every_count_spec_pins_the_only_loss_its_constructor_accepts():
    """No count corner may carry a loss other than ``nll``, on any axis or by inheritance.

    The count constructors accept ``nll`` alone. A count spec declares no ``dist_training_loss``
    axis, so before the pin ``meditate`` fell back to the *cell's* value — and on the six cells
    configured continuous with ``crps`` that reached the constructor and killed every count corner
    they had. Phrased against the registry rather than one spec so a new count family inherits it.
    """
    count_slugs = [
        slug
        for slug in ("SkewNormal", "Mixture", "ZINB", "NegBin", "DPO")
        if distribution_class(slug) == "count"
    ]
    assert count_slugs, "no count families registered"
    for slug in count_slugs:
        spec = get_strategy(slug)
        assert "dist_training_loss" not in spec.axes, f"{slug} must not sweep the loss"
        losses = {controls["dist_training_loss"] for controls in strategy_controls(spec)}
        assert losses == {"nll"}, f"{slug} corners carry {losses}"


def test_a_count_corner_carries_nll_down_both_meditate_paths(monkeypatch):
    """The sweep passes the loss as a flag; confirm persists it. Neither may inherit the cell's."""
    monkeypatch.setattr(
        sweep, "_training_matrix_contract", lambda lg, mkt: (frozenset(), _MATRIX_SHA, True, 4.0)
    )
    context = sweep._cell_context("WNBA", "OREB")
    spec = get_strategy("DPO")
    corner = strategy_controls(spec)[0]

    flags = strategy_cli_args(context, spec, corner)
    edits = strategy_persistence_edits(context, spec, corner)

    assert "--dist-training-loss" in flags
    assert flags[flags.index("--dist-training-loss") + 1] == "nll"
    assert edits["dist_training_loss"] == "nll"


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
            {"dist": "NegBin", "dist_training_loss": "nll", "count_dispersion_objective": "crps"},
            get_strategy("NegBin"),
        )
        == "none"
    )
    assert (
        sweep._dump_subdir(
            {"dist": "DPO", "dist_training_loss": "nll", "count_dispersion_objective": "pit_ks"},
            get_strategy("DPO"),
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
    # This cell's fake contract carries no columns, so neither structural spec is applicable.
    assert sweep._cell_families("NFL", "passing yards") == ("SkewNormal",)
    # A non-integer target keeps the count families out even at a mean well under the ceiling;
    # this cell does carry the structural columns, so both structural methods rank on it.
    assert sweep._cell_families("NFL", "receiving yards") == ("SkewNormal", RECEIVING, RUSHING)
    with pytest.raises(click.UsageError):
        sweep._cell_families("MLB", "hits allowed")
    # --dist-class filters families, not cells: every eligible cell keeps only that class's pool.
    assert sweep._cell_families("WNBA", "AST", "count") == ("ZINB", "NegBin", "DPO")
    assert sweep._cell_families("MLB", "pitcher strikeouts", "continuous") == ("SkewNormal",)
    assert sweep._cell_families("NFL", "passing yards", "count") == ()


def test_cell_families_carries_every_enrolled_structural_spec(monkeypatch):
    """A structural method enrolled for a cell also ranks on it — the pool is applicability alone.

    Both methods now hold their routing fixed across the holdout-blind calibration folds, so the
    scored corner is the corner that trained and there is nothing left to exclude. Both NFL yards
    cells enroll both structural specs today, which makes them where the inclusion has to bite.
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
        assert {RECEIVING, RUSHING} <= set(sweep._cell_families("NFL", market))


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
        == 36
    )
    assert sweep._cell_trial_count("MLB", "pitcher strikeouts", ("NegBin",), 48) == 36
    # ZINB + NegBin + DPO = 144 reachable, so the 48-trial budget binds.
    families = ("ZINB", "NegBin", "DPO")
    assert sweep._cell_trial_count("MLB", "pitcher strikeouts", families, 48) == 48
    assert sweep._cell_trial_count("MLB", "pitcher strikeouts", families, 200) == 144


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
    # 336 SkewNormal + 72 ZINB + 36 NegBin + 36 DPO — a low-mean integer target. Mixture
    # declares more but has no serve path, so SWEEP_CAPABILITIES keeps it out of the pool.
    tds = sweep._cell_context("NFL", "passing tds")
    assert tpe_search.reachable_corners(tds, sweep._cell_families("NFL", "passing tds")) == 480
    # 336 SkewNormal plus one fixed corner per enrolled structural spec: the mean-226.7 target is
    # over the count-admission ceiling, so no count family reaches this cell.
    yards = sweep._cell_context("NFL", "passing yards")
    assert tpe_search.reachable_corners(yards, sweep._cell_families("NFL", "passing yards")) == 338


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


def test_score_corner_reads_n_authentic_validation_from_diagnostics(monkeypatch):
    """The board's evidence-behind-the-weight column comes from the diagnostics block of the pickle
    the sweep's own deterministic retrain just wrote."""
    spec = get_strategy("SkewNormal")
    corner = {
        "dist": "SkewNormal",
        "normalization": "centered_additive_mean10",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending_loss_fn": "nll",
        "posthoc": "none",
    }
    monkeypatch.setattr(
        sweep, "gate_row", lambda df, pred_col, **kw: _canned_row(ship=True, g4_pit_ks=0.03)
    )
    monkeypatch.setattr(sweep, "apply_thresholds", lambda row: row)

    frame, model = _artifact(spec, "WNBA", "AST", corner)
    model["diagnostics"] = {"n_authentic_validation": 42}
    monkeypatch.setattr(sweep, "load_test_set", lambda path, col: frame)
    monkeypatch.setattr(sweep.pd, "read_pickle", lambda path: model)
    assert sweep._score_corner("WNBA", "AST", corner, spec)["n_authentic_validation"] == 42


def test_score_corner_decodes_zinb_with_none_and_no_skew(monkeypatch):
    """A ZINB corner scores through the same gate with ``decode_strategy=none``; a count
    dump has no ``skew_cal`` so it surfaces as 0.0.
    """
    captured = {}
    spec = get_strategy("ZINB")
    corner = {
        "dist": "ZINB",
        "dist_training_loss": "nll",
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

    ``_enqueued_corners`` is stubbed empty so the suggestion sequence depends only on the seeded
    sampler and the fake objective, not on whatever recipe stat_meta carries for the cell today;
    the enqueue path has its own test. With 21 evaluations of a 192-corner grid the sampler is
    expected to find the dominant normalization (worth +0.20 against 0.00 for the runner-up).
    """
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_enqueued_corners", lambda context: [])
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
        for league, market, _, controls in CONFIRM_EVIDENCE_CORNERS
        if (league, market) == ("NFL", "passing tds")
    )
    context = sweep._cell_context("NFL", "passing tds")
    enqueued = [(spec.slug, corner) for spec, corner in sweep._enqueued_corners(context)]
    assert enqueued.count(("DPO", seed)) == 1

    trained = []
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("ZINB", "NegBin", "DPO"))
    monkeypatch.setattr(sweep, "_run_and_score", _spy_run_and_score(trained))
    sweep.search_cell("NFL", "passing tds", max_trials=6)
    assert trained[0] == ("DPO", seed)


def test_frontier_corners_cover_normalization_by_shaping_posthoc_grid():
    """The frontier spans (normalization x shaping-posthoc) with the family defaults elsewhere.

    Gate evidence is axis-structured — normalization moves the mean gates (g2/g3/g6), the
    PIT-shaping posthoc moves Gate 4 — while TPE, optimizing one scalar, can leave whole frontier
    points unsampled (WNBA MIN's board-shipping ratio_meanyr+cdf_recal corner was never proposed:
    its only (ratio, cdf) observation was confounded with ``centered``). Enqueuing the frontier
    makes coverage of the two gate-relevant axes deterministic instead of sampler luck. The
    PIT-neutral class is represented by ``none`` alone: Gate 4 cannot tell a prob-stage
    recalibrator from ``none``, so extra members of that class buy no frontier information.
    """
    cell = CellContext("WNBA", "MIN", "SkewNormal", "continuous", frozenset(), _MATRIX_SHA)
    corners = sweep._frontier_corners(cell)
    spec = get_strategy("SkewNormal")
    # The frontier respects per-cell axis pruning: a normalization the cell cannot train
    # (ratio_projvol without a volume denominator) must not burn an enqueued trial on a
    # guaranteed crash.
    choices = tpe_search.cell_axis_choices(cell, spec)
    assert "ratio_projvol" not in choices["normalization"]
    shaping = tuple(p for p in choices["posthoc"] if p != "none" and p not in PROB_STAGE)
    sn_points = {(c["normalization"], c["posthoc"]) for s, c in corners if s.slug == "SkewNormal"}
    assert sn_points == {
        (norm, posthoc) for norm in choices["normalization"] for posthoc in ("none", *shaping)
    }
    defaults = INCUMBENT_CONTROL_DEFAULTS["SkewNormal"]
    for frontier_spec, controls in corners:
        assert not frontier_spec.structural
        assert controls in strategy_controls(frontier_spec)
        if frontier_spec.slug == "SkewNormal":
            for name in ("dist_training_loss", "sn_param", "blending_loss_fn"):
                assert controls[name] == defaults[name]
    dpo_points = {(c.get("normalization"), c["posthoc"]) for s, c in corners if s.slug == "DPO"}
    assert dpo_points == {(None, "none"), (None, "roe_mean"), (None, "isotonic_mean")}


def test_enqueued_corners_include_the_frontier_deduplicated():
    """Frontier corners enqueue after the seeds and never duplicate a fingerprint."""
    context = sweep._cell_context("WNBA", "MIN")
    enqueued = sweep._enqueued_corners(context)
    prints = [corner_fingerprint(s, c, str(context.matrix_sha256)) for s, c in enqueued]
    assert len(prints) == len(set(prints))
    frontier = {
        corner_fingerprint(s, c, str(context.matrix_sha256))
        for s, c in sweep._frontier_corners(context)
    }
    assert frontier <= set(prints)


def _quiet_progress():
    return progress.SearchProgress("WNBA", "MIN", 4, verbosity=progress.QUIET)


def _probe_scored_row(context, corner, slack):
    """A scored-dict entry for one SkewNormal corner, keyed like the search cache."""
    spec = get_strategy("SkewNormal")
    row = {
        **_fake_row("SkewNormal", corner, slack),
        "controls_json": controls_json(corner),
        "corner_fingerprint": corner_fingerprint(spec, corner, str(context.matrix_sha256)),
    }
    return row["corner_fingerprint"], row


def _probe_corner(norm, dl, bl, posthoc):
    return {
        "dist": "SkewNormal",
        "normalization": norm,
        "dist_training_loss": dl,
        "sn_param": "direct",
        "blending_loss_fn": bl,
        "posthoc": posthoc,
    }


def test_shaping_probes_cover_the_top_two_bases_with_the_cdf_corrector(monkeypatch):
    """brief L1': the ranking statistic is blind to the CDF-stage corrector (slack binds on a
    mean/skill gate on 97% of top rows, so cdf ties ``none``), so after the study the top
    :data:`sweep._SHAPING_PROBE_ANCHORS` distinct base recipes each get their cdf twin scored —
    deterministic coverage of the one mechanism the board cannot rank, budget-exempt.
    """
    context = sweep._cell_context("WNBA", "MIN")
    scored = dict(
        [
            _probe_scored_row(
                context, _probe_corner("ratio_meanyr", "nll", "crps_1se", "prob_recal_platt"), 0.06
            ),
            # Same base as the leader (posthoc differs) — one anchor, not two.
            _probe_scored_row(
                context, _probe_corner("ratio_meanyr", "nll", "crps_1se", "none"), 0.06
            ),
            _probe_scored_row(context, _probe_corner("ratio_meanyr", "crps", "nll", "none"), 0.044),
            # Third distinct base — beyond the anchor budget, never probed.
            _probe_scored_row(
                context, _probe_corner("centered_additive_mean10", "crps", "nll", "none"), 0.03
            ),
        ]
    )
    failed_fp, failed = _probe_scored_row(
        context, _probe_corner("centered_additive_eb_meanyr_k10", "nll", "nll", "none"), 0.9
    )
    failed["slack"] = sweep._FAILED_CORNER_SLACK
    scored[failed_fp] = failed

    trained = []
    monkeypatch.setattr(sweep, "_run_and_score", _spy_run_and_score(trained))
    probes = sweep._shaping_probes(
        context, ("SkewNormal",), dict(scored), scored, {}, timeout_s=60, progress=_quiet_progress()
    )

    assert [c["posthoc"] for _, c in trained] == ["cdf_recal_isotonic"] * 2
    assert [(c["normalization"], c["dist_training_loss"]) for _, c in trained] == [
        ("ratio_meanyr", "nll"),
        ("ratio_meanyr", "crps"),
    ]
    spec = get_strategy("SkewNormal")
    for _, corner in trained:
        assert corner in strategy_controls(spec)
        assert corner_fingerprint(spec, corner, str(context.matrix_sha256)) in probes


def test_shaping_probes_skip_covered_known_and_foreign_anchors(monkeypatch):
    """An anchor already carrying the corrector is covered; a probe fingerprint already evaluated
    is not re-bought; a family without the corrector in its posthoc axis is never probed.
    """
    context = sweep._cell_context("WNBA", "MIN")
    leader_fp, leader = _probe_scored_row(
        context, _probe_corner("ratio_meanyr", "nll", "crps_1se", "cdf_recal_isotonic"), 0.06
    )
    runner_fp, runner = _probe_scored_row(
        context, _probe_corner("ratio_meanyr", "crps", "nll", "none"), 0.044
    )
    probe_fp, probe_row = _probe_scored_row(
        context, _probe_corner("ratio_meanyr", "crps", "nll", "cdf_recal_isotonic"), 0.044
    )
    scored = {leader_fp: leader, runner_fp: runner}
    trained = []
    monkeypatch.setattr(sweep, "_run_and_score", _spy_run_and_score(trained))
    probes = sweep._shaping_probes(
        context,
        ("SkewNormal",),
        {**scored, probe_fp: probe_row},
        scored,
        {},
        timeout_s=60,
        progress=_quiet_progress(),
    )
    assert trained == [] and probes == {}

    count_fp, count_row = _probe_scored_row(
        context, _probe_corner("ratio_meanyr", "crps", "nll", "none"), 0.1
    )
    count_row["family"] = count_row["strategy_slug"] = "ZINB"
    probes = sweep._shaping_probes(
        context,
        ("ZINB",),
        {count_fp: count_row},
        {count_fp: count_row},
        {},
        timeout_s=60,
        progress=_quiet_progress(),
    )
    assert trained == [] and probes == {}


def test_search_cell_scores_the_leader_base_cdf_probe(monkeypatch):
    """End-to-end: after the budget is spent the top-2 bases carry cdf twins the study never ran."""
    spec = get_strategy("SkewNormal")
    enqueued = [
        (spec, _probe_corner("centered_additive_mean10", "nll", "crps", "none")),
        (spec, _probe_corner("centered_additive_mean10", "crps", "crps", "none")),
    ]
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_enqueued_corners", lambda context: list(enqueued))
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    board = sweep.search_cell("WNBA", "AST", max_trials=2)
    base_cols = ["normalization", "dist_training_loss", "blending_loss_fn"]
    cdf = board[board["posthoc"] == "cdf_recal_isotonic"]
    # Both enqueued bases outrank the incumbent's, so both get their cdf twin — and only they do.
    assert sorted(map(tuple, cdf[base_cols].to_numpy())) == [
        ("centered_additive_mean10", "crps", "crps"),
        ("centered_additive_mean10", "nll", "crps"),
    ]
    leader_base = board.iloc[0][base_cols].tolist()
    assert any((cdf[base_cols] == leader_base).all(axis="columns"))


def test_rank_cell_board_breaks_ties_deterministically_by_fingerprint():
    """brief L5: an exact slack tie is ordered by fingerprint, not by dict insertion — the repro
    test's slot-3 'coin flip' was an unstable sort over an insertion-ordered tie.
    """
    tied = {
        "b-corner": {
            **_scored_row("b", "SkewNormal", 0.05, _gate_scalars(0.02, 0.05)),
            "corner_fingerprint": "bbb",
        },
        "a-corner": {
            **_scored_row("a", "SkewNormal", 0.05, _gate_scalars(0.02, 0.05)),
            "corner_fingerprint": "aaa",
        },
        "c-corner": {
            **_scored_row("c", "SkewNormal", 0.10, _gate_scalars(0.01, 0.05)),
            "corner_fingerprint": "ccc",
        },
    }
    forward = sweep._rank_cell_board("WNBA", "AST", tied)
    reversed_in = sweep._rank_cell_board("WNBA", "AST", dict(reversed(list(tied.items()))))
    assert forward["corner_fingerprint"].tolist() == ["ccc", "aaa", "bbb"]
    assert reversed_in["corner_fingerprint"].tolist() == forward["corner_fingerprint"].tolist()


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

    def maybe_fail(league, market, corner, spec, timeout_s):
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
    sweep._run_deterministic_meditate("WNBA", "AST", sn, sn_spec, _TIMEOUT)
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

    sweep._run_deterministic_meditate("NFL", "receiving yards", sn, sn_spec, _TIMEOUT)
    # The retired --structural-strategy axis emits no selector flag for a base corner.
    assert "--structural-strategy" not in calls[-1]

    zinb = {
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending_loss_fn": "nll",
        "posthoc": "none",
    }
    sweep._run_deterministic_meditate(
        "MLB", "pitcher strikeouts", zinb, get_strategy("ZINB"), _TIMEOUT
    )
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
        sweep._run_deterministic_meditate("MLB", "pitcher strikeouts", count, spec, _TIMEOUT)
        command = calls[-1]
        assert command[command.index("--dist") + 1] == family
        assert "--target-normalization" not in command
        assert sweep._log_path("MLB", "pitcher strikeouts", count, spec).parent.name == "none"

    method_spec = get_strategy(RUSHING)
    method = dict(method_spec.fixed_controls)
    sweep._run_deterministic_meditate("NFL", "rushing yards", method, method_spec, _TIMEOUT)
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
                "dist_training_loss": "nll",
                "shipped": "withheld",
            },  # withheld but no cached matrix → missing
        },
        "MLB": {
            "pitcher strikeouts": {
                "dist": "ZINB",
                "dist_training_loss": "nll",
                "shipped": "withheld",
            },  # default board
            "hits allowed": {"dist": "Gamma", "shipped": "withheld"},  # excluded: unswept family
            "1st inning hits allowed": {
                "dist": "ZINB",
                "dist_training_loss": "nll",
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
        "MLB": {
            "pitcher strikeouts": {
                "dist": "ZINB",
                "dist_training_loss": "nll",
                "shipped": "withheld",
            }
        },
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

    sweep._run_board_mode("WNBA", True, "count", "/tmp/board.csv", False, False, 32, jobs=8)
    assert captured["incl"] is True and captured["dist_class"] == "count"
    assert captured["max_trials"] == 32
    assert captured["families_by_cell"] == {("WNBA", "AST"): ("ZINB",)}
    # One cell cannot use eight workers, and saying "8 at a time" over a one-cell board would be a lie.
    assert captured["jobs"] == 1
    out = capsys.readouterr().out
    assert "skip WNBA STL: no cached training matrix" in out
    assert "skip NFL passing yards: no applicable family under --dist-class count" in out
    assert "1 cells to sweep · 2 skipped · 1 at a time · <=12 deterministic trainings" in out


def _cell_frame(league, market, **kwargs):
    """A one-row board frame for a cell — the shape ``search_cell`` returns for the run_board tests.

    Reindexed to the full board schema exactly as ``_rank_cell_board`` does, so a stub can't drift
    into promising columns (``elapsed_s``) the real return value always carries.
    """
    del kwargs  # budget/families/cache are asserted by the tests that care
    return pd.DataFrame(
        {"league": [league], "market": [market], "slack": [0.1], "ships": [True]}
    ).reindex(columns=sweep._BOARD_COLUMNS)


_SN_CORNER = {
    "dist": "SkewNormal",
    "normalization": "ratio_meanyr",
    "dist_training_loss": "crps",
    "sn_param": "direct",
    "blending_loss_fn": "nll",
    "posthoc": "none",
}


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
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda b, **kw: None)

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
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda b, **kw: None)

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
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda board, **kw: None)

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


def test_a_widened_axis_gets_its_own_journal_instead_of_crashing(monkeypatch, tmp_path):
    """A new axis value changes a categorical's choices exactly the way a pool change does."""
    monkeypatch.setattr(tpe_search, "_STUDY_ROOT", tmp_path)
    before = tpe_search.cell_study("NBA", "DREB", ("SkewNormal", "ZINB"))
    sn = get_strategy("SkewNormal")
    widened = dataclasses.replace(
        sn, axes={**sn.axes, "posthoc": (*sn.axes["posthoc"], "prob_recal_widened")}
    )
    monkeypatch.setattr(
        tpe_search,
        "get_strategy",
        lambda slug: widened if slug == "SkewNormal" else get_strategy(slug),
    )
    after = tpe_search.cell_study("NBA", "DREB", ("SkewNormal", "ZINB"))

    assert before.study_name != after.study_name
    assert len(list(tmp_path.glob("NBA_DREB.*.log"))) == 2


def test_a_cell_killed_mid_search_keeps_the_corners_it_already_trained(monkeypatch, tmp_path):
    """Every scored corner reaches the board CSV immediately, not once the cell finishes.

    The Optuna journal replays which corners were proposed but not what they scored, so a cell that
    dies on its 30th corner would otherwise resume with nothing and retrain all 29.
    """
    cell = ("MLB", "pitcher strikeouts")
    out = tmp_path / "board.csv"
    trained = []
    spy = _spy_run_and_score(trained)

    def die_on_the_third(league, market, family, corner, timeout_s=None):
        if len(trained) == 2:
            raise KeyboardInterrupt
        return spy(league, market, family, corner, timeout_s)

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


def test_a_corner_re_proposed_within_a_run_is_reported_cached_not_retrained(monkeypatch):
    """``trained`` is per trial, not "has this corner ever trained" — the second answer is wrong.

    The flag drives the bar, the ships tally and the ETA's duration samples. Testing membership in
    the trained set marks a re-proposal as a retrain, which walked the bar past its own total and
    weighted one corner's duration by how often TPE happened to suggest it again.
    """
    flags: list[bool] = []
    scored = progress.SearchProgress.scored
    monkeypatch.setattr(
        progress.SearchProgress,
        "scored",
        lambda self, corner, **kw: (flags.append(kw["trained"]), scored(self, corner, **kw))[1],
    )
    trained: list[tuple[str, dict]] = []
    monkeypatch.setattr(sweep, "_run_and_score", _spy_run_and_score(trained))

    # A budget above NegBin's 20-corner grid cannot be spent, so the study runs on until patience
    # stops it and the re-proposals this is about are guaranteed rather than hoped for.
    board = sweep.search_cell("MLB", "pitcher strikeouts", families=("NegBin",), max_trials=40)

    assert len(flags) > len(trained), (
        "no corner was re-proposed; the assertion below proves nothing"
    )
    assert sum(flags) == len(trained) == len(board)


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
        "dist_training_loss": "nll",
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
        "dist=ZINB, dist_training_loss=nll, count_dispersion_objective=pit_ks, blending=nll, "
        "posthoc=none, zinb_mode=hurdle, target_normalization=none"
    )
    negbin_controls = {
        "dist": "NegBin",
        "dist_training_loss": "nll",
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
        "dist=NegBin, dist_training_loss=nll, count_dispersion_objective=crps, blending=crps, "
        "posthoc=roe_mean, target_normalization=none"
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
    # The budget bounds study trials; the incumbent backstop and the shaping probes are the two
    # documented retrains beyond it.
    assert 0 < len(board) <= 6 + 1 + sweep._SHAPING_PROBE_ANCHORS
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
        lambda board, *, yes, max_nominees, deadline_hours, fresh_only, auto_promote, min_model_weight: (
            seen.update(
                n=len(board),
                yes=yes,
                max_nominees=max_nominees,
                deadline_hours=deadline_hours,
                fresh_only=fresh_only,
                min_model_weight=min_model_weight,
            )
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
            "--confirm-hours",
            "12",
            "--confirm-fresh-only",
            "--min-model-weight",
            "0.3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["yes"] is True and 0 < seen["n"] <= 5 + 1 + sweep._SHAPING_PROBE_ANCHORS
    assert seen["max_nominees"] == 2
    assert seen["deadline_hours"] == 12.0 and seen["fresh_only"] is True
    assert seen["min_model_weight"] == 0.3


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
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda board, **kw: None)
    monkeypatch.setattr(sweep, "_print_board_rollup", lambda board: None)
    confirmed = {}
    monkeypatch.setattr(
        model_strategy_confirm,
        "run_confirm",
        lambda board, *, yes, max_nominees, deadline_hours, fresh_only, auto_promote, min_model_weight: (
            confirmed.update(
                cells=set(zip(board["league"], board["market"], strict=True)),
                yes=yes,
                max_nominees=max_nominees,
            )
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
        # Discount pairs are signature-scoped: a row recorded under rotated-away code is
        # dropped, so the synthetic rows must carry the live signature to count as pairs.
        "strategy_signature": get_strategy("SkewNormal").canonical_signature,
        "source": "board slack +0.100",
        "league": "WNBA",
        "market": "AST",
        "distribution": "SkewNormal",
        "n_validation": 1500,
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
    cells = [("WNBA", "AST"), ("WNBA", "BLK"), ("WNBA", "STL")]
    ledger_rows, board_rows = [], []
    # Shifts vary per pair, with an outlier last, so the pins below hold for the two fixed
    # quantiles only — a mean (or an OLS fit) would land elsewhere.
    for i, bump in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 20.0]):
        league, market = cells[i % 3]
        controls = f'{{"corner":{i}}}'
        ledger_rows.append(
            _ledger_row(
                league=league,
                market=market,
                strategy_controls_json=controls,
                g4_pit_ks=0.05 + 0.005 * bump,
                g2_star_z=1.0 + 0.1 * bump,
                g3_bench_z=1.0 - 0.05 * bump,
                g5_ece_debiased=0.03 + 0.002 * bump,
                g1_brier_diff_ci_hi=0.001 * bump,
            )
        )
        board_rows.append({**_board_csv_row(controls), "league": league, "market": market})
    # A diverged confirm (dispersion_cal on its 0.1 floor) and a seed row absent from the board.
    ledger_rows.append(
        _ledger_row(strategy_controls_json='{"corner":90}', dispersion_cal=0.1000001, g4_pit_ks=9.0)
    )
    board_rows.append(_board_csv_row('{"corner":90}'))
    ledger_rows.append(_ledger_row(strategy_controls_json='{"corner":91}', source="seed/incumbent"))

    assert sweep._ledger_gate_discounts(str(out)) == ({}, {})  # no ledger file yet

    pd.DataFrame(ledger_rows).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    assert sweep._ledger_gate_discounts(None) == ({}, {})  # no board CSV to pair against
    pd.DataFrame(board_rows).to_csv(out, index=False)

    rank, veto = sweep._ledger_gate_discounts(str(out))
    assert set(rank) == set(veto) == set(sweep._DISCOUNTED_GATES)
    # Rank discounts price the corner at the pessimistic q75 of the measured shifts; the veto
    # keeps the central q50, so a cell loses its nominations only when even the median
    # confirm-vs-board gap sinks every corner.
    assert rank["g4_pit_ks"] == pytest.approx(0.03)
    assert veto["g4_pit_ks"] == pytest.approx(0.02)
    assert rank["g2_star_z"] == pytest.approx(0.6)
    assert veto["g2_star_z"] == pytest.approx(0.4)
    assert rank["g3_bench_z"] == pytest.approx(-0.1)  # a gate confirm lands better on
    assert veto["g3_bench_z"] == pytest.approx(-0.2)
    assert veto["g5_ece_debiased"] == pytest.approx(0.008)
    assert veto["g1_brier_diff_ci_hi"] == pytest.approx(0.004)

    # 7 clean pairs + the diverged one: dropping the diverged row is what makes the ledger thin.
    thin = [*ledger_rows[:7], ledger_rows[9]]
    pd.DataFrame(thin).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    assert sweep._ledger_gate_discounts(str(out)) == ({}, {})

    # Nine pairs but only two distinct cells: row volume alone is not evidence — a per-cell
    # quirk (one campaign's confirms) must not become every cell's discount.
    two_cell_rows = [
        {**row, "league": cells[i % 2][0], "market": cells[i % 2][1]}
        for i, row in enumerate(ledger_rows[:9])
    ]
    pd.DataFrame(two_cell_rows).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    two_cell_board = [
        {**row, "league": cells[i % 2][0], "market": cells[i % 2][1]}
        for i, row in enumerate(board_rows[:9])
    ]
    pd.DataFrame(two_cell_board).to_csv(out, index=False)
    assert sweep._ledger_gate_discounts(str(out)) == ({}, {})


def test_ledger_gate_discounts_are_flat_quantile_scalars_via_echo_columns():
    """Every gate — g4 included — takes the same pooled empirical-quantile treatment; the
    ``board_*`` echo columns pair the rows, so no board CSV join is needed (``out=None``).

    The old conditional-OLS g4 model was deleted on the researcher's verdict
    (/tmp/researcher_ledger_discount_sharpening.md R1): on the 15 live pairs it measured 2.6x
    worse out-of-cell than a flat median, its 1/sqrt(n) regressor was dead (n is constant
    within every cell), and its single-row family effects carried hat values of 1.0.
    """
    echo = {f"board_{gate}": 0.0 for gate in sweep._DISCOUNTED_GATES}
    cells = [("MLB", "hits"), ("MLB", "runs"), ("NBA", "PTS"), ("NBA", "AST")]
    # Eight pairs across four cells; g4 shifts 0.01..0.08 pin q50 = 0.045 and q75 = 0.0625.
    rows = [
        _ledger_row(
            league=league,
            market=market,
            n_validation=n,
            g4_pit_ks=0.01 * (i + 1),
            g2_star_z=1.0 + (0.5 if i % 2 == 0 else -0.5),
            **echo,
        )
        for i, (n, (league, market)) in enumerate((300 + 100 * k, cells[k % 4]) for k in range(8))
    ]
    pd.DataFrame(rows).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)

    rank, veto = sweep._ledger_gate_discounts(None)

    assert set(rank) == set(veto) == set(sweep._DISCOUNTED_GATES)
    assert all(isinstance(value, float) for value in {**rank, **veto}.values())
    assert rank["g4_pit_ks"] == pytest.approx(0.0625)
    assert veto["g4_pit_ks"] == pytest.approx(0.045)
    assert rank["g4_pit_ks"] >= veto["g4_pit_ks"]
    assert rank["g2_star_z"] == pytest.approx(1.5)
    assert veto["g2_star_z"] == pytest.approx(1.0)


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


def _scored_row(marker, family, slack, gates, n=None):
    return {
        "family": family,
        "strategy_slug": family,
        "structural_strategy": BASE_STRUCTURAL_STRATEGY,
        "controls_json": marker,
        "corner_fingerprint": marker,
        "slack": slack,
        "ships": slack > 0,
        "n": n,
        **gates,
    }


def test_rank_cell_board_discounted_slack_reranks_and_matches_slack_when_inert():
    """The same +0.02 g4 shift costs a thin-scale corner its rank while a wide-scale corner
    absorbs it; inert discounts leave the board byte-identical to the slack ranking; a failed
    corner stays ``-inf``; and discounting never lifts a row above its raw slack (the g6 bind
    the board columns don't carry survives via the cap).
    """
    scored = {
        "thin": _scored_row("thin-headroom", "SkewNormal", 0.10, _gate_scalars(0.045, 0.05), n=500),
        "wide": _scored_row("wide-scale", "NegBin", 0.09, _gate_scalars(0.455, 0.5), n=5000),
        "g6": _scored_row("g6-bound", "NegBin", 0.02, _gate_scalars(0.01, 0.05)),
        "dead": {
            "family": "SkewNormal",
            "strategy_slug": "SkewNormal",
            "controls_json": "dead",
            "corner_fingerprint": "dead",
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
    # With no veto discounts given, the veto column mirrors the raw slack (inert veto).
    assert by_marker.loc["thin-headroom", "veto_slack"] == pytest.approx(0.10)

    # The veto column prices the same rows under the milder q50 discounts: the thin-headroom
    # corner survives the veto (+0.045 + 0.004 < 0.05) while ranking already wrote it off.
    two_tier = sweep._rank_cell_board(
        "WNBA", "AST", scored, {"g4_pit_ks": 0.02}, veto_discounts={"g4_pit_ks": 0.004}
    )
    tiers = two_tier.set_index("controls_json")
    assert tiers.loc["thin-headroom", "discounted_slack"] == pytest.approx(-0.3)
    assert tiers.loc["thin-headroom", "veto_slack"] == pytest.approx(0.02)
    assert tiers.loc["wide-scale", "veto_slack"] == pytest.approx(0.082)


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


def _priced_against(monkeypatch, incumbent_slack, *challenger_slacks):
    """Price a synthetic cell's corners against an incumbent scored at ``incumbent_slack``."""
    spec = get_strategy("SkewNormal")
    corners = strategy_controls(spec)[: 1 + len(challenger_slacks)]
    monkeypatch.setattr(sweep, "_incumbent_corner", lambda context: (spec, corners[0]))
    context = sweep._cell_context("WNBA", "AST")
    return sweep._with_incumbent_margin(
        [
            {
                "corner_fingerprint": corner_fingerprint(
                    spec, controls, str(context.matrix_sha256)
                ),
                "slack": slack,
                "discounted_slack": slack,
            }
            for controls, slack in zip(corners, (incumbent_slack, *challenger_slacks), strict=True)
        ],
        context,
    )


def test_board_prices_every_corner_against_the_cells_serving_recipe(monkeypatch):
    """``margin_vs_incumbent`` is a corner's slack less the serving recipe's, on one frame.

    Slack alone ranks a corner against the gates and the book and never against the model it would
    replace, so a supersession ranked on slack is ranked on the wrong quantity.
    """
    priced = _priced_against(monkeypatch, 0.10, 0.25, -0.05)

    assert [row["is_incumbent"] for row in priced] == [True, False, False]
    assert [row["margin_vs_incumbent"] for row in priced] == pytest.approx([0.0, 0.15, -0.15])


@pytest.mark.parametrize(
    "baseline_slack, why",
    [
        (sweep._FAILED_CORNER_SLACK, "the fit crashed, so it measured nothing"),
        (-12.0, "it scored, but disagrees with the cell shipping today"),
    ],
)
def test_a_baseline_that_fails_its_own_gates_leaves_every_margin_unknown(
    monkeypatch, baseline_slack, why
):
    """An incumbent that does not clear the gates is unknown, not a huge win.

    It is serving because it passed all six at full HPO, so a negative re-measure indicts the
    measurement, not the model. Subtracting it anyway inverts the ranking: the first supersession
    pass put NHL points top of the live lane at margin +12.4 off a −12.0 baseline, and it held.
    """
    priced = _priced_against(monkeypatch, baseline_slack, 0.25)

    assert [row["is_incumbent"] for row in priced] == [True, False], why
    assert all(math.isnan(row["margin_vs_incumbent"]) for row in priced), why


def test_search_cell_scores_the_incumbent_even_when_the_study_never_proposes_it(monkeypatch):
    """The baseline is guaranteed, not hoped for.

    ``enqueue_trial`` skips params its study already holds, so a resumed cell whose cached baseline
    row had gone inadmissible silently lost it — 25 of 73 live cells on the board carry no incumbent
    row for that reason. The backstop costs one retrain beyond ``max_trials``.
    """
    sn = get_strategy("SkewNormal")
    monkeypatch.setattr(sweep, "_incumbent_corner", lambda ctx: (sn, strategy_controls(sn)[0]))
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_enqueued_corners", lambda context: [])
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)

    board = sweep.search_cell("WNBA", "AST", max_trials=2)

    assert board["is_incumbent"].sum() == 1
    assert board["margin_vs_incumbent"].notna().all()


def test_a_family_restricted_sweep_prices_no_cross_family_incumbent(monkeypatch):
    """A sweep not allowed to train the incumbent's family cannot compare anything to it."""
    zinb = get_strategy("ZINB")
    monkeypatch.setattr(sweep, "_incumbent_corner", lambda ctx: (zinb, strategy_controls(zinb)[0]))
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_enqueued_corners", lambda context: [])
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)

    board = sweep.search_cell("WNBA", "AST", max_trials=2)

    assert not board["is_incumbent"].any()
    assert board["margin_vs_incumbent"].isna().all()


def test_search_cell_threads_ledger_discounts_into_the_board(monkeypatch, tmp_path):
    seen = {}

    def fake_discounts(out):
        seen["out"] = out
        return {"g4_pit_ks": 0.02}, {"g4_pit_ks": 0.005}

    monkeypatch.setattr(sweep, "_ledger_gate_discounts", fake_discounts)
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_enqueued_corners", lambda context: [])
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    out = str(tmp_path / "board.csv")

    board = sweep.search_cell("WNBA", "AST", max_trials=4, out=out)

    assert seen["out"] == out
    # Every fake row carries g4 0.04/0.05, so the +0.02 shift prices each at (0.05-0.06)/0.05.
    assert board["discounted_slack"].tolist() == pytest.approx([-0.2] * len(board))
    # The milder veto shift prices each row at (0.05-0.045)/0.05, capped at its raw slack.
    expected_veto = [min(0.1, raw) for raw in board["slack"]]
    assert board["veto_slack"].tolist() == pytest.approx(expected_veto)
    assert (board["g4_pit_ks"] == 0.04).all()
    assert board["slack"].max() > 0  # raw slack survives beside the discounted ranking
    persisted = sweep._read_board(pathlib.Path(out))
    assert persisted["discounted_slack"].astype(float).tolist() == pytest.approx(
        [-0.2] * len(board)
    )
    assert persisted["veto_slack"].astype(float).tolist() == pytest.approx(expected_veto)


def test_read_board_backfills_the_ranking_columns_on_legacy_csvs(tmp_path):
    out = tmp_path / "board.csv"
    pd.DataFrame([{"league": "WNBA", "market": "AST", "slack": 0.1, "ships": True}]).to_csv(
        out, index=False
    )
    board = sweep._read_board(out)
    assert list(board.columns) == sweep._BOARD_COLUMNS
    assert board["discounted_slack"].isna().all() and board["confirm_risk"].isna().all()
    assert board["veto_slack"].isna().all()
    order = sweep._BOARD_COLUMNS.index
    assert order("ships") < order("discounted_slack") < order("veto_slack") < order("confirm_risk")


def test_cell_summary_shows_discounted_slack_and_risk(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "120")
    board = pd.DataFrame(
        [
            {
                "league": "WNBA",
                "market": "AST",
                "strategy_slug": "SkewNormal",
                "structural_strategy": BASE_STRUCTURAL_STRATEGY,
                "controls_json": controls_json(
                    {
                        "dist": "SkewNormal",
                        "normalization": "centered_additive_eb_meanyr_k10",
                        "dist_training_loss": "crps",
                        "sn_param": "direct",
                        "blending_loss_fn": "crps",
                        "posthoc": "isotonic_mean",
                    }
                ),
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
    # The raw controls JSON is what used to run this table past 250 columns.
    assert "controls_json" not in out and '{"' not in out
    assert max(len(line) for line in out.splitlines()) <= 120


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


def test_compress_corner_keeps_every_real_axis_value_distinct():
    """No two registered axis values may elide to the same token.

    A left-truncating shortener collides ``centered_additive_mean10`` with
    ``centered_additive_eb_meanyr_k10`` — the two values the normalization axis turns on most —
    which would silently relabel one recipe as the other on every progress line.
    """
    values = {
        value
        for spec in registered_strategies()
        for controls in strategy_controls(spec)
        for value in controls.values()
    }
    assert len(values) > 10, "the catalog should expose many axis values to collide"
    shortened = {value: progress._shorten(value) for value in values}
    assert len(set(shortened.values())) == len(values), "two axis values elided to one token"
    assert all(len(token) <= progress._MAX_TOKEN_CHARS for token in shortened.values())


def test_compress_corner_orders_family_first_and_clamps_to_budget():
    corner = {
        "posthoc": "prob_recal_isotonic",
        "blending_loss_fn": "crps",
        "dist": "SkewNormal",
        "normalization": "centered_additive_eb_meanyr_k10",
    }
    label = progress.compress_corner(sweep._axis_order(corner), 200)
    assert label.startswith("SkewNormal centered")  # family first, then registry axis order
    assert "posthoc=" not in label  # axis names are dropped; values are positional
    assert len(progress.compress_corner(sweep._axis_order(corner), 20)) == 20


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--yes"], "require --confirm"),
        (["--confirm-nominees", "2"], "require --confirm"),
        (["--confirm-hours", "4"], "require --confirm"),
        (["--confirm-fresh-only"], "require --confirm"),
        (["--confirm-auto-promote"], "require --confirm"),
        (["--min-model-weight", "0.3"], "require --confirm"),
        (
            ["--confirm", "--confirm-fresh-only", "--confirm-auto-promote"],
            "only affects the live lane",
        ),
        (["-q", "-v"], "mutually exclusive"),
        (["--market", "AST"], "pass --league with --market"),
    ],
)
def test_cli_rejects_flags_that_would_silently_do_nothing(argv, message):
    """Each of these was accepted and ignored, which on an unattended run reads as having applied."""
    result = CliRunner().invoke(sweep.main, argv)
    assert result.exit_code == 2 and message in result.output


def test_include_shipped_with_market_is_a_note_not_an_error(monkeypatch, tmp_path):
    """A named cell is swept whatever its shipped state, so the flag is redundant — never fatal.

    Erroring here would break `--league X --market Y --include-shipped`, which worked (the withheld
    filter lives in board-cell selection, which the single-cell path never reaches).
    """
    out = str(tmp_path / "board.csv")
    result = CliRunner().invoke(
        sweep.main,
        ["--league", "WNBA", "--market", "AST", "--include-shipped", "--dry-run", "--out", out],
    )
    assert result.exit_code == 0, result.output
    assert "redundant" in result.output and "[dry-run]" in result.output


def test_dry_run_reports_what_resume_would_reuse(monkeypatch, tmp_path):
    """``--dry-run --resume`` prints the per-cell reuse count its help promises."""
    out = str(tmp_path / "board.csv")
    controls = dict(_SN_CORNER)
    pd.DataFrame([_board_row("WNBA", "AST", "SkewNormal", controls)]).to_csv(out, index=False)
    monkeypatch.setattr(
        sweep, "_select_board_cells", lambda *a: ({("WNBA", "AST"): ("SkewNormal",)}, [], [])
    )

    result = CliRunner().invoke(sweep.main, ["--dry-run", "--resume", "--out", out])

    assert result.exit_code == 0, result.output
    assert "[dry-run]" in result.output
    assert "1 prior rows reusable" in result.output


def test_progress_writes_no_carriage_returns_when_redirected(monkeypatch):
    """The overnight driver redirects to a log; a bar rendered there would corrupt every line."""
    monkeypatch.setenv("COLUMNS", "120")
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    bar = progress.SearchProgress("WNBA", "AST", 4, seed_seconds=30.0)
    bar.starting(_SN_CORNER)
    bar.scored(_SN_CORNER, verdict="SHIP", slack=0.2, elapsed_s=31.0, trained=True)
    bar.close()

    written = stream.getvalue()
    assert bar.bar.disable, "tqdm must self-disable off a TTY"
    assert "\r" not in written
    assert "SHIP" in written and max(len(line) for line in written.splitlines()) <= 120


def test_board_eta_prices_untimed_cells_at_the_fallback_not_zero():
    """A cell with no history of its own costs the all-cells mean, never nothing.

    Summing untimed cells as free understated a 29-cell board by 29x while still labelling the
    result a ceiling — the one thing a time estimate must not do.
    """
    cells = [("NBA", "FGA"), ("WNBA", "AST"), ("NFL", "carries")]
    budgets = dict.fromkeys(cells, 48)
    timed_one = {("NBA", "FGA"): 300.0}

    eta = sweep._board_eta(cells, budgets, timed_one, fallback=300.0)

    assert eta == 3 * 48 * 300.0, "untimed cells must be priced, not skipped"
    # With no prior board, the cells this run already finished stand in as the fallback — else a
    # first run reports nothing the whole way through despite having timed most of its own cells.
    assert sweep._board_eta(cells, budgets, timed_one, fallback=None) == 3 * 48 * 300.0
    # Nothing measured anywhere is the one honest "no estimate" case.
    assert sweep._board_eta(cells, budgets, {}, fallback=None) is None
    assert sweep._board_eta([], budgets, timed_one, fallback=300.0) is None


def test_board_countdown_reports_running_total_and_remaining(capsys, monkeypatch):
    """A multi-day sit needs elapsed-so-far and time-left on one line, not just at the end."""
    monkeypatch.setenv("COLUMNS", "120")
    left = [("NBA", f"M{i}") for i in range(8)]
    sweep._echo_board_countdown("NBA FGA", "SHIP +0.076", 720.0, 7800.0, left, 9600.0)
    line = capsys.readouterr().out
    assert "NBA FGA SHIP +0.076 in 12m" in line
    assert "run 2h10m" in line and "8 cells left" in line and "<=2h40m" in line

    sweep._echo_board_countdown("NBA FGA", "no ship (best -0.096)", 720.0, 7800.0, [], None)
    assert "cells left" not in capsys.readouterr().out


def test_upsert_does_not_warn_on_a_board_predating_a_column(tmp_path, recwarn):
    """A legacy board's back-filled columns are all-NA objects; concatenating them warned per corner.

    ``_read_board`` fills any column a board predates with a ``pd.NA`` scalar, so the upsert ran
    straight into pandas 2.x's empty/all-NA concat deprecation — once for every scored corner, which
    is thousands of times across a board run.
    """
    out = tmp_path / "board.csv"
    legacy = pd.DataFrame([_board_row("WNBA", "AST", "SkewNormal", dict(_SN_CORNER))])
    legacy.drop(columns=["elapsed_s", "failure"], errors="ignore").to_csv(out, index=False)
    fresh = pd.DataFrame(
        [{**_board_row("NBA", "AST", "SkewNormal", dict(_SN_CORNER)), "elapsed_s": 18.0}]
    ).reindex(columns=sweep._BOARD_COLUMNS)

    sweep._upsert_cell(fresh, str(out))

    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]
    written = sweep._read_board(out)
    assert list(written.columns) == sweep._BOARD_COLUMNS  # dropped columns come back on reindex
    cells = zip(written["league"], written["market"], strict=True)
    assert set(cells) == {("WNBA", "AST"), ("NBA", "AST")}


def test_progress_prints_one_ship_line_per_cell_not_one_per_shipping_corner(monkeypatch):
    """A later ship that beat nothing is noise — the bar carries `best`, the summary lists them all."""
    monkeypatch.setenv("COLUMNS", "100")
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    bar = progress.SearchProgress("NBA", "AST", 48)
    for _ in range(4):
        bar.scored(_SN_CORNER, verdict="SHIP", slack=0.101, elapsed_s=18.0, trained=True)
    bar.scored(_SN_CORNER, verdict="SHIP", slack=0.140, elapsed_s=20.0, trained=True)
    bar.scored(_SN_CORNER, verdict="KILL: g4", slack=-0.3, elapsed_s=12.0, trained=True)
    bar.close()

    printed = [line for line in stream.getvalue().splitlines() if "SHIP" in line or "KILL" in line]
    assert len(printed) == 2, printed  # the first ship and the better one; four repeats collapse
    assert "+0.101" in printed[0] and "+0.140" in printed[1]


def test_progress_omits_the_eta_until_it_has_one(monkeypatch):
    """With no seed and too few samples the bar showed a bare `<=?`; it now shows nothing."""
    monkeypatch.setenv("COLUMNS", "100")
    bar = progress.SearchProgress("NBA", "AST", 48)
    bar.scored(_SN_CORNER, verdict="KILL: g4", slack=-0.3, elapsed_s=12.0, trained=True)
    assert "<=" not in bar.bar.postfix and "?" not in bar.bar.postfix

    for _ in range(progress._ETA_MIN_CORNERS):
        bar.scored(_SN_CORNER, verdict="KILL: g4", slack=-0.3, elapsed_s=12.0, trained=True)
    assert "<=" in bar.bar.postfix
    bar.close()


def test_run_board_returns_cells_in_board_order_not_completion_order(monkeypatch, tmp_path):
    """A slow cell must not sink to the bottom of the board just for having taken longer.

    With ``jobs > 1`` the cells finish in whatever order they happen to take, so results are written
    into a pre-sized list by position rather than appended as they land.
    """
    cells = [("WNBA", "AST"), ("WNBA", "PTS"), ("WNBA", "REB"), ("WNBA", "STL")]
    delays = dict(zip(cells, [0.20, 0.01, 0.15, 0.02], strict=True))

    def slow_cell(league, market, **kwargs):
        time.sleep(delays[(league, market)])
        return _cell_frame(league, market)

    monkeypatch.setattr(sweep, "search_cell", slow_cell)
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda b, **kw: None)

    board = sweep.run_board(cells, out=str(tmp_path / "board.csv"), jobs=4)
    assert list(zip(board["league"], board["market"], strict=True)) == cells


def test_concurrent_upserts_keep_every_cell(monkeypatch, tmp_path):
    """The board CSV is a read-modify-write of the whole file, and every cell rewrites it.

    Unserialized, two cells that read the same prior board both write it back and the later write
    drops the earlier cell's rows — silently, since each cell's own rows are always present.
    """
    out = str(tmp_path / "board.csv")
    cells = [("WNBA", f"M{i}") for i in range(12)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in [
            pool.submit(sweep._upsert_cell, _cell_frame(lg, mkt), out) for lg, mkt in cells
        ]:
            future.result()

    board = pd.read_csv(out)
    assert set(zip(board["league"], board["market"], strict=True)) == set(cells)


def test_one_cell_crashing_does_not_cost_the_other_cells(monkeypatch, tmp_path, capsys):
    """Before the pool a cell dying outside the per-corner catch ended the whole board run."""
    cells = [("WNBA", "AST"), ("WNBA", "PTS"), ("WNBA", "REB")]

    def sometimes_explode(league, market, **kwargs):
        if market == "PTS":
            raise RuntimeError("scoring blew up")
        return _cell_frame(league, market)

    monkeypatch.setattr(sweep, "search_cell", sometimes_explode)
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda b, **kw: None)

    board = sweep.run_board(cells, out=str(tmp_path / "board.csv"), jobs=3)
    assert list(board["market"]) == ["AST", "REB"]
    assert "cell failed — RuntimeError: scoring blew up" in capsys.readouterr().err


def test_a_ctrl_c_is_not_recorded_as_a_failed_corner(monkeypatch):
    """SIGINT reaches the child meditate but not the worker thread waiting on it.

    Recording the resulting non-zero exit as a failed corner would write one per in-flight cell and
    cache it under its fingerprint, so ``--resume`` would never retrain any of them.
    """
    monkeypatch.setattr(
        sweep,
        "_run_deterministic_meditate",
        lambda *a, **k: (_ for _ in ()).throw(
            subprocess.CalledProcessError(-signal.SIGINT, "meditate")
        ),
    )
    with pytest.raises(subprocess.CalledProcessError):
        sweep._run_and_score("WNBA", "AST", "SkewNormal", dict(_SN_CORNER), _TIMEOUT)

    # An ordinary non-zero exit is still the corner's own problem and still gets banked.
    monkeypatch.setattr(
        sweep,
        "_run_deterministic_meditate",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "meditate")),
    )
    row = sweep._run_and_score("WNBA", "AST", "SkewNormal", dict(_SN_CORNER), _TIMEOUT)
    assert row["failure"] == "CalledProcessError" and row["slack"] == sweep._FAILED_CORNER_SLACK


def test_cell_timeout_tracks_the_cell_and_never_exceeds_the_flat_ceiling(tmp_path):
    """Headroom over the cell's own slowest corner, clamped both ways.

    Anchored on the maximum rather than a quantile because the spread inside one cell is wide — real
    NBA BLK corners run 12 s to 906 s — so anything gentler cuts legitimate work.
    """
    out = str(tmp_path / "board.csv")
    assert sweep._cell_timeout_seconds(out, "NBA", "BLK") == sweep._MEDITATE_TRIAL_TIMEOUT_S

    def board_with(seconds):
        rows = [
            {**_board_row("NBA", "BLK", "SkewNormal", dict(_SN_CORNER)), "elapsed_s": value}
            for value in seconds
        ]
        pd.DataFrame(rows).reindex(columns=sweep._BOARD_COLUMNS).to_csv(out, index=False)

    board_with([9.0, 12.0, 15.0])  # a fast cell still tolerates one slow outlier
    assert sweep._cell_timeout_seconds(out, "NBA", "BLK") == sweep._TIMEOUT_FLOOR_S
    board_with([12.0, 100.0])
    assert sweep._cell_timeout_seconds(out, "NBA", "BLK") == 200
    board_with([12.0, 400.0])  # never looser than the flat ceiling
    assert sweep._cell_timeout_seconds(out, "NBA", "BLK") == sweep._MEDITATE_TRIAL_TIMEOUT_S
    # A cell with rows but no timings at all falls back rather than reading NaN as zero.
    assert sweep._cell_timeout_seconds(out, "NBA", "DREB") == sweep._MEDITATE_TRIAL_TIMEOUT_S


def test_a_timed_out_corner_does_not_raise_the_next_run_s_ceiling(tmp_path):
    """Only a corner that finished measures anything; a timed-out one measures the cap that killed it.

    Counting it compounds — an NBA DREB cap of 366 s produced a 733 s timeout, which would have set
    1466 s, then the ceiling — on precisely the cell the adaptive cap exists to bound. With no
    finished corner left to anchor on the cell falls back rather than inventing a number.
    """
    out = str(tmp_path / "board.csv")

    def board_with(*rows):
        frame = [
            {
                **_board_row("NBA", "DREB", "SkewNormal", dict(_SN_CORNER)),
                "elapsed_s": seconds,
                "failure": failure,
            }
            for seconds, failure in rows
        ]
        pd.DataFrame(frame).reindex(columns=sweep._BOARD_COLUMNS).to_csv(out, index=False)

    board_with((733.1, "TimeoutExpired"))
    assert sweep._cell_timeout_seconds(out, "NBA", "DREB") == sweep._MEDITATE_TRIAL_TIMEOUT_S
    # A real crash still timed something, and a finished corner beside the timeout still anchors it.
    board_with((733.1, "TimeoutExpired"), (110.0, None), (60.0, "CalledProcessError"))
    assert sweep._cell_timeout_seconds(out, "NBA", "DREB") == 220


def test_a_timed_out_corner_is_retried_on_resume_but_a_crash_is_not(tmp_path):
    """A crash is the corner's property and worth caching; a timeout is the machine's and is not.

    The ceiling is derived from the cell's own timings and cells now share the box, so the corner
    that ran long once may well not next time.
    """
    out = str(tmp_path / "board.csv")
    cell = ("MLB", "pitcher strikeouts")
    context = sweep._cell_context(*cell)
    controls = strategy_controls(get_strategy("NegBin"))
    rows = []
    for failure, corner in zip(["TimeoutExpired", "CalledProcessError"], controls, strict=False):
        row = _board_row(*cell, "NegBin", corner)
        rows.append({**row, "failure": failure, "matrix_hash": context.matrix_sha256})
    frame = pd.DataFrame(rows).reindex(columns=sweep._BOARD_COLUMNS)
    frame["eval_split"] = sweep.EVAL_SPLIT_CROSSFIT
    frame.to_csv(out, index=False)

    cached = sweep._cached_corners(out, *cell, context)
    kept = frame[frame["corner_fingerprint"].isin(cached)]
    assert list(kept["failure"]) == ["CalledProcessError"]


def test_a_hopeless_family_stops_being_trained(monkeypatch):
    """A family whose corners all trail the cell's best is budget spent on a known loser.

    Real boards spend 14% of their search there: 66% of non-winning families finish more than
    :data:`tpe_search.ABANDON_SLACK_GAP` behind, and every corner of them still trains.
    """
    cell = ("MLB", "pitcher strikeouts")
    trained = []

    def scored(league, market, family, corner, timeout_s=None):
        trained.append(family)
        slack = 0.4 if family == "NegBin" else -0.9
        return {**_fake_run_and_score(league, market, family, corner), "slack": slack}

    monkeypatch.setattr(sweep, "_run_and_score", scored)
    sweep.search_cell(*cell, families=("NegBin", "ZINB"), max_trials=30)

    losers = trained.count("ZINB")
    assert losers <= tpe_search.ABANDON_MIN_TRIALS, f"{losers} corners of a ruled-out family"
    assert trained.count("NegBin") > losers, "the leading family must keep being searched"


def test_a_family_that_recovers_inside_the_window_keeps_going(monkeypatch):
    """Abandonment judges a family on its best corner, not its last — one bad corner is not a verdict."""
    cell = ("MLB", "pitcher strikeouts")
    trained = []

    def scored(league, market, family, corner, timeout_s=None):
        trained.append(family)
        # ZINB opens badly, then lands a corner right behind the leader before the window closes.
        slack = 0.4 if family == "NegBin" else (0.35 if trained.count("ZINB") == 2 else -0.9)
        return {**_fake_run_and_score(league, market, family, corner), "slack": slack}

    monkeypatch.setattr(sweep, "_run_and_score", scored)
    sweep.search_cell(*cell, families=("NegBin", "ZINB"), max_trials=30)
    assert trained.count("ZINB") > tpe_search.ABANDON_MIN_TRIALS


def test_enqueued_corners_do_not_consume_early_stop_patience():
    """A deterministic enqueue prefix is coverage, not sampler staleness.

    The frontier enqueue can put a dozen same-default corners ahead of the sampler; if each
    non-improving (or failed) one ticked the patience window, the study could stop before TPE's
    first own proposal. Only sampled trials measure "the search stopped finding improvements".
    """
    cell = CellContext("WNBA", "MIN", "SkewNormal", "continuous", frozenset(), _MATRIX_SHA)
    state = tpe_search.CellSearchState(cell, ("SkewNormal",))
    for _ in range(tpe_search._EARLY_STOP_PATIENCE + 4):
        state.observe("SkewNormal", float("-inf"), sampled=False)
    assert state._stale == 0
    state.observe("SkewNormal", float("-inf"))
    assert state._stale == 1


def test_abandonment_survives_a_resume(monkeypatch):
    """A resumed cell must not re-litigate families its reused rows already ruled out."""
    cell = ("MLB", "pitcher strikeouts")
    context = sweep._cell_context(*cell)
    cached = {}
    for spec_slug, slack in (("NegBin", 0.4), ("ZINB", -0.9)):
        spec = get_strategy(spec_slug)
        for corner in strategy_controls(spec)[: tpe_search.ABANDON_MIN_TRIALS]:
            row = {**_fake_run_and_score(*cell, spec_slug, corner), "slack": slack}
            cached[corner_fingerprint(spec, corner, str(context.matrix_sha256))] = row

    trained = []

    def scored(league, market, family, corner, timeout_s=None):
        trained.append(family)
        return {**_fake_run_and_score(league, market, family, corner), "slack": -0.9}

    monkeypatch.setattr(sweep, "_run_and_score", scored)
    sweep.search_cell(*cell, families=("NegBin", "ZINB"), max_trials=30, cached=cached)
    assert "ZINB" not in trained, "a family the reused rows ruled out was searched again"


def test_board_progress_hands_out_one_bar_row_per_running_cell():
    """Concurrent cells drawing on one row would overwrite each other; a released row is reusable."""
    board = progress.BoardProgress(total_cells=5, jobs=3)
    with board.slot() as first, board.slot() as second, board.slot() as third:
        assert len({first, second, third}) == 3, "two cells were handed the same terminal row"
        assert 0 not in {first, second, third}, "row 0 belongs to the board summary"
    with board.slot() as reused:
        assert reused in {first, second, third}
    assert board.done == 4
    board.close()


def test_the_board_line_never_accounts_for_more_cells_than_the_board_has():
    """``done`` and the postfix are one line, so a repaint must read them from one snapshot.

    Taken a moment apart under a pool they render ``3/5 cells, 3 running · 0 queued`` — six cells on
    a board of five — because another worker's completion landed between the two reads.
    """
    board = progress.BoardProgress(total_cells=6, jobs=3)
    samples: list[tuple[int, str]] = []
    postfix = board.bar.set_postfix_str

    def record(text, **kwargs):
        samples.append((board.bar.n, text))  # the count as the repaint would have rendered it
        return postfix(text, **kwargs)

    board.bar.set_postfix_str = record

    def sweep_one() -> None:
        with board.slot():
            time.sleep(0.01)

    with ThreadPoolExecutor(max_workers=3) as pool:
        _ = [future.result() for future in [pool.submit(sweep_one) for _ in range(6)]]
    board.close()

    assert len(samples) == 12  # one repaint entering each slot, one leaving it
    for done, text in samples:
        running, queued = (int(n) for n in re.findall(r"(\d+) (?:running|queued)", text))
        assert done + running + queued == board.total, f"{done} done · {text}"
    assert samples[-1][0] == 6


def test_a_serial_run_keeps_the_single_bar_at_row_zero():
    """One cell at a time gets no summary row and draws exactly where it always has."""
    board = progress.BoardProgress(total_cells=5, jobs=1)
    assert board.bar.disable
    with board.slot() as only:
        assert only == 0
    board.close()


def test_progress_writes_no_carriage_returns_with_concurrent_bars(monkeypatch):
    """The multi-bar block is the case that would put `\\r` back into the overnight driver's log."""
    monkeypatch.setenv("COLUMNS", "120")
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    board = progress.BoardProgress(total_cells=3, jobs=3)
    with board.slot() as a, board.slot() as b:
        for position, market in ((a, "AST"), (b, "PTS")):
            bar = progress.SearchProgress("WNBA", market, 4, position=position)
            bar.scored(_SN_CORNER, verdict="SHIP", slack=0.2, elapsed_s=31.0, trained=True)
            bar.close()
    board.close()

    written = stream.getvalue()
    assert "\r" not in written
    assert written.count("SHIP") == 2


def test_board_eta_divides_by_workers_but_never_beats_the_slowest_cell():
    """A cell is swept by one worker start to finish, so it floors the board's wall clock."""
    cells = [("NBA", "FGA"), ("WNBA", "AST"), ("NFL", "carries"), ("NHL", "saves")]
    budgets = dict.fromkeys(cells, 10)
    means = {("NBA", "FGA"): 100.0, ("WNBA", "AST"): 1.0, ("NFL", "carries"): 1.0}

    serial = sweep._board_eta(cells, budgets, means, 1.0, 1)
    assert serial == pytest.approx(10 * (100.0 + 1.0 + 1.0 + 1.0))
    # No number of workers finishes sooner than the 1000 s cell one of them is stuck with.
    assert sweep._board_eta(cells, budgets, means, 1.0, 4) == pytest.approx(1000.0)

    # Balanced cells have no such floor, so there the workers simply divide the work.
    even = dict.fromkeys(cells, 20.0)
    assert sweep._board_eta(cells, budgets, even, 1.0, 4) == pytest.approx(200.0)


def test_cell_verdict_names_the_outcome_for_a_parallel_run():
    """Under --jobs the per-cell tables move to the end, so this line is the only live verdict."""
    shipped = _cell_frame("NBA", "FGA")
    assert sweep._cell_verdict(shipped) == "SHIP +0.100"

    killed = shipped.copy()
    killed["ships"], killed["slack"] = [False], [-0.096]
    assert sweep._cell_verdict(killed) == "no ship (best -0.096)"

    empty = shipped.copy()
    empty["slack"] = [float("-inf")]
    assert sweep._cell_verdict(empty) == "no corners"


# --- mandatory seeds and incumbent reconstruction ----------------------------------------------


def test_mandatory_corners_lead_the_enqueue_order_ahead_of_evidence_and_the_incumbent(monkeypatch):
    """A mandatory corner is evaluated first, then the proven evidence corner, then the incumbent.

    The two acceptance cells declare structural recipes whose evidence predates the current
    matrix, so the sweep has to re-measure them on it before a budgeted sampler is trusted to have
    considered them. Assert each source's position, not the exact list — the incumbent is whatever
    the cell last shipped.
    """
    monkeypatch.setattr(
        sweep,
        "load_stat_meta",
        lambda path: {
            "NFL": {
                "receiving yards": {
                    "dist": "DPO",
                    "dist_training_loss": "nll",
                    "posthoc": "roe_mean",
                }
            }
        },
    )
    mandatory = next(
        (slug, controls)
        for league, market, slug, controls in MANDATORY_SWEEP_CORNERS
        if (league, market) == ("NFL", "receiving yards")
    )
    monkeypatch.setattr(
        sweep,
        "CONFIRM_EVIDENCE_CORNERS",
        (("NFL", "receiving yards", "DPO", _dpo_controls("crps", "nll", "none")),),
    )

    enqueued = [
        (spec.slug, corner)
        for spec, corner in sweep._enqueued_corners(sweep._cell_context("NFL", "receiving yards"))
    ]

    assert enqueued[0] == mandatory
    assert enqueued[1] == ("DPO", _dpo_controls("crps", "nll", "none"))
    # The stat_meta cell above reconstructs to the DPO/roe_mean incumbent; it follows the
    # evidence corner directly, ahead of every frontier corner.
    assert enqueued[2] == ("DPO", _dpo_controls("crps", "nll", "roe_mean"))
    frontier = {
        str((spec.slug, corner))
        for spec, corner in sweep._frontier_corners(sweep._cell_context("NFL", "receiving yards"))
    }
    assert {str(pair) for pair in enqueued[3:]} <= frontier
    assert len(enqueued) == len(set(map(str, enqueued)))


def test_mandatory_corner_identity_is_matrix_bound_so_a_rebuild_forces_reevaluation(monkeypatch):
    """A seed declares a recipe, never an evaluation: the matrix hash is what a board row keys on."""
    context = sweep._cell_context("NFL", "receiving yards")
    rebuilt = CellContext(
        context.league,
        context.market,
        context.distribution,
        context.distribution_class,
        context.data_columns,
        "matrix-rebuilt",
        context.target_is_integer,
        context.global_mean,
    )
    spec, corner = sweep._enqueued_corners(context)[0]

    assert spec.slug == RECEIVING
    assert corner_fingerprint(spec, corner, str(context.matrix_sha256)) != corner_fingerprint(
        spec, corner, str(rebuilt.matrix_sha256)
    )
    # Same recipe, same matrix ⇒ the same row an exact resume reuses.
    assert corner_fingerprint(spec, corner, str(context.matrix_sha256)) == corner_fingerprint(
        *sweep._enqueued_corners(rebuilt)[0], str(context.matrix_sha256)
    )


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        # Legacy passing-yards metadata: only the family and its transform were ever persisted.
        (
            {"dist": "SkewNormal", "target_normalization": "ratio_meanyr"},
            {
                "dist": "SkewNormal",
                "normalization": "ratio_meanyr",
                "dist_training_loss": "crps",
                "sn_param": "direct",
                "blending_loss_fn": "nll",
                "posthoc": "none",
            },
        ),
        # An explicitly persisted value always beats the historical default.
        (
            {"dist": "SkewNormal", "target_normalization": "ratio_meanyr", "sn_param": "centered"},
            {
                "dist": "SkewNormal",
                "normalization": "ratio_meanyr",
                "dist_training_loss": "crps",
                "sn_param": "centered",
                "blending_loss_fn": "nll",
                "posthoc": "none",
            },
        ),
        # The count families' defaults reproduce what the CLI resolves for an unset knob.
        (
            {"dist": "ZINB"},
            {
                "dist": "ZINB",
                "dist_training_loss": "nll",
                "zinb_mode": "joint",
                "count_dispersion_objective": "crps",
                "blending_loss_fn": "nll",
                "posthoc": "none",
            },
        ),
        # An explicit value the grid no longer carries yields no incumbent, never a rewrite.
        ({"dist": "SkewNormal", "target_normalization": "retired_slug"}, None),
    ],
)
def test_incumbent_reconstruction_fills_unpersisted_controls_from_historical_defaults(
    monkeypatch, cell, expected
):
    monkeypatch.setattr(sweep, "load_stat_meta", lambda path: {"WNBA": {"AST": cell}})

    incumbent = sweep._incumbent_corner(sweep._cell_context("WNBA", "AST"))

    assert (None if incumbent is None else incumbent[1]) == expected


def test_incumbent_control_defaults_reproduce_the_meditate_cli_fallbacks():
    """The defaults are the CLI's own ``auto`` fallbacks, so a reconstruction is not a guess."""
    assert INCUMBENT_CONTROL_DEFAULTS["SkewNormal"]["blending_loss_fn"] == DEFAULT_BLENDING
    assert INCUMBENT_CONTROL_DEFAULTS["ZINB"]["zinb_mode"] == "joint"
    assert INCUMBENT_CONTROL_DEFAULTS["SkewNormal"]["sn_param"] == "direct"
    assert INCUMBENT_CONTROL_DEFAULTS["DPO"]["count_dispersion_objective"] == "crps"
    for slug, defaults in INCUMBENT_CONTROL_DEFAULTS.items():
        assert defaults["posthoc"] == "none"
        assert get_strategy(slug).is_structural is False
