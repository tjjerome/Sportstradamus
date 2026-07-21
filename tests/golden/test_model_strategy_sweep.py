"""Unit tests for the Operation Ship 75 strategy sweep (``training.model_strategy_sweep``).

Two layers are covered. The per-corner *primitive* ``_score_corner`` is the honest scorer: it loads
one trained deterministic dump and runs the production :func:`scorecard.gate_row` on its served
(validation-fit-calibrated) predictive — no test re-fit; the heavy dump load + scorecard gate are
monkeypatched so the test pins only the plumbing. The *orchestration* (family-grid enumeration,
board assembly, verdict formatting) is exercised with the heavy per-corner ``meditate`` train+score
monkeypatched out, so no model trains: the Optuna GridSampler study visits every corner of the
cell's family grid once, scores each by the honest gate, and ranks the board by ship slack.
"""

import json
import subprocess

import click
import pandas as pd
import pytest

from sportstradamus.training import model_strategy_sweep as sweep
from sportstradamus.training.baselines import get_target_normalization
from sportstradamus.training.model_strategy_artifacts import (
    MODEL_STRATEGY_MODEL_KEY,
    SPLIT_FINGERPRINT_CSV_COLUMN,
    InactiveStrategyArtifactError,
    artifact_identity_columns,
    build_artifact_identity,
    validate_strategy_artifacts,
)
from sportstradamus.training.model_strategy_frame import validate_strategy_frame
from sportstradamus.training.model_strategy_registry import (
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
)
from sportstradamus.training.model_strategy_report_identity import resolve_report_identity
from sportstradamus.training.role_specs import role_spec_for
from sportstradamus.training.structural_strategies import (
    AFFINE_STRATEGY as RUSHING,
)
from sportstradamus.training.structural_strategies import (
    TWO_PART_STRATEGY as RECEIVING,
)

_MATRIX_SHA = "matrix-123"
_MATRIX_COLUMNS = frozenset(
    column
    for spec in registered_strategies()
    for column in spec.applicability.required_data_columns
)
# The role×position two-part strategy is gated by the per-(league, market) role registry;
# the NFL rushing-affine strategy is cell-pinned. A cell offers a structural corner only when
# its real matrix carries that strategy's grouping columns, so most cells sweep base families
# alone. These fixtures give each registered cell the role columns its real matrix carries.
_STRUCTURAL_CELL_COLUMNS = {
    ("NFL", "receiving yards"): frozenset(role_spec_for("NFL", "receiving yards").all_columns),
    ("NFL", "rushing yards"): frozenset(role_spec_for("NFL", "rushing yards").all_columns),
    ("NBA", "PTS"): frozenset(role_spec_for("NBA", "PTS").all_columns),
}


@pytest.fixture(autouse=True)
def _fixed_matrix_contract(monkeypatch):
    monkeypatch.setattr(
        sweep,
        "_training_matrix_contract",
        lambda league, market: (
            _STRUCTURAL_CELL_COLUMNS.get((league, market), frozenset()),
            _MATRIX_SHA,
        ),
    )


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
    if family == "SkewNormal":
        slack = (
            {
                "ratio_meanyr": 0.0,
                "centered_additive_mean10": 0.20,
                "centered_additive_eb_meanyr_k10": -0.10,
            }[corner["normalization"]]
            + (0.05 if corner["dist_training_loss"] == "nll" else 0.0)
            + (0.01 if corner["sn_param"] == "centered" else 0.0)
            + (0.02 if corner["blending_loss_fn"] == "crps" else 0.0)
        )
    elif family == "ZINB":
        slack = (
            {"joint": 0.0, "hurdle": 0.15}[corner["zinb_mode"]]
            + (0.05 if corner["count_dispersion_objective"] == "pit_ks" else 0.0)
            + (0.02 if corner["blending_loss_fn"] == "crps" else 0.0)
        )
    else:  # NegBin — kept below ZINB's best so cross-family ranking has an unambiguous winner
        slack = (0.05 if corner["count_dispersion_objective"] == "pit_ks" else 0.0) + (
            0.02 if corner["blending_loss_fn"] == "crps" else 0.0
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
    return [
        {
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
        }
    ]


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
            for control in ("sn_param", "zinb_mode")
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
    # SN: 3 norms × 2 dist-loss × 2 sn-param × 2 blend.
    assert math.prod(len(v) for v in sn.axes.values()) == 24
    # Mixture: 1 dist × 3 norms — no loss axis (component loss pinned nll), no blend axis
    # (off-ship-path fallback trains with the default blend).
    assert math.prod(len(v) for v in mix.axes.values()) == 3
    assert math.prod(len(v) for v in zinb.axes.values()) == 8  # 1 dist × 2 mode × 2 disp × 2 blend
    assert math.prod(len(v) for v in negbin.axes.values()) == 4  # 1 dist × 2 disp × 2 blend
    assert math.prod(len(v) for v in dpo.axes.values()) == 4  # 1 dist × 2 disp × 2 blend
    assert sn.persist == {
        "dist": "dist",
        "normalization": "target_normalization",
        "dist_training_loss": "dist_training_loss",
        "sn_param": "sn_param",
        "blending_loss_fn": "blending",
    }
    assert mix.axes["dist"] == ("Mixture",)
    assert mix.axes["normalization"] == sn.axes["normalization"]
    assert mix.persist == {
        "dist": "dist",
        "normalization": "target_normalization",
    }
    # Count families persist their single-choice dist so a winner's dist writes to stat_meta.
    assert zinb.persist == {
        "dist": "dist",
        "zinb_mode": "zinb_mode",
        "count_dispersion_objective": "count_dispersion_objective",
        "blending_loss_fn": "blending",
    }
    assert negbin.persist == {
        "dist": "dist",
        "count_dispersion_objective": "count_dispersion_objective",
        "blending_loss_fn": "blending",
    }
    assert dpo.persist == negbin.persist  # same gate-free count-family persist surface
    assert zinb.axes["dist"] == ("ZINB",) and negbin.axes["dist"] == ("NegBin",)
    assert dpo.axes["dist"] == ("DPO",)
    for method, slug in (
        (receiving, RECEIVING),
        (rushing, RUSHING),
    ):
        assert method.slug == slug and method.axes == {}
        assert method.fixed_controls == {
            "dist": "SkewNormal",
            "normalization": "ratio_meanyr",
            "dist_training_loss": "crps",
            "sn_param": "direct",
            "blending_loss_fn": "nll",
            "hpo_selection": "loss",
            "stabilization": "None",
            "posthoc": "none",
        }
        assert method.persist == {
            "dist": "dist",
            "normalization": "target_normalization",
            "dist_training_loss": "dist_training_loss",
            "sn_param": "sn_param",
            "blending_loss_fn": "blending",
            "hpo_selection": "hpo_selection",
        }
        assert method.fixed_persist == {"posthoc": "none"}
        assert len(method.canonical_signature) == 64
    # Every swept axis persists — a shipping corner is always reproducible from stat_meta alone
    # (S4: dist_training_loss gained a stat_meta field, deleting the ranks-only corner concept).
    for spec in (sn, mix, zinb, negbin, dpo):
        assert set(spec.persist) == set(spec.axes) | set(spec.fixed_controls)


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
    }
    mixture_controls = {"dist": "Mixture", "normalization": "ratio_meanyr"}
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
    # dist leads the swept-axis columns and rides in the board schema.
    assert sweep._AXIS_COLUMNS[0] == "dist"
    assert "dist" in sweep._BOARD_COLUMNS


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


def test_cell_families_routes_by_distribution_class(monkeypatch):
    """A count cell sweeps every count family (even one pinned NegBin); a plain continuous cell
    sweeps SkewNormal + Mixture, while an NFL yardage cell also sweeps the structural strategies
    its matrix columns qualify for; loud on an unknown dist.
    """
    fake = {
        "MLB": {
            "pitcher strikeouts": {"dist": "ZINB"},
            "pitches thrown": {"dist": "NegBin"},  # already flipped to NegBin — still sweeps all
            "hits allowed": {"dist": "Gamma"},  # unswept family
        },
        "WNBA": {"AST": {"dist": "SkewNormal"}},
        "NFL": {
            "receiving yards": {"dist": "SkewNormal"},
            "rushing yards": {"dist": "SkewNormal"},
            "passing yards": {"dist": "SkewNormal"},
        },
    }
    monkeypatch.setattr(sweep, "load_stat_meta", lambda path: fake)
    assert sweep._cell_families("MLB", "pitcher strikeouts") == ("ZINB", "NegBin", "DPO")
    assert sweep._cell_families("MLB", "pitches thrown") == ("ZINB", "NegBin", "DPO")
    assert sweep._cell_families("WNBA", "AST") == ("SkewNormal", "Mixture")
    assert sweep._cell_families("NFL", "receiving yards") == (
        "SkewNormal",
        "Mixture",
        RECEIVING,
        RUSHING,
    )
    assert sweep._cell_families("NFL", "rushing yards") == (
        "SkewNormal",
        "Mixture",
        RECEIVING,
        RUSHING,
    )
    assert sweep._cell_families("NFL", "passing yards") == ("SkewNormal", "Mixture")
    with pytest.raises(click.UsageError):
        sweep._cell_families("MLB", "hits allowed")


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


def test_cached_matrix_contract_reads_schema_and_changes_sha_when_parquet_changes(
    monkeypatch, tmp_path
):
    monkeypatch.undo()  # exercise the real helper, not this module's fixed-contract fixture
    monkeypatch.setattr(sweep, "_TRAINING_DATA_ROOT", tmp_path)
    sweep._read_training_matrix_contract.cache_clear()
    path = sweep._training_matrix_path("NFL", "receiving yards")
    pd.DataFrame({"Player position": ["WR"], "Result": [10.0], "route_feature": [1.0]}).to_parquet(
        path
    )
    columns_before, sha_before = sweep._training_matrix_contract("NFL", "receiving yards")

    pd.DataFrame(
        {
            "Player position": ["WR", "RB"],
            "Result": [10.0, 11.0],
            "route_feature": [1.0, 2.0],
        }
    ).to_parquet(path)
    columns_after, sha_after = sweep._training_matrix_contract("NFL", "receiving yards")

    assert columns_before == columns_after == {"Player position", "Result", "route_feature"}
    assert len(sha_before) == len(sha_after) == 64
    assert sha_before != sha_after


def test_corner_count_sums_families_per_cell(monkeypatch):
    """A count cell's corner count is ZINB + NegBin + DPO (8+4+4); a SN cell's is its single grid (24)."""
    families = {
        ("MLB", "pitcher strikeouts"): ("ZINB", "NegBin", "DPO"),
        ("WNBA", "AST"): ("SkewNormal",),
    }
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: families[(lg, mkt)])
    assert sweep._cell_corner_count("MLB", "pitcher strikeouts") == 16  # 8 + 4 + 4
    assert sweep._cell_corner_count("WNBA", "AST") == 24
    assert sweep._corner_count([("MLB", "pitcher strikeouts"), ("WNBA", "AST")]) == 40


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
    }
    rows = sweep._run_and_score("WNBA", "AST", "SkewNormal", corner)
    assert captured["corner"] == corner
    assert rows[0]["family"] == "SkewNormal"
    assert rows[0]["slack"] == 0.1


def test_run_and_score_records_failed_corner_non_shipping(monkeypatch):
    """A corner whose meditate errors is caught and returned as one non-shipping row — never scored."""

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
    }
    rows = sweep._run_and_score("NHL", "blocked", "SkewNormal", corner)
    assert len(rows) == 1
    assert rows[0]["ships"] is False
    assert rows[0]["slack"] == sweep._FAILED_CORNER_SLACK
    assert rows[0]["family"] == "SkewNormal"
    assert rows[0]["normalization"] == "ratio_meanyr"  # corner axes preserved for the board


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
    row = sweep._run_and_score("NFL", "rushing yards", spec.slug, dict(spec.fixed_controls))[0]
    assert row["ships"] is False and row["slack"] == sweep._FAILED_CORNER_SLACK

    monkeypatch.setattr(
        sweep,
        "_score_corner",
        lambda *args: (_ for _ in ()).throw(ValueError("wrong matrix")),
    )
    with pytest.raises(ValueError, match="wrong matrix"):
        sweep._run_and_score("NFL", "rushing yards", spec.slug, dict(spec.fixed_controls))


# --- per-cell study over the family grid -----------------------------------------------------


def test_search_cell_enumerates_sn_grid_and_ranks(monkeypatch):
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    board = sweep.search_cell("WNBA", "AST")

    assert len(board) == 24  # 3 norms × 2 dist-loss × 2 sn-param × 2 blend
    # Best corner: centered-mean10 + nll + centered + crps blend (0.20 + 0.05 + 0.01 + 0.02).
    assert board.iloc[0]["normalization"] == "centered_additive_mean10"
    assert board.iloc[0]["dist_training_loss"] == "nll"
    assert board.iloc[0]["sn_param"] == "centered"
    assert board.iloc[0]["blending_loss_fn"] == "crps"
    assert board["slack"].is_monotonic_decreasing
    assert (board["family"] == "SkewNormal").all()
    assert (board["league"] == "WNBA").all() and (board["market"] == "AST").all()
    # The base family is part of the reproducible corner and persists explicitly.
    assert (board["dist"] == "SkewNormal").all()
    # g6 is surfaced on the board.
    assert "g6_pass" in board.columns


def test_search_cell_runs_registered_structural_method_with_full_fixed_recipe(monkeypatch):
    slug = RECEIVING
    spec = get_strategy(slug)
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: (slug,))
    seen = []

    def score(league, market, family, corner):
        seen.append((league, market, family, corner))
        legacy = {"validation_audit": {"split_fingerprint_sha256": "split-123"}}
        identity = build_artifact_identity(
            spec.slug, league, market, corner, legacy, matrix_hash=_MATRIX_SHA
        )
        return [
            {
                **_fake_row(spec.family, corner, 0.2),
                "strategy_slug": spec.slug,
                "structural_strategy": identity.structural_strategy,
                "strategy_signature": identity.signature,
                "strategy_implementation_version": identity.implementation_version,
                "artifact_schema_version": identity.artifact_schema_version,
                "strategy_status": identity.status,
                "controls_json": controls_json(corner),
                "corner_fingerprint": identity.corner_fingerprint,
                "matrix_hash": _MATRIX_SHA,
                "split_fingerprint": identity.split_fingerprint,
            }
        ]

    monkeypatch.setattr(sweep, "_run_and_score", score)
    board = sweep.search_cell("NFL", "receiving yards")

    assert len(board) == 1
    assert seen == [
        (
            "NFL",
            "receiving yards",
            slug,
            dict(spec.fixed_controls),
        )
    ]
    assert board.iloc[0]["structural_strategy"] == slug
    assert board.iloc[0]["strategy_signature"] == spec.canonical_signature
    assert board.iloc[0]["hpo_selection"] == "loss"


def test_search_cell_count_cell_sweeps_both_families_and_unions(monkeypatch):
    """A count cell studies ZINB *and* plain NegBin; the boards union, slack-ranked across families.

    The closure must bind ``family`` per study — a bare loop-variable capture would score every study
    as the last family, collapsing both boards onto one dist. This asserts each family's full corner
    set lands with its own ``dist`` (8 ZINB + 4 NegBin) and the union is one slack-sorted board.
    """
    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("ZINB", "NegBin"))
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    board = sweep.search_cell("MLB", "pitcher strikeouts")

    assert len(board) == 12  # 8 ZINB (2×2×2) + 4 NegBin (2×2)
    assert board["family"].value_counts().to_dict() == {"ZINB": 8, "NegBin": 4}
    # Per-family closure binding: each family's rows carry its own single-choice dist (no bleed).
    assert (board.loc[board["family"] == "ZINB", "dist"] == "ZINB").all()
    assert (board.loc[board["family"] == "NegBin", "dist"] == "NegBin").all()
    # Union is slack-sorted; ZINB's hurdle+pit_ks+crps (0.15+0.05+0.02) tops NegBin's best (0.05+0.02).
    assert board["slack"].is_monotonic_decreasing
    assert board.iloc[0]["family"] == "ZINB" and board.iloc[0]["zinb_mode"] == "hurdle"
    assert board.iloc[0]["count_dispersion_objective"] == "pit_ks"
    assert board.iloc[0]["blending_loss_fn"] == "crps"
    # NegBin corners never carry a zinb_mode key → its column stays blank for them (schema superset).
    assert board.loc[board["family"] == "NegBin", "zinb_mode"].isna().all()
    # SN-only axes are blank on a count board (the schema is a shared superset).
    assert board["normalization"].isna().all()
    assert board["sn_param"].isna().all()


def test_search_cell_survives_a_failing_corner(monkeypatch):
    """One corner erroring does not abort the study: every corner lands, the bad one non-shipping.

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

    assert len(board) == 24  # all corners recorded; no crash aborted the grid
    failed = board[board["slack"] == sweep._FAILED_CORNER_SLACK]
    assert len(failed) == 12 and not failed["ships"].astype(bool).any()  # the 12 crps corners
    assert int(board["ships"].astype(bool).sum()) == 12  # the 12 nll corners scored + shipped


def test_run_deterministic_meditate_builds_corner_flags(monkeypatch, tmp_path):
    """Each corner axis is forwarded as its flag; a ZINB corner omits --target-normalization and the
    output is captured to a per-corner log under the research log root.
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
    }
    sn_spec = get_strategy("SkewNormal")
    sweep._run_deterministic_meditate("WNBA", "AST", sn, sn_spec)
    last = calls[-1]
    assert last[last.index("--target-normalization") + 1] == "ratio_meanyr"
    assert last[last.index("--dist-training-loss") + 1] == "nll"
    assert last[last.index("--sn-param") + 1] == "centered"
    assert last[last.index("--blending-loss-fn") + 1] == "crps"
    assert "--structural-strategy" not in last
    # The posthoc override is a structural-recipe control only — base families never emit it,
    # so a swept SkewNormal corner honours each cell's stat_meta posthoc.
    assert "--posthoc" not in last
    assert sweep._log_path("WNBA", "AST", sn, sn_spec).exists()

    sweep._run_deterministic_meditate("NFL", "receiving yards", sn, sn_spec)
    receiving_base = calls[-1]
    assert receiving_base[receiving_base.index("--structural-strategy") + 1] == "none"

    zinb = {
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending_loss_fn": "nll",
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
    assert structural[structural.index("--posthoc") + 1] == "none"
    assert structural[structural.index("--structural-strategy") + 1] == RUSHING


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

    sweepable, missing = sweep._select_board_cells()
    assert set(sweepable) == {("WNBA", "AST"), ("MLB", "pitcher strikeouts")}
    assert missing == [("WNBA", "STL")]  # withheld + registry, but no cached matrix
    # --include-shipped pulls in the devel cell (it has data); STL still missing.
    incl, _ = sweep._select_board_cells(include_shipped=True)
    assert ("WNBA", "PTS") in incl
    # League filter narrows to that league's sweepable cells.
    sweepable_wnba, _ = sweep._select_board_cells("WNBA")
    assert sweepable_wnba == [("WNBA", "AST")]


def test_select_board_cells_dist_class_filter(monkeypatch):
    """--dist-class narrows the cohort: `count` → ZINB/NegBin cells, `continuous` → SkewNormal, `all`
    → both.
    """
    fake = {
        "WNBA": {"AST": {"dist": "SkewNormal", "shipped": "withheld"}},  # continuous
        "MLB": {
            "pitcher strikeouts": {"dist": "ZINB", "shipped": "withheld"},  # count (ZINB)
            "pitches thrown": {"dist": "NegBin", "shipped": "withheld"},  # count (NegBin)
        },
    }
    fake_markets = {"WNBA": ["AST"], "MLB": ["pitcher strikeouts", "pitches thrown"]}
    monkeypatch.setattr(sweep, "load_stat_meta", lambda path: fake)
    monkeypatch.setattr(sweep, "ALL_MARKETS", fake_markets)
    monkeypatch.setattr(sweep, "_has_training_data", lambda lg, mkt: True)

    count, _ = sweep._select_board_cells(dist_class="count")
    assert set(count) == {("MLB", "pitcher strikeouts"), ("MLB", "pitches thrown")}
    continuous, _ = sweep._select_board_cells(dist_class="continuous")
    assert continuous == [("WNBA", "AST")]
    every, _ = sweep._select_board_cells(dist_class="all")
    assert set(every) == {("WNBA", "AST"), ("MLB", "pitcher strikeouts"), ("MLB", "pitches thrown")}


def test_run_board_mode_warns_missing_data_and_passes_filters(monkeypatch, capsys):
    """The board runner warns per cell skipped for a missing matrix and forwards its scope filters."""
    captured = {}

    def fake_select(league, include_shipped, dist_class):
        captured["incl"] = include_shipped
        captured["dist_class"] = dist_class
        return [("WNBA", "AST")], [("WNBA", "STL")]

    monkeypatch.setattr(sweep, "_select_board_cells", fake_select)
    monkeypatch.setattr(sweep, "_corner_count", lambda cells: 12)
    monkeypatch.setattr(
        sweep,
        "run_board",
        lambda cells, out, resume: pd.DataFrame(
            {"league": ["WNBA"], "market": ["AST"], "ships": [False]}
        ),
    )
    monkeypatch.setattr(sweep, "_print_board_rollup", lambda b: None)

    sweep._run_board_mode("WNBA", True, "count", "/tmp/board.csv", False, False)
    assert captured["incl"] is True and captured["dist_class"] == "count"
    out = capsys.readouterr().out
    assert "skip WNBA STL: no cached training matrix" in out
    assert "1 skipped (no cached matrix)" in out


def _cell_frame(league, market):
    """A one-row board frame for a cell — the shape ``search_cell`` returns for the run_board tests."""
    return pd.DataFrame({"league": [league], "market": [market], "slack": [0.1], "ships": [True]})


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


def test_run_board_resume_skips_cells_already_on_board(monkeypatch, tmp_path):
    """``resume`` skips a cell already on the CSV (keeping its rows) and only sweeps the new one."""
    out = str(tmp_path / "board.csv")
    pd.DataFrame({"league": ["WNBA"], "market": ["AST"], "slack": [0.9], "ships": [True]}).to_csv(
        out, index=False
    )
    swept = []

    def spy(league, market):
        swept.append((league, market))
        return _cell_frame(league, market)

    monkeypatch.setattr(sweep, "search_cell", spy)
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda b: None)
    monkeypatch.setattr(sweep, "_board_done_cells", lambda path: {("WNBA", "AST")})

    board = sweep.run_board([("WNBA", "AST"), ("NBA", "FGA")], out=out, resume=True)
    assert swept == [("NBA", "FGA")]  # the on-board cell was skipped
    assert set(zip(board["league"], board["market"], strict=True)) == {
        ("WNBA", "AST"),
        ("NBA", "FGA"),
    }


def test_run_board_resume_returns_only_requested_scope_but_preserves_unrelated_csv(
    monkeypatch, tmp_path
):
    """A scoped resume cannot leak an unrelated complete prior winner into ``--confirm`` input."""
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
    monkeypatch.setattr(
        sweep,
        "_board_done_cells",
        lambda path: {("WNBA", "AST"), ("NFL", "sacks")},
    )

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


def test_resume_resweeps_legacy_nfl_yards_rows_missing_structural_family(monkeypatch, tmp_path):
    out = tmp_path / "board.csv"
    pd.DataFrame(
        {
            "league": ["NFL"],
            "market": ["receiving yards"],
            "family": ["SkewNormal"],
            "normalization": ["ratio_meanyr"],
            "dist_training_loss": ["crps"],
            "sn_param": ["direct"],
            "blending_loss_fn": ["nll"],
        }
    ).to_csv(out, index=False)
    monkeypatch.setattr(
        sweep,
        "_cell_families",
        lambda lg, mkt: (RECEIVING,),
    )

    assert sweep._board_done_cells(str(out)) == set()
    normalized = sweep._read_board(out)
    assert list(normalized.columns) == sweep._BOARD_COLUMNS
    assert normalized["structural_strategy"].isna().all()

    _, contract = next(iter(sweep._expected_corner_records("NFL", "receiving yards").items()))
    spec = get_strategy(contract["strategy_slug"])
    controls = json.loads(contract["controls_json"])
    split = "split-123"
    fingerprint = corner_fingerprint(spec, controls, _MATRIX_SHA)
    signed = pd.DataFrame(
        [
            {
                "league": "NFL",
                "market": "receiving yards",
                "corner_fingerprint": fingerprint,
                "split_fingerprint": split,
                **contract,
            }
        ]
    )
    signed.to_csv(out, index=False)
    assert sweep._board_done_cells(str(out)) == {("NFL", "receiving yards")}

    contradictory = signed.copy()
    contradictory.loc[0, "hpo_selection"] = "rank"
    contradictory.to_csv(out, index=False)
    assert sweep._board_done_cells(str(out)) == set()

    stale_split = signed.copy()
    stale_split.loc[0, "split_fingerprint"] = "WRONG"
    stale_split.to_csv(out, index=False)
    # Resume can derive spec+controls+matrix before training, not the adapter's split. Confirm
    # separately requires this board value to equal the full-HPO model+CSV split identity.
    assert sweep._board_done_cells(str(out)) == {("NFL", "receiving yards")}

    stale_matrix = signed.copy()
    stale_matrix.loc[0, "matrix_hash"] = "old-matrix"
    stale_matrix.to_csv(out, index=False)
    assert sweep._board_done_cells(str(out)) == set()


def test_resume_accepts_complete_count_board_with_canonical_none_namespaces(monkeypatch, tmp_path):
    cell = ("MLB", "pitcher strikeouts")
    context = CellContext(*cell, "ZINB", "count", _MATRIX_COLUMNS, _MATRIX_SHA)
    monkeypatch.setattr(sweep, "_cell_context", lambda league, market: context)
    monkeypatch.setattr(sweep, "_cell_families", lambda league, market: ("ZINB", "NegBin", "DPO"))
    expected = sweep._expected_corner_records(*cell)
    rows = []
    for contract in expected.values():
        spec = get_strategy(contract["strategy_slug"])
        controls = json.loads(contract["controls_json"])
        rows.append(
            {
                "league": cell[0],
                "market": cell[1],
                **contract,
                "corner_fingerprint": corner_fingerprint(spec, controls, _MATRIX_SHA),
                "split_fingerprint": None,
            }
        )
    out = tmp_path / "count-board.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    assert sweep._board_done_cells(str(out)) == {cell}
    assert {
        spec.slug: {sweep._dump_subdir(controls, spec) for controls in strategy_controls(spec)}
        for spec in map(get_strategy, ("ZINB", "NegBin", "DPO"))
    } == {
        "ZINB": {"none", "none_hurdle"},
        "NegBin": {"none"},
        "DPO": {"none"},
    }


def test_cli_board_dry_run_trains_nothing(monkeypatch, tmp_path):
    """``--board --dry-run`` prints the scope and exits without sweeping a single cell; --dist-class
    threads through to the cell selection.
    """
    from click.testing import CliRunner

    seen = {}

    def fake_select(lg, incl, dist_class):
        seen["dist_class"] = dist_class
        return [("WNBA", "AST")], []

    monkeypatch.setattr(sweep, "_select_board_cells", fake_select)
    monkeypatch.setattr(sweep, "_corner_count", lambda cells: 12)

    def boom(*a, **k):
        raise AssertionError("run_board must not be called on a dry run")

    monkeypatch.setattr(sweep, "run_board", boom)
    out = str(tmp_path / "board.csv")
    result = CliRunner().invoke(
        sweep.main, ["--board", "--dry-run", "--dist-class", "count", "--out", out]
    )
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
        "sn_param=centered, blending=crps, structural_strategy=none"
    )
    receiving_base = {**sn, "league": "NFL", "market": "receiving yards"}
    assert sweep._stat_meta_edit(receiving_base).endswith("structural_strategy=none")
    # Count families persist dist first (pins the winning family, e.g. a ZINB→NegBin flip).
    zinb_controls = {
        "dist": "ZINB",
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending_loss_fn": "nll",
    }
    zinb = {
        "league": "MLB",
        "market": "pitcher strikeouts",
        "family": "ZINB",
        "strategy_slug": "ZINB",
        "structural_strategy": BASE_STRUCTURAL_STRATEGY,
        "controls_json": controls_json(zinb_controls),
    }
    assert (
        sweep._stat_meta_edit(zinb)
        == "dist=ZINB, zinb_mode=hurdle, count_dispersion_objective=pit_ks, blending=nll, "
        "structural_strategy=none"
    )
    negbin_controls = {
        "dist": "NegBin",
        "count_dispersion_objective": "crps",
        "blending_loss_fn": "crps",
    }
    negbin = {
        "league": "MLB",
        "market": "pitcher strikeouts",
        "family": "NegBin",
        "strategy_slug": "NegBin",
        "structural_strategy": BASE_STRUCTURAL_STRATEGY,
        "controls_json": controls_json(negbin_controls),
    }
    assert (
        sweep._stat_meta_edit(negbin)
        == "dist=NegBin, count_dispersion_objective=crps, blending=crps, "
        "structural_strategy=none"
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
    assert sweep._stat_meta_edit(structural) == (
        "dist=SkewNormal, target_normalization=ratio_meanyr, dist_training_loss=crps, "
        "sn_param=direct, blending=nll, hpo_selection=loss, posthoc=none, "
        f"structural_strategy={RUSHING}"
    )


# --- CLI -------------------------------------------------------------------------------------


def test_cli_runs_a_single_cell(monkeypatch, tmp_path):
    from click.testing import CliRunner

    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    out = str(tmp_path / "board.csv")
    result = CliRunner().invoke(sweep.main, ["--league", "WNBA", "--market", "AST", "--out", out])
    assert result.exit_code == 0, result.output
    assert "centered_additive_mean10" in result.output


def test_cli_confirm_invokes_run_confirm(monkeypatch, tmp_path):
    """``--confirm`` hands the ranked board to the confirm loop (which is mocked here)."""
    from click.testing import CliRunner

    from sportstradamus.training import model_strategy_confirm

    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    seen = {}
    monkeypatch.setattr(
        model_strategy_confirm,
        "run_confirm",
        lambda board, *, yes: seen.update(n=len(board), yes=yes),
    )
    out = str(tmp_path / "board.csv")
    result = CliRunner().invoke(
        sweep.main, ["--league", "WNBA", "--market", "AST", "--out", out, "--confirm", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert seen == {"n": 24, "yes": True}


def test_cli_scoped_resume_confirm_excludes_unrelated_complete_prior_cell(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from sportstradamus.training import model_strategy_confirm

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
    complete = {("NFL", "receiving yards"), ("WNBA", "AST")}
    monkeypatch.setattr(sweep, "_select_board_cells", lambda *args: (requested, []))
    monkeypatch.setattr(sweep, "_board_done_cells", lambda path: complete)
    monkeypatch.setattr(sweep, "_corner_count", len)
    monkeypatch.setattr(sweep, "search_cell", _cell_frame)
    monkeypatch.setattr(sweep, "_print_cell_summary", lambda board: None)
    monkeypatch.setattr(sweep, "_print_board_rollup", lambda board: None)
    confirmed = {}
    monkeypatch.setattr(
        model_strategy_confirm,
        "run_confirm",
        lambda board, *, yes: confirmed.update(
            cells=set(zip(board["league"], board["market"], strict=True)), yes=yes
        ),
    )

    result = CliRunner().invoke(
        sweep.main,
        ["--board", "--league", "NFL", "--resume", "--confirm", "--yes", "--out", out],
    )

    assert result.exit_code == 0, result.output
    assert confirmed == {"cells": set(requested), "yes": True}
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

    monkeypatch.setattr(sweep, "_cell_families", lambda lg, mkt: ("SkewNormal",))
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    out = str(tmp_path / "board.csv")
    runner = CliRunner()

    def run(league, market):
        return runner.invoke(sweep.main, ["--league", league, "--market", market, "--out", out])

    assert run("WNBA", "AST").exit_code == 0
    assert run("NBA", "FGA").exit_code == 0
    board = pd.read_csv(out)
    assert set(zip(board["league"], board["market"], strict=True)) == {
        ("WNBA", "AST"),
        ("NBA", "FGA"),
    }
    n_two_cells = len(board)
    assert run("WNBA", "AST").exit_code == 0  # re-run replaces the cell's rows, not appends
    assert len(pd.read_csv(out)) == n_two_cells
