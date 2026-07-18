"""Training report generation: reads saved model pickles and writes model_stats.{parquet,csv}.

One row per ``(league, market)`` cell. The parquet is the authoritative
artifact (read by the page-7 dashboard, the kelly calibration getter, the
graduation gate, and the ship-config promoter); the CSV mirror at
:data:`MODEL_STATS_CSV_PATH` is rewritten alongside the parquet on every
run so the same numbers are browseable from VSCode without a parquet
viewer extension.

Columns owned by ``meditate`` cover the validation-set metrics produced by
``training.pipeline._compute_metrics`` (model + book baseline + brier skill
score), the per-cell diagnostics (model_weight, EV vs line, shape
calibration), and the trained hyperparameters. Columns owned by
``training.scorecard.compute_gates`` (``ece_*``, ``g1_*`` … ``g5_*``,
``ship``) populate as a side effect of ``training.scorecard`` running over
the test-set CSV for the same cell; before that runs they hold NaN / pd.NA.
"""

import importlib.resources as pkg_resources
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sportstradamus import data
from sportstradamus.helpers import get_logger
from sportstradamus.helpers.io import (
    MODEL_STATS_CSV_PATH,
    MODEL_STATS_PATH,
    _atomic_write_csv,
    _atomic_write_parquet,
    market_file_slug,
)
from sportstradamus.training.config import (
    load_shipped_config,
    load_zi_config,
    save_cv_std_config,
    save_zi_config,
)
from sportstradamus.training.scorecard import (
    compute_gates,
    load_test_set,
    recent_form_fired,
)

logger = get_logger(__name__)

# Ship gates score the FUSED mean — the parlay drafts ``Blended_EV``, not the raw model
# ``EV`` — so Gates 2/3 see the number that's actually bet (the raw-EV compression view
# rides along as the reported ``*_z_raw`` columns). Distinct from the scorecard CLI's
# ``--pred-col`` default, which stays ``EV`` for the model-only compression A/B.
_SHIP_PRED_COL = "Blended_EV"

# Tri-state ship-gate columns. Cast at the parquet boundary to the pandas
# nullable BooleanDtype so a missing scorecard run round-trips as pd.NA
# instead of decaying into False.
_GATE_PASS_COLS = (
    "g1_pass",
    "g1_has_edge",
    "g2_pass",
    "g3_pass",
    "g4_pass",
    "g5_pass",
    "g6_pass",
    "ship",
)

# ``training.scorecard.compute_gates`` reads per-cell test-set CSVs from this
# directory. ``meditate`` dumps them inside :class:`training.pipeline.train_market`
# at the end of every cell-training run.
_TEST_SETS_DIR = pkg_resources.files(data) / "test_sets"

# Minimum denominator for the empirical_shape division in _wide_row; guards
# against divide-by-zero when empirical_shape is near zero on sparse markets.
_SHAPE_DENOM_FLOOR: float = 0.01


def report() -> None:
    """Build per-cell training stats from saved model pickles and persist parquet + CSV."""
    model_list = sorted(
        f.name for f in (pkg_resources.files(data) / "models/").iterdir() if ".mdl" in f.name
    )
    # cv/std/zi live in stat_calibration.json (gitignored, runtime-recomputed);
    # shipped lives in stat_meta.json (committed). stat_meta's dist is an INPUT
    # this function must never write back: a pickle-derived dist sync here once
    # reverted a pending family flip off a stale pickle mid-campaign.
    stat_shipped = load_shipped_config()
    stat_zi_local = load_zi_config()
    _cal_path = pkg_resources.files(data) / "config" / "stat_calibration.json"
    if _cal_path.is_file():
        with open(_cal_path) as f:
            _cal = json.load(f)
    else:
        _cal = {}
    stat_cv = {
        lg: {m: cell.get("cv", 1.0) for m, cell in mkts.items()} for lg, mkts in _cal.items()
    }
    stat_std = {lg: {m: cell.get("std") for m, cell in mkts.items()} for lg, mkts in _cal.items()}

    league_models: dict[str, dict[str, dict]] = {}
    for model_str in model_list:
        with open(pkg_resources.files(data) / f"models/{model_str}", "rb") as infile:
            model = pickle.load(infile)

        name = model_str.split("_")
        cv = model["cv"]
        std = model.get("std", 0)
        league = name[0]
        market = name[1].replace("-", " ").replace(".mdl", "")
        h_gate = model.get("hist_gate", 0)

        league_models.setdefault(league, {})[market] = model
        stat_cv.setdefault(league, {})[market] = float(cv)
        stat_std.setdefault(league, {})[market] = float(std)
        stat_zi_local.setdefault(league, {})[market] = h_gate

    save_cv_std_config(stat_cv, stat_std)
    save_zi_config(stat_zi_local)

    write_model_stats(league_models, stat_cv, stat_std, stat_shipped)


def _wide_row(
    model: dict,
    league: str,
    market: str,
    shipped: str,
    stat_cv: dict,
    stat_std: dict,
) -> dict:
    """Flatten one trained model's pickle into a single training-stats row."""
    metrics_block = model.get("metrics") or {}
    model_m = metrics_block.get("model") or {}
    book_m = metrics_block.get("book_baseline") or {}
    diag = model.get("diagnostics") or {}
    params = model.get("params") or {}
    has_book = bool(book_m)

    def _safe_float(value) -> float:
        # dict.get(k, default) returns the stored None when key is present-but-None,
        # bypassing the default — so float(None) raises TypeError. Collapse to NaN.
        return float(value) if value is not None else float("nan")

    emp_shape = _safe_float(diag.get("empirical_shape"))
    mod_shape = _safe_float(diag.get("model_shape"))
    shape_ratio = (
        mod_shape / max(emp_shape, _SHAPE_DENOM_FLOOR) if not np.isnan(emp_shape) else float("nan")
    )

    def _book(key: str) -> float:
        return _safe_float(book_m.get(key)) if has_book else float("nan")

    def _model_val(key: str) -> float:
        return _safe_float(model_m.get(key))

    def _diag(key: str) -> float:
        return _safe_float(diag.get(key))

    def _param(key: str) -> float:
        return _safe_float(params.get(key))

    return {
        # Identity
        "league": league,
        "market": market,
        "distribution": model.get("distribution"),
        "shipped": shipped,
        # Sample
        "n_validation": float("nan"),  # populated by training.scorecard.compute_gates
        "historical_zero_rate": _safe_float(model.get("hist_gate")),
        # Scoring rules
        "brier_book": _book("brier_score"),
        "brier_model": _model_val("brier_score"),
        "brier_skill_score": _safe_float(metrics_block.get("brier_skill_score")),
        "log_loss_book": _book("log_loss"),
        "log_loss_model": _model_val("log_loss"),
        "nll": _model_val("nll"),
        # Calibration ECE — populated by training.scorecard.compute_gates.
        "ece_equal_mass": float("nan"),
        "ece_null_bias": float("nan"),
        "ece_debiased": float("nan"),
        # Discrimination
        "roc_auc": _model_val("roc_auc"),
        "accuracy": _model_val("accuracy"),
        "precision_over": _model_val("precision_over"),
        "precision_under": _model_val("precision_under"),
        "prediction_std": _model_val("prediction_std"),
        # Over rates
        "predicted_over_rate": _model_val("predicted_over_rate"),
        "empirical_over_rate": _model_val("empirical_over_rate"),
        "over_pct_ev_gt": _diag("over_pct_ev_gt"),
        "over_pct_ev_lt": _diag("over_pct_ev_lt"),
        # EV / line
        "model_ev": _diag("model_ev"),
        "mean_line": _diag("mean_line"),
        "result_mean": _diag("result_mean"),
        "mean_ev_diff": _diag("ev_minus_line"),
        "median_ev_diff": _diag("median_ev_diff"),
        "frac_ev_gt_line": _diag("frac_ev_gt_line"),
        # Kelly & blending
        "kelly_shrinkage": _safe_float(metrics_block.get("kelly_shrinkage")),
        "model_weight": _diag("model_weight"),
        # Shape
        "model_shape": float(mod_shape),
        "empirical_shape": float(emp_shape),
        "shape_ratio": float(shape_ratio),
        "marginal_shape": _diag("marginal_shape"),
        "dispersion_cal": _diag("dispersion_cal"),
        "skew_cal": _diag("skew_cal"),
        # Ship gates — populated by training.scorecard.compute_gates.
        "g1_brier_diff_mean": float("nan"),
        "g1_ci_lo": float("nan"),
        "g1_ci_hi": float("nan"),
        "g1_brier_diff_mean_oracle": float("nan"),
        "g1_ci_lo_oracle": float("nan"),
        "g1_ci_hi_oracle": float("nan"),
        "g1_brier_diff_ci_hi_standalone": float("nan"),
        "g2_star_z": float("nan"),
        "g2_star_z_oracle": float("nan"),
        "g2_star_z_raw": float("nan"),
        "g3_bench_z": float("nan"),
        "g3_bench_z_oracle": float("nan"),
        "g3_bench_z_raw": float("nan"),
        "g4_pit_ks": float("nan"),
        "g4_pit_ks_max": float("nan"),
        "g4_tail_pit_ks": float("nan"),
        "g4_iqr_ratio": float("nan"),
        "g4_iqr_ratio_oracle": float("nan"),
        "central50_coverage": float("nan"),
        "central80_coverage": float("nan"),
        "g5_ece_debiased": float("nan"),
        "g1_pass": pd.NA,
        "g1_has_edge": pd.NA,
        "g2_pass": pd.NA,
        "g3_pass": pd.NA,
        "g4_pass": pd.NA,
        "g5_pass": pd.NA,
        "g6_pass": pd.NA,
        "ship": pd.NA,
        # Hyperparameters
        "hp_rounds": _param("opt_rounds"),
        "hp_leaves": _param("num_leaves"),
        "hp_lr": _param("learning_rate"),
        "hp_min_child": _param("min_child_samples"),
        "hp_l1": _param("lambda_l1"),
        "hp_l2": _param("lambda_l2"),
        # Per-cell calibration
        "cv": float(stat_cv.get(league, {}).get(market, float("nan"))),
        "std": float(stat_std.get(league, {}).get(market, float("nan"))),
    }


def _load_prior_g6_fired() -> dict[tuple[str, str], bool]:
    """Map ``(league, market) -> recent-form leg fired last run``, read once from the existing
    ``model_stats.parquet`` before this run overwrites it. Seeds Gate-6 anchor hysteresis; empty on
    the first ever run (every cell then judges at the fire-on threshold).
    """
    if not MODEL_STATS_PATH.is_file():
        return {}
    prior = pd.read_parquet(MODEL_STATS_PATH)
    return {
        (str(r["league"]), str(r["market"])): recent_form_fired(r) for r in prior.to_dict("records")
    }


def _layer_gates_from_test_set(
    row: dict, league: str, market: str, *, prior_g6_fired: bool | None = None
) -> None:
    """Overwrite the placeholder ship-gate columns on ``row`` if the test-set CSV is present.

    Skips silently when the CSV is missing (the cell trained but its test_set
    wasn't dumped), empty (degenerate frame would crash ``pd.qcut``), or
    malformed (missing required columns). The row is mutated in place.
    """
    test_set_path = Path(str(_TEST_SETS_DIR / f"{market_file_slug(league, market)}.csv"))
    if not test_set_path.is_file():
        return
    try:
        df = load_test_set(test_set_path, _SHIP_PRED_COL)
    except (ValueError, KeyError) as e:
        logger.warning("scorecard skip %s/%s: %s", league, market, e)
        return
    if df.empty:
        return
    row.update(
        compute_gates(
            df,
            league=league,
            market=market,
            pred_col=_SHIP_PRED_COL,
            prior_g6_fired=prior_g6_fired,
        )
    )


def write_model_stats(
    league_models: dict,
    stat_cv: dict,
    stat_std: dict,
    stat_shipped: dict | None = None,
) -> None:
    """Persist one wide row per ``(league, market)`` cell to parquet + CSV mirror.

    Args:
        league_models: ``{league: {market: model_dict}}`` keyed by trained cell.
        stat_cv: ``{league: {market: cv}}`` from stat_calibration.
        stat_std: ``{league: {market: std}}`` from stat_calibration.
        stat_shipped: ``{league: {market: shipped}}`` from stat_meta. Defaults
            to an empty dict so callers (tests) that don't care about the
            shipping column don't have to supply one; cells absent from the
            map fall back to ``"withheld"``.
    """
    shipped_map = stat_shipped or {}
    prior_g6 = _load_prior_g6_fired()
    rows = []
    for league, markets in league_models.items():
        for market, model in markets.items():
            shipped = shipped_map.get(league, {}).get(market, "withheld")
            row = _wide_row(model, league, market, shipped, stat_cv, stat_std)
            _layer_gates_from_test_set(
                row, league, market, prior_g6_fired=prior_g6.get((league, market))
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    for col in _GATE_PASS_COLS:
        if col in df.columns:
            df[col] = df[col].astype("boolean")
    # Deployable (ship) vs actually-staking (kelly_shrinkage > 0). A non-inferiority
    # tie cell ships but is sized to ~0 until it proves live edge, so breadth and
    # betting volume diverge — surface both rather than conflate them.
    if {"ship", "kelly_shrinkage"} <= set(df.columns):
        df["betting_active"] = df["ship"].fillna(False).astype(bool) & (
            df["kelly_shrinkage"].fillna(0.0) > 0
        )
    _atomic_write_parquet(df, MODEL_STATS_PATH)
    _atomic_write_csv(df, MODEL_STATS_CSV_PATH)


def get_market_calibration(league: str, market: str) -> dict[str, float]:
    """Return ``{kelly_shrinkage, brier_skill_score, model_weight}`` for one market.

    Reads ``model_stats.parquet``. Returns NaNs when the row is missing so the
    Kelly resolution chain can fall through to its next source. Kelly imports
    this getter — never reaches into the parquet directly.
    """
    nan_result = {
        "kelly_shrinkage": float("nan"),
        "brier_skill_score": float("nan"),
        "model_weight": float("nan"),
    }
    if not MODEL_STATS_PATH.is_file():
        return nan_result
    df = pd.read_parquet(MODEL_STATS_PATH)
    if df.empty:
        return nan_result
    sub = df[(df["league"] == league) & (df["market"] == market)]
    if sub.empty:
        return nan_result
    row = sub.iloc[0]
    return {
        "kelly_shrinkage": float(row.get("kelly_shrinkage", float("nan"))),
        "brier_skill_score": float(row.get("brier_skill_score", float("nan"))),
        "model_weight": float(row.get("model_weight", float("nan"))),
    }
