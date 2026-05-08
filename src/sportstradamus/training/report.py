"""Training report generation: reads saved model pickles and writes training_report.txt.

Per Phase 3 §4.c/§4.e/§4.f/§4.g, ``model_stats.parquet`` carries the migrated
raw-metric set plus a pinned ``row_kind="book_baseline"`` row so downstream
consumers (Kelly, dashboard) compare against "what taking the book's odds gets
you" rather than against a scaled Brier.
"""

import importlib.resources as pkg_resources
import json
import pickle

import numpy as np
import pandas as pd

from sportstradamus import data
from sportstradamus.helpers import get_logger
from sportstradamus.helpers.io import MODEL_STATS_PATH, _atomic_write_parquet
from sportstradamus.training.config import (
    load_distribution_config,
    load_zi_config,
    save_distribution_config,
    save_zi_config,
)

logger = get_logger(__name__)

# Stats matrix row labels — depend on number of rows present in model["stats"].
_STATS_ROW_LABELS_3 = ("raw", "corrected", "calibrated")
_STATS_ROW_LABELS_2 = ("no_filter", "filter")
# Per-row classification stats source key (model["stats"]) → snake_case parquet column.
# Renames per Phase 3 §4.b — a clean break: old names disappear on next meditate.
_STATS_COL_MAP = {
    "Accuracy": "accuracy",
    "Over Prec": "precision_over",
    "Under Prec": "precision_under",
    "Over%": "predicted_over_rate",
    "Sharpness": "prediction_std",
    "NLL": "nll",
}
# Validation-set raw metrics computed in pipeline.py:_compute_metrics, mirrored
# verbatim into both the model's calibrated row and the book_baseline row.
_RAW_METRIC_KEYS = (
    "brier_score",
    "log_loss",
    "roc_auc",
    "expected_calibration_error",
    "accuracy",
    "precision_over",
    "precision_under",
    "predicted_over_rate",
    "empirical_over_rate",
    "prediction_std",
    "nll",
)


def report() -> None:
    """Generate training report summarizing all model performance metrics."""
    model_list = [
        f.name for f in (pkg_resources.files(data) / "models/").iterdir() if ".mdl" in f.name
    ]
    model_list.sort()
    with open(pkg_resources.files(data) / "stat_cv.json") as f:
        stat_cv = json.load(f)
    with open(pkg_resources.files(data) / "stat_std.json") as f:
        stat_std = json.load(f)
    stat_dist = load_distribution_config()
    stat_zi_local = load_zi_config()

    with open(pkg_resources.files(data) / "training_report.txt", "w") as f:
        league_models = {}
        for model_str in model_list:
            with open(pkg_resources.files(data) / f"models/{model_str}", "rb") as infile:
                model = pickle.load(infile)

            name = model_str.split("_")
            cv = model["cv"]
            std = model.get("std", 0)
            league = name[0]
            market = name[1].replace("-", " ").replace(".mdl", "")
            dist = model["distribution"]
            h_gate = model.get("hist_gate", 0)

            league_models.setdefault(league, {})[market] = model

            stat_cv.setdefault(league, {})
            stat_cv[league][market] = float(cv)

            stat_std.setdefault(league, {})
            stat_std[league][market] = float(std)

            stat_dist.setdefault(league, {})
            stat_dist[league][market] = dist

            stat_zi_local.setdefault(league, {})
            stat_zi_local[league][market] = h_gate

            f.write(f" {league} {market} ".center(62, "="))
            f.write("\n")
            f.write(f" Distribution: {dist} | Historical Zero Rate: {h_gate:.4f}\n")

            metrics_block = model.get("metrics") or {}
            book = metrics_block.get("book_baseline")
            mm = metrics_block.get("model")
            if mm is not None:
                if book is not None:
                    f.write(
                        f" BOOK BASELINE  brier={book['brier_score']:.3f}"
                        f" logloss={book['log_loss']:.3f}"
                        f" auc={book['roc_auc']:.3f}"
                        f" ece={book['expected_calibration_error']:.3f}\n"
                    )
                else:
                    f.write(" BOOK BASELINE  unavailable\n")
                f.write(
                    f" MODEL          brier={mm['brier_score']:.3f}"
                    f" logloss={mm['log_loss']:.3f}"
                    f" auc={mm['roc_auc']:.3f}"
                    f" ece={mm['expected_calibration_error']:.3f}\n"
                )
                bss = metrics_block.get("brier_skill_score", float("nan"))
                ks = metrics_block.get("kelly_shrinkage", float("nan"))
                bss_str = f"{bss:+.3f}" if np.isfinite(bss) else "nan"
                ks_str = f"{ks:.3f}" if np.isfinite(ks) else "nan"
                f.write(f" SKILL          brier_skill_score={bss_str}  kelly_shrinkage={ks_str}\n")

            n_rows = len(next(iter(model["stats"].values())))
            if n_rows == 3:
                idx = ["Raw Model", "Corrected", "Calibrated"]
            else:
                idx = ["No Filter", "Filter"]
            f.write(pd.DataFrame(model["stats"], index=idx).to_string())
            f.write("\n")

            if "diagnostics" in model:
                d = model["diagnostics"]
                sl = d["shape_label"]
                emp_shape = d.get("empirical_shape", 0.0)
                mod_shape = d.get("model_shape", 0.0)
                shape_ratio = (
                    mod_shape / max(emp_shape, 0.01) if not np.isnan(emp_shape) else float("nan")
                )
                f.write(f" DIAG model_weight={d.get('model_weight', float('nan')):.3f}\n")
                f.write(
                    f" DIAG start_{sl}={d.get('start_shape', 0.0):.3f}"
                    f" model_{sl}={mod_shape:.3f}"
                    f" empirical_{sl}={emp_shape:.3f}"
                    f" shape_ratio={shape_ratio:.1f}x\n"
                )
                f.write(
                    f" DIAG start_mean={d.get('start_mean', 0.0):.2f}"
                    f" model_ev={d.get('model_ev', 0.0):.2f}"
                    f" mean_line={d.get('mean_line', 0.0):.2f}"
                    f" result_mean={d.get('result_mean', 0.0):.2f}\n"
                )
                f.write(
                    f" DIAG mean_ev_diff={d.get('ev_minus_line', 0.0):+.3f}"
                    f" median_ev_diff={d.get('median_ev_diff', 0.0):+.3f}"
                    f" frac_ev>line={d.get('frac_ev_gt_line', 0.0):.1%}\n"
                )
                f.write(
                    f" DIAG Over%|ev>line={d.get('over_pct_ev_gt', float('nan')):.3f}"
                    f" Over%|ev<line={d.get('over_pct_ev_lt', float('nan')):.3f}"
                    f" CF_Over%(emp_shape)={d.get('cf_over_pct', float('nan')):.3f}\n"
                )
                f.write(
                    f" DIAG shape_ceiling={d.get('shape_ceiling', 'N/A')}"
                    f" marginal_shape={d.get('marginal_shape', 'N/A')}"
                    f" dispersion_cal={d.get('dispersion_cal', 0.0):.3f}\n"
                )
                f.write(
                    f" DIAG ev_meanyr_corr={d.get('ev_meanyr_corr', float('nan')):.3f}"
                    f" result_meanyr_corr={d.get('result_meanyr_corr', float('nan')):.3f}\n"
                )

            if "params" in model:
                p = model["params"]
                f.write(
                    f" HP rounds={p.get('opt_rounds', '?')}"
                    f" leaves={p.get('num_leaves', '?')}"
                    f" lr={p.get('learning_rate', 0):.4f}"
                    f" min_child={p.get('min_child_samples', '?')}"
                    f" L1={p.get('lambda_l1', 0):.2e}"
                    f" L2={p.get('lambda_l2', 0):.2e}\n"
                )

            f.write("\n")

        # === PER-LEAGUE SUMMARY TABLES ===
        for league, markets in sorted(league_models.items()):
            f.write("\n" + "=" * 80 + "\n")
            f.write(f" {league} SUMMARY TABLE\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"{'Market':<16} {'Dist':<8} {'Over%':>6} {'ShpR':>5}"
                f" {'FracEV>':>7} {'MedDiff':>8}"
                f" {'O%|EV>':>7} {'O%|EV<':>7} {'CF_O%':>6}\n"
            )
            f.write("-" * 80 + "\n")
            for mkt, mdl in sorted(markets.items()):
                if "diagnostics" not in mdl:
                    continue
                d = mdl["diagnostics"]
                stats = mdl["stats"]
                dist_name = mdl.get("distribution", "?")[:6]
                over_pct_val = stats["Over%"][1]
                emp_s = d.get("empirical_shape", 0.0)
                mod_s = d.get("model_shape", 0.0)
                sr = mod_s / max(emp_s, 0.01) if not np.isnan(emp_s) else float("nan")
                fev = d.get("frac_ev_gt_line", 0)
                med = d.get("median_ev_diff", 0)
                oeg = d.get("over_pct_ev_gt", float("nan"))
                oel = d.get("over_pct_ev_lt", float("nan"))
                cfo = d.get("cf_over_pct", float("nan"))
                sr_str = f"{sr:>4.1f}x" if not np.isnan(sr) else "  nan"
                f.write(
                    f"{mkt:<16} {dist_name:<8} {over_pct_val:>6.3f} {sr_str}"
                    f" {fev:>7.1%} {med:>+8.3f}"
                    f" {oeg:>7.3f} {oel:>7.3f} {cfo:>6.3f}\n"
                )
            f.write("\n")

    with open(pkg_resources.files(data) / "stat_cv.json", "w") as f:
        json.dump(stat_cv, f, indent=4)

    with open(pkg_resources.files(data) / "stat_std.json", "w") as f:
        json.dump(stat_std, f, indent=4)

    save_distribution_config(stat_dist)
    save_zi_config(stat_zi_local)

    write_model_stats(league_models, stat_cv, stat_std)


def _diag_row(model: dict, league: str, market: str, stat_cv: dict, stat_std: dict) -> dict:
    """Diagnostic + hyperparameter columns repeated on every parquet row for the market."""
    diag = model.get("diagnostics", {}) or {}
    params = model.get("params", {}) or {}
    emp_shape = diag.get("empirical_shape", float("nan"))
    mod_shape = diag.get("model_shape", float("nan"))
    shape_ratio = mod_shape / max(emp_shape, 0.01) if not np.isnan(emp_shape) else float("nan")
    return {
        "model_weight": diag.get("model_weight", float("nan")),
        "start_shape": diag.get("start_shape", float("nan")),
        "model_shape": mod_shape,
        "empirical_shape": emp_shape,
        "shape_ratio": shape_ratio,
        "model_ev": diag.get("model_ev", float("nan")),
        "mean_line": diag.get("mean_line", float("nan")),
        "result_mean": diag.get("result_mean", float("nan")),
        "mean_ev_diff": diag.get("ev_minus_line", float("nan")),
        "median_ev_diff": diag.get("median_ev_diff", float("nan")),
        "frac_ev_gt_line": diag.get("frac_ev_gt_line", float("nan")),
        "over_pct_ev_gt": diag.get("over_pct_ev_gt", float("nan")),
        "over_pct_ev_lt": diag.get("over_pct_ev_lt", float("nan")),
        "cf_over_pct": diag.get("cf_over_pct", float("nan")),
        "dispersion_cal": diag.get("dispersion_cal", float("nan")),
        "marginal_shape": diag.get("marginal_shape", float("nan")),
        "shape_ceiling": diag.get("shape_ceiling", float("nan")),
        "hp_rounds": params.get("opt_rounds", float("nan")),
        "hp_leaves": params.get("num_leaves", float("nan")),
        "hp_lr": params.get("learning_rate", float("nan")),
        "hp_min_child": params.get("min_child_samples", float("nan")),
        "hp_l1": params.get("lambda_l1", float("nan")),
        "hp_l2": params.get("lambda_l2", float("nan")),
        "cv": float(stat_cv.get(league, {}).get(market, float("nan"))),
        "std": float(stat_std.get(league, {}).get(market, float("nan"))),
        "historical_zero_rate": model.get("hist_gate", float("nan")),
    }


def write_model_stats(league_models: dict, stat_cv: dict, stat_std: dict) -> None:
    """Persist a structured per-market metrics table for the dashboard.

    Per Phase 3 §4.c: each (league, market) emits one ``row_kind="book_baseline"``
    row carrying the metrics produced if the model *were* the bookmaker, plus
    one ``row_kind="model"`` row per ``metric_row`` (raw / corrected / calibrated).
    The new raw metrics from ``model["metrics"]["model"]`` (Brier, log-loss,
    ROC-AUC, ECE, etc.) are attached to the calibrated model row only — they
    describe the calibrated output. ``brier_skill_score`` and ``kelly_shrinkage``
    are mirrored on the calibrated row so the Kelly getter can read one row.
    """
    rows = []
    for league, markets in league_models.items():
        for market, model in markets.items():
            stats_block = model.get("stats", {})
            n_rows = len(next(iter(stats_block.values()))) if stats_block else 0
            labels = (
                _STATS_ROW_LABELS_3
                if n_rows == 3
                else _STATS_ROW_LABELS_2
                if n_rows == 2
                else tuple(f"row_{i}" for i in range(n_rows))
            )
            metrics_block = model.get("metrics") or {}
            model_metrics = metrics_block.get("model") or {}
            book_metrics = metrics_block.get("book_baseline")
            bss = metrics_block.get("brier_skill_score", float("nan"))
            ks = metrics_block.get("kelly_shrinkage", float("nan"))
            diag_cols = _diag_row(model, league, market, stat_cv, stat_std)

            # --- Book baseline row (one per market) ---
            book_row = {
                "league": league,
                "market": market,
                "distribution": model.get("distribution"),
                "row_kind": "book_baseline",
                "metric_row": None,
            }
            book_row.update({k: float("nan") for k in _RAW_METRIC_KEYS})
            if book_metrics is not None:
                for k in _RAW_METRIC_KEYS:
                    if k in book_metrics:
                        book_row[k] = float(book_metrics[k])
            book_row["brier_skill_score"] = 0.0 if book_metrics is not None else float("nan")
            book_row["kelly_shrinkage"] = 0.0 if book_metrics is not None else float("nan")
            book_row.update(diag_cols)
            rows.append(book_row)

            # --- Model rows (one per metric_row) ---
            for i, label in enumerate(labels):
                row = {
                    "league": league,
                    "market": market,
                    "distribution": model.get("distribution"),
                    "row_kind": "model",
                    "metric_row": label,
                }
                # Per-row classification stats from model["stats"] (renamed).
                for src, dst in _STATS_COL_MAP.items():
                    vals = stats_block.get(src)
                    row[dst] = (
                        float(vals[i]) if vals is not None and i < len(vals) else float("nan")
                    )
                # Validation-set raw metrics + skill/shrinkage live on the
                # calibrated row (where the calibrated probs were scored).
                is_calibrated = label == "calibrated"
                for k in _RAW_METRIC_KEYS:
                    if is_calibrated and k in model_metrics:
                        # Don't clobber the per-row stats already filled above
                        # (accuracy/precision_over/etc.); prefer the validation-set
                        # value from compute_metrics for the calibrated row.
                        row[k] = float(model_metrics[k])
                    else:
                        row.setdefault(k, float("nan"))
                row["brier_skill_score"] = float(bss) if is_calibrated else float("nan")
                row["kelly_shrinkage"] = float(ks) if is_calibrated else float("nan")
                row.update(diag_cols)
                rows.append(row)

    df = pd.DataFrame(rows)
    _atomic_write_parquet(df, MODEL_STATS_PATH)


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
    mask = (df["league"] == league) & (df["market"] == market) & (df["row_kind"] == "model")
    if "metric_row" in df.columns:
        cal = mask & (df["metric_row"] == "calibrated")
        sub = df[cal] if cal.any() else df[mask]
    else:
        sub = df[mask]
    if sub.empty:
        return nan_result
    row = sub.iloc[0]
    return {
        "kelly_shrinkage": float(row.get("kelly_shrinkage", float("nan"))),
        "brier_skill_score": float(row.get("brier_skill_score", float("nan"))),
        "model_weight": float(row.get("model_weight", float("nan"))),
    }
