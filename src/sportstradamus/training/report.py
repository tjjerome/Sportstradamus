"""Training report generation: reads saved model pickles and writes training_report.txt."""

import importlib.resources as pkg_resources
import json
import pickle

import numpy as np
import pandas as pd

from sportstradamus import data
from sportstradamus.helpers.io import MODEL_STATS_PATH, _atomic_write_parquet
from sportstradamus.training.config import (
    load_distribution_config,
    load_zi_config,
    save_distribution_config,
    save_zi_config,
)

# Stats matrix row labels — depend on number of rows present in model["stats"].
_STATS_ROW_LABELS_3 = ("raw", "corrected", "calibrated")
_STATS_ROW_LABELS_2 = ("no_filter", "filter")
# Column rename: source key in model["stats"] -> snake_case for parquet.
_STATS_COL_MAP = {
    "Accuracy": "accuracy",
    "Over Prec": "over_prec",
    "Under Prec": "under_prec",
    "Over%": "over_pct",
    "Sharpness": "sharpness",
    "NLL": "nll",
}


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
            f.write(f" Distribution Model: {dist}\n")
            f.write(f" Historical Zero Rate: {h_gate:.4f}\n")
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
                f.write(
                    f" DIAG model_weight={d.get('model_weight', float('nan')):.3f}"
                    f" DIAG model_calib={d.get('model_calib', float('nan')):.3f}\n"
                )
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


def write_model_stats(league_models: dict, stat_cv: dict, stat_std: dict) -> None:
    """Persist a structured per-market metrics table for the dashboard.

    One row per (league, market, metric_row). Diagnostics and hyperparameters
    are repeated per metric_row so each row is self-contained for filtering.
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
            diag = model.get("diagnostics", {}) or {}
            params = model.get("params", {}) or {}
            emp_shape = diag.get("empirical_shape", float("nan"))
            mod_shape = diag.get("model_shape", float("nan"))
            shape_ratio = (
                mod_shape / max(emp_shape, 0.01) if not np.isnan(emp_shape) else float("nan")
            )

            for i, label in enumerate(labels):
                row = {
                    "league": league,
                    "market": market,
                    "distribution": model.get("distribution"),
                    "metric_row": label,
                }
                for src, dst in _STATS_COL_MAP.items():
                    vals = stats_block.get(src)
                    row[dst] = float(vals[i]) if vals is not None and i < len(vals) else float("nan")
                row.update(
                    {
                        "model_weight": diag.get("model_weight", float("nan")),
                        "model_calib": diag.get("model_calib", float("nan")),
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
                )
                rows.append(row)

    df = pd.DataFrame(rows)
    _atomic_write_parquet(df, MODEL_STATS_PATH)
