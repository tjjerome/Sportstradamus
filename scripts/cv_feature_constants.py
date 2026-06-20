"""Cross-validate the two §6.3 feature constants. Dev-only; hard-excluded from devel.

Both constants shipped as documented placeholders (see base.py:60 / base.py:72); the §6.3
discipline is to cross-validate before fixing a value ("do not invent a value"). This driver
runs the tiered, held-out-only CV from
docs/superpowers/plans/2026-06-19-cv-eb-prior-comp-recency.md:

  ``eb-sweep``       Tier-1 signal sweep of ``_EXPANDING_EB_PRIOR_K``. ``MeanYr_expanding_eb``
                     recomputes closed-form at any K from the cached matrix columns
                     ``MeanYr_expanding_shifted`` / ``MeanYr_expanding_n`` (with ``pos_baseline``
                     recovered algebraically from the stored K=10 column) -- no matrix rebuild.

  ``recency-sweep``  Tier-1 signal sweep of ``_COMP_RECENCY_HALFLIFE_DAYS``. The recency weights
                     collapse into the matrix and are discarded, so each halflife needs a fresh
                     ``get_training_matrix`` (comp leagues only). Resumable: each (cell, H) rebuild
                     is cached under the work dir.

  ``confirm``        Tier-2 model confirm on the carried values -- deferred until the Tier-1
                     rankings are reviewed (see the plan's checkpoint).

Signal metric: 1-feature out-of-fold R² / NLL under purged GroupKFold-by-player (the dominant
leakage for these same-player-only, online-safe features), with EB also reported on the
low-``n_career`` stratum where the shrinkage K actually bites. Nothing here is promoted to
production; outputs land under ``/tmp/cv_feature_constants``.
"""
import importlib.resources as pkg_resources
import json
import shutil
import subprocess
import time
from datetime import date
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from sportstradamus import data as data_pkg
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA, base
from sportstradamus.training import baselines
from sportstradamus.training.scorecard import (
    DEFAULT_PRED_COL,
    apply_thresholds,
    gate_row,
    load_test_set,
    min_gate_slack,
)
from sportstradamus.training.ship_config import STAT_META_PATH

_STATS = {"NBA": StatsNBA, "NFL": StatsNFL, "WNBA": StatsWNBA}
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_DATA = Path(str(pkg_resources.files(data_pkg) / "training_data"))
_TEST_SETS = Path(str(pkg_resources.files(data_pkg) / "test_sets"))
_WORK = Path("/tmp/cv_feature_constants")
# Fallback deterministic normalization for cells whose configured slug is not a real
# target-normalization (mirrors regen_ab_batch / meditate --bypass-withholding).
_DEFAULT_TN = "ratio_meanyr"

_EB_GRID = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0)  # extended below 3: optimum at floor
_H_GRID = (14.0, 30.0, 45.0, 90.0, 180.0)
_EB_K0 = 10.0  # value the cached MeanYr_expanding_eb column was built at
_H0 = 45.0     # production halflife, restored after each sweep
_N_FOLDS = 5
_EB_LOWN = 10.0  # n_career <= K-midpoint: the stratum where shrinkage moves the value
_EB_COL = "MeanYr_expanding_eb"
_RECENCY_COL = "Player comps z recent"

# EB reads cached matrices, so the sweep is cheap -- cover the volume + 6 shipped cells.
_EB_MARKETS = {
    "NBA": ["PTS", "REB", "AST", "MIN", "FGA", "PR", "RA", "DREB"],
    "WNBA": ["PTS", "REB", "AST", "MIN", "FGA", "PR", "PRA", "RA"],
    "NFL": ["carries", "completions", "attempts", "receiving yards", "rushing yards"],
}
# Recency needs a full rebuild per H, so pool a few comp-heavy markets per league.
_RECENCY_MARKETS = {
    "NBA": ["PTS", "REB", "AST"],
    "WNBA": ["PTS", "REB", "AST"],
    "NFL": ["rushing yards", "receiving yards", "receptions"],
}
# Per-league rebuild cutoff, mirroring the §6.3 regen: NBA needs 2025-01-01 to skip a
# 2-yr gamelog cold-start gap whose per-gameday comp pool divides by zero; NFL/WNBA use
# get_training_matrix's 850-day default (cutoff=None). --cutoff overrides all.
_RECENCY_CUTOFF = {"NBA": date(2025, 1, 1), "NFL": None, "WNBA": None}
# Tier-2 confirm cells: volume + the 6 §6.3 just-shipped cells (deduped).
_CONFIRM_MARKETS = {
    "NBA": ["PTS", "REB", "AST", "MIN", "FGA", "PR", "RA"],
    "WNBA": ["PTS", "REB", "AST", "MIN", "FGA", "RA"],
    "NFL": ["carries", "completions", "attempts", "receiving yards", "rushing yards"],
}


def _eb_at_k(e, n, eb10, k):
    """Recompute ``MeanYr_expanding_eb`` at prior count ``k`` from the cached columns.

    ``pos_baseline`` is recovered per-row from the stored K=10 column: for ``n>0`` rows
    ``exp_filled == e`` so ``pos = (eb10 - s0·e)/(1-s0)``; for ``n==0`` the career mean is
    absent and ``eb10`` *is* the baseline. The recovery is exact algebra (K=10 reproduces
    the stored column), and stays accurate where it matters because any cancellation noise
    for high-``n`` rows carries weight ``1-shrink(k) -> 0``.
    """
    n = np.asarray(n, float)
    eb10 = np.asarray(eb10, float)
    has = n > 0
    e_safe = np.where(has, np.asarray(e, float), 0.0)
    s0 = n / (n + _EB_K0)
    sk = np.where(has, n / (n + k), 0.0)  # n==0,k==0 would be 0/0; no career data -> baseline
    pos = np.where(has, (eb10 - s0 * e_safe) / (1.0 - s0), eb10)
    exp_filled = np.where(has, e_safe, pos)
    return sk * exp_filled + (1.0 - sk) * pos


def _oof_signal(feature, target, groups, n_folds=_N_FOLDS):
    """Pooled out-of-fold signal of one feature vs the target under GroupKFold-by-player."""
    f = np.asarray(feature, float)
    y = np.asarray(target, float)
    g = np.asarray(groups)
    ok = np.isfinite(f) & np.isfinite(y)
    f, y, g = f[ok], y[ok], g[ok]
    n_groups = len(np.unique(g))
    if n_groups < 2 or np.std(f) == 0:
        return {"oof_r2": np.nan, "oof_nll": np.nan, "spearman": np.nan, "n": int(ok.sum())}
    oof = np.empty_like(y)
    sigmas = []
    for tr, te in GroupKFold(n_splits=min(n_folds, n_groups)).split(f, y, g):
        slope, intercept = np.polyfit(f[tr], y[tr], 1)
        oof[te] = intercept + slope * f[te]
        sigmas.append((y[tr] - (intercept + slope * f[tr])).std())
    ss_res = np.sum((y - oof) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    sigma = float(np.mean(sigmas))
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y - oof) / sigma) ** 2))
    return {
        "oof_r2": 1.0 - ss_res / ss_tot,
        "oof_nll": nll,
        "spearman": float(spearmanr(f, y).statistic),
        "n": int(ok.sum()),
    }


def _rank(df, val, name):
    click.echo(f"\n=== {name}: pooled OOF R² per league (mean over markets), ranked ===")
    for league, grp in df.groupby("league"):
        agg = grp.groupby(val)["oof_r2"].mean().sort_values(ascending=False)
        line = "  ".join(f"{val}={k:g}:{r:.4f}" for k, r in agg.items())
        click.echo(f"[{league}] best {val}={agg.index[0]:g}   {line}")


def _eb_sweep(leagues, markets_override):
    rows = []
    for league in leagues:
        for market in markets_override or _EB_MARKETS[league]:
            slug = market_file_slug(league, market)
            path = _TRAINING_DATA / f"{slug}.parquet"
            if not path.exists():
                click.echo(f"  {slug}: skip (no cached matrix)")
                continue
            cols = ["Player", _EB_COL, "MeanYr_expanding_shifted", "MeanYr_expanding_n", "Result"]
            matrix = pd.read_parquet(path, columns=cols)
            e = matrix["MeanYr_expanding_shifted"].to_numpy(float)
            n = matrix["MeanYr_expanding_n"].to_numpy(float)
            eb10 = matrix[_EB_COL].to_numpy(float)
            result = matrix["Result"].to_numpy(float)
            player = matrix["Player"].to_numpy()
            faithful = float(np.nanmax(np.abs(_eb_at_k(e, n, eb10, _EB_K0) - eb10)))
            lown = n <= _EB_LOWN
            for k in _EB_GRID:
                eb_k = _eb_at_k(e, n, eb10, k)
                overall = _oof_signal(eb_k, result, player)
                low = _oof_signal(eb_k[lown], result[lown], player[lown])
                rows.append({
                    "league": league, "market": market, "K": k,
                    "oof_r2": overall["oof_r2"], "oof_nll": overall["oof_nll"],
                    "spearman": overall["spearman"], "oof_r2_lown": low["oof_r2"],
                    "lown_frac": float(lown.mean()), "n": overall["n"],
                    "faithful_max_abs": faithful,
                })
            click.echo(f"  {slug}: faithful={faithful:.2e}  lown_frac={lown.mean():.2f}  n={len(matrix)}")
    return _finish(rows, "K", "eb")


def _install_recency_capture(grid):
    """Capture the served ``Player comps z recent`` at every grid halflife in one pass.

    Wraps ``_nonmlb_comp_features`` to re-run only the comp step (cheap relative to the
    full matrix build) per halflife on a copy of the gameday frame, recording the actual
    served column rather than a raw aggregate -- so any downstream reindex/fill is exactly
    reproduced. The expensive matrix build runs once. Returns (records, restore); records
    accumulate (date, player, H, z_recent). No production base.py change is needed -- the
    wrapper swaps the module constant around each re-run and restores it.
    """
    records = []
    orig_method = base.Stats._nonmlb_comp_features

    def method_wrap(self, stats, defstats, market, opponents, date):
        day = str(pd.Timestamp(date).date())
        for h in grid:
            base._COMP_RECENCY_HALFLIFE_DAYS = h
            probe = stats.copy()
            orig_method(self, probe, defstats, market, opponents, date)
            if "Player comps z recent" in probe:
                records.extend((day, player, h, float(val))
                               for player, val in probe["Player comps z recent"].items())
        base._COMP_RECENCY_HALFLIFE_DAYS = _H0
        return orig_method(self, stats, defstats, market, opponents, date)

    base.Stats._nonmlb_comp_features = method_wrap

    def restore():
        base.Stats._nonmlb_comp_features = orig_method
        base._COMP_RECENCY_HALFLIFE_DAYS = _H0

    return records, restore


def _recency_frame(records, matrix):
    """Align captured per-halflife z-recent onto the matrix's (Date, Player, Result).

    ``_nonmlb_comp_features`` fires twice per gameday -- once in the volume-model
    pass (``_dispatch_volume_stats`` -> ``get_stats``) and once in the served
    ``get_stats`` -- and the two writes differ (different offer context). The matrix
    keeps the served (last) write, so the capture is deduped keep="last" per
    (Date, Player, H); averaging the pair (pivot_table's default aggfunc) corrupts
    the feature by up to the served-vs-volume spread (~3.4 z on WNBA PTS).
    """
    long = (pd.DataFrame(records, columns=["Date", "Player", "H", "z"])
            .drop_duplicates(["Date", "Player", "H"], keep="last"))
    wide = long.pivot_table(index=["Date", "Player"], columns="H", values="z")
    wide.columns = [f"z_{int(h)}" for h in wide.columns]
    frame = matrix[["Player", "Date", "Result", _RECENCY_COL]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    frame = frame.rename(columns={_RECENCY_COL: "prod_z"})
    frame = frame.merge(wide.reset_index(), on=["Date", "Player"], how="left")
    zcols = [f"z_{int(h)}" for h in _H_GRID]
    frame[zcols] = frame[zcols].fillna(0.0)  # no comps -> 0, matching production
    return frame


def _recency_sweep(leagues, markets_override, cutoff_override):
    cache = _WORK / "recency_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for league in leagues:
        cutoff = cutoff_override if cutoff_override is not None else _RECENCY_CUTOFF[league]
        markets = markets_override or _RECENCY_MARKETS[league]
        stats = None
        for market in markets:
            slug = market_file_slug(league, market)
            out = cache / f"{slug}.parquet"
            if out.exists():
                continue
            if stats is None:
                stats = _STATS[league]()
                stats.load()
            records, restore = _install_recency_capture(_H_GRID)
            try:
                started = time.time()
                matrix = stats.get_training_matrix(market, cutoff)
            finally:
                restore()
            frame = _recency_frame(records, matrix)
            frame.to_parquet(out, index=False)
            faithful = float(np.nanmax(np.abs(frame[f"z_{int(_H0)}"] - frame["prod_z"])))
            click.echo(f"  {slug}: rows={len(frame)} faithful(H45)={faithful:.2e} "
                       f"{time.time() - started:.0f}s")

    rows = []
    for league in leagues:
        for market in markets_override or _RECENCY_MARKETS[league]:
            slug = market_file_slug(league, market)
            path = cache / f"{slug}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            for h in _H_GRID:
                signal = _oof_signal(frame[f"z_{int(h)}"], frame["Result"], frame["Player"])
                rows.append({"league": league, "market": market, "H": h,
                             "recency_mean": float(frame[f"z_{int(h)}"].mean()), **signal})
    return _finish(rows, "H", "recency")


def _finish(rows, val, name):
    df = pd.DataFrame(rows)
    _WORK.mkdir(parents=True, exist_ok=True)
    out = _WORK / f"{name}_signal.csv"
    df.to_csv(out, index=False)
    _rank(df, val, name)
    click.echo(f"\n[{name}] signal table -> {out}")
    return df


def _tn_group(cell):
    if cell.get("dist") == "SkewNormal":
        tn = cell.get("target_normalization")
        if tn in baselines.TARGET_NORMALIZATION_SLUGS:
            return tn
    return _DEFAULT_TN


def _eb_overwrite(path, k):
    matrix = pd.read_parquet(path)
    matrix[_EB_COL] = _eb_at_k(
        matrix["MeanYr_expanding_shifted"].to_numpy(float),
        matrix["MeanYr_expanding_n"].to_numpy(float),
        matrix[_EB_COL].to_numpy(float),
        k,
    )
    matrix.to_parquet(path, compression="zstd", index=True)


def _meditate(league, markets, tn):
    click.echo(f"  meditate {league} tn={tn} ({len(markets)} cells)")
    subprocess.run(
        ["poetry", "run", "meditate", "--deterministic", "--bypass-withholding",
         "--league", league, "--market", ",".join(markets), "--target-normalization", tn],
        check=True, cwd=str(_REPO_ROOT),
    )


def _run_arm(cells, arm, stash):
    arm_dir = stash / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for league, market, slug, tn in cells:
        groups.setdefault((league, tn), []).append((market, slug))
    for (league, tn), items in groups.items():
        _meditate(league, [m for m, _ in items], tn)
        for market, slug in items:
            shutil.copy(_TEST_SETS / "deterministic" / tn / f"{slug}.csv", arm_dir / f"{slug}.csv")


def _gate_metrics(csv_path, league, market, tn):
    df = load_test_set(csv_path, DEFAULT_PRED_COL)
    row = apply_thresholds(gate_row(
        df, DEFAULT_PRED_COL, league=league, market=market, strategy="cv", decode_strategy=tn))
    return {"ship": bool(row["ship"]), "slack": min_gate_slack(row),
            "g1_brier": row.get("g1_brier_diff_mean"), "n": row.get("n_validation")}


def _confirm_eb(leagues, ks):
    """Tier-2 confirm: overwrite MeanYr_expanding_eb at each K, deterministic A/B vs K=10."""
    with open(STAT_META_PATH) as fh:
        meta = json.load(fh)
    stash = _WORK / "eb_confirm"
    backup = stash / "prod_parquets"
    backup.mkdir(parents=True, exist_ok=True)
    cells = []
    for league in leagues:
        for market in _CONFIRM_MARKETS[league]:
            slug = market_file_slug(league, market)
            if (_TRAINING_DATA / f"{slug}.parquet").exists():
                cells.append((league, market, slug, _tn_group(meta[league].get(market, {}))))
    for _, _, slug, _ in cells:
        shutil.copy(_TRAINING_DATA / f"{slug}.parquet", backup / f"{slug}.parquet")
    try:
        _run_arm(cells, "baseline", stash)
        for k in ks:
            for _, _, slug, _ in cells:
                _eb_overwrite(_TRAINING_DATA / f"{slug}.parquet", k)
            _run_arm(cells, f"k{int(k)}", stash)
            for _, _, slug, _ in cells:
                shutil.copy(backup / f"{slug}.parquet", _TRAINING_DATA / f"{slug}.parquet")
    finally:
        for _, _, slug, _ in cells:
            shutil.copy(backup / f"{slug}.parquet", _TRAINING_DATA / f"{slug}.parquet")

    rows = []
    for league, market, slug, tn in cells:
        base_m = _gate_metrics(stash / "baseline" / f"{slug}.csv", league, market, tn)
        for k in ks:
            cand = _gate_metrics(stash / f"k{int(k)}" / f"{slug}.csv", league, market, tn)
            rows.append({"league": league, "market": market, "K": k,
                         "ship_base": base_m["ship"], "ship_k": cand["ship"],
                         "slack_base": base_m["slack"], "slack_k": cand["slack"],
                         "slack_delta": cand["slack"] - base_m["slack"],
                         "brier_base": base_m["g1_brier"], "brier_k": cand["g1_brier"]})
    out = pd.DataFrame(rows)
    _WORK.mkdir(parents=True, exist_ok=True)
    out.to_csv(_WORK / "eb_confirm.csv", index=False)
    click.echo("\n=== EB confirm: candidate K vs K=10 baseline (min_gate_slack delta, +=better) ===")
    for k in ks:
        sub = out[out["K"] == k]
        flips = sorted((sub.loc[sub["ship_k"] & ~sub["ship_base"], "league"] + "_"
                        + sub.loc[sub["ship_k"] & ~sub["ship_base"], "market"]).tolist())
        click.echo(f"\n-- K={k:g}: mean slack_delta={sub['slack_delta'].mean():+.4f}  "
                   f"ship {int(sub['ship_base'].sum())}->{int(sub['ship_k'].sum())}/{len(sub)}  "
                   f"new ships: {flips}")
        click.echo(sub[["league", "market", "slack_base", "slack_k", "slack_delta",
                        "ship_base", "ship_k"]].to_string(index=False))
    click.echo(f"\n[eb-confirm] table -> {_WORK / 'eb_confirm.csv'}")
    return out


@click.command()
@click.option("--phase", type=click.Choice(["eb-sweep", "recency-sweep", "confirm"]), required=True)
@click.option("--leagues", default="NBA,WNBA,NFL", help="comma-separated leagues to sweep.")
@click.option("--markets", default=None, help="override market subset (single league at a time).")
@click.option("--cutoff", default=None, help="ISO date lower bound for recency-sweep rebuilds "
              "(skips cold-start gamedays whose comp pool divides by zero).")
@click.option("--eb-k", default="1,2", help="confirm phase: comma-separated K values vs the K=10 baseline.")
def main(phase, leagues, markets, cutoff, eb_k):
    league_list = [lg.strip() for lg in leagues.split(",") if lg.strip()]
    market_list = [m.strip() for m in markets.split(",")] if markets else None
    cutoff_date = date.fromisoformat(cutoff) if cutoff else None
    if phase == "eb-sweep":
        _eb_sweep(league_list, market_list)
    elif phase == "recency-sweep":
        _recency_sweep(league_list, market_list, cutoff_date)
    else:
        _confirm_eb(league_list, [float(x) for x in eb_k.split(",")])


if __name__ == "__main__":
    main()
