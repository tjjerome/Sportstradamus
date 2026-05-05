"""Sweep time-weighted correlation parameters against the beat-line target.

Follow-up to ``scripts/test_correlation_predictive_power.py``. The headline
finding from that script: on real NBA data, the NEW residualize+shrinkage
methodology only shows meaningful direction-matched lift in the extreme
(top ~5%) bucket, ~+0.10. The hypothesis here is that older training games
are noisy — roster turnover, scheme changes, injuries — and that recent-
weighted estimates should sharpen the tail.

We sweep two levers, jointly:

  - ``horizon_days``: drop training rows older than H days before the
    train/test cutoff. Tests "is older data actively harmful?"
  - ``half_life_days``: weight each game by ``0.5^(Δt / half_life)`` where
    Δt is days before the cutoff. Tests "is older data useful, just less
    so?" (``inf`` → uniform weights.)

The two compose: hard horizon picks the support; half-life shapes weights
inside it. Methodology is held fixed at NEW (residualize + pairwise-NaN +
shrinkage + 0.05 floor) — only the time weighting changes. Cell scoring
is the beat-line conditional probability with quantile bucketing from
``test_correlation_predictive_power.py``.

The (gameId, team) pivot construction is the expensive step (~5 min × 3
on NBA), so we cache the residualized + beat-line matrices to parquet
under ``data/test_sets/`` keyed by gamelog fingerprint. Subsequent runs
load the cache and re-correlate in seconds.

Usage:

    poetry run python scripts/test_correlation_time_weighting.py --league NBA
    poetry run python scripts/test_correlation_time_weighting.py --league NBA --no-cache
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import sys
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# importlib.import_module avoids the package-attribute rebinding trap (the
# training package __init__ does ``from .correlate import correlate``, which
# overwrites the submodule attribute with the function).
_correlate = importlib.import_module("sportstradamus.training.correlate")
CORR_MAGNITUDE_FLOOR = _correlate.CORR_MAGNITUDE_FLOOR
MIN_OVERLAP_FOR_FULL_WEIGHT = _correlate.MIN_OVERLAP_FOR_FULL_WEIGHT

# Reuse the empirical beat-line target builder from the comparison script.
# Scoring + same-player detection are reimplemented in vectorized form
# below — at NBA scale the per-pair Python loop in the comparison script
# is minutes per cell.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
_pred = importlib.import_module("test_correlation_predictive_power")
_empirical_beat_line_corr = _pred._empirical_beat_line_corr

# Fraction of (gameId, team) rows used for the train slice. Same value as
# the comparison script so results are directly comparable.
TRAIN_FRACTION: float = 0.70

# Default sweep grids. ``inf`` is encoded as math.inf in code, "inf" on the
# CLI. Picked to span "no filter" through "very recent only" without being
# so dense that the table is unreadable.
DEFAULT_HORIZONS_DAYS: tuple[float, ...] = (365, 180, 90, 60, 30)
DEFAULT_HALF_LIVES_DAYS: tuple[float, ...] = (float("inf"), 365, 180, 90, 60, 30)

CACHE_DIR = REPO_ROOT / "data" / "test_sets" / "correlation_time_weighting_cache"


# ---------------------------------------------------------------------------
# Pivot caching
# ---------------------------------------------------------------------------


def _gamelog_fingerprint(stats) -> str:
    """Short hash of (n_rows, min_date, max_date) — invalidates cache when
    the gamelog grows or shifts. Cheap to compute (no row scan).
    """
    log = stats.gamelog
    date_col = stats.log_strings["date"]
    dates = pd.to_datetime(log[date_col])
    payload = f"{len(log)}|{dates.min().isoformat()}|{dates.max().isoformat()}"
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _build_team_game_records_beat_line(league: str, stats, earliest_date) -> list[dict]:
    """Same beat-line pivot trick as the comparison script: monkey-patch
    _residualize_gamelog so each cell is 1/0/NaN beat-line indicator.
    """
    cmod = _correlate
    orig = cmod._residualize_gamelog

    def beat_line_residualize(gamelog, player_col, date_col, stat_cols):
        residualized = orig(gamelog, player_col, date_col, stat_cols)
        for s in stat_cols:
            if s not in residualized.columns:
                continue
            vals = residualized[s]
            hits = (vals > 0).astype("float")
            hits[vals.isna()] = np.nan
            residualized[s] = hits
        return residualized

    cmod._residualize_gamelog = beat_line_residualize
    try:
        return cmod._build_team_game_records(league, stats, earliest_date)
    finally:
        cmod._residualize_gamelog = orig


def _load_or_build_pivots(
    league: str, use_cache: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (residualized_matrix, beat_line_matrix), using parquet cache.

    The pivots only depend on the gamelog, so a fingerprint of (n_rows,
    min_date, max_date) is sufficient to detect staleness. When the
    fingerprint matches and ``--no-cache`` is not set, the parquet read
    is seconds; otherwise we rebuild via _build_team_game_records (which
    runs _residualize_gamelog internally and takes ~5 min × 2 on NBA).
    """
    from sportstradamus import stats as stats_pkg

    cls = getattr(stats_pkg, f"Stats{league}")
    stats = cls()
    stats.load()
    if not hasattr(stats, "gamelog") or stats.gamelog is None or len(stats.gamelog) == 0:
        raise RuntimeError(f"Stats{league} loaded but gamelog is empty")

    fp = _gamelog_fingerprint(stats)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    res_path = CACHE_DIR / f"{league}_residualized_{fp}.parquet"
    beat_path = CACHE_DIR / f"{league}_beatline_{fp}.parquet"

    if use_cache and res_path.exists() and beat_path.exists():
        click.echo(f"      cache hit: {res_path.name}, {beat_path.name}")
        res_matrix = pd.read_parquet(res_path)
        beat_matrix = pd.read_parquet(beat_path)
        return res_matrix, beat_matrix

    click.echo("      cache miss — rebuilding pivots (this takes ~10 min on NBA)")
    earliest = pd.to_datetime(
        stats.gamelog[stats.log_strings["date"]]
    ).min().to_pydatetime()
    res_records = _correlate._build_team_game_records(league, stats, earliest)
    res_matrix = pd.DataFrame(pd.json_normalize(res_records))
    beat_records = _build_team_game_records_beat_line(league, stats, earliest)
    beat_matrix = pd.DataFrame(pd.json_normalize(beat_records))

    # parquet doesn't like Python date objects; coerce to Timestamp.
    res_matrix["DATE"] = pd.to_datetime(res_matrix["DATE"])
    beat_matrix["DATE"] = pd.to_datetime(beat_matrix["DATE"])

    res_matrix.to_parquet(res_path)
    beat_matrix.to_parquet(beat_path)
    click.echo(f"      cached -> {res_path.name}, {beat_path.name}")
    return res_matrix, beat_matrix


# ---------------------------------------------------------------------------
# Weighted correlation (NEW methodology with per-row weights)
# ---------------------------------------------------------------------------


def _weighted_pairwise_corr(
    X: np.ndarray, M: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized weighted Pearson per column-pair, pairwise-complete on NaN.

    Args:
        X: (n_rows, n_cols) values with NaN positions replaced by 0.
        M: (n_rows, n_cols) 0/1 present-mask (1 = value, 0 = was-NaN).
        weights: (n_rows,) per-row weights.

    Returns:
        ``(corr, eff_overlap)`` — square (n_cols, n_cols) arrays. ESS =
        (Σw)²/Σw², feeds into shrinkage in place of a raw shared-row count
        (matches ``MIN_OVERLAP_FOR_FULL_WEIGHT=30`` semantics under uniform
        weights). All pairwise sums are computed via three matrix products,
        so cost is O(n_cols² · n_rows) but vectorized — ~100× faster than
        the per-pair Python loop on real-data shapes.
    """
    w = weights.astype(np.float64)
    Mw = M * w[:, None]

    # Pairwise sums of weights / squared weights restricted to both-present.
    # W[i,j] = Σ_k w_k · M[k,i] · M[k,j]; W2 uses w² instead of w.
    W = Mw.T @ M
    W2 = (M * (w * w)[:, None]).T @ M

    # X has 0 where missing, so X·M factors out automatically.
    Xw = X * w[:, None]
    Sx = Xw.T @ M
    Sy = Sx.T
    Sxx = ((X * X) * w[:, None]).T @ M
    Syy = Sxx.T
    Sxy = Xw.T @ X

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_W = np.where(W > 0, W, np.nan)
        mx = Sx / safe_W
        my = Sy / safe_W
        cov = Sxy / safe_W - mx * my
        vx = Sxx / safe_W - mx * mx
        vy = Syy / safe_W - my * my
        denom = np.sqrt(np.maximum(vx * vy, 0.0))
        corr = np.where(denom > 0, cov / denom, 0.0)
        eff = np.where(W2 > 0, (W * W) / W2, 0.0)

    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr, eff


def _build_team_caches(matrix: pd.DataFrame) -> dict[str, dict]:
    """Per-team precomputation that does not depend on weights.

    Computing column ranks (weighted Spearman = weighted Pearson on ranks)
    is the only non-trivial per-team work that does not change as we sweep
    half-lives, so we do it once per horizon. ``team_mask`` is a boolean
    selector against the horizon-filtered matrix's row order so the caller
    can pull the matching weight slice without re-deriving it.
    """
    matrix = matrix.drop(columns="DATE", errors="ignore")
    out: dict[str, dict] = {}
    for team in matrix["TEAM"].unique():
        team_mask = (matrix["TEAM"] == team).values
        team_data = matrix.loc[team_mask].drop(columns="TEAM")
        team_data = team_data.loc[:, team_data.notna().any(axis=0)]
        if team_data.shape[1] < 2 or len(team_data) < 2:
            continue
        team_data = team_data.reindex(sorted(team_data.columns), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ranks = team_data.rank(method="average").values  # NaN preserved
        present = ~np.isnan(ranks)
        M = present.astype(np.float64)
        X = np.where(present, ranks, 0.0)
        out[team] = {
            "cols": list(team_data.columns),
            "X": X,
            "M": M,
            "team_mask": team_mask,
        }
    return out


def _build_corr_new_weighted_arrays(
    team_caches: dict[str, dict], weights_full: np.ndarray
) -> dict[str, tuple[list[str], np.ndarray]]:
    """NEW methodology with per-row weights, returning per-team numpy arrays.

    Mirrors ``_team_corr_pairwise(...,shrink=True, magnitude_floor=0.05)``
    from the comparison script. We skip the stacked-Series + per-team-DataFrame
    round-trip used elsewhere because at NBA scale (millions of pairs) those
    pandas conversions are minutes per cell; the downstream scorer takes
    arrays directly. ``weights_full`` is indexed against the horizon-filtered
    matrix used to build the cache; per-team weights are sliced via the
    cached ``team_mask``.
    """
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for team, c in team_caches.items():
        team_w = weights_full[c["team_mask"]]
        corr_arr, eff_arr = _weighted_pairwise_corr(c["X"], c["M"], team_w)
        c_remap = 2 * np.sin(np.pi / 6 * corr_arr)
        # Element-wise shrinkage: full weight at ESS >= 30, linear decay to 0.
        shrink_w = np.minimum(eff_arr / MIN_OVERLAP_FOR_FULL_WEIGHT, 1.0)
        c_shrunk = c_remap * shrink_w
        # Magnitude floor: zero out weak pairs so they look like "no signal"
        # to the scorer (matches the production logic: floor + 0 = drop).
        c_shrunk = np.where(np.abs(c_shrunk) > CORR_MAGNITUDE_FLOOR, c_shrunk, 0.0)
        out[team] = (c["cols"], c_shrunk)
    return out


def _score_arrays_vs_beat_line(
    estimate_arrays: dict[str, tuple[list[str], np.ndarray]],
    beat_line_per_team: dict[str, pd.DataFrame],
) -> dict:
    """Vectorized variant of ``_score_against_beat_line_quantile`` for real data.

    Same bucket layout (bot 25 / mid / top 25 of no-self |pred|, plus
    extreme-tail at max(q95_no_self, q75_full) including same-player pairs),
    but built with numpy operations on per-team upper-triangle slices so
    it scales to NBA-size matrices (~7M pairs) in seconds rather than
    minutes. Same-player detection inlines ``a.split('.')[0] == b.split('.')[0]``
    via a vectorized prefix comparison.
    """
    pred_full_chunks: list[np.ndarray] = []
    actual_full_chunks: list[np.ndarray] = []
    same_player_chunks: list[np.ndarray] = []

    for team, truth in beat_line_per_team.items():
        truth_cols = list(truth.columns)
        if len(truth_cols) < 2:
            continue
        est_entry = estimate_arrays.get(team)
        if est_entry is None:
            est_aligned = np.zeros((len(truth_cols), len(truth_cols)))
        else:
            est_cols, est_arr = est_entry
            # Reindex est into truth's column order; missing pairs → 0.
            col_to_idx = {c: i for i, c in enumerate(est_cols)}
            idx = np.array([col_to_idx.get(c, -1) for c in truth_cols])
            present = idx >= 0
            est_aligned = np.zeros((len(truth_cols), len(truth_cols)))
            if present.any():
                src_idx = idx[present]
                # Build via two-step indexing to avoid creating an
                # (n,n) intermediate when only a subset of cols matches.
                sub = est_arr[np.ix_(src_idx, src_idx)]
                tgt = np.where(present)[0]
                est_aligned[np.ix_(tgt, tgt)] = sub

        truth_arr = truth.values
        n = len(truth_cols)
        iu_i, iu_j = np.triu_indices(n, k=1)
        p = est_aligned[iu_i, iu_j]
        t = truth_arr[iu_i, iu_j].astype(float)
        # Filter NaN truth pairs (pandas Pearson returns NaN for degenerate
        # variance — not a real "no opinion" signal we want scored).
        valid = ~np.isnan(t)
        if not valid.any():
            continue
        p = p[valid]
        t = t[valid]
        # Vectorized same-player detection: prefix-before-dot.
        prefixes = np.array([c.split(".", 1)[0] for c in truth_cols])
        same_mat = prefixes[:, None] == prefixes[None, :]
        same = same_mat[iu_i, iu_j][valid]

        pred_full_chunks.append(p)
        actual_full_chunks.append(t)
        same_player_chunks.append(same)

    if not pred_full_chunks:
        return {"n_pairs": 0.0}

    pa_full = np.concatenate(pred_full_chunks)
    aa_full = np.concatenate(actual_full_chunks)
    same = np.concatenate(same_player_chunks)
    no_self = ~same
    pa_ns = pa_full[no_self]
    aa_ns = aa_full[no_self]

    if len(pa_full) == 0 or len(pa_ns) == 0:
        return {"n_pairs": 0.0}

    abs_full = np.abs(pa_full)
    abs_ns = np.abs(pa_ns)

    # Sign agreement (same convention as the original scorer: only count
    # pairs where both magnitudes clear SIGN_AGREEMENT_MAG_FLOOR=0.05).
    sign_floor = 0.05
    sign_mask = (abs_full >= sign_floor) & (np.abs(aa_full) >= sign_floor)
    sign_total = int(sign_mask.sum())
    sign_match = int((np.sign(pa_full[sign_mask]) == np.sign(aa_full[sign_mask])).sum())

    q25 = float(np.quantile(abs_ns, 0.25))
    q75 = float(np.quantile(abs_ns, 0.75))
    q95_ns = float(np.quantile(abs_ns, 0.95))
    q75_full = float(np.quantile(abs_full, 0.75))
    extreme_cutoff = max(q95_ns, q75_full)

    def _lift(pa: np.ndarray, aa: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
        if not mask.any():
            return float("nan"), 0
        signed = np.where(pa[mask] >= 0, aa[mask], -aa[mask])
        return float(signed.mean()), int(mask.sum())

    bucket_specs = [
        (f"bot25(<{q25:.3f})",      _lift(pa_ns, aa_ns, abs_ns < q25)),
        (f"mid({q25:.3f}-{q75:.3f})", _lift(pa_ns, aa_ns, (abs_ns >= q25) & (abs_ns < q75))),
        (f"top25(>{q75:.3f})",      _lift(pa_ns, aa_ns, abs_ns >= q75)),
        (f"extreme(>{extreme_cutoff:.3f},+self)", _lift(pa_full, aa_full, abs_full >= extreme_cutoff)),
    ]
    bucket_labels = [s[0] for s in bucket_specs]
    bucket_means = [s[1][0] for s in bucket_specs]
    bucket_counts = [s[1][1] for s in bucket_specs]

    # spearmanr on multi-million arrays is the next bottleneck; argsort is
    # cheaper than scipy.stats.spearmanr's setup and gives the same number.
    rho = float(np.corrcoef(np.argsort(np.argsort(pa_full)), np.argsort(np.argsort(aa_full)))[0, 1])

    return {
        "n_pairs": float(len(pa_full)),
        "n_pairs_no_self": float(len(pa_ns)),
        "rho_pred_vs_actual": rho,
        "sign_agreement": (sign_match / sign_total) if sign_total else float("nan"),
        "n_sign_pairs": float(sign_total),
        "bucket_labels": bucket_labels,
        "bucket_means": bucket_means,
        "bucket_counts": bucket_counts,
        "q25": q25,
        "q75": q75,
        "q95_no_self": q95_ns,
        "q75_full": q75_full,
        "extreme_cutoff": extreme_cutoff,
    }


def _row_weights(
    train_dates: pd.Series, cutoff_date: pd.Timestamp, half_life_days: float
) -> np.ndarray:
    """Per-row exponential-decay weight, clamped to [eps, 1].

    ``Δt = (cutoff_date - row_date).days`` is non-negative on the train
    slice. ``half_life=inf`` returns uniform weights.
    """
    if not np.isfinite(half_life_days):
        return np.ones(len(train_dates), dtype=float)
    delta_days = (cutoff_date - pd.to_datetime(train_dates)).dt.days.values.astype(float)
    delta_days = np.maximum(delta_days, 0.0)
    return np.power(0.5, delta_days / half_life_days)


# ---------------------------------------------------------------------------
# Sweep + reporting
# ---------------------------------------------------------------------------


def _format_param(x: float) -> str:
    return "inf" if not np.isfinite(x) else f"{int(x)}"


def _print_grid(
    title: str,
    horizons: tuple[float, ...],
    half_lives: tuple[float, ...],
    grid: dict[tuple[float, float], float],
    fmt: str = "{:>+8.4f}",
    nan_str: str = "    nan ",
) -> None:
    """Render a 2D table of horizon (rows) × half_life (columns)."""
    click.echo("")
    click.echo(f"      {title}")
    label = "horizon \\ half_life"
    header = "      " + f"{label:>20}" + "  " + "  ".join(
        f"{_format_param(h):>10}" for h in half_lives
    )
    click.echo(header)
    click.echo("      " + "-" * 20 + "  " + "  ".join("-" * 10 for _ in half_lives))
    for h in horizons:
        cells = []
        for hl in half_lives:
            v = grid.get((h, hl), float("nan"))
            cells.append(fmt.format(v) if np.isfinite(v) else nan_str)
        click.echo("      " + f"{_format_param(h):>20}" + "  " + "  ".join(cells))


def run_sweep(
    league: str,
    use_cache: bool,
    horizons: tuple[float, ...],
    half_lives: tuple[float, ...],
) -> None:
    click.echo(f"Time-weighted correlation sweep — league={league}")
    click.echo("=" * 72)
    click.echo("[1/3] Loading pivots")
    res_matrix, beat_matrix = _load_or_build_pivots(league, use_cache)

    res_matrix = res_matrix.sort_values("DATE").reset_index(drop=True)
    beat_matrix = beat_matrix.sort_values("DATE").reset_index(drop=True)
    n = len(res_matrix)
    cutoff_date = res_matrix.loc[int(n * TRAIN_FRACTION), "DATE"]
    cutoff_ts = pd.Timestamp(cutoff_date)
    click.echo(
        f"      train/test cutoff = {cutoff_ts.date()} "
        f"({int(n*TRAIN_FRACTION)}/{n} rows)"
    )

    res_train_full = res_matrix.loc[res_matrix["DATE"] <= cutoff_ts]
    beat_test = beat_matrix.loc[beat_matrix["DATE"] > cutoff_ts]
    if len(beat_test) < 50:
        click.echo("      fewer than 50 test rows — aborting")
        return

    click.echo("[2/3] Building empirical beat-line target on test slice")
    test_beat_line = _empirical_beat_line_corr(beat_test)

    click.echo(
        f"[3/3] Running sweep over {len(horizons)} horizons × "
        f"{len(half_lives)} half-lives = {len(horizons) * len(half_lives)} cells"
    )

    top25_grid: dict[tuple[float, float], float] = {}
    extreme_grid: dict[tuple[float, float], float] = {}
    rho_grid: dict[tuple[float, float], float] = {}
    sign_grid: dict[tuple[float, float], float] = {}
    n_top25_grid: dict[tuple[float, float], int] = {}
    n_extreme_grid: dict[tuple[float, float], int] = {}

    train_dates_full = pd.to_datetime(res_train_full["DATE"])

    total_cells = len(horizons) * len(half_lives)
    pbar = tqdm(total=total_cells, desc="sweep", unit="cell")
    for horizon in horizons:
        if not np.isfinite(horizon):
            mask_h = np.ones(len(res_train_full), dtype=bool)
        else:
            mask_h = (
                (cutoff_ts - train_dates_full).dt.days <= horizon
            ).values
        train_h = res_train_full.loc[mask_h]
        train_dates_h = train_dates_full.loc[mask_h]
        if len(train_h) < 100:
            pbar.write(
                f"      horizon={_format_param(horizon)} → only "
                f"{len(train_h)} train rows; skipping"
            )
            pbar.update(len(half_lives))
            continue
        # Ranks don't depend on weights, so cache them per horizon and
        # reuse across all half_life cells (6× speedup for 6 half_lives).
        team_caches = _build_team_caches(train_h)
        for half_life in half_lives:
            weights = _row_weights(train_dates_h, cutoff_ts, half_life)
            estimate_arrays = _build_corr_new_weighted_arrays(team_caches, weights)
            metrics = _score_arrays_vs_beat_line(estimate_arrays, test_beat_line)
            del estimate_arrays
            gc.collect()
            pbar.update(1)
            pbar.set_postfix(
                horizon=_format_param(horizon),
                half_life=_format_param(half_life),
                n_train=len(train_h),
            )
            if metrics.get("n_pairs", 0) == 0:
                continue
            top25_grid[(horizon, half_life)] = metrics["bucket_means"][2]
            n_top25_grid[(horizon, half_life)] = metrics["bucket_counts"][2]
            extreme_grid[(horizon, half_life)] = metrics["bucket_means"][3]
            n_extreme_grid[(horizon, half_life)] = metrics["bucket_counts"][3]
            rho_grid[(horizon, half_life)] = metrics["rho_pred_vs_actual"]
            sign_grid[(horizon, half_life)] = metrics["sign_agreement"]
            pbar.write(
                f"      horizon={_format_param(horizon):>4}d  "
                f"half_life={_format_param(half_life):>4}d  "
                f"n_train={len(train_h):>5d}  "
                f"top25_lift={metrics['bucket_means'][2]:+.4f} "
                f"(n={metrics['bucket_counts'][2]})  "
                f"extreme_lift={metrics['bucket_means'][3]:+.4f} "
                f"(n={metrics['bucket_counts'][3]})  "
                f"rho={metrics['rho_pred_vs_actual']:+.4f}  "
                f"sign={metrics['sign_agreement']:.4f}"
            )
    pbar.close()

    _print_grid(
        "top-25%-no-self bucket direction-matched lift (q75_no_self threshold)",
        horizons,
        half_lives,
        top25_grid,
    )
    _print_grid(
        "extreme-bucket direction-matched lift (higher = better; baseline ~+0.10)",
        horizons,
        half_lives,
        extreme_grid,
    )
    _print_grid(
        "Spearman rho pred vs actual (higher = better)",
        horizons,
        half_lives,
        rho_grid,
    )
    _print_grid(
        "sign agreement (higher = better; chance = 0.5)",
        horizons,
        half_lives,
        sign_grid,
        fmt="{:>10.4f}",
    )

    click.echo("")
    click.echo("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_param_list(s: str) -> tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(float("inf") if p.lower() == "inf" else float(p) for p in parts)


@click.command()
@click.option("--league", type=click.Choice(["NFL", "NBA", "WNBA", "NHL", "MLB"]), default="NBA")
@click.option(
    "--horizons",
    default=",".join(_format_param(h) for h in DEFAULT_HORIZONS_DAYS),
    help="Comma-separated horizon days (use 'inf' for no filter).",
)
@click.option(
    "--half-lives",
    default=",".join(_format_param(h) for h in DEFAULT_HALF_LIVES_DAYS),
    help="Comma-separated half-life days (use 'inf' for uniform weights).",
)
@click.option("--no-cache", is_flag=True, help="Force pivot rebuild (ignores parquet cache).")
def main(league: str, horizons: str, half_lives: str, no_cache: bool) -> None:
    h = _parse_param_list(horizons)
    hl = _parse_param_list(half_lives)
    run_sweep(league, use_cache=not no_cache, horizons=h, half_lives=hl)


if __name__ == "__main__":
    main()
