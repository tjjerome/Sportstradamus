"""Empirically test whether the new correlation methodology beats the old.

Two experiments:

1. Synthetic ground truth (always runs). Generates per-game stats from a known
   correlation structure plus per-player skill drift and missing-game noise,
   then scores how well each method recovers the truth.
2. Real-data time-based holdout (runs only if a Stats gamelog is available
   locally). Splits the user's gamelog 70/30 by date, builds correlations on
   train, scores against the empirical correlation on test.

Usage:

    poetry run python scripts/test_correlation_predictive_power.py --skip-real
    poetry run python scripts/test_correlation_predictive_power.py --league NBA
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import cast

import click
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _load_correlate_module():
    """Load ``sportstradamus.training.correlate`` with two strategies.

    Strategy A (preferred): a normal ``import sportstradamus.training.correlate``.
    This is what runs in any environment with the package properly installed
    (i.e. anywhere ``poetry install`` succeeded), and it leaves the rest of
    the ``sportstradamus`` package importable so experiment 3 can do
    ``from sportstradamus import stats``.

    Strategy B (fallback for sandboxes without the full dep tree): load
    ``correlate.py`` directly by file path, after stubbing the
    ``sportstradamus`` and ``sportstradamus.data`` package entries so the
    ``from sportstradamus import data`` line at the top of correlate.py
    resolves. This path is only taken when strategy A fails — typically
    because optional ML deps or credentials referenced in the package
    ``__init__`` chain are missing. Once strategy B runs, the rest of the
    package can no longer be imported normally; experiment 3 will detect
    that and skip with a clear message.
    """
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    try:
        # NOTE: do not use `import sportstradamus.training.correlate as mod` —
        # the training package __init__ does `from .correlate import correlate`,
        # which rebinds the `correlate` attribute on the package namespace to
        # the function, and `import ... as` resolves via attribute access.
        # importlib.import_module reads sys.modules and returns the submodule.
        return importlib.import_module("sportstradamus.training.correlate")
    except Exception as exc:
        click.echo(
            f"      (note: normal import of sportstradamus.training.correlate failed: "
            f"{type(exc).__name__}: {exc}; falling back to file-load. Experiment 3 "
            f"will be unavailable in this mode.)",
            err=True,
        )

    correlate_path = repo_root / "src/sportstradamus/training/correlate.py"
    for pkg_name, pkg_path in [
        ("sportstradamus", src_root / "sportstradamus"),
        ("sportstradamus.data", src_root / "sportstradamus" / "data"),
    ]:
        if pkg_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            pkg_name, pkg_path / "__init__.py", submodule_search_locations=[str(pkg_path)]
        )
        if spec is None or spec.loader is None:
            stub = importlib.util.module_from_spec(
                importlib.util.spec_from_loader(pkg_name, loader=None)
            )
            stub.__path__ = [str(pkg_path)]
            sys.modules[pkg_name] = stub
        else:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[pkg_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                stub = importlib.util.module_from_spec(
                    importlib.util.spec_from_loader(pkg_name, loader=None)
                )
                stub.__path__ = [str(pkg_path)]
                sys.modules[pkg_name] = stub

    spec = importlib.util.spec_from_file_location(
        "sportstradamus.training.correlate", correlate_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sportstradamus.training.correlate"] = mod
    spec.loader.exec_module(mod)
    return mod


_correlate = _load_correlate_module()
CORR_MAGNITUDE_FLOOR = _correlate.CORR_MAGNITUDE_FLOOR
MIN_OVERLAP_FOR_FULL_WEIGHT = _correlate.MIN_OVERLAP_FOR_FULL_WEIGHT
_residualize_gamelog = _correlate._residualize_gamelog
_shrink_correlations = _correlate._shrink_correlations

# Synthetic experiment defaults — small enough to finish in seconds, large
# enough that low-overlap pairs actually appear after we drop ~20% of rows.
DEFAULT_SEED: int = 42
DEFAULT_N_GAMES: int = 200
DEFAULT_N_TEAMS: int = 20
N_PLAYERS_PER_TEAM: int = 6
N_STATS_PER_PLAYER: int = 2
PLAYER_DROP_PROB: float = 0.20
SKILL_DRIFT_AR1_PHI: float = 0.95
SKILL_DRIFT_INNOVATION_SD: float = 0.4

# How we judge "the methods agreed with the holdout" in experiment 2 — we
# only count pairs with non-trivial magnitude on both sides.
SIGN_AGREEMENT_MAG_FLOOR: float = 0.05

# Fraction of games used for the train slice in the real-data experiment.
TRAIN_FRACTION: float = 0.70


# ---------------------------------------------------------------------------
# Methodology implementations (pure: matrix -> stacked Series)
# ---------------------------------------------------------------------------


def build_corr_old(matrix: pd.DataFrame) -> pd.Series:
    """Replicate the legacy logic from ``src/deprecated/correlation.py:442-457``.

    Drops the DATE column, fills NaN with 0, drops columns with >=50% zeros
    per team, runs Spearman with a 75% min_periods cutoff, applies the
    Fisher-style remap, and keeps pairs above a tiny magnitude floor. No
    residualization, no shrinkage.
    """
    matrix = matrix.drop(columns="DATE", errors="ignore").copy()
    matrix = matrix.fillna(0)
    blocks: dict = {}
    for team in matrix["TEAM"].unique():
        team_matrix = matrix.loc[matrix["TEAM"] == team].drop(columns="TEAM")
        team_matrix = team_matrix.loc[:, ((team_matrix == 0).mean() < 0.5)]
        if team_matrix.shape[1] < 2 or len(team_matrix) < 2:
            continue
        team_matrix = team_matrix.reindex(sorted(team_matrix.columns), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_spearman = team_matrix.corr(
                method="spearman",
                min_periods=int(len(team_matrix) * 0.75),
            )
        c = 2 * np.sin(np.pi / 6 * c_spearman)
        c_stack = c.unstack().dropna()
        c_stack = c_stack.loc[c_stack.abs() > 0.001]
        if not c_stack.empty:
            blocks[team] = c_stack
    if not blocks:
        return pd.Series([], dtype=float, name="R")
    out = pd.concat(blocks)
    out.name = "R"
    return out


def _team_corr_pairwise(
    matrix: pd.DataFrame,
    *,
    shrink: bool,
    magnitude_floor: float,
) -> pd.Series:
    """Per-team Spearman + Fisher remap with pairwise-complete handling.

    Used as a building block for the ablation variants. Skips fillna, drops
    only fully-NaN columns, and (optionally) shrinks low-overlap pairs.
    """
    matrix = matrix.drop(columns="DATE", errors="ignore").copy()
    blocks: dict = {}
    for team in matrix["TEAM"].unique():
        team_matrix = matrix.loc[matrix["TEAM"] == team].drop(columns="TEAM")
        team_matrix = team_matrix.loc[:, team_matrix.notna().any(axis=0)]
        if team_matrix.shape[1] < 2 or len(team_matrix) < 2:
            continue
        team_matrix = team_matrix.reindex(sorted(team_matrix.columns), axis=1)
        present = team_matrix.notna().astype(int)
        overlap = present.T @ present
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_spearman = team_matrix.corr(method="spearman")
        c_remap = 2 * np.sin(np.pi / 6 * c_spearman)
        c = _shrink_correlations(c_remap, overlap) if shrink else c_remap
        c_stack = c.unstack().dropna()
        c_stack = c_stack.loc[c_stack.abs() > magnitude_floor]
        if not c_stack.empty:
            blocks[team] = c_stack
    if not blocks:
        return pd.Series([], dtype=float, name="R")
    out = pd.concat(blocks)
    out.name = "R"
    return out


def build_corr_pairwise_only(matrix_raw: pd.DataFrame) -> pd.Series:
    """Ablation A: same as OLD but with NaN handling like NEW.

    Pairwise-complete Spearman on raw (non-residualized) values, no
    fillna(0), no shrinkage. Isolates the contribution of dropping the
    fillna step from residualization + shrinkage.
    """
    return _team_corr_pairwise(matrix_raw, shrink=False, magnitude_floor=0.001)


def build_corr_residual_only(matrix_residualized: pd.DataFrame) -> pd.Series:
    """Ablation B: residualization + pairwise NaN handling, no shrinkage.

    Isolates the contribution of residualization on top of NaN handling.
    """
    return _team_corr_pairwise(matrix_residualized, shrink=False, magnitude_floor=0.001)


def build_corr_new(matrix_residualized: pd.DataFrame) -> pd.Series:
    """Full new methodology: residualize + pairwise NaN + shrinkage + floor."""
    return _team_corr_pairwise(
        matrix_residualized, shrink=True, magnitude_floor=CORR_MAGNITUDE_FLOOR
    )


# ---------------------------------------------------------------------------
# Experiment 1 — synthetic ground truth
# ---------------------------------------------------------------------------


def _make_true_corr(n_stats: int, rng: np.random.Generator) -> np.ndarray:
    """Build a positive-definite ground-truth correlation matrix.

    Uses a low-rank-plus-diagonal construction so we get a realistic mix of
    near-zero and ±0.3-0.6 entries instead of a uniform random sprinkle.
    """
    rank = max(2, n_stats // 4)
    loadings = rng.normal(scale=0.6, size=(n_stats, rank))
    cov = loadings @ loadings.T + np.eye(n_stats) * 0.5
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def _simulate_gamelog(
    n_teams: int,
    n_games: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Simulate a per-player gamelog with a known per-team correlation matrix.

    Each team has its own (identical-shape) latent stat correlation. Each
    game we draw a team-level multivariate-normal vector, then assign the
    components to player-stat slots. Per-player AR(1) skill drift is added
    on top — the OLD method picks this up as spurious cross-stat correlation
    on the same player; residualization should remove it.

    Returns:
        (gamelog_df, true_corr_per_team) where gamelog has columns
        ``player``, ``team``, ``DATE``, ``gameId``, plus one column per
        stat slot and ``true_corr_per_team`` maps team -> matrix indexed by
        the stat column names.
    """
    n_stat_slots = N_PLAYERS_PER_TEAM * N_STATS_PER_PLAYER

    teams = [f"T{i:02d}" for i in range(n_teams)]
    stat_cols = [
        f"P{p}_S{s}" for p in range(N_PLAYERS_PER_TEAM) for s in range(N_STATS_PER_PLAYER)
    ]

    true_per_team: dict[str, np.ndarray] = {}
    for team in teams:
        true_per_team[team] = _make_true_corr(n_stat_slots, rng)

    # Fixed home/away rotation — every team plays once per "game day".
    # We pair teams across the league each day so each team logs n_games rows.
    perm_order = np.array(teams)
    skills = np.zeros((n_teams, N_PLAYERS_PER_TEAM))

    rows: list[dict] = []
    for game_idx in range(n_games):
        # AR(1) drift on per-player skill (shared across that player's stats).
        skills = (
            SKILL_DRIFT_AR1_PHI * skills
            + SKILL_DRIFT_INNOVATION_SD
            * rng.standard_normal(size=(n_teams, N_PLAYERS_PER_TEAM))
        )
        rng.shuffle(perm_order)
        for pair_idx in range(0, n_teams, 2):
            home, away = perm_order[pair_idx], perm_order[pair_idx + 1]
            game_id = f"G{game_idx:04d}_{home}_{away}"
            for team in (home, away):
                team_i = teams.index(team)
                draw = rng.multivariate_normal(
                    mean=np.zeros(n_stat_slots),
                    cov=true_per_team[team],
                )
                # Add per-player drift to that player's stat slots.
                drift = np.repeat(skills[team_i], N_STATS_PER_PLAYER)
                draw = draw + drift
                for p in range(N_PLAYERS_PER_TEAM):
                    if rng.random() < PLAYER_DROP_PROB:
                        continue  # player out for this game
                    row = {
                        "player": f"{team}_P{p}",
                        "team": team,
                        "DATE": pd.Timestamp("2024-01-01") + pd.Timedelta(days=game_idx),
                        "gameId": game_id,
                    }
                    for s in range(N_STATS_PER_PLAYER):
                        slot = p * N_STATS_PER_PLAYER + s
                        row[f"P{p}_S{s}"] = draw[slot]
                    rows.append(row)
    gamelog = pd.DataFrame(rows)
    # Map each per-team latent matrix into a labeled DataFrame for scoring.
    labeled = {team: pd.DataFrame(true_per_team[team], index=stat_cols, columns=stat_cols)
               for team in teams}
    return gamelog, labeled


def _gamelog_to_team_matrix(gamelog: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    """Pivot a per-player gamelog into per-team-per-game rows.

    For each (gameId, team), the stats from each player slot become columns
    keyed ``{player_slot}.{stat}``. We don't include opponent (_OPP_*)
    columns here — the synthetic experiment only judges within-team
    correlations, which is the comparison we care about.
    """
    rows = []
    grouped = gamelog.groupby(["gameId", "team"])
    for (game_id, team), g in grouped:
        # Map player to their per-team slot index P0..P{N-1} from the
        # player name suffix (synthetic players are named T01_P3 etc.).
        row = {"TEAM": team, "DATE": g["DATE"].iloc[0]}
        for _, prow in g.iterrows():
            slot = prow["player"].split("_")[1]  # "P3"
            for stat in stat_cols:
                if not stat.startswith(slot + "_"):
                    continue
                row[stat] = prow[stat]
        rows.append(row)
    return pd.DataFrame(rows)


def _stacked_to_per_team(stacked: pd.Series) -> dict[str, pd.DataFrame]:
    """Inflate a stacked (team, a, b) -> R series back into per-team matrices."""
    out: dict[str, pd.DataFrame] = {}
    if stacked.empty:
        return out
    for team, sub in stacked.groupby(level=0):
        s = sub.droplevel(0)
        cols = sorted({*s.index.get_level_values(0), *s.index.get_level_values(1)})
        m = pd.DataFrame(0.0, index=cols, columns=cols)
        for (a, b), r in s.items():
            m.loc[a, b] = r
        out[team] = m
    return out


def _score_recovery(
    estimate_per_team: dict[str, pd.DataFrame],
    truth_per_team: dict[str, pd.DataFrame],
    overlap_per_team: dict[str, pd.DataFrame] | None = None,
) -> dict[str, float]:
    """Compute MAE / Spearman / subset MAE between estimate and truth.

    We unroll the upper triangle (i < j) to avoid double-counting and skip
    diagonal self-correlations.
    """
    all_pred: list[float] = []
    all_true: list[float] = []
    low_overlap_pred: list[float] = []
    low_overlap_true: list[float] = []
    confounded_pred: list[float] = []
    confounded_true: list[float] = []

    for team, truth in truth_per_team.items():
        est = estimate_per_team.get(team)
        cols = list(truth.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                t = float(truth.loc[a, b])
                # Estimate may have dropped column or pair (sparse output).
                if est is not None and a in est.index and b in est.columns:
                    p = float(est.loc[a, b])
                else:
                    p = 0.0
                all_pred.append(p)
                all_true.append(t)
                # Confounded == same player, different stats. Truth here is
                # whatever the latent matrix said (often near zero), and the
                # AR(1) drift inflates the OLD estimator's value.
                if a.split("_")[0] == b.split("_")[0]:
                    confounded_pred.append(p)
                    confounded_true.append(t)
                if overlap_per_team is not None:
                    o = overlap_per_team.get(team)
                    if (
                        o is not None
                        and a in o.index
                        and b in o.columns
                        and o.loc[a, b] < MIN_OVERLAP_FOR_FULL_WEIGHT
                    ):
                        low_overlap_pred.append(p)
                        low_overlap_true.append(t)

    pred_arr = np.array(all_pred)
    true_arr = np.array(all_true)
    metrics = {
        "n_pairs": float(len(pred_arr)),
        "mae": float(np.mean(np.abs(pred_arr - true_arr))) if len(pred_arr) else float("nan"),
        "rmse": float(np.sqrt(np.mean((pred_arr - true_arr) ** 2))) if len(pred_arr) else float("nan"),
        "spearman_rho": (
            float(spearmanr(pred_arr, true_arr).statistic) if len(pred_arr) > 1 else float("nan")
        ),
        "confounded_mae": (
            float(np.mean(np.abs(np.array(confounded_pred) - np.array(confounded_true))))
            if confounded_pred
            else float("nan")
        ),
        "n_confounded": float(len(confounded_pred)),
        "low_overlap_mae": (
            float(np.mean(np.abs(np.array(low_overlap_pred) - np.array(low_overlap_true))))
            if low_overlap_pred
            else float("nan")
        ),
        "n_low_overlap": float(len(low_overlap_pred)),
    }
    return metrics


def _per_team_overlap(matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pair-overlap counts per team — used to mark "low-overlap" pairs."""
    matrix = matrix.drop(columns="DATE", errors="ignore")
    out: dict[str, pd.DataFrame] = {}
    for team in matrix["TEAM"].unique():
        team_matrix = matrix.loc[matrix["TEAM"] == team].drop(columns="TEAM")
        present = team_matrix.notna().astype(int)
        out[team] = present.T @ present
    return out


def run_synthetic(seed: int, n_games: int) -> None:
    rng = np.random.default_rng(seed)
    click.echo("[1/3] Synthetic latent-corr recovery experiment")
    click.echo(f"      seed={seed} n_games={n_games} n_teams={DEFAULT_N_TEAMS}")
    click.echo("      simulating gamelog with per-player AR(1) skill drift + 20% missing rows")

    gamelog, truth = _simulate_gamelog(DEFAULT_N_TEAMS, n_games, rng)
    stat_cols = [c for c in gamelog.columns if c.startswith("P")]

    raw_team_matrix = _gamelog_to_team_matrix(gamelog, stat_cols)

    # Residualize the per-player gamelog, then pivot — this is what the
    # production new code does internally.
    residualized_log = _residualize_gamelog(gamelog, "player", "DATE", stat_cols)
    res_team_matrix = _gamelog_to_team_matrix(residualized_log, stat_cols)

    overlap_new = _per_team_overlap(res_team_matrix)
    overlap_raw = _per_team_overlap(raw_team_matrix)

    # Four variants for the ablation:
    #   OLD            = fillna(0) + drop>=50%-zero cols + min_periods spearman   (legacy)
    #   PAIRWISE_ONLY  = pairwise-complete spearman on RAW values                 (NaN-handling delta only)
    #   RESIDUAL_ONLY  = pairwise + residualization, no shrinkage                  (adds residualization)
    #   NEW            = pairwise + residualization + shrinkage + 0.05 floor       (full new method)
    variants = [
        ("OLD", build_corr_old(raw_team_matrix), overlap_raw),
        ("PAIRWISE_ONLY", build_corr_pairwise_only(raw_team_matrix), overlap_raw),
        ("RESIDUAL_ONLY", build_corr_residual_only(res_team_matrix), overlap_new),
        ("NEW", build_corr_new(res_team_matrix), overlap_new),
    ]
    metrics_per_variant: dict[str, dict[str, float]] = {}
    for name, stacked, overlap in variants:
        per_team = _stacked_to_per_team(stacked)
        metrics_per_variant[name] = _score_recovery(per_team, truth, overlap)

    _print_ablation_table(metrics_per_variant)


def _print_ablation_table(metrics_per_variant: dict[str, dict[str, float]]) -> None:
    """Side-by-side table for OLD / PAIRWISE_ONLY / RESIDUAL_ONLY / NEW.

    Reading the columns left-to-right shows what each methodology change
    contributed in isolation: PAIRWISE_ONLY isolates NaN handling,
    RESIDUAL_ONLY adds residualization, NEW adds shrinkage + magnitude floor.
    """
    rows = [
        ("pairs scored", "n_pairs", "{:>10.0f}"),
        ("MAE (lower better)", "mae", "{:>10.4f}"),
        ("RMSE (lower better)", "rmse", "{:>10.4f}"),
        ("Spearman rank rho (higher better)", "spearman_rho", "{:>10.4f}"),
        ("confounded-pair MAE (lower better)", "confounded_mae", "{:>10.4f}"),
        ("low-overlap pair MAE (lower better)", "low_overlap_mae", "{:>10.4f}"),
        ("n confounded", "n_confounded", "{:>10.0f}"),
        ("n low overlap (<30 shared)", "n_low_overlap", "{:>10.0f}"),
    ]
    names = ["OLD", "PAIRWISE_ONLY", "RESIDUAL_ONLY", "NEW"]
    click.echo("")
    header = "      " + f"{'metric':<38}" + "  " + "  ".join(f"{n:>14}" for n in names)
    click.echo(header)
    click.echo("      " + "-" * 38 + "  " + "  ".join("-" * 14 for _ in names))
    for label, key, fmt in rows:
        cells = []
        for n in names:
            v = metrics_per_variant[n][key]
            cells.append(fmt.replace(":>10", ":>14").format(v))
        click.echo(f"      {label:<38}  " + "  ".join(cells))


# ---------------------------------------------------------------------------
# Experiment 2 — beat-line conditional probability (production-relevant)
# ---------------------------------------------------------------------------
#
# Production cares about a downstream conditional, not the latent stat
# correlation in the abstract: if Player A beats their line in market X,
# is Player B more likely to beat their line in market Y? The correlation
# matrix gets multiplied into parlay EV in
# ``prediction/correlation.py:389-402`` — positive value between two "Over"
# legs raises joint probability, negative lowers it. So the right ground
# truth is the empirical correlation of {beat-line indicator}, not the
# Pearson correlation of the raw stat values.
#
# We approximate bookmaker line-setting with each player's leak-free rolling
# 8-game mean (same window the production residualization uses). That makes
# the test slightly favorable to residualization by construction — the line
# is exactly what residualization subtracts — but it's also the most honest
# proxy for what real lines do, so the comparison still tells us which
# method's output best matches the conditional we actually consume.


def _beat_line_team_matrix(
    gamelog: pd.DataFrame,
    stat_cols: list[str],
) -> pd.DataFrame:
    """Build per-team-per-game beat-line booleans (1 / 0 / NaN per stat).

    For each (player, stat) we set the line equal to the player's
    leak-free rolling-N-game mean — the same window used by
    ``_residualize_gamelog``. ``hit[t] = 1`` if ``stat[t] > line[t]``,
    ``0`` otherwise; ``NaN`` when there isn't enough history yet (so the
    line is undefined and the player would not have a posted line in
    real life either).

    The empirical correlation of these per-team boolean columns IS the
    quantity production cares about — it's the realized conditional
    structure of "did A beat its line jointly with B beating its line".
    """
    line_df = _residualize_gamelog(gamelog, "player", "DATE", stat_cols)
    hits = line_df.copy()
    for stat in stat_cols:
        if stat not in line_df.columns:
            continue
        # Residual = stat - rolling_line, so beat-line iff residual > 0.
        # Residuals that are exactly 0 (continuous data: never) count as
        # "not beat" same as the bookmaker convention.
        hits[stat] = (line_df[stat] > 0).astype("float")
        hits.loc[line_df[stat].isna(), stat] = np.nan
    return _gamelog_to_team_matrix(hits, stat_cols)


def _empirical_beat_line_corr(matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-team Pearson correlation of beat-line booleans across games.

    Pearson on 0/1 indicators is the phi coefficient — exactly the binary
    co-occurrence structure production wants the matrix to predict.
    """
    matrix = matrix.drop(columns="DATE", errors="ignore")
    out: dict[str, pd.DataFrame] = {}
    for team in matrix["TEAM"].unique():
        team_matrix = matrix.loc[matrix["TEAM"] == team].drop(columns="TEAM")
        team_matrix = team_matrix.loc[:, team_matrix.notna().any(axis=0)]
        # Need at least 2 games and 2 stats with non-degenerate variance.
        if team_matrix.shape[1] < 2 or len(team_matrix) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[team] = cast(pd.DataFrame, team_matrix.corr(method="pearson"))
    return out


def _score_against_beat_line(
    estimate_per_team: dict[str, pd.DataFrame],
    beat_line_per_team: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Compare each method's predicted pair correlations to beat-line truth.

    We also bucket pairs by predicted-correlation magnitude and report the
    mean realized beat-line correlation per bucket — this is the most
    directly interpretable answer to the user's question. If the method's
    output is meaningful, "high predicted" pairs should have higher mean
    realized beat-line correlation than "low predicted" pairs.
    """
    pred: list[float] = []
    actual: list[float] = []
    sign_match = 0
    sign_total = 0

    for team, truth in beat_line_per_team.items():
        est = estimate_per_team.get(team)
        cols = list(truth.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                t = float(truth.loc[a, b])
                if pd.isna(t):
                    continue
                if est is not None and a in est.index and b in est.columns:
                    p = float(est.loc[a, b])
                else:
                    p = 0.0
                pred.append(p)
                actual.append(t)
                if abs(p) >= SIGN_AGREEMENT_MAG_FLOOR and abs(t) >= SIGN_AGREEMENT_MAG_FLOOR:
                    sign_total += 1
                    if np.sign(p) == np.sign(t):
                        sign_match += 1

    pa = np.array(pred)
    aa = np.array(actual)
    if len(pa) == 0:
        return {"n_pairs": 0.0}

    # Bucket realized lift by predicted-correlation magnitude. Buckets
    # are: |pred| < 0.05 (background), 0.05-0.20 (weak), 0.20-0.40
    # (moderate), > 0.40 (strong). For a method with predictive power,
    # bucket means of |actual| should rise monotonically.
    from itertools import pairwise

    abs_pred = np.abs(pa)
    bucket_edges = [0.0, 0.05, 0.20, 0.40, np.inf]
    bucket_labels = ["bg(<.05)", "weak(.05-.20)", "mod(.20-.40)", "strong(>.40)"]
    bucket_means = []
    bucket_counts = []
    for lo, hi in pairwise(bucket_edges):
        mask = (abs_pred >= lo) & (abs_pred < hi)
        if mask.any():
            # Use signed actual but matched-direction: when predicted is
            # positive we want actual positive; when negative, we want
            # negative. Equivalent to: predicted * actual > 0 for "match".
            signed_actual = np.where(pa[mask] >= 0, aa[mask], -aa[mask])
            bucket_means.append(float(np.mean(signed_actual)))
        else:
            bucket_means.append(float("nan"))
        bucket_counts.append(int(mask.sum()))

    return {
        "n_pairs": float(len(pa)),
        "mae_vs_beat_line": float(np.mean(np.abs(pa - aa))),
        "rho_pred_vs_actual": (
            float(spearmanr(pa, aa).statistic) if len(pa) > 1 else float("nan")
        ),
        "sign_agreement": (sign_match / sign_total) if sign_total else float("nan"),
        "n_sign_pairs": float(sign_total),
        # Per-bucket mean of direction-matched realized beat-line correlation.
        # Higher (positive) = predictions in this bucket really do correspond
        # to legs that co-occur more often (or, for negative predictions, to
        # legs that anti-correlate as predicted).
        "bucket_labels": bucket_labels,
        "bucket_means": bucket_means,
        "bucket_counts": bucket_counts,
    }


def _same_player_real(a: str, b: str) -> bool:
    """Two real-data column names refer to the same player.

    Real-data column format from ``_build_team_game_records`` is
    ``{position}.{stat}`` for own team (e.g. ``PG1.PTS``) and
    ``_OPP_{position}.{stat}`` for opponent. Player identity = everything
    before the first ``.``; opponent vs own-team prefix prevents accidental
    cross-team matches because ``_OPP_PG1`` != ``PG1``.
    """
    return a.split(".")[0] == b.split(".")[0]


def _score_against_beat_line_quantile(
    estimate_per_team: dict[str, pd.DataFrame],
    beat_line_per_team: dict[str, pd.DataFrame],
    same_player_fn,
) -> dict:
    """Real-data variant of beat-line scoring with data-driven bucket cutoffs.

    Bucket layout (4 rows):
      1. bottom 25% of |pred|, same-player pairs excluded
      2. middle 25%-75%, same-player pairs excluded
      3. top 25% (>75th pct), same-player pairs excluded
      4. extreme tail INCLUDING same-player pairs (i<j only, so the trivial
         A.X/A.X self-correlation is already excluded), threshold =
         max(95th pct of no-self |pred|, 75th pct of full |pred|).

    All cutoffs are computed per-variant from the variant's own |pred|
    distribution — the synthetic hardcoded edges (0.05/0.20/0.40) are not
    reused because real-data magnitude scales differ.
    """
    pred_full: list[float] = []
    actual_full: list[float] = []
    pred_no_self: list[float] = []
    actual_no_self: list[float] = []
    sign_match = 0
    sign_total = 0

    for team, truth in beat_line_per_team.items():
        est = estimate_per_team.get(team)
        cols = list(truth.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                t = float(truth.loc[a, b])
                if pd.isna(t):
                    continue
                if est is not None and a in est.index and b in est.columns:
                    p = float(est.loc[a, b])
                else:
                    p = 0.0
                pred_full.append(p)
                actual_full.append(t)
                if not same_player_fn(a, b):
                    pred_no_self.append(p)
                    actual_no_self.append(t)
                if abs(p) >= SIGN_AGREEMENT_MAG_FLOOR and abs(t) >= SIGN_AGREEMENT_MAG_FLOOR:
                    sign_total += 1
                    if np.sign(p) == np.sign(t):
                        sign_match += 1

    pa_full = np.array(pred_full)
    aa_full = np.array(actual_full)
    pa_ns = np.array(pred_no_self)
    aa_ns = np.array(actual_no_self)
    if len(pa_full) == 0 or len(pa_ns) == 0:
        return {"n_pairs": 0.0}

    abs_ns = np.abs(pa_ns)
    abs_full = np.abs(pa_full)
    q25 = float(np.quantile(abs_ns, 0.25))
    q75 = float(np.quantile(abs_ns, 0.75))
    q95_ns = float(np.quantile(abs_ns, 0.95))
    q75_full = float(np.quantile(abs_full, 0.75))
    extreme_cutoff = max(q95_ns, q75_full)

    def _bucket_lift(pa: np.ndarray, aa: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
        if not mask.any():
            return float("nan"), 0
        signed = np.where(pa[mask] >= 0, aa[mask], -aa[mask])
        return float(np.mean(signed)), int(mask.sum())

    bucket_specs = [
        (f"bot25(<{q25:.3f})",      _bucket_lift(pa_ns, aa_ns, abs_ns < q25)),
        (f"mid({q25:.3f}-{q75:.3f})", _bucket_lift(pa_ns, aa_ns, (abs_ns >= q25) & (abs_ns < q75))),
        (f"top25(>{q75:.3f})",      _bucket_lift(pa_ns, aa_ns, abs_ns >= q75)),
        (f"extreme(>{extreme_cutoff:.3f},+self)", _bucket_lift(pa_full, aa_full, abs_full >= extreme_cutoff)),
    ]
    bucket_labels = [s[0] for s in bucket_specs]
    bucket_means = [s[1][0] for s in bucket_specs]
    bucket_counts = [s[1][1] for s in bucket_specs]

    return {
        "n_pairs": float(len(pa_full)),
        "n_pairs_no_self": float(len(pa_ns)),
        "mae_vs_beat_line": float(np.mean(np.abs(pa_full - aa_full))),
        "rho_pred_vs_actual": (
            float(spearmanr(pa_full, aa_full).statistic) if len(pa_full) > 1 else float("nan")
        ),
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


def _print_beat_line_quantile_table(metrics_per_variant: dict[str, dict]) -> None:
    """Print the real-data beat-line table with data-driven quantile buckets.

    Each variant has its own cutoffs (computed from its own |pred|
    distribution), so the bucket-row labels live in the variant column —
    we render one row per logical bucket (bot/mid/top/extreme) and the
    label stays identical, but the cutoff values appear in the cells.
    """
    rows = [
        ("pairs scored (full)", "n_pairs", "{:>22.0f}"),
        ("pairs scored (no self)", "n_pairs_no_self", "{:>22.0f}"),
        ("MAE vs beat-line corr (lower better)", "mae_vs_beat_line", "{:>22.4f}"),
        ("Spearman rho pred vs actual (higher better)", "rho_pred_vs_actual", "{:>22.4f}"),
        ("sign agreement (higher better)", "sign_agreement", "{:>22.4f}"),
        ("n sign-comparable pairs", "n_sign_pairs", "{:>22.0f}"),
    ]
    names = list(metrics_per_variant.keys())
    click.echo("")
    click.echo("      " + f"{'metric':<46}" + "  " + "  ".join(f"{n:>22}" for n in names))
    click.echo("      " + "-" * 46 + "  " + "  ".join("-" * 22 for _ in names))
    for label, key, fmt in rows:
        cells = []
        for n in names:
            v = metrics_per_variant[n].get(key, float("nan"))
            cells.append(fmt.format(v))
        click.echo(f"      {label:<46}  " + "  ".join(cells))

    # Per-variant cutoffs row — buckets are quantile-defined per variant,
    # so we print the actual cutoff values used.
    click.echo("")
    click.echo("      |pred| cutoffs per variant (q25 / q75 / extreme):")
    for n in names:
        m = metrics_per_variant[n]
        click.echo(
            f"        {n:<14} q25={m['q25']:.4f}  q75={m['q75']:.4f}  "
            f"extreme={m['extreme_cutoff']:.4f}  "
            f"(=max of q95_no_self={m['q95_no_self']:.4f}, q75_full={m['q75_full']:.4f})"
        )

    click.echo("")
    click.echo(
        "      direction-matched realized beat-line corr by |pred| quantile bucket"
    )
    click.echo(
        "      buckets 1-3 exclude same-player pairs; bucket 4 includes them"
    )
    bucket_headers = [
        "bot 25% (no self)",
        "mid 25-75% (no self)",
        "top 25% (no self)",
        "extreme (incl self)",
    ]
    click.echo(
        "      " + f"{'bucket':<22}" + "  " + "  ".join(f"{n:>18}" for n in names)
    )
    click.echo("      " + "-" * 22 + "  " + "  ".join("-" * 18 for _ in names))
    for i, header in enumerate(bucket_headers):
        cells = []
        for n in names:
            mean = metrics_per_variant[n]["bucket_means"][i]
            count = metrics_per_variant[n]["bucket_counts"][i]
            cells.append(f"{mean:>+8.4f} (n={count:>5d})")
        click.echo(f"      {header:<22}  " + "  ".join(cells))


def run_beat_line(seed: int, n_games: int) -> None:
    """Score each method against the empirical beat-line correlation.

    We reuse the synthetic gamelog generator from experiment 1 — same
    seed, same data — so the only thing that changes is the ground-truth
    target. Now we measure how well each method predicts the realized
    pairwise beat-line correlation, which is what the parlay EV
    multiplication consumes downstream.
    """
    rng = np.random.default_rng(seed)
    click.echo("[2/3] Beat-line conditional-probability experiment")
    click.echo(f"      seed={seed} n_games={n_games}")
    click.echo("      ground truth = empirical Pearson corr of beat-line booleans per team")

    gamelog, _truth_latent = _simulate_gamelog(DEFAULT_N_TEAMS, n_games, rng)
    stat_cols = [c for c in gamelog.columns if c.startswith("P")]

    raw_team_matrix = _gamelog_to_team_matrix(gamelog, stat_cols)
    residualized_log = _residualize_gamelog(gamelog, "player", "DATE", stat_cols)
    res_team_matrix = _gamelog_to_team_matrix(residualized_log, stat_cols)
    beat_line_matrix = _beat_line_team_matrix(gamelog, stat_cols)

    beat_line_truth = _empirical_beat_line_corr(beat_line_matrix)

    variants = [
        ("OLD", build_corr_old(raw_team_matrix)),
        ("PAIRWISE_ONLY", build_corr_pairwise_only(raw_team_matrix)),
        ("RESIDUAL_ONLY", build_corr_residual_only(res_team_matrix)),
        ("NEW", build_corr_new(res_team_matrix)),
    ]
    metrics_per_variant: dict[str, dict] = {}
    for name, stacked in variants:
        per_team = _stacked_to_per_team(stacked)
        metrics_per_variant[name] = _score_against_beat_line(per_team, beat_line_truth)

    _print_beat_line_table(metrics_per_variant)


def _print_beat_line_table(metrics_per_variant: dict[str, dict]) -> None:
    rows = [
        ("pairs scored", "n_pairs", "{:>14.0f}"),
        ("MAE vs beat-line corr (lower better)", "mae_vs_beat_line", "{:>14.4f}"),
        ("Spearman rho pred vs actual (higher better)", "rho_pred_vs_actual", "{:>14.4f}"),
        ("sign agreement (higher better)", "sign_agreement", "{:>14.4f}"),
        ("n sign-comparable pairs", "n_sign_pairs", "{:>14.0f}"),
    ]
    names = list(metrics_per_variant.keys())
    click.echo("")
    click.echo("      " + f"{'metric':<46}" + "  " + "  ".join(f"{n:>14}" for n in names))
    click.echo("      " + "-" * 46 + "  " + "  ".join("-" * 14 for _ in names))
    for label, key, fmt in rows:
        cells = []
        for n in names:
            v = metrics_per_variant[n].get(key, float("nan"))
            cells.append(fmt.format(v))
        click.echo(f"      {label:<46}  " + "  ".join(cells))

    # Bucketed realized lift: shows whether higher predicted correlation
    # actually corresponds to higher realized beat-line co-occurrence.
    click.echo("")
    click.echo("      direction-matched realized beat-line corr by predicted-magnitude bucket")
    click.echo("      (positive = pred direction agrees with realized direction; higher = better)")
    bucket_labels = metrics_per_variant[names[0]]["bucket_labels"]
    header = "      " + f"{'bucket':<14}" + "  " + "  ".join(f"{n:>14}" for n in names)
    click.echo(header)
    click.echo("      " + "-" * 14 + "  " + "  ".join("-" * 14 for _ in names))
    for i, label in enumerate(bucket_labels):
        cells = []
        for n in names:
            mean = metrics_per_variant[n]["bucket_means"][i]
            count = metrics_per_variant[n]["bucket_counts"][i]
            cells.append(f"{mean:>+8.4f} (n={count:>3d})")
        click.echo(f"      {label:<14}  " + "  ".join(cells))


# ---------------------------------------------------------------------------
# Experiment 3 — real gamelog holdout
# ---------------------------------------------------------------------------


def _try_load_stats(league: str):
    """Try to import + load a Stats subclass for one league. None on failure."""
    try:
        from sportstradamus import stats as stats_pkg
    except Exception as exc:
        click.echo(f"      could not import sportstradamus.stats: {exc}")
        return None
    cls_name = f"Stats{league}"
    cls = getattr(stats_pkg, cls_name, None)
    if cls is None:
        click.echo(f"      no Stats class named {cls_name} — skipping")
        return None
    try:
        inst = cls()
        inst.load()
    except Exception as exc:
        click.echo(f"      Stats{league}().load() failed: {exc}")
        return None
    if not hasattr(inst, "gamelog") or inst.gamelog is None or len(inst.gamelog) == 0:
        click.echo(f"      Stats{league} loaded but gamelog is empty — skipping")
        return None
    return inst


def _empirical_corr(matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Plain Spearman correlation per team — the holdout "ground truth"."""
    matrix = matrix.drop(columns="DATE", errors="ignore")
    out: dict[str, pd.DataFrame] = {}
    for team in matrix["TEAM"].unique():
        team_matrix = matrix.loc[matrix["TEAM"] == team].drop(columns="TEAM")
        team_matrix = team_matrix.loc[:, team_matrix.notna().any(axis=0)]
        if team_matrix.shape[1] < 2 or len(team_matrix) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[team] = cast(pd.DataFrame, team_matrix.corr(method="spearman"))
    return out


def _holdout_metrics(
    train_per_team: dict[str, pd.DataFrame],
    test_per_team: dict[str, pd.DataFrame],
    train_overlap: dict[str, pd.DataFrame],
) -> dict[str, float]:
    pred: list[float] = []
    actual: list[float] = []
    low_overlap_pred: list[float] = []
    low_overlap_actual: list[float] = []
    sign_match = 0
    sign_total = 0

    for team, test_m in test_per_team.items():
        train_m = train_per_team.get(team)
        train_o = train_overlap.get(team)
        cols = list(test_m.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                t = float(test_m.loc[a, b])
                if pd.isna(t):
                    continue
                if train_m is not None and a in train_m.index and b in train_m.columns:
                    p = float(train_m.loc[a, b])
                else:
                    p = 0.0
                pred.append(p)
                actual.append(t)
                if abs(p) >= SIGN_AGREEMENT_MAG_FLOOR and abs(t) >= SIGN_AGREEMENT_MAG_FLOOR:
                    sign_total += 1
                    if np.sign(p) == np.sign(t):
                        sign_match += 1
                if (
                    train_o is not None
                    and a in train_o.index
                    and b in train_o.columns
                    and train_o.loc[a, b] < MIN_OVERLAP_FOR_FULL_WEIGHT
                ):
                    low_overlap_pred.append(p)
                    low_overlap_actual.append(t)

    pa = np.array(pred)
    aa = np.array(actual)
    return {
        "n_pairs": float(len(pa)),
        "mae": float(np.mean(np.abs(pa - aa))) if len(pa) else float("nan"),
        "spearman_rho": (
            float(spearmanr(pa, aa).statistic) if len(pa) > 1 else float("nan")
        ),
        "sign_agreement": (sign_match / sign_total) if sign_total else float("nan"),
        "n_sign_pairs": float(sign_total),
        "low_overlap_mae": (
            float(np.mean(np.abs(np.array(low_overlap_pred) - np.array(low_overlap_actual))))
            if low_overlap_pred
            else float("nan")
        ),
        "n_low_overlap": float(len(low_overlap_pred)),
    }


def run_real(league: str) -> bool:
    click.echo(f"[3/3] Real-data holdout for league={league}")
    stats = _try_load_stats(league)
    if stats is None:
        return False

    if not hasattr(_correlate, "_build_team_game_records"):
        click.echo(
            "      correlate module loaded via file-stub fallback — production "
            "_build_team_game_records is not available in this mode. Re-run in "
            "an env where ``import sportstradamus.training.correlate`` works."
        )
        return False
    _TRACKED_STATS = _correlate._TRACKED_STATS
    if league not in _TRACKED_STATS:
        click.echo(f"      league {league} not in _TRACKED_STATS — skipping")
        return False

    # Build three per-team-per-game matrices over the same (game, team) row
    # set, differing only in per-cell transform:
    #   res_matrix       — residualized stat values (production "new" input)
    #   raw_matrix       — raw stat values (production "old" input)
    #   beat_line_matrix — 1/0/NaN beat-line booleans against rolling line
    #                      (ground truth for the conditional-probability test)
    try:
        earliest = pd.to_datetime(
            stats.gamelog[stats.log_strings["date"]]
        ).min().to_pydatetime()
        res_records = _correlate._build_team_game_records(league, stats, earliest)
        res_matrix = pd.DataFrame(pd.json_normalize(res_records))
    except Exception as exc:
        click.echo(
            f"      _build_team_game_records (residualized) raised "
            f"{type(exc).__name__}: {exc} — skipping"
        )
        return False
    if res_matrix.empty or "DATE" not in res_matrix.columns:
        click.echo("      _build_team_game_records returned no usable rows — skipping")
        return False

    try:
        raw_records = _build_team_game_records_no_residual(league, stats, earliest)
        raw_matrix = pd.DataFrame(pd.json_normalize(raw_records))
        beat_line_records = _build_team_game_records_beat_line(league, stats, earliest)
        beat_line_matrix = pd.DataFrame(pd.json_normalize(beat_line_records))
    except Exception as exc:
        click.echo(
            f"      auxiliary _build_team_game_records pivot raised "
            f"{type(exc).__name__}: {exc} — skipping"
        )
        return False

    # Time-based 70/30 split on the residualized matrix; align the others
    # to the same date cutoff.
    res_matrix = res_matrix.sort_values("DATE").reset_index(drop=True)
    raw_matrix = raw_matrix.sort_values("DATE").reset_index(drop=True)
    beat_line_matrix = beat_line_matrix.sort_values("DATE").reset_index(drop=True)
    n = len(res_matrix)
    cutoff_date = res_matrix.loc[int(n * TRAIN_FRACTION), "DATE"]
    click.echo(f"      train/test cutoff = {cutoff_date} ({int(n*TRAIN_FRACTION)}/{n} rows)")

    res_train = res_matrix.loc[res_matrix["DATE"] <= cutoff_date]
    raw_train = raw_matrix.loc[raw_matrix["DATE"] <= cutoff_date]
    raw_test = raw_matrix.loc[raw_matrix["DATE"] > cutoff_date]
    beat_test = beat_line_matrix.loc[beat_line_matrix["DATE"] > cutoff_date]

    if (raw_matrix["DATE"] > cutoff_date).sum() < 50:
        click.echo("      fewer than 50 test rows — skipping")
        return False

    # Same four variants as the synthetic experiments — same input
    # convention: OLD/PAIRWISE_ONLY consume the raw matrix, RESIDUAL_ONLY/NEW
    # consume the residualized matrix.
    variants = [
        ("OLD", build_corr_old(raw_train)),
        ("PAIRWISE_ONLY", build_corr_pairwise_only(raw_train)),
        ("RESIDUAL_ONLY", build_corr_residual_only(res_train)),
        ("NEW", build_corr_new(res_train)),
    ]
    overlap_raw = _per_team_overlap(raw_train)
    overlap_res = _per_team_overlap(res_train)
    overlap_per_variant = {
        "OLD": overlap_raw,
        "PAIRWISE_ONLY": overlap_raw,
        "RESIDUAL_ONLY": overlap_res,
        "NEW": overlap_res,
    }

    test_spearman = _empirical_corr(raw_test)
    test_beat_line = _empirical_beat_line_corr(beat_test)

    holdout_metrics: dict[str, dict[str, float]] = {}
    beat_line_metrics: dict[str, dict] = {}
    for name, stacked in variants:
        per_team = _stacked_to_per_team(stacked)
        holdout_metrics[name] = _holdout_metrics(
            per_team, test_spearman, overlap_per_variant[name]
        )
        beat_line_metrics[name] = _score_against_beat_line_quantile(
            per_team, test_beat_line, _same_player_real
        )

    click.echo("")
    click.echo("      [3a] Train-vs-test Spearman recovery (analogous to experiment 1)")
    _print_holdout_ablation_table(holdout_metrics)
    click.echo("")
    click.echo("      [3b] Beat-line conditional probability (analogous to experiment 2)")
    _print_beat_line_quantile_table(beat_line_metrics)
    return True


def _print_holdout_ablation_table(metrics_per_variant: dict[str, dict[str, float]]) -> None:
    """Multi-variant version of _print_holdout_table for the real-data run."""
    rows = [
        ("pairs scored", "n_pairs", "{:>14.0f}"),
        ("holdout MAE (lower better)", "mae", "{:>14.4f}"),
        ("Spearman rho train vs test (higher better)", "spearman_rho", "{:>14.4f}"),
        ("sign agreement (higher better)", "sign_agreement", "{:>14.4f}"),
        ("n sign-comparable pairs", "n_sign_pairs", "{:>14.0f}"),
        ("low-overlap MAE (lower better)", "low_overlap_mae", "{:>14.4f}"),
        ("n low overlap pairs", "n_low_overlap", "{:>14.0f}"),
    ]
    names = list(metrics_per_variant.keys())
    click.echo("")
    click.echo("      " + f"{'metric':<46}" + "  " + "  ".join(f"{n:>14}" for n in names))
    click.echo("      " + "-" * 46 + "  " + "  ".join("-" * 14 for _ in names))
    for label, key, fmt in rows:
        cells = []
        for n in names:
            v = metrics_per_variant[n].get(key, float("nan"))
            cells.append(fmt.format(v))
        click.echo(f"      {label:<46}  " + "  ".join(cells))


def _build_team_game_records_no_residual(league: str, stats, earliest_date) -> list[dict]:
    """Same per-team-per-game pivot as production but without residualization.

    Implemented by temporarily monkey-patching _residualize_gamelog to a
    no-op so we don't fork the pivot logic. This keeps the OLD vs NEW
    comparison using the *same* (game, team) row set — only the per-cell
    transformation differs.
    """
    cmod = _correlate
    orig = cmod._residualize_gamelog
    cmod._residualize_gamelog = lambda gamelog, *a, **k: gamelog.copy()
    try:
        return cmod._build_team_game_records(league, stats, earliest_date)
    finally:
        cmod._residualize_gamelog = orig


def _build_team_game_records_beat_line(league: str, stats, earliest_date) -> list[dict]:
    """Same pivot as production, but cells are beat-line booleans.

    We patch ``_residualize_gamelog`` so each tracked stat becomes 1.0 if the
    raw value beat the rolling-mean line (residual > 0), 0.0 otherwise, and
    NaN where rolling history was insufficient (line undefined). The
    surrounding pivot logic in ``_build_team_game_records`` is unchanged, so
    the resulting matrix has identical (gameId, team) rows and column names
    to the residualized/raw variants — only the per-cell semantics differ.

    The Pearson correlation of these per-team boolean columns IS what the
    parlay EV multiplication consumes downstream — see the comment block at
    the top of experiment 2.
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--league", type=click.Choice(["NFL", "NBA", "WNBA", "NHL", "MLB"]), default="NBA")
@click.option("--seed", type=int, default=DEFAULT_SEED)
@click.option("--n-games", type=int, default=DEFAULT_N_GAMES)
@click.option("--skip-real", is_flag=True, help="Skip the real-data holdout experiment.")
@click.option("--skip-synthetic", is_flag=True, help="Skip both synthetic experiments.")
@click.option("--skip-beat-line", is_flag=True, help="Skip the beat-line conditional experiment.")
def main(
    league: str,
    seed: int,
    n_games: int,
    skip_real: bool,
    skip_synthetic: bool,
    skip_beat_line: bool,
) -> None:
    click.echo("Correlation methodology comparison: OLD vs NEW")
    click.echo("=" * 72)
    if not skip_synthetic:
        run_synthetic(seed=seed, n_games=n_games)
    if not skip_synthetic and not skip_beat_line:
        click.echo("")
        run_beat_line(seed=seed, n_games=n_games)
    if not skip_real:
        click.echo("")
        ran = run_real(league=league)
        if not ran:
            click.echo("      (real-data experiment skipped)")
    click.echo("")
    click.echo("Done.")


if __name__ == "__main__":
    main()
