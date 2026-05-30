#!/usr/bin/env python3
"""Compute ICC_1 for every SkewNormal (league, market) cell and route Stage A2.

Task A1 (Track A diagnostic gate). Read-only with respect to model pickles,
``model_prob``, and the training pipeline: this script only reads cached
training-data parquets and decomposes the marginal target's variance into a
between-player and a within-player component.

ICC_1 measures how much of a market's variance is stable player identity vs.
game-to-game noise, via a two-level moment decomposition over (player, season)
groups:

    sigma2_between = Var(player-season means)
    sigma2_within  = mean(within-(player-season) variance)
    ICC            = sigma2_between / (sigma2_between + sigma2_within)

High ICC -> a per-player level (empirical-Bayes centering) is the right lever
(Stage A2 T5); low ICC -> the signal is in the distribution's tail, not its
location (Stage A2 T3/T7). The per-cell empirical-Bayes shrinkage constant
``eb_K = sigma2_within / sigma2_between`` is surfaced next to production's
current single ``EB_SHRINKAGE_K`` so the "K=10 is wrong for NFL" suspicion can
be confirmed or refuted.

Output: one ``data/icc/{LEAGUE}_icc.parquet`` per league (repo-root ``data/``,
not the package data dir), one row per (league, market).

Usage
-----
    poetry run icc-diagnostics
    poetry run icc-diagnostics --league NFL --min-nonzero-fraction 0.4
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import logging
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers.config import stat_dist
from sportstradamus.training.baselines import EB_SHRINKAGE_K
from sportstradamus.training.markets import ALL_MARKETS

logger = logging.getLogger(__name__)

# Leagues Track A covers; MLB/NHL have no SkewNormal markets so they are out of scope.
_COVERED_LEAGUES = ("NBA", "WNBA", "NFL")
# The exact stat_dist label that marks a cell as a SkewNormal candidate.
_SKEWNORMAL_LABEL = "SkewNormal"
# Guard from the plan: the dynamic cell derivation must yield exactly this many.
_EXPECTED_CELL_COUNT = 36
# Columns read from each cached training-data parquet.
_TARGET_COLUMN = "Result"
_PLAYER_COLUMN = "Player"
_DATE_COLUMN = "Date"

# Season-rollover rule replicated from stats/base.py:527-528 (the canonical
# source). Calendar-spanning leagues (NBA/NFL/NHL) roll the season over at
# _SEASON_BOUNDARY_MONTH; WNBA/MLB seasons sit inside one calendar year. The rule
# lives inside a Stats method and cannot be imported without editing an
# inference-path file, which A1 must leave frozen -- so it is replicated here.
_SEASON_BOUNDARY_MONTH = 8
_CALENDAR_YEAR_LEAGUES = frozenset({"WNBA", "MLB"})

# Minimum games for a player-season to have a defined within-variance.
_MIN_GROUP_GAMES = 2
# Participation floor: a player-season counts only if a majority of its games
# produced nonzero output. Drops structural-position zeros (e.g. non-QBs in an
# NFL passing-yards frame) without a hand-maintained position map (STYLE_GUIDE §9).
_MIN_NONZERO_GAME_FRACTION = 0.5
# |Fisher skewness| above which the raw target is too right-skewed for a stable
# variance decomposition and a transform is applied before decomposing.
_SKEW_TRANSFORM_THRESHOLD = 2.0

# Pre-registered routing cutoffs (set before looking at the numbers).
# ICC >= floor -> a per-player level can work (eb_centering).
_ICC_EB_FLOOR = 0.5
# ICC <= ceiling -> needs a distributional-tail head (tail_extension).
_ICC_TAIL_CEILING = 0.3

ROUTING_CHOICES: tuple[str, str, str] = ("eb_centering", "tail_extension", "ambiguous")
TRANSFORM_CHOICES: tuple[str, str, str] = ("raw", "log1p", "rank")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "league",
    "market",
    "n_players",
    "n_player_seasons",
    "n_player_seasons_excluded",
    "n_obs",
    "transform",
    "target_skewness",
    "sigma2_between",
    "sigma2_within",
    "icc",
    "eb_K",
    "eb_K_current",
    "routing_recommendation",
)


class IccDecomposition(NamedTuple):
    """The two-level variance decomposition of one (league, market) cell.

    All values are in the (possibly transformed) target's units-squared, except
    ``icc`` (dimensionless in [0, 1]) and ``eb_k`` (dimensionless, may be inf
    when there is no between-player variance).
    """

    sigma2_between: float
    sigma2_within: float
    icc: float
    eb_k: float


class _ParticipatingGroups(NamedTuple):
    """Kept per-(player, season) raw target arrays plus the participation counts."""

    raw_groups: list[np.ndarray]
    n_players: int
    n_player_seasons: int
    n_player_seasons_excluded: int
    n_obs: int


def skewnormal_cells() -> list[tuple[str, str]]:
    """Return every (league, market) whose stat_dist label is exactly ``SkewNormal``.

    Derived dynamically from ``stat_dist.json`` and ``ALL_MARKETS`` so the set
    tracks config drift. Asserts the plan's guard count of 36.

    Returns:
        Sorted-by-config (league, market) pairs over NBA, WNBA, and NFL.
    """
    cells: list[tuple[str, str]] = []
    for league in _COVERED_LEAGUES:
        for market in ALL_MARKETS[league]:
            if stat_dist.get(league, {}).get(market) == _SKEWNORMAL_LABEL:
                cells.append((league, market))
    if len(cells) != _EXPECTED_CELL_COUNT:
        raise click.ClickException(
            f"Expected {_EXPECTED_CELL_COUNT} SkewNormal cells, found {len(cells)}: {cells}"
        )
    return cells


def _training_data_root() -> Path:
    """Return the cached training-data parquet directory inside the package."""
    return Path(str(pkg_resources.files(data) / "training_data"))


def _market_stem(market: str) -> str:
    """Map a market name to its parquet filename stem (spaces -> hyphens)."""
    return market.replace(" ", "-")


def _load_cell_frame(league: str, market: str) -> pd.DataFrame | None:
    """Load ``[Player, Date, Result]`` for one cell, or ``None`` if absent.

    Returns:
        A frame with non-null ``Player``/``Date``, datetime ``Date``, and finite
        ``Result``, or ``None`` when the parquet is missing (logged as a warning
        so the cell is skipped).
    """
    path = _training_data_root() / f"{league}_{_market_stem(market)}.parquet"
    if not path.is_file():
        logger.warning("Missing training parquet for %s %s: %s", league, market, path)
        return None
    frame = pd.read_parquet(
        path, engine="pyarrow", columns=[_PLAYER_COLUMN, _DATE_COLUMN, _TARGET_COLUMN]
    )
    frame = frame.dropna(subset=[_PLAYER_COLUMN, _DATE_COLUMN, _TARGET_COLUMN]).copy()
    frame[_DATE_COLUMN] = pd.to_datetime(frame[_DATE_COLUMN])
    return frame[np.isfinite(frame[_TARGET_COLUMN].to_numpy(dtype=float))]


def _season_of(dates: pd.Series, league: str) -> pd.Series:
    """Map game dates to integer season labels (stats/base.py:527-528 rule).

    WNBA/MLB seasons sit inside one calendar year; calendar-spanning leagues
    (NBA/NFL/NHL) roll the season over at :data:`_SEASON_BOUNDARY_MONTH`.
    """
    years = dates.dt.year
    if league in _CALENDAR_YEAR_LEAGUES:
        return years
    return years.where(dates.dt.month >= _SEASON_BOUNDARY_MONTH, years - 1)


def _choose_transform(y: np.ndarray) -> str:
    """Pick the variance-stabilizing transform deterministically from skewness.

    Escalates raw -> log1p -> rank: the raw target if it is not strongly
    right-skewed, else log1p if that pulls the skew under the threshold, else a
    rank transform (skew-free by construction). All decisions compare the Fisher
    skewness magnitude against :data:`_SKEW_TRANSFORM_THRESHOLD`.
    """
    if abs(float(stats.skew(y))) <= _SKEW_TRANSFORM_THRESHOLD:
        return "raw"
    if abs(float(stats.skew(np.log1p(np.clip(y, 0.0, None))))) <= _SKEW_TRANSFORM_THRESHOLD:
        return "log1p"
    return "rank"


def _apply_transform(y: np.ndarray, transform: str) -> np.ndarray:
    """Apply the chosen transform to a non-negative observed-stat array."""
    if transform == "log1p":
        return np.log1p(np.clip(y, 0.0, None))
    if transform == "rank":
        return stats.rankdata(y)
    return y


def _participating_groups(
    frame: pd.DataFrame, league: str, min_games: int, min_nonzero_frac: float
) -> _ParticipatingGroups:
    """Group by (player, season) and keep only meaningfully participating seasons.

    A player-season survives when it has at least ``min_games`` games **and** a
    nonzero-output fraction of at least ``min_nonzero_frac``. The second test
    drops structural-position zeros without naming positions.
    ``n_player_seasons_excluded`` counts the games-eligible seasons removed by
    the participation floor specifically.

    Args:
        frame: ``[Player, Date, Result]`` for one cell.
        league: League code (selects the season-boundary rule).
        min_games: Minimum games for a player-season to be games-eligible.
        min_nonzero_frac: Minimum fraction of nonzero-output games to participate.

    Returns:
        A :class:`_ParticipatingGroups` with the kept raw per-group arrays and counts.
    """
    season = _season_of(frame[_DATE_COLUMN], league)
    raw_groups: list[np.ndarray] = []
    kept_players: set[str] = set()
    games_eligible = 0
    n_obs = 0
    for (player, _season), group in frame.groupby([frame[_PLAYER_COLUMN], season]):
        y = group[_TARGET_COLUMN].to_numpy(dtype=float)
        if y.size < min_games:
            continue
        games_eligible += 1
        if float((y > 0).mean()) < min_nonzero_frac:
            continue
        raw_groups.append(y)
        kept_players.add(player)
        n_obs += y.size
    return _ParticipatingGroups(
        raw_groups=raw_groups,
        n_players=len(kept_players),
        n_player_seasons=len(raw_groups),
        n_player_seasons_excluded=games_eligible - len(raw_groups),
        n_obs=n_obs,
    )


def _decompose(group_arrays: list[np.ndarray]) -> IccDecomposition | None:
    """Two-level moment decomposition over already-transformed group arrays.

    ``sigma2_between`` is the sample variance of the per-group means;
    ``sigma2_within`` is the mean of the per-group sample variances (each group
    has >= 2 observations by construction). Returns ``None`` when fewer than two
    groups survive or the total variance is non-positive -- the degenerate guard.
    """
    if len(group_arrays) < 2:
        return None
    means = np.array([g.mean() for g in group_arrays], dtype=float)
    within_vars = np.array([g.var(ddof=1) for g in group_arrays], dtype=float)
    sigma2_between = float(means.var(ddof=1))
    sigma2_within = float(within_vars.mean())
    total = sigma2_between + sigma2_within
    if not np.isfinite(total) or total <= 0:
        return None
    icc = sigma2_between / total
    eb_k = sigma2_within / sigma2_between if sigma2_between > 0 else float("inf")
    return IccDecomposition(sigma2_between, sigma2_within, icc, eb_k)


def _route(row: dict) -> str:
    """Map a cell's ICC to a Stage-A2 routing recommendation (pure function).

    Pre-registered cutoffs: ICC >= :data:`_ICC_EB_FLOOR` -> ``eb_centering``;
    ICC <= :data:`_ICC_TAIL_CEILING` -> ``tail_extension``; the open band between
    them -> ``ambiguous`` (try both in A2, route on outcome).
    """
    icc = row["icc"]
    if icc >= _ICC_EB_FLOOR:
        return "eb_centering"
    if icc <= _ICC_TAIL_CEILING:
        return "tail_extension"
    return "ambiguous"


def _compute_cell_stats(
    league: str, market: str, frame: pd.DataFrame, min_games: int, min_nonzero_frac: float
) -> dict | None:
    """Compute the full diagnostics row for one (league, market) cell.

    Filters to participating player-seasons, picks the transform on the kept raw
    pooled target, decomposes the transformed target, and routes on ICC.

    Args:
        league: League code (e.g. ``"NBA"``).
        market: Market name as it appears in ``ALL_MARKETS``.
        frame: ``[Player, Date, Result]`` for one cell.
        min_games: Minimum games for a player-season to enter the decomposition.
        min_nonzero_frac: Participation floor on the nonzero-output game fraction.

    Returns:
        A dict keyed by :data:`REQUIRED_COLUMNS`, or ``None`` (skip + log) when no
        participating data or a degenerate decomposition leaves the ICC undefined.
    """
    groups = _participating_groups(frame, league, min_games, min_nonzero_frac)
    if not groups.raw_groups:
        logger.warning("No participating player-seasons for %s %s; skipping.", league, market)
        return None
    pooled = np.concatenate(groups.raw_groups)
    transform = _choose_transform(pooled)
    decomposition = _decompose([_apply_transform(g, transform) for g in groups.raw_groups])
    if decomposition is None:
        logger.warning("Degenerate variance decomposition for %s %s; skipping.", league, market)
        return None
    row = {
        "league": league,
        "market": market,
        "n_players": groups.n_players,
        "n_player_seasons": groups.n_player_seasons,
        "n_player_seasons_excluded": groups.n_player_seasons_excluded,
        "n_obs": groups.n_obs,
        "transform": transform,
        "target_skewness": float(stats.skew(pooled)),
        "sigma2_between": decomposition.sigma2_between,
        "sigma2_within": decomposition.sigma2_within,
        "icc": decomposition.icc,
        "eb_K": decomposition.eb_k,
        "eb_K_current": float(EB_SHRINKAGE_K),
    }
    row["routing_recommendation"] = _route(row)
    return row


def _write_league_parquet(rows: list[dict], league: str, out_dir: Path) -> Path:
    """Write one league's diagnostics rows to ``{out_dir}/{league}_icc.parquet``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    path = out_dir / f"{league}_icc.parquet"
    frame.to_parquet(path, engine="pyarrow", index=False)
    return path


@click.command()
@click.option(
    "--league",
    type=click.Choice(_COVERED_LEAGUES),
    default=None,
    help="Restrict to one league (default: all three SkewNormal leagues).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/icc"),
    show_default=True,
    help="Directory for the per-league ICC parquets.",
)
@click.option(
    "--min-games",
    default=_MIN_GROUP_GAMES,
    show_default=True,
    help="Minimum games for a player-season to enter the decomposition.",
)
@click.option(
    "--min-nonzero-fraction",
    default=_MIN_NONZERO_GAME_FRACTION,
    show_default=True,
    help="Participation floor: minimum fraction of a player-season's games with nonzero output.",
)
def main(league: str | None, out_dir: Path, min_games: int, min_nonzero_fraction: float) -> None:
    """Decompose each SkewNormal cell's variance into ICC and recommend a Stage-A2 route."""
    if min_games < _MIN_GROUP_GAMES:
        raise click.UsageError(f"--min-games must be at least {_MIN_GROUP_GAMES}.")
    if not 0.0 <= min_nonzero_fraction <= 1.0:
        raise click.UsageError("--min-nonzero-fraction must be in [0, 1].")
    cells = skewnormal_cells()
    if league is not None:
        cells = [(lg, market) for lg, market in cells if lg == league]
    if not cells:
        click.echo(f"No SkewNormal cells for league filter {league!r}.")
        return

    rows_by_league: dict[str, list[dict]] = {}
    for cell_league, market in tqdm(cells, desc="Scoring SkewNormal cells"):
        frame = _load_cell_frame(cell_league, market)
        if frame is None or frame.empty:
            continue
        row = _compute_cell_stats(cell_league, market, frame, min_games, min_nonzero_fraction)
        if row is None:
            continue
        rows_by_league.setdefault(cell_league, []).append(row)

    out_dir = Path(out_dir)
    for cell_league, rows in rows_by_league.items():
        path = _write_league_parquet(rows, cell_league, out_dir)
        click.echo(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
