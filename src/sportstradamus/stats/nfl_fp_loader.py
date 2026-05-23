"""FantasyPoints (FP) season-aggregate CSV loader for the NFL comp profile.

Phase 1A of the PFF -> FantasyPoints migration. ``build_comp_profile`` in
``stats/nfl.py`` calls :func:`load_one_year` per season in its lookback
window, then ``combine_first``-merges across years. Phase 1G's
``comp_feature_stability.py`` will import the same helper, so the loader
lives in its own module to keep ``stats/nfl.py`` from sliding back toward
its pre-refactor 7k-line size (CLAUDE.md "No new monoliths").

The FP per-player CSVs that this module reads live in
``data/player_data/NFL/{year}/`` (post-Phase-1K canonical location). The
legacy PFF CSVs that used to share that path have been archived under
``data/pff_nfl_archive/`` at repo root.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from importlib import resources as pkg_resources

import nfl_data_py as nfl
import numpy as np
import pandas as pd

from sportstradamus import data
from sportstradamus.helpers.text import remove_accents
from sportstradamus.spiderLogger import logger
from sportstradamus.stats.nfl_pbp_agg import compute_pbp_season_aggregates

# Per-file column-name prefix scheme. Many FP files publish the same short
# headers (``RTE``, ``YPRR``, ``FP/G``); prefixing on read keeps them
# disambiguated when years/files are joined. See plan
# ``before-we-move-on-squishy-squid.md`` §1A "Per-stat-type prefix scheme".
_PER_PLAYER_FILE_PREFIXES: dict[str, str] = {
    "passingBasicExport.csv": "pass_basic_",
    "passingAdvancedExport.csv": "pass_adv_",
    "passingDepthExport.csv": "pass_depth_",
    "receivingBasicExport.csv": "rec_basic_",
    "receivingAdvancedExport.csv": "rec_adv_",
    "receivingManVsZoneExport.csv": "rec_mz_",
    "receivingRoutesRunExport.csv": "rec_routes_",
    "receivingSeparationByAlignmentExport.csv": "rec_sep_align_",
    "receivingSeparationByBreaksExport.csv": "rec_sep_break_",
    "receivingSeparationByCoverageExport.csv": "rec_sep_cov_",
    "receivingSeparationByRoutesExport.csv": "rec_sep_route_",
    "rushingBasicExport.csv": "rush_basic_",
    "rushingAdvancedExport.csv": "rush_adv_",
    "rushingBellCowExport.csv": "rush_bellcow_",
    "offenseSnapsExport.csv": "off_snaps_",
    "efficiencyExport.csv": "eff_",
}

# qbCoverageMatchupExport is keyed by (QB, opponent) so it must be reduced
# to a per-player mean over numeric columns before joining. Tracked
# separately from the simple per-player files above.
_QB_COVERAGE_FILE = "qbCoverageMatchupExport.csv"
_QB_COVERAGE_PREFIX = "qb_cov_"

# coverageMatrixExportOffense is keyed by team and broadcast to each
# player on their team's roster. The other team-grain files
# (defense, runPassReport, lineMatchups) are deliberately skipped at
# Phase 1 — Phase 2 handles them in the team-profile path.
_TEAM_COVERAGE_FILE = "coverageMatrixExportOffense.csv"
_TEAM_COVERAGE_PREFIX = "tm_cov_off_"

# FP CSV "key" columns that must NOT be prefixed -- they're either join
# keys or filterable identifiers shared across files.
_FP_KEY_COLUMNS: frozenset[str] = frozenset(
    {"Name", "Team", "POS", "G", "Season", "Rank"}
)

# coverageMatrixOffense uses full team names (``Cleveland Browns``);
# every other FP per-player file uses an abbreviation that does NOT match
# the ``nfl_data_py`` / Sportstradamus convention (``CLV`` vs ``CLE``,
# ``ARZ`` vs ``ARI``, ``BLT`` vs ``BAL``, ``HST`` vs ``HOU``). This map
# bridges the team-grain coverage file to the per-player FP abbreviation
# used everywhere else, so the broadcast join produces non-empty rows.
_TEAM_NAME_TO_FP_ABBR: dict[str, str] = {
    "Arizona Cardinals": "ARZ",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BLT",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLV",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HST",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

# Skill positions that participate in the comp space. ``HB``/``FB`` are
# remapped to ``RB`` in :func:`load_one_year`; the loader filters on the
# remapped set.
_COMP_POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE"})

# FP CSVs encode running backs with the granular role labels ``HB`` and
# ``FB`` while every other surface (gamelog, market config, comp filter)
# bucket them as ``RB``. :func:`load_one_year` collapses both to ``RB``
# before the position filter so per-year frames carry the canonical label.
_RB_ROLE_ALIASES: frozenset[str] = frozenset({"HB", "FB"})

# Per-player 3-year rolling mean dampens variance for moderate-stability
# features; raises Y/Y r above the 0.55 keep threshold (Phase 1G gate).
TRANSFORM_FEATURES: frozenset[str] = frozenset(
    {
        "block_pct_proxy",
        "pass_adv_aDOT",
        "pbp_intended_air_yards",
        "rec_adv_DESIGN_pct",
        "rec_adv_TPRR",
        "rec_adv_YPRR",
        "win_rate_overall",
    }
)

# Default lookback for the rolling mean. Three seasons matches the FP
# data window (2023+) and the variance-reduction target documented in
# the Phase 1G research brief.
_ROLLING_WINDOW: int = 3


def load_one_year(year: int) -> pd.DataFrame:
    """Load all FP per-player files + PBP aggregates for a single season.

    Returns a DataFrame keyed by the accent-stripped player name with one
    row per player. Each FP file's non-key columns are prefixed per
    :data:`_PER_PLAYER_FILE_PREFIXES` to avoid collisions across files
    (e.g., both ``rec_adv_YPRR`` and ``rec_mz_YPRR`` survive the merge).
    PBP/NGS-derived columns from
    :func:`sportstradamus.stats.nfl_pbp_agg.compute_pbp_season_aggregates`
    are joined on ``gsis_id`` -> player name via :func:`nfl.import_ids`.

    Missing FP files are skipped silently (file inventory varies by year);
    a player who appears in some files but not others simply carries NaNs
    for the missing columns. The caller's downstream ``fillna(0)`` chain
    in :meth:`StatsNFL.update_player_comps` absorbs them.

    Args:
        year: NFL season (e.g., 2024).

    Returns:
        DataFrame indexed by accent-stripped player name, with columns
        that include ``position``, ``player_game_count`` (from the FP
        ``G`` column), and the prefixed FP metrics. Empty DataFrame if
        no FP files exist for ``year``.
    """
    fp_dir = _fp_dir(year)
    if fp_dir is None:
        logger.warning("no FP data directory found for %s", year)
        return pd.DataFrame()

    per_year = pd.DataFrame()

    for filename, prefix in _PER_PLAYER_FILE_PREFIXES.items():
        df = _load_player_file(fp_dir, filename, prefix)
        if df is None:
            continue
        per_year = per_year.combine_first(df) if not per_year.empty else df

    qb_cov = _load_qb_coverage_file(fp_dir)
    if qb_cov is not None and not per_year.empty:
        per_year = per_year.combine_first(qb_cov)

    if not per_year.empty:
        team_cov = _load_team_coverage_file(fp_dir)
        if team_cov is not None and "Team" in per_year.columns:
            per_year = _broadcast_team_columns(per_year, team_cov)

    pbp = _load_pbp_named_by_player(year)
    if pbp is not None and not per_year.empty:
        per_year = per_year.combine_first(pbp)

    if per_year.empty:
        return per_year

    # Collapse HB/FB -> RB before the position filter and the canonical
    # rename so downstream consumers (derive_comp_metrics, the Phase 1G
    # stability diagnostic, build_comp_profile) see one taxonomy.
    if "POS" in per_year.columns:
        per_year.loc[per_year["POS"].isin(_RB_ROLE_ALIASES), "POS"] = "RB"
        per_year = per_year.loc[per_year["POS"].isin(_COMP_POSITIONS)]

    per_year = per_year.rename(columns={"POS": "position", "G": "player_game_count"})
    per_year = derive_comp_metrics(per_year)
    per_year = _join_nfl_players(per_year)
    return per_year


def derive_comp_metrics(profile: pd.DataFrame) -> pd.DataFrame:
    """Add FP-native rate / per-game derived comp features to a per-year profile."""
    # ``position`` / ``player_game_count`` are the canonical post-rename
    # names (load_one_year renames before calling us). The comp-filter
    # aliases ``dropbacks`` / ``attempts`` / ``routes`` are added here too
    # so they exist on every per-year frame the Phase 1G diagnostic and
    # build_comp_profile see.
    if "position" not in profile.columns:
        return profile

    aliases: dict[str, pd.Series] = {}
    if "pass_adv_DB" in profile.columns:
        aliases["dropbacks"] = profile["pass_adv_DB"]
    if "rush_basic_ATT" in profile.columns:
        aliases["attempts"] = profile["rush_basic_ATT"]
    if "rec_adv_RTE" in profile.columns:
        aliases["routes"] = profile["rec_adv_RTE"]
    if aliases:
        profile = profile.assign(**aliases)

    qb_mask = profile["position"] == "QB"
    non_qb_mask = profile["position"] != "QB"
    rb_mask = profile["position"] == "RB"
    games = profile.get("player_game_count")

    # In passingBasic, the second YDS column (auto-suffixed to ``YDS.1``)
    # is total QB rushing yards; the third (``YDS.2``) is scramble yards.
    # designed_yards_per_game uses the total -- the EXP-run share from
    # rush_adv covers the designed/scramble split.
    dropbacks = profile.get("pass_adv_DB")
    scrambles = profile.get("pass_adv_SCRM")
    qb_rush_yards = profile.get("pass_basic_YDS.1")
    rush_att = profile.get("rush_basic_ATT")
    receptions = profile.get("rec_basic_REC")
    ctgt_rate = profile.get("rec_adv_CTGT_pct")
    # receivingSeparationByCoverage publishes 8 unlabeled coverage
    # sub-strats (4 full RTE/SEP/YPRR/TPRR/WIN, 4 simpler RTE/SEP/WIN).
    # The 5th sub-block (auto-suffixed ``.4``) is the first of the simpler
    # set -- per Phase-1A spec we use it as the "deep contested" proxy.
    deep_win = profile.get("rec_sep_cov_WIN_RATE.4")
    # In receivingManVsZone header the sub-strat ordering is
    # [Overall, Man, Zone, OneHi, TwoHi] -- man YPRR is ``YPRR.1``,
    # zone YPRR is ``YPRR.2``.
    overall_yprr = profile.get("rec_adv_YPRR")
    man_yprr = profile.get("rec_mz_YPRR.1")
    zone_yprr = profile.get("rec_mz_YPRR.2")
    sep_overall = profile.get("rec_sep_align_SEP_SCORE")
    win_overall = profile.get("rec_sep_align_WIN_RATE")
    # The route-level header in receivingSeparationByRoutes omits the
    # WIN_RATE column for some bins, so SEP SCORE is the only metric
    # guaranteed across every route. The per-route mean uses every
    # ``SEP SCORE`` column (canonicalized + auto-suffixed).
    route_sep_cols = [
        c for c in profile.columns if c.startswith("rec_sep_route_SEP_SCORE")
    ]
    qb_man = profile.get("qb_cov_QB_MAN_pct")
    tm_man = profile.get("tm_cov_off_MAN_pct")
    rush_success = profile.get("rush_adv_Success_pct")
    exp_run = profile.get("rush_adv_EXP_RUN_pct")
    bell_att = profile.get("rush_bellcow_ATT_pct")
    bell_tgt = profile.get("rush_bellcow_TGT_pct")

    derived: dict[str, pd.Series] = {}

    def _gated(name: str, mask: pd.Series, series: pd.Series) -> None:
        """Write ``series`` into ``derived[name]`` masked by position."""
        sized = pd.Series(np.nan, index=profile.index)
        sized.loc[mask] = series.loc[mask]
        derived[name] = sized

    if dropbacks is not None and games is not None:
        _gated("dropbacks_per_game", qb_mask, dropbacks / games)
    if dropbacks is not None and scrambles is not None:
        _gated("scrambles_per_dropback", qb_mask, scrambles / dropbacks)
    if qb_rush_yards is not None and games is not None:
        _gated("designed_yards_per_game", qb_mask, qb_rush_yards / games)
    if rush_att is not None and receptions is not None and games is not None:
        _gated(
            "total_touches_per_game",
            rb_mask,
            (rush_att.fillna(0) + receptions.fillna(0)) / games,
        )

    # rec_adv carries CTGT_pct directly (contested target rate); the deep
    # variant comes from the deep-coverage sub-strat in
    # receivingSeparationByCoverage.
    if ctgt_rate is not None:
        _gated("contested_target_rate", non_qb_mask, ctgt_rate)
    if deep_win is not None:
        _gated("deep_contested_target_rate", non_qb_mask, deep_win)

    if overall_yprr is not None and man_yprr is not None:
        _gated("man_yprr_diff", non_qb_mask, man_yprr - overall_yprr)
    if overall_yprr is not None and zone_yprr is not None:
        _gated("zone_yprr_diff", non_qb_mask, zone_yprr - overall_yprr)

    # FP-exclusive aggregates from §1D. ``rec_sep_align_SEP_SCORE`` and
    # ``rec_sep_align_WIN_RATE`` (no suffix == first / "Overall" stratum)
    # are the receiver's season-wide separation and win-rate.
    if sep_overall is not None:
        _gated("sep_overall", non_qb_mask, sep_overall)
    if win_overall is not None:
        _gated("win_rate_overall", non_qb_mask, win_overall)
    if route_sep_cols:
        _gated(
            "sep_routes_mean",
            non_qb_mask,
            profile[route_sep_cols].mean(axis=1),
        )

    # Coverage scheme share: QBs see their own MAN%; receivers and backs
    # see the opposing-coverage rate via the team-broadcast file
    # ``coverageMatrixExportOffense`` (proxy for "what coverage does this
    # player's offense face?").
    if qb_man is not None or tm_man is not None:
        cov_share = pd.Series(np.nan, index=profile.index)
        if qb_man is not None:
            cov_share.loc[qb_mask] = qb_man.loc[qb_mask]
        if tm_man is not None:
            cov_share.loc[non_qb_mask] = tm_man.loc[non_qb_mask]
        derived["cov_man_share"] = cov_share

    # First ``Success %`` column (total / overall bucket) for RB success
    # rate -- not the PRO / GAP sub-buckets.
    if rush_success is not None:
        _gated("success_rate", rb_mask, rush_success)
    if exp_run is not None:
        _gated("exp_run_share", rb_mask, exp_run)

    # Bellcow score == ground + air volume relative to the team. The sum
    # produces a single 0-200 ranking number; downstream z-scoring
    # normalizes the scale.
    if bell_att is not None and bell_tgt is not None:
        _gated(
            "bellcow_score",
            rb_mask,
            bell_att.fillna(0) + bell_tgt.fillna(0),
        )

    if not derived:
        return profile
    return pd.concat([profile, pd.DataFrame(derived)], axis=1)


def apply_rolling_transforms(
    stacked: pd.DataFrame,
    features: Iterable[str] = TRANSFORM_FEATURES,
    window: int = _ROLLING_WINDOW,
) -> pd.DataFrame:
    """Replace listed features with their per-player rolling mean over the last ``window`` seasons.

    Operates on a stacked multi-year profile (must have a ``season`` column
    and be indexed by player name). For each (player, year), the feature value
    becomes the mean of that player's last ``window`` seasons of data
    (including the current year). Where a player has fewer than ``window``
    seasons available, the mean is taken over what's available.

    Args:
        stacked: Multi-year per-player profile indexed by accent-stripped
            player name, with a ``season`` column.
        features: Feature columns to rewrite in place. Columns missing from
            ``stacked`` are silently skipped.
        window: Rolling-window length in seasons (inclusive of the current
            season).

    Returns:
        A copy of ``stacked`` with the listed columns replaced by their
        per-player trailing-``window`` mean.

    Raises:
        ValueError: If ``stacked`` lacks a ``season`` column.
    """
    if "season" not in stacked.columns:
        raise ValueError("apply_rolling_transforms requires a 'season' column")

    out = stacked.copy()
    # Stable season order is required for the rolling window to mean what
    # the name says; sort once up front rather than re-sorting per feature.
    out = out.sort_values("season", kind="stable")
    grouped = out.groupby(out.index, sort=False, group_keys=False)
    for feature in features:
        if feature not in out.columns:
            continue
        out[feature] = grouped[feature].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    return out


def _join_nfl_players(profile: pd.DataFrame) -> pd.DataFrame:
    """Join nfl.import_ids() age/height/bmi onto a player-name-indexed profile."""
    try:
        nfl_players = nfl.import_ids()
    except Exception as exc:
        logger.warning("nfl.import_ids() failed: %s", exc)
        return profile

    nfl_players = nfl_players.loc[nfl_players["position"].isin(_COMP_POSITIONS)]
    nfl_players.index = nfl_players["name"].apply(remove_accents)
    nfl_players["bmi"] = (
        nfl_players["weight"] / nfl_players["height"] / nfl_players["height"]
    )
    nfl_players = nfl_players[["age", "height", "bmi"]].dropna()
    nfl_players = nfl_players[~nfl_players.index.duplicated(keep="last")]
    return profile.join(nfl_players, how="left")


def _fp_dir(year: int) -> object | None:
    """Return the canonical FP data directory for ``year`` if it exists."""
    directory = pkg_resources.files(data) / f"player_data/NFL/{year}"
    return directory if os.path.exists(directory) else None


def _load_player_file(fp_dir: object, filename: str, prefix: str) -> pd.DataFrame | None:
    """Load + prefix one per-player FP CSV; key by accent-stripped Name.

    Some FP exports include footer or stratification rows where ``Team`` and
    ``POS`` are NaN (e.g., "Catchable Throw Percentage" entries that
    duplicate the player ``Name`` column for layout reasons). Those rows
    pollute the player index and explode the downstream ``combine_first``
    join into a Cartesian product, so they are dropped here.
    """
    path = fp_dir / filename
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["Name", "Team", "POS"])
    if df.empty:
        return None

    df = _prefix_columns(df, prefix)
    df.index = df["Name"].apply(remove_accents)
    df.index.name = None
    # A few players still appear twice (mid-season trades produce a row per
    # team). Keep the row with the most games so the comp profile reflects
    # the player's dominant tenure.
    if not df.index.is_unique and "G" in df.columns:
        df = df.sort_values("G", ascending=False)
        df = df.loc[~df.index.duplicated(keep="first")]
    elif not df.index.is_unique:
        df = df.loc[~df.index.duplicated(keep="first")]
    return df


def _load_qb_coverage_file(fp_dir: object) -> pd.DataFrame | None:
    """Load qbCoverageMatchup; reduce multi-row (QB, opponent) to per-QB mean."""
    path = fp_dir / _QB_COVERAGE_FILE
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["Name", "POS"])
    if df.empty:
        return None

    # Numeric columns averaged across the QB's matchup rows; the categorical
    # join key (``Name``) survives the groupby. ``OPP`` is opponent-specific
    # and meaningless once aggregated, so it falls out of the reduction.
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None

    grouped = df.groupby("Name", as_index=True)[numeric_cols].mean()
    # Drop ``G`` / ``Season`` / ``Rank`` -- the per-player FP files already
    # carry these as join-key columns, and re-emitting them under the
    # qb-coverage frame would force ``G_x`` / ``G_y`` splits at combine.
    grouped = grouped.drop(
        columns=[c for c in ("G", "Season", "Rank") if c in grouped.columns]
    )
    grouped = _prefix_columns(grouped.reset_index(), _QB_COVERAGE_PREFIX)
    grouped.index = grouped["Name"].apply(remove_accents)
    grouped.index.name = None
    return grouped


def _load_team_coverage_file(fp_dir: object) -> pd.DataFrame | None:
    """Load coverageMatrixOffense; key by FP team abbreviation for broadcast."""
    path = fp_dir / _TEAM_COVERAGE_FILE
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["Name"])
    if df.empty:
        return None

    fp_team = df["Name"].map(_TEAM_NAME_TO_FP_ABBR)

    # ``Name`` is the team full-name on team-grain rows; drop it before
    # prefixing so the player-grain ``Name`` column upstream isn't
    # clobbered. ``G``, ``Season``, ``Rank`` are team-row sentinels --
    # carrying them through the broadcast would split the player columns
    # into ``G_x`` / ``G_y`` and break the ``player_game_count`` alias.
    df = df.drop(columns=[c for c in ("Name", "G", "Season", "Rank") if c in df.columns])
    df = _prefix_columns(df, _TEAM_COVERAGE_PREFIX)
    df.index = fp_team.values
    df = df.loc[df.index.notna()]
    if df.empty:
        return None
    return df


def _broadcast_team_columns(
    per_year: pd.DataFrame, team_cov: pd.DataFrame
) -> pd.DataFrame:
    """Broadcast team-grain coverage columns to every player on that team."""
    if "Team" not in per_year.columns:
        return per_year

    joined = per_year.merge(
        team_cov, how="left", left_on="Team", right_index=True
    )
    joined.index = per_year.index
    return joined


def _load_pbp_named_by_player(year: int) -> pd.DataFrame | None:
    """Pull PBP/NGS season aggregates and re-key from gsis_id -> player name."""
    try:
        pbp = compute_pbp_season_aggregates(year)
    except Exception as exc:
        logger.warning("compute_pbp_season_aggregates(%s) failed: %s", year, exc)
        return None
    if pbp is None or pbp.empty:
        return None

    try:
        ids = nfl.import_ids()
    except Exception as exc:
        logger.warning("import_ids() failed: %s", exc)
        return None

    ids = ids.loc[
        ids["position"].isin(_COMP_POSITIONS), ["name", "gsis_id"]
    ].copy()
    ids = ids.dropna(subset=["gsis_id"])
    ids["key"] = ids["name"].apply(remove_accents)
    # Multiple historical players can share a normalized name; mirror the
    # nfl_pbp_agg lookup's "keep last" preference for the active player.
    ids = ids.drop_duplicates(subset="key", keep="last")
    name_lookup = ids.set_index("gsis_id")["key"]

    pbp = pbp.loc[pbp.index.isin(name_lookup.index)].copy()
    if pbp.empty:
        return None

    pbp.index = pbp.index.map(name_lookup)
    pbp.index.name = None
    pbp = pbp.loc[pbp.index.notna()]
    pbp = pbp[~pbp.index.duplicated(keep="last")]
    return pbp


def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Apply ``prefix`` + canonicalize whitespace/symbols in every non-key column."""
    new_columns: dict[str, str] = {}
    for col in df.columns:
        if col in _FP_KEY_COLUMNS:
            new_columns[col] = col
            continue
        new_columns[col] = prefix + _canonicalize(col)
    return df.rename(columns=new_columns)


def _canonicalize(name: str) -> str:
    """Translate an FP header into a token-safe column suffix.

    Preserves case (``ANY_A`` not ``any_a``) so the JSON feature lists in
    Phase 1E can reference the same casing convention that
    ``playerCompStats.json`` already uses for the existing PFF columns.
    """
    canonical = name
    # BOM (U+FEFF) leaks through when the first header in a UTF-8-SIG file
    # is the empty-string column from a leading comma; strip defensively.
    canonical = canonical.replace("﻿", "")
    canonical = canonical.replace(" ", "_")
    canonical = canonical.replace("/", "_")
    canonical = canonical.replace("%", "pct")
    canonical = canonical.replace("+", "plus")
    return canonical.strip("_")
