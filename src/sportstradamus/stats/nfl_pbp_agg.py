"""PBP/NGS-derived NFL season aggregates for the player-comp profile.

Phase 1B of the PFF -> FantasyPoints migration. The Phase 1A loader builds
per-player season profiles from FantasyPoints CSVs and then ``combine_first``-
merges the rows returned here on ``gsis_id``. Everything in this module is
sourced from ``nfl_data_py`` (PBP and Next Gen Stats) except the TE blocking
proxy, which crosses two FantasyPoints files and so lives with the derived
metrics rather than the FP loader.
"""

import os
from importlib import resources as pkg_resources

import nfl_data_py as nfl
import pandas as pd

from sportstradamus import data
from sportstradamus.helpers.text import remove_accents
from sportstradamus.spiderLogger import logger

# Minimum dropbacks/run plays required before a QB's mean qb_epa per play is
# considered stable (rookies/spot starters with only a few plays produce wildly
# noisy means that pollute the comp space).
MIN_PLAYS_QB_EPA = 100

# Minimum pressured dropbacks before reporting a per-QB pressure EPA mean.
# Pressures are rare even for full-season starters, so a lower floor than the
# overall play threshold is appropriate.
MIN_PRESSURE_PLAYS = 30

# Minimum rushing attempts before reporting a per-RB success rate. PBP success
# is binary per play, so the mean is unreliable for short-yardage / change-up
# backs with limited carries.
MIN_RUSH_ATTEMPTS = 50

# 75th-percentile target for time-to-throw aggregation across a QB's weekly
# NGS rows -- captures the tail (e.g., long-developing concepts) without being
# pinned to the single slowest week.
TTT_PERCENTILE = 0.75

# QB-relevant play types when computing qb_epa per play. Sacks and no-plays
# are excluded by PBP's own filter via the qb_epa column being NaN there.
QB_PLAY_TYPES = ("pass", "run")


def compute_pbp_season_aggregates(year: int) -> pd.DataFrame:
    """Compute PBP- and NGS-derived season aggregates for NFL comp profiles.

    All returned columns are prefixed ``pbp_`` so the FantasyPoints loader can
    disambiguate them. The single FP-derived column (``block_pct_proxy``) keeps
    the legacy name expected downstream.

    Args:
        year: NFL season (e.g., 2024).

    Returns:
        DataFrame indexed by ``gsis_id`` with one row per player who clears
        the minimum-sample thresholds for at least one metric. Players missing
        from the index should be treated as NaN by the caller (the loader
        ``combine_first``-merges on ``gsis_id``).
    """
    pbp = _safe_import_pbp(year)
    ngs_passing = _safe_import_ngs("passing", year)

    frames: list[pd.DataFrame] = []

    if pbp is not None:
        frames.append(_qb_epa_aggregates(pbp))
        frames.append(_rb_success_aggregate(pbp))
        frames.append(_receiver_cpoe_aggregate(pbp))

    if ngs_passing is not None:
        frames.append(_ngs_passing_aggregates(ngs_passing))

    block_proxy = _te_block_pct_proxy(year)
    if block_proxy is not None:
        frames.append(block_proxy)

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()

    aggregates = frames[0]
    for frame in frames[1:]:
        aggregates = aggregates.combine_first(frame)
    aggregates.index.name = "gsis_id"
    return aggregates


def _safe_import_pbp(year: int) -> pd.DataFrame | None:
    """Return the PBP frame for ``year``, or None if the import fails."""
    try:
        return nfl.import_pbp_data([year])
    except Exception as exc:
        logger.warning("import_pbp_data(%s) failed: %s", year, exc)
        return None


def _safe_import_ngs(stat_type: str, year: int) -> pd.DataFrame | None:
    """Return the NGS frame for ``(stat_type, year)``, or None on failure."""
    try:
        return nfl.import_ngs_data(stat_type, [year])
    except Exception as exc:
        logger.warning("import_ngs_data(%s, %s) failed: %s", stat_type, year, exc)
        return None


def _qb_epa_aggregates(pbp: pd.DataFrame) -> pd.DataFrame:
    """QB mean ``qb_epa`` per play and under-pressure variant, keyed by gsis_id."""
    passer = pbp.loc[
        pbp["passer_player_id"].notna()
        & pbp["play_type"].isin(QB_PLAY_TYPES)
        & pbp["qb_epa"].notna(),
        ["passer_player_id", "qb_epa"],
    ]

    if passer.empty:
        return pd.DataFrame()

    grouped = passer.groupby("passer_player_id")["qb_epa"]
    epa_per_play = grouped.mean()
    play_counts = grouped.size()
    epa_per_play = epa_per_play.loc[play_counts >= MIN_PLAYS_QB_EPA]

    pressure_col = "was_pressure" if "was_pressure" in pbp.columns else "qb_hit"
    pressure_plays = pbp.loc[
        pbp["passer_player_id"].notna() & (pbp[pressure_col] == 1) & pbp["qb_epa"].notna(),
        ["passer_player_id", "qb_epa"],
    ]
    pressure_grouped = pressure_plays.groupby("passer_player_id")["qb_epa"]
    epa_pressure = pressure_grouped.mean()
    pressure_counts = pressure_grouped.size()
    epa_pressure = epa_pressure.loc[pressure_counts >= MIN_PRESSURE_PLAYS]

    return pd.DataFrame(
        {
            "pbp_epa_per_play": epa_per_play,
            "pbp_epa_under_pressure": epa_pressure,
        }
    )


def _rb_success_aggregate(pbp: pd.DataFrame) -> pd.DataFrame:
    """RB rushing success rate from PBP's binary ``success`` column."""
    rushes = pbp.loc[
        pbp["rusher_player_id"].notna() & (pbp["play_type"] == "run") & pbp["success"].notna(),
        ["rusher_player_id", "success"],
    ]
    if rushes.empty:
        return pd.DataFrame()

    grouped = rushes.groupby("rusher_player_id")["success"]
    success_rate = grouped.mean()
    attempts = grouped.size()
    success_rate = success_rate.loc[attempts >= MIN_RUSH_ATTEMPTS]
    return pd.DataFrame({"pbp_success_rate_at_distance": success_rate})


def _receiver_cpoe_aggregate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Receiver-side completion percentage over expected from per-target PBP cpoe.

    NGS receiving does not expose an ``expected_catch_percentage`` column, so
    we derive a comparable signal from PBP's per-play ``cpoe`` (completion
    percentage over expected) attributed to the targeted receiver. Returned in
    percentage-point units, matching the source column.
    """
    if "cpoe" not in pbp.columns or "receiver_player_id" not in pbp.columns:
        return pd.DataFrame()

    targets = pbp.loc[
        pbp["receiver_player_id"].notna() & pbp["cpoe"].notna(),
        ["receiver_player_id", "cpoe"],
    ]
    if targets.empty:
        return pd.DataFrame()

    cpoe_mean = targets.groupby("receiver_player_id")["cpoe"].mean()
    return pd.DataFrame({"pbp_catch_rate_over_expected": cpoe_mean})


def _ngs_passing_aggregates(ngs_passing: pd.DataFrame) -> pd.DataFrame:
    """QB 75th-percentile time-to-throw and mean intended air yards.

    NGS publishes per-week rows (``week>0``) plus a season-summary row
    (``week==0``); we use the weekly rows so the percentile reflects real
    week-to-week variation rather than already-aggregated season values.
    """
    weekly = ngs_passing.loc[ngs_passing["week"] > 0]
    if weekly.empty:
        return pd.DataFrame()

    grouped = weekly.groupby("player_gsis_id")
    ttt_p75 = grouped["avg_time_to_throw"].quantile(TTT_PERCENTILE)
    air_yards = grouped["avg_intended_air_yards"].mean()
    return pd.DataFrame(
        {
            "pbp_ttt_p75": ttt_p75,
            "pbp_intended_air_yards": air_yards,
        }
    )


def _te_block_pct_proxy(year: int) -> pd.DataFrame | None:
    """TE blocking-share proxy from FP offense snaps minus routes run.

    Reads the FP CSVs from the canonical ``data/player_data/NFL/{year}``
    location. Resolves FP ``Name`` -> ``gsis_id`` via ``nfl.import_ids``.
    """
    snaps_df, routes_df = _load_fp_te_inputs(year)
    if snaps_df is None or routes_df is None:
        return None

    snaps_te = snaps_df.loc[snaps_df["POS"] == "TE", ["Name", "Snaps"]].copy()
    routes_te = routes_df.loc[routes_df["POS"] == "TE", ["Name", "RTE"]].copy()
    snaps_te["key"] = snaps_te["Name"].apply(remove_accents)
    routes_te["key"] = routes_te["Name"].apply(remove_accents)

    merged = snaps_te.merge(routes_te[["key", "RTE"]], on="key", how="inner")
    merged = merged.loc[(merged["Snaps"] > 0) & merged["RTE"].notna()]
    if merged.empty:
        return None

    merged["block_pct_proxy"] = (merged["Snaps"] - merged["RTE"]) / merged["Snaps"]
    merged["block_pct_proxy"] = merged["block_pct_proxy"].clip(lower=0.0)

    gsis_lookup = _build_gsis_lookup(positions=("TE",))
    if gsis_lookup is None or gsis_lookup.empty:
        return None

    merged = merged.merge(gsis_lookup, on="key", how="inner")
    merged = merged.dropna(subset=["gsis_id"])
    if merged.empty:
        return None

    return merged.set_index("gsis_id")[["block_pct_proxy"]]


def _load_fp_te_inputs(year: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Locate and load the FP offenseSnaps + receivingRoutesRun files for ``year``."""
    fp_dir = pkg_resources.files(data) / f"player_data/NFL/{year}"
    snaps_path = fp_dir / "offenseSnapsExport.csv"
    routes_path = fp_dir / "receivingRoutesRunExport.csv"
    if not os.path.exists(snaps_path) or not os.path.exists(routes_path):
        return None, None

    snaps_df = pd.read_csv(snaps_path)
    routes_df = pd.read_csv(routes_path)
    return snaps_df, routes_df


def _build_gsis_lookup(positions: tuple[str, ...]) -> pd.DataFrame | None:
    """Return a ``(key, gsis_id)`` table for accent-stripped names at ``positions``."""
    try:
        ids = nfl.import_ids()
    except Exception as exc:
        logger.warning("import_ids() failed: %s", exc)
        return None

    ids = ids.loc[ids["position"].isin(positions), ["name", "gsis_id"]].copy()
    ids = ids.dropna(subset=["gsis_id"])
    ids["key"] = ids["name"].apply(remove_accents)
    # Multiple historical players can share a normalized name; keep the most
    # recently issued gsis_id (the loader prefers active players).
    ids = ids.drop_duplicates(subset="key", keep="last")
    return ids[["key", "gsis_id"]]
