"""Main ``prophecize`` CLI entrypoint.

Orchestrates the full prediction pipeline:

1. Load and update ``Stats`` objects for active leagues
2. Scrape DFS offers (Underdog, Sleeper)
3. Score via :func:`process_offers` (feature extraction + distributional model)
4. Snapshot scored offers + parlays as parquet for the Streamlit dashboard
5. Persist a rolling year of predictions to ``data/history.parquet`` and
   the new parlays to ``data/parlay_hist.parquet``
"""

from __future__ import annotations

import datetime
import importlib.resources as pkg_resources
import os
from functools import partialmethod

import click
import line_profiler
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import creds, data
from sportstradamus.analysis import _merge_offers
from sportstradamus.books import get_sleeper, get_ud
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE, LazyArchive, get_logger, stat_map
from sportstradamus.helpers.io import (
    read_history,
    read_parlay_hist,
    write_history,
    write_parlay_hist,
)
from sportstradamus.history_schema import HISTORY_COLS, PREDICTION_LEVEL_COLS
from sportstradamus.prediction.persist import (
    write_current_game_context,
    write_current_game_corr,
    write_current_game_stories,
    write_current_offer_details,
    write_current_offers,
)
from sportstradamus.prediction.scoring import process_offers
from sportstradamus.prediction.stories import (
    attach_offer_why,
    attach_parlay_theses,
    build_game_context,
    build_game_stories,
)
from sportstradamus.prediction.stories.details import build_offer_details
from sportstradamus.spiderLogger import logger
from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA

pd.set_option("mode.chained_assignment", None)
pd.set_option("future.no_silent_downcasting", True)
os.environ["LINE_PROFILE"] = "0"

# LazyArchive defers DuckDB lock acquisition until the first attribute
# access. See LazyArchive docstring in helpers/archive.py.
archive = LazyArchive()

_HISTORY_RETENTION_DAYS = 365

# Per-league gamelog volume-stat column for the deep-dive volume trend. NFL maps by
# base position (the snapshot's depth-rank suffix is stripped before lookup) so each
# player shows their own opportunity stat — pass attempts, carries, or targets.
_VOLUME_STAT: dict[str, str | dict[str, str]] = {
    "NBA": "MIN",
    "WNBA": "MIN",
    "NFL": {"QB": "attempts", "RB": "carries", "WR": "targets", "TE": "targets"},
    "MLB": "plateAppearances",
    "NHL": "TimeShare",
}
# Leagues whose comps use the KNN ``_comp_pairs`` table; MLB's pitcher/hitter comps
# have a different structure and are deferred (empty comps-vs-opponent panel).
_COMP_PAIR_LEAGUES = {"NBA", "WNBA", "NFL", "NHL"}
# Recency window for the detail gamelog — covers the 300-day comps window with margin
# while bounding the per-offer work. ``self.gamelog`` is the full multi-year log.
_DETAIL_GAMELOG_DAYS = 400
_FEATURE_IMPORTANCES_PATH = pkg_resources.files(data) / "training" / "feature_importances.csv"


def _offer_details_frame(snapshot_offers: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Assemble the per-offer detail prerender from the scored slate + Stats objects.

    Uses ``sd.gamelog`` (always loaded) not ``short_gamelog`` (only set inside
    ``get_stats``, absent for book-fallback leagues); the detail helpers window it
    to the comps / sparkline ranges they need.
    """
    today = datetime.date.today()
    if snapshot_offers.empty:
        return build_offer_details(
            snapshot_offers, {}, pd.DataFrame(), exclude=frozenset(), today=today
        )
    importances = pd.read_csv(_FEATURE_IMPORTANCES_PATH, index_col=0)
    empty_pairs = pd.DataFrame(columns=["player", "comp"])
    cutoff = pd.Timestamp(today - datetime.timedelta(days=_DETAIL_GAMELOG_DAYS))
    offer_leagues = set(snapshot_offers["League"].unique())
    league_data = {}
    for league, sd in stats.items():
        if league not in offer_leagues:
            continue
        dcol = sd.log_strings["date"]
        gl = sd.gamelog.copy()
        gl[dcol] = pd.to_datetime(gl[dcol])
        league_data[league] = {
            "comp_pairs": sd._comp_pairs() if league in _COMP_PAIR_LEAGUES else empty_pairs,
            "gamelog": gl[gl[dcol] >= cutoff],
            "cols": sd.log_strings,
            "volume_stat": _VOLUME_STAT.get(league, ""),
        }
    return build_offer_details(
        snapshot_offers, league_data, importances, exclude=frozenset(), today=today
    )


@click.command()
@click.option("--progress/--no-progress", default=True, help="Display progress bars")
@click.option(
    "--legacy-correlation/--no-legacy-correlation",
    default=False,
    help=(
        "Reproduce the pre-2026.05 parlay pipeline verbatim — no PSD repair, "
        "no push-aware EV, mixed insurance/power Boost overwrite. Removed "
        "next release; provided as a one-cycle escape hatch."
    ),
)
@click.option(
    "--contest-variant",
    type=click.Choice(["pooled", "power", "flex", "insurance", "rivals"]),
    default="pooled",
    help=(
        "Underdog payout pool for parlay scoring. Default 'pooled' "
        "combines power (2-3 legs) and flex (4+ legs) into one pool; "
        "single-variant names are kept for the pickem-build path."
    ),
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Verbosity for the structured JSONL log.",
)
@line_profiler.profile
def main(progress, legacy_correlation, contest_variant, log_level):
    """Run the full prediction pipeline and write parquet snapshots for the dashboard."""
    # style: allow-complexity — prophecize entrypoint: a flat top-level pipeline
    # (per-league load/update, per-platform fetch+score, snapshot + history
    # persistence). The residual CC is sequential stages and per-platform/-league
    # guards, not nested logic; decomposing it would only scatter the pipeline.
    cli_log = get_logger("prophecize")
    cli_log.setLevel(log_level)
    cli_log.info(
        "prophecize invoked",
        extra={"contest_variant": contest_variant, "legacy_correlation": legacy_correlation},
    )
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=(not progress))

    sports = []
    nba = StatsNBA()
    nba.load()
    if datetime.datetime.today().date() > (nba.season_start - datetime.timedelta(days=7)):
        sports.append("NBA")
    nfl = StatsNFL()
    nfl.load()
    if datetime.datetime.today().date() > (nfl.season_start - datetime.timedelta(days=7)):
        sports.append("NFL")
    wnba = StatsWNBA()
    wnba.load()
    if datetime.datetime.today().date() > (wnba.season_start - datetime.timedelta(days=7)):
        sports.append("WNBA")

    stats = {}
    if "NBA" in sports:
        nba.update()
        stats.update({"NBA": nba})
    if "NFL" in sports:
        nfl.update()
        stats.update({"NFL": nfl})
    if "WNBA" in sports:
        wnba.update()
        stats.update({"WNBA": wnba})

    all_offers: list[pd.DataFrame] = []
    parlay_df = pd.DataFrame()
    platforms_run: list[str] = []
    corr_sink: list[dict] = []
    story_sink: list = []

    try:
        ud_dict = get_ud()
        ud_offers, ud5 = process_offers(
            ud_dict,
            "Underdog",
            stats,
            contest_variant=contest_variant,
            legacy=legacy_correlation,
            corr_sink=corr_sink,
            story_sink=story_sink,
        )
        parlay_df = pd.concat([parlay_df, ud5])
        ud_offers["Market"] = ud_offers["Market"].map(stat_map["Underdog"])
        ud_offers["Stat"] = ud_offers[
            "Market"
        ]  # preserve gamelog key for dashboard history lookups
        # model_prob pre-multiplied Boost by UNDERDOG_BOOST_BASELINE so that
        # ``Model = Model P * Boost`` is per-$1 EV. The persisted snapshot
        # (current_offers.parquet, history.parquet) is consumed downstream
        # (dashboard display, nightly profit-sim) as the raw UD promo
        # multiplier, so divide the baseline back out here. Matches what the
        # user sees on Underdog (1.00 = no promo, 1.05 = 5% promo).
        ud_offers["Boost"] = ud_offers["Boost"] / UNDERDOG_BOOST_BASELINE
        ud_offers["Platform"] = "Underdog"
        all_offers.append(ud_offers)
        platforms_run.append("Underdog")
    except Exception:
        logger.exception("Failed to get Underdog")

    try:
        sl_dict = get_sleeper()
        sl_offers, sl5 = process_offers(
            sl_dict,
            "Sleeper",
            stats,
            contest_variant=contest_variant,
            legacy=legacy_correlation,
            corr_sink=corr_sink,
            story_sink=story_sink,
        )
        parlay_df = pd.concat([parlay_df, sl5])
        sl_offers["Market"] = sl_offers["Market"].map(stat_map["Sleeper"])
        sl_offers["Stat"] = sl_offers[
            "Market"
        ]  # preserve gamelog key for dashboard history lookups
        # Sleeper Boost stays at the raw books.py value (model_prob only
        # applies UNDERDOG_BOOST_BASELINE to platform == "Underdog"), so no
        # division is needed here for display/storage to match the raw
        # Sleeper promo multiplier.
        sl_offers["Platform"] = "Sleeper"
        all_offers.append(sl_offers)
        platforms_run.append("Sleeper")
    except Exception:
        logger.exception("Failed to get Sleeper")

    snapshot_offers = pd.concat(all_offers) if all_offers else pd.DataFrame()
    if not parlay_df.empty:
        parlay_df.sort_values("Model EV", ascending=False, inplace=True)
        parlay_df.drop_duplicates(inplace=True)
        parlay_df.reset_index(drop=True, inplace=True)
        parlay_df[["Legs", "Misses", "Profit"]] = np.nan

    if not snapshot_offers.empty and "O/U" in snapshot_offers.columns:
        snapshot_offers["O/U"] = snapshot_offers.apply(
            lambda r: archive.get_total(r["League"], r["Date"], r["Team"]) or r["O/U"],
            axis=1,
        )

    if not snapshot_offers.empty:
        key_cols = ["League", "Market", "Date", "Player"]
        line_map = {
            key: archive.get_line(*key)
            for key in snapshot_offers[key_cols]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        snapshot_offers["Consensus Line"] = [
            line_map[key] for key in snapshot_offers[key_cols].itertuples(index=False, name=None)
        ]

    snapshot_offers = attach_offer_why(snapshot_offers)
    game_context = build_game_context(snapshot_offers, archive.default_totals)
    parlay_df = attach_parlay_theses(
        parlay_df, snapshot_offers, corr=corr_sink, context=game_context
    )
    game_stories = build_game_stories(story_sink, snapshot_offers, game_context, corr_sink)
    offer_details = _offer_details_frame(snapshot_offers, stats)

    write_current_offers(
        snapshot_offers,
        parlay_df,
        sports,
        platforms_run,
        contest_variant=contest_variant,
    )
    write_current_game_corr(corr_sink)
    write_current_game_context(game_context)
    write_current_game_stories(game_stories)
    write_current_offer_details(offer_details)

    if not parlay_df.empty:
        old_parlays = read_parlay_hist()
        if not old_parlays.empty:
            combined = pd.concat([parlay_df, old_parlays], ignore_index=True).drop_duplicates(
                subset=["Model EV", "Market EV"], ignore_index=True
            )
        else:
            combined = parlay_df
        write_parlay_hist(combined)

    archive.write()
    logger.info("Checking historical predictions")
    from sportstradamus.analysis import _migrate_flat_history, _offer_tuple

    history = read_history()
    if not history.empty:
        if "Offers" not in history.columns:
            history = _migrate_flat_history(history)
    else:
        history = pd.DataFrame(columns=HISTORY_COLS)

    pred_key = ["Player", "League", "Date", "Market"]

    if all_offers:
        all_df = pd.concat(all_offers)
        all_df.loc[(all_df["Market"] == "AST") & (all_df["League"] == "NHL"), "Market"] = "assists"
        all_df.loc[(all_df["Market"] == "PTS") & (all_df["League"] == "NHL"), "Market"] = "points"
        all_df.loc[(all_df["Market"] == "BLK") & (all_df["League"] == "NHL"), "Market"] = "blocked"
        all_df.dropna(subset="Market", inplace=True, ignore_index=True)
    else:
        all_df = pd.DataFrame()

    new_preds = []
    if not all_df.empty:
        for key, grp in all_df.groupby(pred_key):
            player, league, date, market = key
            latest = grp.iloc[-1]
            offers = [t for _, r in grp.iterrows() if (t := _offer_tuple(r)) is not None]
            row = {
                "Player": player,
                "League": league,
                "Date": date,
                "Market": market,
                "Offers": offers,
            }
            for col in PREDICTION_LEVEL_COLS:
                row[col] = latest.get(col, np.nan)
            new_preds.append(row)

    new_df = pd.DataFrame(new_preds)

    if not history.empty and not new_df.empty:
        history = history.set_index(pred_key)
        new_df = new_df.set_index(pred_key)
        for idx in new_df.index:
            _merge_prediction_row(history, new_df, idx)
        history = history.reset_index()
    elif not new_df.empty:
        history = new_df

    if "Actual" not in history.columns:
        history["Actual"] = np.nan

    gameDates = pd.to_datetime(history.Date).dt.date
    history = history.loc[
        (datetime.datetime.today().date() - datetime.timedelta(days=_HISTORY_RETENTION_DAYS))
        <= gameDates
    ]
    write_history(history)

    logger.info("Success!")


def _merge_prediction_row(history, new_df, idx):
    """Merge one new prediction (both indexed by pred_key) into history at ``idx``.

    Existing rows keep their Offers merged via ``_merge_offers`` and have their
    prediction-level columns overwritten only where the new value is non-NaN; a
    new ``idx`` inserts the whole row. Mutates ``history`` in place.
    """
    if idx not in history.index:
        history.loc[idx] = new_df.loc[idx]
        return
    old_offers = history.at[idx, "Offers"]
    if not isinstance(old_offers, list):
        old_offers = []
    history.at[idx, "Offers"] = _merge_offers(old_offers, new_df.at[idx, "Offers"])
    for col in PREDICTION_LEVEL_COLS:
        val = new_df.at[idx, col]
        if pd.notna(val):
            history.at[idx, col] = val


if __name__ == "__main__":
    main()
