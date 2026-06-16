"""Offer matching and distributional scoring.

:func:`process_offers` is the outer loop: it iterates over every
league/market pair in the scraped offer dict, calls :func:`match_offers`
to build a feature matrix, hands it to :func:`model_prob` for
distributional scoring, and finally passes all scored offers through
:func:`find_correlation` to annotate correlations and build parlays.

:func:`match_offers` loads the model pickle's ``expected_columns`` list
and slices the ``Stats.get_stats`` output down to the schema the model
was trained on.
"""

from __future__ import annotations

import datetime
import importlib.resources as pkg_resources
import os.path
import pickle
import warnings

import line_profiler
import pandas as pd
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import LazyArchive, stat_map
from sportstradamus.prediction.correlation import find_correlation
from sportstradamus.prediction.model_prob import book_fallback_prob, model_prob, normalize_market
from sportstradamus.spiderLogger import logger

# LazyArchive defers DuckDB lock acquisition until the first attribute
# access. See LazyArchive docstring in helpers/archive.py.
archive = LazyArchive()


@line_profiler.profile
def process_offers(
    offer_dict,
    book,
    stats,
    *,
    contest_variant="pooled",
    legacy=False,
    corr_sink=None,
    story_sink=None,
):
    """Score all offers from one platform and return annotated DataFrames.

    Iterates every league/market pair in ``offer_dict``, adds DFS lines to
    the archive, builds feature matrices via ``Stats.get_stats``, scores
    each matrix with :func:`model_prob`, then calls :func:`find_correlation`
    to annotate correlations and enumerate parlays.

    Args:
        offer_dict: ``{league: {market: [offer, ...]}}`` from the scraper.
        book: DFS platform name (e.g. ``"Underdog"``).
        stats: ``{league: Stats}`` dict for currently active leagues.
        contest_variant: Underdog contest variant; passed to find_correlation.
        legacy: Legacy-pipeline escape hatch; passed to find_correlation.
        corr_sink: Optional list the per-game correlation slices accumulate into;
            forwarded to find_correlation. See its docstring.
        story_sink: Optional list the per-game scoring bundles accumulate into;
            forwarded to find_correlation for the story-menu generator.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(offer_df, parlay_df)``.
    """
    new_offers = []
    logger.info(f"Processing {book} offers")
    if len(offer_dict) > 0:
        total = sum(sum(len(i) for i in v.values()) for v in offer_dict.values())
        with tqdm(total=total, desc=f"Matching {book} Offers", unit="offer") as pbar:
            for league, markets in offer_dict.items():
                new_offers.extend(_match_league_offers(league, markets, stats, book, pbar))

    offer_df, parlays = find_correlation(
        new_offers,
        stats,
        book,
        contest_variant=contest_variant,
        legacy=legacy,
        corr_sink=corr_sink,
        story_sink=story_sink,
    )

    logger.info(str(len(offer_df)) + " offers processed")
    return offer_df, parlays


def _match_league_offers(league, markets, stats, book, pbar):
    """Archive and score one league's markets; return its scored offer records.

    A league with no active ``Stats`` object still has its DFS lines archived
    (so the odds land in the archive) but is not scored; a league whose season
    has not started is skipped entirely.
    """
    if league not in stats:
        for offers in markets.values():
            archive.add_dfs(offers, book, stat_map[book])
            pbar.update(len(offers))
        return []

    stat_data = stats.get(league)
    if stat_data.season_start > datetime.datetime.today().date() - datetime.timedelta(days=14):
        logger.info(f"{league} season has not started, skipping stat matching")
        return []

    all_offers = {}
    for offers in markets.values():
        all_offers.update({v["Player"]: v for v in offers})
    all_offers = list(all_offers.values())
    stat_data.get_depth(all_offers)
    stat_data.get_volume_stats(all_offers)
    if league == "MLB":
        stat_data.get_volume_stats(all_offers, pitcher=True)

    scored = []
    for market, offers in markets.items():
        archive.add_dfs(offers, book, stat_map[book])
        pbar.update(len(offers))
        scored.extend(_score_market(offers, league, market, book, stat_data))
    return scored


def _score_market(offers, league, market, book, stat_data):
    """Score one market's offers — trained model when present, else book odds.

    Builds the feature matrix; when it comes back empty (no model pickle, or no
    player matched) falls back to :func:`book_fallback_prob`, which devigs the
    composite book odds into the model slot. Returns the scored offer records.
    """
    playerStats = match_offers(offers, league, market, book, stat_data)
    if len(playerStats) == 0:
        modeled = book_fallback_prob(offers, league, market, book, stat_data)
        if not modeled:
            logger.info(f"{league}, {market} offers not matched")
        return modeled
    return model_prob(offers, league, market, book, stat_data, playerStats)


@line_profiler.profile
def match_offers(offers, league, market, platform, stat_data):
    """Build a feature matrix for ``offers`` from the ``Stats`` object.

    Normalizes the market name, calls ``stat_data.get_stats``, and slices
    the result down to the ``expected_columns`` stored in the model pickle
    so the feature schema exactly matches what LightGBMLSS was trained on.

    Args:
        offers: Raw offer dicts for one market.
        league: League key.
        market: Raw market name from the scraper.
        platform: DFS platform name.
        stat_data: Loaded ``Stats`` instance for ``league``.

    Returns:
        pd.DataFrame: Feature matrix indexed by player name, or an empty
            DataFrame when the market is not in the gamelog or no model
            file exists.
    """
    market = normalize_market(league, market, platform)
    if market in stat_data.gamelog.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            playerStats = stat_data.get_stats(market, offers)
            if playerStats.empty:
                return playerStats

            filename = "_".join([league, market]).replace(" ", "-")
            filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
            if not filepath.is_file():
                return pd.DataFrame()
            with open(filepath, "rb") as infile:
                expected_cols = pickle.load(infile)["expected_columns"]
            playerStats = playerStats[expected_cols]

            return (
                playerStats[~playerStats.index.duplicated(keep="first")]
                .fillna(0)
                .infer_objects(copy=False)
            )
    else:
        return pd.DataFrame()
