"""Offer matching and distributional scoring.

:func:`process_offers` is the outer loop: it archives every league/market pair
in the scraped offer dict, scores each league one game date at a time
(:func:`_match_league_offers`), and finally passes all scored offers through
:func:`find_correlation` to annotate correlations and build parlays. A market's
feature matrix comes from :func:`match_offers` and is scored by :func:`model_prob`,
or by :func:`book_fallback_prob` when the matrix comes back empty.

:func:`match_offers` loads the model pickle's ``expected_columns`` list
and slices the ``Stats.get_stats`` output down to the schema the model
was trained on.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import pickle
import warnings
from datetime import date, datetime
from itertools import chain

import line_profiler
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import LazyArchive, odds_budget, stat_map, stat_meta
from sportstradamus.helpers.io import VOLUME_STATS
from sportstradamus.prediction.correlation import find_correlation
from sportstradamus.prediction.model_prob import book_fallback_prob, model_prob, normalize_market
from sportstradamus.spiderLogger import logger
from sportstradamus.stats.base import earliest_last
from sportstradamus.training.ship_config import WITHHELD

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
        corr_sink=corr_sink,
        story_sink=story_sink,
    )

    logger.info(str(len(offer_df)) + " offers processed")
    return offer_df, parlays


def _match_league_offers(league, markets, stats, book, pbar):
    """Archive and score one league's markets; return its scored offer records.

    Every market's lines reach the archive before anything that can raise, so a
    scoring crash never costs a capture: a feature-prep failure costs only that
    game date's pass, a scoring failure only that market's records in the pass. A
    league with no active ``Stats`` object is archived but never scored, and so is
    a game before the league's opening night.

    A player's features are built as of his earliest offered game, and each game
    date is its own pass, as the training matrix builds each gameday: depth and
    volume for that date's players, then every market's scoring of all their
    offers, a later game's too. Nothing of another date may run inside a pass,
    because ``Stats.base_profile`` rebuilds ``playerProfile`` whenever the date
    moves, wiping the depth and volume columns the pass prepared.
    """
    # style: allow-complexity — one pass a game date, its feature prep and each market's
    # scoring behind their own catches, as get_training_matrix loops its gamedays.
    for offers in markets.values():
        archive.add_dfs(offers, book, stat_map[book])
        pbar.update(len(offers))
    if league not in stats:
        return []
    markets = _in_season_markets(league, markets)
    if not markets:
        return []

    stat_data = stats[league]
    by_player = {
        offer["Player"]: offer for offer in earliest_last(chain.from_iterable(markets.values()))
    }
    today = datetime.today().date()
    passes = {}
    for offer in by_player.values():
        # Never before today: a past date would build the row from its realized game.
        day = max(date.fromisoformat(offer["Date"]), today) if offer["Date"] else today
        passes.setdefault(day, []).append(offer)

    scored = []
    for day, day_offers in sorted(passes.items()):
        try:
            stat_data.get_depth(day_offers, day)
            stat_data.get_volume_stats(day_offers, day)
            if league == "MLB":
                stat_data.get_volume_stats(day_offers, day, pitcher=True)
        except Exception:
            logger.exception(f"{league} features unavailable for {day}, its players not scored")
            continue
        players = {offer["Player"] for offer in day_offers}
        for market, offers in markets.items():
            pass_offers = [offer for offer in offers if offer["Player"] in players]
            if not pass_offers:
                continue
            try:
                scored.extend(_score_market(pass_offers, league, market, book, stat_data, day))
            except Exception:
                logger.exception(f"{league} {market} not scored")
    return scored


def _in_season_markets(league, markets):
    """Drop offers for games before the league's opening night: pre-season is never priced.

    Opening night is the odds feed's (:func:`odds_budget.season_opener`). While the feed
    holds none, and for an offer with no date, nothing is dropped.
    """
    opener = odds_budget.season_opener(league)
    if opener is None:
        return markets
    # Both platforms stamp an offer's Date as YYYY-MM-DD text, which orders as dates do.
    first_day = opener.isoformat()
    kept = {
        market: [offer for offer in offers if not offer["Date"] or offer["Date"] >= first_day]
        for market, offers in markets.items()
    }
    dropped = sum(map(len, markets.values())) - sum(map(len, kept.values()))
    if dropped:
        logger.info(f"{league}: {dropped} offers before opening night {first_day} not scored")
    return {market: offers for market, offers in kept.items() if offers}


def _score_market(offers, league, market, book, stat_data, date):
    """Score one market's offers — trained model when present, else book odds.

    Builds the feature matrix as of ``date``, the game date its pass prepared depth
    and volume for; when it comes back empty (a withheld cell, no model pickle, or no
    player matched) falls back to :func:`book_fallback_prob`, which devigs the composite
    book odds into the model slot. Returns the scored offer records.
    """
    playerStats = match_offers(offers, league, market, book, stat_data, date)
    if len(playerStats) == 0:
        modeled = book_fallback_prob(offers, league, market, book, stat_data, date)
        if not modeled:
            logger.info(f"{league}, {market} offers not matched")
        return modeled
    return model_prob(offers, league, market, book, stat_data, playerStats)


@line_profiler.profile
def match_offers(offers, league, market, platform, stat_data, date):
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
        date: Date the features are built as of, the one the caller prepared
            ``stat_data``'s depth and volume for.

    Returns:
        pd.DataFrame: Feature matrix indexed by player name, or an empty
            DataFrame when the cell is withheld, the market is not in the
            gamelog, or no model file exists.
    """
    market = normalize_market(league, market, platform)
    filename = "_".join([league, market]).replace(" ", "-")
    filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
    if stat_meta.get(league, {}).get(market, {}).get("shipped") == WITHHELD:
        # A volume denominator keeps its pickle on purpose: prune_model_pickle skips it.
        if filepath.is_file() and market not in VOLUME_STATS[league]:
            logger.warning(
                f"{filename} withheld but pickle on disk — serving the book consensus; "
                "run meditate to prune the pickle"
            )
        return pd.DataFrame()
    if market not in stat_data.gamelog.columns:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        playerStats = stat_data.get_stats(market, offers, date)
        if playerStats.empty:
            return playerStats

        if not filepath.is_file():
            return pd.DataFrame()
        with open(filepath, "rb") as infile:
            expected_cols = pickle.load(infile)["expected_columns"]
        playerStats = playerStats.reindex(columns=expected_cols)

        # Training's matrix ends the same way: a 0-target game makes yards per target infinite.
        return (
            playerStats[~playerStats.index.duplicated(keep="first")]
            .fillna(0)
            .infer_objects(copy=False)
            .replace([np.inf, -np.inf], 0)
        )
