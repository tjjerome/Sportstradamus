"""CLI entry point: meditate command orchestrates per-league, per-market training."""

import importlib.resources as pkg_resources
import json
import warnings
from datetime import datetime, timedelta

import click
import numpy as np

from sportstradamus import data
from sportstradamus.helpers import Archive, book_weights, feature_filter
from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA
from sportstradamus.training.calibration import fit_book_weights
from sportstradamus.training.correlate import correlate
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.pipeline import train_market

warnings.simplefilter("ignore", UserWarning)
np.seterr(divide="ignore", invalid="ignore")


@click.command()
@click.option("--force/--no-force", default=False, help="Force update of all models")
@click.option(
    "--league",
    type=click.Choice(["All", "NFL", "NBA", "MLB", "NHL", "WNBA"]),
    default="All",
    help="Select league to train on",
)
@click.option(
    "--rebuild-filter/--no-rebuild-filter",
    default=False,
    help="Train with full feature set (ignore Filtered), then rerun SHAP and rewrite filter",
)
@click.option(
    "--reset-markets",
    default="",
    help="Comma-separated league:market pairs (or just market for active league) to clear from Filtered before training",
)
def meditate(force, league, rebuild_filter, reset_markets):
    """Train or retrain LightGBMLSS models for each configured market."""
    np.random.seed(69)

    if reset_markets.strip():
        ff_path = pkg_resources.files(data) / "feature_filter.json"
        with open(ff_path) as fh:
            ff = json.load(fh)
        for tok in [t.strip() for t in reset_markets.split(",") if t.strip()]:
            if ":" in tok:
                lg, mk = tok.split(":", 1)
            else:
                lg, mk = league, tok
            mk = mk.strip()
            ff.setdefault(lg, {}).setdefault("Filtered", {})
            if mk in ff[lg]["Filtered"]:
                del ff[lg]["Filtered"][mk]
                print(f"Reset filter for {lg}:{mk}")
        with open(ff_path, "w") as fh:
            json.dump(ff, fh, indent=4)
        # Reload module-level feature_filter so this run sees the change
        from sportstradamus import helpers as _hp

        _hp.feature_filter.clear()
        _hp.feature_filter.update(ff)

    nba = StatsNBA()
    nfl = StatsNFL()
    wnba = StatsWNBA()
    # mlb = StatsMLB()
    # nhl = StatsNHL()

    stat_structs = {}
    archive = Archive()

    if (
        league == "All" and datetime.today().date() > (nba.season_start - timedelta(days=7))
    ) or league == "NBA":
        nba.load()
        nba.update()
        stat_structs.update({"NBA": nba})
    if (
        league == "All" and datetime.today().date() > (nfl.season_start - timedelta(days=7))
    ) or league == "NFL":
        nfl.load()
        nfl.update()
        stat_structs.update({"NFL": nfl})
    if (
        league == "All" and datetime.today().date() > (wnba.season_start - timedelta(days=7))
    ) or league == "WNBA":
        wnba.load()
        wnba.update()
        stat_structs.update({"WNBA": wnba})
    # if datetime.today().date() > (mlb.season_start - timedelta(days=7)) or league == "MLB":
    #     mlb.load()
    #     mlb.update()
    #     stat_structs.update({"MLB": mlb})
    # if datetime.today().date() > (nhl.season_start - timedelta(days=7)) or league == "NHL":
    #     nhl.load()
    #     nhl.update()
    #     stat_structs.update({"NHL": nhl})

    active_markets = dict(ALL_MARKETS)
    if league != "All":
        active_markets = {league: ALL_MARKETS[league]}

    for lg, markets in active_markets.items():
        stat_data = stat_structs.get(lg)
        if stat_data is None:
            continue

        # Fit book weights for moneylines and totals before per-market loop
        book_weights.setdefault(lg, {}).setdefault("Moneyline", {})
        book_weights[lg]["Moneyline"] = fit_book_weights(
            lg, "Moneyline", stat_data, archive, book_weights
        )
        book_weights.setdefault(lg, {}).setdefault("Totals", {})
        book_weights[lg]["Totals"] = fit_book_weights(
            lg, "Totals", stat_data, archive, book_weights
        )

        if lg == "MLB":
            for extra_market in ("1st 1 innings", "pitcher win", "triples"):
                book_weights.setdefault(lg, {}).setdefault(extra_market, {})
                book_weights[lg][extra_market] = fit_book_weights(
                    lg, extra_market, stat_data, archive, book_weights
                )
        elif lg == "NHL":
            stat_data.dump_goalie_list()

        with open(pkg_resources.files(data) / "book_weights.json", "w") as outfile:
            json.dump(book_weights, outfile, indent=4)

        stat_data.update_player_comps()
        correlate(lg, stat_data, force)
        league_start_date = stat_data.trim_gamelog()

        for market in markets:
            train_market(lg, market, stat_data, force, rebuild_filter, archive, league_start_date)
