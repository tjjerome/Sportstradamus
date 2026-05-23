"""CLI entry point: meditate command orchestrates per-league, per-market training."""

import importlib.resources as pkg_resources
import json
import warnings
from datetime import datetime, timedelta

import click
import numpy as np

from sportstradamus import data
from sportstradamus.helpers import Archive, book_weights, feature_filter, get_logger
from sportstradamus.helpers.io import prune_model_pickle
from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA
from sportstradamus.training import baselines
from sportstradamus.training.calibration import fit_book_weights
from sportstradamus.training.correlate import correlate
from sportstradamus.training.markets import ALL_MARKETS, select_markets
from sportstradamus.training.pipeline import train_market
from sportstradamus.training.ship_config import (
    WITHHELD,
    load_ship_config,
    resolve_cell_strategy,
    resolve_cell_zinb_mode,
)

warnings.simplefilter("ignore", UserWarning)
np.seterr(divide="ignore", invalid="ignore")

# Reproducibility seed for non-deterministic training runs. Not used under
# --deterministic (which pins RNGs via seed_everything with a fixed value).
_RNG_SEED: int = 69


@click.command()
@click.option(
    "--force/--no-force",
    default=False,
    help=(
        "Skip markets with fresh data (default). "
        "Pass --force to rebuild the prior correlation CSV cache and build "
        "markets whose models are already up to date."
    ),
)
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
@click.option(
    "--rebuild-correlations/--no-rebuild-correlations",
    default=False,
    help="Rebuild only the per-league correlation matrices and exit; skip the per-market training loop",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Verbosity for the structured JSONL log.",
)
@click.option(
    "--deterministic/--no-deterministic",
    default=False,
    help=(
        "DEBUG/EVAL ONLY: pin all RNGs, use fixed fast hyperparameters, and "
        "freeze input to the cached parquet so runs are bit-identical for the "
        "compression eval harness. NEVER publish a model trained with this "
        "flag — it is deliberately low quality."
    ),
)
@click.option(
    "--target-strategy",
    type=click.Choice(list(baselines.STRATEGY_SLUGS)),
    default="ratio_meanyr",
    show_default=True,
    help=(
        "Target/baseline strategy for SkewNormal markets. "
        "Non-default values are A/B experiments (run under --deterministic); "
        "the default 'ratio_meanyr' is current production behavior."
    ),
)
@click.option(
    "--zinb-mode",
    type=click.Choice(list(baselines.ZINB_MODES)),
    default="joint",
    show_default=True,
    help=(
        "Model architecture for ZINB markets. 'joint' is the legacy "
        "jointly-fit LightGBMLSS ZINB. 'hurdle' uses the two-stage "
        "HurdleZINB (calibrated zero classifier + NegBin on positives + "
        "derived-pi gate; see docs/OVERCONFIDENCE_INVESTIGATION.md §2). "
        "Default 'joint' is byte-identical to pre-P2.B production."
    ),
)
@click.option(
    "--market",
    default=None,
    help=(
        "Comma-separated market stem(s) to train (e.g. 'FTM,STL'). "
        "When omitted, all markets for the active league are trained. "
        "A stem absent from every active league is an error."
    ),
)
def meditate(
    force,
    league,
    rebuild_filter,
    reset_markets,
    rebuild_correlations,
    log_level,
    deterministic,
    target_strategy,
    zinb_mode,
    market,
):
    """Train or retrain LightGBMLSS models for each configured market."""
    # --deterministic implies --force: the input-freeze (new_M = empty)
    # otherwise short-circuits train_market when a prior model pickle exists,
    # which is precisely when the eval harness needs a fresh deterministic
    # rebuild. See docs/operation_ship_75.md (Step 0 verification) for the rationale.
    if deterministic and not force:
        force = True
    log = get_logger("meditate")
    log.setLevel(log_level)
    log.info(
        "meditate invoked",
        extra={
            "force": force,
            "league": league,
            "rebuild_filter": rebuild_filter,
            "rebuild_correlations": rebuild_correlations,
            "deterministic": deterministic,
            "target_strategy": target_strategy,
            "zinb_mode": zinb_mode,
            "market": market,
        },
    )
    click.echo(
        f"meditate starting: league={league} force={force} "
        f"rebuild_filter={rebuild_filter} rebuild_correlations={rebuild_correlations} "
        f"deterministic={deterministic}"
    )
    if not deterministic:
        np.random.seed(_RNG_SEED)

    # Per-cell ship config (data/ship_config.json) governs which markets train
    # with which strategy and which are withheld (skipped + pruned). Validated
    # here so a bad entry fails before the expensive gamelog loads below.
    # Deterministic A/B runs ignore it: they target an explicit --market with an
    # explicit --target-strategy and must never mutate production pickles.
    # See docs/operation_ship_75.md "Tier-1 supersession" for the ship mechanism.
    ship_config = {} if deterministic else load_ship_config()

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
                log.info("Reset filter", extra={"league": lg, "market": mk})
        # In deterministic mode the in-memory clear still propagates (so this
        # run sees the reset), but skip persisting to feature_filter.json —
        # deterministic runs use crippled hyperparameters and must never
        # mutate production config.
        if not deterministic:
            with open(ff_path, "w") as fh:
                json.dump(ff, fh, indent=4)
        # Reload module-level feature_filter so this run sees the change
        from sportstradamus import helpers as _hp

        _hp.feature_filter.clear()
        _hp.feature_filter.update(ff)

    nba = StatsNBA()
    nfl = StatsNFL()
    wnba = StatsWNBA()

    stat_structs = {}

    if (
        league == "All" and datetime.today().date() > (nba.season_start - timedelta(days=7))
    ) or league == "NBA":
        click.echo("[NBA] loading cached gamelogs...")
        nba.load()
        click.echo("[NBA] updating from league API (this hits stats.nba.com - may take 30-60s)...")
        nba.update()
        stat_structs.update({"NBA": nba})
    if (
        league == "All" and datetime.today().date() > (nfl.season_start - timedelta(days=7))
    ) or league == "NFL":
        click.echo("[NFL] loading cached gamelogs...")
        nfl.load()
        click.echo("[NFL] updating from league API...")
        nfl.update()
        stat_structs.update({"NFL": nfl})
    if (
        league == "All" and datetime.today().date() > (wnba.season_start - timedelta(days=7))
    ) or league == "WNBA":
        click.echo("[WNBA] loading cached gamelogs...")
        wnba.load()
        click.echo("[WNBA] updating from league API...")
        wnba.update()
        stat_structs.update({"WNBA": wnba})

    active_markets = dict(ALL_MARKETS)
    if league != "All":
        active_markets = {league: ALL_MARKETS[league]}
    active_markets = select_markets(active_markets, market)

    if rebuild_correlations:
        for lg in active_markets:
            stat_data = stat_structs.get(lg)
            if stat_data is None:
                continue
            stat_data.update_player_comps()
            correlate(lg, stat_data, force)
        return

    archive = Archive()

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
            cell_strategy = resolve_cell_strategy(lg, market, target_strategy, ship_config)
            if cell_strategy == WITHHELD:
                prune_model_pickle(lg, market)
                click.echo(f"[{lg}] {market}: withheld — pruned pickle, skipped training")
                continue
            cell_zinb_mode = resolve_cell_zinb_mode(lg, market, zinb_mode, ship_config)
            train_market(
                lg,
                market,
                stat_data,
                force,
                rebuild_filter,
                archive,
                league_start_date,
                deterministic=deterministic,
                target_strategy=cell_strategy,
                zinb_mode=cell_zinb_mode,
            )
