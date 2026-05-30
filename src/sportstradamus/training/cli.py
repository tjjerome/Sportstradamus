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
    SKEW_NORMAL_DIST,
    STAT_META_PATH,
    STRATEGY_NONE,
    WITHHELD,
    load_ship_config,
    load_stat_meta,
    resolve_cell_strategy,
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
    type=click.Choice(["joint", "hurdle"]),
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
@click.option(
    "--branch",
    type=click.Choice(["devel", "main"]),
    default="devel",
    show_default=True,
    help=(
        "Which release branch's ship policy to honor. 'devel' (default, "
        "matches the production server) trains every cell with "
        "shipped in {devel, main}. 'main' only trains cells with "
        "shipped=main. See data/config/stat_meta.json."
    ),
)
@click.option(
    "--bypass-withholding/--no-bypass-withholding",
    default=False,
    help=(
        "One-shot escape from the ship gate: train EVERY market in the "
        "registry regardless of shipped status. Withheld SkewNormal cells "
        "fall back to --target-strategy; non-SkewNormal cells fall back to "
        "STRATEGY_NONE (count-branch ignores the slug). Lets internal "
        "projection markets (NFL attempts/carries/targets) train so their "
        "pickles feed proj_* features into downstream training matrices."
    ),
)
def meditate(
    force,
    league,
    rebuild_correlations,
    log_level,
    deterministic,
    target_strategy,
    zinb_mode,
    market,
    branch,
    bypass_withholding,
):
    """Train or retrain LightGBMLSS models for each configured market."""
    # --deterministic implies --force: the input-freeze (new_M = empty)
    # otherwise short-circuits train_market when a prior model pickle exists,
    # which is precisely when the eval harness needs a fresh deterministic
    # rebuild. See docs/gbdt_mean_regression_plan.md "Bug to fix" note.
    if deterministic and not force:
        force = True
    log = get_logger("meditate")
    log.setLevel(log_level)
    log.info(
        "meditate invoked",
        extra={
            "force": force,
            "league": league,
            "rebuild_correlations": rebuild_correlations,
            "deterministic": deterministic,
            "target_strategy": target_strategy,
            "zinb_mode": zinb_mode,
            "market": market,
            "branch": branch,
            "bypass_withholding": bypass_withholding,
        },
    )
    click.echo(
        f"meditate starting: league={league} force={force} "
        f"rebuild_correlations={rebuild_correlations} "
        f"deterministic={deterministic}"
    )
    if not deterministic:
        np.random.seed(_RNG_SEED)

    # Per-cell ship config (data/config/stat_meta.json) governs which markets
    # train with which strategy and which are withheld (skipped + pruned).
    # Validated here so a bad entry fails before the expensive gamelog loads
    # below. Deterministic A/B runs ignore it: they target an explicit
    # --market with an explicit --target-strategy and must never mutate
    # production pickles. See docs/gbdt_mean_regression_plan.md "Ship
    # mechanism — per-cell strategy".
    ship_config = {} if deterministic else load_ship_config(branch=branch)
    # Raw stat_meta is consulted only by --bypass-withholding to recover the
    # ``dist`` field that ``load_ship_config`` collapses out, so the bypass
    # can substitute a SkewNormal-vs-count-branch-appropriate strategy.
    stat_meta_full = load_stat_meta(STAT_META_PATH) if bypass_withholding else {}

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

        with open(pkg_resources.files(data) / "config" / "book_weights.json", "w") as outfile:
            json.dump(book_weights, outfile, indent=4)

        stat_data.update_player_comps()
        correlate(lg, stat_data, force)
        league_start_date = stat_data.trim_gamelog()

        for market in markets:
            cell_strategy = resolve_cell_strategy(lg, market, target_strategy, ship_config)
            if cell_strategy == WITHHELD:
                if bypass_withholding:
                    cell_dist = stat_meta_full.get(lg, {}).get(market, {}).get("dist")
                    # SkewNormal needs a real strategy slug; count-branch
                    # families (ZINB/NegBin/Gamma/ZAGamma) ignore the slug,
                    # so STRATEGY_NONE is fine and the next clause will
                    # substitute the run-wide default for the pipeline call.
                    cell_strategy = (
                        target_strategy if cell_dist == SKEW_NORMAL_DIST else STRATEGY_NONE
                    )
                    click.echo(
                        f"[{lg}] {market}: BYPASS withhold "
                        f"(dist={cell_dist!r}, strategy={cell_strategy!r})"
                    )
                else:
                    prune_model_pickle(lg, market)
                    click.echo(f"[{lg}] {market}: withheld — pruned pickle, skipped training")
                    continue
            # STRATEGY_NONE marks count-branch cells that don't opt into a
            # SkewNormal strategy slug. The pipeline's count branch ignores
            # the slug anyway, so substitute the CLI default — the run-wide
            # target_strategy — to keep baselines.get_strategy() satisfied.
            if cell_strategy == STRATEGY_NONE:
                cell_strategy = target_strategy
            train_market(
                lg,
                market,
                stat_data,
                force,
                archive,
                league_start_date,
                deterministic=deterministic,
                target_strategy=cell_strategy,
                zinb_mode=zinb_mode,
            )
