"""CLI entry point: meditate command orchestrates per-league, per-market training."""

import importlib.resources as pkg_resources
import json
import warnings

import click
import numpy as np
import pandas as pd

from sportstradamus import data
from sportstradamus.helpers import (
    LazyArchive,
    book_weights,
    feature_filter,
    get_logger,
    odds_budget,
)
from sportstradamus.helpers.io import MODEL_STATS_PATH, prune_model_pickle
from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA
from sportstradamus.training import baselines, calibration
from sportstradamus.training.calibration import fit_book_weights
from sportstradamus.training.correlate import correlate
from sportstradamus.training.markets import ALL_MARKETS, select_markets
from sportstradamus.training.pipeline import (
    _FORCEABLE_DISTS,
    DIST_TRAINING_LOSS_CHOICES,
    LOSS_AUTO,
    train_market,
)
from sportstradamus.training.ship_config import (
    SKEW_NORMAL_DIST,
    STAT_META_PATH,
    TARGET_NORM_NONE,
    WITHHELD,
    ShipConfig,
    load_ship_config,
    load_stat_meta,
    resolve_cell_target_normalization,
    resolve_flag_target_normalization,
)

warnings.simplefilter("ignore", UserWarning)
np.seterr(divide="ignore", invalid="ignore")

# Reproducibility seed for non-deterministic training runs. Not used under
# --deterministic (which pins RNGs via seed_everything with a fixed value).
_RNG_SEED: int = 69


def _enforce_ship_gate(
    active_markets: dict[str, list[str]],
    ship_config: ShipConfig,
    loaded_leagues: set[str],
    log,
) -> int:
    """Dark-out every served cell whose latest offline gates fail.

    A cell may serve only when its ``model_stats`` row has ``ship == True``
    (all five gates; ``training.scorecard``). ``report()`` writes each cell's
    fresh gates during training, so this post-loop pass prunes the production
    pickle of any served cell that came back ``ship == False`` — the same
    dark-out a ``withheld`` cell gets, so inference skips the market. Scoped to
    leagues actually loaded this run and to cells served on the branch
    (``ship_config`` value other than ``WITHHELD``); an empty ``ship_config``
    (``--deterministic`` or the integration smoke test) serves nothing, so this
    no-ops.
    """
    if not MODEL_STATS_PATH.is_file():
        return 0
    stats = pd.read_parquet(
        MODEL_STATS_PATH, columns=["league", "market", "ship", "g4_pass", "g4_pit_ks"]
    )
    failed = stats["ship"].eq(False).fillna(False).astype(bool)
    failing = {(r.league, r.market): r for r in stats[failed].itertuples(index=False)}
    pruned = 0
    for lg, markets in active_markets.items():
        if lg not in loaded_leagues:
            continue
        for market in markets:
            if ship_config.get(lg, {}).get(market, WITHHELD) == WITHHELD:
                continue
            row = failing.get((lg, market))
            if row is None or not prune_model_pickle(lg, market):
                continue
            pruned += 1
            click.echo(
                f"DEMOTE [{lg}] {market}: ship=False "
                f"(g4_pass={row.g4_pass}, pit_ks={row.g4_pit_ks:.3f}) — pruned pickle"
            )
            log.warning("ship-gate demote", extra={"league": lg, "market": market})
    return pruned


def _resolve_cell_knob(stat_meta_full, lg, market, key, default, flag_value):
    """Per-cell training knob from stat_meta, overridden run-wide by an explicit (non-``auto``) flag.

    Shared by the blending-loss and HP-selection axes: ``flag_value == LOSS_AUTO`` honors each
    cell's persisted ``key`` (so the production cron reproduces a cell's shipped config); an
    explicit slug forces every cell for a one-shot A/B.
    """
    cell_value = stat_meta_full.get(lg, {}).get(market, {}).get(key, default)
    return cell_value if flag_value == LOSS_AUTO else flag_value


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
    "--target-normalization",
    type=click.Choice([LOSS_AUTO, *sorted(baselines.TARGET_NORMALIZATION_SLUGS)]),
    default=LOSS_AUTO,
    show_default=True,
    help=(
        "Target-normalization transform for SkewNormal markets. 'auto' (default) "
        "honors each cell's stat_meta target_normalization; an explicit slug "
        "overrides every selected cell, in a real run or under --deterministic — "
        "an Operation Ship 75 search axis and how to confirm a candidate "
        "normalization before persisting it to stat_meta.json."
    ),
)
@click.option(
    "--zinb-mode",
    type=click.Choice([LOSS_AUTO, "joint", "hurdle"]),
    default=LOSS_AUTO,
    show_default=True,
    help=(
        "Model architecture for ZINB markets. 'auto' (default) honors each "
        "cell's stat_meta zinb_mode (else 'joint'); an explicit slug overrides "
        "every cell. 'joint' is the legacy jointly-fit LightGBMLSS ZINB. "
        "'hurdle' uses the two-stage HurdleZINB (calibrated zero classifier + "
        "NegBin on positives + derived-pi gate; see "
        "docs/OVERCONFIDENCE_INVESTIGATION.md §2)."
    ),
)
@click.option(
    "--dist-training-loss",
    type=click.Choice(list(DIST_TRAINING_LOSS_CHOICES)),
    default=LOSS_AUTO,
    show_default=True,
    help=(
        "Training loss for the LightGBMLSS distribution. 'auto' (default) keeps the "
        "per-family production loss — crps for SkewNormal, nll for the count branch. "
        "'nll'/'crps' override every family; an Operation Ship 75 search axis."
    ),
)
@click.option(
    "--dist",
    type=click.Choice([LOSS_AUTO, *sorted(_FORCEABLE_DISTS)]),
    default=LOSS_AUTO,
    show_default=True,
    help=(
        "Training distribution family. 'auto' (default) honors each cell's stat_meta dist "
        "(else the data-driven mean>=2 / zero-rate pick); an explicit family (DPO / NegBin / "
        "ZINB / SkewNormal) overrides every cell — a WS-3 family sweep axis."
    ),
)
@click.option(
    "--blending-loss-fn",
    type=click.Choice([LOSS_AUTO, *sorted(calibration.BLENDING_SLUGS)]),
    default=LOSS_AUTO,
    show_default=True,
    help=(
        "Loss minimized when fitting the model↔book blend weight. 'auto' (default) honors each "
        "cell's stat_meta blending; an explicit slug overrides every cell. An Operation Ship 75 "
        "search axis — choices grow with calibration.BLENDING_SLUGS as new blend objectives ship."
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
        "fall back to --target-normalization; non-SkewNormal cells fall back to "
        "TARGET_NORM_NONE (count-branch ignores the slug). Lets internal "
        "projection markets (NFL attempts/carries/targets) train so their "
        "pickles feed proj_* features into downstream training matrices."
    ),
)
@click.option(
    "--stabilization",
    type=click.Choice(["None", "MAD", "L2"]),
    default="None",
    show_default=True,
    help=(
        "Per-distribution-parameter gradient stabilization for LightGBMLSS. 'None' (default) "
        "is current production. 'MAD'/'L2' damp the scale-head's large/outlier gradients (the "
        "only in-API per-parameter sigma knob); an Operation Ship 75 calibration search axis."
    ),
)
@click.option(
    "--hpo-selection",
    type=click.Choice([LOSS_AUTO, "loss", "calibrated"]),
    default=LOSS_AUTO,
    show_default=True,
    help=(
        "Optuna trial-selection rule for SkewNormal cells. 'auto' (default) honors each cell's "
        "stat_meta hpo_selection (else 'loss'); an explicit slug overrides every cell. 'loss' picks "
        "the lowest CV CRPS; 'calibrated' re-ranks the top trials by validation PIT-KS and picks the "
        "sharpest that clears the Gate-4 threshold (sharpness subject to calibration); a Ship 75 axis."
    ),
)
@click.option(
    "--count-dispersion-objective",
    type=click.Choice([LOSS_AUTO, "crps", "pit_ks"]),
    default=LOSS_AUTO,
    show_default=True,
    help=(
        "Objective the count branch (NegBin/ZINB/Gamma) minimizes when fitting the dispersion "
        "scale. 'auto' (default) honors each cell's stat_meta count_dispersion_objective (else "
        "'crps', current production); an explicit 'crps'/'pit_ks' overrides every cell. 'pit_ks' "
        "targets the served Gate-4 randomized-PIT KS directly, mirroring the SkewNormal branch; a "
        "Ship 75 axis."
    ),
)
@click.option(
    "--sn-param",
    type=click.Choice([LOSS_AUTO, "direct", "centered"]),
    default=LOSS_AUTO,
    show_default=True,
    help=(
        "SkewNormal parametrization. 'auto' (default) honors each cell's stat_meta sn_param "
        "(else 'direct', current production); 'centered' boosts (mean, sd, gamma1) heads and "
        "re-emits direct params at predict — zero serving delta; a WS-3 family sweep axis."
    ),
)
@click.option(
    "--matrix-only/--no-matrix-only",
    default=False,
    help=(
        "Assemble and persist each cell's training matrix (and comps), then stop before "
        "distribution selection and model training. Warms data/training_data/ without training; "
        "pair with --bypass-withholding to reach withheld cells."
    ),
)
def meditate(
    force,
    league,
    rebuild_correlations,
    log_level,
    deterministic,
    target_normalization,
    zinb_mode,
    dist_training_loss,
    dist,
    blending_loss_fn,
    market,
    branch,
    bypass_withholding,
    stabilization,
    hpo_selection,
    count_dispersion_objective,
    sn_param,
    matrix_only,
):
    """Train or retrain LightGBMLSS models for each configured market."""
    # style: allow-complexity — meditate entrypoint: a flat training pipeline
    # (per-league load/update, book-weight fit, then the per-league/per-market
    # train loop with ship-config resolution). The residual CC is sequential
    # stages plus per-league/-cell guards, not nested logic.
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
            "target_normalization": target_normalization,
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
    # --market with an explicit --target-normalization and must never mutate
    # production pickles. See docs/gbdt_mean_regression_plan.md "Ship
    # mechanism — per-cell strategy".
    ship_config = {} if deterministic else load_ship_config(branch=branch)
    # Raw stat_meta carries the per-cell ``posthoc`` slug (read in the market
    # loop) and the ``dist`` field that ``load_ship_config`` collapses out
    # (used by --bypass-withholding to pick a branch-appropriate normalization).
    stat_meta_full = load_stat_meta(STAT_META_PATH)

    stat_structs = {}
    for lg_name, cls in (
        ("NBA", StatsNBA),
        ("NFL", StatsNFL),
        ("WNBA", StatsWNBA),
        ("MLB", StatsMLB),
        ("NHL", StatsNHL),
    ):
        if league not in ("All", lg_name):
            continue
        struct = cls()
        if league == "All" and not odds_budget.league_is_live(lg_name, struct.season_start):
            continue
        click.echo(f"[{lg_name}] loading cached gamelogs...")
        struct.load()
        if not deterministic:
            click.echo(f"[{lg_name}] updating from league API...")
            struct.update()
        stat_structs[lg_name] = struct

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
            correlate(lg, stat_data, force=force)
        return

    # LazyArchive defers the DuckDB connection until first attribute access. A --deterministic run
    # trains from the cached matrix and never touches the archive (_step_init_market skips the
    # archive-hitting book-weight fit), so it takes no write-lock and can run alongside other archive
    # jobs; a real run hits fit_book_weights immediately and connects exactly as before.
    archive = LazyArchive()

    for lg, markets in active_markets.items():
        stat_data = stat_structs.get(lg)
        if stat_data is None:
            continue

        # --deterministic freezes the training matrix to the cached parquet
        # (new_M empty), so book weights, player comps, and correlation matrices
        # are never consumed by the train — skip the whole per-league setup
        # (including correlate's per-run rebuild). Only trim_gamelog, which
        # yields league_start_date, is still needed.
        if not deterministic:
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
            if not matrix_only:
                correlate(lg, stat_data, force=force)

        league_start_date = stat_data.trim_gamelog()

        for market in markets:
            cell_target_norm = resolve_cell_target_normalization(
                lg, market, target_normalization, ship_config
            )
            if cell_target_norm == WITHHELD:
                if bypass_withholding:
                    cell_dist = stat_meta_full.get(lg, {}).get(market, {}).get("dist")
                    # SkewNormal needs a real strategy slug; count-branch
                    # families (ZINB/NegBin/Gamma/ZAGamma) ignore the slug,
                    # so TARGET_NORM_NONE is fine and the next clause will
                    # substitute the run-wide default for the pipeline call.
                    cell_target_norm = (
                        resolve_flag_target_normalization(target_normalization)
                        if cell_dist == SKEW_NORMAL_DIST
                        else TARGET_NORM_NONE
                    )
                    click.echo(
                        f"[{lg}] {market}: BYPASS withhold "
                        f"(dist={cell_dist!r}, strategy={cell_target_norm!r})"
                    )
                else:
                    prune_model_pickle(lg, market)
                    click.echo(f"[{lg}] {market}: withheld — pruned pickle, skipped training")
                    continue
            # TARGET_NORM_NONE marks count-branch cells that don't opt into a
            # SkewNormal strategy slug. The pipeline's count branch ignores
            # the slug anyway, so substitute the CLI default — the run-wide
            # target_normalization — to keep baselines.get_target_normalization() satisfied.
            if cell_target_norm == TARGET_NORM_NONE:
                cell_target_norm = resolve_flag_target_normalization(target_normalization)
            cell_posthoc = stat_meta_full.get(lg, {}).get(market, {}).get("posthoc", "none")
            # Each search-axis flag (blending, hpo_selection, count_dispersion_objective):
            # 'auto' leaves each cell on its configured stat_meta knob; an explicit slug overrides.
            cell_blending = _resolve_cell_knob(
                stat_meta_full,
                lg,
                market,
                "blending",
                calibration.DEFAULT_BLENDING,
                blending_loss_fn,
            )
            cell_hpo_selection = _resolve_cell_knob(
                stat_meta_full, lg, market, "hpo_selection", "loss", hpo_selection
            )
            cell_count_dispersion = _resolve_cell_knob(
                stat_meta_full,
                lg,
                market,
                "count_dispersion_objective",
                "crps",
                count_dispersion_objective,
            )
            cell_zinb_mode = _resolve_cell_knob(
                stat_meta_full, lg, market, "zinb_mode", "joint", zinb_mode
            )
            cell_sn_param = _resolve_cell_knob(
                stat_meta_full, lg, market, "sn_param", "direct", sn_param
            )
            train_market(
                lg,
                market,
                stat_data,
                archive,
                league_start_date,
                force=force,
                deterministic=deterministic,
                target_normalization=cell_target_norm,
                posthoc_slug=cell_posthoc,
                blending=cell_blending,
                zinb_mode=cell_zinb_mode,
                dist_training_loss=dist_training_loss,
                dist=dist,
                stabilization=stabilization,
                hpo_selection=cell_hpo_selection,
                count_dispersion_objective=cell_count_dispersion,
                sn_param=cell_sn_param,
                matrix_only=matrix_only,
            )

    if not deterministic and not matrix_only:
        demoted = _enforce_ship_gate(active_markets, ship_config, set(stat_structs), log)
        if demoted:
            click.echo(f"ship-gate: pruned {demoted} served cell(s) with ship=False")
