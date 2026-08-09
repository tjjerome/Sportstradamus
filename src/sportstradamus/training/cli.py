"""CLI entry point: meditate command orchestrates per-league, per-market training."""

import faulthandler
import importlib.resources as pkg_resources
import json
import os
import warnings
from pathlib import Path

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
from sportstradamus.helpers.cli_options import LOG_LEVEL_OPTION
from sportstradamus.helpers.io import MODEL_STATS_PATH, market_file_slug, prune_model_pickle
from sportstradamus.helpers.locks import (
    ORCHESTRATED_ENV_VAR,
    TRAINING_ARTIFACTS_LOCK_FILE,
    hold_for_process,
)
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
from sportstradamus.training.posthoc import POSTHOC_SLUGS, STRUCTURAL_STAGE
from sportstradamus.training.scorecard import _GATE1_NONINF_MARGIN
from sportstradamus.training.ship_config import (
    CONTINUOUS_DISTS,
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

# A native extension can abort the interpreter outright — LightGBM's cv corrupted the heap and
# took both NFL carries confirm nominees with it — leaving no traceback in a log that captures
# stderr, which reads as a swallowed exception. The fault stack is the only evidence there is.
faulthandler.enable()

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


# The six ship-gate booleans in model_stats.parquet (nullable BooleanDtype;
# pd.NA means the scorecard never ran, which counts as not passing here).
_GATE_COLS = ("g1_pass", "g2_pass", "g3_pass", "g4_pass", "g5_pass", "g6_pass")

# brief R3: separates the 9 addressable near-miss ledger rows from the 4 structural
# failures at excess >= 0.049.
_G4_RETRY_MAX_EXCESS: float = 0.010
# brief R3: noise band on the g1 non-inferiority CI bound.
_G1_RETRY_NOISE_BAND: float = 0.002


def _gate_passed(value) -> bool:
    """NA-safe read of a nullable-boolean gate cell (``bool(pd.NA)`` raises)."""
    return pd.notna(value) and bool(value)


def _model_stats_row(lg: str, market: str) -> pd.Series | None:
    """The cell's model_stats row (fresh after train_market — report() rewrote it), or None."""
    if not MODEL_STATS_PATH.is_file():
        return None
    stats = pd.read_parquet(
        MODEL_STATS_PATH,
        columns=[
            "league",
            "market",
            "distribution",
            "ship",
            "g4_pit_ks",
            "g4_pit_ks_max",
            "g1_brier_diff_ci_hi",
            *_GATE_COLS,
        ],
    )
    rows = stats[(stats["league"] == lg) & (stats["market"] == market)]
    return None if rows.empty else rows.iloc[0]


def _g4_only_retry_wanted(
    row: pd.Series | None, cell_hpo_selection: str, cell_zinb_mode: str
) -> bool:
    """Whether a cell earns the one-shot calibrated-HPO retry.

    Fires only for cells trained under ``loss`` selection whose fresh
    model_stats row failed ship on a near-miss Gate 4, alone or with a
    within-band Gate 1 — the dispersion-calibration failures calibrated trial
    selection targets. Any other failing gate blocks the retry, g6 explicitly
    included (deferred pending the Experiment C measurement, brief R3).
    Hurdle-ZINB and Mixture are excluded: those paths have no calibrated
    trial-selection closure.
    """
    if row is None or cell_hpo_selection != "loss":
        return False
    if row["distribution"] == "Mixture":
        return False
    if row["distribution"] == "ZINB" and cell_zinb_mode == "hurdle":
        return False
    if _gate_passed(row["ship"]) or _gate_passed(row["g4_pass"]):
        return False
    if not row["g4_pit_ks"] - row["g4_pit_ks_max"] <= _G4_RETRY_MAX_EXCESS:
        return False
    return _cofailures_within_band(row)


def _cofailures_within_band(row: pd.Series) -> bool:
    """Non-g4 gate failures are admissible only as exactly {g1} inside its noise band."""
    others_failing = [c for c in _GATE_COLS if c != "g4_pass" and not _gate_passed(row[c])]
    if others_failing == ["g1_pass"]:
        return bool(row["g1_brier_diff_ci_hi"] - _GATE1_NONINF_MARGIN <= _G1_RETRY_NOISE_BAND)
    return not others_failing


def _persist_calibrated_hpo(stat_meta_full: dict, lg: str, market: str) -> None:
    """Pin ``hpo_selection="calibrated"`` into the cell's stat_meta entry, on disk and in memory.

    The weekly warm cron retrains from stat_meta; without the pin the next
    retrain would select by loss again and serve-iff-ship would prune the cell.
    """
    stat_meta_full.setdefault(lg, {}).setdefault(market, {})["hpo_selection"] = "calibrated"
    meta_path = Path(str(STAT_META_PATH))
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stat_meta_full, indent=4) + "\n")
    tmp.replace(meta_path)


def _retry_calibrated_if_g4_only(
    lg, market, stat_data, archive, league_start_date, train_kwargs, stat_meta_full, log
) -> None:
    """One-shot calibrated-HPO retry for a cell that failed ship on a near-miss
    Gate 4, alone or with a within-band Gate 1.

    Reruns ``train_market`` with the same kwargs but ``hpo_selection="calibrated"``
    and ``force=True`` (the first run consumed the new gamedays; without force the
    input-freeze short-circuit would skip the retrain). The retry's ``report()``
    overwrites the cell's model_stats row — the retry row wins. A shipped retry
    persists the knob via :func:`_persist_calibrated_hpo`; a failed one just
    echoes its failing gates. Never retries more than once.
    """
    if (
        train_kwargs["deterministic"]
        or train_kwargs["matrix_only"]
        or train_kwargs.get("artifact_output") is not None
        or train_kwargs["posthoc_slug"] in STRUCTURAL_STAGE
    ):
        return
    row = _model_stats_row(lg, market)
    if not _g4_only_retry_wanted(row, train_kwargs["hpo_selection"], train_kwargs["zinb_mode"]):
        return
    click.echo(f"[{lg}] {market}: g4-only ship fail — retrying with hpo_selection=calibrated")
    log.info("g4-only calibrated retry", extra={"league": lg, "market": market})
    train_market(
        lg,
        market,
        stat_data,
        archive,
        league_start_date,
        **(train_kwargs | {"force": True, "hpo_selection": "calibrated"}),
    )
    row = _model_stats_row(lg, market)
    if row is not None and _gate_passed(row["ship"]):
        _persist_calibrated_hpo(stat_meta_full, lg, market)
        click.echo(
            f"[{lg}] {market}: SHIPPED via calibrated HPO — persisted hpo_selection to stat_meta"
        )
        log.info("calibrated retry shipped", extra={"league": lg, "market": market})
    else:
        failed = (
            "no model_stats row"
            if row is None
            else ", ".join(c for c in _GATE_COLS if not _gate_passed(row[c]))
        )
        click.echo(f"[{lg}] {market}: calibrated retry still ship=False (failed gates: {failed})")


def _validate_mode_flags(
    *,
    deterministic: bool,
    holdout_blind: bool,
    full_rebuild: bool,
    matrix_only: bool,
    matrix_output: Path | None,
    frozen_matrix_dir: Path | None,
    artifact_output: Path | None,
    dependency_namespace: str | None,
    dependency_root: Path | None,
    bypass_withholding: bool,
) -> None:
    """Reject flag combinations that name two different run modes at once."""
    # style: allow-complexity — a flat list of independent boundary checks, no nesting;
    # splitting it would only scatter the mode contract across several helpers.
    if full_rebuild and deterministic:
        raise click.UsageError("--full-rebuild and --deterministic are different modes")
    # A full-HPO confirm has to score the real holdout; a holdout-blind production run
    # would ship a model whose gates never saw half their evaluation data.
    if holdout_blind and not deterministic:
        raise click.UsageError("--holdout-blind is a search mode; pass --deterministic")
    if full_rebuild and (not matrix_only or matrix_output is None):
        raise click.UsageError("--full-rebuild requires --matrix-only and --matrix-output")
    if frozen_matrix_dir is not None and (full_rebuild or deterministic or matrix_only):
        raise click.UsageError(
            "--frozen-matrix-dir pins each cell's training matrix for a full-HPO run; it is "
            "incompatible with --full-rebuild, --deterministic, and --matrix-only"
        )
    if artifact_output is not None and frozen_matrix_dir is None:
        raise click.UsageError("--artifact-output requires --frozen-matrix-dir")
    if artifact_output is not None and (dependency_namespace is None or not bypass_withholding):
        raise click.UsageError(
            "--artifact-output requires --dependency-namespace and --bypass-withholding"
        )
    if dependency_namespace is not None and artifact_output is None and not full_rebuild:
        raise click.UsageError(
            "--dependency-namespace requires --artifact-output or --full-rebuild"
        )
    if full_rebuild and dependency_namespace is not None and dependency_root is None:
        raise click.UsageError(
            "--dependency-namespace on a full rebuild requires --dependency-root"
        )


def _lock_production_artifacts(
    league: str,
    market: str | None,
    *,
    deterministic: bool,
    matrix_only: bool,
    artifact_output: Path | None,
) -> None:
    """Claim the production artifact set for this run, or refuse to start.

    A confirm/supersede walk snapshots and restores that set cell by cell, on the
    assumption that only its own retrain writes it in between; a second writer breaks
    that and tears a model pickle away from its test-set CSV. The three sandbox modes
    write elsewhere, and an orchestrated run is the lock holder's own work.
    """
    if deterministic or matrix_only or artifact_output or os.environ.get(ORCHESTRATED_ENV_VAR):
        return
    try:
        hold_for_process(
            TRAINING_ARTIFACTS_LOCK_FILE,
            timeout_s=0,
            label=f"meditate league={league} market={market or 'all'}",
            contention_hint=(
                "Wait for it to finish, or use a sandbox mode "
                "(--deterministic / --artifact-output)."
            ),
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


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
@LOG_LEVEL_OPTION
@click.option(
    "--deterministic/--no-deterministic",
    default=False,
    hidden=True,
    help=(
        "DEBUG/EVAL ONLY: pin all RNGs, use fixed fast hyperparameters, and "
        "freeze input to the cached parquet so runs are bit-identical for the "
        "compression eval harness. NEVER publish a model trained with this "
        "flag — it is deliberately low quality."
    ),
)
@click.option(
    "--holdout-blind/--no-holdout-blind",
    default=False,
    hidden=True,
    help=(
        "SEARCH ONLY: drop the held-out rows from the run entirely and score on "
        "player-disjoint cross-fit folds of validation, so a recipe search never sees "
        "the frame its ship gate is scored on. Requires --deterministic."
    ),
)
@click.option(
    "--target-normalization",
    type=click.Choice([LOSS_AUTO, *sorted(baselines.TARGET_NORMALIZATION_SLUGS)]),
    default=LOSS_AUTO,
    show_default=True,
    hidden=True,
    help=(
        "Target-normalization transform for SkewNormal markets. 'auto' (default) "
        "honors each cell's stat_meta target_normalization; an explicit slug "
        "overrides every selected cell, in a real run or under --deterministic — "
        "an Operation Ship 75 search axis and how to confirm a candidate "
        "normalization before persisting it to stat_meta.json."
    ),
)
@click.option(
    "--posthoc",
    type=click.Choice([LOSS_AUTO, *sorted(POSTHOC_SLUGS)]),
    default=LOSS_AUTO,
    show_default=True,
    hidden=True,
    help=(
        "Per-cell calibration-method selector (single-valued, mutually exclusive). 'auto' "
        "(default) honors each cell's stat_meta posthoc; an explicit slug overrides every "
        "selected cell, in a real run or under --deterministic. A light corrector slug "
        "adjusts the decoded mean or over-probability after the distribution is formed; a "
        "structural slug (role-position two-part / affine group-conditional CDF) selects "
        "that method's own calibration stage instead."
    ),
)
@click.option(
    "--zinb-mode",
    type=click.Choice([LOSS_AUTO, "joint", "hurdle"]),
    default=LOSS_AUTO,
    show_default=True,
    hidden=True,
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
    hidden=True,
    help=(
        "Training loss for the LightGBMLSS distribution. 'auto' (default) honors each "
        "cell's persisted stat_meta dist_training_loss, falling back to the per-family "
        "production loss — crps for SkewNormal, nll for the count branch. "
        "'nll'/'crps' override every family; an Operation Ship 75 search axis."
    ),
)
@click.option(
    "--dist",
    type=click.Choice([LOSS_AUTO, *sorted(_FORCEABLE_DISTS)]),
    default=LOSS_AUTO,
    show_default=True,
    hidden=True,
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
    hidden=True,
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
    hidden=True,
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
    hidden=True,
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
    hidden=True,
    help=(
        "Optuna trial-selection rule for SkewNormal and LSS count/Gamma cells (hurdle-ZINB has no "
        "Optuna search). 'auto' (default) honors each cell's stat_meta hpo_selection (else 'loss'); "
        "an explicit slug overrides every cell. 'loss' picks the lowest CV loss; 'calibrated' "
        "re-ranks the top trials by validation PIT-KS and picks the sharpest that clears the Gate-4 "
        "threshold. A cell failing ship on Gate 4 alone under 'loss' is retrained once with "
        "'calibrated', and the knob persists to stat_meta when the retry ships; a Ship 75 axis."
    ),
)
@click.option(
    "--count-dispersion-objective",
    type=click.Choice([LOSS_AUTO, "crps", "pit_ks"]),
    default=LOSS_AUTO,
    show_default=True,
    hidden=True,
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
    hidden=True,
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
@click.option(
    "--full-rebuild/--no-full-rebuild",
    default=False,
    hidden=True,
    help=(
        "Rebuild from locally cached raw inputs without reading/appending the canonical matrix. "
        "Requires --matrix-only and --matrix-output."
    ),
)
@click.option(
    "--matrix-output",
    type=click.Path(path_type=Path, file_okay=False),
    hidden=True,
    help="Quarantine directory for full-rebuild parquet and manifest outputs.",
)
@click.option(
    "--dependency-root",
    type=click.Path(path_type=Path, file_okay=False),
    hidden=True,
    help="Root containing a versioned model-dependency namespace.",
)
@click.option(
    "--frozen-matrix-dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    hidden=True,
    help="Directory of lineage-validated matrices to train without rebuilding or rewriting them.",
)
@click.option(
    "--artifact-output",
    type=click.Path(path_type=Path, file_okay=False),
    hidden=True,
    help="Isolated directory for model and test-set artifacts from frozen-matrix training.",
)
@click.option(
    "--dependency-namespace",
    hidden=True,
    help=(
        "Dependency identity namespace stamped onto isolated model artifacts, or selected "
        "from --dependency-root during a full rebuild."
    ),
)
def meditate(
    force,
    league,
    rebuild_correlations,
    log_level,
    deterministic,
    holdout_blind,
    target_normalization,
    posthoc,
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
    full_rebuild,
    matrix_output,
    dependency_root,
    frozen_matrix_dir,
    artifact_output,
    dependency_namespace,
):
    """Train or retrain LightGBMLSS models for each configured market."""
    # style: allow-complexity — meditate entrypoint: a flat training pipeline
    # (per-league load/update, book-weight fit, then the per-league/per-market
    # train loop with ship-config resolution). The residual CC is sequential
    # stages plus per-league/-cell guards, not nested logic.
    # style: allow-length — grandfathered CLI orchestrator; the --posthoc knob adds one
    # param line. Extracting _step_ helpers is a separate refactor, out of this change's scope.
    _validate_mode_flags(
        deterministic=deterministic,
        holdout_blind=holdout_blind,
        full_rebuild=full_rebuild,
        matrix_only=matrix_only,
        matrix_output=matrix_output,
        frozen_matrix_dir=frozen_matrix_dir,
        artifact_output=artifact_output,
        dependency_namespace=dependency_namespace,
        dependency_root=dependency_root,
        bypass_withholding=bypass_withholding,
    )
    _lock_production_artifacts(
        league,
        market,
        deterministic=deterministic,
        matrix_only=matrix_only,
        artifact_output=artifact_output,
    )
    # --deterministic implies --force: the input-freeze (new_M = empty) otherwise
    # short-circuits train_market when a prior model pickle exists, which is precisely
    # when the eval harness needs a fresh deterministic rebuild. See
    # docs/gbdt_mean_regression_plan.md "Bug to fix" note.
    force = force or deterministic
    if full_rebuild:
        os.environ["SPORTSTRADAMUS_ARCHIVE_READ_ONLY"] = "1"
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
            "posthoc": posthoc,
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
    # Raw stat_meta carries the per-cell ``posthoc`` slug (the calibration-method
    # selector, read in the market loop) and the ``dist`` field that
    # ``load_ship_config`` collapses out (used by --bypass-withholding to pick a
    # branch-appropriate normalization).
    stat_meta_full = load_stat_meta(STAT_META_PATH)

    active_markets = dict(ALL_MARKETS)
    if league != "All":
        active_markets = {league: ALL_MARKETS[league]}
    active_markets = select_markets(active_markets, market)
    # A structural calibration method reshapes the target/CDF during training, so it
    # cannot run on the matrix-only or correlation-rebuild paths that stop before the fit.
    structural_cells = {
        (lg, cell_market)
        for lg, markets in active_markets.items()
        for cell_market in markets
        if _resolve_cell_knob(stat_meta_full, lg, cell_market, "posthoc", "none", posthoc)
        in STRUCTURAL_STAGE
    }
    if structural_cells and (rebuild_correlations or matrix_only):
        raise click.UsageError(
            "a structural calibration method requires model training; "
            "--rebuild-correlations and --matrix-only are not valid"
        )

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
        if cls is StatsMLB and (deterministic or full_rebuild or frozen_matrix_dir is not None):
            # Frozen/cold builds consume cached historical inputs only. Probable
            # pitchers describe the live upcoming slate and must not introduce
            # a network dependency before the cached MLB gamelog is loaded.
            struct = cls(load_live_pitchers=False)
        else:
            struct = cls()
        if league == "All" and not odds_budget.league_is_live(lg_name, struct.season_start):
            continue
        click.echo(f"[{lg_name}] loading cached gamelogs...")
        struct.load()
        if not deterministic and not full_rebuild and frozen_matrix_dir is None:
            click.echo(f"[{lg_name}] updating from league API...")
            struct.update()
        stat_structs[lg_name] = struct

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
        if not deterministic and not full_rebuild and frozen_matrix_dir is None:
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
            cell_meta = stat_meta_full.get(lg, {}).get(market, {})
            cell_dist = cell_meta.get("dist") if dist == LOSS_AUTO else dist
            cell_target_norm = resolve_cell_target_normalization(
                lg, market, target_normalization, ship_config
            )
            if cell_target_norm == WITHHELD:
                if bypass_withholding:
                    # Continuous families need a transform; count families carry
                    # canonical ``none`` because they never run normalization.
                    cell_target_norm = (
                        resolve_flag_target_normalization(target_normalization)
                        if cell_dist in CONTINUOUS_DISTS
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
            if cell_dist is not None and cell_dist not in CONTINUOUS_DISTS:
                cell_target_norm = TARGET_NORM_NONE
            cell_posthoc = _resolve_cell_knob(
                stat_meta_full, lg, market, "posthoc", "none", posthoc
            )
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
            cell_dist_training_loss = _resolve_cell_knob(
                stat_meta_full, lg, market, "dist_training_loss", LOSS_AUTO, dist_training_loss
            )
            # Kwargs as a dict so the g4-only calibrated retry below reruns the
            # exact same call with only force/hpo_selection overridden.
            train_kwargs = {
                "force": force,
                "deterministic": deterministic,
                "target_normalization": cell_target_norm,
                "posthoc_slug": cell_posthoc,
                "blending": cell_blending,
                "zinb_mode": cell_zinb_mode,
                "dist_training_loss": cell_dist_training_loss,
                "dist": dist,
                "stabilization": stabilization,
                "hpo_selection": cell_hpo_selection,
                "count_dispersion_objective": cell_count_dispersion,
                "sn_param": cell_sn_param,
                "matrix_only": matrix_only,
                "full_rebuild": full_rebuild,
                "matrix_output": matrix_output,
                "dependency_root": dependency_root,
                "matrix_input": (
                    frozen_matrix_dir / f"{market_file_slug(lg, market)}.parquet"
                    if frozen_matrix_dir is not None
                    else None
                ),
                "artifact_output": artifact_output,
                "dependency_namespace": dependency_namespace,
                "holdout_blind": holdout_blind,
            }
            train_market(lg, market, stat_data, archive, league_start_date, **train_kwargs)
            _retry_calibrated_if_g4_only(
                lg, market, stat_data, archive, league_start_date, train_kwargs, stat_meta_full, log
            )

    if not deterministic and not matrix_only and artifact_output is None:
        demoted = _enforce_ship_gate(active_markets, ship_config, set(stat_structs), log)
        if demoted:
            click.echo(f"ship-gate: pruned {demoted} served cell(s) with ship=False")
