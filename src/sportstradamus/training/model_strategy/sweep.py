"""Per-cell deterministic strategy sweep ranked by six-gate ship slack.

Every applicable, explicitly enrolled :class:`sportstradamus.training.model_strategy.registry.StrategySpec`
contributes its categorical grid to one conditional-TPE study per cell, budgeted at
:data:`MAX_TRIALS_PER_CELL` trials. Each corner trains in a research-only namespace and is scored
through the production scorecard.

The search is **holdout-blind**: every trial runs ``meditate --holdout-blind``, which drops the ship
gate's rows from the run entirely and cross-fits the calibration head out-of-fold over the validation
frame. Selection therefore never adapts against the rows the ship decision uses. Rows carry
:data:`EVAL_SPLIT_CROSSFIT` to distinguish them from legacy rows scored on the holdout.

Canonical spec/control fingerprints make the board resumable per row rather than per cell — a
budgeted search never enumerates a whole cell — and the Optuna journal makes the study itself
resumable.

Deterministic trials rank only. ``--confirm`` walks each cell's nominees and requires a clean
full-HPO 6/6 before a withheld model can ship.
"""

import collections
import functools
import hashlib
import importlib.resources as pkg_resources
import math
import pathlib
import re
import signal
import subprocess
import time
from datetime import UTC, datetime

import click
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
import tabulate

from sportstradamus import data as _data_pkg
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training.baselines import resolve_denom_col
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.model_strategy.identity import (
    InactiveStrategyArtifactError,
    build_artifact_identity,
    validate_strategy_artifacts,
)
from sportstradamus.training.model_strategy.registry import (
    BASE_STRUCTURAL_STRATEGY,
    SWEEP_CAPABILITIES,
    CellContext,
    StrategySpec,
    artifact_namespace,
    controls_json,
    corner_fingerprint,
    distribution_class,
    get_strategy,
    meditate_command,
    parse_controls,
    strategies_for_cell,
    strategy_cli_args,
    strategy_controls,
    strategy_persistence_edits,
)
from sportstradamus.training.model_strategy.specs import SEED_CORNERS
from sportstradamus.training.model_strategy.tpe_search import (
    MAX_TRIALS_PER_CELL,
    TRAINED_TRIAL_ATTR,
    cell_study,
    early_stop_callback,
    enqueue_params,
    reachable_corners,
    suggest_corner,
)
from sportstradamus.training.posthoc import STRUCTURAL_STAGE
from sportstradamus.training.scorecard import (
    apply_thresholds,
    gate_row,
    load_test_set,
    min_gate_slack,
)
from sportstradamus.training.ship_config import (
    STAT_META_PATH,
    TARGET_NORM_NONE,
    WITHHELD,
    load_stat_meta,
)

_SHIP_PRED_COL = "Blended_EV"
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_TEST_SETS_ROOT = pathlib.Path(str(pkg_resources.files(_data_pkg) / "test_sets"))
_DETERMINISTIC_MODEL_ROOT = _REPO_ROOT / "research" / "models" / "deterministic"
_DETERMINISTIC_LOG_ROOT = _REPO_ROOT / "research" / "logs" / "deterministic"
_TRAINING_DATA_ROOT = pathlib.Path(str(pkg_resources.files(_data_pkg) / "training_data"))
# 30-minute ceiling; a deterministic meditate run is fast-HP, but cells with large datasets can
# take ~10 min, so 1800 s keeps CI from hanging without cutting off valid runs.
_MEDITATE_TRIAL_TIMEOUT_S = 1800
# A DuckDB read-write connection takes a process-exclusive lock on archive.duckdb; a meditate that
# opens the archive while a cron job holds it fails immediately with this line (DuckDB throws rather
# than blocking). A --deterministic trial trains from the cached matrix and never opens the archive,
# so this only bites a real-HPO confirm run racing a cron archive job — retry with back-off until the
# holder releases. Lives here because the confirm loop imports its meditate runner from this module.
_LOCK_ERROR_SIGNATURE = "Could not set lock on file"
# Back-off waits (seconds) between archive-lock retries; exhausting them re-raises so a genuinely
# stuck lock fails loud instead of looping forever.
_LOCK_RETRY_WAITS_S: tuple[int, ...] = (15, 30, 60, 120, 240)
# glibc's heap-corruption aborts, as emitted by malloc_printerr. A native fit can kill the
# interpreter with one of these and no traceback, and the message lands glued to the end of a
# multi-kilobyte tqdm line — so pull it out by pattern rather than leaving it in the log tail.
_NATIVE_ABORT_PATTERN = re.compile(
    r"(?:malloc|free|realloc|malloc_consolidate|munmap_chunk)\(\): [^\r\n]+"
    r"|double free or corruption[^\r\n]*"
    r"|corrupted [^\r\n]+"
)
# The six offline ship gates (value + pass); a corner ships iff all pass. Mirrors
# scorecard._SHIP_GATES — apply_thresholds sets `ship` and min_gate_slack folds in all six.
_GATES: tuple[str, ...] = ("g1", "g2", "g3", "g4", "g5", "g6")

# A retrain corner can error mid-sweep — an invalid family/loss combo for the cell's data-driven
# dist (e.g. a SkewNormal `dist_training_loss=crps` corner on a low-mean cell the pipeline trains as
# ZINB), a timeout, or a numerical blow-up. A sweep over hundreds of corners records the bad one
# non-shipping and continues rather than losing the whole board; -inf slack sorts it last and keeps
# it out of every ship/candidate filter.
_FAILED_CORNER_SLACK: float = float("-inf")

# `--dist-class all` sweeps every registered family class.
_DIST_CLASSES: tuple[str, ...] = ("continuous", "count")
_DIST_CLASS_ALL = "all"

# Deterministic trials rank on the holdout-blind cross-fit frame (meditate --holdout-blind), never
# on the rows the ship gate scores. The marker rides every row so the 1161 legacy rows — same gate
# columns, different frame — cannot be mistaken for it: they carry <NA> and are admitted neither to
# the corner cache nor to confirm's nomination lane.
EVAL_SPLIT_CROSSFIT: str = "crossfit_validation"


# Common human-readable controls ride beside the canonical ``controls_json``. A future strategy can
# add a control without changing this schema: the JSON remains authoritative even when the control
# has no convenience column here.
_AXIS_COLUMNS: list[str] = [
    "dist",
    "normalization",
    "dist_training_loss",
    "sn_param",
    "zinb_mode",
    "count_dispersion_objective",
    "blending_loss_fn",
    "hpo_selection",
    "stabilization",
    "posthoc",
]
_BOARD_COLUMNS: list[str] = [
    "league",
    "market",
    "family",
    "strategy_slug",
    "structural_strategy",
    "strategy_signature",
    "strategy_implementation_version",
    "artifact_schema_version",
    "strategy_status",
    "controls_json",
    "corner_fingerprint",
    *_AXIS_COLUMNS,
    "slack",
    "ships",
    "g1_pass",
    "g1_brier_diff_ci_hi",
    "g1_brier_skill",
    "g2_pass",
    "g2_star_z",
    "g3_pass",
    "g3_bench_z",
    "g4_pass",
    "g4_pit_ks",
    "g4_pit_ks_max",
    "g5_pass",
    "g5_ece_debiased",
    "g6_pass",
    "central50_coverage",
    "dispersion_cal",
    "skew_cal",
    "n",
    "matrix_hash",
    "split_fingerprint",
    "eval_split",
    "swept_at",
    "code_rev",
]

# Board identity columns a row must carry nonmissing before resume will reuse its verdict. Always a
# per-row contract; a budgeted search never enumerates a whole cell, so there is no cell-level one.
_IDENTITY_COLUMNS: tuple[str, ...] = (
    "corner_fingerprint",
    "family",
    "strategy_slug",
    "structural_strategy",
    "strategy_signature",
    "strategy_implementation_version",
    "artifact_schema_version",
    "strategy_status",
    "controls_json",
    "matrix_hash",
)

# Default output path for the living board — both CLI modes write here unless --out overrides.
STRATEGY_RESEARCH_BOARD: pathlib.Path = pathlib.Path(
    str(pkg_resources.files(_data_pkg) / "research" / "strategy_research_board.csv")
)


def _cell_context(league: str, market: str) -> CellContext:
    """Build the registry context for one stat-meta cell."""
    meta = load_stat_meta(pathlib.Path(str(STAT_META_PATH)))
    dist = meta.get(league, {}).get(market, {}).get("dist")
    if not isinstance(dist, str):
        raise click.UsageError(f"{league} {market}: missing distribution family")
    try:
        dist_class = distribution_class(dist)
    except ValueError as exc:
        raise click.UsageError(f"{league} {market}: dist {dist!r} is not a swept family") from exc
    data_columns, matrix_sha256, target_is_integer, global_mean = _training_matrix_contract(
        league, market
    )
    return CellContext(
        league=league,
        market=market,
        distribution=dist,
        distribution_class=dist_class,
        data_columns=data_columns,
        matrix_sha256=matrix_sha256,
        target_is_integer=target_is_integer,
        global_mean=global_mean,
    )


def _training_matrix_path(league: str, market: str) -> pathlib.Path:
    return _TRAINING_DATA_ROOT / f"{market_file_slug(league, market)}.parquet"


@functools.lru_cache(maxsize=256)
def _read_training_matrix_contract(
    path_text: str, mtime_ns: int, size: int
) -> tuple[frozenset[str], str, bool, float]:
    """Read a cached matrix's schema and target moments, and stream its exact bytes into SHA256.

    The target's integer lattice and mean drive family admission (registry.Applicability), so only
    the ``Result`` column is materialized — never the full matrix.
    """
    del mtime_ns, size  # cache-key inputs; the path contents are read below
    path = pathlib.Path(path_text)
    columns = frozenset(pq.read_schema(path).names)
    result = pd.read_parquet(path, columns=["Result"])["Result"].to_numpy(dtype=float)
    result = result[np.isfinite(result)]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return (
        columns,
        digest.hexdigest(),
        bool(np.equal(np.mod(np.unique(result), 1), 0).all()),
        float(result.mean()),
    )


def _training_matrix_contract(league: str, market: str) -> tuple[frozenset[str], str, bool, float]:
    """Return cached-matrix schema + SHA + target moments, failing loud on an unusable input."""
    path = _training_matrix_path(league, market)
    try:
        stat = path.stat()
        return _read_training_matrix_contract(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError) as exc:
        raise click.UsageError(
            f"{league} {market}: unreadable cached training matrix {path}"
        ) from exc


def _cell_families(league: str, market: str, dist_class: str = _DIST_CLASS_ALL) -> tuple[str, ...]:
    """Registered sweep strategies applicable and explicitly enrolled for a cell.

    ``dist_class`` narrows to families of one distribution class. It filters *families*, not cells:
    with the class unlocked, a count family can win on a cell whose stat_meta names a continuous one,
    so which cells are eligible no longer follows from the recipe the cell happens to carry today.

    Structural methods are excluded outright: they derive their role support from validation row
    counts, so ``train_market`` refuses them under ``--holdout-blind`` rather than score folds that
    silently disagree about whether the method fell back. They stay selectable in production through
    the ``posthoc`` pool and keep their own preregistered evidence path; this sweep cannot rank them
    until that support is made fold-stable.
    """
    return tuple(
        spec.slug
        for spec in strategies_for_cell(
            _cell_context(league, market), required_capabilities=SWEEP_CAPABILITIES
        )
        if not spec.is_structural
        and dist_class in (_DIST_CLASS_ALL, distribution_class(spec.family))
    )


@functools.lru_cache(maxsize=1)
def _code_rev() -> str:
    """Short git SHA of the tree a board run was swept at — attribution stamped onto every row."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _decode_strategy(corner: dict[str, str]) -> str:
    """The registered continuous transform, or canonical no-transform for a count corner.

    One source of truth for both the scorecard ``decode_strategy`` and the leading token of the
    deterministic dump subdir.
    """
    return corner.get("normalization", TARGET_NORM_NONE)


def _dump_subdir(corner: dict[str, str], spec: StrategySpec) -> str:
    """Resolve the canonical deterministic namespace for one strategy corner."""
    trained_dist = corner.get("dist", spec.family)
    suffix = "_hurdle" if trained_dist == "ZINB" and corner.get("zinb_mode") == "hurdle" else ""
    return artifact_namespace(f"{_decode_strategy(corner)}{suffix}", spec)


def _dump_paths(
    league: str, market: str, corner: dict[str, str], spec: StrategySpec
) -> tuple[pathlib.Path, pathlib.Path]:
    """CSV under ``test_sets/deterministic/<subdir>/`` and pickle under ``research/models/deterministic/<subdir>/``."""
    subdir = _dump_subdir(corner, spec)
    filename = market_file_slug(league, market)
    csv = _TEST_SETS_ROOT / "deterministic" / subdir / f"{filename}.csv"
    mdl = _DETERMINISTIC_MODEL_ROOT / subdir / f"{filename}.mdl"
    return csv, mdl


def _corner_label(corner: dict[str, str]) -> str:
    """Human-scannable ``axis=value · axis=value`` for the trial's progress + log lines."""
    return " · ".join(f"{axis}={value}" for axis, value in corner.items())


def _log_path(league: str, market: str, corner: dict[str, str], spec: StrategySpec) -> pathlib.Path:
    """Per-corner meditate log under ``research/logs/deterministic/<subdir>/``.

    Keyed by the full corner so two corners sharing a dump subdir (e.g. the loss axes of one
    normalization) don't overwrite each other's log — unlike the model/CSV dump, which meditate keys
    by subdir alone and each corner therefore retrains and scores before the next overwrites it.
    """
    filename = market_file_slug(league, market)
    tag = (
        f"structural_strategy={spec.slug}"
        if spec.is_structural
        else "_".join(f"{axis}={corner[axis]}" for axis in sorted(corner))
    )
    return _DETERMINISTIC_LOG_ROOT / _dump_subdir(corner, spec) / f"{filename}__{tag}.log"


def _log_tail(path: pathlib.Path, n: int = 25) -> str:
    return "\n".join(path.read_text().splitlines()[-n:])


def _is_archive_lock_error(log_path: pathlib.Path) -> bool:
    """True iff the run's log tail shows the DuckDB archive write-lock collision (a retryable failure)."""
    return _LOCK_ERROR_SIGNATURE in _log_tail(log_path)


def _failure_reason(exc: subprocess.CalledProcessError | subprocess.TimeoutExpired) -> str:
    """Why a meditate subprocess failed, in the terms triage needs.

    A negative return code is a signal death, not an exit status: the interpreter never raised, so
    there is no traceback to read and the exception-hunting moves are all wasted. Naming the signal
    is what separates a native abort from an ordinary non-zero exit, which a caller reporting a bare
    "error" renders identically.
    """
    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    if exc.returncode >= 0:
        return f"exit {exc.returncode}"
    return f"native abort ({signal.Signals(-exc.returncode).name})"


def _run_meditate_with_lock_retry(cmd: list[str], log_path: pathlib.Path, *, timeout: int) -> None:
    """Run a ``meditate`` subprocess, capturing output to ``log_path``, retrying an archive-lock clash.

    A non-zero exit whose log shows the DuckDB write-lock collision (:data:`_LOCK_ERROR_SIGNATURE`)
    is transient — a cron archive job holds the lock — so wait per :data:`_LOCK_RETRY_WAITS_S` and
    retry (each attempt truncates the log, so a final tail reflects the last try). Every other failure
    — a real non-zero exit, a native abort, a timeout, or exhausted lock retries — surfaces its
    :func:`_failure_reason` and the log tail and re-raises, so the deterministic sweep stays fail-loud
    (the crash kills the Optuna study) and the confirm loop can turn the raise into a HELD/REVERTED
    verdict that names the cause.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n_attempts = len(_LOCK_RETRY_WAITS_S) + 1  # each wait buys one retry, plus the first attempt
    for attempt in range(n_attempts):
        with log_path.open("w") as log:
            try:
                subprocess.run(
                    cmd,
                    cwd=_REPO_ROOT,
                    check=True,
                    timeout=timeout,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                return
            except subprocess.CalledProcessError as exc:
                retries_left = attempt < len(_LOCK_RETRY_WAITS_S)
                if retries_left and _is_archive_lock_error(log_path):
                    wait = _LOCK_RETRY_WAITS_S[attempt]
                    click.echo(f"  archive write-locked; retrying in {wait}s …", err=True)
                    time.sleep(wait)
                    continue
                tail = _log_tail(log_path)
                abort = _NATIVE_ABORT_PATTERN.search(tail)
                signature = f": {abort.group()}" if abort else ""
                click.echo(
                    f"  meditate failed — {_failure_reason(exc)}{signature}"
                    f" — tail of {log_path}:\n{tail}",
                    err=True,
                )
                raise
            except subprocess.TimeoutExpired:
                click.echo(
                    f"  meditate timed out — tail of {log_path}:\n{_log_tail(log_path)}", err=True
                )
                raise


def _run_deterministic_meditate(
    league: str, market: str, corner: dict[str, str], spec: StrategySpec
) -> None:
    """Train one deterministic ``(cell, corner)`` trial via meditate.

    ``--deterministic`` pins RNGs and the fixed fast hyperparameters and dumps to the research
    sandbox (never production); ``--bypass-withholding`` lets a withheld cell train;
    ``--holdout-blind`` drops the ship gate's rows from the run entirely and cross-fits the
    calibration head over the validation frame, so the search never adapts against the rows the ship
    decision uses. Each corner axis is forwarded as its ``--flag value`` so the sweep varies it; an
    axis the family doesn't sweep is left off and meditate resolves its default. The trained model is
    a *ranking* stand-in.

    meditate's full training log is captured to a per-corner file rather than streamed, so the sweep's
    own progress and verdict stay readable; :func:`_run_meditate_with_lock_retry` surfaces a failed
    trial's tail and retries a transient archive-lock clash.
    """
    cmd = meditate_command(
        league,
        market,
        "--deterministic",
        "--bypass-withholding",
        "--holdout-blind",
    )
    cmd += strategy_cli_args(_cell_context(league, market), spec, corner)
    _run_meditate_with_lock_retry(
        cmd, _log_path(league, market, corner, spec), timeout=_MEDITATE_TRIAL_TIMEOUT_S
    )


def _score_corner(
    league: str, market: str, corner: dict[str, str], spec: StrategySpec
) -> dict[str, object]:
    """Score a trained corner's dump through the honest production gate.

    The dump's distribution is inferred from its columns (SkewNormal vs the count R/NB_P/Gate
    triple), so one gate path scores both families; ``decode_strategy`` is the swept SN
    normalization or the canonical no-transform count value. ``dispersion_cal`` / ``skew_cal`` are read
    from the pickle for context only (a count cell dumps ``skew_cal`` absent → 0.0).
    """
    csv_path, mdl_path = _dump_paths(league, market, corner, spec)
    df = load_test_set(csv_path, _SHIP_PRED_COL)
    filedict = pd.read_pickle(mdl_path)
    context = _cell_context(league, market)
    artifact_identity = validate_strategy_artifacts(
        spec,
        corner,
        df,
        filedict,
        league=league,
        market=market,
        matrix_hash=str(context.matrix_sha256),
    )
    decode = _decode_strategy(corner)
    row = apply_thresholds(
        gate_row(
            df,
            _SHIP_PRED_COL,
            league=league,
            market=market,
            strategy=decode,
            decode_strategy=decode,
        )
    )
    return {
        **corner,
        "slack": min_gate_slack(row),
        "ships": bool(row.get("ship")),
        # All six gates (value + pass) so a corner's cost on any gate — g1 Brier non-inferiority,
        # g4 PIT-KS dispersion, g5 ECE, g6 anti-shrinkage — is visible on the board.
        "g1_pass": bool(row.get("g1_pass")),
        "g1_brier_diff_ci_hi": row.get("g1_brier_diff_ci_hi"),
        "g1_brier_skill": row.get("g1_brier_skill_score"),
        "g2_pass": bool(row.get("g2_pass")),
        "g2_star_z": row.get("g2_star_z"),
        "g3_pass": bool(row.get("g3_pass")),
        "g3_bench_z": row.get("g3_bench_z"),
        "g4_pass": bool(row.get("g4_pass")),
        "g4_pit_ks": row.get("g4_pit_ks"),
        "g4_pit_ks_max": row.get("g4_pit_ks_max"),
        "g5_pass": bool(row.get("g5_pass")),
        "g5_ece_debiased": row.get("g5_ece_debiased"),
        "g6_pass": bool(row.get("g6_pass")),
        "central50_coverage": row.get("central50_coverage"),
        "dispersion_cal": filedict.get("dispersion_cal", 1.0),
        "skew_cal": filedict.get("skew_cal") or 0.0,
        "n": row.get("n_rows"),
        "matrix_hash": context.matrix_sha256,
        "split_fingerprint": artifact_identity.split_fingerprint,
        "eval_split": EVAL_SPLIT_CROSSFIT,
    }


def _failed_gates(row: object) -> list[str]:
    """The scored gates a corner fails (empty when it ships). Accepts a dict or a DataFrame row."""
    return [g for g in _GATES if not row.get(f"{g}_pass", True)]


def _verdict(row: object) -> str:
    """``SHIP`` or ``KILL: g4 g5`` naming the failing gates — the human-scannable corner outcome."""
    if row.get("ships"):
        return "SHIP"
    failed = _failed_gates(row)
    return "KILL: " + " ".join(failed) if failed else "KILL"


def _run_and_score(
    league: str, market: str, strategy_slug: str, corner: dict[str, str]
) -> dict[str, object]:
    """Train the ``corner`` retrain trial and score it; tag the row with its family.

    Calibration is not re-fit here: the dump already carries the pipeline's cross-fit
    out-of-fold calibration, and :func:`_score_corner` reads it off the dump via the production
    :func:`scorecard.gate_row` — no test re-fit. The dump is keyed by subdir, so scoring right after
    this trial's train — before the next sequential trial overwrites it — keeps the loss/mode axes
    honest without a per-corner dump path (which is also why the sweep always retrains rather than
    reusing a dump). A corner whose meditate errors or times out is caught, echoed, and returned as a
    non-shipping row (:data:`_FAILED_CORNER_SLACK`) so one bad corner never aborts the board — and,
    cached under its fingerprint, so a resumed run never retrains a reliable crash.
    """
    spec = get_strategy(strategy_slug)
    context = _cell_context(league, market)
    label = _corner_label(corner)
    identity = build_artifact_identity(
        spec.slug,
        league,
        market,
        corner,
        matrix_hash=context.matrix_sha256,
    )
    click.echo(f"  training  {label} …")
    start = time.monotonic()
    try:
        _run_deterministic_meditate(league, market, corner, spec)
        row = _score_corner(league, market, corner, spec)
    except (
        InactiveStrategyArtifactError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        click.secho(
            f"  FAILED    {label} — {type(exc).__name__}; recorded non-shipping, continuing",
            fg="yellow",
            err=True,
        )
        row = {
            **corner,
            "slack": _FAILED_CORNER_SLACK,
            "ships": False,
            "eval_split": EVAL_SPLIT_CROSSFIT,
        }
    row.update(
        {
            "family": spec.family,
            "strategy_slug": identity.strategy_slug,
            "structural_strategy": identity.structural_strategy,
            "strategy_signature": identity.signature,
            "strategy_implementation_version": identity.implementation_version,
            "artifact_schema_version": identity.artifact_schema_version,
            "strategy_status": identity.status,
            "controls_json": controls_json(corner),
            "matrix_hash": context.matrix_sha256,
            "corner_fingerprint": corner_fingerprint(
                spec,
                corner,
                str(context.matrix_sha256),
            ),
        }
    )
    if float(row["slack"]) == _FAILED_CORNER_SLACK:
        return row
    verdict = click.style(_verdict(row), fg="green" if row["ships"] else "red")
    elapsed = f"{time.monotonic() - start:.0f}s"
    click.echo(f"  {verdict}  {label}  slack {float(row['slack']):+.3f}  ({elapsed})")
    return row


def _known_good_corners(context: CellContext) -> list[tuple[StrategySpec, dict[str, str]]]:
    """The cell's stat_meta incumbent plus every seed corner registered for it.

    Both are enqueued ahead of the sampler so a budgeted search always evaluates the do-nothing
    baseline and any recipe already proven under full HPO, rather than hoping TPE rediscovers them.
    Confirm reuses this as its seed/incumbent nomination source.

    A structural incumbent is excluded: this sweep cannot rank it (see :func:`_cell_families`), and
    confirm cannot nominate it either — a structural artifact's split fingerprint only exists after
    the retrain, so a synthesized nominee would carry ``None`` and fail its own identity check.
    """
    candidates = [
        (get_strategy(slug), dict(controls))
        for league, market, slug, controls in SEED_CORNERS
        if (league, market) == (context.league, context.market)
    ]
    incumbent = _incumbent_corner(context)
    if incumbent is not None:
        candidates.append(incumbent)
    # A cell that already ships its seed recipe yields the same corner twice.
    seen: set[str] = set()
    known = []
    for spec, corner in candidates:
        fingerprint = corner_fingerprint(spec, corner, str(context.matrix_sha256))
        if fingerprint not in seen and not spec.is_structural:
            seen.add(fingerprint)
            known.append((spec, corner))
    return known


def _incumbent_corner(context: CellContext) -> tuple[StrategySpec, dict[str, str]] | None:
    """The cell's current stat_meta recipe read back as a registered corner, if it still is one.

    Returns ``None`` when the live recipe predates the current grid — a cell whose persisted fields
    no longer name a registered corner has nothing to enqueue, and the sampler covers it.
    """
    meta = load_stat_meta(pathlib.Path(str(STAT_META_PATH)))
    cell = meta.get(context.league, {}).get(context.market, {})
    slug = cell.get("posthoc") if cell.get("posthoc") in STRUCTURAL_STAGE else cell.get("dist")
    try:
        spec = get_strategy(str(slug))
    except ValueError:
        return None
    corner = {
        **spec.fixed_controls,
        **{
            control: str(cell[field])
            for control, field in spec.persist.items()
            if cell.get(field) is not None
        },
    }
    return (spec, corner) if corner in strategy_controls(spec) else None


def search_cell(
    league: str,
    market: str,
    *,
    max_trials: int = MAX_TRIALS_PER_CELL,
    families: tuple[str, ...] | None = None,
    cached: dict[str, dict[str, object]] | None = None,
    out: str | None = None,
) -> pd.DataFrame:
    """Search one cell's recipe space under a trial budget and rank the board by slack.

    ``cached`` maps ``corner_fingerprint`` to an already-scored row — preloaded from admissible board
    rows — so a TPE re-suggestion or a resumed run never retrains a corner this matrix already has a
    verdict for.

    With ``out``, the cell's rows are upserted to the board CSV after every scored corner rather than
    once at the end. The Optuna journal replays which corners were *proposed* but not what they
    scored, so the board is the only durable record of a trial: without the per-corner write, an
    interrupt anywhere in a multi-hour cell throws away every corner it had already trained.
    """
    context = _cell_context(league, market)
    families = families if families is not None else _cell_families(league, market)
    evaluated = dict(cached or {})
    study = cell_study(league, market, families)
    for spec, corner in _known_good_corners(context):
        params = enqueue_params(context, spec, corner)
        if params is not None and spec.slug in families:
            study.enqueue_trial(params, skip_if_exists=True)

    scored: dict[str, dict[str, object]] = {}
    trained: set[str] = set()

    def objective(trial: optuna.Trial) -> float:
        spec, corner = suggest_corner(trial, context, families)
        fingerprint = corner_fingerprint(spec, corner, str(context.matrix_sha256))
        row = evaluated.get(fingerprint)
        if row is None:
            row = _run_and_score(league, market, spec.slug, corner)
            evaluated[fingerprint] = row
            trained.add(fingerprint)
        else:
            click.echo(f"  cached    {_corner_label(corner)}  slack {float(row['slack']):+.3f}")
        trial.set_user_attr(TRAINED_TRIAL_ATTR, fingerprint in trained)
        scored[fingerprint] = row
        if out is not None:
            _upsert_cell(_rank_cell_board(league, market, scored), out)
        return -float(row["slack"])

    def stop_when_budget_spent(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        del trial
        if len(trained) >= max_trials:
            study.stop()

    # The budget counts retrains, not trials: a re-proposed or cached corner costs no training, so
    # it must not cost budget either. n_trials is the reachable grid, the point past which every
    # further suggestion is necessarily a repeat.
    study.optimize(
        objective,
        n_trials=reachable_corners(context, families),
        callbacks=[early_stop_callback(context, families), stop_when_budget_spent],
    )
    return _rank_cell_board(league, market, scored)


def _rank_cell_board(
    league: str, market: str, scored: dict[str, dict[str, object]]
) -> pd.DataFrame:
    """One cell's scored corners as board rows, best slack first."""
    board = pd.DataFrame([{"league": league, "market": market, **row} for row in scored.values()])
    ranked = board.sort_values("slack", ascending=False, ignore_index=True)
    ranked["swept_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    ranked["code_rev"] = _code_rev()
    return ranked.reindex(columns=_BOARD_COLUMNS)


def _has_training_data(league: str, market: str) -> bool:
    """True iff the cell's cached training matrix exists.

    The deterministic sweep freezes input to this parquet and never rebuilds it, so a cell without
    one is skipped rather than triggering an expensive matrix rebuild mid-board.
    """
    return _training_matrix_path(league, market).is_file()


def _iter_eligible_cells(meta: dict, league: str | None, include_shipped: bool):
    """Yield ``(league, market, cell)`` for every registered-market cell in scope.

    Honors the ``--league`` filter, the trainable ``ALL_MARKETS`` registry (stat_meta carries
    non-market entries meditate rejects), and the withheld-only default (``include_shipped`` also
    yields already-shipped cells).
    """
    for lg, markets in meta.items():
        if league is not None and lg != league:
            continue
        for mkt, cell in markets.items():
            if mkt not in ALL_MARKETS.get(lg, []):
                continue
            if include_shipped or cell.get("shipped") == WITHHELD:
                yield lg, mkt, cell


def _candidate_cells(
    league: str | None = None,
    include_shipped: bool = False,
) -> list[tuple[str, str]]:
    """Registered-family cells eligible for the board, before the trainable/data filters.

    Withheld only by default (the ship path); ``include_shipped`` adds already-shipped cells so the
    board can hunt a better strategy for a live cell (evaluated by the separate supersession test,
    not the fresh-ship confirm). Self-maintaining — it follows stat_meta's ``shipped`` field.
    """
    meta = load_stat_meta(pathlib.Path(str(STAT_META_PATH)))
    cells: list[tuple[str, str]] = []
    unsupported: list[tuple[str, str, object]] = []
    for lg, mkt, cell in _iter_eligible_cells(meta, league, include_shipped):
        try:
            distribution_class(cell.get("dist"))
        except (TypeError, ValueError):
            unsupported.append((lg, mkt, cell.get("dist")))
            continue
        cells.append((lg, mkt))
    if unsupported:
        rendered = ", ".join(f"{lg} {mkt} ({dist!r})" for lg, mkt, dist in unsupported)
        raise click.UsageError(f"board contains unsupported distribution cells: {rendered}")
    return cells


def _select_board_cells(
    league: str | None = None,
    include_shipped: bool = False,
    dist_class: str = _DIST_CLASS_ALL,
) -> tuple[dict[tuple[str, str], tuple[str, ...]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Split eligible cells into ``(sweepable → families, missing_data, no_families)``.

    Sweepable = eligible (see :func:`_candidate_cells`), present in the trainable ``ALL_MARKETS``
    registry (stat_meta carries non-market entries like inning props that meditate rejects), and with
    a cached training matrix. ``missing_data`` are eligible registry cells whose matrix is absent —
    the deterministic sweep can't build one, so they are surfaced as a warning rather than swept.
    ``no_families`` are cells the ``dist_class`` filter or the count-admission gates leave with
    nothing to search; they are warned the same way rather than silently dropped.
    """
    missing: list[tuple[str, str]] = []
    no_families: list[tuple[str, str]] = []
    sweepable: dict[tuple[str, str], tuple[str, ...]] = {}
    for cell in _candidate_cells(league, include_shipped):
        if not _has_training_data(*cell):
            missing.append(cell)
            continue
        families = _cell_families(*cell, dist_class)
        if families:
            sweepable[cell] = families
        else:
            no_families.append(cell)
    return sweepable, missing, no_families


def _cell_trial_count(league: str, market: str, families: tuple[str, ...], max_trials: int) -> int:
    """The deterministic trainings one cell contributes: its reachable grid, capped by the budget."""
    return min(reachable_corners(_cell_context(league, market), families), max_trials)


def _read_board(path: pathlib.Path) -> pd.DataFrame:
    """Load any legacy board into the current ordered schema."""
    board = pd.read_csv(path, keep_default_na=False).replace("", pd.NA)
    for column in _BOARD_COLUMNS:
        if column not in board:
            board[column] = pd.NA
    return board.reindex(columns=_BOARD_COLUMNS)


def _split_contract_ok(spec: StrategySpec, split: object) -> bool:
    """Whether a board row's split-fingerprint presence matches the spec's contract."""
    if spec.split_fingerprint_path:
        return isinstance(split, str)
    return split is None


def _row_matches_contract(source: pd.Series, context: CellContext) -> bool:
    """Whether one board row is a reusable verdict for the cell's *current* corner contract.

    A budgeted search never enumerates a whole cell, so admission is per row: the row must name a
    registered corner of a registered spec, carry that spec's current signature and versions, be
    fingerprinted against the matrix in hand, and — the marker that separates this board from the
    1161 legacy rows scored on the ship holdout — have been evaluated on the cross-fit frame.
    """
    missing = any(pd.isna(source.get(column)) for column in _IDENTITY_COLUMNS)
    if missing or str(source.get("eval_split")) != EVAL_SPLIT_CROSSFIT:
        return False
    try:
        spec = get_strategy(str(source["strategy_slug"]))
        controls = parse_controls(source["controls_json"])
    except (TypeError, ValueError):
        return False
    if controls not in strategy_controls(spec):
        return False
    split = source.get("split_fingerprint")
    if not _split_contract_ok(spec, None if pd.isna(split) else split):
        return False
    return all(
        str(source.get(field)) == str(value)
        for field, value in _current_row_identity(spec, controls, context).items()
    )


def _current_row_identity(
    spec: StrategySpec, controls: dict[str, str], context: CellContext
) -> dict[str, object]:
    """The board-identity fields a row for this corner would carry if swept right now."""
    return {
        "family": spec.family,
        "structural_strategy": spec.slug if spec.is_structural else BASE_STRUCTURAL_STRATEGY,
        "strategy_signature": spec.canonical_signature,
        "strategy_implementation_version": spec.implementation_version,
        "artifact_schema_version": spec.artifact_schema_version,
        "controls_json": controls_json(controls),
        "matrix_hash": context.matrix_sha256,
        "corner_fingerprint": corner_fingerprint(spec, controls, str(context.matrix_sha256)),
    }


def _cached_corners(
    out: str | None, league: str, market: str, context: CellContext
) -> dict[str, dict[str, object]]:
    """Admissible prior rows for one cell, keyed by corner fingerprint, for reuse without retraining."""
    if out is None or not pathlib.Path(out).exists():
        return {}
    prior = _read_board(pathlib.Path(out))
    cell_rows = prior[(prior["league"] == league) & (prior["market"] == market)]
    return {
        str(source["corner_fingerprint"]): source.drop(labels=["league", "market"]).to_dict()
        for _, source in cell_rows.iterrows()
        if _row_matches_contract(source, context)
    }


def run_board(
    cells: list[tuple[str, str]],
    out: str | None = None,
    resume: bool = False,
    *,
    max_trials: int = MAX_TRIALS_PER_CELL,
    families_by_cell: dict[tuple[str, str], tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Search every cell in ``cells``, printing each verdict as it lands and upserting the board CSV
    per scored corner so an interrupt keeps partial progress and a ``--league``-scoped run leaves
    other leagues' rows intact. With ``resume``, each cell reopens its Optuna journal and reuses every
    admissible prior row, so a crashed multi-hour run continues instead of retraining scored corners.
    Returns the requested cells' board; CSV upserts preserve all unrelated rows.
    """
    boards: list[pd.DataFrame] = []
    for league, market in cells:
        cell_board = search_cell(
            league,
            market,
            max_trials=max_trials,
            families=(families_by_cell or {}).get((league, market)),
            cached=(
                _cached_corners(out, league, market, _cell_context(league, market))
                if resume
                else None
            ),
            out=out,
        )
        boards.append(cell_board)
        _print_cell_summary(cell_board)
        if out is not None:
            _upsert_cell(cell_board, out)
    return pd.concat(boards, ignore_index=True)


def _upsert_cell(cell_board: pd.DataFrame, out: str) -> pd.DataFrame:
    """Merge one cell's rows into the board CSV at ``out`` — replacing any prior rows for that
    cell — so a single-cell run refreshes the living board instead of clobbering it. Returns the
    rows written for this cell (what the CLI summarizes).
    """
    path = pathlib.Path(out)
    if path.exists():
        prior = _read_board(path)
        league, market = cell_board["league"].iloc[0], cell_board["market"].iloc[0]
        keep = prior[~((prior["league"] == league) & (prior["market"] == market))]
        pd.concat([keep, cell_board], ignore_index=True).to_csv(path, index=False)
    else:
        cell_board.to_csv(path, index=False)
    return cell_board


def _stat_meta_edit(row: object) -> str:
    """The exact stat_meta.json fields to persist a winning corner, resolved for its family.

    Reads the family's ``persist`` map so the operator doesn't have to translate board columns to
    field names: SkewNormal → ``target_normalization=…, dist_training_loss=…, sn_param=…,
    blending=…``; a count family → ``dist=…, [zinb_mode=…,] count_dispersion_objective=…,
    blending=…`` (the persisted ``dist`` pins the winning family, e.g. flipping a cell
    ZINB→NegBin).
    """
    spec = get_strategy(row["strategy_slug"])
    edits = strategy_persistence_edits(
        _cell_context(row["league"], row["market"]), spec, parse_controls(row["controls_json"])
    )
    return ", ".join(f"{field}={value}" for field, value in edits.items())


def _print_cell_summary(board: pd.DataFrame) -> None:
    """One cell's verdict + the exact stat_meta.json edit to ship it, over a narrow per-corner table.

    The board CSV keeps all columns; the screen shows the verdict, what to change in stat_meta.json
    (for a shipping cell), and each corner's family axes / slack / blocking gate.
    """
    best = board.iloc[0]
    ships = bool(best["ships"])
    click.echo(
        click.style(f"\n{best['league']} {best['market']} — ", bold=True)
        + click.style(_verdict(best), fg="green" if ships else "red", bold=True)
        + click.style(f" (best slack {float(best['slack']):+.3f})", bold=True)
    )
    if ships:
        click.echo(f"  → stat_meta.json: {_stat_meta_edit(best)}")
    table = [
        [
            r["strategy_slug"],
            r["structural_strategy"],
            r["controls_json"],
            f"{float(r['slack']):+.3f}",
            "yes" if r["ships"] else "no",
            " ".join(_failed_gates(r)) or "-",
        ]
        for _, r in board.iterrows()
    ]
    click.echo(
        tabulate.tabulate(
            table,
            headers=[
                "strategy",
                "structural strategy",
                "controls",
                "slack",
                "ships",
                "failed gates",
            ],
            tablefmt="github",
        )
    )


def _print_board_rollup(board: pd.DataFrame) -> None:
    """Board-wide tally + a scannable 'what to edit' table: one row per shipping cell naming the
    exact stat_meta.json fields to set. This is the takeaway — no cross-referencing the board CSV.
    """
    cells = list(board.groupby(["league", "market"], sort=False))
    shipping = [(lg, mkt, sub) for (lg, mkt), sub in cells if bool(sub["ships"].any())]
    click.echo(f"\n{len(cells)} cells swept · {len(shipping)} with a shipping corner")
    if not shipping:
        return
    click.echo("\nTo ship a cell, set these fields in its data/config/stat_meta.json entry:")
    rows = []
    for lg, mkt, sub in shipping:
        best = sub.sort_values("slack", ascending=False).iloc[0]
        rows.append(
            [
                f"{lg} {mkt}",
                best["family"],
                _stat_meta_edit(best),
                f"{float(best['slack']):+.3f}",
            ]
        )
    click.echo(
        tabulate.tabulate(
            rows,
            headers=["cell", "family", "set in stat_meta.json", "slack"],
            tablefmt="github",
        )
    )


def _run_board_mode(
    league: str | None,
    include_shipped: bool,
    dist_class: str,
    out: str,
    resume: bool,
    dry_run: bool,
    max_trials: int,
) -> pd.DataFrame:
    """Derive the board, warn per skipped cell, print the scope, sweep.

    ``dist_class`` narrows each cell's family pool; ``resume`` reuses admissible prior rows and
    reopens each cell's Optuna journal; ``dry_run`` prints the resolved scope then returns without
    training a single corner.
    """
    families_by_cell, missing, no_families = _select_board_cells(
        league, include_shipped, dist_class
    )
    for lg, mkt in missing:
        click.secho(
            f"  skip {lg} {mkt}: no cached training matrix — train it first "
            "(the deterministic sweep won't rebuild it)",
            fg="yellow",
        )
    for lg, mkt in no_families:
        click.secho(
            f"  skip {lg} {mkt}: no applicable family under --dist-class {dist_class}", fg="yellow"
        )
    if not families_by_cell:
        raise click.UsageError(
            f"no trainable cells with cached data to sweep{f' in {league}' if league else ''}."
        )
    cells = list(families_by_cell)
    trials = sum(
        _cell_trial_count(lg, mkt, families_by_cell[(lg, mkt)], max_trials) for lg, mkt in cells
    )
    scope = f" ({league})" if league else ""
    skipped = len(missing) + len(no_families)
    note = f" · {skipped} skipped" if skipped else ""
    click.echo(
        f"board{scope}: {len(cells)} cells to sweep{note} · <={trials} deterministic trainings"
    )
    if dry_run:
        click.secho("  [dry-run] no corners trained", fg="cyan")
        return pd.DataFrame(columns=_BOARD_COLUMNS)
    result = run_board(
        cells, out=out, resume=resume, max_trials=max_trials, families_by_cell=families_by_cell
    )
    _print_board_rollup(result)
    return result


@click.command(name="model-strategy-sweep")
@click.option(
    "--league",
    default=None,
    help="League code, e.g. WNBA. A single cell also needs --market; alone it narrows the board.",
)
@click.option(
    "--market",
    default=None,
    help="Market stem, e.g. AST. With --league, sweeps just that cell; omit it to sweep the board.",
)
@click.option(
    "--include-shipped",
    is_flag=True,
    default=False,
    help="Board mode: also sweep already-shipped (devel/main) cells to hunt a better strategy — "
    "evaluated by the supersession test, not the fresh-ship --confirm (which only auto-ships "
    "withheld cells). Off by default.",
)
@click.option(
    "--dist-class",
    type=click.Choice([*_DIST_CLASSES, _DIST_CLASS_ALL]),
    default=_DIST_CLASS_ALL,
    show_default=True,
    help="Narrow each cell's family pool to one distribution class — 'count' searches only "
    "ZINB/NegBin/DPO, 'continuous' only SkewNormal/Mixture. Filters families, not cells.",
)
@click.option(
    "--max-trials",
    type=click.IntRange(min=1),
    default=MAX_TRIALS_PER_CELL,
    show_default=True,
    help="Per-cell trial budget. A cell whose reachable grid is smaller runs the whole grid.",
)
@click.option(
    "--confirm",
    is_flag=True,
    default=False,
    help="After ranking, persist each cell's best persistable corner to stat_meta.json and confirm "
    "it with a full-HPO retrain; failures auto-revert (stat_meta + pickle).",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the --confirm prompt (unattended). No effect without --confirm.",
)
@click.option(
    "--confirm-nominees",
    type=click.IntRange(min=1),
    default=None,
    help="Cap each cell's confirm walk at its first N nominees, highest board slack first. "
    "Unset walks every nominee. No effect without --confirm.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Reopen each cell's Optuna journal and reuse every admissible prior board row — resume a "
    "crashed multi-hour run instead of retraining corners this matrix already has a verdict for.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the resolved scope (cells, ~trainings, what --resume would skip) and exit without "
    "training a single corner.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default=None,
    help="Board CSV path. Defaults to the package data dir "
    "(data/research/strategy_research_board.csv): a board upserts per cell, a single cell upserts.",
)
def main(
    league: str | None,
    market: str | None,
    include_shipped: bool,
    dist_class: str,
    max_trials: int,
    confirm: bool,
    yes: bool,
    confirm_nominees: int | None,
    resume: bool,
    dry_run: bool,
    out: str | None,
) -> None:
    """Operation Ship 75 strategy sweep — one conditional-TPE study per cell over the strategy
    catalog, each corner scored on a holdout-blind cross-fit gate row. Naming ``--league`` and
    ``--market`` sweeps one cell; omitting ``--market`` sweeps the board (``--league`` narrows it).
    ``--confirm`` then full-HPO-retrains each cell's nominees until one ships.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = out or str(STRATEGY_RESEARCH_BOARD)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    if market is None:
        result = _run_board_mode(
            league, include_shipped, dist_class, out, resume, dry_run, max_trials
        )
    else:
        if not league:
            raise click.UsageError(
                "pass --league with --market for a single cell, or omit --market to sweep the board"
            )
        families = _cell_families(league, market, dist_class)
        if not families:
            raise click.UsageError(
                f"{league} {market}: no applicable family under --dist-class {dist_class}"
            )
        if dry_run:
            click.secho(
                f"[dry-run] {league} {market}: families {', '.join(families)} "
                f"· <={_cell_trial_count(league, market, families, max_trials)} trials",
                fg="cyan",
            )
            return
        result = search_cell(
            league,
            market,
            max_trials=max_trials,
            families=families,
            cached=(
                _cached_corners(out, league, market, _cell_context(league, market))
                if resume
                else None
            ),
            out=out,
        )
        _upsert_cell(result, out)
        _print_cell_summary(result)
    click.echo(f"\nboard: {out}")

    if confirm and not dry_run:
        from sportstradamus.training.model_strategy.confirm import run_confirm

        run_confirm(result, yes=yes, max_nominees=confirm_nominees)


if __name__ == "__main__":
    main()
