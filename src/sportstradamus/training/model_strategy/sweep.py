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
full-HPO 6/6 before a withheld model can ship. Because those confirms land systematically below
their board scores, the board sorts on ``discounted_slack`` — raw slack re-priced under the
confirm ledger's measured per-gate confirm-minus-board medians (:func:`_ledger_gate_discounts`).
"""

import collections
import contextlib
import functools
import hashlib
import importlib.resources as pkg_resources
import math
import os
import pathlib
import re
import signal
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from statistics import fmean

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
from sportstradamus.training.model_strategy.progress import (
    NORMAL,
    QUIET,
    VERBOSE,
    BoardProgress,
    SearchProgress,
    compress_corner,
    echo_above_bar,
    human_seconds,
    line_width,
    wrapped,
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
    CellSearchState,
    cell_study,
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
# Hard ceiling on one deterministic trial, and the cap for a cell nothing has been measured for yet.
# A corner's median is 33 s and its p95 is 385 s, so 300 s cuts 6.5% of the 2208 measured corners —
# but slow corners are also *bad* ones (median slack -1.162 against -0.194, ships 20.8% against
# 35.8%), so only 4 of 81 cells have their best corner above it and only MLB hits ships uncapped and
# not at 300 s, its winner sitting at 342 s. A cut corner is retried rather than cached, so the cost
# of the ones this does catch is one retry, not a lost recipe.
_MEDITATE_TRIAL_TIMEOUT_S = 300
# Once a cell has timings, its own worst *finished* corner sets the cap instead: a timeout is a hang
# detector, not a work limit, and five minutes of silence on a cell whose corners run 15 s is a long
# time to find out. The floor is what keeps that from becoming a work limit in turn — the 7 cells
# whose slowest corner is under a minute run ~44 s at worst, so 120 s tolerates an outlier without
# cutting one. At this ceiling 60 of 81 cells sit at it and 21 are genuinely adaptive.
_TIMEOUT_HEADROOM = 2.0
_TIMEOUT_FLOOR_S = 120
# How far n_trials runs past the reachable grid. A corner from a family the cell has abandoned is
# pruned before it trains, and TPE keeps proposing it because pruned trials are not in its model, so
# without slack a cell that rules out a family early could exhaust its suggestions before its budget.
_PRUNED_SUGGESTION_SLACK = 3
# The `failure` value _run_and_score records for a timed-out corner; _cached_corners refuses to reuse
# one, so `--resume` retrains it.
_TIMEOUT_FAILURE = subprocess.TimeoutExpired.__name__
# Serializes _upsert_cell's read-modify-write of the board CSV across concurrently swept cells.
_BOARD_WRITE_LOCK = threading.Lock()
# Concurrent cells, when --jobs is not given. Half the cores leaves room for the operator's shell and
# whatever else the box is doing; the cap is where the returns flatten, because a board finishes no
# faster than its slowest single cell. Deterministic training is core-bound, not memory-bound — the
# largest cached matrix is 117k x 165 and a worker's floor is its interpreter, ~580 MB.
_MAX_AUTO_JOBS = 8
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
    "discounted_slack",
    "confirm_risk",
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
    "elapsed_s",
    "failure",
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

# Where confirm's _record_nominee_gates appends each full-HPO nominee's model_stats row; the board
# reads it back to price how confirm verdicts land relative to board scores.
NOMINEE_LEDGER_PATH: pathlib.Path = _REPO_ROOT / "research" / "confirm_nominee_gates.csv"

# The gate scalars with a board-side counterpart to discount. g6's legs are CI bounds against
# per-row references, so it has no discountable scalar and is left alone.
_DISCOUNTED_GATES: tuple[str, ...] = (
    "g4_pit_ks",
    "g2_star_z",
    "g3_bench_z",
    "g5_ece_debiased",
    "g1_brier_diff_ci_hi",
)

# The dispersion calibrator floors at 0.1, so a ledger row on the floor (float wobble included) is
# a diverged SkewNormal fit, not a calibration measurement — it never informs the discounts.
_DIVERGED_DISPERSION_CAL: float = 0.1005

# Below this many usable ledger observations the per-gate medians and the rollup's ship-rate cells
# are noise, so the discounts and the calibrated-expectations table stay inert.
_MIN_LEDGER_ROWS: int = 8

# A board nominee's ledger `source` label carries the slack it was nominated at.
_BOARD_SLACK_PATTERN = re.compile(r"board slack ([+-]\d+\.\d+)")

# Rollup P(ship) bands on board slack: confirm odds measured non-monotone — the mid band shipped
# ~10/17 while the top band shipped 1/9 — so banded rates, not the raw rank, carry the expectation.
_SLACK_BAND_EDGES: tuple[float, float] = (0.05, 0.15)
_SLACK_BAND_LABELS: tuple[str, ...] = ("<0.05", "0.05-0.15", ">=0.15")

# Cell-summary layout. The four fixed columns plus the github borders cost this much, and what is
# left goes to the corner label; below the floor the label stops being identifiable at all.
_SUMMARY_FIXED_CHARS: int = 55
_MIN_CORNER_CHARS: int = 24
# A cell scores ~30 corners and the tail is uniformly deep-negative, so the screen shows the leading
# ones and the board CSV keeps the rest. `-v` prints them all.
_SUMMARY_ROWS: int = 12


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
                    echo_above_bar(f"  archive write-locked; retrying in {wait}s …", err=True)
                    time.sleep(wait)
                    continue
                abort = _NATIVE_ABORT_PATTERN.search(_log_tail(log_path))
                signature = f": {abort.group()}" if abort else ""
                echo_above_bar(
                    f"  meditate failed — {_failure_reason(exc)}{signature} — see {log_path}",
                    fg="red",
                    err=True,
                )
                raise
            except subprocess.TimeoutExpired:
                echo_above_bar(
                    f"  meditate timed out after {timeout}s — see {log_path}", fg="red", err=True
                )
                raise


def _run_deterministic_meditate(
    league: str, market: str, corner: dict[str, str], spec: StrategySpec, timeout_s: int
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
    _run_meditate_with_lock_retry(cmd, _log_path(league, market, corner, spec), timeout=timeout_s)


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
    league: str,
    market: str,
    strategy_slug: str,
    corner: dict[str, str],
    timeout_s: int = _MEDITATE_TRIAL_TIMEOUT_S,
) -> dict[str, object]:
    """Train the ``corner`` retrain trial and score it; tag the row with its family.

    Calibration is not re-fit here: the dump already carries the pipeline's cross-fit
    out-of-fold calibration, and :func:`_score_corner` reads it off the dump via the production
    :func:`scorecard.gate_row` — no test re-fit. The dump is keyed by subdir, so scoring right after
    this trial's train — before the next sequential trial overwrites it — keeps the loss/mode axes
    honest without a per-corner dump path (which is also why the sweep always retrains rather than
    reusing a dump). A corner whose meditate errors or times out is caught and returned as a
    non-shipping row (:data:`_FAILED_CORNER_SLACK`) naming the exception in ``failure``, so one bad
    corner never aborts the board — and, cached under its fingerprint, so a resumed run never
    retrains a reliable crash. Display is the caller's: :class:`SearchProgress` renders the row.

    An interrupt is the one failure that is not the corner's fault. SIGINT reaches the whole process
    group, so a Ctrl-C lands on the child as a signal death while the pool's worker threads keep
    running; recording that as a failed corner would write a row per in-flight cell and cache it, and
    ``--resume`` would then never retrain any of them.
    """
    spec = get_strategy(strategy_slug)
    context = _cell_context(league, market)
    identity = build_artifact_identity(
        spec.slug,
        league,
        market,
        corner,
        matrix_hash=context.matrix_sha256,
    )
    # Spans the archive-lock back-off too (up to _LOCK_RETRY_WAITS_S of sleep), which is what an ETA
    # built from these samples needs to include — that wait is real time the operator waits.
    start = time.monotonic()
    try:
        _run_deterministic_meditate(league, market, corner, spec, timeout_s)
        row = _score_corner(league, market, corner, spec)
    except (
        InactiveStrategyArtifactError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.returncode == -signal.SIGINT:
            raise
        row = {
            **corner,
            "slack": _FAILED_CORNER_SLACK,
            "ships": False,
            "eval_split": EVAL_SPLIT_CROSSFIT,
            "failure": type(exc).__name__,
        }
    row.update(
        {
            "elapsed_s": round(time.monotonic() - start, 1),
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
    verbosity: int = NORMAL,
    seed_seconds: float | None = None,
    position: int = 0,
) -> pd.DataFrame:
    """Search one cell's recipe space under a trial budget and rank the board by slack.

    ``cached`` maps ``corner_fingerprint`` to an already-scored row — preloaded from admissible board
    rows — so a TPE re-suggestion or a resumed run never retrains a corner this matrix already has a
    verdict for.

    With ``out``, the cell's rows are upserted to the board CSV after every scored corner rather than
    once at the end. The Optuna journal replays which corners were *proposed* but not what they
    scored, so the board is the only durable record of a trial: without the per-corner write, an
    interrupt anywhere in a multi-hour cell throws away every corner it had already trained.

    ``seed_seconds`` is this cell's mean corner duration from a prior run; it carries the bar's ETA
    until enough of this run's own corners have landed to estimate from. ``position`` is the terminal
    row this cell's bar draws on, handed out by :class:`BoardProgress` when cells run concurrently.
    """
    context = _cell_context(league, market)
    families = families if families is not None else _cell_families(league, market)
    evaluated = dict(cached or {})
    discounts = _ledger_gate_discounts(out)
    timeout_s = _cell_timeout_seconds(out, league, market)
    study = cell_study(league, market, families)
    for spec, corner in _known_good_corners(context):
        params = enqueue_params(context, spec, corner)
        if params is not None and spec.slug in families:
            study.enqueue_trial(params, skip_if_exists=True)

    scored: dict[str, dict[str, object]] = {}
    trained: set[str] = set()
    state = CellSearchState(context, families, _prior_standings(evaluated))
    progress = SearchProgress(
        league,
        market,
        min(reachable_corners(context, families), max_trials),
        verbosity=verbosity,
        seed_seconds=seed_seconds,
        reused=len(evaluated),
        position=position,
    )

    def objective(trial: optuna.Trial) -> float:
        spec, corner = suggest_corner(trial, context, families)
        fingerprint = corner_fingerprint(spec, corner, str(context.matrix_sha256))
        row = evaluated.get(fingerprint)
        # Asked once, before the cache is written: `fingerprint in trained` would answer True again
        # on a re-proposal of a corner this run already trained, and TPE re-proposes about a quarter
        # of the time — enough to walk the bar past its total and double-weight the ETA.
        retrained = row is None
        if retrained:
            if state.abandoned(spec.slug):
                trial.set_user_attr(TRAINED_TRIAL_ATTR, False)
                raise optuna.TrialPruned
            progress.starting(corner)
            row = _run_and_score(league, market, spec.slug, corner, timeout_s)
            evaluated[fingerprint] = row
            trained.add(fingerprint)
            state.observe(spec.slug, float(row["slack"]))
        trial.set_user_attr(TRAINED_TRIAL_ATTR, retrained)
        progress.scored(
            corner,
            verdict=_verdict(row),
            slack=float(row["slack"]),
            elapsed_s=row.get("elapsed_s"),
            trained=retrained,
            failure=_board_text(row, "failure"),
        )
        scored[fingerprint] = row
        if out is not None:
            _upsert_cell(_rank_cell_board(league, market, scored, discounts), out)
        return -float(row["slack"])

    def stop_when_budget_spent(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        del trial
        if len(trained) >= max_trials:
            study.stop()

    # The budget counts retrains, not trials: a re-proposed, cached, or abandoned-family corner costs
    # no training, so it must not cost budget either. n_trials is a ceiling on *suggestions*, so it
    # carries slack over the reachable grid — a family the cell has ruled out keeps being proposed
    # (TPE does not model pruned trials), and each of those costs a trial and no time.
    with contextlib.closing(progress):
        study.optimize(
            objective,
            n_trials=_PRUNED_SUGGESTION_SLACK * reachable_corners(context, families),
            callbacks=[state.stop_if_done, stop_when_budget_spent],
        )
    return _rank_cell_board(league, market, scored, discounts)


def _prior_standings(evaluated: dict[str, dict[str, object]]) -> list[tuple[str, float]]:
    """``(family, slack)`` per reused row, so a resumed cell keeps the family verdicts it already had."""
    return [(str(row["family"]), float(row["slack"])) for row in evaluated.values()]


def _ledger_gate_discounts(out: str | None) -> dict[str, float]:
    """Per-gate median confirm-minus-board gate shifts measured from the nominee ledger.

    Full-HPO confirms land systematically worse than the holdout-blind board on the same corner
    (g4_pit_ks +0.0115, g2_star_z +0.034 on the 37-row ledger), so raw board slack oversells its
    nominees. Non-diverged ledger rows are paired with their board-side gate values — the ledger's
    own ``board_*`` echo columns when a confirm-side change has added them, else an identity join
    against the board CSV at ``out`` — and each gate's median shift is returned. ``{}`` (inert)
    when the ledger is absent, unpairable, or thinner than :data:`_MIN_LEDGER_ROWS` pairs.
    """
    if not NOMINEE_LEDGER_PATH.exists():
        return {}
    ledger = pd.read_csv(NOMINEE_LEDGER_PATH)
    diverged = pd.to_numeric(ledger["dispersion_cal"], errors="coerce") <= _DIVERGED_DISPERSION_CAL
    ledger = ledger[~diverged]
    echo_columns = [f"board_{gate}" for gate in _DISCOUNTED_GATES]
    if set(echo_columns) <= set(ledger.columns):
        paired, board_columns = ledger, echo_columns
    else:
        if out is None or not pathlib.Path(out).exists():
            return {}
        paired = ledger.merge(
            _read_board(pathlib.Path(out)),
            left_on=["league", "market", "strategy_slug", "strategy_controls_json"],
            right_on=["league", "market", "strategy_slug", "controls_json"],
            suffixes=("", "_board"),
        )
        board_columns = [f"{gate}_board" for gate in _DISCOUNTED_GATES]
    confirm_values = paired[list(_DISCOUNTED_GATES)].apply(pd.to_numeric, errors="coerce")
    board_values = (
        paired[board_columns]
        .set_axis(list(_DISCOUNTED_GATES), axis="columns")
        .apply(pd.to_numeric, errors="coerce")
    )
    shifts = confirm_values - board_values
    shifts = shifts[shifts.notna().any(axis="columns")]
    if len(shifts) < _MIN_LEDGER_ROWS:
        return {}
    return {gate: float(median) for gate, median in shifts.median().items() if pd.notna(median)}


def _board_float(row: dict[str, object], column: str) -> float | None:
    """A board row's numeric field as float, ``None`` when blank — cached CSV rows carry strings."""
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _board_text(row: dict[str, object], column: str) -> str | None:
    """A board row's text field, ``None`` when blank — a cached CSV row carries ``pd.NA``."""
    value = row.get(column)
    return None if value is None or pd.isna(value) else str(value)


def _cell_mean_seconds(out: str | None) -> tuple[dict[tuple[str, str], float], float | None]:
    """Mean corner seconds per cell from the board's ``elapsed_s`` history, plus the all-cells mean.

    Mean rather than median: a search ETA is a *sum* over the corners still to run, and E[sum] is
    k·mean. A median would systematically under-call a distribution whose p95 (221 s) is twelve times
    its median (19 s). Empty when no board has timings yet, and then no ETA is shown at all.
    """
    if out is None or not pathlib.Path(out).exists():
        return {}, None
    board = _read_board(pathlib.Path(out))
    elapsed = pd.to_numeric(board["elapsed_s"], errors="coerce")
    timed = board.assign(elapsed_s=elapsed).dropna(subset=["elapsed_s"])
    if timed.empty:
        return {}, None
    per_cell = timed.groupby(["league", "market"])["elapsed_s"].mean()
    return {cell: float(mean) for cell, mean in per_cell.items()}, float(timed["elapsed_s"].mean())


def _cell_timeout_seconds(out: str | None, league: str, market: str) -> int:
    """This cell's trial ceiling: headroom over its own slowest measured corner, else the flat one.

    Maximum rather than a quantile because the spread *inside* one cell is wide — NBA BLK's corners
    run 12 s to 906 s on the same target — so anything but the worst legitimate corner cuts real work.
    Clamped both ways by :data:`_TIMEOUT_FLOOR_S` and :data:`_MEDITATE_TRIAL_TIMEOUT_S`.

    Only corners that *finished* are evidence. A timed-out one measures the cap that killed it, so
    counting it would make each run's cap twice the last one's until it pinned at the ceiling — on
    exactly the cell the cap exists for.
    """
    path = pathlib.Path(out) if out else None
    if path is None or not path.exists():
        return _MEDITATE_TRIAL_TIMEOUT_S
    board = _read_board(path)
    # A blank `failure` reads back as pd.NA; `!= _TIMEOUT_FAILURE` against pd.NA is pd.NA, not
    # True, so every finished corner would silently drop out of the mask without the cast + fillna.
    cell = board[
        (board["league"] == league)
        & (board["market"] == market)
        & (board["failure"].astype("string").fillna("") != _TIMEOUT_FAILURE)
    ]
    slowest = pd.to_numeric(cell["elapsed_s"], errors="coerce").max()
    if not np.isfinite(slowest):
        return _MEDITATE_TRIAL_TIMEOUT_S
    return int(min(_MEDITATE_TRIAL_TIMEOUT_S, max(_TIMEOUT_FLOOR_S, _TIMEOUT_HEADROOM * slowest)))


def _discounted_slack(row: dict[str, object], discounts: dict[str, float]) -> float:
    """``slack`` recomputed with each gate scalar shifted by its measured confirm inflation.

    Equal to ``slack`` when the discounts are inert, and a failed corner stays ``-inf``. Capped at
    the raw ``slack``: the board row carries no g6 legs, so an uncapped recompute would float a
    g6-bound row above its honest rank.
    """
    slack = float(row["slack"])
    if not discounts or slack == _FAILED_CORNER_SLACK:
        return slack
    adjusted: dict[str, object] = {"g4_pit_ks_max": _board_float(row, "g4_pit_ks_max")}
    for gate in _DISCOUNTED_GATES:
        value = _board_float(row, gate)
        adjusted[gate] = None if value is None else value + discounts.get(gate, 0.0)
    return min(min_gate_slack(adjusted), slack)


def _confirm_risk(row: dict[str, object], context: CellContext, discounts: dict[str, float]) -> str:
    """``"high"`` when the corner matches a shape the ledger shows failing confirm, else ``""``.

    Two measured shapes: a continuous-class family on an integer-lattice target (the diverged
    ``dispersion_cal``-floor confirms), and g4 headroom inside the measured PIT-KS inflation.
    """
    if distribution_class(str(row["family"])) == "continuous" and context.target_is_integer:
        return "high"
    g4, g4_max = _board_float(row, "g4_pit_ks"), _board_float(row, "g4_pit_ks_max")
    if g4 is not None and g4_max is not None and g4 + discounts.get("g4_pit_ks", 0.0) >= g4_max:
        return "high"
    return ""


def _rank_cell_board(
    league: str,
    market: str,
    scored: dict[str, dict[str, object]],
    discounts: dict[str, float] | None = None,
) -> pd.DataFrame:
    """One cell's scored corners as board rows, best empirically discounted slack first.

    ``discounted_slack`` re-prices each row's raw slack under the ledger's confirm-vs-board gate
    shifts and is the sort key; ``confirm_risk`` flags the corner shapes that measured worst on
    confirm. Both fall back to the raw ranking / stay blank on a thin ledger.
    """
    discounts = discounts or {}
    context = _cell_context(league, market)
    board = pd.DataFrame(
        [
            {
                "league": league,
                "market": market,
                **row,
                "discounted_slack": _discounted_slack(row, discounts),
                "confirm_risk": _confirm_risk(row, context, discounts),
            }
            for row in scored.values()
        ]
    )
    ranked = board.sort_values("discounted_slack", ascending=False, ignore_index=True)
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
    """Admissible prior rows for one cell, keyed by corner fingerprint, for reuse without retraining.

    A row that timed out is not admissible. Every other failure is a property of the corner and
    reusing it is the point — a resumed run should not retrain a reliable crash — but a timeout is a
    property of the *machine*: the ceiling is derived from the cell's own timings, and cells now run
    concurrently, so the corner that ran long once may well not next time.
    """
    if out is None or not pathlib.Path(out).exists():
        return {}
    prior = _read_board(pathlib.Path(out))
    cell_rows = prior[(prior["league"] == league) & (prior["market"] == market)]
    return {
        str(source["corner_fingerprint"]): source.drop(labels=["league", "market"]).to_dict()
        for _, source in cell_rows.iterrows()
        if _row_matches_contract(source, context)
        and _board_text(source, "failure") != _TIMEOUT_FAILURE
    }


def run_board(
    cells: list[tuple[str, str]],
    out: str | None = None,
    resume: bool = False,
    *,
    max_trials: int = MAX_TRIALS_PER_CELL,
    families_by_cell: dict[tuple[str, str], tuple[str, ...]] | None = None,
    verbosity: int = NORMAL,
    jobs: int = 1,
) -> pd.DataFrame:
    """Search every cell in ``cells``, printing each verdict as it lands and upserting the board CSV
    per scored corner so an interrupt keeps partial progress and a ``--league``-scoped run leaves
    other leagues' rows intact. With ``resume``, each cell reopens its Optuna journal and reuses every
    admissible prior row, so a crashed multi-hour run continues instead of retraining scored corners.
    Returns the requested cells' board in ``cells`` order; CSV upserts preserve all unrelated rows.

    Each finished cell folds its own measured mean back into the timing history, so the remaining-work
    estimate sharpens as the board proceeds rather than staying pinned to the previous run's numbers.

    ``jobs`` cells run at once. A ``--deterministic`` corner is a single-threaded subprocess that
    never opens the archive, and cells share nothing but the board CSV — which :func:`_upsert_cell`
    serializes — so the pool is bounded by cores, not by any lock of ours. Threads rather than
    processes because the training already happens in a child process: the GIL is released for the
    whole of it, and staying in one process is what lets the board write take an ordinary lock and
    every cell's bar share one terminal.
    """
    swept = [
        cell_board
        for cell_board in _sweep_cells(
            cells,
            out=out,
            resume=resume,
            max_trials=max_trials,
            families_by_cell=families_by_cell or {},
            verbosity=verbosity,
            jobs=jobs,
        )
        if cell_board is not None
    ]
    if jobs > 1:
        # Deferred to here: cells finishing at unpredictable times would interleave their tables into
        # the bar block. Board order, not completion order, so the tail of the run reads as a report.
        for cell_board in swept:
            _print_cell_summary(cell_board, verbosity=verbosity)
    return pd.concat(swept, ignore_index=True) if swept else pd.DataFrame(columns=_BOARD_COLUMNS)


def _sweep_cells(
    cells: list[tuple[str, str]],
    *,
    out: str | None,
    resume: bool,
    max_trials: int,
    families_by_cell: dict[tuple[str, str], tuple[str, ...]],
    verbosity: int,
    jobs: int,
) -> list[pd.DataFrame | None]:
    """Run ``cells`` through the pool, ``None`` where a cell crashed, in ``cells`` order.

    Results are written into a pre-sized list rather than appended, because ``jobs > 1`` finishes them
    in whatever order they happen to take and the board must not be re-ordered by how long its cells
    ran. Each finished cell folds its own measured mean into ``means`` before the next countdown line
    reads it, so the remaining-work estimate sharpens as the board proceeds.
    """
    means, fallback = _cell_mean_seconds(out)
    budgets = {
        cell: _cell_trial_count(*cell, families_by_cell.get(cell, ()), max_trials)
        for cell in cells
        if families_by_cell
    }
    boards: list[pd.DataFrame | None] = [None] * len(cells)
    board = BoardProgress(len(cells), jobs)
    started_at = time.monotonic()

    def sweep_cell(index: int, league: str, market: str) -> None:
        with board.slot() as position:
            started = time.monotonic()
            cell_board = search_cell(
                league,
                market,
                max_trials=max_trials,
                families=families_by_cell.get((league, market)),
                cached=(
                    _cached_corners(out, league, market, _cell_context(league, market))
                    if resume
                    else None
                ),
                out=out,
                verbosity=verbosity,
                seed_seconds=means.get((league, market), fallback),
                position=position,
            )
            boards[index] = cell_board
            if out is not None:
                _upsert_cell(cell_board, out)
            if jobs < 2:
                _print_cell_summary(cell_board, verbosity=verbosity)
            timed = pd.to_numeric(cell_board["elapsed_s"], errors="coerce").dropna()
            if not timed.empty:
                means[(league, market)] = float(timed.mean())
            left = [c for c, done in zip(cells, boards, strict=True) if done is None]
            _echo_board_countdown(
                f"{league} {market}",
                _cell_verdict(cell_board),
                time.monotonic() - started,
                time.monotonic() - started_at,
                left,
                _board_eta(left, budgets, means, fallback, jobs),
            )

    with contextlib.closing(board), ThreadPoolExecutor(max_workers=jobs) as pool:
        _drain([pool.submit(sweep_cell, i, *cell) for i, cell in enumerate(cells)])
    return boards


def _drain(futures: list[Future]) -> None:
    """Wait out every cell, surfacing a crashed one instead of losing the rest of the board.

    A cell can die outside :func:`_run_and_score`'s per-corner catch — a scoring bug, an Optuna
    internal — and before the pool that was one exception ending the whole run. Ctrl-C is the
    exception to the exception: it re-raises, because the operator asked for the run to stop.
    """
    for future in futures:
        try:
            future.result()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # one cell's crash must not cost the other cells
            echo_above_bar(f"  cell failed — {type(exc).__name__}: {exc}", fg="red", err=True)


def _board_eta(
    cells: list[tuple[str, str]],
    budgets: dict[tuple[str, str], int],
    means: dict[tuple[str, str], float],
    fallback: float | None,
    jobs: int = 1,
) -> float | None:
    """Wall-clock ceiling on what ``cells`` still need, or ``None`` when nothing has been timed yet.

    A cell with no history of its own is priced at the all-cells mean rather than at zero: on a fresh
    board only a handful of cells are timed, and summing the rest as free understated a 29-cell run
    by 29x while still presenting the number as a ceiling. With no prior board at all, the cells this
    run has already finished become that mean — otherwise a first run would report nothing the whole
    way through, having measured most of its own cells.

    ``jobs`` workers divide the total, but never below the longest single cell: a cell is swept by one
    worker start to finish, so the board cannot finish before its slowest one does however many
    workers are idle by then.
    """
    typical = fallback if fallback is not None else (fmean(means.values()) if means else None)
    per_cell = [(budgets.get(cell), means.get(cell, typical)) for cell in cells]
    if not per_cell or not all(budget and mean for budget, mean in per_cell):
        return None
    seconds = [budget * mean for budget, mean in per_cell]
    return max(sum(seconds) / jobs, *seconds)


def _cell_verdict(cell_board: pd.DataFrame) -> str:
    """``SHIP +0.076`` / ``no ship (best -0.096)`` / ``no corners`` — a finished cell in one phrase.

    Under ``--jobs`` this is the only place a cell's outcome appears while the run is going, the
    per-cell tables having moved to the end.
    """
    slack = pd.to_numeric(cell_board["slack"], errors="coerce")
    scored = slack[np.isfinite(slack)]
    if scored.empty:
        return "no corners"
    best = float(scored.max())
    ships = bool(cell_board["ships"].eq(True).any())
    return f"SHIP {best:+.3f}" if ships else f"no ship (best {best:+.3f})"


def _echo_board_countdown(
    cell: str,
    verdict: str,
    cell_elapsed: float,
    run_elapsed: float,
    remaining: list[tuple[str, str]],
    eta: float | None,
) -> None:
    """``NBA FGA SHIP +0.076 in 12m · run 2h10m · 8 cells left · <=2h40m`` as each cell lands.

    The running total lives here rather than on the bar because this is the line a multi-day sit is
    actually read by; a bar is scoped to the one cell in front of it. Under ``--jobs`` this is also
    where a cell's verdict shows up at all, the per-cell tables having moved to the end of the run.
    """
    line = f"  {cell} {verdict} in {human_seconds(cell_elapsed)} · run {human_seconds(run_elapsed)}"
    if remaining:
        line += f" · {len(remaining)} cell{'' if len(remaining) == 1 else 's'} left"
        if eta is not None:
            line += f" · <={human_seconds(eta)}"
    echo_above_bar(line)


def _upsert_cell(cell_board: pd.DataFrame, out: str) -> pd.DataFrame:
    """Merge one cell's rows into the board CSV at ``out`` — replacing any prior rows for that
    cell — so a single-cell run refreshes the living board instead of clobbering it. Returns the
    rows written for this cell (what the CLI summarizes).

    Read-modify-write over the whole file, so it takes :data:`_BOARD_WRITE_LOCK`: cells run
    concurrently and each of them rewrites the board after every scored corner. The lock lives here
    rather than at the call sites because there are two of them, per-corner and per-cell.

    ``keep``'s all-NA columns are dropped before the concat and restored by the reindex after it.
    :func:`_read_board` backfills any column a board predates as a ``pd.NA`` scalar, which lands as
    an all-NA *object* column; concatenating one of those against this cell's real dtypes is what
    pandas 2.x warns about, once per scored corner. Round-trips the same values either way — the
    reindex puts every dropped column back, still empty.
    """
    path = pathlib.Path(out)
    with _BOARD_WRITE_LOCK:
        if path.exists():
            prior = _read_board(path)
            league, market = cell_board["league"].iloc[0], cell_board["market"].iloc[0]
            keep = prior[~((prior["league"] == league) & (prior["market"] == market))]
            merged = pd.concat(
                [keep.dropna(axis="columns", how="all"), cell_board], ignore_index=True
            )
            merged.reindex(columns=_BOARD_COLUMNS).to_csv(path, index=False)
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


def _axis_order(controls: dict[str, str]) -> dict[str, str]:
    """``controls`` in canonical axis order — family first, then its own axes.

    ``controls_json`` sorts its keys so the canonical form is stable to hash, which leaves a corner
    parsed back out of it alphabetical: `blending_loss_fn` ahead of `dist`. A positional label has to
    read the same everywhere, and a live corner comes off ``suggest_corner`` in this order already.
    """
    known = {axis: controls[axis] for axis in _AXIS_COLUMNS if axis in controls}
    return known | {axis: value for axis, value in controls.items() if axis not in known}


def _print_cell_summary(board: pd.DataFrame, *, verbosity: int = NORMAL) -> None:
    """One cell's verdict + the exact stat_meta.json edit to ship it, over a narrow per-corner table.

    The board CSV keeps every column and every corner; the screen shows the verdict, what to change in
    stat_meta.json (for a shipping cell), and the leading corners by discounted slack. The old raw
    ``controls_json`` column is gone — it restated axis columns the board already carries and by
    itself ran the table past 250 characters.
    """
    best = board.iloc[0]
    ships = bool(best["ships"])
    click.echo(
        click.style(f"\n{best['league']} {best['market']} — ", bold=True)
        + click.style(_verdict(best), fg="green" if ships else "red", bold=True)
        + click.style(f" (best slack {float(best['slack']):+.3f})", bold=True)
    )
    if ships:
        click.echo(wrapped(f"→ stat_meta.json: {_stat_meta_edit(best)}", "  ", "    "))
    if verbosity <= QUIET:
        return
    shown = board if verbosity >= VERBOSE else board.head(_SUMMARY_ROWS)
    budget = max(_MIN_CORNER_CHARS, line_width() - _SUMMARY_FIXED_CHARS)
    table = [
        [
            compress_corner(_axis_order(parse_controls(r["controls_json"])), budget),
            f"{float(r['slack']):+.3f}",
            f"{float(r['discounted_slack']):+.3f}",
            r["confirm_risk"] or "-",
            " ".join(_failed_gates(r)) or "-",
        ]
        for _, r in shown.iterrows()
    ]
    click.echo(
        tabulate.tabulate(
            table,
            headers=["corner", "slack", "disc slack", "risk", "failed gates"],
            tablefmt="github",
            # Without this tabulate re-parses the slack strings as numbers and drops the leading
            # `+`, so a positive and a negative slack stop being distinguishable at a glance.
            disable_numparse=True,
            colalign=("left", "right", "right", "left", "left"),
        )
    )
    if len(shown) < len(board):
        click.echo(f"  +{len(board) - len(shown)} more corners on the board")


def _swept_class(*names: object) -> str | None:
    """The distribution class of the first name that is a swept family, else ``None``."""
    for name in names:
        try:
            return distribution_class(str(name))
        except ValueError:
            continue
    return None


def _ledger_board_slack(ledger: pd.DataFrame) -> pd.Series:
    """Each ledger row's board-side slack: the ``board_slack`` echo when a confirm-side change has
    added it, else parsed off the ``source`` label (seed/incumbent rows have none).
    """
    if "board_slack" in ledger.columns:
        return pd.to_numeric(ledger["board_slack"], errors="coerce")
    labels = ledger["source"].astype(str)
    return labels.str.extract(_BOARD_SLACK_PATTERN, expand=False).astype(float)


def _print_ledger_expectations() -> None:
    """Calibrated confirm odds: P(ship) per board-slack band × family class, from the ledger.

    Raw board rank is anti-predictive at the top (slack >= +0.19 shipped 1/9), so the rollup shows
    what nominees of each shape actually did on full-HPO confirm. Silent when the ledger is absent
    or has fewer than :data:`_MIN_LEDGER_ROWS` usable rows.
    """
    if not NOMINEE_LEDGER_PATH.exists():
        return
    ledger = pd.read_csv(NOMINEE_LEDGER_PATH)
    families = (
        ledger["distribution"] if "distribution" in ledger.columns else ledger["strategy_slug"]
    )
    nominees = pd.DataFrame(
        {
            "band": pd.cut(
                _ledger_board_slack(ledger),
                bins=[-np.inf, *_SLACK_BAND_EDGES, np.inf],
                labels=_SLACK_BAND_LABELS,
                right=False,
            ),
            "family_class": [
                _swept_class(family, slug)
                for family, slug in zip(families, ledger["strategy_slug"], strict=True)
            ],
            "shipped": ledger["ship"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}),
        }
    ).dropna()
    if len(nominees) < _MIN_LEDGER_ROWS:
        return
    nominees["band"] = nominees["band"].astype(str)
    rates = nominees.groupby(["band", "family_class"])["shipped"].agg(["mean", "count"])
    table = []
    for band in _SLACK_BAND_LABELS:
        cells = [band]
        for family_class in _DIST_CLASSES:
            if (band, family_class) in rates.index:
                rate, n = rates.loc[(band, family_class)]
                cells.append(f"{rate:.0%} ({int(n)})")
            else:
                cells.append("-")
        table.append(cells)
    click.echo(
        f"\nconfirm P(ship) by board slack × family class ({len(nominees)} ledger nominees):"
    )
    click.echo(tabulate.tabulate(table, headers=["board slack", *_DIST_CLASSES], tablefmt="github"))


def _print_board_rollup(board: pd.DataFrame) -> None:
    """Board-wide tally + a scannable 'what to edit' table: one row per shipping cell naming the
    exact stat_meta.json fields to set. This is the takeaway — no cross-referencing the board CSV.
    """
    cells = list(board.groupby(["league", "market"], sort=False))
    shipping = [(lg, mkt, sub) for (lg, mkt), sub in cells if bool(sub["ships"].any())]
    click.echo(f"\n{len(cells)} cells swept · {len(shipping)} with a shipping corner")
    _print_ledger_expectations()
    if not shipping:
        return
    click.echo("\nTo ship a cell, set these fields in its data/config/stat_meta.json entry:")
    # A stanza, not a table: the edit line is the thing you copy, so it must never be truncated to
    # fit a column, and at ~145 characters it is what pushed the old rollup table past 250 wide.
    for lg, mkt, sub in shipping:
        best = sub.sort_values("discounted_slack", ascending=False).iloc[0]
        risk = f"  risk {best['confirm_risk']}" if best["confirm_risk"] else ""
        click.echo(
            click.style(f"\n  {lg} {mkt}", bold=True)
            + f"  {best['family']}  slack {float(best['slack']):+.3f}"
            + f"  disc {float(best['discounted_slack']):+.3f}{risk}"
        )
        click.echo(wrapped(_stat_meta_edit(best), "    ", "    "))


def _warn_unsweepable(
    missing: list[tuple[str, str]], no_families: list[tuple[str, str]], dist_class: str
) -> int:
    """Warn once per cell the board cannot sweep, and return how many were skipped."""
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
    return len(missing) + len(no_families)


def _run_board_mode(
    league: str | None,
    include_shipped: bool,
    dist_class: str,
    out: str,
    resume: bool,
    dry_run: bool,
    max_trials: int,
    verbosity: int = NORMAL,
    confirm: bool = False,
    jobs: int = 1,
) -> pd.DataFrame:
    """Derive the board, warn per skipped cell, print the scope, sweep.

    ``dist_class`` narrows each cell's family pool; ``resume`` reuses admissible prior rows and
    reopens each cell's Optuna journal; ``dry_run`` prints the resolved scope — per cell, including
    what ``--resume`` would reuse — then returns without training a single corner. ``jobs`` cells
    are swept at once.
    """
    families_by_cell, missing, no_families = _select_board_cells(
        league, include_shipped, dist_class
    )
    skipped = _warn_unsweepable(missing, no_families, dist_class)
    if not families_by_cell:
        raise click.UsageError(
            f"no trainable cells with cached data to sweep{f' in {league}' if league else ''}."
        )
    cells = list(families_by_cell)
    budgets = {cell: _cell_trial_count(*cell, families_by_cell[cell], max_trials) for cell in cells}
    trials = sum(budgets.values())
    means, fallback = _cell_mean_seconds(out)
    scope = f" ({league})" if league else ""
    note = f" · {skipped} skipped" if skipped else ""
    jobs = min(jobs, len(cells))
    eta = _board_eta(cells, budgets, means, fallback, jobs)
    click.echo(
        f"board{scope}: {len(cells)} cells to sweep{note} · {jobs} at a time"
        f" · <={trials} deterministic trainings"
        + (f" · <={human_seconds(eta)}" if eta is not None else "")
    )
    if dry_run:
        click.secho("  [dry-run] no corners trained", fg="cyan")
        _print_dry_run_scope(
            cells, families_by_cell, budgets, means, fallback, out, resume, confirm
        )
        return pd.DataFrame(columns=_BOARD_COLUMNS)
    result = run_board(
        cells,
        out=out,
        resume=resume,
        max_trials=max_trials,
        families_by_cell=families_by_cell,
        verbosity=verbosity,
        jobs=jobs,
    )
    _print_board_rollup(result)
    return result


def _print_dry_run_scope(
    cells: list[tuple[str, str]],
    families_by_cell: dict[tuple[str, str], tuple[str, ...]],
    budgets: dict[tuple[str, str], int],
    means: dict[tuple[str, str], float],
    fallback: float | None,
    out: str,
    resume: bool,
    confirm: bool,
) -> None:
    """Per cell: families, training ceiling, what ``--resume`` would reuse, and the time ceiling."""
    for cell in cells:
        n_families = len(families_by_cell[cell])
        line = (
            f"  {cell[0]} {cell[1]}  {n_families} famil{'y' if n_families == 1 else 'ies'}"
            f" · <={budgets[cell]} trainings"
        )
        if resume:
            reusable = len(_cached_corners(out, *cell, _cell_context(*cell)))
            line += f" · {reusable} prior rows reusable"
        seconds = means.get(cell, fallback)
        if seconds:
            line += f" · <={human_seconds(budgets[cell] * seconds)}"
        click.echo(line)
    if confirm:
        click.secho("  [dry-run] --confirm not exercised", fg="cyan")


def _run_single_cell(
    league: str,
    market: str,
    include_shipped: bool,
    dist_class: str,
    out: str,
    resume: bool,
    dry_run: bool,
    max_trials: int,
    verbosity: int,
    confirm: bool,
    jobs: int | None = None,
) -> pd.DataFrame:
    """Sweep one named cell, upsert it, print its summary; an empty board under ``dry_run``."""
    _note_board_only_flags(league, market, include_shipped, jobs)
    families = _cell_families(league, market, dist_class)
    if not families:
        raise click.UsageError(
            f"{league} {market}: no applicable family under --dist-class {dist_class}"
        )
    means, fallback = _cell_mean_seconds(out)
    budgets = {(league, market): _cell_trial_count(league, market, families, max_trials)}
    if dry_run:
        click.secho(f"[dry-run] {league} {market}: no corners trained", fg="cyan")
        _print_dry_run_scope(
            [(league, market)],
            {(league, market): families},
            budgets,
            means,
            fallback,
            out,
            resume,
            confirm,
        )
        return pd.DataFrame(columns=_BOARD_COLUMNS)
    result = search_cell(
        league,
        market,
        max_trials=max_trials,
        families=families,
        cached=(
            _cached_corners(out, league, market, _cell_context(league, market)) if resume else None
        ),
        out=out,
        verbosity=verbosity,
        seed_seconds=means.get((league, market), fallback),
    )
    _upsert_cell(result, out)
    _print_cell_summary(result, verbosity=verbosity)
    return result


def _reject_inert_flags(
    league: str | None,
    market: str | None,
    confirm: bool,
    yes: bool,
    confirm_nominees: int | None,
    confirm_hours: float | None,
    confirm_fresh_only: bool,
    confirm_auto_promote: bool,
    quiet: bool,
    verbose: bool,
) -> None:
    """Fail on flag combinations that would silently do nothing.

    Each of these was accepted and ignored before, which on an unattended multi-day run reads as the
    flag having taken effect. Flags that are merely *redundant* on a named cell get a note from
    :func:`_note_board_only_flags` instead — they are not mistakes, just inert at that scope.
    """
    if market is not None and not league:
        raise click.UsageError(
            "pass --league with --market for a single cell, or omit --market to sweep the board"
        )
    passed = [
        name
        for name, is_set in (
            ("--yes", yes),
            ("--confirm-nominees", confirm_nominees is not None),
            ("--confirm-hours", confirm_hours is not None),
            ("--confirm-fresh-only", confirm_fresh_only),
            ("--confirm-auto-promote", confirm_auto_promote),
        )
        if is_set
    ]
    if passed and not confirm:
        raise click.UsageError(f"{', '.join(passed)} require --confirm")
    if confirm_fresh_only and confirm_auto_promote:
        raise click.UsageError(
            "--confirm-auto-promote only affects the live lane that --confirm-fresh-only drops"
        )
    if quiet and verbose:
        raise click.UsageError("--quiet and --verbose are mutually exclusive")


def _default_jobs() -> int:
    """Concurrent cells when ``--jobs`` is unset."""
    return min(_MAX_AUTO_JOBS, max(1, (os.cpu_count() or 2) // 2))


def _note_board_only_flags(
    league: str, market: str, include_shipped: bool, jobs: int | None
) -> None:
    """Say so when a flag that only shapes a *board* run was passed alongside a named cell.

    Neither is a mistake worth failing on — one is the board default and the other widens cell
    selection this path never reaches — but silently ignoring a flag on a multi-hour run reads as
    the flag having taken effect.
    """
    notes = []
    if include_shipped:
        notes.append(
            f"--include-shipped is redundant here — {league} {market} sweeps whatever its shipped "
            "state; the flag only widens which cells a board run picks up"
        )
    if jobs is not None and jobs > 1:
        notes.append(
            f"--jobs {jobs} does nothing here — a named cell is one cell, and the corners within "
            "one are swept in sequence"
        )
    for note in notes:
        click.secho(wrapped(f"note: {note}.", "  ", "  "), fg="cyan")


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
    "withheld cells). Off by default. Redundant with --market: a named cell is swept whatever its "
    "shipped state.",
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
    help="Per-cell budget of *trained* corners; a re-proposed or cached corner is free. Early "
    "stopping usually ends a cell well under it, so every '<=N' count is a ceiling, not an estimate.",
)
@click.option(
    "-j",
    "--jobs",
    type=click.IntRange(min=1),
    default=None,
    help="Cells to sweep at once (default: half the cores, capped at 8). A deterministic corner is a "
    "single-threaded subprocess that never opens the archive, so cells parallelize cleanly; corners "
    "within a cell do not. No effect with --market, which is one cell.",
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
    help="Skip the --confirm prompt (unattended). Requires --confirm.",
)
@click.option(
    "--confirm-nominees",
    type=click.IntRange(min=1),
    default=None,
    help="Cap each cell's confirm walk at its first N nominees, highest board slack first. "
    "Unset walks every nominee. Requires --confirm.",
)
@click.option(
    "--confirm-hours",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Wall-clock budget for the confirm walks; cells not yet started when it runs out are "
    "skipped. Cells walk best board rank first, so the cut lands on the weakest tail. "
    "Requires --confirm.",
)
@click.option(
    "--confirm-fresh-only",
    is_flag=True,
    default=False,
    help="Confirm withheld cells only; skip live-cell supersession tests, whose promotions prompt "
    "per cell and so need an attended run. Requires --confirm.",
)
@click.option(
    "--confirm-auto-promote",
    is_flag=True,
    default=False,
    help="Answer the per-cell live-promotion prompt yes, so an unattended run can swap live cells. "
    "The S1/S2/S3 verdict and the six gates still decide each swap; only the human veto is gone, "
    "and every swap stays local until the stat_meta.json diff is reviewed. Requires --confirm.",
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
    help="Print the resolved scope — per cell: families, training ceiling, what --resume would "
    "reuse, and the time ceiling — then exit without training a single corner.",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="Progress bar and the per-cell verdict only. A redirected run still gets a periodic "
    "heartbeat, so a multi-hour cell is never a blank log.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="One line per corner, cache hits included, rather than only the notable ones.",
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
    jobs: int | None,
    confirm: bool,
    yes: bool,
    confirm_nominees: int | None,
    confirm_hours: float | None,
    confirm_fresh_only: bool,
    confirm_auto_promote: bool,
    resume: bool,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
    out: str | None,
) -> None:
    """Operation Ship 75 strategy sweep — one conditional-TPE study per cell over the strategy
    catalog, each corner scored on a holdout-blind cross-fit gate row. Naming ``--league`` and
    ``--market`` sweeps one cell; omitting ``--market`` sweeps the board (``--league`` narrows it).
    ``--confirm`` then full-HPO-retrains each cell's nominees until one ships.
    """
    _reject_inert_flags(
        league,
        market,
        confirm,
        yes,
        confirm_nominees,
        confirm_hours,
        confirm_fresh_only,
        confirm_auto_promote,
        quiet,
        verbose,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    verbosity = QUIET if quiet else VERBOSE if verbose else NORMAL
    out = out or str(STRATEGY_RESEARCH_BOARD)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if market is None:
        result = _run_board_mode(
            league,
            include_shipped,
            dist_class,
            out,
            resume,
            dry_run,
            max_trials,
            verbosity,
            confirm,
            jobs if jobs is not None else _default_jobs(),
        )
    else:
        result = _run_single_cell(
            league,
            market,
            include_shipped,
            dist_class,
            out,
            resume,
            dry_run,
            max_trials,
            verbosity,
            confirm,
            jobs,
        )
    if dry_run:
        return
    click.echo(f"\nswept in {human_seconds(time.monotonic() - started)} · board: {out}")

    if confirm:
        from sportstradamus.training.model_strategy.confirm import run_confirm

        run_confirm(
            result,
            yes=yes,
            max_nominees=confirm_nominees,
            deadline_hours=confirm_hours,
            fresh_only=confirm_fresh_only,
            auto_promote=confirm_auto_promote,
        )


if __name__ == "__main__":
    main()
