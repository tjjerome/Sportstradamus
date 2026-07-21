"""Per-cell deterministic strategy sweep ranked by six-gate ship slack.

Every applicable, explicitly enrolled :class:`model_strategy_registry.StrategySpec` contributes
its categorical grid. Each corner trains in a research-only namespace and is scored through the
production scorecard with its validation-fit calibration intact. Canonical spec/control
fingerprints make the board resumable without accepting stale or legacy strategy coverage.

Deterministic trials rank only. ``--confirm`` persists the winner and requires a clean full-HPO
6/6 before a withheld model can ship.
"""

import functools
import hashlib
import importlib.resources as pkg_resources
import math
import pathlib
import subprocess
import time
from datetime import UTC, datetime

import click
import optuna
import pandas as pd
import pyarrow.parquet as pq
import tabulate

from sportstradamus import data as _data_pkg
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.model_strategy_artifacts import (
    InactiveStrategyArtifactError,
    build_artifact_identity,
    validate_strategy_artifacts,
)
from sportstradamus.training.model_strategy_execution import (
    artifact_namespace,
    meditate_command,
    strategy_cli_args,
    strategy_persistence_edits,
)
from sportstradamus.training.model_strategy_registry import (
    BASE_STRUCTURAL_STRATEGY,
    SWEEP_CAPABILITIES,
    CellContext,
    StrategySpec,
    controls_json,
    corner_fingerprint,
    distribution_class,
    get_strategy,
    parse_controls,
    strategies_for_cell,
    strategy_controls,
)
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
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
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
# The six offline ship gates (value + pass); a corner ships iff all pass. Mirrors
# scorecard._SHIP_GATES — apply_thresholds sets `ship` and min_gate_slack folds in all six.
_GATES: tuple[str, ...] = ("g1", "g2", "g3", "g4", "g5", "g6")

# A retrain corner can error mid-sweep — an invalid family/loss combo for the cell's data-driven
# dist (e.g. a SkewNormal `dist_training_loss=crps` corner on a low-mean cell the pipeline trains as
# ZINB), a timeout, or a numerical blow-up. A sweep over hundreds of corners records the bad one
# non-shipping and continues rather than losing the whole board; -inf slack sorts it last and keeps
# it out of every ship/candidate filter.
_FAILED_CORNER_SLACK: float = float("-inf")

# `--dist-class all` sweeps every registered distribution class.
_DIST_CLASSES: tuple[str, ...] = ("continuous", "count")
_DIST_CLASS_ALL = "all"

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
    "swept_at",
    "code_rev",
]

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
    data_columns, matrix_sha256 = _training_matrix_contract(league, market)
    return CellContext(
        league=league,
        market=market,
        distribution=dist,
        distribution_class=dist_class,
        data_columns=data_columns,
        matrix_sha256=matrix_sha256,
    )


def _training_matrix_path(league: str, market: str) -> pathlib.Path:
    return _TRAINING_DATA_ROOT / f"{market_file_slug(league, market)}.parquet"


@functools.lru_cache(maxsize=256)
def _read_training_matrix_contract(
    path_text: str, mtime_ns: int, size: int
) -> tuple[frozenset[str], str]:
    """Read only a cached matrix's schema and stream its exact bytes into SHA256."""
    del mtime_ns, size  # cache-key inputs; the path contents are read below
    path = pathlib.Path(path_text)
    columns = frozenset(pq.read_schema(path).names)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return columns, digest.hexdigest()


def _training_matrix_contract(league: str, market: str) -> tuple[frozenset[str], str]:
    """Return cached-matrix schema + SHA, failing loud when the frozen input is unusable."""
    path = _training_matrix_path(league, market)
    try:
        stat = path.stat()
        return _read_training_matrix_contract(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError) as exc:
        raise click.UsageError(
            f"{league} {market}: unreadable cached training matrix {path}"
        ) from exc


def _cell_families(league: str, market: str) -> tuple[str, ...]:
    """Registered sweep strategies applicable and explicitly enrolled for a cell."""
    return tuple(
        spec.slug
        for spec in strategies_for_cell(
            _cell_context(league, market), required_capabilities=SWEEP_CAPABILITIES
        )
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


def _dump_subdir(corner: dict[str, str], spec: StrategySpec) -> str:
    """Resolve the canonical deterministic namespace for one strategy corner."""
    norm = corner.get("normalization", TARGET_NORM_NONE)
    trained_dist = corner.get("dist", spec.family)
    suffix = "_hurdle" if trained_dist == "ZINB" and corner.get("zinb_mode") == "hurdle" else ""
    return artifact_namespace(f"{norm}{suffix}", spec)


def _decode_strategy(corner: dict[str, str]) -> str:
    """Use a registered continuous transform, or canonical no-transform for a count corner."""
    return corner.get("normalization", TARGET_NORM_NONE)


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


def _run_meditate_with_lock_retry(cmd: list[str], log_path: pathlib.Path, *, timeout: int) -> None:
    """Run a ``meditate`` subprocess, capturing output to ``log_path``, retrying an archive-lock clash.

    A non-zero exit whose log shows the DuckDB write-lock collision (:data:`_LOCK_ERROR_SIGNATURE`)
    is transient — a cron archive job holds the lock — so wait per :data:`_LOCK_RETRY_WAITS_S` and
    retry (each attempt truncates the log, so a final tail reflects the last try). Every other failure
    — a real non-zero exit, a timeout, or exhausted lock retries — surfaces the log tail and re-raises,
    so the deterministic sweep stays fail-loud (the crash kills the Optuna study) and the confirm loop
    can turn the raise into a HELD/REVERTED verdict.
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
            except subprocess.CalledProcessError:
                retries_left = attempt < len(_LOCK_RETRY_WAITS_S)
                if retries_left and _is_archive_lock_error(log_path):
                    wait = _LOCK_RETRY_WAITS_S[attempt]
                    click.echo(f"  archive write-locked; retrying in {wait}s …", err=True)
                    time.sleep(wait)
                    continue
                click.echo(
                    f"  meditate failed — tail of {log_path}:\n{_log_tail(log_path)}", err=True
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
    sandbox (never production); ``--bypass-withholding`` lets a withheld cell train. Each corner axis
    is forwarded as its ``--flag value`` (:data:`_AXIS_FLAG`) so the sweep varies it; an axis the
    family doesn't sweep is left off and meditate resolves its default. The trained model is a
    *ranking* stand-in.

    meditate's full training log is captured to a per-corner file rather than streamed, so the sweep's
    own progress and verdict stay readable; :func:`_run_meditate_with_lock_retry` surfaces a failed
    trial's tail and retries a transient archive-lock clash.
    """
    cmd = meditate_command(
        league,
        market,
        "--deterministic",
        "--bypass-withholding",
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
) -> list[dict[str, object]]:
    """Train the ``corner`` retrain trial and score it; tag the row with its family.

    Calibration is not re-fit here: the dump already carries the pipeline's validation-fit
    calibration, and :func:`_score_corner` reads it off the dump via the production
    :func:`scorecard.gate_row` — no test re-fit. The dump is keyed by subdir, so scoring right after
    this trial's train — before the next sequential trial overwrites it — keeps the loss/mode axes
    honest without a per-corner dump path (which is also why the sweep always retrains rather than
    reusing a dump). Returns a one-row list so the GridSampler objective and board assembly read
    uniformly. A corner whose meditate errors or times out is caught, echoed, and returned as a
    single non-shipping row (:data:`_FAILED_CORNER_SLACK`) so one bad corner never aborts the board.
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
        row = {**corner, "slack": _FAILED_CORNER_SLACK, "ships": False}
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
        return [row]
    verdict = click.style(_verdict(row), fg="green" if row["ships"] else "red")
    elapsed = f"{time.monotonic() - start:.0f}s"
    click.echo(f"  {verdict}  {label}  slack {float(row['slack']):+.3f}  ({elapsed})")
    return [row]


def _run_family_study(league: str, market: str, strategy_slug: str) -> list[dict[str, object]]:
    """One GridSampler study over a registered strategy; return every scored corner.

    ``objective`` binds ``family`` as a default arg rather than a free variable — belt-and-suspenders
    against the classic loop-closure-capture bug even though it can't fire here: ``family`` is this
    function's parameter, fixed for the whole call, and :func:`search_cell` calls this function once
    per family rather than defining ``objective`` itself inside its own family loop.
    """
    spec = get_strategy(strategy_slug)
    if not spec.axes:
        return _run_and_score(league, market, strategy_slug, dict(spec.fixed_controls))
    grid = {axis: list(choices) for axis, choices in spec.axes.items()}
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(grid))

    def objective(trial: optuna.Trial, strategy_slug: str = strategy_slug) -> float:
        corner = {
            **spec.fixed_controls,
            **{axis: trial.suggest_categorical(axis, choices) for axis, choices in grid.items()},
        }
        rows = _run_and_score(league, market, strategy_slug, corner)
        trial.set_user_attr("rows", rows)
        return -max(float(row["slack"]) for row in rows)

    study.optimize(objective, n_trials=math.prod(len(choices) for choices in grid.values()))
    return [row for trial in study.trials for row in trial.user_attrs["rows"]]


def search_cell(league: str, market: str) -> pd.DataFrame:
    """Score every registered corner for a cell and rank the unified board by slack."""
    rows = [
        {"league": league, "market": market, **row}
        for strategy_slug in _cell_families(league, market)
        for row in _run_family_study(league, market, strategy_slug)
    ]
    board = pd.DataFrame(rows)
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


def _candidate_cells(
    league: str | None = None,
    include_shipped: bool = False,
    dist_class: str = _DIST_CLASS_ALL,
) -> list[tuple[str, str]]:
    """Registered-family cells eligible for the board, before the trainable/data filters.

    Withheld only by default (the ship path); ``include_shipped`` adds already-shipped cells so the
    board can hunt a better strategy for a live cell (evaluated by the separate supersession test,
    not the fresh-ship confirm). ``dist_class`` narrows to one distribution class (``count`` /
    ``continuous``); the ``all`` default keeps every class. Self-maintaining — it follows stat_meta's
    ``shipped`` field.
    """
    meta = load_stat_meta(pathlib.Path(str(STAT_META_PATH)))
    cells: list[tuple[str, str]] = []
    unsupported: list[tuple[str, str, object]] = []
    for lg, markets in meta.items():
        if league is not None and lg != league:
            continue
        for mkt, cell in markets.items():
            if mkt not in ALL_MARKETS.get(lg, []):
                continue
            if not include_shipped and cell.get("shipped") != WITHHELD:
                continue
            try:
                cell_dist_class = distribution_class(cell.get("dist"))
            except (TypeError, ValueError):
                unsupported.append((lg, mkt, cell.get("dist")))
                continue
            if dist_class in (_DIST_CLASS_ALL, cell_dist_class):
                cells.append((lg, mkt))
    if unsupported:
        rendered = ", ".join(f"{lg} {mkt} ({dist!r})" for lg, mkt, dist in unsupported)
        raise click.UsageError(f"board contains unsupported distribution cells: {rendered}")
    return cells


def _select_board_cells(
    league: str | None = None,
    include_shipped: bool = False,
    dist_class: str = _DIST_CLASS_ALL,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split eligible family cells into ``(sweepable, missing_data)``.

    Sweepable = eligible (see :func:`_candidate_cells`), present in the trainable ``ALL_MARKETS``
    registry (stat_meta carries non-market entries like inning props that meditate rejects), and with
    a cached training matrix. ``missing_data`` are eligible registry cells whose matrix is absent —
    the deterministic sweep can't build one, so they are surfaced as a warning rather than swept.
    ``dist_class`` narrows the cohort to one distribution class (the WS-3 count residual runs
    ``--dist-class count``).
    """
    in_registry = _candidate_cells(league, include_shipped, dist_class)
    sweepable = [c for c in in_registry if _has_training_data(*c)]
    missing = [c for c in in_registry if not _has_training_data(*c)]
    return sweepable, missing


def _cell_corner_count(league: str, market: str) -> int:
    """The deterministic trainings one cell contributes across all registered strategies."""
    return sum(
        len(strategy_controls(get_strategy(slug))) for slug in _cell_families(league, market)
    )


def _corner_count(cells: list[tuple[str, str]]) -> int:
    """Total deterministic trainings a board run will do — each cell's family grids summed."""
    return sum(_cell_corner_count(lg, mkt) for lg, mkt in cells)


def _board_done_cells(out: str | None) -> set[tuple[str, str]]:
    """Cells carrying the exact current set of signed strategy-corner fingerprints."""
    if out is None or not pathlib.Path(out).exists():
        return set()
    prior = _read_board(pathlib.Path(out))
    complete: set[tuple[str, str]] = set()
    for (league, market), cell_rows in prior.groupby(["league", "market"], sort=False):
        try:
            expected = _expected_corner_records(league, market)
        except (click.UsageError, ValueError):
            continue
        if _cell_rows_match_expected(cell_rows, expected):
            complete.add((league, market))
    return complete


def _read_board(path: pathlib.Path) -> pd.DataFrame:
    """Load any legacy board into the current ordered schema."""
    board = pd.read_csv(path, keep_default_na=False).replace("", pd.NA)
    for column in _BOARD_COLUMNS:
        if column not in board:
            board[column] = pd.NA
    return board.reindex(columns=_BOARD_COLUMNS)


def _expected_corner_records(league: str, market: str) -> dict[str, dict[str, object]]:
    """Current canonical board identity for every corner registered for one cell."""
    expected: dict[str, dict[str, object]] = {}
    context = _cell_context(league, market)
    for slug in _cell_families(league, market):
        spec = get_strategy(slug)
        for controls in strategy_controls(spec):
            identity = build_artifact_identity(
                spec.slug,
                league,
                market,
                controls,
                matrix_hash=str(context.matrix_sha256),
            )
            key = controls_json({"strategy_slug": spec.slug, "controls": controls})
            expected[key] = {
                "family": spec.family,
                "strategy_slug": identity.strategy_slug,
                "structural_strategy": identity.structural_strategy,
                "strategy_signature": identity.signature,
                "strategy_implementation_version": identity.implementation_version,
                "artifact_schema_version": identity.artifact_schema_version,
                "strategy_status": identity.status,
                "controls_json": controls_json(controls),
                "matrix_hash": context.matrix_sha256,
                **{name: value for name, value in controls.items() if name in _AXIS_COLUMNS},
            }
    return expected


def _cell_rows_match_expected(rows: pd.DataFrame, expected: dict[str, dict[str, object]]) -> bool:
    identity_columns = (
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
    if rows.empty or any(rows[column].isna().any() for column in identity_columns):
        return False
    if len(rows) != len(expected) or rows["corner_fingerprint"].duplicated().any():
        return False
    for _, source in rows.iterrows():
        row = source.loc[list(identity_columns)].to_dict()
        try:
            controls = parse_controls(row["controls_json"])
        except (TypeError, ValueError):
            return False
        key = controls_json({"strategy_slug": row["strategy_slug"], "controls": controls})
        contract = expected.get(key)
        if contract is None:
            return False
        for field, value in contract.items():
            actual = source.get(field)
            if pd.isna(actual) or str(actual) != str(value):
                return False
        spec = get_strategy(str(row["strategy_slug"]))
        split = source.get("split_fingerprint")
        if pd.isna(split):
            split = None
        if spec.split_fingerprint_path and not isinstance(split, str):
            return False
        if not spec.split_fingerprint_path and split is not None:
            return False
        fingerprint = corner_fingerprint(spec, controls, str(row["matrix_hash"]))
        if row["corner_fingerprint"] != fingerprint:
            return False
    return True


def run_board(
    cells: list[tuple[str, str]], out: str | None = None, resume: bool = False
) -> pd.DataFrame:
    """Search every cell in ``cells``, printing each verdict as it lands and upserting the board CSV
    per cell so an interrupt keeps partial progress and a ``--league``-scoped run leaves other
    leagues' rows intact. With ``resume``, cells already on the CSV are skipped and their rows carry
    through only when they are in the requested scope, so confirmation cannot consume unrelated
    prior winners. Returns the requested cells' board; CSV upserts preserve all unrelated rows.
    """
    boards: list[pd.DataFrame] = []
    done = _board_done_cells(out) if resume else set()
    if resume and out is not None and pathlib.Path(out).exists():
        prior = _read_board(pathlib.Path(out))
        prior_cells = pd.MultiIndex.from_frame(prior[["league", "market"]])
        requested_done = done & set(cells)
        boards.append(prior[prior_cells.isin(requested_done)])
    for league, market in cells:
        if (league, market) in done:
            continue
        cell_board = search_cell(league, market)
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
) -> pd.DataFrame:
    """Derive the board, warn per cell skipped for a missing training matrix, print the scope, sweep.

    ``dist_class`` narrows the cohort to one distribution class; ``resume`` skips cells already on the
    board CSV; ``dry_run`` prints the resolved scope (and what resume would skip) then returns without
    training a single corner.
    """
    cells, missing = _select_board_cells(league, include_shipped, dist_class)
    for lg, mkt in missing:
        click.secho(
            f"  skip {lg} {mkt}: no cached training matrix — train it first "
            "(the deterministic sweep won't rebuild it)",
            fg="yellow",
        )
    if not cells:
        raise click.UsageError(
            f"no trainable cells with cached data to sweep{f' in {league}' if league else ''}."
        )
    done = _board_done_cells(out) if resume else set()
    todo = [c for c in cells if c not in done]
    scope = f" ({league})" if league else ""
    note = f" · {len(missing)} skipped (no cached matrix)" if missing else ""
    resumed = f" · {len(done & set(cells))} already on board (resume)" if resume else ""
    click.echo(
        f"board{scope}: {len(todo)}/{len(cells)} cells to sweep{note}{resumed} "
        f"· ~{_corner_count(todo)} deterministic trainings"
    )
    if dry_run:
        click.secho("  [dry-run] no corners trained", fg="cyan")
        return pd.DataFrame(columns=_BOARD_COLUMNS)
    result = run_board(cells, out=out, resume=resume)
    _print_board_rollup(result)
    return result


@click.command(name="model-strategy-sweep")
@click.option(
    "--league", default=None, help="League code, e.g. WNBA. Single-cell mode, or narrows --board."
)
@click.option("--market", default=None, help="Market stem, e.g. AST (single-cell mode).")
@click.option(
    "--board/--no-board",
    default=False,
    help="Sweep every withheld cell with cached training data (all strategies) instead of one cell; "
    "--league narrows it.",
)
@click.option(
    "--include-shipped",
    is_flag=True,
    default=False,
    help="Also sweep already-shipped (devel/main) cells to hunt a better strategy — evaluated by the "
    "supersession test, not the fresh-ship --confirm (which only auto-ships withheld cells). Off by default.",
)
@click.option(
    "--dist-class",
    type=click.Choice([*_DIST_CLASSES, _DIST_CLASS_ALL]),
    default=_DIST_CLASS_ALL,
    show_default=True,
    help="Board mode: narrow to one distribution class — 'count' (ZINB + NegBin) sweeps the WS-3 "
    "count residual, 'continuous' the SkewNormal cells.",
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
    "--resume",
    is_flag=True,
    default=False,
    help="Board mode: skip cells already on the board CSV and keep their rows — resume a crashed "
    "multi-hour run instead of re-sweeping from scratch.",
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
    "(data/research/strategy_research_board.csv): --board upserts per cell, a single cell upserts.",
)
def main(
    league: str | None,
    market: str | None,
    board: bool,
    include_shipped: bool,
    dist_class: str,
    confirm: bool,
    yes: bool,
    resume: bool,
    dry_run: bool,
    out: str | None,
) -> None:
    """Operation Ship 75 strategy sweep — a per-cell GridSampler over the strategy catalog, one
    honest val-fit→test gate row per corner. ``--confirm`` then ships the winners end-to-end.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = out or str(STRATEGY_RESEARCH_BOARD)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    if board:
        result = _run_board_mode(league, include_shipped, dist_class, out, resume, dry_run)
    else:
        if not (league and market):
            raise click.UsageError("pass --league and --market, or --board")
        if dry_run:
            families = _cell_families(league, market)
            click.secho(
                f"[dry-run] {league} {market}: families {', '.join(families)} "
                f"· {_cell_corner_count(league, market)} corners",
                fg="cyan",
            )
            return
        result = search_cell(league, market)
        _upsert_cell(result, out)
        _print_cell_summary(result)
    click.echo(f"\nboard: {out}")

    if confirm and not dry_run:
        from sportstradamus.training.model_strategy_confirm import run_confirm

        run_confirm(result, yes=yes)


if __name__ == "__main__":
    main()
