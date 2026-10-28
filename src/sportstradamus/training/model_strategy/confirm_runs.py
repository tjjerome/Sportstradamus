"""Confirm-walk run machinery: what one nominee goes through between nomination and verdict.

The walk (:mod:`sportstradamus.training.model_strategy.confirm`) calls in here to pin a cell's
training matrix, run a nominee blind at full HPO (plus the g4-only calibrated retry), write
stat_meta.json and snapshot/restore the cell's serve-read artifacts around the pick's canonical
refit, and read that refit's gates and identity back from model_stats. Every run's gate row lands in
the nominee ledger.
"""

import json
import pathlib
import shutil
import subprocess
import time
from typing import NamedTuple

import click
import pandas as pd

from sportstradamus.helpers.io import MODEL_STATS_PATH, market_file_slug, model_pickle_path
from sportstradamus.stats.model_dependencies import DEPENDENCY_NAMESPACE
from sportstradamus.training.lineage import (
    MATRIX_BUILDER_VERSION,
    MATRIX_SCHEMA_VERSION,
    file_sha,
)

# _cell_context and NOMINEE_LEDGER_PATH are read off sweep at call time, not imported: other
# confirm modules read them too, and one binding means one monkeypatch covers every caller.
from sportstradamus.training.model_strategy import sweep
from sportstradamus.training.model_strategy.identity import (
    ArtifactIdentity,
    corner_fingerprint,
    validate_strategy_artifacts,
)
from sportstradamus.training.model_strategy.progress import elapsed_ticker, human_seconds
from sportstradamus.training.model_strategy.registry import (
    controls_json,
    get_strategy,
    meditate_command,
    strategy_cli_args,
    strategy_full_hpo_cli_args,
)
from sportstradamus.training.model_strategy.sweep import (
    _DIVERGED_DISPERSION_CAL,
    _GATES,
    _TEST_SETS_ROOT,
    EVAL_SPLIT_CROSSFIT,
    EVAL_SPLIT_TEST,
    _crossfit_ledger,
    _failed_gates,
    _failure_reason,
    _run_meditate_with_lock_retry,
    _score_corner,
    _training_matrix_path,
)
from sportstradamus.training.report import _IDENTITY_COLUMNS
from sportstradamus.training.scorecard import _GATE1_NONINF_MARGIN
from sportstradamus.training.ship_config import STAT_META_PATH

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
STAT_META = pathlib.Path(str(STAT_META_PATH))
_CONFIRM_LOG_ROOT = _REPO_ROOT / "research" / "logs" / "confirm"
# A full-HPO blind run is ~1 h; a large cell can run longer, so a 4 h ceiling keeps a hung run
# from blocking the loop forever. A timeout is a failure: the nominee fails blind, or a canonical
# refit auto-reverts.
_CONFIRM_TIMEOUT_S = 4 * 3600
# The exact model_stats identity columns the official ship verdict is bound to.
_SHIP_IDENTITY_COLUMNS = [
    "league",
    "market",
    "strategy_slug",
    "structural_strategy",
    "strategy_signature",
    "strategy_implementation_version",
    "artifact_schema_version",
    "strategy_status",
    "strategy_controls_json",
    "strategy_corner_fingerprint",
    "strategy_matrix_hash",
    "strategy_split_fingerprint",
    "ship",
]
# The six ship-gate booleans in model_stats.parquet (nullable BooleanDtype; pd.NA means the
# scorecard never ran, which counts as not passing here).
_GATE_COLS: tuple[str, ...] = tuple(f"{g}_pass" for g in _GATES)
# brief R3: separates the 9 addressable near-miss ledger rows from the 4 structural
# failures at excess >= 0.049.
_G4_RETRY_MAX_EXCESS: float = 0.010
# brief R3: noise band on the g1 non-inferiority CI bound.
_G1_RETRY_NOISE_BAND: float = 0.002
# The blind scorer speaks the board's column names, the ledger model_stats'. These fields have a
# model_stats twin; the scorer's other fields keep their names, and its controls are dropped for the
# ``strategy_controls_json`` the identity columns carry.
_LEDGER_NAMES: dict[str, str] = {
    "ships": "ship",
    "n": "n_validation",
    "matrix_hash": "strategy_matrix_hash",
    "split_fingerprint": "strategy_split_fingerprint",
}


def atomic_write_meta(meta: dict) -> None:
    """Write stat_meta.json atomically at its native 4-space indent for a minimal review diff."""
    tmp = STAT_META.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(meta, fh, indent=4)
        fh.write("\n")
    tmp.replace(STAT_META)


def backup_stat_meta() -> pathlib.Path:
    """Copy the whole stat_meta.json to a timestamped sibling — the crash/abort recovery point."""
    backup = STAT_META.with_name(f"stat_meta.{time.strftime('%Y%m%dT%H%M%S')}.bak.json")
    shutil.copy2(STAT_META, backup)
    return backup


def _cell_artifacts(league: str, market: str) -> list[pathlib.Path]:
    """Every serve-read artifact a canonical refit rewrites for one cell — the restore set.

    Restoring these puts the incumbent back exactly as it served: the model pickle (which carries
    all calibrators), the test-set CSV (also the snapshotted S2/S3 baseline), ``model_stats``
    (parquet + csv mirror), the two SHAP CSVs, and the two config files read at serve time —
    ``stat_calibration`` and ``book_weights`` (shared, its per-cell key reverted by the whole-file
    restore). The shared files are safe to whole-file restore because a cell's snapshot→restore
    window is isolated (only that cell's ``meditate`` runs between them, so the revert touches only
    that cell's row/key).

    Deliberately excluded: the per-cell training-matrix cache (``training_data/{slug}.parquet``) and
    the per-league caches (gamelog, correlation matrices). Those are training inputs,
    never read at serve time — and a confirm retrain trains from the walk's frozen snapshot
    (``--frozen-matrix-dir``) without touching any of them.
    """
    slug = market_file_slug(league, market)
    training_dir = MODEL_STATS_PATH.parent
    config_dir = STAT_META.parent
    return [
        model_pickle_path(league, market),
        _TEST_SETS_ROOT / f"{slug}.csv",
        MODEL_STATS_PATH,
        MODEL_STATS_PATH.with_suffix(".csv"),
        training_dir / "feature_importances.csv",
        training_dir / "feature_correlations.csv",
        config_dir / "stat_calibration.json",
        config_dir / "book_weights.json",
    ]


def snapshot_cell(league: str, market: str) -> pathlib.Path:
    """Copy the incumbent's artifacts to a per-cell backup dir and return it."""
    backup = _CONFIRM_LOG_ROOT / "incumbent_backup" / market_file_slug(league, market)
    shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir(parents=True, exist_ok=True)
    for art in _cell_artifacts(league, market):
        if art.exists():
            shutil.copy2(art, backup / art.name)
    return backup


def restore_cell(
    league: str, market: str, backup: pathlib.Path, meta: dict, original: dict
) -> None:
    """Restore every snapshotted artifact byte-identical and put the original stat_meta entry back.

    Copies files from ``backup`` over the canonical paths — it never prunes — so a live cell that
    loses the supersession test keeps serving exactly what it served before.
    """
    for art in _cell_artifacts(league, market):
        saved = backup / art.name
        if saved.exists():
            shutil.copy2(saved, art)
        elif art.exists():
            art.unlink()
    meta[league][market] = original
    atomic_write_meta(meta)


def _cell_row(league: str, market: str, columns: list[str] | None) -> pd.Series | None:
    """One cell's row of the requested model_stats.parquet columns, or ``None`` when absent.

    ``None`` columns reads the whole row.
    """
    stats = pd.read_parquet(MODEL_STATS_PATH, columns=columns)
    hit = stats[(stats["league"] == league) & (stats["market"] == market)]
    return None if hit.empty else hit.iloc[0]


def _split_matches(actual: object, expected: str | None) -> bool:
    """Whether a model_stats split fingerprint matches the expected value (both-absent counts)."""
    if expected is None:
        return bool(pd.isna(actual))
    return actual == expected


def _ship_from_model_stats(league: str, market: str, expected: ArtifactIdentity) -> bool:
    """The official ship verdict bound to the exact reported strategy contract."""
    try:
        row = _cell_row(league, market, _SHIP_IDENTITY_COLUMNS)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if row is None or not bool(row["ship"]) or row["strategy_status"] != "active":
        return False
    checks = {
        "strategy_slug": expected.strategy_slug,
        "structural_strategy": expected.structural_strategy,
        "strategy_signature": expected.signature,
        "strategy_implementation_version": expected.implementation_version,
        "artifact_schema_version": expected.artifact_schema_version,
        "strategy_controls_json": expected.controls_json,
        # Not strategy_matrix_hash: expected is rebound to the hash this row reported, so comparing
        # them here would assert nothing. The fingerprint below re-derives from that hash instead.
        "strategy_corner_fingerprint": expected.corner_fingerprint,
    }
    if any(row[column] != value for column, value in checks.items()):
        return False
    return _split_matches(row["strategy_split_fingerprint"], expected.split_fingerprint)


def _record_ledger_row(candidate: dict, gates: dict, eval_split: str) -> bool:
    """Append one nominee's gate row to the research ledger, tagged with the half it was scored on.

    Each blind run writes an :data:`EVAL_SPLIT_CROSSFIT` row from the blind scorer, and a cell's one
    canonical refit writes its :data:`EVAL_SPLIT_TEST` row from model_stats. That model_stats row
    is the only record of the refit's gates: a losing refit's pickle is pruned, and ``report()``
    rebuilds model_stats from the pickles on disk, so by the time the walk reports its verdict the
    gate values behind it are gone. A board nominee's row also carries its board-side gate values
    (``board_*``, from its ``board_gates``); seed/incumbent rows leave them NaN. Ledger only;
    nothing production reads it.

    Returns whether the fit diverged — ``dispersion_cal`` on its floor
    (:data:`_DIVERGED_DISPERSION_CAL`) — which is also recorded as the ``diverged`` column.
    """
    dispersion = gates.get("dispersion_cal")
    diverged = bool(pd.notna(dispersion) and float(dispersion) <= _DIVERGED_DISPERSION_CAL)
    ledger = pd.DataFrame(
        [
            {
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_s": candidate.get("elapsed_s"),
                "strategy_slug": candidate["strategy_slug"],
                "source": candidate["source"],
                "diverged": diverged,
                **candidate.get("board_gates", {}),
                **gates,
                "eval_split": eval_split,
            }
        ]
    )
    if sweep.NOMINEE_LEDGER_PATH.exists():
        # Rewrite rather than append: model_stats widens as gates are added, and appending a
        # wider row under a header written by a narrower one yields a CSV no reader can parse.
        ledger = pd.concat([pd.read_csv(sweep.NOMINEE_LEDGER_PATH), ledger], ignore_index=True)
    sweep.NOMINEE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    return diverged


def _failed_gates_after(league: str, market: str) -> list[str]:
    """The gates a just-confirmed cell fails, read from model_stats — diagnostics for the report."""
    row = _cell_row(league, market, ["league", "market", *_GATE_COLS])
    if row is None:
        return ["(no model_stats row)"]
    return [g for g in _GATES if not bool(row[f"{g}_pass"])]


def _frozen_matrix_dir(league: str, market: str) -> pathlib.Path:
    """The per-cell dir a walk pins its training matrix into (shared by pin and retrain)."""
    return _CONFIRM_LOG_ROOT / "frozen_matrix" / market_file_slug(league, market)


def pin_cell_matrix(league: str, market: str) -> pathlib.Path:
    """Copy the cell's cached training matrix + lineage manifest into the walk's frozen dir.

    Every blind run of the walk, and the cell's canonical refit, then trains via
    ``--frozen-matrix-dir`` on this one frame — an unpinned ``--force`` rewrote the matrix between
    nominees, so they scored different frames (n_validation drifted 373/363/358 within one walk).
    Two side effects of meditate's frozen-input mode are deliberate: a blind run cold-starts its
    HPO (warm pickle params are ignored, so every nominee gets the same full search budget) and it
    skips the per-market book-weight refits (the same isolation as the board's deterministic runs,
    keeping board and confirm aligned). A swept cell always has a cached parquet — the sweep
    requires one — so a missing source fails loud here. The snapshot stays on disk after the walk
    (research/ is gitignored), matching the ``incumbent_backup`` pattern.
    """
    source = _training_matrix_path(league, market)
    frozen_dir = _frozen_matrix_dir(league, market)
    shutil.rmtree(frozen_dir, ignore_errors=True)
    frozen_dir.mkdir(parents=True)
    frozen = frozen_dir / source.name
    shutil.copy2(source, frozen)
    matrix = pd.read_parquet(frozen)
    # Exactly the five fields lineage.validate_matrix_manifest checks on the frozen path.
    manifest = {
        "builder_version": MATRIX_BUILDER_VERSION,
        "schema_version": MATRIX_SCHEMA_VERSION,
        "row_count": len(matrix),
        "feature_schema": list(matrix.columns),
        "matrix_sha256": file_sha(frozen),
    }
    frozen.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return frozen_dir


def _run_meditate(
    candidate: dict,
    args: list[str],
    *,
    log_path: pathlib.Path,
    label: str,
    expected: float | None = None,
) -> str:
    """One ``meditate`` of a nominee on the walk's pinned matrix; "" iff the subprocess exits clean.

    Every run of a walk — each blind run and the cell's canonical refit — trains on the frame
    :func:`pin_cell_matrix` froze (``--frozen-matrix-dir``), so they all score the same rows.
    ``--force`` is still required: a frozen-input run appends no new rows, and without it a cell
    whose pickle already exists skips silently and never rewrites its outputs. A failure returns its
    :func:`_failure_reason` rather than a bare flag — a native abort inside the fit and an ordinary
    non-zero exit call for completely different triage, and reporting both as one "retrain error"
    is what sent the last investigation looking for a Python exception that never existed. The
    caller must not trust a possibly-stale artifact either way; a transient archive-lock clash is
    retried first (:func:`_run_meditate_with_lock_retry`).
    """
    lg, mkt = candidate["league"], candidate["market"]
    cmd = meditate_command(
        lg, mkt, "--force", "--frozen-matrix-dir", str(_frozen_matrix_dir(lg, mkt)), *args
    )
    click.echo(f"  {label} {lg} {mkt} — {_expectation(expected)} …")
    started = time.monotonic()
    try:
        with elapsed_ticker(f"  {lg} {mkt} {label}", expected):
            _run_meditate_with_lock_retry(cmd, log_path, timeout=_CONFIRM_TIMEOUT_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return _failure_reason(exc)
    finally:
        # Carried on the candidate rather than a module global: the same dict already threads the
        # walk, and _record_ledger_row is the next thing to read it.
        candidate["elapsed_s"] = round(time.monotonic() - started, 1)
    click.echo(f"  {label} {lg} {mkt} done in {human_seconds(candidate['elapsed_s'])}")
    return ""


def _expectation(expected: float | None) -> str:
    """How long this run should take, and the wall-clock time that lands at.

    A redirected run gets no ticker, so the finish time is what makes a stalled run obvious in a
    log — an elapsed count would need the reader to know when the line was written.
    """
    if expected is None:
        return "no timing estimate"
    finish = time.strftime("%H:%M", time.localtime(time.time() + expected))
    return f"~{human_seconds(expected)}, expect ~{finish}"


def confirm_median_seconds(league: str | None = None, market: str | None = None) -> float | None:
    """Median blind-run seconds for one cell from the nominee ledger, else across all cells.

    Naming no cell asks for the all-cells median, which is what sizes the up-front confirm prompt.

    Only blind rows count: each times one full-HPO search (a calibrated retry records its own
    row), while a canonical refit reuses its blind run's parameters and takes minutes. A run that
    errored never reaches the ledger, so the estimate is blind to the slowest failures — hence it is
    always shown as approximate. Median guards the estimate against that skew.
    """
    if not sweep.NOMINEE_LEDGER_PATH.exists():
        return None
    ledger = _crossfit_ledger(pd.read_csv(sweep.NOMINEE_LEDGER_PATH))
    if "elapsed_s" not in ledger.columns:
        return None
    ledger = ledger.assign(elapsed_s=pd.to_numeric(ledger["elapsed_s"], errors="coerce")).dropna(
        subset=["elapsed_s"]
    )
    if ledger.empty:
        return None
    cell = ledger[(ledger["league"] == league) & (ledger["market"] == market)]
    scope = ledger if league is None or cell.empty else cell
    return float(scope["elapsed_s"].median())


def _candidate_identity(candidate: dict, matrix_hash: str | None = None) -> ArtifactIdentity:
    """The nominated recipe's identity, rebound to ``matrix_hash`` when a retrain moved the matrix.

    ``_run_meditate`` passes ``--force``, which updates the league gamelogs and appends gamedays, so
    a confirm retrain routinely lands on a different training matrix than the search scored — every
    cell of the first overnight run did. What confirmation tests is the *recipe*; the board's
    pre-retrain matrix hash is stale by construction, and the corner fingerprint binds it, so
    comparing either against the retrain rejects recipes the six gates just passed. The same-retrain
    invariant that actually matters — pickle and model_stats agreeing — is enforced in
    :func:`_produced_artifacts_match`.
    """
    matrix = candidate["matrix_hash"] if matrix_hash is None else matrix_hash
    return ArtifactIdentity(
        strategy_slug=candidate["strategy_slug"],
        structural_strategy=candidate["structural_strategy"],
        signature=candidate["strategy_signature"],
        implementation_version=candidate["strategy_implementation_version"],
        artifact_schema_version=candidate["artifact_schema_version"],
        league=candidate["league"],
        market=candidate["market"],
        status=candidate["strategy_status"],
        controls_json=controls_json(candidate["controls"]),
        corner_fingerprint=(
            candidate["corner_fingerprint"]
            if matrix_hash is None
            else corner_fingerprint(
                get_strategy(candidate["strategy_slug"]), candidate["controls"], matrix
            )
        ),
        matrix_hash=matrix,
        split_fingerprint=candidate["split_fingerprint"],
    )


def _retrained_matrix_hash(league: str, market: str) -> str | None:
    """The matrix hash the just-finished retrain reported, or ``None`` when it wrote no row."""
    try:
        row = _cell_row(league, market, ["league", "market", "strategy_matrix_hash"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if row is None or not isinstance(row["strategy_matrix_hash"], str):
        return None
    return row["strategy_matrix_hash"]


def served_weight(path: pathlib.Path) -> float:
    """The blend weight in a model pickle: how much of the served predictive is model, not book."""
    return float(pd.read_pickle(path)["weight"])


def _produced_artifacts_match(
    league: str, market: str, candidate: dict, expected: ArtifactIdentity
) -> bool:
    """Fail closed unless the retrain produced the exact signed board strategy and cell.

    ``expected`` carries the matrix hash model_stats reported, so a pickle written against a
    different matrix — a stale artifact the retrain never replaced — fails here.
    """
    spec = get_strategy(candidate["strategy_slug"])
    csv_path = _TEST_SETS_ROOT / f"{market_file_slug(league, market)}.csv"
    try:
        frame = pd.read_csv(csv_path, keep_default_na=False)
        model = pd.read_pickle(model_pickle_path(league, market))
        actual = validate_strategy_artifacts(
            spec,
            candidate["controls"],
            frame,
            model,
            league=league,
            market=market,
            matrix_hash=expected.matrix_hash,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return actual == expected


def _gate_passed(value) -> bool:
    """NA-safe read of a nullable-boolean gate cell (``bool(pd.NA)`` raises)."""
    return pd.notna(value) and bool(value)


def _g4_only_retry_wanted(row: pd.Series, cell_hpo_selection: str, cell_zinb_mode: str) -> bool:
    """Whether a cell earns the one-shot calibrated-HPO retry.

    Fires only for cells trained under ``loss`` selection whose fresh
    gate row failed ship on a near-miss Gate 4, alone or with a
    within-band Gate 1 — the dispersion-calibration failures calibrated trial
    selection targets. Any other failing gate blocks the retry, g6 explicitly
    included (deferred pending the Experiment C measurement, brief R3).
    Hurdle-ZINB and Mixture are excluded: those paths have no calibrated
    trial-selection closure.
    """
    if cell_hpo_selection != "loss":
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


def _retry_calibrated_wanted(candidate: dict, row: dict[str, object], cell: dict) -> bool:
    """Whether a nominee's blind run earns the walk's calibrated-HPO fallback.

    ``row`` is that run's blind gate row and ``cell`` the nominee's stat_meta entry with its edits
    applied — the knobs a canonical refit of it would resolve. Structural corners never retry
    (their spec pins ``hpo_selection``, and the structural stage has no calibrated trial-selection
    closure).
    """
    if get_strategy(candidate["strategy_slug"]).is_structural:
        return False
    return _g4_only_retry_wanted(
        pd.Series(row), cell.get("hpo_selection", "loss"), cell.get("zinb_mode", "joint")
    )


class BlindRun(NamedTuple):
    """One nominee's full-HPO sandbox run, scored on the cross-fit validation half."""

    # The nominee as run: a calibrated retry carries its hpo_selection pin in its edits.
    candidate: dict
    dump: pathlib.Path
    # The sandbox pickle a canonical refit takes its parameters from (``--params-from``).
    model: pathlib.Path
    # Why the run fails on its own (meditate, identity, gates); empty on a blind 6/6.
    failed: list[str]
    # The ledger-named gate row; None when the run never scored.
    row: dict[str, object] | None


def _run_blind(candidate: dict, hpo_selection: str) -> BlindRun:
    """Train one nominee at full HPO in its own sandbox and score it on the cross-fit half.

    ``--holdout-blind`` drops the ship holdout from the run and cross-fits calibration over the
    validation frame, the half the board scores. ``--artifact-output`` writes the dump and pickle
    to a research sandbox keyed by cell, corner fingerprint and trial-selection rule, and the run
    skips ``report()``, so neither the canonical artifact set nor model_stats moves. Nothing is
    read from a persisted edit: every control rides as its flag, and so does ``hpo_selection``
    unless the corner pins it, so stat_meta.json is never written for a blind run. The namespace is
    the one a canonical run reads (:data:`DEPENDENCY_NAMESPACE`), and ``--bypass-withholding`` lets
    a withheld cell train. The board's own scorer (:func:`_score_corner`) reads the dump, and its
    row lands in the ledger as a cross-fit row.
    """
    lg, mkt = candidate["league"], candidate["market"]
    spec = get_strategy(candidate["strategy_slug"])
    slug = market_file_slug(lg, mkt)
    sandbox = (
        _CONFIRM_LOG_ROOT / "blind" / slug / f"{candidate['corner_fingerprint']}__{hpo_selection}"
    )
    # A dump left by an earlier walk carries this corner's identity, so a run that exits clean
    # without writing one would be scored on it.
    shutil.rmtree(sandbox, ignore_errors=True)
    selection = (
        [] if "hpo_selection" in candidate["controls"] else ["--hpo-selection", hpo_selection]
    )
    error = _run_meditate(
        candidate,
        [
            "--holdout-blind",
            "--artifact-output",
            str(sandbox),
            "--dependency-namespace",
            DEPENDENCY_NAMESPACE,
            "--bypass-withholding",
            *strategy_cli_args(sweep._cell_context(lg, mkt), spec, candidate["controls"]),
            *selection,
        ],
        log_path=sandbox / "meditate.log",
        label="blind full-HPO run",
        expected=confirm_median_seconds(lg, mkt),
    )
    run = BlindRun(
        candidate, sandbox / "test_sets" / f"{slug}.csv", sandbox / f"{slug}.mdl", [], None
    )
    if error:
        return run._replace(failed=[f"retrain {error}"])
    try:
        scored = _score_corner(lg, mkt, candidate["controls"], spec, run.dump, run.model)
    except (OSError, ValueError, KeyError, TypeError):
        return run._replace(failed=["artifact identity"])
    identity = _candidate_identity(candidate, scored["matrix_hash"])
    row = {
        "league": lg,
        "market": mkt,
        "distribution": candidate["family"],
        "hpo_selection": hpo_selection,
        **{column: getattr(identity, attr) for column, attr in _IDENTITY_COLUMNS.items()},
        **{
            _LEDGER_NAMES.get(name, name): value
            for name, value in scored.items()
            if name not in candidate["controls"]
        },
    }
    diverged = _record_ledger_row(candidate, row, EVAL_SPLIT_CROSSFIT)
    failed = _failed_gates(row)
    click.echo(
        f"  blind {' '.join(failed) or 'pass'}: slack {row['slack']:+.3f}, "
        f"w={row['model_weight']:.3f}"
    )
    if diverged and failed:
        # Named ahead of whatever it fails: the gates already reject a diverged fit, but
        # "diverged g1 g4" sends triage to the fit rather than the anonymous gate list.
        failed.insert(0, "diverged")
    return run._replace(failed=failed, row=row)


def measure_blind(meta: dict, candidate: dict) -> list[BlindRun]:
    """A nominee's blind runs: the first, then a calibrated retry when a g4-only miss earns one.

    The last run is the one that ranks. A live cell's incumbent keeps its first run as the
    baseline every contender meets, because that run measures the recipe the cell serves.
    """
    lg, mkt = candidate["league"], candidate["market"]
    cell = {**meta[lg][mkt], **candidate["edits"]}
    first = _run_blind(candidate, cell.get("hpo_selection", "loss"))
    if first.row is None or not _retry_calibrated_wanted(candidate, first.row, cell):
        return [first]
    click.echo(f"  {lg} {mkt}: g4-only blind fail — retrying with hpo_selection=calibrated")
    retry = {
        **candidate,
        "source": f"{candidate['source']} +calibrated-retry",
        "edits": {**candidate["edits"], "hpo_selection": "calibrated"},
    }
    return [first, _run_blind(retry, "calibrated")]


def run_canonical(league: str, market: str, candidate: dict) -> tuple[str, bool]:
    """Refit the walk's pick canonically and record the cell's one test-half ledger row.

    The pick's edits are already persisted, so the run reads its strategy from stat_meta as the
    weekly cron does, with only the unpersisted controls as flags. ``--params-from`` reuses the
    blind run's LightGBM params, cv and shape ceiling instead of searching, so the booster that
    ships is the one the walk ranked. Returns ``(error, diverged)``; a run that wrote no
    model_stats row records nothing, and the caller reports it.
    """
    spec = get_strategy(candidate["strategy_slug"])
    error = _run_meditate(
        candidate,
        [
            "--params-from",
            str(candidate["params_from"]),
            *strategy_full_hpo_cli_args(
                sweep._cell_context(league, market), spec, candidate["controls"]
            ),
        ],
        log_path=_CONFIRM_LOG_ROOT / f"{market_file_slug(league, market)}.log",
        label="canonical refit",
    )
    if error:
        return error, False
    row = _cell_row(league, market, None)
    if row is None:
        return "", False
    return "", _record_ledger_row(candidate, row.to_dict(), EVAL_SPLIT_TEST)


def refit_failures(league: str, market: str, candidate: dict) -> list[str]:
    """Why a canonical refit cannot ship, read back from model_stats and its pickle; empty: it can.

    Reasons rather than a bool because the legs fail for very different operator-facing causes, and
    a bare False reads on the report as a gate failure with no gate named.
    """
    matrix_hash = _retrained_matrix_hash(league, market)
    if matrix_hash is None:
        return ["no model_stats row"]
    # Gates first: ``ship`` is the six gates' conjunction, so a plain Gate-4 miss also fails the
    # model_stats leg, and checking identity ahead of the gates reports it with no gate named. The
    # two identity legs still gate the ship — they just answer a later question.
    failed = _failed_gates_after(league, market)
    if failed:
        return failed
    expected = _candidate_identity(candidate, matrix_hash)
    if not _produced_artifacts_match(league, market, candidate, expected):
        return ["artifact identity"]
    if not _ship_from_model_stats(league, market, expected):
        return ["model_stats identity/ship"]
    return []


def confirm_meditate(league: str, market: str, candidate: dict) -> list[str]:
    """Withheld-path confirm: refit canonically, then why the cell did not ship (empty: it did)."""
    error, diverged = run_canonical(league, market, candidate)
    if error:
        return [f"retrain {error}"]
    # A diverged fit is named ahead of whatever it fails: the gates already reject it, but
    # "diverged g1 g4" sends triage to the fit rather than the anonymous gate list.
    prefix = ["diverged"] if diverged else []
    failed = refit_failures(league, market, candidate)
    return prefix + failed if failed else []
