"""Operation Ship 75 strategy sweep: a per-cell Optuna study over a family's retrain grid.

For each ``(league, market)`` cell the sweep runs one Optuna study per family of the cell's
distribution class — a SkewNormal cell sweeps ``normalization × dist-loss × blend-loss``; a count
cell sweeps BOTH ``ZINB`` (``zinb-mode × count-dispersion-objective × blend-loss``) AND plain
``NegBin`` (``count-dispersion-objective × blend-loss``), cross-family ranked by ship slack —
training one deterministic ``meditate`` trial per grid corner and scoring it through the *honest*
production gate: the deterministic dump
already carries the pipeline's validation-fit calibration, and :func:`_score_corner` runs the same
:func:`scorecard.gate_row` the production scorecard does — no test re-fit. The sweep is a fixed-HP
replica of the production HPO pipeline: same calibration, same gate, same dump decode; the *only*
differences are fixed hyperparameters in place of the Optuna search and the deterministic sandbox
write locations (so a trial never clobbers a real trained market). The objective minimizes the
negative ship slack, so the study's best trial is the most-shippable corner.

Families live in :data:`_FAMILIES`, a small registry; :data:`_CLASS_FAMILIES` routes a cell's
``dist`` (via its distribution class) to the families to sweep. Each :class:`FamilySpec` names its
grid axes — a count family carries a single-choice ``dist`` axis so its winner persists ``dist`` and
each corner forces ``--dist`` regardless of the cell's current pin — the ``stat_meta.json`` fields a
winning corner persists, and the shipped defaults for any non-persistable axis. Adding a family
(e.g. Double Poisson) is one :class:`FamilySpec` plus its :data:`_DIST_CLASS` entry, not an engine
change. Every axis is categorical, so the sampler is :class:`optuna.samplers.GridSampler` —
exhaustive and deterministic, the right tool for a discrete space.

Research scaffolding: the deterministic trials *rank* only — nothing ships off them. The confirm
loop (``--confirm``, :mod:`sportstradamus.training.model_strategy_confirm`) persists a winner and a
clean full-HPO 5/5 on the official scorecard is what actually ships.
"""

import functools
import importlib.resources as pkg_resources
import math
import pathlib
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import click
import optuna
import pandas as pd
import tabulate

from sportstradamus import data as _data_pkg
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training import calibration
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.scorecard import (
    _DECODE_FALLBACK_STRATEGY,
    apply_thresholds,
    gate_row,
    load_test_set,
    min_gate_slack,
)
from sportstradamus.training.ship_config import STAT_META_PATH, WITHHELD, load_stat_meta

# The normalization corners the SkewNormal gate can decode (and therefore score). The EB slug
# `centered_additive_eb_meanyr_k10` decodes off the dumped `GlobalMean` column (the gate re-adds
# the empirical-Bayes prior) — see docs/ship_gate.md.
_DECODABLE_SN_NORMS: tuple[str, ...] = (
    "ratio_meanyr",
    "centered_additive_mean10",
    "centered_additive_eb_meanyr_k10",
)
_BLENDING: tuple[str, ...] = tuple(sorted(calibration.BLENDING_SLUGS))

# The SkewNormal family default training loss (the one that ships). A corner won under the other
# dist-loss can't be reproduced from stat_meta.json — dist-loss is a training-time knob that does
# not persist — so the actionable summary flags it and the confirm loop skips it.
_SN_DEFAULT_DIST_LOSS = "crps"

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

# Each swept axis' meditate CLI flag. A corner is realized by appending `--flag value` per axis;
# an axis a family doesn't sweep is simply absent (e.g. ZINB never forces --target-normalization,
# so meditate resolves it to the ratio_meanyr fallback — see _dump_subdir).
_AXIS_FLAG: dict[str, str] = {
    "dist": "--dist",
    "normalization": "--target-normalization",
    "dist_training_loss": "--dist-training-loss",
    "blending_loss_fn": "--blending-loss-fn",
    "zinb_mode": "--zinb-mode",
    "count_dispersion_objective": "--count-dispersion-objective",
}


@dataclass(frozen=True)
class FamilySpec:
    """One distribution family's sweep definition.

    Attributes:
        axes: Grid axes ``{axis: choices}`` the GridSampler enumerates (all categorical today).
        persist: ``{axis: stat_meta.json field}`` a winning corner writes to ship the cell.
        defaults: ``{axis: shipped value}`` for each swept axis that does *not* persist — the value
            production actually ships. A corner is confirmable only when its non-persistable axes
            sit at these defaults (the confirm loop's reproducibility check).
    """

    axes: dict[str, tuple[str, ...]]
    persist: dict[str, str]
    defaults: dict[str, str]


_FAMILIES: dict[str, FamilySpec] = {
    "SkewNormal": FamilySpec(
        axes={
            "normalization": _DECODABLE_SN_NORMS,
            "dist_training_loss": ("crps", "nll"),
            "blending_loss_fn": _BLENDING,
        },
        persist={"normalization": "target_normalization", "blending_loss_fn": "blending"},
        defaults={"dist_training_loss": _SN_DEFAULT_DIST_LOSS},
    ),
    "ZINB": FamilySpec(
        axes={
            "dist": ("ZINB",),
            "zinb_mode": ("joint", "hurdle"),
            "count_dispersion_objective": ("crps", "pit_ks"),
            "blending_loss_fn": _BLENDING,
        },
        persist={
            "dist": "dist",
            "zinb_mode": "zinb_mode",
            "count_dispersion_objective": "count_dispersion_objective",
            "blending_loss_fn": "blending",
        },
        defaults={},
    ),
    "NegBin": FamilySpec(
        axes={
            "dist": ("NegBin",),
            "count_dispersion_objective": ("crps", "pit_ks"),
            "blending_loss_fn": _BLENDING,
        },
        persist={
            "dist": "dist",
            "count_dispersion_objective": "count_dispersion_objective",
            "blending_loss_fn": "blending",
        },
        defaults={},
    ),
    "DPO": FamilySpec(
        axes={
            "dist": ("DPO",),
            "count_dispersion_objective": ("crps", "pit_ks"),
            "blending_loss_fn": _BLENDING,
        },
        persist={
            "dist": "dist",
            "count_dispersion_objective": "count_dispersion_objective",
            "blending_loss_fn": "blending",
        },
        defaults={},
    ),
}

# A cell's stat_meta `dist` names its distribution class; the sweep sweeps every family in that
# class (cross-family ranked by slack). A count cell sweeps ZINB, plain NegBin *and* DPO regardless
# of which one it is currently pinned to, so a re-sweep after any family flip still evaluates all.
# Adding a family is one FamilySpec above plus its class entry here.
_DIST_CLASS: dict[str, str] = {
    "SkewNormal": "continuous",
    "ZINB": "count",
    "NegBin": "count",
    "DPO": "count",
}
_CLASS_FAMILIES: dict[str, tuple[str, ...]] = {
    "continuous": ("SkewNormal",),
    "count": ("ZINB", "NegBin", "DPO"),
}
# `--dist-class all` sweeps every class; the two real classes are _CLASS_FAMILIES' keys.
_DIST_CLASS_ALL = "all"

# One wide board schema across both families: a cell fills only its family's axis columns, the rest
# are blank. Kept as a fixed superset so the board CSV has a stable header regardless of which
# families were swept.
_AXIS_COLUMNS: list[str] = [
    "dist",
    "normalization",
    "dist_training_loss",
    "zinb_mode",
    "count_dispersion_objective",
    "blending_loss_fn",
]
_BOARD_COLUMNS: list[str] = [
    "league",
    "market",
    "family",
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
    "swept_at",
    "code_rev",
]

# Default output path for the living board — both CLI modes write here unless --out overrides.
STRATEGY_RESEARCH_BOARD: pathlib.Path = pathlib.Path(
    str(pkg_resources.files(_data_pkg) / "research" / "strategy_research_board.csv")
)


def _cell_families(league: str, market: str) -> tuple[str, ...]:
    """The sweep families for a cell's distribution class, from its stat_meta ``dist``.

    A count cell sweeps ``("ZINB", "NegBin", "DPO")`` — even one already pinned to a single family —
    so a re-sweep after any family flip re-evaluates all; a SkewNormal cell sweeps
    ``("SkewNormal",)``. Loud on a dist with no registered family/class.
    """
    meta = load_stat_meta(pathlib.Path(str(STAT_META_PATH)))
    dist = meta.get(league, {}).get(market, {}).get("dist")
    if dist not in _DIST_CLASS:
        raise click.UsageError(
            f"{league} {market}: dist {dist!r} is not a swept family; known: {sorted(_DIST_CLASS)}"
        )
    return _CLASS_FAMILIES[_DIST_CLASS[dist]]


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


def _dump_subdir(corner: dict[str, str]) -> str:
    """The deterministic dump subdir meditate keys a corner by: ``{target_normalization}{_hurdle}``.

    SkewNormal corners force a ``normalization`` slug; ZINB corners force none, so meditate resolves
    the count-branch target-normalization to the ``ratio_meanyr`` fallback (cli.meditate, via
    :func:`ship_config.resolve_flag_target_normalization`), and a ``hurdle`` mode appends ``_hurdle``
    (pipeline.py). This mirrors that formula exactly so scoring reads the file the trial wrote.
    """
    norm = corner.get("normalization", _DECODE_FALLBACK_STRATEGY)
    suffix = "_hurdle" if corner.get("zinb_mode") == "hurdle" else ""
    return f"{norm}{suffix}"


def _decode_strategy(corner: dict[str, str]) -> str:
    """The gate's decode strategy for a corner: the swept SN normalization, or ratio_meanyr for a count cell."""
    return corner.get("normalization", _DECODE_FALLBACK_STRATEGY)


def _dump_paths(
    league: str, market: str, corner: dict[str, str]
) -> tuple[pathlib.Path, pathlib.Path]:
    """CSV under ``test_sets/deterministic/<subdir>/`` and pickle under ``research/models/deterministic/<subdir>/``."""
    subdir = _dump_subdir(corner)
    filename = market_file_slug(league, market)
    csv = _TEST_SETS_ROOT / "deterministic" / subdir / f"{filename}.csv"
    mdl = _DETERMINISTIC_MODEL_ROOT / subdir / f"{filename}.mdl"
    return csv, mdl


def _corner_label(corner: dict[str, str]) -> str:
    """Human-scannable ``axis=value · axis=value`` for the trial's progress + log lines."""
    return " · ".join(f"{axis}={value}" for axis, value in corner.items())


def _log_path(league: str, market: str, corner: dict[str, str]) -> pathlib.Path:
    """Per-corner meditate log under ``research/logs/deterministic/<subdir>/``.

    Keyed by the full corner so two corners sharing a dump subdir (e.g. the loss axes of one
    normalization) don't overwrite each other's log — unlike the model/CSV dump, which meditate keys
    by subdir alone and each corner therefore retrains and scores before the next overwrites it.
    """
    filename = market_file_slug(league, market)
    tag = "_".join(f"{axis}={corner[axis]}" for axis in sorted(corner))
    return _DETERMINISTIC_LOG_ROOT / _dump_subdir(corner) / f"{filename}__{tag}.log"


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


def _run_deterministic_meditate(league: str, market: str, corner: dict[str, str]) -> None:
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
    cmd = [
        "poetry",
        "run",
        "meditate",
        "--league",
        league,
        "--market",
        market,
        "--deterministic",
        "--bypass-withholding",
    ]
    for axis, value in corner.items():
        cmd += [_AXIS_FLAG[axis], value]
    _run_meditate_with_lock_retry(
        cmd, _log_path(league, market, corner), timeout=_MEDITATE_TRIAL_TIMEOUT_S
    )


def _score_corner(league: str, market: str, corner: dict[str, str]) -> dict[str, object]:
    """Score a trained corner's dump through the honest production gate.

    The dump's distribution is inferred from its columns (SkewNormal vs the count R/NB_P/Gate
    triple), so one gate path scores both families; ``decode_strategy`` is the swept SN
    normalization or the ratio_meanyr count fallback. ``dispersion_cal`` / ``skew_cal`` are read
    from the pickle for context only (a count cell dumps ``skew_cal`` absent → 0.0).
    """
    csv_path, mdl_path = _dump_paths(league, market, corner)
    df = load_test_set(csv_path, _SHIP_PRED_COL)
    filedict = pd.read_pickle(mdl_path)
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
    league: str, market: str, family: str, corner: dict[str, str]
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
    label = _corner_label(corner)
    click.echo(f"  training  {label} …")
    start = time.monotonic()
    try:
        _run_deterministic_meditate(league, market, corner)
        row = _score_corner(league, market, corner)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        click.secho(
            f"  FAILED    {label} — {type(exc).__name__}; recorded non-shipping, continuing",
            fg="yellow",
            err=True,
        )
        return [{**corner, "family": family, "slack": _FAILED_CORNER_SLACK, "ships": False}]
    row["family"] = family
    verdict = click.style(_verdict(row), fg="green" if row["ships"] else "red")
    elapsed = f"{time.monotonic() - start:.0f}s"
    click.echo(f"  {verdict}  {label}  slack {float(row['slack']):+.3f}  ({elapsed})")
    return [row]


def _run_family_study(league: str, market: str, family: str) -> list[dict[str, object]]:
    """One GridSampler study over ``family``'s grid; the scored row of every corner it visits.

    ``objective`` binds ``family`` as a default arg rather than a free variable — belt-and-suspenders
    against the classic loop-closure-capture bug even though it can't fire here: ``family`` is this
    function's parameter, fixed for the whole call, and :func:`search_cell` calls this function once
    per family rather than defining ``objective`` itself inside its own family loop.
    """
    grid = {axis: list(choices) for axis, choices in _FAMILIES[family].axes.items()}
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(grid))

    def objective(trial: optuna.Trial, family: str = family) -> float:
        corner = {axis: trial.suggest_categorical(axis, choices) for axis, choices in grid.items()}
        rows = _run_and_score(league, market, family, corner)
        trial.set_user_attr("rows", rows)
        return -max(float(row["slack"]) for row in rows)

    study.optimize(objective, n_trials=math.prod(len(choices) for choices in grid.values()))
    return [row for trial in study.trials for row in trial.user_attrs["rows"]]


def search_cell(league: str, market: str) -> pd.DataFrame:
    """Run one Optuna GridSampler study per family of the cell's distribution class, ranked by slack.

    A count cell studies both ZINB and plain NegBin and the boards union; a SkewNormal cell studies
    one family. One honest row per retrain corner across all families; the board carries each corner's
    slack / ship verdict / gate passes so the top row is the real-HPO confirm candidate. Sorted by
    ``slack`` descending.
    """
    rows = [
        {"league": league, "market": market, **row}
        for family in _cell_families(league, market)
        for row in _run_family_study(league, market, family)
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
    return (_TRAINING_DATA_ROOT / f"{market_file_slug(league, market)}.parquet").is_file()


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
    return [
        (lg, mkt)
        for lg, markets in meta.items()
        if league is None or lg == league
        for mkt, cell in markets.items()
        if cell.get("dist") in _FAMILIES
        and (include_shipped or cell.get("shipped") == WITHHELD)
        and (dist_class == _DIST_CLASS_ALL or _DIST_CLASS[cell["dist"]] == dist_class)
    ]


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
    in_registry = [
        (lg, mkt)
        for lg, mkt in _candidate_cells(league, include_shipped, dist_class)
        if mkt in ALL_MARKETS.get(lg, [])
    ]
    sweepable = [c for c in in_registry if _has_training_data(*c)]
    missing = [c for c in in_registry if not _has_training_data(*c)]
    return sweepable, missing


def _cell_corner_count(league: str, market: str) -> int:
    """The deterministic trainings one cell contributes — its family grids summed (count cells: ZINB + NegBin)."""
    return sum(
        math.prod(len(c) for c in _FAMILIES[f].axes.values())
        for f in _cell_families(league, market)
    )


def _corner_count(cells: list[tuple[str, str]]) -> int:
    """Total deterministic trainings a board run will do — each cell's family grids summed."""
    return sum(_cell_corner_count(lg, mkt) for lg, mkt in cells)


def _board_done_cells(out: str | None) -> set[tuple[str, str]]:
    """The ``(league, market)`` cells already on the board CSV — what ``--resume`` skips."""
    if out is None or not pathlib.Path(out).exists():
        return set()
    prior = pd.read_csv(out)
    return set(map(tuple, prior[["league", "market"]].drop_duplicates().to_numpy()))


def run_board(
    cells: list[tuple[str, str]], out: str | None = None, resume: bool = False
) -> pd.DataFrame:
    """Search every cell in ``cells``, printing each verdict as it lands and upserting the board CSV
    per cell so an interrupt keeps partial progress and a ``--league``-scoped run leaves other
    leagues' rows intact. With ``resume``, cells already on the CSV are skipped and their rows carry
    through, so a crashed multi-hour run picks up where it stopped. Returns the full board.
    """
    boards: list[pd.DataFrame] = []
    done = _board_done_cells(out) if resume else set()
    if resume and out is not None and pathlib.Path(out).exists():
        boards.append(pd.read_csv(out))
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
        prior = pd.read_csv(path)
        league, market = cell_board["league"].iloc[0], cell_board["market"].iloc[0]
        keep = prior[~((prior["league"] == league) & (prior["market"] == market))]
        pd.concat([keep, cell_board], ignore_index=True).to_csv(path, index=False)
    else:
        cell_board.to_csv(path, index=False)
    return cell_board


def _stat_meta_edit(row: object) -> str:
    """The exact stat_meta.json fields to persist a winning corner, resolved for its family.

    Reads the family's ``persist`` map so the operator doesn't have to translate board columns to
    field names: SkewNormal → ``target_normalization=…, blending=…``; a count family →
    ``dist=…, [zinb_mode=…,] count_dispersion_objective=…, blending=…`` (the persisted ``dist`` pins
    the winning family, e.g. flipping a cell ZINB→NegBin). A non-persistable axis (SN's dist-loss) is
    intentionally omitted — the shipped model uses the family default.
    """
    persist = _FAMILIES[row["family"]].persist
    return ", ".join(f"{field}={row[axis]}" for axis, field in persist.items())


def _repro_note(row: object) -> str:
    """Warn when the winning corner used a non-default axis it can't carry into stat_meta.

    Only SkewNormal's dist-loss is non-persistable today; a count corner's value is NaN (float), so
    the ``isinstance`` str guard keeps this quiet for families whose every axis persists.
    """
    dist_loss = row.get("dist_training_loss")
    if isinstance(dist_loss, str) and dist_loss != _SN_DEFAULT_DIST_LOSS:
        return (
            f"won under dist-loss={dist_loss}, which is not saved — "
            f"confirm under the default {_SN_DEFAULT_DIST_LOSS}"
        )
    return ""


def _print_cell_summary(board: pd.DataFrame) -> None:
    """One cell's verdict + the exact stat_meta.json edit to ship it, over a narrow per-corner table.

    The board CSV keeps all columns; the screen shows the verdict, what to change in stat_meta.json
    (for a shipping cell), and each corner's family axes / slack / blocking gate.
    """
    best = board.iloc[0]
    ships = bool(best["ships"])
    axes = list(_FAMILIES[best["family"]].axes)
    click.echo(
        click.style(f"\n{best['league']} {best['market']} — ", bold=True)
        + click.style(_verdict(best), fg="green" if ships else "red", bold=True)
        + click.style(f" (best slack {float(best['slack']):+.3f})", bold=True)
    )
    if ships:
        note = _repro_note(best)
        click.echo(
            f"  → stat_meta.json: {_stat_meta_edit(best)}" + (f"   ⚠ {note}" if note else "")
        )
    table = [
        [
            *(r[axis] for axis in axes),
            f"{float(r['slack']):+.3f}",
            "yes" if r["ships"] else "no",
            " ".join(_failed_gates(r)) or "-",
        ]
        for _, r in board.iterrows()
    ]
    click.echo(
        tabulate.tabulate(
            table, headers=[*axes, "slack", "ships", "failed gates"], tablefmt="github"
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
                _repro_note(best),
            ]
        )
    click.echo(
        tabulate.tabulate(
            rows,
            headers=["cell", "family", "set in stat_meta.json", "slack", "note"],
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
    help="Sweep every withheld cell with cached training data (both families) instead of one cell; "
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
    type=click.Choice([*_CLASS_FAMILIES, _DIST_CLASS_ALL]),
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
    """Operation Ship 75 strategy sweep — a per-cell GridSampler over the cell's family grid, one
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
