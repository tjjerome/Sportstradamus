"""Operation Ship 75 confirm-and-ship loop: persist a swept winner, retrain at full HPO, keep or revert.

The strategy sweep (:mod:`sportstradamus.training.model_strategy.sweep`) *ranks* corners on fixed-HP
deterministic trials — it never ships. This module turns a ranked board into shipped cells:

1. For each cell, nominate the top :data:`CONFIRM_TOP_K` corners **across the cell's swept
   families** by ship slack — a count cell's slice carries ZINB, NegBin and DPO corners ranked
   together — then its seed corners and its incumbent. Every swept axis has a ``stat_meta.json``
   field, and each nominee's own family's ``persist`` map builds the edits (so a NegBin corner
   outranking ZINB persists ``dist=NegBin`` and flips the cell's family).
2. Prompt, then per nominee in order: write its persist fields + ``shipped="devel"`` to
   ``stat_meta.json`` (so the confirm ``meditate`` reads the exact config being shipped), run a
   **full-HPO** ``meditate`` on the walk's pinned training matrix, and read the official ``ship``
   verdict from ``model_stats.parquet``.
3. A pass keeps the cell on devel and ends its walk; a failure **auto-reverts** — restore the
   original stat_meta entry *and* :func:`prune_model_pickle` — and the next nominee is tried. The
   pickle prune is mandatory: inference loads pickles by path and never consults ``shipped``, so
   reverting stat_meta alone would leave a failed cell serving.

Everything is local and uncommitted — the human reviews the ``shipped: devel`` diff and commits it.
A whole-file ``stat_meta.json`` backup is taken before any write as the crash/abort safety net.
"""

import json
import math
import pathlib
import shutil
import subprocess
import time
from copy import deepcopy

import click
import pandas as pd
import tabulate

from sportstradamus.helpers.io import (
    MODEL_STATS_PATH,
    market_file_slug,
    model_pickle_path,
    prune_model_pickle,
)
from sportstradamus.training.lineage import (
    MATRIX_BUILDER_VERSION,
    MATRIX_SCHEMA_VERSION,
    file_sha,
)
from sportstradamus.training.model_strategy.identity import (
    ArtifactIdentity,
    build_artifact_identity,
    corner_fingerprint,
    validate_strategy_artifacts,
)
from sportstradamus.training.model_strategy.registry import (
    BASE_STRUCTURAL_STRATEGY,
    CAP_CONFIRM,
    CAP_FULL_HPO,
    CAP_SCORE,
    CAP_SERVE,
    controls_json,
    distribution_class,
    get_strategy,
    meditate_command,
    parse_controls,
    strategies_for_cell,
    strategy_controls,
    strategy_full_hpo_cli_args,
    strategy_persistence_edits,
)
from sportstradamus.training.model_strategy.sweep import (
    _DIVERGED_DISPERSION_CAL,
    _GATES,
    _SHIP_PRED_COL,
    _TEST_SETS_ROOT,
    EVAL_SPLIT_CROSSFIT,
    NOMINEE_LEDGER_PATH,
    _cell_context,
    _failure_reason,
    _known_good_corners,
    _run_meditate_with_lock_retry,
    _training_matrix_path,
)
from sportstradamus.training.scorecard import _supersede_headline, load_test_set, supersede_verdict
from sportstradamus.training.ship_config import STAT_META_PATH, WITHHELD, load_stat_meta

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_STAT_META = pathlib.Path(str(STAT_META_PATH))
_CONFIRM_LOG_ROOT = _REPO_ROOT / "research" / "logs" / "confirm"
_SHIPPED_DEVEL = "devel"
# Board-row gate values echoed onto each board nominee (as ``board_*`` ledger columns), so a
# board↔confirm comparison reads one ledger row instead of joining two CSVs. Seed/incumbent
# nominees have no board row and leave them NaN.
_BOARD_ECHO_FIELDS: tuple[str, ...] = (
    "slack",
    "g1_brier_diff_ci_hi",
    "g2_star_z",
    "g3_bench_z",
    "g4_pit_ks",
    "g4_pit_ks_max",
    "g5_ece_debiased",
    "n",
    "eval_split",
    "swept_at",
)
# Leagues whose withheld cells confirm may never auto-flip: a board-passing cell is announced and
# skipped until the owner's activation gates (D1/D2) go GO, then the league is removed here in the
# same PR that ships its first cells. Empty since the MLB+NHL GO (2026-07-09,
# docs/handoffs/mlb-nhl-activation.md); machinery stays for the next league onboarding.
_ACTIVATION_GATED_LEAGUES: tuple[str, ...] = ()
# A full-HPO meditate confirm is ~1 h; a large cell can run longer, so a 4 h ceiling keeps a hung
# run from blocking the loop forever. A timeout is treated as a failure (the cell auto-reverts).
_CONFIRM_TIMEOUT_S = 4 * 3600
# Confirm outcomes that leave a shippable cell on devel (vs REVERTED / HELD, which change nothing).
_WIN_OUTCOMES: tuple[str, ...] = ("SHIPPED", "SUPERSEDED")
_CONFIRM_CAPABILITIES = frozenset({CAP_CONFIRM, CAP_FULL_HPO, CAP_SCORE, CAP_SERVE})
# Board rows a cell nominates for full-HPO confirmation. Deterministic fixed-HP scoring does not
# ship the recipes that only pass under real HPO, so nominating the top few — rather than requiring
# a deterministic `ships` — is what makes confirmation reachable at all for the popular NFL passing
# markets, whose boards carry zero shipping rows.
CONFIRM_TOP_K: int = 3
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


def _nominees(sub: pd.DataFrame) -> list[dict]:
    """Ordered confirm nominees for one cell: top board corners, then seeds, then the incumbent.

    ``ships`` is deliberately not read. Fixed-HP deterministic scoring will not ship a recipe that
    only passes under full HPO, so requiring it left the popular NFL passing cells with no candidate
    at all; the six gates still decide, just after the retrain rather than before it.

    Ordering is the argument each source can make. Board rows first — the search's own ranked
    recommendation on honest out-of-fold data, by ``discounted_slack`` when the sweep prices it
    (falling back to raw ``slack`` on older boards) — with the best count-class corner interleaved
    second on an integer-target cell whose top corners are all continuous-family
    (:func:`_count_class_backup`). Seeds next: independent full-HPO held-out evidence the
    deterministic ranking demonstrably cannot see. The incumbent last, so a cell is never downgraded
    by an unlucky list. Rows scored against the ship holdout (legacy ``eval_split``) never nominate.
    """
    lg, mkt = sub["league"].iloc[0], sub["market"].iloc[0]
    context = _cell_context(lg, mkt)
    admissible = sub[sub["eval_split"].astype("string").eq(EVAL_SPLIT_CROSSFIT)]
    rank_column = "discounted_slack" if "discounted_slack" in admissible.columns else "slack"
    ranked = admissible.sort_values(rank_column, ascending=False)
    nominated: list[dict] = []
    for _, row in ranked.iterrows():
        if len(nominated) >= CONFIRM_TOP_K:
            break
        cand = _board_candidate_row(row, context, lg, mkt)
        if cand is not None:
            nominated.append({**cand, "source": f"board slack {cand['slack']:+.3f}"})
    backup = _count_class_backup(ranked, nominated, context, lg, mkt)
    if backup is not None:
        nominated.insert(1, backup)
    for spec, controls in _known_good_corners(context):
        cand = _corner_candidate(context, spec, lg, mkt, controls, slack=math.nan, split=None)
        if cand is not None:
            nominated.append({**cand, "source": "seed/incumbent"})
    seen: set[str] = set()
    return [
        cand
        for cand in nominated
        if not (cand["corner_fingerprint"] in seen or seen.add(cand["corner_fingerprint"]))
    ]


def _count_class_backup(
    ranked: pd.DataFrame, nominated: list[dict], context, league: str, market: str
) -> dict | None:
    """The best count-class board corner to slot second on an integer-target cell, or ``None``.

    An integer-target cell's top corners are often all SkewNormal — the family whose full-HPO fit
    can diverge — while stable count-family corners rank just below; whole walks burned on that
    pattern. Interleaving the best count corner at slot 2 costs nothing when the leader confirms
    and saves the walk when it diverges. No-op when the cell's target is not on the integer
    lattice, nothing was nominated, a count corner already sits in the top slots, or the board has
    no admissible count-class row. Insert-only: no continuous nominee is dropped.
    """
    if not context.target_is_integer or not nominated:
        return None
    if any(distribution_class(cand["family"]) == "count" for cand in nominated):
        return None
    for _, row in ranked.iterrows():
        if distribution_class(str(row["family"])) != "count":
            continue
        cand = _board_candidate_row(row, context, league, market)
        if cand is not None:
            return {**cand, "source": f"board slack {cand['slack']:+.3f}"}
    return None


def _corner_candidate(
    context, spec, league: str, market: str, controls: dict[str, str], *, slack: float, split
) -> dict | None:
    """The confirm-candidate dict for one registered corner, or ``None`` if it cannot confirm.

    A family without the confirm capabilities (Mixture ranks but never serves) drops out here, as
    does a corner whose spec is not applicable to the cell — a seed or incumbent can outlive the
    admission gates that once let it in.
    """
    if spec not in strategies_for_cell(context) or not spec.capabilities >= _CONFIRM_CAPABILITIES:
        return None
    identity = build_artifact_identity(
        spec.slug, league, market, controls, matrix_hash=context.matrix_sha256
    )
    return {
        "league": league,
        "market": market,
        "family": spec.family,
        "strategy_slug": spec.slug,
        "structural_strategy": identity.structural_strategy,
        "strategy_signature": identity.signature,
        "strategy_implementation_version": identity.implementation_version,
        "artifact_schema_version": identity.artifact_schema_version,
        "strategy_status": identity.status,
        "matrix_hash": identity.matrix_hash,
        "split_fingerprint": split,
        "controls": controls,
        "corner_fingerprint": identity.corner_fingerprint,
        "edits": strategy_persistence_edits(context, spec, controls),
        "slack": slack,
    }


def _board_candidate_row(row: pd.Series, context, league: str, market: str) -> dict | None:
    """Validate one board row's full identity contract; ``None`` if not confirm-capable.

    Raises on any stale/mismatched identity; returns the confirm-candidate dict for a clean,
    confirm-capable corner, or ``None`` when the corner's family lacks the confirm capabilities so
    the caller can fall through to the next-best row.
    """
    family = _required_text(row, "family", league, market)
    strategy_slug = _required_text(row, "strategy_slug", league, market)
    structural = _required_text(row, "structural_strategy", league, market)
    spec = get_strategy(strategy_slug)
    expected_structural = spec.slug if spec.is_structural else BASE_STRUCTURAL_STRATEGY
    if spec.family != family or structural != expected_structural:
        raise ValueError(f"{league} {market}: strategy/family identity mismatch")
    if spec not in strategies_for_cell(context):
        raise ValueError(f"{league} {market}: strategy {spec.slug!r} is not enrolled/applicable")
    controls = parse_controls(row.get("controls_json"))
    if controls_json(controls) != row["controls_json"] or controls not in strategy_controls(spec):
        raise ValueError(f"{league} {market}: stale or noncanonical strategy controls")
    _validate_control_columns(row, controls, league, market)
    identity = build_artifact_identity(
        spec.slug, league, market, controls, matrix_hash=context.matrix_sha256
    )
    split = _validate_board_identity(row, spec, identity, league, market)
    slack = float(row["slack"])
    if not math.isfinite(slack):
        raise ValueError(f"{league} {market}: candidate slack must be finite")
    cand = _corner_candidate(context, spec, league, market, controls, slack=slack, split=split)
    if cand is not None:
        cand["board_gates"] = {f"board_{field}": row.get(field) for field in _BOARD_ECHO_FIELDS}
    return cand


def _required_text(row: pd.Series, field: str, league: str, market: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value or pd.isna(value):
        raise ValueError(f"{league} {market}: candidate has missing {field}")
    return value


def _validate_control_columns(
    row: pd.Series, controls: dict[str, str], league: str, market: str
) -> None:
    for name, expected in controls.items():
        if name not in row.index:
            continue
        actual = row.get(name)
        if pd.isna(actual) or str(actual) != expected:
            raise ValueError(f"{league} {market}: control column {name} contradicts controls_json")


def _int_matches(actual: object, expected: int) -> bool:
    """Whether a board cell's integer field equals ``expected`` (NaN / non-numeric → False)."""
    try:
        return not pd.isna(actual) and float(actual) == expected
    except (TypeError, ValueError):
        return False


def _validate_split_contract(spec, split: object, league: str, market: str) -> None:
    if spec.split_fingerprint_path and (not isinstance(split, str) or not split):
        raise ValueError(f"{league} {market}: missing strategy split fingerprint")
    if not spec.split_fingerprint_path and split is not None:
        raise ValueError(f"{league} {market}: unexpected strategy split fingerprint")


def _validate_board_identity(
    row: pd.Series,
    spec,
    identity: ArtifactIdentity,
    league: str,
    market: str,
) -> str | None:
    core_matches = (
        _required_text(row, "strategy_slug", league, market) == identity.strategy_slug
        and _required_text(row, "strategy_signature", league, market) == identity.signature
        and _required_text(row, "strategy_status", league, market) == "active"
        and _required_text(row, "matrix_hash", league, market) == identity.matrix_hash
        and _int_matches(
            row.get("strategy_implementation_version"), identity.implementation_version
        )
        and _int_matches(row.get("artifact_schema_version"), identity.artifact_schema_version)
    )
    if not core_matches:
        raise ValueError(f"{league} {market}: stale or mismatched strategy identity")
    fingerprint = _required_text(row, "corner_fingerprint", league, market)
    if fingerprint != identity.corner_fingerprint:
        raise ValueError(f"{league} {market}: stale strategy corner fingerprint")
    split = row.get("split_fingerprint")
    split = None if pd.isna(split) else split
    _validate_split_contract(spec, split, league, market)
    return split


def _candidates(board: pd.DataFrame, max_nominees: int | None = None) -> list[list[dict]]:
    """Each cell's ordered nominee list; cells with nothing confirmable drop out.

    ``max_nominees`` truncates each list after dedup, trading the tail of a cell's walk for wall
    clock. The ordering :func:`_nominees` establishes is what makes that trade sound — the corners
    most likely to ship come first.
    """
    return [
        nominated[:max_nominees]
        for _cell, sub in board.groupby(["league", "market"], sort=False)
        if (nominated := _nominees(sub))
    ]


def _atomic_write_meta(meta: dict) -> None:
    """Write stat_meta.json atomically at its native 4-space indent so the review diff stays minimal."""
    tmp = _STAT_META.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(meta, fh, indent=4)
        fh.write("\n")
    tmp.replace(_STAT_META)


def _sync_cell_from_disk(meta: dict, league: str, market: str) -> None:
    """Re-read one cell's stat_meta entry after a meditate subprocess shipped it.

    The g4-only calibrated retry inside meditate persists ``hpo_selection`` to disk
    from the subprocess; without this re-sync the walk's next whole-file
    ``_atomic_write_meta`` (another cell's persist or revert) would silently erase
    that pin from confirm's stale in-memory copy.
    """
    with _STAT_META.open() as fh:
        meta[league][market] = json.load(fh)[league][market]


def _backup_stat_meta() -> pathlib.Path:
    """Copy the whole stat_meta.json to a timestamped sibling — the crash/abort recovery point."""
    backup = _STAT_META.with_name(f"stat_meta.{time.strftime('%Y%m%dT%H%M%S')}.bak.json")
    shutil.copy2(_STAT_META, backup)
    return backup


def _cell_artifacts(league: str, market: str) -> list[pathlib.Path]:
    """Every serve-read artifact a full-HPO ``meditate`` rewrites for one cell — the restore set.

    Restoring these puts the incumbent back exactly as it served: the model pickle (which carries all
    calibrators), the test-set CSV (also the snapshotted S2/S3 baseline), ``model_stats`` (parquet +
    csv mirror), the two SHAP CSVs, and the two config files read at serve time — ``stat_calibration``
    and ``book_weights`` (shared, its per-cell key reverted by the whole-file restore). The shared
    files are safe to whole-file restore because a cell's snapshot→restore window is isolated (only
    that cell's ``meditate`` runs between them, so the revert touches only that cell's row/key).

    Deliberately excluded: the per-cell training-matrix cache (``training_data/{slug}.parquet``) and
    the per-league caches (gamelog, ``comps.json``, correlation matrices). Those are training inputs,
    never read at serve time — and a confirm retrain trains from the walk's frozen snapshot
    (``--frozen-matrix-dir``) without touching any of them.
    """
    slug = market_file_slug(league, market)
    training_dir = MODEL_STATS_PATH.parent
    config_dir = _STAT_META.parent
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


def _snapshot_cell(league: str, market: str) -> pathlib.Path:
    """Copy the incumbent's artifacts to a per-cell backup dir and return it (the S2/S3 baseline lives there)."""
    backup = _CONFIRM_LOG_ROOT / "incumbent_backup" / market_file_slug(league, market)
    shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir(parents=True, exist_ok=True)
    for art in _cell_artifacts(league, market):
        if art.exists():
            shutil.copy2(art, backup / art.name)
    return backup


def _restore_cell(
    league: str, market: str, backup: pathlib.Path, meta: dict, original: dict
) -> None:
    """Restore every snapshotted artifact byte-identical and put the original stat_meta entry back.

    Copies files from ``backup`` over the canonical paths — it never prunes — so a live cell that loses
    the supersession test keeps serving exactly what it served before.
    """
    for art in _cell_artifacts(league, market):
        saved = backup / art.name
        if saved.exists():
            shutil.copy2(saved, art)
        elif art.exists():
            art.unlink()
    meta[league][market] = original
    _atomic_write_meta(meta)


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


def _record_nominee_gates(league: str, market: str, candidate: dict) -> bool:
    """Record the just-retrained nominee's whole model_stats row in the research ledger.

    This is the only moment that row exists. A losing nominee gets its pickle pruned, and
    ``report()`` rebuilds model_stats from the pickles on disk, so by the time the walk reports
    its verdict the gate values behind it are gone — leaving no way to compare a cell's board
    numbers against its own confirm. A board nominee's row also carries its board-side gate values
    (``board_*``, from ``_board_candidate_row``); seed/incumbent rows leave them NaN. Ledger only;
    nothing production reads it.

    Returns whether the fit diverged — ``dispersion_cal`` on its floor
    (:data:`_DIVERGED_DISPERSION_CAL`) — which is also recorded as the ``diverged`` column.
    """
    row = _cell_row(league, market, None)
    if row is None:
        return False
    dispersion = row.get("dispersion_cal")
    diverged = bool(pd.notna(dispersion) and float(dispersion) <= _DIVERGED_DISPERSION_CAL)
    ledger = pd.DataFrame(
        [
            {
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "strategy_slug": candidate["strategy_slug"],
                "source": candidate["source"],
                "diverged": diverged,
                **candidate.get("board_gates", {}),
                **row.to_dict(),
            }
        ]
    )
    if NOMINEE_LEDGER_PATH.exists():
        # Rewrite rather than append: model_stats widens as gates are added, and appending a
        # wider row under a header written by a narrower one yields a CSV no reader can parse.
        ledger = pd.concat([pd.read_csv(NOMINEE_LEDGER_PATH), ledger], ignore_index=True)
    NOMINEE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(NOMINEE_LEDGER_PATH, index=False)
    return diverged


def _failed_gates_after(league: str, market: str) -> list[str]:
    """The gates a just-confirmed cell fails, read from model_stats — diagnostics for the report."""
    row = _cell_row(league, market, ["league", "market", *(f"{g}_pass" for g in _GATES)])
    if row is None:
        return ["(no model_stats row)"]
    return [g for g in _GATES if not bool(row[f"{g}_pass"])]


def _frozen_matrix_dir(league: str, market: str) -> pathlib.Path:
    """The per-cell dir a walk pins its training matrix into (shared by pin and retrain)."""
    return _CONFIRM_LOG_ROOT / "frozen_matrix" / market_file_slug(league, market)


def _pin_cell_matrix(league: str, market: str) -> pathlib.Path:
    """Copy the cell's cached training matrix + lineage manifest into the walk's frozen dir.

    Every nominee of the walk then retrains via ``--frozen-matrix-dir`` on this one frame — an
    unpinned ``--force`` rewrote the matrix between nominees, so they scored different frames
    (n_validation drifted 373/363/358 within one walk). Two side effects of meditate's frozen-input
    mode are deliberate: the run cold-starts its HPO (warm pickle params are ignored, so every
    nominee gets the same full search budget) and it skips the per-market book-weight refits (the
    same isolation as the board's deterministic runs, keeping board and confirm aligned). A swept
    cell always has a cached parquet — the sweep requires one — so a missing source fails loud
    here. The snapshot stays on disk after the walk (research/ is gitignored), matching the
    ``incumbent_backup`` pattern.
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


def _run_meditate(league: str, market: str, candidate: dict) -> str:
    """Full-HPO retrain of a cell from its just-persisted stat_meta strategy; "" iff meditate exits clean.

    Trains on the walk's pinned matrix (``--frozen-matrix-dir``, see :func:`_pin_cell_matrix`) so
    every nominee of a cell scores the same frame. ``--force`` is still required: a frozen-input
    run appends no new rows, and without it a cell whose pickle already exists skips silently and
    never rewrites its outputs. A failure returns its :func:`_failure_reason` rather than a bare
    flag — a native abort inside the fit and an ordinary non-zero exit call for completely
    different triage, and reporting both as one "retrain error" is what sent the last investigation
    looking for a Python exception that never existed. The caller must not trust a possibly-stale
    model_stats row either way; a transient archive-lock clash is retried first
    (:func:`_run_meditate_with_lock_retry`).
    """
    spec = get_strategy(candidate["strategy_slug"])
    context = _cell_context(league, market)
    cmd = meditate_command(
        league,
        market,
        "--force",
        "--frozen-matrix-dir",
        str(_frozen_matrix_dir(league, market)),
        *strategy_full_hpo_cli_args(context, spec, candidate["controls"]),
    )
    log_path = _CONFIRM_LOG_ROOT / f"{market_file_slug(league, market)}.log"
    click.echo(f"  retraining {league} {market} (full HPO, ~1h) …")
    try:
        _run_meditate_with_lock_retry(cmd, log_path, timeout=_CONFIRM_TIMEOUT_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return _failure_reason(exc)
    return ""


def _candidate_identity(candidate: dict, matrix_hash: str | None = None) -> ArtifactIdentity:
    """The nominated recipe as an identity, rebound to ``matrix_hash`` when a retrain moved the matrix.

    ``_run_meditate`` passes ``--force``, which updates the league gamelogs and appends gamedays, so a
    confirm retrain routinely lands on a different training matrix than the search scored — every cell
    of the first overnight run did. What confirmation tests is the *recipe*; the board's pre-retrain
    matrix hash is stale by construction, and the corner fingerprint binds it, so comparing either
    against the retrain rejects recipes the six gates just passed. The same-retrain invariant that
    actually matters — pickle and model_stats agreeing — is enforced in :func:`_produced_artifacts_match`.
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


def _produced_artifacts_match(
    league: str, market: str, candidate: dict, expected: ArtifactIdentity
) -> bool:
    """Fail closed unless the retrain produced the exact signed board strategy and cell.

    ``expected`` carries the matrix hash model_stats reported, so a pickle written against a different
    matrix — a stale artifact the retrain never replaced — fails here.
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


def _confirm_meditate(league: str, market: str, candidate: dict) -> list[str]:
    """Withheld-path confirm: retrain, then the reasons the cell did not ship (empty means it did).

    Reasons rather than a bool because the legs fail for very different operator-facing causes, and a
    bare False reads on the report as a gate failure with no gate named.
    """
    error = _run_meditate(league, market, candidate)
    if error:
        return [f"retrain {error}"]
    # A diverged fit is named ahead of whatever it fails: the gates already reject it, but
    # "diverged g1 g4" sends triage to the fit rather than the anonymous gate list.
    diverged = _record_nominee_gates(league, market, candidate)
    prefix = ["diverged"] if diverged else []
    matrix_hash = _retrained_matrix_hash(league, market)
    if matrix_hash is None:
        return [*prefix, "no model_stats row"]
    # Gates first: serve-iff-ship leaves no pickle behind for a cell that failed one, so checking
    # artifact identity ahead of them reports a missing serving artifact as the cause of a plain
    # Gate-4 miss. The two identity legs still gate the ship — they just answer a later question.
    failed = _failed_gates_after(league, market)
    if failed:
        return prefix + failed
    expected = _candidate_identity(candidate, matrix_hash)
    if not _produced_artifacts_match(league, market, candidate, expected):
        return [*prefix, "artifact identity"]
    if not _ship_from_model_stats(league, market, expected):
        return [*prefix, "model_stats identity/ship"]
    return []


def _confirm_one(meta: dict, cand: dict) -> tuple[str, str, str, list[str]]:
    """Persist one candidate, confirm at full HPO, and keep it (devel) or revert (stat_meta + pickle).

    The pickle prune on failure is what actually dark-outs the cell — inference loads pickles by
    path and ignores ``shipped``, so a reverted stat_meta entry alone would still serve.
    """
    lg, mkt = cand["league"], cand["market"]
    original = deepcopy(meta[lg][mkt])
    backup = _snapshot_cell(lg, mkt)
    keep = False
    try:
        meta[lg][mkt].update(cand["edits"])
        meta[lg][mkt]["shipped"] = _SHIPPED_DEVEL
        _atomic_write_meta(meta)
        failed = _confirm_meditate(lg, mkt, cand)
        if not failed:
            _sync_cell_from_disk(meta, lg, mkt)
            keep = True
            click.secho(f"  SHIPPED (devel) {lg} {mkt}", fg="green")
            return (lg, mkt, "SHIPPED", [])
        click.secho(f"  REVERTED {lg} {mkt} — failed {' '.join(failed)}", fg="red")
        return (lg, mkt, "REVERTED", failed)
    finally:
        if not keep:
            _restore_cell(lg, mkt, backup, meta, original)
            prune_model_pickle(lg, mkt)


def _failed_legs(verdict: dict) -> list[str]:
    """The supersession legs a HOLD verdict failed (for the report's failed-gates column)."""
    return [leg for leg in ("S1", "S2", "S3") if not verdict[f"{leg.lower()}_pass"]]


def _supersede_one(meta: dict, cand: dict) -> tuple[str, str, str, list[str]]:
    """Supersession-test one live cell: snapshot, retrain the candidate in place, require its exact
    artifact identity and official six-gate ``model_stats`` ship, then run S1/S2/S3 and promote it
    (on a passing verdict + operator yes) or restore the incumbent byte-identical.

    A live-cell swap needs the test to pass AND an explicit promote confirmation; every other exit —
    HOLD, decline, retrain error, or an exception — hits the ``finally`` restore, which copies the
    snapshot back and never prunes, so the incumbent keeps serving.
    """
    lg, mkt = cand["league"], cand["market"]
    slug = market_file_slug(lg, mkt)
    original = deepcopy(meta[lg][mkt])
    backup = _snapshot_cell(lg, mkt)
    keep = False
    try:
        meta[lg][mkt].update(cand["edits"])  # shipped left as-is; the cell stays live
        _atomic_write_meta(meta)
        error = _run_meditate(lg, mkt, cand)
        if error:
            return (lg, mkt, "HELD", [f"retrain {error}"])
        diverged = _record_nominee_gates(lg, mkt, cand)
        prefix = ["diverged"] if diverged else []
        matrix_hash = _retrained_matrix_hash(lg, mkt)
        if matrix_hash is None:
            return (lg, mkt, "HELD", [*prefix, "no model_stats row"])
        expected = _candidate_identity(cand, matrix_hash)
        if not _produced_artifacts_match(lg, mkt, cand, expected):
            return (lg, mkt, "HELD", [*prefix, "artifact identity"])
        if not _ship_from_model_stats(lg, mkt, expected):
            return (lg, mkt, "HELD", [*prefix, "model_stats identity/ship"])
        baseline = load_test_set(backup / f"{slug}.csv", _SHIP_PRED_COL)
        candidate = load_test_set(_TEST_SETS_ROOT / f"{slug}.csv", _SHIP_PRED_COL)
        verdict = supersede_verdict(baseline, candidate, _SHIP_PRED_COL, league=lg, market=mkt)
        click.echo("  " + _supersede_headline(verdict))
        if not verdict["ship"]:
            return (lg, mkt, "HELD", prefix + _failed_legs(verdict))
        edits = ", ".join(f"{k}={v}" for k, v in cand["edits"].items())
        if not click.confirm(f"  Promote {lg} {mkt} to {edits}?"):
            return (lg, mkt, "HELD", ["declined"])
        keep = True
        _sync_cell_from_disk(meta, lg, mkt)
        return (lg, mkt, "SUPERSEDED", [])
    finally:
        if not keep:
            _restore_cell(lg, mkt, backup, meta, original)


def _split_shippable(
    ready: list[list[dict]], meta: dict
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Partition *cells* by current release surface: withheld (fresh) vs already-shipped (live).

    Withheld cells auto-ship on a clean 6/6 (a fresh cell has no live pickle, so a failed confirm's
    revert+prune restores its dark state). Live cells route to the supersession test, which restores
    the incumbent byte-identical on a loss rather than pruning it.
    """
    fresh, shipped = [], []
    for nominated in ready:
        lg, mkt = nominated[0]["league"], nominated[0]["market"]
        target = fresh if meta[lg][mkt].get("shipped") == WITHHELD else shipped
        target.append(nominated)
    return fresh, shipped


def _drop_activation_gated(fresh: list[list[dict]]) -> list[list[dict]]:
    """Announce and drop withheld cells in activation-gated leagues — they never auto-ship."""
    kept = []
    for nominated in fresh:
        lg, mkt = nominated[0]["league"], nominated[0]["market"]
        if lg in _ACTIVATION_GATED_LEAGUES:
            click.secho(
                f"  ACTIVATION-GATED {lg} {mkt} — withheld {lg} cells ship "
                "only after the D1/D2 owner gate; skipping.",
                fg="yellow",
            )
        else:
            kept.append(nominated)
    return kept


def _announce_plan(fresh: list[list[dict]], shipped: list[list[dict]]) -> None:
    """List the run's plan: each cell's nominees in the order they will be tried, with their source."""
    for label, group, verb in (
        ("withheld cell", fresh, "persist + confirm"),
        ("live cell", shipped, "supersession-test (S1/S2/S3)"),
    ):
        if group:
            click.secho(f"\n{len(group)} {label}(s) to {verb}:", bold=True)
            for nominated in group:
                head = nominated[0]
                click.echo(f"  {head['league']} {head['market']} — {len(nominated)} nominee(s):")
                for n, cand in enumerate(nominated, start=1):
                    edits = ", ".join(f"{k}={v}" for k, v in cand["edits"].items())
                    click.echo(f"    {n}. [{cand['source']}] {edits}")


def _walk_nominees(
    meta: dict, nominated: list[dict], attempt
) -> tuple[str, str, str, list[str], str]:
    """Retrain each nominee in order until one wins; report the deciding attempt and how deep it went.

    The cell's training matrix is pinned once here (:func:`_pin_cell_matrix`), so every nominee
    retrains and scores on the same frame. No new revert machinery is needed: ``_confirm_one``'s
    ``finally`` already restores stat_meta and prunes the pickle on every non-win, and
    ``_supersede_one``'s restores byte-identically. The loop just stops at the first win.
    """
    outcome = ("", "", "REVERTED", ["no nominee"], "-")
    _pin_cell_matrix(nominated[0]["league"], nominated[0]["market"])
    for n, cand in enumerate(nominated, start=1):
        click.secho(
            f"\n  nominee {n}/{len(nominated)} [{cand['source']}] {cand['league']} {cand['market']}",
            bold=True,
        )
        lg, mkt, verdict, failed = attempt(meta, cand)
        outcome = (lg, mkt, verdict, failed, f"{n}/{len(nominated)} {cand['source']}")
        if verdict in _WIN_OUTCOMES:
            break
    return outcome


def run_confirm(board: pd.DataFrame, *, yes: bool = False, max_nominees: int | None = None) -> None:
    """Confirm the sweep's winners: auto-ship withheld cells on a clean 6/6, supersession-test live cells.

    ``board`` is the in-memory sweep result (a row per corner). Each cell nominates its top corners
    plus any seed and its incumbent; the nominees are retrained in order until one ships. Withheld
    cells are persisted + retrained and kept on a clean 6/6 (else reverted+pruned). Already-shipped
    cells (present when the sweep ran ``--include-shipped``) run the S1/S2/S3 test and swap the live
    cell only on a passing verdict AND an operator yes; any loss restores the incumbent. ``yes``
    skips only the upfront gate — live-cell promotions always prompt individually. ``max_nominees``
    caps each cell's walk at its first N nominees.
    """
    ready = _candidates(board, max_nominees)
    if not ready:
        click.echo("no confirmable nominees on the board.")
        return

    meta = load_stat_meta(_STAT_META)
    fresh, shipped = _split_shippable(ready, meta)
    fresh = _drop_activation_gated(fresh)
    if not fresh and not shipped:
        click.echo("no confirmable candidates after the activation gate.")
        return
    _announce_plan(fresh, shipped)
    retrains = sum(len(n) for n in fresh) + sum(len(n) for n in shipped)
    prompt = (
        f"\nConfirm {len(fresh)} withheld and supersession-test {len(shipped)} live cell(s) — up to "
        f"{retrains} full-HPO retrains (~1h each), stopping at each cell's first shipping nominee? "
        "Withheld failures auto-revert; live promotions prompt"
    )
    if not yes and not click.confirm(prompt):
        click.echo("aborted; stat_meta.json unchanged.")
        return

    backup = _backup_stat_meta()
    click.echo(f"stat_meta.json backed up to {backup}")
    results = [_walk_nominees(meta, n, _confirm_one) for n in fresh]
    results += [_walk_nominees(meta, n, _supersede_one) for n in shipped]
    _print_confirm_report(results, backup)


def _print_confirm_report(
    results: list[tuple[str, str, str, list[str], str]], backup: pathlib.Path
) -> None:
    """Final table (cell → nominee → outcome → failing gates) + tally, backup path, commit reminder."""
    rows = [
        [f"{lg} {mkt}", nominee, outcome, " ".join(failed) or "-"]
        for lg, mkt, outcome, failed, nominee in results
    ]
    click.secho("\nconfirm results", bold=True)
    click.echo(
        tabulate.tabulate(
            rows, headers=["cell", "nominee", "outcome", "failed gates"], tablefmt="github"
        )
    )
    n_win = sum(1 for r in results if r[2] in _WIN_OUTCOMES)
    click.secho(
        f"\n{n_win} shipped/superseded (devel), {len(results) - n_win} reverted/held. Backup: {backup}",
        fg="green" if n_win else "yellow",
    )
    if n_win:
        click.echo("Review the stat_meta.json diff and commit the shipped/superseded cells.")
