"""Operation Ship 75 confirm-and-ship loop: persist a swept winner, retrain at full HPO, keep or revert.

The strategy sweep (:mod:`sportstradamus.training.model_strategy.sweep`) *ranks* corners on fixed-HP
deterministic trials — it never ships. This module turns a ranked board into shipped cells:

1. For each cell, pick the top-slack shipping corner **across the cell's swept families** — a
   count cell's slice carries both ZINB and NegBin corners, ranked together by ship slack. Every
   swept axis has a ``stat_meta.json`` field, and the winner's own family's ``persist`` map builds
   the edits (so a NegBin corner outranking ZINB persists ``dist=NegBin`` and flips the cell's
   family).
2. Prompt, then per candidate: write its persist fields + ``shipped="devel"`` to ``stat_meta.json``
   (so the confirm ``meditate`` reads the exact config being shipped), run a **full-HPO** ``meditate``,
   and read the official ``ship`` verdict from ``model_stats.parquet``.
3. A pass keeps the cell on devel; a failure **auto-reverts** — restore the original stat_meta entry
   *and* :func:`prune_model_pickle`. The pickle prune is mandatory: inference loads pickles by path
   and never consults ``shipped``, so reverting stat_meta alone would leave a failed cell serving.

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
from sportstradamus.training.model_strategy.identity import (
    ArtifactIdentity,
    build_artifact_identity,
    validate_strategy_artifacts,
)
from sportstradamus.training.model_strategy.registry import (
    BASE_STRUCTURAL_STRATEGY,
    CAP_CONFIRM,
    CAP_FULL_HPO,
    CAP_SCORE,
    CAP_SERVE,
    controls_json,
    get_strategy,
    meditate_command,
    parse_controls,
    strategies_for_cell,
    strategy_controls,
    strategy_full_hpo_cli_args,
    strategy_persistence_edits,
)
from sportstradamus.training.model_strategy.sweep import (
    _GATES,
    _SHIP_PRED_COL,
    _TEST_SETS_ROOT,
    _cell_context,
    _run_meditate_with_lock_retry,
)
from sportstradamus.training.scorecard import _supersede_headline, load_test_set, supersede_verdict
from sportstradamus.training.ship_config import STAT_META_PATH, WITHHELD, load_stat_meta

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_STAT_META = pathlib.Path(str(STAT_META_PATH))
_CONFIRM_LOG_ROOT = _REPO_ROOT / "research" / "logs" / "confirm"
_SHIPPED_DEVEL = "devel"
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


def _candidate(sub: pd.DataFrame) -> dict | None:
    """Return the top signed, reproducible, confirm-capable shipping corner for one cell."""
    shipping = sub[sub["ships"].eq(True)].sort_values("slack", ascending=False)
    if shipping.empty:
        return None
    lg, mkt = sub["league"].iloc[0], sub["market"].iloc[0]
    context = _cell_context(lg, mkt)
    for _, row in shipping.iterrows():
        cand = _board_candidate_row(row, context, lg, mkt)
        if cand is not None:
            return cand
    return None


def _board_candidate_row(row: pd.Series, context, league: str, market: str) -> dict | None:
    """Validate one shipping board row's full identity contract; ``None`` if not confirm-capable.

    Raises on any stale/mismatched identity; returns the confirm-candidate dict for a clean,
    confirm-capable corner, or ``None`` when the corner's family lacks the confirm capabilities so
    the caller can fall through to the next-best shipping row.
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
    if not spec.capabilities >= _CONFIRM_CAPABILITIES:
        return None
    slack = float(row["slack"])
    if not math.isfinite(slack):
        raise ValueError(f"{league} {market}: candidate slack must be finite")
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


def _candidates(board: pd.DataFrame) -> list[dict]:
    """One confirm candidate per cell that shipped at least one corner."""
    out = []
    for _cell, sub in board.groupby(["league", "market"], sort=False):
        cand = _candidate(sub)
        if cand is not None:
            out.append(cand)
    return out


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
    never read at serve time, and strategy-independent — a candidate run reproduces them identically,
    so a HOLD that leaves them in candidate state changes nothing served and the next ``meditate``
    rebuilds them.
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


def _cell_row(league: str, market: str, columns: list[str]) -> pd.Series | None:
    """One cell's row of the requested model_stats.parquet columns, or ``None`` when absent."""
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
        "strategy_corner_fingerprint": expected.corner_fingerprint,
        "strategy_matrix_hash": expected.matrix_hash,
    }
    if any(row[column] != value for column, value in checks.items()):
        return False
    return _split_matches(row["strategy_split_fingerprint"], expected.split_fingerprint)


def _failed_gates_after(league: str, market: str) -> list[str]:
    """The gates a just-confirmed cell fails, read from model_stats — diagnostics for the report."""
    row = _cell_row(league, market, ["league", "market", *(f"{g}_pass" for g in _GATES)])
    if row is None:
        return ["(no model_stats row)"]
    return [g for g in _GATES if not bool(row[f"{g}_pass"])]


def _run_meditate(league: str, market: str, candidate: dict) -> bool:
    """Full-HPO retrain of a cell from its just-persisted stat_meta strategy; True iff meditate exits clean.

    ``--force`` is required or a cell with no new gamedays skips silently and never rewrites its
    outputs. A non-zero exit or a timeout returns False (don't trust a possibly-stale model_stats
    row); a transient archive-lock clash is retried first (:func:`_run_meditate_with_lock_retry`).
    """
    spec = get_strategy(candidate["strategy_slug"])
    context = _cell_context(league, market)
    cmd = meditate_command(
        league,
        market,
        "--force",
        *strategy_full_hpo_cli_args(context, spec, candidate["controls"]),
    )
    log_path = _CONFIRM_LOG_ROOT / f"{market_file_slug(league, market)}.log"
    click.echo(f"  retraining {league} {market} (full HPO, ~1h) …")
    try:
        _run_meditate_with_lock_retry(cmd, log_path, timeout=_CONFIRM_TIMEOUT_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _candidate_identity(candidate: dict) -> ArtifactIdentity:
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
        corner_fingerprint=candidate["corner_fingerprint"],
        matrix_hash=candidate["matrix_hash"],
        split_fingerprint=candidate["split_fingerprint"],
    )


def _produced_artifacts_match(league: str, market: str, candidate: dict) -> bool:
    """Fail closed unless the retrain produced the exact signed board strategy and cell."""
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
            matrix_hash=candidate["matrix_hash"],
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return actual == _candidate_identity(candidate)


def _confirm_meditate(league: str, market: str, candidate: dict) -> bool:
    """Withheld-path confirm: retrain, then True iff the official scorecard ships the cell."""
    expected = _candidate_identity(candidate)
    return (
        _run_meditate(league, market, candidate)
        and _produced_artifacts_match(league, market, candidate)
        and _ship_from_model_stats(league, market, expected)
    )


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
        if _confirm_meditate(lg, mkt, cand):
            _sync_cell_from_disk(meta, lg, mkt)
            keep = True
            click.secho(f"  SHIPPED (devel) {lg} {mkt}", fg="green")
            return (lg, mkt, "SHIPPED", [])
        failed = _failed_gates_after(lg, mkt)
        click.secho(
            f"  REVERTED {lg} {mkt} — failed {' '.join(failed) or '(retrain error)'}",
            fg="red",
        )
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
        if not _run_meditate(lg, mkt, cand):
            return (lg, mkt, "HELD", ["retrain error"])
        if not _produced_artifacts_match(lg, mkt, cand):
            return (lg, mkt, "HELD", ["artifact identity"])
        if not _ship_from_model_stats(lg, mkt, _candidate_identity(cand)):
            return (lg, mkt, "HELD", ["model_stats identity/ship"])
        baseline = load_test_set(backup / f"{slug}.csv", _SHIP_PRED_COL)
        candidate = load_test_set(_TEST_SETS_ROOT / f"{slug}.csv", _SHIP_PRED_COL)
        verdict = supersede_verdict(baseline, candidate, _SHIP_PRED_COL, league=lg, market=mkt)
        click.echo("  " + _supersede_headline(verdict))
        if not verdict["ship"]:
            return (lg, mkt, "HELD", _failed_legs(verdict))
        edits = ", ".join(f"{k}={v}" for k, v in cand["edits"].items())
        if not click.confirm(f"  Promote {lg} {mkt} to {edits}?"):
            return (lg, mkt, "HELD", ["declined"])
        keep = True
        _sync_cell_from_disk(meta, lg, mkt)
        return (lg, mkt, "SUPERSEDED", [])
    finally:
        if not keep:
            _restore_cell(lg, mkt, backup, meta, original)


def _split_shippable(ready: list[dict], meta: dict) -> tuple[list[dict], list[dict]]:
    """Partition candidates by current release surface: withheld (fresh) vs already-shipped (live).

    Withheld cells auto-ship on a clean 6/6 (a fresh cell has no live pickle, so a failed confirm's
    revert+prune restores its dark state). Live cells route to the supersession test, which restores
    the incumbent byte-identical on a loss rather than pruning it.
    """
    fresh, shipped = [], []
    for c in ready:
        target = fresh if meta[c["league"]][c["market"]].get("shipped") == WITHHELD else shipped
        target.append(c)
    return fresh, shipped


def _drop_activation_gated(fresh: list[dict]) -> list[dict]:
    """Announce and drop withheld candidates in activation-gated leagues — they never auto-ship."""
    kept = []
    for c in fresh:
        if c["league"] in _ACTIVATION_GATED_LEAGUES:
            click.secho(
                f"  ACTIVATION-GATED {c['league']} {c['market']} — withheld {c['league']} cells ship "
                "only after the D1/D2 owner gate; skipping.",
                fg="yellow",
            )
        else:
            kept.append(c)
    return kept


def _announce_plan(fresh: list[dict], shipped: list[dict]) -> None:
    """List the run's plan: which withheld cells get persisted+confirmed, which live cells get tested."""
    for label, group, verb in (
        ("withheld candidate", fresh, "persist + confirm"),
        ("live cell", shipped, "supersession-test (S1/S2/S3)"),
    ):
        if group:
            click.secho(f"\n{len(group)} {label}(s) to {verb}:", bold=True)
            for c in group:
                click.echo(
                    f"  {c['league']} {c['market']}: "
                    + ", ".join(f"{k}={v}" for k, v in c["edits"].items())
                )


def run_confirm(board: pd.DataFrame, *, yes: bool = False) -> None:
    """Confirm the sweep's winners: auto-ship withheld cells on a clean 6/6, supersession-test live cells.

    ``board`` is the in-memory sweep result (a row per corner, ``ships`` bool). Withheld candidates are
    persisted + retrained and kept on a clean 6/6 (else reverted+pruned). Already-shipped cells (present
    when the sweep ran ``--include-shipped``) run the S1/S2/S3 test and swap the live cell only on a
    passing verdict AND an operator yes; any loss restores the incumbent. ``yes`` skips only the upfront
    gate — live-cell promotions always prompt individually.
    """
    ready = _candidates(board)
    if not ready:
        click.echo("no shipping candidates to confirm.")
        return

    meta = load_stat_meta(_STAT_META)
    fresh, shipped = _split_shippable(ready, meta)
    fresh = _drop_activation_gated(fresh)
    if not fresh and not shipped:
        click.echo("no confirmable candidates after the activation gate.")
        return
    _announce_plan(fresh, shipped)
    prompt = (
        f"\nPersist {len(fresh)} withheld config(s) and supersession-test {len(shipped)} live cell(s) "
        "with a full-HPO retrain (~1h each)? Withheld failures auto-revert; live promotions prompt"
    )
    if not yes and not click.confirm(prompt):
        click.echo("aborted; stat_meta.json unchanged.")
        return

    backup = _backup_stat_meta()
    click.echo(f"stat_meta.json backed up to {backup}")
    results = [_confirm_one(meta, c) for c in fresh] + [_supersede_one(meta, c) for c in shipped]
    _print_confirm_report(results, backup)


def _print_confirm_report(
    results: list[tuple[str, str, str, list[str]]], backup: pathlib.Path
) -> None:
    """Final table (cell → outcome → failing gates) + the tally, backup path, and commit reminder."""
    rows = [
        [f"{lg} {mkt}", outcome, " ".join(failed) or "-"] for lg, mkt, outcome, failed in results
    ]
    click.secho("\nconfirm results", bold=True)
    click.echo(
        tabulate.tabulate(rows, headers=["cell", "outcome", "failed gates"], tablefmt="github")
    )
    n_win = sum(1 for r in results if r[2] in _WIN_OUTCOMES)
    click.secho(
        f"\n{n_win} shipped/superseded (devel), {len(results) - n_win} reverted/held. Backup: {backup}",
        fg="green" if n_win else "yellow",
    )
    if n_win:
        click.echo("Review the stat_meta.json diff and commit the shipped/superseded cells.")
