"""Operation Ship 75 confirm-and-ship loop: persist a swept winner, retrain at full HPO, keep or revert.

The strategy sweep (:mod:`sportstradamus.training.model_strategy_sweep`) *ranks* corners on fixed-HP
deterministic trials — it never ships. This module turns a ranked board into shipped cells:

1. For each cell, pick the best-by-slack shipping corner whose config is **fully persistable** — every
   swept axis it wins on has a ``stat_meta.json`` field. SkewNormal's ``dist_training_loss`` has no
   field (the family default ships), so a corner that only wins under the non-default dist-loss is
   reported ``RANKS-ONLY`` and skipped — never persist a config the confirm can't reproduce.
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
from sportstradamus.training.model_strategy_sweep import (
    _FAMILIES,
    _GATES,
    _SHIP_PRED_COL,
    _TEST_SETS_ROOT,
    FamilySpec,
    _run_meditate_with_lock_retry,
)
from sportstradamus.training.scorecard import _supersede_headline, load_test_set, supersede_verdict
from sportstradamus.training.ship_config import STAT_META_PATH, WITHHELD, load_stat_meta

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_STAT_META = pathlib.Path(str(STAT_META_PATH))
_CONFIRM_LOG_ROOT = _REPO_ROOT / "research" / "logs" / "confirm"
_SHIPPED_DEVEL = "devel"
# MLB/NHL ship only after their owner activation gates (D1/D2 — docs/handoffs/mlb-nhl-activation.md).
# A board-passing withheld cell in a gated league is announced and skipped, never auto-flipped; on a
# GO the owner removes the league here in the same PR that ships its first cells. Already-live cells
# in these leagues still supersession-test (a strategy swap never changes the release surface).
_ACTIVATION_GATED_LEAGUES: tuple[str, ...] = ("MLB", "NHL")
# A full-HPO meditate confirm is ~1 h; a large cell can run longer, so a 4 h ceiling keeps a hung
# run from blocking the loop forever. A timeout is treated as a failure (the cell auto-reverts).
_CONFIRM_TIMEOUT_S = 4 * 3600
# Confirm outcomes that leave a shippable cell on devel (vs REVERTED / HELD, which change nothing).
_WIN_OUTCOMES: tuple[str, ...] = ("SHIPPED", "SUPERSEDED")


def _is_persistable(row: pd.Series, spec: FamilySpec) -> bool:
    """True iff every non-persistable axis of ``row`` sits at its shipped default.

    ``spec.defaults`` is empty for a family whose every axis persists (ZINB), so all its corners are
    persistable; SkewNormal's only non-persistable axis is ``dist_training_loss`` (default ``crps``).
    """
    return all(row[axis] == default for axis, default in spec.defaults.items())


def _candidate(sub: pd.DataFrame) -> dict | None:
    """The confirm candidate for one cell's board slice: a persistable shipping corner, or a marker.

    Returns ``None`` when the cell shipped no corner (nothing to confirm), a ``ranks_only`` marker
    when it shipped only non-persistable corners, else a ``candidate`` with the stat_meta edits.
    """
    lg, mkt = sub["league"].iloc[0], sub["market"].iloc[0]
    family = sub["family"].iloc[0]
    spec = _FAMILIES[family]
    shipping = sub[sub["ships"].astype(bool)]
    if shipping.empty:
        return None
    persistable = shipping[shipping.apply(lambda r: _is_persistable(r, spec), axis=1)]
    if persistable.empty:
        return {"league": lg, "market": mkt, "family": family, "status": "ranks_only"}
    best = persistable.sort_values("slack", ascending=False).iloc[0]
    edits = {spec.persist[axis]: str(best[axis]) for axis in spec.persist}
    return {
        "league": lg,
        "market": mkt,
        "family": family,
        "status": "candidate",
        "edits": edits,
        "slack": float(best["slack"]),
    }


def _candidates(board: pd.DataFrame) -> list[dict]:
    """One confirm candidate (or ranks-only marker) per cell that shipped at least one corner."""
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
    meta[league][market] = original
    _atomic_write_meta(meta)


def _ship_from_model_stats(league: str, market: str) -> bool:
    """The official ``ship`` verdict report() wrote for a cell; False if the row is absent."""
    stats = pd.read_parquet(MODEL_STATS_PATH, columns=["league", "market", "ship"])
    hit = stats[(stats["league"] == league) & (stats["market"] == market)]
    return bool(hit["ship"].iloc[0]) if not hit.empty else False


def _failed_gates_after(league: str, market: str) -> list[str]:
    """The gates a just-confirmed cell fails, read from model_stats — diagnostics for the report."""
    cols = ["league", "market", *(f"{g}_pass" for g in _GATES)]
    stats = pd.read_parquet(MODEL_STATS_PATH, columns=cols)
    hit = stats[(stats["league"] == league) & (stats["market"] == market)]
    if hit.empty:
        return ["(no model_stats row)"]
    r = hit.iloc[0]
    return [g for g in _GATES if not bool(r[f"{g}_pass"])]


def _run_meditate(league: str, market: str) -> bool:
    """Full-HPO retrain of a cell from its just-persisted stat_meta strategy; True iff meditate exits clean.

    ``--force`` is required or a cell with no new gamedays skips silently and never rewrites its
    outputs. A non-zero exit or a timeout returns False (don't trust a possibly-stale model_stats
    row); a transient archive-lock clash is retried first (:func:`_run_meditate_with_lock_retry`).
    """
    cmd = ["poetry", "run", "meditate", "--league", league, "--market", market, "--force"]
    log_path = _CONFIRM_LOG_ROOT / f"{market_file_slug(league, market)}.log"
    click.echo(f"  retraining {league} {market} (full HPO, ~1h) …")
    try:
        _run_meditate_with_lock_retry(cmd, log_path, timeout=_CONFIRM_TIMEOUT_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _confirm_meditate(league: str, market: str) -> bool:
    """Withheld-path confirm: retrain, then True iff the official scorecard ships the cell."""
    return _run_meditate(league, market) and _ship_from_model_stats(league, market)


def _confirm_one(meta: dict, cand: dict) -> tuple[str, str, str, list[str]]:
    """Persist one candidate, confirm at full HPO, and keep it (devel) or revert (stat_meta + pickle).

    The pickle prune on failure is what actually dark-outs the cell — inference loads pickles by
    path and ignores ``shipped``, so a reverted stat_meta entry alone would still serve.
    """
    lg, mkt = cand["league"], cand["market"]
    original = deepcopy(meta[lg][mkt])
    meta[lg][mkt].update(cand["edits"])
    meta[lg][mkt]["shipped"] = _SHIPPED_DEVEL
    _atomic_write_meta(meta)
    if _confirm_meditate(lg, mkt):
        click.secho(f"  SHIPPED (devel) {lg} {mkt}", fg="green")
        return (lg, mkt, "SHIPPED", [])
    meta[lg][mkt] = original
    _atomic_write_meta(meta)
    prune_model_pickle(lg, mkt)
    failed = _failed_gates_after(lg, mkt)
    click.secho(f"  REVERTED {lg} {mkt} — failed {' '.join(failed) or '(retrain error)'}", fg="red")
    return (lg, mkt, "REVERTED", failed)


def _failed_legs(verdict: dict) -> list[str]:
    """The supersession legs a HOLD verdict failed (for the report's failed-gates column)."""
    return [leg for leg in ("S1", "S2", "S3") if not verdict[f"{leg.lower()}_pass"]]


def _supersede_one(meta: dict, cand: dict) -> tuple[str, str, str, list[str]]:
    """Supersession-test one live cell: snapshot, retrain the candidate in place, run S1/S2/S3, and
    promote it (on a passing verdict + operator yes) or restore the incumbent byte-identical.

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
        if not _run_meditate(lg, mkt):
            return (lg, mkt, "HELD", ["retrain error"])
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


def _announce_ranks_only(cands: list[dict]) -> None:
    for c in (c for c in cands if c["status"] == "ranks_only"):
        click.secho(
            f"  RANKS-ONLY {c['league']} {c['market']} — best shipping corner needs a non-default "
            "dist loss (not persistable); skipping.",
            fg="yellow",
        )


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
    cands = _candidates(board)
    ready = [c for c in cands if c["status"] == "candidate"]
    _announce_ranks_only(cands)
    if not ready:
        click.echo("no fully-persistable shipping candidates to confirm.")
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
