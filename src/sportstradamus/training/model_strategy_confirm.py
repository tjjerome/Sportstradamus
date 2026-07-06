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

from sportstradamus.helpers.io import MODEL_STATS_PATH, market_file_slug, prune_model_pickle
from sportstradamus.training.model_strategy_sweep import _FAMILIES, _GATES, FamilySpec
from sportstradamus.training.ship_config import STAT_META_PATH, WITHHELD, load_stat_meta

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_STAT_META = pathlib.Path(str(STAT_META_PATH))
_CONFIRM_LOG_ROOT = _REPO_ROOT / "research" / "logs" / "confirm"
_SHIPPED_DEVEL = "devel"
# A full-HPO meditate confirm is ~1 h; a large cell can run longer, so a 4 h ceiling keeps a hung
# run from blocking the loop forever. A timeout is treated as a failure (the cell auto-reverts).
_CONFIRM_TIMEOUT_S = 4 * 3600


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


def _log_tail(path: pathlib.Path, n: int = 25) -> str:
    return "\n".join(path.read_text().splitlines()[-n:])


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


def _confirm_meditate(league: str, market: str) -> bool:
    """Full-HPO retrain of a cell now marked shipped=devel; True iff it ships on the official scorecard.

    Reads the just-persisted strategy from stat_meta.json (no strategy flags); ``--force`` is required
    or a cell with no new gamedays skips silently and never rewrites model_stats. A non-zero exit or a
    timeout is a failure — don't trust a possibly-stale model_stats row.
    """
    cmd = ["poetry", "run", "meditate", "--league", league, "--market", market, "--force"]
    log_path = _CONFIRM_LOG_ROOT / f"{market_file_slug(league, market)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"  confirming {league} {market} (full HPO, ~1h) …")
    with log_path.open("w") as log:
        try:
            subprocess.run(
                cmd,
                cwd=_REPO_ROOT,
                check=True,
                timeout=_CONFIRM_TIMEOUT_S,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            click.echo(
                f"  meditate {type(exc).__name__} — tail of {log_path}:\n{_log_tail(log_path)}",
                err=True,
            )
            return False
    return _ship_from_model_stats(league, market)


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


def _split_shippable(ready: list[dict], meta: dict) -> tuple[list[dict], list[dict]]:
    """Partition candidates into withheld (auto-shippable, fresh) and already-shipped (skip).

    Only a withheld cell is safe to auto-ship: a fresh cell has no live pickle, so a failed confirm's
    revert+prune restores its original dark state. An already-shipped cell's confirm would overwrite a
    live pickle and its revert would dark-out a serving cell — that is the supersession test's job, so
    the board ranks it but the confirm skips it.
    """
    fresh, shipped = [], []
    for c in ready:
        target = fresh if meta[c["league"]][c["market"]].get("shipped") == WITHHELD else shipped
        target.append(c)
    return fresh, shipped


def _announce_ranks_only(cands: list[dict]) -> None:
    for c in (c for c in cands if c["status"] == "ranks_only"):
        click.secho(
            f"  RANKS-ONLY {c['league']} {c['market']} — best shipping corner needs a non-default "
            "dist loss (not persistable); skipping.",
            fg="yellow",
        )


def _announce_already_shipped(shipped: list[dict], meta: dict) -> None:
    for c in shipped:
        surface = meta[c["league"]][c["market"]].get("shipped")
        click.secho(
            f"  ALREADY-SHIPPED {c['league']} {c['market']} (on {surface}) — ranked, but re-shipping a "
            "live cell is a manual supersession call; skipping.",
            fg="yellow",
        )


def run_confirm(board: pd.DataFrame, *, yes: bool = False) -> None:
    """Persist each withheld cell's best persistable corner, confirm at full HPO, keep passers, revert failures.

    ``board`` is the in-memory sweep result (a row per corner, ``ships`` bool). Only withheld cells are
    auto-shipped; already-shipped cells (present when the sweep ran ``--include-shipped``) are ranked
    but left to the manual supersession test. ``yes`` skips the interactive prompt for unattended runs.
    """
    cands = _candidates(board)
    ready = [c for c in cands if c["status"] == "candidate"]
    _announce_ranks_only(cands)
    if not ready:
        click.echo("no fully-persistable shipping candidates to confirm.")
        return

    meta = load_stat_meta(_STAT_META)
    fresh, shipped = _split_shippable(ready, meta)
    _announce_already_shipped(shipped, meta)
    if not fresh:
        click.echo(
            "no withheld candidates to confirm (shipped cells need the manual supersession test)."
        )
        return

    click.secho(f"\n{len(fresh)} candidate(s) to persist + confirm:", bold=True)
    for c in fresh:
        click.echo(
            f"  {c['league']} {c['market']}: "
            + ", ".join(f"{k}={v}" for k, v in c["edits"].items())
        )
    prompt = (
        f"\nPersist {len(fresh)} config(s) to stat_meta.json and confirm each with a full-HPO "
        "retrain (~1h each)? Failures auto-revert"
    )
    if not yes and not click.confirm(prompt):
        click.echo("aborted; stat_meta.json unchanged.")
        return

    backup = _backup_stat_meta()
    click.echo(f"stat_meta.json backed up to {backup}")
    results = [_confirm_one(meta, c) for c in fresh]
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
    n_ship = sum(1 for r in results if r[2] == "SHIPPED")
    click.secho(
        f"\n{n_ship} shipped (devel), {len(results) - n_ship} reverted. Backup: {backup}",
        fg="green" if n_ship else "yellow",
    )
    if n_ship:
        click.echo("Review the stat_meta.json diff and commit the shipped cells.")
