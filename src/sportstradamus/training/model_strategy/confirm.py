"""Operation Ship 75 confirm-and-ship loop: rank swept nominees blind at full HPO, ship the top one.

The strategy sweep (:mod:`sportstradamus.training.model_strategy.sweep`) *ranks* corners on fixed-HP
deterministic trials — it never ships. This module turns a ranked board into shipped cells:

1. For each cell, nominate the top :data:`confirm_nominees._CONFIRM_TOP_K` corners **across the
   cell's swept families** by ship slack — a count cell's slice carries ZINB, NegBin and DPO
   corners ranked together — then its seed corners and its incumbent. Every swept axis has a
   ``stat_meta.json`` field, and each nominee's own family's ``persist`` map builds the edits (so
   a NegBin corner outranking ZINB persists ``dist=NegBin`` and flips the cell's family).
2. Prompt, then per cell: pin the training matrix and run **every** nominee at full HPO in a
   research sandbox (``meditate --holdout-blind --artifact-output``, each control a flag, so
   stat_meta.json is never written for it). A live cell's incumbent runs first: its run is the
   baseline every contender meets, so one that never scores holds the cell before the rest train.
   The board's own scorer reads each sandbox dump on the cross-fit validation half, the half the
   board ranks on; the ship holdout stays unread.
3. Rank the blind rows. A priced cell (rule PR-1,
   :mod:`sportstradamus.training.model_strategy.priced_select`) picks the eligible nominee whose
   shrunk priced contrast with the incumbent (the book, on a withheld cell) clears the margin. A
   tier-U nominee passes on a blind 6/6 — on a live cell, on S1/S2/S3 against the incumbent's own
   blind run — and passers sort by blind slack. A cell with no pick is held, its ship holdout
   never read.
4. Persist the pick alone (its edits, plus ``shipped="devel"`` on a withheld cell) and refit
   it canonically with ``--params-from`` its sandbox pickle, so the booster that ships is the
   booster that ranked. Its ship-holdout verdict is read **once**: a pass keeps the cell on devel
   (a live cell promotes on S1/S2/S3 against the served incumbent; a priced cell on S1 and the
   harm guard, S2/S3 reported only), and a failure ends the cell —
   no second nominee is read on the test half. A failure **auto-reverts**: restore the original
   stat_meta entry *and* :func:`prune_model_pickle`. The prune keeps a failed cell's pickle from
   riding a later send; serving reads ``shipped`` (a withheld cell's quoted offers serve the book
   consensus, its unquoted props leave the board), so the stat_meta restore alone already stops
   its model probabilities.

The walk also owns the g4-only calibrated fallback: a nominee whose blind run fails ship on a
near-miss Gate 4 alone runs blind once more under ``hpo_selection: "calibrated"``; the retry's row
is the one that ranks, and a retry that wins persists that pin with the cell's other edits. The
weekly ``meditate`` never explores — it trains each cell once with whatever this walk persisted.

Known limits: a calibrated-HPO nominee ranks optimistically, because its trial pick reads the same
validation half its blind row is scored on. A nominee whose blind row reached the ledger is decided
on that matrix, so an interrupted walk's blind-run nominees do not re-rank on a rerun, and a live
cell walks once per matrix (its incumbent's baseline run decides it too). A SkewNormal nominee on a
live count cell fails its blind run: meditate keeps a live cell's ``target_normalization: none``
over the flag. The compute is every nominee at full HPO plus one pinned refit per cell.

Step 1 lives in :mod:`sportstradamus.training.model_strategy.confirm_nominees` and the machinery
one nominee runs through in :mod:`sportstradamus.training.model_strategy.confirm_runs`; this
module is the walk that drives them.

Everything is local and uncommitted — the human reviews the ``shipped: devel`` diff and commits it.
A whole-file ``stat_meta.json`` backup is taken before any write as the crash/abort safety net.
"""

import functools
import pathlib
import time
from copy import deepcopy
from datetime import UTC, datetime

import click
import pandas as pd
import tabulate

from sportstradamus.helpers.io import market_file_slug, model_pickle_path, prune_model_pickle
from sportstradamus.scripts.priced_score import STATISTICS, p2_rungs
from sportstradamus.scripts.tail_pricing import price_dumps

# _cell_context is read off sweep at call time, the one binding all three confirm modules share.
from sportstradamus.training.model_strategy import (
    confirm_nominees,
    confirm_runs,
    priced_select,
    sweep,
)
from sportstradamus.training.model_strategy.identity import corner_fingerprint
from sportstradamus.training.model_strategy.progress import human_seconds, wrapped
from sportstradamus.training.model_strategy.registry import CellContext
from sportstradamus.training.model_strategy.sweep import (
    _SHIP_PRED_COL,
    _TEST_SETS_ROOT,
    _incumbent_corner,
)
from sportstradamus.training.scorecard import (
    _supersede_headline,
    load_test_set,
    supersede_verdict,
)
from sportstradamus.training.ship_config import WITHHELD, load_stat_meta

_SHIPPED_DEVEL = "devel"
# Leagues whose withheld cells confirm may never auto-flip: a board-passing cell is announced and
# skipped until the owner's activation gates (D1/D2) go GO, then the league is removed here in the
# same PR that ships its first cells. Empty since the MLB+NHL GO (2026-07-09,
# docs/archive/mlb-nhl-activation.md); machinery stays for the next league onboarding.
_ACTIVATION_GATED_LEAGUES: tuple[str, ...] = ()
# Confirm outcomes that leave a shippable cell on devel (vs REVERTED / HELD, which change nothing).
_WIN_OUTCOMES: tuple[str, ...] = ("SHIPPED", "SUPERSEDED")
# A guard read prices every test-half rung: no game date precedes this one.
_EVERY_DATE = "1900-01-01"
# E1's columns for each test-half statistic of a priced pick (``priced_select.contrast``'s).
_PICK_LOG_COLUMNS = (
    "n_rows",
    "n_dates",
    "estimate",
    "lo",
    "hi",
    "se",
    "deff_player",
    "s",
    "shrunk",
)


def _confirm_one(meta: dict, cand: dict) -> tuple[str, str, str, list[str]]:
    """Persist and refit the walk's pick, then keep it (devel) or revert it (stat_meta + pickle).

    The pickle prune on failure keeps the artifact set honest: serving already prices a
    withheld cell at the book consensus, and a volume denominator (``VOLUME_STATS``) keeps
    its pickle.

    A priced pick must also clear the harm guard against the book (:func:`_priced_guard`). The
    refit fits its own blend, so both weights print: the nominee's, which ranked, and the refit's,
    which would serve. A pick reverted here keeps its ledger row's 6/6 verdict, written before
    the guard, so the corner is decided on this matrix and walks again only once the matrix hash
    moves.
    """
    lg, mkt = cand["league"], cand["market"]
    original = deepcopy(meta[lg][mkt])
    backup = confirm_runs.snapshot_cell(lg, mkt)
    keep = False
    try:
        meta[lg][mkt].update(cand["edits"])
        meta[lg][mkt]["shipped"] = _SHIPPED_DEVEL
        confirm_runs.atomic_write_meta(meta)
        failed = confirm_runs.confirm_meditate(lg, mkt, cand)
        if not failed:
            weight = confirm_runs.served_weight(model_pickle_path(lg, mkt))
            click.echo(f"  w {cand['nominee_weight']:.3f} (nominee) -> {weight:.3f} (refit)")
            failed = [] if cand["priced_tier"] == "U" else _priced_guard(cand, None, weight)
            if not failed:
                keep = True
                click.secho(f"  SHIPPED (devel) {lg} {mkt} w={weight:.3f}", fg="green")
                return (lg, mkt, "SHIPPED", [])
        click.secho(f"  REVERTED {lg} {mkt} — failed {' '.join(failed)}", fg="red")
        return (lg, mkt, "REVERTED", failed)
    finally:
        if not keep:
            confirm_runs.restore_cell(lg, mkt, backup, meta, original)
            prune_model_pickle(lg, mkt)


def _priced_guard(cand: dict, backup: pathlib.Path | None, w_refit: float) -> list[str]:
    """PR-1.7's harm guard on a priced pick's canonical refit; empty keeps it.

    The refit's test half meets the incumbent's snapshot in ``backup``, or the book on authentic
    rows when ``backup`` is ``None`` (a withheld cell). P1 reads each dump as training served it,
    as the selection half did. Tier P2 prices both arms' posted rungs through their own pickles,
    as ``ship priced`` does; a refit whose rungs cannot be replayed reads no ``p2_dbrier`` and
    reverts. Every read is logged for E1 (:func:`_log_pick`).
    """
    lg, mkt = cand["league"], cand["market"]
    slug = market_file_slug(lg, mkt)
    refit = _TEST_SETS_ROOT / f"{slug}.csv"
    baseline = None if backup is None else load_test_set(backup / refit.name, _SHIP_PRED_COL)
    p2 = None
    if cand["priced_tier"] == "P2":
        with sweep._archive_read_only() as con:
            arms = [
                price_dumps([dump], con, _EVERY_DATE, models)
                for dump, models in (
                    (refit, model_pickle_path(lg, mkt).parent),
                    (backup / refit.name, backup),
                )
            ]
        for *_, excluded in arms:
            for reason in excluded.values():
                click.echo(f"  no P2 rungs: {reason}")
        p2 = p2_rungs(arms[0][0], arms[1][0])
    table = priced_select.contrast(
        priced_select.served(load_test_set(refit, _SHIP_PRED_COL)),
        None if baseline is None else priced_select.served(baseline),
        slug,
        p2,
    )
    failed = priced_select.harm(table, cand["priced_tier"], cand["nominee_weight"], w_refit)
    _log_pick(cand, table, w_refit, failed)
    for statistic, read in table.iterrows():
        click.echo(
            f"  test half {statistic}: {read['estimate']:+.3f} [{read['lo']:+.3f}, "
            f"{read['hi']:+.3f}] s {read['s']:.3f}, {read['n_dates']} dates"
        )
    return failed


def _log_pick(cand: dict, table: pd.DataFrame, w_refit: float, failed: list[str]) -> None:
    """E1's record of a priced pick: its selection-half read beside every test-half statistic.

    One row per ``ship priced`` statistic, blank where the test half has none, appended beside the
    nominee ledger so the selection-to-confirmation slope and tau0 can be refit on the pairs.
    """
    path = sweep.NOMINEE_LEDGER_PATH.with_name("priced_picks.csv")
    read = table.reindex(index=pd.Index(STATISTICS, name="statistic"), columns=_PICK_LOG_COLUMNS)
    picks = read.reset_index().assign(
        logged_at=datetime.now(UTC).isoformat(timespec="seconds"),
        league=cand["league"],
        market=cand["market"],
        corner_fingerprint=cand["corner_fingerprint"],
        matrix_hash=cand["matrix_hash"],
        **{f"sel_{key.removeprefix('priced_')}": cand[key] for key in priced_select.COLUMNS},
        w_nominee=cand["nominee_weight"],
        w_refit=w_refit,
        reverted=" ".join(failed),
    )
    if path.exists():
        # Rewritten rather than appended, as the nominee ledger is: a column added later would
        # land under an old header no reader can parse.
        picks = pd.concat([pd.read_csv(path), picks], ignore_index=True)
    picks.to_csv(path, index=False)


def _failed_legs(verdict: dict) -> list[str]:
    """The supersession legs a HOLD verdict failed (for the report's failed-gates column)."""
    return [leg for leg in ("S1", "S2", "S3") if not verdict[f"{leg.lower()}_pass"]]


def _supersede_failures(verdict: dict, w_in: float, w_out: float) -> list[str]:
    """Why an S1/S2/S3 verdict holds the incumbent; empty means the candidate may replace it.

    The walk's blind ranking and the canonical refit's test-half read apply this one rule. Both
    weights, the incumbent's and the candidate's, print on the headline.
    """
    click.echo(f"  {_supersede_headline(verdict)}  w {w_in:.3f}→{w_out:.3f}")
    return _failed_legs(verdict)


def _supersede_one(
    meta: dict, cand: dict, *, auto_promote: bool = False
) -> tuple[str, str, str, list[str]]:
    """Supersession-test the walk's pick on one live cell; promote it or restore the incumbent.

    Snapshots the incumbent, refits the pick canonically in place, holds the refit to the withheld
    lane's post-refit verdict (:func:`confirm_runs.refit_failures`: the six gates, then exact
    artifact identity and the official ``model_stats`` ship), then runs S1/S2/S3 against the served
    incumbent's test set. A swap needs the test to pass AND an explicit promote confirmation; every
    other exit — HOLD, decline, refit error, or an exception — hits the ``finally`` restore, which
    copies the snapshot back byte-identical and never prunes, so the incumbent keeps serving.
    ``auto_promote`` answers that confirmation yes so an unattended run can swap live cells: the
    S1/S2/S3 verdict and the six gates still decide, only the human veto is gone. Every swap is
    still local and uncommitted, so the review of the ``stat_meta.json`` diff is where a promotion
    is actually accepted. A priced pick (PR-1.7) is decided by S1 and the harm guard against the
    snapshot (:func:`_priced_guard`); its S2 and S3 print but never hold it. Both of the pick's
    weights print, the nominee's and the refit's.
    """
    lg, mkt = cand["league"], cand["market"]
    slug = market_file_slug(lg, mkt)
    original = deepcopy(meta[lg][mkt])
    backup = confirm_runs.snapshot_cell(lg, mkt)
    keep = False
    try:
        meta[lg][mkt].update(cand["edits"])  # shipped left as-is; the cell stays live
        confirm_runs.atomic_write_meta(meta)
        error, diverged = confirm_runs.run_canonical(lg, mkt, cand)
        if error:
            return (lg, mkt, "HELD", [f"retrain {error}"])
        prefix = ["diverged"] if diverged else []
        failed = confirm_runs.refit_failures(lg, mkt, cand)
        if failed:
            return (lg, mkt, "HELD", prefix + failed)
        verdict = supersede_verdict(
            load_test_set(backup / f"{slug}.csv", _SHIP_PRED_COL),
            load_test_set(_TEST_SETS_ROOT / f"{slug}.csv", _SHIP_PRED_COL),
            _SHIP_PRED_COL,
            league=lg,
            market=mkt,
        )
        weight = confirm_runs.served_weight(model_pickle_path(lg, mkt))
        failed = _supersede_failures(
            verdict, confirm_runs.served_weight(backup / model_pickle_path(lg, mkt).name), weight
        )
        click.echo(f"  w {cand['nominee_weight']:.3f} (nominee) -> {weight:.3f} (refit)")
        if cand["priced_tier"] != "U":
            failed = [leg for leg in failed if leg == "S1"] + _priced_guard(cand, backup, weight)
        if failed:
            return (lg, mkt, "HELD", prefix + failed)
        edits = ", ".join(f"{k}={v}" for k, v in cand["edits"].items())
        if auto_promote:
            click.secho(f"  auto-promoting {lg} {mkt} to {edits}", fg="yellow")
        elif not click.confirm(f"  Promote {lg} {mkt} to {edits}?"):
            return (lg, mkt, "HELD", ["declined"])
        keep = True
        return (lg, mkt, "SUPERSEDED", [])
    finally:
        if not keep:
            confirm_runs.restore_cell(lg, mkt, backup, meta, original)


def _split_shippable(
    ready: list[list[dict]], meta: dict, *, fresh_only: bool = False
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Partition *cells* by current release surface: withheld (fresh) vs already-shipped (live).

    Withheld cells auto-ship on a clean 6/6 (a fresh cell has no live pickle, so a failed confirm's
    revert+prune leaves it withheld, serving the book consensus). Live cells route to the
    supersession test, which restores the incumbent byte-identical on a loss rather than pruning
    it. ``fresh_only`` drops the live lane entirely: supersession promotions always prompt per
    cell, so an unattended run would burn each live cell's retrains into guaranteed HELDs.
    """
    fresh, shipped = [], []
    for nominated in ready:
        lg, mkt = nominated[0]["league"], nominated[0]["market"]
        target = fresh if meta[lg][mkt].get("shipped") == WITHHELD else shipped
        target.append(nominated)
    if fresh_only and shipped:
        click.secho(
            f"  fresh-only: skipping {len(shipped)} live cell(s) — supersession promotions "
            "prompt per cell, so they need an attended run.",
            fg="yellow",
        )
        shipped = []
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
    """List the run's plan: each cell's nominees in nomination order, with their source."""
    for label, group, verb in (
        ("withheld cell", fresh, "rank blind, then confirm the top passer"),
        ("live cell", shipped, "rank blind against the incumbent, then S1/S2/S3-test the top one"),
    ):
        if group:
            click.secho(f"\n{len(group)} {label}(s) to {verb}:", bold=True)
            for nominated in group:
                head = nominated[0]
                click.echo(f"  {head['league']} {head['market']} — {len(nominated)} nominee(s):")
                for n, cand in enumerate(nominated, start=1):
                    edits = ", ".join(f"{k}={v}" for k, v in cand["edits"].items())
                    click.echo(wrapped(f"{n}. [{cand['source']}] {edits}", "    ", "       "))


def _undecided(nominated: list[dict]) -> list[tuple[str, dict]]:
    """The cell's nominees the ledger has not decided on this matrix, each with its report label.

    Nomination already drops ledger-decided board corners (brief L4), but it never filters seeds or
    the incumbent, and a concurrent walk can append a verdict between selection and this walk.
    """
    decided = confirm_nominees.decided_pairs()
    walkable = [
        (f"{n}/{len(nominated)} {cand['source']}", cand)
        for n, cand in enumerate(nominated, start=1)
        if (str(cand["corner_fingerprint"]), str(cand["matrix_hash"])) not in decided
    ]
    if len(walkable) < len(nominated):
        click.secho(
            f"  {len(nominated) - len(walkable)} nominee(s) already decided on this matrix — "
            "skipping",
            fg="yellow",
        )
    return walkable


def _incumbent_fingerprint(league: str, market: str) -> str | None:
    """Corner fingerprint of the recipe a live cell serves, or ``None`` once it left the grid."""
    context = sweep._cell_context(league, market)
    incumbent = _incumbent_corner(context)
    if incumbent is None:
        return None
    return corner_fingerprint(*incumbent, str(context.matrix_sha256))


def _blind_failures(
    run: confirm_runs.BlindRun, reference: confirm_runs.BlindRun | None
) -> list[str]:
    """Why a tier-U blind run cannot be the cell's pick; empty means it passes.

    ``reference`` is a live cell's incumbent baseline and ``None`` on a withheld cell. A run that
    fails a gate already fails S1, so the gates name the failure on both lanes.
    """
    if run.failed or reference is None:
        return run.failed
    verdict = supersede_verdict(
        load_test_set(reference.dump, _SHIP_PRED_COL),
        load_test_set(run.dump, _SHIP_PRED_COL),
        _SHIP_PRED_COL,
        league=run.candidate["league"],
        market=run.candidate["market"],
    )
    return _supersede_failures(
        verdict, float(reference.row["model_weight"]), float(run.row["model_weight"])
    )


def _priced_pick(
    contenders: list[tuple[str, confirm_runs.BlindRun]],
    baseline: priced_select.Baseline,
    context: CellContext,
) -> tuple[str, confirm_runs.BlindRun, dict[str, object]] | str:
    """PR-1.6's pick among a priced cell's blind runs, each read against ``baseline``.

    A run is eligible (PR-1.4) on a blind 6/6 whose served mean passes the sanity scan. Axis
    changes, the first tie-break, count the strategy and controls that differ from the cell's
    stat_meta recipe.
    """
    slug = market_file_slug(context.league, context.market)
    incumbent = _incumbent_corner(context)
    recipe = {} if incumbent is None else {"strategy": incumbent[0].slug, **incumbent[1]}
    reads, rows = {}, []
    for label, run in contenders:
        if run.row is None:
            click.echo(f"  {label}: blind FAIL {' '.join(run.failed)}")
            continue
        read = priced_select.selection_columns(
            load_test_set(run.dump, _SHIP_PRED_COL), baseline, slug
        )
        sane = read["priced_mean_ratio"] <= priced_select.MEAN_SANITY
        excluded = [*run.failed, *([] if sane else ["mean scan"])]
        corner = {"strategy": run.candidate["strategy_slug"], **run.candidate["controls"]}
        click.echo(
            f"  {label}: blind {'FAIL ' + ' '.join(excluded) if excluded else 'PASS'}, "
            f"{read['priced_stat']} {read['priced_est']:+.3f} s {read['priced_se']:.3f} "
            f"(dates alone {read['priced_se_date']:.3f}), shrunk {read['priced_shrunk']:+.3f}"
        )
        reads[label] = read
        rows.append(
            {
                "label": label,
                "eligible": not excluded,
                "shrunk": read["priced_shrunk"],
                "s": read["priced_se"],
                "axes": sum(corner.get(k) != recipe.get(k) for k in corner.keys() | recipe.keys()),
                "slack": run.row["slack"],
            }
        )
    label, why = priced_select.pick(
        pd.DataFrame(rows, columns=["label", *priced_select.NOMINEE_COLUMNS]).set_index("label")
    )
    click.echo(f"  tier {baseline.tier}: {why}")
    if label is None:
        return f"tier {baseline.tier}: {why}"
    return label, dict(contenders)[label], reads[label]


def _slack_pick(
    contenders: list[tuple[str, confirm_runs.BlindRun]],
    reference: confirm_runs.BlindRun | None,
) -> tuple[str, confirm_runs.BlindRun, dict[str, object]] | str:
    """A tier-U cell's pick: the passing run with the most blind slack, or why the cell is held.

    Slack is the value the board ranks on, and a strict comparison keeps nomination order on a tie.
    """
    best = None
    for label, run in contenders:
        failed = _blind_failures(run, reference)
        click.echo(f"  {label}: blind {'FAIL ' + ' '.join(failed) if failed else 'PASS'}")
        if not failed and (best is None or run.row["slack"] > best[1].row["slack"]):
            best = (label, run)
    return "no blind pass" if best is None else (*best, {"priced_tier": "U"})


def _top_blind_passer(
    measured: list[tuple[str, list[confirm_runs.BlindRun]]],
    reference: confirm_runs.BlindRun | None,
) -> tuple[str, confirm_runs.BlindRun, dict[str, object]] | str:
    """The cell's pick with its selection-half priced read, or why the cell is held.

    Each nominee's last run contends. The live baseline is never its own contender; an incumbent
    whose calibrated retry fired still contends through that retry. The cell's tier comes from the
    baseline's dump, else the first scored contender's (a withheld cell meets the book): a priced
    cell picks by PR-1.6 (:func:`_priced_pick`), a tier-U cell on slack (:func:`_slack_pick`).
    """
    contenders = [(label, runs[-1]) for label, runs in measured if runs[-1] is not reference]
    source = reference or next((run for _, run in contenders if run.row is not None), None)
    baseline = None
    if source is not None:
        context = sweep._cell_context(source.candidate["league"], source.candidate["market"])
        baseline = sweep._priced_baseline(context, source.dump)
    if baseline is None:
        return _slack_pick(contenders, reference)
    return _priced_pick(contenders, baseline, context)


def _walk_nominees(
    meta: dict,
    nominated: list[dict],
    attempt,
    *,
    live: bool = False,
) -> tuple[str, str, str, list[str], str]:
    """Rank a cell's nominees blind, then read the ship holdout once, for the pick only.

    The cell's training matrix is pinned once here (:func:`confirm_runs.pin_cell_matrix`) and every
    undecided nominee runs blind on it (:func:`confirm_runs.measure_blind`); none touches
    stat_meta.json. ``attempt`` — :func:`_confirm_one` or :func:`_supersede_one` — then persists the
    pick and refits it from its sandbox pickle, and that refit's ship-holdout verdict is the cell's:
    a failure is reverted by ``attempt``'s own ``finally`` and ends the walk, because reading a
    second nominee on the test half would be selection on the test half again. With no pick
    (:func:`_top_blind_passer`) the cell is held and its test half never read. The pick carries its
    tier, its selection-half read and its blind blend weight into ``attempt``.

    A live cell measures every nominee against its incumbent's blind run, so its incumbent's recipe
    must be among the undecided nominees; without it (cut by ``--confirm-nominees``, or already
    decided on this matrix) the cell is held before anything runs. The incumbent then trains first,
    and a baseline that never scores holds the cell before any other nominee trains. Ranking still
    reads the nominees in nomination order, so a slack tie goes to the earlier nominee.
    """
    lg, mkt = nominated[0]["league"], nominated[0]["market"]
    walkable = _undecided(nominated)
    if not walkable:
        return (lg, mkt, "SKIPPED", ["prior verdict on this matrix"], "-")
    baseline = _incumbent_fingerprint(lg, mkt) if live else None
    if live and baseline not in {cand["corner_fingerprint"] for _, cand in walkable}:
        return (lg, mkt, "HELD", ["no incumbent baseline"], "-")
    confirm_runs.pin_cell_matrix(lg, mkt)
    blind, reference = {}, None
    # Stable, so only the live baseline moves up; a withheld cell has none and keeps its order.
    order = sorted(walkable, key=lambda pair: pair[1]["corner_fingerprint"] != baseline)
    for label, cand in order:
        click.secho(f"\n  nominee {label} {lg} {mkt}", bold=True)
        blind[label] = confirm_runs.measure_blind(meta, cand)
        if cand["corner_fingerprint"] != baseline:
            continue
        reference = blind[label][0]
        if reference.row is None:
            click.secho(
                f"  HELD {lg} {mkt} — incumbent baseline failed {' '.join(reference.failed)}; "
                f"{len(walkable) - 1} other nominee(s) not run",
                fg="red",
            )
            return (lg, mkt, "HELD", ["incumbent baseline", *reference.failed], "-")
    measured = [(label, blind[label]) for label, _ in walkable]
    pick = _top_blind_passer(measured, reference)
    if isinstance(pick, str):
        return (lg, mkt, "HELD", [pick], "-")
    label, run, read = pick
    click.secho(f"\n  canonical refit of {label} {lg} {mkt}", bold=True)
    weight = float(run.row["model_weight"])
    cand = {**run.candidate, "params_from": run.model, "nominee_weight": weight, **read}
    return (*attempt(meta, cand), label)


def _walk_lanes(
    meta: dict,
    fresh: list[list[dict]],
    shipped: list[list[dict]],
    deadline_hours: float | None,
    auto_promote: bool = False,
) -> list[tuple[str, str, str, list[str], str]]:
    """Walk both lanes cell by cell; past the deadline, remaining cells record SKIPPED instead.

    The deadline clock starts here — after the operator prompt — and is checked between cells, never
    mid-walk: a cell that starts before the deadline finishes its walk, so the budget can overrun by
    at most one cell. :func:`confirm_nominees.candidates` ordered the cells best-first, which is
    what makes the cut land on the weakest tail.
    """
    supersede = functools.partial(_supersede_one, auto_promote=auto_promote)
    deadline = None if deadline_hours is None else time.monotonic() + deadline_hours * 3600
    results = []
    for lane, attempt, live in ((fresh, _confirm_one, False), (shipped, supersede, True)):
        for nominated in lane:
            if deadline is not None and time.monotonic() > deadline:
                lead = nominated[0]
                results.append((lead["league"], lead["market"], "SKIPPED", ["deadline"], "-"))
                continue
            results.append(_walk_nominees(meta, nominated, attempt, live=live))
    return results


def run_confirm(
    board: pd.DataFrame,
    *,
    yes: bool = False,
    max_nominees: int | None = None,
    deadline_hours: float | None = None,
    fresh_only: bool = False,
    auto_promote: bool = False,
) -> None:
    """Confirm the sweep's winners: rank each cell's nominees blind, then refit the top one.

    ``board`` is the in-memory sweep result (a row per corner). Each cell nominates its top corners
    plus any seed and its incumbent. Every nominee runs at full HPO in a research sandbox, scored
    on the cross-fit validation half, and only the cell's pick is persisted and refit canonically,
    its ship holdout read once (:func:`_walk_nominees`). A priced cell picks on its priced contrast
    with the incumbent (rule PR-1); a tier-U cell keeps the slack ranking. A withheld cell keeps
    its pick on a clean 6/6 and, when priced, the harm guard (else reverted+pruned). An
    already-shipped cell (present when the sweep ran ``--include-shipped``) swaps its pick in only
    on a passing test-half verdict (S1/S2/S3 on tier U, S1 and the harm guard when priced) AND an
    operator yes; any loss restores the incumbent. ``yes`` skips only the upfront gate — live-cell
    promotions prompt individually unless ``auto_promote`` answers them. ``max_nominees`` caps each
    cell's walk at its first N nominees; ``deadline_hours`` skips cells not yet started when the
    budget runs out; ``fresh_only`` drops the live lane for unattended runs.
    """
    meta = load_stat_meta(confirm_runs.STAT_META)
    ready = confirm_nominees.candidates(board, meta, max_nominees)
    if not ready:
        click.echo("no confirmable nominees on the board.")
        return

    fresh, shipped = _split_shippable(ready, meta, fresh_only=fresh_only)
    fresh = _drop_activation_gated(fresh)
    if not fresh and not shipped:
        click.echo("no confirmable candidates after the activation gate.")
        return
    _announce_plan(fresh, shipped)
    runs = sum(len(n) for n in fresh) + sum(len(n) for n in shipped)
    each = confirm_runs.confirm_median_seconds()
    budget = f" (~{human_seconds(each)} each, ~{human_seconds(runs * each)} in all)" if each else ""
    prompt = (
        f"\nRank {len(fresh)} withheld and {len(shipped)} live cell(s) blind — {runs} full-HPO "
        f"sandbox runs{budget}, plus calibrated retries — then refit each cell's top passer and "
        "read its ship holdout once? Withheld failures auto-revert; live promotions prompt"
    )
    if not yes and not click.confirm(prompt):
        click.echo("aborted; stat_meta.json unchanged.")
        return

    backup = confirm_runs.backup_stat_meta()
    click.echo(f"stat_meta.json backed up to {backup}")
    _print_confirm_report(_walk_lanes(meta, fresh, shipped, deadline_hours, auto_promote), backup)


def _print_confirm_report(
    results: list[tuple[str, str, str, list[str], str]], backup: pathlib.Path
) -> None:
    """Final table (cell → refit nominee → outcome → failing gates), tally, backup, reminder."""
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
    n_skip = sum(1 for r in results if r[2] == "SKIPPED")
    skipped = f", {n_skip} skipped" if n_skip else ""
    click.secho(
        f"\n{n_win} shipped/superseded (devel), {len(results) - n_win - n_skip} reverted/held"
        f"{skipped}. Backup: {backup}",
        fg="green" if n_win else "yellow",
    )
    if n_win:
        click.echo("Review the stat_meta.json diff and commit the shipped/superseded cells.")
