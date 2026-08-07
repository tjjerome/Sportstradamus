# Supersede-in-Confirm Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `model-strategy-sweep --confirm` re-ship an already-shipped (live) cell via the supersession test (S1/S2/S3) in the same mixed run that auto-ships withheld cells, restoring the incumbent byte-identical on any loss or failure.

**Architecture:** In `training/model_strategy_confirm.py`, add a supersede path parallel to the existing withheld `_confirm_one`: snapshot the incumbent's seven on-disk artifacts, persist the candidate strategy, full-HPO retrain in place, run `scorecard.supersede_verdict` (baseline = snapshot CSV, candidate = fresh CSV), print the comparison and prompt to promote on a passing verdict, and restore the incumbent from the snapshot on HOLD / decline / error / exception via a `try/finally` guard. `run_confirm` routes withheld candidates to `_confirm_one` (unchanged) and shipped candidates to the new `_supersede_one`.

**Tech Stack:** Python 3.11, click, pandas, pytest (all tests mocked — no model trains, no real `stat_meta`/data touched). Reuses `scorecard.supersede_verdict` / `_supersede_headline` / `load_test_set` and `helpers.io` path builders.

---

## File structure

- **Modify** `src/sportstradamus/training/model_strategy_confirm.py` — all new functions live here; the module stays well under the monolith ceiling.
- **Modify** `tests/golden/test_model_strategy_confirm.py` — new snapshot/restore + supersede tests; rewrite the now-obsolete `test_run_confirm_skips_already_shipped`.
- **Modify** `README.md` — extend the `--confirm` paragraph in §1.
- **Modify** `/home/trevor/.claude/projects/-home-trevor-Sportstradamus/memory/project_model_strategy_driver.md` — record the supersede behavior + snapshot/restore landmine.

No new files. No changes to `scorecard.py`, `pipeline.py`, or `model_strategy_sweep.py`.

---

## Task 1: Cell-artifact snapshot and restore

**Files:**
- Modify: `src/sportstradamus/training/model_strategy_confirm.py` (imports + three new functions near the other IO helpers, after `_backup_stat_meta`)
- Test: `tests/golden/test_model_strategy_confirm.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/golden/test_model_strategy_confirm.py`:

```python
def test_snapshot_restore_round_trips_all_artifacts(monkeypatch, tmp_path):
    """Snapshot copies the incumbent artifacts aside; restore puts them back byte-identical and
    restores the stat_meta entry — the safety primitive the supersede path relies on."""
    arts = [tmp_path / name for name in ("NBA_PTS.mdl", "NBA_PTS.csv", "model_stats.parquet")]
    for a in arts:
        a.write_text("incumbent")
    monkeypatch.setattr(mc, "_cell_artifacts", lambda lg, mkt: arts)
    monkeypatch.setattr(mc, "_CONFIRM_LOG_ROOT", tmp_path / "logs")

    backup = mc._snapshot_cell("NBA", "PTS")
    assert (backup / "NBA_PTS.csv").read_text() == "incumbent"  # the S2/S3 baseline copy

    for a in arts:  # a candidate meditate overwrites every artifact in place
        a.write_text("CANDIDATE")
    meta = {"NBA": {"PTS": {"target_normalization": "candidate"}}}
    monkeypatch.setattr(mc, "_atomic_write_meta", lambda m: None)

    mc._restore_cell("NBA", "PTS", backup, meta, {"target_normalization": "ratio_meanyr"})
    assert all(a.read_text() == "incumbent" for a in arts)
    assert meta["NBA"]["PTS"] == {"target_normalization": "ratio_meanyr"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_model_strategy_confirm.py::test_snapshot_restore_round_trips_all_artifacts -v`
Expected: FAIL with `AttributeError: module 'sportstradamus.training.model_strategy_confirm' has no attribute '_cell_artifacts'`

- [ ] **Step 3: Write minimal implementation**

In `model_strategy_confirm.py`, extend the imports. Change the `helpers.io` import line to add `model_pickle_path`:

```python
from sportstradamus.helpers.io import (
    MODEL_STATS_PATH,
    market_file_slug,
    model_pickle_path,
    prune_model_pickle,
)
```

Change the sweep import line to add `_SHIP_PRED_COL` and `_TEST_SETS_ROOT`:

```python
from sportstradamus.training.model_strategy_sweep import (
    _FAMILIES,
    _GATES,
    _SHIP_PRED_COL,
    _TEST_SETS_ROOT,
    FamilySpec,
)
```

Add a new scorecard import line (the module does not import scorecard today):

```python
from sportstradamus.training.scorecard import _supersede_headline, load_test_set, supersede_verdict
```

Then add the three functions after `_backup_stat_meta`:

```python
def _cell_artifacts(league: str, market: str) -> list[pathlib.Path]:
    """Every file a full-HPO ``meditate`` rewrites for one cell — the byte-identical restore set.

    The test-set CSV doubles as the S2/S3 baseline once snapshotted. ``model_stats`` is two files
    (parquet + csv mirror); the three ``data/training`` CSVs and ``stat_calibration.json`` are shared
    files, safe to whole-file restore because a cell's snapshot→restore window is isolated (only that
    cell's ``meditate`` runs between them).
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/golden/test_model_strategy_confirm.py::test_snapshot_restore_round_trips_all_artifacts -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/training/model_strategy_confirm.py tests/golden/test_model_strategy_confirm.py
git commit -m "feat(ship75): cell-artifact snapshot/restore for the supersede path"
```

---

## Task 2: Split the meditate subprocess out of the ship-verdict read

The supersede path needs to run a full-HPO `meditate` and then run *its own* verdict (S1/S2/S3), not the withheld path's bare `ship` read. Extract the subprocess run so both paths share it.

**Files:**
- Modify: `src/sportstradamus/training/model_strategy_confirm.py:129-156` (`_confirm_meditate`)
- Test: `tests/golden/test_model_strategy_confirm.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_meditate_true_on_clean_exit_false_on_error(monkeypatch, tmp_path):
    """`_run_meditate` reports subprocess success/failure only — it does not read the ship verdict."""
    monkeypatch.setattr(mc, "_CONFIRM_LOG_ROOT", tmp_path)
    monkeypatch.setattr(mc.subprocess, "run", lambda *a, **k: None)
    assert mc._run_meditate("NBA", "PTS") is True

    def boom(*a, **k):
        raise mc.subprocess.CalledProcessError(1, "meditate")

    monkeypatch.setattr(mc.subprocess, "run", boom)
    assert mc._run_meditate("NBA", "PTS") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_model_strategy_confirm.py::test_run_meditate_true_on_clean_exit_false_on_error -v`
Expected: FAIL with `AttributeError: ... has no attribute '_run_meditate'`

- [ ] **Step 3: Write minimal implementation**

Replace the existing `_confirm_meditate` (lines 129-156) with a `_run_meditate` subprocess primitive plus a thin composition that keeps the withheld path's behavior identical:

```python
def _run_meditate(league: str, market: str) -> bool:
    """Full-HPO retrain of a cell from its just-persisted stat_meta strategy; True iff meditate exits clean.

    ``--force`` is required or a cell with no new gamedays skips silently and never rewrites its
    outputs. A non-zero exit or a timeout returns False (don't trust a possibly-stale model_stats row).
    """
    cmd = ["poetry", "run", "meditate", "--league", league, "--market", market, "--force"]
    log_path = _CONFIRM_LOG_ROOT / f"{market_file_slug(league, market)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"  retraining {league} {market} (full HPO, ~1h) …")
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
    return True


def _confirm_meditate(league: str, market: str) -> bool:
    """Withheld-path confirm: retrain, then True iff the official scorecard ships the cell."""
    return _run_meditate(league, market) and _ship_from_model_stats(league, market)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/golden/test_model_strategy_confirm.py -v`
Expected: PASS — the new test passes and the existing `test_run_confirm_yes_persists_and_confirms` (which monkeypatches `_confirm_meditate`) is unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/training/model_strategy_confirm.py tests/golden/test_model_strategy_confirm.py
git commit -m "refactor(ship75): extract _run_meditate subprocess from _confirm_meditate"
```

---

## Task 3: The supersede orchestration (`_supersede_one`)

**Files:**
- Modify: `src/sportstradamus/training/model_strategy_confirm.py` (add `_failed_legs` and `_supersede_one` after `_confirm_one`)
- Test: `tests/golden/test_model_strategy_confirm.py`

- [ ] **Step 1: Write the failing tests**

Add a shared helper and four tests. The `_verdict` helper builds a full verdict dict so the real `_supersede_headline` (called inside `_supersede_one`) never crashes on a missing key.

```python
import pytest


def _shipped_meta():
    return {
        "NBA": {
            "PTS": {
                "dist": "SkewNormal",
                "shipped": "devel",
                "target_normalization": "ratio_meanyr",
                "blending": "nll",
            }
        }
    }


def _supersede_cand():
    return {
        "league": "NBA",
        "market": "PTS",
        "family": "SkewNormal",
        "status": "candidate",
        "edits": {"target_normalization": "centered_additive_mean10", "blending": "crps"},
        "slack": 0.2,
    }


def _verdict(*, ship, s1=True, s2=True, s3=True):
    return {
        "s1_pass": s1,
        "s2_n": 120,
        "s2_mean": 0.02,
        "s2_ci_lo": 0.005 if s2 else -0.004,
        "s2_ci_hi": 0.03,
        "s2_pass": s2,
        "s3_sharpe_baseline": 1.10,
        "s3_sharpe_candidate": 1.40,
        "s3_memmel_z": 2.33 if s3 else 0.5,
        "s3_pass": s3,
        "ship": ship,
    }


def _patch_supersede_io(monkeypatch, *, verdict, meditate_ok=True):
    """Patch the heavy IO of _supersede_one; return (restored, pruned) spy lists."""
    monkeypatch.setattr(mc, "_snapshot_cell", lambda lg, mkt: mc.pathlib.Path("/tmp/bk"))
    monkeypatch.setattr(mc, "_run_meditate", lambda lg, mkt: meditate_ok)
    monkeypatch.setattr(mc, "load_test_set", lambda path, col: pd.DataFrame())
    monkeypatch.setattr(mc, "supersede_verdict", lambda *a, **k: verdict)
    monkeypatch.setattr(mc, "_atomic_write_meta", lambda m: None)
    restored, pruned = [], []
    monkeypatch.setattr(mc, "_restore_cell", lambda lg, mkt, bk, m, orig: restored.append((lg, mkt, orig)))
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt)))
    return restored, pruned


def test_supersede_hold_restores_incumbent_and_never_prunes(monkeypatch, capsys):
    """The safety-critical path: a HOLD verdict restores the incumbent and never prunes the live pickle."""
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=False, s3=False))
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "HELD")
    assert result[3] == ["S3"]  # only S3 failed
    assert restored[0][:2] == ("NBA", "PTS")
    assert pruned == []  # live cell keeps serving
    assert "S3" in capsys.readouterr().out  # the comparison was printed


def test_supersede_pass_and_yes_keeps_candidate(monkeypatch):
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))
    monkeypatch.setattr(mc.click, "confirm", lambda *a, **k: True)
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "SUPERSEDED")
    assert restored == []  # winning candidate kept in place
    assert pruned == []


def test_supersede_pass_but_no_restores_incumbent(monkeypatch):
    meta = _shipped_meta()
    restored, _ = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))
    monkeypatch.setattr(mc.click, "confirm", lambda *a, **k: False)
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "HELD")
    assert result[3] == ["declined"]
    assert restored[0][:2] == ("NBA", "PTS")


def test_supersede_meditate_error_restores_incumbent(monkeypatch):
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True), meditate_ok=False)
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "HELD")
    assert result[3] == ["retrain error"]
    assert restored[0][:2] == ("NBA", "PTS")
    assert pruned == []


def test_supersede_restores_on_verdict_exception(monkeypatch):
    """A crash mid-verdict still restores the incumbent via the finally guard, then re-raises."""
    meta = _shipped_meta()
    restored, _ = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))

    def boom(*a, **k):
        raise RuntimeError("verdict blew up")

    monkeypatch.setattr(mc, "supersede_verdict", boom)
    with pytest.raises(RuntimeError):
        mc._supersede_one(meta, _supersede_cand())
    assert restored[0][:2] == ("NBA", "PTS")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/golden/test_model_strategy_confirm.py -k supersede -v`
Expected: FAIL with `AttributeError: ... has no attribute '_supersede_one'`

- [ ] **Step 3: Write minimal implementation**

Add after `_confirm_one`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/golden/test_model_strategy_confirm.py -k supersede -v`
Expected: PASS (all five supersede tests)

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/training/model_strategy_confirm.py tests/golden/test_model_strategy_confirm.py
git commit -m "feat(ship75): _supersede_one — S1/S2/S3 test with byte-identical incumbent restore"
```

---

## Task 4: Route shipped cells through supersede in `run_confirm`

**Files:**
- Modify: `src/sportstradamus/training/model_strategy_confirm.py` (`_split_shippable` docstring, delete `_announce_already_shipped`, add `_announce_plan`, rewrite `run_confirm`, extend `_print_confirm_report`)
- Test: `tests/golden/test_model_strategy_confirm.py` (rewrite the obsolete skip test, add a mixed-board test)

- [ ] **Step 1: Write the failing test**

Delete the existing `test_run_confirm_skips_already_shipped` (its behavior is gone) and add:

```python
def test_run_confirm_mixed_board_routes_withheld_and_shipped(monkeypatch, capsys):
    """One --confirm run auto-ships the withheld cell via _confirm_one and supersession-tests the live
    cell via _supersede_one — a single combined report."""
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", True, 0.25),  # WNBA AST (withheld)
            {
                "league": "NBA",
                "market": "PTS",
                "family": "SkewNormal",
                "normalization": "centered_additive_mean10",
                "dist_training_loss": "crps",
                "blending_loss_fn": "crps",
                "ships": True,
                "slack": 0.30,
            },
        ]
    )
    meta = {
        "WNBA": {"AST": _sn_original()},  # withheld
        "NBA": {"PTS": {"dist": "SkewNormal", "shipped": "devel",
                        "target_normalization": "ratio_meanyr", "blending": "nll"}},
    }
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(mc, "_backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    calls = {"confirm": [], "supersede": []}
    monkeypatch.setattr(mc, "_confirm_one", lambda m, c: calls["confirm"].append(c["market"]) or ("WNBA", "AST", "SHIPPED", []))
    monkeypatch.setattr(mc, "_supersede_one", lambda m, c: calls["supersede"].append(c["market"]) or ("NBA", "PTS", "SUPERSEDED", []))

    mc.run_confirm(board, yes=True)
    assert calls["confirm"] == ["AST"]
    assert calls["supersede"] == ["PTS"]
    out = capsys.readouterr().out
    assert "SHIPPED" in out and "SUPERSEDED" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_model_strategy_confirm.py::test_run_confirm_mixed_board_routes_withheld_and_shipped -v`
Expected: FAIL — `_supersede_one` is never called because `run_confirm` still routes shipped cells to `_announce_already_shipped` + skip (`calls["supersede"] == []`).

- [ ] **Step 3: Write minimal implementation**

Update `_split_shippable`'s docstring (behavior unchanged) to reflect the new routing:

```python
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
```

Delete `_announce_already_shipped` entirely and add `_announce_plan`:

```python
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
```

Rewrite `run_confirm`:

```python
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
```

Extend `_print_confirm_report` to count both win outcomes. Add a module-level constant near the other constants (after `_CONFIRM_TIMEOUT_S`):

```python
# Confirm outcomes that leave a shippable cell on devel (vs REVERTED / HELD, which change nothing).
_WIN_OUTCOMES: tuple[str, ...] = ("SHIPPED", "SUPERSEDED")
```

Replace the tally in `_print_confirm_report`:

```python
    n_win = sum(1 for r in results if r[2] in _WIN_OUTCOMES)
    click.secho(
        f"\n{n_win} shipped/superseded (devel), {len(results) - n_win} reverted/held. Backup: {backup}",
        fg="green" if n_win else "yellow",
    )
    if n_win:
        click.echo("Review the stat_meta.json diff and commit the shipped/superseded cells.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/golden/test_model_strategy_confirm.py -v`
Expected: PASS (the mixed-board test plus all prior confirm/supersede tests; the deleted skip test is gone).

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/training/model_strategy_confirm.py tests/golden/test_model_strategy_confirm.py
git commit -m "feat(ship75): --confirm supersession-tests live cells instead of skipping them"
```

---

## Task 5: Docs — README and memory

**Files:**
- Modify: `README.md` (§1, the `--confirm` paragraph)
- Modify: `/home/trevor/.claude/projects/-home-trevor-Sportstradamus/memory/project_model_strategy_driver.md`

- [ ] **Step 1: Update the README `--confirm` paragraph**

Replace the current sentence about already-shipped cells being skipped:

```
it). Already-shipped cells (only present under `--include-shipped`) are listed and skipped, not
re-shipped — swapping a live cell's strategy has to clear the supersession test (below).
```

with:

```
it). For an already-shipped cell (only present under `--include-shipped`), `--confirm` runs the
supersession test: it snapshots the incumbent, retrains the candidate in place, and scores S1/S2/S3
(candidate clears the six gates standalone, is paired-Brier sharper, and paired-Sharpe sharper). It
prints the comparison and swaps the live cell only when all three pass **and** you confirm the
promotion; a loss (or a declined prompt) restores the incumbent byte-identical and it keeps serving.
```

- [ ] **Step 2: Update the memory note**

In `project_model_strategy_driver.md`, under the "Family-aware sweep + auto-ship confirm loop" section, append a sentence:

```
`--confirm` also supersession-tests already-shipped cells swept under `--include-shipped`: snapshot the
incumbent's 7 artifacts (pickle/.mdl, test-set CSV, model_stats parquet+csv, feature_importances/_correlations,
stat_calibration.json) → retrain candidate in place → `scorecard.supersede_verdict` S1/S2/S3 → promote on
pass+operator-yes, else `try/finally` restores the incumbent byte-identical (RESTORE, never prune — a live
cell must keep serving). Baseline = the incumbent's on-disk test-set CSV, snapshotted before the retrain.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(ship75): document the --confirm supersession path"
```

(The memory file lives outside the repo and is not committed.)

---

## Task 6: Quality gates

**Files:** none (verification only).

- [ ] **Step 1: Run the refactoring-specialist**

Dispatch the `refactoring-specialist` subagent (per CLAUDE.md, mandatory before review/commit-as-done) on the one touched source file: `src/sportstradamus/training/model_strategy_confirm.py`. Address any items it raises.

- [ ] **Step 2: Run the authoritative gates once**

```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/
poetry run pytest -m integration -n0 && touch /home/trevor/Sportstradamus/.claude/.state/integration_green
```

Expected: ruff clean; golden all pass; integration all pass.

- [ ] **Step 3: Final commit if the specialist changed anything**

```bash
git add -A
git commit -m "style(ship75): refactoring-specialist follow-up on the supersede path"
```

(Skip if the specialist made no changes.)

---

## Self-review notes

- **Spec coverage:** flow (Task 4), snapshot/restore six-artifact set + isolation window (Task 1), try/finally restore-not-prune (Task 3), S2/S3 baseline = snapshot CSV (Task 3), reuse of `supersede_verdict`/`_supersede_headline`/`load_test_set` (Tasks 1/3), all-mocked tests incl. the critical restore-not-prune test (Task 3), README + memory (Task 5), gates + specialist (Task 6). The spec's `_render_supersede_comparison` was dropped in favor of calling `_supersede_headline` directly (a dedicated wrapper would be a banned pure-forwarder); the comparison is still printed and asserted.
- **Type consistency:** every function returns the `tuple[str, str, str, list[str]]` shape `_print_confirm_report` consumes (`_confirm_one` and `_supersede_one` match). `_supersede_headline` reads only keys present in the `_verdict` test fixture and the real `supersede_verdict` output.
- **No placeholders:** every step carries complete code and exact commands.
