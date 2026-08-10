# Hygiene & Closeout

> Status: ACTIVE — stage 1 (stage 0, the roadmap-v3 migration, landed with this file)

## 1. Mission & money logic

Close the small open items the v2→v3 review surfaced and keep the doc/config
surface honest. Stale docs are the most expensive failure mode of a
months-long agent-driven repo: sessions burn time — and make wrong ships —
acting on dead claims. One item here is direct money evidence: the committed
parlay calibration plot is a placeholder, so the system's joint-probability
accuracy has never been verified on production data.

## 2. Read first (in order)

1. [`../sportstradamus_roadmap_v3.md`](../sportstradamus_roadmap_v3.md) — the
   swimlane index this lane maintains.
2. [`../archive/sportstradamus_roadmap_v2.md`](../archive/sportstradamus_roadmap_v2.md)
   §1.6 — the deprecated-triage acceptance spec (decision-only).
3. `src/deprecated/README.md` + `src/deprecated/` contents — the triage
   subject.
4. [`../../src/sportstradamus/scripts/audit_parlay_calibration.py`](../../src/sportstradamus/scripts/audit_parlay_calibration.py)
   — the calibration harness stage 2 re-runs.
5. [`model_improvement_track.md`](model_improvement_track.md) §8 — the
   no-defer / matrix-exhaustion policy stage 3's drift sweeps guard
   (`deferred-90` is a retired tag).

## 3. Verify before you trust

```bash
git fetch origin && git log --oneline origin/devel -3
ls docs/DEPRECATED_TRIAGE.md 2>/dev/null          # stage 1 done?
ls src/deprecated/                                 # triage subject
grep -rn "deferred-90" docs/*.md   # only model_improvement_track.md's retirement notes should hit
grep -rn "roadmap_v2" docs/*.md .claude/ 2>/dev/null | grep -v archive  # drift
```

### Volatile product assumptions

- `data/config/underdog_payouts.json` multipliers vs the live Underdog
  product — the recurring per-season check this lane owns (stage 3).

## 4. Locked decisions

- 2026-06-10 — Triage is decision-only: `docs/DEPRECATED_TRIAGE.md` records
  REVIVE / DELETE / ARCHIVE per file; deletions land only after owner review
  (archived v2 §1.6 spec).
- 2026-06-10 — The calibration re-run needs production archive data; it is
  owner-assisted, never faked from fixtures.

## 5. Module footprint & canonical paths

Docs (`docs/`, `docs/handoffs/`), `src/deprecated/` (decisions about, not
edits to), `src/sportstradamus/scripts/` (read/run), `pyproject.toml`
(script-registration check only). No `sportstradamus.*` runtime modules.

## 6. Stage plan

1. **Deprecated triage** (archived v2 §1.6). For each file in
   `src/deprecated/` and each README TODO: does a live replacement exist
   (CONTRIBUTING §Package Map)? Write a one-paragraph decision — REVIVE (with
   target lane), DELETE (rationale), ARCHIVE (rationale) — to
   `docs/DEPRECATED_TRIAGE.md`; remove matching README TODOs for DELETEs;
   report counts. Known inputs: `correlation.py` and `opt_parlay.py` are
   superseded by live code; `opt_kelley_bet.py` already revived into
   `strategies/kelly.py` (sits in `.archived/`); `drafts/` feeds the
   `bestball-2027` lane's stage-0 retire-vs-evolve call. Acceptance: doc
   exists, every file covered, counts reported. 1–2 sessions. No code
   deletion in this stage.
2. **Parlay calibration re-run** (owner-assisted). Run
   `audit_parlay_calibration.py` against production archive data; commit the
   real plot + CSV over the placeholder, and report PSD-repair distortion
   stats alongside the reliability deciles (PARLAY_AUDIT.md §2.2 rider). The
   populated artifacts double as the incumbent baseline
   [`parlay-dependence.md`](parlay-dependence.md) stage 4 must beat.
   Acceptance: committed artifacts derive from production data (dates + row
   counts in the ledger line). 1 session + owner time. *If production data
   can't be exported:* park the stage, note in ledger, move on.
3. **Drift sweeps + recurring checks.** (a) `deferred-90` drift grep (§3
   block) — the tag is retired
   ([`model_improvement_track.md`](model_improvement_track.md) §8);
   only its retirement notes should match. (b) Verify `pyproject.toml`
   script registrations match shipped CLIs. (c) Run the recurring checks and
   record results: per-season `underdog_payouts.json` verification vs the
   live product (archived v2 §3.2 mandate); monthly free-passer re-score
   reminder to the model-track lane (model_improvement_track.md §6.0).
   (d) Docs relative-link sweep across `docs/` — every relative link
   resolves; no stale paths.
   Acceptance: each check's result in the ledger. ~1 session each visit;
   repeats.

Lane-level: items here are independent; any can be parked without blocking
the others.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > home-of-record
  doc > this brief > roadmap v3.
- Doc edits follow STYLE_GUIDE §16: revise stale statements in place, one
  canonical home per fact, changelogs short and caveman.
- This lane maintains roadmap v3's §4 status column when lanes flip state —
  any session may flip the enum; only this lane does broader v3 edits.

## 8. Escalation & stop conditions

**Stop and ask the owner:** any actual deletion under `src/deprecated/`
(stage 1 is decision-only); production-data export (stage 2); payout-table
mismatches found by a recurring check (product drift — triggers the v3 §6
product-change protocol for affected lanes).

**Park and pivot:** per stage, freely — items are independent.

**Dispatch:** `refactoring-specialist` if any `.py` is touched (unlikely);
`devel-ship-curator` for devel-bound PRs.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session (if any).
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- One ledger line appended below; status line updated on stage boundaries.
- Never push `devel` directly — the curator carves ship PRs.
- Durable non-obvious lesson? Offer a memory capture.

## 10. Ledger (append-only, newest first, cap ~15)

- 2026-07-11 · stage 3 scope+ · recurring checks gain (d) docs relative-link sweep (roadmap audit found a stale model_stats path in v3 §2/§9) · next: unchanged
- 2026-07-10 · stage 2 scope+ · calibration re-run gains PSD-distortion rider + parlay-dependence stage-4 baseline role (PARLAY_AUDIT.md refresh dispositions) · next: unchanged
- 2026-06-10 · stage 0 · roadmap v3 + 7 briefs + v2 archived + pointers repointed (this migration) · next: stage 1 deprecated triage
