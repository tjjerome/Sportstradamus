# Simulated-Bettor Ledger

> Status: QUEUED (entry: D6 — owner signs off policy spec v1)

## 1. Mission & money logic

A pre-registered paper-trading ledger: a defined *reasonable-bettor policy*
selects and sizes entries from each day's recommendations **at decision time**,
the entries are committed to an append-only ledger, settled nightly, and rolled
into a simulated bankroll with ROI/CLV and circuit-breaker analytics.

Nobody can place every entry the engine outputs; real-bet logging is
logistically impossible (owner). This lane is the substitute that still
answers the only question that matters: **is the system actually making money,
forward, without hindsight?** It converts "the gates passed" into measured
walk-forward ROI/CLV, detects live drift the offline gates can't see, and is
the evidence gate D7 (real-stake scaling) reads. Every week not logging is
evidence permanently lost — the lane's value accrues with calendar time.

The existing [`strategies/profit_sim.py`](../../src/sportstradamus/strategies/profit_sim.py)
+ [`pages/6_Stats_Profit_Sim.py`](../../src/sportstradamus/pages/6_Stats_Profit_Sim.py)
are **retrospective** Monte Carlo over resolved history — strategy exploration
with hindsight and selection effects. This ledger is the forward sibling, not a
replacement: `profit_sim.py` is load-bearing for the S3 supersede gate
([`ship_gate.md`](../ship_gate.md) §S3 paired Sharpe) and for Gate-2 yield
(`nightly._profit_sim_kelly_yield`). **Do not modify it in this lane.**

## 2. Read first (in order)

1. [`strategies/underdog_pickem.py`](../../src/sportstradamus/strategies/underdog_pickem.py)
   — `construct_entries` + the recommendations YAML this policy consumes.
2. [`strategies/kelly.py`](../../src/sportstradamus/strategies/kelly.py) —
   sizing the policy reuses (`fractional_kelly_stake`, shrinkage blend).
3. [`clv.py`](../../src/sportstradamus/clv.py) +
   [`nightly.py`](../../src/sportstradamus/nightly.py) `reflect` — the
   settlement + CLV machinery stage 2 extends.
4. [`strategies/profit_sim.py`](../../src/sportstradamus/strategies/profit_sim.py)
   — read-only; the retrospective sibling whose presets inform the policy.
5. CLAUDE.md §Production deployment — `run_job.sh` semantics (flock,
   healthchecks) before any cron wiring.
6. `data/config/underdog_payouts.json` — payout/push rules settlement applies.

## 3. Verify before you trust

If command output contradicts brief prose, the output wins — fix in place
(minor) or stop and ask (material).

```bash
git fetch origin && git log --oneline origin/devel -3
ls data/recommendations/ 2>/dev/null | tail -3   # pickem-build output present?
ls data/ledger/ 2>/dev/null                       # prior-stage artifacts
grep -n "pickem" scripts/run_job.sh               # cron wiring state
poetry run pickem-build --help                    # CLI alive
```

Known at creation: `pickem-build` is **not** in the production crontab
(CLAUDE.md §Production deployment lists prophecize/confer/meditate/reflect/
close-lines/gate-status/fp-fetch only) — stage 1 adds the daily decision-time
job; cron edits are owner-approved.

### Volatile product assumptions

- Underdog payout multipliers + push/void rules
  (`data/config/underdog_payouts.json`) — re-verify against the live product
  each season; settlement math inherits them.
- Sleeper payout/push rules once the `sleeper-parity` lane lands — this
  ledger's schema is platform-aware from day one and starts logging Sleeper
  entries the day that lane ships.

## 4. Locked decisions

- 2026-06-10 — Simulated bettor replaces placed-bet logging; never resurrect a
  `tracking/` package (owner; roadmap v3 §8).
- 2026-06-10 — Entries are committed **at decision time** and are immutable:
  append-only store, no backfill code path may exist, settlement never mutates
  an entry. Pre-registration is the lane's entire epistemic value.
- 2026-06-10 — Policy is versioned (`policy_v1`, `policy_v2`, …); any change
  to its parameters is a new version logged alongside, never an in-place edit
  — mid-stream changes break comparability.
- 2026-06-10 — Money is `Decimal`, never float (CLAUDE.md §General Rules).
- 2026-06-10 — `strategies/profit_sim.py` untouched by this lane (§1).

## 5. Module footprint & canonical paths

New `sportstradamus.strategies` module (e.g. `strategies/ledger.py`) +
`sportstradamus.nightly` (settlement extension) + one new `pages/` view +
`scripts/run_job.sh` job entry (owner-approved) + `tests/golden/`. Reads
`sportstradamus.strategies.kelly` and `clv`; the dashboard page reads parquet
snapshots only — never DuckDB (CLAUDE.md §Hard rules;
`tests/golden/test_dashboard_no_archive_lock.py`).

## 6. Stage plan

0. **Policy spec v1 → D6.** Draft the reasonable-bettor policy for owner
   sign-off: max entries/day, EV floor, allowed entry sizes/variants,
   quarter-Kelly sizing via `fractional_kelly_stake` with its shrinkage blend,
   fixed paper bankroll, one decision snapshot per day (at the daily
   `pickem-build` run). Acceptance: owner flips D6. 1 session.
   *If it stalls:* park lane `BLOCKED (on: D6)`.
1. **Commit path.** `policy_v1` selects from the day's recommendations and
   appends to `data/ledger/entries/{date}.jsonl`: legs, lines, model probs,
   book devig, stake, policy version, git SHA, `committed_at`. Idempotent
   (re-runs don't duplicate); wired as a daily job through `run_job.sh`
   (owner approves crontab line). Acceptance: two consecutive live days
   produce entries; re-run produces no dupes. 2 sessions.
   *If recommendations are empty on quiet days:* that's data, not failure —
   log the empty decision.
2. **Settlement.** Extend `nightly.py:reflect`: resolve each ledger entry's
   legs, apply push/void rules and the payout curve, join per-leg CLV via
   `clv.fill_from_archive`, write `data/ledger/bankroll.parquet` (daily
   series) + settled-entry parquet. Acceptance: hand-checked settlement of a
   known day matches; CLV populated for ≥ 90% of settled legs. 2 sessions.
3. **Analytics + dashboard.** New page (parquet only): bankroll curve, ROI,
   CLV per segment, drawdown; circuit-breaker state written to a small file
   the daily job reads — drawdown > 20% from peak ⇒ policy halves stakes,
   > 30% ⇒ halt new entries until owner reset. Acceptance: breaker state
   flips on a synthetic drawdown fixture. 1–2 sessions.
4. **Hindsight-proofing tests.** Golden tests: `committed_at` strictly before
   the earliest leg's game start; settlement is append/derive-only (entry
   files byte-identical after settle); no code path writes a past date's
   entry file. Acceptance: tests in `tests/golden/`, red if anyone adds a
   backfill. 1 session.

Lane-level if-it-fails: if the policy proves degenerate (e.g. EV floor never
met, near-zero entries/week), that is a finding about the engine — record it,
loosen only via `policy_v2` with owner sign-off, never silently.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > this brief >
  roadmap v3.
- Owner is open to improvements on the Monte Carlo methodology — propose in a
  ledger note or dispatch `research-analyst` (Opus) for breaker-threshold /
  policy-design literature; don't improvise silent changes.
- Timestamps UTC; one decision snapshot per day — intraday re-runs replace
  nothing, they no-op (idempotency).

## 8. Escalation & stop conditions

**Stop and ask the owner:** D6 unmet; any crontab/`run_job.sh` change; any
schema change after live entries exist (migration needs sign-off); breaker
thresholds (owner parameters); gates red at session start; grind detector
(two sessions, no acceptance movement).

**Park and pivot:** blocked externally ⇒ ledger line + `BLOCKED (on: …)` +
flip roadmap v3 §4 row, pick another lane.

**Dispatch:** `research-analyst` (Opus) optional for policy/breaker design;
`devel-ship-curator` for every devel-bound PR; `refactoring-specialist` per
the five CLAUDE.md triggers.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session.
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- One ledger line appended below; status line updated on stage boundaries.
- Never push `devel` directly — the curator carves ship PRs.
- Durable non-obvious lesson? Offer a memory capture.

## 10. Ledger (append-only, newest first, cap ~15)

- 2026-06-10 · created · brief drafted from roadmap-v3 migration · next: stage 0 policy spec v1 for D6
