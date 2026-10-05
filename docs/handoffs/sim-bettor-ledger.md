# Simulated-Bettor Ledger

> Status: ACTIVE (stages 1–2 live from cron on `devel` — stage 3 next)

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
+ the Receipts strategy simulator ([`dashboard/components/profit_sim.py`](../../src/sportstradamus/dashboard/components/profit_sim.py))
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
sportstradamus bet pickem --help                    # CLI alive
```

Known at creation: `pickem-build` is **not** in the production crontab
(CLAUDE.md §Production deployment lists prophecize/confer/meditate/reflect/
close-lines/gate-status/fp-fetch only) — stage 1 adds the daily decision-time
job; cron edits are owner-approved.

### Volatile product assumptions

- Underdog payout multipliers + push/void rules
  (`data/config/underdog_payouts.json`) — re-verify against the live product
  each season; settlement math inherits them.
- Sleeper payout/push rules: `sleeper-parity` stage 4 landed 2026-07-14 (local
  `feature/sleeper-parity`, not yet merged `devel`) — `build_candidate_universe`
  now pulls both platforms into one shared candidate pool/budget per persona;
  see `sleeper-parity.md` §10 stage-4 line for the merge-design rationale and
  two push-refund pricing bugs it fixed along the way.

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
   series) + settled-entry parquet. Resolve the *union of distinct legs*
   once per day and map outcomes onto every replicate entry that cites
   them — never per entry (§10 Ensemble sizing; keeps settlement cost flat
   across the 40-replicate ensemble). Acceptance: hand-checked settlement
   of a known day matches; CLV populated for ≥ 90% of settled legs.
   2 sessions.
3. **Analytics + dashboard.** New page (parquet only): bankroll curve, ROI,
   CLV per segment, drawdown; circuit-breaker state written to a small file
   the daily job reads — drawdown > 20% from peak ⇒ policy halves stakes,
   > 30% ⇒ halt new entries until owner reset (reads **replicate 0's**
   lived drawdown, never the ensemble mean — §10 Ensemble sizing). Reports
   each persona's ensemble-mean ROI/CLV/bankroll plus an IQR/±1 SE spread
   band, labeled as selection-policy spread, not outcome uncertainty; a
   one-time knee-check (ensemble-mean ROI vs `M` at `M ∈ {10,20,40,80,160}`,
   after ~2 weeks live) validates or revises `LEDGER_REPLICATES = 40`.
   Acceptance: breaker state flips on a synthetic drawdown fixture.
   1–2 sessions.
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

## 10. Policy

### Policy v1

Three independent simulated-bettor personas, each run as a **40-replicate
Monte Carlo ensemble** (§ Ensemble sizing) against its own fixed $5,000
paper bankroll per replicate. All three read the same candidate universe
but differ in how they score and select from it. Decisions commit twice
daily. Any future change is a new version (`policy_v2`, …) appended below
policy v1, never an in-place edit of this section (§4 locked decision).

**Bankroll.** $5,000 fixed per persona, per replicate — every one of the 40
replicates tracks its own independent bankroll trajectory (needed for a
well-defined per-replicate ROI). Passed to `construct_entries` as a
constant `bankroll=Decimal("5000")` on every run — never recomputed against
a replicate's drifting settled balance. Keeps stake sizing from fighting
the stage-3 circuit breaker's drawdown response (a compounding bankroll
would shrink stakes mid-drawdown right when the breaker is also halving
them). Only **replicate 0's** bankroll is the canonical pre-registered
capital path (§ Ensemble sizing); replicates 1–39 are notional, for
variance measurement only.

**Candidate universe (shared).** Power (2–3 legs) + Flex (4+ legs) only —
Rivals retired (Underdog dropped the product 2026-09-10); Ladders and Game
Lines out of scope (unimplemented — `dfs-products.md` Stages 3/4). **Scar:** when
either ships, policy_v2 decides whether to admit it. `PickemConfig
.entry_sizes = (2, 3, 4, 5, 6)` (widened from the live dashboard's `(3, 5)`
default), `contest_variants = ("power", "flex")`, `min_ev = 0.05` (upstream
default, unchanged), `top_k = 100` (raised from the live default of 20 so
the size/variant partition — keep `power` + size∈{2,3}, keep `flex` +
size≥4, drop everything else — never
truncates a valid candidate before it's evaluated; `underdog_payouts.json`
currently defines a flex-3 tier this partition treats as not applicable to
selection, flagged not fixed).

**Cross-game / cross-league candidates (new, additive).** The same-game-only
limitation is real, but it lives in
`prediction/correlation.py:_process_league_games`, which partitions legs by
team/game before `beam_search_parlays` ever runs — not in the beam search
itself, which has no game-awareness and would combine legs from anywhere if
given the chance. Natively fixing this means editing
`correlation.py`/`parlay.py`/`joint.py`'s Σ assembly, which both
`dfs-products.md` ("No `parlay.py`/`correlation.py` edits at v1... an edit
to either file is a stop condition, not a judgment call") and the roadmap's
file-conflict-queueing rule (§5.1) gate behind cross-lane coordination this
lane doesn't have. Instead, a new module in the stage-1
`strategies/ledger.py` footprint (import-only against
`prediction.scoring.process_offers`) builds cross-game/cross-league
candidates directly from the day's already-scored single-leg offers — the
same input `_parlays_per_variant` starts from, upstream of the per-game
split — and treats every cross-game leg pair as independent (ρ = 0). This
is the same fallback the current code already applies to any leg pair with
no correlation-matrix entry, so it's a conservative extension of existing
behavior, not a new modeling claim (same-game correlation is the entire
reason the copula machinery exists; cross-game correlation between
unrelated players/games is presumably close to zero anyway). A cross-game
candidate's probability is the plain product of its legs' individual
(shrinkage-adjusted) win probabilities — no MVN/copula machinery needed.
Every committed entry carries a `game_span` field (count of distinct games)
so stage-3 analytics can split same-game vs. cross-game profit.

**Personas.**
- *Safe* — priority score = `joint_prob` (win probability), descending.
  Skews toward smaller, higher-hit-rate entries, since joint probability
  falls as leg count grows.
- *High-EV* — priority score = today's existing `rank_and_dedupe` key,
  `(ev * joint_prob) / stake` (EV-per-dollar-staked). Chases edge magnitude
  regardless of size or hit rate.
- *Kelly-growth* — candidates are first de-overlapped by the same
  Jaccard-decay draw used for selection (below), then sized by
  `strategies/kelly.py:joint_kelly_portfolio` — the existing cvxpy/SCS
  portfolio log-growth optimizer — instead of independent per-entry Kelly
  stakes. This is a genuinely different *selection*, not a relabeled
  ranking: the portfolio solve can allocate zero to a candidate simple
  EV-ranking would take, if funding it doesn't improve joint log-growth
  enough relative to spending that budget elsewhere.
  `joint_kelly_portfolio`'s own docstring assumes independent candidates
  ("callers should de-overlap legs before invocation") — the Jaccard draw
  is what satisfies that precondition; the two are complementary, not
  duplicated machinery.

**Selection (Monte Carlo, Jaccard-similarity-weighted).** Within a run,
candidates are drawn one at a time, without replacement, from the
remaining pool — sequential weighted sampling, rhyming with the pattern in
`strategies/profit_sim.py:_pick_day_bets` (not imported — that function is
private, in a file this lane doesn't touch; policy_v1 uses the same
*shape* in new code). Each candidate's draw weight is its persona-specific
priority score, percentile-normalized against the day's full pool, decayed
by similarity to what's already been drawn:

```
effective_weight = percentile_score * exp(-K * max_jaccard * (1 - percentile_score))
```

`max_jaccard` is the highest Jaccard similarity — on player sets, not legs
or markets — between this candidate and any already-selected entry,
checked against both this run's picks so far and anything the same persona
already committed earlier today (see Cadence). A high-percentile candidate
tolerates more overlap than a marginal one, matching "if the shared legs
rank highly, they'd still take both." `K` is a single named module
constant — an explicit v1 default, not a fitted value; this doc's own §7
already invites tuning knobs like this later. The number of entries
actually taken per draw is itself stochastic, not a hard target of 5 — a
small, mean-reverting-to-the-cap distribution, so a persona doesn't
robotically fill its budget every run. That's the v1 lever for "a real
bettor misses some opportunities"; simulating literal intraday check-in
timing would need offer-arrival timestamps `construct_entries` doesn't
expose today, and is out of scope.

**Cadence.** Two decision runs per day per persona: morning and
mid-afternoon (exact clock times are a stage-1/owner-approved cron
decision, not fixed here). Each run queries fresh live data — it does not
replay the morning's candidate list. The **daily entry budget (5) is
shared across both runs, not five per run**: before drawing, the afternoon
run reads what that persona already committed today and (a) folds those
entries' players into the similarity-decay "already chosen" set, so it
won't stack a near-duplicate parlay on a morning pick, and (b) draws only
up to the remaining budget. For the Kelly-growth persona specifically, the
afternoon run re-solves `joint_kelly_portfolio` using the *remaining*
Kelly-fraction budget (original fraction minus what morning's entries
already allocated) — true portfolio accounting across the whole day, not
two independent half-days. This supersedes the original draft's "one
decision snapshot per day" language; §6 stage 1 picks up the matching
update when that stage is built. The cadence and shared daily budget apply
independently within each of the 40 replicates — a replicate's afternoon
run only checks dissimilarity against, and shares budget with, its *own*
morning run, never another replicate's.

**Ensemble sizing (Monte Carlo replicates).** Each persona runs as a
**40-replicate ensemble** (`LEDGER_REPLICATES = 40`; 120 replicate-ledgers
total), not the single realized draw the first cut of this policy
specified. A `research-analyst` review
(`docs/archive/researcher_sim_bettor_mc_ensemble.md`, 2026-07-12) found
that framing this as a finite ensemble of a stochastic selection policy —
not "more Monte Carlo is safer" — puts the knee of the ensemble-mean
convergence curve near `M ≈ 40` (proper-score convergence goes as
`1 + 1/M`; 40→100 replicates buys under 1%, invisible against real-world
ROI noise at this bankroll/entry-cap scale), while a 5th/95th-percentile
**range** off any live-forward ensemble this size would need 2,000+
replicates to stabilize and would just reintroduce noisy-path jitter one
level up. Two consequences:

- **Replicate 0 is the canonical pre-registered ledger** — seeded exactly
  as § RNG/idempotency specifies, and the *only* replicate that drives the
  stage-3 circuit breaker and the stage-4 hindsight-proofing tests.
  Replicates 1–39 exist solely to quantify selection-variance around it;
  the breaker must never read the ensemble mean (a fragile policy's
  occasional bad draw would wash out in an average that never lives
  replicate 0's actual bad week).
- **Reporting is mean + central-spread, never tail percentiles.** Stage-3
  analytics report each persona's ensemble-mean ROI/CLV/bankroll as the
  headline skill estimate, plus a ±1 SE band and the replicate
  distribution's IQR (25th–75th) as the selection-variance envelope —
  explicitly *not* 5/95 or min/max, which are unstable at `M = 40`. The
  band must be labeled "spread across equivalent policy draws," not "range
  of outcomes" — it is selection-algorithm noise, not uncertainty about
  whether the system makes money. The full tail-distribution question is
  deferred to a retrospective, `profit_sim.py`-style large-`N`
  characterization run on the *accumulated real ledger* once enough has
  settled — that is what `profit_sim.py` already exists for (§1); the live
  ledger stays lean.

Within a given `(date, run_slot, replicate_id)`, all three personas draw
from **one shared random stream, consumed in a fixed order** (Safe, then
High-EV, then Kelly-growth) — common random numbers, so the day's shared
slate noise cancels out of the cross-persona *comparison*, which is the
lane's actual deliverable. Different `replicate_id`s get independent
streams (§ RNG/idempotency). This trades a small, bias-free coupling
between personas for a real variance reduction on the contrast between
them — flag to the owner if "three independent personas" (as phrased
above) should instead mean fully independent randomness; the tradeoff is
standard CRN practice, not a free lunch. A one-time knee-check (stage 3,
after ~2 weeks live: plot ensemble-mean ROI vs `M ∈ {10,20,40,80,160}` per
persona) validates or revises `LEDGER_REPLICATES = 40` via `policy_v2` —
not a recurring job.

**EV floor.** Unchanged: reuse `PickemConfig.min_ev = 0.05`, enforced
upstream in `construct_entries`; no second ledger-only threshold.

**RNG / idempotency.** Each replicate's draw is seeded deterministically
from `(date, run_slot, replicate_id)` — shared across all three personas
within that tuple (the CRN mechanic above), not a single fixed constant
like `profit_sim.py`'s backtest seed (`42`). Streams are derived via
`numpy.random.default_rng(...).spawn(...)`-style independent spawning, one
child per `replicate_id`, never seed-plus-`replicate_id` arithmetic (which
can silently correlate streams). A retry of an already-simulated run
reproduces the same draw instead of drifting, preserving the
append-only/no-backfill requirement (§4 locked decision) even under a job
retry.

**Schema additions stage 1 will need** (beyond what §6 stage 1 already
lists — legs, lines, model probs, book devig, stake, policy version, git
SHA, `committed_at`): `persona` (`safe` / `high_ev` / `kelly_growth`),
`run_slot` (`morning` / `afternoon`), `replicate_id` (int, 0–39; 0 is
canonical — § Ensemble sizing), `game_span` (int; see Cross-game above).
These feed the stage-3 analytics split (same-game vs. cross-game profit,
per-persona ROI/CLV, ensemble mean/spread) — not built this session, only
tagged at commit time so it's available later. Stage 2's settlement must
resolve the *union of distinct legs* once per day and map outcomes onto
every replicate entry that cites them, never resolve leg-by-leg per entry
— verified against `clv.py`'s already-group-invariant fill and
`analysis.py`'s per-(player, date) resolution, this keeps nightly
settlement cost flat in replicate count (~3,000 entries/day at `M = 40` is
sub-second additional work, not a bottleneck).

**Config shape for stage 1** (illustrative — not yet wired):
```
shared = PickemConfig(
    entry_sizes=(2, 3, 4, 5, 6),
    contest_variants=("power", "flex"),
    min_ev=0.05,
    kelly_fraction=0.25,            # Safe/High-EV personas; Kelly-growth uses joint_kelly_portfolio directly
    max_stake_pct_bankroll=0.005,
    top_k=100,
    max_overlap=2,                  # coarse backstop only; the Jaccard draw is the real dissimilarity control
)
PERSONAS = ("safe", "high_ev", "kelly_growth")
bankroll_per_replicate = Decimal("5000")
max_entries_per_day = 5             # shared across the morning + afternoon runs, per replicate
RUN_SLOTS = ("morning", "afternoon")
LEDGER_REPLICATES = 40              # ensemble-mean knee (1+1/M law); replicate 0 = canonical path
# per (date, run_slot, replicate_id): one shared RNG stream (CRN across personas, fixed order)
# per persona, per replicate: fresh candidates -> partition filter -> cross-game module (independence-assumed)
#          -> persona priority score -> Jaccard-decay weighted draw
#          -> Kelly-growth only: joint_kelly_portfolio over the draw, remaining-budget-aware
# replicate 0 drives the circuit breaker + hindsight tests; 1-39 are the variance envelope only
```

### Policy v2

Policy v1 with the changes below; whatever v1 states and this section does
not replace still holds. The commit path stamps every new record
`policy_v2`.

**Payouts.** Candidates are priced, and entries settled, on the platform's
live payout table, and no Underdog same-game pair modifier exceeds 1.0
([`underdog_api.md`](../underdog_api.md) §6.8 holds the table, the
modifiers and the payout formula). A settled entry pays its table tier for
the real outcome, times the pick multipliers the platform quotes that tier
on, times the entry's pair modifier. One function,
`prediction.payouts.outcome_payouts`, applies that rule to the pricers'
sampled outcomes and to settlement's real one, so an entry is scored the
way it was priced; its docstring carries the push and refund cases.
Sleeper pays its own curve times the product of its posted multipliers.
**Scar:** a Sleeper Flex entry settles on the stand-in table
(`payout_curve_for("Sleeper", "flex")`), which has no loss tiers; the
per-entry curve the search prices it on is not on the record.

**Records.** Each `canonical_legs` entry's `boost` is the platform's raw
multiplier for that pick, on the same-game and the cross-game path alike,
and the record adds `pair_modifier`: the product of the entry's same-game
pair modifiers as the search priced it, 1.0 for a cross-game entry.

**Cross-game candidates.** Priced on each pick's raw multiplier, pick by
pick rather than as one fused product, so their Flex loss tiers follow the
settlement rule. v1 priced an Underdog pick on its multiplier scaled by
the per-pick baseline, which overstated every cross-game Underdog EV. A
side the platform does not post is never a leg. A cross-game candidate is
sized, ranked and recorded as a same-game one: `payout_multiplier` is what
the entry pays when every pick hits (table tier × pick multipliers) and
`joint_prob` the win probability that payout makes worth the priced EV.
v1 sized on the bare table tier and the all-hit probability, blind to the
pick multipliers and the Flex loss tiers: every Sleeper cross-game entry
and nearly every Flex one sized to $0, and a $0 stake ranks first for the
EV-per-dollar personas.

**Personas.** A fourth, *Even-picks* (`even_picks`), takes only Underdog
entries whose every pick carries a raw multiplier of exactly 1.00 and
ranks them as High-EV does. Its pool is built beside the shared universe,
by the same builders under the same shared config, from Underdog's even
picks alone: the all-even parlays of the same-game search, and cross-game
combos over the even-pick offers. The shared universe filtered to even
picks would hold only what the `top_k` cut and the cross-game beam keep
after ranking those picks against every other leg. Building the pool
leaves the shared universe as it is, and entry sizes, the Power/Flex
split, the daily budget and same-game stake sizing are those of v1. Even-picks
draws last on each replicate's shared stream, so for a given
`(date, run_slot, replicate_id)` the first three personas draw exactly
what they would without it. **Scar:** the pool is thin on many days all
the same: the same-game search's EV floors and the shared `min_ev` bind
on even picks before any cut does.

**Versions stay apart.** `policy_v1` records stay in the store as
committed and settle as they always have: on the bare tier of the live
table, with no pick multiplier and no pair modifier. Settlement never
reads a v1 record's `boost` (on a v1 cross-game Underdog leg it is the raw
multiplier scaled by the per-pick baseline). Every settled row carries its
`policy_version`, and `bankroll.parquet` keeps one trajectory per
`(policy_version, persona, replicate_id)`, so a v2 replicate starts from
the $5,000 seed instead of continuing a v1 balance. Stage-3 analytics
report the versions apart, never pooled (§4).

**Store scar (v1 to v3).** `_ledger_store.append_entries` refuses a
record whose `id` is already in the day's file, whichever bettor wrote
it. Replicate 0's morning draw is therefore whole, and every later draw
(replicates 1–39, the afternoon run) keeps only the entries no bettor
committed earlier that day: the store holds close to one replicate plus a
de-duplicated union, not 40 independent ledgers. Keying the check on
`(id, persona, replicate_id)` restores the ensemble and is a new policy
version (§4); a same-slot retry then needs its own guard, because a
second draw sees the first one's budget and draws differently.

### Policy v3

Policy v2 with the Even-picks bettor rebuilt and one key fixed; whatever
v1 and v2 state and this section does not replace still holds. The commit
path stamps every new record `policy_v3`. No `policy_v2` record exists:
v2 committed nothing before it was replaced (2026-10-05).

**Even-picks.** This replaces v2's paragraph on the bettor. It takes the
Underdog even picks the product recommends for games on the slate date:
a side whose raw multiplier is exactly 1.00 and whose edge at the
per-pick payout clears the Receipts floor (`realized.recommended`: served
`Win Prob` × 1.83 − 1 ≥ 0.05, a read of .574 or more). Its legs skip the
shared leg gate and the trust haircut and are priced, sized and recorded
at the served read, because the bettor exists to put the recommendation
rule on paper as real entries (honest-receipts decision 12). Entries are
dealt evenly: up to 200 at each size from 2 to 6, each a random set of
the day's legs with a game per leg, so the table payout is exact and no
leg carries the pool. The pool holds no same-game entry. Pricing on the
pooled curve, the `min_ev` floor, quarter Kelly under the 0.5 % cap, the
High-EV ranking and the daily budget of five entries are v2's cross-game
path.

**Why v2's pool was empty.** v2 built the pool behind the shared leg
gate (`filter_legs`: read and book both above .5 and within .04 of each
other). A recommended leg has the model well away from the book. On
production's morning board of 2026-10-05, 25 of the 155 even picks were
recommended and none of the 25 passed the gate; the 12 even picks that
did pass priced a best entry of +0.8 % after the haircut, against the
+5 % floor. Replayed on that board, v3's pool holds 99 candidates (25
two-pick, 38 three-pick, 28 four-pick, 8 five-pick) over the day's 8
recommended even picks in 5 games: every entry those legs allow. A first
cut that reused the shared beam (the 200 entries with the highest joint
read at each size, legs of any game date) put one leg in 53 % of its
candidates and would have re-entered a Sunday leg on every commit day of
the week; hence the even deal and the game-day rule.

**Scars.** Measured by running the commit path on two boards production
saved on 2026-10-05 (99 and 735 candidates), into a scratch store. (1)
The draw still ranks as High-EV does, so it leans inside the pool: the
stronger half of the legs held 55 % and 59 % of the leg slots, and on
the larger board the four-pick Flex took 45 % of the 104 records against
27 % of the candidates, the six-pick one record against 15 %. In the five
weeks before the bettor started the strongest reads hit no better than
the rest (honest-receipts §7). (2) A slate needs recommended even
picks in two games to give an entry and in six to give a six-pick, and a
game that starts before the morning run (14:05 UTC) is missed. (3) The
pool is scored at commit time, while Receipts grades the stored snapshot,
which for half of a game-day board is an earlier one (honest-receipts §8
ask 10): until that is fixed the two recommended sets differ. (4) The
store scar above costs more on a small pool: from the 99 candidates the
store kept 58 records where forty independent copies would hold about
122, and ten copies held none.

**Cross-game leg keys.** A cross-game leg's `stat`, and the shared
path's trust lookup, use the league's own market key
(`helpers.archive_market`). An NHL Assists leg carries `assists`, the
gamelog's column; v1 and v2 wrote `AST`, which the gamelog lacks and
settlement would have read as a push. NHL Points moves the same way
(`PTS` to `points`), and NBA and WNBA Underdog fantasy points move to
the model cell's key, a column that holds the same number. No stored
record holds an NHL leg.

**The three shared bettors are unchanged but for that key, and
silent**; their last entries are dated 2026-09-20. On the 2026-10-05
board 598 of the 1,267 Underdog rows passed the leg gate, at a mean pick
multiplier of 0.71 and 2 % even picks: discounted favorites. The trust
figure (`resolve_market_shrinkage`) was 0 for 81 % of them, so a mean read of
.681 became .505, and the best two-pick the beam kept priced at −24.9 %
(−17.5 % at the served read; Sleeper −19.3 % and −12.5 %). Across the 70
served cells the figure is 0 in 34, at most .05 in 27 more and above .3
in four (NFL receptions .475, NFL rushing yards .455, NHL goals .361,
NHL skater fantasy points .331). Training skill is above 0 in 40 of the
70, and the live rung sets 24 of those to 0. That rung is
`clip(2 × (share of closed legs that beat the close − .5))`; a line that
did not move counts as not beaten (20 % of 498,882 closed rows, against
39.5 % moved for the bet side and 40.1 % against), and the close it
reads is stamped before the game on most rows (honest-receipts §8
ask 10).

## 11. Ledger (append-only, newest first, cap ~15)

- 2026-10-05 · policy_v3 (`6adcfae8`) · Even-picks rebuilt on the
  recommendation rule at the served read: game-day legs, entries dealt
  evenly, a game per leg · v2's pool sat behind the shared leg gate,
  which no recommended leg passes (0 of 25 on the 2026-10-05 board), and
  v2 committed nothing · cross-game `stat` and trust lookup on the
  league's own key · shared bettors unchanged, silent since 2026-09-20
  (trust figure 0 in 34 of 70 served cells) · replay on that board: 99
  candidates · detail §10 Policy v3 · next: first production runs, then
  the four-week read
- 2026-10-05 · policy_v2 built (local, unpushed) · settle = table × pick
  multipliers × pair modifier, one rule shared with pricers; record
  +`pair_modifier`; cross-game priced per pick on raw multipliers; persona
  `even_picks` (dev data, 53 days with parlay history: own pool under 5
  candidates on 23, shared universe's even entries under 5 on 27);
  cross-game stake on the all-hit payout and the priced EV; bankroll per
  version; v1 keeps bare table · store scar found (replicate 0 holds 223
  of 341 live records) · detail §10 Policy v2 · next: owner review + push
- 2026-07-14 · cross-lane note (sleeper-parity stage 4, landed on
  `feature/sleeper-parity`, not yet merged `devel`) · Sleeper candidates now
  flow through this ledger: `build_candidate_universe`/`run_commit` draw both
  platforms from one shared pool/budget per persona — no bankroll-schema
  change needed (`_ledger_bankroll.py`'s `(persona, replicate_id)` grouping
  was already platform-agnostic), but the committed-record schema gained an
  additive `platform` field (pre-existing records on disk have none;
  `settle_entry` reads it via `.get("platform", "Underdog")`, never
  `record["platform"]`, since settlement never rewrites old append-only
  entries) · fixed two latent push-refund pricing bugs on the Sleeper path
  found while wiring this in (cross-game combo pricing, settlement's
  `realized_multiplier`) — detail in `sleeper-parity.md` §10 · covered by new
  `tests/golden/test_ledger_cross_platform.py` +
  `tests/integration/test_ledger_fake_mode.py` · next: unchanged — this
  lane's own stage 1+2 still await owner review + push, then stage 3
- 2026-07-12 · stage 2 settlement built (local, unpushed): additive
  prerequisite patch first — entries stored legs as rendered display
  strings, not this codebase's canonical structured-leg schema
  (`leg_schema.py`; `leg_label`'s own docstring: "never parsed back").
  Threaded `canonical_legs` (the `build_leg`-shaped dicts already sitting
  on the parlay search DataFrame, previously dropped) through
  `RecommendedEntry` → `LedgerCandidate` → the JSONL record, and fixed a
  latent bug where `_ledger_cross_game.py` left `stat` as the raw platform
  market name instead of the canonical key. Free to amend — zero live
  entries existed anywhere on disk. Settlement itself: 2 new modules
  (`_ledger_settlement.py` — dedup/resolve/CLV-join/payout,
  `_ledger_bankroll.py` — parquet persistence + compounding bankroll), a
  thin `nightly.py:reflect` hook (`_resolve_ledger`), `_gameday_rows_for`
  relocated `nightly.py` → `analysis.py` (removes a backwards
  `strategies/ → nightly` import). Resolves the day's distinct legs once
  and broadcasts outcomes across every citing entry, not per-entry, per
  §10; CLV joined via one `clv.fill_from_archive` call over the
  deduplicated set, with `Dist`/`CV`/`Gate`/`Step` bridged from
  `history.parquet` (the likely source of the "≥90%, not 100%" acceptance
  bar). `README.md` §Recommended Cron gained the `ledger-commit` lines
  stage 1 never added. refactoring-specialist + all 3 gates green (ruff,
  3549/3550 golden [same pre-existing dashboard-render flake noted in the
  stage-1 entry below], 24/24 integration) · next: owner review + push,
  then stage 3 (analytics + dashboard, circuit breaker)
- 2026-07-12 · stage 1 commit path built (local, unpushed): 4 new modules
  (`_ledger_store.py` JSONL append+idempotency, `_ledger_selection.py`
  `LedgerCandidate`+RNG+Jaccard-draw+personas, `_ledger_cross_game.py`
  independence-assumed cross-game builder pricing via
  `prediction.payouts.expected_payout_with_pushes` not a hand-rolled
  Poisson-binomial, `strategies/ledger.py` orchestrator) + 4 helpers in
  `underdog_pickem.py` promoted public for reuse; one live scrape shared
  across same-game+cross-game paths; `ledger-commit` CLI wired into
  `pyproject.toml`+`run_job.sh` (crontab lines proposed only, NOT
  activated — owner-gated per §8); refactoring-specialist + all 3 gates
  green (ruff, 3516/3517 golden [1 pre-existing unrelated dashboard-render
  flake], 24/24 integration) · next: owner review + push, then stage 2
  settlement (`nightly.py:reflect` extension)
- 2026-07-12 · stage 0 refinement · research-analyst (Opus) verdict on MC
  ensemble sizing folded into policy v1: 40 replicates/persona (120 total,
  not the originally floated 300) at the ensemble-mean convergence knee;
  replicate 0 canonical (drives breaker + hindsight tests), 1–39 variance
  envelope only; common random numbers across personas per replicate;
  mean + IQR/SE reporting, never 5/95 percentiles; tail-distribution
  question deferred to a retrospective `profit_sim.py`-style study on
  accumulated real data · brief:
  `docs/archive/researcher_sim_bettor_mc_ensemble.md` · next: stage 1
  commit path unchanged in shape, sized for 120 replicate-ledgers
- 2026-07-12 · stage 0 · policy v1 drafted + owner-approved via plan review
  (D6 resolved): 3 personas (safe / high-ev / kelly-growth) over a shared
  candidate universe, Jaccard-similarity-weighted MC selection replacing
  strict top-N, twice-daily (morning + afternoon) cadence sharing one daily
  budget, cross-game/cross-league via a new independence-assumed module (no
  correlation.py/parlay.py edits) · next: stage 1 commit path
  (`strategies/ledger.py`, `data/ledger/entries/{date}.jsonl` append,
  `run_job.sh` two-slot cron wiring — owner approves crontab lines)

- 2026-06-10 · created · brief drafted from roadmap-v3 migration · next: stage 0 policy spec v1 for D6
