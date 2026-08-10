# Sportstradamus Improvement Roadmap (v2)

> **ARCHIVED — superseded by [`../sportstradamus_roadmap_v3.md`](../sportstradamus_roadmap_v3.md)
> (swimlane index + per-workstream briefs in `docs/handoffs/`).** Status claims
> below are stale and non-normative. Retained for the *original design (for
> reference)* blocks and phase acceptance specs that the v3 lanes mine
> (ledger §2.2/2.3 schema, streaks §3.5, alerts §4.1, draft products §5.1–5.6,
> model refinements §6.x).

A six-phase plan for evolving `tjjerome/Sportstradamus` from a sophisticated modeling
factory with partial decision-engine coverage into a complete Underdog edge system. The
prior roadmap mis-assumed several production features were missing; they exist (`training/correlate.py`,
`prediction/correlation.py:beam_search_parlays` and `find_correlation`, `nightly.py:reflect`,
`helpers/distributions.no_vig_odds`). The deprecated items in `src/deprecated/` are *older
replaced implementations*, not gaps.

The current headline work is a **model-correctness-and-breadth track** prompted by defects
found in the live implementation. Phase 1 is therefore audit-and-fix, not revive-missing-code;
genuinely missing features (Kelly, CLV, contest-variant handling, model versioning) moved into
later phases — most of which have since landed.

---

## Table of Contents

- [Status at a glance](#status-at-a-glance)
- [Active Track — Model Correctness & Market Breadth](#active-track--model-correctness--market-breadth-in-progress)
- [Tools and CLIs Built](#tools-and-clis-built)
- [Standing Rules (every session)](#standing-rules-every-session)
- [Phase 1 — Audit and Strengthen the Foundation](#phase-1--audit-and-strengthen-the-foundation)
- [Phase 2 — Bet Logger and CLV Tracker](#phase-2--bet-logger-and-clv-tracker)
- [Phase 3 — Underdog-Specific Decision Engine](#phase-3--underdog-specific-decision-engine)
- [Phase 4 — Real-Time Alerts and Dashboard Extensions](#phase-4--real-time-alerts-and-dashboard-extensions)
- [Phase 5 — Best Ball and Battle Royale](#phase-5--best-ball-and-battle-royale)
- [Phase 6 — Modeling Refinements](#phase-6--modeling-refinements-ongoing)
- [Critical Path](#critical-path-post-2026-05-08-audit-scope-reduced)
- [Decisions & Trade-offs](#decisions--trade-offs)
- [Suggestions for Further Improvement](#suggestions-for-further-improvement-beyond-the-roadmap)

---

## Status at a glance

Status markers are inlined per sub-phase below; this is the executive summary. Since the
2026-05-08 audit, the headline change is the model-correctness-and-breadth track, now the
project's **active** work (Phases 1–6 keep their original numbers).

**ACTIVE — Model Correctness & Market Breadth (IN PROGRESS).** The lead work; home of record
[`docs/operation_ship_75.md`](../operation_ship_75.md). Prompted by defects in the live
implementation (see [Active Track](#active-track--model-correctness--market-breadth-in-progress)),
not optional polish. **Goal: ≥ 75% of markets per league carry a set baseline** (NBA ≥ 16/21,
WNBA ≥ 14/18, NFL ≥ 15/20). The live count, the lever stack, and the next step all live in
[`operation_ship_75.md`](../operation_ship_75.md); this roadmap does not restate them — they move
on every ship.

| Phase | State | Notes |
|---|---|---|
| 1 — Audit & Strengthen Foundation | ~80% | 1.1/1.4/1.5 ✅; 1.2 audit ✅ fixes ⚠️; 1.3 ⚠️ dropped; 1.6 ❌ |
| 2 — CLV Tracker | ✅ (in scope) | Retrospective CLV done; placed-bet logging out of scope |
| 3 — Underdog Decision Engine | Shipped (Power/Flex/Rivals) | 3.1/3.2/3.3 ✅; 3.4 REMOVED; 3.5 Rivals ✅, Streaks/Ladders deferred |
| 4 — Alerts & Dashboard | Deferred / scope-reduced | 4.1 deferred to end; 4.2 recs tab shipped, Open/Settled/Bankroll dropped |
| 5 — Best Ball / Battle Royale | ~15% (legacy shape) | Legacy `drafts/` pipeline only; roadmap modules unbuilt |
| 6 — Modeling Refinements | 0% | None started; home for deferred model tail |

**Off-roadmap landings:** `helpers/archive.py` migrated klepto → DuckDB (bulk-deduped flush);
`scripts/` migration scripts for archive/training-data/gamelogs/correlations/pickles → parquet;
`tests/golden/` covers correlate, parlay search, CLV explode, pipeline snapshots, CLI-help snapshots.

---

## Active Track — Model Correctness & Market Breadth (IN PROGRESS)

**This leads the remaining work**, ahead of deferred Phase 4 (alerts/dashboard) and Phase 5
(best ball). It is not a Phase-6 refinement: it was prompted by **defects discovered in the
live implementation**, making it production-quality *correction* work. Home of record:
[`docs/operation_ship_75.md`](../operation_ship_75.md) — this entry is a pointer +
status, not a duplicate of the stage detail.

**Discovered defects (why it is urgent):**

- **GBDT regression-toward-the-mean** — top-decile compression, a family-invariant leaf-averaging bias.
- **SkewNormal overconfidence** / level bias.
- **ZINB joint-fit catastrophic per-row blowups** — compression up to **~5357×**, predicted means
  up to **~1437** on BLK/OREB/PF/BLST — fixed by HurdleZINB.
- **Blanket ZINB label wrong for ≥ 13 markets** (underdispersed → should route to CMP).
- **NFL position-confound** that put the passing-yards training mean at **38 instead of 216** (and
  similar across passing/rushing markets).

**Goal:** ≥ 75% of markets per league carry a *set baseline* (per-league targets in
[Status at a glance](#status-at-a-glance)). The live count and per-league gaps move on every
ship — see [`operation_ship_75.md`](../operation_ship_75.md).

**Scope (pre-break, active):** reach 75% breadth (Tier-0 audit code → Stage B1.6 feature/bias
track), then the core depth methods expected to pay off — A2 (T3 tail-head), A3 (calibration
polish), B2 (routing + feature engineering), B3 (MZINB / marginalized-hurdle family build). The
post-break speculative tail (Stage A4 / B4 / long-shots) is **deferred into Phase 6**. Follow the
track until diminishing returns, then stop it per-cell.

**Lever stack and next step:** owned by [`operation_ship_75.md`](../operation_ship_75.md) — §5
(lever stack), §6 (per-league path), §7 (stop-the-track principle). This roadmap points rather
than restates, because it moves on every ship.

---

## Tools and CLIs Built

Infrastructure hardened during the model-correctness track. Most diagnostics are **dev-only** —
they stay off `devel` per the CONTRIBUTING.md "Shipping to Production (`devel`)" denylist; the
live-metrics + graduation tooling is production runtime.

**Offline A/B + verdict**

- `scorecard` (`src/sportstradamus/training/scorecard.py`) — offline A/B harness
  over cached `data/test_sets/`; `--baseline` / `--candidate` / `--live-window N`. Its
  `compute_gates` returns the per-cell five-gate scorecard (g1–g5 + `ship`); `report()`
  inline-calls it on every `meditate` run, so the gate logic is production — only the standalone
  A/B CLI is a dev exercise. Thresholds mirrored in [`docs/ship_gate.md`](../ship_gate.md).

**Live metrics + lifecycle (production runtime)**

- `check-graduation` (`scripts/check_graduation.py`) — joins Gate 1 (`data/model_stats.parquet`)
  × Gate 2 (`data/live_metrics_per_market.parquet`, 30-day window); prints per-(league, market)
  lifecycle state: not-shipped / in-test / graduated / demoted.
- `backfill-live-metrics` (`scripts/backfill_live_metrics.py`) — walks settled history backwards
  with `--days` / `--step`; idempotent day-precision dedup.

**Diagnostics (dev-only)**

- `icc-diagnostics` (`scripts/icc_diagnostics.py`) — ICC₁ per (league, market) → `data/icc/{LEAGUE}_icc.parquet`.
- `zinb-routing-diagnostics` (`scripts/zinb_routing_diagnostics.py`) — per-league × per-market
  dispersion routing → `data/zinb_routing/{LEAGUE}_diagnostics.parquet` (pulls `statsmodels`,
  denylisted off `devel`).

**`meditate` flags**

- `meditate --deterministic` — opt-in deterministic mode (debug-only, never publish): RNGs pinned,
  Optuna swapped for fixed params, writes redirected to `data/{test_sets,models}/deterministic/`.
- `meditate --market <name>` — per-market scoped training (added Stage B1).

**Persisted artifacts**

- `data/live_metrics_per_market.parquet` — live BSS, 10-column / 2-window schema (Gate 2).
- `data/model_stats.parquet` — offline training report (Gate 1).
- `data/icc/`, `data/zinb_routing/` — diagnostic parquets.

**Shipping mechanism**

- `data/ship_config.json` — per-cell strategy config, nested `{league: {market: strategy}}`; the
  single toggle a ship PR flips.
- `devel-ship-curator` agent (`.claude/agents/devel-ship-curator.md`) — carves per-market
  production-delta ship PRs to `devel`, enforcing the research-scaffolding denylist
  (`zinb_routing_diagnostics`, `icc_diagnostics`, `statsmodels`, `/tmp`
  harnesses). Never pushes; the human approves.
- [`docs/ship_gate.md`](../ship_gate.md) — human-readable mirror of the `scorecard` threshold
  constants. Update it whenever a threshold changes.

---

## Standing Rules (every session)

Each session prompt opens with "Read `CLAUDE.md` and `CONTRIBUTING.md` first" (strict repo rules:
no monoliths, no commented-out code, no orphan methods, golden tests must pass). Prompts are scoped
**one module per subagent** (multi-module work dispatches subagents in parallel; single-module work
stays in the main session), specify canonical import paths (`sportstradamus.training`,
`sportstradamus.prediction`, `sportstradamus.helpers`, `sportstradamus.stats`) so deleted shims
aren't recreated, and state acceptance criteria.

- **Always-on gate set — all must pass at session close:**
  ```bash
  poetry run ruff check src/sportstradamus/
  poetry run pytest tests/golden/
  poetry run pytest -m integration      # fake-mode, no network
  REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py   # when CLI flags change
  ```
- **`refactoring-specialist` subagent before any of five triggers** on Python edits: a `git push`,
  a PR create/update, replying "done", dispatching a code-review subagent, or asking the user for
  review. Mandated by `CLAUDE.md`; reviewers should not spend attention on style nits the specialist
  catches. Hand it the explicit list of `.py` files touched this session.
- **`research-analyst` subagent** (`subagent_type: "research-analyst"`) when a diagnostic result is
  ambiguous or a path-forward decision needs literature + stats synthesis. Read-only; writes a cited
  brief to `/tmp/researcher_{topic}.md`.
- **End-of-phase handoff via the `prompt-engineer` agent** (Definition of Done): produce the
  next-stage handoff prompt in the 10-section structure, written to `/tmp/{stage}_handoff_prompt.md`;
  on user acceptance, commit to `docs/handoffs/{stage}.md`.
- **Cross-league testing policy:** run a smoke phase (1–2 markets per league) first; only after
  smoke passes run full verification (all markets in all covered leagues for the affected
  distribution branch). A smoke regression is a hard stop.
- **Determinism gate before any cross-league A/B:**
  `poetry run pytest tests/integration/test_determinism_gate.py -v -m integration`.
- Scope to one module per subagent (parallel for multi-module), then commit.
- When adding a dependency, specify the Poetry group (core, `[bayes]`, `[strategy]`, `[alerts]`).
- Money values are always `Decimal`, never `float`.

---

## Phase 1 — Audit and Strengthen the Foundation

**Goal:** validate that the existing correlation/parlay/EV pipeline does what it claims, fix
methodology gaps, and add the observability + contest-variant features the live code is missing.
Each sub-phase is audit-then-surgery: the first session reads the code and writes findings; the
fix is gated on the audit.

### 1.1 Audit `training/correlate.py` — ✅ DONE (audit + fix shipped)

**Audit** at `docs/archive/CORRELATE_AUDIT.md`. **Fixes shipped:** 8-game rolling residualization
(`_residualize_gamelog`), stratified per-team matrices, `--rebuild-correlations` flag on `meditate`.

The audit answered, per quoted source line: inputs/time-window; raw vs residualized correlation +
time-decay; minimum-overlap threshold and low-overlap shrinkage; same-team vs opposing vs cross-game
stratification; CSV schema + metadata; callers; recency via `git log --follow`.

**Acceptance (fix):** residualization makes a shared-trend/independent-residual synthetic pair
correlate ~0; a 5-shared-game pair shrinks closer to 0 than the same relationship at 100 games;
metadata JSON written with date range / observation counts / timestamp / git SHA;
`--rebuild-correlations` runs without touching `data/models/`. Stratified CSVs
(`{LEAGUE}_corr_same_team.csv`, `_opposing.csv`, `_cross_game.csv`) consumed by
`prediction/correlation.py:find_correlation` (signature unchanged, matrix chosen internally).
Confined to `training/correlate.py`, `training/cli.py`, `prediction/correlation.py`, new tests; no
change to the LightGBMLSS pipeline; magic numbers → named constants (STYLE_GUIDE.md §9). Gates: ruff,
golden, CLI snapshots; `poetry run meditate --rebuild-correlations --league NBA` produces the CSVs.

### 1.2 Audit `find_correlation` + `beam_search_parlays` — ✅ AUDIT / ⚠️ FIXES PARTIAL

**Audit** at `docs/PARLAY_AUDIT.md`; calibration plot/CSV at `docs/archive/evidence/PARLAY_CALIBRATION_*`.
**Note:** `beam_search_parlays` was since split out into `prediction/parlay.py`
(`find_correlation` stays in `prediction/correlation.py`). **Shipped:** Gaussian copula, `contest_variant` parameter
(power/flex/insurance/rivals), `data/underdog_payouts.json`, push-aware EV via
`_expected_payout_with_pushes`, nearest-PSD Σ-repair via `parlay.py:_nearest_psd`.

The audit answered, per quoted source: `find_correlation`'s joint-probability formula (copula vs
Pearson product vs log-odds), same-player guarding, `banned_combos.json` usage; `beam_search_parlays`
beam width / configurability, partial-parlay scoring, constraints, payout-multiplier source +
per-variant parametrization, output ranking/dedup. The empirical calibration check
(`scripts/audit_parlay_calibration.py`) bins the last 90 days of system-produced candidates into
deciles by predicted joint probability, computes actual decile hit rates from
`Stats.{league}.gamelog`, and plots predicted-vs-actual.

**Acceptance (fix, gated on audit):** switch Pearson→Gaussian copula (50K draws, marginals from
the trained LightGBMLSS objects) behind a feature flag with a `--legacy-correlation` escape hatch
on `prophecize` for one release; add the `data/underdog_payouts.json` variants and the
`contest_variant` parameter (default `power`); add `min_pairwise_correlation` (default 0.10); push-aware
EV summing across the `2^k` over/push/under outcomes for integer-line legs; tighter decile
calibration after the change. Tests: copula vs hand-computed 3-leg joint within 1% at 50K;
no-correlated-pair parlays filtered; push-aware EV matches a hand calc; Power→Flex changes a 5-pick
EV correctly. Existing callers unbroken.

**Open audit findings (Phase 1.2.x follow-ups):**

- `find_correlation`'s pairwise EV `exp(C * sqrt(V_i V_j)) * p_i p_j` is not a probability and is
  unbounded — wrap/replace with a bounded score.
- Beam width 1000, EV cutoff 1.05, boost bands, final EV floors are inline magic numbers (STYLE_GUIDE.md §9).
- `data/banned_combos.json` is a soft modifier; no pair is hard-banned. Decide whether a hard-ban
  path is desired.
- Same-player guarding is fragile substring matching — switch to canonical `player_id` joins.
- The `Boost` column displayed downstream is overwritten with a *different* payout table than the
  one that drove ranking (`correlation.py:498`).
- `Sleeper`/`ParlayPlay`/`Chalkboard` payout tables are stub `1`s — fill them or gate Model EV on
  platform support.
- Dedup is by exact bet-id; overlap-aware dedup uses a 3-cluster Ward linkage but does not enforce
  one-per-family selection.
- The committed calibration plot is a placeholder (no `parlay_hist.dat`); rerun on a production host
  with archive data and commit the result.
- **Copula over calibrated marginals (deferred here from the Ship-75 audit).** Once Ship-75 lands
  well-calibrated marginal predictives, replace the Pearson/Gaussian-copula approximation with a copula
  fit on PIT-transformed historical residuals (`U_{i,t} = F̂_i(y_{i,t})`) within same-game leg groups
  (and per leg-type pair, e.g. QB pass-yds × WR rec-yds on the same offense): fit a Gaussian/t-copula,
  EB-shrink the per-pair correlations across teams, and price by sampling jointly then inverting through
  the marginals. Add a **dependence diagnostic** — average pairwise rank correlation of residual PITs
  within same-game groups vs the under-independence prediction. At parlay dimensions of 2–6 a Gaussian
  copula suffices; vines are overkill. This is the audit's single largest *product*-EV lever: the five
  gates are marginal-only, the product is parlays, and the DFS pick'em apps largely don't tax leg
  correlation — the asymmetry that makes them beatable. See [`operation_ship_75.md`](../operation_ship_75.md) §10.

### 1.3 Closing-line freeze — ⚠️ DROPPED (workaround sufficient)

No `Archive.freeze_close`, no `data/closing_lines/`, no `freeze-close` CLI. `clv.fill_from_archive`
treats the last archived sample before kickoff as the close, and the Odds API stops surfacing
prematch odds for in-progress games, so this is correct in practice. The original motivation was
the placed-bet logger (Phase 2.2, now out of scope); without that consumer the implicit close is
good enough. Revisit only if the CLV monitoring (2.4) shows the assumption breaking. See
[Decisions & Trade-offs](#decisions--trade-offs).

*Original design (for reference):* `Archive.freeze_close(event_id, freeze_ts)` snapshotting line
records to `data/closing_lines/{event_id}.json` (book / player_id / market / line / over+under odds /
devig); a cron-friendly idempotent `freeze-close` CLI freezing events whose `commence_time` just
passed, with a `.last_freeze_check` marker; `Archive.get_closing_line(...)`. Additive — no change to
`Archive.write` / `Archive.get_line`.

### 1.4 Structured logging — ✅ DONE

`src/sportstradamus/helpers/logging.py` is in place and used across CLI entry points (`reflect` log
lines in `nightly.py`). Provides `get_logger(name)` writing JSON lines to
`logs/{YYYY-MM-DD}/{cli_name}.jsonl` (RotatingFileHandler, 50MB) with a `JsonFormatter`
(`ts`/`level`/`module`/`message` + extras), WARNING/ERROR also to stderr, re-exported from
`helpers/__init__.py`. `print()` calls migrated to logger calls (except `tqdm` progress UI); a
`--log-level` flag on all five CLIs. Stdlib `logging` only (no `loguru`/`structlog`).

### 1.5 Integration tests — ✅ DONE

`tests/integration/test_end_to_end.py` plus fixtures. One end-to-end fixture-based test runs
`confer → meditate --league WNBA → prophecize` via a `--fixture-dir` flag in `moneylines.get_props`
(canned API responses, no live Odds API), trains on a small WNBA points fixture (~500 player-games),
mocks the Sheets export, and asserts EV computed for ≥10 offers and ≥1 parlay candidate. Runs <90s
under `pytest -m integration` (excluded from default `tests/golden/`). Confirm coverage matches this
spec when revisiting. No production path changed beyond the `--fixture-dir` branch.

### 1.6 Audit `src/deprecated/` — ❌ NOT DONE

No `docs/DEPRECATED_TRIAGE.md`. `src/deprecated/` still contains `correlation.py`, `opt_parlay.py`
(both superseded by live code), `opt_kelley_bet.py` (REVIVE candidate for Phase 3.1), and the rest
of the 2026-04 sweep. The triage decisions are still owed.

**Acceptance:** for each file and each README TODO, check whether a live replacement exists (per
CONTRIBUTING.md Package Map) and write a one-paragraph decision in `docs/DEPRECATED_TRIAGE.md` —
REVIVE (with target phase), DELETE (rationale), or ARCHIVE PERMANENTLY (rationale). For each DELETE,
remove the matching README TODO entry. Decision-only — no code changes or deletions yet (deletions
follow review). Report counts: REVIVE / DELETE / ARCHIVE.

---

## Phase 2 — Bet Logger and CLV Tracker

**Goal:** measure whether the model beats the close. ROI alone is too noisy at recreational scale;
closing-line value is the standard quant-betting skill-vs-variance metric, and the one diagnostic
`nightly.py:reflect` did not produce. This phase extends `reflect` rather than building a parallel
system.

### 2.1 Audit `nightly.py:reflect` — ✅ SUPERSEDED

Audit doc never produced because `clv.py` was built directly. `reflect` now (a) resolves history,
(b) calls `clv.fill_from_archive`, (c) prints `clv.summarize` + per-segment results honoring
`CLV_SEGMENT_MIN_N=20`. Adequate for this package's scope.

### 2.2 / 2.3 Placed-bet logger and per-bet CLV — ❌ OUT OF SCOPE

Tracking which entries the user actually placed is a personal-bookkeeping concern users own
themselves. The SQLAlchemy schema, `track place|close|settle` CLIs, and bet-history dashboard tabs
are intentionally not built (**dropped, not deferred** — see
[Decisions & Trade-offs](#decisions--trade-offs)).

*Original design (for reference):* a `tracking/` package with hand-written SQL migrations at
`data/tracking.db` (Entry + Leg tables holding stake/payout/model-prob/joint-prob/model-version/CLV/
push/void), `place_entry()` snapshotting model prob + sharp devig + model SHA at insertion,
`track place --from <yaml>`, plus `track close` (fetch closing line, de-vig sharp or
`book_weights.json`-weighted consensus, compute `clv`) and `track settle` (look up actual outcomes,
apply Underdog push/void rules, compute payout from `data/underdog_payouts.json`). All money `Decimal`.

### 2.4 CLV reporting — ✅ DONE (in scope); monitor

`clv.summarize` produces per-(League × Market × Platform) segment tables logged by `reflect`.
Bootstrap CIs and CSV/JSON exports are nice-to-haves, not blockers — defer until the existing summary
can't answer a real question.

**Monitor:**

- `frac_beat_close` near 50% on no-edge segments, >50% where the model is sharp. Persistent <50% on
  a segment is an archive/close-snapshot bug or a model bug.
- `Market CLV mean` vs `Model CLV mean` divergence on a market signals retrain / feature-set review.
- Segments hitting `CLV_SEGMENT_MIN_N=20` — never accumulating 20 legs means it isn't surfacing in
  recommendations or the threshold is too high for low-volume markets.
- The implicit "last archive sample before kickoff = close" assumption silently breaks if Odds API
  behavior changes. Add a sanity check that the close timestamp is within N minutes of `commence_time`
  and warn otherwise.

---

## Phase 3 — Underdog-Specific Decision Engine

**Goal:** turn EV signals into ranked Underdog entries with proper bankroll sizing across Underdog's
contest variants. Implementation log:
[docs/archive/PHASE_3_IMPLEMENTATION.md](PHASE_3_IMPLEMENTATION.md).

### 3.1 Kelly sizing module — ✅ DONE

`src/sportstradamus/strategies/kelly.py` (archive/PHASE_3_IMPLEMENTATION.md §5). The kwarg was renamed
`model_calibration` → `model_shrinkage` to match the migrated training-report schema;
`opt_kelley_bet.py` moved to `src/deprecated/.archived/`.

API: `fractional_kelly_stake(bankroll, win_prob, payout_multiplier, fraction=0.25, model_shrinkage=1.0,
max_fraction_of_bankroll=0.005) -> Decimal` (returns 0 if -EV) and
`joint_kelly_portfolio(bankroll, candidates, fraction=0.25) -> dict[str, Decimal]` (cvxpy SCS,
`[tool.poetry.group.strategy]`). Fractional Kelly default 0.25, capped per entry; effective-probability
shrinkage `0.5 + (win_prob - 0.5) * model_shrinkage` reading the calibration via
`training.report.get_market_calibration(league, market)`; all math `Decimal`. The `kelly` CLI loads
candidates from a recommendations YAML and prints (candidate, EV, stake). Acceptance: -EV → 0; +EV →
analytic fractional Kelly; cap enforced past 0.5%; two independent +EV bets give positive allocations
summing ≤ fraction × bankroll; shrinkage equivalence (0.6 @ 0.5 ≡ 0.55 @ 1.0).

### 3.2 Underdog contest-variant payouts — ✅ DONE

`data/underdog_payouts.json` covers power/flex/insurance/rivals; `beam_search_parlays(contest_variant=...)`
consumes it, and Flex's miss-one tier is integrated into push-aware EV. Verify multipliers against the
current Underdog product each major release.

The EV calc handles Flex's miss-one/miss-two branches via partial-hit probabilities from the same
Gaussian copula sampling (fraction of 50K samples where exactly *k* of *n* legs hit). Acceptance: Power
EV unchanged when `contest_variant='power'`; Flex 5-pick > Power 5-pick at ~55% leg probs (downside
protection helps marginal edges); Flex 5-pick < Power 5-pick at ~70% (higher multiplier wins when legs
are sharper). Default behavior unchanged for callers not passing `contest_variant`.

### 3.3 Underdog-native strategy module — ✅ DONE

`src/sportstradamus/strategies/underdog_pickem.py` plus the `pickem-build` CLI
(archive/PHASE_3_IMPLEMENTATION.md §6). Sheets export was deprecated on `devel`. `pickem-build`
writes `data/recommendations/{date}.yaml` (which `kelly` re-sizes offline); the dashboard's Pickem
page reads the recommendations from the `prophecize` snapshot (`data/runtime/current_pickem.parquet`),
not the YAML — see [§4.2](#42-dashboard-extensions--%E2%9A%A0%EF%B8%8F-scope-reduced). Bankroll is a
CLI flag only (no `data/bankroll.json`). Rivals is folded into the same orchestrator (2- and 3-leg
sizes only, both sides of the matchup required).

API: `PickemConfig` (edge/disagreement/correlation/EV thresholds, `entry_sizes`, `contest_variants`,
`top_k`, `max_overlap`, `kelly_fraction`, `max_stake_pct_bankroll`) and
`construct_entries(date, bankroll, config=None) -> list[RecommendedEntry]`. It loads today's offers
(same source as `prophecize`), filters to Underdog markets with model coverage + sharp devig + model/sharp
agreement within `disagreement_threshold` + both edge thresholds, calls `beam_search_parlays` once per
(entry_size, contest_variant), sizes each candidate via `fractional_kelly_stake`, and writes the YAML.
This module orchestrates only — math stays in `kelly.py`/`correlation.py`. Acceptance: per-threshold
filtering; disagreement threshold skips a divergent leg; both Power and Flex appear when enabled;
non-empty YAML on a small fixture.

### 3.4 Pick'em Champions — ❌ REMOVED

Dropped per archive/PHASE_3_IMPLEMENTATION.md §0. Pari-mutuel optimization is a different problem shape (you
play other users, not the house — static-line-vs-sharp arbitrage doesn't apply); revisit only after
Phase 3 has measurable CLV. See [Decisions & Trade-offs](#decisions--trade-offs).

*Original design (for reference):* `strategies/underdog_champions.py` optimizing **expected percentile
rank** in the field's score distribution — estimate `field_pick_rate` per leg (from Underdog's
popularity indicator, scraped via `books.get_ud()`, or an ADP-style ~58%-Higher fallback), simulate
100K opponent entries, score your candidate's percentile distribution, convert to pari-mutuel prize
equity via a `champions` curve in `data/underdog_payouts.json`. Separate from `underdog_pickem.py`.

### 3.5 Streaks / Ladders / Rivals — Rivals ✅ DONE; Streaks/Ladders ❌ DEFERRED

**Rivals** shipped folded into `pickem-build` (archive/PHASE_3_IMPLEMENTATION.md §6) — restricted to 2- and
3-leg sizes; both sides of the matchup must be covered for the same market or the entry is dropped with
a logged WARNING.

**Streaks** and **Ladders** are deferred — each warrants its own design pass (see
[Decisions & Trade-offs](#decisions--trade-offs)). Streaks is a sequential decision problem.

*Original design (for reference):* `strategies/underdog_streaks.py` with
`recommend_streak_action(current_streak, used_team_ids, available_offers, config) -> StreakRecommendation`
— continue iff the best available next-leg win probability exceeds a threshold derived from the
geometric payout ratios (`data/underdog_payouts.json` `streaks`) by a margin (default 2%); first two
picks from different teams; cash-out allowed at intermediate lengths; CLI
`streaks-recommend --streak 4 --used-teams KC,SF`. Ladders needs a "lowest-shared-rung" expected-payout
calculation.

---

## Phase 4 — Real-Time Alerts and Dashboard Extensions

**Goal:** push high-edge opportunities to your phone the moment they appear and extend the Streamlit
dashboard for review-and-act workflows.

### 4.1 Alerts package — ⚠️ DEFERRED to end of roadmap

Dropped from the active critical path. Line-update cadence today (60s+ polls, manual Underdog
observation) makes push alerts low-value: by the time a Telegram message arrives, the user has already
seen the line on the next dashboard refresh. Revisit if polling cadence drops below ~10s, a real
websocket feed lands, or the user reports missing high-edge windows. See
[Decisions & Trade-offs](#decisions--trade-offs).

*Original design (for reference):* `alerts/` (`rules.py`, `dispatcher.py`, `telegram.py`, `discord.py`)
with an `AlertEvent` / `AlertRule` Protocol / async-polling `Dispatcher`; rules
`HighEdgeAppearanceRule`, `InjuryNewsExposureRule`, `BankrollDrawdownRule`, `CLVDriftRule`,
`ScraperFailureRule`, `ModelStaleRule`; Telegram + Discord channels reading tokens from `creds/keys.json`;
24-hour in-memory dedup; foreground `alert-watch` CLI. Reads from cached files only — never calls into
`prediction/` mid-run.

### 4.2 Dashboard extensions — ⚠️ SCOPE-REDUCED

Baseline `dashboard.py`, `dashboard_app.py`, `dashboard_data.py`, and `pages/` exist. The Open Entries /
Settled History / Bankroll tabs are **dropped** (they depended on `tracking.db`). The **Today's
Recommendations** tab **shipped** as `pages/2_Predictions_Pickem.py`: it reads the `prophecize`
Pick'em snapshot (`data/runtime/current_pickem.parquet` via `load_current_pickem()`) and folds Kelly
sizing in through a bankroll slider (see [§3.3](#33-underdog-native-strategy-module--%E2%9C%85-done)). A
sibling `pages/2_Predictions_Parlays.py` covers beam-search parlays. See
[Decisions & Trade-offs](#decisions--trade-offs).

---

## Phase 5 — Best Ball and Battle Royale

**Goal:** play Underdog's draft products, which the existing system ignores. NFL-season-aligned: Best
Ball drafts run March–August, Battle Royale weekly during the regular season. The roadmap
effectively **replaces** the legacy `drafts/` package.

### 5.1 ADP ingestion — ❌ NOT DONE (legacy `update_ez_adp.py` exists)

`drafts/update_ez_adp.py` is a legacy ADP fetcher that does not match this spec (no parquet, no
stochastic ADP, no contest-aware output). Decide whether to retire-and-replace or evolve in place; the
target presupposes replacement.

**Acceptance:** `drafts/adp.py` ingesting Underdog ADP (HTML via `Scrape`) → FantasyLife fallback →
manual CSV, writing `data/adp/{contest_slug}/{YYYY-MM-DD}.parquet` with stochastic ADP (mean **and**
stdev, plus counts/source/timestamp); idempotent `drafts-adp-update --contest ...` CLI. Tests on a
captured HTML fixture (no live HTTP): right player count + ADP, manual-CSV round-trip, player-ID
resolution against `nfl_data_py` for 10 players. Leaf-dependency only.

### 5.2 Season-long projection distributions — ❌ NOT DONE

Legacy `drafts/forecast.py` exists but is structured around the older training pipeline;
`drafts/projections.py` is unbuilt.

**Acceptance:** `project_season(league='NFL', contest=..., n_seasons=10_000, seed=42) -> SeasonProjections`
— a translation layer (no new model) pulling marginals from existing LightGBMLSS pickles, converting to
fantasy points via the contest scoring system (`data/contests/{contest}.json`), sampling
`n_seasons × 17` weeks per player with week-to-week correlation modeled as AR(1) on residuals, caching
to `data/projections/{contest}_{date}.parquet`. Tests: deterministic with fixed seed; star-RB mean
weekly points within 5% of consensus (fixture); non-trivial AR(1) autocorrelation.

### 5.3 Battle Royale optimizer — ❌ NOT STARTED

Smaller scope than Best Ball — ship before advance equity.

**Acceptance:** `optimize_battle_royale(slate, ownership_estimates, n_field_pods=50_000,
n_my_rosters=100, seed=42) -> list[OptimizedRoster]` — project each player's weekly distribution
(`projections.project_week()`), simulate ADP-following field pods, generate candidate constructions via
heuristic templates, rank by **expected prize equity** (percentile → payout curve), mandatory stacking
(≥2 from one team). CLI `battle-royale-build --slate sunday-main`. Tests: deterministic; stacked
dominate non-stacked; top roster beats a random one. Performance: 50K pods × 100 candidates × 6 slots
in <60s (vectorized).

### 5.4 Best Ball advance equity — ❌ NOT STARTED

**Acceptance:** `expected_payout(roster, structure: TournamentStructure, n_simulations=10_000, seed=42)
-> AdvanceEquityResult` — sample independent full seasons from `drafts.projections`, build a stochastic
12-team pod (user + 11 ADP-sampled opponents), apply Underdog's weekly optimal-lineup rule, run the
14-week regular season (top 2 advance) → weeks 15/16/17 playoff structure, award prizes per
`payout_curve`, return mean + per-round advancement breakdowns. Pure simulation (no HTTP, no disk
writes). Tests: deterministic; league-average roster ≈ 1× entry (slightly under for 10–13% rake); elite
roster well over; round-1 advancement for a 50th-pct roster ~16.7%. Performance: 10K × 12 × 17 × 18 in
<60s (Numba-JIT the inner score loop).

### 5.5 Live-draft companion — ❌ NOT STARTED

**Acceptance:** `recommend(state: DraftState, k=5, candidate_pool_size=30, n_simulations=2_000)
-> list[Recommendation]` — take top `candidate_pool_size` available players by ADP, hypothetically add
each to the roster, score by resulting `advance_equity.expected_payout`, return top `k` with diagnostics
(stack synergy, weekly variance, late-season schedule strength) in <10s. Share the non-candidate sample
matrix and opponent simulation across evaluations (module-level lru_cache); only the candidate's weekly
contribution changes. CLI `draft-recommend --state path.yaml`. Tests: deterministic; recommendations in
the available pool; adding a top rec beats a random player; QB-stack candidate beats an equivalent
player on another team.

### 5.6 Portfolio exposure tracker — ❌ NOT STARTED

**Acceptance:** `compute_exposure(contest, entry_pool) -> ExposureReport` surfacing player concentration
(flag any player >35% of entries), stack concentration (flag any QB+top-WR combo >25%), archetype
distribution (% Hero-RB / Zero-RB / balanced), and leverage (user_exposure / field_exposure per player).
CLI `drafts-exposure --contest ...`. Tests: concentration flags fire; leverage matches hand math on a
fixture.

---

## Phase 6 — Modeling Refinements (Ongoing)

**Goal:** squeeze additional CLV out of edge cases and underserved markets. None required for
profitability; all upside.

Phase 6 is the home for **deferred** model work of two kinds: (1) the **speculative tail** of the
active model track — the post-diminishing-returns stages from
[`docs/operation_ship_75.md`](../operation_ship_75.md): **Stage A4** (novel risky retries),
**Stage B4** (tuning/polish — optional), and any long-shot method; plus (2) the **original refinements**
6.1–6.5 below. None of Phase 6 was ever the urgent work — the urgent work is fixing the discovered
defects and reaching 75% breadth (see [Active Track](#active-track--model-correctness--market-breadth-in-progress)).
Everything here is **deferred until breadth is met**; the A4/B4 stage detail is preserved in the
archived [`docs/archive/gbdt_mean_regression_plan.md`](gbdt_mean_regression_plan.md).

The full-distribution audit adds to this deferred tail (kind 1): **distributional conformal / CQR** for
the alt-line ladder (refines [§6.4](#64-conformal-prediction-wrappers--%E2%9D%8C-not-started)); a **CLV
CRPS-edge dashboard** (model vs the de-vigged *closing* distribution, per market, weekly); and two
backbone swings — **TabPFN-as-platform** and a **multi-task shared-trunk NN** pooling across
cells/leagues (the per-cell, small-n use of TabPFN stays in Ship-75 §5.7). The heaviest tail-head
rebuilds (spliced/Pareto, MZINB) also land here. The marginal-breadth levers from the same audit are
**not** deferred — they live in [`operation_ship_75.md`](../operation_ship_75.md) §5.

### 6.1 Bayesian hierarchical for low-sample players — ❌ NOT STARTED

`training/bayes_hier.py` — PyMC NegBin with player random effects partially pooled to position priors;
NBA points only this session. Output per-player posterior mean + std to
`data/bayes_predictions/NBA_points/{date}.parquet`; nightly `bayes-update --league NBA --market points`
CLI. `train_market` reads `bayes_mean`/`bayes_std` as optional features (NaN fallback; LightGBM handles
NaN). PyMC + ArviZ in `[tool.poetry.group.bayes]`. Tests: deterministic; r_hat < 1.05 on fixture; rookie
(5 games) closer to position prior than veteran (500 games); sampling <60s on fixture.

### 6.2 Monte Carlo NFL game simulator — ❌ NOT STARTED

`training/game_sim_nfl.py` — Vegas-anchored Monte Carlo. `simulate_game` returns
`dict[player_market, np.ndarray (n_sims,)]` for pass/rush/receiving markets: sample Vegas-anchored
game-flow → per-team drives → pass/rush attempts → Dirichlet target shares → per-player yards-per-target.
`joint_prob_from_sim(sim_output, predicates)` helper; `find_correlation` optionally uses the simulator
instead of the copula when sim outputs exist for all legs. Numba in core; calibrate via
`game-sim-calibrate-nfl` → `data/game_sim/nfl_coefficients.json`. Tests: deterministic; total points
within 0.5 of Vegas total across 10K sims; QB-pass-yds × WR1-rec-yds correlation > +0.5; 10K sims <500ms.

### 6.3 News and weather feeds — ❌ NOT STARTED

`feeds/news.py` + `feeds/weather.py`. News: poll Underdog feed every 60s (6am–1am ET), RotoWire RSS
backup, persist to `data/news/{date}.jsonl` (resolved player_id + severity), emit alert events at
severity ≥ high. Weather: OpenWeatherMap hourly forecast for upcoming outdoor NFL games →
`data/weather/{game_id}.json`, expose `weather_features(game_id)` (wind/temp/precip/is_dome) for
`Stats.NFL.get_stats()`. CLI `feeds-update`. OpenWeatherMap key in `keys.json`; news best-effort (log
WARNING, try next source, never crash the loop). Tests: RSS-fixture parsing; dome game → is_dome=True;
outdoor + wind >15mph → windy=True.

### 6.4 Conformal prediction wrappers — ❌ NOT STARTED

`training/conformal.py` — split conformal around LightGBMLSS marginals (wrapper only, no training
change). `fit_conformal(model, X_calib, y_calib, alpha=0.1) -> ConformalCalibration` and
`conformal_prob_over(model, x, line, calib) -> (point_estimate, half_width)` at (1−alpha) coverage. Add
a `kelly.py` option to size on the conformal interval lower bound (conservative under high uncertainty);
`meditate --fit-conformal` persists to `data/conformal/{LEAGUE}_{market}.pkl`. Tests: observed coverage
in [85%, 95%] at alpha=0.1; half-width monotonic in n_calibration.

**Audit extension (deferred from Ship-75):** for the full alt-line ladder the parlay builder prices, the
relevant tool is a *distributional* conformal predictive (Chernozhukov et al. 2021) or conformalized
quantile regression (Romano et al. 2019) over the whole CDF, not just `prob_over` at one line —
sequenced after Ship-75's marginals are calibrated. Conformal guarantees are marginal, not conditional,
unless DCP / multivalid conformal is used.

### 6.5 Push-aware EV refinement — ❌ NOT STARTED

Phase 1.2 added basic push handling; this models integer-valued markets (NFL TDs, NHL goals, NBA 3PM)
with discrete distributions explicitly. Audit every `data/config/stat_meta.json` market's support
(continuous/integer/hybrid) → `docs/MARKET_SUPPORT_AUDIT.md`. Refine `helpers/distributions.get_odds`:
exact `P(stat == line)` from the discrete PMF for integer support (`nbinom.pmf` for NegBin);
continuity correction `P(line-0.5 < stat < line+0.5)` for continuous distributions on integer-valued
markets. Verify improved calibration via the Phase 1.2 audit script. Tests: anytime-TD push prob
(line=1.0, mean=1.2) matches `nbinom.pmf(1, ...)`; Gamma-modeled TD continuity correction gives a
non-zero push probability.

---

## Critical Path (post-2026-05-08 audit, scope-reduced)

The model-correctness-and-breadth track now **leads** the critical path. The modeling, CLV, Kelly
(3.1), and Underdog-native (3.3) foundations all landed since the 2026-05-08 audit (1.1 + 1.2 + 2.x +
3.x). With placed-bet tracking out of scope, the remaining non-model work is narrow.

**Already landed:** Phase 3.1 Kelly (`strategies/kelly.py` + `kelly` CLI — a recommendation, not a
placed-bet trigger), Phase 3.3 Underdog-native (`pickem-build` → `data/recommendations/{date}.yaml`),
Phase 4.2 Today's Recommendations tab (`pages/2_Predictions_Pickem.py`).

**Open, in recommended order:**

1. **Model Correctness & Market Breadth — ACTIVE (lead).** Reach ≥ 75% baseline breadth per league,
   then follow until diminishing returns; the speculative tail (A4/B4) is deferred to Phase 6. Live
   count, lever stack, and next step: [`docs/operation_ship_75.md`](../operation_ship_75.md).
2. **Phase 1.2 follow-up — open audit findings.** Fix the `find_correlation` pairwise-EV unbounded
   score, replace inline magic numbers with named constants, swap substring same-player guarding for
   `player_id` joins, reconcile the `Boost`-column overwrite at `correlation.py:498` so the displayed
   multiplier matches the EV that ranked the parlay, and re-run `audit_parlay_calibration.py` on
   production archive data to commit a real plot.
3. **Phase 1.6 — deprecated triage.** `correlation.py` and `opt_parlay.py` in `src/deprecated/` can be
   deleted; `opt_kelley_bet.py` marked REVIVE. Cheap; unblocks future-agent confusion.

(Phase 4.1 alerts/push are deferred to the very end — see §4.1 for revisit triggers.) That sequence
finishes the package's mandate — ranked, sized, EV-positive Underdog recommendations with measurable
CLV — and then stops. Personal bet tracking is a downstream concern users own themselves.

Phase 1 audit work is undramatic but it is where most of the edge correction happens. With the
placed-bet logger dropped, only **3.1 + 3.3 + the 1.2 follow-ups** were the minimum-viable path to a
ranked-and-sized recommendation pipeline; everything else is multipliers on that core.

---

## Decisions & Trade-offs

Scope decisions consolidated from across the roadmap. Each links back to where the detail lives.

| Decision | Status | Rationale |
|---|---|---|
| **Placed-bet logger + `tracking/` package** (orig. 2.2/2.3) | **OUT OF SCOPE** (dropped, not deferred) | Tracking what the user actually placed is personal bookkeeping users own; package scope ends at retrospective system-recommended CLV. See [§2.2/2.3](#22--23-placed-bet-logger-and-per-bet-clv--%E2%9D%8C-out-of-scope). |
| **Dashboard Open Entries / Settled History / Bankroll tabs** (orig. 4.2) | DROPPED | Depended on `tracking.db`, which is out of scope. A Today's-Recommendations tab off the YAML remains. See [§4.2](#42-dashboard-extensions--%E2%9A%A0%EF%B8%8F-scope-reduced). |
| **Phase 1.3 closing-line freeze** | DROPPED (workaround sufficient) | Its only consumer was the placed-bet logger; `clv.fill_from_archive`'s implicit "last sample before kickoff = close" is correct in practice. See [§1.3](#13-closing-line-freeze--%E2%9A%A0%EF%B8%8F-dropped-workaround-sufficient). |
| **Phase 3.4 Pick'em Champions** | REMOVED | Pari-mutuel (peer-to-peer) is a different problem shape — percentile-rank, not static-line-vs-sharp arbitrage; revisit only after Phase 3 shows measurable CLV. See [§3.4](#34-pickem-champions--%E2%9D%8C-removed). |
| **Streaks / Ladders** (Phase 3.5) | DEFERRED | Each warrants its own design pass (Streaks is a sequential decision problem). Rivals shipped folded into `pickem-build`. See [§3.5](#35-streaks--ladders--rivals--rivals--done-streaksladders--%E2%9D%8C-deferred). |
| **Phase 4.1 alerts/push notifications** | DEFERRED to roadmap end | Line-update cadence (60s+ polls) makes push low-value vs the next dashboard refresh. Revisit if cadence <~10s, a websocket feed lands, or the user misses high-edge windows. See [§4.1](#41-alerts-package--%E2%9A%A0%EF%B8%8F-deferred-to-end-of-roadmap). |
| **Model speculative tail — Stage A4 / B4 / long-shots** | DEFERRED into Phase 6 | Post-diminishing-returns model work; not urgent vs reaching 75% breadth. A4/B4 detail preserved in the archived `docs/archive/gbdt_mean_regression_plan.md`. See [Phase 6](#phase-6--modeling-refinements-ongoing) + [Active Track](#active-track--model-correctness--market-breadth-in-progress). |
| **`prediction/parlay.py`** | SPLIT OUT (resolved) | `beam_search_parlays` now lives in `prediction/parlay.py`; `find_correlation` stays in `prediction/correlation.py` — matches the CONTRIBUTING.md package map. |
| **Bankroll** | CLI flag, not state | With the bet logger dropped, bankroll is a parameter to `kelly`/`pickem-build` — no `data/bankroll.json`, no DB row. See [§3.3](#33-underdog-native-strategy-module--%E2%9C%85-done). |
| **`clv.py` location** | Stays at package root | Would have moved into `tracking/`; without the bet logger, leaving it next to `nightly.py` is the right shape. |
| **Full-distribution audit** | Marginal-breadth levers → Ship-75 §5; parlay-dependence + conformal-ladder + backbone swings deferred here | Folded into [`operation_ship_75.md`](../operation_ship_75.md) §5 (four-axis: normalization / model-loss / blend / calibration); the non-marginal-breadth tail (copula §1.2; conformal / CLV-dashboard / TabPFN-platform / multi-task-NN Phase 6) defers to just after 75%. |

---

## Suggestions for Further Improvement (beyond the roadmap)

Items the audit surfaced that aren't in any phase but matter for long-term health.

**Modeling hygiene**

- **Closing-line bias check** — once a freeze exists (1.3), compare open-at-scrape vs close. Open moving
  *toward* the model = you're sharper than the open; close moving *away* = a model bug hiding behind a
  momentary mis-price.
- **Per-market shape-ratio dashboard** — `model_stats.parquet` exposes `shape_ratio` as a snapshot; a
  per-retrain historical CSV surfaces drift (1.0→1.3 over six months = creeping dispersion). Cheap add to
  `training/report.py`.
- **Parquet-everything migration** — confirm no production path still reads CSV/pickle; the archive is
  DuckDB but `data/{LEAGUE}_corr.csv` is still CSV (per archive/CORRELATE_AUDIT.md) — convert to parquet for
  read time + typed metadata.
- **Continuity-correction audit** — Phase 6.5's per-market support audit is a ~2-hour task that catches
  mis-applied push probabilities (anytime-TD, NHL goals, NBA 3PM); worth promoting to Phase 1.

**Engineering hygiene**

- **Magic-number purge in `correlation.py`** — the audit's list (`K=1000`, EV cutoffs, boost bands, the
  `[0, 0, 3.5, 6.5, 6, 10, 25]` payout array) deserves its own commit (STYLE_GUIDE.md §9).
- **`src/deprecated/` README cleanup** — the TODO list references replaced items (`correlation.py`,
  `opt_parlay.py`); Phase 1.6 fixes this, but until it lands the README misleads new agents.
- **`book_weights.json` provenance** — referenced in the prediction pipeline and Phase 2.3 but the source
  and refit cadence are undocumented; add a provenance note + `scripts/refit_book_weights.py`.

**New diagnostics worth building**

- **CLV-by-day-of-week × time-to-lock heatmap** — `clv.py` has the data; one plot reveals whether the
  edge is lock-window specific (early-week NFL lines carry more error than Sunday-morning).
- **Parlay-shape ablation** — a sister to `audit_parlay_calibration.py` that holds out individual shape
  constraints (min_correlation, max-per-game, beam_width) and reports calibration delta per ablation —
  tells you which knob matters before tuning.
- **Model-vs-market disagreement log** — when the copula joint estimate and per-leg sharp devig disagree
  beyond a threshold, log to `data/disagreements/{date}.jsonl`. A model-bug detector now; an edge surfacer later.

**Operational**

- **`pyproject.toml` script registration** — 5 scripts wired today (`prophecize`, `confer`, `meditate`,
  `dashboard`, `reflect`); register `kelly`, `pickem-build`, `alert-watch` in one batch with consistent
  verb-noun, dash-separated naming.
- **`creds/keys.json` schema doc** — `keys.json` is referenced in several places (Odds API, Telegram,
  Discord, OpenWeatherMap) with no canonical schema; add `docs/CREDS_SCHEMA.md` with example keys and
  which module reads each.

**Bigger swings (optional, multi-week)**

- **Beam-search-by-shape** — split by `(contest_variant, entry_size)` upstream and run separate beams so
  cutoffs tune per shape without cross-contamination. Cheap refactor, real win.
- **Conformal before Bayesian (swap 6.4 and 6.1)** — conformal wraps the existing pickles and gives a
  calibrated uncertainty knob immediately; PyMC adds a dependency, sampler, and infrastructure for value
  mostly on low-sample players. Conformal is the higher-EV-per-engineer-week swing.
- **Streamlit → FastAPI + Vue dashboard** — Streamlit's rerun model fights real-time push (Phase 4.1
  alerts feeding the dashboard); consider a one-weekend FastAPI + WebSocket prototype before investing
  more in Streamlit pages.
