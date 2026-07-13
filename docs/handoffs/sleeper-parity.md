# Sleeper decision-layer parity

> Status: QUEUED (entry: stage-1 Sleeper EV engine — stage 0 complete
> 2026-07-12) — CRITICAL PATH: blocks D3/parlay-dependence + dfs-products
> 2b/2c/5; recommended next code-heavy lane, target merge ~Aug 2026 (pre NFL
> Wk 1)

## 1. Mission & money logic

Bring Sleeper to full decision-layer parity with Underdog: payout-aware EV,
entry construction, Kelly sizing, recommendations YAML, and a dashboard view.
The observation and scoring layers already treat Sleeper as a first-class
platform; only the decision layer is Underdog-hardcoded.

A second app roughly doubles the slate surface at near-zero modeling cost:
models, scrape, and scoring are already platform-agnostic —
[`books.py:344`](../../src/sportstradamus/books.py) `get_sleeper` scrapes it,
[`prediction/cli.py:155-176`](../../src/sportstradamus/prediction/cli.py)
scores it via `process_offers`. Sleeper also prices each leg with a dynamic
per-leg multiplier (`payout_multiplier` → `Boost_Over`/`Boost_Under`,
[`books.py:339-340`](../../src/sportstradamus/books.py); alternates included,
[`books.py:328-329`](../../src/sportstradamus/books.py)) — pricing variance
Underdog's fixed tables do not have, hence more mispricings to harvest.

The load-bearing design fact: **the two apps have different engine shapes.**
Underdog pays from fixed tables keyed by (entry size, miss count)
([`underdog_payouts.json`](../../src/sportstradamus/data/config/underdog_payouts.json));
[`payouts.py:118`](../../src/sportstradamus/prediction/payouts.py)
`expected_payout_with_pushes` prices off that count-based curve. Sleeper's
payout depends on *which* legs hit (product of per-leg multipliers), so the
count-based curve does not generalize: Sleeper needs a per-leg-payout-vector
EV path — new code, not a config entry beside the legacy stub
`"Sleeper": [1.0, 1.0]` at [`payouts.py:84`](../../src/sportstradamus/prediction/payouts.py).

## 2. Read first (in order)

1. [`CLAUDE.md`](../../CLAUDE.md) §Hard rules + §MANDATORY refactoring-specialist — assumed-read law; this lane leans on the dashboard-parquet rule and the duplicate-code gate.
2. [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §Package Map + §Shipping to Production (`devel`) — canonical import paths and the devel-ship-curator PR mechanics.
3. [`sportstradamus_roadmap_v3.md`](../sportstradamus_roadmap_v3.md) §5 — cross-lane constraints: this lane lands **before** `parlay-dependence`; the `sim-bettor-ledger` schema preferably lands before this lane finishes.
4. [`books.py`](../../src/sportstradamus/books.py) — the Sleeper scrape (`get_sleeper`, `_sleeper_prop_offers`, the four `SLEEPER_*_URL` endpoints at :31-34). Read-only in this lane (§4).
5. [`prediction/payouts.py`](../../src/sportstradamus/prediction/payouts.py) — the payout engine to extend: `payout_curve_for`, `expected_payout_with_pushes` (post parlay.py seam-split; `beam_search_parlays` stays in [`parlay.py`](../../src/sportstradamus/prediction/parlay.py), joint pricing in [`joint.py`](../../src/sportstradamus/prediction/joint.py)).
6. [`prediction/correlation.py`](../../src/sportstradamus/prediction/correlation.py) — `find_correlation(offers, stats, platform, ...)` (:625), the platform plumb-through point.
7. [`strategies/underdog_pickem.py`](../../src/sportstradamus/strategies/underdog_pickem.py) + [`strategies/README.md`](../../src/sportstradamus/strategies/README.md) — the decision layer to generalize (`live_load` :336 calls `get_ud()` and hardcodes `"Underdog"` in the same line, :356).
8. [`archive/sportstradamus_roadmap_v2.md`](../archive/sportstradamus_roadmap_v2.md) — background only; status claims stale.

## 3. Verify before you trust

If command output contradicts brief prose, the output wins — fix the brief in
place (minor) or stop and ask the owner (material).

```bash
git fetch origin && git log --oneline origin/devel -3
# Decision layer still Underdog-only? (expect get_ud + "Underdog" together at ~:356)
grep -n 'get_ud()\|"Underdog"' src/sportstradamus/strategies/underdog_pickem.py
# Legacy Sleeper stub still in place? (expect '"Sleeper": [1.0, 1.0]' at ~:84, payouts.py not parlay.py)
grep -n '"Sleeper"' src/sportstradamus/prediction/payouts.py
# Picks-board shape unchanged? books.py parses: subject_id, wager_type, game_id, options[].outcome/outcome_value/payout_multiplier
curl -s https://api.sleeper.app/lines/available | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d), sorted({x['sport'] for x in d})); print(json.dumps(d[0], indent=2)[:800])"
poetry run pytest tests/golden/ -k "pickem or kelly or parlay" -q
```

### Volatile product assumptions

External facts this lane's math depends on. On drift: stop, re-verify all of
stage 0, revise this brief in place, then resume.

1. **Leg caps.** Verified 2026-07-12 (owner, in-app; corroborated by
   [Sleeper Support Center](https://support.sleeper.com/en/articles/9047931-sleeper-player-picks-rules):
   "2–8 players"): real app max is **8 legs**. The
   [`payouts.py:83`](../../src/sportstradamus/prediction/payouts.py) comment
   claiming a 3-leg cap is wrong — marked for correction, folded into stage
   1's `payouts.py` work rather than a standalone edit here (stage 0 is
   no-code). **New product decision** (owner, 2026-07-12): our construction
   caps recommended Sleeper entries at **6 legs**, split 2-3 = Max/power
   (all-or-nothing) and 4-6 = Flex (partial-hit-tolerant) — mirroring the
   existing [`_pooled_underdog_curve`/`POWER_MAX_SIZE`](../../src/sportstradamus/prediction/payouts.py)
   pattern (:19-45). This split is *our own* recommendation-construction
   convention on both platforms, not a platform rule — Sleeper and Underdog
   both let a user freely toggle Max/Flex-equivalent mode at any size in-app.
   **Still open:** the exact Sleeper Flex payout multiplier table (per size
   4/5/6 × miss-count) isn't published — checked
   [Sleeper's Flex-contest article](https://support.sleeper.com/en/articles/9261402-sleeper-player-picks-flex-contests)
   2026-07-12: the mechanic is confirmed (1-miss tolerance at 3+ picks,
   2-miss tolerance at 5+ picks) but no numeric table. See §3.1's test
   protocol — run before stage 1 implements the Flex pricing path.
2. **Multiplier composition rule.** Verified 2026-07-12 (owner test entry):
   entry payout = stake × product of the selected legs' `payout_multiplier`
   values; no promo stacking or entry-level cap found. Corroborated by
   Sleeper's own docs ("entry fee × VS Score" — their name for the composite
   multiplier).
3. **Push/void semantics per leg.** Verified 2026-07-12 (owner; corroborated
   by Sleeper Support Center): entries with 3+ picks — the void leg drops and
   the entry re-prices on the remaining legs, same rule as Underdog
   ([`payouts.py:118`](../../src/sportstradamus/prediction/payouts.py)
   `expected_payout_with_pushes`). **Divergence from Underdog:** 2-pick
   entries cancel and refund in full on any push, rather than dropping to a
   1-leg entry — stage 1's push-aware EV branch needs this as a
   platform-specific special case at the minimum size.
4. **Picks-board API shape.** Verified 2026-07-12 via the `curl` above: 246
   active lines returned; all expected fields present (`subject_id`,
   `wager_type`, `game_id`, `outcome`, `outcome_value`, `payout_multiplier`).
   Endpoints ([`books.py:31-34`](../../src/sportstradamus/books.py)) and
   field parsing (`_sleeper_prop_offers`, `books.py:303`) still match.
5. **Kill-criteria sanity check** (gates stage 0; not originally a numbered
   assumption). Verified 2026-07-12 (owner): Sleeper Picks prices as a
   fixed/quoted multiplier per leg, matching what the API returns — not
   pari-mutuel, not live-adjusting. Stage 0 is **not** blocked.

### 3.1 Sleeper Flex payout test protocol (new, carried into stage 1)

To populate the Flex payout table without risking real money: in the
Sleeper app's entry builder, **build but do not submit** one entry at each
size — 4, 5, and 6 legs. The builder shows the payout ladder per miss-count
before submission. Record, for each size, the multiplier shown at 0 misses,
1 miss, and (for size 5-6) 2 misses; also confirm whether size 6 offers a
3-miss tier (Sleeper's docs don't rule one in or out). Nine data points
total. Feed the results into a `sleeper_payouts.json`-style config
(mirroring `data/config/underdog_payouts.json`) at the start of stage 1.

## 4. Locked decisions

Owner decisions, dated. Sessions may not relitigate; changes are owner-only.
Cross-lane gates live in [roadmap v3 §6](../sportstradamus_roadmap_v3.md).

- Full parity is the scope (owner, 2026-06-10): payout modeling, push/void
  handling, EV, entry construction, Kelly sizing, recommendations YAML — not
  a screener-only surface (the screener is stage 1's fallback, not the goal).
- Money math uses `Decimal`, never float (owner, 2026-06-10). Already the
  practice in [`strategies/kelly.py`](../../src/sportstradamus/strategies/kelly.py)
  and [`underdog_pickem.py`](../../src/sportstradamus/strategies/underdog_pickem.py);
  probability internals stay numpy. The rule is being rehomed to
  [`CLAUDE.md`](../../CLAUDE.md) — cite it there once landed.
- Observation stays within the existing
  [`books.py`](../../src/sportstradamus/books.py) endpoints (owner,
  2026-06-10): no new scrape surface (ToS exposure); adding one is owner-only.
- Platform-aware from day one (owner, 2026-06-10): every entry artifact
  carries its platform so the `sim-bettor-ledger` lane can log Sleeper entries
  from its first slate (roadmap v3 §5.3).
- Sleeper recommendation construction caps entries at 6 legs (owner,
  2026-07-12), even though the real app allows up to 8 — split 2-3 legs =
  Max/power (all-or-nothing), 4-6 legs = Flex (partial-hit-tolerant),
  mirroring the existing Underdog `_pooled_underdog_curve`/`POWER_MAX_SIZE`
  convention ([`payouts.py:19-45`](../../src/sportstradamus/prediction/payouts.py)).
  This is a recommendation-construction choice, not a platform restriction —
  see §3 item 1.

## 5. Module footprint & canonical paths

Canonical import paths per [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §Package
Map; do not recreate deleted shims. Editing outside this list is a stop
condition (§8).

| Module | Role in this lane |
|---|---|
| `sportstradamus.prediction` — [`payouts.py`](../../src/sportstradamus/prediction/payouts.py) + [`joint.py`](../../src/sportstradamus/prediction/joint.py) (post seam-split; the per-leg-payout-vector EV path lands here, not in `parlay.py`), [`correlation.py`](../../src/sportstradamus/prediction/correlation.py), [`cli.py`](../../src/sportstradamus/prediction/cli.py), [`persist.py`](../../src/sportstradamus/prediction/persist.py) | New per-leg-payout-vector EV path; platform plumb-through; snapshot hook (today [`cli.py:275`](../../src/sportstradamus/prediction/cli.py) builds the pick'em snapshot from Underdog offers only) |
| `sportstradamus.strategies` — [`underdog_pickem.py`](../../src/sportstradamus/strategies/underdog_pickem.py), [`_pickem_emit.py`](../../src/sportstradamus/strategies/_pickem_emit.py), [`kelly.py`](../../src/sportstradamus/strategies/kelly.py) | Generalize the pick'em orchestrator (platform parameter) **or** add a sibling module — naming decision flagged for the owner at stage 2; Kelly is consumed, not rewritten |
| `sportstradamus.books` | **READ-ONLY** (locked, §4) — the scrape already delivers everything the decision layer needs |
| [`dashboard/`](../../src/sportstradamus/dashboard/) | Snapshot-reading dashboard view (new surface or extension — legacy `pages/` retired by dashboard-ux P1) |
| [`tests/golden/`](../../tests/golden/) | New EV-engine and orchestrator tests; existing pickem/kelly/dashboard gates stay green |

This lane touches the serving path (`prophecize` → snapshots): see the
inference-path compatibility checklist in
[`model_improvement_track.md`](model_improvement_track.md) §7.3.

## 6. Stage plan

### Stage 0 — product-rules verification

- **Goal:** convert every §3 assumption into a dated, verified fact.
- **Entry:** owner available with Sleeper app access.
- **Scope:** this brief only (owner-assisted deep dive; no code).
- **Acceptance:** §3 list rewritten in place with verified-on dates and
  evidence pointers; the [`payouts.py:83`](../../src/sportstradamus/prediction/payouts.py)
  leg-cap comment confirmed or marked for correction; ledger line appended.
- **Est. sessions:** 1.
- **Kill criteria:** product is not a static-multiplier pick'em (pari-mutuel,
  live-only pricing) → `BLOCKED`, owner re-scopes; precedent: the Pick'em
  Champions removal (roadmap v3 §8).

### Stage 1 — Sleeper EV engine

- **Goal:** per-leg payout-vector pricing for Sleeper, `Decimal` at money
  boundaries (§4). Two payout regimes, mirroring Underdog's existing
  `_pooled_underdog_curve` pattern
  ([`payouts.py:19-45`](../../src/sportstradamus/prediction/payouts.py)):
  **Max** (2-3 legs, all-or-nothing — EV from the product of the selected
  legs' `payout_multiplier`s) and **Flex** (4-6 legs, partial-hit-tolerant —
  needs a `sleeper_payouts.json`-style table populated via the §3.1 test
  protocol before this half can be implemented). Push-aware per stage-0
  semantics (§3 item 3), including the 2-leg full-refund special case.
  Owner-routed into this stage (2026-07-10, PARLAY_AUDIT §2.6 option a):
  replace the inline parlay-path Kelly gate at
  [`parlay.py:184`](../../src/sportstradamus/prediction/parlay.py)
  (`_KELLY_BANKROLL_FRACTION`, inside `_evaluate_parlay`) with
  `strategies/kelly.py::fractional_kelly_stake` conventions (shrinkage blend
  + cap, already used downstream at `underdog_pickem.py:149`) — highest
  live-money audit finding.
- **Entry:** stage 0 complete. The §3.1 Flex payout test protocol has not
  been run yet — it blocks only the Flex half of this stage; the Max path
  and the Kelly-gate fix can start immediately.
- **Scope:** `sportstradamus.prediction.parlay` + `payouts.py` + their golden
  tests.
- **Acceptance:** new golden unit tests pass against hand-computed 2- and
  3-leg Max entries and 4-6-leg Flex entries (push cases included);
  `poetry run pytest tests/golden/ -q` and
  `poetry run ruff check src/sportstradamus/` clean.
- **Est. sessions:** 3-4 (grew from 2-3 with the Max/Flex dual-regime split —
  see ledger).
- **Kill criteria / branch:** multiplier data unreliable or stale at decision
  time → drop to screener-only scope (surface edges, no entry construction),
  record the verdict in the ledger, re-scope with the owner.

### Stage 2 — decision-layer plumb-through

- **Goal:** lift the Underdog hardcodes — a platform parameter, not a
  copy-paste sibling
  ([`tests/golden/test_no_duplicate_code.py`](../../tests/golden/test_no_duplicate_code.py),
  [`CLAUDE.md`](../../CLAUDE.md) §Reuse) — so entry construction, Kelly
  sizing, and recommendations YAML run for Sleeper.
- **Entry:** stage 1 merged.
- **Scope:** `sportstradamus.strategies` + `sportstradamus.prediction`
  (`correlation.py`, `cli.py`); one module per subagent. Flag the naming
  decision (generalize `underdog_pickem.py` vs sibling) to the owner first.
- **Acceptance:** `pickem-build` (or successor flag) emits YAML with
  platform-tagged Sleeper entries; `poetry run pytest tests/golden/ -q` and
  `poetry run pytest -m integration -n0` clean.
- **Est. sessions:** 2-3.
- **Kill criteria / branch:** generalization forces banned pure-forwarders or
  unjustified parallel blocks → stop before the second copy and put the
  structure question to the owner.

### Stage 3 — dashboard + snapshots

- **Goal:** Sleeper recommendations on the dashboard from parquet snapshots
  only — **never DuckDB from a page** ([`CLAUDE.md`](../../CLAUDE.md) §Hard
  rules; pinned by [`tests/golden/test_dashboard_no_archive_lock.py`](../../tests/golden/test_dashboard_no_archive_lock.py)).
- **Entry:** stage 2 merged; snapshot hook writing Sleeper entries.
- **Scope:** `pages/` + `prediction/persist.py`.
- **Acceptance:** dashboard renders the Sleeper view from snapshots;
  `poetry run pytest tests/golden/ -q` clean (archive-lock test included).
- **Est. sessions:** 1-2.
- **Kill criteria / branch:** a needed field exists only in the archive →
  export parquet from a cron job per CLAUDE.md; schema change breaks existing
  pages → revert and re-land additive-only.

### Stage 4 — tests + ledger integration

- **Goal:** golden/integration coverage for the full Sleeper path; entries
  logging into the `sim-bettor-ledger` schema.
- **Entry:** stage 3 merged.
- **Scope:** `tests/golden/`, `tests/integration/`, the ledger adapter.
- **Acceptance:** all three quality gates clean; a fake-mode run produces
  platform-tagged Sleeper ledger rows.
- **Est. sessions:** 1-2.
- **Kill criteria / branch:** ledger schema not landed (D6 unresolved) → ship
  parity without ledger hooks; ledger line names the dependency; do not block.

### Stage 5 — live soak

- **Goal:** Sleeper recommendations running on live summer slates
  (WNBA/MLB), landed before NFL Week 1 (Sept 2026).
- **Entry:** stage 4 merged to `devel`.
- **Scope:** observation only; brief + ledger updates.
- **Acceptance:** consecutive daily `prophecize` runs write Sleeper entries to
  snapshots with no exceptions in the structured logs; owner reviews the YAML.
- **Est. sessions:** ongoing, low-touch.
- **Kill criteria / branch:** product-assumption drift → §3 protocol (stop,
  re-verify stage 0, revise this brief in place, resume).

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > home-of-record
  doc > this brief > roadmap v3.
- The dashboard reads parquet snapshots only, never DuckDB —
  [`CLAUDE.md`](../../CLAUDE.md) §Hard rules.
- `books.py` is read-only here (§4); platform quirks live downstream, the way
  [`model_prob.py:418`](../../src/sportstradamus/prediction/model_prob.py)
  applies `UNDERDOG_BOOST_BASELINE` to Underdog only, leaving Sleeper boosts raw.
- Parameterize, don't duplicate: parallel code only where the *knowledge*
  differs (payout mechanics); everything else takes a platform argument —
  [`CLAUDE.md`](../../CLAUDE.md) §Reuse.
- One subagent per module on multi-module stages ([`CLAUDE.md`](../../CLAUDE.md)
  §General Rules). This lane never edits `stat_meta.json` or the model
  lifecycle — out of footprint.

## 8. Escalation & stop conditions

**STOP and ask the owner when:** entry criteria unmet; gates red at session
start through no fault of yours; smoke regression; any change to gate
constants or test tolerances; anything touching credentials, paid APIs, cron,
or ToS surface — any new Sleeper endpoint is ToS surface (§4); two
consecutive sessions with no acceptance criterion moving (the grind detector).

**PARK AND PIVOT when blocked externally:** append a ledger line with the
blocking reason, set status to `BLOCKED (on: …)`, and point the owner at the
[roadmap v3](../sportstradamus_roadmap_v3.md) §4 swimlane index.

**DISPATCH a subagent when:** research-analyst (Opus-backed) — *optional*
here, with no named research-gated triggers: the open questions are product
mechanics (owner questions), not literature questions; devel-ship-curator for
**every** devel-bound PR; prompt-engineer for new briefs / major re-briefs;
refactoring-specialist per the five [`CLAUDE.md`](../../CLAUDE.md) triggers.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session
  (CLAUDE.md five-trigger rule).
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- One ledger line appended to §10; status line updated if a stage boundary
  was crossed.
- Never push `devel` directly — devel-ship-curator carves ship PRs.
- Durable non-obvious lesson? Offer a memory capture (CLAUDE.md §Agentic
  workflow conventions).

## 10. Ledger (append-only, newest first, cap ~15 — older lines live in git)

- 2026-07-12 · stage 0 complete · owner-verified in-app: leg cap=8 (payouts.py:83 comment wrong, marked for correction into stage 1); our construction locks 2-3=Max/power + 4-6=Flex capped at 6 legs (new decision, mirrors `_pooled_underdog_curve`, bumps stage 1 est. to 3-4 sessions); multiplier=stake×product confirmed; push=leg-drop+reprice except 2-pick=full refund (Underdog divergence); kill-criteria passed (fixed multiplier, not pari-mutuel); fixed 10 stale file/line citations from the parlay.py→payouts.py seam-split drift · OPEN: Sleeper Flex payout table unpublished, needs §3.1 in-app test · next: stage 1 — run §3.1 test, then payouts.py/parlay.py:184 Kelly-gate work
- 2026-07-11 · flagged critical path · roadmap audit: this lane blocks both D3 (parlay-dependence) and dfs-products 2b/2c/5; owner marked it the recommended next code-heavy lane, target merge ~Aug (pre NFL Wk 1) · next: stage 0
- 2026-07-10 · owner decision · PARLAY_AUDIT §2.6 parlay-path Kelly fix routed into stage 1 (option a — shrinkage-aware sizing lands with the parlay.py rebuild) · next: unchanged
- 2026-07-10 · heads-up · new dfs-products lane may touch books.py Sleeper ingestion (alt-line de-vig, its stage 2a) — books.py is read-only in THIS lane so no footprint collision, but re-verify this brief's stage-0 payload facts if that lands first; PARLAY_AUDIT.md §2.6 flags parlay-path Kelly for possible routing into this lane's parlay.py rebuild (owner call) · next: unchanged
- 2026-06-10 · created · brief drafted from roadmap-v3 migration · next: stage 0 product verification with owner
