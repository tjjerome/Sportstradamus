# Sleeper decision-layer parity

> Status: QUEUED (entry: stage-0 product-rules verification with the owner)

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
[`books.py:337-338`](../../src/sportstradamus/books.py); alternates included,
[`books.py:364-366`](../../src/sportstradamus/books.py)) — pricing variance
Underdog's fixed tables do not have, hence more mispricings to harvest.

The load-bearing design fact: **the two apps have different engine shapes.**
Underdog pays from fixed tables keyed by (entry size, miss count)
([`underdog_payouts.json`](../../src/sportstradamus/data/config/underdog_payouts.json));
[`parlay.py:293`](../../src/sportstradamus/prediction/parlay.py)
`_expected_payout_with_pushes` prices off that count-based curve. Sleeper's
payout depends on *which* legs hit (product of per-leg multipliers), so the
count-based curve does not generalize: Sleeper needs a per-leg-payout-vector
EV path — new code, not a config entry beside the legacy stub
`"Sleeper": [1.0, 1.0]` at [`parlay.py:159`](../../src/sportstradamus/prediction/parlay.py).

## 2. Read first (in order)

1. [`CLAUDE.md`](../../CLAUDE.md) §Hard rules + §MANDATORY refactoring-specialist — assumed-read law; this lane leans on the dashboard-parquet rule and the duplicate-code gate.
2. [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §Package Map + §Shipping to Production (`devel`) — canonical import paths and the devel-ship-curator PR mechanics.
3. [`sportstradamus_roadmap_v3.md`](../sportstradamus_roadmap_v3.md) §5 — cross-lane constraints: this lane lands **before** `parlay-dependence`; the `sim-bettor-ledger` schema preferably lands before this lane finishes.
4. [`books.py`](../../src/sportstradamus/books.py) — the Sleeper scrape (`get_sleeper`, `_sleeper_prop_offers`, the four `SLEEPER_*_URL` endpoints at :31-34). Read-only in this lane (§4).
5. [`prediction/parlay.py`](../../src/sportstradamus/prediction/parlay.py) — the payout engine to extend: `_payout_curve_for` (:123), `_expected_payout_with_pushes` (:293), `beam_search_parlays` (:521).
6. [`prediction/correlation.py`](../../src/sportstradamus/prediction/correlation.py) — `find_correlation(offers, stats, platform, ...)` (:548), the platform plumb-through point.
7. [`strategies/underdog_pickem.py`](../../src/sportstradamus/strategies/underdog_pickem.py) + [`strategies/README.md`](../../src/sportstradamus/strategies/README.md) — the decision layer to generalize (`_live_load` :322 calls `get_ud()` only; `find_correlation(..., "Underdog")` :296).
8. [`archive/sportstradamus_roadmap_v2.md`](../archive/sportstradamus_roadmap_v2.md) — background only; status claims stale.

## 3. Verify before you trust

If command output contradicts brief prose, the output wins — fix the brief in
place (minor) or stop and ask the owner (material).

```bash
git fetch origin && git log --oneline origin/devel -3
# Decision layer still Underdog-only? (expect get_ud at ~:337, "Underdog" at ~:296)
grep -n 'get_ud()\|"Underdog"' src/sportstradamus/strategies/underdog_pickem.py
# Legacy Sleeper stub still in place? (expect '"Sleeper": [1.0, 1.0]' at ~:159)
grep -n '"Sleeper"' src/sportstradamus/prediction/parlay.py
# Picks-board shape unchanged? books.py parses: subject_id, wager_type, game_id, options[].outcome/outcome_value/payout_multiplier
curl -s https://api.sleeper.app/lines/available | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d), sorted({x['sport'] for x in d})); print(json.dumps(d[0], indent=2)[:800])"
poetry run pytest tests/golden/ -k "pickem or kelly or parlay" -q
```

### Volatile product assumptions

External facts this lane's math depends on. On drift: stop, re-verify all of
stage 0, revise this brief in place, then resume.

1. **Leg caps.** [`parlay.py:158`](../../src/sportstradamus/prediction/parlay.py)
   comments that Sleeper caps real parlays at 3 legs — **unverified product
   fact**. Re-verify: stage 0, with the owner, in the app's entry builder.
2. **Multiplier composition rule.** Assumed: entry payout = stake × product of
   the selected legs' `payout_multiplier` values; promo/boost stacking and any
   entry-level cap unconfirmed. Re-verify: owner builds a small test entry,
   compares app-shown payout to the hand product; read the house rules.
3. **Push/void semantics per leg.** Voids to ×1.0, refunds the entry, or
   size-dependent rules? (Underdog drops the entry one leg,
   [`parlay.py:307`](../../src/sportstradamus/prediction/parlay.py).)
   Re-verify: house rules + owner; drives stage 1's push-aware EV branch.
4. **Picks-board API shape.** The four endpoints at
   [`books.py:31-34`](../../src/sportstradamus/books.py) and the fields
   `_sleeper_prop_offers` (:301) parses. Re-verify: the `curl` above.

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

## 5. Module footprint & canonical paths

Canonical import paths per [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §Package
Map; do not recreate deleted shims. Editing outside this list is a stop
condition (§8).

| Module | Role in this lane |
|---|---|
| `sportstradamus.prediction` — [`parlay.py`](../../src/sportstradamus/prediction/parlay.py), [`correlation.py`](../../src/sportstradamus/prediction/correlation.py), [`cli.py`](../../src/sportstradamus/prediction/cli.py), [`persist.py`](../../src/sportstradamus/prediction/persist.py) | New per-leg-payout-vector EV path; platform plumb-through; snapshot hook (today [`cli.py:275`](../../src/sportstradamus/prediction/cli.py) builds the pick'em snapshot from Underdog offers only) |
| `sportstradamus.strategies` — [`underdog_pickem.py`](../../src/sportstradamus/strategies/underdog_pickem.py), [`_pickem_emit.py`](../../src/sportstradamus/strategies/_pickem_emit.py), [`kelly.py`](../../src/sportstradamus/strategies/kelly.py) | Generalize the pick'em orchestrator (platform parameter) **or** add a sibling module — naming decision flagged for the owner at stage 2; Kelly is consumed, not rewritten |
| `sportstradamus.books` | **READ-ONLY** (locked, §4) — the scrape already delivers everything the decision layer needs |
| [`pages/`](../../src/sportstradamus/pages/) | Snapshot-reading dashboard view (extend [`2_Predictions_Pickem.py`](../../src/sportstradamus/pages/2_Predictions_Pickem.py) or sibling) |
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
  evidence pointers; the [`parlay.py:158`](../../src/sportstradamus/prediction/parlay.py)
  leg-cap comment confirmed or marked for correction; ledger line appended.
- **Est. sessions:** 1.
- **Kill criteria:** product is not a static-multiplier pick'em (pari-mutuel,
  live-only pricing) → `BLOCKED`, owner re-scopes; precedent: the Pick'em
  Champions removal (roadmap v3 §8).

### Stage 1 — Sleeper EV engine

- **Goal:** per-leg payout-vector pricing in `prediction/parlay.py` — EV from
  the product of the selected legs' multipliers, push-aware per stage-0
  semantics, `Decimal` at money boundaries (§4). Owner-routed into this stage
  (2026-07-10, PARLAY_AUDIT §2.6 option a): replace the inline parlay-path
  Kelly at [`parlay.py:416`](../../src/sportstradamus/prediction/parlay.py)
  with `strategies/kelly.py::fractional_kelly_stake` conventions (shrinkage
  blend + cap) — highest live-money audit finding.
- **Entry:** stage 0 complete.
- **Scope:** `sportstradamus.prediction.parlay` + its golden tests.
- **Acceptance:** new golden unit tests pass against hand-computed 2- and
  3-leg entries (push cases included); `poetry run pytest tests/golden/ -q`
  and `poetry run ruff check src/sportstradamus/` clean.
- **Est. sessions:** 2-3.
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
  [`model_prob.py:300`](../../src/sportstradamus/prediction/model_prob.py)
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

- 2026-07-10 · owner decision · PARLAY_AUDIT §2.6 parlay-path Kelly fix routed into stage 1 (option a — shrinkage-aware sizing lands with the parlay.py rebuild) · next: unchanged
- 2026-07-10 · heads-up · new dfs-products lane may touch books.py Sleeper ingestion (alt-line de-vig, its stage 2a) — books.py is read-only in THIS lane so no footprint collision, but re-verify this brief's stage-0 payload facts if that lands first; PARLAY_AUDIT.md §2.6 flags parlay-path Kelly for possible routing into this lane's parlay.py rebuild (owner call) · next: unchanged
- 2026-06-10 · created · brief drafted from roadmap-v3 migration · next: stage 0 product verification with owner
