# DFS Products — game lines, Ladders, Rivals, alt-line hardening

> Status: QUEUED (entry: stage-0 product verification; stages 0–1 startable now)

## 1. Mission & money logic

Build the decision engines that turn calibrated marginals into +EV entries on the DFS apps'
newer product surfaces: Underdog Combo Entries / game-line Prediction Picks, Ladders,
Sleeper Markets, and the Rivals head-to-head. This lane does **not** train models — it
prices and sizes products from marginals the model track already certifies.

Money logic: `model_improvement_track.md` §1 — "Gate-4 PIT-KS calibration **is** alt-line
pricing accuracy — the alt-line/Ladders/Rivals surfaces where the profit concentrates."
This lane is where that calibration cashes out. §6.11 (WS-4) named three of these products
(Rivals, Ladders, the copula) as the product-EV lever the model track defers to a
downstream lane; this brief is that lane, generalized to the newer surfaces (game-line
combos, alt-line hardening) two Opus stage-0 briefs since scoped. See also
[`PARLAY_AUDIT.md`](../PARLAY_AUDIT.md) for the incumbent `find_correlation` /
`beam_search_parlays` audit + dispositions this lane's pricers plug into (its
empirical-calibration run stays unpopulated in dev — production host only).

## 2. Read first (in order)

1. [`model_improvement_track.md`](model_improvement_track.md) §1.1 + §6.11 — the lens
   (calibration is the product) and the WS-4 verdicts this lane inherits (Rivals-first,
   copula default, ladder table census).
2. [`archive/researcher_ladders_stage0.md`](../archive/researcher_ladders_stage0.md) —
   Ladders pricing brief in full (A1–A8 + VERIFY register). Load-bearing for Stage 3.
3. [`archive/researcher_gamelines_stage0.md`](../archive/researcher_gamelines_stage0.md) —
   game-line combo brief in full (B1–B8 + kill rule). Load-bearing for Stage 4; **B7-P4**
   is the single fact that gates whether Stage 4 builds at all.
4. [`archive/researcher_copula_stage0.md`](../archive/researcher_copula_stage0.md) —
   R3, the dependence-estimator conventions (hierarchical Fisher-z EB, PIT extraction,
   `census_parlay_pairs.py` design) both product briefs extend rather than reinvent.
5. [`parlay-dependence.md`](parlay-dependence.md) §5 + §7 — the sibling lane rebuilding
   `parlay.py`/`correlation.py` internals; this lane's Stage 2/5 items queue behind it,
   never interleave (roadmap §5.1).
6. `src/sportstradamus/strategies/underdog_pickem.py` — the orchestrator pattern to rhyme
   with (`RecommendedEntry`, YAML emit, `pickem-build` CLI); it already **selects and sizes**
   Rivals entries via the incumbent copula (2/3-leg cap, `_validate_rivals_coverage`) — this
   lane adds the missing **difference pricer**, it does not build Rivals from scratch.
7. `src/sportstradamus/strategies/kelly.py` — `fractional_kelly_stake` conventions
   (quarter-Kelly, 0.5% cap, shrinkage blend) every new pricer wraps, generalized to
   discrete outcome vectors per the ladders brief §A5.

## 3. Verify before you trust

Rule, verbatim: if command output contradicts brief prose, the output wins — fix the brief
in place (minor) or stop and ask the owner (material).

```bash
git fetch origin && git log --oneline origin/devel -3

# ladder table census — is the pricing substrate as large as the stage-0 brief measured?
python3 -c "
import duckdb
con = duckdb.connect('archive/archive.duckdb', read_only=True)
print(con.execute(\"SELECT league, COUNT(*) FROM ladder GROUP BY league\").fetchall())
"

# payout config this lane consumes, never owns
ls -la src/sportstradamus/data/config/underdog_payouts.json

# census script status — design-only per both stage-0 briefs; confirm still absent
# before building it a second time
find . -iname "census_parlay_pairs.py" 2>/dev/null

# sleeper-parity status — Stage 2/5 queue behind it (roadmap §5.1)
head -3 docs/handoffs/sleeper-parity.md
```

### Volatile product assumptions

Each re-verify step is a stage-0 recapture, not a spot-check — on drift, stop, re-run the
relevant stage-0 capture, revise this brief in place, resume.

- **Combo fee split ~90/10, roll-into-fantasy-stake, partial-payout floor.** CONDITIONAL,
  web-sourced 2026-07 (gamelines brief B7 P2/P3). Re-verify: place/inspect a real Combo
  Entry payload; the whole B1 EV algebra rescales on drift.
- **Ladders: 3–5 picks, 3 rungs, payout keyed to lowest rung reached, rung-1 = 1× refund,
  ≥2 teams required, rung lines app-fixed (not user-chosen).** Web-sourced (ladders brief
  VERIFY-1..8). Re-verify: capture a live Ladders slip payload. **Never hardcode a
  multiplier table** — read it per-slip (VERIFY-4/5); the "1000×" marketing figure does not
  reconcile against the derived Pick-3/4/5 table.
- **Sleeper Markets contract shape + $0.02 fee; no confirmed fantasy-pairing combo
  analogue.** Re-verify at Stage-0 capture; if Sleeper has no combo, that sub-lane is
  standalone-contract-only for Sleeper.
- **App pairing / banned-combo rules — B7-P4 preliminary answer (owner field test,
  2026-07-10): UD DOES reprice event×pick correlation.** MLB 3-leg A/B: ML 1.79x +
  same-team pitcher-Ks-over 1.62x + other-game filler 2.33x quoted 6.11x; correlated
  leg swapped for an uncorrelated 1.62x leg quoted 6.50x (leg product 6.76x) ⇒ base
  parlay haircut ~3.8%, correlation penalty ~6.0% ≈ implied pair ρ ~0.08. Pairing is
  ALLOWED but requires an unrelated filler leg (game line + correlated player alone is
  refused — same shape as the single-team player rule); legs quote as unified
  multipliers, so whether the contract+roll structure (B1) or a plain multiplicative
  parlay applies is still open (checkout fee-split screenshot owed). Stage-4 edge
  thesis narrows: hunt tax-vs-true-ρ mismatch per pair-type (B2 sources (i)-residual +
  (iv)), not an untaxed coupling — B8's "taxed ⇒ kill" is NOT triggered until the
  tax-curve sweep (weak/strong/negative-ρ pairs, filler variation) prices the mismatch.
  Single sample; screenshots + sweep owed to `docs/archive/`. **UD alt-rung API
  breadth** (`get_ud`) still unverified.
- **Payout tables are consumed, not owned.** UD per-season table drift is a
  `hygiene-closeout` housekeeping item, not this lane's; Sleeper's table lives in
  `sleeper-parity` stage 0. This lane reads whatever those lanes/stage-0 captures land.

## 4. Locked decisions

- 2026-07-10 — **Verify-first for game lines.** No combo-engine code until Stage-0 payload
  capture adjudicates B7-P4 (owner decision at plan review). The pricing algebra (B1) and
  MVN composition (B5) are known-cost engineering; the edge (B2-i) is unproven until
  captured — the two are tracked separately in the ledger.
- 2026-07-10 — **Book-implied game-line marginals only; no team-market model training.**
  `model_improvement_track.md` §1.1 — the sharp de-vigged consensus is the truth estimate;
  this lane never trains a team-outcome model.
- 2026-07-10 — **Serve-time budget (owner).** `prophecize` stays a few minutes typical, 15
  minutes MAX end-to-end on a heavy day. Every stage's acceptance includes measured
  wall-time impact. Per-stage compute-budget targets from the stage-0 briefs: Ladders
  ~20–30s worst case (10× headroom); game-line combos ~1–60s worst case. A stage that can't
  show its number is not accepted.
- 2026-07-10 — **File-conflict queueing.** Stages touching `parlay.py`/`correlation.py`/
  `correlate.py`, `prediction/cli.py`, `persist.py`, or `sleeper-parity`'s declared
  footprint queue behind roadmap §5.1's serialization. v1 work for every stage is
  **import-only** in new modules; see §5.
- 2026-07-10 — **Constellation grammar changes are owner-only** (DESIGN.md §4a FIXED);
  presentation design for any new surface lives in `dashboard_ux_redesign.md`, not here.
- 2026-07-10 — **CFTC contract surface (UD Predict, Sleeper Markets) is pricing/display
  only** until the owner clears anything transactional — a ToS/escalation item (§8).
- 2026-07-10 — **Research gates.** The two archived stage-0 briefs discharge research-first
  for Stages 3–4 (CLAUDE.md "research-first" convention). Any later dependence-mechanism
  change (e.g. adopting the ladders brief's t-copula branch, or a new copula family)
  needs a fresh `research-analyst` dispatch — no waiver.

## 5. Module footprint & canonical paths

Per `CONTRIBUTING.md` §Package Map; do not recreate deleted shims.

| Path | Status | Role |
|---|---|---|
| `strategies/underdog_ladders.py` (NEW) | conflict-free | Ladders pricer + candidate builder, rhyming with `underdog_pickem.py` (ladders brief §0, A6, A8) |
| `strategies/` Rivals difference-pricer (NEW module, name TBD at Stage 1) | conflict-free | `P(A−B>k)` pricer feeding the existing `underdog_pickem.py` Rivals orchestration — does not replace it |
| `strategies/` combo-EV module (NEW, Stage 4 only) | conflict-free | B1 combo-EV algebra + B5 mixed-slip composition, import-only against `parlay.py` |
| `scripts/audit_ladder_calibration.py` (NEW) | conflict-free | Ladders brief A7 validation harness (Track 1 reliability, Track 2 joint calibration) |
| `scripts/estimate_game_line_corr.py` (NEW) | conflict-free | Gamelines brief B3 standalone ρ estimator; **do not build a second census** — extends `census_parlay_pairs.py --source gameline` per both briefs |
| `scripts/census_parlay_pairs.py` (NEW when built) | conflict-free | Shared with `parlay-dependence` Stage 0; add `--source gameline` mode here rather than forking |
| `books.py` (ingestion only) | conflict-free | Sleeper alt-line de-vig (Stage 2a); game-line contract/combo payload capture (Stage 0/4); **never touches `prediction/`** |
| `data/config/stat_map.json`, payout config | conflict-free | New ingestion entries; payout tables consumed, never owned (§3) |
| `prediction/correlation.py`, `prediction/parlay.py` | **import-only** | Every new pricer imports Σ assembly / MVN+push-MC machinery; zero edits at v1 (all stage-0 briefs confirmed this composes cleanly) |
| `prediction/cli.py`, `persist.py` | **QUEUED** | Alt-line columns into snapshots (Stage 2c), scoring-path wiring (Stage 2b) — behind `sleeper-parity`'s declared footprint |
| `training/correlate.py`, `prediction/correlation.py` Σ assembly | **QUEUED (Stage 5)** | Team markets / game-line legs into the correlation engine — behind `sleeper-parity`, never interleaved with `parlay-dependence` |
| `tests/golden/` | conflict-free | One golden suite per new pricer, vs hand-computed cases |

Serving path is touched at Stages 2b/2c/4/5 ⇒ the inference-path compatibility checklist
applies (`operation_ship_references.md` — reference, don't restate).

## 6. Stage plan

**Stage 0 — Product verification & payload capture.** Conflict-free; owner-assisted for
in-app-only facts.
- Entry: none.
- Scope: `books.py` read paths (capture only, no scoring changes); manual payload logging.
- Capture list: UD Prediction Picks + Combo Entries payloads (fee split, roll mechanics,
  pairing restrictions, cash-out, per-contract fee — gamelines brief B7 checklist in full);
  Ladders offer payloads (rung structure, per-slip payout table — never hardcode); `get_ud`
  alt-rung breadth; Sleeper Markets payloads + alt-line shape; app pairing/banned-combo
  ground truth. **The load-bearing capture is B7-P4** — preliminarily answered TAXED
  (§3, owner field test 2026-07-10); remaining capture = evidence artifacts + the
  tax-curve sweep that decides whether the penalty is flat (exploitable mismatch) or
  pair-specific (B8 kill).
- Acceptance: §3 volatile-assumptions table fully adjudicated with evidence pointers
  (payload samples committed under `docs/archive/`); a game-line go/no-go packet built on
  B7-P4 + the B8 kill conditions, ready for owner sign-off.
- Est. 1–2 sessions (owner time for in-app capture is the bottleneck, not code).
- Kill: product is unpriceable or has no edge → close that sub-lane DONE(no-ship). Ladders
  API-absent (VERIFY-6 fails) → Stage 3 flips to the manual-entry mode (ladders brief §A8)
  rather than killing.

**Stage 1 — Rivals difference-pricer.** Conflict-free (import-only; verify at entry).
- Entry: Stage 0 Rivals-relevant facts adjudicated (contract/pairing rules don't touch
  Rivals directly — this stage can start once the plain margin-behavior audit is done).
- Scope: `strategies/` new module (import-only against `correlation.py`); `tests/golden/`.
- Work: margin-behavior audit on Rivals history first, then a `P(A−B>k)` pricer with push
  handling at integer `k`, consuming the incumbent ρ at d=2 (copula brief Q5 — Rivals-first
  is a YES; bivariate Gaussian suffices, the upgrade path improves the ρ number, not the
  machinery). Golden tests vs hand-computed bivariate-normal cases; offline EV table on
  historical Rivals offers.
- Acceptance: golden suite green; offline EV table shows the pricer's `P(A−B>k)` tracks
  realized margin outcomes within the audit harness's reliability band.
- Est. 1–2 sessions.
- Kill: audit shows Rivals margins are already sharply priced with no exploitable gap →
  record verdict + evidence pointer, close this sub-item DONE(no-ship). Note: this is the
  "cheap early product win" `model_improvement_track.md` §6.11 names — that doc's Rivals
  bullet now points here.

**Stage 2 — Alt-line hardening.** Mixed conflict status by sub-item.
- **2a — Sleeper alt-line de-vig in `books.py` ingestion.** Conflict-free (sleeper-parity
  holds `books.py` read-only in its own lane per its §4; append a heads-up ledger line to
  `sleeper-parity.md` §10 on any Sleeper-path change). Entry: Stage 0 Sleeper capture done.
- **2b — `ladder` table consumption for rung-level book probs at alt lines.** Ingestion-side
  work is free; any scoring-path edit to `prediction/cli.py` or `model_prob.py` **QUEUES**
  behind `sleeper-parity`. Entry: `ladder` census (§3 command) confirms sufficient rung
  density at current offers.
- **2c — alt-line columns into snapshots** (`persist.py`). **QUEUES** behind
  `sleeper-parity`; coordinate with `dashboard-ux` before adding any new column the
  dashboard will read.
- Acceptance: 2a — de-vigged alt lines land in the archive with a golden test; 2b — rung
  join density measured and either lands or is deferred per the kill rule; 2c — new
  snapshot columns documented in `persist.py`'s schema comment, dashboard-ux notified.
- Est. 1 session (2a) + 1–2 sessions each for 2b/2c once unqueued.
- Kill (2b only): ladder-table join too sparse at current offer volume to beat the
  incumbent dist-inversion path → keep dist-inversion, record verdict. 2a/2c have no kill
  condition (pure hardening); if blocked, PARK AND PIVOT per §8.

**Stage 3 — Ladders decision engine.** Conflict-free (new `strategies/underdog_ladders.py`
+ `scripts/`, import-only on `prediction/`).
- Entry: Stage 0 Ladders facts adjudicated (VERIFY-1..8 closed; the ladders research brief
  already answers the pricing/staking/selection design in full, discharging research-first).
- Scope: `strategies/underdog_ladders.py`, `scripts/audit_ladder_calibration.py`,
  `tests/golden/`.
- Work: three-orthant nested-survival pricer via a per-game reused Sobol'-QMC draw pool
  (N=8,192, CRN across candidates — ladders brief A1/A2); `DeepScore` two-stage selection
  pre-filter (A6); discrete-payout-vector Kelly via 1-D Brent solve (A5); the A7 validation
  harness on the `ladder` table.
- Acceptance: the brief's numeric gates — rung-survival reliability ≤0.03 absolute (r1/r2)
  and ≤0.04 (r3) count-weighted per league; joint deep calibration ≥20% relative
  improvement vs independence; golden tests vs hand-computed 2-pick cases; measured
  worst-case wall-time ≤60s on an NBA-heavy fixture (10× headroom under the 15-min budget).
- Est. 3–4 sessions (pricer + harness + Kelly + golden suite).
- Kill: Track-1 rung reliability fails for ≥3 of a league's top-EV markets by >0.05
  absolute in a consistent direction → that league is ladder-ineligible; route the finding
  to `model_improvement_track.md` §6.11 as a Gate-4/alt-line-accuracy deficit, set this
  sub-item BLOCKED (on: model-track calibration) for that league. A t-copula does not
  rescue a marginal-survival miscalibration (ladders brief, KILL rule) — don't try.

**Stage 4 — Game-line combo engine v1 (CONDITIONAL).** Conflict-free (`books.py` ingestion
+ `stat_map`/payout config + new `strategies/` module + standalone `scripts/`).
- Entry: Stage 0 verified **including B7-P4**, and the owner go/no-go from the Stage-0
  packet recorded in this brief's §4 (a new dated line) before any pricer code lands.
- Scope: new combo-EV `strategies/` module; `scripts/estimate_game_line_corr.py`
  (standalone — **not** `correlate.py`; census reuse as `census_parlay_pairs.py --source
  gameline`); `tests/golden/`.
- Work per gamelines brief §B: combo-EV closed form (B1); book-implied event marginals via
  the Odds-API de-vigged consensus, power-de-vig for lopsided moneylines (B4); book-derived
  `p_push` for spread/total legs feeding the incumbent's existing push-MC path (B5 — the
  one real gap: `get_push_prob` returns 0 for `dist="Normal"`, fix by supplying a nonzero
  `p_push`, no `distributions.py` edit); mixed-slip composition via `parlay.py`'s MVN/MC,
  import-only.
- Acceptance: golden tests vs hand-computed combo-EV cases; offline calibration A/B shows
  the augmented MVN `P11` tracks empirical joint hit rate within ±2% absolute on a held-out
  slate sample (gamelines brief Stage-2 gate); measured wall-time within budget (§4).
- Est. 3–4 sessions.
- Kill: any of the B8 conditions fires (app prices event×pick correlation; pairing rules
  block correlated combos; fees exceed median edge; contract prices are strictly sharper
  than our consensus) → close the combo sub-lane DONE(no-ship); game-line rows stay
  dashboard-only book-implied per `dashboard_ux_redesign.md` §8 scar item 4.

**Stage 5 — Correlation-engine + snapshot integration.** **CONFLICTED** — queues entirely
behind `sleeper-parity`; never interleave with `parlay-dependence` (roadmap §5.1).
- Entry: `sleeper-parity` merged; if `parlay-dependence` (D3) fired first, this stage
  delivers through `parlay-dependence`'s stages instead, with this lane supplying the ρ
  design (B3's estimand + estimator) as an input, not a competing implementation.
- Scope: team markets into `correlate.py`/`correlation.py` Σ assembly; game-line legs into
  beam/slip scoring; snapshot artifacts flip the `dashboard-ux` scars this lane's earlier
  stages left open.
- Acceptance: held-out joint calibration on team-market pairs beats independence per the
  copula brief's estimator conventions; `dashboard_ux_redesign.md` §8 scar item 4 clears.
- Est. 2–3 sessions once unblocked.
- If-it-fails: `parlay-dependence` replaced Σ assembly first → re-express this stage's ρ
  work on the new path (branch, not kill — the estimator design survives either way).

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md >
  [`model_improvement_track.md`](model_improvement_track.md) > this brief > roadmap v3.
- **Reuse before you write.** `_residualize_gamelog`, the matmul Spearman+overlap trick, and
  the Fisher-z EB machinery live in `training/correlate.py` / the copula brief — every new
  ρ estimator in this lane parameterizes or extends them, never re-implements beside them.
- **Marginals are read-only here.** This lane never edits distribution families,
  `stat_meta.json`, gate code, or `training/scorecard.py` — pricing and staking only.
- **No `parlay.py`/`correlation.py` edits at v1.** Every new pricer is import-only against
  them; the stage-0 briefs confirmed this composes cleanly (B5, ladders brief §0). An edit
  to either file is a stop condition (§8), not a judgment call.
- **Payout tables are consumed, not derived.** Never hardcode a multiplier (the "1000×"
  marketing trap, ladders brief VERIFY-4) — read it per-slip from the captured payload.

## 8. Escalation & stop conditions

**STOP and ask the owner when:** entry criteria for the active stage are unmet; gates red
at session start through no fault of yours; smoke regression; any change to gate constants,
harness thresholds, or test tolerances; anything touching credentials, paid APIs, cron, or
ToS surface (the CFTC contract surface is explicitly named, §4); two consecutive sessions
with no acceptance criterion moving (grind detector).

**PARK AND PIVOT when blocked externally:** append a ledger line with the blocking reason,
set the status line to `BLOCKED (on: …)`, flip the roadmap v3 §4 row, and point the owner
at the swimlane index for the next lane.

**DISPATCH a subagent when:**
- `research-analyst` (Opus-backed) — named triggers: any dependence-mechanism change beyond
  what the two archived stage-0 briefs already answered (e.g. adopting the ladders brief's
  t-copula branch, a new copula family, or a materially different pricing formulation).
- `devel-ship-curator` — every devel-bound PR.
- `refactoring-specialist` — per the five CLAUDE.md triggers.
- `prompt-engineer` — new briefs / major re-briefs, including the Stage-1 naming decision
  for the Rivals pricer module once it's made.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session (CLAUDE.md five-trigger
  rule).
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- One ledger line appended to §10; status line updated if a stage boundary was crossed.
- Never push `devel` directly — devel-ship-curator carves ship PRs.
- Durable non-obvious lesson? Offer a memory capture (CLAUDE.md §Agentic workflow
  conventions).

## 10. Ledger (append-only, newest first, cap ~15 — older lines live in git)

- 2026-07-10 · stage-0 field result · B7-P4 = TAXED, preliminary (owner MLB A/B: ~6.0%
  correlation penalty over a ~3.8% base haircut; pairing allowed w/ filler-leg
  requirement); §3 bullet revised in place · next: evidence screenshots + tax-curve
  sweep before any Stage-4 go/no-go.
- 2026-07-10 · lane created · stage-0 research done during planning (Opus research-analysts):
  ladders brief (`docs/archive/researcher_ladders_stage0.md`) + gamelines brief
  (`docs/archive/researcher_gamelines_stage0.md`) committed; audit dispositions cross-checked
  against `PARLAY_AUDIT.md`; serve-time budget locked (≤15 min heavy day) · next: Stage-0
  payload capture (owner-assisted; B7-P4 first) or Stage-1 Rivals pricer.
