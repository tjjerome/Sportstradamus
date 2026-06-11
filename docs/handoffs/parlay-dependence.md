# Parlay Dependence — copula on PIT residuals

> Status: BLOCKED (on: D3 — see roadmap v3 §7)

## 1. Mission & money logic

Upgrade parlay joint-probability pricing from the incumbent Gaussian copula
over a pairwise-assembled Pearson correlation matrix to a copula fit on
PIT-transformed historical residuals `U_{i,t} = F̂_i(y_{i,t})` — fit within
same-game leg groups and per leg-type pair (e.g. QB pass-yds × WR rec-yds on
the same offense), per-pair correlations EB-shrunk across teams, priced by
sampling jointly and inverting through the marginals. Add a dependence
diagnostic: average pairwise rank correlation of residual PITs within
same-game groups vs the under-independence prediction. At parlay dimensions
2–6 a Gaussian copula suffices and vines are overkill (archived v2 §1.2);
Gaussian-vs-t is the open question for the research brief.

Money logic: the five ship gates certify **marginals only**; the product is
**parlays**; and the DFS apps largely don't tax leg correlation — that
asymmetry is the core of why they're beatable. The predecessor design calls
this "the audit's single largest *product*-EV lever"
([archived v2 §1.2](../archive/sportstradamus_roadmap_v2.md));
[`operation_ship_75.md`](../operation_ship_75.md) §10 defers it here as "the
audit-deferred tail (post-Ship-75): the parlay copula / dependence layer
(Phase 1.2 follow-ups)". Better joint pricing multiplies the EV of every
entry on both apps.

## 2. Read first (in order)

1. [`operation_ship_75.md`](../operation_ship_75.md) §Purpose + §10 — the
   lens (calibration is the product; Gate 4 PIT-KS is the real bar) and the
   deferral this lane resumes.
2. [`archive/sportstradamus_roadmap_v2.md`](../archive/sportstradamus_roadmap_v2.md)
   §1.2 — the design sketch this brief restates, plus the open Phase 1.2.x
   audit findings.
3. [`PARLAY_AUDIT.md`](../PARLAY_AUDIT.md) — prior audit of
   `find_correlation` / `beam_search_parlays`; its line cites predate the
   `parlay.py` split but the logic analysis stands.
4. `src/sportstradamus/prediction/parlay.py` — incumbent pricing: analytical
   `multivariate_normal.cdf` path (parlay.py:426), 50K-draw MC push/flex path
   (`_PUSH_MC_SAMPLES`, parlay.py:71; parlay.py:323), `_nearest_psd` repair
   (parlay.py:193), the `legacy` flag pattern to mirror (parlay.py:533).
5. `src/sportstradamus/prediction/correlation.py` — how Σ is assembled today:
   stratified parquet lookups (correlation.py:632–638) →
   `_build_game_corr_map` (correlation.py:120) → per-leg-pair sums in
   `_build_correlation_matrices` (correlation.py:201).
6. `src/sportstradamus/training/correlate.py` — the residual machinery to
   REUSE: `_residualize_gamelog` 8-game rolling residualization
   (correlate.py:45, 461), stratified same-team/opposing matrices,
   Spearman→Pearson remap `2·sin(πρ_s/6)` (correlate.py:839), linear
   overlap-credibility shrinkage (correlate.py:54, 678).
7. `src/sportstradamus/scripts/audit_parlay_calibration.py` — the calibration
   harness this lane must re-run; already separates copula vs independence
   gaps (`gap_copula` / `gap_indep`).
8. [`handoffs/sleeper-parity.md`](sleeper-parity.md) — the lane serialized
   ahead of this one (roadmap v3 §5.1); read its status before touching code.

## 3. Verify before you trust

Rule, verbatim: if command output contradicts brief prose, the output wins —
fix the brief in place (minor) or stop and ask the owner (material).

```bash
git fetch origin && git log --oneline origin/devel -3

# D3 input 1 — breadth: ≥2 leagues at target (roadmap v3 §2; targets
# NBA ≥ 16/21 · WNBA ≥ 14/18 · NFL ≥ 15/20). PITs of uncalibrated
# marginals are noise (roadmap v3 §5.2).
python3 -c "import json,collections; m=json.load(open('src/sportstradamus/data/config/stat_meta.json')); [print(l, dict(collections.Counter(c['shipped'] for c in v.values()))) for l,v in m.items()]"

# D3 input 2 — the Opus research brief exists (stage 0 output)
ls /tmp/researcher_*copula* docs/archive/researcher_*copula* 2>/dev/null

# D3 input 3 — sleeper-parity merged (roadmap v3 §5.1: do not interleave;
# both lanes rebuild parlay.py / correlation.py internals)
head -3 docs/handoffs/sleeper-parity.md

# Stage-4 input — resolved parlay history (production host only; absent in
# dev checkouts, see PARLAY_AUDIT.md §3)
ls -la data/parlay_hist.parquet 2>/dev/null
```

### Volatile product assumptions

- **Apps don't fully tax leg correlation.** The whole lane's edge. Re-verify
  at stage 0 and stage 4: the audit harness's `gap_indep` vs `gap_copula`
  split shows what the apps' pricing already absorbs.
- **Payout curves / leg caps** (`data/underdog_payouts.json`, Sleeper 3-leg
  cap) feed `_payout_curve_for` (parlay.py:123). On drift: stop, re-verify
  stage-0 facts, revise this brief in place, resume. The decision lanes own
  the payout tables; this lane only consumes them.

## 4. Locked decisions

- 2026-06-10 — Entry gate D3 is **owner-only** (roadmap v3 §7). Sessions
  prepare decision packets; they never flip the gate.
- 2026-06-10 — `research-analyst` (Opus) dispatch is **HARD-REQUIRED before
  any stage-1 code**. Questions it must answer: Gaussian vs t copula at
  dimensions 2–6; EB shrinkage strength and structure across teams; minimum
  same-game pair counts per league; PIT extraction window and handling of
  discrete/hurdle marginals (randomized PIT). No waiver — do not use
  `.claude/.state/research_waiver` for this lane.
- 2026-06-10 — The incumbent path stays behind a flag until the offline A/B
  verdict; mirror the existing `legacy` flag pattern (parlay.py:533,
  correlation.py:554). Default stays incumbent until stage 3 acceptance.
- 2026-06-10 — Never loosen gates, harness thresholds, or test tolerances to
  pass (ship_75 §Purpose discipline).
- 2026-06-10 — Vine copulas are out of scope at dims 2–6 (archived v2 §1.2);
  relitigating that is owner-only.

## 5. Module footprint & canonical paths

- `sportstradamus.prediction` — `prediction/parlay.py` (pricing, sampling,
  PSD repair), `prediction/correlation.py` (Σ/group assembly,
  `find_correlation`).
- `sportstradamus.training` — `training/correlate.py`. The residual machinery
  already lives here (`_residualize_gamelog` 8-game rolling residualization;
  stratified same-team/opposing/cross-game matrices) — **REUSE it, don't
  duplicate it**. PIT extraction extends this module or a sibling under
  `training/`, not a new parallel pipeline.
- `src/sportstradamus/scripts/audit_parlay_calibration.py` — harness; note it
  mirrors the payout table by hand (audit_parlay_calibration.py:42) and must
  stay in sync with any pricing change.
- `tests/golden/` — copula-vs-hand-computed joints, flag-path equivalence.

Serving path is touched ⇒ the inference-path compatibility checklist applies
([`operation_ship_references.md`](../operation_ship_references.md) — reference,
don't restate). **File-conflict constraint:** this lane must not run while
`sleeper-parity` is mid-flight in the same files (roadmap v3 §5.1). Editing
outside this footprint is a stop condition (§8).

## 6. Stage plan

**Stage 0 — Research dive + data census.**
- Goal: an Opus statistician's brief answering the §4 questions, plus a
  census of usable history.
- Entry: none — read-only; may run early pre-D3 if the owner wants (it is a
  D3 input). Does not touch `prediction/`.
- Scope: `research-analyst` dispatch (definition:
  `.claude/agents/research-analyst.md`; Opus-backed per its frontmatter;
  output `/tmp/researcher_{topic}.md`); census script against gamelogs.
- Acceptance: brief exists with an implementable verdict (durable verdict
  pointer committed to the §10 ledger); a census **table** — historical
  same-game leg-pair counts per league and market-pair, from the same
  gamelog window `correlate.py` uses.
- Est. 1–2 sessions.
- Kill: census shows pair counts too thin to beat the incumbent's shrunk
  matrix in any league → record verdict, close lane DONE(no-ship).

**Stage 1 — PIT-residual extraction.**
- Goal: per-cell PITs `U_{i,t} = F̂_i(y_{i,t})` from shipped marginals on
  historical gamelogs, leak-free, reusing `correlate.py` residual plumbing.
- Entry: D3 flipped; stage-0 brief in hand.
- Scope: `sportstradamus.training` only.
- Acceptance: PITs of shipped cells ≈ Uniform(0,1) by KS on holdout;
  discrete/count markets via randomized PIT per the brief's prescription.
- Est. 1–2 sessions.
- Kill: PITs materially non-uniform on shipped cells → the marginals aren't
  ready; set BLOCKED (on: model-track calibration), park.

**Stage 2 — Copula fit + EB shrinkage.**
- Goal: per leg-type-pair copula correlations within same-game groups,
  EB-shrunk across teams (strength/structure per the brief).
- Entry: stage-1 PITs accepted.
- Scope: `sportstradamus.training`.
- Acceptance: held-out joint log-likelihood beats independence **and** beats
  the incumbent Pearson-matrix approach on the same groups.
- Est. 1–2 sessions.
- Kill: no held-out log-lik gain over incumbent → lane-level if-it-fails
  (below).

**Stage 3 — Pricing integration behind a flag + offline A/B.**
- Goal: sample jointly from the fitted copula, invert through the marginals,
  price; incumbent stays the default behind the flag.
- Entry: stage-2 accepted; sleeper-parity not mid-flight in these files.
- Scope: `sportstradamus.prediction` (one module per subagent if both files
  move).
- Acceptance: offline A/B vs incumbent on cached test sets shows a
  joint-calibration gain; golden tests pin both flag paths.
- Est. 2–3 sessions.
- Kill: A/B flat or negative → lane-level if-it-fails (below).

**Stage 4 — Dependence diagnostic + production calibration re-run.**
- Goal: the diagnostic (avg pairwise rank correlation of residual PITs within
  same-game groups vs under-independence) and a populated
  `audit_parlay_calibration.py` re-run on production archive data
  (owner-assisted; production host — dev checkouts lack
  `data/parlay_hist.parquet`).
- Acceptance: diagnostic table committed; populated reliability artifacts
  with the new path's decile gap ≤ incumbent's.
- Est. 1 session + owner time.
- Then ship via `devel-ship-curator`; flag default flips only in that PR.

**Lane-level if-it-fails:** if the dependence diagnostic shows the incumbent
already captures same-game dependence (no exploitable gap), or the A/B shows
no joint-calibration gain → keep the incumbent, record the verdict + evidence
pointer in the ledger, close the lane DONE(no-ship). A KILL is a valid,
valuable verdict.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md >
  [`operation_ship_75.md`](../operation_ship_75.md) > this brief > roadmap v3.
- Reuse before you write (CLAUDE.md): residualization, stratification, and
  shrinkage exist in `training/correlate.py`; the lane parameterizes or
  extends them, never re-implements them beside themselves.
- Marginals are read-only here. This lane never edits distribution families,
  `stat_meta.json`, or gate code — joint structure only.
- Keep the audit harness's hand-mirrored payout table
  (audit_parlay_calibration.py:42) in sync with any pricing change, or the
  inverse `Model EV → joint_p` recovery silently drifts.

## 8. Escalation & stop conditions

**STOP and ask the owner when:** entry criteria unmet (D3 not flipped; Opus
brief missing); gates red at session start through no fault of yours; smoke
regression; any change to gate constants, harness thresholds, or test
tolerances; anything touching credentials, paid APIs, cron, or ToS surface;
two consecutive sessions with no acceptance criterion moving (grind
detector).

**PARK AND PIVOT when blocked externally:** append a ledger line with the
blocking reason, set the status line to BLOCKED (on: …), flip the roadmap v3
§4 row, and point the owner at the swimlane index.

**DISPATCH a subagent when:**
- `research-analyst` (Opus-backed) — named triggers: the stage-0 brief
  (hard-required, §4) and any later copula-family or dependence-mechanism
  change. The research-gate hook's path list (`.claude/research_gated.txt`)
  doesn't see `prediction/` files, so this rides the CLAUDE.md research-first
  convention — the judgment call a path matcher cannot see; the Opus brief
  satisfies it.
- `devel-ship-curator` — every devel-bound PR.
- `prompt-engineer` — new briefs / major re-briefs.
- `refactoring-specialist` — per the five CLAUDE.md triggers.

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

- 2026-06-10 · created · brief drafted from roadmap-v3 migration · next: wait for D3; stage-0 census can run early if owner wants
