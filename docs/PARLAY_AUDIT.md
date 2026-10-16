# Parlay Audit

Audit of the correlated-parlay pricing path: `find_correlation`
(`prediction/correlation.py`) and `beam_search_parlays` (`prediction/parlay.py`).
Read-only analysis; remediation is dispositioned in §4, not landed here. This doc is
the canonical home for parlay-engine audit findings and their dispositions; lane
briefs point here rather than restating. Older findings that remediation has since
closed live in git history, not in this body.

---

## 1. `find_correlation`

### 1.1 Joint-probability formula

`find_correlation` builds two pairwise matrices and hands them to
`beam_search_parlays`; it computes no final joint probability itself.

**Pairwise EV matrix** (`_build_correlation_matrices`, `prediction/correlation.py`):

```
EV[i, j] = exp(C[i,j] * V[i,j]) * P[i,j] * boosts[i] * M[i,j] * boosts[j] * payouts[0]
```

with `V[i,j] = sqrt(p_i(1-p_i) p_j(1-p_j))`, `P[i,j] = p_i p_j`. This is a log-linear
boost on the independent product — **not a probability, unbounded above 1** for high
`C` and intermediate `p`. It only ranks candidates inside the beam (final pricing is
the copula, §2.2), so it is a ranking-only defect — but an EV leak: overconfident
pseudo-EV can promote weak candidates into the `_BEAM_WIDTH=1000` beam and crowd out
better ones. Disposition: bound or replace with the true pairwise copula CDF — parked,
scoped into parlay-dependence stage 3 (gated on D3).

**Final pricing** (`parlay_payout_prob`, `prediction/joint.py`) is a Gaussian copula:
`Φ_n(Φ⁻¹(p); Σ)` via `scipy.stats.multivariate_normal.cdf` — called with **default
tolerances**. At d = 5–6 with tight Σ the integrator is numerically noisy; a 1–2% CDF
error is a 3–6% EV error at flex payouts. Disposition: pass explicit `abseps`/`releps`
or route d ≥ 5 through the push-MC path — parked with the beam fix (parlay-dependence
stage 3).

**Correlation estimator verdict.** The Σ entries come from residualized
8-game-rolling Spearman correlations remapped `2·sin(πρ_s/6)` (`_correlate_teams`,
`training/correlate.py`). That remap **is** the principled rank→Gaussian-copula
parameter map, robust to marginals — the estimator is sound, not a hack. The two real
estimator-level items: the <30-overlap credibility weighting shrinks toward **zero**,
biasing against the product's own edge (superseded by design: the R3 copula brief
prescribes two-level hierarchical Fisher-z EB — `docs/archive/researcher_copula_stage0.md`,
model_improvement_track.md §6.11); and Σ is assembled pair-by-pair with no global PSD
projection — §2.2 covers the repair in place.

**Cross-game legs are independent** (block-diagonal Σ; `correlation.py` processes one
game at a time). Verdict: **accepted** for player-only slips — same-day cross-game
player dependence is weak and R3 builds on within-game groups. It becomes wrong for
mixed player×game-line slips; that case is owned by the `dfs-products` lane
(game-line combo brief B3/B5, `docs/archive/researcher_gamelines_stage0.md`).

### 1.2 Same-player guard

The boost zero-out uses substring matching (`n1 in n2 or n2 in n1`) and the enforcing
dedup lives in the beam search (`leg_players` set), not in `C` — two legs on the same
player still get a nonzero correlation entry. Fragile on suffixed names ("Mike
Williams" vs "Mike Williams Jr."). Disposition: move to ID/exact-match guarding —
minor correctness, parked into parlay-dependence stage 3.

### 1.3 Banned-combos usage

`banned_combos.json` holds per-platform `[same-direction, opposite-direction]` pair
payout multipliers keyed by position-market pair, and `0.0` is a hard ban: every Sleeper
entry carries one, and Underdog's NFL, NHL and MLB sections carry a few. A banned pair
zeroes the parlay's product boost, so the `boost <= _MIN_PRODUCT_BOOST` gate in
`_parlay_admissible` (`prediction/parlay.py`) drops it from the beam search. prophecize
publishes the modifiers it priced with to `current_pair_modifiers.parquet`, so the
dashboard slip prices the same pairs and refuses to lock a banned one.

Cap rule: no Underdog value in the committed file exceeds 1.0. Underdog pays the table
entry × the picks' own multipliers × a per-game modifier `m ≤ 1`, and quotes opposite
sides and negatively associated pairs at exactly 1.0
([underdog_api.md §6.8](underdog_api.md#68-entry-slip-pricing)), so a value above 1.0 is
a bonus the app does not pay. `tests/golden/test_correlation_helpers.py` holds the
committed file to the cap. The runtime overlay of captured quotes is merged on top
unchanged, and a capture stores the pair's whole `[same, opposite]` list, so the slot
it did not quote keeps the value it had at the capture.

The map is hand-documented and drifts. Disposition: corrections come from real app
quotes through the Modifiers reconciler (dfs-products brief §3 and §4); the sub-1
values and bans still waiting on an owner quote are listed under I5a-3 in
[honest-receipts.md §6](handoffs/honest-receipts.md).

---

## 2. `beam_search_parlays`

### 2.1 Beam width and configurability

`_BEAM_WIDTH = 1000` is a named module constant (`prediction/parlay.py`) with no
CLI/kwarg surface; tuning requires a source edit. `max_bet_size` is implicit in the
payout-curve length.

### 2.2 Per-step scoring and PSD handling

Per-step ranking is the geometric mean of pairwise EV (§1.1 defect) with floor
`_PARLAY_GEO_MEAN_FLOOR = 1.05`; survivors are re-priced by the copula.

Non-PSD Σ submatrices are always repaired, never dropped: `psd_or_none`
(`prediction/joint.py`) projects any submatrix whose smallest eigenvalue is below
`_PSD_EIG_TOLERANCE` to the nearest PSD matrix with `_nearest_psd` (eigenvalue clip,
unit diagonal restored). Residual: eigenvalue clipping distorts tight high-ρ
submatrices and the distortion is unmeasured — disposition: report repair-distortion
stats as a read-only rider on the production calibration re-run (hygiene-closeout
stage 2).

Pushes and multi-tier (flex) payouts route through a 50K-draw Monte-Carlo classifier
(`expected_payout_with_pushes` in `prediction/payouts.py`, reached through
`parlay_payout_prob`) — 3-way lose/push/win cuts at `norm.ppf` boundaries. Sound.
The Ladders research brief generalizes exactly this classifier to 4-way ordinal rungs
and finds 50K IID draws borderline for rare-deep-tail payout EV, prescribing a Sobol'
pool with common random numbers (`docs/archive/researcher_ladders_stage0.md`); if that
lands for ladders, the same upgrade applies here.

### 2.3 Constraints enforced

Strictly-increasing leg index; same-player dedup (exact + substring); per-step
geo-mean floor; beam width; both-teams rule; boost band; independent books-EV and
model-EV floors; PSD repair; final EV/units floors. The thresholds are named
module-level constants: `_BEAM_WIDTH`, `_PARLAY_GEO_MEAN_FLOOR`, `_MIN_PRODUCT_BOOST`,
`_BOOKS_EV_FLOOR`, `_MODEL_EV_PRECHECK_FLOOR`, `_MODEL_EV_FINAL_FLOOR` and
`_KELLY_UNITS_FLOOR` in `prediction/parlay.py`; `_PSD_EIG_TOLERANCE` and
`_PUSH_PROB_FLOOR` in `prediction/joint.py`; `_PUSH_MC_SAMPLES` in
`prediction/payouts.py`.

### 2.4 Payout source

Payout curves load from `data/config/underdog_payouts.json` via `payout_curve_for`
(`prediction/payouts.py`) with contest variants (power / flex; pooled default). Search
and display read the same curve.

The Underdog table holds the live terms the owner confirmed in the app in October 2026:
Power for 2–8 picks and the Flex tiers Underdog offers (numbers and capture:
[underdog_api.md §6.8](underdog_api.md#68-entry-slip-pricing)). The pooled variant
prices each size on the schedule an entry of even-money picks beats at the lower
per-pick hit rate, which is Power at 2–3 picks and Flex at 4–6, and a Flex tier with
`k` losses multiplies only the `n − k` largest pick multipliers. The table runs to
8 picks; the engine builds entries of at most 6 (`UNDERDOG_FLEX_CAP` in
`prediction/payouts.py`).

The tables are still **static config**: the app can change payouts mid-season and
nothing detects it, so the per-season re-verify stays hygiene-closeout stage 3's
recurring check. Sleeper's constants (`data/config/sleeper_payouts.json`) belong to
sleeper-parity. Pointers only — no new item here.

### 2.5 Output ranking and dedup

`_append_parlay_rows` (`prediction/correlation.py`) keeps the top
`_PARLAY_TOP_N_PER_SORT` candidates by each of Model EV, Rec Bet and Fun and dedups the
union on exact `Bet ID`; there is no family clustering. Overlap-aware selection
remains a consumer concern.

### 2.6 Parlay-path Kelly sizing

`Rec Bet` is a shrinkage-aware Kelly score. `_parlay_kelly_units`
(`prediction/parlay.py`) recovers the win probability as `Model EV / payout` (exact on
the analytical path, an approximation on the push-aware Monte-Carlo path) and passes it
to `strategies/kelly.py::kelly_edge` with the slip's smallest per-leg shrinkage weight;
`kelly_edge` blends the probability with the payout's implied `1 / payout`, trusting the
model by that weight. Each leg's weight is `resolve_market_shrinkage` for its cell
(`resolve_shrinkage` over the cell's training and live Brier skill); a game leg with no
cell keeps 1.0. The edge is multiplied by `DEFAULT_KELLY_FRACTION` and divided by
`MAX_FRACTION_OF_BANKROLL`, so `Rec Bet` is the quarter-Kelly fraction in multiples of
the 0.5% hard cap, not clipped to it. It is a ranking and filter score, not a stake: a
candidate below `_KELLY_UNITS_FLOOR` is dropped, and `Rec Bet` is one of the three sort
keys in §2.5. Pick'em entries are staked by `strategies/kelly.py::fractional_kelly_stake`
(called from `underdog_pickem._row_to_entry`). Sleeper-parity stage 1 (`b9f8a95e`) put
this in place of the old unshrunk inline formula.

---

## 3. Empirical calibration

Open: `src/sportstradamus/scripts/audit_parlay_calibration.py` runs end-to-end on the
day-partitioned `data/runtime/parlay_hist/` (production writes it; the dev box carries
a copy); it recovers each parlay's payout from its stored `Boost`, which already
includes the payout base. The honest-receipts lane's R2 checked the engine against
898,592 resolved production parlays: the selected set realizes 0.08 of priced, and the
pricer is sound while the admitted legs read above their hit rates
([honest-receipts §7](handoffs/honest-receipts.md#7-records-not-tasks),
[`archive/researcher_parlay_engine.md`](archive/researcher_parlay_engine.md)). The script
has two customers: the standing re-run vehicle is **hygiene-closeout stage 2**
(owner-assisted, on the parlay history `scripts/sync_from_prod.sh` brings over), and its
populated artifacts double as the **incumbent baseline** parlay-dependence stage 4 must beat.
One rider on that re-run: report PSD-repair distortion stats (§2.2) alongside the
reliability deciles. Keep the script's `PAYOUT_FLOOR` / `PAYOUT_CEIL` in step with
`PAYOUT_CLIP_LO` / `PAYOUT_CLIP_HI` in `prediction/payouts.py`, or the inverse
`Model EV → joint_p` recovery silently drifts.

---

## 4. Dispositions

**Overall verdict.** The engine core is right: the Spearman→Gaussian-copula remap is
the principled estimator; the analytical-MVN / push-MC pricing split is correct; PSD
repair is on; payout curves are variant-aware config; every offer row is priced at its
own line (alt lines included — `model_prob` evaluates the CDF per-row, and the
`(Player, Market)` dedup is archive-write-only, not a scoring-path collapse). The real
exposure is operational, not mathematical: on production outcomes the selected parlays
realize **0.08 of their priced probability** (§3); candidate ranking runs on an
**unbounded pseudo-EV** (§1.1); and the credibility weighting **shrinks toward zero** against the product's
own edge (§1.1, already sentenced by R3). The copula-on-PIT rebuild (parlay-dependence
lane, gated on D3) remains the right structural upgrade — nothing in this audit
justifies unparking it early.

| Weakness | Verdict | Actionable now? | Recorded where |
|---|---|---|---|
| MVN cdf default tolerances at d=5–6 (§1.1) | real, cheap | parked behind D3 | here + parlay-dependence §6 stage-3 scope |
| Beam heuristic unbounded (§1.1) | ranking-only EV leak | parked | here + same stage-3 scope |
| Cross-game ρ=0 (§1.1) | accepted for player-only slips | mixed-slip case → dfs-products | here + dfs-products brief |
| PSD-repair distortion (§2.2) | drop-bias fixed; distortion unmeasured | YES — read-only rider on production re-run | hygiene-closeout stage 2 + here |
| Payout staleness (§2.4) | Underdog table at the owner-confirmed live terms (October 2026); still static config | recurring per-season re-verify | hygiene stage 3 + sleeper-parity stage 0 |
| Substring same-player guard (§1.2) | minor correctness | parked | here + parlay-dependence stage-3 scope |
| Shrink-to-zero credibility (§1.1) | superseded by R3 hierarchical Fisher-z EB | resolved-by-design pending D3 | R3 brief / model track §6.11 |
| Production calibration (§3) | R2 ran it (0.08 of priced); script = re-run vehicle + D3 stage-4 baseline | YES (owner-assisted) | hygiene-closeout stage 2 + here |
| banned_combos drift (§1.3) | hand-documented; Underdog values capped at 1.0; `0.0` bans enforced in beam search + live slip | ongoing — owner quotes via the Modifiers reconciler | here + dfs-products §3/§4 |
| ρ not line-stratified | tail-dependence question | no new item — R3 t-branch test gates it | R3 brief |
