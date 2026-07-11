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

**Pairwise EV matrix** (`correlation.py:237-249`):

```
EV[i, j] = exp(C[i,j] * V[i,j]) * P[i,j] * boosts[i] * M[i,j] * boosts[j] * payouts[0]
```

with `V[i,j] = sqrt(p_i(1-p_i) p_j(1-p_j))`, `P[i,j] = p_i p_j`. This is a log-linear
boost on the independent product — **not a probability, unbounded above 1** for high
`C` and intermediate `p`. It only ranks candidates inside the beam (final pricing is
the copula, §2.2), so it is a ranking-only defect — but an EV leak: overconfident
pseudo-EV can promote weak candidates into the `_BEAM_WIDTH=1000` beam and crowd out
better ones. Disposition: bound or replace with the true pairwise copula CDF — parked
behind sleeper-parity (conflict file), scoped into parlay-dependence stage 3.

**Final pricing** (`parlay.py:352`) is a Gaussian copula: `Φ_n(Φ⁻¹(p); Σ)` via
`scipy.stats.multivariate_normal.cdf` — called with **default tolerances**. At
d = 5–6 with tight Σ the integrator is numerically noisy; a 1–2% CDF error is a 3–6%
EV error at flex payouts. Disposition: pass explicit `abseps`/`releps` or route d ≥ 5
through the push-MC path — parked with the beam fix (same file).

**Correlation estimator verdict.** The Σ entries come from residualized
8-game-rolling Spearman correlations remapped `2·sin(πρ_s/6)`
(`training/correlate.py:867`). That remap **is** the principled rank→Gaussian-copula
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

`banned_combos.json` is a soft-modifier table (`[same_side, opp_side]` multipliers
keyed by frozenset of position-markets); no entry ≤ 0.7, so nothing is hard-banned
today — the `boost <= _MIN_PRODUCT_BOOST` gate (`parlay.py:325`) is a back-door ban no
pair triggers. The missing piece is **ground truth**: the apps' actual
pairing/rejection rules have never been captured. Disposition: pairing-rule capture is
a `dfs-products` stage-0 acceptance item; enforcement changes wait on it.

---

## 2. `beam_search_parlays`

### 2.1 Beam width and configurability

`_BEAM_WIDTH = 1000` is a named module constant (`parlay.py:66`) with no CLI/kwarg
surface; tuning requires a source edit. `max_bet_size` is implicit in the payout-curve
length.

### 2.2 Per-step scoring and PSD handling

Per-step ranking is the geometric mean of pairwise EV (§1.1 defect) with floor
`_PARLAY_GEO_MEAN_FLOOR = 1.05`; survivors are re-priced by the copula.

Non-PSD Σ submatrices are repaired, not dropped: `_psd_or_none` (`parlay.py:330-336`)
applies `_nearest_psd` eigenvalue-clip repair by default (`legacy=True` reproduces the
old drop behavior for historical comparison). Residual: eigenvalue clipping distorts
tight high-ρ submatrices and the distortion is unmeasured — disposition: report
repair-distortion stats as a read-only rider on the production calibration re-run
(hygiene-closeout stage 2).

Pushes and multi-tier (flex) payouts route through a 50K-draw Monte-Carlo classifier
(`_expected_payout_with_pushes`, `parlay.py:219-290`) — 3-way lose/push/win cuts at
`norm.ppf` boundaries. Sound. The Ladders research brief generalizes exactly this
classifier to 4-way ordinal rungs and finds 50K IID draws borderline for
rare-deep-tail payout EV, prescribing a Sobol' pool with common random numbers
(`docs/archive/researcher_ladders_stage0.md`); if that lands for ladders, the same
upgrade applies here.

### 2.3 Constraints enforced

Strictly-increasing leg index; same-player dedup (exact + substring); per-step
geo-mean floor; beam width; both-teams rule; boost band; independent books-EV and
model-EV floors; PSD repair; final EV/units floors. The thresholds are named
module-level constants in `parlay.py` (`_BEAM_WIDTH`, `_PARLAY_GEO_MEAN_FLOOR`,
`_MIN_PRODUCT_BOOST`, `_PUSH_PROB_FLOOR`, `_PUSH_MC_SAMPLES`,
`_KELLY_BANKROLL_FRACTION`, `_PSD_EIG_TOLERANCE`).

### 2.4 Payout source

Payout curves load from `data/config/underdog_payouts.json` via `_payout_curve_for`
(`parlay.py:128-195`) with contest variants (power / flex / insurance / rivals; pooled
default); the old hardcoded search table and display-time `Boost` overwrite survive
only behind `legacy=True` (`correlation.py:88-99`, default off), so search and display
use the same regime on the current path.

Open items: the tables are **static config** — the app can change payouts mid-season
and nothing detects it (per-season re-verify = hygiene-closeout stage 3); Sleeper's
curve is a `[1.0, 1.0]` placeholder, so Sleeper Model EV degenerates to the raw joint
probability (owned by sleeper-parity stage 0). Pointers only — no new item here.

### 2.5 Output ranking and dedup

Three-way sort (Model EV / Rec Bet / Fun), exact bet-id dedup, Ward-linkage `Family`
clustering without one-per-family enforcement. Overlap-aware selection remains a
consumer concern.

### 2.6 Parlay-path Kelly sizing

The parlay path sizes stakes inline: `units = (p − 1) / (payout − 1) /
_KELLY_BANKROLL_FRACTION` (`parlay.py:416`) — no shrinkage, no live/training Brier
blending, no per-leg cap. The repo already owns the correct machinery:
`strategies/kelly.py::fractional_kelly_stake` (quarter-Kelly, `resolve_shrinkage`
blend, 0.5% cap) — but only the pick'em path uses it. **This is the highest
live-money finding of the audit**: every parlay recommendation is sized from the raw
copula probability with no model-confidence discount, exactly the overbetting failure
mode the shrinkage machinery exists to prevent.

Disposition — **owner routing decision required** (recorded here, deliberately not
added to any lane's scope unilaterally): the fix is small but `parlay.py` is
conflict-gated (roadmap §5.1). Options: (a) land it inside sleeper-parity's parlay.py
rebuild, (b) fold it into parlay-dependence stage 3, (c) authorize a surgical
standalone PR ahead of both (accepting the file-conflict exception). Recommendation:
(a) if sleeper-parity is scheduled soon, else (c).

---

## 3. Empirical calibration

Open: `scripts/audit_parlay_calibration.py` runs end-to-end but dev checkouts have no
resolved parlay history (`data/parlay_hist.parquet` is production-only), so local
artifacts are placeholders. **This remains the only empirical check of the whole
engine** — and it has two customers: the standing re-run vehicle is
**hygiene-closeout stage 2** (owner-assisted, production host), and its populated
artifacts double as the **incumbent baseline** parlay-dependence stage 4 must beat.
One rider on that re-run: report PSD-repair distortion stats (§2.2) alongside the
reliability deciles. Keep the script's hand-mirrored payout table in sync with
`_payout_curve_for` or the inverse `Model EV → joint_p` recovery silently drifts.

---

## 4. Dispositions

**Overall verdict.** The engine core is right: the Spearman→Gaussian-copula remap is
the principled estimator; the analytical-MVN / push-MC pricing split is correct; PSD
repair is on; payout curves are variant-aware config; every offer row is priced at its
own line (alt lines included — `model_prob` evaluates the CDF per-row, and the
`(Player, Market)` dedup is archive-write-only, not a scoring-path collapse). The real
exposure is operational, not mathematical: the engine has **never been validated on
production outcomes** (§3); the **parlay path bypasses the Kelly-shrinkage machinery**
(§2.6); candidate ranking runs on an **unbounded pseudo-EV** (§1.1); and the
credibility weighting **shrinks toward zero** against the product's own edge (§1.1,
already sentenced by R3). The copula-on-PIT rebuild (parlay-dependence lane, gated on
D3) remains the right structural upgrade — nothing in this audit justifies unparking
it early.

| Weakness | Verdict | Actionable now? | Recorded where |
|---|---|---|---|
| MVN cdf default tolerances at d=5–6 (§1.1) | real, cheap, conflicted file | parked behind sleeper-parity/D3 | here + parlay-dependence §6 stage-3 scope |
| Beam heuristic unbounded (§1.1) | ranking-only EV leak | parked | here + same stage-3 scope |
| Cross-game ρ=0 (§1.1) | accepted for player-only slips | mixed-slip case → dfs-products | here + dfs-products brief |
| PSD-repair distortion (§2.2) | drop-bias fixed; distortion unmeasured | YES — read-only rider on production re-run | hygiene-closeout stage 2 + here |
| Payout staleness (§2.4) | known; homes exist | already homed | hygiene stage 3 + sleeper-parity stage 0 |
| Parlay-path Kelly no-shrinkage (§2.6) | **highest live-money finding** | small fix, conflict-gated → owner routing decision | here §2.6 |
| Substring same-player guard (§1.2) | minor correctness | parked | here + parlay-dependence stage-3 scope |
| Shrink-to-zero credibility (§1.1) | superseded by R3 hierarchical Fisher-z EB | resolved-by-design pending D3 | R3 brief / model track §6.11 |
| Production calibration never run (§3) | only empirical check + D3 stage-4 baseline | YES (owner-assisted) | hygiene-closeout stage 2 + here |
| banned_combos soft-only (§1.3) | product-rule ground truth missing | capture → dfs-products stage 0 | here + dfs-products stage 0 |
| ρ not line-stratified | tail-dependence question | no new item — R3 t-branch test gates it | R3 brief |
