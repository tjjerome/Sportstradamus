# Stage-0 Research Brief — Underdog Ladders: pricing + decision-engine design

> **Plan-mode note.** Plan mode restricts edits to this plan file. This file **is** the
> stage-0 research brief in full; per the lane convention (cf. the R3 copula brief header),
> it is the plan-mode-permitted location. On acceptance, copy this content **verbatim** to
> `/tmp/researcher_ladders_stage0.md` — that path is the deliverable named in the dispatch.
> No production code is edited by this brief.

Statistician's brief for a new `dfs-products` lane (Underdog Ladders). Read-only session.
Repo facts verified this session against: `prediction/parlay.py` (60–453), `prediction/correlation.py`
(full), `training/correlate.py` (40–59, 695–718, 855–879), `helpers/archive.py` (108–124, 900–934),
`strategies/kelly.py` (1–215), `data/config/underdog_payouts.json`, `/tmp/researcher_copula_stage0.md`
(R3), `docs/handoffs/parlay-dependence.md` §1–3, and **a live read-only query of the `ladder`
DuckDB table** (16,889,205 rows; see the coverage census in A7). Product rules web-verified against
Underdog help + rotogrinders; unverifiable items tagged **VERIFY** with the exact stage-0 capture that
must confirm them.

Date: 2026-07-10.

---

## 0. Verdict summary (the design in one breath)

Build Ladders as **two new modules under `strategies/`** (`strategies/ladders.py` = pricer +
candidate builder, rhyming with `strategies/underdog_pickem.py`) **plus a validation script under
`scripts/`** (`scripts/audit_ladder_calibration.py`), all **importing** `prediction/parlay.py`'s
copula/PSD machinery and `prediction/correlation.py`'s Σ assembly but editing **neither** (roadmap
§5.1 file-conflict lock). Price each entry as a **d∈{3,4,5}-pick joint of independent 4-way ordinals**
(fail / r1 / r2 / r3 per pick) cut from **one latent Gaussian per pick** at `norm.ppf` of the model's
**survival** values at the three rung lines, coupled by the incumbent same-game Gaussian copula Σ. The
priced event `P(lowest rung = r)` is a **monotone-nested orthant** quantity: compute it by a **single
reused Sobol'-QMC draw pool** (one d-dim standard-normal QMC block per game, Cholesky-transformed per
candidate) — **not** by inclusion–exclusion of `mvn.cdf` calls, which would need up to `4^d`≈1024
rectangle evaluations per candidate and blow the serve budget. **8,192 Sobol' points per game**, shared
across all candidates in that game, gives ≤±2% relative EV error even on the rare `P(lowest=3)` tail
(A2) and costs <1s/game. **Rung cuts carry ≈zero push mass** because Underdog ladder lines are
half-integers (verified: 5,562,332 of 5,562,336 NBA ladder lines end in .5), so the push-band logic in
`_expected_payout_with_pushes` collapses to the clean 4-band cut — a real simplification over pick'em.
Size with a **1-D numeric Kelly solve** over the ≤4 non-zero payout outcomes, wrapped in the
`fractional_kelly_stake` conventions (quarter-Kelly × shrinkage, 0.5% cap). **Dependence:** default
Gaussian per R3; add a **rung-depth-stratified** re-test of R3's t-branch on the `ladder` table (adopt t
only if the deep-rung joint-exceedance Spearman clears the R3 band **and** ΔAIC≥10 with ν̂≤15). Selection
pre-filter: a **cheap scalar "deep-rung edge" score** (Σ model-vs-book log-survival gap at r2/r3, penalized
by adverse copula sign) ranks a filtered pool before exact pricing (two-stage, budget-aware). If Ladders
prove app-UI-only with no API (**VERIFY**), ship a **manual-entry mode**: owner keys picks + 3 rung lines +
the payout row; the same pricer/sizer runs. This is an **engineering project** (known methods: MVN
rectangle probabilities, QMC, discrete-outcome Kelly, the repo's own copula), **not** a research bet — the
only genuinely open empirical question is the deep-rung t-copula trigger (A4), and it has a concrete
adopt/reject gate runnable today on 1.98M archived ladder offers.

---

## A1. Exact joint formulation of P(lowest rung = r)

**Setup.** Entry has `d ∈ {3,4,5}` active picks. Pick `i` has three ascending rung lines
`ℓ_{i,1} < ℓ_{i,2} < ℓ_{i,3}` (fixed by the app — **VERIFY-1**). From the calibrated model CDF `F_i`
for that cell, define the three **survival** probabilities at the rungs:

```
s_{i,k} = P(Y_i ≥ ℓ_{i,k})           k = 1,2,3     (reach recipe in A3; ≥ vs > → VERIFY-2)
```

Ascending lines ⇒ `s_{i,1} ≥ s_{i,2} ≥ s_{i,3}` (survival is non-increasing in the line; enforce
monotonicity per A3). Each pick's outcome is a **4-way ordinal** `R_i ∈ {0(fail),1,2,3}`:

```
P(R_i = 0) = 1 − s_{i,1}       (fails rung 1)
P(R_i = 1) = s_{i,1} − s_{i,2} (reaches r1 not r2)
P(R_i = 2) = s_{i,2} − s_{i,3}
P(R_i = 3) = s_{i,3}
```

**Latent-Gaussian representation (matches `parlay.py:_expected_payout_with_pushes` at 249–262).**
Introduce one standard-normal latent `Z_i` per pick with `Corr(Z) = Σ` (the same-game copula matrix from
`correlation.py`; `Σ = I` across different games). Three ascending **cut points** partition the real line:

```
c_{i,1} = Φ⁻¹(1 − s_{i,1})   (below ⇒ fail)
c_{i,2} = Φ⁻¹(1 − s_{i,2})
c_{i,3} = Φ⁻¹(1 − s_{i,3})    c_{i,1} ≤ c_{i,2} ≤ c_{i,3}
R_i = #{k : Z_i ≥ c_{i,k}}    (0,1,2,3 = how many cuts cleared)
```

This is the exact 4-way generalization of the incumbent's 3-band LOSS/PUSH/WIN cut
(`parlay.py:254–262`), where the incumbent's two cuts become three and the classification counts
cleared cuts instead of banding.

**The priced quantity.** Let `L = min_i R_i` (Underdog keys the entry payout to the **lowest** rung
reached across **all** picks — verified, see Sources). The full law is a **nested-orthant** family. Using
the model-survival-monotone tail events `A_r = {all picks reach at least rung r} = {min_i R_i ≥ r}`:

```
P(L ≥ r) = P( Z_i ≥ c_{i,r}  ∀ i )       = Φ_Σ,d( lower = c_{·,r}, upper = +∞ )   (an upper orthant)
P(L = r) = P(L ≥ r) − P(L ≥ r+1)          r = 1,2 ;   P(L = 3) = P(L ≥ 3)
P(L = fail) = P(L ≥ 1)ᶜ = 1 − P(L ≥ 1)
```

So the **entire** payout law reduces to **three upper-orthant probabilities** `P(L≥1), P(L≥2), P(L≥3)`
of the *same* `d`-dim Gaussian with the *same* Σ, evaluated at three different lower-corner vectors
`c_{·,1}, c_{·,2}, c_{·,3}`. **No inclusion–exclusion over `4^d` cells is needed** — the min-structure
collapses to a survival-difference of three orthants. (This is the key structural simplification; a naïve
"enumerate all `4^d` ordinal cells and sum" is 64–1024 evaluations, the nested form is 3.)

**Analytical vs Monte Carlo at d=3–5 — verdict: reused-QMC Monte Carlo.**

- **Analytical (`scipy.stats.multivariate_normal.cdf`, Genz 1992 QMC under the hood).** An upper orthant
  is one `mvn` call. Three orthants per candidate → 3 `mvn.cdf` calls. `scipy`'s `mvn.cdf` at d=5 with
  default `abseps` is ~1–5 ms **per call** but is **not vectorized across candidates** and re-derives its
  own internal QMC lattice each call — at ~hundreds of candidate ladders per game that is 3×hundreds×~3ms
  ≈ seconds-to-tens-of-seconds **per game**, times ~10 games. Under the 15-min-max budget (A2/compute
  section) this is the wrong default: it pays the QMC setup cost once **per candidate** instead of
  amortizing it.
- **Reused-QMC Monte Carlo (recommended).** Draw **one** `(N × d_max)` Sobol'-QMC block of standard
  normals **per game** (d_max=5; slice the first `d` columns for smaller entries). For each candidate:
  Cholesky `L_Σ` of its Σ (cheap, d≤5), transform `X = draws @ L_Σᵀ`, classify `R_i` by the three cuts
  (fully vectorized `searchsorted`/comparison, exactly `parlay.py:258`), take `L = X_ord.min(axis=1)`,
  and read the payout. **The expensive object (the QMC point set) is built once per game and reused across
  every candidate**, so the per-candidate cost is a d×d Cholesky + an `(N×d)` matmul + a min — microseconds
  at N=8,192, d≤5. This is strictly the incumbent MC pattern (`parlay.py:248–290`) with (a) 4 bands not 3,
  (b) Sobol' not IID, (c) the draw pool hoisted out of the candidate loop. **Accuracy trade vs analytical:**
  QMC at N=8,192 gives orthant-probability error ~1e-3–1e-4 absolute (Genz-transform QMC error scales
  ~`log(N)^d/N`, which is *better* at low d — see A2 and Sources), well inside the ±2% EV gate; the only
  loss vs exact `mvn.cdf` is in the deepest tail, handled by the N-sizing in A2.

**Recommendation A1.** Price via the **three-orthant nested-survival form** above, evaluated by a
**per-game reused Sobol' draw pool**. Do **not** call `mvn.cdf` per candidate for the full price. *Do*
keep `mvn.cdf` available for one narrow use: **pre-filter bounds** (A6) — a single `mvn.cdf(P(L≥3))` on a
*candidate-agnostic* independence proxy is a cheap upper bound to reject no-hope ladders before the QMC
classify runs.

---

## A2. Monte-Carlo error analysis for payout-weighted EV

**The hard part.** `EV = Σ_r P(L=r)·m_r` where `m_3` is large (Pick-3 r3 = 25×, Pick-4 = 100×, Pick-5 =
250× — verified table, A5) and `P(L=3)` is small (all `d` picks simultaneously deep). The EV is
**tail-dominated**: the `P(L=3)·m_3` term is where both the money and the MC variance concentrate. A
relative-error target on total EV is therefore effectively a relative-error target on the small
probability `P(L=3)`.

**Target (proposed gate).** **±2% relative error on entry EV at the 95% level**, i.e.
`SE(ÊV)/EV ≤ 0.01` (2σ ≈ 2%). Because `m_3` dominates, this is approximately
`SE(P̂(L=3))/P(L=3) ≤ ~0.01–0.02` for the deep tail (looser when lower rungs carry material EV mass).

**IID MC sizing (baseline).** For a Bernoulli-like tail event of probability `p₃ = P(L=3)`, IID MC
gives `SE(p̂₃) = √(p₃(1−p₃)/N) ≈ √(p₃/N)`, so relative SE `= √((1−p₃)/(N p₃)) ≈ 1/√(N p₃)`. For a
representative deep NBA ladder, `p₃` is on the order of `10⁻²`–`10⁻³` (three correlated picks each with
r3 survival ~0.2–0.35 — cf. the Al Horford PTS ladder in A7: s at 9.5 ≈ 0.32; `0.32³ ≈ 0.033`
independent, less under positive dependence... actually **more** in the joint deep tail under positive
copula, ~0.04–0.06). Take `p₃ ≈ 0.03` as typical, `p₃ ≈ 0.005` as a stress case:

| p₃ (deep-hit prob) | N for rel-SE ≤ 2% (IID) | N for rel-SE ≤ 2% (Sobol', ~4–10× fewer) |
|---|---|---|
| 0.05 | ~47,500 | ~5,000–12,000 |
| 0.03 | ~80,800 | ~8,000–20,000 |
| 0.01 | ~247,500 | ~25,000–60,000 |
| 0.005 | ~497,000 | ~50,000–120,000 |

So **the incumbent 50,000 IID draws are borderline-to-inadequate** for the deepest ladders (rel-SE ≈
2–3% at p₃=0.03, ≥4% at p₃=0.005) — acceptable for a *display* EV, **not** for a Kelly stake on a 250×
payout where a 4% EV error is a meaningful mis-size. **Fix without raising wall-time: variance reduction.**

**Recommended variance-reduction stack (in priority order):**

1. **Randomized QMC (Sobol' with a Genz-style transform), CRN across candidates.** The single biggest win
   and already the incumbent's spiritual path. Sobol' converges ~`log(N)^d/N` vs `N^{-1/2}` for IID; at
   d≤5 the empirical gain is typically **4–10× fewer points** for the same accuracy (Genz 1992; l'Ecuyer
   RQMC). **Reusing one scrambled Sobol' block as Common Random Numbers across all candidates in a game**
   also makes candidate-to-candidate EV *differences* far lower-variance than their levels — which is
   exactly what the selection ranking (A6) needs. **N = 8,192** (2¹³, a natural Sobol' length) is the
   recommended default: it lands at rel-SE ≤ ~2% down to p₃ ≈ 0.01 and ≤ ~3% at p₃ ≈ 0.005, and costs
   ~0.6 ms/candidate for the transform+classify at d=5 (A-compute).
2. **Stratification on the tail latent (optional, if deep ladders dominate the book).** Because `L=3`
   requires the *min* latent to be large, stratify the draw of `min_i Z_i` — or more simply, importance-
   sample by shifting the sampling mean of `Z` toward the deep-tail corner and reweighting. A mean-shift
   importance sampler targeting the `A_3` orthant can cut the deep-tail variance another 3–10× (standard
   rare-event MC), but it **complicates the reused-pool trick** (the shift is candidate-specific). **Defer
   to stage-2** unless the ±2% gate fails at N=8,192 on the deep-ladder decile in the A7 harness.
3. **Antithetic pairing** (`±Z`): trivially free with Sobol' scrambling, modest gain on the symmetric part
   of the integrand; include it, don't rely on it.

**Interaction with the budget.** The reused-pool design means **N scales the per-game cost once, not the
per-candidate cost** (the Sobol' generation is `O(N·d)` once per game; each candidate is `O(N·d)` for the
matmul but with a tiny constant). Doubling N from 8,192 to 16,384 roughly doubles the matmul cost but
leaves the per-candidate constant and the Sobol' setup ~linear — a heavy NBA day stays well under budget
either way (A-compute). **Do not** raise N by looping IID draws per candidate (the incumbent structure);
that multiplies N by the candidate count and *is* what would blow 15 minutes.

**A2 verdict.** Adopt **N = 8,192 scrambled Sobol' per game, CRN-reused across candidates, antithetic on**.
Gate the choice empirically in A7: on the deep-ladder decile, `sd of ÊV across 20 independent scrambles ≤
2% of mean ÊV`. If it fails there (very high-payout, very deep books), enable the mean-shift importance
sampler for the `A_3` term only, priced separately and added back.

---

## A3. Per-pick cut construction from the calibrated model CDF

**Survival recipe.** For pick `i`, cell `(league, market)`, calibrated predictive CDF `F_i` (the same
object the ship gates certify — model_improvement_track §1: "Gate-4 PIT-KS calibration **is** alt-line
pricing accuracy"; the rung price *is* the alt-line price the PIT gate audits):

- **Continuous cells (Gamma, ZAGamma, SkewNormal — the `global_mean ≥ 2` branch):**
  `s_{i,k} = 1 − F_i(ℓ_{i,k})`. No push term (continuous ⇒ `P(Y = ℓ) = 0`), and lines are half-integer
  anyway. Use the **serving decode path**, not a re-implementation — offset/normalization decode drift
  (offset_mode, `MeanYr_nonzero` denominators, ZI decode) has bitten this repo repeatedly; the rung pricer
  must call the identical decode the prediction pipeline uses (mirror how `model_prob` builds `Win Prob`).
- **Count cells (NegBin, ZINB — the `global_mean < 2` branch):** with a **half-integer line** `ℓ = n+0.5`
  there is no push mass and `≥` vs `>` is moot: `P(Y ≥ n+0.5) = P(Y ≥ n+1) = 1 − F_i(n)`. This is the
  common case and is clean. **If** a rung line is ever an **integer** `ℓ = n` (should not happen on
  Underdog — VERIFY-2/VERIFY-3), then push mass `P(Y = n)` is real and the reach semantics matter:
  - reach `≥`: `s = P(Y ≥ n) = 1 − F_i(n−1)`, and the exact-tie value `Y=n` counts as reached;
  - reach `>`: `s = P(Y > n) = 1 − F_i(n)`, exact tie fails;
  - **Underdog's own pick'em rule is that an exact-line result VOIDS the leg** (verified: "if the total is
    exactly equal to the offered line … all selections on this offer will be declared void"). For a ladder
    rung this most likely means the pick is **rescued/voided at that rung boundary**, not counted
    over/under (VERIFY-3). Model it by removing the tie mass symmetrically: treat `Y=n` as a *rung-void*
    that drops the pick to its next-lower resolved rung — but because lines are half-integer this branch is
    **dead code in practice**. Capture a real integer-line ladder in stage-0 before writing it.

**Push mass — verified near-zero.** Live query: **5,562,332 / 5,562,336 NBA ladder lines end in `.5`**
(4 stragglers are data noise). So the `p_push` band that forces `parlay.py`'s MC path is **structurally
absent for rung cuts** — the 4-band cut is exact with `p_push ≡ 0`. (Keep a `p_push` hook for the integer-
line contingency, defaulted to 0, mirroring `correlation.py:217`'s "fill 0.0 so the analytical path runs.")

**Monotonicity enforcement.** The three survivals **must** satisfy `s_{i,1} ≥ s_{i,2} ≥ s_{i,3}` for the
cut points to be ordered; a mis-calibrated or noisy CDF can violate it near the tail, and — separately —
if you ever borrow **book** rung prices (`ladder.p_over`), the de-vigged book ladder can be non-monotone
across rungs (book noise). Two guards:

1. **Model side:** because all three survivals come from the *same* fitted `F_i`, monotonicity is
   guaranteed by construction as long as `ℓ_{i,1}<ℓ_{i,2}<ℓ_{i,3}` and `F_i` is a proper CDF — no repair
   normally needed. Assert it; if it fails, the decode is wrong (fail loud, per house rules — do not paper
   over with a clamp).
2. **Book side (only if used as a prior/anchor):** enforce isotonic non-increasing survival across rungs
   via `sklearn.isotonic.IsotonicRegression(increasing=False)` on `(line, p_over)` before use — the same
   whole-CDF isotonic-PIT discipline the repo already uses (memory: "Rung C whole-CDF recal"). This is a
   *book-price cleanup*, not a model change.

**≥ vs > (VERIFY-2).** Rotogrinders and Underdog marketing use "**200+ passing yards**" (the `+` glyph =
"at or above"), and half-integer lines make it moot for the priced quantity. Proceed with **`s = 1 − F(ℓ)`
at half-integer `ℓ`** (which equals both `P(Y≥ℓ)` and `P(Y>ℓ)` when `ℓ∉ℤ`). Register VERIFY-2 to confirm
the glyph semantics on a real slip and VERIFY-3 to confirm no integer rung lines exist.

---

## A4. Dependence at depth — does Gaussian bias joint deep-rung probabilities?

**The concern, stated precisely.** The priced deep event `A_3 = {all d picks reach rung 3}` is a **joint
upper-tail** event: every pick's latent `Z_i` simultaneously large. A Gaussian copula has **zero tail
dependence** (`λ_U = 0` for any ρ<1; Demarta & McNeil 2005, cited in R3), so if the *true* same-game
dependence has upper-tail dependence (e.g. an NBA blowout where every starter's minutes/usage co-spike, or
a shootout inflating every pass-catcher), the Gaussian copula **understates** `P(A_3)` and therefore
**understates** the 250×-payout probability — a systematic *under*-pricing of the best outcome, i.e. we'd
leave the sharpest ladders under-bet. This is the one place Ladders is more tail-sensitive than pick'em
(pick'em's priced event is a single orthant at moderate p; Ladders' money is in the *deep* orthant).

**R3's verdict and how this extends it.** R3 concluded: **default Gaussian**; adopt **t** only if a
two-part gate fires — (i) pooled joint-**exceedance** Spearman above the Gaussian-null band in **both**
tails across a majority of pair-types, **and** (ii) pooled-MLE `ΔAIC ≥ 10` with `ν̂ ≤ 15` on `N ≥ 2,000`
pairs, one ν per league. Ladders **inherits that verdict unchanged for the pairwise fit**, and **adds a
depth-stratified re-test** because the priced event lives specifically in the joint tail, where R3's
*full-pool* Spearman is least informative.

**Recommended rung-depth-stratified re-test (runnable today on the `ladder` table; design only).**
The `ladder` table gives, for 1.98M (entity, game_date, market) offers across 5 leagues (A7 census), the
**exact archived rung lines** books used, with de-vigged `p_over` per rung and — after joining the league
gamelog on (entity, game_date) — the **realized** outcome `Y`. That yields, per historical would-be pick,
its realized rung `R∈{0,1,2,3}` at book-set lines. Then, within same-game groups (≥2 picks sharing a
`game_date` and opposing/same team, keyed exactly like `correlation.py`'s Σ):

1. **Empirical deep-orthant hit rate vs Gaussian prediction.** For each realized same-game **pair** of
   ladder picks, form PITs `U = F̂(Y)` through the *model* marginals (leak-free re-score on the archived
   date, per R3's PIT extraction discipline). Bin by **rung depth** `q ∈ {r1-line, r2-line, r3-line}`
   (i.e. the survival level the rung sits at). Compute the **joint-exceedance Spearman** on the subset
   `{both U ≥ 1−s_{r}}` for `s_r` = the r2 and r3 survival levels — this is R3's exceedance test **but
   evaluated at the actual rung-depth quantiles**, not generic q∈{0.75,0.9}. Simulate the Gaussian-copula
   null (500 draws at the full-pool ρ) for a 95% band at each depth.
2. **Numeric adopt/reject gate (extends R3):**
   - **REJECT t (keep Gaussian):** deep-rung (r3-level) empirical joint-exceedance Spearman **inside** the
     Gaussian null band in ≥ half of the top-10 same-game pair-types per league. *Expected outcome*,
     stated for the ledger: Gaussian survives (post-shrinkage same-game |ρ| is mostly 0.05–0.4; at ρ=0.3,
     ν=10, λ≈0.03 — immaterial at the pair level), **except possibly the NBA blowout/pace regime**, which
     is the one plausible real joint-tail source (R3 flagged the same regime).
   - **ADOPT t (as a tested branch, not default):** deep-rung empirical Spearman **above** the band in
     **both** tails for a **majority** of tested pair-types in a league **AND** R3's confirmatory pooled
     `ΔAIC ≥ 10` with `ν̂ ≤ 15` on that league's `N ≥ 2,000` pooled pairs. Then price *that league's* deep
     ladders with a **t-copula** (one ν/league), sampled by scaling the reused Sobol' normals by a shared
     `√(ν/χ²_ν)` mixing draw — a **one-line change to the draw block**, not a new abstraction. Shallow
     ladders (money in r1/r2) stay Gaussian; the t-branch only moves the `P(A_3)` term.
   - **Never gate on nonparametric `λ̂`** (Frahm–Junker–Schmidt 2005, per R3): at these pooled n it can't
     separate λ=0 from λ=0.05.
3. **Direct calibration cross-check (the real arbiter).** Independent of the copula-family test, A7's
   reliability harness measures **realized deep-rung hit rate vs model-predicted `P(A_3)`** directly on the
   archive. If predicted `P(A_3)` tracks realized within the A7 band under Gaussian, the family question is
   moot regardless of the Spearman test — **calibration of the priced quantity is the verdict**, the
   copula test only explains *why* if it misses.

**Consistency with R3.** This uses R3's estimator (pooled exceedance-Spearman + hierarchical Fisher-z EB
for the pairwise ρ, `2·sin(πρ_s/6)` remap already in `correlate.py:867`), R3's confirmatory ΔAIC/ν gate,
and R3's Gaussian default. It **extends** R3 by evaluating the exceedance test **at rung-depth quantiles**
(because that is where Ladders' money is) and by adding the **direct deep-orthant calibration** arbiter.
It contradicts R3 nowhere.

---

## A5. Staking — Kelly for a discrete payout vector

**The bet.** One entry, stake fraction `f` of bankroll, payout **multiplier** `m_r` (gross per dollar) on
outcome `r` with probability `π_r = P(L=r)`. Verified Underdog Ladder multipliers (rotogrinders,
2026; a refund = 1× returns the stake, net 0):

| outcome | Pick-3 `m` | Pick-4 `m` | Pick-5 `m` |
|---|---|---|---|
| fail (L=0) | 0 | 0 | 0 |
| rung 1 (refund) | 1 | 1 | 1 |
| rung 2 | 3 | 5 | 10 |
| rung 3 | 25 | 100 | 250 |

(**VERIFY-4:** rotogrinders' "1000x" / the help page's "$1,000 on a $10 entry" is **inconsistent** with a
5-pick r3 = 250× — 250× × $10 = $2,500, and 100× × $10 = $1,000. Either the max multiplier is slate-varying
(**VERIFY-5**) or the marketing example conflates pick counts. Do not hardcode 1000×; **read the per-slip
payout table from the app payload** — the table is shown per-slip, exactly like pick'em, so capture it,
don't assume it.)

**Objective (exact, discrete).** Maximize expected log growth:

```
g(f) = Σ_r π_r · log(1 + f·(m_r − 1))
```

Note `m_1 = 1` (refund) contributes `log(1 + f·0) = 0` — the refund outcome is **growth-neutral**, it
neither helps nor hurts log-growth, it only reduces variance. The fail outcome contributes
`π_0·log(1−f)`.

**Closed form vs numeric.** With >2 non-degenerate outcomes there is **no closed form** in general (the
FOC `Σ_r π_r (m_r−1)/(1+f(m_r−1)) = 0` is a rational equation of degree = #outcomes−1). Solve the **1-D
root** of `g'(f)=0` on `f∈(0, 1/(1−m_min⁺))` — here `f∈(0,1)` since the worst *non-refund* loss is the
fail outcome `m=0` giving the `1/(1−0)=1` upper bound. `g` is **strictly concave** in `f` on `(0,1)`
(sum of concave logs), so `g'` is strictly decreasing → a unique interior root iff `g'(0) = Σ_r π_r(m_r−1)
= EV−1 > 0` (i.e. the entry is +EV). Use **Brent** (`scipy.optimize.brentq` on `g'`) or 30 iterations of
bisection — microseconds. This is the exact analogue of `joint_kelly_portfolio`'s cvxpy log-objective
(`kelly.py:196–204`) specialized to one bet with a discrete outcome vector; a 1-D solve is lighter than
invoking SCS for a scalar and is the right tool.

**Wrap in `fractional_kelly_stake` conventions (`kelly.py:114–159`).** The full-Kelly `f*` from the solve
is then shrunk and capped identically to the incumbent:

```
f_final = min( f* · fraction , MAX_FRACTION_OF_BANKROLL )        # quarter-Kelly × 0.5% cap
```

and the **edge shrinkage** applies to the *probabilities* before the solve, mirroring
`p_eff = 0.5 + (p−0.5)·shrinkage` — but for a multi-outcome vector, shrink the **distribution toward the
book's implied distribution** (or toward the refund-heavy no-edge distribution), not toward 0.5. Concretely,
per-cell `model_shrinkage` from `resolve_shrinkage` (`kelly.py:65`) blends model `π` with the book-implied
`π^book` (from the ladder's own de-vigged rung survivals):
`π_eff = shrinkage·π_model + (1−shrinkage)·π_book`. This keeps the exact `kelly.py` shrinkage/fraction/cap
semantics while generalizing the scalar `p` to the outcome vector. Below `SHRINKAGE_FLOOR` → stake 0
(unchanged).

**Worked numeric example (Pick-3, illustrative).** Suppose a 3-pick ladder prices (after copula) to
`π = (fail 0.45, r1 0.20, r2 0.20, r3 0.15)` with `m = (0, 1, 3, 25)`.
- `EV = 0.45·0 + 0.20·1 + 0.20·3 + 0.15·25 = 0 + 0.20 + 0.60 + 3.75 = 4.55` per dollar → strongly +EV.
- `g'(f) = 0.45·(−1)/(1−f) + 0.20·0/(1+0) + 0.20·2/(1+2f) + 0.15·24/(1+24f)`.
- Solve `g'(f)=0`: at `f=0.05`, `g' = −0.4737 + 0 + 0.3636 + 2.093 = +1.98 > 0`; at `f=0.30`,
  `g' = −0.643 + 0.2439 + 0.2885 = −0.111 < 0`; root ≈ `f* ≈ 0.285` (full Kelly — aggressive because the
  25× term dominates).
- Quarter-Kelly: `0.25·0.285 = 0.071`; **capped at `MAX_FRACTION_OF_BANKROLL = 0.005`** → **stake = 0.5%
  of bankroll**. (The 0.5% cap binds hard on high-payout ladders — as designed; the cap, not the raw
  Kelly, sizes essentially every deep ladder, which is the correct risk posture given the estimate variance
  on a 25×+ tail.)

**A5 verdict.** 1-D concave Brent solve on the discrete-outcome log-growth, `π_book`-blended shrinkage,
then the verbatim `fractional_kelly_stake` fraction+cap. The 0.5% cap will bind on most Ladders; that is
the point.

---

## A6. Selection — what makes a +EV ladder, and the cheap pre-filter

**When a ladder is +EV.** The edge lives where the **model survival at deep rungs** exceeds the **book's
de-vigged survival** at those same rungs, *jointly and correlation-aware*:

1. **Deep-rung marginal gap (primary).** For each pick, `Δ_{i,k} = log s^model_{i,k} − log s^book_{i,k}`
   at k=2,3 (the r2/r3 lines). Positive Δ at deep rungs = model thinks the deep reach is likelier than the
   book's rung price implies = the source of the 3×/25×+ edge. r1 gaps barely matter (r1 only refunds).
2. **Right-tail mass / shape.** A cell whose calibrated predictive has **heavier right tail than the book's
   ladder implies** (model survival decays slower across rungs than the book's `p_over` ladder) is a ladder
   engine. This is directly readable from the archived `ladder.p_over` vs the model CDF at the same lines —
   and is exactly the Gate-4 alt-line-accuracy quantity (model_improvement_track §6.11). Prefer cells with
   `shape_ratio ≈ 1` (well-calibrated dispersion; `model_stats.parquet`) so the deep-rung survival is
   trustworthy — a mis-dispersed cell's r3 survival is the least reliable number in the pipeline.
3. **Correlation sign among picks.** Because the payout is keyed to the **min**, **positive** same-game
   dependence among the chosen picks **helps** the deep event `A_3` (they co-reach) and **helps** avoid the
   fail event (they co-fail together less often than... — careful: positive dependence *raises* both
   `P(all deep)` and `P(all fail)`; it *lowers* the "one drags the min down" middle). Net, for a **deep,
   +EV** ladder you **want positively-correlated picks** (co-reach the deep rung); for a **safety/refund**
   ladder you want low correlation. The selection rule must know which regime it's building for.

**When a Ladder beats the flat parlay of the same picks.** A flat Underdog Power/Flex parlay of the same
`d` legs at a single line pays only on the all-hit (or partial-hit flex) event at *one* threshold. A Ladder
(a) **monetizes the deep tail** (r3 = 25–250× vs flex's ~10–25×) and (b) **refunds** at r1 instead of
busting. A Ladder dominates when the picks have **fat, well-calibrated right tails** and **positive
same-game dependence** (so the joint deep reach is materially more likely than the independence product),
because that is precisely the region a single-line parlay under-monetizes and the min-keyed ladder rewards.
Conversely, when picks are near-independent and tails are thin, the flat parlay's cleaner all-hit event is
better priced — route those to the existing pick'em engine, not Ladders.

**Cheap scalar pre-filter (two-stage, budget-aware).** Before any QMC pricing, rank the *filtered* pick
pool (reuse `correlation.py:_select_bet_offers`'s book-support/model-EV gates, 375–400) by a **deep-rung
edge score** computable in closed form from per-pick survivals + the pairwise ρ already in `g.C`:

```
DeepScore(candidate) =  Σ_i [ w2·Δ_{i,2} + w3·Δ_{i,3} ]                      # summed deep-rung log-gap
                        + λ_corr · mean_{i<j} sign(Δ_deep)·ρ_ij              # reward aligned positive corr
                        − penalty·(1 − min_i s^model_{i,3})                  # penalize any pick with a weak deep reach
```

with `w3 > w2` (r3 pays far more) and `λ_corr` small. This needs **only** the marginal survivals and the
existing pairwise ρ — **no orthant probability** — so it costs `O(d²)` per candidate and pre-ranks
thousands of candidate ladders in milliseconds. **Stage 1:** score all admissible `d∈{3,4,5}` combos with
`DeepScore`, keep the top-K (K≈ `_BEAM_WIDTH`=1000, mirroring `parlay.py:66`). **Stage 2:** exact-QMC-price
only those K with the reused Sobol' pool. This is the same **beam-search-then-exact-evaluate** two-stage
shape as `beam_search_parlays` (`parlay.py:456`), so it inherits a proven budget profile.

**A6 verdict.** Pre-filter by the closed-form `DeepScore` (deep-rung log-survival gap + aligned-correlation
bonus), exact-QMC-price the top-K, size the survivors by A5. Build a candidate class that **rhymes with
`RecommendedEntry`/`underdog_pickem.py`** (YAML emit) — do not fork the parlay beam search.

---

## A7. Validation harness on the `ladder` table

**The asset (live-verified census, read-only query this session):**

| league | ladder rows | markets | players | books | date range | **ladder-shaped offers (≥3 rungs)** |
|---|---|---|---|---|---|---|
| MLB | 7,007,818 | 18 | 1,123 | 9 | 2025-03-28 → 2026-07-09 | 733,416 |
| NBA | 5,562,336 | 12 | 576 | 12 | 2023-10-24 → 2026-06-13 | 659,081 |
| NHL | 3,482,453 | 7 | 1,392 | 12 | 2023-10-10 → 2026-06-14 | 480,695 |
| NFL | 522,687 | 4 | 514 | 8 | 2024-09-05 → 2026-02-08 | 64,487 |
| WNBA | 313,911 | 8 | 152 | 8 | 2025-05-16 → 2026-07-09 | 44,637 |

**≈1.98M archived ≥3-rung offers** with per-rung de-vigged `p_over` from up to 12 books — an enormous,
already-on-disk validation set. Books offer 3–50+ rungs per player-game (distribution: 561,889 at exactly
3 rungs, tapering to 50+); for validation, pick the **3 rungs nearest a product-ladder template** per
offer, or validate at **all** archived rung lines (richer). Lines are 99.99% half-integer → clean cuts.

**Script (design only): `scripts/audit_ladder_calibration.py`** (read-only; click CLI; tqdm; **imports**
the pricer from `strategies/ladders.py`, edits no prediction module). Two evidence tracks + numeric gates:

**Track 1 — rung-probability reliability by depth/league/market (the core gate).**
Join `ladder` (entity, game_date, market, line, p_over) to the league gamelog on (entity, game_date) →
realized `Y` → realized rung outcome at each archived line. For each shipped cell, re-score the model CDF
on the archived date (leak-free, serving decode) → model survival `s^model` at each rung line. Bin by
**survival depth** (r1≈0.5–0.7, r2≈0.3–0.45, r3≈0.15–0.25 bands, per the observed ladder structure — cf.
Al Horford PTS: 0.62/0.47/0.32/0.20 at 5.5/7.5/9.5/11.5). Reliability = `|empirical reach rate −
model-predicted survival|` per (depth-bin, league, market).
- **GO gate (enter):** count-weighted mean `|empirical − model survival|` ≤ **0.03 absolute** at r1/r2 and
  ≤ **0.04** at r3 (deepest, noisiest), per league; **and** no systematic sign (mean signed error within
  ±0.02) — i.e. the model isn't uniformly over/under-stating deep reaches.
- **KILL gate:** if deep-rung (r3) model survival is biased by **> 0.05 absolute** in a consistent
  direction across ≥3 of a league's markets, the marginals aren't deep-tail-calibrated **for ladder
  pricing** → the cell is **ladder-ineligible** (route to model-track dispersion work; this is the Gate-4
  "alt-line accuracy" failure, model_improvement_track §6.11) even if it ships for pick'em.

**Track 2 — model-vs-book rung price, and joint deep calibration.**
- **Marginal:** compare `s^model` to the book's de-vigged `p_over` at each rung (the current single-EV
  dist-inversion vs the real book ladder). Report the **deep-rung edge distribution** `Δ_{·,3}`; a cell
  with a persistent positive median `Δ_{·,3}` that *also* passes Track-1 reliability is a validated ladder
  engine.
- **Joint (the A4 arbiter):** among same-game archived pick pairs/triples, compare **model-predicted
  `P(A_2), P(A_3)`** (Gaussian-copula priced) to the **realized** joint-deep-reach rate.
  - **GO gate:** realized joint deep-reach rate within the block-bootstrap 95% band of predicted `P(A_3)`
    (block by game_date), pooled per league; count-weighted mean `|empirical − predicted|` for `P(A_3)`
    **improves ≥ 20% relative** vs an **independence** baseline (proves the copula earns its keep).
  - **t-branch trigger:** only if Track-2 joint calibration **fails GO under Gaussian** *and* the A4
    depth-stratified exceedance-Spearman clears the R3 band *and* ΔAIC≥10/ν̂≤15 — then re-price that league
    deep with t and re-test. (Three gates must agree; a single one is not enough — R3 discipline.)

**Style/consistency with R3.** Same GO/KILL structure, same block-bootstrap-by-game CIs, same "improve ≥
20% relative vs independence" bar, same "thresholds live in the harness, never met by loosening."

---

## A8. Fallback — manual-entry mode if Ladders is app-UI-only

**Trigger (VERIFY-6).** Stage-0 must confirm whether Underdog exposes Ladders offers via the same
`beta/v*` endpoints that already feed `books.py` (pick'em/rivals) or whether ladders are UI-only. If no
API, ship a **manual-entry mode** — the pricer/sizer is identical; only the offer source changes.

**Minimal input schema** (`scripts/price_ladder_manual.py`, click CLI reading a small YAML the owner
keys; emits a `RecommendedEntry`-shaped record):

```yaml
entry:
  league: NBA
  slate_date: 2026-07-10
  picks:                    # 3–5 entries; ≥2 distinct teams (assert)
    - player: "LeBron James"
      team: LAL
      opponent: DEN
      market: PTS           # canonical market name (stat_map)
      rungs: [24.5, 29.5, 34.5]   # 3 ascending half-integer lines from the app
    - player: "Nikola Jokic"
      team: DEN
      opponent: LAL
      market: REB
      rungs: [9.5, 12.5, 15.5]
    - ...
  payout_table:             # read verbatim off the slip (do NOT hardcode)
    fail: 0
    rung1: 1
    rung2: 5                 # pick-count-specific; owner copies the shown table
    rung3: 100
  bankroll: 1000.00
```

The engine: (1) resolves each `(league, market)` cell → calibrated CDF → three survivals via A3; (2)
assembles Σ for the same-game picks via `prediction/correlation.py` (import, don't edit); (3) prices
`P(L=r)` via the A1 reused-QMC path; (4) sizes via A5 with the keyed `payout_table`; (5) prints EV, the
`P(L=r)` vector, the deep-rung edge per pick, and the quarter-Kelly-capped stake. **This mode is also the
right stage-1 deliverable regardless of API status** — it lets the owner price real slips by hand while the
auto-ingest/candidate-builder is built, and it's the cleanest way to close VERIFY-1..5 (key a real slip,
compare the engine's `P(L=r)` to the app's shown payout structure).

---

## Compute-budget section (HARD CONSTRAINT: few min typical, 15 min MAX end-to-end)

**Slate scale (live-verified from `ladder`).** NBA heavy day: **~76 players/date average, 214 max**,
**~565 scored offers average** — matches the dispatch's 300–600. ~10 games. After the
`_select_bet_offers` filter (book-support + model-EV gates, `correlation.py:375–400`) and per-game caps
(`_MAX_OFFERS_PER_GAME=40`), each game feeds ~10–40 candidate legs; `d∈{3,4,5}` ladders from a ~15-leg
filtered per-game pool is `C(15,3)+C(15,4)+C(15,5) ≈ 455+1365+3003 ≈ 4,800` raw combos/game before the
`DeepScore` top-K cut.

**Two-stage cost per game:**

| stage | work | cost/game (d≤5, N=8,192) |
|---|---|---|
| Sobol' pool build | one scrambled `(8192 × 5)` normal block | ~1 ms (once/game) |
| Stage-1 `DeepScore` | `O(d²)` per raw combo × ~4,800 combos | ~5–15 ms (pure numpy) |
| Stage-2 exact QMC price | top-K=1,000 candidates × (Cholesky d≤5 + `(8192×d)` matmul + min + payout lookup) | ~1,000 × ~0.6 ms ≈ **0.6 s** |
| A5 Kelly solve | Brent on top survivors (~100 priced +EV) | ~1 ms |

**Per game ≈ 0.6–0.7 s. ~10 NBA games ≈ 6–7 s.** Add the two other live leagues on a heavy day (NBA +
MLB + NHL) and worst-case ladder pricing is **~20–30 s end-to-end**, dwarfed by the existing prophecize
scoring/correlation passes. **Comfortably inside the few-minutes-typical / 15-min-max budget** with an
order of magnitude to spare.

**Why this fits (the three budget levers):**

1. **Reused Sobol' pool across candidates** (A1/A2): the `(N×d)` point set is built **once per game**, so N
   scales the per-game setup, not the per-candidate loop. Naïve per-candidate IID (the incumbent shape at
   50K) would be `1,000 candidates × 50,000 draws = 5×10⁷` draws/game = ~seconds/game *just in RNG* — the
   reused pool cuts the RNG to one 8,192-row block.
2. **Two-stage filtering** (A6): `DeepScore` (closed-form, no orthant) culls ~4,800→1,000 before any QMC,
   so exact pricing runs on the top ~20%.
3. **QMC not `mvn.cdf`-per-candidate** (A1): `scipy.mvn.cdf` at ~1–5 ms/call, 3 calls/candidate, 1,000
   candidates = 3–15 s/game *and not vectorizable* — 20–40× slower than the vectorized reused-pool matmul.

**Worst-case guardrail.** If a pathological slate (playoffs, every game live) pushes candidate counts up,
the budget scales **linearly in K** (top-K) and **linearly in N** (draws). Cap `K` at `_BEAM_WIDTH` and `N`
at 8,192 by default; a `--fast` path can drop to N=4,096 (rel-SE ~3%, still fine for display) if a wall-
clock watchdog trips. **Pre-filter bound:** a single candidate-agnostic `mvn.cdf` independence upper-bound
on `P(A_3)` per game rejects no-hope deep ladders before the K-loop (cheap, one call/game).

**Determinism.** Seed the Sobol' scramble per (game_date, game) so prophecize is reproducible (the repo's
serve path values determinism — cf. golden tests); pass an explicit `np.random.Generator`/scramble seed
exactly as `_expected_payout_with_pushes` accepts `rng` (`parlay.py:243`).

---

## Stage-acceptance gates + kill rule (for the BUILD stage, R3 style)

**Enter the build** iff stage-0 capture closes VERIFY-1..3 (rung semantics: app-fixed lines, ≥-reach,
half-integer/no-integer) and VERIFY-4..5 (payout table read per-slip). If rung lines turn out
**user-selectable** (VERIFY-1 fails "app-fixed"), the candidate-builder scope changes (the engine must
*choose* rungs, not just price given ones) but the pricer/sizer is unchanged — re-scope, don't kill.

**Build-stage GO gates (all must pass on the A7 harness before serving any ladder EV):**
1. **Rung reliability (A7 Track 1):** count-weighted mean `|empirical − model survival|` ≤ 0.03 (r1/r2),
   ≤ 0.04 (r3), per league, on ≥N cells; no systematic deep-rung bias (signed mean within ±0.02).
2. **Joint deep calibration (A7 Track 2):** predicted `P(A_3)` within the block-bootstrap 95% band of
   realized, per league; ≥ 20% relative improvement vs independence.
3. **EV MC accuracy (A2):** on the deep-ladder decile, sd of ÊV across 20 Sobol' scrambles ≤ 2% of mean.
4. **Budget:** measured worst-case ladder-pricing wall-time on an NBA-heavy fixture ≤ 60 s (10× headroom
   under the 15-min end-to-end cap).
5. **Staking sanity:** the 0.5% cap binds on deep ladders; no ladder sized above `MAX_FRACTION_OF_BANKROLL`.

**KILL rule (close the lane DONE(no-ship)):** if **≥3 of the top-EV markets per league fail Track-1
deep-rung reliability by >0.05 absolute** in a consistent direction, the marginals are not deep-tail-
calibrated for ladder pricing and Ladders cannot be priced honestly on the current models → **kill the
pricing lane**, route the finding to the model-improvement track as a Gate-4/§6.11 alt-line-accuracy
deficit (do **not** ship a mis-calibrated deep-tail price behind a 250× payout). A t-copula does **not**
rescue a marginal-survival miscalibration (it only reshapes dependence) — so a Track-1 failure is terminal
for the *pricing* build regardless of A4.

**Reality check / what could make this wrong.** (a) The whole edge assumes the model's **deep-rung
survival** is trustworthy; it is the *least* certified quantity (ship gates certify calibration broadly,
but the r3 tail at survival ~0.15–0.25 is where dispersion errors bite hardest — a cell that ships for
pick'em can still be ladder-ineligible). Track-1 is the guard and can kill the lane. (b) Positive same-game
tail dependence, if real and un-modeled (A4), *under*-prices the best ladders — a *conservative* error
(we'd under-bet, not over-bet), so Gaussian-default is the safe default while A4 runs. (c) The payout table
is assumed readable per-slip; if it silently varies by slate/player (VERIFY-5) and we hardcode, every EV is
wrong — hence "read the table, never assume 1000×."

---

## VERIFY register (each keyed to a stage-0 capture task)

| ID | Claim to verify | Stage-0 capture task |
|---|---|---|
| **VERIFY-1** | Rung lines are **fixed by the app** (not user-selectable) | Capture a live Ladders slip payload / screenshot; confirm the 3 rung lines per player are app-set. If user-selectable, re-scope the candidate builder (engine must *choose* rungs). |
| **VERIFY-2** | Reaching a rung means **`Y ≥ ℓ`** (the "+" glyph = at-or-above) | Grade a real slip where a player lands exactly at a rung's integer-adjacent value; confirm "+"=≥. Half-integer lines make it moot for pricing but confirm for grading. |
| **VERIFY-3** | Rung lines are **half-integer** (no integer lines ⇒ no push mass) | Capture 20+ real ladder slips across markets; confirm all rung lines end in .5 (matches the 5.56M/5.56M archive rate). If any integer line exists, implement the A3 integer-line/void branch. |
| **VERIFY-4** | Multiplier table: Pick-3 {1,3,25}, Pick-4 {1,5,100}, Pick-5 {1,10,250}; **1000× is marketing, not a real 5-pick r3** | Read the per-slip payout table off 3-, 4-, 5-pick slips; reconcile the "$1,000 on $10" example (implies 100×, a Pick-4 r3, not Pick-5). |
| **VERIFY-5** | Payout table is **constant** (not slate/player-varying) | Compare payout tables across ≥3 slates and ≥3 player sets; if it varies, the engine must read the table per-slip (already the design) — confirm the payload carries it. |
| **VERIFY-6** | Ladders offers are **API-exposed** (same `beta/v*` endpoints as pick'em/rivals) | Inspect the app's network calls / existing Underdog endpoints in `books.py`; if UI-only, ship the A8 manual-entry mode first. |
| **VERIFY-7** | **DNP/void/rescue** grading: a void drops the entry to the next-viable pick count; a Pick-3 void → refund; a Rescued pick caps the entry at **refund** even if others go deep | Confirmed in help text (see Sources); re-confirm the *rung* interpretation (does a rescued pick set `R_i`=r1-equivalent, forcing `min`=r1?) on a real graded rescued slip. Model a void/DNP pick by **dropping it and re-pricing at the smaller `d`** (mirrors pick'em push→next-size), and a **Rescued** pick by forcing that pick's `R_i ≤ 1` (caps the min at refund). |
| **VERIFY-8** | Minimum **≥2 distinct teams** and **≥3 active picks to grade** | Confirmed in help text; assert both in the candidate builder and manual-entry schema. |

---

## Sources

| # | Source | Identifier / URL | Bears on |
|---|---|---|---|
| S1 | Underdog Ladders — General Rules (help center) | help.underdogsports.com/en/articles/11084685-ladders-general-rules (redirect from help.underdogfantasy.com) | Payout keyed to lowest rung; ≥2 teams; ≥3 active picks; void/rescue grading |
| S2 | RotoGrinders — Underdog Ladders: How to Play & Win (June 2026) | rotogrinders.com/sports-betting/underdog-fantasy/ladders | Multiplier table (Pick-3 1/3/25, Pick-4 1/5/100, Pick-5 1/10/250); "+" reach glyph; $1,000-on-$10 example (VERIFY-4) |
| S3 | Underdog — Tied or Voided Picks | help.underdogsports.com/en/articles/8974260-tied-or-voided-picks-champions | Exact-line result voids the leg; void → next-viable size / refund |
| S4 | Underdog — Pick'em Rescues | help.underdogsports.com/en/articles/8970218-pick-em-rescues | Rescue caps a ladder entry at refund even if others go deep (VERIFY-7) |
| S5 | OddsJam — How Do Pushes Work on Underdog | oddsjam.com/betting-education/how-do-pushes-work-on-underdog | Half-integer lines prevent ties; push→next-size; 2-leg push→refund |
| S6 | Demarta & McNeil (2005), *The t Copula and Related Copulas* | ressources-actuarielles.net/…/t copula demarta mcneil.pdf | Gaussian zero tail dependence; t tail-dependence formula (A4) |
| S7 | Frahm, Junker & Schmidt (2005), *Estimating the tail-dependence coefficient: properties and pitfalls* | (as cited in R3) | Why nonparametric λ̂ can't gate at these n (A4) |
| S8 | Dey & Stephens (2018), *CorShrink* | researchgate.net/publication/326383607 | EB Fisher-z correlation shrinkage template (A4, via R3) |
| S9 | Genz (1992), *Numerical computation of multivariate normal probabilities*, J. Comp. Graph. Stat. 1:141–149 | doi:10.1080/10618600.1992.10477010 | MVN rectangle QMC transform; convergence favors low d (A1/A2) |
| S10 | Genz & Bretz (2009), *Computation of Multivariate Normal and t Probabilities*, Springer LNS 195 | doi:10.1007/978-3-642-01689-9 | Orthant/rectangle probability algorithms; t-probability sampling (A1/A4) |
| S11 | mvtnorm R package manual (2026) — Genz/Miwa methods | cran.r-project.org/web/packages/mvtnorm/mvtnorm.pdf | Confirms QMC default for MVN prob up to d≈1000, Miwa exact ≤20 (A1) |
| S12 | l'Ecuyer, *Randomized Quasi-Monte Carlo* (Springer, RQMC advances) | link.springer.com/chapter/10.1007/0-306-48102-2_20 | RQMC as variance reduction; CRN across candidates (A2) |
| S13 | Kelly (1956), *A New Interpretation of Information Rate* / Thorp, *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market* | (standard) | Log-growth staking; concavity of `g(f)` for discrete outcomes (A5) |
| S14 | Repo: `prediction/parlay.py` | parlay.py:66,94,193–216,219–290,339–353,376–453,456 | Incumbent MC cut/classify/payout pattern to generalize; PSD repair; beam-search two-stage |
| S15 | Repo: `prediction/correlation.py` | correlation.py:120–157,168–198,201–261,375–400 | Σ assembly (import, don't edit); 0.75/0.25 offense/defense weighting; leg pre-filter |
| S16 | Repo: `training/correlate.py` | correlate.py:40–59,695–718,855–879 | Estimator conventions: 8-game residualization, `2·sin(πρ_s/6)` remap (:867), overlap shrinkage (:698–718) |
| S17 | Repo: `helpers/archive.py` + live query | archive.py:115–124,911–934; `ladder` table (16,889,205 rows) | Validation asset census; de-vigged per-rung `p_over`; half-integer lines |
| S18 | Repo: `strategies/kelly.py` | kelly.py:33,37,114–159,162–215 | quarter-Kelly, 0.5% cap, shrinkage blend, discrete log-objective analogue |
| S19 | Repo: `data/config/underdog_payouts.json` | (file) | Payout-table storage convention to mirror for the ladder table |
| S20 | R3 copula verdict | /tmp/researcher_copula_stage0.md | Gaussian default; t-branch triggers (ΔAIC≥10, ν̂≤15); hierarchical Fisher-z EB — A4 must stay consistent |
| S21 | model_improvement_track §1, §6.11 | docs/handoffs/model_improvement_track.md | "Gate-4 PIT-KS calibration IS alt-line pricing accuracy" — the rung price is the alt-line price (A3/A6/A7 kill rule) |

---

## Load-bearing conclusions for the plan (where the main session files each)

1. **New-lane scaffolding (roadmap §5.1 / new `dfs-products` lane brief, "Read first" + "Mission"):**
   Ladders = two modules under `strategies/` (`strategies/ladders.py` pricer+builder rhyming with
   `underdog_pickem.py`) + `scripts/audit_ladder_calibration.py`, all **importing**
   `prediction/parlay.py`+`prediction/correlation.py`, editing neither.
2. **"Stage 0 — research verdict" block (new lane brief):** engineering project, not a research bet;
   pricer = 3-orthant nested-survival form priced by a **per-game reused Sobol' pool** (N=8,192, CRN across
   candidates, antithetic on); **not** `mvn.cdf`-per-candidate. Push mass ≈0 (half-integer lines verified).
3. **Cross-lane consistency note (parlay-dependence §6 / R3 cross-ref):** Ladders inherits R3's Gaussian
   default and t-trigger unchanged, and **adds** a rung-depth-stratified exceedance re-test on the `ladder`
   table with a three-gate adopt rule (deep-Spearman band **AND** ΔAIC≥10 **AND** ν̂≤15) — file as an
   extension of R3, contradicts nothing.
4. **"Open questions" entries (new lane brief):** the eight-row **VERIFY register** (rung semantics, payout
   table, API existence, DNP/rescue rung mapping) — each keyed to a concrete stage-0 capture task.
5. **Build-stage gates + KILL rule (new lane brief §gates):** the five GO gates + the terminal KILL rule
   (≥3 markets/league with >0.05 deep-rung survival bias ⇒ ladder-ineligible, route to model-track §6.11);
   a t-copula cannot rescue a marginal miscalibration.
6. **Compute-budget entry (new lane brief §serve-budget, mirroring the owner's 2026-07-10 constraint):**
   worst-case ~20–30 s end-to-end for ladder pricing on an NBA+MLB+NHL heavy day, 10× under the 15-min cap;
   levers = reused Sobol' pool, two-stage `DeepScore` pre-filter, QMC-not-`mvn.cdf`.
7. **Staking note (new lane brief §sizing):** 1-D Brent solve on the discrete-outcome log-growth,
   `π_book`-blended shrinkage, verbatim `fractional_kelly_stake` fraction+cap; the 0.5% cap binds on deep
   ladders by design.
8. **Validation-asset note (new lane brief §validation):** the `ladder` DuckDB table (16.9M rows / ~1.98M
   ≥3-rung offers / 12 books / 5 leagues) is the harness substrate; currently write-only, now the read
   spine for `audit_ladder_calibration.py`.
