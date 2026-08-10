# Stage-0 Research Brief — Parlay Dependence: copula on PIT residuals

Statistician's brief for docs/handoffs/parlay-dependence.md §6 Stage 0. Read-only session;
repo claims verified against: docs/handoffs/parlay-dependence.md, training/correlate.py (full),
prediction/correlation.py:100–260, prediction/parlay.py:60–210, one test-set CSV header
(MLB_total-bases.csv), books.py rival greps. Intended home per lane convention:
`/tmp/researcher_copula_parlay.md` / plans path named in the dispatch — this file is the
plan-mode-permitted location; copy verbatim when flipping out of plan mode.

---

## Q1. Gaussian vs t copula at dims 2–6

**Prior from theory.** The Gaussian copula has zero tail dependence at any ρ<1; the t copula
has symmetric tail dependence λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) (Demarta & McNeil 2005).
At the correlation magnitudes seen in same-game prop residuals (post-shrinkage |ρ| mostly
0.05–0.4 given `CORR_MAGNITUDE_FLOOR=0.05` and the 8-game residualization), λ is small even
for moderately heavy tails: ρ=0.3, ν=10 gives λ≈0.03. So tail dependence per pair is likely
immaterial — **but** the priced event is a joint upper/lower orthant (all 2–6 legs hit), and
orthant probabilities compound: at p≈0.55 marginals and 5–6 legs, even λ≈0.03-per-pair-level
tail thickening can move P(all hit) several % relative, which is material against a fixed
payout curve. Verdict must be empirical, and the test is cheap.

**Recommended empirical test (run at stage 1/2 boundary, on pooled PIT pairs per
pair-type):**

1. **Exceedance rank correlation** (primary). For each of the ~10 highest-count same-team
   pair-types per league, pool PIT pairs (U_a, U_b) across teams/games. Compute Spearman ρ_S
   on the full pool, then on the joint-exceedance subsets {both U > q} and {both U < 1−q} for
   q = 0.75 and 0.9. Simulate the Gaussian-copula null: 500 draws of matched n at the
   full-sample ρ, same subsetting → 95% band for the exceedance ρ_S. Empirical exceedance
   correlation sitting above the band in **both** tails, in a majority of tested pair-types,
   is the t-copula signal. (Symmetric-both-tails matters: one-sided excess suggests an
   asymmetric family, which at these dims we'd still approximate with t per the lane's
   no-vines lock.)
2. **Pooled pseudo-MLE with one ν per league** (confirmatory). Semi-parametric copula fit on
   pooled normal scores z = Φ⁻¹(U): fit Gaussian (ρ per pair-type) vs t (same ρ structure +
   single shared ν). Compare by ΔAIC and a likelihood-ratio test on the boundary
   (ν→∞ ⇒ Gaussian). Ship t only if ΔAIC ≥ 10 and ν̂ ≤ 15.
3. **Do not gate on nonparametric λ̂.** Estimators like λ̂_U = 2 − log Ĉ(u,u)/log u are
   high-variance and badly biased at n in the hundreds–low-thousands (the classic
   "properties and pitfalls" result of Frahm, Junker & Schmidt 2005); at our per-pair-type
   pooled n they cannot distinguish λ=0 from λ=0.05. Use them descriptively only.

**df-estimability caveat.** ν is weakly identified once the true ν exceeds ~10–15: the
log-likelihood is nearly flat in 1/ν near 0, so SE(ν̂) explodes and per-pair ν̂ at n≈200–500
is noise. Never fit ν per pair or per team. Fit **one ν per league** (or one global), pooled
across pair-types; require pooled N ≥ ~2,000 pairs before trusting ν̂ at all. If pooled
ν̂ ≥ 15 or the CI includes ν=30 → use Gaussian; the extra parameter buys nothing and costs MC
sampling complexity (t sampling needs a chi-square mixing draw; trivial but one more thing).

**Recommendation.** Default Gaussian, run the two tests above as a mandatory stage-2
diagnostic, and structure the fit code so the copula family is a two-branch switch (Gaussian
/ t-with-league-ν) — not an abstraction layer. Expected outcome, stated for the ledger:
Gaussian survives; the test exists to catch the NBA blowout/pace regime (both players' minutes
and possessions co-crash), which is the one plausible source of real joint-tail mass.

Sources: [Demarta & McNeil, The t Copula and Related Copulas](https://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/303eb11b4d617b79c1257b0800744575/$FILE/t%20copula%20demarta%20mcneil.pdf);
[t copula with multiple df parameters (arXiv:0710.3959)](https://arxiv.org/pdf/0710.3959);
[tail-dependence estimator overview](https://metricgate.com/docs/tail-dependence-coefficient/).

---

## Q2. EB shrinkage design for per-(leg-type-pair) correlations across teams

**What exists in `training/correlate.py` (verified, reuse):**

- `_residualize_gamelog` (correlate.py:481): leak-free per-player rolling-8-game mean
  subtraction, `MIN_ROLLING_OBSERVATIONS=3` warmup → NaN. This is exactly the right
  residual concept; the copula lane swaps the *residual value* for a *PIT value* but keeps
  the same leak-free windowing discipline.
- Stratified same-team / opposing matrices with position-role keys
  (`_TRACKED_STATS`, `_build_team_game_records`, `_stratify_team_pairs`): the (team, game)
  record cache at `data/training_data/{LEAGUE}_corr.parquet` with `_OPP_`-prefixed opponent
  columns is the natural same-game pairing spine — one row per (team, game), columns
  role-keyed (e.g. `QB1.passing yards`, `_OPP_WR2.receiving yards`).
- `_pairwise_spearman_with_overlap` (correlate.py:637): the three-matmul rank-correlation +
  pairwise-overlap-count trick. The overlap matrix **is** the census raw material.
- Spearman→Pearson remap `2·sin(π ρ_S/6)` (correlate.py:867): this is precisely the
  rank-to-Gaussian-copula-parameter map — for a Gaussian copula, ρ_S(ρ) = (6/π)·asin(ρ/2),
  so the remap is already estimating the *copula* correlation, not a raw Pearson. Keep it,
  or equivalently compute Pearson on normal scores Φ⁻¹(U) once PITs exist (slightly more
  efficient, same estimand).
- Linear overlap-credibility shrinkage `_shrink_correlations` (correlate.py:698):
  ρ ← ρ·min(1, n/30) with `MIN_OVERLAP_FOR_FULL_WEIGHT=30`.

**What's wrong / missing for a copula fit:**

1. **Shrinkage target is zero.** The incumbent pulls every thin pair toward independence.
   For a lane whose entire thesis is "the apps under-tax correlation," shrinking toward 0 is
   a systematic bias *against* the edge: a QB×WR1 pass-yds pair on a new team with n=10
   games gets weight 1/3 even though the league-wide pair-type mean is strongly positive
   and well-estimated. The EB fix is to shrink toward the **pair-type mean**, not zero.
2. **No between-team variance model.** The linear n/30 weight is ad hoc; credibility should
   come from the ratio of sampling noise to true between-team spread.
3. **Raw-stat residual space, not PIT space.** Correlations of raw residuals conflate
   marginal shape with dependence; the copula fit needs U = F̂(y) through the *model*
   marginals so dependence is estimated on the same scale it will be applied.
4. **No export of overlap counts** (census) and **no joint-likelihood machinery** (t-ν fit,
   held-out log-lik) — both new, both small.

**Recommended design: two-level hierarchical EB in Fisher-z space (with a type-level
guard), not flat.**

Work in z = atanh(ρ) where the sampling variance is ≈ 1/(n−3) independent of ρ:

- **Level 1 (team | pair-type):** for pair-type g (league × scope × role-market-pair) and
  team t with n_t overlapping games and raw normal-scores correlation z_t:
  ẑ_t = B_t·μ̂_g + (1−B_t)·z_t, with B_t = (1/(n_t−3)) / (τ̂²_g + 1/(n_t−3)).
  Estimate τ²_g (between-team spread) by DerSimonian–Laird method-of-moments across the
  teams in the pair-type. If the league has <8 teams with n_t ≥ 10, or τ̂²_g ≈ 0, collapse to
  complete pooling (every team gets μ̂_g). This is standard partial pooling — MSE-optimal
  across groups (Efron–Morris lineage; see also CorShrink, Dey & Stephens 2018, which does
  exactly adaptive EB shrinkage of Fisher-z correlations and is the closest published
  template).
- **Level 2 (pair-type | league):** μ̂_g itself gets a credibility shrink toward 0 with
  weight N_g/(N_g + N₀), N₀ ≈ 200 effective pairs. This preserves the incumbent's one good
  instinct (unsupported pair-types shouldn't invent correlation) but applies it at the type
  level where it belongs, instead of per team. A full third level (pair-type → league grand
  mean over all types) is over-engineering at this data size — types are heterogeneous in
  sign (e.g. QB pass yds × opposing QB pass yds vs QB×own-WR), so the grand mean is not a
  meaningful prior. Two levels + zero-guard is the whole design.
- **Assembly:** per parlay, the leg-pair ρ comes from tanh(ẑ_t) for the involved team(s),
  falling back to tanh(μ̂_g) when the team is unseen. PSD repair stays as-is
  (`_nearest_psd`, parlay.py:198) — EB-shrunk pairwise matrices are not guaranteed PSD.

Flat (one shrinkage to a global mean) is strictly dominated here: the pair-type structure is
real, known, and cheap; the hierarchy is one groupby + two formulas, no new class.

Sources: [CorShrink: EB shrinkage of correlations (Dey & Stephens 2018)](https://www.researchgate.net/publication/326383607_CorShrink_Empirical_Bayes_shrinkage_estimation_of_correlations_with_applications);
[partial pooling / shrinkage overview](https://jrnold.github.io/bayesian_notes/shrinkage-and-hierarchical-models.html).

---

## Q3. Census spec (design only — do not run in stage 0 planning)

**Script:** `scripts/census_parlay_pairs.py` (read-only; click CLI per repo convention;
tqdm over leagues). **Input:** the existing warm-start caches
`data/training_data/{LEAGUE}_corr.parquet` — these are exactly the (team, game) records over
the `LOOKBACK_DAYS=300` window correlate.py uses, with role-keyed stat columns and `_OPP_`
prefixes, so the census inherits the production window by construction. If a league's cache
is absent/stale, the script calls `_build_team_game_records` (import, don't copy). A second
pass joins `stat_meta.json` to flag which markets have shipped marginals (PITs only exist
for shipped cells — pairs where either side is unshipped are fit-ineligible until promotion).

**Computation:** per league, per scope (same_team: both columns unprefixed; opposing:
exactly one `_OPP_`), build the 0/1 present-mask and count pairwise both-present rows with
the same `M.T @ M` matmul used by `_pairwise_spearman_with_overlap` — once on role-keyed
columns (QB1×WR2 grain), once after stripping position digits and pooling to market-pair
grain (the pair-type grain the EB prior uses). Per-team counts come from grouping the mask
by TEAM before the matmul.

**Output table columns** (one row per league × scope × pair-type, parquet + printed
summary):

| column | meaning |
|---|---|
| `league`, `scope` | NFL/NBA/…; `same_team` / `opposing` |
| `role_a`, `market_a`, `role_b`, `market_b` | position-role and stripped market names |
| `n_pair_obs` | pooled both-present (team, game) rows in window |
| `n_teams` | teams with ≥1 pair observation |
| `n_teams_ge_10`, `n_teams_ge_30` | teams meeting per-team EB / full-weight thresholds |
| `median_n_per_team`, `p90_n_per_team` | per-team overlap distribution |
| `shipped_a`, `shipped_b` | `stat_meta.json` shipped state (withheld/devel/main) |
| `fit_eligible` | shipped_a ∧ shipped_b (both marginals certified) |
| `window_start`, `window_end` | from the cache DATE range |

**Minimum n under the recommended shrinkage.** Per-team estimates are usable at any n ≥ 10
(they just shrink hard toward μ̂_g); keep 30 as the "mostly-own-data" threshold (matches
`MIN_OVERLAP_FOR_FULL_WEIGHT`). The binding constraint is the **pair-type prior**:
SE(z̄_g) ≈ √(τ²/k + 1/(N_g−3k)) — practically, pooled **N_g ≥ 300** gives SE(ρ̄) ≈ 0.06,
tight enough to beat the incumbent's zero-target shrinkage. For the league-level t-ν fit,
pooled **N ≥ 2,000** pairs per league (across types) before ν̂ means anything (Q1 caveat).
**Stage-0 kill criterion, concrete:** a league is copula-viable iff ≥15 fit-eligible
pair-types reach N_g ≥ 300; if no league qualifies, the incumbent's shrunk matrix stands and
the lane closes DONE(no-ship).

---

## Q4. PIT extraction: source and leak-freedom

**Verified fact:** the test-set CSVs (e.g.
`src/sportstradamus/data/test_sets/MLB_total-bases.csv`) carry the **full feature vector**
per holdout row plus `Result`, `Line`, `Odds`, `EV`, `P`, `P_standalone`, `NB_P`, `R`,
`Gate`, `Blended_EV`, `Player`, `Date`. They do **not** carry predictive-distribution
parameters, and they carry **no team/opponent/gameId columns**. `P` is the probability at
the *line*, not F̂ at the observed outcome — so the CSVs **as-is do not suffice** as the PIT
source.

**But they are the right row spine.** Because the full feature matrix is stored, PITs are a
re-score, not a re-build: load the production model pickle per cell, predict distribution
parameters on the CSV's stored feature rows, compute U = F̂(Result). This is leak-free by
construction (the rows are that cell's holdout — excluded from training), and it reuses the
exact serving decode. Two hard requirements:

1. **Use the serving decode path, not a re-implementation.** Offset/normalization decode
   drift (offset_mode, `MeanYr_nonzero` denominators, ZI decode) has bitten this repo
   before; the PIT extractor must call the same decode the prediction pipeline uses.
2. **Randomized PIT for discrete/hurdle marginals** (Brockwell 2007, already the ship-gate
   precedent): U = V·F̂(y) + (1−V)·F̂(y−1), V ~ Unif(0,1), applied to NB/ZINB/hurdle cells;
   continuous (Gamma/SkewNormal) cells use plain F̂(y). Without it, discrete PITs are
   non-uniform by construction and every KS test fails spuriously.

**Same-game grouping:** recover (team, opponent, gameId) by joining (Player, Date) to the
league gamelog. That join is exact for these leagues (one game per player-date; doubleheader
MLB dates are the known exception — drop ambiguous joins rather than guess).

**Coverage caveat → the census must count twice.** Holdout rows per cell are
n_validation-sized; the *intersection* across two cells' holdouts within the same game is
what the copula fit consumes, and it is strictly thinner than the gamelog pair counts from
Q3. So the census script gets a second mode: `--source test_sets`, counting same-game pair
rows where **both** cells' holdout CSVs contain the game. Decision rule: if test-set
intersections meet the Q3 thresholds, fit and validate entirely on holdout (cleanest). If
thin, fit on a walk-forward re-predict over the correlate.py 300-day window (predicting each
date with the then-production pickle if archived, else accepting mild in-sample optimism
*for dependence estimation only*) — and validate stage-1 uniformity and stage-2 held-out
log-lik **strictly on the holdout intersections**. Dependence (rank) structure is far less
sensitive to in-sample marginal optimism than the PITs' marginal uniformity, which is why
the fit/validate split is acceptable; the brief flags it for the ledger either way.

---

## Q5. Rivals-first assessment

**Yes — Rivals first.** Reasons, in money order:

1. **Dimension 2 makes the copula question trivial.** A bivariate Gaussian copula with one ρ
   captures essentially all the dependence that matters at |ρ| ≤ 0.4; family choice (Q1) is
   a second-order refinement there. P(A−B>k) = E_B[1−F_A(k+B)] under independence, or a
   cheap 2-dim MC / Gaussian-copula integral with the incumbent's ρ when A and B share a
   game. **The incumbent Pearson matrix suffices at 2 dims** — the upgrade path (EB priors,
   PIT marginals) improves the number ρ, not the machinery.
2. **Most Rivals pairings are cross-game** (two players from different games), where
   independence is *correct* and the entire edge is marginal quality — which the six ship
   gates already certify. No dependence work at all for that slice.
3. **Ingestion already exists.** `books.py:28` (`UD_RIVALS_URL = …/beta/v3/rival_lines`) and
   `_ud_augment_from_rivals` (books.py:113) already parse rivals players/games/appearances;
   `_pooled_underdog_curve` (parlay.py:104) confirms rivals legs sit in the standard payout
   pool; `correlation.py:160` `_leg_bets` already flips the second cMarket for `"vs."`
   props — H2H legs are partially plumbed through scoring today.
4. **What's actually missing** is small: a difference-distribution pricer
   (P(A−B>k) by MC through the two fitted marginals, ρ-coupled when same-game, with push
   handling at integer k) and a calibration check of that price. That is a fraction of the
   stage-3 scope and yields a shippable product while stages 1–2 of the full lane proceed.

Risk note: Rivals lines are set on the *difference*, where book margin behavior differs
from single-leg props; run the same `gap_indep`/`gap_copula` audit split on rivals history
before trusting the edge.

---

## Q6. Acceptance thresholds (stage 1/2/3)

**Stage 1 — PIT uniformity (per shipped cell, randomized PIT, holdout rows):**
- Per-cell two-sided KS vs U(0,1): **p ≥ 0.01** to enter the copula pool (D ≤ 1.63/√n).
- Portfolio: **≥ 90% of shipped cells with p ≥ 0.05**; any cell failing p ≥ 0.01 is
  *excluded from the fit* (its pairs dropped), not patched.
- If **> 30% of shipped cells fail p ≥ 0.05** → marginals not ready; stage-1 kill
  (BLOCKED on model-track calibration) per the lane brief.

**Stage 2 — held-out joint log-likelihood (temporal split or grouped 5-fold by game
date; block-bootstrap by game for CIs):**
- vs **independence**: mean OOS copula log-density gain ≥ **+0.01 nats per leg-pair** with
  the 95% block-bootstrap CI excluding 0.
- vs **incumbent** (Gaussian copula over today's shrunk-Spearman Σ, same groups): mean OOS
  gain > 0 with **95% CI excluding 0** (no magnitude floor — any real joint-pricing gain
  compounds across every entry; but CI discipline is non-negotiable).
- t over Gaussian only on Q1's rule: ΔAIC ≥ 10 **and** ν̂ ≤ 15 on pooled N ≥ 2,000.

**Stage 3 — offline A/B joint calibration (audit_parlay_calibration.py harness,
reusing its `gap_copula`/`gap_indep` split):**
- Decile reliability of predicted all-hit probability: count-weighted mean |empirical −
  predicted| improves ≥ **20% relative** vs incumbent, and no decile worsens beyond its
  bootstrap noise band.
- Parlay-hit **Brier not worse** than incumbent (guard against a calibration win bought
  with discrimination loss).
- Aggregate honesty: mean predicted joint p within **±2% absolute** of empirical hit rate
  on the A/B sample (no systematic EV inflation from the new path).
- Never met by loosening: thresholds live in the harness, locked per §4.

---

## Literature index

- Demarta & McNeil (2005), *The t Copula and Related Copulas* — tail-dependence formula, df
  behavior. [pdf](https://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/303eb11b4d617b79c1257b0800744575/$FILE/t%20copula%20demarta%20mcneil.pdf)
- Frahm, Junker & Schmidt (2005), *Estimating the tail-dependence coefficient: properties
  and pitfalls* — why nonparametric λ̂ can't gate at our n.
- Dey & Stephens (2018), *CorShrink* — EB adaptive shrinkage of Fisher-z correlations across
  groups; closest template for Q2. [link](https://www.researchgate.net/publication/326383607_CorShrink_Empirical_Bayes_shrinkage_estimation_of_correlations_with_applications)
- Brockwell (2007), *Universal residuals* — randomized PIT for discrete marginals (already
  the ship-gate precedent in this repo).
- DerSimonian & Laird (1986) — method-of-moments between-group variance for the level-1
  credibility weight.
- McNeil, Frey & Embrechts, *Quantitative Risk Management* — elliptical-copula estimation,
  weak identification of ν.
- Sports-specific SGP copula literature is thin; no directly citable same-game-parlay copula
  paper surfaced in search — treat the financial-risk copula literature as the methods
  source and this repo's audit harness as the domain evidence.

## Verification appendix (repo facts checked this session)

- `training/correlate.py`: residualization at :481 (window 8, min 3), matmul
  Spearman+overlap at :637, remap `2·sin(π/6·ρ_S)` at :867, linear shrink n/30 at :698,
  floor 0.05 at :58, window 300d at :40; per-(team,game) cache
  `data/training_data/{LEAGUE}_corr.parquet`.
- `prediction/correlation.py`: `_build_game_corr_map` :120 (offensive/defensive pair
  weights, `_OPP_` conventions), `_leg_pair_corr_boost` :168 (cMarket-averaged ρ, sign flip
  on Over/Under mismatch, `"vs."` flip at :160), C/M assembly :201.
- `prediction/parlay.py`: `_PUSH_MC_SAMPLES=50_000` :94, PSD repair :198, pooled Underdog
  curve with rivals in-pool :104, legacy-flag pattern :128–159.
- Test-set CSV header (MLB_total-bases.csv): full features + Result/Line/P/EV/…/Player/Date;
  **no** distribution params, **no** team/game keys.
- `books.py`: `UD_RIVALS_URL` :28; `_ud_augment_from_rivals` :113.
