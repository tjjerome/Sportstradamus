# Plan: Mitigate GBDT Regression-Toward-the-Mean in the Training Pipeline

> **Multi-session handoff doc.** Status below is updated each session. The
> attached source report lives in the originating session; this file is the
> durable plan + progress log for the project on branch
> `claude/fix-gbdt-mean-regression-GcY1g` (PR #46 → `devel`).
>
> **All phases of this plan are worked and documented in PR #46.** Each
> phase's code, the harness run-log entries, and the status updates below land
> as commits on that one PR; do not open separate PRs per phase.

## Status / progress log

| Phase | State | Notes |
|---|---|---|
| **P0 — offline eval harness** | ✅ done (PR #46) | `src/sportstradamus/scripts/compression_eval.py` + `tests/golden/test_compression_eval.py`. ruff clean, 6 unit tests pass, CLI single+diff smoke-tested on synthetic data. Full `poetry` gates NOT run in the build env — network policy blocks the PyTorch CPU wheel source so `poetry install` fails on `torch`; needs a normal-network run before merge. |
| **P0.5 — determinism gate** | ✅ done (PR #46) | Opt-in `meditate --deterministic` (debug-only, never publish) + `tests/integration/test_determinism_gate.py`. Pure helpers `seed_everything` / `fit_lss_model` / `predict_lss_params` / `fit_predict_params` in `pipeline.py`; under `--deterministic`: RNGs pinned (random/numpy/torch + `torch.use_deterministic_algorithms`), Optuna swapped for `DETERMINISTIC_FIXED_PARAMS`, input frozen to cached parquet. Persistent writes are **redirected to `data/{test_sets,models}/deterministic/`** (training parquet + whole-suite `report()` stay fully suppressed) so a `--deterministic` run produces consumable artifacts without ever overwriting production paths. Gate runs on real cached `NBA_FGA.parquet` (4000 rows, ~5s) with stochastic LightGBM (`feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`) so it actually tests the seeding mechanism — different seed produces `loc` max-abs diff ~0.34, same seed bit-identical. Default `meditate` byte-identical. P1 unblocked. |
| **P1 — centered-target bridge (SkewNormal)** | ✅ done (PR #46), result: **FGA-only SHIP, family-wide KILL** | Two centered-target variants A/B'd path-wide under `meditate --deterministic` against the `ratio_meanyr` baseline. (a) `centered_additive_eb_meanyr_k10` (Phase-A's EB(MeanYr, K=10)): FGA SHIPS (+5.3% top-decile MAE, brier_skill +0.096→+0.112), every other SkewNormal market KILLs (PTS −3.5%, PA −4.1%, PR −2.9%, RA −2.2%, FG3A −3.8%, FGM −2.6%, MIN +3.7%, PRA +0.8%, REB +0.2%, fantasy-points-prizepicks brier_skill regressed). (b) `centered_additive_mean10` (trailing-10 baseline, more responsive to recent form — added post-Phase-A as the obvious "level shifts with form" hypothesis): **every SkewNormal market KILLs** including FGA (+4.6% — close but under the 5% bar), with PA −6.6% and PR −6.7% notably *worse* than Phase-A. Count-family markets (FG3M, FTM, OREB, PF, STL, TOV, BLK, BLST) showed exactly 0% delta under both strategies as expected — the centered-target transform is a no-op for NegBin/ZINB. **Both runs together confirm the OVERCONFIDENCE_INVESTIGATION §3.2 "decisive negative result", strengthened: the SkewNormal level bias is not the dominant compression cause path-wide, regardless of baseline horizon (long-term EB or short-term trailing-10).** FGA is genuinely special — its win comes from EB(MeanYr) capturing structural shot-volume; Mean10 is too noisy for FGA itself and not the right lever for volume-shifting markets either. Default `--target-strategy=ratio_meanyr` stays. The infrastructure (`baselines.py`, the registry, the offset_meta pickle field, the brier_skill gate, the live-path test) is reusable for P3 (rate decomposition) and P2 (init_score baseline for count markets) — the next levers worth trying based on the path-wide negative result. |
| **P2.B — HurdleZINB (derived-π gate)** | ✅ done (PR #46), result: **6/8 NBA ZINB markets SHIP** | New `meditate --zinb-mode=hurdle` (orthogonal to `--target-strategy`; default `joint` stays byte-identical to pre-P2.B). `HurdleZINB` (in `src/sportstradamus/hurdle.py`) is a two-stage drop-in for joint ZINB: calibrated binary classifier estimates `q = P(Y=0)`; NegBin LightGBMLSS on `Y>0` supplies count shape; structural-inflation π derived from the ZINB identity `π = clip((q − NB(0))/(1 − NB(0)), 0, 1)` (NOT the simpler `gate = 1 − p_nonzero` from the original Phase B spec — corrected because downstream `fused_loc` in `helpers/distributions.py` explicitly treats `gate` as zero-inflation, not marginal P(Y=0)). Returned `total_count/probs/gate` columns match the LightGBMLSS ZINB contract so `model_prob` ZINB decode (lines 252-257) is untouched and legacy pickles still load via `getattr(model, "is_hurdle", False)`. Path-wide A/B under `meditate --deterministic --league NBA --zinb-mode hurdle` against the joint baseline: **SHIP** on FG3M (+9.7% top-decile MAE, brier_skill +0.115→+0.290), OREB (+44.9%, +0.019→+0.109), PF (+19.2%, −0.238→−0.002), TOV (+26.8%, −0.049→+0.058), BLK (+40.4%, +0.237→+0.299), BLST (+11.6%, −0.002→+0.093). **KILL** on FTM (+1.3%, under 5% bar) and STL (global MAE +14.1% regression). The joint ZINB had per-row catastrophic blowups in mid-deciles on BLK/OREB/PF/BLST under deterministic mode (compression_ratio 24–5357×; predicted means up to 1437) that the hurdle eliminates entirely — global MAE drops 60–99% on those markets. **Default stays `--zinb-mode=joint`** — shipping the infrastructure + verdict here; the per-market routing question (FTM/STL stay joint, the rest move to hurdle) is a follow-up. **Note on the verdict criterion**: the parent plan said "predicted gate mean ≈ hist_gate" — that criterion was mis-stated under derived-π semantics. Derived-π gate is π_zi (the inflation parameter), structurally ≤ q with equality only in the zero-truncated-NB limit. For FG3M (positives mean ≈ 2.2) NB(0) ≈ 0.20 even with a well-fit NB, so derived-π gate ≈ 0.17 (similar to joint's 0.18) but the *total reconstructed P(Y=0)* matches `q ≈ 0.33` exactly by construction. The meaningful SHIP/KILL signal is the downstream compression_eval verdict on `P(over@line)` proxies (top-decile MAE + brier_skill_score), not gate mean. New `tests/integration/test_zinb_hurdle_live_path.py` asserts the identity reconstruction `π + (1−π)·NB(0) ≈ q` per-row (mean tolerance 0.02) and two-run bit-identity under `DETERMINISTIC_SEED`. Determinism gate extended with a parallel hurdle assertion. |
| **P2.A — `init_score` baseline (NegBin/ZINB)** | ✅ closed: **DEAD** | In-process spike on FG3M: LightGBMLSS accepts per-row `init_score` (as a length-2n flat array, `[log_EB, zeros]` per-parameter concatenation) without raising — but the produced predictions are **byte-identical** to a plain NegBin fit, every decile. Either LightGBMLSS overrides init_score with its own `start_values` seeding, or the 30-round deterministic fit converges to the same answer regardless of starting point. Either way, the bias signature does not move. Also: FG3M's plain-NegBin top-decile bias is already −0.013 — there is no meaningful compression signature on the count-branch NegBin mean to fix; the overconfidence was the gate, which P2.B already addresses. **P2.A is dead** on the count branch as a one-line `init_score` transform. Per parent plan, the fallback is P5 (leakage-safe target-encoded player-baseline feature) or P3 (rate decomposition) — both require their own design sessions. |
| P3–P10 | ⬜ | see priority list; P10 (GPBoost) already prototyped and failed deterministically — annotated below |

## Research handoffs that fed this plan

The next-session plan below integrates findings from two researcher passes that
ran after P1 and P2.B landed. The reports themselves are not committed (they
live at `/tmp/researcher_skewnormal.md` and `/tmp/researcher_zinb.md` in the
originating session); the load-bearing citations and recommendations are
copied below. Future sessions should treat this file as self-contained.

- **SkewNormal track** — surfaces 10 new methods (T1–T10) for top-decile bias
  on SkewNormal markets (PTS/REB/AST/PRA/FGA/MIN/PA/PR/FG3A/FGM). Headline:
  ICC₁ per-market routing comes first; the original P3 basic rate
  decomposition / P5 target-encoded features / P10 GPBoost retry are all
  KILLED in favor of T-method replacements.
- **ZINB track** — focused on the FTM/STL kill markets from P2.B. Headline:
  the current "fit on positives" Stage-2 NegBin is a misspecified ZTNB; a
  single-line PyTorch fix likely resolves FTM/STL by construction. Per-market
  routing diagnostics (ziNB index, Wilson-Einbeck, Schwarz-corrected Vuong)
  are the second item; the Stage 3 architectural choice (MZINB vs GPBoost)
  is a strategic fork chosen by residual analysis, not run in parallel.

Bibliography (DOIs preserved so the plan is self-contained):

| Tag | Citation |
|---|---|
| MEGB | Olaniran et al., *Scientific Reports* 15:30927, 22 Aug 2025, DOI 10.1038/s41598-025-16526-z (CRAN + github.com/rid4stat/MEGB). |
| GBMixed | Prevett, Hui, Tho, Welsh, Westveld, ANU, arXiv 2511.00217, 31 Oct 2025. |
| CatBoost ordered TS | Prokhorenkova et al., NeurIPS 2018, arXiv 1706.09516. |
| DEGPD / ZIDEGPD | Ahmad & Hussain, arXiv 2510.27365, 2025. |
| gbex | Velthoen, Dombry, Cai, Engelke, *Extremes* (Springer), 2023. |
| FAGTB / M²FGB | Grari et al., arXiv 1911.05369; Cruz et al., arXiv 2504.12458, Apr 2025. |
| CQR / LCMQR | Romano et al., NeurIPS 2019; LCMQR arXiv 2411.19523, late 2024. |
| Normalizing-flow heads | LightGBMLSS v0.3.0, 20 Jul 2023. |
| PGBM | Sprangers et al., KDD 2021, DOI 10.1145/3447548.3467278. |
| MZINB foundations | Long, Preisser, Herring & Golin, *Stat Med* 2014, DOI 10.1002/sim.6293; Preisser, Das, Long, Divaris, *Stat Med* 2016, DOI 10.1002/sim.6804. |
| MZINB Stata | Cummings & Hardin, *Stata J* 19(3) 2019, DOI 10.1177/1536867X19874209. |
| MZINB spatial | Mutiso et al., *Biometrical Journal* 2024, DOI 10.1002/bimj.202300182. |
| Marginalized hurdle | Kassahun et al., *Stat Med* 2014, DOI 10.1002/sim.6237; Liu, Zhang, Tang et al., *HSORM* 2018, DOI 10.1007/s10742-018-0183-6. |
| ZTNB | Hilbe 2011 *Negative Binomial Regression*; UCLA-OARC ZTNB tutorial; Grodri notes on count moments. |
| ziNB / Wilson-Einbeck | Blasco-Moreno et al., *Methods Ecol Evol* 2019, DOI 10.1111/2041-210X.13185; Wilson & Einbeck, *Statistical Modelling* 2019, DOI 10.1177/1471082X18762277. |
| Corrected Vuong | Desmarais & Harden 2013; Wilson 2015, *Economics Letters*, DOI 10.1016/j.econlet.2014.12.029. |
| Feng GOF framework | Feng, *J Stat Distrib Appl* 2021, DOI 10.1186/s40488-021-00121-4. |
| CMP head | Philipson & Huang, *Statistics and Computing* 2023, DOI 10.1007/s11222-023-10244-0. |
| cyc-GBM | Delong, Lindholm & Zakrisson, SSRN 4352505, 9 Feb 2023, DOI 10.2139/ssrn.4352505. |
| CyclicBoosting | Wick et al., Blue Yonder, arXiv 2009.07052; *SN OR Forum* 2021, DOI 10.1007/s43069-021-00079-8. |
| Balanced GAMLSS boosting | Daub et al., *Computational Statistics* 2025, DOI 10.1007/s00607-023-01224-3; arXiv 2602.17272, 2026. |
| Multi-parametric GBM benchmark | Chevalier & Côté, *European Actuarial Journal* 2025, DOI 10.1007/s13385-025-00428-5. |
| Arctan pinball | Sluijterman et al., *Int J Mach Learn Cybern* 2025, DOI 10.1007/s13042-025-02671-4. |

## Context

LightGBMLSS predictions in this repo compress toward the global mean: high-volume
players are systematically under-predicted, low-volume over-predicted. Two
branches: the SkewNormal branch (`global_mean >= 2.0`, e.g. NBA PTS/FGA) uses a
`Result/MeanYr` target and multiplicative denorm; the NegBin/ZINB branch
(`global_mean < 2.0`) uses raw counts. P1 closed the centered-target question on
SkewNormal markets (FGA-only ship); P2.B closed the joint-vs-hurdle question on
ZINB markets (6/8 ship). The next-session plan attacks (a) the remaining
SkewNormal top-decile bias path-wide and (b) the FTM/STL kill on the ZINB track.

For deeper context see [docs/OVERCONFIDENCE_INVESTIGATION.md](OVERCONFIDENCE_INVESTIGATION.md)
(determinism, live-path confound, the original investigation) and
[docs/CENTERED_TARGET_NEGATIVE_RESULT.md](CENTERED_TARGET_NEGATIVE_RESULT.md)
(the path-wide P1 KILL verdict).

## Architectural principle (applies to all phases)

Make the **target/baseline transform a single configurable strategy**, selected
by a CLI flag on `meditate` (and a matching env var for the harness),
defaulting to current behavior. Every experiment becomes a new strategy value,
not a destructive rewrite. This is what makes the multi-session A/B tractable
and keeps `devel` shippable between sessions. Centralize the forward
transform, the inverse (de-norm) transform, and the inference-time mirror so
train/predict cannot drift. The inference mirror lives in
[stats/base.py:597](src/sportstradamus/stats/base.py#L597) (`get_stats`); any
new baseline must be computed there identically and leakage-safe. STYLE_GUIDE
§9 (named constants), §18.9 (no orphan methods), and the CLAUDE.md
"no new monoliths" rule all apply.

**Universal decision threshold (every experiment):** ship a strategy only if
it reduces **top-mean-decile MAE by ≥ 5%** vs the current production strategy
without worsening **global MAE by > 1%** and without worsening
`brier_skill_score` on the existing report. Otherwise kill it and move on.
The harness — [src/sportstradamus/scripts/compression_eval.py](../src/sportstradamus/scripts/compression_eval.py)
— is the ship/kill gate.

## Two-track next-session plan

Work proceeds on two independent tracks. Stages within a track are sequential
(later stages depend on diagnostics from earlier ones). The two tracks share
infrastructure (the harness, the determinism gate, the per-strategy registry)
but the diagnostics and method choices diverge.

---

## Track A — SkewNormal markets

Source: `/tmp/researcher_skewnormal.md`. Scope: PTS, REB, AST, PRA, FGA, MIN,
PA, PR, FG3A, FGM, fantasy points. After P1, FGA ships with EB(MeanYr, K=10)
centering; every other SkewNormal market killed under both EB-centered and
Mean10-centered strategies. The researcher's hypothesis is that the dominant
compression cause is **volume-efficiency entanglement** (PTS = FGA × eFG% × …),
not leaf-averaging — and that ICC per market predicts which strategies can work.

### Stage A1 — Diagnostic gate (~1 day)

**T1. ICC per market.** Compute ICC₁ per market on three seasons of held-out
data using a two-level ANOVA decomposition: σ²_between = Var(player season
means), σ²_within = mean(Var(game logs within player)), ICC =
σ²_between / (σ²_between + σ²_within). For highly skewed markets (STL/BLK)
compute on `log(1+Y)` or rank-transformed target.

Pre-register the routing cutoff before running A/Bs:
- ICC ≥ 0.5 → EB-style centering can work (T5 four-stage factorization).
- ICC ≤ 0.3 → needs distributional tail extension (T3/T7).
- 0.3 < ICC < 0.5 → ambiguous; try both, route based on outcome.

Expected: FGA > 0.6, PTS in 0.3–0.45, STL/BLK < 0.2. Implementation site:
a new notebook or a one-shot CLI in `src/sportstradamus/scripts/`. No model
changes. Source: researcher T1 (Tier 0).

**Decision points (gate logic for Stage A1):**
- If ICC_PTS > 0.5 — entanglement hypothesis is wrong; leaf-averaging
  hypothesis revives; T7 (extreme-tail boosting via gbex) jumps to Stage A2.
- If ICC is uniformly low across the family — the family is form-driven and
  T3 (heavy-tail / normalizing-flow head) takes priority over T5.
- If ICC clusters bimodally (e.g. FGA/MIN/REB-for-bigs high, PTS/AST low) —
  the path-wide negative result is fully explained; per-market routing in
  Stage A2 is justified.

### Stage A2 — Highest-leverage structural fixes (2–3 sprints)

Two methods, both worth running because they attack different mechanisms.
Pick order based on Stage A1 ICCs.

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **T5. Four-stage multiplicative factorization** *(replaces original P3)* | Tier 1 (Skew) | 2–3 weeks | Predict (a) P(plays), (b) MIN \| plays, (c) per-100-poss rate \| MIN, (d) for PTS: FGA-per-100 × FG% × points-per-make; recombine via Monte Carlo. Routes each factor to its own ICC-appropriate strategy. DFS-industry consensus approach. | New `src/sportstradamus/factorize/` package or extend [pipeline.py](../src/sportstradamus/training/pipeline.py); inference mirror in [stats/base.py](../src/sportstradamus/stats/base.py); per-market wiring via the existing strategy registry from P1. |
| **T3. Spliced / Pareto-tail or normalizing-flow head** | Tier 1 (Skew) | 1–2 weeks per distribution | Body ~ SkewNormal up to learned threshold u, tail ~ Generalized Pareto above u, mixing weight per-row. LightGBMLSS v0.3.0 normalizing-flow head is the simplest production path. Direct attack on top-decile MAE without touching loc. | Custom PyTorch distribution alongside [skew_normal.py](../src/sportstradamus/skew_normal.py); dist selection in [pipeline.py:245-324](../src/sportstradamus/training/pipeline.py#L245). |

**Decision points (gate logic for Stage A2):**
- If T5 ships globally → make it the new baseline and re-run EB centering on
  each factor independently (factor-specific ICCs dominate the decision).
  Stage A3/A4 become polish.
- If T3 ships on the high-ICC, heavy-tail markets but T5 doesn't → distribution
  was the bottleneck; route those markets to T3 and continue T5 work on
  low-ICC volume-driven markets only.
- If neither ships → either ICCs are pathologically low across the family
  (re-check T1) or the live-path confound (Model Skew=NaN, see
  `OVERCONFIDENCE_INVESTIGATION` §3.4) is consuming the gain. Stop and
  resolve the live path before more training-side work.

### Stage A3 — Calibration polish (1 sprint, mostly orthogonal — stack them)

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **Original P7. Isotonic on loc** | Original plan | hours | Fixes residual average bias on the location parameter. Cheap, monotone. | Wrap predictions in `IsotonicRegression(out_of_bounds="clip")` post-fit in [pipeline.py](../src/sportstradamus/training/pipeline.py) before the test-set dump. |
| **T8. CQR with player-decile-local conditioning (LCMQR)** | Tier 2 (Skew) | 2–3 days | Post-hoc per-player-decile calibration of *scale*. Complements P7 (loc) — they are orthogonal. Romano-Patterson-Candès CQR (NeurIPS 2019) and LCMQR (arXiv 2411.19523) give finite-sample marginal-coverage guarantees. | New module under `src/sportstradamus/calibration/` if it grows past 50 lines; otherwise an inline helper in the report path. |
| **Original P6. Reduce tree regularization** | Original plan | ~1 day | Widen Optuna ranges: larger `num_leaves`/`max_depth`, smaller `min_child_samples`/`min_child_weight` at [pipeline.py:348-368](../src/sportstradamus/training/pipeline.py#L348). | Optuna search space dict in `pipeline.py`. |
| **T9. Monotone constraint MeanYr → loc** | Tier 2 (Skew) | 1 day | Smoke test only — LightGBM `monotone_constraints` forces non-decreasing dependence of loc on MeanYr. If violated, that tells you something about the feature set. Diagnostic, not a primary fix. | `monotone_constraints` arg on the LightGBM call in [pipeline.py](../src/sportstradamus/training/pipeline.py); MeanYr feature column must be identified by index. |

**Decision points (gate logic for Stage A3):**
- If P7 + T8 combined ship ≥ 5% on top-decile MAE — calibration was the
  bottleneck; Stage A4 (novel retries) becomes lower-ROI.
- If T9 monotone constraint is violated by the trained model — feature set
  is missing a key volume driver; loop back to A1 ICC analysis with that
  feature included.

### Stage A4 — Novel risky retries (only if Stage A2/A3 leave a gap)

| Method | Source | Cost | Direct effect | Notes |
|---|---|---|---|---|
| **T4. MEGB / GBMixed** *(replaces original P10 GPBoost retry)* | Tier 3 (Skew) | 1–2 weeks (MEGB), more for GBMixed | EM/BLUP-based mixed-effects boosting that fixes the bias GPBoost was criticized for in Prevett et al. 2025. MEGB is on CRAN + github.com/rid4stat/MEGB; GBMixed has no public code. **Different mechanism from GPBoost** — prior failure does not predict failure here. | Point prediction only for MEGB; use for loc, keep LightGBMLSS for scale/shape. Port GBMixed from the paper if MEGB ships. |
| **T2. CatBoost ordered TS** *(replaces original P5 leakage-safe target encoding)* | Tier 3 (Skew) | 3–5 days per market | Only published GBDT mechanism with a proof of unbiasedness for high-cardinality categoricals (`player_id`). Strictly dominates greedy mean encoding per Prokhorenkova et al. NeurIPS 2018. | Re-fit SkewNormal/NegBin LSS heads using CatBoost instead of LightGBM. CatBoostLSS forks exist; alternatively wire an external objective. **Caveat:** unbiasedness proof is for log-loss / squared-error, not SkewNormal NLL — validate empirically. |
| **T7. gbex** | Tier 3 (Skew) | 1–2 weeks | Generalized Pareto tail boosting on exceedances. Layered on top of the existing LSS body model. Good parallel experiment to T3. | Velthoen et al. *Extremes* 2023. Published validation is on rainfall extremes, not sports data. |
| **T6. FAGTB adversarial penalty against MeanYr decile** | Tier 3 (Skew) | 1 week | Quantile-bucket MeanYr (10 deciles); adversary tries to predict the decile from the residual; penalize the loc gradient by adversary loss. Principled "bias-by-group" penalty per Grari et al. (arXiv 1911.05369) and M²FGB (arXiv 2504.12458). | Custom LightGBM objective + one notebook. FAGTB was designed for binary protected attributes; for continuous MeanYr, quantile-bucket first. |
| **T10. PGBM** | Tier 3 (Skew) | 1 week | Alternative scale predictor: mean + variance from a single ensemble without parametric distribution assumption. Avoids the SkewNormal shape bound. | Sprangers et al. KDD 2021. Per Chevalier & Côté 2025, NB-class targets do **not** uniformly benefit from probabilistic GBM over point-prediction GBM — validate per market. |

**Decision points (gate logic for Stage A4):**
- If CatBoost ordered TS alone ships > 5% on a low-ICC market → bias was
  largely `player_id` encoding leakage, not structural compression.
  Deprioritize the rest of Stage A4; MEGB and gbex become nice-to-have.
- If MEGB ships on PTS but not on FG3M → confirms the high-ICC vs low-ICC
  dichotomy; route count markets to T3/T7 only and stop Track-A work there.

### What is dropped from the Track-A plan and why

- **Original P3 basic rate decomposition** → folded into T5 four-stage
  factorization. No reason to run the basic version when the industry-standard
  four-stage version is the actual approach DFS shops use.
- **Original P5 leakage-safe target-encoded player features** → replaced by
  T2 CatBoost ordered TS. Ordered TS strictly dominates expanding-mean
  encoding per the NeurIPS 2018 proof.
- **Original P10 GPBoost retry** → replaced by T4 MEGB/GBMixed. Same goal
  (mixed-effects boosting) but the EM-pseudo-residual mechanism fixes the
  documented GPBoost bias. Already-failed GPBoost prototype stands as the
  baseline to beat, not a reason to abandon mixed-effects entirely.

---

## Track B — ZINB markets (FTM, STL kill recovery + 6 SHIP hardening)

Source: `/tmp/researcher_zinb.md`. Scope: FG3M, FTM, OREB, PF, STL, TOV, BLK,
BLST. After P2.B, 6/8 ship under hurdle mode; FTM and STL kill. The
researcher's hypothesis is that the FTM/STL kill is a **mathematical artifact
of using "fit on positives" instead of a true zero-truncated NegBin (ZTNB)
likelihood** in [hurdle.py:201](../src/sportstradamus/hurdle.py#L201). The
hurdle Stage 2 is currently a misspecified ZTNB.

### Stage B1 — Isolate and diagnose (1 week)

Two tasks in parallel. Tier 1 from the researcher.

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **ZTNB Stage 2 likelihood fix** | Tier 1 (ZINB) #1 | hours | Replace `NegativeBinomial(μ, α).log_prob(y)` in the Stage-2 hurdle loss with `nb.log_prob(y) − log1p(−exp(nb.log_prob(zeros_like(y))))`. The optimizer recovers an unbiased μ; the derived-π identity then gives the correct ZINB marginal mean (1−ψ)·μ. Eliminates the +NB(0)·(1−q)·μ_pos/(1−NB(0)) over-prediction on FTM/STL **by construction**. | Wrap the existing NegBin loss in [hurdle.py](../src/sportstradamus/hurdle.py); PyTorch already has `torch.distributions.NegativeBinomial`. |
| **Per-market routing diagnostics** | Tier 1 (ZINB) #2 | days | Build a one-page per-market dashboard with: observed mean, variance, zero-rate p₀; ziP and ziNB indices with bootstrap CIs (Blasco-Moreno et al. 2019); Wilson-Einbeck p-value (Stat Modelling 2019); Schwarz-corrected Vuong (HurdleNB vs ZINB) per Wilson 2015; var/mean ratio; conditional positive mean E[Y\|Y>0] vs μ. Routes each market to plain NB / HurdleNB(ZTNB) / MZINB / CMP. | New `src/sportstradamus/scripts/zinb_routing_diagnostics.py`; output to `data/zinb_routing/{LEAGUE}_diagnostics.parquet`. |

Routing rule (precomputable from training data alone, survives temporal split):
- ziNB CI contains 0 + low overdispersion → **plain NB**
- ziNB CI contains 0 + hurdle structurally appropriate → **HurdleNB with ZTNB Stage 2**
- ziNB > 0 robustly → **MZINB** (after Stage B3) or stay on HurdleZINB-with-ZTNB
- var/mean < 1.3 → **CMP**

**Decision points (gate logic for Stage B1):**
- FTM/STL flip to ship/neutral under ZTNB Stage 2 → bias was likelihood-level
  only. Stop and harden; Tier 2 work becomes lower-ROI.
- FTM/STL still kill → routing diagnostics tell you whether the problem is
  structural (use plain HurdleNB instead of ZINB on those markets) or deeper
  (proceed to Stage B2).
- If sample variance/mean ratio for STL is < 1.0 → pivot STL specifically to
  CMP (Stage B4) and run that A/B before any other Stage B3 work.

### Stage B2 — Routing + orthogonal feature engineering (2 weeks, in parallel)

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **Per-market routing wiring** | Tier 1 (ZINB) #2 | days | Implement the routing logic from B1's table. `data/zinb_mode_per_market.json` config + per-market lookup in [pipeline.py](../src/sportstradamus/training/pipeline.py). The 6 SHIP markets under hurdle continue to use hurdle; FTM/STL route per Stage B1 diagnostic outcome. | `pipeline.py` per-market dispatch; new JSON config under `data/`. |
| **Leakage-safe target-encoded player features** | Tier 2 (ZINB) #5; was original P5 | days | `groupby(player_id).expanding().mean().shift(1)` for stat and stat×opponent. Orthogonal to architectural choice; ships regardless of which Stage B3 winner emerges. | New columns in [stats/base.py:597](../src/sportstradamus/stats/base.py#L597) `get_stats`; same leakage audit as the MeanYr/Mean10 columns. |

**Decision points (gate logic for Stage B2):**
- 8/8 ship under routing + encoded features → hold off on Stage B3 unless
  ROI is needed elsewhere. The system is in a good state.
- <8/8 ship → proceed to Stage B3 on the markets that still kill.

### Stage B3 — Strategic fork (4–6 weeks, pick ONE — not both)

This is the canonical decision point of the ZINB track. The researcher
explicitly flags that running MZINB and GPBoost in parallel doubles cost
without obvious gain. The decision criterion is **residual structure after
Stage B2**.

| Option | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **MZINB head in LightGBMLSS** | Tier 2 (ZINB) #4 | 2–4 weeks | Reparameterize so the marginal mean ν = E[Y] is boosted directly; the latent NB conditional mean μ = ν/(1−ψ) is reconstructed. Three heads: logit(ψ), log(ν), log(α). The boosted ν IS the quantity the downstream pipeline already consumes. Cleaner than current derived-π trick. Sidesteps the gate-vs-NB(0) trade-off as a side effect. **No published GBDT implementation — novel contribution.** | New `MZINB` class alongside the existing distributions in [skew_normal.py](../src/sportstradamus/skew_normal.py) or under [helpers/distributions.py](../src/sportstradamus/helpers/distributions.py); foundational papers Long 2014 + Preisser 2016. Use Mutiso et al. 2024 (Pólya-gamma augmentation) as a likelihood-structure reference. |
| **GPBoost with NegBin likelihood + player random intercept** | Tier 2 (ZINB) #3; was original P10 | 1–2 weeks | NegBin is one of GPBoost's native likelihoods, so the LSS-flexibility loss is smaller on counts than on SkewNormal. Sigrist's published benchmarks (~10pp gap vs LightGBM-Cat, ~93pp vs naive numeric ID) transfer to NB. The earlier GPBoost prototype failed on SkewNormal FGA, not on counts. | New GPBoost dependency (user pre-approved for a phase that needs it); custom training path branched off [pipeline.py](../src/sportstradamus/training/pipeline.py). Treat as a sub-project with its own task list. |

**Decision criterion (researcher-specified):**
- **Choose GPBoost** if residuals plotted by `player_id` (with bootstrap CIs)
  show systematic per-player offsets statistically distinguishable from zero
  — i.e. the problem is missing player effects. Trade-off: lose LSS-style
  distributional flexibility for future markets that need exotic distributions
  (CMP, mixtures, normalizing flows).
- **Choose MZINB** if residuals are tail-driven or shape-driven
  (heavy-tailed within-player residuals, not location-shifted) — i.e. the
  problem is identifiability of the gate vs the count head, not missing
  player effects. Trade-off: novel implementation, ~4 weeks of debugging,
  no GBDT precedent.

In either case, also implement **Per-parameter Optuna search** (Tier 3 #6,
days of work, see Stage B4) — you want a fair tuned baseline against which to
evaluate the bigger structural change.

**Decision points (gate logic for Stage B3):**
- If residuals after Stage B2 are ambiguous (no clear per-player pattern AND
  no clear tail-driven pattern) → run a 2-week MZINB spike first (cheaper
  exit if it fails); GPBoost is the fallback.
- If Stage B4 per-parameter Optuna alone eliminates the deterministic-mode
  blowups across all SHIP markets → MZINB becomes much weaker as a
  recommendation; identifiability was hyperparameter-induced not
  parameterization-induced. Reroute Stage B3 effort to GPBoost-only.

### Stage B4 — Tuning, polish, specialized fixes (optional)

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **Per-parameter Optuna search** | Tier 3 (ZINB) #6 | days | Separate `learning_rate` and `n_estimators` for gate vs NB heads inside the existing LightGBMLSS Optuna sweep. cyc-GBM-inspired without porting. May resolve the deterministic-30-round blowups without architectural change. Daub et al. 2026 ("balanced step length") provides theoretical justification. | Extend the Optuna search-space dict in [pipeline.py:348-368](../src/sportstradamus/training/pipeline.py#L348) to distinguish gate vs count hyperparameters. |
| **CMP head for STL specifically** | Tier 4 (ZINB) #8 | 1–2 weeks | Only worth implementing if STL's empirical variance/mean ratio < 1.3 (per Stage B1 diagnostic). NB always has variance > mean for finite α; CMP corrects in either direction. Philipson & Huang 2023 provide a fast look-up-table mean-parameterized CMP. | New PyTorch custom distribution; pre-computed Z(λ, ν) lookup table at module load. |
| **Reduced regularization on location parameter** | Tier 3 (ZINB) #7; was original P6 | hours | Larger `num_leaves`, smaller `min_data_in_leaf`, deeper `max_depth`. Marginal effect. Try only after Tier 1–2. | Optuna search-space dict. |
| **MERF-style iteration with LightGBMLSS** | Tier 4 (ZINB) #9; was original P9/P10 fallback | 2–3 weeks | Alternating fit-residual / re-estimate-shrunken-per-player-baseline loop. Reserve as fallback if Stage B3 (both MZINB and GPBoost) prove infeasible. | New module under `src/sportstradamus/training/`. |
| **Quantile / expectile heads alongside the ZINB** | Tier 5 (ZINB) #11; was original P7-equivalent | days | **Different use case** — DFS ceiling and over bets, not the bias fix. Add alongside, not instead of, Tier 1–2. Sluijterman et al. 2025 arctan pinball loss is a drop-in replacement for standard pinball with better-calibrated extremes. | New quantile head pickled alongside the ZINB pickle. |
| **Isotonic post-hoc calibration on ZINB-mean** | Tier 5 (ZINB) #12; was original P8 | hours | Polish. Mops up residual average bias after architectural fixes. Don't lead with it. | Wrap predictions in `IsotonicRegression(out_of_bounds="clip")`. |
| **Sample reweighting on high-scoring games** | Tier 5 (ZINB) #13; was original P9 | hours | Last resort. Increases variance in the upweighted region. | LightGBM `sample_weight` arg. |

**What NOT to do (researcher-specified warnings):**
- Do **not** lead with quantile/expectile heads as a bias fix — they change
  which point of the distribution is predicted, not the underlying bias.
- Do **not** lead with isotonic calibration. It's polish, not a primary fix.
- Do **not** port to cyc-GBM or CyclicBoosting before trying per-parameter
  Optuna inside the existing LightGBMLSS stack. Port cost is unjustified
  until you've ruled out the cheap version of the same idea.
- Do **not** pursue per-minute decomposition on count markets in parallel
  with Track A — leverage is materially lower on counts (STL/BLK are
  low-mean stochastic events, not rate-times-minutes phenomena) and the
  engineering cost overlaps.
- Do **not** commit to both MZINB and GPBoost. Pick based on Stage B2
  residual diagnostics.

**Decision points (gate logic for Stage B4):**
- If per-parameter Optuna alone hits the threshold → declare done; don't
  proceed to CMP/MERF.
- If STL's var/mean ratio < 1.0 → CMP becomes a primary fix, not Stage B4
  polish. Move it up to Stage B3.
- If everything in B4 fails → loop back to a Track-A-style live-path audit:
  the FTM/STL kill may actually originate downstream in
  [prediction/model_prob.py](../src/sportstradamus/prediction/model_prob.py),
  not in training. See OVERCONFIDENCE_INVESTIGATION §3.4 for the live-path
  confound playbook.

---

## Open questions (researcher-flagged, unresolved)

1. **Feature-predictive-power asymmetry between zero-vs-count splits at the
   market level is not in the published literature.** Your TOV-vs-STL puzzle
   (similar surface stats, opposite hurdle outcomes) is not directly
   addressed; the closest analogue is Feng (2021)'s finding that NB
   outperforms ZINB at ~20% zero rates (STL at 48% is well above that
   regime). You may be observing a genuine domain-specific phenomenon that
   would be publishable in its own right. Capture data and patterns
   discovered in this project for a possible write-up after Stage B
   concludes. (ZINB researcher caveat #5.)
2. **No GBDT precedent for MZINB.** First implementation in boosting will be
   yours. Expect 3–5 cycles of debugging the gradient computation for
   log(ν) → μ_implied = ν/(1−ψ) before the model trains stably. The
   empirical case for boosted MZINB resolving the deterministic-mode
   blowups would itself be a novel contribution. (ZINB researcher caveat #1.)
3. **MEGB's headline 35–76% MSE improvement is from simulations**, not
   from a real high-n low-p panel matching the NBA regime. Transfer is the
   bet, not a confirmed result. (SkewNormal researcher caveat.)
4. **DEGPD / ZIDEGPD count distributions** (Ahmad & Hussain 2025) are very
   new with no production-grade Python implementation. Implementation from
   the paper required if the routing diagnostic points there. Deliberately
   not staged into Track B — too speculative for the current plan.
5. **CMP normalizing-constant lookup table** implementation cost is non-trivial.
   Worth it only if Stage B1 var/mean diagnostics strongly suggest under-dispersion
   for at least one market.
6. **CatBoost ordered TS unbiasedness was proven for log-loss / squared-error**;
   the proof does not directly extend to a distributional SkewNormal NLL.
   Mechanism should still help, but validate empirically. (SkewNormal
   researcher caveat.)
7. **The Chevalier & Côté (2025) benchmark warns that probabilistic GBM is
   not uniformly better than point-prediction GBM on NB-class targets.**
   Validate any architectural switch (MZINB, cyc-GBM, CMP) against the
   current LightGBMLSS-ZINB baseline on probabilistic metrics (CRPS,
   log-score) **and** on the downstream MAE metric, not just one of these.
8. **None of these approaches solves "high-volume players get under-predicted"
   if it is fundamentally a player-effect issue.** Best long-term architecture
   probably combines (a) MZINB likelihood, (b) per-player random intercept
   à la GPBoost, and (c) per-parameter early stopping à la cyc-GBM — but
   each piece is an independent A/B and should be staged as above.

---

## Critical files

| File | Role | Key lines |
|---|---|---|
| [src/sportstradamus/training/pipeline.py](../src/sportstradamus/training/pipeline.py) | target build, dist select, training, denorm, test_set dump | 245–324 (branch/target), 328 (`lgb.Dataset` — `init_score` injection point), 341/394–409 (`set_model_start_values`), 345–346 (MeanYr monotone), 348–368 (Optuna search space), 439–452 (SkewNormal denorm), ~960/981 (test_set dump) |
| [src/sportstradamus/training/report.py](../src/sportstradamus/training/report.py) | diagnostics → `training_report.txt`, `model_stats.parquet` | `ev_meanyr_corr`/`result_meanyr_corr` (~850), `write_model_stats` |
| [src/sportstradamus/stats/base.py](../src/sportstradamus/stats/base.py) | baseline features + target; inference-time mirror lives here | 597 (`get_stats`), 676–702 (`MeanYr`, `Mean10`, `*_Ratio`), 1005/1011/1082 (`Result`) |
| [src/sportstradamus/stats/nba.py](../src/sportstradamus/stats/nba.py) | NBA `MIN`, `USG_PCT`, per-48 stats | 127–135, 359, 366 |
| [src/sportstradamus/helpers/distributions.py](../src/sportstradamus/helpers/distributions.py) | `set_model_start_values`; `fused_loc` (book blend) | 425–504 |
| [src/sportstradamus/skew_normal.py](../src/sportstradamus/skew_normal.py) | custom SkewNormal (location-scale, supports negatives) | 30–199 |
| [src/sportstradamus/hurdle.py](../src/sportstradamus/hurdle.py) | HurdleZINB (Stage 2 ZTNB lives here for Track-B Stage B1) | ~201 (NegBin loss for Stage 2) |
| [src/sportstradamus/scripts/compression_eval.py](../src/sportstradamus/scripts/compression_eval.py) | **P0 harness** — decile table, compression ratio, run log, diff verdict | — |
| [src/sportstradamus/prediction/model_prob.py](../src/sportstradamus/prediction/model_prob.py) | **Live-path confound** — where shipped strategies must survive end-to-end | SkewNormal decode, `fused_loc` w≈0.9 blend, `temperature`≈1.37 |
| [docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md](superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md) | Source spec for the **ZINB derived-π gate** fix (P2.B precursor) | Phase B "SUPERSEDED → derived-π" |

## Verification (every code session)

- `poetry run ruff check src/sportstradamus/`
- `poetry run pytest tests/golden/` (incl. `test_compression_eval.py`)
- `poetry run pytest -m integration` (fake-mode, no network)
- Regenerate CLI help snapshots if `meditate` flags change:
  `REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py`
- Determinism gate (P0.5): `poetry run pytest tests/integration/test_determinism_gate.py -v -m integration`
- Functional gate: harness scorecard delta vs current strategy meets the P0
  threshold before a strategy is promoted to default.
- Live-path gate: the promoted strategy is confirmed end-to-end through
  [prediction/model_prob.py](../src/sportstradamus/prediction/model_prob.py)
  (no `Model Skew`=NaN, EV not collapsed by the book blend), not only on the
  dumped test set.

## Session handoff

### Per-session rules

- One strategy/experiment per session where feasible (aligns with CLAUDE.md
  "one module per subagent" — the per-strategy scope discipline, not the
  serial execution); commit + push to `claude/fix-gbdt-mean-regression-GcY1g`
  and update the harness run log so the next session sees the scorecard history.
- Keep the default strategy = current production behavior until an experiment
  clears the threshold, so `devel`-tracking production is never regressed
  mid-project.
- Record each experiment's scorecard verdict (ship/kill) in the run log committed
  to the repo (not a scratch doc), and update the **Status / progress log** table
  at the top of this file.
- Track A and Track B can be worked in separate sessions / by separate
  subagents; they share no mutable state beyond the harness. The strategic
  fork in Stage B3 is the only place where a single decision blocks
  multiple downstream sessions.

### Tooling note: `gh` is a userspace install on this workstation

PR #46 CI status and review-comment monitoring use the GitHub CLI. `gh` is
**not a system package** — it lives at `~/.local/bin/gh` (installed
2026-05-19 via the official static tarball release, since `sudo apt` would
have required an interactive password). Future sessions must ensure
`~/.local/bin` is on `PATH`; on this workstation it already is (set in
`~/.profile` / `~/.bashrc`), but a sandboxed or non-login shell may not
inherit it. If `gh --version` fails, run:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Authentication is also a one-time setup the user completes locally
(`gh auth login` interactive, or `export GH_TOKEN=…` from a PAT with `repo`
scope). Agent sessions don't have credentials by default — if `gh api …`
returns `HTTP 401`, the user needs to re-auth.

### Branch / PR / commit refs

- Branch: `claude/fix-gbdt-mean-regression-GcY1g`
- PR: #46 (→ `devel`)
- HEAD at this plan rewrite: `6e913b1` ("docs: add research handoff for
  centered-target negative result")
- Latest shipped: P2.B HurdleZINB (commit `cee5625` ships
  `centered_additive_eb_meanyr_k10`; subsequent commits add the verdict and
  the HurdleZINB landing); P1 follow-up `1d0e65e` adds
  `centered_additive_mean10` as the path-wide A/B counterexample.
