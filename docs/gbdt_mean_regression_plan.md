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
| **Stage 0 — live-data instrumentation** | ✅ done (PR #46) | All five deliverables shipped: 0.1 `compute_book_brier_skill_score` in [analysis.py](../src/sportstradamus/analysis.py) (8 unit tests, hand-computed-reference within 1e-6); 0.2 `_compute_live_metrics` + Step 6 in [nightly.py](../src/sportstradamus/nightly.py) writing `data/live_metrics_per_market.parquet` with the locked 10-column / 2-window schema (6 round-trip tests including empty-window n_settled=0 case); 0.3 `compression_eval --live-window N` in [scripts/compression_eval.py](../src/sportstradamus/scripts/compression_eval.py) using `_history_to_eval_frame` + per-league `_load_league_stats_lookup` (functools.cached) with Stats-backed MeanYr lookup, mockable via monkeypatch (8 new tests in [tests/golden/test_compression_eval.py](../tests/golden/test_compression_eval.py)); 0.4 [scripts/check_graduation.py](../src/sportstradamus/scripts/check_graduation.py) CLI joining Gate 1 (model_stats.parquet) × Gate 2 (live_metrics_per_market.parquet, 30d window) with `_classify_lifecycle` → {not-shipped, in-test, graduated, demoted}, colored stdout table (11 tests including 7 parametrized state-machine cases); 0.5 [scripts/backfill_live_metrics.py](../src/sportstradamus/scripts/backfill_live_metrics.py) walking history backwards with `--days/--step` controls + idempotent dedup on day-precision `computed_at` (5 tests). All three always-on gates green (ruff clean, 113 golden pass, 9 integration pass); 30 new Stage 0 tests added. refactoring-specialist sweep applied two minor fixes (nightly docstring step list, `_classify_lifecycle` line-length split). Two new console scripts registered in `pyproject.toml`: `check-graduation`, `backfill-live-metrics`. Default behavior of `meditate`/`prophecize`/`confer` byte-identical; `reflect` gains tail Step 6 only. **Track A / Track B graduation lookups are now a parquet read** — see "Stage 0 — Live-data instrumentation" section below for the lifecycle classifier rules. |
| **Stage B1 — ZTNB likelihood fix + routing diagnostics** | ✅ done, result: **ZTNB hypothesis REFUTED; routing rescope delivered** | **B1.1 (ZTNB):** correct in isolation — the zero-truncated NB recovers an unbiased count component (`tests/test_ztnb_loss.py`, scipy-referenced) — but **incompatible with the frozen derived-π hurdle decode**. On FG3M the ZTNB count component implies `NB(0) ≈ 0.41` vs observed `q ≈ 0.31`, so on **65 % of rows `NB(0) > q`** → derived-π clips to 0 → the reconstruction identity breaks (`tests/integration/test_zinb_hurdle_live_path.py` diff **0.136** vs 0.02 tol). E[Y\|Y>0] is ~unchanged (old 2.2 vs ZTNB 2.12); ZTNB only re-decomposes the positive mean, which is exactly what breaks the decode. The fix would **regress the 6 P2.B SHIP markets**, not just fail FTM/STL. Wire-in **reverted**; `_ZeroTruncatedNB` kept as an unwired, test-covered building block for MZINB (B3). Smoke A/B not run (no stats.nba.com network in the build env) — analytical verdict is KILL; commands handed to the user. **B1.2 (routing):** new read-only `scripts/zinb_routing_diagnostics.py` + golden test + `statsmodels` dep; writes `data/zinb_routing/{LEAGUE}_diagnostics.parquet` for all 23 cells. Result: **0/23 route to `hurdle_nb_ztnb`; 13 → `cmp` (underdispersed, var/mean ≤ 1.3), 10 → `mzinb`.** The blanket ZINB label in `stat_dist.json` is wrong for ≥ 13 markets. **STL → cmp** (underdispersed, the kill's real cause), **FTM → mzinb** (genuinely inflated+overdispersed). Tooling: new `meditate --market` flag; `select_markets` relocated `cli.py → training/markets.py`. See "Stage B1 — outcome & Track-B rescope" below. |
| **Stage A1 — SkewNormal ICC diagnostic gate** | ✅ done, result: **family clusters AMBIGUOUS — ICC alone does not cleanly route the SkewNormal family** | New read-only `scripts/icc_diagnostics.py` (console script `icc-diagnostics`) + `tests/golden/test_icc_diagnostics.py` (15 tests). Computes ICC₁ via a two-level moment decomposition (σ²_between = Var(player-season means), σ²_within = mean within-(player-season) variance) over **(Player, season)** groups — season via the `stats/base.py:527-528` Aug-boundary rule, since the NFL/WNBA cached parquets are multi-season (NFL 2021–26, WNBA 2022–25; NBA single-season). Participation filter (nonzero-game fraction ≥ 0.5, **no position map**) resolves the NFL position-confound; skew-driven transform escalation raw→log1p→rank (all 36 cells landed `raw`). Writes `data/icc/{NBA,WNBA,NFL}_icc.parquet`, one row per (league, market); all 36 cells produced. **Routing verdict: 25 ambiguous, 10 eb_centering, 1 tail_extension.** NBA (ICC 0.27–0.51): only PA 0.514→eb_centering and DREB 0.274→tail_extension, other 11 ambiguous; **FGA 0.489 (NOT the predicted >0.6)**, **PTS 0.473** (just over the predicted 0.3–0.45 band). WNBA (0.37–0.57, slightly *higher* than NBA, not noisier — 4-season (player,season) pooling, n_player_seasons ≈ 480–530, is stable): 5 eb_centering (MIN/FGA/PRA/PA/PR). NFL (0.41–0.79, widest spread): qb-yards 0.790, carries 0.666, targets 0.507, rushing-yards 0.502 → eb_centering; the participation filter excluded ~1380–1391 non-QB player-seasons on passing-yards/attempts/completions (kept n_players ≈ 90–92 = QBs), confirming the position-confound fix. **Decision triggers:** "ICC_PTS > 0.5 on any league" did NOT fire (entanglement hypothesis stands; T7 does not jump to A2); there is NO bimodal split and the family is NOT uniformly-low → the 25 ambiguous cells sit in the plan's "try both, route on outcome" band, so A2 should run EB-centering *and* tail-extension per-market rather than route off ICC alone. **ICC does not predict the P1 EB ship/kill** — FGA SHIPPED at ICC 0.489 while PA (highest NBA ICC, 0.514) KILLED; caveat: ICC is unconditional and the production model already carries a wide feature matrix (~280 columns) that captures much between-player level, so unconditional ICC is necessary context, not a sufficient router. **eb_K** per-league median ≈ NBA 1.4 / WNBA 1.1 / NFL 1.2 — **NFL is NOT an outlier**, so the "K=10 is wrong for NFL *specifically*" suspicion is unconfirmed; further caveat: the moment eb_K = σ²_within/Var(player-season means) is a downward-biased estimate of the Casella–Berger EB constant (observed between-player variance is inflated by σ²_within/n̄), so the table cannot assert "K=10 too high" — re-derive K with a bias-corrected estimator before acting on it in A2. No model/inference/pickle change; default flags byte-identical. All three always-on gates green (ruff clean, golden 140 incl. 15 new, integration 11); refactoring-specialist run (added an `Args` block + test type hints, no logic change). |
| P3–P10 | ⬜ | see priority list; P10 (GPBoost) already prototyped and failed deterministically — annotated below |

## Scope — leagues this plan covers

The training pipeline ships models for three leagues. Every method below is
applicable to all three unless explicitly flagged as league-specific.
Market counts (from `data/stat_dist.json`):

| League | SkewNormal markets | ZINB markets | Games/season per player | EB K (current) |
|---|---|---|---|---|
| NBA | 13 (incl. PTS, REB, AST, PRA, FGA, MIN, PA, PR, FG3A, FGM, fantasy points) | 8 (FG3M, FTM, OREB, PF, STL, TOV, BLK, BLST) | ~82 | 10 |
| WNBA | 11 | 7 (same names as NBA minus PF) | ~40 | 10 (likely fine) |
| NFL | 12 (incl. passing/rushing/receiving yards, attempts, carries, targets) | 8 (passing tds, tds, rushing tds, receiving tds, qb tds, interceptions, sacks taken, passing first downs) | ~17 (regular season) | 10 (almost certainly wrong; see caveats) |

NBA and WNBA share the same ZINB market names by construction — the same
basketball stat universe with PF dropped on WNBA. NFL's ZINB markets are
different names (touchdown counts, sacks, interceptions, etc.) but the same
structural problem (low-mean count stats with zero inflation).

`meditate --league {NBA,NFL,WNBA}` is already wired in
[training/cli.py:36-38](../src/sportstradamus/training/cli.py#L36-L38); the
per-league dispatch loop at
[training/cli.py:218-261](../src/sportstradamus/training/cli.py#L218-L261)
trains each market in each requested league through the same
`train_market` orchestrator. The compression_eval harness itself is
league-agnostic — it scores per-market CSVs and does not care which
league they came from.

## Cross-league testing policy (applies to every method below)

Every change goes through two test phases before shipping:

1. **Smoke phase (start of work):** pick 1–2 representative markets per
   league. For Track A pick the canonical SkewNormal market of the league
   (NBA: FGA + PTS; WNBA: FGA + PTS; NFL: passing yards + receiving
   yards). For Track B pick a SHIP and a KILL from the existing P2.B
   verdict where they exist (NBA: FG3M + FTM/STL; WNBA: FG3M + STL;
   NFL: pick the highest-zero-rate ZINB market + the lowest). Smoke
   phase must pass before further development.
2. **Full-verification phase (before any default-flag flip):** run the
   compression_eval A/B on **every market in every covered league** that
   uses the affected distribution branch. SkewNormal-only changes touch
   13 + 11 + 12 = 36 markets; ZINB-only changes touch 8 + 7 + 8 = 23
   markets. SHIP requires the universal decision threshold below to be
   met on every market in every league, or per-league/per-market
   routing config that documents the exceptions.

The deterministic test sets land under `data/test_sets/deterministic/{strategy}/`
keyed by filename (`{LEAGUE}_{market}.csv` already), so the per-league
output is already separable — no schema change needed to support the
cross-league A/B.

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

**Universal decision threshold (every experiment, every market, every covered league):**
ship a strategy only if it reduces **top-mean-decile MAE by ≥ 5%** vs the
current production strategy without worsening **global MAE by > 1%** and
without worsening `brier_skill_score` on the existing report. The threshold
must hold on **every market in every covered league** — or the routing
config records the per-market/per-league exceptions. Otherwise kill it
and move on. The harness —
[src/sportstradamus/scripts/compression_eval.py](../src/sportstradamus/scripts/compression_eval.py)
— is the ship/kill gate; the cross-league A/B is the
full-verification phase from the testing policy above.

## Diminishing returns — stop-the-track principle

Don't let perfect be the enemy of good. Each stage exists because the
*previous* stage's verdict suggested more lift was available; if live data
shows the deployed model is already calibrated and profitable on a given
market or league, **stop that track for that market/league and redeploy
the engineering effort elsewhere**. The plan is a backlog, not a queue.

### Lifecycle: offline gate → production test → live gate → graduation

Every (league, market) cell moves through three states. Two gates control
the transitions. Both gates appear in the same `check_graduation` table
(Stage 0 deliverable 0.4) so a single view answers "what changed, what's
in test, what's graduated."

```
not-shipped  ─[Gate 1: offline]→  in-production-test  ─[Gate 2: live]→  graduated
   ▲                                       │                                │
   └────── re-entry / revert ←─────────────┴────[live regression]───────────┘
```

### Gate 1 — Offline ship gate (qualifies a cell for production test run)

Computed on the **held-out validation + test split** that `train_market`
already produces (lines 547-553 in [pipeline.py](../src/sportstradamus/training/pipeline.py)
and the deterministic test_set CSVs under `data/test_sets/`). This is the
existing universal decision threshold restated as a per-cell gate.

| Offline metric | Threshold | Where it lives |
|---|---|---|
| Top-mean-decile MAE on the test split | ≥ 5% better than current default strategy on the same cell | `compression_eval --baseline ... --candidate ...` output |
| Global MAE on the test split | not worse by > 1% vs current default | `compression_eval` global summary |
| `brier_skill_score` on the validation split (book baseline) | not worse than current default | `model_stats.parquet` for the candidate run |
| Determinism gate (when changing the deterministic-mode pipeline) | green for every league with cached parquets | `tests/integration/test_determinism_gate.py` (current NBA-only; Stage 0 prerequisite extends to WNBA + NFL) |

If **all four** clear on every cell in every covered league (or the
routing config records the exceptions), the strategy is allowed to
promote: change becomes the new default for those cells on `devel`, runs
in production for the **mandatory ≥ 14-day soak window** before the live
gate is evaluated. During the soak, the previous version's pickle stays
archived under `data/old_models/` so revert is one cron-pull away.

### Gate 2 — Live graduation gate (cell graduates from the track)

Computed on the **last 30 days of settled production offers** by the
Stage 0 `compute_book_brier_skill_score` and rolling-window aggregator
in `nightly.py`. A cell graduates from the track — no further stage work
— if all four hold:

| Live metric | Threshold | Where it lives |
|---|---|---|
| Settled book-BSS (30 days, ≥ 200 offers) | ≥ 0 AND ≥ training-set `brier_skill_score − 0.02` (no live-vs-offline regression > 0.02) | Stage 0 deliverable 0.1 (`compute_book_brier_skill_score`); persisted by 0.2 in `data/live_metrics_per_market.parquet` |
| Empirical over-rate vs predicted over-rate on settled offers | within ±0.03 over ≥ 200 settled offers | same parquet — Stage 0 0.2 |
| Top-decile live MAE on settled bets | ≥ 5% better than prior-version live MAE on the same cell, OR within 5% of the offline compression_eval test-set MAE (i.e. no live-vs-offline drift) | Stage 0 deliverable 0.3 (`compression_eval --live-window 30`) |
| Profit-sim parlay yield | non-negative on slates containing the cell | dashboard Stats Profit Sim page; Stage 0 0.2 aggregates per-cell into the same parquet |

If a cell graduates, mark it ✅ in the Status table with the graduating
stage and the live metrics that triggered graduation. The track continues
on the non-graduating cells only. This is the live-data analog of P1's
"FGA-only SHIP" verdict — FGA graduated at Stage A2 (effectively), the
rest of the SkewNormal family did not.

### Why both gates need to exist

Gate 1 (offline) without Gate 2 (live) is the failure mode P1 actually
hit — `centered_additive_eb_meanyr_k10` cleared the offline gate on FGA
under deterministic mode, but the production-mode pipeline had different
hyperparameters and the live data could have looked different from
test-set CSVs. The 14-day soak window is what catches that drift.

Gate 2 without Gate 1 is the inverse failure: shipping changes to
production on live-data intuition alone, no offline ship verdict to
revert to. The plan rejects both modes — every cell must clear Gate 1,
soak, then clear Gate 2 before graduating.

### Track-wide stop condition

A whole track stops when **every cell has graduated**. At that point any
remaining staged work (e.g. Stage B3 MZINB if Stage B1 ZTNB already
graduated 8/8 ZINB cells across NBA/WNBA/NFL) is filed under "future
research" — code in `src/deprecated/` if prototyped, otherwise just
noted in the Status table as deprioritized with a one-line reason
(e.g. "B3 not pursued: B1 graduated 23/23 cells; further structural
change would be perfect-as-enemy-of-good").

### Re-entry condition

If a graduated cell *regresses* — settled brier_skill_score drops below
the graduation threshold for two consecutive 7-day windows — it re-enters
the track at the stage where it graduated. Track work resumes on that
cell only. Re-entry is logged in the Status table with the regression
metric so future sessions can read the pattern (e.g. "WNBA FG3M re-entered
2026-09-12 after settled brier_skill dropped to −0.04 vs graduation 0.11
— preseason rotations changed minutes patterns").

### Stage 0 — Live-data instrumentation (prerequisite for everything else)

**The graduation criteria above reference metrics that are not currently
persisted in a usable form.** Before any track work that intends to use
the stop-the-track principle, Stage 0 must ship. Without it, "settled
brier_skill ≥ 0 over the last 30 days" is rhetorical, not actionable.

What exists today:

| Component | State |
|---|---|
| [analysis.py:877](../src/sportstradamus/analysis.py#L877) `compute_brier_skill_score(subset, base_rate=0.5)` | Computes BSS against base-rate 0.5 (chance), **not against the book** — different quantity than training-side `brier_skill_score` which uses the book baseline. Not directly comparable to the training metric the plan references. |
| `history.parquet` with settled `Actual` column | Exists; filled by [nightly.py](../src/sportstradamus/nightly.py) on every `reflect` run. Per-offer granularity; not aggregated per (league, market) or over rolling windows. |
| Dashboard pages 3/4/6 | Compute brier / BSS / profit-sim **on demand** from `history.parquet` when a user opens the page. Not persisted; future sessions cannot read these values programmatically. |
| Live top-decile MAE on settled bets | **Does not exist.** [compression_eval.py](../src/sportstradamus/scripts/compression_eval.py) only scores `data/test_sets/` CSVs (offline held-out test rows), not live settled offers. |
| Per-(league, market) persistent metric store | **Does not exist.** No `data/live_metrics_per_market.parquet`. |

Stage 0 deliverables (in dependency order):

| # | Deliverable | Cost | Where it lives |
|---|---|---|---|
| 0.1 | **Live brier_skill_score against the book baseline** — variant of [analysis.py:877](../src/sportstradamus/analysis.py#L877) that uses the bookmaker's implied prob (already stored per-offer in `history.parquet` as the original `Odds` column) as the reference, mirroring the training-side `brier_skill_score` exactly. Add as `compute_book_brier_skill_score(subset)` alongside the existing function — keep the chance-baseline version (some dashboards already rely on it). | ~2 hours | [analysis.py](../src/sportstradamus/analysis.py) |
| 0.2 | **Rolling-window aggregation per (league, market)** — extend `reflect` to compute, for each (league, market) cell at the end of every nightly run: 7-day and 30-day rolling book-BSS, empirical-vs-predicted over-rate, total settled bets, and total profit-sim yield. Write to `data/live_metrics_per_market.parquet` keyed by (league, market, computed_at). | ~1 day | New `compute_live_metrics()` step at the tail of [nightly.py](../src/sportstradamus/nightly.py) `run()`. |
| 0.3 | **Live top-decile MAE harness mode** — new `compression_eval --live-window N` flag that reads `history.parquet` instead of a test-set CSV, filters to the last N days of settled bets per (league, market), and runs the existing decile-bias scoring code path. Output schema matches the offline mode so the comparison "live vs offline top-decile MAE" is a parquet join. | ~1 day | [compression_eval.py](../src/sportstradamus/scripts/compression_eval.py); add unit tests in [tests/golden/test_compression_eval.py](../tests/golden/test_compression_eval.py). |
| 0.4 | **Lifecycle-status table view** — small CLI (`poetry run check-graduation`) or dashboard page that joins `data/live_metrics_per_market.parquet` against **both gates** and emits a per-(league, market) status: not-shipped / in-test (days into soak) / graduated. Renders as a 36-cell × 8-metric table — 4 Gate 1 (offline) columns + 4 Gate 2 (live) columns + a state column. Future sessions read this to decide both "can this candidate ship to test?" (Gate 1) and "can this cell graduate?" (Gate 2). | ~half day | New [src/sportstradamus/scripts/check_graduation.py](../src/sportstradamus/scripts/check_graduation.py) or a new dashboard page. |
| 0.5 | **Backfill rolling metrics on existing history** — one-shot script that walks back through `history.parquet` and emits the rolling-window points for the last ~90 days so Stage 0's first publication of `live_metrics_per_market.parquet` has historical context to compare against. | ~half day | One-shot script under [src/sportstradamus/scripts/](../src/sportstradamus/scripts/). |

**Total Stage 0 cost: ~3 days.** Modest — and the entire stop-the-track
principle becomes empirically grounded instead of rhetorical.

**Stage 0 ship gate** (separate from the universal threshold — Stage 0 is
infrastructure, not a model change):
- `live_metrics_per_market.parquet` exists and has ≥ 7 days of history at
  publication
- Live book-BSS computed by 0.1 matches a hand-computed reference within
  1e-6 on a 100-row spot check from `history.parquet`
- `compression_eval --live-window 30` produces a deterministic output on a
  fixed history.parquet snapshot (frozen for the test) and the golden
  test asserts row-count + schema
- A `check_graduation` invocation on the cached snapshot produces a
  non-empty table covering NBA / WNBA / NFL × all distributions, with
  the lifecycle state column populated for every cell (not-shipped /
  in-test with day count / graduated)
- The 8-metric output (4 Gate 1 + 4 Gate 2) reads from the same
  parquet on every invocation — no recomputation of Gate 1 from
  `model_stats.parquet` and Gate 2 from a separate source

**Stage 0 stop-the-track check:** there is no "we have enough live data
already" exit for Stage 0 — it's the infrastructure that makes
graduation decisions possible. Skip Stage 0 only if you also skip the
stop-the-track principle (which means committing to running every
remaining stage on every cell — exactly the perfect-as-enemy-of-good
mode the principle is designed to prevent).

**Why Stage 0 doesn't compete with the actual model work:** Stage 0 ships
once. Every subsequent stage's graduation check is a 30-second parquet
read. Across the remaining ~12 stages × 36+23 cells, the amortized cost
is negligible. Without Stage 0, every graduation check requires either a
manual dashboard inspection (slow, not programmatic, won't survive
future sessions) or re-deriving the metric in an ad-hoc script (which is
Stage 0 done badly).

## Inference-path compatibility (applies to every shipped change)

Every change in this plan must land its inference-side mirror in the
**same PR** before promotion to production. Gate 1 above lets a change
into the test-run window; the inference-path checklist below is what
makes that window safe to enter.

The architectural principle near the top of this plan already requires
this for target/baseline transforms ("Centralize the forward transform,
the inverse (de-norm) transform, and the inference-time mirror"). This
section restates it as a hard requirement covering **every** change
type, not just target transforms — and it names the concrete files and
seams so the work is unambiguous.

### Inference path (where things live)

| Component | File / lines | Role |
|---|---|---|
| CLI entry | `prophecize` → `sportstradamus.prediction.cli` | Loads slate, dispatches per-market |
| Model load | [prediction/__init__.py](../src/sportstradamus/prediction/__init__.py) `main()` | Reads pickle from `data/models/{LEAGUE}_{market}.mdl` |
| Per-offer scoring | [model_prob.py:114](../src/sportstradamus/prediction/model_prob.py#L114) `model_prob()` | The function the user is asking about — runs once per offer |
| Per-offer features | [stats/base.py:597](../src/sportstradamus/stats/base.py#L597) `get_stats()` | Builds the `playerStats` row consumed by `model_prob`. **Inference mirror of `get_training_matrix`** — any new training feature must be computed identically here (leakage-safe). |
| Per-distribution decode | [model_prob.py:259-272](../src/sportstradamus/prediction/model_prob.py#L259-L272) (NegBin/ZINB/Gamma/ZAGamma) + [model_prob.py:276](../src/sportstradamus/prediction/model_prob.py#L276) (SkewNormal via `_decode_skewnormal`) | Per-`dist` block that turns `predict(pred_type="parameters")` output into `Model EV` + `Model Gate` + `Model R/Alpha` columns |
| Hurdle dispatch | [model_prob.py:205](../src/sportstradamus/prediction/model_prob.py#L205) `getattr(model, "is_hurdle", False)` | The pattern P2.B introduced for opting a model into a non-default predict path. **Every new model class follows this pattern.** |
| Distribution-specific blend | [helpers/distributions.py:69](../src/sportstradamus/helpers/distributions.py#L69) `get_ev`, [314](../src/sportstradamus/helpers/distributions.py#L314) `fused_loc`, [163](../src/sportstradamus/helpers/distributions.py#L163) `get_odds`, [425](../src/sportstradamus/helpers/distributions.py#L425) `set_model_start_values` | Every distribution name the plan introduces must round-trip through all four; the `dist` string is the dispatch key. |
| Pickle schema | [pipeline.py:1940](../src/sportstradamus/training/pipeline.py#L1940) `_build_filedict` (writer) ↔ [model_prob.py](../src/sportstradamus/prediction/model_prob.py) + [training/pipeline.py](../src/sportstradamus/training/pipeline.py) (readers) | Any new pickle key must be added to the writer AND read back by every consumer. Legacy pickles must still load — use `filedict.get("new_key", legacy_default)`. |

### Per-change-type inference checklist

Use this table to identify what needs to land alongside each Track A / B
method. The table is keyed on change type, not stage — many stages
include multiple change types.

| Change type | Inference-side work required | Reference precedent |
|---|---|---|
| **Training-only** (T6 FAGTB adversarial objective, T9 monotone constraint, B1 ZTNB likelihood, B4 per-parameter Optuna, B4 reduced regularization, B4 sample reweighting) | **None.** Output schema (the `(total_count, probs, gate)` triple for ZINB; `(loc, scale, alpha)` for SkewNormal; etc.) is unchanged. The pickle round-trips. | B1 ZTNB fix: only loss changes; `predict(pred_type="parameters")` still returns the same columns. |
| **New target/baseline strategy** (P1-style — already done; future Track-A variants in the same family) | Inverse decode in `model_prob.py:_decode_skewnormal` via `baselines.STRATEGY_REGISTRY[strategy].decode_loc/decode_scale`; matching `*_Ratio` feature in `stats/base.py:get_stats`; `target_strategy` + `offset_meta` pickle keys round-trip. | P1 `centered_additive_*` strategies — registry entry, live-path test, pickle field, decode dispatch. |
| **New distribution head** (T3 spliced/Pareto, T10 PGBM, B3 MZINB, B4 CMP, B4 quantile heads, T4 MEGB output, T7 gbex) | (a) New decode block in [model_prob.py:259-272](../src/sportstradamus/prediction/model_prob.py#L259-L272) that turns `predict(pred_type="parameters")` into `Model EV` columns; (b) [helpers/distributions.py](../src/sportstradamus/helpers/distributions.py) `get_ev`, `get_odds`, `fused_loc`, `set_model_start_values` each accept the new `dist` name; (c) `dist` string in `_build_filedict` + the legacy `dist=…` fallback in `model_prob`; (d) new live-path integration test mirroring [test_zinb_hurdle_live_path.py](../tests/integration/test_zinb_hurdle_live_path.py) asserting `Model EV` is finite, `Model Gate ∈ [0,1]` where applicable, and two runs identical under `DETERMINISTIC_SEED`. | P2.B HurdleZINB: `getattr(model, "is_hurdle", False)` gate, `zinb_mode` + `is_hurdle` pickle keys, identity-reconstruction test, determinism gate parallel assertion. |
| **Post-hoc calibration object** (A3 isotonic on loc, T8 CQR/LCMQR, B4 isotonic on ZINB-mean) | Pickle the calibration object as a new key (`isotonic`, `cqr`, `temperature` already exists as precedent); load it in `model_prob` and apply after the distribution decode but before `fused_loc` (or after — depends on what's calibrated). Round-trip test asserts byte-identical predictions across save/load. | `temperature` field on the existing pickle dict ([pipeline.py:1958](../src/sportstradamus/training/pipeline.py#L1958)) and the calibrated proba path. |
| **New player-level feature** (B2 leakage-safe target-encoded `expanding().mean().shift(1)` per player) | Feature column added to BOTH `Stats.get_training_matrix` (training) AND [stats/base.py:get_stats](../src/sportstradamus/stats/base.py#L597) (inference) — computed identically, leakage-safe at inference time (no peek at the current game). Same dtype + index across train/inference. Add the feature to `feature_filter.json` whitelist for affected markets. | `MeanYr` / `Mean10` / `*_Ratio` columns at `stats/base.py:676-702` and the leakage tests at `test_meanyr_mean10_leakage.py`. |
| **Multi-head factorization** (T5-basketball, T5-NFL) | `prophecize` loads N pickles per market (one per factor in the factor tree); `model_prob` Monte Carlos: sample from each factor's predicted distribution at inference time, multiply, derive the marginal distribution of the composed target; `fused_loc` may need a multi-output blend variant that respects the joint distribution. New live-path test drives multi-head decode end-to-end. Pickle schema: new `factor_pickles: dict[str, Path]` field on the parent market pickle. | None in-repo. Closest analogue: the per-market book_weights handling. Expect this to be the largest inference-side change in the plan. |
| **Different model class** (T2 CatBoost ordered TS, T4 MEGB native R, B3 GPBoost) | New `getattr(model, "is_catboost", False)` / `"is_gpboost", False` flag on the model class; `model_prob` dispatches accordingly. `prediction/__init__.py` model-load path branches on the new class. Determinism gate extended with the new class. If the alternative class doesn't have the LSS `predict(pred_type="parameters")` API, the dispatch must adapt. | P2.B `is_hurdle` pattern — identical structure, just a new attribute name. |

### Inference-path test as a hard ship gate

Any change requiring inference-side work must have a passing live-path
integration test under `tests/integration/` **before promotion to
production** (i.e. before Gate 1 from the lifecycle section is even
checked — the test asserts the inference path doesn't blow up on cached
real data). The test must assert:

1. `Model EV` is finite for every offer.
2. For ZI-class distributions: `Model Gate ∈ [0, 1]`.
3. Two runs with `DETERMINISTIC_SEED` produce identical predictions.
4. Legacy pickles (those without the new pickle keys) still load and
   predict — the `filedict.get(key, default)` pattern from P2.B is the
   contract.

If the inference test does not exist, the change cannot ship. This is
the lesson from OVERCONFIDENCE_INVESTIGATION §3.4 (the live-path
confound): offline A/B verdicts are meaningless if the change crashes
or silently drifts in `prophecize`.

### Pickle-schema discipline (the seam where train/predict drift hides)

The pickle dict written by [pipeline.py:1940 `_build_filedict`](../src/sportstradamus/training/pipeline.py#L1940)
is the contract between training and inference. Every new field added
during this plan **must** also have:

1. A reader site in `model_prob.py` (or wherever the field is consumed).
2. A legacy-default fallback so pre-change pickles still load:
   `filedict.get("new_key", "joint")` — the P2.B `zinb_mode` pattern.
3. A round-trip test in `tests/test_*.py` that pickles the new model,
   re-loads, asserts byte-identical predictions.

The list of fields written today (as of commit `77e4a41`): `model`,
`step`, `stats`, `metrics`, `diagnostics`, `params`, `distribution`,
`cv`, `std`, `temperature`, `dispersion_cal`, `weight`, `r_book`,
`hist_gate`, `shape_ceiling`, `normalized`, `offset_meta`,
`target_strategy`, `zinb_mode`, `is_hurdle`, `expected_columns`. Any
addition lands in this list and gets the three-step contract above.

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

**T1. ICC per league × per market.** Compute ICC₁ for every SkewNormal
market in every covered league (NBA 13 + WNBA 11 + NFL 12 = 36 cells) on
three seasons of held-out data using a two-level ANOVA decomposition:
σ²_between = Var(player season means), σ²_within = mean(Var(game logs
within player)), ICC = σ²_between / (σ²_between + σ²_within). For highly
skewed markets (NBA STL/BLK; NFL interceptions/sacks) compute on
`log(1+Y)` or rank-transformed target.

Output a 36-row table keyed by (league, market). Pre-register the routing
cutoff per cell before running A/Bs:
- ICC ≥ 0.5 → EB-style centering can work (T5 four-stage factorization).
- ICC ≤ 0.3 → needs distributional tail extension (T3/T7).
- 0.3 < ICC < 0.5 → ambiguous; try both, route based on outcome.

Expected (NBA): FGA > 0.6, PTS in 0.3–0.45, STL/BLK < 0.2. WNBA likely
similar to NBA (same sport, fewer games per player so estimator is
noisier). NFL likely *higher* than basketball for position-dependent
stats (a top QB always passes for 250+ yds, a RB1 always rushes) but
small-sample variance inflates the within-player term — net effect is
empirical, not predicted.

Implementation site: a new one-shot CLI in
`src/sportstradamus/scripts/icc_diagnostics.py` that consumes
`data/training_data/{LEAGUE}_{market}.parquet` for every cached league.
Output to `data/icc/{LEAGUE}_icc.parquet` (per-league file keeps the diff
trivial when only one league's gamelogs change). No model changes.
Source: researcher T1 (Tier 0).

**Decision points (gate logic for Stage A1):**
- If ICC_PTS > 0.5 on any league — entanglement hypothesis is wrong for
  that league; leaf-averaging hypothesis revives; T7 (extreme-tail
  boosting via gbex) jumps to Stage A2 on that league.
- If ICC is uniformly low across a family within a league — that league's
  family is form-driven and T3 (heavy-tail / normalizing-flow head) takes
  priority over T5 there.
- If ICC clusters bimodally within a league (e.g. FGA/MIN high, PTS/AST
  low) — per-market routing in Stage A2 is justified for that league.
- **NFL-specific:** if ICC is high but EB shrinkage K=10 produces noisy
  per-player MeanYr (small sample), re-derive K per the
  Casella–Berger empirical-Bayes formula
  `K = σ²_within / σ²_between` evaluated *per league*; record the per-league
  K alongside the per-league ICC table.
- **Stop-the-track check:** Stage A1 is a diagnostic — it does not by
  itself ship a model change, so there are no live-data graduations here.
  But if the ICC table reveals the family is *already* well-routed by the
  existing `ratio_meanyr` default (e.g. low-ICC markets are already kept
  out of EB centering by P1's verdict), Stage A2 should be skipped for
  those cells and the track-resumption discussion happens only if live
  metrics regress.
- **Inference-path check:** Stage A1 ships no model change — diagnostic
  only. No inference touchpoints.

### Stage A2 — Highest-leverage structural fixes (2–3 sprints)

Two methods, both worth running because they attack different mechanisms.
Pick order based on Stage A1 ICCs.

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **T5-basketball. Four-stage multiplicative factorization (NBA + WNBA)** *(replaces original P3)* | Tier 1 (Skew) | 2–3 weeks | Predict (a) P(plays), (b) MIN \| plays, (c) per-100-poss rate \| MIN, (d) for PTS: FGA-per-100 × FG% × points-per-make; recombine via Monte Carlo. NBA and WNBA share this structure exactly. Routes each factor to its own ICC-appropriate strategy. DFS-industry consensus approach. | New `src/sportstradamus/factorize/` package or extend [pipeline.py](../src/sportstradamus/training/pipeline.py); inference mirror in [stats/base.py](../src/sportstradamus/stats/base.py); per-market wiring via the existing strategy registry from P1. |
| **T5-NFL. Position-dependent factorization** | Tier 1 (Skew), adapted | 2–3 weeks (depends on T5-basketball landing first) | Football has no per-100-possession equivalent and stats are position-locked. Candidate factor trees: passing yards = P(plays) × Snaps × Attempts/snap × Yards/attempt; rushing yards = P(plays) × Snaps × Carries/snap × Yards/carry; receiving yards = P(plays) × Snaps × Targets/snap × Catch-rate × Yards/catch. The Stage A1 ICC table tells you whether each factor is high-ICC (route to EB centering) or low-ICC (route to T3 distributional tail). Position-locked features (`Player position` already a category in `X` per [pipeline.py](../src/sportstradamus/training/pipeline.py)) make per-position model variants cleanly separable if needed. | Same `src/sportstradamus/factorize/` package; NFL-specific factor definitions in a config file under `data/factorize_nfl.json`. **Defer until T5-basketball ships** — same architectural code, different factor lists. |
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
- **Stop-the-track check:** after T5 or T3 ships and runs in production
  for ≥ 14 days, compute the live-data graduation criteria above per
  cell. Cells that graduate skip Stage A3 and A4 entirely. Common
  expected outcome: high-ICC markets graduate after T5; low-ICC
  heavy-tail markets graduate after T3; the remaining gap is what
  Stage A3 polish targets.
- **Inference-path check:** Stage A2 is the largest inference-side change
  in the plan. T5 needs `prophecize` to load N pickles per market and
  Monte Carlo recompose; `model_prob` and `fused_loc` may need multi-head
  variants. T3 introduces a new distribution name (spliced or
  normalizing-flow) requiring decode + blend in `model_prob.py` and
  `helpers/distributions.py`. Inference-path test must exist before Gate 1
  is evaluated. Use the P2.B HurdleZINB pattern as the template.

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
- **Stop-the-track check:** Stage A3 is intentionally cheap calibration
  polish. If P7+T8+P6 combined push live-data graduation across the
  remaining cells (high probability — these target the residual loc/scale
  miscalibration after A2), do NOT proceed to Stage A4. Stage A4 is
  reserved for cells that fail to graduate despite both structural fix
  (A2) and calibration polish (A3); if A3 fixes them, A4 is
  perfect-as-enemy-of-good.
- **Inference-path check:** P7 (isotonic on loc) and T8 (CQR) are
  post-hoc calibration objects — new pickle keys (`isotonic`, `cqr`)
  loaded by `model_prob` and applied after the distribution decode but
  before `fused_loc`. Round-trip test required. P6 (reduce regularization)
  is training-only, no inference change. T9 (monotone constraint) is
  training-only, no inference change.

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
- **Stop-the-track check:** Stage A4 is the last resort. If any cell is
  still failing graduation here, *first* check whether it has crossed into
  "live data shows we're already profitable on this market with the
  current model" — the bar for justifying a novel risky retry (T4, T2,
  T7, T6, T10) is much higher than for the cheap fixes earlier in the
  track. A cell that has settled brier_skill ≥ 0 in production but
  doesn't pass the strict 5% top-decile MAE bar on offline A/B is
  almost certainly NOT worth Stage A4 effort — file as "deprioritized:
  acceptable live performance, offline gap is academic."
- **Inference-path check:** T4 MEGB (different model class, R-native —
  needs Python wrapper and `is_megb` dispatch flag), T2 CatBoost
  (different model class with `cat_features` + ordered TS — needs
  `is_catboost` dispatch and a CatBoostLSS adapter or external objective
  wrapper for SkewNormal NLL), T10 PGBM (different model class) all
  require the most invasive inference work in Stage A4. T7 gbex (tail
  model layered on body — multi-pickle Monte Carlo at inference). T6
  FAGTB (training-only objective change, no inference touchpoint).
  Inference cost is a tie-breaker: prefer T7 over T2/T4/T10 when
  inference-engineering capacity is the constraint.

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
| **ZTNB Stage 2 likelihood fix** | Tier 1 (ZINB) #1 | hours | Replace `NegativeBinomial(μ, α).log_prob(y)` in the Stage-2 hurdle loss with `nb.log_prob(y) − log1p(−exp(nb.log_prob(zeros_like(y))))`. The optimizer recovers an unbiased μ; the derived-π identity then gives the correct ZINB marginal mean (1−ψ)·μ. Eliminates the +NB(0)·(1−q)·μ_pos/(1−NB(0)) over-prediction on FTM/STL **by construction**. League-agnostic — applies to every ZINB market in every covered league. | Wrap the existing NegBin loss in [hurdle.py](../src/sportstradamus/hurdle.py); PyTorch already has `torch.distributions.NegativeBinomial`. |
| **Per-league × per-market routing diagnostics** | Tier 1 (ZINB) #2 | days | Build a per-league × per-market dashboard with: observed mean, variance, zero-rate p₀; ziP and ziNB indices with bootstrap CIs (Blasco-Moreno et al. 2019); Wilson-Einbeck p-value (Stat Modelling 2019); Schwarz-corrected Vuong (HurdleNB vs ZINB) per Wilson 2015; var/mean ratio; conditional positive mean E[Y\|Y>0] vs μ. Routes each (league, market) cell to plain NB / HurdleNB(ZTNB) / MZINB / CMP. 23 cells total (NBA 8 + WNBA 7 + NFL 8). | New `src/sportstradamus/scripts/zinb_routing_diagnostics.py`; output to `data/zinb_routing/{LEAGUE}_diagnostics.parquet` (one file per league keeps diffs minimal when only one league updates). |

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
- **Stop-the-track check:** the ZTNB fix is the cheapest single change
  in the entire plan and the researcher's strongest claim ("likely resolves
  FTM/STL by construction"). After it ships and runs ≥ 14 days, expect
  most ZINB cells to graduate. The bar for proceeding past Stage B1 is
  *unambiguous* live-data failure on specific cells — not "we could
  probably squeeze out more on FG3M with MZINB."
- **Inference-path check:** ZTNB likelihood fix is **training-only** —
  `HurdleZINB.predict()` returns the same `(total_count, probs, gate)`
  triple, the derived-π formula is unchanged, downstream
  `helpers/distributions.py` consumers are untouched. The existing
  `tests/integration/test_zinb_hurdle_live_path.py` is sufficient
  inference coverage. Per-market routing diagnostics (`zinb_routing_diagnostics.py`)
  are read-only — no inference touchpoint.

### Stage B1 — outcome & Track-B rescope

**B1.1 — ZTNB likelihood fix: hypothesis REFUTED.** The zero-truncated NB
(`_ZeroTruncatedNB` in `hurdle.py`, verified against scipy in
`tests/test_ztnb_loss.py`) is correct in isolation but is **incompatible with the
frozen derived-π hurdle decode**. Diagnostic on FG3M (a P2.B *SHIP*), 4000-row
deterministic fit:

| quantity | value |
|---|---|
| observed zero rate | 0.326 |
| classifier `q` (pred P(Y=0)) | 0.311 |
| ZTNB count-component `NB(0)` | **0.412** |
| frac rows `NB(0) > q` → π clips to 0 | **0.652** |
| ZTNB `μ_NB` (= base_ev) | 1.249 |
| `E[Y\|Y>0]` under ZTNB = 1.249/(1−0.412) | 2.12 ≈ empirical 2.22 |

The derived-π identity `q = π + (1−π)·NB(0)` requires `q ≥ NB(0)` per row. ZTNB
correctly recovers a count component with a *higher* zero mass than the old
full-support-NB-on-positives fit, so `NB(0)` exceeds `q` on most rows, π clips to
0, and the reconstruction overshoots q (`test_zinb_hurdle_live_path` identity diff
**0.136** ≫ 0.02 tol). Crucially `E[Y|Y>0]` is essentially unchanged (old ≈ 2.2,
ZTNB ≈ 2.12) — ZTNB only re-decomposes the positive mean into (lower count mean,
higher count-zero mass), which is precisely what breaks the decode. The fix would
**regress the 6 markets P2.B shipped**, not just fail FTM/STL. Decision: revert the
one-line wire-in; keep `_ZeroTruncatedNB` as an unwired, test-covered building
block for the MZINB head (B3). The smoke A/B was not run (no stats.nba.com network
in the build env); the analytical verdict is KILL.

**B1.2 — routing diagnostics: the ZINB label is wrong for most cells.** The
marginal diagnostics (`data/zinb_routing/{LEAGUE}_diagnostics.parquet`, 23 cells)
split into two physical clusters:

- **Underdispersed / near-Poisson → CMP (13 cells):** `var/mean ≤ 1.3`, ziNB ≈ 0
  (three *negative* = zero-deflation). NBA STL, BLST, TOV, OREB, PF; WNBA BLK, STL,
  BLST, TOV, OREB; NFL tds, rushing tds, receiving tds. NB/ZINB **cannot** fit
  var < mean — it forces overdispersion these markets do not have; their high zero
  rates are low-mean *sampling* zeros, not structural inflation.
- **Overdispersed + mild inflation → MZINB (10 cells):** NBA FG3M, FTM, BLK; WNBA
  FG3M, FTM; NFL passing tds, qb tds, interceptions, sacks taken, passing first
  downs. Genuine overdispersion (var/mean 1.35–7.9), small-but-positive ziNB,
  Vuong favors ZINB over the hurdle.

**0/23 cells route to `hurdle_nb_ztnb`.** The two P2.B kills have *different* root
causes: **STL** is underdispersed (var/mean 0.99–1.17) → mis-labeled ZINB → wants
**CMP** (B4, as the parent plan's STL note anticipated); **FTM** is genuinely
overdispersed+inflated (highest ziNB ≈ 0.063) → wants **MZINB** (B3). Neither wants
the ZTNB hurdle.

**Rescope recommendation (supersedes the original B2/B3/B4 ordering):**

1. **Revisit `stat_dist.json` ZINB labeling.** ≥ 13 of the 23 "ZINB" cells are
   underdispersed/near-Poisson and are being over-dispersed by the NB family — a
   plausible source of the compression on those markets independent of the gate.
2. **CMP track (was B4) is now first-class, not optional** — it owns the 13
   underdispersed cells. Conway-Maxwell-Poisson (or plain Poisson) handles
   var ≤ mean, which NB/ZINB structurally cannot.
3. **MZINB (B3)** owns the 10 genuinely-inflated cells and is the consumer of the
   `_ZeroTruncatedNB` building block (a marginalized ZINB estimates the count
   component and the gate jointly, avoiding the derived-π `q ≥ NB(0)` constraint
   that broke B1.1).
4. **B2 routing wiring** is still the mechanism, but the routing config should be
   seeded directly from `zinb_routing_diagnostics.py` output (cmp / mzinb), not
   from a hurdle-vs-joint flag.

Research-agent handoff: `/tmp/track_b_rescope_research_prompt.md`; the returned
statistician's brief is archived at `/tmp/researcher_track_b_rescope_response.md`
and its load-bearing conclusions are folded into the next subsection. Regenerate
the diagnostics with `poetry run zinb-routing-diagnostics` if the parquet is stale.

### Stage B1 — research verdict (distribution-family routing)

A claude.ai statistician reviewed the B1.2 rescope (brief at
`/tmp/researcher_track_b_rescope_response.md`; conclusions copied here so the plan
stays self-contained). It **confirms the two-cluster rescope** and sharpens it
into an implementable routing protocol, with two reality checks that change the
economics of the stages below.

**Headline.** Route the 13 underdispersed cells to **mean-parameterized
Conway-Maxwell-Poisson (CMPμ)** and the 10 inflated cells to **marginalized ZINB
(MZINB)**; keep the derived-π hurdle as the *default* for borderline cells. The
two new families are **add-ons, not a wholesale replacement** of the shipped
hurdle.

**1. Route on *conditional* dispersion, not the marginal var/mean alone.** The
marginal var/mean is a necessary screen but not a sufficient router. It misleads
once features explain the mean: (a) *mean-mixing* inflates it — pooling a star
(3.5 threes/g) with a benchwarmer (0.4) looks overdispersed even if each player is
Poisson(μ_i), and a good feature set absorbs the role mixture so residual
conditional dispersion → 1; (b) floor/ceiling effects (PF capped at 6) create
*conditional* underdispersion features cannot remove. Safe protocol:
1. Marginal (var/mean, ZI-index, score-test p) per cell with bootstrap CIs — what
   `zinb_routing_diagnostics.py` already produces.
2. Fit a baseline Poisson GBM; compute *conditional* dispersion from
   randomized-quantile residuals (Dunn & Smyth).
3. Route only when marginal and conditional agree, conditional overriding:
   **CMPμ** iff conditional var/mean < 0.90 AND marginal < 1.0; **MZINB** iff
   conditional var/mean > 1.20 AND ZI-index CI excludes 0.
4. Disagreement zone → keep the derived-π hurdle (least to lose).
5. Tiny-sample cells (rare markets, late-season WNBA) → single-stage Poisson +
   sandwich SE beats data-driven family selection.
So the B2 routing config must be seeded from a *two-stage* diagnostic (marginal +
a Poisson-GBM residual pass), not from the marginal parquet alone.

**2. Tightened dispersion bands.** B1.2's `var/mean < 1.3 → cmp` lumps
equi-dispersed and mildly-overdispersed regimes. Literature-matched bands: ≤ 0.85
strong underdispersion (CMPμ) · 0.85–1.15 equi-dispersed (Poisson) · 1.15–1.5 mild
overdispersion (NB) · ≥ 1.5 NB/hurdle/ZINB. Adopt **1.15**, not 1.3, as the NB
lower edge.

**3. Do NOT use the Vuong test** to pick ZINB-vs-NB — they are nested at γ=0 so
the Vuong assumptions fail (Wilson 2015, Economics Letters). Read the
`zinb_routing_diagnostics.py` "Schwarz-corrected Vuong" column as descriptive
only; use a boundary likelihood-ratio / score test for the inflation decision.

**4. CMPμ is an engineering project, not a research project — but the ceiling is
modest.** Use Huang's (2017, Statistical Modelling) *mean*-parameterized CMP
(log-link μ, orthogonal dispersion ν), NOT canonical (λ,ν) where λ ≠ E[Y] — boost
the quantity you price against. Neither LightGBMLSS nor XGBoostLSS ships CMP
(verified to XGBoostLSS 0.6.1 / current LightGBMLSS), so it is a custom
distribution class; the frameworks accept any distribution with first/second
derivatives, and CMPμ's score/Hessian are tractable once Z(λ,ν) is tabulated
(truncate the series at K≈64 in our μ≈0.3–3 regime; well-conditioned info matrix
from Huang & Rathouz 2017 mean/dispersion orthogonality). Proven pattern: a
precomputed (μ,ν)→λ look-up grid with bilinear interpolation, refreshed once per
market (Philipson & Huang 2023, Stat. Comput.; CMPBoost / Chatla & Shmueli 2020
JCGS is the boosting reference, code `SuneelChatla/CMPTree`). Stabilize the ν
gradient (clip log(ν)∈[−1, 2]). **Reality check:** projected top-decile-MAE gain
is **~3–8% relative, not 30%** — for conditional var/mean ≈ 0.75 fit as NB ≈ 1.4,
σ is overstated ~1.37×, moving a half-line tail probability only a few points.
Worth building to amortize across 13 cells; **not** worth it for one. When
conditional var/mean ∈ [0.90, 1.10], plain **Poisson** is the right call — CMPμ
collapses to it and forcing NB on equi-dispersed data destabilizes the MLE (Yang
et al. 2026, arXiv:2404.07457).

**5. For the inflated cluster, prefer the *marginalized hurdle* (Kassahun et al.
2014) over MZINB — and that is where `_ZeroTruncatedNB` belongs.** Two joint-fit
families remove the derived-π `q ≥ NB(0)` clip:
- **Marginalized hurdle (Kassahun et al. 2014, Stat. Med. doi:10.1002/sim.6237) —
  recommended, smaller lift.** It is the literal joint-fit version of the
  two-stage hurdle we already ship: a logistic for P(Y>0) and a **zero-truncated**
  NB on the positives, with the marginal mean reparameterized as the regression
  target. No `q ≥ NB(0)` constraint — the count component is genuinely defined on
  positives and the zero mass is its own free parameter, so the ZTNB that broke
  B1.1 inside the *derived-π decode* has a *natural home* here (this is what
  `_ZeroTruncatedNB` was kept for). Preserves the structural cleanliness of
  derived-π while making it one joint fit. (Molenberghs et al. 2018,
  doi:10.1002/sim.7596, for the design choices.)
- **MZINB (Preisser et al. 2016) — the alternative, only for true structural-zero
  excess.** Parameterizes log(ν_i)=X_i′β (β = marginal mean = the line),
  back-computes μ_i = ν_i/(1−ψ_i), separate logistic gate; the count component is
  a *full* NB at all y (**NOT** zero-truncated — ZTNB does not belong in MZINB).
  Pick it only when a meaningful subpopulation is plausibly at *structural* zero
  beyond the NB's own zeros (e.g. a player who simply never shoots threes).

**Either way, joint fitting *relocates* the gate-vs-count identifiability problem,
it does not solve it** — for fixed ν_i a larger ψ_i is offset by a larger μ_i (α
absorbs the residual). Three mitigations that work: (a) **separate covariate
sets** — drive ψ off zero-risk-only variables (minutes, availability,
blowout/garbage-time flags) and ν off count variables (ability, opponent defense,
pace); identifiability is much better when the inflation predictors are not a
subset of the mean predictors; (b) **warm-start from the derived-π fit** — init
the gate from the classifier log-odds at ψ̂_i = clip(q_i), β from log(ȳ_i), α from
the NB MLE on positives; (c) **constrain α weakly** (log-normal penalty around the
method-of-moments estimate, sd≈0.5) so the gate is not swamped by NB
over-dispersion. Preisser's ~100% convergence is from their own simulations; the
2017 follow-up calls MZIP/MZINB "prone to convergence problems to a degree shared
by ZIP and ZINB" — budget real validation, keep the derived-π hurdle as fallback.

**6. The routing will drift across seasons.** NBA 3PA/game rose ~1000% (2.8 →
32.0) from 1979 to 2018-19 with 3P% up 28%→36% (Zając et al. 2023) — distribution
shape, zero mass, and dispersion regime all move. Treat single-season routing as
stale: re-run the diagnostic each offseason, route on a **hysteresis band** (flip
a cell to CMPμ/MZINB only if it sits outside [0.85, 1.30] var/mean in the last
*two* seasons), default to the hurdle inside the band, and if a cell's routing
flips year-over-year force the more robust NB-based hurdle.

**7. Cheap pre-checks before building anything.** (a) **Confirm the compression is
a likelihood problem at all:** refit the suspect cells with a plain **Poisson GBM**
(no over-dispersion). If top-decile compression *persists*, the likelihood is part
of the story; if it *vanishes*, the cause was the **feature set** (missing
opponent-defense interaction, no garbage-time/blowout flag), not the distribution
family — fix features first, far cheaper than CMPμ/marginalized-hurdle. (b) **Audit
for Vuong misuse:** any routing decision that used a Vuong p-value to choose
ZINB-vs-NB is invalid (nested at γ=0) — redo with Wilson-Einbeck or a boundary LR
test. (c) **ZICMP is research territory:** `mpcmp` does not yet ship a
zero-inflated CMP, so a cell needing *both* inflation and under-dispersion has no
off-the-shelf family — flag it rather than forcing one.

**Recommended sequencing (supersedes the B2/B3/B4 ordering below; effort is the
researcher's estimate):**
1. **Diagnostics infrastructure (1–2 weeks)** — extend `zinb_routing_diagnostics.py`
   into a per-market panel: marginal mean, var/mean + bootstrap CI, ZI-index + CI,
   Wilson-Einbeck score test, **conditional** var/mean from randomized-quantile
   residuals of a baseline Poisson GBM, and a 4-season stability flag (fraction of
   seasonal var/mean in the same routing zone over the last 4 seasons). Lock the
   protocol *before* refitting any family (Campbell 2021 selection-bias trap). Run
   the Poisson-GBM compression pre-check (§7a) here.
2. **Marginalized hurdle for the 10 inflated cells (3–4 weeks)** — smallest delta
   from what we ship; consumes `_ZeroTruncatedNB`; warm-start + separate covariate
   sets per §5. **Gate: promote iff top-decile MAE improves ≥ 3% on the inflated
   cluster on a held-out season; else revert.**
3. **CMPμ for the 13 underdispersed cells (6–8 weeks)** — promoted from B4
   "STL-only/optional" to first-class. **Gate: ship CMPμ only where conditional
   var/mean < 0.90 AND held-out top-decile MAE improves ≥ 2%; in the 0.90–1.15 band
   ship plain Poisson (strictly safer at ν≈1).** Ceiling ~3–8% — worth it amortized
   across 13 cells, not for one.
4. **Routing governance (ongoing)** — offseason refresh on a rolling 3-season
   window; hysteresis band (var/mean < 0.85 last 2 seasons → CMPμ; > 1.30 with ZI
   CI excluding 0 → marg-hurdle/MZINB; else hurdle); stability < 0.75 → derived-π
   hurdle; mid-season regime-flip detector → hurdle fallback. The B2 config gains
   `poisson`/`cmp`/`marg_hurdle`/`mzinb` values, seeded by the two-stage diagnostic.

**What would change this:** if LightGBMLSS/XGBoostLSS ship a native CMPμ head
(watch the StatMixedML repos) CMPμ drops from 6–8 weeks to ~1; if held-out
top-decile MAE on the underdispersed cells is already within 2% of a Poisson
oracle, skip CMPμ; if > 3–4 cells show conditional var/mean < 0.70 (deep
underdispersion) CMPμ rises in priority — only CMPμ or generalized/double-Poisson
(Efron 1986) can fit those. The GPBoost fork (B3 below) stays an alternative *only*
if post-step-1 residuals show systematic per-player offsets rather than dispersion
misfit.

### Stage B2 — Routing + orthogonal feature engineering (2 weeks, in parallel)

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **Per-league × per-market routing wiring** | Tier 1 (ZINB) #2 | days | Implement the routing logic from B1's table. `data/zinb_mode_per_market.json` config schema: `{LEAGUE: {market: "joint"|"hurdle"|"plain_nb"|"poisson"|"cmp"|"marg_hurdle"|"mzinb"}}` (keyed by both league and market — NBA STL and WNBA STL may route differently). **Seeded by the two-stage routing diagnostic (marginal `zinb_routing_diagnostics.py` + a Poisson-GBM conditional-dispersion residual pass) under the Stage B1 research verdict's hysteresis band — not the marginal parquet alone.** Per-cell lookup in [training/pipeline.py:1869](../src/sportstradamus/training/pipeline.py#L1869) `_step_select_distribution` consults the config; default `"joint"` for any unrouted cell keeps legacy production byte-identical. | `pipeline.py` per-cell dispatch; new JSON config under `data/`. |
| **Leakage-safe target-encoded player features** | Tier 2 (ZINB) #5; was original P5 | days | `groupby(player_id).expanding().mean().shift(1)` for stat and stat×opponent. Orthogonal to architectural choice; ships regardless of which Stage B3 winner emerges. League-agnostic — same expanding-mean recipe works for NBA/WNBA/NFL. | New columns in [stats/base.py:597](../src/sportstradamus/stats/base.py#L597) `get_stats`; same leakage audit as the MeanYr/Mean10 columns. |

**Decision points (gate logic for Stage B2):**
- 8/8 ship under routing + encoded features → hold off on Stage B3 unless
  ROI is needed elsewhere. The system is in a good state.
- <8/8 ship → proceed to Stage B3 on the markets that still kill.
- **Stop-the-track check:** after Stage B2's routing config lands and
  runs ≥ 14 days, expect the remaining ZINB cells (across NBA/WNBA/NFL)
  to graduate. Stage B3 is a 4–6 week investment in a novel architecture
  (MZINB) or a non-trivial dependency add (GPBoost); do not start it
  unless ≥ 3 cells have *not* graduated AND their offline residual
  diagnostics point clearly at one of MZINB or GPBoost. "We can probably
  do better than B2" without that evidence is the perfect-as-enemy-of-good
  trap.
- **Inference-path check:** Per-(league, market) routing config is the
  cleanest inference change in Track B — `model_prob` already dispatches
  on `getattr(model, "is_hurdle", False)` (the P2.B pattern), so the
  config just changes which pickle gets loaded per cell; per-cell pickle
  already records `zinb_mode`. Leakage-safe target-encoded features
  require the same `stats/base.py:get_stats` mirror as MeanYr/Mean10 —
  feature column must be computed identically at inference with no peek
  at the current game; leakage test pattern from
  [test_meanyr_mean10_leakage.py](../tests/test_meanyr_mean10_leakage.py).

### Stage B3 — Strategic fork (4–6 weeks, pick ONE — not both)

This is the canonical decision point of the ZINB track. The researcher
explicitly flags that running MZINB and GPBoost in parallel doubles cost
without obvious gain. The decision criterion is **residual structure after
Stage B2**.

| Option | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **MZINB head in LightGBMLSS** | Tier 2 (ZINB) #4 | 2–4 weeks | Reparameterize so the marginal mean ν = E[Y] is boosted directly; the latent NB conditional mean μ = ν/(1−ψ) is reconstructed. Three heads: logit(ψ), log(ν), log(α). The boosted ν IS the quantity the downstream pipeline already consumes. Cleaner than the current derived-π trick. Removes the `q ≥ NB(0)` clip **by construction**, but *relocates* (does not eliminate) the gate-vs-count trade-off — see the **Stage B1 research verdict §5**, which recommends the **marginalized hurdle (Kassahun 2014)** as the smaller-lift joint fit for the inflated cluster (the natural home for `_ZeroTruncatedNB`), with MZINB reserved for true structural-zero excess. **No published GBDT implementation — novel contribution.** | New `MZINB` class alongside the existing distributions in [skew_normal.py](../src/sportstradamus/skew_normal.py) or under [helpers/distributions.py](../src/sportstradamus/helpers/distributions.py); foundational papers Long 2014 + Preisser 2016. Use Mutiso et al. 2024 (Pólya-gamma augmentation) as a likelihood-structure reference. |
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
- **Stop-the-track check:** Stage B3 is the canonical "is it worth it"
  decision. The researcher explicitly flags MZINB as "no GBDT precedent
  — novel contribution." That implementation risk is only worth eating
  if (a) live data shows specific cells still failing graduation after
  B1+B2 AND (b) their residual structure clearly favors MZINB over the
  cheaper GPBoost alternative. If live performance is acceptable on all
  cells but offline metrics still show top-decile gap, the right
  decision is "file MZINB as future research" and move to Stage B4 or
  a different project.
- **Inference-path check:** Both options change the parameter contract.
  MZINB heads return `(ν, ψ, α)` instead of `(total_count, probs, gate)`
  — `model_prob.py` needs a new ZINB-MZINB decode block that derives
  `Model EV = ν` directly (the marginal mean IS the boosted quantity)
  and reconstructs the `Model Gate` from `ψ` for downstream `fused_loc`
  compatibility. The `dist` string can stay `"ZINB"` (mode field
  distinguishes), but a new `mzinb_mode` pickle key with a legacy
  default of `"hurdle"` keeps existing pickles loadable. GPBoost is a
  different model class entirely — `is_gpboost` dispatch flag in
  `model_prob`, model-load path branches in `prediction/__init__.py`,
  and a GPBoost-specific live-path test. **Inference-engineering cost
  alone is enough reason to prefer one fork over the other when
  residual diagnostics are ambiguous** — GPBoost's class change is
  larger than MZINB's parameter-contract change.

### Stage B4 — Tuning, polish, specialized fixes (optional)

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **Per-parameter Optuna search** | Tier 3 (ZINB) #6 | days | Separate `learning_rate` and `n_estimators` for gate vs NB heads inside the existing LightGBMLSS Optuna sweep. cyc-GBM-inspired without porting. May resolve the deterministic-30-round blowups without architectural change. Daub et al. 2026 ("balanced step length") provides theoretical justification. | Extend the Optuna search-space dict in [pipeline.py:348-368](../src/sportstradamus/training/pipeline.py#L348) to distinguish gate vs count hyperparameters. |
| **CMPμ head — PROMOTED to first-class (owns the 13 underdispersed cells, not STL-only)** | was Tier 4 → Tier 1 (ZINB) #8 | 6–8 weeks | Per the Stage B1 research verdict, no longer optional STL polish — owns the 13 underdispersed cells. Mean-parameterized CMP (Huang 2017; canonical λ ≠ E[Y], unusable for line-pricing). Custom distribution class — neither LightGBMLSS nor XGBoostLSS ships CMP. Series truncation K≈64; precomputed (μ,ν)→λ look-up grid (Philipson & Huang 2023); init ν from method-of-moments; clip log(ν)∈[−1,2]. Ceiling ~3–8% top-decile MAE amortized across 13 cells. Plain **Poisson** when conditional var/mean ∈ [0.90, 1.10]. | New PyTorch custom distribution; pre-computed Z(λ, ν) + (μ,ν)→λ lookup table at module load. Reference: CMPBoost (Chatla & Shmueli 2020), code `SuneelChatla/CMPTree`. |
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
- **Stop-the-track check:** Stage B4 items are individually cheap (hours
  to days), so the perfect-as-enemy-of-good risk is lower than at Stage
  B3. But the *cumulative* time across all 6 items is multiple weeks. If
  per-parameter Optuna alone graduates the holdout cells, do not run the
  other 5 — file as deprioritized with the graduating metrics noted.
- **Inference-path check:** Per-parameter Optuna and reduced
  regularization are training-only — no inference change. Sample
  reweighting is training-only — no inference change. CMP head is a new
  distribution: `model_prob` decode + `helpers/distributions.py`
  consumers all need to accept `dist == "CMP"`; live-path test required.
  Isotonic on ZINB-mean is a post-hoc calibration object — new
  `isotonic_zinb` pickle key + load + apply in `model_prob` after the
  ZINB decode block; round-trip test. Quantile heads pickle alongside
  the ZINB model with a new key (`quantile_heads`) and a new live-path
  decode path. MERF-style iteration changes the model architecture
  (alternating fit-residual / re-estimate-shrunken-baseline) — pickle
  schema changes, `is_merf` dispatch flag, new live-path test.

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

## Cross-league caveats (read before running any cross-league A/B)

1. **NFL sample sizes are an order of magnitude smaller than NBA.** ~17 games
   per player per regular season vs ~82 for NBA. EB(MeanYr, K=10) is
   aggressive shrinkage with that sample size — Stage A1 should re-derive
   `K = σ²_within / σ²_between` per league. Expect NFL K to be much lower
   (or for the EB target transform to fail outright on form-volatile NFL
   markets — file as a Stage A1 finding, not a P1-style negative result).
2. **NFL stats are position-locked in a way basketball isn't.** `Player
   position` is already a categorical feature
   ([pipeline.py](../src/sportstradamus/training/pipeline.py)
   `_step_build_splits`). Researcher Track-A methods that train one
   cross-player model per market may not transfer cleanly to NFL — a QB
   and a WR don't share the same "passing yards" distribution. The Stage
   A1 ICC table should be computed *within position* for NFL where
   relevant (e.g. passing yards ICC computed across QBs only).
3. **WNBA shares NBA's structure but has half the games/season.** EB K=10
   is probably fine but should still be verified in Stage A1. The
   per-100-possession factorization (T5-basketball) transfers
   exactly — WNBA uses the same court geometry and stat universe.
4. **The compression_eval A/B harness is league-agnostic but file paths
   are league-specific.** Cached parquets live at
   `data/training_data/{LEAGUE}_{market}.parquet`; deterministic test
   sets at `data/test_sets/deterministic/{strategy}/{LEAGUE}_{market}.csv`.
   The full-verification phase (cross-league A/B) iterates the existing
   league loop — no harness rewrite needed.
5. **Determinism gate currently covers NBA only.** The two
   `test_deterministic_mode_*` integration tests use NBA_FGA + NBA_FG3M
   parquets ([tests/integration/test_determinism_gate.py:37,102](../tests/integration/test_determinism_gate.py)).
   Before shipping a cross-league change, add parallel determinism
   assertions on WNBA_FGA + WNBA_FG3M + a representative NFL market.
   Without them the cross-league A/B verdict is noise for the new leagues
   (P1's hard-learned lesson — see `CENTERED_TARGET_NEGATIVE_RESULT.md`).
6. **For low-mean NFL markets** (interceptions mean ~0.5, sacks ~1.5),
   the ZINB diagnostic formulae in Stage B1 may need to compute on
   `log(1+Y)` per the Blasco-Moreno et al. STL/BLK caveat — the
   asymptotic Vuong test in particular degrades badly at very low means.
   Wilson-Einbeck's non-asymptotic Poisson-binomial test handles this
   better but should be the only test trusted for NFL interceptions/sacks
   in Stage B1.
7. **Two-track parallelism still holds across leagues.** Track A and
   Track B touch different distribution branches and different markets;
   they can be worked in parallel per league. They share no mutable
   state — the only shared resource is the compression_eval harness,
   which is read-only during scoring.

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

**Always-on quality gates** (run on every commit, every session, every PR):
- `poetry run ruff check src/sportstradamus/`
- `poetry run pytest tests/golden/` (incl. `test_compression_eval.py`)
- `poetry run pytest -m integration` (fake-mode, no network)
- Regenerate CLI help snapshots if `meditate` flags change:
  `REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py`
- Determinism gate (P0.5): `poetry run pytest tests/integration/test_determinism_gate.py -v -m integration`

**Smoke phase** (at the start of a new experiment, before full A/B):
- Pick 1–2 representative markets per league (see "Cross-league testing
  policy" above for the per-league smoke list).
- Run `meditate --deterministic --league {NBA,WNBA,NFL} --market <smoke-market>`
  per league × per smoke-market. Confirm the strategy doesn't immediately
  blow up determinism, produces sensible `compression_eval` output, and
  doesn't regress the smoke markets vs baseline.
- A smoke regression is a hard stop — fix or revert before further work.

**Full-verification phase** (before any default-flag flip or `--zinb-mode`
config change):
- Run the A/B on every market in every covered league for the affected
  distribution branch (SkewNormal: 36 cells; ZINB: 23 cells).
- Strategy SHIPs only if it clears the universal decision threshold on
  every cell, OR the per-cell routing config records the exceptions.
- **Inference-path test** (required for any change that touches the
  prediction-side schema — every change in the "Per-change-type
  inference checklist" above except the training-only rows): live-path
  integration test under `tests/integration/` that loads the new
  pickle, runs `model_prob` on a cached 100-row fixture, asserts:
  (a) `Model EV` finite for every offer; (b) `Model Gate ∈ [0, 1]` if
  ZI-class; (c) two runs with `DETERMINISTIC_SEED` produce identical
  predictions; (d) legacy pickles (without the new pickle keys) still
  load and predict. The test exists **before promotion to test
  production**, not after — Gate 1's "soak window" assumes the
  inference path is structurally sound.
- **Pickle round-trip test** (required for any new `_build_filedict`
  key): save the new pickle, reload, assert byte-identical predictions
  on a cached fixture. Mirrors the test pattern from
  [tests/test_hurdle_zinb.py](../tests/test_hurdle_zinb.py) test 3
  ("Pickle round-trip preserves predictions exactly").
- Live-path gate (catches behaviors only visible end-to-end): the
  promoted strategy is confirmed through
  [prediction/model_prob.py](../src/sportstradamus/prediction/model_prob.py)
  end-to-end (no `Model Skew`=NaN, EV not collapsed by the book blend),
  not only on the dumped test set. Run live-path verification on one
  representative market per league per affected distribution branch.

**Cross-league determinism additions** (needed before the full-verification
phase is meaningful on WNBA + NFL):
- Add `test_deterministic_mode_is_bit_reproducible_wnba` (analog of the
  existing NBA test, on `WNBA_FGA.parquet`) to
  [tests/integration/test_determinism_gate.py](../tests/integration/test_determinism_gate.py).
- Add `test_deterministic_mode_is_bit_reproducible_nfl` on a representative
  NFL SkewNormal market (`NFL_passing-yards.parquet`).
- Add `test_deterministic_mode_hurdle_is_bit_reproducible_wnba` and
  `_nfl` on WNBA FG3M + an NFL ZINB market (e.g. `NFL_interceptions.parquet`).
- Without these, the cross-league A/B verdict is noise on the new
  leagues — P1's hard-learned lesson, see `CENTERED_TARGET_NEGATIVE_RESULT.md`.

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

### Phase-to-phase handoff prompts

Each completed phase produces the handoff prompt for the next phase as
part of its Definition-of-Done. The repo ships a
[prompt-engineer subagent](../.claude/agents/prompt-engineer.md) for
this purpose — invoke it via the `Agent` tool with
`subagent_type: "prompt-engineer"` at the end of each phase. The agent's
project-specific addendum documents the required reading list, the
standard 10-section structure (opener / reading list / scope / locked
decisions / inference-path checklist / decision threshold / verification
gates / branch state / out-of-scope / definition-of-done), and the
`/tmp/{stage}_handoff_prompt.md` output convention.

Handoff prompts are scratch artifacts until the next session accepts
them; on acceptance they move to `docs/handoffs/{stage}.md` for durable
git history. The handoff for **Stage 0** lives at
`/tmp/stage0_handoff_prompt.md` after this plan revision (initial
production by the prompt-engineer agent in commit
`{stage0-handoff-commit}`).

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
