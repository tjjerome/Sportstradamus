# Context: GBDT Mean-Regression Plan — Decision History & Research Detail

> **Companion context doc** to [`gbdt_mean_regression_plan.md`](gbdt_mean_regression_plan.md).
> Decision history, completed-phase results, research verdicts, and references.
> Implementation agents can usually skip this; research agents should read it.
> Home-of-record execution plan: [`gbdt_mean_regression_plan.md`](gbdt_mean_regression_plan.md).

## Table of contents

- [Status / progress log (detailed)](#status--progress-log-detailed)
- [Prior-phase results](#prior-phase-results)
- [Context](#context)
- [Architectural principle](#architectural-principle-applies-to-all-phases)
- [Research handoffs that fed this plan](#research-handoffs-that-fed-this-plan)
- [Stage 0 — Live-data instrumentation](#stage-0--live-data-instrumentation-prerequisite-for-everything-else)
- [Two-track depth plan (secondary to breadth)](#two-track-depth-plan-secondary-to-breadth)
- [Track A — SkewNormal markets](#track-a--skewnormal-markets)
- [Track B — ZINB markets](#track-b--zinb-markets-ftm-stl-kill-recovery--6-ship-hardening)
- [Open questions](#open-questions-researcher-flagged-unresolved)
- [Decisions & trade-offs](#decisions--trade-offs)
- [References](#references)

## Status / progress log (detailed)

Every result note keeps its numbers and verdict; prose tightened only.

| Phase | State | Notes |
|---|---|---|
| **P0 — offline eval harness** | ✅ done (PR #46) | `scripts/compression_eval.py` + `tests/golden/test_compression_eval.py`. Ruff clean, 6 unit tests pass, CLI single+diff smoke-tested on synthetic data. Full `poetry` gates NOT run in build env — network policy blocks the PyTorch CPU wheel source (`poetry install` fails on `torch`); needs a normal-network run before merge. |
| **P0.5 — determinism gate** | ✅ done (PR #46) | Opt-in `meditate --deterministic` (debug-only, never publish) + `tests/integration/test_determinism_gate.py`. Pure helpers `seed_everything`/`fit_lss_model`/`predict_lss_params`/`fit_predict_params` in `pipeline.py`; under `--deterministic`: RNGs pinned (random/numpy/torch + `torch.use_deterministic_algorithms`), Optuna swapped for `DETERMINISTIC_FIXED_PARAMS`, input frozen to cached parquet. Persistent writes **redirected to `data/{test_sets,models}/deterministic/`** (training parquet + whole-suite `report()` suppressed) so a run produces consumable artifacts without overwriting production. Gate runs on real cached `NBA_FGA.parquet` (4000 rows, ~5s) with stochastic LightGBM (`feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`): different seed → `loc` max-abs diff ~0.34, same seed bit-identical. Default `meditate` byte-identical. P1 unblocked. |
| **P1 — centered-target bridge (SkewNormal)** | ✅ done, result: **FGA-only SHIP, family-wide KILL** | Two centered-target variants A/B'd path-wide under `--deterministic` vs `ratio_meanyr`. (a) `centered_additive_eb_meanyr_k10` (EB(MeanYr, K=10)): FGA SHIPS (+5.3% top-decile MAE, brier_skill +0.096→+0.112), every other SkewNormal market KILLs (PTS −3.5%, PA −4.1%, PR −2.9%, RA −2.2%, FG3A −3.8%, FGM −2.6%, MIN +3.7%, PRA +0.8%, REB +0.2%, fpp brier_skill regressed). (b) `centered_additive_mean10` (trailing-10): **every market KILLs** incl. FGA (+4.6%, under the 5% bar), PA −6.6% / PR −6.7% notably worse. Count-family markets show exactly 0% delta under both (transform is a no-op for NegBin/ZINB). Confirms the OVERCONFIDENCE_INVESTIGATION §3.2 "decisive negative result": the SkewNormal level bias is not the dominant compression cause path-wide regardless of baseline horizon. FGA is genuinely special (EB(MeanYr) captures structural shot-volume; Mean10 too noisy). Default `--target-strategy=ratio_meanyr` stays. Infrastructure (`baselines.py`, registry, `offset_meta` pickle field, brier_skill gate, live-path test) is reusable for P3/P2. |
| **P2.B — HurdleZINB (derived-π gate)** | ✅ done, result: **6/8 NBA ZINB markets SHIP** | New `meditate --zinb-mode=hurdle` (orthogonal to `--target-strategy`; default `joint` byte-identical to pre-P2.B). `HurdleZINB` (`src/sportstradamus/hurdle.py`): calibrated binary classifier estimates `q = P(Y=0)`; NegBin LightGBMLSS on `Y>0` supplies count shape; structural-inflation π derived via `π = clip((q − NB(0))/(1 − NB(0)), 0, 1)` (NOT the simpler `gate = 1 − p_nonzero` — corrected because downstream `fused_loc` treats `gate` as zero-inflation, not marginal P(Y=0)). Returns `total_count/probs/gate` matching the ZINB contract so `model_prob` decode (lines 252-257) is untouched; legacy pickles load via `getattr(model, "is_hurdle", False)`. Path-wide A/B vs joint: **SHIP** FG3M (+9.7%, brier_skill +0.115→+0.290), OREB (+44.9%, +0.019→+0.109), PF (+19.2%, −0.238→−0.002), TOV (+26.8%, −0.049→+0.058), BLK (+40.4%, +0.237→+0.299), BLST (+11.6%, −0.002→+0.093). **KILL** FTM (+1.3%, under bar), STL (global MAE +14.1% regression). Joint ZINB had per-row catastrophic blowups in mid-deciles on BLK/OREB/PF/BLST under deterministic mode (compression_ratio 24–5357×; predicted means up to 1437) that the hurdle eliminates entirely — global MAE drops 60–99% there. **Default stays `--zinb-mode=joint`** (per-market routing is a follow-up). Verdict-criterion note: the parent "predicted gate mean ≈ hist_gate" was mis-stated under derived-π — derived-π gate is π_zi (≤ q, equality only in the ZTNB limit). For FG3M (positives mean ≈ 2.2) NB(0) ≈ 0.20 → derived-π gate ≈ 0.17 (≈ joint's 0.18) but total reconstructed P(Y=0) matches `q ≈ 0.33` by construction. The meaningful signal is the downstream compression_eval verdict on `P(over@line)` proxies, not gate mean. New `tests/integration/test_zinb_hurdle_live_path.py` asserts `π + (1−π)·NB(0) ≈ q` per-row (tol 0.02) + two-run bit-identity. Determinism gate extended with a parallel hurdle assertion. |
| **P2.A — `init_score` baseline (NegBin/ZINB)** | ✅ closed: **DEAD** | In-process FG3M spike: LightGBMLSS accepts per-row `init_score` (length-2n flat `[log_EB, zeros]` per-parameter concat) without raising — but predictions are **byte-identical** to a plain NegBin fit, every decile. Either LightGBMLSS overrides init_score with its own `start_values`, or the 30-round deterministic fit converges regardless of start. FG3M's plain-NegBin top-decile bias is already −0.013 — no meaningful compression on the count-branch NegBin mean; the overconfidence was the gate, which P2.B addresses. **DEAD** as a one-line `init_score` transform. Fallback is P5 (target-encoded baseline feature) or P3 (rate decomposition) — own design sessions. |
| **Stage 0 — live-data instrumentation** | ✅ done (PR #46) | All five deliverables shipped: **0.1** `compute_book_brier_skill_score` in [analysis.py](../src/sportstradamus/analysis.py) (8 unit tests, hand-ref within 1e-6); **0.2** `_compute_live_metrics` + Step 6 in [nightly.py](../src/sportstradamus/nightly.py) writing `data/live_metrics_per_market.parquet` (locked 10-col / 2-window schema; 6 round-trip tests incl. empty-window `n_settled=0`); **0.3** `compression_eval --live-window N` using `_history_to_eval_frame` + per-league `_load_league_stats_lookup` (Stats-backed MeanYr lookup, monkeypatch-mockable; 8 new tests); **0.4** [scripts/check_graduation.py](../src/sportstradamus/scripts/check_graduation.py) joining Gate 1 (model_stats.parquet) × Gate 2 (live_metrics, 30d) with `_classify_lifecycle` → {not-shipped, in-test, graduated, demoted}, colored table (11 tests incl. 7 parametrized state-machine cases); **0.5** [scripts/backfill_live_metrics.py](../src/sportstradamus/scripts/backfill_live_metrics.py) walking history backwards with `--days/--step` + idempotent day-precision dedup (5 tests). All three always-on gates green (ruff, 113 golden, 9 integration); 30 new tests. refactoring-specialist applied two minor fixes. Two console scripts registered: `check-graduation`, `backfill-live-metrics`. `meditate`/`prophecize`/`confer` byte-identical; `reflect` gains tail Step 6. Track-A/B graduation lookups are now a parquet read. |
| **Stage B1 — ZTNB likelihood fix + routing diagnostics** | ✅ done, result: **ZTNB hypothesis REFUTED; routing rescope delivered** | **B1.1 (ZTNB):** correct in isolation (`tests/test_ztnb_loss.py`, scipy-referenced) but **incompatible with the frozen derived-π hurdle decode**. On FG3M the ZTNB count component implies `NB(0) ≈ 0.41` vs observed `q ≈ 0.31`, so on **65% of rows `NB(0) > q`** → π clips to 0 → identity breaks (`test_zinb_hurdle_live_path` diff **0.136** vs 0.02 tol). E[Y\|Y>0] ~unchanged (old 2.2 vs ZTNB 2.12); ZTNB only re-decomposes the positive mean — exactly what breaks the decode. The fix would **regress the 6 P2.B SHIP markets**. Wire-in **reverted**; `_ZeroTruncatedNB` kept as an unwired, test-covered block for MZINB (B3). Smoke A/B not run (no stats.nba.com network); analytical verdict KILL. **B1.2 (routing):** read-only `scripts/zinb_routing_diagnostics.py` + golden test + `statsmodels` dep; writes `data/zinb_routing/{LEAGUE}_diagnostics.parquet` for all 23 cells. **0/23 route to `hurdle_nb_ztnb`; 13 → `cmp` (var/mean ≤ 1.3), 10 → `mzinb`.** The blanket ZINB label is wrong for ≥ 13 markets. **STL → cmp** (underdispersed, the kill's real cause), **FTM → mzinb**. Tooling: new `meditate --market`; `select_markets` relocated `cli.py → training/markets.py`. |
| **Stage B1.5 — §7a likelihood-vs-features pre-check** | ✅ done, result: **FEATURES, not likelihood — family build DEFERRED** | Poisson-GBM pre-check (`research-analyst`) on 9 cells: NBA FTM/STL/TOV/FG3M, WNBA STL/FG3M/FTM, NFL interceptions/rushing-tds. Top-decile compression **persists under a plain Poisson mean head** (Poisson top-decile CR 0.16–0.35 vs NB/ZINB 0.12–0.37) → family-INVARIANT GBT leaf-averaging bias [3][4], not a likelihood problem; **no CMPμ/MZINB swap can fix it.** Conditional Dunn–Smyth RQR variance collapses the "underdispersed" reps to ~1.0–1.08 (STL/TOV/WNBA-STL equi-dispersed → fail `CMPμ iff cond<0.90`); **no CMPμ candidate among NBA/WNBA.** Only 2/9 clear ≥5% (FG3M +6.3%, rushing-tds +10.3%) and both are **upward mean-bias** (over-predict low-volume players — inverse of Track B's symptom); ~half the gain is pure bias-recentering. **Pivot Track B to a ~1–2 wk feature/bias track**; re-enter the 9–12 wk family build only on a cell that still kills. No pickle written. |
| **Stage A1 — SkewNormal ICC diagnostic gate** | ✅ done, result: **family clusters AMBIGUOUS — ICC alone does not cleanly route SkewNormal** | Read-only `scripts/icc_diagnostics.py` (console `icc-diagnostics`) + `tests/golden/test_icc_diagnostics.py` (15 tests). ICC₁ via two-level moment decomposition (σ²_between = Var(player-season means), σ²_within = mean within-(player-season) variance) over (Player, season) groups (season via the `stats/base.py:527-528` Aug-boundary rule, since NFL/WNBA caches are multi-season: NFL 2021–26, WNBA 2022–25; NBA single-season). Participation filter (nonzero-game fraction ≥ 0.5, **no position map**) resolves the NFL position-confound; skew-driven transform escalation raw→log1p→rank (all 36 cells landed `raw`). Writes `data/icc/{NBA,WNBA,NFL}_icc.parquet`. **Routing verdict: 25 ambiguous, 10 eb_centering, 1 tail_extension.** NBA (ICC 0.27–0.51): only PA 0.514→eb_centering, DREB 0.274→tail_extension, other 11 ambiguous; **FGA 0.489 (NOT the predicted >0.6)**, **PTS 0.473**. WNBA (0.37–0.57, slightly *higher* than NBA, not noisier — 4-season pooling, n_player_seasons ≈ 480–530, is stable): 5 eb_centering (MIN/FGA/PRA/PA/PR). NFL (0.41–0.79): qb-yards 0.790, carries 0.666, targets 0.507, rushing-yards 0.502 → eb_centering; participation filter excluded ~1380–1391 non-QB player-seasons on passing-yards/attempts/completions (kept ≈ 90–92 = QBs). **Decision triggers:** "ICC_PTS > 0.5 on any league" did NOT fire (T7 does not jump to A2); no bimodal split, family not uniformly-low → the 25 ambiguous cells sit in "try both, route on outcome" → A2 runs EB-centering *and* tail-extension per-market. **ICC does not predict the P1 EB ship/kill** — FGA SHIPPED at 0.489 while PA (highest NBA ICC 0.514) KILLED; ICC is unconditional and the production model already carries a ~280-col feature matrix capturing much between-player level. **eb_K** per-league median ≈ NBA 1.4 / WNBA 1.1 / NFL 1.2 — **NFL is NOT an outlier**; further, the moment eb_K is a downward-biased estimate of the Casella–Berger [6] EB constant, so the table cannot assert "K=10 too high" — re-derive K bias-corrected before acting in A2. No model/inference change. Gates green; refactoring-specialist run. |
| **Stage A1.5 — factor-ICC de-risk (T5 fork gate)** | ✅ done, result: **T5 KILLED as a wholesale architecture; A2 pivots to T3 tail head** | research-analyst brief computed factor-level ICC read-only by reusing A1's tested engine — reference markets reproduce the A1 parquet ICCs to 1e-6 (NBA MIN 0.3468, FGA 0.4890), inheriting A1's 15-test coverage. **Band verdict (median ICC(volume) − median ICC(efficiency); pre-registered bands 0.20/0.10):** NBA gap **+0.232** (vol 0.391/eff 0.158) → **MIXED** (volume clause FAILED: 0/3 NBA volume factors ≥ 0.5 — MIN 0.347, FGA 0.489, FGA-per-MIN 0.391, the same band where P1 shipped EB on FGA yet KILLED PA); WNBA gap **+0.456** (vol 0.559/eff 0.103) → **CONFIRMED but low-confidence** (single computable efficiency factor — no `FGM`/`FG3A` markets); NFL gap **+0.291** (vol 0.507/eff 0.216) → **CONFIRMED** (carries 0.666, targets 0.507; yards-per-* heavy-tailed, skew +2.9 to +4.4). Efficiency factors low-ICC in all three leagues (**8/8 ≤ 0.30**) — "efficiency = noise" strongly confirmed, matches [2]; "volume = stable identity" FALSE for NBA. **Literature OVERRIDES the band-only CONFIRMED to a T5 KILL on the body of the stat:** Goodman [1] gives CV²(XY) = CV²(X)+CV²(Y)+CV²(X)CV²(Y), and on the actual NBA top-mean-decile PTS data, recomposing FGA × (PTS/FGA) inflates within-player-season CV to **0.423 vs 0.334 modeling PTS directly (+27%)** — recomposition discards the structural volume↔efficiency negative covariance and re-inflates the tail; disaggregate forecasting wins only with known-DGP components, not estimated GBDT heads [7]. The line-492 "DFS-industry consensus" is practitioner **lore** (no peer-reviewed tail-bias validation). **A2 fork: build T3 (1–2 wk) as primary; do NOT build T5 (2–3 wk + largest inference change, 36 cells).** T5's logic survives only as a narrow **NFL per-factor route** (carries/targets → EB, yards-per-* → T3) as an A3/A4 follow-on — routing, not recomposition, avoiding the +27% penalty. Verdict-only gate, no `src/` change. |
| **Stage A1.6 — NFL position-split matrix cleanup + WNBA test-case fix** | ✅ done | Moved the NFL position confound from a read-side workaround (A1's nonzero-fraction filter) to a **write-side fix**: `NFL_MARKET_POSITIONS` constant + `_market_position_filter` hook (no-op default in [base.py](../src/sportstradamus/stats/base.py), NFL override in [nfl.py](../src/sportstradamus/stats/nfl.py)) called in `get_training_matrix` before the usage cutoff. Scoping: passing + QB total-offense → **QB**; rushing → **QB+RB**; receiving + skill scrimmage → **WR+RB+TE**; fantasy-points composites stay all-position. Cached parquets cleaned in-place via [scripts/prune_nfl_matrix_positions.py](../src/sportstradamus/scripts/prune_nfl_matrix_positions.py): **passing-yards 15000→2646 rows, mean 38.1→215.9**; attempts 5.4→30.4; qb-yards→QB-only 45.3→228.0. **ICC re-run:** NBA/WNBA value-identical; all NFL shifts downward (exact scoping removes gadget players): passing-yards 0.420→0.405, receiving/targets/receptions ≤0.001, **qb-yards 0.790→0.423** (632→222 player-seasons — the 0.790 was a QB-vs-RB between-position artifact), yards 0.470→0.440, carries 0.666→0.627 / rushing-yards 0.502→0.475, attempts 0.458→0.436 / completions 0.451→0.429. New NFL ICCs supersede A1's for A2 routing. No pickle/inference change; parquets gitignored ⇒ PR diff is code + script + plan. WNBA efficiency test cases: `FTM_per_FGA` + `FG3M_per_FGA` with `PTS_per_FGA` (0.103) anchor. **New Stage A4 entry T11: per-position model-split bias experiment.** Two flagged inference edges for A2: QB-only `qb yards` vs Underdog "Total Yards"; QB+RB `carries` excludes gadget WR-rushers. Gates green; refactoring-specialist run. |
| P3–P10 | ⬜ | See priority list; P10 (GPBoost) already prototyped and failed deterministically — annotated below. |
| **Docs rework — breadth-led reorg + roadmap sync** | ✅ done (docs-only) | Reorganized this plan so the breadth North Star + per-cell status table lead, depth Tracks A/B labeled secondary, the diminishing-returns pre/post-break split marked, and a "Branches & model-promotion flow" section added (four-branch pipeline + `devel-ship-curator`). Synced `docs/sportstradamus_roadmap_v2.md`: snapshot → 2026-05-21, model work promoted to a leading "Active Track" phase, post-break tail deferred into Phase 6, Standing rules + "Tools and CLIs built" added. No `.py`/`.json`/threshold changes; no retrain. |
| **B1.6 — breadth research verdict (research-analyst)** | ✅ done (research-only) | A cited 2026-05-22 brief (`/tmp/researcher_breadth_75.md`) confirming the breadth gap (NBA +3, WNBA +4, NFL +2) is a **bias-gate** problem, with post-hoc mean-bias correction as the **rank-1 on-mechanism lever** (the only method moving both gated bands by construction), the expanding-mean/EB-shrunk player feature rank 2, opponent features rank 3, BSS-floor gate-tuning as mop-up. Per-league targets set (NBA TOV/BLST/OREB/PF; WNBA TOV/BLST/OREB/STL; NFL rushing-tds/receiving-tds via affine ROE). Flagged NBA/WNBA AST as SkewNormal (Track-A treatment, not count-family). No data re-run (clean checkout — no cached parquets/models); the Tier-0 absolute-only `compression_eval.verdict()` mode named as the hard precondition. New refs [43]–[48]. |

## Prior-phase results

The breadth-audit narratives behind the Roadmap summary table in the plan. The plan
keeps the summary breadth-counts table and a one-line pointer here; the per-cell BSS
lists, methodology findings, the WNBA TOR-fix prose, and the NFL pruning detail live
below.

**Step 1 — breadth baseline (2026-05-21).** Tier-0 audit over four candidate
strategies (`ratio_meanyr` incumbent, `centered_additive_mean10`,
`centered_additive_eb_meanyr_k10`, `ratio_meanyr` + hurdle) on the deterministic
A/B test sets, after fixing the NFL gamelog numeric-coercion bug (NFL candidates
were previously un-generatable). BSS here is on crippled-HP deterministic models —
**conservative**; the final per-cell pick uses the highest *full-retrain* BSS. The
absolute bias gates are HP-robust.

| League | passes bias gates | bias + BSS ≥ 0 | 75% target | gap |
|---|---|---|---|---|
| NBA | 14/21 | 13/21 | 16 | −3 |
| WNBA | 10/18 | 10/18 | 14 | −4 |
| NFL | 17/20 | 16/20 | 15 | **MEETS (+1)** |

The binding constraint is the **absolute bias gates, not the BSS floor**. Failing
cells are the low-mean count markets — NBA AST/BLK/BLST/FG3M/OREB/STL/TOV, WNBA
those + FTM, NFL passing-tds/receiving-tds/rushing-tds — exactly the
family-invariant-compression cells Stage B1.5 routed to the feature/bias track. The
crippled-HP BSS floor excludes only one extra cell per league beyond the bias
failures (NBA PF, NFL interceptions — both "bias-only"), so a full retrain should
recover several. Winning strategy varies by cell: `ratio_meanyr` takes most NFL
high-volume markets + every fantasy-points cell; `centered_additive_eb_meanyr_k10`
takes the NBA/WNBA volume markets (FGA, MIN, DREB, PA, PR, PRA, REB); the hurdle
wins once (NFL sacks-taken). Per-cell selection is material — no single strategy
dominates. Gap to close (Step 2): NBA +3, WNBA +4. NFL's screen 16/20 fell to
**13/20 confirmed at full HP** (qb-tds, rushing-yards, sacks-taken failed Tier-0 on
the real retrain — see Step 3).

**Step 3 — baselines locked (2026-05-21).** Full-HP retrain driven by per-cell
`ship_config.json` (`--force`, non-deterministic). Methodology finding: a fresh
retrain's test sets ARE `compression_eval`-scoreable (they carry the book-odds
columns), unlike the pre-existing May-8 production sets — so full-HP BSS
confirmation is available going forward.

- **NBA — 13/13 locked, full-HP confirmed.** Every baseline-able cell passes Tier-0
  (BSS +0.027 … +0.352: MIN +0.284, fpp +0.352, DREB +0.251, REB +0.131, PTS
  +0.097, FGM +0.094, FGA +0.093, FTM +0.087, PR +0.074, FG3A +0.061, RA +0.060, PA
  +0.049, PRA +0.027). The crippled-HP "centered wins" in Step 1 were all 0–2pp
  noise; `ratio_meanyr` is best-or-equal, so all 13 lock as `ratio_meanyr`.
- **WNBA — 10/10 baseline-able locked, full-HP confirmed (TOR fix landed).** The
  retrain had crashed in `get_stats`: `teamProfile.loc[…]` raised
  `KeyError: ['TOR']` because Toronto Tempo (2026 WNBA expansion) is absent from
  `teamProfile`; root cause was a pair of hardcoded `"GSV"` (2025 Golden State
  Valkyries) guards that papered over only the *prior* expansion. Fixed with a
  league-agnostic `_profile_rows_for_teams` helper
  ([base.py:51](../src/sportstradamus/stats/base.py#L51)) that `reindex`es the
  profile so any absent franchise yields an all-NaN row (consumed as a missing
  feature) — robust to future expansions; both GSV guards deleted; regression test
  `tests/golden/test_profile_team_lookup.py`. All 10 markets then completed cleanly
  (BSS +0.013 … +0.230: fpp +0.230, MIN +0.189, FGA +0.137, DREB +0.096, PA +0.050,
  PR +0.048, REB +0.041, RA +0.036, PTS +0.033, PRA +0.013). DREB locks as
  `centered_additive_mean10` (sole Tier-0 passer in the screen — incumbent fails the
  bench gate — and fully plumbed, no `zinb_mode` gap); the other 9 as `ratio_meanyr`.
  This is 10/18 of all WNBA markets; the remaining +4 is the Step-2 feature/bias
  track on the count markets (FG3M/FTM/STL …).
- **NFL — 13/16 baseline-able locked, full-HP confirmed.** Per-cell `zinb_mode`
  plumbing was built first (object-form `ship_config` cells + `resolve_cell_zinb_mode`;
  `ZINB_MODES` in `baselines.py`) so sacks-taken *could* ship as `ratio_meanyr`+hurdle.
  The full-HP `--force` retrain confirmed only **13 of the 16 screen-passers** — the
  crippled-HP screen was optimistic for 3: `qb tds` (full-HP bench +0.39 / star +0.68
  over bounds), `rushing yards` (`centered_additive_mean10` BSS −0.006; incumbent also
  failed it), and **`sacks taken`** (hurdle: star-bias +0.67 > 0.50 — plumbing works
  end-to-end, pickle `is_hurdle=True`/`zinb_mode=hurdle`, but the cell fails Tier-0
  anyway). Locked 13: `passing first downs`→`eb`, `receptions`→`mean10`, the other 11
  →`ratio_meanyr`. The centered transforms blow up on NFL yardage (passing-yards `eb`
  top-decile MAE 2629, qb-yards `mean10` 44902) — incumbent unambiguously correct
  there. **NFL lands 13/20, short of 15/20 by 2**; the 3 pruned + 4 never-baseline-able
  = a 7-cell Step-2 pool. Full per-cell results + the devel ship handoff are in
  [docs/archive/lockin_ship_handoff.md](archive/lockin_ship_handoff.md).

Depth work (superseding baselines via the ≥ 5% bar, Track A tail-head, Track B
family builds) is **secondary** to reaching 75% breadth.

## Context

LightGBMLSS predictions compress toward the global mean. Two branches: SkewNormal
(`global_mean >= 2.0`, e.g. NBA PTS/FGA) uses a `Result/MeanYr` target +
multiplicative denorm; NegBin/ZINB (`global_mean < 2.0`) uses raw counts. P1 closed
the centered-target question on SkewNormal (FGA-only ship); P2.B closed the
joint-vs-hurdle question on ZINB (6/8 ship). The next-session plan attacks (a) the
remaining SkewNormal top-decile bias path-wide and (b) the FTM/STL ZINB kill. Deeper
context: [docs/OVERCONFIDENCE_INVESTIGATION.md](OVERCONFIDENCE_INVESTIGATION.md)
(determinism, live-path confound) and
[docs/CENTERED_TARGET_NEGATIVE_RESULT.md](CENTERED_TARGET_NEGATIVE_RESULT.md) (the
path-wide P1 KILL).

## Architectural principle (applies to all phases)

Make the **target/baseline transform a single configurable strategy**, selected by a
CLI flag on `meditate` (and a matching env var for the harness), defaulting to
current behavior. Every experiment is a new strategy value, not a destructive
rewrite — this keeps the multi-session A/B tractable and `devel` shippable between
sessions. Centralize the forward transform, the inverse (de-norm) transform, and the
inference-time mirror so train/predict cannot drift. The inference mirror lives in
[stats/base.py:597](src/sportstradamus/stats/base.py#L597) (`get_stats`); any new
baseline must be computed there identically and leakage-safe. STYLE_GUIDE §9 (named
constants), §18.9 (no orphan methods), and CLAUDE.md "no new monoliths" apply.

(The Universal decision threshold / Tier-1 gate — condition (4) asymmetric betting
risk and all — is restated in the plan's North Star section, where the gate work is
executed.)

## Research handoffs that fed this plan

The next-session plan integrates two researcher passes after P1 and P2.B. The seed
reports are not committed (`/tmp/researcher_skewnormal.md`, `/tmp/researcher_zinb.md`
in the originating session); load-bearing citations and recommendations are below.
Future research handoffs are produced **in-repo** by the
[research-analyst subagent](../.claude/agents/research-analyst.md) (brief lands at
`/tmp/researcher_{topic}.md`; the main session distills its conclusions here). See
"Research handoffs (in-repo)" below.

- **SkewNormal track** — 10 new methods (T1–T10) for top-decile bias on SkewNormal
  markets (PTS/REB/AST/PRA/FGA/MIN/PA/PR/FG3A/FGM). Headline: ICC₁ per-market routing
  first; original P3 rate decomposition / P5 target-encoded features / P10 GPBoost
  retry are all KILLED in favor of T-method replacements.
- **ZINB track** — focused on the FTM/STL kill markets. Headline: the "fit on
  positives" Stage-2 NegBin is a misspecified ZTNB; a single-line PyTorch fix likely
  resolves FTM/STL by construction. Per-market routing diagnostics (ziNB index,
  Wilson-Einbeck, Schwarz-corrected Vuong) second; the Stage 3 architectural choice
  (MZINB vs GPBoost) is a fork chosen by residual analysis, not run in parallel.

All citations are collected, numbered, in **References** at the end of this document.

### Research handoffs (in-repo)

When a diagnostic is ambiguous or a path-forward decision needs literature + statistical
synthesis, dispatch the [research-analyst subagent](../.claude/agents/research-analyst.md)
(`subagent_type: "research-analyst"`) — the in-repo replacement for the claude.ai research
round-trip. It reads the diagnostic outputs (may re-run read-only diagnostics), searches
the primary literature, and writes a cited statistician's brief.

- **Input** (optional): a prompt at `/tmp/{topic}_research_prompt.md`, or inline.
- **Output**: a brief at `/tmp/researcher_{topic}.md` (TL;DR / Key Findings with DOIs /
  Recommendation / Reality checks / Open questions / Bibliography), ending with a
  "Load-bearing conclusions for the plan" list.
- **Distillation**: the main session copies the load-bearing conclusions into this plan's
  "research verdict" / "Open questions" / "Cross-league caveats" sections. The agent does
  not edit the plan and is read-only w.r.t. production (no pickles, no default flips, no
  inference-path or `src/` edits).

## Stage 0 — Live-data instrumentation (prerequisite for everything else)

The graduation criteria reference metrics not currently persisted in usable form.
Before any track work that uses the stop-the-track principle, Stage 0 must ship —
otherwise "settled brier_skill ≥ 0 over 30 days" is rhetorical. (Stage 0 is **done** —
see Status log.)

What existed before Stage 0:

| Component | State |
|---|---|
| [analysis.py:877](../src/sportstradamus/analysis.py#L877) `compute_brier_skill_score(subset, base_rate=0.5)` | Against chance, **not the book** — not comparable to the training metric. |
| `history.parquet` with settled `Actual` | Exists (filled by [nightly.py](../src/sportstradamus/nightly.py) on every `reflect`). Per-offer; not aggregated per cell or over windows. |
| Dashboard pages 3/4/6 | Compute brier/BSS/profit-sim **on demand**; not persisted; not programmatically readable. |
| Live top-decile MAE on settled bets | **Did not exist** (compression_eval only scored `data/test_sets/` CSVs). |
| Per-(league, market) persistent metric store | **Did not exist** (no `live_metrics_per_market.parquet`). |

Stage 0 deliverables (dependency order):

| # | Deliverable | Cost | Where |
|---|---|---|---|
| 0.1 | **Live book-BSS** — variant of [analysis.py:877](../src/sportstradamus/analysis.py#L877) using the bookmaker implied prob (per-offer `Odds` in `history.parquet`) as reference, mirroring training `brier_skill_score`. Added as `compute_book_brier_skill_score(subset)`; keep the chance-baseline version. | ~2h | [analysis.py](../src/sportstradamus/analysis.py) |
| 0.2 | **Rolling-window aggregation per (league, market)** — extend `reflect` to compute, each nightly run, 7d/30d rolling book-BSS, empirical-vs-predicted over-rate, total settled bets, profit-sim yield → `live_metrics_per_market.parquet` keyed by (league, market, computed_at). | ~1d | New `compute_live_metrics()` at tail of [nightly.py](../src/sportstradamus/nightly.py) `run()`. |
| 0.3 | **Live top-decile MAE harness mode** — `compression_eval --live-window N` reads `history.parquet`, filters to last N days of settled bets per cell, runs the decile-bias path. Output schema matches offline mode (parquet join). | ~1d | [compression_eval.py](../src/sportstradamus/scripts/compression_eval.py) + tests. |
| 0.4 | **Lifecycle-status table view** — `poetry run check-graduation` joins `live_metrics_per_market.parquet` against **both gates** → per-cell not-shipped / in-test (days into soak) / graduated. 36-cell × 8-metric table (4 Gate 1 + 4 Gate 2 + state). | ~½d | New [scripts/check_graduation.py](../src/sportstradamus/scripts/check_graduation.py). |
| 0.5 | **Backfill rolling metrics** — one-shot script walking back ~90 days through `history.parquet` so the first publication has historical context. | ~½d | One-shot under [scripts/](../src/sportstradamus/scripts/). |

**Total Stage 0 cost: ~3 days.** Stage 0 ships once; every subsequent graduation check
is a 30-second parquet read.

**Stage 0 ship gate** (infrastructure, separate from the universal threshold):
`live_metrics_per_market.parquet` exists with ≥ 7 days of history; live book-BSS (0.1)
matches a hand-computed reference within 1e-6 on a 100-row spot check;
`compression_eval --live-window 30` is deterministic on a frozen snapshot (golden test
asserts row-count + schema); a `check_graduation` invocation produces a non-empty table
covering NBA/WNBA/NFL × all distributions with the state column populated; the 8-metric
output reads from the same parquet on every invocation (no recompute from separate
sources). **Stop-the-track check:** there is no "enough live data" exit for Stage 0 — it
*is* the infrastructure; skip it only if you also skip the stop-the-track principle.

## Two-track depth plan (secondary to breadth)

> **Depth is secondary to the breadth North Star.** Both tracks supersede a *set*
> baseline with a better one (the Tier-1 ≥ 5% bar); they **begin only once the
> 75%-per-league breadth gap closes** (NBA −3, WNBA −4, NFL −2 as of 2026-05-21).
> Active pre-break work is the breadth push (Tier-0 audit → Stage B1.6) plus core
> depth A2/A3/B2/B3; the post-break tail (A4/B4) is deferred —
> [roadmap](sportstradamus_roadmap_v2.md) Phase 6.

Work proceeds on two independent tracks. Stages within a track are sequential (later
stages depend on earlier diagnostics). The tracks share infrastructure (harness,
determinism gate, per-strategy registry) but diagnostics and method choices diverge.

---

## Track A — SkewNormal markets

*Depth work — secondary to 75% breadth; begins after the gap closes. A2/A3 active
(pre-break); A4 deferred (roadmap Phase 6).*

Source: `/tmp/researcher_skewnormal.md`. Scope: PTS, REB, AST, PRA, FGA, MIN, PA, PR,
FG3A, FGM, fantasy points. After P1, FGA ships with EB(MeanYr, K=10) centering; every
other SkewNormal market killed under both EB-centered and Mean10-centered strategies.
Researcher hypothesis: the dominant compression cause is **volume-efficiency
entanglement** (PTS = FGA × eFG% × …), not leaf-averaging, and ICC per market predicts
which strategies work.

### Stage A1 — Diagnostic gate (~1 day) — DONE (see Status log)

**T1. ICC per league × per market.** ICC₁ for every SkewNormal market in every league
(36 cells) on three seasons of held-out data via a two-level ANOVA decomposition
(σ²_between = Var(player season means), σ²_within = mean within-player variance,
ICC = σ²_between/(σ²_between + σ²_within)); highly-skewed markets on `log(1+Y)` or rank.
Output a 36-row table keyed by (league, market). Pre-registered routing cutoffs: ICC ≥
0.5 → EB-style centering can work (T5 factorization); ICC ≤ 0.3 → distributional tail
extension (T3/T7); 0.3 < ICC < 0.5 → ambiguous, try both. Expected (NBA): FGA > 0.6,
PTS 0.3–0.45, STL/BLK < 0.2. Implementation: `scripts/icc_diagnostics.py` consuming
`data/training_data/{LEAGUE}_{market}.parquet`, output `data/icc/{LEAGUE}_icc.parquet`.
No model changes. (Source: researcher T1, Tier 0.)

Decision triggers (and what fired): ICC_PTS > 0.5 on any league → entanglement wrong,
T7 jumps to A2 (**did not fire**); uniformly-low family → T3 priority over T5; bimodal
clusters → per-market routing in A2 justified; **NFL-specific:** high ICC but noisy
K=10 → re-derive `K = σ²_within / σ²_between` per the Casella–Berger [6] formula per
league. Result (see Status log): **25 ambiguous, 10 eb_centering, 1 tail_extension** —
no trigger fired; A2 runs EB-centering *and* tail-extension per-market. Stop-the-track:
diagnostic only, no graduations. Inference: no model change.

### Stage A1.5 — research verdict (factor-ICC de-risk of T5) — DONE

The `research-analyst` reviewed the T5 factor-ICC question (brief
`/tmp/researcher_factor_icc.md`). It computed **factor-level ICC** read-only by reusing
the A1 engine (reference markets reproduce A1 parquet ICCs to 1e-6, inheriting A1's
15-test coverage). The verdict **KILLs T5 as a wholesale multiplicative architecture**
and routes A2 to **T3**.

**Headline.** The T5 premise is *half* true, and the wrong half is true for the lift.
Efficiency factors are low-ICC in all three leagues (8/8 ≤ 0.30) — exactly the half a
T3 tail head serves directly on the whole stat, without recomposition. The "volume =
high-ICC stable identity" half holds only for WNBA/NFL, not NBA. And the variance
algebra of multiplying separately-modeled factors inflates the priced tail. Build T3
(1–2 wk); do not build T5 (2–3 wk + the largest inference change).

**1. Band verdict** (median ICC(volume) − median ICC(efficiency); pre-registered bands
0.20/0.10):
- **NBA — MIXED.** Gap **+0.232** (vol 0.391/eff 0.158); volume clause FAILED (0/3 NBA
  volume factors reach the 0.5 EB floor — MIN 0.347, FGA 0.489, FGA-per-MIN 0.391, all
  ambiguous). Same band where P1 shipped EB on FGA (0.489) yet KILLED PA (0.514), so
  **decomposing PTS into volume × efficiency does not explain the P1 ship/kill split.**
- **WNBA — CONFIRMED, low-confidence.** Gap **+0.456** (vol 0.559/eff 0.103), but a
  *single* computable efficiency factor (`PTS_per_FGA`; no `FGM`/`FG3A` markets), so the
  median is one point and the gap is fragile (Cross-league caveat #3).
- **NFL — CONFIRMED.** Gap **+0.291** (vol 0.507/eff 0.216); volume clause 0.67,
  efficiency 1.00; NFL efficiency factors genuinely heavy-tailed (yards-per-carry skew
  +4.43, yards-per-attempt +2.92, all `log1p`) — the tail a Pareto/spliced T3 fits.

Gap robust to factor-classing (NBA 0.22–0.26, WNBA 0.37–0.46, NFL 0.29 across three
variants). Efficiency-low-ICC matches the basketball reliability literature [2] (FG3%
ICC 0.066 here = "most 3P% differences are sampling variability"; see also [13]).

**2. Literature OVERRIDES the band-only CONFIRMED to a T5 KILL on the body of the
stat.** Goodman's exact variance-of-products [1] gives
**CV²(XY) = CV²(X)+CV²(Y)+CV²(X)·CV²(Y) ≥ CV²(X)+CV²(Y)** — relative variances add
super-additively. On the *actual* NBA top-mean-decile PTS player-seasons, PTS modeled
directly has within-player-season CV **0.334**, but the recomposed FGA × (PTS/FGA)
product has CV **0.423 — a +27% predictive-variance inflation on the cell we price.**
Because PTS = FGA × (PTS/FGA) is an identity, the factors are negatively correlated
within games; modeling them independently and Monte-Carlo recomposing discards that
negative covariance and re-inflates the tail toward the Goodman independent bound. For
top-decile MAE (set by tail mass around the line), T5 fixes location and inflates the
tail — the wrong trade. Aggregate-vs-disaggregate forecasting says the same [7]: the
disaggregate forecast beats the direct one only when component DGPs are *known*; once
estimated (four–five GBDT heads each with its own SkewNormal mis-specification) "the
superiority is no longer assured."

**3. The line-492 "DFS-industry consensus" is practitioner LORE, not published method.**
The minutes × usage × efficiency funnel is genuine industry practice but has no
peer-reviewed validation that it reduces top-decile bias; one practitioner source warns
projections "are not some kind of simple multiplication problem." Restated below as
*convention*, not ship evidence — the only primary sources on products of
separately-modeled components ([1], [7]) point the other way for the tail.

**Reality checks.** Projected T3 lift is **single-digit percent**, best where the body
model gets location roughly right and the miss is tail mass — NFL yardage best regime,
NBA scoring worst. Mirrors the Track-B CMP "3–8%, not 30%"; T3's own validation is
out-of-domain (gbex on rainfall; LightGBMLSS normalizing-flow general-purpose). Neither
T5 nor T3 is a ship until it clears the universal threshold on every covered cell. What
would revive T5: (a) NFL per-factor *conditional* signal turning out large once features
are added (Open question #10 — the ICCs are unconditional and the ~280-col matrix
already absorbs much volume identity); (b) modeling the negative volume↔efficiency
covariance jointly (copula / joint multi-output head) — *more* engineering than T5, not
less. If T5 is trialed anyway (NFL only, lowest-risk cell): Gate 1 must verify
top-mean-decile MAE improves ≥5% **and** the recomposed predictive scale is not inflated
(the +27% CV) on both metric families (CRPS/log-score and top-decile MAE) — Open
question #9.

### Stage A2 — Highest-leverage structural fixes (2–3 sprints)

**A2 fork (set by A1.5): build T3 as primary; T5 KILLED as a wholesale multiplicative
architecture.** The low-ICC half of the volume/efficiency split is *efficiency* — what
T3 serves directly — while Goodman's algebra shows recomposition inflates the priced
top-decile tail (+27% CV on NBA PTS). T3 captures the realizable lift for ~half the
build cost and a fraction of inference risk. T5's logic survives only as a narrow **NFL
per-factor route** (carries/targets → EB, yards-per-* → T3) as an A3/A4 follow-on.

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **❌ T5-basketball — KILLED by A1.5. Four-stage multiplicative factorization (NBA + WNBA)** *(replaces P3)* | Tier 1 (Skew) | 2–3 weeks | Predict (a) P(plays), (b) MIN \| plays, (c) per-100-poss rate \| MIN, (d) for PTS FGA-per-100 × FG% × points-per-make; recombine via Monte Carlo. Routes each factor to its own ICC-appropriate strategy. The minutes × usage × efficiency funnel is DFS *convention*, not peer-reviewed tail-bias evidence (A1.5 Finding 3). | New `src/sportstradamus/factorize/` or extend [pipeline.py](../src/sportstradamus/training/pipeline.py); inference mirror in [stats/base.py](../src/sportstradamus/stats/base.py); per-market wiring via the P1 strategy registry. |
| **⚠️ T5-NFL — deferred by A1.5 to a narrow per-factor route (A3/A4; run a conditional residual pass first). Position-dependent factorization** | Tier 1 (Skew), adapted | 2–3 weeks (after T5-basketball) | No per-100-poss equivalent; stats position-locked. Candidate trees: passing yards = P(plays) × Snaps × Attempts/snap × Yards/attempt; rushing = P(plays) × Snaps × Carries/snap × Yards/carry; receiving = … × Targets/snap × Catch-rate × Yards/catch. A1 ICC tells each factor's route. `Player position` already a category in `X`. | Same `factorize/` package; NFL factor defs in `data/factorize_nfl.json`. **Defer until T5-basketball ships.** |
| **✅ T3 — A1.5 PRIMARY A2 BUILD. Spliced / Pareto-tail or normalizing-flow head** | Tier 1 (Skew) | 1–2 weeks per distribution | Body ~ SkewNormal up to learned threshold u, tail ~ Generalized Pareto above u, mixing weight per-row. LightGBMLSS v0.3.0 normalizing-flow head is the simplest production path. Direct attack on top-decile MAE without touching loc. | Custom PyTorch distribution alongside [skew_normal.py](../src/sportstradamus/skew_normal.py); dist selection in [pipeline.py:245-324](../src/sportstradamus/training/pipeline.py#L245). |

Decision points: [A1.5 fork — T3 primary, T5 killed wholesale; the "T5 ships globally"
branch applies only to the narrow NFL per-factor route, and only after the Open
question #10 conditional residual pass]. If T5 ships globally → new baseline, re-run EB
centering per factor; A3/A4 become polish. If T3 ships on high-ICC heavy-tail markets
but T5 doesn't → distribution was the bottleneck. If neither ships → ICCs pathologically
low (re-check T1) or the live-path confound (Model Skew=NaN, OVERCONFIDENCE_INVESTIGATION
§3.4) is consuming the gain — resolve the live path first. Stop-the-track: after ≥ 14
days live, cells that graduate skip A3/A4. **Inference: A2 is the largest inference-side
change in the plan** — T5 needs N pickles/market + Monte Carlo recompose; T3 introduces
a new distribution name (decode + blend in `model_prob.py` and `distributions.py`).
Inference-path test must exist before Gate 1; use the P2.B HurdleZINB template.

### Stage A3 — Calibration polish (1 sprint, mostly orthogonal — stack them)

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **P7. Isotonic on loc** | Original plan | hours | Fixes residual average bias on location. Cheap, monotone. | `IsotonicRegression(out_of_bounds="clip")` post-fit in [pipeline.py](../src/sportstradamus/training/pipeline.py) before test-set dump. |
| **T8. CQR with player-decile-local conditioning (LCMQR)** | Tier 2 (Skew) | 2–3 days | Post-hoc per-player-decile calibration of *scale*; orthogonal to P7 (loc). CQR [14] and LCMQR [15] give finite-sample marginal-coverage guarantees. | New `src/sportstradamus/calibration/` if > 50 lines; else inline in report path. |
| **P6. Reduce tree regularization** | Original plan | ~1 day | Widen Optuna: larger `num_leaves`/`max_depth`, smaller `min_child_samples`/`min_child_weight` at [pipeline.py:348-368](../src/sportstradamus/training/pipeline.py#L348). | Optuna search-space dict. |
| **T9. Monotone constraint MeanYr → loc** | Tier 2 (Skew) | 1 day | Smoke test only — LightGBM `monotone_constraints` forces non-decreasing loc-on-MeanYr. Diagnostic. | `monotone_constraints` arg; MeanYr column identified by index. |

Decision points: P7 + T8 ship ≥ 5% on top-decile MAE → calibration was the bottleneck,
A4 lower-ROI. T9 violated → feature set missing a volume driver; loop back to A1 with
that feature. Stop-the-track: if P7+T8+P6 push live graduation across remaining cells
(high probability), do NOT proceed to A4. Inference: P7/T8 are post-hoc calibration
objects (new pickle keys `isotonic`/`cqr`, applied after decode before `fused_loc`,
round-trip test); P6 and T9 are training-only.

### Stage A4 — Novel risky retries (only if A2/A3 leave a gap)

> **Deferred — post-break speculative tail; tracked under [roadmap](sportstradamus_roadmap_v2.md) Phase 6.** Body kept here (home of record); not active until A2/A3 leave a gap *and* breadth is met.

| Method | Source | Cost | Direct effect | Notes |
|---|---|---|---|---|
| **T4. MEGB / GBMixed** *(replaces P10 GPBoost retry)* | Tier 3 (Skew) | 1–2 wk (MEGB), more (GBMixed) | EM/BLUP mixed-effects boosting that fixes the bias GPBoost was criticized for in [9]. MEGB on CRAN + github.com/rid4stat/MEGB; GBMixed no public code. **Different mechanism from GPBoost** — prior failure does not predict failure. | Point prediction only for MEGB (loc); keep LightGBMLSS for scale/shape. |
| **T2. CatBoost ordered TS** *(replaces P5 leakage-safe target encoding)* | Tier 3 (Skew) | 3–5 days/market | Only published GBDT mechanism with a proof of unbiasedness for high-cardinality categoricals (`player_id`); dominates greedy mean encoding [17]. | Re-fit LSS heads using CatBoost. **Caveat:** proof is for log-loss/squared-error, not SkewNormal NLL — validate empirically. |
| **T7. gbex** | Tier 3 (Skew) | 1–2 weeks | Generalized Pareto tail boosting on exceedances, layered on the LSS body. Good parallel to T3. | [18]. Published validation on rainfall, not sports. |
| **T6. FAGTB adversarial penalty against MeanYr decile** | Tier 3 (Skew) | 1 week | Quantile-bucket MeanYr (10 deciles); adversary predicts decile from residual; penalize loc gradient by adversary loss [19][20]. | Custom LightGBM objective. Designed for binary attributes; quantile-bucket continuous MeanYr first. |
| **T10. PGBM** | Tier 3 (Skew) | 1 week | Mean + variance from a single ensemble without parametric distribution; avoids the SkewNormal shape bound. | [21]. Per [8], NB-class targets do **not** uniformly benefit from probabilistic over point GBM — validate per market. |
| **T11. Per-position model split** *(enabled by A1.6 position scoping)* | A1.6 follow-on | 1–2 wk (selective) | Train a separate model per (position, market) instead of one pooled cross-position model with `Player position` categorical. Removes GBDT pooling bias where *eligible* positions diverge (rushing-yards QB-scramble ~19 vs RB-workhorse ~37). Complementary to T3's tail fix. | **Selective, not wholesale.** Split only where eligible-position marginals diverge materially; tight receiving stays pooled. NFL ~17 games/season (caveat #1) → over-splitting starves/overfits; min-row guard + fallback to pooled+categorical below threshold. A/B per cell. Prereq: A1.6 clean scoping. Generalizes the A1.5 NFL per-factor route from factors to positions. |

Decision points: CatBoost alone ships > 5% on a low-ICC market → bias was `player_id`
encoding leakage; deprioritize the rest. MEGB ships on PTS not FG3M → confirms high-ICC
vs low-ICC dichotomy; route count markets to T3/T7. T11 ships on rushing-yards but not
tighter receiving → split those, keep rest pooled; if it overfits even on rushing-yards
(thin QB-rusher sample), pooled+categorical is near-optimal — drop T11. Stop-the-track:
A4 is last resort; a cell with settled brier_skill ≥ 0 in production but failing the
strict 5% offline bar is almost certainly NOT worth A4 — file "deprioritized: acceptable
live performance, offline gap academic." Inference: T4/T2/T10 are the most invasive
(different model classes — `is_megb`/`is_catboost`); T7 layers a tail model (multi-pickle
Monte Carlo); T6 is training-only. Prefer T7 over T2/T4/T10 when inference-engineering
capacity is the constraint.

(The Track-A drops — P3→T5, P5→T2, P10→T4 — are consolidated in **Decisions &
trade-offs**.)

---

## Track B — ZINB markets (FTM, STL kill recovery + 6 SHIP hardening)

*Depth work — secondary to 75% breadth; begins after the gap closes. B1.6 (feature/bias
track) is the active breadth lever; B2/B3 active (pre-break); B4 deferred (roadmap Phase
6).*

Source: `/tmp/researcher_zinb.md`. Scope: FG3M, FTM, OREB, PF, STL, TOV, BLK, BLST.
After P2.B, 6/8 ship under hurdle mode; FTM and STL kill. Researcher hypothesis: the
FTM/STL kill is a **mathematical artifact of "fit on positives" instead of a true
zero-truncated NegBin (ZTNB) likelihood** in
[hurdle.py:201](../src/sportstradamus/hurdle.py#L201) — the hurdle Stage 2 is a
misspecified ZTNB.

### Stage B1 — Isolate and diagnose (1 week) — DONE (verdict below)

Two tasks in parallel (Tier 1 from the researcher):

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **ZTNB Stage 2 likelihood fix** | Tier 1 (ZINB) #1 | hours | Replace `NegativeBinomial(μ,α).log_prob(y)` in the Stage-2 hurdle loss with `nb.log_prob(y) − log1p(−exp(nb.log_prob(zeros_like(y))))`. The optimizer recovers an unbiased μ; the derived-π identity gives the correct ZINB marginal mean (1−ψ)·μ. Eliminates the over-prediction on FTM/STL **by construction**. League-agnostic. | Wrap the existing NegBin loss in [hurdle.py](../src/sportstradamus/hurdle.py). |
| **Per-league × per-market routing diagnostics** | Tier 1 (ZINB) #2 | days | Dashboard with observed mean, variance, zero-rate p₀; ziP/ziNB indices with bootstrap CIs [22]; Wilson-Einbeck p-value [23]; Schwarz-corrected Vuong (HurdleNB vs ZINB) [24]; var/mean; E[Y\|Y>0] vs μ. Routes each cell to plain NB / HurdleNB(ZTNB) / MZINB / CMP. 23 cells. | New `scripts/zinb_routing_diagnostics.py`; output `data/zinb_routing/{LEAGUE}_diagnostics.parquet`. |

Routing rule (precomputable from training data, survives temporal split): ziNB CI
contains 0 + low overdispersion → **plain NB**; ziNB CI contains 0 + hurdle appropriate
→ **HurdleNB with ZTNB Stage 2**; ziNB > 0 robustly → **MZINB** (after B3) or
HurdleZINB-with-ZTNB; var/mean < 1.3 → **CMP**.

Decision points: FTM/STL flip to ship under ZTNB → bias was likelihood-level only, stop
and harden. Still kill → routing tells you structural vs deeper. STL var/mean < 1.0 →
pivot STL to CMP (B4) first. Stop-the-track: the ZTNB fix is the cheapest single change
and the researcher's strongest claim; the bar for proceeding past B1 is *unambiguous*
live-data failure. Inference: ZTNB is **training-only** (`HurdleZINB.predict()` returns
the same triple, derived-π unchanged); routing diagnostics are read-only.

### Stage B1 — outcome & Track-B rescope

**B1.1 — ZTNB likelihood fix: hypothesis REFUTED.** The zero-truncated NB
(`_ZeroTruncatedNB`, scipy-verified in `tests/test_ztnb_loss.py`) is correct in
isolation but **incompatible with the frozen derived-π hurdle decode.** Diagnostic on
FG3M (a P2.B *SHIP*), 4000-row deterministic fit:

| quantity | value |
|---|---|
| observed zero rate | 0.326 |
| classifier `q` (pred P(Y=0)) | 0.311 |
| ZTNB count-component `NB(0)` | **0.412** |
| frac rows `NB(0) > q` → π clips to 0 | **0.652** |
| ZTNB `μ_NB` (= base_ev) | 1.249 |
| `E[Y\|Y>0]` under ZTNB = 1.249/(1−0.412) | 2.12 ≈ empirical 2.22 |

The identity `q = π + (1−π)·NB(0)` requires `q ≥ NB(0)` per row. ZTNB recovers a count
component with *higher* zero mass than the old full-support-NB-on-positives fit, so
`NB(0)` exceeds `q` on most rows, π clips to 0, and the reconstruction overshoots q
(`test_zinb_hurdle_live_path` identity diff **0.136** ≫ 0.02 tol). E[Y|Y>0] is
essentially unchanged (old ≈ 2.2, ZTNB ≈ 2.12) — ZTNB only re-decomposes the positive
mean into (lower count mean, higher count-zero mass), which is exactly what breaks the
decode. The fix would **regress the 6 markets P2.B shipped.** Decision: revert the
one-line wire-in; keep `_ZeroTruncatedNB` as an unwired, test-covered block for the
MZINB head (B3). Smoke A/B not run (no stats.nba.com network); analytical verdict KILL.

**B1.2 — routing diagnostics: the ZINB label is wrong for most cells.** The marginal
diagnostics (23 cells) split into two clusters:

- **Underdispersed / near-Poisson → CMP (13 cells):** `var/mean ≤ 1.3`, ziNB ≈ 0
  (three *negative* = zero-deflation). NBA STL, BLST, TOV, OREB, PF; WNBA BLK, STL,
  BLST, TOV, OREB; NFL tds, rushing tds, receiving tds. NB/ZINB **cannot** fit var <
  mean — it forces overdispersion these markets lack; their high zero rates are low-mean
  *sampling* zeros, not structural inflation.
- **Overdispersed + mild inflation → MZINB (10 cells):** NBA FG3M, FTM, BLK; WNBA FG3M,
  FTM; NFL passing tds, qb tds, interceptions, sacks taken, passing first downs. Genuine
  overdispersion (var/mean 1.35–7.9), small-but-positive ziNB, Vuong favors ZINB.

**0/23 cells route to `hurdle_nb_ztnb`.** The two P2.B kills have *different* root
causes: **STL** is underdispersed (var/mean 0.99–1.17) → mis-labeled ZINB → wants **CMP**
(B4); **FTM** is genuinely overdispersed+inflated (highest ziNB ≈ 0.063) → wants
**MZINB** (B3).

**Rescope (supersedes the original B2/B3/B4 ordering):** (1) revisit `stat_dist.json`
ZINB labeling — ≥ 13 cells are underdispersed and over-dispersed by NB, a plausible
compression source independent of the gate; (2) **CMP track (was B4) is now first-class**
— owns the 13 underdispersed cells (CMP handles var ≤ mean, which NB/ZINB cannot); (3)
**MZINB (B3)** owns the 10 inflated cells, consuming `_ZeroTruncatedNB` (a marginalized
ZINB estimates count + gate jointly, avoiding the derived-π `q ≥ NB(0)` constraint); (4)
**B2 routing wiring** seeded directly from `zinb_routing_diagnostics.py` (cmp / mzinb).
Research handoff `/tmp/track_b_rescope_research_prompt.md`; brief
`/tmp/researcher_track_b_rescope_response.md`. Regenerate with `poetry run
zinb-routing-diagnostics` if stale.

### Stage B1 — research verdict (distribution-family routing)

A claude.ai statistician reviewed the B1.2 rescope (brief
`/tmp/researcher_track_b_rescope_response.md`). It **confirms the two-cluster rescope**
and sharpens it into a routing protocol with two reality checks.

**Headline.** Route the 13 underdispersed cells to **mean-parameterized
Conway-Maxwell-Poisson (CMPμ)** and the 10 inflated cells to **marginalized ZINB
(MZINB)**; keep the derived-π hurdle as the *default* for borderline cells. The two new
families are **add-ons, not a wholesale replacement** of the shipped hurdle.

**1. Route on *conditional* dispersion, not marginal var/mean alone.** Marginal var/mean
is a necessary screen but not sufficient — (a) *mean-mixing* inflates it (pooling a star
3.5/g with a benchwarmer 0.4 looks overdispersed even if each is Poisson(μ_i); a good
feature set absorbs the role mixture → residual conditional dispersion → 1); (b)
floor/ceiling effects (PF capped at 6) create *conditional* underdispersion features
cannot remove. Safe protocol: (1) marginal (var/mean, ZI-index, score-test p) with
bootstrap CIs; (2) fit a baseline Poisson GBM, compute *conditional* dispersion from
randomized-quantile residuals [10]; (3) route only when marginal and conditional agree,
conditional overriding — **CMPμ** iff conditional var/mean < 0.90 AND marginal < 1.0;
**MZINB** iff conditional > 1.20 AND ZI-index CI excludes 0; (4) disagreement → keep the
derived-π hurdle; (5) tiny-sample cells → single-stage Poisson + sandwich SE. So the B2
routing config must be seeded from a *two-stage* diagnostic, not the marginal parquet
alone.

**2. Tightened dispersion bands.** B1.2's `< 1.3 → cmp` lumps equi-dispersed and mildly-
overdispersed. Literature-matched: ≤ 0.85 strong underdispersion (CMPμ) · 0.85–1.15
equi-dispersed (Poisson) · 1.15–1.5 mild overdispersion (NB) · ≥ 1.5 NB/hurdle/ZINB.
Adopt **1.15**, not 1.3, as the NB lower edge.

**3. Do NOT use the Vuong test** to pick ZINB-vs-NB — nested at γ=0 so the assumptions
fail [24]. Read the "Schwarz-corrected Vuong" column as descriptive only; use a boundary
LR / score test for the inflation decision.

**4. CMPμ is an engineering project, not a research project — but the ceiling is
modest.** Use Huang's [5] *mean*-parameterized CMP (log-link μ, orthogonal dispersion ν),
NOT canonical (λ,ν) where λ ≠ E[Y] — boost the quantity you price. Neither LightGBMLSS
nor XGBoostLSS ships CMP (verified to XGBoostLSS 0.6.1), so it is a custom distribution;
CMPμ's score/Hessian are tractable once Z(λ,ν) is tabulated (truncate at K≈64 in our
μ≈0.3–3 regime; well-conditioned info matrix from Huang & Rathouz [5]). Proven pattern: a
precomputed (μ,ν)→λ look-up grid with bilinear interpolation, refreshed once per market
[11][12]. Stabilize the ν gradient (clip log(ν)∈[−1, 2]). **Reality check:** projected
top-decile-MAE gain is **~3–8% relative, not 30%**. Worth building to amortize across 13
cells; **not** for one. When conditional var/mean ∈ [0.90, 1.10], plain **Poisson** is
right — CMPμ collapses to it, and forcing NB on equi-dispersed data destabilizes the MLE
[16].

**5. For the inflated cluster, prefer the *marginalized hurdle* [25] over MZINB — and
that is where `_ZeroTruncatedNB` belongs.** Two joint-fit families remove the derived-π
`q ≥ NB(0)` clip:
- **Marginalized hurdle [25] — recommended, smaller lift.** The literal joint-fit version
  of the two-stage hurdle we ship: a logistic for P(Y>0) and a **zero-truncated** NB on
  the positives, with the marginal mean reparameterized as the target. No `q ≥ NB(0)`
  constraint — the count component is genuinely defined on positives, the zero mass is its
  own free parameter, so the ZTNB that broke B1.1 *inside the derived-π decode* has a
  natural home here. (Design choices: [26].)
- **MZINB [27] — alternative, only for true structural-zero excess.** Parameterizes
  log(ν_i)=X_i′β (β = marginal mean = the line), back-computes μ_i = ν_i/(1−ψ_i), separate
  logistic gate; the count component is a *full* NB at all y (**NOT** zero-truncated).
  Pick it only when a subpopulation is plausibly at *structural* zero beyond the NB's own
  zeros.

**Either way, joint fitting *relocates* the gate-vs-count identifiability problem, it
does not solve it** — for fixed ν_i a larger ψ_i is offset by a larger μ_i. Three working
mitigations: (a) **separate covariate sets** (drive ψ off zero-risk-only variables:
minutes, availability, blowout/garbage-time; ν off count variables: ability, opponent
defense, pace); (b) **warm-start from the derived-π fit** (init the gate from the
classifier log-odds, β from log(ȳ_i), α from the NB MLE on positives); (c) **constrain α
weakly** (log-normal penalty around the MoM estimate, sd≈0.5). Preisser's ~100%
convergence is from their own simulations; the 2017 follow-up calls MZIP/MZINB "prone to
convergence problems to a degree shared by ZIP and ZINB" — budget real validation, keep
the derived-π hurdle as fallback.

**6. The routing will drift across seasons.** NBA 3PA/game rose ~1000% (2.8 → 32.0) from
1979 to 2018-19 with 3P% up 28%→36% [28] — shape, zero mass, and dispersion all move.
Treat single-season routing as stale: re-run the diagnostic each offseason, route on a
**hysteresis band** (flip a cell to CMPμ/MZINB only if outside [0.85, 1.30] var/mean in
the last *two* seasons), default to the hurdle inside the band, and force the more robust
NB-based hurdle if a cell's routing flips year-over-year.

**7. Cheap pre-checks before building anything.** (a) **Confirm it is a likelihood
problem at all:** refit suspect cells with a plain Poisson GBM. If top-decile compression
*persists*, the likelihood is part of the story; if it *vanishes*, the cause is the
**feature set** — fix features first, far cheaper. (b) **Audit for Vuong misuse:** any
routing decision using a Vuong p-value to choose ZINB-vs-NB is invalid (nested at γ=0) —
redo with Wilson-Einbeck or a boundary LR test. (c) **ZICMP is research territory:**
`mpcmp` does not yet ship a zero-inflated CMP, so a cell needing both inflation and
under-dispersion has no off-the-shelf family — flag it.

**Recommended sequencing (supersedes B2/B3/B4 below; effort is the researcher's
estimate):**
1. **Diagnostics infrastructure (1–2 wk)** — extend `zinb_routing_diagnostics.py` into a
   per-market panel (marginal mean, var/mean + CI, ZI-index + CI, Wilson-Einbeck,
   **conditional** var/mean from RQR of a baseline Poisson GBM, 4-season stability flag).
   Lock the protocol *before* refitting any family [29]. Run the §7a Poisson-GBM pre-check
   here.
2. **Marginalized hurdle for the 10 inflated cells (3–4 wk)** — smallest delta; consumes
   `_ZeroTruncatedNB`; warm-start + separate covariate sets per §5. **Gate: promote iff
   top-decile MAE improves ≥ 3% on a held-out season; else revert.**
3. **CMPμ for the 13 underdispersed cells (6–8 wk)** — promoted from B4 to first-class.
   **Gate: ship CMPμ only where conditional var/mean < 0.90 AND held-out top-decile MAE
   improves ≥ 2%; in 0.90–1.15 ship plain Poisson.** Ceiling ~3–8%, worth it amortized
   across 13 cells.
4. **Routing governance (ongoing)** — offseason refresh on a rolling 3-season window;
   hysteresis band; stability < 0.75 → derived-π hurdle; mid-season regime-flip detector →
   hurdle fallback. The B2 config gains `poisson`/`cmp`/`marg_hurdle`/`mzinb`.

**What would change this:** native CMPμ head in LightGBMLSS/XGBoostLSS → CMPμ drops to
~1 wk; held-out top-decile MAE within 2% of a Poisson oracle → skip CMPμ; > 3–4 cells
with conditional var/mean < 0.70 → CMPμ rises (only CMPμ or generalized/double-Poisson
[Efron 1986] fit those). The GPBoost fork (B3) stays an alternative *only* if post-step-1
residuals show systematic per-player offsets rather than dispersion misfit.

### Stage B1.5 — §7a pre-check verdict (likelihood vs features) — DONE

The §7(a) Poisson-GBM compression pre-check was run by the in-repo `research-analyst`
**before** paying for the family build (brief `/tmp/researcher_track_b_precheck.md`). It
covered **9 cells across all three leagues** — NBA FTM/STL/TOV/FG3M (the 4 mandatory: two
P2.B kills + one rep per cluster), WNBA STL/FG3M/FTM, NFL interceptions/rushing-tds —
each fit with a throwaway in-memory Poisson GBM (LightGBM `objective="poisson"`, seed
1729, **no pickle saved**) and scored against the production NB/ZINB baseline via
`compression_eval` on identical held-out rows.

**Verdict: FEATURES, not likelihood — DEFER/CANCEL the 9–12-week CMPμ +
marginalized-hurdle/MZINB build as Track B's next step.** The likelihood signature
("Poisson tracks the top decile while NB compresses it") did not appear on any cell.

| cell | cluster | marg var/mean | cond RQR var | top-MAE Δ (Pois vs NB) | §7a verdict |
|---|---|---|---|---|---|
| NBA FTM | inflated | 2.00 | 1.54 | +3.3% | KILL → features |
| NBA STL | underdisp | 1.17 | 1.08 | +3.5% | KILL → features |
| NBA TOV | underdisp | 1.16 | 1.04 | +1.4% | KILL → features |
| NBA FG3M | inflated | 1.57 | 1.17 | +6.3% | KILL → **mean-bias** (SHIP is bias, not family) |
| WNBA STL | underdisp | 0.99 | 1.00 | −0.9% | KILL → features |
| WNBA FG3M | inflated | 1.58 | 1.12 | −0.6% | KILL → features |
| WNBA FTM | inflated | 1.92 | 1.46 | +0.9% | KILL → features |
| NFL interceptions | inflated | 1.60 | 1.04 | −1.1% | KILL → features (RQR≈1: low-count, not inflated cond.) |
| NFL rushing-tds | underdisp | 0.96 | 0.96 | +10.3% | KILL → **mean-bias** (SHIP is bias, not family) |

**1. Top-decile compression is distribution-family-INVARIANT.** It persists under a
Poisson mean head, so it cannot be a likelihood problem: Poisson top-decile CR
`std(pred)/std(actual)` 0.16–0.35 vs production NB/ZINB 0.12–0.37 — indistinguishable,
both severely compressed; a model with **no over-dispersion freedom at all** compresses
the high-mean tail just as hard. Textbook ensemble-tree dynamic-range bias (extreme
values estimated by leaf-averages of neighbours; persists regardless of sample size or
family) — [3], [30]; mechanism = the regularized leaf-average itself [4]. Neither CMPμ
(re-parameterizes dispersion, leaves the boosted mean unchanged) nor MZINB/hurdle (re-
parameterizes the zero gate) touches the mean head's leaf-averaging. **Load-bearing —
gates the entire family build.** Independently predicted by [8] (point ≡ probabilistic in
Poisson; a global dispersion parameter beat covariate-specific dispersion in their
benchmark).

**2. The 2 cells passing the ≥5% gate are upward mean-bias, not dispersion, and not
Track B's symptom.** NBA FG3M (+6.3%) and NFL rushing-tds (+10.3%) "SHIP" but show
**uniform upward bias** — the production model over-predicts, worst at the *bottom*
decile (FG3M predicts 0.68 threes where actual is 0.20), the inverse of Track B's
under-prediction-of-stars symptom. A trivial bias re-centering recovers 41% (FG3M) / 47%
(rushing-tds) of the gain with **no family change**; the remainder is better mean-fit
from a fresh GBM, not orthogonal dispersion. Under the calibration ship gate (Universal
threshold condition 4, added in response to this finding), both cells **fail Gate 1** —
the low-volume over-prediction their wins rely on is exactly what that gate blocks.

**3. No CMPμ candidate among the NBA/WNBA cells.** Conditional Dunn–Smyth RQR variance
[10] (power for count GOF: Feng et al. 2020 [31]) collapses marginal var/mean toward 1
once the ~280-col feature set conditions the mean: STL 1.17→1.08, TOV 1.16→1.04, WNBA STL
0.99→1.00 — **equi-dispersed**, failing the `CMPμ iff conditional < 0.90 AND marginal <
1.0` gate. At ν≈1 there is nothing for CMPμ to learn; its ~3–8% ceiling does not survive
conditional RQR ≈ 1. The inflated cells keep genuine conditional overdispersion (FTM
1.46–1.54, FG3M 1.12–1.17 → NB is right there) but a Poisson/NB mean swap moves top-decile
MAE < 5% on every inflated cell except the bias-driven FG3M, so the
marginalized-hurdle/MZINB build is **not** justified by this pre-check either.

**Pivot — Track B's next step is a ~1–2 week feature/bias track, not the family build
(supersedes the §7-verdict B2/B3 sequencing):** (1) **post-hoc mean-bias / dynamic-range
correction** — cheapest of the six [3] methods (ROE = regress-observed-on-estimated, or
EDM = empirical-distribution matching), fit on validation, applied before `fused_loc`; a
post-hoc calibration object (one pickle key + apply step + round-trip test); **days**;
directly attacks the family-invariant compression, captures ~half the FG3M/rushing-tds
gain by construction; (2) the leakage-safe target-encoded player features already staged
as B2 (`groupby(player_id).expanding().mean().shift(1)` for stat and stat×opponent);
**days**; (3) opponent-defense interaction + garbage-time/blowout flag — the FG3M
bottom-decile overshoot (low-volume players predicted ~3.4× actual) points at not
conditioning on *whether the player attempts threes at all*, far cheaper than a
structural-zero gate.

**Cost the pre-check gates:** ~1–2 weeks (feature/bias) vs ~9–12 weeks (CMPμ +
marginalized hurdle + the largest inference-path change in the plan + the Open-question-#2
MZINB-gradient-debugging risk). **Re-entry condition:** build the family only on a cell
that **still kills after** the cheap fixes — specifically conditional RQR variance < 0.70
AND the Poisson GBM tracking the top decile while NB compresses it. None of the 9 cells
does. Caveat: single-snapshot, single-fit pre-check (untuned Poisson, one HP set); a
*tuned* Poisson would only strengthen the read, and the brier_skill gate did not fire
(Open question #12). The 13-cell `cmp` label and 10-cell inflated label from B1.2 are both
unsupported as a *family-build* trigger by this evidence (Open question #11).

### Stage B1.6 — Feature/bias track (the §7a pivot — Track B's operative next step, ~1–2 weeks)

Per the B1.5 §7a verdict the top-decile compression is a **mean-head + feature** problem,
not a likelihood problem, so this stage attacks those two layers cheapest-first and
**ships per market the instant a cell clears Gate 1** (per "Ship incrementally"). It
**supersedes the family build (B2 routing + B3 fork) as the next step**; the family build
is deferred behind the re-entry condition at the end. Lock the eval protocol before
refitting [29].

| Workstream | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **1. Post-hoc mean-bias / dynamic-range correction** (NEW — the only piece targeting the actual mechanism) | [3], ROE / EDM | days | Attacks the family-invariant leaf-averaging compression *directly* (B1.5 Finding 1). Fit `ŷ_corrected = f(ŷ)` on validation (ROE or EDM), apply to the predicted mean **before** `fused_loc`. Symmetric: pulls bottom-decile over-prediction *down*, serving the new Gate-1/Gate-2 bias gates. Recovers ~half the FG3M/rushing-tds gain by construction. | New pickle key (e.g. `bias_correction`) per Pickle-schema discipline; apply step in `model_prob` before [helpers/distributions.py:314](../src/sportstradamus/helpers/distributions.py#L314) `fused_loc`/`get_ev`; fit on validation only; one round-trip test. |
| **2. Leakage-safe target-encoded player features** (pulled forward from B2) | Tier 2 (ZINB) #5 | days | `groupby(player_id).expanding().mean().shift(1)` for stat and stat×opponent. League-agnostic; ships regardless of family. | New columns in [stats/base.py:597](../src/sportstradamus/stats/base.py#L597) `get_stats`; same leakage audit as MeanYr/Mean10 ([test_meanyr_mean10_leakage.py](../tests/test_meanyr_mean10_leakage.py)). |
| **3. Opponent-defense interaction + garbage-time/blowout flag** | B1.5 Finding 2 | days–1 wk | The named "deficient feature set" items: opponent-defense interaction (player stat × opponent defensive profile from `profile_market`); a blowout/garbage-time flag (projected point-differential bucket / minutes-at-risk). Addresses the FG3M low-volume overshoot. | New feature columns + the [get_stats](../src/sportstradamus/stats/base.py#L597) inference mirror; leakage-safe, same audit. |

**Sequencing (cheapest, highest-leverage first): (1) → (2) → (3).** Do the post-hoc
correction first. Re-run `compression_eval` per cell after each workstream and **promote
every cell that clears Gate 1's five conditions to the 14-day live soak immediately.**

Decision points: after (1), any cell clearing Gate 1 (incl. the new bottom-quartile
condition) ships — expect FG3M/rushing-tds to flip from "bias-driven nominal SHIP" to a
*legitimate* SHIP. After (1)+(2)+(3), cells that clear ship; cells that still kill carry
to the re-entry check. Stop-the-track: if a cell graduates via Gate 2, stop track work on
it. Inference: workstream (1) is the cleanest possible change (one pickle key + one apply
step + round-trip); (2)/(3) need the `get_stats` mirror, leakage-tested like MeanYr/Mean10.
**Validate (1) does not worsen `brier_skill_score`** — a post-hoc mean shift must not
distort the probabilistic shape (Open question #12: check CRPS/log-score + brier_skill, not
MAE alone).

**Re-entry to the family build (supersedes B2/B3 sequencing).** Build CMPμ or the
marginalized hurdle **only** on a cell that *still* kills after B1.6 — specifically
conditional Dunn–Smyth RQR variance **< 0.70** AND the Poisson GBM tracking the top decile
while NB compresses it. None of the 9 §7a cells qualifies today. The diagnostic that
re-checks this is the conditional-RQR pass from Open question #11 (run it on the 11
untested `cmp`-labelled cells before any CMPμ build); only then do B2's routing wiring +
the B3 fork apply, scoped to that cell.

### Stage B1.6 — breadth research verdict (2026-05-22, research-analyst)

The `research-analyst` reviewed the per-league baseline-breadth gap (NBA +3, WNBA +4,
NFL +2) and wrote a cited statistician's brief (`/tmp/researcher_breadth_75.md`,
References [43]–[48]). It **confirms the B1.5 §7a pivot and ranks the breadth levers**.
The actionable summary lives in the plan's "Path to 75%" + roadmap; this block is the
detail.

**Bottleneck restated.** Every failing cell is a **low-mean count market** and fails
Tier-0 on the **absolute bias gates (conditions 4b/4c), not the BSS floor** (the BSS
floor excludes only ~1 extra cell/league beyond the bias failures — NBA PF, NFL
interceptions). Two symptoms: bottom-quartile **over**-prediction (gate 4b, one-sided —
e.g. FG3M bottom decile 0.68 predicted vs 0.20 actual) and top-decile **under**-prediction
(gate 4c — STL −0.59, FTM −1.16, TOV −0.39, PF −0.46 on the healthy joint cells).
Family swaps are **dead per B1.5 §7a**: leaf-averaging compression is family-invariant
(Poisson top-decile CR 0.16–0.35 ≈ NB/ZINB 0.12–0.37), so CMPμ/MZINB/hurdle re-parameterize
dispersion or the zero gate and **leave the boosted mean head untouched** ([3][4][30][8]).
Only two layers touch the mean head: **post-hoc correction of the prediction** and
**features that hand the tree the per-player level it averages away**.

**Ranked methods (effect on the two gated bands).**

1. **Post-hoc per-decile / isotonic / affine mean-bias correction — RANK 1.** The *only*
   method that moves **both** gated bands by construction (a monotone map pulls bottom-band
   over-prediction down and top-band under-prediction up at once). Prefer **isotonic** where
   the decile miscalibration is **curved** (over-low/under-high — the compression shape);
   plain affine/ROE [3] captures the uniform part (≈ the §7a "41%/47% recovered" finding).
   **Per-decile multicalibration is the formal frame** (Globus-Harris et al., arXiv:2301.13767
   [44] — per-subgroup squared-error de-biasing *is* a boosting procedure) but a **fallback,
   not the lead**: Hansen et al., arXiv:2406.06487 [45], find trained GBDTs are often already
   near-multicalibrated, so budget per-decile only for a band just outside the bound.
2. **Leakage-safe expanding-mean / EB-shrunk player feature — RANK 2** (the deferred P5).
   `groupby(player_id).expanding().mean().shift(1)` for stat and stat × opponent as a
   *feature*; optionally a James-Stein / empirical-Bayes-shrunk variant (Efron–Morris [47];
   shrinkage weight `K = σ²_within/σ²_between`). Regularized target encoding beats one-hot and
   naive mean encoding for high-cardinality `player_id` (Pargent et al. 2022, DOI
   10.1007/s00180-022-01207-6 [43]); unregularized encoding overfits.
3. **Opponent-defense × player + blowout/garbage-time features — RANK 3.** Narrowest, most
   build cost; targets the FG3M low-volume overshoot (does this player attempt the stat at
   all tonight). Mostly helps the bottom band.
4. **BSS-floor gate-tuning toward −0.02 — mop-up only.** A policy lever for the ~1
   bias-passes/BSS-fails cell per league (NBA PF, NFL interceptions); does **not** address
   the binding bias-gate failures. Widen the constant, never per-cell; flag widened cells.

**Per-league nearest-miss triage.**

- **NBA (+3):** target **TOV, BLST, OREB, PF** (STL, FG3M as backups) — near-equi-dispersed
  conditionally, so they want post-hoc correction, not family.
- **WNBA (+4):** target **TOV, BLST, OREB, STL** — same family as the NBA cheap wins, but
  **re-validate, never assume** (half the games: ~40/season makes isotonic tails and the
  expanding-mean feature noisier; fall back to affine ROE / strongly-shrunk player mean).
- **NFL (+2):** target **rushing-tds + receiving-tds** via **affine ROE — not isotonic /
  per-decile** (too few positive events at those means: interceptions 0.5, TD zero-rates
  0.78–0.92; per-bin error is worst in low-base-rate groups [48]). Keep **rushing-yards,
  qb-tds, sacks-taken** on the Track-A T11 (per-position) bench.

**Ordered experiment plan** (ship each cell to Gate-2 the instant it clears Tier-0):

- **Step 0 — Tier-0 absolute-only `compression_eval.verdict()` mode (hard precondition).**
  No baseline-only path exists today (`compression_eval.py:337-426` encodes only the Tier-1
  pairwise path); the +3/+4/+2 counts are unmeasurable until it does. Run the read-only
  per-cell 4b/4c-slack + BSS triage audit on this mode first.
- **Step 1 — post-hoc correction** (affine first, then isotonic), fit on validation, applied
  before `fused_loc`; re-validate BSS/CRPS does not regress.
- **Step 2 — expanding-mean / EB-shrunk player feature**, leakage-audited like MeanYr.
- **Step 3 — opponent-defense × player + blowout features** for cells still failing.
- **Step 4 — BSS-floor gate-tuning** for the bias-passes/BSS-fails stragglers.

### Stage B2 — Routing + orthogonal feature engineering (2 weeks, in parallel)

> **Status after the B1.5 §7a verdict:** this stage is now **downstream of B1.6 and gated
> by the re-entry condition** — the family build (CMPμ / marginalized-hurdle, B3) and the
> routing wiring below are pursued **per cell, only for cells that still kill after B1.6.**
> The "leakage-safe target-encoded player features" row is **pulled forward into B1.6**;
> the routing-config wiring stays here for when a re-entry cell needs a non-default family.

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **Per-league × per-market routing wiring** | Tier 1 (ZINB) #2 | days | Implement the routing from B1's table. `data/zinb_mode_per_market.json` schema: `{LEAGUE: {market: "joint"\|"hurdle"\|"plain_nb"\|"poisson"\|"cmp"\|"marg_hurdle"\|"mzinb"}}` (keyed by both — NBA STL and WNBA STL may route differently). **Seeded by the two-stage routing diagnostic (marginal `zinb_routing_diagnostics.py` + a Poisson-GBM conditional-dispersion residual pass) under the B1 verdict's hysteresis band — not the marginal parquet alone.** Per-cell lookup in [pipeline.py:1869](../src/sportstradamus/training/pipeline.py#L1869) `_step_select_distribution`; default `"joint"` keeps legacy byte-identical. | `pipeline.py` per-cell dispatch; new JSON under `data/`. |
| **Leakage-safe target-encoded player features** | Tier 2 (ZINB) #5; was P5 | days | `groupby(player_id).expanding().mean().shift(1)` for stat and stat×opponent. Orthogonal to architectural choice; ships regardless of B3 winner. League-agnostic. | New columns in [get_stats](../src/sportstradamus/stats/base.py#L597); same leakage audit as MeanYr/Mean10. |

Decision points: 8/8 ship under routing + encoded features → hold off on B3 unless ROI
needed elsewhere. <8/8 → proceed to B3 on the markets that still kill. Stop-the-track:
after the routing config runs ≥ 14 days, expect remaining ZINB cells to graduate; do not
start B3 (a 4–6 wk novel architecture or non-trivial dependency add) unless ≥ 3 cells have
*not* graduated AND their offline residuals point clearly at MZINB or GPBoost. Inference:
per-cell routing config is the cleanest Track-B change — `model_prob` already dispatches on
`is_hurdle`, so the config just changes which pickle loads per cell (pickle records
`zinb_mode`); target-encoded features need the same `get_stats` mirror as MeanYr/Mean10.

### Stage B3 — Strategic fork (4–6 weeks, pick ONE — not both)

The canonical decision point of the ZINB track. Running MZINB and GPBoost in parallel
doubles cost without obvious gain. The criterion is **residual structure after B2**; if
ambiguous, dispatch the `research-analyst` agent to adjudicate before committing 4–6 weeks.

| Option | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **MZINB head in LightGBMLSS** | Tier 2 (ZINB) #4 | 2–4 weeks | Reparameterize so the marginal mean ν = E[Y] is boosted directly; latent NB conditional mean μ = ν/(1−ψ) reconstructed. Three heads: logit(ψ), log(ν), log(α). The boosted ν IS the quantity the downstream pipeline consumes. Removes the `q ≥ NB(0)` clip **by construction**, but *relocates* (does not eliminate) the gate-vs-count trade-off — see **B1 research verdict §5**, which recommends the **marginalized hurdle [25]** as the smaller-lift joint fit (the natural home for `_ZeroTruncatedNB`), with MZINB reserved for true structural-zero excess. **No published GBDT implementation — novel.** | New `MZINB` class alongside the distributions in [skew_normal.py](../src/sportstradamus/skew_normal.py) or [helpers/distributions.py](../src/sportstradamus/helpers/distributions.py); foundational [27]; Mutiso et al. 2024 [32] (Pólya-gamma) as a likelihood-structure reference. |
| **GPBoost with NegBin likelihood + player random intercept** | Tier 2 (ZINB) #3; was P10 | 1–2 weeks | NegBin is a native GPBoost likelihood, so LSS-flexibility loss is smaller on counts than on SkewNormal. Sigrist's benchmarks (~10pp gap vs LightGBM-Cat, ~93pp vs naive numeric ID) transfer to NB. The earlier GPBoost prototype failed on SkewNormal FGA, not on counts. | New GPBoost dependency (user pre-approved for a phase that needs it); custom training path branched off [pipeline.py](../src/sportstradamus/training/pipeline.py). |

**Decision criterion (researcher-specified):** **Choose GPBoost** if residuals plotted by
`player_id` (bootstrap CIs) show systematic per-player offsets distinguishable from zero
(missing player effects); trade-off: lose LSS distributional flexibility for future exotic
distributions. **Choose MZINB** if residuals are tail/shape-driven (heavy-tailed
within-player, not location-shifted) — gate-vs-count identifiability, not missing player
effects; trade-off: novel implementation, ~4 weeks debugging, no GBDT precedent. In either
case also implement **per-parameter Optuna search** (Tier 3 #6, days, see B4) for a fair
tuned baseline.

Decision points: residuals ambiguous → 2-week MZINB spike first (cheaper exit), GPBoost
fallback. B4 per-parameter Optuna alone eliminating the deterministic blowups across all
SHIP markets → MZINB much weaker (identifiability was hyperparameter-induced); reroute to
GPBoost-only. Stop-the-track: MZINB's "no GBDT precedent" risk is only worth eating if (a)
live data shows specific cells still failing after B1+B2 AND (b) residuals clearly favor
MZINB over the cheaper GPBoost; if live performance is acceptable but offline shows a gap,
file MZINB as future research. Inference: both options change the parameter contract —
MZINB returns `(ν, ψ, α)` (new ZINB-MZINB decode in `model_prob.py` deriving `Model EV =
ν` and reconstructing `Model Gate` from `ψ`; `dist` stays `"ZINB"`, new `mzinb_mode` key
with legacy default `"hurdle"`); GPBoost is a different model class (`is_gpboost` dispatch,
load-path branch, GPBoost-specific live-path test). **Inference-engineering cost alone
favors one fork over the other when residuals are ambiguous** — GPBoost's class change is
larger than MZINB's parameter-contract change.

### Stage B4 — Tuning, polish, specialized fixes (optional)

> **Deferred — post-break speculative tail; tracked under [roadmap](sportstradamus_roadmap_v2.md) Phase 6.** Body kept here (home of record); optional, not active until B2/B3 leave a gap *and* breadth is met.

| Method | Source | Cost | Direct effect | Implementation site |
|---|---|---|---|---|
| **Per-parameter Optuna search** | Tier 3 (ZINB) #6 | days | Separate `learning_rate`/`n_estimators` for gate vs NB heads inside the existing Optuna sweep. cyc-GBM-inspired without porting. May resolve the deterministic-30-round blowups without architectural change. [8a] (Daub "balanced step length") justifies. | Extend the Optuna search-space dict in [pipeline.py:348-368](../src/sportstradamus/training/pipeline.py#L348). |
| **CMPμ head — PROMOTED to first-class (owns the 13 underdispersed cells)** | was Tier 4 → Tier 1 (ZINB) #8 | 6–8 weeks | Per the B1 verdict, no longer optional STL polish. Mean-parameterized CMP [5] (canonical λ ≠ E[Y], unusable for line-pricing). Custom distribution. Series truncation K≈64; precomputed (μ,ν)→λ grid [11]; init ν from MoM; clip log(ν)∈[−1,2]. Ceiling ~3–8% amortized across 13 cells. Plain **Poisson** when conditional var/mean ∈ [0.90, 1.10]. | New PyTorch distribution; pre-computed Z(λ,ν) + (μ,ν)→λ lookup at module load. Reference: CMPBoost [12]. |
| **Reduced regularization on location parameter** | Tier 3 (ZINB) #7; was P6 | hours | Larger `num_leaves`, smaller `min_data_in_leaf`, deeper `max_depth`. Marginal. Try only after Tier 1–2. | Optuna search-space dict. |
| **MERF-style iteration with LightGBMLSS** | Tier 4 (ZINB) #9; was P9/P10 fallback | 2–3 weeks | Alternating fit-residual / re-estimate-shrunken-per-player-baseline loop. Reserve as fallback if B3 (both MZINB and GPBoost) prove infeasible. | New module under `src/sportstradamus/training/`. |
| **Quantile / expectile heads alongside the ZINB** | Tier 5 (ZINB) #11; was P7-equiv | days | **Different use case** — DFS ceiling and over bets, not the bias fix. Add alongside, not instead of, Tier 1–2. Sluijterman et al. 2025 arctan pinball [33] is a drop-in for standard pinball with better-calibrated extremes. | New quantile head pickled alongside the ZINB. |
| **Isotonic post-hoc calibration on ZINB-mean** | Tier 5 (ZINB) #12; was P8 | hours | Polish. Mops up residual average bias after architectural fixes. Don't lead with it. | `IsotonicRegression(out_of_bounds="clip")`. |
| **Sample reweighting on high-scoring games** | Tier 5 (ZINB) #13; was P9 | hours | Last resort. Increases variance in the upweighted region. | LightGBM `sample_weight`. |

**What NOT to do (researcher-specified):** do not lead with quantile/expectile heads as a
bias fix (they change which point is predicted, not the bias); do not lead with isotonic
(polish); do not port to cyc-GBM/CyclicBoosting before trying per-parameter Optuna inside
the existing stack; do not pursue per-minute decomposition on count markets in parallel
with Track A (leverage materially lower on counts); do not commit to both MZINB and
GPBoost.

Decision points: per-parameter Optuna alone hits the threshold → done; don't proceed to
CMP/MERF. STL var/mean < 1.0 → CMP becomes a primary fix, move to B3. Everything in B4
fails → loop back to a Track-A-style live-path audit (the FTM/STL kill may originate
downstream in [model_prob.py](../src/sportstradamus/prediction/model_prob.py),
OVERCONFIDENCE_INVESTIGATION §3.4). Stop-the-track: items are individually cheap but
cumulatively multiple weeks; if per-parameter Optuna alone graduates the holdout cells,
do not run the other 5. Inference: per-parameter Optuna, reduced regularization, sample
reweighting are training-only; CMP head is a new distribution (`dist == "CMP"` decode +
`distributions.py` consumers, live-path test); isotonic on ZINB-mean is a post-hoc object
(`isotonic_zinb` key + apply, round-trip); quantile heads pickle alongside with a new
`quantile_heads` key + decode path; MERF changes the architecture (`is_merf` dispatch,
pickle schema, live-path test).

---

## Open questions (researcher-flagged, unresolved)

1. **Feature-predictive-power asymmetry between zero-vs-count splits at the market level
   is not in the published literature.** The TOV-vs-STL puzzle (similar surface stats,
   opposite hurdle outcomes) is not directly addressed; closest analogue is Feng (2021)
   [34] (NB outperforms ZINB at ~20% zero rates; STL at 48% is well above). Possibly a
   genuine domain-specific phenomenon publishable in its own right — capture data for a
   write-up after Stage B. (ZINB caveat #5.)
2. **No GBDT precedent for MZINB.** Expect 3–5 cycles debugging the gradient for log(ν) →
   μ_implied = ν/(1−ψ) before it trains stably. Boosted MZINB resolving the
   deterministic-mode blowups would itself be a novel contribution. (ZINB caveat #1.)
3. **MEGB's headline 35–76% MSE improvement is from simulations**, not a real high-n
   low-p panel matching the NBA regime. Transfer is the bet. (SkewNormal caveat.)
4. **DEGPD / ZIDEGPD count distributions** [16a] are very new with no production-grade
   Python implementation. Deliberately not staged into Track B — too speculative.
5. **CMP normalizing-constant lookup-table** cost is non-trivial. Worth it only if Stage
   B1 var/mean strongly suggests under-dispersion for at least one market.
6. **CatBoost ordered TS unbiasedness was proven for log-loss / squared-error**; does not
   directly extend to a distributional SkewNormal NLL. Validate empirically. (SkewNormal
   caveat.)
7. **The Chevalier & Côté (2025) [8] benchmark warns probabilistic GBM is not uniformly
   better than point-prediction GBM on NB-class targets.** Validate any architectural
   switch (MZINB, cyc-GBM, CMP) against the LightGBMLSS-ZINB baseline on probabilistic
   metrics (CRPS, log-score) **and** the downstream MAE, not just one.
8. **None of these solves "high-volume players get under-predicted" if it is fundamentally
   a player-effect issue.** Best long-term architecture probably combines (a) MZINB
   likelihood, (b) per-player random intercept à la GPBoost, (c) per-parameter early
   stopping à la cyc-GBM — each an independent A/B.
9. **The T5 recomposition-variance penalty must be checked at Gate 1 if T5 is ever
   trialed.** A1.5 measured **+27% within-player-season CV inflation** from recomposing
   FGA × (PTS/FGA) vs modeling NBA PTS directly [1]. The +27% is an identity-decomposition
   floor, not a trained-model A/B — the *direction* (inflation) is robust (structural
   identity), the *magnitude* is an estimate. Any T5 trial (NFL only) must verify the
   recomposed predictive scale is not inflated vs the direct SkewNormal baseline on
   **both** metric families (CRPS/log-score and top-decile MAE per #7), not MAE alone.
   (A1.5 verdict.)
10. **Marginal vs conditional, for the NFL per-factor route.** Every factor ICC in A1.5 is
    *unconditional*; the ~280-col matrix already absorbs much between-player volume
    identity, so the marginal volume/efficiency split overstates the residual signal.
    Before A2 commits to the NFL per-factor route, run a one-pass diagnostic — does a
    SkewNormal-GBM on each NFL factor beat the whole-stat SkewNormal-GBM on held-out
    top-decile MAE? Flagged, not blocking. (A1.5 verdict.)
11. **The marginal "underdispersed (CMP)" cluster label is mostly a mean-mixing artifact
    for NBA/WNBA.** 13 cells routed to `cmp` on marginal var/mean ≤ 1.3 (B1.2); the two
    reps Poisson-GBM-checked in B1.5 (STL, TOV) are conditionally **equi-dispersed** (RQR
    ≈ 1.0–1.08). The other 11 (NBA BLST/OREB/PF, WNBA BLK/STL/BLST/TOV/OREB, NFL
    tds/receiving-tds) were not individually checked — each needs the same conditional-RQR
    pass before any CMPμ build; strong prior most land at RQR ≈ 1 (plain Poisson). (B1.5
    §7a verdict.)
12. **The brier_skill_score third gate did not fire in the B1.5 pre-check** — the test CSVs
    carry no `Odds` column and the candidate left `P` unchanged (only `EV` swapped), so the
    gate reduced to the two MAE conditions. Correct for a mean-head pre-check, but it means
    the pre-check does not speak to probabilistic calibration: any eventual family **or**
    bias-correction switch must still be validated on CRPS/log-score **and** brier_skill on
    a report run that carries the book columns. (Extends #7; B1.5 verdict.)
13. **Post-hoc mean correction can distort the probabilistic shape** — point calibration ≠
    distributional calibration (Venn-Abers, Cortes-Gomez et al. arXiv:2502.05676 [46]).
    Validate BSS/CRPS at every B1.6 step, not the corrected mean alone. (Extends #12;
    B1.6 breadth verdict.)
14. **The expanding-mean player feature may add little over the existing `MeanYr`** — the
    §7a Poisson GBM already carried the full feature matrix and still compressed. Measure the
    marginal lift; its novelty is causal within-season updating + the stat × opponent cross,
    not the per-player level itself. (B1.6 breadth verdict.)

---

## Decisions & trade-offs

Every deferral / prioritization / kill decision scattered through the body, consolidated.
Each is one decision + one rationale; the detail lives where cited.

| Decision | Rationale | Detail in |
|---|---|---|
| **First-ship gate reframed from "≥ 5% top-decile-MAE improvement" to absolute breadth (Tier 0).** | Top-decile compression is family-invariant; the ≥ 5% bar baselined only 2/21 NBA markets. Breadth (75%/league) makes money on more markets. | North Star |
| **Default target strategy stays `ratio_meanyr`.** | P1 KILL: centered-target variants lost path-wide (FGA-only ship); the SkewNormal level bias is not the dominant compression cause. | P1 / Track A |
| **Default ZINB mode stays `joint` (hurdle not made default).** | P2.B shipped infrastructure + a 6/8 verdict, but per-market routing (FTM/STL stay joint, rest → hurdle) is a follow-up, not a default flip. | P2.B |
| **P2.A `init_score` baseline — DEAD.** | LightGBMLSS produces byte-identical predictions to plain NegBin; the bias signature does not move; FG3M count-branch bias is already −0.013. | P2.A |
| **ZTNB Stage-2 wire-in — REVERTED (kept as unwired block).** | Correct in isolation but incompatible with the frozen derived-π decode (NB(0) > q on 65% of FG3M rows → identity diff 0.136); would regress the 6 P2.B SHIP markets. `_ZeroTruncatedNB` reserved for the marginalized hurdle (B3). | B1.1 |
| **T5 (multiplicative factorization) — KILLED as a wholesale architecture.** | Goodman variance-of-products: recomposing FGA × (PTS/FGA) inflates within-player-season CV +27%; the low-ICC half is *efficiency* (served directly by T3); volume-clause fails for NBA. T5 survives only as a narrow NFL per-factor *route*. | A1.5 |
| **A2 primary build = T3 tail head (not T5).** | T3 captures the realizable lift for ~half the cost and a fraction of the inference risk. | A1.5 / A2 |
| **CMPμ / marginalized-hurdle / MZINB family build — DEFERRED behind a re-entry condition.** | §7a pre-check: compression persists under a plain Poisson head (family-invariant), and conditional RQR collapses the "underdispersed" cells to ≈ 1 — no CMPμ candidate among NBA/WNBA. Build only on a cell with conditional RQR < 0.70 that still kills after B1.6. None of the 9 cells qualifies. | B1.5 §7a |
| **Track B's operative next step = B1.6 feature/bias track (~1–2 wk), not the family build (~9–12 wk).** | Cheapest fix that targets the actual (mean-head + feature) mechanism; ships per cell on Gate 1 clear. | B1.5 / B1.6 |
| **The two §7a "SHIP" cells (FG3M, rushing-tds) FAIL Gate 1 under the new calibration condition.** | Their top-decile wins came from over-predicting low-volume players (FG3M ~3.4× actual); Universal threshold condition 4 (added in response) blocks exactly that. | B1.5 §7a |
| **`stat_dist.json` ZINB labeling is wrong for ≥ 13 cells.** | Routing diagnostics: 0/23 route to `hurdle_nb_ztnb`; 13 underdispersed (→ CMP/Poisson), 10 inflated (→ MZINB/marg-hurdle). Revisit labeling. | B1.2 |
| **For the inflated cluster, prefer the marginalized hurdle [25] over MZINB.** | Smaller delta from what we ship; the ZTNB has a natural home there (no `q ≥ NB(0)` clip); MZINB reserved for true structural-zero excess. | B1 verdict §5 |
| **Hurdle / borderline cells default to the derived-π hurdle.** | Joint fitting *relocates* the gate-vs-count identifiability problem; keep the proven hurdle as the least-to-lose fallback in the disagreement zone. | B1 verdict §1, §5 |
| **Do NOT use the Vuong test for ZINB-vs-NB.** | Nested at γ=0; assumptions fail. Use a boundary LR / Wilson-Einbeck score test; read the Schwarz-corrected Vuong column as descriptive only. | B1 verdict §3 |
| **Stage A4 and Stage B4 — DEFERRED to roadmap Phase 6 (post-break tail).** | Speculative / optional; only if A2/A3 (resp. B2/B3) leave a gap *and* breadth is met. Bodies kept here as home of record. | Diminishing returns; A4/B4 |
| **The "DFS-industry consensus" funnel is convention, not ship evidence.** | Practitioner lore with no peer-reviewed tail-bias validation; the only primary sources on products of separately-modeled components ([1], [7]) point the other way for the tail. | A1.5 Finding 3 |
| **Diminishing-returns pre/post-break split drives the roadmap phase split.** | Pre-break (active): breadth → A2/A3/B2/B3. Post-break (deferred): A4/B4/long-shots. Stop a track when a cell is good live; the plan is a backlog, not a queue. | Diminishing returns |
| **Track-A drops: P3 → folded into T5; P5 → T2 CatBoost ordered TS; P10 GPBoost retry → T4 MEGB/GBMixed.** | T5 is the four-stage version DFS shops actually use; ordered TS dominates expanding-mean encoding [17]; MEGB's EM-pseudo-residual fixes the documented GPBoost bias. (T5 itself later killed — see above.) | "What is dropped from Track A" |
| **NBA/WNBA AST are SkewNormal, not count markets.** | Their Tier-0 failure is the P1 family-wide SkewNormal compression; take the Track-A treatment (post-hoc + per-player feature). Do NOT group with FG3M/STL count-family reasoning (CMP/Poisson, zero-gate). | B1.6 breadth verdict / Track A |
| **Post-hoc mean-bias correction is the RANK-1 breadth lever.** | The only method that moves both gated bands (bottom-quartile over-prediction, top-decile under-prediction) by construction; ranked above the feature work. | B1.6 breadth verdict |
| **NFL low-mean cells use affine ROE, not isotonic / per-decile.** | Too few positive events (interceptions 0.5, TD zero-rates 0.78–0.92); per-bin error worst in low-base-rate groups [48]. Target rushing-tds + receiving-tds for the +2. | B1.6 breadth verdict |
| **Breadth is a 3-rung ladder, not a one-shot target.** | Rung 1: 75%/league at the current ~30%-of-band absolute-gate bar (floor 0.10). Rung 2: tighten the absolute gates to 20% of band mean (floor 0.10 unchanged), re-reach 75% at that stricter bar (a fresh research agent is dispatched then). Rung 3: push to >90% coverage at the tightened bar. Ratchet quality and breadth alternately rather than over-fitting one. | North Star |

## References

Inline citations in the body use the marker `[n]` (or author-year where a name reads more
naturally); the full list, including the original bibliography table, is here. DOIs/arXiv
IDs preserved so the plan is self-contained. Grepped from the file for `DOI `, `doi:`,
`arXiv:`, `SSRN`, `Working Paper`, and author-year patterns to collect all.

1. **Goodman**, *On the Exact Variance of Products*, JASA 55(292):708–713, 1960, DOI
   10.1080/01621459.1960.10483369; K-variable generalization JASA 57(297):54–60, 1962, DOI
   10.1080/01621459.1962.10482151. (Product-variance algebra; the +27% CV result.)
2. **Franks, D'Amour, Cervone & Bornn**, *J. Quantitative Analysis in Sports*
   12(4):151–165, 2016, DOI 10.1515/jqas-2016-0098, arXiv:1609.09830. (Meta-analytics /
   discrimination ≈ ICC; most 3P% differences are sampling variability.)
3. **Belitz & Stackelberg** 2021, doi:10.1016/j.envsoft.2021.105006. (Dynamic-range bias of
   ensemble trees; ROE / EDM post-hoc correction — six methods.)
4. **Boulevard** — Zhou & Hooker, arXiv:1806.09762. (Regularized leaf-average mechanism of
   the compression.)
5. **Huang** (2017, *Statistical Modelling*) mean-parameterized CMP; **Huang & Rathouz**
   2017 (mean/dispersion orthogonality).
6. **Casella–Berger** — empirical-Bayes shrinkage constant `K = σ²_within / σ²_between`.
7. **Hubrich et al.**, *Understanding and forecasting aggregate and disaggregate price
   dynamics*, European Central Bank Working Paper No. 1365, 2011. (Aggregate vs disaggregate
   forecasting; disaggregate wins only with known-DGP components.)
8. **Chevalier & Côté**, *European Actuarial Journal* 2025, DOI 10.1007/s13385-025-00428-5.
   (Multi-parametric GBM benchmark; point ≡ probabilistic on NB-class targets; global
   dispersion beat covariate-specific.)
   - 8a. **Daub et al.**, balanced GAMLSS boosting (balanced step length), *Computational
     Statistics* 2025, DOI 10.1007/s00607-023-01224-3; arXiv 2602.17272, 2026.
9. **Prevett, Hui, Tho, Welsh, Westveld** (GBMixed), ANU, arXiv 2511.00217, 31 Oct 2025.
   (Criticizes GPBoost bias; MEGB/GBMixed mixed-effects boosting.)
10. **Dunn & Smyth** 1996, randomized-quantile residuals (RQR), doi:10.1080/10618600.1996.10474708.
11. **Philipson & Huang** 2023, *Statistics and Computing*, DOI 10.1007/s11222-023-10244-0.
    (Precomputed (μ,ν)→λ look-up grid.)
12. **CMPBoost / Chatla & Shmueli** 2020, JCGS (boosting reference; code
    `SuneelChatla/CMPTree`).
13. **Modeling Player and Team Performance in Basketball**, *Annual Review of Statistics and
    Its Application* 2021, DOI 10.1146/annurev-statistics-040720-015536, arXiv:2007.10550.
14. **CQR** — Romano, Patterson & Candès, NeurIPS 2019.
15. **LCMQR** — arXiv 2411.19523, late 2024.
16. **Yang et al.** 2026, arXiv:2404.07457. (Forcing NB on equi-dispersed data destabilizes
    the MLE; Poisson collapse.)
    - 16a. **DEGPD / ZIDEGPD** — Ahmad & Hussain, arXiv 2510.27365, 2025.
17. **CatBoost ordered TS** — Prokhorenkova et al., NeurIPS 2018, arXiv 1706.09516.
    (Unbiasedness proof for high-cardinality categoricals.)
18. **gbex** — Velthoen, Dombry, Cai, Engelke, *Extremes* (Springer), 2023.
19. **FAGTB** — Grari et al., arXiv 1911.05369.
20. **M²FGB** — Cruz et al., arXiv 2504.12458, Apr 2025.
21. **PGBM** — Sprangers et al., KDD 2021, DOI 10.1145/3447548.3467278.
22. **ziNB index / Wilson-Einbeck (Blasco-Moreno)** — Blasco-Moreno et al., *Methods Ecol
    Evol* 2019, DOI 10.1111/2041-210X.13185.
23. **Wilson & Einbeck**, *Statistical Modelling* 2019, DOI 10.1177/1471082X18762277.
24. **Corrected Vuong** — Desmarais & Harden 2013; Wilson 2015, *Economics Letters*, DOI
    10.1016/j.econlet.2014.12.029. (Vuong invalid for nested ZINB-vs-NB at γ=0.)
25. **Marginalized hurdle** — Kassahun et al., *Stat Med* 2014, DOI 10.1002/sim.6237.
26. **Marginalized-hurdle design choices** — Molenberghs et al. 2018, doi:10.1002/sim.7596;
    Liu, Zhang, Tang et al., *HSORM* 2018, DOI 10.1007/s10742-018-0183-6.
27. **MZINB foundations** — Long, Preisser, Herring & Golin, *Stat Med* 2014, DOI
    10.1002/sim.6293; Preisser, Das, Long, Divaris, *Stat Med* 2016, DOI 10.1002/sim.6804.
28. **Zając et al.** 2023. (NBA 3PA/game rose ~1000% 1979→2018-19, 3P% 28%→36%.)
29. **Campbell** 2021, doi:10.1111/2041-210X.13559. (Selection-bias trap — lock the protocol
    before refitting.)
30. **Zhang & Lu** 2012, doi:10.1080/02664763.2011.578621. (Dynamic-range bias.)
31. **Feng et al.** 2020, power for count GOF, doi:10.1186/s12874-020-01055-2.
32. **MZINB spatial / Pólya-gamma** — Mutiso et al., *Biometrical Journal* 2024, DOI
    10.1002/bimj.202300182.
33. **Arctan pinball** — Sluijterman et al., *Int J Mach Learn Cybern* 2025, DOI
    10.1007/s13042-025-02671-4.
34. **Feng GOF framework** — Feng, *J Stat Distrib Appl* 2021, DOI 10.1186/s40488-021-00121-4.
    (NB outperforms ZINB at ~20% zero rates.)

Additional citations referenced in the seed bibliography or method tables, retained for
completeness:

35. **MEGB** — Olaniran et al., *Scientific Reports* 15:30927, 22 Aug 2025, DOI
    10.1038/s41598-025-16526-z (CRAN + github.com/rid4stat/MEGB).
36. **MZINB Stata** — Cummings & Hardin, *Stata J* 19(3) 2019, DOI 10.1177/1536867X19874209.
37. **CMP head** — Philipson & Huang, *Statistics and Computing* 2023, DOI
    10.1007/s11222-023-10244-0. (See [11]; canonical-vs-mean CMP head reference.)
38. **cyc-GBM** — Delong, Lindholm & Zakrisson, SSRN 4352505, 9 Feb 2023, DOI
    10.2139/ssrn.4352505.
39. **CyclicBoosting** — Wick et al., Blue Yonder, arXiv 2009.07052; *SN OR Forum* 2021, DOI
    10.1007/s43069-021-00079-8.
40. **ZTNB references** — Hilbe 2011, *Negative Binomial Regression*; UCLA-OARC ZTNB
    tutorial; Grodri notes on count moments.
41. **Normalizing-flow heads** — LightGBMLSS v0.3.0, 20 Jul 2023.
42. **Efron** 1986 — double-Poisson (deep underdispersion fit).

New citations [43]–[48] from the 2026-05-22 breadth brief (`/tmp/researcher_breadth_75.md`):

43. **Pargent, Pfisterer, Thomas & Bischl**, *Regularized target encoding outperforms
    traditional methods in supervised ML with high cardinality features*, Computational
    Statistics 37:2671–2692, 2022. DOI 10.1007/s00180-022-01207-6; arXiv:2104.00629.
    (Regularized/out-of-fold target encoding beats one-hot & naive mean; `player_id`.)
44. **Globus-Harris, Harrison, Kearns, Roth & Sorrell**, *Multicalibration as Boosting for
    Regression*, ICML 2023. arXiv:2301.13767. (Per-decile squared-error de-biasing is a
    boosting procedure.)
45. **Hansen, Devic, Nakkiran & Sharan**, *When is Multicalibration Post-Processing
    Necessary?*, NeurIPS 2024. arXiv:2406.06487. (Trained tree ensembles often need little
    post-hoc multicalibration.)
46. **Cortes-Gomez et al.**, *Generalized Venn and Venn-Abers Calibration*, 2025.
    arXiv:2502.05676. (Point calibration ≠ distributional calibration.)
47. **Efron & Morris**, *Data Analysis Using Stein's Estimator and its Generalizations*, JASA
    70:311–319, 1975. DOI 10.1080/01621459.1975.10479864. (James-Stein/EB shrinkage for
    few-obs-per-unit.)
48. **Becker et al.**, *Fair admission risk prediction with proportional multicalibration*,
    2023. PMC10417639. (Percent-calibration error is worst in low-base-rate groups.)
