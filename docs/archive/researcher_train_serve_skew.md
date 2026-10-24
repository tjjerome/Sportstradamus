# In-repo research brief (R3): which train/serve skews and tail-calibration defects explain the selected-tail overstatement, ranked, with a training- or serving-side fix for each — 2026-10-04, revised after phase 2, R1's KILL and R2

## TL;DR

- **No train/serve skew in the calibration chain or the quotes explains the gap. The phase-1 verdict stands, and it survives R2's payout convention.** Post-fix recommended legs read .611 and hit .486: +12.5 pp on 6,138 legs. Walk-forward, the gap is bulk miscalibration of +1.5 pp [0.8, 2.1] plus a selection residual of +11.7 pp [9.2, 14.4] (89%). At R2's √3.5 Underdog convention it is +1.4 [0.7, 2.0] plus +9.5 [7.5, 11.8] (87%, 8,209 legs). Every pre-registered skew moves the cohort gap by 0.3 pp or less, and only the two decomposition components survive Holm. C1, C2, C4 and C6 are KILLed, as is C5 as a shape defect; C3 is a correctness fix.
- **Rank 1 is selection on model–market disagreement where the served mean knows little beyond the market (Smith & Winkler 2006, doi:10.1287/mnsc.1050.0451). R1's KILL closes the layer route, so the fix has to be model information.** R1 shows decision-time market observables encompass served p out of sample. Measured like for like (fused P, cell fixed effects, logistic encompassing; Clements & Harvey 2010, doi:10.1002/jae.1097), MLB live b_model is .26 (.07), against .38 (.19) offline in September and .49–.66 offline in May–August. NFL is ≈ 0 offline at every distance from the booster's training cutoff, and ≈ 0 live (−.03 (.16)). The offline-to-live drop is mostly the September calendar window plus about 0.1 from book-leg reference timing; it is not a serve-time skew.
- **The largest new effect is exploratory: in MLB, a version's first four days are far more honest than the rest of its life.** Recommended legs served 0–3 days after a version's first served date have a gap of +5.6 pp [4.0, 8.3] (1,059 legs, 10 days); from day 4 on, +13.5 [9.4, 17.8] (2,349 legs, 21 days). The difference is −7.9 pp [−12.3, −2.9] (−5.1 [−8.7, −1.1] at √3.5), and within-market b_model is .51 against .16. It repeats in all three post-fix MLB retrains. It is not a level drift, booster staleness, a stale serving gamelog or the comp snapshot, and NFL shows none. Retrains were not randomized and the mechanism is unidentified, so it is a lead to test (I6g), not yet a lever.
- **Phase 2 found real skews whose measured effects are small or inconclusive.**
  - Serve-path parity (same pickle, line and book leg; only the features differ): sd(ΔP) 2.8–5.9 pp. Training-path features would cut NFL total overstatement to 0.68× [0.58, 1.11].
  - Training's book leg is the latest quote at or before game-day 12:00 UTC; serving decides 5–11 h later (median).
  - Training game lines include in-game quotes: MLB enriched totals correlate .57 with team runs against .16 pre-game. The effect on P is sd ≈ 0.5 pp.
  - Each booster is fit on the first 70% of its matrix, so NFL boosters end in December 2024. Information does not decay with distance from that cutoff, so this is not a lever.
- **What I6 should build, in order:**
  - (1) the tail scorecard (§6, with six small deltas);
  - (2) I6e, serve-time feature logging plus a parity monitor;
  - (3) I6g, a paired version-age test before any retrain-cadence change;
  - (4) I6f, as-of alignment of training's archive inputs;
  - (5) hygiene: I6a fractional push labels, and removing the T ridge or setting it by CV;
  - (6) I6d, NFL information research accepted on fixed-effect encompassing.

  Do not build a trust layer, w = 1 serving or price-blind recalibration as a gap fix; no model is pulled or demoted. For parlays, R2's [0.9, 1.1] band needs per-leg hit/read ≥ 0.95 at two legs and ≥ 0.98 at five. Nothing measured gets there; even fresh-version MLB legs sit at 0.91.

## 1. Decision question and the statistic that decides it

**Decision question (dispatch, verbatim):** "Which train/serve skews and tail-calibration defects explain why the served probability overstates the hit rate on edge-selected legs (.650 read vs .496 realized), ranked by measured contribution, and what training- or serving-side fix closes each, with every model staying in production?"

**Statistic.**
- The **selected-tail gap**: mean(served chosen-side `Win Prob` − Hit) over recommended legs. Recommended means posted, edge ≥ 0.05 at platform payout, payout in (1, 2.5].
- Each candidate's **contribution** is Δgap after removing that skew. The served probability is recomputed, the live selection rule is re-run on the same posted legs (same side only; an unposted side cannot be priced), and the two cohorts are compared on the same day-resampling weights (paired day-clustered bootstrap, 4,000 draws).
- Share = −Δgap / gap.
- A complementary additive split is gap = (read − E[y | read]) + (E[y | read] − hit). E[y | read] is a walk-forward weekly isotonic of Hit on `Win Prob`, per league × side, fit on *all posted* legs dated before the week. The first term is bulk miscalibration; the second is the selection residual.

## 2. Pre-registration (method rules verbatim from the dispatch)

> "Post-fix data only for live evidence (history Date ≥ 2026-08-31, the book-leg fix era); walk-forward for anything fitted; day-clustered CIs (block bootstrap by Date, ≥ 2,000 resamples); pre-register the candidate list below, Holm across candidates, report everything tried; NFL reported separately; per-defect KILL is a valid verdict; primary sources with DOIs / arXiv IDs."

**Pre-registered candidates (from the dispatch):**

| # | Candidate | Pre-registered measurement |
|---|---|---|
| C1 | Unquoted-row pooling skew. Training fuses non-authentic rows at w = 1. Serving pools every row at w with `books_ev = Market Projection.fillna(Projection)`, so the shape is pulled toward the generic book shape (NegBin r→1/cv, DPO φ, SkewNormal symmetric leg with 0.8 floor, ZI gate → hist_gate). | Served vs trained-convention probability on unquoted rows; effect at DFS lines |
| C2 | Combo-sum quotes: derived (never blended) in training, blended at w in serving | Exposure, then effect |
| C3 | Tie label: training labels `Result >= Line` as Over (pipeline ~3204, ~5002), while live voids pushes | Calibration and Under-side effect on integer-line count cells |
| C4 | `Market Prob` at DFS rungs is a shape-decode extrapolation of the consensus. The `ladder` holds real sportsbook rungs for a subset. | Decode vs real rung, and which is right against outcomes |
| C5 | Tail reliability at offered lines, per cell, by tail z and alt flag, both sides | Cells, sides and z bands that carry the gap |
| C6 | NFL Under bias (+3.6 pp, λ_all −0.14) | Split by `Model Version` around the matchup-leak removal (96b573fe) and the fp-feature era boundaries |
| C7 | Evaluation design: offline g1 covers authentic rows only, at the modal line, Over only, tie = Over, with no payout and no selection | Distance of that population from the live recommended one; specify the "tail scorecard" |

**Holm family.** Holm (1979) is applied over the pre-registered inferential tests: C1, C3 (fractional), C4, C4b, C5 (two τ arms), C6, and the two decomposition components. C2 (exposure) and C7 (descriptive) carry no p-value.

**Exploratory, not pre-registered.** These are reported separately and are not Holm-corrected. They came up while chasing C5–C7 and the main session's fallback note:
- E1: temperature-penalty mechanism;
- E2: price-blind posted-leg recalibration cap;
- E3: DFS-vs-consensus line distance (main-session request);
- E4: forecast encompassing;
- E5: mean compression;
- E6: model-removed (book-only) selection.

**Phase 2** came after R1's KILL, R1's regime map and R2's result. It added the items below, which are equally exploratory and not Holm-corrected:
- E7: serve-path parity (same pickle; matrix-row features against serve-time features);
- E8: archive-input timing — the book-leg cutoff, quote age, the game-line leak and miss default, the MLB comp snapshot, the booster training window;
- E9: level-robust information by distance from the booster's training cutoff;
- E10: non-participation (no plate appearance) legs;
- E11: book-leg reference timing (12:00 UTC against decision time);
- E12: within-cell information by calendar period, offline and live;
- E13: the September mechanism (call-ups);
- E14: version age (R1 Finding 7), with three mechanism probes;
- robustness: the decomposition at R2's √3.5 Underdog convention.

## 3. Data actually used

- **Live cohort.**
  - Source: `data/runtime/history.parquet` (533,411 rows, 2026-02-19 → 10-05), deduplicated on (Date, Player, Market, Line, Bet, Platform).
  - Filter: settled Over/Under, `Boost > 0`.
  - Payout: Underdog Boost × 1.78, Sleeper decimal (memory `history_side_price_conventions`).
  - This reproduces the plan cohort exactly: 16,731 recommended all-time, hit .4961, read .6503, book .5597, payout 1.828, ROI −10.48%.
  - Post-fix live evidence covers 131,798 settled posted legs and 6,138 recommended (08-31 → 10-02): hit .4858, read .6109, book .4927, payout 1.851, ROI −11.8%. By league: MLB 3,442, NFL 2,528, WNBA 134, NHL 34.
  - The walk-forward decomposition needs ≥ 500 prior legs per league × side, which leaves 5,523 legs (09-07 → 10-02).
- **Model knobs per served leg.**
  - A `Model Version` → pickle registry, built by loading pickles read-only from `data/models/` and two backup trees. It covers 64 of 106 post-fix versions and 53.4% of recommended legs. Current pickles alone serve 26.2% of recommended legs, and NFL 0%.
  - Persisted T and Disp Cal match history exactly.
  - Live count-family re-serves reconstruct `Win Prob` to ≤ 7.7e-7 (55,533 rows). SkewNormal cannot be re-served live because `Model Skew` is not persisted.
- **Offline.**
  - `data/test_sets/*.csv`, 48 cells that persist the model-only columns (MLB 09-17, NFL 09-29/10-03, WNBA 09-17, NHL hits; NBA and most NHL lack them).
  - Re-serving with the training convention reproduces each CSV's `P` to ≤ 1.1e-13.
  - The final booster is fit on the train split only, with a fixed `opt_rounds` from train-split CV (`training/pipeline.py:367-378`). Validation and test rows are therefore out-of-sample for the calibration chain.
- **Archive** (`archive/archive.duckdb`, opened `read_only=True` only).
  - `ladder` holds DFS polls from August 2026 (164,545 in August, 6.84M in September) and sportsbook alt rungs from February.
  - `odds` gives per-book (line, under_prob).
  - Decision time is proxied by the platform's last poll of the leg's exact line.
  - Decision-time rung coverage of recommended legs is 39.0%; decision-time consensus-line coverage is 51.4%.
- **Tail-scorecard prototype.** Test rows dated ≥ 2026-08-30 × each DFS rung that Underdog/Sleeper held at its last poll gives 40,568 rung rows over 32 cells. This is the only overlap between held-out test rows and the ladder era.
- **Phase-2 inputs.**
  - **Cached matrices** `data/training_data/{LEAGUE}_{market}.parquet`: MLB through 2026-09-16 (built 09-17), NFL through 09-28 (rebuilt 09-26..29), WNBA through 08-30, NHL through 06-14.
  - **Parity pickles**, loaded read-only:
    - MLB `20260829.*` (`research/logs/restamp_identity/20260829T132406Z/`, served 08-31..09-03);
    - NFL `20260917.*` pre-bump incumbents (`~/backups/sportstradamus/2026-09-19-nfl-fp-bump/models/`, served 09-20..10-01).

    The MLB `20260903.*` versions (52,280 posted legs) are not on disk. Parity re-serves 9,382 posted legs on 13 days: MLB 1,833 and NFL 7,549. The NFL count includes 120 passing-tds legs served 09-09..09-20 by the pre-identity pickle `20260806.none.e997c067`. Dropping legs clipped at 0.9 or not yet graded leaves 9,358 (MLB 1,831, NFL 7,527), and those feed the era rows of Finding 13.
  - **Machinery check.** Re-serving matrix rows through `prediction.model_prob`'s own chain reproduces the test-set `P` exactly on 98.5% of rows with own-scale ≥ `_OWN_SCALE_MIN`. The residue is zero-history rows, which serving drops.
  - **R2's per-pick convention.** Wherever it changes the cohort, results are also reported at Underdog Boost × √3.5 = 1.8708: 9,198 recommended legs instead of 6,138.
- **Subsampling: none.** Every script ran in under 10 minutes on full data.
- **Not used.** The pre-fix era appears only as context, for the main session's all-time fallback figures (flagged where it appears). NBA is not in the post-fix live sample.

## 4. Key findings (results per candidate)

### Finding 1: about 89% of the gap is a selection residual; bulk miscalibration is about 11%

Walk-forward decomposition on 5,523 recommended legs, 09-07 → 10-02, with day-clustered 95% CIs (4,000 resamples):

| Component | Estimate | 95% CI | Share |
|---|---|---|---|
| Gap (read − hit) | +.1322 | [+.1062, +.1589] | 100% |
| Bulk miscalibration (read − E[y \| read]) | +.0149 | [+.0080, +.0210] | ≈11% |
| Selection residual (E[y \| read] − hit) | +.1173 | [+.0924, +.1443] | ≈89% |

The gap / bulk / residual split by group:

| Group | Gap | Bulk | Residual |
|---|---|---|---|
| MLB | .123 | .007 | .116 |
| NFL | .149 | .024 | .126 |
| Over | .160 | .008 | .152 |
| Under | .119 | .018 | .101 |
| Quoted | .151 | .020 | .132 |
| Unquoted | .121 | .012 | .109 |

On all posted legs the served probability is calibrated within about 1–2 pp in every bin except the top one, (0.8, 0.9]: Over +5.9 pp, Under +3.0 pp.

On *recommended* legs, the hit rate tracks the platform's price in every served-probability bin, not the model's read:

| Served bin | n | Hit | Read | Platform/book (`Market Prob`) |
|---|---|---|---|---|
| (0.50, 0.55] | 1,545 | .394 | .527 | .416 |
| (0.55, 0.60] | 1,437 | .479 | .577 | .468 |
| (0.60, 0.65] | 1,750 | .507 | .623 | .503 |
| (0.65, 0.70] | 713 | .533 | .674 | .542 |
| (0.70, 0.75] | 339 | .602 | .721 | .589 |
| (0.75, 0.80] | 146 | .589 | .774 | .644 |
| (0.80, 0.90] | 207 | .618 | .856 | .714 |

This is the optimizer's / winner's curse (Smith & Winkler 2006, doi:10.1287/mnsc.1050.0451; Capen, Clapp & Campbell 1971, SPE-2993-PA; Thaler 1988, doi:10.1257/jep.2.1.191). A rule that picks the legs where the model's read most exceeds the platform's price picks the model's largest errors. The selected subset is then miscalibrated even though the bulk is not. Marginal calibration does not imply calibration on a subpopulation defined by the price and the decision rule. The formal versions are multicalibration (Hébert-Johnson et al. 2018, arXiv:1711.08513) and decision calibration (Zhao et al. 2021, arXiv:2107.05719). The prior brief's λ_sel ≈ −0.10 ± 0.10 (`docs/archive/researcher_selection_shrink.md`) is the same fact measured as a shrink coefficient.

**Where the gap sits.** Post-fix recommended legs; share = that group's Σ(read − hit) over the cohort's:

| Split | Groups (n; gap; share) |
|---|---|
| League | MLB 3,442; .112; 50.1% · NFL 2,528; .149; 49.2% · WNBA 134; .022 · NHL 34 |
| Side | Over 2,003; .155; 40.6% · Under 4,135; .110; 59.4% |
| Quote class | quoted 2,165; .144; 40.5% · unquoted 3,935; .114; 58.4% · fallback 38 |
| Line type | main 5,162; .122 · alt 976; .140 |
| Integer vs half line | integer 0 · half-point 6,138 (100%) |
| Payout band | (2.0, 2.5]: 33.2% of the gap at .153 |
| Family | SkewNormal 50.7% of the gap |
| Top cells | MLB H+R+RBI 11.1% · NFL rushing yards 10.1% · NFL receptions 10.0% · MLB hits allowed 7.9% |

No split isolates the gap. It is present in every quote class, both sides, main and alt lines, and both large leagues. That is what a selection effect looks like, as opposed to one broken path.

### Finding 2 (C1): unquoted pooling skew is real in mechanism, absent in effect. KILL.

- **Live, count families.** Exact re-serve to ≤ 7.7e-7, inverting the served pool to model-only parameters (DPO φ, NegBin r, ZINB gate) and re-serving at w = 1.
  - 34,389 posted unquoted legs: served read .6407 against .6418 at w = 1; mean |Δ| = .0114; log-loss .6406 against .6410.
  - 1,185 recommended legs: hit .484. Removing the skew *raises* the read, .608 → .611, which is the wrong direction for closing the gap.
  - Cohort re-selection: gap 12.50 → 12.32 pp, Δ −0.18 [−0.60, +0.12], p = .27.
- **Offline.** On 48,647 non-authentic test rows, mean ΔP = −0.00036 and log-loss is .60177 against .60178. SkewNormal cells split: receiving yards and receptions do better at w = 1, rushing yards and WNBA PR do better pooled.
- **At DFS rungs (prototype).** Selected gap is 10.11 pp under the serving convention and 10.65 pp under the training convention.
- **Verdict: KILL.** Do not build w = 1 serving for unquoted rows. If the two conventions are aligned for hygiene (I6b), align training to serving, since serving is the slightly better of the two at the rungs. Expect no movement in the gap.

### Finding 3 (C2): combo-sum quotes have no exposure. KILL.

- Post-fix there are 9,558 quoted combo legs. A *direct* sportsbook quote existed for 100% of MLB H+R+RBI, 100% of NFL tds and 98–100% of WNBA combos.
- Only 7 legs (6 recommended) were priced off a component sum, about 0.3% of the gap.
- One subtle skew is left. Serving inverts direct-quoted combo means under the component-sum shape, while training uses the generic shape. On the affected recommended H+R+RBI legs the model is *under*-confident (hit .662, read .628), so this skew cannot contribute to the overstatement.
- Optional alignment touches `prediction/book_quotes.py` (`with_component_sum_shape`). It is not research-gated unless it reaches `*/combined_markets.py`, which is listed in `.claude/research_gated.txt`.

### Finding 4 (C3): the tie label is a correctness bug with negligible gap effect. FIX for hygiene; it is not a gap lever.

- **Direct exposure is zero.** Post-fix, 0.0% of recommended legs sit on an integer line; every one is a half-point line.
- **Indirect channel.** `Result >= Line` = Over enters the T and PROB_STAGE fits (`training/pipeline.py:3204`, `5002`). Integer validation lines are common in some NFL and WNBA cells and nearly absent in MLB (2.9% of authentic test rows overall):

  | Cell | Integer validation lines | Pushes |
  |---|---|---|
  | NFL qb tds | 66% | 18.6% |
  | NFL targets | 66% | 15% |
  | WNBA TOV | 60% | 19% |
  | WNBA STL | 39% | 13% |

- **Cell-level effect.** Refitting the chain on the test half under a fractional label shifts P(Over) at half-point lines. The fractional label is y = ½ at a push, which matches the half-split `cdf − pmf/2` the serving decode uses. Shift of tie-label P(Over) relative to drop-push, unpenalized fits:

  | Cell | Shift |
  |---|---|
  | WNBA TOV | +9.2 pp |
  | NFL qb tds | +8.5 pp |
  | NFL targets | +5.0 pp |
  | NFL completions | +2.6 pp |
  | NFL carries | +1.7 pp |

  The sign depends on the stage. In T-only cells the tie label *flattens*; in Platt cells it moves the intercept toward Over.
- **Live, applied to post-fix legs** (per-cell monotone map from the test-half refit; same side only):
  - Fractional label: read on the fixed recommended set falls by 0.15 pp [0.09, 0.20]. Re-selected gap 12.50 → 12.30, Δ −0.21 [−0.52, +0.18], p = .26.
  - Drop-push variant: Δ −0.06 [−0.32, +0.28].
  - The largest cell-side moves are NFL targets Over, whose read falls 9.7 pp (116 legs); MLB runs allowed Over, down 2.5 pp (257 legs); and NFL completions Under, *up* 3.4 pp (49 legs).
- **Verdict.** Ship I6a's shared label helper (`training/labels.py`) for correctness, using the fractional label so training matches the serving half-split. Expected gap effect is ≤ 0.2 pp. Retrain is required; files are `training/pipeline.py` and the new `training/labels.py`, neither research-gated.

### Finding 5 (C4): the `Market Prob` decode at DFS rungs agrees with real rungs. KILL. Rung coverage (C4b) reduces volume, not the per-leg gap.

- **Coverage.** Of recommended legs, 59.7% have a sportsbook rung at the platform line at any time; only 39.0% (2,393) have one at decision time.
- **Decode vs rung.** On covered recommended legs:

  | | Hit | Read | `Market Prob` (decode) | Real rung |
  |---|---|---|---|---|
  | Mean | .474 | .613 | .495 | .491 |
  | Log-loss | — | .732 | .687 | .684 |

  Decode−rung MAD is 2.4 pp. The paired log-loss difference is +0.0029 [−0.0026, +0.0094], p = .33. The decode is not the defect; both market numbers sit near the outcome and the model sits 12 pp above.
- **C4b, rung as the book leg on unquoted rows.** Live count families, exact chain, reconstruction 3.2e-7:
  - Only 231 of 1,185 unquoted count recommended legs (19.5%) had a decision-time rung.
  - On the 212 non-mean-stage legs: hit .392, read .608, rung .501. The rung-quoted blend at the trained w reads .563.
  - Re-selection keeps 47 of 212 legs. The per-leg gap is unchanged (21.6 → 20.1 pp); total overstatement falls from 45.9 to 9.5 legs.
  - Cohort: Δgap −0.26 [−0.96, +0.13], p = .32.
  - The offline prototype (all families) shows no change (+9.97 → +10.27 pp), because test rows that are non-authentic but have a decision-time rung are rare.
- **Verdict: KILL as a gap lever.** Rung coverage is defensible as a quote-coverage or volume lever (`prediction/book_quotes.py`, serving-side, no retrain). Expect −15% recommended volume on covered unquoted legs and no change in per-leg honesty.

### Finding 6 (C5): the gap is not a tail-shape defect. KILL for shape; price-blind tail recalibration is capped near the bulk component.

Recommended legs by signed model-relative z, where z > 0 means the line sits on the bet's favoured side of the served mean. SkewNormal σ is approximated by the scale ω, because skew is not persisted.

| z band | n | Gap | Share |
|---|---|---|---|
| (−∞, −0.5] | 64 | .088 | 0.7% |
| (−0.5, 0] | 1,661 | .089 | 19.3% |
| (0, 0.25] | 2,100 | .121 | 33.0% |
| (0.25, 0.5] | 1,400 | .126 | 23.0% |
| (0.5, 0.75] | 527 | .168 | 11.6% |
| (0.75, 1.0] | 174 | .226 | 5.1% |
| (1.0, 1.5] | 100 | .235 | 3.1% |
| (1.5, ∞) | 60 | .329 | 2.6% |

- About 75% of the gap sits at |z| < 0.5, near the money. The far tail is worse per leg but small in volume. All posted legs are calibrated within 1–2 pp except z_side > 0.75 (+2.9 to +5.6 pp).
- Price-blind recalibration on the tail-scorecard prototype, an extra temperature τ per cell:
  - τ fit walk-forward on authentic modal-line test rows: selected gap 10.1 → 7.2 pp, Δ −2.95 [−5.27, −0.03], p = .049, Holm .34. Bulk log-loss gets *worse*, .5750 → .5815.
  - τ cross-fit on all DFS rungs: Δ −0.82 [−1.48, +0.10], p = .072; bulk neutral.
- Any recalibration that cannot see the price is bounded by the bulk component (Finding 1), and it buys selected-tail gap with bulk log-loss.

### Finding 7 (C6): the NFL Under bias is not the matchup leak. KILL as the explanation. The mechanisms are served-mean compression against the market and failure to encompass it.

- **Era split.** All posted post-fix NFL Under legs, by `Model Version`:

  | Era | Legs | Days | Under gap |
  |---|---|---|---|
  | Leak era (< 20260901) | 1,120 | 6 | +2.25 pp |
  | Post-leak | 8,884 | 11 | +3.78 pp |

  The difference is +1.54 [−2.39, +13.25], p = .34; the leak hypothesis predicts a negative number. Finer eras, Over/Under gap in pp:

  | Era | Over | Under |
  |---|---|---|
  | A (0829) | +2.9 | +2.2 |
  | B (0901–02) | +0.2 | +4.9 |
  | C (0917) | +2.0 | +2.7 |
  | D (0927+, n 119 / 467) | −13.9 | +11.0 |

  Every era CI includes 0, with about 6 game-days per era. 17 D-era rows carry game date 09-09 (unexplained). The fp team-feature blackout from 2026 week 5 (memory `nfl_line_matchups_leak_and_fp_2026_cliff`) is a forward risk the D-era sample cannot resolve.
- **Held-out test calibration slope on authentic (modal-line) rows** (should be about 1):

  | Cell | Slope ± SE |
  |---|---|
  | receptions | .138 ± .111 |
  | receiving yards | .164 ± .184 |
  | rushing yards | .033 ± .206 |
  | interceptions | −.203 ± .164 |
  | passing tds | .192 ± .269 |

  MLB authentic cells range .79–1.23. Non-authentic (synthetic-line) rows often run above 1, e.g. receiving yards 1.58. *At the market line, the NFL served probability carries almost no information.*
- **E4, forecast encompassing** (exploratory; Chong & Hendry 1986, doi:10.2307/2297611; Fair & Shiller 1990, AER 80(3):375–389; Harvey, Leybourne & Newbold 1998, doi:10.1080/07350015.1998.10524759). The regression is Actual ~ a + b_model·served mean + b_market·market line, one row per post-fix player-game, day-clustered bootstrap.

  | Cell | Market proxy | n | b_model [95% CI] | b_market [95% CI] | R² model / market |
  |---|---|---|---|---|---|
  | NFL receiving yards | consensus | 517 | −0.04 [−0.36, 0.34] | 1.07 [0.77, 1.33] | .30 / .39 |
  | NFL receptions | consensus | 529 | −0.10 [−0.51, 0.09] | 1.03 [0.89, 1.29] | .21 / .35 |
  | NFL receptions | DFS main line | 576 | 0.21 [−0.23, 0.47] | 0.79 [0.57, 1.12] | .22 / .30 |
  | MLB hits | consensus | 5,404 | 0.82 [0.62, 1.03] | 0.30 [0.16, 0.45] | .027 / .018 |
  | MLB hits+runs+rbi | consensus | 5,387 | 0.34 [0.08, 0.61] | 0.62 [0.41, 0.85] | .028 / .042 |
  | MLB total bases | consensus | 5,346 | 0.50 [0.24, 0.76] | 0.44 [0.26, 0.61] | .027 / .033 |
  | MLB runs allowed | consensus | 549 | −0.17 [−0.43, 0.09] | 0.86 [0.58, 1.17] | .001 / .058 |
  | MLB hits allowed | consensus | 532 | 0.53 [0.32, 0.94] | 0.66 [0.46, 0.84] | .074 / .099 |

  In the two biggest NFL volume cells the market *encompasses* the served mean. MLB models carry partial independent information. This is the mechanism behind "NFL = selection + model bias" in the prior brief.
- **E5, mean compression against the market.** A mean that is calibrated given the market line implies slope(served ~ consensus) = slope(actual ~ consensus) by iterated expectations. Measured slopes:

  | Cell | Served | Actual | Difference [95% CI] |
  |---|---|---|---|
  | NFL receiving yards | 0.68 | 1.04 | +0.36 [0.25, 0.56] |
  | NFL receptions | 0.68 | 0.96 | +0.28 [0.23, 0.38] |
  | MLB hits+runs+rbi | 0.44 | 0.77 | +0.33 [0.15, 0.53] |
  | MLB total bases | 0.35 | 0.61 | +0.27 [0.14, 0.38] |
  | MLB runs allowed | 0.30 | 0.81 | +0.51 [0.23, 0.81] |

  In the top consensus tercile the served mean − actual is −11.6 yards (receiving yards) and −0.63 receptions. Mincer–Zarnowitz slopes of actual on the served mean alone are about 1 (0.89–1.15), so the compression is invisible without the market. The edge rule picks the compression side: Under on top-tercile players, Over on bottom-tercile players.

  | Group | Legs | Hit | Read | Gap | Share of league gap |
  |---|---|---|---|---|---|
  | NFL, compression-consistent | 919 | .397 | .604 | +20.6 | 61% |
  | NFL, against compression | 335 | .549 | .597 | +4.7 | 5% |
  | MLB, compression-consistent | 1,159 | .494 | .623 | +12.9 | 51% |
  | MLB, against compression | 601 | .504 | .600 | +9.6 | 20% |

  Selection rates are 19.6% against 9.4% for NFL and 4.0% against 2.3% for MLB. Mid-tercile legs carry the rest.

  A price-blind de-compression (walk-forward OLS of actual on the served mean, pushed through the local density) does nothing: Δgap −0.50 [−1.06, +0.05].
- **Verdict.** The NFL Under bias is a model-information defect, with compression as the visible symptom; it is not a leak artefact. The fix is training-side, in the NFL lane (I6d). Retrain on the post-fix era, and adopt *encompassing* as the acceptance metric: b_model's CI lower bound > 0 on held-out player-games with a real quote. Do not adopt g1 Brier, which this defect passes; every NFL volume cell shows `g1_pass = True`. Lowering w toward the book would be the mechanical patch, but that is the killed shrink lever and it cannot reach unquoted legs, so it is not proposed. Phase 2 sharpens the verdict. NFL model-only information is ≈ 0 even in the first 60 days after the booster's training cutoff (Finding 14f), so retraining on post-leak rows alone is unlikely to create it. The open lever is new at-market information (I6d), after the parity repair (I6e).

### Finding 8 (C7): the offline g1 population is far from the live recommended one; a tail scorecard on DFS rungs reproduces the live gap

| | g1 test population (44 cells, MLB/NFL/WNBA/NHL) | Live recommended, post-fix |
|---|---|---|
| Rows | 167,568 authentic of 251,625 test rows | 6,138 of 131,798 posted legs (4.7%) |
| Quote class | 100% authentic (non-authentic rows excluded) | quoted 35.3%, unquoted 64.1%, fallback 0.6% |
| Line | modal sportsbook line; integer 2.9%, pushes 0.4% | DFS platform line; integer 0.0%, alt 15.9% |
| Side | Over only (Brier is side-symmetric; selection is not) | argmax side; Under 67.4% |
| Selection | none | edge ≥ 0.05 at platform payout, payout in (1, 2.5] |
| \|p − 0.5\| | mean .182 | mean .111 |
| \|model − market\| | mean .050; share > 0.10: 10.1% | mean .120; share > 0.10: 74.2% |
| Payout | none | mean 1.851; (2.0, 2.5] share 27.1% |
| League mix | MLB 58.7%, NHL 30.4%, NFL 7.4%, WNBA 3.5% | MLB 56.1%, NFL 41.2%, WNBA 2.2%, NHL 0.6% |

g1 samples the region where the model agrees with the market. Live selection samples the region where it disagrees by more than 10 pp three times in four; across all posted legs that share is 7.9%. Every NFL volume cell passes g1 while carrying the worst selected-tail gap. This is the textbook gap between calibration "in the large" and calibration on the decision-relevant subpopulation (Van Calster et al. 2019, doi:10.1186/s12916-019-1466-7; Zhao et al. 2021, arXiv:2107.05719).

**Tail-scorecard prototype** (`scratchpad/r3/tail_proto*.py`).
- Construction: test rows dated ≥ 2026-08-30 × each Underdog/Sleeper rung at its last poll, giving 40,568 rung rows.
- Repricing: serving convention, with payout = 1/(p_dfs,side × overround). Overround is Sleeper 1.128 and Underdog 1.178 (2/1.78 at the standard rung); these are two-sided medians measured on the ladder.
- Selection: the live rule is replayed.
- Result: 1,615 rows selected, read .686, hit .585, market .613, gap +10.1 pp [3.5, 13.9]. The live gap is +13.2 [10.6, 15.9]. The bulk gap is −1.2 pp.
- Splits: main-rung gap .135, alt-rung .086. NFL authentic Over .230 and Under .148; MLB authentic Under .050.
- It reproduces the live phenomenon from held-out rows, which no current gate does.

### Finding 9 (fallback class, main-session §6 measurement, 2026-10-04): the all-time book-only gap is a pre-fix decode bug, but the curse survives with the model removed

All-time, per the main session: the book_fallback class (Win Prob = decoded sportsbook probability, no model) has 4,463 recommended legs, read .670, hit .492, ROI −9.9%. On all posted legs every class is calibrated within 1–2 pp. This brief's measurement of the same class:

- **Era.**
  - 4,425 of 4,463 legs (99.1%) are pre-fix: 443 in July and 3,982 in August. The other 38 are post-fix (read .619, hit .395).
  - The pre-fix class is MLB 2,821 (read .627, hit .484) and WNBA 1,642 (read .742, hit .508), and 96% of it is Under.
  - The mechanism is the one `6c21665c` (2026-08-29) removed. The old `book_fallback_prob` "averaged per-book EV MEANS across different lines and re-derived probs through a fitted-skew decode of a symmetric encode — … the path Sleeper's discounted lines poisoned (WNBA PRA claimed 0.74 unders hitting 0.60)". The WNBA fallback Unders here read .746 and hit .510.
- **Split by |DFS line − consensus line|.** Consensus is the median of each sportsbook's last `odds.line` before game day + 30 h. This is a closing proxy, because no decision-time DFS polls exist before August; pre-fix coverage is 63.8%.

  | \|d\| band | Recommended fallback legs | Gap | Share of class gap |
  |---|---|---|---|
  | 0 | 1,532 | +19.8 | 38.4% |
  | (0, 0.5] | 379 | +14.4 | 6.9% |
  | (0.5, 1] | 655 | +20.3 | 16.9% |
  | (1, 2] | 183 | +19.9 | 4.6% |
  | > 2 | 113 | +25.7 | 3.7% |
  | no consensus | 1,601 | — | 29.5% |

  | Signed distance (bet side) | Legs | Hit | Read | Gap | Share |
  |---|---|---|---|---|---|
  | ≤ −1 (DFS line harder) | 587 | .394 | .692 | +29.9 | 22.2% |
  | (−1, 0) harder | 245 | .518 | .656 | +13.7 | 4.3% |
  | 0 | 1,532 | .473 | .671 | +19.8 | 38.4% |
  | (0, 1] easier | 382 | .607 | .744 | +13.6 | 6.6% |
  | > 1 easier | 116 | .819 | .753 | −6.6 | −1.0% |

  | Alt flag | Legs | Gap | Share |
  |---|---|---|---|
  | main | 4,016 | +16.4 | 83.4% |
  | alt | 447 | +29.3 | 16.6% |

  By platform: Sleeper 2,334 at +17.1 and Underdog 2,129 at +18.4.
- **Bulk decode reliability.**
  - All-time, all posted fallback legs:

    | Band | Legs | Read | Hit | Gap |
    |---|---|---|---|---|
    | At consensus | 21,523 | — | — | +1.1 |
    | DFS line ≥ 1 harder | 1,747 | .585 | .349 | +23.6 |
    | > 1 easier | 4,223 | — | — | −5.9 |

    The pre-fix decode did not move enough with line distance, which is the cross-line averaging signature.
  - **Post-fix** all posted fallback legs (8,102) are calibrated at every |d| band (−0.7 to +2.3 pp) and every signed band with n > 50. *The decode defect is fixed.*
- **E6, model-removed replay, post-fix.**
  - Setup: on all 51,911 quoted posted legs, serve the decoded book probability (`Market Prob`) instead of `Win Prob` and re-run the live rule (same side only).
  - Result: 374 legs selected, read .787, hit .684, payout 1.48, gap +10.3 pp [+0.4, +20.8]. Only 26.5% of them overlap the model's own selection.
  - Comparison: the model-served selection on the same legs is 2,165 legs at +14.4 [9.8, 17.8], payout 1.90.
  - By league: MLB 242 legs at +5.8, NFL 91 at +17.2, NHL 41 at +21.5.
- **Verdict.**
  - As a live defect, the fallback row is **FIXED** (`6c21665c`, `8bebd0e3`); nothing to build.
  - The main session's inference holds in a precise form: the gap needs no model. Selecting legs where *any* probability exceeds the platform's payout-implied price produces it, because the platform's price carries information the comparison probability lacks on exactly those legs.
  - The model's contribution is volume and size: 5.8× the selected legs, a higher payout and a larger gap. Phase 1 routed this to R1. R1 has since KILLed the layer, because decision-time market observables encompass served p. It therefore routes to model information (rank 1: I6d, I6g), not to a decode fix and not to a layer.

### Finding 10 (E3): line distance from consensus localizes a third of the post-fix gap; a distance-conditional recalibration removes the band, but the gap re-forms elsewhere

Decision-time consensus is the median of the last line per sportsbook observed at or before the platform's last poll of the leg's line. It covers 58.6% of posted and 51.4% of recommended legs, and lands in the same band as the closing consensus 93.4% of the time.

| Signed distance (bet side) | Recommended legs | Hit | Read | Platform | Payout | Gap [95% CI] | Share of post-fix gap | Selection rate |
|---|---|---|---|---|---|---|---|---|
| ≤ −1 harder | 830 | .369 | .554 | .412 | 2.10 | +18.6 [14.0, 22.3] | 20.1% | 39.6% |
| (−1, 0) harder | 60 | .500 | .593 | .466 | 1.90 | +9.3 [−9.2, 21.7] | 0.7% | 17.4% |
| 0 (at consensus) | 1,948 | .468 | .615 | .500 | 1.84 | +14.8 [11.3, 18.7] | 37.4% | 3.8% |
| (0, 1] easier | 212 | .646 | .695 | .590 | 1.63 | +4.8 [−2.6, 12.0] | 1.3% | 1.3% |
| > 1 easier | 105 | .543 | .685 | .598 | 1.68 | +14.2 [−0.5, 26.2] | 1.9% | 1.4% |
| no decision-time consensus | 2,983 | .517 | .616 | .500 | 1.81 | +9.9 | 38.5% | — |

- **Harder band share.** On covered legs, legs harder than consensus carry 33.8% [16.8%, 43.0%] of the gap while making up 28.2% of the legs.
- **Alt flag within the harder band:** main 428 legs at +19.3, alt 402 at +17.8.
- **Composition.** The band is dominated by NFL Underdog Unders on lines far below consensus:

  | Market | DFS line | Consensus | Legs |
  |---|---|---|---|
  | receiving yards | 34.5 | 47.0 | 161 |
  | attempts | 28.5 | 31.5 | 127 |
  | rushing yards | 49.5 | 58.25 | 108 |
  | receptions | 3.5 | 4.5 | 50 |

  These are the compression-side legs of Finding 7.
- **Bulk at harder lines.** The model is overconfident even across all posted legs: quoted 1,480 legs read .539, hit .406 (+13.3); unquoted 583 legs read .545, hit .395 (+15.0). The platform's own implied probability (.435 / .474) is also above the hit, a favourite–longshot pattern (Snowberg & Wolfers 2010, doi:10.1086/655844; Thaler & Ziemba 1988, doi:10.1257/jep.2.2.161). It is much closer to the hit than the model.
- **Recalibration attempt.** A walk-forward Platt per league × side × distance band, fit weekly on *all posted* legs and never on the selected subset:
  - Bulk log-loss .6429 → .6425.
  - The harder band drops out of the selection entirely (406 → 0 legs).
  - But the cohort gap moves only 13.17 → 12.33 pp, Δ −0.84 [−3.31, +1.44], p = .54. Volume falls 4,116 → 2,824 and total overstatement 542 → 348 legs; the newly selected easier-band legs carry +13.5 pp.
  - Without the price, the selection residual re-forms wherever the edge threshold lands.

### Finding 11 (E1, E2): the temperature penalty pins T near 1; removing it is a correctness fix on the validation population with a small, mixed live payoff

- **Mechanism.** `_brier_temperature_loss` (`training/pipeline.py:2841`) minimises Brier + 0.01·(T − 1)². Refitting it on the test half (exchangeable with validation) *reproduces* the pickle T: correlation .905, median |ΔT| .032.

  | Cell | Pickle T | Unpenalized T |
  |---|---|---|
  | NFL receptions | 1.38 | 5.03 |
  | NFL rushing yards | 1.42 | 6.79 |
  | NFL carries | 1.51 | 4.39 |
  | NFL receiving yards | 1.27 | 4.11 |
  | NFL attempts | 1.30 | 3.57 |
  | NFL interceptions | 1.27 | 10.0 (bound) |
  | NFL passing tds | 1.27 | 10.0 (bound) |
  | MLB runs allowed | 1.27 | 4.85 |

  At the unpenalized optimum the penalty term is 0.02–0.81 against a Brier gain of 0.0015–0.012, so the regulariser, not the data, sets T. In 12 of 48 cells T_free > 1.5·T_pen; on authentic rows only, most NFL volume cells hit the 10.0 bound. Temperature scaling exists to flatten overconfident nets (Guo et al. 2017, arXiv:1706.04599). A (T − 1)² ridge at this weight disables it for exactly the overconfident cells.
- **Live payoff.** An extra unpenalized temperature fit walk-forward on test-half rows dated before each live week:

  | Fit population | Δgap [95% CI] | p | Recommended n | Overstatement (legs) | Bulk log-loss |
  |---|---|---|---|---|---|
  | All test rows | −1.03 [−2.32, +0.49] | .20 | 6,138 → 4,472 | 767 → 513 | .6470 → .6479 |
  | Authentic rows | −1.20 [−2.12, −0.10] | .029 | 6,138 → 4,244 | 767 → 480 | .6470 → .6486 |
  | DFS-rung rows (τ < 1 allowed) | +0.20 [−0.77, +1.26] | — | 6,138 → 8,784 | — | → .6488 |

  On the authentic-rows fit, NFL bulk log-loss goes .6839 → .6868. On the rung population the τ fit *sharpens* most cells (0.5–1.0) and bulk log-loss also worsens.
- **E2, cap test.** Serving the walk-forward posted-leg isotonic map E[y | read, league, side] itself makes the selected gap *worse*: 13.22 → 14.50, Δ +1.28 [−0.40, +3.10]; bulk log-loss .6461 → .6498.
- **Reading.** The three fit populations disagree on τ, flattening 3–10× versus sharpening, because the model's information depends on line distance. NFL has almost none at the money (Finding 7) and real discrimination at far rungs. No single price-blind temperature is right for both. The penalty is still a defect, since it silently overrides the fit, and should be removed or set by cross-validation. But it is a hygiene fix with an expected gap effect of about −1 pp and a small bulk cost, not the cure.

### Finding 12: NFL reported separately (method rule)

| NFL, post-fix recommended (11 game-days) | n | Gap [95% CI] | Bulk | Selection residual |
|---|---|---|---|---|
| All | 2,528 | +14.9 [12.9, 20.8] | +2.4 [1.0, 3.5] | +12.6 [10.7, 18.2] |
| Over | 871 | +16.9 [10.8, 37.4] | +1.5 [−1.4, 5.2] | +15.4 [8.2, 36.7] |
| Under | 1,657 | +13.9 [8.2, 18.8] | +2.8 [1.5, 4.6] | +11.1 [3.7, 16.3] |
| *MLB for contrast (24 days)* | 2,827 | +12.3 [8.7, 16.2] | +0.7 [0.6, 0.8] | +11.6 [8.0, 15.5] |

NFL carries the only material bulk component, +2.4 pp, mostly on Unders. It also has the at-market information failure, measured three ways:
- held-out authentic slopes of 0.03–0.19 (Finding 7);
- encompassing b_model ≈ 0 in receiving yards and receptions (E4);
- the strongest compression (E5).

The NFL volume cells also need the largest unpenalized temperatures, 3.6–10 (E1). NFL Under legs on lines harder than consensus are the single largest localized block (Finding 10). None of this is attributable to the matchup leak (C6). Every NFL CI is wide. The live window has 11 NFL game-days against 24 for MLB, and NFL training samples are about 10× smaller per player-season (cross-league caveat, plan §7.4). The D-era (0927+) Under gap of +11.0 rests on 467 legs. Phase 2 adds three facts. NFL information is ≈ 0 at every distance from the booster's training cutoff (Finding 14f). NFL shows no version-age effect (Finding 15). Serve-path parity is worst in NFL (Finding 13).

### Finding 13 (E7): serve-path parity is a real skew in mechanism, with an inconclusive effect on the selected tail. FIX as engineering (I6e); it is not a measured gap lever.

**Setup.**
- Take each post-fix leg whose serving pickle is on disk. Re-serve the same pickle at the leg's own line and book leg (`Market Projection`), but feed it the training-matrix row for (Player, Date) instead of the serve-time features.
- The chain is `prediction.model_prob`'s own: `_build_prob_params` → `_decode_model_params` → `_clamp_shape_ceiling` → `_sanitize_model_ev` → `_blend_with_book` → `_dispersion_calibrate` → `_model_over_and_push` → PIT recal → temperature → PROB_STAGE.
- Only the feature vector differs from the served `Win Prob`. Volume, two-part and affine cells are skipped.

| Cell | Legs | corr(served mean: matrix path, live) | Median \|Δmean\|/mean | Mean \|ΔP(Over)\| | Share identical (\|ΔP\| < 1e-4) |
|---|---|---|---|---|---|
| MLB hits | 350 | .903 | 2.9% | 1.87 pp | 1% |
| MLB hits+runs+rbi | 293 | .866 | 4.5% | 2.09 pp | 1% |
| MLB rbi | 328 | .913 | 5.1% | 1.73 pp | 2% |
| MLB runs | 209 | .842 | 6.1% | 2.14 pp | 7% |
| MLB singles | 43 | .840 | 3.0% | 1.51 pp | 0% |
| MLB total bases | 351 | .889 | 3.8% | 1.47 pp | 10% |
| MLB walks | 259 | .965 | 3.4% | 1.19 pp | 1% |
| NFL completions | 327 | .339 | 13.6% | 6.99 pp | 0% |
| NFL fantasy (Underdog) | 185 | .982 | 3.8% | 3.97 pp | 44% |
| NFL interceptions | 120 | .616 | 0.0% | 3.15 pp | 2% |
| NFL passing tds | 204 | .752 | 5.3% | 3.37 pp | 33% |
| NFL passing yards | 580 | .918 | 0.3% | 1.89 pp | 3% |
| NFL qb yards | 576 | .679 | 0.0% | 0.60 pp | 6% |
| NFL receiving yards | 2,116 | .985 | 3.1% | 0.88 pp | 70% |
| NFL receptions | 1,499 | .844 | 10.8% | 6.00 pp | 0% |
| NFL rushing yards | 1,194 | .965 | 6.5% | 3.17 pp | 1% |
| NFL tds | 264 | .907 | 0.0% | 2.21 pp | 44% |
| NFL yards | 364 | .944 | 4.3% | 2.61 pp | 1% |

The table covers the 18 cells on the two pickle sets of section 3 (9,262 legs). The 120 passing-tds legs on `20260806.none.e997c067` are in the era rows below but not in this table.

Parity is worst where serving builds an input that the matrix stores differently: completions (a `ratio_projvol` cell whose denominator is projected attempts), receptions, interceptions and quarterback yards. Attributing the gap to individual features is the monitor's first job. These legs cannot do it, because serve-time features are not persisted.

| Era | Legs | Player-games | Days | Log-loss, live − matrix path [95% CI] | sd(ΔP) | Recommended: live vs matrix path |
|---|---|---|---|---|---|---|
| MLB 08-31..09-03 | 1,831 | 208 | 3 | +.0035 [−.0041, +.0108] | .028 | 37 legs at +23.5 pp vs 35 at +22.4 |
| NFL 09-09..09-24 (09-09..09-17: the 120 `20260806` passing-tds legs only) | 3,616 | 267 | 8 | +.0045 [−.0015, +.0112] | .048 | 576 at +15.0 vs 461 at +14.7 |
| NFL 09-27..09-28 | 3,911 | 212 | 2 | +.0040 [−.0038, +.0117] | .059 | 622 at +14.5 vs 526 at +9.9 |

The quoted NFL market's log-loss on the same legs is .6565 and .6620, below both model paths.

- **Size relative to the signal.**
  - var(live − matrix) / var(live − m) is .169 for MLB quoted legs (263), .352 for MLB unquoted (1,568), .259 for NFL quoted (4,100) and .778 for NFL unquoted (3,427).
  - So on NFL unquoted legs the feature noise between serving and matrix is 78% as large as the model's whole deviation from the platform price.
- **Information, centered slope λ_c, live against matrix path** (player-game cluster bootstrap):

  | Legs | λ_c live | λ_c matrix path | Matrix − live [95% CI] | Note |
  |---|---|---|---|---|
  | MLB unquoted | −.003 | +.423 | +.425 [−.300, +1.173] | 203 player-games on 3 days; no information |
  | NFL quoted | −.149 | +.001 | +.149 [−.003, +.303] | |
  | NFL unquoted | +.136 | +.200 | +.067 [−.194, +.330] | |

- **Selection** (paired day bootstrap, NFL, 10 days).
  - Live: 1,198 recommended legs at +14.7 pp. Matrix path: 987 at +12.1. Difference −2.6 pp [−4.6, +5.0].
  - Total overstatement falls from 176 to 120 legs, a ratio of 0.68 [0.58, 1.11].
  - MLB: 0.90 [0.17, 2.23] on 3 days.
- **Ruled out.**
  - *Not version age.* sd(ΔP) shows no trend with version age: .028 / .016 / .034 for MLB at 2 / 4 / 5 days, and .051 / .035 / .038 / .063 / .046 for NFL at 3 / 4 / 7 / 10 / 11 days.
  - *Not the game lines.* Swapping the matrix game lines for decision-time values explains R² .015 (MLB) and .047 (NFL) of the parity variance (Finding 14c).
- **Caveats.**
  - (i) NFL code changed inside the window. Parity after these fixes (09-27..28, sd .059) is no better than before (.048).
    - The team-code map NaN'd 79% of serve-time rosters (fixed `6eac660c`, about 09-25).
    - The NEP KeyError dropped NFL on 09-25..27 (`abb82025`).
    - pbp ids were roster-gated (`4c8b4a91`, 09-26).
    - Prod served on legacy-era FP snapshot history until the 09-28 16:14 sync.
  - (ii) The matrices are dev-box builds, while serving uses prod data.
  - (iii) The matrix can carry residual look-ahead. Rebuild and append differ by design on the usage-quantile window, and MLB comps use the retrain-day snapshot (Finding 14e). The matrix path is therefore an upper bound on what a parity repair buys.
- **Verdict.**
  - Training and serving compute materially different feature values. This is the training/serving-skew test that the ML Test Score calls "perhaps the most important and least implemented" (Monitor 3; Breck et al. 2017, doi:10.1109/BigData.2017.8258038). It is also an unstable data dependency in the sense of Sculley et al. 2015 (NeurIPS 28). Zinkevich's Rules #29, #32 and #37 prescribe logging serve-time features, reusing code and measuring the skew.
  - The fix is engineering (I6e): log each scored leg's serve-time feature vector and compare it with the matrix row once the matrix catches up.
  - Expected gap effect: NFL total overstatement −32% at the point estimate, CI −42% to +11%. MLB cannot be measured on 3 days.

### Finding 14 (E8): training and serving time their archive inputs differently. Two of the differences are leaks; none moves the gap measurably today. FIX for consistency (I6f).

**(a) The book-leg cutoff.** This corrects phase 1, which implicitly read training's book leg as a closing quote.
- *Training* takes the latest quote per book at or before game-day 12:00 UTC: `TRAINING_LOOKBACK_HOURS = 8` before a fixed 20:00 UTC stand-in for game time (`helpers/archive.py:111–114`, `stats/base.py:2393–2401`). Matrix `QuoteObservedAt` is always before game time; MLB rows for 09-16 were observed at 09-16 01:55 UTC.
- *Serving* takes the latest quote before the decision.

| League | Quoted posted legs | Decision after the cutoff | Median offset | Fresher serving quote exists | Its median offset | Recommended, median offset |
|---|---|---|---|---|---|---|
| MLB | 42,702 | 99.8% | +10.9 h | 97.3% | +7.5 h | +10.9 h |
| NFL | 7,434 | 86.8% | +4.9 h | 57.7% | +4.5 h | +5.0 h |
| NHL | 351 | 100% | +13.0 h | 92.3% | +10.5 h | +14.0 h |
| WNBA | 1,350 | 93.0% | +11.0 h | 91.9% | +10.5 h | +11.0 h |

- **Effect on the information test (E11).**
  - Data: 6,199 held-out ladder-era test rows. Both references exist on 23.6%, and the median decision is 8.0 h after the cutoff.
  - Moving the market reference from the 12:00 UTC quote to the decision-time quote changes it by .0076 on average. That lowers MLB model-only b_model from .772 (.167) to .656 (.266), and fused from .962 to .847.
  - The later quote is the sharper reference: market log-loss goes .644 → .640. Betting prices keep moving toward the outcome between open and close (Moskowitz 2021, doi:10.1111/jofi.13082), so an earlier reference flatters the model.
  - NFL: −.072 → −.079.
  - About 0.1 of b_model is reference timing.

**(b) Quote age does not drive the gap.**
- **Quoted legs (51,837 posted).**
  - Median age is 2.48 h on all posted legs and 2.57 h on recommended legs. Posted legs are calibrated within ±1 pp in every age band.
  - Recommended gap by quote age: ≤ 1 h +18.0 pp (898 legs); 1–3 h +13.3 (238); 3–6 h +12.7 (297); 6–12 h +6.8 (287); > 12 h +13.4 (407).
  - Quotes ≤ 3 h old carry +17.0 [11.3, 22.7] on 1,136 legs; older quotes carry +11.3 [5.5, 15.5] on 991.
  - Quotes older than 3 h hold .367 [.229, .512] of the gap but .466 of the legs. The freshest quotes carry the largest gap.
- **Fallback legs.**
  - Here age *is* a selection variable. Median age is 0.50 h posted against 5.59 h recommended, and the selection rate rises from .0011 to .0252 across the age bands.
  - This is R1's stale-slot reversal: hit − implied of −0.063 on 199 legs.
  - Fallback edge flags stay the owner's decision (R1).

**(c) Training game lines include in-game quotes, and NFL's are mostly missing.**
- **Training.**
  - The features are `Moneyline`, `Total`, `OppTotal`, `Spread` and `GameTotal`. They come from the gamelog, enriched with the latest per-book archive quote at ingestion: `_enrich_team_markets`, `get_team_market_map(at=None)`, `stats/base.py:~630–690`.
  - A miss fills the moneyline with 0.5.
- **Serving** reads decision-time archive values (`_game_context`, upcoming branch, `stats/base.py:1568–1630`).
- **The MLB leak.** Over 4,276 MLB 2026 team-games, 77.9% of the latest per-book quotes were observed at or after 23:00 UTC on game day. Correlation with the team's runs:

  | Value | Total | Moneyline |
  |---|---|---|
  | Enriched (training) | .574 | .364 |
  | Archive-latest | .627 | .319 |
  | Noon-ET pre-game | .163 | .158 |

  The enrichment is picking up live in-game quotes. That is target leakage (Kaufman et al. 2012, doi:10.1145/2382577.2382579; Kapoor & Narayanan 2023, doi:10.1016/j.patter.2023.100804).
- **Exposure: share of rows at the 0.5 default.**

  | Rows | MLB | NFL |
  |---|---|---|
  | 2026 rows | 89.6% | 62.0% |
  | Held-out last 30% | 84.6% | 16.3% |
  | Dated ≥ 08-30 (today's scorecard window) | 0% (100% enriched) | 94.1% |

  NFL's problem is the miss default, not the leak: 9.5% of its latest quotes are late, and enriched ≈ pre-game (correlation .95–1.0).
- **Effect.** Replacing a matrix row's game lines with decision-time values moves P by sd .0053 (MLB) and .0168 (NFL). That explains R² .015 and .047 of the parity variance.

**(d) The opposing pitcher.**
- Training uses the realized opposing starter from the gamelog. Serving uses the probable pitcher from `upcoming_games`.
- This is unmeasured, because the serve-time probable pitcher is not persisted (I6e).

**(e) The MLB comp snapshot.**
- **Mechanism.** MLB comps come from Baseball Savant affinity CSVs that only `meditate` refreshes (`training/cli.py:644, 687` → `StatsMLB.update_player_comps`, `stats/mlb.py:887`). The files on disk are dated 2026-09-17 12:15, the last MLB retrain.
- **Training** gives every row back to 2025-04-01 the retrain-day, current-state snapshot. That is look-ahead, and the code flags it (`TODO(comp-leakage-mlb)`, `stats/base.py:760–769`).
- **Serving** uses the snapshot frozen at the last retrain.
- **Weight in the models.** Comp features carry 0.7–8.9% of MLB |SHAP|: hits+runs+rbi 8.9%, total bases 7.9%, runs 6.8%, walks and singles 2.5%.
- **Not the version-age mechanism.** The per-market age effect does not track comp share (Spearman −0.16 over 9 markets, Finding 15).
- **Effect: unmeasured.** Savant publishes current-state tables only.

**(f) The booster's training window (E9).**
- **Mechanism.** The final booster is fit on `X_train` only, the first 70% of matrix rows by date: `_TRAIN_FRACTION = 0.7` (`training/pipeline.py:150`), split at `:1116–1124`, fit at `:4915`. The last 30% is hash-split into validation (calibration) and test.
- **Cutoffs:**
  - MLB 2026-04-14 → 05-05 (19 cells; matrices start 2025-04-01);
  - NFL 2024-11-10 → 12-29 (20 cells; no NFL booster has seen a 2025 game);
  - WNBA 2025-06-21 → 06-29;
  - NHL 2025-11-18 → 12-31.
- **What a retrain refreshes.** It moves the cutoff by about 0.7 × the days since the last retrain. What it really refreshes is the calibration stage.
- **Age-versus-quality curve.** This is the ML Test Score's staleness test (Breck et al. 2017, Model 4: "The impact of model staleness is known"; Zinkevich Rule #8). It uses within-cell, model-only `P_standalone`, with cell fixed effects.
  - MLB, by days past the cutoff:

    | Days | b_model | λ_c |
    |---|---|---|
    | 0–30 | .35 (.05) | .55 |
    | 31–60 | .22 (.06) | .49 |
    | 61–90 | .37 (.08) | .66 |
    | 91–120 | .32 (.15) | .63 |
    | > 120 | .24 (.14) | .62 |

    Flat.
  - NFL, b_model:

    | Days past the cutoff | b_model |
    |---|---|
    | 0–60 | .01 (.08) |
    | 61–300 | .07 (.10) |
    | 301–600 (the 2025 season) | .03 (.06) |
    | > 600 (2026) | −.15 (.21) |

    ≈ 0 from the first day.
- **Verdict: KILL booster staleness as a mechanism.** Do not move `_TRAIN_FRACTION` for the gap. A fresher booster would not recover NFL's missing information, which is absent at every distance.

### Finding 15 (E9–E14): within cell, offline and live information agree; the offline-to-live drop is calendar plus reference timing; MLB version age is the largest live modifier

**Two estimator traps, now avoided.**
- *A through-origin slope of (y − m) on (p − m)* is contaminated whenever the market reference m is mis-levelled. NFL receptions m runs +9.4 pp and tds −13 pp. Phase 1's version-age reading (c15) was superseded on this ground.
- *Encompassing pooled across cells without fixed effects* credits the model with cross-cell level differences. For MLB model-only rows in August, it gives .78 pooled against .32 within cell.

Every number below is either the logistic encompassing coefficient b_model or the centered slope λ_c with an intercept, estimated with cell (or market) fixed effects: a cell-specific intercept and market slope, and one common b_model. CIs are day-clustered. Logistic encompassing for probability forecasts follows Clements & Harvey 2010 (doi:10.1002/jae.1097). Decoding m from odds follows Štrumbelj 2014 (doi:10.1016/j.ijforecast.2014.02.008).

| Population | MLB b_model | MLB λ_c | NFL b_model | NFL λ_c |
|---|---|---|---|---|
| Offline, model-only `P_standalone`, May–Aug (monthly range) | .23–.40 | .51–.69 | .03 (.04) over 2024-12..2025 | .18 (.05) |
| Offline, model-only, September / NFL 2026 | .18 (.13) | .11 (.15) | −.15 (.21) | −.02 (.17) |
| Offline, fused `P`, May–Aug | .49–.66 | .77–.92 | (contaminated) | — |
| Offline, fused `P`, September | .38 (.19) | .28 (.23) | (contaminated) | — |
| Live, served `Win Prob`, all post-fix | .26 (.07) | .21 (.07) | −.03 (.16) | −.04 (.11) |
| Live, quoted / unquoted | .39 (.18) / .13 (.10) | .35 / .10 | −.28 (.17) / +.25 (.14) | −.19 / +.19 |

The NFL fused-`P` rows are contaminated because `ev` and `under_prob` disagree in the NFL count quotes (memory `nfl_count_quote_columns_disagree`).

- **MLB.**
  - The live level (fused .26) matches offline September (.38 ± .19) within noise. The remaining ≈ .1 is the size of the reference-timing effect (Finding 14a).
  - Offline September is itself well below May–August: fused λ_c falls from .86 to .28 (z ≈ 2.4) and model-only from .59 to .11 (z ≈ 2.8).
  - The whole live MLB window falls in September.
- **NFL.**
  - It has no at-market information offline at any distance from the booster cutoff (Finding 14f), and none live.
  - Quoted NFL legs are anti-informative live, at −.28 (.17). Their matrix-path λ_c is +.001 (Finding 13), so part of the anti-information is parity and the rest is absence.
- **Conclusion.** The offline-to-live drop is mostly not a serve-time skew. It is the models' at-market information in that calendar window, minus about 0.1 for reference timing and minus the parity term.

**The September drop is not call-ups (E13).**
- Pooled b_model by the player's prior games in the cell's matrix, August → September:
  - fewer than 50: .66 (.10) → .35 (.19);
  - 50–149: .91 (.09) → .25 (.26);
  - 150 or more: .79 (.05) → .08 (.22).
- These are pooled within each history band, without fixed effects.
- Players first seen on or after 07-15 are 5.6% of September rows.
- The established players fall the most. The mechanism is open. Concept drift is the generic name (Gama et al. 2014, doi:10.1145/2523813); end-of-season rosters, rest days and pennant races are the MLB-specific suspects.

**Non-participation is not the information story (E10).**
- 1.3% of 52,710 MLB hitter legs had no plate appearance (Over hit .033, Under .981). They are 1.5% of recommended legs.
- b_model is .273 with them and .256 without.
- Excluding them raises the recommended gap from .0904 to .0972, because the ledger grades an Under that the platform would void as a hit.
- **KILL** as an information explanation. The grading defect stays with R1 Finding 11.

**Version age (E14; R1 Finding 7, replicated with a level-robust within-market estimator).** Age is the days since the version family's first served post-fix date.

| MLB recommended legs | n | Days | Read | Hit | Gap [95% CI] |
|---|---|---|---|---|---|
| Version days 0–3 (fresh) | 1,059 | 10 | .609 | .553 | +5.6 [+4.0, +8.3] |
| Version days 4+ (aged) | 2,349 | 21 | .612 | .477 | +13.5 [+9.4, +17.8] |
| Fresh − aged | | | | | −7.9 [−12.3, −2.9] |
| At √3.5: fresh / aged / difference | 1,704 / 3,549 | | | | +6.5 / +11.6 / −5.1 [−8.7, −1.1] |

- **Per retrain window.**

  | Version | Fresh (days 0–3) | Aged (days 4+) |
  |---|---|---|
  | `20260829` | +5.8 [2.6, 23.0] (08-31..09-03) | — |
  | `20260903` | +5.6 [3.9, 7.5] (09-03..06) | +11.1 [7.2, 15.1] (09-07..16) |
  | `20260917` | +5.6 [3.5, 12.0] (09-17..20) | +16.6 [9.0, 24.0] (09-21..) |

- **Posted legs.** Within-market b_model is .51 (.10) fresh against .16 (.08) aged; λ_c is .43 (.10) against .11 (.07).
- **Daily series.** On 09-13..17 the aged `20260903` gives daily estimates between −.15 and +.10. The fresh `20260917` gives .44, .66 and .66 on 09-18..20, then .18, .69, −.59, .15, .02 and .25.
- **Transition day.** On 09-17 two versions served the same calendar day: `20260903` at age 14 gives −.06 (.22), `20260917` at age 0 gives .62 (.46).
- **NFL** shows no age effect: fresh − aged is −0.3 pp [−13.4, +17.4], because its information is ≈ 0 at every age.
- **Probes.**
  - *Not a level drift.* Bulk read − hit is +0.2 to +0.8 pp in every age × side cell. The model's deviation from the market keeps the same size (sd .044–.055), and the selection rate is unchanged (3.0% against 3.3%). Aged deviations are just as large but carry less information.
  - *Not booster staleness* (Finding 14f).
  - *Not a stale serving gamelog.* Take 29,588 consecutive-day pairs of unquoted hitter legs. The day-over-day change in the served mean loads on yesterday's surprise at about the same slope fresh and aged (.011 against .014), so prod's `Stats.update()` keeps the gamelog current.
  - *Not the comp snapshot* (Finding 14e).
- **Verdict: exploratory and observational.** The owner scheduled the retrains; they were not randomized, so part of the fresh-day gain may be regression to the mean after bad runs. The pattern repeats across three windows and on a same-day transition, which makes chance unlikely. Only a paired same-leg comparison of two versions separates version from calendar (I6g). This is the age-versus-quality A/B with older models that Breck et al. 2017 (Model 4) recommend.

### Finding 16: how R1's regime map and R2's "the pricer is sound, the legs fail" change or confirm the ranking

**R1 confirms rank 1 and removes its serving route.**
- **The routing changes.** R1's K2 ablation shows that decision-time market observables encompass served p out of sample (logistic +0.00038 nats [−0.00011, +0.00088], p = .072).
  - Phase 1 routed the 89% to "R1 price-conditioned trust". That routing is withdrawn.
  - The selection residual can now shrink only if the models carry more information at the market (I6d to I6g). Without that, the per-leg selected gap stays.
- **The regime map survives a level-robust estimator.** R1 used the through-origin λ_all. Live per-market λ_c (c21) gives the same broad ranking: the contact markets (singles, walks, hits) lead and hits+runs+rbi trails in both. rbi and runs read higher here.

  | Market | λ_c (this brief) | R1's λ_all |
  |---|---|---|
  | singles | .64 | .57 |
  | walks | .51 | .59 |
  | hits | .41 | .38 |
  | rbi | .36 | .10 |
  | runs | .28 | .16 |
  | hits allowed | .16 | .23 |
  | total bases | .14 | .19 |
  | hits+runs+rbi | .085 | .00 |
  | NFL overall | −.005 (.11) | −.00 / .07 |

  Within NFL, carries sit at −.49 (.13) and rushing yards at −.30. R1's protect list is where the information lives: MLB hits, singles and walks; the ≤ 1.5 payout band; main lines; quoted MLB Unders. Every I6 retrain must hold those cells' within-market b_model.
- **R1 target 1: the 2.0–2.5 band and alt rungs ("decode and tail calibration").** Not a decode defect: decode against real rungs has MAD 2.4 pp and log-loss difference +0.003 [−0.003, +0.009] (C4). Not a tail-shape defect either (C5). The anti-information there is selection on disagreement. The only calibration item with a measurable effect is the T ridge, at −1.0 to −1.2 pp (rank 2).
- **R1 target 2: NFL quoted Unders.** Not the leak (C6) and not booster staleness (Finding 14f). Partly serve-path parity: matrix-path λ_c is +.001 against −.149 live (Finding 13). The rest is absent information. Route: I6e, then I6d.
- **R1 target 3: MLB hits+runs+rbi and runs allowed.** Not combo-sum quotes (C2: 0.3% exposure). The served mean is compressed against the market: slope .44 against .77 for hits+runs+rbi, .30 against .81 for runs allowed (E5). That is model information. MLB is out of season, so this is 2027 work.
- **R1 Finding 7, version decay.** Confirmed live with the within-market estimator (Finding 15). Booster staleness cannot explain it. It is the largest live modifier this brief finds.
- **R1 Finding 11, DNP grading.** KILL as an information explanation (c22). It stays a ledger fix.

**R2 confirms rank 1 and fixes the parlay acceptance.**
- **The legs fail.** Admitted legs read 8–11 pp above their hit rate (hit / `Win Prob` 0.82–0.88). Realized/priced is 0.71–0.77 at two legs and 0.41–0.44 at five, and book marginals calibrate every size. That is rank 1, compounded.
- **The beam selects twice.** Its Model EV ≥ 2.0 floor is a second selection on the same disagreement: realized/priced falls from .50 to .00 across the priced-EV bands.
- **What the band requires.**
  - Under independence the k-leg joint ratio is roughly r^k, where r = hit/read per admitted leg.
  - R2's [0.9, 1.1] band therefore needs r ≥ 0.949 at two legs and r ≥ 0.979 at five. At a read of .61 those are per-leg overstatements of ≤ 3.1 pp and ≤ 1.3 pp.
  - No item in ranks 2–17 gets there; the T ridge moves r by about 0.02. Fresh-version MLB legs (r = .553 / .609 = 0.91) would still miss at two legs (0.82).
  - I6's parlay acceptance is R2's joint ratio on R2's harness: synthetic pool B and the beam re-run, in `scratchpad/r2/`. It cannot pass until rank 1 moves.
- **The convention change does not move the ranking.** The decomposition (c20) and the version-age contrast (c29) both hold at R2's √3.5.

## 5. Ranked defect table and routing protocol

**Holm (1979) across the pre-registered inferential tests** (m = 9; p from paired day-clustered bootstrap draws, 4,000 resamples; floor 1/4,000):

| Test | Raw p | Holm-adjusted p |
|---|---|---|
| Bulk miscalibration (decomposition) | < .0003 | .002 |
| Selection residual (decomposition) | < .0003 | .002 |
| C5 τ fit on authentic modal-line rows (prototype) | .049 | .343 |
| C5 τ cross-fit on DFS rungs (prototype) | .072 | .432 |
| C3 fractional tie label (live re-selection) | .263 | 1.000 |
| C1 unquoted pooling, w = 1 (live re-selection) | .270 | 1.000 |
| C4b rung quote on unquoted count rows (live) | .320 | 1.000 |
| C4 decode vs real rung (log-loss) | .332 | 1.000 |
| C6 NFL Under gap = matchup leak | .335 | 1.000 |

C2 (exposure) and C7 (descriptive) carry no p-value. The exploratory E1–E14 are reported with their own CIs and are not Holm-corrected. Of those, only E14 (MLB version age) has a selected-tail CI that excludes 0, and its pre-registered confirmation is I6g.

**Ranked by measured contribution to the post-fix selected-tail gap.** The gap is +12.5 pp on 6,138 legs, +13.2 pp on the 5,523-leg walk-forward subset and +10.9 pp on 8,209 legs at √3.5.
- The gated families in `.claude/research_gated.txt` are `*/skew_normal.py`, `*/skew_normal_centered.py`, `*/double_poisson.py` and `*/combined_markets.py`. No fix below touches them unless stated.
- Rows 1a–1d are mechanisms inside rank 1. They overlap and are not additive.

| Rank | Defect | Share of gap [95% CI] | Mechanism | Fix (side) | Files touched | Retrain | Expected effect | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **Selection on model–market disagreement (optimizer's curse) where the served mean carries little at-market information** | **≈89%**: selection residual +11.7 pp [9.2, 14.4]; 87% at √3.5. Overlapping localizations: harder-than-consensus lines 34% [17, 43] of the covered gap; compression side 61% of NFL's gap, 51% of MLB's | The edge rule picks the largest gaps between served p and the platform price; where the model lacks information those gaps are model error. Present with the model removed (+10.3 pp, E6). R1 K2: market observables encompass served p, so no serving layer can take it | *Training only:* raise at-market information (1a–1d). Every model stays served | I6d, I6e, I6f, I6g (below) | Yes | The per-leg gap closes only as information is gained | **ROUTE to I6 (model information)** |
| 1a | NFL carries no at-market information | NFL holds 49.2% of the gap; within-cell b_model .03 (.04) offline, −.03 (.16) live | Not the leak (C6), not booster staleness (14f); partly parity (13); served mean compressed against the market (E5) | *Training:* NFL information research after the I6e parity repair. Accept on fixed-effect encompassing (b_model CI lower bound > 0 on held-out quoted player-games at the decision-time reference) plus the tail scorecard | NFL lane; `training/pipeline.py` (extract helpers, never grow it); NFL feature builders | Yes | Unknown: research project | **RESEARCH (I6d)** |
| 1b | MLB version age (exploratory, E14) | Aged − fresh +7.9 pp [2.9, 12.3]. If causal: 49% [18, 77] of MLB's overstatement, about 25% [9, 39] of the cohort's | Posted-leg information falls from .51 to .16 after day 3, in all three retrain windows. Not level drift, booster or gamelog staleness, or comps | *Training/ops:* the I6g paired test first. Re-serve each logged leg (I6e) through the prior pickle set offline, giving a paired same-leg age-versus-quality curve (Breck et al. 2017, Model 4). If it confirms, retrain every 3–4 days | An offline script over the I6e log; keep the prior pickle set for 14 days in a backup tree. No serving change beyond I6e | Cadence only | If causal, the aged-day MLB gap goes from +13.5 to about +6 | **TEST (I6g)** before any build |
| 1c | Serve-path parity (E7) | NFL covered overstatement 0.68× [0.58, 1.11], i.e. −32% [−42, +11]; MLB unmeasurable | Same pickle and inputs except the features: sd(ΔP) 2.8–5.9 pp; served-mean correlation .34 (completions) to .98 | *Serving/engineering:* log the serve-time `expected_columns` slice per scored leg; run a nightly parity check against matrix rows; repair feature by feature | `prediction/scoring.py` (`_score_market`), a new diagnostics parquet and script | No (a repair may need one) | Upper bound: −32% of NFL overstatement | **FIX (I6e)** |
| 1d | Book-leg reference timing (E11) | ≈0.1 of b_model (MLB model-only .77 → .66); no direct gap measure | Training quote ≤ game-day 12:00 UTC; serving decides +5 to +11 h later | *Training:* align the matrix book leg to decision time where ladder polls exist; otherwise record the offset | `helpers/archive.py:111–114` (`TRAINING_LOOKBACK_HOURS`), `stats/base.py:2393–2401` | Yes | Small; makes offline information honest rather than raising live information | **FIX for consistency (I6f)** |
| 2 | **Bulk miscalibration** | **≈11%**: +1.5 pp [0.8, 2.1]; NFL 2.4, Under 1.8 | T pinned near 1 by the `0.01·(T−1)²` ridge (E1); calibration fit on a validation population unlike live (C7); tie label (C3) | *Training:* remove the ridge or set its weight by CV; fit T and PROB_STAGE on a population that includes tail-scorecard DFS-rung rows, reporting both; fractional push labels | `training/pipeline.py:2841` (`_brier_temperature_loss`), `:3194` (`_step_calibrate_temperature`), `:3204`/`:5002` → `training/labels.py`; `training/posthoc.py` (soft-label Platt needs sample weights) | Yes (calibration stage) | −1.0 to −1.2 pp gap, −27 to −31% volume, overstatement −33 to −37%; bulk log-loss +0.001 to +0.002 | **PARTIAL FIX** (hygiene, small gain, owner trade-off) |
| 3 | C3 tie label (`Result >= Line` = Over) | 1.7%: −0.21 pp [−0.52, +0.18] | Pushes counted as Overs in the T / posthoc fits of integer-line cells | *Training:* I6a shared label helper with y = ½ at a push (matches the serving half-split) | `training/pipeline.py:3204`, `:5002`; new `training/labels.py` | Yes | ≤ 0.2 pp; NFL targets Over read −9.7 pp | **FIX for correctness only** |
| 4 | C4b no book leg on unquoted rows where a real rung exists | 2.1%: −0.26 pp [−0.96, +0.13] | Rung covers 19.5% of unquoted count recommended legs at decision time | *Serving:* admit decision-time sportsbook rungs as quotes | `prediction/book_quotes.py` | No | Volume −15% on covered legs; per-leg gap unchanged | **KILL** as gap lever (optional volume lever) |
| 5 | C1 unquoted pooling skew | 1.4%: −0.18 pp [−0.60, +0.12] | Generic book shape pooled into unquoted rows at serve time but not in training | *Training:* I6b, align training's non-authentic fusion to serving (serving is the better convention at rungs: 10.11 vs 10.65 pp) | `training/pipeline.py:~3522, ~3668, ~3766` | Yes if aligned | ≈0 | **KILL** (hygiene only) |
| 6 | C4 `Market Prob` decode vs real rung | 0%: log-loss diff +0.003 [−0.003, +0.009] | Decode agrees with real rungs (MAD 2.4 pp) | none | — | — | — | **KILL** |
| 7 | C2 combo-sum quotes | ≈0.3% (6 legs) | Direct quotes exist for ~all combo legs | Optional shape alignment | `prediction/book_quotes.py`; **research-gated** if it reaches `combined_markets.py` | No | ≈0 | **KILL** |
| 8 | C5 tail position / shape | n/a (75% of the gap at \|z\| < 0.5) | Not a tail-shape defect; price-blind τ bounded by bulk | Folded into rank 2 | — | — | — | **KILL** (shape) |
| 9 | C6 NFL Under bias = matchup leak | Under gap +1.5 pp [−2.4, +13.3] *after* the leak removal | Leak era is not worse; the bias is ranks 1/E5 (compression against the market) | Folded into rank 1a (I6d) | — | — | — | **KILL** as explanation |
| 10 | **Fallback class (book_fallback; main-session row)** | All-time +17.8 pp on 4,463 legs, 99.1% pre-fix. Signed-distance shares: harder ≤ −1 22.2%, at consensus 38.4%. Alt 16.6% of class gap at +29.3 pp vs main +16.4 | Pre-fix cross-line mean-averaging decode (fixed `6c21665c` 2026-08-29, `8bebd0e3` 08-30). Post-fix decode calibrated at every distance band (8,102 posted legs); 38 recommended | None for the decode. The model-removed curse (+10.3 pp, E6) is rank 1 | — | — | — | **FIXED**; residual = rank 1. Quote age is a selection variable on fallback rows (Finding 14b); their edge flag is the owner's call (R1) |
| 11 | SkewNormal `step` parity | ≈0 (55 recommended NFL fantasy-underdog legs) | Training's SkewNormal `get_odds` omits `step`; serving passes the pickle's (1e-15 to 0.5 on fantasy, minutes and NHL skater cells) | *Training:* pass `step` in `_step_compute_test_probabilities` and `_step_calibrate_temperature` | `training/pipeline.py` (not the gated `skew_normal.py`) | Yes for affected cells | ≈0 | **FIX for parity** |
| 12 | Training game lines carry in-game quotes; NFL enrichment mostly misses (E8) | P effect sd .005 (MLB) / .017 (NFL); parity R² .015 / .047 | `_enrich_team_markets` takes the latest per-book quote at ingestion: 77.9% observed at or after 23:00 UTC, and the total correlates .57 with runs against .16 pre-game. NFL: 62% of 2026 rows at the 0.5 default | *Training:* an as-of pre-game cutoff (`get_team_market_map(at=…)`); find the NFL enrichment miss | `stats/base.py:~630–690`; `helpers/archive.py:714–800` | Yes | Small today; grows as the archive fills (MLB late-season rows are 100% enriched) | **FIX (I6f, correctness)** |
| 13 | MLB comp snapshot: every training row uses the current-state Savant affinity; serving uses the one frozen at the last retrain | Unmeasured; comp features 0.7–8.9% of MLB \|SHAP\| | Look-ahead, flagged `TODO(comp-leakage-mlb)` | *Data:* archive the daily affinity CSVs from now on; rebuild point-in-time once a season of snapshots exists | `stats/mlb.py:887`; `training/cli.py:644, 687` | Later | Unknown | **FIX when the data exists (I6f)** |
| 14 | Opposing pitcher: realized in training, probable at serve | Unmeasured | The serve-time probable pitcher is not persisted | *Serving:* persist it with the I6e log, then measure | `stats/base.py:1568–1630` | — | — | **MEASURE (I6e)** |
| 15 | Booster window = first 70% of matrix rows (NFL ends December 2024) | n/a: no decay with distance in either league | Final booster fit on `X_train` only | None for the gap | — | — | — | **KILL** as a gap lever |
| 16 | Book-quote staleness on model rows | Gap ≤ 3 h +17.0 against > 3 h +11.3 | The freshest quotes carry the larger gap | None | — | — | — | **KILL** (fallback staleness stays the owner's call, R1) |
| 17 | Non-participation (no plate appearance) legs | Gap .090 with them, .097 without | Grading, not information | Ledger (R1 Finding 11) | — | — | — | **KILL** as an explanation |
| — | C7 evaluation design | n/a | g1 scores a population where the model agrees with the market (mean \|model − market\| .050 vs .120 live) | **BUILD** the tail scorecard (§6) as a training-time diagnostic, not a gate | `scripts/tail_scorecard.py`, `tests/` `-m diagnostics` | No | Lets every rank-1/2 change be judged on the selected tail before it ships | **BUILD** |

**Routing onto the plan's I6 items.**
- **E** (tail scorecard): build first, with the six §6 deltas. It is the only offline instrument that reproduces the defect.
- **I6a** (labels): go, with the fractional label. Hygiene; ≤ 0.2 pp.
- **I6b** (fusion consistency): align training to serving, or leave it. No gap effect either way.
- **I6c** (tail recalibration at DFS rungs): reframe as "remove the T ridge or set its weight by CV, and fit the calibration stage on a live-like population". Every price-blind or distance-conditional variant measured here closes ≤ 1.2 pp.
- **I6d** (NFL retrain on the post-leak era): keep it as a research project.
  - NFL has no at-market information at any distance from the cutoff, so a retrain on post-leak rows alone is unlikely to create it.
  - Order: the I6e parity repair first, then new decision-time inputs.
  - Acceptance: fixed-effect encompassing against the decision-time reference, plus the tail scorecard. g1 passes every NFL volume cell today and cannot judge this.
- **I6e** (new; engineering, no retrain):
  - For each scored leg, persist the serve-time feature vector, `Model Skew`, `Step`, the probable pitcher and the decision-time book leg. Write them to a diagnostics parquet, never `history.parquet` or the dashboard snapshots.
  - Run a nightly parity job against matrix rows once they exist, alarming on per-cell sd(ΔP) > 1 pp.
  - This is Zinkevich Rule #29 and Breck Monitor 3. It also unblocks the live SkewNormal re-serve and I6g.
- **I6f** (new; training-side, needs a retrain):
  - an as-of pre-game cutoff for game lines in `_enrich_team_markets`;
  - repair of the NFL enrichment miss;
  - a book-leg cutoff aligned to decision time where ladder polls exist;
  - archiving the daily Savant affinity snapshot.

  This is correctness work with a small expected gap effect; it makes offline numbers comparable to live ones.
- **I6g** (new; a pre-registered test, then possibly ops):
  - Keep the prior pickle set for 14 days after each retrain. Re-serve the same legs offline from the I6e feature log (preferred; no serving change beyond I6e), or shadow-score them at serve time into the I6e diagnostics parquet.
  - Primary statistic: fresh (days 0–3) minus aged (days 4+) recommended gap, day-clustered, on NBA and NHL from late October. Secondary: within-market b_model, and the paired same-leg difference between versions.
  - If confirmed, retrain every 3–4 days. This is training-side and pulls no model. If the paired difference is null, KILL it: the MLB pattern was calendar or regression to the mean.
- **Parlays (R2).** I6's parlay acceptance is R2's realized/priced joint ratio in [0.9, 1.1] at every entry size, measured on R2's harness. It needs per-leg r ≥ 0.95 (two legs) and ≥ 0.98 (five legs). No I6a–f item reaches that alone.
- **The plan's I6 success number** ("the selected-tail gap halves on the first clean window after a retrain") is met by version age alone: MLB's fresh-day gap is 41% of the aged one. Compare like ages, fresh against fresh, or the metric certifies the retrain calendar.

**Do not build:**
- a trust layer of any kind (R1 KILL);
- w = 1 serving for unquoted rows (C1);
- price-blind or distance-conditional recalibration as a gap fix (C5, E2, E3);
- rung admission as a gap lever (C4b; it remains a volume lever);
- combo-shape alignment (C2);
- a fixed scalar shrink toward the book (killed);
- a staleness gate on model rows (Finding 14b);
- a `_TRAIN_FRACTION` change for the gap (Finding 14f);
- any pull, demotion or withholding of a model, or a model-free engine.

The fallback edge flag is the owner's decision (R1).

## 6. Tail-scorecard specification (`scripts/tail_scorecard.py`, tests under `-m diagnostics`)

*Revised after phase 2 with six marked deltas, Δ1–Δ6. Nothing else in this section changed.*

**Purpose.**
- Replay the live recommendation rule on *held-out* test rows, at the DFS rungs the ladder actually held at the time.
- Report the selected-tail gap per cell, with day-clustered CIs.
- This is a training-time diagnostic and acceptance input for rank-1/2 work, **not a ship gate**.
- It is the instrument g1 lacks (Finding 8). The prototype reproduces the live gap from held-out rows (+10.1 [3.5, 13.9] against live +13.2 [10.6, 15.9]).

**Inputs** (all read-only):

1. **`data/test_sets/{LEAGUE}_{market}.csv`.**
   - Required columns: `Player`, `Date`, `Result`, `Line`, `EV`, `Book_EV`, `P`, `Odds`, `QuoteAuthenticity`. The information test also needs `P_standalone`, the model-only P at the row's line, which every dumped test set carries today. *(Δ1)*
   - Model-only columns by family:

     | Family | Columns |
     |---|---|
     | SkewNormal | `SN_Sigma_model`, `SN_Alpha_model`; plus `Gate` when `hist_gate > GATE_PUBLISH_THRESHOLD` |
     | NegBin | `R_model` |
     | ZINB | `R_model`, `Gate_model` |
     | DPO | `DP_PHI_model` |

   - A cell without these columns is listed as "not reconstructable — re-dump" and never scored. Today that is NBA and most NHL.
2. **`data/models/{cell}.mdl`.**
   - Fields used: `distribution`, `weight`, `cv`, `dispersion_cal`, `skew_cal`, `hist_gate`, `step`, `temperature`, `posthoc`, `posthoc_blob`, `pit_recal_blob`, `model_version`.
   - Structural-strategy cells (two-part / affine) are skipped and listed.
3. **Archive `ladder`.**
   - Open with `duckdb.connect(path, read_only=True)` or `SPORTSTRADAMUS_ARCHIVE_READ_ONLY=1`.
   - For each test row's (league, market, game_date, entity), take every (platform ∈ {Underdog, Sleeper}, line) rung. Use `p_over` at that platform's last poll of the line, `arg_max(p_over, observed_at)`, and keep `last_poll`.
   - Sportsbook rungs: books outside the DFS list, at the same line, observed ≤ `last_poll`. Take the latest per book, then the mean `p_over` and `n_books`.
4. **Archive `odds`.** Decision-time consensus line: the median over sportsbooks of each book's last `line` observed ≤ `last_poll`. It gives the signed DFS-vs-consensus distance on the bet side.
5. **Optional `data/runtime/history.parquet`.** Posted payouts where (Date, Player, Market, Line, Platform, side) matches. On Underdog the payout is `Boost` × `UNDERDOG_BOOST_BASELINE`, imported from `sportstradamus.helpers` and never written as a literal. It is 1.78 today and becomes √3.5 = 1.8708 when R2's I5a-1 lands. *(Δ3)*

**Steps.**

1. **Reconstruction self-check (blocking per cell).**
   - Re-serve every test row at its own line with the *training* convention: `w_row = weight` on authentic rows and 1.0 otherwise, `ev_b = Book_EV`.
   - The chain is `fused_loc` → `correct_fused_mean` → (c, s) → `get_odds` → `apply_cdf_recal` → `apply_temperature` → PROB_STAGE `apply_posthoc`. Count families pass the pickle `step`; SkewNormal passes no `step`, as training does.
   - Require max |P_rebuilt − P| ≤ 1e-6; the prototype achieved ≤ 1.1e-13 on 48 cells. A failing cell is excluded and printed, never silently scored.
2. **Reprice at each rung with the *serving* convention.**
   - w everywhere; `ev_b = Book_EV` on authentic rows, else the model `EV`; SkewNormal with the pickle `step`.
   - This is the same function with two arguments switched. Emit both conventions per cell, which is the C1 contrast.
   - Optional third arm: the sportsbook rung, decoded with `get_ev(line, 1 − p_rung, cv, dist, gate=book_gate(...))`, as the book leg where one exists (C4b).
3. **Payout.**
   - Use the posted payout where history matches, and flag the row `posted`.
   - Otherwise use 1 / (p_dfs,side × OR_platform) and flag it `assumed`. OR is the two-sided median from the ladder over the scored window: today Sleeper 1.128 and Underdog 1.178, with 2/`UNDERDOG_BOOST_BASELINE` at Underdog's standard rung. *(Δ3)*
   - Report the posted / assumed share per cell.
4. **`Market Prob`.**
   - Authentic rows: the decode at the rung line, via the same `book_over_prob` path serving uses (`book_skewnormal_shape` / count dispersion).
   - Other rows: the payout-implied side probability, which is the live fill.
5. **Live-rule replica.** Import the constants; never re-type them:
   - `prediction/offer_records._MAX_CONFIDENCE` (0.90 clip), with the side chosen by argmax;
   - `realized.RECOMMENDED_EDGE_MIN` (0.05) applied at the payout;
   - payout in (1, `MAX_FAVORED_PAYOUT` = 2.5];
   - `UNQUOTED_BOOK_DISAGREEMENT_MAX` (0.15) on unquoted rows;
   - `_MAX_OFFERS_PER_PLAYER` (3), keeping those closest to the standard payout.
   - The fixture test asserts row-for-row agreement with `finalize_records`.
6. **Outcomes.** Drop pushes, since live voids them; y = `Result > Line` at the rung.

**Outputs.**
- **Destination.** A sandbox CSV, default `/tmp/tail_scorecard.csv`, plus a printed summary. Never `model_stats.parquet`, following the `scorecard` CLI's sandbox rule.
- **Granularity.** Per cell, per league and overall.
- **Counts.** Test rows, rung rows and selected rows.
- **Selected tail.** Read, hit, market, payout, and the gap with a day-clustered 95% CI (≥ 2,000 multinomial day resamples). Total overstatement, Σ(read − hit).
- **Bulk on all rung rows.** Log-loss, and read − hit.
- **Splits.**
  - side;
  - quote class (authentic / derived / synthetic / pickem);
  - main vs alt rung (rung line == test line);
  - signed consensus-distance band (≤ −1, (−1, 0), 0, (0, 1], > 1, none);
  - |z| band;
  - decision-time sportsbook-rung coverage.
- **Information test.** This is the per-cell answer to "does the model know anything at the money". Run it on test rows with a real quote (`QuoteAuthenticity == "authentic"`).
  - *Forecast:* the model-only columns. That is the mean `EV` (pre-blend) and `P_standalone` at the row's line. Never use `Blended_EV` or the fused `P`, which already contain the book leg; NFL's fused numbers are contaminated by the `ev`/`under_prob` disagreement (Finding 15). *(Δ1)*
  - *Market reference, primary — decision time:* `t_dec` is the Underdog/Sleeper last poll of that (league, market, game_date, entity). Use each sportsbook's latest `under_prob` at the row's line observed ≤ `t_dec`, averaged over books, and the decision-time consensus line (Input 4). *(Δ2)*
  - *Market reference, secondary — training time:* `1 − Odds` and `Book_EV`, the 12:00 UTC quote. Report both references. Their difference is the reference-timing skew: MLB model-only b_model is .77 against the 12:00 UTC quote and .66 against the decision-time one (Finding 14a). *(Δ2)*
  - *Estimators, each with an intercept:*
    - linear: Actual ~ a + b_model·`EV` + b_market·line;
    - logistic: y ~ a + b_market·logit(p_market) + b_model·logit(`P_standalone`);
    - the log-loss ablation: the logistic fit with the model term against the fit without it.

    Use day-clustered CIs with ≥ 2,000 day resamples. Never use a through-origin slope of (y − m) on (p − m): it is level-contaminated whenever the market reference is mis-levelled, as the NFL count quotes are. *(Δ1)*
  - *League and overall rows:* pool with cell fixed effects (a cell-specific intercept and market slope, one common b_model). Without them the pooled fit credits the model with cross-cell level differences: MLB model-only August reads .78 pooled against .32 within cell. *(Δ1)*
- **Diagnostics.** Both fusion conventions side by side, reconstruction max error, and posted / assumed payout share.
- **Reproduction line.** Live selected gap for the same cell and window, from `realized` (`settled_offers` when it lands; until then the I1a rule in `realized._offers_by_split`). Split it by Model Version age: days 0–3 since the version's first served date against 4+. Live MLB gaps are +5.6 and +13.5 pp in those bands (Finding 15), so an unsplit comparison mixes retrain timing into the agreement guard below. *(Δ4)*

**Routing and guards.**
- **Who reads it.** The NFL lane (I6d), I6c, I6g and any rank-2 calibration change, as an acceptance input. R1 is KILLed and no longer a reader. *(Δ6)*
- **Promotion.** Promote into a sweep objective only after ≥ 4 weekly runs show per-cell agreement with live, measured as the rank correlation of the per-cell selected gap. This is the Goodhart guard: proxies diverge under search (memory `proxy_goodhart_under_search`).
- **Power.**
  - Today the test half overlaps the ladder era from 2026-08-30 only: 40,568 rung rows over 32 cells, 1,615 selected, 23 days.
  - Report n and CI width, and treat cells with < 100 selected rows as descriptive.
  - Power grows with every retrain whose test window sits inside the ladder era.
- **Known optimism.** Scorecard rows use training-matrix features. Three measured feature skews make them optimistic relative to serving:
  - serve-path parity (Finding 13);
  - in-game quotes in enriched game lines, present in 100% of MLB and WNBA rows in today's window, with a P effect of sd ≈ 0.5 pp (Finding 14c);
  - MLB's retrain-day comp snapshot (Finding 14e).

  Read the selected gap as a lower bound on the live gap. The prototype gives +10.1 against live +13.2. *(Δ5)*
- **Diagnostics tests** (`-m diagnostics`):
  - (a) the reconstruction self-check passes on a fixture cell;
  - (b) the live-rule replica matches `finalize_records` on a fixture frame;
  - (c) the archive is opened read-only (assert the connection mode).
- **CLI.** `click`, with a `tqdm` bar over cells (CLAUDE.md rules).

## 7. What was tried and failed (everything run)

| Arm | Population | Result | Verdict |
|---|---|---|---|
| C1 w = 1 on unquoted count rows (live, exact) | 34,389 posted / 1,185 recommended | read moves the wrong way (+0.26 pp); cohort Δ −0.18 [−0.60, +0.12] | failed |
| C1 offline both conventions | 48,647 non-authentic test rows; 40,568 rung rows | ΔP −0.00036; selected 10.11 vs 10.65 pp | failed |
| C3 fractional / drop-push label, refit on the test half, mapped to live | 6,138 recommended | Δ −0.21 / −0.06 pp, CIs span 0 | failed as gap lever |
| C4 decode vs real rung | 2,393 recommended with a decision-time rung | log-loss diff +0.003 [−0.003, +0.009] | no defect |
| C4b rung as book leg (live count, exact) | 212 recommended | volume 212 → 47, per-leg gap 21.6 → 20.1 pp; cohort Δ −0.26 | volume only |
| C4b prototype (all families) | 40,568 rung rows | +9.97 → +10.27 pp | failed |
| C5 τ on authentic modal rows (prototype) | 1,615 selected | −2.95 pp [−5.27, −0.03], Holm .34; bulk log-loss .5750 → .5815 | fails Holm; bulk cost |
| C5 τ cross-fit on rungs (prototype) | same | −0.82 [−1.48, +0.10] | failed |
| C6 leak era vs post-leak | 10,004 NFL Under legs | gap rose after removal | refuted |
| E1 unpenalized τ, live walk-forward (all / authentic / rungs) | 6,138 recommended | −1.03 / −1.20 / +0.20 pp; bulk log-loss worse in all three | small, trade-off |
| E2 posted-leg isotonic served directly | 5,523 recommended | +1.28 pp (worse) | failed |
| E3 distance-conditional walk-forward Platt | 4,116 covered recommended | removes harder band; Δ −0.84 [−3.31, +1.44] | failed (re-forms) |
| E5 price-blind mean de-compression | 5,936 recommended | Δ −0.50 [−1.06, +0.05] | failed (compression only visible against the market) |
| E6 model-removed (book decode) selection | 51,911 quoted posted | 374 selected, +10.3 pp [0.4, 20.8] | the curse needs no model |
| E7 serve-path parity, same pickle (c16, c16b) | 9,382 posted legs, 13 days | sd(ΔP) 2.8–5.9 pp; NFL overstatement 0.68× [0.58, 1.11]; log-loss live − matrix +.004, CIs span 0 | real skew, effect inconclusive → I6e |
| E8 decision-time game lines into matrix rows (c17) | parity legs | R² .015 (MLB) / .047 (NFL) of parity variance | small |
| E8 in-game game-line leak (c18) | 4,276 MLB team-games | 77.9% of latest quotes at or after 23:00 UTC; total vs runs .57 enriched, .16 pre-game | leak confirmed; small exposure today |
| E8 quote age (c14) | 51,837 quoted posted / 2,127 recommended | ≤ 3 h +17.0 vs > 3 h +11.3; bulk calibrated in every band | staleness is not the driver |
| E8 decision time against the 12:00 UTC cutoff (c19) | 51,837 quoted posted | MLB decides +10.9 h after the cutoff (median), NFL +4.9 h | timing skew confirmed |
| E9 information by distance from the booster cutoff (c19, c27) | held-out test rows | MLB flat (.22–.37); NFL ≈ 0 at every distance | booster staleness KILL |
| E10 no-plate-appearance legs (c22) | 52,710 MLB hitter legs | b_model .273 vs .256; gap .090 vs .097 | KILL |
| E11 reference timing (c23) | 6,199 ladder-era test rows | MLB model-only b .77 → .66; NFL −.07 → −.08 | ≈0.1 of b_model |
| E12 within-cell information by month, offline and live (c24, c26) | 98k MLB and 12k NFL test rows; 121k live legs | MLB offline fused May–Aug .49–.66, September .38, live .26; NFL ≈ 0 | calendar, not skew |
| E13 September call-ups (c25) | 4,073 September rows | established players fall most (.79 → .08) | refuted |
| E14 version age (c28b, c29, c29b) | 105,914 posted / 3,408 recommended MLB legs | fresh +5.6 vs aged +13.5 pp; b .51 vs .16 | replicated; exploratory → I6g |
| E14 probes: level drift (c31), gamelog staleness (c30), comp snapshot (c32) | MLB | bulk ≈ 0 in both bands; served mean loads on yesterday's surprise at .011 vs .014; Spearman −0.16 | all refuted |
| c15 through-origin λ by test-row age | test rows | λ ≈ 1 driven by level offsets | estimator superseded (c19, c26) |
| c20 decomposition at √3.5 | 8,209 walk-forward legs | selection 86.9% | robust |

**Not tried:**
- A joint C1 + C3 + C4b removal. The sum of the individual Δ, −0.65 pp, bounds the joint effect, and the arms overlap only on unquoted count rows.
- A live SkewNormal re-serve. `Model Skew` is not persisted in history; SkewNormal carries 50.7% of the gap, so its skews are measured only offline and on the prototype. Persisting `Model Skew` (and the step) in `history.parquet` would close this blind spot.
- NBA, which is not in the post-fix sample.
- A booster refit through the validation window. Pickle writes are out of scope, and E9 shows no distance decay to recover.
- A paired same-leg comparison of two versions. It needs serve-time features (I6e) or shadow scoring (I6g).
- Distribution-family changes, shape-borrow, book-dist and scalar book shrink, all excluded by the dispatch.

## 8. Reality checks

- **Size of the skew fixes.**
  - All calibration-chain and quote skew fixes together are worth ≤ 0.7 pp of a 12.5 pp gap.
  - Phase 2's feature and timing skews add one inconclusive NFL term (parity, −32% [−42, +11] of NFL overstatement) and small ones (game lines, reference timing).
  - Do them for correctness: I6a, I6b, I6e, I6f and step parity. Never treat them as the gap programme, and never sell a retrain built on them as "fixing the Receipts".
- **The T-ridge fix is real but small.**
  - It is worth about −1 pp on the selected tail, at a 0.001–0.002 cost in bulk log-loss, concentrated in NFL.
  - The live estimate adds an extra τ on top of the served chain. That is exact for T-only cells and a proxy where a PROB_STAGE map follows T.
  - The ridge is a regulariser silently overriding the data, which is a defect regardless of payoff.
- **The 89% is not a skew, and no layer can take it.**
  - Nothing in this brief closes it without using the price. R1 has shown that, once the price is used, the model adds nothing (K2), so a price-aware layer is the model-free engine the owner ruled out.
  - What remains is information: the models must know more than the market at the decision. That is a research project with no guaranteed method (below).
  - Product consequence: until information rises, the per-leg selected gap stays near +10 to +13 pp, and parlays compound it (R2).
- **The one large live effect, version age, is observational.**
  - The owner scheduled the retrains; they were not randomized. If retrains follow bad runs, part of the fresh-day gain is regression to the mean.
  - Three windows and one same-day transition make chance unlikely. Only a paired same-leg comparison (I6g) separates version from calendar.
  - Until then, the projected cadence gain (aged-day MLB gap +13.5 → about +6) is a bet, not a measured lever.
- **Research project vs engineering project.**
  - *Engineering, known methods, cheap:* the tail scorecard (E), the label helper (I6a), T-ridge removal (I6c), step parity, serve-time feature logging and the parity monitor (I6e), game-line and book-leg as-of alignment (I6f), shadow scoring for the age test (I6g), and rung admission as a volume lever.
  - *Research, unproven here:* making NFL models carry at-market information (I6d), and whatever drives MLB version decay. I6g's test is engineering; its cure may not be.
  - NFL's encompassing failure appears at every distance from the training cutoff, so it is a feature or information problem, not a staleness or calibration one.
  - No method in the literature guarantees beating a liquid market's line. Kaunitz, Zhong & Kreiner 2017 (arXiv:1710.02824) profit by *using* the consensus, and bookmaker prices are themselves strong forecasts (Levitt 2004, doi:10.1111/j.1468-0297.2004.00207.x; Spann & Skiera 2009, doi:10.1002/for.1091).
- **Estimator traps.**
  - Through-origin λ is level-contaminated.
  - Pooled encompassing without cell fixed effects credits cross-cell levels (MLB model-only, August: .78 pooled against .32 within cell).
  - Fused-P coefficients are contaminated for NFL.
  - Every phase-2 number uses cell fixed effects, plus model-only P where it matters. R1's regime map used through-origin λ, but its ranking survives the level-robust estimator (Finding 16).
- **Power.**
  - *Phase 1:* the post-fix window is 33 days, with about 11 NFL game-days (about 6 per era). The NFL encompassing estimates rest on about 520 player-games with a decision-proxy consensus. The tail-scorecard prototype covers 23 days and 1,615 selected rows.
  - *Phase 2:* parity rests on 13 days (3 for MLB). The version-age contrast uses 10 fresh and 21 aged MLB days across three retrains. Offline September has 15 days, and offline NFL 2026 has 10.
  - All CIs are day-clustered and wide where noted.
- **Look-ahead.**
  - The pre-fix fallback distance split uses a closing consensus, because no decision-time DFS polls exist before August. It is context only.
  - Post-fix conclusions use the decision-time consensus, which falls in the same band as the closing consensus 93.4% of the time.
  - The training matrices carry three measured look-ahead or timing skews: in-game game lines, the MLB comp snapshot, and the 12:00 UTC book leg against later decisions. The tail scorecard inherits them, so its gap is a lower bound on the live gap.
- **Coverage.**
  - Live exact re-serves cover count families on registry versions, 53% of recommended legs.
  - Parity covers 9,382 legs: the two pickle sets of section 3 plus 120 NFL passing-tds legs on a pre-identity pickle. The MLB `20260903` versions are not on disk.
  - The C3 live map comes from unpenalized refits, an upper bound for the production (penalized) chain.
  - Prototype payouts are assumed (overround medians) for unposted rungs.
- **Convention robustness.** The decomposition (c20), the version-age contrast (c29) and every verdict hold at √3.5.
- **What could make this wrong.**
  - (i) NBA enters from late October. No NBA cell is measured here, and NBA models may encompass the market differently.
  - (ii) Encompassing is linear. Segment-level information (position, role, injury news) could exist and be averaged away.
  - (iii) Consensus is the median of the books' stored lines. If a book stores an alt tier as its main line, distance bands blur; the 93.4% band agreement bounds the damage.
  - (iv) The version-age effect could be calendar or regression to the mean. I6g decides.
  - (v) The September drop could be an MLB end-of-season effect (expanded rosters, rest, pennant races) that is absent in NBA and NHL. Then the in-season information gap is smaller than this window shows.
  - (vi) The NFL count quotes are mis-levelled (`ev` against `under_prob`). Every fused NFL coefficient inherits that, which is why the NFL numbers here use model-only P.

## 9. Open questions / caveats (for the plan's "Open questions")

1. **NFL at-market information by segment.** After the I6d retrain, does b_model become positive in any segment (position, role, snap share), or for any NFL cell other than tds and interceptions (whose BSS is 0.16–0.25)?
2. **Why validation w stays high where live encompassing says about 0.** Receptions w = 0.80 against b_model −0.10. Phase 2 removes two candidates: booster staleness (NFL information is ≈ 0 even 0–60 days after the cutoff) and the matchup leak. The lead that remains is the mis-levelled NFL count quote. `ev` and `under_prob` disagree (memory `nfl_count_quote_columns_disagree`), so a model that corrects the level of a mis-levelled book leg earns weight on validation Brier without carrying ranking information at the market. The fp team-feature blackout from 2026 week 5 is still a forward risk.
3. **DFS main line as a median quote for unquoted legs.** The encompassing test says the DFS main line carries the market's information (receptions b_market 0.79 [0.57, 1.12]). Admitting it as a book leg conflicts with the owner's 2026-09-29 decision that pickem rows are book-less for g1 and blend fit. Doing it serving-side only would create a train/serve skew by construction. **Owner call.**
4. **Harder-than-consensus Underdog Unders** (Finding 10). Are these Underdog's discounted or boosted lines, the same family `8bebd0e3` describes? If the platform's line-plus-payout is a structural informed price, the model should learn it as a decision-time input (I6d). R1's layer is dead, so no layer can trust it instead.
5. **Calibration-only refit.** Can T and PROB_STAGE be refit from stored validation predictions without a full `meditate`? That would make the T-ridge fix a cheap batch job.
6. **Data hygiene.** 17 D-era NFL rows carry game date 09-09 under a later-dated version. `Model Skew` and `Step` are not persisted in history.
7. **Overround drift.** Underdog prices at about 1.178 when the price moves (1.1236 standard). The scorecard should re-estimate OR each run.
8. **Version-age mechanism (I6g).**
   - Pre-registered confirmation on NBA and NHL from late October:
     - primary: fresh (days 0–3) minus aged (days 4+) recommended gap, day-clustered, one-sided;
     - secondary: within-market b_model;
     - identification: shadow scores of the prior version on the same legs.
   - Candidate mechanisms not yet excluded:
     - the calibration stage refit at each retrain;
     - another input that `meditate` refreshes (`stat_calibration.json`, `book_weights.json`);
     - regression to the mean around retrain timing.
9. **The September regime.** Offline September within-cell information falls (fused λ_c .86 → .28), and established players fall the most. Is this an MLB end-of-season effect, absent in other leagues? The first read is NBA and NHL in October–November.
10. **Probable vs realized opposing pitcher.** Training uses the realized starter; serving uses the probable one. Persist the serve-time probable pitcher (I6e) and measure the share of legs where the two differ.
11. **NFL enrichment miss.**
    - 62% of 2026 NFL matrix rows sit at the 0.5 moneyline default; in the scorecard window it is 94%.
    - Root cause unknown. Candidates: the team-code join, archive coverage, or the `get_team_market_map` keys.
12. **NFL count quote mis-levelling.** It contaminates every fused NFL number. Fix it, or derive the book leg from one consistent column, before I6d's acceptance runs.
13. **Serve-time persistence (I6e).** Persist the serve-time features, `Model Skew` and `Step`. That closes the SkewNormal live re-serve blind spot and enables I6g.
14. **Grading of non-participants.**
    - 1.3% of MLB hitter legs had no plate appearance; R1 counts 5.4% non-starters.
    - The ledger grades a voidable Under as a hit. Excluding these legs raises the recommended gap by 0.7 pp (c22), so every hit rate here is slightly flattering.
15. **MLB comp snapshot.** Start archiving the daily Savant affinity CSVs now, so a point-in-time rebuild is possible for 2027.

## 10. Bibliography

| # | Reference | Identifier |
|---|---|---|
| B1 | Smith, J. E. & Winkler, R. L. (2006). The optimizer's curse: skepticism and postdecision surprise in decision analysis. *Management Science* 52(3):311–322 | doi:10.1287/mnsc.1050.0451 |
| B2 | Capen, E. C., Clapp, R. V. & Campbell, W. M. (1971). Competitive bidding in high-risk situations. *J. Petroleum Technology* 23(6):641–653 | SPE-2993-PA (doi:10.2118/2993-PA) |
| B3 | Thaler, R. H. (1988). Anomalies: the winner's curse. *J. Economic Perspectives* 2(1):191–202 | doi:10.1257/jep.2.1.191 |
| B4 | Thaler, R. H. & Ziemba, W. T. (1988). Anomalies: parimutuel betting markets: racetracks and lotteries. *J. Economic Perspectives* 2(2):161–174 | doi:10.1257/jep.2.2.161 |
| B5 | Snowberg, E. & Wolfers, J. (2010). Explaining the favorite–long shot bias: is it risk-love or misperceptions? *J. Political Economy* 118(4):723–746 | doi:10.1086/655844 |
| B6 | Chong, Y. Y. & Hendry, D. F. (1986). Econometric evaluation of linear macro-economic models. *Review of Economic Studies* 53(4):671–690 | doi:10.2307/2297611 |
| B7 | Fair, R. C. & Shiller, R. J. (1990). Comparing information in forecasts from econometric models. *American Economic Review* 80(3):375–389 | RePEc:aea:aecrev:v:80:y:1990:i:3:p:375-89 (no DOI) |
| B8 | Harvey, D. I., Leybourne, S. J. & Newbold, P. (1998). Tests for forecast encompassing. *J. Business & Economic Statistics* 16(2):254–259 | doi:10.1080/07350015.1998.10524759 |
| B9 | Hébert-Johnson, Ú., Kim, M. P., Reingold, O. & Rothblum, G. N. (2018). Multicalibration: calibration for the (computationally-identifiable) masses. ICML 2018, PMLR 80 | arXiv:1711.08513 |
| B10 | Zhao, S., Kim, M. P., Sahoo, R., Ma, T. & Ermon, S. (2021). Calibrating predictions to decisions: a novel approach to multi-class calibration. NeurIPS 2021 | arXiv:2107.05719 |
| B11 | Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML 2017 | arXiv:1706.04599 |
| B12 | Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML 2005, 625–632 | doi:10.1145/1102351.1102430 |
| B13 | Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*, MIT Press, 61–74 | (no DOI; book chapter) |
| B14 | Van Calster, B. et al. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine* 17:230 | doi:10.1186/s12916-019-1466-7 |
| B15 | Ranjan, R. & Gneiting, T. (2010). Combining probability forecasts. *JRSS B* 72(1):71–91 | doi:10.1111/j.1467-9868.2009.00726.x |
| B16 | Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477):359–378 | doi:10.1198/016214506000001437 |
| B17 | Andrews, I., Kitagawa, T. & McCloskey, A. (2024). Inference on winners. *Quarterly J. Economics* 139(1):305–358 | doi:10.1093/qje/qjad043 |
| B18 | Efron, B. (2011). Tweedie's formula and selection bias. *JASA* 106(496):1602–1614 | doi:10.1198/jasa.2011.tm11181 |
| B19 | Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics* 17(3):1217–1241 | doi:10.1214/aos/1176347265 |
| B20 | Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics* 90(3):414–427 | doi:10.1162/rest.90.3.414 |
| B21 | Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian J. Statistics* 6(2):65–70 | JSTOR 4615733 |
| B22 | Kaunitz, L., Zhong, S. & Kreiner, J. (2017). Beating the bookies with their own numbers — and how the online sports betting market is rigged | arXiv:1710.02824 |
| B23 | Hubáček, O., Šourek, G. & Železný, F. (2019). Exploiting sports-betting market using machine learning. *Int. J. Forecasting* 35(2):783–796 | doi:10.1016/j.ijforecast.2019.01.001 |
| B24 | Walsh, C. & Joshi, A. (2024). Machine learning for sports betting: should model selection be based on accuracy or calibration? | arXiv:2303.06021 |
| B25 | Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data* 6(4):15 | doi:10.1145/2382577.2382579 |
| B26 | Breck, E., Cai, S., Nielsen, E., Salib, M. & Sculley, D. (2017). The ML test score: a rubric for ML production readiness and technical debt reduction. *IEEE International Conference on Big Data*, 1123–1132 (Model 4 "The impact of model staleness is known"; Monitor 3 training/serving skew; Monitor 7) | doi:10.1109/BigData.2017.8258038 |
| B27 | Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4(9):100804 | doi:10.1016/j.patter.2023.100804 |
| B28 | Moskowitz, T. J. (2021). Asset pricing and sports betting. *Journal of Finance* 76(6):3153–3209 | doi:10.1111/jofi.13082 |
| B29 | Zinkevich, M. *Rules of Machine Learning: Best Practices for ML Engineering*. Google (grey literature); Rules #8, #29, #31, #32, #37 | developers.google.com/machine-learning/guides/rules-of-ml (no DOI) |
| B30 | Sculley, D. et al. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems* 28, 2503–2511 | no DOI (proceedings.neurips.cc, paper 86df7dcfd896fcaf2674f757a2463eba) |
| B31 | Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys* 46(4):44 | doi:10.1145/2523813 |
| B32 | Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? *Economic Journal* 114(495):223–246 | doi:10.1111/j.1468-0297.2004.00207.x |
| B33 | Clements, M. P. & Harvey, D. I. (2010). Forecast encompassing tests and probability forecasts. *Journal of Applied Econometrics* 25(6):1028–1062 | doi:10.1002/jae.1097 |
| B34 | Spann, M. & Skiera, B. (2009). Sports forecasting: a comparison of the forecast accuracy of prediction markets, betting odds and tipsters. *Journal of Forecasting* 28(1):55–72 | doi:10.1002/for.1091 |
| B35 | Štrumbelj, E. (2014). On determining probability forecasts from betting odds. *International Journal of Forecasting* 30(4):934–943 | doi:10.1016/j.ijforecast.2014.02.008 |
| P1 | Prior in-repo brief: selection shrink toward the book (KILL, λ_sel ≈ −0.10) | `docs/archive/researcher_selection_shrink.md` |
| P2 | Main-session §6 measurement, 2026-10-04 (fallback / quoted / unquoted cohort). The scratch file was lost with /tmp; the numbers live in the lane brief | `docs/handoffs/honest-receipts.md` §5, §7 |
| P3 | Commits: fallback modal-line decode `6c21665c` (2026-08-29); model-path book leg never self-quotes `8bebd0e3` (2026-08-30); NFL matchup-leak removal `96b573fe` (2026-09-01); NFL serve-path fixes `6eac660c`, `abb82025`, `4c8b4a91` (09-25..26) | git |
| BR1 | R1 brief: selection-aware trust layer — KILL (K2 ablation, regime map, version decay, DNP grading) | `docs/archive/researcher_trust_layer.md` |
| BR2 | R2 brief: parlay engine and the per-pick Underdog convention (√3.5; the pricer is sound, the legs fail) | `docs/archive/researcher_parlay_engine.md` |

Scratch scripts and outputs (read-only reproductions; every number above comes from these): `/tmp/claude-1000/-home-trevor-Sportstradamus/7c225c1a-7c1e-4cde-bede-cf7b5fe0cce8/scratchpad/r3/`. Phase 1 scripts include `decomp.py`, `paired.py`, `c1_live.py`, `c3_live.py`, `c4_rungs.py`, `c4b_live.py`, `c5_tailz.py`, `c7_population.py`, `c8_tpen.py`, `c8_live.py`, `c10_distcal.py`, `c12_encompass.py`, `c6_compress.py`, `c13_bookonly.py`, `fb_linedist.py`, `fb_linedist2.py`, `tail_proto*.py` and `holm_extra.py`. Phase 2 adds `c14_quoteage.py`, `c15_agedecay.py`, `parity_lib.py`, `c16_parity.py`, `c16b_eval.py`, `c17_gameline.py`, `c18_gameline_leak.py`, `c19_misc.py`, `c20_decomp_r2conv.py`, `c21_live_info.py`, `c22_participation.py`, `c23_reference_timing.py`, `c24_info_split.py`, `c25_september.py`, `c26_within_cell.py`, `c27_nfl_age.py`, `c28_version_fresh.py`, `c28b_daily.py`, `c29_age_gap.py`, `c29b_age_robust.py`, `c30_serve_freshness.py`, `c31_age_level.py` and `c32_age_by_market.py`. /tmp was wiped at about 18:41Z; the data artifacts are gone, and every script regenerates its inputs from `history.parquet`, the test sets, the matrices, the pickles and the read-only archive.
