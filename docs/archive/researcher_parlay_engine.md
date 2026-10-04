# In-repo research brief R2: parlay engine repairs and the per-pick Underdog payout convention

**Question:** which parlay-pricer repairs make priced joint probability and payout match realized outcomes, which change beam-search selection, and which per-pick Underdog payout convention the single-leg ledger and `Model EV` should use.
**Date:** 2026-10-04 · branch `devel` · read-only (no pickles, no config edits, no git writes, archive opened `read_only=True`). Scripts and intermediate parquet: `/tmp/claude-1000/-home-trevor-Sportstradamus/7c225c1a-7c1e-4cde-bede-cf7b5fe0cce8/scratchpad/r2/`.

**STATUS: Stage 1 READY FOR THE OWNER (payout tables + per-pick convention). Stage 2 COMPLETE (Σ, Underdog modifier, leg marginals, beam selection, ranked I5 list).**

## TL;DR

- **Stage 1: set the per-pick Underdog convention to √3.5 = 1.8708, the 2-pick Power root, replacing 1.78.** Price every multi-leg entry from the live table at its own size. The Underdog recommended cohort goes from 3,055 legs at −13.1% to 6,115 legs at −6.3% [−9.4, −3.3]. Both platforms together go from −11.8% to −7.7%. No sign flips. The owner confirms the live table in the app first.
- **Parlay money is lost in the legs, not in the pricer.** The test set is 818,844 synthetic parlays built from post-fix posted settled legs. The legs the beam admits are read 8–11 pp above their hit rate (hit / `Win Prob` 0.82–0.88). The current pricer's realized/priced all-hit ratio is 0.71–0.77 at 2 legs, 0.57–0.69 at 3 and 0.41–0.44 at 5, which is that per-leg ratio compounded. Replacing only the leg marginals with `Market Prob` (R4) puts every entry size's CI over 1 (point 0.88–1.11) with the same Σ and copula. So the copula machinery is sound, and served `Win Prob` on admitted legs is what fails. R4 is a diagnostic, not an engine: it is model-free, and R1 was KILLED. The walk-forward Trust proxy (R4b) calibrates only by collapsing onto the market; its weight on logit `Win Prob` falls from 0.41 to 0.19.
- **KILL I5b (symmetric Σ).** On same-game parlays from all posted legs it changes log-loss by −0.00011 [−0.00024, +0.00003] (p = .12). On admitted legs it is worse: +0.00079 [+0.00028, +0.00130]. Its top-20 Jaccard against today is 0.94. The shipped Σ cannot be told apart from independence (+0.00029, p = .07). NFL team matrices hold 7–11 games each, so each entry is a noisy estimate shrunk to 23–37% of itself. **KILL I5c (`underdog_tax.py`) as a pricing repair.** It fails Holm (p = .052 against a .017 threshold), and Underdog's modifier m is below 1 on only 3–7% of entries, averaging 0.99 when it is.
- **GO on I5a and I5d as truth, not as calibration.** I5a: live tables, √3.5, file pair modifiers above 1 set to 1, and Flex tiers on the n−k largest pick multipliers. I5d: the ledger settles at T × Π b × m. On the entries the beam selects, the stored payout runs high (median true/priced 0.94), because selection picks up stale pair bonuses: 68% of gated entries carry one, mean ×1.24. The worst case is backwards: opposing RB carries [1.16, 0.86] pays extra for a same-direction pair whose ρ is −0.26. The sim-bettor ledger settles at the bare table, which underpays stored winners by a median 1.68×. Live tables alone (R2) fail Holm (p = .28). They raise gated volume 43% with no detectable change in realized return: 0.25 [0.10, 0.62] against 0.18 [0.07, 0.42] per $1.
- **The Model EV ≥ 2.0 floor amplifies the curse.** No pricer repair can reach I5's acceptance band [0.9, 1.1] while legs are served `Win Prob`. Across NFL Underdog priced-EV bands <1.5, 1.5–2, 2–2.5, 2.5–3 and ≥3, realized/priced falls 0.50, 0.34, 0.09, 0.05, 0.00. Today's gates keep 1,283 entries: 2.3% hit against 29.7% priced (ratio 0.08 [0.04, 0.18]), returning $0.18 per $1. No floor or book-EV gate tested returns ≥ 1. The lever is the leg probabilities (R3 → I6, the model track), and I5 shrinks to the factual refresh.

---

## Decision question (verbatim)

"Which parlay-pricer repairs (symmetric game Σ, live payout tables, Underdog's per-game correlation modifier, book-anchored or Trust leg marginals) make the priced joint probability and payout match realized parlay outcomes, which change what the beam search selects, and what per-pick Underdog payout convention should the single-leg ledger and Model EV use? KILL any repair that does not improve out-of-sample calibration or selection."

## Pre-registration (written before any stage-2 test was run)

Method rules: post-fix data only (Date ≥ 2026-08-31, the book-leg fix era); walk-forward (expanding window, weekly refits, never fit on the evaluation week, every feature cut at decision time); day-clustered CIs (block bootstrap by Date, 2,000 resamples); pre-registered candidate set, Holm across it, everything tried reported; NFL reported separately; KILL is a valid verdict; primary sources with DOIs / arXiv IDs.

Candidate repairs (each alone, then all together):

| ID | Repair | Exact definition |
|---|---|---|
| R0 | current | `Win Prob` marginals; Σ as `_build_game_corr_map` builds it (first-processed team's same-team ρ × 0.75, other team × 0.25, cross pairs summed from both sides × 0.75 each); stored payout = stale table × leg multipliers × `banned_combos.json` pair modifiers |
| R1 | symmetric Σ | both teams' same-team ρ at one weight (1.0; the matrices are already overlap-shrunk estimates), cross pair = mean of the two perspectives (or the one present), `_nearest_psd` kept |
| R2 | live payout tables | Power 3.5/6.5/12/20/35/65/120 for 2–8 picks, live Flex tiers, Flex partial tiers multiply the n−k largest pick multipliers, every `banned_combos.json` Underdog modifier > 1 set to 1 |
| R3 | Underdog per-game modifier m | m_g = Π p_i / P(all same-game picks hit) under Underdog's published Gaussian ρ per pair type (untaxed pairs ρ = 0), clipped at 1, multiplied across games (docs/underdog_api.md §6.8); Underdog only |
| R4 | book-anchored leg marginals | `Market Prob` (sportsbook decode where quoted, platform-implied otherwise) replaces `Win Prob` in the joint |
| R4b | walk-forward selection-aware marginal (Trust proxy) | weekly expanding-window logistic of hit on logit(`Win Prob`), logit(`Market Prob`), log payout, fit on all posted settled legs before the evaluation week |
| ALL | R1 + R2 + R3 + R4 | primary comparison: ALL vs R0 |

Primary statistic: realized/priced all-hit ratio by entry size (2, 3) with day-clustered 95% CI; acceptance = CI covers 1 at every entry size under the repaired pricer. Secondary: Δ log-loss of the all-hit event (repair − R0) with day-clustered CI; EV calibration (realized payout / priced EV). Holm family = {R1, R2, R3, R4, R4b} vs R0, two-sided, α = 0.05, on the synthetic-parlay sample. Selection: top-K overlap (Jaccard) of beam output before/after, and realized all-hit of the selected sets.

Synthetic parlays (to escape parlay_hist's selection): drawn from all post-fix posted settled legs in `history.parquet` (model-chosen side, one offer per platform, no combo or `vs.` legs, no pushes), sizes 2 and 3, stratified by pair type (cross-game, same-game opposing, same-game same-team), fixed seed, distinct players within a parlay.

## Data and coverage actually used

- `history.parquet` (dev copy, written 2026-10-03 09:09): 533,411 rows; settled posted offers (the `realized.settled_offers` rule, replicated because the function is not on disk yet: Over/Under result, `Win Prob` present, `Boost > 0`, dedup on `(Date, Player, Market, Line, Bet, Platform)`): 277,957 all-time, **131,798 post-fix** (2026-08-31 → 10-02, 32 days; Underdog 72,688, Sleeper 59,110; MLB 110,445, NFL 14,979, WNBA 5,652, NHL 722). This copy has no `Payout Over`/`Payout Under` columns yet.
- `parlay_hist` (127 day files): 3,530,369 candidates, 16,708,141 legs. **Post-fix it holds 1,691 candidates on 7 NFL dates** (09-13, 09-20, 09-24, 09-27, 09-28, 10-01, 10-04); MLB and WNBA parlays stop after 2026-08-30. Reason in Finding S2-1.
- Grading verified (the coordinator's check): on 898,592 parlays fully resolved both by `parlay_hist` and by `history.parquet` outcomes (no pushes), `Misses` agrees with the history-derived miss count on 98.4%; all-hit rates agree within 0.2 pp in every platform × size cell (e.g. Sleeper 2-leg .1152 vs .1153, Underdog 3-leg .1349 vs .1364). The low all-hit rates are real, not a resolution defect.
- Archive (`archive.duckdb`, read-only): `ladder` has Underdog 4.85M and Sleeper 2.90M rows from 2026-08-30 to 10-03 observations.

---

## Stage 1 verdict: payout tables and the per-pick Underdog convention (READY)

### S1-a. Every stale cell in `underdog_payouts.json` and `payouts.py`

Reference: the live `entry_slips/estimate` quotes captured 2026-09-10 for even-money (`payout_multiplier` 1.0) picks from different games (`docs/underdog_api.md` §6.8, P-32). The quote, not the file, settles a slip.

| Cell | File today | Live | Status |
|---|---|---|---|
| Power 2 | 3.0 | 3.5 | stale (−14%) |
| Power 3 | 6.0 | 6.5 | stale (−8%) |
| Power 4 | 10.0 | 12.0 | stale (−17%) |
| Power 5 | 20.0 | 20.0 | matches |
| Power 6 | 25.0 | 35.0 | stale (−29%) |
| Power 7, 8 | absent | 65, 120 | missing (Underdog allows 8 picks, `maximum_selection_size: 8`) |
| Flex 3 | 2.25 / 1.25 | 3.25 / 1.09 | stale both tiers |
| Flex 4 | 5.0 / 1.5 / 0.4 | 7.2 / 1.8 | stale; the 2-loss tier does not exist live (`max_losses` 1 up to 5 picks) |
| Flex 5 | 10 / 2.0 / 0.4 / 0.4 | 10 / 2.5 | 1-loss tier stale; 2- and 3-loss tiers do not exist live |
| Flex 6 | 25 / 2.0 / 0.4 / 0.4 / 0.4 | 25 / 2.6 / 0.25 | 1- and 2-loss tiers stale; 3- and 4-loss tiers do not exist |
| Flex 7, 8 | absent | 40 / 2.75 / 0.5; 80 / 3.0 / 1.0 | missing |
| `insurance` 2–4 | 3 / 6 / 10 all-or-nothing | — | stale alias (payouts.py never reads it in the pooled path) |
| `insurance` 5, 6 | 10 / 2.5; 25 / 2.6 / 0.25 | equals live Flex 5, 6 | the only rows that match |
| `_comment` | "public Underdog terms current as of 2026-05" | — | stale |

`payouts.py` defects against the same capture:

1. `_pooled_underdog_curve` prices 2–3 picks on Power and 4–6 on the (stale) Flex rows and stops at 6. Live, Power and Flex both run to 8 picks; at live terms a 4-pick Flex (7.2 / 1.8) breaks even at p = 0.518 per even pick, a 4-pick Power (12) at 0.537, so the pool choice is a pricing decision the file currently makes with wrong numbers.
2. `expected_payout_with_pushes` multiplies **every** non-push leg's multiplier into a partial tier. Live Flex multiplies only the n − k **largest** pick multipliers into the k-loss tier (picks at 0.87 / 1.16 / 0.74: 1-loss quoted 1.10 = 1.09 × 0.87 × 1.16). With today's mostly discounted picks (post-fix median Underdog multiplier 0.76), the code underprices partial tiers.
3. `banned_combos.json` Underdog modifiers: 86 of 108 NFL same-team pairs, all 10 NFL opposing pairs, all 49 MLB same-team pairs and 228 of 235 NHL same-team pairs carry a value above 1. Live Underdog never pays above the table on a correlated pair (m ≤ 1; opposite sides and negative association come back at exactly 1.0). Pairs the file bans that Underdog prices: QB pass TDs × same-team WR TDs (file 0.0 / 0.0; live 7.13 vs 7.8 untaxed, m = 0.914). Pairs the file taxes that Underdog leaves untaxed: QB pass yds × same-team RB rush yds (file 1.03 / 0.95), QB pass yds × opposing QB pass yds on Power (file 0.9 / 1.1; Power 6.5 untaxed, Flex 0.898), RB rush yds × opposing RB rush yds (file 1.03 / 0.95). Example of the effect in storage: a 2026-09-28 NFL slip (Bigsby Over carries, Monangai Over carries, Barkley Under TDs) carries `Boost` 12.006 = 6.0 × 1.725 × 1.16, where 1.16 is a stale RB-carries pair bonus Underdog does not pay.
4. `_ledger_settlement.realized_multiplier` reads the same stale curve and ignores per-pick multipliers and m (already recorded in the plan as I5d).
5. `parlay_rake.json` (`Underdog: {"3": 0.962}`) feeds only the dashboard Modifiers reconciliation, not pricing; it has no counterpart in the live formula (Power = T_n × Π pick multipliers × m).

Sleeper (`sleeper_payouts.json`: Max 2 = 1.0, Max 3 = 1.0797, Flex K(n,k)) has no live capture in this repo to verify against; not verified here.

### S1-b. The per-pick Underdog convention

**Recommendation: replace `UNDERDOG_BOOST_BASELINE = 1.78` with the 2-pick Power root √3.5 = 1.8708 for single-leg `Model EV`, Kelly, the 5% edge rule, the payout-implied `Market Prob` of one-sided rows, and the realized ledger. Price every multi-leg entry from the table at its own size, never through a root.** Treat the switch as a new selection era (it changes which legs are recommended); tag it and never pool across it.

Why this anchor:

1. **There is no single-pick player-prop product.** Underdog's own API sets `minimum_selection_size: 2` (`state_config.pick_em`) and `min_selection_count: 2` (`entry_slips/estimate`). Streaks and prediction-market contracts are separate products with their own prices. A per-pick price is therefore always a share of an entry, and the 2-pick Power is the smallest real entry.
2. **It factors exactly.** A k-pick Power entry pays T_k × Π b_i, so under independence its expected return is Π (p_i · b_i · T_k^(1/k)). A leg's EV at r_k is exactly the factor it multiplies into a k-pick entry's EV, which is what a single-leg ledger is meant to report.
3. **The roots of the sizes the product builds are within 0.3% of each other:** 3.5^(1/2) = 1.8708, 6.5^(1/3) = 1.8663, 12^(1/4) = 1.8612. Post-fix Underdog beam output is 94% 3-pick and 6% 2-pick; the sim-bettor ledger plays Power at 2–3 picks (`strategies/ledger.py` `_POWER_SIZES`). Only 5–8-pick Power (1.81–1.82) sits materially lower, and those sizes are priced from the table by the parlay pricer anyway.
4. **Underdog displays it.** Every even (1.0×) pick in the 2026-09-10 lobby capture carries `odds.fantasy.decimal` = 1.87 (−115), i.e. √3.5 to three decimals; the `american_price` field shows −112 (1.893) (`docs/underdog_api.md` §7.4, [INF] which field is the experiment variant).
5. **One rule for both platforms.** Sleeper's `Boost` is already its posted per-pick decimal, and Sleeper's 2-pick Max carries no bonus (1.0), so Sleeper is on the 2-pick anchor today. Underdog at √3.5 makes the ledger use the same rule on both apps.
6. **1.78 has no live basis.** It is 10^(1/4), the root of the old 4-pick Power 10×. Under the live table it under-credits every Underdog winner by 4.9% of payout relative to the 2-pick root.

The Flex products cannot be represented by any single per-pick number: at even picks their breakeven per pick is 0.554 (3), 0.518 (4), 0.548 (5), 0.538 (6), 0.554 (7), 0.551 (8). Those prices belong in the parlay pricer (I5a), not the single-leg convention.

### S1-c. Effect on the post-fix single-leg ledger (Underdog, 72,688 posted settled legs, 32 days)

Cohort rule throughout: `Recommended` = `Win Prob` × payout − 1 ≥ 0.05 and 1 < payout ≤ 2.5. ROI CIs are day-clustered (2,000 resamples).

| Per-pick r | Legs clearing 5% (no cap) | Recommended n | New / lost vs 1.78 | Hit | Read | Avg payout | Breakeven | ROI [95% CI] | Same 3,055 legs re-graded at r | All posted: ROI, breakeven |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.78 (current) | 3,165 | 3,055 | — | .471 | .603 | 1.88 | .532 | −13.1% [−17.9, −7.7] | −13.1% | −13.8%, .714 |
| 1.808 (6-pick) | 4,004 | 3,888 | +839 / −6 | .482 | .603 | 1.88 | .531 | −11.0% [−14.9, −6.6] | −11.7% | −12.4%, .703 |
| 1.821 (5-pick) | 4,387 | 4,268 | +1,222 / −9 | .491 | .604 | 1.88 | .531 | −9.6% [−13.3, −5.6] | −11.2% | −11.9%, .698 |
| 1.861 (4-pick) | 5,828 | 5,684 | +2,663 / −34 | .507 | .609 | 1.88 | .533 | −7.1% [−10.2, −4.1] | −9.2% | −9.9%, .683 |
| 1.866 (3-pick) | 6,024 | 5,869 | +2,859 / −45 | .507 | .609 | 1.88 | .533 | −7.0% [−10.2, −4.0] | −8.9% | −9.7%, .681 |
| **1.871 (2-pick, recommended)** | **6,270** | **6,115** | **+3,105 / −45** | **.511** | **.609** | **1.88** | **.533** | **−6.3% [−9.4, −3.3]** | **−8.7%** | **−9.4%, .679** |
| 1.893 (UD −112 field) | 7,289 | 7,120 | +4,124 / −59 | .517 | .613 | 1.87 | .535 | −5.7% [−8.4, −2.7] | −7.6% | −8.4%, .671 |

Both platforms together (Sleeper unchanged: 3,083 recommended, hit .501, ROI −10.5% [−15.4, −6.3]): the recommended cohort goes from 6,138 legs at −11.8% [−16.2, −7.4] (1.78) to 9,198 legs at −7.7% [−11.2, −4.5] (1.871).

What moves and why:

- **Re-grading alone** is exact arithmetic: (1 + ROI) scales by r / 1.78, +4.4 pp on the same 3,055 legs (−13.1% → −8.7%).
- **Composition** adds about 2 pp. The 3,105 legs that newly clear 5% are favorites (mean raw multiplier 0.96, payout 1.79) that hit .549 against a .613 read and return −3.9% [−7.8, −0.7]. The 3,010 kept legs return −8.8% [−13.8, −3.0]. The 45 dropped legs are the 2.5-cap boundary (raw 1.36, payout 2.55 at the new root, hit .378). Under side: −10.5% → −4.0%; Over: −18.3% → −11.8%.
- **League split at 1.78 → 1.871:** MLB 1,260 legs −10.7% → 3,126 legs −6.5% [−11.1, −2.3]; NFL 1,703 −16.4% → 2,761 −7.0% [−14.5, −3.4]; WNBA 85 → 199 legs, CIs span zero.
- **Side effects of the constant:** 26 one-sided unquoted rows (of 795) would newly fail the ±0.15 phantom gate because their payout-implied `Market Prob` = 1/(raw × r) falls; 7 rows would exceed the `_MAX_UNDERDOG_BOOST` = 3.65 filter; `Distance` dedup and the persisted raw `Boost` are invariant; `helpers/archive._dfs_offer_probs` devigs one-sided Underdog quotes through the same constant.
- **No sign flips.** At every candidate root the recommended cohort's ROI CI stays below 0. Every posted-side payout band still realizes below its breakeven at r = 1.871 (e.g. 1.4–1.6: hit .605 vs breakeven .671; 1.8–2.0: .501 vs .532). The convention fixes the ledger's arithmetic; it does not create edge, and the 9.8 pp read-over-hit gap on the new cohort is R1's and R3's problem, not this one.

### S1-d. Stage-1 reality checks

- The live table is a single capture (2026-09-10, one account, one state). When Underdog moved from 3 / 6 / 10 / 20 / 25 to 3.5 / 6.5 / 12 / 20 / 35 is not on record in the repo; third-party guides still quote the old table. Before I5a lands, the owner should confirm the table in the app for his state.
- The comparison assumes the history `Boost` is the settle value (`payout_multiplier`, §7.4) and that picks are independent. A same-game Underdog pair with positive ρ settles at T × Π b × m with m < 1, so the per-pick root overstates the realized payout of stacked legs; the engine does not build such stacks today (Stage 2).
- Build cost: one constant in `helpers/distributions.py` plus the `_comment`; the golden tests that pin 1.78-derived numbers move. Engineering, not research.

---

## Stage 2 verdict: Σ, Underdog's modifier, leg marginals, beam selection (COMPLETE)

Per R1's KILL (coordinator, 2026-10-04), no `Trust Prob` will exist. Leg marginals are evaluated as served `Win Prob` and book-anchored `Market Prob` only. R4b, the pre-registered walk-forward Trust proxy, is reported because it was run before the KILL; it is not a candidate.

### S2-0. Stage-2 data and coverage

- **Leg universe.** Post-fix posted settled legs, main lines only, one row per (Date, Platform, Player, Market), no combo or `vs.` legs: 90,317 legs (MLB 82,788, NFL 5,048, WNBA 2,481). NHL (722 rows) and NBA (no post-fix games) are not evaluated. Today's `_select_bet_offers` gates at the 1.78 convention admit 8,429 of them (9.3%). Across all post-fix posted legs, alt lines included, the admitted share is 8.8%: NFL 21–25%, MLB 4–11%, WNBA 2–7%.
- **Synthetic parlays.** 818,844 in total: pool A (all legs) 543,835 and pool B (admitted legs) 275,009. Each cell of stratum × (Date, Platform, League) gets up to 1,000 draws with seed 20261004. The strata are X2–X5 (cross-game), O2 (same-game opposing), S2 (same-game same-team) and G3 (same-game three-leg). This is a sample of the possible combinations, not an enumeration. Sizes 4–5 are tested on probability only, with no payout. They extend the pre-registered sizes 2–3, so they are reported but kept out of Holm.
- **Σ.** The shipped `training.correlate._correlate_teams` (imported read-only) is rebuilt for each evaluation week from `training_data/{LG}_corr.parquet` rows dated before that week: 5 weeks × NFL, MLB, WNBA. NFL teams carry 7–11 observations each (median 8).
- **R4b.** A weekly expanding-window logistic, so it exists from the second week on (W2+, 25–26 days).
- **Stored parlays.** 417 fully resolved post-fix entries on 6 NFL dates (Underdog 36 two-leg and 350 three-leg; Sleeper 2 and 29), single-tier, no pushes. The rebuilt Σ reproduces only 1.7% of stored `Corr Pairs` within 0.01, because weekly NFL matrices drift and production and dev matrices differ. So the stored-parlay arms price on the stored Σ and apply R1 as stored + (R1_rebuilt − R0_rebuilt).
- **Beam re-run.** 837 post-fix games, sizes 2–3, under today's gates: pairwise geo-mean ≥ 1.05, beam 1,000, both teams, boost product in (0.7, 2.5] on Underdog, books EV ≥ 0.9, Model EV ≥ 1.5 at precheck and ≥ 2.0 final. The Kelly-units gate is omitted because it is a sizing gate tied to training BSS. R0's first-processed team is assigned by a hash per game, since production's order depends on processing order. The run produces 608,709 candidates across eight variants. It reproduces production's NFL-only post-fix output and the stored entries' ratio (0.08 in both). That is the fidelity check.
- **Runtime and resampling.** Every script ran in under 10 minutes, and nothing beyond the synthetic draws was subsampled. All CIs are day-clustered bootstraps with 2,000 resamples.
- **Pre-fix context** (coordinator; fixed audit script I3d; trailing 90 days, mostly pre-fix; payout = clip(Boost, 1, 100), joint = Model EV / Boost, independence = Indep P / Boost; Underdog 4–6-leg rows from 2026-05-03 to 05 excluded; only fully resolved rows kept, 9–28% are partial). Underdog priced vs empirical all-hit: 2 legs .719 vs .216, 3 legs .378 vs .123, 4 legs .565 vs .060, 5 legs .301 vs .018, 6 legs .190 vs .012. Sleeper: 2 legs .49 vs .10, 3 legs .37 vs .07. The top priced decile: .50 vs .051. For multi-tier (Flex) entries, Model EV / Boost includes the partial tiers and is not P(all hit), which is why the 4-leg "priced" figure exceeds the 3-leg one. Single-tier entries of up to 3 legs are the clean comparison.

### Key findings

**S2-1. Post-fix `parlay_hist` is NFL-only because of the Model EV ≥ 2.0 floor.** In the re-run, 1,270 of the 6,351 NFL candidates that pass the prechecks clear the final 2.0 floor (20%). Only 12 of 367 MLB candidates (3%) and 1 of 12 WNBA candidates do. Production's MLB and WNBA entries stop on 2026-08-30. NFL dominates the floor for three reasons:

- NFL admits 21–25% of its posted legs, against 4–11% for MLB.
- Its admitted legs read 10–11 pp over the book (MLB: 9 pp).
- 68% of the gated entries carry a stale pair bonus (S2-6).

So the only live post-fix evidence about selected parlays covers one league on 6 settled dates.

**S2-2. Parlay over-prediction is leg over-prediction compounded.** Legs admitted post-fix:

| League | Platform | n | days | hit | `Win Prob` | `Market Prob` | hit / WP [95% CI] | hit / MP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| MLB | Sleeper | 5,267 | 30 | .549 | .627 | .539 | 0.88 [0.85, 0.90] | 1.02 [0.99, 1.05] |
| MLB | Underdog | 2,473 | 27 | .508 | .601 | .510 | 0.85 [0.80, 0.89] | 1.00 [0.95, 1.04] |
| NFL | Sleeper | 1,228 | 11 | .502 | .613 | .504 | 0.82 [0.71, 0.87] | 1.00 [0.86, 1.06] |
| NFL | Underdog | 2,123 | 11 | .512 | .609 | .507 | 0.84 [0.75, 0.88] | 1.01 [0.91, 1.05] |
| WNBA | Sleeper / Underdog | 72 / 178 | 12 | .569 / .545 | .589 / .592 | .533 / .518 | 0.97 / 0.92 (CIs span 1) | 1.07 / 1.05 (CIs span 1) |

Raising the per-leg ratio to the k-th power predicts the joint: 0.67–0.77 at two legs and 0.55–0.67 at three. The pool-B joint under today's pricer is 0.65 at two legs and 0.54 at three for NFL, and 0.75 and 0.64 for MLB.

The legs' error is the optimizer's curse, not bulk miscalibration. R3 attributes 89% of the single-leg gap to selection (`/tmp/researcher_train_serve_skew.md` TL;DR; Smith & Winkler 2006, DOI 10.1287/mnsc.1050.0451). In every quote class, the platform price lands within 1–8 pp of the selected tail's hit rate, while served p runs 13–23 pp high (R1, `docs/archive/researcher_trust_layer.md` Finding 12). The beam inherits this exactly. It admits legs on `Model EV > 1` and ranks on priced EV: the same selection on a noisy edge.

**S2-3. With calibrated marginals the joint machinery is calibrated. With served `Win Prob`, no pricer repair is.** Realized/priced all-hit on pool B (admitted legs, all leagues):

| Stratum | Platform | n | days | all-hit | R0 current | R1 symmetric Σ | R4 book marginals | R1 + R4 |
|---|---|---|---|---|---|---|---|---|
| X2 | Sleeper | 26,011 | 29 | .278 | 0.72 [0.67, 0.79] | 0.72 | 0.98 [0.90, 1.07] | 0.98 |
| X2 | Underdog | 20,452 | 26 | .252 | 0.71 [0.65, 0.79] | 0.71 | 1.00 [0.90, 1.10] | 1.00 |
| X3 | Sleeper | 29,647 | 29 | .147 | 0.62 [0.53, 0.71] | 0.62 | 0.97 [0.84, 1.12] | 0.97 |
| X3 | Underdog | 26,361 | 26 | .118 | 0.57 [0.47, 0.67] | 0.57 | 0.93 [0.78, 1.09] | 0.93 |
| X4 | Sleeper | 29,239 | 28 | .074 | 0.50 [0.42, 0.60] | 0.50 | 0.92 [0.76, 1.10] | 0.92 |
| X4 | Underdog | 26,497 | 24 | .059 | 0.48 [0.37, 0.59] | 0.48 | 0.93 [0.73, 1.13] | 0.93 |
| X5 | Sleeper | 27,981 | 25 | .037 | 0.41 [0.32, 0.50] | 0.41 | 0.88 [0.69, 1.07] | 0.88 |
| X5 | Underdog | 24,990 | 22 | .033 | 0.44 [0.34, 0.57] | 0.44 | 1.01 [0.77, 1.27] | 1.01 |
| O2 | Sleeper | 11,484 | 31 | .294 | 0.76 [0.72, 0.81] | 0.76 | 1.05 [0.98, 1.11] | 1.04 |
| O2 | Underdog | 5,300 | 29 | .262 | 0.72 [0.63, 0.78] | 0.72 | 1.01 [0.89, 1.11] | 1.01 |
| S2 | Sleeper | 10,581 | 32 | .300 | 0.77 [0.73, 0.81] | 0.76 | 1.05 [0.99, 1.11] | 1.04 |
| S2 | Underdog | 5,537 | 31 | .266 | 0.72 [0.65, 0.79] | 0.71 | 1.00 [0.90, 1.10] | 0.99 |
| G3 | Sleeper | 19,936 | 31 | .165 | 0.69 [0.61, 0.77] | 0.69 | 1.11 [0.98, 1.24] | 1.10 |
| G3 | Underdog | 10,993 | 29 | .135 | 0.60 [0.48, 0.71] | 0.60 | 1.00 [0.79, 1.18] | 0.98 |

- **Pool A (all posted legs).** R0 runs 0.88–0.98. Every Underdog CI excludes 1, and Sleeper's upper bounds touch 1. R4 runs 0.98–1.08; Sleeper's five-leg 1.08 [1.02, 1.14] is the one CI above 1.
- **By league (pool B, W2+).** NFL R0 is 0.65, 0.54, 0.41 and 0.35 at k = 2–5; R4 moves these to 0.96, 0.96, 0.88 and 0.91 (NFL k = 4–5 rests on 3 days). MLB moves from 0.75, 0.64, 0.50 and 0.44 to 1.02, 1.01, 0.93 and 0.93. WNBA's 743 and 417 parlays give CIs of 0.5–1.7 and 0.1–4.7, which says nothing.
- **Acceptance.** The pre-registered acceptance is a CI covering 1 at every entry size. R4 meets it, and so does the pre-registered primary ALL (R1 + R2 + R3 + R4): its probability ratio equals R1 + R4 above, and its EV ratio is 0.97–1.01 (S2-6). Every repair that keeps served `Win Prob` (R0–R3) misses it at every size in pool B. ALL is not admissible, because its R4 component is model-free.

**S2-4. Per-repair Δ log-loss** (pre-registered secondary; all-hit event, sizes 2–3; R1 rows are same-game only, the only parlays where Σ enters):

| Repair vs R0 | Sample | n | days | Δ log-loss [95% CI] | p |
|---|---|---|---|---|---|
| R1 symmetric Σ | pool A | 263,048 | 32 | −0.00011 [−0.00024, +0.00003] | .12 |
| R1 symmetric Σ | pool B | 63,831 | 32 | **+0.00079 [+0.00028, +0.00130]** (worse) | — |
| R4 book marginals | pool A | 415,887 | 32 | −0.00340 [−0.00511, −0.00166] | .0005 |
| R4 book marginals | pool B | 166,302 | 32 | −0.0242 [−0.0316, −0.0168] | — |
| R4b Trust proxy | pool A (W2+) | 356,517 | 26 | −0.00370 [−0.00555, −0.00165] | .0005 |
| R4b Trust proxy | pool B (W2+) | 138,522 | 26 | −0.0252 [−0.0332, −0.0169] | — |
| R1 + R4 | pool A | 415,887 | 32 | −0.00351 [−0.00519, −0.00178] | .0005 |
| independence at `Market Prob` (no Σ) | pool A | 415,887 | 32 | −0.00359 [−0.00538, −0.00178] | .0005 |

By league, R4 on pool A: NFL −0.0134 [−0.0239, −0.0034] (11 days), MLB −0.0017 [−0.0031, −0.0004] (30 days), WNBA −0.0020 [−0.0050, +0.0006]. Two things follow:

- The leg-marginal swap does all the work. Independence at `Market Prob` scores as well as any Σ variant.
- NFL's legs carry eight times MLB's per-parlay excess loss.

**S2-5. The shipped Σ carries no measurable information, and NFL Σ is noise around zero.**

- **Against independence** (exploratory, not in Holm). On pool-A same-game parlays at book marginals, the shipped Σ scores +0.00029 [−0.00004, +0.00059] against independence (p = .07; at `Win Prob` +0.00030, p = .06). The symmetric Σ scores +0.00012 (p = .47). On NFL same-game strata at book marginals, independence, R4, R1 + R4 and a pair-type pooled Σ agree to two decimals (0.95–0.98).
- **Why.** NFL team matrices hold 7–11 games per team (median 8). The credibility weight min(1, n/30) shrinks each entry to 23–37% of a noisy raw estimate; `docs/PARLAY_AUDIT.md` §1.1 already sentenced this shrink-toward-zero. The engine then weights by processing order. Compared against hit-event (tetrachoric) correlations at posted lines over NFL 2022–25:
  - QB pass yds × same-team WR rec yds: +0.38 [0.32, 0.42] (post-fix +0.44 [0.32, 0.62]), against shipped entries averaging +0.06 (range −0.17 to +0.33).
  - Opposing RB carries: −0.26 [−0.34, −0.16], against shipped −0.01 (range −0.26 to +0.26).
  - Same-team QB pass yds × RB rush yds: −0.16 [−0.24, −0.08], against −0.02.
- **Method.** The engine's 2 sin(πρ_s/6) remap (Kruskal 1958, DOI 10.1080/01621459.1958.10501481) is the right rank-to-Gaussian map for continuous margins. TDs, receptions and carries are discrete with heavy ties, and there the rank–copula identities break (Genest & Nešlehová 2007, DOI 10.2143/AST.37.2.2024077). A parlay pricer needs the orthant probability at the posted lines, and the tetrachoric correlation of the hit events estimates it directly (Olsson 1979, DOI 10.1007/BF02296207). Shrinkage should pull toward a pair-type mean, not toward zero (Schäfer & Strimmer 2005, DOI 10.2202/1544-6115.1175; Efron & Morris, ref [47] in `docs/operation_ship_references.md`). The R3 copula brief already prescribes exactly that: hierarchical Fisher-z EB (`docs/archive/researcher_copula_stage0.md`; model track §6.11).
- **What such a fix shows today.** A pair-type pooled NFL Σ (2022–25 tetrachoric) scores −0.00031 [−0.00066, +0.00008] against R4 (p = .10, 11 days). That is exploratory, and too small to detect at this sample.

**S2-6. The payout the engine prices is wrong in both directions, and selection picks up the errors.**

- **Underdog's live formula is Power = T_n × Π b_i × m, with m ≤ 1** (`docs/underdog_api.md` §6.8). The modifier rarely binds: on synthetic Underdog parlays m < 1 on 6.6% (17,863 of 269,627; mean 0.988 when it binds), and on the R0 gated entries on 6.7% (mean ≈ 1.00). Random same-game pairs carry a file bonus (> 1) 10% of the time (mean file modifier 0.97).
- **Selection picks up the stale bonuses.** On what the beam keeps, 68% of gated entries carry one (mean product ×1.24), and the priced payout sits above the true live payout: median true/priced 0.94, 53% of entries overpriced. On stored post-fix Underdog entries, true/stored is 0.97 at three legs (the live table adds 8%; removing the bonuses takes back more) and 1.09 at two.
- **The worst row is backwards.** `banned_combos.json` gives opposing RB carries [1.16, 0.86]: a 16% bonus on the same-direction pair, whose hit events are negatively correlated (ρ −0.26). So the file pays more for the less likely combination; Underdog pays 1.0. The 417 stored entries hold 283 such Under/Under pairs. Both legs hit .163 of the time, against .266 at independent book probabilities.
- **The sim-bettor ledger settles at the bare table.** `strategies/_ledger_settlement.realized_multiplier` pays T_file(n) and ignores the entry's own `payout_multiplier` (the priced `Boost`, which carries Π b). The engine picks boosted legs (stored Π b median 1.61), so on stored post-fix Underdog entries the true live payout is 1.67× what the ledger pays (median 1.68, q10–q90 1.18–2.16); on R0 gated re-run entries it is 1.24× (median). The ledger understates winners by more than any table cell is stale.
- **EV calibration** (synthetic Underdog, W2+, realized at the live price T_live × Π b × m; ratio = realized / priced EV):

| Pool | k | n | R0 | R2 live tables | R3 Underdog m | R4 book marginals | ALL |
|---|---|---|---|---|---|---|---|
| A | 2 | 98,257 | 1.08 [1.04, 1.11] | 0.93 [0.90, 0.96] | 1.07 [1.04, 1.10] | 1.15 [1.12, 1.19] | 0.98 [0.95, 1.01] |
| A | 3 | 72,281 | 0.96 [0.91, 1.02] | 0.89 [0.84, 0.95] | 0.95 [0.90, 1.01] | 1.06 [1.00, 1.13] | 0.97 [0.91, 1.03] |
| B | 2 | 26,197 | 0.84 [0.76, 0.92] | 0.72 [0.66, 0.79] | 0.83 [0.75, 0.91] | 1.19 [1.07, 1.30] | 1.01 [0.91, 1.10] |
| B | 3 | 30,179 | 0.64 [0.55, 0.75] | 0.60 [0.51, 0.69] | 0.63 [0.54, 0.73] | 1.08 [0.91, 1.25] | 0.97 [0.82, 1.13] |

  In R0, two wrongs offset. The stale low table hides part of the legs' overstatement (pool A, two legs: 1.08), so the correct table alone looks worse (0.93). R4 alone over-corrects (1.15–1.19) because it keeps the stale table. Only ALL is calibrated, and ALL is not admissible.
- **Holm across the pre-registered family.** R2 and R3 change only the payout, so their statistic is Δ|log(realized/priced EV)| against R0 on pool A; R1, R4 and R4b use Δ log-loss.

| Repair | p | Holm threshold | H0 rejected? |
|---|---|---|---|
| R4 | .0005 | .0100 | yes |
| R4b | .0005 | .0125 | yes |
| R3 | .0515 | .0167 | no |
| R1 | .123 | — | no |
| R2 | .279 | — | no |

  The step-down stops at R3, so R1 and R2 are not rejected regardless of their thresholds (.025, .05).

**S2-7. Underdog's tax against the truth: where a correlation edge could exist, and how small it is.** NFL pair types. Hit-event ρ is tetrachoric at posted lines; "Underdog ρ" comes from the 2026-09-10 quotes (§6.8); file values are [same direction, opposite direction].

| Pair type | Hit-event ρ, NFL 2022–25 [CI] | Post-fix | Shipped Σ (mean) | Underdog ρ | File, Underdog | File, Sleeper |
|---|---|---|---|---|---|---|
| QB pass yds × same WR rec yds | +0.38 [0.32, 0.42] | +0.44 [0.32, 0.62] | +0.06 to +0.13 | 0.40 (Flex 0.48) | [0.79, 1.05] | banned |
| QB pass yds × same WR receptions | +0.26 [0.20, 0.30] | +0.30 [0.20, 0.46] | +0.09 | 0.15 | [0.82, 1.05] | — |
| QB pass yds × same TE rec yds | +0.34 [0.26, 0.42] | — | +0.08 | 0.17–0.19 | [0.77, 1.05] | banned |
| QB pass yds × same RB rec yds | +0.24 [0.16, 0.32] | — | — | 0.31 (over-taxed) | — | — |
| QB completions × same WR receptions | +0.32 [0.28, 0.36] | +0.32 [0.16, 0.48] | — | not captured | [0.73, 1.05] | banned |
| QB pass yds × same RB rush yds | −0.16 [−0.24, −0.08] | — | −0.02 to −0.04 | untaxed | [1.03, 0.95] | — |
| Opposing QB pass yds × QB pass yds | +0.08 [−0.06, 0.22] | — | +0.01 | untaxed on Power (Flex 0.898) | [0.90, 1.10] | — |
| Opposing RB carries × RB carries | −0.26 [−0.34, −0.16] | — | −0.01 | untaxed | [1.16, 0.86] | — |
| Opposing RB rush yds × RB rush yds | −0.14 [−0.20, −0.06] | — | −0.01 | untaxed | [1.03, 0.95] | — |
| QB pass TDs × same WR TDs | — | — | — | 0.13 (m = 0.914) | [0, 0] banned | — |

- **Two MLB rows.** Same-team runs × rbi +0.18 (shipped +0.26); opposing pitcher runs allowed × batter runs +0.38. Sleeper bans both.
- **Break-even arithmetic.** At even picks with p = .5, a 2-pick Power at 3.5 needs P(both) ≥ 1/3.5 = .286, a lift of 1.143 over independence. A stack's EV is 3.5 × m × P_true, i.e. 0.875 × P_true / P_Underdog:
  - **Taxed pairs.** QB pass yds × WR receptions (true ρ .26–.30 vs Underdog .15) and QB pass yds × TE yds (.34 vs .17–.19) lift 1.07–1.09 and 1.10, for EV ≈ 0.94–0.96. QB pass yds × WR rec yds is taxed at the truth (0.87, independence-like). QB pass yds × RB rec yds is over-taxed.
  - **Untaxed pairs, played on the opposite sides.** QB pass yds × same RB rush yds lifts 1.10 (EV 0.96) and opposing RB rush yds lifts 1.09 (EV 0.95). Opposing RB carries, Over/Under, lifts 1.17 (EV 1.02; 0.96–1.07 across the ρ CI). It is the only pair type whose point estimate clears break-even.
- **Conditions.** All of this assumes legs at their book probability. It also assumes Underdog's untaxed both-higher quote means ρ_Underdog = 0 for the opposite-direction combination, not a negative ρ clipped at m = 1; only an owner in-app quote of the Over/Under combination settles that.
- **What the engine builds instead.** It builds the opposite: the same-direction Under/Under, the less likely combination, paid a stale bonus (S2-6).
- **Prior evidence.** Correlated combinations are the books' known exposure, which is why books ban or tax same-game combos (Davis, Dawson & Krieger 2018, DOI 10.5750/jpm.v12i2.1562). Sleeper bans and Underdog prices are that pattern.

**S2-8. The selected set: stored post-fix NFL entries.**

- **Effective sample.** The 417 resolved entries share 240 distinct legs (mean 5.1 appearances, max 52) over 6 dates, so the effective sample is a few dozen leg outcomes.
- **Joint calibration.** Underdog three-leg entries hit .020 against .249 priced (ratio 0.08 [0.03, 0.21]); two-leg entries .056 against .414 (0.13 [0.00, 0.20]). Sleeper's 29 three-leg entries come in at 0.35 [0.00, 0.64]. Re-priced at book marginals (R4), the ratios are 0.20–0.72 with every CI below 1. The selected legs beat neither their own read nor the book.
- **Leg hit rates.** Legs hit .392 slot-weighted (.45 distinct) against a `Win Prob` of .62 and a `Market Prob` of .485. Carries Unders hit .389; carries Overs went 0 for 11.
- **Money.** Realized return at the live price is 0.14 [0.06, 0.37] per $1 (−86%, n = 386 Underdog). Realized over priced EV is 0.06 [0.03, 0.16] under R0 and 0.17 [0.06, 0.42] under ALL.
- **R3 corroboration.** Working independently, R3 localizes a third of the single-leg gap in NFL Underdog Unders on lines far below consensus (`/tmp/researcher_train_serve_skew.md` TL;DR): the same legs.

**S2-9. What each repair does to the beam's selection** (re-run, sizes 2–3, post-fix pools):

| Variant | Entries at today's gates | days | priced all-hit | realized | realized / priced | return per $1 at live price | top-20 Jaccard vs R0 |
|---|---|---|---|---|---|---|---|
| R0 current | 1,283 | 11 | .297 | .023 | 0.08 [0.04, 0.18] | 0.18 [0.07, 0.42] | 1.000 |
| R1 symmetric Σ | 1,254 | 10 | .296 | .022 | 0.08 [0.04, 0.18] | 0.17 [0.09, 0.39] | 0.937 |
| R2 live tables | 1,841 | 10 | .294 | .036 | 0.12 [0.05, 0.29] | 0.25 [0.10, 0.62] | 0.674 |
| R3 Underdog m (replaces file modifiers) | 621 | 10 | .310 | .016 | 0.05 [0.02, 0.20] | 0.12 [0.04, 0.54] | 0.783 |
| R4 book marginals | 13 | 2 | .415 | .308 | 0.74 [0.24, 2.52] | 1.79 [0.67, 5.55] | 0.000 |
| ALL | 68 | 3 | .390 | .147 | 0.38 [0.00, 1.95] | 0.84 [0.00, 4.32] | 0.015 |
| R4b, ALL-b | none survive the prechecks (29 and 1,167 raw candidates) | | | | | | |

- **Looser floor (EV ≥ 1.0, books EV ≥ 0.9).** R0 keeps 5,996 entries: ratio 0.32 [0.15, 0.44], return 0.56 [0.28, 0.83]. ALL keeps 622: 0.31 [0.17, 1.24], return 0.54 [0.28, 2.20]. R4 keeps 98: 0.34 [0.00, 1.71].
- **No floor: the top 20 per (Date, Platform, League) by priced EV.** Returns are R0 0.90 [0.39, 1.48], R1 0.92, R2 1.08 [0.52, 1.66], R3 0.90, R4 1.20 [0.00, 3.37] (41 entries, 4 days), and ALL 1.27 [0.26, 2.37] (118 entries, 8 days). Every CI spans 1, and the variants cannot be told apart. The ungated top 20 reaches MLB cells the 2.0 floor never admits, which is why it beats the gated set.
- **Concentration.** R0's gated entries use 657 distinct legs (about 60 per day). Distinct legs hit .466 against a `Market Prob` of .527 and a `Win Prob` of .640; slot-weighted, they hit .336. The top 20 legs fill 27.5% of slots.
- **Bonus entries.** Gated entries carrying a stale bonus are not worse than the rest: realized over book-priced is 0.19 against 0.10.
- **Book-EV gates do not rescue it:**
  - Model EV ≥ 2 and books EV ≥ 1.0: 807 entries, return 0.14.
  - Books EV ≥ 1.0 alone: 3,424 entries, return 0.57.
  - Books EV ≥ 1.1: 1,843 entries, return 0.64.

**S2-10. The Model EV ≥ 2.0 floor is a curse amplifier.** NFL Underdog R0 beam candidates by priced-EV band:

| Priced EV | n | days | priced all-hit | realized | realized / priced | return per $1 |
|---|---|---|---|---|---|---|
| < 1.5 | 342 | 6 | .223 | .111 | 0.50 [0.13, 1.10] | 0.77 [0.19, 1.80] |
| 1.5–2.0 | 4,478 | 10 | .263 | .091 | 0.34 [0.13, 0.42] | 0.57 [0.23, 0.72] |
| 2.0–2.5 | 980 | 8 | .295 | .028 | 0.09 [0.04, 0.19] | 0.20 [0.08, 0.40] |
| 2.5–3.0 | 133 | 4 | .292 | .015 | 0.05 [0.00, 0.18] | 0.12 [0.00, 0.41] |
| ≥ 3.0 | 87 | 3 | .304 | .000 | 0.00 | 0.00 |

- **Rank correlation.** Within-day Spearman(priced EV, realized return) is −0.11 for NFL Underdog (4 days with ≥ 30 entries). Pooled it is −0.11 for Underdog (n = 6,155) and −0.09 for Sleeper (n = 575).
- **Pre-fix stored entries** (single-tier, ≤ 3 legs, deduped; 60,019 entries over up to 45 days, all leagues). Underdog realized/priced is 0.37, 0.28, 0.29 and 0.15 at priced EV 2–2.5, 2.5–3, 3–4 and 4–6. Sleeper is 0.35, 0.22, 0.16, 0.05 and 0.00 (the last at ≥ 6).
- **Recalibration slope** of logit all-hit on logit priced (Cox 1958, DOI 10.1093/biomet/45.3-4.562; Van Calster et al. 2019, DOI 10.1186/s12916-019-1466-7). Pre-fix Underdog is 0.335, a priced spread three times too wide. Sleeper is 0.955 with intercept −1.82, uniformly overstated. Post-fix Underdog cannot be identified (9 all-hit events in 386).
- **MLB shows no gradient** post-fix (Spearman +0.09 and +0.16, n = 131 and 236), because almost no MLB candidate passes 1.5.
- **Why the floor backfires.** Taking the best of many noisy, unbiased estimates guarantees disappointment after the decision, and the disappointment grows with the noise and with the number of alternatives (Smith & Winkler 2006). A floor on the estimate selects further into that tail. Raising the floor to fight overconfidence moves the wrong way, and lowering it to 1.0 still returns 0.56. No floor setting makes the beam profitable under served `Win Prob`.

**S2-11. Selection on book EV is cursed too, and the Trust proxy is the market.**

- **Book-EV admission.** Legs that ALL would admit (MP × payout at √3.5 > 1) hit 0.86 (Underdog) and 0.83 (Sleeper) of their `Market Prob`.
- **Trust-proxy admission.** Legs admitted on the Trust proxy hit 0.86 and 0.63 of TP.
- **Today's admission.** Legs admitted on `Win Prob` hit 1.01 of their `Market Prob`. MP is calibrated on legs selected by something else, not on legs selected by MP.
- **The Trust proxy's weights.** As its window grows, its weight on logit(`Win Prob`) falls from 0.41 to 0.19. Book logit goes from 0.22 to 0.34, and log payout sits at −1.35 to −1.47, about 0.7 on the platform's implied logit. That is the same collapse onto the market that R1 KILLED.
- **Consequence.** Book-anchored engine marginals are model-free, which the owner's rule excludes, and they are still cursed by their own selection: R4 at EV ≥ 1 comes in at 0.34 [0.00, 1.71].
- **Literature.** Profit requires a model that disagrees with the bookmaker and is right where it disagrees (Hubáček, Šourek & Železný 2019, DOI 10.1016/j.ijforecast.2019.01.001). Calibration on the bet set, not accuracy, drives returns (Walsh & Joshi 2024, arXiv:2303.06021, DOI 10.1016/j.mlwa.2024.100539).

**S2-12. Sleeper's bans look correlation-based, and they cannot be checked offline.** Every Sleeper entry in `banned_combos.json` is [0, 0]:

- **NFL same-team: 76 bans.** 21 have an empirical estimate. QB-receiver pairs run +0.20 to +0.44; three sit near zero (QB rushing yds × RB carries 0.00, same-team RB carries −0.02, RB carries × RB rush yds −0.06).
- **MLB same-team: 13 bans,** at +0.02 to +0.18.
- **MLB opposing: 55 bans.** Pitcher-vs-batter |ρ| runs up to 0.38–0.48; some sit near zero (pitcher walks allowed × batter hits −0.04).
- **NBA opposing: 30 bans,** with no data.

This matches the industry pattern of banning correlated same-game combos. No Sleeper slip capture exists in the repo, so validity is an owner in-app check. A wrong near-zero ban only narrows the candidate pool.

### Verdict per repair

| Repair | Calibration (pool B, served legs) | Holm | Selection change | Verdict |
|---|---|---|---|---|
| R1 symmetric Σ | none; worse on admitted legs | not rejected (p .12) | Jaccard 0.94 | **KILL** |
| R2 live payout tables + caps | none (EV ratio worse: the two wrongs stop offsetting) | not rejected (p .28) | Jaccard 0.67, +43% gated volume | **GO as truth** (I5a), not as a calibration repair |
| R3 Underdog m | none (m < 1 on 3–7%, ≈ 0.99) | not rejected (p .052) | Jaccard 0.78, mostly from dropping file bonuses | **KILL as a module** (I5c); m stays a settlement term (I5d) |
| R4 book marginals | calibrates every size, both pools | rejected (p .0005) | Jaccard 0.00, gated volume 1% of R0 | **diagnostic only**: model-free (owner rule, R1 KILL), and cursed under its own selection |
| R4b Trust proxy | calibrates (collapses onto the market) | rejected (p .0005) | nothing survives the prechecks | **not a candidate** (R1 KILL) |
| ALL (primary) | calibrates every size | — | Jaccard 0.015 | inadmissible (contains R4); the admissible part (R1 + R2 + R3) fails acceptance |

### Recommendation: ranked I5 list

1. **I5a-1, per-pick convention.** **GO.** Engineering; lands after the owner confirms the live table in the app.
   - **File:** `helpers/distributions.py`, `UNDERDOG_BOOST_BASELINE` 1.78 → √3.5 = 1.8708, with its `_comment`.
   - **What the constant feeds:** single-leg `Model EV`, Kelly, the 5% rule, the payout-implied `Market Prob` of one-sided rows, `helpers/archive._dfs_offer_probs`'s devig of one-sided Underdog quotes, and leg admission in `prediction/correlation._select_bet_offers` (via leg `Model EV` / `Market EV`).
   - **Move them together.** If the payout root changes but the payout-implied MP does not, about 800 one-sided unquoted Underdog rows (2%) get a spurious +5% book EV.
   - **Tests and era:** golden pins on 1.78-derived numbers move. Tag the change as a new selection era.
   - **Effect:** S1-c.
2. **I5a-2, live tables.** **GO as truth, not as a calibration fix.**
   - **`data/config/underdog_payouts.json`:**
     - Power 2–8 = 3.5 / 6.5 / 12 / 20 / 35 / 65 / 120.
     - Flex 3–8 at the live tiers (S1-a), dropping the tiers that do not exist live.
     - Drop the `insurance` alias and date the `_comment`.
   - **`prediction/payouts.py`:**
     - `_pooled_underdog_curve` runs to 8 picks and makes the Power-vs-Flex choice at live terms.
     - `expected_payout_with_pushes`: the Flex k-loss tier multiplies only the n − k largest pick multipliers.
   - **Expect:** +43% gated volume and no detectable change in realized return (S2-9). Era tag.
3. **I5a-3, pair modifiers.** **GO as truth.**
   - **File:** the Underdog section of `data/config/banned_combos.json`.
   - **Values > 1 → 1.0** everywhere. Underdog's formula has m ≤ 1, and the live capture returns exactly 1.0 for opposite sides and negative association. This covers NFL 86 of 108 same-team and 10 of 10 opposing, MLB 49 of 49 same-team, and NHL 228 of 235 same-team.
   - **Same-direction sub-1 values on pairs Underdog quoted untaxed → 1.0.** This covers opposing QB pass yds × QB pass yds on Power (file 0.9).
   - **Opposite-direction sub-1 values on the negatively correlated pairs wait for the owner's Over/Under quote** (open question 2). These are QB pass yds × same RB rush yds 0.95, opposing RB rush yds 0.95, and opposing RB carries 0.86. Whether Underdog taxes that combination is exactly what decides S2-7.
   - **Bans on pairs Underdog prices are lifted only after an owner re-quote** through the Modifiers reconciler (QB pass TDs × same WR TDs: one quote, m = 0.914).
   - Corrections stay on the reconciler path (`docs/handoffs/dfs-products.md` §3–4; no automated authed probing, owner rule 2026-07-10).
4. **I5d, settlement truth.** **GO.**
   - **File:** `strategies/_ledger_settlement.py::realized_multiplier`.
   - **All-hit Power:** settle at T_live(n) × Π b_i × m, using the legs' own multipliers. Each `parlay_hist` leg carries `boost`; confirm on a committed record that `canonical_legs` keeps it.
   - **Fallback:** where a record lacks per-leg multipliers, use its `payout_multiplier` (the priced `Boost`, already on the committed record via `strategies/ledger.py`) re-based to the live table, × T_live / T_file.
   - **Flex k-loss:** settle at the tier × the n − k largest b_i.
   - **m:** stays 1 except on Underdog-taxed same-game NFL pairs, where it moves current selection by ≤ 0.1% of payout.
   - **Size of the fix:** stored winners are underpaid by a median 1.68× today (S2-6). Records committed before the fields exist settle as today and are flagged; the ledger is append-only.
5. **I5e, premise rewrite.** **GO.**
   - `docs/handoffs/parlay-dependence.md:21–28` (money logic) and `:88–91` (volatile assumption): text below.
   - `docs/PARLAY_AUDIT.md`: the §2.4 open item closes with I5a, and §1.3 states the cap rule.
6. **I5b, symmetric Σ.** **KILL.** No build. If the parlay-dependence lane unparks, fold equal weighting into its Σ rebuild (pair-type pooled tetrachoric ρ with hierarchical shrinkage; R3 copula brief).
7. **I5c, `prediction/underdog_tax.py` + `underdog_correlation_tax.json`.** **KILL** as a pricing repair. The ρ-per-pair table stays documented in `docs/underdog_api.md` §6.8. It comes back with the parlay-dependence lane only if the engine starts targeting same-game positive stacks, where m becomes first-order.
8. **`_MODEL_EV_FINAL_FLOOR` (2.0) and `_BOOKS_EV_FLOOR` (0.9) in `prediction/parlay.py`.** Owner decision; no change recommended. No tested value of either returns ≥ 1 on the selected set, and raising either selects deeper into the curse (S2-10).
9. **Acceptance re-scope.** The plan's I5 number, realized/priced joint ratio in [0.9, 1.1] at every entry size, cannot be met by any pricer repair while legs are served `Win Prob` (S2-3, S2-9).
   - **Move the joint-ratio target to I6** (leg calibration on the selected set). Measure it on this brief's pool B and beam re-run; the scripts are in the scratch directory.
   - **I5's own acceptance** becomes: priced payout equals the quote on owner-captured slips (one Power, one Flex, one stacked same-game pair), and the ledger settles at it.

### Premise rewrite, for `docs/handoffs/parlay-dependence.md:21–28` and `:88–91`

Money logic (replaces "the DFS apps largely don't tax leg correlation … Better joint pricing multiplies the EV of every entry on both apps"):

> Underdog does tax same-game correlation: it prices each same-game pick set with its own Gaussian ρ per pair type and pays T × Π b × m with m ≤ 1, and Sleeper bans the correlated pairs outright, so a correlated stack earns its independence EV times P_true / P_platform — at most break-even on the one untaxed pair type where the truth departs from the platform (opposing RB carries Over/Under, EV 0.96–1.07 at even picks) and never enough to carry an entry without leg edge. Parlay EV is therefore leg EV compounded: post-fix, admitted legs read 8–11 pp above their hit rate, the joint over-prediction grows with entry size (realized/priced 0.71–0.77 at two legs, 0.41–0.44 at five), and no Σ or payout repair moves it; only selection-calibrated leg probabilities do.

Volatile assumption (replaces "Apps don't fully tax leg correlation. The whole lane's edge."):

> Platforms price leg correlation (Underdog's per-game m, Sleeper's bans). Re-verify the payout table and the pair-type ρ by owner in-app quote each season (`docs/underdog_api.md` §6.8, §7.4); any edge in this lane lives only in pair types where the platform's ρ departs from the hit-event ρ at posted lines.

### Reality checks (stage 2)

- **Effect sizes and regime.** R4's calibration holds on random combinations: pool A unselected, pool B admission-filtered. It does not hold on what a bettor or the beam builds; on the beam's own output nothing is calibrated, R4 included. Per parlay, the leg swap moves log-loss by only −0.0034 nats in pool A, about seven times that on admitted legs (−0.024), and eight times more in NFL than in MLB.
- **Few clusters.** NFL rests on 11 days, and the stored set on 6. A day-clustered percentile bootstrap with ≤ 11 clusters under-covers (Cameron, Gelbach & Miller 2008, DOI 10.1162/rest.90.3.414), so the NFL CIs here are directional. MLB has 24–30 days.
- **Reconstruction.** The rebuilt Σ matches stored `Corr Pairs` on 1.7% of entries. The beam re-run omits the Kelly gate and re-implements the admission caps. It does reproduce production's NFL-only post-fix output and the stored ratio (0.08), which is the check that matters.
- **Payout truth.** It rests on one live capture (2026-09-10, one account, one state). The Underdog ρ table covers NFL pair types only; MLB, WNBA, NHL and NBA m are assumed to be 1.
- **What could make this wrong:**
  - The live table is state-specific or has changed since the capture.
  - Underdog's untaxed quotes are clipped negatives rather than ρ = 0. That changes only S2-7.
  - NBA, where pace and blowouts create the strongest same-game dependence (the R3 copula brief's t-tail question), has no post-fix data and could make Σ matter.
- **Build cost.** I5a-1/2/3, I5d and I5e come to about 1–2 engineering days, and the golden pins move. There is no research component. KILLing I5b and I5c saves the new module and its config.

### What was tried and failed

- **Reproducing stored `Corr Pairs` from rebuilt matrices.** Only 1.7% came within 0.01, so the stored-parlay arms price on the stored Σ with R1 as a delta.
- **R4b as an engine marginal.** It calibrates and is rejected under Holm, but only by collapsing onto the market (weight on logit WP 0.41 → 0.19). It was killed with R1.
- **Book-EV gates and other Model EV floors as selection rescues.** Every setting tested returned 0.14–0.64. The ungated top-20 per cell (0.90–1.27) cannot be told apart from 1 or from each other.
- **Pair-type pooled NFL Σ** (exploratory): −0.00031 [−0.00066, +0.00008].
- **Post-fix calibration slope.** It cannot be identified from 9 all-hit events.
- **Sleeper ban validity and Underdog MLB/WNBA m.** There is no data on disk. They need owner quotes; automated authed probing is ruled out.
- **Flex (multi-tier) parlay calibration.** `parlay_hist` stores Model EV / Boost, which for Flex includes the partial tiers and is not P(all hit). It cannot be tested without the tier distribution.

### Open questions / caveats

1. **Live table.** The owner confirms the Power/Flex table in the app for his state. This blocks I5a-1 and I5a-2.
2. **Untaxed pair quotes.** Owner quotes are needed through the Modifiers reconciler: Over/Under on the untaxed pair types (opposing RB carries, opposing RB rush yds, QB pass yds × same-team RB rush yds), and a re-quote of QB pass TDs × same WR TDs. They decide whether any structural pair edge exists (S2-7) and fill the I5a-3 rows.
3. **Other leagues and Sleeper.** Underdog m for MLB/WNBA/NHL/NBA pair types and the validity of Sleeper's bans are unknown; both need owner quotes.
4. **NFL Σ estimation.** Pair-type pooled tetrachoric ρ at posted lines, with hierarchical shrinkage and attention to discrete-margin ties, becomes a pre-registered parlay-dependence research item. At 11 NFL days its gain cannot be detected.
5. **Beam floor stance (owner).** The evidence says no floor setting, in either direction, makes the beam profitable.
6. **Mid-November confirmatory re-run.** Re-run pool B, the beam re-run and S2-10 on the I2 decision-time columns once there are ≥ 8 NFL weeks. MLB ends in late October. NBA needs its own window from late November.
7. **Parlay Kelly.** Parlay Kelly still sizes from the raw copula probability (`docs/PARLAY_AUDIT.md` §2.6). With the selected set realizing 0.08 of priced, this is the parlay path's live-money exposure until I6 lands.

### Cross-league caveats

- **NFL:** Σ comes from 7–11 games per team. The leg swap's Δ log-loss is 8× MLB's. There are 6–11 clusters. The curse gradient (S2-10) is clearest here.
- **MLB:** the best-powered league (24–30 days). It has had no post-fix production parlays because of the floor. Admitted legs are calibrated against `Market Prob` (O/E 1.00–1.02). The season ends in late October.
- **WNBA:** too small to read, and the season is over.
- **NHL:** 722 settled rows, not evaluated. 228 of 235 Underdog same-team file modifiers carry stale bonuses.
- **NBA:** no post-fix data, no Underdog NBA entries in `banned_combos.json`, and Underdog's NBA ρ is unknown. Nothing in this brief transfers to NBA without a re-run.

## Bibliography

| Source | Identifier | Used for |
|---|---|---|
| Smith, J. E., Winkler, R. L. (2006). The optimizer's curse: skepticism and postdecision surprise in decision analysis. *Management Science* 52(3), 311–322. | DOI 10.1287/mnsc.1050.0451 | selection on noisy edge; the floor as curse amplifier |
| Kruskal, W. H. (1958). Ordinal measures of association. *JASA* 53(284), 814–861. | DOI 10.1080/01621459.1958.10501481 | the 2 sin(πρ_s/6) remap in `training/correlate.py` |
| Genest, C., Nešlehová, J. (2007). A primer on copulas for count data. *ASTIN Bulletin* 37(2), 475–515. | DOI 10.2143/AST.37.2.2024077 | ties in discrete stats break rank–copula identities |
| Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation coefficient. *Psychometrika* 44(4), 443–460. | DOI 10.1007/BF02296207 | hit-event (tetrachoric) ρ at posted lines |
| Plackett, R. L. (1954). A reduction formula for normal multivariate integrals. *Biometrika* 41(3–4), 351–360. | DOI 10.1093/biomet/41.3-4.351 | bivariate orthant computation in the re-pricing |
| Genz, A. (1992). Numerical computation of multivariate normal probabilities. *J. Comput. Graph. Stat.* 1(2), 141–149. | DOI 10.1080/10618600.1992.10477010 | MVN orthant (the shipped `multivariate_normal.cdf`) |
| Higham, N. J. (2002). Computing the nearest correlation matrix — a problem from finance. *IMA J. Numer. Anal.* 22(3), 329–343. | DOI 10.1093/imanum/22.3.329 | PSD repair of 3-leg Σ |
| Schäfer, J., Strimmer, K. (2005). A shrinkage approach to large-scale covariance matrix estimation and implications for functional genomics. *Stat. Appl. Genet. Mol. Biol.* 4(1), Art. 32. | DOI 10.2202/1544-6115.1175 | shrink toward a structured target, not zero |
| Efron, B., Morris, C. (1973). Stein's estimation rule and its competitors — an empirical Bayes approach. *JASA* 68(341), 117–130. | DOI 10.1080/01621459.1973.10481350; `docs/operation_ship_references.md` [47] | empirical-Bayes shrinkage of pair-type ρ |
| Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *REStat* 90(3), 414–427. | DOI 10.1162/rest.90.3.414 | few-cluster under-coverage (NFL) |
| Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Stat.* 6(2), 65–70. | JSTOR 4615733 | multiplicity across R1, R2, R3, R4, R4b |
| Cox, D. R. (1958). Two further applications of a model for binary regression. *Biometrika* 45(3–4), 562–565. | DOI 10.1093/biomet/45.3-4.562 | calibration slope |
| Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., Steyerberg, E. W. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine* 17, 230. | DOI 10.1186/s12916-019-1466-7 | calibration-in-the-large vs slope |
| Davis, J., Dawson, J., Krieger, K. (2018). Correlated parlay betting: an analysis of betting market profitability scenarios in college football. *J. Prediction Markets* 12(2), 68–84. | DOI 10.5750/jpm.v12i2.1562 | correlated parlays are the books' known exposure; bans and taxes |
| Hubáček, O., Šourek, G., Železný, F. (2019). Exploiting sports-betting market using machine learning. *Int. J. Forecasting* 35(2), 783–796. | DOI 10.1016/j.ijforecast.2019.01.001 | profit needs disagreement with the book that is right |
| Walsh, C., Joshi, A. (2024). Machine learning for sports betting: should model selection be based on accuracy or calibration? *Machine Learning with Applications* 16, 100539. | arXiv:2303.06021; DOI 10.1016/j.mlwa.2024.100539 | calibration on the bet set drives returns |
| Chong, Y. Y., Hendry, D. F. (1986). Econometric evaluation of linear macro-economic models. *Rev. Econ. Stud.* 53(4), 671–690. | DOI 10.2307/2297611 | forecast encompassing (via R3) |
| In-repo: `docs/underdog_api.md` §6.8, §7.4 (live capture 2026-09-10); `docs/archive/researcher_trust_layer.md` (R1); `/tmp/researcher_train_serve_skew.md` (R3); `docs/archive/researcher_selection_shrink.md`; `docs/archive/researcher_copula_stage0.md`; `docs/PARLAY_AUDIT.md`; `docs/handoffs/dfs-products.md` §3–4 | — | live payout and ρ facts; sibling verdicts |

**Artifacts** (scratchpad `r2/`):
- Shared helpers: `base.py` (paths), `boot.py` (day-clustered ratio / mean-difference / p-value bootstraps), `bvn.py` (Plackett bivariate normal CDF, tetrachoric inversion), `pricing.py` (`joint2`, `joint3`, `psd3`), `udtax.py` (Underdog ρ per pair type, §6.8).
- Stage 1: `stage1.py`, `stage1b.py`, `stage1_ud.csv`.
- Legs and pair types: `legs_post.parquet`, `adm_calib.py` → `legs_post_adm.parquet` + `adm_calib.log`, `keys.py` → `legs_keys.parquet`, `pair_types_post.csv`, `nfl_long_pair_types.csv`.
- Σ: `sigma.py`, `validate_sigma.py` → `sigma_validation.parquet`, `wf_sigma.py` → `wf_sigma.pkl` (walk-forward `_correlate_teams`).
- Trust proxy: `trust_proxy.py` → `legs_keys_tp.parquet`.
- Synthetic: `synth.py` → `synth.parquet`, `synth_eval.py` → `synth_eval.log`, `synth_dll.csv`, `synth_eval_in.parquet`; `ev_eval.py` → `ev_eval.log`, `synth_ud_ev.parquet`; `synth_league.py` → `synth_league.log`.
- Stored: `stored_reprice.py` → `stored_reprice.pkl`; `curse.py` → `curse.log`.
- Beam: `beam_rerun.py` → `beam_rerun.parquet` + `beam_rerun.log`; `beam_eval.py` → `beam_eval.log`.
- Code read (read-only): `prediction/correlation.py`, `prediction/parlay.py`, `prediction/payouts.py`, `prediction/offer_records.py`, `training/correlate.py`, `helpers/config.py`, `strategies/_ledger_settlement.py`, `strategies/_ledger_selection.py`, `strategies/underdog_pickem.py`, `data/config/banned_combos.json`, `data/config/underdog_payouts.json`, `data/config/sleeper_payouts.json`.
