# Lane record — low-weight models: the model legs the book is hiding

**Status: ACTIVE (opened 2026-09-01).** Context spine: CLAUDE.md,
[docs/ARCHITECTURE.md](../ARCHITECTURE.md), [ship_gate.md](../ship_gate.md),
[model_improvement_track.md](model_improvement_track.md) §6.5 / §8.2.

## Mission

Breadth-75 is met in every league, but some cells pass the six ship gates only because the
model↔book blend weight `model_weight` sits at its 0.05 floor: the served predictive is the
sportsbook line and the cell's model contributes nothing. Diagnose those model legs and repair
them until each cell passes all six gates with the **free** weight fit landing at **≥ 0.3**. No
forced floor — the model has to earn the weight.

## Owner constraints (binding)

1. **Every lever is testable through `ship sweep`.** No per-cell hand fixes. Levers are values
   on existing sweep axes (`sn_param`, `blending_loss_fn`, `posthoc`, `normalization`, `dist`);
   a new method enters as a slug in the matching pool, wired through the three control sites
   (`_CONTROL_FLAGS`, `runtime_controls`, the CLI resolve), and a method piloted on one cell is
   graduated by the `experiment-graduation-specialist`. A bug fix that changes how an artifact is
   produced is an `implementation_version` bump so the board re-scores the whole family.
2. **Everything keeps working as training data accumulates.** Thresholds are minima or ratios,
   never tuned to today's row counts; guards relax on their own as evidence arrives; verdicts are
   re-earned per sweep on the current matrix hash; scripts derive their cohort from the current
   artifacts, never a pinned cell list.

## What the weight is

`model_weight` is a precision-pool weight (`fused_loc`), not a probability share; a cell at
`w = 0.05` still carries 30–45% of the model's departure from the book in probability space. It
is fit on authentic-quoted validation rows only (`calibration.fit_blend_weight`; bounds
`[0.05, 0.9]`); `1.0` means no authentic validation quotes. It never enters stake sizing.

## The census (authentic test rows; Spearman vs `Result`)

Reproduce with `poetry run python research/scripts/blend_weight_probe/census.py`; the cohort is
derived from the class labels and `w` on every run.

| Cell | w | ρ(model) | ρ(Line) | ρ(Mean10) | Read |
|---|---|---|---|---|---|
| NFL attempts | 0.05 | −0.03 | 0.29 | 0.13 | model leg is noise; 18 negative means; diverged SkewNormal head |
| MLB hits allowed | 0.05 | 0.08 | 0.19 | 0.11 | 139 negative means, CITL 1.23, shape_ratio 4.99 |
| NFL passing yards | 0.05 | 0.14 | 0.33 | 0.16 | weak small-n model (n=358); the 1-SE rule floors `w` by design |
| MLB batter strikeouts | 0.05 | — | — | — | one authentic test row; weight fit on two validation rows (feed blackout) |
| NHL skater fantasy (UD) | 0.33 | 0.01 | 0.16 | 0.05 | model mean 0.65× the line; skew_cal railed |
| MLB runs allowed | 0.63 | −0.01 | 0.14 | 0.10 | model leg is noise |
| NBA BLK | 0.80 | 0.31 | 0.35 | 0.37 | below naive, and the book leg is a known ingestion defect |
| NHL hits | 0.78 | 0.22 | 0.39 | 0.30 | below naive |

## Why the sweep lands there

Board slack is the minimum gate headroom on the served predictive and Gate 1 is zero at the book,
so on a weak-model cell the lowest weight wins. The ledger has passed over 6/6 arms at healthy
weights twice (MLB hits allowed DPO at 0.90; NFL passing yards at 0.315), and the live S2/S3
supersession tests cannot separate two legs that both sit on the book at the line. The
`--min-model-weight` knob on `ship sweep` (nomination constraint, ship-time check, S1-only
waiver against a book-riding incumbent) is the lane's answer; see [ship_gate.md](../ship_gate.md).

## Guards landed by this lane

- Weight fit requires `calibration._ONE_SE_MIN_CLUSTERS` distinct authentic validation clusters
  (players; dates for team markets); below it the cell serves model-only (`w = 1.0`), logs why,
  and its skill / Kelly columns are NaN — the same rows are no book to score against (batter
  strikeouts carried `kelly_shrinkage` 0.052 from two rows). `n_authentic_validation` and
  `n_blend_fit_clusters` are persisted per cell (`model_stats`; the board carries the rows).
- The cluster ids behind that fit — and the 1-SE clustered SE and the post-hoc cluster folds —
  now align by label. `_blend_fit_clusters` picked positionally against an index-sorted mask,
  so on the production path 88–99% of rows carried another player's id (pre-existing; the
  cross-fit board was unaffected, so no `implementation_version` bump; served `crps_1se` weights
  refit honestly on the next retrain).
- Gate 1 / BSS blank on fewer than the same number of distinct priced players (dates when the
  dump has no `Player`) in `scorecard._priced_rows` — a one-row book is no book.
- The test-set dump carries the book base mean (`Book_EV`) and the model-only shape
  (`SN_Sigma_model` / `SN_Alpha_model`, `R_model`, `Gate_model`, `DP_PHI_model`) so
  `research/scripts/blend_weight_probe/reblend.py` can re-run the pool at any `w`.

## Per-cell status

The re-predict fork (`repredict.py`) is closed for all three SkewNormal cells: the serve chain
reproduces the dumped model mean to float noise (max |Δ| ≤ 2e-14; NHL skater fantasy has one
float32 boundary row at 4e-3), so the broken legs are **training pathologies, not decode bugs**,
and the lever is a sweep axis value, not a version bump. Head detail: MLB hits allowed's centered
`mean` head goes negative on 139 rows (range −2.5…4.6 in ratio space) with 0.7% of rows at the
sd ceiling; NFL attempts sits at its per-cell scale ceiling on 4.6% of rows; NHL skater fantasy
rails nowhere but under-predicts by 32% specifically on the book-quoted rows (level 0.95× overall,
0.68× on the 234 authentic rows) — its matrix is 86% `derived` (combo EV inversion) quotes with
no authentic rows at all in 2025.

| Cell | Class | Lever | Status |
|---|---|---|---|
| MLB hits allowed | BIASED (CITL 1.23) | re-confirm the ledger's DPO arm under the knob | fork closed: training pathology |
| NFL passing yards | WEAK | re-confirm the `ratio_meanyr / direct / nll / none` arm post-leak | 2026-09-02 sweep under the floor: 14/15 scored corners fit `w = 0.05`, the other 0.145 (325 authentic validation rows; the 1-SE rule floors it), so nothing nominated; keeps its coverage ship, lifts as seasons accumulate |
| NFL attempts | BROKEN | `sn_param: centered` / count corners under the knob | fork closed: training pathology |
| NHL skater fantasy (UD) | BIASED (CITL 0.68) | centered / count corners under the knob | fork closed: training pathology; quote classes checked |
| MLB runs allowed | BROKEN | sweep under the knob | pending |
| NHL hits | BROKEN | sweep under the knob | pending |
| NBA BLK | BROKEN | book-side packet first | blocked on the ingestion packet |
| MLB batter strikeouts | DATA | guard result stands until the feed returns | retrained 2026-09-02: `w = 1.0` (2 authentic rows / 2 clusters), Gate 1 blank, skill / Kelly NaN; ships model-only as a book-less cell |

The census (50 book-quoted cells) also flags a second wave outside the owner's cohort: NHL
powerPlayPoints (NegBin, CITL 0.67), NFL tds (NegBin, 0.76), NFL interceptions (DPO, 0.83), MLB
stolen bases (NegBin, 0.84), and — provisionally, because their dumps predate `Gate_model` and the
deflation used the blended gate — MLB home runs (1.73) and NHL goals (1.37). Re-run the census after
the next `meditate` regenerates the dumps before acting on the last two.

## Owner packets (not fixed in this lane)

- **NBA BLK / STL ingestion.** Re-checked on the archive 2026-09-01 (sportsbook rows only,
  betmgm / draftkings / fanduel / bovada): at line 0.5 the stored `ev` averages 0.92 while
  `under_prob` averages 0.73; at line 1.5, `ev` 2.0 against `under_prob` 0.68; 34% of rows carry
  no `ev` at all. No count distribution puts P(0) = 0.73 on a mean of 0.92 — the two columns come
  from different producers and disagree, the `nfl_count_quote_columns_disagree` pattern. The
  2026-06-02 real-price repair did not reach this market. No BLK model verdict is trustworthy
  until the book leg is; the fix is on the `confer` / archive-seed side.
- **MLB batter strikeouts feed.** `betmgm` is the only sportsbook that has ever priced this
  market in the archive (Mar–May 2026: 223 / 7971 / 5149 rows), it drops to 2 rows in June and
  none after (13 stray `betrivers` rows in August), while `hits allowed` keeps betmgm / bovada /
  draftkings / fanatics through the same window. The Odds API key `batter_strikeouts` is still
  mapped in `stat_map.json`, so the market is requested; the book set in `prop_books.json` simply
  no longer contains one that prices it. Owner call: add a book that prices batter strikeouts, or
  accept the cell as book-less (it serves model-only under the cluster floor).

## Do not retry without new evidence

Raising `_MODEL_WEIGHT_MAX`; a Brier-at-line weight objective; a second book-quality weight;
shaped book inside the blend; BLP / decoupled blend; model-only quality as an HPO objective
(Goodharts). Lowering `_MODEL_WEIGHT_MIN` is an owner policy call. Sources:
`docs/archive/researcher_blend_weight_slug.md`, `docs/archive/researcher_passing_g1.md`,
[mean_corrector_stage_order.md](mean_corrector_stage_order.md).

## Follow-ups outside the lane

- The `WEAK` cells (model at par with or slightly below the line) are the honest sharp-book case;
  the levers there are model-quality research (hierarchical pooling, more seasons).
- `model_prob._blend_with_book` applies the scalar `w` to every row and still pools the book's
  constant-CV dispersion on rows with no quote, while training pins such rows at `w = 1.0`.

## Changelog

- 2026-09-02 guards, cluster-id alignment fix, `--min-model-weight`, probes landed on devel.
- 2026-09-01 lane opened.
