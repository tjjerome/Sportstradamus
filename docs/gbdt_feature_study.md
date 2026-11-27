# Applying the GBDT feature guide to this repo: a measured study (2026-11-16)

This report answers one question: how can the data we already gather become better signal?
It looks past column pruning, and past the training matrices to how rows are collected and
built. It reads [gbdt_feature_engineering.md](gbdt_feature_engineering.md) against the
pipeline and then tests the guide's advice on the dev box's current matrices, test sets, game
logs and serve-time feature log.

It is a point-in-time study. No code, config or model changed. Every number comes from a
read-only script saved under `~/backups/sportstradamus/2026-11-16-gbdt-feature-study/` (§13).
Every keep-or-kill call rests on the rows we price: the props an app or sportsbook posted, at
the posted line (§2). §7.3 and §7.4 answer two follow-up questions: does the pipeline use the
play-level data it fetches to its full potential, and what would NBA.com's play-by-play and
tracking endpoints add? §11 applies the findings to the profit question and reads the live
record through the games of 11-15. §14 lists the defects found on the way. §15 lists earlier
decisions, made before the priced-row rule, that it may overturn.

## 1. The short answer

1. **The inputs are not the bottleneck. The way the mean is fit is.** In 12 of 13 NBA and NFL
   cells, a plain LightGBM regression fed the served model's exact inputs predicts the stat
   better than the served model's own mean, by 1.5 to 7 percent of squared error. It does so
   with untuned settings and fewer training rows. On the rows we price it cuts the error by 2
   to 7 percent in NBA PRA and five NFL cells, and the Brier score at the apps' posted lines by
   16 to 30 × 10⁻⁴ in NBA PRA, REB and FG3M and NFL receptions; no cell reads worse. NHL and
   MLB show no gap. The served hyperparameters are not the cause (§4.3, §4.4).
2. **Data we already store adds a little, some of it at the line.** Shot attempts in NHL,
   minutes history and per-minute rates in NBA, lineup-spot history and handedness in MLB each
   cut the error on priced rows by 0.1 to 0.7 percent. NHL shot attempts also cut the Brier
   score at the line by 5 to 8 × 10⁻⁴. Learning curves have flattened at the current sizes, so
   more rows of the same kind would add little.
3. **Play-level data that collection downloads and throws away is worth more, in a few
   cells.** The pitch counts and times through the order of an MLB starter's recent outings
   cut the error on pitches thrown by 4.2 percent. The pitcher's own pitch-level record and the
   opposing lineup's strikeout rates cut it on pitcher strikeouts by 2.7 percent, and the
   Brier score at the priced line by 36 × 10⁻⁴. Faceoffs taken cut it on NHL faceoff wins by
   2.5 percent, and power-play ice time cuts both error and Brier on NHL points and power-play
   points. The game-log update already downloads all of it, so keeping it needs no new live
   request (§7.3). NBA.com's play-by-play and tracking endpoints, which the pipeline never
   calls, add little: across 48 candidates in 14 NBA cells they help NBA assists on priced
   rows and the minutes model, whose market no app posts, and cost PRA, PR and REB (§7.4).
4. **The largest effect on hitters is knowledge the board never prices.** Knowing which MLB
   hitters start today is worth 2.6 percent of squared error on hits, more than every other
   hitter candidate combined. The apps pull a non-starter's props about three hours before
   the game, when lineups post, so serving never prices a posted line with that knowledge
   (§7.2).
5. **Judged on the rows we price, more candidates survive than the all-row screens showed.**
   The plain mean, a fit through the newest rows, NHL shot attempts, MLB handedness, NBA
   per-minute rates for assists and most MLB and NHL play-level sets lower the error at the
   posted line. Row re-weighting, quoted-only training, the book's mean alone as a label and
   most NBA.com families do not. Distance to the book is reported as information, never as a
   verdict (§2).
6. **Correctness items surfaced on the way.** Serving marks most NBA home teams as away,
   because its schedule source lists only the games with an international broadcast (§8).
   Underdog's NHL shots props never reach the board, because its label for them is missing
   from `stat_map.json`, and four MLB pitcher labels Underdog renamed on 09-29 will miss
   Opening Day the same way. Unquoted NBA training rows are chosen by the minutes the player went
   on to play in that game (§6.1). §14 lists every defect found.
7. **None of it is an edge yet.** Recommended legs have lost 9.5 percent a leg since 10-09,
   and the repo's own offline replay reads the same rule at .51 to .52 on posted standard
   picks against a claimed .60. §11 sets out what the record supports and the order of work.
8. **Earlier decisions may not survive the same test.** Many model decisions made before this
   study were judged on all held-out rows, on what a candidate adds beyond the book, or on the
   recommended legs alone, and some counted a move toward the book against a candidate. The
   most exposed are the blend-weight floor that holds NFL passing yards withheld, the gates
   that pick each cell's recipe, the live-gap lane's pulls toward the book and Part B's
   feature screen (§15).

| Finding | Size | Whose call |
|---|---|---|
| Served mean loses to a plain regression on the same inputs (§4) | 1.5–7 % MSE, 12 of 13 NBA/NFL cells; 2–7 % on priced rows; 16–30 × 10⁻⁴ Brier at posted lines in 4 cells | Owner: touches target normalization, research-first |
| Final fit stops at the 70 % mark (§5) | about 1 % MSE in 3 of 8 cells; 8–10 × 10⁻⁴ Brier at posted lines in NFL receiving yards | Owner: split design |
| NBA serve-time `Home` flag (§8) | wrong on 41 % of logged rows; cause found | Owner: serve-side fix |
| App labels missing from the stat map (§14) | 5,695 Underdog NHL shots props since 09-19, none scored; four MLB pitcher labels renamed 09-29 | Owner: config fix |
| Recommended legs by kind (§11) | −5.0 % a leg on standard picks, −7.6 to −13.5 % on priced legs | Owner: serving rule |
| NBA rows chosen on realized minutes (§6.1) | 53 % of rows affected; effect on the mean not shown | Inside the feature grant |
| Stored but unused columns (§7.1) | 0.1–0.7 % MSE on priced rows; 5–8 × 10⁻⁴ Brier at the line in NHL shots | Inside the feature grant |
| MLB lineup of the day (§7.2) | 2.6–2.9 % MSE; the apps pull non-starters' props when lineups post | Owner: log the lineup source; starters' spot only |
| Play-level data discarded at collection (§7.3) | 2.5–4.2 % MSE in three cells; 36 × 10⁻⁴ Brier at the line on pitcher strikeouts | Inside the feature grant; a new source needs the owner's licence call |
| NBA play-by-play and tracking endpoints (§7.4) | 0.7 % MSE and 7 × 10⁻⁴ Brier on priced NBA AST rows; costs PRA, PR, REB; 0.5 % on minutes, which no app posts | None needed now: AST waits for more posted rungs |
| Earlier decisions judged another way (§15) | one pull toward the book, set aside in October, cut Brier on quoted legs by 60–100 × 10⁻⁴ | Owner: which to re-judge first, and whether #0d stands |

## 2. How to read the numbers

- **Cell**: one league and market, such as NBA PRA.
- **Held-out rows**: the newest 30 percent of a cell's matrix by date. The served model never
  trained on them. The test-set CSV holds about half of them, and every comparison with the
  served model uses that half.
- **Served mean**: the `EV` column of the test-set CSV, the model's own mean before the book
  blend and before any correction.
- **Plain model**: a LightGBM squared-error regression on the stat itself, with fixed settings
  (learning rate 0.03, 15 leaves, at least 60 rows per leaf, half the columns per tree, early
  stopping). It trains on the oldest 59.5 percent of rows and stops on the next 10.5 percent,
  so it sees fewer rows than the served model. Its inputs are the served model's inputs: in
  all eight cells checked, every input value in the matrix equals its value in the test-set
  CSV.
- **MSE change**: percent change in mean squared error on held-out rows. Negative is better.
  Brackets hold a 95 percent interval from resampling whole game dates 500 times. "ns" means
  the interval includes zero.
- **Priced rows**: held-out rows whose line an app or a sportsbook posted (`QuoteAuthenticity`
  `pickem` or `authentic`), scored at that line. The rest of the held-out rows, about half,
  carry a stand-in line the pipeline made up (`synthetic` or `derived`), and nothing prices
  them.
- **Posted rungs**: every line Underdog or Sleeper still showed for a held-out row at the
  prop's last poll, read from the archive ladder as the realized replay reads it
  ([ship_gate.md](ship_gate.md), "The realized replay"). The ladder starts on 08-30 in MLB and
  WNBA, 09-09 in NFL, 09-19 in NHL and 10-03 in NBA, so posted rungs cover only the newest
  weeks of each held-out period. Props posted under a label the stat map lacks (§14 item 14)
  count under the market they name.
- **Brier at the line**: each mean becomes P(over) through one logistic map on its distance
  from the line, raw and divided by √(line + 1), fitted on four fifths of the game dates and
  applied to the fifth. Pushes are dropped. Changes are in units of 10⁻⁴; negative is better.
- **Verdict**: a candidate is kept or killed on priced rows and posted rungs, by MSE and Brier
  at the line ([CLAUDE.md](../CLAUDE.md), "Judging model changes"). The all-row figure is a
  diagnostic.
- **Quoted rows**: priced rows with a sportsbook quote (`authentic`), the rows where a book
  mean exists. §6 uses "quoted" for the matrix's own `Archived` flag, which marks every
  priced row.
- **Book mean**: `Book_EV`, the mean decoded from the book's line and price. The decode is
  poor in count cells: its error runs 6 to 23 percent above the model's in NBA REB, NBA FG3M,
  NHL shots and blocked, MLB hits and WNBA AST. I trust book comparisons only in NBA PRA and
  PR and in the NFL yardage, reception and carry cells.
- **Distance to book**: the average absolute difference between a model mean and the book
  mean on quoted rows. It is information, not a verdict: an accurate model must first
  represent the line, so moving toward the book or away from it is neither a gain nor a loss.

These are screens, not confirm walks. Each uses one date split, generic settings and no
retrain of a served model. §12 lists the limits.

## 3. The guide against the pipeline

The guide closes with an ordered workflow. This table sets each step beside what the
pipeline does today.

| Guide step | Pipeline today | Reading |
|---|---|---|
| 1. Split to match deployment | Date-ordered 70/30; the newest 30 % is halved by a hash of player and date into validation and test. The final fit uses the oldest 70 % only ([training/pipeline.py](../src/sportstradamus/training/pipeline.py), `_TRAIN_FRACTION`). Hyperparameter search scores four contiguous date blocks, each model trained on the other three. | The split is honest. Its cost is staleness (§5). Three of four search folds train on rows newer than the rows they score; not measured here. |
| 2. Leakage audit, adversarial validation | No post-outcome column among the inputs. One post-outcome value picks rows (§6.1). | A classifier separates the train period from the held-out period with AUC 1.00 in NBA and NFL: team and opponent profile columns act as period fingerprints. One input is an era marker by construction (§6.2). |
| 3. Build a baseline | No plain baseline is kept beside the distributional model. | The baseline wins on the mean in NBA and NFL, on priced rows and at posted lines (§4). |
| 4. Handle the target; match the loss | Per-cell target normalization (`ratio_meanyr`, `centered_additive_*`) chosen by the strategy sweep on probability gates. | The multiplier form costs mean accuracy where I could isolate it (§4.3). |
| 5. Features trees cannot synthesize | Rolling medians and means of the target; 300-day, last-5 and slope profiles of per-game rates; team and opponent profiles; comps; expanding means; two products. | Candidate ratios, lags and group aggregates of stored columns add at most 0.7 % on priced rows (§7.1). Play-level data that collection discards adds up to 4.2 % in a few cells (§7.3). NBA.com's play-level endpoints help assists on priced rows and minutes, which no app posts (§7.4). |
| 6. Constraints | Monotone on `MeanYr` only. | Not tested here; Part B dropped this screen too. |
| 7. Prune | Part B pruned two NFL cells. | Not repeated. |
| 8. Learning curve, residual slices | Not routine. | Curves have flattened at current sizes (§5). Slices by tier, month and lineup spot are in §4.2 and §7.2. |
| 9. Decide rows versus information | — | Flattened curves call for new information, not more rows. The large new information found either arrives too late for any posted line (§7.2) or is downloaded and then discarded (§7.3); the new NBA sources hold little outside assists (§7.4). |
| 10. Drift | One row window per league, no recency weights. | All history beats a recent window; recency weights add nothing beyond a fit through the newest rows; those rows are worth about 1 % on the mean and up to 10 × 10⁻⁴ of Brier at posted lines (§5). |
| Leave missing values as NaN | Blanket `fillna(0)` at the end of `get_stats`. | Part B tested NaN placeholders and dropped them. Not repeated. |

Part B (11-13 to 11-16) already tried the following, and this study did not re-test them:
comps extras, schedule features, NaN placeholders, NBA rates per minute and team share as
profile columns, NFL and NHL availability, NFL weather, per-cell pruning, and dropping team
or defense blocks. It found in-sample projection columns leak and walk-forward projections
add little. Its screen judged changes partly on all rows and on what they add beyond the
book, so its verdicts may not hold under §2 (§15).

## 4. Finding: the served mean loses to a plain regression on the same inputs

### 4.1 The result

MSE change of the plain model against the served mean, all held-out rows of the test half
(about 2,900 rows per NBA cell, 1,500 to 3,000 per NFL cell, 8,000 to 10,000 in NHL shots and
MLB hits).

| Cell | Family, target normalization | Plain vs served |
|---|---|---|
| NBA PRA | SkewNormal, `centered_additive_eb_meanyr_k10` | −3.3 % [−4.9, −2.0] |
| NBA AST | SkewNormal, `centered_additive_mean10` | −3.2 % [−5.1, −1.4] |
| NBA PR | SkewNormal, `centered_additive_mean10` | −1.5 % [−2.5, −0.4] |
| NBA fantasy | SkewNormal, `centered_additive_mean10` | −2.0 % [−3.4, −0.5] |
| NBA MIN | SkewNormal, `ratio_meanyr` | −4.2 % [−6.9, −1.9] |
| NBA REB | NegBin, none | −2.0 % [−3.4, −0.3] |
| NBA FG3M | DPO, none | −2.1 % [−3.2, −0.9] |
| NFL receptions | SkewNormal, `ratio_meanyr` | −7.0 % [−8.7, −5.4] |
| NFL rushing yards | SkewNormal, `ratio_meanyr` | −6.0 % [−9.3, −2.9] |
| NFL receiving yards | SkewNormal, `ratio_meanyr` | −2.4 % [−4.0, −1.0] |
| NFL carries | DPO, none | −5.2 % [−8.7, −2.1] |
| NFL passing yards | SkewNormal, `centered_additive_eb_meanyr_k10` | −5.9 % [−10.9, −1.5] (379 of 438 rows joined) |
| NFL fantasy (Underdog) | SkewNormal, `centered_additive_eb_meanyr_k10` | −0.5 % (ns) |
| NHL shots, blocked, points, skater fantasy | DPO or SkewNormal | −0.3, +0.4, −0.3, −0.8 % (all ns) |
| MLB hits, total bases | DPO, NegBin | 0.0, −0.2 % (ns) |

NBA PTS is left out: only 1,163 of its 2,677 test rows joined, so its test set is stale.

The plain model in this table averages three fits. As a single fit it leads by nearly the
same margin: five single fits give −2.9 to −3.5 % in NBA PRA and −6.6 to −7.0 % in NFL
receptions.

### 4.2 It is the ranking, not the level

A level error would vanish under recalibration. This one does not. After a cross-fitted
straight-line recalibration of both means, the plain model still leads: NBA PRA −3.4 %
[−4.8, −2.0], NBA MIN −4.5 % [−6.6, −2.6], NBA AST −2.2 % [−4.2, −0.6], NFL receptions −4.6 %
[−6.8, −2.6], NFL rushing yards −6.2 % [−8.9, −3.4], NFL receiving yards −2.3 % [−3.9, −0.7].

The gap holds in every month of the held-out period in five of the six cells sliced; NFL
receiving yards reverses in two of its eight months. It is widest for low-volume players in
NFL receptions (−14.6 % in the bottom third by trailing mean) and for high-volume players in
NFL rushing yards (−8.1 % in the top third).

### 4.3 Where it comes from

**Not the hyperparameters.** The served settings are already tight (8 to 16 leaves, 54 to 146
rows per leaf). A plain squared-error model run with each cell's own served settings keeps
the lead:

| Cell | Generic settings | Served settings, early stop | Served settings and served round count |
|---|---|---|---|
| NBA PRA | −3.3 % | −3.1 % [−4.5, −1.8] | −1.1 % (ns) |
| NBA MIN | −4.2 % | −3.9 % [−6.5, −1.8] | −4.5 % [−7.7, −1.9] |
| NFL receptions | −7.0 % | −7.3 % [−9.1, −5.5] | −5.7 % [−7.7, −3.9] |
| NFL carries | −5.2 % | −4.9 % [−9.1, −0.5] | −5.1 % [−8.5, −1.5] |
| NHL shots | −0.3 % (ns) | −0.1 % (ns) | −0.3 % (ns) |

Two causes show through.

*The multiplier form.* `ratio_meanyr` asks the model for a multiple of the player's 300-day
mean. I fit a plain model to that multiple with weights that make its loss identical to
squared error on the stat. In NBA MIN and NFL receiving yards that model lands on the served
model's error (+0.3 % and +0.9 % against served), while the plain model on the stat itself is
−4.2 % and −2.4 %. Same inputs, same loss, same learner: the form of the target alone costs
the whole gap in those two cells. In NFL rushing yards it costs a third of the gap, and in
NFL receptions a sixth. The likely reason is that the 300-day mean is a poor anchor when a
role has changed, and a multiple of a poor anchor is a noisy target.

*One round count for every parameter.* The NBA PRA model stops at 258 rounds. A squared-error
model with the same settings wants 600. The distributional fit stops when its likelihood
stops improving, and the shape parameters may reach that point before the mean has finished
learning. That is a reading, not a proof, but holding the plain model to 258 rounds erases
two thirds of its lead in NBA PRA.

The gap appears only where rows are thin against columns: NBA and NFL cells train on 14 to
48 rows per input column, NHL shots and MLB hits on 280 to 400. The guide warns that wide,
short tables erode a tree model's advantage, and a fit that spreads each row across two or
three parameters feels the shortage first.

### 4.4 What a better mean buys on the rows we price

The plain model against the served mean on the test half's priced rows and posted rungs
(§2). Brier changes are in units of 10⁻⁴.

| Cell | Priced rows: MSE | Brier at the priced line | Brier at the posted rungs |
|---|---|---|---|
| NBA PRA | −3.1 % [−5.0, −1.1] | −1.8 (ns) | −27.7 [−44.4, −10.3] |
| NBA REB | −0.4 % (ns) | −1.7 (ns) | −21.2 [−39.5, −4.5] |
| NBA FG3M | −1.3 % (ns) | −1.5 (ns) | −16.1 [−28.8, −3.3] |
| NBA AST | −1.1 % (ns) | −0.7 (ns) | −21.6 (ns) |
| NBA PR | −1.2 % (ns) | +0.4 (ns) | −5.6 (ns) |
| NBA fantasy | +0.8 % (ns), 327 rows | −9.3 (ns) | −14.7 (ns) |
| NFL receptions | −5.9 % [−7.7, −3.9] | −6.7 (ns) | −30.2 [−54.1, −1.6] |
| NFL rushing yards | −7.0 % [−10.4, −3.4] | +2.3 (ns) | −34.6 (ns) |
| NFL receiving yards | −2.4 % [−4.1, −0.5] | +1.4 (ns) | −21.3 (ns) |
| NFL carries | −5.6 % [−9.6, −1.6] | −7.6 (ns) | −62.3 (ns) |
| NFL fantasy (Underdog) | −2.1 % [−4.3, −0.5] | −7.6 (ns) | −1.6 (ns) |
| NFL passing yards | −4.6 % (ns), 368 rows | +3.9 (ns) | too few rungs |
| NHL shots, blocked, points, skater fantasy | −0.3, +0.4, −0.4, −2.1 % (all ns) | all ns | all ns |
| MLB hits, total bases | 0.0, −0.2 % (ns) | ns | ns |

NBA MIN has no priced row: no app posts NBA minutes.

The gain carries to the rows we price. Squared error falls in NBA PRA and five NFL cells by
about what it falls on all rows, and rises in none. At the posted rungs the Brier score falls
in four cells; 14 of 18 estimates sit below zero, and none above zero clears noise. The
priced rows' own lines separate the two means less. They sit near each player's median,
where the probability stays near one half whatever the mean, and there every Brier interval
spans zero. The posted rungs add the apps' alternate lines, where an accurate mean matters
most. Offers with no sportsbook quote rest on the model's distribution alone; the `pickem`
class is 12 percent of NBA PRA rows and sits inside every number above.

The plain mean also sits closer to the book. That is information, not a verdict (§2). On
quoted rows:

| Cell | Served mean | Plain mean | Change |
|---|---|---|---|
| NFL receptions | 0.654 | 0.522 | −20 % |
| NFL rushing yards | 8.80 | 7.48 | −15 % |
| NFL carries | 2.18 | 2.03 | −7 % |
| NFL receiving yards | 8.13 | 7.59 | −7 % |
| NBA PRA | 2.13 | 2.03 | −5 % |
| NBA REB | 1.56 | 1.49 | −4 % |
| NBA PR | 1.81 | 1.75 | −3 % |

In a cross-fitted linear blend with the book mean, the plain mean earns a larger weight than
the served mean in all eight cells tested (for example 0.16 against 0.05 in NFL receptions
and 0.21 against 0.10 in NFL rushing yards). Inside that blend the two means end level on
quoted rows (−0.6 % to 0.0 %, every interval spanning zero), so how much of the gain survives
the served blend is the replay's question (§11.6). Against the served standalone probability
itself, recalibrated, the plain mean wins on quoted NFL receptions rows by 20 [7, 34].

Two side facts from the same runs. The raw standalone model probability scores worse than a
coin flip on quoted rows in all seven cells (Brier 0.260 to 0.287 against 0.250), which the
blend then repairs. And the model's own inputs predict a quarter to a third of its gap to
the book on held-out rows (23 to 33 percent in the cells where the book mean can be
trusted), with outcomes following the predicted gap one for one in NBA PRA, NBA PTS and four
NFL cells: the matrix already carries much of the information in the line.

### 4.5 What this suggests

The fix lies in target normalization and the fit, which sit outside the feature grant and
under the research-first rule. The decision is the owner's. Two steps follow from the
evidence, the first of them safe.

- **Keep the baseline.** Fit the plain model beside each cell at training time and write its
  held-out MSE ratio to `model_stats`. It costs seconds per cell. It turns this finding into a
  per-cell number on every retrain and shows at once whether a new strategy gives up mean
  accuracy. This is step 3 of the guide's workflow.
- **Give the normalization a learned anchor.** The existing normalizations anchor on a
  rolling statistic (`MeanYr`, `Mean10`, the shrunk expanding mean). A walk-forward plain
  mean is a better anchor by the margins above, and it fits the mechanism the sweep already
  has: one more anchor column for `centered_additive_*`. A brief should precede it, because
  it changes what the distributional model is asked to learn. Expect 1.5 to 7 percent on the
  mean, about the same on priced rows, and a lower Brier score at the posted rungs; the
  priced rows' own near-median lines may not show it (§4.4).

## 5. Finding: every final fit stops 30 percent of the rows before today

The final fit uses the oldest 70 percent of the matrix and is never refit. On the current
matrices that fit ends on these dates:

| Cell | Fit ends | Gap to today |
|---|---|---|
| NFL rushing yards | 2024-12-29 | 23 months, two seasons |
| WNBA AST | 2025-08-30 | 14 months |
| NFL receptions, receiving yards | 2025-09-07 | 14 months |
| NHL shots | 2026-01-06 | 10 months |
| NBA PRA, AST | 2026-04-05 | 7 months, last season |
| MLB hits | 2026-05-05 | 6 months |

The NBA train part runs from December to early April. It holds no early-season game and no
playoff game, and it serves in November.

I scored the newest 15 percent of rows with the plain model, changing only which older rows
it trained on.

| Cell | Same row count, shifted to end at the 85 % mark | All rows to the 85 % mark | Newest 35 % only | Recency weights |
|---|---|---|---|---|
| NBA PRA | −1.2 % [−2.1, −0.2] | −1.3 % [−2.3, −0.5] | −0.2 % | −0.7 % (ns) |
| NBA AST | −0.9 % (ns) | −0.6 % (ns) | +1.0 % | −0.9 % (ns) |
| NFL receptions | −1.3 % [−2.3, −0.4] | −1.4 % [−2.3, −0.4] | −1.0 % | −1.5 % [−2.9, −0.1] |
| NFL receiving yards | −0.8 % [−1.8, 0.0] | −1.1 % [−2.0, −0.3] | +0.4 % | −1.0 % (ns) |
| NFL rushing yards | −0.6 % (ns) | +0.1 % (ns) | +1.0 % | +0.4 % (ns) |
| WNBA AST | +0.2 % (ns) | +0.5 % (ns) | +1.3 % | +0.8 % (ns) |
| NHL shots | −0.1 % (ns) | 0.0 % (ns) | +0.4 % | +0.1 % (ns) |
| MLB hits | −0.1 % (ns) | −0.2 % [−0.4, 0.0] | +0.1 % | −0.2 % (ns) |

The same arms on the rows we price (§2), with Brier in units of 10⁻⁴:

| Cell | Same row count, shifted: priced MSE | Brier at posted rungs | All rows to the 85 % mark: priced MSE | Brier at posted rungs |
|---|---|---|---|---|
| NBA PRA | −2.1 % [−4.2, −0.1] | −3.6 (ns) | −1.6 % (ns) | −5.1 (ns) |
| NBA AST | −1.2 % (ns) | −10.0 [−19.6, −0.2] | −0.9 % (ns) | −5.7 (ns) |
| NFL receptions | −1.1 % [−2.1, −0.0] | −6.5 (ns) | −1.1 % [−2.1, −0.2] | −3.6 (ns) |
| NFL receiving yards | −0.7 % (ns) | −8.3 [−15.8, −2.6] | −1.0 % [−1.9, −0.2] | −9.7 [−15.1, −4.6] |
| NFL rushing yards | −0.5 % (ns) | −7.3 (ns) | +0.1 % (ns) | +0.3 (ns) |
| WNBA AST | +1.0 % (ns) | +5.0 (ns) | +1.1 % (ns) | +11.0 (ns) |
| NHL shots | −0.1 % (ns) | −1.7 (ns) | −0.1 % (ns) | −2.1 (ns) |
| MLB hits | −0.1 % (ns) | −3.4 (ns) | −0.2 % (ns) | −4.4 [−7.2, −0.8] |

In NBA the priced rows of the newest 15 percent span 20 game dates, so those intervals are
wide. Brier at the priced rows' own lines clears noise for no arm.

Three readings.

- **Freshness is worth about 1 percent in three of eight cells and nothing in the rest.** That
  agrees with the live-gap lane's E1 verdict on blended Brier (no wins and no losses in five
  cells). On priced rows it holds in NBA PRA and NFL receptions and receiving yards. At the
  posted rungs it cuts the Brier score in NFL receiving yards by 8 to 10 in both fresher
  arms, and by 13 with recency weights, which read slightly worse at the priced rows' own
  lines (+1.1 [0.0, +2.5]); in NBA AST and MLB hits one arm each clears noise (−10, −4.4).
  Only the newest 35 percent alone costs anything: WNBA AST, by 2.4 % on priced rows and 33
  at the posted rungs.
- **It also shortens the distance to the book in NFL**, as information (§2). On quoted rows
  of the newest 15 percent, the fresher fit moves from 0.517 to 0.464 in receptions (−10 %),
  7.27 to 6.69 in receiving yards (−8 %) and 7.16 to 6.82 in rushing yards (−5 %). Against the
  served mean on the same rows (0.635, 7.80, 8.31), the fresh plain mean is 27, 14 and 18
  percent closer.
- **More rows would add little.** The curve was still falling at half the current size: at
  fixed recency, doubling the rows from the newest 35 percent to the newest 70 percent bought
  0.3 to 1.9 % in the NBA, NFL and WNBA cells and 0.1 to 0.6 % in NHL and MLB. It has
  flattened since: the next 21 percent more rows, reaching back to the oldest, bought between
  −0.7 and +0.3 %, zero on average. Another doubling would likely buy under 1 %. A longer NBA
  window is still worth one test, less for the rows than for the early-season and playoff
  games the train part lacks. By the guide's rule, the larger limit is information, not rows.

Refitting through the newest rows changes the split design, because the calibration layers
and gates are measured on the 70 percent fit. It is the owner's call: small on the mean,
larger at the posted line in NFL receiving yards.

## 6. Finding: how training rows are chosen

### 6.1 Unquoted NBA rows are picked by the minutes the player went on to play

`get_training_matrix` keeps a row when it is quoted, or when the player's usage **in that
game** clears the 15th percentile of the game log
([stats/base.py](../src/sportstradamus/stats/base.py), the `usage > usage_cutoff` test). Usage is
minutes in NBA and WNBA, snap share in NFL and ice-time share in NHL. The value is known only
after the game.

| League | Cutoff in practice | Unquoted share of rows | Effect |
|---|---|---|---|
| NBA PRA | more than 11 minutes | 53 % | large for low-minute players |
| NHL shots | ice-time share above 0.19 | 17 % | about 1 % on the mean outcome |
| NFL receptions, rushing yards | any snap | 60 %, 49 % | none |

In NBA the filter removes the short-minute games (early injury exits, blowouts, benchings)
from unquoted rows and leaves them in quoted rows. Applying the same cut to quoted rows shows
its size:

| Trailing-mean quartile of quoted NBA PRA rows | Games at or under 11 minutes | Mean outcome rises by |
|---|---|---|
| lowest | 7.7 % | 5.9 % |
| second | 2.8 % | 2.0 % |
| third | 1.2 % | 0.9 % |
| highest | 0.5 % | 0.4 % |

Unquoted rows carry the mark: their outcomes run 1.27 PRA above the player's own trailing
mean, against 0.22 for quoted rows.

I could not show harm to the served mean. Its bias on quoted held-out rows is small and of
mixed sign across five NBA cells (−2.4 % to +6.1 %). The harm to expect is in the lower tail:
on half its rows the model never sees the short-minute games that every live offer is exposed
to. Those games are 3.8 percent of quoted held-out rows and carry 5.6 percent of the absolute
error.

The clean form of the filter tests the minutes known before the game. The depth-chart code
already computes that value (`usage short`). The change falls under training rows, inside the
feature grant, and needs an NBA and WNBA matrix rebuild.

### 6.2 The quoted share shifts by era, and one new input marks the era

The archive holds no NFL quotes before 2024. NFL receptions rows are 0 percent quoted from
2021 to 2023 and 79 to 86 percent quoted since. The train part is 20 percent quoted and the
held-out part 86 percent. The model learns mostly from unquoted players and serves quoted
ones.

`weather lead` is the forecast lead in days. It is 0 on every NFL row before 2024, where the
weather is observed history, and 1 on about three quarters of rows since. It carries 15 to
24 percent of the split gain in a classifier that tells the train period from the held-out
period. In NFL it is a near-perfect marker of the quoted era, and nothing about the weather.

### 6.3 Re-weighting rows does not help

I tested four ways to pull training toward the serving population: tripling the weight of
quoted rows, weighting unquoted rows by their resemblance to quoted ones, weighting rows by
their resemblance to the held-out period, and training on quoted rows alone. Scored on
priced rows and posted rungs in five cells (NBA PRA, NFL receptions, receiving and rushing
yards, NHL shots), no arm gains: of 76 intervals none clears zero on the better side and 23
clear it on the worse. Quoted-only training costs NBA PRA 3.1 % [1.7, 4.6] on priced rows and
19 × 10⁻⁴ [11, 26] of Brier at the posted rungs. Unquoted rows earn their place.

## 7. Finding: data that never reaches the model

### 7.1 Stored columns, re-expressed

Each test adds columns to the plain model and scores the newest 30 percent. Every column
uses earlier games only. Brier changes are in units of 10⁻⁴ (§2).

| Cell | Added columns | All held-out rows | Priced rows: MSE | Brier at the priced line | Brier at the posted rungs |
|---|---|---|---|---|---|
| NHL shots | shot attempts, expected goals, attempts per minute, share of attempts on net | −0.41 % [−0.57, −0.24] | −0.41 % [−0.60, −0.23] | −5.5 [−8.2, −2.5] | −5.9 [−9.7, −2.0] |
| NHL shots | all 52 columns from stored but unused fields | −0.49 % [−0.70, −0.30] | −0.56 % [−0.79, −0.34] | −7.1 [−10.7, −3.2] | −7.8 [−12.1, −3.5] |
| NHL shots | ice-time history | −0.10 % (ns) | −0.13 % (ns) | −1.4 (ns) | −3.1 (ns) |
| NHL shots | zone starts | −0.05 % (ns) | 0.00 % (ns) | +0.6 (ns) | −0.7 (ns) |
| NHL blocked | all 52 | −0.30 % [−0.55, −0.04] | +0.10 % (ns) | +2.0 (ns) | +1.7 (ns) |
| NBA PRA | minutes history: last game, 3-, 5- and 10-game means, spread, trend | −0.39 % [−0.73, −0.02] | −0.57 % [−0.98, −0.11] | −0.3 (ns) | −2.5 (ns) |
| NBA PRA | per-minute rates as ratios of sums, three half-lives | −0.22 % (ns) | −0.14 % (ns) | −0.2 (ns) | −0.8 (ns) |
| NBA PRA | per-minute rate adjusted for opponents faced | −0.03 % (ns) | +0.03 % (ns) | +0.9 (ns) | −1.2 (ns) |
| NBA AST | all 51 candidates | −0.28 % (ns) | −0.62 % (ns) | −4.6 (ns) | −0.3 (ns) |
| NBA AST | per-minute rates as ratios of sums | −0.33 % (ns) | −0.73 % [−1.21, −0.30] | −4.6 [−7.5, −1.6] | 0.0 (ns) |
| NBA AST | per-minute rate adjusted for opponents faced | +0.14 % (ns) | +0.05 % (ns) | −2.7 [−5.2, −0.2] | 0.0 (ns) |
| MLB hits | batter side, pitcher hand, platoon advantage | −0.13 % [−0.18, −0.07] | −0.13 % [−0.19, −0.07] | −1.1 (ns) | −1.2 (ns) |
| MLB hits | those, plus the batter's and the opposing pitcher's split history | −0.12 % [−0.19, −0.04] | −0.12 % [−0.19, −0.04] | −1.8 [−3.1, −0.4] | −2.5 [−4.0, −0.8] |
| MLB hits | lineup-spot and start history | −0.13 % [−0.21, −0.04] | −0.13 % [−0.21, −0.05] | −1.6 (ns) | −1.9 (ns) |
| MLB total bases | platoon advantage alone | −0.09 % [−0.19, −0.01] | −0.09 % [−0.19, −0.01] | −1.7 [−3.2, −0.1] | +1.0 (ns) |
| MLB total bases | lineup-spot, start and plate-appearance history | −0.15 % [−0.26, −0.03] | −0.15 % [−0.26, −0.03] | −1.6 (ns) | −1.5 (ns) |

Every held-out row in these MLB cells is priced. Of the table's 48 intervals on priced rows
and posted rungs, 18 clear zero on the better side and none on the worse; chance alone would
put about one on each.

Shot attempts and expected goals sit in the NHL game log and never enter the profile. The
MLB game log stores batter side and pitcher hand, and no input uses them. Each adds a real but
small amount to the mean, and on the rows we price some of it lands at the line: NHL shot
attempts cut the Brier score by 5 to 8 at the priced lines and the posted rungs alike, and
MLB handedness with split history by about 2 on hits. The game logs hold little untapped
signal about the mean, which agrees with the learning curves.

Two expected wins appear in one cell only. Per-minute rates built as ratios of sums, which fix
the profile's habit of averaging per-game ratios with zero-attempt games counted as zero, add
nothing in NBA PRA but cut priced-row error in NBA AST by 0.73 percent and the Brier score at
the line by 4.6. Adjusting past production for the opponents faced moves AST's Brier by 2.7
and nothing else. Neither gain shows yet at AST's posted rungs, which span 26 game dates.

### 7.2 The MLB lineup of the day

The MLB hitter models take no lineup or plate-appearance input. The matrix carries
`Player proj plateAppearances mean`, built from the lineup spot, but it is not among the
inputs: on settled games it reads the realized spot, the kind of leak Part B found in
projection columns. `Player depth` is zero on every row.

What the realized lineup is worth, as an upper bound:

| Added to the plain model | MLB hits | MLB total bases |
|---|---|---|
| Whether the player started today | −2.64 % [−2.96, −2.32] | −1.59 % [−1.81, −1.36] |
| Started, plus lineup spot | −2.89 % [−3.19, −2.55] | −1.78 % [−2.03, −1.54] |
| Lineup spot, among starters only | −0.15 % [−0.25, −0.06] | −0.23 % [−0.35, −0.13] |

Nearly all of it is the start itself. About 6 percent of quoted held-out rows are hitters who
did not start. They average 0.20 hits against a served mean of 0.77, and 0.34 total bases
against 1.27. Among starters the served mean runs 10 percent low for leadoff hitters in hits
[−15.3, −4.7], 9 percent low in total bases [−17.3, −2.2] and 8 percent low in
hits+runs+RBI [−14.4, −1.9].

Every held-out row in both cells is priced, and the start shows at the line too: it cuts the
Brier score by 75 × 10⁻⁴ [65, 85] at the priced lines in hits and by 48 [41, 54] in total
bases, and by 28 and 14 at the posted rungs.

The lineup posts a few hours before first pitch, and the apps act on it. In the archive
ladder from 08-30 to 10-31, a starter's batter props stay up until the game's last poll
(median 2 minutes before it; 93 percent within 30 minutes). A non-starter's props come down a
median 186 minutes before it, and 1 percent last within 30 minutes: the apps pull them when
the lineup posts. Non-starters are 5.2 percent of posted batter props, and serving prices
them only while no one knows they sit. The start is therefore knowledge the board never
prices: its gain is an upper bound no posted line offers, and a rule not to price a hitter
missing from a posted lineup would rarely fire.

What remains is the spot among starters, known for the three hours their props stay up and
worth 0.15 to 0.23 percent. Serving already reads a posted lineup when one exists and falls
back to the player's usual spot. The game log cannot say which of the two serving had at
scoring time, so a model trained on realized spots learns from knowledge it will often lack.
The way out is to record what serving knew. The feature log exists for this. If it also
records where the spot came from (posted lineup or fallback), a season of logs gives rows
with the true serve-time state, and the spot can become an input without leaking. MLB is out
of season, so the logging change costs nothing now and pays in April.

### 7.3 Play-level data: fetched every game, mostly thrown away

The game-log update downloads a play-level file for every MLB and NHL game and keeps a box
score from it.

| League | What the update downloads | What it keeps |
|---|---|---|
| NFL | nflverse play-by-play and Next Gen Stats | Most of it: usage columns in the game log, EPA and pressure aggregates in the comp profiles ([stats/nfl_pbp_agg.py](../src/sportstradamus/stats/nfl_pbp_agg.py)) |
| MLB | Savant's game feed, `gf?game_pk=` ([stats/mlb.py](../src/sportstradamus/stats/mlb.py), `parse_game`): every pitch with 98 fields, among them speed, spin, location, the call and times through the order, and every batted ball's exit velocity, launch angle and expected outcome | Box-score counts |
| NHL | moneypuck's per-game file ([stats/nhl.py](../src/sportstradamus/stats/nhl.py), `parse_game`): 746 columns, one row per player per situation (all, 5-on-5, 5-on-4, 4-on-5, other) plus rows for lines and pairings | The `all` row, and points from the 5-on-4 row (`pp_df`) |
| NBA | No play-level data (§7.4) | — |

To price what MLB and NHL throw away, I rebuilt the discarded fields for past games from
public history: Baseball Savant's search export for MLB (612 game dates, 2.18 million
pitches) and the NHL stats service's per-game reports for NHL (162,842 player-games since
October 2023). On the game checked, the reports' power-play time, short-handed time and
faceoffs taken equal the moneypuck fields for every skater. Savant's pitch counts equal the
game log's on 249 of 270 sampled starts and run at most four pitches high on the rest. Each
candidate is a recency-weighted state per player or team, built from earlier dates only and
joined by player id; opponents come from the schedule. The screen is §7.1's.

| Candidate set | Cell | All held-out rows | Priced rows: MSE | Brier at the priced line | Brier at the posted rungs |
|---|---|---|---|---|---|
| MLB starter's leash: pitch counts, batters faced and times through the order in recent starts; fastball velocity | pitches thrown | −4.21 % [−5.53, −2.80] | — | — | — |
| | pitching outs | −1.01 % [−1.68, −0.41] | −1.01 % [−1.60, −0.47] | +0.9 (ns) | +3.2 (ns) |
| MLB pitcher's own pitch-level record: the leash plus strikeout, walk, whiff, chase and contact rates | pitcher strikeouts | −1.69 % [−2.62, −0.76] | −1.69 % [−2.57, −0.74] | −19.6 [−30.9, −7.8] | +1.8 (ns) |
| | hits allowed | −0.86 % [−1.57, −0.04] | −0.86 % [−1.54, −0.16] | −9.6 [−18.5, −1.7] | −21.7 (ns) |
| | runs allowed | −0.22 % [−0.41, −0.02] | −0.22 % [−0.40, −0.02] | −0.8 (ns) | −5.8 (ns) |
| The pitcher's record plus the opposing lineup's rates against the pitcher's hand | pitcher strikeouts | −2.70 % [−4.06, −1.39] | −2.70 % [−4.06, −1.43] | −35.6 [−52.8, −18.7] | −1.8 (ns) |
| | pitching outs | −1.48 % [−2.20, −0.76] | −1.48 % [−2.26, −0.77] | +0.1 (ns) | −2.0 (ns) |
| MLB batter's contact and discipline: barrel, hard-hit and sweet-spot rates, exit velocity, bat speed, chase, whiff, expected wOBA | home runs | −0.24 % [−0.42, −0.06] | −0.24 % [−0.44, −0.07] | −2.3 [−3.9, −0.6] | −1.7 (ns) |
| | runs | −0.23 % [−0.37, −0.11] | −0.23 % [−0.34, −0.12] | −1.5 (ns) | +1.6 (ns) |
| | hits+runs+RBI | −0.18 % [−0.32, −0.06] | −0.18 % [−0.31, −0.07] | −1.9 [−3.8, −0.1] | −1.9 (ns) |
| | walks | −0.17 % [−0.31, −0.03] | −0.17 % [−0.32, −0.03] | −3.2 [−5.9, −1.0] | −2.3 (ns) |
| | total bases | −0.15 % [−0.30, −0.01] | −0.15 % (ns) | +0.6 (ns) | +0.5 (ns) |
| | hits | −0.14 % [−0.23, −0.05] | −0.14 % [−0.23, −0.05] | −0.5 (ns) | −0.2 (ns) |
| NHL faceoffs taken: recent counts, rate per minute, zone shares | faceoff wins | −2.46 % [−3.49, −1.44] | — | — | +6.3 (ns), 480 Underdog rungs |
| NHL power-play role: power-play ice time and its share of the team's, power-play shots and points | power-play points | −0.69 % [−0.96, −0.41] | −0.59 % [−0.92, −0.24] | −3.7 [−8.1, −0.0] | −2.2 (ns) |
| | points | −0.21 % [−0.37, −0.06] | −0.20 % [−0.43, −0.00] | −4.3 [−8.3, −0.4] | −3.3 (ns) |
| | skater fantasy | −0.31 % [−0.67, +0.01] (ns) | +0.17 % (ns) | +0.8 (ns) | −0.8 (ns) |
| NHL rink factor: how each arena's scorer counts blocks and hits | blocked shots | −0.17 % [−0.29, −0.04] | −0.14 % (ns) | −0.6 (ns) | −1.9 (ns) |

A dash means no priced row or posted rung. Every held-out row in these MLB cells is priced,
so their priced MSE repeats the all-row figure. No faceoff-wins row in the matrix carries a
quote and the cell is withheld, but Underdog posts faceoff wins under a label the board does
not map (§14 item 14). On its 480 posted rungs since 09-19 the faceoff set reads −1.4 % (ns)
on squared error, and −2.4 % [−4.6, −0.3] with team faceoff rates added. Its 273 rows with no
earlier faceoff state average fewer wins than the rest, but the all-row gain holds at −2.60 %
without them, so the blank does not carry it. On pitching outs the full 30-column pitcher
record reads only −0.13 % (ns) while its leash subset reads −1.01 %, so treat the outs
figures as the least stable here.

On the rows we price the gains hold where books and apps post the market. Pitcher strikeouts
gains most: the pitcher's record with the opposing lineup cuts the Brier score at the priced
line by 36 × 10⁻⁴ [19, 53]. Power-play role lowers both error and Brier on points and
power-play points, and the batter's contact quality does the same on walks, home runs and
hits+runs+RBI. The posted rungs, which in MLB run only from 08-30 to the World Series, are
too few to settle most of these, and no interval there contradicts the priced rows. They
confirm squared error in pitching outs and batter walks and Brier in hits allowed and NHL
points, each for the widest set; pitcher strikeouts reads −1.8 there (ns). Pitches thrown has
no priced row or posted rung and pays through the projection that feeds every pitcher cell
(below). Across every MLB and NHL play-level set scored, 59 intervals on priced rows and
posted rungs clear zero on the better side and 9 on the worse, against about 8 each by
chance.

Measured and null or worse: team and opponent power-play and penalty-kill rates, which clear
zero in one of 16 intervals across four NHL cells; the opposing lineup for runs allowed
(+0.61 % [+0.37, +0.89] on priced rows and +12 × 10⁻⁴ [3, 21] of Brier at the posted rungs)
and for walks allowed (+2.75 % [+0.09, +5.94] at the posted rungs), whose matrix holds hitter
rows (§14 item 13); the batter's contact and discipline for batter strikeouts (+1.45 %
[+0.08, +2.59] at the posted rungs) and doubles; power-play role for goals, assists, shots
and skater fantasy; and the rink factor for hits and skater fantasy, which costs 5.7
[2.2, 8.9] and 10.3 [4.1, 17.1] × 10⁻⁴ of Brier at the priced line. The opposing starter's
record in batter cells is small and mixed: −0.12 % in runs and −0.08 % in singles on priced
rows, better at the posted rungs in walks and worse there in hitter fantasy (+3.3 × 10⁻⁴
[1.6, 4.8]). NHL time on ice read −3.56 % [−6.00, −1.34] with even-strength and short-handed
ice time, but 88 percent of that came from nine goalie rows in the held-out set (§14 item
11); on skaters it is about −0.4 %, and no app posts the market. Shot attempts in NHL shots
repeat §7.1.

**What each league would take.**

- **NHL: no new requests.** The 5-on-4 row's `I_F_iceTime` is power-play time, the 4-on-5
  row's is short-handed time, and `I_A_faceOffsWon` counts faceoffs lost, so faceoffs taken is
  its sum with `I_F_faceOffsWon`. `parse_game` already holds all three. History back to
  October 2023 comes from the NHL stats service: four reports took about six minutes.
- **MLB: no new live requests.** The game feed `parse_game` reads carries every pitch. The
  history pull used Savant's search export, one request per date: 612 dates took about 20
  minutes and 115 MB. The live feed carries no expected wOBA, so a build should use only fields
  both carry or rebuild such rates from exit velocity and launch angle. The leash belongs first
  in the pitches-thrown model, whose projection (`Player proj pitches thrown mean`) feeds every
  pitcher cell. MLB is out of season until April.
- **NBA: new requests, for little (§7.4).** The update fetches no play-level data, so every
  NBA candidate costs new calls, and only the minutes model gains.
- **NFL: little left.** The profiles already read nflverse play-by-play. The weak spot is
  routes: `route participation` is team pass plays times snap share, which correlates 0.994
  with snap share. Fantasy Points' data carries real routes and targets per route run.
  Opponent-adjusted EPA is untested.

Savant, NBA.com and the MLB stats API publish under personal-use terms, so any new collector
waits on the owner's licence call
([researcher_information_audit.md](archive/researcher_information_audit.md)).

### 7.4 NBA: play-by-play and tracking endpoints the pipeline does not call

The NBA update reads box scores and season-to-date tables. It calls none of the play-by-play,
rotation, tracking, matchup, hustle or shot-chart endpoints in the nba_api reference. I probed
them from the dev box through `nba_client`, the repo's throttled client
([stats/nba_client.py](../src/sportstradamus/stats/nba_client.py)).

| Endpoint | What it returns | From the dev box |
|---|---|---|
| `PlayByPlayV3` | every action with the clock, the score, the shooter, shot distance and location, and the assist credit | works, about one second a game |
| `GameRotation` | every stint: player, time in, time out | works |
| `BoxScorePlayerTrackV3` | touches, passes, secondary assists, rebound chances, distance run, contested shots | works |
| `BoxScoreMatchupsV3` | possessions and fouls by attacker and defender | works; potential assists read 0 on every row |
| `BoxScoreHustleV2` | box outs, deflections, contested shots, charges drawn, loose balls | works; not pulled |
| `LeagueDashPtStats` over a one-day range | potential assists, assist points created and drives, per player | works, one call per date and measure |
| `ShotChartDetail` for a season | every shot with its zone | stops at 102,400 rows (10-21 to 01-29); V3 carries the same shots |
| `PlayByPlayV2` | the older play-by-play | no data |
| `WinProbabilityPBP` | win probability by play | empty response |
| `cdn.nba.com` live play-by-play | V3's content, during the game | 403 Access Denied |

I pulled the play-by-play, rotations, tracking box and matchups for all 1,519 games from
2025-10-21 to 2026-11-15: 6,076 calls in 78 minutes at the client's throttle, three of which
failed after five tries. Potential assists and drives took 474 calls in six and a half
minutes. The NBA matrices start on 2025-12-09, so even the first matrix row has seven weeks of
games behind its states.

Each family below applies a technique from the guide
([gbdt_feature_engineering.md](gbdt_feature_engineering.md)) to data the inputs lack; "guide
§1" and the like name the guide's own sections. As in §7.3, every state is recency-weighted
with a half-life of ten games, built from earlier dates only and joined by player id; team
states join by the row's opponent or the player's own team. Rates are ratios of weighted sums
(§7.1). No family is blank on more than 0.1 percent of rows, so no blank can carry a gain
(§12). The screen is §7.1's, run on 14 NBA cells for each family alone and for all 48 columns
together.

| Family (columns) | Source | Guide technique | What the columns measure |
|---|---|---|---|
| Minutes structure (9) | `GameRotation`; the score from V3 | Rolling windows; group-by aggregates over game states (guide §1) | stints a game; start rate; first-stint length; share of minutes in the fourth quarter; share of the last two minutes played; share of minutes in garbage time (margin of 20 or more); minutes while the margin is within 8; minutes in games decided by 15 or more over minutes in games decided by under 10; rate of early foul trouble |
| Shot profile (10) | V3 shots | Ratios; a target encoding computed out of time (guide §1) | shares of attempts at the rim, in the short two, from mid-range, from the corner three and from above the break; expected points per shot from league make rates by zone on earlier dates; points and threes above expected; assisted share of makes; mean shot distance |
| Tracking box (11) | `BoxScorePlayerTrackV3` | Ratios that split opportunity from conversion (guide §1) | rebound chances per minute (all, offensive, defensive) and the share converted; touches, passes, and secondary and free-throw assists per minute; assists per pass; contested share of shots; distance run per minute |
| Passing and drives (7) | `LeagueDashPtStats` by date | The same split; information the inputs lack (guide §5) | potential assists per minute and the share converted; assist points created per minute; drives per minute; free throws, passes and points per drive |
| Matchups (2) | `BoxScoreMatchupsV3` | Ratios (guide §1) | possessions per matchup minute; shooting fouls drawn per minute |
| Opponent and team (9) | all of the above, by team | Group-by aggregates (guide §1) | what each opponent allows: rebound chances, offensive rebound chances, potential assists, drives, possessions, and the rim and three-point shares of attempts; the garbage-time share of the opponent's games and of the player's own team's |

The table holds the full set in each cell where one of its intervals clears zero, and each
family with a gain; the families' costs are in the readings below. A dash means no priced row
or posted rung: no app posted an NBA minutes prop.

| Candidate set | Cell | All held-out rows | Priced rows: MSE | Brier at the priced line | Brier at the posted rungs |
|---|---|---|---|---|---|
| All 48 columns | MIN | −0.70 % [−1.17, −0.20] | — | — | — |
| | AST | −0.42 % (ns) | −0.70 % [−1.36, −0.03] | −7.3 [−12.7, −2.1] | −2.2 (ns) |
| | PA | +0.05 % (ns) | −0.69 % [−1.12, −0.22] | −0.9 (ns) | +1.7 (ns) |
| | fantasy | +0.22 % (ns) | +0.75 % (ns) | −4.7 [−7.7, −1.1] | +2.9 (ns) |
| | PRA | +0.40 % [+0.02, +0.78] | +0.33 % (ns) | +0.3 (ns) | +4.9 [+0.5, +9.1] |
| | PR | +0.27 % (ns) | +0.15 % (ns) | +0.8 (ns) | +8.8 [+1.8, +14.6] |
| | REB | +0.75 % [+0.07, +1.42] | +1.04 % [+0.20, +1.90] | +0.9 (ns) | +2.3 (ns) |
| Minutes structure | MIN | −0.54 % [−0.96, −0.19] | — | — | — |
| | PA | −0.16 % (ns) | −0.61 % [−1.05, −0.18] | −1.5 (ns) | +1.1 (ns) |
| Tracking box | AST | −0.10 % (ns) | −0.33 % (ns) | −4.1 [−8.1, −0.5] | −1.1 (ns) |
| | PA | +0.01 % (ns) | −0.53 % [−0.86, −0.25] | −0.3 (ns) | +5.3 [+0.4, +12.7] |
| Passing and drives | AST | −0.39 % (ns) | −0.47 % (ns) | −4.3 [−8.8, −0.1] | +0.8 (ns) |
| | PA | −0.03 % (ns) | −0.43 % [−0.79, −0.02] | −1.3 (ns) | −0.7 (ns) |
| | STL | −0.13 % (ns) | −0.19 % (ns) | −3.4 (ns) | −3.7 [−6.5, −0.8] |
| Shot profile | PA | +0.10 % (ns) | −0.18 % (ns) | −1.6 [−2.9, −0.4] | +3.3 (ns) |
| Matchups | fantasy | −0.02 % (ns) | +0.25 % (ns) | −2.1 [−4.3, −0.0] | −0.8 (ns) |
| | PR | −0.29 % [−0.56, −0.03] | −0.11 % (ns) | +0.2 (ns) | +0.4 (ns) |

The 12 cells other than MIN with priced rows give 336 intervals: seven sets, each with squared
error and Brier on priced rows and at the posted rungs. Of these, 15 clear zero on the better
side and 26 on the worse; chance alone would put about 8 on each. Five readings.

- **Assists gain at the priced line.** All 48 columns cut AST's priced-row error by 0.70
  percent and its Brier score at the priced line by 7.3 × 10⁻⁴; passing and drives, led by
  potential assists, and the tracking box each give about 4 of that alone. At AST's posted
  rungs, 26 game dates since the ladder began, the full set reads −2.2 (ns). §7.1's
  ratio-of-sums assist rates gain the same way, so AST is the one NBA cell where both stored
  and new columns pay on the rows we price.
- **Points+assists gains on squared error, in a withheld cell.** Three families and the full
  set cut PA's priced-row error by 0.4 to 0.7 percent, and shot profile its Brier score at
  the priced line by 1.6. None helps at the posted rungs, and the tracking box costs 5.3 ×
  10⁻⁴ of Brier there. PA is withheld and has no current model; its test set dates from 10-18
  (§14 item 8), so its screen, like PTS's, took NBA PRA's input list. The gain matters only if
  PA trains again.
- **Minutes structure pays on a cell no app posts.** It cuts the error on NBA MIN by 0.54
  percent, and all 48 columns by 0.70 percent. The model splits most on minutes while the
  margin is within 8, the start rate, the fourth-quarter share and the closing minutes. Those
  separate a player's role from the game script, which §7.1's raw minutes history mixes in. No
  other NBA cell reads the minutes model: the matrices carry its projection
  (`Player proj MIN mean`) but not as an input, and no NBA cell uses `ratio_projvol`, the
  normalization that divides by it. Given straight to the scoring cells, the same columns help
  PA on priced rows, cost REB there, cost FG3M at the posted rungs and cost BLK on all rows.
- **The full set costs where it does not help.** All 48 columns cost PRA and PR at the posted
  rungs, by 4.9 and 8.8 × 10⁻⁴ of Brier and 0.77 and 0.61 percent of squared error, and cost
  REB 1.04 percent on priced rows; squared error at the posted rungs also rises in fantasy,
  BLK and BLST. Family by family, the costs fall on REB (four families, on priced rows or at
  the rungs), on FG3M and BLK at the posted rungs, and on PTS and PR at the priced line from
  shot profile and passing. That is the warning about noise columns in guide §3: boosted
  trees lose accuracy as uninformative predictors are added, and NBA cells already train on
  fewer than 50 rows per input column (§4.3).
- **The few other gains are isolated.** Passing and drives helps STL at the posted rungs,
  the tracking box lowers squared error there in PTS and shot profile in STL, and the full
  set and matchups help fantasy at the priced line, on 675 priced props. Outside AST the
  inputs already carry 300-day, last-five and slope profiles of per-game rates, team and
  opponent profiles, and comps (§3), and rebound chances, touches and drives describe the same
  production from another side. By the test in guide §5 a source helps only where it explains
  variance the inputs miss; on the rows we price, that shows in AST and perhaps PA.

**What a build would take.**

- **Assists, once more posted rungs accrue.** Passing and drives takes one `LeagueDashPtStats`
  call per date and measure (474 calls for the season, six and a half minutes); the tracking
  box takes one call a game. Pair them with §7.1's ratio-of-sums assist rates, and rescreen
  AST when its posted rungs span more than 26 dates.
- **Minutes structure, once the minutes model has a consumer.** Two calls a game
  (`GameRotation` and V3): about 20 on a ten-game night and 40 minutes of backfill for the
  current window. It pays only if the apps post NBA minutes props or a cell anchors on the
  minutes projection through `ratio_projvol`. Screen that cell first.
- **Dated tables for the comps need no play-by-play.** The comps read season-to-date
  shot-location, play-type and bio tables as last fetched, the likely cause of the drift in
  §14 item 6. `LeagueDashPlayerShotLocations` and `LeagueDashPlayerBioStats` take a date
  range, so a rebuild can read them as of each row's date. `SynergyPlayTypes` takes none, so
  only a stored daily capture can date it (§7.5). The fix is for consistency between training
  and serving; its effect on the mean is unmeasured.
- **Hustle box outs, untested.** Rebound chances, the nearer measure, added nothing to REB,
  OREB or DREB, so expect little. The residual pilot of guide §5, run on the pulled season,
  would settle it before any collector.

The other families cost or do nothing on the rows we price and are not worth a collector
(§9). NBA.com publishes under
the personal-use terms of §7.3, so any NBA collector waits on the same licence call.

### 7.5 Other discards

- **NBA starters.** The absences collector fetches the traditional box score for every game
  and keeps only the players who sat
  ([collectors/game_absences.py](../src/sportstradamus/collectors/game_absences.py)). The same
  response lists who started.
- **Snapshot sources have no past.** The Cleaning the Glass, Savant and Fantasy Points
  collectors store season-to-date tables stamped with the capture date, and the NBA comps read
  such tables with no stamp at all (§14 item 6). A row before the first capture can never
  carry them. Where a source offers game-level or date-ranged data, collecting that form lets
  any feature be rebuilt as of any date. Savant's search export (§7.3) and NBA.com's
  shot-location and bio tables (§7.4) are such forms.

§7.4 measured the start rate within minutes structure: it helps the minutes model and nothing
else. The starters cost no extra requests.

## 8. Serve-time inputs against rebuilt inputs

The feature log on the dev box holds what serving fed each model from 2026-10-05. I joined it
to the matrix rows for the same player and date. The parity tool already measures the effect
on probability, and the lane's NBA read called it small (RMS 0.009); this is the same
comparison in raw feature units. I did not check prod's copy of the log.

**The NBA `Home` flag is wrong at serve time.** Across 2,862 NBA PRA rows from 10-20 to
11-16, serving marked 17.5 percent of players as home. The game log says 50.2 percent. The
flag differs on 41 percent of rows, alike on Sleeper and Underdog and in NBA PR and NBA
fantasy. The error is all or nothing by team and game: of 394 team-games, 161 are wrong for
every player. Serving marked 145 of 196 home teams as away and 16 of 198 away teams as home.
Team, opponent and game lines are right on those rows, so only the flag is wrong. NHL shows
no mismatch.

The cause is the schedule source. `fetch_upcoming_games`
([stats/base.py](../src/sportstradamus/stats/base.py)) reads stats.nba.com's
`internationalbroadcasterschedule`, which lists only the games an international broadcaster
carries. It listed 2 of the 11 games played on 10-21, 2 of 10 on 11-11, 2 of 9 on 11-13 and
1 of 5 on 11-14. It listed all 3 on opening night, the one date with no mismatch. A team in
an unlisted game has no entry, and `_game_context` then writes False (the
`games[x].get("Home", False)` line). An unlisted team with a listed game on another day in
the three-day window inherits that game's flag, which is how away teams come to read as
home. The WNBA update path calls the same function. This closes the NBA home check that the
Sleeper fix left open.

**Some inputs barely survive the trip.** Correlation between the serve-time value and the
rebuilt value:

| Input | NBA PRA | NHL shots |
|---|---|---|
| `Player moneyline gain` | — | 0.02 |
| `Player totals gain` | 0.57 | 0.19 |
| `Defense moneyline gain` | 0.99 | 0.57 |
| `Defense totals gain` | — | 0.57 |
| `Player comps z` | 0.38 | 0.70 |
| `Defense comp distance` | 0.57 | 0.64 |
| `Player depth` | 0.58 | 0.82 |
| `Player age` | 0.55 | — |
| `GameTotal`, `Total`, `OppTotal`, `Spread`, `Moneyline` | 0.83–0.90 | 0.76–0.95 |
| `MeanYr`, `Mean10`, `GamesPlayed` | 1.00 | 1.00 |

A dash means the input is not among that cell's 22 largest differences.

The rolling statistics match. The game lines differ because training reads the game log's
stored line and serving reads the archive at scoring time, a known gap. The per-player
`gain` slopes in NHL and the comps columns in NBA come back nearly unrelated. An input that
changes this much between two builds of the same row cannot carry signal from training to
serving, and the small probability effect the parity tool reports suggests the model leans
on them little. They are candidates to stabilize or drop; I did not measure the cost. In NBA
the gap narrows toward the rebuild date, which points at the season snapshots the comps read
(§14 item 6).

## 9. Tested here and not worth pursuing

Each of these costs or does nothing on the rows we price (§2).

- **The book's mean alone as the training label.** Trained on the archived book mean instead
  of the outcome, the plain model raises priced-row error in four of nine cells, by 5 to 12
  percent, and lowers it in none. At the posted rungs it costs NBA PR and PTS 56 to 65 × 10⁻⁴
  of Brier and helps NFL receiving yards by 20 and NHL blocked shots by 11. A half-and-half
  blend of the book mean and the outcome is a different matter and stays open (§10 item 4):
  it cuts priced-row error by 0.35 to 0.6 percent in NBA PRA, NBA PTS and NFL receptions, and
  costs NHL shots 1.2 percent.
- **Row re-weighting and quoted-only training** (§6.3).
- **Training on the newest 35 percent alone** (§5). Recency weights go with the refit
  (§10 item 10).
- **NHL zone starts and ice-time history** (§7.1).
- **NHL team power-play and penalty-kill rates**, and the rink factor for hits and skater
  fantasy (§7.3).
- **The opposing lineup for runs and walks allowed**, and pitch-level inputs for batter
  strikeouts (§7.3).
- **NBA.com families outside AST and PA** (§7.4): shot profiles, tracking, passing and drives,
  matchups and opponent allowances cost PRA, PR, REB, FG3M and BLK on the rows we price, and
  minutes structure helps only the minutes cell.
- **More rows of the same kind** (§5): expect under 1 %.

## 10. Recommended order

Each item is one change with its facts.

1. **Give NBA and WNBA serving a full schedule.** `fetch_upcoming_games` reads a list of
   broadcast games only, so 145 of 196 home team-games since 10-20 were served as away (§8).
   Point it at a source that lists every game. Serve-side; no retrain.
2. **Map the app labels the board drops.** Underdog's NHL "Shots on Goal" and "Faceoffs
   Won", Sleeper's "faceoffs_won" and the four MLB pitcher labels Underdog renamed on 09-29
   are missing from `stat_map.json` (§14 item 14). Underdog's NHL shots props, 5,695 posted
   since 09-19, reach the board the day the label maps. Config only; no retrain.
3. **Write a plain-baseline column at training time.** One extra LightGBM fit per cell, its
   held-out MSE ratio stored in `model_stats`. It makes §4 a standing per-cell number.
   Report-only.
4. **Commission the brief on a learned anchor.** The questions for it: should
   `centered_additive_*` and `ratio_*` anchor on a walk-forward plain mean instead of a
   rolling statistic, should the mean get its own round count, and should the cells where it
   helps train on a half-and-half blend of the outcome and the book mean (§9)? The evidence is
   §4.3 and §4.4. The target is the 12 cells of §4.1.
5. **Filter unquoted NBA and WNBA rows on minutes known before the game.** One condition in
   `get_training_matrix`, a matrix rebuild, then the §2 verdict with the lower tail in view.
6. **Record the lineup source in the MLB feature log.** One column: posted or fallback. It
   costs nothing in the off-season and makes the lineup spot among starters trainable next
   year (§7.2).
7. **Keep the NHL fields the game file already holds.** Power-play time, short-handed time
   and faceoffs lost sit in the moneypuck rows `parse_game` reads (§7.3). Power-play role
   lowers both error and Brier at the priced line in points and power-play points. Store the
   fields, backfill them from the NHL stats service, and retrain those two cells, plus faceoff
   wins once item 2 lets the board score it. NHL is in season and is the one league §11 keeps
   as a candidate.
8. **Sum the MLB pitch feed per player and game before Opening Day.** The feed is already
   downloaded. The pitcher's record with the opposing lineup cuts the Brier score at the
   priced line in pitcher strikeouts by 36 × 10⁻⁴, and the leash cuts the error on pitches
   thrown by 4.2 percent, whose projection feeds every pitcher cell.
9. **Batch the small inputs.** NHL shot attempts and expected goals as recency-weighted
   per-minute rates; NBA last-game and three-game minutes, and per-minute assist rates as
   ratios of sums; MLB handedness with split history, lineup-spot history and batter contact
   quality (§7.1, §7.3). Each is worth 0.1 to 0.7 percent on priced rows and several lower the
   Brier score at the line, so judge them together by the §2 verdict, once.
10. **Decide on refitting through the newest rows.** About 1 percent on priced rows in NBA
    PRA and NFL receptions and receiving yards, and 8 to 10 × 10⁻⁴ of Brier at the posted
    rungs in NFL receiving yards (§5). It changes what the gates measure, so it comes after
    item 4 and only if that lands.
11. **Trace the NBA comps drift, then date the comps' season tables.** Rebuilt comps agree
    with serving more as the row nears the rebuild date (§14 item 6). The shot-location and
    bio tables take a date range; the play-type tables need a stored daily capture (§7.4). NBA
    play-by-play earns a collector for AST once its posted rungs confirm the priced-row gain,
    or for minutes once the minutes model has a consumer.

## 11. An edge against the DFS apps

The sections above improve the model's mean. This one asks what would make the dashboard's
recommendations pay. It restates the problem as the live-gap lane defined it, reads the live
record and the offline replay as they stand, and ends with steps in order.

Four terms. A **standard pick** is an Underdog pick that pays the same on either side (1.00×
in the app); the repo values it at 1.83 a pick. A **priced leg** is any other: an Underdog
pick whose payout the app raised or cut, and every Sleeper leg. **Claimed** is the model's win
probability for the side it chose. **Needs** is the hit rate at which a leg breaks even.

### 11.1 The problem as the record defines it

The dashboard recommends a leg when claimed probability times payout, less one, is at least
0.05 and the payout is at most 2.5 (`realized.recommended`). Three results from the live-gap
lane explain why those legs lose. Their homes are [live-gap-2.md](handoffs/live-gap-2.md) §5
and the briefs it cites.

- **The price.** A standard pick needs .546 at 1.83. Real entries move that a little: 2-, 3-
  and 4-pick Power and 4-pick Flex need .535 to .537 a pick, while 5- and 6-pick Power and 3-
  and 5-pick Flex need .547 to .554 ([honest-receipts.md](handoffs/honest-receipts.md)). The
  app keeps about 6.5 percent of a standard pick, 10.8 percent of a priced Underdog pick, 19.9
  percent of a pick offered on one side only and 12.3 percent of a Sleeper leg
  ([consensus_sanity_check.md](archive/consensus_sanity_check.md)).
- **The cause.** The app's line already sits at the market's median, and the rule picks the
  props where the model disagrees with that line most. Where a sportsbook prices the same line
  near even, the model's lean adds nothing to the book's in NBA (weight −.11 [−.29, +.11]),
  little and unsteadily in NFL, and a real amount only in NHL (+.67 [.39, .94]). A leg needs a
  weight near .40 to pay
  ([researcher_market_centre_anchor.md](archive/researcher_market_centre_anchor.md)).
- **The ceiling.** A forecaster who used every knowable input perfectly would hit about .601
  at the median line in NBA and .549 in NFL. In NHL goals, assists and points, MLB batter
  counts and NFL touchdowns the book already sits on the noise floor. The model captures 0 to
  7 percent of the room there is
  ([researcher_model_class_and_edge.md](archive/researcher_model_class_and_edge.md)).

The owner's rule for any fix: prove it on an offline replay of realized money that tracks the
live record, and let live confirm the replay. No rule without a model behind it, and no live
result used as a threshold.

### 11.2 The live record through 11-15

Source: the dev box's copy of `history.parquet`, priced by `realized.settled_offers`, games
from 10-09 (each prop counted once at the lines still posted when it was last scored).
Brackets resample whole game dates.

**By kind of leg.** Recommended legs: 8,137, hit .489, return −9.5 % a leg [−12.5, −6.5].

| Kind of leg | Legs | Claimed | Hit | Needs | Return per leg |
|---|---|---|---|---|---|
| Underdog standard pick | 2,273 | .605 | .519 [.494, .544] | .546 | −5.0 % [−9.7, −0.5] |
| Underdog pick with a raised payout | 2,101 | .550 | .451 [.432, .472] | .485 | −7.6 % [−11.6, −3.3] |
| Underdog pick with a cut payout | 523 | .720 | .585 [.540, .633] | .651 | −11.4 % [−18.0, −4.4] |
| Sleeper leg | 3,229 | .611 | .477 [.454, .499] | .542 | −13.5 % [−17.8, −9.2] |

Priced legs are 72 percent of what the dashboard recommends, and each kind loses with its
interval clear of zero. The lane read the same three signs on an earlier sample. (Eleven legs
priced 1.83 on one side only are left out of the table.)

**Standard picks by league.**

| League | Legs | Hit | Return per leg at 1.83 |
|---|---|---|---|
| NBA | 1,216 | .497 [.458, .530] | −9.1 % [−16.4, −2.5] |
| NFL | 414 | .524 [.466, .551] | −4.1 % [−14.2, +0.9] |
| NHL | 609 | .557 [.516, .603] | +1.9 % [−5.6, +10.1] |

NBA has fallen week by week since opening night: .538, .502, .495, .451. NHL has read .552,
.569, .495, .573 and .628 over its last five weeks.

**Standard picks by what the sportsbooks showed at scoring time.**

| Class | Recommended legs | Share of that class recommended | Hit |
|---|---|---|---|
| No sportsbook line for the prop | 1,480 (65 %) | 32 % | .513 [.486, .538] |
| Sportsbook line equals the app's line | 417 (18 %) | 7 % | .506 [.447, .559] |
| Sportsbook line differs from the app's | 376 (17 %) | 14 % | .559 [.491, .615] |

Two thirds of the recommended standard picks sit on props no sportsbook prices. Most are the
app's own fantasy scores: NHL skater fantasy 383 legs, NBA fantasy 279, NFL fantasy 245, plus
247 NBA PRA legs with no sportsbook line at scoring time. No book restrains the model there,
so it recommends a third of everything it sees. Where a book agrees with the app's line the
blend holds the model to 7 percent, and those legs lose at the same rate.

Five further reads, each a candidate edge or a candidate cause:

- **The book's line as a guide.** On 2,763 standard picks the sportsbook line sat at least
  half a point from the app's. The side toward the book's line won .526 [.505, .548], and
  .542 [.515, .570] where the gap was exactly one point. The lane called this the one classic
  edge left to read forward. It does not clear .546 here.
- **A side lean.** Blind Under won .514 [.502, .526] of all 13,033 standard picks. The
  model's recommended Unders, which the lane read at .554 through early October, have hit
  .511 [.476, .543] since; its Overs have hit .538.
- **The model against a blind side, on props no book prices.** In NHL, blind Under won .501 of
  2,198 legs and the model's picks won .553 [.508, .599] of 557: the model adds five points
  there. In NFL fantasy, blind Under won .541 of 580 legs and the model's picks, 92 percent of
  them Unders, won .502. In NBA fantasy the model's picks won .448 [.387, .509] of 279, worse
  than a coin.
- **Absent teammates.** NBA recommended standard picks hit .424 with no rotation player out,
  .519 with one and .507 with two or more. Missing news about teammates is not where the NBA
  losses come from.
- **The `Home` flag.** NBA picks on team-games served the wrong flag hit .485, against .506 on
  the rest, with intervals that overlap. The defect is real (§8); its cost is not measurable
  here.

### 11.3 The offline replay on the same rule

`sportstradamus admin realized-replay` prices each cell's held-out rows at the lines the app
posted, applies the 5 %-edge rule, and keeps the recommended standard picks
([ship_gate.md](ship_gate.md), "The realized replay"). I ran it on the dev box's current test
sets and models, the 11-16 retrain, with its outputs sent to scratch.

| League | Replay legs | Claimed | Replay hit (lower bound) | Live hit on the same rows |
|---|---|---|---|---|
| NBA | 540 | .603 | .522 (.487) | .493 on 402 |
| NFL | 173 | .601 | .514 (.484) | .479 on 165 |
| NHL | 278 | .596 | .514 (.486) | .570 on 200 |

The replay already holds most of the shortfall: it reads a claimed .60 at .51 to .52 before
any live effect. No league's lower bound reaches .546. By market it agrees with live on the
three largest: NBA fantasy .505 against .474, NFL fantasy .503 against .504, NHL skater
fantasy .552 against .563. On the older held-out rows at market lines (the replay's league
scope) NBA reads .534, NFL .545 and NHL .586 with a lower bound of .569, and the lane's
league-level screen (the L2 column) clears NHL alone.

Two cautions. None of the live legs came from the models in these dumps, so this run cannot
certify that the replay tracks live; the lane's forward read, each served version on the dates
it served, came within 1.3 points a league. And the NHL figure deserves a look before the
11-16 NHL models serve: the replay recommends 47 points legs and 15 assists legs where the
serving models recommended 10 and 4, and those 62 legs hit .452. Points and assists are two
of the markets where the book sits on the noise floor.

**Why the gates pass cells that lose.** Applied to the held-out rows by the kind of line each
row carries, the rule reads:

| Line on the held-out row | NBA | NFL | NHL |
|---|---|---|---|
| The pipeline's own stand-in line | .702 | .769 | .737 |
| Sportsbook line priced within .06 of even | .540 | .543 | .581 |
| App line priced exactly even | .541 (338 picks) | .521 (457) | .691 (68) |
| Share of held-out rows on a stand-in line | 50 % | 46 % | 48 % |

Calibration and the gates read all of these rows together, and half carry a line no market
set. The lane named this hole on 10-29; the table gives its present size.

### 11.4 What this study adds

- **The mean fit is the one model change found here that could raise the weight the cause
  turns on.** In NBA and NFL the served mean is less accurate than its own inputs allow (§4).
  On the rows we price the plain mean cuts squared error by 2 to 7 percent, and the Brier
  score at the posted rungs by 16 to 30 × 10⁻⁴ in NBA PRA, REB and FG3M and NFL receptions.
  It also earns about twice the blend weight (.05 to .16 in NFL receptions, .10 to .21 in
  rushing yards, .01 to .06 in NBA PRA). None of this has been through the replay.
- **A fresher fit pays at the posted line in one cell.** Fitting through the newest rows cuts
  the Brier score at NFL receiving yards' posted rungs by 8 to 10 × 10⁻⁴, and priced-row
  error by about 1 percent in NBA PRA and two NFL cells (§5).
- **Discarded play-level data pays at the line where books and apps post the market.** The
  pitcher's record with the opposing lineup cuts the Brier score at the priced line in
  pitcher strikeouts by 36 × 10⁻⁴; NHL power-play role and MLB batter contact cut it by 2 to 4
  in points, power-play points, walks, home runs and hits+runs+RBI (§7.3). NBA.com's endpoints
  help AST on priced rows and cost PRA, PR and REB (§7.4). NHL skater fantasy, the app-only
  cell where the model shows live skill, gains from none of them.
- **Late knowledge is the one large effect, and the board never prices it.** The served model
  prices quoted MLB hitters who do not start at 0.77 hits; they make 0.20 (§7.2). The apps
  pull a non-starter's props a median 186 minutes before the game's last poll, when lineups
  post, so no posted line carries that knowledge. What the board does price is the lineup spot
  among starters, worth 0.15 to 0.23 percent.
- **Most defects in §14 are worth fixing and unlikely to move the hit rate.** The lane measured
  the serve-against-rebuild effect on probability as small, and the `Home` split above agrees.
  The unmapped labels differ: they keep Underdog's NHL shots props off the board entirely
  (§14 item 14).

### 11.5 Recommendations

1. **Recommend standard picks only.** Priced legs cost 10.8 to 19.9 percent before the model
   says a word, against 6.5 percent on a standard pick, and they have lost in every read. This
   is a rule about the app's cut and has no model behind it, so it needs the owner's
   exception to the 10-04 rule. The ask is already open in honest-receipts §8.
2. **Treat NBA and NFL standard picks as unproven and NHL as the one candidate.** NHL is the
   only league where the offline weight, the older replay rows and the live record point the
   same way. Even there the live interval starts at .516 and the replay on the retrained
   models reads .514.
3. **Do not pick sides by the sportsbook's line alone.** The lane found the apps already cut
   the payout wherever the books lean, and the line-side read above stops short of
   break-even. Representing the line inside the model is a separate matter: a candidate that
   brings the model closer to the book is judged on the rows we price like any other (§2).
4. **Look for edge where no sharp market exists to copy.** The app's own fantasy scores are
   the bulk of the board that no book prices. The model shows skill there in NHL, none in NFL
   and less than none in NBA. NBA fantasy is the largest single loser and the first cell to
   hold or repair.
5. **Build only the entry kinds that need .535 to .537.** That means 2-, 3- and 4-pick Power
   and 4-pick Flex, one leg a game. All four pair modifiers the owner checked against the app
   were wrong in the config and the rest are unproven, so a same-game pair's payout is a
   guess until the app quotes it.
6. **Do not size on the claimed probability.** It reads .605 where the record reads .519.
   Stake flat until a league's replay lower bound clears break-even.

### 11.6 Steps toward profit, in order

No step makes today's list profitable. The first two stop losses that are certain; the rest
are the work that can raise the hit rate, each judged by the replay.

1. **Cut the recommended list to Underdog standard picks.** On this record the return per
   recommended leg moves from −9.5 % to −5.0 % with no model change. Owner's decision.
2. **Hold NBA and NFL out of the recommended list and leave NHL in.** This is the lane's
   release gate (L1). The owner's rule lets it read only a replay shown to track live,
   so the step before it is the lane's open one: run the forward, same-version read through
   the command. Expect it to need pooling. A hit rate on 540 legs carries an interval of about
   ±.04, and a .03 tolerance needs near 2,000 matched legs a league.
3. **Fix the free defects.** The NBA schedule source, the unmapped app labels, the dead
   columns and the others in §14.
4. **Put the mean fit through the replay.** After the research brief of §10 item 4, score the
   candidate on the replay's posted standard picks by its error at the posted line, with the
   weight at the median line read beside it as information. Start with NBA PRA, REB and FG3M
   and NFL receptions, where the posted rungs already show a gain, then NBA fantasy, NFL
   rushing yards and NFL fantasy. Score the NHL play-level inputs of §7.3 the same way,
   starting with points and power-play points, where they gain at the line.
5. **Refit through the newest rows and score it the same way** (§10 item 10).
6. **Turn a league on only when its replay lower bound clears break-even.** Then stake flat,
   in 2- and 3-pick Power entries, one leg a game. For scale: at NHL's live .557 a 2-pick
   Power entry returns +8.6 %, and −6.8 % at the interval's low end of .516.
7. **Collect what the line may lack.** The app's line states and entry counts, lineups,
   injuries and weather now ride every run (added between 11-06 and 11-13) and become
   trainable through 2027. Test the MLB non-starter question in April.

The ceiling sets expectations. NBA has room (.601 against .546). NFL has almost none (.549).
Among the NHL markets the books price, only shots, blocks and hits have room; the app's own
fantasy scores have no ceiling estimate.

### 11.7 What the lane killed

The live-gap lane killed these as serving rules or as sources of edge in themselves:
shrinking picks toward the book, the trust layer, anchoring on the app's line, importance
weighting toward the live board, the near-even blend weight, a temperature for pick'em rows,
a learner swap, a wider model pool, the lead-time blend, selection on the book's lean,
predicting the close, and injuries, goalies or line movement as sources of edge
([live-gap-2.md](handoffs/live-gap-2.md) §5,
[researcher_information_audit.md](archive/researcher_information_audit.md)). This study
re-tested none of them. §15 separates those that stand on the rows we price from those
judged another way. An input one of them touches, such as a teammate's absence, stays open
as a model input, judged on the rows we price (§2).

## 12. Limits

- One date split per cell. Intervals cover row sampling by game date, not the choice of
  split.
- The plain model has fixed settings and no search. It understates what a tuned mean model
  would do, and it says nothing about distribution shape.
- No served model was retrained. §4.5 and §10 item 4 are proposals, not results.
- The priced rows' own lines make a weak test at the line. They sit near each player's
  median, where the probability stays near one half whatever the mean, and a Brier interval
  on 1,000 to 2,500 such rows is about ±0.001 wide, the size of the effects one would hope
  for. The posted rungs add alternate lines and separate means better.
- The posted rungs cover only the weeks since the archive ladder began (§2): 26 NBA game
  dates, and MLB from 08-30 to the World Series. Their intervals are wide, so a gain on
  priced rows that the rungs leave level is unconfirmed, not refuted.
- Brier at the line scores each mean through a logistic map fitted within the same
  population, cross-fitted by date. It measures how well a mean ranks outcomes against the
  line, not the served probability, which adds the distribution's shape and the book blend.
- The decoded book mean is unreliable in count cells (§2).
- Artifacts are the dev box's as of 2026-11-16. Prod may serve older models.
- §11 reads the dev box's copy of `history.parquet`, 38 game dates, and its intervals resample
  whole game dates 2,000 times. Prod holds the record of truth; re-read any single number
  there before acting on it. I cut the record many ways, so a lone split that stands out is a
  lead and no more.
- A trap met twice while testing MLB handedness: a joined column left blank only for players
  who did not bat produced a false 2.7 percent gain, because the blank marked the outcome.
  Any new column needs the same check: compare the mean outcome of its blank rows with the
  rest. Filling the blank with zero does not remove the leak.
- §7.3 prices discarded data with history pulled for the study from public sources, not from
  the files the update reads. The fields match where I checked, but a build must use only
  fields both carry.
- §7.3 and §7.4 read 300 and 336 intervals on priced rows and posted rungs, so about 8 in
  each would clear zero on each side by chance. The sets overlap (the full set holds every
  family), so the intervals are not independent. A gain in one cell from one set is a lead;
  AST and pitcher strikeouts stand because several sets agree.

## 13. Scripts

All in `~/backups/sportstradamus/2026-11-16-gbdt-feature-study/scripts/`, except the §7.3 and
§7.4 sets, which sit in `pbp/` beside it with the downloaded history (`pbp/data/`,
`pbp/nba/data/`) and the screen logs (`pbp/results/`, `pbp/nba/results/`), and the scoring on
priced rows and posted rungs, which sits in `priced/` with its scores (`*_rescore.csv`) and
logs. Each reads the matrices, test sets, game logs, archive (read-only), feature log, live
history or public sources, and writes nothing to the repo.

| Script | Section |
|---|---|
| `x4_same_rows.py`, `x5_decomp.py`, `x15_seed.py`, `x16_hp.py` | §4.1–4.3 |
| `x6_pover.py`, `x11_checks.py`, `x2_gap.py` | §4.4 and the input-parity check of §2 |
| `x8_fresh.py` | §5 |
| `x7_usage.py`, `x7b_bias.py`, `x12_shift.py` | §6 |
| `x10_cands.py`, `x13_lineup.py`, `nba_feats.py` | §7.1, §7.2 |
| `x14_parity.py` | §8 |
| `x3_teacher.py` | §9 |
| `h2.py`, `h6.py` (leg kinds, leagues), `h3.py`, `h4.py` (book-line classes, line side), `h5.py` (absences), `h10.py` (blind sides) | §11.2 |
| `h8.py`, `h9.py` (rule on held-out rows by line type); `replay.csv` (the repo's replay, run to scratch) | §11.3 |
| `home_probe.py`, `home_verify.py`, `bugchk.py`, `agechk.py` | §8, §14 |
| `pbp/`: `dl_statcast.py`, `dl_nhlrest.py` (history pulls), `mp_rest_check.py`, `sc_starts_check.py` (source checks), `mlb_cands.py`, `nhl_cands2.py` (candidates), `screen.py`, `run_jobs.sh` (the screen), `inspect_*.py` (feed contents), `defects_check.py` | §7.3, §14 items 11–13 |
| `pbp/nba_probe.py` and, in `pbp/nba/`, `probe2.py` to `probe4.py` (endpoints), `pull.py`, `pull_pt.py` (history pulls), `nba_cands.py` (candidates), `blank_check.py`, `nba_jobs.txt` with copies of `screen.py` and `run_jobs.sh` (the screen) | §7.4 |
| `pbp/nba_comps_week.py` (weekly agreement of the comps columns) | §8, §14 item 6 |
| `priced/`: `rungs.py` (posted rungs from the archive ladder), `screen2.py` and `run_dump.sh` (the screen, keeping per-row predictions), `x10_build.py` (§7.1 and §7.2 candidates for it), `arms_dump.py` (per-row predictions of the §4, §5, §6.3 and §9 arms), `rescore.py` (priced rows and posted rungs), `summarize.py` (intervals counted against chance) | §2 and every priced column |

## 14. Defects found

Everything the study turned up that is wrong, dead or fragile, in one list. "Verified" means I
reproduced it from code and data on the dev box. Nothing here has been fixed.

| # | Defect | Where | Measured effect | Status |
|---|---|---|---|---|
| 1 | NBA serving marks most home teams as away (WNBA shares the code) | `fetch_upcoming_games` | flag wrong on 41 % of NBA rows; cost not measurable | Verified, cause found |
| 2 | Unquoted training rows are chosen on that game's realized usage | `get_training_matrix` | 53 % of NBA PRA rows; harm to the mean not shown | Verified |
| 3 | `weather lead` marks the era, not the weather | weather features | 15–24 % of a period classifier's split gain in NFL | Verified |
| 4 | Columns that are constant or use zero for unknown | matrices | none measured | Verified |
| 5 | NBA `Player age` served one year low for four weeks | serving | none measured | Verified, cause not traced |
| 6 | Inputs that differ between serving and the rebuilt matrix | serving and matrix | small on probability (lane) | Verified, wrong side unknown |
| 7 | MLB hitter rows for players who never batted | MLB matrices | 916 quoted rows in MLB hits | Verified; settlement rule unchecked |
| 8 | Stale test-set files beside current ones | `data/test_sets/` | none measured | Verified |
| 9 | Search folds and the validation split are not forward in time | training pipeline | not measured | Design weakness |
| 10 | NFL usage cutoff appears not to bind | `get_training_matrix` | none | Not verified |
| 11 | NHL time-on-ice matrix holds goalie rows | NHL `timeOnIce` matrix (served on devel) | 22 rows; the 9 held out carry 9 % of held-out squared error | Verified |
| 12 | NHL game log keeps 4 Nations Face-Off games | `_collect_game_ids` | 267 player rows; 12 to 212 rows in each NHL matrix | Verified |
| 13 | Hitter rows in MLB walks allowed | MLB `walks-allowed` matrix | 131 quoted rows on 2026-03-26 | Verified in the matrix; source not traced |
| 14 | App labels missing from the stat map: Underdog's NHL shots and faceoffs, four MLB pitcher labels Underdog renamed on 09-29, Sleeper's faceoffs | `stat_map.json`, Underdog and Sleeper sections | no Underdog NHL shots prop scored since 09-19; 5,695 posted | Verified |

**1. NBA `Home` flag.** `fetch_upcoming_games`
([stats/base.py](../src/sportstradamus/stats/base.py)) builds the schedule from stats.nba.com's
`internationalbroadcasterschedule`, which lists only games with an international broadcast.
Probed on 2026-11-17: 2 of 11 games on 10-21, 2 of 6 on 11-10, 2 of 10 on 11-11, 2 of 9 on
11-13, 1 of 5 on 11-14, 3 of 3 on 10-20. A team with no entry is served `Home = False`. The
live history shows it: 18.8 percent of NBA team-dates served as home in October and 12.4
percent in November, against 49 to 50 percent in NHL and NFL. The WNBA path calls the same
function; October's 18 WNBA team-dates are too few to judge, so check before May. Fix: one
function, pointed at a source that lists every game. Evidence in §8 and §11.2.

**2. Row selection on a post-game value.** An unquoted row stays in the matrix only when the
player's usage in that same game clears the 15th percentile
([stats/base.py](../src/sportstradamus/stats/base.py), `usage > usage_cutoff`). In NBA PRA 53
percent of rows are unquoted, and none of them is a game of 11 minutes or fewer. Fix: test
the usage known before the game. Evidence in §6.1.

**3. `weather lead`.** The column is 0 on every NFL row before 2024, where the weather is
observed history, and 1 on about three quarters of rows since. A model can read it as "this
row is from the quoted era". Fix: drop it from the inputs, or give training rows the lead
serving will have. Evidence in §6.2.

**4. Constant columns and zero for unknown.**

- `Defense comps` is 0.0 on every row of all five matrices checked (MLB hits, NBA PRA, NFL
  receptions, NHL shots, WNBA AST).
- `Player age` is 0 on every row of the MLB hits, NFL receptions and NHL shots matrices. Age
  reaches no model in three leagues.
- In the NBA PRA matrix `Player age` is 0.0 on 7.5 percent of rows: 160 of 543 players have no
  age on file, and zero stands in for it. NBA age is a model input, so the model must learn
  that zero means unknown.
- MLB hits carries five more all-zero columns (`Player team`, `Player position`,
  `Player depth`, `Player position z`, `Defense position`), and `Player position` is still
  among that model's inputs.

Fix: leave unknown ages blank rather than zero, and either fill or delete the dead columns.

**5. NBA `Player age` at serve time.** From opening week to 11-13 the served age was exactly
one year below the rebuilt value on 2,247 of 2,747 joined NBA PRA rows. The rebuilt value is
the right one: a player born 1998-09-19 was served as 27.1 on 11-12, a year short of 28.1. On
11-14 and 11-15 the two agree on all 200 rows. On 260 more rows one side reads zero. I did
not trace what changed on 11-14. Check that the next season rollover does not bring it back.

**6. Inputs that do not survive the trip.** Between the serve-time feature log and the rebuilt
matrix, NBA `Player comps z` correlates 0.38, `Defense comp distance` 0.57, `Player depth`
0.58 (different on 29 percent of rows) and `Player position` differs on 19.5 percent; in NHL
`Player moneyline gain` correlates 0.02 and `Player totals gain` 0.19. I did not establish
which side is right. Evidence in §8.

In NBA the agreement rises toward the rebuild date: `Player comps z` correlates
.29, .28, .36 and .54 over the four weeks to 11-15, and `Defense comp distance` .44 to .76.
That is the mark of an input built from a snapshot taken later than the row. The NBA comps
weight season-to-date shot-zone, play-type and bio tables (`_fetch_league_endpoints` in
[stats/nba.py](../src/sportstradamus/stats/nba.py), weights in `playerCompStats.json`), and a
rebuilt row reads those tables as last fetched. That is the likely cause; I did not trace it.
The shot-location and bio tables take a date range, so a rebuild could read them as of each
row's date; the play-type tables take none, and only a stored daily capture dates them (§7.4).

**7. MLB hitter rows with no plate appearance.** The MLB hits matrix holds 916 quoted rows for
players who never batted, and about 6 percent of quoted held-out rows are hitters who did not
start. They train and grade the model on props the apps may void. Check the apps' settlement
rules, then decide whether those rows belong in the matrix. Evidence in §7.2.

**8. Stale test-set files.** `data/test_sets/` holds dumps for cells that no longer train
beside the 11-16 ones: NBA PTS, FG3A, FTM and PA from 10-18, NBA FGA from 08-06, NBA PF and TOV
from 07-02, NFL passing yards from 09-26, NFL qb yards from 10-18 and NFL fantasy points
prizepicks from 10-27. Anything that reads the folder whole mixes eras. More than half the
NBA PTS rows no longer join the current matrix (§4.1).

**9. Splits that look ahead.** The hyperparameter search scores four contiguous date blocks,
each with a model trained on the other three, so three of four folds train on rows newer than
the rows they score. The newest 30 percent is halved into validation and test by a hash of
player and date, so anything fit on the validation half shares game dates with the test half
that grades it. Both make the offline read kinder than a forward one. I did not measure by
how much.

**10. NFL usage cutoff (not verified).** Unquoted NFL rows with a snap share of 0.01 are in
the matrices, yet the 15th percentile of snap share over the whole game log is 0.18. Either
the cutoff is computed on a different slice than I assumed or it does not bind in NFL. I did
not trace it.

**11. Goalies in the NHL time-on-ice matrix.** The skater market's matrix holds 22 goalie rows
(`Player position` 4.0), all on the pipeline's stand-in line. The plain model prices the nine
held-out ones at 23 to 37 minutes; eight of them played 56 to 63. Those nine rows carry 9
percent of the held-out squared error, so they distort any read of the cell, §7.3's included.
The cell serves on devel. Fix: drop goalies from skater markets when the matrix is built.

**12. 4 Nations games in the NHL game log.** `_collect_game_ids`
([stats/nhl.py](../src/sportstradamus/stats/nhl.py)) keeps every game the schedule lists. The
game log holds 229 player rows from the 4 Nations Face-Off round robin (game type 19, Canada,
Finland, Sweden and the United States, 2025-02-12 to 02-17) and 38 from its final (type 20,
2025-02-20), with the national teams as team and opponent. They reach the matrices: 212 rows
in power-play points, 206 in goals, 108 in time on ice and 12 to 184 in each of the rest. The
log also holds 38 rows from one preseason game (type 01, 2023-10-06) that predate every
matrix. Fix: keep regular-season and playoff games (types 02 and 03) when collecting ids.

**13. Hitter rows in MLB walks allowed.** The `walks-allowed` matrix holds 131 rows on
2026-03-26 for hitters such as Mike Trout and Mookie Betts. None of them faced a batter; all
carry a sportsbook quote and an outcome of 0. Most likely a batter-walks line was filed under
the pitcher market on opening day; I did not trace whether the archive or the matrix build
does it. Five other pitcher matrices hold 2 to 4 rows each for pitchers who faced no batter;
pitches thrown holds none. Fix: trace the opening-day market name, then drop rows with no
batter faced.

**14. App labels the board never scores.** Underdog posts NHL shots as "Shots on Goal" and
faceoff wins as "Faceoffs Won". The Underdog section of
[stat_map.json](../src/sportstradamus/data/config/stat_map.json) maps "Shots" but neither of
these, so neither has a cell. The archive ladder holds 5,695 Underdog shots props still posted
at their last poll from 09-19 to 11-15, 5,446 of them on players Sleeper also posted that day,
and 961 faceoff-wins props on 124 players. The live history holds 10,704 NHL shots rows, every
one from Sleeper, and no Underdog shots row. Sleeper's "faceoffs_won" (55 props since 10-31)
is unmapped too. On 09-29 Underdog renamed four MLB pitcher labels: "Pitching Outs" became
"Pitching Outs Recorded", and "Hits Allowed", "Walks Allowed" and "Earned Runs Allowed" took
a "Pitcher" prefix. The old labels stop that day and none of the new ones is mapped, so
Underdog's 78 to 84 postseason props in each never reached the board; "Strikeouts" kept its
label. Fix: add the labels to the Underdog and Sleeper sections before Opening Day. NHL shots
then reaches the Underdog board; faceoff wins stays withheld until its cell ships.

Two related items are not code defects. Half of all held-out rows carry a line the pipeline
made up, and calibration and the gates read them with the rest (§11.3; the lane's open
research hole). And any test of a new column must first compare the outcomes of its blank
rows with the rest (§12).

## 15. Earlier decisions that may not survive the rule

The rule in §2 ([CLAUDE.md](../CLAUDE.md), "Judging model changes") is newer than most model
decisions in the repo. Many of them were judged on all held-out rows, about half of which
carry a stand-in line, on what a candidate adds beyond the book, or on the recommended legs
alone, and some counted a move toward the book against a candidate. Such a verdict is
unverified, not wrong. Re-judge it on priced rows and posted rungs before building on it or
citing it to rule a candidate out. These are the most exposed.

- **The blend-weight floor.** Under `ship sweep --confirm --min-model-weight T`, a recipe
  whose fitted blend weight falls below `T` never nominates, and a refit that lands below it
  is reverted ([ship_gate.md](ship_gate.md), "research → devel, supersede an incumbent"). The
  NFL lane has run every confirm walk at 0.20 since 09-29, and its notes name the floor as
  what holds NFL passing yards withheld ([nfl-ship15-recovery.md](handoffs/nfl-ship15-recovery.md),
  "Gate 1 evidence and the weight floor"). The floor answers a real flaw: ranking by gate
  headroom favours the lowest weight on a weak cell
  ([low_weight_models.md](handoffs/low_weight_models.md)). But it answers by barring closeness
  to the book, not by scoring error at the posted line.
- **Recipes chosen on the gates.** The strategy sweep ranks a cell's recipes (family, target
  normalization, post-hoc) by their least headroom across the six gates, and the same gates
  decide whether a cell may serve. Gate 1 and the blend weight read sportsbook quotes only,
  but Gate 5 and the temperature are fit on rows 40 to 61 percent of which carry a stand-in
  line ([live-gap-2.md](handoffs/live-gap-2.md) §5; §11.3 here). The multiplier normalization
  the sweep chose for NBA MIN and NFL receiving yards costs those cells the whole of the plain
  mean's lead (§4.3).
- **The live-gap lane's pulls toward the book.** The lane judged serving changes by the gap
  between claimed and realized hit rates on recommended legs. One pull of the served
  probability toward the book, fitted on quoted NFL and NHL legs from October, cut their Brier
  score by .006 to .010 (60 to 100 × 10⁻⁴, more than any gain in this study). The brief handed
  its level to the I6d lane and noted that it made the recommended tail no better
  ([researcher_lead_time_blend.md](archive/researcher_lead_time_blend.md)). The selection
  shrink was judged on the selected legs alone, and the pick'em shrink was killed because it
  removed every pick'em-only recommended leg. The DFS-line centre anchor was killed because it
  fits to near-full strength, which the lane read as abstention, and under #0d, the trust
  layer's 10-04 kill, which bars any probability consumer from anchoring on a learned market
  layer as model-free in disguise
  ([model_improvement_track.md](handoffs/model_improvement_track.md) §3.4 and §8.2).
  #0d counts closeness to the market against a candidate, so whether it stands is the owner's
  call.
- **Kills by what a candidate adds beyond the book.** Part B's plan kept a change only when
  one of two screen measures improved beyond seed noise and neither worsened beyond it: the
  cell's own loss over all rows, and its standing on sportsbook-quoted rows, which included
  the mean's incremental R² over the line and the book. That R² gives no credit for what the
  line already holds, and the screen never read pick'em rows or posted rungs. §3 lists what
  Part B tried. The learner swap (X1, 11-11) read the same R², and the wider model pool was
  killed because the book encompassed its members
  ([researcher_model_class_and_edge.md](archive/researcher_model_class_and_edge.md)).
- **Sources killed as edge.** The information audit tested injuries, goalies, line movement
  and predicting the close as sources of edge beyond the decision-time price. That closes
  them as edge in themselves; as model inputs they stay open (§11.7).

Three earlier kills already stand on priced rows: row re-weighting toward the live board
(§6.3), the near-even blend weight (near-even Brier improved in at most one league of three)
and a blend weight that varies with lead time (nothing in walk-forward Brier on quoted
legs). Selection on the book's lean is a rule for picking sides, not a model change, and
§11.5 item 3 keeps it closed.
