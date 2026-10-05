# Sportsbook-only consensus line: trace, measurements, design

Read-only investigation, 2026-10-04 to 2026-10-05, branch `devel`. Implements decisions 5 and 11 of
`docs/handoffs/honest-receipts.md`. Nothing in the repository was edited.

Line numbers were re-checked against the working tree at 00:15 on 2026-10-05. Other sessions
were editing `helpers/archive.py`, `prediction/cli.py`, `prediction/model_prob.py`,
`training/pipeline.py` and `stats/base.py` while this was written (one of them shifted most of
`archive.py` by four lines), so every citation also names its function; find by name if a
number is off.

Terms used throughout:

- **DFS platform**: a book in `helpers/training_quotes.DFS_PLATFORM_BOOKS` (ParlayPlay, PrizePicks,
  Sleeper, Thrive, Underdog). Only Underdog and Sleeper have rows in the archive.
- **Sportsbook**: any other book in `odds` (DraftKings, FanDuel, BetMGM, Caesars and so on).
- **Key**: one (league, market, game date, player), a player-market-day.
- **DFS-only key**: a key no sportsbook has posted a line for.
- **Mixed key**: a key with both a sportsbook and a DFS platform.
- **Line of record**: the median of every distinct line the `lines` table holds for a key. This is
  exactly what `Archive.get_line` returns today.
- **Sportsbook consensus line**: the median of each sportsbook's latest posted line, floored to the
  half point. This is what `Archive.get_line` should return.
- **Serving window**: the five weeks of game dates the brief measured,
  2026-08-30 to 2026-10-03.
- **Decision time**: the archive as it stood at a key's last DFS poll (what prophecize saw).
- **Final state**: the archive as it stands now, after close-lines has run.
- **Training cutoff**: game date 12:00 UTC, the `at=` training reads with
  (`Stats.get_training_matrix`, `base.py` ~2398: `gameDate + 20h − TRAINING_LOOKBACK(8h)`).

---

## 0. The answer on one page

**What is wrong.** `Archive.get_line` takes the median of every distinct line ever logged in
`lines`. That table has no book column and receives two kinds of row: each DFS platform's main
rung (from `add_dfs`, every prophecize run) and the cross-book sportsbook median (from
`merge_player_books`, every confer poll). So a DFS line moves the "consensus" whenever it differs
from the sportsbooks. A second, separate leak sits in the quote resolver: DFS rows vote on which
line the sportsbook consensus price is read at (`_direct_line_cohort`).

**What to change.** Section 4.2 has the code. The edit numbers below are used throughout.

| Edit | Change |
|---|---|
| 1 | `Archive.get_line` becomes the sportsbook consensus line, read from `odds.line` (the only place a line carries its book). It returns `0.0` for a DFS-only key. |
| 2 | New `Archive.get_reference_line`: the sportsbook consensus, else the line of record. For readers that need "the line this entry's price goes with". |
| 3 | The resolver's `legacy_line` (`get_training_quote_inputs`) and the NHL batch read (`get_ev_line_inputs`) return the reference line. |
| 4 | The three combo-subline call sites read `get_reference_line`. |
| 5 | `_direct_line_cohort` narrows to sportsbooks before the modal-line vote. **The only edit that changes a served probability.** |
| 6 | `_weighted_book_ev` narrows to sportsbooks before the divergence filter. |
| 7 | `fit_book_weights` drops the DFS platform columns. |
| 8 | `to_pandas` loses its unused `Line` column. |
| 9 | The board's `Consensus Line` is NaN (not 0.0) when no sportsbook posts a line, the lookup uses the archive's market name, and the `Alt Line` flag is judged against the reference line. |
| 10 | Comments and docstrings. |

`add_dfs` keeps staging into `lines`. No new table, no schema change, no data migration.

**The numbers that drive it.**

| Question | Answer |
|---|---|
| Does any model feature read the consensus line? | No. Zero features. No retrain is forced. |
| How often does the sportsbook consensus differ from today's `get_line`? (serving window, mixed keys, decision time) | 13.1% of keys: NFL 45.1%, WNBA 60.5%, MLB 8.2%, NHL 3.7%. Mean gap 0.76, median 0.5. |
| Same, on the board as it stands now (4,544 rows) | `Consensus Line` changes on 33.5% of rows (NFL 56%, WNBA 59%, MLB 15%, NHL 2.5%) and becomes NaN on 10.2%. |
| How many keys have no sportsbook line? (same window, modeled markets) | 46.5% at decision time (MLB 48.9%), 24.7% once close-lines has run. |
| Served probability: how many keys change? | 722 of 48,230 mixed keys in five weeks (1.5%): NFL 6.6%, WNBA 7.4%, MLB 0.8%, NHL 0.4%. 56 keys on the current board (44 NFL). About 200 more simple-combo keys see one component of their sum re-priced. |
| By how much? | Book-leg under-probability moves by a median 0.02 (NFL, WNBA, NHL) to 0.05 (MLB), p90 0.14 on MLB. The served probability moves by that times (1 − model weight); median model weight of served cells is 0.85. |
| Is the new quote better? | No worse where a sportsbook was already in the cohort (Brier 0.2441 → 0.2467, CI spans zero). Better where a DFS-only cohort had displaced book evidence (0.2438 → 0.2219, n=128). |
| Existing tests that move | 7 of 473 in the 47 test files that touch these paths, verified by running them against an in-memory patch: 1 semantic pin, 6 fake-archive attribute errors. |

**Owner decisions needed** (section 7): how the Alt Line flag treats a DFS-only entry; whether
the fantasy `_book_mean_shift` blend and the mixed line-movement diagnostic also fall under
decision 5.

---

## 1. The map

### 1.1 Tables and the reducer

`src/sportstradamus/helpers/archive.py`, local archive `archive/archive.duckdb`:

| Table | Columns | Rows | Book attribution |
|---|---|---|---|
| `odds` | league, market, game_date, entity, book, ev, observed_at, under_prob, line | 41.07M | yes |
| `lines` | league, market, game_date, entity, line, observed_at | 21.38M | **none** |
| `ladder` | per-book rungs with prices | 31.46M | yes |

`_consensus_line(values)` (`archive.py:190`) is `floor(2 × median) / 2`, and `0.0` when the list
is empty or the result is NaN. Every "consensus line" in the code goes through it.

Books present in `odds`: Underdog (8.66M rows, from 2026-03-16), Sleeper (3.95M, from 2026-03-16)
and fifteen sportsbooks. Since 2026-08-30 the two DFS platforms are 5.36M of the 5.99M player-prop
`odds` rows (89%).

### 1.2 Writers into `lines`

`_stage_line` (`archive.py:1188`) is the only staging function; `write()` flushes with
`INSERT INTO lines SELECT * FROM lines_df` (`archive.py:1424`). Rows staged in a run are invisible
to readers until that flush.

| # | Writer | Site | What line it stages | `observed_at` | When |
|---|---|---|---|---|---|
| 1 | `Archive.add_dfs` | `archive.py:1317` (def 1252); called from `prediction/scoring.py:97` (`_match_league_offers`) | Each DFS platform's rung priced nearest even money per (player, market), higher line on a tie. One row per platform per run. | now | every prophecize run |
| 2 | `Archive.merge_player_books` | `archive.py:1362` (def 1319); called from `moneylines._archive_event_props` (`moneylines.py:912`, line computed at 911 as `np.median(entry["Lines"])`) | The median of the sportsbooks' main lines in one Odds API event response. One row per poll. It can be a value no book posts (45.5 and 46.5 stage 46.0). | now, or the snapshot time on a backfill | confer (`_store_sport_props`, `moneylines.py:640`), close-lines (`_close_lines_pass`, 1082), fixture replay (`_get_props_from_fixtures`, 981), historical backfill (`scripts/backfill_historical_odds.py`: the feature layer is stamped at its snapshot hour, at or before 12:00 UTC; the close layer at 23:00) |
| 3 | `scripts/merge_archives.py:205` | `SELECT … FROM lines UNION SELECT … FROM src.lines` | Whatever the source archive held | preserved | manual |
| 4 | `scripts/add_observed_at_to_archive.py:54–113` | one-time migration; rewrote the table with an `observed_at` (game-date midnight for legacy rows) | n/a | n/a | done |
| 5 | `clean_archive` | `archive.py:279, 281` | deletes old rows and `' + '` / `' vs. '` entities | n/a | maintenance |

`merge_player_books` also writes each book's own `(line, under_prob)` to `odds` through
`book_quotes` (`archive.py:1348–1358`). That per-book line is what the recommended `get_line`
reads.

Row-level attribution of the 15.31M `lines` rows in the five leagues (method in Appendix A):
DFS-written 7,248,468 (47%); legacy midnight-stamped 4,390,501 (29%, origin unrecoverable);
sportsbook backfill 3,072,619 (20%); sportsbook live 561,820 (3.7%); unmatched 34,644 (0.2%).

### 1.3 Readers

Every read of a line, R1 to R9. "Missing" is what the reader gets when there is no line.

| # | Reader | Callers | What the value is used for | Train / serve | `at=` | Missing |
|---|---|---|---|---|---|---|
| R1 | `Archive.get_line` (`archive.py:822`, SQL at 834) | `prediction/cli.py:351` (in `main`) | `Consensus Line` column on the board (`current_offers.parquet`, the dashboard marker at `dashboard/components/deep_dive_tabs.py:334`) and in history (`history_schema.py:64`); the reference the `Alt Line` flag is judged against (`_stamp_alt_line`, `cli.py:178`, applied at 414) | serve | no | `0.0` (int `0` on a bad date). The code's NaN guard never fires, so a miss is judged against 0. |
| R2 | same | `Stats._submarket_ev` (`stats/base.py:2467`) | Component subline for the legacy combo scalar: the pivot of the family conversion (`_convert_to_market_dist`, 2458) and the default line in `_fantasy_default_contribution` (2480) | train only. `check_combo_markets` has two callers: `resolve_player_market_odds` (`base.py:2308`, resolver rung 4) and the research script `scripts/backtest_combo_quotes.py:105`. Nothing in the prediction path calls it. | **no**, although training | `0`: `_fantasy_default_contribution` swaps in the player's last-10 median + 0.5; the conversion would compute garbage at a 0 pivot |
| R3 | same | `StatsMLB._mlb_hits_proportional_ev` (`stats/mlb.py:1409`) | Pivot to convert the hits EV for singles/doubles/triples/home runs | train only | no | returns 0 when `subline == 0` |
| R4 | same | `StatsMLB._check_mlb_fantasy` (`stats/mlb.py:1431`) | Component subline; for `pitcher win` it prices `1 − get_odds(subline, v)` directly | train only | no | conversion at a 0 pivot |
| R5 | same | `Archive.get_closing_line` (`archive.py:950`) | **No caller anywhere** in `src/`, `scripts/` or `tests/` | orphan | n/a | n/a |
| R6 | `Archive._observed_lines` (`archive.py:612`, SQL at 624) → `get_training_quote_inputs` (565) | `Stats.resolve_player_market_odds` (`base.py:2277`, `at=target_at`); `Stats.combo_quote` (`base.py:2636, 2642`, `at=at`); `prediction/book_quotes.servable_fallback_quotes` (143, no `at`); `scripts/backtest_combo_quotes.py:86`; a research script under `data/research/` | (a) `legacy_line`: the line the resolver prices at on rungs 2–5 (`resolve_training_quote`, `training_quotes.py:403–410`); (b) the anchor for `pickem_quote` (`training_quotes.py:253`), the stand-in 50/50 quote for a fantasy line nothing priced; (c) in `combo_quote` training mode, the combo market's own line (`base.py:2646`) | both | yes in training, no in serving | `0.0` → the resolver falls to `fallback_line` (Avg10 in training, the offered line in serving) |
| R7 | `Archive.get_ev_line_inputs` (`archive.py:637`, lines SQL at 675) | `StatsNHL._prepare_combo_archive_cache` (`stats/nhl.py:948`), `snapshot_only_rebuild` mode only | Batch form of R2 for NHL | train only | no | `0.0` |
| R8 | `Archive.to_pandas` (`archive.py:843`, lines SQL at 886) | `training/calibration.fit_book_weights` (208) | A `Line` column that `fit_book_weights` drops at 219–220. **Dead output.** | train | n/a | `0.0` |
| R9 | `Archive.get_line_history` (984, SQL at 1006) → `get_movement` (1099) | `clv._row_movement` (`clv.py:244`) | `open_line`, `close_line`, `n_moves` for the `frac_lines_moved_toward_model` diagnostic (`clv.py:313`) | evaluation | `until=` | NaN-filled dict |

Serving-side fact worth stating once: `legacy_line` never reaches a served probability. Serving
keeps only `book_direct` quotes with a real sportsbook, or component sums
(`_has_serving_support`, `book_quotes.py:89`), and neither uses `legacy_line`. So R1 and R6 are
display, evaluation and training only. The served book leg is reached by the cohort vote instead
(section 1.5, item a).

### 1.4 Raw SQL on `lines` outside those readers

| Site | Use | Affected by this change? |
|---|---|---|
| `scripts/sweep_runaway_odds.py:96`, `scripts/delete_corrupt_seed.py:63` | `SELECT MAX(l.line) FROM lines l` as the slot's line bound when deciding an EV is a runaway | No. Needs the log to keep holding every line, DFS included. |
| `scripts/migrate_archive_shapefree.py:59–66, 143–150` | One-time WS1 backfill: wrote the mixed consensus into `odds.line` for legacy rows | Already run. It is why legacy `odds.line` is not a per-book line (Appendix A). |
| `scripts/merge_archives.py:53, 205, 211` | Union merge | No |
| `scripts/add_observed_at_to_archive.py:54–113` | One-time migration | No |
| `scripts/quarantine_sleeper_wnba.py` | Nulls `ev`/`under_prob` on poisoned Sleeper WNBA rows; "line and lines table kept" | No |
| `archive.py:279, 281` | `clean_archive` deletes | No |
| Tests | `tests/test_archive_history.py` (83, 143–173), `tests/golden/test_archive_shapefree_storage.py` (133, 152, 310), `tests/golden/test_archive_wal_recovery.py:46`, `tests/test_merge_archives.py`, `tests/test_sweep_runaway_odds.py`, `tests/test_delete_corrupt_seed.py`, `tests/test_quarantine_sleeper_wnba.py` | Section 5 |

All scripts are under `src/sportstradamus/scripts/`. The repo-root `scripts/` directory has no
reader or writer of `lines`.

### 1.5 Other places a DFS line reaches something called consensus

The brief named `get_line`. The trace found eight more paths.

| | Path | Effect today | In the recommended change? |
|---|---|---|---|
| a | `_direct_line_cohort` (`training_quotes.py:273`): DFS rows vote on the modal line; `sportsbook_cohort` is applied only afterwards, in `_authentic_quote` (324) | A DFS line can pick which line the sportsbook price is read at, or win the vote alone and turn a key with sportsbook quotes into a `PICKEM` quote with no book leg. Reaches **served** probabilities. | Yes (edit 5) |
| b | `_weighted_book_ev` (`archive.py:454`): `_drop_divergent_lines` runs before `sportsbook_cohort` | DFS rows move the median the divergence filter uses and can get a lone sportsbook dropped as "divergent". Readers: CLV closing probability, the training combo scalar. Nothing served. | Yes (edit 6) |
| c | `fit_book_weights` (`calibration.py:209`) drops only `pinnacle` | DFS platforms are fitted as books; their columns move the sportsbook weights | Yes (edit 7) |
| d | `to_pandas` `Line` column | Mixed median, unused | Yes (edit 8, deleted) |
| e | `Stats._book_mean_shift` (`base.py:2538`) | On fantasy markets the training-side component-sum mean is moved halfway to the market's own quoted mean, "admitted from any book including the DFS platforms". Deliberate and documented in the docstring. Training `DERIVED` rows only; fantasy cells do not take the combo pass at serve time (`book_quotes.py:129–132`). | **No. Owner decision** (section 7) |
| f | `get_movement` / `get_line_history` | `open_line`, `close_line` and `n_moves` come from the mixed log, so a key that alternates between a DFS rung and a sportsbook median counts a "move" on every poll. Diagnostic only. | **No. Owner decision** |
| g | `Archive.get_ev` on a DFS-only key | `sportsbook_cohort` falls back to the DFS rows, so the "consensus EV" is the DFS EV. Readers: CLV closing probability on DFS-only offers, the training combo scalar. | No. Consistent with "a DFS line may be evaluated against". Flagged. |
| h | Legacy `odds.line` | The WS1 migration baked the mixed consensus into every pre-live sportsbook row. Unrecoverable. | No |

---

## 2. What changes for the models

### 2.1 Features: none

No model feature reads the consensus line.

- The feature matrix is `M.reindex(columns=stat_data.get_stat_columns(market))`
  (`training/pipeline.py:1097`). `Line`, `Odds`, `EV`, `Archived` and the `Quote*` provenance
  columns are not in `feature_filter.json` or in any profile column list.
- The team features that do come from the archive (`Moneyline`, `Total`, `OppTotal`, `Spread`,
  `GameTotal`) read `get_moneyline` / `get_total`. Those markets have no DFS rows.
- No target normalization uses a book column. The slugs in `stat_meta.json` are `none`,
  `centered_additive_mean10`, `centered_additive_eb_meanyr_k10`, `ratio_meanyr` and
  `ratio_projvol`: all player or volume means.

So an already-trained model is fed an identical feature vector at serve time before and after
the change. **No retrain is forced.**

### 2.2 Book columns and labels in the training matrix

These are produced by `Stats.resolve_player_market_odds` (`base.py:2258`) and stored in the
cached matrix. They change only when a cell's quote block is next re-resolved
(`scripts/inject_backfilled_odds.py --all-cached`) or its matrix rebuilt; `meditate` only appends
new game dates.

| Column | Consumers | Changes through |
|---|---|---|
| `Line` | labels `Result >= Line` (`pipeline.py` ~3208, 4302, 5006): gates, posthoc calibrator fits, Brier; `fit_book_shape` conditioner (`pipeline.py` ~4190); `_balance_lines` and trim (`training/data.py:158–233`) | cohort vote (edit 5) on 2026 mixed rows |
| `Odds`, `EV` | book Brier and log loss, blend-weight fit, `mean_ev_diff` diagnostics (`pipeline.py` ~3484–3769) | cohort vote, on the row's own quote and on the component quotes of a `combo_sum` row; component sublines for `combo_ev_inversion` rows |
| `Archived`, `QuoteAuthenticity`, `QuoteBookCount` | row admission (`base.py` ~2407: MLB keeps `Archived` rows only), Gate 1 and blend-fit sample (authentic rows only) | cohort vote turns some `pickem` rows into `authentic` |

Sizes on a re-resolve:

- **Cohort vote.** Only 2026 rows: no DFS `odds` row exists before mid-March 2026. Share of 2026
  sportsbook keys at the training cutoff whose quote changes: MLB 1.22% (3,281 keys), NFL 3.04%
  (173), WNBA 4.73% (669), NHL 0.14% (118), NBA 0%. Of those, 3,005 are `pickem` today and become
  `authentic`, which enlarges the Gate 1 and blend-fit sample.
- **Resolver `legacy_line`.** Zero rows. Among the 1,845,805 cached-matrix rows since 2023-09,
  8,752 were resolved on rungs 2–5 off an observed line. Every one of them has a line of record
  and no sportsbook line, so "sportsbook consensus, else line of record" returns the same number
  as today.
- **Legacy combo scalar sublines.** `combo_ev_inversion` rows (`DERIVED`, outside Gate 1 and the
  blend fit; MLB 16,057, NBA 35,316, NHL 19,979, WNBA 16,949 rows) read component sublines through
  R2–R4. On a 3,000-row sample of the simple-combo cells, a component subline moves on 5–30% of
  rows by cell, and the row's EV changes on 9.4% (only where the moved leg needs a family
  conversion), by a median 2.6% of the mean (p90 6.7%). Before 2026 this is the change of
  statistic (latest per book, not distinct over time), not DFS removal.
- **`fit_book_weights`.** Takes effect at the next refit and `sync_to_prod.sh`, not on deploy.

A matrix whose quote block is re-resolved gets a new file hash (`pipeline.py:4748`) like any
refresh, so matrix-scoped ship verdicts restamp as they would for any backfill.

### 2.3 Served today vs display and evaluation

| Changes what a trained model's pipeline serves | Display or evaluation only |
|---|---|
| Edit 5 (cohort vote): the book leg of the fused mean and the book-fallback probability, on about 1.5% of mixed keys, plus the simple-combo sums built on those components (about 200 more keys in five weeks, section 3e) | `Consensus Line` column and dashboard marker |
| Nothing else | `Alt Line` flag (receipts main-vs-alt split, `realized.py:171, 258`; `dashboard/surfaces/receipts.py:174`) |
| | CLV closing probability (edit 6, 0.2% of mixed keys) |
| | `Market Prob`, `Books EV`, `Quote Source`, `Quote Books` on the 1.5% of keys in the left column |

No served cell uses `prob_recal_book_citl`. The posthoc slugs across the 78 served cells are:
none 39, `prob_recal_platt` 14, `prob_recal_isotonic` 8, `isotonic_mean` 8, `cdf_recal_isotonic`
6, `roe_mean` 3. So the book quote reaches a served probability only through the fused mean
(scaled by 1 − model weight) or the book-fallback path (full size).

---

## 3. Measurements

All from `archive/archive.duckdb` opened with `duckdb.connect(path, read_only=True)`. Scripts and
intermediate tables are listed in Appendix B.

### 3a. Sportsbook `odds` rows with a non-null `line`

Player-prop rows from non-DFS books. Share with `line IS NOT NULL`:

| Year | MLB | NBA | NFL | NHL | WNBA |
|---|---|---|---|---|---|
| 2022 | 70.1% | 88.5% | 71.7% | 86.9% | n/a |
| 2023 | 84.1% | 91.3% | 90.3% | 89.9% | 100.0% |
| 2024 | 93.2% | 97.3% | 98.3% | 95.5% | 99.8% |
| 2025 | 99.96% | 99.98% | 99.9% | 97.9% | 99.96% |
| 2026 | 100.0% | 99.85% | 99.99% | 100.0% | 99.7% |
| All | 91.9% | 96.6% | 96.6% | 93.5% | 99.8% |

Months under 95%:

- MLB: 2022-05 37.0%; 2022-06 to 2022-11 69.8–73.8%; 2023-03 33.3%; 2023-04 35.8%; 2023-05 to
  2023-11 82.5–90.6%; 2024-02 0.0% (1,736 rows); 2024-03 89.5%; 2024-04 to 2024-10 91.9–94.2%.
- NBA: 2022-10 to 2023-04 86.9–89.8%; 2023-05 and 2023-06 94.1%; 2023-11 to 2024-02 91.0–94.2%.
- NFL: 2022-09 to 2023-01 70.3–72.6%; 2023-02 57.1%; 2023-11 88.8%; 2023-12 81.4%; 2024-01 79.9%;
  2024-02 69.8%.
- NHL: 2022-10 to 2023-06 85.8–87.0%; 2023-12 to 2024-02 94.4–94.9%; 2024-12 to 2025-03
  93.7–94.0%.
- WNBA: never under 98.0%.

Every month from 2025-04 on is at or above 98.0% in every league. The full month table is
`m1_odds_line_share.csv`.

Two qualifications:

- **A non-null line is not always a per-book line.** Rows stamped at game-date midnight (the
  pre-live "legacy" rows) carry the mixed consensus the WS1 migration wrote. They never disagree
  across books on a key. Share of sportsbook rows with a real (non-legacy) stamp: 2023 MLB 21.6%,
  NBA 15.8%, NFL 51.1%, NHL 16.6%; 2024 MLB 26.8%, NBA 45.5%, NFL 74.3%, NHL 46.4%; 2025 MLB
  99.5%, NBA 51.0%, NFL 77.9%, NHL 77.3%, WNBA 51.8%; 2026 MLB 100%, NBA 59.3%, NFL 96.8%, NHL
  100%, WNBA 100%.
- **The missing lines are concentrated.** They sit in MLB `triples` (49,284 keys), `stolen bases`
  (12,431) and `pitcher win` (4,404) from 2023-09 to 2025-03: sportsbook `ev` rows with NULL
  `line` and NULL `under_prob`, whose 0.5 line survives only in `lines`. This is why the readers
  that pivot on a line need the line-of-record fallback (section 4.3).

"Latest row has a NULL line while an earlier row has one" occurs on 235 of 11.8M sportsbook
book-keys since 2023-09, so reading the latest row needs no "latest non-null" variant.

### 3b. Where do the lines on a key come from?

Key level, all time, five leagues, 5,142,347 keys with a line today:

| Origin of the key's `lines` rows | Keys | Share |
|---|---|---|
| Sportsbook only (attributed) | 913,291 | 17.76% |
| Sportsbook plus legacy or unknown rows | 1,403,773 | 27.30% |
| Legacy only, sportsbook `odds` on the key (origin unknown) | 1,823,837 | 35.47% |
| Legacy only, no `odds` row at all (a bare line; DFS by elimination) | 513,837 | 9.99% |
| Legacy only, DFS `odds` on the key | 65,059 | 1.27% |
| Both (attributed) | 261,588 | 5.09% |
| DFS only (attributed) | 159,865 | 3.11% |
| Other | 1,097 | 0.02% |

The bare-line row (9.99%) is NBA 24.6%, NFL 22.0%, NHL 14.0%, WNBA 13.0%, MLB 0.4% of that
league's keys.

Where attribution is exact (game date from 2026-05-10; 522,222 keys):

| | All | MLB | NBA | NFL | NHL | WNBA |
|---|---|---|---|---|---|---|
| Both | 49.68% | 52.07 | 44.21 | 15.43 | 43.21 | 52.00 |
| DFS only | 30.50% | 26.01 | 41.05 | 82.64 | 50.95 | 33.52 |
| Sportsbook only | 19.56% | the remainder in each league | | | | |

The serving window (130,681 keys): both 66,101 (50.58%: MLB 57,456,
NFL 3,476, NHL 2,665, WNBA 2,504), which reproduces the brief's 66,101 exactly; DFS only 52,015
(39.80%: MLB 27,911, NFL 18,130, NHL 4,299, WNBA 1,675); sportsbook only 12,565 (9.62%).

**How reliable the inference is.**

- From 2026-05-09 (live `observed_at` stamps): exact. Each `lines` row is matched to the `odds`
  row staged immediately before it on the same key; both writers stage the book row and the line
  row in the same call. Row-level attribution agrees with key-level book presence on 100% of the
  522,222 keys. 0.2% of rows found no partner.
- 2026-03-16 to 2026-05-08: DFS `odds` rows exist but stamps are legacy, so only key-level book
  presence is available.
- Before 2026-03-16: DFS platforms wrote to `lines` only. A legacy line cannot be attributed
  except by elimination (a key with a line and no `odds` row must be DFS). A legacy line on a key
  that also has sportsbook rows may or may not include a DFS line; the old klepto archive mixed
  them and kept no source.

### 3c. Sportsbook-only median vs today's `get_line`

Three candidates were measured against today's value:

- **S1**: today's statistic (distinct lines over time) with the DFS-written rows removed.
- **S3**: the recommended statistic, the median of each sportsbook's latest `odds.line`.
- **S3d**: S3 over distinct lines rather than per book.

**Serving window, mixed keys, all markets.** Share of keys where the candidate differs from
today's `get_line`:

| | Keys | S3 differs | mean gap | median | S1 differs | mean gap |
|---|---|---|---|---|---|---|
| **Decision time** | | | | | | |
| MLB | 39,860 | 8.15% | 0.56 | 0.5 | 7.70% | 0.56 |
| NFL | 3,325 | 45.11% | 1.21 | 0.5 | 38.83% | 1.00 |
| NHL | 2,584 | 3.68% | 0.51 | 0.5 | 2.24% | 0.53 |
| WNBA | 2,461 | 60.50% | 0.75 | 0.5 | 47.14% | 0.74 |
| All | 48,230 | **13.13%** | 0.76 | 0.5 | **11.57%** | 0.70 |
| **Final state** | | | | | | |
| MLB | 69,156 | 7.93% | 0.57 | 0.5 | 6.22% | 0.56 |
| NFL | 3,903 | 41.40% | 2.44 | 0.5 | 34.00% | 1.18 |
| NHL | 2,898 | 3.62% | 0.51 | 0.5 | 2.38% | 0.51 |
| WNBA | 2,708 | 61.78% | 1.12 | 0.5 | 43.87% | 0.76 |
| All | 78,665 | 11.28% | 1.01 | 0.5 | 8.75% | 0.72 |

The brief's "about 12%, NFL 38, WNBA 48, NHL 3, by 0.7" is S1 at decision time. S3 moves more
keys than S1 because it also drops the "every distinct line ever seen" behavior.

Size of the gap when S3 differs (final state): MLB 4,838 keys at 0.5 and 548 at 1.0; NFL 851 at
0.5, 239 at 1.0 and a tail to 12.5 (yardage); WNBA 987 at 0.5, 349 at 1.0 and a tail to 12.5.

Cells that move most (final state, share of mixed keys, mean gap):

- MLB: pitcher strikeouts 64.7% (0.75), hits allowed 64.0% (0.79), pitching outs 30.8%, total
  bases 24.0%, hits+runs+rbi 20.7%, runs allowed 16.2%, hits 12.9%.
- NFL: receiving yards 87.6% (3.42), rushing yards 80.1% (3.10), carries 72.3%, receptions 52.0%,
  tds 5.0%.
- WNBA: PA 75.4% (1.65), PR 75.1% (1.96), PRA 65.4%, PTS 64.5%, RA 64.5%, AST 51.3%, REB 50.6%,
  FG3M 49.4%.
- NHL: shots 8.0%, points 4.7%.

**The board as it stands now.** `current_offers.parquet`, 4,544 rows over three game dates,
read against the archive's final state:

| | Rows | `Consensus Line` changes | mean gap | becomes NaN | whole-number consensus today → after |
|---|---|---|---|---|---|
| MLB | 1,316 | 15.0% | 0.50 | 18.6% | 15.0% → 0.0% |
| NFL | 1,987 | 56.2% | 1.19 | 4.0% | 41.8% → 9.5% |
| NHL | 927 | 2.5% | 0.50 | 14.9% | 0.4% → 0.2% |
| WNBA | 314 | 59.2% | 0.74 | 0.6% | 36.0% → 7.0% |
| All | 4,544 | 33.5% | 1.03 | 10.2% | 25.2% → 4.7% |

A whole-number consensus is a line no book posts (player props are quoted at half points). It
comes from the median of an even number of distinct lines.

The "changes" column compares the new value with today's rule re-read on the same archive state
under the archive's market name. Against the values the last run actually stored on the board,
the column changes on 43.8% of rows (MLB 14.4%, NFL 55.9%, NHL 59.7%, WNBA 43.6%). NHL is higher
there because 69.5% of NHL rows hold 0.0 today (section 8, D2).

**Training cutoff, modeled markets, by season.** Share of keys with a sportsbook line where S3
differs from today's value read at the same cutoff:

| League | Season | Keys | S3 differs | mean gap | S3 differs among keys with a DFS-written `lines` row |
|---|---|---|---|---|---|
| MLB | 2024 | 523,389 | 2.39% | 0.63 | none exist |
| MLB | 2025 | 461,909 | 0.04% | 0.50 | none exist |
| MLB | 2026 | 268,960 | 3.58% | 0.60 | 7.47% of 38,512 |
| NBA | 2023-24 | 170,808 | 20.81% | 0.61 | none exist |
| NBA | 2024-25 | 172,256 | 0.00% | n/a | none exist |
| NBA | 2025-26 | 152,744 | 0.47% | 0.85 | 25.44% of 2,476 |
| NFL | 2024-25 | 24,544 | 0.00% | n/a | none exist |
| NFL | 2025-26 | 22,857 | 0.86% | 1.10 | none exist |
| NFL | 2026-27 (to date) | 3,608 | 38.69% | 1.25 | 43.62% of 3,198 |
| NHL | 2023-24 | 156,738 | 3.64% | 0.53 | none exist |
| NHL | 2024-25 | 147,741 | 0.16% | 0.50 | none exist |
| NHL | 2025-26 | 145,211 | 0.63% | 0.57 | 2.68% of 3,137 |
| WNBA | 2024 | 9,704 | 0.00% | n/a | none exist |
| WNBA | 2025 | 11,922 | 0.00% | n/a | none exist |
| WNBA | 2026 | 14,137 | 48.00% | 0.98 | 55.64% of 11,161 |

Reading it:

- Before 2026 every difference is the change of statistic, not DFS removal. NBA 2023-24 is the
  large one (20.8%): legacy rows where the lines log holds several distinct values and the
  migrated `odds.line` holds one.
- The DFS effect proper is the last column: 7% of MLB, 25% of NBA, 44% of NFL and 56% of WNBA
  keys that carry a DFS line. That column counts keys whose lines log holds a row attributed to
  a DFS write at the cutoff, which is possible from 2026-05-09 only. It is narrower than the
  mixed-key count in 3e, which goes by book presence in `odds` and starts 2026-03-16.
- Read at the final state instead of the cutoff, the same seasons differ more, from the statistic
  alone, because close-layer rows add distinct lines: NBA 2024-25 20.55%, 2025-26 27.34%; NFL
  2024-25 28.64%, 2025-26 30.45%; WNBA 2025 22.72%; MLB 2025 2.20%, 2026 7.06%. Readers R2–R4
  read at the final state today (no `at=`).

**Which statistic agrees with the line the resolver already prices at?** On decision-time keys
with a sportsbook quote, share where the candidate equals the resolver's sportsbook modal line:

| | MLB | NFL | NHL | WNBA |
|---|---|---|---|---|
| S3, per-book latest (recommended) | 99.05% | 91.79% | 99.57% | 88.22% |
| S3d, distinct lines | 92.25% | 70.74% | 98.37% | 66.52% |
| Today's `get_line` | 91.48% | 54.17% | 96.05% | 33.08% |

### 3d. Keys with no consensus line under a sportsbook-only rule

A key "loses its line" when it has a line today and no sportsbook `odds.line`.

All time, all markets: MLB 392,034 of 2,526,385 (15.5%); NBA 413,093 of 1,172,535 (35.2%); NFL
85,299 of 178,027 (47.9%); NHL 283,809 of 1,197,057 (23.7%); WNBA 20,090 of 68,343 (29.4%).
Total 1,194,325 of 5,142,347 (23.2%).

All time, modeled markets only, split by reason:

| | Lose the line | share | DFS `odds` only | bare line, no `odds` row | sportsbook row with NULL line |
|---|---|---|---|---|---|
| MLB | 198,639 | 8.7% | 111,909 | 1,409 | 85,321 |
| NBA | 268,428 | 26.1% | 6,730 | 261,672 | 26 |
| NFL | 38,939 | 29.6% | 3,012 | 35,898 | 29 |
| NHL | 177,487 | 17.2% | 7,180 | 161,829 | 8,478 |
| WNBA | 8,201 | 14.5% | 4,471 | 3,721 | 9 |

Serving window, modeled markets:

| | Final state | Decision time |
|---|---|---|
| MLB | 21,839 of 89,999 (24.3%) | 38,162 of 78,022 (48.9%) |
| NFL | 1,729 of 5,632 (30.7%) | 1,846 of 5,171 (35.7%) |
| NHL | 1,407 of 4,305 (32.7%) | 1,371 of 3,955 (34.7%) |
| WNBA | 554 of 3,262 (17.0%) | 509 of 2,970 (17.1%) |
| All | 25,529 of 103,198 (24.7%) | 41,888 of 90,118 (46.5%) |

All markets, final state: 51,235 of 129,900 (39.4%: MLB 28.5%, NFL 82.1%, NHL 59.0%, WNBA 37.2%).
The decision-time figure is higher because sportsbook rows often arrive after the last DFS poll
(the close-lines pass). Half of MLB modeled player-market-days had no sportsbook quote when
prophecize scored them.

Training cutoff, modeled markets, keys that lose the line by season: MLB 2024 13,331 (2.48%: 11,926
NULL-line sportsbook rows, 1,405 bare lines), 2025 9, 2026 63,043 (18.99%, DFS only); NBA 2023-24
72,446 (29.78%), 2024-25 18,259 (9.58%), 2025-26 11,160 (6.81%); NFL 2023-24 9,208 (22.81%),
2024-25 325 (1.31%), 2025-26 414 (1.78%), 2026-27 2,120 (37.01%); NHL 2023-24 21,976 (12.30%),
2024-25 171 (0.12%), 2025-26 7,517 (4.92%), 2026-27 563 (27.22%); WNBA 2024 2,214 (18.58%), 2025
12, 2026 1,978 (12.27%).

### 3e. The cohort vote (edit 5)

Mixed keys with at least one directly priced row, modeled markets. "Pickem today" means the DFS
line wins the vote alone, so today's quote has no sportsbook in it. "Minority line" means the
vote lands on a line where the DFS rows outnumber the sportsbooks' own most-quoted line.

| Cutoff | League | Mixed keys | Pickem today → sportsbook quote | Minority line → sportsbook modal line | Share changed |
|---|---|---|---|---|---|
| Decision time | MLB | 39,860 | 111 | 198 | 0.78% |
| | NFL | 3,325 | 8 | 211 | 6.59% |
| | NHL | 2,584 | 2 | 9 | 0.43% |
| | WNBA | 2,461 | 38 | 145 | 7.44% |
| | All | 48,230 | 159 | 563 | **1.50%** |
| Current board | MLB / NFL / NHL / WNBA | 578 / 1,005 / 886 / 173 | 2 / 3 / 1 / 0 | 0 / 41 / 3 / 6 | 56 of 2,642 (2.1%) |
| Training cutoff (2026 rows) | MLB | 116,058 | 2,739 | 542 | 1.22% of sportsbook keys |
| | NFL | 3,198 | 12 | 161 | 3.04% |
| | NHL | 27,713 | 109 | 9 | 0.14% |
| | WNBA | 10,954 | 145 | 524 | 4.73% |
| | NBA | 22,584 | 0 | 0 | 0% |

The line gap between the two cohorts is 1.0 at the median (0.5 for the pickem class at the
training cutoff).

Effect on the served book leg, 722 decision-time keys. The measure is the under-probability the
quote implies at the DFS platform's own main line, today's quote against the sportsbook-first
quote:

| | Keys | median | p90 | max |
|---|---|---|---|---|
| **Book leg served today, cohort changes** | 563 | | | |
| MLB | 198 | 0.048 | 0.135 | 0.182 |
| NFL | 211 | 0.019 | 0.038 | 0.219 |
| NHL | 9 | 0.022 | 0.043 | 0.060 |
| WNBA | 145 | 0.022 | 0.049 | 0.167 |
| **No book leg served today, one starts** | 159 | | | |
| MLB | 111 | 0.065 | 0.173 | 0.341 |
| WNBA | 38 | 0.049 | 0.121 | 0.222 |
| NFL | 8 | 0.029 | 0.095 | 0.139 |

In the second block nothing is served from the book today, so the figure is how far the
sportsbook quote that starts serving sits from the platform's own price.

Busiest cells: NFL receiving yards 105 keys (median 0.021), MLB total bases 88 (0.043), MLB hits
71 (0.029), MLB hits+runs+rbi 63 (0.131), NFL rushing yards 56 (0.019), WNBA PR 51 (0.023).

Knock-on through component sums. A served simple-combo quote (`combo_props`: MLB hits+runs+rbi,
WNBA PRA and the like) takes its shape from the component sum, or its whole price when the combo
has no direct quote (`servable_fallback_quotes`, `book_quotes.py:157–179`), and the components are
resolved by the same vote. Among the simple-combo keys offered in the serving window, at least one
component's quote changes on 218: MLB 73 of 7,730, NFL 81 of 913 (`yards` 47, `qb yards` 34),
WNBA 64 of 725. On 200 of those the combo's own quote is unchanged, so they come on top of the
722. The size of the move on a sum was not measured; the component's own shift is the one tabled
above, entering a sum of two or three.

The served probability moves by about (1 − model weight) × that figure in a fused cell, and by
the full figure in a book-fallback cell. Model weight of served cells: median 0.851 overall; MLB
0.653, NBA 0.754, NHL 0.848, NFL 1.000, WNBA 1.000; 47% of served cells are at 0.9 or above.

Accuracy on settled keys whose quote changes. The measure is the Brier score of the quote's
under-probability at the DFS platform's main line against the settled result, pushes excluded:

| | n | today | after | difference |
|---|---|---|---|---|
| Sportsbook already in today's cohort, decision time | 457 | 0.2441 | 0.2467 | +0.0026, 95% CI [−0.0020, +0.0076] |
| Same, training cutoff | 446 | 0.2458 | 0.2435 | −0.0023, 95% CI [−0.0049, +0.0002] |
| DFS-only cohort today (book evidence discarded), decision time | 128 | 0.2438 | 0.2219 | better |
| Same, training cutoff | 121 | 0.2315 | 0.2197 | better |

So the change is accuracy-neutral where it reshuffles sportsbooks and an improvement where it
restores discarded sportsbook evidence. The case for it is the owner's rule, not a measured edge.

### 3f. Divergence filter order (edit 6)

242,286 mixed keys, 2026-05-10 to 2026-10-03. Narrowing to sportsbooks before the divergence
filter changes `_weighted_book_ev` on 480 keys (0.2%): WNBA 299 (1.87% of its mixed keys), MLB 96
(0.045%), NFL 84 (2.42%), NBA 1, NHL 0. When it changes, the EV moves by 20–40% of the mean. On
28 of those keys (NFL 13, WNBA 9, MLB 6) today's surviving cohort contains no sportsbook at all:
the DFS EV is returned as the consensus EV although sportsbooks quoted the key.

### 3g. Alt Line (edit 9)

262,547 history rows in the window; the stored flag is True on 46.3%. An emulation of today's
rule from the archive agrees with the stored flag on 97.6% of rows. 39.0% of rows had no
sportsbook line at decision time (MLB 43.5%, NFL 20.8%, NHL 12.2%, WNBA 1.9%), and 47.3% of those
carry the flag today.

| Rule for a row with no sportsbook line | Flag share after | Rows that flip | True → False | False → True |
|---|---|---|---|---|
| **A1**: no reference, flag False | 29.0% | 21.8% | 19.5% | 2.3% |
| **A2** (recommended): judge against the line of record | 46.7% | 4.4% | 2.0% | 2.4% |

A2 flips by league: MLB 3.4%, NFL 11.5%, NHL 3.7%, WNBA 8.0%. Among rows that do have a sportsbook
line, 5.5% flip under either rule. On the current board: A2 flips 5.1% of rows, A1 6.6%.

Both rows of the table assume the board lookup uses the archive's market name (section 8, D2).

### 3h. Book weights (edit 7)

`book_weights.json` carries a DFS weight in NBA 15 of 23 cells (median DFS share of the cell's
weight mass 0.26), NFL 14 of 28 (0.44), MLB 18 of 29 (0.01), NHL 11 of 18 (0.20), WNBA 12 of 23
(0.21). Largest: MLB runs allowed 90%, NFL yards 78%, NFL rushing yards 74%, NBA REB 64%.

Refit experiment on eight well-conditioned cells (fit on the earlier 60% of DFS-era dates with and
without the DFS columns; test on later rows with at least two sportsbooks):

| Cell | Largest shift in a sportsbook's normalized weight | Out-of-sample NLL, without − with |
|---|---|---|
| MLB total bases | 0.378 | −0.0058 |
| NBA REB | 0.356 | +0.0010 |
| WNBA PTS | 0.218 | +0.0033 |
| NBA PTS | 0.185 | −0.0001 |
| MLB pitcher strikeouts | 0.183 | +0.0033 |
| NHL points | 0.104 | +0.0000 |
| MLB hits+runs+rbi | 0.044 | +0.0002 |
| NHL shots | 0.023 | −0.0000 |

Equal weights land within 0.003–0.02 NLL of either fit. The fit's objective is ill-conditioned on
NFL and WNBA REB SkewNormal cells (NLL 1e9–1e12 on near-zero EVs), so those were left out.

### 3i. Latency

Per key on the current board's 1,955 distinct keys: today's `lines` query 1.1 ms; the per-book
query the new `get_line` runs 2.4 ms as bare SQL, and 2.3 ms and 3.0 ms on two runs through the
patched `Archive.get_line`. A colder 600-key sample measured 6.2 ms. The reference fallback for
the 255 keys with no sportsbook line costs 1.4 s (5.3 ms each: the consensus is read again, then
the lines log).

That is about +4 to +5 s per prophecize run on this board (+2.5 to +3.6 s for the consensus pass,
+1.4 s for the fallback), and up to about +13 s on a cold cache.

---

## 4. Recommended implementation

### 4.1 The rule

- **Consensus line** = `_consensus_line` over each sportsbook's latest posted line at or before
  `at`. Sportsbooks only. `0.0` when none posts a line.
- **Reference line** = the consensus line, else the line of record. It is the line an entry is
  graded at. It is never called consensus and never feeds one.
- A DFS line reaches a reader only (i) as the line of record of a key no sportsbook lines, and
  (ii) as the DFS platform's own quote in the `pickem` evaluation cohort.

The statistic already exists in the code: `scripts/tail_pricing._RUNG_SQL` (lines 85–113)
computes `median(arg_max(o.line, o.observed_at))` per sportsbook with
`NOT list_contains($dfs, o.book)`, and the tail scorecard reads a NULL from it as band `none`.
This change makes `Archive.get_line` agree with the evaluation tool.

### 4.2 The edits

Edits 1–8 below exist as a working in-memory patch, `patchdir/consensus_patch.py` in the scratch
directory (the same code, applied by text substitution at import). It was run against the real
archive read-only: the patched `get_line` equals the measured S3 column on 2,400 of 2,400 sampled
keys (final state and both `at=` eras), and both batch readers equal the scalar
`get_reference_line` on 1,600 of 1,600. Edit 9 was run separately, as written, on the current
board (the result is at the end of edit 9). What no run covered: the two delegations in 9a (two
lines each) and the placement of 9b and 9c inside `main`, which only a prophecize run or the
integration suite exercises.

**Edit 1. `helpers/archive.py`: `get_line` and a module-level reducer.**

Import `DFS_PLATFORM_BOOKS` beside `sportsbook_cohort` (the import block at 53–57). Add next to
`_consensus_line` (190):

```python
def _sportsbook_line(quotes: Iterable[tuple[str, float | None]]) -> float:
    """Consensus over the lines the sportsbooks post; ``0.0`` when none posts one.

    A pick'em platform's line never counts: it is the entry being graded, not
    evidence of where the market sits.
    """
    return _consensus_line(
        [line for book, line in quotes if line is not None and book not in DFS_PLATFORM_BOOKS]
    )
```

Replace the body of `get_line` (822–841):

```python
def get_line(self, league, market, date, player, *, at: datetime.datetime | None = None):
    """Sportsbook consensus line for ``player`` on ``date``: median, floored to ½.

    The median runs over each sportsbook's latest posted line at-or-before ``at``
    (``at=None``: latest available). ``0.0`` when no sportsbook has posted one. A
    pick'em platform's line is never a consensus; :meth:`get_reference_line` falls
    back to it.
    """
    quotes = self.get_training_book_quotes(league, market, date, player, at=at)
    return _sportsbook_line((quote.book, quote.line) for quote in quotes)
```

Why `get_training_book_quotes` (524) and not `_book_rows` (466): it breaks same-timestamp ties
deterministically, and it is the scalar twin of the SQL `get_training_quote_inputs` runs, so the
scalar and batch lines cannot disagree. 13,041 sportsbook book-keys (2023-10-14 to 2026-07-08)
have two rows with one timestamp and different lines; `_book_rows` picks between them
arbitrarily. It also gives that method its first caller in `src/`.

A bad date returns `0.0` (today: int `0`). Equal under `==`.

**Edit 2. `helpers/archive.py`: `get_reference_line`.**

```python
def get_reference_line(self, league, market, date, player, *, at: datetime.datetime | None = None):
    """The line an entry is graded at: the sportsbook consensus, else its line of record.

    The line of record is the median of every distinct line the archive logged for the
    entry: a pick'em platform's own main rung, or a pre-2025 sportsbook row archived
    without its line. It anchors a price that has no sportsbook line beside it. It is
    never a consensus and never feeds one.
    """
    line = self.get_line(league, market, date, player, at=at)
    d = _safe_date(date)
    if line or d is None:
        return line
    values, _ = self._observed_lines(league, market, d, [player], at).get(player, ([], None))
    return _consensus_line(values)
```

`_observed_lines` (612) stays as it is and becomes the one read of the line-of-record log. Reword
its docstring to say so.

**Edit 3. `helpers/archive.py`: the two batch readers.**

`get_training_quote_inputs`, the loop at 605–609:

```python
for entity in ordered_entities:
    values, seen_at = observed_lines.get(entity, ([], None))
    line_of_record = _consensus_line(values)
    rows = grouped[entity] or pickem_quote(market, line_of_record, seen_at)
    line = _sportsbook_line((row.book, row.line) for row in grouped[entity])
    inputs[entity] = (rows, line or line_of_record)
```

No extra SQL: the per-book rows are already fetched. `pickem_quote` keeps its anchor.

`get_ev_line_inputs` (637): delete its private `lines` query (673–684) and reuse
`_observed_lines`, which is the same statement:

```python
observed_lines = self._observed_lines(league, market, d, ordered_entities, at)
inputs = {}
for entity in ordered_entities:
    rows = ev_rows[entity]
    ev = self._weighted_book_ev(league, market, rows) if rows else float("nan")
    line = _sportsbook_line((book, book_line) for book, _, book_line in rows)
    values, _ = observed_lines.get(entity, ([], None))
    inputs[entity] = (ev, line or _consensus_line(values))
return inputs
```

That removes one of the three copies of the lines query.

One caveat. This reader's per-book query orders by `observed_at DESC` alone, as `_book_rows` does.
On a same-timestamp tie it can take a different row than the scalar read: 453 NHL sportsbook
book-keys have such a tie (NHL is this reader's only caller). The EV it returns has that
ambiguity today; the line now shares it. Copying the tie-break columns from
`get_training_quote_inputs` into its `ORDER BY` closes it, as a separate fix.

**Edit 4. Combo sublines: three call sites move to `get_reference_line`.**

- `stats/base.py:2467` (`Stats._submarket_ev`)
- `stats/mlb.py:1409` (`StatsMLB._mlb_hits_proportional_ev`)
- `stats/mlb.py:1431` (`StatsMLB._check_mlb_fantasy`)

Each is `archive.get_line(` → `archive.get_reference_line(`, nothing else. The NHL batch path
(`stats/nhl.py:948`) picks the same value up through edit 3.

**Edit 5. `helpers/training_quotes.py`: sportsbooks vote first.** Suggest its own commit.

`_direct_line_cohort` (273–285), first statement:

```python
direct = sportsbook_cohort(
    [row for row in rows if _positive(row.line) and _probability(row.under_probability)]
)
```

Then delete `cohort = sportsbook_cohort(cohort)` in `_authentic_quote` (324), which becomes a
no-op (the cohort is already all sportsbooks, or all DFS when no sportsbook priced the key), and
move its docstring paragraph (319–322) to `_direct_line_cohort`. The comment at 282–283 gains one
clause: sportsbooks only, when any priced the entry.

**Edit 6. `helpers/archive.py`: `_weighted_book_ev` (454–457).**

```python
usable = _drop_divergent_lines(
    sportsbook_cohort([row for row in rows if row[1] is not None], operator.itemgetter(0))
)
```

Update the docstring's order of events (445–453).

**Edit 7. `training/calibration.py`: `fit_book_weights` (209).**

```python
df = df[[col for col in df.columns if col != "pinnacle" and col not in DFS_PLATFORM_BOOKS]]
```

with the import from `sportstradamus.helpers.training_quotes`.

**Edit 8. `helpers/archive.py`: `to_pandas` loses its dead `Line` column.**

Delete 882–901 (the `_TEAM_ONLY_MARKETS` early return, the `lines` read and the join) and return
`wide`. Keep the `_TEAM_ONLY_MARKETS` constant (78): `scripts/migrate_archive_shapefree.py:36`
imports it and `books/underdog.py:225` names it. In `fit_book_weights`, the `"Line"` entry in the
list at 210 and the drop at 219–220 go too. Fix the docstring (846–847). The other two callers of
`to_pandas` (`tests/golden/test_archive_shapefree_storage.py:565, 569` and
`tests/integration/test_end_to_end.py:81`) do not read the column.

**Edit 9. The board lookup and the flag** (`prediction/cli.py`, plus one shared function).

*9a. One rule for the archive's market name.* The board label is not always the archive key.
`add_dfs` files NHL `AST` / `PTS` / `BLK` under `assists` / `points` / `blocked`, and NBA or WNBA
`… underdog` under `… prizepicks` (`_resolve_market`, `archive.py:236–244`). The board lookup asks
under the board label and misses (section 8, D2). Give the two league fixups one home in
`helpers/archive.py` and make the writer call it:

```python
def archive_market(league: str, market: str) -> str:
    """Archive key for a market label already renamed through its platform's ``stat_map``."""
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points", "BLK": "blocked"}.get(market, market)
    if league in ("NBA", "WNBA"):
        market = market.replace("underdog", "prizepicks")
    return market


def _resolve_market(league: str, raw_market: str, key: dict) -> str:
    """Rename a sportsbook-native market string to its canonical per-league name."""
    market = raw_market.replace("H2H ", "")
    return archive_market(league, key.get(market, market))
```

`normalize_market` (`prediction/model_prob.py:153`) holds the same four lines after its alias
lookup. Its body becomes `return archive_market(league, stat_map[platform].get(market, market))`,
which leaves one copy of the fixups instead of two. Behavior of both callers is unchanged. Export
`archive_market` from `helpers/__init__.py` beside `Archive` (the import block at 31–36 and
`__all__` at 98); `cli.py` and `model_prob.py` already import from there.

Do **not** call `normalize_market` on a board label. The board's `Market` has already been through
`stat_map` once (`cli.py:264, 300`), and the alias is not idempotent: Sleeper maps `bat_walks` to
`walks` and `walks` to `walks allowed`. A second pass sends the 60 MLB Sleeper `walks` keys on the
current board to the pitcher market. (An earlier draft of this note made that mistake; running the
edit on the real board caught it.)

*9b. The lookup.* In `main`, replacing 349–358:

```python
key_cols = ["League", "Market", "Date", "Player"]
# The archive files a row under its own market key, not the board's label
# (an NHL board says AST where the archive says assists).
archive_keys = [
    (league, archive_market(league, market), date, player)
    for league, market, date, player in snapshot_offers[key_cols].itertuples(index=False, name=None)
]
consensus = {key: archive.get_line(*key) for key in set(archive_keys)}
reference = {key: line or archive.get_reference_line(*key) for key, line in consensus.items()}
snapshot_offers["Consensus Line"] = [consensus[key] or np.nan for key in archive_keys]
snapshot_offers["Reference Line"] = [reference[key] or np.nan for key in archive_keys]
```

`get_reference_line` is asked only for the keys with no consensus. It re-reads the consensus before
it reads the lines log: 255 keys and 1.4 s on the current board.

One input to check. A row whose raw label is missing from its platform's `stat_map` reaches this
point with a NaN `Market` (`Series.map` at 264 and 300 leaves NaN; the history path drops such rows
at 412). Today's lookup hands the NaN to SQL and reads no line. `archive_market` would raise on it
for an NBA or WNBA row (`float` has no `replace`). The current board has no such row (0 of 4,544),
and the Sleeper map carries identity entries to prevent them, so this note adds no guard. If the
implementer wants the old tolerance, the least code is to drop those rows from `snapshot_offers`
before the lookup, as line 412 does for history. That removes them from the board too, which is
the owner's call.

*9c. The flag.* Carry `Reference Line` through the merge at 401–408 beside `Consensus Line`
(`snapshot_offers[[*PREDICTION_KEY, "Consensus Line", "Reference Line"]]`), and judge the flag
against it in `_stamp_alt_line` (187–188):

```python
diff = (offers["Line"] - offers["Reference Line"]).abs()
offers["Alt Line"] = (diff > tol).where(offers["Reference Line"].notna(), False)
```

`Reference Line` is a working column. History keeps `all_df[HISTORY_COLS]` (`cli.py:424`) and the
board keeps `_OFFER_KEEP_COLS` (`persist.py:268`), so neither writer persists it; the fallback
branch at 415–417 needs no new column. Update the `_stamp_alt_line` docstring.

Under rule A1 instead (section 7), 9b and 9c shrink: no `reference` dict, no `Reference Line`,
`_stamp_alt_line` unchanged. 9a stays.

Both parts were run as written, read-only, on the real current board:

- 9b through the patched archive (`m17_edit9_board.py`): `Consensus Line` and `Reference Line`
  equal the SQL-measured columns on 4,544 of 4,544 rows, and every row gets a reference line.
- 9c with the repository's own `_stamp_alt_line` patched in memory and the merge statement above
  (`m19_edit9c_stamp.py`): 22.0% of rows flagged, against 22.2% from today's function on the
  stored column; 5.1% of rows flip (MLB 3.9%, NFL 5.0%, NHL 7.1%, WNBA 5.4%). Under A1 6.6% flip.
  These are the board figures in section 3g.

**Edit 10. Comments and docstrings that state today's behavior.**

- `archive.py` module docstring (8–23) and the `add_dfs` docstring (1253–1264): `lines` is the
  line-of-record log.
- `history_schema.py:46` (Alt Line) and `:63` (Consensus Line: currently "The sportsbook modal
  line", which was never true; it becomes true as "sportsbook consensus line, NaN when no
  sportsbook posts one").
- `prediction/persist.py:36–37` ("weighted-avg book line from archive.get_line").
- `helpers/training_quotes.py:22`: `CONSENSUS_LINE_POLICY = "modal-nearest-median-v1"` has no
  reader anywhere. Delete it, or leave it; do not restamp it.

### 4.3 What each reader does for a DFS-only entry

`get_line` returns `0.0`. Then:

| Reader | Receives | Behavior | Change from today |
|---|---|---|---|
| Board `Consensus Line` (R1) | NaN | No consensus marker on the Model tab; history stores NaN | Today: the DFS line shown as "consensus", or 0.0 on a first poll |
| `Alt Line` flag (R1), rule A2 | line of record | Judged against the platform's own main line; False when there is no line at all | Same reference as today for these rows |
| `Alt Line` flag, rule A1 | NaN | Always False | 19.5% of rows lose the flag |
| Combo sublines (R2–R4, R7) | line of record | Same pivot as today | None. A strict 0.0 here would break: see 4.6, alternative 3 |
| Resolver `legacy_line` (R6) | line of record | Rungs 2–5 price at the same line as today | None |
| `pickem_quote` (R6) | line of record | The fantasy 50/50 stand-in quote is unchanged | None |
| Quote resolver direct cohort | DFS rows only | `PICKEM` quote, as today. Never served (`_has_serving_support`). | None |
| Tail scorecard | NULL consensus | Band `none`, as today | None |

For a mixed entry: `Consensus Line` is the sportsbook number; the Alt flag is judged against it;
combo sublines and `legacy_line` use it; the direct cohort is the sportsbooks' modal line.

### 4.4 Should `add_dfs` stop staging into `lines`? No.

The log is the line of record for DFS-only keys. If `add_dfs` stopped staging (`archive.py:1317`):

| Would break or degrade | How |
|---|---|
| Alt Line reference for DFS-only entries (rule A2) | No line of record going forward → flag False, the A1 outcome by the back door |
| `pickem_quote` | The stand-in for an unpriced fantasy line needs the logged line and its timestamp. Unreached for new rows (DFS rows are priced since 2026-03), so no practical loss, but the pre-2026 path and `test_unpriced_pickem_line_resolves_as_the_platforms_own_symmetric_quote` rely on the table. |
| Combo sublines on DFS-only components | Line of record missing → `_fantasy_default_contribution` falls back to the player's last-10 median |
| `sweep_runaway_odds.py:96`, `delete_corrupt_seed.py:63` | Their `MAX(l.line)` bound would not see DFS lines, so a DFS row's EV would be judged against a missing or sportsbook-only bound |
| `get_movement` | `open_line` / `close_line` NaN for DFS-only keys |
| `tests/golden/test_archive_shapefree_storage.py:310` | Asserts the `lines` row `add_dfs` writes |

Not affected either way:

- `archived_players_by_date` (`archive.py:904`) reads `odds`, not `lines`.
- The close-lines job is driven by the game ledger and writes through `merge_player_books`.
- CLV on DFS offers: the closing probability comes from `get_ev` /
  `get_composite_under_prob`, which read `odds`.
- The tail scorecard and the line-movement snapshot read `ladder`.

Stopping the staging saves nothing the new `get_line` needs (it no longer reads `lines`) and
costs the items above.

### 4.5 Should `fit_book_weights` drop the DFS platforms? Yes.

- **What the weights are used for.** `_weighted_book_ev` and the resolver's `_weighted_value`
  average only over `sportsbook_cohort`, so a DFS weight is used only inside a DFS-only cohort,
  which is never served. But the DFS columns sit in the fit and absorb weight mass (median 20–44%
  of a cell outside MLB), which moves the relative sportsbook weights that are used: by up to
  0.38 of normalized weight (section 3h).
- **Accuracy.** A wash: out-of-sample NLL differences of ±0.006 or less on eight cells, inside the
  gap to equal weights.
- **So the reason is correctness**, fitting what is used, not a measured gain.
- **After the next refit** a DFS-only cohort averages its platforms at equal weight (`_weight`
  defaults to 1.0). That touches `pickem` quotes only.
- **Timing.** `book_weights.json` is gitignored and reaches production through `sync_to_prod.sh`.
  `_step_init_market` (`pipeline.py:728–735`) replaces a cell's dict wholesale on refit, so the DFS
  keys disappear at the next dev `meditate` or `scripts/refit_book_weights.py` run. Nothing changes
  on deploy day.

### 4.6 Rejected alternatives

1. **Keep reading `lines`; stop DFS staging and purge DFS rows (S1).** Destructive on both
   archives; legacy rows cannot be attributed, so the purge is incomplete by construction; and it
   keeps the "every distinct line ever seen" statistic, which is the source of the whole-number
   lines (32% of NFL and 51% of WNBA mixed keys at decision time today, against 6% and 11% under
   S3).
2. **Add a `book` column to `lines`.** A schema change on a 21M-row table for information `odds`
   already holds; history stays unattributed.
3. **Strict everywhere: no reference line.** Simplest code. But measured: the resolver would lose
   its line on 8,752 cached-matrix rows since 2023-09 and re-anchor them on Avg10 (NFL 2,985, NHL
   2,922, NBA 1,148, WNBA 995, MLB 702; mostly `neutral_fallback` rows), and on about 22,500 over
   all cached history (15,931 of them NFL `neutral_fallback`); the Alt flag would fall from 46% to
   29% of rows; and a 0.0 pivot would reach `_convert_to_market_dist` and the `pitcher win`
   `get_odds` on the 2023–24 NULL-line rows. The conversion does not raise at a zero pivot, it
   returns garbage (a NegBin mean of 2.3 converts to 0.78; a SkewNormal 18.4 converts to 1.0).
4. **`get_line` falls back to the DFS line itself** (cohort-first, like `sportsbook_cohort`).
   Least code of all, but the `Consensus Line` column would show a DFS line under the name
   consensus, which is the thing decision 11 forbids.
5. **Median over distinct sportsbook lines (S3d).** Agrees with the resolver's sportsbook modal
   line far less often (NFL 70.7% against 91.8%, WNBA 66.5% against 88.2%).
6. **Make `get_line` modal, matching the resolver.** A second policy change in one step, and it
   moves more numbers for no measured gain; per-book median already agrees 88–99%.
7. **Batch the board lookup per (league, market, date).** Faster, but more code than a saving of a
   few seconds justifies. Revisit only if the per-key loop shows up in the prophecize profile.

### 4.7 Suggested commits

Production self-deploys from `devel` on push, so the split controls what changes when.

| Commit | Edits | Changes on deploy |
|---|---|---|
| A | 1, 2, 3, 4, 8, 9, 10 and their tests | `Consensus Line`, `Alt Line`. No served probability. |
| B | 5, 6 and their tests | Served book leg on about 1.5% of mixed keys; CLV closing probability on 0.2% |
| C | 7 and its test | Nothing until the next weight refit and sync |

---

## 5. Tests and docs

### 5.1 Existing tests that move

Verified, not inferred. The 47 non-integration test files that touch an archive object or any
affected function were run on unpatched code (473 passed) and again with the proposed change
applied in memory (`-p consensus_patch`): **7 fail, 466 pass.**

| Test | Why | Fix |
|---|---|---|
| `tests/test_archive_history.py::test_get_line_at_aggregates_distinct_lines_observed_through_cutoff` (143–173) | Inserts only into `lines` and asserts `get_line` returns 23.0 / 22.5 / 0. Pins today's semantics. | Point the same assertions at `get_reference_line` (they hold unchanged: no sportsbook row exists, so it returns the line of record), and assert `get_line` returns 0.0 for that key. |
| `tests/test_combo_market_helpers.py` × 6: `test_submarket_ev_no_conversion`, `…_missing_is_nan`, `…_missing_league_no_keyerror`, `test_combo_market_ev_sums_legs`, `…_zero_when_leg_missing`, `…_zero_when_leg_zero` | `_FakeArchive` (line 25) defines `get_line` only | Rename the fake's method to `get_reference_line` |

Applied one part at a time, edits 5, 6, 7 and 8 fail nothing. **No existing test pins the DFS
vote, the divergence order, or the DFS columns in the weight fit.** Those behaviors need new
tests (5.2).

A re-run twenty minutes later, with other sessions' Underdog payout work in progress in this
checkout, showed three failures on unpatched code that are theirs, not this change's:
`tests/golden/test_get_ev_robustness.py::test_add_dfs_one_sided_underdog_offer_converts_through_baseline`,
`tests/golden/test_kelly_parity_annotate_vs_finalize.py::test_annotate_recomputes_the_board_kelly`
and `tests/golden/test_offer_records_kelly.py::test_underdog_payout_at_or_below_one_zeroes_kelly`.
With the patch the same run fails those three plus the same seven (10 fail, 463 pass). An
implementer who sees those three red should not look for them here.

Edit 9 moves no non-integration test:

- No test names `_stamp_alt_line`, and none outside `tests/integration/` runs `cli.main`.
- `tests/integration/test_end_to_end.py:187, 276` does run `main` in fake mode, so the lookup and
  the stamp execute there. It asserts nothing on `Consensus Line` or `Alt Line`. It was not run
  here (integration), so it is the first place a slip in edit 9 would show.
- `tests/test_book_fallback_prob.py:104`, the one test that calls `normalize_market`, passes a raw
  platform label, which 9a leaves unchanged.
- `tests/golden/test_history_decision_cols.py:38–41` requires every decision column except
  `Scored At` and `Consensus Line` to come off the scored record. It stays green because
  `Reference Line` is a working column and not a history column; adding it to `DECISION_COLS`
  would turn that test red.

Tests that pass but should be touched:

- `tests/golden/test_archive_shapefree_storage.py::test_training_quote_batch_matches_scalar_rows_and_lines`
  (122–144): line 144 compares the batch line with `get_line`. It passes because the fixture has
  only sportsbooks. The parity it means is with `get_reference_line`; change the call and add a
  DFS-only entity so the two differ.
- Fakes that define an unused `get_line`: `tests/golden/test_book_prob_feature.py:25`,
  `tests/golden/test_train_live_feature_parity.py:136`,
  `tests/golden/test_training_quote_resolution.py:322`. Not exercised today; rename or delete so
  they do not suggest a live call path.
- `tests/golden/test_archive_shapefree_storage.py::test_multi_tier_ingest_keeps_one_odds_row_and_ladders_every_tier`
  (278–310) asserts the `lines` row `add_dfs` writes. Stays green because staging stays.
- `tests/integration/test_end_to_end.py:80–82` filters `"Line"` out of `to_pandas` columns. Works
  with or without the column; the filter becomes dead. Not run here (integration).
- `tests/test_refit_book_weights.py:24–30`: the fixture has `"PTS": {"Sleeper": 0.63}` and the
  fitter is monkeypatched. Unaffected.

### 5.2 New tests

Small, one behavior each, in the files that already hold their neighbours.

| # | File | Pins |
|---|---|---|
| 1 | `tests/test_archive_history.py` | `get_line` ignores a DFS `odds` row: two sportsbooks at 22.5 and 23.5 plus Underdog at 30.5 → 23.0 |
| 2 | same | `get_line` returns 0.0 for a key with only an Underdog row and a `lines` row; `get_reference_line` returns that line |
| 3 | same | `get_line(at=…)` uses each sportsbook's latest line at or before `at`, not every line seen (a book that moved 22.5 → 24.5 counts once, at the value for the cutoff) |
| 4 | same | `get_reference_line` returns the line of record for a sportsbook row with NULL `line` (the 2023–24 MLB shape) |
| 5 | `tests/golden/test_archive_shapefree_storage.py` | Batch equals scalar: `get_training_quote_inputs(...)[e][1] == get_reference_line(...)` for a sportsbook, a mixed and a DFS-only entity; same for `get_ev_line_inputs` |
| 6 | `tests/golden/test_training_quote_resolution.py` | Mixed cohort, DFS line outvotes: one sportsbook at 1.5, Underdog and Sleeper at 2.5 → quote line 1.5, `authentic`, `books == (sportsbook,)`. Today this is 2.5 / `pickem`. |
| 7 | same | Minority line: two sportsbooks at 20.5; one sportsbook, Underdog and Sleeper at 21.5 → line 20.5 with the two sportsbooks |
| 8 | `tests/golden/test_consensus_guards.py` | `_weighted_book_ev`: one sportsbook at 20.5, two DFS rows at 14.5 → the sportsbook's EV (today the sportsbook is dropped as divergent) |
| 9 | `tests/test_refit_book_weights.py` or a calibration test | `fit_book_weights` returns no DFS key when `to_pandas` has an `Underdog` column |
| 10 | new, beside the cli tests | `_stamp_alt_line` reads `Reference Line`: within tolerance → False; beyond it → True, with the count and continuous tolerances; NaN reference → False even when `Consensus Line` is NaN too |
| 11 | `tests/test_archive_history.py` or beside the cli tests | `archive_market`: NHL `AST` → `assists`; WNBA `fantasy points underdog` → `fantasy points prizepicks`; MLB `walks` stays `walks`. And through `_resolve_market` with Sleeper's map: `bat_walks` → `walks`, `walks` → `walks allowed` (the pair a second alias pass gets wrong). |

The no-duplicate-code gate (`tests/golden/test_no_duplicate_code.py`) should be run by the
implementer; edit 3 removes a clone rather than adding one.

### 5.3 Docs that state today's behavior

| File | Where | Today | After |
|---|---|---|---|
| `CLAUDE.md` | Archive paragraph, ~366–371 | `lines(league, market, game_date, entity, line)` described beside `odds` with no role; public-method list | Say `get_line` is the sportsbook consensus read from `odds.line`, `lines` is the line-of-record log; add `get_reference_line` to the method list |
| `CLAUDE.md` | config table, ~475 | `book_weights.json`: "Sportsbook reliability weights for consensus lines" | "for the consensus price"; sportsbooks only |
| `README.md` | 128 | same wording | same fix |
| `docs/ARCHITECTURE.md` | `archive.py` row (~68); prophecize flow (~220, `Archive.get_line() / get_ev()`); "Add/remove a sportsbook from consensus lines" (~260) | Describe the mixed read | One sentence on the two reads; the ~260 row is already correct in spirit |
| `docs/STYLE_GUIDE.md` | glossary, 501–508 | Lists PrizePicks, Underdog, Sleeper and ParlayPlay under "Book (sportsbook)"; lists the Archive read methods | Split DFS platform from sportsbook; add the method |
| `docs/pipeline_diagrams.md` | 37 | "Archive book consensus get_line / get_ev / get_total" | No change needed |
| `docs/handoffs/honest-receipts.md` | §7, the "Where the code stands against decision 5" bullet (~366–378) | Describes the leak as open | Revise in place to the landed state; cross-reference rather than restate |
| `docs/handoffs/cleanup-pass.md` | 2026-09-12 log line (~179) | "real leak was `lines`" | Historical log entry; leave |
| `docs/merge_archives.md` | 9 | Mentions the `lines` table | No change |

Per `docs/STYLE_GUIDE.md` §16, the fact "what `get_line` returns" should live in one place (the
`get_line` docstring and the ARCHITECTURE row) and be cross-referenced elsewhere.

---

## 6. What can change on the day this deploys

Production pulls `devel` before every cron job, so each commit takes effect at the next
prophecize run after the push.

| # | What | Size | Reaches a served probability? |
|---|---|---|---|
| 1 | **Cohort vote (commit B).** Book leg re-priced off the sportsbooks' modal line. | About 1.5% of mixed keys: NFL 6.6%, WNBA 7.4%, MLB 0.8%, NHL 0.4%. 56 keys on the current board, 44 of them NFL. Book-leg shift median 0.02–0.05, p90 0.14 (MLB count cells), times (1 − model weight). About 159 keys per five weeks gain a book leg they do not have today (MLB 111). About 200 more simple-combo keys per five weeks see a component of their sum re-priced (size not measured). | **Yes** |
| 2 | Same keys' book-side columns: `Market Prob`, `Books EV`, `Quote Source`, `Quote Books` | Full size on those keys | Display and evaluation; `Market Prob` feeds Market CLV |
| 3 | `Consensus Line` values | Changes on about a third of current board rows against today's rule (NFL 56%, WNBA 59%, MLB 15%, NHL 2.5%), mean gap 1.0, and on 44% against the stored values (NHL 60%, its zeros becoming real lines). NaN on 10% now; 39% of history rows had no sportsbook line at decision time. | No |
| 4 | `Alt Line` flag | 4.4% of rows flip under A2 (NFL 11.5%, WNBA 8.0%); 21.8% under A1 | No. It moves rows between the main and alt halves of the receipts split. |
| 5 | **History has two regimes.** `Consensus Line` and `Alt Line` are stamped at write time and never backfilled. Open rows are restamped by the next run; rows already closed keep the old stamp. | Every receipts or research cut on these columns that spans the deploy date mixes two definitions. `Scored At` separates them. | No |
| 6 | Prophecize run time | +4 to +5 s on the current board, up to about +13 s cold | No |
| 7 | CLV closing probability (commit B, edit 6) | 0.2% of mixed keys | No |
| 8 | NHL board rows get a real `Consensus Line` | 644 of 927 NHL rows on the current board show 0.0 today (section 8, D2) | No |

Not on deploy day:

- Book-weight refit (commit C): at the next dev refit and sync.
- Matrix book columns: at the next re-resolve or rebuild of a cell (section 2.2).

Cannot be fixed by any code change: legacy lines and the migrated `odds.line` keep whatever DFS
lines the old klepto archive mixed in. For pre-2026 history "sportsbook-only" means "from
sportsbook rows", not "provably free of a DFS line".

One thing to check before pushing commit B: the 56 current-board keys are mostly NFL yardage
cells, and NFL served cells run at model weight 1.0 at the median, so the served move there is
close to zero; the visible change is in `Market Prob`. The cells where a served probability moves
measurably are the fused MLB count cells (median model weight 0.65).

---

## 7. Decisions for the owner

1. **Alt Line on a DFS-only entry: A2 or A1?** A2 judges the flag against the platform's own main
   line (today's behavior for those rows, 4.4% of rows flip overall). A1 leaves the flag False
   with no sportsbook line (21.8% flip; the alt rungs of DFS-only entries would be counted as main
   lines in the receipts split). Recommended: A2, on the reading that the flag is an evaluation
   cut and decision 5 allows a DFS line there.
2. **`_book_mean_shift`** (`base.py:2538`). On fantasy markets the training-side component-sum
   mean is moved halfway to the market's own quote, and on a fantasy market that quote is the DFS
   platform's line. The docstring's evidence (MLB hitter fantasy, n=1,345): PIT KS 0.221
   component sum, 0.072 book only, 0.067 averaged. It produces `DERIVED` training rows only and is
   not served. Is this "blending with the DFS main line" under decision 5? Left untouched here.
3. **`get_movement`** reads the mixed log, so `frac_lines_moved_toward_model` counts alternation
   between a DFS rung and a sportsbook median as movement. Diagnostic only. Rebuild it on
   sportsbook history, or leave it?
4. **`Archive.get_ev` on a DFS-only key** returns the DFS platform's EV as the consensus EV. It
   feeds the CLV closing probability for DFS-only offers. Consistent with "evaluate against", but
   it carries the consensus name.
5. **Half of MLB modeled player-market-days had no sportsbook quote at decision time** (48.9%),
   against 24.3% once close-lines has run. Not caused by this change, but it bounds how much of the
   MLB board can ever show a consensus line or serve a book leg at scoring time.

---

## 8. Adjacent defects found (not part of the change unless noted)

| | Defect | Evidence |
|---|---|---|
| D1 | `Consensus Line` is 0.0, not NaN, on a miss, so `_stamp_alt_line`'s NaN guard never fires and the dashboard draws a consensus marker at 0. | 659 of 4,544 current board rows. **Fixed by edit 9.** |
| D2 | The board lookup passes the board's market label; the archive files under its own key. The rename to archive names (`cli.py:409–411`) runs after the lookup and covers NHL only. | NHL `AST` (250 rows) and `PTS` (325 rows) are 100% zero on the current board; NHL overall 69.5%. The same miss hits NBA and WNBA `fantasy points underdog`, archived as `fantasy points prizepicks`: 1,062 WNBA Underdog rows in history, 2026-07-04 to 2026-08-29, none on the board now. **Fixed by edit 9a and 9b**, which make the reader use the writer's rule. |
| D3 | `Archive.get_closing_line` (`archive.py:918`) and its helpers have no caller in `src/`, `scripts/` or `tests/`. | grep. CLAUDE.md's orphan rule sends it to `src/deprecated/`. |
| D4 | `CONSENSUS_LINE_POLICY` (`training_quotes.py:22`) has no reader. | grep |
| D5 | `Stats._submarket_ev` and the two MLB readers call `get_ev` and `get_line` with no `at=`, although they run only in training, so the legacy combo scalar reads close-layer rows. | `base.py:2466–2467`, `mlb.py:1408–1409, 1430–1431` |
| D6 | A simple-combo key whose only direct quote is a DFS platform's keeps that quote in `servable_fallback_quotes` (`book_quotes.py:176–179`), and `_has_serving_support` then drops it, discarding a component sum built entirely on sportsbook components. | Read from code; not measured. |
| D7 | Cached MLB and WNBA matrices (dated 2026-09-17) and NBA and NHL (August) predate the `pickem` authenticity (2026-09-29, `daab5fef`) and still label DFS-only cohorts `authentic`. Only NFL has been re-resolved. | `matrix_book_cols.parquet`: NFL is the only league with `pickem` rows (11,496). A re-resolve is owed regardless of this change. |
| D8 | The three stale comments in edit 10. | |

---

## Appendix A. Method and reliability

- **Archive access.** Every query used `duckdb.connect(path, read_only=True)` on
  `archive/archive.duckdb`, or the `Archive` singleton with `SPORTSTRADAMUS_ARCHIVE_READ_ONLY=1`.
  Never read-write, never under `flock`.
- **Leagues.** MLB, NBA, NFL, NHL, WNBA. "Modeled markets" are the cells in `stat_meta.json`.
- **Row attribution of `lines`.** ASOF join of each `lines` row to the latest `odds` row on the
  same key at or before its timestamp. Reliability by era is in section 3b.
- **Decision time.** The local `history.parquet` predates the `Scored At` column, so decision
  time is emulated as each key's last DFS poll in `ladder`. The stored `Alt Line` flag reproduces
  on 97.6% of rows under that emulation.
- **Training cutoff.** Game date 12:00 UTC, as `get_training_matrix` computes it.
- **Legacy `odds.line` is not a per-book line.** Rows stamped at game-date midnight got the mixed
  `lines` consensus from `migrate_archive_shapefree.py`. They agree across books on every key.
  Backfill and live rows are per-book except some pre-WS1 batches (NFL backfill 01:00 layer
  2023–2026, NBA 2025–26 and WNBA 2025 backfill, NBA live May–June 2026).
- **Data quality in `odds.line`.** No NaN values. 13 non-positive lines among 23.78M sportsbook
  player-prop rows.
- **Cohort accuracy** (3e) uses settled history rows and a bootstrap over keys; n is small
  (128–457), so read the direction and the interval, not the third decimal.
- **Tests.** Only the 47 targeted files were run (`-n0`, cache provider off). No integration run,
  no full golden run.
- **Side effect to know about.** Importing the package writes a per-process log file under
  `src/sportstradamus/logs/` and touches the day's `logs/<date>/*.jsonl`. Every script here that
  imported the package, and every targeted test run, left those. Both locations are gitignored.
  `git status --porcelain` was identical before and after the test runs and the board run; a
  later comparison differed only by five `strategies/` files another session was editing.

## Appendix B. Scratch files

All in `/tmp/claude-1000/-home-trevor-Sportstradamus/7c225c1a-7c1e-4cde-bede-cf7b5fe0cce8/scratchpad/`.
Run with `/home/trevor/Sportstradamus/.venv/bin/python`.

| File | What it produces |
|---|---|
| `m1_odds_line_share.py` → `m1_odds_line_share.csv` | 3a, by league and month |
| `m1b_native.py` | Whether a non-null line is per-book |
| `m2_build.py` → `lines_attrib.parquet`; `m2_attrib_summary.py` | Row attribution of `lines` |
| `m2_keys.py` → `m2_keys.parquet` | 3b and 3d, key level |
| `m3_build.py` → `m3_serve.parquet`, `m3_final.parquet`, `m3_train.parquet`; `m3_summary_serve.py`, `m3_summary_train.py` | 3c: today vs S1, S3, S3d at three cutoffs |
| `m5_cohort_build.py` → `m5_*.parquet`; `m5_summary.py` | 3e, cohort vote counts |
| `m11_cohort_effect.py serve`, `m11b_cohort_accuracy.py` | 3e, served book-leg shift and Brier |
| `m6_divergence.py` | 3f |
| `m7_altline.py`, `m7b_altline_options.py` | 3g |
| `m10_bookweights.py` | 3h |
| `m12_perf_and_parity.py`, `m12b_perf_stable.py` → `m12b_board.parquet` | 3i and the current-board table |
| `m8_matrix_vs_lines.py`, `m9_matrix_provenance.py` → `matrix_book_cols.parquet` | Section 2.2, cached-matrix rows by resolver rung |
| `m13_sanity.py` | NaN lines, same-timestamp duplicates |
| `patchdir/consensus_patch.py` | Edits 1–8 as an in-memory pytest plugin (`CONSENSUS_PATCH_PARTS` selects edits) |
| `m14_patched_parity.py` | Patched methods against the real archive, read-only |
| `m15_combo_scalar.py` | Section 2.2, legacy combo scalar sensitivity |
| `m16_board_market_key.py` | Board label against the archive's market key (edit 9a, D2) |
| `m17_edit9_board.py` | Edit 9b and 9c run on the current board through the patched archive |
| `m18_combo_knockon.py` | 3e, simple-combo keys whose component quotes change |
| `m19_edit9c_stamp.py` | Edit 9c: the patched `_stamp_alt_line` and the merge statement on the current board |
| `pytest_baseline.log`, `pytest_patched.log`, `pytest_part_*.log`, `pytest_extra_*.log` | Section 5.1, first runs |
| `pytest_final_baseline.log`, `pytest_final_patched.log` | Section 5.1, the re-run |
| `target_sorted.txt`, `archive_touching_tests.txt` | The test files that were run |
