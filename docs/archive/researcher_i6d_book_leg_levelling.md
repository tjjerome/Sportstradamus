# In-repo research brief: levelling the historical sportsbook leg (I6d step 1, live-gap-2 week 2)

**Question.** Every served cell is trained, gated and blend-weighted against the matrix `Odds` column. That column is the sportsbook probability resolved per row from the archive `odds` table. The brief asks:
- where its level error comes from, per era, cell and book;
- how to level it without leaking into training, Gate 1 or the blend-weight fit;
- which read proves the levelling worked.

**Date and scope.** 2026-10-27. Read-only.

**Data sources.**
- **Archive numbers:** the 10-26 06:31 backup `archive/archive.duckdb.bak-1793104621`, opened read-only so the live file's DuckDB lock stayed free for cron.
- **Matrix numbers:** the cached matrices in `src/sportstradamus/data/training_data/`.
- **Served-split numbers:** `src/sportstradamus/data/test_sets/*.csv`.
- **Blend weights and BSS:** `model_stats.parquet` (10-18).

**What "held-out" means in matrix tables.** It is the last 30% of rows by date. The served split is 70/30 by date, then a (Player, Date) hash into validation and test.

**Scripts.** In `~/backups/sportstradamus/2026-10-21-live-gap-2/research/i6d_scripts/` (Appendix A).

---

## TL;DR

- **Root cause (Class I): the WS1 migration decoded every row written before 2026-06-28 under the wrong shape. Devig, the push label and the line of record are not the cause.**
  - **What went wrong.** Those rows stored only `ev`. Each `ev` was encoded under the family, cv and leg gate in force when the row was written. `migrate_archive_shapefree` (`377972ce`) converted each one to `under_prob` under the 06-28 shape. Its round-trip check is the identity under any shape, so it could not fail.
  - **The signature is a gate floor.** Every migrated NFL tds row reads under ≥ .812 (median .967). The native close layer spans .035 to .99 (median .871).
  - **The information survived in `ev`.** Decoding it under the original encoding family reproduces native prices on the same game, player, book and line:

    | Rows | Decode family | Mean diff | MAD |
    |---|---|---|---|
    | NHL goals 00:00 | Poisson | −.0002 | .0000 |
    | NBA 2025 01:00 (BLK, STL, FG3M) | NegBin | — | ≤ .001 |
    | NFL tds | NegBin | −.0015 | .008 |
    | NFL interceptions | SkewNormal | −.008 | — |
    | NFL passing tds | SkewNormal | +.002 | — |

- **A second defect (Class II) is live in the writer today, and it accounts for the NHL and MLB anomalies.**
  - **What it does.** `moneylines._resolve_prop_lines` lets two outcome rows on the same side through as a pair, and `no_vig_odds(p, p)` returns .5. The ladder rung written from the same rows reads about .18.
  - **Served test sets:**

    | Cell | Rows at .5 | BSS on .5 rows | BSS on the rest | Cell BSS now | Cell BSS with .5 rows at .18 |
    |---|---|---|---|---|---|
    | MLB home runs | 2,048 of 7,572 | +.473 | +.027 | +.256 | +.017 |
    | NHL goals | 601 of 16,621 | +.625 | +.000 | +.044 | +.002 |

  - **What it explains.** These rows are NHL's > .10 disagreement band and the reason week 1's "MLB home runs BSS .154 after re-levelling" did not move.
  - **It is still happening.** `williamhill_us` has written 9,829 such live MLB home-run rows since 07-11.

- **Measured: on the NFL count cells, all offline skill sits on migrated rows.** Served test sets split by quote provenance:

  | Cell | BSS, migrated rows (n) | BSS, native rows (n) |
  |---|---|---|
  | tds | +.168 (2,751) | −.037 (742) |
  | interceptions | +.301 | −.072 |
  | passing tds | +.144 | +.009 |
  | receptions | +.059 | −.010 |
  | carries | +.026 | +.001 |

  NBA's current matrices hold no native rows at all.

- **Recommend one candidate: a provenance-keyed re-read at training read time.**
  1. Classify each archived book-row as native or migrated, from the archive alone.
  2. Repair Class II rows from the same-observation ladder.
  3. Handle each migrated block on prices alone:
     - re-decode it if its encoding family passes a pre-registered test against native rows (|mean| ≤ .01, MAD ≤ .03);
     - keep a stored decode that already passes;
     - demote the rest out of the authentic cohort.
  4. Fix the writer.
  5. Leave the archive raw. `merge_archives` keeps `MAX(under_prob)`, which silently undoes any edit that lowers it.
  6. Re-inject and retrain per league.

  This is engineering: about 2–3 dev-days plus one retrain per league. Re-decoding the live-poll layers by month or epoch is the one bounded research piece.

- **KILL list.** The per-era logit shift; the close layer as the book leg; in-place archive edits; stand-alone era exclusion (it leaves NBA with zero authentic rows and NFL interceptions and passing tds with about 80); re-buying raw quotes; any decode choice fitted on outcomes. The book-CITL column ships report-only.

- **The deciding read is offline book level against live, before any retrain.**
  - **NFL.** H3's L over the seven NFL count cells must fall to ≤ .03. Today it is .102 [.084, .118] offline and .019 [.013, .046] live. The offline-minus-live CI must cover 0.
  - **Every served cell.** |book CITL| ≤ .03, except pre-named residuals (one-sided longshot props and the Class III cells).
  - **Before anything is consumed.** The price-only decode and writer checks pass.
  - **After the retrain (expected).**
    - BSS on the affected cells falls to about −.05…+.05.
    - The blend weight w falls from .72–.90 toward riding the book.
    - Gate 1 mostly keeps passing, because non-inferiority lets the blended probability ride a level book.
    - So I6d's fixed encompassing test and the tail scorecard, not Gate 1, decide what skill remains.

## 1. Decision and the statistic that decides it

**Decision.** Which levelling build to make before any I6d model work, and the read that accepts it.

**Statistic.** The book leg's mean calibration (calibration-in-the-large, CITL), per cell:
- CITL = mean(P_book,over − y), with y from `training/labels.over_label` (a push counts as ½);
- measured on the served split's priced held-out rows;
- compared with the same quantity on the live board's decision-time quotes.

**Build test.** A test on prices alone: does the levelled value reproduce a native price on the same game, player, book and line?

The I6d acceptance (honest-receipts §6 item 5) is fixed and sits downstream of this brief.

## 2. Key findings

### Finding 1: the offset is a decode-shape error written by the WS1 migration (Class I)

**Mechanism.**
- **How pre-WS1 rows were stored.** Before WS1 (`377972ce`, 2026-06-28 20:24 CDT), `_event_player_props` stored only `ev = get_ev(line, devigged under, cv, dist)` per book, using the config live at write time.
- **The 06-02 NFL backfill was ungated.** The 06-02 NFL feature backfill (code at `7a624294`) passed no gate at all. Its `get_ev` sent an ungated ZINB call down the NegBin branch. `book_gate` arrived on 06-06 (`91379e10`).
- **What the migration did.** `migrate_archive_shapefree._migrate_cell` wrote `under_prob = get_odds(line, ev, <06-28 stat_dist, stat_cv, book_gate>)`.
- **Why its check could not fail.** It verified the round trip under that same shape. A decode followed by its own inverse is the identity for any shape, so the check tests the arithmetic, not the shape. It is a pseudo-oracle that cannot fail (Weyuker 1982; Barr et al. 2015).
- **Vintage framing.** In vintage terms (Croushore & Stark 2001), WS1 replaced each row's original reading with a 06-28 re-interpretation of it.

**Signature: the gate floor.** All rows; quantiles of `under_prob`:

| NFL market | 01:00 migrated rows | min | p01 | median | native close rows | min | p01 | median |
|---|---|---|---|---|---|---|---|---|
| tds | 170,561 | .812 | .866 | .967 | 123,249 | .035 | .402 | .871 |
| interceptions | 9,613 | .711 | .812 | .834 | 6,838 | .302 | .371 | .500 |
| passing tds | 11,939 | .528 | .595 | .714 | 8,241 | .294 | .319 | .522 |

No migrated tds row can fall below the 06-28 gate, because `gate + (1 − gate)·NB_cdf` has that floor. The natively written close spans the full range.

**Recovery.** Method:
- decode the stored `ev` under one family, with at most one dispersion parameter fitted on prices only;
- compare against a native reference on matched rows (same game_date, entity, book and line; 6,000 rows sampled);
- MAD is the median absolute difference. It includes genuine price moves between the stored observation and the reference.

| block | reference | stored − ref | decode that recovers it | mean diff | MAD | rmse |
|---|---|---|---|---|---|---|
| NFL tds 01:00, 2023–25 (06-02 fetch) | 23:00 close | +.133 | NegBin cv .586 | −.0015 | .0078 | .047 |
| NFL interceptions 01:00 | 23:00 close | +.326 | SkewNormal cv 1.02 (NegBin +.099) | −.0080 | .0287 | .058 |
| NFL passing tds 01:00 | 23:00 close | +.199 | SkewNormal cv .53 (NegBin +.060) | +.0022 | .0213 | .051 |
| NFL receptions 00:00, 2024–25 | 23:00 close | −.071 | NegBin cv .19 | +.005 | .018 | — |
| NBA BLK / STL / FG3M 01:00, 2025-26 | 23:00 close | +.25 / +.22 / +.16 | NegBin cv .55 / .54 / .57 | +.0001 | ≤ .001 | — |
| WNBA FG3M 01:00, 2025 | 23:00 close | +.235 | Gamma cv 1.04 (NegBin cv .76: +.012, .024) | +.0092 | .0233 | .070 |
| WNBA FG3M 00:00, 2023–25 | 23:00 close | +.071 | NegBin cv 3.43 | −.0013 | .018 | .030 |
| NHL goals 00:00, 2023–25 | native 01:00 | −.181 | Poisson | −.0002 | .0000 | .013 |
| MLB home runs 00:00, 2023–25 | native 01:00 | −.203 | Poisson | −.0006 | .0044 | .020 |
| NBA PTS 00:00, 2023-24 | native 01:00 | +.084 | none (SkewNormal at the cv bound ≡ 0.5: not a recovery) | | | |
| NBA BLK 00:00, 2023-24 | native 01:00 | −.020 | none better than the stored value | | | |
| NFL carries 00:00, 2024–26 | 23:00 close | −.20 to −.30 by month, MAD .30–.35 | none | | | |

**Code-history check.**
- **What reproduces.** At the 06-02 fetch, stat_meta set tds, interceptions and passing tds to ZINB. The ungated NegBin route reproduces tds.
- **What does not.** Interceptions and passing tds decode only under a continuous family; NegBin and Poisson leave +.06 to +.10. Their encoding path is therefore not reconstructed.
- **Unverified hypothesis.** A fall-through to the `"SkewNormal"` default in `stat_dist.get(league, {}).get(market_name, "SkewNormal")`.
- **Consequence for the harness.** It must identify the family from prices, with code history only as the prior.

**Interpretation.** The information is intact in `ev`; the defect is the 06-28 interpretation of it. A re-decode is a deterministic inversion, not an estimate of the book's level, and that is what makes it leak-free (Finding 7).

### Finding 2: a live writer defect prices duplicated same-side quotes at exactly 0.5 (Class II)

**Mechanism.**
- **The odds-row path.** In `_event_player_props`, one book's outcome rows for a player pass through `_resolve_prop_lines`:
  - rename Yes→Over;
  - collapse multi-point quotes;
  - trim to one Over/Under pair, but only when more than two rows exist.
- **Where the .5 comes from.** The rows then go to `no_vig_odds(*prices)`. Two rows on the same side, at the same point and price, pass as a pair, and `no_vig_odds(p, p)` returns [.5, .5].
- **The ladder path is correct.** `_devig_ladder` reads the same rows into a dict keyed by side, keeps one Over and de-vigs it one-sided.
- **The payload is not archived.** On this code path, equal same-side prices are the only route to an exact .5.

**Evidence (`a10`).** Every odds row with a same-observation ladder rung on the same line, all leagues. Pairs where `1 − under_prob` and the rung differ by more than .05:

| league | market | book | layer | rows | share at .5 against the ladder | mean diff |
|---|---|---|---|---|---|---|
| NHL | goals | betrivers | 01:00 | 224,894 | .239 | +.077 |
| NHL | goals | betrivers | 23:00 | 66,345 | .788 | +.255 |
| MLB | home runs | williamhill_us | 01:00 | 110,588 | .123 | +.042 |
| MLB | home runs | williamhill_us | 23:00 | 39,532 | .226 | +.087 |
| MLB | total bases | betrivers | 01:00 | 16,121 | .468 | +.059 |
| MLB | total bases | betrivers | 23:00 | 38,696 | .372 | +.051 |
| WNBA | FG3M | betrivers | 23:00 (evaluation only) | 802 | .481 | +.014 |

**Everything else agrees.**
- Matched row counts per league: 2.49M NHL, 5.43M MLB, 1.97M NBA, 288k NFL and 71k WNBA.
- Every other (market, book, layer) agrees to within ±.016 on average, with no .5 pattern.
- The NFL 23:00 pairs show 7–16% of rows differing by more than .05, but with a mean diff of about −.001. That is main-market against alternate-market pricing at one point: noise, not level.

**Example.** NHL goals, betrivers, January 2026, 01:00: 10,224 odds rows read exactly .5. Their ladder rungs average .1745, and none is at .5.

**By season (`a13`).**

| Book and market | Layer | Rows at .5 | Dates |
|---|---|---|---|
| williamhill_us, MLB home runs | 01:00 | 1,876 | 2025-08-06 → 11-01 |
| | 01:00 | 11,731 | 2026-03-25 → 07-09 |
| | live | 9,829 (7 have a ladder row) | 2026-07-11 → 10-26 |
| betrivers, NHL goals | 01:00 | 54,062 | 2025-10-07 → 2026-06-14 |
| | live | 774 | 2026-09-29 |

The defect is native, not a migration artifact. The MLB and NHL backfill layers were written by the 07-09/10 re-buy (model_improvement_track §3.2).

**Other books on the same player-game (`a9`).**
- williamhill_us MLB home runs: 22,758 of 29,090 .5 rows have other sportsbooks. Those books read a mean under of .823, and 98.9% read above .70.
- betrivers NHL goals: 90,824 of 106,587, mean .834, 88.8% above .70.

**Where it lands.**
- **Multi-book consensus is barely moved.** `fit_book_weights` already gives the defective book almost no weight (normalized):
  - williamhill_us on MLB home runs .006;
  - betrivers on NHL goals .001;
  - betrivers on MLB total bases .001.
- **The damage is on single-book rows.** It reaches the matrix where the defective book is the only sportsbook at the modal line (held-out priced rows):

| cell | rows at .5 | y on them | book CITL on them | the rest: rows / CITL | cell CITL |
|---|---|---|---|---|---|
| MLB home runs | 4,091 (27%) | .162 | +.338 | 11,098 / +.040 | +.120 |
| NHL goals | 1,217 (3.7%) | .122 | +.378 | 32,108 / +.0105 | +.024 |
| MLB total bases | 80 | .388 | +.113 | 14,918 / +.021 | +.022 |

**Why a mean re-level cannot fix it.** A mean re-level cannot remove a point mass at .5 on a quarter of the rows. That is the mechanism behind week 1's MLB home runs reading.

**What the bad rows are.**
- The NHL > .10 band in E4 is these rows: 994 NHL goals rows, book CITL +.271, book calibration slope .75, band BSS +.42.
- Of MLB home runs' bad rows, 90% sit inside the held-out window, so they feed Gate 1, the blend fit and the calibrators.

**Served test sets (`t3`).** The .18 stand-in is the average ladder rung; the fix uses each row's own rung.

| Cell | Rows at .5 | BSS on .5 rows | BSS on the rest | Cell BSS now | Cell BSS with .5 rows at .18 |
|---|---|---|---|---|---|
| MLB home runs | 2,048 of 7,572 (89 dates) | +.473 | +.027 | +.256 | +.017 |
| NHL goals | 601 of 16,621 (160 dates) | +.625 | +.000 | +.044 | +.002 |
| MLB total bases | 48 | — | — | +.0016 | +.0015 |

### Finding 3: on the served NFL count test sets, all offline skill lives on migrated rows

Served test CSVs, authentic rows, joined to the matrix on (Player, Date) for `QuoteObservedAt` (`t2`):

| cell | provenance | rows | dates | book CITL | BSS |
|---|---|---|---|---|---|
| NFL tds | migrated 01:00 | 2,751 | 72 | −.153 | +.168 |
| | native live 2026 | 742 | 17 | +.008 | −.037 |
| NFL interceptions | migrated 01:00 | 373 | 75 | −.343 | +.301 |
| | native live | 77 | 15 | +.034 | −.072 |
| NFL passing tds | migrated 01:00 | 373 | 75 | −.197 | +.144 |
| | native live | 78 | 15 | −.078 | +.009 |
| NFL receptions | migrated 00:00 | 2,047 | 84 | +.106 | +.059 |
| | native live | 417 | 17 | +.001 | −.010 |
| NFL carries | migrated 01:00 / 00:00 | 838 / 82 | 87 / 31 | +.043 / +.153 | +.026 / +.071 |
| | native live | 133 | 16 | −.006 | +.001 |
| NBA BLK / STL / FG3M | migrated 01:00 | 1,272 / 572 / 1,469 | 72–73 | −.225 / −.240 / −.177 | +.171 / +.203 / +.127 |
| NBA BLST / AST | migrated 00:00 | 84 / 1,186 | 22 / 57 | −.195 / +.069 | +.166 / +.034 |
| WNBA FG3M | migrated 01:00 | 292 | 46 | −.139 | +.084 |
| | live 2026 (mostly native) | 565 | 84 | −.031 | +.025 |

**Caveat.** The native rows come from a different season (2026) and only 15–17 dates. So this split bounds the post-fix skill rather than measuring it.

**Direction.** The direction matches week 1's live NFL quoted main-line BSS of −.0285 [−.0434, −.0144]. It also matches R3's "receptions w = .80 against b_model −.10; lead: the mis-levelled NFL count quote": receptions' validation rows are 00:00 migrated rows reading +.106.

### Finding 4: which rows are mis-levelled and which are fine

**Producer class is identifiable from the archive alone, without outcomes.**
- **Backfill rows.** A backfill row (01:00 feature layer, 23:00 close or 02:00 probe) has a same-observation ladder row if and only if code after `92f918ab` (2026-06-18, ladder capture) fetched it. The ladder share per (season, layer) block is exactly 0 or 1 in every block probed (`a12`):

| block | ladder share | reads as |
|---|---|---|
| NFL tds 01:00, 2023 / 2024 / 2025 | 0.000 / 0.000 / 0.000 | migrated (06-02 fetch) |
| NFL tds 23:00, 2024 / 2025 | 1.000 / 1.000 | native |
| NBA BLK 01:00, 2023 / 2025 | 1.000 / 0.000 | native / migrated |
| NBA BLK 23:00, 2024 / 2025 | 1.000 / 1.000 | native |
| WNBA FG3M 01:00, 2025 / 2026 | 0.000 / 0.413 | migrated / straddles the cutover |
| MLB home runs 01:00, 2023–2026 | 1.000 | native |
| every 00:00 block | 0.000 | migrated |

- **The straddling block.** WNBA 2026 01:00 needs the row-level test.
- **Live polls.**
  - A live poll is native if the WS1 writer wrote it.
  - Live polls rarely write ladder rows (NFL 2026 live .017), so they need the deploy boundary instead.
  - MLB shows a clean polling gap from 06-26 to 06-30.
  - WNBA straddles the boundary, so it must be pinned from the devel deploy history (run_job `PULL` lines), not from the commit time.
- **00:00 rows.** Every 00:00 row is migrated.

**The live-poll layers keep one encoding shape for months at a time (`a11`).** Stored WS1 value minus the native close, per calendar month, on matched rows:

| Market | Months | Offset | Pattern |
|---|---|---|---|
| NBA PTS | Oct–Feb | +.081 to +.084 (MAD ≈ the mean) | uniform |
| | March | +.040 | mixed |
| | April / May | −.001 / +.001 | level |
| NBA FG3M | Oct–Feb | +.070 to +.074 | uniform |
| | April / May | +.20 / +.29 | shifted |
| NBA REB, PR | Oct–Feb | +.061 to +.064 | uniform |
| | April–May | near level | |
| NBA AST | Oct–Feb | +.005 to +.007 | level |
| | March | −.060 | shifted |
| NFL receptions | every month, Sep 2024 → Feb 2026 | −.068 to −.092 | one shape across two seasons |
| WNBA PTS | May–Oct 2025 | +.084 to +.093 | uniform |
| WNBA FG3M | May–Oct 2025 | +.053 to +.077 | uniform |

Uniform offsets are re-decodable per epoch. The breaks mark config changes (for example NBA in March 2026).

**The consensus mixes producers within one player-game.** `get_training_quote_inputs` keeps each book's latest row at or before 12:00 UTC.
- **Mixing rate.** On three NHL goals dates in 2023-24 (`a14`), 20–23% of book-rows were migrated 00:00 rows. They touched 64–67% of players and lifted the unweighted consensus P(over) from .177–.182 (native only) to .206–.216.
- **Where it shows in the matrix.** The NHL goals matrix `Odds` reads .24–.29 from 2024-02 to 2025-03, then .16–.18 (y ≈ .15).
- **Consequence for the fix.** It must act per book-row, before the consensus forms.

**Which windows reach a consumer.** Rows in each window are mostly from these blocks; the census is in `m2`:
- **NFL** (held-out from 2024-12-08): migrated 01:00 plus 00:00, and native 2026 live.
- **NBA:** the 2025-26 season only. Every row predates the cutover; there are no native rows at all.
- **WNBA** (from 2025-08-30): migrated 2025 plus mixed 2026.
- **NHL** (from 2025-12-13) and **MLB** (from 2026-04-30): native rows plus Class II.
- **NHL count cells' train-era rows** feed nothing: the count HPO branch is model-only.

### Finding 5: what the error is not

- **The push label.** The tds, interceptions and passing tds lines are half-integers. Week 1's push share was about 0. A push-as-half label cannot move CITL by .1 to .3.
- **The line of record.** Count cells quote 0.5 or 1.5. The modal-line rule cannot shift a 0.5 line.
- **Devig method.**
  - Proportional and power devig differ by about .02 or less at these prices.
  - The one-sided 6.52% hold assumption does leave a small positive residual on longshot one-sided props:
    - NHL goals native rows +.0105;
    - MLB home runs non-.5 rows +.040;
    - MLB stolen bases +.043, single book.
  - This is the favourite–longshot pattern (Snowberg & Wolfers 2010; Clarke, Kovalchik & Ingram 2017; Štrumbelj 2014). It is small, positive, independent of era, and the wrong sign for the NFL negatives. It goes to Open questions.
- **Book mix.** Class I is uniform across books within a block. In NHL goals 2023-24, every book's 00:00 rows read under .59–.62, against .81–.83 at 01:00. Only Class II is book-specific.
- **Class III (genuine, not a conversion defect).**
  - NFL carries reads +.051 on 01:00 rows whose prices equal the native close to within +.0004. That is a historical book lean (books shade toward where bettors lean; Levitt 2004) or a population difference.
  - NFL attempts and completions read +.031/+.009 offline against −.063/−.048 live, on only 250 and 191 live rows.
  - Levelling will not move these cells and must not be tuned to them.

### Finding 6: how a mis-levelled leg reaches gating, the blend and the calibrators

**Skill against a biased reference.** A reference with bias δ pays about δ² in reliability. A skill score against it books that as the model's skill (Murphy 1973; Murphy 1988; Hamill & Juras 2006; Mason 2004). Week 1 measured exactly this: ρ .75 between offline BSS and |book CITL|.

**The blend weight runs to its clamp.**
- **How the weight is fit.** w is fit on validation authentic rows against `B_validation["EV"]`. `_authentic_quote` derives that EV from the mis-levelled `under_prob` at the current config, so the book's mean is off by the level error and loses the likelihood contest.
- **Values by cell:**

  | Cell | w |
  |---|---|
  | NFL interceptions | .900 |
  | NFL passing tds | .900 |
  | NBA STL | .900 |
  | WNBA FG3M | .900 |
  | NFL carries | .861 |
  | NBA BLST | .858 |
  | NBA FG3M | .840 |
  | NFL tds | .832 |
  | NFL receptions | .824 |
  | NBA BLK | .818 |
  | NHL goals | .761 |
  | MLB home runs | .723 |

- **Level cells for comparison:**

  | Cell | w |
  |---|---|
  | NFL attempts | .324 |
  | NBA AST | .385 |
  | MLB total bases | .377 |
  | MLB stolen bases | .475 |
  | NFL completions | .639 |

**Every consumer of the book leg.** None is a GBDT feature: `get_stat_columns` carries no book column.
1. Gate 1 reads the test authentic `Odds`.
2. The blend weight reads the validation authentic `EV`.
3. The fused validation and test outputs, and so every posthoc calibrator fit on them.
4. The SkewNormal HPO search gate. `_calibration_penalty` blends the train-era OOF pool with `B_train["EV"]` per trial, so on SkewNormal cells the train-era rows matter too. The count branch is model-only.
5. `fit_book_weights`, which reads the stored per-book `ev` via `to_pandas`.
6. `prob_recal_book_citl`, used only by NFL passing yards (withheld).

**For I6d.** A mismeasured book covariate in the encompassing regression attenuates b_book and shifts part of its signal onto b_model (errors-in-variables: Carroll et al. 2006; probability encompassing: Clements & Harvey 2010).
- The estimator rules already require model-only probabilities and the decision-time reference.
- NFL's decision-time book has no Class II rows.
- MLB's and NHL's do, so any encompassing read in those leagues needs the writer fix first.

### Finding 7: leakage audit of the candidates

"Leakage" here means any path by which outcomes, or post-decision prices, shape the book leg that Gate 1, w or the calibrators then score against. This follows Kaufman et al. 2012's legitimacy rule and learn–predict separation.

**(a) Re-derive from raw quotes with a corrected conversion: ADOPT (the core).**
- **Leakage path.** None, provided the conversion is chosen from prices alone. Raw payloads do not exist for migrated rows.
- **Leak-free form.**
  - (a1) re-decode the stored `ev` under a family identified from prices;
  - (a2) take Class II rows from the same-observation ladder.
- **Type.** Engineering for the backfill layers; research for the live-poll epochs.

**(b) A per-cell, per-era logit shift fit on training rows: KILL.**
- **The shift is an outcome fit.** It is mean recalibration (Cox 1958; Steyerberg et al. 2004).
- **The eras do not line up with the split.** The held-out windows mix migrated and native blocks, so:
  - a train-era shift is the wrong shift on held-out rows;
  - a held-out shift is fit on Gate 1's own rows.
- **It absorbs genuine bias.** It would also absorb Class III.
- **No leak-free form fixes the defect.** A shift cannot undo a gate floor (tds ≥ .81) or a .5 point mass on 27% of rows.
- **Type.** Research, and already measured.
- **Prior verdict.** Already a kill in `researcher_count_blend_location` §9: it cost MLB home runs, stolen bases and NHL hits their ship, and it carries a coherence trap.

**(c) Leave the mis-levelled era out of Gate 1 and w: FALLBACK ONLY (the demote branch).**
- **Leakage path.** It is leak-free only if "era" is defined by provenance, not by measured CITL. The leak-free form is demotion keyed on provenance.
- **Type.** Engineering.
- **Why not on its own.** It throws away rows the re-decode recovers exactly.
  - NFL interceptions and passing tds would keep 77 and 78 of about 450 test rows, on 15 dates.
  - NBA would keep zero.
- **The cluster floor.** Below `_ONE_SE_MIN_CLUSTERS` = 10 clusters, w becomes 1.0 and Gate 1 goes blank, which auto-passes.

**(d) Refit w and Gate 1 on levelled rows only: ADOPT as the retrain step.**
- **Leakage path.** None; it is the consequence of (a) plus (c).
- **Form.** The retrain itself, with a native-only Gate 1 read as a sensitivity check.
- **Type.** Engineering.

**(e) Re-level and relabel mis-levelled archive rows as `derived`: KILL the in-place form; ADOPT the read-time relabel.**
- **The relabel half** is (c).
- **Re-levelling in the archive** is either (b), or an edit that `merge_archives` undoes.
  - `merge_archives` keeps `MAX(under_prob)` per key (league, market, game_date, entity, book, ev, observed_at).
  - Every repair that lowers `under_prob` (every negative-CITL cell) loses to prod's row at the next sync. Repairs that raise it survive.
- **Form.** The relabel at read time, inside the recommendation.
- **Type.** Engineering.

**(f) A per-cell book-CITL monitor column: ADOPT, report-only.**
- **Leakage path.** Report-only is leak-free. A gate or selector built on it would be an outcome-based choice on held-out rows (Dwork et al. 2015).
- **Form.** Report-only, plus a provenance-mix column.
- **Type.** Engineering, small.

**(g) The 23:00 close as the book leg: KILL as the book leg.**
- **Leakage path.** It is a post-decision price.
- **Why it misleads.** It is sharper than the decision-time quote. That biases w toward the book, and it grades Gate 1 against a book that serving never sees. It breaks the as-of alignment.
- **Leak-free use.** As a validation reference only.

**(h) Re-buy the historical raw quotes: KILL this season.**
- **Leakage path.** None; the method is exact.
- **Why not.** The cost: 10 credits per market per event. The NFL count cells alone come to about 3 seasons × 285 games × 3 markets × 10 ≈ 26k credits, against about 1k left at reset.
- **Reopen trigger.** A budget cycle with slack.

**(i) Choose the decode family or cv by the book's fit to outcomes: KILL.** An outcome-fitted shape is (b) by another name; the leak-free alternative is to use prices only.

**Why (a1) is leak-free.**
- The stored `ev` was written at observation time, so it is pre-decision information.
- The shape is identified from native prices on matched rows, with at most one parameter per block of thousands of rows. A 23:00 reference can therefore pass on only its average level, which open and close share on matched rows (carries 01:00 against close: +.0004). No row-level close information passes through.
- Prefer a native 01:00 reference wherever one exists.

### Finding 8: the repair has to happen at read time

**Why the archive cannot be edited in place.**
- `merge_archives._odds_select` keeps `MAX(under_prob), MAX(line)` grouped by `_ODDS_KEY`.
- The training reader's tie-break (`under_prob DESC NULLS LAST`) also prefers the higher `under_prob`.
- So any in-place repair ends up half-applied after the next sync.

**What to do instead.** Use a read-time transform keyed on provenance, plus a committed decode table. That keeps the archive raw and makes the provenance explicit and reversible (where-provenance: Buneman, Khanna & Tan 2001).

**A stale statement to fix.**
- model_improvement_track §3.2 says "the book we grade against is real". That holds for native rows only and should be revised in place.
- The §6 note "the matrix build should read the WS-1 `under_prob` column directly" holds for native rows only.

## 3. Recommendation / routing protocol

### A. One candidate: the provenance-keyed book re-read

**Step 0: writer fix. Ship first, on its own push.** Engineering, about an hour plus a characterization test.
- In `_resolve_prop_lines`, key rows by side, as `_devig_ladder` does, so two same-side rows can never be priced as a pair. A lone Over de-vigs one-sided.
- This stops new Class II rows on the live board and in future backfills. `williamhill_us` MLB home runs are writing them today.

**Step 1: a producer class per archived book-row.** Outcome-blind; engineering.
- **native:** a backfill stamp with a same-observation ladder row, or a live poll written by the WS1 writer (boundary pinned from the deploy history on each box).
- **migrated:** everything else.
- **Implementation:** precompute a per-(league, market, layer, season) table, plus a row-level ladder test for straddling blocks.

**Step 2: Class II repair on native rows.**
- **With a ladder row.** Where `1 − under_prob` and the same-observation ladder rung on the same line differ by more than .05, take the rung.
- **Without one (live polls).** Drop the book-row from the direct cohort when it reads exactly .5 while the other sportsbooks on the same player-game read under above .70.
- Both rules use prices only.

**Step 3: Class I migrated blocks. Re-decode, keep, or demote.**
- **What a block is.**
  - For backfill layers: (league, market, layer, producer run).
  - For live-poll layers (00:00 seeds and pre-cutover live): (league, market, epoch). An epoch is a calendar month by default, split at config changes. NBA March 2026 is mixed.
- **The test.**
  1. Match native rows. The reference is the native 01:00 layer where one exists, otherwise the native 23:00 close with its Class II rows removed.
  2. Split the matched rows in half by date.
  3. On the first half, fit at most one dispersion parameter per candidate, in this order:
     - the stored WS1 decode (no parameter);
     - the code-history family at the fetch date;
     - SkewNormal, NegBin, Poisson, then Gamma.
  4. On the second half, accept the first candidate with |mean diff| ≤ .01, MAD ≤ .03, and an interior parameter.
- **On acceptance.** At read time, set `under_prob := get_odds(line, ev, family, cv, step)`. The row stays `authentic`, with `QuoteSource = book_redecoded`.
- **On failure.**
  - The book-row is not a price.
  - It leaves the direct cohort, and also skips the ev-inversion rung, which decodes under today's config and so repeats WS1's error.
  - A player-game with no admissible book-row becomes non-authentic and claims no level.
- **Blocks with no native overlap.**
  - They inherit the verdict of a validated block from the same producer run (NFL 2023 01:00 inherits from the same 06-02 fetch's 2024–25 rows).
  - Live-poll epochs without overlap are demoted.

**Step 4: the w guard. An owner call, before the retrain.**
- **The hazard.** Below `_ONE_SE_MIN_CLUSTERS` = 10 authentic validation clusters, `_blend_fit_supported` sets w = 1.0 ("serving model-only") and `_priced_rows` blanks Gate 1, which auto-passes. That is the no-evidence-full-trust pattern.
- **Rule to adopt.** A blank Gate 1 means withheld, never pass.
- **Where it binds.**
  - Under this recommendation it should not bind on any served cell.
  - Under stand-alone (c) it binds on every NBA cell.

**Step 5: re-resolve and retrain.**
1. Check first with `inject_backfilled_odds --all-cached --dry-run`; it reports the repair delta and writes nothing.
2. Re-resolve every cached matrix's quote block with `inject_backfilled_odds --all-cached`.
3. Refit `fit_book_weights`. Its inputs are stored per-book `ev`, so check it does not concentrate weight on books whose history is mostly demoted.
4. Retrain each league on the campaign recipe. This refits w, the calibrators, the SkewNormal HPO gate and Gate 1.

**Step 6: monitor, report-only.** Add to `model_stats`, read by no gate:
- per-cell `book_citl_offline`;
- its provenance split (native, re-decoded, ladder-repaired);
- the provenance share.

### B. Per-block routing

From the measured blocks; the harness decides the rest.

| block | route | evidence |
|---|---|---|
| NFL tds 01:00 (06-02) | re-decode NegBin cv ≈ .59 | −.0015 / .0078 |
| NFL interceptions 01:00 | re-decode SkewNormal cv ≈ 1.02 | −.0080 / .0287 (MAD .0013 under the bar) |
| NFL passing tds 01:00 | re-decode SkewNormal cv ≈ .53 | +.0022 / .0213 |
| NFL carries 01:00 | keep stored | equals the close to +.0004 |
| NFL receptions 00:00 | re-decode NegBin cv ≈ .19, one epoch | uniform −.07 every month |
| NFL carries and rushing yards 00:00 | demote | no family fits |
| NBA BLK / STL / FG3M 01:00, 2025-26 | re-decode NegBin cv ≈ .55 | +.0001 / ≤ .001 |
| NBA 00:00 and pre-cutover live, 2025-26 | per-epoch validation against the 2025 close | Oct–Feb uniform; March mixed; Apr–May varies |
| WNBA FG3M 01:00, 2025 | code-history family (ZINB) first, then Gamma or NegBin | borderline: +.009–.012 / .023, rmse .07 |
| WNBA 00:00, 2023–25 | per-epoch validation | uniform +.05 to +.09 |
| NHL goals 00:00 | re-decode Poisson | −.0002 / .0000 |
| NHL goals betrivers; MLB home runs williamhill_us; MLB total bases betrivers | ladder repair | 12–79% of rows |
| MLB home runs williamhill_us, live | drop the .5 rows; writer fix | 9,829 rows since 07-11 |
| MLB home runs 00:00 | re-decode Poisson | −.0006 / .0044 |

### C. Pre-registration

- **Freeze before the harness runs:** the tolerances, the candidate order, the reference choice and the epoch rule, in a committed file.
- **Record:** every block's numbers.
- **Never** retune after an outcome read (§4 M1).
- **Changing a bar** means a new pre-registration and a full re-run.
- **The owner sets the MAD bar.** The NFL interceptions verdict hangs on it, since its MAD of .0287 sits .0013 under .03.

### D. KILL verdicts

- **(b) The per-cell, per-era logit shift.** Outcome-fitted, cannot fix the defect shapes, and already killed in §9.
- **(g) The close as the book leg.** A post-decision price.
- **(e, in place) Archive edits.** Undone by the `MAX` merge.
- **(c) Stand-alone era exclusion.** Discards exactly recoverable rows and triggers the w = 1.0 floor.
- **(h) Re-buying raw quotes.** About 26k credits against about 1k.
- **(i) Any decode choice fitted on outcomes.**

### E. Cells touched and build cost

**Cells touched.** Every cell's matrix is re-resolved, because 00:00 rows exist in every league. The level moves materially (|ΔCITL| > .03 expected) on:
- **NFL:** tds, interceptions, passing tds, receptions; also carries and rushing yards on a few rows.
- **NBA:** BLK, STL, FG3M, BLST, and epoch by epoch the 00:00-dominated SkewNormal cells (AST, PR, PRA, REB, RA, PA).
- **WNBA:** FG3M, and the 00:00-dominated SkewNormal cells.
- **NHL:** goals.
- **MLB:** home runs, and total bases slightly.

**Build cost.**

| piece | size |
|---|---|
| Step 0, writer fix and test | S (≤ 1 h) |
| Step 1, provenance table and reader filter | M (½ day) |
| Step 2, ladder repair and the exact-.5 rule | S–M (½ day) |
| Step 3, harness (fit and validate, writes the committed decode table) and read-time decode | M (1–1½ days) |
| Step 5, re-inject (minutes per league) and retrain | one overnight per league |
| Step 6, monitor columns | S |

**Total:** about 2–3 dev-days plus one retrain cycle per league.

**The research part is the live-poll epochs.** Recovery is not guaranteed. An NBA epoch that fails demotes, and because NBA has no native rows, those rows are lost until 2026-27 native rows accrue.

## 4. Measurement (pre-registered)

### M0: checks on prices alone, before anything is consumed

- **(a) Decode test.** Every re-decoded block passes its validation half (|mean| ≤ .01, MAD ≤ .03, interior parameter). A failing block is demoted, not retuned.
- **(b) Writer consistency.** After Step 2, at most 1% of ladder-matched native rows in any (market, book, layer) disagree with their rung by more than .05. Today the four flagged pairs sit at .12 to .79.
- **(c) Writer fix in production.** One week after Step 0 ships, the live archive holds zero exact-.5 `williamhill_us` MLB home-run rows or betrivers NHL goals rows where the other books read above .70.

### M1: the read that decides. Offline book level against live, before the retrain.

Outcomes enter, so read it once, after M0.
- **(i) NFL.** Re-run H3's T1 (`e3/h3_t1`) on the same live population. L_off over the seven NFL count cells must be ≤ .03, and the offline-minus-live D 95% CI must include 0. Today D is .0827 [.0504, .0990].
- **(ii) Every served cell with at least 100 priced held-out rows.** |book CITL_off| ≤ .03. Pre-named exceptions are reported, not failed:
  - one-sided longshot props (MLB home runs, MLB stolen bases, NHL goals), allowed up to +.05;
  - Class III cells (NFL carries, attempts, completions).
- **(iii) NHL served > .10 band.** The 95% lower bound of its BSS falls below +.10. Today it is +.284.
- **(iv) Report only.** Spearman ρ between offline BSS and |book CITL| (today .75). Per cell, BSS on levelled-migrated rows against native rows (the `t2` split).

### M2: after the retrain

- **The w guard.** No served cell shows a blank Gate 1 from the cluster floor unless the owner has withheld it.
- **Gate 1.** The standard date-clustered CI (ci_hi < .005), plus a native-only sensitivity read wherever native rows reach 10 or more clusters.
- **Report.** w per cell.
- **Tail scorecard.** The weekly per-cell offline-against-live gap ρ. The promotion rule is unchanged: four agreeing weeks.

### M3: live, with the fixed acceptance

- **I6d.** Fixed-effect logistic encompassing, b_model's day-clustered lower bound above 0, plus the tail scorecard. Model-only probabilities and the decision-time reference, per the estimator rules.
- **Report only, not a gate.** Live BSS of the recommended population per league, day-clustered. Use a wild cluster bootstrap under about 30 days (MacKinnon & Webb 2017).

### Decision rules

- **An M0 failure on a block** demotes that block, as pre-declared.
- **M1 passes:** retrain (M2), then send per league on live-gap-2 §5's cadence.
- **M1 fails while M0 passes:** a mis-levelling class not found here remains. Stop and do not retrain on it; diagnose with `t2`, `a10` and `a11`.
- **Never use M1 to pick between levelling variants.** It reads held-out outcomes (Dwork et al. 2015).

## 5. Reality checks

**Projected effects in this regime.** The "after" values are projections from the decode diagnostics and the `t2`/`t3` splits, not measured post-fix numbers:

| cell | held-out book CITL now | projected after re-read | basis | model_stats BSS now | w now |
|---|---|---|---|---|---|
| NFL interceptions | −.253 | ≈ 0 ±.02 | 718 of 870 held rows re-decode (−.008); native +.021 | .218 | .900 |
| NFL passing tds | −.186 | ≈ 0 ±.02 | 719 of 874 (+.002); native −.022 | .185 | .900 |
| NFL tds | −.120 | ≈ 0 | 5,551 of 6,990 (−.0015); native +.009 | .144 | .832 |
| NFL receptions | +.085 | ≈ 0 | 4,182 of 5,117 (+.005); native −.002 | .049 | .824 |
| NFL carries | +.055 | ≈ +.04 (Class III stays) | 1,644 01:00 rows kept; 166 00:00 rows demoted | .048 | .861 |
| NBA BLK / STL / FG3M | −.206 / −.205 / −.171 | ≈ 0 | 2,549 / 1,149 / 2,980 01:00 rows (+.0001) | .150 / .146 / .127 | .818 / .900 / .840 |
| NBA BLST | −.168 | ≈ 0 if its 00:00 epochs validate; otherwise most of its 248 held rows are lost | 179 of 248 rows are 00:00 | .128 | .858 |
| WNBA FG3M | −.053 | ≈ −.02 | 567 01:00 rows (+.009–.012) | .071 | .900 |
| NHL goals | +.024 | ≈ +.01 | 1,217 .5 rows to the ladder; test BSS +.044 → ≈ +.002 | .034 | .761 |
| MLB home runs | +.120 | ≈ +.03 | 4,091 .5 rows to the ladder; test BSS +.256 → ≈ +.017 | .227 | .723 |

**Expected consequences of the projections.**
- **BSS.** BSS on the Class I cells moves to about −.05…+.05. The anchors are week 1's re-levelled .005–.052 and native-row BSS of −.072 to +.009.
- **w.** w falls from .72–.90 toward riding the book.
- **Gate 1.** It mostly keeps passing, since a fused probability near a level book has a Brier difference near 0 and a tight CI.
- **Recommended volume and the recommended gap.** Both shrink on these cells. Week 1 measured that gap at +.10 to +.14 with ROI of −8% to −19%.
- **Noise ceiling.** Residual BSS inflation from decode noise is at most rmse²/BS_book: about .013 for NFL tds, interceptions and passing tds, and negligible for NBA 01:00.

**Where it holds and where it does not.**
- **Exact recoveries:** NHL and MLB 00:00, NBA 2025 01:00, NFL tds. Close recoveries: NFL interceptions and passing tds.
- **Partial (rmse .07):** WNBA FG3M 01:00.
- **Not recoverable:** NFL carries and rushing yards 00:00.
- **Partly research:** the NBA and WNBA live-poll epochs.

**Build cost.** As in §3E: about 2–3 dev-days plus one retrain per league.

**What would make the recommendation wrong.**
1. **The native reference is itself off level.** The re-decode would inherit that bias. Native rows' CITL is small (Finding 3), and M1 would catch it.
2. **The close passes on information.** If line moves within a block were one-directional, the fitted parameter would absorb them. Prefer native 01:00 references.
3. **Shape drift inside a block.** Per-row noise would persist (rmse .05–.07) and mildly inflate b_model in I6d. The MAD bar bounds this.
4. **Class III is larger than seen here.** Then M1(i) can fail on carries, attempts and completions. That is not a reason to tune the decode.
5. **`fit_book_weights` is fit on stored `ev`.** After the re-read the weights move, and the consensus with them.
6. **The live board carries Class II rows.** MLB and NHL decision-time quotes are contaminated until Step 0 ships.
7. **Forking paths.** I chose the tolerances after viewing blocks, so the owner must freeze them first. The NFL interceptions verdict hangs on the MAD bar.
8. **The inheritance assumption fails.** NFL 2023 01:00 assumes the 06-02 fetch used one shape across all seasons. No fetch log was found on disk (`archive/backfill_progress.json` is absent).

## 6. Open questions / caveats

- **The encoding path for NFL interceptions and passing tds.** stat_meta said ZINB, but only a continuous family decodes them. The `"SkewNormal"` market-name fall-through hypothesis is unverified.
- **Epoch boundaries for the live-poll layers.** Options: change-points in the stored-against-close offset, or the dates of meditate runs. The `stat_calibration.json` history is gitignored.
- **What encoded NFL carries and rushing yards 00:00.** Their MAD is .30–.35 under every family.
- **The one-sided longshot devig convention.** It leaves +.01 to +.04 on home runs, stolen bases, anytime goal and anytime TD. Power or Shin devig needs two-sided prices, and a fixed one-sided hold is a convention, not an estimate. Research; not in scope.
- **Class III on NFL carries, attempts and completions.** Regime change or population difference; it needs its own analysis.
- **Book weights.** `book_weights.json` was rewritten today (10-27 08:42), and large weights sit on books that exist only in old data (NFL tds barstool .59, NHL goals pointsbetus .57). Refit after the re-read.
- **NBA 2026-27 native rows** will enter the matrices as the season accrues, and the re-decoded share falls with them.
- **The deploy boundary for live polls** on each box (dev and prod) needs pinning from the PULL logs, not the commit time.

## 7. Bibliography

Every identifier I checked was confirmed via Crossref or doi.org this session; the base column names prior briefs whose identifiers I reused without re-checking.

| # | Reference | Identifier |
|---|---|---|
| 1 | Murphy, A. H. (1973). A new vector partition of the probability score. *J. Appl. Meteorol.* 12(4):595–600 | doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2 |
| 2 | Murphy, A. H. (1988). Skill scores based on the mean square error and their relationships to the correlation coefficient. *Mon. Wea. Rev.* 116(12):2417–2424 | doi:10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2 |
| 3 | Hamill, T. M. & Juras, J. (2006). Measuring forecast skill: is it real skill or is it the varying climatology? *QJRMS* 132(621C):2905–2923 | doi:10.1256/qj.06.25 |
| 4 | Mason, S. J. (2004). On using "climatology" as a reference strategy in the Brier and ranked probability skill scores. *Mon. Wea. Rev.* 132:1891–1895 | doi:10.1175/1520-0493(2004)132<1891:OUCAAR>2.0.CO;2 |
| 5 | Bröcker, J. & Smith, L. A. (2007). Scoring probabilistic forecasts: the importance of being proper. *Wea. Forecasting* 22(2):382–388 | doi:10.1175/WAF966.1 |
| 6 | Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM TKDD* 6(4):15 | doi:10.1145/2382577.2382579 |
| 7 | Croushore, D. & Stark, T. (2001). A real-time data set for macroeconomists. *J. Econometrics* 105(1):111–130 | doi:10.1016/S0304-4076(01)00072-0 |
| 8 | Buneman, P., Khanna, S. & Tan, W.-C. (2001). Why and where: a characterization of data provenance. *ICDT 2001*, LNCS 1973:316–330 | doi:10.1007/3-540-44503-X_20 |
| 9 | Weyuker, E. J. (1982). On testing non-testable programs. *The Computer Journal* 25(4):465–470 | doi:10.1093/comjnl/25.4.465 |
| 10 | Barr, E. T., Harman, M., McMinn, P., Shahbaz, M. & Yoo, S. (2015). The oracle problem in software testing: a survey. *IEEE TSE* 41(5):507–525 | doi:10.1109/TSE.2014.2372785 |
| 11 | Cox, D. R. (1958). Two further applications of a model for binary regression. *Biometrika* 45(3–4):562–565 | doi:10.1093/biomet/45.3-4.562 |
| 12 | Steyerberg, E. W., Borsboom, G. J. J. M., van Houwelingen, H. C., Eijkemans, M. J. C. & Habbema, J. D. F. (2004). Validation and updating of predictive logistic regression models. *Stat. Med.* 23(16):2567–2586 | doi:10.1002/sim.1844 |
| 13 | Van Calster, B. et al. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Med.* 17:230 (base B18) | doi:10.1186/s12916-019-1466-7 |
| 14 | Carroll, R. J., Ruppert, D., Stefanski, L. A. & Crainiceanu, C. M. (2006). *Measurement Error in Nonlinear Models*, 2nd ed. Chapman & Hall/CRC | doi:10.1201/9781420010138 |
| 15 | Clements, M. P. & Harvey, D. I. (2010). Forecast encompassing tests and probability forecasts. *J. Appl. Econometrics* 25(6):1028–1062 (base B29) | doi:10.1002/jae.1097 |
| 16 | Harvey, D. I., Leybourne, S. J. & Newbold, P. (1998). Tests for forecast encompassing. *JBES* 16(2):254–259 | doi:10.1080/07350015.1998.10524759 |
| 17 | Štrumbelj, E. (2014). On determining probability forecasts from betting odds. *IJF* 30(4):934–943 | doi:10.1016/j.ijforecast.2014.02.008 |
| 18 | Shin, H. S. (1993). Measuring the incidence of insider trading in a market for state-contingent claims. *Econ. J.* (refs [73]) | doi:10.2307/2234240 |
| 19 | Clarke, S. R., Kovalchik, S. & Ingram, M. (2017). Adjusting bookmaker's odds to allow for overround. *Am. J. Sports Sci.* 5(6):45–49 | doi:10.11648/j.ajss.20170506.12 |
| 20 | Snowberg, E. & Wolfers, J. (2010). Explaining the favorite–long shot bias: is it risk-love or misperceptions? *JPE* 118(4):723–746 | doi:10.1086/655844 |
| 21 | Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? *Econ. J.* 114(495):223–246 | doi:10.1111/j.1468-0297.2004.00207.x |
| 22 | Dwork, C. et al. (2015). The reusable holdout: preserving validity in adaptive data analysis. *Science* 349:636–638 (base B4) | doi:10.1126/science.aaa9375 |
| 23 | Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *REStat* 90(3):414–427 (base B19) | doi:10.1162/rest.90.3.414 |
| 24 | MacKinnon, J. G. & Webb, M. D. (2017). Wild bootstrap inference for wildly different cluster sizes. *J. Appl. Econometrics* 32(2):233–254 (base B22) | doi:10.1002/jae.2508 |
| 25 | Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102:359–378 (base B15) | doi:10.1198/016214506000001437 |

## Appendix A: re-runnable commands

All commands are read-only and run from the repo root. Each needs these two settings:
- `S=~/backups/sportstradamus/2026-10-21-live-gap-2/research/i6d_scripts`
- the prefix `SPORTSTRADAMUS_ARCHIVE_READ_ONLY=1 nice -n 10 .venv/bin/python`

The archive scripts open only the backup copy, read-only, with 4 threads.

| Command | Produces |
|---|---|
| `$S/a4_family.py NFL "tds,interceptions,passing tds" 1` | Finding 1 recovery against the close; also NBA `"BLK,STL,FG3M" 1`, WNBA `"FG3M" 1` and `"FG3M" 0` |
| `$S/a4b_family_ref.py NHL "goals" 0 1`; `$S/a4b_family_ref.py MLB "home runs" 0 1`; `$S/a4b_family_ref.py NBA "BLK,STL,PTS" 0 1` | recovery against the native 01:00 layer |
| `$S/a5_layers.py NBA "BLK,STL,FG3M" 9` | layer census |
| `$S/a12_ladder_share.py NFL tds 8` (also NBA BLK 9; WNBA FG3M 1; MLB "home runs" 1) | provenance discriminator |
| `$S/a10_ladder_consistency.py <LEAGUE>` | Class II writer check, all leagues |
| `$S/a13_half_by_season.py MLB "home runs"`; `$S/a13_half_by_season.py NHL goals` | Class II by season |
| `$S/a9_halfbook.py <LEAGUE> "<market>" <book> <start> <end>` | other books when the defective book reads .5 |
| `$S/a11_month_blocks.py NBA "AST,PR,REB,PTS,BLST,FG3M" 2025-10-01 2026-06-30` (also NFL `"receptions,carries" 2024-09-01 2026-02-28`; WNBA `"FG3M,PTS" 2025-05-01 2025-10-31`) | live-poll epochs |
| `$S/a14_consensus_mix.py NHL goals 0.5 2024-01-15 2024-02-20 2023-11-10` | consensus mixing |
| `$S/m1_matrix_eras.py NFL "tds,interceptions,passing tds"`; `$S/m2_matrix_held.py <LEAGUE> "<cells>"` | matrix census |
| `$S/t2_stamp_bss.py NFL "tds,interceptions,passing tds,receptions,carries"` (also NBA and WNBA) | Finding 3 |
| `$S/t3_half_split.py MLB_home-runs`; `$S/t3_half_split.py NHL_goals` | Class II effect on the served test set |
| `$S/t1_bands.py <LEAGUE> "<cells>"` | the > .10 band anatomy |

## Load-bearing conclusions for the plan

1. **Root cause of H3: two defects, neither devig, push nor line of record.**
   - **Class I.** The WS1 migration decoded every row written before 06-28 under the 06-28 shape, and its round-trip check cannot see that. The stored `ev` still holds the right value.
   - **Class II.** A live writer bug: a duplicated same-side quote becomes exactly .5, while the same-fetch ladder reads about .18.
   - **Where.** Copy into live-gap-2.md §5 (week 2, "the book-leg levelling it designs"), and as a new §8.2 entry closing the "historical book leg" hole. Separately, revise model_improvement_track §3.2's "the book we grade against is real" in place: it is true for native rows only.
2. **Ship the writer fix first, alone.** In `_resolve_prop_lines`, key rows by side, as `_devig_ladder` does. `williamhill_us` MLB home runs (9,829 live rows since 07-11) and betrivers NHL goals contaminate the live decision-time book until it lands. **Where:** live-gap-2.md §5 week 2, as the first item.
3. **The one recommended candidate is the provenance-keyed re-read at training read time.**
   - **Steps.** Producer class from the ladder and the deploy boundary; Class II ladder repair; per-block re-decode, keep, or demote under a pre-registered price-only test (|mean| ≤ .01, MAD ≤ .03); no archive edit; re-inject and retrain per league.
   - **Cost.** About 2–3 dev-days plus one retrain per league.
   - **Where.** live-gap-2.md §5 week 2 (build and order), and §8.2 (the hole's resolution).
4. **Lever verdicts.**
   - KILL the per-cell, per-era logit shift on the book leg (outcome-fitted; cannot undo a gate floor or a .5 point mass).
   - KILL the 23:00 close as the training book leg (post-decision).
   - KILL in-place archive repair (`merge_archives` `MAX(under_prob)` undoes lowering edits).
   - KILL stand-alone era exclusion (NBA would keep zero authentic rows; NFL interceptions and passing tds about 80).
   - KILL re-buying historical quotes this season (about 26k credits against about 1k).
   - Book-CITL column: ALIVE, report-only.
   - **Where.** model_improvement_track §3.4 (lever-verdict rows).
5. **Measured: on the NFL count test sets all offline skill sits on migrated rows.** BSS +.14 to +.30 on migrated rows against −.07 to +.01 on native rows. MLB home runs' +.256 and NHL goals' +.044 fall to about +.017 and +.002 once the .5 rows are priced. **Where:** §8.2 (#0d / I6d entry, as evidence) and live-gap-2.md §5.
6. **The deciding read, with floors, before any retrain.**
   - H3 L_off ≤ .03 on the seven NFL count cells, with the offline-minus-live D CI covering 0.
   - Per served cell |book CITL| ≤ .03, with pre-named exceptions.
   - NHL > .10 band BSS lower bound below +.10.
   - Outcome-blind decode and writer checks first.
   - Never used to choose between variants.
   - **Where.** live-gap-2.md §5, with a pointer from §6's live-reads table.
7. **The w guard (owner call before the retrain).** Below 10 authentic validation clusters, w = 1.0 and Gate 1 goes blank and auto-passes. Adopt "blank Gate 1 = withheld". **Where:** §8.2 #0d rules ("abstain is the no-fit default").
8. **Expectation to set.** After levelling, w on these cells falls toward riding the book, and Gate 1 will mostly still pass because non-inferiority lets the fused probability ride a level book. I6d's fixed encompassing test and the tail scorecard decide the skill; Gate 1 cannot. **Where:** live-gap-2.md §5, and §8.2's I6d entry.
9. **New research holes.**
   - Epoch-level decoding of the live-poll layers (NBA March 2026 is mixed).
   - The encoding path for NFL interceptions and passing tds (configured ZINB, decodes as SkewNormal).
   - The unrecoverable NFL carries and rushing yards 00:00 rows.
   - The one-sided longshot devig residual (+.01 to +.04).
   - Class III on NFL carries, attempts and completions.
   - Book weights fit on stored `ev`.
   - **Where.** model_improvement_track §8.2 (open holes).
