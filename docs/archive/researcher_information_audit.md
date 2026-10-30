# In-repo research brief — information audit: what moves player-prop lines that we do not read, and which of it can we collect both live and historically (research-analyst, 2026-10-29)

**Question (owner, 2026-10-29 evening, verbatim):** "I thought we had tapped out the information that's available out there. What else can be done to get more of an edge? Is there any information I'm missing or another way to process it that would give us a breakthrough?" Binding constraint, verbatim: "keep it grounded in what I am actually able to collect in real time and historical, for training."

**Decision this brief unblocks.** Which new sources, or which new processing of stored data, get collectors and features, and in what order. Ranking rule (dispatch): information beyond the decision-time market line × collectability of both halves (live before our scoring hour; point-in-time history for training) ÷ build cost.

**Scope.** 2026-10-29, 21:25–23:20 CDT. Read-only:
- Nothing under `src/` was written. No meditate, ship, pickle, `stat_meta.json` entry, plan, doc or git ref was touched.
- The archive was read only through the copy `/tmp/arch_nhl_i6f_20261029.duckdb` (read-only connection, 2 threads).
- Every external source got one small request (r08). Each of the two MLB statsapi schedule calls returns a whole season in one response.
- Scripts and outputs: `$L/research/information_audit_reads/` r01–r12, where `$L` = `/home/trevor/backups/sportstradamus/2026-10-21-live-gap-2`. Downloads sit in `ext/` subfolders and were parsed with `python -I`.
- Sibling briefs are cited, not repeated: `researcher_gates_vs_pickem_profit.md` (selection levers L1–L3) and `researcher_model_class_and_edge.md` (learner, pooling, pivot, restart). This is the inputs brief.

**Conventions.**
- pp = percentage points of win probability at our line. A positive move is toward our side.
- Decision price = the sportsbook consensus when we pick; for test-set rows, training's noon-UTC (7 a.m. CDT) read. Close = the last read before the lock (sportsbooks) or Underdog's last stamp.
- Consensus instrument (r02, r03, r06, r09): per book, fair line f = line + s·Φ⁻¹(P(over)), with s = the player's trailing-20 SD; median over books; probability at the consensus line. MLB reads (r10, r11) use the median P(over) at the prop's modal line.
- res = outcome − price. Game-level tests treat each game as one unit.
- Intervals are 90 % date-cluster bootstraps or date-clustered SEs (Cameron, Gelbach & Miller 2008, doi:10.1162/rest.90.3.414) unless stated.
- Break-even .5464 at the 1.83 payout; count-once from 2026-10-09 (`realized.settled_offers`).
- Owner-facing times are CDT.

**Information grades** (information beyond the decision-time market line):
- **A** — beats the price in-house and replicates on a holdout with point-in-time inputs.
- **B** — beats the price in-house and replicates on a holdout, but measured on game-time actuals (leak-bounded).
- **C** — small (about 1 pp per SD or less), or helps the model only where the market already prices it.
- **D** — tested here and null against both the price and the model.
- **E** — not tested here; literature or prior only.

Research project = unproven, may not transfer. Engineering project = known method, build cost given.

---

## TL;DR (plain English)

1. Yes, the model misses game-day information the market has — mostly who is playing: in the NBA, when important teammates are out (41 % of rows), the model puts a player's chance of going over 4.5 percentage points below the closing line's, because it does not know about the absences — but the sportsbook price at the moment we pick already contains it, and replaying past NBA picks with that information moved the hit rate between −2 and +3 percentage points, none distinguishable from zero.
2. The market does not turn against our picks after we pick: across 1,182 recommended legs since 10-09, the pick'em apps' price moved 0.34 percentage points against us on average and 84–99 % did not move at all; the damage happens before we pick, when in a third of recommended legs the price had already moved away from our side, and those legs won 49 % against 57 % when it had moved our way.
3. Of everything we can collect both live and historically, one input showed information the books missed: wind at outdoor NFL games, where passing Overs fell 12–15 percentage points further short of the sportsbook price in games with 15 mph or more of wind than in calm games, in 2024–25 and again in 2022–23, seasons no earlier read had touched (receiving Overs 5–6 percentage points).
4. Wind is collectable on both halves (free forecasts live; forecast archives back to 2021–2024 for training), but it touches about 15–20 games a season, mostly November to January, and it must still pass a re-test with forecasts instead of measured game-time wind, so it is a narrow edge, not a breakthrough.
5. Everything else tested adds about 1 percentage point per standard deviation or less beyond the price when we pick: the NFL injury report, NHL goalies and scratches, MLB umpires and weather, line movement in the NBA and WNBA, and "predict the closing line", where the close recovers at most a tenth of our disagreement, and none in the NBA.
6. Underdog's injury flags and line suspensions arrive in every poll we already make and are thrown away, and nobody else keeps their history; storing them, with Underdog's popularity counts and the NBA's pre-game lineup states, costs two to three days and is the only way to test them next season.
7. Bottom line: no collectible source is a breakthrough; the honest gains are one narrow NFL weather edge, cleaner but not more profitable NBA models, and archives that start paying next season.

---

## Key findings

### 1. The line does not move against our picks after we pick; it moved before (the owner's read)

**Sample (r02).** 5,732 legs at the 1.83 payout since 10-09 (NBA, NFL, NHL, scored before kickoff), 1,182 of them recommended (side probability ≥ .574), counted once. Recommended legs are graded at their last scoring, a median 0.16 h (10 min) before lock.

**Post-decision move (last scoring → close), signed to our side:**

| League | Platform n | Platform move, pp [90 %] | Share flat / against | Books n | Books move, pp [90 %] |
|---|---|---|---|---|---|
| NBA | 460 | −0.42 [−0.60, −0.30] | .841 / .102 | 320 | −0.30 [−0.87, +0.02] |
| NFL | 319 | −0.04 [−0.10, −0.01] | .994 / .006 | 165 | +0.15 [+0.00, +0.55] |
| NHL | 376 | −0.47 [−0.79, −0.19] | .957 / .043 | 47 | +0.00 [−0.04, +0.05] |
| All | 1,155 | −0.34 [−0.50, −0.21] | .921 / .056 | 532 | −0.14 [−0.44, +0.08] |

- Non-recommended legs drift −0.07 pp [−0.12, −0.04] on the platform.
- From the *first* scoring, for legs first scored well before their last: platform −0.94 pp (NBA, 312 legs, first lead 9.2 h), −0.13 (NFL, 54, 26 h), −2.99 (NHL, 87, 9.2 h); books −0.59, −5.39, −1.01.
- Dropping the legs whose market later moved against us does not help: platform .520 → .517 (the 65 "against" legs hit .569).
- Against a 6–11 pp gap between claimed and realized, post-decision news explains at most ~0.3–0.5 pp. **Missing post-decision information is not the proximate cause of the live gap.**

**Pre-decision move (platform's first price → our decision), r02b:**

| League | n | Mean move, pp [90 %] | Fade share | Ride share | Hit when fading [90 %] | Hit flat | Hit riding |
|---|---|---|---|---|---|---|---|
| NBA | 460 | −0.89 [−1.39, −0.59] | .361 | .196 | .488 [.436, .521] | .505 | .578 |
| NFL | 319 | −1.89 [−2.52, −1.56] | .476 | .210 | .513 [.429, .544] | .480 | .552 |
| NHL | 376 | −0.41 [−0.89, +0.07] | .125 | .112 | .447 [.333, .543] | .547 | .571 |
| All | 1,155 | −1.01 [−1.25, −0.71] | .316 | .172 | .493 [.452, .522] | .521 | .568 |

- Non-recommended legs: −0.52 pp [−0.73, −0.36]; fade .302, ride .220; hit .490 / .519 / .502.
- Where Underdog had moved the *line itself* toward our side before the decision, NBA recommended legs hit .679 (n 53) against .487 unmoved (n 310); NFL .597 (67) against .426 (115).
- **Reading.** The information that beats us arrives before we score. The platform prices it, and the model, blind to it, rates the moved price as value. This is the market-line selection mechanism (winner's curse at a sharp median; market-centre brief) seen in the price path.
- Removing fades lifts the recommended hit to about .533, still below .546. A filter on the platform's own move is a market-observable selection rule, the family the 10-04 trust-layer verdict killed (`docs/archive/researcher_trust_layer.md`). So it is a diagnostic here, not a lever.

### 2. Predicting the close is feasible on our archive, but it is not a hidden edge

Feasibility counts are in the "Predict-the-close feasibility" section below. The test (r03, r03b, r03c) regresses the move (close − decision price) on the model's disagreement with the decision price, over **all** authentic quoted test rows, not only recommended ones. Placebo forecasters (a coin at .5, and a naive trailing-10 average) remove regression to the mean of a noisy decision price.

| League | Era / decision | n | Model b [90 %] | Coin b | Net model slope (joint fit) [90 %] |
|---|---|---|---|---|---|
| NBA | backfill, evening-before read | 8,172 | +.486 [+.456, +.515] | +.604 | +.199 [+.176, +.222] |
| NFL | backfill | 1,106 | +.444 | +.374 | +.183 [+.142, +.228] |
| NHL | backfill | 37,128 | +.297 | +.027 | +.280 [+.216, +.342] |
| WNBA | backfill | 3,811 | +.281 | +.574 | +.103 [+.085, +.122] |
| NFL | live 2026-07+ | 1,264 | +.016 | −.012 | −.006 [−.042, +.032] |
| NHL | live | 2,632 | +.085 | +.006 | +.083 [+.063, +.103] |
| WNBA | live | 2,908 | +.103 | +.039 | +.100 [+.065, +.143] |
| NBA | live, 8:45 a.m. CDT (13:45 UTC) game-day decision | 371 (5 dates) | +.006 | −.056 | −.018 [−.091, +.011] |
| NFL | live, 8:45 a.m. CDT | 2,881 | +.027 | −.004 | +.022 [+.000, +.042] |
| NHL | live, 8:45 a.m. CDT | 2,172 | +.042 | +.003 | +.041 [+.027, +.055] |
| WNBA | live, 8:45 a.m. CDT | 2,842 | +.058 | +.055 | +.058 [+.031, +.089] |

- In the backfill era a coin "predicts" the move as well as or better than the model (NBA .60 vs .49; WNBA .57 vs .28). The decision price is a stale evening-before read, and any forecast near .5 "predicts" its reversion.
- In the live era a 10 pp disagreement buys at most ~1 pp of closing-line movement (NHL, WNBA), and none in the NBA or NFL.
- Live-era moves are small: SD 3.4 pp (NBA), 4.0 (NFL), 1.4 (NHL), 4.2 (WNBA) between 8:45 a.m. CDT and the last pre-kick read; 60 / 38 / 23 / 51 % move more than 1 pp.
- **Outcome encompassing (r03).** The close encompasses the model in the NBA (coefficient on the model's disagreement at the close +.01, SE .32) and the WNBA (+.08, SE .33). The NFL (+1.21, .30) and the NHL (+1.89, .20) keep information beyond the close. That is the house at-median β order (NHL .67 > NFL unstable > NBA ≈ 0).
- **Verdict.** A target that predicts the close would learn the market's own drift. **KILL as a breakthrough**; keep an NHL-only closing-line-value monitor.
- **Literature.** Prices absorb news fast and fully (Croxson & Reade 2014, doi:10.1111/ecoj.12033). Lines become more accurate and move less as information accrues (Kang & Salaga 2022, doi:10.32731/IJSF/171.022022.03). Informed money moves NBA lines from open to close (Gandar et al. 1998, doi:10.1111/0022-1082.155346). Moves can overshoot and reverse (Simon 2024, doi:10.1287/mnsc.2022.00456; Moskowitz 2021, doi:10.1111/jofi.13082); our NFL reversal (finding 7) matches.

### 3. NBA/WNBA availability: the largest input the model lacks, already in the price

- **Reconstruction (r04).** From box scores, 78 % of NBA team-games miss at least one rotation player (mean vacated minutes 44). The WNBA share is 50 %.
- **The model lacks it.** Outcome minus the standalone model rises +1.8 pp per 10 points of vacated usage (SE 0.5). The model's disagreement with the close falls −2.0 pp per 10 (SE 0.2). In the 16+ bin (3,360 of 8,175 rows, 41 %) the model sits 4.5 pp below the close.
- **The market has it.** Beyond the decision price: −.001 logit per 10 (SE .022), about 0 pp (SE ≈ 0.6 pp). Beyond the close: −.008 logit (SE .023). Underdog's even close under-prices it slightly (+0.9 pp per 10, SE 0.4; r04c), far below the 4.6 pp a 1.83 pick needs.
- **Live.** NBA recommended Unders in games with vacated usage ≥ 8 hit .355 (n 31) against .488 for all other recommended legs (n 82), over 3 dates. WNBA, same split: .568 (n 495) against .609 (n 1,723), over 68 dates (r04d).
- **Pre-build gate, already run (r12).** Leak-favourable box-score absences, a logistic correction cross-fitted on 5 date blocks:
  - It removes the false-disagreement slope (−.021 → .000) and improves Brier by .0007 (.2570 → .2563).
  - It does **not** improve the recommended near-even tail. NBA: +2.8 pp [−2.3, +8.2] at model weight .5, −2.4 [−7.7, +3.1] at 1.0. WNBA: +1.2 [−2.0, +4.3] and −0.2 [−2.6, +1.9].
  - The legs it drops do lose (.44–.50), but it promotes about as many new legs that win ~.50.
- **Literature.** Absence mispricing sits in opening NBA lines and is gone by the close (Dare, Dennis & Paul 2015, doi:10.1016/j.frl.2015.02.004).
- **Collectability answers the L-3 leakage question.** The official injury-report PDFs exist point-in-time back to at least the 2018-19 season. HEAD on 2019-01-15 and 2020-01-15 returned 200. Last-Modified stamps each file's final write: 22:30 UTC (5:30 PM ET) for the 2019–20 "05PM" files, 21:00:04 UTC (5:00 PM ET) for 2026's "05_00PM". Reports come every 15 minutes in 2026-27 (93 links on 10-29), and the 2022 sample also lists the next day's games. No reconstruction leak is needed: the as-of is strict < on Last-Modified.
- **Verdict:** hygiene, not edge. It makes the model agree with the market where the market is right; the replay shows no profit from that.

### 4. NFL wind: the one collectible input that beats the price, on a narrow slice

- **2025 season (r07).** Passing props: wind ≥ 15 mph −.95 logit (SE .26) beyond the close. The model's residual (outcome − standalone P) on 1,328 passing test rows: −27 pp (SE 7) in windy games.
- **2024–26 archive (r09).** 5,095 passing props, 677 games, 37 windy:
  - Passing: wind ≥ 15 −.61 logit (game-clustered SE .21); per 5 mph above 10, −.37 logit (.13).
  - Receiving: −.24 logit (.09). Rushing: +.02 (.10). Touchdowns: +.16 (.09).
  - The books barely move in windy games (+0.1 pp since open, against +0.8 in calm games).
- **Game level (r09b).** Windy minus calm outdoor games, mean residual: passing −12.5 pp (t −2.44, p .019; 37 vs 378 games), receiving −5.6 (p .032), both −7.3 (p .012).
  - Dose-response: −6.8 pp at ≥ 8 mph, rising to −26.8 at ≥ 20.
  - Spearman ρ between wind and game residual −.133 (p .0065) over 415 outdoor games.
- **Holdout (r09c).** The 2022–23 seasons, with the hypothesis fixed before the run; the decision price is the evening-before read. Passing −14.6 pp (p .026), receiving −5.5 (p .020), both −7.1 (p .006). ρ −.063 (p .31) for passing, −.118 (p .056) for both.
- **Leak bound.** nflverse wind is the gamebook measurement at kickoff, not a forecast. Recommended legs are scored a median 10 min before lock, so a short-lead forecast should carry most of it. The forecast re-test is Build 1's step 0.
- **Literature.** NFL totals under-weighted weather in 1984–2004, and a weather rule beat the 52.38 % break-even out of sample (Borghesi 2008, doi:10.1080/09603100701335432). Player-prop markets are thinner than totals.
- **Slice size.** 14, 14, 16 and 21 windy outdoor games in 2022–25, about 5–7 % of games; three quarters fall in November–January. 2026 so far has none recorded: nflverse fills wind late (44 of 182 outdoor games filled).
- **Contrast.** MLB books do price weather (finding 6).

### 5. Tested null: the NFL injury report, and NHL goalies and scratches

- **NFL (r07).** 2025-26 season, nflverse weekly report final statuses (point-in-time).
  - Beyond the close (logit): receiving vacated targets/5 −.055 (.048); own Questionable +.007 (.120); passing QB-out +.052 (.164); rushing ≈ 0.
  - Model residual (probability): vacated targets/5 +1.8 pp (1.9); own Questionable +5.8 pp (4.7).
  - The NFL model's usage windows (FantasyPoints snaps, routes, targets) and the market both already carry it.
- **NHL (r05).** 38,432 authentic test rows.
  - Opposing goalie GSAx/100 (SD 1.34): on scoring props, model residual −0.5 pp per unit (0.3), about −0.7 pp per SD; beyond the decision price −0.4 pp per unit (0.3). Shots/blocks +0.6 pp (0.5).
  - Backup starting: +0.3 pp (0.9). Vacated TOI ≈ 0.
  - Null even though the starter is the actual, which favours the feature.
  - The NHL API publishes no projected starter (landing for an upcoming game verified). The dobbersports projected goalie is fetched every NHL run and never used; leave it.

### 6. MLB: umpires small, weather priced

- **Umpire (r10).** statsapi officials for the 2024 season (the call asked for 2024–25, but the API caps one call at a year): 2,431 games, 90 umpires, one request.
  - K/PA tendency persists only weakly: slope .34 (.21) after shrinkage; the quintile spread is ~0.7 pp of K/PA.
  - Beyond the decision price, 2024 props: batter K +0.68 pp per SD (z 2.1; 26,787 props); pitcher K +1.10 pp per SD (z 1.4; 3,531 props).
- **Weather (r11).** statsapi game weather 2025, one request. Beyond the decision price:
  - Home runs: +0.64 pp per 5 mph blowing out (z 2.0); +0.30 pp per 10 °F (not significant).
  - Total bases and hits: ≤ 0.6 pp per unit.
  - Temperature's physical effect on home runs is established (Callahan et al. 2023, doi:10.1175/BAMS-D-22-0235.1), and the books price it.
- **Umpire literature.** Calls vary with umpire identity and monitoring (Parsons et al. 2011, doi:10.1257/aer.101.4.1410). No peer-reviewed test of umpire mispricing in totals or props turned up; practitioner sources put typical umpire variation under one strikeout per game.

### 7. Processing stored market data (C-3): little beyond the NHL

- **Movement since open, beyond the close (r06; logit per unit of probability moved).**
  - NBA −.056 (.095) and WNBA −.021 (.375): null.
  - NHL +.96 (.14) in the backfill: continuation, the books under-react. Live era +1.03 (1.47), with moves a quarter the size (SD .015 against .064). The backfill "open" is the evening-before read, so the feature means something different by era: a train/serve mismatch risk.
  - NFL −.35 (.13): reversal, Simon's overshoot.
- **Calibration of the consensus close by the number of books quoting.**
  - NFL slope .40 (1–2 books), .48 (3–4), .69 (5–7), .96 (8+).
  - NHL by cross-book dispersion tercile: 1.11 / .99 / .82.
  - This is a book-leg weighting question for the I6d lane, not a new input. NFL's slopes also carry the known inconsistency in the count columns.

### 8. What we fetch and throw away, and what has no history

- **Underdog's feed** (one pull, 22:10 CDT 10-29; 27.6 MB):
  - `lineup_status_id` is set on 85 of 2,072 appearances (Q 59, P 16, O 5, D 3, I 2).
  - `pre_game_data` carries venue type and weather on 92 of 113 games.
  - The scraper drops suspended lines, and the archive has no status column, so suspensions are never stored.
  - `entry_count` (popularity) lives on the lobby endpoints (`docs/underdog_api.md`).
- **NBA daily lineups JSON.** History exists but holds only the final state. The per-player timestamps are not a reliable as-of: −46 to +72 min against tip on 10-28, and +129 to +163 on 2025-01-15 (r08).
- **WNBA.** The league's injury report is live HTML only (rule: designate status by 5 p.m. local the day before); no archive was found. No lineups JSON exists at the NBA-style path.
- **PrizePicks.** The API answered 403 to a plain request (bot protection); not sampled. Rotowire and ProSportsTransactions: not sampled (ToS; bot protection).

### 9. A leakage trap in the plan's starter-flag item (M-4 / D-3)

- §6.4 says box-score `START_POSITION` "backfills leakage-clean". It is an actual: confirmed starters exist only ~0–60 min before tip, and history keeps no reliable as-of stamp (finding 8).
- It leaks for any leg scored before the confirmation, and for training's noon-UTC (7 a.m. CDT) book read.
- Fixes, any one: use the injury report's as-of status; forward-capture the Expected/Confirmed states; or restrict the feature to decisions inside the confirmation window.

---

## Per-league audit table

Ranked by the dispatch rule: information beyond the decision-time price × collectability of both halves ÷ cost. "PIT" = point-in-time. Rebuild times are the house figures (cold matrix rebuild 6–20 h per league; NFL ~6–7 h).

| Rank | League | Source | Live: how · cost · latency | Historical: how · coverage · PIT or actuals | Feature (as-of rule · fallback · join key) | Grade · evidence | Effort |
|---|---|---|---|---|---|---|---|
| 1 | NFL | Game wind (D-4) | Open-Meteo forecast API (keyless; free tier non-commercial, < 10k calls/day) or NWS `api.weather.gov` (public domain, User-Agent only). Models update hourly; kickoff wind is known days ahead, so there is no race with Underdog and every :50 scoring can read it | Open-Meteo Previous Runs API (fixed 1–7 day leads; most models from Jan 2024) and Historical Forecast API (short lead; GFS from 2021-03-23, HRRR 2018): **PIT**. nflverse game-time **actuals** 2022–25 (leak = forecast error) | Mean forecast 10 m wind over kickoff → +3 h at the stadium; latest fetch strictly before the decision; dome or closed roof → 0; missing → NaN; key (season, week, home team) → stadium (M-3 table) | **B**: passing −12.5 pp (2024–25, p .019) and −14.6 pp (2022–23 holdout, p .026) relative to calm games; receiving −5.6 / −5.5 pp; dose-response ρ −.13 | 1 d pre-test + 2–3 d build + NFL rebuild ~6–7 h |
| 2 | All | Forward capture: Underdog `lineup_status_id`, line suspensions, lobby `entry_count`; NBA injury PDFs live; NBA lineups JSON pre-tip; WNBA report (May 2027) | Already in hand each hourly poll (Underdog: zero marginal cost, same latency as Underdog's own line changes); NBA PDFs every 15 min; lineups JSON every 15 min from tip − 2 h | **None**: Underdog statuses, suspensions and popularity were never stored; lineups history is final state only; WNBA report has no archive → **forward-only** | Dated snapshots per poll (`DatedSnapshotStore.latest_before`, strict <); key (league, player, game); used as monitors and report columns until a season accrues | **E**: 85 of 2,072 appearances flagged at one poll; popularity mispricing reported in tennis (Ramirez et al. 2023), weakened on replication (Clegg & Cartlidge) | 2–3 d, no rebuild |
| 3 | MLB | Home-plate umpire K tendency (D-5) and park weather (D-4) | statsapi game feed officials, posted pre-game (crews rotate predictably); Open-Meteo / NWS forecast at the park; MLB copyright notice: personal, non-commercial use | statsapi schedule `hydrate=officials,weather`: one request per season, every game. Umpire **PIT**; weather game-time **actuals** (leak = forecast error) | `ump_k`: shrunk expanding K/PA residual over prior plate games (k0 1,500 PA, strict prior dates); wind out / in mph; temperature; closed roof → 0; key gamePk | **C**: batter K +0.68 pp/SD (z 2.1), pitcher K +1.1 pp/SD (z 1.4), 2024; HR +0.64 pp per 5 mph out (z 2.0), 2025; temperature +0.30 pp/10 °F (n.s.) | 0.5 d replication + 1–2 d + MLB rebuild, before Opening Day 2027 |
| 4 | NBA | Official injury report (L-3) | `ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH_MMAM.pdf` every 15 min, free. Status changes reach Underdog within minutes, so this gives parity, not a lead | PDF archive back to ≥ 2018-19 (HEAD 2019-01-15 → 200); Last-Modified stamps the final write: **PIT**. Parser: `nbainjuries` (MIT, 2021-22 on, needs Java) or our own | Own status; teammates Out/Doubtful vacated usage (trailing minutes × usage) from the latest report with Last-Modified < decision; no report → 0; key (team, game date, normalized name → player id) | **C**: model residual +1.8 pp/10 vacated usage (SE .5); beyond price −.001 logit (.022); r12 tail replay −2.4 to +2.8 pp, CIs ±5 | 3–5 d + NBA rebuild 6–20 h |
| 5 | WNBA | Injury report / absences | wnba.com HTML, "updated on a continual basis", free | No archive → box-score reconstruction (**actuals**; leak = late scratches and coach's decisions) | As rank 4 | **C**: false disagreement −1.6 pp/10 vacated usage (SE .2); outcome residual −.016 logit (.032); r12 tail replay −0.2 to +1.2 pp | Forward from May 2027; 2 d + WNBA rebuild |
| 6 | NHL | Stored line movement (C-3) | Archive polls, free | Backfill paths are two points (evening before, game day); live 5 slots → era mismatch | mv = decision price − first price | **C**: backfill +.96 logit per unit (.14); live +1.03 (1.47), moves ¼ the size | 1–2 d; research only |
| 7 | NFL | Few-book overconfidence of the consensus (C-3) | Archive, free | Archive | Book-count-dependent shrink of the book leg | **C**: close calibration slope .40 / .48 / .69 / .96 at 1–2 / 3–4 / 5–7 / 8+ books (instrument caveat) | Belongs to the I6d book-leg lane |
| 8 | NBA | Referee crew (D-5) | official.nba.com, posted ~9 a.m. ET game day (HTML) | stats.nba.com box-score summary, one request per game (cdn liveData answers 403); **PIT** (assigned pre-game) | Crew foul / free-throw tendency, shrunk, prior games only | **E**: crews move fouls (Price & Wolfers 2010); no evidence of mispricing found | 2 d + NBA rebuild |
| 9 | NFL | Officials | nflverse `officials.parquet` (HEAD 200) | Same file, **PIT** | Crew penalty rate | **E** | 1–2 d |
| 10 | NBA | Confirmed starters (M-4 / D-3) | stats.nba.com daily lineups JSON (Expected → Confirmed ~0–60 min pre-tip) | History is final state; timestamps unreliable (−46 to +163 min vs tip) → **actuals** | Starter flag as-of decision; history leaks for decisions before confirmation | **E** (mostly implied by the injury report) | Forward capture in rank 2 |
| 11 | NFL | Injury report / inactives (L-3) | nflverse weekly report (Wed–Fri); inactives 90 min pre-kick | nflverse 2009 on, **PIT** | Vacated targets; own Questionable; QB out | **D**: beyond price (logit) −.055 (.048), +.007 (.120), +.052 (.164); model residual +1.8 pp (1.9) | KILL |
| 12 | NHL | Opposing / own starting goalie | dobbersports `predictedGoalies` (fetched each run, unused); DailyFaceoff (robots disallow `/api/`; ToS risk); NHL API has no projected starter | Box-score `starter` flag (**actuals**; leak = surprise starts) | Opposing GSAx/100; backup flag | **D**: scoring props, model residual −0.5 pp per GSAx/100 (0.3; SD 1.34, so ≤ 0.7 pp per SD), beyond the decision price −0.4 (0.3); backup +0.3 pp (0.9); this small even with the actual-starter leak | KILL |
| 13 | NHL | Scratches / vacated TOI | Right-rail scratches (post-game) | Box score (**actuals**) | Vacated TOI | **D**: ≈ 0 | KILL |
| 14 | NBA / WNBA | Stored movement and dispersion (C-3) | Archive | Archive | mv; cross-book dispersion | **D**: NBA −.056 logit (.095), WNBA −.021 (.375); calibration flat across dispersion | KILL |
| 15 | All | Predict the close (processing) | Archive polls | Backfill paths are two-point and stale | Target = closing price | **D** (NBA, NFL) / **C** (NHL, WNBA): live net slopes −.018 to +.10 | KILL as breakthrough; NHL CLV monitor |
| 16 | All | PrizePicks lines | `api.prizepicks.com` behind bot protection (403) | None | — | **E** | Not recommended |
| 17 | NBA | ESPN injuries API | Undocumented live JSON, keyless | None | Redundant with rank 4 | **E** | Skip |
| 18 | NHL | Line combinations / power-play units | DailyFaceoff (ToS risk) | Per-game NHL shift charts (one request per game); MoneyPuck season aggregates only | Linemate quality | **E**, low prior | 3–5 d |

---

## Recommendation

### Top three builds, each with a pre-registered test and kill criteria

**Build 1 — NFL game wind for the passing and receiving cells.** Engineering project after a one-day research pre-test.

- **Step 0, pre-test (1 day, research folder, no `src/` edit).**
  - Fetch forecasts at stadium coordinates for every outdoor or open-roof game in 2022–25 (~760 games). Use the Historical Forecast API for a short lead (all four seasons) and the Previous Runs API at a 1-day lead (2024–25). That is under 2,000 requests, inside one day's free quota for this research use.
  - Rerun r09b and r09c with forecast wind in place of gamebook wind.
  - Hypotheses, fixed now, one-sided: H1, passing: the mean game residual in games forecast at ≥ 15 mph minus calm outdoor games is below 0. H2, the same for passing plus receiving. H3: Spearman ρ between forecast wind and the game residual is below 0.
  - **Accept:** H1 difference ≤ −6 pp with p < .05 and H2 difference ≤ −3 pp with p < .05, pooled 2022–25 on the short-lead forecast; the 1-day-lead forecast (2024–25) has the same sign.
  - **Kill:** H1 difference > −3 pp, or p > .20. Then the gamebook result was forecast-error leak or noise, and D-4 closes for the NFL.
  - **Neither** (between the bands): no build yet. The live test below still runs, since it needs no collector, and the pooled 2022–26 forecast read after the 2026 regular season decides with the same bars.
  - Power: about 65 windy games against ~600 calm, game-level SD ≈ .30. A true −12 pp is detected about 94 % of the time; a true −6 pp only about 45 %. A pass is informative; a marginal fail does not prove zero.
- **Build (2–3 days, plus the NFL rebuild of ~6–7 h).**
  - Stadium table (the M-3 dependency; include neutral and international sites, and the retractable-roof status).
  - A `weather` source in `src/sportstradamus/collectors/` that snapshots the forecast at each prophecize run for games in the next 72 h, plus the archive backfill.
  - Features `wind_kick` and `wind15`, as-of strictly before the decision. Join on (season, week, team). Dome → 0; missing → NaN (B-5 parity).
  - Retrain the passing and receiving cells through the six gates.
- **Live acceptance (weeks 10–18 and playoffs, Nov 2026–Jan 2027; expect 12–18 windy games).**
  - Accept: Underdog passing and receiving Unders in games forecast at ≥ 15 mph at decision time hit ≥ .546, with a game-clustered 90 % lower bound > .50 after ≥ 10 windy games; and the residual against Underdog's own close is ≤ −3 pp (Underdog does not already shade for wind).
  - Kill: Unders hit ≤ .50 after ≥ 10 windy games, or the residual against Underdog's close is > −1 pp.
  - Between the two: keep reading through the playoffs and decide on the pooled windy games at season end with the same bars.
  - This live test needs no model change and no collector: the Previous Runs and Historical Forecast APIs return the forecast as it stood before each game even when asked afterwards, and Underdog's lines and the outcomes are already in the archive and history.

**Build 2 — Forward capture of what we already see.** Engineering project; optionality, not a proven edge.

- **Contents (about 2–3 days in the collectors framework, no rebuild):**
  1. Underdog `appearances[].lineup_status_id` per poll, decoded with `/v1/lineup_statuses` (0.5 day).
  2. Underdog line status transitions (active → suspended → active or removed) with the poll stamp. The scraper drops them today (0.5 day).
  3. Underdog lobby `entry_count` per line, converted from strings such as "75.2K" (0.5 day).
  4. The official NBA injury report: the newest PDF at :45 on game days, parsed to rows with Last-Modified (1 day; the parser is shared with rank 4's backfill).
  5. NBA daily lineups JSON every 15 min from tip − 2 h, keeping the Expected and Confirmed states (0.5 day).
  6. The WNBA injury report HTML, from the May 2027 opener (0.5 day, then).
- **Meanwhile.** Monitors and report-only columns only. No model features until a season accrues: a serving-only feature is a train/serve mismatch.
- **Pre-registered tests:**
  - **T2a, news guard** (NBA, NHL, NFL; run at ≥ 300 flagged recommended legs, about 2027-01-15). Compare recommended legs whose game saw an Underdog suspension, or a teammate's status change to O or D, within 3 h before our decision, against the other recommended legs. Accept, and propose an abstention rule to the owner in the L1 family: hit difference ≤ −4 pp with a 90 % date-cluster interval below 0. If neither at 300 legs, read again at 600. Kill: not accepted at 600 flagged legs.
  - **T2b, popularity** (≥ 5,000 legs with `entry_count`, about 2027-01-15). Logistic y ~ logit(Underdog's implied p) + popularity z (log `entry_count` within game and market), date-clustered. Accept: |coefficient| ≥ 2 pp per SD with p < .01 and the same sign in both halves of the sample. Kill otherwise.

**Build 3 — MLB umpire and weather for the 2027 season.** Engineering project if the replication passes.

- **Pre-test now (0.5 day, research folder).** Three statsapi schedule calls, one season each (the API caps a call at a year): officials for 2025 and 2026, weather for 2026. Rerun r10 on 2025–26 props and r11 on 2026 props, out of sample.
  - **Accept:** a batter-K or pitcher-K umpire coefficient ≥ +0.5 pp per SD with date-clustered p < .05; or home runs ≥ +0.5 pp per 5 mph blowing out with p < .05.
  - **Kill:** neither replicates (p > .20). Then D-5 and D-4 close for MLB with no build.
  - **Neither:** no MLB build for 2027; re-read once on 2027 props with the same bars, then close.
- **Build if accepted (1–2 days, plus the MLB rebuild of 6–20 h, before Opening Day 2027).** statsapi backfill (one call per season), live officials from the game feed at each run, park forecasts; as-of strict <. The expected effect is ≤ 1 pp per SD: a refinement of the K and home-run means, not an edge on its own.

### Forward-only list (start storing now)

"Monitor-ready" means a pre-registered test can run. "Trainable" means a full season of history exists, which is what a feature needs before it can enter training without a train/serve mismatch.

| Source | Start | Monitor-ready | Trainable as a model feature | Meanwhile |
|---|---|---|---|---|
| Underdog `lineup_status_id` (all leagues) | 2026-10-30 | NFL ~2027-01-10 (end of regular season); NBA, NHL ~2027-01-15 | NFL 2027-09; NBA, NHL 2027-10; MLB and WNBA a season after their 2027 openers (2028) | T2a news guard; report column |
| Underdog line suspensions | 2026-10-30 | ~2027-01-15 | 2027-10 (NBA, NHL); 2027-09 (NFL) | T2a |
| Underdog lobby `entry_count` | 2026-10-30 | ~2027-01-15 (≥ 5,000 legs) | 2027-10 | T2b |
| NBA daily lineups, Expected / Confirmed states | 2026-10-30 | ~2027-01-15 | 2027-10 | Fixes M-4 / D-3's as-of |
| WNBA injury report (HTML) | 2027 opener (May) | ~2027-08 | 2028-05 | Box-score reconstruction, leak-bounded |
| NHL projected goalie (dobbersports, already fetched) | Optional, zero cost | — | 2027-10 | Tested null (finding 5); low value |
| PrizePicks projections | Not recommended (bot protection, ToS) | — | — | — |

Not forward-only: the NBA injury report (archive to ≥ 2018-19), wind forecasts (Open-Meteo archives 2021 / 2024 on) and MLB umpires and weather (statsapi, every game). Forward capture of their live values still removes any archive-versus-live mismatch.

### Demote or KILL

| Item (plan id) | Verdict | Reason |
|---|---|---|
| L-3 NBA / WNBA availability | Demote to hygiene backlog | Priced at decision time; r12 tail replay null. The leakage question is answered (PIT PDFs). Build after 1–3, or if the owner wants cleaner NBA models |
| L-3 NFL injury report / inactives | KILL | Null beyond the price and the model (finding 5) |
| NHL goalie, scratches | KILL | ≤ 0.7 pp per SD even with the actual-starter leak; bears on the goalie half of §6.7 M-9 / D-2 (its xG half is untested here) |
| C-3 NBA / WNBA movement and dispersion | KILL | Null beyond the close |
| C-3 NHL movement | Research only | Era mismatch; live not significant |
| C-3 NFL few-book overconfidence | Route to the I6d book-leg lane | A book-leg calibration question |
| Predict the close | KILL as breakthrough | Live net slopes ≤ .10; NBA ≈ 0; keep an NHL CLV monitor |
| D-5 referees (NBA, NFL, NHL) | Park | Untested, low prior; one request per game to backfill |
| PrizePicks | Not recommended | Bot-protected API; ToS |

---

## The recommended-leg line-movement read (owner's request; detail behind finding 1)

Scripts r02 and r02b; legs in `r02_legs.parquet` and `r02b_legs.parquet`. "Platform" is Underdog or Sleeper at the leg's own line. "Books" is the sportsbook consensus at our line from the archive. Moves are in pp toward our side.

**Coverage.** Of 1,182 recommended legs, 1,155 have a platform read at the close, and 532 have books at both ends. NBA 460 / 320, NFL 319 / 165, NHL 376 / 47. The platform's close is stale by more than 1.5 h on 29 / 24 / 15 legs. The median lead from the last scoring to the lock is 0.16 h in all three leagues.

**By lead time (last scoring → kickoff), recommended legs:**

| League | Lead | n | Hit | Claimed | Platform move, pp | Share against |
|---|---|---|---|---|---|---|
| NBA | < 1 h | 426 | .516 | .605 | −0.36 | .099 |
| NBA | 24 h + | 28 | .464 | .600 | −0.94 | .071 |
| NFL | < 1 h | 291 | .498 | .618 | −0.01 | .003 |
| NFL | 6–24 h | 21 | .571 | .620 | −0.47 | .048 |
| NHL | < 1 h | 348 | .549 | .605 | −0.10 | .026 |
| NHL | 6–24 h | 13 | .538 | .619 | −8.73 (n 11) | .308 |

**By platform:** NBA Underdog −0.54 pp (n 321, hit .526), Sleeper −0.17 (155, .477); NFL Underdog −0.04 (276, .514), Sleeper −0.09 (44, .477); NHL Underdog −0.51 (358, .539), Sleeper −0.03 (28, .571).

**By side:** NBA Overs −0.55 pp (237, hit .544), Unders −0.31 (239, .477); NFL Overs −0.01 (44, .568), Unders −0.05 (276, .500); NHL Overs −0.08 (46, .500), Unders −0.53 (340, .547).

**Bottom line.** Post-decision drift is real but small (−0.3 to −0.5 pp in NBA and NHL, ≈ 0 in the NFL) and concentrated in legs scored hours early. The pre-decision fade (−1.0 pp, a third of legs, hit .493) is the larger signature. Both say the same thing: news reaches the price before or around our scoring, and the model does not see it.

---

## Predict-the-close feasibility (the numbers this brief owns)

**Sportsbook polls per player-prop (non-DFS books), r01.** Share of props with ≥ 2 polls spanning ≥ 6 h, with counts:

| League | 2024-25 | 2025-26 | 2026-27 (live era) | Live: median polls, median span |
|---|---|---|---|---|
| NBA | .761 (153,725 of 201,918) | .787 (138,435 of 175,844) | .794 (4,638 of 5,839) | 4 polls, 9.3 h |
| NFL | .885 (22,813 of 25,775) | .906 (22,022 of 24,311) | .926 (8,621 of 9,305) | 5, 102.5 h |
| NHL | .161 (25,750 of 159,681) | .857 (138,664 of 161,818) | .678 (14,921 of 21,996) | 3, 9.3 h |
| WNBA | .302 (3,428 of 11,342) | .759 (13,422 of 17,691) | .888 (13,489 of 15,194) | 8, 30.3 h |
| MLB | .244 (143,820 of 589,326) | .535 (296,256 of 554,178) | .421 (82,152 of 194,908) | 2, 3.8 h |

- Before 2026-07, 98–100 % of sportsbook stamps sit on two hours: about 7 p.m. Central the evening before (00–01 UTC), and about 6 p.m. Central on game day (23 UTC). A pre-2026-07 "path" is two points, and the first is stale.
- From 2026-07, polls land at 8 and 11 a.m. and 2 and 5 p.m. CDT (13, 16, 19 and 22 UTC) plus close-lines captures 5–25 min before commence. A full-path era exists only from 2026-07.

**Underdog polls per prop, October 2026, r01.** Share with ≥ 2 polls spanning ≥ 6 h: NBA .879 (8,442 of 9,603; median 12 polls), NFL .959 (23,264 of 24,251; 47), NHL .891 (30,323 of 34,019; 10), WNBA .931 (1,684 of 1,809; 23), MLB .897 (5,367 of 5,985; 13). Underdog and Sleeper history starts 2026-03, with single polls until May.

**Rows for the test (r03).** Authentic test rows: NBA 9,180, NFL 12,460, NHL 43,758, WNBA 7,079. Of these, 72,036 join a decision read and 57,229 have a later pre-kick read at least 1 h after it. Instrument check: corr(1 − Odds, decision price) = .918; median |difference| .0026.

**Result.** See finding 2. Net of the placebos, the live-era close recovers −.02 to +.10 of the model's disagreement. The feasible sample is large; the signal is not.

---

## Reality checks

- **NFL wind (Build 1).**
  - *Effect.* −12.5 / −14.6 pp on passing and −5.5 / −5.6 pp on receiving, relative to calm games, are on gamebook actuals. Forecast error and the winner's curse of a selected finding will shrink them. Plan on half: about −6 pp passing and −3 pp receiving relative to the line.
  - *Money.* At 1.83, a passing Under hitting .57 is worth about +4 % per leg. Receiving alone does not clear break-even unless the model's own lean adds to it.
  - *Volume.* About 15–20 windy games a season and ~25 quoted passing and receiving props per game. Outcomes within a game move together, so the effective sample is games: a season is roughly 15–20 independent bets, and a true edge will take more than a season to show in profit (Wunderlich & Memmert 2020, doi:10.1016/j.ijforecast.2019.08.009).
  - *What could make it wrong.* Underdog may already shade lines for wind (untested: no windy 2026 game yet). NFL yardage props sit near .5, so "beyond the price" is close to "beyond the line". Retractable roofs need their game-day status. The 15 mph threshold came from the 2025 read; the dose-response and the 2022–23 holdout argue against a threshold artefact, not against chance.
  - *Licence.* Open-Meteo's free tier is non-commercial; profit-seeking use may need its paid plan. NWS is public domain for the live half. Decide before production.
- **Forward capture (Build 2).** It cannot be graded yet; that is its point. Popularity mispricing was found in tennis (Ramirez et al. 2023, doi:10.1016/j.ijforecast.2022.07.011) and shrank on replication (Clegg & Cartlidge, arXiv:2306.01740). Expect T2b to fail; it is cheap to run.
- **MLB (Build 3).** The umpire read covers one season (2024) and the weather read one (2025). ≤ 1 pp per SD cannot clear a 4.6 pp hurdle alone. The value is a sharper strikeout and home-run mean, worth it only because both halves are one request per season.
- **NBA availability.** r12 uses a linear correction on box-score absences. A retrained model with player-specific features could extract more, but the market already prices absences at decision time (finding 3), and that bounds any gain on the recommended tail. Brier gains of ~.0007 are real model hygiene, not profit.
- **Predict the close.** The live-era samples are short (5–68 dates). The NHL's +.04 to +.08 could grow with more data, but the house rule is clear: CLV is a monitor, not a target, while the owner's no-model-free rule stands.
- **Generality.** Every in-house number here is this project's archive instrument on this project's test rows. None is a law of the market. The 2022–23 holdout is the only out-of-sample replication in this brief.

## Open questions / caveats (for §8.2)

1. **NFL wind with point-in-time forecasts.** Build 1 step 0; decides D-4 for the NFL.
2. **Does Underdog shade lines for wind?** Needs windy 2026 games (Nov–Jan) and the forecast snapshots running.
3. **Starter-flag as-of (M-4 / D-3).** Box-score `START_POSITION` is an actual. Daily-lineups history has no reliable as-of stamp, so forward-capture the Expected / Confirmed states.
4. **NHL movement era mismatch (C-3).** The backfill open is the evening-before read. Any NHL movement feature must be defined on live-era slots only, or rescaled per era, before it is tested again.
5. **NFL few-book overconfidence.** Close calibration slopes .40–.69 below 8 books, on an instrument with known count-column issues. Route to the I6d book-leg lane.
6. **MLB 2026 replication** (Build 3 pre-test) before any MLB build for 2027.
7. **Referees** (NBA, NFL, NHL) are untested. Backfill costs one request per game; prior low.
8. **Licensing.** Open-Meteo free tier non-commercial; NBA.com, stats.nba.com and MLB statsapi are personal-use terms; DailyFaceoff, Rotowire and PrizePicks were not sampled. A human decides the risk before any production collector.
9. **Cross-league (§7.4).** The WNBA has no injury-report archive (forward-only from May 2027). Wind applies only to outdoor and open-roof NFL games; NBA, WNBA and NHL are indoor. MLB weather is priced. NFL yardage props sit near .5, so tests there read the line more than the price.

---

## Bibliography

New to the house list unless marked; identifiers verbatim.

| Ref | Source | Identifier | Used for |
|---|---|---|---|
| L1 | Borghesi, R. (2008). Weather biases in the NFL totals market. *Applied Financial Economics* 18(12), 947–953 | doi:10.1080/09603100701335432 | Weather under-weighted in NFL totals, 1984–2004 (finding 4) |
| L2 | Dare, W. H., Dennis, S. A., Paul, R. J. (2015). Player absence and betting lines in the NBA. *Finance Research Letters* 13, 130–136 | doi:10.1016/j.frl.2015.02.004 | Absence mispricing in openers, gone by the close (finding 3) |
| L3 | Croxson, K., Reade, J. J. (2014). Information and efficiency: goal arrival in soccer betting. *Economic Journal* 124(575), 62–91 | doi:10.1111/ecoj.12033 | Prices impound news fast and fully (finding 2) |
| L4 | Gandar, J. M., Dare, W. H., Brown, C. R., Zuber, R. A. (1998). Informed traders and price variations in the betting market for professional basketball games. *Journal of Finance* 53(1), 385–401 | doi:10.1111/0022-1082.155346 | Informed money moves NBA lines open → close |
| L5 | Simon, J. (2024). Inefficient forecasts at the sportsbook: an analysis of real-time betting line movement. *Management Science* 70(12), 8583–8611 | doi:10.1287/mnsc.2022.00456 | Overshoot and reversal in line paths (NFL reversal, finding 7) |
| L6 | Moskowitz, T. J. (2021). Asset pricing and sports betting. *Journal of Finance* 76(6), 3153–3209 | doi:10.1111/jofi.13082 | Momentum / delayed overreaction, small and unexploitable after costs |
| L7 | Kang, B., Salaga, S. (2022). The impact of information release on line accuracy and line movement. *International Journal of Sport Finance* 17(1) | doi:10.32731/IJSF/171.022022.03 | Lines grow more accurate and move less as information accrues |
| L8 | Callahan, C. W., Dominy, N. J., DeSilva, J. M., Mankin, J. S. (2023). Global warming, home runs, and the future of America's pastime. *Bulletin of the American Meteorological Society* | doi:10.1175/BAMS-D-22-0235.1 | Temperature raises home runs (physical effect the books price, finding 6) |
| L9 | Parsons, C. A., Sulaeman, J., Yates, M. C., Hamermesh, D. S. (2011). Strike three: discrimination, incentives, and evaluation. *American Economic Review* 101(4), 1410–35 | doi:10.1257/aer.101.4.1410 | Umpire calls vary with identity and monitoring |
| L10 | Price, J., Wolfers, J. (2010). Racial discrimination among NBA referees. *Quarterly Journal of Economics* 125(4), 1859–1887 | doi:10.1162/qjec.2010.125.4.1859 (NBER w13206) | Crew composition moves fouls |
| L11 | Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? *Economic Journal* 114(495), 223–246 | doi:10.1111/j.1468-0297.2004.00207.x | Bookmakers set informed prices |
| L12 | Kaunitz, L., Zhong, S., Kreiner, J. (2017). Beating the bookies with their own numbers — and how the online sports betting market is rigged | arXiv:1710.02824 | Consensus as the information set; book limits on winners |
| L13 | Hubáček, O., Šourek, G., Železný, F. (2019). Exploiting sports-betting market using machine learning. *International Journal of Forecasting* 35(2), 783–796 | doi:10.1016/j.ijforecast.2019.01.001 | Profit needs information the bookmaker lacks |
| L14 | Wunderlich, F., Memmert, D. (2020). Are betting returns a useful measure of accuracy in (sports) forecasting? *International Journal of Forecasting* 36(2), 713–722 | doi:10.1016/j.ijforecast.2019.08.009 | Small samples of returns are noisy (Build 1 volume) |
| L15 | Ramirez, P., Reade, J. J., Singleton, C. (2023). Betting on a buzz: mispricing and inefficiency in online sportsbooks. *International Journal of Forecasting* 39(3), 1413–1423 | doi:10.1016/j.ijforecast.2022.07.011 | Attention predicts pricing errors (tennis); T2b prior |
| L16 | Clegg, L., Cartlidge, J. (2023; IJF 2025). Not feeling the buzz: correction study of mispricing and inefficiency in online sportsbooks | arXiv:2306.01740 | Replication: mispricing holds, profits mostly vanish |
| L17 | Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics* 90(3) | doi:10.1162/rest.90.3.414 (house) | Date-cluster intervals |
| D1 | Open-Meteo Historical Forecast API; Previous Runs API; Terms | open-meteo.com/en/docs/historical-forecast-api; /en/docs/previous-runs-api; /en/terms | Coverage (GFS 2021-03-23, HRRR 2018, IFS 2017; Previous Runs from Jan 2024); free tier non-commercial, < 10k calls/day, CC BY 4.0 |
| D2 | NBA official injury report PDFs | ak-static.cms.nba.com/referee/injury/ ; official.nba.com/nba-injury-report-2025-26-season/ | 15-minute PIT reports, archive ≥ 2018-19 |
| D3 | `nbainjuries` (PyPI 1.1.1, MIT) | pypi.org/project/nbainjuries | Parser for the official PDFs (2021-22 on; Java via tabula-py) |
| D4 | MLB Stats API schedule `hydrate=officials`, `hydrate=weather` | statsapi.mlb.com/api/v1/schedule | Umpires and game weather for every game, one call per season |
| D5 | nflverse (games, injuries, depth charts, officials) | github.com/nflverse/nflverse-data releases | NFL weather actuals, PIT injury reports |
| D6 | WNBA injury report | www.wnba.com/wnba-injury-report | Live only; 5 p.m. local designation rule |
| H1 | House: market-centre brief; gates-vs-pick'em brief; model-class brief; trust-layer brief | `docs/archive/researcher_market_centre_anchor.md`; `/tmp/researcher_gates_vs_pickem_profit.md`; `/tmp/researcher_model_class_and_edge.md`; `docs/archive/researcher_trust_layer.md` | Selection mechanism; L1–L3; ceiling; killed selection layers |

---

## Files written

All under `$L/research/information_audit_reads/` unless noted (`$L` = `/home/trevor/backups/sportstradamus/2026-10-21-live-gap-2`).

- Brief: `/tmp/researcher_information_audit.md`, copied to `$L/research/researcher_information_audit.md`.
- r01 poll feasibility: `r01_poll_feasibility.py`, `.txt`, `r01_sportsbook_poll_feasibility.csv`, `r01_dfs_poll_feasibility.csv`.
- r02 recommended-leg movement: `r02_reco_line_movement.py`, `.txt`, `r02_legs.parquet`, `r02_summary_dec.csv`.
- r02b pre-decision movement: `r02b_pre_decision_movement.py`, `.txt`, `r02b_legs.parquet`, `r02b_summary.csv`.
- r03 predict-the-close: `r03_predict_the_close.py`, `.txt`, `r03_rows.parquet`, `r03_slopes.csv`; `r03b_close_robustness.py`, `.txt`, `r03b_live_rows.parquet`; `r03c_placebo.py`, `.txt`.
- r04 NBA / WNBA absence: `r04_nba_absence.py`, `.txt`, `r04_NBA_absence.parquet`, `r04_WNBA_absence.parquet`, `r04_live_nba_legs.parquet`; `r04b_absence_dfs_lines.py`, `.txt`, `r04b_dfs_lines_absence.parquet`; `r04c_underdog_close_absence.py`, `.txt`, `r04c_NBA_dfs_close.parquet`, `r04c_WNBA_dfs_close.parquet`; `r04d_reco_by_absence.py`, `.txt`, `r04d_NBA_reco_absence.parquet`, `r04d_WNBA_reco_absence.parquet`.
- r05 NHL goalies and absences: `r05_nhl_goalie_absence.py`, `.txt`, `r05_nhl_features.parquet`.
- r06 archive features (C-3): `r06_archive_features.py`, `.txt`, `r06_props.parquet`.
- r07 NFL injuries and weather: `r07_nfl_injuries_weather.py`, `.txt`, `r07_nfl_features.parquet`.
- r08 source verification: `r08_source_verification.txt`; `r08_inspect_probes.py`, `r08_inspect_probes2.py`, `r08_inspect_probes3.py` and their `_part1–3.txt`; `r08_pdf_peek.py`; `r08_lineup_confirm_lead.py`, `.txt`.
- r09 NFL wind: `r09_nfl_wind_all_seasons.py`, `.txt`, `.csv`; `r09b_nfl_wind_game_level.py`, `.txt`; `r09c_nfl_wind_holdout_2022_2023.py`, `.txt`, `r09c_nfl_wind_holdout_props.parquet`.
- r10 MLB umpire: `r10_mlb_umpire_k.py`, `.txt`, `r10_mlb_umpire_props.parquet`.
- r11 MLB weather: `r11_mlb_weather.py`, `.txt`, `r11_mlb_weather_props.parquet`.
- r12 absence replay: `r12_nba_absence_denoise_replay.py`, `.txt`, `r12_{NBA,WNBA}_{0,1}_denoise_rows.parquet`.
- Downloads (untrusted): `ext/` (nflverse games, injuries 2025 / 2026, depth charts 2026), `ext/probes/`, `ext/probes_ud/`, `ext/probes_wnba/`, `ext/mlb_officials/`.
