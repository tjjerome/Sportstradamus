# I6f: a pre-game cutoff for the game lines training reads

Design note, 2026-10-05. Nothing in the repository or the archive was changed to write it.
Every script named here is kept with its output on the dev box only, in `~/backups/sportstradamus/2026-10-04-honest-receipts/main/wave4/i6f/`;
the list is at the end. What was built from it, and what waits on the owner, is in
`docs/handoffs/honest-receipts.md` §6 (I6f) and §8.

## Ten-line summary

1. **Rule.** For a finished game, training takes each sportsbook's newest Moneyline and Totals quote stamped at or before **15:00 UTC on the game date**. One named constant, one extra SQL condition in the bulk reader, two call lines.
2. **Why 15:00 and not the props' 12:00 UTC.** Both remove the leak (MLB team total against runs scored: .584 as read today, .181 at 15:00, .167 at 12:00; clean seasons sit at .16 to .19). 12:00 admits only the previous evening's poll and leaves 24.0% of live-polled MLB team-games with no quote; 15:00 admits the morning poll and leaves 4.5%.
3. **No start time exists** in any gamelog, team log or the archive, so a cutoff tied to each game's own start cannot be built from data on disk.
4. **There are no NULL stamps.** Of 2,238,705 game-line rows, 1,097,983 carry a midnight placeholder (any cutoff keeps them) and 780,976 were written by historical backfills and stamped with the day the backfill ran. A plain cutoff drops those and empties whole seasons (MLB 2025: 99.4% of team-games quoted falls to 0.6%; NHL 2025: 100% to 0%).
5. **So the change has three parts:** the reader cutoff; a small writer fix so a historical game-line write carries its snapshot time; one UPDATE on the dev archive that moves the 780,976 rows onto their game date. Dry run on a copy: 0 of 1,800,407 keys without a live poll read differently.
6. **What the cutoff itself changes:** only games polled live, which is 1,364 MLB, 454 WNBA, 90 NFL, 54 NBA and 0 NHL team-games in the dev gamelog (6.0%, 9.7%, 3.2%, 2.4% and 0% of cached matrix rows). In MLB those rows are 98.1% of the matrix rows that carry any real game line today.
7. **Rows already on disk do not change by themselves.** Lines are written into the gamelog once, at ingestion, and the matrix cache only appends. For the next retrain to see the change: re-enrich the gamelogs (seconds per league) and rebuild or patch 93 cached matrices (2,020,447 rows; rebuild time not measured).
8. **A whole re-enrichment does more than cut.** It also fills seasons whose lines were backfilled after ingestion (MLB matrix rows with a real line go from 4.6% to 99.2%, NHL from 49.1% to 99.7%), and for NBA it would import 382 inflated team totals (+33 points) unless NBA is left out or limited to dates from 2026-05-08.
9. **NFL.** 84 of the 90 defaulted 2026 team-games have archive rows; the lookup uses the week's first game day. The cutoff neither fixes nor worsens this, but an NFL repair without the cutoff would swap defaults for in-game quotes (total against points: .724).
10. **Unknown.** What prod's archive and gamelog hold; poll times in winter; the snapshot hour of the old backfills; whether NFL 2024-25 and NBA 2024 single-stamp rows are in-game; the effect on model metrics (needs the retrain).

## How to read this note

Each claim carries one of two tags.

- **[M name]** means measured: the script `name.py` prints the number.
- **[C file:line]** means inferred from reading code, not from running it.

Words used throughout:

| Word | Meaning |
|---|---|
| Game line | One sportsbook's quote for one team in one game. `Moneyline` is the team's win probability with the bookmaker margin removed. `Totals` is the team's implied score: half of (game total plus or minus the spread). [C `moneylines.py:451-481`] |
| Archive | The DuckDB file `archive/archive.duckdb`. Its `odds` table holds one row per observation: league, market, game date, team, book, value, stamp. |
| Game date | The Chicago calendar date of the scheduled start. [C `moneylines.py:453-458`] |
| Stamp | The column `observed_at`: when the row was observed, in UTC. |
| Key | One league, market, game date, team and book. |
| Team-game | One team in one game on one date in a gamelog. |
| Default | What the gamelog gets when the archive returns nothing: moneyline 0.5 and the league-average team total (MLB 4.671, NBA 111.667, WNBA 81.667, NFL 22.668, NHL 2.674). [C `stats/base.py:674-677`, `helpers/archive.py:397`] |
| Season | Calendar year for MLB and WNBA; the year the season began for NBA, NHL and NFL. |
| Leak test | The correlation between a team's archived implied total and the points or runs it then scored, and between its moneyline and whether it won. A quote taken during the game already knows part of the score, so the correlation jumps. |

## 1. Where the values come from and where they go

### 1.1 The path

1. Prod's `confer` job polls the Odds API five times a day and appends game-line rows to prod's archive, stamped with the time of the poll. [C `moneylines.py:484-516`, `helpers/archive.py:1361-1381`]
2. `Stats.update()` fetches finished games from the league API. For each new row it asks the archive for that team's Moneyline and Totals on that date and writes the answer into two gamelog columns, `moneyline` and `totals`. The function is `Stats._enrich_team_markets`; it calls `Archive.get_team_market_map` once per market. [C `stats/base.py:634-677`]
3. `get_team_market_map` takes, for each book, the **newest observation the archive holds**, whenever it was stamped, and averages the books with their weights. It has an `at` parameter that would cut by time; no caller passes it. [C `helpers/archive.py:764-830`; the only two calls are `stats/base.py:672-673`]
4. The gamelog is saved as `src/sportstradamus/data/leagues/{league}/gamelog.parquet`. [C `helpers/io.py:629`]

Because step 2 runs the day after a game, "newest" is often a quote taken while the game was being played.

### 1.2 Every call of `_enrich_team_markets`

| League | New rows, on every update | Whole gamelog, rarely |
|---|---|---|
| MLB | `stats/mlb.py:455`, inside `parse_game`: the rows of one finished game | `stats/mlb.py:936-938` |
| NFL | `stats/nfl.py:840`, inside `_enrich_new_rows`: rows not yet in the gamelog | `stats/nfl.py:557-561` |
| NBA | `stats/nba.py:650`: the new player rows | `stats/nba.py:673-677` |
| WNBA | `stats/wnba.py:130` | `stats/wnba.py:152-160` |
| NHL | `stats/nhl.py:585`, inside the new-gamelog merge | `stats/nhl.py:532-537` |

The whole-gamelog call runs only when `season_start` is more than 300 days old, or when the module constant `clean_data` is true. `clean_data` is `False`. [C `stats/base.py:224`]

`season_start` begins as a hand-set date and is replaced by the season opener that the activity feed observed, when that is newer. [C `stats/base.py:606-629`, `helpers/odds_budget.py:349-365`] Today the feed holds openers for NFL, MLB, WNBA and NHL, all from 2026, so the whole-gamelog call is off for those leagues in a normal run; NBA is not in the feed yet, so its update does not run at all. [C `src/sportstradamus/data/runtime/league_activity.json`, snapshot of 2026-10-03, read directly] The call fires for NHL and NBA when `SPORTSTRADAMUS_FORCE_UPDATE=1` is set, or when the feed snapshot is stale or missing, because their hand-set dates (2025-10-07 and 2025-10-21) are more than 300 days old. [C `stats/nhl.py:92`, `stats/nba.py:86`, `stats/base.py:619-628`, `helpers/odds_budget.py:379-383`]

So in practice a row's `moneyline` and `totals` are written once, on the day it is ingested, and never revisited.

### 1.3 Who reads the stored values

| Reader | Code | Training | Serving | What it reads |
|---|---|---|---|---|
| Game context of a past game: `Moneyline`, `Total`, `OppTotal`, and from those `Spread`, `GameTotal`, `Blowout` | `stats/base.py:1581-1592`, `1625-1629` | Yes, every matrix row | No. Serving takes the other branch (`1595-1624`) and reads the archive's newest quote for today's game | The game's own stored value |
| Profile slopes: `Player moneyline gain`, `Player totals gain`, `Defense moneyline gain`, `Defense totals gain` (MLB adds pitcher slopes) | `stats/base.py:1245-1358`, `stats/mlb.py:1056-1190` | Yes | **Yes** | Every past game in the 300-day window |
| MLB starter-win curve, used to price pitcher fantasy points | `stats/mlb.py:1531-1553` | Yes | **Yes** | Every past start whose stored moneyline is not 0.5 |

So the answer to "does any serve-time feature read a past game's stored value" is yes, in two places: the four slope features (they are in the shared feature list, `data/config/feature_filter.json:46-51`) and the MLB starter-win curve. Both read the gamelog of the box they run on. Gamelogs are not copied between dev and prod in either direction. [C `scripts/sync_to_prod.sh:6-38, 76-92`, `scripts/sync_from_prod.sh:50-56`]

The matrix cache holds the values under ten columns: `Moneyline`, `Total`, `OppTotal`, `Spread`, `GameTotal`, `Blowout` and the four slopes. [M m4c] For every league, the cached `Moneyline` and `Total` equal the gamelog's stored value on 100.0% of rows. [M m4_final, table F5]

### 1.4 Training reads of game lines that do not pass through the gamelog

These also take the newest quote for a past date. This change does not touch them.

| Reader | Code | Feeds |
|---|---|---|
| MLB market factor | `stats/mlb.py:1332-1335`, used at `1296` | The projected plate-appearance multiplier, clipped to plus or minus 8% |
| MLB starter-win probability | `stats/mlb.py:1502` | The book leg of pitcher fantasy cells |
| NHL goalie-win probability and goals against | `stats/nhl.py:865`, `912`, `987-989` | The book leg of NHL goalie cells |
| Book weights for Moneyline and Totals | `training/cli.py:666-672`, `training/calibration.py:203-209` | The weights `get_team_market_map` averages with, on dev and (through `book_weights.json`) on prod |

## 2. Which archive training reads, and what it holds

Training runs on the dev box and opens `archive/archive.duckdb` under the repository root, unless `SPORTSTRADAMUS_ARCHIVE_DB` points elsewhere. [C `helpers/archive.py:96, 388`] The file is 2.4 GB and holds 41,065,780 `odds` rows. [M m0_schema]

Game lines reach it three ways:

1. **Prod's live polls.** `scripts/sync_from_prod.sh` copies prod's archive and `merge-archives` folds it into dev's as a union of rows. Two rows are the same only if league, market, game date, team, book, value and stamp all match. [C `scripts/sync_from_prod.sh:87-113`, `scripts/merge_archives.py:8-17, 50`]
2. **Historical backfills run on dev.** `backfill_historical_odds` asks the Odds API for a past snapshot and writes it through the same `set_team_books`, which takes no stamp, so each row is stamped with the moment of writing. [C `scripts/backfill_historical_odds.py:283-287`, `helpers/archive.py:1361-1381, 1178`]
3. **Rows that predate stamps.** When stamps were added, every existing row got its game date at midnight as a placeholder. [C `helpers/archive.py:420-438`]

Nothing carries dev's archive to prod. [C `scripts/sync_to_prod.sh:6-38`]

### Table A. Game-line rows by kind of stamp [M m10_stamp_classes]

| League | Rows | No stamp | Placeholder (game date 00:00) | Backfill (stamped more than 28 h after the game date) | Live poll |
|---|---|---|---|---|---|
| MLB | 964,974 | 0 | 420,304 | 368,258 | 176,412 |
| NBA | 332,374 | 0 | 268,596 | 59,962 | 3,816 |
| NFL | 129,628 | 0 | 55,654 | 0 | 73,974 |
| NHL | 590,978 | 0 | 215,842 | 351,776 | 23,360 |
| WNBA | 128,621 | 0 | 45,457 | 980 | 82,184 |
| NCAAF (not trained) | 92,130 | 0 | 92,130 | 0 | 0 |
| All | 2,238,705 | 0 | 1,097,983 | 780,976 | 359,746 |

The 28-hour line separates the two stamped kinds cleanly: the latest live poll for any game is 25.5 hours after the start of its game date, and the earliest backfill write is 31.3 hours after. [M m10_stamp_classes]

### Table B. Moneyline by league and season [M m2_archive_profile]

Totals row counts are in the fourth column. The Totals shares are within about three points of Moneyline's, with one exception: in NBA 2025, 22.9% of Totals keys hold two or more observations against 2.2% for Moneyline. Those are the duplicate rows of section 5.4. Every Totals figure is in `m2_rows.csv` and `m2_keys.csv`.
"Keys" here fold the case of the team code, so an old placeholder row and a backfill row for one team and book count as one key with two observations.
This script drew the backfill line at 36 hours instead of 28; the only rows that fall between are 516 MLB rows for 2026-07-09.

| League | Season | Moneyline rows | Totals rows | Stamp present | Placeholder | Backfill | Live | Keys | Keys with 2 or more observations |
|---|---|---|---|---|---|---|---|---|---|
| MLB | 2022 | 78,034 | 68,216 | 100% | 100.0% | 0% | 0% | 78,034 | 0.0% |
| MLB | 2023 | 138,282 | 130,334 | 100% | 60.0% | 40.0% | 0% | 86,287 | 60.1% |
| MLB | 2024 | 103,268 | 102,390 | 100% | 54.0% | 46.0% | 0% | 58,707 | 75.7% |
| MLB | 2025 | 52,640 | 52,260 | 100% | 0.7% | 99.3% | 0% | 51,652 | 1.9% |
| MLB | 2026 | 119,292 | 120,258 | 100% | 0.0% | 26.3% | 73.7% | 49,780 | 34.0% |
| NBA | 2021 | 1,140 | 1,084 | 100% | 100.0% | 0% | 0% | 1,140 | 0.0% |
| NBA | 2022 | 52,610 | 50,724 | 100% | 100.0% | 0% | 0% | 52,610 | 0.0% |
| NBA | 2023 | 66,596 | 67,838 | 100% | 55.9% | 44.1% | 0% | 37,310 | 78.5% |
| NBA | 2024 | 21,528 | 21,624 | 100% | 100.0% | 0% | 0% | 21,528 | 0.0% |
| NBA | 2025 | 22,688 | 26,542 | 100% | 91.6% | 0% | 8.4% | 21,260 | 2.2% |
| NFL | 2022 | 11,984 | 10,986 | 100% | 100.0% | 0% | 0% | 11,984 | 0.0% |
| NFL | 2023 | 8,572 | 8,584 | 100% | 100.0% | 0% | 0% | 8,572 | 0.0% |
| NFL | 2024 | 3,649 | 3,561 | 100% | 100.0% | 0% | 0% | 3,649 | 0.0% |
| NFL | 2025 | 4,222 | 4,096 | 100% | 100.0% | 0% | 0% | 4,222 | 0.0% |
| NFL | 2026 | 36,738 | 37,236 | 100% | 0.0% | 0% | 100.0% | 1,326 | 99.8% |
| NHL | 2021 | 2,412 | 1,906 | 100% | 100.0% | 0% | 0% | 2,412 | 0.0% |
| NHL | 2022 | 107,672 | 96,354 | 100% | 50.9% | 49.1% | 0% | 55,084 | 92.9% |
| NHL | 2023 | 112,808 | 96,698 | 100% | 34.2% | 65.8% | 0% | 38,776 | 96.3% |
| NHL | 2024 | 44,876 | 44,436 | 100% | 40.2% | 59.8% | 0% | 27,156 | 64.2% |
| NHL | 2025 | 30,354 | 30,102 | 100% | 0.0% | 100.0% | 0% | 30,290 | 0.0% |
| NHL | 2026 | 11,698 | 11,662 | 100% | 0.0% | 0% | 100.0% | 778 | 94.9% |
| WNBA | 2022 | 4,810 | 4,552 | 100% | 100.0% | 0% | 0% | 4,810 | 0.0% |
| WNBA | 2023 | 6,610 | 6,546 | 100% | 100.0% | 0% | 0% | 6,610 | 0.0% |
| WNBA | 2024 | 5,658 | 5,693 | 100% | 100.0% | 0% | 0% | 5,658 | 0.0% |
| WNBA | 2025 | 5,650 | 5,702 | 100% | 100.0% | 0% | 0% | 5,650 | 0.0% |
| WNBA | 2026 | 38,552 | 44,848 | 100% | 0.3% | 1.3% | 98.5% | 6,329 | 93.3% |

### Table C. Where each key's newest observation sits, Moneyline [M m2_archive_profile]

Hours are UTC on the game date. Seasons not listed have 100% in "placeholder". Totals differ from these by up to five points (`m2_keys.csv`).

| League | Season | Placeholder | Before the game date | 00:00 to 12:00 | 12:00 to 17:00 | 17:00 to 23:00 | 23:00 to next-day 12:00 (live) | Backfill, days later |
|---|---|---|---|---|---|---|---|---|
| MLB | 2023 | 36.0% | 0% | 0% | 0% | 0% | 0% | 64.0% |
| MLB | 2024 | 19.3% | 0% | 0% | 0% | 0% | 0% | 80.7% |
| MLB | 2025 | 0.0% | 0% | 0% | 0% | 0% | 0% | 100.0% |
| MLB | 2026 | 0.0% | 0.5% | 0.0% | 7.9% | 17.2% | 11.6% | 62.8% |
| NBA | 2023 | 21.2% | 0% | 0% | 0% | 0% | 0% | 78.8% |
| NBA | 2025 | 97.6% | 0.3% | 0.0% | 0.0% | 1.9% | 0.2% | 0% |
| NFL | 2026 | 0.0% | 24.1% | 0.0% | 26.7% | 38.6% | 10.6% | 0% |
| NHL | 2022 | 6.6% | 0% | 0% | 0% | 0% | 0% | 93.4% |
| NHL | 2023 | 3.7% | 0% | 0% | 0% | 0% | 0% | 96.3% |
| NHL | 2024 | 2.4% | 0% | 0% | 0% | 0% | 0% | 97.6% |
| NHL | 2025 | 0.0% | 0% | 0% | 0% | 0% | 0% | 100.0% |
| NHL | 2026 | 0.0% | 3.9% | 0.0% | 36.8% | 19.0% | 40.4% | 0% |
| WNBA | 2026 | 1.2% | 1.5% | 0.3% | 6.2% | 56.0% | 27.1% | 7.7% |

For 320 of 1,326 NFL keys the newest quote was taken before the game date. Only 32 of those belong to games not yet played on 2026-10-05; I did not look into why the other 288 have no game-day quote. [M m12_newest_before_gameday]

Among keys that were polled live, the newest quote is later than 12:00 UTC on the game date for 98.7% (MLB, 18,508 keys), 98.1% (WNBA, 5,763), 96.1% (NHL, 778), 87.8% (NBA, 508) and 75.9% (NFL, 1,326). It is at or after 23:00 UTC for 31.3%, 29.8%, 40.4%, 6.7% and 10.6%. [M m2_archive_profile, file `m2_late.csv`]

### Table D. When the backfills were written [M m2_archive_profile, file `m2_backfill_days.csv`]

| Written on | League | Moneyline rows | Totals rows | Game dates covered |
|---|---|---|---|---|
| 2026-07-01 | WNBA | 488 | 492 | 2026-06-20 to 2026-06-30 |
| 2026-07-10 | MLB | 81,740 | 81,348 | 2025-03-28 to 2026-07-08 |
| 2026-07-10 | NHL | 72,840 | 67,426 | 2023-10-10 to 2026-06-14 |
| 2026-07-10 | NBA | 29,394 | 30,568 | 2023-10-24 to 2024-06-17 |
| 2026-08-04 | MLB | 24,402 | 22,012 | 2023-05-04 to 2023-06-29 |
| 2026-08-05 | MLB | 79,000 | 76,506 | 2023-06-30 to 2025-03-28 |
| 2026-08-05 | NHL | 111,470 | 100,040 | 2022-10-07 to 2025-03-26 |
| 2026-08-06 | MLB | 1,372 | 1,362 | 2026-08-01 to 2026-08-05 |

A backfill snapshot is taken "as of" an hour on the game date: 06:00 UTC unless the operator chose another, and never later than 12:00 UTC for this kind of run since 2026-07-10. [C `scripts/backfill_historical_odds.py:109, 235-236, 321, 392-395`] The hour used for each run above was not logged.

### Table E. When live polls run [M m10_stamp_classes]

| Month | Poll hours seen (UTC) | Live Moneyline rows |
|---|---|---|
| 2026-05 | 01, 03, 13, 17, 23 | 3,892 |
| 2026-06 | 13, 17 | 3,254 |
| 2026-07 | 01, 13, 14, 16, 17, 19, 22 | 26,364 |
| 2026-08 | 01, 13, 16, 19, 22 | 40,508 |
| 2026-09 | 01, 13, 16, 19, 22 | 92,896 |
| 2026-10 | 01, 13, 16, 19, 22 | 9,066 |

The polls land at half past the hour. The morning poll stamped between 13:30.1 and 13:30.2 on all 85 days since 2026-07-11. [M m10_stamp_classes]
First live stamp per league: NBA and WNBA 2026-05-08, MLB 2026-07-11, NFL 2026-07-24, NHL 2026-09-23. [M m10_stamp_classes]
The crontab line is `30 8,11,14,17,20 * * *` in the server's local time. [C `docs/OPERATIONS.md:43`] That matches the summer stamps if the server keeps Chicago time, and would put the polls one hour later in UTC during winter. No winter poll exists yet to confirm it.

### Table F. The earlier "77.9% of quotes are late" figure [M m6_brief_figure]

The research note that prompted this change reported that 77.9% of the newest MLB 2026 quotes were observed at or after 23:00 UTC on game day. The figure reproduces (77.0% over 46,400 keys), but most of it is backfill rows carrying the date the backfill ran.

| Newest observation of the key | Share of MLB 2026 Moneyline keys |
|---|---|
| At or after 23:00 UTC on the game date, all | 77.0% |
| ... of which a backfill row written days later (a pre-game snapshot) | 68.0% |
| ... of which a live poll | 9.1% |
| A live poll between 12:00 and 23:00 UTC | 22.5% |
| At or before 12:00 UTC | 0.5% |

The leak is real all the same; section 3 shows it. It is confined to games that were polled live.

## 3. Which cutoff

### 3.1 A fixed time on the game date, and the rule the prop quotes already follow

The prop readers work like this. For a game date D, training assumes a start of D 00:00 plus 20 hours, which is 20:00 UTC, and subtracts `TRAINING_LOOKBACK_HOURS = 8`. The result is **12:00 UTC on D**. Each book's newest observation stamped at or before that moment is used. [C `stats/base.py:2393-2399`, `helpers/archive.py:115-118`] Prop backfills are stamped D 01:00 so they pass it. [C `moneylines.py:716-727`]

Is that rule sound for game lines? As a guard against in-game quotes, yes: 12:00 UTC is 7 am in Chicago in summer, before every regular start. As a match for what serving reads, less so. With today's poll times (Table E) the only poll at or before 12:00 UTC on a game date is the 01:30 UTC one, which is the evening before in the United States. Many next-day MLB lines are not up by then (Table H: 56.8% of July team-games had one). The first serving run of a day is at 13:50 UTC and already reads the 13:30 poll. [C `docs/OPERATIONS.md:42`, crontab line `50 8-20 * * *`]

### 3.2 The game's own start time

| Place looked | Holds a start time? |
|---|---|
| The five gamelogs | No. Each has one date column and nothing finer. NFL's `gametime` is dropped on load. [M m1_gamelog_inventory] [C `stats/nfl.py:537`] |
| The five team logs | No. [M m1_gamelog_inventory] |
| The archive (`odds`, `lines`, `ladder`) | No. The schema stops at the game date. [M m0_schema] |
| The matrix column `GameTime` | No. It is the stand-in "game date plus 20 hours". [C `stats/base.py:2398-2405`] |

Coverage of this candidate is 0% in every league. The league schedule feeds do carry start times when they are fetched; using them means storing a new column and filling four years of history, which is a larger change than this one. I did not trace the other four feeds line by line.

### 3.3 Simpler candidates

| Candidate | Result |
|---|---|
| Each book's **first** observation instead of its newest; no hour needed | Coverage 100%, but 4.5% of MLB, 7.4% of NBA and 1.8% of WNBA live team-games were first seen after 15:00 UTC on game day, so it is not pre-game without a cutoff of its own. NFL lines are first seen a median of 124.5 hours ahead, and sit five times further from the 17:00 UTC value than the 15:00 cutoff does (.0233 against .0045 on moneyline). [M m8_first_quote] Rejected. |
| Stop archiving game lines once a game has started | Would need a start time for the 359,746 live rows already stored, changes what every other reader of the archive gets, and leaves training on closing lines that serving never sees. Not pursued. |
| A fixed UTC hour on each row's own game date, inside the one bulk query | The form recommended below. It needs no per-game data and no loop. |

### 3.4 Coverage: how many team-games still get a quote

Table G covers whole seasons and shows what a cutoff does when the stamps are left as they are ("plain") and after the backfill rows are moved onto their game date ("moved"; section 4). [M m4_final, table F1]

| League | Season | Team-games | Newest (today) | Plain 12:00 | Plain 15:00 | Moved 12:00 | Moved 15:00 |
|---|---|---|---|---|---|---|---|
| MLB | 2024 | 4,874 | 99.2% | 99.0% | 99.0% | 99.2% | 99.2% |
| MLB | 2025 | 4,884 | 99.4% | 0.6% | 0.6% | 99.4% | 99.4% |
| MLB | 2026 | 4,524 | 94.4% | 22.9% | 28.8% | 87.2% | 93.1% |
| NBA | 2023 | 176 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| NBA | 2024 | 2,642 | 91.7% | 91.7% | 91.7% | 91.7% | 91.7% |
| NBA | 2025 | 2,644 | 89.6% | 89.4% | 89.5% | 89.4% | 89.5% |
| NFL | 2020 | 442 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| NFL | 2021 | 570 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| NFL | 2022 | 568 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| NFL | 2023 | 570 | 99.6% | 99.6% | 99.6% | 99.6% | 99.6% |
| NFL | 2024 | 570 | 91.9% | 91.9% | 91.9% | 91.9% | 91.9% |
| NFL | 2025 | 570 | 93.3% | 93.3% | 93.3% | 93.3% | 93.3% |
| NFL | 2026 | 96 | 93.8% | 93.8% | 93.8% | 93.8% | 93.8% |
| NHL | 2022 | 2,800 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| NHL | 2023 | 2,802 | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% |
| NHL | 2024 | 2,810 | 99.2% | 76.5% | 76.5% | 99.2% | 99.2% |
| NHL | 2025 | 2,788 | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| WNBA | 2021 | 379 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| WNBA | 2022 | 456 | 46.1% | 46.1% | 46.1% | 46.1% | 46.1% |
| WNBA | 2023 | 491 | 51.5% | 51.5% | 51.5% | 51.5% | 51.5% |
| WNBA | 2024 | 504 | 94.8% | 94.8% | 94.8% | 94.8% | 94.8% |
| WNBA | 2025 | 622 | 97.1% | 97.1% | 97.1% | 97.1% | 97.1% |
| WNBA | 2026 | 539 | 87.2% | 82.4% | 84.2% | 85.3% | 87.2% |

The archive has no NFL game lines before 2022-09 and no WNBA lines for 2021, which is why those seasons read 0% under every rule.

Table H keeps only the team-games that were polled live. This is where the choice of hour matters. [M m4_final, tables F1b and F1c]

| League | Team-games | Dates | Newest | 12:00 UTC | 15:00 UTC | 17:00 UTC |
|---|---|---|---|---|---|---|
| MLB | 1,364 | 2026-07-11 to 2026-09-16 | 100% | 76.0% | 95.5% | 95.5% |
| MLB, July | 458 | | 100% | 56.8% | 93.4% | 93.4% |
| MLB, August | 476 | | 100% | 83.2% | 93.3% | 93.3% |
| MLB, September | 430 | | 100% | 88.4% | 100.0% | 100.0% |
| NBA | 54 | 2026-05-08 to 2026-06-13 | 100% | 88.9% | 92.6% | 92.6% |
| NFL | 90 | 2026-09-09 to 2026-09-28 | 100% | 100.0% | 100.0% | 100.0% |
| WNBA | 454 | 2026-05-15 to 2026-08-30 | 100% | 97.8% | 100.0% | 100.0% |
| NHL | 0 | dev's gamelog ends 2026-06-14; live polls began 2026-09-23 | | | | |

### 3.5 Leak test

Table I uses the same team-games under all three rules: polled live, one game that day, a quote available under each rule. The last column is the rough standard error of one correlation. [M m4_final, table F2]

| League | Team-games | Total against score: newest | 12:00 | 15:00 | Moneyline against win: newest | 12:00 | 15:00 | s.e. |
|---|---|---|---|---|---|---|---|---|
| MLB | 1,028 | .584 | .167 | .181 | .581 | .260 | .262 | .031 |
| NFL | 90 | .724 | .339 | .352 | .512 | .243 | .241 | .107 |
| WNBA | 444 | .577 | .464 | .473 | .540 | .483 | .493 | .048 |
| NBA | 48 | .216 | .151 | .194 | .236 | .216 | .234 | .149 |

What the dev gamelog holds today for those live-polled team-games is as leaky as "newest": stored total against score is .582 for MLB (948 team-games) and .559 for WNBA (346). [M m11_live_era_stored]

Table J is the reference: seasons whose only quotes are backfilled pre-game snapshots. [M m3b_tables, file `m3b_T2b_leak_old_eras.csv`]

| League | Season | Team-games | Total against score | Moneyline against win |
|---|---|---|---|---|
| MLB | 2025 | 4,771 | .164 | .174 |
| MLB | 2026 | 2,866 | .185 | .122 |
| NHL | 2024 | 638 | .160 | .273 |
| NHL | 2025 | 2,788 | .128 | .167 |

MLB at either cutoff sits inside that clean band. NFL at either cutoff matches its 2022 and 2023 seasons (.302 and .344, Table M). WNBA stays high at .46 to .47, but its undated 2024 and 2025 seasons read .495 and .400, so a high pre-game correlation looks normal for that league; with no start times I cannot prove it.

Table K asks when the score enters the quote. It correlates how far the quote moved after 12:00 UTC with how far the result landed from the 12:00 quote. A pre-game move predicts the result a little; an in-game move predicts it a lot. [M m4b_extra, table G1]

| League | Pair | Team-games | Move to 15:00 | to 17:00 | to 20:00 | to 23:00 | to newest |
|---|---|---|---|---|---|---|---|
| MLB | total, score | 1,028 | .073 | .099 | .245 | .308 | .571 |
| MLB | moneyline, win | 1,028 | .050 | .050 | .238 | .309 | .544 |
| NFL | total, score | 90 | .207 | .100 | .575 | .599 | .678 |
| NFL | moneyline, win | 90 | -.049 | -.016 | .372 | .457 | .476 |
| WNBA | total, score | 444 | .103 | .110 | .138 | .178 | .386 |
| WNBA | moneyline, win | 444 | .121 | .147 | .155 | .185 | .276 |
| NBA | total, score | 48 | .201 | .201 | .265 | .265 | .215 |
| NBA | moneyline, win | 48 | .364 | .364 | .290 | .290 | .281 |

The score is in the MLB and NFL quotes by 20:00 UTC. Nothing of that size appears between 12:00 and 15:00. The NBA rows rest on 48 team-games and say little.

### 3.6 Distance from what serving reads

Serving runs hourly from 13:50 UTC and reads the newest quote at that moment. The 17:00 UTC snapshot (the 16:30 poll) stands for a typical afternoon decision. Later snapshots are shown too, but Table K says they already contain in-game quotes for early games, so they overstate the distance. [M m4_final, table F3]

| League | Value | Cutoff | Mean distance to 17:00 | to 20:00 | to 23:00 | to newest | Spread (s.d.) of the value itself |
|---|---|---|---|---|---|---|---|
| MLB | moneyline | 12:00 | .0134 | .0522 | .0811 | .1946 | .0839 |
| MLB | moneyline | 15:00 | .0076 | .0436 | .0729 | .1711 | .0854 |
| MLB | total | 12:00 | .183 | .397 | .611 | 1.476 | .731 |
| MLB | total | 15:00 | .082 | .305 | .511 | 1.279 | .749 |
| NFL | moneyline | 12:00 | .0059 | .0682 | .1221 | .1549 | .1705 |
| NFL | moneyline | 15:00 | .0045 | .0669 | .1217 | .1552 | .1694 |
| NFL | total | 12:00 | .205 | 2.686 | 3.552 | 4.360 | 3.140 |
| NFL | total | 15:00 | .153 | 2.642 | 3.538 | 4.366 | 3.181 |
| WNBA | moneyline | 12:00 | .0255 | .0351 | .0408 | .0669 | .2081 |
| WNBA | moneyline | 15:00 | .0078 | .0214 | .0276 | .0543 | .2114 |
| WNBA | total | 12:00 | 1.044 | 1.193 | 1.324 | 2.439 | 5.686 |
| WNBA | total | 15:00 | .483 | .966 | 1.071 | 2.231 | 5.742 |
| NBA | moneyline | 12:00 | .0057 | .0085 | .0085 | .0084 | .1548 |
| NBA | moneyline | 15:00 | .0000 | .0056 | .0056 | .0055 | .1529 |
| NBA | total | 12:00 | .712 | .967 | .967 | .992 | 3.349 |
| NBA | total | 15:00 | .000 | .810 | .810 | .854 | 3.270 |

The 15:00 value is about twice as close to the afternoon value as the 12:00 value in MLB and WNBA. What serving actually read cannot be compared directly: the serve-time feature log began on 2026-10-05.

### 3.7 The choice

**15:00 UTC on the game date.**

| Question | 12:00 UTC (the prop rule) | 15:00 UTC |
|---|---|---|
| Removes the leak (Table I) | Yes | Yes, same within error |
| Live-polled MLB team-games left without a quote (Table H) | 24.0% | 4.5% |
| Live-polled WNBA / NBA / NFL left without a quote | 2.2% / 11.1% / 0% | 0% / 7.4% / 0% |
| Polls it admits on game day | 01:30 UTC only (the evening before) | 01:30 and the morning poll (13:30 UTC; 14:30 in winter if the server shifts) |
| Equals what a serving run reads | Only the late-evening run the day before | The first run of the day, 13:50 UTC |
| Games that may have started | None in regular play | Morning-kickoff NFL games abroad: the poll lands at kickoff (13:30 UTC). About 14 team-games a season if seven such games are played; my estimate, not a measurement |
| One rule for props and game lines | Yes | No, two constants |

15:00 is the earliest whole hour that admits the morning poll whether the server's clock shifts in winter or not, and the latest that precedes the earliest regular United States starts (11 am Eastern day games, 15:00 UTC in summer). 16:00 or later reads the same polls today, but would admit a quote taken after such a start if a poll slot were ever added between.

The start times in this paragraph and in the table come from general knowledge of league schedules. No file on disk can confirm them (section 3.2).

If the owner prefers one rule for every archive input, 12:00 UTC is the fallback; the cost is the first two rows of the table.

## 4. Rows with no stamp, and rows with one observation

No game-line row has an empty stamp. The brief expected some; the migration that added stamps filled them all with the placeholder. [M m10_stamp_classes]

### Table L. Keys by how many observations they hold [M m10_stamp_classes]

Keys here keep the team code exactly as stored.

| League | Keys | One observation | ... a placeholder | ... a backfill row | ... a live poll | Backfilled more than once | With a live poll |
|---|---|---|---|---|---|---|---|
| MLB | 823,080 | 787,326 | 420,304 | 364,234 | 2,788 | 2,012 | 36,530 |
| NBA | 325,268 | 320,094 | 259,984 | 59,962 | 148 | 0 | 1,016 |
| NFL | 58,316 | 55,658 | 55,654 | 0 | 4 | 0 | 2,662 |
| NHL | 497,428 | 425,596 | 215,842 | 209,674 | 80 | 70,356 | 1,556 |
| WNBA | 58,029 | 46,279 | 45,457 | 492 | 330 | 0 | 12,080 |
| All, with NCAAF | 1,854,251 | 1,727,083 | 1,089,371 | 634,362 | 3,350 | 72,368 | 53,844 |

93.1% of keys hold exactly one observation.

### What each kind is, and what a cutoff does to it

| Kind | Rows | Seasons | A cutoff with stamps left as they are | Proposed |
|---|---|---|---|---|
| No stamp | 0 | none | nothing to do | nothing to do |
| Placeholder (game date 00:00) | 1,097,983 | Everything before live polling: all leagues 2022; NFL and WNBA to 2025; NBA to 2026-05-07; MLB to 2024; NHL to 2024 | Kept: midnight is before any cutoff | Keep as they are |
| Backfill, stamped when written | 780,976 | MLB 2023-05-04 to 2026-07-09 and 2026-08-01 to 08-05; NHL 2022-10-07 to 2026-06-14; NBA 2023-10-24 to 2024-06-17; WNBA 2026-06-20 to 06-30 | **Dropped**: the stamp is weeks to years after the game | Move the stamp onto the game date, once |
| Live poll, single | 3,350 keys | 2026 | Kept if stamped by the cutoff | Same |
| Live poll, several | 50,494 keys | 2026 | Newest one stamped by the cutoff | Same |

Dropping the backfill rows is what empties MLB 2025 and NHL 2025 in Table G: 11,189 team-games in the dev gamelog (MLB 7,747, NHL 3,426, WNBA 16) have backfill rows and nothing else. [M m4_final, table F4b]

### Are the single-stamp values pre-game?

For backfill rows, yes by the leak test (Table J): .13 to .19 on total against score.

For placeholder rows the stamp says nothing, so only the leak test can speak. [M m3b_tables, file `m3b_T2b_leak_old_eras.csv`]

### Table M. Placeholder-only team-games

| League | Season | Team-games | Total against score | Moneyline against win | Reading of the total |
|---|---|---|---|---|---|
| MLB | 2024 | 478 | .172 | .212 | Looks pre-game |
| NFL | 2022 | 568 | .302 | .405 | Looks pre-game |
| NFL | 2023 | 568 | .344 | .368 | Looks pre-game |
| NFL | 2024 | 524 | .591 | .592 | Looks partly in-game |
| NFL | 2025 | 532 | .501 | .400 | Looks partly in-game |
| NBA | 2024 | 2,422 | .492 | .444 | Unclear; high |
| NBA | 2025 | 2,316 | .211 | .472 | Looks pre-game |
| WNBA | 2022 | 210 | .206 | .533 | Looks pre-game |
| WNBA | 2023 | 253 | .312 | .454 | Looks pre-game |
| WNBA | 2024 | 476 | .495 | .518 | In line with WNBA at the cutoff in 2026 (.47) |
| WNBA | 2025 | 604 | .400 | .415 | Same |

These readings are judgements from one number per season, set against the same league's clean values in Tables I and J. They are not proof either way.

No cutoff can repair a placeholder row: there is one observation and no time on it. NFL 2024-25 (1,056 team-games) and possibly NBA 2024 stay as they are under this change. Only a fresh historical snapshot would replace them, and with the writer fix below it would win on recency. That is a spend decision, not part of this design.

### The rule for backfill rows, with numbers

Move each backfill row's stamp to 01:00 UTC on its own game date, plus its write order in microseconds so that where a key was backfilled twice the later write still wins. 01:00 is the convention prop backfills already use. [C `moneylines.py:716-727`]

```sql
UPDATE odds SET observed_at = s.new_at
FROM (
    SELECT rowid AS rid,
           CAST(game_date AS TIMESTAMP) + INTERVAL 1 HOUR
           + to_microseconds(ROW_NUMBER() OVER (
                 PARTITION BY league, market, game_date, entity, book ORDER BY observed_at)) AS new_at
    FROM odds
    WHERE market IN ('Moneyline', 'Totals') AND observed_at > game_date + INTERVAL 28 HOUR
) AS s
WHERE odds.rowid = s.rid;
```

Tested on an in-memory copy of all 2,238,705 game-line rows; the archive file itself was opened read-only. [M m7_restamp_dryrun]

| Check | Result |
|---|---|
| Rows moved | 780,976 |
| Rows still stamped more than 28 h late | 0 |
| New stamps | 01:00:00.000001 to 01:00:00.000017 on the game date |
| Keys with no live poll: the cutoff reader after the move against today's reader | 1,800,407 keys; 0 lose their quote; 0 pick a different row. Same at 12:00 and at 15:00 |
| Keys with a live poll, 15:00 cutoff | 53,844 keys; 2,052 have no quote by the cutoff; 44,339 read an earlier quote than today |
| Keys with a live poll, 12:00 cutoff | 53,844 keys; 16,606 have no quote by the cutoff; 33,851 read an earlier quote |
| Time for the UPDATE on the copy | 0.1 s |

Two side notes that the move does not change:

- Placeholder rows store the team code in mixed case (`Nyy`); backfill and live rows store it in capitals. [M m10_stamp_classes: no exact key holds both kinds] The bulk reader splits by the exact code and then merges by the capitalised one, so where a team-game has both kinds, both sets of books enter the average. That is so today and stays so. [C `helpers/archive.py:803-829`]
- 4,306 NBA Totals keys hold two different values under one stamp (section 5.4). The reader's pick between them is arbitrary, before and after. [M m7_restamp_dryrun, m9_nba_duplicates]

## 5. What changes on disk

### 5.1 Four different things happen when a stored row is read again

| Effect | Which rows | Size |
|---|---|---|
| The cutoff | Team-games polled live | Large per row: Table P |
| A fill | Rows stored at the default because the archive had no line when they were ingested, and has one now | Whole seasons: MLB 2025, NHL 2025, most of MLB 2026 |
| A small shift in a quote that was already there | Every row with a quote: the book weights were refit since, and where a season was backfilled later the new books join the average | Small: median moneyline shift .001 to .011 in seasons with no live polls (Table O) |
| The archive holds something worse than before | NBA 2026-03-14 to 04-24 | +33 points on 382 team-games (section 5.4) |

Only the first is the approved change. The others ride along with any re-enrichment and are larger in row count.

### Table N. Gamelog, every row re-enriched at 15:00 UTC, by league [M m4_final, table F4]

| League | Team-games | Changed | Default to quote | Quote to default | Moneyline moves by more than .02 | Moneyline change: mean | s.d. | Total change: mean | s.d. |
|---|---|---|---|---|---|---|---|---|---|
| MLB | 14,282 | 97.6% | 56.7% | 0.3% | 56.5% | .0000 | .0948 | -.229 | .901 |
| NBA | 5,462 | 91.0% | 1.3% | 0.1% | 8.0% | -.0017 | .0649 | +2.416 | 9.507 |
| NFL | 3,386 | 67.7% | 2.5% | 0.4% | 6.1% | .0000 | .0295 | -.020 | .575 |
| NHL | 11,200 | 99.8% | 30.6% | 0.0% | 35.9% | .0000 | .0654 | +.104 | .395 |
| WNBA | 2,991 | 67.4% | 3.7% | 0.1% | 9.8% | .0004 | .0479 | +.133 | 1.794 |

The moneyline mean is zero by construction: the two teams in a game move in opposite directions.

### Table O. The same, by season [M m4c_by_season]

| League | Season | Team-games | Polled live | Changed | Default to quote | Quote to default | Quote to another quote | Median moneyline shift among those | Moneyline s.d. | Total mean | Total s.d. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MLB | 2024 | 4,874 | 0 | 99.2% | 0.0% | 0.0% | 99.2% | .0063 | .0375 | +.027 | .347 |
| MLB | 2025 | 4,884 | 0 | 99.4% | 98.8% | 0.0% | 0.6% | .0052 | .0941 | -.420 | .755 |
| MLB | 2026 | 4,524 | 1,364 | 93.8% | 72.1% | 0.8% | 20.8% | .1163 | .1316 | -.299 | 1.304 |
| NBA | 2023 | 176 | 0 | 100.0% | 0.0% | 0.0% | 100.0% | .0062 | .0119 | -.141 | .681 |
| NBA | 2024 | 2,642 | 0 | 91.7% | 0.0% | 0.1% | 91.7% | .0033 | .0086 | -.001 | .286 |
| NBA | 2025 | 2,644 | 54 | 89.6% | 2.6% | 0.2% | 86.8% | .0023 | .0929 | +5.001 | 13.178 |
| NFL | 2020 | 442 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | | .0000 | .000 | .000 |
| NFL | 2021 | 570 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | | .0000 | .000 | .000 |
| NFL | 2022 | 568 | 0 | 100.0% | 0.0% | 0.0% | 100.0% | .0033 | .0099 | +.009 | .129 |
| NFL | 2023 | 570 | 0 | 99.6% | 0.0% | 0.0% | 99.6% | .0034 | .0095 | -.004 | .198 |
| NFL | 2024 | 570 | 0 | 93.7% | 0.0% | 2.1% | 91.9% | .0068 | .0188 | -.044 | .371 |
| NFL | 2025 | 570 | 0 | 93.3% | 0.0% | 0.0% | 93.3% | .0044 | .0103 | -.024 | .297 |
| NFL | 2026 | 96 | 90 | 93.8% | 87.5% | 0.0% | 6.2% | .1581 | .1646 | -.325 | 3.163 |
| NHL | 2022 | 2,800 | 0 | 100.0% | 0.0% | 0.0% | 100.0% | .0070 | .0386 | +.026 | .132 |
| NHL | 2023 | 2,802 | 0 | 99.9% | 0.0% | 0.0% | 99.9% | .0067 | .0366 | -.025 | .157 |
| NHL | 2024 | 2,810 | 0 | 99.2% | 22.6% | 0.0% | 76.7% | .0108 | .0726 | +.038 | .400 |
| NHL | 2025 | 2,788 | 0 | 100.0% | 100.0% | 0.0% | 0.0% | | .0950 | +.378 | .566 |
| WNBA | 2021 | 379 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | | .0000 | .000 | .000 |
| WNBA | 2022 | 456 | 0 | 46.1% | 0.0% | 0.0% | 46.1% | .0012 | .0125 | -.074 | .720 |
| WNBA | 2023 | 491 | 0 | 51.1% | 0.0% | 0.4% | 50.9% | .0016 | .0120 | -.071 | .604 |
| WNBA | 2024 | 504 | 0 | 94.8% | 0.0% | 0.0% | 94.8% | .0013 | .0129 | -.005 | .328 |
| WNBA | 2025 | 622 | 0 | 97.4% | 0.0% | 0.3% | 97.1% | .0017 | .0041 | -.053 | .183 |
| WNBA | 2026 | 539 | 454 | 87.2% | 20.8% | 0.0% | 66.4% | .0176 | .1110 | +.930 | 4.024 |

### Table P. The cutoff alone: 15:00 against "newest", both read today [M m4_final, table F4b]

| League | Kind of team-game | Team-games | Changed by the cutoff | Lose their quote | Moneyline change s.d. | Total change s.d. |
|---|---|---|---|---|---|---|
| MLB | Polled live | 1,364 | 92.4% | 4.5% | .2489 | 2.259 |
| WNBA | Polled live | 454 | 98.7% | 0.0% | .1101 | 3.957 |
| NFL | Polled live | 90 | 100.0% | 0.0% | .2017 | 6.032 |
| NBA | Polled live | 54 | 88.9% | 7.4% | .0368 | 1.648 |
| MLB | Backfill and placeholder kinds | 12,600 | 0.0% | 0.0% | .0000 | .000 |
| NHL | Backfill and placeholder kinds | 11,176 | 0.0% | 0.0% | .0000 | .000 |
| NFL | Placeholder | 2,192 | 0.0% | 0.0% | .0000 | .000 |
| WNBA | Backfill and placeholder kinds | 1,561 | 0.0% | 0.0% | .0000 | .000 |
| NBA | Backfill and placeholder kinds | 4,916 | 0.1% | 0.0% | .0000 | .015 |

The NBA 0.1% is the arbitrary pick among same-stamp duplicates, not the cutoff.

### Table Q. Cached training matrices [M m4_final table F5; m4b_extra table G2]

Each matrix row was matched to the gamelog by player and date (100.0% matched in every league) and its stored `Moneyline` and `Total` compared with the re-enriched value.

| | MLB | NBA | WNBA | NHL | NFL |
|---|---|---|---|---|---|
| Matrix files | 19 | 21 | 18 | 15 | 20 |
| Rows | 732,791 | 299,565 | 260,005 | 518,628 | 209,458 |
| Dates | 2025-04-01 to 2026-09-16 | 2025-11-12 to 2026-06-13 | 2022-05-07 to 2026-08-30 | 2024-02-10 to 2026-06-14 | 2021-09-12 to 2026-09-28 |
| Rows at the default today | 95.4% | 12.1% | 26.4% | 50.9% | 23.1% |
| Rows with a real line today | 34,007 (4.6%) | 263,254 (87.9%) | 191,464 (73.6%) | 254,743 (49.1%) | 161,050 (76.9%) |
| ... of which from a live-polled game | 33,348 (98.1%) | 6,564 (2.5%) | 19,168 (10.0%) | 0 | 418 (0.3%) |
| Live-polled rows sitting at the default | 10,935 | 554 | 5,934 | 0 | 6,222 |
| Rows from live-polled games (all the cutoff can touch) | 6.0% | 2.4% | 9.7% | 0.0% | 3.2% |
| Rows the cutoff alone changes | 5.7% | 2.1% | 9.4% | 0.0% | 3.1% |
| After a full re-enrichment at 15:00: rows changed | 99.2% | 90.3% | 75.9% | 99.7% | 79.8% |
| ... default to quote | 94.6% | 2.4% | 2.4% | 50.6% | 3.0% |
| ... moneyline moves by more than .02 | 84.3% | 14.9% | 6.7% | 52.8% | 7.3% |
| ... moneyline change: mean / s.d. | .0007 / .1025 | -.0026 / .0985 | -.0021 / .0409 | .0011 / .0778 | .0002 / .0321 |
| ... total change: mean / s.d. | -.404 / .915 | +5.855 / 14.111 | -.019 / 1.296 | +.174 / .485 | -.021 / .621 |
| ... rows with a real line | 99.2% | 90.2% | 75.9% | 99.7% | 79.5% |

Read the MLB column twice. Today 95.4% of MLB matrix rows have no game line at all, and 98.1% of the few that do come from live-polled games, where the stored total correlates .582 with runs scored. The MLB models have learned what a game line means almost only from leaked values, and then meet a real pre-game line on every row at serving.

Why so many live-polled rows sit at the default on dev (MLB 416 of 1,364 team-games, WNBA 108 of 454 [M m11_live_era_stored]) I did not establish. It is consistent with dev ingesting games before the next `sync_from_prod` brought their polls over.

### 5.2 Can old rows be re-enriched with the code as it stands?

Not by any normal run. The whole-gamelog call needs one of these (section 1.2):

| Way | Leagues it reaches today | Side effects |
|---|---|---|
| `SPORTSTRADAMUS_FORCE_UPDATE=1` on an update | NHL and NBA only | Runs a full update against the league APIs |
| Set `clean_data = True` for one run | All five | Edits a source file; also runs a full update |
| Call the function directly on a loaded gamelog and save | All five, any subset of rows | None beyond the write. Uses only existing functions: `load()`, `_enrich_team_markets(frame, date_col=..., team_col=..., mask=...)`, `write_gamelog(...)` [C `stats/base.py:584, 634`, `helpers/io.py:629`] |

The third is about five lines per league in a Python shell. I did not run it: it writes repository files.

The matrix cache is a separate matter. A training run reads the cached matrix and builds only game days after its last date. [C `training/pipeline.py:836-852`] Old matrix rows keep their six game-context columns and four slopes until the matrix is rebuilt from nothing, or patched.

### 5.3 Cost of a one-off re-enrichment

| Step | Cost | Basis |
|---|---|---|
| Move the backfill stamps in the dev archive | 0.1 s on an in-memory copy; not timed on the 2.4 GB file. Take a file copy first | [M m7_restamp_dryrun] |
| Read the lines for one league and one market, every date | About 0.4 s: 80 such reads plus five gamelog loads took 35 s | [M m4_final] |
| Read and write one gamelog | Under 1 s each (2.3 to 9.1 MB, 27,098 to 213,335 rows) | [M m4c_by_season] |
| Patch six columns in 93 matrices | 1,394 MB on disk; the largest file reads in 0.1 s and writes in 0.7 s. Minutes in total. Needs a short script that joins each matrix to the gamelog by player and date | [M m4c_by_season] |
| Rebuild the matrices from nothing | Not measured: it needs `meditate`, which this task may not run. The repository describes it as "a full feature rebuild" and keeps `scripts/inject_backfilled_odds.py` to avoid one for the odds columns | [C `scripts/inject_backfilled_odds.py:1-20`] |

A patch fixes `Moneyline`, `Total`, `OppTotal`, `Spread`, `GameTotal` and `Blowout` exactly, because the cache equals the gamelog on 100% of rows and the last three are sums and differences of the first. [C `stats/base.py:1625-1629`] It leaves the four slope features stale in old rows; only a rebuild refreshes those. How much the slopes move was not measured.

### 5.4 The NBA hazard

For game dates 2026-03-14 to 2026-04-24 the dev archive holds two NBA Totals values for the same team, book and stamp: one about 1.4427 times the other. [M m9_nba_duplicates] The likely reason, which I did not verify on prod: the inflated rows were corrected on dev but not on prod, and each sync from prod brings them back beside the corrected ones, because a row's value is part of its identity in the merge. [C `scripts/merge_archives.py:50`]

| | |
|---|---|
| NBA Totals keys holding two values under one stamp | 4,306, over 482 team-games |
| Median ratio between the two values | 1.4427 |
| Dev gamelog team-games that would move by more than 5 points if re-read | 382 of 4,900 |
| Their mean move | +33.4 points, under "newest" and under the 15:00 cutoff alike |
| Their stored mean today / actual mean score | 114.7 / 116.0 |
| Every other NBA team-game: mean absolute move | .266 points |

The dev gamelog is right today and a whole NBA re-enrichment would make it wrong. The cutoff does not help: both values carry one stamp.

### 5.5 Is re-enrichment needed for the change to matter at the next retrain?

Yes for MLB, WNBA and NFL. No for NBA and NHL.

| League | Without re-enrichment, what the next retrain sees |
|---|---|
| MLB | The same 34,007 real-line matrix rows, 98.1% of them leaked. The season is over; almost no new rows arrive |
| WNBA | The same 19,168 leaked rows. The season is ending |
| NFL | New rows still fall to the default (section 6), so nothing changes |
| NBA | Its live-polled history is 54 team-games from the 2026 playoffs, with no leak signal (stored total against score .150, cutoff .182, about 50 team-games [M m11_live_era_stored]). The new season's rows will be read with the cutoff from the first game |
| NHL | No live-polled game is in the gamelog yet. Same as NBA |

A suggested scope, for the owner to decide:

| League | Gamelog rows to re-enrich | Why |
|---|---|---|
| MLB | All | Cuts the 1,364 live team-games and fills 2025 and 2026 from the backfill |
| NHL | All | No cut needed; fills 2025 and part of 2024 from the backfill |
| WNBA | All | Cuts 454 live team-games; fills 20.8% of 2026 |
| NFL | All | Fills 84 of the 90 defaulted 2026 team-games with pre-game values |
| NBA | None | Nothing to gain; 382 team-games to lose |

The fill is a bigger change to the training data than the cut. If both land in one retrain, a change in the scorecard cannot be assigned to either.

## 6. The NFL miss

Of the 96 NFL team-games in the 2026 season, 90 sit at the 0.5 default. For 84 of them the archive holds quotes under the right date and team today, first seen days before the game. The cause is a date mismatch at the moment of enrichment, not a missing row and not a team-name mismatch. New NFL rows are joined to the schedule on the week alone, because the schedule frame names its team column `recent_team` and the player frame does not; every row of a week therefore carries the week's first game day when the archive is asked. The true game day is written afterwards. [C `stats/nfl.py:543-554, 725-734, 902-930`] The six team-games that did get a line are exactly the two teams per week that played on the week's first game day. [M m5_nfl_miss] I did not run the NFL update to watch it happen, so the code reading is an inference that the measured pattern supports.

| Reason for the stored value | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| Default, though the archive has that team on that date today | 0 | 0 | 0 | 0 | 0 | 0 | 84 |
| Default: archive has no row for that date | 442 | 570 | 0 | 2 | 8 | 6 | 2 |
| Default: date archived, this team missing | 0 | 0 | 0 | 0 | 26 | 32 | 4 |
| Real line stored | 0 | 0 | 568 | 568 | 536 | 532 | 6 |

What this means for the cutoff, in one line each:

- The cutoff is keyed on the game date, so a row looked up under the wrong date misses with or without it.
- Any repair that enriches after the true game day is set will pass through the cutoff reader without further work.
- The repair should not land before the cutoff: it would replace 84 defaults with quotes whose total correlates .724 with points scored, instead of .352 (Table I).
- The six team-games enriched today were given the newest quote, which for a night game is a poll taken during play (inferred; six rows are too few to test). A re-enrichment replaces them with the 15:00 values.

## 7. Design

### 7.1 Code: three files, about fifteen changed lines

**`helpers/archive.py`**, beside `TRAINING_LOOKBACK`:

```python
# Game lines for a finished game are read as of 15:00 UTC on its game date: after the
# morning confer poll (13:30 UTC, 14:30 in winter) and before any regular US start.
GAME_LINE_TRAINING_CUTOFF = timedelta(hours=15)
```

`get_team_market_map` swaps its unused `at` for a required `cutoff`:

```python
def get_team_market_map(
    self,
    league: str,
    market: str,
    *,
    cutoff: timedelta,
    dates: Iterable[str | datetime.date] | None = None,
) -> dict[tuple[str, str], float]:
    ...
    params: list = [league, market, cutoff]
    sql = (
        ...
        "  FROM odds "
        "  WHERE league=? AND market=? AND observed_at <= game_date + ?"
    )
```

The `if at is not None` block goes. A single instant cannot express "each row's own game date", which is why `at` is replaced and not reused. DuckDB binds the Python `timedelta` as an interval, and a date plus an interval is a timestamp; the predicate ran against the real archive. [M m4_final, table F6]

`set_team_books` gains an optional stamp and passes it on:

```python
def set_team_books(self, league, market, date, team, book_evs, observed_at=None) -> None:
    ...
            self._stage_book_ev(league, market, d, team, book, ev, observed_at)
```

**`stats/base.py:672-673`**: the two calls add `cutoff=GAME_LINE_TRAINING_CUTOFF`, and the import at line 45 adds the name. These are the only callers. The ten league call sites of `_enrich_team_markets` do not change.

**`moneylines.py`**: a historical fetch stamps its rows with the snapshot's own time.

```python
def get_moneylines(...):
    historical = date.date() != datetime.today().date()
    # A historical snapshot is stamped with its own as-of time, not the day the backfill
    # ran, so the training cutoff reads it as the pre-game quote it is.
    observed_at = date.astimezone(pytz.utc).replace(tzinfo=None) if historical else None
    ...
            _store_game_moneylines(archive, game, league, date, dayDelta, observed_at)
```

`_store_game_moneylines` takes `observed_at=None` and hands it to its four `set_team_books` calls. A live run passes `None` and is stamped "now", as today.

Why the writer is in scope: without it, the next game-line backfill writes rows the new reader cannot see, and the credits spent on them buy nothing for training.

Why the true snapshot time and not the props' fixed 01:00: two backfill runs of one date then order by what they saw, not by which ran last, and no two runs tie unless they asked for the same hour. A snapshot taken after 15:00 UTC is correctly ignored by training. `backfill_historical_odds` already caps these snapshots at 12:00 UTC. The older hand-edited `scripts/moneylines_hist.py` asks for noon Chicago, which is 17:00 or 18:00 UTC; rows from it would not be read. [C `scripts/moneylines_hist.py:27-44`]

Nothing else is added: no option to switch the cutoff off, no per-league hour, no new class.

### 7.2 The alternative that needs no change to stored rows, and why not

Leave the stamps alone and let the reader accept late rows:

```sql
AND (observed_at <= game_date + ? OR observed_at > game_date + INTERVAL 28 HOUR)
```

For a key with no live poll it reads what today's reader reads, by construction: every row of such a key passes. I did not run this variant. It was not chosen because the stamp stays untrue, every later reader that cuts by time would need the same exception (the single-key readers of section 1.4 already take an `at` argument [C `helpers/archive.py:728-762`]), and a live poll added later than 04:00 UTC would be let through as if it were a backfill. It is the fallback if the owner does not want the archive touched.

### 7.3 Data steps on dev, in order

1. Copy `archive/archive.duckdb`.
2. Run the UPDATE of section 4, once, with no other process on the archive. Check that no Moneyline or Totals row is left stamped more than 28 hours after its game date.
3. Re-enrich the gamelogs chosen in section 5.5.
4. Rebuild or patch the matrix cache.
5. Retrain.

Step 2 must come before step 3. A re-enrichment against unmoved stamps reads the backfilled seasons as "no quote" (Table G, "plain" columns).

No script is proposed for steps 2 and 3: each is run once, on one box.

### 7.4 Rows already enriched

The code change does not touch them. A row changes only when it is ingested after the change lands, when one of the whole-gamelog triggers of section 1.2 fires, or in step 3 above.

One thing to know: if the whole-gamelog call ever fires unattended for NBA (a stale activity snapshot is enough), it will import the inflated totals of section 5.4. That is true today, before this change, and stays true after it.

### 7.5 Tests

| Test | File | Asserts | Before | After |
|---|---|---|---|---|
| `test_team_market_map_stops_at_the_cutoff_on_each_game_date` | `tests/test_archive_history.py` | One team with a quote at 13:30 and another at 23:40 on its game date reads the 13:30 value; a team with only the 23:40 quote is absent; a placeholder row at 00:00 is returned; the same team on the next date reads that date's own quote | Fails: no `cutoff` argument, and with the old call the 23:40 values come back | Passes |
| `test_enrich_team_markets_writes_the_pre_game_line` | `tests/test_archive_history.py` | Same rows; after `_enrich_team_markets` on a three-row frame the `moneyline` column is the 13:30 value, the 0.5 default, and the placeholder value | Fails: the 23:40 values are written | Passes |
| `test_set_team_books_keeps_a_given_stamp` | `tests/test_archive_history.py` | A write with `observed_at=2025-06-01 06:00` reads back with that stamp and is returned under the cutoff | Fails: unexpected argument | Passes |
| `test_get_moneylines_stamps_a_historical_snapshot_with_its_as_of_time` | `tests/golden/test_moneylines_get_moneylines.py` | A fetch dated 2025-06-01 06:00 UTC passes that time on all four writes; the existing live test asserts the four writes pass no stamp | Fails: no stamp is passed | Passes |

`tests/test_archive_history.py` already has a temporary-archive fixture and a helper that inserts a row with a chosen stamp. [C `tests/test_archive_history.py:29-52`] It runs in CI's `poetry run pytest`, not in the local `tests/golden/` gate. [C `pyproject.toml:249-259`, `.github/workflows/ci.yml:44`] If the leak test should block locally too, the second test fits `tests/golden/test_archive_shapefree_storage.py`, which has the same fixture. [C lines 61-64]

Existing tests:

| File | Effect |
|---|---|
| `tests/golden/test_moneylines_get_moneylines.py:107` | Its stand-in `set_team_books` must accept the stamp |
| `tests/test_mlb_parse_game.py:44` | None: it replaces `_enrich_team_markets` |
| `tests/golden/test_train_live_feature_parity.py` | None: it uses a stand-in archive. Its header needs rewording (below) |
| `tests/test_backfill_historical_odds.py:130`, `tests/golden/test_parser_ev_dist.py:36` | None: their stand-ins accept any arguments |
| `tests/golden/test_moneylines_get_props.py:97`, `tests/golden/test_moneylines_archive_event_props.py:120` | None: the props path is untouched |

### 7.6 Text that states the old behaviour

| Where | Says now | Change |
|---|---|---|
| `helpers/archive.py`, `get_team_market_map` docstring | "`at`: Observation cutoff; `None` means latest available per book" | Describe `cutoff`; drop `at` |
| `helpers/archive.py`, `set_team_books` docstring | Nothing on stamps | One sentence: defaults to now; a historical fetch passes its snapshot time |
| `stats/base.py`, `_enrich_team_markets` docstring | Nothing on timing | One sentence: read as of the cutoff on each game date |
| `moneylines.py`, `get_moneylines` docstring | Nothing on stamps | One clause on the historical stamp |
| `tests/golden/test_train_live_feature_parity.py:16-18` | The gamelog's values "were themselves baked from that same archive at load time" | Baked as of 15:00 UTC on the game date |
| `docs/handoffs/honest-receipts.md:236-240` | I6f listed as to do; cites `get_team_market_map(at=...)`; NFL cause unknown | Revise in place when it lands |
| `docs/archive/researcher_train_serve_skew.md:569-592, 785, 1052` | "77.9% observed at or after 23:00 UTC"; NFL "root cause unknown" | An archived record. The two facts that changed are in Table F and section 6; whether archived notes get corrections is the owner's call |

`CLAUDE.md` and `docs/ARCHITECTURE.md:68` list the archive's methods without saying when quotes are read, so neither states the old behaviour.

### 7.7 Production

`Stats.update()` runs inside `prophecize` on prod every hour from 08:50 to 20:50 local.

| Question | Answer |
|---|---|
| What differs after deploy | Each newly ingested game's `moneyline` and `totals` are the 15:00 UTC values, not the newest. Nothing already in prod's gamelog changes |
| Served features for today's games (`Moneyline`, `Total`, `OppTotal`, `Spread`, `GameTotal`) | Unchanged: they come from the other branch, the archive's newest quote [C `stats/base.py:1595-1624`] |
| Served slope features and the MLB starter-win curve | They read prod's gamelog over 300 days, so they drift as cut rows replace uncut ones. The size was not measured |
| Cost of the bulk query | Lower. A scan of every MLB Moneyline date took 255 ms without the condition and 144 ms with it, best of five. The daily call passes a handful of dates [M m4_final, table F6] |
| Does prod need the stamp move | Not if prod holds no backfill rows. Nothing sends dev's archive to prod, and backfills are run on dev, so I expect none. One count on prod settles it |
| Models | Unchanged until a retrain is synced. When it is, prod's gamelog should be re-enriched once as well, so that the slopes it serves are built from the same kind of value the models were trained on |

## 8. Unknowns and risks, ranked

| # | Risk or unknown | Size | What would settle it |
|---|---|---|---|
| 1 | **Scope of the re-enrichment.** The fill changes far more rows than the cut (MLB matrix rows with a real line: 4.6% to 99.2%). The effect on model metrics is unknown and the two effects cannot be told apart in one retrain | High | Decide the scope per league before step 3; retrain |
| 2 | **NBA duplicates.** A whole NBA re-enrichment imports +33 points on 382 team-games. It can also fire unattended | High | Leave NBA out of step 3; the archive rows themselves are a separate fix |
| 3 | **Order of the data steps.** Re-enriching before the stamp move reads backfilled seasons as empty (NHL 2024 falls from 99.2% to 76.5% quoted; MLB 2025 and NHL 2025 stay at the default they already have) | High if done out of order | Follow section 7.3 |
| 4 | **The matrix cache.** Neither the code change nor a gamelog re-enrichment reaches cached rows. Rebuild time unmeasured; a patch leaves four slope features stale | Medium | Time one league's rebuild |
| 5 | **Prod is unseen.** What prod's archive and gamelog hold, and whether the polls move to 14:30 UTC in winter, are inferred | Medium | One read-only count on prod; look at the first November stamps |
| 6 | **Other training reads still take the newest quote** (section 1.4): the MLB plate-appearance multiplier, the MLB pitcher and NHL goalie book legs, and the game-line book weights | Medium | Each single-key reader already accepts a time; passing it for past dates is a follow-up |
| 7 | **Placeholder rows that look in-game** (NFL 2024-25, perhaps NBA 2024; Table M). No cutoff can fix them | Medium | A fresh historical snapshot for those seasons |
| 8 | **Morning kickoffs abroad.** At 15:00 the morning poll lands at kickoff for NFL games starting 13:30 UTC | Low: about 14 team-games a season | Use 12:00 if this matters |
| 9 | **Doubleheaders.** One date and team is one key, so both games share one value, as today | Low | Not addressed |
| 10 | **Same-stamp ties.** Re-running a backfill for one date and hour writes a second row under the same stamp; the reader's pick is arbitrary | Low | Not addressed |
| 11 | **Thin leak tests.** NBA rests on 48 team-games and NFL on 90. WNBA's .47 at the cutoff cannot be separated from a normal pre-game level without start times | Low | More live-polled games; a stored start time |
| 12 | **Old backfill hours.** The as-of hour of the July and August runs was not logged. Their values pass the leak test (Table J) | Low | None needed |

What I could not establish, stated once:

- Game start times: none are stored, so no per-game cutoff could be tested.
- Prod's archive, prod's gamelog, and the server's winter clock.
- What serving actually read on past days: the serve-time feature log began on 2026-10-05.
- The effect of the change on any model metric or served probability.
- Rebuild time for the matrix cache.
- Why 252 MLB team-games between 2026-06-21 and 2026-08-15, and the other gaps listed in `m3b_T5_gaps.csv`, have no archive row at all.
- Why a quarter to a third of the live-polled team-games on dev were stored at the default.

## Scripts

All in `~/backups/sportstradamus/2026-10-04-honest-receipts/main/wave4/i6f/`. Each opens the archive read-only and writes only beside itself.

| Script | Output | Measures |
|---|---|---|
| `m0_schema.py` | printed | Archive schema and row counts |
| `m1_gamelog_inventory.py` | printed | Gamelog and matrix inventory, time-like columns, stored defaults |
| `m2_archive_profile.py` | `m2_rows.csv`, `m2_keys.csv`, `m2_late.csv`, `m2_backfill_days.csv`, `m2_hour_hist.csv` | Tables B, C, D |
| `m3_rules.py`, `m3b_tables.py` | `m3_values.parquet`, `m3b_*.csv` | First pass at the rules (36-hour line, flat restamp); kept for Tables J and M and the gap list |
| `m4_final.py` | `m4_output.txt`, `m4_values.parquet`, `m4_F*.csv` | Tables G, H, I, N, P, Q, the distance table, the query timing |
| `m4b_extra.py` | `m4b_output.txt`, `m4b_G*.csv` | Table K, the real-line rows of Table Q |
| `m4c_by_season.py` | `m4c_output.txt`, `m4c_gamelog_change_by_season.csv` | Table O, file timings |
| `m5_nfl_miss.py` | `m5_nfl_miss.csv` | Section 6 |
| `m6_brief_figure.py` | `m6_output.txt` | Table F |
| `m7_restamp_dryrun.py` | `m7_output.txt` | The UPDATE of section 4 on an in-memory copy |
| `m8_first_quote.py` | `m8_output.txt`, `m8_first_quote.csv` | The "first observation" candidate |
| `m9_nba_duplicates.py` | `m9_output.txt` | Section 5.4 |
| `m10_stamp_classes.py` | `m10_output.txt` | Tables A, E, L |
| `m11_live_era_stored.py` | `m11_output.txt`, `m11_live_era_stored.csv` | What the gamelog stores for live-polled team-games |
| `m12_newest_before_gameday.py` | `m12_output.txt` | Keys whose newest quote predates the game date |
