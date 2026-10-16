# Lane brief — honest receipts: one pricing truth, three research verdicts, model-side repairs

**Read first:** CLAUDE.md, [docs/ARCHITECTURE.md](../ARCHITECTURE.md), [docs/ship_gate.md](../ship_gate.md)
(Gate 2 and the devel → main row), [story-balance.md](story-balance.md) §1 and §6 (the ledger fixes and
the killed selection-shrink lever this lane continues), and the three research briefs in
`docs/archive/` (§5). Status: OPEN — waves 1–15 are on `devel` (§4); all three briefs returned (R1
KILL, R2 DONE, R3 DONE; §5); the tail scorecard (E) is built; the consensus sanity check found no
profit in the sportsbook consensus at the platforms' real payouts (§7); the owner's decisions 7–29
and 32–76 (§2) are built or run (§4, §6), and the stored game lines of four leagues are re-read on
the training box (§6, I6f). On production: the rebuilt MLB models and the NFL ship-15 models
(decisions 48, 49), the NBA's same-player correlation file (65), the 13 MLB games the gamelog lacked
(51), MLB's two retrained cells (67), the WNBA gamelog repair (68), and the NHL's, the NFL's and
the WNBA's rebuilt and retrained models with their game-line copies (69, 70, 72). The handoff of
2026-10-08 is done. What stays open is in §6: the I6g version-age test (late October) and the two
reads I6 marks still open. §8 holds the owner's own two items.

## 1. Mission & money logic

The owner asked why Receipts reported a 60–70 % hit rate on the model's recommendations while realized
ROI was negative. Both numbers were true of different populations, and the report hid it. The old hero
("If you'd tailed every rec", n = 341k, hit .725, "+38 % at flat −110") mixed three groups: sides the
platform never posted (`Boost == 0`, hit .839, never a bet, half the rows), posted legs the model did
not recommend (hit .62, calibrated, but the platforms pay 1.3–1.5× on favorites so breakeven sits at
.67–.70), and the recommended legs (3 % of rows). Priced at the platforms' real payouts, posted sides
only, one offer per platform, all-time on `data/runtime/history.parquet`:

| cohort | n | hit | model read | book | avg payout | breakeven | ROI |
|---|---|---|---|---|---|---|---|
| posted, model side chosen | 277,957 | .611 | .620 | .609 | 1.48 | .676 | −11.7 % |
| posted, not recommended | 261,004 | .619 | .618 | .612 | 1.46 | .687 | −11.8 % |
| recommended (edge ≥ 5 %, payout ≤ 2.5) | 16,731 | .496 | .650 | .560 | 1.83 | .547 | −10.5 % |
| recommended, payout > 2.5 | 222 | .248 | .541 | .408 | 2.85 | .350 | −30.5 % |

So the answer to "is it the boosts or the reporting?" is both. The payout is where the money goes; the
flat −110 grade was fiction for discounted favorites; and the model's read on the legs it recommends is
15 pp too high. The models are calibrated in bulk (every served-probability bucket on posted legs
realizes within 1–2 pp of its read) but the recommender ranks on `edge = p × payout − 1`, which selects
exactly the legs where the model's error is most positive: the optimizer's curse. On the selected set
the claimed edge carries no information (λ_sel = −0.10 ± 0.10,
[researcher_selection_shrink.md](../archive/researcher_selection_shrink.md)). Three things sat beside
it when the lane opened: parlays compound the per-leg overconfidence; no offline gate scores the
population the product recommends; and the reporting graded everything at −110.

The owner's constraints are binding and shape every stage here: **no model is pulled, demoted or
withheld on realized money; no model-free engine; research first (Opus subagents); one honest report.**
Reporting was fixed first (§4) so every later step is judged by one honest number. Three briefs then
measured the candidate fixes (§5). Shrinking to the sportsbook consensus was already killed
(story-balance §6). A learned, selection-aware probability layer was killed by R1: once the layer sees
the platform price, the served probability adds nothing out of sample, so the layer would be the
model-free engine the owner ruled out. R2 found the parlay pricer sound and the legs at fault. R3 found
that train/serve skews in the calibration chain and the quotes are worth at most 0.7 pp of the 12.5 pp
post-fix gap; 89 % of it is selection on model–market disagreement where the served mean knows little
beyond the market. So the "fix the models" track is giving the models information the market does not
have (labelled I6, §6), judged on the recommended legs by the tail scorecard (labelled E, §6).

## 2. Locked decisions (owner)

Decisions 1 to 14 are of 2026-10-04. Decisions 15 to 25 are the owner's answers of 2026-10-08 to
the asks that then stood in §8, each named here by what it decides. Decision 26 is of 2026-10-09.
Decisions 27 to 35 are the owner's go of 2026-10-08 on the story page and on the asks that
building 15 to 25 raised ("Take all of your recommendations except waiting on the MLB offseason.
Let's fix it now"), in the order they stood in §8. Decisions 36 to 42 are the owner's answers
of 2026-10-11 to the eight asks that building those raised. Decisions 43 to 68 are the owner's
yes of 2026-10-11 and 2026-10-15 to the asks then in §8, each moved here when it was built or
run. Decisions 69 to 76 are the owner's yes of 2026-10-16 to asks 32 to 35 of §8 and 36 to 39
of the close-out report.

1. **Hero cohort = recommended**: edge ≥ 5 % at the platform payout, payout in (1, 2.5], with all
   posted sides as the context row. The ledger's `recommended` definition carries the same cap
   (`realized.RECOMMENDED_PAYOUT_MAX`, set to `helpers.MAX_FAVORED_PAYOUT`).
2. **All three briefs run in parallel** (R1 trust layer, R2 parlay engine, R3 train/serve skew),
   `research-analyst` subagents, model opus, read-only.
3. **`Trust Prob` drives recommendations, Kelly and parlay legs (I4d) only at Tier B**: walk-forward,
   day-clustered 95 % CI lower bound of ROI at platform payouts above 0 after Holm. Tier A unlocks
   display and the D6 shadow policy only. (Moot: R1 KILL, §5.)
4. **Menu stance under a thin honest menu is decided after R1 reports its volume/ROI frontier** per
   league with volume per day at every grid threshold; I4d is not specified until the owner picks the
   operating point. (Moot: R1 KILL, §5.)
5. **The DFS main line is an evaluation reference only.** A model may be scored against it; it is
   never a book leg in the blend and never part of the consensus line. Where the code stands against
   this rule: §7.
6. **The live Underdog Power table is confirmed**: 3.5 / 6.5 / 12 / 20 / 35 / 65 / 120 for 2–8 picks
   ([underdog_api.md](../underdog_api.md) P-32). The owner answered the ask to confirm that capture in
   the app with "Underdog payouts are confirmed correct"; it is read here as confirming the capture,
   which makes `underdog_payouts.json` (3 / 6 / 10 / 20 / 25) the stale one.
7. **The session pushes `devel` for this lane's approved work** ("Go ahead and push yourself"), each
   push after the refactoring-specialist and the three gates. A push deploys: production pulls `devel`
   before every cron job.
8. **A 1.00× Underdog pick is valued for the entries the owner plays**: 3-, 5- or 6-leg on Underdog
   and 2-leg on Sleeper, "but not exclusively those". `UNDERDOG_BOOST_BASELINE` becomes 1.83, the
   geometric mean of the 3-, 5- and 6-pick Power roots (1.866, 1.821, 1.809). A Sleeper leg keeps its
   posted multiplier.
9. **The three parlay fact fixes go**: the payout file, the pair bonuses above 1.0, the ledger
   settlement (§6, I5).
10. **A leg with no model keeps its recommended flag**: "Treat the consensus book as the model if the
    model doesn't exist." No code change.
11. **The consensus line is built from sportsbooks only** (decision 5 applied to `Archive.get_line`),
    after every reader is traced.
12. **A paper bettor takes Underdog 1.00× legs only**, in the sim ledger, beside the current ones.
13. **The model-information work opens** in the §6 I6 order: the serve-time log and the 14-day
    model-file retention now, the training-side cuts at the next retrain, NFL research after the log.
14. **The temperature penalty is dropped only if** the models are better without it and every cell
    that passes the gates today still passes.
15. **A closing price is stamped only once the game is under way.** A history row closed before
    its game is released, and the close is read as of the game's start (§6, repairs).
16. **A prop whose line moved counts once** in Receipts and the realized ledger, at the lines its
    last scoring held, for games from 2026-10-09; earlier games keep every posted line.
17. **The paper ledger's forty copies are made independent as a new version**, `policy_v4`.
18. **The bankroll table's `policy_version` column is signed off**: a new policy version starts
    its trajectories from the $5,000 seed.
19. **A push counts as half an Over in training**, at every label site, the ship gates' own label
    included. A cell changes at its next retrain.
20. **The game lines of games already stored on the training box are re-read** for MLB, NHL, WNBA
    and NFL, every row; none for NBA.
21. **The NFL game-line lookup is fixed**: a new row is read under its own game day.
22. **The dashboard restarts once per deploy**, not at every job after it.
23. **NBA fantasy-points legs get their own trust figure in the dashboard's parlay search**: the
    lookup uses the league's own market key.
24. **Training's fantasy-points average stays** ("leave it"): moving the summed-components mean
    halfway to the platform's own fantasy quote in training rows is not read as blending with the
    DFS line (decision 5).
25. **The app's payout tables as the owner read them on 2026-10-08 are the record.** Power 2 to 6
    picks pays 3.5 / 6.5 / 12 / 20 / 35; Flex pays 3.25 and 1.09 at three picks, 7.2 and 1.4 at
    four, 10 and 2.5 at five, 25, 2.6 and 0.25 at six. Four NFL same-game Power payouts read the
    same day set four pair values (§6, I5).
26. **A pre-season game is never priced, in any league**, and no cron line is held for it. An
    offer dated before the odds feed's opening night for its league is archived and not scored;
    the rule is off while the feed holds no opening night (silent over three days). A new NBA
    season reads the newest earlier roster seed file until its own is dropped in; what a new
    season's roster does not hold yet is decision 37.
27. **The Tonight card shows no side arrow, and a date's Over leads fill toward half of its
    games.** The rule picks which story leads a game. The legs inside a story are the models',
    so no balanced leg mix is promised. Where the platforms lead a game on different sides the
    card keeps the higher-EV lead ([story-balance.md](story-balance.md) §2, §5).
28. **49ers and 76ers games get their game lines again**: a team name is matched under one
    spelling on both sides, in every league. Past games stay at the default line (decision 36).
29. **The MLB profile defect is fixed now, not in the off-season**: the fix, then a cold rebuild
    of the MLB matrices and a retrain. Nothing is synced to production before the owner has
    seen the gate table.
30. **The NHL matrices are rebuilt before the season's first NHL retrain**, for the four slope
    features (§7).
31. **Production's gamelog gets the same game-line re-read** when the first retrained models
    are synced.
32. **NBA and WNBA Underdog fantasy-points rows get a closing price**: a close is read under
    the league's own market name, and history keeps its own.
33. **The Underdog scrape fails loud the day a full feed holds no even pick**
    ([underdog_api.md](../underdog_api.md) §7.4).
34. **A 4-pick Underdog entry stays Flex** (§7, the entry-kind record).
35. **Sleeper NHL assists, points and blocks legs go by the league's own market key**, in the
    parlay search, the stories and a slip seeded from a story.
36. **The past game lines of 49ers and 76ers games are not bought.** Those games stay at the
    default line in the archive, the matrices and the models (decision 28, §7).
37. **A player the new season's roster does not hold yet is read from last season's**, in the
    NBA, the WNBA and the NHL, on the old team if the player has moved since. The NHL lookup
    no longer fails while the new season is not stored, and a roster pull the site blocks
    leaves the NHL's stored roster as it was.
38. **One league's failed gamelog update does not cost the others their board.** That league
    sits the run out, with its lines still archived; the board run still ends failed, so the
    alarm fires.
39. **The deep dive's pairing cards find their board rows**: a stored partner names its market
    by the board's own name.
40. **Portland's WNBA players get their roster, and their game rows are stored again.**
41. **A league's opening night is pinned for the season.** The odds feed's opening night is
    captured only when a league returns from an off-season, and a league keeps it through a
    break. A season start already held is not moved by a feed date inside that season, which
    puts MLB back on 2026-03-25 and the WNBA on 2026-05-15.
42. **A cleanup pass on the style-review leftovers follows the MLB and NHL rebuilds.** It
    followed MLB's and did not wait for the NHL's, which was stopped then (§7).
43. **MLB's two comp tables travel to production with its models.** Production's were from
    March (§7).
44. **The depth rank reads the usage column alone**, so a roster's text attribute cannot stop
    an NHL matrix build (§7).
45. **An MLB game keeps its schedule entry until it is over.** Two costs are accepted with
    it: a doubleheader's second game reads game one's entry until game one is over, and the
    next day's starter's Underdog fantasy-points offer is filed as a hitter's while today's
    game is played (§7).
46. **A Sleeper home team reads its real opponent**: the opponent is picked against the
    remapped team code (§7).
47. **The session runs the production jobs the owner named** (the MLB runbook, a league's
    send, the gamelog refills of §8): one job in a gap between two board runs, every
    replaced file kept on production, undone on a failed check. Everything else on
    production stays read-only, and a session never imports the package there, opens its
    archive or tests its lock files.
48. **The rebuilt MLB models are on production** (2026-10-12 00:01 UTC), with the comp tables
    (decision 43). The training box's game lines of past games were copied into production's
    gamelog first (decision 31): 2025's team-games at the default line went from 4,855 to 28.
49. **The NFL ship-15 models are on production** (2026-10-12 00:57 UTC): 19 model files, the
    NFL's entries of the three shared files and its correlation files, by a one-league send.
50. **A later date's offer reads its own game's schedule.** MLB, the NBA and the WNBA keep a
    second map keyed by team and game date, read through one lookup that falls back to the
    team's next game, and a player offered on two dates is built for the earlier one. The NHL
    and the NFL keep the next game, and an offer beyond the three-day window reads it too.
51. **An MLB game called early is finished**: `Completed Early` joins `Final`. The 13 games the
    gamelog lacked were refilled on both boxes on 2026-10-15, each at its date: 11 called early
    (4 in 2024, 5 in 2025, 2 in 2026) and two `Final` games of 2025-03-27.
52. **`scripts/sync_to_prod.sh --league` sends one league**: its entries of the shared files
    alone, nothing deleted, every replaced file kept on production, the league's
    `corr_market_summary.parquet` (and MLB's comp tables) with it.
53. **An empty NHL roster pull fails the update once the season has games**: the board run
    exits 1, and the stored roster and the parsed games are kept.
54. **A league whose constructor or first load raises costs only its own board**; `StatsMLB`
    no longer calls the schedule when it is built.
55. **The board's `Stat` is the league's own key** (`assists` for an NHL `AST` row), and labels
    fold through it, so the History tab, the form sparks, comps, slips and the Lab's lookups
    find their cell.
56. **`Fun` reads a market's spread under the league's own name.**
57. **The parlay search builds one game frame per game date**, earliest first. The dashboard
    still keys a game without its date (§7).
58. **The board run no longer builds the pick'em snapshot**; `bet pickem` and the ledger build
    are unchanged.
59. **The NHL keeps four years of gamelog**, as the NBA and MLB do.
60. **A WNBA ratio reads 0 on a zero denominator**, as the NBA's do; the stored rows are cleaned
    by ask 14's repair.
61. **A second cleanup pass**: code nothing calls to `src/deprecated/`, the imports no module
    reads gone with the lint rule for them on, docstrings trued. Nothing computed differently.
62. **Every NHL update writes the goalie list** (`config/goalies.json`), the hourly board run's
    included.
63. **The training box's archive files Portland's WNBA game lines under `POR`**: 1,722 rows of
    2026-07-11 to 08-06 renamed from `PDX`. Production's archive is untouched.
64. **MLB's schedule map files Washington under `WAS` and Arizona under `ARI`**, through the
    map both scrapers use.
65. **The NBA's same-player correlation file is on production**, sent alone on 2026-10-15;
    production's copy of July held no same-player rows.
66. **The Lab › Diagnostics CRPS panel is gone**: history has held no CRPS since June.
67. **MLB's two retrained cells are on production** (ask 3): `batter strikeouts` and `pitcher
    fantasy points underdog`, retrained on their recomputed matrices, went with MLB's correlation
    files, `corr_market_summary.parquet` included, by `scripts/sync_to_prod.sh --league MLB` on
    2026-10-16 at 00:07 UTC. The files' checksums match on both boxes. The 00:50 board run held no
    offer in either market (the 10-16 game's lines were not yet posted), so their first scoring on
    production is still to be read.
68. **Production's WNBA gamelog is repaired** (asks 14 and 31), as the training box's was on
    2026-10-15: 281 rows added at their dates (Portland's 275 of 27 games, 2026-06-24 to 09-22, and
    the 6 of the 9 bench rows production lacked), `PDX` renamed `POR` (147 player rows and 132
    opponent cells; team log 13 and 12) and 5,256 infinite cells set to 0 (gamelog `BLK_PCT` 4,923
    and `FTR` 173, team log 160), on 2026-10-16 at 01:05 UTC. Every after-write check passed and a
    second run found nothing to change; both old files are kept in
    `~/repair-backup/P5-20261016T0105Z/`.
69. **The NHL's rebuilt and retrained models are on production** (ask 32): twelve model files,
    hits included, with the NHL's entries in the shared files and its correlation files, by
    `scripts/sync_to_prod.sh --league NHL` on 2026-10-16 at 03:35 UTC. At 03:34 production's NHL
    gamelog first took the training box's `moneyline` and `totals` for team-games dated before
    2026-10-08 (decision 31): 10,768 of 10,928 team-games changed, and those at the default line
    fell from 3,446 to 24. The write's five checks passed, a second run changed nothing, and the
    sent files' checksums match on both boxes. Backups: `~/repair-backup/P3-20261016T0334Z/`,
    `~/league-send-backup/NHL-20261016T033507Z/`.
70. **The NFL's rebuilt and retrained models are on production** (ask 33): 19 model files by
    `sync_to_prod.sh --league NFL` at 03:36 UTC, after the game-line copy at 03:35 (2,322 of 3,390
    team-games changed; at the default line 1,148 before, 1,036 after). Production keeps its
    withheld `NFL_passing-yards.mdl`, which serving skips. Backups:
    `~/repair-backup/P7-20261016T0335Z/`, `~/league-send-backup/NFL-20261016T033558Z/`.
71. **`Archive.to_pandas` keeps the Totals rows dated before 2023-05-03** (ask 34), as it keeps
    the Moneyline rows: its exemption named `"Total"`, a market no row carries. The Totals
    book-weight fit gains 7 to 26 % of each league's rows at that league's next `meditate` (§7).
72. **The WNBA's rebuilt and retrained models are on production** (ask 35): 14 model files at
    03:34 UTC, after the game-line copy at 03:32 (2,095 of 3,106 team-games changed; at the default
    line 1,037 before, 977 after), which ran after decision 68's repair so Portland's rows match on
    `POR`. Backups: `~/repair-backup/P8-20261016T0332Z/`,
    `~/league-send-backup/WNBA-20261016T033341Z/`.
73. **The constellation-art lane is closed** (ask 36 of the close-out report): 99 of 101 templates
    have art, and `the-chain-gang` and `the-crossover` stay on the owner's hand-hunt list. Its
    brief is in `docs/archive/`.
74. **The model-strategy reachability lane is closed, its code kept** (ask 37): no acceptance cell
    was ever scored on a structural recipe, and NFL receiving and rushing yards pass all six gates
    on the canonical recipe after the 2026-10-15 retrain. Its brief is in `docs/archive/`.
75. **Sleeper-parity's stage 5 acceptance is `sportstradamus bet pickem --platform sleeper` output
    on live slates** (ask 38); the pick'em snapshot it named went with decision 58.
76. **A cleanup push** (ask 39): the pre-commit integration hook runs with `-n0`; `Scrape()` reads
    the ScrapeOps key alone; the Plotly template's font is IBM Plex Mono, one of DESIGN.md's fixed
    faces; `training.calibration.select_distribution` (to `src/deprecated/`), `helpers.books` with
    `prop_books.json`, `ship_config.json` and the `fetch fp discover` help snapshot go, none of them
    read; 13 comments that cited research briefs not in the repo are repointed or keep only their
    substance; three stale comments are trued; Mixture is marked research-only in ARCHITECTURE and
    MODEL_LIFECYCLE; story-balance §7 and §8 and model_improvement_track's two dated blocks follow
    STYLE_GUIDE §16.

Standing rules from CLAUDE.md apply: one module per subagent, refactoring-specialist before any push,
the three gates once per wave, `stat_meta.json` never committed from a session.

## 3. Verify before you trust

```bash
git log --oneline 3685161b^..devel                # the wave commits (§4) among the others since
poetry run pytest tests/golden/test_realized_offers.py tests/golden/test_realized_by_side.py \
    tests/golden/test_receipts_reconciles_ledger.py tests/golden/test_receipts_page_render.py -n0 -q
poetry run python - <<'PY'
from sportstradamus.helpers.io import read_history
from sportstradamus.realized import settled_offers, cohort_summary
o = settled_offers(read_history())
print("posted", cohort_summary(o))
print("recommended", cohort_summary(o[o["Recommended"]]))
PY
ls docs/archive/researcher_trust_layer.md docs/archive/researcher_parlay_engine.md docs/archive/researcher_train_serve_skew.md
ls ~/backups/sportstradamus/2026-10-04-honest-receipts/   # dev box only: the briefs' harness scripts (§5)
sportstradamus admin tail-scorecard                      # E (§6): ~90 s, read-only, writes /tmp/tail_scorecard.csv
```

The Receipts hero must reconcile to the ledger to the leg: `cohort_summary` of the recommended cohort
of the windowed posted frame equals the sum over sides of `compute_realized_by_side` rows for that
window and cohort (`tests/golden/test_receipts_reconciles_ledger.py`). The string "-110" appears
nowhere on the page.

### Volatile product assumptions

- `helpers.distributions.UNDERDOG_BOOST_BASELINE = 1.83` is what one unboosted Underdog pick is
  worth: the geometric mean of the 3-, 5- and 6-pick Power roots (decision 8; the live table is
  [underdog_api.md](../underdog_api.md) P-32). Model EV, Kelly, the 5 % rule, the payout-implied
  `Market Prob` and leg admission all read it, and `Archive.add_dfs` prices a one-sided rung through
  it, so a leg scored or a rung archived before 2026-10-05 12:00 UTC sits at 1.78: a new selection
  era, never pooled with the old one (the tail scorecard decodes a rung by its last poll time).
  Replayed on post-fix legs, 1.83 takes the Underdog recommended cohort from 3,055 legs at −13.1 % to
  4,566 at −8.2 % and both platforms from −11.8 % to −9.1 %; the 2-pick root 1.871 would give 6,123
  at −6.3 %. No sign flips.
- An even Underdog pick is a `Boost` of exactly 1.0 in the feed, and every even-pick rule in the
  code tests for that value (the per-pick valuation above, the payout tables' base case, the
  `even_picks` paper bettor). The app stopped showing that number in October 2026: it prints the
  pick's decimal price, so an even pick reads 1.87× (√3.5). The feed still sends 1.0. The day it
  follows the app every Underdog payout would price about 1.87 times too high, so the scrape
  raises when a full feed holds no even pick (decision 33; the check, its blind spot and the
  last count are in [underdog_api.md](../underdog_api.md) §7.4).
- `underdog_payouts.json` is the owner's app read of 2026-10-08 (decision 25). One value moved
  with it: the 4-pick Flex pays 1.4× with one miss, not the 1.8× captured in September. Anything
  that priced or settled a 4-pick Flex before that date used 1.8 (§7, the entry-kind record).
- A same-game pair's Power modifier in `banned_combos.json` is one value per pair type, while
  Underdog prices each player pair on its own. Four values the owner checked in the app were all
  off (§6, I5), so a sub-1 or banned value is a guess until a quote backs it.
- Clean data is thin: five weeks post-fix (Date ≥ 2026-08-31); per-side payouts since 2026-10-03; MLB
  (the largest share of recommended legs) ends late October; NBA starts late October with no clean data
  until late November. No brief measured an NBA cell, so every NBA read starts from zero.

## 4. What landed (all on `devel`)

| Wave | Commits | What |
|---|---|---|
| 1 | `3685161b` | `helpers.platform_payout(boost, platform)`: the one Boost → decimal-payout conversion (Underdog raw × 1.78 per pick, Sleeper as posted) |
| 1 | `3a2265d5` | `strategies/profit_sim.compute_payout` nets every platform at its real payout (Underdog was 100/110 × boost) |
| 1 | `7fc7f4ea` | `scripts/audit_parlay_calibration.py` drops its own payout table: stored parlay `Boost` already includes the payout base |
| 1 | `38090895` | serving provenance: `book_evs_for_players` returns quotes; `annotate_quote_provenance` writes Quote Source / Authenticity / Books / Line / Observed At; `Model Weight` on both paths. Served probabilities unchanged (hermetic before/after harness) |
| 2 | `e5ebbf42`, `b34f14c0` | `history_schema.DECISION_COLS` (22 decision-time columns incl. `Scored At`, `Consensus Line`, `Push Prob`, `Projection STD`, `Books STD`, `Model Weight`, the Quote columns) persisted from the next prophecize run; the dashboard loader projects them out |
| 2 | `efe5106e` | `realized.py`: `settled_offers`, `window`, `by_split`, `cohort_summary`, `compute_realized_by_side`, `worst_month`, `calibration_summary` (cohorts posted / recommended); `breakeven_rate` and the `quote` split join the ledger |
| 2 | `73e7821b` | `reflect`: calibration summary from realized pricing; the fair-odds profit sims stop scaling Sleeper's posted payout as a promo (I3a) |
| 3 | `1c9fc738`, `ccedaa3f`, `0f35e617`, `b3084815`, `2142ec91` | `dashboard.data.posted_offers_or_stop`; Receipts hero = the recommended cohort at the platform payout with the all-posted context row, by-side panel live from `by_split` with "edge captured", calibration panel by cohort, rolling-accuracy line at breakeven; Lab › Diagnostics on the same universe; the −110 helpers retired to `src/deprecated/`; `analysis._add_kelly_columns` sized at the platform payout (I3b) |
| 3 | `2fb154c8` | E: `sportstradamus admin tail-scorecard` replays the live recommendation rule on held-out test rows at the archived DFS rungs and reports the selected-tail gap, the live gap for the same cells, the Model Version age split and the information test (§6, §7); a diagnostic, never a gate |
| 4 | `1d2efb58` | Parlay payout facts (decision 9): `underdog_payouts.json` carries the live Power and Flex tables to 8 picks, a Flex loss tier pays on the largest remaining pick multipliers, the pair modifier travels beside them, the 523 Underdog pair values above 1.0 become 1.0 |
| 4 | `f6138bab` | `UNDERDOG_BOOST_BASELINE` 1.78 → 1.83 (decision 8); the archive prices a one-sided rung through the same constant, so `Market Prob` and the close `clv` reads stay on one scale |
| 4 | `308db57c` | Sim ledger `policy_v2` (decisions 9, 12): settlement at table × pick multipliers × pair modifier through `payouts.outcome_payouts`, the rule the pricers use; the `even_picks` paper bettor; cross-game stakes on the all-hit payout and the priced EV ([sim-bettor-ledger.md](sim-bettor-ledger.md) §10) |
| 4 | `fe394eb6`, `6f6dcb3d` | Serve-time feature log (`prediction/feature_log.py`) and `sportstradamus admin feature-parity` (decision 13, I6e) |
| 4 | `a7ca8e21` | The temperature fit without its penalty (decision 14, I6c) |
| 4 | `dcf084fe` | 14-day prior model files (`training/prior_models.py`, I6g) |
| 4 | `1a8cabd8` | Sportsbook-only consensus line (decision 11): `Archive.get_line`, `get_reference_line`, the quote resolver's line vote, the book-weight fit, the board stamp ([archive/consensus_line_trace.md](../archive/consensus_line_trace.md)) |
| 5 | `01b52c66` | Pre-game cutoff for the game lines training reads (decision 13, I6f): each sportsbook's newest Moneyline and Totals quote stamped by 15:00 UTC on the game date; a historical game-line fetch keeps its snapshot time ([archive/game_line_cutoff_design.md](../archive/game_line_cutoff_design.md)) |
| 6 | `6adcfae8` | Sim ledger `policy_v3` (decision 12, rebuilt): the `even_picks` paper bettor takes the Underdog even picks the product recommends for the day's games, at the served read, in evenly dealt entries with a game per leg; a cross-game leg's `stat` and trust lookup use the league's own market key ([sim-bettor-ledger.md](sim-bettor-ledger.md) §10) |
| 7 | `23890f6d` | `scripts/run_job.sh` restarts the dashboard once per deploy (decision 22): it compares against the commit it noted before its own pull |
| 7 | `a5a3b78e` | Four same-game Power pair values from payouts the owner read in the app (decision 25; §6, I5) |
| 7 | `db5ab012` | A closing price waits for the game (decision 15): `clv.commence_times` gives each row its kickoff; `fill_from_archive` fills a row once that instant has passed, releases one closed before it, and reads the close as of it |
| 7 | `0c464fae` | Sim ledger `policy_v4` (decisions 17, 18, 25): an entry is one copy's (`id`, persona, `replicate_id`), and the retry guard and the settlement match follow; the bankroll chain continues from the last row written; a 4-pick Flex with one miss pays 1.4×; a leg's close is read at its kickoff ([sim-bettor-ledger.md](sim-bettor-ledger.md) §10) |
| 7 | `5b7ca2dd` | Receipts and the realized ledger count a prop once (decision 16): `realized.settled_offers` keeps the lines a prop's last scoring held, for games from 2026-10-09 |
| 7 | `e339e047` | The dashboard's parlay search looks a leg's trust up under the league's own market key (decision 23) |
| 7 | `21e00ebd` | A new NFL row takes the game line of its own game day (decision 21): the schedule joins on team and week |
| 7 | `33ce9b28` | A push is half an Over in training (decision 19, I6a): `training/labels.over_label` at the eleven label sites, the ship gates' label included |
| 8 | `cf0f374e` | A new NBA or WNBA season starts on last season's rosters (decision 26): `Stats._roster_player_df` falls back to last season's (player by player since decision 37), `nba._roster_seed_path` reads the newest earlier seed file, and one `_merge_season_roster` adds a team the season lacks (production's WNBA 2026 roster held 6 of 15 teams) |
| 8 | `f3713b95` | A pre-season game is never priced (decision 26): `prediction/scoring._in_season_markets` drops an offer dated before the feed's opening night, after it is archived |
| 9 | `d35a8aa3` | The Tonight card shows the lead story's headline with no side arrow (decision 27); `narrative.game_lead_side` retired to `src/deprecated/` |
| 9 | `0212ca65` | Over leads fill toward half of each date's games (decision 27): `stories/lead._pick_sides` ([story-balance.md](story-balance.md) §5) |
| 9 | `0039f100` | 49ers and 76ers games get their game lines (decision 28): `moneylines._TEAM_ABBR` folds the config's team names the way the odds feed's are folded |
| 9 | `d19a597b` | A close is read under the league's own market name (decision 32): `clv._fill_one_group`, `clv._row_movement` and the ledger's `join_clv` fold through `helpers.archive_market`; history's `Market` is unchanged |
| 9 | `439bbd34` | The Underdog scrape raises when a full feed holds no even pick (decision 33): `UD_EVEN_PICK_MIN_LINES`, `UNDERDOG_EVEN_PICK_MULTIPLIER` |
| 9 | `b2f3274c`, `0c4fc099`, `09f64934` | Parlay and story legs name a market by the league's own key (decision 35): `parlay.leg_market_map` builds on `helpers.cell_market`, and a slip seeded from a story finds its board rows under that key |
| 9 | `67309a51` | A rebuilt base profile forgets the market it held (decision 29): `Stats.base_profile` clears `profiled_market`, so an MLB matrix build profiles the market on every game day |
| 10 | `148f81a1`, `2fbc8d14` | Last season's roster row stands in for a player the season does not hold (decision 37): `Stats._roster_player_df` for the NBA, the WNBA and the NHL; the NHL update merges a pull into the season's roster, so a blocked pull cannot empty it |
| 10 | `911fc9a6` | One loader for the board run and the ledger build (decision 38): `stats.load_live_leagues`; a league whose update raises sits the run out, and `prophecize` exits 1 after its snapshots |
| 10 | `d7ba4766` | A stored correlation partner names its market as the board does and carries its league (decision 39): `correlation._corr_partners` |
| 10 | `e71f748e` | The WNBA shot map's team code goes through the alias map (decision 40), so Portland's players stay in the player frame |
| 10 | `03395751`, `59e79056` | A league's opening night is pinned (decision 41): `moneylines._is_opening_night` captures it only after more than 60 days without a game and idle records keep it; `Stats.update` adopts it only when more than 300 days after the start it holds |
| 11 | `6b9e00c7`, `d55a49eb`, `e91f6d1a`, `9fce9cf1`, `25976d05`, `3f5f5382` | The cleanup pass (decision 42), nothing computed differently: `check_bet`, `beam_search_parlays`, `construct_entries` and `build_entries_from_scored` drop a parameter each never read; `odds_budget.feed_ts` goes for `datetime.fromisoformat`; "no season runs 300 days" is one constant, `stats.base.SEASON_MAX_DAYS`, where it was written seven times, and the test five league updates each made with it is one method, `Stats._full_clean_due` (`9ddfe5af`, `5d0cc5c5`); the unused parlay-history plot goes to `src/deprecated/`. One thing a reader of the box can see: the empty `logs/<date>/pickem-build.jsonl` no longer appears |
| 12 | `4dc766e5` | A Sleeper home team with a remapped code reads its real opponent (decision 46): `books/sleeper.py` picks the opponent against the game's remapped codes |
| 12 | `82d04c5b` | An MLB game keeps its schedule entry until it is over (decision 45): `helpers/text.mlb_game_is_over`, read by `stats/mlb._build_mlb_upcoming_games` and the probable-pitcher table |
| 12 | `5982fc4e` | MLB's two comp tables are in the file list of `scripts/sync_to_prod.sh` (decision 43) |
| 12 | `cf64d023`, `6a3602fb` | `Stats.get_depth` ranks the usage column alone (decision 44), so every NHL matrix build no longer raises on its first game day |
| 13 | `6e13a11b`, `a63b9107`, `3bd8ff59`, `6a6df14e` | A later date's offer reads its own game's schedule (decision 50): `Stats.upcoming_game(team, date)` over the dated map; the date maps, the feature log and the parlay search's depth records take a player's earliest offer |
| 13 | `37b6bbe4` | An MLB game called early is parsed (decision 51): `helpers.text.MLB_FINAL_BOX_SCORE_STATES` |
| 13 | `11f6949e` | `scripts/sync_to_prod.sh --league` (decision 52), with `scripts/build_league_only_serving.py` and its golden test |
| 13 | `74b40d02` | An empty NHL roster pull fails the update once the season has games (decision 53) |
| 13 | `91da623d` | `stats.load_live_leagues` guards the constructor and the first load (decision 54); `StatsMLB.pitchers` and `load_live_pitchers` go |
| 13 | `54592607`, `de8273ae` | The board's `Stat` is `archive_market(League, Market)`, and `market_display_name` folds through `archive_market` (decision 55) |
| 13 | `fba7ccc4` | `parlay._parlay_fun` reads the spread through `resolve_leg_stat` (decision 56) |
| 13 | `ebb7d4c0` | `correlation._process_league_games` walks the board one game date at a time (decision 57) |
| 13 | `1bafd0a5` | The pick'em snapshot's writer and its plumbing go to `src/deprecated/pickem_snapshot.py` (decision 58) |
| 13 | `3ab2ba3d` | The NHL's retention is the 1,461 days the NBA and MLB read (decision 59) |
| 13 | `2b923049` | The five unguarded WNBA ratios are guarded (decision 60) |
| 13 | `6504d3a1`, `cf30f91f`, `aa67944f` | The second cleanup pass (decision 61) |
| 13 | `142f73e9` | `StatsNHL._update` writes `config/goalies.json` (decision 62) |
| 13 | `f03e4a1a` | `stats/mlb.py` maps the feed's team codes through `helpers.text.ABBR_MAP` (decision 64) |
| 13 | `6dcc7625`, `86965b06` | The CRPS panel leaves Lab › Diagnostics; its chart goes to `src/deprecated/lab_diagnostics_crps.py` (decision 66) |
| 13 | `8363fdc1`, `67546f5d`, `efd13041`, `28866a4e` | The style pass on wave 13: `prediction.cli._name_markets`, `dashboard.legs.leg_cell_market`, one loop for the five extra book-weight fits; nothing computed differently |
| 14 | `a5cdf2aa` … `62a49e89` | Living docs, docstrings, help and hints trued to the code; seven closed briefs to `docs/archive/`; nothing computed differently |
| 15 | `83eec8f0` | `Archive.to_pandas` exempts `Totals` from the 2023-05-03 cutoff (decision 71) |
| 15 | `e0b292ce`, `2e024bdb`, `08e65275`, `387bfee6`, `107571d7`, `76ff458c`, `fed36c21`, `efe81edc` | The cleanup push's code and text (decision 76): pre-commit `-n0`, `Scrape()`'s key, the chart font, four orphans, the research-brief citations, three comments |
| 15 | `6307e007`, `2e8dffff`, `7d03088a` | Docs of decisions 73 to 76: two lanes archived, sleeper-parity stage 5, Mixture, story-balance, model_improvement_track |

Recorded, not in scope: Gate 2's `n`, `book_bss` and over-rates still count unposted rows
(`nightly._build_cell_row`); Kelly's "live_bss" is a CLV beat-rate remap with 1–19 legs → full trust (§6,
the I4 bullet); the `SPORTSTRADAMUS_ARCHIVE_DB` default-path expression now repeats in four scripts
beside `helpers/archive.py`. `docs/ship_gate.md`'s devel → main row no longer describes a
`profit_sim_kelly_yield ≥ 0` money rule: no code implemented it and the owner's rule forbids it.
Three grading rules the cleanup pass's style review found, none reached today:
`analysis._resolve_leg` grades a leg whose realized value is NaN as a hit (no NaN sits in the
78 served market columns of the training box's gamelogs); a `" vs. "` parlay leg is graded on
the sum of its two players where its offer row is graded on their difference (no matchup offer
is scraped); and `underdog_pickem._parlay_shrinkage` gives a parlay whose legs all lack a cell
mapping full trust, where Kelly's own no-evidence case gets 0.01.

## 5. Research program (three briefs, `research-analyst`, model opus, read-only)

Rules in every brief: post-fix data only; walk-forward with weekly refits; day-clustered CIs (block
bootstrap by Date); pre-registered candidate sets with Holm; price at the platform payout via
`realized.settled_offers`; NFL reported separately; no pickles, default flips or serving-path edits; no
model-free engine, no pull/demote/withhold; KILL is a valid verdict; archive opened read-only.

| Brief | Decision question (short) | Deliverable | Gates |
|---|---|---|---|
| R1 `researcher_trust_layer.md` | Add a selection-aware probability layer (identity / L2 logistic in logit space / monotone shallow GBM; fit on all posted legs, applied post-argmax, monotone in served p, keyed to Model Version) whose `Trust Prob` drives recommendations at a threshold on `Trust EV − 1 ∈ {0, .02, .05, .08}`? KILL if nothing beats served `Win Prob` on the selected tail, or if dropping served p does not worsen day-clustered log-loss | ablation, regime map (routed to R3), volume/ROI frontier per league per day, the no-fit default | **KILL (2026-10-04)** — [archive/researcher_trust_layer.md](../archive/researcher_trust_layer.md): the pre-registered ablation fires (dropping served p costs no out-of-sample log-loss: logistic +0.00038 nats, p = .07; GBM p = .75), Tier B 0 of 12 after Holm; a market-anchored `Trust Prob` is the model-free engine the owner ruled out. Reopen only if the mid-November confirmatory re-run (frozen `r1/prereg.py`, I2 columns) passes K2 and live λ_sel's trailing-30-day CI lower bound clears 0 |
| R2 `researcher_parlay_engine.md` | Which pricer repairs (symmetric game Σ, live payout tables, Underdog's per-game correlation modifier, book-anchored or Trust leg marginals) make priced joint probability and payout match realized parlays; what per-pick Underdog convention should the single-leg ledger use? | stage 1: payout tables vs live capture + the 1.78 verdict; stage 2: Σ, m, marginals, the beam gate as curse amplifier, ranked I5 list | **DONE (2026-10-04)** — [archive/researcher_parlay_engine.md](../archive/researcher_parlay_engine.md). Stage 1 READY: √3.5 = 1.8708 replaces 1.78 on the owner's go (the live table is confirmed, §2). Stage 2: the pricer is sound and the legs fail — on 818,844 synthetic parlays from post-fix posted legs the beam's admitted legs read 8–11 pp above their hit rate, compounding to realized/priced 0.71–0.77 at 2 legs and 0.41–0.44 at 5, while book-anchored marginals (a diagnostic, model-free) calibrate every size. I5b symmetric Σ KILL (Δ log-loss −0.00011, p = .12; the shipped Σ is indistinguishable from independence); I5c `underdog_tax.py` KILL (m < 1 on 3–7 % of entries, ≈ 0.99); I5a/I5d/I5e GO as truth, not calibration (68 % of gated entries carry a stale pair bonus, mean ×1.24; the ledger underpays stored winners by a median 1.68×). The Model EV ≥ 2.0 beam floor amplifies the curse (realized/priced 0.50 → 0.00 across priced-EV bands; today's gated entries hit 2.3 % against 29.7 % priced). The joint-ratio acceptance moves to I6 |
| R3 `researcher_train_serve_skew.md` | Which train/serve skews (unquoted pooling at w = 1.0 vs w, combo-sum quotes, tie label, DFS-rung Market Prob from the consensus shape, tail reliability by z and alt flag, NFL Under bias by era, the evaluation design) explain the .650-read vs .496-hit gap, ranked by measured share? | ranked defect table with CIs, fix per defect, files touched, retrain need; the tail-scorecard spec (E) | **DONE (2026-10-04)** — [archive/researcher_train_serve_skew.md](../archive/researcher_train_serve_skew.md). No skew in the calibration chain or the quotes explains the gap. Post-fix recommended legs read .611 and hit .486 (+12.5 pp, 6,138 legs); walk-forward that is bulk miscalibration +1.5 pp [0.8, 2.1] (11 %) plus a selection residual +11.7 pp [9.2, 14.4] (89 %; 87 % at √3.5), the only two tests that survive Holm. Every pre-registered skew moves the cohort gap by ≤ 0.3 pp: unquoted pooling, combo-sum quotes, the rung decode, tail shape and the NFL matchup leak are KILLed as gap levers; the tie label is a correctness fix. Rank 1 is selection on model–market disagreement where the served mean carries little at-market information (within-cell logistic encompassing, b_model: MLB .26 (.07) live against .49–.66 offline May–August; NFL ≈ 0 offline at every distance from the booster's training cutoff, and live). R1's KILL closes the layer route, so the fix is model information (I6, §6). Largest live effect, exploratory: MLB recommended legs overstate by +5.6 pp in a version's first four days against +13.5 pp from day 4 on (difference −7.9 [−12.3, −2.9]; repeats in all three post-fix retrains; cause unknown; I6g is the test). Real but small or inconclusive: serve-path feature parity (same pickle, line and book leg: sd(ΔP) 2.8–5.9 pp; NFL overstatement 0.68× [0.58, 1.11] on training-path features), book-leg timing (training takes the quote at or before 12:00 UTC on game day, decisions come 5–11 h later, ≈ 0.1 of b_model), in-game quotes in training game lines (MLB total against team runs .57 enriched, .16 pre-game), the MLB comp snapshot look-ahead |

R1 ran on proxies (decision time = the platform's last ladder poll of the line, quote class from
`Market Projection` + `Model Version`); its confirmatory re-run needs ~4 weeks of `DECISION_COLS`
(mid-November). The scripts the briefs cite under `scratchpad/r1`, `r2` and `r3` are kept on the dev
box at `~/backups/sportstradamus/2026-10-04-honest-receipts/` (`/tmp` does not survive a restart
there; data artifacts are not kept, rerun the producing script): R1's re-run starts from
`r1/prereg.py`, the parlay joint-ratio acceptance from `r2/`, the estimator prototypes behind E and
I6g from `r3/`.

## 6. Gated implementation (one module per subagent; entry = the brief's verdict + the owner's read)

The labels, in plain words. **I4** is the trust layer: a second probability fitted on top of the
model's (killed). **I5** makes the parlay builder and the ledger pay what Underdog really pays: the
per-pick value of a leg, the payout table, the pair bonuses, the settlement. **I6** gives the models
more information than the market has and makes training see what serving sees. **E** is the tail
scorecard, the offline replay of the live recommendation rule. A suffix (I5a-1, I6e) names one change
inside its group; §8 restates each open one as a plain question.

- **I4 trust layer — KILLED with R1 (§5); nothing below is built.** The one piece that needs a new
  home is the Kelly no-evidence case: `strategies/kelly.py::resolve_shrinkage` still returns
  `clip(live_bss)` for any `live_n > 0` (full trust on 1–19 legs); R1's abstain rule (no recommendation,
  Kelly 0, below 2,000 legs and 5 dates per league) is the evidence-backed default. Original sketch,
  for the record (Tier A unlocked I4a–c, e, f; Tier B unlocked I4d; behind a selection knob, off by
  default): `prediction/trust.py` (≤ 250 lines: `trust_features`, `fit_trust`, `apply_trust`; artifact
  via `helpers/io`; NaN where no fit, never full trust); a `reflect` refit step after the realized
  ledger; `offer_records.finalize_records` sets `Trust Prob`, `Trust EV`, `Rec Rule` and a decision-time
  `Recommended` flag after clip/argmax/phantom gate; consumers one subagent each (`stories/menu.py`,
  `correlation.py::_select_bet_offers`, `strategies/kelly.py`); a `policy_v2` ledger persona A/B'd
  against `policy_v1`; Receipts shows model read and trust read side by side.
- **I5 parlay pricer — built (decisions 8, 9; R2's ranked list, §5): a factual refresh, not a
  calibration fix.** I5a-1 `UNDERDOG_BOOST_BASELINE` = 1.83 (§3). I5a-2 `underdog_payouts.json` Power
  2–8 = 3.5/6.5/12/20/35/65/120 and the live Flex tiers; the payout tables run to 8 picks while the
  search still builds to 6; a Flex tier with k losses pays on the n−k largest pick multipliers.
  I5a-3 `banned_combos.json`: every Underdog value above 1 is 1.0 (523 values), and so is the one
  same-direction 0.9 on a pair Underdog quotes untaxed (opposing quarterbacks' passing yards). I5d the
  ledger settles
  at T_live × Π b × m through `payouts.outcome_payouts`, the one rule the pricers and settlement
  share, under `policy_v2`. The `even_picks` paper bettor of decision 12 is `policy_v3`: v2 built
  it behind the shared leg gate, which no recommended leg passes, so it was rebuilt on the
  recommendation rule. `policy_v4` then made each bettor's forty copies independent and took the
  4-pick Flex tier the app pays (decisions 17 and 25;
  [sim-bettor-ledger.md](sim-bettor-ledger.md) §10, which also carries the ledger's open scars).
  I5e the premise rewrite in
  `parlay-dependence.md` §1 and `PARLAY_AUDIT.md`. Four pair values come from payouts the owner
  read in the app (decision 25; the quotes are in [underdog_api.md](../underdog_api.md) §6.8):
  opposing running backs' rush attempts, one Higher and one Lower, 0.748 (the file held 0.86); the
  same on rush yards, 0.783 (0.95); a quarterback's pass yards Higher with his own running back's
  rush yards Lower, 1.0 (0.95); a quarterback's pass TDs Higher with his own receiver's TDs
  Higher, 0.794 (the file banned the pair). All four file values were off, so the other sub-1
  values and bans are unproven; the opposite-direction slot of the lifted ban and the nine other
  banned keys of that family stay as they were. **Struck:** I5b symmetric Σ, I5c
  `underdog_tax.py`. `_MODEL_EV_FINAL_FLOOR` (2.0) and `_BOOKS_EV_FLOOR` (0.9) unchanged: no tested
  value returns ≥ 1. I5's own acceptance = priced payout equals the quote on owner-captured slips
  (one Power, one Flex, one stacked same-game pair) and the ledger settles at it.
- **I6 model information and train/serve alignment — R3's build order (§5), opened by decision 13.**
  None pulls or demotes a model. `pipeline.py` is over 5,000 lines: extract helpers, never grow it.
  Retrains go through `meditate` and the existing gates unchanged.
  1. **I6e serve-time feature log + parity monitor — built** (no retrain, no served probability
     moves). `model_prob` upserts one row per scored player and game to
     `data/runtime/feature_log/date=<game date>/` (`prediction/feature_log.py`: the frame fed to the
     model, the decision-time book leg, the model's outputs, the probable pitcher; 30 days kept; never
     `history.parquet` or a dashboard snapshot). `sportstradamus admin feature-parity` replays it by
     hand, not from cron: it checks each logged batch reproduces through the file carrying its
     `Model Version`, compares each row with its training-matrix row once the matrix catches up
     (flag at a per-cell sd(ΔP) above 1 pp), re-scores under the other versions on disk
     (`--versions`) and lists the settled cells with no log rows. No logged game is in a matrix yet,
     so the first real parity read follows the next `meditate`. First finding, with no matrix needed:
     serving feeds `inf` where training holds 0 (all 48 logged WNBA rows carry it in `Team
     BLK_RATIO` or `Team OPP_BLK_RATIO`; setting it to 0 moves WNBA PR by −0.7 pp on average). Then
     repair feature by feature (R3's worst: NFL completions, receptions, interceptions, qb yards).
  2. **I6g version-age test** (pre-registered; a cadence change only if it confirms). `meditate`
     keeps each model file it replaces for 14 days (`training/prior_models.py`, `data/models/prior/`)
     and `feature-parity --versions` re-serves the logged legs through it offline.
     Primary statistic: fresh (version days 0–3) minus aged (days 4+) recommended gap, day-clustered,
     on NBA and NHL from late October; secondary: within-market b_model and the paired same-leg
     difference between versions. Confirmed → retrain every 3–4 days. Null → KILL: the MLB pattern
     was calendar or regression to the mean.
  3. **I6f as-of alignment of training's archive inputs.** The pre-game cutoff for game lines is
     built for every game ingested from now on, on both boxes: `_enrich_team_markets` takes each
     sportsbook's newest Moneyline and Totals quote stamped at or before 15:00 UTC on the game date
     (`GAME_LINE_TRAINING_CUTOFF`, `Archive.get_team_market_map(cutoff=…)`), and a historical
     game-line fetch is stamped with its snapshot time. Serving's read of today's game is another
     branch and does not change. On live-polled MLB team-games the team total correlates .58 with
     the runs then scored as it was read before, .18 at the cutoff; seasons that hold only pre-game
     snapshots sit at .16–.19
     ([archive/game_line_cutoff_design.md](../archive/game_line_cutoff_design.md)). Games already
     stored were re-read on the training box for MLB, NHL, WNBA and NFL (decision 20; NBA stays
     out, because 382 of its team-games would take in a total inflated ×1.44): the 780,976
     backfilled game-line stamps moved onto their game dates, the four gamelogs were re-read at
     the cutoff, and the six game-context columns of the 72 cached matrices were patched from
     them. Training rows with a real game line went from 4.6 % to 99.2 % in MLB and from 49.1 %
     to 99.7 % in NHL (WNBA 73.6 → 76.1 %, NFL 76.9 → 79.9 %), and on MLB's 2026 games the team
     total's correlation with the runs then scored fell from .58 to .18. What the re-read leaves:
     the four slope features keep their cached values until a matrix is rebuilt (§7); every
     patched matrix has a new hash, so a stored model-selection verdict is no longer evidence for
     its cell; production's own gamelog is untouched (decision 31, not yet run); and a model
     changes only at its next retrain. The scripts, their output and the pre-run copies are in
     `~/backups/sportstradamus/2026-10-08-gameline-reread/`. The NFL enrichment miss is fixed
     (decision 21): since the player column was renamed in September 2025 the schedule joined new
     rows on the week alone, so a new row was looked up under its week's first game day and 84 of
     the 90 team-games of 2026 sat at the default. The join is on team and week now, and 90 of
     the 96 hold their line; the other six are the 49ers' games (§7). On the live player and
     schedule frames for weeks 1 to 4 the old join gave a player row its own game day on 89 of
     1,410 rows and the new one on all 1,410, with no other column changed.
     Still open: the other training reads that take the newest quote for a past date (the
     MLB plate-appearance multiplier and starter-win leg, the NHL goalie legs, the Moneyline and
     Totals book-weight fit); the book-leg cutoff aligned to decision time where ladder polls exist
     (`TRAINING_LOOKBACK_HOURS`); archiving the daily Savant affinity CSVs, so MLB comps can be
     rebuilt point-in-time for 2027.
  4. **Hygiene.** I6c — done (decision 14): `_brier_temperature_loss` is the Brier alone, with no
     penalty toward T = 1. T lives in each model file, so a cell changes at its next `meditate`.
     Refit on the 78 served cells, 73 pass the gates with the penalty and the same 73 without; the
     recommended-leg gap on the tail scorecard falls about 3 pp and recommended volume about a
     quarter; the cost is log-loss on the alternate rungs of four NFL cells (receptions, rushing
     yards, interceptions, passing TDs). The retrain checklist and the reversal rule are
     [archive/researcher_temperature_ridge.md](../archive/researcher_temperature_ridge.md) §5.
     I6a — built (decision 19): a push is y = ½ at the eleven sites that derive the label, through
     `training/labels.over_label`, the ship gates' own label included. Brier, calibration error
     and the empirical Over rate are scored on the half label; log-loss, AUC, accuracy and
     precision on the rows that settled. The 73 cells that ship on the stored test sets still
     ship. A cell's served read changes at its next retrain: 0.5 points down on average and up to
     11.5 in NFL `qb tds` ([archive/push_label_design.md](../archive/push_label_design.md)). Until
     then three stored cells read a worse calibration error under the new label, all inside the
     gate's .075 (NFL `qb tds` .015 → .063, NFL targets .000 → .060, WNBA TOV .020 → .033), and
     `model_stats.parquet` mixes the two labels from the first `meditate` after the change until
     every cell has retrained. Still open: `step` passed to SkewNormal `get_odds` in
     `_step_compute_test_probabilities` and `_step_calibrate_temperature`. I6b is optional: align
     training's non-authentic fusion to serving, never the reverse.
  5. **I6d NFL at-market information** (research; no method is guaranteed). Order: after I6e's parity
     repair, and after the NFL count quote is levelled (`ev` and `under_prob` disagree, which
     contaminates every fused NFL number). Acceptance: fixed-effect logistic encompassing with
     b_model's day-clustered CI lower bound above 0 on held-out quoted player-games at the
     decision-time reference, plus the tail scorecard. g1 passes every NFL volume cell today and
     cannot judge this.

  **Not to build** (each measured; brief §5 and §7): w = 1 serving for unquoted rows; price-blind or
  distance-conditional recalibration as a gap fix; rung admission as a gap lever (it is a volume
  lever); combo-shape alignment; a staleness gate on model rows; a `_TRAIN_FRACTION` change; any trust
  layer or fixed scalar shrink toward the book; the DFS main line as a book leg for unquoted legs
  (owner decision 5, §2).
- **E tail scorecard — built** (`sportstradamus admin tail-scorecard`; `scripts/tail_scorecard.py`
  with `tail_pricing.py` and `tail_information.py`, tests under `-m diagnostics`; spec = R3 brief
  §6). It re-serves every held-out test row at each Underdog and Sleeper rung the archive `ladder`
  held (the rebuilt probability matches the persisted one to 1e-13 on every scored cell), hands each
  slate to the live `finalize_records`, grades the survivors with `realized.settled_offers`, and
  prints the selected-tail gap per cell, league and overall with day-clustered CIs. Beside it: the
  live gap for the same cells and windows, split by Model Version age, and the information test (the
  model-only forecast against the market at the decision-time and training-time references, cell
  fixed effects). It never bets a side history shows was not posted; a side history never priced
  pays 1 / (price × overround) and is flagged `assumed`, so the `posted` row of the payout split is
  the clean read. It scores 32 of 91 test sets today (MLB 14, NFL 13, WNBA 5): 30 predate the
  columns the re-serve needs (`Book_EV` and the family's model-only parameters; 18 NBA, 11 NHL and
  WNBA FGA, until a retrain re-dumps them), 13 have no pickle and 16 have no ladder rung in the
  window. Readers: I6c, I6d and I6g, as an acceptance input. It routes to training, never to
  demotion, and becomes a sweep objective only after four weekly runs agree with live per cell.
- **Repairs found on the way — built (decisions 15, 16, 22, 23).** None moves a served
  probability.
  1. **A closing price waits for the game.** The nightly `reflect` used to give every history row
     without a closing price the newest sportsbook quote, games not yet played included, and a
     closed row then won over every later scoring of the same prop and line: before any game of
     one measured day, 53 % of the board's rows were frozen at an earlier snapshot.
     `clv.commence_times` now gives each row a kickoff (its own `Commence`; else its team's that
     day, because Sleeper posts none; else 20:00 UTC on the game date), and
     `clv.fill_from_archive` fills a row once that instant has passed, releases a row closed
     before it, and reads the close as of it. The read moved with the gate because 20:00 UTC is
     before kickoff on 77 % of the rows that carry one: on 200 evening predictions of one day
     the close as of kickoff differed from the 20:00 read on 78 %, by .009 on average. A ledger
     leg is read at the same instant (`_ledger_settlement.join_clv`). Rows closed before the
     change keep the close they have, so closing-line value and the Kelly trust figure built on
     it mix the two reads until the old rows age out of their windows.
  2. **A prop counts once.** `realized.settled_offers` keeps, for each prop and platform (`Date`,
     `Player`, `Market`, `Platform`), the rows whose `Scored At` is that prop's latest, for games
     from `realized.COUNT_ONCE_FROM`, the first game day whose rows close only after kickoff.
     Alternate rungs posted side by side at that scoring stay separate legs; earlier games count
     every posted line, and the page says so under the hero. The dashboard loader keeps
     `Scored At` for it. One scar: the page drops pushes before the count, so when a prop's
     last-scored line pushes, an older line of that prop is counted (no push among the 510,933
     settled rows on the training box; nine of them sit on a whole-number line).
  3. **Parlay trust under the league's key.** `correlation._leg_shrinkage` and
     `underdog_pickem._parlay_shrinkage` look a leg's trust up under the league's own market key,
     as the ledger's legs have since `policy_v3`; `helpers.cell_market` states the rule once for
     all three. NBA fantasy points on Underdog resolves its cell's figure (.163 on the training
     box's files) where it resolved the no-evidence .01, which dropped every parlay holding such
     a leg; NHL points and assists resolve their cells' 0.
  4. **One dashboard restart per deploy.** `scripts/run_job.sh` notes the commit before its pull
     and compares against it. It compared the checkout's last move, so after a deploy that
     touched dashboard code or a config file every job restarted the dashboard until the next
     deploy (26 restarts in one day).

Rollout per [model_improvement_track.md](model_improvement_track.md) §6.10: replay-validate in the
fixed profit sim, then A/B live through the D6 sim-bettor ledger. I5a-1, I6c and an I6g cadence change
move which legs clear the edge rule: tag each by Model Version / policy version and never pool across
eras.

## 7. Records, not tasks

- Success in numbers: the hero reconciles with the ledger to the leg (done); three verdicts (done, §5);
  I5 = the priced payout equals the quote on owner-captured slips (§6); I6 = the tail scorecard's
  selected-tail gap falls at like version ages (fresh against fresh: MLB's fresh-day gap is already 41 %
  of the aged one, so an unmatched before/after certifies the retrain calendar, not the retrain) and
  within-cell b_model's CI lower bound clears 0; parlays = realized/priced joint ratio in [0.9, 1.1] at
  every entry size on R2's harness, which needs per-leg hit/read ≥ .949 at two legs and ≥ .979 at five
  (fresh-version MLB legs sit at .91; nothing measured gets there).
- Standing risks: until the models carry more at-market information the per-leg selected gap stays near
  +10 to +13 pp and parlays compound it (R3 §8); I6d is research with no guaranteed method; the
  version-age effect is observational until I6g. The NFL team-feature cliff feared for 2026-10-11
  did not arrive: both boxes hold `team_data/NFL/2026` weeks 1–3, and the serve-time log shows the
  Team and Defense blocks filled on every row for the games of 10-11 and 10-12 (checked
  2026-10-05). It rests on the weekly `fp-fetch` from here.
- Phantom rows (unquoted, |Win Prob − Market Prob| > 0.15, dropped at `offer_records.py`) are not
  persisted, and R1 keeps it that way: keep the phantom gate, never write those rows into history.
- `strategies/profit_sim._settle_day` recomputes Kelly from prob and payout and ignores the
  `MAX_FAVORED_PAYOUT` cap, so a sampled above-cap leg is still staked in the dashboard sim (the
  nightly precompute's ranking column now zeroes it, so such legs only reach the sim through the
  floor sampling weight). Owner call whether the sim should mirror the board's cap.
- The orphans left in `analysis.py` after the retirement (`backfill_crps`, whose caller was
  already archived, the CRPS helpers reachable only through it, and `reconstruct_prob`) are in
  `src/deprecated/analysis_flat_odds.py` (decision 61). `analysis.py` is 581 lines; the natural
  split is outcome resolution / profit-sim precompute / scoring rules.
- Lab › Diagnostics' interval-coverage panel calls `compute_coverage` per row, uncached, twice
  (overall, then per league): ~4 minutes per rerun on the default 90-day window (158k
  predictions at ~0.7 ms each, measured 2026-10-04). Predates this lane; a vectorized or
  cached pass is the fix.
- R1 side findings for this lane: Sleeper voids a batter who does not start or bat, and the ledger grades
  those legs (about 5.4% of MLB hitter legs; Underdog's rule unverified); 24 of the 92 legs the best trust
  spec picked at the 5% threshold fell on 2026-09-23 behind one stale pre-lineup sportsbook slot (quote age
  is a selection variable on fallback rows; on model rows R3 finds the freshest quotes carry the larger gap);
  the model's information share is 0.47–0.61 in a version's first days and 0.05–0.14 one to two weeks later
  (version-age decay; R3 replicated it live with a level-robust estimator, §5; I6g is the test). R3 counts
  1.3 % of MLB hitter legs with no plate appearance; dropping them raises the recommended gap by 0.7 pp, so
  every hit rate here is slightly flattering.
- `book_fallback` recommendations are a model-free engine: 4,463 legs all-time at −9.9% on the decoded-book
  edge alone, 38 since 2026-08-31 at −26.0%. They keep their recommended flag (decision 10).
- The Receipts calibration panel guards a pre-cohort `calibration_summary.parquet` (no `Cohort` column → "No
  calibration data yet"); drop the guard after the first post-deploy `reflect`. The panel reads the nightly
  snapshot of every resolved leg, so its tiles do not follow the sidebar range or the page window (caption says so).
- The low parlay all-hit rates are real, not a resolution defect: on 898,592 parlays resolved both by
  `parlay_hist` and by history outcomes, `Misses` agrees on 98.4 % and all-hit rates agree within 0.2 pp
  in every platform × size cell (Sleeper 2-leg .115). Parlay Kelly (`Rec Bet`) sizes from the copula
  probability, shrunk toward 1/payout by the smallest leg weight but not corrected for the joint
  (`PARLAY_AUDIT.md` §2.6), while the selected set realizes 0.08 of priced: the parlay path's
  live-money exposure until I6.
- R2 open questions: NFL Σ rests on 7–11 games per team (pair-type pooled tetrachoric ρ with hierarchical
  shrinkage is a pre-registered parlay-dependence research item); Underdog's modifier m for MLB / WNBA /
  NHL / NBA pairs is unknown and assumed 1; Sleeper's bans are unverified; Flex calibration cannot be
  tested without the tier distribution; a mid-November re-run on the I2 columns needs ≥ 8 NFL weeks.
- R3 records: the September regime (offline within-cell MLB information falls in September, fused λ_c
  .86 → .28, established players most; the live MLB window is all September, so the first cross-league
  read is NBA and NHL in October–November); each booster is fit on the first 70 % of its matrix (NFL
  boosters end December 2024) and information does not decay with distance from that cutoff, so it is
  not a lever; training uses the realized opposing starter and serving the probable one (unmeasured
  until I6e persists it); validation weight stays high where live encompassing says ≈ 0 (receptions
  w = .80 against b_model −.10; lead: the mis-levelled NFL count quote); Underdog prices at an overround
  near 1.178 when the price moves, against 2 / `UNDERDOG_BOOST_BASELINE` standard (1.1236 at the 1.78
  R3 measured under; the scorecard re-estimates it each run); 17
  NFL rows carry game date 09-09 under a later-dated version. Estimator rules for any information read:
  cell fixed effects, an intercept, model-only probabilities and the decision-time reference (a
  through-origin slope is level-contaminated; pooled encompassing without fixed effects read .78 where
  the within-cell value is .32).
- Scorecard first run (rungs from 2026-08-30 to each cell's last test row; 40,568 rung rows, 1,149
  selected): read .653, hit .524, gap +12.9 pp [6.7, 16.1], against live +12.8 [9.7, 15.2] on the
  same cells and windows. NFL reads +15.8 [11.5, 21.2] on 904 legs against live +15.3. MLB reads
  +3.1 [−3.3, 8.0] on 240 legs against live +9.7 [6.6, 13.0]; its CI overlaps the live gap of fresh
  versions on the same cells (+7.9 [4.8, 11.6]; aged +10.5 [6.4, 14.8]), and 240 legs cannot tell
  version age from R3's known optimism (training-matrix features; in-game quotes sit in every MLB
  row of the window), so MLB needs more weekly runs before the scorecard is trusted there. By payout
  source: posted +15.5 (594 legs), assumed +10.1 (555). The brief's prototype (+10.1 on 1,615 legs)
  paid an assumed price on every side, so it bet sides the platforms never posted; the scorecard
  does not. Information test, within cell, logistic b_model at decision time: MLB −.13 [−.66, .45]
  (3,276 rows, 15 days, 9 cells), NFL −.16 [−.59, .30] (933 rows, 4 cells); neither can be told
  from zero. The brief's MLB .77 → .66 reference-timing pair reproduces exactly, but only on its
  530-row, 5-day subset holding both references, so the size of the timing effect is unmeasured.
- Consensus sanity check (owner request; record with tables in
  [archive/consensus_sanity_check.md](../archive/consensus_sanity_check.md)): the sportsbook consensus
  is not profitable on DFS legs where it shows an edge. With no model anywhere, the same-line mean of
  two or more sportsbooks and each platform's real payout rule, the consensus' edge legs return −17.5 %
  per pick [−23.7, −11.0] (1,760 legs over 31 days; −37.0 % as real 2-pick entries), and no edge floor
  turns it positive. The arithmetic reproduces two known answers: every side of every Underdog 1.00×
  leg returns −12.5 % per 2-pick entry (3.5 × ¼ − 1), and the consensus is calibrated in bulk (+0.65 pp
  on 39,332 lines). The cause is the market: the platforms price within 1.0–1.3 pp of the books (5 pp
  apart on about a dozen rungs a day), and 79 % of the apparent edges are Underdog one-sided longshot
  rungs where our consensus runs 3.6 pp high (one-way prices de-vigged at a flat 6.52 %, often two
  books, quotes hours old).
- The platform's cut is the hurdle, and it differs by kind of leg: 6.5 % per pick on an Underdog 1.00×
  leg, 10–12 % on a priced two-sided leg on either platform, about 20 % on a one-sided rung. The
  recommendation rule returns the cut on every kind but the Underdog 1.00× leg, where it sits near
  break-even post-fix when every posted line counts as a leg (2,153 recommended legs hit .533
  against .535 needed and .490 for the legs it passed, +4.4 pp [−0.4, 7.6]; no such gap before the
  fix date). That count is generous. History keeps a row for each line a prop was posted at, so a
  prop whose line moved counts once per line (46 % of the recommended 1.00× rows), and a bettor
  holds one of them. Each prop once, the same rule's legs hit .515 [.485, .545] on 1,640 props
  against .499 for the props it passed, below what a 2-pick needs. Receipts counts a prop once
  for games from 2026-10-09 (§6, repairs). A lead for selection scope and for I6, not a proven
  edge.
- Payout rule for any replay: bet a side only when its multiplier is provable from one poll (the
  record's "Method trap"; the check's first pass read a false +17 % without it). The tail scorecard's
  `assumed` payout class (§6) uses a looser tolerance, so read its `posted` row until that is tightened.
- Where the code stands against decision 5 (§2): the consensus is sportsbook-only in price and in
  line (decision 11; every reader and the ten edits are in
  [archive/consensus_line_trace.md](../archive/consensus_line_trace.md)). `Archive.get_line` is the
  median of each sportsbook's latest posted line and 0.0 when none posts one; an entry only a DFS
  platform posts is graded at `get_reference_line`, its line of record, which never feeds a
  consensus. A DFS line no longer votes on the line a sportsbook price is read at, no longer sets the
  median a sportsbook row is judged divergent against, and gets no fitted book weight
  (`book_weights.json` keeps its DFS keys until the next refit). Measured on the archive: the old
  line differed from the sportsbook consensus on 13 % of entries both kinds post (NFL 45 %, WNBA
  61 %, MLB 8 %, NHL 4 %); the board's `Consensus Line` changes on a third of rows and is blank on
  10 % (46 % of modeled player-market-days have no sportsbook line at decision time); the one served
  probability that moves is the book leg on about 1.5 % of mixed entries (median 0.02–0.05 before
  the model weight); no model feature reads the consensus line, so no retrain is forced. Left as
  they are: `Stats._book_mean_shift`
  (decision 24), `Archive.get_movement` and `get_line_history` (diagnostics on the mixed log), and
  `Archive.get_ev` on an entry only a DFS platform posts.
- The per-pick move to 1.83 (decision 8), replayed on post-fix legs: Underdog 1.00× recommended legs
  go from 1,082 at −6.2 % per pick to 1,657 at −1.8 % (hit .5365 against .5464 needed), counting
  every posted line; each prop once they go from 814 at −8.9 % to 1,262 at −4.5 % [−10.2, +2.5]
  (hit .522). 77 unquoted one-sided Underdog rows newly fail the ±0.15 phantom gate (14 of them
  posted NFL Overs that read .767 and hit .462); the boost cap and the per-player trim pass exactly
  the same legs.
- What an entry kind asks of a leg, from the tables the owner read in the app (decision 25), at
  even picks from different games. The hit rate per pick at which an entry breaks even: 2-, 3-
  and 4-pick Power .535, .536 and .537; 4-pick Flex .537; 6-pick Flex .538; 5-pick Flex .547;
  5-pick Power .549; 6-pick Power .553; 3-pick Flex .554. No kind is cheap: the best five sit
  within .003 of each other. This bullet once put the 4-pick Flex at .518 and 10.8 points above
  the 3-pick Power on the same legs. That rested on a one-miss payout of 1.8× in the payout file,
  a September capture the app does not pay (1.4×), and is withdrawn: a paired difference between
  entry kinds is tight because the legs cancel, and it says nothing about whether the table is
  right. The recommended even picks settled as real entries (history through 2026-10-07, 36
  days, each prop once, every leg in one entry a day, never two legs of a game, day-block
  bootstrap; 1,482 props that read .609 and hit .528): 2-pick Power −0.4 % [−11.4, +13.7],
  3-pick Power −0.1 % [−15.8, +20.5], 4-pick Power +2.3 % [−19.8, +32.4], 4-pick Flex −0.3 %
  [−16.7, +23.3], 6-pick Flex +1.7 % [−26.0, +48.9], 3-pick Flex −7.3 % [−17.0, +4.6], 5-pick
  Flex −8.0 % [−27.9, +20.0], 5-pick Power −10.0 % [−35.8, +30.0], 6-pick Power −11.8 % [−42.3,
  +38.6]. No kind is told from zero, and the 4-pick Flex is not told from the 3-pick Power (−0.2
  points [−2.3, +3.5]). Decision 8's 1.83 is the mean of the Power roots at the owner's sizes,
  while `payouts.py` builds 4- to 6-pick Underdog entries as Flex (`POWER_MAX_SIZE` = 3) and the
  paper ledger plays Power at 2–3 picks and Flex at 4–6; at the corrected tier a 4-pick entry
  whose legs read above .5385 prices higher as a 12× Power; it stays Flex (decision 34), since
  both kinds break even at the same .537 and the read that would choose Power is the one that
  runs 8 points high. By side, each prop once:
  recommended Unders hit .541 (1,074 props; .553 [.520, .592] per posted line) and recommended
  Overs .493 (410 props; .496 [.405, .560] per posted line). One window, cut after the fact, and
  4,142 of the 9,811 settled even-pick legs found no game in the training box's gamelogs, so
  "never two legs of a game" fell back to the team for them. Scripts and logs:
  `~/backups/sportstradamus/2026-10-04-honest-receipts/main/entry_types/` (`replay4.py` takes
  the tier as its argument).
- The production paper ledger committed nothing between its entries of 2026-09-20 and
  `policy_v3`, and the cause is its own leg gate and trust haircut, not a fault. The gate keeps
  legs whose read sits within .04 of the book, which are discounted favorites; the Kelly trust
  figure is 0 in 34 of the 70 served cells; and the best entry the beam kept on the 2026-10-05
  board priced at −24.9 % against a +5 % floor. `policy_v2`'s Even-picks bettor sat behind the
  same gate, which no recommended leg passes (0 of 25 that day), so it could not bet either;
  `policy_v3` rebuilds it on the recommendation rule at the served read. The replay, the trust
  figure cell by cell and the three bettors left as they are:
  [sim-bettor-ledger.md](sim-bettor-ledger.md) §10 Policy v3. The same figure gates the
  dashboard's parlay search (a parlay is dropped when its least-trusted leg sizes it under half a
  unit), so that board holds legs from the few cells above about .05: three parlays on the
  morning board of 2026-10-05, all NFL.
- On Underdog 1.00× legs the hit rate does not rise with the model's read (2026-08-31 to 10-02,
  each prop once, day-block bootstrap). Every posted leg by read: .50–.52 hit .500 (2,296 props),
  .52–.54 .495 (1,442), .54–.56 .498 (965), .56–.574 .494 (404), then .538 [.495, .586] on the
  430 just above the bar (.574–.59), .517 at .59–.61 (296), .499 at .61–.64 (304) and .497 at .64
  and up (125). Among recommended legs the eight strongest reads of a day hit .502 (237 props)
  against .526 for the rest (1,025): −2.8 points [−10.0, +5.1]. NFL's strongest (.64 and up) hit
  .425 [.281, .475] on 74. One window, cut after the fact. It is why the `policy_v3` Even-picks
  bettor deals its entries evenly over the day's recommended legs instead of keeping the entries
  with the highest joint read.

- MLB training skipped a market's profile on every game day of a matrix build but the first. In
  the twelve hitter matrices and pitches thrown, `Player z`, `Player home`, `Player moneyline
  gain` and `Player totals gain` are non-zero on 0.1 % to 5.6 % of rows (5 to 9 of the 122 to
  368 dates), against 92 % to 99 % in the six other pitcher matrices; the same skip left the
  team-defense columns at zero and kept the pitcher profile of the build's first game day. The
  cause was a guard meant to save work: `StatsMLB.get_depth` calls `base_profile(date)`, which
  moved the profile date and emptied the profile frames, and `profile_market` then saw the same
  market on the same date and returned early (`Stats._begin_profile_market`). Serving did run
  it. In the serve-time log the four columns are filled on every row of the cells whose model
  kept them (batter strikeouts, where training saw them on 5.5 % of rows), and the ten other
  hitter models carry no such feature at all. So those models have never learned from the
  player's standardized level or home split in the market they price. `base_profile` now
  forgets the market it held when it rebuilds (decision 29): walked in the matrix builder's
  order over the training box's last two game days, the share of a game's hitters with a
  `Player z` was .89 then 0 before the fix and about .9 on both after. The rebuild and retrain
  ran on the training box on 2026-10-11 for 16 of the 19 MLB cells: each matrix built cold, and
  the 13 that have a model retrained. In the 11 rebuilt matrices that had the defect `Player z`
  is non-zero on 88 % to 99.7 % of rows (home runs 60 %, stolen bases 33 %: only a player who
  records the event often enough gets one), and no game day is left without a profile, where
  337 to 359 a matrix had none. Inside the dates both cover, a rebuilt matrix keeps 100.0 % to
  100.3 % of the old one's quoted rows. On the ship gates no cell went from pass to fail and
  runs allowed went from fail (Gate 4) to pass. Skill against the book moved most in home runs
  (0.305 to 0.227), stolen bases (0.218 to 0.035, model weight 0.77 to 0.48) and hits allowed
  (0.026 to 0.002, weight 0.76 to 0.10), and the two tables are not like for like: the rebuild
  resolved every quote again, so about a quarter fewer validation rows count as
  sportsbook-priced (hits 10,299 to 7,775), `DaysIntoSeason` counts from one season start, the
  pitch-count projection sits on every pitcher row, and the test set reaches three weeks
  later. On held-out recommended legs the tail scorecard reads .709 against .660 hit before
  (262 legs, gap .049 [−.018, .111]) and .624 against .539 after (687 legs, gap .086 [.047,
  .118]; 636 of them Unders): the fix did not change the over-read or the Under lean. The
  rebuilt models expect 118 features where production's expect 108 (hitters, pitches thrown)
  or 112 (hits allowed, runs allowed): six game-line features and four player features that
  were constant in the old matrices, which training drops. Made-up offers for each team's last
  starting nine and starter, sent through the serve path in a sandbox on both file sets, score
  in all 15 cells (2,928 rows each), every number finite and all 118 features in the frame fed
  to each rebuilt model; no made-up offer found a sportsbook quote, so the blend with a quote
  did not run. The models went to production on 2026-10-12 and were checked on its 00:50 UTC
  board run (decisions 43, 48). Three cells were left out because a cold rebuild
  keeps quoted rows only and would cost them four months of rows: batter strikeouts and the
  hitter and pitcher fantasy points on Underdog keep their matrices, and batter strikeouts'
  model its gap (decision 67). Tables and logs:
  `~/backups/sportstradamus/2026-10-09-mlb-nhl-rebuild/logs/`; the probe:
  `~/backups/sportstradamus/2026-10-11-league-only-send/README.md`.
- The four slope features after the re-read (`Player` and `Defense` `moneyline gain` and
  `totals gain`: a stat regressed on the stored game lines of the trailing 300 days). The matrix
  patch leaves them at their cached values. Recomputed from the re-read gamelog on sampled dates
  of one market a league, against the cache, the rank correlation is .98 to .995 for NFL
  receiving yards (30 dates; .77 to .93 on the 2026 rows alone), .96 to .99 for WNBA points,
  .48 to .69 for NHL shots and .16 to .23 for MLB pitcher strikeouts, where the cached value is
  zero on 60 % of the rows that would now hold one. A patch is enough for NFL and WNBA; MLB and
  NHL need a matrix rebuild to see the re-read lines in these four columns (decisions 29 and
  30). MLB's 16 rebuilt matrices hold them; the NHL's rebuild is stopped (the NHL bullet
  below). They carry a median
  1.4 % of a cell's stored importance in MLB, 1.7 % in NHL, 0.6 % in NFL and 0.8 % in WNBA.
- No game line was stored for a 49ers or a 76ers game from 2024-05-07 until decision 28.
  `helpers.text.remove_accents` title-cases a team name, "San Francisco 49ers" became "San
  Francisco 49Ers", the abbreviation lookup missed, and `moneylines._store_game_moneylines`
  then stored neither side of the game. The archive's newest 49ers Moneyline row was for
  2024-02-11 and the 76ers' for 2024-05-02. The lookup now folds the config's names the same
  way (`moneylines._TEAM_ABBR`). No other lookup changes in the five polled leagues; in the two
  college lists, which no job polls, 159 more names resolve and two (Augustana (IL) and (SD))
  fold to one key. What the fix does not reach: 39 49ers games (78 NFL team-games: 34, 38 and 6
  in 2024 to 2026) and 176 76ers games sit at the default in training (decision 36), and in serving
  every player of both teams in such a game was scored at a 0.5 win chance and the
  league-average team total (22 players in the serve-time log for the Seattle game of
  2026-10-11) until an odds poll on the fixed code stores the game.
- A WNBA fantasy-points row on Underdog never got a closing price: 0 of the 1,062 in history.
  History files the market as `fantasy points underdog` and the archive as
  `fantasy points prizepicks` (`helpers.archive_market` renames it for NBA and WNBA), and the
  close read passed history's name through, as NBA's rows would have from opening night. Both
  close readers fold the name now (decision 32) and history keeps its own. Replayed on
  production's rows against the training box's archive, all 1,062 take a close; 776 older rows
  of the two leagues hold no line to read one at. The close is Underdog's own last quote, the
  archive's only book for that market as for NFL's: an evaluation reference (decision 5).
- Sleeper NHL assists, points and blocks legs read no correlation and could not settle until
  decision 35. Two hand-kept maps renamed a market to the league's key for Underdog's
  capitalized labels only, so the parlay search keyed a Sleeper leg `C1.AST` against matrices
  that hold `assists`, and a same-game parlay stored a `stat` the NHL gamelog has no column
  for: at settlement the leg would have counted as a push. These legs are 58 % of Sleeper's NHL
  rows since 2026-09-20 (2,641 of 4,578); the training box holds no Sleeper NHL parlay yet. One
  rule (`helpers.cell_market`) builds the map now, and all 23 NHL labels of the two platforms
  resolve to a market in the stored correlation matrices and to a gamelog column, where four
  did not (Sleeper's assists, points and blocks, Underdog's Blocks). The board itself still
  names NHL assists and points `AST` and `PTS`, and NBA and WNBA Underdog fantasy points by the
  platform's name, so a slip seeded from a story folds both sides to the league's key.

- The odds feed's opening night moved in mid-season for two leagues. The activity classifier
  (`moneylines._league_activity_record`) recorded an opener whenever a league was tiered
  preseason and dropped it from an idle record, and a league whose feed empties over a break
  comes back tiered preseason, as one nearing opening night does. MLB's opener has read
  2026-07-16 since the All-Star break and the WNBA's 2026-09-17 since a gap in September.
  `Stats.update` adopted each as the season start, so serving has counted `DaysIntoSeason` 113
  and 125 days short since, and so does every matrix row built in between; the pre-season rule
  (decision 26) would have dropped an offer dated before the first game after such a gap. Both
  ends are fixed (decision 41): an opener is captured only when the next game falls more than
  60 days after the last one seen, and a start a class holds gives way only to an opener more
  than 300 days after it. On production's first board run on that code the MLB day-of-season
  count in the serve-time log went from 85 or 86 to 200. What the fix does not reach: the two
  wrong dates stay in the feed's record until next season's capture replaces them, and the
  300-day rule is what ignores them; the NBA's and the NHL's classes hold last season's
  starts, so for those two the capture rule is the only guard this season; and a matrix row
  built while a wrong start was held keeps its short count until its matrix is rebuilt (16 of
  MLB's 19 are, decision 29).
- A player the season's roster did not hold was priced by the book alone, and the NHL's opening
  night went unscored. The lookup knew two states: the new season's roster, or last season's
  whole while the new one held nobody (decision 26). A WNBA or NHL roster learns a player at
  that player's first game of the season, and the NBA's holds whoever the seed file names. The
  NHL branch had no fall-back at all: it raised on a season the store did not hold and on one
  stored empty, the two errors in production's board log from 2026-09-19 to 09-30.
  `Stats._roster_player_df` now returns the season's rows and, for each name the season does
  not hold, last season's row (decision 37). Replayed on the training box in both of those
  states, 25 of 30 opening-night offers are scored where none was (the other 5 fall to the
  price rule for unquoted offers). The NHL update also assigned each pull to the season's
  roster, and the scraper answers a blocked pull with nothing, so one blocked pull emptied the
  season; it merges now. On production's first board run on that code the six NBA players
  offered for opening night are read from last season's row, each with a position, and the
  NHL's 2026 roster went from 676 to 677 rows, merged and not replaced. What the fix does not
  reach: a player who changed teams reads on the
  old team until a first game with the new one. A blocked pull, which returns nothing, fails
  the NHL's update once the season has games, so the board run exits 1 (decision 53).
- No Portland player game row was stored for the last 27 games of the WNBA's 2026 season.
  stats.wnba.com names the team `PDX` and ours is `POR`. The alias went in on 2026-06-15 for
  the player index, the player logs and the team logs, but not for the shot map, and
  `_build_wnba_player_profiles` inner-joins the player frame with the shot map on player and
  team: every Portland player dropped out of that frame, and `_assemble_wnba_player_rows`
  takes each game row's position from it. On production the gamelog holds 147 Portland player
  rows through 2026-06-17, all under `PDX`, and none for the 27 games from 06-24 to 09-22
  (about 270 rows), while the team log holds all 41 games; the 2026 roster held 14 of the 15
  teams. The shot map's code goes through the alias map now (decision 40), and since its first
  update on that code production's roster holds all 15 (236 players, 220 before). What the fix does
  not reach: the lost rows are not refetched, because an update pulls from the latest stored
  date on, and the 147 older rows read `PDX` until decision 68's repair renamed them.
- The deep dive's pairing cards were empty from 2026-07-03 (`13d01613`). A board row's
  partners were stored under the platform's label ("Receiving Yards") and looked up on a board
  renamed through the platform's map: on production's board of 2026-10-08, 0 of the 1,204
  partners stored on Underdog rows and 70 of the 123 on Sleeper rows named a market the board
  holds. A partner is stored under the board's name now, with its league (decision 39): on
  production's first board run on that code every stored partner names a row of the board (603
  same-team and 1,928 opposing on Underdog, 40 and 125 on Sleeper). A card names a market as
  its board row does (the platform's code for the four markets of decision 55); since that
  decision the label shown is the league's own name.
- One league's failed gamelog update cost every league its board. `prophecize` and the ledger
  build each updated every live league before scoring anything, with no guard: on 2026-10-10
  the NBA update raised on its missing seed file, and production's board runs of 19:50 and
  20:50 UTC and its 20:05 ledger run each ended in five seconds, until decision 26's fix was
  pulled at 21:10. Both load through `stats.load_live_leagues` now (decision 38). A league
  whose update raises is logged with its traceback and left out of the run, its lines still
  archived, and the board run exits 1 once its snapshots are written. The league is not scored
  on its cached gamelog, because the day's schedule is fetched only inside an update: without
  it every player reads as away and an MLB hitter faces no pitcher, on a board that would look
  like any other. The guard covers a league's constructor and first load too (decision 54).
  What it does not reach: the ledger build, which logs a failed update and raises no alarm of
  its own. No update has raised on production since, so the guard has run there only where
  nothing fails.
- An MLB row scored near first pitch, or for a game of a later date, was scored without the
  day's schedule until decisions 45 and 50, and history keeps that scoring for about nine rows
  in ten of the games measured. The schedule map
  (`upcoming_games`) is the only source of an upcoming row's `Home` flag, opposing starter and
  posted batting order. `_build_mlb_upcoming_games` kept a game only while the feed called it
  `Pre-Game` or `Scheduled`; the feed moves a game to `Warmup` before first pitch (a delayed
  one to `Delayed Start: Rain`), and the board run of :50 is the last before a game listed on
  the hour. The map also stops at the server's today, so an offer for tomorrow has no entry.
  A row with no entry reads every player as away, the opposing team's average in place of its
  starter, no head-to-head, a batting slot guessed from recent games and, in October, a
  regular-season game. History keeps a line's newest scoring: of the 4,863 MLB rows for the
  games of 2026-10-05 to 10-10, about 595 (12.2 %) keep one made with the entry (the run ten
  minutes before the start had it for one game of nine, and no run of the day before had it).
  Against the matrix row of the same player and date, a row served without the entry differs
  on 22 of 109 features, the opposing starter's by about one standard deviation, and those
  features carry 9 to 20 % of |SHAP| in the MLB cells. In history's own pairs of one line
  scored both ways the projection moves by 4.4 % on average and by more than 10 % for one pair
  in eight, where two runs that both had the entry differ by 0.08 %. Training is untouched: a
  matrix row reads the game as it was played. So MLB's live record (Receipts, the live gates,
  the closing-line comparisons) is mostly of readings the holdout never scored. Not measured:
  the regular season, and the cost in hit rate. The same status test decides who starts today
  in `helpers/text.get_mlb_pitchers`, which the Underdog scrape uses to tell a pitcher's
  fantasy-points offer from a hitter's. Both halves are fixed: a game keeps its entry until
  it is over (decision 45), and a later date's offer reads its own game's entry (decision 50).
  An offer dated beyond the three-day window still reads the team's next game.
- A Sleeper offer for a home team whose code is remapped read that team as its own opponent.
  `_sleeper_games_map` passes a game's two team codes through `ABBR_MAP` (`WSH` to `WAS`), and
  `_sleeper_prop_offers` took as opponent the first of the two that differs from the player's
  raw code: for a home `WSH` player, `WAS`. It dates from `93748aac` (2024-09-29), when the
  game's codes began to be remapped and the player's was not. History holds 8,841 such rows among its 308,644 Sleeper rows,
  every home date of five teams for as long as it goes back (July 2026): MLB Arizona 4,241
  and Washington 3,626, WNBA Phoenix 389 and Portland 377, NHL Washington 208 (42 of the 171
  Sleeper NHL rows on the board of 2026-10-11). History's own `Opponent` column is blank
  before 2026-10-05, so the count is taken from the gamelog's home dates; the stored parlays
  agree on every team-date they hold (the game key names the team twice, `ARI/ARI`, on all 33
  home team-dates, on none of the 30 away ones, and never on Underdog). Such a row is scored
  against the team's own defense and sits in a game of its own for stories and parlays.
  Where both platforms offer one player at one line, the projection differs on 88 % of
  Arizona's home pairs and 95 % of Washington's (median gap 1.3 % and 4.7 %) against 14 to
  24 % of MLB's other pairs, and on all 153 of the NHL's against 16 to 27 %; the WNBA's two
  teams show no move (38 and 32 % against 23 to 45 %). Sleeper's NFL codes and the WNBA's
  `WAS` need no remap and are not affected. The map holds three NBA codes (`GS`, `PHO`,
  `NOP`); last season's stored parlays show no sign of it on the five home team-dates they
  hold for those teams (decision 46).
- Production's MLB comp tables are from March. MLB comps are read from two Baseball Savant
  match-score files under `player_data/MLB/` that only `meditate` refreshes; production runs
  no `meditate`, and the sync's file list never held them. Production's are dated 2026-03-06,
  the training box's 2026-10-10. Of the hitters with comps on both boxes the two lists share
  22 % of their members on average (pitchers 31 %), and 112 hitters have comps only on the
  training box and stand as their own comp on production. `Player comps z` and `Defense comp
  n` differ between a served row and the matrix row of the same player and date on 94 to 97 %
  of rows, by 0.7 to 0.8 standard deviations; they carry 2 to 5.6 % of |SHAP| in the hitter
  cells (decision 43).
- A rain-shortened MLB game never reached the gamelog. `_mlb_final_game_ids` took a game
  whose state was exactly `Final`, and the feed ends a called game as `Completed Early` for
  good. The gamelog lacked 11 such games (4 in 2024, 5 in 2025, 2 in 2026) and two `Final`
  games of 2025-03-27 that parse today: each of their players lacked one game in the rolling
  features, and the game gave no training row. The 58 postponed listings are all in, because
  a made-up game keeps its id and ends `Final`. Both states count now, and the 13 games were
  refilled on both boxes on 2026-10-15, each at its date (the feature code reads a player's
  last games by row position), so the matrices built before do not hold them (decision 51).
- The sync to production mirrors every league, and seven NFL flags reached production before
  their model files. `scripts/sync_to_prod.sh` has no league switch: it mirrors the model
  folder with delete and sends the book weights, the calibration file and the stats table
  whole, and each of those holds every league. The 19 NFL model files of the ship-15 campaign
  (2026-09-29 and 10-03) were never sent, and `8dfbe48c` (2026-10-10) moved seven NFL cells
  from `withheld` to `devel`. A withheld cell is not served even with a file on disk, and a
  cell that stops being withheld is served by whatever file is there: from the board run of
  21:50 UTC that day production scored five of the seven (yards, targets, completions, qb
  yards, fantasy points underdog; the other two had no offers) on its files of 09-27. Six of
  the seven cells hold another recipe there than the campaign confirmed, and blend weights
  fitted before only sportsbook quotes counted: on the rows that carry a sportsbook quote
  the book's share is 47 % for yards, 10 % for qb yards and 72 % for completions, where the
  campaign's files give it none, none and 32 %. Production's stats table marks those files
  `ship` False, but four of the five fail only a Gate 1 scored on rows since classed as
  DFS-only; under the rule in force qb yards alone fails a gate (Gate 4). By the 14:02 UTC
  board of 10-11 that was 1,328 history rows for the games of 10-11 and 10-12, and 566 of the
  board's 2,668 NFL rows. It was found by counting, league by
  league, what a sync meant for MLB would send, which is a step before any sync now. The
  owner was told at 14:59 UTC that day; the send is run by the session (decision 47). A one-league send
  (`~/backups/sportstradamus/2026-10-11-league-only-send/`) takes production's own three
  shared files, replaces one league's entries, and sends them with that league's model and
  correlation files; it deletes nothing and keeps every file it replaces on production. It
  was rehearsed on a copy of production's own files, undo included, and is now
  `scripts/sync_to_prod.sh --league` (decisions 49 and 52).
- Every NHL matrix build raised on its first game day until decision 44's fix. On a past
  date `Stats.get_depth` left-merges the day's gamelog players onto the roster (`6eac660c`, 2026-09-25, so a player
  the roster lacks still gets a depth slot). Such a player takes a filled-in 0 in every roster
  attribute, the NHL's roster rows carry a text one (`shootsCatches`), and the depth rank ran
  over every column before one was picked, so it compared text with a number and raised
  `TypeError`. 486 of the 570 NHL game days since 2024-02-10 hold such a player, and all 12 of
  the new season: a cold rebuild and a retrain's top-up both raised, where a cold rebuild
  worked in July. Serving reads today's roster without the merge and is not affected, and the
  NBA's and the WNBA's rosters hold no text attribute beside the position. It was found on
  2026-10-11 when decision 30's rebuild raised: the 15 cached matrices were put back byte for
  byte and no NHL model changed. The fix (`cf64d023`) has the rank read the usage column
  alone. With it the matrix builder builds the game days of 2024-02-10 to 02-16 on the
  training box for a quoted and an unquoted market, at about ten seconds a game day for the
  quoted one: a cold start of the 15 matrices is about a day in one process. The pipeline's
  full-rebuild mode, which builds outside the canonical folder with the NHL's caches on, is
  no shortcut: on two weeks of one quoted market it gave the same 3,039 rows cell for cell,
  at 9.7 seconds a game day against 11.2 (decision 44). The 15 matrices were rebuilt on
  2026-10-12 and the ten dependent models retrained on 2026-10-15 (decision 69). Record:
  `~/backups/sportstradamus/2026-10-09-mlb-nhl-rebuild/nhl_crash/README.md`.
- The WNBA gamelog kept the infinities its ratio stats divide into.
  `StatsWNBA._assemble_wnba_player_rows` and `_assemble_wnba_team_rows` cleaned the frame of
  them and dropped the result, where the NBA's builders guard each division. The training box's
  WNBA gamelog holds 4,948 infinite cells in 27,098 rows (`BLK_PCT` 4,780, `FTR` 168) and its
  team log 156 (`BLK_RATIO`, `OPP_BLK_RATIO`); the NBA's hold none. A window that holds one
  such game averages to infinity and the feature is zeroed: `Team BLK_RATIO`,
  `Team OPP_BLK_RATIO`, `Defense BLK_RATIO` and `Defense OPP_BLK_RATIO` read 0 on 21 to 23 %
  of the WNBA points matrix's rows, never in the NBA's, and `Player FTR` on 8 % against
  0.4 %. Each division is guarded now (decision 60). Training and serving read the same
  values, so the fix moves the features only with the WNBA's repair, rebuild and retrain
  (decisions 68 and 72).
- The parlay search's `Fun` score read no calibration on Underdog until decision 56. A game keeps its top
  parlays under three sort keys, model EV, recommended stake and `Fun`
  (`correlation._append_parlay_rows`), and `parlay._parlay_fun` scores an Over leg by its line
  in units of the market's spread, read from `stat_std` and `stat_cv` under the leg's `Market`.
  Inside the search that is still the platform's label ("Points"); both tables are keyed by
  the league's own name (`PTS`), and the board's rename runs after the search. Of the label
  and league pairs that have a calibration, none of Underdog's 96 is a key and 13 of Sleeper's
  65 are. The lookup then takes its default of 1 and the score is one plus the raw line: the
  parlays `Fun` keeps are the ones whose Over legs carry the biggest numbers, a passing-yards
  Over before any receptions Over, on every board.
- A team with offers on two game dates got one game frame. `correlation._assemble_game_frame`
  takes every leg of a team whatever its date and names the frame by the team's most common
  opponent and date. Both games' legs are then scored with same-game correlations and can
  share a parlay, and a line offered at one number on both dates is dropped as a duplicate.
  In production's parlay store 23,974 of 2,236,339 parlays hold legs on two game dates (WNBA
  20,000, MLB 3,974), every one stored for July or August 2026 (2.3 % of August's). None
  since, and no team held two dates on the board of 2026-10-11; it returns whenever a platform
  posts a team's next game before the current one is played. Each game date has its own
  frame now, earliest first (decision 57). Not reached: the dashboard keys its correlation and
  modifier maps and its builder pool by game alone, so a matchup on the board on two dates (an
  MLB series, first due April 2027) still reads as one game there. A leg with no date joins no
  frame: no history row of either platform lacks one (0 of 571,233 on 2026-10-15).
- `current_pickem.parquet` had a writer and no reader. The dashboard's batch Pick'em page is
  retired and nothing else in `src/` reads the file. Every board run built it
  (`cli._write_pickem_snapshot`): a parlay search per contest variant per platform, about ten
  seconds of a ten-to-fourteen-minute run by the progress bars of production's 17:50 UTC run
  of 2026-10-11, and a failure there was logged and never reached the exit code. The board run
  no longer builds it (decision 58); production's last copy stays where it is. CLAUDE.md,
  [ARCHITECTURE.md](../ARCHITECTURE.md) and `strategies/README.md` no longer say the dashboard
  reads it, name the recommendations file `{date}.yaml` (the code writes
  `{date}_{platform}.yaml`) or spell the commands `poetry run pickem-build` and `poetry run
  kelly`.
- Ask 11's first build could have cost an MLB platform its board. It dated the records the
  parlay search hands `StatsMLB.get_depth`, so a combo leg offered on two dates was listed
  twice, and the second split of its name raises `KeyError` into `main`, which drops that
  platform's board for the run. Found in review before any push: the records are one a leg,
  for its earliest game (`3bd8ff59`, `6a6df14e`). No MLB combo leg is on production's board or
  in its history (0 of 4,429 and 0 of 585,032 rows on 2026-10-15).
- The NHL hits model reads its high-volume players about 15 % low. After the 2026-10-15
  retrain it fails Gate 4 (randomized PIT KS 0.108 against a 0.05 bar; the old file 0.027) and
  Gate 6 (the fused mean's top-tier ratio 0.849, upper bound 0.928). Neither gate reads the
  temperature. The low read is not new: the old file's raw model mean reads 0.847 of the
  outcome on its own test rows, where its blend with the DFS line read 0.995, and production's
  live rows on that file read 0.844 (Projection over Actual, 273 settled rows since 10-01). The
  new file has no sportsbook-priced validation row (1,396 before, all since classed as
  DFS-only), so its blend weight is 1.0 and no line pulls the mean back. Sent as it is
  (decision 69).
- The Totals book-weight fit dropped its rows before 2023-05-03 until decision 71.
  `Archive.to_pandas` dropped rows dated before that day for every market except `Moneyline`
  and `"Total"`, and the archive names the market `Totals`: MLB lost 83,414 of its totals rows
  (15 %), the NHL 96,040 (26 %), the NBA 50,512 (25 %), the NFL 10,986 (14 %), the WNBA 4,552
  (7 %), all from 2022-05-08 on.
- Two MLB models know nothing of the postseason. `batter strikeouts` and `pitcher fantasy
  points underdog` (retrained 2026-10-12, ask 3) lack the six playoff features (`Playoff`,
  `SeriesWins`, `SeriesLosses`, `CanClinch`, `FacingElimination`, `Blowout`). Their matrices
  start on 2026-03-30 and the temporal split's training 70 % ends on 08-23 and 08-10, so no
  training row is a playoff game (the first is 09-29) and
  `pipeline._prune_uninformative_features` drops each column as constant. Matrices that
  start in 2025 keep them (`hits`: two values of `Playoff` in training). Serving builds the
  columns each model lists, so nothing fails; the two cells price a playoff game as a
  regular-season one until a retrain's training split holds a postseason.

## 8. Owner asks (one each)

The owner answered every ask from 1 to 23 with yes on 2026-10-11, the two retrains of 7 and 8
included ("retrain both now"), asks 24 to 31 with yes on 2026-10-15, and asks 32 to 35 here and 36
to 39 of the close-out report with yes on 2026-10-16. Each became a decision in §2 and a row in §4
(decisions 43 to 76). No ask is open. Not asked because no change is proposed: the parlay floor
(`_MODEL_EV_FINAL_FLOOR` = 2.0; no setting tested returns $1 per $1 while the legs are over-read).

The owner's own, no answer needed:

- **Read the nine other banned touchdown pairs in the app** (ten minutes; no automated
  probing). The one banned pair checked, a quarterback's pass TDs with his own receiver's TDs,
  is allowed at 0.794 (§6, I5). The nine other keys of that family, and every sub-1 value
  without a quote, are guesses until a payout read in the app backs them. From a payout, the
  pair's modifier is the payout over the table value over the product of each pick's shown
  price divided by 1.8708. In `banned_combos.json` a pair's first slot is the same side and
  its second the opposite; the read is recorded in [underdog_api.md](../underdog_api.md) §6.8.
- **Drop the season's Cleaning the Glass export on both boxes** when it is published
  (`player_data/NBA/nba_players_2026-27.csv`). Until then the NBA reads last season's file: ages
  and positions are a year old (decision 26).
