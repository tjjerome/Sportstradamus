# Lane brief — story balance: an honest ledger by bet side, Over-led and Under-led star stories

**Read first:** CLAUDE.md, [docs/ARCHITECTURE.md](../ARCHITECTURE.md),
[docs/story_voice.md](../story_voice.md), [docs/dashboard_ux_redesign.md](../dashboard_ux_redesign.md)
(Tonight and Games rows), [docs/ship_gate.md](../ship_gate.md) (Gate 2), and the research brief
[docs/archive/researcher_selection_shrink.md](../archive/researcher_selection_shrink.md).
Status: OPEN — every workstream is live on `devel`; the lead rule of §5 and the Tonight card without
its side arrow landed 2026-10-10; the dek about the lead's own leg, with no clause against its bet,
landed 2026-10-17 (`STORIES_VERSION` p6). The live read is §7; the measured after-numbers, the
options for what a lead rule cannot reach and the side-rule record are in §11.

## 1. Mission & money logic

The Games and Tonight tabs read as "this game will be exciting, but take the unders anyway". The
owner asked, in order: are the under-heavy recommendations actually +EV; if not, fix the models;
either way give Overs and Unders near-equal representation with the stars in front, in every
league, without giving up profitability.

The first question has an answer, and it is not the one the dashboard was showing. Priced at the
platforms' real payouts, **neither side of the recommended cohort is realized +EV**:

| recommended legs (Model EV ≥ 1.05) | n | hit | model read | book read | flat ROI |
|---|---|---|---|---|---|
| 90d Over (2026-07-04 → 10-02) | 5,656 | 0.470 | 0.663 | 0.483 | −14.8% |
| 90d Under | 11,237 | 0.504 | 0.642 | 0.492 | −8.8% |
| 30d Over | — | 0.443 | 0.609 | 0.483 | — |
| 30d Under | — | 0.495 | 0.609 | 0.492 | — |

Unconditional calibration is exact (mean model P(Over) 0.442, realized 0.442, book 0.445). The loss
is overconfidence conditional on selection: on the legs the rule picks, the sportsbook consensus
beats the model by 2–8 pp, and the hit rate falls as the claimed edge over the market grows. Legs
paying above 2.5× hit 0.15–0.35. The under skew itself is structural (DFS x.5 lines on low-count
stats, argmax-probability side selection, right-skewed distributions, Kelly ranking that favours
high-probability low-payout legs, Over stories needing unanimous Over legs), not a bias of the
distributions.

Four reporting bugs had hidden all of that from every nightly number the owner reads: Kelly went
positive on −EV legs whose decimal payout sat below 1; the profit sims priced Unders as `boost/(1 − p)`
although `Market Prob` is the chosen side's probability (MLB home runs showed a +615% Kelly yield);
CLV carried an extra Under sign flip (390k stored rows wrong-signed); Gate 2 demoted on Over
precision only while 52% of Under history rows were never posted on the platform yet counted. All
four are fixed (§4). The model lever the brief would have justified was researched and **killed** (§6).
So the lane's money logic is: stop lying to the owner, stop ranking −EV and deep-alt legs first, and
let the stories balance by construction at the unchanged 0.05 edge floor rather than by pretending
the Overs are better than they are.

## 2. Locked decisions (owner)

- Model scope: diagnostic + guardrails now; the selection lever research-gated (it was, and it died).
- Balance: Over leads fill toward half of each date's games (§5), at the unchanged
  `_MENU_EDGE_FLOOR` (0.05); the top-EV story stays in the menu, often behind the led ones (median
  cost ≈ 0.05 EV on the first slate; accepted). The rule chooses which story leads a game. It does
  not change the legs inside stories: that pool is the models', and no balanced leg mix is promised.
- Tonight shows the lead story's headline with no side mark. Where the two platforms lead a game on
  different sides, the card takes the higher-EV lead.
- Star: computed prominence from the snapshot alone (`Star` = rank-1 depth label + line percentile +
  market breadth, in [0, 3]); no new data; **no MLB depth term** (B1..B9 is the batting slot, not
  usage), so MLB stars top out at 2.0 on line percentile + breadth.
- Voice: the "live analyst" register (`docs/story_voice.md`), pinned by density tests, not strings.
- Gate 2 symmetric: either recommended side's live precision below 0.50 demotes (see §8; two
  mandatory NFL cells demote on the dev-box read).
- Orphan lead seeds pair with the top-edge strong leg from the other team (any other player in a
  one-sided game); the dek drops its correlation line below the ρ floor, so no false "move together".
- Both sides' full payouts persist to history (`Payout Over`, `Payout Under`) for the argmax-EV
  backtest the research brief specifies.

## 3. Verify before you trust

```bash
git log --oneline devel -24 | grep -E "stories|realized|nightly|clv|training|history|prediction|dashboard"
ls -la src/sportstradamus/data/runtime/ | grep -E "realized_by_side|current_game_stories|current_offers"
poetry run python - <<'EOF'
import pandas as pd
o = pd.read_parquet("src/sportstradamus/data/runtime/current_offers.parquet")
s = pd.read_parquet("src/sportstradamus/data/runtime/current_game_stories.parquet")
print("Star column:", "Star" in o.columns, "| lead columns:", {"lead", "lead_side", "lead_player", "star"} <= set(s.columns))
print("-EV legs with Kelly > 0:", int(((o["Model EV"] < 1) & (o["Kelly"] > 0)).sum()))
EOF
```

## 4. What landed (all on `devel`)

| commit | change | where it shows |
|---|---|---|
| `1d0d7ba9` | `Kelly` = `(Model EV − 1)/(Boost − 1)` only for `1 < payout ≤ MAX_FAVORED_PAYOUT` (2.5), else 0; `Model EV` untouched | Board Kelly sort, "favored" counts, constellation, satellite picker, story legs |
| `ad599623` | `Payout Over` / `Payout Under` on every history row (0 when the side is not posted; NaN before 2026-10-03) | `history.parquet` |
| `ed7a32b9` | `realized.py`: `compute_realized_by_side` — long ledger (window × cohort × split × key × side) at platform payouts, `Boost > 0` only | `data/runtime/realized_by_side.parquet` |
| `bd6548d5` | nightly prices every row as `boost / p` (chosen side), counts posted sides only in precision and both profit sims, writes the ledger + one INFO ROI line | `live_metrics_per_market.parquet`, `reflect` log |
| `82c0adb2` | CLV = close − open on bet-side probabilities, no Under flip; `fill_from_archive` recomputes every row with a close (repairs the 390k rows on the next `reflect`) | Receipts beat-close rate, pick'em CLV shrinkage |
| `a6d96fd8` | Gate 2 demotes on either side's precision < `MIN_PRECISION_SIDE` (0.50) | `lifecycle_table`, `gate-status`, Lab › Training |
| `cc2f8942` | Receipts panel "Realized by side — platform payouts" (`components/by_side.py`) | Receipts, after Skeptic checks |
| `6a4b7dc9`, `c9c303a8` | `stories/lead.py`: `attach_prominence` (`Star`), `lead_seeds`, `assign_leads`; `prophecize` scores and persists `Star` | `current_offers.Star`, story `lead*` columns |
| `d37ef588`, `db6e834c`, `0806d11a` | engine takes a `lead`; the menu grows an Over-led and an Under-led story per game (`Kelly > 0` and edge ≥ 0.05 required of every leg, lead in both presets, led stories first, slate-wide headline dedup); pricing helpers moved to `stories/pricing.py` | `current_game_stories` |
| `a3ca2684`, `c12b90af`, `624a45a8`, `d35a8aa3` | `game_headline` prefers the lead story; Tonight sorts `(urgent, −star, −favored, minutes)` and shows the headline with no side mark; Games opens on the lead story with "Over-led · " / "Under-led · " labels | Tonight, Games |
| `0212ca65` | `lead._pick_sides`: Over leads fill toward half of each date's games (§5) | story `lead` flags, Tonight, Games |
| `31a70732` | why/dek clauses read the defense (`gives`/`takes`) and the bet (`above_for` … `below_against`); `bank_cell` resolves `mistakes` before any `production` fallback | offer Why text, story deks |
| `6fc2dbd4` … `c1354dfe`, `0450274f` | all five voice banks rewritten to the live-analyst register; `STORIES_VERSION` p5 (every persisted headline reshuffles once; `user_slips` headlines stay frozen); density pins in `test_bank_coverage.py` | every headline |
| `eea99c05` | `why.story_dek` anchors a led story's dek on the lead's own leg (an unled story keeps the highest-conviction leg) | story deks |
| `6792d763` | the dek says a form or matchup read only when it backs the lead's bet; the two `dek.form` against branches leave the bank; `STORIES_VERSION` p6 | story deks |

Untouched on purpose: `strategies/profit_sim.py`, `strategies/_ledger_*.py` (sim-bettor lane),
`analysis._add_kelly_columns` and the flat −110 grading in Receipts — both since fixed by the
honest-receipts lane ([honest-receipts.md](honest-receipts.md): Kelly at the platform payout, the
`record_grid` and flat hero retired).

## 5. How the balance works

`attach_prominence` scores every offer row once per `prophecize`. In the menu, `_strong_legs` keeps
legs with `Model EV − 1 ≥ 0.05` **and** `Kelly > 0` (so no −EV leg and no payout above 2.5× can seed
or join a story). `lead_seeds` picks, per side, the strong leg with the highest `(Star, Model EV)`.
The Over-led cluster grows first, then the Under-led one, then the old greedy clusters; a lead whose
cluster prices nothing borrows one leg and is priced once more. `_best_subsets` only enumerates
subsets containing the lead, so Builder and Moon stay true argmaxes with the lead inside. Led stories
rank first. `assign_leads` then gives each game one side for both platforms (`_pick_sides`). A game
led on one side keeps it. Games led on both take Over until the date's Over leads reach half of its
games, and the rest take Under. Half counts every game the date has a story for, on either platform,
and rounds up; an Over-only game counts toward it. The games with the brightest Over-led story
(`star`) take Over first, ties by tip, then by game key. A platform with no story led on the game's
side shows its other one, so both platforms show the same side whenever both offer it. Tonight's
headline and the Games default story read that `lead` flag.

The dek (the line under a story's headline) is about the lead's own leg: `why.story_dek` takes the
lead player the menu routes the headline to and anchors the form, matchup and lineup clauses on
that leg (`_dek_anchor`); an unled story keeps the core's highest-`Win Prob` leg. A dek says a form
or matchup read only when it backs the bet (`_anchor_clauses` skips the `*_against` form branches
and a matchup opposing the side), so a led story never argues against its own bet; the per-offer
Why keeps both reads. Before this, the anchor was the highest-`Win Prob` leg of the core, so on the
10-15 board 28 of 36 Over-led stories carried a dek about an Under leg.

## 6. Research verdict: the selection-shrink lever is KILLED

Question put to `research-analyst`: should the served probability of a recommended leg be shrunk
toward the sportsbook consensus by a factor fitted on the realized selected subset (grain, estimator,
placement), against the status quo, a higher floor, `kelly_shrinkage` as λ, unconditional book CITL,
and argmax-EV side selection? Walk-forward on fit < 2026-09-01, eval ≥ 2026-09-01, cohort re-derived
after the shrink. The full brief is archived at
[docs/archive/researcher_selection_shrink.md](../archive/researcher_selection_shrink.md).

- **Every option fails.** Best shrink +1.7 pp ROI (per-cell λ, day-clustered CI [−0.3, +3.4]) against
  a +3 pp bar; Over hit among survivors ≤ 0.475 against 0.50. A hindsight oracle fitted on the eval
  window itself passes nothing.
- **Nothing to tune.** On selected legs the served `Win Prob` carries no information beyond the
  market: λ_sel = −0.10 ± 0.10, logistic slope 0.01 ± 0.15; hit rates track the book in every
  disagreement bin. λ = 0 is "replace the model with the market".
- **64% of recommended legs have no sportsbook quote** (`Market Prob` is the payout-implied fill), so
  a book shrink cannot reach most of the overconfidence. Those legs hit 0.466 / 0.514 against 0.610.
- A **higher floor hurts** at every level: claimed edge is anti-correlated with outcome because a big
  edge is a big payout, which is the side the platform priced as unlikely.
- **MLB is selection on noise** (calibrated, selected hit == book). **NFL is selection plus a model
  bias** (Unders +3.6 pp overconfident on 10,004 legs, λ_all −0.14 ± 0.03): a training problem for the
  NFL lanes, not a serving patch. The 2026-10-16 read confirmed and located it: the training
  matrix's outcome-band trim deletes stars' big games on unquoted rows, so the model's mean on the
  top quarter of lines runs low wherever no sportsbook quote blends it out;
  [nfl-ship15-recovery.md](nfl-ship15-recovery.md) §11 holds the tables, the acceptance test and the
  proposal.
- **Payout cap 2.0** is the most consistent single lever (+2.3 pp, CI [+0.3, +3.8]) but its Over hit
  0.498 fails the bar. Its trigger is pre-registered below; the cap stays at 2.5.

If the lever is ever reopened (about 2026-11-01, one book regime, NBA in sample): post-argmax only,
in `finalize_records` after the confidence clip, on real-quote rows,
`Win Prob' = Market Prob + λ_L · (Win Prob − Market Prob)` with λ_L per league keyed on
`Model Version`, through-origin least squares on trailing recommended quoted legs, at least 1,000
legs and a CI lower bound above 0, else λ = 0, never 1.

## 7. Live read

Runtime parquets only; nothing here touches DuckDB. Run it on the dev box after a `sync` from
prod. It reruns as is after any `prophecize` and `reflect`.

```bash
poetry run python - <<'EOF'
import pandas as pd
from sportstradamus.helpers import platform_payout
R = "src/sportstradamus/data/runtime/"
o = pd.read_parquet(R + "current_offers.parquet")
s = pd.read_parquet(R + "current_game_stories.parquet")
led = s[s["lead"] & (s["objective"] == "moon")]
leads = led.groupby(["Date", "platform"])["lead_side"].agg(over=lambda side: int((side == "Over").sum()), menus="size").unstack("platform")
leads.insert(0, "games", s.groupby("Date")["Game"].nunique())
print(leads.to_string())
legs = pd.DataFrame([dict(l, story=r.story_id) for r in s.itertuples() for l in r.legs])
edge = legs["win_prob"] * platform_payout(legs["boost"], legs["platform"]) - 1
print("story legs below the floor or with Kelly <= 0:", int(((edge < 0.05) | (legs["kelly"] <= 0)).sum()), "(want 0)")
print("-EV offers with Kelly > 0:", int(((o["Model EV"] < 1) & (o["Kelly"] > 0)).sum()), "| payout > 2.5 with Kelly > 0:",
      int(((platform_payout(o["Boost"], o["Platform"]) > 2.5) & (o["Kelly"] > 0)).sum()), "(want 0, 0)")
r = pd.read_parquet(R + "realized_by_side.parquet")
print(r[(r["window_days"] == 30) & (r["cohort"] == "recommended") & (r["split"] == "side")][["side", "n", "hit_rate", "pred_rate", "book_rate", "roi"]].to_string(index=False))
h = pd.read_parquet(R + "history.parquet")
u = h[(h["Bet"] == "Under") & h["Close Market Prob"].notna()]
print("Under rows whose Market CLV == close − open:", f"{((u['Market CLV'] - (u['Close Market Prob'] - u['Market Prob'])).abs() < 1e-9).mean():.3f}", "(want 1.000 after the first reflect)")
EOF
```

The first table is each date's games beside each platform's Over leads (`over`) and story menus
(`menus`). The rule fills Over toward half of a date's games, counted across both platforms, so a
platform's `over` reaches half only where the pool allows: a date falls short when too few of its
games have an Over-led story at all, and a platform falls short when it has no Over-led story for a
game the rule gave to Over. On the slate of 2026-10-08 the rule gave Over 14 of 29 games (by date 6
of 12, 1 of 2, 7 of 14, 0 of 1); Underdog led Over in 11 of 29 menus and Sleeper in 11 of 26, up
from 8 and 10 under the strict alternation it replaced. The other lines carry their targets in
parentheses. Story legs carry lowercase keys (`win_prob`, `boost`, `kelly`), so the edge is
rebuilt from the payout. The measured boards after the rule are in §11.

Then eyeball Tonight: star games above non-star games within a tier, and the Games tab opening on
the led story.

## 8. Behaviour changes the lane made (2026-10-03)

- **Gate 2 demotes cells on Under precision.** On the dev-box history of the landing the
  symmetric rule demoted five cells that passed on Over precision: NFL `rushing yards` (Under precision 0.39,
  Over has too few bets to read), NFL `receiving yards` (0.46), WNBA `PR` (0.43), `RA` (0.44),
  `PA` (0.47). Two of those are mandatory cells of
  [nfl-ship15-recovery.md](nfl-ship15-recovery.md). Demotion changes `lifecycle_state`, the monthly
  `gate-status` PR and Lab › Training; it does not stop serving (serving is pickle-exists). To restore
  the old asymmetric rule, revert `a6d96fd8`; the precision inputs stay posted-only either way.
- **Board Kelly reads 0** on every leg paying ≤ 1× or > 2.5× (61 MLB Underdog Unders on the first
  slate were −EV with Kelly 3.5–40 and ranked first everywhere). Tooltip updated.
- **Every persisted headline reshuffled once** (`STORIES_VERSION` p5); `user_slips.parquet` keeps its
  frozen text. Every persisted dek reshuffled once more at p6 (2026-10-17), when the dek moved to
  the lead's leg and dropped the clauses against its bet.
- **CLV flipped sign on every Under segment** at the first `reflect` after the landing (NFL Under
  beat-close ≈ 68%, not 31%); pick'em CLV shrinkage reads the bet-side value.
- **Live metrics moved** on the dev-box history of the landing: MLB home runs' Kelly yield left the
  hundreds of percent; flat and Kelly yields share a sign on cells with n ≥ 100;
  `precision_under_live` dropped where unposted Unders had propped it up.
- `voice_bank.json` lives under `data/config/`, so the production pull restarts the dashboard by
  itself (`run_job.sh`); nothing manual.

## 9. Records, not tasks (left open on purpose)

Monitoring the research brief asked for; nothing is wired, nothing is yours to do now:

- A `quote` split (quoted / unquoted / fallback) in `realized.by_split`, so `book_rate` stops
  pooling sportsbook and payout-implied anchors (that pooling is why the 90d "book" read 0.586
  against 0.504 realized).
- The selection residual per league × side: `hit_rate − book_rate` and a trailing-30d λ_sel with its
  day-clustered CI. Reopen the shrink only when the CI lower bound clears 0.
- Gate 2 cannot see this failure (its precision uses all posted `Bet` rows, ≈ 0.59 / 0.64); the
  realized ledger is the only monitor for it.
- Pre-registered trigger for lowering `MAX_FAVORED_PAYOUT` to 2.0: two consecutive monthly 30d reads
  in which the recommended 2.0–2.5 band's ROI sits ≥ 5 pp below the 1.5–2.0 band on **both** sides
  (today's 30d gap: 6.6 pp Over, 9.0 pp Under; 90d: 5.1 pp Over, 0.2 pp Under). Owner packet, not a
  session flip.

Open questions the data could not settle:

- Honest, market-anchored probabilities recommend about 1% of today's legs. Should the `recommended`
  label and the 0.05 menu floor keep firing on unproven claimed edge? Product call.
- Unquoted legs (64% of the cohort) get full trust (λ = 1); requiring a quote makes ROI worse (−1.6 pp)
  because quoted disagreements are the most adversely selected. Unresolved, and the larger lever by
  count.
- Is a sportsbook alt-line quote at the rung in the archive? It would beat the shape decode as
  `Market Prob` for rung payouts (a WS-4 question).
- Star weights: NBA/WNBA depth labels rank minutes within team × position (A'ja Wilson posts as F3);
  line percentile + breadth carry her anyway. If a true star loses a seed to a role player on the
  first slates, re-weight `depth` in `stories/lead.py` (owner call; not pre-built).
- `legs._stat_category` maps the "Goals Against" display name to `scoring` while the `goalsAgainst`
  slug reads `mistakes`; the leg's `stat` carries the slug, so the slug path fires. Worth a glance
  because mistakes cells are the valence-sensitive ones.
- Re-run the walk-forward around 2026-11-01 within one book regime and with NBA in sample; NFL had no
  fit-window rows in the prescribed split.

## 10. Owner asks (one each)

1. Decide on the Gate-2 demotions in §8 (keep the symmetric rule, or revert `a6d96fd8`).
2. The no-lead Tonight card (§11, option A): yes or no.
3. The lower story floor (§11, option B): no is the recommendation; a yes names 0.03 or 0.0 and
   whether the Receipts "recommended" floor moves with it.
4. The star-lean lever: [nfl-ship15-recovery.md](nfl-ship15-recovery.md) §11 names the lever, the
   cells, the retrain and the date; one yes.

## 11. After the rule: measured boards, what a lead rule cannot reach, the side-rule record

Scripts and outputs: `~/backups/sportstradamus/2026-10-16-under-lean/part1b/` (boards) and
`part3/` (side rule), each with a `REPORT.md` whose numbers are tagged measured or inferred.
The 10-08 board is the last one before the lead rule; 10-15 and 10-16 are production boards after
it. "Menu" is one (platform, date, game) with at least one story; "legs" counts each story once
(the union of its two presets).

| | 10-08 (before) | 10-15 | 10-16 |
|---|---|---|---|
| games with a story | 29 | 29 | 23 |
| Underdog: Over-led headlines / menus | 8 / 29 | 12 / 24 | 10 / 18 |
| Sleeper: Over-led headlines / menus | 10 / 26 | 12 / 26 | 11 / 20 |
| Over-led headline share | 18 of 55 (33 %) | 24 of 50 (48 %) | 21 of 38 (55 %) |
| Tonight cards quoting an Over-led story | 8 of 29 | 15 of 29 | 13 of 23 |
| games with no Over-led story on either platform | 10 | 7 | 3 |
| legs inside stories, Over / Under | 64 / 195 (25 %) | 93 / 236 (28 %) | 78 / 146 (35 %) |
| headline is also the menu's top-EV story | 42 of 55 | 35 of 50 | 28 of 38 |
| EV given up by the headline: median / mean / max | 0 / 0.08 / 1.41 | 0 / 0.16 / 1.31 | 0 / 0.08 / 0.60 |

The saved lead flags on both after boards equal a fresh `assign_leads` run, and the rule meets its
half target on every date but one (10-16's single game has no Over-led story). The rule alone,
replayed on the 10-08 stories, lifts Over-led headlines to 11 of 29 and 11 of 26 and the Over-quoting
Tonight cards to 11 of 29; the rest of the move between the columns is the slate.

Three gaps a lead rule cannot reach (counts on the three boards): games with no Over-led story on
either platform, 10 / 7 / 3 (13 NHL, 5 NFL, 2 NBA across the three); a platform lacking the Over-led
story in a game the rule gave to Over, 4 / 1 / 4 menus; and, because of that, a Tonight card
quoting an Under-led story in a game picked for Over, 3 / 1 / 0 cards (the card takes the higher-EV
lead by the §2 decision; `game_headline` in `narrative.py` could prefer the rule's side instead).

**Option A, a no-lead Tonight card (display only).** When no story row of a game has
`lead_side == "Over"`, the card shows the game's shape as its headline and quotes no bet. Count:
10 of 29 cards on 10-08, 7 of 31 on 10-15 (6 NHL, 1 NBA), 3 of 31 on 10-16. Effect: cards quoting
an Under-led story fall from 14 to 7 of 29 (10-15) and from 10 to 7 of 23 (10-16). Cost: none in
money; a game loses its bet quote on the card (the Games tab still opens on its led story). Recipe:
`dashboard/surfaces/tonight.py` (the headline branch near line 164) renders the shape line when the
game has no Over-led story, with its own bank line (the storyless "no story binds them" reads wrong
for a game that has stories). One yes.

**Option B, a lower story floor (`_MENU_EDGE_FLOOR`, `menu.py` line 53).** Replayed offline with the
floor patched (73 of 76 stories rebuild exactly on 10-16; the older boards half, since the code
moved): at 0.03 the Over-less games gain an Over-led story in 5 of 10 (10-08), 2 of 7 (10-15) and 0
of 3 (10-16); Tonight cards quoting an Over-led story go 11 → 13, 14 → 15, 13 → 14 at 0.03 and
17, 16, 14 at 0.0. On 10-16 at 0.03 the count of Over-less games rises from 3 to 5, because the
floor admits whole new games led only Under. Most of the change is not balance: stories change in
about two-thirds of menus at 0.03 and nearly all at 0.0; leads rank by prominence first, so stars'
sub-floor legs take over leads (5 to 11 of about 25 Over-led headlines at 0.03, 9 to 20 at 0.0);
the admitted legs run about two Unders per Over. Cost on production history (09-16 to 10-15, each
prop once, Kelly > 0, day-block CIs): the 0.03 to 0.05 band returned −8.7 % per pick [−13.0, −3.9]
on 1,545 legs against −9.0 % [−12.8, −5.5] for today's ≥ 0.05 band on 4,648; its Overs alone
−7.0 % [−17.8, +4.8] on 371. No band makes money, so the floor is not a money lever either way.
Recipe: line 53 from 0.05 to 0.03 (or 0.0); two golden pins move with it
(`test_story_menu.py::test_edge_floor_value_is_owner_locked`,
`test_realized_by_side.py::test_recommended_edge_matches_the_story_menu_floor`), and
`RECOMMENDED_EDGE_MIN` in `realized.py` must equal the floor, so the Receipts "recommended" cohort
moves with it. Recommendation: no.

**Record, not a task: `offer_index` keys.** `legs.offer_index` keys on (player, side, line) with no
market. Lead keys matching two or more markets: 27 / 22 / 14 on the three boards, with the lead's
own market first in 21 / 22 / 13; every other-market case is a Sleeper key finding an Underdog row
of another market, and on 10-16 the one case is not a headline. Nothing to do now.

**Record, not a task: the side rule.** Showing the better-paying side (the argmax-EV side) instead
of the likelier one was replayed on production history from 10-05 (both sides' payouts persist
from then), each prop once, through `realized.settled_offers`. It never flips a leg the board
already recommends: a flip would need both sides to clear a 5 % edge, and the platforms' margin
(1/pay_over + 1/pay_under ≥ 0.9604 on every two-sided row) rules that out. It only adds legs: 186
on 11 game days, which hit .349 at a mean payout of 2.34 and returned −18.8 % per pick
[−30.0, −3.8]; the recommended cohort went from −5.4 % to −7.1 % per pick (−1.8 [−3.9, −0.1]) and
its Over share from 18 % to 24 %. On every two-sided row the EV side hit .315 where the model said
.370 and the book .320. Preliminary: 11 game days (NFL 4, WNBA 2); the fair read is 2026-11-04 with
the same scripts (`part3/REPORT.md` holds the one-line re-run). Nothing to do now.
