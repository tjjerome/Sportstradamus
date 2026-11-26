# Lane brief — profit engine: one EV fitted where we bet, plays scored in money

> Status: ACTIVE. Built: the Modifiers page fix (step 0), the money table (phase 0, offline and
> nightly) and EV with its scoreboard (phase 1: the stamp on every scored record, the nightly
> weights table, the fit's two plays, the replay's fit check). Next: the engine reads EV (phase 2,
> §5.3); until then ranking and stakes read the model's claim. The model lane (§7) is open.

## 1. Mission & money logic

The board's recommendations lose money, and model work that lowered error at posted lines has
not changed that. This lane attacks the loss where it is made: in what the engine recommends and
what the apps pay for it. Three moves, in order. Score every pick rule in money (the money table,
§5.2). Let one number fitted on settled posted picks, "EV", lead ranking, the "+EV" mark and
stakes (§5.1, §5.3). Aim model work at the information the app's price lacks, measured by the
weights that fit reports (§7).

The record covers games from 2026-10-09 to 2026-11-24, each prop counted once
(`realized.settled_offers` over production's history). `sportstradamus admin play-replay`
reproduces the first five rows.

| Finding | Number |
|---|---|
| Recommended legs | 9,386 legs; the board claims a .601 win chance, they hit .493; return −8.8 % a leg |
| By pick kind | Underdog even picks −5.2 % (2,734 legs); raised payout −6.9 % (2,342); cut payout −10.4 % (604); Sleeper −12.3 % (3,706) |
| Even picks by league | NBA .497 on 1,433; NFL .527 on 526; NHL .551 on 741 (+0.8 %). Break-even is .546 a pick, about .536 inside a 2- to 4-pick Power entry |
| Every side the apps posted, taken blind | Underdog even picks −8.5 % a leg; cut payout −11.8 %; raised payout −19.1 %; Sleeper −12.0 % |
| The engine's own same-game entries | 7,799 Underdog 2- and 3-pick entries, 7,664 of them NFL; claimed value 2.22 for 1 staked; all picks hit 9.0 % where 12.0 % was needed (the mean of each entry's own break-even). The search keeps only entries claiming 2.0 or more, which selects the largest compounded errors |
| Rules that need no model | Following Underdog's own line moves .493 on 4,829; each market's prior-month winning side .51 to .52; every Under on an even pick .515 on 17,258 |

**Where information beyond the app's price comes from.** On settled posted picks, how much weight
does the data put on a source's disagreement with the app's own price (0: the app already knows
it; 1: trust the source fully)? Legs fall into three classes by what the scoring run could see
of the sportsbooks, in quotes at most 90 minutes old. Standard errors in brackets, clustered by
game date. The shares and the class-A book weight are Underdog's; the other weights pool both
apps, and the model's are read on rows a model served
([archive/researcher_ev_fit.md](../archive/researcher_ev_fit.md), E2, E9 and E12).

| Class (share of legs NBA / NFL / NHL) | Sportsbooks | Our model |
|---|---|---|
| A: a sportsbook quotes the app's exact line (28 % / 18 % / 40 %) | The plain mean of the books' own prices: NBA 0.53 (.23), NFL 1.24 (.47), NHL 0.92 (.23) | Beside the raw book it adds nothing: NBA 0.10 (.15), NFL −0.02 (.17), NHL −0.01 (.13) |
| B: a sportsbook quotes another line (21 % / 36 % / 14 %) | Our translation of that quote to the app's line: NBA 0.07 (.10), NFL −0.04 (.11), NHL −0.16 (.33). The information is lost in translation | NBA 0.01 (.10), NFL 0.03 (.07), NHL 0.59 (.27) |
| C: no sportsbook quotes the prop (52 % / 46 % / 47 %) | none | NBA −0.05 (.07), NFL 0.14 (.10), NHL 0.48 (.08) |

Three things follow. Read raw at the same line, the sportsbooks beat the app's price, and the
engine does not use them that way: its `Market Prob` is the consensus turned into a mean and
re-priced through our curve, from a quote a median 79 minutes old (§7, M1). The edge the books
offer is thin, because the app tracks them closely: fitted walk-forward, the books and the model
together beat the app's price in log loss, and still no set of legs credibly returns 1 or more.
The model's own information lives where the books are silent or quote another line, NHL first.
Those weights, not the error at the line, are what the model lane must move.

## 2. Read first (in order)

1. `CLAUDE.md`, "Judging model changes": the priced-population rule, the floor under this lane's
   keep rule (§5.2).
2. [`ship_gate.md`](../ship_gate.md), "The realized replay" and the devel → main table.
3. [`underdog_api.md`](../underdog_api.md) §6.8: how an entry is priced (table × multipliers × m).
4. `src/sportstradamus/realized.py` and `src/sportstradamus/plays.py`: the per-offer frame and
   the money table built on it.
5. [`gbdt_feature_study.md`](../gbdt_feature_study.md) §15: earlier verdicts this lane re-reads.

## 3. Verify before you trust

Run `sportstradamus admin play-replay --history <frozen history> --entries <frozen parlay_hist>
--since 2026-10-09 --until 2026-11-24` on the frozen copies in the lane directory (§8). A correct
build prints:

- Play `model_edge`, all leagues and kinds: 9,386 legs, claim .601, hit .493, return −8.8 %.
  Underdog even picks 2,734 at .518: NBA 1,433 at .497, NFL 526 at .527, NHL 741 at .551.
  Sleeper 3,706 at .484.
- Play `all` on Underdog even picks: 2-, 3- and 4-pick Power entries return −12.6 %, −18.9 % and
  −25.1 %. Both sides of an even pick return exactly −12.5 %, −18.75 % and −25 %; the gap is the
  offers with one side posted.
- The engine's entries: Underdog 2-pick 200 and 3-pick 7,599 (7,799 together, 135 of them NBA).

Add `--walk-forward` and point `--history` at `history_stamped.parquet` (the same history with
its book stamps rebuilt): the replay prices each of the last 33 game dates from fits on earlier
dates only and prints the fit check. A correct build prints:

- The fit's log-loss gain over the app's price, × 10,000, every leg pooled: +5.7 [+2.0, +9.3] on
  122,627 legs. NBA modified-payout picks: +10.2 on 36,866.
- What the model's side paid, by band of EV: says .871 and paid .887 on 78,213 legs; .915 and
  .916 on 42,030; .948 and .935 on 2,166; .980 and .982 on 173; at 1 or more, says 1.027 and
  paid .803 on 45.
- "+EV" marks: 0 of 139,279 legs. `ev_rank` minus `model_edge`: +0.028 a leg [−0.003, +0.060],
  on 7,115 legs each.

### Volatile product assumptions

- History carries both sides' payouts (`Payout Over`, `Payout Under`) only from 2026-10-05, so
  nothing before it has an unchosen side to score.
- The payout tables (`underdog_payouts.json`, `sleeper_payouts.json`) and the per-pick baseline
  (`UNDERDOG_BOOST_BASELINE`) are the apps' to change.
- The cut Underdog takes on a same-game entry (its `m`) shows only in the app. Until the owner
  reads it, a stack's worth is a "bet if the app pays at least X" number, never a price.

## 4. Locked decisions (owner)

1. Every pick kind and league stays on the board. The aim is to fix what is recommended, not to
   remove it; modified-payout picks are growing and cannot be ignored.
2. `devel` is the lab: nothing is turned off there. `main` may be gated on a live money test.
3. The fitted number leads: ranking, the "+EV" mark and stake size follow it, and the dashboard
   calls it "EV". The model's claim stays visible as "Model EV". Today's 5 %-edge rule keeps
   running as a tracked paper play (`model_edge`).
4. Main's money test: a league's recommended even picks over the trailing 90 days, at least 300
   legs and a return at or above zero; it drops off `main` again below −3 %.
5. The owner reads real payouts in the app. No scripted quotes on the owner's account.
6. The engine is patched first, and model work keeps going: it resumes as soon as the engine can
   measure it, aimed at where the model can add information (§7).

`stat_meta.json` stays the owner's to commit.

## 5. Design

### 5.1 EV: one fit on the picks we bet

For each posted offer, the chance behind "EV" is fitted on settled posted picks, as the chance
that the Over hits:

    logit(P) = c + a * logit(app's fair price)
                 + w_book  * (logit(book price)  - logit(app's fair price))
                 + w_model * (logit(model's P)   - logit(app's fair price))    a model served
                 + w_fb    * (logit(model's P)   - logit(app's fair price))    a withheld cell

- **App's fair price**: the posted pair of payouts with the cut removed; one half on an even pick.
- **Book price**: the plain mean of each sportsbook's own latest price at the app's exact line,
  at most 90 minutes old, as the scoring run saw it, read off the archive's `ladder` table
  (which holds every same-line quote the `odds` table holds). Present only in class A; a
  translated price joins class B when the model lane earns it weight (§7, M2).
- **Model's P**: the served `Win Prob`. A withheld cell serves the book consensus under that
  name, so its rows carry their own weight `w_fb` and `w_model` reads served models only.
- One fit per league, pick kind (Underdog even, Underdog modified payout, Sleeper) and class,
  refit nightly on the trailing 90 days. A prior worth 4,000 legs pulls every coefficient toward
  the no-evidence point (c = 0, a = 1, every w = 0), which is the app's fair price. Each
  coefficient's prior is scaled by the legs that carry its term: scaled by every leg, a term
  few legs carry (`w_fb` in a league with few withheld cells) is held too loosely and overfits
  them. Below 400 settled legs a fit is not used and the leg takes that no-evidence value:
  zero edge, on purpose. An offer with one side posted has no fair price and gets no EV: a third
  of history's rows, nearly all alternate rungs, and on one in five of them (7 % of all rows,
  mostly Sleeper's) the posted side is the model's.
- The side a leg takes stays the model's. The fit values that side; it does not choose it.
- `EV` is the fitted win chance times the posted payout, and it ranks legs. The "+EV" mark and
  the stake read the fit's one-sided 95 % lower bound instead: a leg is marked when the bound
  times the payout reaches 1, and it is staked at half Kelly on the bound. `Win Prob` and
  `Model EV` keep their meaning.

A scoring run stamps seven columns on each record (`prediction/ev_stamp.stamp_records`):
`Book Prob`, `Book Count`, `Book Class` and `Book Observed At`, the books' price as that run
read the archive (`book_stamp.stamp`), then `Fit Prob`, `Fit Prob Low` and `EV`
(`live_fit.score` against the nightly table). They ride to the board snapshot and to history.
`reflect` refits the table every night (`data/runtime/live_fit.parquet`), and for history rows
scored before the stamp existed it fills the four Book columns from the archive as their run
could read it (`book_stamp.fill_history`), so the fit's window is full from its first night.

The prior, the lower bound and the side rule come from the research brief
([archive/researcher_ev_fit.md](../archive/researcher_ev_fit.md)), walk-forward over the last 33
game dates. Without the prior the fit loses to the app's own price out of sample, and in a
simulated world that holds no edge it still marks about a thousand legs "+EV". With the prior it
beats the app's price in log loss and is no worse in any league and kind, yet the legs it values
at 1 or more paid 0.79 to 0.90 on fewer than a hundred legs; the lower bound marks none of them.
Where the fit's better side differs from the model's, taking it returned 2.7 points less a leg.
So marks and stakes stay rare until the evidence grows, and the board is carried by rank. The
brief's six acceptance tests and five kill lines (its §3) are this fit's own.

The fit does not double count the model blend. The served number already passes two trust
stages, the blend of model and book and the `book_pull` knob; the fit sits after both and is
estimated on their output, so it measures only what is left to correct. The end state is one
owner of "how much app, book and model": this fit. The blend keeps its other job, the shape of
the distribution. `book_pull` retires in M1.

### 5.2 Plays and the money table

A **play** is a named rule that picks sides from a day's board (`plays.PLAYS`). Registered:
`model_edge` (the board's rule, `realized.recommended`, so Receipts, the live gap and the replay
keep their cohort), two controls that need no model, `under` (every Under on an even pick) and
`all` (every posted side), and the fit's two: `ev_edge` (the "+EV" mark of §5.1 in the same
payout band) and `ev_rank` (per date, league and kind, as many legs as `model_edge` took, the
highest by EV). A new play registers last, so the plays before it keep their bounds.

`plays.play_record` scores each play per league and pick kind:

- legs, hit rate, break-even and return a leg, with intervals that resample whole game dates,
  and an anytime-valid bound on the hit rate's excess over break-even, since the table is read
  every night;
- the exact return of every one-leg-per-game Power entry of 2, 3 and 4 picks the play's legs
  could have built that day, at the real payout table and each pick's own multiplier. Nothing is
  sampled: per game take the count of legs and the summed multipliers of the legs that hit, and
  combine games with elementary symmetric polynomials. A league's figure weights each date by
  its legs, so entries are staked as the legs are.

`plays.entry_record` scores the engine's own graded entries the same way. An entry with a pushed
or ungraded leg is left out until phase 2 fills `Profit` (about one in ten today).

The table is built in two places. Nightly, `reflect` writes `data/runtime/play_record.parquet`
over the trailing 90 days (`plays.trailing_record`; games before props counted once, 2026-10-09,
stay out): the window main's money test reads (§5.5). On the dev box, `sportstradamus admin
play-replay` scores any window of a history, this box's or a frozen copy.

The **information weights** of §1 are the fit's own table, `data/runtime/live_fit.parquet`: one
row per league, kind and class with its coefficients, their standard errors and its legs. They
are the model lane's scoreboard. `play-replay --walk-forward` prices each game date of a stamped
history from fits on earlier dates and prints the fit's checks: its log-loss gain over the app's
price, what each band of EV paid, the count of "+EV" marks and `ev_rank` against `model_edge`
(`live_fit.fit_gain`, `live_fit.ev_bands`; §3 holds the known answers).

**Keep rule.** The priced-population rule stays as the floor: a change may not make error at the
posted line worse. Above that floor, a change of any kind (rule, model, feature, data source) is
kept only if, on later dates, its information weight on the rows it targets rises and its play's
return does not fall. Both are read against the control on the same dates, the weight as a
paired difference on the same legs: that sees a change about half the size two separate reads
can (0.31 against 0.56 on NBA's class B today).

### 5.3 The engine reads EV

- The leg filters that read the claimed number are redirected or deleted: the story menu's edge
  floor, the entry search's claimed-value floors, the leg-pool floors and the pick'em filter's
  thresholds.
- Selection becomes rank-based, so no league or kind goes empty: per game the top legs by EV,
  entries ranked by entry EV, the best few kept per game and per slate whether or not they clear
  1. At or above 1 an entry gets the "+EV" mark; the rest read as "best available".
- Cross-game entries reach the board through a `current_entries` snapshot, built only at the
  sizes with the lowest break-even (2-, 3- and 4-pick Power, 4-pick Flex).
- The paper ledger moves to `policy_v5`: personas price with the fitted win chance, and one
  persona per registered play is appended.
- The dashboard shows "EV", "Model EV", the mark and the play's own record beside each leg and
  entry. [DESIGN.md](../../DESIGN.md) governs the UI work.
- A rung with one side posted carries no EV (§5.1), so a rank on EV alone would never list the
  7 % of rows whose only posted side is the model's. What ranks them is decided here.

### 5.4 Stacks: "bet if the app pays at least X"

Underdog pays a same-game entry `table × multipliers × m`, where `m` comes from its own estimate
of how often the picks hit together. An entry is worth a bet only when the app's `m` is lighter
than the truth. We can measure the truth; the app's `m` shows only in the app.

- A census per league and structure (one anchor pick plus one to three linked picks): the share
  of settled posted even main lines on which every pick hit, each date priced only from earlier
  dates, with the pay-at-least number (1 / share) and a cautious one (1 / the interval's low end).
- The board lists tonight's candidates with their pay-at-least number and a field for the app's
  payout; the owner reads the slip in the app and types the payout into Lab › Modifiers.
- A quote of three or more picks is stored whole, per structure. Spreading it evenly over pairs
  is wrong: the app's cut is not a product of pair cuts. Until phase 3 builds that store, only
  a 2-pick read is saved through Lab › Modifiers (`helpers/parlay_modifiers.reconcile` spreads a
  larger slip's cut over its pairs or calls it ambiguous); a read of three or more picks goes in
  the lane's `STATE.md`.
- Serving changes only after quotes show a gap.

### 5.5 Main's money test

`training/graduation.classify_lifecycle` gains one league-level condition, read from the nightly
money table: play `model_edge`, Underdog even picks, trailing 90 days, at least 300 legs, return
at or above zero; a cell already on `main` holds until the return falls below −3 %. The live
Gate 2 stays. `devel` is never gated by it.

## 6. Stage plan

| Stage | What | State |
|---|---|---|
| Step 0 | Lab › Modifiers loads with a cut-payout leg, so the owner's reads (§5.4) can start | Built |
| Phase 0 | The money table, offline: `plays.py`, `sportstradamus admin play-replay` | Built |
| Phase 1 | EV and the scoreboard: `live_fit.py` (fit, score, walk-forward, the fit's checks); the EV stamp on each scored record; nightly writes `live_fit.parquet` and `play_record.parquet`; the plays `ev_edge` and `ev_rank`; `play-replay --walk-forward`; history records only a board a bettor could act on (§7, M1). The fit follows the research brief ([archive/researcher_ev_fit.md](../archive/researcher_ev_fit.md)) and passes its acceptance tests on the frozen history (§3) | Built. `EV` is stamped and recorded; ranking, the mark and stakes wait for phase 2 |
| Phase 2 | The engine reads EV (§5.3), one subagent per module | Queued |
| Phase 3 | Stacks (§5.4), after the owner's reads | Queued |
| Phase 4 | Main's money test (§5.5) | Queued |

Where the code lives: `plays.py` holds the play registry, the posted sides, both money tables
and the anytime-valid bound (`sequence_bounds`, which `scripts/books_lean.py` also reads);
`helpers.leg_kind` names a side's pick kind beside `platform_payout`; `live_fit.py` holds the
fit, the pricing and the fit's checks; `book_stamp.py` reads the books' same-line price for a
run's offers and for stored rows; `prediction/ev_stamp.py` lays both on a run's records. The
structure census gets its own module with phase 3.

## 7. The model lane

**The scoreboard.** The information weights of §1 by league, pick kind and class, with the
play's return beside them: `reflect` writes both every night (`live_fit.parquet`,
`play_record.parquet`), and `play-replay --walk-forward` reads a stamped history the same way
offline. Two parts are not built: the weights by market, and the replay of a candidate model
version (the realized replay's posted legs carry no book stamp). A candidate is judged on the
scoreboard, by the keep rule of §5.2, before it ships. Brier and the six gates remain the test
for serving a price; they no longer decide whether model work is paying.

**When.** M1 ships inside phase 1. M2 and M3 run beside phases 2 to 4. The weekly retrain and
defect fixes never stop.

- **M1. Use the books raw** (class A; no retraining). The fit's book input is the plain mean of
  same-line book prices, widened by any rung a book posts at the app's exact line.
  `book_pull` stays for now: the fit is estimated after it, so nothing is counted twice, and its
  cells have carried it only since 2026-11-23; its retirement is read on the money table with the
  kill lines' first reading. Target: `w_book` near 1 in every league.
- **M1, open: a fitted average of the books.** A weighted consensus should beat the plain mean
  of books. The first read did not settle it. On offers two or more books price, the engine's
  `book_weights.json` mean adds nothing to the plain mean (the weight on their difference is
  −0.05 (.36) over the three leagues), and no single book's gap to the plain mean points at the
  outcome by more than 1.4 standard errors. That tests one fit (weights on each book's implied
  mean against the stat result, unshrunk, near one-hot) on 46 dates at the app's line, not the
  class. The study fits the average on outcomes at posted lines, over every line two or more
  books share in four seasons of archive quotes, walk-forward, and is read on class-A offers by
  the EV fit's own gain (lane `research/m1/`).
- **M1. The board a bettor can act on.** A run takes about 25 minutes and publishes once, at its
  end; games start on the hour or the half hour. History records a row only when its game was at
  least five minutes ahead as the board went out (`prediction.cli._recordable`): a later scoring
  leaves the stored row alone and an offer first seen that late is never stored, so the money
  record, the closing-line value and the fit read boards a bettor could act on. Rows of games
  through 2026-11-25 hold each offer's last scoring instead, which landed too late for about
  three quarters of offers, and the fit's 90-day window carries them until they age out. The run
  fires at :20 and the broad poll at :05: over the 46 days measured, 77 % of last passes then
  land in time, on a board 16 to 21 minutes old at the start, scored on sportsbook quotes at
  most 30 minutes old for 59 % of offers (24 %, 46 to 51 minutes and 37 % with the run at :50
  and the poll at :30). Weights are earned on fresh quotes (M2's slope price reads 0.52 at 20
  minutes or less against 0.12 at 45 to 90). Open: a capture before every run lifts the
  fresh-quote share to 99 % for about 4,300 credits a month, and the feature log and the entries
  record (`parlay_hist`) still keep scorings that landed too late.
- **M2. Translate a book's price to another line** (class B, and every modified-payout rung): the
  largest pool of unused information. Step 1 interpolates between the same book's two nearest
  rungs where they bracket the app's line (4 % of class B). Step 2 shifts the nearest quote by
  that market's own slope between adjacent rungs, built from earlier dates' ladders and never
  from outcomes (69 % of class B: nine NBA markets and three NFL yardage markets). Step 3 puts
  the translated price into the fit as class B's book term. The model's own curve is not used
  for the shift: on a held-out rung it misses by two to four times the book slope's error. NHL
  has no ladders, so its class B stays without a book term. Target: class B's book weight at 0.5
  or more in NBA and NFL; the slope reads 0.26 (.16) in NBA against 0.07 (.10) for the engine's
  number. Kill line, set before any run: a step is dropped when its paired gain over today's
  number, on the same legs, is under one standard error of that difference.
- **M3. Price what no book quotes** (class C). NHL first, where the model already carries 0.48
  (blocked shots, skater and goalie fantasy points, points and assists for players the books
  skip); target 0.8. NBA gets a diagnosis before any build: class C carries −0.05, so split its
  legs by player tier, minutes volatility and market first. NFL: fantasy points and receptions
  for players the books skip. MLB from Opening Day 2027.
- **M4. A model of the app's price**, trained on posted picks themselves. History has carried
  both sides' payouts only since 2026-10-05, so the first fit runs at 120 days, in early
  February 2027.

Each step is written into the lane's `STATE.md` with its target rows, its expected weight change
and its kill line before it runs.

Closed by the audit, not to be retried without new data: following line moves and the line-move
agreement rule; the prior-month side; a favourite-or-longshot tilt on modified-payout picks; the
engine's `Market Prob` as a line-shopping signal.

## 8. Rules superseded, and working rules

| Rule | What replaces it |
|---|---|
| No adjustment of the served win chance from market data; the trust layer kill (#0d) | The EV fit leads ranking, the "+EV" mark and stakes |
| "Realized money is never a live threshold" | `main` gets the money test of §5.5; `devel` is never gated, and iteration stays offline |
| "No serving change; models only" (Part C) | Engine changes; model work re-aimed by the scoreboard |
| Entry floors (claimed value at least 2.0), the menu floor, the pick'em filter thresholds | Rank by EV; floors deleted |
| "Parlay EV is leg EV compounded", the premise that holds `parlay-dependence` behind gate D3 | Same-game stacks priced as joint hit rates at posted lines against the app's payout, in this lane's phase 3 (§5.4) |
| `book_pull` as the way to trust the book | The fit's `w_book` on the raw same-line mean; the knob retires in M1 |

Kept: no scripted quotes on the owner's account; the priced-population rule as the floor; the six
offline gates as the test for serving a price; the dashboard never opens the archive.

Each rule's own doc is revised in the phase that changes it: `CLAUDE.md` "Judging model
changes", `ship_gate.md` and `model_improvement_track.md` with phase 1, `sim-bettor-ledger.md`
with phase 2, `parlay-dependence.md` and the roadmap's D3 row with phase 3, the devel → main
table of `ship_gate.md` with phase 4. Until then this table is the record of the owner's
decision, and it wins where an older doc disagrees.

Working rules. Build on a side branch off `devel` and push by commit. The lane directory
`~/backups/sportstradamus/2026-11-25-profit-engine/` holds `STATE.md`, the research briefs and
the frozen `history.parquet` and `parlay_hist/` behind §3. One module per subagent;
`refactoring-specialist` on every touched Python file before review or push; the three quality
gates before "done". The table's returns are ratios; ledger money stays `Decimal`.
