# Lane brief — honest receipts: one pricing truth, three research verdicts, model-side repairs

**Read first:** CLAUDE.md, [docs/ARCHITECTURE.md](../ARCHITECTURE.md), [docs/ship_gate.md](../ship_gate.md)
(Gate 2 and the devel → main row), [story-balance.md](story-balance.md) §1 and §6 (the ledger fixes and
the killed selection-shrink lever this lane continues), and the three research briefs in
`docs/archive/` (§5). Status: OPEN — waves 1–5 are on `devel` (§4); all three briefs returned (R1
KILL, R2 DONE, R3 DONE; §5); the tail scorecard (E) is built; the consensus sanity check found no
profit in the sportsbook consensus at the platforms' real payouts (§7); the owner's decisions 7–14 (§2)
are built (§6), bar the push label and the re-read of stored game lines, which went back to the
owner with their measured size (§8 asks 5 and 6), and the four pair quotes (§8 ask 1).

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

## 2. Locked decisions (owner, 2026-10-04)

1. **Hero cohort = recommended**: edge ≥ 5 % at the platform payout, payout in (1, 2.5], with all
   posted sides as the context row. The ledger's `recommended` definition carries the same cap
   (`realized.RECOMMENDED_PAYOUT_MAX`, pinned equal to `offer_records.MAX_FAVORED_PAYOUT`).
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

Standing rules from CLAUDE.md apply: one module per subagent, refactoring-specialist before any push,
the three gates once per wave, `stat_meta.json` never committed from a session.

## 3. Verify before you trust

```bash
git log --oneline devel -14                       # the wave commits (§4)
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

Recorded, not in scope: Gate 2's `n`, `book_bss` and over-rates still count unposted rows
(`nightly._build_cell_row`); Kelly's "live_bss" is a CLV beat-rate remap with 1–19 legs → full trust (§6,
the I4 bullet); the `SPORTSTRADAMUS_ARCHIVE_DB` default-path expression now repeats in four scripts
beside `helpers/archive.py`. `docs/ship_gate.md`'s devel → main row no longer describes a
`profit_sim_kelly_yield ≥ 0` money rule: no code implemented it and the owner's rule forbids it.

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
  share, under `policy_v2` ([sim-bettor-ledger.md](sim-bettor-ledger.md) §10, which also carries the
  `even_picks` paper bettor of decision 12 and the ledger's open scars). I5e the premise rewrite in
  `parlay-dependence.md` §1 and `PARLAY_AUDIT.md`. **Open:** the sub-1 pair values and the lifted bans
  wait for the owner's quotes through the Modifiers reconciler (§8). **Struck:** I5b symmetric Σ, I5c
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
     stored keep their values until re-read, and a retrain reads the stored values: §8 ask 6. The
     NFL enrichment miss has a cause: a new row is looked up under its week's first game day, so 84
     of the 90 team-games of 2026 sit at the default although the archive holds their lines (§8
     ask 7). Still open: the other training reads that take the newest quote for a past date (the
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
     [archive/researcher_temperature_ridge.md](../archive/researcher_temperature_ridge.md) §5. Still
     open: I6a, y = ½ at a push in training's labels, waits on §8 ask 5 (the label is derived at
     eleven sites, not two, and "≤ 0.2 pp" was a pooled figure for recommended legs: the served Over
     read falls 0.5 points on average and up to 11.5 in one cell,
     [archive/push_label_design.md](../archive/push_label_design.md)); `step` passed to SkewNormal
     `get_odds` in `_step_compute_test_probabilities` and `_step_calibrate_temperature`. I6b is
     optional: align training's non-authentic fusion to serving, never the reverse.
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
- Orphans left in `analysis.py` after the retirement (zero callers before this lane): `backfill_crps`
  (its caller is already archived) and the CRPS helpers reachable only through it, and
  `reconstruct_prob`. `analysis.py` is 731 lines; the natural split is outcome resolution /
  profit-sim precompute / scoring rules.
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
  in every platform × size cell (Sleeper 2-leg .115). Parlay Kelly still sizes from the raw copula
  probability (`PARLAY_AUDIT.md` §2.6) while the selected set realizes 0.08 of priced: the parlay path's
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
  against .499 for the props it passed, below what a 2-pick needs (§8 ask 8). A lead for selection
  scope and for I6, not a proven edge.
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
  (§8), `Archive.get_movement` and `get_line_history` (diagnostics on the mixed log), and
  `Archive.get_ev` on an entry only a DFS platform posts.
- The per-pick move to 1.83 (decision 8), replayed on post-fix legs: Underdog 1.00× recommended legs
  go from 1,082 at −6.2 % per pick to 1,657 at −1.8 % (hit .5365 against .5464 needed), counting
  every posted line; each prop once they go from 814 at −8.9 % to 1,262 at −4.5 % [−10.2, +2.5]
  (hit .522). 77 unquoted one-sided Underdog rows newly fail the ±0.15 phantom gate (14 of them
  posted NFL Overs that read .767 and hit .462); the boost cap and the per-player trim pass exactly
  the same legs.
- What an entry kind asks of a leg, from the live tables at 1.00× picks from different games. The
  hit rate per pick at which an entry breaks even: 4-pick Flex .518; 2-, 3- and 4-pick Power .535,
  .536 and .537; 6-pick Flex .538; 5-pick Flex .548; 5-pick Power .549; 6-pick Power .553; 3-pick
  Flex .554. For every hit rate from .49 to .565 the 4-pick Flex returns the most per entry of
  those nine. The post-fix recommended 1.00× legs settled as real entries (each prop once, every
  leg in one entry a day, never two legs of a game, 31 days, day-block bootstrap): 4-pick Flex
  +6.0 % [−12.8, +34.0], 3-pick Power −4.9 % [−21.1, +21.1], 5-pick Power −17.2 % [−42.6, +26.8],
  6-pick Power −21.1 % [−52.1, +43.2]. No kind is told from zero. The gaps between kinds are: on
  the same days the 4-pick Flex returns 10.8 points more than the 3-pick Power [7.7, 15.1] and
  23.2 more than the 5-pick Power [6.3, 32.5]. Decision 8's 1.83 is the mean of the Power roots at
  the owner's sizes, while `payouts.py` builds 4- to 6-pick Underdog entries as Flex
  (`POWER_MAX_SIZE` = 3) and the paper ledger plays Power at 2–3 picks and Flex at 4–6. By side,
  each prop once: recommended Unders hit .537 [.501, .584] (903 props) and recommended Overs .482
  [.408, .546] (359), while Under wins .508 of all posted 1.00× lines. One window, cut after the
  fact. Scripts and logs: `~/backups/sportstradamus/2026-10-04-honest-receipts/main/entry_types/`.
- The production paper ledger's last committed entries are dated 2026-09-20: every run since ends
  "committed 0 new entries", and the cause is not pinned down. One lead: the 40 highest-probability
  gated Underdog legs are sides the platform does not post on nearly every day, the v1 cross-game
  beam ranks on probability alone, and such a side prices an entry at zero. `policy_v2` drops those
  sides before the beam; its first runs are the test.

## 8. Owner asks (one each)

The eight asks that stood here were answered on 2026-10-04 and are decisions 7–14 (§2). The first
below was deferred by the owner until the rest was built; the others came out of building it. Not
asked because no change is proposed: the parlay floor (`_MODEL_EV_FINAL_FLOOR` = 2.0; no setting
tested returns $1 per $1 while the legs are over-read).

1. **Four pair quotes from the app** (fills I5a-3; ten minutes; NFL same-game pairs only).
   For each slip: put the legs on the dashboard slip, open Model Lab › Modifiers, build the same entry
   in the Underdog app, type the payout the app shows into "Actual quoted payout (x)", then "Save
   corrected modifiers" and "Confirm save". The slips: two opposing running backs' rush attempts, one
   Higher and one Lower; the same on rush yards; a quarterback's pass yards Higher with his own running
   back's rush yards Lower, plus any third pick from another game (the app refuses a same-team pair
   alone); a quarterback's pass TDs Higher with his own receiver's TDs Higher, plus any third pick.
   No automated probing. One more number while the app is open: the Flex payouts for four 1.00×
   picks from different games. The table on file says 7.2× for four of four and 1.8× for three of
   four, the lowest bar of any entry kind (§7).
2. **Does training's fantasy-points average count as blending with the DFS line?** (yes / no.) For
   a fantasy-points market, training moves the summed-components mean halfway to the platform's own
   fantasy quote (`Stats._book_mean_shift`). It touches training rows only, never a served
   probability, and it is the best-calibrated of the three choices measured (distribution fit error
   0.221 on the component sum alone, 0.072 on the platform quote alone, 0.067 averaged). Left as it
   is; "yes" removes it at the next retrain.
3. **Fix the paper ledger's 40 copies as a new policy version?** (yes / no.) Each paper bettor is
   meant to run as 40 independent copies. The store refuses an entry any copy already holds that
   day, so copy 0 carries 223 of the 341 entries ever committed and the other 39 are near empty. The
   fix is small and, by the ledger's own rule, a new version (`policy_v3`).
4. **Sign off the bankroll table's new column?** (yes / no.) `bankroll.parquet` gained
   `policy_version` so a `policy_v2` bettor starts from $5,000 instead of continuing a v1 balance;
   old rows are stamped `policy_v1` at the next settle. The ledger brief asks for the owner's
   sign-off on any schema change after live entries exist. "No" is a small revert, and v2 then
   continues the v1 balances.
5. **Count a push as half an Over in training, now that its size is measured?** (yes / no.)
   Decision 13 took it on a figure of "at most 0.2 pp", a pooled number for recommended legs and no
   bound. The model's probabilities already count a tie as half an Over; the label they are fitted
   to and graded on counts it as an Over win, so the calibration steps push the Over read up by
   about half the tie share. A tie is 2.3 % of held-out rows, 5 % or more in 26 of the 78 served
   cells and 18.8 % in WNBA TOV. With the label at ½ the served Over read falls 0.5 points on
   average and 6 to 11 points in four cells (NFL `qb tds` 11.5, WNBA TOV 9.2, NFL targets 8.1, NBA
   DREB 6.6); Gate 1 and Gate 5 read the same label and change with it, and the 73 cells that ship
   today still ship (a replay that refits the temperature and the post-hoc map on the stored model,
   not a retrain); the Kelly haircut moves by more than 0.01 in 11 cells. At half-point lines, where
   no tie can happen, the read lands closer to the hit rate pooled over the 26 tie-heavy cells
   (+1.4 → −0.5 points) and further from it in some (WNBA TOV +3.2 → −5.8). "Yes" is one commit
   across eleven label sites; a cell changes at its next retrain. Record:
   [archive/push_label_design.md](../archive/push_label_design.md).
6. **Re-read the game lines of games already stored on the training box?** (yes / no.) Without it
   the cutoff changes nothing a retrain sees for MLB and WNBA, whose seasons are over or ending. It
   is more than the approved cut, in four ways. It needs one data fix first: 780,976 game-line rows
   in the dev archive were written by historical backfills and carry the day the backfill ran, so a
   cutoff cannot see them; moving each stamp onto its own game date changes no value that a key
   without a live poll reads (dry run, 1,800,407 keys). It fills seasons that sit at the default
   today because their lines were backfilled after the games were ingested: MLB training rows with a
   real game line go from 4.6 % to 99.2 % and NHL from 49.1 % to 99.7 %, against the 6.0 % of MLB
   rows, 9.7 % of WNBA and 3.2 % of NFL that the cut itself touches, and one retrain cannot tell the
   fill from the cut. NBA has to stay out: 382 team-games would take in a total inflated ×1.44,
   because the March–April 2026 archive rows repaired on this box return with every sync from
   production. And the 93 cached training matrices are patched or rebuilt, which moves their
   hashes, so stored confirm verdicts stop being evidence for them. Proposed scope: MLB, NHL, WNBA
   and NFL, every row; NBA none. Record:
   [archive/game_line_cutoff_design.md](../archive/game_line_cutoff_design.md) §5 and §7.3.
7. **Fix the NFL game-line lookup?** (yes / no.) A new NFL row asks the archive for its week's
   first game day (`stats/nfl.py` joins the schedule on the week alone), so 84 of the 90 team-games
   of 2026 are stored at the default, a 0.5 win chance and the league-average total, while serving
   reads the real line. Read from the code and matched by the pattern (the six rows that got a line
   are the teams that played on the week's first game day); the update was not run to watch it. The
   fix enriches after the true game day is set; stored rows are ask 6.
8. **Count a prop once in Receipts when its line moved?** (yes / no.) Receipts and the realized
   ledger count every line a prop was posted at as its own leg, so a rushing-yards prop that moved
   33.5 → 34.5 → 35.5 counts three times while a bettor holds one. Over all recommended legs since
   2026-08-31 the difference is small (7,649 lines at −9.1 % per pick, 6,250 props at −9.7 %); on
   Underdog 1.00× legs it is not (1,657 lines at −1.8 %, 1,262 props at −4.5 %). "Yes" keeps, for
   each prop and platform, the lines still posted at its last scoring (`Scored At`, stored since
   2026-10-04); alternate rungs posted side by side stay separate legs; rows from before the stamp
   keep today's count, and the page says so.
9. **Restart the dashboard once per deploy, not at every job?** (yes / no.) `scripts/run_job.sh`
   restarts the dashboard when a pull moved dashboard code or a config file, but it compares the
   checkout's last move (`HEAD@{1}..HEAD`), not this pull's. After such a deploy every job restarts
   the dashboard again until the next deploy that touches neither: 26 restarts on 2026-10-05, 22 of
   them from the ten-minute closing-line job, each dropping an open dashboard session. The same
   comparison decides `poetry install`. "Yes" is three lines: note the commit before the pull and
   compare against it.
