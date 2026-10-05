# Lane brief — honest receipts: one pricing truth, three research verdicts, model-side repairs

**Read first:** CLAUDE.md, [docs/ARCHITECTURE.md](../ARCHITECTURE.md), [docs/ship_gate.md](../ship_gate.md)
(Gate 2 and the devel → main row), [story-balance.md](story-balance.md) §1 and §6 (the ledger fixes and
the killed selection-shrink lever this lane continues), and the three research briefs in
`docs/archive/` (§5). Status: OPEN — waves 1–3 landed on `devel` 2026-10-04, unpushed; all three briefs
returned the same day (R1 KILL, R2 DONE, R3 DONE; §5); the tail scorecard (E) is built; the consensus
sanity check found no profit in the sportsbook consensus at the platforms' real payouts (§7); the gated
work (§6) waits on the owner's answers to §8.

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

Standing rules from CLAUDE.md apply: one module per subagent, refactoring-specialist before any push,
the three gates once per wave, the owner pushes, `stat_meta.json` never committed from a session.

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

- `helpers.distributions.UNDERDOG_BOOST_BASELINE = 1.78` is the per-pick convention selection used when
  every persisted leg was recommended, so the ledger keeps it on purpose. Live Underdog Power tables
  ([underdog_api.md](../underdog_api.md) P-32: 3.5/6.5/12/20/35/65/120) imply per-pick roots 1.81–1.87,
  so 1.78 under-credits Underdog winners. R2 (2026-10-04) measured it: at the 2-pick root √3.5 = 1.8708
  the post-fix Underdog recommended cohort goes from 3,055 legs at −13.1 % to 6,115 legs at −6.3 %
  [−9.4, −3.3], both platforms from −11.8 % to −7.7 %; no sign flips. Changing it changes selection
  (Model EV, Kelly, the 5 % rule, the payout-implied `Market Prob` and leg admission all move together),
  so it lands as I5a-1 on the owner's go (the table is confirmed, §2), never through reporting.
- Clean data is thin: five weeks post-fix (Date ≥ 2026-08-31); per-side payouts since 2026-10-03; MLB
  (the largest share of recommended legs) ends late October; NBA starts late October with no clean data
  until late November. No brief measured an NBA cell, so every NBA read starts from zero.

## 4. What landed (all on `devel`, 2026-10-04)

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

Recorded, not in scope: Gate 2's `n`, `book_bss` and over-rates still count unposted rows
(`nightly._build_cell_row`); `_ledger_settlement.realized_multiplier` omits per-pick multipliers and the
Underdog modifier (I5d); Kelly's "live_bss" is a CLV beat-rate remap with 1–19 legs → full trust (§6,
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
- **I5 parlay pricer — R2's ranked list (§5), a factual refresh, not a calibration fix; each item
  waits on the owner's go (the live table is confirmed, §2):** I5a-1 `UNDERDOG_BOOST_BASELINE` 1.78 →
  √3.5 = 1.8708 (moves Model EV, Kelly, the 5 % rule, the payout-implied `Market Prob`,
  `_dfs_offer_probs` and leg admission together; golden pins on 1.78 move; new selection era);
  I5a-2 `underdog_payouts.json` Power 2–8 = 3.5/6.5/12/20/35/65/120 and live Flex tiers,
  `payouts._pooled_underdog_curve` to 8 picks, Flex k-loss on the n−k largest multipliers (+43 % gated
  volume, no detectable return change); I5a-3 `banned_combos.json` Underdog values > 1 → 1.0, same-direction
  sub-1 values on untaxed pairs → 1.0, opposite-direction sub-1 values and lifted bans wait for owner quotes
  through the Modifiers reconciler; I5d `_ledger_settlement.realized_multiplier` settles at T_live × Π b × m
  from the legs' own multipliers (fallback: the committed `payout_multiplier` re-based); I5e the premise
  rewrite in `parlay-dependence.md` §1 and the volatile assumption (text in the brief) plus
  `PARLAY_AUDIT.md` §1.3 / §2.4 / §4. **Struck:** I5b symmetric Σ, I5c `underdog_tax.py`.
  `_MODEL_EV_FINAL_FLOOR` (2.0) and `_BOOKS_EV_FLOOR` (0.9) unchanged: no tested value returns ≥ 1, so no
  change is proposed (§8). I5's own acceptance = priced payout equals the quote on owner-captured slips (one Power, one
  Flex, one stacked same-game pair) and the ledger settles at it.
- **I6 model information and train/serve alignment — R3's build order (§5); each item waits on the
  owner's read.** None pulls or demotes a model. `pipeline.py` is 5,109 lines: extract helpers,
  never grow it. Retrains go through `meditate` and the existing gates unchanged.
  1. **I6e serve-time feature log + parity monitor** (engineering; no retrain, no served probability
     moves). Per scored leg, persist the `expected_columns` feature slice, `Model Skew`, `Step`, the
     probable pitcher and the decision-time book leg to a diagnostics parquet
     (`prediction/scoring.py::_score_market`; never `history.parquet` or a dashboard snapshot). A
     nightly job compares each leg with its matrix row once the matrix catches up and alarms on a
     per-cell sd(ΔP) above 1 pp; repair feature by feature (worst today: NFL completions, receptions,
     interceptions, qb yards). It also unblocks I6g and the live SkewNormal re-serve.
  2. **I6g version-age test** (pre-registered; a cadence change only if it confirms). Keep the prior
     pickle set for 14 days after each retrain and re-serve the logged legs through it offline.
     Primary statistic: fresh (version days 0–3) minus aged (days 4+) recommended gap, day-clustered,
     on NBA and NHL from late October; secondary: within-market b_model and the paired same-leg
     difference between versions. Confirmed → retrain every 3–4 days. Null → KILL: the MLB pattern
     was calendar or regression to the mean.
  3. **I6f as-of alignment of training's archive inputs** (training-side; needs a retrain;
     correctness with a small expected gap effect). A pre-game cutoff for game lines in
     `_enrich_team_markets` (`get_team_market_map(at=…)`, `stats/base.py:~630–690`): do it before the
     next MLB retrain, because the leak grows as the archive fills (late-season MLB rows are 100 %
     enriched). The NFL enrichment miss (62 % of 2026 rows sit at the 0.5 moneyline default; cause
     unknown). The book-leg cutoff aligned to decision time where ladder polls exist
     (`TRAINING_LOOKBACK_HOURS`, `helpers/archive.py:111–114`). Archiving the daily Savant affinity
     CSVs, so MLB comps can be rebuilt point-in-time for 2027.
  4. **Hygiene.** I6a: one label helper (`training/labels.py`) with y = ½ at a push, used at
     `pipeline.py:3204` and `5002` (≤ 0.2 pp). I6c, reframed: remove the `0.01·(T−1)²` ridge in
     `_brier_temperature_loss` (`pipeline.py:2841`) or set its weight by CV, and fit T and
     `PROB_STAGE` on a population that includes the scorecard's DFS-rung rows (gap −1.0 to −1.2 pp,
     recommended volume −27 to −31 %, bulk log-loss +0.001 to +0.002: an owner trade-off). Pass
     `step` to SkewNormal `get_odds` in `_step_compute_test_probabilities` and
     `_step_calibrate_temperature`. I6b is optional: align training's non-authentic fusion to
     serving, never the reverse.
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
  version-age effect is observational until I6g; NFL fp team features go dark 2026-10-11 unless
  `team_data/NFL/2026` is backfilled (memory note, not re-verified).
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
- `book_fallback` recommendations are a model-free engine today: 4,463 legs all-time at −9.9% on the decoded-book
  edge alone, 38 since 2026-08-31 at −26.0%. Whether those legs keep an edge flag is the owner's call (§8); it
  pulls no model.
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
  near 1.178 when the price moves against 1.1236 standard (the scorecard re-estimates it each run); 17
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
  recommendation rule returns the cut on every kind but the Underdog 1.00× leg, where it sits at
  break-even post-fix (2,153 recommended legs hit .533 against .535 needed and .490 for the legs it
  passed, +4.4 pp [−0.4, 7.6]; no such gap before the fix date). A lead for selection scope and for
  I6, not a proven edge (§8).
- Payout rule for any replay: bet a side only when its multiplier is provable from one poll (the
  record's "Method trap"; the check's first pass read a false +17 % without it). The tail scorecard's
  `assumed` payout class (§6) uses a looser tolerance, so read its `posted` row until that is tightened.
- Where the code stands against decision 5 (§2). The consensus *price* already excludes the DFS
  platforms whenever a sportsbook quotes the entry (`training_quotes.sportsbook_cohort`, used by
  `Archive._weighted_book_ev` and the quote resolver), and a leg whose only quote is a DFS platform is
  scored model-only (`book_quotes._has_serving_support`). The consensus *line* does not:
  `Archive.get_line` takes the median of every distinct line in the `lines` table, which has no book
  column and receives each DFS platform's main rung from `add_dfs` beside the sportsbook lines. Where
  both exist (66,101 player-market-days in the ladder era) the DFS lines move it on about 12 % (NFL
  38 %, WNBA 48 %, MLB 10 %, NHL 3 %), by 0.7 on average when they do. Its readers: the `Consensus
  Line` column, `get_closing_line`, the combo component sublines (`Stats._submarket_ev`,
  `stats/mlb.py`) and the legacy line of the training quote resolver; not yet traced into every model
  feature. The DFS platforms also sit as columns in the book-weight fit (`book_weights.json` carries
  an Underdog weight in 68 cells and a Sleeper weight in 53), unused whenever a sportsbook quotes the
  entry.

## 8. Owner asks (one each)

Every ask is a yes/no or a pick, with the facts beside it. Settled and off this list: the live Underdog
table (confirmed, §2); the DFS main line as a book leg (no, §2); the menu operating point (moot under
R1's KILL). Not asked because no change is proposed: the parlay floor (`_MODEL_EV_FINAL_FLOOR` = 2.0;
no setting tested returns $1 per $1 while the legs are over-read).

1. **Push `devel`** after reading §4 and the PR body's before/after tables.
2. **Value an Underdog pick at 1.871 instead of 1.78?** (I5a-1.) The code values a 1.00× Underdog pick
   at 1.78× stake, the fourth root of the old 4-pick payout (10×). The confirmed table pays 3.5× on a
   2-pick, so a pick is worth √3.5 = 1.871 (3-pick 1.866, 4-pick 1.861, 5-pick 1.821, 6-pick 1.809).
   The change credits Underdog winners 5 % more (the same 3,055 post-fix recommended legs go from
   −13.1 % to −8.7 %) and lowers the model read a 1.00× leg needs to be recommended from 59.0 % to
   56.1 %, which recommends 3,105 more legs (they returned −3.9 %). A truth fix; it creates no edge.
3. **Three factual fixes to parlay pricing, one go?** (I5a-2, I5a-3, I5d.) The payout file takes the
   confirmed table (it still says 3 / 6 / 10 / 20 / 25, so a 2-pick is priced 14 % low and a 6-pick
   29 % low). Pair bonuses above 1.0 in `banned_combos.json` become 1.0 (live Underdog never pays a
   same-game pair above the table; 68 % of the parlays the builder selects carry such a bonus, mean
   ×1.24). The sim ledger settles a winning parlay at table × each leg's multiplier (it leaves the leg
   multipliers out today and underpays stored winners by a median 1.68×). No probability moves.
4. **Keep the "recommended" flag on legs with no model?** A leg with no model is scored from the
   sportsbook consensus alone (`book_fallback`) and can still be flagged: 4,463 such legs all-time
   returned −9.9 %, 38 post-fix returned −26 %, and the consensus shows no edge at the platforms'
   payouts (§7). Dropping the flag keeps the legs on the board and touches no model.
5. **Make the consensus line sportsbook-only?** Decision 5 applied to the one place that breaks it
   (§7). Step 1 traces every reader of `Archive.get_line`, so no training input shifts unseen; step 2
   changes that one function to the median of sportsbook lines, with no consensus line where no
   sportsbook posts one; the models pick it up at their next retrain.
6. **Paper-trade "Underdog 1.00× legs only"?** A second paper bettor in the sim ledger that takes only
   those legs, compared with the current one after four weeks. It is the one kind of leg where the
   rule does not lose the platform's cut (§7). Nothing live changes.
7. **Open the model-information work (I6) in this order?**
   - First, log what the model saw (I6e): at scoring time, save each leg's model inputs to a
     diagnostics file. Nothing can show today that the live model sees what training fed it; the same
     model at the same line differs by 3–6 pp (standard deviation) between the two paths, worst on NFL.
     No retrain, no served number moves. Clock: late October, when NBA and NHL open; a leg served before the log exists can never be
     checked.
   - With it, keep the previous model files for 14 days after each retrain (I6g): MLB recommended legs
     are over-read by 5.6 pp in a model's first four days and 13.5 pp after, in all three post-fix
     retrains. Old files let both versions score the same legs; if the pattern holds, retrain every
     3–4 days.
   - At the next retrain, cut training's game-line quotes at pre-game (I6f: late-season MLB rows carry
     in-game quotes; the game-total quote correlates .57 with the team's runs where they leak in, .16
     on pre-game quotes) and count a push as half an Over (I6a: a tie is an Over win today; ≤ 0.2 pp).
   - After the log, NFL research (I6d): the NFL models add nothing measurable beyond the market on the
     legs tested, and no method is guaranteed.
8. **Drop the temperature penalty at the next retrain?** (I6c.) Each model has one number T that
   flattens its probabilities toward 50 % when it is overconfident. Training holds T near 1 (no
   flattening) with the penalty `0.01·(T−1)²`; in 12 of 48 cells the free T is more than 1.5× the
   penalized one. Without it the over-read on recommended legs falls 1.0–1.2 pp (of 12.5), recommended
   legs fall 27–31 %, and overall log-loss rises 0.001–0.002.
9. **Optional: four pair quotes from the app** (fills I5a-3; ten minutes; NFL same-game pairs only).
   For each slip: put the legs on the dashboard slip, open Model Lab › Modifiers, build the same entry
   in the Underdog app, type the payout the app shows into "Actual quoted payout (x)", then "Save
   corrected modifiers" and "Confirm save". The slips: two opposing running backs' rush attempts, one
   Higher and one Lower; the same on rush yards; a quarterback's pass yards Higher with his own running
   back's rush yards Lower, plus any third pick from another game (the app refuses a same-team pair
   alone); a quarterback's pass TDs Higher with his own receiver's TDs Higher, plus any third pick.
   No automated probing.
