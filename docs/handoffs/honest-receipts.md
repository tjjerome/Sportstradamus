# Lane brief — honest receipts: one pricing truth, a selection-aware trust layer, model-side repairs

**Read first:** CLAUDE.md, [docs/ARCHITECTURE.md](../ARCHITECTURE.md), [docs/ship_gate.md](../ship_gate.md)
(Gate 2 and the devel → main row), [story-balance.md](story-balance.md) §1 and §6 (the ledger fixes and
the killed selection-shrink lever this lane continues), and the three research briefs once they land
in `docs/archive/` (§5). Status: OPEN — waves 1–3 landed on `devel` 2026-10-04, unpushed; R1 returned KILL
(§5); the gated model-side work (§6) waits on R2's and R3's verdicts and the owner's read.

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
[researcher_selection_shrink.md](../archive/researcher_selection_shrink.md)). Four things amplify it:
parlays compound per-leg overconfidence through a pricer with its own defects; no offline gate scores
the population the product recommends; train and serve disagree on unquoted rows and on ties; and the
reporting graded everything at −110.

The owner's constraints are binding and shape every stage here: **no model is pulled, demoted or
withheld on realized money; no model-free engine; research first (Opus subagents); one honest report.**
Shrinking to the sportsbook consensus was measured and killed (story-balance §6). What was never tried
is a learned, selection-aware probability layer fit on all posted settled legs, with the served model
probability as its primary monotone input plus the observables selection depends on. That is R1 (§5).
The parlay pricer's factual bugs (R2 → I5) and the train/serve skews that create the tail
overconfidence (R3 → I6) are the real "fix the models" track. Reporting was fixed first so every later
step is judged by one honest number.

## 2. Locked decisions (owner, 2026-10-04)

1. **Hero cohort = recommended**: edge ≥ 5 % at the platform payout, payout in (1, 2.5], with all
   posted sides as the context row. The ledger's `recommended` definition carries the same cap
   (`realized.RECOMMENDED_PAYOUT_MAX`, pinned equal to `offer_records.MAX_FAVORED_PAYOUT`).
2. **All three briefs run in parallel** (R1 trust layer, R2 parlay engine, R3 train/serve skew),
   `research-analyst` subagents, model opus, read-only.
3. **`Trust Prob` drives recommendations, Kelly and parlay legs (I4d) only at Tier B**: walk-forward,
   day-clustered 95 % CI lower bound of ROI at platform payouts above 0 after Holm. Tier A unlocks
   display and the D6 shadow policy only.
4. **Menu stance under a thin honest menu is decided after R1 reports its volume/ROI frontier** per
   league with volume per day at every grid threshold; I4d is not specified until the owner picks the
   operating point.

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
ls /tmp/researcher_trust_layer.md /tmp/researcher_parlay_engine.md /tmp/researcher_train_serve_skew.md 2>/dev/null
ls docs/archive/researcher_trust_layer.md docs/archive/researcher_parlay_engine.md docs/archive/researcher_train_serve_skew.md 2>/dev/null
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
  so it lands as I5a-1 after the owner confirms the live table in the app, never through reporting.
- Clean data is thin: five weeks post-fix (Date ≥ 2026-08-31); per-side payouts since 2026-10-03; MLB
  (the largest share of recommended legs) ends late October; NBA starts late October with no clean data
  until late November and needs an explicit no-fit default in any trust layer.

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

Recorded, not in scope: Gate 2's `n`, `book_bss` and over-rates still count unposted rows
(`nightly._build_cell_row`); `_ledger_settlement.realized_multiplier` omits per-pick multipliers and the
Underdog modifier (I5d); Kelly's "live_bss" is a CLV beat-rate remap with 1–19 legs → full trust (I4
sizing). `docs/ship_gate.md`'s devel → main row no longer describes a `profit_sim_kelly_yield ≥ 0`
money rule: no code implemented it and the owner's rule forbids it.

## 5. Research program (three briefs, `research-analyst`, model opus, read-only)

Rules in every brief: post-fix data only; walk-forward with weekly refits; day-clustered CIs (block
bootstrap by Date); pre-registered candidate sets with Holm; price at the platform payout via
`realized.settled_offers`; NFL reported separately; no pickles, default flips or serving-path edits; no
model-free engine, no pull/demote/withhold; KILL is a valid verdict; archive opened read-only.

| Brief | Decision question (short) | Deliverable | Gates |
|---|---|---|---|
| R1 `researcher_trust_layer.md` | Add a selection-aware probability layer (identity / L2 logistic in logit space / monotone shallow GBM; fit on all posted legs, applied post-argmax, monotone in served p, keyed to Model Version) whose `Trust Prob` drives recommendations at a threshold on `Trust EV − 1 ∈ {0, .02, .05, .08}`? KILL if nothing beats served `Win Prob` on the selected tail, or if dropping served p does not worsen day-clustered log-loss | ablation, regime map (routed to R3), volume/ROI frontier per league per day, the no-fit default | **KILL (2026-10-04)** — [archive/researcher_trust_layer.md](../archive/researcher_trust_layer.md): the pre-registered ablation fires (dropping served p costs no out-of-sample log-loss: logistic +0.00038 nats, p = .07; GBM p = .75), Tier B 0 of 12 after Holm; a market-anchored `Trust Prob` is the model-free engine the owner ruled out. Reopen only if the mid-November confirmatory re-run (frozen `r1/prereg.py`, I2 columns) passes K2 and live λ_sel's trailing-30-day CI lower bound clears 0 |
| R2 `researcher_parlay_engine.md` | Which pricer repairs (symmetric game Σ, live payout tables, Underdog's per-game correlation modifier, book-anchored or Trust leg marginals) make priced joint probability and payout match realized parlays; what per-pick Underdog convention should the single-leg ledger use? | stage 1: payout tables vs live capture + the 1.78 verdict; stage 2: Σ, m, marginals, the beam gate as curse amplifier, ranked I5 list | **DONE (2026-10-04)** — [archive/researcher_parlay_engine.md](../archive/researcher_parlay_engine.md). Stage 1 READY: √3.5 = 1.8708 replaces 1.78 once the owner confirms the live table. Stage 2: the pricer is sound and the legs fail — on 818,844 synthetic parlays from post-fix posted legs the beam's admitted legs read 8–11 pp above their hit rate, compounding to realized/priced 0.71–0.77 at 2 legs and 0.41–0.44 at 5, while book-anchored marginals (a diagnostic, model-free) calibrate every size. I5b symmetric Σ KILL (Δ log-loss −0.00011, p = .12; the shipped Σ is indistinguishable from independence); I5c `underdog_tax.py` KILL (m < 1 on 3–7 % of entries, ≈ 0.99); I5a/I5d/I5e GO as truth, not calibration (68 % of gated entries carry a stale pair bonus, mean ×1.24; the ledger underpays stored winners by a median 1.68×). The Model EV ≥ 2.0 beam floor amplifies the curse (realized/priced 0.50 → 0.00 across priced-EV bands; today's gated entries hit 2.3 % against 29.7 % priced). The joint-ratio acceptance moves to I6 |
| R3 `researcher_train_serve_skew.md` | Which train/serve skews (unquoted pooling at w = 1.0 vs w, combo-sum quotes, tie label, DFS-rung Market Prob from the consensus shape, tail reliability by z and alt flag, NFL Under bias by era, the evaluation design) explain the .650-read vs .496-hit gap, ranked by measured share? | ranked defect table with CIs, fix per defect, files touched, retrain need; the tail-scorecard spec (E) | per-defect KILL valid |

R1 runs now on proxies (decision time = the platform's last ladder poll of the line, quote class from
`Market Projection` + `Model Version`); its confirmatory re-run needs ~4 weeks of `DECISION_COLS`
(mid-November). Briefs land at `/tmp/researcher_<topic>.md` and are copied to `docs/archive/`.

## 6. Gated implementation (one module per subagent; entry = the brief's verdict + the owner's read)

- **I4 trust layer — KILLED with R1 (2026-10-04); nothing below is built.** The one piece that needs a new
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
- **I5 parlay pricer — R2's ranked list (2026-10-04), a factual refresh, not a calibration fix; each item
  waits on the owner's read and the live-table confirmation:** I5a-1 `UNDERDOG_BOOST_BASELINE` 1.78 →
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
  `_MODEL_EV_FINAL_FLOOR` (2.0) and `_BOOKS_EV_FLOOR` (0.9) unchanged pending the owner: no tested value
  returns ≥ 1. I5's own acceptance = priced payout equals the quote on owner-captured slips (one Power, one
  Flex, one stacked same-game pair) and the ledger settles at it.
- **I6 training-side fixes** (after R3, per defect): one label helper for ties at `pipeline.py:3204` and
  `5002`; fusion consistency for unquoted and combo rows; tail recalibration on validation rows at DFS
  rungs with E's machinery; NFL retrain on the post-leak era. `pipeline.py` is 5,109 lines: extract
  helpers, never grow it. Retrains go through `meditate` and the existing gates unchanged.
- **E tail scorecard** (`scripts/tail_scorecard.py`, `-m diagnostics`, after R3's spec): reprices every
  test-set row at the DFS rungs held before Commence, replays the live recommendation rule, and reports
  the selected-tail gap per cell with day-clustered CIs. Routes to training, never to demotion.

Rollout per [model_improvement_track.md](model_improvement_track.md) §6.10: replay-validate in the
fixed profit sim, then A/B live through the D6 sim-bettor ledger. I4 and I5 change selection: tag each
by Model Version / policy version and never pool across eras.

## 7. Records, not tasks

- Success in numbers: the hero reconciles with the ledger to the leg (done); ≈ 2 weeks, three verdicts;
  I4 Tier A = |mean Trust Prob − hit| ≤ 3 pp on its own recommended set with a CI covering 0 (Win Prob's
  gap today: 15 pp); I4 Tier B = ROI CI lower bound > 0 at the platform payout at whatever volume that
  implies; I5 = realized/priced joint ratio in [0.9, 1.1] at every entry size; I6 = the tail scorecard's
  selected-tail gap halves on the first clean window after a retrain.
- Risks the owner accepted: the honest menu may get very thin (S5 recommended 45 of 3,469 legs); R1 may
  KILL, which routes the lever to R3/I6; NFL fp team features go dark 2026-10-11 unless
  `team_data/NFL/2026` is backfilled (memory note, not re-verified).
- Phantom rows (unquoted, |Win Prob − Market Prob| > 0.15, dropped at `offer_records.py`) are not
  persisted; if R1 finds Trust Prob still rising near |D| = 0.15, write them to a separate file through
  `helpers/io`, never into history.
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
  spec picked at the 5% threshold fell on 2026-09-23 behind one stale pre-lineup sportsbook slot (book-quote
  staleness is a selection variable); the model's information share is 0.47–0.61 in a version's first days and
  0.05–0.14 one to two weeks later (version-age decay, exploratory; R3 tests it on the test sets).
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

## 8. Owner asks (one each)

- Push `devel` after reading §4 and the PR body's before/after tables.
- Read each brief as it lands and record the verdict (ship spec / KILL) here under §5 before any I4/I5/I6
  subagent is dispatched.
- Confirm the live Underdog Power/Flex table in the app for your state (blocks I5a-1 and I5a-2; the
  convention itself is decided: √3.5).
- Quote, through the Modifiers reconciler, the Over/Under side of the untaxed pair types (opposing RB
  carries, opposing RB rush yds, QB pass yds × same-team RB rush yds) and re-quote QB pass TDs × same WR
  TDs; no automated probing. These fill I5a-3 and decide whether any structural pair edge exists (only
  opposing RB carries Over/Under reads above break-even, EV 0.96–1.07, and the engine builds Under/Under).
- Decide the beam floor stance (`_MODEL_EV_FINAL_FLOOR`): no setting tested returns ≥ 1 per $1 on the
  selected set.
- Decide whether `book_fallback` legs keep an edge flag (R1 conclusion 6): they are priced off the decoded
  sportsbook consensus alone and lose at the DFS payout; dropping the flag pulls no model.
- Decision 4 (menu operating point) is moot under the KILL; the frontier in the brief §Frontier stays the
  reference for what volume an honest menu would have had.
