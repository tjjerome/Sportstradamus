# Lane brief — combo-market pricing: sum distributions, not mean sums

Status: **open.** Fantasy-points markets (primary) and the simpler component sums
(PRA, `yards`, `hits+runs+rbi`, …) are priced today by summing component *means*
and reading a tail off a single generic-cv family. Both the book-consensus
inference and the model predictions for these markets need rework. Opened
2026-08-30 alongside the stop-serve gate described below.
Context spine: CLAUDE.md, [docs/ARCHITECTURE.md](../ARCHITECTURE.md),
predecessor incident: the Sleeper discount-line poisoning repair (2026-08-29,
devel `e1246576..55587e66`) whose fallback rewrite exposed this lane.

## Why this lane exists (graded evidence)

`history.parquet` grading of fantasy cells, DNP zeros excluded
(`Actual > 0`, pushes dropped):

| market | path | n | claimed | hit |
|---|---|---|---|---|
| hitter fantasy points underdog (MLB) | model (`ratio_meanyr`) | 2612 | 0.678 | 0.593 |
| hitter fantasy points underdog (MLB) | book_fallback | 1154 | 0.503 | 0.586 |
| pitcher fantasy points underdog (MLB) | model | 500 | 0.615 | 0.564 |
| pitcher fantasy points underdog (MLB) | book_fallback | 199 | 0.506 | 0.477 |
| fantasy points underdog (NBA/WNBA) | model | 1533 | 0.569 | **0.264** |
| fantasy points underdog (NBA/WNBA) | book_fallback | 201 | 0.509 | 0.517 |

Directional extremes are worse: fallback rows claiming ≥ 0.85 hit 1/6 (the
0.90-cap `combo_ev_inversion` unders that triggered this lane). The NBA/WNBA
model row (claimed 0.569, hit 0.264) is bad enough that the lane's first data
task is a grading audit — a scale or settlement mismatch is plausible and must
be ruled out before that cell's model is condemned.

The MLB hitter cell currently has **no pickle** (culled), so before the gate it
served 100% fallback → combo inversion. As of this lane's opening commit,
`_servable_fallback_quotes` (`prediction/model_prob.py`) folds the
`check_combo_markets` second pass in **only for `combo_props` markets**;
fantasy markets no-serve on the fallback path entirely. Pinned by
`tests/golden/test_book_fallback_quotes.py::test_fantasy_market_never_serves_combo_fallback`.

## Current mechanics (what you are replacing)

`check_combo_markets` (per-league, `stats/{mlb,nba,nhl}.py`; NFL is a stub
returning 0 "pending reimplementation") produces a single scalar **mean** for a
derived market. Three consumers:

1. **Training quotes** — `stats/base.py:2217` (`resolve_player_market_odds`):
   the combo mean becomes `fallback_ev` for `resolve_training_quote`, i.e. the
   book columns of the training matrix. **Deliberately ungated** by the
   stop-serve commit: training matrices for fantasy cells still consume the
   mean quote until Lane A replaces it — a known, temporary train/serve
   divergence on this one branch.
2. **Model-path Market EV** — `prediction/model_prob.py:744`
   (`_book_evs_for_players`): when the archive has no direct quote, the combo
   mean is the "book consensus" fed into the model↔book blend and the exported
   `Market EV`/`Market Prob`.
3. **Fallback serving** — `prediction/model_prob.py` `_servable_fallback_quotes`
   second pass (now `combo_props`-gated, see above).

Mechanics of the mean:

* **Simple combos** (`combo_props.json`: PRA, RA, PR, PA, BLST, sogBS, yards,
  hits+runs+rbi, qb tds, qb yards): `Stats._combo_market_ev` — sum of
  per-component archive EVs, each converted to the combo's distribution family
  via `_convert_to_market_dist`. All-or-nothing: any missing component → 0.
* **Fantasy scores** (`stats/mlb.py:1215 _check_mlb_fantasy`, NBA analog in
  `stats/nba.py`): weighted sum over ~6–8 components (UD hitter: singles×3,
  doubles×6, triples×8, HR×10, walks×3, RBI×2, runs×2, SB×4; UD pitcher:
  win×5, K×3, runs allowed×−3, outs×1, QS×5; NBA: PTS + REB×1.2 + AST×1.5 +
  BLK×3 + STL×3 − TOV). Components without an archived quote are filled from
  the player's recent **gamelog empirical CDF**
  (`base.py:2386 _fantasy_default_contribution`); pitcher QS gets an analytic
  normal×poisson estimate; missing hit-type EVs are prorated shares of the
  `hits` quote. One real book quote among the components is enough to return a
  value (`return ev if book_odds else 0`).
* **Tail pricing**: whoever consumes the mean prices P(over line) as
  `CDF(line | mean, stat_cv[combo market])` under the combo cell's single
  family — the dispersion of the *sum* is never computed from the components.

Two structural errors: the sum's **variance/shape is fabricated** (a generic cv
on the summed mean, blind to component dispersions and cross-component
correlation), and the inputs **silently mix** sharp book means with gamelog
empirics. That is how an 8-component mean plus a generic-cv tail served 0.90
unders that hit 17%.

### The orphaned honest kernel

`helpers/combined_markets.py` already contains the right primitives —
`normal_sum_over_prob` (bivariate-normal sum with rho),
`count_sum_over_prob` (PMF convolution), `derived_book_under_prob_row` — built
for qb-yards/qb-tds and **never wired in** (zero callers; the NFL stub is the
hole it was meant to fill). It is two-component and Normal/count-only today.
This lane generalizes it instead of writing a parallel module.

Same-player cross-stat correlations exist:
`data/leagues/{league}/corr_same_team.parquet` keys `(team, "POS.market",
"POS.market")` — same position slot ⇒ same player (e.g. `B1.singles ×
B1.walks`). Written by `training/correlate.py`.

## Lane A — book consensus on combo markets: price the sum

Goal: replace the scalar combo mean with a **component-sum distribution**, and
hand consumers an honest `(under_prob, line)` quote instead of an EV.

Sketch (research brief refines this):

1. Per component, resolve a modal-cohort quote via `resolve_training_quote`
   (the WS1 native `under_prob`/`line` columns are already served to it) and
   invert to that component's fitted family/shape — the shape-consistent
   invert machinery from the fallback rewrite (`_quote_pricing_params`)
   applies per component.
2. Combine into the weighted-sum distribution: PMF convolution for count
   components, Monte-Carlo with a Gaussian copula (rho from
   `corr_same_team.parquet` same-slot pairs) when families are mixed or
   weights non-integer — the general fantasy case.
3. Emit `under_prob` at the offered combo line. Feed all three consumers the
   same object; retire the mean-only path.
4. **Admission policy**: no silent mixing. Either every component is
   book-quoted (or an explicitly whitelisted analytic like QS), or the combo
   quote is absent — consumers already handle absent honestly post-repair.
   The gamelog fill-in either dies or survives only as a clearly-labeled
   degraded tier that never reaches serving.
5. Wire NFL `check_combo_markets` through the generalized kernel (closes the
   stub and the `combined_markets.py` orphan).

Validation: backtest the derived `under_prob` against graded combo outcomes
(claimed-vs-hit by bucket) before any consumer flips; the nightly
calibration-divergence WARN (n≥150, gap>0.08) watches it live afterward.

## Lane B — model predictions on combo markets: blend the components

Goal: a combo cell's prediction built from the **component models'
distributions** instead of (or blended with) a directly-trained combo model.

* Nearly every MLB hitter component has a live pickle (`singles`, `doubles`,
  `home-runs`, `rbi`, `runs`, `walks`, `stolen-bases`; `triples` missing —
  gamelog rate or hits-prorated analytic is the candidate fill). NBA combo
  components (PTS/REB/AST/BLK/STL/TOV) are modeled. Pitcher fantasy is
  thinner (no `pitcher-strikeouts`/`quality-start` models) — start with
  hitters.
* Combiner is the **same kernel as Lane A** applied to model-predicted
  per-component distributions (LightGBMLSS params → PMF/samples), with the
  same copula. One implementation, two input sources.
* Evaluate as a candidate against the direct combo model with the existing
  A/B harness: `sportstradamus ship scorecard --baseline <direct-model CSV>
  --candidate <blend CSV>` on frozen matrices; the six offline gates decide
  per cell, human ships. Also test a convex blend of direct-model and
  component-blend probabilities as a third arm.
* Order: MLB hitter fantasy (dead cell, pure upside) → NBA fantasy (after the
  grading audit) → PRA/yards-class simple combos (models exist and pass gates
  today; component-blend must *beat* the incumbent, not just exist).

## Sequencing and gates

0. **Research brief first** (mandatory — this changes a distribution
   mechanism): dispatch `research-analyst` on sum-of-correlated-counts
   pricing — convolution vs copula-MC, correlation estimation error, family
   choice for weighted count sums, calibration of the derived tail. Cite the
   brief in the implementing PR.
1. Grading audit of `fantasy points underdog` (NBA/WNBA) settlement scale.
2. Lane A kernel + backtest; flip consumers 1→2→3 only on green backtest.
3. Lane B behind the scorecard A/B; per-cell ship decisions.

Usual gates apply throughout (`ruff`, golden, `-m integration -n0`,
refactoring-specialist before push). No serving behavior changes without the
backtest/scorecard evidence attached.
