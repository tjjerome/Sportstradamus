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

`history.parquet` grading of fantasy cells — legacy pre-flat-schema rows
excluded (see audit note below), DNP zeros excluded (`Actual > 0`), pushes
dropped:

| market | path | n | claimed | hit |
|---|---|---|---|---|
| hitter fantasy points underdog (MLB) | model (`ratio_meanyr`) | 2612 | 0.678 | 0.593 |
| hitter fantasy points underdog (MLB) | book_fallback | 1311 | 0.513 | 0.567 |
| pitcher fantasy points underdog (MLB) | model | 544 | 0.619 | 0.557 |
| pitcher fantasy points underdog (MLB) | book_fallback | 199 | 0.506 | 0.477 |
| fantasy points underdog (WNBA) | model | 797 | 0.569 | 0.507 |
| fantasy points underdog (WNBA) | book_fallback | 213 | 0.510 | 0.526 |

Directional extremes carry the damage: clean fallback rows claiming ≥ 0.85 hit
0.40 (n=10; the 0.90-cap `combo_ev_inversion` unders that triggered this lane).

Grading caveat, resolved by the audit: legacy pre-flat-schema history rows
(nested `Offers`, top-level `Line`/`Bet`/`Win Prob` NaN — all NBA fantasy rows,
616/1678 WNBA rows) score `Hit = 0` under naive `(Bet == Result)` aggregation,
which is what produced this brief's original "hit 0.264" NBA/WNBA figure.
Settlement scale is fine (WNBA line median 27.55 vs actual median 27.75); the
cell is ~6 pt overconfident like the MLB cells, not condemned. The cell itself:
`normalize_market` scores NBA/WNBA underdog fantasy via the
`fantasy points prizepicks` cell; the history label stays
`fantasy points underdog`.

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
2. **Model-path Market EV** — `prediction/model_prob.py`
   (`_book_evs_for_players`): rebuilt 2026-08-30 on the same modal-cohort
   admission as the fallback path. A player with no real-book cohort (and no
   `combo_props` consensus) gets a NaN book leg: the blend runs model-only and
   `_finalize_records` prices `Market Prob` payout-implied with the 0.15
   disagreement drop. Fantasy combo means no longer reach the model path's
   market side — a served fantasy model's only market reference is the
   platform's own payout until Lane A prices the sum.
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
   degraded tier that never reaches serving. "Book-quoted" means a real
   sportsbook: a pick'em platform pays evenly at its posted line, so its implied
   probability is anchored near 0.5 however far the truth sits from there, and a
   platform-only component carries that anchor rather than a price. MLB stolen
   bases archive 100% of Underdog rows at exactly under 0.50 where the
   sportsbooks quote 0.87-0.89 and the market settles under 0.90 — weighted by 4
   in the fantasy sum, one such component moves the combo further than every
   honest one together.
5. Wire NFL `check_combo_markets` through the generalized kernel (closes the
   stub and the `combined_markets.py` orphan).

Validation: backtest the derived `under_prob` against graded combo outcomes
(claimed-vs-hit by bucket) before any consumer flips; the nightly
calibration-divergence WARN (n≥150, gap>0.08) watches it live afterward.

## Lane B — model predictions on combo markets: blend the components — **CLOSED, NO-GO**

Built and graded. A combo cell's prediction assembled from its **component models'
own predictives** is well-formed and cheap to produce, and it does not beat the
directly-trained combo model anywhere. **0 of 16 gradeable cells supersede**, across
three prediction arms and both correlation arms.

Nothing in serving changed. `component_sum_frame` produces no `TrainingQuote` and is
not on any serving path; the lane's deliverable is this verdict.

### What was built

* `training/component_cells.py` — which components, at what weight, decoded per row.
  Weights come from the tables serving already uses (`combo_props`, each league's
  `Stats._fantasy_combo_spec`), never a copy. Per-row predictives are read out of each
  component cell's `data/test_sets` dump — its **test split only**, so every component
  row is out-of-sample for its own model, with no pickle, feature matrix or archive read.
* `training/component_sum.py` — `component_sum_frame(league, market, ...)`, which feeds
  those components to the *same* `helpers/combined_markets.combo_sum_quote` kernel Lane A
  uses and returns a row-aligned `(candidate, baseline)` pair for the ship scorecard.
* `training/scorecard.py` — a `ComponentSum` family that grades a distribution-free
  predictive from six persisted endpoints (`SUM_CDF`, `SUM_PMF`, `SUM_Q10/Q25/Q75/Q90`).
  Without it Gate 4 — the only gate that sees dispersion, and the whole point of a
  component sum — silently fell back to a point-IQR estimator.
* `scripts/backtest_component_sum.py` — the sweep driver.

Reproduce:

    poetry run python -m sportstradamus.scripts.backtest_component_sum --rho both
    poetry run python -m sportstradamus.scripts.backtest_component_sum \
        --rho book --mixture-weight 0.5

### The arms

* **A0** — the incumbent combo model, restricted to the joined rows.
* **A1** — the NORTA component sum (`mixture_weight` unset).
* **A2** — a linear pool `0.5 * F_A1 + 0.5 * F_A0`, the handoff's convex-blend arm. It
  is built by pooling the two arms' **draw vectors**, not their CDFs: all six endpoints
  read off a sorted draw vector and a mixture CDF has no closed-form quantile to
  persist. The incumbent side is resampled through the same kernel as a one-component
  sum, which is what makes `mixture_weight=0` reproduce the incumbent's own gate row
  (pinned in `tests/test_component_sum.py`) and lets both sides carry the same
  sampling error.

### Per-cell verdict

`A0`/`A1`/`A2` columns are Gate 4's randomized-PIT KS (lower is better; the gate's
threshold is `1.358/sqrt(n)`). Every cell holds.

| cell | n / combo rows | A0 | A1 | A2 | verdict |
|---|---|---|---|---|---|
| MLB `hits+runs+rbi` | 8792 / 9858 (89%) | **0.0208** | 0.0555 | 0.0365 | HOLD |
| MLB `hitter fantasy points underdog` | 586 / 3234 (18%) | **0.0452** | 0.2184 | — | HOLD |
| NBA `BLST` | 1570 / 2192 (72%) | **0.0206** | 0.0321 | 0.0255 | HOLD |
| NBA `PA` | 1779 / 2189 (81%) | **0.0400** | 0.0696 | 0.0489 | HOLD |
| NBA `PR` | 2022 / 2195 (92%) | **0.0150** | 0.0478 | 0.0272 | HOLD |
| NBA `PRA` | 1806 / 2212 (82%) | **0.0279** | 0.0556 | 0.0397 | HOLD |
| NBA `RA` | 1828 / 2217 (82%) | **0.0348** | 0.0528 | 0.0394 | HOLD |
| NBA `fantasy points prizepicks` | 501 / 1873 (27%) | **0.0359** | 0.1036 | 0.0677 | HOLD |
| NFL `fantasy points prizepicks` | 1414 / 1938 (73%) | **0.0457** | 0.1264 | 0.0925 | HOLD |
| NFL `fantasy points underdog` | 1368 / 1851 (74%) | **0.0520** | 0.1322 | 0.0963 | HOLD |
| NFL `tds` | 457 / 2466 (19%) | **0.0405** | 0.0496 | 0.0436 | HOLD |
| NFL `yards` | 441 / 2057 (21%) | **0.0595** | 0.1509 | 0.0943 | HOLD |
| NFL `qb tds` | 277 / 341 (81%) | 0.0932 | 0.0564 | **0.0520** | HOLD (thin) |
| NFL `qb yards` | 232 / 279 (83%) | **0.0568** | 0.1797 | 0.0945 | HOLD (thin) |
| NHL `sogBS` | 234 / 2391 (10%) | **0.0691** | 0.0891 | 0.0713 | HOLD (thin) |
| WNBA `BLST` | 1850 / 2256 (82%) | 0.0234 | 0.0247 | **0.0198** | HOLD |

Cells under ~300 paired rows are reported, not judged: Gate 4's threshold and Gate 1's
bootstrap CI both widen enough that a thin cell fails for the wrong reason.

**Eight cells could not be graded at all**, none of them for a reason about the method:

* WNBA `PA` / `PR` / `PRA` / `RA` / `fantasy points prizepicks` — the `PTS` and `REB`
  test-set dumps carry a strategy identity that no longer resolves, so `load_test_set`
  refuses them. A stale artifact; re-dumping those two cells unblocks five combos.
* MLB `pitcher fantasy points underdog`, NHL `goalie fantasy points underdog` — their
  specs need sampled/Bernoulli/post-hook terms (pitcher win, quality start, goalie win)
  that no component cell models.
* NHL `skater fantasy points underdog` — only 12 of 2040 rows join every component;
  `blocked` reaches 95 of 2040. Same co-occurrence wall Lane A hit.

### Why it loses: the sum is honestly dispersed, the incumbent is fitted

The component sum's spread is *computed* — it adds the component variances and their
correlation — while the incumbent's is *fitted to this cell's outcomes*. That predicts
exactly what Gate 4's IQR ratio shows, and the split is clean:

| incumbent `g4_iqr_ratio` | cells | the sum moves it toward 1.0 |
|---|---|---|
| more than 0.10 off 1.0 | 10 | **9** |
| within 0.10 of 1.0 | 5 | **0** |

The repairs are large where they happen — NFL `qb tds` 1.689 → 1.000, NHL `sogBS`
0.667 → 1.000, MLB hitter fantasy 1.574 → 1.333 — and the damage where the incumbent
was already right is what costs the whole board. This is the same two-sidedness Lane A
measured from the book side (`combo_kernel_honest_not_predictive`): the sum is honest,
not sharp. A well-trained combo cell has already absorbed the dependence structure into
its fitted scale, and rebuilding that scale from parts throws information away.

The Brier picture agrees and is what actually blocks supersession. A1's paired-Brier
S2 mean is negative in every single cell; even where all six offline gates pass on the
candidate — MLB `hits+runs+rbi` under A2 ships G1–G6 outright — S2 lands at
−0.0006 [−0.0009, −0.0002], a small but statistically clean loss to the incumbent.

### Two findings worth keeping

**Correlation is not the lever.** ρ_book (the shipped same-player residual Spearman)
and ρ_model (the components' own NORTA `ρ_Z`, estimated on shared rows *outside* the
graded set) are interchangeable here: across 16 cells the largest PIT-KS difference is
0.0197, the mean is 0.0057, and **zero** cells flip either their gate row or their
verdict. On MLB `hits+runs+rbi` the pairwise estimates land within 0.06 and disagree in
sign (`hits|rbi` 0.468 book vs 0.507 model, `runs|rbi` 0.407 vs 0.349) — the
expectation that a better model would leave uniformly *less* residual correlation does
not hold. Do not spend another lane on the correlation input.

**The convex blend interpolates; it does not ensemble.** A2 lands strictly between A0
and A1 on Gate 4 in 13 of 15 cells. It beats both endpoints in exactly two — WNBA
`BLST` (0.0198 against 0.0234 / 0.0247, n=1850) and NFL `qb tds` (thin) — and neither
converts to a supersede. A blend is therefore a way to *bound* the damage from a
mis-specified sum, not a way to extract value the incumbent lacks.

### What would change the answer

Not a kernel change and not a correlation change. The one lead the evidence supports
is narrow: on a cell whose incumbent is badly mis-dispersed and whose components are
sound, the sum's spread is the better one. Grading that as a **dispersion prior** for
the incumbent — borrow the sum's scale, keep the incumbent's location and ranking —
is a different experiment from replacing the predictive, and it is the only Lane B
descendant worth opening. Re-dumping WNBA `PTS`/`REB` would add five more cells to
test it on.

## Sequencing and gates

0. **Research brief first** (mandatory — this changes a distribution
   mechanism): dispatch `research-analyst` on sum-of-correlated-counts
   pricing — convolution vs copula-MC, correlation estimation error, family
   choice for weighted count sums, calibration of the derived tail. Cite the
   brief in the implementing PR.
1. ~~Grading audit~~ **done** — legacy-row artifact, see the grading caveat
   above; settlement scale confirmed fine.
2. Lane A kernel + backtest; flip consumers 1→2→3 only on green backtest.
   Execution plan (phases, thresholds, admission policy):
   `~/.claude/plans/investigate-and-improve-the-abstract-lerdorf.md`.
   Blocking sub-lever found while grading the kernel: both count conventions
   (`NegBin r = 1/cv`, `DPO phi = 1/(1+cv·ev)`) impose `var = μ(1 + cv·μ)` on the
   book quote, over-dispersing the inversion. MLB `hits` settles at mean 0.847
   with P(0) = 0.4295, Poisson-exact, yet reproducing that P(0) under DPO at
   cv=0.456 demands μ=1.066 — 26% high, and the sum weights it by ~4.6. A single
   market's probability is unharmed (`get_odds` re-derives it from the same mean
   under the same cv, so the round trip is exact), but a component *sum* adds
   means, and `_blend_with_book` pools that same mean into every served count
   cell. Fix is the research brief's fitted `var = a·μ^b`, not a blanket Poisson:
   `rbi` genuinely clumps and stays over-dispersed.
3. ~~Lane B behind the scorecard A/B~~ **done, NO-GO** — graded on all 16
   reachable cells, none supersedes; see the Lane B section above.

Usual gates apply throughout (`ruff`, golden, `-m integration -n0`,
refactoring-specialist before push). No serving behavior changes without the
backtest/scorecard evidence attached.
