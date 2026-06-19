# Feature Improvement Plan

> **ARCHIVED — superseded by [`../model_improvement_track.md`](../model_improvement_track.md)
> (feature stages → §7.0/§7.3/§7.4/§7.6/§7.7, validation protocol + worked example → §11.2,
> feature-count verdict → §7.4 item 5).** Status claims below are stale and non-normative.
> Retained for the full per-item hypothesis/evidence prose the consolidation compressed.

> **Status: ACTIVE — stage 0 (hygiene + quick wins).** Canonical home for feature-lever
> detail across all five leagues; [`operation_ship_75.md`](operation_ship_75.md) §5.8
> cross-references this doc. Gate thresholds live in [`ship_gate.md`](ship_gate.md);
> MLB/NHL activation gates (data freshness, D1/D2 GO) live in
> [`handoffs/mlb-nhl-activation.md`](handoffs/mlb-nhl-activation.md).

## 0. Changelog

- 2026-06-11 §11 executor-hardened: verified regen↔deterministic ordering, scorecard CLI, `--bypass-withholding`, worked example; QW-1/M-1 constants = build-time decisions.
- 2026-06-11 doc created: full-board feature audit → workstreams A–E, roadmap, risks.

## 1. Mission & scope

**Improve the signal in the X-matrix** — the feature columns
`get_stat_columns(market)` hands to training — for player-prop models in NBA, WNBA,
MLB, NFL, NHL.

In scope, the four asks this plan answers:

1. better calculation/use of player comps (workstream A),
2. better aggregation/processing of data already collected (workstream B),
3. neglected data — in the database or free to add (workstreams C, D),
4. an honest "too many features?" evaluation (workstream E).

Out of scope (cross-ref, don't restate): normalization / model-loss / blend /
calibration axes ([`operation_ship_75.md`](operation_ship_75.md) §5.2–§5.6); the
B-frame design — book `Line`/`Odds`/`EV` are deliberately **not** features
([`pipeline.py`](../src/sportstradamus/training/pipeline.py) keeps them in separate
B-frames for the blend and calibration; the blend owns book information); MLB/NHL
activation gates ([`handoffs/mlb-nhl-activation.md`](handoffs/mlb-nhl-activation.md)).

Operator constraints: **new external data must be free sources only**; **all five
leagues get equal effort budget** — refinement for NBA/WNBA/NFL (machinery exists),
foundation parity for MLB/NHL (machinery starved).

## 2. The feature system today

One row per feature family; builders live in
[`stats/base.py`](../src/sportstradamus/stats/base.py) unless noted.

| Family | Builder | Leagues | Emitted columns |
|---|---|---|---|
| Rolling windows | `_rolling_features` | all | `Avg1/3/5/10`, `AvgYr`, `AvgH2H` (medians), `Mean10/MeanYr/MeanH2H`, `STD10/STDYr`, ratios vs `MeanYr` |
| Game context | `_game_context` | all | `Home`, `Moneyline`, `Total`, `DaysOff`, `DaysIntoSeason`, `GamesPlayed`, `H2HPlayed`, `ZeroYr` |
| Player profile | `base_profile` + `profile_market` | all | per `stat_types` entry: raw mean, `" short"` (last-5), `" growth"` (last-5 trend slope); plus `z`, `position z`, `home` split, `moneyline gain`, `totals gain` |
| KNN comps | `_apply_comp_features`, `_nonmlb_comp_features`, `_mlb_comp_games` | all | `comps mean`, `comps mean (EB)` (K=10), `comps p25/p75`, `Player comps z` (opponent-conditional, distance-weighted), `Defense comp n`, `Defense comp distance` |
| Defense profile | `profile_market` | all | `Defense avg` (league-scaled), `home`, `moneyline gain`, `totals gain`, per-position splits, `Defense {stat}` |
| Team profile | `base_profile` | all | `Team {stat}` from last-10 team games |
| Park factors | `_join_defense_and_parks` | MLB | `PF R/OBP/H/1B/2B/3B/HR/BB/K` per home park |
| FantasyPoints as-of | [`nfl.py`](../src/sportstradamus/stats/nfl.py) `_join_fp_player_features` / `_join_fp_team_features` | NFL | ~155 `Player {col}_asof` season-to-date aggregates (leakage-clean, prior-week cut) + team snapshots |
| Volume projections | `load_volume_model_params` | NFL/NHL/MLB (+NBA MIN) | `proj {market} loc/scale/alpha` (or `mean/std`) joined as features |

**Column counts and the breadth gap.** NBA/WNBA ≈ 460–480 columns, NFL ≈ 440–550
(with FP as-of), MLB ≈ 100–120, NHL ≈ 100–120. The driver is raw `stat_types`
breadth: NBA 138, NFL 88, **MLB 13, NHL 14**. MLB/NHL are not "efficient cells with
nothing to learn" — they are starved of inputs. Closing that gap is workstreams
C/D's largest item (L-1, M-9).

**Candidate-set policy.** Since the no-filter rewire (researcher Option C; Akhiat &
Touchanti 2024, arXiv:2411.05937), `get_stat_columns` returns the full unfiltered
candidate set; the only pruning is `_prune_uninformative_features`
([`pipeline.py`](../src/sportstradamus/training/pipeline.py)) dropping all-NaN /
zero-variance columns. See §9 before proposing any filtering.

## 3. Evidence base

From `src/sportstradamus/data/training/feature_importances.csv` (transposed: rows =
features, columns = cells; MLB 23 / NFL 23 / NBA 22 / WNBA 19 cells — **NHL absent,
never prop-trained**) and `model_stats.parquet` (29 trained cells: NBA 14 / NFL 5 /
WNBA 10):

- **Sparsity**: 70–92% of features per cell carry near-zero |SHAP| (MLB worst at
  ~92%). Expected for a wide-candidate GBDT (§9), but it means audits must read
  importances per cell, never globally.
- **Concentration**: top-10 features capture 35–53% of total |SHAP| per cell.
- **Dominant families**: rolling means (`Mean10`, `Avg10`, `STDYr`) and player
  profile stats everywhere; defense profile dominates MLB (`Defense avg`,
  `Defense PASO`); `Player comps z` is a stable top-15 feature but only ~10–11% of
  total importance — the comp pool is informative and under-harvested (§5).
- **Board state** ([`stat_meta.json`](../src/sportstradamus/data/config/stat_meta.json)):
  NBA 21 cells (14 devel), WNBA 18 (11), NFL 20 (5), MLB 24 (all withheld), NHL 16
  (all withheld).
- **Where features are the named lever** (ship75 §2 census): the NFL g1+g4 volume
  five (attempts, carries, completions, receiving-yards, rushing-yards), WNBA STL,
  NBA PF — cells that need *signal*, not width. Gate-4 under-dispersion, the
  board's dominant symptom, belongs to the calibration/normalization axes, not
  features; this plan complements, never replaces, that work.

## 4. Verify before you trust

Numbers above drift with every `meditate`. Re-derive before acting:

```bash
# per-league cell counts + ship status
python3 -c "
import json, collections
m = json.load(open('src/sportstradamus/data/config/stat_meta.json'))
for lg, mkts in m.items():
    print(lg, len(mkts), dict(collections.Counter(v.get('shipped') for v in mkts.values())))"

# importances: cells per league, stale double-prefix rows, per-cell active features
python3 -c "
import pandas as pd, collections
fi = pd.read_csv('src/sportstradamus/data/training/feature_importances.csv', index_col=0)
print(dict(collections.Counter(c.split('_')[0] for c in fi.columns)))
print('stale Player Player rows:', sum(i.startswith('Player Player ') for i in fi.index))
nz = (fi > fi.max().max()*0.001).sum(); print('active features per cell (median):', int(nz.median()))"

# training-cache column counts per league (one cell each)
python3 -c "
import pandas as pd
for f in ['NBA_PTS','NFL_attempts','MLB_hits','WNBA_PTS']:
    try: print(f, len(pd.read_parquet(f'src/sportstradamus/data/training_data/{f}.parquet').columns))
    except FileNotFoundError: print(f, 'no cache')"

# archive odds coverage per league (movement-feature feasibility, C-3)
python3 -c "
import duckdb
con = duckdb.connect('archive/archive.duckdb', read_only=True)
print(con.execute('select league, min(game_date), max(game_date), count(*) from odds group by league').fetchall())"
```

## 5. Workstream A — player comps

The KNN machinery (z-scored profiles → weighted BallTree from
[`playerCompStats.json`](../src/sportstradamus/data/config/playerCompStats.json) →
distance-weighted comp outcomes) is sound and week-gated against population leakage
for NBA/WNBA/NFL/NHL. Three gaps:

**A-1 (= QW-5). Harvest more from the existing pool.** Effort S–M.
*Hypothesis*: the pool already computes pairs + distances (`_comp_pairs`, cached);
emitting more aggregates is near-free signal. *Evidence*: `comps z` stable top-15
at ~10% importance. *Implementation*: in `_apply_comp_features` and
`_nonmlb_comp_features` add (a) distance-weighted comp **std** (dispersion of the
pool, complements p25/p75), (b) comp **trend** (weighted mean of comps' `growth`),
(c) opponent-conditional comp mean on the **raw scale** (today only the z-score is
emitted), (d) **recency weighting** of comp outcomes in the opponent merge (decay
by game age). *Validation/Revert*: §11 standard; SHAP-inert columns reverted.

**A-2 (= M-5). Re-optimize weights with the stability gate.** Effort M (mostly compute).
*Hypothesis*: per-position comp weights were tuned at different times per league;
YoY-unstable comp features inject noise. *Evidence*: idle tooling exists —
[`comp_feature_stability.py`](../src/sportstradamus/scripts/comp_feature_stability.py)
(YoY Pearson keep/transform/drop),
[`evaluate_comp_features.py`](../src/sportstradamus/scripts/evaluate_comp_features.py)
(greedy add/remove),
[`optimize_comp_weights.py`](../src/sportstradamus/scripts/optimize_comp_weights.py)
(differential evolution on Spearman). *Implementation*: run stability → evaluate →
optimize per league; extend the optimizer to co-optimize K (currently min 5 / max
15–20) and the distance-kernel exponent (`1/(1+d)`). *Validation*: Spearman is
in-sample to the comp objective — the arbiter is the §11 deterministic A/B.
*Revert*: restore prior `playerCompStats.json` (versioned).

**A-3 (= L-2). Time-gated MLB comps.** Effort L; depends on L-1.
*Hypothesis/Evidence*: the one documented open comp leakage —
`TODO(comp-leakage-mlb)` in `_ensure_comps`
([`base.py`](../src/sportstradamus/stats/base.py)): MLB reuses today-state Savant
affinity CSVs across every training gameday (no `_compute_comps` override).
*Implementation*: once statcast ingest (L-1) lands, build MLB comp profiles from
as-of statcast aggregates, add an MLB block to `playerCompStats.json`, retire the
affinity CSVs to cold-start fallback. *Validation*: extend the comp week-gating
tests; deterministic A/B. *Revert*: affinity CSVs.

## 6. Workstream B — better aggregation of existing data

**B-1 (= QW-1). Spread, opponent implied total, game total, blowout flag.** Effort S, all leagues.
*Hypothesis*: game script (favored/dog × pace) conditions volume markets; the model
sees only own-team `Moneyline`/`Total`. *Evidence*: ship75 §5.8 3b calls for a
blowout flag "from the book moneyline+spread"; no spread feature exists; zero new
data needed — [`moneylines.py`](../src/sportstradamus/moneylines.py) stores archive
`Totals` as the implied team total (`(game_total + team_spread)/2`), so
`Spread = Total(team) − Total(opp)` and `GameTotal = Total(team) + Total(opp)` come
from two `archive.get_total()` calls, leakage-safe via the existing `at=target_at`
mechanism. *Implementation*: `_game_context` adds `Opp Total`, `Spread`,
`GameTotal`, `Blowout = |Spread| > threshold` — the threshold is an operator
decision at build time, landed as a named per-league module-level constant (no
magic numbers); register the columns in the `Common` bucket of
[`feature_filter.json`](../src/sportstradamus/data/config/feature_filter.json).
Both `_game_context` branches change: historical (gamelog) and upcoming
(archive lookups).
*Targets*: NFL volume five, NBA PF. *Validation/Revert*: §11.

**B-2 (= M-2). EWMA recency family.** Effort M, all leagues.
*Hypothesis*: `Avg1/3/5` are high-variance tail snapshots; exponentially-weighted
mean/std at 2–3 half-lives adds a continuous recency axis to the family that
already dominates SHAP. *Implementation*: `_rolling_features` (leakage-safe by
construction — `short_gamelog` is strictly `< date`). *Expectation*: some EWMA
columns displace `Avg3/Avg5` rather than add — fine. *Validation/Revert*: §11.

**B-3 (= M-1). The ship75 §5.8 3a/3b build.** Effort M; the highest-confidence item.
`MeanYr_expanding_shifted` = `groupby(player).expanding().mean().shift(1)`, plus a
`× opponent` variant and EB / James-Stein shrinkage where expanding is sparse
(shrinkage strength cross-validated per ship75 §5.7, landed as a named constant —
do not invent a value); defense×player interaction columns. Pre-registered targets: NFL volume five, WNBA
STL, NBA PF. Implement in `_rolling_features` / `profile_market`; extend
[`test_meanyr_mean10_leakage.py`](../tests/test_meanyr_mean10_leakage.py).
Mirrored exactly in training and `get_stats` (strict `<` date filter).

**B-4 (= M-3). Schedule-derived fatigue: B2B, 3-in-4, opponent-rest differential, travel/timezone.** Effort M.
*Hypothesis*: fatigue moves minutes/TOI and efficiency; opponent-rest differential
is the priced version. *Evidence*: `DaysOff` already earns importance; team B2B,
opponent B2B, timezone shift are zero-scrape derivables from schedules already
fetched + one new static stadium lat/lon/dome table under
`src/sportstradamus/data/`. *Order*: NBA/WNBA/NHL first (schedule density), then
NFL/MLB. *Validation/Revert*: §11.

**B-5 (= M-7). Missing-value semantics: stop conflating missing with zero.** Effort M build, L validation. **Research-brief flagged.**
*Hypothesis*: the matrix fill (`get_training_matrix` tail,
[`base.py`](../src/sportstradamus/stats/base.py)) and the live fill
([`model_prob.py`](../src/sportstradamus/prediction/model_prob.py)) both
`fillna(0)` — parity holds, but 0 conflates "no H2H games" with "averaged zero";
LightGBM's native missing-value handling is strictly more expressive.
*Implementation*: exempt a curated column subset (H2H, comps, movement) from the
fill in **both** files in one PR (parity rule). *Validation*: changes every cell's
input distribution → per-league deterministic A/B on 2–3 cells before board-wide
adoption; brief first. *Revert*: restore fills.

## 7. Workstream C — neglected data already in the database

**C-1 (= QW-3). MLB batting order reaches the matrix.** Effort S after a 30-min verification.
*Hypothesis*: batting order (1–9) and `starting batter` are first-order PA-volume
drivers for every hitter market. *Evidence*: both are already stored per gamelog
row ([`mlb.py`](../src/sportstradamus/stats/mlb.py) `_player_game_row`) and reach
`playerProfile["depth"]` — but `_join_defense_and_parks`
([`base.py`](../src/sportstradamus/stats/base.py)) then overwrites
`Player depth = Player position` for MLB. *Implementation*: verify what
`Player position` holds for MLB, stop the overwrite so batting order survives as
`Player depth` (keep a separate position/`starting batter` flag). Document the
skew: training sees the realized same-day lineup, live sees announced-or-mode.
*Validation/Revert*: §11; matrix build validates now, training ships post-D1.

**C-2 (= QW-4). NFL schedule context.** Effort S.
`weekday` / `gametime` are fetched and discarded
([`nfl.py`](../src/sportstradamus/stats/nfl.py)). Emit `Weekday`, `PrimeTime`,
and short-week rest differential. *Targets*: NFL volume + usage cells.

**C-3 (= M-6). Line-movement and book-disagreement features.** Effort M–L. **Research-brief flagged.**
*Hypothesis*: opening→T drift, n_moves, and cross-book EV dispersion on the player
prop encode news (lineups, injuries) the gamelog lacks. *Evidence*:
`Archive.get_movement()` already computes opening/closing/peak/trough/n_moves and
`get_ev_history()` gives per-book EVs
([`archive.py`](../src/sportstradamus/helpers/archive.py)) — never fed to models.
*Implementation*: resolve at `target_at = game_time − TRAINING_LOOKBACK` in
`get_training_matrix` and at now() live; columns go into X, not the B-frames.
*Why the brief is required*: (a) the X/B boundary is deliberate — pre-register
whether g1 gain is allowed to come from book echo; (b) the odds time-series is
recent-era and league-uneven (§4 check), and under fill-0 "no movement data"
collides with "line never moved" — depends on B-5 or a per-feature NaN exemption.
*Revert*: SHAP-inert or g5 degradation.

**C-4 (= QW-2). Stale-importances hygiene.** Effort S.
155 `Player Player *_asof` rows in `feature_importances.csv` are zero-importance
residue of a fixed double-prefix bug (current NFL caches are clean — verified
against `NFL_attempts.parquet`). Batch-rebuild via `see_features()`
([`training/shap.py`](../src/sportstradamus/training/shap.py)) so rows derive only
from current model pickles; add a golden assert that no `Player Player ` rows
exist. Not a model change — it keeps every future feature audit honest.

## 8. Workstream D — new data (free sources only)

**D-1 (= L-1). MLB statcast breadth via pybaseball.** Effort XL. **The MLB parity centerpiece.**
*Hypothesis*: MLB's 13 raw `stat_types` vs NBA's 138 is the largest signal gap on
the board; statcast event data (barrel%, exit velocity, xBA/xwOBA, hard-hit%,
whiff%, chase%, pitcher arsenal/usage/velocity) is free, historical, and
event-dated — point-in-time aggregates are leakage-clean by construction.
*Evidence*: MLB's defense profile is already a top SHAP family with crude inputs;
24 withheld cells await activation. *Implementation*: new ingest module patterned
on the FP loader family
([`nfl_fp_loader.py`](../src/sportstradamus/stats/nfl_fp_loader.py)); extend
`stat_types`; join through the existing `_join_fp_player_features` /
`_join_fp_team_features` base hooks rather than new plumbing. *Validation*: §11;
training ships post-D1 (activation lane).

**D-2 (= M-9). NHL foundation: goalie quality + xG breadth.** Effort L.
*Hypothesis*: shots/saves/goals markets hinge on opponent-goalie quality and shot
generation; NHL has never had prop models trained. *Evidence*: the scraping
already exists — dobbersports predicted goalies populate `upcoming_games`,
`opponent goalie` is stored per gamelog row, MoneyPuck CSVs are already pulled
([`nhl.py`](../src/sportstradamus/stats/nhl.py)); the gap is **joining
goalie-quality features into the matrix**, plus widening skater/team `stat_types`
with MoneyPuck xG aggregates. *Implementation*: opponent-goalie SV/GSAx features;
extend `stat_types`; validate NHL comps (machinery + `playerCompStats.json[NHL]`
exist). *Greenfield rule*: start matrices lean — there is no incumbent baseline,
so the first scorecard IS the baseline; do not import NFL-width column counts
without SHAP evidence.

**D-3 (= M-4 + L-3). Availability: starters, inactives, usage vacuum.**
- **M-4** (effort M): NBA/WNBA starter flag — nba_api boxscores carry
  `START_POSITION` historically, so it backfills leakage-clean; emit starter flag +
  "starters missing" count. Both leagues in one PR (shared base machinery).
- **L-3** (effort XL, **research brief REQUIRED**): injury/inactive reports
  (nfl-data-py injuries, NBA official inactives) + a usage-vacuum feature (sum of
  OUT teammates' `MeanYr`). The brief must settle report-timestamp leakage design
  (announcement time vs `target_at`) and backfill honesty before any build.

**D-4 (= L-4). Weather.** Effort L. NFL/MLB outdoor parks.
Open-Meteo (free) historical + forecast; needs the stadium table from B-4. Train
on realized weather, archive forecasts going forward, document the
forecast-vs-realized serve skew; revisit at first scorecard.

**D-5 (= L-5). Referee/umpire assignments.** Effort L. Lowest priority.
Free scrapes; targets single cells (MLB K/BB via umpire K-zone tendencies, NBA PF
via crew foul rates). Only after the owning league's medium items are exhausted.

## 9. Workstream E — feature-count honesty ("too many features?")

Four findings, two actions. **Do not re-add global pre-train filtering.**

1. **The no-filter verdict stands, scoped precisely.** The rewire's evidence (tree
   ensembles statistically tie FS-filtered vs no-FS) covers *global pre-train
   filtering on the cells tested*. Per the repo's scope-kill norm ("unproven ≠
   refuted"), it does **not** cover: small-n NFL cells (300–1000 rows × 440+
   candidates), duplicate/derived-column redundancy, or Optuna wall-time.
2. **Near-zero SHAP ≠ harmful.** 70–92% inert features is the expected signature
   of a healthy GBDT on a wide candidate set. The *measured* cost is wall-time
   (linear in feature count — `_prune_uninformative_features` docstring); the
   small-n variance cost is a **hypothesis, not a finding** — treat it as exactly
   that.
3. **Sanctioned subtraction is narrow.** (a) C-4 stale-row hygiene; (b) the
   existing all-NaN/zero-variance prune stays; (c) **M-8, a bounded
   pre-registered ablation**: 2–3 small-n NFL cells from the g1-blocked volume
   five, full set vs per-cell top-K |SHAP| (K ∈ {50, 100}) vs family-pruned, under
   `meditate --deterministic` at fixed HP, scored on the honest val→test gate row.
   Decision rule, registered now: adopt per-cell top-K **only if** it beats full
   on g1 `ci_lo` across the tested cells **and** survives a real-HPO confirm;
   otherwise record the verdict here and close the axis.
4. **The durable answer is addition discipline, not subtraction.** Every
   experiment in this plan carries the SHAP < 0.001 ⇒ revert rule, so the
   candidate set cannot accrete junk going forward. `feature_correlations.csv`
   feeds a redundancy *audit*: collapse |ρ| > 0.98 pairs only where one column is
   derived from the other by construction, and only through the standard A/B.

## 10. Per-league plans

Equal treatment = equal effort budget (~12–15 person-days each), not identical
items.

| League | Target cells (evidence) | Items | Sequencing / dependencies |
|---|---|---|---|
| **NBA** | PF (g1 — needs signal); calibration-blocked cells get signal alongside ship75 axes | QW-1/2/5, M-1, M-2, M-3, M-4, M-5, M-6, L-3, L-5 | feedback-rich; first to validate every shared-base build |
| **WNBA** | STL (g1) | every shared-base item lands for WNBA in the same PR (shared `base.py` machinery); EB-shrunk variants preferred (small n); M-4 explicitly dual-league | piggybacks NBA regens |
| **NFL** | attempts, carries, completions, receiving-yards, rushing-yards (g1+g4); qb-yards, passing-first-downs (multi-gate) | QW-1/2/4, **M-1 primary**, M-2, M-5, M-6, **M-8 ablation**, L-3, L-4 | richest source set already (FP suite); the feature-count question is tested *here* |
| **MLB** | 24 withheld; hitter volume markets first (batting order), Ks later (umpire) | QW-3, M-2/M-3 shared, **L-1 centerpiece**, L-2, L-4, L-5 | matrix/feature builds proceed against refreshed gamelogs now; training ships post-D1 ([activation lane](handoffs/mlb-nhl-activation.md)) |
| **NHL** | 16 withheld, greenfield; goalie SV + skater shots/points first | **M-9 foundation**, QW-1/5, M-2/M-3 shared | post-D2 (same lane); lean-first — first board IS the baseline |

**Anti-drift guard.** Foundation leagues (MLB/NHL) have slow feedback — no trained
cells until D1/D2 — while refinement leagues confirm fast. The §12 roadmap pins
per-league milestones so effort does not silently flow to wherever feedback is
quickest.

## 11. Validation protocol & batching policy

Stated once; every experiment block above references it.

1. **Leakage test first.** Extend
   [`test_meanyr_mean10_leakage.py`](../tests/test_meanyr_mean10_leakage.py):
   temporal features assert strict `<` date visibility; as-of/external features
   assert the training value is reconstructable from data observable at
   `game_time − TRAINING_LOOKBACK`.
2. **Train/live parity test per batch.** One frozen gameday: `get_training_matrix`
   vs `get_stats` column sets and values. Paired surfaces change in one PR — the
   two fill sites, any new context column's historical + upcoming branches.
   **Build-first:** no parity harness exists yet — creating it (a golden test
   comparing the two paths on a frozen gameday) is part of the first quick-win
   batch, before any feature lands.
3. **Regen + deterministic A/B.** Two verified facts in
   [`pipeline.py`](../src/sportstradamus/training/pipeline.py) force the ordering:
   `--deterministic` **never rebuilds the matrix** (`_step_load_matrix` returns
   cache-only under the flag; `_step_persist_matrix_and_comps` skips the parquet
   write), and a **plain `meditate` run publishes production artifacts** (model
   pickle, test CSV — only `--deterministic` redirects to the sandbox). So a new
   feature is invisible until the cache parquet is rebuilt, and the rebuild must
   never go through a plain run mid-experiment. Recipe, per cell:
   1. preserve baselines *before* the code change: copy the cache parquet aside
      and run a baseline deterministic train (worked example below);
   2. land the feature edit — training + live mirror together (§11.2);
   3. rebuild the cache **directly**: delete the parquet, call
      `get_training_matrix` + `trim_matrix` ([`training/data.py`](../src/sportstradamus/training/data.py))
      and write the parquet yourself (snippet below);
   4. run the candidate deterministic train with flags identical to the baseline;
   5. compare the two sandbox test CSVs with the scorecard CLI —
      `poetry run python -m sportstradamus.training.scorecard --baseline … --candidate …`
      (writes a sandbox scorecard CSV, never `model_stats.parquet`).
   Deterministic runs are sandboxed by construction: fixed HP, pinned seed, test
   CSVs under `data/test_sets/deterministic/{target_normalization}/`, pickles
   under `research/models/deterministic/{strategy}/`.
4. **Inert-revert rule** (ship75 §5.8, verbatim): SHAP importance < 0.001 ⇒ the
   feature is inert ⇒ revert. Never carry dead columns.
5. **Ship path.** Deterministic A/B improvement ⇒ full-HPO `meditate` ⇒ official
   5-gate scorecard; incumbent cells additionally need `supersede_verdict()`.
   Never ship on in-sample screens.
6. **Batching policy.** Features land per-league in batches — one regen per league
   per batch, never one feature at a time (regen costs hours per league board).
   Quality gates (ruff + golden + integration) before any push.
7. **Research-first triggers.** Plain feature levers need no brief. Items that
   move a leakage surface or the book/model information boundary do: **C-3**
   (movement features), **B-5** (missing-value semantics), **L-3** (injury
   timestamps), **D-4** (weather forecast skew — brief-note level).

### Worked example — QW-1 on NFL `carries`, end-to-end

Every target cell on this plan is **withheld**, so every `meditate` invocation
needs `--bypass-withholding` or the cell is silently skipped. Read the cell's
`target_normalization` from
[`stat_meta.json`](../src/sportstradamus/data/config/stat_meta.json) (`carries` →
`ratio_meanyr`) and pass it explicitly so baseline and candidate match production
config — **identical flags on both runs**.

```bash
# 0. baseline insurance + baseline deterministic run (BEFORE any code change)
cp src/sportstradamus/data/training_data/NFL_carries.parquet /tmp/NFL_carries.cache.bak
poetry run meditate --league NFL --market carries --deterministic \
    --bypass-withholding --target-normalization ratio_meanyr
cp src/sportstradamus/data/test_sets/deterministic/ratio_meanyr/NFL_carries.csv \
    /tmp/NFL_carries.baseline.csv

# 1. implement the feature in BOTH paths (QW-1: spread columns in _game_context,
#    historical + upcoming branches), extend the leakage test, run quality gates.

# 2. rebuild the cache with the new columns (direct — NOT via plain meditate)
rm src/sportstradamus/data/training_data/NFL_carries.parquet
poetry run python - <<'EOF'
from sportstradamus.stats import StatsNFL
from sportstradamus.training.data import trim_matrix
s = StatsNFL(); s.load(); s.update()   # update(): gamelog must be current
M = trim_matrix(s.get_training_matrix("carries"), 15000)
M.to_parquet("src/sportstradamus/data/training_data/NFL_carries.parquet",
             compression="zstd", index=True)
EOF

# 3. candidate deterministic run (identical flags) + scorecard comparison
poetry run meditate --league NFL --market carries --deterministic \
    --bypass-withholding --target-normalization ratio_meanyr
poetry run python -m sportstradamus.training.scorecard \
    --baseline /tmp/NFL_carries.baseline.csv \
    --candidate src/sportstradamus/data/test_sets/deterministic/ratio_meanyr/NFL_carries.csv

# 4. decide: gate-row improvement => real-HPO confirm (step 5 above);
#    SHAP < 0.001 or regression => restore /tmp/NFL_carries.cache.bak, revert edit.
```

Notes for the executor: the deterministic CSV subdir is the normalization slug
(`{target_normalization}{_hurdle?}`); the parquet snippet mirrors what
`_step_persist_matrix_and_comps` writes (same `trim_matrix(…, 15000)`, same
compression) so the candidate cache is shaped like a production one; `update()`
requires league-API access — run after the daily jobs or accept a slightly stale
gamelog on both sides (fine: A/B only needs both sides identical).

## 12. Prioritized roadmap

Effort: S ≤ 1 day · M 2–5 days · L 1–3 wk · XL 3+ wk. Status values: todo /
brief / building / A-B / shipped / closed.

| ID | What | Leagues | Effort | Expected impact (basis) | Depends on | Status |
|---|---|---|---|---|---|---|
| QW-2 / C-4 | importances stale-row purge | all | S | audit honesty (155 dead rows) | — | todo |
| QW-1 / B-1 | spread / opp total / blowout | all | S | game-script axis for volume cells (§5.8 3b) | — | todo |
| QW-3 / C-1 | MLB batting-order fix | MLB | S | first-order PA-volume driver | verify | todo |
| QW-4 / C-2 | NFL weekday/primetime | NFL | S | short-week usage shifts | — | todo |
| QW-5 / A-1 | comp-output extensions | NBA/WNBA/NFL/NHL | S–M | comps informative, under-harvested (~10% SHAP) | — | todo |
| M-1 / B-3 | §5.8 3a/3b expanding means + interactions | all (NFL first) | M | pre-registered ship75 lever, RANK 2/3 | — | todo |
| M-2 / B-2 | EWMA family | all | M | sharpen the dominant SHAP family | — | todo |
| M-3 / B-4 | B2B / travel / rest-diff | NBA/WNBA/NHL → all | M | fatigue axis, zero new scraping | stadium table | todo |
| M-4 / D-3 | NBA/WNBA starter flag | NBA+WNBA | M | minutes conditioning, leakage-clean backfill | — | todo |
| M-5 / A-2 | comp weight re-optimization | all | M | stability-gated noise reduction | — | todo |
| M-8 | NFL small-n ablation | NFL | M | settles §9 hypothesis with pre-registered rule | QW-2 | todo |
| M-6 / C-3 | line-movement features | all | M–L | news signal the gamelog lacks | **brief**, B-5 or NaN exemption | brief |
| M-7 / B-5 | missing-value semantics | all | M+L | stop missing≡0 conflation | **brief** | brief |
| M-9 / D-2 | NHL foundation | NHL | L | greenfield: goalie + xG into matrix | D2 lane | todo |
| L-1 / D-1 | MLB statcast breadth | MLB | XL | largest signal gap on the board (13 vs 138 stat_types) | — (train post-D1) | todo |
| L-2 / A-3 | time-gated MLB comps | MLB | L | closes documented comp leakage | L-1 | todo |
| L-3 | injuries + usage vacuum | NBA/WNBA/NFL | XL | availability shock signal | **brief required** | brief |
| L-4 / D-4 | weather | NFL/MLB | L | outdoor passing/HR effects | M-3 stadium table, brief-note | todo |
| L-5 / D-5 | referee/umpire | MLB/NBA | L | single-cell targets | league mediums done | todo |

## 13. Structural risks register

1. **Cache-append inertness (highest).** New columns are NaN over cached rows →
   pruned → silently inert. Control: mandatory per-cell parquet delete + full
   regen per batch (§11.3); add a golden test that a sentinel new column survives
   `_prune_uninformative_features` after regen.
2. **Train/live parity.** Dual implementation surfaces (per-gameday training loop
   vs live `get_stats`) and dual fill sites. Control: §11.2 parity test; one-PR
   rule for paired sites.
3. **Leakage surfaces.** Open: MLB comps (`TODO(comp-leakage-mlb)`). Skew class to
   document, not eliminate: realized-vs-announced (MLB lineups, NBA starters,
   weather). Movement features resolve at `target_at`, never at
   scrape-time-of-training-run.
4. **Archive sparsity per league.** Odds history is recent-era; MLB/NHL book
   honesty is gated on the activation lane's audit. Under fill-0, "no movement
   data" collides with "line never moved" — C-3 depends on B-5 or a per-feature
   NaN exemption.
5. **Stale diagnostics.** `feature_importances.csv` carries dead rows until QW-2;
   `model_stats.parquet` covers 29 cells only — read importances per cell.
6. **Regen feasibility.** The 850-day training window must be resolvable from the
   archive at regen time — verify per league (§4 block) before deleting any cache.
7. **Greenfield NHL over-reach.** No baseline to A/B against — lean-first, treat
   the first scorecard as the baseline.
8. **Doc drift.** This doc is the canonical home of feature-lever detail; ship75
   §5.8 holds the summary + cross-ref. When a fact changes, fix it here.

## 14. Cross-references

1. [`operation_ship_75.md`](operation_ship_75.md) — lever stack, §5.8 features
   summary, validation loop this plan reuses
2. [`ship_gate.md`](ship_gate.md) — g1–g5 thresholds (authoritative)
3. [`handoffs/mlb-nhl-activation.md`](handoffs/mlb-nhl-activation.md) — MLB/NHL
   data freshness + D1/D2 GO gates
4. [`CONTRIBUTING.md`](../CONTRIBUTING.md) §Package Map
5. [`STYLE_GUIDE.md`](STYLE_GUIDE.md) §16 — doc conventions this file follows
