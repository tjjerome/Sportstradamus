# Architecture

Reference map of the Sportstradamus codebase: how the packages fit together,
where data flows, and where to find things. For setup, quality gates, and the
PR workflow, see [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Package Map](#package-map)
3. [Data Flow](#data-flow)
4. [Where to Find Things](#where-to-find-things)
5. [Stable Seams](#stable-seams)
6. [Adding a New League](#adding-a-new-league)
7. [Adding a New Market](#adding-a-new-market)
8. [Modifying the Training Pipeline](#modifying-the-training-pipeline)

---

## Repository Layout

```
Sportstradamus/
├── src/
│   ├── sportstradamus/          # installable package
│   │   ├── helpers/             # shared utilities (HTTP, archive, distributions, config, IO)
│   │   ├── stats/               # per-league Stats classes + feature engineering
│   │   ├── training/            # meditate pipeline (train, calibrate, report, gates)
│   │   ├── prediction/          # prophecize pipeline (score, parlay search, snapshots)
│   │   ├── strategies/          # Kelly sizing, Underdog pick'em builder, simulated ledger
│   │   ├── collectors/          # authenticated data-collector framework (fp/ctg/savant)
│   │   ├── dashboard/           # Streamlit dashboard package
│   │   ├── scripts/             # standalone maintenance / backfill / diagnostic scripts
│   │   ├── books.py             # Underdog + Sleeper scrapers
│   │   ├── moneylines.py        # Odds API ingest (confer command)
│   │   ├── nightly.py           # reflect command (prediction resolution)
│   │   ├── analysis.py          # shared metric functions
│   │   ├── clv.py               # closing-line-value computation
│   │   ├── skew_normal.py       # custom LightGBMLSS distributions, with
│   │   │                        #   skew_normal_centered.py, double_poisson.py, hurdle.py
│   │   ├── history_schema.py    # prediction-history frame schema (+ leg_schema.py)
│   │   ├── creds/               # API keys (git-ignored; see README)
│   │   └── data/                # JSON config, per-league data, models, snapshots
│   └── deprecated/              # archived modules with no active callers
├── tests/                       # behavioral unit tests, golden/ snapshots, integration/ smoke
├── docs/                        # this file, STYLE_GUIDE.md, subsystem guides
├── scripts/                     # ops shell scripts (cron wrapper, prod sync)
├── archive/                     # runtime DuckDB odds archive (git-ignored contents)
├── .github/                     # CI workflow, PR/issue templates
├── pyproject.toml
└── README.md
```

---

## Package Map

### `helpers/` — Shared utilities

| Module | What's in it |
|---|---|
| `config.py` | Loads the JSON config files and `creds/keys.json` at import time. Exposes `stat_meta` (per-cell union of committed `stat_meta.json` and the runtime-recomputed `stat_calibration.json`), the per-field views `stat_cv` / `stat_dist` / `stat_std` / `stat_zi`, plus `stat_map`, `book_weights`, `feature_filter`, `name_map`, `abbreviations`, and the other config dicts |
| `archive.py` | `Archive` — DuckDB singleton at `archive/archive.duckdb` with `odds(league, market, game_date, entity, book, ev, under_prob, line)`, `lines(...)`, and `ladder(...)` tables. Reads: `get_ev`, `get_line`, `get_moneyline`, `get_total`, `get_team_market`, `to_pandas`, `archived_players_by_date`. Writes buffer in memory and flush on `write()`: `add_dfs`, `merge_player_books`, `set_team_books`. `LazyArchive` defers connection for modules the dashboard imports |
| `scraping.py` | `Scrape` — `requests.Session` with ScrapeOps browser-header rotation and ScrapingFish proxy fallback |
| `odds_budget.py` | Odds API credit-budget governor: usage ledger, cycle math, cost estimates, broad-run league admission, season/activity windows (`league_is_live`, `update_window_open`, `season_opener`). Loads `config/odds_api_budget.json` itself — the one exception to `config.py` owning config loads |
| `distributions.py` | Model/bookmaker fusion math: `fused_loc`, `get_ev`, `get_odds`, `fit_distro`, `no_vig_odds`, `odds_to_prob`, `prob_to_odds` |
| `integer_distribution.py` | Exact CDF endpoints and settlement for nonnegative-integer outcomes |
| `combined_markets.py` | Honest book references for combined QB markets with no direct sportsbook line |
| `io.py` | Atomic parquet/JSON IO and schema converters for the data hot path (history, parlay history, model pickles) |
| `locks.py` | Advisory file locks serializing the long-running jobs that share one checkout |
| `logging.py` | Structured JSON logging for CLI entry points (`get_logger`) |
| `market_display.py` | Market slug → human display name |
| `parlay_modifiers.py` | Expected-vs-actual quote reconciliation for pick'em correlation modifiers |
| `provenance.py` | Code-version stamps for training artifacts |
| `training_quotes.py` | Training-quote resolution with explicit provenance |
| `text.py` | String normalization and small collection utilities (`remove_accents`, `merge_dict`, `hmean`) |
| `__init__.py` | Re-exports the public API — `from sportstradamus.helpers import X` is the canonical import |

### `stats/` — Player statistics and feature engineering

| Module | What's in it |
|---|---|
| `base.py` | `Stats` base class. All shared logic: `load`, `update`, `parse_game`, `get_training_matrix`, `get_stats`, `get_volume_stats`, `profile_market`, player-comp KNN (`update_player_comps`), `trim_gamelog` |
| `nba.py` | `StatsNBA(Stats)` — NBA loading and features via `nba_api`; `nba_client.py` is its reliable transport wrapper for stats.nba.com / stats.wnba.com |
| `wnba.py` | `StatsWNBA(StatsNBA)` — WNBA, inherits from `StatsNBA` |
| `mlb.py` | `StatsMLB(Stats)` — MLB via `statsapi` |
| `nfl.py` | `StatsNFL(Stats)` — NFL via `nfl_data_py` / `nflreadpy` |
| `nhl.py` | `StatsNHL(Stats)` — NHL; `nhl_position_policy.py` holds the fail-closed training-market position policy |
| `nfl_fp_*.py`, `nfl_pbp_agg.py` | FantasyPoints snapshot loaders/aggregators and PBP/NGS-derived season aggregates feeding the NFL comp and team/defense profiles |
| `collector_snapshots.py` | Date-keyed snapshot loader for the collector framework (CTG / Savant) |
| `model_dependencies.py` | Versioned, serving-independent model dependencies for matrix reconstruction |
| `__init__.py` | Re-exports the five league classes |

### `training/` — The `meditate` pipeline

| Module | What's in it |
|---|---|
| `cli.py` | `meditate` click command — orchestrator: validates per-cell ship config, loads/updates each league's `Stats`, loops markets calling `train_market` |
| `pipeline.py` | `train_market(league, market, ...)` — the full per-market training loop: matrix load, distribution selection, target normalization, Optuna search, LightGBMLSS fit, dispersion calibration, temperature scaling, calibration-method selection, evaluation, model save |
| `baselines.py` | Target-normalization strategy registry for the SkewNormal branch — slug → forward transform + decode pair |
| `calibration.py` | `fit_book_weights`, `fit_model_weight`, `select_distribution`, blending slugs |
| `posthoc.py` | Per-cell calibration-method selector, fit on the validation split and reapplied at inference |
| `group_conditional_cdf/` | Structural calibration methods (group-conditional CDF maps: affine, two-part) |
| `hyperparams.py` | `tune_hyperparameters` — Optuna search utilities |
| `data.py` | `count_training_rows`, `trim_matrix` — training-data preparation |
| `markets.py` | `ALL_MARKETS` per-league market lists; `select_markets` narrowing for `--market` |
| `correlate.py` | `correlate(league, stat_data)` — builds `corr_same_team.parquet`, `corr_opposing.parquet`, and `corr_metadata.json` under `data/leagues/{league}/` from player stat history (intermediate cache at `data/training_data/{LEAGUE}_corr.parquet`) |
| `report.py` | `report()` — walks model pickles and writes the wide one-row-per-cell `data/training/model_stats.parquet` + `.csv` mirror; `get_market_calibration` exposes `{kelly_shrinkage, brier_skill_score, model_weight}` |
| `scorecard.py` | `compute_gates` — the offline ship gates (paired Brier CI, star/bench z, IQR ratio, debiased ECE, anti-shrinkage), called inline by `report()` and exposed as a standalone A/B CLI that never writes `model_stats.parquet`. Thresholds: [docs/ship_gate.md](ship_gate.md) |
| `graduation.py` | Shared lifecycle classification for `(league, market)` cells |
| `ship_config.py` / `config.py` | Per-cell ship-config resolution and JSON I/O for the split config files |
| `shap.py` | Post-train drift-monitoring SHAP only — writes `feature_importances.csv` / `feature_correlations.csv`; does not drive feature selection |
| `lineage.py` / `matrix_audit.py` | Deterministic matrix manifests for quarantined rebuilds; read-only cache integrity audit |
| `model_strategy/` | Per-cell strategy sweep/confirm harness (registry, sweep, confirm, TPE search) |
| `role_specs.py`, `structural_context.py`, `structural_strategies.py` | Role/position group columns and shared context for the structural calibration methods |
| `__init__.py` | Re-exports the public API |

### `prediction/` — The `prophecize` pipeline

| Module | What's in it |
|---|---|
| `cli.py` | `main` click command — loads/updates `Stats` for active leagues, scrapes DFS offers (Underdog, Sleeper), scores via `process_offers`, snapshots scored offers + parlays as parquet for the dashboard, persists `history.parquet` / the `parlay_hist/` day partitions |
| `model_prob.py` | `model_prob` — loads a trained model, computes blended probability distributions for a league/market/platform batch, decodes per the cell's target normalization |
| `scoring.py` | `process_offers`, `match_offers` — offer-level EV scoring and deduplication |
| `correlation.py` | `find_correlation` — loads the per-league correlation parquets, assembles Σ, scores parlay legs (calls `parlay.beam_search_parlays`) |
| `parlay.py` | `beam_search_parlays` (beam-search core), `GameArrays`, `GameScoringContext`, `resolve_leg_stat` |
| `payouts.py` | `payout_curve_for`, `expected_payout_with_pushes`, clip constants — platform payout tables/curves |
| `joint.py` | `parlay_payout_prob`, `psd_or_none` — Gaussian-copula joint pricing, the swappable Σ seam |
| `stories/` | Narrative generation for the dashboard: game context, offer "why" text, parlay theses, offer details |
| `persist.py` | Atomic parquet snapshot writers (`write_current_offers`, `write_current_pickem`, game context/stories/details) — the only files the dashboard reads |
| `__init__.py` | Re-exports the public API (including `beam_search_parlays`) |

### `strategies/` — Bet sizing and contest construction

| Module | What's in it |
|---|---|
| `kelly.py` | `fractional_kelly_stake`, `joint_kelly_portfolio` (cvxpy SCS), `resolve_shrinkage` (explicit > live CLV-segment BSS > training BSS > fallback `1.0`), and the `kelly` CLI for offline re-sizing of a recommendations YAML. cvxpy / pyyaml / tabulate are ordinary runtime dependencies imported at module top |
| `underdog_pickem.py` | `PickemConfig`, `RecommendedEntry`, `construct_entries`, and the `pickem-build` CLI. Pure orchestrator — no math. Covers Power, Flex, and Rivals; emits `data/recommendations/{date}.yaml` |
| `_pickem_emit.py` | YAML-emit helpers split out of `underdog_pickem.py` |
| `ledger.py` + `_ledger_*.py` | Simulated-bettor ledger: twice-daily commit orchestrator with selection, cross-game candidate, settlement, bankroll, and JSONL store layers |
| `profit_sim.py` | Kelly-sized profit / ROI / Sharpe / drawdown backtest |
| `README.md` | Module-level docs: resolution chain, blending ramp, contest variants |

### `collectors/` — Authenticated data collectors

Shared framework (`transport.py`, `auth.py`, `catalog.py`, `runner.py`,
`tabular.py`, `dispatch.py`, `commands*.py`, `cli.py`) for cookie/bearer
authenticated sources, plus one subpackage per source: `fantasypoints/`
(`fp-fetch`), `cleaningtheglass/` (`ctg-fetch`), `baseballsavant/`
(`savant-fetch`). Each writes date-keyed snapshots that `stats/` loaders fold
into features. Canonical guide: [docs/data_collectors.md](data_collectors.md).

### `dashboard/` — Streamlit dashboard

| Module | What's in it |
|---|---|
| `__init__.py` | The `dashboard` console script — launches Streamlit on `dashboard/app.py` |
| `app.py` | Main Streamlit app: page registry and navigation |
| `data.py` | Mtime-keyed cached loading of the parquet snapshots |
| `theme.py` | Non-Streamlit mirror of the design tokens (see [DESIGN.md](../DESIGN.md)) |
| `columns.py`, `legs.py`, `lenses.py`, `narrative.py`, `viewport.py` | Scoring-column semantics, leg lookup, preset filter lenses, narrative display, mobile detection |
| `slip_engine.py` | Live slip scoring for the builders — the one sanctioned live calc |
| `surfaces/` | The pages: board, tonight, games, receipts, and the lab_* diagnostics surfaces |
| `components/` | Reusable widgets: slip dock/builder, offer cards, tickets, constellation, deep-dive, gate matrix |

The dashboard reads pre-computed parquet snapshots only — it never opens the
DuckDB archive. DuckDB holds an exclusive file lock for the lifetime of any
read-write connection, and the dashboard is the only long-lived process in the
system, so an accidental `Archive()` import would block every writer job.
Enforced by `tests/golden/test_dashboard_no_archive_lock.py`; modules shared
with the pipelines use `LazyArchive` from `sportstradamus.helpers`.

### Other top-level modules

| Module | CLI command | What it does |
|---|---|---|
| `moneylines.py` | `confer` | Odds API ingest for game-level and player-prop markets: `get_moneylines`, `get_props` |
| `books.py` | (called from `prediction/cli.py`) | Underdog (`get_ud`) and Sleeper (`get_sleeper`) scrapers |
| `nightly.py` | `reflect` | Resolves predictions against results; historical parlay performance |
| `analysis.py` / `clv.py` | — | Shared metric functions; closing-line-value computation |
| `skew_normal.py`, `skew_normal_centered.py`, `double_poisson.py`, `hurdle.py` | — | Custom PyTorch distributions for LightGBMLSS (SkewNormal, centered parametrization, Double Poisson, HurdleZINB) |
| `history_schema.py`, `leg_schema.py` | — | Canonical schemas for the prediction-history frame and structured parlay legs |

---

## Data Flow

```
confer  (moneylines.py)
  get_moneylines() / get_props()   ← Odds API
        │
        ▼
  Archive.write()  →  archive/archive.duckdb

meditate  (training/cli.py)
  Stats{League}.load() / update()  ← league APIs / cached artifacts in data/player_data/
  Stats{League}.get_training_matrix(market)
        │
        ▼
  training/pipeline.train_market()
    ├─ calibration.select_distribution()
    ├─ hyperparams.tune_hyperparameters()      (Optuna)
    ├─ LightGBMLSS.fit()
    ├─ dispersion calibration + temperature scaling
    └─ model pickle → data/models/{LEAGUE}_{market}.pkl
  training/report.report() → data/training/model_stats.parquet (+ .csv mirror)

prophecize  (prediction/cli.py)
  books.get_ud() / get_sleeper()   ← Underdog / Sleeper APIs
  Archive.get_line() / get_ev()
  Stats{League}.get_stats(market, offers, date)
        │
        ▼
  prediction/scoring.process_offers()  →  model_prob.model_prob()
  prediction/correlation.find_correlation()
  prediction/parlay.beam_search_parlays()
  prediction/stories.*
        │
        ▼
  prediction/persist.py  →  parquet snapshots the dashboard reads
  history.parquet / parlay_hist/ day partitions

pickem-build  (strategies/underdog_pickem.py)
  prediction/parlay.beam_search_parlays(contest_variant=...)
  strategies/kelly.fractional_kelly_stake(...)
        │
        ▼
  data/recommendations/{date}.yaml
        ├─ sportstradamus bet kelly    (offline re-sizing)
        └─ dashboard                   (live review)
```

**NFL comp data:** Player-comp aggregates for NFL come from FantasyPoints
season exports (refreshed to `src/sportstradamus/data/player_data/NFL/{year}/`)
plus PBP-derived aggregates from `nfl_data_py` via `stats/nfl_pbp_agg.py`.
These are distinct from the `nfl_data_py` weekly logs that drive training
features. `scripts/comp_feature_stability.py` validates year-over-year
stability as a gate before `scripts/optimize_comp_weights.py --save`.

---

## Where to Find Things

| I want to... | Look here |
|---|---|
| Change which markets a league trains | `training/markets.py` → `ALL_MARKETS` |
| Change which distribution a stat uses | `src/sportstradamus/data/config/stat_meta.json` (`dist` field) |
| Control which cells train and serve | `stat_meta.json` per-cell `shipped` field — `"withheld"` (matrix kept warm, training skipped, pickle pruned) / `"devel"` / `"main"` release surfaces |
| Add/remove a sportsbook from consensus lines | `src/sportstradamus/data/config/prop_books.json` |
| Add a player name alias | `src/sportstradamus/data/config/name_map.json` |
| Understand a training stats metric | `training/report.py` (schema owner) + [docs/ship_gate.md](ship_gate.md) for the gate columns |
| Know what `kelly_shrinkage` Kelly reads | `training/report.py:get_market_calibration` → `{kelly_shrinkage, brier_skill_score, model_weight}` from `data/training/model_stats.parquet` |
| Tune the Kelly sizing constants | `strategies/kelly.py` — `DEFAULT_KELLY_FRACTION`, `MAX_FRACTION_OF_BANKROLL`, `LIVE_BLEND_FLOOR`, `LIVE_BLEND_FULL` |
| Change the Optuna hyperparameter search | `training/hyperparams.py` → `tune_hyperparameters` |
| Change how distributions blend with bookmaker lines | `helpers/distributions.py` → `fused_loc` |
| Update the season start date for a league | Usually nothing: `Stats.update` adopts the feed-observed opening night (`helpers/odds_budget.py:season_opener`) when newer than the `Stats{League}.season_start` seed constant |
| Run a `Stats.update()` in the offseason | `SPORTSTRADAMUS_FORCE_UPDATE=1` — bypasses the season window (`helpers/odds_budget.py:update_window_open`) |
| Refresh the per-cell SHAP diagnostic CSVs | `training/shap.py:see_features` — diagnostics only; training uses the full unfiltered candidate feature set |
| Change book reliability weights | `training/calibration.py` → `fit_book_weights`, or edit `data/config/book_weights.json` |
| Find why a comp feature has a certain weight | `data/config/playerCompStats.json` + `scripts/optimize_comp_weights.py` |
| Read archived / removed code | `src/deprecated/` (reintroduction protocol in its README) |

---

## Stable Seams

These boundaries are designed to be extended by **importing** them, not by
editing them. New work plugs into a seam; a seam itself changes only when the
change is about the seam.

| Seam | Home | Who imports it |
|---|---|---|
| Data collectors | `collectors/` — subclass `Source`, wire with the command builders | any new data source (the fp/ctg/savant pattern) |
| Cell lifecycle | `stat_meta.json` `shipped` flag; serving a new strategy touches `stat_meta.json`, `training/baselines.py`, and the `prediction/model_prob.py` decode | model ships/demotes |
| Target normalizations | `training/baselines.py` registry — add a slug + forward/decode pair; no pipeline edit | new target transforms |
| Payout curves | `prediction/payouts.py` + `data/config/underdog_payouts.json`; Sleeper is a per-leg-multiplier EV path, not a table entry | every product pricer |
| Joint pricing (Σ/copula) | `prediction/joint.py`; Σ is assembled only in `correlation.py` → `GameArrays`; consumed only via `joint.parlay_payout_prob` | parlay pricing; dashboard/stories read-only |
| Money sizing | `strategies/kelly.py` — `resolve_shrinkage`, `fractional_kelly_stake` | every pricer sizes through these |
| Dashboard snapshots | `prediction/persist.py` atomic parquet writers; the dashboard reads parquet only, never `Archive` (golden-enforced) | anything surfacing data in the dashboard |
| Game scoring bundle | `prediction/parlay.py:GameScoringContext` (subset re-pricing) | story menu, slip engine, product engines |

---

## Adding a New League

1. **Create `stats/{league}.py`** — subclass `Stats` from `stats/base.py`:
   - seed `self.league` and the `season_start` fallback constant
   - implement `parse_game(game)` — parse one raw API game into the standard
     gamelog columns
   - override `load()` / `update()` only for league-specific post-processing;
     the base class provides the uniform artifact load and the season-windowed
     update loop
   - ensure `get_stats` and `get_training_matrix` produce the same feature
     columns (most of this is inherited; league hooks fill the league-specific
     profiles)

2. **Export it** from `stats/__init__.py`.

3. **Add markets** to `training/markets.py` → `ALL_MARKETS["{LEAGUE}"]`.

4. **Add per-cell entries** in `data/config/stat_meta.json` for each market —
   see [Adding a New Market](#adding-a-new-market) for the cell shape. New
   cells start `"shipped": "withheld"`.

5. **Add stat name mappings** in `data/config/stat_map.json` and
   **abbreviations** in `data/config/abbreviations.json`.

6. **Register the class** in the explicit `(league, class)` tuples in
   `training/cli.py` and `prediction/cli.py`, and add the league to the
   `--league` `click.Choice` list in `training/cli.py`.

---

## Adding a New Market

A "market" is a betting category for a single stat in a single league (e.g.
"assists", "strikeouts"). To add one:

1. **`data/config/stat_meta.json`** — add a per-cell entry:

   ```json
   "{LEAGUE}": {
     "{market}": {
       "dist": "SkewNormal",
       "shipped": "withheld",
       "target_normalization": "ratio_meanyr",
       "posthoc": "none"
     }
   }
   ```

   Pick the distribution family from whether the stat is continuous or
   count-valued and whether it is zero-inflated (`SkewNormal`, `NegBin`,
   `ZINB`, `DPO`, `Mixture`; Gamma families are selected data-driven).
   `shipped: "withheld"` keeps the cell out of training and serving until it
   clears the offline gates (see [docs/ship_gate.md](ship_gate.md)). Optional
   knobs (`blending`, `zinb_mode`, `dist_training_loss`, `sn_param`,
   `hpo_selection`, `count_dispersion_objective`) default sensibly when absent.

2. **`data/config/stat_map.json`** — map the sportsbook's name for the stat to
   the internal name.

3. **`training/markets.py` → `ALL_MARKETS`** — append the market string to that
   league's list.

4. **`stats/{league}.py`** — ensure `get_training_matrix` handles the new
   market key and produces a valid `(X, y)` pair, and `get_stats` produces the
   same feature columns for live offers.

5. Run `sportstradamus meditate --league {LEAGUE} --market {market}` and review the
   new row in `data/training/model_stats.csv` (mirror of the parquet).

---

## Modifying the Training Pipeline

The training loop is `training/pipeline.py` → `train_market`. The stages in
order:

1. **Data loading** — `Stats.get_training_matrix` → `data.trim_matrix`
2. **Distribution selection** — `calibration.select_distribution`, unless the
   cell pins a family via `stat_meta.json` `dist`; stats with
   `global_mean >= 2` route to SkewNormal, lower means to a count family
3. **Target normalization** — SkewNormal targets pass through the
   `training/baselines.py` strategy named by the cell's
   `target_normalization` slug (decode happens in `model_prob`)
4. **Hyperparameter search** — Optuna via `hyperparams.tune_hyperparameters`,
   warm-seeded from the previous pickle's best params when one exists
5. **Model fit** — `LightGBMLSS.fit`
6. **Dispersion calibration** — CRPS-loss scalar minimization on the
   validation set
7. **Temperature scaling** — Brier-loss minimization on the validation set
8. **Calibration-method selection** — `training/posthoc.py` fits the cell's
   `posthoc` slug on the validation split
9. **Evaluation + diagnostics** — held-out test set scoring; diagnostics are
   written into the model pickle alongside the model
10. **Save** — pickle to `data/models/{LEAGUE}_{market}.pkl`; `report()`
    then rebuilds `model_stats.parquet` and computes the ship gates

There is no feature selection at training time: every cell trains on the full
candidate set returned by `Stats.get_stat_columns(market)`. `training/shap.py`
runs after each fit for drift monitoring only.
