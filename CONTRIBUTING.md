# Contributing to Sportstradamus

This document is for anyone making changes to the codebase — new features, bug fixes,
model improvements, or adding a new league. It covers where everything lives, how the
packages fit together, the style rules, and the change workflow.

Read [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) once before writing any code. The
style guide is the mechanical source of truth; this document gives you the map.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Package Map](#package-map)
3. [Data Flow](#data-flow)
4. [Where to Find Things](#where-to-find-things)
5. [Making Changes](#making-changes)
6. [Style Rules Summary](#style-rules-summary)
7. [Adding a New League](#adding-a-new-league)
8. [Adding a New Market](#adding-a-new-market)
9. [Modifying the Training Pipeline](#modifying-the-training-pipeline)
10. [Shipping to Production (devel)](#shipping-to-production-devel)
11. [Tests](#tests)
12. [Archived / Deprecated Code](#archived--deprecated-code)

---

## Repository Layout

```
Sportstradamus/
├── src/
│   ├── sportstradamus/         # installable package
│   │   ├── helpers/            # shared utilities (HTTP, archive, distributions, config)
│   │   ├── stats/              # per-league Stats classes
│   │   ├── training/           # meditate pipeline
│   │   ├── prediction/         # prophecize pipeline
│   │   ├── books.py            # Underdog + Sleeper scrapers
│   │   ├── moneylines.py       # Odds API scraper (confer command)
│   │   ├── nightly.py          # reflect command
│   │   ├── dashboard.py        # dashboard command entry point
│   │   ├── dashboard_app.py    # Streamlit app
│   │   ├── skew_normal.py      # custom PyTorch distribution
│   │   ├── analysis.py
│   │   ├── creds/              # API keys + Google OAuth (git-ignored)
│   │   ├── data/               # JSON config + CSV correlation + trained models
│   │   └── scripts/            # standalone analysis/maintenance scripts
│   └── deprecated/             # archived modules with no active callers
├── tests/
│   └── golden/                 # CLI snapshot tests + fixtures
├── docs/
│   └── STYLE_GUIDE.md
├── .github/workflows/ci.yml
├── .pre-commit-config.yaml
├── pyproject.toml
├── CLAUDE.md                   # guidance for AI contributors
└── README.md
```

---

## Package Map

### `helpers/` — Shared utilities

| Module | What's in it |
|---|---|
| `config.py` | Loads every JSON config file at import time; exposes `stat_meta` (per-cell `{dist, shipped, strategy, cv, std, zi}` union of committed `stat_meta.json` and gitignored `stat_calibration.json`), the legacy per-field views `stat_cv` / `stat_dist` / `stat_std` / `stat_zi` derived from it, plus `stat_map`, `book_weights`, `books`, `feature_filter`, `banned`, `abbreviations`, `combo_props`, `nhl_goalies`, `name_map`, `odds_api` |
| `archive.py` | `Archive` class — DuckDB singleton at `archive/archive.duckdb` with `odds(league, market, game_date, entity, book, ev)` and `lines(...)` tables. Public read methods: `get_ev`, `get_line`, `get_moneyline`, `get_total`, `get_team_market`, `to_pandas`, `archived_players_by_date`. Public write methods: `add_dfs`, `merge_player_books`, `set_team_books`, `write`. Module-level helper: `clean_archive`. |
| `scraping.py` | `Scrape` class — `requests.Session` with ScrapeOps browser-header rotation and ScrapingFish proxy fallback |
| `distributions.py` | `fused_loc`, `get_ev`, `get_odds`, `fit_distro`, `no_vig_odds`, `odds_to_prob`, `prob_to_odds`, `set_model_start_values` |
| `text.py` | `remove_accents`, `merge_dict`, `hmean`, `get_trends`, `get_mlb_pitchers` |
| `__init__.py` | Re-exports all of the above — existing code that does `from sportstradamus.helpers import X` keeps working |

### `stats/` — Player statistics and feature engineering

| Module | What's in it |
|---|---|
| `base.py` | `Stats` abstract base class. All shared logic: `load`, `update`, `get_training_matrix`, `get_stats`, `get_volume_stats`, `profile_market`, `_build_comps`, `update_player_comps`, `trim_gamelog` |
| `nba.py` | `StatsNBA(Stats)` — NBA game log loading, feature engineering via `nba_api` |
| `wnba.py` | `StatsWNBA(StatsNBA)` — WNBA, inherits from `StatsNBA` |
| `mlb.py` | `StatsMLB(Stats)` — MLB via `statsapi` |
| `nfl.py` | `StatsNFL(Stats)` — NFL via `nfl_data_py` and `nflreadpy` |
| `nhl.py` | `StatsNHL(Stats)` — NHL |
| `__init__.py` | Re-exports all six classes |

### `training/` — The `meditate` pipeline

| Module | What's in it |
|---|---|
| `cli.py` | `meditate` click command — thin orchestrator: stats init, league setup (book weights, comps, correlations), per-market loop calling `train_market` |
| `pipeline.py` | `train_market(league, market, stat_data, ...)` — the full per-market training loop: data load, distribution selection, normalization, Optuna search, LightGBMLSS fit, dispersion calibration, temperature scaling, model save |
| `calibration.py` | `fit_book_weights`, `fit_model_weight`, `select_distribution` |
| `shap.py` | Post-train drift-monitoring SHAP only: `compute_market_importance` writes per-cell |SHAP| + corr columns to `feature_importances.csv` / `feature_correlations.csv` after each model trains. `see_features` rebuilds the CSVs from all pickles in batch. SHAP no longer drives feature selection (2026-05-27 no-filter rewire) |
| `correlate.py` | `correlate(league, stat_data)` — builds `{LEAGUE}_corr.csv` from player stat history |
| `report.py` | `report()` — walks model pickles, builds the wide one-row-per-cell training stats, and writes `data/training/model_stats.parquet` + `model_stats.csv` mirror. `get_market_calibration` exposes `{kelly_shrinkage, brier_skill_score, model_weight}` for Kelly to read. Inline calls `training.scorecard.compute_gates` per cell |
| `scorecard.py` | `compute_gates(test_set_df, *, league, market)` — five offline ship gates (G1 paired Brier CI, G2/G3 star/bench z, G4 IQR ratio, G5 Roelofs-debiased ECE) called inline by `report()` and exposed via a standalone click CLI (`poetry run python -m sportstradamus.training.scorecard ...`) for A/B-test runs. The CLI never writes `model_stats.parquet` |
| `data.py` | `count_training_rows`, `trim_matrix`, `_histogram_weights` |
| `hyperparams.py` | `warm_start_hyper_opt`, `_BoundedResponseFn` |
| `markets.py` | `ALL_MARKETS` — per-league market name lists |
| `config.py` | `load/save_distribution_config`, `load_shipped_config`, `load/save_zi_config` |
| `__init__.py` | Re-exports the public API |

### `prediction/` — The `prophecize` pipeline

| Module | What's in it |
|---|---|
| `cli.py` | `main` click command — Google auth, stats init, fetch offers, score, write sheet. Also `_get_sheets_client` |
| `model_prob.py` | `model_prob` — loads a trained model, computes blended probability distributions for every offer |
| `scoring.py` | `process_offers`, `match_offers` — offer-level EV scoring and deduplication |
| `correlation.py` | `find_correlation` — loads correlation CSVs, scores parlay legs (calls `parlay.beam_search_parlays`) |
| `parlay.py` | `beam_search_parlays` plus its helpers `_payout_curve_for`, `_expected_payout_with_pushes`, `_nearest_psd`, and the per-search constants. Re-exported at the package level. |
| `persist.py` | `save_data` — writes scored offers to disk |
| `__init__.py` | Re-exports the public API (including `beam_search_parlays` from `parlay.py`) |

> `prediction/sheets.py` was deprecated on `devel`; the live path is
> `data/recommendations/{date}.yaml` (written by
> `strategies/underdog_pickem.py`) which the dashboard reads directly.

### `strategies/` — Underdog-native decision engine

| Module | What's in it |
|---|---|
| `kelly.py` | `fractional_kelly_stake`, `joint_kelly_portfolio`, `KellyCandidate`, the resolution chain (explicit > live CLV-segment BSS > training BSS > fallback `1.0`), and the `kelly` CLI. cvxpy / pyyaml / tabulate are lazy-imported from `[tool.poetry.group.strategy]`. |
| `underdog_pickem.py` | `PickemConfig`, `RecommendedEntry`, `construct_entries`, and the `pickem-build` CLI. Pure orchestrator — no math. Covers Power, Flex, and Rivals (Rivals restricted to 2/3-leg sizes). |
| `_pickem_emit.py` | YAML-emit helpers split out so `underdog_pickem.py` stays under the 300-line cap. |
| `__init__.py` | Re-exports `kelly` and `underdog_pickem`. |
| `README.md` | Module-level docs: resolution chain, blending ramp, contest variants. |

### Other top-level modules

| Module | CLI command | What it does |
|---|---|---|
| `moneylines.py` | `confer` | Odds API scraper: `get_moneylines`, `get_props` |
| `books.py` | (called from `prediction/cli.py`) | Underdog (`get_ud`) and Sleeper (`get_sleeper`) scrapers |
| `nightly.py` | `reflect` | Historical parlay performance analysis |
| `dashboard.py` / `dashboard_app.py` | `dashboard` | Streamlit dashboard |
| `skew_normal.py` | — | Custom PyTorch `SkewNormal` distribution for LightGBMLSS |

---

## Data Flow

```
confer
  moneylines.get_props()       ← Odds API
  books.get_ud()               ← Underdog API
  books.get_sleeper()          ← Sleeper API
        │
        ▼
  Archive.write()  →  archive/archive.duckdb

meditate
  Stats{League}.load()         ← league APIs / cached CSVs in data/player_data/
  Stats{League}.get_training_matrix(market)
        │
        ▼
  training/pipeline.train_market()
    ├─ LightGBMLSS.fit()
    ├─ calibration.fit_model_weight()
    ├─ calibration.fit_book_weights()
    └─ model pickle → data/models/{LEAGUE}_{market}.pkl

prophecize
  Archive.get_line() / get_ev()
  Stats{League}.get_stats(offer, game_date)
        │
        ▼
  prediction/model_prob.model_prob()
  prediction/scoring.process_offers()
  prediction/correlation.find_correlation()
  prediction/parlay.beam_search_parlays()
        │
        ▼
  prediction/persist.save_data()  →  scored-offer cache

pickem-build (Phase 3)
  prediction/parlay.beam_search_parlays(contest_variant=...)
  strategies/kelly.fractional_kelly_stake(...)
        │
        ▼
  data/recommendations/{date}.yaml
        │
        ▼
  poetry run kelly --from <yaml>           ← offline re-sizing
  dashboard "Today's Recommendations" tab  ← live review
```

**NFL comp data:** Player-comp aggregates for NFL come from FantasyPoints
season exports (manually refreshed to
`src/sportstradamus/data/player_data/NFL/{year}/`) plus PBP-derived
aggregates from `nfl_data_py` via `stats/nfl_pbp_agg.py`. These are
distinct from the `nfl_data_py` weekly logs that drive training features.
The comp feature list is evidence-based per established stickiness
research; `scripts/comp_feature_stability.py` validates Y/Y stability as
a gate before `optimize_comp_weights.py --save`.

---

## Where to Find Things

| I want to... | Look here |
|---|---|
| Change which markets a league trains | `training/markets.py` → `ALL_MARKETS` |
| Change which distribution a stat uses | `src/sportstradamus/data/config/stat_meta.json` (`dist` field) |
| Ship a cell that cleared Gate 1 | `src/sportstradamus/data/config/stat_meta.json` (`shipped: "withheld"` → `"devel"`) |
| Add/remove a sportsbook from consensus lines | `src/sportstradamus/data/config/prop_books.json` |
| Add a player name alias | `src/sportstradamus/data/config/name_map.json` |
| Understand a training stats metric | [CLAUDE.md](CLAUDE.md) §Training stats — covers the wide one-row-per-cell schema (identity, scoring rules, ECE, discrimination, EV/line, Kelly, shape, ship gates, hyperparameters) and the CSV mirror |
| Know what `kelly_shrinkage` Kelly reads | `training/report.py:get_market_calibration` → returns `{kelly_shrinkage, brier_skill_score, model_weight}` for a `(league, market)` from `data/training/model_stats.parquet` |
| Tune the Kelly resolution chain | `strategies/kelly.py` constants `LIVE_BLEND_FLOOR=25`, `LIVE_BLEND_FULL=100`, `DEFAULT_KELLY_FRACTION=0.25`, `MAX_FRACTION_OF_BANKROLL=0.005` |
| Change the confidence cutoff for picks | `prediction/scoring.py` → `MIN_CONFIDENCE` |
| Change the Optuna hyperparameter search space | `training/pipeline.py` → `train_market` objective |
| Change how distributions are blended with bookmaker lines | `helpers/distributions.py` → `fused_loc` |
| Update the season start date for a league | `stats/{league}.py` → `Stats{League}.season_start` |
| Refresh per-cell SHAP diagnostic CSV from scratch | `training/shap.py:see_features` — re-runs SHAP on every saved pickle, rewrites `feature_importances.csv` + `feature_correlations.csv`. Does NOT change feature selection — selection is unfiltered since the 2026-05-27 rewire |
| Change book reliability weights | `training/calibration.py` → `fit_book_weights`, or edit `data/book_weights.json` directly |
| Find why a comp feature has a certain weight | `data/playerCompStats.json` + `scripts/optimize_comp_weights.py` |
| Read archived / removed code | `src/deprecated/` |

---

## Making Changes

### Setup

```bash
poetry install
poetry run pre-commit install
```

`pre-commit install` wires `ruff` (lint + format) into the commit hook. Every
commit runs `ruff check --fix` and `ruff format`. CI runs the same checks plus
`pytest tests/golden/`.

### Workflow

1. Work on a feature branch off `devel` (production tracks `devel`; `main`
   lags — see CLAUDE.md §Production deployment).
2. Run `poetry run ruff check src/ --fix && poetry run ruff format src/` before
   committing. The pre-commit hook does this, but running it manually first avoids
   a hook-rejected commit.
3. Run `poetry run pytest tests/golden/` to verify CLI help text hasn't changed.
   If you intentionally changed a CLI flag or added a new command, regenerate the
   snapshot:
   ```bash
   REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py
   ```
4. For changes that touch the training pipeline, run `meditate` on a small league
   (`--league WNBA` if in-season) to confirm models still train without errors and
   the `data/training/model_stats.csv` values look plausible.
5. For changes that touch `prophecize`, run it against the live data and spot-check
   the exported sheet.

### Adding a new dependency

```bash
poetry add <package>          # runtime dependency
poetry add --group dev <package>   # dev-only
```

PyTorch must stay CPU-only (the `torch-cpu` source in `pyproject.toml`). Do not
change the `torch` dependency without verifying the new wheel exists in that source.

---

## Style Rules Summary

Code conventions live in [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) — read it
once, cite sections by number (`§N`). It is the single source of truth; this
document deliberately does not restate the rules, so the two can't drift apart.

The posture in one line: **less code, written for a human to maintain** — no
wrappers that only forward a call, no fallbacks for cases that can't happen, no
comments that narrate the code. The §18 smells table is the fastest checklist.

Mechanical enforcement (`ruff format` + `ruff check --fix`, line length 100,
Google docstrings, the three quality gates) is configured in `pyproject.toml`
and runs automatically — see STYLE_GUIDE §19.

---

## Adding a New League

1. **Create `stats/{league}.py`** — subclass `Stats` from `stats/base.py`. Implement:
   - `season_start: date`
   - `load()` — download game logs from the league API, write CSVs to
     `data/player_data/{LEAGUE}/{YEAR}/`
   - `update()` — fetch only new games since last load
   - `parse_game(game)` — parse a raw API game into the standard gamelog columns
   - `get_stats(offer, game_date)` — build a feature vector for a single offer
   - `get_training_matrix(market)` — build (X, y) for one market

2. **Export it** from `stats/__init__.py`.

3. **Add markets** to `training/markets.py` → `ALL_MARKETS["{LEAGUE}"]`.

4. **Add per-cell entries** in `data/config/stat_meta.json` for each new
   market: `{"dist": "Gamma" | "NegBin" | ..., "shipped": "withheld",
   "strategy": "none"}`. The cell ships only after passing Gate 1 (then
   edit `shipped` to `"devel"`).

5. **Add stat name mappings** in `data/config/stat_map.json` to normalize
   API names to internal names.

6. **Add abbreviations** to `data/config/abbreviations.json`.

7. **Wire into `meditate`** — `training/cli.py` already reads `ALL_MARKETS` and
   instantiates each league's Stats class dynamically; if your class follows the naming
   convention `Stats{LEAGUE}` (e.g. `StatsXFL`) it will be picked up automatically.

8. Add a `--league` option value in `training/cli.py` → the `click.Choice` list.

---

## Adding a New Market

A "market" is a betting category for a single stat in a single league (e.g. "assists",
"strikeouts"). To add one:

1. **`data/config/stat_meta.json`** — add a per-cell entry:
   `"{LEAGUE}": {"{market}": {"dist": "Gamma", "shipped": "withheld",
   "strategy": "none"}}` (pick a distribution family based on whether the
   stat is continuous or count-valued and whether it has zero-inflation).
   `shipped: "withheld"` keeps the cell out of training until it clears
   Gate 1.

2. **`data/config/stat_map.json`** — map the sportsbook's name for the stat to your internal name.

3. **`training/markets.py` → `ALL_MARKETS`** — append the market string to the list for
   that league.

4. **`stats/{league}.py` → `get_training_matrix`** — ensure the method handles the new
   market key and produces a valid `(X, y)` pair.

5. **`stats/{league}.py` → `get_stats`** — ensure the method produces the same feature
   columns for a single offer as `get_training_matrix` produces for training.

6. Run `poetry run meditate --league {LEAGUE}` — the new market will train on the first
   pass. Review its row in `data/training/model_stats.csv` (CSV mirror of the parquet
   the dashboard reads).

---

## Modifying the Training Pipeline

The training loop is in `training/pipeline.py` → `train_market`. The stages in order:

1. **Data loading** — `Stats.get_training_matrix` → `data.trim_matrix`
2. **Distribution selection** — `calibration.select_distribution`; `global_mean >= 2`
   switches to SkewNormal
3. **Normalization** — SkewNormal targets are normalized by mean before fitting
4. **Hyperparameter search** — Optuna via `hyperparams.warm_start_hyper_opt`, seeded
   from the previous best params if the model pickle exists
5. **Model fit** — `LightGBMLSS.fit`
6. **Dispersion calibration** — `minimize_scalar` on CRPS loss over the validation set
7. **Temperature scaling** — Brier loss minimization on the validation set
8. **Diagnostics** — written into the model pickle alongside the model itself
9. **Save** — pickle to `data/models/{LEAGUE}_{market}.pkl`

To change the objective function, edit the `objective` closure inside `train_market`.
To change dispersion calibration, edit the `minimize_scalar` call that follows model fit.
To change the feature filter logic, see `training/shap.py`.

---

## Shipping to Production (`devel`)

**The production server tracks `devel` and pulls the entire branch.** Anything
merged to `devel` runs in production — including its dependencies and console
scripts. So `devel` must carry **production-runtime code and operator tools
only**, never the dev-only research scaffolding used to *decide* a change.

This is the process for shipping **any** model change for **any** market to
production. The ship mechanism — `data/config/stat_meta.json`'s per-cell
`shipped` field, the per-cell strategy resolve + withhold/prune in
`meditate`, and the Gate 1 / Gate 2 lifecycle — applies to **every
`(league, market)` cell**, not to one track or league. A research line
(e.g. the GBDT mean-regression work on
`claude/fix-gbdt-mean-regression-*`) is just where the *evidence* for a
given ship is produced; the pipeline itself is project-wide.

Model improvements are developed on a long-lived research branch that accumulates
diagnostics, A/B harnesses, and experiment flags. **Do not merge a research
branch wholesale into `devel`.** Ship in two phases.

### Phase A — foundation (one-time)

Land the production substrate the ship mechanism needs:

- the strategy registry (`training/baselines.py`) + pipeline `target_strategy` /
  `zinb_mode` dispatch + `model_prob` decode;
- the Gate 2 live-metrics machinery (`analysis.compute_book_brier_skill_score`,
  the `nightly.py` live-metrics step, `check-graduation`, `backfill-live-metrics`);
- the per-cell ship plumbing (`training/ship_config.py`, the `helpers/io.py`
  model-pickle helpers, the `meditate` wiring) + a `data/config/stat_meta.json`
  in which every cell starts `shipped: "withheld"`.

With every cell `"withheld"` and default flags, production behavior is
unchanged — the mechanism is simply now live.

### Phase B — per-market ship PRs (repeating)

When a `(league, market)` cell clears **Gate 1** (see
`docs/gbdt_mean_regression_plan.md`, "Ship mechanism — per-cell strategy config
on devel"), open a focused PR to `devel` carrying the **production delta only**:

1. **Training** — if the cell ships a *new* strategy: its slug + forward/decode in
   `training/baselines.py`. If it needs new features: those `stats/` + `pipeline`
   additions. If the strategy is already deployed: nothing here.
2. **Inference** — the matching `model_prob` decode branch for that strategy, with
   an inference-path test (a Gate 1 requirement).
3. **Toggle** — the one-field edit in `stat_meta.json` (set the cell's
   `shipped` to `"devel"` to ship, or back to `"withheld"` to dark-out a
   cell under rework).

The offline **evidence** that justifies the ship (compression-eval runs,
diagnostic verdicts) **travels as prose in the PR description, never as committed
code.**

### Keep on `devel` vs leave on the research branch

| Keep (production runtime / operator tools) | Leave off `devel` (dev-only research) |
|---|---|
| `baselines.py`, pipeline dispatch, `model_prob` decode, `training/scorecard.py` (inline-called by `report()`) | `training/scorecard.py` standalone-CLI A/B mode (single/diff/live-window flags) — keep the module, leave the research CLI exercises off |
| `ship_config.py` + `helpers/io.py` model-pickle helpers + `meditate` wiring | `zinb-routing-diagnostics` + the **`statsmodels`** dependency it pulls in |
| `nightly` live-metrics, `check-graduation`, `backfill-live-metrics` | `icc-diagnostics` |
| Production data / feature fixes | the diagnostics' test suites; any `/tmp` harness |

The `--target-strategy` / `--zinb-mode` / `--deterministic` flags default to
current production behavior, so they are inert on `devel`; the heavy determinism
*integration tests* are dev scaffolding and need not ship.

### Use the `devel-ship-curator` agent

Phase B PRs (and any further foundation layers) **must be carved by the
`devel-ship-curator` agent** (`.claude/agents/devel-ship-curator.md`), which
enforces the keep/drop split above and the production-delta-only discipline.
The initial Phase A foundation PR is the one exception — it is carved by hand.

---

## Tests

`tests/golden/` contains snapshot tests for every CLI command's `--help` output.

```bash
poetry run pytest tests/golden/         # run all golden tests
REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py  # regenerate snapshots
```

When you add a new CLI flag or rename an existing one, regenerate the affected snapshot
and commit the new fixture file alongside your code change.

There are no unit tests for model behavior — the golden tests guard the CLI surface and
the training report is the behavioral regression check.

### Integration smoke test

`tests/integration/test_end_to_end.py` exercises the
`confer -> meditate -> prophecize` wiring against cached Odds API fixtures
and stubbed external dependencies (`nba_api`, Optuna, Google Sheets, Underdog,
Sleeper). It is marked `integration` and is excluded from the default
`pytest` collection; opt in explicitly:

```bash
poetry run pytest -m integration         # fake mode (default; ~5s, no network)
SPORTSTRADAMUS_INTEGRATION_REAL_APIS=1 \
  poetry run pytest -m integration       # live Stats loaders (manual / on-demand)
```

The fake-mode run is **mandatory on every commit** — the
`integration-smoke` pre-commit hook in `.pre-commit-config.yaml` runs
`pytest -m integration` with `SPORTSTRADAMUS_INTEGRATION_REAL_APIS`
forcibly unset so the hook never hits the network. If your change touches
any module the test imports (the three CLIs, `Stats`, `Archive`, scoring,
parlay search, or sheet export), the smoke test will exercise the import
graph before the commit lands. The hook is set up by
`poetry run pre-commit install`.

The test never writes data: `Archive.write`, model pickle writes, and
prediction-history writes are all stubbed, and a `preserve_data_files`
fixture snapshots/restores the JSON config files in `src/sportstradamus/data/`
that ``meditate`` and ``confer`` would otherwise rewrite. Adding a new
CLI flag, renaming a stubbed function, or moving a CLI entry point will
likely break the smoke test — update its monkeypatches in lockstep with
the production change.

---

## Archived / Deprecated Code

Code that has no active callers lives in `src/deprecated/` rather than being deleted.
Each file carries an archive header:

```python
# ARCHIVED YYYY-MM-DD from src/sportstradamus/<original_path>
# Reason: <why it was removed>
# Last git SHA where it was live: <short sha>
```

See [`src/deprecated/README.md`](src/deprecated/README.md) for the full reintroduction
protocol. To reintroduce a deprecated module:

1. Copy the body back to its original path (or a new path if the structure has changed).
2. Remove the archive header.
3. Wire it into the appropriate caller.
4. Remove the `TODO` entry from `README.md`.
5. Delete the file from `src/deprecated/` if no other deprecated code remains there.