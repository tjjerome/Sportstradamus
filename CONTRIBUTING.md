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
10. [Tests](#tests)
11. [Archived / Deprecated Code](#archived--deprecated-code)

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
│   │   ├── feature_selection.py
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
| `config.py` | Loads every JSON config file at import time; exposes `stat_cv`, `stat_dist`, `stat_map`, `stat_zi`, `stat_std`, `book_weights`, `books`, `feature_filter`, `banned`, `abbreviations`, `combo_props`, `nhl_goalies`, `name_map`, `odds_api` |
| `archive.py` | `Archive` class — klepto HDF wrapper for persisting odds keyed by `(date, league, market, player)`. Methods: `get_line`, `get_ev`, `write`, `clip`. Also `clean_archive`. |
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
| `cli.py` | `meditate` click command — thin orchestrator: reset-markets handling, stats init, league setup (book weights, comps, correlations), per-market loop calling `train_market` |
| `pipeline.py` | `train_market(league, market, stat_data, ...)` — the full per-market training loop: data load, distribution selection, normalization, Optuna search, LightGBMLSS fit, dispersion calibration, temperature scaling, model save |
| `calibration.py` | `fit_book_weights`, `fit_model_weight`, `select_distribution` |
| `shap.py` | SHAP importance computation, feature filter management: `compute_market_importance`, `filter_market`, `filter_features`, `see_features`, `_scouting_shap_and_filter` |
| `correlate.py` | `correlate(league, stat_data)` — builds `{LEAGUE}_corr.csv` from player stat history |
| `report.py` | `report()` — reads model pickles, writes `training_report.txt` |
| `data.py` | `count_training_rows`, `trim_matrix`, `_histogram_weights` |
| `hyperparams.py` | `warm_start_hyper_opt`, `_BoundedResponseFn` |
| `markets.py` | `ALL_MARKETS` — per-league market name lists |
| `config.py` | `load/save_distribution_config`, `load/save_zi_config` |
| `__init__.py` | Re-exports the public API |

### `prediction/` — The `prophecize` pipeline

| Module | What's in it |
|---|---|
| `cli.py` | `main` click command — Google auth, stats init, fetch offers, score, write sheet. Also `_get_sheets_client` |
| `model_prob.py` | `model_prob` — loads a trained model, computes blended probability distributions for every offer |
| `scoring.py` | `process_offers`, `match_offers` — offer-level EV scoring and deduplication |
| `correlation.py` | `find_correlation` — loads correlation CSVs, scores parlay legs |
| `parlay.py` | `beam_search_parlays`, `save_data` |
| `sheets.py` | `write_to_sheet`, `format_sheet` — gspread write paths |
| `__init__.py` | Re-exports the public API |

### Other top-level modules

| Module | CLI command | What it does |
|---|---|---|
| `moneylines.py` | `confer` | Odds API scraper: `get_moneylines`, `get_props` |
| `books.py` | (called from `prediction/cli.py`) | Underdog (`get_ud`) and Sleeper (`get_sleeper`) scrapers |
| `nightly.py` | `reflect` | Historical parlay performance analysis |
| `dashboard.py` / `dashboard_app.py` | `dashboard` | Streamlit dashboard |
| `skew_normal.py` | — | Custom PyTorch `SkewNormal` distribution for LightGBMLSS |
| `feature_selection.py` | — | Feature selection helpers used during training |

---

## Data Flow

```
confer
  moneylines.get_props()       ← Odds API
  books.get_ud()               ← Underdog API
  books.get_sleeper()          ← Sleeper API
        │
        ▼
  Archive.write()  →  data/   (klepto HDF)

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
  prediction/sheets.write_to_sheet()  →  Google Sheets
```

---

## Where to Find Things

| I want to... | Look here |
|---|---|
| Change which markets a league trains | `training/markets.py` → `ALL_MARKETS` |
| Change which distribution a stat uses | `src/sportstradamus/data/stat_dist.json` |
| Add/remove a sportsbook from consensus lines | `src/sportstradamus/data/prop_books.json` |
| Add a player name alias | `src/sportstradamus/data/name_map.json` |
| Understand a training report metric | [CLAUDE.md](CLAUDE.md) §Training Report Diagnostics |
| Change the confidence cutoff for picks | `prediction/scoring.py` → `MIN_CONFIDENCE` |
| Change the Optuna hyperparameter search space | `training/pipeline.py` → `train_market` objective |
| Change how distributions are blended with bookmaker lines | `helpers/distributions.py` → `fused_loc` |
| Update the season start date for a league | `stats/{league}.py` → `Stats{League}.season_start` |
| Change SHAP importance thresholds | `training/shap.py` → `filter_features` / run `meditate --rebuild-filter` |
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

1. Work on a feature branch off `main`.
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
   the `training_report.txt` values look plausible.
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

The full rules are in [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md). Quick reference:

- **Formatter:** `ruff format`. Line length 100. Double quotes.
- **Linter:** `ruff check`. Enforces PEP 8, import order, pydocstyle (Google convention),
  bugbear, and more.
- **Docstrings:** Every public function and class. Google format (Args / Returns / Raises).
  Module-level docstring on every file under `src/sportstradamus/`.
- **Type hints:** Required on all public function signatures.
- **Naming:** `snake_case` for modules/functions/variables, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for module-level constants.
- **Comments:** Explain *why*, not *what*. No commented-out code — delete it and let
  `git log` keep the history. If code will return, move it to `src/deprecated/` instead.
- **Function length:** Target ≤ 60 logical lines. Hard cap ~120. Extract helpers when
  nesting exceeds 4 levels.
- **No speculative abstraction.** Three similar lines of code are better than a premature
  abstraction. Extract only after the third concrete reuse.

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

4. **Add distribution types** in `data/stat_dist.json` for each new market.

5. **Add stat name mappings** in `data/stat_map.json` to normalize API names to
   internal names.

6. **Add abbreviations** to `data/abbreviations.json`.

7. **Wire into `meditate`** — `training/cli.py` already reads `ALL_MARKETS` and
   instantiates each league's Stats class dynamically; if your class follows the naming
   convention `Stats{LEAGUE}` (e.g. `StatsXFL`) it will be picked up automatically.

8. Add a `--league` option value in `training/cli.py` → the `click.Choice` list.

---

## Adding a New Market

A "market" is a betting category for a single stat in a single league (e.g. "assists",
"strikeouts"). To add one:

1. **`data/stat_dist.json`** — add `"{LEAGUE}: {market}": "Gamma"` (or NegBin / ZINB /
   ZAGamma based on whether the stat is continuous or count-valued and whether it has
   zero-inflation).

2. **`data/stat_map.json`** — map the sportsbook's name for the stat to your internal name.

3. **`training/markets.py` → `ALL_MARKETS`** — append the market string to the list for
   that league.

4. **`stats/{league}.py` → `get_training_matrix`** — ensure the method handles the new
   market key and produces a valid `(X, y)` pair.

5. **`stats/{league}.py` → `get_stats`** — ensure the method produces the same feature
   columns for a single offer as `get_training_matrix` produces for training.

6. Run `poetry run meditate --league {LEAGUE}` — the new market will train on the first
   pass. Review its block in `training_report.txt`.

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
