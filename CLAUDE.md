# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Mandatory reading — do this first

Before touching any code, read these two documents once:

1. **[CONTRIBUTING.md](CONTRIBUTING.md)** — package map, data flow, where to find things,
   how to make changes, how to add a league or market. Required reading. Not optional.
2. **[docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md)** — formatting, naming, docstrings, type
   hints, dead-code rules. Cite sections by number in commits and comments.

If you skip these and make changes that violate the layout or style, you will be asked
to redo the work.

## General Rules

* Talk like caveman except when /writing-clearly-and-concisely overrides
* Use `click` over `argparse` for CLI args
* Long-running scripts: add a status bar with `tqdm`
* One module per session. Commit and start fresh before moving to the next file.
* Before claiming anything is "done", run `poetry run pytest tests/golden/` and
  `poetry run ruff check src/sportstradamus/`. Both must be clean.

## Hard rules — these caused the last major refactor

The codebase was refactored from several 1,000–7,000 line monoliths into packages.
Do not undo that work:

* **No new monoliths.** If a file you are editing exceeds ~300 lines, stop and check
  whether you are adding to the right module. Consult CONTRIBUTING.md §Package Map.
* **No back-compat shims.** The old `train.py`, `sportstradamus.py`, and `stats.py`
  shims have been deleted. Import from the canonical package paths:
  - Stats classes → `sportstradamus.stats`
  - Training pipeline → `sportstradamus.training`
  - Prediction pipeline → `sportstradamus.prediction`
  - Shared utilities → `sportstradamus.helpers`
* **No commented-out code.** Delete it. If it might return, move it to `src/deprecated/`
  with the archive header (see `src/deprecated/README.md`).
* **No orphan methods.** Before finishing any work that removes a caller, grep for all
  call sites of the affected method. Zero-caller methods go to `src/deprecated/`, not
  into the next refactor's surprise pile.
* **No magic numbers.** Named constants at module level with a one-line reason comment.
  See STYLE_GUIDE.md §8.

## Commands

```bash
# Install dependencies
poetry install
poetry run pre-commit install   # required once after clone

# CLI entry points
poetry run prophecize        # prediction pipeline → Google Sheets
poetry run confer            # fetch current odds/props
poetry run meditate          # train/retrain ML models
poetry run reflect           # historical parlay performance
poetry run dashboard         # Streamlit dashboard

# Quality gates — both must pass before committing
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/

# Regenerate CLI help snapshots after an intentional flag change
REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py
```

Python 3.11 required. PyTorch CPU-only (2.1.2) via custom Poetry source.

## Package structure (canonical paths)

The old single-file modules no longer exist. Use these paths:

| What you need | Import from |
|---|---|
| `Stats`, `StatsNBA`, `StatsMLB`, `StatsNFL`, `StatsNHL`, `StatsWNBA` | `sportstradamus.stats` |
| `Archive`, `Scrape`, `fused_loc`, `get_ev`, config dicts | `sportstradamus.helpers` |
| `meditate` CLI, `train_market`, `report`, `correlate` | `sportstradamus.training` |
| `main` (prophecize) CLI, `model_prob`, `find_correlation` | `sportstradamus.prediction` |
| `confer`, `get_props`, `get_moneylines` | `sportstradamus.moneylines` |
| `get_ud`, `get_sleeper` | `sportstradamus.books` |

Full per-submodule breakdown is in CONTRIBUTING.md §Package Map.

## Architecture

### Data Pipeline

1. **Collection** (`books.py`, `moneylines.py`): Scrapes Underdog and Sleeper directly;
   fetches all other sportsbook props via the Odds API. Uses `Scrape` helper with
   ScrapeOps header rotation.
2. **Enrichment** (`stats/`): `Stats` subclasses fetch player game logs from league APIs
   (mlb-statsapi, nba-api, nfl-data-py), compute rolling features, and build KNN
   player-comparable feature sets.
3. **Training** (`training/`): `train_market` in `training/pipeline.py` builds feature
   matrices per market, tunes LightGBMLSS with Optuna, calibrates against bookmaker
   lines, and writes model pickles.
4. **Prediction** (`prediction/`): `model_prob` loads trained models, `process_offers`
   scores each offer for EV, `find_correlation` scores parlay legs, `sheets.py` exports
   to Google Sheets via gspread.

### Core Class Hierarchy

**`Stats`** (`stats/base.py`) → `StatsNBA`, `StatsWNBA`, `StatsMLB`, `StatsNFL`, `StatsNHL`

Key methods:
- `load()` / `update()` — load/fetch game logs from league APIs
- `get_training_matrix(market)` — feature matrix (X) and targets (y) for one market
- `get_stats(offer, game_date)` — feature vector for a single prediction
- `profile_market()` — aggregate stats for defense/offense profiling

**`Archive`** (`helpers/archive.py`): klepto HDF wrapper persisting odds by
`(date, league, market, player)`. Methods: `get_line()`, `get_ev()`, `write()`, `clip()`.

**`Scrape`** (`helpers/scraping.py`): HTTP client with ScrapeOps browser-header rotation
and ScrapingFish proxy fallback.

### ML Pipeline

**LightGBMLSS** for distributional regression — predicts full probability distributions.

Distribution types (set per stat in `data/stat_dist.json`):
- **Gamma** / **ZAGamma** — continuous stats, optional zero-inflation
- **Negative Binomial** / **ZINB** — count stats, optional zero-inflation
- **SkewNormal** (`skew_normal.py`) — custom PyTorch distribution; used when
  `global_mean >= 2` in `training/pipeline.py:train_market`

**Player Comparables**: z-scored profiles → weighted BallTree KNN → comp outcomes as
features. Weights optimized via `scripts/optimize_comp_weights.py`.

**Feature filtering**: SHAP-based importance thresholds in `feature_filter.json`,
updated by `meditate --rebuild-filter` via `training/shap.py`.

### Training Report Diagnostics (`data/training_report.txt`)

Generated by `training/report.py:report()` after `meditate`. Each market block:

**Header**: Distribution type (NegBin/Gamma/ZINB/ZAGamma/SkewNormal) + historical zero rate.

**Performance Table** (row 1 = raw LightGBMLSS, row 2 = after model correction
(`fit_model_weight` + `dispersion_cal`), row 3 = after temperature scaling; all filtered
at confidence > 0.54):

| Metric | Meaning |
|--------|---------|
| Accuracy | `accuracy_score` on over/under classification |
| Over Prec | Precision of "Over" predictions |
| Under Prec | Precision of "Under" predictions |
| Over% | Proportion of "Over" predictions |
| Sharpness | Std dev of predicted probabilities (lower = more decisive) |
| NLL | Negative log-loss |

**DIAG — Model Blending & Calibration**:
- `model_weight`: Blend weight [0.05–0.9] model vs bookmaker, optimized via log-likelihood. Low = bookmaker dominates. Logarithmic opinion pool for NegBin, precision-weighted blend for Gamma.
- `model_calib`: `1 - mean((calibrated_prob - actual)²)` on validation set post-temperature scaling. Higher = better.

**DIAG — Shape / Dispersion**:
- `start_r` / `start_alpha`: Method-of-moments shape estimate from training features (NegBin: `μ²/(σ²-μ)`, Gamma: `(μ/σ)²`).
- `model_r` / `model_alpha`: Mean shape parameter from LightGBMLSS (NegBin: `total_count`, Gamma: `concentration`).
- `empirical_r` / `empirical_alpha`: Shape from actual test outcomes — ground truth dispersion.
- `shape_ratio`: `model_shape / empirical_shape`. >1 = over-dispersed, <1 = under-dispersed, 1.0 = ideal.
- `shape_ceiling`: Upper bound on shape during training (2× `marginal_shape`).
- `marginal_shape`: Empirical shape from full training targets.
- `dispersion_cal`: Post-hoc shape scaling [0.1–1.0] learned on validation set. 1.0 = no correction, <1.0 = model over-dispersed.

**DIAG — Expected Value**:
- `start_mean`: Average of training feature `MeanYr`.
- `model_ev`: Mean blended expected value (model + bookmaker fusion via `fused_loc()`).
- `mean_line`: Average bookmaker line.
- `result_mean`: Actual test set mean outcome.
- `mean_ev_diff` / `median_ev_diff`: Blended EV excess over bookmaker line (mean/median).
- `frac_ev>line`: Fraction where model EV > bookmaker line (~50% = well-calibrated).

**DIAG — Conditional Prediction Quality**:
- `Over%|ev>line`: Over rate when model EV > line (should be high).
- `Over%|ev<line`: Over rate when model EV < line (should be low).
- `CF_Over%(emp_shape)`: Counterfactual Over% using empirical shape vs model shape.

**HP — Hyperparameters** (Optuna tuning):
- `rounds`: Boosting iterations | `leaves`: Max leaves per tree | `lr`: Learning rate
- `min_child`: Min samples per leaf | `L1`/`L2`: Regularization

**Quick interpretation**:
- Sharpness low + NLL low = good confidence separation
- `Over%|ev>line` high & `Over%|ev<line` low = properly discriminative
- `shape_ratio` ≈ 1.0 = well-calibrated dispersion
- `dispersion_cal` = 1.0 = no fix needed; <1.0 = model over-dispersed
- `model_weight` near 0 = bookmaker dominates; near 1 = model dominates

### Key Configuration Files (`src/sportstradamus/data/`)

| File | Purpose |
|------|---------|
| `stat_dist.json` | Distribution type per stat (Gamma, NegBin, ZINB, etc.) |
| `stat_cv.json` | Coefficient of variation per stat |
| `stat_zi.json` | Zero-inflation parameters |
| `stat_map.json` | Stat name mappings across APIs/sportsbooks |
| `feature_filter.json` | SHAP-based feature importance thresholds |
| `playerCompStats.json` | Learned player comp weights per league/position |
| `book_weights.json` | Sportsbook reliability weights for consensus lines |
| `{LEAGUE}_corr.csv` | Pre-computed player stat correlation matrices |

### Data Storage

- `data/models/` — Trained LightGBMLSS model pickles
- `data/training_data/` — Cached training matrices
- `data/player_data/{LEAGUE}/{YEAR}/` — Historical game log CSVs
- `data/test_sets/` — Holdout test data
- `creds/` — API keys (`keys.json`), Google OAuth (`credentials.json`, `token.json`)
