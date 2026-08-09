# Sportstradamus

A Python package that scrapes sportsbook odds and player stats, trains distributional ML models
that predict player performance, and surfaces value recommendations through a Streamlit
dashboard. Supports MLB, NBA, NFL, NHL, and WNBA.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [API Keys and Credentials](#api-keys-and-credentials)
5. [First-Run Setup](#first-run-setup)
6. [CLI Commands](#cli-commands)
7. [Operations](#operations)
8. [Configuration Files](#configuration-files)
9. [Data Storage Layout](#data-storage-layout)

---

## What it does

The system runs a four-stage pipeline:

1. **Collect** — `confer` fetches current player prop odds from the Odds API and supplemental
   lines from Underdog and Sleeper. Results are stored in a DuckDB archive at
   `archive/archive.duckdb`.
2. **Enrich** — The `Stats` classes (`StatsNBA`, `StatsMLB`, `StatsNFL`, `StatsNHL`,
   `StatsWNBA`) pull game logs from official league APIs, compute rolling features, and build
   player-comparable (KNN) feature sets.
3. **Train** — `meditate` fits a LightGBMLSS distributional regression model per market
   (e.g. "NBA: points", "NFL: receiving yards"), tunes hyperparameters with Optuna, and
   calibrates the output distribution against bookmaker lines.
4. **Predict** — `prophecize` loads trained models, scores every live offer against the model,
   computes expected value, and writes parquet snapshots (`current_offers.parquet`,
   `current_pickem.parquet`, `history.parquet`) that the Streamlit dashboard reads.

---

## Prerequisites

- **Python 3.11** (exact — PyTorch is pinned to a CPU wheel that requires it)
- **Poetry** ≥ 1.7 — [install](https://python-poetry.org/docs/#installation)
- **Git**

---

## Installation

```bash
git clone <this repo>
cd Sportstradamus
poetry install
poetry run pre-commit install   # wires ruff lint/format into the commit hook
```

`poetry install` pulls PyTorch CPU-only from a custom source defined in `pyproject.toml`.
The first install is slow (~1–2 GB download). Subsequent installs use the cache.

---

## API Keys and Credentials

Three external services need API keys. All secrets live in
`src/sportstradamus/creds/` — **this directory is git-ignored and must be
created manually**.

```
src/sportstradamus/creds/
└── keys.json           # API keys for Odds API, ScrapeOps, ScrapingFish
```

### `keys.json`

Create this file with the following structure:

```json
{
  "odds_api": "<your key>",
  "scrapeops": "<your key>",
  "scrapingfish": "<your key>"
}
```

| Key | Service | Purpose | Cost |
|---|---|---|---|
| `odds_api` | [The Odds API](https://the-odds-api.com) | Fetches player prop lines and moneylines from all major sportsbooks | Free tier: 500 req/month; paid tiers available |
| `scrapeops` | [ScrapeOps](https://scrapeops.io) | Rotates realistic browser headers on HTTP requests to avoid bot detection | Free tier: 1k req/month |
| `scrapingfish` | [ScrapingFish](https://scrapingfish.com) | Proxy/rendering fallback for harder-to-scrape endpoints | Pay-as-you-go |

---

## First-Run Setup

After installing and adding credentials, run the commands in this order:

```bash
# 1. Fetch current odds into the local archive (~2–5 minutes)
poetry run sportstradamus confer

# 2. Train models for one league to verify the pipeline (~10–30 minutes)
poetry run sportstradamus meditate --league NBA

# 3. Score current offers and write dashboard snapshots
poetry run sportstradamus prophecize
```

On the first `meditate` run, player game-log CSVs are downloaded from the league
APIs and cached under `src/sportstradamus/data/player_data/`. This takes several
minutes per league the first time; subsequent runs only fetch new games.

---

## CLI Commands

All commands are defined as Poetry scripts in `pyproject.toml`.

### `poetry run sportstradamus confer`

Fetches current player prop odds from the Odds API and supplemental lines from
Underdog and Sleeper. Writes results to the local archive.

```bash
poetry run sportstradamus confer
```

No flags. Typically run once per day before `prophecize`.

---

### `poetry run sportstradamus meditate`

Trains or retrains LightGBMLSS distributional models, one per market per league.
Reads game logs via the `Stats` classes, fits Optuna hyperparameter search,
calibrates against bookmaker lines, and writes model pickles to
`src/sportstradamus/data/models/`.

```bash
poetry run sportstradamus meditate                          # train only stale/missing models
poetry run sportstradamus meditate --league NBA             # one league only
poetry run sportstradamus meditate --market points,assists  # only the named market stem(s)
poetry run sportstradamus meditate --force                  # don't skip markets with fresh data
poetry run sportstradamus meditate --rebuild-correlations   # rebuild the {LEAGUE}_corr.csv matrices
# --help lists the rest: --target-normalization, --zinb-mode, --branch, --deterministic, --bypass-withholding
```

After training, per-cell diagnostics are written to
`src/sportstradamus/data/training/model_stats.parquet` (with a `model_stats.csv`
mirror) and surfaced on the dashboard's Model Training page. `training/report.py`
owns the schema; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline
that writes it and [docs/ship_gate.md](docs/ship_gate.md) for the gate columns.

---

### `poetry run sportstradamus prophecize`

Scores all current offers against trained models, computes expected value,
filters by confidence threshold, and writes parquet snapshots
(`current_offers.parquet`, `current_pickem.parquet`, `history.parquet`) that the
Streamlit dashboard reads.

```bash
poetry run sportstradamus prophecize
```

Requires `confer` to have been run at least once today and models to exist for
the active leagues.

---

### `poetry run sportstradamus reflect`

Analyzes historical parlay performance from the archive.

```bash
poetry run sportstradamus reflect
```

---

### `poetry run sportstradamus dashboard`

Launches a Streamlit dashboard for interactive review of picks and parlay
performance. Replaces `reflect` for visual exploration.

```bash
poetry run sportstradamus dashboard
```

---

### `poetry run sportstradamus bet pickem`

Underdog-native orchestrator covering Power, Flex, and Rivals contest
variants. Loads today's offers, filters by edge / disagreement / EV
thresholds, runs `beam_search_parlays` once per `(entry_size, variant)`
cross, sizes each candidate via `strategies/kelly.py`, and writes
`data/recommendations/{date}.yaml`.

```bash
poetry run sportstradamus bet pickem --date today --bankroll 500
```

Bankroll is a CLI flag only — there is no `data/bankroll.json`. Rivals
is restricted to 2- and 3-leg sizes regardless of `--entry-sizes`.

---

### `poetry run sportstradamus bet kelly`

Re-sizes a recommendations YAML offline. Reads the file produced by
`pickem-build`, applies fractional Kelly with the resolution chain
(explicit kwarg > live CLV-segment BSS > training BSS > fallback `1.0`),
and prints a tabulated stake table.

```bash
poetry run sportstradamus bet kelly --bankroll 500 --from data/recommendations/2026-05-08.yaml
```

cvxpy (SCS solver), pyyaml, and tabulate are required base dependencies
installed by `poetry install`; the `kelly` / `pickem-build` paths import
them at module top.

---

## Operations

- Training, ship gates, and the `withheld` → `devel` → `main` release
  surfaces: [docs/MODEL_LIFECYCLE.md](docs/MODEL_LIFECYCLE.md)
- Daily workflow and the production cron schedule:
  [docs/OPERATIONS.md](docs/OPERATIONS.md)

---

## Configuration Files

These files live under `src/sportstradamus/data/` (the JSON configs in
`data/config/`) and control model behavior. Most are updated automatically by
`meditate`; a few need manual attention.

| File | Updated by | Purpose |
|---|---|---|
| `stat_meta.json` | manual + `meditate` | Per-cell `{dist, shipped, strategy}` — distribution family, release surface (`withheld`/`devel`/`main`), training-strategy slug |
| `stat_calibration.json` | `meditate` (gitignored) | Per-cell `{cv, std, zi}` — coefficient of variation, spread, zero-inflation; recomputed each run |
| `stat_map.json` | manual | Stat name mappings across APIs and sportsbooks |
| `feature_filter.json` | manual | League-shared (`Common`) + per-market locked-in (`Always`) feature lists (the SHAP-ranked `Filtered` buckets were removed in the 2026-05-27 no-filter rewire) |
| `playerCompStats.json` | `scripts/optimize_comp_weights.py` | Learned comp-weight vectors per league/position |
| `book_weights.json` | `meditate` (`fit_book_weights`, gitignored) | Per-sportsbook reliability weights for consensus lines |
| `{LEAGUE}_corr.csv` | `meditate` (`correlate`) | Player stat correlation matrices for parlay EV |
| `prop_books.json` | manual | Which sportsbooks to query per league |
| `banned_combos.json` | manual | Player pairs excluded from parlay correlation |

---

## Data Storage Layout

```
src/sportstradamus/data/
├── models/                      # trained LightGBMLSS pickles ({LEAGUE}_{market}.pkl)
├── training_data/               # cached feature matrices ({LEAGUE}_{market}.csv)
├── player_data/{LEAGUE}/{YEAR}/ # per-player game log CSVs
├── test_sets/                   # holdout test data
├── training/model_stats.parquet # per-cell meditate diagnostics (+ .csv mirror)
├── config/                      # JSON config (see table above)
└── *.csv                        # correlation matrices, etc.

src/sportstradamus/creds/
└── keys.json                    # API keys (git-ignored, create manually)
```
