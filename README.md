# Sportstradamus

[![CI](https://github.com/tjjerome/Sportstradamus/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tjjerome/Sportstradamus/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)

Distributional sports-prop modeling: scrape sportsbook odds and player stats, train
models that predict the **full probability distribution** of each player stat — not
just a point estimate — and surface value recommendations in a Streamlit dashboard.
Supports MLB, NBA, NFL, NHL, and WNBA.

<!-- Dashboard screenshot goes here once captured:
![Dashboard](docs/img/dashboard.png) -->

> **Disclaimer.** Sportstradamus is an independent research project with no
> affiliation to any sportsbook, DFS platform, or sports league. Its output is
> not betting advice: sports betting involves real financial risk, and
> historical model performance never guarantees future results. Scraping and
> API access are subject to each data source's terms of service. The software
> is provided as-is, without warranty, under the [MIT license](LICENSE).

## How it works

1. **Collect** — `confer` fetches player-prop odds from the Odds API plus
   Underdog and Sleeper lines, into a DuckDB archive (`archive/archive.duckdb`).
2. **Enrich** — per-league `Stats` classes pull game logs from official league
   APIs, compute rolling features, and build KNN player-comparable feature sets.
3. **Train** — `meditate` fits a LightGBMLSS distributional regression per
   `(league, market)` cell, tunes hyperparameters with Optuna, calibrates against
   bookmaker lines, and only *ships* a cell when it clears six offline gates
   ([docs/ship_gate.md](docs/ship_gate.md)).
4. **Predict** — `prophecize` scores every live offer, computes expected value,
   and writes the parquet snapshots the dashboard reads.

Architecture detail — package map, data flow, adding a league or market — lives in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart

Requires **Python 3.11** (exact — PyTorch is pinned to a CPU wheel), **Poetry ≥ 1.7**, and Git.

```bash
git clone https://github.com/tjjerome/Sportstradamus.git
cd Sportstradamus
poetry install                  # first install downloads ~1-2 GB (CPU torch)
poetry run pre-commit install   # wires ruff lint/format into the commit hook

# Puts `sportstradamus` on PATH so the commands below work as written. The stub's
# shebang is absolute, so it needs no venv activation and runs from any directory.
ln -sf "$(poetry env info --path)/bin/sportstradamus" ~/.local/bin/sportstradamus
```

That symlink shortens the project CLI only; dev tools (`pytest`, `ruff`,
`pre-commit`) still go through `poetry run`.

Create `src/sportstradamus/creds/keys.json` (the `creds/` directory is
git-ignored — see [SECURITY.md](SECURITY.md) for the credential layout):

```json
{
  "odds_api": "<your key>",
  "scrapeops": "<your key>",
  "scrapingfish": "<your key>"
}
```

| Key | Service | Purpose |
|---|---|---|
| `odds_api` | [The Odds API](https://the-odds-api.com) | Player-prop lines + moneylines from major sportsbooks (free tier: 500 req/month) |
| `scrapeops` | [ScrapeOps](https://scrapeops.io) | Realistic browser-header rotation (free tier available) |
| `scrapingfish` | [ScrapingFish](https://scrapingfish.com) | Proxy fallback for hard endpoints (pay-as-you-go) |

Then run the pipeline once, in order:

```bash
sportstradamus confer                   # fetch odds (~2-5 min)
sportstradamus meditate --league NBA    # train one league (~10-30 min)
sportstradamus prophecize               # score offers, write snapshots
sportstradamus dashboard                # browse picks
```

The first `meditate` downloads and caches player game logs per league; later
runs fetch only new games.

## The CLI

Everything is one umbrella command — `sportstradamus <command>` (or
`python -m sportstradamus <command>`, which works even mid-reinstall):

```
sportstradamus
├── prophecize            score offers, write dashboard snapshots
├── confer                fetch current odds/props (--close-lines for closing lines)
├── meditate              train/retrain models (--league, --market, --force, ...)
├── reflect               nightly resolution: grade history, settle the ledger
├── dashboard             launch the Streamlit dashboard
├── bet
│   ├── kelly             re-size a recommendations YAML offline
│   ├── pickem            build Underdog Power/Flex/Rivals entries -> YAML
│   └── ledger-commit     twice-daily simulated-bettor ledger commit
├── fetch
│   ├── fp ...            Fantasy Points snapshots (NFL)
│   ├── ctg ...           Cleaning the Glass snapshots (NBA)
│   └── savant ...        Baseball Savant snapshots (MLB)
├── ship
│   ├── scorecard         offline six-gate A/B harness
│   ├── graduation        Gate-2 graduation check
│   ├── config            generate/validate the ship config
│   └── sweep             model-strategy sweep across cells
└── admin                 archive + data maintenance (merge-archives, ...)
```

`--help` on any node lists its flags. `meditate --help` shows the production
surface; the research axes are documented in
[docs/MODEL_LIFECYCLE.md](docs/MODEL_LIFECYCLE.md).

## Configuration

JSON configs live in `src/sportstradamus/data/config/`. The ones you might touch:

| File | Updated by | Purpose |
|---|---|---|
| `stat_meta.json` | manual + `meditate` | Per-cell `{dist, shipped, ...}` — distribution family and release surface (`withheld`/`devel`/`main`) |
| `stat_calibration.json` | `meditate` (gitignored) | Per-cell `{cv, std, zi}`, recomputed each run |
| `stat_map.json` | manual | Stat-name mappings across APIs and sportsbooks |
| `feature_filter.json` | manual | League-shared (`Common`) + per-market locked-in (`Always`) feature lists |
| `playerCompStats.json` | `scripts/optimize_comp_weights.py` | Learned player-comp weights per league/position |
| `book_weights.json` | `meditate` (gitignored) | Per-sportsbook reliability weights for consensus lines |
| `prop_books.json` | manual | Which sportsbooks to query per league |
| `odds_api_budget.json` | manual | Odds API credit-governor knobs |

Pre-computed correlation matrices land in
`src/sportstradamus/data/leagues/{league}/corr_*.parquet` (written by `meditate`).

## Data layout

```
src/sportstradamus/data/
├── models/                      # trained LightGBMLSS pickles ({LEAGUE}_{market}.pkl)
├── training_data/               # cached feature matrices
├── player_data/{LEAGUE}/{YEAR}/ # per-player game-log CSVs
├── leagues/{league}/            # correlation parquets
├── training/model_stats.parquet # per-cell training diagnostics (+ .csv mirror)
├── test_sets/                   # holdout test data
└── config/                      # JSON configs (table above)

src/sportstradamus/creds/
└── keys.json                    # API keys (git-ignored, create manually)
```

## Running it daily

- Cron schedule, job wrapper, healthchecks, dashboard-as-a-service:
  [docs/OPERATIONS.md](docs/OPERATIONS.md)
- Training cadence, six ship gates, and the `withheld` → `devel` → `main`
  release surfaces: [docs/MODEL_LIFECYCLE.md](docs/MODEL_LIFECYCLE.md)

## Contributing

Setup, quality gates, and PR expectations: [CONTRIBUTING.md](CONTRIBUTING.md).
Architecture orientation: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Vulnerability reports: [SECURITY.md](SECURITY.md).
