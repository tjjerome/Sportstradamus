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

1. **Collect** — `confer` fetches sportsbook player-prop and game-line odds from
   the Odds API into a DuckDB archive (`archive/archive.duckdb`).
2. **Enrich** — per-league `Stats` classes pull game logs from official league
   APIs, compute rolling features, and build KNN player-comparable feature sets.
3. **Train** — `meditate` fits a LightGBMLSS distributional regression per
   `(league, market)` cell, tunes hyperparameters with Optuna, calibrates against
   bookmaker lines, and only *ships* a cell when it clears six offline gates
   ([docs/ship_gate.md](docs/ship_gate.md)).
4. **Predict** — `prophecize` scrapes the live Underdog and Sleeper offers
   (archiving their lines), scores each one, computes expected value, and writes
   the parquet snapshots the dashboard reads.

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
  "odds_api_plus": "<your key>",
  "odds_api": "<your key>",
  "scrapeops": "<your key>",
  "scrapingfish": "<any value>"
}
```

| Key | Service | Purpose |
|---|---|---|
| `odds_api_plus` | [The Odds API](https://the-odds-api.com) | The key `confer` polls with: player-prop lines and game lines from the major sportsbooks, paced by the credit governor (`odds_api_budget.json`) |
| `odds_api` | [The Odds API](https://the-odds-api.com) | Fallback key `confer --close-lines` switches to when the plus key runs out; read at import, so it must be present |
| `scrapeops` | [ScrapeOps](https://scrapeops.io) | Realistic browser-header rotation (free tier available) |
| `scrapingfish` | [ScrapingFish](https://scrapingfish.com) | Read when `Scrape()` is built but unused since the proxy path was archived; any placeholder value works |

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
├── export-line-movement  rebuild the Board's DFS line-movement snapshot
├── bet
│   ├── kelly             re-size a recommendations YAML offline
│   ├── pickem            build Underdog/Sleeper Power/Flex entries -> YAML
│   └── ledger-commit     twice-daily simulated-bettor ledger commit
├── fetch
│   ├── fp ...            Fantasy Points snapshots (NFL)
│   ├── ctg ...           Cleaning the Glass snapshots (NBA)
│   ├── savant ...        Baseball Savant snapshots (MLB)
│   └── headshots         player-headshot cache
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
| `stat_meta.json` | manual + `ship sweep --confirm` / `ship config --branch main` | Per-cell `{dist, shipped, ...}` — distribution family and release surface (`withheld`/`devel`/`main`) |
| `stat_calibration.json` | `meditate` (gitignored) | Per-cell `{cv, std, zi, book_shape}`, recomputed each run |
| `stat_map.json` | manual | Stat-name mappings across APIs and sportsbooks |
| `feature_filter.json` | manual | League-shared (`Common`) + per-market locked-in (`Always`) feature lists |
| `playerCompStats.json` | `src/sportstradamus/scripts/optimize_comp_weights.py` | Learned player-comp weights per league/position |
| `book_weights.json` | `meditate` (gitignored) | Per-sportsbook reliability weights for the consensus price. Sportsbooks only: a DFS pick'em platform gets no weight |
| `odds_api_budget.json` | manual | Odds API credit-governor knobs |

Pre-computed correlation matrices land in
`src/sportstradamus/data/leagues/{league}/corr_*.parquet` (written by `meditate`).

## Data layout

```
src/sportstradamus/data/
├── models/                      # trained LightGBMLSS pickles ({LEAGUE}_{market-slug}.mdl)
├── training_data/               # cached feature matrices
├── player_data/                 # roster seeds, MLB affinity CSVs, collector snapshots
├── leagues/{league}/            # gamelog/teamlog parquets, players.json, correlation parquets
├── runtime/                     # prediction snapshots, history, parlay store, live metrics
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
