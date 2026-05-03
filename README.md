# Sportstradamus

A Python package that scrapes sportsbook odds and player stats, trains distributional ML models
that predict player performance, and exports value recommendations to Google Sheets. Supports
MLB, NBA, NFL, NHL, and WNBA.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [API Keys and Credentials](#api-keys-and-credentials)
5. [First-Run Setup](#first-run-setup)
6. [CLI Commands](#cli-commands)
7. [Daily Workflow](#daily-workflow)
8. [Configuration Files](#configuration-files)
9. [Data Storage Layout](#data-storage-layout)
10. [Deferred / Archived Code](#deferred--archived-code)

---

## What it does

The system runs a four-stage pipeline:

1. **Collect** — `confer` fetches current player prop odds from the Odds API and supplemental
   lines from Underdog and Sleeper. Results are stored in a klepto HDF archive on disk.
2. **Enrich** — The `Stats` classes (`StatsNBA`, `StatsMLB`, `StatsNFL`, `StatsNHL`,
   `StatsWNBA`) pull game logs from official league APIs, compute rolling features, and build
   player-comparable (KNN) feature sets.
3. **Train** — `meditate` fits a LightGBMLSS distributional regression model per market
   (e.g. "NBA: points", "NFL: receiving yards"), tunes hyperparameters with Optuna, and
   calibrates the output distribution against bookmaker lines.
4. **Predict** — `prophecize` loads trained models, scores every live offer against the model,
   computes expected value, and exports picks to a Google Sheet.

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

Four external services need credentials. All secrets live in
`src/sportstradamus/creds/` — **this directory is git-ignored and must be
created manually**.

```
src/sportstradamus/creds/
├── keys.json           # API keys for Odds API, ScrapeOps, ScrapingFish
├── credentials.json    # Google OAuth client secret (downloaded from Cloud Console)
└── token.json          # Auto-generated on first prophecize run — do not create manually
```

### 1. `keys.json`

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

### 2. Google Sheets credentials (`credentials.json`)

`prophecize` writes picks to a Google Sheet. To enable this:

1. Go to [Google Cloud Console](https://console.cloud.google.com) and create a project.
2. Enable the **Google Sheets API** and the **Google Drive API** for the project.
3. Create an **OAuth 2.0 Client ID** (type: Desktop application).
4. Download the client secret JSON and save it as
   `src/sportstradamus/creds/credentials.json`.
5. On the first `prophecize` run, a browser window will open asking you to authorize
   the app. After you approve, a `token.json` is written automatically and subsequent
   runs are silent.

The app requests these scopes:
- `https://www.googleapis.com/auth/spreadsheets`
- `https://www.googleapis.com/auth/drive`

---

## First-Run Setup

After installing and adding credentials, run the commands in this order:

```bash
# 1. Fetch current odds into the local archive (~2–5 minutes)
poetry run confer

# 2. Train models for one league to verify the pipeline (~10–30 minutes)
poetry run meditate --league NBA

# 3. Score current offers and export to Google Sheets
poetry run prophecize
```

On the first `meditate` run, player game-log CSVs are downloaded from the league
APIs and cached under `src/sportstradamus/data/player_data/`. This takes several
minutes per league the first time; subsequent runs only fetch new games.

---

## CLI Commands

All commands are defined as Poetry scripts in `pyproject.toml`.

### `poetry run confer`

Fetches current player prop odds from the Odds API and supplemental lines from
Underdog and Sleeper. Writes results to the local archive.

```bash
poetry run confer
```

No flags. Typically run once per day before `prophecize`.

---

### `poetry run meditate`

Trains or retrains LightGBMLSS distributional models, one per market per league.
Reads game logs via the `Stats` classes, fits Optuna hyperparameter search,
calibrates against bookmaker lines, and writes model pickles to
`src/sportstradamus/data/models/`.

```bash
poetry run meditate                          # train only stale/missing models
poetry run meditate --force                  # retrain every model
poetry run meditate --league NBA             # one league only
poetry run meditate --rebuild-filter         # full feature set → rerun SHAP → rewrite filter
poetry run meditate --reset-markets NBA:points,NBA:assists
```

After training, a diagnostic report is written to
`src/sportstradamus/data/training_report.txt`. See [CLAUDE.md](CLAUDE.md) for a
full explanation of every metric in that report.

---

### `poetry run prophecize`

Scores all current offers against trained models, computes expected value,
filters by confidence threshold, and exports picks to Google Sheets.

```bash
poetry run prophecize
```

Requires `confer` to have been run at least once today and models to exist for
the active leagues.

---

### `poetry run reflect`

Analyzes historical parlay performance from the archive.

```bash
poetry run reflect
```

---

### `poetry run dashboard`

Launches a Streamlit dashboard for interactive review of picks and parlay
performance. Replaces `reflect` for visual exploration.

```bash
poetry run dashboard
```

---

## Daily Workflow

```
morning:
  poetry run confer          # pull today's lines
  poetry run prophecize      # score + export to Sheets

weekly (or when model accuracy drops):
  poetry run meditate        # retrain stale models
```

If a new season has started and the season-start date in the relevant Stats class
has not been updated, `meditate` will skip the league. Update
`src/sportstradamus/stats/{league}.py` → `Stats{League}.season_start`.

---

## Configuration Files

These files live in `src/sportstradamus/data/` and control model behavior.
Most are updated automatically by `meditate`; a few need manual attention.

| File | Updated by | Purpose |
|---|---|---|
| `stat_dist.json` | manual / `meditate` | Distribution type per stat (Gamma, NegBin, ZINB, ZAGamma, SkewNormal) |
| `stat_cv.json` | manual | Coefficient of variation per stat — controls spread of Gamma/NegBin priors |
| `stat_zi.json` | `meditate` | Zero-inflation gate parameters per stat |
| `stat_map.json` | manual | Stat name mappings across APIs and sportsbooks |
| `feature_filter.json` | `meditate --rebuild-filter` | SHAP-based feature importance thresholds |
| `playerCompStats.json` | `scripts/optimize_comp_weights.py` | Learned comp-weight vectors per league/position |
| `book_weights.json` | `meditate` (`fit_book_weights`) | Per-sportsbook reliability weights for consensus lines |
| `{LEAGUE}_corr.csv` | `meditate` (`correlate`) | Player stat correlation matrices for parlay EV |
| `prop_books.json` | manual | Which sportsbooks to query per league |
| `banned_combos.json` | manual | Player pairs excluded from parlay correlation |

---

## Data Storage Layout

```
src/sportstradamus/data/
├── models/                     # trained LightGBMLSS pickles ({LEAGUE}_{market}.pkl)
├── training_data/              # cached feature matrices ({LEAGUE}_{market}.csv)
├── player_data/{LEAGUE}/{YEAR}/ # per-player game log CSVs
├── test_sets/                  # holdout test data
├── training_report.txt         # latest meditate diagnostic output
└── *.json / *.csv              # config files (see table above)

src/sportstradamus/creds/
├── keys.json                   # API keys (git-ignored, create manually)
├── credentials.json            # Google OAuth client secret (git-ignored)
└── token.json                  # auto-generated OAuth token (git-ignored)
```

---

## Deferred / Archived Code

The following modules were moved to [`src/deprecated/`](src/deprecated/) during
the 2026-04-21 maintainability refactor because they had no caller in any CLI
entry point or `scripts/` module. They are preserved verbatim under
`src/deprecated/` (with a header comment recording the original path and the
last live git SHA) and should be reintroduced if the corresponding feature
returns. See [`src/deprecated/README.md`](src/deprecated/README.md) for the
header protocol and reintroduction process.

- [ ] **TODO: reimplement Kelly bet sizing** (`opt_kelley_bet.py`) — stake
      optimization on +EV picks. Needs integration with `prophecize`'s sheet
      export or a new `kelly` subcommand.
- [ ] **TODO: reimplement parlay search** (`unused_funcs.py::find_bets`,
      `opt_parlay.py`) — combinatorial search over +EV legs. May have been
      replaced by in-line parlay logic in `sportstradamus.py`; decide whether
      to delete or rewire.
- [ ] **TODO: reimplement BettingPros NFL ingest** (`get_lines.py`) —
      redundant with `books.py` scrapers, but a useful fallback.
- [ ] **TODO: reimplement team correlation generator** (`correlation.py`) —
      produces `{LEAGUE}_corr.csv` consumed by `sportstradamus.py`'s parlay
      logic. Those CSVs are currently stale; a script or CLI is needed.
- [ ] **TODO: reimplement LightGBM feature-importance plot**
      (`see_features.py`).
- [ ] **TODO: reimplement testing utilities** (`test.py`) — ad-hoc
      experimentation harness, not pytest tests. Decide whether to convert to
      proper tests or delete.
- [ ] **TODO: decide fate of orphaned `helpers.py` math utilities**
      (`prob_diff`, `prob_sum`, `accel_asc`, `get_active_sports`) — preserved
      in [`src/deprecated/helpers_orphans.py`](src/deprecated/helpers_orphans.py).
- [ ] **TODO: orphan methods** (`Archive.add`, `Archive.clip`, `Archive.merge`,
      `Archive.rename_market`, `Scrape.get_proxy`, `Scrape.post`) — preserved
      in [`src/deprecated/helpers_orphans.py`](src/deprecated/helpers_orphans.py)
      as de-methodized top-level functions. `Archive.add` looks like the
      intended write path for the `confer` pipeline but was never wired in;
      decide whether to wire it or delete.
- [ ] **TODO: Stats orphan methods** (21 methods across all league subclasses) —
      preserved in [`src/deprecated/stats_deprecated.py`](src/deprecated/stats_deprecated.py).
      The `obs_*` family (`obs_get_stats`, `obs_get_training_matrix`, `obs_profile_market`,
      `dvpoa`, `bucket_stats`) was an older per-observation prediction API superseded by
      the current vectorized offer-based API. `get_fantasy` (StatsNFL) was unused NFL
      fantasy scoring logic. Reintroduce if the obs_* API is revived for analysis tooling.
- [ ] **TODO: deprecated sportsbook scrapers** (`get_dk`, `get_fd`,
      `get_pinnacle`, `get_caesars`, `get_thrive`, `get_pp`, `get_parp`)
      preserved in [`src/deprecated/books_deprecated.py`](src/deprecated/books_deprecated.py).
      Superseded by `moneylines.get_props`. Reintroduce only if direct-book
      scraping becomes preferable to the odds aggregator. The remaining live
      scrapers (`get_ud` for Underdog, `get_sleeper` for Sleeper) stay in
      `books.py`.
