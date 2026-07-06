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
7. [Training and Shipping a New Model](#training-and-shipping-a-new-model)
8. [Daily Workflow](#daily-workflow)
9. [Configuration Files](#configuration-files)
10. [Data Storage Layout](#data-storage-layout)
11. [Deferred / Archived Code](#deferred--archived-code)

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
poetry run confer

# 2. Train models for one league to verify the pipeline (~10–30 minutes)
poetry run meditate --league NBA

# 3. Score current offers and write dashboard snapshots
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
poetry run meditate --league NBA             # one league only
poetry run meditate --market points,assists  # only the named market stem(s)
poetry run meditate --force                  # don't skip markets with fresh data
poetry run meditate --rebuild-correlations   # rebuild the {LEAGUE}_corr.csv matrices
# --help lists the rest: --target-normalization, --zinb-mode, --branch, --deterministic, --bypass-withholding
```

After training, per-cell diagnostics are written to
`src/sportstradamus/data/training/model_stats.parquet` (with a `model_stats.csv`
mirror) and surfaced on the dashboard's Model Training page. See
[CLAUDE.md](CLAUDE.md) §Training stats for a full explanation of every column.

---

### `poetry run prophecize`

Scores all current offers against trained models, computes expected value,
filters by confidence threshold, and writes parquet snapshots
(`current_offers.parquet`, `current_pickem.parquet`, `history.parquet`) that the
Streamlit dashboard reads.

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

### `poetry run pickem-build`

Underdog-native orchestrator covering Power, Flex, and Rivals contest
variants. Loads today's offers, filters by edge / disagreement / EV
thresholds, runs `beam_search_parlays` once per `(entry_size, variant)`
cross, sizes each candidate via `strategies/kelly.py`, and writes
`data/recommendations/{date}.yaml`.

```bash
poetry run pickem-build --date today --bankroll 500
```

Bankroll is a CLI flag only — there is no `data/bankroll.json`. Rivals
is restricted to 2- and 3-leg sizes regardless of `--entry-sizes`.

---

### `poetry run kelly`

Re-sizes a recommendations YAML offline. Reads the file produced by
`pickem-build`, applies fractional Kelly with the resolution chain
(explicit kwarg > live CLV-segment BSS > training BSS > fallback `1.0`),
and prints a tabulated stake table.

```bash
poetry run kelly --bankroll 500 --from data/recommendations/2026-05-08.yaml
```

cvxpy (SCS solver), pyyaml, and tabulate are required base dependencies
installed by `poetry install`; the `kelly` / `pickem-build` paths import
them at module top.

---

## Training and Shipping a New Model

Each model covers one **cell** — a single league-and-market pair, such as *NBA points*. Before a
cell's model is allowed to serve real recommendations it has to clear the **ship gates**: a fixed
battery of automatic quality checks (Is it as accurate as the sportsbook? Are its predictions
unbiased and well-calibrated?). A cell's release status is the `shipped` field in `stat_meta.json`,
and it moves through three stages:

- `withheld` — not served; still being worked on.
- `devel` — served on the tracking server and watched on real settled bets.
- `main` — fully live.

The path from "I want to improve a cell" to "it's live" is five steps.

### 1. Sweep the cell's strategy options

`model-strategy-sweep` trains a quick throwaway model for every combination of the knobs that move
the cell's distribution family (SkewNormal sweeps target shape × training loss × blend loss; ZINB
sweeps count mode × dispersion objective × blend loss) and ranks them by how comfortably each would
clear the ship gates — the margin it calls **slack**.

```bash
poetry run model-strategy-sweep --league NBA --market FGM   # one cell
poetry run model-strategy-sweep --board                     # every withheld cell with cached data
poetry run model-strategy-sweep --board --league WNBA       # just one league's withheld cells
poetry run model-strategy-sweep --board --include-shipped   # also re-check already-shipped cells
```

`--board` sweeps every **withheld** cell in `stat_meta.json` — both SkewNormal and ZINB — that has a
cached training matrix, and prints an up-front count of how many cells (and trainings) that is;
`--league` narrows it, and naming a single `--league` / `--market` sweeps just that cell. A cell with
no cached matrix is **skipped with a yellow warning** rather than swept — the throwaway trainings
reuse the cached matrix and never rebuild one, so train the cell for real once first if you want it in
the board. Add `--include-shipped` to also rank already-shipped (devel/main) cells when hunting a
better strategy for a live cell; that path is judged by the supersession test, and `--confirm` never
auto-re-ships a live cell. As it runs it prints one line per combination, then a short table per cell
with the winning strategy marked `SHIP` (green) or `KILL` (red). The full ranked results are saved to
`data/research/strategy_research_board.csv`. **Nothing ships from this step** — the throwaway models
only *rank* the options so you know which one to train for real.

Prefer to skip the manual walkthrough below? Add `--confirm`: for each cell it persists the best
reproducible winner to `stat_meta.json`, retrains it for real, and keeps or reverts it on the
official gates — steps 2–4 automated, with a prompt before it touches anything (`--yes` to skip it).

### 2. Confirm the winner with a real training run

The board's top row names the winner in three columns, but only two of them get saved. Match them
to that cell's entry in `src/sportstradamus/data/config/stat_meta.json`:

| Board column | What it is | Where it goes in `stat_meta.json` |
|---|---|---|
| `normalization` | the target shape | the `target_normalization` field |
| `blend` | how the model and book are combined | the `blending` field (add it if it's missing; leaving it out means the `nll` default) |
| `dist` | the training loss — a training-time knob | **nothing** — the shipped model always uses the family default, so ignore this column |

Watch the name clash: the `dist` **field** already in `stat_meta.json` is the *distribution family*
(`SkewNormal`, `ZINB`, …) — a different thing from the board's `dist` column. Leave the `dist` field
alone. (A ZINB cell has no `normalization`; its persistable columns are `zinb_mode` and
`count_dispersion_objective`, plus the same `blend` → `blending`.) You don't have to map columns by
hand either way — the sweep prints the exact `field=value` line to copy for each shipping cell.

So a board winner of `centered_additive_mean10 · crps/crps` for NBA FGM makes the cell read:

```json
"FGM": { "dist": "SkewNormal", "shipped": "withheld",
         "target_normalization": "centered_additive_mean10", "blending": "crps", "posthoc": "none" }
```

Set those fields *before* you train — `meditate` reads them:

```bash
poetry run meditate --league NBA --market FGM --bypass-withholding
```

A real training run (unlike the sweep) is the one whose result actually counts. One caveat: if the
winning row's `dist` column is `nll` rather than `crps`, that exact corner can't be reproduced from
`stat_meta.json` — only the `crps` default is saved — so prefer a `crps` row, or accept that you're
confirming under `crps`.

### 3. Read the ship-gate scorecard

`meditate` writes each cell's gate results into `model_stats.csv` (and its `.parquet` twin). To see
the verdict for one cell on its own:

```bash
poetry run python -m sportstradamus.training.scorecard --league NBA --market FGM
```

It prints a **SHIP SUMMARY** naming any gate the cell fails. A cell that passes every gate is ready
to ship. In plain terms the gates check:

- **Accuracy** — the model is at least as good as the sportsbook.
- **Star / bench bias** — predictions aren't systematically too high or too low for the best or the
  lowest-usage players.
- **Calibration** — the whole predicted distribution matches what actually happens.
- **Confidence** — the model is neither over- nor under-confident.
- **No shrinkage** — it doesn't quietly pull star players back toward the average.

The exact thresholds behind each gate are in [docs/ship_gate.md](docs/ship_gate.md).

### 4. Ship it to devel

Change the cell's `shipped` field from `"withheld"` to `"devel"` in `stat_meta.json`, then
sanity-check the config:

```bash
poetry run generate-ship-config --branch devel   # validates and summarizes; changes nothing
```

On `devel` the server serves the model and records how it does on real settled bets.

### 5. Graduate to main

Once a cell has proven itself on live bets, `check-graduation` shows where every cell stands and
`generate-ship-config` promotes the ones that passed:

```bash
poetry run check-graduation                      # status of every cell
poetry run generate-ship-config --branch main    # promote graduated cells to fully live
```

---

## Daily Workflow

```
morning:
  poetry run confer          # pull today's lines
  poetry run prophecize      # score + write dashboard snapshots

weekly (or when model accuracy drops):
  poetry run meditate        # retrain stale models
```

If a new season has started and the season-start date in the relevant Stats class
has not been updated, `meditate` will skip the league. Update
`src/sportstradamus/stats/{league}.py` → `Stats{League}.season_start`.

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

---

## Deferred / Archived Code

The following modules were moved to [`src/deprecated/`](src/deprecated/) during
the 2026-04-21 maintainability refactor because they had no caller in any CLI
entry point or `scripts/` module. They are preserved verbatim under
`src/deprecated/` (with a header comment recording the original path and the
last live git SHA) and should be reintroduced if the corresponding feature
returns. See [`src/deprecated/README.md`](src/deprecated/README.md) for the
header protocol and reintroduction process.

- [x] **DONE: parlay search reimplemented** — superseded by
      `prediction/parlay.py:beam_search_parlays` (Gaussian-copula beam search).
      `unused_funcs.py::find_bets` / `opt_parlay.py` remain in `src/deprecated/`
      pending a triage-delete.
- [ ] **TODO: reimplement BettingPros NFL ingest** (`get_lines.py`) —
      redundant with `books.py` scrapers, but a useful fallback.
- [x] **DONE: team correlation generator reimplemented** — `training/correlate.py`
      now builds the `{LEAGUE}_corr.csv` matrices (`meditate --rebuild-correlations`).
      The old `correlation.py` remains in `src/deprecated/` pending a triage-delete.
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
