# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Writing code in this repo — applies every session

This is a solo-maintained project. Every abstraction, fallback, and helper is
something **one person** has to understand and fix later, often months from now.
The default is *less code*, not more. Match the codebase you find; do not impose
textbook patterns over it. The rules in this section are binding on their own —
they do not depend on you having opened another file first.

**Avoid over-engineering.** Only make changes that are directly requested or
clearly necessary for the task. Keep solutions simple and focused.

* **Scope.** Don't add features, refactor, or make "improvements" beyond what was
  asked. A bug fix does not need the surrounding code cleaned up. If you notice
  unrelated work worth doing, say so — don't just do it.
* **Defensive coding.** Don't add error handling, fallbacks, or validation for
  scenarios that can't happen. Trust internal code and framework guarantees. Catch
  only the specific exceptions you can actually handle and let everything else fail
  loud — a clean traceback beats a swallowed error. No bare `except:`. Python is
  EAFP: prefer `try`/act over pre-checking conditions that are almost always true,
  and don't sprinkle `hasattr`/`getattr`-with-default to paper over our own code.
* **Edge cases.** Handle what the task names and what demonstrably occurs in this
  project's data. Ask before writing code for speculative inputs.

**Don't narrate code with comments.** Over-commenting is the most common tell of
machine-written code and the fastest way to make a file tiring to read.

* Comments explain **why**, never **what**. Delete any comment that just restates
  the line under it (`# increment counter`, `# loop over the rows`).
* No section-divider banners and no `# Note:` / `# Important:` spam. Flag a real
  gotcha when one exists; otherwise stay quiet.
* Docstrings go on public functions, classes, and modules only, and only when they
  add something the signature doesn't. Don't add docstrings, comments, or type
  annotations to code you didn't otherwise change. (See STYLE_GUIDE.md for the
  ratchet — D1xx is being lifted file by file, not back-filled in bulk.)

**Prefer a few deep functions over many thin ones.** A coherent 40-line function
beats six 7-line fragments you have to read together to follow one thought.

* A function must earn its name with real logic. No wrappers that only rename and
  forward a call. Keep a single logical operation within ~3 layers of our own calls.
* No new class where a function does the job. No factory / strategy / config-object /
  dependency-injection scaffolding unless **three** real implementations already
  need it (rule of three). Build for today's task, not a hypothetical future one.

**Reuse before you write.** Duplicated logic is the largest measured source of
drift in AI-assisted codebases.

* Before writing a utility, grep `helpers/` and the relevant package for one that
  already exists, and use it. `sportstradamus.helpers` is the shared home (see the
  package table below).
* Don't reimplement the standard library or our existing stack. Reach for
  `itertools`, `collections`, `pathlib`, and vectorized pandas/numpy/polars before
  hand-rolling a loop.
* Find the same logic in two places? Consolidate it — don't copy it a third time.

**Type hints in moderation.** Annotate public signatures and module boundaries;
skip the obvious locals. Avoid `Any` — model real structures with `dataclass`,
`TypedDict`, or `Protocol`. Don't build elaborate generics where a concrete type
reads fine.

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
* **Multi-module work uses subagent-driven development by default.** When a
  task touches two or more modules, dispatch one subagent per module rather
  than serializing through the main session. The main session orchestrates,
  reviews diffs, and runs the quality gates. **One module per subagent** —
  the same scope discipline as the old "one module per session" rule, just
  parallelized. Single-module work stays in the main session; deviating from
  the per-subagent scope (e.g. one subagent touching two modules) needs an
  explicit reason recorded in the plan.
* Before claiming anything is "done", run `poetry run pytest tests/golden/`,
  `poetry run pytest -m integration` (fake-mode, no network), and
  `poetry run ruff check src/sportstradamus/`. All three must be clean.
* Dashboard banner/caption timestamp lines show only the timestamp. Do not append
  feature descriptions, announcements, or other text to them.

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
  See STYLE_GUIDE.md §9.
* **Dashboard never touches the DuckDB archive.** The Streamlit dashboard reads
  pre-computed parquet snapshots only (`data/history.parquet`,
  `data/parlay_hist.parquet`, `data/model_stats.parquet`). DuckDB holds an
  exclusive file lock for the entire lifetime of any read-write connection;
  the dashboard is the only long-lived process in the system, so any archive
  connection it opens — even accidentally via a module-level `Archive()` in
  a transitively-imported module — blocks every cron job (prophecize, confer,
  close-lines) that needs to write odds. If a dashboard page ever needs
  archive-derived data, export it to parquet from a cron job and read the
  parquet from the dashboard; do not query DuckDB directly. New dashboard
  pages must not import any module whose top-level code constructs
  `Archive()` — use `LazyArchive` from `sportstradamus.helpers` if a module
  needs an `archive` binding shared with the prediction or training pipelines.
  Pinned by `tests/golden/test_dashboard_no_archive_lock.py`.

## MANDATORY: run refactoring-specialist before any push, PR update, or review

This is not a suggestion. You MUST invoke the `refactoring-specialist` subagent
on every Python file you touched in the current session **before** any of:

1. `git push` to any remote
2. Calling any GitHub MCP tool that creates or updates a PR
3. Replying "done" on a task that edited Python sources
4. **Dispatching any code-review subagent** (`superpowers:code-reviewer`, the
   spec-compliance reviewer or code-quality reviewer in the subagent-driven
   development workflow, or any future review agent)
5. **Asking the user for review feedback** on Python edits (e.g., "does this
   look right?", "ready for your review", surfacing a diff for sign-off)

Triggers 4 and 5 exist because reviewers — human or subagent — should spend
their attention on substance, not style nits the refactoring-specialist
would have caught. Running it first compresses the review loop.

No exceptions for "small" edits, "obvious" fixes, doc tweaks that grazed a
`.py`, or tests-only changes. If you wrote to a `.py` file under
`src/sportstradamus/`, the subagent runs before any of the five gates above.

How to invoke:

* Use the `Agent` tool with `subagent_type: "refactoring-specialist"`.
* In the prompt, list every Python file you modified, created, or moved this
  session. The subagent refuses to scan the whole repo on its own; you must
  hand it the scope.
* Wait for its report. Do not push while it is still running.
* If it reverts any of your edits or flags a behavior risk per STYLE_GUIDE
  §18.9, address the items it raised — re-invoke if needed — before pushing.

Skipping this step is the single most common way Claude sessions ship code
that violates the style guide. The PR review will catch it and you will redo
the work. Just run the subagent.

The subagent definition lives at `.claude/agents/refactoring-specialist.md`.
Read it once per session so you know its scope and limits.

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
poetry run pickem-build      # Underdog Power/Flex/Rivals recommendations YAML
poetry run kelly             # re-size a recommendations YAML offline

# Quality gates — all three must pass before committing
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/
poetry run pytest -m integration            # fake-mode end-to-end, no network

# Regenerate CLI help snapshots after an intentional flag change
REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py
```

Python 3.11 required. PyTorch CPU-only (2.1.2) via custom Poetry source.

## Production deployment

* **The remote server tracks `devel`, not `main`.** Cron pulls run against the
  `devel` branch — `main` is allowed to lag. Don't assume the production code
  matches `main`; check `devel` HEAD when reasoning about server behavior.
* **All cron jobs go through `scripts/run_job.sh`.** The wrapper adds:
  - per-job `flock -n` (a second invocation of the same job is skipped),
  - a shared archive `flock -w 900` (serializes against DuckDB's
    single-writer lock so jobs don't collide on `archive.duckdb`),
  - Healthchecks.io `/start` / `/fail` / success pings,
  - structured `START` / `OK` / `FAIL` / `WAIT` log lines per job.
* **Production crontab** (run as `sportstradamus@<host>`):

  ```cron
  50 8-20 * * *          /home/sportstradamus/Sportstradamus/scripts/run_job.sh prophecize
  30 8,12 * * *          /home/sportstradamus/Sportstradamus/scripts/run_job.sh confer
  0 1 * * 5              /home/sportstradamus/Sportstradamus/scripts/run_job.sh meditate
  0 23 * * *             /home/sportstradamus/Sportstradamus/scripts/run_job.sh reflect
  */10 11-23,0-1 * * *   /home/sportstradamus/Sportstradamus/scripts/run_job.sh close-lines
  0 2 1 * *              /home/sportstradamus/Sportstradamus/scripts/run_job.sh gate-status
  0 10 * * 3             /home/sportstradamus/Sportstradamus/scripts/run_job.sh fp-fetch
  ```

  The `fp-fetch` job runs weekly during NFL season: it walks the
  Fantasy Points Data Suite endpoint catalog
  (`src/sportstradamus/data/config/fantasypoints_endpoints.json`) and
  writes per-tool snapshots to
  `src/sportstradamus/data/fantasypoints/{season}/week_NN/`. Needs a
  fresh session cookie in `creds/keys.json` (see
  `docs/fantasypoints.md`) and `HEALTHCHECK_URL_FP_FETCH` set so
  cookie-expiry surfaces as an immediate alert.

  The `gate-status` job runs monthly: it promotes/demotes cells in `main`'s
  `stat_meta.json` based on live Gate-2 graduation and opens a PR (a human
  merges — `main` is the public branch). It needs `gh` authenticated on
  the box (`GH_TOKEN` or `gh auth`) and `HEALTHCHECK_URL_GATE_STATUS` set.
  On `devel`, ship promotions are one-line edits to `stat_meta.json`
  (`shipped: "withheld"` → `shipped: "devel"`) the human commits directly;
  `generate-ship-config --branch devel` only validates + summarizes.

  `prophecize` and `close-lines` both fire at `:50` during peak hours; the
  `run_job.sh` archive flock serializes them, so the second-to-acquire just
  waits and emits a `WAIT job=… archive_lock_wait=Ns` line.

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

**`Archive`** (`helpers/archive.py`): DuckDB singleton persisting odds at
`archive/archive.duckdb`. Two tables — `odds(league, market, game_date, entity, book, ev)`
and `lines(league, market, game_date, entity, line)` — no PRIMARY KEY (the PK index alone
bloats the DB ~10× for this row count; sorted-on-disk data + zone-map pruning give
~1 ms point lookups without it). Writes accumulate in in-memory buffers
(`_pending_odds`, `_pending_lines`, `_replace_keys`) and flush bulk-deduped on
`Archive().write()` — same in-memory-mutate-then-dump semantics as the old klepto
backend. Public methods: `get_ev`, `get_line`, `get_moneyline`, `get_total`,
`get_team_market`, `to_pandas`, `add_dfs`, `merge_player_books`, `set_team_books`,
`archived_players_by_date`, `write`, `clean_archive`.

**`Scrape`** (`helpers/scraping.py`): HTTP client with ScrapeOps browser-header rotation
and ScrapingFish proxy fallback.

### ML Pipeline

**LightGBMLSS** for distributional regression — predicts full probability distributions.

Distribution types (set per cell in `data/config/stat_meta.json`):
- **Gamma** / **ZAGamma** — continuous stats, optional zero-inflation
- **Negative Binomial** / **ZINB** — count stats, optional zero-inflation
- **SkewNormal** (`skew_normal.py`) — custom PyTorch distribution; used when
  `global_mean >= 2` in `training/pipeline.py:train_market`

**Player Comparables**: z-scored profiles → weighted BallTree KNN → comp outcomes as
features. Weights optimized via `scripts/optimize_comp_weights.py`.

**Feature filtering**: SHAP-based importance thresholds in `feature_filter.json`,
updated by `meditate --rebuild-filter` via `training/shap.py`.

### Training Report Diagnostics (`data/training_report.txt` + `data/model_stats.parquet`)

Generated by `training/report.py:report()` after `meditate`. Phase 3 Step 4
migrated the diagnostic schema to **raw data-science metrics** — `1 − Brier`
and the other folded-up ratios are gone. The parquet is regenerated from
scratch on every `meditate` run; there is no legacy alias.

**`row_kind` column** — every metric row carries one of two values:
- `"book_baseline"`: pinned reference row, computed by treating the bookmaker's
  implied probabilities as the model. Tells you what taking the book's odds
  alone would score on the same metric.
- `"model"`: the real model.

The companion `metric_row` column still distinguishes `raw / corrected /
calibrated` for the model rows.

**Header**: Distribution type (NegBin/Gamma/ZINB/ZAGamma/SkewNormal) + historical zero rate.

**Scoring rules** (lower is better unless noted):

| Metric | Meaning | Direction |
|--------|---------|-----------|
| `brier_score` | Mean squared error of probabilities vs. binary outcomes | ↓ lower is better |
| `log_loss` | Cross-entropy of probabilities vs. outcomes | ↓ lower is better |
| `nll` | Negative log-likelihood (renamed from `NLL`) | ↓ lower is better |
| `expected_calibration_error` | 10-bin ECE | ↓ lower is better |
| `brier_skill_score` | `1 − model_brier / book_brier` (vs. book baseline) | ↑ higher is better; 0 = matches book |

**Discrimination**:

| Metric | Meaning | Direction |
|--------|---------|-----------|
| `roc_auc` | Area under the ROC curve | ↑ higher is better; 0.5 = random |
| `accuracy` | Fraction correct on over/under (renamed from `Accuracy`) | ↑ higher is better |
| `precision_over` | Precision of "Over" predictions (renamed from `Over Prec`) | ↑ higher is better |
| `precision_under` | Precision of "Under" predictions (renamed from `Under Prec`) | ↑ higher is better |
| `prediction_std` | Std dev of predicted probabilities (renamed from `Sharpness`) | informational |

**Rates**:

| Metric | Meaning |
|--------|---------|
| `predicted_over_rate` | Fraction predicted Over (renamed from `Over%`) — compare to `empirical_over_rate` |
| `empirical_over_rate` | Actual hit rate of legs in the validation set |
| `frac_ev_gt_line` | Fraction where model EV > book line (~0.5 = calibrated) |
| `over_pct_ev_gt` | Over rate when model EV > line (should be high) |
| `over_pct_ev_lt` | Over rate when model EV < line (should be low) |

**Kelly & blending**:

- `model_weight`: Blend weight [0.05–0.9] model vs bookmaker, optimized via
  log-likelihood. Logarithmic opinion pool for NegBin, precision-weighted
  blend for Gamma.
- `kelly_shrinkage`: `clip(brier_skill_score, 0, 1)` — the value
  `strategies/kelly.py` reads via `training.report.get_market_calibration`.
  Higher = trust the model more. `0` = bookmaker dominates the blend.

**Shape / Dispersion**:

- `start_r` / `start_alpha`: Method-of-moments shape estimate from training
  features (NegBin: `μ²/(σ²-μ)`, Gamma: `(μ/σ)²`).
- `model_r` / `model_alpha` (`model_shape`): Mean shape parameter from
  LightGBMLSS (NegBin: `total_count`, Gamma: `concentration`).
- `empirical_r` / `empirical_alpha` (`empirical_shape`): Shape from actual
  test outcomes — ground-truth dispersion.
- `shape_ratio`: `model_shape / empirical_shape`. >1 = over-dispersed,
  <1 = under-dispersed, 1.0 = ideal.
- `shape_ceiling`: Upper bound on shape during training (2× `marginal_shape`).
- `marginal_shape`: Empirical shape from full training targets.
- `dispersion_cal`: Post-hoc shape scaling [0.1–1.0] learned on validation
  set. 1.0 = no correction, <1.0 = model over-dispersed.

**EV & lines**:

- `start_mean`: Average of training feature `MeanYr`.
- `model_ev`: Mean blended expected value (model + bookmaker fusion via `fused_loc()`).
- `mean_line`: Average bookmaker line.
- `result_mean`: Actual test set mean outcome.
- `mean_ev_diff` / `median_ev_diff`: Blended EV excess over bookmaker line.
- `cf_over_pct`: Counterfactual Over% using empirical shape vs model shape.

**HP — Hyperparameters** (Optuna tuning):
- `rounds`: Boosting iterations | `leaves`: Max leaves per tree | `lr`: Learning rate
- `min_child`: Min samples per leaf | `L1`/`L2`: Regularization

**Reading the report**:

- The `BOOK BASELINE` line at the top of each market block is "what taking
  the book's odds gets you" on every metric. The model's job is to beat it.
- `brier_skill_score > 0` ⇔ model beats book on Brier.
  `kelly_shrinkage = 0` ⇔ no Kelly stake (Kelly degenerates to no-information).
- `prediction_std` low + `nll` low = confident and correct.
- `over_pct_ev_gt` high & `over_pct_ev_lt` low = properly discriminative.
- `shape_ratio` ≈ 1.0 = well-calibrated dispersion;
  `dispersion_cal` = 1.0 = no fix needed; <1.0 = model over-dispersed.
- `model_weight` near 0 = bookmaker dominates; near 1 = model dominates.

The Streamlit dashboard (`pages/7_📊_Stats_Model_Training.py`) renders
these metrics in family-grouped tabs (Scoring rules, Discrimination, Rates,
Kelly & blending, Dispersion, EV & lines, Hyperparameters) with a pinned
book-baseline row above each table and column-direction help text.

### Key Configuration Files (`src/sportstradamus/data/`)

| File | Purpose |
|------|---------|
| `config/stat_meta.json` | **Committed.** Per-cell `{dist, shipped, strategy}` (distribution family, release surface — `"withheld"` / `"devel"` / `"main"`, training strategy slug) |
| `config/stat_calibration.json` | **Gitignored.** Per-cell `{cv, std, zi}` — runtime-recomputed by `meditate` each run |
| `config/stat_map.json` | Stat name mappings across APIs/sportsbooks |
| `config/feature_filter.json` | SHAP-based feature importance thresholds |
| `config/playerCompStats.json` | Learned player comp weights per league/position |
| `config/book_weights.json` | **Gitignored.** Sportsbook reliability weights for consensus lines |
| `config/prop_books.json` | List of sportsbooks consulted for player-prop consensus |
| `{LEAGUE}_corr.csv` | Pre-computed player stat correlation matrices |

### Data Storage

- `data/models/` — Trained LightGBMLSS model pickles
- `data/training_data/` — Cached training matrices
- `data/player_data/{LEAGUE}/{YEAR}/` — Historical game log CSVs
- `data/test_sets/` — Holdout test data
- `creds/` — API keys (`keys.json`), Google OAuth (`credentials.json`, `token.json`)