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
  forward a call, and no build-and-invoke thunks (`def main(): _build_cli()()`): name
  the command at module level and point the entry point at it (`module:command`). Keep
  a single logical operation within ~3 layers of our own calls.
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
* **Any parallelism must be justified.** Per-league / per-grain blocks that look
  alike are not automatically fine: if they encode the *same* knowledge,
  consolidate (base-class method, shared helper, or a store parameterized by the
  differing values). A block may stay parallel only when it is the *same shape
  over genuinely different knowledge* — e.g. league-specific physical constants —
  and consolidating would force a banned pure-forwarder; then it carries an
  explicit `# pylint: disable=duplicate-code` pragma with a one-line rationale.
  The blocking `tests/golden/test_no_duplicate_code.py` gate fails on any
  unjustified clone (pylint R0801).

**Type hints in moderation.** Annotate public signatures and module boundaries;
skip the obvious locals. Avoid `Any` — model real structures with `dataclass`,
`TypedDict`, or `Protocol`. Don't build elaborate generics where a concrete type
reads fine.

## Mandatory reading — do this first

Before touching any code, read these documents once:

1. **[CONTRIBUTING.md](CONTRIBUTING.md)** — contribution workflow: how to make changes,
   how to add a league or market — plus **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
   for the package map, data flow, and where to find things. Required reading. Not optional.
2. **[docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md)** — formatting, naming, docstrings, type
   hints, dead-code rules. Cite sections by number in commits and comments.

If you skip these and make changes that violate the layout or style, you will be asked
to redo the work.

## General Rules

* Talk like caveman except when /writing-clearly-and-concisely overrides
* **Documentation: write to not drift.** Before editing any `.md`, skim
  [docs/STYLE_GUIDE.md §16](docs/STYLE_GUIDE.md). One canonical home per fact
  (cross-ref, don't restate); revise stale statements in place rather than layering
  new ones beside them; living docs describe current state. Changelogs stay short
  and caveman — one-line entries, newest-first, cap the recent few, detail in git;
  no dated narrative build-log blocks in the body. The `docs-style.py` hook nudges
  when an edit adds one.
* Use `click` over `argparse` for CLI args
* Long-running scripts: add a status bar with `tqdm`
* Money values are always `Decimal`, never `float`
* Program roadmap: [docs/sportstradamus_roadmap_v3.md](docs/sportstradamus_roadmap_v3.md)
  (swimlane index). Working a lane? Read its brief in `docs/handoffs/` first.
* **Multi-module work uses subagent-driven development by default.** When a
  task touches two or more modules, dispatch one subagent per module rather
  than serializing through the main session. The main session orchestrates,
  reviews diffs, and runs the quality gates. **One module per subagent** —
  the same scope discipline as the old "one module per session" rule, just
  parallelized. Single-module work stays in the main session; deviating from
  the per-subagent scope (e.g. one subagent touching two modules) needs an
  explicit reason recorded in the plan.
* Before claiming anything is "done", run `poetry run pytest tests/golden/`
  (now parallel via pytest-xdist), `poetry run pytest -m integration -n0`
  (fake-mode, no network; `-n0` because the integration suite is not xdist-safe),
  and `poetry run ruff check src/sportstradamus/`. All three must be clean.
* Dashboard banner/caption timestamp lines show only the timestamp. Do not append
  feature descriptions, announcements, or other text to them.
* **Dashboard UI work: read [DESIGN.md](DESIGN.md) first.** It holds the committed visual
  identity — FIXED design tokens mirrored in `.streamlit/config.toml` — plus the NEVER list
  that keeps the app from looking AI-generated (no default-red, no purple gradients, no
  Inter/Roboto, Material icons not emoji). Treat the FIXED tokens as inviolable and do not
  supplement them with your own defaults. The `design-lint` hook nudges live and
  `tests/golden/test_design_tokens.py` is the hard gate. The UX redesign spec (six surfaces,
  slip rail, taxonomy, scars) is
  [docs/dashboard_ux_redesign.md](docs/dashboard_ux_redesign.md); its lane brief is
  [docs/handoffs/dashboard-ux.md](docs/handoffs/dashboard-ux.md).

## Agentic workflow conventions

These conventions pair with the hooks in `.claude/hooks/`.

* **The refactoring-specialist runs no pytest.** It refactors and runs `ruff`
  on its scope only. The main agent owns the single authoritative gate run —
  after the specialist returns, run `poetry run ruff check src/sportstradamus/`,
  `poetry run pytest tests/golden/`, and the integration command below, exactly
  once. If they fail, isolate cause via `git diff` of the specialist's changes.

* **Integration gate before a push.** The push-gate hook reads
  `.claude/.state/integration_green`. Run the authoritative integration suite as:
  `poetry run pytest -m integration -n0 && touch "$CLAUDE_PROJECT_DIR/.claude/.state/integration_green"`
  so a clean run clears the push prompt. Editing any `.py` afterward re-arms it.

* **Research-first.** Before building any `docs/handoffs/model_improvement_track.md`
  §8.2-flagged lever, or
  changing a distribution family / dispersion mechanism, dispatch the
  `research-analyst` subagent first and cite its `/tmp/researcher_*.md` brief.
  The research-gate hook enforces the discrete cases (a `shipped:` flip in
  `stat_meta.json`, edits to a distribution-family file in
  `.claude/research_gated.txt`); this convention covers the judgment calls a
  path matcher cannot see. To proceed without a brief on a gated edit, write a
  one-line justification to `.claude/.state/research_waiver`.

* **Session memory capture.** When a unit of work completes — notably before or
  at a push — offer to capture any durable, non-obvious, repeatable lesson to the
  memory dir in the standard format. Do not force a memory from every session.

## Hard rules — these caused the last major refactor

The codebase was refactored from several 1,000–7,000 line monoliths into packages.
Do not undo that work:

* **No new monoliths.** If a file you are editing exceeds ~300 lines, stop and check
  whether you are adding to the right module. Consult docs/ARCHITECTURE.md §Package Map.
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
  pre-computed parquet snapshots only (`data/runtime/history.parquet`,
  `data/runtime/parlay_hist.parquet`, `data/training/model_stats.parquet`,
  `data/runtime/current_pickem.parquet`). DuckDB holds an
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
poetry run prophecize        # prediction pipeline → dashboard snapshots
poetry run confer            # fetch current odds/props
poetry run meditate          # train/retrain ML models
poetry run reflect           # historical parlay performance
poetry run dashboard         # Streamlit dashboard
poetry run pickem-build      # Underdog Power/Flex/Rivals recommendations YAML
poetry run kelly             # re-size a recommendations YAML offline

# Quality gates — all three must pass before committing
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/             # parallel via pytest-xdist (-n auto lives in addopts)
poetry run pytest -m integration -n0        # fake-mode end-to-end, no network; -n0: integration is not xdist-safe

# Dev-only diagnostic-script tests (zinb-routing, icc), excluded from the default loop
poetry run pytest -m diagnostics

# Regenerate CLI help snapshots after an intentional flag change
REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py
```

Python 3.11 required. PyTorch CPU-only (2.9.1) via custom Poetry source.

## Production deployment

* **The remote server tracks `devel`, not `main`.** Cron pulls run against the
  `devel` branch — `main` is allowed to lag. Don't assume the production code
  matches `main`; check `devel` HEAD when reasoning about server behavior.
* **All cron jobs go through `scripts/run_job.sh`.** The wrapper adds:
  - per-job `flock -n` (a second invocation of the same job is skipped),
  - a self-deploy `git pull --ff-only origin devel` before each job (own
    `flock -n`; a failed pull runs the existing checkout; the two modifier
    configs are reset then re-folded from `modifier_overrides.json` with
    `--prune` so dashboard-captured corrections never block the pull;
    `GIT_PULL=0` skips),
  - a shared archive `flock -w 900` (serializes against DuckDB's
    single-writer lock so jobs don't collide on `archive.duckdb`),
  - Healthchecks.io `/start` / `/fail` / success pings,
  - structured `START` / `OK` / `FAIL` / `WAIT` / `PULL` log lines per job.
* **Production crontab** (run as `sportstradamus@<host>`): the canonical
  schedule lives in [docs/OPERATIONS.md](docs/OPERATIONS.md).
  Key coupling: the number of confer slots must match `broad_slots_per_day`
  in `data/config/odds_api_budget.json`.

  The `fp-fetch` job runs weekly during NFL season: it walks the
  Fantasy Points Data Suite endpoint catalog
  (`src/sportstradamus/data/config/fantasypoints_endpoints.json`) and
  writes per-tool snapshots to
  `src/sportstradamus/data/fantasypoints/{season}/week_NN/`. Needs a
  fresh session cookie in `creds/keys.json` (see
  `docs/fantasypoints.md`) and `HEALTHCHECK_URL_FP_FETCH` set so
  cookie-expiry surfaces as an immediate alert.

  The `ctg-fetch` (NBA, Cleaning the Glass) and `savant-fetch` (MLB,
  Baseball Savant) collectors are **dev-side**, not prod-cron: run them on
  the dev box beside the manual weekly `meditate`, then `sync_to_prod.sh`
  pushes their date-stamped snapshots up. They share the `fp-fetch`
  framework (`sportstradamus.collectors`); `StatsNBA`/`StatsMLB` fold the
  snapshots in via the `_join_fp_*_features` hooks once each source's join
  schema is pinned. Canonical guide: `docs/data_collectors.md`.

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

Full per-submodule breakdown is in docs/ARCHITECTURE.md §Package Map.

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
   scores each offer for EV, `find_correlation` scores parlay legs, and `persist.py`
   writes parquet snapshots (`current_offers`, `current_pickem`, `history`) the Streamlit
   dashboard reads.

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

**Feature selection**: None at training time. Per the 2026-05-27 no-filter rewire
(researcher Option C; Akhiat & Touchanti 2024 arXiv:2411.05937 — tree ensembles
statistically tie FS-filtered vs no-FS across 960 XGBoost experiments), every cell
trains on the full unfiltered candidate set returned by
`Stats.get_stat_columns(market)` (Common + per-league Common + ALL player/team/defense
profile columns, ~440 features per NFL cell). `feature_filter.json` still ships the
`Common`/`Always` buckets for shared/locked-in lists but `Filtered` is gone.
`training/shap.py:compute_market_importance` runs after each cell trains, writing
per-cell |SHAP| + corr to `feature_importances.csv` / `feature_correlations.csv` for
drift monitoring only — those CSVs no longer drive selection.

### Training stats (`data/training/model_stats.parquet` + `.csv` mirror)

Single source of truth for per-cell training diagnostics. Written by
`training/report.py:report()` after every `meditate` run as **one wide row
per `(league, market)` cell**; consumed by:

* The Streamlit dashboard (`pages/7_Stats_Model_Training.py`) — tab views
  by metric family with `lifecycle_state` joined in from
  `training/graduation.py:lifecycle_table()`.
* `training.report.get_market_calibration(league, market)` — kelly's
  per-cell `{kelly_shrinkage, brier_skill_score, model_weight}` getter.
* `training.graduation.read_gate1` → `check_graduation` /
  `generate-ship-config` for Gate 1 ↔ Gate 2 promotion logic.

`model_stats.csv` is a literal mirror written next to the parquet so the
same numbers are browseable from VSCode without a parquet viewer; the
parquet is authoritative on disagreement and is the only file readers
consult.

**Column groups** (one row per cell, ~50 columns):

| Group | Columns | Owner |
|---|---|---|
| Identity | `league`, `market`, `distribution`, `shipped` | `meditate` + `stat_meta.json` |
| Sample | `n_validation`, `historical_zero_rate` | `meditate` + scorecard |
| Scoring | `brier_book`, `brier_model`, `brier_skill_score` (↑), `log_loss_book`, `log_loss_model`, `nll` | `meditate` |
| ECE | `ece_equal_mass`, `ece_null_bias`, `ece_debiased` (↓) | `training.scorecard.compute_gates` |
| Discrimination | `roc_auc` (↑), `accuracy` (↑), `precision_over` (↑), `precision_under` (↑), `prediction_std` | `meditate` |
| Over rates | `predicted_over_rate`, `empirical_over_rate`, `over_pct_ev_gt` (↑), `over_pct_ev_lt` (↓) | `meditate` |
| EV / line | `model_ev`, `mean_line`, `result_mean`, `mean_ev_diff` (↑), `median_ev_diff` (↑), `frac_ev_gt_line` | `meditate` |
| Kelly | `kelly_shrinkage` (↑), `model_weight` | `meditate` |
| Shape | `model_shape`, `empirical_shape`, `shape_ratio`, `marginal_shape`, `dispersion_cal` | `meditate` |
| Ship gates | `g1_brier_diff_mean` (↓), `g1_brier_diff_ci_lo/hi`, `g1_brier_diff_mean_oracle` (+ oracle CI), `g2_star_z` (↓), `g3_bench_z` (↓), `g4_iqr_ratio`, `g5_ece_debiased` (↓), `g6_star_ci_hi`/`g6_star_ref`/`g6_recent_corr` (`ratio_meanyr` SkewNormal cohort), `g1_pass`…`g6_pass`, `ship` | `training.scorecard.compute_gates` |
| HP | `hp_rounds`, `hp_leaves`, `hp_lr`, `hp_min_child`, `hp_l1`, `hp_l2` | `meditate` |
| Calibration | `cv`, `std` | `meditate` (from `stat_calibration.json`) |

**Reading the parquet**:

- `brier_skill_score > 0` ⇔ model beats book on Brier;
  `kelly_shrinkage = clip(brier_skill_score, 0, 1)` — the value
  `strategies/kelly.py` reads via `training.report.get_market_calibration`.
- `over_pct_ev_gt` high & `over_pct_ev_lt` low ⇒ EV-positive picks
  actually win more.
- `shape_ratio` ≈ 1.0 ⇒ well-calibrated dispersion;
  `dispersion_cal = 1.0` ⇒ no fix needed; `< 1.0` ⇒ model over-dispersed.
- `model_weight` near 0 ⇒ bookmaker dominates; near 1 ⇒ model dominates.
- `ship == True` ⇔ all six offline gates pass on the most recent
  test-set CSV; see `docs/ship_gate.md` for the threshold rationale.

**Standalone A/B-test harness** for testing model updates without
touching production:

```bash
poetry run python -m sportstradamus.training.scorecard \
    --baseline data/test_sets/NBA_PTS.csv \
    --candidate /tmp/NBA_PTS_centered.csv
```

The CLI never writes `model_stats.parquet` — that file is owned by
`training.report.report()`. In `--scorecard-out` mode the CLI writes a
sandbox CSV (defaults to `/tmp/scorecard.csv`) so A/B runs can't clobber
production data.

### Key Configuration Files (`src/sportstradamus/data/`)

| File | Purpose |
|------|---------|
| `config/stat_meta.json` | **Committed.** Per-cell `{dist, shipped, strategy}` (distribution family, release surface — `"withheld"` / `"devel"` / `"main"`, training strategy slug) |
| `config/stat_calibration.json` | **Gitignored.** Per-cell `{cv, std, zi}` — runtime-recomputed by `meditate` each run |
| `config/stat_map.json` | Stat name mappings across APIs/sportsbooks |
| `config/odds_api_budget.json` | **Committed.** Odds API credit governor knobs: cycle quota/reset day, `enforce` kill switch, slots/day (must match the confer crontab), floor safety factor, per-league seed costs + priority. Consumed by `helpers/odds_budget.py`; ledger at `data/runtime/odds_api_usage.jsonl` |
| `config/feature_filter.json` | League-shared (`Common`) + per-market locked-in (`Always`) feature lists. The historical `Filtered` SHAP-ranked buckets were removed in the 2026-05-27 no-filter rewire; production trains on the full candidate set |
| `config/playerCompStats.json` | Learned player comp weights per league/position |
| `config/book_weights.json` | **Gitignored.** Sportsbook reliability weights for consensus lines |
| `config/prop_books.json` | List of sportsbooks consulted for player-prop consensus |
| `leagues/{league}/corr_same_team.parquet` + `corr_opposing.parquet` | Pre-computed player stat correlation matrices (written by `training/correlate.py`; never loaded by the dashboard — NBA is 2.85M rows, per-game slices only) |

### Data Storage

- `data/models/` — Trained LightGBMLSS model pickles
- `data/training_data/` — Cached training matrices
- `data/player_data/{LEAGUE}/{YEAR}/` — Historical game log CSVs
- `data/test_sets/` — Holdout test data
- `creds/` — API keys (`keys.json`)