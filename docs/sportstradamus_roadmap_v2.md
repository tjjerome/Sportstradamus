# Sportstradamus Improvement Roadmap (v2)

A six-phase plan for evolving `tjjerome/Sportstradamus` from a sophisticated modeling factory with partial decision-engine coverage into a complete Underdog edge system.

This version corrects a significant error in the prior roadmap: many things I assumed were missing actually exist in production. The current live code includes `training/correlate.py` (correlation matrix generator), `prediction/parlay.py:beam_search_parlays` (combinatorial parlay search via beam search), `prediction/correlation.py:find_correlation` (joint-probability scorer), `nightly.py:reflect` (historical parlay analysis), and `helpers/distributions.no_vig_odds` (sharp de-vigging). The deferred items in `src/deprecated/` are *older alternative implementations* that were replaced — the README's TODO list partly references them by their old paths.

The result: **Phase 1 is no longer about reviving missing code; it's about auditing existing code, fixing methodology gaps, and surfacing the contest-variant and push-handling features that were never built**. Several genuinely missing features (Kelly sizing, CLV tracking, Underdog contest-variant handling, closing-line freeze, model versioning) move into later phases.

---

## Current Status Snapshot (audited 2026-05-08)

A walk through the live tree against this roadmap. Status markers are inlined per
sub-phase below; this section is the executive summary.

**Phase 1 — Audit and Strengthen the Foundation: ~80% complete.**

- 1.1 Correlate audit ✅ landed (`docs/CORRELATE_AUDIT.md`). Correlate fixes ✅ shipped:
  `training/correlate.py` now residualizes on a rolling 8-game window
  (`_residualize_gamelog`, line 404), produces *stratified* matrices, and
  `meditate` exposes `--rebuild-correlations` (`training/cli.py:50`).
- 1.2 Parlay audit ✅ landed (`docs/PARLAY_AUDIT.md`, `PARLAY_CALIBRATION_*.png/csv`,
  `scripts/audit_parlay_calibration.py`). Parlay fixes ✅ mostly shipped:
  `beam_search_parlays` is Gaussian-copula-based (`correlation.py:603`), takes a
  `contest_variant: Literal['power','flex','insurance','rivals']` parameter,
  reads `data/underdog_payouts.json`, and applies push-aware EV via
  `_expected_payout_with_pushes`. **Note**: `prediction/parlay.py` does not
  exist; both functions live in `prediction/correlation.py`. Several audit
  findings remain open (PSD repair, magic-number cleanup, `banned_combos`
  semantics, `Boost`-column inconsistency).
- 1.3 Closing-line freeze ❌ not built as a discrete step. Functionally,
  `clv.fill_from_archive` uses the last archived sample before kickoff as the
  "close" because the Odds API stops surfacing prematch odds for in-progress
  games. That works for the existing CLV summary but doesn't give us an
  explicit, queryable `closing_lines/` snapshot for downstream tracking.
- 1.4 Structured logging ✅ shipped (`helpers/logging.py`).
- 1.5 Integration tests ✅ shipped (`tests/integration/test_end_to_end.py`).
- 1.6 Deprecated triage ❌ doc not produced; `src/deprecated/` still carries
  TODOs that reference replaced modules.

**Phase 2 — CLV Tracker: ✅ done for this package's scope.**

- CLV computation exists (`src/sportstradamus/clv.py`) and is wired into
  `nightly.reflect`, including segmented summaries with a `CLV_SEGMENT_MIN_N`
  threshold. The retrospective CLV pipeline against system-recommended
  history is the right scope for this project.
- **Placed-bet logging is explicitly out of scope.** Tracking which entries
  a user actually placed is a personal-bookkeeping concern, not a package
  concern; users handle that themselves. The roadmap's original
  `tracking/` package (Phase 2.2 / 2.3) and the dashboard tabs that
  depended on `tracking.db` (Phase 4.2) are dropped, not deferred.
- Keep an eye on the existing CLV pipeline: revisit if segment sample
  sizes stay below `CLV_SEGMENT_MIN_N=20` for too long, if the implicit
  "last sample before kickoff = close" assumption in
  `clv.fill_from_archive` starts diverging from sharp closes (e.g.,
  Odds API behavior changes), or if `frac_beat_close` drifts away from
  ~50% on no-edge segments.

**Phase 3 — Underdog-Specific Decision Engine: ~25% complete.**

- 3.1 Kelly sizing ❌ no `strategies/kelly.py`. The only Kelly logic is in
  `src/deprecated/opt_kelley_bet.py` and an unrelated Streamlit page.
- 3.2 Contest-variant payouts ✅ done — `data/underdog_payouts.json` covers
  power/flex/insurance/rivals; `beam_search_parlays` honors them and Flex's
  miss-one tier is integrated into push-aware EV.
- 3.3 Underdog-native strategy module ❌ no `strategies/` package at all.
- 3.4 Champions ❌ not started.
- 3.5 Streaks / Ladders / Rivals strategies ❌ not started (Rivals payout
  table exists in JSON; no decision engine).

**Phase 4 — Real-Time Alerts and Dashboard Extensions: scope-reduced.**

- 4.1 Alerts package ❌ no `alerts/` directory. Useful subset (high-edge
  appearance, scraper failure, model staleness, CLV drift on system
  history) is still in scope; the rules that needed the placed-bet log
  (open-entry exposure, bankroll drawdown) are dropped with the bet
  logger.
- 4.2 Dashboard ✅ baseline `dashboard.py`, `dashboard_app.py`,
  `dashboard_data.py`, and a `pages/` multi-page set exist. The
  proposed Open Entries / Settled History / Bankroll tabs are dropped
  (they depended on `tracking.db`); a Today's Recommendations tab built
  off `data/recommendations/{today}.yaml` is still worthwhile if Phase
  3.3 lands.

**Phase 5 — Best Ball / Battle Royale: ~15% complete (different shape).**

- A `drafts/` package exists but contains the *legacy* Best Ball pipeline
  (`data_merge.py`, `data_process.py`, `forecast.py`, `train.py`,
  `update_ez_adp.py`), not the new modules the roadmap proposed.
- ❌ No `drafts/adp.py`, `drafts/projections.py`, `drafts/battle_royale.py`,
  `drafts/advance_equity.py`, `drafts/best_ball_optimizer.py`, or
  `drafts/exposure.py`. The roadmap effectively asks to replace this
  package; do that as a Phase 5 sub-phase rather than additively.

**Phase 6 — Modeling Refinements: 0% complete.**

- 6.1 Bayesian hierarchical, 6.2 NFL game sim, 6.3 News/weather feeds,
  6.4 Conformal prediction, 6.5 Push refinement audit — none started.
  Push handling itself is in production via 1.2; the *per-market support
  audit* and discrete-PMF refinement are not.

**Other landings worth noting (off-roadmap):**

- `helpers/archive.py` migrated from klepto to DuckDB (with bulk-deduped
  flush semantics) — see CLAUDE.md.
- Migration scripts under `scripts/` for archive, training data, gamelogs,
  correlations, and pickles → parquet.
- `tests/golden/` covers correlate, parlay search, CLV explode, pipeline
  snapshots, and CLI-help snapshots.

---

## How to use the prompts in this document

Every prompt is written for a Claude Code or GitHub Copilot Chat session inside the repo. Each one:

1. Tells the agent to read `CLAUDE.md` and `CONTRIBUTING.md` first. The repo has unusually strict rules (no monoliths, no commented-out code, no orphan methods, golden tests must pass) and surfacing those up front prevents rejected PRs.
2. Is scoped to one module per session. The repo's own `CLAUDE.md` mandates this.
3. Specifies canonical import paths (`sportstradamus.training`, `sportstradamus.prediction`, `sportstradamus.helpers`, `sportstradamus.stats`) so the agent doesn't recreate deleted shims.
4. States acceptance criteria so the agent knows what "done" looks like.

**Standing rules for every session, regardless of phase:**

- Open with: "Read `CLAUDE.md` and `CONTRIBUTING.md` first."
- Close with: "Run `poetry run ruff check src/sportstradamus/` and `poetry run pytest tests/golden/`. Both must pass."
- Scope to one module per session, then commit.
- When adding a CLI flag or command, regenerate snapshots: `REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py`
- When adding a dependency, specify the Poetry group (core, `[bayes]`, `[strategy]`, `[alerts]`) so it doesn't dump into core.
- Money values are always `Decimal`, never `float`.

---

## Phase 1 — Audit and Strengthen the Foundation

**Goal:** validate that the existing correlation/parlay/EV pipeline is actually doing what it claims to be doing, identify and fix methodology gaps, and add the observability + contest-variant features that the live code is missing.

This is mostly *audit work* — for each existing module the agent reads the code, writes findings, and proposes targeted improvements. The first session of every sub-phase is "audit and report"; only then do you do the surgery.

**Estimated time:** 3–4 weekends

### 1.1 Audit `training/correlate.py` — ✅ DONE (audit + fix shipped)

**Status:** audit at `docs/CORRELATE_AUDIT.md`. Improvements shipped:
8-game rolling residualization (`_residualize_gamelog`), stratified
per-team matrices, `--rebuild-correlations` flag on `meditate`. Original
audit prompts retained below for traceability.


The function exists and is called automatically as part of `meditate`'s "league setup" stage (per the package map in `CONTRIBUTING.md`). What we don't know without reading it:

- Does it compute correlations on **residuals** (player stat minus rolling mean) or on raw values? Raw correlations are almost always inflated by shared trend and produce systematically over-correlated parlay estimates.
- Does it apply a **minimum overlap threshold** for player pairs? Pairs with <30 shared games tend to produce noise that looks like signal.
- Does it apply **empirical Bayes shrinkage** toward zero based on overlap count?
- Does it distinguish **same-team, opposing-team, and cross-game** correlations? These three matter differently for Underdog Pick'em (same-team and opposing-team are stackable, cross-game is just diversification).
- Does it weight **recent seasons** higher? Player roles change; the 2022 Tyreek Hill–Tua correlation isn't the 2026 Tyreek Hill–Tua correlation.
- Does it write **metadata** alongside the CSV (date range, observation counts, generation timestamp)?

The audit prompt asks the agent to read the code, answer those questions, and write findings into `docs/CORRELATE_AUDIT.md`. The fix prompt is gated on the audit results — don't write it until you've seen the findings.

**Audit prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `src/sportstradamus/training/correlate.py` first. Then write a methodology audit at `docs/CORRELATE_AUDIT.md` answering:
>
> 1. **Inputs**: where does this function read its data from? (`Stats.gamelog`? a cached CSV? the archive?) What's the time window?
> 2. **Computation**: is the correlation computed on raw stat values or on residuals (de-meaned per player)? Is there any time-decay weighting?
> 3. **Pairwise filtering**: does it apply a minimum-overlap threshold? Is there shrinkage toward 0 for low-overlap pairs?
> 4. **Stratification**: does it separate same-team, opposing-team, and cross-game correlations? If so, are these stored as separate matrices?
> 5. **Output**: what does the CSV schema look like? Does it carry metadata (date range, sample sizes)?
> 6. **Consumers**: grep for callers. Is it called only inside `meditate`? Could it be called standalone?
> 7. **Recency**: when was the function last meaningfully changed? `git log --follow -p src/sportstradamus/training/correlate.py | head -200`. Is there a known issue in the commit history?
>
> For each question, quote the relevant lines (with line numbers) from the source. Do not propose changes yet. Do not modify any code. The output is the audit document only.
>
> When done: ruff clean (the doc-only change shouldn't trigger any lint), golden tests pass. Summarize findings in 5–10 bullets at the top of the audit doc.

**Improvement prompt (only after reading the audit):**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, the audit at `docs/CORRELATE_AUDIT.md`, and `src/sportstradamus/training/correlate.py` first. Then implement the targeted improvements identified by the audit.
>
> Target improvements (only apply those the audit confirms are missing):
>
> 1. **Residualization**: subtract each player's rolling-N-game mean (N=8 default) before computing pairwise correlations. The N goes in a module-level named constant.
> 2. **Minimum overlap shrinkage**: pairs with fewer than 30 shared games get correlation pulled toward 0 with strength proportional to (30 - overlap) / 30. The 30 goes in a named constant. Match the empirical Bayes pattern used elsewhere in `helpers/`.
> 3. **Stratified matrices**: produce three CSVs per league: `{LEAGUE}_corr_same_team.csv`, `{LEAGUE}_corr_opposing.csv`, `{LEAGUE}_corr_cross_game.csv`. Drop the existing single `{LEAGUE}_corr.csv` only after `prediction/correlation.py` is updated to read the new files.
> 4. **Metadata**: write `data/correlations/{LEAGUE}_corr_metadata.json` with date range, total game-pair observations, generation timestamp, git SHA.
> 5. **Add a `--rebuild-correlations` flag** to `meditate` (in `training/cli.py`) that runs only the correlation step and exits, bypassing the per-market training loop. Useful when the player pool changes (trades, injuries, season start) without needing a full retrain.
>
> Update `prediction/correlation.py:find_correlation` to read the new stratified CSVs. The function signature should not change; pick the right matrix internally based on whether the two legs are on the same team, opposing teams, or different games.
>
> Tests at `tests/golden/test_correlate.py`:
> - Residualization: a synthetic dataset where two players' raw values share a trend but their residuals are independent should produce a correlation near 0 after the change.
> - Shrinkage: a pair with only 5 shared games gets correlation closer to 0 than the same statistical relationship with 100 shared games.
> - Metadata file is written and contains the right keys.
> - The `--rebuild-correlations` flag runs without touching `data/models/`.
>
> Constraint: do not change the LightGBMLSS training pipeline. This work is confined to `training/correlate.py`, `training/cli.py`, `prediction/correlation.py`, and the new tests. Magic numbers go in named module-level constants per `STYLE_GUIDE.md` §8.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots since `--rebuild-correlations` is new. Run `poetry run meditate --rebuild-correlations --league NBA` end-to-end and confirm the new CSVs appear. Summarize the methodological changes and any audit findings that turned out not to need a fix.

### 1.2 Audit `prediction/correlation.py:find_correlation` and `beam_search_parlays` — ✅ AUDIT + ⚠️ FIXES PARTIAL

**Status:** audit at `docs/PARLAY_AUDIT.md`; calibration plot/CSV at
`docs/PARLAY_CALIBRATION_*`. **Note:** `prediction/parlay.py` was never split
out — `beam_search_parlays` lives in `prediction/correlation.py:504+`.
Shipped: Gaussian copula, `contest_variant` parameter (power/flex/insurance/
rivals), `data/underdog_payouts.json`, push-aware EV via
`_expected_payout_with_pushes`. **Open audit findings (Phase 1.2.x
follow-ups):**

- `find_correlation`'s pairwise EV `exp(C * sqrt(V_i V_j)) * p_i p_j` is
  not a probability and is unbounded. Wrap or replace with a bounded score.
- The Σ matrix passed to `multivariate_normal.cdf` is built pair-by-pair;
  singular submatrices are dropped rather than nearest-PSD-projected.
- Beam width 1000, EV cutoff 1.05, boost bands, and final EV floors are
  inline magic numbers — violates STYLE_GUIDE.md §8.
- `data/banned_combos.json` is a soft modifier; no pair is hard-banned.
  Decide whether banlist semantics are desired and add a hard-ban path.
- Same-player guarding is fragile substring matching. Switch to canonical
  `player_id` joins.
- The `Boost` column displayed downstream is overwritten with a *different*
  payout table than the one that drove ranking (`correlation.py:498`).
- `Sleeper`/`ParlayPlay`/`Chalkboard` payout tables are stub `1`s; either
  fill them in or gate Model EV on platform support.
- Dedup is by exact bet-id; overlap-aware dedup uses a 3-cluster Ward
  linkage but does not enforce one-per-family selection.
- Calibration plot in repo is a placeholder (no `parlay_hist.dat`); rerun
  `scripts/audit_parlay_calibration.py` on a production host with archive
  data and commit the result.



Both exist and run today. Open questions:

- **`find_correlation`**: when scoring a parlay candidate, how does it combine pairwise correlations into a joint probability? Is it using a **Gaussian copula**, a **simple Pearson product** (which is wrong for non-Gaussian marginals), or some other approximation?
- **`beam_search_parlays`**: what's the beam width? What's the candidate ordering? Are the Underdog-specific constraints (max 2 legs per game, push handling, payout multipliers per contest variant) baked in or generic?
- **Calibration check**: do the model's predicted joint probabilities match the empirically observed joint hit rates on past parlays? The audit script should answer this.

**Audit prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, `src/sportstradamus/prediction/correlation.py`, and `src/sportstradamus/prediction/parlay.py` first. Then write `docs/PARLAY_AUDIT.md` answering:
>
> **`find_correlation`**:
> 1. Given N legs and a correlation matrix, what's the exact formula for joint probability? Quote the lines.
> 2. Is the formula a Gaussian copula, a Pearson product, an additive log-odds combination, or something else?
> 3. How does it handle the case where two legs are on the same player (which is impossible per Underdog rules but should be guarded against)?
> 4. Does it use `data/banned_combos.json` to exclude specific pairs?
>
> **`beam_search_parlays`**:
> 1. What's the beam width? Is it configurable?
> 2. How are partial parlays scored at each step?
> 3. What constraints are enforced? (Max legs per game? Max parlay size? Min EV? Min correlation?)
> 4. Where do payout multipliers come from? Are they parametrized per contest variant (Power, Flex, etc.)?
> 5. Is the output ranked? Deduplicated against overlapping parlays?
>
> **Calibration check** (this part is empirical):
> - Write a one-off script `src/sportstradamus/scripts/audit_parlay_calibration.py` that:
>   1. Pulls the last 90 days of parlay candidates that the system would have produced (from the archive), grouped into deciles by predicted joint probability.
>   2. For each decile, computes the actual hit rate based on settled outcomes from `Stats.{league}.gamelog`.
>   3. Plots predicted-vs-actual and writes the result to `docs/PARLAY_CALIBRATION_{date}.png` plus a CSV with the decile data.
> - Run the script and include the resulting plot reference in the audit doc.
>
> Do not change `find_correlation` or `beam_search_parlays`. Audit only. The improvements come in the next session, gated on what we find.
>
> When done: ruff clean, golden tests pass. Summarize findings in 5–10 bullets, including whether the empirical calibration matches the predicted joint probabilities.

**Improvement prompt** (gated on audit findings):

> Read `CLAUDE.md`, `CONTRIBUTING.md`, the audit at `docs/PARLAY_AUDIT.md`, and the corresponding source files first. Then make the targeted fixes identified by the audit.
>
> Likely targets (only those the audit confirms are gaps):
>
> 1. **Switch from Pearson product to Gaussian copula** for joint-probability calculation if the audit shows the current method fails calibration. The marginals come from the trained LightGBMLSS distribution objects (which are stored in the model pickles and re-instantiated by `model_prob.py`). The copula sampling is 50K draws.
> 2. **Add Underdog contest-variant payouts** to a new `data/underdog_payouts.json` config and have `beam_search_parlays` accept a `contest_variant: Literal['power','flex','insurance','rivals']` parameter. The default stays Power for backward compatibility.
> 3. **Tighten constraints** if the audit shows the current beam search is producing too many low-correlation entries. Add a `min_pairwise_correlation` parameter (default 0.10) that requires at least one pair in the entry to exceed it.
> 4. **Push-aware EV** for integer-line markets. For each leg, compute `P(stat > line)`, `P(stat == line)` (the push probability), and `P(stat < line)`. Per Underdog rules a push drops the entry one leg, so the entry's expected payout is a weighted sum across the 2^k possible (over/push/under) outcomes. The marginals are already in the LightGBMLSS distribution; the integration changes from a single `cdf` call to two (the over and the >= line).
> 5. **Calibration recheck**: rerun the audit script after the changes; the new decile calibration should be tighter than before.
>
> Tests at `tests/golden/test_parlay_search.py`:
> - Gaussian copula: a 3-leg parlay with hand-computed joint probability matches the function's output within 1% at 50K samples.
> - Constraints: parlays with no correlated pair are filtered out when `min_pairwise_correlation` is set.
> - Push handling: a parlay with one integer-line leg has the right push-aware EV (verify against a known-correct hand calculation).
> - Contest variants: switching from Power to Flex changes the EV correctly for a 5-pick entry.
>
> Constraint: do not break existing callers of `find_correlation`. The new copula-based calculation should be wrapped behind a feature flag in `prediction/correlation.py` (default ON for new runs but with a `--legacy-correlation` escape hatch on `prophecize` for one release cycle).
>
> When done: ruff clean, golden tests pass, calibration plot shows tighter decile alignment than the audit baseline. Summarize.

### 1.3 Closing-line freeze — ⚠️ DROPPED (workaround is sufficient)

**Status:** no `Archive.freeze_close`, no `data/closing_lines/`, no
`freeze-close` CLI. `clv.fill_from_archive` treats the last archived
sample before kickoff as the close, and the Odds API stops surfacing
prematch odds for in-progress games, so this is correct in practice.
The original motivation for the explicit freeze step was the placed-bet
logger (Phase 2.2, now out of scope). Without that consumer, the
implicit close is good enough; revisit only if the CLV monitoring in
2.4 shows the assumption is breaking.



CLV tracking (Phase 2) requires a clean answer to the question "what was the line at game lock?" The archive captures everything but doesn't explicitly mark a snapshot as the closing line. Add an explicit freeze step.

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `src/sportstradamus/helpers/archive.py` first. Then add a closing-line freeze pipeline.
>
> Requirements:
>
> 1. Add a new method `Archive.freeze_close(event_id, freeze_ts)` that snapshots all line records for the event into a separate `data/closing_lines/{event_id}.json` file with the schema:
>    ```json
>    {
>      "event_id": "...",
>      "freeze_ts": "2026-09-12T17:00:00-04:00",
>      "commence_time": "2026-09-12T17:00:00-04:00",
>      "lines": [
>        { "book": "pinnacle", "player_id": "...", "market": "...",
>          "line": 248.5, "over_odds": -110, "under_odds": -110,
>          "devig_over": 0.524 }
>      ]
>    }
>    ```
> 2. Add a CLI command `poetry run freeze-close` that iterates all events whose `commence_time` is in the past 4 hours but in the future at the time of the most recent `freeze_close` for the event, and freezes them. Use a marker file `data/closing_lines/.last_freeze_check` to avoid double-processing.
> 3. Cron-friendly: idempotent if run twice. Designed to run every 60 seconds via systemd or launchd starting 30 minutes before each game.
> 4. Add `Archive.get_closing_line(event_id, player_id, market)` returning the frozen record or `None`.
>
> Tests at `tests/golden/test_archive_freeze.py`:
> - Freeze pulls the latest line in the archive at or before `freeze_ts`.
> - Freeze is idempotent: running it twice on the same event produces identical output.
> - `get_closing_line` returns `None` if the event hasn't been frozen.
>
> Constraint: do not modify the existing `Archive.write` or `Archive.get_line` paths. The freeze is additive.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Summarize.

### 1.4 Structured logging — ✅ DONE

**Status:** `src/sportstradamus/helpers/logging.py` is in place and used
across the CLI entry points (`reflect` log lines visible in `nightly.py`).



Right now scraper failures and pipeline issues are visible only in stdout if you happen to be watching. Add structured logging.

**Prompt:**

> Read `CLAUDE.md` and `CONTRIBUTING.md` first. Then add structured logging across the three main CLI entry points: `confer` (in `moneylines.py`), `meditate` (in `training/cli.py`), `prophecize` (in `prediction/cli.py`), `reflect` (in `nightly.py`), and the new `freeze-close`.
>
> Create `src/sportstradamus/helpers/logging.py` exposing `get_logger(name: str) -> logging.Logger` that:
>
> - Returns a logger writing JSON lines (one record per line) to `logs/{YYYY-MM-DD}/{cli_name}.jsonl`, with stdlib `logging.handlers.RotatingFileHandler` rotation at 50MB.
> - Uses a custom `JsonFormatter` emitting fields: `ts`, `level`, `module`, `message`, plus any extras passed via `extra={...}`.
> - Also writes WARNING and ERROR records to stderr in plain text.
> - Add to `helpers/__init__.py` re-exports.
>
> Migrate the existing `print()` calls to logger calls. Don't migrate `print()` calls inside `tqdm` callbacks — those are progress UI, not log output.
>
> Add a `--log-level` flag (`click.Choice(['DEBUG','INFO','WARNING','ERROR'])`, default INFO) to all five CLIs.
>
> Constraint: stdlib `logging` only. No `loguru`, no `structlog`.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Verify a sample run of `poetry run confer` produces a `logs/` file. Summarize.

### 1.5 Integration tests — ✅ DONE

**Status:** `tests/integration/test_end_to_end.py` plus fixtures live in
`tests/integration/`. Confirm coverage matches the original spec
(WNBA points, 10+ scored offers, ≥1 parlay candidate) when revisiting.



Golden tests cover CLI surface; integration tests catch wiring breaks. Add one end-to-end fixture-based test.

**Prompt:**

> Read `CLAUDE.md` and `CONTRIBUTING.md` first. Then add `tests/integration/` with one end-to-end test that runs `confer → meditate --league WNBA → prophecize` on cached fixtures.
>
> Requirements:
>
> - Use a fixture-mode flag I'll add to `confer` (`--fixture-dir tests/integration/fixtures/`) that reads canned API responses from disk instead of hitting the live Odds API. Implement that flag in `moneylines.py`.
> - The test trains on a small WNBA fixture (one market: points; ~500 player-games), runs prediction, asserts the Google Sheets export step is mocked (don't actually hit Google), and checks that EV is computed for at least 10 offers and at least one parlay candidate is returned.
> - The fixture data lives under `tests/integration/fixtures/` as small JSON/CSV files committed to the repo.
> - The test runs in under 90 seconds.
> - Add `pytest -m integration` mark; default `pytest tests/golden/` should not run it.
>
> Constraint: do not modify production code paths to support fixture mode beyond the `--fixture-dir` flag. The flag should branch in `moneylines.get_props` only; everything downstream must be path-identical to a live run.
>
> When done: ruff clean, golden tests pass, integration test passes. Summarize.

### 1.6 Audit `src/deprecated/` — ❌ NOT DONE

**Status:** no `docs/DEPRECATED_TRIAGE.md`. `src/deprecated/` still
contains `correlation.py`, `opt_parlay.py` (both superseded by live code),
`opt_kelley_bet.py` (REVIVE candidate for Phase 3.1), and the rest of
the 2026-04 sweep. The triage decisions are still owed.



The deprecated directory has TODOs that are partly outdated (the correlation generator and parlay search are no longer missing — they were replaced). Make explicit decisions for each.

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `src/deprecated/README.md`. Then for each file in `src/deprecated/` and each TODO listed at the bottom of `README.md`, check whether a live replacement exists in the current package layout (per `CONTRIBUTING.md` Package Map). For each, write a one-paragraph decision in a new doc `docs/DEPRECATED_TRIAGE.md`:
>
> ```
> ## opt_kelley_bet.py
> Decision: REVIVE (Phase 3)
> Replacement exists in production: NO
> Rationale: Kelly sizing is genuinely missing from the live codebase.
> Owner: <user>
>
> ## correlation.py
> Decision: DELETE
> Replacement exists in production: YES — training/correlate.py
> Rationale: live module supersedes this. Older alternative; keep git history only.
>
> ## opt_parlay.py
> Decision: DELETE
> Replacement exists in production: YES — prediction/parlay.py:beam_search_parlays
> Rationale: live module supersedes this with beam search. Older brute-force enumeration; keep git history only.
> ```
>
> Decisions are: REVIVE (with target phase from `docs/ROADMAP.md`), DELETE (with rationale), or ARCHIVE PERMANENTLY (with rationale).
>
> For each DELETE decision, also remove the corresponding entry from the README's TODO list.
>
> Do not change any code in `src/deprecated/` yet. Do not actually delete anything. The triage doc is decision-only; deletions happen in a follow-up session after I review.
>
> When done: ruff clean, golden tests pass. Summarize the decisions: how many REVIVE, how many DELETE, how many ARCHIVE.

---

## Phase 2 — Bet Logger and CLV Tracker

**Goal:** measure whether the model actually beats the close. Without CLV, ROI alone is too noisy at recreational scale to tell signal from luck. Closing-line value is the standard quant-betting metric for separating skill from variance — and it's the one diagnostic the existing `nightly.py:reflect` does not produce.

This phase extends `reflect` rather than building a parallel system, since `reflect` already does historical parlay performance.

**Estimated time:** 1–2 weekends

### 2.1 Audit `nightly.py:reflect` — ✅ SUPERSEDED

**Status:** the audit doc was never produced because `clv.py` was built
directly. `reflect` now (a) resolves history, (b) calls
`clv.fill_from_archive`, (c) prints `clv.summarize` + per-segment results
honoring `CLV_SEGMENT_MIN_N=20`. Adequate for this package's scope.

### Phase 2.2 / 2.3 — Placed-bet logger and per-bet CLV — ❌ OUT OF SCOPE

Tracking which entries the user actually placed is a personal-bookkeeping
concern. Users own that themselves. The remainder of this Phase 2 section
(the SQLAlchemy schema, `track place|close|settle` CLIs, the bet-history
dashboard tabs) is intentionally not built. Original prompts retained
below for historical reference only — do not implement.



**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `src/sportstradamus/nightly.py` first. Then write a short audit at `docs/REFLECT_AUDIT.md` answering:
>
> 1. What does `reflect` actually compute? Quote the relevant lines.
> 2. What inputs does it read? (The archive? Past Google Sheets exports? Cached pickles?)
> 3. Does it know which entries were actually placed by the user, or only which entries the system recommended?
> 4. Does it compute CLV? If it tracks anything closing-line-related, what?
> 5. What's the output format?
>
> No code changes. The audit informs whether Phase 2.2 extends `reflect` or builds alongside it.

### 2.2 Bet logger schema — ❌ OUT OF SCOPE (see note above)



Build the placed-bet logger as a sibling to `reflect`, not a replacement. `reflect` analyzes "what would the system have done"; the new tracker analyzes "what did I actually do."

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `docs/REFLECT_AUDIT.md` first. Then create a new module `src/sportstradamus/tracking/` with three files: `schema.py`, `place.py`, and `__init__.py` exposing a `track` CLI.
>
> `schema.py` — SQLAlchemy 2.0 declarative models stored at `data/tracking.db` (SQLite). Tables:
>
> ```python
> class Entry:
>     id: str (uuid4)
>     placed_at: datetime
>     contest_type: str         # 'underdog_pickem_3leg_power', 'underdog_pickem_5leg_flex', etc.
>     platform: str
>     stake: Decimal
>     payout_multiplier: Decimal
>     legs: list[Leg]
>     modeled_ev: float          # at placement time
>     modeled_joint_prob: float  # joint probability all legs hit
>     model_version: str         # SHA of model pickle used for the prediction
>     settled: bool
>     payout: Decimal | None
>     notes: str | None
>
> class Leg:
>     entry_id: str (FK)
>     leg_idx: int
>     sport: str
>     player_id: str
>     player_name: str
>     market: str
>     line: float
>     side: str                  # 'over' | 'under' | 'higher' | 'lower'
>     model_prob_at_placement: float
>     sharp_devig_prob_at_placement: float | None
>     closing_line: float | None         # populated by Phase 2.3
>     closing_devig_prob: float | None
>     clv: float | None
>     actual_outcome: float | None
>     won: bool | None
>     pushed: bool                       # Underdog drops a pushed leg
>     voided: bool
> ```
>
> Hand-write the initial schema as `tracking/migrations/001_initial.sql`; run on first import. No Alembic.
>
> `place.py` — function `place_entry(entry: dict) -> str` that inserts an Entry + Legs and returns the entry id. Snapshot model probability and sharp devig at insertion time using `prediction.model_prob.model_prob` and `helpers.distributions.no_vig_odds` against `Archive.get_line(...)`. Store the model SHA: read it from the model pickle's metadata if present, otherwise hash the pickle file content.
>
> CLI in `tracking/__init__.py` exposing `track place --from <yaml-file>` (use `click`). Sample YAML:
>
> ```yaml
> contest_type: underdog_pickem_3leg_power
> platform: underdog
> stake: 5.00
> payout_multiplier: 6.0
> legs:
>   - sport: NFL
>     player_id: "00-0036442"
>     market: player_pass_yds
>     line: 248.5
>     side: over
> ```
>
> `pyproject.toml` script: `track = "sportstradamus.tracking:cli"`.
>
> Tests at `tests/golden/test_tracking_place.py`:
> - Successful placement.
> - Missing model probability — log WARNING, store None, succeed.
> - Missing sharp probability — log WARNING, store None, succeed.
> - Duplicate entry by client-side hash is rejected.
> - Model SHA is captured.
>
> Constraints:
> - All money values are `Decimal`.
> - Do not modify `prediction/` or `moneylines/` — only consume their existing public APIs.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Verify `poetry run track place --from examples/sample_entry.yaml` works against a fresh DB. Summarize.

### 2.3 CLV computation and settlement — ✅ DONE for in-scope work

**Status:** retrospective CLV exists for system-recommended history
(`clv.fill_from_archive` + `_signed_clv`). The placed-bet `track close` /
`track settle` flow is dropped (see 2.2 note).



**Prompt:**

> Read `CLAUDE.md` and the existing `src/sportstradamus/tracking/place.py` first. Then add `tracking/close.py` and `tracking/settle.py` with corresponding CLI subcommands.
>
> `track close` requirements:
>
> - Iterate all entries with `settled=False` and any leg with `closing_line=None`.
> - For each leg, fetch the closing line via `Archive.get_closing_line(event_id, player_id, market)` (added in Phase 1.3).
> - If the closing line is from a sharp book (Pinnacle, Circa), de-vig directly. If the closing-line snapshot only has soft books, take a `book_weights.json`-weighted consensus and de-vig that.
> - Compute `closing_devig_prob` and `clv = closing_devig_prob - model_prob_at_placement`. Sign correctly for the bet side.
> - Idempotent.
>
> `track settle` requirements:
>
> - For each entry with all legs having `closing_line` set and at least one leg with `actual_outcome=None`, look up the player's actual stat for the game via `Stats.{league}.get_player_game_stat(player_id, market, game_id)`. Add this thin lookup method to `Stats` if it doesn't exist; it should query the existing `data/player_data/` cache, not hit a live API.
> - Determine `won` per leg. Apply Underdog push/void rules:
>   - Push: `pushed=True`, the leg drops, the entry's effective size and payout multiplier shift accordingly.
>   - Void: `voided=True`, leg drops, entry treated as if it were one leg shorter.
> - Compute the final `payout` using the entry's `contest_type` payout structure (lookup in `data/underdog_payouts.json`, added in Phase 1.2).
> - Set `settled=True`.
>
> Tests at `tests/golden/test_tracking_close_settle.py`:
> - Closing successfully on a fully archived event.
> - No sharp closing line available — store None, log WARNING, continue.
> - Settle with a pushed leg — entry treated as one leg shorter, payout adjusted.
> - Settle with a voided leg — same, payout adjusted differently per Underdog's rules.
> - Settle with all legs winning — full payout.
>
> Constraint: do not add a new dependency for sportsbook fetching. Use only `Archive`. If the archive doesn't have the closing snapshot (Phase 1.3 hadn't run), mark `closing_line=None` and continue.
>
> When done: ruff clean, golden tests pass. Summarize.

### 2.4 CLV reporting — ✅ DONE for in-scope work; monitor

**Status:** `clv.summarize` produces per-(League × Market × Platform)
segment tables logged by `reflect`. Bootstrap CIs and CSV/JSON exports
are nice-to-haves but not blockers — defer until the existing summary
shows it can't answer a real question.

**Things to keep an eye on:**

- `frac_beat_close` should hover near 50% on segments with no real edge
  and >50% on segments where the model is sharp. Persistent <50% on a
  segment is either an archive/close-snapshot bug or a model bug.
- `Market CLV mean` and `Model CLV mean` drift apart only when
  bookmakers move toward (or away from) our model after open. Big
  divergence on a market is a signal to retrain or to investigate the
  feature set.
- Segment counts hitting `CLV_SEGMENT_MIN_N=20` — if a segment never
  accumulates 20 legs in a season, either it's not surfacing in the
  recommendation pipeline or the threshold is too high for low-volume
  markets.
- The implicit "last archive sample before kickoff = close"
  assumption: if Odds API behavior changes (they start surfacing
  prematch lines into in-play, or stop providing some sharp books
  prematch), `clv.fill_from_archive` silently breaks. Add a sanity
  check that the close timestamp is within N minutes of `commence_time`
  and warn otherwise.



**Prompt:**

> Read `CLAUDE.md` and the new `tracking/close.py` and `tracking/settle.py` first. Then add `tracking/report.py` and a `track report` CLI subcommand.
>
> Requirements:
>
> - Aggregate settled entries by `contest_type`, `sport`, `market`, and the cross of (sport × market). For each segment with ≥50 entries:
>   - Mean CLV per leg
>   - 95% bootstrap CI of mean CLV (10K resamples, percentile method, set seed for reproducibility)
>   - Mean ROI
>   - Win rate
>   - Sample size
> - Segments with <50 entries: report "insufficient sample" with current count.
> - Output formats: pretty terminal table by default, `--csv path.csv` and `--json path.json` flags.
> - Highlight (terminal: bold green/red) segments where the 95% CLV CI excludes zero.
>
> Constraint: do not pull in pandas or scipy if they're not already runtime deps (check `pyproject.toml`). They are; you can use them.
>
> Tests at `tests/golden/test_tracking_report.py`:
> - Segment with positive CLV.
> - Segment with negative CLV.
> - Segment with insufficient sample.
> - CSV/JSON export round-trip.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Run `track report` against the fixture DB and paste the output in your summary.

---

## Phase 3 — Underdog-Specific Decision Engine

**Goal:** turn EV signals into ranked Underdog entries with proper bankroll sizing, accounting for Underdog's five contest variants. The existing `prediction/parlay.py` produces parlay candidates against a generic payout structure; this phase makes them Underdog-native.

**Estimated time:** 4–5 weekends

### 3.1 Kelly sizing module — ❌ NOT DONE (still in deprecated/)



`opt_kelley_bet.py` is in `src/deprecated/`. Revive and modernize it.

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `src/deprecated/opt_kelley_bet.py` first. Then create `src/sportstradamus/strategies/kelly.py` with the deferred Kelly logic modernized.
>
> Public API:
>
> ```python
> def fractional_kelly_stake(
>     bankroll: Decimal,
>     win_prob: float,
>     payout_multiplier: Decimal,
>     fraction: float = 0.25,
>     model_calibration: float = 1.0,
>     max_fraction_of_bankroll: float = 0.005,
> ) -> Decimal:
>     """Returns recommended stake. Returns 0 if -EV."""
>
> def joint_kelly_portfolio(
>     bankroll: Decimal,
>     candidates: list[KellyCandidate],
>     fraction: float = 0.25,
> ) -> dict[str, Decimal]:
>     """Solve for joint Kelly allocations across a candidate set. Uses cvxpy."""
> ```
>
> Requirements:
>
> - Fractional Kelly, default 0.25. Cap at `max_fraction_of_bankroll` per entry.
> - Effective probability shrinkage: `effective_p = 0.5 + (win_prob - 0.5) * model_calibration`. The `model_calibration` value is read from the training report's `model_calib` field for the relevant market — pull it via `training.report.get_market_calibration(league, market)` (add this thin getter to `training/report.py`).
> - All math in `Decimal`.
> - Portfolio variant: `max sum(log(1 + r_i * x_i))` s.t. `sum(x_i) ≤ fraction * bankroll`, `x_i ≥ 0`. Use cvxpy SCS solver. Add cvxpy to `[tool.poetry.group.strategy]`, not core.
>
> Add a CLI: `poetry run kelly --bankroll 500 --from data/recommendations/{date}.yaml` that loads parlay candidates from the YAML produced by `prediction/parlay.py`'s output (see Phase 3.3) and prints a table of (candidate, EV, recommended stake).
>
> Tests at `tests/golden/test_kelly.py`:
> - -EV bet returns 0.
> - +EV bet returns the analytically correct fractional Kelly.
> - Cap is enforced when uncapped Kelly exceeds 0.5%.
> - Portfolio with two independent +EV bets returns positive allocations summing ≤ fraction × bankroll.
> - Calibration shrinkage works: a 0.6 win prob with calibration 0.5 should produce stake equivalent to a 0.55 win prob with calibration 1.0.
>
> Constraint: do not call this from any other module yet. Wiring into the parlay pipeline happens in Phase 3.3.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Summarize. Move `src/deprecated/opt_kelley_bet.py` to `src/deprecated/.archived/` per the deprecated README protocol once the new code is shipped.

### 3.2 Underdog contest-variant payouts — ✅ DONE

**Status:** `data/underdog_payouts.json` covers power/flex/insurance/rivals;
`beam_search_parlays(contest_variant=...)` consumes it. Verify multipliers
against the current Underdog product on each major release.



Underdog has five variants with different payouts. The existing parlay system treats Power as the default. Make all five first-class.

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `src/sportstradamus/prediction/parlay.py` first. Then add Underdog-specific payout config and contest-variant handling.
>
> Create `src/sportstradamus/data/underdog_payouts.json` with the current payout structure for each contest variant. Include at minimum:
>
> ```json
> {
>   "power": {
>     "2": { "all_hit": 3.0 },
>     "3": { "all_hit": 6.0 },
>     "4": { "all_hit": 10.0 },
>     "5": { "all_hit": 20.0 },
>     "6": { "all_hit": 35.0 }
>   },
>   "flex": {
>     "3": { "all_hit": 2.25, "minus_one": 1.25 },
>     "4": { "all_hit": 5.0, "minus_one": 1.5 },
>     "5": { "all_hit": 10.0, "minus_one": 2.0, "minus_two": 0.4 },
>     "6": { "all_hit": 25.0, "minus_one": 2.0, "minus_two": 0.4 }
>   },
>   "rivals": {
>     "2": { "all_hit": 3.0 },
>     "3": { "all_hit": 6.0 }
>   }
> }
> ```
>
> Verify the exact multipliers against the current Underdog product (the values above are illustrative; pull current ones from a fresh app screenshot or the help-center pages cited in the strategy doc).
>
> Modify `prediction/parlay.py:beam_search_parlays` to accept a `contest_variant: Literal['power','flex','rivals']` parameter (default 'power' for backward compat). The EV calculation now needs to handle Flex's "miss one or two" branches:
>
> ```python
> # Power: pays only if all legs hit
> ev_power = joint_p_all_hit * payout - 1
>
> # Flex on 5-pick: pays full if 5/5, partial if 4/5
> ev_flex_5 = (
>     p_all_5_hit * payout_all
>     + p_exactly_4_hit * payout_minus_1
>     - 1
> )
> ```
>
> For Flex, you need partial-hit probabilities, not just the joint all-hit probability. Compute these via the same Gaussian copula sampling: count fractions of 50K samples where exactly k of the n legs hit.
>
> Tests at `tests/golden/test_parlay_variants.py`:
> - Power EV matches the existing implementation when `contest_variant='power'`.
> - Flex 5-pick EV is higher than Power 5-pick EV when leg probabilities are around 55% (Flex's downside protection helps in marginal-edge regimes).
> - Flex 5-pick EV is lower than Power 5-pick EV when leg probabilities are around 70% (Power's higher multiplier wins when legs are sharper).
>
> Constraint: do not break callers that don't pass `contest_variant`. The default behavior must be unchanged.
>
> When done: ruff clean, golden tests pass. Summarize.

### 3.3 Underdog-native strategy module — ❌ NOT DONE



Pull everything together into a CLI that produces ranked, Kelly-sized recommendations.

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, the new `strategies/kelly.py`, and the updated `prediction/parlay.py` first. Then create `src/sportstradamus/strategies/underdog_pickem.py`.
>
> Public API:
>
> ```python
> @dataclass
> class PickemConfig:
>     min_model_edge: float = 0.020
>     min_sharp_edge: float = 0.015
>     disagreement_threshold: float = 0.04
>     min_correlation: float = 0.10
>     min_ev: float = 0.05
>     entry_sizes: tuple[int, ...] = (3, 5)
>     contest_variants: tuple[str, ...] = ('power', 'flex')
>     top_k: int = 20
>     max_overlap: int = 2
>     kelly_fraction: float = 0.25
>     max_stake_pct_bankroll: float = 0.005
>
> def construct_entries(
>     date: datetime.date,
>     bankroll: Decimal,
>     config: PickemConfig | None = None,
> ) -> list[RecommendedEntry]:
>     """Pull today's offers, filter, search across variants, size, return top-K."""
> ```
>
> The function:
>
> 1. Loads today's offers from the same source `prophecize` reads.
> 2. Filters to Underdog markets where the model has coverage AND sharp devig is available AND model+sharp agree within `disagreement_threshold` AND both clear their respective edge thresholds.
> 3. Calls `prediction/parlay.py:beam_search_parlays` once per (entry_size, contest_variant) combination.
> 4. For each returned candidate, calls `strategies/kelly.py:fractional_kelly_stake` with the calibration from the training report.
> 5. Writes to two sinks:
>    - Extends `prediction/sheets.py` with a "Pickem Recommendations" tab that includes a "Variant" column.
>    - YAML at `data/recommendations/{date}.yaml` formatted for `track place`.
>
> CLI: `poetry run pickem-build --date today --bankroll 500`. Defaults from `data/pickem_config.json` and `data/bankroll.json`.
>
> Tests at `tests/golden/test_underdog_pickem.py`:
> - Filtering excludes legs failing each threshold.
> - The disagreement threshold catches a model-vs-sharp divergence and skips the leg.
> - Both Power and Flex variants appear in the output when both are enabled in config.
> - End-to-end on a small fixture produces a non-empty YAML output.
>
> Constraint: this module orchestrates other modules; algorithm logic stays in `kelly.py`, `parlay.py`, and `correlation.py`. If you find yourself writing math here, move it.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Summarize.

### 3.4 Pick'em Champions — ❌ NOT DONE



In Champions, the static-line-vs-sharp arbitrage doesn't apply because you're playing other users, not the house. Different optimization.

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `src/sportstradamus/strategies/underdog_pickem.py` first. Then create `src/sportstradamus/strategies/underdog_champions.py`.
>
> Champions is peer-to-peer; the prize pool is pari-mutuel. The optimization changes:
>
> 1. Instead of EV against a static line, use **expected percentile rank in the field's score distribution**.
> 2. Estimate field behavior from the popularity weighting Underdog itself shows in the app (every offer has a "more popular side" indicator). For each leg, an estimated `field_pick_rate` ∈ [0, 1].
> 3. For each candidate entry, simulate 100K opponent entries by sampling legs according to field popularity. Score each opponent's hits using the model's joint distribution. Score your candidate the same way. Compute your candidate's percentile rank distribution.
> 4. Convert percentile rank to expected prize equity using the contest's pari-mutuel structure (top X% cash, prize curve in `data/underdog_payouts.json` under `champions`).
>
> Public API:
>
> ```python
> def construct_champions_entries(
>     date: datetime.date,
>     bankroll: Decimal,
>     config: ChampionsConfig | None = None,
> ) -> list[RecommendedEntry]:
>     ...
> ```
>
> CLI: `poetry run champions-build --date today --bankroll 500`.
>
> Where to get `field_pick_rate`: this requires scraping the Underdog Champions board's popularity indicators. Add this to the existing `books.get_ud()` scraper as an optional field. If it's unavailable, fall back to ADP-style estimation: bias toward "Higher" for stat lines below the player's season average, "Lower" for above (people pick Higher more often, ~58% of legs). Document this fallback in the code.
>
> Tests at `tests/golden/test_underdog_champions.py`:
> - Contrarian entries score higher than chalky entries when both have equal model EV (because contrarian gets you to top percentiles when chalky tickets clump).
> - With fallback field_pick_rate, output is non-empty and reasonable.
> - End-to-end on fixture produces YAML output.
>
> Constraint: this is a separate strategy module from `underdog_pickem.py`. Do not merge.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Summarize.

### 3.5 Streaks, Ladders, Rivals — ❌ NOT DONE



These three contest variants have very different payout structures and require their own logic. Streaks especially is a sequential decision problem.

**Prompt (Streaks):**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `src/sportstradamus/strategies/underdog_pickem.py` first. Then create `src/sportstradamus/strategies/underdog_streaks.py`.
>
> Streaks is a sequential parlay: pick up to 11 consecutive correctly, with payouts up to 1000x. Cash-out is allowed at intermediate streak lengths. The decision is: at each step, given the current streak length `n`, do I add another pick (continue) or cash out (stop)?
>
> Algorithm:
>
> 1. The decision at streak length `n` is to continue iff `E[payout | continue] > payout_at_n`. With the geometric payout structure, this is essentially: continue iff next-leg win probability > some threshold derived from the payout ratios.
> 2. Compute the threshold from `data/underdog_payouts.json` under `streaks`.
> 3. For each potential next pick, score its win probability using the model. Recommend continuing only if the best available pick exceeds the threshold by a margin (default 2%).
> 4. The first two picks must be from different teams (Underdog rule). Voids/ties replaced by next pick after pick 2.
>
> Public API:
>
> ```python
> def recommend_streak_action(
>     current_streak: int,
>     used_team_ids: list[str],   # for the two-different-teams rule
>     available_offers: list[Offer],
>     config: StreaksConfig,
> ) -> StreakRecommendation:  # action: 'continue' with offer, or 'cash_out'
>     ...
> ```
>
> CLI: `poetry run streaks-recommend --streak 4 --used-teams KC,SF`.
>
> Tests:
> - At streak length 1 with a 60% available pick, recommend continue.
> - At streak length 8 with only 52% available picks, recommend cash out.
> - Two-different-teams rule is enforced for picks 1 and 2.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Summarize.

I'm leaving Ladders and Rivals as follow-up sessions — same shape of work but smaller scope. Ladders needs a "lowest-shared-rung" expected payout calculation; Rivals is essentially a 2-pick or 3-pick Power with player-vs-player matchups (the hardest part is making sure your model has both players covered for the same market).

---

## Phase 4 — Real-Time Alerts and Dashboard Extensions

**Goal:** push high-edge opportunities to your phone the moment they appear and extend the existing Streamlit dashboard for review-and-act workflows.

**Estimated time:** 2 weekends

### 4.1 Alerts package — ❌ NOT DONE



**Prompt:**

> Read `CLAUDE.md` and `CONTRIBUTING.md` first. Then create `src/sportstradamus/alerts/` with `rules.py`, `dispatcher.py`, `telegram.py`, `discord.py`.
>
> Architecture:
>
> ```python
> @dataclass
> class AlertEvent:
>     id: str                # idempotency key
>     rule_name: str
>     priority: Literal['low','medium','high','critical']
>     title: str
>     body: str
>     metadata: dict
>
> class AlertRule(Protocol):
>     name: str
>     poll_interval_seconds: int
>     def should_fire(self, state: AppState) -> AlertEvent | None: ...
>
> class Dispatcher:
>     def __init__(self, rules: list[AlertRule], channels: list[AlertChannel]):
>         ...
>     def run_forever(self):
>         """Async loop polling each rule at its interval."""
> ```
>
> Initial rules:
> - `HighEdgeAppearanceRule(min_model_edge=0.04, min_sharp_edge=0.03)` — reads from `data/recommendations/{today}.yaml`.
> - `InjuryNewsExposureRule()` — reads open entries from `tracking.db`, polls news feed (Phase 6.3 will populate this; for now it's a no-op until news feed exists).
> - `BankrollDrawdownRule(threshold_pct=0.20)` — reads bankroll history.
> - `CLVDriftRule(window=50, max_negative_clv=0.0)` — reads from `tracking.db`.
> - `ScraperFailureRule(min_offers=50)` — reads logs from Phase 1.4.
> - `ModelStaleRule(max_days_old=14)` — checks `data/models/` mtimes.
>
> Channels:
> - Telegram and Discord. Read tokens from `creds/keys.json` (extend the existing schema).
> - For each event: title + body, deep link to the dashboard.
>
> Idempotency: 24-hour in-memory dedup cache keyed on event id.
>
> Add a CLI: `poetry run alert-watch` (foreground only).
>
> Tests at `tests/golden/test_alerts.py`:
> - Each rule fires when condition met (use a fake AppState).
> - Each rule does NOT fire when condition not met.
> - Dispatcher dedupes within 24h.
> - Telegram/Discord channels mock HTTP and verify payload format.
>
> Constraint: alerts read from `tracking.db` and cached files only — they don't call into `prediction/` while it might be mid-run.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Demonstrate by triggering a fake event against a Telegram test chat. Summarize.

### 4.2 Dashboard extensions — ⚠️ SCOPE-REDUCED

**Status:** baseline `dashboard.py`, `dashboard_app.py`, `dashboard_data.py`,
and `pages/` exist. The Open Entries / Settled History / Bankroll tabs
are dropped along with the placed-bet logger. A Today's Recommendations
tab reading `data/recommendations/{today}.yaml` (produced by Phase 3.3)
is still worth building once that module lands.



**Prompt:**

> Read `CLAUDE.md` and the existing `dashboard.py` and `dashboard_app.py` first. Then add tabs to the Streamlit app.
>
> New tabs:
>
> 1. **Today's Recommendations** — read from `data/recommendations/{today}.yaml`. Table with: rank, contest_type, variant (Power/Flex/Champions), modeled_ev, joint_prob, recommended_stake, top correlated pair, "Place" button. Click a row to expand leg-by-leg detail with sharp comparison and correlation callouts. "Place" button calls `tracking.place_entry()` and removes the row.
> 2. **Open Entries** — `tracking.db` query, all `settled=False` entries.
> 3. **Settled History** — `tracking.db` query, with CLV column color-coded.
> 4. **Bankroll** — current balance, peak, drawdown%, today's pending P&L. Pulled from `tracking.db`.
>
> Auto-refresh every 60 seconds.
>
> Constraints:
> - No new dependencies.
> - Wrap any DB read >200ms in `@st.cache_data(ttl=30)`.
>
> Smoke test at `tests/integration/test_dashboard_smoke.py` that imports the dashboard module without errors and asserts new tab functions exist.
>
> When done: ruff clean, smoke test passes. Run the dashboard locally and screenshot the new tabs in your summary.

---

## Phase 5 — Best Ball and Battle Royale

**Goal:** play Underdog's draft products, which the existing system completely ignores. NFL-season-aligned: Best Ball drafts run March–August, Battle Royale weekly during the regular season.

**Estimated time:** 8–12 weekends, season-aligned

### 5.1 ADP ingestion — ❌ NOT DONE (legacy `update_ez_adp.py` exists)

**Status:** `drafts/update_ez_adp.py` is a legacy ADP fetcher that does
not match this spec (no parquet, no stochastic ADP, no contest-aware
output). Decide whether to retire and replace, or evolve in place. The
roadmap target presupposes replacement.



**Prompt:**

> Read `CLAUDE.md` and `CONTRIBUTING.md` first. Then create `src/sportstradamus/drafts/adp.py` for Underdog Best Ball ADP ingestion.
>
> Sources (try in order):
> 1. Underdog's public ADP page at `underdogfantasy.com/adp` (HTML, use existing `Scrape` helper from `helpers/scraping.py`).
> 2. FantasyLife's republished ADP (HTML, fallback).
> 3. Manual CSV at `data/adp/manual/{contest}_{date}.csv`.
>
> Output: `data/adp/{contest_slug}/{YYYY-MM-DD}.parquet` with columns `player_id, player_name, position, team, mean_pick, stdev_pick, count_drafted, total_drafts, source, ingested_at`. Stochastic ADP — store stdev, not just mean.
>
> CLI: `poetry run drafts-adp-update --contest best_ball_mania_VII`. Idempotent.
>
> Tests at `tests/golden/test_adp.py`:
> - HTML parsing on captured fixture extracts right player count and ADP.
> - Manual CSV path round-trips.
> - Player ID resolution against `nfl_data_py` succeeds for 10 sample players.
>
> Constraints: no live HTTP in tests (`tests/golden/fixtures/adp_underdog_sample.html`). ADP module is leaf-dependency only.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Run against today's live ADP and report player count.

### 5.2 Season-long projection distributions — ❌ NOT DONE

**Status:** legacy `drafts/forecast.py` exists but is structured around
the older training pipeline. The new `drafts/projections.py` is unbuilt.



**Prompt:**

> Read `CLAUDE.md` and the existing `Stats` classes first. Then create `src/sportstradamus/drafts/projections.py` that translates the existing distributional models into season-long weekly fantasy point distributions.
>
> Public API:
>
> ```python
> def project_season(
>     league: str = "NFL",
>     contest: str = "best_ball_mania_VII",
>     n_seasons: int = 10_000,
>     seed: int = 42,
> ) -> SeasonProjections:
>     """
>     Returns a 3D array (player, week, sample) of fantasy points,
>     plus metadata: player_ids, week_numbers, scoring_system.
>     """
> ```
>
> Algorithm:
>
> 1. For each player, for each week, pull the marginal distributions from the existing LightGBMLSS models for relevant markets (pass_yds, pass_tds, rush_yds, rush_tds, receptions, receiving_yds, receiving_tds for skill players; goalie save percentage etc. for other sports).
> 2. Convert to fantasy points using the contest's scoring system (Underdog half-PPR for NFL — load from `data/contests/{contest}.json`).
> 3. Sample `n_seasons` × 17 weeks of joint outcomes per player, accounting for week-to-week correlation (a player who's in form one week is more likely in form the next — model this as an AR(1) on week-over-week residuals).
> 4. Cache results to `data/projections/{contest}_{date}.parquet` for reuse across roster evaluations.
>
> Tests at `tests/golden/test_projections.py`:
> - Deterministic with fixed seed.
> - For a star RB, mean weekly fantasy points across the n_seasons samples is within 5% of consensus projections (sanity check against industry projections — store a fixture in `tests/golden/fixtures/consensus_projections.json`).
> - AR(1) auto-correlation on week-over-week residuals is non-trivial.
>
> Constraint: this module is a translation layer; it doesn't define a new model. All distributional information comes from existing pickles.
>
> When done: ruff clean, golden tests pass. Run for NFL and report total player-weeks projected.

### 5.3 Battle Royale optimizer

Smaller scope than Best Ball. Ship this before tackling advance equity.

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `src/sportstradamus/drafts/projections.py` first. Then create `src/sportstradamus/drafts/battle_royale.py`.
>
> Battle Royale: 6-player snake draft per pod, scored against tens of thousands of pods. Top ~10% of pods cash; ~10% of prize pool to first.
>
> Public API:
>
> ```python
> def optimize_battle_royale(
>     slate: list[Player],          # players available in this week's slate
>     ownership_estimates: dict[str, float],  # ADP-derived field ownership
>     n_field_pods: int = 50_000,
>     n_my_rosters: int = 100,
>     seed: int = 42,
> ) -> list[OptimizedRoster]:
>     """
>     Generate top-N roster constructions ranked by expected prize equity.
>     """
> ```
>
> Algorithm:
>
> 1. Project each player's weekly fantasy point distribution using `drafts.projections.project_week()` (single-week version of project_season).
> 2. Simulate `n_field_pods` field pods, each drafting via ADP-following opponents.
> 3. Generate `n_my_rosters` candidate roster constructions for the user's pod via heuristic templates (single-stack, double-stack, naked-QB, RB-heavy, etc.).
> 4. For each candidate, compute the percentile rank distribution against field scores.
> 5. Convert percentile to prize equity using the payout curve.
> 6. Rank by expected prize equity, not by median percentile.
>
> Stacking is mandatory: no roster with fewer than 2 players from one team is included.
>
> CLI: `poetry run battle-royale-build --slate sunday-main`.
>
> Tests at `tests/golden/test_battle_royale.py`:
> - Deterministic with fixed seed.
> - Stacked rosters dominate non-stacked rosters in expected prize equity.
> - Top-recommended roster has higher expected prize equity than a randomly constructed roster.
>
> Performance: 50K field pods × 100 candidates × 6 slots in <60 seconds. Vectorize aggressively.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Summarize.

### 5.4 Best Ball advance equity

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `drafts/projections.py` first. Then create `src/sportstradamus/drafts/advance_equity.py`.
>
> Public API:
>
> ```python
> @dataclass
> class TournamentStructure:
>     n_pods: int
>     pod_size: int
>     regular_season_weeks: int
>     playoff_weeks: list[int]
>     advance_per_pod_round_1: int
>     advance_per_pod_round_2: int
>     advance_per_pod_round_3: int
>     payout_curve: dict[int, Decimal]
>     regular_season_prizes: list[Decimal]
>
> def expected_payout(
>     roster: list[Player],
>     structure: TournamentStructure,
>     n_simulations: int = 10_000,
>     seed: int = 42,
> ) -> AdvanceEquityResult:
>     """Returns expected_payout, p_advance_round_1/2/3, regular_season_ev, playoff_ev."""
> ```
>
> Algorithm:
>
> 1. Sample `n_simulations` independent full seasons of weekly fantasy points for every NFL player from `drafts.projections`. Cache.
> 2. For each simulation: build a stochastic 12-team pod, the user's roster as one team, 11 ADP-sampled opponents.
> 3. Apply Underdog's optimal-lineup rule weekly.
> 4. Sum 14-week regular season scores; top 2 advance.
> 5. Week 15: top 1 of 12. Week 16: top 1 of 12. Week 17: place against all surviving teams.
> 6. Award prizes per `payout_curve`.
> 7. Return mean and breakdowns.
>
> Performance: 10K sims × 12 teams × 17 weeks × 18 slots in <60 seconds for one roster. Numba-JIT the inner score-the-lineup loop.
>
> Tests at `tests/golden/test_advance_equity.py`:
> - Deterministic with fixed seed.
> - League-average roster returns expected payout near 1× entry fee (10–13% rake means slightly under).
> - Elite roster returns substantially over entry fee.
> - Round-1 advancement for 50th-percentile roster ~16.7% (2 of 12).
>
> Constraint: pure simulation. No HTTP, no disk writes from this module. CLI wrapper next session.
>
> When done: ruff clean, golden tests pass. Benchmark single roster evaluation. Summarize.

### 5.5 Live-draft companion

**Prompt:**

> Read `CLAUDE.md` and the existing `drafts/advance_equity.py`, `drafts/adp.py`, `drafts/projections.py` first. Then create `src/sportstradamus/drafts/best_ball_optimizer.py`.
>
> Public API:
>
> ```python
> @dataclass
> class DraftState:
>     pod_size: int
>     pick_number: int
>     my_slot: int
>     my_roster: list[Player]
>     all_picks: list[tuple[int, Player]]
>     contest: str
>
> def recommend(
>     state: DraftState,
>     k: int = 5,
>     candidate_pool_size: int = 30,
>     n_simulations: int = 2_000,
> ) -> list[Recommendation]:
>     ...
> ```
>
> Algorithm:
>
> 1. Available players = anyone not in `all_picks`.
> 2. Take top `candidate_pool_size` by current ADP.
> 3. For each candidate: hypothetically add to my_roster, run `advance_equity.expected_payout`. Score = resulting expected payout.
> 4. Return top k with diagnostic metadata: stack synergy, projected weekly variance, late-season schedule strength.
> 5. Total wall time: <10 seconds for k=5, candidate_pool_size=30.
>
> Caching tricks for the inner loop:
> - 17-week sample matrix for non-candidate players is shared across candidate evaluations.
> - Opponent simulation is shared.
> - Only the candidate's contribution to weekly scores changes.
>
> CLI: `poetry run draft-recommend --state path/to/state.yaml`.
>
> Tests at `tests/golden/test_best_ball_optimizer.py`:
> - Deterministic with fixed seed.
> - Recommendations all in the available pool.
> - Adding a top recommendation increases expected payout more than a random available player.
> - Stack synergy: a candidate stacking with a drafted QB scores higher than equivalent-projection player on a different team.
>
> Constraints: no live HTTP. Caches go in module-level lru_cache.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Run synthetic 12-pod draft from slot 6, post top-5 recommendations at picks 1, 6, 12, and timing. Summarize.

### 5.6 Portfolio exposure tracker

**Prompt:**

> Read `CLAUDE.md` and the existing `tracking/` package first. Then create `src/sportstradamus/drafts/exposure.py`.
>
> Public API:
>
> ```python
> def compute_exposure(
>     contest: str,
>     entry_pool: list[BestBallEntry],
> ) -> ExposureReport:
>     """
>     For each player, stack pattern, and roster archetype:
>     percentage of user's entries containing them, vs. field exposure (from ADP).
>     """
> ```
>
> Surfaces:
> - Player concentration: any player >35% of entries → flag.
> - Stack concentration: any (QB, top-WR) combo >25% → flag.
> - Archetype distribution: % Hero-RB, % Zero-RB, % balanced.
> - Leverage: user_exposure / field_exposure ratio per player.
>
> CLI: `poetry run drafts-exposure --contest best_ball_mania_VII`.
>
> Tests:
> - Concentration flags fire correctly.
> - Leverage calculation matches hand math on a small fixture.
>
> When done: ruff clean, golden tests pass. Summarize.

---

## Phase 6 — Modeling Refinements (Ongoing)

**Goal:** squeeze additional CLV out of edge cases and underserved markets. None required for profitability; all upside.

**Estimated time:** opportunistic, 1–4 weekends per item

### 6.1 Bayesian hierarchical for low-sample players

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `training/pipeline.py:train_market` first. Then add `src/sportstradamus/training/bayes_hier.py` for Bayesian player-level shrinkage in low-sample situations.
>
> Scope: NBA points only this session. Other markets and sports follow.
>
> PyMC model (NegBin with player random effects partially pooled to position priors). Output: per-player posterior mean + std stored at `data/bayes_predictions/NBA_points/{date}.parquet`.
>
> CLI: `poetry run bayes-update --league NBA --market points`. Nightly cron.
>
> Modify `training/pipeline.py:train_market` to read `bayes_mean` and `bayes_std` as input features when available; fall back to NaN. LightGBM handles NaN natively.
>
> Tests:
> - Deterministic with fixed seed.
> - r_hat < 1.05 on test fixture.
> - Rookie with 5 games has posterior mean closer to position prior than veteran with 500 games.
>
> Constraints: PyMC and ArviZ go in `[tool.poetry.group.bayes]`. Sampling on test fixture <60 seconds.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Run on full NBA data; report sampling time, r_hat range, marginal CLV improvement on next 100 NBA points props. Summarize.

### 6.2 Monte Carlo NFL game simulator

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `training/pipeline.py` first. Then create `src/sportstradamus/training/game_sim_nfl.py` — Vegas-anchored NFL Monte Carlo game simulator.
>
> Public API: simulate_game returns `dict[player_market, np.ndarray of shape (n_sims,)]` for markets pass_yds, pass_tds, pass_completions, pass_attempts, rush_yds, rush_attempts, receiving_yds, receptions, targets.
>
> Algorithm: sample game-flow params (Vegas-anchored), translate to per-team drives, then to pass/rush attempts, sample target shares from Dirichlet posteriors, sample yards-per-target from per-player distributions, aggregate.
>
> Performance: 10K sims of one game in <500ms with Numba.
>
> Add `joint_prob_from_sim(sim_output, predicates)` helper for parlay legs.
>
> Modify `prediction/correlation.py:find_correlation` to optionally use this simulator instead of Gaussian copula when sim outputs are available for all legs.
>
> Tests:
> - Deterministic with fixed seed.
> - Total points in sims match Vegas total within 0.5 across 10K sims.
> - QB pass yards × WR1 receiving yards correlation > +0.5.
> - 10K sims in <500ms.
>
> Constraints: Numba goes in core. Calibrate drives-from-pace and pass-attempts regressions via `poetry run game-sim-calibrate-nfl`, persist to `data/game_sim/nfl_coefficients.json`.
>
> When done: ruff clean, golden tests pass, benchmark passes, regenerate CLI snapshots. Summarize.

### 6.3 News and weather feeds

**Prompt:**

> Read `CLAUDE.md` and `CONTRIBUTING.md` first. Then add `src/sportstradamus/feeds/` with `news.py` and `weather.py`.
>
> `news.py`:
> - Poll Underdog's news feed every 60s during 6am–1am ET.
> - Poll RotoWire's RSS as backup.
> - Persist to `data/news/{date}.jsonl` with ts, source, sport, league, player_id (resolved), player_name, headline, body, severity.
> - Emit alert events when severity ≥ 'high'.
>
> `weather.py`:
> - For each upcoming NFL game in next 7 days at outdoor stadium, fetch hourly forecast from OpenWeatherMap free tier.
> - Persist to `data/weather/{game_id}.json`.
> - Compute features: wind_mph, temp_f, precipitation_pct, is_dome.
> - Add `weather_features(game_id) -> dict` for `Stats.NFL.get_stats()`.
>
> CLI: `poetry run feeds-update`. Cron-friendly.
>
> Tests:
> - News parsing on RSS fixture.
> - Weather feature extraction for known dome game returns is_dome=True.
> - Weather feature for outdoor game with wind > 15mph flags windy=True.
>
> Constraints: OpenWeatherMap key in `keys.json`. News best-effort — log WARNING and try next source on failure; never crash polling loop.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Summarize.

### 6.4 Conformal prediction wrappers

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `training/pipeline.py` first. Then add `src/sportstradamus/training/conformal.py` — split conformal prediction wrappers around LightGBMLSS marginals.
>
> Public API:
>
> ```python
> @dataclass
> class ConformalCalibration:
>     market: str
>     league: str
>     calibration_quantiles: np.ndarray
>     fit_at: datetime
>
> def fit_conformal(model, X_calib, y_calib, alpha: float = 0.1) -> ConformalCalibration:
>     """Fit split conformal on a held-out calibration set."""
>
> def conformal_prob_over(
>     model, x: np.ndarray, line: float, calib: ConformalCalibration
> ) -> tuple[float, float]:
>     """Returns (point_estimate, half_width) of P(stat > line) with (1-alpha) coverage."""
> ```
>
> In `strategies/kelly.py`, add option to size based on lower bound of conformal interval rather than point estimate. Conservative when model uncertainty is high.
>
> Add `--fit-conformal` flag to `meditate` that runs calibration on holdout and persists to `data/conformal/{LEAGUE}_{market}.pkl`.
>
> Tests:
> - Coverage on synthetic example: with alpha=0.1, observed coverage on holdout in [85%, 95%].
> - Half-width is monotonic in n_calibration.
>
> Constraint: do not change LightGBMLSS training. Wrapper only.
>
> When done: ruff clean, golden tests pass, regenerate CLI snapshots. Fit conformal on NBA points, report empirical coverage and median half-width. Summarize.

### 6.5 Push-aware EV refinement

I added basic push handling in Phase 1.2, but a more thorough version models the integer-valued markets (NFL TDs, NHL goals, NBA 3PM) with discrete distributions explicitly rather than treating their continuous-distribution CDF integration as approximate.

**Prompt:**

> Read `CLAUDE.md`, `CONTRIBUTING.md`, and `helpers/distributions.py` first. Then audit and refine push handling in the EV calculation.
>
> Audit first: for each market in `data/stat_dist.json`, classify its support: continuous (yards, %), integer (TDs, goals), or hybrid. Document in `docs/MARKET_SUPPORT_AUDIT.md`.
>
> Then refine `helpers/distributions.get_odds`:
> - For integer-support markets, compute `P(stat == line)` exactly using the discrete PMF rather than approximating via CDF differencing.
> - For NegBin distributions, use `nbinom.pmf(line, ...)`.
> - For continuous distributions on integer-valued markets (e.g., TDs modeled as Gamma), apply a continuity correction: P(push) = P(line - 0.5 < stat < line + 0.5).
>
> Verify the refinement improves calibration on past parlays involving integer-support legs (rerun the audit script from Phase 1.2).
>
> Tests:
> - Push probability for an NFL anytime-TD market with line=1.0 and modeled mean=1.2 matches `nbinom.pmf(1, ...)` to within rounding.
> - Continuity correction for Gamma-modeled TDs gives a non-zero push probability.
>
> When done: ruff clean, golden tests pass. Summarize calibration improvement.

---

## Pacing Recommendation

> **Note (2026-05-08):** original pacing assumed a placed-bet logger and
> per-bet CLV. With those out of scope, the table below is the
> scope-reduced version.

| Month | Focus |
|---|---|
| 1 | Phase 1.2 follow-ups + Phase 1.6 deprecated triage. CLV monitoring on existing `clv.summarize` output. |
| 2 | Phase 3.1 (Kelly) + Phase 3.3 (Underdog-native `pickem-build`). |
| 3 | Phase 4.1 alerts (in-scope subset) + Phase 4.2 Today's Recommendations dashboard tab. Review CLV on accumulated segments. |
| 4–5 | Phase 3.4 (Champions) + 3.5 (Streaks/Ladders/Rivals) as appetite allows. |
| 6–9 | Phase 5 (Best Ball + Battle Royale rebuild), aligned with NFL season. |
| 10–12 | Phase 6 refinements (conformal first, Bayesian after) as edge erodes. |

Phase 1 audit work is undramatic but it's where most of the actual edge correction happens. The existing modeling is already solid; the gains come from confirming the correlation methodology is sound, the parlay search is well-calibrated, push handling is correct, and contest-variant payouts are right. Skipping it to jump to Phase 3 means building the strategy module on a foundation you haven't verified.

A realistic minimum-viable path was: **Phase 1.1 (correlate) + 1.2 (parlay) + 2 (CLV) + 3.1 (Kelly) + 3.3 (Underdog-native)**. With the placed-bet logger dropped, only **3.1 + 3.3 + the Phase 1.2 follow-ups** remain on that path. Roughly 3–4 weekends of focused work to reach a ranked-and-sized recommendation pipeline; everything else is multipliers on that core.

---

## Updated Critical Path (post-2026-05-08 audit, scope-reduced)

The modeling and CLV foundations are in place (1.1 + 1.2 + 2.x + 3.2).
With placed-bet tracking explicitly out of scope, the remaining work is
narrower: close out Phase 1 follow-ups, ship Kelly + Underdog-native
recommendation, and add useful alerts. Recommended order:

1. **Phase 1.2 follow-up — open audit findings (1 weekend).** Fix the
   `find_correlation` pairwise-EV unbounded score, replace inline magic
   numbers with named constants, swap substring same-player guarding for
   `player_id` joins, and reconcile the `Boost`-column overwrite at
   `correlation.py:498` so the displayed multiplier matches the EV that
   ranked the parlay. Re-run `audit_parlay_calibration.py` on production
   archive data and commit a real plot.
2. **Phase 1.6 — deprecated triage (½ weekend).** `correlation.py` and
   `opt_parlay.py` in `src/deprecated/` can be deleted; `opt_kelley_bet.py`
   should be marked REVIVE for the next session. Cheap and unblocks
   confusion for future agents.
3. **Phase 3.1 — Kelly module (1 weekend).** Genuinely missing. The
   parlay search produces ranked candidates that nothing currently sizes.
   Output is a recommendation, not a placed-bet trigger — users decide
   whether to actually wager.
4. **Phase 3.3 — Underdog-native strategy CLI (1 weekend).** Single
   `pickem-build` CLI that produces today's ranked, Kelly-sized
   recommendations as YAML / Sheets export.
5. **Phase 4.1 alerts (in-scope subset only, 1 weekend).** High-edge
   appearance, scraper failure, model staleness, and CLV drift on
   system history. Drop the rules that depended on placed-bet state
   (open-entry exposure, bankroll drawdown).
6. **Phase 4.2 dashboard — Today's Recommendations tab only.** Reads
   `data/recommendations/{today}.yaml`. Skip the Open Entries / Settled
   History / Bankroll tabs.

That sequence finishes the package's actual mandate — surface ranked,
sized, EV-positive Underdog recommendations with measurable CLV — and
then stops. Personal bet tracking is a downstream concern users own
themselves.

---

## Suggestions for Further Improvement (beyond the roadmap)

The original v2 roadmap covered the structural gaps. The audit surfaced a
handful of items that aren't in any phase but matter for long-term health.

### Modeling hygiene

- **Closing-line bias check.** Once Phase 1.3 freeze is in place, build a
  diagnostic that compares the *open* line at scrape time vs the close.
  Markets where open consistently moves toward your model's prediction
  are markets where you're sharper than the market opens; markets where
  the close moves *away* from your model are model bugs hiding behind
  momentary mis-pricing.
- **Per-market shape ratio dashboard.** `training_report.txt` exposes
  `shape_ratio` per market but only as a snapshot. A historical CSV per
  retrain would surface drift (a market trending from 1.0 to 1.3 over six
  months means dispersion is creeping). Cheap to add to `training/report.py`.
- **Parquet-everything migration.** Migration scripts under `scripts/`
  exist for archive, training data, gamelogs, correlations, and pickles.
  Confirm none of the production paths still read CSV or pickle. The
  archive itself is DuckDB now, but `data/{LEAGUE}_corr.csv` is still a
  CSV (per CORRELATE_AUDIT.md) — convert to parquet to save read time
  and to carry typed metadata columns.
- **Continuity correction audit.** Phase 6.5 asks for a per-market
  support audit (continuous vs integer vs hybrid). This is a 2-hour task
  that would catch mis-applied push probabilities for markets like
  anytime-TD, NHL goals, NBA 3PM. Worth promoting to Phase 1.

### Engineering hygiene

- **Magic-number purge in `correlation.py`.** Roadmap mentions this in
  passing for Phase 1.2 follow-up; the audit's lit list (`K=1000`, EV
  cutoffs, boost bands, `[0, 0, 3.5, 6.5, 6, 10, 25]` payout array) is
  long enough that it deserves its own commit per STYLE_GUIDE.md §8.
- **`prediction/parlay.py` split.** CONTRIBUTING.md advertises this
  module; in practice both `find_correlation` and `beam_search_parlays`
  live in `correlation.py`. Either split per the contract or update
  CONTRIBUTING.md. Don't leave the inconsistency.
- **`src/deprecated/` README cleanup.** The TODO list at the bottom of
  `src/deprecated/README.md` references items that have replacements in
  the live tree (`correlation.py`, `opt_parlay.py`). Phase 1.6 was meant
  to fix this; until it lands, the README actively misleads new agents
  reading the repo.
- **`clv.py` location is fine.** Originally this would have moved into
  a `tracking/` namespace alongside placed-bet logging; with the bet
  logger out of scope, leaving `clv.py` at the package root next to
  `nightly.py` is the right shape.
- **`book_weights.json` provenance.** It's referenced in the prediction
  pipeline and in Phase 2.3 of this roadmap but the source of the
  weights and how often they're refit is undocumented. Add a provenance
  note + a `scripts/refit_book_weights.py` so the next refit is one
  command, not a memory exercise.

### New diagnostics worth building

- **CLV-by-day-of-week × time-to-lock heatmap.** `clv.py` already has
  the data; one plot would tell you whether your edge is lock-window
  specific (it usually is — early-week NFL lines have more error than
  Sunday-morning lines).
- **Parlay-shape ablation.** `audit_parlay_calibration.py` produces
  decile calibration. Add a sister script that holds out individual
  shape constraints (min_correlation, max-per-game, beam_width) and
  reports calibration delta per ablation. Tells you which knob actually
  matters before you tune it.
- **Model-vs-market disagreement log.** When the Gaussian copula joint
  estimate and the per-leg sharp devig disagree by more than a
  configurable threshold, log to `data/disagreements/{date}.jsonl` for
  retrospective analysis. Currently a model-bug detector; eventually a
  potential-edge surfacer.

### Operational

- **Bankroll input is a CLI flag, not state.** With the placed-bet
  logger out of scope, "bankroll" is just a parameter the user passes
  to `kelly` and `pickem-build`. No `data/bankroll.json`, no DB row —
  one less piece of state to drift.
- **`pyproject.toml` script registration.** Currently 5 scripts wired
  (`prophecize`, `confer`, `meditate`, `dashboard`, `reflect`).
  Future-proof: when adding `kelly`, `pickem-build`, `alert-watch`,
  register them in one batch with consistent naming (verb-noun,
  dash-separated) rather than a slow drift toward inconsistency.
- **`creds/keys.json` schema doc.** The repo references `keys.json`
  several places (Odds API, Telegram, Discord, OpenWeatherMap) but
  there's no canonical schema. A single `docs/CREDS_SCHEMA.md` with
  example keys and which module reads each would avoid the
  next-onboarding bug.

### Bigger swings (optional, multi-week)

- **Replace beam search with beam-search-by-shape.** Currently the search
  ranks all parlays jointly. Splitting by `(contest_variant, entry_size)`
  upstream and running separate beams lets you tune cutoffs per shape
  without cross-contamination. Cheap refactor, real win.
- **Conformal *before* Bayesian (swap 6.4 and 6.1).** Conformal is a
  wrapper around the existing LightGBMLSS pickles; PyMC introduces a new
  dependency, a new sampler, and new infrastructure. Conformal gives you
  a calibrated uncertainty knob immediately; Bayesian hierarchical
  delivers value mostly for low-sample players. Conformal is the
  higher-EV-per-engineer-week swing.
- **Streamlit → simple FastAPI + Vue dashboard.** Streamlit is fine until
  it isn't; the moment you need real-time push (Phase 4.1 alerts feeding
  the dashboard), Streamlit's rerun model fights you. Consider a
  one-weekend FastAPI + WebSocket prototype before investing more in
  Streamlit pages.
