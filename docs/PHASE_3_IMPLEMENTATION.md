# Phase 3 Implementation Plan — Underdog-Specific Decision Engine

**Status target:** turn the existing parlay candidate stream into ranked,
Kelly-sized, contest-variant-aware Underdog recommendations covering
Power, Flex, and Rivals.

**Source documents:**
- `docs/sportstradamus_roadmap_v2.md` §Phase 3 (3.1, 3.3, 3.5 Rivals
  only) and §Updated Critical Path (post-2026-05-08).
- `docs/underdog_edge_suite.md` §4 (Decision Engines) and §5 (Execution
  & Risk).
- Audit notes in §1.3 (closing-line freeze — dropped, workaround
  sufficient) and §2.4 (CLV reporting — monitor the implicit-close
  assumption).

**Estimated time:** 4 weekends, sequenced so each step ships
independently and respects CLAUDE.md's one-module-per-session rule.

---

## 0. Scope and Non-Goals

**In scope:**

1. CLV-side bring-up needed before Kelly can lean on calibration data
   (Step 1).
2. Magic-number purge in `prediction/correlation.py` (Step 2).
3. `prediction/parlay.py` split per CONTRIBUTING.md contract (Step 3).
4. Training pipeline: dedicated Kelly-shrinkage diagnostic (Step 4).
5. Kelly sizing module (Step 5, roadmap §3.1).
6. Underdog-native `pickem-build` orchestrator covering Power, Flex,
   and Rivals (Step 6, roadmap §3.3 + §3.5 Rivals folded in).
7. Documentation refresh (Step 7).

**Out of scope:**

- **Pick'em Champions (§3.4) — removed.** Pari-mutuel optimization is a
  different problem shape; revisit only after Phase 3 has measurable
  CLV.
- **Streaks and Ladders (§3.5) — deferred.** Each warrants a dedicated
  design pass.
- Placed-bet logger (`tracking/`) and per-bet CLV. Roadmap §2.2/2.3
  dropped.
- Phase 4.1 alerts. Best Ball / Battle Royale (Phase 5). Bayesian /
  conformal modeling (Phase 6).

**Roadmap §3.2 (contest payouts) is already DONE.** This plan consumes
`data/underdog_payouts.json` and the `contest_variant` arg on
`beam_search_parlays`; it does not redo that work.

---

## 1. CLV Bring-Up (Step 1, ≈½ weekend)

Roadmap §1.3 dropped the explicit `freeze_close` pipeline. Phase 3 has
two new consumers that need a narrower bring-up — not the full freeze
system.

### 1.a Close-timestamp sanity check

Per §2.4 monitor note: warn when the archive snapshot used as "close"
is more than `CLOSE_LOOKBACK_WARN_MINUTES = 90` from `commence_time`.
One log per (league, market, date) segment via
`helpers.logging.get_logger`.

**Files touched:** `src/sportstradamus/clv.py`.
**Tests:** `tests/golden/test_clv_close_sanity.py`.

### 1.b Per-segment CLV getter for Kelly

Add `clv.get_segment_calibration(league: str, market: str) -> float`.
Reads `data/clv_segments.parquet` (newly persisted at end of `reflect`
as a side effect of `clv.summarize`). Returns `frac_beat_close`
remapped to a `[0.0, 1.0]` shrinkage weight; documented fallback of
`1.0` when the segment has fewer than `CLV_SEGMENT_MIN_N=20` legs.

**Files touched:** `src/sportstradamus/clv.py`,
`src/sportstradamus/nightly.py` (just persists the segments parquet at
the end of `reflect`; no behavior change).

### 1.c Read-only `Archive.get_closing_line` shim

Wraps `Archive.get_line` + `Archive.get_ev` for the latest pre-kickoff
sample. Returns a small dataclass `(line, devig_over, sample_ts,
book_set)`. Used by 1.a and consumed by Step 6's recommendation YAML.

**Files touched:** `src/sportstradamus/helpers/archive.py`.

---

## 2. Magic-Number Purge in `prediction/correlation.py` (Step 2, ½ weekend)

Roadmap §1.2 follow-up. Inline literals like `0.95`, `0.05`, `0.9`,
`1.78`, `0.54`, the historical Boost-overwrite payout array, and
several EV/correlation cutoffs at `correlation.py:624` and `:411`
ride alongside the already-named `_BEAM_WIDTH` and
`_KELLY_BANKROLL_FRACTION`. Promote them to module-level constants per
STYLE_GUIDE §8 (named, ALL_CAPS, single-line "why" comment).

**Constraint:** behavior must be unchanged. This is a rename pass, not
a tune.

**Files touched:** `src/sportstradamus/prediction/correlation.py`.
**Tests:** existing `tests/golden/test_correlation*.py` and parlay
calibration audit must produce byte-identical output (re-run
`scripts/audit_parlay_calibration.py` and diff the plot).

This step lands **before** Step 3 to avoid carrying unexplained
literals across the file split.

---

## 3. `prediction/parlay.py` Split (Step 3, ½ weekend)

CONTRIBUTING.md §Package Map advertises that `find_correlation` lives
in `prediction/correlation.py` and `beam_search_parlays` lives in
`prediction/parlay.py`, but today both live in `correlation.py`. Split
to match the contract.

**Move to `prediction/parlay.py`:** `beam_search_parlays` (line 732+),
its helpers `_payout_curve_for`, `_expected_payout_with_pushes`,
`_nearest_psd`, plus the constants those helpers consume.

**Stays in `prediction/correlation.py`:** `find_correlation`,
`_team_slice`, `_build_game_corr_map`, plus the
`_legacy_underdog_overwrite_payouts` / `_legacy_underdog_search_payouts`
shims (find_correlation calls `beam_search_parlays`, so it imports
from `parlay.py`).

**`prediction/__init__.py`:** re-export `beam_search_parlays` from the
new module so external imports keep working.

**Files touched:** `src/sportstradamus/prediction/correlation.py`,
`src/sportstradamus/prediction/parlay.py` (new),
`src/sportstradamus/prediction/__init__.py`,
`src/sportstradamus/prediction/scoring.py` (one import line).

**Tests:** existing tests must pass unchanged. Add a thin
`tests/golden/test_prediction_imports.py` asserting
`from sportstradamus.prediction.parlay import beam_search_parlays` and
the package-level re-export both work.

---

## 4. Training Pipeline: Migrate to Raw Data-Science Metrics + Kelly Shrinkage (Step 4, 1 weekend)

### 4.a Why `model_calib` (and several siblings) are the wrong fields

Verified at `src/sportstradamus/training/pipeline.py:688`:

```python
val_calibrated = expit(val_logits / T_opt)
model_calib = 1 - np.mean((val_calibrated - y_class_val) ** 2)
```

That is `1 − Brier(temperature_calibrated_probs, actual)` on the
validation set — a "scaled-for-context" score in roughly `[0.75, 1.0]`
where even a useless model that always predicts 0.5 lands at ~0.75.
The transformation existed to give human readers an at-a-glance
quality bar; it actively hurts dashboard analysis and any downstream
consumer (Kelly, drift detection, model selection).

The same pattern shows up across the report: per-row classification
metrics in `model["stats"]` (`Accuracy`, `Over Prec`, `Under Prec`,
`Over%`, `Sharpness`, `NLL`) are a mix of raw values and folded-up
ratios. Now that the dashboard exists to provide context (Step 4.d),
migrate to the raw metrics a data scientist would expect.

### 4.b Metric migration table

Compute raw metrics in `pipeline.py` and emit them via `report.py`'s
parquet (`model_stats.parquet`). The migration is a clean break — the
parquet is regenerated from scratch by `poetry run meditate`, so old
column names disappear on the first retrain after PR `3-train` lands.
No legacy aliases; consumers (the dashboard page 7, this plan)
update in lockstep.

| Old field | Replacement(s) | Type | Direction |
|---|---|---|---|
| `model_calib` (`1 − Brier`) | `brier_score`, `log_loss` | raw scoring rules | lower is better |
| (none) | `brier_skill_score` (vs. book baseline) | skill | higher is better; 0 = matches book |
| (none) | `roc_auc` | discrimination | higher is better; 0.5 = random |
| (none) | `expected_calibration_error` (10-bin) | calibration | lower is better |
| `Accuracy` | `accuracy` (kept; raw) | raw | higher is better |
| `Over Prec` / `Under Prec` | `precision_over` / `precision_under` (renamed; raw) | raw | higher is better |
| `Over%` | `predicted_over_rate` (renamed; raw) | rate | informational; compare to `empirical_over_rate` |
| (none) | `empirical_over_rate` | rate | informational; the actual hit rate of legs |
| `Sharpness` | `prediction_std` (renamed; raw) | raw | informational |
| `NLL` | `nll` (kept; raw) | raw | lower is better |
| (none, derived) | `kelly_shrinkage` = `clip(brier_skill_score, 0, 1)` | derived | higher = trust the model more |

`kelly_shrinkage` is now derived from `brier_skill_score` rather than
duplicated math, and the dashboard surfaces both.

### 4.c Pinned book-baseline row

The dashboard requires a baseline-of-comparison row pinned to the top
of the diagnostic table: the same metrics computed if the model
**were** the bookmaker (i.e., predicting `book_implied_prob` for every
validation example). Concretely, in `pipeline.py`'s validation block:

```python
def compute_metrics(probs, y):
    return {
        "brier_score":               brier(probs, y),
        "log_loss":                  log_loss(probs, y),
        "roc_auc":                   roc_auc(probs, y),
        "expected_calibration_error": ece(probs, y, n_bins=10),
        "accuracy":                  accuracy(probs > 0.5, y),
        "precision_over":            precision(probs > 0.5, y, pos=1),
        "precision_under":           precision(probs < 0.5, y, pos=0),
        "predicted_over_rate":       float(np.mean(probs > 0.5)),
        "empirical_over_rate":       float(np.mean(y)),
        "prediction_std":            float(np.std(probs)),
        "nll":                       nll(probs, y),
    }

model_metrics = compute_metrics(val_calibrated, y_class_val)
book_metrics  = compute_metrics(val_book_proba, y_class_val)
brier_skill_score = 1 - (model_metrics["brier_score"]
                         / max(book_metrics["brier_score"], 1e-9))
```

Both metric dicts are written to `model_stats.parquet`. Add a
`row_kind` column with values `"book_baseline"` and `"model"` (the
existing `metric_row` column is repurposed or kept beside it; favor
keeping `metric_row` for `raw / corrected / calibrated` and
introducing a new `row_kind` sibling for `book_baseline / model`).

If `val_book_proba` is unavailable for a (league, market), skip the
book row for that segment with a logged WARNING. Skill score → `nan`.

### 4.d Dashboard updates — `pages/7_📊_Stats_Model_Training.py`

1. **Higher/lower-is-better column annotations.** Streamlit's
   `st.dataframe` accepts a `column_config` dict that supports
   custom help-text per column. Add a help string to every metric
   column reading `"↑ higher is better"` or `"↓ lower is better"` or
   `"informational"`. Define the direction map next to the rendered
   `cols` list so the source of truth is one place.
2. **Pin the book-baseline row.** Filter the dataframe to the active
   `(league, market)` selection plus the `row_kind == "book_baseline"`
   row for the same scope. Render the baseline row as a styled
   pinned row above the model rows (Streamlit dataframes do not
   natively pin; either render the baseline as a separate one-row
   `st.dataframe` immediately above the main table with a caption
   "Book-only baseline (what taking the book's odds gets you):", or
   `pd.concat` the baseline on top and use a row-styler to highlight
   it). Recommend the separate-table approach; it survives column
   reorders and sort interactions.
3. **Tab restructure.** Split the existing "Per-market accuracy" and
   "Diagnostics" tabs into:
   - **Scoring rules:** `brier_score`, `log_loss`, `nll`,
     `expected_calibration_error`.
   - **Discrimination:** `roc_auc`, `accuracy`, `precision_over`,
     `precision_under`, `prediction_std`.
   - **Rates:** `predicted_over_rate`, `empirical_over_rate`,
     `frac_ev_gt_line`, `over_pct_ev_gt`, `over_pct_ev_lt`.
   - **Kelly & blending:** `brier_skill_score`, `kelly_shrinkage`,
     `model_weight`.
   - **Dispersion:** `shape_ratio`, `dispersion_cal`,
     `model_shape`, `empirical_shape`.
   - **EV & lines:** `model_ev`, `mean_line`, `result_mean`,
     `mean_ev_diff`, `median_ev_diff`, `cf_over_pct`.
   - **Hyperparameters:** unchanged.
4. **Captions.** Each tab gets a one-line caption explaining what the
   pinned baseline represents and how to read the column directions.

### 4.e `training_report.txt` (human-readable companion)

Update the DIAG block to print raw metrics and the baseline:

```
================ NFL player_pass_yds ================
 Distribution: Gamma | Historical Zero Rate: 0.0124
 BOOK BASELINE  brier=0.218 logloss=0.682 auc=0.557 ece=0.041
 MODEL          brier=0.197 logloss=0.628 auc=0.628 ece=0.018
 SKILL          brier_skill_score=+0.096  kelly_shrinkage=0.10
 ...
```

The baseline row makes the human-readable report self-contained for
non-dashboard consumers (e.g., reviewing on a remote box without
Streamlit running).

### 4.f Thin getter for Kelly

```python
# in training/report.py
def get_market_calibration(league: str, market: str) -> dict[str, float]:
    """Return {'kelly_shrinkage', 'brier_skill_score', 'model_weight'} for one (league, market).

    Reads model_stats.parquet. Returns NaNs when the row is missing.
    """
```

Kelly imports this getter — never reaches into the parquet directly.

### 4.g Migration mechanics

`model_stats.parquet` is fully regenerated by `poetry run meditate`
on the next training run, so the migration completes the moment the
user retrains. There is no schema bridge to maintain. `nightly.py:reflect`
and `clv.summarize` don't read any of the renamed fields, so their
behavior is unchanged.

**Files touched:** `src/sportstradamus/training/pipeline.py`,
`src/sportstradamus/training/report.py`,
`src/sportstradamus/dashboard_data.py` (only if it exposes a column
allowlist; today it just `read_parquet`-s),
`src/sportstradamus/pages/7_📊_Stats_Model_Training.py`.
**Tests:** `tests/golden/test_training_metrics.py`:
- Each metric in the table matches a hand-worked example on a
  3-sample fixture.
- Useless model (Brier matching book) → `brier_skill_score = 0`,
  `kelly_shrinkage = 0`.
- Missing book column → both = `nan` and WARNING logged.
- Book-baseline row written to parquet with `row_kind="book_baseline"`.
- Getter returns NaNs when (league, market) absent.
- Streamlit-page smoke test: page renders without exception against a
  parquet containing both row kinds.

---

## 5. Kelly Sizing Module (Step 5, 1 weekend, roadmap §3.1)

Revive `src/deprecated/opt_kelley_bet.py` into
`src/sportstradamus/strategies/kelly.py`.

### 5.a Module layout

```
src/sportstradamus/strategies/
├── __init__.py          # re-exports kelly
├── kelly.py
└── README.md
```

### 5.b Public API

```python
def fractional_kelly_stake(
    bankroll: Decimal,
    win_prob: float,
    payout_multiplier: Decimal,
    fraction: float = DEFAULT_KELLY_FRACTION,
    model_shrinkage: float = 1.0,
    max_fraction_of_bankroll: float = MAX_FRACTION_OF_BANKROLL,
) -> Decimal: ...

def joint_kelly_portfolio(
    bankroll: Decimal,
    candidates: list[KellyCandidate],
    fraction: float = DEFAULT_KELLY_FRACTION,
) -> dict[str, Decimal]: ...
```

Note: kwarg renamed from `model_calibration` to `model_shrinkage`
because Step 4 demonstrated those names mean different things. Keep
the value semantics: `1.0` = no shrink, `0.0` = degenerate to
no-information.

### 5.c Constants (STYLE_GUIDE §8)

| Name | Value | Reason |
|---|---|---|
| `DEFAULT_KELLY_FRACTION` | `0.25` | Quarter-Kelly hedge against estimate variance. |
| `MAX_FRACTION_OF_BANKROLL` | `0.005` | 0.5%-per-entry hard cap from edge-suite §5.1. |
| `SHRINKAGE_FLOOR` | `0.0` | Below this, treat as no-information (no Kelly stake). |

### 5.d Shrinkage resolution and blending

```
effective_p = 0.5 + (win_prob - 0.5) * shrinkage
```

The training-time validation set is large; the CLV-segment value is
not, especially early in a season. So instead of a hard "first
non-NaN wins" chain, blend the two as the per-segment sample size
ramps up.

Let `n` be the number of CLV-segment legs for the `(league, market)`
returned by `clv.get_segment_calibration` (the same count surfaced in
`clv_segments.parquet`). Define a linear ramp:

```python
LIVE_BLEND_FLOOR = 25      # below this, ignore live BSS entirely
LIVE_BLEND_FULL  = 100     # at or above this, ignore training BSS

w_live = clamp((n - LIVE_BLEND_FLOOR)
               / (LIVE_BLEND_FULL - LIVE_BLEND_FLOOR), 0.0, 1.0)
shrinkage = w_live * live_bss + (1 - w_live) * training_bss
```

Resolution rules, in order:

1. Explicit `model_shrinkage` kwarg — overrides the blend.
2. Both `live_bss` (from `clv.get_segment_calibration`) and
   `training_bss` (from `training.report.get_market_calibration`)
   present → blended per the ramp above.
3. Only `training_bss` present (n < `LIVE_BLEND_FLOOR`, or CLV segment
   missing) → use `training_bss` directly.
4. Only `live_bss` present (training BSS missing — unusual) → use
   `live_bss` directly.
5. Neither present → fallback `1.0`. Logged at DEBUG.

The ramp is intentional: by ~25 segment legs, the live signal is no
longer pure noise but should still be smoothed by training; by 100+
legs, current-season conditions have moved enough that training-time
BSS is stale and the live signal should dominate. Constants are
roadmap-aligned with `CLV_SEGMENT_MIN_N=20` (we use 25 here so the
ramp starts above the bare reporting threshold).

Document the blending rule and constants in the `kelly.py` module
docstring with the same one-line "why" comment per STYLE_GUIDE §8.

**Tests:** `tests/golden/test_kelly.py` (extends Step 5.g):
- `n=10` → output equals `training_bss`.
- `n=100` → output equals `live_bss`.
- `n=62` → output equals `0.5 * training_bss + 0.5 * live_bss` ±
  float tolerance.
- Missing `training_bss`, `n=10` → fallback `1.0`, DEBUG log.

### 5.e Portfolio variant

cvxpy SCS solver, lazy import. Add cvxpy under
`[tool.poetry.group.strategy]` so core deps stay untouched.

### 5.f CLI

`poetry run kelly --bankroll 500 --from data/recommendations/{date}.yaml`

Reads the YAML produced in Step 6, prints a tabulate table.

### 5.g Tests — `tests/golden/test_kelly.py`

- −EV bet → `Decimal("0")`.
- +EV closed-form expected stake matches analytical fractional Kelly.
- Hard cap kicks in when uncapped Kelly exceeds 0.5%.
- Two-bet portfolio sums to ≤ `fraction × bankroll`.
- Resolution chain: explicit > CLV > training > fallback, with each
  source mocked.

### 5.h Deprecated cleanup

Move `src/deprecated/opt_kelley_bet.py` → `src/deprecated/.archived/`.
Update `src/deprecated/README.md` TODO list.

---

## 6. Underdog-Native Strategy — `pickem-build` (Step 6, 1 weekend, roadmap §3.3 + Rivals)

`src/sportstradamus/strategies/underdog_pickem.py`. Pure orchestrator —
no math.

### 6.a Public API

```python
@dataclass
class PickemConfig:
    min_model_edge: float = 0.020
    min_sharp_edge: float = 0.015
    disagreement_threshold: float = 0.04
    min_correlation: float = 0.10
    min_ev: float = 0.05
    entry_sizes: tuple[int, ...] = (3, 5)
    contest_variants: tuple[str, ...] = ('power', 'flex', 'rivals')
    top_k: int = 20
    max_overlap: int = 2
    kelly_fraction: float = 0.25
    max_stake_pct_bankroll: float = 0.005

def construct_entries(
    date: datetime.date,
    bankroll: Decimal,
    config: PickemConfig | None = None,
) -> list[RecommendedEntry]: ...
```

### 6.b Pipeline

1. **Load offers** — reuse the loader `prophecize` already calls.
2. **Filter legs** — Underdog-only, model & sharp coverage,
   `|model − sharp| ≤ disagreement_threshold`, edge thresholds.
3. **Search** — `beam_search_parlays(..., contest_variant=v)` once per
   `(entry_size, variant)` cross. Power and Flex honor user
   `entry_sizes`; **Rivals is restricted to 2- and 3-leg sizes** and
   requires both sides of the matchup covered for the same market —
   drop with logged WARNING when only one side is covered.
4. **Size** — `fractional_kelly_stake` per candidate, resolution chain
   per Step 5.d.
5. **Rank** — sort by EV-per-dollar, dedupe by leg overlap, cut to
   `top_k`.
6. **Emit:** `data/recommendations/{date}.yaml`. Sheets export was
   deprecated on `devel` (`prediction/sheets.py` lives in
   `src/deprecated/`); the dashboard reads the YAML directly. No
   Sheets tab.

### 6.c YAML schema

```yaml
generated_at: 2026-09-12T15:30:00-04:00
date: 2026-09-12
bankroll: 500
config: {shrinkage_source: clv_segment, ...}
entries:
  - id: <hash of legs>
    contest_variant: rivals       # power | flex | rivals
    entry_size: 2
    legs: [...]
    joint_prob: 0.276
    payout_multiplier: 3.0
    ev: 0.276 * 3 - 1
    recommended_stake: 1.25
```

### 6.d CLI

`poetry run pickem-build --date today --bankroll 500`. Bankroll is a
CLI flag only (no `data/bankroll.json` per roadmap §Operational).

### 6.e Tests — `tests/golden/test_underdog_pickem.py`

- Filtering excludes legs failing each threshold.
- Disagreement threshold catches a divergence.
- Power, Flex, Rivals each appear when enabled.
- Rivals matchup with one-sided coverage → dropped + WARNING.
- Rivals respects 2/3-leg-only constraint regardless of `entry_sizes`.
- E2E on the existing WNBA fixture produces non-empty YAML.
- YAML round-trips via `yaml.safe_load`.

---

## 7. Documentation Refresh (Step 7, ≈¼ weekend)

Lands in the same PR as Step 6 (or a sibling doc-only PR if the diff is
big). Updates:

- **`CLAUDE.md`:** add `kelly` and `pickem-build` to the CLI entry list.
- **`CONTRIBUTING.md`:** §Package Map row for `prediction/parlay.py`
  changes from "(advertised, not yet split)" to actual contents; add
  `strategies/` package row; add `kelly_shrinkage` to the training
  diagnostic glossary.
- **`docs/sportstradamus_roadmap_v2.md`:** flip §3.1, §3.3, and §3.5
  Rivals to ✅ DONE with a cross-link to this plan; mark §3.4 and the
  rest of §3.5 as removed/deferred.
- **`README.md` (top-level):** if it lists CLI scripts, add
  `kelly` and `pickem-build`.
- **Training-report glossary in CLAUDE.md:** rewrite the §Training
  Report Diagnostics section to reflect the migrated raw-metric
  schema. Cover all new fields, the `row_kind` column, the book
  baseline, and the higher/lower-is-better convention.

Documentation completeness is checked into the done criteria (§9 below).

---

## 8. Module-by-Module Execution Plan (one module per session)

CLAUDE.md mandates one module per session, with commit and fresh start
between modules. This plan groups the work above into single-module
sessions and orders them so each session lands on a clean, tested
foundation.

| # | Session focus (one module / file) | Step refs | Depends on | Notes |
|---|---|---|---|---|
| 1 | `helpers/archive.py` — add `get_closing_line` shim | 1.c | – | Smallest possible starter; unblocks others. |
| 2 | `clv.py` — close-timestamp sanity warn + segment getter | 1.a, 1.b | session 1 | Also persists `clv_segments.parquet` from `reflect`'s call site, but no logic changes in `nightly.py`. |
| 3 | `nightly.py` — call segment-persist hook | 1.b | session 2 | One-line edit; tiny session, but kept separate to respect the rule and allow a clean revert if it perturbs `reflect`. |
| 4 | `prediction/correlation.py` — magic-number purge only | 2 | session 3 | No file moves yet. Behavior must be byte-identical (re-run audit script). |
| 5 | `prediction/parlay.py` — new module + `beam_search_parlays` move | 3 | session 4 | This session edits two files (source + new), but both are within the single-module file-split semantics CLAUDE.md envisions. |
| 6 | `prediction/__init__.py` + `prediction/scoring.py` — re-exports & import fix | 3 | session 5 | Tiny adjustment-only session. |
| 7 | `training/pipeline.py` — compute raw metrics + book-baseline metrics + `brier_skill_score` + `kelly_shrinkage` | 4.a-c | session 6 | Pipeline is the heart of training; isolate the change. Validation block grows by `compute_metrics()` helper applied to model and book probs. |
| 8 | `training/report.py` — emit raw metrics + book-baseline row to parquet + txt; add `get_market_calibration` getter | 4.c, 4.e, 4.f, 4.g | session 7 | New `row_kind` column. Clean break — parquet is regenerated on next `meditate`. |
| 9 | `pages/7_📊_Stats_Model_Training.py` — restructured tabs, column-direction annotations, pinned book-baseline row | 4.d | session 8 | Largest dashboard change in the plan. Keep ≤300 lines per CLAUDE.md hard rule; if it exceeds, split tab renderers into `dashboard_data.py` helpers. |
| 10 | `strategies/kelly.py` — new module + CLI registration | 5 | session 9 | Adds cvxpy under `[strategy]` group in `pyproject.toml`; this is a one-line dependency edit done in the same session. |
| 11 | `src/deprecated/opt_kelley_bet.py` archive move + README update | 5.h | session 10 | Tiny housekeeping. |
| 12 | `strategies/underdog_pickem.py` — orchestrator + YAML emit | 6 | session 11 | Sheets is deprecated on `devel`, so YAML is the only sink. Keep ≤300 lines per CLAUDE.md hard rule; if it grows, split rank/emit helpers into a `strategies/_pickem_emit.py` sibling and continue in a follow-up session. |
| 13 | Documentation refresh (`CLAUDE.md`, `CONTRIBUTING.md`, roadmap, README) | 7 | session 12 | Doc-only, but qualifies as "one module" worth of work. |

**Sequencing constraints:**

- Steps 1.a/b/c can happen in any order internally; sessions 1–3 just
  pick the smallest first.
- The magic-number purge (session 4) must precede the parlay split
  (session 5) — moving constants while also moving code multiplies
  diff review pain.
- The training diagnostic (sessions 7–8) must precede Kelly (session
  10) so Kelly's resolution chain has a real value to read.
- Documentation (session 13) must be the last session so it can
  describe what actually shipped, not what was planned.

**Per-session ritual (from CLAUDE.md):**

1. Pull latest `claude/phase-3-impl-<step>` working branch.
2. Edit exactly the module listed above.
3. `poetry run ruff check src/sportstradamus/`
4. `poetry run pytest tests/golden/`
5. Commit with the step ref in the message.
6. Push and stop the session — do not start the next module in the
   same context.

---

## 9. Sequencing and PR Plan

Phase 3 lands as four PRs, each holding several of the per-module
sessions above. Each PR ships ruff-clean, golden-tests-green, and CLI
snapshots regenerated only on intentional flag changes.

| PR | Sessions | Scope |
|---|---|---|
| 3-pre | 1–6 | CLV bring-up, magic-number purge, parlay.py split |
| 3-train | 7–9 | Training-pipeline `kelly_shrinkage` diagnostic |
| 3-kelly | 10–11 | Kelly module + deprecated cleanup |
| 3-build | 12–13 | `pickem-build` (Power/Flex/Rivals) + docs |

## 10. Quality Gates (per PR)

1. `poetry run ruff check src/sportstradamus/`
2. `poetry run pytest tests/golden/`
3. `poetry run pytest tests/integration -m integration` if the change
   crosses module boundaries (PRs `3-pre`, `3-build`).
4. `REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py`
   on intentional CLI flag changes only (`3-kelly`, `3-build`).
5. For PR `3-pre`: re-run `scripts/audit_parlay_calibration.py` and
   diff the resulting plot — must be byte-identical pre/post the
   magic-number purge and the parlay split.

## 11. Done Criteria

Phase 3 is complete when:

1. `poetry run pickem-build --date today --bankroll 500` produces a
   ranked, sized YAML covering Power, Flex, and Rivals.
2. `poetry run kelly --from <yaml>` re-sizes from the same YAML
   offline.
3. `clv.summarize` continues to surface segments with no regressions,
   the new close-timestamp sanity warning fires in tests, and
   `data/clv_segments.parquet` is written by `reflect`.
4. After a fresh `poetry run meditate`, `data/training_report.txt`
   and `data/model_stats.parquet` carry the migrated raw-metric set
   (`brier_score`, `log_loss`, `roc_auc`,
   `expected_calibration_error`, `precision_over`, `precision_under`,
   `predicted_over_rate`, `empirical_over_rate`, `prediction_std`,
   `nll`, `brier_skill_score`, `kelly_shrinkage`) plus a book-baseline
   row (`row_kind="book_baseline"`). Dashboard page 7 renders the
   metric tabs with higher/lower-is-better column annotations and
   pins the book-baseline row above each table.
5. `prediction/parlay.py` exists with `beam_search_parlays` and its
   helpers; `prediction/correlation.py` retains `find_correlation`
   only; CONTRIBUTING.md §Package Map matches reality.
6. `src/deprecated/opt_kelley_bet.py` has moved to `.archived/` and
   the deprecated README's TODO list is shorter by one entry.
7. **Documentation reflects the shipped state:** `CLAUDE.md`,
   `CONTRIBUTING.md`, `docs/sportstradamus_roadmap_v2.md`, and the
   top-level `README.md` are updated per Step 7. Roadmap §3.1, §3.3,
   and §3.5 (Rivals only) are marked ✅ DONE; §3.4 and the rest of
   §3.5 are marked removed/deferred with a link to this plan.

## 12. Risks and Open Questions

- **Rivals payout multipliers.** Verify
  `data/underdog_payouts.json` reflects current Rivals product before
  PR `3-build`. Bad multipliers silently miscalibrate Rivals EV.
