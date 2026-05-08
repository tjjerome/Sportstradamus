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

## 4. Training Pipeline: Dedicated Kelly-Shrinkage Diagnostic (Step 4, ½ weekend)

### 4.a Why `model_calib` is the wrong field for Kelly

Verified at `src/sportstradamus/training/pipeline.py:688`:

```python
val_calibrated = expit(val_logits / T_opt)
model_calib = 1 - np.mean((val_calibrated - y_class_val) ** 2)
```

That is `1 − Brier(temperature_calibrated_probs, actual)` on the
validation set. It is a **calibration quality** score in roughly
`[0.75, 1.0]` — even a useless model that always predicts 0.5 has
Brier ≈ 0.25 → `model_calib ≈ 0.75`. Plugging it into
`effective_p = 0.5 + (p − 0.5) * model_calibration` would barely
shrink a useless model and barely modulate a great one.

What Kelly actually wants is a **discrimination / skill** score:
how much better than the bookmaker baseline our calibrated
probabilities are.

### 4.b New diagnostic: `kelly_shrinkage`

Add alongside `model_calib`:

```python
brier_model = np.mean((val_calibrated - y_class_val) ** 2)
brier_book  = np.mean((val_book_proba - y_class_val) ** 2)
bss = 1.0 - (brier_model / max(brier_book, 1e-9))
kelly_shrinkage = float(np.clip(bss, 0.0, 1.0))
```

Brier Skill Score against the bookmaker baseline, clamped to `[0, 1]`:
`0.0` = no better than the book (Kelly degenerates to half-Kelly on
book-implied prob), `1.0` = perfect discrimination on the holdout
(no shrink).

If the validation set has no book-implied probability column, log a
WARNING and store `kelly_shrinkage = nan`; Kelly's resolution chain
(Step 5.c) handles `nan` by falling through to the next source.

### 4.c Where the diagnostic surfaces

Two sinks already exist — no JSON refactor needed:

1. **Human-readable** `data/training_report.txt` — extend the DIAG
   block in `report.py:write_report` to print
   `kelly_shrinkage=0.43` next to `model_calib=0.82`.
2. **Machine-readable** `data/model_stats.parquet` — add a
   `kelly_shrinkage` column in `report.py:write_model_stats` (line
   227-ish, alongside `model_weight` and `model_calib`).

The dashboard already reads `model_stats.parquet` via
`dashboard_data.load_model_stats` and renders it in
`pages/7_📊_Stats_Model_Training.py`. Extending the parquet adds the
column transparently; the page renders it because it auto-discovers
columns. Update the page only to include `kelly_shrinkage` in any
explicit column list it constructs (verify on read).

### 4.d Thin getter

```python
# in training/report.py
def get_market_calibration(league: str, market: str) -> dict[str, float]:
    """Return {'kelly_shrinkage', 'model_calib', 'model_weight'} for one (league, market).

    Reads model_stats.parquet. Returns NaNs when the row is missing.
    """
```

Kelly imports this getter — never reaches into the parquet directly.

**Files touched:** `src/sportstradamus/training/pipeline.py`,
`src/sportstradamus/training/report.py`,
`src/sportstradamus/pages/7_📊_Stats_Model_Training.py` (only if it
hard-codes a column allowlist).
**Tests:** `tests/golden/test_training_kelly_shrinkage.py`:
- BSS computation matches a hand-worked example.
- Useless model (Brier matching book) → `kelly_shrinkage = 0`.
- Missing book column → `nan` and WARNING logged.
- Getter returns NaNs when (league, market) absent.

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

### 5.d Shrinkage resolution chain

```
effective_p = 0.5 + (win_prob - 0.5) * shrinkage
```

`shrinkage` resolved in this order, first non-NaN wins:

1. Explicit `model_shrinkage` kwarg.
2. `clv.get_segment_calibration(league, market)` (Step 1.b) — recent
   season-long sharpness signal.
3. `training.report.get_market_calibration(league, market)`
   `['kelly_shrinkage']` (Step 4.b) — training-time skill on holdout.
4. Fallback `1.0` (no shrinkage). Logged at DEBUG.

CLV-segment value wins over training-time when both exist because it
reflects current-season market conditions. Document this priority
explicitly in the kelly.py module docstring.

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
6. **Emit:**
   - "Pickem Recommendations" tab in Sheets export with `Variant`
     column.
   - `data/recommendations/{date}.yaml`.

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
- **Training-report glossary in CLAUDE.md:** add a line for
  `kelly_shrinkage` next to the existing `model_calib` line.

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
| 7 | `training/pipeline.py` — compute `kelly_shrinkage` next to `model_calib` | 4.b | session 6 | Pipeline is the heart of training; isolate the change. |
| 8 | `training/report.py` — emit `kelly_shrinkage` to txt + parquet + add `get_market_calibration` getter | 4.c, 4.d | session 7 | |
| 9 | `pages/7_📊_Stats_Model_Training.py` — surface new column if needed | 4.c | session 8 | Verify the page; no-op if it auto-discovers columns. |
| 10 | `strategies/kelly.py` — new module + CLI registration | 5 | session 9 | Adds cvxpy under `[strategy]` group in `pyproject.toml`; this is a one-line dependency edit done in the same session. |
| 11 | `src/deprecated/opt_kelley_bet.py` archive move + README update | 5.h | session 10 | Tiny housekeeping. |
| 12 | `strategies/underdog_pickem.py` — orchestrator + Sheets tab + YAML | 6 | session 11 | The largest session in this plan; keep ≤300 lines per CLAUDE.md hard rule. If it grows, split rank/emit helpers into a `strategies/_pickem_emit.py` sibling and continue in a follow-up session. |
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
   ranked, sized YAML covering Power, Flex, and Rivals and updates the
   Sheets tab.
2. `poetry run kelly --from <yaml>` re-sizes from the same YAML
   offline.
3. `clv.summarize` continues to surface segments with no regressions,
   the new close-timestamp sanity warning fires in tests, and
   `data/clv_segments.parquet` is written by `reflect`.
4. `data/training_report.txt` and `data/model_stats.parquet` both
   contain the new `kelly_shrinkage` column. Dashboard renders it.
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

- **`kelly_shrinkage` early-season noise.** BSS on a small validation
  set is itself noisy. The clamp to `[0, 1]` plus the resolution-chain
  fallback (CLV-segment > training > 1.0) keeps a noisy retrain from
  destabilizing Kelly, but consider a minimum validation-set size
  before trusting the value.
- **Rivals payout multipliers.** Verify
  `data/underdog_payouts.json` reflects current Rivals product before
  PR `3-build`. Bad multipliers silently miscalibrate Rivals EV.
- **Sheets tab churn.** Adding a `Variant` column to the
  recommendation tab may break downstream sheet consumers if the user
  has linked cells. Surface in the PR description so they can adjust
  before merge.
