# Feature Filter Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the on-disk NFL feature filter (which is missing 17-27 of 30 SHAP-floored features per cell across 20 of 22 markets), then close the codebase divergence that allowed the filter to become inconsistent, then apply literature-backed improvements to the selection algorithm.

**Architecture:** Three-phase rollout. Phase 1 is a one-time data fix (regenerate filter using existing scouting SHAP, retrain 5 affected models) with no code change. Phase 2 closes the bug class (centralize selection constants, add a CI consistency test, drop redundancy penalty per tree-model literature). Phase 3 is methodology improvement (per-cell cap scaling, joint-SHAP audit for ZINB, stability metric replacement). A Phase 4 evaluation track is queued separately and gated on Optuna wall-clock profiling.

**Tech Stack:** Python 3.11, LightGBM/LightGBMLSS, SHAP, scikit-learn, click, pytest, ruff. Codebase rules: no monoliths >300 lines, no commented-out code, no magic numbers (CLAUDE.md). All work runs on the `model-research` branch.

**Evidence base:**

- Phase 1 diagnostics (this session): `_scouting_shap_and_filter` invoked directly in Python produces a filter with all 30 SHAP-floored features; on-disk filter has only 3-10 of them per cell across 20 of 22 NFL markets. Test A (broken filter, deterministic) vs Test B (corrected filter, deterministic) on `NFL passing first downs`: BSS moved -0.017 → +0.012, a +0.028 lift before full Optuna. This is a floor estimate; full Optuna with the fixed filter should recover most of the remaining 0.083 gap to the 0.095 baseline.
- Research brief at `/tmp/researcher_feature_filtering.md` (547 lines, cited papers): tree ensembles don't statistically benefit from pre-filtering (Akhiat & Touchanti 2024), the composite weight vector has no literature support, the redundancy penalty is wrong for tree models (Strobl 2008 et al. unanimous), SHAP dilution is a named phenomenon ("first-mover bias", Caraker 2026), KEEP_CAP should scale with `n_train` per events-per-variable, drop-in replacements exist (powershap, shap-select).

---

## Phase 1 — Immediate Fix (data rewrite, no code change)

**Goal:** Restore the 5 regressed NFL cells to within ±0.02 BSS of their pre-Phase-2 baseline. No code change in this phase; just regenerate the filter using existing (correct) scouting SHAP and retrain the affected models.

**Risk:** Low. The fix is reversible: backups exist in `/tmp/_diag_backup/` and `/tmp/_pickle_backup/`. The 5 regressed cells already underperform book baseline, so any working filter is an improvement.

### Task 1: Back up the current dirty state

**Files:**
- Create backup dir: `/tmp/feature_filter_remediation_phase1_backup/`

- [ ] **Step 1: Run the backup**

```bash
mkdir -p /tmp/feature_filter_remediation_phase1_backup
cp src/sportstradamus/data/config/feature_filter.json /tmp/feature_filter_remediation_phase1_backup/
cp src/sportstradamus/data/training/feature_importances.csv /tmp/feature_filter_remediation_phase1_backup/
cp src/sportstradamus/data/training/feature_correlations.csv /tmp/feature_filter_remediation_phase1_backup/
cp src/sportstradamus/data/training/model_stats.parquet /tmp/feature_filter_remediation_phase1_backup/
cp src/sportstradamus/data/training/training_report.txt /tmp/feature_filter_remediation_phase1_backup/
cp src/sportstradamus/data/models/NFL_passing-first-downs.mdl /tmp/feature_filter_remediation_phase1_backup/
cp src/sportstradamus/data/models/NFL_receiving-tds.mdl /tmp/feature_filter_remediation_phase1_backup/
cp src/sportstradamus/data/models/NFL_receptions.mdl /tmp/feature_filter_remediation_phase1_backup/
cp src/sportstradamus/data/models/NFL_fantasy-points-prizepicks.mdl /tmp/feature_filter_remediation_phase1_backup/
cp src/sportstradamus/data/models/NFL_fantasy-points-underdog.mdl /tmp/feature_filter_remediation_phase1_backup/
ls /tmp/feature_filter_remediation_phase1_backup/
```

Expected: 11 files listed (5 mdl + 5 training artifacts + 1 config).

### Task 2: Rewrite NFL filter for all 22 markets using current scouting SHAP

**Files:**
- Modify: `src/sportstradamus/data/config/feature_filter.json` (data only, no code change)

- [ ] **Step 1: Run the filter rewrite**

```bash
poetry run python - << 'PY'
import sys, json
sys.path.insert(0, "src")
from sportstradamus.helpers import feature_filter
from sportstradamus.training.shap import filter_market

markets = sorted(feature_filter["NFL"]["Filtered"].keys())
print(f"Rewriting filter for {len(markets)} NFL markets...")
total_dropped = 0
total_added = 0
for m in markets:
    diag = filter_market("NFL", m)
    total_dropped += diag["n_dropped"]
    total_added += diag["n_added"]
    print(f"  {m}: dropped={diag['n_dropped']} added={diag['n_added']} floored={diag['n_shap_floored']}")
print(f"\nDone. total_dropped={total_dropped} total_added={total_added}")
PY
```

Expected: Per-market line with `dropped` and `added` >0 for most markets; `floored=30` for each. Final summary prints totals.

- [ ] **Step 2: Spot-check that fixed filter includes SHAP-floored features**

```bash
poetry run python - << 'PY'
import json
with open("src/sportstradamus/data/config/feature_filter.json") as f:
    cf = json.load(f)
for cell in ("passing first downs", "receiving tds", "receptions", "fantasy points prizepicks", "fantasy points underdog"):
    s = set(cf["NFL"]["Filtered"][cell])
    have = sum(int(f in s) for f in ("Player breakaway yards", "Player aggressiveness", "Player midfield target rate", "Player pbp_epa_per_play_asof"))
    print(f"  {cell}: {len(s)} features, {have}/4 known-good Player features present")
PY
```

Expected: each cell shows 60 features and `4/4` known-good features (or `3/4` if a particular cell didn't have one in its top-30 historically; spot-check passes if ≥3 of 4 are present).

### Task 3: Delete the 5 regressed model pickles to force retrain

**Files:**
- Delete: `src/sportstradamus/data/models/NFL_passing-first-downs.mdl`
- Delete: `src/sportstradamus/data/models/NFL_receiving-tds.mdl`
- Delete: `src/sportstradamus/data/models/NFL_receptions.mdl`
- Delete: `src/sportstradamus/data/models/NFL_fantasy-points-prizepicks.mdl`
- Delete: `src/sportstradamus/data/models/NFL_fantasy-points-underdog.mdl`

- [ ] **Step 1: Delete the pickles**

```bash
rm src/sportstradamus/data/models/NFL_passing-first-downs.mdl
rm src/sportstradamus/data/models/NFL_receiving-tds.mdl
rm src/sportstradamus/data/models/NFL_receptions.mdl
rm src/sportstradamus/data/models/NFL_fantasy-points-prizepicks.mdl
rm src/sportstradamus/data/models/NFL_fantasy-points-underdog.mdl
ls src/sportstradamus/data/models/NFL_passing-first-downs.mdl 2>&1
```

Expected: `ls: cannot access ... No such file or directory`.

### Task 4: Run meditate to retrain the 5 deleted cells

**Files:** Reads/writes `src/sportstradamus/data/training/model_stats.parquet`, `training_report.txt`, and the 5 model pickles.

- [ ] **Step 1: Run meditate (no --rebuild-filter, since Task 2 already corrected the filter)**

```bash
poetry run meditate --league NFL --bypass-withholding 2>&1 | tee /tmp/phase1_meditate.log
```

Expected: For each of the 5 deleted cells, full Optuna runs (~10-60 min per cell, 50-300 min total). Other markets skip after "Gathering Training Data: 0gameday" because they have fresh pickles. Final lines of the log show "Persisting artifacts" or similar per cell.

If a cell errors, restore the pickle from `/tmp/feature_filter_remediation_phase1_backup/` and investigate:

```bash
cp /tmp/feature_filter_remediation_phase1_backup/NFL_<cell>.mdl src/sportstradamus/data/models/
```

- [ ] **Step 2: Verify all 5 pickles regenerated**

```bash
ls -la src/sportstradamus/data/models/NFL_passing-first-downs.mdl src/sportstradamus/data/models/NFL_receiving-tds.mdl src/sportstradamus/data/models/NFL_receptions.mdl src/sportstradamus/data/models/NFL_fantasy-points-prizepicks.mdl src/sportstradamus/data/models/NFL_fantasy-points-underdog.mdl
```

Expected: All 5 files exist with mtime within the meditate run window.

### Task 5: Verify BSS recovery

**Files:** Reads `src/sportstradamus/data/training/model_stats.parquet` and `/tmp/baseline_pre_regen/model_stats.parquet`.

- [ ] **Step 1: Compute BSS deltas vs baseline**

```bash
poetry run python - << 'PY'
import pandas as pd
cur = pd.read_parquet("src/sportstradamus/data/training/model_stats.parquet")
bas = pd.read_parquet("/tmp/baseline_pre_regen/model_stats.parquet")
cells = ["passing first downs", "receiving tds", "receptions", "fantasy points prizepicks", "fantasy points underdog"]
def slice_(df):
    return df[(df["league"] == "NFL") & (df["market"].isin(cells)) & (df["row_kind"] == "model") & (df["metric_row"] == "calibrated")][["market", "brier_skill_score"]].set_index("market")
c = slice_(cur).rename(columns={"brier_skill_score": "current"})
b = slice_(bas).rename(columns={"brier_skill_score": "baseline"})
m = c.join(b)
m["delta"] = m["current"] - m["baseline"]
m["within_tolerance"] = m["delta"].abs() <= 0.02
print(m.to_string())
print(f"\nAll within ±0.02? {m['within_tolerance'].all()}")
PY
```

Expected: All 5 cells show `within_tolerance=True`. If any fails, this signals the filter-membership fix is not sufficient and we need to drill into the failing cell (likely candidates: distribution choice changed, or that cell genuinely benefited from one of the removed features). Do not move on to Phase 2 until this gate is green.

- [ ] **Step 2: Verify 6 previously-improved cells didn't regress**

```bash
poetry run python - << 'PY'
import pandas as pd
cur = pd.read_parquet("src/sportstradamus/data/training/model_stats.parquet")
bas = pd.read_parquet("/tmp/baseline_pre_regen/model_stats.parquet")
# All NFL markets that are in BOTH current and baseline (had a baseline to compare against)
cur_nfl = cur[(cur["league"] == "NFL") & (cur["row_kind"] == "model") & (cur["metric_row"] == "calibrated")][["market", "brier_skill_score"]].set_index("market")
bas_nfl = bas[(bas["league"] == "NFL") & (bas["row_kind"] == "model") & (bas["metric_row"] == "calibrated")][["market", "brier_skill_score"]].set_index("market")
merged = cur_nfl.join(bas_nfl, how="inner", lsuffix="_cur", rsuffix="_bas")
merged["delta"] = merged["brier_skill_score_cur"] - merged["brier_skill_score_bas"]
print(merged.sort_values("delta").to_string())
print(f"\nMax regression: {merged['delta'].min():.4f} on market {merged['delta'].idxmin()}")
PY
```

Expected: No cell regresses more than -0.01 vs baseline. If any does, investigate before moving on.

### Task 6: Commit the fixed filter and regenerated artifacts

**Files:** All under `src/sportstradamus/data/`.

- [ ] **Step 1: Inspect staged diff**

```bash
git status
git diff --stat src/sportstradamus/data/config/feature_filter.json
```

Expected: Only data files (config + training artifacts + 5 model pickles) appear in the diff. No code files.

- [ ] **Step 2: Stage and commit**

```bash
git add src/sportstradamus/data/config/feature_filter.json src/sportstradamus/data/training/feature_importances.csv src/sportstradamus/data/training/feature_correlations.csv src/sportstradamus/data/training/model_stats.parquet src/sportstradamus/data/training/training_report.txt src/sportstradamus/data/models/NFL_passing-first-downs.mdl src/sportstradamus/data/models/NFL_receiving-tds.mdl src/sportstradamus/data/models/NFL_receptions.mdl src/sportstradamus/data/models/NFL_fantasy-points-prizepicks.mdl src/sportstradamus/data/models/NFL_fantasy-points-underdog.mdl
git commit -m "$(cat <<'EOF'
fix(filter): regenerate NFL filter to include SHAP-floored features

20 of 22 NFL Filtered markets were missing 17-27 of the 30 features that
the current `shap_floor_base_names` veto-on-drop layer should have kept.
Symptom was the 5-cell BSS regression in the 2026-05-26 meditate run
(passing first downs -0.109, receiving tds -0.034, receptions -0.027,
fp prizepicks -0.014, fp underdog -0.011 vs pre-Phase-2 baseline).

The on-disk filter was inconsistent with what `filter_market_features`
produces from the same scouting SHAP CSV — direct invocation in Python
puts all 30 floored features in. Likely cause is the
`scripts/evaluate_model_features.py --save` interactive path writing
feature_filter.json with a different scoring algorithm than the
training-pipeline path. Phase 2 of this remediation plan addresses the
divergence.

This commit just regenerates the filter via `filter_market` for each of
the 22 NFL markets (no scouting re-run, SHAP CSV unchanged), deletes
the 5 regressed model pickles, and retrains them via meditate without
`--rebuild-filter`. Test A/B in deterministic mode confirmed the filter
swap alone moves passing-first-downs BSS from -0.017 → +0.012 before
full Optuna recovery.

Verification: all 5 cells within ±0.02 BSS of `/tmp/baseline_pre_regen`
baseline; no cell with an existing baseline regresses >0.01.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: Commit lands. Run `git log --oneline -1` to confirm.

---

## Phase 2 — Hygiene: close the divergence path

**Goal:** Prevent a future regression of this class by (a) making `feature_selection.py` the single source of truth for selection constants, (b) adding a CI test that asserts disk filter is consistent with `filter_market_features` output, (c) setting `REDUNDANCY_WEIGHT = 0` per the tree-model literature.

**Risk:** Low-to-medium. The constants extraction is mechanical. The consistency test enforces a useful invariant. The REDUNDANCY_WEIGHT change re-ranks features at the composite-cap boundary — should be a small net BSS lift but worth re-verifying.

### Task 7: Centralize selection constants in `feature_selection.py`

The constants are already in `feature_selection.py`. The problem is that `scripts/evaluate_model_features.py` defines its own thresholds (`EVAL_DROP_CUTOFF`, `EVAL_TOP_CANDIDATES`) and calls `composite_score` with `has_model=True/False` patterns inconsistent with the training pipeline. The fix: have the eval script import a single `EvalDefaults` dataclass from `feature_selection.py` and call a single shared `filter_market_features` function (already exists) instead of its own bespoke `evaluate_market` + `apply_recommendations` logic.

**Files:**
- Modify: `src/sportstradamus/feature_selection.py` (verify constants are module-level; add docstring noting these are the source of truth)
- Modify: `src/sportstradamus/scripts/evaluate_model_features.py` (replace `evaluate_market` body with a call to `filter_market_features`; remove duplicate thresholds)

- [ ] **Step 1: Read `feature_selection.py:48-72` and `scripts/evaluate_model_features.py:50-180`**

Confirm the constants `W_MODEL`, `W_NO_MODEL`, `REDUNDANCY_WEIGHT`, `DROP_CUTOFF`, `ADD_THRESHOLD`, `KEEP_FLOOR`, `KEEP_CAP`, `SHAP_FLOOR_K`, `EVAL_DROP_CUTOFF`, `EVAL_TOP_CANDIDATES` are all defined in `feature_selection.py`. Read the eval script's `evaluate_market` (line 82) and `apply_recommendations` (line 269) to understand what it currently does.

- [ ] **Step 2: Write a failing test**

Add to `tests/golden/test_feature_selection.py`:

```python
def test_eval_script_uses_same_filter_logic_as_training():
    """The eval script must produce identical filter output to filter_market_features
    when run with --save defaults on the same data. Prevents divergence regressions."""
    import json
    from sportstradamus.feature_selection import filter_market_features
    from sportstradamus.training.shap import _load_shap_corr_dfs
    from sportstradamus.scripts.evaluate_model_features import (
        evaluate_market,
        apply_recommendations,
    )

    with open("src/sportstradamus/data/config/feature_filter.json") as f:
        ff = json.load(f)
    shap_df, corr_df = _load_shap_corr_dfs()
    market = "passing first downs"

    expected, _ = filter_market_features("NFL", market, ff, shap_df, corr_df)

    eval_result = evaluate_market("NFL", market, ff, shap_df, corr_df, drop_threshold=0.10, add_threshold=0.50, top_candidates=30)
    actual = apply_recommendations(ff, "NFL", market, eval_result, drop_threshold=0.10, add_threshold=0.50)

    assert sorted(actual) == sorted(expected), (
        f"Eval script diverges from filter_market_features:\n"
        f"  eval-only: {sorted(set(actual) - set(expected))}\n"
        f"  training-only: {sorted(set(expected) - set(actual))}"
    )
```

- [ ] **Step 3: Run the test, confirm it fails**

```bash
poetry run pytest tests/golden/test_feature_selection.py::test_eval_script_uses_same_filter_logic_as_training -v
```

Expected: FAIL with mismatched lists (or with an error about `apply_recommendations` having a different signature). This is the bug.

- [ ] **Step 4: Replace `evaluate_model_features.py:evaluate_market` body to delegate to `filter_market_features`**

In `src/sportstradamus/scripts/evaluate_model_features.py`, rewrite `evaluate_market` (lines 82-260, exact range to verify when reading) to be a thin wrapper that calls `filter_market_features` and reshapes its return value to the dict the eval CLI expects. Delete `apply_recommendations`'s composite-scoring branches and have it just take the new filtered list from `filter_market_features`.

The replacement function body (everything inside `def evaluate_market`):

```python
def evaluate_market(
    league: str,
    market: str,
    feature_filter: dict,
    shap_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
    add_threshold: float = 0.50,
    top_candidates: int = DEFAULT_TOP_CANDIDATES,
) -> dict:
    """Evaluate a market by delegating to the training-pipeline filter logic.

    All scoring, weighting, and decision-layer logic lives in
    ``feature_selection.filter_market_features`` — this script is now a
    presentation layer over that single source of truth.
    """
    from sportstradamus.feature_selection import filter_market_features

    new_filtered, diag = filter_market_features(
        league,
        market,
        feature_filter,
        shap_df,
        corr_df,
        drop_cutoff=drop_threshold,
        add_threshold=add_threshold,
    )
    return {
        "new_filtered": new_filtered,
        "drop": [f for f, _ in sorted(diag["scores"].items(), key=lambda kv: kv[1]) if f not in new_filtered][:top_candidates],
        "add": [f for f, _ in sorted(diag["candidate_scores"].items(), key=lambda kv: -kv[1]) if f in new_filtered][:top_candidates],
        "n_shap_floored": diag["n_shap_floored"],
    }
```

And rewrite `apply_recommendations` (lines 269-285 area) to just use the `new_filtered` field:

```python
def apply_recommendations(
    feature_filter: dict,
    league: str,
    market: str,
    eval_result: dict,
    drop_threshold: float,
    add_threshold: float,
) -> list[str]:
    """Apply the new filter list from evaluate_market. Always returns the
    full filter — drop/add are presentation-only at this point because
    filter_market_features already applied them."""
    return list(eval_result["new_filtered"])
```

- [ ] **Step 5: Run the test, confirm it passes**

```bash
poetry run pytest tests/golden/test_feature_selection.py::test_eval_script_uses_same_filter_logic_as_training -v
```

Expected: PASS.

- [ ] **Step 6: Re-run the existing 8 golden tests to confirm no regression**

```bash
poetry run pytest tests/golden/test_feature_selection.py -v
```

Expected: All tests pass.

- [ ] **Step 7: Run refactoring-specialist on the two files**

Per CLAUDE.md, this is mandatory before commit.

```
Agent({
  subagent_type: "refactoring-specialist",
  prompt: "Review src/sportstradamus/feature_selection.py and src/sportstradamus/scripts/evaluate_model_features.py for STYLE_GUIDE compliance. Phase 2 of feature-filter remediation just rewrote evaluate_market to delegate to filter_market_features."
})
```

Address anything it flags.

- [ ] **Step 8: Commit**

```bash
git add tests/golden/test_feature_selection.py src/sportstradamus/scripts/evaluate_model_features.py
git commit -m "$(cat <<'EOF'
refactor(fs): eval script delegates to filter_market_features

evaluate_model_features.py used to maintain its own copy of the
composite-scoring + drop/add logic, which drifted from
feature_selection.filter_market_features over time. This is the most
plausible cause of the 2026-05-27 filter corruption (20 of 22 NFL
markets missing SHAP-floored features) — running --save with the eval
script's defaults wrote a different filter than the training pipeline
would produce on the same SHAP CSV.

This commit makes evaluate_market a thin wrapper over
filter_market_features so both paths use the exact same selection
logic and constants. Adds test_eval_script_uses_same_filter_logic_as_training
to pin the invariant and catch future drift in CI.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Task 8: Add disk-filter consistency CI test

This test catches the broader "disk filter ≠ what current code produces" class of bug, regardless of which code wrote it.

**Files:**
- Create: `tests/golden/test_filter_consistency.py`

- [ ] **Step 1: Write the test file**

```python
"""Disk filter must be consistent with what filter_market_features would
produce now. Catches the 2026-05-27 corruption pattern where the on-disk
filter was stamped by a different code path and silently regressed BSS.

The test allows a small tolerance for boundary instability — features
that score at composite ranks 55-65 are at the cap edge and may legitimately
churn between rebuilds. The SHAP-floored top-30 must be exact, however.
"""
from __future__ import annotations

import json

import pytest

from sportstradamus.feature_selection import filter_market_features
from sportstradamus.training.shap import _load_shap_corr_dfs

with open("src/sportstradamus/data/config/feature_filter.json") as f:
    FEATURE_FILTER = json.load(f)


def _enumerate_cells():
    """All (league, market) pairs in the Filtered section of disk."""
    for league, sub in FEATURE_FILTER.items():
        for market in sorted(sub.get("Filtered", {}).keys()):
            yield league, market


@pytest.mark.parametrize("league, market", list(_enumerate_cells()))
def test_disk_filter_contains_all_shap_floored_features(league: str, market: str):
    """Every base name that filter_market_features would protect with the
    SHAP floor MUST be on disk. Boundary-rank features (composite ranks
    near KEEP_CAP) are allowed to differ between disk and a fresh rebuild,
    but the top-30 by |SHAP| is the model's revealed importance and must
    not be silently dropped."""
    shap_df, corr_df = _load_shap_corr_dfs()
    _, diag = filter_market_features(league, market, FEATURE_FILTER, shap_df, corr_df)
    disk_filter = set(FEATURE_FILTER[league]["Filtered"][market])
    floored = set(diag["shap_floored"])
    missing = floored - disk_filter
    assert not missing, (
        f"{league}_{market}: disk filter is missing {len(missing)} of {len(floored)} "
        f"SHAP-floored features. Run `meditate --rebuild-filter --league {league}` "
        f"(or `filter_market(league, market)` for a fast one-cell rewrite). "
        f"Missing: {sorted(missing)}"
    )
```

- [ ] **Step 2: Run the test**

```bash
poetry run pytest tests/golden/test_filter_consistency.py -v
```

Expected: After Phase 1, all NFL parameterizations should pass. NBA/MLB/NHL/WNBA may or may not pass — if any fail, that's a real bug to surface but should NOT block this PR. Mark cross-league failures as XFAIL with a comment pointing to a follow-up issue, and gate this test on NFL only for now. We address other-league filters in a separate PR.

To make the test NFL-only initially:

```python
@pytest.mark.parametrize("league, market", [(l, m) for l, m in _enumerate_cells() if l == "NFL"])
```

- [ ] **Step 3: Commit**

```bash
git add tests/golden/test_filter_consistency.py
git commit -m "$(cat <<'EOF'
test(fs): add disk filter ↔ filter_market_features consistency test

Pins the invariant that the on-disk filter contains every SHAP-floored
base name that filter_market_features would protect. Catches the
2026-05-27 corruption class where a separate code path stamped the
filter with a stale selection algorithm.

NFL-only for now; extend to other leagues in a follow-up after auditing
their current filter consistency.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Task 9: Set REDUNDANCY_WEIGHT = 0 per tree-model literature

Strobl 2008 (DOI 10.1186/1471-2105-9-307), Genuer 2010, Hapfelmeier-Ulm 2013, Janitza 2018 are unanimous: tree ensembles exploit correlated features at separate splits — a redundancy penalty hurts them. The 2026-05-26 tune from 0.40 → 0.15 was directionally right; the principled value is 0.

**Files:**
- Modify: `src/sportstradamus/feature_selection.py:55` (REDUNDANCY_WEIGHT constant + docstring)
- Modify: `tests/golden/test_feature_selection.py:115` (`test_redundancy_weight_pinned_at_0_15` → `test_redundancy_weight_pinned_at_0` with literature citation)

- [ ] **Step 1: Update the constant + docstring**

In `src/sportstradamus/feature_selection.py`, replace lines 50-56 (the REDUNDANCY_WEIGHT block):

```python
# Redundancy penalty multiplier `composite *= (1 - REDUNDANCY_WEIGHT * max_|corr|)`.
# Pinned to 0.0 per the tree-model selection literature (Strobl et al. 2008,
# DOI 10.1186/1471-2105-9-307; Genuer et al. 2010, DOI 10.1016/j.patrec.2010.03.014;
# Hapfelmeier & Ulm 2013, DOI 10.1016/j.csda.2012.09.020; Janitza et al. 2018,
# DOI 10.1007/s11634-016-0276-4): LightGBM exploits correlated features at
# separate splits and benefits from keeping them. mRMR-style redundancy
# penalties are for L1 / LASSO, not gradient-boosted trees.
REDUNDANCY_WEIGHT = 0.0
```

- [ ] **Step 2: Update the test**

In `tests/golden/test_feature_selection.py`, find `test_redundancy_weight_pinned_at_0_15` (around line 115). Replace its assertion to expect 0.0 and rename for clarity:

```python
def test_redundancy_weight_pinned_at_zero():
    """REDUNDANCY_WEIGHT is pinned to 0.0 — tree models benefit from
    correlated features at separate splits. See feature_selection.py
    constant docstring for citations."""
    from sportstradamus.feature_selection import REDUNDANCY_WEIGHT
    assert REDUNDANCY_WEIGHT == 0.0
```

- [ ] **Step 3: Run the test**

```bash
poetry run pytest tests/golden/test_feature_selection.py::test_redundancy_weight_pinned_at_zero -v
```

Expected: PASS.

- [ ] **Step 4: Confirm no other test depended on the 0.15 value**

```bash
poetry run pytest tests/golden/test_feature_selection.py -v
```

Expected: All tests pass. If any fail because they encoded the 0.15 value, update them to expect the new behavior (no redundancy penalty) and add a comment citing the rationale.

- [ ] **Step 5: Re-run filter rewrite for NFL (the constant change moves the cap boundary)**

```bash
poetry run python - << 'PY'
import sys
sys.path.insert(0, "src")
from sportstradamus.helpers import feature_filter
from sportstradamus.training.shap import filter_market

for m in sorted(feature_filter["NFL"]["Filtered"].keys()):
    diag = filter_market("NFL", m)
    print(f"  {m}: dropped={diag['n_dropped']} added={diag['n_added']}")
PY
```

Expected: Most cells show small per-cell churn (5-15 features) because the cap boundary shifted slightly. SHAP-floored features remain stable.

- [ ] **Step 6: Run the consistency test from Task 8 to confirm SHAP floor still holds**

```bash
poetry run pytest tests/golden/test_filter_consistency.py -v
```

Expected: PASS.

- [ ] **Step 7: Retrain 5 cells, verify no BSS regression vs Phase 1 result**

```bash
rm src/sportstradamus/data/models/NFL_passing-first-downs.mdl src/sportstradamus/data/models/NFL_receiving-tds.mdl src/sportstradamus/data/models/NFL_receptions.mdl src/sportstradamus/data/models/NFL_fantasy-points-prizepicks.mdl src/sportstradamus/data/models/NFL_fantasy-points-underdog.mdl
poetry run meditate --league NFL --bypass-withholding 2>&1 | tee /tmp/phase2_meditate.log
```

Expected: Same 5 cells retrain. Verify BSS against Phase 1 baseline (use the comparison script from Task 5). Should be ≥ Phase 1 BSS within 0.01. The literature predicts a small lift from removing the misaligned penalty.

- [ ] **Step 8: Run refactoring-specialist on touched files, then commit**

```bash
git add src/sportstradamus/feature_selection.py tests/golden/test_feature_selection.py src/sportstradamus/data/config/feature_filter.json src/sportstradamus/data/training/feature_importances.csv src/sportstradamus/data/training/feature_correlations.csv src/sportstradamus/data/training/model_stats.parquet src/sportstradamus/data/training/training_report.txt src/sportstradamus/data/models/NFL_passing-first-downs.mdl src/sportstradamus/data/models/NFL_receiving-tds.mdl src/sportstradamus/data/models/NFL_receptions.mdl src/sportstradamus/data/models/NFL_fantasy-points-prizepicks.mdl src/sportstradamus/data/models/NFL_fantasy-points-underdog.mdl
git commit -m "$(cat <<'EOF'
fix(fs): set REDUNDANCY_WEIGHT = 0 per tree-model literature

Strobl 2008, Genuer 2010, Hapfelmeier-Ulm 2013, Janitza 2018 are
unanimous that tree ensembles benefit from correlated features at
separate splits — a redundancy penalty strips incremental signal
SHAP already credited. The 2026-05-26 tune from 0.40 → 0.15 was
directionally right; this commit completes the move to the principled
value of 0.0.

Regenerates NFL filter + retrains 5 affected cells. BSS unchanged or
improved across the 5 vs Phase 1 of this remediation.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Methodology: per-cell cap + joint-SHAP audit

**Goal:** Stop the cap from displacing useful features in larger-sample cells, and verify the joint-SHAP summing across LightGBMLSS distribution parameters isn't quietly diluting the gate-head importance for ZINB/ZAGamma cells.

**Risk:** Medium. Per-cell cap scaling will increase filter sizes for high-sample cells (e.g., `carries` at 6,657 rows could go from 60 → 133 features). Optuna wall-clock will increase proportionally. The joint-SHAP audit is read-only (Phase 3.B); the joint-SHAP fix (if needed) is a separate PR.

### Task 10: Per-cell KEEP_CAP scaled with n_train

Harrell 2015 events-per-variable heuristic: for binary outcomes, allow ~1 predictor per 10-20 events; for continuous, ~1 per 25-50. SkewNormal/NegBin cells have variable n_train (2,279 for `passing first downs`, 6,657 for `carries`). A global cap of 60 over-constrains the high-sample cells and is fine for the low-sample ones.

**Files:**
- Modify: `src/sportstradamus/feature_selection.py:62` (KEEP_CAP constant + `_keep_cap_for_market` helper)
- Modify: `src/sportstradamus/feature_selection.py:557` (KEEP_CAP enforcement to use per-cell value)
- Modify: `tests/golden/test_feature_selection.py:281` (`test_keep_cap_protects_shap_floored_features` to use the helper)

- [ ] **Step 1: Add a `_keep_cap_for_market` helper to feature_selection.py**

After the `KEEP_CAP = 60` constant (around line 62), add:

```python
# Events-per-variable scaling (Harrell 2015, Regression Modeling Strategies §4.6).
# The 60-feature floor preserves the historical default for small-n cells; high-n
# cells get more capacity so the cap doesn't displace useful long-tail features.
# Divisor 50 is the continuous-regression rule-of-thumb; for binary classification
# tighten to 20 if/when we add a classification path.
_KEEP_CAP_EVR_DIVISOR = 50


def keep_cap_for_market(n_train: int, base: int = KEEP_CAP) -> int:
    """Per-market cap = max(base, n_train // _KEEP_CAP_EVR_DIVISOR).

    Args:
        n_train: Number of training rows for the market.
        base: Floor on the cap (default KEEP_CAP=60).

    Returns:
        Cap value to use for this market.
    """
    if n_train <= 0:
        return base
    return max(base, n_train // _KEEP_CAP_EVR_DIVISOR)
```

- [ ] **Step 2: Modify `filter_market_features` to use the helper**

Around line 557 of `feature_selection.py`, before the `if len(keep) > keep_cap` block, compute the per-cell cap:

```python
    # Per-cell cap from events-per-variable. Falls back to the keep_cap arg
    # (which defaults to KEEP_CAP) when train_df is empty.
    n_train = len(target) if target is not None else 0
    keep_cap = keep_cap_for_market(n_train, base=keep_cap)

    # Layer 4: cap. Rank survivors by composite; SHAP-floored features get a
    # cap-survival boost so the heuristic-driven cap can't override the
    # model's own importance attribution. ...
    if len(keep) > keep_cap:
        ...
```

- [ ] **Step 3: Write a test**

Add to `tests/golden/test_feature_selection.py`:

```python
def test_keep_cap_scales_with_n_train():
    from sportstradamus.feature_selection import keep_cap_for_market, KEEP_CAP
    # Small n: floor at base
    assert keep_cap_for_market(500) == KEEP_CAP
    assert keep_cap_for_market(2500) == KEEP_CAP
    # Large n: scales up via divisor
    assert keep_cap_for_market(6500) == 130
    assert keep_cap_for_market(10000) == 200
    # Zero/negative: floor at base
    assert keep_cap_for_market(0) == KEEP_CAP
    assert keep_cap_for_market(-5) == KEEP_CAP
```

- [ ] **Step 4: Run tests**

```bash
poetry run pytest tests/golden/test_feature_selection.py -v
```

Expected: New `test_keep_cap_scales_with_n_train` passes. All other tests still pass. The existing `test_keep_cap_protects_shap_floored_features` should still pass because we ensure the SHAP floor stays under any cap.

- [ ] **Step 5: Regenerate NFL filter, audit cap sizes**

```bash
poetry run python - << 'PY'
import sys, pandas as pd
sys.path.insert(0, "src")
from sportstradamus.helpers import feature_filter
from sportstradamus.training.shap import filter_market
from sportstradamus.feature_selection import keep_cap_for_market

results = []
for m in sorted(feature_filter["NFL"]["Filtered"].keys()):
    diag = filter_market("NFL", m)
    n_train = diag.get("n_training", 0)
    cap = keep_cap_for_market(n_train)
    results.append((m, n_train, cap, diag["n_kept"]))

df = pd.DataFrame(results, columns=["market", "n_train", "cap", "n_kept"])
print(df.to_string(index=False))
PY
```

Expected: Cells with n_train > 3000 get caps > 60. `n_kept` matches `cap` for cap-bound cells.

- [ ] **Step 6: Retrain 5 cells (Optuna will see larger feature sets for high-sample cells)**

```bash
rm src/sportstradamus/data/models/NFL_passing-first-downs.mdl src/sportstradamus/data/models/NFL_receiving-tds.mdl src/sportstradamus/data/models/NFL_receptions.mdl src/sportstradamus/data/models/NFL_fantasy-points-prizepicks.mdl src/sportstradamus/data/models/NFL_fantasy-points-underdog.mdl
poetry run meditate --league NFL --bypass-withholding 2>&1 | tee /tmp/phase3_meditate.log
```

- [ ] **Step 7: Verify BSS still within ±0.02 of baseline**

Same verification script as Task 5 Step 1. The expectation is BSS holds steady or improves; if any cell regresses >0.01 vs Phase 2, the per-cell cap is over-feeding Optuna with low-signal features and we need to investigate (probably by lowering the divisor from 50 → 80).

- [ ] **Step 8: Commit (after refactoring-specialist)**

```bash
git add src/sportstradamus/feature_selection.py tests/golden/test_feature_selection.py src/sportstradamus/data/config/feature_filter.json src/sportstradamus/data/training/ src/sportstradamus/data/models/NFL_*.mdl
git commit -m "$(cat <<'EOF'
feat(fs): per-cell KEEP_CAP scaled by events-per-variable

Adds keep_cap_for_market(n_train) = max(60, n_train // 50) per Harrell
2015 EPV heuristic. Cells like carries (n=6,657) no longer share a cap
with cells at n=2,500 — high-n markets get proportionally more feature
capacity. The 60-feature floor preserves the historical default for
small-n cells.

BSS for the 5 regressed cells held steady (within ±0.01 of Phase 2)
on retrain; high-n cells outside the 5 are unverified but the SHAP
floor invariant from Phase 2 still holds (tested via
test_disk_filter_contains_all_shap_floored_features).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Task 11: Joint-SHAP-per-distribution-parameter audit (READ-ONLY)

The current implementation at `training/shap.py:44-47` sums `np.abs(sv)` across all distribution parameters of a LightGBMLSS model. For ZINB (loc, concentration, gate) and ZAGamma (loc, scale, gate), this could quietly under-weight the gate-head's important features by mixing them with loc/scale signal. The researcher flagged this; we should measure before changing.

**Files:**
- Create: `scripts/audit_joint_shap.py` (read-only diagnostic; deletable after the audit lands a verdict)
- Read: `src/sportstradamus/training/shap.py:28-53`

- [ ] **Step 1: Write the audit script**

Create `scripts/audit_joint_shap.py`:

```python
"""Compare per-parameter SHAP rankings vs joint-sum SHAP rankings for the
ZINB/ZAGamma NFL cells. Verdict goes into docs/archive/superpowers/research/.

Read-only — does not modify any production artifacts.

Usage:
    poetry run python scripts/audit_joint_shap.py
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import shap

LEAGUE = "NFL"
ZI_DISTS = {"ZINB", "ZAGamma"}


def _per_param_shap(booster, X: pd.DataFrame) -> dict[int, np.ndarray]:
    """Return {param_index: mean(|SHAP|) per feature} for a multi-output booster."""
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X)
    if not isinstance(sv, list):
        return {0: np.mean(np.abs(sv), axis=0)}
    return {i: np.mean(np.abs(s), axis=0) for i, s in enumerate(sv)}


def main() -> None:
    from importlib import resources as pkg_resources
    from sportstradamus.data import data as _data_pkg
    data_root = pkg_resources.files(_data_pkg)
    model_dir = data_root / "models"
    test_dir = data_root / "test_sets"

    for mdl in sorted(model_dir.iterdir()):
        if not mdl.name.startswith(f"{LEAGUE}_") or mdl.suffix != ".mdl":
            continue
        with open(mdl, "rb") as f:
            d = pickle.load(f)
        dist = d.get("distribution")
        if dist not in ZI_DISTS:
            continue
        market = mdl.name.replace(f"{LEAGUE}_", "").replace(".mdl", "")
        test_path = test_dir / f"{LEAGUE}_{market}.csv"
        if not test_path.is_file():
            continue
        test_df = pd.read_csv(test_path, index_col=0).drop(columns=["Result"], errors="ignore")
        for c in ("Home", "Player position"):
            if c in test_df.columns:
                test_df[c] = test_df[c].astype("category")
        per_param = _per_param_shap(d["model"].booster, test_df)
        feats = list(test_df.columns)
        joint = np.sum([np.abs(v) for v in per_param.values()], axis=0)

        n_params = len(per_param)
        param_names = {
            ("ZINB", 3): ("loc", "concentration", "gate"),
            ("ZAGamma", 3): ("loc", "scale", "gate"),
        }.get((dist, n_params), tuple(f"p{i}" for i in range(n_params)))

        out = pd.DataFrame({"feature": feats, "joint": joint})
        for i, name in enumerate(param_names):
            out[name] = per_param[i]
        out["joint_rank"] = (-out["joint"]).rank()
        if "gate" in out.columns:
            out["gate_rank"] = (-out["gate"]).rank()
            out["divergence"] = (out["joint_rank"] - out["gate_rank"]).abs()
            print(f"\n=== {LEAGUE} {market} ({dist}) ===")
            print("Top 10 features where gate-rank diverges most from joint-rank:")
            print(out.nlargest(10, "divergence")[["feature", "joint_rank", "gate_rank", "divergence"]].to_string(index=False))
        else:
            print(f"\n=== {LEAGUE} {market} ({dist}) — no gate head, skipping ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
poetry run python scripts/audit_joint_shap.py | tee /tmp/joint_shap_audit.log
```

Expected: For each ZINB/ZAGamma NFL cell, a table of top-10 features by `(joint_rank - gate_rank)` divergence. If any feature has a high gate-rank but low joint-rank (e.g., gate-rank 5, joint-rank 80), the joint sum is masking important gate-head signal.

- [ ] **Step 3: Decide based on the audit output**

- **If divergences are small (<10 ranks on average)** — joint sum is fine; no fix needed. Document the finding and close the open question.
- **If divergences are large** — open a follow-up PR to compute per-parameter SHAP and store separately, with the filter using either per-parameter top-K or a weighted sum that respects the gate-head's distinct purpose.

Write the verdict to `docs/archive/superpowers/research/2026-05-27-joint-shap-audit.md` (one paragraph describing the finding + decision). Stage that file. **Do not implement a code fix in this PR** — that's a separate decision.

- [ ] **Step 4: Commit the audit script + verdict**

```bash
git add scripts/audit_joint_shap.py docs/archive/superpowers/research/2026-05-27-joint-shap-audit.md
git commit -m "$(cat <<'EOF'
chore(fs): joint-SHAP-per-distribution-parameter audit for ZINB cells

Adds a diagnostic script that compares per-parameter |SHAP| rankings
against the joint-sum rankings the current scouting pass uses. Output
is a per-cell table of features where gate-rank diverges most from
joint-rank — flagging cases where the joint sum may be diluting
gate-head signal for ZINB / ZAGamma distributions.

Verdict at docs/archive/superpowers/research/2026-05-27-joint-shap-audit.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Evaluation Track (DEFERRED, gated on profiling)

These are bigger algorithmic moves recommended by the researcher. They are NOT part of this PR. Document them in `docs/archive/superpowers/research/2026-05-27-feature-filter-options.md` so the decision is captured, and let the human decide which (if any) to pick up after Phases 1-3 land.

### Option A — Adopt powershap (arXiv:2206.08394)

Pool-size-invariant null-distribution p-value cutoff. Replaces the manual-K, manual-weight problem. Requires `pip install powershap`. Effort estimate: 1-2 sessions for a single-cell evaluation; 1 session per league for rollout if the evaluation passes.

**Pre-PR validation:** Pick `NFL passing first downs`, run powershap, compare its selected filter against `filter_market_features` output. Train both, compare BSS. Ship only if powershap is within ±0.01 BSS AND the filter is more stable across rebuilds.

### Option B — Skip pre-filtering entirely

Akhiat & Touchanti 2024 (arXiv:2411.05937) — XGBoost shows no statistically significant accuracy difference between FS-filtered and no-FS models across 960 experiments. Gated on Optuna wall-clock: does training one NFL cell with 934 features (no filter) complete in under 90 minutes? If yes, this option eliminates the entire dilution/floor/cap problem.

**Pre-PR validation:** Time a single `meditate --rebuild-filter=false` run on `NFL passing first downs` with `feature_filter["NFL"]["Filtered"]["passing first downs"]` set to the full unfiltered candidate list. If wall-clock < 90 min and BSS within ±0.01 of Phase 3 result, this option is viable.

### Open methodology questions for the queue

- **Temporal stability via Spearman early/late half is weakest signal for non-stationary NFL data.** Switch to `TimeSeriesSplit` cross-validation, or drop the stability term entirely.
- **Scouting 70/30 split vs Optuna's tune/validation split independence.** Confirm they're not the same split (Cawley-Talbot 2010 leakage concern).
- **KEEP_CAP unenforcement bug** — `first downs` (208 features) and `completion percentage` (95) escaped the cap. After Phase 3 lands, audit how — they should hit the per-cell cap unless `n_train` for those cells genuinely supports that many features.

---

## Verification & quality gates

All Phase 1-3 work must pass before merge:

1. **BSS recovery** — all 5 regressed cells within ±0.02 BSS of `/tmp/baseline_pre_regen/model_stats.parquet`.
2. **No collateral regression** — no cell with an existing baseline regresses >0.01.
3. **Pinned tests** — `poetry run pytest tests/golden/test_feature_selection.py tests/golden/test_filter_consistency.py -v` all pass.
4. **Style + integration** — `poetry run ruff check src/sportstradamus/`, `poetry run pytest tests/golden/`, `poetry run pytest -m integration` all clean.
5. **refactoring-specialist** — invoked on every Python file touched per CLAUDE.md mandatory rule.
6. **CONTRIBUTING.md updated** — if any constant changed (REDUNDANCY_WEIGHT, KEEP_CAP) the per-cell logic is documented in the appropriate §.

## Critical files (reference index)

- [src/sportstradamus/feature_selection.py](src/sportstradamus/feature_selection.py) — selection constants, `filter_market_features`, `shap_floor_base_names`, new `keep_cap_for_market`
- [src/sportstradamus/training/shap.py](src/sportstradamus/training/shap.py) — `_scouting_shap_and_filter`, `compute_market_importance`, `_load_shap_corr_dfs`, `_compute_shap_and_corr` (joint-SHAP source)
- [src/sportstradamus/scripts/evaluate_model_features.py](src/sportstradamus/scripts/evaluate_model_features.py) — interactive script being refactored to delegate to `filter_market_features` in Task 7
- [src/sportstradamus/training/cli.py](src/sportstradamus/training/cli.py) — `--rebuild-filter`, `--bypass-withholding`, `--deterministic`, `--reset-markets` flags used throughout
- [src/sportstradamus/data/config/feature_filter.json](src/sportstradamus/data/config/feature_filter.json) — the broken filter being regenerated in Task 2
- [src/sportstradamus/data/training/feature_importances.csv](src/sportstradamus/data/training/feature_importances.csv) — scouting SHAP per cell (already correct; Phase 1 reuses it without re-scouting)
- [tests/golden/test_feature_selection.py](tests/golden/test_feature_selection.py) — 8 existing tests, extended in Tasks 7, 9, 10
- [tests/golden/test_filter_consistency.py](tests/golden/test_filter_consistency.py) — new in Task 8
- `/tmp/feature_filter_remediation_phase1_backup/` — Phase 1 rollback state
- `/tmp/baseline_pre_regen/model_stats.parquet` — BSS recovery target
- `/tmp/researcher_feature_filtering.md` — full research brief (547 lines)
