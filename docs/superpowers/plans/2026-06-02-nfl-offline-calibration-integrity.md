# NFL Offline Win-Banking + Gate Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bank the honest offline NFL g1 wins available now (affine-ROE mean correction), and make the model/book blend and the g1 gate honest enough to justify the follow-on Phase 2 g1-loosening — without deferring any cell.

**Architecture:** Five sequenced changes to the training/prediction pipeline. (1) Wire the already-built `MEAN_STAGE` post-hoc corrector (`roe_mean`) into `train_market` at the decode→fuse seam and into `model_prob` inference. (2) Add a behavior-preserving `blending` strategy seam (registry + per-cell config field, default `nll`). (3) Build convolution-derived honest books for the no-market QB combined cells. (4) Persist `Player`/`Date` and add a player-clustered g1 recheck. (5) Roll out per-cell with BSS + supersede guardrails. Immediate breadth comes from change (1); changes (2)–(4) are integrity/extensibility groundwork for Phase 2.

**Tech Stack:** Python 3.11, LightGBMLSS, scipy/numpy/pandas, DuckDB archive, pytest. Spec: [`docs/superpowers/specs/2026-06-02-nfl-offline-calibration-integrity-design.md`](../specs/2026-06-02-nfl-offline-calibration-integrity-design.md).

**Worktree:** Execute in a dedicated worktree off `model-research` (the production-tracking branch is `devel`; `model-research` is where this work lives). No production retrain happens until Task 8.

---

## File map

| File | Responsibility | Change |
|---|---|---|
| `src/sportstradamus/training/pipeline.py` | training orchestration | wire MEAN_STAGE at decode→fuse; corrected EV persist; call blending registry |
| `src/sportstradamus/prediction/model_prob.py` | inference | apply mean-stage posthoc before `fused_loc` |
| `src/sportstradamus/training/calibration.py` | blend-weight fit | extract `fit_model_weight` into a `blending` registry; default `nll` |
| `src/sportstradamus/training/ship_config.py` | per-cell config validation | validate new `blending` field |
| `src/sportstradamus/training/scorecard.py` | offline gates | clustered g1 bootstrap; keep `Player`/`Date` |
| `src/sportstradamus/helpers/combined_markets.py` | **new** — convolution-derived books | `qb-yards`/`qb-tds` synthetic book from components |
| `src/sportstradamus/data/config/stat_meta.json` | per-cell config | set `posthoc: roe_mean`, add `blending` defaults |
| `tests/golden/test_posthoc_mean_stage.py` | **new** | MEAN_STAGE wiring + inference parity |
| `tests/golden/test_blending_registry.py` | **new** | behavior-preservation + validation |
| `tests/golden/test_combined_markets.py` | **new** | convolution correctness |
| `tests/golden/test_gate1_clustered.py` | **new** | clustered bootstrap |

---

## Task 1: Wire `roe_mean` (MEAN_STAGE) into `train_market`

The corrector exists in `posthoc.py` (`MEAN_STAGE = {"roe_mean","isotonic_mean"}`, `fit_posthoc`/`apply_posthoc` fully implemented). Only the `PROB_STAGE` branch is wired in `train_market` (pipeline.py:2226–2235). This task adds the `MEAN_STAGE` branch at the decode→fuse seam so the correction flows into the blend → `P` → g1/g5. `decoded` carries both `ev` (test μ) and `ev_validation`.

**Files:**
- Modify: `src/sportstradamus/training/pipeline.py:2200-2201` (insert between `_step_decode_predictions` and `_step_fuse_predictions`)
- Test: `tests/golden/test_posthoc_mean_stage.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/golden/test_posthoc_mean_stage.py
import numpy as np
from sportstradamus.training import posthoc


def test_roe_mean_corrects_compression_on_validation():
    # Compressed model: predicted mean is shrunk toward the grand mean (slope < 1
    # vs the truth), exactly the leaf-averaging failure roe_mean targets.
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 4, size=500)
    grand = truth.mean()
    compressed_mu = grand + 0.6 * (truth - grand)  # slope 0.6 => compressed
    blob = posthoc.fit_posthoc("roe_mean", compressed_mu, truth)
    assert blob is not None and blob["kind"] == "affine"
    corrected = posthoc.apply_posthoc("roe_mean", blob, compressed_mu)
    # The affine fit should restore the decompressed slope: corrected tracks truth
    # with slope ~1.
    slope = np.polyfit(corrected, truth, 1)[0]
    assert 0.9 < slope < 1.1
    # MEAN_STAGE output is clipped non-negative.
    assert (corrected >= 0).all()
```

- [ ] **Step 2: Run test to verify it passes already (corrector exists)**

Run: `poetry run pytest tests/golden/test_posthoc_mean_stage.py::test_roe_mean_corrects_compression_on_validation -v`
Expected: PASS — this confirms the dormant corrector works. (It is the contract the wiring must preserve.)

- [ ] **Step 3: Add the MEAN_STAGE wiring in `train_market`**

In `src/sportstradamus/training/pipeline.py`, the current seam reads:

```python
    decoded = _step_decode_predictions(
        prob_params,
        preds["prob_params_validation"],
        splits["X_test"],
        splits["X_validation"],
        dist,
        dist_info["target_normalization"],
        dist_info["global_mean"],
        dist_info["denom_col"],
        hist_gate,
    )
    fused = _step_fuse_predictions(decoded, splits, dist, cv, hist_gate)
```

Insert the mean-stage correction between them:

```python
    decoded = _step_decode_predictions(
        prob_params,
        preds["prob_params_validation"],
        splits["X_test"],
        splits["X_validation"],
        dist,
        dist_info["target_normalization"],
        dist_info["global_mean"],
        dist_info["denom_col"],
        hist_gate,
    )

    # Mean-stage post-hoc (orthogonal to target_normalization and to the
    # prob-stage corrector below): fit on the validation decoded mean, then
    # correct both test and validation means BEFORE fusion so the correction
    # flows through the blend into P (Gate 1/5) and into the persisted EV
    # (Gate 2/3). roe_mean undoes leaf-averaging compression; it deliberately
    # does not touch dispersion (Gate 4).
    mean_posthoc_blob = None
    if posthoc_slug in posthoc.MEAN_STAGE:
        val_result = splits["y_validation"]["Result"].to_numpy(dtype=float)
        mean_posthoc_blob = posthoc.fit_posthoc(
            posthoc_slug, decoded["ev_validation"], val_result
        )
        decoded["ev"] = posthoc.apply_posthoc(posthoc_slug, mean_posthoc_blob, decoded["ev"])
        decoded["ev_validation"] = posthoc.apply_posthoc(
            posthoc_slug, mean_posthoc_blob, decoded["ev_validation"]
        )

    fused = _step_fuse_predictions(decoded, splits, dist, cv, hist_gate)
```

- [ ] **Step 4: Make `posthoc_blob` carry the mean blob and persist the corrected count EV**

The pickle persists one `posthoc_blob` (pipeline.py:1280-1281, 2281). A cell carries one `posthoc` slug, so reuse that single blob key. Replace the prob-stage-only blob init (pipeline.py:2225) so the mean blob is the persisted blob when the slug is mean-stage:

```python
    posthoc_blob = mean_posthoc_blob  # None unless the slug is a MEAN_STAGE corrector
    if posthoc_slug in posthoc.PROB_STAGE:
        posthoc_blob = posthoc.fit_posthoc(posthoc_slug, val_calibrated, y_class_val)
        val_calibrated = posthoc.apply_posthoc(posthoc_slug, posthoc_blob, val_calibrated)
```

For count cells, `_step_persist_artifacts` recomputes `base_ev` from raw params (pipeline.py:1326, 1333), which would discard the mean correction in the persisted `EV` column (Gate 2/3 read it). For SkewNormal the persist already uses `decoded["ev"]` (line 1319), so it is already corrected. Make the count branch use the corrected decoded mean when a mean-stage correction was applied. In `_step_persist_artifacts`, add a parameter and use it:

```python
def _step_persist_artifacts(
    *,
    filedict: dict,
    splits: dict,
    prob_params: pd.DataFrame,
    decoded: dict,
    weighted_mean: np.ndarray,
    y_proba_filt: np.ndarray,
    dist: str,
    hist_gate: float,
    filename: str,
    deterministic: bool,
    target_normalization: str,
    zinb_mode: str,
    mean_corrected: bool = False,
) -> None:
```

and in the count branch replace the base-mean assignment:

```python
    elif dist in ("NegBin", "ZINB"):
        base_ev = prob_params["total_count"] * prob_params["probs"] / (1 - prob_params["probs"])
        # A mean-stage corrector already adjusted decoded["ev"]; persist that so the
        # bias gates (EV column) see the same corrected mean the blend used. The
        # native R / NB_P below stay uncorrected — Gate 4 reads them and measures
        # dispersion, which roe_mean intentionally leaves alone.
        X_test["EV"] = decoded["ev"] if mean_corrected else base_ev
        if dist == "ZINB":
            X_test["Gate"] = prob_params["gate"]
        X_test["R"] = prob_params["total_count"]
        X_test["NB_P"] = prob_params["probs"]
```

Pass the flag at the call site (pipeline.py:2286-2299):

```python
    _step_persist_artifacts(
        filedict=filedict,
        splits=splits,
        prob_params=prob_params,
        decoded=decoded,
        weighted_mean=fused["weighted_mean"],
        y_proba_filt=y_proba_filt,
        dist=dist,
        hist_gate=hist_gate,
        filename=filename,
        deterministic=deterministic,
        target_normalization=target_normalization,
        zinb_mode=zinb_mode,
        mean_corrected=posthoc_slug in posthoc.MEAN_STAGE,
    )
```

- [ ] **Step 5: Add a wiring regression test**

```python
# tests/golden/test_posthoc_mean_stage.py (append)
def test_mean_stage_is_noop_when_slug_is_none():
    # apply with blob=None is identity; the train_market guard only fits when the
    # slug is in MEAN_STAGE, so a "none"/prob-stage cell never touches decoded ev.
    import numpy as np
    from sportstradamus.training import posthoc
    mu = np.array([1.0, 2.0, 3.0])
    assert np.allclose(posthoc.apply_posthoc("none", None, mu), mu)
    assert "roe_mean" in posthoc.MEAN_STAGE
    assert "roe_mean" not in posthoc.PROB_STAGE
```

- [ ] **Step 6: Run the golden + the determinism/integration smoke**

Run: `poetry run pytest tests/golden/test_posthoc_mean_stage.py -v`
Expected: PASS (both tests).
Run: `poetry run ruff check src/sportstradamus/training/pipeline.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/sportstradamus/training/pipeline.py tests/golden/test_posthoc_mean_stage.py
git commit -m "feat(training): wire MEAN_STAGE posthoc (roe_mean) into train_market"
```

---

## Task 2: Apply mean-stage posthoc at inference (`model_prob`)

`model_prob` loads `posthoc`/`posthoc_blob` (model_prob.py:218-219) and applies only the prob-stage corrector (`_apply_prob_posthoc`, line 125, called at 520). Mirror it for the mean stage, applied to the decoded model μ **before** `fused_loc` (model_prob.py:365-368) so live inference matches the training-side test CSV event-for-event.

**Files:**
- Modify: `src/sportstradamus/prediction/model_prob.py` (import, helper, call before `fused_loc`)
- Test: `tests/golden/test_posthoc_mean_stage.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/golden/test_posthoc_mean_stage.py (append)
def test_apply_mean_posthoc_helper_matches_posthoc_module():
    import numpy as np
    from sportstradamus.prediction.model_prob import _apply_mean_posthoc
    from sportstradamus.training import posthoc
    mu = np.array([0.5, 1.5, 2.5, 3.5])
    blob = {"kind": "affine", "a": 0.2, "b": 1.3}
    got = _apply_mean_posthoc(mu, "roe_mean", blob)
    want = posthoc.apply_posthoc("roe_mean", blob, mu)
    assert np.allclose(got, want)
    # prob-stage slug must be a no-op on the mean path
    assert np.allclose(_apply_mean_posthoc(mu, "prob_recal_platt", blob), mu)
    # legacy pickle: no blob => identity
    assert np.allclose(_apply_mean_posthoc(mu, "none", None), mu)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_posthoc_mean_stage.py::test_apply_mean_posthoc_helper_matches_posthoc_module -v`
Expected: FAIL with `ImportError: cannot import name '_apply_mean_posthoc'`.

- [ ] **Step 3: Implement the helper and call it before `fused_loc`**

In `src/sportstradamus/prediction/model_prob.py`, extend the posthoc import (line 34):

```python
from sportstradamus.training.posthoc import MEAN_STAGE, PROB_STAGE, apply_posthoc
```

Add the helper next to `_apply_prob_posthoc` (after line 135):

```python
def _apply_mean_posthoc(
    model_mu: np.ndarray, posthoc_slug: str, posthoc_blob: dict | None
) -> np.ndarray:
    """Apply a fitted mean-stage corrector to the decoded model mean.

    No-op unless the cell's slug is a :data:`MEAN_STAGE` corrector. Mirrors the
    training-side correction in ``pipeline.train_market`` so the live blend sees
    the same corrected mean.
    """
    if posthoc_slug in MEAN_STAGE:
        return apply_posthoc(posthoc_slug, posthoc_blob, model_mu)
    return model_mu
```

Apply it to `offer_df["Model EV"]` once, immediately after the shape-ceiling clamp (model_prob.py:363) and before the `# Blend model and book distributions via fused_loc` dispatch (line 365). Every dist branch passes `offer_df["Model EV"]` to `fused_loc` as the model-EV argument (e.g. SkewNormal at line 370), so a single correction here covers all of them. `posthoc_slug` / `posthoc_blob` are already loaded from the pickle at model_prob.py:218-219 and in scope here:

```python
        # Mean-stage post-hoc correction of the model EV before blending, mirroring
        # pipeline.train_market so live predictions match the offline test CSV
        # event-for-event.
        offer_df["Model EV"] = _apply_mean_posthoc(
            offer_df["Model EV"].to_numpy(), posthoc_slug, posthoc_blob
        )

        # Blend model and book distributions via fused_loc
        if dist == "SkewNormal":
            ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/golden/test_posthoc_mean_stage.py::test_apply_mean_posthoc_helper_matches_posthoc_module -v`
Expected: PASS.

- [ ] **Step 5: Run the prediction integration smoke**

Run: `poetry run pytest -m integration -k model_prob -v` (fake-mode; if no such test exists, run the full `poetry run pytest -m integration`)
Expected: PASS — legacy pickles (no mean blob) still load and predict.

- [ ] **Step 6: Commit**

```bash
git add src/sportstradamus/prediction/model_prob.py tests/golden/test_posthoc_mean_stage.py
git commit -m "feat(prediction): apply mean-stage posthoc before fused_loc at inference"
```

---

## Task 3: `blending` strategy seam (behavior-preserving)

Extract `fit_model_weight` (calibration.py:178) into a registry keyed by a `blending` slug, register `nll` as the sole default that reproduces today's behavior (objective + `[0.05, 0.9]` bounds), and validate the new `blending` field in `ship_config`. No new objective and no floor change here — that is the future blending-research session.

**Files:**
- Modify: `src/sportstradamus/training/calibration.py` (registry + `BLENDING_SLUGS`, `DEFAULT_BLENDING`)
- Modify: `src/sportstradamus/training/ship_config.py:95-127` (validate `blending`)
- Test: `tests/golden/test_blending_registry.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/golden/test_blending_registry.py
import numpy as np
from sportstradamus.training import calibration


def test_nll_is_default_and_registered():
    assert calibration.DEFAULT_BLENDING == "nll"
    assert "nll" in calibration.BLENDING_SLUGS


def test_nll_strategy_reproduces_fit_model_weight():
    # The registry's nll entry must equal the legacy fit_model_weight output.
    rng = np.random.default_rng(1)
    n = 300
    model_ev = rng.uniform(0.5, 3.0, n)
    odds_ev = model_ev + rng.normal(0, 0.3, n)
    result = np.maximum(0, np.round(rng.normal(model_ev, 1.0)))
    legacy = calibration.fit_model_weight(
        model_ev, odds_ev, result, "NegBin", model_r=np.full(n, 5.0), cv=0.5
    )
    via_registry = calibration.fit_blend_weight(
        "nll", model_ev, odds_ev, result, "NegBin", model_r=np.full(n, 5.0), cv=0.5
    )
    assert abs(legacy - via_registry) < 1e-9


def test_unknown_blending_slug_rejected():
    import pytest
    from sportstradamus.training import ship_config
    with pytest.raises(ValueError, match="unknown blending"):
        ship_config._validate_cell(
            "NFL", "passing-tds",
            {"shipped": "withheld", "dist": "ZINB",
             "target_normalization": "none", "posthoc": "none", "blending": "bogus"},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_blending_registry.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'DEFAULT_BLENDING'` / `fit_blend_weight`.

- [ ] **Step 3: Add the registry to `calibration.py`**

The existing `fit_model_weight` stays (the `nll` implementation). Add a thin registry above it that owns the slug→(weight-fitter, bounds) mapping:

```python
# Per-cell blend strategy: how the model and book distributions are combined.
# Each entry owns its weight-fitting objective and its weight bounds, so a future
# strategy can change the objective (e.g. Brier-at-line) and/or the bounds (e.g.
# drop the 0.05 floor) without touching the others. Default `nll` reproduces the
# historical behavior exactly. New strategies are added in the blending-research
# session; this seam ships behavior-preserving.
DEFAULT_BLENDING: str = "nll"
BLENDING_SLUGS: frozenset[str] = frozenset({"nll"})


def fit_blend_weight(blending: str, *args, **kwargs) -> float:
    """Dispatch to the blend strategy's weight fitter. ``nll`` is the legacy
    full-distribution-likelihood objective in :func:`fit_model_weight`."""
    if blending not in BLENDING_SLUGS:
        raise ValueError(f"Unknown blending slug {blending!r}; valid: {sorted(BLENDING_SLUGS)}")
    return fit_model_weight(*args, **kwargs)
```

- [ ] **Step 4: Route the pipeline call through the registry**

In `_step_fuse_predictions` (pipeline.py:1713+), the two `fit_model_weight(...)` calls (lines 1769, 1816) become `fit_blend_weight(blending, ...)`. Thread the cell's `blending` slug into `_step_fuse_predictions` (add a `blending: str = calibration.DEFAULT_BLENDING` parameter; pass `dist_info["blending"]` from `train_market`). Because only `nll` is registered, output is unchanged.

```python
# at the two call sites inside _step_fuse_predictions:
        model_weight = calibration.fit_blend_weight(
            blending,
            ...  # identical positional/keyword args as the existing fit_model_weight call
        )
```

- [ ] **Step 5: Validate `blending` in `ship_config._validate_cell`**

Add to `src/sportstradamus/training/ship_config.py` imports:

```python
from sportstradamus.training.calibration import BLENDING_SLUGS, DEFAULT_BLENDING
```

and inside `_validate_cell` (after the `posthoc` check at line 116), mirror the posthoc validation:

```python
    blending = cell.get("blending", DEFAULT_BLENDING)
    if blending not in BLENDING_SLUGS:
        raise ValueError(
            f"stat_meta.json: cell {league}/{market} has unknown blending "
            f"value {blending!r}; valid: {sorted(BLENDING_SLUGS)}"
        )
```

`load_ship_config` resolves a missing field to `DEFAULT_BLENDING`, so all 41 shipped cells (which lack the field) stay on `nll` unchanged.

- [ ] **Step 6: Run tests**

Run: `poetry run pytest tests/golden/test_blending_registry.py -v`
Expected: PASS (3 tests).
Run: `poetry run pytest tests/golden/ -k ship_config -v`
Expected: PASS — existing config validation unaffected.

- [ ] **Step 7: Commit**

```bash
git add src/sportstradamus/training/calibration.py src/sportstradamus/training/ship_config.py src/sportstradamus/training/pipeline.py tests/golden/test_blending_registry.py
git commit -m "feat(training): add behavior-preserving blending strategy seam (default nll)"
```

---

## Task 4: Persist `Player` + `Date`; keep them through `load_test_set`

The clustered g1 recheck (Task 5) needs a player key and game date per test row. `Date` survives in `M`; `Player` is dropped from the dumped test set. Persist both, and make `load_test_set` keep them as optional columns.

**Files:**
- Modify: `src/sportstradamus/training/pipeline.py:1309-1339` (`_step_persist_artifacts`)
- Modify: `src/sportstradamus/training/scorecard.py:219-247` (`load_test_set` optional columns)
- Test: `tests/golden/test_gate1_clustered.py` (create; schema assertion here, bootstrap in Task 5)

- [ ] **Step 1: Write the failing test**

```python
# tests/golden/test_gate1_clustered.py
import pandas as pd
from sportstradamus.training.scorecard import load_test_set, DEFAULT_PRED_COL


def test_load_test_set_keeps_player_and_date_when_present(tmp_path):
    csv = tmp_path / "NFL_passing-tds.csv"
    pd.DataFrame(
        {
            "MeanYr": [1.0, 2.0],
            "Result": [1, 0],
            "EV": [1.1, 1.9],
            "P": [0.6, 0.4],
            "Odds": [0.5, 0.5],
            "Line": [1.5, 1.5],
            "Player": ["A", "B"],
            "Date": ["2025-09-07", "2025-09-07"],
        }
    ).to_csv(csv, index=False)
    df = load_test_set(csv, DEFAULT_PRED_COL)
    assert "Player" in df.columns and "Date" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_gate1_clustered.py::test_load_test_set_keeps_player_and_date_when_present -v`
Expected: FAIL — `load_test_set` drops `Player`/`Date` (not in the kept set).

- [ ] **Step 3: Persist `Player`/`Date` in `_step_persist_artifacts`**

**3a. Thread the player key + game date into the splits dict.** In `_step_build_splits`, `X_test.index` indexes into `M`, which holds `Player` and `Date` (pipeline.py:551). Add both to the returned dict (pipeline.py:711-724), guarding `Player` since some leagues omit it:

```python
    players_test = M.loc[X_test.index, "Player"].values if "Player" in M.columns else None
    dates_test = M.loc[X_test.index, "Date"].values
    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "X_validation": X_validation,
        "y_train": y_train,
        "y_test": y_test,
        "y_validation": y_validation,
        "B_train": B_train,
        "B_test": B_test,
        "B_validation": B_validation,
        "y_train_labels": y_train_labels,
        "players_test": players_test,
        "dates_test": dates_test,
    }
```

**3b. Persist them.** After `X_test["P"] = y_proba_filt[:, 1]` (pipeline.py:1339), add:

```python
    # Persist the player key + game date so the offline scorecard can run a
    # player-clustered Gate-1 bootstrap (the i.i.d. bootstrap over-credits
    # repeated-player panels by 18-48%).
    if splits.get("players_test") is not None:
        X_test["Player"] = splits["players_test"]
    X_test["Date"] = splits["dates_test"]
```

- [ ] **Step 4: Keep `Player`/`Date` in `load_test_set`**

In `scorecard.py:244`, extend the opportunistic optional set:

```python
    optional = {"P", "Odds", "Line", "Player", "Date"} & set(df.columns)
```

and ensure the column-keep logic downstream retains them (they are non-required, so they must be added to whatever projection `load_test_set` returns — keep the existing required ∪ optional ∪ dist-param selection).

- [ ] **Step 5: Run test to verify it passes**

Run: `poetry run pytest tests/golden/test_gate1_clustered.py::test_load_test_set_keeps_player_and_date_when_present -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sportstradamus/training/pipeline.py src/sportstradamus/training/scorecard.py tests/golden/test_gate1_clustered.py
git commit -m "feat(training): persist Player/Date on test set for clustered Gate-1"
```

---

## Task 5: Player-clustered Gate-1 bootstrap

Add a block-bootstrap variant of the paired-Brier CI that resamples whole players (preserving within-player correlation), mirroring `_bootstrap_mean_ci` (scorecard.py:453). Compute it in `compute_gates` next to the i.i.d. g1 and expose `g1_clustered_ci_hi`. It is an integrity recheck used at promotion (Task 8): only makes passing harder.

**Files:**
- Modify: `src/sportstradamus/training/scorecard.py` (add `_bootstrap_mean_ci_clustered`, `_gate1_brier_ci_clustered`, wire into `compute_gates`)
- Test: `tests/golden/test_gate1_clustered.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/golden/test_gate1_clustered.py (append)
import numpy as np
from sportstradamus.training.scorecard import _bootstrap_mean_ci_clustered, _bootstrap_mean_ci


def test_clustered_ci_is_wider_on_correlated_panel():
    # 40 players x 25 games; the per-event statistic is identical within a player
    # (max within-player correlation). The clustered bootstrap must yield a wider
    # CI than the i.i.d. one, which ignores the correlation and over-credits.
    rng = np.random.default_rng(7)
    player_means = rng.normal(0.0, 0.02, size=40)
    players = np.repeat(np.arange(40), 25)
    values = np.repeat(player_means, 25)  # constant within player
    _, lo_i, hi_i = _bootstrap_mean_ci(values, np.random.default_rng(1))
    _, lo_c, hi_c = _bootstrap_mean_ci_clustered(
        values, players, np.random.default_rng(1)
    )
    assert (hi_c - lo_c) > (hi_i - lo_i)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_gate1_clustered.py::test_clustered_ci_is_wider_on_correlated_panel -v`
Expected: FAIL — `ImportError: cannot import name '_bootstrap_mean_ci_clustered'`.

- [ ] **Step 3: Implement the clustered bootstrap**

Add to `scorecard.py` after `_bootstrap_mean_ci` (line 473):

```python
def _bootstrap_mean_ci_clustered(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = _GATE1_N_BOOT,
) -> tuple[float, float, float]:
    """Cluster (player) block bootstrap of the mean of ``values``.

    Resamples whole clusters with replacement, so within-cluster correlation is
    preserved and the CI is not anti-conservative on repeated-player panels.
    Returns ``(mean, ci_lo, ci_hi)``; ``(nan, nan, nan)`` if empty.
    """
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    finite = np.isfinite(values)
    values, cluster_ids = values[finite], cluster_ids[finite]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(cluster_ids)
    groups = [values[cluster_ids == c] for c in uniq]
    n_clusters = len(uniq)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        draws[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(draws, [_CI_LOW_PCT, _CI_HIGH_PCT])
    return float(values.mean()), float(lo), float(hi)


def _gate1_brier_ci_clustered(
    p_model: np.ndarray,
    p_book: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Gate 1 paired Brier CI under a player-clustered bootstrap."""
    d = (p_model - y) ** 2 - (p_book - y) ** 2
    return _bootstrap_mean_ci_clustered(d, cluster_ids, rng)
```

- [ ] **Step 4: Wire it into `compute_gates`**

In `compute_gates`, where the i.i.d. g1 is computed (scorecard.py:891-902), add the clustered recheck when a player key is present. `_brier_inputs(df)` already aligns to finite rows; pass the matching `Player` column:

```python
    g1_clustered_hi = None
    if brier_in is not None and "Player" in df.columns:
        # Align Player to the same finite-row mask _brier_inputs used.
        sub = df.dropna(subset=["P", "Odds", "Line"])
        _, _, g1_clustered_hi = _gate1_brier_ci_clustered(
            p_model_b, p_book, y_b, sub["Player"].to_numpy(),
            np.random.default_rng(_GATE1_SEED),
        )
```

Add `"g1_clustered_ci_hi": r(g1_clustered_hi)` to the returned gate dict (next to the existing `g1_*` keys). `r(...)` is the existing 4-dp rounder used for the other gate fields. The i.i.d. `g1_pass` verdict is unchanged; `g1_clustered_ci_hi` is informational in the parquet and enforced at promotion in Task 8.

- [ ] **Step 5: Run tests**

Run: `poetry run pytest tests/golden/test_gate1_clustered.py -v`
Expected: PASS (schema + clustered-width tests).
Run: `poetry run pytest tests/golden/ -k scorecard -v`
Expected: PASS — existing gate tests unaffected (new column is additive).

- [ ] **Step 6: Commit**

```bash
git add src/sportstradamus/training/scorecard.py tests/golden/test_gate1_clustered.py
git commit -m "feat(scorecard): add player-clustered Gate-1 bootstrap recheck"
```

---

## Task 6: Convolution-derived book helpers

New module computing an honest combined-market book over/under probability from the component markets: `qb-yards = passing-yards + rushing-yards` (Normal sum, correlation from `NFL_corr.parquet`); `qb-tds = passing-tds + rushing-tds` (count PMF convolution). Pure functions, no I/O — archive sourcing happens in the Task 8 injection script.

**Files:**
- Create: `src/sportstradamus/helpers/combined_markets.py`
- Test: `tests/golden/test_combined_markets.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/golden/test_combined_markets.py
import numpy as np
from sportstradamus.helpers.combined_markets import (
    normal_sum_over_prob,
    count_sum_over_prob,
)


def test_normal_sum_mean_and_monotonicity():
    # Two component books N(220, 40^2) and N(25, 20^2), independent (rho=0).
    # Combined mean 245, sd sqrt(40^2+20^2). P(over line) decreases as the line rises.
    p_low = normal_sum_over_prob(line=200.0, mu1=220, sd1=40, mu2=25, sd2=20, rho=0.0)
    p_mid = normal_sum_over_prob(line=245.0, mu1=220, sd1=40, mu2=25, sd2=20, rho=0.0)
    p_high = normal_sum_over_prob(line=290.0, mu1=220, sd1=40, mu2=25, sd2=20, rho=0.0)
    assert p_low > p_mid > p_high
    assert abs(p_mid - 0.5) < 1e-6  # line at the combined mean => 0.5


def test_negative_rho_tightens_variance():
    # Game-script substitution: negative rho shrinks combined variance, so an
    # over line above the mean is LESS likely (mass pulled toward the mean).
    p_indep = normal_sum_over_prob(line=300.0, mu1=220, sd1=40, mu2=25, sd2=20, rho=0.0)
    p_negcorr = normal_sum_over_prob(line=300.0, mu1=220, sd1=40, mu2=25, sd2=20, rho=-0.4)
    assert p_negcorr < p_indep


def test_count_sum_matches_independent_convolution():
    # Sum of two small Poisson-like count books; P(over 1.5) = P(total >= 2).
    p = count_sum_over_prob(line=1.5, pmf1=np.array([0.5, 0.3, 0.2]), pmf2=np.array([0.6, 0.4]))
    # total pmf via direct convolution
    conv = np.convolve([0.5, 0.3, 0.2], [0.6, 0.4])
    expected = conv[2:].sum()  # P(total >= 2)
    assert abs(p - expected) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_combined_markets.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the helpers**

```python
# src/sportstradamus/helpers/combined_markets.py
"""Honest book references for combined QB markets with no direct sportsbook line.

``qb-yards = passing-yards + rushing-yards`` and ``qb-tds = passing-tds +
rushing-tds`` are offered on DFS sites but not quoted by sportsbooks, so the
archive carries a fabricated ``p_book = 0.5`` placeholder. These helpers build an
honest combined-market over/under probability by convolving the sharp component
books — a Normal sum for the (continuous) yardage market and a discrete PMF
convolution for the (count) TD market. The pass/rush correlation comes from the
per-league correlation matrix; it is negative (game-script substitution), so an
independence assumption overstates the combined variance.
"""
import numpy as np
from scipy.stats import norm


def normal_sum_over_prob(
    line: float, mu1: float, sd1: float, mu2: float, sd2: float, rho: float
) -> float:
    """P(X1 + X2 > line) for jointly-Normal components with correlation ``rho``."""
    mu = mu1 + mu2
    var = sd1**2 + sd2**2 + 2.0 * rho * sd1 * sd2
    sd = np.sqrt(max(var, 1e-12))
    return float(norm.sf(line, loc=mu, scale=sd))


def count_sum_over_prob(line: float, pmf1: np.ndarray, pmf2: np.ndarray) -> float:
    """P(N1 + N2 >= ceil(line)) for independent count components given their PMFs.

    The line for a count prop sits on a half-integer (e.g. 1.5), so "over" is
    ``total >= ceil(line)``. Independence is the documented approximation for the
    TD convolution; game-script dependence is second-order at these low counts.
    """
    total = np.convolve(np.asarray(pmf1, dtype=float), np.asarray(pmf2, dtype=float))
    threshold = int(np.ceil(line))
    return float(total[threshold:].sum())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/golden/test_combined_markets.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/helpers/combined_markets.py tests/golden/test_combined_markets.py
git commit -m "feat(helpers): convolution-derived book for combined QB markets"
```

---

## Task 7: `derived_book_under_prob_row` adapter (injection script deferred to Task 8)

**Re-scoped 2026-06-02 during execution.** The original Task 7 also built an offline `inject_combined_book.py` script. Verifying the data sources showed the script cannot be built or validated as specified yet: (a) the correlation matrix ships as `NFL_corr.parquet` / per-league `corr_same_team.parquet`, not `NFL_corr.csv`; (b) the component test sets (`NFL_passing-yards.csv`, etc.) and the qb target CSVs carry no `Player`/`Date` join key — those columns only appear after a retrain (Task 4's persist), so the `(player, game_date)` join is impossible until the qb cells are retrained; (c) the CSVs store only `Line`/`Odds` plus the model's own params, not the book's `mu`/`sd`/`pmf`, so the script must source component prices from the archive and impose an implied-component-book model — a modeling choice best made against real data. Per the user decision, Task 7 ships ONLY the pure, fully-testable adapter; the archive-sourcing injection script is built and validated in Task 8 (after the qb retrains), where its coverage and implied-book model can be checked against the freshly-persisted `Player`/`Date`.

This task adds one row-level adapter mapping a per-offer dict (component `pass`/`rush` book params) to the archive's `Odds` convention (book under-probability; book over = `1 − Odds`), returning `None` when a component book is missing so the future caller skips rather than fabricates.

**Files:**
- Modify: `src/sportstradamus/helpers/combined_markets.py` (add `derived_book_under_prob_row`)
- Test: `tests/golden/test_combined_markets.py` (append adapter tests)

- [ ] **Step 1: Write the failing test (coverage + convention)**

```python
# tests/golden/test_combined_markets.py (append)
from sportstradamus.helpers.combined_markets import derived_book_under_prob_row


def test_derived_book_returns_none_when_component_missing():
    # No rushing component available => cannot synthesize; must return None so the
    # caller skips the row rather than fabricating.
    row = {"line": 250.0, "pass": {"mu": 220, "sd": 40}, "rush": None}
    assert derived_book_under_prob_row(row, market="qb-yards", rho=-0.3) is None


def test_derived_book_under_prob_convention():
    # Odds stores book UNDER-probability; book over = 1 - Odds. Combined mean 245,
    # line 245 => over 0.5 => under 0.5.
    row = {"line": 245.0, "pass": {"mu": 220, "sd": 40}, "rush": {"mu": 25, "sd": 20}}
    under = derived_book_under_prob_row(row, market="qb-yards", rho=0.0)
    assert abs(under - 0.5) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_combined_markets.py -k derived_book -v`
Expected: FAIL — `derived_book_under_prob_row` not defined.

- [ ] **Step 3: Add the row-level adapter to `combined_markets.py`**

```python
def derived_book_under_prob_row(row: dict, market: str, rho: float) -> float | None:
    """Return the archive `Odds` (book UNDER-probability) for one combined-market
    offer, or ``None`` when a component book is missing (caller skips the row).

    ``row`` carries the offer ``line`` and the two component book params:
    ``pass``/``rush`` dicts with Normal ``mu``/``sd`` for ``qb-yards`` or count
    ``pmf`` arrays for ``qb-tds``.
    """
    comp1, comp2 = row.get("pass"), row.get("rush")
    if comp1 is None or comp2 is None:
        return None
    if market == "qb-yards":
        over = normal_sum_over_prob(
            row["line"], comp1["mu"], comp1["sd"], comp2["mu"], comp2["sd"], rho
        )
    elif market == "qb-tds":
        over = count_sum_over_prob(row["line"], comp1["pmf"], comp2["pmf"])
    else:
        raise ValueError(f"derived book undefined for market {market!r}")
    return float(1.0 - over)
```

- [ ] **Step 4: Run the adapter tests + commit**

Run: `poetry run pytest tests/golden/test_combined_markets.py -v`
Expected: PASS (Task-6 convolution tests + the new adapter tests, including a `qb-tds` count-dispatch case).

```bash
git add src/sportstradamus/helpers/combined_markets.py tests/golden/test_combined_markets.py
git commit -m "feat(helpers): add derived_book_under_prob_row adapter for combined QB markets"
```

**Deferred to Task 8 (the injection script).** Building `inject_combined_book.py` requires the qb test sets to carry `Player`/`Date` (only present after the Task-4 retrain) and an implied-component-book model sourced from the archive (the CSVs store no book `mu`/`sd`/`pmf`). Both are best decided against real data, so the script is written and validated in Task 8 Step 6, after `qb-tds` is retrained.

---

## Task 8: Per-cell rollout with guardrails

Apply the levers to NFL cells, retrain, re-score, and promote only the cells that pass all five gates **and** the clustered recheck. `passing-tds` first (highest confidence); then `qb-tds` (after Task 7's honest book), `interceptions`, `carries`.

**Files:**
- Modify: `src/sportstradamus/data/config/stat_meta.json` (per-cell `posthoc` / `blending`)
- No test file — this is an operational task gated on the existing harnesses.

- [ ] **Step 1: Set `passing-tds` to mean-stage correction**

Edit `stat_meta.json` NFL `passing-tds`: set `"posthoc": "roe_mean"` (leave `blending` absent → `nll`). Confirm `ship_config.load_ship_config("devel")` loads without error:

Run: `poetry run python -c "from sportstradamus.training.ship_config import load_ship_config; load_ship_config('devel'); print('ok')"`
Expected: `ok`.

- [ ] **Step 2: Retrain the cell**

Run: `poetry run meditate --league NFL --market passing-tds --force` (use the project's exact per-cell retrain flags; confirm against `poetry run meditate --help`).
Expected: writes `data/models/NFL_passing-tds.mdl`, `data/test_sets/NFL_passing-tds.csv`, and the cell's row in `data/training/model_stats.parquet`.

- [ ] **Step 3: Re-score and read the gates + clustered recheck**

```bash
poetry run python -c "
import pandas as pd
df = pd.read_parquet('src/sportstradamus/data/training/model_stats.parquet')
r = df[(df.league=='NFL') & (df.market=='passing-tds')].iloc[0]
print('ship', bool(r['ship']), 'g1_ci_hi', r['g1_brier_diff_ci_hi'],
      'g1_clustered_ci_hi', r.get('g1_clustered_ci_hi'),
      'brier_skill', r['brier_skill_score'])
"
```
Expected: `g1_brier_diff_ci_hi < 0` AND `g1_clustered_ci_hi < 0` AND `brier_skill_score` not dropped > 0.01 vs the pre-change baseline (BSS guardrail). If the clustered recheck does **not** clear, do **not** promote — record it as a fragile/false pass.

- [ ] **Step 4: Promote if and only if both gates clear**

If Step 3 passes: edit `stat_meta.json` NFL `passing-tds` `"shipped": "withheld" → "devel"`. Re-run `load_ship_config` to confirm validity.

- [ ] **Step 5: Run the full quality gates**

```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/
poetry run pytest -m integration
```
Expected: all clean.

- [ ] **Step 6: Repeat Steps 1–5 for `qb-tds`, `interceptions`, `carries`**

- For `qb-tds`: it needs the honest convolution book before scoring, and that requires building the deferred injection script. Order: (a) retrain `qb-tds` FIRST so its test set carries `Player`/`Date`; (b) write `src/sportstradamus/scripts/inject_combined_book.py` (a `click` CLI mirroring `inject_backfilled_odds.py`) — source the `passing-tds` + `rushing-tds` component prices from the archive per `(Player, Date)`, impose an implied-component-book count model, build each component PMF, call `derived_book_under_prob_row` per offer, write the derived under-prob into the test set's `Odds` column where both components are present, skip rows missing a component, and report coverage under `--dry-run`; (c) run `--dry-run`, validate coverage against the real retrained test set, then apply; (d) set `posthoc: roe_mean`, re-score. Expect it may stay withheld (the honest book is sharp). `qb-yards` (Normal-sum) follows the same pattern only if time permits; `passing-first-downs` has no clean decomposition and stays on its current book.
- For `interceptions`: low prior (μ ~uncorrelated with INTs); apply `roe_mean`, re-score, promote only if both g1 gates clear, else leave withheld.
- For `carries`: apply `roe_mean`, re-score, promote only if both gates clear.

- [ ] **Step 7: Commit the config + record outcomes**

```bash
git add src/sportstradamus/data/config/stat_meta.json
git commit -m "feat(ship): promote NFL cells passing g1 under mean-stage correction"
```

Update [`docs/operation_ship_75.md`](../../operation_ship_75.md) Current-state table + lifecycle rows with the new NFL count and a one-line note per cell (shipped / stayed-withheld + reason). Commit the doc separately.

---

## Final verification

- [ ] `poetry run ruff check src/sportstradamus/` — clean
- [ ] `poetry run pytest tests/golden/` — clean (incl. the 4 new test files)
- [ ] `poetry run pytest -m integration` — clean (fake-mode)
- [ ] Determinism gate green (`poetry run pytest tests/integration/test_determinism_gate.py`)
- [ ] `refactoring-specialist` run on every touched `.py` before any push (CLAUDE.md hard rule)
- [ ] `docs/operation_ship_75.md` updated with the new NFL count + per-cell outcomes

## Self-review notes (author)

- **Spec coverage:** Component 1 → Tasks 1–2; Component 2 → Task 3; Component 3 → Tasks 6–7; Component 4 → Tasks 4–5; Component 5 → Task 8. All covered.
- **Behavior preservation:** Task 3 `nll` reproduces `fit_model_weight` (asserted); missing `blending`/`posthoc` fields resolve to defaults, so the 41 shipped cells are untouched until Task 8 edits a cell.
- **Type consistency:** `fit_blend_weight(blending, *args, **kwargs)` forwards to `fit_model_weight`; `_apply_mean_posthoc(mu, slug, blob)` mirrors `_apply_prob_posthoc(cal_over, slug, blob)`; `_gate1_brier_ci_clustered` mirrors `_gate1_brier_ci` with an added `cluster_ids` arg.
- **Out of scope (do not implement here):** `brier_line` blend strategy, dropping the `0.05` floor, DFS-gate machinery, Phase-2 g1 loosening.
