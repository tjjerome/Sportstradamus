# Gate 6 Outcome-Directional Legs + Anchor Hysteresis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework Gate 6 (anti-shrinkage) from one recent-form leg into three OR-ed one-sided legs — recent-form (with a hysteresis anchor), CITL-under vs the realized outcome (all cells), and a guarded count/ZINB over-leg — then re-score every cell under the new gates plus the `ratio_projvol` normalization and update the handoff docs.

**Architecture:** All gate logic stays in `src/sportstradamus/training/scorecard.py` as pure functions of one test-set CSV; the only new input is an optional `prior_g6_fired` boolean (for anchor hysteresis) that `training/report.py` derives from the previous `model_stats.parquet` row and threads in. The authoritative ship decision is `apply_thresholds`; `min_gate_slack` is the ranking signal for the board sweep.

**Tech Stack:** Python 3.11, numpy, pandas, pytest (golden suite via xdist), poetry. Design spec: `docs/superpowers/specs/2026-06-21-gate6-outcome-directional-legs-design.md`. Research: `/tmp/researcher_gate6_scope_and_drift.md`.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/sportstradamus/training/scorecard.py` | All Gate-6 logic | Constants; `_gate6_anchored`; replace `_gate6_star_ratio` with `_gate6_legs`; rewrite `_gate6_passes`; `recent_form_fired` helper; `gate_row`/`compute_gates` gain `prior_g6_fired`; `apply_thresholds` + `min_gate_slack` read the new legs |
| `src/sportstradamus/training/report.py` | Production scorecard writer | Read prior `model_stats.parquet` g6 verdict, thread `prior_g6_fired` into `compute_gates` |
| `tests/golden/test_scorecard.py` | Gate unit tests | Update the 3 existing g6 tests; add CITL-under, over-leg, hysteresis tests |
| `docs/ship_gate.md` | Gate reference | OR-of-three semantics, new constants, hysteresis |
| `docs/handoffs/model_improvement_track.md` | Lane handoff | Research-verdict block, ledger entry, post-CV results |

The recent-form leg's math, the basketball/NFL `star_ref`, the stable band, and `_bootstrap_ratio_ci_clustered` are **unchanged** — reused.

---

## Task 1: Gate-6 constants

**Files:**
- Modify: `src/sportstradamus/training/scorecard.py:190-200` (the `_GATE6_*` constants block)
- Test: `tests/golden/test_scorecard.py`

- [ ] **Step 1: Write the failing test**

```python
def test_gate6_hysteresis_and_over_constants_exist():
    from sportstradamus.training.scorecard import (
        _GATE6_FIRE_ON,
        _GATE6_KEEP_ON,
        _GATE6_OVER_MIN_MEAN,
    )
    # Deadband: a fresh cell starts judging at FIRE_ON, a flagged cell keeps down to KEEP_ON.
    assert _GATE6_FIRE_ON == 0.58
    assert _GATE6_KEEP_ON == 0.52
    assert _GATE6_KEEP_ON < _GATE6_FIRE_ON
    # Over-leg degenerate-count guard: only test a count bench whose realized mean clears 1.
    assert _GATE6_OVER_MIN_MEAN == 1.0
```

- [ ] **Step 2: Run it, verify ImportError**

Run: `poetry run pytest tests/golden/test_scorecard.py::test_gate6_hysteresis_and_over_constants_exist -v`
Expected: FAIL — `ImportError: cannot import name '_GATE6_FIRE_ON'`.

- [ ] **Step 3: Add the constants, remove the retired one**

In the `_GATE6_*` block, replace the `_GATE6_MIN_RECENT_CORR` definition (the anchor) with the deadband pair, and add the over-leg guard. Find:

```python
# corr(Mean10, Result) anchor: below it recent form isn't a valid yardstick and the gate
# can't fire — exempts MIN (corr ~0.3-0.5) and the bursty counts, retains FGA/PR/PRA (~0.6).
_GATE6_MIN_RECENT_CORR: float = 0.55
```

Replace with:

```python
# corr(Mean10, Result) anchor with hysteresis: recent form is a valid yardstick only above it
# (exempts MIN ~0.3-0.5 and bursty counts, retains FGA/PR/PRA ~0.6). Two thresholds form a
# deadband so a near-0.55 cell can't flip ship state on a retrain wobble: a fresh cell starts
# being judged at FIRE_ON; a cell whose recent-form leg fired last run keeps being judged down
# to KEEP_ON. Below KEEP_ON (or no prior fire) the recent-form leg is exempt.
_GATE6_FIRE_ON: float = 0.58
_GATE6_KEEP_ON: float = 0.52
# Over-leg guard: count/ZINB bench over-prediction is only testable where the stable bench's
# realized mean clears this — below it Σ Result → 0 and the ratio is discreteness, not a defect.
_GATE6_OVER_MIN_MEAN: float = 1.0
```

- [ ] **Step 4: Run it, verify pass** — `poetry run pytest tests/golden/test_scorecard.py::test_gate6_hysteresis_and_over_constants_exist -v` → PASS. (Other tests still reference `_GATE6_MIN_RECENT_CORR` and will break — fixed in Task 8; that is expected here.)

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/training/scorecard.py tests/golden/test_scorecard.py
git commit -m "feat(scorecard): Gate-6 hysteresis + over-leg constants"
```

---

## Task 2: `_gate6_anchored` — the hysteresis decision

**Files:**
- Modify: `src/sportstradamus/training/scorecard.py` (add above `_gate6_star_ratio`, ~line 1346)
- Test: `tests/golden/test_scorecard.py`

- [ ] **Step 1: Write the failing test**

```python
def test_gate6_anchored_hysteresis():
    from sportstradamus.training.scorecard import _gate6_anchored
    # Above fire-on: always judged, regardless of prior.
    assert _gate6_anchored(0.60, None) is True
    assert _gate6_anchored(0.58, False) is True
    # In the deadband: judged only if it fired last run.
    assert _gate6_anchored(0.56, True) is True
    assert _gate6_anchored(0.56, None) is False
    assert _gate6_anchored(0.56, False) is False
    # Below keep-on: never judged.
    assert _gate6_anchored(0.51, True) is False
    # Non-finite corr (degenerate cell): not anchored.
    import numpy as np
    assert _gate6_anchored(np.nan, True) is False
```

- [ ] **Step 2: Run it, verify ImportError** — FAIL: `cannot import name '_gate6_anchored'`.

- [ ] **Step 3: Implement**

```python
def _gate6_anchored(corr: float, prior_g6_fired: bool | None) -> bool:
    """Gate-6 recent-form anchor with hysteresis. A fresh cell starts being judged at
    ``_GATE6_FIRE_ON``; a cell whose recent-form leg fired on the prior run keeps being judged
    down to ``_GATE6_KEEP_ON``. Outside that the recent-form leg is exempt — ``Mean10`` is not a
    valid yardstick where recent form doesn't predict the outcome (the corr-anchor rationale).
    """
    if not np.isfinite(corr):
        return False
    if corr >= _GATE6_FIRE_ON:
        return True
    return corr >= _GATE6_KEEP_ON and bool(prior_g6_fired)
```

- [ ] **Step 4: Run it, verify pass** — PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(scorecard): Gate-6 anchor hysteresis helper"
```

---

## Task 3: `_gate6_legs` — the three-leg statistic (replaces `_gate6_star_ratio`)

**Files:**
- Modify: `src/sportstradamus/training/scorecard.py:1346-1385` (replace `_gate6_star_ratio`)
- Test: `tests/golden/test_scorecard.py`

The existing `_cohort_frame` helper (top of the Gate-6 test block) builds a stable-star SkewNormal
frame whose `Blended_EV = shrink × Mean10` and (when `anchored=True`) ties `Result` to `Mean10`.
The new legs need `Result` controllable independently of `Mean10`; extend the helper and add a
count-frame helper.

- [ ] **Step 1: Write the failing tests**

```python
def _legs_frame(n=200, seed=11, *, ev_over_form=0.8, result_over_form=1.0, anchored=True):
    """Stable-star frame: Blended_EV = ev_over_form × Mean10, Result = result_over_form × Mean10
    plus noise. `anchored` ties Result's *correlation* to Mean10 (set False for a MIN-like cell).
    """
    rng = np.random.default_rng(seed)
    meanyr = rng.uniform(2.0, 20.0, n)
    mean10 = meanyr * (1.0 + rng.uniform(-0.05, 0.05, n))  # inside the ±0.12 stable band
    base = mean10 if anchored else rng.uniform(2.0, 20.0, n)
    return pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Mean10": mean10,
            "Result": result_over_form * base + rng.normal(0.0, 0.3, n),
            "Blended_EV": ev_over_form * mean10,
            "Player": [f"P{i % 40}" for i in range(n)],
            "SN_Loc": np.zeros(n), "SN_Scale": np.ones(n), "SN_Alpha": np.zeros(n),
        }
    )


def _count_legs_frame(n=200, seed=7, *, bench_ev_over_result=1.3, bench_mean=2.0):
    """Count/ZINB frame (R/NB_P/Gate cols ⇒ _infer_dist == 'ZINB') whose stable BENCH (low MeanYr)
    is over-predicted vs a non-degenerate realized mean ≥ 1.
    """
    rng = np.random.default_rng(seed)
    meanyr = np.concatenate([rng.uniform(0.5, 3.0, n // 2), rng.uniform(8.0, 20.0, n // 2)])
    mean10 = meanyr * (1.0 + rng.uniform(-0.05, 0.05, n))
    bench = meanyr <= np.quantile(meanyr, 0.25)
    result = np.where(bench, bench_mean, mean10) + rng.normal(0.0, 0.2, n)
    ev = np.where(bench, bench_ev_over_result * bench_mean, mean10)
    return pd.DataFrame(
        {
            "MeanYr": meanyr, "Mean10": mean10, "Result": result, "Blended_EV": ev,
            "Player": [f"P{i % 40}" for i in range(n)],
            "R": np.ones(n), "NB_P": np.full(n, 0.5), "Gate": np.zeros(n),
        }
    )


def test_gate6_legs_recent_form_fires_when_anchored():
    from sportstradamus.training.scorecard import _gate6_legs, _GATE6_MARGIN
    g6 = _gate6_legs(_legs_frame(ev_over_form=0.8), "Blended_EV", league="WNBA", prior_g6_fired=None)
    assert g6["g6_recent_corr"] >= 0.58  # anchored (fire-on)
    assert g6["g6_star_ci_hi"] is not None
    assert g6["g6_star_ci_hi"] < g6["g6_star_ref"] - _GATE6_MARGIN  # over-shrinks vs recent form


def test_gate6_legs_recent_form_exempt_below_keep_on():
    from sportstradamus.training.scorecard import _gate6_legs
    # Unanchored (Result independent of Mean10) ⇒ low corr ⇒ recent-form leg blank, but CITL still runs.
    g6 = _gate6_legs(_legs_frame(ev_over_form=0.8, anchored=False), "Blended_EV", league="WNBA",
                     prior_g6_fired=None)
    assert g6["g6_star_ci_hi"] is None  # recent-form exempt
    assert g6["g6_citl_ci_hi"] is not None  # CITL is not anchored


def test_gate6_legs_citl_under_fires_against_outcome_on_any_cell():
    from sportstradamus.training.scorecard import _gate6_legs, _GATE6_MARGIN
    # EV 0.85× the realized outcome, low corr (unanchored): recent-form is blank, CITL catches it.
    g6 = _gate6_legs(_legs_frame(ev_over_form=0.85, result_over_form=1.0, anchored=False),
                     "Blended_EV", league="NBA", prior_g6_fired=None)
    assert g6["g6_star_ci_hi"] is None
    assert g6["g6_citl_ci_hi"] < 1.0 - _GATE6_MARGIN


def test_gate6_legs_citl_silent_on_legitimate_regression():
    from sportstradamus.training.scorecard import _gate6_legs, _GATE6_MARGIN
    # EV below recent form but ABOVE the realized outcome (outcome confirms the regression).
    g6 = _gate6_legs(_legs_frame(ev_over_form=0.8, result_over_form=0.74, anchored=False),
                     "Blended_EV", league="NBA", prior_g6_fired=None)
    assert g6["g6_citl_ci_hi"] >= 1.0 - _GATE6_MARGIN  # does not fire


def test_gate6_legs_over_leg_fires_on_count_bench():
    from sportstradamus.training.scorecard import _gate6_legs, _GATE6_MARGIN
    g6 = _gate6_legs(_count_legs_frame(bench_ev_over_result=1.3), "Blended_EV", league="WNBA",
                     prior_g6_fired=None)
    assert g6["g6_over_ci_lo"] is not None
    assert g6["g6_over_ci_lo"] > 1.0 + _GATE6_MARGIN


def test_gate6_legs_over_leg_silent_under_degenerate_guard():
    from sportstradamus.training.scorecard import _gate6_legs
    # Same over-prediction but bench realized mean < 1 ⇒ guard blanks the over-leg.
    g6 = _gate6_legs(_count_legs_frame(bench_ev_over_result=1.3, bench_mean=0.4), "Blended_EV",
                     league="WNBA", prior_g6_fired=None)
    assert g6["g6_over_ci_lo"] is None


def test_gate6_legs_over_leg_blank_for_non_count_family():
    from sportstradamus.training.scorecard import _gate6_legs
    g6 = _gate6_legs(_legs_frame(), "Blended_EV", league="WNBA", prior_g6_fired=None)  # SkewNormal
    assert g6["g6_over_ci_lo"] is None
```

- [ ] **Step 2: Run them, verify they FAIL** — `cannot import name '_gate6_legs'`.

- [ ] **Step 3: Replace `_gate6_star_ratio` with `_gate6_legs`**

Delete the whole `_gate6_star_ratio` function (currently ~1346-1385) and add:

```python
_GATE6_BLANK_LEGS: dict[str, float | None] = {
    "g6_recent_corr": None,
    "g6_star_ratio": None,
    "g6_star_ci_hi": None,
    "g6_star_ref": None,
    "g6_citl_ratio": None,
    "g6_citl_ci_hi": None,
    "g6_over_ratio": None,
    "g6_over_ci_lo": None,
}


def _gate6_legs(
    df: pd.DataFrame,
    pred_col: str,
    *,
    league: str,
    prior_g6_fired: bool | None = None,
) -> dict[str, float | None]:
    """Gate 6 (anti-shrinkage): three one-sided legs on the stable top/bottom-MeanYr segments,
    each reading the served ``pred_col`` (so all are normalization-agnostic):

    * **recent-form** (star vs ``Mean10``): the original leg, gated by the corr anchor with
      hysteresis — catches the ``ratio_meanyr`` holdout-corruption class CITL is blind to.
    * **CITL-under** (star vs ``Result``): calibration-in-the-large, run on *every* cell (the
      outcome is a valid yardstick without the anchor) — catches outcome under-shrinkage that
      Gate 2's σ-normalization launders.
    * **over** (bench vs ``Result``): count/ZINB bench over-prediction, guarded by a realized
      segment-mean floor so it can't fire on degenerate rare counts.

    Returns the ``g6_*`` measurement subset; a leg's keys are ``None`` where it can't or
    shouldn't test (the gate auto-passes a blank leg).
    """
    if not {"Mean10", "Player"}.issubset(df.columns):
        return dict(_GATE6_BLANK_LEGS)
    work = df[(df[DECILE_COL] > 0) & (df["Mean10"] > 0)]
    meanyr = work[DECILE_COL].to_numpy()
    mean10 = work["Mean10"].to_numpy()
    result = work[ACTUAL_COL].to_numpy()
    pred = work[pred_col].to_numpy()
    players = work["Player"].to_numpy()
    rng = np.random.default_rng(_GATE1_SEED)
    out = dict(_GATE6_BLANK_LEGS)

    corr = _corr(mean10, result)
    out["g6_recent_corr"] = corr if np.isfinite(corr) else None

    stable = np.abs(mean10 / meanyr - 1.0) <= _GATE6_STABLE_BAND
    star = stable & (meanyr >= np.quantile(meanyr, 1.0 - BOTTOM_QUARTILE_FRAC))
    if int(star.sum()) >= _GATE6_MIN_STAR_ROWS:
        citl, _, citl_hi = _bootstrap_ratio_ci_clustered(
            pred[star], result[star], players[star], rng
        )
        out["g6_citl_ratio"], out["g6_citl_ci_hi"] = citl, citl_hi
        if _gate6_anchored(corr, prior_g6_fired):
            ratio, _, ratio_hi = _bootstrap_ratio_ci_clustered(
                pred[star], mean10[star], players[star], rng
            )
            out["g6_star_ratio"], out["g6_star_ci_hi"] = ratio, ratio_hi
            out["g6_star_ref"] = (
                _GATE6_STAR_REF_NFL if league == "NFL" else _GATE6_STAR_REF_BASKETBALL
            )

    if _infer_dist_from_columns(df) in ("NegBin", "ZINB"):
        bench = stable & (meanyr <= np.quantile(meanyr, BOTTOM_QUARTILE_FRAC))
        if int(bench.sum()) >= _GATE6_MIN_STAR_ROWS and result[bench].mean() >= _GATE6_OVER_MIN_MEAN:
            over, over_lo, _ = _bootstrap_ratio_ci_clustered(
                pred[bench], result[bench], players[bench], rng
            )
            out["g6_over_ratio"], out["g6_over_ci_lo"] = over, over_lo
    return out
```

- [ ] **Step 4: Run the Task-3 tests, verify they PASS** — `poetry run pytest tests/golden/test_scorecard.py -k "gate6_legs" -v`. (Tests calling the old `_gate6_star_ratio` will break — fixed in Task 8.)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(scorecard): Gate-6 three-leg statistic (recent-form + CITL-under + over)"
```

---

## Task 4: `_gate6_passes` (OR of three legs) + `recent_form_fired` helper

**Files:**
- Modify: `src/sportstradamus/training/scorecard.py:1582-1587` (`_gate6_passes`)
- Test: `tests/golden/test_scorecard.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_gate6_passes_ors_three_legs():
    from sportstradamus.training.scorecard import _gate6_passes
    ok = {"g6_star_ci_hi": None, "g6_star_ref": None, "g6_citl_ci_hi": None, "g6_over_ci_lo": None}
    assert _gate6_passes(ok) is True  # all blank ⇒ auto-pass
    assert _gate6_passes({**ok, "g6_star_ci_hi": 0.80, "g6_star_ref": 0.95}) is False  # recent-form
    assert _gate6_passes({**ok, "g6_citl_ci_hi": 0.90}) is False  # CITL-under (< 1 - 0.03)
    assert _gate6_passes({**ok, "g6_citl_ci_hi": 0.99}) is True
    assert _gate6_passes({**ok, "g6_over_ci_lo": 1.10}) is False  # over (> 1 + 0.03)
    assert _gate6_passes({**ok, "g6_over_ci_lo": 1.00}) is True


def test_recent_form_fired_reads_prior_row():
    from sportstradamus.training.scorecard import recent_form_fired
    assert recent_form_fired({"g6_star_ci_hi": 0.80, "g6_star_ref": 0.95}) is True
    assert recent_form_fired({"g6_star_ci_hi": 0.99, "g6_star_ref": 0.95}) is False
    assert recent_form_fired({"g6_star_ci_hi": None, "g6_star_ref": None}) is False
```

- [ ] **Step 2: Run, verify FAIL** (`_gate6_passes` currently takes `(hi, ref)`; `recent_form_fired` missing).

- [ ] **Step 3: Rewrite `_gate6_passes`, add `recent_form_fired`**

Replace `_gate6_passes`:

```python
def _gate6_passes(row: Mapping[str, object]) -> bool:
    """Gate-6 ship test: pass iff none of the three one-sided legs fires. Each leg auto-passes
    when blank (not applicable / untestable) — the deliberate blank-is-pass, unlike g2-g5.

    * recent-form: ``star_ci_hi`` (CI UB of Σpred/ΣMean10) at/above ``star_ref − margin``.
    * CITL-under: ``citl_ci_hi`` (UB of Σpred/ΣResult) at/above ``1 − margin``.
    * over: ``over_ci_lo`` (LB of bench Σpred/ΣResult) at/below ``1 + margin``.
    """
    star_hi, star_ref = row.get("g6_star_ci_hi"), row.get("g6_star_ref")
    citl_hi = row.get("g6_citl_ci_hi")
    over_lo = row.get("g6_over_ci_lo")
    recent_ok = star_hi is None or star_ref is None or star_hi >= star_ref - _GATE6_MARGIN
    citl_ok = citl_hi is None or citl_hi >= 1.0 - _GATE6_MARGIN
    over_ok = over_lo is None or over_lo <= 1.0 + _GATE6_MARGIN
    return recent_ok and citl_ok and over_ok


def recent_form_fired(row: Mapping[str, object]) -> bool:
    """Whether the recent-form leg flagged this cell on the row's run. Seeds Gate-6 hysteresis
    from the prior ``model_stats`` row using ``g6_star_ci_hi``/``g6_star_ref`` (columns that
    predate this rework, so the first run after it lands is still seeded — cold-start safe).
    """
    hi, ref = row.get("g6_star_ci_hi"), row.get("g6_star_ref")
    return hi is not None and ref is not None and hi < ref - _GATE6_MARGIN
```

Ensure `from collections.abc import Mapping` is imported at the top of the file (add if absent).

- [ ] **Step 4: Run, verify PASS** — `poetry run pytest tests/golden/test_scorecard.py -k "gate6_passes or recent_form_fired" -v`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(scorecard): Gate-6 OR-of-three pass test + prior-fire seed"
```

---

## Task 5: Wire `gate_row`, `apply_thresholds`, `min_gate_slack`, `compute_gates`

**Files:**
- Modify: `src/sportstradamus/training/scorecard.py` — `gate_row` (1388, 1501-1553), `apply_thresholds` (1620), `min_gate_slack` (1654-1662), `compute_gates` (1672)
- Test: `tests/golden/test_scorecard.py`

- [ ] **Step 1: Write the failing test (end-to-end through `gate_row`/`apply_thresholds`)**

```python
def test_gate6_citl_fails_ship_through_gate_row():
    # A low-corr cell Gate 2 would pass but CITL catches: built so only g6 fails.
    df = _legs_frame(ev_over_form=0.85, result_over_form=1.0, anchored=False, n=400)
    row = apply_thresholds(
        gate_row(df, "Blended_EV", league="NBA", market="fantasy-points", strategy="x")
    )
    assert row["g6_star_ci_hi"] is None  # recent-form exempt (low corr)
    assert row["g6_citl_ci_hi"] is not None
    assert row["g6_pass"] is False
    assert row["ship"] is False


def test_gate_row_threads_prior_g6_fired():
    # A deadband cell (corr in [0.52, 0.58)) is judged iff prior fired.
    df = _legs_frame(ev_over_form=0.8, n=400, seed=3)  # tune seed so corr lands in the band
    judged = gate_row(df, "Blended_EV", league="WNBA", market="x", strategy="x", prior_g6_fired=True)
    exempt = gate_row(df, "Blended_EV", league="WNBA", market="x", strategy="x", prior_g6_fired=None)
    # If the frame's corr is in [0.52, 0.58): judged has a recent-form leg, exempt does not.
    if 0.52 <= judged["g6_recent_corr"] < 0.58:
        assert judged["g6_star_ci_hi"] is not None
        assert exempt["g6_star_ci_hi"] is None
```

- [ ] **Step 2: Run, verify FAIL** (`gate_row` has no `prior_g6_fired`; `apply_thresholds`/`min_gate_slack` read old keys).

- [ ] **Step 3: Implement the wiring**

In `gate_row` signature add the kwarg:

```python
def gate_row(
    df: pd.DataFrame,
    pred_col: str,
    *,
    league: str,
    market: str,
    strategy: str,
    decode_strategy: str | None = None,
    prior_g6_fired: bool | None = None,
) -> dict[str, object]:
```

Replace the Gate-6 block + the four `g6_*` lines in the return dict (1501-1504 and 1549-1552). Delete:

```python
    g6_corr, g6_ratio, g6_hi, g6_ref = _gate6_star_ratio(df, pred_col, league=league)
```

and the four `"g6_recent_corr"…"g6_star_ref"` entries in the literal. In their place, after computing `r = _round_gate_value`, spread the legs dict into the return:

```python
    g6 = _gate6_legs(df, pred_col, league=league, prior_g6_fired=prior_g6_fired)
    r = _round_gate_value
    return {
        "league": league,
        # ... all existing g1-g5 entries unchanged ...
        "g5_ece_debiased_oracle": r(g5_ece_db_o),
        **{k: r(v) for k, v in g6.items()},
    }
```

In `apply_thresholds`, replace the g6 line:

```python
    g6_pass = _gate6_passes(out)
```

In `min_gate_slack`, replace the g6 term (the `g6_hi`/`g6_ref` lines and the final tuple entry) with all three legs, each `+inf` when blank (auto-pass never binds), correctly signed:

```python
    star_hi, star_ref = row.get("g6_star_ci_hi"), row.get("g6_star_ref")
    citl_hi = row.get("g6_citl_ci_hi")
    over_lo = row.get("g6_over_ci_lo")
    return min(
        np.inf if hi is None else (_GATE1_NONINF_MARGIN - hi) / _GATE1_NONINF_MARGIN,
        _normalized_gate_slack(row.get("g2_star_z"), _GATE2_STAR_Z_MAX),
        _normalized_gate_slack(row.get("g3_bench_z"), _GATE3_BENCH_Z_MAX),
        -np.inf if g4 is None or g4_max is None else (g4_max - g4) / g4_max,
        _normalized_gate_slack(row.get("g5_ece_debiased", row.get("g5_ece")), _GATE5_ECE_MAX),
        # recent-form / CITL are lower-bounded (value must clear a floor); over is upper-bounded.
        np.inf if star_hi is None or star_ref is None
        else (star_hi - (star_ref - _GATE6_MARGIN)) / (star_ref - _GATE6_MARGIN),
        np.inf if citl_hi is None else (citl_hi - (1.0 - _GATE6_MARGIN)) / (1.0 - _GATE6_MARGIN),
        np.inf if over_lo is None
        else _normalized_gate_slack(over_lo, 1.0 + _GATE6_MARGIN),
    )
```

> Note: this corrects a latent sign inversion in the prior recent-form slack term (it used
> `_normalized_gate_slack(hi, ref − margin)`, positive when *failing*, contradicting the
> "positive ⇒ passing headroom" docstring). Ranking-only — `apply_thresholds` was always correct.

In `compute_gates` add the kwarg and forward it:

```python
def compute_gates(
    test_set_df: pd.DataFrame,
    *,
    league: str,
    market: str,
    strategy: str = "meditate",
    pred_col: str = DEFAULT_PRED_COL,
    prior_g6_fired: bool | None = None,
) -> dict[str, object]:
```

and in its body, where it calls `gate_row(...)`, pass `prior_g6_fired=prior_g6_fired`.

- [ ] **Step 4: Run, verify PASS** — the Task-5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(scorecard): wire Gate-6 legs through gate_row/apply_thresholds/min_gate_slack/compute_gates"
```

---

## Task 6: Thread `prior_g6_fired` from `report.py`

**Files:**
- Modify: `src/sportstradamus/training/report.py` — `_layer_gates_from_test_set` (249-266) + its caller loop
- Test: covered by the integration suite (no new unit test; the behavior is pure I/O wiring exercised end-to-end)

- [ ] **Step 1: Add a prior-verdict loader and thread it**

Near the other helpers in `report.py`, add (it reuses `MODEL_STATS_PATH`, already imported, and `recent_form_fired` from scorecard):

```python
def _load_prior_g6_fired() -> dict[tuple[str, str], bool]:
    """Map ``(league, market) -> recent-form leg fired last run``, read once from the existing
    ``model_stats.parquet`` before this run overwrites it. Seeds Gate-6 anchor hysteresis; empty
    on the first ever run (every cell then uses the fire-on threshold).
    """
    if not MODEL_STATS_PATH.is_file():
        return {}
    prior = pd.read_parquet(MODEL_STATS_PATH)
    return {
        (str(row["league"]), str(row["market"])): recent_form_fired(row)
        for row in prior.to_dict("records")
    }
```

Import `recent_form_fired` from `sportstradamus.training.scorecard` alongside `compute_gates` (line 45).

Add the kwarg to `_layer_gates_from_test_set` and forward it:

```python
def _layer_gates_from_test_set(
    row: dict, league: str, market: str, *, prior_g6_fired: bool | None = None
) -> None:
    ...
    row.update(
        compute_gates(
            df, league=league, market=market, pred_col=_SHIP_PRED_COL,
            prior_g6_fired=prior_g6_fired,
        )
    )
```

- [ ] **Step 2: Thread the map at the single call site**

Run `grep -n "_layer_gates_from_test_set" src/sportstradamus/training/report.py` to find the caller loop. Before the loop, add `prior_g6 = _load_prior_g6_fired()`, and pass `prior_g6_fired=prior_g6.get((league, market))` into the call.

- [ ] **Step 3: Run the integration suite** — `poetry run pytest -m integration -n0 -q`. Expected: PASS (the meditate integration path now reads the prior parquet and threads the bool; the fake-mode run writes a fresh parquet as before).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat(report): seed Gate-6 anchor hysteresis from prior model_stats"
```

---

## Task 7: Update the existing Gate-6 golden tests

**Files:**
- Modify: `tests/golden/test_scorecard.py` — the 3 existing g6 tests (594-700 region) + imports (lines 12-55)

- [ ] **Step 1: Fix the imports**

Remove `_GATE6_MIN_RECENT_CORR` from the import block; remove `_gate6_star_ratio`; add `_gate6_legs` and `recent_form_fired`. `_GATE6_MARGIN`, `_GATE6_STAR_REF_BASKETBALL`, `_gate6_anchored`, `_GATE6_FIRE_ON`/`_KEEP_ON`/`_OVER_MIN_MEAN` as the new tests need them.

- [ ] **Step 2: Update the three existing tests**

`test_gate6_flags_overshrunk_ratio_meanyr_cohort` and `test_gate6_gates_non_ratio_meanyr_normalization`: the WNBA frames now need `corr ≥ 0.58` (the new fire-on) to be anchored; they already build `anchored=True` cohorts with corr ~0.6, so they stay green — but assert against the leg via `gate_row(...)` unchanged. `test_gate6_exempts_unanchored_cell`: replace the `< _GATE6_MIN_RECENT_CORR` assertion with `_gate6_anchored(row["g6_recent_corr"], None) is False`. `test_gate6_applies_regardless_of_distribution_family`: call `_gate6_legs(frame, "Blended_EV", league="WNBA")` and assert `g6["g6_citl_ci_hi"] is not None` (the family-agnostic leg).

- [ ] **Step 3: Run the full Gate-6 block** — `poetry run pytest tests/golden/test_scorecard.py -k "gate6 or recent_form" -v`. All green.

- [ ] **Step 4: Run the whole scorecard file + the min_gate_slack tests** — `poetry run pytest tests/golden/test_scorecard.py -q`. If a `min_gate_slack` test sets `g6_star_ci_hi` and pinned the old sign, update its expected value to the corrected (positive-when-passing) slack.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test(scorecard): update Gate-6 golden tests for the three-leg rework"
```

---

## Task 8: Quality gates + refactoring-specialist

- [ ] **Step 1** — `poetry run ruff check src/sportstradamus/`. Fix any lint.
- [ ] **Step 2** — `poetry run pytest tests/golden/ -q`. Expected: green except the pre-existing `test_find_correlation_offer_correlations_real_nba` red (unrelated, confirmed by `git stash` in prior sessions).
- [ ] **Step 3** — `poetry run pytest -m integration -n0 && touch "$CLAUDE_PROJECT_DIR/.claude/.state/integration_green"`.
- [ ] **Step 4** — Dispatch the `refactoring-specialist` subagent on `src/sportstradamus/training/scorecard.py` and `src/sportstradamus/training/report.py` (the two Python files touched). Address anything it flags; do not push while it runs.
- [ ] **Step 5: Commit** any specialist edits — `git add -A && git commit -m "style(scorecard,report): refactoring-specialist pass"`.

---

## Task 9: Docs — ship_gate.md + handoff research-verdict block + ledger

**Files:**
- Modify: `docs/ship_gate.md`, `docs/handoffs/model_improvement_track.md`

- [ ] **Step 1: `docs/ship_gate.md`** — In the Gate-6 table row (line ~63) and the Gate-6 paragraph (~71-85): state Gate 6 = **(recent-form leg) OR (CITL-under leg) OR (count over-leg)**, strictly harder to pass than any one leg; the corr anchor (now a `0.58/0.52` hysteresis deadband) gates only the recent-form leg; CITL-under (`Σpred/ΣResult` UB `< 1 − 0.03`) runs on all cells; the guarded over-leg (`Σpred/ΣResult` LB `> 1 + 0.03`, count/ZINB, bench mean `Result ≥ 1`). Update the constant list (`_GATE6_FIRE_ON`/`_KEEP_ON`/`_OVER_MIN_MEAN`; drop `_GATE6_MIN_RECENT_CORR`). Update the "Gate 6 blank" bullet (~165) to "anchor-miss / too few stable rows / over-leg guard".

- [ ] **Step 2: `docs/handoffs/model_improvement_track.md`** — Update the §7.1 gate table row 6 to the OR-of-three. Add a §6-style **"Gate 6 graded-tolerance research verdict + outcome-directional legs"** block citing `/tmp/researcher_gate6_scope_and_drift.md` (Q1 graded KILL + hysteresis cure; Q2 CITL-under SHIP, sign-test ruled out; Q3 per-family; over-leg deferred-then-built-guarded). Add an **"Open questions / methods ruled out"** line: the **sign test is rejected** (reads mean-vs-median skew, not bias). Add a ledger entry (newest-first) summarizing the rework + the NBA-fantasy-points demote-candidate.

- [ ] **Step 3: Commit** — `git add -A && git commit -m "docs: Gate-6 three-leg rework (ship_gate + handoff verdict block + ledger)"`.

---

## Task 10: CV run for all cells (reworked gates + `ratio_projvol`)

This realizes the new gates on every cell's fresh scorecard and exercises the `ratio_projvol`
normalization. It is a long production run — kick it off and capture results.

- [ ] **Step 1: Back up the current served state** — `cp src/sportstradamus/data/training/model_stats.parquet /tmp/model_stats_pre_g6rework.parquet` so the before/after gate verdicts are diffable.

- [ ] **Step 2: Full CV/meditate across all cells** — `poetry run meditate` (no args = every league/market). This retrains and re-scores every cell; `report()` writes the new `model_stats.parquet` with the three-leg Gate-6 columns and seeds hysteresis from the backed-up prior. Run in the background; it is multi-hour.

- [ ] **Step 3: `ratio_projvol` board sweep on eligible cells** — for the NFL efficiency + NBA/WNBA per-minute cells the resolver accepts, run the deterministic strategy board including `ratio_projvol`, e.g. `poetry run model-strategy-driver --board --league NFL --market receiving-yards --normalizations ratio_meanyr,ratio_projvol` (repeat per eligible cell, or via the board's per-cell loop). This tests the new normalization against the reworked gates without committing it to production. **CHECKPOINT:** whether any `ratio_projvol` cell ships under real HPO is an owner call (it was parked on g4 regression); surface the board slacks, do not flip `stat_meta.json` without sign-off.

- [ ] **Step 4: Diff the gate verdicts** — compare `model_stats.parquet` vs the backup on `g6_pass`/`ship` and the new `g6_citl_ci_hi`/`g6_over_ci_lo` columns. Confirm the predicted catches: NBA fantasy-points fails g6 (CITL + recent-form); the 6 count cells' over-leg verdicts; no unexpected demotions. Surface any newly-failing shipped cell to the owner (do not auto-demote — re-confirm on the next run per §8.1).

---

## Task 11: Post-CV handoff update

**Files:**
- Modify: `docs/handoffs/model_improvement_track.md`

- [ ] **Step 1** — Append the CV-run results to the ledger entry / verdict block: which cells the three-leg Gate 6 caught, the `ratio_projvol` board slacks per eligible cell, and the demote decisions (gated on owner sign-off). Update the §7.1 / served-count rollups if any cell's ship verdict changed.
- [ ] **Step 2: Commit** — `git add -A && git commit -m "docs(handoff): Gate-6 rework CV-run results + ratio_projvol board"`.
- [ ] **Step 3: Offer a memory capture** — the durable lesson (Gate 6 = OR-of-three; CITL-vs-outcome catches what g2's σ-normalization launders; recent-form leg is the ratio_meanyr-artifact regression test) updates the existing `[[gate6_anti_shrinkage_holdout_blind]]` memory.

---

## Self-review

**Spec coverage:** recent-form leg + hysteresis (Tasks 1-3, 5-6) ✓; CITL-under all-cells (Task 3, 5) ✓; guarded over-leg (Task 3) ✓; pure-function state-threading via `report()` (Task 6) ✓; OR-of-three pass + slack (Tasks 4-5) ✓; cold-start seed from existing columns (`recent_form_fired`, Task 4/6) ✓; constants incl. removal of `_GATE6_MIN_RECENT_CORR` (Task 1) ✓; tests for every new behavior (Tasks 2-4, 7) ✓; docs incl. research-verdict block + sign-test-ruled-out (Task 9) ✓; CV run + `ratio_projvol` + post-run handoff (Tasks 10-11) ✓.

**Placeholder scan:** every code step shows the full function; no TBD/TODO; the one "tune seed so corr lands in the band" note is wrapped in an `if` so the test is robust whatever the seed yields.

**Type/name consistency:** `_gate6_legs` returns the exact `g6_*` keys read by `_gate6_passes`, `apply_thresholds`, `min_gate_slack`, and emitted by `gate_row`; `prior_g6_fired: bool | None` is consistent across `_gate6_anchored` → `_gate6_legs` → `gate_row` → `compute_gates` → `_layer_gates_from_test_set`; `recent_form_fired` reads the same `g6_star_ci_hi`/`g6_star_ref` keys `_gate6_legs` writes.

---

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks. Good fit: Tasks 1-7 are tight TDD units; Task 8 (gates + specialist) and 10 (the multi-hour CV run) are main-session checkpoints.
2. **Inline Execution** — execute in this session with checkpoints.

Note: Tasks 1-9 are the code+docs fix; Task 10 is the long CV run the owner asked to kick off once the fix lands; Task 11 closes the loop. The `ratio_projvol` ship decision in Task 10 is explicitly owner-gated.
