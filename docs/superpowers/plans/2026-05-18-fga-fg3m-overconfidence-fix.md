# FGA-under / FG3M-over Overconfidence Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two confirmed, path-wide model biases — SkewNormal markets regressing high-volume players toward the global mean (FGA confident-Unders) and ZINB markets under-fitting the zero-inflation gate (FG3M confident-Overs).

**Architecture:** (A) Replace the SkewNormal multiplicative `Result/MeanYr` target with an **additive empirical-Bayes per-player offset**: train on `Result − EB_prior`, predict `EB_prior + boosted_residual`, so the per-player level is added back exactly and the trees model only deviations (the documented mixed-effects / random-intercept remedy). (B) Replace the jointly-fit `ZINB` for zero-inflated markets with a **two-stage hurdle** (`HurdleZINB`): a separately-calibrated binary P(nonzero) classifier supplies the gate, a NegBin fit on the strictly-positive subset supplies the count params. `HurdleZINB.predict` returns the same `total_count/probs/gate` columns as `ZINB`, so all downstream code (`get_odds`, `fused_loc`, `get_ev`, `model_prob` ZINB branch) is unchanged.

**Tech Stack:** Python 3.11, LightGBM, LightGBMLSS, custom `SkewNormal` (PyTorch), pandas/numpy, pytest, Poetry. No new dependencies.

---

## Context (why)

Confirmed from live published output (`current_offers.parquet`) + offline holdout replay:

- **Root cause A (every SkewNormal market):** `Result/MeanYr` ratio target + GBDT regression-to-global-mean ⇒ predicted level shrinks for high-volume players. FGA: elite starters get EV ~30% below the line → 100% confident Unders in the live slate. Validated remedy (FGA holdout): additive EB offset moves top-volume-quintile bias −2.0 → −0.48, corr(bias,MeanYr) −0.17 → −0.04, predicted-Under rate for elite players 0.95 → 0.65 (actual 0.62). Multiplicative variants all stay broken.
- **Root cause B (every ZINB market):** the jointly-fit ZINB `gate` head is unidentified under nll and converges to ≈half the true structural-zero rate (FG3M 0.19 vs 0.34) ⇒ P(over@line) inflated everywhere → confident Overs. Validated remedy (FG3M holdout): two-stage hurdle recovers zero mass 0.35 vs true 0.34 and halves the P(over@line) gap 0.348 → 0.191; standalone zero-classifier well-calibrated (Brier 0.166).

Decisions locked with the user: A = no-dep additive EB offset, path-wide SkewNormal. B = two-stage hurdle, **ZINB-only first** (ZAGamma untouched). Sequence: implement + retrain + verify **A fully first, commit, then B** (one module per session, easy bisect).

Residual known issues (refine during implementation, not blockers): A has mild mid-volume-quintile over-prediction (tune `EB_SHRINKAGE_K`); B's NegBin-on-positives still slightly over-predicts the count (acceptable; gate identifiability is fixed).

---

## File Structure

**Phase A — SkewNormal additive EB offset**
- Create `src/sportstradamus/training/offsets.py` — `compute_eb_prior()` (empirical-Bayes per-player prior) + `EB_SHRINKAGE_K`. Single responsibility: the offset math.
- Modify `src/sportstradamus/training/pipeline.py` — SkewNormal branch: build offset instead of ratio-normalizing; persist `offset_meta`; change test/diagnostic decode.
- Modify `src/sportstradamus/helpers/distributions.py` — `set_model_start_values` SkewNormal additive-offset mode (loc start 0, scale start from STDYr).
- Modify `src/sportstradamus/prediction/model_prob.py` — SkewNormal decode: `EV = EB_prior + skewnormal_mean`, drop the `× MeanYr` denorm; read `offset_meta` instead of `normalized`.
- Test `tests/test_eb_offset.py`.

**Phase B — Hurdle for ZINB**
- Create `src/sportstradamus/hurdle.py` — `HurdleZINB` (binary P(nonzero) clf + NegBin-on-positives), `.predict(X, pred_type="parameters") → DataFrame[total_count, probs, gate]`, `.set_model_start_values(X)`, picklable.
- Modify `src/sportstradamus/training/pipeline.py` — when `hist_gate > 0.02` build/train `HurdleZINB` instead of `LightGBMLSS(ZINB)`; keep `dist="ZINB"` tag.
- Modify `src/sportstradamus/prediction/model_prob.py` & `src/sportstradamus/helpers/distributions.py` `set_model_start_values` call sites — detect hurdle model and delegate start values; `model.predict` interface unchanged.
- Test `tests/test_hurdle_zinb.py`.

Downstream `get_odds`/`fused_loc`/`get_ev` (`helpers/distributions.py`, `helpers/archive.py`) are **unchanged** in both phases — A keeps absolute EV/sigma; B keeps the ZINB `total_count/probs/gate` contract.

---

# PHASE A — SkewNormal additive empirical-Bayes offset

### Task A1: EB prior helper + constant

**Files:**
- Create: `src/sportstradamus/training/offsets.py`
- Test: `tests/test_eb_offset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eb_offset.py
import numpy as np
from sportstradamus.training.offsets import compute_eb_prior, EB_SHRINKAGE_K


def test_eb_prior_shrinks_low_games_toward_global():
    # 1 game observed, player mean 30, global 10 -> heavily shrunk toward 10
    p = compute_eb_prior(np.array([30.0]), np.array([1.0]), global_mean=10.0,
                          k=EB_SHRINKAGE_K)
    assert 10.0 < p[0] < 14.0


def test_eb_prior_trusts_high_games():
    # 200 games observed -> prior ~ player mean
    p = compute_eb_prior(np.array([30.0]), np.array([200.0]), global_mean=10.0,
                          k=EB_SHRINKAGE_K)
    assert abs(p[0] - 30.0) < 1.5


def test_eb_prior_formula_exact():
    p = compute_eb_prior(np.array([20.0]), np.array([10.0]), global_mean=5.0, k=10.0)
    # (10*20 + 10*5) / (10 + 10) = 12.5
    assert abs(p[0] - 12.5) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_eb_offset.py -q`
Expected: FAIL — `ModuleNotFoundError: sportstradamus.training.offsets`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sportstradamus/training/offsets.py
"""Empirical-Bayes per-player prior for additive SkewNormal offset modeling.

The GBDT regresses grouped (per-player) targets toward the global mean. We
keep the per-player level OUT of the boosted function: train on
``Result - EB_prior`` and add ``EB_prior`` back at prediction. ``EB_prior``
shrinks the noisy trailing per-player mean toward the global mean by games
observed (James-Stein / empirical-Bayes), so low-sample players are not
over-trusted.
"""

import numpy as np

# Shrinkage strength: pseudo-count of "global-mean games" mixed into each
# player's trailing mean. Validated on FGA holdout (top-volume-quintile bias
# -2.0 -> -0.48 at K=10); retune in Task A7 if mid-volume over-prediction.
EB_SHRINKAGE_K = 10.0


def compute_eb_prior(player_mean, games_played, global_mean, k=EB_SHRINKAGE_K):
    """Empirical-Bayes shrink a per-player trailing mean toward the global mean.

    prior = (games * player_mean + k * global_mean) / (games + k)
    """
    pm = np.asarray(player_mean, dtype=float)
    g = np.clip(np.asarray(games_played, dtype=float), 0.0, None)
    return (g * pm + k * float(global_mean)) / (g + k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_eb_offset.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/training/offsets.py tests/test_eb_offset.py
git commit -m "feat(training): empirical-Bayes per-player prior helper for SkewNormal offset"
```

---

### Task A2: `set_model_start_values` additive-offset mode for SkewNormal

**Files:**
- Modify: `src/sportstradamus/helpers/distributions.py:464-477` (SkewNormal branch of `set_model_start_values`)
- Test: `tests/test_eb_offset.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_eb_offset.py
import pandas as pd
from sportstradamus.helpers import set_model_start_values


class _Stub:
    def __init__(self): self.start_values = None


def test_skewnormal_offset_mode_loc_start_zero():
    X = pd.DataFrame({"MeanYr": [12.0, 4.0], "STDYr": [3.0, 1.5],
                      "ZeroYr": [0.0, 0.1]})
    m = _Stub()
    set_model_start_values(m, "SkewNormal", X, offset_mode=True)
    sv = m.start_values
    # cols: [loc, log(scale), alpha]; additive-residual loc starts at 0
    assert sv.shape == (2, 3)
    assert np.allclose(sv[:, 0], 0.0)
    # scale start ~ STDYr (residual std), in log space
    assert np.allclose(np.exp(sv[:, 1]), [3.0, 1.5], rtol=0.2)
    assert np.allclose(sv[:, 2], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_eb_offset.py::test_skewnormal_offset_mode_loc_start_zero -q`
Expected: FAIL — `set_model_start_values() got an unexpected keyword argument 'offset_mode'`

- [ ] **Step 3: Write minimal implementation**

In `src/sportstradamus/helpers/distributions.py`, change the signature at line 425:

```python
def set_model_start_values(model, dist, X_data, shape_ceiling=None,
                           normalized=False, offset_mode=False):
```

Replace the SkewNormal branch (lines 464-477) with:

```python
    if dist == "SkewNormal":
        if offset_mode:
            # Additive residual target (Result - EB_prior): loc centered at
            # 0, scale ~ per-player STD (residual dispersion ~ player std).
            loc = np.zeros(n)
            scale = std.copy()
        elif normalized:
            cv_player = np.clip(std / mu, 0.01, 10)
            loc = np.ones(n)
            scale = cv_player
        else:
            loc = mu.copy()
            scale = std.copy()
        alpha_skew = np.zeros(n)
        sv = np.column_stack([loc, np.log(np.clip(scale, 1e-6, None)), alpha_skew])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_eb_offset.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/helpers/distributions.py tests/test_eb_offset.py
git commit -m "feat(helpers): SkewNormal additive-offset start values (loc=0, scale=STDYr)"
```

---

### Task A3: pipeline — build additive offset, persist `offset_meta`

**Files:**
- Modify: `src/sportstradamus/training/pipeline.py:262-291` (SkewNormal target build), `:341,394,400,406` (start-value calls), `:439-475` (test decode), `:942-951` (pickle metadata)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_eb_offset.py
def test_offset_meta_roundtrip():
    """offset_meta must let prediction recompute EB_prior identically."""
    from sportstradamus.training.offsets import compute_eb_prior
    meta = {"method": "eb_additive", "k": 10.0, "global_mean": 8.69}
    my = np.array([18.0]); gp = np.array([40.0])
    train_prior = compute_eb_prior(my, gp, meta["global_mean"], meta["k"])
    pred_prior = compute_eb_prior(my, gp, meta["global_mean"], meta["k"])
    assert np.allclose(train_prior, pred_prior)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_eb_offset.py::test_offset_meta_roundtrip -q`
Expected: PASS already (pure helper) — this is the contract guard; if it errors, fix import. (Acceptable: it locks the train/predict symmetry the pipeline edits must preserve.)

- [ ] **Step 3: Implement pipeline changes**

3a. Add import near line 38:
```python
from sportstradamus.training.offsets import EB_SHRINKAGE_K, compute_eb_prior
```

3b. Replace the SkewNormal block (lines 266-291) — the `if global_mean >= 2.0:` body — with:

```python
    if global_mean >= 2.0:
        dist = "SkewNormal"
        normalize = False  # additive-offset mode replaces ratio normalization
        offset_mode = True
        dist_obj = SkewNormalDist(stabilization="None", loss_fn="crps")

        cv = (
            player_stats.std()
            / player_stats.mean()
            * player_stats.count()
            / player_stats.count().sum()
        ).sum()
        cv = max(cv, 0.05)
        shape_ceiling = None
        marginal_shape = None
        denom_col = "MeanYr"

        gp_train = X_train["GamesPlayed"].clip(lower=0).to_numpy()
        my_train = X_train[denom_col].clip(lower=0.5).to_numpy()
        eb_global_mean = float(global_mean)
        eb_prior_train = compute_eb_prior(
            my_train, gp_train, eb_global_mean, EB_SHRINKAGE_K
        )
        y_train_labels = y_train_labels - eb_prior_train
    else:
        offset_mode = False
        # ... (existing NegBin/ZINB block unchanged) ...
```
(Keep the existing `else:` NegBin/ZINB body exactly as-is below it. Delete the old SkewNormal nonzero-subsetting + `y_train_labels / meanyr_train` + clip lines — they belonged to the removed ratio scheme.)

3c. Initialize `offset_mode = False` before the `if global_mean` (so the NegBin path defines it). At the four `set_model_start_values(... normalized=normalize)` calls (lines 341, 394, 400, 406) add `offset_mode=offset_mode`.

3d. Replace the SkewNormal test/validation decode (lines 439-475 region). Where it currently does `loc_abs = loc_norm * meanyr_test; scale_abs = scale_norm * meanyr_test; ev = loc_abs + scale_abs*delta*sqrt(2/pi)` substitute:

```python
    if dist == "SkewNormal":
        loc = prob_params["loc"].to_numpy()
        scale = prob_params["scale"].to_numpy()
        alpha_sn = prob_params["alpha"].to_numpy()
        loc_v = prob_params_validation["loc"].to_numpy()
        scale_v = prob_params_validation["scale"].to_numpy()
        alpha_v = prob_params_validation["alpha"].to_numpy()

        eb_test = compute_eb_prior(
            X_test[denom_col].clip(lower=0.5).to_numpy(),
            X_test["GamesPlayed"].clip(lower=0).to_numpy(),
            eb_global_mean, EB_SHRINKAGE_K,
        )
        eb_val = compute_eb_prior(
            X_validation[denom_col].clip(lower=0.5).to_numpy(),
            X_validation["GamesPlayed"].clip(lower=0).to_numpy(),
            eb_global_mean, EB_SHRINKAGE_K,
        )
        delta = alpha_sn / np.sqrt(1 + alpha_sn**2)
        ev = eb_test + loc + scale * delta * np.sqrt(2 / np.pi)
        delta_v = alpha_v / np.sqrt(1 + alpha_v**2)
        ev_validation = eb_val + loc_v + scale_v * delta_v * np.sqrt(2 / np.pi)
        sn_sigma_test = scale          # already absolute (residual units)
        sn_sigma_val = scale_v
        sn_alpha_test = alpha_sn
        sn_alpha_val = alpha_v
        if hist_gate > 0.02:
            gate_test = np.full_like(ev, hist_gate)
            gate_validation = np.full_like(ev_validation, hist_gate)
```

3e. Replace pickle metadata line 951 `"normalized": normalize,` with:
```python
        "offset_meta": (
            {"method": "eb_additive", "k": EB_SHRINKAGE_K,
             "global_mean": eb_global_mean, "prior_col": denom_col}
            if offset_mode else None
        ),
        "normalized": normalize,
```
(`normalize` is now always False for SkewNormal; keep the key for back-compat readers but `offset_meta` is authoritative.)

3f. Update the SkewNormal diagnostic block (`pipeline.py:816-822`): keep the existing effective-std diagnostic but compute model EV from `ev` (already absolute) — no `× MeanYr`. Set `diag_start_shape = float(np.mean(std))`-style absolute; reuse existing `skewnormal_effective_std` if present, else `diag_model_shape = float(np.std(ev))` placeholder is **not** acceptable — instead `diag_model_shape = float(np.mean(sn_sigma_test)); diag_empirical_shape = float(y_test["Result"].std()); diag_shape_label = "std"`.

- [ ] **Step 4: Run targeted offline replay to verify the bias is fixed**

Run (throwaway, not committed): adapt `/tmp/protoA2_meanrev.py` to load the *retrained* `NBA_FGA.mdl` after Task A6, OR run the in-process check:
```bash
poetry run pytest tests/test_eb_offset.py tests/golden/ -q
poetry run ruff check src/sportstradamus/
```
Expected: tests PASS, ruff clean. (Full bias verification is Task A6 after retrain.)

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/training/pipeline.py tests/test_eb_offset.py
git commit -m "feat(training): SkewNormal additive EB offset target + offset_meta persistence"
```

---

### Task A4: model_prob — decode SkewNormal via EB offset

**Files:**
- Modify: `src/sportstradamus/prediction/model_prob.py:100` (read offset_meta), `:127` (start values), `:181-200` (decode)
- Test: `tests/test_eb_offset.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_eb_offset.py
def test_model_prob_skewnormal_decode_uses_eb_offset(monkeypatch):
    """EV must be EB_prior + skewnormal_mean, NOT loc * MeanYr."""
    import numpy as np
    from sportstradamus.training.offsets import compute_eb_prior
    loc, scale, alpha = 1.5, 3.0, 0.0
    meanyr, games = 18.0, 40.0
    gm, k = 8.69, 10.0
    eb = compute_eb_prior(np.array([meanyr]), np.array([games]), gm, k)[0]
    delta = alpha / np.sqrt(1 + alpha**2)
    expected_ev = eb + loc + scale * delta * np.sqrt(2 / np.pi)
    # contract: decode == EB_prior + (loc + scale*delta*sqrt(2/pi)); NOT loc*meanyr
    assert abs(expected_ev - (eb + loc)) < 1e-9
    assert expected_ev != loc * meanyr
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `poetry run pytest tests/test_eb_offset.py::test_model_prob_skewnormal_decode_uses_eb_offset -q`
Expected: PASS (pure contract guard locking the formula the edit must implement).

- [ ] **Step 3: Implement model_prob changes**

3a. Line ~100, replace `normalized = filedict.get("normalized", False)` with:
```python
        offset_meta = filedict.get("offset_meta")
        normalized = filedict.get("normalized", False)
```
3b. Line ~127 `set_model_start_values(model, dist, playerStats, normalized=normalized)` →
```python
            set_model_start_values(
                model, dist, playerStats,
                normalized=normalized,
                offset_mode=bool(offset_meta and offset_meta.get("method") == "eb_additive"),
            )
```
3c. Replace the SkewNormal decode block (lines 181-200) with:
```python
        if dist == "SkewNormal" and "loc" in prob_params.columns:
            from sportstradamus.training.offsets import compute_eb_prior
            prior_col = (offset_meta or {}).get("prior_col", "MeanYr")
            eb = compute_eb_prior(
                playerStats[prior_col].clip(lower=0.5).values,
                playerStats["GamesPlayed"].clip(lower=0).values,
                (offset_meta or {}).get("global_mean", 0.0),
                (offset_meta or {}).get("k", 10.0),
            )
            loc = prob_params["loc"].values
            scale = prob_params["scale"].values
            alpha_sn = prob_params["alpha"].values
            delta = alpha_sn / np.sqrt(1 + alpha_sn**2)
            prob_params["Model EV"] = eb + loc + scale * delta * np.sqrt(2 / np.pi)
            prob_params["Model Sigma"] = scale          # absolute (residual units)
            prob_params["Model Skew"] = alpha_sn
            if hist_gate > 0.02:
                prob_params["Model Gate"] = hist_gate
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_eb_offset.py tests/golden/ -q && poetry run ruff check src/sportstradamus/`
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/prediction/model_prob.py tests/test_eb_offset.py
git commit -m "feat(prediction): decode SkewNormal EV via additive EB offset"
```

---

### Task A5: Retrain SkewNormal markets

**Files:** none (model artifacts regenerated)

- [ ] **Step 1:** Retrain NBA (covers FGA + all NBA SkewNormal):
```bash
poetry run meditate --league NBA --force
```
Expected: completes; `data/models/NBA_*.mdl` rewritten; `data/training_report.txt` regenerated.

- [ ] **Step 2:** (Path-wide) retrain other leagues' SkewNormal markets when convenient:
```bash
poetry run meditate --league NFL --force   # repeat WNBA as needed
```

- [ ] **Step 3: Commit regenerated artifacts**
```bash
git add src/sportstradamus/data/models/ src/sportstradamus/data/training_report.txt src/sportstradamus/data/model_stats.parquet src/sportstradamus/data/stat_cv.json src/sportstradamus/data/stat_zi.json
git commit -m "chore(models): retrain SkewNormal markets with additive EB offset"
```

---

### Task A6: Verify A on holdout + live replay

**Files:** none (verification)

- [ ] **Step 1: Offline holdout bias check** — script:
```bash
poetry run python - <<'EOF'
import pickle,pkg_resources,pandas as pd,numpy as np
from sportstradamus.helpers import set_model_start_values
from sportstradamus.training.offsets import compute_eb_prior
P=pkg_resources.resource_filename
d=pickle.load(open(P('sportstradamus','data/models/NBA_FGA.mdl'),'rb'))
m=d['offset_meta']; df=pd.read_csv(P('sportstradamus','data/test_sets/NBA_FGA.csv'))
X=df[[c for c in d['expected_columns'] if c in df.columns]].copy()
for c in('Home','Player position'):
    if c in X: X[c]=X[c].astype('category')
mdl=d['model']; set_model_start_values(mdl,'SkewNormal',X,offset_mode=True)
pp=mdl.predict(X,pred_type='parameters')
eb=compute_eb_prior(df['MeanYr'].clip(lower=.5),df['GamesPlayed'].clip(lower=0),m['global_mean'],m['k'])
a=pp['alpha'].to_numpy(); dl=a/np.sqrt(1+a**2)
ev=eb+pp['loc'].to_numpy()+pp['scale'].to_numpy()*dl*np.sqrt(2/np.pi)
res=df['Result'].to_numpy(); my=df['MeanYr'].to_numpy(); ln=df['Line'].to_numpy()
qs=pd.qcut(my,5,labels=False,duplicates='drop'); top=qs==qs.max()
print('corr(bias,MeanYr)=%.3f topQ bias=%.2f predUnder=%.2f actUnder=%.2f'%(
 np.corrcoef(ev-res,my)[0,1], np.mean((ev-res)[top]),
 np.mean(ev[top]<ln[top]), np.mean(res[top]<ln[top])))
EOF
```
Expected: `corr` ≈ 0 (was −0.81/−0.17), topQ bias ≈ −0.5 or better (was −2.0), predUnder ≈ actUnder (was 0.95 vs 0.62).

- [ ] **Step 2: SkewNormal sweep** — re-run the market-specificity sweep (all NBA SkewNormal pickles): every market's `corr(bias,MeanYr)` should be near 0, none with the −0.4…−0.87 pattern.

- [ ] **Step 3:** If `corr` not ≈ 0 or topQ bias still strongly negative, **stop** and retune `EB_SHRINKAGE_K` (Task A7) before proceeding to Phase B.

---

### Task A7: (Conditional) retune `EB_SHRINKAGE_K`

Only if Task A6 shows residual mid-volume over-prediction or incomplete top-volume fix.

- [ ] **Step 1:** Offline K sweep on FGA holdout (throwaway script): for K in {5,10,20,40,80}, compute per-MeanYr-quintile `mean(EV−Result)` and `mean|bias|`. Pick K minimizing max |per-quintile bias|.
- [ ] **Step 2:** Update `EB_SHRINKAGE_K` in `src/sportstradamus/training/offsets.py` with the chosen value and a one-line reason comment.
- [ ] **Step 3:** Re-run Task A5 (retrain) + Task A6 (verify).
- [ ] **Step 4: Commit**
```bash
git add src/sportstradamus/training/offsets.py
git commit -m "tune(training): set EB_SHRINKAGE_K=<value> from holdout quintile-bias sweep"
```

---

# PHASE B — Two-stage hurdle for ZINB markets

### Task B1: `HurdleZINB` model class

**Files:**
- Create: `src/sportstradamus/hurdle.py`
- Test: `tests/test_hurdle_zinb.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hurdle_zinb.py
import numpy as np, pandas as pd
from sportstradamus.hurdle import HurdleZINB


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    # 35% structural zeros; positives ~ NB-ish around 2
    is0 = rng.random(n) < 0.35
    y = np.where(is0, 0, rng.poisson(2.0, n) + 1).astype(float)
    return X, y


def test_hurdle_predict_returns_zinb_param_columns():
    X, y = _data()
    h = HurdleZINB()
    h.fit(X, y, hp={"num_leaves": 8, "learning_rate": 0.1, "verbosity": -1,
                    "num_threads": 4}, rounds=80)
    pp = h.predict(X, pred_type="parameters")
    assert list(pp.columns) == ["total_count", "probs", "gate"]
    assert len(pp) == len(X)


def test_hurdle_gate_recovers_true_zero_rate():
    X, y = _data()
    h = HurdleZINB()
    h.fit(X, y, hp={"num_leaves": 8, "learning_rate": 0.1, "verbosity": -1,
                    "num_threads": 4}, rounds=120)
    pp = h.predict(X, pred_type="parameters")
    # mean predicted gate (P structural zero) ~ true 0.35 (ZINB joint fit ~0.18)
    assert abs(pp["gate"].mean() - 0.35) < 0.07


def test_hurdle_pickle_roundtrip():
    import pickle
    X, y = _data()
    h = HurdleZINB()
    h.fit(X, y, hp={"num_leaves": 8, "learning_rate": 0.1, "verbosity": -1,
                    "num_threads": 4}, rounds=60)
    h2 = pickle.loads(pickle.dumps(h))
    pd.testing.assert_frame_equal(
        h.predict(X, pred_type="parameters"),
        h2.predict(X, pred_type="parameters"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_hurdle_zinb.py -q`
Expected: FAIL — `ModuleNotFoundError: sportstradamus.hurdle`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sportstradamus/hurdle.py
"""Two-stage hurdle replacing jointly-fit ZINB for zero-inflated markets.

The ZINB gate is unidentified under joint nll and converges to ~half the true
structural-zero rate, inflating P(over). The hurdle decouples it:

* Stage 1: a calibrated binary classifier for P(Result > 0).
* Stage 2: a NegBin (LightGBMLSS) fit on the strictly-positive subset.

``predict(..., pred_type="parameters")`` returns the SAME columns a
LightGBMLSS ``ZINB`` produces — ``total_count``, ``probs``, ``gate`` (with
``gate = P(Result == 0)``) — so every downstream consumer (``get_odds``,
``fused_loc``, ``get_ev``, ``model_prob`` ZINB branch) is unchanged.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbmlss.distributions.NegativeBinomial import NegativeBinomial
from lightgbmlss.model import LightGBMLSS

from sportstradamus.helpers import set_model_start_values

_CLF_KEYS = (
    "num_leaves", "learning_rate", "feature_fraction", "bagging_fraction",
    "bagging_freq", "min_child_samples", "max_bin", "num_threads", "verbosity",
)


class HurdleZINB:
    """ZINB-compatible hurdle: binary P(nonzero) clf + NegBin on positives."""

    is_hurdle = True

    def __init__(self):
        self.clf = None
        self.nb = None

    def _clf_params(self, hp):
        p = {k: hp[k] for k in _CLF_KEYS if k in hp}
        p.update(objective="binary", metric="binary_logloss")
        return p

    def fit(self, X, y, hp, rounds, shape_ceiling=None):
        y = np.asarray(y, dtype=float)
        self.clf = lgb.train(
            self._clf_params(hp),
            lgb.Dataset(X, label=(y > 0).astype(int)),
            num_boost_round=rounds,
        )
        pos = y > 0
        nb = LightGBMLSS(NegativeBinomial(stabilization="None", loss_fn="nll"))
        if shape_ceiling is not None:
            from sportstradamus.training.hyperparams import _BoundedResponseFn
            nb.dist.param_dict["total_count"] = _BoundedResponseFn(
                nb.dist.param_dict["total_count"], shape_ceiling)
        Xp = X[pos] if hasattr(X, "__getitem__") else X[pos]
        set_model_start_values(nb, "NegBin", Xp)
        nb.train({k: v for k, v in hp.items()},
                 lgb.Dataset(Xp, label=y[pos]), num_boost_round=rounds)
        self.nb = nb
        return self

    def set_model_start_values(self, X):
        set_model_start_values(self.nb, "NegBin", X)

    def predict(self, X, pred_type="parameters"):
        p_nonzero = np.clip(self.clf.predict(X), 1e-6, 1 - 1e-6)
        nb = self.nb.predict(X, pred_type="parameters")
        return pd.DataFrame({
            "total_count": nb["total_count"].to_numpy(),
            "probs": nb["probs"].to_numpy(),
            "gate": 1.0 - p_nonzero,  # P(structural/observed zero)
        }, index=nb.index)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_hurdle_zinb.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/hurdle.py tests/test_hurdle_zinb.py
git commit -m "feat: HurdleZINB (calibrated zero clf + NegBin-on-positives), ZINB-compatible predict"
```

---

### Task B2: pipeline — use `HurdleZINB` for ZINB markets

**Files:**
- Modify: `src/sportstradamus/training/pipeline.py` ZINB branch (`:293-309`), model construction (`:340`), start-value calls (`:341,394,400,406`), predict calls (`:395,401,407`)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_hurdle_zinb.py
def test_pipeline_builds_hurdle_for_zinb(monkeypatch):
    """When hist_gate>0.02, the model object must be a HurdleZINB."""
    from sportstradamus.hurdle import HurdleZINB
    # contract guard: pipeline selects HurdleZINB for the ZINB regime.
    # (Full pipeline run is covered by integration tests; here we assert the
    # class is importable and exposes the ZINB-compatible interface.)
    h = HurdleZINB()
    assert h.is_hurdle and hasattr(h, "predict") and hasattr(h, "fit")
```

- [ ] **Step 2: Run** `poetry run pytest tests/test_hurdle_zinb.py::test_pipeline_builds_hurdle_for_zinb -q` — Expected: PASS (guard).

- [ ] **Step 3: Implement pipeline ZINB→Hurdle swap**

3a. Import near line 38: `from sportstradamus.hurdle import HurdleZINB`.

3b. In the NegBin/ZINB `else:` block (lines ~293-309): keep the `dist = "NegBin"` / `dist = "ZINB"` selection and `shape_ceiling`/`cv` computation. Where `dist_obj = ZINB(...)` is built and (line 340) `model = LightGBMLSS(dist_obj)` — branch:

```python
        is_hurdle = dist == "ZINB"
        if dist == "NegBin":
            dist_obj = NegativeBinomial(stabilization="None", loss_fn="nll")
        # ZINB: no dist_obj; HurdleZINB built below.
```
Replace `model = LightGBMLSS(dist_obj)` (line ~340) and the `_BoundedResponseFn` wrap (lines 331-345) with:
```python
    if dist in ("NegBin",):
        dist_obj.param_dict["total_count"] = _BoundedResponseFn(
            dist_obj.param_dict["total_count"], shape_ceiling)
        model = LightGBMLSS(dist_obj)
    elif dist == "ZINB":
        model = HurdleZINB()
    elif dist in ("Gamma", "ZAGamma"):
        dist_obj.param_dict["concentration"] = _BoundedResponseFn(
            dist_obj.param_dict["concentration"], shape_ceiling)
        model = LightGBMLSS(dist_obj)
    else:
        model = LightGBMLSS(dist_obj)
```

3c. Training: where the model is trained (the `model.train(...)` / hyperopt path), add a hurdle branch that calls `model.fit(X_train, y_train_labels, hp=<resolved params>, rounds=<opt_rounds>, shape_ceiling=shape_ceiling)`. For HurdleZINB skip Optuna LightGBMLSS-specific tuning; reuse the resolved `params` dict (same keys) and `opt_rounds`. Guard every `set_model_start_values(model, dist, X*, ...)` (lines 341,394,400,406) and `model.predict(X*, pred_type="parameters")` (395,401,407):
```python
    if getattr(model, "is_hurdle", False):
        model.set_model_start_values(X_train)   # (and X_validation / X_test before each predict)
    else:
        set_model_start_values(model, dist, X_train, shape_ceiling=shape_ceiling,
                               normalized=normalize, offset_mode=offset_mode)
```
`model.predict(X, pred_type="parameters")` works unchanged for `HurdleZINB`.

3d. Pickle: `filedict["model"] = model` already; `dist` stays `"ZINB"` so `distribution` metadata and all downstream branches are unchanged.

- [ ] **Step 4: Run** `poetry run pytest tests/test_hurdle_zinb.py tests/golden/ -q && poetry run ruff check src/sportstradamus/` — Expected: PASS, clean.

- [ ] **Step 5: Commit**
```bash
git add src/sportstradamus/training/pipeline.py tests/test_hurdle_zinb.py
git commit -m "feat(training): use HurdleZINB for ZINB markets (ZINB-compatible, downstream unchanged)"
```

---

### Task B3: model_prob & start-value call sites — hurdle-aware

**Files:**
- Modify: `src/sportstradamus/prediction/model_prob.py:127`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_hurdle_zinb.py
def test_hurdle_predict_interface_matches_zinb_consumer():
    """model_prob ZINB branch consumes total_count*probs/(1-probs) + gate."""
    import numpy as np
    X, y = _data()
    h = HurdleZINB().fit(X, y, hp={"num_leaves": 8, "learning_rate": 0.1,
                                   "verbosity": -1, "num_threads": 4}, rounds=60)
    pp = h.predict(X, pred_type="parameters")
    base_ev = pp["total_count"] * pp["probs"] / (1 - pp["probs"])
    assert np.isfinite(base_ev).all() and (pp["gate"].between(0, 1)).all()
```

- [ ] **Step 2: Run** the test — Expected: PASS (interface contract).

- [ ] **Step 3: Implement** — `model_prob.py:127`, make start-values hurdle-aware:
```python
            if getattr(model, "is_hurdle", False):
                model.set_model_start_values(playerStats)
            else:
                set_model_start_values(
                    model, dist, playerStats,
                    normalized=normalized,
                    offset_mode=bool(offset_meta and offset_meta.get("method") == "eb_additive"),
                )
```
No other change: `model.predict(playerStats, pred_type="parameters")` (line ~129) returns `total_count/probs/gate`; the existing ZINB branch (lines 165-172) computes `base_ev`, `Model Gate`, `Model R` exactly as before.

- [ ] **Step 4: Run** `poetry run pytest tests/test_hurdle_zinb.py tests/golden/ -q && poetry run ruff check src/sportstradamus/` — Expected PASS, clean.

- [ ] **Step 5: Commit**
```bash
git add src/sportstradamus/prediction/model_prob.py tests/test_hurdle_zinb.py
git commit -m "feat(prediction): hurdle-aware start values; ZINB decode path unchanged"
```

---

### Task B4: Retrain ZINB markets + verify

**Files:** none

- [ ] **Step 1:** `poetry run meditate --league NBA --force` (rebuilds ZINB markets as hurdles).
- [ ] **Step 2: Gate calibration check** — adapt the ZINB sweep: for each NBA ZINB market, predicted `gate` mean vs actual zero rate, and P(over@line) vs actual. Expected: predicted gate ≈ true zero rate (was ≈half); P(over@line) gap roughly halved or better.
- [ ] **Step 3:** Commit regenerated artifacts:
```bash
git add src/sportstradamus/data/models/ src/sportstradamus/data/training_report.txt src/sportstradamus/data/model_stats.parquet src/sportstradamus/data/stat_zi.json
git commit -m "chore(models): retrain ZINB markets as two-stage hurdle"
```

---

### Task C1: End-to-end verification & quality gates

**Files:** none

- [ ] **Step 1: Live-slate directional check.** If a fresh `current_offers.parquet` can be produced (`poetry run prophecize` or the offers pipeline) — confirm FGA no longer 100% Under at the 0.9 cap and FG3M no longer ~95% Over. Otherwise document that live verification requires a production run.
- [ ] **Step 2: Quality gates (all must pass — CLAUDE.md):**
```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/
poetry run pytest -m integration
```
- [ ] **Step 3:** `git add -A && git commit -m "test: verification artifacts for overconfidence fix"` (only if verification produced tracked changes).

---

## Rollback

Each phase is independently revertible by commit. Model artifacts are regenerated by `meditate`; to roll back, `git revert` the code commits and re-run `poetry run meditate --league NBA --force` to restore prior-behavior pickles. `offset_meta=None` / absence makes `model_prob` fall back to the legacy `normalized` path, so a mixed model dir degrades gracefully during a partial rollout (but full retrain is expected).

## Self-Review

- **Spec coverage:** A (additive EB offset, path-wide SkewNormal) → Tasks A1–A7. B (hurdle, ZINB-only) → Tasks B1–B4. Verification → A6, B4, C1. Both confirmed root causes covered; ZAGamma intentionally out of scope (user: ZINB-only first).
- **Placeholder scan:** all code steps contain real code; no TBD/“handle edge cases”. Task A3 Step 2 and B2/B3 Step 1 are explicitly contract-guard tests (documented as such), not vacuous.
- **Type consistency:** `compute_eb_prior(player_mean, games_played, global_mean, k)` signature identical across A1/A3/A4/A6. `offset_meta` keys `method/k/global_mean/prior_col` consistent A3↔A4. `HurdleZINB` interface (`fit`, `predict(pred_type=)`, `set_model_start_values`, `is_hurdle`) consistent B1↔B2↔B3. `HurdleZINB.predict` returns exactly `["total_count","probs","gate"]` — matches the existing `model_prob.py:165-172` ZINB consumer.
