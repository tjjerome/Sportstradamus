# SkewNormal + CRPS Implementation Summary

## Overview
Implemented SkewNormal distribution with CRPS loss for distributional ML models to address systematic compression bias. Markets with global mean ≥ 2 now use normalized SkewNormal targets, while low-mean markets (mean < 2) remain NegBin. Zero-inflation is handled via hurdle model (no ZASN class).

## Core Changes

### 1. train.py
- **Target Normalization**: Markets with `mean ≥ 2` now normalize targets as `Result / MeanYr ≈ 1.0` for SkewNormal training
- **Distribution Selection** (L490-526):
  - Check if original distribution is Gamma/ZAGamma and mean ≥ 2 → switch to SkewNormal + CRPS
  - NegBin/ZINB markets unchanged
  - Targets clipped to 0.01 minimum for continuous SkewNormal
- **Parameter Extraction** (L670-723):
  - Denormalize SkewNormal params: `loc_abs = loc * MeanYr`, `scale_abs = scale * MeanYr`
  - Compute EV: `EV = loc + sigma * delta * sqrt(2/π)` where `delta = alpha / sqrt(1+alpha²)`
  - Hurdle gate from `stat_zi.json` applied externally (no model gate parameter)
- **Model Blending** (L749-799):
  - SkewNormal-specific `fit_model_weight` call with `model_sigma` and `model_skew_alpha` params
  - Precision-weighted blend of loc/sigma, linear blend of alpha via `fused_loc`
- **Dispersion Calibration** (L801-879):
  - **SKIPPED for SkewNormal** — CRPS loss handles dispersion automatically
  - NegBin/Gamma still use post-hoc calibration
- **Shape Diagnostics** (L968-1002):
  - SkewNormal: report `scale` (sigma) instead of shape parameter
  - No counterfactual diagnostics for SkewNormal
- **Filedict** (L1051-1095):
  - Added `"normalized": normalize` flag for inference time denormalization
  - Store SkewNormal params: `SN_Loc`, `SN_Scale`, `SN_Alpha` in test predictions
- **Monotone Constraints** (L546-553):
  - SkewNormal loc constrained to increase with MeanYr
- **set_model_start_values** (L550, 603, 609, 615):
  - Pass `normalized=normalize` flag to initialize start values correctly
- **fit_model_weight** (L1460-1510):
  - New SkewNormal branch: uses `skewnorm.logpdf` for likelihood calculation
  - Hurdle model gate handling: only `gate_book` (no model gate)

### 2. sportstradamus.py (Inference Path)
- **Load Configuration** (L908-909):
  - Read `normalized` flag from filedict
  - Load `hist_gate` for SkewNormal markets (hurdle model)
- **set_model_start_values** (L927):
  - Pass `normalized=normalized` to set correct start values
- **Parameter Extraction** (L983-1000):
  - Denormalize SkewNormal params using `playerStats["MeanYr"]`
  - Compute EV from denormalized loc/scale/alpha
  - Store in prob_params: `Model EV`, `Model Sigma`, `Model Skew`, `Model Gate`
- **Book-Side Probabilities** (L1005-1010):
  - SkewNormal uses `sigma=ev*cv, skew_alpha=0` (books assumed symmetric Normal)
- **Blending** (L1022-1087):
  - SkewNormal-specific `fused_loc` call returning `(ev, sigma, alpha, gate_blend)`
  - Skip dispersion calibration for SkewNormal
- **Probability Computation** (L1089-1103):
  - SkewNormal get_odds call with `sigma` and `skew_alpha` parameters
  - Hurdle gate applied if `hist_gate > 0.02`
- **Display** (L1078-1079):
  - Convert Books EV from base mean to overall mean for SkewNormal with gate
- **Model Param** (L1151):
  - SkewNormal reports `Model Sigma` as dispersion parameter

### 3. helpers.py (Utility Functions)
- **get_ev** (L354-426):
  - SkewNormal branch (L406-416): Root-finding on `skewnorm.cdf` with `sigma = mean * cv`
  - Derives loc from mean and alpha: `loc = mean - sigma * delta * sqrt(2/π)`
  - Supports `skew_alpha` parameter (defaults to 0 for Normal)

- **get_odds** (L428-514):
  - SkewNormal branch (L488-500): CDF computation via `skewnorm.cdf`
  - Supports hurdle gate: `CDF_hurdle = gate + (1-gate) * base_CDF`
  - Book side uses `skew_alpha=0` (symmetric Normal)

- **fused_loc** (L978-1077):
  - SkewNormal branch (L1041-1067):
    - Precision-weighted blend: `prec = 1/sigma²`
    - Derives book loc from EV (book alpha=0 → loc = EV)
    - Linear blend of alpha: `blended_skew = w * model_skew` (book alpha=0)
    - Returns: `(blended_ev, blended_sigma, blended_skew, gate_blend)`
  - Hurdle gate handling: use `gate_book` directly if no model gate

- **set_model_start_values** (L1134-1180):
  - SkewNormal branch: if `normalized=True`, start with `loc=1.0, scale=cv, alpha=0`
  - Raw space: `[loc, log(scale), alpha]` matching model parameter order

- **fit_model_weight** (L1460-1510):
  - SkewNormal branch: uses `skewnorm.logpdf` to compute log-likelihood
  - Hurdle gate: only `gate_book` (no model gate parameter)
  - Maximizes likelihood via `scipy.optimize.minimize`

### 4. Migration Script: migrate_archive_to_skewnorm.py
- Converts stored EVs from old Gamma/NegBin encoding → SkewNormal (alpha=0) encoding
- For each stored EV:
  1. **Decode**: Use `get_odds` with old distribution to recover implied under-probability
  2. **Re-encode**: Call `get_ev` with `dist="SkewNormal", skew_alpha=0` to get new EV
- Markets eligible for migration defined in `SKEWNORM_MARKETS`
- Preserves NaN/invalid EVs unchanged
- Summary stats: converted, skipped (no change), skipped (bad data), errors
- **Uncomment `archive.write(all=True)` to persist changes**

## Key Design Decisions

### Hurdle Model (vs ZASN)
- **Rationale**: `torch.bernoulli()` in gate rsample blocks autograd under CRPS loss
- **Solution**: No gate parameter in LightGBMLSS model; apply gate externally from `stat_zi.json`
- **Implementation**: `gate_model=None` in all SkewNormal fit_model_weight/fused_loc calls

### Target Normalization
- **Why**: Normalized targets ≈1.0 stabilize SkewNormal training; loc learns additive corrections with no nonlinear compression
- **Conditions**: Only for Gamma/ZAGamma markets with mean ≥ 2 (low-mean markets remain NegBin)
- **Denormalization**: `loc_abs = loc * MeanYr`, `scale_abs = scale * MeanYr` at inference; alpha unchanged (dimensionless)

### Archive Migration
- **Book Assumption**: Books encode using symmetric Normal (alpha=0)
- **Encoding**: Round-trip through implied under-probability preserves consistency across distributions
- **Reversibility**: Can be run in dry-run mode (comment out `archive.write()`) to verify before committing

## Testing Recommendations

1. **Unit Tests**:
   - Run `python3 -m pytest` on get_ev, get_odds, fused_loc with SkewNormal examples
   - Verify denormalization: `loc_abs * (1 + scale_abs * delta * sqrt(2/π) / loc_abs) ≈ EV`

2. **Integration Tests**:
   - Run `poetry run meditate --league NBA --force` on a subset (e.g., just PRA market)
   - Check filedict contains `"normalized": True` and `"distribution": "SkewNormal"`
   - Verify test set predictions include `SN_Loc`, `SN_Scale`, `SN_Alpha` columns

3. **Compression Ratio Validation**:
   - Before: Compare (model EV - global mean) std vs (results - global mean) std
   - After: Should improve from ~2.5x compression to ~1.25-1.5x for SkewNormal markets

4. **Archive Migration**:
   - Dry run: `python3 migrate_archive_to_skewnorm.py` (no write)
   - Inspect stats output for converted/skipped counts
   - Spot-check a few EVs before uncommenting `archive.write()`

## Files Modified

- `src/sportstradamus/train.py`: Training pipeline (60+ line changes)
- `src/sportstradamus/sportstradamus.py`: Inference path (90+ line changes)
- `src/sportstradamus/helpers.py`: Utility functions (already modified in prior session)
- `src/sportstradamus/scripts/migrate_archive_to_skewnorm.py`: NEW archive migration script

## Not Implemented

- Parlay combinations (player + player) with SkewNormal — currently skips to fallback if dist is SkewNormal
- Explicit ZASN class — hurdle model achieves same goal without gate autograd issues
- Backwards compatibility with old models — new filedict format includes `normalized` flag

## Next Steps

1. Run `poetry run meditate --force` on full dataset to retrain all markets
2. Execute `python3 src/sportstradamus/scripts/migrate_archive_to_skewnorm.py` to update stored EVs
3. Run `poetry run prophecize` to verify inference path works end-to-end
4. Monitor compression ratio and calibration metrics in training reports
