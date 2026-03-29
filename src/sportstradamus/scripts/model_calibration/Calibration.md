# Model Calibration Investigation

This document records diagnostic findings about LightGBMLSS model calibration issues,
specifically why high-scale NBA markets (PTS, MIN, REB, AST, etc.) have `model_weight`
pushed to the 0.05 floor and see accuracy degrade after model correction.

---

## 1. The Symptom

Training report for NBA PTS (ZAGamma, n=2100 test obs):

```
              Accuracy  Over Prec  Under Prec  Over%  Sharpness    NLL
Raw Model        0.545      0.557       0.533  0.487      0.045  0.694
Corrected        0.529      0.540       0.518  0.487      0.047  0.720
Calibrated       0.529      0.540       0.518  0.487      0.032  0.720

DIAG model_weight=0.050  model_calib=0.647
DIAG start_alpha=4.179   model_alpha=3.698   empirical_alpha=2.312   shape_ratio=1.60
DIAG dispersion_cal=2.026
DIAG start_mean=12.36  model_ev=12.63  mean_line=12.57  result_mean=13.36
```

Accuracy falls from 0.545 (raw model) to 0.529 (after blending with model_weight=0.05).
The correction makes things *worse*.

---

## 2. Mean Level Diagnostics

### 2a. EV vs MeanYr vs Line vs Result (NBA PTS test set)

```python
# Computed from src/sportstradamus/data/test_sets/NBA_PTS.csv
Model EV mean:   10.90   std: 2.94
MeanYr mean:     12.36   std: 5.96
Book EV mean:    13.01   (from Blended_EV, which is near-100% book at w=0.05)
Line mean:       12.57
Result mean:     13.36
```

The model EV is the **lowest** of all estimates. The book line is a much better predictor
of actual results than the model.

### 2b. EV compression by player tier (NBA PTS)

```python
# pd.qcut(MeanYr, 5) groupby
          MeanYr  start_EV_clamped  model_EV   Result
VLow        5.5              6.8       8.3       8.4
Low         8.5              9.1      10.2      10.6
Mid        11.3             11.6      11.5      12.4
High       14.4             14.5      12.8      14.1
VHigh      22.0             22.0      11.7      21.3
```

The model predicts ~8–12 for everyone regardless of MeanYr. Start EV (derived from MeanYr
via clamped `(mu/std)^2`) correctly spans 6.8–22.0, but the model collapses this to 8.3–11.7.

### 2c. Compression ratios across NBA markets

```python
# Computed by loading each model pickle and comparing start_ev std vs model_ev std
market  dist      weight  compress  mean_shift  start_ev_mean  start_ev_std
BLK     NegBin    0.579   1.040     0.736       0.57           0.35
STL     NegBin    0.588   0.938     0.955       0.85           0.39
TOV     NegBin    0.900   0.764     0.903       1.39           0.79
FTM     NegBin    0.900   0.524     0.832       1.72           1.34
FGM     NegBin    0.149   0.746     0.658       3.95           1.96
REB     NegBin    0.074   0.886     0.580       4.56           2.14
MIN     Gamma     0.050   1.160     0.723       23.84          6.61
PTS     ZAGamma   0.050   0.524     0.850       12.81          5.62

# compress = model_ev.std() / start_ev.std()  (1.0 = no compression)
# mean_shift = model_ev.mean() / start_ev.mean()  (1.0 = no shift)
```

- **All markets** have `mean_shift < 1` (model systematically underpredicts vs start values)
- FTM has **identical compression** (0.52) as PTS but gets weight=0.90 because absolute
  scale is small (mean 1.72 vs 12.81)
- Compression is not the discriminator — absolute scale of the error is

### 2d. EV/MeanYr ratio paradox (NBA PTS)

```python
# Simple ratio of means: EV.mean() / MeanYr.mean()
Ratio of means:           0.8819   # model underpredicts by 12%

# Mean of per-observation ratios:
Mean of (EV/MeanYr):      1.0409   # looks fine if you average per-obs
```

These differ because high-MeanYr players contribute more to the mean but have EV/MeanYr
well below 1, while low-MeanYr players have EV/MeanYr > 1. Equal-weight averaging hides
the compression.

---

## 3. Distribution Parameter Diagnostics

### 3a. NBA PTS alpha/beta analysis

```python
# From test set CSV: Alpha column = model alpha, Gate column = model gate
# beta = alpha / EV (derived)
Model alpha:  mean=3.698   std=0.897   cv=0.243
Model beta:   mean=0.371   std=0.310   cv=0.834
Model EV:     mean=10.898  std=2.940   cv=0.270

# Expected from MeanYr/STDYr (pre-ceiling):
alpha_expected = (mu/std)^2  -- extremely variable, mean=332B (outliers from low-STD players)

# With ceiling=4.624 applied (as in training):
alpha_start_clamped: mean=3.424  std=1.211
beta_start:          mean=0.347  std=0.285
ev_start:            mean=12.807 std=5.615

# Rank correlations (model output vs expected):
Spearman(model_alpha, expected_alpha) = 0.9388   # well-ranked individually
Spearman(model_beta, expected_beta)   = 0.9395   # well-ranked individually
Spearman(model_EV, MeanYr)           = 0.4753   # ratio is poorly ranked
```

Alpha and beta are individually well-ranked, but their ratio (EV = alpha/beta) loses
dynamic range because compression of both parameters distorts the quotient.

### 3b. Tree residual direction (NBA PTS)

```python
# Computed as: softplus_inv(model_param) - softplus_inv(start_param)
tree_alpha_residual: mean=0.313  std=0.563
tree_beta_residual:  mean=0.253  std=0.498

Spearman(tree_alpha_residual, MeanYr) = -0.602  # trees push high-scorers alpha DOWN
Spearman(tree_beta_residual, MeanYr)  = +0.433  # trees push high-scorers beta UP
```

The tree residuals systematically work against player-level differentiation:
high-MeanYr players get lower alpha and higher beta → EV = alpha/beta collapses.

### 3c. Shape ceiling effect

```python
# NBA PTS stored shape_ceiling = 4.624
# This equals marginal_shape * 2 where marginal_shape comes from player_stats.quantile(0.95)

# 37% of players hit the alpha ceiling in start values:
at_ceiling = (alpha_start_raw > 4.624).mean()  # = 0.369

# Those players have alpha_expected ranging from 4.62 to 5.98e14
# All get clamped to 4.624, losing differentiation at the top end
```

The ceiling correctly reflects empirical distribution width, but it limits how much alpha
can vary, putting most of the EV work on beta alone.

---

## 4. Discrimination vs Calibration

### 4a. The model's actual accuracy source

```python
# Spearman rank correlation of (EV - Line) vs (Result - Line)
model_rank_corr = 0.1202
book_rank_corr  = 0.0506
# Model is 2.4x better at ranking who will beat their line

# EV > Line means model predicts "over"
model_EV_gt_line_pct   = 0.459  # model says "over" 45.9% of the time
book_EV_gt_line_pct    = 0.569  # book says "over" 56.9% of the time

# Conditional accuracy
acc_when_model_gt_line = 0.576  # 57.6% accurate when model EV > line
acc_when_book_gt_line  = 0.547  # 54.7% accurate when book line is used
```

The model knows *which* players will exceed expectations (good discrimination) but
its absolute mean is too low (poor calibration). Accuracy is a rank-based metric — it
rewards discrimination, not calibration. All proper scoring rules (NLL, CRPS, Brier)
evaluate both — so they penalize the mean bias and push `model_weight` to 0.05.

---

## 5. Feature Signal Analysis

### 5a. Residual predictability (Spearman of top feature vs Result − MeanYr)

```python
# For each market, best single-feature Spearman correlation with the residual
market  best_feature         best_corr  MeanYr→Result  Line→Result
PTS     Player MIN            0.218      0.535          0.590
REB     Player MIN            0.186      0.513          0.538
AST     ZeroYr                0.195      0.559          0.568
MIN     MeanYr/Player MIN     0.748      0.429          0.479
BLK     MeanYr                0.312      0.299          0.210
STL     MeanYr                0.202      0.234          0.173
TOV     Player position z     0.271      0.346          0.311
FTM     Player FT_PCT         0.371      0.382          0.370
```

PTS, REB, and AST have the weakest feature signal (max ~0.22). This means the model
has limited ability to predict game-specific deviations from historical means. BLK/STL/TOV
work better partly because the residuals are more driven by stable player characteristics.

### 5b. Existing PTS features (from feature_filter.json)

Common NBA features (35 total): Avg1/3/5/10/Yr, AvgH2H, Mean10/Yr, MeanH2H, STD10/Yr,
ZeroYr, DaysOff, DaysIntoSeason, GamesPlayed, H2HPlayed, Home, Moneyline, Total, Player z,
Player position, Player position z, Player home, Player moneyline gain, Player totals gain,
Defense avg/home/moneyline gain/totals gain/position/comps, Player comps mean/z,
Defense comp n/comp distance.

NBA PTS-specific features (24): Player AST_48, AST_PCT, AST_TO, BLKA_48, E_OFF_RATING,
FG3_RATIO, FGA_48, FG_PCT, FTR, FT_PCT, MIN, PCT_AST, PCT_BLKA, PCT_FG3A, PCT_FGA,
PCT_TOV, PFD, PIE, TOV_48, USG_PCT, age, depth, **proj MIN mean**, **proj MIN std**.

Note: `Player proj MIN mean` and `Player proj MIN std` are already included. The projected
minutes model is already being used as a feature for PTS.

---

## 6. Approaches Tested

### 6a. CRPS as `fit_model_weight` objective
**Outcome: w=0.05 for all high-scale markets**

CRPS is a marginal scoring rule. It evaluates overall distributional quality but cannot
detect per-observation conditional discrimination. A tight distribution centered on the
book's more accurate mean scores well on CRPS regardless of whether the model has better
discrimination. Reverted.

### 6b. Clamped NLL as `fit_model_weight` objective (current)
**Outcome: w=0.05 for PTS/MIN/REB/PA/PR/PRA/RA; w=0.14 for FGM**

Better than CRPS (per-observation evaluation preserves discrimination signal). But the mean
bias penalty (model consistently underpredicts high-scorers) still overwhelms the
discrimination benefit. The optimizer correctly identifies that reducing model weight reduces
the absolute error cost more than it loses the discrimination benefit.

### 6c. Debiasing before weight optimization
**Outcome (from test_model_weight.py::debias_data()): FGM w=0.416 acc=0.634; MIN w=0.90**

Scale model EVs by `book_ev_mean / model_ev_mean` before calling `fit_model_weight`.
Preserves rank ordering, removes absolute level bias. Improved several markets.
Not pursued further — user prefers root cause fix over post-hoc correction.

### 6d. Two-weight optimization (separate w_mean, w_shape)
**Outcome: marginal improvement, similar to debiasing**

### 6e. Dispersion calibration before blending
**Outcome: no improvement — problem is mean location, not shape**

---

## 7. Root Cause: Gradient Boosting Shrinkage

LightGBMLSS initializes each training observation with per-observation start values from
`MeanYr` and `STDYr` (see `helpers.py:set_model_start_values`). The boosted trees learn
**residuals in pre-response-function space** added to those start values.

Regularization compresses residuals toward zero:
- Learning rate multiplies every tree contribution (NBA PTS: lr=0.0046 — very conservative)
- L1/L2 penalize large leaf values (NBA PTS: L1=0.298, L2=0.120)
- `min_child_samples=137` prevents splits on small subgroups

Over 796 boosting rounds, the cumulative residuals cannot fully recover the per-player
variation in the start values. **This is a fundamental property of regularized gradient
boosting, not a bug.** It affects all GBDT distributional models.

The ratio parameterization (alpha/beta for Gamma, r*p/(1-p) for NegBin) amplifies this:
if both numerator and denominator are compressed toward their respective means, the ratio
loses range faster than either component individually. This explains why Gamma/ZAGamma
markets tend to suffer more than NegBin markets with low shape ceilings.

---

## 8. Open Questions

### 8a. Target normalization
Train on `Result / MeanYr` (or `Result / proj_MIN`) instead of raw `Result`. The model
would learn *relative* deviations from historical mean rather than absolute values. After
prediction, multiply the predicted relative deviation back by MeanYr to get the final EV.

This removes scale dependence entirely. The target range becomes ~0.3–3.0 for all markets
instead of 0–30+. The compression would still occur, but a 50% compression of a 0.3–3.0
target is far less damaging than a 50% compression of a 5–22 target.

Implementation: in `train.py`, before building `dtrain`:
```python
y_train_labels_raw = y_train_labels.copy()
mean_yr_train = X_train['MeanYr'].values
y_train_labels = y_train_labels / np.clip(mean_yr_train, 0.1, None)
# ... train model ...
# At prediction time: multiply model output by MeanYr feature
```

Risk: changes the NLL gradient shape; LightGBMLSS NLL for Gamma/NegBin is no longer
equivalent to the original. Need to verify the math still holds.

### 8b. Response function experiment
NBA PTS uses `softplus`. The `exp` response function (standard log-link) is scale-invariant
and might interact differently with per-observation init values. The trade-off is numerical
stability — exp produces larger gradients for large raw values.

### 8c. Hyperparameter tuning bias
NBA PTS Optuna result: lr=0.0046, rounds=796. This is at the low end of the search space
(lr range: 0.001–0.15). The 4-fold CV NLL objective may be systematically preferring very
low lr for high-scale markets because lower lr = more conservative trees = less chance of
severely mispredicting high-value outliers. Worth checking whether relaxing the search space
or changing the CV objective changes the selected hyperparameters.

### 8d. Additional features from nba_api
Features not currently in the model that could improve game-specific PTS predictions:
- **Play-type breakdown** (`playerdashptstats`): isolation %, PnR ball-handler %, post-up %
  — more stable player profiles for how they generate offense
- **Shot quality allowed by opponent** (`teamdashptshots`): how many open shots the defense
  allows at each zone — more precise than the generic DEF_RATING
- **Lineup context**: how much the player's minutes/usage changes when key teammates are
  out — injury-adjusted projections
- **Recent pace-adjusted stats**: scoring per 100 possessions vs raw scoring accounts for
  game tempo better than Total alone

All of the above are available historically via nba_api and cleaning the glass.

---

## 9. Data Files

Test set CSVs: `src/sportstradamus/data/test_sets/NBA_{MARKET}.csv`

Columns: `Result, Line, Blended_EV, EV, Gate` (ZI only), `Alpha` (Gamma/ZAGamma),
`R, NB_P` (NegBin/ZINB), `MeanYr, STDYr, ZeroYr`, all feature columns.

Model pickles: `src/sportstradamus/data/models/NBA_{MARKET}.mdl`

Keys: `model, step, stats, diagnostics, params, distribution, cv, std, temperature,
dispersion_cal, weight, r_book, hist_gate, shape_ceiling`
