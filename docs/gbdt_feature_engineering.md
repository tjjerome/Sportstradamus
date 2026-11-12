# Feature Engineering, Preprocessing and Data Strategy for Gradient-Boosted Trees: An Evidence-Based Guide

For a GBDT, most of the gain comes from three things. First, a validation scheme that matches how the model will be used. Second, features that hand the trees relationships they cannot cheaply build from axis-aligned splits: ratios, group aggregates, lags and leak-free encodings. Third, honest diagnostics (learning curves and residual slicing) that tell you whether the bottleneck is features, rows or label noise. Classic preprocessing like scaling mostly does nothing for trees. "More data" helps only when it is relevant to the deployment distribution and adds information the model does not already have.

## TL;DR

- **Engineer what trees can't synthesize, then prune with held-out tests.** Trees approximate functions with piecewise-constant, axis-aligned splits. Ratios, differences of ratios, group-by statistics, lags/rolling windows and out-of-fold target encodings therefore often pay off. Monotonic transforms of inputs and scaling do not. Select features with permutation or null-importance tests on held-out data, never with gain importance alone, and always inside the resampling loop.
- **Preprocess the target and the rows, not the feature scales.** Log or otherwise transform skewed regression targets, deduplicate, remove leaky or ID-like columns, and leave NaNs as NaN, since XGBoost and LightGBM learn a default branch for missing values. Use native categorical handling or cross-fitted target encoding rather than wide one-hot for high-cardinality columns.
- **Decide "more data vs. more features" empirically.** Plot error against training-set size. If validation error is still falling on a roughly power-law slope and the train/validation gap is large, more rows of the same kind will help. If the curve is flat and both errors are high, you need new *information*: orthogonal sources, better labels or new features. Choose which data to acquire by slicing residuals, checking what a candidate signal explains about current residuals, and running small pilots before buying anything at scale.

## Key Findings

1. **Trees remain the default for medium-sized tabular data, but the frontier has moved.**
   - Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022 Datasets & Benchmarks) built a benchmark of 45 datasets. They found tree-based models "remain state-of-the-art on medium-sized data (∼10K samples) even without accounting for their superior speed."\[1\]\[2\]
   - Shwartz-Ziv & Armon (Information Fusion, 81:84–90, 2022) compared XGBoost with TabNet, NODE, DNF-Net and 1D-CNN: "For 8 of the 11 datasets, XGBoost outperformed" them. Average relative performance deterioration was 3.34 for XGBoost vs. 7.56 (1D-CNN), 10.51 (TabNet), 11.96 (DNF-Net) and 14.21 (NODE).
   - McElfresh et al. (NeurIPS 2023) compared 19 methods on 176 datasets. GBDTs did better on "irregular" datasets (skewed or heavy-tailed feature distributions), on larger datasets, and where the ratio of rows to features is high. For a "surprisingly high fraction of datasets," light tuning of a GBDT mattered more than choosing the best neural net.\[3\]\[4\]
   - TabArena (Erickson et al., NeurIPS 2025) curated 51 datasets from 1,053 candidates and evaluated 16 curated models, including 3 tabular foundation models; the authors "trained ∼25 000 000 instances of models." Its finding: "gradient-boosted trees are still strong contenders on practical tabular datasets," deep learning "caught up under larger time budgets with ensembling," and "foundation models excel on smaller datasets."
2. **Tabular foundation models are a real alternative on small data, and a useful baseline.**
   - TabPFN (Hollmann et al., *Nature* 637:319–326, 2025) targets datasets under about 10,000 samples. A University of Freiburg press release (9 January 2025, team led by Prof. Frank Hutter) says "TabPFN requires only 50% of the data to achieve the same accuracy as the previously best model," especially for small tables with fewer than 10,000 rows.
   - TabPFN-2.5 and TabPFN-3 technical reports (2025–2026) claim leading TabArena results, but those reports come from the vendor (Prior Labs).\[5\]\[6\] Treat them as promising until they are independently replicated.
   - For this guide, the practical point is that if a TabPFN-class model beats your engineered GBDT on a small dataset, your features are probably not adding much.
3. **Trees are robust to uninformative features, but not immune.**
   - Grinsztajn et al. found a GBT's accuracy "is not much affected by removing up to half of the features," and that MLPs are much less robust to such features.\[1\]
   - Kuhn & Johnson's simulation (*Feature Engineering and Selection*, §10.3) found that boosted trees "showed a moderate decrease in performance as predictors were added." It also found that tree ensembles "show poor results for irrelevant predictors by selecting many of the noise variables."\[7\]
4. **Leakage in encodings and in feature selection is the most common way to fool yourself.**
   - CatBoost's authors (Prokhorenkova et al., NeurIPS 2018) show that naive target statistics cause a "prediction shift caused by a special kind of target leakage." Their fix is ordered target statistics plus ordered boosting.\[8\]
   - Kuhn & Johnson cite a reanalysis (Ambroise & McLachlan, 2002) of SVM recursive feature elimination. Leave-one-out error was near zero, but true held-out error was 15–20% higher, and "a zero LOO error rate could [still] be achieved even when the class labels were scrambled."\[9\]
5. **Learning curves are the right instrument, but they are not always well-behaved.**
   - Hestness et al. (Baidu Research, 2017) studied machine translation, language modeling, image processing and speech recognition, and describe three regions in deep-learning learning curves: small-data, power-law and irreducible-error. They report that the exponent "usually settles between −0.07 and −0.35, exponents that are unexplained by prior theoretical work" (theory had predicted −0.5 or −1).
   - For tabular classifiers, the LCDB 1.1 benchmark (NeurIPS 2025) found that "learning curves are less often well-behaved than previously thought."\[10\] Extrapolate cautiously.

## Details

### 1. Techniques that actually help GBDTs, and why

**The mechanism.**
- Each tree splits on one feature at a time against a threshold. The ensemble is therefore a sum of axis-aligned, piecewise-constant functions.
- Anything that is a smooth function of *several* features takes many splits to approximate, and so needs a lot of data. Examples are a ratio x1/x2, a difference, a distance, or "this value relative to its group's typical value."
- The same mechanism means monotonic one-to-one transforms of a single input (log, sqrt, standardization) do not change which partitions a tree can form. That is why they rarely help.
- Grinsztajn et al. frame tree models' advantage as not being rotationally invariant and coping well with "irregular" target functions. Rotating the data reversed the ranking in their experiments.\[1\] The corollary: features that put the signal on a natural axis help trees.

**Arithmetic and interaction features.**
- Heaton (SoutheastCon 2016) created synthetic targets from engineered formulas and tested whether models could learn them. "Engineered features based on a ratio of differences were not synthesized well by any of the models," gradient boosting included.\[11\]
- Domain-motivated ratios are the classic win. Examples are BMI, debt-to-income, power = V·I, specific energy, and normalized errors. For a controls engineer, physically meaningful dimensionless groups (Reynolds-number-like quantities, error/setpoint, rate × dt) are the high-prior candidates.
- Categorical crosses are a cheap version of the same idea. The NVIDIA Kaggle Grandmasters playbook (Onodera, Viel, Titericz, Deotte, 2025) describes taking 8 categorical columns and building "28 new categorical features" from all pairs.\[12\]

**Group-by aggregations.**
- Chris Deotte (NVIDIA, 1st place in a 2025 Kaggle Playground competition) calls `groupby(COL1)[COL2].agg(STAT)` "the most powerful feature engineering technique."\[13\]
- In the IEEE-CIS fraud competition, the features that "made the difference" for the winning team were "new columns created from group aggregations of other columns."\[14\] Typical aggregates are a transaction amount relative to that card's mean, counts per entity, and per-group std, quantiles or histogram bins.
- The reason it works: a row-level model cannot see the context of the row's group unless you give it to the model.
- Caution: when the target's statistics are used, or when the groups cross the train/validation boundary in time, compute the aggregates within folds.

**Lag, rolling and time-window features (temporal data).**
- Lags, rolling means/stds/min/max over several windows, time since last event, and exponentially weighted averages turn a sequence problem into a tabular one.
- The non-negotiable rule is that every window ends strictly before the prediction timestamp. Validate with forward-chaining splits; the NVIDIA playbook recommends `TimeSeriesSplit` for time-dependent data and `GroupKFold` for grouped data.\[12\]
- This is practitioner consensus rather than a benchmarked result. TabArena explicitly excludes non-IID and temporal tasks.\[15\]

**Date/time decomposition.** Hour, day-of-week, month, holiday flags, and "time since a reference event" let trees split on cyclic or calendar effects. Unlike neural nets, trees do not need sin/cos encodings to handle cyclic features, but those encodings do no harm.

**Categorical encoding.**
- *Native handling:*
  - LightGBM partitions categories using an approach based on Fisher (1958), which "often performs better than one-hot encoding." Use `min_data_per_group` and `cat_smooth` against overfitting.\[16\]
  - LightGBM's own docs say that for high-cardinality features "it often works best to treat the feature as numeric… or by embedding the categories in a low-dimensional numeric space."\[17\]
  - Categories unseen during training are treated as missing at `predict()`.\[18\]
  - A categorical feature cannot carry a monotone constraint.\[19\]
- *Target/mean encoding:*
  - Must be computed out-of-fold. scikit-learn's `TargetEncoder.fit_transform` "uses a cross fitting scheme to prevent target leakage." Its docs warn that `fit(X, y).transform(X)` "is discouraged… because it can introduce data leakage."\[20\]
  - With `smooth="auto"` it shrinks toward the global mean with an empirical-Bayes estimate. It treats NaN as its own category and encodes unseen categories as the global target mean.\[20\]\[21\]
  - CatBoost's ordered target statistics compute each row's encoding only from rows "before" it in a random permutation, which avoids the conditional shift. The original paper reports that ordered TS beat greedy, holdout and leave-one-out TS.\[22\]\[23\]
  - Be careful with third-party encoders. Their `fit_transform` may *not* cross-fit.\[24\]
- *Frequency/count encoding:* replace a category with how often it occurs. It needs no target, so it cannot leak the target, and it often helps for IDs. Heaton reports trees have difficulty synthesizing count features on their own.\[25\]

**Missing-value indicators.** XGBoost and LightGBM already route NaNs to a learned branch (see §2). An explicit `is_missing` flag is mostly redundant for those features. It is useful when you impute for some other reason, or when missingness of a *combination* of fields is informative.

**Constraints as feature engineering.**
- XGBoost's monotone constraints (e.g., `monotone_constraints="(1,-1)"`) build domain knowledge into the model and reduce spurious wiggles. The docs warn that with `hist`/`approx` they "may produce unnecessarily shallow trees."\[26\]\[27\]
- Interaction constraints exist because deep trees let variables interact "on the sole basis of minimizing training loss," which "may capture a spurious relationship (noise)."\[28\] Use them when physics or business logic says two groups of variables act additively.
- An independent paper (Monotone GAMI, arXiv 2309.02426) reports that XGBoost's interaction-constraint documentation misdescribes how overlapping constraint sets behave.\[29\] Check what your trees actually do.

**Out-of-fold target encoding (manual, works with any GBDT):**
```python
import numpy as np, pandas as pd
from sklearn.model_selection import KFold

def oof_target_encode(train, test, col, target, n_splits=5, smooth=20, seed=0):
    prior = train[target].mean()
    oof = np.full(len(train), np.nan)
    for tr_idx, va_idx in KFold(n_splits, shuffle=True, random_state=seed).split(train):
        stats = train.iloc[tr_idx].groupby(col)[target].agg(['sum', 'count'])
        enc = (stats['sum'] + smooth * prior) / (stats['count'] + smooth)
        oof[va_idx] = train.iloc[va_idx][col].map(enc).fillna(prior).values
    stats = train.groupby(col)[target].agg(['sum', 'count'])  # full-train fit for test
    enc = (stats['sum'] + smooth * prior) / (stats['count'] + smooth)
    return oof, test[col].map(enc).fillna(prior).values
# Must be nested INSIDE your outer CV loop when you evaluate, or use time-ordered folds for temporal data.
```

**Feature selection and pruning.**
- *Gain/split importance:*
  - These are in-sample and biased. scikit-learn's docs state impurity-based importance is "strongly biased and favor high cardinality features." In their example, a purely random numeric column ranks as the most important feature.\[30\]\[31\]
  - Split-count importance has the same problem.
  - Kuhn & Johnson add that duplicating a predictor "will numerically dilute the importance scores," so correlated features split credit between them.\[7\]
- *Permutation importance on held-out data:* this is the default to use. Features "deemed of low importance for a bad model… could be very important for a good model," so evaluate the model first.\[32\] With correlated features, permuting one feature understates it because its twin still carries the signal.
- *SHAP:* mean |SHAP| is consistent within the model and useful for diagnostics and interactions. It is still *in-model* attribution, not a test of generalization. Use it to explain the model, and use held-out permutation or ablation to decide what to drop.
- *Null importance / target permutation (PIMP):* Altmann, Toloşi, Sander & Lengauer (Max Planck Institute for Informatics; *Bioinformatics* 26(10):1340–1347, 2010) permute the outcome many times, refit, and form a null distribution of each feature's importance. "The P-value of the observed importance provides a corrected measure." A later analysis by Nembrini (*Bioinformatics* 35(15):2701–2705, 2019, "On what to permute in test-based approaches for variable importance measures in Random Forests") shows that "permuting the outcome may produce unexpected results, including p-values with undesirable properties," and that the scheme can have low power when many predictors are associated with the target.
- *Boruta:* Kursa & Rudnicki (*J. Stat. Software* 36(11), 2010) add shuffled "shadow" copies of every feature and "iteratively remov[e] the features which are proved by a statistical test to be less relevant than random probes." It aims for *all-relevant* selection, not a minimal set.\[33\]\[34\]
- *Adversarial validation:*
  - Train a classifier to tell train rows from test or production rows. AUC ≈ 0.5 means the two sets look alike. A high AUC, together with that classifier's top features, points to the drifting columns.\[35\]
  - An Uber paper (arXiv 2004.03045) uses this for concept drift. A credit-scoring study (arXiv 2112.10078) measured an adversarial AUC of 0.9681 between its train and test periods on Lending Club data.\[35\]\[36\]
- *Do selection inside resampling.* Kuhn & Johnson: "feature selection needs to be part of the resampling process… The most common mistake is to only conduct resampling inside of the feature selection procedure." Their own caveat is that large datasets "greatly reduce the risk."\[9\]

**Null-importance test (LightGBM):**
```python
import numpy as np, lightgbm as lgb
def importances(X, y, shuffle, seed):
    y_ = np.random.default_rng(seed).permutation(y) if shuffle else y
    m = lgb.LGBMRegressor(n_estimators=300, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1)
    m.fit(X, y_)
    return m.booster_.feature_importance(importance_type='gain')
actual = np.mean([importances(X, y, False, s) for s in range(5)], axis=0)
null = np.array([importances(X, y, True, s) for s in range(50)])
pval = (null >= actual).mean(axis=0)          # empirical p-value per feature
keep = X.columns[pval < 0.05]
# Then confirm with held-out ablation: drop the candidates and re-run CV.
```

**Adversarial validation:**
```python
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
Xa = pd.concat([X_train, X_test]); ya = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
clf = lgb.LGBMClassifier(n_estimators=300, verbose=-1)
p = cross_val_predict(clf, Xa, ya, cv=5, method='predict_proba')[:, 1]
print('adversarial AUC', roc_auc_score(ya, p))   # ~0.5 = similar distributions
clf.fit(Xa, ya)
drift = pd.Series(clf.booster_.feature_importance('gain'), Xa.columns).sort_values(ascending=False)
# Inspect top drifting features: drop, re-define (e.g. use relative instead of absolute), or reweight.
```

### 2. Should you preprocess? What matters and what doesn't

| Step | Needed for GBDT? | Why |
|---|---|---|
| Scaling / standardization / normalization of features | No | Split thresholds don't depend on scale. Exceptions: `linear_tree`, or if you also ensemble with linear models, NNs or kNN. |
| Log / Box-Cox of *input* features | Usually no | Monotone transforms keep the order of values, so the same partitions are available. Small effects can come from histogram bin placement, but expect little. |
| Transform of a skewed *regression target* | Often yes | Grinsztajn et al. log-transformed heavy-tailed targets (e.g., house prices) and searched over Gaussianizing the target.\[1\] Optimize the loss you care about: training on log(y) minimizes something closer to relative error, and the back-transformed mean is biased low. Alternatives are Tweedie/Poisson/Gamma objectives. |
| Target outliers / label errors | Yes, investigate | Squared loss is driven by extreme targets. Fix errors, or use Huber/quantile losses. |
| Deduplication | Yes | Duplicates that land in both train and validation inflate CV. TabArena treated "duplicate instances" as a curation defect.\[37\] |
| Missing values | Leave as NaN (usually) | XGBoost learns "the best direction to handle missing values" per split (sparsity-aware split finding; over 50× faster than a naive implementation on sparse data). LightGBM uses NaN by default, with `zero_as_missing=true` to also treat zeros as missing.\[38\]\[39\]\[40\] Don't encode missing as 0 or −999 unless that matches reality. |
| Binning (`max_bin`) | Rarely a tuning lever | Histogram methods bucket features. Raising the bin count can help features with fine-grained signal, at the cost of memory and speed. |
| Class imbalance | Metric-dependent | For AUC/log-loss, keep the natural distribution and calibrate. Use `scale_pos_weight`/class weights only when the decision threshold or metric needs it (practitioner heuristic). |
| ID-like, leaky, post-outcome columns | Remove | High-cardinality IDs memorize; post-event fields leak. Adversarial validation and "too-good-to-be-true" importance help find them. |
| Categoricals | Native or cross-fitted TE | See §1. Avoid one-hot for high cardinality with trees. |

### 3. Is there such a thing as too much data?

**Too many rows: rarely harmful on their own; harmful when irrelevant.**
- More IID rows almost never hurt a well-regularized, early-stopped GBDT. The cost is compute and diminishing returns along a learning curve.
- Rows *do* hurt when they come from a different distribution than deployment: old regimes, other sites, changed instrumentation, policy changes. The model then spends capacity fitting relationships that no longer hold.
- Use adversarial validation between old and recent data to measure the shift.
- Then compare, on a forward-in-time validation window: (a) training on all history, (b) training on a recent window, and (c) recency weighting, e.g. sample weights w = exp(−age/τ) with τ tuned on the validation window.
- This is a practitioner procedure. No benchmark gives a universal window size. Note that the major benchmarks cited here (Grinsztajn, TabArena) deliberately exclude time-series and non-IID data, so they say little about drift.\[15\]

**Too many features: trees cope, but you pay.**
- Grinsztajn et al. found trees are much more robust than MLPs to uninformative columns.\[1\] Kuhn & Johnson's simulation still shows boosted trees losing accuracy as noise predictors are added, and tree ensembles selecting many noise variables.\[7\]
- McElfresh et al. found GBDTs do relatively better when the row-to-feature ratio is high. That suggests very wide, short tables erode their advantage.\[3\]
- The mechanism: with p candidate features at every split, the chance that some noise feature *looks* good on a finite sample grows with p. This is multiple comparisons inside the split search.
- Countermeasures are column subsampling, `min_child_samples`/`min_data_in_leaf`, stronger regularization, and the selection procedures in §1.
- Mass feature generation (the Grandmaster approach) is safe only with a validation set you trust and fold-correct evaluation of each batch of features.

**Label noise** caps achievable accuracy and makes the model fit the noise, especially with deep trees and many rounds. Ng's data-centric argument applies directly: cleaning labels can beat adding rows.\[41\]

**Compute** grows with rows × features × rounds. GPU training, histogram binning and early stopping usually make this manageable before it becomes a statistical problem.

### 4. When to look for more data instead of more feature engineering

**Learning curve: train on nested subsets of increasing size and plot train and validation error.**
```python
import numpy as np, lightgbm as lgb
from sklearn.model_selection import learning_curve, KFold
sizes, tr, va = learning_curve(
    lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1),
    X, y, train_sizes=np.geomspace(0.05, 1.0, 8),
    cv=KFold(5, shuffle=True, random_state=0),   # use TimeSeriesSplit for temporal data
    scoring='neg_root_mean_squared_error', n_jobs=-1)
tr_err, va_err = -tr.mean(1), -va.mean(1)
slope = np.polyfit(np.log(sizes[-4:]), np.log(va_err[-4:]), 1)[0]  # local power-law exponent
print(sizes, va_err, 'local log-log slope:', slope)
```

**How to read it:**
- **Validation error still falling on a log-log line, large train/validation gap (variance-limited).** More rows of the same kind will help. Fit the local slope and extrapolate to estimate how much doubling the data would buy. Hestness et al. found such power laws in deep learning, with "model improvements only shift[ing] the error but [not] the power-law exponent."\[42\] That is a useful mental model, but it was not established for GBDTs.
- **Validation error flat, train and validation error both high and close together (bias- or information-limited).** More rows won't help. You need new features, a more flexible model, or new *information* sources.
- **Validation error flat, train error far below it, gap not closing with size.** Overfitting to noise or leakage. Regularize, clean labels, and check for duplicates or leaks.
- **Curve flat near a floor that a much stronger or different model also hits.** You are probably near the noise floor of these inputs. Estimate it from repeated measurements, inter-rater disagreement, or sensor noise specifications. This is often far easier in controls and robotics, where you know sensor noise, than in business data.

**Caveat:** LCDB 1.1 (Mohr, van Rijn et al., NeurIPS 2025) found tabular learning curves are often not monotone or convex.\[10\] Extrapolate only from the last several points, with CV error bars.

**Signs feature engineering is saturating:**
- Successive feature batches give gains within fold-to-fold noise. Measure that noise with repeated CV over several seeds.
- Gains on CV don't transfer to a later time window.
- A strong out-of-the-box alternative matches your engineered model. Examples are TabPFN on small data, or a tuned and ensembled AutoGluon.

**Data-centric framing.** Andrew Ng (IEEE Spectrum, 2022): "Data-centric AI is the discipline of systematically engineering the data needed to successfully build an AI system… it's now more productive to hold the neural network architecture fixed, and instead find ways to improve the data." He also warns: "Collecting more data often helps, but if you try to collect more data for everything, that can be a very expensive activity."\[41\] That second line is the bridge to §5.

### 5. Deciding what new data would be most useful

1. **Slice the residuals.**
   - Group out-of-fold errors by segment (operating regime, site, device, customer type, time-of-day, feature quantile bins) and rank segments by error × volume (or × business cost).
   - High error with low count suggests collecting more rows *in that region*.
   - High error with high count suggests the features can't separate cases there, so you need new *information*.
2. **SHAP diagnostics.**
   - SHAP dependence plots show where the model leans on a feature in a region with sparse support. Large SHAP values on few rows warn that the model is extrapolating.
   - SHAP interaction values can reveal interactions worth encoding explicitly or constraining.
3. **Find underrepresented regions.** Train a density or "domain" classifier on deployment inputs versus training inputs (adversarial validation again). Deployment rows the classifier finds easy to separate are where training coverage is thin.
4. **Uncertainty/active learning.**
   - Use disagreement across seeds, folds or quantile models as an uncertainty proxy, and label the most uncertain or most impactful candidates first.
   - Data Shapley (Ghorbani & Zou, ICML 2019) assigns each training point its marginal contribution to performance.\[43\] Low or negative values flag likely mislabeled or harmful points, and high values show which kinds of data help.
   - It is expensive; Monte-Carlo approximations exist. Use it for diagnosis, not routinely.
5. **Prefer orthogonal information over more of the same.** A new source helps only if it explains variance the current features don't. Cheap pre-acquisition test: obtain the candidate signal for a sample, then measure how well it predicts the *current out-of-fold residuals*.
```python
import numpy as np, lightgbm as lgb
from sklearn.model_selection import cross_val_predict, KFold
resid = y - oof_pred                                    # from your current CV pipeline
cv = KFold(5, shuffle=True, random_state=0)
r_hat = cross_val_predict(lgb.LGBMRegressor(n_estimators=200, verbose=-1),
                          Z_candidate, resid, cv=cv)    # Z_candidate: new source on the pilot rows
gain = 1 - np.mean((resid - r_hat)**2) / np.mean(resid**2)
print('fraction of residual variance explained by candidate:', gain)
# Confirm with a full ablation: retrain with [X, Z] vs X on the same folds; compare with repeated-CV noise.
```
   - If the candidate cannot explain residual variance even nonlinearly, it is unlikely to help much.
   - If it can, run the full ablation on the pilot subset before acquiring at scale.
   - For "more rows" decisions, the learning-curve slope gives the expected gain per added row. For "new source" decisions, the residual test plus ablation does the same job.

## Recommendations: an ordered workflow

1. **Define the deployment question and split accordingly:** time-forward for temporal data, GroupKFold for entities, random K-fold only for truly IID data. Hold out a final untouched test window.
2. **Run a leakage audit:** remove IDs, post-outcome fields and duplicates. Run adversarial validation between train and test or recent periods.
3. **Build a baseline:** a default LightGBM/XGBoost/CatBoost with NaNs left as-is, native categoricals, and early stopping. For a dataset under about 10k rows, also run TabPFN as a reference. Record mean ± std over repeated CV.
4. **Handle the target:** choose a loss matching the business metric, and try a log or Tweedie target for skewed regression.
5. **Feature engineering, in priority order:**
   - domain ratios and differences;
   - group-by aggregates;
   - lags and rolling windows (temporal);
   - count encodings;
   - cross-fitted target encodings or CatBoost;
   - categorical crosses.
   
   Evaluate in batches, with all target-dependent steps inside the folds.
6. **Add constraints:** monotone constraints where physics or business logic dictates; interaction constraints where additivity is known.
7. **Prune:** held-out permutation importance plus null-importance or Boruta, confirmed by drop-column CV. Don't trust gain importance.
8. **Diagnose:** draw the learning curve; slice residuals; check SHAP for low-support extrapolation.
9. **Decide:**
   - Still on a power-law slope with a big gap: get more rows like the current ones, targeted at high-error segments.
   - Flat with both errors high: find new orthogonal signals and test them with the residual-explanation pilot.
   - Flat near the noise floor: stop. Improve labels or accept the limit.
10. **Handle drift:** compare all-history, recent-window and recency-weighted training on a forward window. Monitor adversarial AUC in production.
11. **Only then tune and ensemble.** TabArena found that post-hoc ensembling of tuned configurations gives the best individual-model results.\[15\]

### Most useful references
- Grinsztajn, Oyallon, Varoquaux (2022), "Why do tree-based models still outperform deep learning on tabular data?" (arXiv 2207.08815)
- McElfresh et al. (2023), "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" (arXiv 2305.02997)
- Erickson et al. (2025), "TabArena: A Living Benchmark for Machine Learning on Tabular Data" (arXiv 2506.16791)
- Hollmann et al. (2025), "Accurate predictions on small data with a tabular foundation model," *Nature* 637:319–326
- Prokhorenkova et al. (2018), "CatBoost: unbiased boosting with categorical features" (arXiv 1706.09516)
- Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System" (arXiv 1603.02754)
- Kuhn & Johnson, *Feature Engineering and Selection* (bookdown.org/max/FES), chapters 10–12
- LightGBM "Advanced Topics" and XGBoost monotonic/interaction-constraint docs; scikit-learn `TargetEncoder` and permutation-importance guides
- Altmann et al. (2010), "Permutation importance: a corrected feature importance measure," *Bioinformatics*; Kursa & Rudnicki (2010), Boruta, *JSS*
- NVIDIA "Kaggle Grandmasters Playbook" (2025) and Deotte's cuDF feature-engineering write-up
- Hestness et al. (2017), "Deep Learning Scaling is Predictable, Empirically"; Mohr et al., LCDB 1.0/1.1
- Ghorbani & Zou (2019), "Data Shapley," ICML

## Caveats

- **Benchmark scope.** Grinsztajn et al. removed missing data, balanced classes, dropped categoricals with more than 20 levels, and truncated training sets to 10,000 rows.\[1\] TabArena v0.1 covers only IID data with no temporal tasks.\[15\] Their conclusions may not transfer to your drifting, high-cardinality, time-indexed data.
- **Practitioner heuristics vs. evidence.** These are well-established practice but lack controlled benchmark evidence specific to GBDTs:
  - group-by aggregates being the "most powerful" technique;
  - recency weighting;
  - class-weighting advice;
  - residual-correlation pilots.
  
  Kaggle results are real but specific to their competitions.
- **Vendor claims.** TabPFN-2.5/3 performance claims come from Prior Labs' own technical reports. The 2025 *Nature* paper is peer-reviewed, and the later versions are not.
- **Power laws.** Power-law scaling with data is well documented for deep learning.\[44\] For GBDTs on tabular data, learning curves are often irregular (LCDB 1.1), so treat extrapolation as a rough estimate.
- **Importance methods all have failure modes:**
  - correlated features split or mask importance;
  - permutation creates unrealistic rows;
  - outcome-permutation null tests can be low-power.
  
  Final decisions should rest on drop-column CV with repeated seeds.
- **Documentation drift.** Library behaviour changes between versions (e.g., LightGBM's handling of ordered pandas categoricals; XGBoost's interaction-constraint semantics). Pin versions and verify.\[45\]

## Sources

1. <https://arxiv.org/html/2207.08815v1>
2. [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)
3. [When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://scispace.com/pdf/when-do-neural-nets-outperform-boosted-trees-on-tabular-data-4l9j6096.pdf)
4. [When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://arxiv.org/pdf/2305.02997)
5. [TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models](https://arxiv.org/pdf/2511.08667)
6. [A new performance standard.](https://arxiv.org/html/2605.13986v2)
7. <https://bookdown.org/max/FES/feature-selection-simulation.html>
8. [CatBoost](https://dl.acm.org/doi/abs/10.5555/3327757.3327770)
9. <https://bookdown.org/max/FES/selection-overfitting.html>
10. [LCDB 1.1: A Database Illustrating Learning Curves Are ...](https://arxiv.org/pdf/2505.15657)
11. [\[1701.07852\] An Empirical Analysis of Feature Engineering for Predictive Modeling](https://ar5iv.labs.arxiv.org/html/1701.07852)
12. [The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data | NVIDIA Technical Blog](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)
13. [Data Science](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas)
14. [Leveraging Machine Learning to Detect Fraud: Tips to Developing a Winning Kaggle Solution](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)
15. <https://arxiv.org/pdf/2506.16791>
16. [\- LightGBM enables the missing value handle by default. Disable it by setting](https://github.com/microsoft/LightGBM/blob/af401561468a92cb407bea8157ff8e6dc57c4325/docs/Advanced-Topics.rst)
17. [Advanced Topics — LightGBM 4.7.0.99 documentation](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html)
18. [Advanced Topics — LightGBM 4.7.0 documentation](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html)
19. [Parameters — LightGBM 4.7.0.99 documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
20. [TargetEncoder — scikit-learn 1.9.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
21. [sklearn.preprocessing.TargetEncoder — scikit-learn 1.3.2 documentation](https://scikit-learn.org/1.3/modules/generated/sklearn.preprocessing.TargetEncoder.html)
22. [CatBoost for big data: an interdisciplinary review - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7610170/)
23. [CatBoost: unbiased boosting with categorical features](https://www.alphaxiv.org/abs/1706.09516)
24. [Categorical Encoding: One-Hot vs Label vs Target](https://letsdatascience.com/target-encoding/)
25. [Importance of Feature Engineering methods - Towards Data Science](https://towardsdatascience.com/importance-of-feature-engineering-methods-73e4c41ae5a3?gi=5accd2a44741)
26. [Monotonic Constraints — xgboost 3.5.0-dev documentation](https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html)
27. [xgboost/doc/tutorials/monotonic.rst at master · dmlc/xgboost · GitHub](https://github.com/dmlc/xgboost/blob/master/doc/tutorials/monotonic.rst)
28. [Feature Interaction Constraints — xgboost 3.4.2 documentation](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html)
29. [Monotone Tree-Based GAMI Models by Adapting XGBoost](https://arxiv.org/pdf/2309.02426)
30. [5.2. Permutation feature importance — scikit-learn 1.9.1 documentation](https://scikit-learn.org/stable/modules/permutation_importance.html)
31. [Permutation Importance vs Random Forest Feature Importance (MDI) — scikit-learn 1.9.1 documentation](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)
32. [4.2. Permutation feature importance — scikit-learn 0.24.2 documentation](https://scikit-learn.org/0.24/modules/permutation_importance.html)
33. [Discriminating among cosmological models by data-driven methods](https://arxiv.org/pdf/2408.01563)
34. [Feature Selection with the Boruta Package | Journal of Statistical Software](https://www.jstatsoft.org/article/view/v036i11)
35. [\[2004.03045\] Adversarial Validation Approach to Concept Drift Problem in User Targeting Automation Systems at Uber](https://ar5iv.labs.arxiv.org/html/2004.03045)
36. [(PDF) Managing dataset shift by adversarial validation for credit scoring](https://www.researchgate.net/publication/357203459_Managing_dataset_shift_by_adversarial_validation_for_credit_scoring)
37. [Explaining Tabular Foundation Model Differences Through Meta-Features](https://arxiv.org/pdf/2605.28418)
38. [Advanced Topics — LightGBM 3.2.1 documentation](https://lightgbm.readthedocs.io/en/v3.2.1/Advanced-Topics.html)
39. [XGBoost: A Scalable Tree Boosting System](https://www.alphaxiv.org/abs/1603.02754)
40. [XGBoost: A Scalable Tree Boosting System Tianqi Chen University of Washington](https://arxiv.org/pdf/1603.02754)
41. [Andrew Ng: Farewell, Big Data](https://spectrum.ieee.org/andrew-ng-data-centric-ai)
42. [\[1712.00409\] Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409)
43. [\[PDF\] Data Shapley: Equitable Valuation of Data for Machine Learning](https://www.semanticscholar.org/paper/Data-Shapley:-Equitable-Valuation-of-Data-for-Ghorbani-Zou/b7a717233ec3ff37385ab1b06816d0ca375f5bb3)
44. [Deep Learning Scaling is Predictable, Empirically](https://www.researchgate.net/publication/321487863_Deep_Learning_Scaling_is_Predictable_Empirically)
45. [Advanced Topics — LightGBM 4.7.0.99 documentation](https://lightgbm-readthedocs-io.translate.goog/en/latest/Advanced-Topics.html?_x_tr_sl=en&_x_tr_tl=id&_x_tr_hl=id&_x_tr_pto=tc)
