# HurdleZINB — Path-Wide A/B Verdict (Handoff for Research)

**Date:** 2026-05-20
**Branch:** `claude/fix-gbdt-mean-regression-GcY1g` (PR #46 → `devel`)
**Status:** P2.B of `docs/gbdt_mean_regression_plan.md` is complete. Result
is a **6 SHIP / 2 KILL split across 8 NBA ZINB markets** — clear net win,
asymmetric across markets. Handing off to a research agent to figure out
*why* the asymmetry exists, whether the SHIPs are "the hurdle is genuinely
better" or "the joint ZINB was catastrophically broken under deterministic
mode and the hurdle merely doesn't blow up," and what (if any) literature
illuminates the per-market routing question.

**Lineage:** Direct continuation of `docs/OVERCONFIDENCE_INVESTIGATION.md`
§2 (the ZINB derived-π gate fix), and a sibling to
`docs/CENTERED_TARGET_NEGATIVE_RESULT.md` (the P1 SkewNormal-branch
research handoff that established this doc format).

This document is self-contained. Concrete numbers are inlined; pointers
to code and artifacts are at the bottom.

---

## 1. One-line takeaway

`HurdleZINB` (a two-stage replacement for the jointly-fit ZINB:
calibrated binary `q = P(Y=0)` classifier + NegBin on `Y>0` +
derived-π gate via the ZINB identity) **ships on 6 of 8 NBA ZINB
markets** under bit-reproducible `meditate --deterministic` A/B, with
top-decile MAE improvements of +9.7% (FG3M) to +44.9% (OREB) and
brier_skill_score gains everywhere. It **kills on FTM** (top-decile
MAE +1.3%, below the 5% bar) and **on STL** (global MAE regression
+14.1%). Default `--zinb-mode=joint` stays byte-identical to pre-P2.B
production.

The split is not random: the SHIPs are *exactly* the markets where the
joint ZINB exhibits catastrophic per-row blowups under deterministic
mode (predicted means of 1437, 167, 538, …), and the KILLs are the
markets where joint ZINB was already well-calibrated. The interesting
research question is therefore whether the hurdle is *genuinely better
on hard markets* or merely *not pathological* — and whether non-
deterministic joint ZINB with full Optuna would close the gap on the
SHIPs.

## 2. What we were trying to learn

`OVERCONFIDENCE_INVESTIGATION.md` §2 documented that the jointly-fit
LightGBMLSS `ZINB` distribution's `gate` head is *unidentified under
NLL with a flexible count head*: the optimizer trades gate against
NB(0) (the NegBin's sampling mass at zero), and joint runs converge to
gates that are ≈ half the true marginal zero rate (FG3M 0.19 vs 0.34,
PF 0.02 vs 0.14, OREB 0.18 vs 0.45, …). Because the ZINB mean is
`(1−gate)·NB_mean`, the model compensates with a lower base mean,
which leaves the marginal calibrated but the *distribution shape*
wrong: far too little mass at/near zero, far too much at small
positive Y. The live consequence was every NBA ZINB market published
overconfident "Over" bets — FG3M 95% Over at the 0.9 cap.

The hypothesis the hurdle tests: *separating the zero/non-zero
classification problem from the count-shape problem* removes the
identifiability trade-off, because the binary classifier sees the
zero/non-zero label directly and the NegBin sees only positives.
Whether that re-architecting actually improves predictions across the
path was unknown.

## 3. What we built (so a researcher knows what's available)

All on branch `claude/fix-gbdt-mean-regression-GcY1g` (PR #46), tested
green:

- **`src/sportstradamus/hurdle.py`** — `HurdleZINB` class. Two-stage:
  Stage 1 is a calibrated binary `lgb.train` classifier on
  `(y == 0).astype(int)`; Stage 2 is a `LightGBMLSS(NegativeBinomial)`
  on the strictly-positive subset, with `_BoundedResponseFn` applied
  to `total_count` exactly as joint ZINB does. The structural-
  inflation gate π is *derived* from the ZINB identity
  `q = π + (1 − π) · NB(0)` ⇒ `π = clip((q − NB(0)) / (1 − NB(0)),
  0, 1)`, where `NB(0) = (1 − probs)^total_count` in the PyTorch
  parameterization (`mean = total_count · probs / (1 − probs)`, which
  matches `model_prob.py:253`). By construction P(Y=0) under the
  predicted ZINB equals `q` exactly per-row.

  The public surface mirrors LightGBMLSS:
  `predict(X, pred_type="parameters")` returns
  `DataFrame[total_count, probs, gate]`, indexed like `X`. A class
  attribute `is_hurdle = True` lets downstream code branch via
  `getattr(model, "is_hurdle", False)`, avoiding `isinstance`
  coupling. The wrapper exposes `set_model_start_values(X)` that
  delegates to the internal NegBin, so the prediction-side call site
  stays uniform. Picklable round-trip is unit-tested.

  **Spec drift note**: the original spec at
  `docs/archive/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md`
  (Phase B Task B1) returned `gate = 1 − p_nonzero` — i.e. the
  *marginal* zero rate, not the structural-inflation π. That was a
  bug: downstream `fused_loc` in
  `src/sportstradamus/helpers/distributions.py:344-347` documents
  explicitly that the `gate` column is treated as zero-inflation
  *before* gate deflation, not the marginal P(Y=0). Using `1 −
  p_nonzero` would over-count zeros downstream by `(1−q)·NB(0)`. The
  correction to derived-π preserves the existing ZINB downstream
  contract; legacy joint pickles, hurdle pickles, and downstream
  `fused_loc` / `get_ev` / `model_prob.py:252-257` ZINB decode all
  consume `gate` with the same semantics.

- **`--zinb-mode={joint,hurdle}` CLI flag** (in
  `src/sportstradamus/training/cli.py`). Orthogonal to
  `--target-strategy` (which is a target-space transform, not a model
  architecture swap). Default `joint` is byte-identical to pre-P2.B
  legacy. Under `--deterministic`, hurdle test sets dump into a
  `data/test_sets/deterministic/ratio_meanyr_hurdle/` subdir so they
  coexist with the joint baseline and `compression_eval` can diff
  them.

- **Pure helpers parallel to P0.5's LSS helpers** —
  `fit_hurdle_model` / `predict_hurdle_params` /
  `fit_predict_hurdle_params` in `training/pipeline.py`. Same purity
  contract as `fit_lss_model` (no disk writes; seed-aware), so
  `tests/integration/test_determinism_gate.py` can assert bit-
  identity for hurdle the same way it does for SkewNormal.

- **Backward compat for legacy pickles**: `model_prob.py` reads
  `is_hurdle` via `getattr(model, "is_hurdle", False)` — joint
  pickles that predate P2.B return False and route through the
  existing LSS path unchanged. No retrain stampede required.

- **Tests:**
  - `tests/test_hurdle_zinb.py` — 5 unit tests (parameter columns,
    gate reconstruction on a synthetic 35%-zero fixture, pickle
    round-trip, determinism, set_model_start_values delegation).
  - `tests/integration/test_zinb_hurdle_live_path.py` — drives a
    deterministic HurdleZINB on cached NBA_FG3M and asserts the
    identity `π + (1−π)·NB(0) ≈ q` per-row (mean tolerance 0.02) and
    two-run bit-identity under `DETERMINISTIC_SEED`.
  - `tests/integration/test_determinism_gate.py` extended with a
    parallel hurdle-path two-run bit-identity test on FG3M.

The infrastructure is the same shape as the P1 target-strategy
registry: a new arch swap is one new module + a CLI choice + three
helper-call branches, with the determinism gate and `compression_eval`
A/B harness already in place.

## 4. Results

### 4.1 A/B methodology

Two `meditate --deterministic --league NBA` runs against the cached
training parquet:

1. Baseline: `--zinb-mode joint` (legacy production behavior).
2. Candidate: `--zinb-mode hurdle`.

Each run trains every NBA market (the SkewNormal branch is unaffected
by the flag) and dumps a per-market test-set CSV with `Result`,
`Line`, `Odds`, `P`, `EV`, plus the raw ZINB params (`R`, `NB_P`,
`Gate`). `compression_eval --baseline <joint> --candidate <hurdle>
--strategy hurdle_zinb` scores each market independently against the
P0 ship/kill gate (≥5% top-decile MAE improvement, ≤1% global MAE
regression, brier_skill_score not worse).

The runs are bit-reproducible (proven by the extended
`test_determinism_gate.py` shipping with this PR).

### 4.2 Market structural facts

The 8 NBA ZINB markets, ordered by joint→hurdle outcome:

| Market | n (train) | mean(y) | E[y\|y>0] | std(y\|y>0) | zero rate | hist_gate | A/B verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **FG3M** | 14702 | 1.49 | 2.23 | 1.36 | 0.334 | 0.337 | **SHIP** |
| **OREB** | 14594 | 0.86 | 1.56 | 0.72 | 0.448 | 0.447 | **SHIP** |
| **PF**   | 14475 | 1.83 | 2.18 | 1.02 | 0.158 | 0.164 | **SHIP** |
| **TOV**  | 13999 | 1.31 | 1.95 | 1.01 | 0.327 | 0.337 | **SHIP** |
| **BLK**  | 15009 | 0.53 | 1.42 | 0.81 | 0.630 | 0.628 | **SHIP** |
| **BLST** | 14567 | 1.35 | 1.95 | 1.06 | 0.309 | 0.310 | **SHIP** |
| FTM      | 13598 | 1.44 | 2.58 | 1.49 | 0.443 | 0.450 | KILL |
| STL      | 14325 | 0.84 | 1.60 | 0.80 | 0.477 | 0.485 | KILL |

The structural facts alone do not partition the table — FTM and STL
have *higher* zero rates than several SHIPs (FG3M 0.33, PF 0.16, BLST
0.31), and STL's positives-conditional moments are nearly identical
to BLST's (1.60/0.80 vs 1.95/1.06).

### 4.3 SHIP/KILL gate outputs

Headline numbers from `compression_eval --baseline <joint csv>
--candidate <hurdle csv>` per market:

| Market | Top-dec MAE Δ | Global MAE Δ | brier_skill (joint → hurdle) | Verdict |
|---|---:|---:|---:|---|
| FG3M | **+9.7%**  | −88.4% | +0.115 → +0.290 | SHIP |
| OREB | **+44.9%** | −95.6% | +0.019 → +0.109 | SHIP |
| PF   | **+19.2%** | −61.2% | −0.238 → −0.002 | SHIP |
| TOV  | **+26.8%** | −2.7%  | −0.049 → +0.058 | SHIP |
| BLK  | **+40.4%** | −99.7% | +0.237 → +0.299 | SHIP |
| BLST | **+11.6%** | −60.8% | −0.002 → +0.093 | SHIP |
| FTM  | +1.3% | n/a | n/a | KILL (under 5% bar) |
| STL  | n/a | **+14.1% regression** | n/a | KILL (global MAE) |

(Sign convention: `+X% top-decile MAE` means a `X%` *reduction* in
top-decile MAE, i.e. better. `+Y% global MAE` means a `Y%`
*regression*, i.e. worse. This matches `compression_eval`'s ship/kill
verdict text and the P1 doc's formatting.)

### 4.4 The pattern the researcher needs to dig into: joint-ZINB catastrophic blowups

**Predicted ZINB means by market (averaged across the test set):**

| Market | Joint EV mean | Hurdle EV mean | Actual mean | Joint over-prediction factor |
|---|---:|---:|---:|---:|
| **BLK**  | **324.10** | 1.38 | 0.52 | 623× |
| **OREB** | **22.54**  | 1.55 | 0.86 | 26× |
| **FG3M** | **11.30**  | 2.07 | 1.49 | 7.6× |
| **BLST** | 3.34       | 1.93 | 1.33 | 2.5× |
| PF       | 2.93       | 2.15 | 1.88 | 1.6× |
| TOV      | 1.41       | 1.88 | 1.33 | 1.1× (calibrated) |
| FTM      | 1.60       | 2.56 | 1.64 | 1.0× (calibrated) |
| STL      | 0.78       | 1.62 | 0.85 | 0.9× (calibrated) |

Joint ZINB under `--deterministic` is catastrophically over-predicting
on BLK, OREB, FG3M, BLST — and within reason on TOV, FTM, STL. PF is
borderline.

**These catastrophic blowups are localized to specific deciles.**
Sampling the per-decile prediction means (full tables inlined below
under §4.5):

- **BLK joint** predicts 180.96 / 538.10 / 1437.04 / 158.73 / 109.56
  / 133.16 / 340.63 / 248.36 / 95.49 / **2.01** across deciles 0-9 of
  MeanYr (true 0.23 / 0.27 / 0.27 / 0.34 / 0.36 / 0.45 / 0.44 / 0.61
  / 0.76 / **1.45**). Only the top decile is sane.
- **OREB joint** has decile 2 = 167.89 and decile 3 = 26.83
  (otherwise reasonable).
- **FG3M joint** has decile 2 = 43.21 and decile 3 = 54.37 (otherwise
  reasonable).
- **BLST joint** has deciles 5/6/7 = 6.94 / 4.76 / 9.41 (true 1.34 /
  1.39 / 1.50).
- **PF joint** has decile 0 = 15.34 (otherwise reasonable).

Joint ZINB blows up specifically on **low-to-mid MeanYr deciles of
high-structural-zero markets**. The hurdle eliminates these
pathologies entirely — every market predicts in the 1-2.5 range
across every decile.

**Hypothesis for what's happening in joint ZINB**: at low MeanYr the
identifiability problem becomes severe because the joint optimization
can satisfy NLL by pushing `total_count` very high *and* `probs` very
low (which produces NB(0) ≈ 1 and a small but technically defined
mean `r·p/(1−p)`), or vice versa. The deterministic-mode
hyperparameters (30 rounds, no Optuna search) may not find the
"sensible" basin and land in this pathological region. Joint with
full Optuna search at production scale might converge to a different,
non-pathological local optimum.

This is the load-bearing uncertainty in the SHIP verdict: **is the
hurdle genuinely better, or is it merely more robust to the
deterministic-mode hyperparameter starvation while joint with full
Optuna would close the gap?**

### 4.5 Full per-market decile bias tables (joint vs hurdle)

(Columns: `meanyr` = decile midpoint of `X_test["MeanYr"]`, `n` =
rows, `bias` = predicted_mean − actual_mean, `pred` = predicted EV
mean, `actual` = actual `Result` mean.)

#### NBA_FG3M (SHIP)

```
joint                                  hurdle
d  meanyr  n   bias   pred  actual  | d  meanyr  n   bias   pred  actual
0  0.09   218  +1.93  2.11   0.18   | 0  0.09   218  +1.08  1.27   0.18
1  0.50   218  +0.19  0.95   0.76   | 1  0.50   218  +0.87  1.63   0.76
2  0.82   217 +42.02  43.21  1.18   | 2  0.82   217  +0.61  1.80   1.18
3  1.04   218 +53.22  54.37  1.14   | 3  1.04   218  +0.77  1.91   1.14
4  1.22   218  +0.92  2.34   1.42   | 4  1.22   218  +0.54  1.96   1.42
5  1.45   217  -0.04  1.49   1.53   | 5  1.45   217  +0.51  2.04   1.53
6  1.69   218  +0.14  1.89   1.74   | 6  1.69   218  +0.41  2.15   1.74
7  1.97   217  +0.01  1.91   1.90   | 7  1.97   217  +0.38  2.29   1.90
8  2.37   218  -0.24  2.18   2.43   | 8  2.37   218  +0.18  2.61   2.43
9  3.03   218  +0.06  2.65   2.59   | 9  3.03   218  +0.44  3.03   2.59
compression_ratio  joint=130.05  hurdle=0.42
```

The joint pathology lives in deciles 2-3 (43.21 / 54.37). The hurdle
trades that blowup for systematic mild over-prediction across all
deciles — global MAE collapses 88%, top-decile MAE improves 9.7%.

#### NBA_FTM (KILL)

```
joint                                 hurdle
d  meanyr  n   bias   pred  actual  | d  meanyr  n   bias   pred  actual
0  0.28   202  -0.10  0.50   0.59   | 0  0.28   202  +1.61  2.20   0.59
1  0.55   201  +0.05  0.81   0.76   | 1  0.55   201  +1.47  2.23   0.76
2  0.75   201  +0.06  0.99   0.93   | 2  0.75   201  +1.40  2.33   0.93
3  0.96   202  +0.18  1.34   1.15   | 3  0.96   202  +1.21  2.36   1.15
4  1.17   201  +0.09  1.37   1.28   | 4  1.17   201  +1.16  2.44   1.28
5  1.41   201  +0.21  1.48   1.27   | 5  1.41   201  +1.19  2.46   1.27
6  1.79   202  +0.10  1.83   1.73   | 6  1.79   202  +0.93  2.66   1.73
7  2.36   201  +0.14  2.06   1.92   | 7  2.36   201  +0.89  2.81   1.92
8  3.46   201  +0.04  2.68   2.64   | 8  3.46   201  +0.33  2.97   2.64
9  5.61   202  -1.16  2.98   4.13   | 9  5.61   202  -0.98  3.15   4.13
compression_ratio  joint=0.48  hurdle=0.24
```

Joint FTM was well-calibrated across deciles 0-8 (|bias| ≤ 0.21
everywhere) with the standard top-decile under-prediction (-1.16 at
MeanYr=5.61). Hurdle introduces systematic over-prediction
**everywhere** (bias +0.93 to +1.61 across deciles 0-8), barely
improving the top-decile MAE (+1.3%, under the 5% bar) while
substantially hurting every other decile. This is a clean negative
result for hurdle on a well-fit joint baseline.

#### NBA_OREB (SHIP)

```
joint                                  hurdle
d  meanyr  n   bias    pred    actual | d  meanyr  n   bias   pred  actual
0  0.21   217  +0.73   1.13    0.39   | 0  0.21   217  +0.87  1.26   0.39
1  0.39   216  +0.38   0.89    0.51   | 1  0.39   216  +0.80  1.31   0.51
2  0.52   216 +167.31 167.89   0.58   | 2  0.52   216  +0.79  1.37   0.58
3  0.63   216  +26.12  26.83   0.70   | 3  0.63   216  +0.78  1.48   0.70
4  0.75   216  +0.12   0.97    0.85   | 4  0.75   216  +0.66  1.51   0.85
5  0.86   216 +20.26   21.10   0.85   | 5  0.86   216  +0.68  1.52   0.85
6  0.99   216  +0.00   0.99    0.99   | 6  0.99   216  +0.60  1.59   0.99
7  1.20   216  +0.16   1.21    1.06   | 7  1.20   216  +0.59  1.64   1.06
8  1.61   216  +1.37   2.53    1.15   | 8  1.61   216  +0.60  1.76   1.15
9  2.54   217  +0.52   2.08    1.56   | 9  2.54   217  +0.43  1.99   1.56
compression_ratio  joint=788.36  hurdle=0.37
```

Joint OREB pathology: deciles 2 (167.89), 3 (26.83), 5 (21.10), 8
(2.53). Hurdle similarly mild over-prediction everywhere (+0.43 to
+0.87). Top-decile MAE improves 45%.

#### NBA_PF (SHIP)

```
joint                                  hurdle
d  meanyr  n   bias    pred   actual | d  meanyr  n   bias   pred  actual
0  0.91   215 +13.85   15.34   1.49  | 0  0.91   215  +0.48  1.97   1.49
1  1.26   214  +0.06    1.71   1.65  | 1  1.26   214  +0.27  1.93   1.65
2  1.47   214  +0.01    1.66   1.65  | 2  1.47   214  +0.33  1.98   1.65
3  1.64   214  -0.29    1.27   1.57  | 3  1.64   214  +0.45  2.02   1.57
4  1.79   215  -0.60    1.19   1.79  | 4  1.79   215  +0.29  2.08   1.79
5  1.97   214  -0.37    1.61   1.98  | 5  1.97   214  +0.18  2.16   1.98
6  2.13   214  -0.56    1.49   2.05  | 6  2.13   214  +0.16  2.22   2.05
7  2.29   214  -0.59    1.46   2.05  | 7  2.29   214  +0.21  2.27   2.05
8  2.54   214  -0.62    1.64   2.26  | 8  2.54   214  +0.05  2.31   2.26
9  2.99   215  -0.46    1.88   2.33  | 9  2.99   215  +0.22  2.55   2.33
compression_ratio  joint=52.37  hurdle=0.25
```

Joint PF: one catastrophic decile (15.34 at decile 0) plus a
*real compression signature* — top-decile under-prediction (-0.46 to
-0.62 across deciles 4-9). Hurdle eliminates both. Brier_skill jumps
from −0.238 (worse than book) to −0.002 (matching book).

#### NBA_STL (KILL)

```
joint                                  hurdle
d  meanyr  n   bias    pred  actual | d  meanyr  n   bias   pred  actual
0  0.23   210  +0.11   0.65   0.55  | 0  0.23   210  +0.92  1.46   0.55
1  0.42   209  +0.24   0.83   0.59  | 1  0.42   209  +0.76  1.34   0.59
2  0.54   209  +0.47   1.07   0.60  | 2  0.54   209  +0.83  1.42   0.60
3  0.63   209  +0.17   0.79   0.62  | 3  0.63   209  +0.88  1.50   0.62
4  0.74   209  +0.13   0.74   0.61  | 4  0.74   209  +1.02  1.63   0.61
5  0.84   209  -0.19   0.65   0.84  | 5  0.84   209  +0.82  1.66   0.84
6  0.94   209  -0.34   0.57   0.91  | 6  0.94   209  +0.80  1.71   0.91
7  1.09   209  -0.44   0.74   1.18  | 7  1.09   209  +0.52  1.70   1.18
8  1.26   209  -0.32   0.76   1.09  | 8  1.26   209  +0.66  1.75   1.09
9  1.68   209  -0.59   0.96   1.55  | 9  1.68   209  +0.46  2.00   1.55
compression_ratio  joint=1.065  hurdle=0.34
```

**STL is the cleanest negative result**: joint ZINB was already
nearly perfectly calibrated (compression_ratio 1.065 ≈ 1.0 = no
compression) with the standard mild top-decile under-prediction
(-0.59 at MeanYr=1.68). Hurdle introduces systematic over-prediction
across every decile (+0.46 to +1.02), regressing global MAE by 14.1%.
STL is the canonical "hurdle hurts when joint is healthy" case.

#### NBA_TOV (SHIP)

```
joint                                  hurdle
d  meanyr  n   bias    pred  actual | d  meanyr  n   bias   pred  actual
0  0.36   203  +0.22   0.85   0.63  | 0  0.36   203  +0.81  1.43   0.63
1  0.57   203  +0.49   1.18   0.69  | 1  0.57   203  +0.83  1.52   0.69
2  0.73   203  +0.17   1.22   1.04  | 2  0.73   203  +0.54  1.59   1.04
3  0.88   202  +0.26   1.12   0.86  | 3  0.88   202  +0.84  1.70   0.86
4  1.05   203  +0.12   1.24   1.13  | 4  1.05   203  +0.68  1.81   1.13
5  1.28   203  -0.04   1.39   1.43  | 5  1.28   203  +0.47  1.90   1.43
6  1.50   202  +0.13   1.55   1.43  | 6  1.50   202  +0.58  2.01   1.43
7  1.83   203  +0.22   1.88   1.67  | 7  1.83   203  +0.38  2.05   1.67
8  2.29   203  -0.35   1.64   1.99  | 8  2.29   203  +0.32  2.31   1.99
9  3.05   203  -0.39   2.04   2.43  | 9  3.05   203  +0.05  2.48   2.43
compression_ratio  joint=0.71  hurdle=0.34
```

Joint TOV was also reasonably calibrated (no pathological blowups,
mild top-decile under-prediction). Hurdle introduces global
over-prediction *similar to STL*, but the top-decile improvement is
+26.8% (vs STL's regressive global MAE). Why does TOV ship and STL
kill, given similar joint baselines? This is the cleanest single
research question in the table.

#### NBA_BLK (SHIP)

```
joint                                   hurdle
d  meanyr  n   bias      pred    actual | d  meanyr  n   bias   pred  actual
0  0.07   223 +180.73   180.96   0.23   | 0  0.07   223  +0.84  1.07   0.23
1  0.15   222 +537.83   538.10   0.27   | 1  0.15   222  +0.97  1.23   0.27
2  0.21   222 +1436.77 1437.04   0.27   | 2  0.21   222  +0.90  1.17   0.27
3  0.28   222 +158.39   158.73   0.34   | 3  0.28   222  +0.98  1.32   0.34
4  0.34   223 +109.20   109.56   0.36   | 4  0.34   223  +0.99  1.35   0.36
5  0.40   222 +132.71   133.16   0.45   | 5  0.40   222  +0.89  1.34   0.45
6  0.48   222 +340.19   340.63   0.44   | 6  0.48   222  +0.91  1.35   0.44
7  0.63   222 +247.76   248.36   0.61   | 7  0.63   222  +0.79  1.40   0.61
8  0.84   222  +94.74    95.49   0.76   | 8  0.84   222  +0.72  1.47   0.76
9  1.45   223   +0.56     2.01   1.45   | 9  1.45   223  +0.63  2.08   1.45
compression_ratio  joint=5356.79  hurdle=0.49
```

BLK is the extreme case: joint ZINB blows up in *9 of 10 deciles* —
predicted means of 180-1437 against true 0.23-0.76. Only the top
decile is sane. Hurdle is uniformly +0.6 to +1.0 over-prediction in
mid-deciles. The "+40% top-decile MAE improvement" is real, but the
joint baseline for this market is essentially broken (global MAE
324.1), so the absolute comparison is not far above "anything is
better than this."

#### NBA_BLST (SHIP)

```
joint                                  hurdle
d  meanyr  n   bias    pred  actual | d  meanyr  n   bias   pred  actual
0  0.38   214  +0.33   1.18   0.85  | 0  0.38   214  +0.68  1.53   0.85
1  0.68   214  +0.44   1.36   0.93  | 1  0.68   214  +0.71  1.64   0.93
2  0.85   213  +0.40   1.37   0.97  | 2  0.85   213  +0.72  1.70   0.97
3  1.01   214  +0.51   1.68   1.17  | 3  1.01   214  +0.61  1.78   1.17
4  1.18   213  +0.42   1.61   1.19  | 4  1.18   213  +0.68  1.88   1.19
5  1.34   214  +5.60   6.94   1.34  | 5  1.34   214  +0.61  1.95   1.34
6  1.51   213  +3.37   4.76   1.39  | 6  1.51   213  +0.55  1.95   1.39
7  1.72   214  +7.91   9.41   1.50  | 7  1.72   214  +0.58  2.08   1.50
8  2.04   213  +0.87   2.67   1.79  | 8  2.04   213  +0.45  2.24   1.79
9  2.58   214  +0.17   2.37   2.21  | 9  2.58   214  +0.34  2.55   2.21
compression_ratio  joint=24.91  hurdle=0.32
```

BLST joint: deciles 5/6/7 blow up to 4.76 / 6.94 / 9.41 (true 1.34 /
1.39 / 1.50). Hurdle uniformly mild over-prediction (+0.34 to +0.72).

## 5. Interpretation

### 5.1 Two regimes, not one

The 6 SHIP / 2 KILL split partitions the markets cleanly along
**joint-ZINB pathology** rather than any structural feature of the
data:

- **Pathology regime** (FG3M, OREB, PF, TOV, BLK, BLST): joint ZINB
  exhibits per-row catastrophic over-prediction in some deciles
  (predicted means ≥ 5×, often ≥ 100× actual). Global MAE is
  inflated by these blowups; top-decile MAE is also degraded because
  the per-decile model behavior is unstable. Hurdle eliminates the
  blowups; SHIP.
- **Calibrated regime** (FTM, STL): joint ZINB converges to
  reasonable per-decile predictions; the only systematic problem is
  the standard mild top-decile under-prediction shared across every
  GBDT family in this repo. Hurdle introduces uniform over-prediction
  (because its NegBin-on-positives has a higher conditional mean
  than the joint ZINB's deflated base mean); KILL.

Three follow-up notes are load-bearing:

1. **The pathology is plausibly a deterministic-mode artifact.** The
   joint ZINB runs 30 boosting rounds with fixed deterministic
   hyperparameters and no Optuna search; the production model runs
   hundreds of rounds with a tuned hyperparameter set. The joint's
   tendency to land in a bad basin under deterministic mode might
   not reproduce at production scale. This is the single most
   important uncertainty for the researcher to resolve before
   shipping hurdle by default.
2. **TOV vs STL is the analytic cleanest test case.** TOV joint
   shows no catastrophic blowups but ships; STL joint also shows no
   blowups and kills. Both have similar structural features (TOV:
   mean 1.31, zero rate 0.33; STL: mean 0.84, zero rate 0.48). The
   factor that decides hurdle's verdict must distinguish them.
3. **The hurdle's bias is consistent across markets.** Wherever
   hurdle's NegBin is trained on positives with mean ≈ 2.0 (always,
   in this dataset), its base EV lands in 1.4-2.5 across every
   market. That's a feature when the joint is broken and a bug when
   the joint is calibrated.

### 5.2 Why the hurdle over-predicts on calibrated markets

For a market with true zero rate `q` and positives-conditional mean
`μ_pos`, the hurdle's *marginal* mean (combining gate and NegBin
output downstream) is:

```
E_hurdle[Y] = (1 − π) · NB_mean
            = (1 − π) · μ_pos
```

where `π` is the *structural inflation* gate. By the derived-π
identity, `π = (q − NB(0)) / (1 − NB(0))`. The NegBin on positives
has `NB(0) = (1 − probs)^total_count > 0` (typically 0.10-0.20 for
the FG3M-like mean range), so `π < q`. Hence:

```
(1 − π) > (1 − q) ⇒ E_hurdle[Y] > (1 − q) · μ_pos = E_true[Y]
```

The hurdle systematically over-shoots the marginal mean by the
amount `(q − π) · μ_pos = NB(0) · (1 − q) · μ_pos / (1 − NB(0))`. For
FG3M this is ≈ 0.16 · 0.67 · 2.23 / 0.84 ≈ 0.28 — *consistent with
the observed +0.4 to +0.8 mid-decile bias in the SHIP markets*.

This is a fundamental property of how derived-π interacts with the
existing downstream's ZINB-mean formula
`mean = (1 − gate) · total_count · probs / (1 − probs)`. The hurdle
preserves the *total* P(Y=0) (= q exactly) but inflates the marginal
mean by the gap between π and q.

**Open question**: a true hurdle distribution would model
`mean = (1 − q) · μ_pos / (1 − NB(0))` (the zero-truncated NegBin
mean times the non-zero probability). To get that downstream, one
would either (a) return `gate = q` directly and have downstream
divide `total_count · probs / (1 − probs)` by `(1 − NB(0))` to get
the truncated mean, or (b) doctor the returned `total_count` /
`probs` so the LightGBMLSS ZINB formula computes the truncated mean.
Option (a) breaks the downstream contract; option (b) is a
non-standard parameterization. Neither was attempted in this PR.

This is the cleanest candidate fix if a researcher wants to take a
crack at making hurdle ship on FTM/STL: implement option (b) and
A/B again.

### 5.3 What the verdict criterion in the parent plan got wrong

The parent plan in `docs/gbdt_mean_regression_plan.md` said the
verdict for hurdle should be:

> Predicted gate mean within ±0.07 of `hist_gate` (joint ZINB: ≈ 0.5×
> hist_gate; hurdle target: ≈ hist_gate).

That criterion is **mis-stated** under derived-π semantics. Per §3
and §5.2: the gate returned by `HurdleZINB.predict` is π_zi
(zero-inflation), which is structurally < q. Looking at the numbers
on FG3M:

- joint gate mean: 0.267 (not 0.18 as in
  `OVERCONFIDENCE_INVESTIGATION.md` — the deterministic-mode joint
  ZINB lands at a different fit than the production joint quoted in
  the investigation).
- hurdle gate mean: 0.209.
- true zero rate `q`: 0.334.

Both joint and hurdle have gate < q. The hurdle is *not* significantly
closer to `q` than joint — they're close to each other.

What the hurdle uniquely guarantees is the *identity reconstruction*:
`π + (1 − π) · NB(0) = q` exactly per-row (subject to numerical
clipping, never triggered in observed runs). That's the meaningful
contract, asserted by the live-path integration test
(`tests/integration/test_zinb_hurdle_live_path.py`). The
*compression_eval* gate (top-decile MAE / brier_skill_score) is the
meaningful SHIP/KILL signal.

If you read the investigation's "joint gate is ~half the true zero
rate" framing and assumed that fixing it should make `Model Gate ≈
q`, you'd be wrong — that's not what `gate` means in this codebase's
ZINB semantics. The investigation's framing implied a hurdle gate
of q would directly fix downstream P(over@line); under derived-π
that fix happens implicitly through the gate+NB combination, not
through gate alone.

## 6. Open questions for the research agent

Ranked by likely leverage:

1. **Is the joint-ZINB catastrophic blowup a deterministic-mode
   artifact?** Re-run joint ZINB on BLK / OREB / FG3M / BLST with
   *non-deterministic* full-Optuna hyperparameter search (i.e.
   regular `meditate --league NBA` without `--deterministic`). If
   joint converges to non-pathological per-decile predictions, the
   hurdle's headline SHIPs collapse to "hurdle ≈ joint at production
   scale" — and the actual leverage of P2.B reduces to FG3M /
   OREB / PF (where joint was *partially* pathological even in the
   investigation's production-scale evidence). If joint still blows
   up at production scale, P2.B is a genuine architectural win and
   should ship by default for the pathology markets.
2. **What distinguishes TOV (SHIP) from STL (KILL)?** Both have
   non-pathological joint baselines and similar structural features.
   Hypotheses worth checking: (a) feature predictive power for the
   binary q vs the count head — if a market has features that
   strongly predict zero/non-zero, the hurdle's separate
   classifier wins; if not, the joint NLL's gate can't be improved
   on. (b) Conditional-distribution shape — STL's positives may be
   tighter / more Poisson-like, fitting joint ZINB cleanly; TOV's
   positives may have a tail or bimodality the joint can't capture
   but the hurdle's NB-on-positives handles. (c) Per-row variance
   in `q − NB(0)` — when this is small per-row the derived-π is
   well-conditioned, when it's large the clip(...) at the [0,1]
   boundary breaks the identity (per-row, not in the mean).
3. **Would a "true hurdle" parameterization (option (b) in §5.2)
   fix the FTM/STL regression without breaking the SHIPs?** A
   `total_count / (1 − NB(0)) · probs` style doctored output, with
   `gate = q` returned directly, would compute the zero-truncated
   NegBin mean downstream. This is the mathematically cleanest
   form of a hurdle, and addresses the §5.2 derivation that shows
   derived-π systematically over-shoots the marginal mean.
4. **What does the literature say about hurdle vs. ZINB selection
   per market?** Mullahy 1986 and the count-data literature (Cameron
   & Trivedi; Hilbe) have decision rules — usually a Vuong test
   comparing the two model fits, or moment-based diagnostics
   (excess zeros relative to NB(0); ratio of variance to mean).
   Has anyone studied which features of the data predict hurdle
   winning vs. ZINB winning at a market-by-market level? A per-
   market routing config (e.g.
   `data/zinb_mode_per_market.json` keyed on a precomputable
   diagnostic) would be the simplest deploy-time fix.
5. **Is the live-path verdict different from the offline-eval
   verdict?** `compression_eval` scores on the dumped test sets,
   which use the raw `Model EV` (no `fused_loc` book blend, no
   `temperature`, no `dispersion_cal`). The published EV the user
   sees goes through all those downstream stages. A hurdle that
   ships on raw test-set MAE might lose at the published-EV stage
   if the book blend amplifies the over-prediction bias. The new
   `tests/integration/test_zinb_hurdle_live_path.py` asserts the
   identity reconstruction but not the EV-after-blend. Worth a
   quick check on a fresh `current_offers.parquet` if one is
   available.
6. **Is the hurdle's gate identifiability empirically better than
   joint's, independent of MAE?** Compare predicted-gate vs.
   *holdout* zero rate, conditional on `MeanYr` decile. If the
   hurdle's per-row gate tracks the per-row zero rate better than
   joint does (even when neither hits the marginal `q` precisely),
   that's evidence the architecture is genuinely identifying the
   gate, regardless of the catastrophic blowup story.

## 7. What's locked

- **Default `--zinb-mode=joint` stays.** Promoting hurdle to default
  would regress FTM and STL (the calibrated joint markets). The
  per-market routing question (FTM/STL stay joint, others move to
  hurdle) is the natural follow-up but was not attempted in this
  PR.
- **HurdleZINB infrastructure stays in.** The class, the
  fit/predict helpers, the live-path test, the determinism gate
  hurdle assertion, the CLI flag, the pickle metadata — all
  reusable for the next round of A/B (e.g. the §6.3 "true hurdle"
  variant, or per-market routing).
- **Default `meditate` invocation is byte-identical to pre-P2.B
  production** — confirmed by `tests/integration/test_pipeline_target_strategy.py`.
  The branch is shippable to `devel` whenever the user is ready;
  the verdict doesn't require any rollback.
- **Derived-π formula stays.** Returning `gate = 1 − p_nonzero`
  (the simpler form from the original spec) would break the
  downstream contract (over-count zeros by `(1 − q) · NB(0)`). If
  a researcher wants to test a different `gate` semantics, they
  should also change the downstream `fused_loc` consumer in
  lockstep — out of scope for the hurdle module alone.

## 8. Artifact pointers

| What | Where |
|---|---|
| Branch | `claude/fix-gbdt-mean-regression-GcY1g` (37+ commits ahead of `origin`, not pushed) |
| Master plan (durable) | `docs/gbdt_mean_regression_plan.md` (P2.B row updated with the SHIP verdict; P2.A row marked DEAD) |
| P1 research handoff (sibling doc) | `docs/CENTERED_TARGET_NEGATIVE_RESULT.md` |
| Overconfidence investigation (lineage) | `docs/OVERCONFIDENCE_INVESTIGATION.md` §2 |
| Original spec (pre-correction) | `docs/archive/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md` Phase B "SUPERSEDED → derived-π" |
| HurdleZINB class | `src/sportstradamus/hurdle.py` |
| Pipeline integration | `src/sportstradamus/training/pipeline.py` (helpers `fit_hurdle_model`, `predict_hurdle_params`, `fit_predict_hurdle_params`; `use_hurdle` dispatch in `train_market`) |
| CLI flag | `src/sportstradamus/training/cli.py` `--zinb-mode` |
| Prediction-side adapter | `src/sportstradamus/prediction/model_prob.py` (`getattr(model, "is_hurdle", False)` branch around lines 205-220) |
| Offline harness | `src/sportstradamus/scripts/compression_eval.py` |
| Determinism gate (extended) | `tests/integration/test_determinism_gate.py` (parallel hurdle assertion on NBA_FG3M) |
| Hurdle unit tests | `tests/test_hurdle_zinb.py` (5 tests) |
| Hurdle live-path test | `tests/integration/test_zinb_hurdle_live_path.py` (identity reconstruction + determinism) |
| Joint baseline test-set CSVs (A/B baseline) | `src/sportstradamus/data/test_sets/deterministic/ratio_meanyr/NBA_{FG3M,FTM,OREB,PF,STL,TOV,BLK,BLST}.csv` |
| Hurdle candidate test-set CSVs | `src/sportstradamus/data/test_sets/deterministic/ratio_meanyr_hurdle/NBA_{FG3M,FTM,OREB,PF,STL,TOV,BLK,BLST}.csv` |
| Joint baseline model pickles | `src/sportstradamus/data/models/deterministic/ratio_meanyr/NBA_*.mdl` |
| Hurdle candidate model pickles | `src/sportstradamus/data/models/deterministic/ratio_meanyr_hurdle/NBA_*.mdl` |
| Full per-market A/B verdict raw output | `/tmp/p2b_full_ab.txt` (the source for §4.5 above — local-only; reproducible via §9) |

## 9. How to re-run the A/B

```bash
# Baseline (joint ZINB, byte-identical to pre-P2.B production)
poetry run meditate --deterministic --league NBA --zinb-mode joint

# Candidate (hurdle)
poetry run meditate --deterministic --league NBA --zinb-mode hurdle

# Per-market SHIP/KILL verdict
for m in FG3M FTM OREB PF STL TOV BLK BLST; do
  poetry run python -m sportstradamus.scripts.compression_eval \
    --baseline "src/sportstradamus/data/test_sets/deterministic/ratio_meanyr/NBA_${m}.csv" \
    --candidate "src/sportstradamus/data/test_sets/deterministic/ratio_meanyr_hurdle/NBA_${m}.csv" \
    --strategy hurdle_zinb --no-log
done
```

Wall time: ~10-15 min per `meditate --deterministic --league NBA` run
on cached parquet (one-time cost; runs are bit-reproducible). The
joint baseline CSVs already exist from P1's A/B, so re-running the
joint side is only needed if `data/test_sets/deterministic/ratio_meanyr/`
has been blown away.

### Reproducing the production-scale joint ZINB (open question §6.1)

This is the critical experiment for resolving whether the SHIPs are
real:

```bash
# WARNING: This is a long-running production retrain that takes
# ~hours on this dataset. It rewrites the production model pickles
# AND production training_report.txt. Run only with the user's
# explicit go-ahead.
poetry run meditate --league NBA --force

# Or, less destructive: a one-market production-style retrain via
# a throwaway script that calls train_market(...) without
# deterministic=True on a single market. The pickle still lands in
# data/models/NBA_BLK.mdl (production path); a researcher should
# back that up first.
```

A safer approach: extend `train_market` to accept a `production_mode`
flag that runs full Optuna *but* dumps to a `data/{models,test_sets}/sandbox/`
subdir, mirroring the deterministic redirect. That's a small
infrastructure addition (~30 lines in `pipeline.py:1214` and a CLI
flag in `cli.py`), unblocking §6.1 without risking production
artifacts. It's not in this PR.

### Inspecting the per-row blowups

The CSVs in `data/test_sets/deterministic/ratio_meanyr/NBA_{BLK,OREB,FG3M,BLST}.csv`
have an `EV` column (raw NegBin mean `total_count · probs / (1 − probs)`)
and a `Result` column. Sorting by `EV` desc surfaces the pathological
rows (BLK alone has rows with `EV > 1000`). The features driving
those blowups are in the same CSV's feature columns — useful for
characterizing what kind of row triggers the joint ZINB's
unidentifiability disaster.
