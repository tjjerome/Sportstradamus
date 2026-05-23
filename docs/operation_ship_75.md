# Operation Ship 75

> **Home of record for the model-research push to 75% breadth.** Supersedes
> the deprecated `docs/archive/gbdt_mean_regression_plan.md` /
> `docs/archive/gbdt_mean_regression_context.md` track. Decision history,
> research verdicts, and citations live in
> [`docs/operation_ship_references.md`](operation_ship_references.md). Stub
> for the next-rung push is at
> [`docs/operation_ship_90.md`](operation_ship_90.md).

## North Star

**Get ≥ 75% of markets in every covered league past the Tier-0 quality
gates (g1–g5) in [`src/sportstradamus/data/tier0_scorecard.csv`](../src/sportstradamus/data/tier0_scorecard.csv).**

- NBA ≥ 16/21
- WNBA ≥ 14/18
- NFL ≥ 15/20

Mantra: *"don't let perfect be the enemy of good."* A profitable model
goes to production once it proves it — Tier-0 gate offline → 14-day
Gate-2 soak live → graduate. A cell promoted but still in 14-day soak
counts toward the 75% numerator (ship-incrementally).

## Current state (scorecard snapshot 2026-05-23, post-Step-0.5 audit)

| League | Shipped | Markets | % | 75% target | Gap |
|---|---|---|---|---|---|
| NBA | 18 | 21 | 86% | 16 | **+2** |
| NFL | 11 | 20 | 55% | 15 | **−4** |
| WNBA | 12 | 18 | 67% | 14 | **−2** |

Source: [`data/training/tier0_scorecard.csv`](../src/sportstradamus/data/training/tier0_scorecard.csv),
strategy `ship75-step0-g4audit-2026-05-23`. `ship = g1_pass AND g2_pass AND
g3_pass AND g4_pass AND g5_pass`
([compression_eval.py:653](../src/sportstradamus/scripts/compression_eval.py)).

**Step 0 verdict: G4 was measuring a measurement artifact.** The old
gate `_iqr(point_pred) / _iqr(actual)` compared point-prediction spread
to realized-outcome spread — a sharpness-vs-calibration category error
([Gneiting & Raftery 2007], brief at `/tmp/researcher_g4_audit.md`).
Step 0.2 replaced the predicted-IQR estimator with the **pooled
analytical IQR** of the per-row predictive distribution via
`scipy.stats.<dist>.ppf(0.75) − ppf(0.25)`. Outcome: 10 cells flipped
to ship with no retrain (NBA +6, NFL +4, WNBA +0). WNBA cells now pass
G4 but multi-gate failures (mostly G5 ECE) hold them; that's Step 1
territory.

**Step 0 G4 audit recovery (2026-05-23):**

- NBA flips (6): AST, BLK, FG3A, FTM, OREB, STL — promote via Step 0.3.
- NFL flips (4): passing-tds, receiving-tds, rushing-tds, tds — the
  last three are the degenerate `IQR(actual) = 0` cells; the new
  `0/0 → 1.0` convention from Step 0.4 ships them.
- WNBA flips (0): G4 now passes for 5 of 8 WNBA failing cells, but each
  still fails G5 (ECE) — passes flipped but the cell still kills, so
  the count is unchanged until Step 1 attacks G5.

**Dominant failure mode after Step 0: G5 ECE.** ≥ 14 cells now fail on
the equal-mass ECE ceiling (0.075). Step 1 (post-hoc bias correction)
must extend to ECE-driven post-hoc calibration of `P` (probabilities),
not just mean bias.

## Step 0.5 — Lifecycle gate audit (G5 + S3 fixes; DONE 2026-05-23)

Triggered by the user-requested objective audit of the remaining four
gates after Step 0. Brief at `/tmp/researcher_lifecycle_gate_audit.md`.

**Verdicts:**

- G1 paired-Brier CI = **SOUND** (re-audited at user request — brief at
  `/tmp/researcher_g1_audit.md`). 88 % pass rate is genuine: paired
  correctly, effectively one-sided α=0.025, MDE 1 % on full-season
  cells / 2–3 % on NFL-N≈400. All 7 failures have CIs straddling zero.
  No block-bootstrap fix is currently constructible — `test_set` CSVs
  lack a `Date` column. Backlog: schema upgrade enables block bootstrap
  for both offline G1 and live Gate-2.
- G2 star-bias z, G3 bench-bias z = SOUND. Closed-form 2-sample z on
  inflation-conditional means; thresholds calibrated to ½ σ.
- G4 IQR ratio = SOUND post-Step-0 (analytical IQR, pooled).
- G5 equal-mass ECE = **DEFECT (false-fails)**. Raw equal-mass ECE is
  positively biased at finite N — the bias scales O(1/√N_bin) and
  falsely fails up to 44.6 % of perfectly-calibrated NFL-N≈240 cells.
  Per [Roelofs 2022], the fix is to subtract the matched null-bias
  offset estimated via Monte-Carlo bootstrap under
  `y ~ Bernoulli(p_model)`.
- S2 paired Brier CI = SOUND.
- S3 paired Sharpe (`sc > sb`) = **DEFECT (coin-flip)**. Bare ratio
  comparison has ~50 % Type-I rate. Per [Memmel 2003], replace with the
  closed-form paired-Sharpe z and ship at `z > 1.645` (one-sided
  α=0.05).

**Step 0.5 implementation:**

- `_ece_debias_offset(p_model)` — 200-resample Monte-Carlo null bias.
- `_gate5_ece_debiased(p_model, y)` — `raw_ECE - offset`.
- `apply_thresholds` now reads `g5_ece_debiased` (was raw `g5_ece`).
  Falls back to raw if the debiased column is absent.
- `_memmel_sharpe_z(b, c) -> (SR_b, SR_c, z)` — closed-form paired-Sharpe z.
- `_supersede_paired_sharpe` returns the z; `supersede_verdict.s3_pass`
  fires on `z > _SUPERSEDE_S3_Z_MIN = 1.645`.

**Step 0.5 recovery (9 cells flip to ship, 0 break):**

- NBA flips (5): BLST, DREB, FG3M, FGM, TOV (all g5 raw 0.077–0.091,
  debiased 0.04–0.064; gate threshold 0.075).
- NFL flips (2): passing-first-downs (raw 0.095 → debiased 0.025),
  passing-yards (raw 0.093 → debiased 0.030).
- WNBA flips (2): DREB (raw 0.083 → debiased 0.057), STL (raw 0.096 →
  debiased 0.070).

**Counts after Step 0.5:** NBA 18/21 (PAST 75 % target of 16),
NFL 11/20 (need +4 more), WNBA 12/18 (need +2 more). Total 41/59 = 69 %.

## Strategy encoding (how cells declare what they train with)

Per-cell ML config lives in
[`src/sportstradamus/data/config/stat_meta.json`](../src/sportstradamus/data/config/stat_meta.json)
(committed). Each cell has three relevant fields for Ship 75:

```json
{
    "NBA": {
        "PTS": {"dist": "SkewNormal", "shipped": "devel", "strategy": "ratio_meanyr"},
        "FG3M": {"dist": "ZINB", "shipped": "devel", "strategy": "none"},
        "FGA": {"dist": "SkewNormal", "shipped": "withheld", "strategy": "none"}
    }
}
```

- **`dist`** — distribution family (`"SkewNormal"` / `"ZINB"` /
  `"NegBin"` / `"Gamma"` / `"ZAGamma"`). Drives which pipeline branch
  consumes the strategy slug.
- **`shipped`** — release surface. `"withheld"` skips training + prunes
  the pickle; `"devel"` ships on the production-tracking branch (in
  14-day Gate-2 soak or already graduated); `"main"` is Gate-2
  graduated. The production server tracks `devel`, so any cell with
  `shipped in {"devel", "main"}` is live in production.
- **`strategy`** — training-pipeline strategy slug. Currently encodes
  *target normalization* (the SkewNormal forward / decode transform in
  `training/baselines.py`); future strategies are intended to encompass
  any per-cell pipeline tweak — post-processing (Step 1's isotonic /
  affine-ROE bias correction), calibration overrides, alternate
  priors, etc. Count-branch cells (ZINB / NegBin / Gamma / ZAGamma)
  carry `"none"` today because no strategy targets that branch yet.
  When a strategy lands that count cells can opt into (e.g. Step 4.3's
  `zinb_mode`), it gets registered in `baselines._STRATEGIES` and a
  cell adopts it by editing the one field.

**Hard invariants** enforced at
[`training/ship_config.py:load_ship_config`](../src/sportstradamus/training/ship_config.py):

- SkewNormal cells MUST have a real strategy slug (never `"none"`) when
  shipped — `"none"` for a SkewNormal cell means the pipeline can't pick
  a target transform.
- Count-branch cells MUST have `strategy == "none"` — the SkewNormal
  branch's slug would be silently ignored otherwise, which is
  misleading.
- `shipped` must be one of `{"withheld", "devel", "main"}`.

**Promotion (one-line edits):**

| From → to | Means |
|---|---|
| `"withheld"` → `"devel"` | Cell cleared Gate 1; ship to production. Manual edit. |
| `"devel"` → `"main"` | Cell cleared Gate 2 graduation. Done by `generate-ship-config --branch main` (monthly cron). |
| `"main"` → `"devel"` | Cell stopped passing Gate 2. Same cron, same PR. |
| `"devel"` → `"withheld"` | Cell pulled for rework. Manual edit. |

**Calibration values** (`cv`, `std`, `zi`) live in
`stat_calibration.json` (**gitignored** — runtime-recomputed by
`meditate`'s `report()` on every run). The committed `stat_meta.json`
holds only the semi-stable ML config above.

## Lifecycle table (operator maintains this)

Status values:

| Status | Meaning |
|---|---|
| `shipped` | `stat_meta.json` has `shipped: "main"` (Gate-2 graduated) or `shipped: "devel"` (in 14-day soak); both ship on the production-tracking `devel` branch |
| `soak` | In 14-day Gate-2 window (`shipped: "devel"`) — counts toward 75% numerator |
| `ready` | Cleared Gate-1 offline, awaiting promotion via `stat_meta.json` edit (`shipped: "withheld"` → `"devel"`) |
| `g{n}-fail` | Failing one or more gates; primary failure listed first |
| `withheld` | Under rework, model pruned (`stat_meta.json` has `shipped: "withheld"`) |
| `deferred-90` | Failed ≥ 4 levers, kicked to Operation Ship 90 |
| `degenerate` | Structurally untestable (e.g. actual IQR = 0); decision pending Step 0 |

Lever-attempts counter is the safety on the per-cell pivot policy. User
mandate: *push every cell until it ships OR fails ≥ 4 levers — no easy
out.*

**NBA — 18/21 shipped post-Step-0.5, PAST 75 % target of 16/21**

| Market | Family | Status | Strategy | Gates failing | Next step | Levers |
|---|---|---|---|---|---|---|
| MIN | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak (counts) | 0 |
| PA | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| PR | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| PRA | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| PTS | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| RA | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| REB | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| AST | SkewNormal | ready (Step 0.2 flip) | `ratio_meanyr` | — (g4 0.46→0.68) | Step 0.3 promote | 0 |
| BLK | ZINB | ready (Step 0.2 flip) | `ratio_meanyr` | — (g4 0.28→1.00) | Step 0.3 promote | 0 |
| FG3A | SkewNormal | ready (Step 0.2 flip) | `ratio_meanyr` | — (g4 0.47→0.67) | Step 0.3 promote | 0 |
| FTM | ZINB | ready (Step 0.2 flip) | `ratio_meanyr` | — (g4 0.36→0.67) | Step 0.3 promote | 0 |
| OREB | ZINB | ready (Step 0.2 flip) | `ratio_meanyr` | — (g1 auto-pass, g4 0.49→1.00) | Step 0.3 promote | 0 |
| STL | ZINB | ready (Step 0.2 flip) | `ratio_meanyr` | — (g4 0.41→1.00) | Step 0.3 promote | 0 |
| BLST | ZINB | ready (Step 0.5 flip) | `ratio_meanyr` | — (g5 0.091→0.064 debiased) | Step 0.5 promote | 0 |
| DREB | SkewNormal | ready (Step 0.5 flip) | `ratio_meanyr` | — (g5 0.087→0.061 debiased) | Step 0.5 promote | 0 |
| FG3M | ZINB | ready (Step 0.5 flip) | `ratio_meanyr` | — (g5 0.077→0.053 debiased) | Step 0.5 promote | 0 |
| FGA | SkewNormal | g5-fail | `ratio_meanyr` | g5 debiased 0.067 (raw 0.093); g2 z=0.91 | Step 1 isotonic + ECE | 0 |
| FGM | SkewNormal | ready (Step 0.5 flip) | `ratio_meanyr` | — (g5 0.080→0.054 debiased) | Step 0.5 promote | 0 |
| PF | ZINB | g5-fail | `ratio_meanyr` | g5 debiased 0.121 (raw 0.147); g1 auto-pass | Step 1 ECE calibration | 0 |
| TOV | ZINB | ready (Step 0.5 flip) | `ratio_meanyr` | — (g5 0.078→0.049 debiased) | Step 0.5 promote | 0 |
| fantasy-points-prizepicks | SkewNormal | g2-fail | `ratio_meanyr` | g2 (z=0.52); g5 0.032 debiased | Step 1 isotonic | 0 |

**WNBA — 12/18 shipped post-Step-0.5, need +2 to hit 14/18**

Step 0.5 G5 debias flipped 2 cells (DREB, STL) to ship. Six WNBA cells
still kill on non-G5 gates (AST g3-fail, BLK g3+g4-fail, BLST g5-fail,
FG3M g4-fail, PRA g1-fail, TOV g5-fail) — Step 1 attacks these.

| Market | Family | Status | Strategy | Gates failing | Next step | Levers |
|---|---|---|---|---|---|---|
| FGA | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| FTM | ZINB | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| MIN | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| OREB | ZINB | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| PA | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| PR | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| PTS | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| RA | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| REB | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| fantasy-points-prizepicks | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| AST | SkewNormal | g3-fail | `ratio_meanyr` | g3 (z=0.59); g4 now 0.70 | Step 1 affine ROE (bench) | 0 |
| BLK | ZINB | g3+g4-fail | `ratio_meanyr` | g3 (z=0.61), g4 (0.00 — under-spread, model over-pinned at 0) | Step 1 + dispersion review | 0 |
| BLST | ZINB | g5-fail | `ratio_meanyr` | g5 debiased 0.095 (raw 0.121) | Step 1 ECE calibration | 0 |
| DREB | SkewNormal | ready (Step 0.5 flip) | `centered_additive_mean10` | — (g5 0.083→0.057 debiased) | Step 0.5 promote | 0 |
| FG3M | ZINB | g4-fail | `ratio_meanyr` | g4 (0.50, borderline) | Step 1 + Step 4 widen | 0 |
| PRA | SkewNormal | g1-fail | `ratio_meanyr` | g1 (CI_HI=0.0028, barely fails) | Step 4 widen OR retrain | 0 |
| STL | ZINB | ready (Step 0.5 flip) | `ratio_meanyr` | — (g5 0.096→0.070 debiased) | Step 0.5 promote | 0 |
| TOV | ZINB | g5-fail | `ratio_meanyr` | g5 debiased 0.095 (raw 0.121) | Step 1 ECE calibration | 0 |

**NFL — 11/20 shipped post-Step-0.5, need +4 to hit 15/20**

| Market | Family | Status | Strategy | Gates failing | Next step | Levers |
|---|---|---|---|---|---|---|
| fantasy-points-prizepicks | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| fantasy-points-underdog | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| receiving-yards | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| receptions | SkewNormal | shipped | `centered_additive_mean10` | — | Gate-2 soak | 0 |
| yards | SkewNormal | shipped | `ratio_meanyr` | — | Gate-2 soak | 0 |
| passing-tds | ZINB | ready (Step 0.2 flip) | `ratio_meanyr` | — (g4 0.23→1.14) | Step 0.3 promote | 0 |
| receiving-tds | ZINB | ready (Step 0.4 degenerate-pass) | `ratio_meanyr` | — (g4 0/0→1.0) | Step 0.3 promote | 0 |
| rushing-tds | ZINB | ready (Step 0.4 degenerate-pass) | `ratio_meanyr` | — (g4 0/0→1.0) | Step 0.3 promote | 0 |
| tds | ZINB | ready (Step 0.4 degenerate-pass) | `ratio_meanyr` | — (g4 0/0→1.0) | Step 0.3 promote | 0 |
| attempts | SkewNormal | g1-fail | `ratio_meanyr` | g1 (CI_HI>0); g4 now 0.73 | Step 4 small-n widen | 0 |
| carries | SkewNormal | g1-fail | `ratio_meanyr` | g1 (CI_HI>0); g4 0.89 | Step 4 small-n widen | 0 |
| completions | SkewNormal | g1-fail | `ratio_meanyr` | g1 (CI_HI>0); g4 0.76 | Step 4 small-n widen | 0 |
| interceptions | ZINB | g1+g5-fail | `ratio_meanyr` | g1, g5 (0.09); g4 now 1.00 | Step 1 affine + Step 4 | 0 |
| passing-first-downs | SkewNormal | ready (Step 0.5 flip) | `centered_additive_eb_meanyr_k10` | — (g5 0.095→0.025 debiased) | Step 0.5 promote | 0 |
| passing-yards | SkewNormal | ready (Step 0.5 flip) | `ratio_meanyr` | — (g5 0.093→0.030 debiased) | Step 0.5 promote | 0 |
| qb-tds | ZINB | g2+g5-fail | `ratio_meanyr` | g2 (z=0.83), g5 (0.096); g4 now 1.00 | Step 1 isotonic + ECE | 0 |
| qb-yards | SkewNormal | g2+g5-fail | `ratio_meanyr` | g2 (z=0.68), g5 (0.174); g4 1.18 | Step 1 isotonic | 0 |
| rushing-yards | SkewNormal | g1+g5-fail | `ratio_meanyr` | g1, g5 (0.092); g4 18.7 (over-spread) | Step 4 + dispersion ceiling | 0 |
| sacks-taken | ZINB | multi-fail | `ratio_meanyr` | g1, g2 (z=0.55), g5 (0.084); g4 now 1.00 | Step 4 (likely defer-90) | 0 |
| targets | SkewNormal | g2-fail | `ratio_meanyr` | g2 (z=0.52); g4 0.89 | Step 1 isotonic | 0 |

> Update this table after every `compression_eval` re-run. The
> `lever_attempts` column is the safety on the per-cell pivot policy.
> Step 0 gate-definition changes do NOT count as a lever attempt.

## Step plan (cheapest-first, ships per cell on Gate-1 pass)

### Step 0 — G4 IQR gate audit (PRECONDITION; DONE 2026-05-23)

**Verdict: Outcome B (analytical IQR, pooled across rows) shipped.**
10 cells flipped to ship under the new gate with no retrain — NBA +6,
NFL +4, WNBA +0 (WNBA cells flipped G4 but stayed killed on G5 ECE).
Per the recovery in the 5–11 band, Step 1 proceeds with **reduced
scope** focused on the G5-only failures (now the dominant mode).

Full deliverables:
- Research brief: `/tmp/researcher_g4_audit.md`
  (Outcome B, pooled, `0/0 → 1.0` for degenerate cells)
- Reproducible numerical experiment: `/tmp/g4_iqr_experiment.py`
- New helpers in
  [scripts/compression_eval.py](../src/sportstradamus/scripts/compression_eval.py):
  `_zinb_ppf`, `_infer_dist_from_columns`, `_decode_sn_loc_scale`,
  `_iqr_pred_analytical`, extended `_gate4_iqr_spread` signature.
- Golden tests: 14 new in
  [tests/golden/test_compression_eval.py](../tests/golden/test_compression_eval.py)
  covering NB / ZINB / SkewNormal / Gamma analytical IQR, degenerate
  `0/0 → 1.0`, oracle back-compat, gate_row column-detection dispatch.
- Per-cell stat_meta strategy lookup wired in compression_eval main
  (the SkewNormal decode mirrors `baselines._ratio_decode_scale`).
- `tier0_scorecard.csv` regenerated as
  `ship75-step0-g4audit-2026-05-23`.

Historic plan (for the record):

**0.1 — Dispatch `research-analyst` subagent.** Question: is
`_iqr(point_pred) / _iqr(actual)` a fair sharpness gate for discrete
low-mean count distributions? Compare to literature standards: PIT
histograms, CRPS, sample-based IQR from M predictive draws, analytical
IQR from the parametric distribution. Brief lands at
`/tmp/researcher_g4_audit.md`.

Sub-questions:

- What's the right denominator for sharpness on integer-valued, low-mean
  targets where `p75(actual) − p25(actual) ∈ {0, 1, 2}`?
- For NB(r, p), ZINB(r, p, π), SkewNormal(loc, scale, alpha): give
  analytical IQR formulae. Worked example: NBA OREB and NBA AST.
- Per-row average IQR (one draw set per offer) vs pooled IQR (all draws
  pooled across offers)?

**0.2 — Implement the audit outcome.** Three possible code paths in
[scripts/compression_eval.py:488-498](../src/sportstradamus/scripts/compression_eval.py)
`_gate4_iqr_spread`:

- **Outcome A (most likely):** sample-based predicted IQR. Per row,
  draw `M = 1000` from the predicted distribution; pooled draws'
  `p75 − p25` is `g4_iqr_pred`.
- **Outcome B:** analytical IQR. `_iqr_analytical(dist_name, params)`
  uses `scipy.stats.<dist>.ppf(0.75) − ppf(0.25)` per row, averaged.
- **Outcome C (unlikely):** G4 correctly defined; Step 1 onward
  absorbs the load.

Re-run scorecard. Cells flipping to pass G4 → `status=ready`.

**0.3 — Promote G4-flips to `stat_meta.json`.** Per cell: set
`strategy` to a real slug (`ratio_meanyr` unless A/B says otherwise) for
SkewNormal cells, leave `strategy: "none"` for count-branch cells; set
`shipped: "devel"`. Run `meditate --force`, re-run `compression_eval`,
ship via `devel-ship-curator`.

**0.4 — Degenerate-IQR cells** (NFL receiving-tds, rushing-tds, tds).
Decide per cell:

- If audit adds an analytical IQR path: NB quantiles can be 0 on both
  ends — ratio convention `0/0 → 1.0` ships them.
- Else: `status=degenerate`. Audit whether to exclude from the 75%
  denominator. Either fix or kick to Operation Ship 90.

**Step 0 verification:**

- New golden test in `tests/golden/test_compression_eval.py` asserting
  deterministic IQR computation on synthetic NB / SkewNormal samples.
- Re-run `compression_eval` writes new `tier0_scorecard.csv`; diff vs
  the 2026-05-22 snapshot is committed.
- Sanity: cells that pass G2/G3/G5 and were g4-fail should pass G4 now
  under Outcome A or B. If not, audit is wrong; revert and treat G4 as
  correct.

**Step 0 stop criterion:** if recovery ≥ 12 cells, Step 1 becomes the
NFL-specific track. If < 5, Step 1/2 carry the full load.

### Step 1 — Post-hoc mean-bias correction (1–2 weeks)

Targets residual G2 (star) / G3 (bench) bias cells after Step 0.
Per 2026-05-22 research verdict (see
[references doc](operation_ship_references.md) §8): post-hoc
correction is the only lever that moves both bias bands by construction.

**1.1 — Affine ROE** first: fit `y = a + b·ŷ` on validation;
`ŷ_corrected = a + b·ŷ`. Works at any sample size.

**1.2 — Isotonic on prediction** where decile miscalibration is curved
(compression shape). `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`.
Higher-mean NBA / WNBA cells.

**1.3 — Per-decile multicalibration** as fallback for one cell *just*
outside bound. Don't lead with this — GBDTs already near-multicalibrated
([ref 45]).

**Implementation seams:**

- New `_step_fit_bias_correction` between `_step_decode_predictions` and
  `_step_fuse_predictions` in
  [training/pipeline.py](../src/sportstradamus/training/pipeline.py)
  `train_market`. Fit on validation.
- New pickle key `bias_correction` (dict:
  `{"type": "roe"|"isotonic", "a": ..., "b": ..., "isotonic_blob": ...}`).
- New apply step in
  [prediction/model_prob.py](../src/sportstradamus/prediction/model_prob.py)
  before SkewNormal decode (~line 276) and NegBin / ZINB decode
  (~lines 259-272). Apply **before** `fused_loc` in
  [helpers/distributions.py:314](../src/sportstradamus/helpers/distributions.py).
- Pickle round-trip test; legacy pickles load via
  `filedict.get("bias_correction", None)`.
- Inference-path integration test (see "Inference-path checklist" in
  the [references doc](operation_ship_references.md) §12).

**Selection per cell** (encoded via a future per-cell `bias_correction`
field on `stat_meta.json` — the same one-cell-per-line shape that today
carries `dist` / `shipped` / `strategy`):

- **NBA / WNBA bias-failing cells:** isotonic.
- **NFL count cells** (interceptions, TDs, sacks): affine ROE only
  (too few positive events for isotonic tails — [ref 48]).
- **Bench-quartile-only failures** (WNBA AST g3-fail): affine ROE
  with one-sided constraint (correct only the upward bottom-decile bias).

**1.4 — Validate.** Critical guardrail: post-hoc must NOT worsen
`brier_skill_score`. Compute CRPS + log-loss + Brier on validation
before/after. **Reject** if BSS drops > 0.01.

**Step 1 stop criterion:** re-run `compression_eval` per cell; each
passer → soak. Cells with both G2/G3 fixed but G4 still failing →
Step 2.

### Step 2 — Leakage-safe expanding-mean / EB-shrunk player feature (1 week)

Rank-2 lever. Trees need per-player level the leaf-averaging compresses
away.

**2.1 — `MeanYr_expanding_shifted`.**
`groupby(player_id).expanding().mean().shift(1)` of the target stat.
Plus stat × opponent variant:
`groupby([player_id, opponent_team]).expanding().mean().shift(1)`.

**2.2 — EB-shrunk variant** ([ref 47]):
`μ̂_player = (n·μ̄_player + K·μ̄_global) / (n + K)`
with `K = σ²_within / σ²_between`. Fallback when expanding is sparse.

**2.3 — Inference mirror** at
[stats/base.py:597](../src/sportstradamus/stats/base.py) `get_stats`.
Leakage-safe (strict `<` date filter, mirroring `MeanYr` / `Mean10`).

**2.4 — Leakage test.** Extend
[`tests/test_meanyr_mean10_leakage.py`](../tests/test_meanyr_mean10_leakage.py).

**2.5 — Feature filter.** `meditate --rebuild-filter` regenerates
`feature_filter.json`; verify new feature passes (or pin via SHAP
override).

**Step 2 stop criterion:** retrain affected cells; re-run
`compression_eval`. Each passer → soak. SHAP importance < 0.001 →
revert and route to Step 3.

### Step 3 — Opponent-defense × player + blowout flag (1 week)

Rank-3 lever. Narrowest; targets FG3M-style bottom-decile overshoot.

**3.1 — Opponent-defense interaction.** Player stat × opponent
defensive profile from
[`stats/base.py:623`](../src/sportstradamus/stats/base.py)
`profile_market` (extract from logging to a callable).
New feature: `player_stat_vs_opponent_defense_rank`.

**3.2 — Blowout / garbage-time flag.** Projected point-differential
bucket from book moneyline + spread. Optional: minutes-at-risk.

**3.3 — Same inference mirror + leakage audit** as Step 2.

**Step 3 stop criterion:** cells still failing after Steps 0/1/2/3 →
lever count 3, eligible for Step 4.

### Step 4 — Per-cell pivots and gate widening (rolling)

For cells failing 3+ levers (user policy: push until ships OR fails
≥ 4 levers).

**4.1 — Targeted gate audit.** Example: G1 fails on small-sample NFL
(carries 895 rows, interceptions 383 rows) — wide CI is a sample
issue. Widen `_GATE1_CI_HI_MAX` from `0.0` to `+0.005` only for
small-n (< 1000) cells, AND only after demonstrating profitability in a
14-day live A/B. Document every widening in this doc with rationale and
flag for Gate-2.

**4.2 — Strategy A/B per cell.** If `ratio_meanyr` fails, try
`centered_additive_eb_meanyr_k10` or `centered_additive_mean10`. P1
audit was crippled-HP; fresh A/Bs may flip cells.

**4.3 — Hurdle vs joint ZINB.** Per-cell `zinb_mode` becomes a fourth
field on each `stat_meta.json` cell:
`{"dist": "ZINB", "shipped": "devel", "strategy": "none", "zinb_mode": "joint" | "hurdle"}`.
Loader + validation in `training/ship_config.py`; defaulted to `"joint"`
for cells without the field so the schema migration is backward
compatible.

**4.4 — Lever cap.** After 4 attempts, cell moves to `deferred-90`
with a one-line documented reason.

### Step 5 — Family-build re-entry (deferred)

CMPμ / marginalized-hurdle / MZINB family build is **deferred** per
Stage B1.5 §7a verdict (see [references doc](operation_ship_references.md) §7).
Re-entry condition (per cell): still kills after Steps 0–4 **AND**
conditional Dunn–Smyth RQR variance < 0.70 **AND** Poisson GBM tracks
top decile while NB compresses. Score only on cells reaching Step 5;
if zero qualify (likely), family build stays parked.

## Tier-1 supersession (preserves shipped cells)

Every change to a **baselined** cell must clear
[`compression_eval.supersede_verdict()`](../src/sportstradamus/scripts/compression_eval.py)
(line 851) before replacing the baseline (i.e. before editing the cell's
`strategy` / `dist` in `stat_meta.json`). Computes:

- **S1**: five Tier-0 gates on candidate
- **S2**: paired Brier 95% CI (candidate vs baseline)
- **S3**: paired Sharpe on Kelly sim
- `ship = s1_pass AND s2_pass AND s3_pass`

**Plan integration:**

- Step 0 IQR audit re-scores baselines too. Shipped cells that fail
  the new G4 → `status=g4-fail`, returned to the queue.
- New-baseline first-ship uses Tier-0 only (absolute gates), not Tier-1.
  This is the "Tier-0 absolute-only" mode the old plan promised; confirm
  via Step 0.
- Add `tests/golden/test_supersede_verdict.py`:
  - Candidate clearing Tier-0 but losing paired Brier → ship = False.
  - Candidate tying Tier-0, winning paired Sharpe, BSS not worse → ship = True.

**`devel-ship-curator` agent** at
[`.claude/agents/devel-ship-curator.md`](../.claude/agents/devel-ship-curator.md)
carves every per-cell PR. Confirm its denylist (`compression_eval.py`,
`zinb_routing_diagnostics.py`, `icc_diagnostics.py`, `statsmodels`,
`/tmp` harnesses) covers any new Step-0 / Step-1 research scaffolding.

## Verification

**Always-on (every commit, every PR):**

```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/         # incl. compression_eval, gate tests
poetry run pytest -m integration        # fake-mode, no network
```

**Per-step gates:**

- Step 0: golden test for new `_gate4_iqr_*` path; diff-scorecard committed.
- Step 1: pickle round-trip test for `bias_correction`; live-path
  integration test (legacy pickle + new pickle).
- Step 2: extended leakage test in
  [`tests/test_meanyr_mean10_leakage.py`](../tests/test_meanyr_mean10_leakage.py).
- Step 3: opponent-defense feature has its own leakage test.

**Determinism gate**
([`tests/integration/test_determinism_gate.py`](../tests/integration/test_determinism_gate.py))
must stay green. Before Step-1+ ships, extend to WNBA + NFL
(`test_deterministic_mode_*_wnba`, `_nfl`); cross-league determinism is
the [old plan's lesson][lesson] — without it cross-league verdicts are
noise.

[lesson]: archive/gbdt_mean_regression_plan.md "P1 CENTERED_TARGET_NEGATIVE_RESULT lesson"

**`refactoring-specialist`** runs on every Python file touched in a
session before push (CLAUDE.md hard rule).

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Step 0 audit finds G4 correct (Outcome C) | low–med | Step 1 carries load; no structure change. |
| Post-hoc correction worsens BSS | med | Per-cell BSS guardrail in Step 1.4 — reject correction. |
| Step 1/2/3 don't recover NFL to 15/20 | med | NFL at "best effort"; degenerate cells exclude or kick to Ship 90. |
| Degenerate-IQR cells can't pass | med | Audit-driven decision in Step 0.4. |
| Determinism flakes under new pipeline step | low | Integration test before production retrain. |
| Post-baseline retrain regresses shipped cell | med | `supersede_verdict()` catches; old pickle archived for 1-cron revert. |
| > 4 lever attempts per cell | high | Defer to Ship 90 — don't grind. |
| NFL ~17 games/season noise on new feature | med | Smoke phase first; full-verification per cross-league policy. |
| `feature_filter.json` SHAP rebuild discards new feature | low | SHAP override pin; document. |

## Execution sequencing

| Week | Step | Goal |
|---|---|---|
| 1 | 0.1–0.4 | G4 audit + recompute scorecard. Promote G4-only-fails. Target: NBA 14–18, WNBA 13–16, NFL 6–10. |
| 2–3 | 1 | Post-hoc bias correction. Target: NBA 16, WNBA 14, NFL 11–13. |
| 4 | 2 | Expanding-mean player feature. |
| 5 | 3 | Opponent-defense + blowout. NFL focus. |
| 6+ | 4 | Per-cell pivots / gate widening. Lever cap 4. |
| 7+ | 5 | Family-build re-entry triage. |
| Throughout | Tier-1 | `supersede_verdict()` on every baselined-cell change. |

Done = each league shows ≥ 75% on a fresh `tier0_scorecard.csv`,
counting `shipped` + `soak` toward the numerator.

## Reading list (before touching code)

1. [`CONTRIBUTING.md`](../CONTRIBUTING.md) §Package Map
2. [`docs/STYLE_GUIDE.md`](STYLE_GUIDE.md)
3. [`docs/operation_ship_references.md`](operation_ship_references.md) — research verdicts, citations, critical files map, inference-path checklist
4. [`docs/ship_gate.md`](ship_gate.md) — current Tier-0 / Tier-1 thresholds (kept; will update if Step 0 redefines G4)

## Cross-references

- Old plan + context (read-only history):
  [`docs/archive/gbdt_mean_regression_plan.md`](archive/gbdt_mean_regression_plan.md),
  [`docs/archive/gbdt_mean_regression_context.md`](archive/gbdt_mean_regression_context.md)
- Next-rung stub:
  [`docs/operation_ship_90.md`](operation_ship_90.md)
- Research preservation:
  [`docs/operation_ship_references.md`](operation_ship_references.md)
- Live-data instrumentation (Stage 0 — shipped):
  [`src/sportstradamus/scripts/check_graduation.py`](../src/sportstradamus/scripts/check_graduation.py),
  [`src/sportstradamus/scripts/backfill_live_metrics.py`](../src/sportstradamus/scripts/backfill_live_metrics.py)
