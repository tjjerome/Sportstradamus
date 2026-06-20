# Ship gate — current thresholds (quick reference)

At-a-glance reference for the model that promotes a `(league, market)` cell research
→ `devel` → `main`. Keep this updated whenever a threshold changes; it is the
human-readable mirror of the code.

**Top priority:** get a baseline *set* for as many markets in every league as the gate
allows. The breadth aspiration is **≥ 75% of each league's markets** (NBA ≥ 16/21,
WNBA ≥ 14/18, NFL ≥ 15/20). The binding constraint is **model quality**, not the gate:
**75% is a model-improvement target, not a reason to loosen the gate.**

This doc describes the *thresholds*, not the *standings* — any cell count written into
prose goes stale the next `meditate`. For where each league actually stands, read the
ground truth:

- **Shipped count (the numerator):** the `shipped` field in
  [`stat_meta.json`](../src/sportstradamus/data/config/stat_meta.json) —
  `"devel"` / `"main"` ship; `"withheld"` does not.
- **The track that ships against these gates:**
  [`handoffs/model_improvement_track.md`](handoffs/model_improvement_track.md) (standings are
  never prose — its §3 ground-truth commands).
- **Per-cell gate pass/fail:** a fresh `python -m sportstradamus.training.scorecard`
  sweep (`model_stats.parquet` can lag a gate redefinition).

**Source of truth (code):** the gate constants, `gate_row()`, `apply_thresholds()` and
the `_gate*` helpers in
[src/sportstradamus/training/scorecard.py](../src/sportstradamus/training/scorecard.py).

_Last updated: 2026-06-19._

---

## The 2×2 lifecycle

A cell moves through two **edges** (set-a-baseline vs supersede an incumbent), each
with an **offline** check at research→devel and a **live** check at devel→main.

|  | research → devel (offline) | devel → main (live) |
|---|---|---|
| **Set first baseline** | The **6 gates** below | **Profitability:** positive Kelly-sized ROI on settled data |
| **Supersede incumbent** | **S1** pass all 6 + **S2** paired Brier CI + **S3** paired Sharpe (backdated Kelly sim) | **≥ +0.5% ROI vs the incumbent on live data for ≥ 2 weeks** |

A served model that drifts back outside any of its track's bounds is **withheld and
re-enters the set-baseline track** for a fresh baseline (drift monitor).

---

## research → devel, set baseline: the 6 gates (strict)

Computed by `gate_row()` + `apply_thresholds()` on the test-set CSVs dumped by
`meditate`. A cell **ships** iff all six pass. Star = top-mean **decile**; bench =
bottom-mean **quartile** (pooled coarser on purpose — low-volume players generalize
more than stars). Gate 6 is **cohort-scoped** — it only constrains `ratio_meanyr`
SkewNormal cells and auto-passes everything else.

| # | Gate | Formula | Threshold | Constant |
|---|------|---------|-----------|----------|
| 1 | **Brier vs book, paired bootstrap (non-inferiority)** | `d_i = (p_model_i − y_i)² − (p_book_i − y_i)²`; 95% percentile CI of `mean(d)` (2000 resamples, seeded) | `ci_hi < 0.005` — 95% confident the fused ensemble's Brier is at most δ worse than the book's (a tight tie or a win passes; mild-worse-beyond-δ and underpowered-wide-CI fail) | `_GATE1_NONINF_MARGIN = 0.005` |
| 2 | **Star σ-match** | `z = |mean(EV) − mean(Result)| / std(Result)` on the top-mean decile | `z < 0.5` — bias under half the segment's spread | `_GATE2_STAR_Z_MAX = 0.5` |
| 3 | **Bench σ-match** | Same on the bottom-mean quartile | `z < 0.5` | `_GATE3_BENCH_Z_MAX = 0.5` |
| 4 | **PIT-KS calibration** | `pit_ks = KS(randomized-PIT, Uniform)` of the predictive CDF (seeded draws, averaged) | `pit_ks < max(δ, 1.358/√n)`, `δ = 0.05` — whole-CDF mispricing under the larger of the vig-scale effect floor and the cell's KS α=0.05 noise floor | `_GATE4_PIT_KS_DELTA = 0.05` |
| 5 | **Equal-mass ECE** | 10 equal-mass `p_model` bins; `ece = Σ (n_b/N)·|mean(p_model) − mean(y)|` | `ece < 0.075` | `_GATE5_ECE_MAX = 0.075` |
| 6 | **Anti-shrinkage (recent-form floor)** | (`ratio_meanyr` SkewNormal only) `r_star = Σ Blended_EV / Σ Mean10` on *stable* (`\|Mean10/MeanYr−1\| ≤ 0.12`) top-MeanYr-quartile players; player-clustered bootstrap 97.5% upper bound `star_hi`, gated on `corr(Mean10, Result) ≥ 0.55` | `star_hi ≥ star_ref − 0.03` — the stable star not projected materially below the causal real-game floor (`star_ref`: basketball `0.98`, NFL `0.94`) | `_GATE6_STAR_REF_BASKETBALL = 0.98`, `_GATE6_MARGIN = 0.03` |

**Denominators are σ, not σ/√N.** Gates 2/3 use the segment's **standard deviation**
(not the standard error of the mean). With ~2000 rows per cell, σ/√N collapses to
near-zero on low-variance bench segments and the gate fires on a negligible bias;
σ keeps the yardstick at "what a typical event in the segment looks like" regardless
of N.

**Gate 6 catches MeanYr over-shrinkage the other five are blind to.** The `ratio_meanyr`
cells divide the target by a 365-day MeanYr that conflates "high historical average" with
"will regress", so the holdout itself teaches a high-volume regression real games don't
show. The model fits that holdout faithfully — top-decile `mean(pred)/mean(Result) ≈ 1.0`
— so every outcome-scored gate passes (a relative-bias rework of Gates 2/3 is equally
blind: the holdout's own stable stars are suppressed). Gate 6 instead scores the model's
*stable* top-MeanYr prediction against the players' **recent form** (`Mean10`), the one
yardstick the artifact doesn't suppress, and fails the cell when a stable star is projected
materially below the causal real-game floor (~0.99× recent form, measured from 6 seasons of
gamelog; ~0.94 for NFL position-mixed yardage). The `corr ≥ 0.55` anchor exempts cells
where recent form isn't predictive (minutes, bursty counts); the gate is **star-side only**
because the `ratio_meanyr` denominator deflates the whole distribution and so never inflates
the bench. Live symptom that motivated it: served WNBA FGA projecting a stable 13.4-shooter
at 10.1 (Win-Prob pinned at the 0.90 clamp). [research:
`/tmp/researcher_overshrinkage_gate.md`]

**Gate 1 is non-inferiority, not superiority.** Intent #3 is "the deployed ensemble
is *at least as good as* the book," so a statistical **tie passes** — we do not demand
the model *provably beat* the book offline. The ship test is `ci_hi < δ` with
`δ = _GATE1_NONINF_MARGIN = 0.005` (≈ 2% of the ≈0.25 book Brier ≈ one SE of these
estimates): the largest ensemble-vs-book Brier degradation we still call a tie. This
is a **tie tolerance, not a degradation allowance** — it admits genuine ties and wins,
still rejects provably-worse and mild-worse-beyond-δ cells, and the "don't loosen the
gate to chase the 75% breadth target" rule is unchanged (δ was set to the tie scale,
not to manufacture ships; more breadth comes from model work, not a looser δ).

Whether it is then **+EV to bet** the cell — and how much — is *not* the gate's job:
that is owned downstream by the EV calc, the Kelly sizer (`kelly_shrinkage =
clip(brier_skill_score, 0, 1)`, so a tie cell is sized to ~0 until it earns edge), and
the 14-day live Gate-2 soak. The gate certifies **deployable** (calibrated +
uncompressed + tie-or-better); the sizer + soak certify **profitable**. That split is
what makes shipping a tie safe.

**Gate 4 is whole-CDF calibration (PIT-KS), not IQR spread.** `pit_ks =
sup|F_model − F_true|` over the predictive CDF *is* the worst-case alt-line probability
mispricing: an alt line at quantile `q` is +EV exactly to the degree `F_model(q)`
matches the truth there, so the KS supremum bounds how wrong any alt line can be. The
PIT is **randomized** (each integer's probability jump spread by `V ~ U(0,1)`) so it is
exactly Uniform under calibration for count *and* continuous families — one threshold
spans both; the non-randomized mid-PIT is lattice-inflated on low counts and would fail
calibrated count cells on discreteness alone. The threshold is `max(δ, 1.358/√n)`:
`δ = 0.05` is the effect-size floor (worst-case mispricing at most ≈ the house vig, the
scale below which the model's own error is smaller than the edge it hunts), and
`1.358/√n` is the cell's KS α=0.05 critical value (never fail a cell below the
miscalibration its sample size can resolve). The old `IQR(EV)/IQR(Result)` compression
proxy was retired here — it conflated between- vs within-player spread and was
fiat-blind on count cells (`IQR(Result) = 0` ⇒ ratio 1.0 by definition); it rides along
as the reported `g4_iqr_ratio`.

**Reported, not gated** (legibility columns on `model_stats.parquet`; never in the
`ship` AND):

- `g1_has_edge` — the old strict-superiority test (`ci_hi < 0`). True ⇔ the cell
  *provably* beats the book, distinct from merely tying it. Lets dashboards/sizing see
  which passers carry a real offline edge.
- `betting_active = ship AND kelly_shrinkage > 0` — the deployable-and-staking subset.
  Breadth has two honest readings: **deployable** (`ship`) and **betting-active**.
- `g1_brier_diff_ci_hi_standalone` — the same paired-Brier upper bound on the
  *pre-blend* model probabilities (`P_standalone`), so each cell shows whether the
  standalone model or the book drives the fused pass. Populated on the next `meditate`
  (the column is dumped by the training pipeline).
- `g4_tail_pit_ks` — the Gate-4 KS restricted to the over-tail (`u ≥ 0.80` of the same
  randomized PIT), the worst-case **alt-OVER** mispricing where boosted parlay legs
  live. It is a sub-supremum of `pit_ks` (so always ≤ it): it localizes *where* the
  deviation sits. The whole-CDF `pit_ks` is a sup that **nets compensating directional
  errors** — a cell can pass Gate 4 while over-pricing the alt-over tail and
  under-pricing the standard line that cancel globally (NFL `receiving-tds`: −4% at
  Over 0.5, +2% at Over 1.5, `pit_ks ≈ 0.04` but `g4_tail_pit_ks ≈ 0.037`, i.e. ~90% of
  its deviation is in the over-tail). It is **not** a gate — the deep tail is too
  sample-starved per cell to threshold without gaming the cutoff — but it flags the
  alt-over wobblers for the fix-queue and the live soak.
- `central50_coverage`, `central80_coverage` — central predictive-interval coverage of
  the per-row distribution; names the *direction* a KS scalar cannot. Coverage
  materially below nominal (0.50 / 0.80) ⇒ the predictive is too narrow / mislocated
  (under-dispersed); above ⇒ too wide. A cell can win Gate 1 (binary over/under Brier)
  while its predictive is badly shaped — e.g. NFL `receptions` passes g1 yet reads
  `central50_coverage ≈ 0.24` — so these are the route to fixing the NFL SkewNormal
  cells *on merit* rather than loosening a gate.
- `g4_iqr_ratio` (+ `_oracle`) — the retired `IQR(EV)/IQR(Result)` compression proxy,
  kept for back-comparison only; superseded as the Gate-4 statistic by `pit_ks`.

**Auto-pass / fail conventions for blank metrics:**

- **Gate 1 blank** (no book `Odds` in the dump): **auto-pass** — no book to beat,
  model wins by default.
- **Gate 5 blank** (no `P` or no `Line`): **fail** — the cell couldn't compute
  calibration; that's a model artifact, not a free pass. Gate 5 only needs `P` +
  `Line`, NOT `Odds`, so unpriced-but-lined markets still get a real ECE.
- **Gate 4 blank** (no per-row distribution params in the dump ⇒ no `pit_ks`): **fail**
  — no credit for absent calibration evidence. The PIT statistic resolves the old
  `IQR(Result) = 0` degeneracy that made sparse "tds"-style markets undefined under the
  retired IQR gate (randomized-PIT is well-defined on count cells), so that Step 0.4
  blank-pass workaround is closed.
- **Gates 2/3 blank**: fail (couldn't compute).
- **Gate 6 blank** (off-cohort — not `ratio_meanyr` SkewNormal — or untestable: recent
  form unanchored / too few stable stars): **auto-pass** — "not applicable", the only
  blank-is-pass besides Gate 1. Gate 6 never fails for absence of evidence; it fires only
  on a positive over-shrinkage signal.

**Oracle columns.** Gates 1–5 each emit a sibling "oracle" value computed under the
deterministic-1/0 oracle (`pred = Result`; over-prob `= 1 if Result>=Line else 0`):
- Gate 1 oracle `mean = −book Brier` (and CI sits below 0), exposing the book's own
  Brier so the achievable headroom is visible.
- Gate 2/3 oracle `z = 0`.
- Gate 4 oracle `ratio = 1.0`.
- Gate 5 oracle `ece = 0`.
- Gate 6 has **no oracle** — its reference is the external causal floor, not the row's
  own (artifact-suppressed) outcome.

The σ / IQR_true denominators equal the model row, so the oracle row sizes each
gate's natural threshold.

---

## research → devel, supersede an incumbent: S1 + S2 + S3 (Phase 3)

A challenger replaces an established baseline only if **all three** hold. Computed by
`supersede_verdict()` on two row-aligned test-set CSVs via
`python -m sportstradamus.training.scorecard --baseline ... --candidate ...`.

| # | Gate | Rule |
|---|------|------|
| S1 | **Pass all 6** | The challenger clears every set-baseline gate above. |
| S2 | **Paired Brier CI** | `d_i = brier_current_i − brier_new_i` per shared event; 95% CI of `mean(d)` must have `ci_lo > 0` (CI excludes 0 in the new model's favor). |
| S3 | **Paired Sharpe (backdated)** | Run the dashboard's Kelly-sized profit-sim (`strategies/profit_sim.py`) on the shared events for each model; `sharpe_new > sharpe_current`. |

---

## Serving control — default-deny via stat_meta `shipped` field

`data/config/stat_meta.json` is the **canonical, hand-curated** per-cell
record. Every cell carries `{"dist": ..., "shipped": ..., "strategy": ...}`
where `shipped` is one of `"withheld"` / `"devel"` / `"main"`:

- `"withheld"` — `meditate` prunes the cell's pickle so `prophecize`
  dark-outs the market.
- `"devel"` — the production-tracking branch (`devel`) trains + ships
  the cell. In 14-day Gate-2 soak.
- `"main"` — the cell also passed Gate-2 graduation; it's locked in on
  `main`.

This is **default-deny**: only cells the human explicitly promotes
(`shipped` ≠ `"withheld"`) serve. Promoting a Gate-1 passer is a
one-line edit (`"withheld"` → `"devel"`). `generate-ship-config --branch
main` mutates `stat_meta.json` monthly (cron via `run_job.sh
gate-status`) to promote graduated cells `"devel"` → `"main"` and demote
the rest back to `"devel"`; the cron opens a PR a human merges.
`generate-ship-config --branch devel` only validates the current state
and prints a summary — devel promotions are direct edits.

**Known gap:** the graduated classifier (`training/graduation.py`) uses a
proxy of Gate 2 — positive Gate-1 BSS + ≥ 200 settled offers in the 30d window
+ non-negative live book-BSS — not the full live metric set above. `main` is
dormant until the live aggregator produces data, so the proxy is acceptable
for now.

---

## devel → main (live)

| Case | Threshold | Where |
|---|---|---|
| **No incumbent in main** | Positive Kelly-sized ROI on the cell's settled offers within the 30-day graduation window (`profit_sim_kelly_yield ≥ 0` AND live `book_bss ≥ 0`) | `nightly._profit_sim_kelly_yield` writes `data/live_metrics_per_market.parquet`; `check_graduation._classify_lifecycle` reads it |
| **Incumbent in main** | Challenger's live ROI **≥ incumbent's + 0.5%** over **≥ 2 weeks** of settled offers (`_SUPERSEDE_LIVE_ROI_DELTA = 0.005`, `_SUPERSEDE_LIVE_WINDOW_DAYS = 14`) | `check_graduation.supersede_live_delta`; the challenger-vs-incumbent A/B record needs `stat_meta.json` + per-model-version aggregation in `nightly.py` to fire automatically (dependency: live aggregator runs) |

A cell that drifts outside its set-baseline bounds is **withheld and re-enters the
set-baseline track** (drift monitor; the prior pickle stays archived under
`data/old_models/` for one-pull revert). Live data is dormant in dev, so the live
gate fires only after `nightly.py`'s aggregator has produced rows for the cell.

Plus, when changing the deterministic-mode pipeline: the determinism gate must be
green for every league with cached parquets
(`tests/integration/test_determinism_gate.py`).

---

## How to tighten / loosen later

- **Too many cells fail Gate 4** → raise `_GATE4_PIT_KS_DELTA` toward the next
  effect-size tier (0.06–0.075) or loosen Gate 5 (raise `_GATE5_ECE_MAX` to 0.10–0.15).
  G4 is the binding constraint at the strict bar (most NBA/WNBA skill-stat cells fail it
  on under-dispersion); G2/G3 are gentle by default under σ-norm. Do **not** tune δ to a
  cell's own KS to manufacture a pass — δ is the worst-case-mispricing-we-call-a-tie, a
  vig-scale tolerance, not a knob to hit a breadth target.
- **Quality too low** → tighten G4 (lower `_GATE4_PIT_KS_DELTA` toward `1.358/√n`, the
  pure-noise floor) or G2/G3 (`< 0.3`).
- **NFL low-N false KILLs on Gate 1** — *resolved* by the non-inferiority margin. The
  95% CI is naturally wide on cells with < 1000 events, so under the old strict
  `ci_hi < 0` a real-tie thin-N cell would straddle 0 and KILL. `ci_hi < 0.005` admits
  the genuine ties (e.g. NFL `passing-yards`, `ci_hi ≈ 0.002`) while still failing a
  wide CI that reaches past δ. This is a principled tie tolerance, **not** an N-aware
  floor that would silently widen with small N — degradation past δ still fails at any N.
