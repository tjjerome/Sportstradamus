# Ship gate — current thresholds (quick reference)

At-a-glance reference for the model that promotes a `(league, market)` cell from
research → `devel` → graduated. Keep this updated whenever a threshold changes; it
is the human-readable mirror of the code.

**Top priority:** get a baseline *set* for **≥ 75% of the markets in every league**
(NBA ≥ 16/21, WNBA ≥ 14/18, NFL ≥ 15/20). Breadth first — see the "Top priority —
baseline breadth" section of [docs/gbdt_mean_regression_plan.md](gbdt_mean_regression_plan.md).

**Source of truth (code):** the constants and `verdict()` in
[src/sportstradamus/scripts/compression_eval.py](../src/sportstradamus/scripts/compression_eval.py).
Note: `verdict()` currently encodes only the **Tier 1** path; the **Tier 0**
absolute-only mode is the next code step.

The bounds are **deliberately wide for Phase 0**. Tighten the fractions/floor as the
bias-correction track lands improvements.

_Last updated: 2026-05-21._

---

## The two-tier ship model

A cell is in one of two regimes. The split: relative checks only make sense once a
baseline exists to improve on, so the *first* ship is absolute-only.

- **Tier 0 — set the first baseline (the priority).** A cell with no baseline gets
  one as soon as *any* candidate (including the incumbent default) clears the
  **absolute** gates. Pick the highest-BSS passer (see Tiebreaker). No relative /
  ≥5%-improvement requirement.
- **Tier 1 — supersede an established baseline.** A challenger must clear **both**
  the absolute gates **and** the relative gates to replace a set baseline. The
  ≥5% top-decile bar lives here — it stops needless churn, it does not block the
  first ship.
- **Gate 2 — live graduation.** A newly-set baseline soaks ≥ 14 days, then
  graduates on settled-offer metrics.

---

## Gate 1 — offline gate

Computed on the held-out validation + test split (`compression_eval --baseline … --candidate …`).
The **Tier** column says which conditions apply when.

| # | Condition | Current threshold | Tier | Constant |
|---|-----------|-------------------|------|----------|
| 1 | Top-mean-decile MAE improvement vs baseline | **≥ 5%** better | Tier 1 only | `MIN_TOP_DECILE_MAE_IMPROVEMENT = 0.05` |
| 2 | Global MAE regression vs baseline | **≤ 1%** worse | Tier 1 only | `MAX_GLOBAL_MAE_REGRESSION = 0.01` |
| 3 | `brier_skill_score` (BSS) | **Tier 0:** ≥ 0 (beats book); **Tier 1:** no regression vs baseline | both | `MAX_BRIER_SKILL_REGRESSION = 0.0` |
| 4a | Bottom-quartile bias — **relative** | not more **positive** than baseline | Tier 1 only | — |
| 4b | Bottom-quartile bias — **absolute** | over-predict ≤ **+30%** of quartile mean (floor **0.10**) | both | `BOTTOM_QUARTILE_BIAS_MAGNITUDE_FRAC = 0.30`, `BIAS_ABS_FLOOR = 0.10` |
| 4c | Top-decile bias — **absolute** | **\|bias\| ≤ 30%** of decile mean (floor **0.10**) | both | `TOP_DECILE_BIAS_MAGNITUDE_FRAC = 0.30`, `BIAS_ABS_FLOOR = 0.10` |

**Tier 0 (set first baseline):** rows **3, 4b, 4c** must hold (BSS ≥ 0 + both
absolute bias bounds). The **BSS ≥ 0 floor is the breadth knob** — if a league
can't get three-quarters of its markets to ≥ 0, widen toward the Gate-2 −0.02
tolerance rather than ship a book-losing cell.

**Tier 1 (supersede):** all rows (1, 2, 3, 4a, 4b, 4c) must hold.

Plus, when changing the deterministic-mode pipeline: the determinism gate must be
green for every league with cached parquets (`tests/integration/test_determinism_gate.py`).

### Condition 4 semantics — the calibration gate (both tiers)

The compression pathology has two symptoms; condition 4 guards each according to
its betting risk, **not** symmetrically:

- **Bench warmers (bottom quartile = lowest 25% of players by season mean):** the
  failure mode is **over-prediction only** (predicting a low-volume player higher
  than reality → false EV on overs). Both 4a (relative) and 4b (absolute) are
  **one-sided on the positive side**. **Under-prediction of bench warmers is
  tolerated for now.** Pooled into a quartile (coarser than a decile) on purpose:
  low-volume players generalize more than stars.
- **Stars (top decile = highest 10%):** gated **bidirectionally** (4c, on
  `|bias|`). No systematic over- *or* under-prediction — either is a calibration
  defect at the high-stakes end.

### How to tighten later

- Bench-warmer over-prediction stricter → lower `BOTTOM_QUARTILE_BIAS_MAGNITUDE_FRAC`
  (and/or `BIAS_ABS_FLOOR`).
- Star calibration stricter → lower `TOP_DECILE_BIAS_MAGNITUDE_FRAC`.
- Breadth too loose / too tight → move the Tier-0 BSS floor.
- If bench-warmer **under**-prediction ever becomes a concern, make 4b two-sided
  (`abs(...) >`), as the top-decile bound already is.

### Tiebreaker — when more than one candidate passes (mainly Tier 0)

`verdict()` is pairwise and does **not** rank candidates; `ship_config.json` holds
one strategy per cell, and live production can A/B only **one** strategy per cell at
a time. When several clear the gate (the common Tier-0 case — incumbent + each
strategy compete), pick in two stages:

1. **Screen (deterministic A/B):** rank survivors by **top-decile MAE improvement**
   — the clean signal the crippled-HP deterministic models isolate. Brier on
   crippled models is noisy; don't tiebreak on it here. Treat near-ties (~1-2 pp)
   as equal; carry the top 1-2 forward.
2. **Final pick (retrain survivors at full, non-deterministic HPs):** choose the
   highest **`brier_skill_score`** — the money metric (drives Kelly staking; what
   Gate 2 measures live). Ties → smaller global-MAE regression, then operational
   simplicity (prefer a strategy already shipped elsewhere; avoid `*_hurdle` until
   per-cell `zinb_mode` plumbing exists).

The offline brier pick is a pre-ship heuristic; the 14-day live soak (Gate 2) is
the real arbiter.

---

## Serving control — default-deny via generated ship_config

`ship_config.json` is a **generated artifact**, not hand-edited. The canonical
human-curated source is `data/gate1_decisions.json` (`{league: {market:
strategy}}` for Gate-1 passers). `generate-ship-config --branch {devel|main}`
writes `ship_config.json` exhaustively over **all** `ALL_MARKETS` cells:

- a cell that passed the branch's gate gets its decisions strategy (served);
- every other cell gets `"withheld"` — `meditate` prunes its pickle so
  `prophecize` dark-outs the market.

This is **default-deny**: only gate-passing cells serve. `--branch devel` =
Gate-1 passers (regenerate manually when `gate1_decisions.json` changes);
`--branch main` = Gate-2 graduated cells (regenerated monthly by the
`run_job.sh gate-status` cron, which opens a PR a human merges).

**Known gap:** the graduated classifier (`training/graduation.py`) uses a
proxy of Gate 2 — positive Gate-1 BSS + ≥ 200 settled offers in the 30d window
+ non-negative live book-BSS — not the full metric set below. `main` is dormant
until the live aggregator produces data, so the proxy is acceptable for now.

---

## Gate 2 — live graduation gate

Computed on the last 30 days of settled production offers
(`compression_eval --live-window 30`, `nightly.py` rolling aggregator). A cell
graduates (no further track work) when all hold:

| Live metric | Current threshold |
|-------------|-------------------|
| Settled book-BSS (30 d, ≥ 200 offers) | ≥ 0 **and** ≥ training `brier_skill_score − 0.02` |
| Empirical vs predicted over-rate | within **±0.03** over ≥ 200 settled offers |
| Top-decile live MAE | ≥ 5% better than prior-version live MAE, **or** within 5% of offline test-set MAE |
| Calibration (bottom-quartile + top-decile bias) | **mirrors Gate 1 cond. 4** over ≥ 100 settled offers: bottom-quartile over-prediction ≤ +30% of band mean (floor 0.10) and not more positive than prior; top-decile \|bias\| ≤ 30% (floor 0.10) |
| Profit-sim parlay yield | non-negative on slates containing the cell |

Promotion requires a **mandatory ≥ 14-day soak** between setting a baseline and
Gate 2; the prior pickle stays archived under `data/old_models/` for a one-pull
revert.
