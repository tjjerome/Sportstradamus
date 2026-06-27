# Model Improvement Track

> Status: ACTIVE — Ship-75 rung
>
> **The single home of record for the model improvement track** — the lead lane of
> [`sportstradamus_roadmap_v3.md`](../sportstradamus_roadmap_v3.md). Every lever, stage,
> per-league path, validation recipe, and session rule for pushing market cells through the
> offline ship gates lives here, from the current Ship-75 rung through the D5 decision to the
> Ship-90 rung.
>
> Consolidated 2026-06-12 from four docs, now archived: `operation_ship_75.md`,
> `operation_ship_90.md`, `feature_improvement_plan.md`, `handoffs/model-track.md`
> (see [`../archive/`](../archive/); pre-consolidation text recoverable via
> `git show 805cea7:docs/operation_ship_75.md` on devel and
> `git show 46338ef:docs/operation_ship_75.md` on model-research, whose operating-loop
> refinements are folded in below). Three companions stay canonical and are **not** restated
> here: [`../ship_gate.md`](../ship_gate.md) (gate thresholds — owner-only),
> [`../operation_ship_references.md`](../operation_ship_references.md) (research verdicts,
> citations [1]–[71], commit refs), and the roadmap (program index, other lanes, deferred
> register).

## 1. Mission & money logic

**Get ≥ 75% of each covered league's markets past the six offline ship gates (g1–g6) in
[`training/scorecard.py`](../../src/sportstradamus/training/scorecard.py), then take the
Ship-90 rung when gate D5 fires.** This is the lead lane: calibrated full predictive
distributions are the input every decision engine consumes, breadth is the number of +EV
opportunities per slate, and Gate-4 PIT-KS calibration *is* alt-line pricing accuracy. Nothing
downstream out-earns its marginals.

| League | Ship-75 target | Ship-90 target |
|---|---|---|
| NBA | ≥ 16 / 21 | ≥ 19 / 21 |
| WNBA | ≥ 14 / 18 | ≥ 17 / 18 |
| NFL | ≥ 15 / 20 | ≥ 18 / 20 |

A cell counts toward the numerator once it is `shipped ∈ {"devel", "main"}` in
[`stat_meta.json`](../../src/sportstradamus/data/config/stat_meta.json) — the production server
tracks `devel`, so a Gate-1-clearing cell in its 14-day Gate-2 soak is already live and already
counts. Mantra: *don't let perfect be the enemy of good.* The gate certifies **deployable**; the
Kelly sizer and the live soak certify **profitable**. MLB and NHL have cells in `stat_meta.json`
but no breadth target until their activation gates decide (D1/D2 —
[`mlb-nhl-activation.md`](mlb-nhl-activation.md)).

**Failure is not an option at the operation level.** Individual *levers* are allowed to fail —
when one doesn't move a cell, scrap that path and take the next. The plan is built so every
league has more independent levers than it has cells to flip, and the failure branches are
written down in advance (§6, §8).

### 1.1 The lens — beat the DFS apps; the sharp book is an asset, not an adversary

Read this before anything about a "gate." It is the lens for the whole operation and the
correction that had to be repeated most often during the gate audits.

**We are beating the DFS pick'em apps (Underdog, PrizePicks, Sleeper), not the sportsbooks.**
The apps price slates of over/under and alt-line picks that players combine into parlays. We win
by handing the parlay builder **more accurate, better-calibrated probabilities than the app's
implied prices** — across the standard line and the whole alt-line ladder.

**The sharp sportsbook consensus is an asset we blend into, never a wall to beat.** It is a
weighted, closing, de-vigged consensus — by construction it sits very close to the true
probability, so beating it *standalone* is genuinely hard and is **not the goal**. `fused_loc`
blends our model with that sharp line (the ensemble that actually prices our offers) precisely so
we inherit its sharpness and add whatever marginal signal the model carries.

**Calibration is the product.** The parlay builder needs well-shaped full predictive
distributions to price alt-lines and to assemble correlated legs without compounding mispricing.
That is why Gate 4 (PIT-KS calibration) is the real quality bar — and why fixing
under-dispersion is central to the plan (§6, §7).

**Gate 1 is a no-regression guardrail, not a "beat the book" demand.** It is a *non-inferiority*
test: it certifies that blending our model into the sharp line does not make the ensemble *worse*
than the line alone. A tie passes — that is the entire point. It follows that:

- A cell that fails Gate 1 is one where the current model is **regressing the blend** — adding
  noise, not signal (typically a small-sample, under-calibrated cell). The fix is a better model
  (calibration, real signal via features), or leaning harder on the book in the blend. It is
  **never** evidence that "the market is too efficient to win," and **never** a reason to loosen
  a gate.
- A sharp book can never be a wall. The worst case is a cell where the model adds nothing and we
  ride the book line (a tie) — and even then the calibration work (g4) still matters, because the
  parlay builder still needs a well-shaped distribution around that line.

## 2. Read first (in order)

| Fact | Canonical home |
|---|---|
| Gate thresholds g1–g6, Gate-2, S1–S3, tighten/loosen | [`../ship_gate.md`](../ship_gate.md) |
| Release surface per cell | `stat_meta.json` `shipped` (git carries flip history) |
| Per-cell gate numbers | `data/training/model_stats.csv` (mirror of the parquet) |
| Lever stack, stages, per-league routing, session rules | **this doc** |
| Research verdicts, citations [1]–[71], commit refs | [`../operation_ship_references.md`](../operation_ship_references.md) |
| Program index, other lanes, decision gates D1–D7, deferred register | [`../sportstradamus_roadmap_v3.md`](../sportstradamus_roadmap_v3.md) |
| Ship mechanics how-to, package map | `CONTRIBUTING.md` |
| Session law (gates, subagents, hard rules) | `CLAUDE.md` |

## 3. Verify before you trust

Numbers drift on every ship and every `meditate`. Per-cell standings are **never** restated as
prose anywhere in this doc; derive them:

```bash
# What production tracks
git fetch origin && git log --oneline origin/devel -5

# Shipped counts per league (withheld / devel / main)
python3 -c "import json,collections; m=json.load(open('src/sportstradamus/data/config/stat_meta.json')); [print(l, dict(collections.Counter(c['shipped'] for c in v.values()))) for l,v in m.items()]"

# Per-cell gate numbers (g1–g6, pit_ks slack, ship) — rewritten by every meditate
ls -la data/training/model_stats.csv   # stale ⇒ re-run meditate before trusting

# Lifecycle per (league, market): not-shipped / in-test / graduated / demoted
poetry run check-graduation

# Feature-importance hygiene (cells per league, stale rows, active features per cell)
python3 -c "
import pandas as pd, collections
fi = pd.read_csv('src/sportstradamus/data/training/feature_importances.csv', index_col=0)
print(dict(collections.Counter(c.split('_')[0] for c in fi.columns)))
print('stale Player Player rows:', sum(i.startswith('Player Player ') for i in fi.index))
nz = (fi > fi.max().max()*0.001).sum(); print('active features per cell (median):', int(nz.median()))"

# Training-cache column counts per league (one cell each)
python3 -c "
import pandas as pd
for f in ['NBA_PTS','NFL_attempts','MLB_hits','WNBA_PTS']:
    try: print(f, len(pd.read_parquet(f'src/sportstradamus/data/training_data/{f}.parquet').columns))
    except FileNotFoundError: print(f, 'no cache')"

# Archive odds coverage per league (movement-feature / regen feasibility)
python3 -c "
import duckdb
con = duckdb.connect('archive/archive.duckdb', read_only=True)
print(con.execute('select league, min(game_date), max(game_date), count(*) from odds group by league').fetchall())"
```

Column dictionary for `model_stats.csv` / `.parquet`: CLAUDE.md §Training stats.

### Volatile product assumptions

None directly — this track prices stats, not app products. App-side payout drift is the
decision lanes' problem.

### 3.1 The gates are trustworthy (audits closed)

Every gate was independently audited; several were measuring artifacts and were fixed. A cell
that fails a gate today is failing on real model quality, not a scorer bug:

| Audit | Verdict | Status today |
|---|---|---|
| G4 IQR ratio (orig) | Sharpness-vs-calibration category error + decode/EV artifacts | **Retired** — replaced by PIT-KS; survives as report-only `g4_iqr_ratio` |
| G5 equal-mass ECE | Positively biased at finite N (false-fails up to 44% of calibrated NFL-N cells) | **Fixed** — Monte-Carlo null-bias offset; gate reads `g5_ece_debiased` |
| G2/G3 bias gates | ZINB/ZAGamma stored base-distribution μ vs zero-inclusive mean, overstating by `1/(1−π)` | **Fixed** — `_zero_inflated_mean`; gates score the fused `Blended_EV` |
| G1 strict-superiority | Over-specified ("at least as good as the book") | **Reframed** — non-inferiority `ci_hi < 0.005`; strict superiority survives as reported `g1_has_edge` |
| G1 selection-gate proposal | Precision-on-value-picks gate = multiplicity trap (0/6 NFL cells survive Bonferroni) | **KEEP g1 as-is.** Standing rule: never gate on a model-conditioned statistic |
| G1 −0.0 rounding | Sign-bit preserved; genuinely-negative CI bound in (−5e-5, 0) false-failed | **Fixed** — `np.signbit` |

### 3.2 The book we grade against is real (archive repair, permanent infrastructure)

The archived per-book `ev` for the passing/count families was a legacy klepto-era seed (every
book = consensus median ⇒ `p_book ≡ 0.5`), not a real price. Real two-sided historical prices
were re-fetched (Odds API, `backfill_historical_odds.py`) and injected rebuild-equivalently
(`inject_backfilled_odds.py`); every g1 verdict now grades against an honest book. The repair
also confirmed how sharp the asset is: honest continuous NFL volume lines sit at the median
(`p_book ≈ 0.5`, `brier_book ≈ 0.25` by construction) — per §1.1 that makes them good blend
ingredients, not walls. NFL is the hard league because its small samples (≈300–1000 rows/cell)
make the *don't-regress-the-blend* bar harder to clear — not because any market is unwinnable.

### 3.3 Diagnosis — under-dispersion is the dominant symptom

The 2026-06-03 full-board re-score (every test CSV through `compute_gates`, the exact production
path) found failure modes overwhelmingly concentrated on Gate 4, pointing one direction: **too
narrow**. Census of the then-40 withheld cells (evidence snapshot — regenerate from
`model_stats.csv`, never cite these counts as current):

| Primary failure | Count | Routed to |
|---|---|---|
| Gate 4 only (g1/g2/g3/g5 all pass) | 24 | calibration (§6.1) / normalization (§6.2) |
| Gate 4 + marginal Gate 1 (`ci_hi` 0.007–0.018) | 6 (all NFL) | calibration → features (§6.3) / blend (§6.5) |
| Gate 4 + Gate 2/3 (bias) | 3 | mean rung then scale rung (§6.1) |
| Multi-gate (g1+g3+g4) | 2 (NFL passing-first-downs, qb-yards) | hardest; features + per-position (§6.6) |
| Gate 1 only / Gate 1+5 (edge) | 2 (WNBA STL; NBA PF) | features (§6.3) |
| Pass now, un-promoted | 3 (WNBA BLK, FG3M, TOV) | promoted since — free-passer sweep is §6.0 |

The two modeling **heads fail in opposite directions** (Czado-Gneiting-Held 2009) — there is no
single "widen everything" fix:

- **Continuous (SkewNormal)** — *under-dispersed* (PIT U-shaped, central-50 below nominal:
  shipped cells sat at ≈0.44–0.46, withheld ones as low as 0.23–0.38 in the snapshot). Needs
  widening.
- **Count (ZINB / NegBin / ZAGamma)** — *over-covers* (PIT inverted-U, central-50 above
  nominal). Needs narrowing — and the negative-binomial conditional variance is bounded below by
  its mean, so a near-equidispersed-or-tighter truth is *forced* too wide: the deepest count
  cells need a both-directions family (§6.6), a structural ceiling post-hoc cannot clear.

**Root cause and what remains.** The original sin — SkewNormal received no dispersion
calibration at all (two hardcoded exclusions) — **is fixed**: the pipeline now fits a joint
`(c, skew_cal)` scale/skew against PIT-KS for SkewNormal and applies it on the served path
(§6.1 Rung B). What remains is the *residual* under-dispersion a scalar can't reach: every
SkewNormal cell starts on the raw GBDT scale — leaf-averaged, dynamic-range-compressed
(refs [3][4][30]) — and on moderate/severe cells the miss is *shape* or *location*, not pure
scale. The count families get `dispersion_cal` too, but the original objective was CRPS, not
PIT-KS (re-targeted in §6.1).

**Which axis fixes a given cell is not assumed — the honest search adjudicates (§6).** The
headroom is real: on Gate-4-only cells the g1 margins are comfortable (snapshot: NBA AST `ci_hi`
−0.028, WNBA DREB −0.064) — the blend already ties-or-beats the book on a proper score; only
the predictive shape that prices alt-lines is too tight. The mirror-image risk: over-widening
pushes probabilities toward 0.5 and can *reduce* Brier skill (g1) and shift ECE (g5) — so every
width fix is fit to a calibration target and guard-railed on g1/g5 (§6.1 go/no-go).

### 3.4 Lever verdicts — what is dead, what is alive

| Lever | Verdict | Evidence |
|---|---|---|
| Centered-target normalization (`centered_additive_mean10`) | **Alive & shipping** — out-calibrates `ratio_meanyr` on Gate 4 for several cells the scalar width fix can't reach (run the §3 shipped-counts block; several cells carry it in `stat_meta.json` today). The old P1 "dead" call judged it as a *mean-compression* fix under the pre-PIT-KS gate — superseded | refs §3 + sweep board |
| `init_score` warm-start baseline | **Dead** — byte-identical to plain NegBin | refs §3 |
| ZTNB-hurdle likelihood | **Refuted** — incompatible with the derived-π decode; would regress the shipped hurdle markets | refs §6 |
| T5 multiplicative factorization (volume × efficiency) | **Killed** — Goodman variance-of-products gives +27% predictive-variance inflation on the priced cell | refs §9 |
| Family build (CMP / GenPoisson; SHASH / JSU; skew-t) | **On the table — research-gated** (§6.6). Top-decile *mean* compression is family-invariant (the leaf average itself), but a both-directions count family fixes the over-covering count cells NegBin can't, and SHASH/JSU fixes heavy-kurtosis continuous cells | refs §7 |
| HurdleZINB (per-cell `zinb_mode`) | **Alive & shipped** — 6/8 NBA ZINB markets | refs §4 |
| Post-hoc mean correction (`roe_mean` / `isotonic_mean`) | **Alive & shipped** — `MEAN_STAGE` in [`posthoc.py`](../../src/sportstradamus/training/posthoc.py); use skeptically (§6.1 Rung A) | refs §8 |
| Post-hoc probability recalibration (`prob_recal_*`) | **Alive** — `PROB_STAGE` built, available per-cell | posthoc.py |
| Post-hoc scale/dispersion (joint `(c, skew_cal)` vs PIT-KS) | **Shipped** — route 1a-hybrid; Levi closed-form σ-scaling is a dead end (diverges 5–7000× on skewed cells) | §6.1 Rung B |
| Player-level features (expanding-mean, EB-shrunk, opp-defense) | **Alive, unbuilt** — RANK 2/3 in the breadth verdict; §6.3 M-1 | refs §8 |
| Per-position model split (NFL, T11) | **Alive, on the table** — a live lever now, not held for Ship-90 | refs §9 |
| Context-conditioned book `cv` scale law (Fitting-half, no-ladder) | **Refuted** — 0% out-of-sample book-PIT-KS gain on 45/45 cells (the context slope fits to 0); same class as `ratio_projvol`/WS2. The book CDF *is* mis-shaped (≈30/45 fail a book-only PIT-KS, magnitudes directional per the `EV`≠`ev_b` caveat) but is decoupled from the served gate at the CRPS-fit `w≈0.90` ⇒ both a standalone book rebuild and the Pooling-half blend rebuild are NO-GO (next row); residual routes to §6.6/§6.1 | audit brief `/tmp/researcher_book_distribution.md`; §10 2026-06-26 |
| Pooling-half blend-structure rebuild (BLP + decoupled location/shape) | **Refuted (this cohort)** — a two-structure honest-OOS probe (12 SkewNormal cells, 8 folds): decoupled (A) is ill-posed (PIT-KS-fit overfits location, CRPS-fit explodes PIT-KS — no common feasible region; EMOS/NGR fits by CRPS, never PIT-KS); BLP (B) is a wash (cohort-median OOS ΔKS ≈ 0; single-fold wins evaporate on resampling — `[[deterministic_ab_g4_oversell]]`; the lone g4-failer rescued 0/8). At `w≈0.90` the served predictive is already calibrated ⇒ no decalibration for the beta wrapper to repair; over-wide cells are family/shape-bound → §6.6 / §6.1. Probe-v2 (operator weight-challenge — raw dispersion-freed leg, weight+dispersion jointly refit by CRPS) hardens it: low weights explored (`λ`→0.63, `w_mix`→0.40) but 0/9 OOS gate wins. Re-probe per cohort; pre-committed design in §6.5 if reopened | probe brief `/tmp/researcher_blp_pooling.md`; §10 2026-06-26 |

## 4. Locked decisions

- 2026-06-10 — the model track stays the lead lane; other lanes never preempt it.
- 2026-06-10 — gate definitions and thresholds are owner-only; a gate change never counts as a
  lever (§8).
- Standing — ship incrementally: a Gate-1-clearing cell goes to `shipped: "devel"` and counts;
  don't hold ships for batches.
- Standing — new external data is free sources only; all five leagues get equal feature-effort
  budget (refinement for NBA/WNBA/NFL, foundation parity for MLB/NHL).

**Conflict order:** command output > `CLAUDE.md` / `CONTRIBUTING.md` > this doc > roadmap v3.
If command output contradicts prose here, the output wins — fix the doc in place (minor) or stop
and ask (material).

## 5. Module footprint & canonical paths

`sportstradamus.training` (pipeline, scorecard, report, calibration, strategy driver),
`sportstradamus.stats` (feature columns), `data/config/{stat_meta.json, ship_config.json}`,
`tests/golden/`. Prediction-side edits only via §7.3.

## 6. Stage plan

### The four axes and the strategy search

A served predictive is built in four independently-swappable stages,
`normalization → model/loss → blend → calibration`
(`target_normalization ⊥ {dist_training_loss, variance_reg} ⊥ {blending_loss_fn, fused_loc/BLP, book-recovery} ⊥ {posthoc, dispersion_cal, skew_cal}`):

| Axis | Values | Executable today | Unbuilt |
|---|---|---|---|
| **Normalization** (retrain) | `ratio_meanyr`, `centered_additive_mean10`, `centered_additive_eb_meanyr_k10`, `ratio_projvol` | 3 of 4 carry a Gate-4 SkewNormal decode (`scorecard._decode_sn_loc_scale`; EB off the dumped `GlobalMean`) | `ratio_projvol` refuted → §6.9 count offset (§6.2) |
| **Model/loss** (retrain) | dist-loss `nll`/`crps`; variance / soft-cal regularizer; calibration-constrained HP selection; σ-head stabilization | dist-loss via `--dist-training-loss`; HP selection via `--hpo-selection calibrated`; stabilization via `--stabilization MAD\|L2` (§6.1) | in-training PIT regularizer / decoupled-σ fit (§6.5) |
| **Blend** (retrain weight + research-gated structure) | blend-loss `nll`/`crps`; `fused_loc` pool; book recovery; BLP wrapper; p_book noise | blend-loss via `--blending-loss-fn` (`fit_model_weight_crps` built); current `fused_loc` pool | density-LOP fix, power de-vig, book recovery, p_book noise, free post-hoc `w`-refit (§6.5); **BLP + decoupled blend probed NO-GO** (§3.4, §10 2026-06-26) |
| **Calibration** (auto-fit, not searched) | location (`roe_mean`/`isotonic_mean`); scale+shape (`dispersion_cal` + joint `skew_cal`; count PIT-KS retarget); full-CDF (isotonic-PIT / IDR) | location + scale+shape shipped; count PIT-KS retarget via `--count-dispersion-objective pit_ks` (§6.1 Rung B′) | isotonic-PIT / IDR full-CDF recal (§6.1 Rung C) |

**The search interface.** The per-market searcher is
`training/model_strategy_driver.py` (entry point `model-strategy-driver`;
`model_strategy_search.py` underneath). `--board` searches the covered-league board and appends
the ranked result to `data/research/strategy_research_board.csv` after each cell (an interrupt
keeps partial progress); `--league L --market M [--out PATH]` runs one cell. Each corner trains
one `meditate --deterministic --target-normalization … --dist-training-loss {nll|crps}
--blending-loss-fn {nll|crps} --bypass-withholding`; `--deterministic` pins RNGs and fixed fast
hyperparameters and writes to a sandbox (`research/models/deterministic/` +
`data/test_sets/deterministic/`) so a trial never clobbers a production market.

> **Branch asymmetry — check before you start.** The driver pair lives on the
> `model-research` branch only (`ls src/sportstradamus/training/model_strategy_driver.py`).
> On `devel` the fallback is manual ranking from `model_stats.csv` (sort withheld cells by
> min-gate slack / `pit_ks` distance to threshold) plus per-cell `meditate --deterministic`
> A/Bs via the scorecard CLI. Porting the driver to devel is a §6.0 task — until it lands,
> board generation happens on model-research and ships are confirmed wherever the cell will
> ship from.

The driver searches only the **retrain** axes (normalization × dist-loss × blend-loss) as an
Optuna `GridSampler` grid (≤12 discrete corners, exhaustive and deterministic; the
`[kind, spec, stage]` `SEARCH_SPACE` flips to TPE the moment a continuous axis lands).
Calibration is auto-fit per corner — a post-hoc transform fit in milliseconds never sits in the
training loop; that cost asymmetry is the design's core. The objective is the negative
**min-gate slack**: a single scalar, positive iff the corner ships, larger with more headroom
across all six gates at once — it optimizes "ships, with margin," not Gate 4 alone.

Each corner is scored by the **honest val-fit→test gate row**
(`model_strategy_search._score_normalization`): the deterministic dump already carries the
pipeline's own validation-fit joint calibration, so the ranker calls `scorecard.gate_row` on it —
the *same* code production ships on. No test re-fit (an earlier build re-fit calibration on test
rows and oversold the screen; removed in `10306ee`). The driver is a faithful fixed-HP replica of
the production HPO pipeline; the only differences are fixed hyperparameters and sandbox write
locations.

**Nothing is deferred. Every withheld cell and every lever is a live Ship-75 candidate.** The
old `deferred-90` / "defer" / lever-cap tags are retired — they were per-axis verdicts (almost
always the calibration axis under a single normalization), and the honest sweep showed cells
stamped `deferred-90` ship under a *different* normalization. A cell leaves the board **only**
on matrix-wide exhaustion (§8) or the operator's explicit, documented denominator call.

**Two caveats on "wire in a new parameter."** (a) **Research-gate** — a parameter that changes
a *distribution family or dispersion mechanism* needs a `research-analyst` brief before it is
wired or built (§8.2); a plain knob (a normalization slug, a loss choice) does not.
(b) **Wiring an axis-value ≠ it sweeps.** A value can sit in `SEARCH_SPACE` yet not sweep
until its machinery exists, and its cost tier can shift once it does (`blending_loss_fn` carried
`crps` before `fit_model_weight_crps` existed; building it made it sweepable — on the *retrain*
tier, because the deterministic dump doesn't carry the pre-blend components a free `w`-refit
needs). "Wire it in" is sometimes "wire the axis, build the value, decide the tier — then it
sweeps."

### The operating loop (per cell)

The deterministic study only **ranks**; the real-HPO scorecard **ships**. Fixed workflow:

1. **Board = candidate generator.** `model-strategy-driver` (or the devel manual fallback above)
   returns the ranked board, one deterministic train per corner, scored on the honest
   val-fit→test gate row. A `ships=True` row is a *candidate flag*, never a ship.

2. **Carry forward the top-K (2–3) production-reproducible corners — including near-misses.**
   Include corners that *fail* the deterministic gate by ≤3% of threshold
   (`min_gate_slack ≥ −0.03`): the val→test discount tips knife-edge cells in *either* direction
   (a board passer can fail HPO; a board near-miss can clear it). **Reproducible** means the
   corner survives the server's plain `meditate`: `target_normalization`, `posthoc`, `blending`,
   and (SkewNormal) `hpo_selection` persist per-cell in `stat_meta.json`, but `--dist-training-loss`
   does **not** — so for a SkewNormal cell only the family-default `crps` dist-loss is shippable; an
   `nll`-dist corner can clear the run yet will silently retrain to `crps` on the server. The top board
   corner is not always the one that survives HPO (evidence: NBA DREB — `eb` topped the board,
   `mean10` is the corner that shipped), which is why the list is walked, not just its head.

3. **Withheld cell → real-HPO confirm → ship to devel.** Set the corner's
   `target_normalization`, `posthoc`, and `blending` in `stat_meta.json` **before** the full-HPO
   `meditate` — `scorecard._resolve_decode_strategy` reads `stat_meta.json`, not the CLI flag,
   so a stale strategy there yields a spurious Gate-4 miss. Pass the corner's
   `--target-normalization` / `--blending-loss-fn` flags to match during the confirm. Read the
   official scorecard (`model_stats.csv`). A clean **5/5** → flip
   `shipped: "withheld" → "devel"`. Walk down the reproducible top-K, shipping the first that
   clears; list exhausted ⇒ route the cell to the calibration ladder (§6.1) and record the axis
   attempt (§8). If the winner is the cell's current default strategy it is a straight confirm
   (no strategy edit). Knife-edge cells (g4 within ~0.003 of 0.05) are exactly where the
   val→test discount bites — the confirm is mandatory; never ship the deterministic score.
   **HP-selection is a confirm-time axis the board cannot see.** The deterministic study fixes one
   HP set, so it never runs the search `--hpo-selection calibrated` gates on validation PIT-KS
   (§6.1 Lever 1); the selection policy is therefore orthogonal to the board's
   normalization pick and decided only here. **Confirm every SkewNormal candidate under `calibrated`
   first** — it picks the sharpest trial that *clears* Gate-4, so it weakly dominates `loss` on g4 at
   a small g1-sharpness cost; a clean 5/5 ships with `hpo_selection: "calibrated"` persisted in
   `stat_meta.json` so the plain cron reproduces the calibrated trial instead of retraining to the
   sharper, g4-failing one. **If calibrated does not ship** (its wider σ tips g1, or no trial clears
   g4 and the logged fallback fires), **re-confirm under the default `loss`** and ship that if it
   clears, leaving `hpo_selection` unset so the cell rides the production default. Neither clears ⇒
   route to §6.1 Rung C / §6.6. (Worked example — the PA/PR/DREB re-confirm: PR shipped *only* under
   calibrated, g4 0.0516 → 0.0423, and carries the persisted field; PA was a calibrated no-op
   — fallback fired, g4 ≈ 0.058 either way — and DREB a 0.0508 near-miss, both held.)

4. **Incumbent cell with a better corner → supersede.** The higher bar of §7.1 (S1+S2+S3)
   applies. Sweep the **whole** board, shipped cells included — a shipped cell may have a better
   corner than the scale-only default it settled for.

Exploration runs always use `meditate --deterministic` (sandboxed writes) with `--market`
scoping; full-league retrains are expensive — don't run one to answer a one-cell question. Every
target cell that is withheld needs `--bypass-withholding` or the run silently skips it.

### §6.0 Stage 0 — Bookkeeping & trust (standing + one-time, all small)

Entry: none — most of this recurs.

1. **Free-passer re-score sweep (standing, monthly).** The Gate-4 redefinition was applied as
   demotions only; cells the old gate killed but the new gate passes were never auto-promoted
   (the 2026-06 sweep found three WNBA cells, since promoted). After any lever or data change
   and at least monthly, `generate-ship-config` reports `ship == True` rows still
   `shipped: "withheld"` (`graduation.free_passer_cells`, report-only on both branches, so the
   monthly `gate-status` cron logs them as `FREE-PASSER` lines); re-confirm each on the official
   scorecard and flip it by hand. The flip stays manual: if the official scorecard disagrees with
   a sweep pass, that disagreement is a scorer bug to chase before anything else ships.
   *Acceptance:* no `ship == True ∧ withheld` rows older than a month.
2. **Hole #0b decision packet — Gate-4 baseline hysteresis (owner call, prepare don't flip).**
   A fresh re-score fails several `devel` cells by a hair (a finite-sample KS statistic is a
   step function at the hard 0.05 cutoff). Options: (a) asymmetric hysteresis band — demote a
   baselined cell only at `pit_ks ≥ 0.055` (first-ship stays strict 0.05), pinned by a golden
   test and reconciled with the monthly auto-demote; (b) ship the §6.1 scale fit first (moves
   salvageable cliff cells to comfortable margin), re-score, accept the genuine demotions.
   **Recommendation: (b) first** — it needs no gate-policy change; revisit (a) only if
   demote/promote churn persists across two monthly cycles. Prepare the packet (cells, slacks,
   churn count); the owner flips.
3. **Stale-importances purge (QW-2/C-4, DONE).** The 155 all-zero `Player Player *_asof` residue
   rows (a fixed double-prefix bug) are gone from both `feature_importances.csv` and
   `feature_correlations.csv`, and `tests/golden/test_feature_importances_hygiene.py` pins their
   absence (skips where the gitignored CSVs are absent). The server purge is a full
   `see_features()` rebuild ([`training/shap.py`](../../src/sportstradamus/training/shap.py)) once
   every cell is pickled; on a partial local model set a `see_features()` rebuild would clobber
   the other cells' history, so a targeted drop of the residue rows is the local equivalent
   (`compute_market_importance`'s outer-join would otherwise re-accumulate them — only an explicit
   drop or full rebuild clears them). *Acceptance met:* §3 hygiene block reports 0 stale rows.
4. **Train/live parity harness (BUILT — `tests/golden/test_train_live_feature_parity.py`).**
   `get_training_matrix` and the live serving path share `get_stats`; the only per-gameday
   divergence is `_game_context`'s `date < today` branch (historical reads the gamelog row,
   upcoming reads `upcoming_games` + `archive.get_moneyline`/`get_total`) plus the fillna
   asymmetry (training `replace([inf,-inf],0)`, serving not). The harness drives the real
   `StatsNBA.get_stats` twice on one synthetic frozen gameday — flipping only that branch (real
   today vs a frozen `today == D`) — and asserts identical column sets, identical values on the
   `get_stat_columns` shared surface, no inf on the shared surface, and matching
   `Home`/`Moneyline`/`Total`. Synthetic logs + stubbed archive + pre-seeded comps + injected
   depth keep it a fast (~3.5 s) network-free golden test; those stubs are branch-independent
   shared state so they cannot mask the divergence. **Every later feature batch extends it**
   (the in-file parity-surface extension point) — per §7.2 step 2, a new per-gameday feature
   that does not extend this gate does not ship.
5. **Pre-register the NFL g1×dispersion experiment (hole #4 — the difference between NFL
   reaching 10 and reaching 15).** Before/after paired Brier-CI on the five g1+g4 NFL volume
   cells (attempts, carries, completions, receiving-yards, rushing-yards) under a candidate §6.1
   scale fit: does calibrating dispersion pull the marginal `ci_hi` (0.007–0.018 at the
   2026-06-03 snapshot) under 0.005 as a side effect? Run it as soon as a candidate fit exists
   for one of the five — the answer reorders the NFL queue (§6.9), so measure early, not when
   NFL stalls. *Registered decision rule:* if ≥3 of 5 cells move `ci_hi` below threshold → NFL
   path runs through Stage 1 alone; if ≤1 does → pull Stage 3 features and §6.6 per-position
   forward for NFL immediately.
6. **Port the strategy driver to devel (small code PR, separate session).**
   `model_strategy_driver.py` + `model_strategy_search.py` are research-branch-only (branch
   asymmetry note above); porting removes the branch asymmetry and lets devel sessions generate
   boards. Until then the manual fallback applies.

### §6.1 Stage 1 — Calibration axis: the post-hoc ladder (location → scale+shape → full-CDF)

Post-hoc transforms on the served predictive, free or near-free (fit on validation, applied to
the disjoint test split). Close to exhausted as a *breadth* lever alone — it fixes
width/shape/location, never signal — but it is the cheapest move on every too-narrow cell and
the prerequisite measurement for hole #4.

Entry: cell fails g4 (either direction) or g2/g3.

- **Rung A — location (`roe_mean` / `isotonic_mean`, shipped).** Affine ROE / isotonic
  `MEAN_STAGE` correctors, selected per cell via `posthoc` in `stat_meta.json`. Targets g2/g3
  bias failures. At NFL count means use affine ROE only (isotonic tails overfit at low base
  rates, ref [48]). **Operator note — skeptical:** post-hoc correction of the *predicted mean*
  edits the central tendency the model is supposed to learn; prefer fixing location at the
  source (normalization §6.2, features §6.3), carry mean correction as last resort, and ship a
  mean-corrected cell only if it also holds the g1 BSS guardrail and survives the val→test
  discount.
- **Rung B — scale + shape (joint `(c, skew_cal)`, shipped).** Per-cell scale `c` fit so the
  served predictive PIT is Uniform, inside
  [`pipeline._step_calibrate_dispersion`](../../src/sportstradamus/training/pipeline.py) with
  objective `scorecard._randomized_pit_ks`, applied via
  [`model_prob._dispersion_calibrate`](../../src/sportstradamus/prediction/model_prob.py). The
  additive skew `skew_cal` on the served SkewNormal `alpha` is fit *jointly* with `c` against
  the same PIT-KS — additive not multiplicative because the direct parameterization has a
  Fisher-information singularity at `alpha = 0` (Hallin & Ley 2014), so `alpha × k` injects no
  skew; joint strictly dominates sequential on the binding under-skew cells. Fit in
  centered-skewness space (clamp `|s| ≤ 3`), warm-start `(c, 0)`, ship at val
  `pit_ks < 0.040` (val→test discount +0.008–0.010). Fit, dump, and gate all act on the *served*
  (model × book) predictive, re-encoded to normalized space so the scorecard decode recovers it
  byte-for-byte.
- **Rung B′ — re-target the count objective (built, opt-in).** The count-branch `dispersion_cal`
  minimized CRPS, but the gate is PIT-KS — re-targeting the fit to PIT-KS is the one change
  that tightens the whole over-wide count branch (snapshot: all 21 ZINB/NegBin cells
  over-covered). Minimizing CRPS does not guarantee a calibrated PIT. Built as
  [`pipeline._dispersion_pit_ks_loss`](../../src/sportstradamus/training/pipeline.py) (mirrors
  `_dispersion_crps_loss`, swaps the objective for `scorecard._randomized_pit_ks`), selected by
  `meditate --count-dispersion-objective pit_ks`; default `crps` keeps production. The
  `0.01·log(c)²` brake is retained as the only sharpness guard — a pure PIT-KS objective has none
  (CRPS implicitly penalizes over-confidence), so watch the over-leg of g6 and the g1 acceptance
  on count cells (§8.2 open-Q). The count miscalibration is
  *uniform* (over-wide on both central-50 and central-80, 7/7), exactly the regime a single
  scale clears.
- **Rung C — full CDF (isotonic-PIT / IDR, build, free) — now the recommended fix for the
  mixed-direction SkewNormal cohort.** The scalar `(c, s)` is the bottom of
  an expressiveness ladder: a monotone spline on the PIT (Kuleshov 2018) or isotonic
  distributional regression (Henzi, Ziegel & Gneiting 2021) recalibrates the *whole* predictive
  CDF — the entire alt-line ladder — not just the single-line over-probability
  `prob_recal_isotonic` fixes. It acts on `Z = F(Y|X)` (randomized PIT for counts) so it ports
  to both heads unchanged. Prefer isotonic/IDR over conformal *calibration* on the count lattice
  — conformal yields discontinuous randomized CDFs (Marx 2022), bad for pricing a ladder.
  (Conformal *predictive distributions* are separately deferred — roadmap v3 §8.)
  **Why it is re-ranked above the family rebuilds (§6.6) for the g4-failing SkewNormal cells:**
  those cells are not uniformly under-dispersed — they are *mixed-direction shape*
  miscalibration. On the live numbers WNBA
  `fantasy-points` is grossly *over*-wide (central-50 0.91, central-80 0.99 vs nominal 0.50/0.80)
  while NFL `attempts`/`carries` are *under*-wide (central-50 ≈ 0.27/0.34); a single scale `c`
  moves every quantile together and structurally cannot fix a peaked-and-heavy-tailed cell. Rung
  C is the minimum-DOF post-hoc object with the freedom to, without retraining or touching μ.
  Dheur & Ben Taieb 2023 (arXiv:2306.02738, the largest controlled study) finds post-hoc
  recalibration attains the best probabilistic calibration — but only the *whole-CDF* rung
  realizes that, not the scalar one. Caveat: a PIT-spline overfits as readily as `skew_cal`;
  apply the same val→test discount + clamp. It recalibrates *probability*, so it cannot rescue a
  signal deficit — the under-wide-AND-mislocated NFL cells need §6.3 features in parallel.

**Training-time calibration levers (Lever 1 selection + Lever 4 stabilization, built, opt-in).**
The post-hoc rungs above polish a *fixed* trained σ-head; these two change which head you get,
upstream of the polish. Both default to current production.

- **Lever 1 — calibration-constrained HP *search* (`--hpo-selection calibrated`, SkewNormal
  cells).** Why full HPO often *fails* g4 where the deliberately under-fit deterministic run
  passes: μ and σ share one LightGBMLSS boosting process and one HP set, so the standard knobs
  move sharpness and calibration *together*; full HPO walks to the CRPS minimum (the sharpest σ
  surface), not the calibrated one, while the under-fit run's near-constant σ ≈ the marginal
  residual SD is homoscedastically calibrated "for free." The fix is the *selection objective*, not
  the loss: the Optuna objective is **search-gated** — `CRPS + M·max(0, PIT-KS − τ)`, a one-sided
  hinge on the served validation PIT-KS over the Gate-4 threshold τ (M=10) — so the TPE sampler
  *explores* the wider-σ region instead of re-ranking only the sharpest cluster; the feasible region
  stays ranked by pure CRPS (zero hinge once calibrated ⇒ no pull toward the over-dispersed marginal
  predictor, the GBR degeneracy guard). Final pick: the lowest-CRPS trial clearing τ, else the
  best-calibrated, *logged* (Gneiting-Balabdaoui-Raftery 2007 "sharpness subject to calibration" as a
  hard constraint; research brief `/tmp/researcher_hpo_objective.md`). Built across
  [`hyperparams.run_hyper_opt(calibration_penalty=…)`](../../src/sportstradamus/training/hyperparams.py)
  (+ `_penalized_objective`) and the per-trial served-PIT-KS closure
  [`pipeline._calibration_penalty`](../../src/sportstradamus/training/pipeline.py); cost is one extra
  refit per Optuna trial (the brief's `return_cvbooster` OOF variant is a deferred optimization). The
  measure is the model-only (pre-book-blend) predictive; sound because the model dominates the failing
  cells, but it does not capture a cell the *blend* would rescue.

  **Scale-bound vs shape-bound — what the search-gate can and cannot fix.** A g4-failing SkewNormal
  cell is one of two kinds, separated by the PIT central-coverage triple already in the gate row:
  *scale-bound* — cov50/cov80 *below* nominal (uniformly too narrow); the wide-σ trial exists but
  CRPS-selection never sampled it, so the search-gate rescues it (DREB: g4 0.0651 → 0.0508, the
  near-miss the gate should now close). *shape-bound* — cov50/cov80 *near* nominal yet KS still fails
  because the PIT defect *wanders* (a lumpy histogram no scalar σ can move); the family cannot
  represent the cell's shape at any σ, the logged fallback fires, and the search-gate is a no-op (PA:
  cov50 0.493 / cov80 0.813, g4 0.0581 — the calibrated arm widened σ +34% *and* raised book-skill and
  still failed). Shape-bound cells route to §6.6 **manual per-cell** family escalation
  (centered-parametrization SN → SHASH/JSU → skew-t), not to any HP knob. Diagnose with cov50/cov80
  before spending a real-HPO confirm; the scale-bound vs shape-bound *mix* across the live g4-failing
  SN cohort decides search-gate-vs-§6.6 ROI and is the cheap next read (open-Q #9).
- **Lever 4 — per-parameter gradient stabilization (`--stabilization MAD|L2`).** The only in-API
  σ-vs-μ-asymmetric knob: LightGBMLSS stabilizes each distributional parameter's gradient/Hessian
  independently, and `MAD`/`L2` damp the σ-head's large/outlier gradients (the over-fit mechanism
  Lever 1 selects around). Blunt — it also calms μ — so it is a per-cell A/B long-shot, tried only
  after Lever 1; do not promote it to a swept axis unless it ever wins (brief open-Q3, YAGNI).
  True σ-decoupling (separate σ learning rate, two-stage fit, natural gradient) is out of
  LightGBMLSS's API and statistically dominated by joint MLE — that is the deferred §6.5 research
  build, not this knob.

**Acceptance (per cell, on validation, before ship):** `pit_ks` below `max(0.05, 1.358/√n)`
**and** g1 Brier-skill drop ≤ 0.01 **and** g5 ECE < 0.075. Any served calibration object
(scale, skew, Rung-C map) must be written to the test-CSV dump (`_step_persist_artifacts`) and
applied identically at inference — pickle key + legacy fallback + byte-identical round-trip test
(§7.3).

**If it fails:** widening flips g4 but breaks g1/g5, or `pit_ks` won't move because the
mislocation is mean/skew not scale → route bias-direction cells (g2/g3) through Rung A; route
cells trading g4 for g1 to features (§6.3 — signal, not width); residual *shape* the SkewNormal
can't reach by `(c, s)` → normalization (§6.2) and blend (§6.5); family rebuild (§6.6) only on
matrix-wide exhaustion. Do not pre-tag cells dead.

### §6.2 Stage 2 — Strategy sweep: normalization × loss

Entry: driver (or fallback) available; cell on the board.

The retrain axes, searched per above. Normalization is **often decisive over calibration**: the
honest sweep ships moderate/severe cells under `centered_additive_mean10` that no scalar width
fix reaches — a centered/rate target reshapes the residual so the predictive is calibratable at
all. The EB / hierarchical-shrinkage strategy (`centered_additive_eb_meanyr_k10`) is built and
decode-tested. Training loss `nll` vs `crps` per cell: min-CRPS is more robust to
misspecification, max-likelihood slightly more efficient under correct specification
(Gebetsberger 2018) — keep the per-cell winner, subject to the operating-loop reproducibility
rule (dist-loss does not persist; only the family default ships).

**`ratio_projvol` — built, swept, REFUTED as a SkewNormal target normalization.**
The ratio form trained on `y / projected-volume` and decoded `rate × projected-volume` to model
*efficiency × opportunity*. Swept on 40 eligible NBA/WNBA + NFL cells: **ships 0/40**, −5 to −16
`min_gate_slack` worse than the incumbent on every cell the incumbent ships. The root cause is **not**
the linear dispersion decode (`scale × volume`) the symptom first suggested — a learned `volume^p`
minimises global PIT-KS at p≈1.0, and a principled compound-variance scale (√-law + usage-uncertainty)
still leaves the volume-stratified PIT-KS-max ≥ 0.111 (NBA PTS) / 0.148 (WNBA PTS) / 0.342 (NFL
rushing-yards). The binding defect is the **high-skew, partially-zero-inflated low-volume tail the
ratio target manufactures** (low-quartile actual skew +0.99 / +1.30 / +2.90; `outcome ≤ 0.5` up to 33%
for low-carry RBs) that the SkewNormal family cannot shape (research
`/tmp/researcher_projvol_dispersion.md`). Normalization axis recorded **tried-and-refuted** for
`ratio_projvol` on the swept cells (§8). The efficiency × opportunity *idea* survives only as the
count-branch **`log(volume)` NB/Tweedie offset** (§6.9) — `log(volume)` GLM offset with variance tied
to the mean and a delta-method / Monte-Carlo decode back to totals scale, **not** a width patch on the
SkewNormal decode. The `proj_*` volume features it divided by are sound (proj-MIN corr 0.79 with
realized minutes); ignoring their projection uncertainty under-states dispersion ~20–40% (fold the
volume *distribution*, not its mean, into any future opportunity head).

**Confirm selection — only confirm what can change served state.** The deterministic board ranks
every cell, but a real-HPO confirm (~1 h each) is spent only where it can move the served set:
(a) a `withheld` cell whose top *reproducible* corner ships — a net-new ship candidate; (b) a
shipped (`devel`/`main`) cell whose board-best corner is a *different* persistable config than its
current one **and** improves it — a supersede candidate (§7.1 S1–S3). **Skip** a shipped cell whose
board-best corner equals its current config: the served pickle already encodes it, so the confirm
only reproduces a known ship. Match configs on the persistable axes only (`target_normalization` +
`blending` + `posthoc`); `--dist-training-loss` does not persist, so treat it as the family default
(`crps` for SkewNormal) when comparing.

**Acceptance:** per the operating loop above — official 5/5 on the full-HPO confirm (withheld)
or S1–S3 (incumbent). **If it fails:** the cell records the normalization axis as tried (§8) and
routes to features (§6.3) or blend (§6.5).

### §6.3 Stage 3 — Feature batch 1: quick wins + the pre-registered build

Entry: §6.0 parity harness green. Targets the cells that need *signal*, not width: the NFL
g1+g4 volume five, WNBA STL, NBA PF — Gate-4 under-dispersion belongs to §6.1/§6.2, and feature
work complements, never replaces it.

All items follow the §7.2 validation protocol (leakage test → parity → regen → deterministic
A/B → real-HPO confirm) and the batching policy (features land per-league in batches — one regen
per league per batch, never one feature at a time; regen costs hours per league board). The
inert-revert rule applies to every item: SHAP importance < 0.001 ⇒ inert ⇒ revert; never carry
dead columns.

1. **QW-1 — spread, opponent implied total, game total, blowout flag (all leagues).** Game
   script conditions volume markets; the model sees only own-team `Moneyline`/`Total`. Zero new
   data: [`moneylines.py`](../../src/sportstradamus/moneylines.py) stores archive `Totals` as
   the implied team total (`(game_total + team_spread)/2`), so `Spread = Total(team) − Total(opp)`
   and `GameTotal = Total(team) + Total(opp)` come from two `archive.get_total()` calls,
   leakage-safe via the existing `at=target_at` mechanism. Implement in `_game_context` —
   **both** branches (historical gamelog + upcoming archive lookups); `Blowout = |Spread| >
   threshold` with the threshold a named per-league module-level constant (build-time operator
   decision, no magic numbers); register columns in the `Common` bucket of
   [`feature_filter.json`](../../src/sportstradamus/data/config/feature_filter.json).
2. **QW-4 — NFL schedule context.** `Weekday` (game-date day-of-week) is emitted from the
   schedule ([`nfl.py`](../../src/sportstradamus/stats/nfl.py)). `PrimeTime` + the short-week
   rest differential were built alongside it but **reverted 2026-06-20** (dead — zero importance,
   their `gametime`/`own rest`/`opp rest` inputs never backfilled; see the §10 ledger).
3. **QW-5 / A-1 — harvest more from the existing comp pool (NBA/WNBA/NFL/NHL).** The pool
   already computes pairs + distances (`_comp_pairs`, cached); emitting more aggregates is
   near-free. In `_apply_comp_features` / `_nonmlb_comp_features` add: distance-weighted comp
   **std**, comp **trend** (weighted mean of comps' `growth`), opponent-conditional comp mean on
   the **raw scale** (today only the z-score is emitted), recency weighting of comp outcomes in
   the opponent merge. Evidence: `comps z` is a stable top-15 feature at ~10% importance —
   informative and under-harvested.
4. **M-1 / B-3 — the pre-registered player-level build (highest confidence; NFL first).**
   `MeanYr_expanding_shifted` = `groupby(player_id).expanding().mean().shift(1)`, plus a
   `× opponent` variant and EB / James-Stein shrinkage where expanding is sparse (shrinkage
   strength cross-validated, landed as a named constant — do not invent a value); plus
   opponent-defense × player interaction columns (`profile_market`) and the blowout flag from
   QW-1. Mirrored exactly in training and `get_stats` (strict `<` date filter); extend
   [`test_meanyr_mean10_leakage.py`](../../tests/test_meanyr_mean10_leakage.py) before any ship.
   Pre-registered targets: NFL volume five, WNBA STL, NBA PF.

WNBA lands every shared-base item in the same PR as NBA (shared `base.py` machinery; EB-shrunk
variants preferred at half the games). **Acceptance:** per §7.2 step 5. **If it fails:** a
feature that doesn't carry g1 for an NFL cell escalates to the per-position split (§6.6); a cell
the model still can't improve leans harder on the book in the blend — rides the sharp line and
ships if calibration holds; never "unwinnable."

### §6.4 Stage 4 — Feature batch 2 + the feature-count ablation

Entry: Stage-3 batch confirmed (its regen + parity infrastructure is reused).

1. **M-2 / B-2 — EWMA recency family (all leagues).** `Avg1/3/5` are high-variance tail
   snapshots; exponentially-weighted mean/std at 2–3 half-lives adds a continuous recency axis
   to the family that already dominates SHAP. In `_rolling_features` (leakage-safe by
   construction). Expect some EWMA columns to displace `Avg3/Avg5` — fine.
2. **M-3 / B-4 — schedule-derived fatigue.** B2B, 3-in-4, opponent-rest differential,
   travel/timezone — zero-scrape derivables from schedules already fetched + one new static
   stadium lat/lon/dome table under `src/sportstradamus/data/`. Order: NBA/WNBA/NHL first
   (schedule density), then NFL/MLB. The stadium table is also D-4's dependency.
3. **M-4 / D-3 — NBA/WNBA starter flag.** nba_api boxscores carry `START_POSITION`
   historically, so it backfills leakage-clean; emit starter flag + "starters missing" count.
   Both leagues in one PR.
4. **M-5 / A-2 — comp-weight re-optimization with the stability gate.** Idle tooling exists:
   [`comp_feature_stability.py`](../../src/sportstradamus/scripts/comp_feature_stability.py)
   (YoY keep/transform/drop),
   [`evaluate_comp_features.py`](../../src/sportstradamus/scripts/evaluate_comp_features.py)
   (greedy add/remove),
   [`optimize_comp_weights.py`](../../src/sportstradamus/scripts/optimize_comp_weights.py)
   (differential evolution on Spearman). Run stability → evaluate → optimize per league; extend
   the optimizer to co-optimize K (currently min 5 / max 15–20) and the distance-kernel exponent
   (`1/(1+d)`). Spearman is in-sample to the comp objective — the arbiter is the §7.2
   deterministic A/B. Revert = restore prior `playerCompStats.json` (versioned).
5. **M-8 — the NFL small-n ablation (pre-registered; settles "too many features?").** 2–3
   small-n NFL cells from the g1-blocked volume five: full candidate set vs per-cell top-K
   |SHAP| (K ∈ {50, 100}) vs family-pruned, under `meditate --deterministic` at fixed HP, scored
   on the honest val→test gate row. **Decision rule, registered now:** adopt per-cell top-K only
   if it beats full on g1 `ci_lo` across the tested cells **and** survives a real-HPO confirm;
   otherwise record the verdict here and close the axis. Context: the no-filter rewire stands —
   70–92% near-zero-SHAP features is the expected signature of a healthy wide-candidate GBDT;
   the measured cost is wall-time, the small-n variance cost is a hypothesis. **Do not re-add
   global pre-train filtering.** The durable answer is addition discipline (the inert-revert
   rule), not subtraction; `feature_correlations.csv` feeds a redundancy audit — collapse
   |ρ| > 0.98 pairs only where one column is derived from the other by construction, and only
   through the standard A/B.

### §6.5 Stage 5 — Structural width fixes: blend rebuild + training-time regularizer (research-gated)

Entry: a cell where §6.1–§6.3 left the served predictive miscalibrated, or NFL needs the
blend-side escalation (§6.9). **Structure changes here alter the dispersion mechanism →
`research-analyst` brief first (§8.2).**

**Why the blend is the under-built axis.** `fused_loc`
([`helpers/distributions.py`](../../src/sportstradamus/helpers/distributions.py)) blends summary
*parameters*, with five measured immaturities: (1) parameter blend ≠ density pool — the NegBin
path log-blends `(μ, r)` but a true logarithmic opinion pool multiplies the PMFs pointwise and
renormalizes (parameter-blending coincides with the LOP only in the Gaussian case, so the
SkewNormal loc/σ precision-blend is correct, NegBin/ZINB is an approximation wearing the label);
(2) the book side is a crude symmetric `N(ev_b, (ev_b·cv)²)` with a single per-cell `cv` from
`stat_calibration.json`, skew forced to 0; (3) skew is a linear shrink (`w·model_skew`); (4) the
pool can only sharpen, never widen — widening is bolted on afterward as the scalar `c_opt`; (5)
no noise model on `p_book`. The de-vig (`no_vig_odds`) is proportional, and one-sided lines
fabricate a flat 6.5% under.

**Fitting half — turn one de-vigged point into a distribution. Power de-vig (a) + count-tail reg
(c) BUILT + committed; the shape-borrow book-skew mechanism (b) was BUILT, A/B-REFUTED, and
REVERTED** — production books are symmetric (skew=0), as before this track, until real alt-line
ladder data (WS1) accrues. (a) Replace proportional
de-vig with the **power (logarithmic) method** — preserves [0,1], handles favourite-longshot
bias on asymmetric / anytime-TD lines (Clarke, Kovalchik & Ingram 2017); flag cells with
`|p_over − 0.5| > 0.3` for extra downstream shrinkage. (b) The book *distribution* recovery —
fix the model's shape `(σ̂, α̂)` / `(θ̂, π̂)` and solve the single location so the model-shaped
CDF passes through the de-vigged point (the line is a *median*, not a mean — that is why skew
matters) — was built closed-form (no root-find) with a **non-circular** per-cell skew prior
(each player's standardized residuals of *realized* outcomes → SkewNormal α, **not** the live
model row's shape, which would flatter the model in g4 per Ranjan-Gneiting). **A/B-refuted and
reverted:** the within-player *marginal* residual skew (α≈3–4 on NBA PTS/REB) over-states the
book's *conditional* skew — which the model already carries — so it shifts the blended mean 8–11%
and **worsens g4/g2 on every tested cell even at w=0.9** (g4: PTS 0.016→0.063, REB 0.027→0.083,
MIN 0.031→0.053). The right source is a per-line ladder fit (WS1, once rungs accrue), not the
marginal residual; until then the book stays symmetric. (c) Count tail case (anytime-TD):
`λ = −log(1−p)` / NegBin `μ = θ((1−p)^(−1/θ)−1)` are ill-conditioned as p→1 — regularize toward
the model's `μ̂` and cap `|μ_book − μ̂| ≤ K·SD` (`_sanitize_book_ev`, BUILT + committed).

**The book reconstruction itself is audited and exonerated as a standalone rebuild.** Turning one
de-vigged point into a full book distribution is genuinely mis-shaped (≈30/45 cells fail a
book-only PIT-KS — SkewNormal book too narrow, count too wide, the same directions as the served
heads), but the mis-shape is **decoupled from the served gate** at the CRPS-fit `w≈0.90` (the
served blend passes ≈15/18 where the book fails; the lone book-dominated cell, NFL attempts
w=0.05, has served g4 0.214 ≫ book 0.088). A context-conditioned `cv = f(STDYr/MeanYr)` scale law
is **refuted** (0% out-of-sample on 45/45 — a single point is sufficient for *location* but cannot
carry *shape*: Dmochowski 2023; Ranjan-Gneiting 2010 Thm 1 puts a mis-shaped input's fix in the
**pool**, not the input). So the book's mis-shape pointed to the **Pooling half** (probed next,
below) rather than a standalone Fitting-half rebuild — the only legitimate book-shape source being
the WS1 ladder, a year+ out; the exoneration is conditional on `w≈0.90`, so a Pooling-half `w`-refit
that trusts the book more must re-audit at the new weights. **Caveat (recorded 2026-06-26):** the
audit's `book_calib.py` used the dumped `EV` column as the book mean, but `EV` is the *model* base
mean ([pipeline.py:1497](../../src/sportstradamus/training/pipeline.py#L1497)); the true `ev_b`
(recovered from `Odds`) differs by ≈1.3–2.7 points/cell, so the audit's *verdict* stands (a
more-mislocated book is even less the lever) but its book-only PIT-KS / coverage *magnitudes* are
directional, not exact. Any future book audit recovers `ev_b` from `Odds`, never reads `EV`. SkewNormal residual after the best constant `cv` is *shape* not
scale (→ §6.1 Rung C); count over-coverage is the NegBin variance-≥-mean floor (→ §6.6 count
family, which fixes the book leg for free). Report-only carve-out: a `book_pit_ks` companion to
`brier_book` (never a gate). Detail: §10 (2026-06-26).

**The Pooling-half rebuild is probed across both candidate structures and NO-GO this pass.** A cheap
two-structure honest-OOS probe (reconstruct `F_model` by inverting the precision blend at the
pickle's `w`, `F_book` from `Odds`; re-blend reproduces the served CDF to 3e-16) scored a **decoupled
location/shape blend** (A: a free location-trust `λ` toward the book line, dispersion from the model)
and the **beta-transformed CDF-mixture BLP** (B) against the served incumbent over a 12-cell
SkewNormal cohort across 8 resampled folds. Both fail. **A is ill-posed:** a PIT-KS-fit `λ` overfits
the fit-half mean (loc shift 2–7 units, ΔCRPS −0.5 to −3.7, g1 −0.01 to −0.11); fit to CRPS instead
(the correct EMOS objective, Gneiting 2005) its PIT-KS explodes — the two objectives share no
feasible region (Gneiting 2007 calibration-vs-sharpness). **B is a wash:** cohort-median OOS ΔKS ≈ 0;
its best single-fold wins (NFL carries, WNBA REB) flip sign / evaporate on resampling (the
`[[deterministic_ab_g4_oversell]]` signature), and the lone genuinely g4-failing cell (NFL fantasy
underdog, g4 0.082) is rescued by neither (0/8). Root cause matches the book audit: at `w≈0.90` the
served predictive is already the model's and near-calibrated, so the beta wrapper has no
decalibration to repair (Ranjan-Gneiting's premise unmet), and the over-wide cells are over-wide
because their *family/shape* is wrong — a pool of a wrong-shape model and a symmetric book cannot
manufacture the right shape. A follow-up probe closing the *weight* objection (that v1 tested at a
fixed `w` carrying the current dispersion-cal) — recover the **raw** dispersion-freed model leg (undo
`dispersion_cal` before inverting the blend; sanity 5e-16, `frac_ok`→~1.0 on the book-heavy cells)
and re-fit weight *and* dispersion **jointly by CRPS+hinge** — **hardens** the verdict: the joint fit
explores the low-weight operating points (`λ` to 0.63, `w_mix` to 0.40 on the book-heavy NBA
RA/PR/AST) but **0/9 cells** clear the gate OOS, and v1's lone fragile positive (NFL targets) reverses
sign (+0.013 → −0.017) under the correct objective. The weight *can* differ; it does not win.
**Route the residual to §6.6 (centered-param / count family) + §6.1 Rung C, not the pool.** Detail:
§10 (2026-06-26).

**Pooling half.** (a) Keep the log pool as base operator but fix NegBin/ZINB to a **real density
LOP** (grid-multiply the PMFs, renormalize). (b) The **beta-transformed linear pool (BLP)**
wrapper `F^BLP = B_{α,β}(w·F_model + (1−w)·F_book)` (Ranjan & Gneiting 2010) — flexibly dispersive
(narrows *or* widens), fit **outside** the five gate calculations — and its decoupled location/shape
cousin are the pre-committed structures, **both probed NO-GO this pass** (above). Revisit only for a
*genuinely over-wide* cohort at `w<0.9` (post-§6.6, or NHL/MLB on activation), re-probing per cohort
— never generalize the NO-GO. **If reopened, the design is settled:** fit `(w, α, β)` by **CRPS
(+ a PIT-KS hinge), never PIT-KS** (PIT-KS overfits location for the decoupled variant and generates
the `[[deterministic_ab_g4_oversell]]` for the BLP); bounds `α,β∈[0.5,3]`; **ship guard = the raw
linear pool `α=β=1` must sit inside the CI** (the fitted beta must strictly beat the un-warped mix
OOS, else collapse to `α=β=1`, which must itself beat the scalar blend); XOR with §6.1 Rung C per
cell (both are monotone CDF maps — stacking double-fits the held-out PIT). **Do not** substitute a
raw *linear* pool: its widening is disagreement-driven and degrades sharpness and the KS/ECE gates
(Gneiting & Ranjan 2013; Hora 2004). **Coupling — do
not sequence the two halves.** The BLP's beta transform absorbs a misshapen book input (Ranjan &
Gneiting 2010, Thm 1: a linear pool of calibrated forecasts is necessarily uncalibrated, and the
beta wrapper is precisely what re-calibrates it), so the book-shape choice (Fitting half) and the
BLP / learned `w` must be **co-fit on one held-out objective** — a book-input change must not
presume a later independent pool fit, and a pool rebuild must not presume a frozen book shape.
(c) Conjugate
noise on `p_book`: treat `logit(p_book)` as a Gaussian observation on the model's logit-CDF at
the line, per-cell variance from residual studies → precision weight. (d) Learn `w` by CRPS
(continuous) / log-score (count), shrunk per-cell toward a global prior ∝ 1/n_cell; never
hard-code book = truth. (e) Time-varying `w_book` ramp toward close (the de-vigged close is the
best probability estimate, but at close). The CLV-edge dashboard defers to roadmap v3 §8; the
weight schedule lives here.

**Model/loss adjunct (research-gated — dispersion mechanism).** The training-time variance /
soft-calibration regularizer (untried, not refuted): an MMD-to-uniform-PIT penalty (Chung 2021)
or held-out variance penalty that widens σ where the model is overconfident — attacking
under-dispersion at the source. Pick it up if normalization + blend leave a cell overconfident.
CRPS-stacking the model's own CDF variants across loss × transform via a log / beta-transformed
pool (never raw linear) is the cheap cousin. Honest caveat for both: LightGBMLSS's CRPS path
sets the Hessian to 1 (first-order), which is why a properly-curved CRPS head (NGBoost-style)
is a narrow-use Hail-Mary, not the default.

**Why this stays deferred behind §6.1 Lever 1 + Rung C.** The
in-training calibration-regularization line is real — `CRPS + λ·CumKL(PIT‖U)` (Utpala & Rai 2020,
arXiv:2002.12860) and Dheur & Ben Taieb 2024's "Quantile Recalibration Training" (arXiv:2403.11964)
report single-digit-% CRPS for a large PIT gain. But Dheur 2023 (arXiv:2306.02738) finds in-training
regularization wins the calibration-*sharpness trade-off* yet does **not** beat *post-hoc* on
calibration itself, and two-stage / decoupled-σ estimation is statistically dominated by joint MLE —
so the expected value over the *free* moves (Lever-1 selection, which spends zero new training, and
the free whole-CDF Rung C) is low while the build cost (a custom LightGBMLSS objective with a
differentiable PIT penalty, or a second library) is high. Exhaust Lever 1 + Rung C per cell before
funding this; it is the genuine escalation only for a cell that needs a properly-*curved* σ surface
no first-order CRPS trial can produce.

**Acceptance:** `pit_ks` below threshold with g1 BSS drop ≤ 0.01 and g5 < 0.075, on
validation, before ship; an inference-path round-trip test for every new served object (de-vig
method, recovered book params, BLP coefficients — §7.3). **If it fails:** a cell the blend can't
widen into calibration without a g1 hit is signal-starved → features (§6.3) or normalization
(§6.2); a genuine heavy tail the BLP can't reach → family (§6.6).

### §6.6 Stage 6 — Research-gated escalations: family, hierarchical, per-position

Entry: per-cell, when the cheaper axes are recorded-tried (§8); for NFL, when §6.9's escalation
fires. **Every family / dispersion item needs a `research-analyst` brief before build (§8.2).**
Each family swap is a one-field `stat_meta.json` edit + retrain behind `supersede_verdict()`.

- **Continuous family ladder, escalating expressiveness.** (a) **Centered-parametrization
  SkewNormal / skew-t** (Arellano-Valle & Azzalini 2008) — a loss-function change that removes
  the `alpha = 0` Fisher-information singularity at the source (distinct from, complementary to,
  the post-hoc `skew_cal` patch); try first; fixes the singularity, **not** the tails.
  (b) **SHASH / Johnson-SU** (Jones & Pewsey 2009) — 4-parameter, separate skew and kurtosis —
  for heavy-kurtosis cells the centered family leaves too thin. (c) **skew-t / Student-t** for
  the heaviest tails. **Tweedie / compound-Poisson-Gamma**
  (Smyth & Jørgensen 2002) with a **`log(volume)` offset** for zero-mass continuous cells (NFL
  rushing/receiving yards) — the preserved efficiency × opportunity fork after the SkewNormal
  `ratio_projvol` form was refuted (§6.2: the ratio target manufactures a zero-inflated low-volume
  tail SkewNormal can't shape, and no dispersion-scale law clears Gate 4). Validate the offset
  plumbing on `carries` (NB) first; gate on the **low-carry-quartile PIT-KS-max** — the stratum
  `ratio_projvol` couldn't reach — not just global PIT-KS. Research
  `/tmp/researcher_projvol_dispersion.md`.
- **Count family — the structural ceiling.** Over-covering count cells that won't narrow under
  recalibration (NegBin variance ≥ mean) need a both-directions family: **COM-Poisson** (Sellers
  & Shmueli 2010; truncate and round-trip-test the infinite `Z(λ,ν)`), **Generalized Poisson**
  (Harris 2012 — cheaper, no infinite normalizing constant), or **Double Poisson** (Efron 1986).
  Run a per-cell **plain-NB vs hurdle vs ZINB vs COM-Poisson screen** on the honest val→test
  PIT; stop defaulting ZINB on cells that aren't genuinely zero-inflated (a ZINB mixture
  inflates variance to fit zeros a single process explains, feeding the over-coverage). Hurdle
  already exists per-cell via `zinb_mode`. Re-entry evidence bar from the §7a pre-check
  (refs §7): build a count family only on a cell that still kills after the cheap fixes AND
  conditional Dunn–Smyth RQR variance < 0.70 AND a Poisson GBM tracks the top decile while NB
  compresses.
- **Small-sample / hierarchical layer (the NFL wall), cheapest-first.** Partial pooling
  dominates no-pooling and complete-pooling at n ≈ 300–1000/group (Gelman & Hill 2007).
  (a) **EB-shrink the distributional parameters** (μ, σ, ν, τ) per player toward a per-position
  mean, cross-validated shrinkage — stays in the LightGBMLSS stack. (b) **TabPFN v2
  head-to-head** on the small-n NFL/WNBA cells (Hollmann 2025): native full predictive
  distribution, no per-cell tuning, sweet-spot ≤ ~10k rows; recalibrate through the existing
  PIT gate; judge on the same honest val→test criterion; try *before* the full hierarchical
  build. GBDTs stay the backbone (Grinsztajn 2022; McElfresh 2023). (c) **Hierarchical-Bayes**
  (player ⊂ position ⊂ team) — escalation if (a)+(b) are insufficient. (The multi-task
  shared-trunk NN defers to roadmap v3 §8.)
- **Per-position model split (T11, NFL).** Train separate model per (position, market) where
  eligible-position marginals diverge materially (rushing-yards QB-scramble ~19 vs RB-workhorse
  ~37); selective, never wholesale — tight receiving stays pooled; min-row guard + fallback to
  pooled+categorical. Enabled by the Stage-A1.6 position-scoping work (refs §9).
- **Monotone priors** (`monotone_priors.json`, layered default→league→market) for NFL small-n
  volume cells — commit only priors with mechanical meaning; a wrong-sign prior is worse than
  none.
- **Research-brief-flagged feature items** (the brief settles the leakage / information-boundary
  design before any build): **C-3 line-movement & book-disagreement features**
  (`Archive.get_movement()` / `get_ev_history()` exist, never fed to models; brief must
  pre-register whether g1 gain may come from book echo — the X/B-frame boundary is deliberate —
  and resolve at `target_at`, never scrape-time; depends on B-5 or a per-feature NaN exemption).
  **B-5 missing-value semantics** (both fill sites `fillna(0)` — training matrix tail in
  `base.py`, live fill in `model_prob.py`; exempting a curated subset (H2H, comps, movement) is
  strictly more expressive but changes every cell's input distribution → brief + per-league
  deterministic A/B on 2–3 cells first; both files in one PR — parity rule). **L-3
  injuries/inactives + usage vacuum** (brief must settle report-timestamp leakage —
  announcement time vs `target_at` — and backfill honesty). **D-4 weather** (brief-note level:
  Open-Meteo free, NFL/MLB outdoor; train on realized, archive forecasts forward, document the
  forecast-vs-realized serve skew; needs the M-3 stadium table). **D-5 referee/umpire** (lowest
  priority; single-cell targets — MLB K/BB umpire zones, NBA PF crew rates — only after the
  owning league's medium items are exhausted).

### §6.7 Stage 7 — Foundation leagues (MLB / NHL)

Entry: matrix/feature builds proceed **now**; training ships only post-D1 (MLB) / post-D2 (NHL)
— gates owned by [`mlb-nhl-activation.md`](mlb-nhl-activation.md) (data freshness, book
honesty, GO/NO-GO; not restated here).

MLB/NHL are not "efficient cells with nothing to learn" — they are starved of inputs (raw
`stat_types`: MLB 13, NHL 14, vs NBA 138; ~100–120 matrix columns vs ~460–480). Closing that
input gap is this stage:

1. **L-1 / D-1 — MLB statcast breadth via pybaseball (the MLB parity centerpiece, XL).**
   Barrel%, exit velocity, xBA/xwOBA, hard-hit%, whiff%, chase%, pitcher
   arsenal/usage/velocity — free, historical, event-dated, so point-in-time aggregates are
   leakage-clean by construction. New ingest module patterned on the FP loader family
   ([`nfl_fp_loader.py`](../../src/sportstradamus/stats/nfl_fp_loader.py)); extend `stat_types`;
   join through the existing `_join_fp_player_features` / `_join_fp_team_features` base hooks.
2. **QW-3 / C-1 — MLB batting order reaches the matrix (S, after a 30-min verification).**
   Batting order and `starting batter` are already stored per gamelog row and reach
   `playerProfile["depth"]` — but `_join_defense_and_parks` then overwrites
   `Player depth = Player position` for MLB. Verify what `Player position` holds, stop the
   overwrite so batting order survives (keep a separate position flag). Document the skew:
   training sees the realized same-day lineup, live sees announced-or-mode.
3. **A-3 / L-2 — time-gated MLB comps (depends on L-1).** The one documented open comp leakage
   (`TODO(comp-leakage-mlb)` in `_ensure_comps`): MLB reuses today-state Savant affinity CSVs
   across every training gameday. Build MLB comp profiles from as-of statcast aggregates, add an
   MLB block to `playerCompStats.json`, retire the affinity CSVs to cold-start fallback; extend
   the comp week-gating tests.
4. **M-9 / D-2 — NHL foundation: goalie quality + xG breadth (greenfield).** The scraping
   already exists (dobbersports predicted goalies → `upcoming_games`, `opponent goalie` per
   gamelog row, MoneyPuck CSVs pulled); the gap is **joining** opponent-goalie SV/GSAx features
   into the matrix and widening skater/team `stat_types` with MoneyPuck xG aggregates. Validate
   NHL comps (machinery + `playerCompStats.json[NHL]` exist). **Greenfield rule: start lean** —
   there is no incumbent baseline, the first scorecard IS the baseline; do not import NFL-width
   column counts without SHAP evidence.

Anti-drift guard: foundation leagues have slow feedback (no trained cells until D1/D2) while
refinement leagues confirm fast — hold the equal-effort budget (§1) so effort does not silently
flow to wherever feedback is quickest.

### §6.8 Stage 8 — D5 → the Ship-90 rung

Entry: §1 Ship-75 targets met on a fresh scorecard (run the §3 block — never prose).

Ship-90 is the same operation at a higher bar (NBA ≥ 19/21, WNBA ≥ 17/18, NFL ≥ 18/20) — the
lever stack does not change; the remaining cells are by construction the ones that resisted the
most axes. The session's job at this stage is the **D5 decision packet** for the owner
(roadmap v3 §7); the owner flips the rung. The packet contains:

1. **Standings + resistance map.** Per-league fresh scorecard; for every still-withheld cell,
   the axes tried and verdicts (§8 records) — the Ship-90 queue is ordered by what is *left*
   per cell, mostly §6.5/§6.6 structural items by then.
2. **Gate-tightening proposal (owner-only).** Candidates: G2/G3 z-bound 0.5 → 0.3; G5 ECE
   0.075 → 0.05. Decide tighten-early (forces some Ship-75 graduates back into queue) vs
   tighten-late. **Before proposing any tightening, run the profit-sim** (`dashboard` pages
   3/4/6 over resolved history): are profitable cells currently locked out that tightening would
   lock out harder? Never tighten on aesthetics.
3. **Live-drift readiness.** By D5 the Gate-2 soak archive should hold months of live per-cell
   metrics (`data/live_metrics_per_market.parquet`, 7d/30d windows) — report whether
   drift/regime detection (e.g., league-wide 3PA trends) is signal-worthy yet.
4. **Branch-ladder question.** Ship-75 ships per-cell from research → devel via the curator. If
   Ship-90's tightened gates need a more disciplined staging pipeline, the `devel-foundation`
   buffer branch is the named option — owner decides.
5. **Inference-cost check.** Per-cell calibration objects accrete (pickle keys: `temperature`,
   `dispersion_cal`, `skew_cal`, Rung-C maps, BLP coefficients…); measure `prophecize` load
   time before the rung adds more.

Out of scope at any rung: changing the LightGBMLSS framework upstream; adding new leagues
(MLB/NHL have their own activation lane); replacing the GBDT base learner wholesale (until the
full §6 ladder is exhausted on every cell — not the case while zero cells are matrix-exhausted).

### §6.9 Per-league routing

Live counts: §3 block. Per-cell gate numbers and current candidates: `model_stats.csv` + the
sweep board. What follows is the durable routing — which lever classes plausibly flip which
cell classes — not a snapshot.

#### NBA — comfortable

- **Calibration (§6.1) / sweep (§6.2)** on the Gate-4-only cells (snapshot examples: DREB,
  FG3A, FGM, FTM, STL, TOV; AST shipped via the centered normalization, MIN and RA via the
  scale fit).
- **Rung A then Rung B (§6.1)** on the g2-bias + g4 cells (FGA, fantasy-points-prizepicks).
- **Features (§6.3)** on PF (g1+g5 — the one genuinely hard NBA cell; also a D-5 referee
  candidate, last).
- Verdict: five of the seven calibration-class cells clear the target; FGA/FP/PF are backups,
  not load-bearing.

#### WNBA — achievable

- **Free promotes** banked (§6.0 sweep — BLK, FG3M, TOV shipped 2026-06).
- **Calibration (§6.1) / normalization (§6.2)** on the Gate-4-only cells (snapshot: AST, BLST,
  DREB, FTM, OREB, PTS, RA, REB, fantasy-points-prizepicks); cells the calibration axis can't
  reach route to `centered_additive_mean10` in the sweep — current candidate status is the sweep
  board, never an in-sample floor.
- **Features (§6.3)** on STL (g1 edge). Small-n caveats: half NBA's games — EB-shrunk feature
  variants preferred, affine ROE over isotonic (§7.4.9).
- Verdict: six of the eight realistic calibration/normalization cells.

#### NFL — the binding league, with a real failure mode

- **Calibration (§6.1), g1 already passes:** passing-tds, passing-yards, yards,
  fantasy-points-underdog, sacks-taken (snapshot); receptions screened severe on the calibration
  axis but stays live with the §6.9 count-branch `log(volume)` offset and blend (§6.5) untried
  (`ratio_projvol` tried-refuted §6.2).
- **Calibration + edge — the crux (hole #4 verdict: NEGATIVE — §10):** the continuous-volume five
  (attempts, carries, completions, receiving-yards, rushing-yards) + qb-tds fail g4 plus a
  *marginal* g1. The §6.1 scale fit does **not** pull g1 under threshold (`carries` full-ladder
  real-HPO g1 0.0087; deterministic screen 0/5). The **§6.2 normalization axis does**: under
  `centered_additive_mean10`, `carries` beats the book at real-HPO (bss +0.036, g1 passes) but
  stays g4-bound (PIT-KS ~0.08 floor). g1 is reachable by target-shape, not by calibrating
  dispersion; and these cells are **not feature-starved** (game-script + interactions + recency
  already trained), so the lever is normalization + family/regularization, **not** a §6.3
  raw-feature build.
- **Escalation ladder, in order:** §6.1 scale fit → hole-#4 verdict → **§6.2 normalization**
  (`centered_additive_mean10`, the decisive axis — gets carries/targets/fantasy to 5/6) → **§6.6
  family / centered-param SkewNormal + regularization** (the binding g4/g6 wall once normalization
  is in) → §6.5 blend rebuild → **count-branch `log(volume)` NB/Tweedie offset** (the preserved
  efficiency × opportunity fork after `ratio_projvol` was refuted §6.2 — a family change at the
  deep end, attempt only if normalization/family/blend can't ship NFL yards) / monotone priors /
  TabPFN. **§6.3 raw-feature build is demoted** — the volume cells' 485-col set already carries
  game-script + interactions + recency (§10 audit); only *targeted* interactions (e.g.
  implied-total × usage) remain, not a broad build. Hardest cells (qb-yards, passing-first-downs
  multi-gate; receptions severe) sit at the ladder's deep end.
- For any cell the model can't improve: lean harder on the book in the blend — rides the sharp
  line, ships if calibration holds. No cell is shelved (§8); never loosen a gate.

#### MLB / NHL — foundation, then activation

Feature foundations are §6.7 and proceed now; training/shipping is gated on D1/D2
([`mlb-nhl-activation.md`](mlb-nhl-activation.md)). First targets when active: MLB hitter
volume markets (batting order), Ks later (umpire); NHL goalie SV + skater shots/points.

## 7. Working rules

### §7.1 The bar — six gates, lifecycle, supersession

Full thresholds, rationale, and the tighten/loosen procedure live in
[`../ship_gate.md`](../ship_gate.md) (canonical; owner-only). Convenience snapshot of the six
first-ship gates:

| # | Gate | Statistic | Threshold |
|---|---|---|---|
| 1 | Brier vs book (non-inferiority) | paired-bootstrap 95% CI of `(p_model−y)² − (p_book−y)²` | `ci_hi < 0.005` |
| 2 | Star σ-match | `\|mean(Blended_EV) − mean(Result)\| / std` on top-mean decile | `z < 0.5` |
| 3 | Bench σ-match | same on bottom-mean quartile | `z < 0.5` |
| 4 | **PIT-KS calibration** | `KS(randomized-PIT, Uniform)` of the predictive CDF | `pit_ks < max(0.05, 1.358/√n)` |
| 5 | Equal-mass debiased ECE | 10 equal-mass `p_model` bins | `ece < 0.075` |
| 6 | **Anti-shrinkage** (all cells; OR of 3 one-sided legs) | (a) recent-form `Σ Blended_EV / Σ Mean10` stable top-MeanYr, anchored on `corr(Mean10,Result)` w/ 0.58/0.52 hysteresis; (b) CITL-under `Σ Blended_EV / Σ Result` same stars, every cell; (c) over `Σ Blended_EV / Σ Result` bottom-MeanYr, count/ZINB, guarded `mean(Result)≥1` | recent `star_hi ≥ star_ref − 0.03` (bball 0.95/NFL 0.94); CITL `citl_hi ≥ 0.97`; over `over_lo ≤ 1.03` |

Gate 4 is the load-bearing one: `pit_ks = sup|F_model − F_true|` **is** the worst-case alt-line
mispricing, and the randomized PIT (Brockwell 2007) is exactly Uniform under calibration for
count *and* continuous families, so one threshold spans both. Report-only companions name the
*direction* a KS scalar can't: `central50_coverage` / `central80_coverage` (below nominal ⇒
predictive too narrow), `g4_tail_pit_ks` (alt-over wobble), `g1_has_edge`, `betting_active`, and
the retired IQR ratio survives as `g4_iqr_ratio`. Advisory diagnostics worth adding (report-only,
never a gate): Anderson-Darling PIT (tail-weighted), conditional/stratified PIT-KS (by
mean-decile / blowout / position / home-away), the non-randomized PIT for the count head, and a
CRPS reliability decomposition (Arnold et al. 2024).

**Lifecycle.** Six offline gates (Gate 1) certify a first ship → `shipped: "withheld" → "devel"`
(one-line `stat_meta.json` edit; production tracks devel) → 14-day live Gate-2 soak →
`check-graduation` classifies {not-shipped, in-test, graduated, demoted} → the monthly
`gate-status` cron promotes graduates to `main` via PR. Demotions flow back the same way.

**Supersession (Tier 1).** A *baselined* (already-shipped) cell never re-ships on a fresh 6/6 —
the candidate must beat the incumbent via `scorecard.supersede_verdict(baseline, candidate)`:
**S1** candidate clears the 6 gates standalone, **S2** paired Brier CI lower-bound > 0, **S3**
paired Kelly-Sharpe Memmel-z > min. All three → SUPERSEDE (swap the strategy in
`stat_meta.json`); any fail → HOLD. The S2/S3 asymmetry is deliberate — it stops strategy-churn
on noise. First-ships use Tier-0 absolute gates only. CLI:
`python -m sportstradamus.training.scorecard --baseline … --candidate …` (both sides full-HPO,
row-aligned test dumps).

### §7.2 Feature validation protocol (every §6.3/§6.4/§6.7 item)

1. **Leakage test first.** Extend
   [`test_meanyr_mean10_leakage.py`](../../tests/test_meanyr_mean10_leakage.py): temporal
   features assert strict `<` date visibility; as-of/external features assert the training value
   is reconstructable from data observable at `game_time − TRAINING_LOOKBACK`.
2. **Train/live parity test per batch** (the §6.0.4 harness). Paired surfaces change in one PR —
   the two fill sites, any new context column's historical + upcoming branches.
3. **Regen + deterministic A/B.** Two verified `pipeline.py` facts force the ordering:
   `--deterministic` **never rebuilds the matrix** (cache-only load; parquet write skipped), and
   a **plain `meditate` publishes production artifacts** (only `--deterministic` redirects to
   the sandbox). So a new feature is invisible until the cache parquet is rebuilt, and the
   rebuild must never go through a plain run mid-experiment. Recipe per cell: (i) preserve
   baselines *before* the code change — copy the cache parquet aside, run a baseline
   deterministic train; (ii) land the feature edit, training + live mirror together; (iii)
   rebuild the cache **directly** — delete the parquet, call `get_training_matrix` +
   `trim_matrix` and write it yourself (worked example below); (iv) run the candidate
   deterministic train with flags identical to the baseline; (v) compare the two sandbox CSVs
   with the scorecard CLI (writes a sandbox scorecard CSV, never `model_stats.parquet`).
4. **Inert-revert rule:** SHAP importance < 0.001 ⇒ inert ⇒ revert. Never carry dead columns.
5. **Ship path.** Deterministic A/B improvement ⇒ full-HPO `meditate` ⇒ official 6-gate
   scorecard; incumbents additionally need `supersede_verdict()`. Never ship on in-sample
   screens.
6. **Batching.** One regen per league per batch, never one feature at a time (regen costs hours
   per league board). Quality gates before any push.
7. **Cache-append inertness (highest structural risk).** New columns are NaN over cached rows →
   pruned → silently inert. Control: mandatory per-cell parquet delete + full regen per batch;
   golden test that a sentinel new column survives `_prune_uninformative_features` after regen.
   Verify regen feasibility per league first (the 850-day window must be resolvable from the
   archive — §3 archive block) before deleting any cache.

**Worked example — QW-1 on NFL `carries`, end-to-end.** Read the cell's `target_normalization`
from `stat_meta.json` (`carries` → `ratio_meanyr`) and pass it explicitly so baseline and
candidate match production config — identical flags on both runs; withheld cells need
`--bypass-withholding` or they are silently skipped:

```bash
# 0. baseline insurance + baseline deterministic run (BEFORE any code change)
cp src/sportstradamus/data/training_data/NFL_carries.parquet /tmp/NFL_carries.cache.bak
poetry run meditate --league NFL --market carries --deterministic \
    --bypass-withholding --target-normalization ratio_meanyr
cp src/sportstradamus/data/test_sets/deterministic/ratio_meanyr/NFL_carries.csv \
    /tmp/NFL_carries.baseline.csv

# 1. implement the feature in BOTH paths (historical + upcoming branches),
#    extend the leakage test, run quality gates.

# 2. rebuild the cache with the new columns (direct — NOT via plain meditate)
rm src/sportstradamus/data/training_data/NFL_carries.parquet
poetry run python - <<'EOF'
from sportstradamus.stats import StatsNFL
from sportstradamus.training.data import trim_matrix
s = StatsNFL(); s.load(); s.update()   # update(): gamelog must be current
M = trim_matrix(s.get_training_matrix("carries"), 15000)
M.to_parquet("src/sportstradamus/data/training_data/NFL_carries.parquet",
             compression="zstd", index=True)
EOF

# 3. candidate deterministic run (identical flags) + scorecard comparison
poetry run meditate --league NFL --market carries --deterministic \
    --bypass-withholding --target-normalization ratio_meanyr
poetry run python -m sportstradamus.training.scorecard \
    --baseline /tmp/NFL_carries.baseline.csv \
    --candidate src/sportstradamus/data/test_sets/deterministic/ratio_meanyr/NFL_carries.csv

# 4. decide: gate-row improvement => real-HPO confirm (§7.2 step 5);
#    SHAP < 0.001 or regression => restore /tmp/NFL_carries.cache.bak, revert edit.
```

Executor notes: the deterministic CSV subdir is the normalization slug
(`{target_normalization}{_hurdle?}`); the parquet snippet mirrors what
`_step_persist_matrix_and_comps` writes (same `trim_matrix(…, 15000)`, same compression);
`update()` requires league-API access — run after the daily jobs or accept a slightly stale
gamelog on both sides (fine: the A/B only needs both sides identical).

### §7.3 Inference-path compatibility checklist (per change type)

Every change lands its inference-side mirror **in the same PR, before promotion**. Gate 1 lets a
change into the test window; this checklist makes the window safe.

| Change type | Inference-side work | Precedent |
|---|---|---|
| Training-only (loss change, monotone constraint, per-parameter Optuna, reweighting) | None — output schema unchanged | Stage B1 ZTNB attempt |
| New target / baseline strategy | Inverse decode in `model_prob._decode_skewnormal` via `baselines.STRATEGY_REGISTRY[strategy].decode_loc/decode_scale`; matching `*_Ratio` feature in `get_stats`; `target_strategy` + `offset_meta` keys round-trip | P1 `centered_additive_*` |
| New distribution head (SHASH, CMP, MZINB, PGBM, MEGB, gbex) | (a) decode block in `model_prob.py`; (b) `get_ev` / `get_odds` / `fused_loc` / `set_model_start_values` accept the new `dist`; (c) `dist` in `_build_filedict` + legacy fallback; (d) live-path test mirroring [`test_zinb_hurdle_live_path.py`](../../tests/integration/test_zinb_hurdle_live_path.py) | P2.B HurdleZINB |
| Post-hoc calibration object (Rung-C map, CQR, `bias_correction`) | Pickle as a new key; load in `model_prob`, apply after decode (before/after `fused_loc` per what's calibrated); byte-identical round-trip test | `temperature` field |
| New player-level feature | Column in BOTH `get_training_matrix` and `get_stats`, computed identically, leakage-safe; same dtype/index; `feature_filter.json` registration | `MeanYr` / `Mean10` / `*_Ratio` |
| Multi-head factorization | N pickles/market, Monte-Carlo combine, multi-output blend — **largest inference-side change in plan**; T5 is killed, so only relevant if a future factorization survives a brief | none in-repo |
| Different model class (CatBoost, MEGB, GPBoost, TabPFN) | New `is_*` flag; `model_prob` + `prediction/__init__.py` load-path branch; determinism gate extended; adapt if no LSS `predict(pred_type="parameters")` API | P2.B `is_hurdle` |

**Hard ship gate:** any change requiring inference-side work must have a passing live-path
integration test under `tests/integration/` before promotion.

**Pickle-schema discipline:** every new field needs (1) a reader site in `model_prob.py`, (2) a
legacy default fallback (`filedict.get("new_key", default)`), (3) a round-trip test asserting
byte-identical predictions. Fields written by `_build_filedict` as of the consolidation:
`model`, `step`, `stats`, `metrics`, `diagnostics`, `params`, `distribution`, `cv`, `std`,
`temperature`, `dispersion_cal`, `weight`, `r_book`, `hist_gate`, `shape_ceiling`, `normalized`,
`offset_meta`, `target_strategy`, `zinb_mode`, `is_hurdle`, `expected_columns` — verify against
the code before relying on the list.

### §7.4 Cross-league caveats (read before any cross-league A/B)

1. NFL sample sizes are an order of magnitude smaller than NBA (~17 vs ~82
   games/player/season); re-derive EB shrinkage `K = σ²_within / σ²_between` per league.
2. NFL stats are position-locked (`Player position` categorical; per-position scoping shipped) —
   cross-player models per market may not transfer.
3. WNBA shares NBA's structure at half the games/season; verify EB K. WNBA has no `FGM` or
   `FG3A` markets.
4. The scorecard harness is league-agnostic; file paths are league-specific
   (`data/training_data/{LEAGUE}_{market}.parquet`,
   `data/test_sets/deterministic/{strategy}/{LEAGUE}_{market}.csv`).
5. The determinism gate covers NBA only (NBA_FGA + NBA_FG3M) — add parallel WNBA + NFL
   assertions before any cross-league change, else the verdict is noise.
6. Low-mean NFL markets (interceptions ~0.5): asymptotic Vuong degrades badly; trust only
   Wilson-Einbeck's non-asymptotic zero-inflation test.
7. Track-parallelism across leagues is fine; the shared resource is the read-only scorecard
   harness.
8. Low-mean conditional-dispersion diagnostics need the Dunn–Smyth RQR, not Pearson, at mean
   ≲ 0.11 (they diverge badly there); bootstrap the RQR variance.
9. Post-hoc bias correctors at NFL count means must be affine ROE, not isotonic/per-decile
   (low-base-rate overfit, ref [48]); reserve isotonic for higher-mean NBA/WNBA count cells.

### §7.5 Ship & session mechanics

- **Ship to devel = one-line `stat_meta.json` edit** (`shipped: "withheld" → "devel"`) the human
  commits; `generate-ship-config --branch devel` validates + summarizes. Promotion to `main` is
  the monthly `gate-status` cron's PR. Never train or ship on `main`; production tracks `devel`.
- **Never push `devel` directly** — the `devel-ship-curator` agent carves every devel-bound ship
  PR and enforces the dev-only denylist (`zinb-routing-diagnostics`, `icc-diagnostics`,
  statsmodels, /tmp harnesses never ship). Confirm its denylist covers any new research
  scaffolding.
- **`refactoring-specialist`** runs on every touched Python file before any push / PR / review
  (CLAUDE.md hard rule, five triggers).

## 8. Escalation & stop conditions

### §8.1 Failure protocol & matrix exhaustion

- **Per-lever:** every §6 stage carries a go/no-go and an if-it-fails branch. When a lever's
  go/no-go fails on a cell, record it (lever-attempt +1, with the axis named) and take the
  branch. Do not grind a dead lever.
- **Per-cell:** push every cell until it ships or has actually failed across the **whole
  matrix** — all four axes (normalization × model/loss × blend × calibration) **plus** the
  family and hierarchical tracks. Four failed calibration/feature levers is *not* an exit: the
  cell moves to the next axis with a one-line note naming the axes tried. The only ways off the
  board are genuine matrix-wide exhaustion (true of **zero** cells today) or the operator's
  explicit, documented denominator call. The heaviest tail-head rebuilds (spliced/Pareto,
  MZINB) are deferred long-shots (roadmap v3 §8), tried only after the cheaper §6.6 moves.
- **Operation-level: failure is not an option.** §6.9 shows NBA and WNBA clear on §6.0+§6.1
  alone, with backups; NFL's escalation ladder is deep enough to reach 15, and the terminal
  fallback (ride the book line, calibrated) ships.
- **Never loosen a gate to hit breadth.** Gate constants are effect-size floors (vig-scale),
  not breadth knobs. Standing rule: any search over bet-definition knobs must be
  multiplicity-corrected before it informs a ship, and never gate on a model-conditioned
  statistic.
- **Grind detector:** two consecutive sessions with no acceptance criterion moving is a hard
  stop — escalate to the owner; a cell that resists an axis moves to the next axis, never gets
  re-ground.
- **Park & pivot:** if blocked (e.g., NFL Gate-2 soak needs live games that won't exist until
  September), append a §10 ledger line with the reason, set `BLOCKED (on: …)` in the status
  line, flip the roadmap v3 §4 row, and point the owner at the swimlane index.

### §8.2 Research holes & research-first triggers

**Dispatch `research-analyst` before** (CLAUDE.md research-first; the research-gate hook
enforces the file-level cases): any §6.6 family/distribution change; any §6.5 blend-structure
change or training-time dispersion regularizer; §6.6 hierarchical/TabPFN escalation; the
§6.6-flagged feature items C-3 / B-5 / L-3 (D-4 brief-note level); any hole below before
betting the plan on it. Plain knobs (normalization slug, loss choice, ordinary features) need
no brief. To proceed without a brief on a hook-gated edit, write a one-line justification to
`.claude/.state/research_waiver`.

**Open holes:**

- **#0b — Gate-4 baseline hysteresis** (highest priority; owner call). §6.0.2 carries the
  decision packet and the recommendation (ship the scale fit first; hysteresis only if churn
  persists). It gates trust in the ship-incrementally premise.
- **#4 — do the marginal NFL g1s improve when dispersion is calibrated?** Pre-registered as
  §6.0.5 with its decision rule. The difference between NFL reaching 10 and reaching 15.
- **#6 — the block-bootstrap / clustered-g1 backlog.** Test-set CSVs still lack `game_date` on
  combined-stat cells, blocking the player-clustered Gate-1 recheck and a closing-line-value
  gate. The concrete method is CPCV + a player/date embargo (López de Prado 2018) — a
  validation refinement, not a gate change; the one principled selection-style criterion worth
  building.
- **#7 — the MeanYr over-shrinkage root cause** (Gate 6 detects the symptom; this re-ships the
  pulled-back WNBA FGA/PR/PRA). The `ratio_meanyr` 365-day denominator conflates "high historical
  average" with "will regress", so the holdout target teaches a high-volume regression real games
  don't show (6-season causal: a *stable* star produces ~0.99× recent form, not the holdout's
  ~0.83×). The fix is a normalization change — a recency-weighted / decline-aware baseline
  denominator, or a posthoc `isotonic_mean` corrector lifting the stable-star prediction — which
  must clear all six gates, Gate 6 as the regression test. Research-gated (normalization axis).
  PR/PRA may re-ship faster than FGA: part of their flag is real even by the outcome gates
  (`pred/Result` 0.92 / 0.96 vs FGA's 1.00). [research: `/tmp/researcher_overshrinkage_gate.md`]
- **#8 — do the *shipping incumbents* fail a volume-stratified PIT-KS-max?** The `ratio_projvol`
  refutation (§6.2) turned on a per-volume-quartile PIT-KS-max the global Gate-4 does not compute.
  `ratio_meanyr` scales width by a per-player *constant* (no per-game volume heteroscedasticity) so
  should be far better stratified — but it is asserted, not measured (the `ratio_meanyr` test-set
  dumps lack the `Player proj … mean` column). If the gate ever adopts a stratified-max, re-dump the
  incumbents with the proj column and verify current ship states first.
  [research: `/tmp/researcher_projvol_dispersion.md`]
- **#9 — do the calibration-aware HP levers (§6.1 Lever 1 / Rung B′ / Lever 4) move the served
  set, or only re-rank trials that ship/fail together?** Built + opt-in; unanswered until run on
  real cells. (a) Lever 1 is now the **search-gate** (the objective steers the TPE study toward the
  calibrated region, not just a top-K re-rank); the open question is the **scale-bound vs shape-bound
  mix** across the live g4-failing SN cohort — the cheap cov50/cov80 gate-row query decides
  search-gate-vs-§6.6 ROI before any retrain. Confirm on the scale-bound near-miss WNBA DREB (g4
  0.0651 → 0.0508 under the old re-rank; the search-gate should close the last 0.001); shape-bound
  cells (cov ≈ nominal, KS wandering — PA) are a search-gate no-op and route to §6.6. (b) Rung B′'s PIT-KS count objective
  has no sharpness brake — watch g6's over-leg and the g1 acceptance for over-tightening. (c) only
  promote `--stabilization` to a swept axis if MAD/L2 ever wins on ≥1 cell (YAGNI). (d) the
  under-wide-AND-mislocated NFL SkewNormal cells (attempts/carries, central-50 ≈ 0.27/0.34) are out
  of reach for any selection/post-hoc width fix — that defect is signal/family, not width; confirm
  via §6.3 features and/or the §6.9 `log(volume)` offset before spending a calibration lever on
  them. [research: `/tmp/researcher_calibration_hp.md`, `/tmp/researcher_hpo_objective.md`]

**Resolved (one-liners; detail in [`../operation_ship_references.md`](../operation_ship_references.md)):**
post-hoc scale moves PIT-KS without breaking g1/g5, but only under the right normalization and
with a cell-specific val→test discount; the SkewNormal dispersion-cal exclusion was an a-priori
skip, not a tested negative; the count branch needed the PIT-KS re-target (coverage was the
wrong objective); the severe-coverage cells were a decode artifact + plain under-dispersion, not
a new-tail-head problem — they ship under the centered normalization; the in-sample 2-parameter
screen oversells the honest val→test gate (nothing ships on an in-sample floor); normalization
is a first-class ship axis, often decisive; the variance regularizer is untried, not refuted; the
ZINB hurdle-hyperinflation π-cap was considered and parked — `decode_predictive_mean` returns the
base NB mean `r·p/(1−p)` with the gate `π` separate, so the model-EV winsorize already bounds the
decoded mean and a high gate cannot hide an inflated μ; the failure mode is not observed live (owner call).

**STOP and ask the owner when:**

- gate-constant or test-tolerance changes (always);
- smoke regression;
- gates red at session start through no fault of yours;
- anything touching cron, credentials, or paid APIs;
- the §8.1 grind detector fires.

**DISPATCH a subagent when:**

- `research-analyst` (Opus-backed) — for the §8.2 triggers above;
- `devel-ship-curator` — every devel-bound ship PR;
- `refactoring-specialist` — per the five CLAUDE.md triggers.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session (CLAUDE.md five-trigger rule).
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean (incl. scorecard / gate tests).
- `poetry run pytest -m integration -n0` clean, then `touch .claude/.state/integration_green`.
  The determinism gate
  ([`test_determinism_gate.py`](../../tests/integration/test_determinism_gate.py)) must stay
  green; extend it to WNBA + NFL before any cross-league lever ship (§7.4.5). Cross-league
  testing policy: smoke (1–2 markets/league) before full verification; a smoke regression is a
  hard stop.
- One ledger line appended to §10; status line updated if a stage boundary was crossed.
- Never push `devel` directly — devel-ship-curator carves ship PRs.
- Durable non-obvious lesson? Offer a memory capture (CLAUDE.md §Agentic workflow conventions).

## 10. Ledger (append-only, newest first, cap ~15 — older lines live in git)

- 2026-06-27 · **WNBA 12→13/18: FTM ships on `--zinb-mode hurdle`+`pit_ks` (6/6); the prior entry's "blocked" hurdle screen was a read-the-wrong-file bug, NOT a harness quirk.** Root-cause: `meditate --deterministic` **does** train but never writes `model_stats.parquet` (owned by `report()`, skipped under deterministic) — it dumps holdout CSVs to `data/test_sets/deterministic/{norm}[_hurdle]/`; the screen read `model_stats` so joint/hurdle came back byte-identical (looked like no train). Re-ran the A/B via the scorecard on the dumped CSVs: hurdle clears the binding **non-g4** gates (g1, g5) on FTM+TOV, hurts BLST (adds g2). Real-HPO confirm (HurdleZINB trains ~6 s, bypasses the Optuna sweep): **FTM 6/6** (g1 ci_hi −0.0059, g4 pass, g5 0.027) → **ships**; **TOV 5/6** — only g4 (iqr 0.875, mild under-dispersion), Rung C `cdf_recal_isotonic` a no-op on g4 (0.875→0.875, only nudged g5) → genuine count under-dispersion → §6.6 → **book-lean**; BLST excluded. **Code:** per-cell `zinb_mode` persistence (`cli._resolve_cell_knob` + FTM/TOV stat_meta `zinb_mode:hurdle`/`count_dispersion:pit_ks` + extended golden pin; `--zinb-mode` default `joint`→`auto`, snapshot regen) so the hurdle ship reproduces on the warm cron instead of darking to joint. Gates: refactoring-specialist clean; ruff clean, golden 1937 pass (same lone pre-existing `test_find_correlation_offer_correlations_real_nba` red, untouched by this change), integration green. FTM flip + cli.py carved as **one commit onto the un-pushed `ship/wnba-5cells-runtime`** → one PR, WNBA 7→13. Hurdle is the validated P2.B mechanism [[p2_hurdle_zinb_verdict]] (no new research). **Built-lever lane now closed for WNBA at 13/18** (72 %); the 14th (75 %) needs §6.6 (DREB/PA shape, BLST/TOV count family) or §6.3 (STL) — deferred.
- 2026-06-27 · **WNBA Ship-75: 7→12/18 on built non-family levers; the un-pulled-lever brainstorm paid where the doc predicted, the rest confirmed §6.6-bound.** Walked the 11 withheld cells cheapest-lever-first (capped at built levers + book-lean, **no §6.6 family work** per scope). **+5 ships:** PR/RA/AST (g6 over-shrink → `centered_additive_mean10` Mean10-anchor antidote; AST stacks calibrated HP-sel + Rung C), OREB (ZINB over-coverage → `--count-dispersion-objective pit_ks` Rung-B′ retarget), FGA (g6 stars 0.898→pass via `posthoc isotonic_mean`, §8.2 hole #7 confirmed; real-HPO **6/6**, g4 0.047, bss +0.145). **Brainstorm (no new families) → 3 un-pulled levers, Tier-1 confirm 1/4:** FGA isotonic_mean **ships**; **Rung-C-on-count REFUTED** — BLST broke g1 (whole-CDF reshape worsens the book-blend non-inferiority), FTM g4 0.0511→0.0523 (wrong way); DREB loss+RungC fixed g4 (0.035) but g5 0.081 stays (calibrated+RungC's 0.0776 was closer) → **shape-bound**. **Built non-family levers now EXHAUSTED on the last 6:** BLST/FTM at the **ZINB variance≥mean floor** (g4 0.056/0.051, can't narrow) → §6.6 count family; DREB/PA **shape-bound** (g4-fix breaks g5) → §6.6 centered-param SN; TOV g4+g5 → §6.6; STL g1 signal-starved → §6.3 features (caps WNBA at 13 — 14 needs §6.6 regardless; deferred). **Code:** `count_dispersion_objective` now persists per-cell (`cli._resolve_cell_knob` + OREB stat_meta field + extended golden pin) so the pit_ks ship reproduces on the warm devel cron instead of darking out; `--count-dispersion-objective` default `crps`→`auto` (mirrors hpo_selection/blending), CLI-help snapshot regen. Gates: refactoring-specialist clean; ruff clean, golden 1937 pass (lone red = pre-existing `test_find_correlation_offer_correlations_real_nba` NBA-correlation fixture, untouched by this WNBA/cli change), integration 24 pass. 5 stat_meta flips + cli.py carved to devel via curator. · next: TWO un-pulled NON-family levers to try BEFORE §6.6 (post-carve exploration, NOT ruled out): **(a) posthoc stage-pipeline** — chain CDF_STAGE→PROB_STAGE (small build; `apply_posthoc` takes a single slug today) to rescue the *two-gate-split* cells where one stage fixes g4 **or** g5 but never both (DREB proves it: cdf_recal → g4 0.035 pass / g5 0.081 fail; also TOV g4+g5, NFL carries/fantasy g4+g6) — stays in the calibration ladder, cheaper + lower-risk than a family rebuild; **(b) `--zinb-mode hurdle`** for BLST/FTM/TOV — a built toggle (P2.B, NBA-screened) never tried on WNBA, but run-wide so it needs the same per-cell persistence just added for `count_dispersion_objective`, and its fast deterministic screen was *blocked* (the `--deterministic` WNBA run exits after the data refresh without entering the train loop — harness quirk; use a real-HPO run). THEN the §6.6 families session (centered-param SkewNormal for DREB/PA, COM-Poisson/count family for BLST/FTM/TOV) + §6.3 STL features — the built-lever lane is closed for WNBA at 12/18.
- 2026-06-26 · **§6.5 Pooling-half NO-GO HARDENED by Probe-v2 — the operator's weight challenge tested and also fails.** Operator pushed back that v1 scored at a fixed `w≈0.90` carrying the current dispersion-cal, under-testing a *jointly-refit sharper blend* on the book-heavy cells (where v1's reconstruction dropped rows). Probe-v2 recovers the **raw** dispersion-freed model leg (undo `dispersion_cal` BEFORE inverting the blend — re-apply+re-blend sanity **5e-16**; `frac_ok`→~1.0, so the book-heavy NBA AST 0.60→0.99 / NFL carries 0.87→1.00 are now FULLY reconstructed — the drop concern fixed) and re-fits weight AND dispersion jointly by **CRPS+hinge** (the correct retrain objective; the LightGBM model is blend-independent so the raw leg is what a retrain blends). The joint fit DID explore the low-weight regime the operator wanted (`λ_A`→0.63, `w_mix`→0.40 on the book-heavy cells, B discarding the model) and **every one still loses OOS** (dA, dB < 0); freed dispersion shaves KS only on NFL fantasy-underdog (dA +0.023) but A/B still pass **0/6** (shape/family-bound → §6.6/§6.1); v1's lone fragile positive (NFL targets) **reverses sign** (+0.013→−0.017) under CRPS — [[deterministic_ab_g4_oversell]] made explicit; on model-dominated cells the joint fit is actively harmful (NBA PTS dA −0.036, WNBA REB −0.031). Incidental: `c>1` on 9/12 cells (the served predictive RELIES on dispcal-widening — raw model under-dispersed, consistent with the under-dispersion-is-the-disease theme). **0/9 cells produce a robust OOS g4 win — NO-GO confirmed and firmer**, closing the "you didn't test a jointly-refit sharper blend" objection in the record. No production code (probe-only; `scratchpad/blp_v2_*.py`). · next: §6.6 centered-param / count family for the over-wide cohort — the pool is closed, jointly-refit or not.
- 2026-06-26 · **§6.5 Pooling-half blend-structure rebuild PROBED across both candidate structures → NO-GO on both; routes to §6.6/§6.1.** Operator's design call (a scalar `w` couples location-trust and shape-trust; a CDF-mixture widens, fighting the book's sharp line) reframed the lever as a 2+-param structure decoupling location from shape, then chose "probe both, data decides." Cheap two-structure honest-OOS probe (brief `/tmp/researcher_blp_pooling.md`; `scratchpad/blp_*.py`): reconstruct `F_model` by inverting the `fused_loc` precision blend at the pickle's `w`, `F_book` from `Odds` — **re-blend reproduces the served CDF to 3e-16** (machine epsilon, the WS2 1e-14 bar) on all 11 reconstructable cells; score **(A) decoupled location/shape** (free location-trust `λ`, dispersion from the model) and **(B) beta-transformed CDF-mixture BLP** `B_{α,β}(w·F_model+(1−w)·F_book)` vs the served incumbent over a 12-cell SkewNormal cohort, 8 resampled 50/50 within-test folds. **(A) ill-posed:** a PIT-KS-fit `λ` overfits the fit-half mean (loc shift 2–7 units, ΔCRPS −0.5 to −3.7, g1 −0.01 to −0.11); fit to CRPS instead (the correct EMOS objective, Gneiting 2005 [74]) PIT-KS explodes (NBA AST 0.047→0.47) — the two objectives share no feasible region (Gneiting 2007 [75] calibration-vs-sharpness). **(B) wash:** cohort-median OOS ΔKS ≈ 0; its best single-fold wins (NFL carries +0.018, WNBA REB fail→pass) **flip sign / evaporate** across folds (carries → −0.017 mean, B 6/8 vs serv 8/8 — would REGRESS it; WNBA REB rescue not reproduced) — the [[deterministic_ab_g4_oversell]] signature a third time; the lone genuinely g4-failing cell (NFL fantasy-underdog, g4 0.082) rescued by **neither** (B 0/8). Root cause = the book audit's: at the CRPS-fit `w≈0.90` the served predictive is already the model's and near-calibrated, so the beta wrapper has no decalibration to repair (Ranjan-Gneiting 2010 Thm 1 premise unmet); the over-wide cells are over-wide because their *family/shape* is wrong — a pool of a wrong-shape model and a symmetric book cannot manufacture the right shape. **Routes the cohort to §6.6 (centered-param / count family) + §6.1 Rung C, not the pool.** Pre-committed design recorded (§6.5/§3.4) should a future genuinely-over-wide cohort at `w<0.9` reopen it (fit by CRPS+PIT-hinge never PIT-KS; bounds `α,β∈[0.5,3]`; ship guard = raw pool `α=β=1` in CI; XOR with Rung C). **Caveat surfaced — corrects the book-audit entry below:** that audit's `book_calib.py` used the dumped `EV` column as the book mean, but `EV` is the *model* base mean ([pipeline.py:1497](../../src/sportstradamus/training/pipeline.py#L1497)) — the true `ev_b` (recovered from `Odds`) differs by ≈1.3–2.7 pts/cell; the audit's NO-GO verdict is unaffected (a more-mislocated book is even less the lever) but its book-only PIT-KS / coverage magnitudes are directional, not exact. No production code changed (probe-only; .md doc + refs [74][75]); blend-structure rebuild recorded tried-refuted (§3.4/§8). · next: §6.6 centered-parametrization SkewNormal for the over-wide cohort (NFL fantasy-underdog/carries g4, WNBA DREB) — the binding under-dispersion wall, research-gated; the Pooling half is closed for this cohort.
- 2026-06-26 · **§6.5 Step 1 book-distribution audit RAN → NO-GO on a standalone book rebuild; redirects to the Pooling-half BLP.** Operator re-sequenced §6.5: audit how one de-vigged point becomes a full book distribution *before* pooling ("don't blend into garbage"). Built the missing book-only-CDF diagnostic (`scratchpad/book_calib.py`, exact scorecard randomized-PIT machinery, all 45 cells); brief `/tmp/researcher_book_distribution.md`. **Verdict = qualified exoneration.** Book IS genuinely mis-shaped — **30/45 fail a book-only PIT-KS**, SkewNormal book too narrow (cov50/80 ~0.51/0.79), count book too wide (~0.81/0.96), the SAME two directions as the served heads — so the suspicion is confirmed at the standalone-CDF level. **But it does not propagate to the served gate:** where the book fails g4 the served blend already passes **15/18** (book-fail∧served-fail=3, book-pass∧served-fail=0), because `model_weight` pins to its 0.90 cap on 17/22 cells (the CRPS `w`-fit rides the model); Spearman(book PIT-KS, served g4)=0.35 SN (n.s.)/0.09 count. The lone book-dominated cell (NFL attempts w=0.05) has served g4 **0.214 ≫ book 0.088** — fixing the book is not the lever even there. **The within-constraint fix is refuted:** a context-conditioned `cv=a+b·(STDYr/MeanYr)` (book-observable only) gives **0% out-of-sample book-PIT-KS gain on 45/45** (optimizer drives b→0) — same failure class as `ratio_projvol` and WS2. Theory agrees: a single point is sufficient for *location* but not *shape* (Dmochowski 2023 [72]); a mis-shaped pool input is the **pool's** problem, the beta wrapper recalibrates it (Ranjan-Gneiting 2010 Thm 1). So the mis-shape is the **Pooling-half BLP's** job (co-fit with `w`), the true shape source is the WS1 ladder (a year+ out) — §6.5 sequencing vindicated, not overturned. SkewNormal post-constant-cv residual is *shape* not scale (→ §6.1 Rung C); count over-coverage is the NegBin variance-≥-mean floor (→ §6.6 count family, fixes the book leg free). Exoneration **conditional on w≈0.90** — a Pooling-half `w`-refit must re-run this audit. No production code changed (audit-only; .md doc + refs [72][73]). Recorded: context-book-cv scale-law tried-refuted (§3.4/§8); `book_pit_ks` report-only diagnostic carved (unbuilt, optional). · next: operator call — build the report-only `book_pit_ks` companion (low-risk, ~1 file), or pivot to the §6.5 Pooling-half BLP (the vindicated move).
- 2026-06-25 · **NFL §6.1→§6.2 rung RAN + feature audit reframes the volume-cell lever — hole #4 NEGATIVE, `centered_additive_mean10` decisive but real-HPO 5/6, 0 ships.** **hole #4 (§6.0.5) = NEGATIVE:** dispersion calibration does NOT pull the volume cells' g1 `ci_hi` under 0.005 — `carries` full ladder (Lever 1 calibrated + Rung C, both confirmed applied: `pit_recal_blob lam:1.0` in pickle) real-HPO g1 0.0087; deterministic screen 0/5; bss signal-starved (attempts −0.11, completions −0.04, passing −0.12). **§6.1 calibration ladder EXHAUSTED for the g1-passing cells** (`targets`, `fantasy-points-underdog` on `ratio_meanyr`): Lever 1 + Rung C real-HPO = 0/3, g4 floors 0.07–0.13 (val→test discount — Rung C reshapes the val PIT, the test KS keeps the gap). **§6.2 `centered_additive_mean10` = the decisive axis:** deterministic `carries` 6/6 (g1 flips to beat book), `targets`/`fantasy` 5/6; real-HPO **cold-calibrated confirm = 0/3 ship but all 5/6** — `carries` **beats the book at real-HPO** (bss +0.036, g1 ci_hi 0.0039 < 0.005 — was g1+g4+g6-fail under ratio), `targets` **passes g4** (the gate the ratio ladder never cleared); blocking gates split clean — carries/fantasy **g4** (PIT-KS ~0.08 floor, calibrated fallback logged "best 0.0812"), targets **g6** (over-shrinks stable stars even centered). Kept `centered` as the withheld config (strictly better, sets up the last-gate push); **0 devel flips.** **FEATURE AUDIT (owner ask — "better features from existing data, not more data"):** the NFL volume cells are **NOT feature-starved** — the 485-col candidate set already carries full game-script (`Total`/`OppTotal`/`GameTotal`/`Spread`/`Moneyline`, all trained-`Common`), pre-built interactions (`PlayerExp_x_DefAvg`, `PlayerZ_x_DefPos`, `Player`/`Defense moneyline gain`/`totals gain`), multi-window recency (`Avg1/3/5/10/Yr` + `short`+`growth`+`_asof` point-in-time + EB + vs-opp); `Line`/`Odds` are correctly **excluded** from the trained X (training on the line just echoes the book — `get_stat_columns` omits them, they live in the matrix only for the gate). So negative bss is **target-shape + capacity, not missing features** → **corrected the §6.9 routing: volume five → §6.2 normalization + §6.6 family/regularization, NOT §6.3 raw-feature build.** The `carries` g1 flip on a pure target transform (zero new features) is the proof. **BUG (latent, production-relevant):** warm-start `study.enqueue_trial(seed_params)` (`hyperparams.py:155`) crashes `ValueError: 0.0 invalid for log=True` when a pickle stored `lambda_l1/l2 = 0.0` (LightGBM default, below the 1e-6 search floor, both `log=True`) — killed both --force confirm runs on `carries`; deterministic unaffected (fixed HP, no enqueue). Workaround = delete the pickle → cold start (`initial_params=None`); proper fix = clamp/drop sub-floor log-params from the seed before enqueue. Budget gotcha logged: cold = 3600s/300-trial (`pipeline.py:951`), warm = 300s/150-trial (line 964) — splits on cold-vs-warm, NOT calibrated-vs-loss. Only `stat_meta.json` touched (3 norm flips, no .py ⇒ no refactoring-specialist). Gates: ruff clean (no .py), golden 1936 pass (the 1 red = pre-existing `test_find_correlation_offer_correlations_real_nba`, an NBA-correlation fixture untouched by this NFL-only change), integration 24 pass. · next: §6.6 family / centered-param SkewNormal for carries/fantasy **g4** + targets **g6** (research-gated — the calibration axis is closed for these); the volume-five work is target-shape/regularization per the audit, not new columns; fix the warm-start seed clamp; production-faithful re-confirm under warm-300s once the seed clamp lands.
- 2026-06-23 · **Lever 1 reworked — HP *search-gate* replaces the top-8 re-rank (research-gated build).** Brief `/tmp/researcher_hpo_objective.md`: keep CV-CRPS as the base selection score (do NOT switch to NLL — it loses g1–g4 + EV in-house and is tail-fragile); the change is the constraint *form*. The old Lever 1 re-ranked only the top-8 lowest-CRPS trials — structurally blind to the wider-σ corner CRPS never samples (PA's whole top-8 was under-dispersed). Now the Optuna objective is **search-gated**: `CRPS + 10·max(0, val_PIT-KS − τ)`, a one-sided hinge (zero once calibrated ⇒ the feasible region stays ranked by pure CRPS, the GBR degeneracy guard) so the TPE sampler *explores* the calibrated region; final pick unchanged (`_pick_calibrated_candidate`). **Inert by default** — no-op unless `hpo_selection: calibrated` + SkewNormal; `calibration_penalty is None` reproduces plain CV-CRPS byte-for-byte. `hyperparams.run_hyper_opt(calibration_penalty=…)` + `_penalized_objective` + `_collect_calibrated_candidates`; `pipeline._calibration_penalty` closure replaces the removed `_select_calibrated_hp`/`_calibrated_top_k`; cost = one extra refit/trial (`return_cvbooster` OOF variant deferred). **Brief reframe:** PA was the wrong poster child — it is **shape-bound** (cov50/cov80 ≈ nominal, PIT supremum wanders → family-misspecified), a search-gate no-op routing to §6.6 **manual per-cell** family; **DREB** is the **scale-bound** beneficiary (0.0651 → 0.0508, the gate should now close it). TDD (`_penalized_objective` golden), gates: ruff clean, golden 1908 pass (1 pre-existing correlation red), integration 22 pass, refactoring-specialist clean. · next: cov50/cov80-stratify the live g4-failing SN cohort (scale-vs-shape mix = search-gate-vs-§6.6 ROI, open-Q #9); real-HPO confirm DREB under the search-gate; re-confirm PR (it shipped under the old re-rank, PR #84).
- 2026-06-23 · **PA/PR/DREB calibrated re-confirm — Lever 1 VALIDATED, ships WNBA PR; per-cell `hpo_selection` persistence + §6.2 confirm policy.** Ran the 3 in-regime g4-bound WNBA SkewNormal cells (PA/PR net-new, DREB the knife-edge) as a controlled **loss-vs-calibrated A/B** (same `ca_mean10` norm, full cold HPO, differing only in `--hpo-selection`). **PR RESCUED** — loss g4 **0.0516 (fail) → calibrated 0.0423 (pass)**, ships 6/6: calibrated selection traded a hair of sharpness for the wider-σ trial that clears Gate-4, exactly the Lever-1 mechanism. **DREB near-miss** (0.0651 → 0.0508; slack −0.302 → −0.016 — a Rung-C candidate now), **PA no-op** (0.0581 → 0.0583 — the logged fallback fired, no calibratable trial in the top-K). All three fail **g4-only** under loss (gates `111011`), the exact regime Lever 1 targets. **Durability gap found + fixed:** `--hpo-selection` was a *run-wide* flag, not per-cell, so PR's calibrated ship would **dark out on the next devel cron** (warm HPO walks back to the sharper g4-failing trial → `_enforce_ship_gate` prunes). Made it per-cell like `blending`: new `cli._resolve_cell_knob` consolidates the blending + hpo_selection reads, `--hpo-selection` default `loss`→`auto` (honors stat_meta, explicit slug still forces a run-wide A/B), and WNBA PR carries `hpo_selection: "calibrated"`. **§6.2 operating loop updated** — HP-selection is a confirm-time axis the deterministic board can't see (it fixes one HP set, no trial spread to re-rank); **confirm every SN candidate under `calibrated` first** (weakly dominates loss on g4), fall back to `loss` if its wider σ tips g1, persist the winner per-cell. **Ships WNBA PR + PRA → devel** (PRA durable under the default loss from the prior confirm; +2 WNBA served). Gates: ruff clean, golden green (+ `_resolve_cell_knob` persistence test), integration green, refactoring-specialist clean. · next: §6.1 Rung C (whole-CDF PIT recal) for DREB's 0.0508 near-miss + the mixed-direction SN cohort; persist `stabilization`/`count_dispersion_objective` per-cell only if a cell ever ships on Lever 2/4 (YAGNI until then).
- 2026-06-22 · **`centered_additive_mean10`-over-`ratio_meanyr` confirm queue RAN (full HPO) + three calibration-aware HP levers BUILT.** Confirm queue (6 cells, the `ca_mean10`-board-best-differs set): 3 supersede (NBA RA/PR, WNBA fantasy) + 3 net-new (WNBA PA/PR/PRA). **Only WNBA PRA SHIPS** (net-new, real `min_gate_slack` +0.068 — the tiny det board margin +0.036 survived full HPO), the other 5 HOLD: the deterministic board oversold g4 again ([[deterministic_ab_g4_oversell]]) — WNBA PA/PR det g4 margins +0.33/+0.32 **evaporated** at full HPO; supersede trio failed S2/S3. WNBA 7→8. **The HPO-vs-deterministic g4 paradox** (sharp μ surface ≠ calibrated σ surface; μ/σ share one LightGBMLSS HP set so standard knobs move both together) was researched (`/tmp/researcher_calibration_hp.md`) → **3 cheap opt-in levers built, defaults = production:** **Lever 1** `--hpo-selection calibrated` ([`hyperparams.run_hyper_opt(top_k)`](../../src/sportstradamus/training/hyperparams.py) returns top-K lowest-CRPS trials → [`pipeline._select_calibrated_hp`](../../src/sportstradamus/training/pipeline.py) refits each + re-ranks by validation PIT-KS, picks sharpest that clears g4, logged fallback; SkewNormal only); **Lever 2** `--count-dispersion-objective pit_ks` (count `dispersion_cal` CRPS→PIT-KS via `_dispersion_pit_ks_loss`; §6.1 Rung B′); **Lever 4** `--stabilization MAD|L2` (σ-head gradient damping; the only in-API per-param knob). Levers 3 (whole-CDF Rung C) + 5 (in-training PIT reg) folded to §6.1/§6.5 as next/deferred; the g4-failing SN cells are **mixed-direction** (WNBA fantasy over-wide, NFL attempts/carries under-wide) so a scalar `c` can't fix them — Rung C re-ranked above family rebuilds. Gates: ruff clean, refactoring-specialist clean, golden 1905 pass (1 pre-existing real-NBA correlation red), integration 22 pass. WNBA PRA flip staged (uncommitted). · next: re-confirm the 5 HOLD SN cells (NBA RA/PR, WNBA PA/PR/fantasy) with `--hpo-selection calibrated` (does calibrated selection rescue any?); commit WNBA PRA + any new winners → devel via devel-ship-curator.
- 2026-06-22 · **§6.2 all-cells CV re-sweep vs the reworked Gate 6 + `ratio_projvol` — g6 VALIDATED, projvol REFUTED.** Re-scored all 59 cells' existing deterministic dumps against the 3-leg g6 (instant — g6 changed scoring, not training): g6 fires on exactly **4 corners via the right legs** — NBA fantasy-points `ratio_meanyr` via **CITL** at corr 0.556 (*below* the 0.58 recent-form anchor, the σ-laundered case the rework targets), WNBA FGA + NFL targets/rushing-yards via recent-form; **0 served cells false-flag**; shippable set unchanged (all 4 g6-fails were already non-shipping corners). Trained `ratio_projvol` on 41 eligible cells (39 ok, 2 NFL timeouts = same low-mean data-prep hang): **0/40 ship, g4 fails 39/40**, −5 to −16 `min_gate_slack` worse than every incumbent. Research (`/tmp/researcher_projvol_dispersion.md`): the linear `scale × volume` dispersion decode is a real symptom but **NOT the binding defect** — no `volume^p` or compound-variance scale clears the volume-stratified PIT-KS-max (≥ 0.111 NBA / 0.148 WNBA / 0.342 NFL rushing-yards); the binding defect is the **ratio-induced high-skew, zero-inflated low-volume tail SkewNormal can't shape** (low-q actual skew +0.99/+1.30/+2.90, `outcome ≤ 0.5` up to 33%) → **scoped abandonment of the projvol implementation**, efficiency × opportunity preserved only as the §6.9 count-branch `log(volume)` NB/Tweedie offset. proj-MIN itself is sound (corr 0.79 with realized minutes); ignoring its uncertainty under-states dispersion ~20–40% but doesn't clear g4 alone. `ratio_projvol` recorded tried-refuted (§6.2/§8). · next: NFL `log(volume)` NB/Tweedie offset folded into §6.6 as a deep-end family escalation (revisit only if features/blend can't ship NFL yards); firing the `centered_additive_mean10`-over-`ratio_meanyr` confirm queue on full HPO now.
- 2026-06-22 · **Gate 6 REDESIGNED — one recent-form leg → OR of three one-sided legs + anchor hysteresis** (research `/tmp/researcher_gate6_scope_and_drift.md`; spec `docs/superpowers/specs/2026-06-21-gate6-outcome-directional-legs-design.md`). `_gate6_star_ratio → _gate6_legs` (dict return); `_gate6_passes` ORs the legs; `gate_row`/`apply_thresholds`/`compute_gates`/`report.write_model_stats` threaded with `prior_g6_fired`. **(a) recent-form** (unchanged statistic) now anchored with a **`0.58/0.52` hysteresis deadband** — fire-on `0.58`, keep-on `0.52` if it fired last run, seeded from the prior `model_stats` row's `g6_star_ci_hi`/`g6_star_ref` (cold-start safe) — so a cell whose `corr` straddles the old hard `0.55` can't flip ship state on a retrain wobble; retained because it is the only leg that catches the `ratio_meanyr` holdout-corruption class (the held-out `Result` is itself suppressed, so an outcome-scored leg is structurally blind to it). **(b) CITL-under** `Σpred/ΣResult` on the same stable stars, **every cell**, fail `citl_hi < 1 − 0.03` — the σ-denominator-free counterpart to g2 that catches outcome-confirmed proportional under-prediction (NBA fantasy-points, `0.88×` recent form, g2 `z=0.22`). **(c) over** `Σpred/ΣResult` on the stable bottom-MeanYr quartile, **count/ZINB only**, guarded `mean(Result) ≥ 1`, fail `over_lo > 1 + 0.03` — owner asked to also catch systematic bench over-shooting. **Methods ruled OUT (research):** *graded* continuous-corr tolerance — over-shrinkers cluster *just above* the anchor, so a graded knob relaxes exactly where it should bite (hysteresis instead); the *sign test* — reads mean-vs-median skew not bias, false-flags shipped NBA PTS/PRA/REB. `g6_pass` now nullable-cast in the parquet (`_GATE_PASS_COLS` + `_wide_row` placeholder) — completes the wiring the original g6 left half-done. Gates: ruff clean, golden green (the 2 real-NBA correlation `ArrowInvalid` reds are pre-existing — stash-verified), integration 20 pass. · next: the all-cells CV run confirms the blast radius (expect NBA fantasy-points to fail g6 via recent-form + CITL; re-score the 6 shipped count cells against the new over-leg before any demote).
- 2026-06-21 · **Gate 6 scope WIDENED to all cells** (owner call). Was the `ratio_meanyr` SkewNormal cohort only; the anti-shrinkage check reads only the served `Blended_EV` vs `Mean10`, so it is normalization- and family-agnostic — the `corr(Mean10,Result) ≥ 0.55` anchor (not the cohort) scopes it. Motivated by g2 being insufficient: g2's bias `z` divides by the outcome σ, so a real proportional star shrinkage launders into a tiny `z` on high-variance stats. **Floor relaxed `0.98→0.95`** (`_GATE6_STAR_REF_BASKETBALL`; allow ~5% star shrinkage). Blast radius (force-scored all 11 shipped non-`ratio_meanyr` cells): exactly ONE flips — shipped **NBA fantasy-points** now fails g6 (stable stars served at `0.88×` recent form, the `z=0.22` that passed g2) → routes to demotion; NBA AST / WNBA PTS / WNBA REB pass, NBA FG3A + all 6 count cells anchor-skip (corr 0.22–0.49 `<` 0.55). `scorecard.py`: dropped the `decode_strategy`/SkewNormal guard from `_gate6_star_ratio` (param removed); golden pins updated. · next: let the NBA fantasy-points demotion flow on the next gate-status run, or investigate a composite-stat floor (fantasy points is a sum of individually-regressing stats, may warrant a floor below the single-stat 0.95).
- 2026-06-21 · §6.2 full strategy board GENERATED (first run) + the three 06-20 `next:` items closed. **Confirm-selection rule now canonical in §6.2** (spend a ~1 h real-HPO confirm only where it can move the served set: a `withheld` cell whose top *reproducible* corner ships, or a shipped cell whose board-best persistable config differs AND improves; skip a shipped cell already on its board-best config; match on `target_normalization`+`blending`+`posthoc`, dist-loss treated as the family default since it doesn't persist). **Board** (`model-strategy-driver --board`, deterministic GridSampler, 3 norms × {nll,crps}dist × {nll,crps}blend = 12 corners/cell): **13/15 cells swept**; `centered_additive_mean10` is the decisive normalization (top corner for 7/8 WNBA+NBA cells) and **blend-loss is gate-inert** (crps≡nll per cell on g1–g5, so blending stays the family default). The confirm filter cut 13→4 real-HPO confirms. **2 NFL cells EXCLUDED — `receptions`+`sacks-taken` HANG in deterministic `meditate`** (>30 min, killed at the 1800 s per-corner ceiling; stuck in DATA-PREP, not training — progress bar frozen ~1442/1693 with `2025 done`/`Downcasting floats` looping; NFL-low-mean-specific, `carries` swept fine), both book-wall non-shippers (`carries` board slack −0.064; `sacks-taken` 06-19 ship=False [[project_passing_book_degenerate]]) so **no ship lost** — logged as a separate deterministic-meditate tooling bug, normalization axis recorded tried (§8). **Real-HPO confirms (4):** (1) WNBA **fantasy-points** (the omitted 06-19 next-item) SHIPS 6/6 on `ratio_meanyr`, brier_skill +0.110 → `withheld→devel`; (2) NBA **fantasy-points** SHIPS 6/6 (`none→centered_additive_mean10`+`blending:crps`; board slack 0.194 → real brier +0.044 / g4 0.041) → `devel`; (3) NBA **FG3A** SHIPS 6/6 (`ratio_meanyr→centered_additive_eb_meanyr_k10`; board 0.174 → brier +0.044 / g4 0.043) → `devel`; (4) WNBA **DREB** NO-SHIP — g4 pit_ks **0.0597 > 0.05** (5/6); the knife-edge board slack 0.034 (det g4 0.0483) did NOT survive full HPO — [[deterministic_ab_g4_oversell]] held exactly; stays withheld, routes §6.6 centered-param SN. **WNBA RA supersede = HOLD:** candidate `centered_additive_mean10` ships standalone 6/6 and beats the `ratio_meanyr` incumbent on Brier (S2 d_mean +0.0259, CI [+0.0155,+0.0363]) and g4 (0.0342 vs 0.0446), but **S3 paired-Sharpe z +0.33 < 1.645** (and S1-on-intersection fails) → not decisive → incumbent preserved (stat_meta reverted, pickle/CSV restored byte-identical from backup). **The 4 06-19 regressors (next-item-3):** only `receiving-tds` was shipped → re-confirmed full-HPO, **REGRESSED** (g2 star z 0.83), demoted `devel→withheld`, surfaced to owner (fix fork: §6.1 `roe_mean` affine-ROE vs per-cell M-1 feature exclusion); NBA_STL / WNBA_OREB / NFL qb-yards were multi-gate KILLs in baseline too (deterministic-oversell signature) → no ship, routed (§6.6 count family / per-position). scorecard-fixture date time-bomb (next-item-1) already closed by `aff75a1`. **Net: +3 net-new ships − 1 demote.** devel served **19→21**: NBA 10→12, WNBA 6→7, NFL 3→2 — no league at 75% yet (NBA short 4, WNBA short 7, NFL short 13; the memory's "30/59" predated the serve-iff-ship prune to 19). Gates: ruff clean, golden 1873 pass (the 2 real-NBA correlation reds are pre-existing — stash-verified identical with my changes removed), integration 17 pass. devel-ship-curator carves the PR (4 stat_meta flips + this doc; human approves, never pushes). · next: §6.6 centered-parametrization SkewNormal for the `centered_additive_mean10` g4 cohort (WNBA DREB + the QW-1 volume cells — the binding under-dispersion wall, research-gated); fix the deterministic-meditate NFL low-mean data-prep hang so `receptions`/`sacks-taken` can be swept; §6.1 affine-ROE re-confirm for `receiving-tds`
- 2026-06-20 · §6.3 CV the two placeholder feature constants + revert `PrimeTime`/`RestDiff` **DONE** (closes two prior `next:` items; the "do not invent a value" discipline). **EB `_EXPANDING_EB_PRIOR_K` → per-league `{NFL: 1.0}`, else keep 10.0** (`base.py`): tiered held-out CV via the new dev-only driver `scripts/cv_feature_constants.py` (devel-denylisted like `regen_ab_batch.py`). **Tier-1** cheap 1-feature OOF (purged GroupKFold-by-player, low-`n_career` stratum where the shrink bites) said K=1 globally — but the **Tier-2 deterministic `meditate` confirm (NFL+WNBA, K∈{1,2} vs 10, `min_gate_slack` Δ) DIVERGED**: NFL **carries +2.96** (both low-K agree, large — though nothing ships either way, book-wall [[project_passing_book_degenerate]]), WNBA **uniformly prefers K=10** and **K=1 flips WNBA MIN out of ship** (the univariate OOF oversold K=1; the fitted gates caught the variance cost the OOF couldn't see). NBA scoped out (lown_frac 0.4–2.6%, inert) → keep 10. Per-league dict per the plan's divergence rule; latent (no net-shippable gain today — the only cell that moves is book-walled NFL volume). **Recency `_COMP_RECENCY_HALFLIFE_DAYS` = keep 45 (CV-inert)**: `Player comps z recent` is a recency-reweighted **convex combination**, so H only re-weights *within* the mean — held-out OOF R² spread ≤ 0.0006 across H∈{14..180}, all 7 cells / 3 leagues. (Driver faithfulness bug fixed along the way: the comp step fires 2×/gameday — the `_dispatch_volume_stats` volume pass + the served `get_stats` — and the matrix keeps the LAST, so the recency capture must `drop_duplicates(keep="last")`, not `pivot_table` mean; the mean corrupted the faithful-replay check 0→1.72.) **`PrimeTime`/`RestDiff` REVERTED** (`nfl.py` / `feature_filter.json` / `test_nfl_schedule_features`, keep `Weekday`): both dead — zero `feature_importances` (0 variance), and their `gametime`/`own rest`/`opp rest` inputs were 100% NaN on all 33,303 history rows (the QW-4 schedule backfill was deferred and never run). That all-NaN col ALSO **broke `StatsNFL.update()` perf** — the `gamelog.isna().any(axis=1)` backfill selector matched 100% of rows every run, reprocessing the full pbp history (minutes, hourly in-season once NFL starts); dropping the cols + an `update()` cleanup that sheds them from any pre-revert cache (gamelog is gitignored → per-env, the server self-heals on pull) collapses the selection **33303→0** (verified read-only). Gates: refactoring-specialist clean (base.py+nfl.py, 0 refactors), ruff clean, integration 17 pass; golden = my code clean — the 5 reds all reproduce with my changes stashed (2 known correlation reds + **3 NEW pre-existing time-bombs unrelated to this work**: `_build_live_history_fixture` hardcodes `today=2026-05-20` while `_history_to_eval_frame` filters by the *real* `datetime.now()` 30-day window → the 06-19→06-20 calendar rollover pushed the fixture's newest row outside the window → empty eval frame; flagged for a separate 1-line fixture fix). · next: fix the scorecard-fixture date time-bomb (separate, not this landing); re-confirm WNBA fantasy-points (still omitted); investigate the 4 A/B regressors (NBA_STL, WNBA_OREB, NFL qb-yards/receiving-tds)
- 2026-06-19 · **Gate 6 (anti-shrinkage) ADDED** — sixth offline ship gate, `ratio_meanyr` SkewNormal cohort only (auto-pass elsewhere). Catches the MeanYr over-shrinkage g1–g5 are blind to: the 365-day MeanYr denominator teaches a high-volume regression real games don't show, the model fits the holdout faithfully (top-decile pred/Result≈1.0) so every outcome-scored gate passes — a relative-bias g2/g3 rework is **equally blind** (the holdout's own stable stars are suppressed; verified). Gate 6 instead scores the *stable* (`|Mean10/MeanYr−1|≤0.12`) top-MeanYr-quartile `Σ Blended_EV / Σ Mean10` (player-clustered bootstrap 97.5% upper bound) vs the **causal recent-form floor** (~0.99 bball / ~0.94 NFL, off 6 gamelog seasons), anchored on `corr(Mean10,Result)≥0.55` (exempts MIN + bursty counts), **star-side only** (the ratio_meanyr denom deflates the whole distribution → never inflates the bench; the bench leg was research-refuted — would false-flag NBA PA). `scorecard.py`: `_gate6_star_ratio` + `_bootstrap_ratio_ci_clustered` + `_gate6_passes`, wired into `apply_thresholds` (`ship`) + `min_gate_slack`; 3 golden pins. **Pulled back WNBA FGA/PR/PRA** (`shipped:devel→withheld`) — gate-confirmed over-shrinkers (live symptom: FGA projecting a stable 13.4-shooter at 10.1, Win-Prob pinned at the 0.90 clamp); NBA PTS/PA/PRA/REB clear, WNBA/NBA MIN + NBA RA exempt (anchor). Research `/tmp/researcher_overshrinkage_gate.md`. worktree off origin/devel → PR. Gates: ruff clean, golden 1859 pass (the 3 live-window failures are a **pre-existing** date-fixture time-bomb — identical on stashed origin/devel, NOT mine), integration green. · next: the §8.2 root-cause fix (MeanYr-denominator artifact → recency-weighted baseline) to re-ship FGA/PR/PRA — a SEPARATE session; PR/PRA may return faster than FGA (their flag is partly real even by outcome gates)
- 2026-06-19 · §6.3 real-HPO ship-confirm + serving guards **DONE** (the deterministic A/B's follow-up verdict). Walked the 17 deterministic flippers through full production `meditate` (official 5-gate scorecard on the fused `Blended_EV` = `report._SHIP_PRED_COL`), thread-bumped serial tail (NFL + WNBA-ratio + WNBA-centered + NBA-ratio + NBA-centered, all `rc=0`, ~08:07→15:37). **16 of 17 confirmed** — WNBA **fantasy-points** was omitted from the confirm groups (gap, re-run next). **Verdict = 6 net-new promotes** (`shipped:withheld` ∧ ship=True, fresh `M_cand` pickle today): NBA **AST/PR/RA**, WNBA **PTS/RA/REB**; the 9 already-`devel` cells (NBA MIN/PA/PRA/PTS/REB, WNBA FGA/MIN/PR/PRA) re-confirm ship=True; NFL **sacks-taken** = ship=False (g1 book-wall, [[project_passing_book_degenerate]]). **Brier caveat:** the NBA trio passes all 5 gates but `brier_skill_score ≤ 0` (AST −0.021, PR −0.017, RA −0.00002) → `kelly_shrinkage≈0` (served, ~no bet); only the WNBA trio carries positive Brier skill (+0.017 / +0.033 / +0.010). So the deterministic A/B oversold 17→6 net-new, exactly the [[operation_ship_75_state]] caveat — direction was the deliverable, the real-HPO confirm is the verdict. Process note: the already-`devel` warm cells did NOT re-train (only the withheld/cold cells logged BYPASS-withhold + got fresh pickles); already shipped ⇒ no action, their §6.3 re-fit rides the devel server's next `meditate`. **Serving guards committed (model-research):** model-EV runaway winsorize `_sanitize_model_ev`+`_drop_no_history_offers` (book-independent clamp toward `K=10·max(MeanYr,Mean10,STDYr)`, SkewNormal `Sigma` rescaled to hold CV; symmetric with `_sanitize_book_ev`; research `/tmp/researcher_model_ev_runaway.md`) [`46461dd`]; scorecard `DEFAULT_PRED_COL` EV→`Blended_EV` to match the ship gate, with `--live-window` scoped back to raw `EV` (history.parquet carries no fused column, only `Model EV`→`EV`) [`46461dd`+`fa02f84`]; LightGBM `num_threads` default 1→8, `--deterministic` forces 1 (behaviorally neutral — non-deterministic HPO already forced 8 via the `_suggest_params` search space; the bump just documents the default) [`c44a6f7`]. **Curator:** `devel-ship-curator` dispatched (worktree-isolated) to carve the production delta + the 6 stat_meta flips — prepares the devel PR, human approves, never pushes. Gates: ruff clean, golden 263 pass (same pre-existing `test_find_correlation_offer_correlations_real_nba` red), integration 17 pass. · next: CV `_EXPANDING_EB_PRIOR_K`+`_COMP_RECENCY_HALFLIFE_DAYS` per league; revert-or-backfill `PrimeTime`/`RestDiff`; re-confirm WNBA fantasy-points (omitted); investigate the 4 A/B regressors (NBA_STL, WNBA_OREB, NFL qb-yards/receiving-tds)
- 2026-06-19 · §6.3 batch 1 + Playoff/series — per-league regen + deterministic §7.2 A/B **DONE** (the three BUILT entries' follow-up). **3 pre-regen gap fixes** (base.py/nfl.py, refactoring-specialist clean): (a) `comps trend` was all-zero — it read the per-player `{market} growth` col, which `_profile_stat_types` only emits for the 37 efficiency stats and never for a market → rewired to a shared `_player_recent_slope` over the MARKET, threaded as a new `_all_trend` arg to `_apply_comp_features` (`test_comp_aggregates` updated to the new contract); nunique 1→1031 on NFL passing-yards. (b) QW-4 `_schedule_context` KeyError'd on cached gamelog rows predating `gametime`/`own rest`/`opp rest` → graceful-degrade to NaN/empty (`Weekday` still derives off the always-present gameday). (c) a recent gameday whose players are all off the depth chart empties `stats` at the `Player depth>0` filter inside `_join_profiles` → `.iloc[:,[]]` crash → 2-part guard (early-return after `_join_profiles` + `len(stats)` guard on the position diag). Rebuilt NFL `attempts`/`carries` volume models (deterministic → data/models/) so `proj_*` survives the direct-`get_training_matrix` regen — they'd been ship-pruned and `get_volume_stats` early-returns on the first missing model, so all 10 `proj_*` were GONE; restored (total_GONE=0). **Regen** all 59 cells (NBA 21 @ `cutoff=2025-01-01` to skip the 2-yr-gamelog cold start; NFL 20 + WNBA 18 default 850d); clean isolation = drop-new-cols baseline (`M_base`=`M_cand`−the 16 NBA/WNBA or 19 NFL new cols, identical rows). **3 leagues regen'd in PARALLEL** via per-process `SPORTSTRADAMUS_ARCHIVE_DB` archive copies (sidesteps DuckDB's process-lifetime exclusive lock; `get_training_matrix` only reads odds); Phase B SERIAL (book_weights.json is a shared read-modify-write — though Phase B safely overlapped the remaining Phase A since `book_weights` is an in-memory module global immune to disk writes). New driver `scripts/regen_ab_batch.py` (resumable). **A/B** (deterministic, both arms identical `target_normalization`): **17 cells flip baseline-KILL(g4)→candidate-SHIP** — NBA AST/MIN/PA/PR/PRA/PTS/RA/REB (PR also SUPERSEDEs), WNBA FGA/MIN/PR/PRA/PTS/RA/REB/fantasy-points, NFL sacks-taken; ~6 more g4-fixed (NFL passing family — g4 fixed, g1 book-wall remains; WNBA BLK; NBA DREB/TOV); **4 REGRESS** (NBA_STL, WNBA_OREB, NFL qb-yards, NFL receiving-tds); rest HOLD. **CAVEAT — effect inflated by the deterministic small-HP**: the feature-poor `M_base` under-fits variance (iqr_ratio 0.005–0.72; NFL passing-yards 0.005, WNBA_MIN 0.032) and the variance-signal cols (`MeanYr_expanding_*`, `comps std`/`trend`) rescue it to ~0.87–1.23; real HPO would partly close that baseline gap, so the proxy oversells ([[operation_ship_75_state]]) — direction is the deliverable, the real-HPO ship-confirm is the verdict. **Inert** (model-agnostic variance across the 59 candidate matrices): `PrimeTime`+`RestDiff` globally inert (0 variance, NFL-only) — DORMANT pending the gametime/rest schedule backfill (deferred), not useless; `SeriesWins`/`SeriesLosses` inert for NFL (single-elim, correct) but active in 39 NBA+WNBA cells; the other 15 batch cols carry signal in every cell they appear. Benign: `Odds_synthetic` drops from 7 sparse-odds NFL/NBA cells (a train-pipeline artifact baked into the old cache, not a `get_training_matrix` output; both arms identical). Gates: ruff clean, golden green save the 1 **pre-existing** fixture-based `test_find_correlation_offer_correlations_real_nba` red (NOT mine — the test never calls `get_stats`), integration 17 pass. Artifacts: `/tmp/regen_ab/` (sidecars + 59 scorecards + `AB_SUMMARY.txt`); canonical training_data parquets now = `M_cand` (gitignored), `.regen_backup/` preserves originals. · next: real-HPO ship-confirm on the g4-flippers (`supersede_verdict` for incumbents) → devel flips; revert-or-backfill `PrimeTime`/`RestDiff`; investigate the 4 regressors; CV `_EXPANDING_EB_PRIOR_K`+`_COMP_RECENCY_HALFLIFE_DAYS` per league
- 2026-06-18 · Playoff **series-context** feature **BUILT (code+tests only)** — owner ask after the `Playoff` flag, scope "Elimination only" (build-only; regen + MLB backfill PENDING). Emits `SeriesWins`/`SeriesLosses` + `FacingElimination`/`CanClinch`. The within-series W-L record is tallied from the **teamlog** (the clean game-result source — `WL` + opponent + date present every league, unlike the player gamelog which lacks `WL` for NHL/MLB/NFL) over prior playoff games vs the same opponent inside `_SERIES_LOOKBACK_DAYS`=30 (two teams meet in ≤1 series/postseason, so the window isolates the current series from any prior-season meeting). Flags derive from a per-league wins-to-clinch (`_series_games_to_win`): **NBA/NHL** best-of-7 (base const 4); **MLB** by round via `game type` (WC `F`→bo3, LDS `D`→bo5, LCS/WS `L`/`W`→bo7) — game_type now stamped onto BOTH gamelog and teamlog in `update()` (+ carried on the upcoming dict for the serve-branch round), 60-day window/run so historical backfill rides the regen follow-up (pre-backfill → regular/0); **WNBA** round-detected (bo3 first round, bo5 later — round = distinct prior-postseason opponents within `_POSTSEASON_LOOKBACK_DAYS`=75, since the WNBA game ID carries no round and the format drifts); **NFL** single-elimination → `FacingElimination`=`CanClinch`=`Playoff`, no record. Architecture: base `_series_context` orchestrator (short-circuits to 0 when nobody on the slate is in the postseason) + shared `_tally_series_record` + hooks `_playoff_teamlog` (game-ID scope) / `_series_games_to_win`; NFL overrides the orchestrator, MLB overrides both hooks, WNBA overrides `_series_games_to_win` (R0801-clean per the specialist — genuinely different per-league knowledge). 4 cols seeded in `_BASE_STAT_COLUMNS` + registered in all 5 `Common`. Tests: new `test_series_context_features.py` (10 — record tally, the 30-day window, bo7/bo3/bo5/per-MLB-round elimination+clinch arithmetic, non-playoff short-circuit, MLB serve-from-upcoming) + the 4 cols added to the parity `_game_context` pin. Gates: ruff clean, golden 263 pass (same pre-existing correlation red), integration 17 pass. · next: folds into the §6.3 regen follow-up (MLB needs the game_type backfill on gamelog+teamlog before the series cols carry signal).
- 2026-06-18 · Cross-league `Playoff` context flag **BUILT (code+tests only)** — added alongside §6.3 (owner ask), same build-only scope (regen + the MLB historical backfill below are PENDING). Source per league chosen for accuracy: **NBA/WNBA/NHL decode the native game-ID season-type code** (`_PLAYOFF_GAMEID_CODE`, verified vs cached gamelogs: NBA `GAME_ID[2]` 4=playoffs/5=play-in, WNBA `[2]` 4=playoffs (5=Commissioner's Cup, NOT playoff), NHL `gameId[4:6]` 03=playoffs) — exact and **inherently excludes** preseason/All-Star/NBA Cup/Commissioner's Cup/4-Nations, which a date-window or game-count would mislabel (owner first suggested game-count; the same per-game type signal needed to drop exhibitions makes counting redundant, and WNBA's regular-season length drifts 40→44, so the code is strictly better). Historical reads the realized row's ID; the serve branch (upcoming schedule carries no ID) infers the phase from **the season's latest completed game's code** — games-based, no calendar window, no drift-prone length constant. **NFL** override: `season type == "POST"` historical, `week > NFL_REGULAR_SEASON_WEEKS` serve (plumbed `week`→`playoff` through `_compute_upcoming_games`). **MLB** override: `_MLB_POSTSEASON_GAME_TYPES`={F,D,L,W,P}; MLB IDs carry no code so `game type` is stamped onto the gamelog in `update()` (gameId→type map + fillna-preserve across incremental runs) — only the 60-day window per run, so **historical backfill = a full-range schedule fetch in the regen follow-up**; pre-backfill rows read regular (0). Base `_playoff_flag` hook after `_schedule_context` in `_game_context`; `Playoff` seeded in `_BASE_STAT_COLUMNS` + registered in all 5 `Common` (5 adds). The 3 `_playoff_flag` impls stay parallel (R0801 clean — genuinely different per-league encodings, refactoring-specialist confirmed). Tests: new `test_playoff_features.py` (14 — per-league decode incl. the special-event exclusions, games-based serve inference, historical↔serve agreement, MLB missing-column degrade) + `Playoff` added to the parity-harness `_game_context` pin. Gates: ruff clean, golden 263 pass (same pre-existing correlation red), integration 17 pass. · next: folds into the §6.3 regen follow-up below (MLB needs the game_type backfill before its column carries signal).
- 2026-06-18 · §6.3 Feature batch 1 (QW-4 + QW-5 + M-1) **BUILT (code+tests only)** — co-batched across all 5 leagues per the amortization rule; per-league cache regen + deterministic A/B + real-HPO ship-confirm are the explicit PENDING follow-up (build-only scope, owner call). QW-1 was already done/A/B'd (kept as candidate). **QW-5 comp-pool harvest** (NBA/WNBA/NFL/NHL; MLB auto-excluded via empty `_comp_pairs`): `comps std`/`comps trend` (profile-level in `_apply_comp_features`, reuse `_comp_wmean`/`_wcnt_p` and the `{market} growth` cols) + opp-conditional `Player comps raw` (raw-scale mean) and `Player comps z recent` (recency-weighted z **sibling**, leaves shipped `comps z` untouched), recency `exp(-days_ago/_COMP_RECENCY_HALFLIFE_DAYS=45)`. **No feature_filter edits** for QW-5 — profile-prefixed cols auto-register via `get_stat_columns`; opp-conditional ones use the `comps z` trick (init `playerProfile[[…]]=0.0`). **M-1 player-level build** (all 5 leagues for expanding/eb; interaction non-MLB): `MeanYr_expanding_shifted` = career mean over the FULL gamelog strict-`<` date (distinct from the 1yr-window `MeanYr`), `_vsopp` career H2H variant, `_eb` Efron-Morris `K/(K+n)` shrink toward the per-position baseline (`_EXPANDING_EB_PRIOR_K=10` documented placeholder, per-league CV deferred), opp-defense×player interaction `PlayerExp_x_DefAvg`/`PlayerZ_x_DefPos`; new `_expanding_features`/`_interaction_features` called after `_join_profiles`. **QW-4 NFL schedule** (NFL only): `Weekday` from the gameday **dayofweek** (NOT the ragged `weekday.str[:-3]` slice — R2), `PrimeTime` = kickoff ≥ `_NFL_PRIMETIME_HOUR`=20, `RestDiff` = own−opp rest; plumbed `gametime`/`own rest`/`opp rest` through the `_assemble_gamelog_frame` merge + `rest_diff` into `upcoming_games` (R1, required for serve parity); new NFL `_schedule_context` via a no-op base hook at the end of `_game_context`. Registration: M-1 bare-stats + QW-4 in per-league `Common` (26 adds, `MeanYr_expanding_*` also seeded in `_BASE_STAT_COLUMNS`); the `_n` EB-input cols deliberately unregistered. Tests: leakage extended (full-gamelog strict-`<` sentinel reading >300d back + vsopp + static `<` source check), parity harness extended (explicit M-1 pin), 3 new derivation-pin files (`test_player_expanding_features`/`test_comp_aggregates`/`test_nfl_schedule_features`). Gates: ruff clean, golden 263 pass + the **pre-existing** `test_find_correlation_offer_correlations_real_nba` red (proven pre-existing via stash, NOT mine), integration 17 pass. refactoring-specialist: 1 consistency fix — added `comps std`/`comps trend` to the `profile_market` zero-init (closes a same-date cross-market staleness gap, matching `comps mean`). · next: §6.3 follow-up — preserve baselines → per-league regen (NBA incremental-cutoff per [[training-matrix-regen-concat]]) → deterministic A/B → inert-revert SHAP<0.001 → real-HPO ship-confirm; then CV `_EXPANDING_EB_PRIOR_K` + `_COMP_RECENCY_HALFLIFE_DAYS` per league
- 2026-06-18 · §6.5 Fitting-half book-input track — final state: WS1/WS3/WS4 committed, WS2 (book-skew) BUILT→A/B-REFUTED→**REVERTED**. Committed correctness: WS3 power de-vig `no_vig_odds(method="power")` (lopsided-only, `|p_over−0.5|>_DEVIG_LOPSIDED_FLAG`=0.3; proportional on the body, byte-identical at even money); WS4 count-tail reg `_sanitize_book_ev` shrinks a runaway book EV to `μ̂+K·SD` (`_BOOK_EV_MODEL_SD_CAP`, keeps the 5×line corruption trigger); WS1 ingest alt-line ladder capture `archive.add_ladder` (per-rung de-vigged, accrues the §1.1 asset — invisible to the gates, usable only once accumulated). WS2 was the shape-borrow book recovery (closed-form book `loc` holding the de-vigged median, skew from a non-circular within-player standardized-residual prior, threaded serve+train-blend via a pickle `book_skew`). A/B (deterministic §7.2; NBA PTS/REB/MIN; faithful: invert the baseline blend → re-blend both arms through production `fused_loc`, book_skew=0 reproduces the CSV to 1e-14) **REFUTED** it: the within-player *marginal* residual skew (α≈3–4) over-states the book's CONDITIONAL skew (which the model already carries) → shifts the blended mean +8/+11/−1% and **worsens g4 on all three** (PTS 0.016→0.063, REB 0.027→0.083, MIN 0.031→0.053) plus g2, even at w=0.9. Operator call: assume skew=0 for books until real ladder data, so WS2 fully reverted (`_book_loc_with_skew`/`skewnorm_alpha_from_skewness`/`_within_player_book_skew`/`_BOOK_SKEW_PRIOR_ENABLED`/the `fused_loc(line=, book_skew=)` params/the pickle `book_skew` key/the train-blend threading all removed; commits `eaa19a7`/`dffc855`/`2018199`/`10ee215`/`0233a75`/`9605d41` hold the build history); production books are symmetric `N(ev_b,(ev_b·cv)²)`, byte-unchanged from before the track. Coupling RECORDED (Pooling half): the BLP / learned `w` must be co-fit with the book-shape, never sequenced (Ranjan-Gneiting 2010 Thm 1). · next: let WS1 alt-line ladders accrue, then fit the book skew per-line from the ladder (the marginal residual is the wrong quantity), co-fit with the pool (Ranjan-Gneiting)
- 2026-06-17 · §6.0 Stage 0 finished off (the two remaining actionable items; #4 parity harness + #6 driver-on-devel already done per audit — driver pair present on origin/devel; #2 hysteresis + #5 NFL g1×dispersion stay deferred/pre-registered, blocked on the §6.6 centered-SN modeling fix per the entry below). **#6.0.3 stale-importances purge DONE:** the 155 all-zero `Player Player *_asof` residue rows (every one ends `_asof`, has a single-prefix counterpart, 0 non-zero cells) dropped from BOTH `feature_importances.csv` + `feature_correlations.csv`; new `tests/golden/test_feature_importances_hygiene.py` pins absence in both (skips when the gitignored CSVs are absent). Gotcha: the CSVs are gitignored/runtime-recomputed and locally only 19 of 83 cells are pickled, so the doc's `see_features()` full-rebuild would CLOBBER 64 cells' drift history here → used a byte-preserving line-drop (server still uses `see_features()` after a full meditate); `compute_market_importance`'s `how="outer"` join re-accumulates stale rows across runs, so only an explicit drop or full rebuild clears them. **#6.0.1 free-passer detector WIRED (report-only):** `graduation.free_passer_cells` (mirror of `served_cells_failing_ship`, predicate inverted to `ship==True ∧ shipped=withheld`, NA/missing→[]) echoed by `generate-ship-config` on BOTH branches, so the monthly `gate-status` cron (`--branch main`) logs `FREE-PASSER` lines; NEVER auto-flips (manual scorecard re-confirm; a sweep pass disagreeing with the scorecard is a scorer bug). 0 free-passers today. TDD throughout (unit + CLI tests). · next: §6.6 centered-parametrization SkewNormal for the `centered_additive_mean10` g4 cohort (real under-dispersion, research-gated), or §6.3 QW-4/QW-5/M-1 feature batch
- 2026-06-17 · §6.5 g4 dispersion lever CLOSED — root cause was NOT a missing dispersion mechanism, and NOT (as a same-day investigation first concluded) a skew fit "gated off" by `_DISPERSION_SKEW_MIN_GAIN` needing a per-cell val→test discount. That earlier read was a wrong-PIT-space artifact: the served fit's `_served_sn_pit_ks` scores in observed/`NONE` space (the fit's own objective, e.g. 0.0403), which for a zero-inflated cell ≠ the gate's decoded score (0.1225) until the denom matches — the fit was fine, the GATE was wrong. The real cause is a GATE-ONLY decode bug: the scorecard's `_decode_sn_loc_scale` hardcoded `denom_col="MeanYr"`, but a zero-inflated SkewNormal cell (`hist_gate > NONZERO_DENOM_GATE`=0.05) encodes — and the betting path serves (`model_prob.py:122`) — against `MeanYr_nonzero`. So encode↔decode was not identity and g4 scored a predictive ×`MeanYr/MeanYr_nonzero` too narrow → spurious under-dispersion. Tell: `g4_iqr_ratio` == that denom ratio (WNBA_AST 0.766; g4 0.1225 → 0.0403 with the right denom = the served fit's own value). Serving was always correct ⇒ no mispricing; the bug only blocked the offline ship gate. FIX: persist the cell's `denom_col` as a constant `DenomCol` column (`_step_persist_artifacts`, mirrors the `GlobalMean` persist), read it back in `_decode_sn_loc_scale` (fallback `MeanYr` ⇒ zero regression on legacy CSVs, exact on retrain); regression pin `test_scorecard.py::test_decode_sn_loc_scale_uses_persisted_denom_col`. NOT a gate loosening — it is *stricter* on some NFL `ratio` cells the bug had been flattering (yards 0.086→0.119). Blast radius (fresh re-score of all 33 test_sets, not cached): flips exactly ONE cell fail→pass — WNBA_AST (its other 4 gates already pass → ships on next retrain); the `centered_additive_mean10` cells (NBA AST/DREB/FGM, WNBA PTS/REB/DREB, NFL receptions) are UNAFFECTED (their `decode_scale` ignores the denom) and fail g4 at 0.08–0.18 for GENUINE under-dispersion. Process lesson: `report()` is end-of-meditate, not per-market — a killed meditate leaves fresh per-market test CSVs but a stale `model_stats`; re-score the CSV, never trust `model_stats` after a partial run. · next: the remaining g4 wall is the `centered_additive_mean10` cohort (real under-dispersion, not a bug) → §6.6 centered-parametrization SkewNormal, a modeling fix (research-gated)
- 2026-06-17 · §6.3 QW-1 game-script A/B (deterministic, §7.2) across 15 volume cells / 3 leagues = NO SHIP, NO SUPERSEDE anywhere (all 15 `supersede`=HOLD). Method: baseline = the 4 QW-1 cols dropped from `feature_filter` {league}/Common (cache untouched; `get_stat_columns` stops selecting them — isolates the feature, no prune ambiguity); candidate = with them; both `--deterministic --target-normalization ratio_meanyr` (production for the devel cells, default for withheld norm=none placeholders — normalization held identical across arms). NFL five (carries/completions/attempts/receiving-yards/rushing-yards): candidate equal-or-worse, all KILL g1+g4; attempts NOT revived (book-axis death [[project_passing_book_degenerate]] holds on the feature axis). NBA five (MIN/PTS/REB/AST/FGA): faint **+** lean — AST significant on S2 (d_mean +0.0012, CI excludes 0) AND S3 sharpe (z+2.33), FGA S3 sharpe z+11.18, MIN +0.0014; REB worse (sharpe z−3.13) — but every cell still HOLD and KILL g4. WNBA five: leans **−** (PTS d_mean −0.0028, AST −0.0012, FGA −0.0023, all CIs exclude 0). The NBA/WNBA run froze `comps.json` across arms to kill a deterministic-write float-jitter seen on the NFL run. **g4 status — VERIFIED by re-running the scorecard on the production test_sets (not cached `model_stats`):** the withheld good-discrimination volume cells genuinely FAIL g4 in production — NBA AST `g4_pit_ks` 0.135 / FGA 0.097, WNBA PTS 0.077 / REB 0.088 / AST 0.110, NFL carries 0.133 / completions 0.146 / attempts 0.214 (all above the `max(δ, 1.36/√n)` threshold); the served dispersion calibration (`fit_skewnorm_dispersion_c`/`_skew`) is insufficient for them. The devel-shipped cells DO pass g4 (NBA MIN/PTS/REB 0.016–0.031, WNBA MIN/FGA 0.030/0.049) — for those alone the deterministic A/B's g4-KILL was an artifact (fast fixed-HP skips the calibration). NFL additionally fails g1 (book wall, [[project_passing_book_degenerate]]). So g4 under-dispersion IS the binding wall for the withheld volume cells, and QW-1 does not move it. Lesson: cached `model_stats` held only the last local meditate's cells (the withheld ones absent), which nearly led to a wrong "g4 is just an artifact" read — resolve a gate-data conflict by RE-RUNNING the scorecard on production test_sets, never cached stats alone. Deviation: the drop-cols baseline stands in for the pre-regen baseline (bit-identical for the concat-backfill leagues). · operator call: KEEP QW-1 as a candidate (no revert; faint+ on NBA AST/FGA; no-filter philosophy). · next: the g4 dispersion lever — the served calibration clears the easy cells but not the withheld good-discrimination volume cells; improving it is research-gated → `research-analyst` brief before any dispersion-mechanism change
- 2026-06-17 · QW-1 cache regen DONE (all 5 leagues). WNBA (18) + NFL (20) via concat-backfill — `OppTotal`/`Spread`/`GameTotal`/`Blowout` reconstructed per (Player, Date) from the gamelog, validated faithful by cached `Total` == reconstruction at rate 1.0 outside the corruption window; `fillna(0)` order + zstd/index=True match production. NBA needed more (totals corruption propagated into the date-varying `Player`/`Defense totals gain` slopes, corr 0.925, mean|Δ| ≈ 70% of IQR → only a per-gameday `get_training_matrix` reproduces them), but a full COLD rebuild hits two pre-existing crashes (empty-comps `KeyError`, zero-std `ZeroDivisionError`). Resolved via the operator's INCREMENTAL-CUTOFF third path: keep the pre-March warm rows (+ backfill the 4 cols from the corrected gamelog) and refill only March-onward via `get_training_matrix(cutoff=2026-02-28)`. The warm cutoff sidesteps both crashes — recent gamedays have mature comp pools, so no empty-comps and no zero-std column. All 21 NBA cells regenerated: corruption-window `Total` mean 114.6 (corrected; corrupt was ~165 = ×1/ln 2), blowout ≈ 0.26, the 4 QW-1 cols present, fantasy cell included. See [[training-matrix-regen-concat]]. · next: §6.3 deterministic A/B + ship-confirm on the regenerated cells (§7.2; NFL volume five first, co-batch QW-4/QW-5/M-1 to amortize)
- 2026-06-17 · gamelog re-baked from corrected archive + blowout thresholds DATA-DERIVED. (1) NBA `gamelog.parquet` `totals` re-read from the fixed archive over the corruption window (2026-03-14→04-24): the 2308 inflated rows (old>135) deflated at ratio 1.4435 ≈ 1/ln 2, the 3274 `default_total` rows held or improved, window mean 133.8→114.7, max 194.3→134.8 (realistic), zero rows >145 remain, neighbors untouched — scoped to the window (inflation provably doesn't leak outside) so historical rows aren't degraded to default. (2) `_BLOWOUT_SPREAD_THRESHOLD` retuned from per-player-normalized starter playing-time vs |spread| knees (MIN/snap-pct/TimeShare): **the effect is FAVORITE-ONLY** (favorite rests starters; underdog starters play full/more, chasing) — the signed `Spread` feature carries the asymmetry so `Blowout` stays |spread|-based. NBA/WNBA 11→10 (knee ~10 ≈ p80), NFL 10→11 (snaps resist past p80), NHL 1→1.5 (TimeShare flat/rising in blowouts ⇒ NO effect, value nominal, expect Blowout inert + reverted at validation), MLB 2.5 unchanged (0% local line coverage, undated default). · next: apply `ev *= ln(2)` window fix to the PRODUCTION server archive, then the QW-1 per-league matrix regen picks up the corrected gamelog totals
- 2026-06-17 · archive NBA `Totals` corruption ROOT-CAUSED + stopgapped + guardrailed (found while validating QW-1's `OppTotal`/`Total`, which read archived totals). Mechanism: `get_ev`'s default flipped to `"Gamma"` in `83c4b43` (3-12); Gamma at cv=1 is exponential (mean = median/ln 2 ≈ 1.4427×), so every NBA team total written 2026-03-14→04-24 inflated ×1.4427 until `a323049` (4-24) reset the default to SkewNormal. Moneyline clean (never calls `get_ev`); NBA-only; all books uniform. Recovery validated WITHOUT the paid API (corrected median 230.1 sits between clean neighbors; raw was 0.5% outside realistic range). Stopgap applied to local `archive.duckdb`: `ev *= ln(2)` over the window with an `ev>135` idempotency guard — window mean 165.6→114.8, clean neighbors untouched. Guardrail (operator ask, "game-line EV always normal"): new symmetric `dist="Normal"` branch in `get_odds` (forced alpha=0) + `moneylines._GAME_LINE_DIST="Normal"` pinning all four game-line `get_ev` calls, immune to any future default flip; pinned by `tests/golden/test_book_round_trip.py` (even-money price → line; Gamma diverges ×1.44). · next: apply the same `ev *= ln(2)` window fix to the PRODUCTION server archive, and re-bake gamelog `totals` from the corrected archive at QW-1 regen time
- 2026-06-17 · §6.3 QW-1 game-script features BUILT (code+tests only): `OppTotal` + derived `Spread`/`GameTotal`/`Blowout` in `_game_context` both branches (historical = gamelog team→total map, upcoming = symmetric `archive.get_total(opp)`; existing `Total` untouched, doc's "two get_total calls" satisfied via the archive-baked gamelog), `_BLOWOUT_SPREAD_THRESHOLD` per-league constant, registered in `feature_filter` Common ×5 leagues. Parity harness extended + new `test_game_context_features.py` derivation pins (both blowout sides); 3 brittle integration fixtures fixed `M[cols]`→`M.reindex(columns=cols)` to match production `_step_build_splits` stale-cache resilience (QW-1 exposed the latent gap; determinism gate already did this). Real-NBA eyeball: `totals` = implied team total confirmed (2731 games, 13.9% equal, median spread 5.55, GameTotal ≈223, OppTotal non-null 1.0). All gates green (1 pre-existing golden red = untracked 0-byte `nba/corr_same_team.parquet`, unrelated). · next: §6.3 per-league regen + deterministic A/B on the NFL volume five (co-batch QW-4/QW-5/M-1 to amortize regen), then ship-confirm
- 2026-06-17 · §6.0.4 train/live parity harness BUILT (`tests/golden/test_train_live_feature_parity.py`, ~3.5 s, sanity-flip verified) — Stage-3/4 feature work now unblocked. Free-passer sweep null (0 ship-True∧withheld in the 19 fresh `model_stats` rows); hole-#0b deferred (cycle-1 g4 churn only). also synced model-research `stat_meta.json` to origin/devel — demoted WNBA PA + NBA BLST `devel`→`withheld` (fail g4; `test_ship_gate_invariant` now green; served set == production). · next: §6.0.3 stale-importances purge or §6.3 QW-1 features
- 2026-06-15 · INBOUND BUG (from dashboard-ux) · `strategies/profit_sim.py` has a payout/Kelly accounting bug that feeds the **S3 paired-Sharpe gate** (`training/scorecard.py`) and **Gate-2 Kelly yield** (`nightly._profit_sim_kelly_yield`): Sleeper `compute_payout` returns the gross boost (median 1.74) but `_settle_day` adds it as NET profit (should be `boost − 1`); the Kelly branch's `if payout <= 1: continue` skips every Underdog bet (payout 0.909) so Kelly mode bets only the overpaid Sleeper legs → MC bankroll explodes (+3.2M%); snapshot also carries `inf` boosts. Owner: fix the canonical engine (no duplicate sim), but it moves gate outputs ⇒ this lane + owner sign-off (§8). Fix payout-net + Kelly net/decimal + inf-guard, AND add optional staking params (flat-off-initial / daily-exposure cap / fractional-Kelly+cap) with **defaults = current behavior**, then revalidate S3 + Gate-2 and curate. Unblocks the dashboard-ux Strategy-simulator rework. Full diagnosis: memory `profit-sim-payout-kelly-bug`.
