# Model Improvement Track

> Status: ACTIVE — live-alignment P1 + standing breadth harvest (Ship-75 feeds D3)
>
> **The single home of record for the model improvement track** — the lead lane of
> [`sportstradamus_roadmap_v3.md`](../sportstradamus_roadmap_v3.md). Every workstream, lever,
> per-league path, validation recipe, and session rule for making the market models profitable —
> live alignment, the standing breadth harvest, MLB/NHL activation execution, family escalation,
> and ladder/tail calibration — lives here, through the D3/D5 decisions.
>
> Consolidated 2026-06-12 from four docs, now archived: `operation_ship_75.md`,
> `operation_ship_90.md`, `feature_improvement_plan.md`, `handoffs/model-track.md`
> (see [`../archive/`](../archive/); pre-consolidation text recoverable via
> `git show 805cea7:docs/operation_ship_75.md` on devel and
> `git show 46338ef:docs/operation_ship_75.md` on model-research, whose operating-loop
> refinements are folded in below). Reworked 2026-07-07 to the profit-first workstream layout
> (owner reframing; supporting briefs
> `/tmp/researcher_{count_family,continuous_family,copula_stage0,sweep_ux_versioning}.md`).
> Three companions stay canonical and are **not** restated
> here: [`../ship_gate.md`](../ship_gate.md) (gate thresholds — owner-only),
> [`../operation_ship_references.md`](../operation_ship_references.md) (research verdicts,
> citations, commit refs), and the roadmap (program index, other lanes, deferred
> register).

## 1. Mission & money logic

**Make the served models money: beat the DFS apps (Underdog, Sleeper) and books mispriced
against consensus, live, with calibrated full predictive distributions.** This is the lead lane:
calibrated distributions are the input every decision engine consumes, and Gate-4 PIT-KS
calibration *is* alt-line pricing accuracy — the alt-line/Ladders/Rivals surfaces where the
profit concentrates. The six offline ship gates (g1–g6 in
[`training/scorecard.py`](../../src/sportstradamus/training/scorecard.py)) certify
**deployable**; the live Gate-2 soak certifies **profitable**; the live evidence says the
deployable→profitable gap — selection, sizing, benchmark alignment — is now the binding
constraint (§3.3, §6.10). Breadth stays instrumented (targets below) because breadth is the
number of +EV opportunities per slate and the D3/D5 gate input — but it is an input, not the
goal. Work is ordered by payoff-to-effort per the §6 workstream table; individual levers are
expected to fail, and every workstream carries pre-written escalation branches.

Breadth targets (D3/D5 inputs; MLB/NHL targets are set by the D1/D2 activation packets —
[`mlb-nhl-activation.md`](../archive/mlb-nhl-activation.md)):

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
[`mlb-nhl-activation.md`](../archive/mlb-nhl-activation.md)).

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
| Gate-2 live-soak thresholds (n, book-BSS, precision) — owner-only | [`graduation.py`](../../src/sportstradamus/training/graduation.py) |
| Release surface per cell | `stat_meta.json` `shipped` (git carries flip history) |
| Per-cell gate numbers | `data/training/model_stats.csv` (mirror of the parquet) |
| Sweep board (per-cell best corner + gate slacks) | `data/research/strategy_research_board.csv` |
| Live telemetry / recorded picks / graduation | `data/runtime/*.parquet` + `sportstradamus ship graduation` |
| Served-offer snapshots the dashboard reads | `data/runtime/{current_pickem,current_offers,history}.parquet` |
| Product-payout rules (Rivals, Ladders, multipliers) | the decision lanes (`strategies/`, `books.py`), **not** this track |
| Lever stack, stages, per-league routing, session rules | **this doc** |
| Research verdicts, citations, commit refs | [`../operation_ship_references.md`](../operation_ship_references.md) |
| Program index, other lanes, decision gates D1–D7, deferred register | [`../sportstradamus_roadmap_v3.md`](../sportstradamus_roadmap_v3.md) |
| Ship mechanics how-to, package map | `CONTRIBUTING.md` |
| Session law (gates, subagents, hard rules) | `CLAUDE.md` |

## 3. Verify before you trust

Numbers drift on every ship and every `meditate`. Per-cell standings are **never** restated as
prose anywhere in this doc; derive them:

```bash
# Concurrency guard — a running sweep OWNS stat_meta.json; never meditate or edit it under one
# (bracket on [m] so pgrep's own command line doesn't self-match)
pgrep -af "[m]editate|model.strategy.(sweep|confirm)" || echo "no sweep/train running — safe to edit stat_meta"

# What production tracks
git fetch origin && git log --oneline origin/devel -5

# Shipped counts per league (withheld / devel / main)
python3 -c "import json,collections; m=json.load(open('src/sportstradamus/data/config/stat_meta.json')); [print(l, dict(collections.Counter(c['shipped'] for c in v.values()))) for l,v in m.items()]"

# Board rollup — cells with a passing corner per league (offline headroom)
python3 -c "
import pandas as pd
b = pd.read_csv('src/sportstradamus/data/research/strategy_research_board.csv')
best = b.sort_values('slack').groupby(['league','market']).tail(1)
print(best.assign(pass_=best.ships).groupby('league').agg(passing=('ships','sum'), cells=('market','nunique')))"

# Per-cell gate numbers (g1–g6, pit_ks slack, ship) — rewritten by every meditate
ls -la data/training/model_stats.csv   # stale ⇒ re-run meditate before trusting

# Lifecycle per (league, market): not-shipped / in-test / graduated / demoted
sportstradamus ship graduation

# Recorded live picks (Bet populated) per league — the era-aware WS-1 spine (§6.10)
python3 -c "
import pandas as pd
h = pd.read_parquet('src/sportstradamus/data/runtime/history.parquet')
p = h[h['Bet'].notna() & h['Actual'].notna() & h['Line'].notna()]
print('recorded picks per league:', dict(p.groupby('League').size()))"

# Ladder / alt-line accrual (seeded 15.5M rows by the 2026-07 backfill; must keep growing)
python3 -c "
import duckdb
con = duckdb.connect('archive/archive.duckdb', read_only=True)
print('ladder rows:', con.execute('select count(*) from ladder').fetchone()[0])
print(con.execute('select league, count(*) from ladder group by league').fetchall())"

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

This track prices stats, not app products — but three product-surface facts feed the money
lens (§1.1) and go stale silently, so verify them at the start of any WS-1/WS-4 session:

- **App coverage** — which apps/markets are actually served today (`current_offers` platform
  split); a market we model but nobody serves earns nothing.
- **Sleeper multipliers / Underdog boosts** — the payout curves live in the decision lanes
  (`books.py`, `strategies/`); a multiplier change moves EV without any model change.
- **Ladder / alt-line accrual** — the `ladder` archive table must keep filling (the §3 probe).
  Seeded to 15.5M rungs across all five leagues by the 2026-07 historical backfill; live accrual
  adds only primary-market rungs (the `*_alternate` keys are backfill-only by design —
  `ALT_MARKET_KEYS` in `scripts/backfill_historical_odds.py`, deliberately not in `stat_map`), so
  a live-side alternate fetch is a future cost decision. The right tail those rows price is
  Gate-4's job (§6.11).

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

The **same repair recipe is MLB's activation gate** (§6.7, WS-2): MLB's archived book prices
carry the same klepto-era seed, so its cells fail g1 on a degenerate book, not a bad model. The
audit → Odds API sign-off → `backfill_historical_odds.py` → `inject_backfilled_odds.py` →
matrix rebuild → re-sweep sequence is the D1 critical path; nothing MLB ships until it grades
against an honest book.

### 3.3 Diagnosis — two-sided: offline shape and live alignment

Diagnose from both ends now. The offline board says where predictive **shape** is still wrong;
the era-aware live read says whether the shape that ships actually **makes money** — and the two
no longer agree.

**Offline (board).** The dominant offline symptom is still Gate-4 under-calibration, and the
residual no-pass cohort has a known structure: the ZINB count cells over-cover at the NegBin
variance≥mean floor and the shape-bound SkewNormal cells rail their skew head — the §6.6 family
escalation, research-done (§3.4) and unbuilt. Regenerate the cohort from the §3 board rollup;
never cite fixed counts as current. The 2026-06-03 full-board re-score that launched the
calibration ladder (every test CSV through `compute_gates`, the production path) is preserved as
the **founding snapshot** — evidence of the shape of the problem, not a live census:

| Primary failure (2026-06-03 snapshot) | Count | Routed to |
|---|---|---|
| Gate 4 only (g1/g2/g3/g5 all pass) | 24 | calibration (§6.1) / normalization (§6.2) |
| Gate 4 + marginal Gate 1 (`ci_hi` 0.007–0.018) | 6 (all NFL) | calibration → features (§6.3) / blend (§6.5) |
| Gate 4 + Gate 2/3 (bias) | 3 | mean rung then scale rung (§6.1) |
| Multi-gate (g1+g3+g4) | 2 (NFL passing-first-downs, qb-yards) | hardest; features + per-position (§6.6) |
| Gate 1 only / Gate 1+5 (edge) | 2 (WNBA STL; NBA PF) | features (§6.3) |
| Pass now, un-promoted | 3 (WNBA BLK, FG3M, TOV) | promoted since — free-passer sweep is §6.0 |

**Live (era-aware) — the benchmarks disagree (WS-1, §6.10).** Reading recent windows rather than
the 5-month aggregate (which mixes model generations) surfaces a conflict the offline gates can't
see: recent recorded picks (~last 30 days, WNBA, n≈492 resolved) hit **~64% against the app
lines**, and even the cells the live soak **demoted** (PTS/REB) hit ~67%, with pick calibration
monotone and slightly **under**-confident — yet the live Gate-2 soak, which keys on **sharp-book
BSS** rather than app-line profit, demoted five WNBA cells at scale. Older mixed-era aggregates
show the opposite tail: weak-edge adverse selection (bottom deciles ~37% vs ~50% predicted). So
three benchmarks — app-line hit rate (the money), book-BSS (the graduation criterion), and CLV —
point different ways, and reconciling them is WS-1's job. Two hard constraints on that work:
Gate-2's thresholds are **owner-only** (findings become a decision packet, never a session edit),
and there is **no model-version stamp on live rows today** (the `Step` column is line
granularity), so era attribution must be built (Stage-0) or approximated by the per-cell
calibration fingerprint until it is.

The two modeling **heads still fail in opposite offline directions** (Czado-Gneiting-Held 2009) —
there is no single "widen everything" fix:

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
| Strategy sweep + confirm engine (`model_strategy/sweep.py` + `model_strategy/confirm.py`) | **Alive, registry-driven, identity-bound** — `model-strategy-sweep --confirm` ranks exact signed base corners by min-gate slack under a budgeted conditional-TPE study, scored holdout-blind; confirm walks each cell's nominees (top-K + seeds + incumbent) at full HPO, exact-checks model/CSV/model_stats, and keeps the first all-six-gate ship or restores the prior cell. Deterministic ranks, real-HPO ships (§6); generic selector and serving validation landed. Final gate audit is recorded in §6 | §6 + R4 brief |
| Forceable distribution family (`dist` via `stat_meta.json`) | **Alive** — `_resolve_dist` reads the cell's `dist` as authoritative input (SkewNormal / ZINB / NegBin); makes family a one-line sweep axis, no code edit per cell. The data-driven mean≥2 / zero-rate rule is now only the fallback | `[[dist_selection_forceable_via_stat_meta]]` |
| Centered-target normalization (`centered_additive_mean10`) | **Alive & shipping** — out-calibrates `ratio_meanyr` on Gate 4 for several cells the scalar width fix can't reach (run the §3 shipped-counts block; several cells carry it in `stat_meta.json` today). The old P1 "dead" call judged it as a *mean-compression* fix under the pre-PIT-KS gate — superseded | refs §3 + sweep board |
| Calibrated HP-selection search-gate (Lever-1, `--hpo-selection calibrated`) | **Alive, rebuilt as v3** — raw CV loss objective + OOF PIT-KS as a TPE feasibility constraint, 1-SE final pick (§6.1 Lever 1); ship-deciding knobs persist per-cell so cron re-fits reproduce them | `[[calibration_hp_selection_lever]]` |
| `init_score` warm-start baseline | **Dead** — byte-identical to plain NegBin | refs §3 |
| ZTNB-hurdle likelihood | **Refuted** — incompatible with the derived-π decode; would regress the shipped hurdle markets | refs §6 |
| T5 multiplicative factorization (volume × efficiency) | **Killed** — Goodman variance-of-products gives +27% predictive-variance inflation on the priced cell | refs §9 |
| Family build — count wall + shape-bound | **Both families BUILT + wired end-to-end** (§6.6): Double Poisson (`DPO`) in the count sweep (`ZINB ∪ NegBin ∪ DPO`, 16 corners/cell) and centered-SN behind the `sn_param` axis (SkewNormal 24 corners/cell); full-cohort screen routes 15 DP-mandatory / 8 hurdle / 18 plain-NB of 41 count cells. Pilots (NBA PF, WNBA TOV, NHL points; then WNBA PA + NFL receptions) decide ship/kill; SHASH still gated on centered-SN pilot evidence | R1/R2 briefs; §6.6 |
| HurdleZINB (per-cell `zinb_mode`) | **Alive & shipped** — 6/8 NBA ZINB markets | refs §4 |
| Post-hoc mean correction (`roe_mean` / `isotonic_mean`) | **Alive & shipped** — `MEAN_STAGE` in [`posthoc.py`](../../src/sportstradamus/training/posthoc.py); use skeptically (§6.1 Rung A) | refs §8 |
| Post-hoc probability recalibration (`prob_recal_*`) | **Alive** — `PROB_STAGE` built, available per-cell | posthoc.py |
| Post-hoc scale/dispersion (joint `(c, skew_cal)` vs PIT-KS) | **Shipped** — route 1a-hybrid; Levi closed-form σ-scaling is a dead end (diverges 5–7000× on skewed cells) | §6.1 Rung B |
| Player-level features (expanding-mean, EB-shrunk, opp-defense) | **Alive, unbuilt — narrowed** (§6.3 M-1). Pilots WNBA STL + NBA PF; NFL volume cells are **not** feature-starved (485 candidate cols) — their negative BSS is target-shape, not features, so NFL runs 1–2 falsification pilots only, routing the rest to §6.6/§6.1 | `[[nfl_volume_cells_feature_mature]]` |
| Per-position model split (NFL, T11) | **Alive, on the table** — a live lever now, not held for Ship-90 | refs §9 |
| TabPFN / tabular foundation model | **Deferred backlog** — potential prior-data-fit alternative to per-cell GBDT; not researched, not scheduled; parked until the family and live-alignment lanes resolve | roadmap deferred register |
| Shaped-book leg in the served blend (WS2) | **Split verdict** — GO as a **book-only fallback** leg (shipped, WS1/WS3/WS4 kept), NO-GO **inside the served blend** (co-fit drives the shape weight to the floor; worsens g4). Never re-add to the blend without new evidence | `[[ws2_settling_split_verdict]]`, `[[book_skew_shape_borrow_refuted]]` |
| Whole-CDF isotonic-PIT recal on count cells (Rung-C-on-count) | **Dead** — the monotone CDF map degrades low-mean count cells; Rung C ships on continuous cells only (PA), count residual routes to §6.6 | `[[rung_c_whole_cdf_recal]]` |
| Context-conditioned book `cv` scale law | **Refuted** — 0% OOS book-PIT-KS gain (45/45; slope fits to 0). Book CDF is mis-shaped but decoupled from the served gate at `w≈0.90` ⇒ standalone book rebuild NO-GO | `[[book_distribution_audit_nogo]]`; refs |
| Pooling-half blend rebuild (BLP + decoupled location/shape) | **Refuted (this cohort)** — decoupled (A) ill-posed, BLP (B) a wash (OOS ΔKS≈0); at `w≈0.90` nothing for the wrapper to repair; over-wide cells are family/shape-bound → §6.6. Re-probe per cohort; design in §6.5 | `[[pooling_half_blp_nogo]]`; refs |

## 4. Locked decisions

- 2026-06-10 — the model track stays the lead lane; other lanes never preempt it.
- 2026-06-10 — gate definitions and thresholds are owner-only; a gate change never counts as a
  lever (§8).
- Standing — ship incrementally: a Gate-1-clearing cell goes to `shipped: "devel"` and counts;
  don't hold ships for batches.
- Standing — new external data is free sources only; all five leagues get equal feature-effort
  budget (refinement for NBA/WNBA/NFL, foundation parity for MLB/NHL).
- 2026-07-07 — mission reframed profit-first: beat the DFS apps and mispriced books **live** with
  calibrated distributions; calibration is the product; breadth is a D3/D5 input, not the goal.
  Live alignment (§6.10, WS-1) is Priority 1.
- 2026-07-07 — MLB and NHL activation is folded into this track (WS-2, §6.7/§6.9); no MLB/NHL
  cell ships before its owner gate (D1/D2 — [`mlb-nhl-activation.md`](../archive/mlb-nhl-activation.md)).
- 2026-07-07 — a Fable **Stage 0** cleans the sweep/confirm engine and builds structure (corner
  registry + family axis, resume/queue, version stamping, new-family scaffold, this doc rework)
  before any Sonnet grunt work; the execution queue after it is Sonnet-sized.
- 2026-07-07 — research for this rework was done by Fable and folded in before the doc edits;
  standing research-first still binds every distribution-family / dispersion build (§7, §8.2).

**Conflict order:** command output > `CLAUDE.md` / `CONTRIBUTING.md` > this doc > roadmap v3.
If command output contradicts prose here, the output wins — fix the doc in place (minor) or stop
and ask (material).

## 5. Module footprint & canonical paths

Breadth / family / calibration work: `sportstradamus.training` (pipeline, scorecard, report,
calibration, `model_strategy/sweep.py` + `model_strategy/confirm.py`), `sportstradamus.stats`
(feature columns), `data/config/{stat_meta.json, ship_config.json}`, `tests/golden/`.
Prediction-side edits only via §7.3.

WS-1 (live alignment, §6.10) footprint: `strategies/kelly.py` (selection-aware sizing),
`strategies/profit_sim.py` (replay harness — validate before any code change),
`data/runtime/*.parquet` (**read-only** — the dashboard's snapshots), and
`training/graduation.py`. The **Gate-2 thresholds in `graduation.py` are owner-only** — WS-1
produces decision packets against them, never session edits.

## 6. Stage plan

### The four axes and the sweep + confirm engine

A served predictive is built in four independently-swappable stages,
`normalization → model/loss → blend → calibration`
(`target_normalization ⊥ {dist_training_loss, variance_reg} ⊥ {blending_loss_fn, fused_loc/BLP, book-recovery} ⊥ {posthoc, dispersion_cal, skew_cal}`):

| Axis | Values | Executable today | Unbuilt |
|---|---|---|---|
| **Normalization** (retrain) | `ratio_meanyr`, `centered_additive_mean10`, `centered_additive_eb_meanyr_k10`, `ratio_projvol` | 3 of 4 carry a Gate-4 SkewNormal decode (`scorecard._decode_sn_loc_scale`; EB off the dumped `GlobalMean`) | `ratio_projvol` refuted → §6.9 count offset (§6.2) |
| **Model/loss** (retrain) | dist-loss `nll`/`crps`; variance / soft-cal regularizer; calibration-constrained HP selection; σ-head stabilization | dist-loss via `--dist-training-loss`; HP selection via `--hpo-selection calibrated`; stabilization via `--stabilization MAD\|L2` (§6.1) | in-training PIT regularizer / decoupled-σ fit (§6.5) |
| **Blend** (retrain weight + research-gated structure) | blend-loss `nll`/`crps`; `fused_loc` pool; book recovery; BLP wrapper; p_book noise | blend-loss via `--blending-loss-fn` (`fit_model_weight_crps` built); current `fused_loc` pool | density-LOP fix, power de-vig, book recovery, p_book noise, free post-hoc `w`-refit (§6.5); **BLP + decoupled blend probed NO-GO** (§3.4, §10 2026-06-26) |
| **Calibration** (auto-fit, not searched) | location (`roe_mean`/`isotonic_mean`); scale+shape (`dispersion_cal` + joint `skew_cal`; count PIT-KS retarget); full-CDF (isotonic-PIT / IDR) | location + scale+shape shipped; count PIT-KS retarget via `--count-dispersion-objective pit_ks` (§6.1 Rung B′); whole-CDF isotonic-PIT recal shipped on **continuous** cells (§6.1 Rung C — PA ships, `[[rung_c_whole_cdf_recal]]`) | Rung-C-on-**count** dead (`[[rung_c_whole_cdf_recal]]`); IDR variant unbuilt |

**The sweep + confirm engine.** One entry point on devel:
`model-strategy-sweep` (`training/model_strategy/sweep.py`, with
`training/model_strategy/confirm.py` for the ship half). It replaces the retired
`model_strategy_driver.py` / `model-research`-branch split — everything is on devel now.

- Board mode (omit `--market`) enumerates every registered `StrategySpec` whose exact-cell
  enrollment, cached-matrix schema, distribution-class applicability, and
  `deterministic_train`/`score` capabilities admit the cell. Base distribution families and structural heads use the same registry contract; there
  is no NFL-yard special branch. Results upsert into
  `data/research/strategy_research_board.csv` after each cell, and `--include-shipped` still routes
  live cells to supersession rather than fresh ship.
- Every row carries the exact strategy slug/family and generic `structural_strategy` selector,
  canonical spec signature, implementation/artifact-schema versions, status and cell, sorted
  compact `controls_json`, cached-matrix SHA, corner fingerprint, and structural split fingerprint
  where declared. The corner fingerprint SHA binds
  `{strategy signature, family, exact controls, matrix SHA}`. The split fingerprint is a separate
  exact identity from the structural artifact's validation audit; it is deliberately **not** folded
  into the pre-training corner fingerprint. `--resume` admits prior rows **per row** — a budgeted
  search never enumerates a whole cell — rejecting any with stale specs, controls, matrix/schema
  identity, a missing/unexpected structural split contract, or an `eval_split` that is not the
  cross-fit marker. It does not independently derive a future structural split; full-HPO confirm
  separately exact-compares that split across board, pickle, CSV, and `model_stats`.
- Per cell the sweep is **one budgeted conditional-TPE study** over every applicable spec's
  registered controls: `family` is a categorical, each family's axes are suggested under it as
  namespaced `family__axis` parameters, and `TPESampler(multivariate=True, group=True)` is the only
  Optuna 3.5 configuration that models that conditionality rather than falling back to independent
  sampling. `MAX_TRIALS_PER_CELL = 48` against a ~312-corner declared grid, so the enqueued
  incumbent and seed corners matter as much as the sampler. A per-cell journal file makes the study
  itself resumable. Each corner trains one
  `meditate --deterministic --bypass-withholding --holdout-blind` into a sandbox
  (`research/models/deterministic/` + `data/test_sets/deterministic/`) so a trial never clobbers a
  production market. `--deterministic` pins RNGs + fast fixed HPs.
- Each corner is scored **holdout-blind**: `--holdout-blind` drops the ship gate's rows from the run
  entirely and cross-fits the calibration head out-of-fold, player-disjoint, over the validation
  frame, so selection never adapts against the rows the ship decision uses. The ranker calls
  `scorecard.gate_row` on the dump, the same code production ships on (no test re-fit; that oversold
  the screen and was removed in `10306ee`). Rows carry `eval_split = crossfit_validation`; the older
  holdout-scored rows carry `<NA>` and are inert to both the resume cache and confirm's nomination
  lane. The objective is negative **min-gate slack**: one scalar, positive iff
  the corner ships, larger with more headroom across all six gates — "ships, with margin," not
  Gate 4 alone.
- Structural methods are **not searchable** by this sweep: they derive role support from validation
  row counts, so `train_market` refuses them under `--holdout-blind` rather than score folds that
  disagree about whether the method fell back. They stay selectable in production through the
  `posthoc` pool and keep their own preregistered evidence path.
- `--confirm` requires the selected spec to expose `confirm`, `full_hpo`, `score`, and `serve`; a
  research-only family such as Mixture cannot slip through. It rejects noncanonical controls or
  stale corner/matrix/split identity, persists the exact controls and structural selector, and
  launches full HPO through `--structural-strategy auto`. It then exact-compares the pickle, test
  CSV, and `model_stats.parquet` identities before accepting the official scorecard verdict. An
  all-six-gate pass keeps `withheld → devel`; any failure auto-reverts the cell's config and
  artifacts. `--yes` runs it unattended.

**Registered-corner reproducibility rule.** Every swept control, including
`dist_training_loss`, either persists in `stat_meta.json` or is forwarded explicitly to full HPO.
The old ranks-only exception and top-K workaround were deleted by S4. A board leader is eligible
for confirm only when its registered exact corner can be reproduced and identity-checked
end-to-end.

**Generic selector and artifact identity.** `meditate --structural-strategy` accepts
`none|auto|<registered structural slug>`: `auto` honors the cell's persisted selector, while an
explicit slug is deterministic-only and must match its registered cell and fixed controls.
The pickle stores `model_strategy`. CSV identity is constant in `ModelStrategy`,
`StructuralStrategy`, `StrategySignature`, `StrategyImplementationVersion`,
`StrategyArtifactSchemaVersion`, `StrategyStatus`, `StrategyLeague`, `StrategyMarket`,
`StrategyControlsJSON`, `StrategyCornerFingerprint`, `StrategyMatrixHash`, and
`StrategySplitFingerprint`. `model_stats` repeats the contract as `strategy_slug`,
`structural_strategy`, `strategy_signature`, `strategy_implementation_version`,
`artifact_schema_version`, `strategy_status`, `strategy_controls_json`,
`strategy_corner_fingerprint`, `strategy_matrix_hash`, and `strategy_split_fingerprint` beside
its normal `league`/`market`. The scorecard rejects malformed or adapter-inconsistent CSV identity.
Serving validates active status, `serve` capability, exact cell, distribution class and canonical
controls, the pickle's actual distribution and `expected_columns`, plus corner/matrix/split
identity; a structural adapter dispatches only from that validated selector. Identity-absent
base-family legacy artifacts remain compatible, but a legacy structural blob without generic
identity fails closed. The signed recipe is also checked against every independently persisted
runtime field (distribution, normalization/normalized semantics, SN parameterization, ZINB mode,
and fixed posthoc settings). Live supersession requires the fresh candidate-bound `model_stats`
row before S1/S2/S3, and S2/S3 pair exact shared events by Player + UTC Date + Line + Result
rather than row number; missing, mismatched, or ambiguous identities hold the incumbent.

**Scoped board-wiring verification (not lifecycle evidence).** The combined
sweep/confirm/pipeline/report/scorecard/serving regression suites passed **431 tests** after the
mandatory refactor and duplicate-code repair; the final line-cap extraction passed its 145-test
affected suite. Real deterministic registry paths then returned
receiving yards SHIP 6/6 (slack `+0.052371`, n=2,238) and rushing yards SHIP 6/6 (slack
`+0.036703`, n=1,009).
Receiving model/CSV SHA-256 were
`50c94a77d4f0eca24c6142dc270633429d191670392e670c5665493f1a361153` /
`fb9da6af6f67557fad00a0a2aa8d1537f75ab7f382b202c7c02bd0ef45406570`; rushing were
`9a9c8f333a86b4e07a01da3349c89d68e97d7350dcf451e43fe6aa7ba8215740` /
`696e2a78b6d7454156a4db1b988f9919b7f3eb313c926496bfffc446cc95752d`.
Both pairs exactly matched canonical controls plus matrix, corner, and separate split identity;
the pre/post production audit stayed byte-identical for `stat_meta.json`
`5bcb1f46fc0fe4717e1f509f87c83b556e63c049e89127b45197ac34b4a3dff8`,
`stat_calibration.json` `5bbb14b41e5a63522e24c0455055067a42b636ad6bfd7f99e9b37de7904cfffe`,
`book_weights.json` `1ca7a3a182762ab827101b68d9b3701cb95495306ee62f30847346e334162926`,
production `models/NFL_rushing-yards.mdl`
`c9557e5a30f5909ceb1042b3544172b0831f5891a687c0fcbcd66e8310d433d1`, and production test CSVs
receiving `027cf0387171ea8d0d7a631e7d136ed52d68d3016417060cc324ec94f3cb454b` / rushing
`798cb685b1dbf09b2f10837d6e9c5052c31ee1c7cc3b3dd2c77a46e7b07e6738`. No production receiving
model existed in the audited path, so none is claimed. No confirm or `stat_meta` persistence ran.
These are scoped wiring smokes, not another full-HPO claim, enrollment, or ship-state change;
the authoritative Ruff gate passed. The one full golden run finished 3,756 passed / 1 expected
xfail / 2 failed: its task-owned R0801 failure was repaired and the focused duplicate gate then
passed, while `test_ship_gate_invariant` remains red on the unchanged runtime-recomputed
`model_stats.parquet` (SHA-256
`e66f139871e5d8524dc99abe1de36b280d241bafd04199671e7ac8f414e0903a`) because eleven previously
shipped cells currently report a failed gate. The one full fake integration run finished 28/30;
both failures were fixture-contract defects (missing synthetic `hist_gate`, plus an MLB fake-mode
network leak), were fixed without production-code changes, and passed directly 2/2 afterward.
The full suites were not rerun; the stale prior `.claude/.state/integration_green` marker was
removed, so the repository does not advertise a green integration state.

**Deterministic ranks, real-HPO ships — the law.** A `ships=True` board row is a candidate flag,
never a ship; the val→test discount tips knife-edge cells both ways, so the full-HPO confirm is
mandatory and a deterministic score is never shipped.

**League activation guard.** `--confirm` **never** auto-flips a withheld MLB or NHL cell — those
ship only after their owner gate (D1/D2, §6.7); a board-passing withheld cell in a gated league
is announced and skipped (`model_strategy.confirm._drop_activation_gated`). Already-live cells in
those leagues still supersession-test (a strategy swap never changes the release surface).

**Shipped cell with a better corner → supersede, not auto-ship.** `--include-shipped` routes a
shipped cell through the higher §7.1 bar (S1+S2+S3), never the fresh-ship flip.

**Nothing is deferred. Every withheld cell and every lever is a live candidate.** The old
`deferred-90` / lever-cap tags are retired — they were per-axis verdicts under a single
normalization, and the honest sweep showed such cells ship under a *different* corner. A cell
leaves the board **only** on matrix-wide exhaustion (§8) or the operator's documented denominator
call.

**Stage-0 hardened the engine (owner requirement — the sweep was a bottleneck run by hand).**
Landed: the declarative `StrategySpec` registry in `model_strategy/specs.py` /
`model_strategy/registry.py` (base families and cell-scoped structural heads, capabilities,
controls, persistence, selector, required columns, and artifact paths); exact cached-matrix schema
and streamed-SHA scoping; signature/matrix/split-aware resumable board runs; `--dry-run` scope
preview; and **model-version stamping**
(`model_version = yyyymmdd.norm-slug.sha8`, train→serve→history→dashboard, plus
`scripts/backfill_history_eras.py` for legacy rows — the §6.10 era-attribution spine). Family is a
swept registry strategy today. Still deferred: the confirm **queue manifest** (ETA + outcome
journal; per-cell logs in `research/logs/confirm/` cover today's need) and automatic archive
**snapshot copy** per confirm run (a manual `cp` + `SPORTSTRADAMUS_ARCHIVE_DB` is the current
recipe).
**Research-gate still binds** a family/dispersion-mechanism value
(`research-analyst` brief first, §8.2); a plain knob (normalization slug, loss choice) does not.
And **wiring an axis-value ≠ it sweeps** — a value can sit in the grid yet not sweep until its
machinery exists (`blending_loss_fn` carried `crps` before `fit_model_weight_crps` was built).

### The operating loop — three manual residues on top of the engine

`--confirm` automates the persist → full-HPO → keep-on-all-six-gates / auto-revert loop. Three judgment
calls the automation cannot fully encode still need an operator:

1. **Near-miss walks.** Confirm the reproducible top-K (2–3), including corners that *fail* the
   deterministic gate by ≤3% of threshold (`min_gate_slack ≥ −0.03`): the val→test discount tips
   knife-edge cells in *either* direction (a board passer can fail HPO; a board near-miss can
   clear it), and the board's top row isn't always the one that survives HPO (NBA DREB — `eb`
   topped the board, `mean10` is what shipped). List exhausted ⇒ route the cell to the calibration
   ladder (§6.1) and record the axis attempt (§8).

2. **Calibrated-first `hpo_selection` policy (a confirm-time axis the board cannot see).** The
   deterministic study fixes one HP set, so it never runs the search `--hpo-selection calibrated`
   gates on blended OOF PIT-KS (§6.1 Lever 1); the selection policy is orthogonal to the board's
   normalization pick and decided only here. **Confirm every SkewNormal candidate under
   `calibrated` first** — it picks the sharpest trial that *clears* Gate-4, so it weakly dominates
   `loss` on g4 at a small g1-sharpness cost; a clean six-gate pass ships with
   `hpo_selection: "calibrated"`
   persisted in `stat_meta.json` so the plain cron reproduces the calibrated trial instead of
   retraining to the sharper, g4-failing one. **If calibrated does not ship** (its wider σ tips g1,
   or no trial clears g4 and the logged fallback fires), **re-confirm under the default `loss`**
   and ship that if it clears, leaving `hpo_selection` unset so the cell rides the production
   default. Neither clears ⇒ route to §6.1 Rung C / §6.6. (Worked example — the PA/PR/DREB
   re-confirm: PR shipped *only* under calibrated, g4 0.0516 → 0.0423, and carries the persisted
   field; PA was a calibrated no-op — fallback fired, g4 ≈ 0.058 either way — and DREB a 0.0508
   near-miss, both held.) The loss→calibrated direction is now automated in the walk: a nominee
   failing ship on a g4-only near-miss is retried once under `calibrated` and a shipping retry
   persists the pin (`confirm._retrain_with_calibrated_retry`); the calibrated-*first* doctrine
   above stays manual — automating it would double every SkewNormal nominee's cost. The weekly
   `meditate` never retries; it trains each cell once with the persisted knob. **The Experiment-2
   sunset path is dead:** the calibrated-everywhere default flip was killed by its pre-registered
   criteria (§6.1 Lever 1 — 1/4 failing cells improved, MLB g1 breach), so the retry machinery
   and this two-attempt doctrine stay. What changed instead: v3 is now the implementation behind
   every `calibrated` pin (v2's hinge is deleted), and its SkewNormal measure scores the
   book-blended predictive on both the OOF and validation frames, so low-`model_weight` pins
   are safe to retry into.

3. **Supersede shipped cells (S1+S2+S3).** `--include-shipped` routes a shipped cell through the
   higher §7.1 bar — a shipped cell may have a better corner than the scale-only default it
   settled for, but a strategy swap on a live cell needs the incumbent-supersede evidence, not the
   fresh-ship flip.

Exploration runs always use `meditate --deterministic` (sandboxed writes) with `--market`
scoping; full-league retrains are expensive — don't run one to answer a one-cell question. Every
target cell that is withheld needs `--bypass-withholding` or the run silently skips it.

### Workstream priority (payoff-to-effort — the order work is pulled)

Work is pulled by payoff-to-effort, not section number. The stage subsections below hold the
mechanics; this table is the priority overlay.

| WS | Name | Payoff/Effort | Home |
|---|---|---|---|
| WS-1 | **Live alignment** (era-aware profitability, benchmark reconciliation, selection/sizing) | Highest / M | §6.10 |
| WS-0 | Standing breadth harvest (monthly registry sweep + confirm, free-passer, near-miss walks) | High / Low | §6.0 |
| WS-2 | MLB (now) + NHL (by Sep) activation | High / M | §6.7 |
| WS-3 | Family escalation — count wall (Double Poisson) + shape-bound (centered-SN / SHASH) | High / High | §6.6 |
| WS-4 | Ladder/tail + Rivals difference-pricer + D3 | High / Low-M | §6.11 |
| WS-5 | Targeted features (M-1), narrowed | Med / M-High | §6.3 |

WS-1 is Priority 1 (§4, 2026-07-07). Breadth (WS-0/WS-3) stays a standing harvest feeding D3/D5;
it is an input, not the goal (§1).

### §6.0 WS-0 — Standing breadth harvest & bookkeeping (automated, high-payoff, low-effort)

The breadth engine, run on a cadence rather than a campaign. **Cadence:** a registry-driven
`model-strategy-sweep --confirm` monthly and after any lever or data change, plus the
free-passer sweep below; NFL candidates first (the binding league, §6.9). **Scope guard:** a
running sweep OWNS `stat_meta.json` — never `meditate` or hand-edit it while one runs (the §3
concurrency probe). **Counts:** always re-derive from the §3 board-rollup and shipped-counts
blocks; this section never restates them. Ships flow through the deterministic-ranks / real-HPO-
ships law and the devel-ship-curator PR.

Entry: none — this recurs.

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
The two one-time items that used to live here are resolved: the strategy engine is on devel (no
more branch asymmetry), and the NFL g1×dispersion question is answered — NFL volume cells are
feature-mature, their negative BSS is target-shape not dispersion, so they route to §6.6/§6.1
(`[[nfl_volume_cells_feature_mature]]`), not to a scale-fit side-effect.

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
  byte-for-byte. *λ-shrinkage refinement — NULL (`/tmp/researcher_lever1_v3.md` R5, Exp-4
  replay over 79 cells):* cross-fit λ-shrinking the fitted `(c, s)` toward `(1, 0)` missed its
  ship bar (median held-out KS −0.0001 vs the −0.0005 bar; worst degrade +0.0058 vs the 0.002
  cap; CvM-fit arm no better). The replay also measured the unshrunk fit itself as net-neutral
  at median (+0.0002 held-out) — it pays only where real miscalibration exists (WNBA FGA
  0.065→0.028) and taxes slightly elsewhere, which is exactly what the
  `_DISPERSION_SKEW_MIN_GAIN` hard gate protects: keep the gate, no λ machinery.
- **Rung B′ — re-target the count objective (built, opt-in; default-flip candidate,
  Experiment-3 gated; kickoff brief: [count_dispersion_flip.md](count_dispersion_flip.md)).**
  The count-branch `dispersion_cal`
  minimized CRPS, but the gate is PIT-KS — re-targeting the fit to PIT-KS is the one change
  that tightens the whole over-wide count branch (snapshot: all 21 ZINB/NegBin cells
  over-covered; the 2026-08 confirm-ledger read sharpens this: 26 of 30 over-wide count rows
  sit on the `crps` objective, so the flip to `pit_ks`-by-default is Experiment-3 gated — ship
  iff ≥2 of 3 pilot cells improve g4 ≥0.010 with no control losing `ship` and the g6 over-leg
  flat). Minimizing CRPS does not guarantee a calibrated PIT. Built as
  [`pipeline._dispersion_pit_ks_loss`](../../src/sportstradamus/training/pipeline.py) (mirrors
  `_dispersion_crps_loss`, swaps the objective for `scorecard._randomized_pit_ks`), selected by
  `meditate --count-dispersion-objective pit_ks`; default `crps` keeps production. The
  `0.01·log(c)²` brake is retained as the only sharpness guard — a pure PIT-KS objective has none
  (CRPS implicitly penalizes over-confidence), so watch the over-leg of g6 and the g1 acceptance
  on count cells (§8.2 open-Q). The count miscalibration is
  *uniform* (over-wide on both central-50 and central-80, 7/7), exactly the regime a single
  scale clears.
- **Rung C — full CDF (isotonic-PIT / IDR, BUILT & shipped on continuous cells; dead on count) —
  the recommended fix for the mixed-direction SkewNormal cohort.** Shipped: WNBA PA rides the
  whole-CDF isotonic-PIT recal; on count cells the monotone map degrades the low-mean lattice, so
  count is a confirmed dead end there (`[[rung_c_whole_cdf_recal]]`) and its residual routes to
  §6.6. The scalar `(c, s)` is the bottom of
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
  **Extension kill (`/tmp/researcher_lever1_v3.md` R7):** μ-conditional and
  higher-DOF post-hoc are refuted as defaults — the μ-conditional dispersion defect is real
  (tercile `c*` 1.082/1.031/0.923 vs a 1.04/1.01/1.00 null; lattice artifact ruled out) but the
  one-parameter μ-power fix does not transfer (median held-out −0.0001 KS, wins 13/30). No
  mean-bucketed isotonic, no IDR-on-μ, no Rung C beyond SkewNormal. Per-cell exceptions only:
  NFL receiving yards (−0.0099) and NBA FGA (−0.0036). Revisit only if the *served* fit moves to
  the OOF frame (n≈5000+, the one path that changes the variance regime). Sequential
  `(c, s)` → Rung C is likewise null: the WNBA FGA half-split pilot scored −0.0001 held-out KS
  vs recal-only (kill bar −0.005) even though the joint fit finds a large defect
  (c ≈ 0.75, s ≈ 2.9) — the isotonic map alone absorbs it, so CDF_STAGE's `(c, s)` bypass
  stands as designed.

**Training-time calibration levers (Lever 1 selection + Lever 4 stabilization, built, opt-in).**
The post-hoc rungs above polish a *fixed* trained σ-head; these two change which head you get,
upstream of the polish. Both default to current production.

- **Lever 1 — calibration-constrained HP *search* (`--hpo-selection calibrated`; SkewNormal +
  count cells; v3 2026-08-23).** Why full HPO often *fails* g4 where the deliberately under-fit
  deterministic run passes: μ and σ share one LightGBMLSS boosting process and one HP set, so the
  standard knobs move sharpness and calibration *together*; full HPO walks to the CRPS minimum
  (the sharpest σ surface), not the calibrated one, while the under-fit run's near-constant σ ≈
  the marginal residual SD is homoscedastically calibrated "for free." The fix is the *selection
  mechanism*, not the loss — v3 (brief `/tmp/researcher_lever1_v3.md`): the Optuna objective is
  the **raw CV loss** and each trial's served PIT-KS — measured **out-of-fold** from the fold
  boosters `model.cv` already trains (`pipeline.predict_cv_oof_params`; explicit contiguous
  folds; no per-trial refit; ~4.7× the validation rows ⇒ 2.16× tighter KS) — reaches the sampler
  as a **feasibility constraint** (`TPESampler(constraints_func=…)`, optuna 4.9.0, Deb rule).
  This retires the v2 `CRPS + 10·max(0, KS − τ)` hinge, which was lexicographic whenever
  feasibility was scarce (9/31 real searches had <5% feasible trials — the study degenerated to a
  pure PIT-KS minimizer). Final pick: lowest CV loss under the 1-SE effective threshold
  `τ_eff = max(τ, min_KS + 0.2606/√n)` — the old zero-feasible fallback (bare min-KS, the most
  winner's-cursed of ~300 noisy draws) is retired (Gneiting-Balabdaoui-Raftery 2007 "sharpness
  subject to calibration"; ESL §7.10 one-SE rule). Built across
  [`hyperparams.run_hyper_opt(calibration_penalty=…)`](../../src/sportstradamus/training/hyperparams.py)
  and the per-trial evaluator
  [`pipeline._calibration_penalty`](../../src/sportstradamus/training/pipeline.py). The measure
  fits the gate's joint `(c, s)` (the v2 c-only closure's `s = 0` blindfold was worth 0.03–0.04
  KS on the NFL SN cells), and the SkewNormal statistic is the **max over two book-blended
  frames** — the OOF pool and the validation split, the latter predicted refit-free by averaging
  the fold boosters' raw margins (`pipeline.predict_cv_ensemble_params`;
  `pipeline._blend_oof_skewnormal` mirrors `_fuse_skewnormal` on both). Both legs are mandatory,
  not optional fidelity, and each was forced by a live failure mode Exp-2 exposed on MLB hits
  allowed (model_weight ≈ 0.12): a **model-only** measure Goodharts on book-dominated cells (the
  static ladder's ≤0.004 blend gap on *fixed* artifacts inverted under search pressure — 280
  trials walked the served g4 from 0.021 to 0.153 while the measure kept improving), and a
  **blended but OOF-only** measure passes era-fragile trials (train-era recal-fixable, but the
  validation-era serve fit demanded c=1.80/s=−2.82 which collapsed to g4 0.104 on test — the R2
  era caveat realized, escalated to its pre-registered two-frame form). The count branch stays
  model-only OOF (its risk cohort is all SN); post-hoc threading stays NOT built. Cost ≈ the
  plain `loss` path (the +33% refit is gone; the validation leg is invisible at ~8 s/trial); a
  `calibrated` pin on a hurdle or Mixture cell is inert and now warns.

  **Exp-2 verdict (6 cells × 3 arms, frozen-matrix, 60-min wall each): default flip KILLED by
  its pre-registered criteria** — g4 improved on 1 of 4 failing cells (≥3 required), 1 flip to
  pass (≥2 required), and the MLB arm breached g1. v3 stays an **opt-in per-cell pin**, but it
  *replaces v2 as the implementation behind that pin* (the hinge is deleted) and it is the better
  opt-in where the model dominates: WNBA MIN ships under the final two-frame v3 (g4 0.0609 loss
  / 0.0542 v2 fail / 0.0464 v3 **ship**, +5 trials over loss), and both controls (NBA PTS, NHL
  points) held `ship` with equal-or-better trial counts. On MLB hits allowed the final measure
  is safe at low model_weight: g4 0.0371 vs the v2 incumbent's 0.0390 (the interim measures'
  0.104–0.153 catastrophes eliminated), with g1 +0.0024 — inside the ±0.003 replicate band the
  `loss` arm also sat in, so the pin no longer risks poisoning on retrain. NBA PF was flat
  across all three arms (~0.07 — over-wide NegBin, evidence for the Rung B′ / R4
  count-objective flip, not for selection). Side finding: v2's refit cost capped real searches
  at ~10–30 trials/hour, the historical reason the lever under-performed.

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

### §6.2 Stage 2 — Normalization axis (reference)

Entry: cell on the board.

The normalization axis of the sweep engine above. Normalization is **often decisive over
calibration**: the
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

The confirm-selection economics (only confirm what can change served state; skip a shipped cell
already on its board-best config; match the complete signed registered identity) are the engine's
`--confirm` semantics — see the sweep+confirm spec atop §6.

**Acceptance:** per the operating loop above — official all-six-gate pass on the full-HPO confirm
(withheld) or S1–S3 (incumbent). **If it fails:** the cell records the normalization axis as tried
(§8) and routes to features (§6.3) or blend (§6.5).

### §6.3 WS-5 — Targeted features (narrowed): quick wins + the pre-registered build

Entry: §6.0 parity harness green; WS-1 underway. Targets the cells that need *signal*, not width:
**WNBA STL and NBA PF are the primary pilots.** The NFL g1+g4 volume five are **feature-mature**
(485 candidate columns already carry game-script, interactions, multi-window recency), so their
negative BSS is target-shape not missing features (`[[nfl_volume_cells_feature_mature]]`); NFL
therefore gets **1–2 falsification pilots only**, and the rest of the volume cohort routes to
§6.6/§6.1. Gate-4 under-dispersion belongs to §6.1/§6.2, and feature work complements, never
replaces it.

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

### §6.4 Deferred feature backlog (batch 2 + the feature-count ablation)

Deferred behind WS-1/WS-3/WS-5 — this is the standing reference list of feature levers not yet
pulled, each reusing WS-5's regen + parity infrastructure. Pull one only when a cell's routing
(§6.9) points here and the higher-payoff workstreams are underway.

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

### §6.5 Stage 5 — Structural width fixes: blend rebuild (CLOSED this cohort — research-gated to reopen)

Entry: a cell where §6.1–§6.3 left the served predictive miscalibrated. **Structure changes here
alter the dispersion mechanism → `research-analyst` brief first (§8.2).**

**Status — both halves probed NO-GO for the current SkewNormal cohort; the blend is not the lever
at `w≈0.90`.** The full narrative and citations live in the references doc (Ship-75 sweep-era
verdicts) and memory; the durable conclusions:

- **Built + committed:** power (logarithmic) de-vig (favourite-longshot bias on asymmetric /
  anytime-TD lines, Clarke-Kovalchik-Ingram 2017) and the count-tail regularizer `_sanitize_book_ev`
  (caps `|μ_book − μ̂| ≤ K·SD` as p→1).
- **Fitting half (book-shape rebuild) — REFUTED.** The shape-borrow book-skew mechanism was built,
  A/B-refuted and reverted (marginal residual skew over-states the book's *conditional* skew;
  worsens g4/g2 even at w=0.9) — production books stay symmetric until the WS1 ladder accrues
  (`[[book_skew_shape_borrow_refuted]]`). A standalone book rebuild is decoupled from the served
  gate at `w≈0.90` and a context-conditioned `cv` scale law is 0% OOS (`[[book_distribution_audit_nogo]]`).
- **Pooling half (BLP + decoupled location/shape) — REFUTED this cohort** across two probes
  including an operator weight-challenge (low weights explored, 0/9 OOS gate wins) —
  `[[pooling_half_blp_nogo]]`. At `w≈0.90` the served predictive is already the model's and
  near-calibrated, so the beta wrapper has no decalibration to repair; over-wide cells are
  family/shape-bound → §6.6.
- The five `fused_loc` immaturities (parameter-blend ≠ density pool; symmetric book; linear-shrink
  skew; sharpen-only pool; no `p_book` noise) are documented in the references doc.

**Reopen only for a genuinely over-wide cohort at `w<0.9`** (post-§6.6, or NHL/MLB on activation),
re-probing per cohort — never generalize the NO-GO. **If reopened, the design is settled:** fit
`(w, α, β)` by **CRPS (+ a PIT-KS hinge), never PIT-KS** (PIT-KS overfits location for the decoupled
variant and generates the `[[deterministic_ab_g4_oversell]]` for the BLP); bounds `α,β∈[0.5,3]`;
**ship guard = the raw linear pool `α=β=1` must sit inside the CI** (the fitted beta must strictly
beat the un-warped mix OOS, else collapse to `α=β=1`, which must itself beat the scalar blend); XOR
with §6.1 Rung C per cell (both are monotone CDF maps — stacking double-fits the held-out PIT). Do
**not** substitute a raw *linear* pool (disagreement-driven widening degrades the KS/ECE gates —
Gneiting & Ranjan 2013), and **co-fit** the book-shape choice with the BLP / learned `w` on one
held-out objective (Ranjan & Gneiting 2010 Thm 1 — never sequence the two halves).

**Model/loss adjunct (research-gated, deferred behind §6.1 Lever 1 + Rung C).** The training-time
variance / soft-calibration regularizer (MMD-to-uniform-PIT, Chung 2021; `CRPS + λ·CumKL(PIT‖U)`,
Utpala & Rai 2020) is untried, not refuted — but Dheur 2023 finds in-training regularization does
not beat *post-hoc* on calibration itself, so the free moves (Lever-1 selection, whole-CDF Rung C)
come first. It is the genuine escalation only for a cell needing a properly-curved σ surface no
first-order CRPS trial can produce (LightGBMLSS's CRPS path sets the Hessian to 1).

**Acceptance:** `pit_ks` below threshold with g1 BSS drop ≤ 0.01 and g5 < 0.075, on validation,
before ship; an inference-path round-trip test for every new served object (§7.3). **If it fails:**
signal-starved → features (§6.3); genuine heavy tail → family (§6.6).

### §6.6 WS-3 — Family escalation: count wall + shape-bound (DPO live — 4 ships: WNBA TOV, NHL points/blocked/shots; centered-SN pilots next; shape-bound kickoff brief: [shape_bound_triage.md](shape_bound_triage.md))

Entry: a cell whose cheaper axes are recorded-tried (§8), or a whole cohort the sweep leaves at
the family ceiling. **This is the main funded build.** The two research questions are answered —
briefs `/tmp/researcher_count_family.md` (R1) and `/tmp/researcher_continuous_family.md` (R2) —
so this section carries the *verdicts*, not options. Each family is built against the Stage-0
distribution scaffold, then **swept via the registry over its residual cohort** (a family is a
grid axis, not a manual per-cell edit); pilots first, then registry entry → board run. Standing
research-gate still binds each new family (§8.2) — the R1/R2 briefs discharge it for the two
below; a third family needs its own.

**Build status — both families are landed code; the remaining work is verdicts.** The generic
registry carries SkewNormal, Mixture, ZINB, NegBin, and DPO base strategies (count cells sweep
8+4+4 corners; SkewNormal cells 24 via the `sn_param` axis),
`meditate --dist [auto|DPO|NegBin|ZINB|SkewNormal]` + `--sn-param [auto|direct|centered]`,
the exact-normalized **Double Poisson** (`double_poisson.py` + a torch-free numpy serve kernel;
the inverse-CDF `sample()` is mandatory — LightGBMLSS predict draws samples even for
`pred_type="parameters"`), the **centered-SN class** (`skew_normal_centered.py`; as-built γ1
response `0.99·tanh`, not R2's 0.9952 — at that rail the α radicand is ~4e-5, float32
cancellation exploded hessians and killed boosting in ~4 rounds; stabilization defaults L2 for
the same reason), DPO wiring end-to-end (pipeline / scorecard `DP_MU`/`DP_PHI` contract /
model_prob + persist `Model Phi` / analysis / dashboard deep-dive; `fit_dpo_weight` lives in
`training/calibration.py`), `sn_param` persisted per-cell (pickle + stat_meta + serve-side seed
read with a call-site pin — the offset_mode drift class), cross-family confirm, and live-path
integration tests for both families. Family-scoped integration checks passed at that milestone;
this section makes no claim about the campaign's still-pending authoritative repository-wide
final gates. The **full-cohort screen**
(`count-family-screen --all-count`, 41 cells) routes **15 DP-mandatory (zero-deflated) / 8
hurdle-NB / 18 plain-NB** — the DP cohort is far larger than the 3 pilots. *Resume gotcha
retired:* `--resume` admits prior rows per row against the current spec signature, controls, matrix
SHA, structural split contract, and cross-fit `eval_split` marker, so a family/axis or input change
invalidates exactly the affected rows rather than the cell; confirm, not resume, exact-compares the
resulting split value. *Cohort-filter gotcha retired:* `--dist-class` now filters **families**, not
cells, and family admission is read from the cell's own target (integer lattice, mean ≤ 50) rather
than its meta `dist` — so integer-count cells wearing `SkewNormal` meta (NFL
receptions/targets/carries, NBA FGA, NHL sogBS) are searched by the count families directly, with no
pre-sweep `dist` flip. A count winner persists `target_normalization=none` via `fixed_persist`.
Refs:
`[[ws3_phase0_count_screen_verdict]]`, `[[deterministic_ab_g4_oversell]]`,
`[[data_dir_under_package]]`, `[[breadth75_campaign]]`.

**Continuous — centered-parametrization SkewNormal (R2 verdict; ~2 sessions, zero serving delta).**
The headline diagnosis: the trained α-head is **frozen at ~0 on every SkewNormal cell** — raw
`SN_Alpha ≈ 0` after subtracting `skew_cal`, which **rails at the ±3 clamp on 9–12/12 corners** of
the yards/fantasy family. That is the α=0 Fisher-information singularity (Arellano-Valle & Azzalini
2008; Hallin & Ley 2014) live in production; the centered parametrization is the designed fix. New
distribution class with a closed-form (mean, sd, γ1)→(ξ, ω, α) map, γ1 response `0.99·tanh`
(as-built; see Build status for the 0.9952 float32 deviation),
centered seed, per-cell `sn_param` knob; `predict_dist` re-emits direct `loc/scale/alpha` so
`model_prob`/`fused_loc`/`get_ev`/scorecard are untouched (§7.3 "training-only" — byte-identical
round-trip test only). Build order: **(0) re-baseline the 5 decode-drift `model_stats` rows first**
(NFL receptions/targets, WNBA DREB/PA, NBA DREB — g4≈0.46–0.50 are decode-drift artifacts, not
model failures); (1) centered-SN + **pilots WNBA PA and NFL receiving-yards** (receptions rerouted to the
count board via the meta-dist input — see the second engine gotcha above; expect PA KS 0.088→0.03–
0.05); (2) riders on NBA FGA + NFL receiving/rushing-yards (retrains only); (3) **SHASH gated on
pilot evidence** (3–4 sessions, hand-rolled on the `skew_normal.py` pattern, n≳1000 cells) — the
**WNBA DREB kurtosis class** (+10.4 z) routes to SHASH, *not* a centered pilot; NFL targets (pure
kurtosis) to StudentT/SHASH after g5/g6. **skew-t deferred indefinitely** (density embeds t-CDF;
torch lacks `betainc`; SHASH δ<1 covers the regime). **Kill (per cell, after calibrated-HPO
confirm):** val PIT-KS gain <15% vs incumbent, or γ1(x) degenerate, or g1 BSS drop >0.01 / g5 ≥0.075
= no-ship; 0/2 pilots ⇒ cohort → SHASH. Stacking: centered-SN supersedes `skew_cal`'s role but the
joint (c, s) stays as residual polisher; never carry a Rung-C blob across a family swap; per cell max
= parametric (c,s) **or** one monotone CDF map, never both (enforced pipeline.py:1730).

**Count — exact-normalized Double Poisson (R1 verdict; ~3 build sessions + 1 pilot).** A repo-local
custom LightGBMLSS distribution mirroring `skew_normal.py` (torch `Distribution` + ~70-line
`DistributionClass`; autograd supplies grad/hess; no lightgbmlss fork). Natively **mean-parametrized**
(µ trees move only the mean — the property whose absence broke joint-ZINB) with **both dispersion
directions unbounded**; verified in-venv (normalizer = 1.000000, finite gradients, V/M 0.36–1.45 at
target means; the µ-vs-true-mean gap neutralized by exact-series mean decode + Newton inversion).
**Key finding — ZINB is a confirmed misroute on all 20 count cells:** `_data_driven_dist` escalates
on raw zero-rate >0.02 but measured zeros ≈ NB-implied on every cell (four are zero-*deflated*), and
**plain-NB — a one-line stat_meta `dist` edit — has never been tried on any of the 20.** That is the
cheapest lever in the cohort, now a registry family axis. **Per-cell screen (pre-build):** a
non-asymptotic zero test under the NB null → route {no-modification → plain-NB; inflation → hurdle;
**deflation → DP mandatory** (gates can't subtract zeros)}; escalation bar = Dunn–Smyth RQR variance
<0.70 + Poisson-GBM-tracks-top-decile-while-NB-compresses; then fit {plain-NB, hurdle-NB}×{crps,
pit_ks} + the DP arm on honest val→test PIT, cheapest passing corner wins, DP must beat the best NB
corner without g1/g5 regression. **Pilot verdicts (dates: §10):** WNBA TOV = **first DPO ship** — all 4 DPO corners board-pass
(best slack +0.122) while every NB-family corner dies g4; full-HPO confirm 5/5 → devel (BSS
+0.042, dispersion_cal 1.014 — the φ head fits dispersion natively, nothing left to calibrate).
NBA PF = all-16-corner kill: DPO posts best-in-family coverage (0.663 vs hurdle 0.769 / plain-NB
0.692, target 0.50) but closes only 15–39% of the central-50 gap and loses PIT-KS to hurdle
(0.083 vs 0.066); with g5 co-failing, PF's binding constraint is not family — routes §6.1/§6.2.
NHL points (low-mean 0.5 stress) shipped 5/5 (BSS +0.012, pit_ks 0.0153 vs the 0.083 NB floor —
low-mean ceiling thesis validated). The pre-registered kill rule is dead; DPO stays in
`_CLASS_FAMILIES`. Lesson: screen labels are weak per-cell predictors — the ship came from the
plain-NB *control* while the DP-mandatory pilot killed; the sweep decides, not the screen.
**W1 count boards (NBA, NHL, MLB — dates: §10):** NBA 6 cells → 0 board ships (FTM's lone
passing corner, hurdle +0.02, reverted at full-HPO confirm on g4 alone; TOV −0.300 kill); BLST +
OREB shipped via direct full-HPO re-earn. NHL 6 cells → 3/3 confirms shipped: blocked (DPO,
BSS +0.024), powerPlayPoints (hurdle-ZINB, BSS +0.449), shots (DPO, BSS +0.012); goalsAgainst
and faceOffWins killed. MLB 9 cells → 8/8 confirms shipped (doubles/hits/runs/singles/walks/
hits+runs+rbi DPO; rbi/total-bases NegBin); home runs + stolen bases all-corner kills. W2
reroute shipped NHL sogBS (DPO); W5 direct re-earns shipped NHL assists, MLB batter strikeouts,
NBA STL (all DPO). WNBA count board deferred — league already over target. Family tally across
the campaign: **DPO 14 ships, NegBin 2, hurdle-ZINB 1, joint-ZINB 0**.
Runner-up mean-parametrized CMP and rejected GenPoisson/Gamma-count/Tweedie rationale is in the R1
brief.

**Past families — model-class escalation (all §8.2-gated, pilot evidence first).** LightGBMLSS
binds four ways: a pre-committed closed-form family per cell; weak per-row gradient to the
scale/shape heads (why g4 is the graveyard gate); Fisher pathologies at singular points (the α=0
saga); one loss and round schedule shared by all heads. When a cell exhausts families, go
nonparametric on the axis serving actually needs: (1) **ordinal threshold stack** for counts — K
binary `P(X≥k)` LightGBMs + isotonic de-crossing = a family-free conditional PMF; serve only
needs `P(X > line)` at half-integer lines, and blend/push/mean/sims all work off the PMF; first
pilot on the count residual (PF-class cells). (2) **Mixture head** for continuous — check the
pinned lightgbmlss for Mixture support *before* any SHASH packet; a 2-component normal covers
heavy tails + skew + bimodality (blowout/benching regimes no single-mode family represents) and
likely supersedes the SHASH ask. (3) **Comp-weighted empirical PMF** as the cheap diagnostic: the
KNN player-comp machinery already exists — if the comp-empirical distribution *also* fails
coverage, the cell's problem is conditioning, not family; stop escalating families and route
§6.2/§6.3. NGBoost skipped (natural gradients fix an optimization disease we already stabilized);
global multi-task pooling across cells is the horizon item, not this push.

**Machinery simplification track (owner-adopted; no-regression rule — nothing lands mid-sweep).**
S1 *evidence-gated*: drop the dominated joint-ZINB sweep axis once the W1 boards confirm it never
wins a cell (ZINB grid 8→4 corners; the mode stays supported for pinned cells). Evidence through
the NBA + NHL boards: 132 joint corners on the board, 0 shipping, best slack −0.166 — MLB/WNBA
boards are the last word before landing. S2: retire
`count_family_screen.py` to `src/deprecated/` post-campaign (verdict delivered; labels proved
weak per-cell predictors). S3 *guard*: a knob-contract golden asserting stat_meta knob ↔ pickle
filedict ↔ serve-side seed parity — the offset_mode / sn_param drift class, caught by hand twice.
S4 **BUILT** (2026-07-19, commit 4538861): `dist_training_loss` persists per cell (stat_meta
field + meditate per-cell knob resolution); every swept axis now ships, so the ranks-only-corner
concept and its confirm/sweep machinery are deleted. Unblocks NHL saves (+0.011) and MLB pitches
thrown (+0.052), both previously skipped twice. S5 *post-campaign*: consolidate the per-family serve branches
(~6 files; DPO needed 9 sites) into one dispatch table in `helpers/distributions` behind the
existing parity goldens — family #7 becomes one registry entry plus its torch class. The
torch/numpy dual kernels stay as designed (serve-graph isolation).

**NFL popular-market re-baseline (2026-07-20; evidence quarantined, no enrollment).** The old
“efficient-book kill” conclusion for passing yards / passing TDs and the Mixture-first route for
the yardage cells are superseded. All four popular markets now have fresh isolated full-HPO
artifacts and independent heldout six-gate passes. None of these results changes `shipped`, the
NFL numerator, or a live-test state.

- **Passing yards — isolated full-HPO + independent heldout PASS.** The exact recipe was SkewNormal,
  `ratio_meanyr`, distribution CRPS, blend NLL, loss-selected HPO, direct SN, no stabilization,
  and no posthoc. On n=369: g1 Brier delta −0.0010, row CI [−0.0051, 0.0032] and
  player-clustered CI-high 0.0033; g2 0.1206; g3 0.1211; g4 PIT-KS 0.0572 ≤ 0.0707 (tail
  0.0332); g5 −0.0075; g6 CITL 0.9904 (CI-high 1.0697). Pickle version
  `20260720.ratio_meanyr.e59505d4`, temperature 1.1540011, dispersion 0.9908284, model weight
  0.1093457. Quarantine: `/tmp/NFL_passing-yards.full-hpo-pass-20260720.mdl` SHA-256
  `6108de3a8aa90a90f34e4c0b13de7b003c6568129cf58a23c8ca183a1c9ca3bc`; scorecard
  `/tmp/passing-yards-fresh-scorecard.csv` SHA-256
  `f5d76a0fee5dc1229411d6f40382ee838baaf6bf4b9dbeb6c023ae4664c3a652`.

  ```bash
  sportstradamus meditate --league NFL --market "passing yards" --force \
    --bypass-withholding --dist SkewNormal --target-normalization ratio_meanyr \
    --dist-training-loss crps --blending-loss-fn nll --hpo-selection loss \
    --sn-param direct --stabilization None
  ```

- **Passing TDs — isolated full-HPO + independent heldout PASS.** The evidence recipe was DPO,
  legacy-labeled `ratio_meanyr`,
  distribution NLL, blend NLL, loss-selected HPO, count-dispersion objective `pit_ks`, direct
  invocation flag `--sn-param direct`, no stabilization, and **`roe_mean` posthoc** (affine a=0.2841855,
  b=0.8736016). Canonical `stat_meta` now records `target_normalization: none`. This is an identity
  cleanup, not a model change: the DPO count branch never applies the continuous target transform,
  so `none` and legacy `ratio_meanyr` both train/decode the same raw count labels. Goldens
  `test_dpo_count_path_target_normalization_is_probability_inert` and
  `test_dpo_distribution_selection_accepts_canonical_none_without_transform` in
  `tests/golden/test_pipeline_artifact_parity.py` pin bitwise-equal mean/dispersion arrays and
  equal DPO CDF/PMF probabilities. The command and version below retain their historical
  `ratio_meanyr` evidence identity. On n=373: g1 −0.0187, row CI [−0.0284, −0.0087] and
  clustered CI-high −0.0087;
  g2 0.1780; g3 0.1499; g4 0.0498 ≤ 0.0703 (tail 0.0187); g5 0.0439; g6 CITL 1.0479
  (CI-high 1.2394) and over ratio 0.9103 (CI-low 0.7901). Pickle version
  `20260720.ratio_meanyr.afa8f007`, temperature 1.3753360, dispersion 1.1951890, model weight
  0.6822488. Quarantine: `/tmp/NFL_passing-tds.full-hpo-pass-20260720.mdl` SHA-256
  `e00646834f2646ba2c304a863755608862396691df5c32302209f96d8242ab6b`; scorecard
  `/tmp/passing-tds-fresh-scorecard.csv` SHA-256
  `e8b16b4e05efc712f58d8f48749cf5a06f329d9bd37f3ad99012ef2b063232b2`.

  ```bash
  sportstradamus meditate --league NFL --market "passing tds" --force \
    --bypass-withholding --dist DPO --target-normalization ratio_meanyr \
    --dist-training-loss nll --blending-loss-fn nll --hpo-selection loss \
    --count-dispersion-objective pit_ks --sn-param direct --stabilization None
  ```

- **Rushing yards — isolated cold full-HPO + independent heldout PASS.** Experiment
  `rushing-qb-rb-affine-groupcdf-bookpool-v1` adds an affine marginal, separate QB/RB endpoint
  CDF maps, line-only temperature, and a coherent model/book probability pool to the existing
  SkewNormal `ratio_meanyr` / CRPS / blend-NLL / loss-selected / direct / no-stabilization /
  no-posthoc path. The player-grouped conditional-OOF calibration audit passed on n=1010
  (g1 −0.00315, row CI-high 0.000735, player CI-high 0.000566; PIT-KS 0.04586; QB 0.03583,
  RB 0.05384). The isolated run completed 95 trials at 1:00:27, best trial 53 / validation loss
  0.765541, version `20260720.ratio_meanyr.b45d9dae`; full-fit head
  `4.4653484193 + 1.0039169446 × marginal`, QB/RB lambdas 1/1, temperature 1.6410711827,
  pool weight 0.4211669563. Its untouched heldout then passed on n=1009: g1 −0.0024, row CI
  [−0.0061, 0.0011], clustered CI-high 0.0011; g2 0.1913; g3 0.1098; g4 0.0368 ≤ 0.05
  (tail 0.0367); g5 0.0272; g6 recent ratio 0.9309 (CI-high 0.9656) and CITL 0.9621
  (CI-high 1.0473).

  ```bash
  MPLCONFIGDIR=/tmp/mpl-rushing-confirm \
    poetry run python /tmp/run_rushing_full_hpo_confirm.py
  ```

  Evidence root `/tmp/rushing-full-hpo-confirm-k9md4fgm/`: confirmation SHA-256
  `2e229934d8e7314e0872d16d8e91aa3a2ba46244713ea9f98c8d1d326a800aee`, scorecard
  `a7d12f4fa6de30dacfd63f171ca968460403cef9ed2c9be447058c5e6e8c11bb`, model
  `47d9576b78a43623d0cf5c614f7b8d75e4159948b8355590d5f99ed37d345479`, test CSV
  `0dd6f0e730f5d76a5fa72be780207c17ed6997656d54d53419e9353778bb552a`. All ten protected
  production targets retained their content hashes/sizes. The strict fingerprint is false only
  because an external process rewrote `stat_calibration.json` with identical content but a new
  mtime; the restoration-audit SHA-256 is
  `324d754f0dbe86ba29350a38f6c1ba09a87250efb12625df55822c69f73e7da6`.

- **Receiving yards — isolated cold full-HPO + independent heldout PASS.** Experiment
  `receiving-role-position-two-part-groupcdf-fixedlinear-v3` uses two role boundary intercepts
  plus an RB residual, six role×position positive CDF maps, two role nonpositive maps, shared
  positive/nonpositive lambda selection, line-only temperature, exact endpoint-first settlement,
  and a policy-fixed authentic-quote pool `0.8 × book + 0.2 × model` (missing/synthetic quotes
  remain unpooled). The exact base recipe was SkewNormal, `ratio_meanyr`, distribution CRPS,
  blend NLL, loss-selected HPO, direct SN, no stabilization, no posthoc, and the frozen cached
  matrix. The 60-minute cold search used no initial parameters, completed 146 of 300 configured
  trials, and selected trial 72 at the best logged output loss 0.629173; the final pickle version
  is `20260720.ratio_meanyr.b9a6cba1` and its line temperature is 1.4702421.

  Before heldout scoring, nested Player GroupKFold on the canonical 2,239 rows / 382 players
  passed all ten named support guards and all 12 validation gate guards. The exact validation
  audit was: g1 −0.0003677, row CI [−0.0020198, 0.0013740], player-clustered CI
  [−0.0020539, 0.0013343]; g2 0.0927752; g3 0.2569047; g4 global 0.0432616, role full
  high/low 0.0325552/0.0590294, role-positive 0.0344159/0.0715984, position full WR/RB/TE
  0.0420076/0.0516557/0.0520554, and position-positive 0.0464157/0.0692140/0.0540941;
  g5 debiased ECE 0.0173928 and rate error 0.0038382; g6 recent correlation 0.5605996 and
  CITL ratio 0.9593831, CI [0.8978878, 1.0292329]. Fold lambdas (+/−) were (1/.5),
  (1/.5), (1/.25), (1/.25), (1/.75); fold temperatures ranged 1.4551961–1.4956771.

  Its untouched heldout then passed all six gates on n=2,238: g1 −0.0005, row CI
  [−0.0022, 0.0012] and player-clustered CI-high 0.0012; g2 0.0239; g3 0.1932; g4
  PIT-KS 0.0340 ≤ 0.05 (tail 0.0340; central-50/80 coverage 0.4960/0.7976); g5 debiased
  ECE 0.0159 (raw 0.0423); g6 recent correlation 0.5508 and CITL ratio 0.9670
  (CI-high 1.0240).

  ```bash
  MPLCONFIGDIR=/tmp/mpl-receiving-confirm \
    poetry run python /tmp/run_receiving_full_hpo_confirm.py
  ```

  Canonical split fingerprint
  `9589cedab5e06ad318bb4e7b4c6e4e6c75bc776e87a03e480c8f32836b49f10a`. Evidence root
  `/tmp/receiving-full-hpo-confirm-k4ipe86a/`: confirmation SHA-256
  `a83c6684b9f283572f0f7fc722069d604356b366d4f709cc2be9325362458afb`, scorecard
  `9abd76d3e8e18793de4e3499746e288db1de76ee5ab1f5a170746feab63d4660`, model
  `5706a153b77464ffcb00036bd122f4eb00c1a6194660ddfa8a97e8f165703607`, and test CSV
  `8e22c48c12ad9cd80194946428426c1868b0038e9c32217f3230296ff43160bb`. The harness redirected
  model/test/scorecard output into its isolated root, suppressed matrix/config/report/SHAP writes,
  and verified all ten protected production targets content- and metadata-identical before/after;
  restoration-audit SHA-256
  `4b2200cbfc8103f28f3d9f665574ece0bbff66156e2aed04e0b04f05cab8998b`. The prior direct-read
  result remains retired because 1,118 purported validation rows overlapped the heldout.

**Research rationale.** Proper scoring keeps selection tied to the served distribution, while
calibration is pursued without discarding sharpness and randomized PIT handles discrete TD
outcomes ([Gneiting & Raftery 2007](https://doi.org/10.1198/016214506000001437), [Gneiting,
Balabdaoui & Raftery 2007](https://doi.org/10.1111/j.1467-9868.2007.00587.x), [Brockwell
2007](https://doi.org/10.1016/j.spl.2007.02.008)). Isotonic/PIT recalibration and the receiving
two-part maps follow the recalibration literature, with fold-held fitting used to avoid
in-sample calibration claims ([Ranjan & Gneiting
2010](https://doi.org/10.1111/j.1467-9868.2009.00726.x), [Henzi, Ziegel & Gneiting
2021](https://doi.org/10.1111/rssb.12450)). Forecast combination motivates the constrained
book/model pools but does not replace the repository gates ([Bates & Granger
1969](https://doi.org/10.1057/jors.1969.103), [Gneiting & Ranjan
2013](https://doi.org/10.1214/13-EJS823)). The decisive acceptance evidence remains the untouched
six-gate heldout, not a favorable validation screen.

**Structured integer follow-on — research-only, owner-blocked.** An exact pure kernel now exists
in `src/sportstradamus/helpers/integer_distribution.py`, with goldens in
`tests/golden/test_integer_distribution.py`: it owns integer endpoints `F(y−1), F(y)`, seeded
randomized PIT, and mutually exclusive whole-/half-line `(under, push, over)` settlement. It is
deliberately **not production-wired** into training, scorecard, or serving. The owner must first
select one book/score contract: the recommended push-void conditional binary comparison with a
separately evaluated push, or a full three-category book/score contract. Until that selection,
there is **NO model pilot, heldout-outcome access, registry enrollment, `shipped` flip, or live
mutation**.

The two research-only preregistrations are pinned as
`/tmp/researcher_structured_count_transfer_20260720.md` SHA-256
`413c7e2049ddd9a0918a7d315dd1340f6bc26f22ac9069a1434abae0c002edf2` and
`/tmp/researcher_cross_sport_groupcdf_20260720.md` SHA-256
`07cdcd5aae90f5301955cefd136e50b377de1131dae3cdff2946c42a08a4c16b`. They permit only these
conditional next slugs:

- NBA FGA: `nba-fga-role3-integer-affine-groupcdf-v1`, only after a fresh signature-checked base
  control remains g4-bound under exact integer parity.
- NFL carries: `carries-qb-rb-integer-affine-groupcdf-bookpool-v1`, only if its simpler registered
  full-HPO control fails; a control ship stops the first-ship branch.
- NFL receptions: `receptions-role-position-integer-two-part-groupcdf-fixedlinear-v1`, only as an
  incumbent S1/S2/S3 supersession experiment.

Applicability/support and a registered slug are not enrollment, and none of these three is a
production or live-state action in this campaign.

**Generic role registry (training infra landed).** The receiving role×position two-part corner
(`receiving-role-position-two-part-groupcdf-fixedlinear-v3`) is no longer NFL-locked. Its role
columns live in `training/role_specs.py` — one `RoleSpec` per continuous cell across NFL/NBA/WNBA/NHL
(MLB excluded), each a small market-appropriate pick (basketball `PACE × PCT_* × rate`; NFL
target/carry share × yards-per; NHL `Corsi × TimeShare × Shot60`); composites sum their components.
Positions are **league-wide** (one roster per league, mirroring `Stats.positions`); the context
builder tiers only the codes a market actually fields, so a QB-only cell tiers just QB and receiving
just WR/RB/TE — NFL receiving stays byte-identical. The corner enrolls via `role_registry_gated`
applicability, so the board sweep can now search the method on all 43 continuous cells; nothing
ships without the six gates. Serving (`model_prob`) still validates the blob against the NFL
receiving roster and **fails loud** on any other shape — a non-NFL ship first needs the deferred
blob-driven serving generalization (no live gap: no non-NFL cell ships this method). The two NFL
heads now share one config-driven engine in `training/group_conditional_cdf/`
(`RECEIVING_CONFIG`/`RUSHING_CONFIG` → `fit_group_conditional_cdf`); the slugs and the persisted
`nfl_yards_experiment` blob schema are byte-identical.

**Small-sample / hierarchical layer (the NFL wall), cheapest-first.** Partial pooling dominates at
n ≈ 300–1000/group (Gelman & Hill 2007). (a) **EB-shrink the distributional parameters** per player
toward a per-position mean, cross-validated — stays in the LightGBMLSS stack. (b) **TabPFN v2
head-to-head** on the small-n cells (Hollmann 2025; ≤~10k rows), recalibrated through the PIT gate —
try before the full hierarchical build; GBDTs stay the backbone. (c) **Hierarchical-Bayes** only if
(a)+(b) fall short.

**Per-position model split (T11, NFL)** where eligible-position marginals diverge materially
(rushing-yards QB ~19 vs RB ~37); selective, min-row guard + pooled fallback. **Monotone priors**
(`monotone_priors.json`) for NFL small-n volume cells — only priors with mechanical meaning.

**Research-brief-flagged feature items** (brief settles the leakage / information boundary before
build): **C-3** line-movement & book-disagreement (`get_movement`/`get_ev_history` exist, resolve at
`target_at`); **B-5** missing-value semantics (both `fillna(0)` sites, one PR — parity rule); **L-3**
injuries/inactives + usage vacuum (report-timestamp leakage); **D-4** weather (Open-Meteo, needs the
M-3 stadium table); **D-5** referee/umpire (lowest priority).

### §6.7 WS-2 — MLB (now) + NHL (by Sep) activation

Entry: matrix/feature builds proceed **now**; training ships only post-D1 (MLB) / post-D2 (NHL)
— gates owned by [`mlb-nhl-activation.md`](../archive/mlb-nhl-activation.md) (data freshness, book
honesty, GO/NO-GO; not restated here).

**Reality (re-derive counts from §3).** MLB is in season with **0 cells served**; its board cells
fail mostly on **g1**, which is the known book-price degeneracy, not a bad model — so **MLB's
critical path is the book-honesty repair** (§3.2 recipe: audit → Odds API sign-off → `backfill_
historical_odds.py` → `inject_backfilled_odds.py` → matrix rebuild → re-sweep → **D1 packet**). NHL
has a few passing corners pre-staged but is **dark until October**, so its **D2 packet lands ~Sep**.
**No MLB or NHL cell flips before its gate** — the sweep engine's league-activation guard
(`model_strategy.confirm._drop_activation_gated`) announces and skips any withheld MLB/NHL board
passer, and the D1/D2 packets (owned by the activation doc) are where the owner removes the league
from the guard in the same PR that ships its first cells. Kill: unfixable data → a NO-GO packet is
a valid deliverable.

MLB/NHL are also not "efficient cells with nothing to learn" — they are starved of inputs (raw
`stat_types`: MLB 13, NHL 14, vs NBA 138; ~100–120 matrix columns vs ~460–480). Closing that
input gap is the parity build:

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

Read routing from **both** the offline board (§3 board rollup — where predictive shape is wrong)
and the live lifecycle (`check-graduation` — what actually earns), not counts alone. Per-cell gate
numbers and current candidates: `model_stats.csv` + the sweep board; live graduation/demotion:
`check-graduation`. A demotion is a §6.10 (WS-1) input, not automatically a routing verdict — the
benchmarks disagree (§3.3), so a book-BSS demotion is reconciled there before it reorders a queue.
What follows is the durable routing — which lever classes plausibly flip which cell classes — not a
snapshot.

#### NBA — comfortable

- **Calibration (§6.1) / sweep (§6.2)** on the Gate-4-only cells (snapshot examples: DREB,
  FG3A, FGM, FTM, STL, TOV; AST shipped via the centered normalization, MIN and RA via the
  scale fit).
- **Rung A then Rung B (§6.1)** on the g2-bias + g4 cells (FGA, fantasy-points-prizepicks).
- **Features (§6.3)** on PF (g1+g5 — the one genuinely hard NBA cell; also a D-5 referee
  candidate, last). The §6.6 DPO pilot confirmed family is not PF's constraint.
- **Free re-earns:** BLST + OREB hold `ship=True` all-six-gate rows while `shipped: withheld` —
  a fresh confirm re-earns the flip; queued in the breadth waves (`[[breadth75_campaign]]`).
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
- **Live-demotion cross-ref (§6.10, WS-1):** the live soak demoted PTS/REB/PA/PR/FGA on 30-day
  book-BSS, yet recent recorded picks on the *app line* hit ~64–67% including the demoted cells —
  the benchmark disagreement (§3.3). Do **not** re-route a demoted WNBA cell to a family/feature
  build on the demotion alone; WS-1 reconciles book-BSS vs app-line profit first (owner packet).
- **Count-family (§6.6 Double Poisson) / centered-SN** on the shape-bound residual (BLST/FTM/TOV
  count; DREB/PA shape) once the family lands — the built-lever lane is closed for WNBA at 13/18.
- Verdict: six of the eight realistic calibration/normalization cells.

#### NFL — the binding league, with a real failure mode

- **Popular-market status:** passing yards, passing TDs, rushing yards, and receiving yards all
  have fresh isolated full-HPO artifacts and independent heldout six-gate passes. Exact recipes,
  slugs, metrics, artifact hashes, commands, and evidence-scope caveats are in the 2026-07-20
  re-baseline in §6.6. All four remain unenrolled in live config/lifecycle by this work.
- **Board/confirm seam:** the generic `StrategySpec` registry, exact controls/corner/matrix/split
  identity, `--structural-strategy`, confirm artifact parity, and scorecard validation are
  implemented without NFL-yard branches in board/confirm. Serving dispatch begins only from the
  centrally validated selector. The final focused regression suites passed 431 tests, and real
  deterministic registry smokes returned receiving/rushing SHIP 6/6 with exact model/CSV identity
  and unchanged protected production hashes. The authoritative-run status and remaining unrelated
  runtime invariant are recorded once in §6. Until this seam is exercised through normal confirm
  and lifecycle controls, each isolated pass remains evidence rather than enrollment or a
  live-state change.
- **Residual ladder:** apply §6.1 calibration → §6.2 normalization → §6.6 family/structural head →
  §6.5 blend only to still-failing cells. The broad §6.3 raw-feature build remains demoted for the
  audited volume cohort; use only pre-registered targeted interactions or the QB/RB/role structure
  supported by clean validation. Never infer enrollment from conditional OOF, deterministic
  screening, or a quarantined full-HPO artifact; normal board/confirm and lifecycle controls remain
  binding even after an isolated heldout pass.
- **Book state (audited): the big-7 book is real; the blockers are code-side.** The July archive
  repair already re-priced attempts/completions/passing-yards etc. — their brier_book ≈ 0.250 is
  a sharp book the model honestly loses to, not a coin flip. The Odds API has zero NFL prop
  history before 2023-05-03 and post-cutoff refetch resolves to different entity names, so
  further paid backfill is a dead end (probe-verified). Honest-g1 blockers, all code-side:
  (a) **8 bookless cells** (qb-yards, qb-tds, yards, targets, sacks-taken, passing-first-downs,
  rushing-tds, receiving-tds) grade against synthetic 0.5 rows — **owner ruling: that stands;
  bookless markets must beat the coin flip** (no autopass; yards does, qb-yards honestly
  doesn't); (b) 17–21% synthetic-row dilution on priced cells (`scorecard._priced_rows` counts
  `Odds_synthetic`) — known measurement caveat, no gate change planned; (c) **tds ev-clamp bug — FIXED**:
  the population zi (0.777) refutes every quoted-star tds price (selection: books quote likely
  scorers), so `get_ev` clamped and confer persisted ev = 5×line on all 20,840 rows since
  2026-06. Fix: the moneylines parser stores a NULL ev + shape-free quote for gate-refuted
  prices (`_event_player_props`; both live confer and close-lines route through it; the DFS
  clamp semantics stay pinned); the poisoned evs are NULLed in the archive (quotes kept) and
  the tds matrix purged pending regen; (d) count-cell book decode round-trips ev→odds through
  drifting cv/zi — the matrix build should read the WS-1 `under_prob` column directly. Seed-row
  cleanup applied + matrices swapped from `staging_nfl_book_repair/` (`[[breadth75_campaign]]`).
- For any cell the model can't improve: lean harder on the book in the blend — rides the sharp
  line, ships if calibration holds. No cell is shelved (§8); never loosen a gate.

#### MLB / NHL — foundation, then activation

Feature foundations are §6.7 and proceed now; training/shipping is gated on D1/D2
([`mlb-nhl-activation.md`](../archive/mlb-nhl-activation.md)). First targets when active: MLB hitter
volume markets (batting order), Ks later (umpire); NHL goalie SV + skater shots/points.
Book-less MLB cells (pitches-thrown, both fantasy) grade against the synthetic coin-flip book —
**owner ruling: they must beat it** (g1–g3 vs 0.5 baseline, no autopass); the
continuous/centered-SN board is still their ship path, with g1 a real bar. MLB count matrices
are the fattest in the system (55–65k rows); sample size is not MLB's blocker.

### §6.10 WS-1 — Live alignment (Priority 1)

The highest-payoff workstream: the offline gates certify **deployable**, but the live evidence
says the deployable→profitable gap — selection, sizing, and *which benchmark you optimize* — is
now the binding constraint (§3.3). Read live from recent windows, reconcile the benchmarks, and
fix selection/sizing in the replay harness before any code change.

Entry: recorded-pick history exists (§3 probe — currently WNBA n≈492). **Footprint (§5):**
`strategies/kelly.py`, `strategies/profit_sim.py`, `data/runtime/*.parquet` (read-only). The
**Gate-2 thresholds in `graduation.py` are owner-only** — WS-1's benchmark findings become a
decision packet, never a session edit.

1. **Era-aware profitability read.** Never aggregate the 5-month history (it mixes model
   generations). Segment by model era and read recent windows. Era attribution comes from the
   model-version stamp Stage-0 adds (`model_version = yyyymmdd.norm-slug.sha8`, stamped in
   `persist.py` → flows to `history.parquet`); until legacy rows carry it, approximate with the
   per-cell **calibration fingerprint** `(Dist, CV, Temperature, Disp Cal, Step)` + `stat_meta.json`
   git flip dates (one-shot `scripts/backfill_history_eras.py` → `history_eras.parquet`).
2. **Benchmark reconciliation (the core finding → owner packet).** Three benchmarks disagree
   (§3.3): app-line hit rate (the money), sharp-book BSS (the graduation criterion), CLV. The five
   demoted WNBA cells hit ~67% on app lines while the book-BSS soak demoted them. **Autopsy each
   demotion** — is Gate-2's book-BSS criterion mis-aligned with app-line profit? — and **re-check
   weak-edge adverse selection** on the *recorded-pick* era (older mixed-era aggregates showed
   bottom-decile ~37% vs ~50% predicted). The output is an owner decision packet against the
   owner-only Gate-2 thresholds, never a session flip.
3. **Selection/sizing fixes, replay-validated first.** Validate every change in
   `strategies/profit_sim.py` (replay over resolved history) **before** touching code: an EV
   threshold by decile, selection-aware shrinkage in `strategies/kelly.py`, a CLV / time-decay
   discount. Ship the config change through refactoring-specialist + the devel-ship-curator PR.

**Kill:** replay too confounded to adjudicate → a pre-registered **live A/B via the sim-bettor
ledger** (roadmap D6). A serving-side *mechanism* implicated (not just sizing) → `research-analyst`
brief first (§8.2 — live serving-distribution changes are research-gated).

### §6.11 WS-4 — Ladder / tail calibration + Rivals + D3

The product-EV surface where calibration pays off directly: alt-line ladders, the right tail, and
Rivals head-to-heads. Copula research is **done** (R3 brief `/tmp/researcher_copula_stage0.md`).

- **Rivals difference-pricer — the cheap early product win.** Rivals is 2-dimensional; cross-game
  pairings are independence-correct (edge = the certified marginals alone) and same-game needs a
  single ρ the **incumbent correlation matrix already supplies at d=2**. Ingestion exists
  (`books.py` `rival_lines`, payout curve, `"vs."` flip); the only missing piece is a small
  **P(A−B>k) difference pricer** with push handling. The pricer **build is homed in the
  `dfs-products` lane, stage 1** ([`dfs-products.md`](dfs-products.md)); this track keeps the
  tail/ladder read below.
- **Ladder / tail read (query, not build).** The `ladder` table holds **15.5M historical rungs
  across all five leagues**: MLB 7.0M (18 markets, 4 alt-enriched: hits / total bases / home runs
  / pitcher strikeouts), NBA 4.2M (PTS/REB/AST/FG3M alternates, 2024-10→2026-06), NHL 3.5M (7
  markets, 2023-10→2026-06), NFL 523k (yardage + receptions alternates, two seasons), WNBA 314k.
  MLB/NHL additionally carry a **close-layer dual snapshot** (`observed_at` 23:00,
  evaluation-only vs the 01:00 feature layer) over their core windows — open→close movement and
  CLV are computable per slot (C-3's raw material; the feature build stays §8.2-gated). The
  standing tail read on `g4_tail_pit_ks` + central50/80 is runnable at scale, and the
  §6.5-deferred ladder-lift re-test (book-CDF fit on real rungs, previously blocked on an empty
  ladder table) is unblocked.
- **Parlay dependence (copula) — R3 verdict.** Gaussian copula default; **t-copula only as a
  tested branch** (adopt iff pooled exceedance-Spearman clears a simulated Gaussian null in both
  tails AND pooled pseudo-MLE with one ν per league gives ΔAIC≥10, ν̂≤15 — never per-pair ν). EB
  shrinkage is **two-level hierarchical in Fisher-z** (team → pair-type mean → 0), not the
  incumbent's shrink-thin-pairs-toward-zero (a bias against the lane's own edge). Reuse
  `correlate.py` (`_residualize_gamelog`, the cache, matmul Spearman, `2·sin(πρ_S/6)` remap,
  `_nearest_psd`). **Census script** `census_parlay_pairs.py` (read-only Sonnet task over the
  `{LEAGUE}_corr.parquet` caches); **kill rule: a league is viable iff ≥15 fit-eligible pair-types
  reach N≥300.** PIT source = re-score holdout rows through the production pickle + serving decode
  (randomized PIT for discrete/hurdle), same-game grouped by (Player, Date). **Stage acceptance:**
  (1) per-cell randomized-PIT KS p≥0.01 to enter, ≥90% of shipped cells p≥0.05, >30% failing ⇒
  kill; (2) OOS joint log-lik ≥+0.01 nats/pair vs independence (block-bootstrap CI>0) AND >0 vs
  incumbent; (3) decile joint-reliability gap ↓≥20% vs incumbent, parlay Brier not worse.
- **D3 packet** when breadth + the stage-0 copula brief (done) + **sleeper-parity** all exist.
  Sleeper-parity is a hard D3 blocker serialized with parlay-dependence in the same files
  ([`parlay-dependence.md`](parlay-dependence.md)) — it **needs owner scheduling** (Risks, §8).

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
`gate-status` cron promotes graduates to `main` via PR. Demotions flow back the same way — but a
Gate-2 **demotion is a §6.10 (WS-1) input, not automatically a routing verdict**: the soak keys on
sharp-book BSS while the money is the app line, and the two disagree (§3.3), so reconcile a
demotion in WS-1 before it reorders a queue. The Gate-2 thresholds in `graduation.py` are
**owner-only** — a reconciliation becomes a decision packet, never a session edit.

**Supersession (Tier 1).** A *baselined* (already-shipped) cell never re-ships on a fresh 6/6 —
the candidate must beat the incumbent via `scorecard.supersede_verdict(baseline, candidate)`:
**S1** candidate clears the 6 gates standalone, **S2** paired Brier CI lower-bound > 0, **S3**
paired Kelly-Sharpe Memmel-z > min. All three → SUPERSEDE (swap the strategy in
`stat_meta.json`); any fail → HOLD. The S2/S3 asymmetry is deliberate — it stops strategy-churn
on noise. First-ships use Tier-0 absolute gates only. CLI:
`sportstradamus ship scorecard --baseline … --candidate …` (both sides full-HPO,
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
sportstradamus meditate --league NFL --market carries --deterministic \
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
sportstradamus meditate --league NFL --market carries --deterministic \
    --bypass-withholding --target-normalization ratio_meanyr
sportstradamus ship scorecard \
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
- **A running sweep OWNS `stat_meta.json`.** `model-strategy-sweep --confirm` persists and reverts
  per-cell strategy fields live; never `meditate`, hand-edit `stat_meta.json`, or start a second
  sweep while one runs. Check the §3 concurrency probe (`pgrep`) before any training or config
  edit — a concurrent write races the confirm's revert and can strand a cell mid-flip.

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
§6.6-flagged feature items C-3 / B-5 / L-3 (D-4 brief-note level); **any §6.10 (WS-1) change to a
live serving distribution or its dispersion — not the sizing/selection config, which is a plain
knob**; any hole below before betting the plan on it. Plain knobs (normalization slug, loss choice,
ordinary features, Kelly/selection config) need no brief. The four rework briefs
`/tmp/researcher_{count_family,continuous_family,copula_stage0,sweep_ux_versioning}.md` discharge
the gate for WS-3's two families, the copula, and the sweep/versioning build. To proceed without a
brief on a hook-gated edit, write a one-line justification to `.claude/.state/research_waiver`.

**Open holes:**

- **#0a — Mixture is out of the sweep pool until it can serve** (owner call).
  `SWEEP_CAPABILITIES` requires `CAP_CONFIRM`, which drops Mixture from every cell's family pool.
  It was earning that budget on the board and nowhere else: on the first holdout-blind run it was
  the rank-1 corner on 3 of the 4 cells searched — NFL passing tds +0.202, NBA FGA +0.150, NFL
  passing yards +0.100 — against best-confirmable corners of +0.033, +0.076, and −0.040. Passing
  yards is the sharp case: both of its two gate-passing corners were Mixture, so the cell has no
  shippable recipe on any family the sweep can now propose. **Put Mixture back in the pool the day
  it can serve** — that needs Mixture decode in `prediction/model_prob.py` and carries the
  ZINB/NegBin identity merge as a migration (research-gated per the family rule above). Until then
  the boards under-report what the data supports, and that gap is the argument for building the
  serve path, not for re-ranking a family that cannot ship.

- **#0b — Gate-4 baseline hysteresis** (highest priority; owner call). §6.0.2 carries the
  decision packet and the recommendation (ship the scale fit first; hysteresis only if churn
  persists). It gates trust in the ship-incrementally premise.
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
  set, or only re-rank trials that ship/fail together?** Lever 1 **rebuilt as v3** (§6.1 bullet;
  the earlier per-cell validation `[[calibration_hp_selection_lever]]` was measured through the
  v2 closure, whose model-only `s = 0` frame was 3.5× pessimistic on SN and optimistic on
  counts — one closure, two opposite biases); Rung B′ / Lever 4 built + opt-in. The companion
  HP-*range* hunch is **closed negative**: within-cell capacity sign test 11/30 n.s., failing
  fits skew *under*-fit, prior frozen-matrix null — no range narrowing anywhere. (a) The
  **scale-bound vs shape-bound mix** is now **read** — R2 routed every SN cell by normal-scores z
  of the gate-matched PIT (`/tmp/researcher_continuous_family.md`): the shape/kurtosis cohort is
  §6.6-bound (centered-SN / SHASH), the scale-bound residual is the search-gate's. So the cheap
  cov50/cov80 query is no longer the open question — it is answered per cell in R2's routing table;
  what remains is executing the confirms. (b) Rung B′'s PIT-KS count objective
  has no sharpness brake — watch g6's over-leg and the g1 acceptance for over-tightening. (c) only
  promote `--stabilization` to a swept axis if MAD/L2 ever wins on ≥1 cell (YAGNI). (d) the
  under-wide-AND-mislocated NFL SkewNormal cells (attempts/carries, central-50 ≈ 0.27/0.34) are out
  of reach for any selection/post-hoc width fix — that defect is signal/family, not width; confirm
  via §6.3 features and/or the §6.9 `log(volume)` offset before spending a calibration lever on
  them. (e) **hurdle cells bypass Optuna entirely** (`_step_select_hyperparams` short-circuits),
  so no `hpo_selection` — v3 included — reaches the 10 hurdle cells (a `calibrated` pin there now
  warns as inert); whether the hurdle path needs its own calibrated search is open.
  [research: `/tmp/researcher_calibration_hp.md`, `/tmp/researcher_hpo_objective.md`,
  `/tmp/researcher_lever1_v3.md`]
- **#10 — the benchmark disagreement (WS-1, §6.10, highest live priority).** App-line hit rate,
  sharp-book BSS, and CLV point different ways: recent WNBA picks hit ~64% (demoted cells ~67%)
  on the app line while the book-BSS soak demoted five cells. Root cause unknown — is Gate-2's
  book-BSS criterion mis-aligned with app-line profit, or is the app-line hit rate selection-biased
  (only the picks we bet resolve)? WS-1 autopsies it into an owner packet; the Gate-2 thresholds
  are owner-only (#12).
- **#11 — the MLB book-repair cost (WS-2, §6.7).** MLB's g1 failures are the klepto-era book seed,
  fixable by the §3.2 recipe — but the Odds API spend is unknown until audited and needs owner
  sign-off, and the D1 activation option decays weekly. Read-only audit first; then the packet.
- **#12 — Gate-2 criteria are owner-only (standing constraint, not a hole to close).** WS-1 may
  find the live-soak demotion criterion misaligned with app-line profit; the finding is a decision
  packet against `graduation.py`, never a session edit (§4, 2026-06-10).
- **#13 — sleeper-parity is the live D3 blocker (needs owner scheduling).** D3 needs {breadth,
  the copula stage-0 brief (done), sleeper-parity}. Sleeper-parity is serialized with
  parlay-dependence in the same files ([`parlay-dependence.md`](parlay-dependence.md)) and has no
  session owner — the owner must schedule it or D3 cannot fire.

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
decoded mean and a high gate cannot hide an inflated μ; the failure mode is not observed live (owner call);
the NFL g1×dispersion question (old hole #4) is answered — the volume five are feature-mature, their
negative BSS is target-shape not dispersion, so a scale fit does not pull g1 under threshold and they
route to §6.2 normalization + §6.6 family (`[[nfl_volume_cells_feature_mature]]`).

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
- Counts re-derived from the §3 blocks (shipped-counts + board rollup + `check-graduation`), never
  carried forward as prose — a ledger line that cites a count cites the command that produced it.
- One ledger line appended to §10; status line updated if a stage boundary was crossed.
- Never push `devel` directly — devel-ship-curator carves ship PRs.
- Durable non-obvious lesson? Offer a memory capture (CLAUDE.md §Agentic workflow conventions).

## 10. Ledger (append-only, newest first, cap ~15 — older lines live in git)

- 2026-08-24 · Exp-2 verdict: calibrated-everywhere default flip KILLED (1/4 improved, MLB g1
  breach); v2 hinge gone — v3 IS the `calibrated` pin; retry machinery + two-attempt doctrine
  stay. Two live failures on MLB hits allowed forced the final SN measure to max(blended OOF,
  blended validation-ensemble): model-only Goodharts at low model_weight (g4 0.153), OOF-only
  passes era-fragile trials (0.104). Final: MLB 0.0371 ≈ v2's 0.039, WNBA MIN ships 0.0464,
  controls hold.
- 2026-08-23 · Lever 1 v3 landed (worktree `g4-calibrated-v3`, brief
  `/tmp/researcher_lever1_v3.md`): hinge deleted — raw CV loss + OOF PIT-KS as
  `TPESampler(constraints_func=…)` feasibility (optuna 4.9.0), per-trial KS from the cv fold
  boosters (`predict_cv_oof_params`, no refit, ~4.7× rows), 1-SE ε final pick; C2 joint `(c, s)`
  closure fix; fidelity ladder killed blend/post-hoc threading (gaps ≤0.004). Board g4 discount
  now conditioned on (family, 1/√n) in `sweep._ledger_gate_discounts`. Exp 4 read: R5
  λ-shrinkage NULL (§6.1 Rung B). Pending: Exp 2 default flip + retry deletion, Exp 3
  count-objective flip.
- 2026-07-21 · Role×position two-part method generalized off NFL. New `training/role_specs.py`
  registers a `RoleSpec` for every continuous cell in NFL/NBA/WNBA/NHL (43 total, MLB excluded):
  league-wide positions (mirror `Stats.positions`, tier only codes a market fields) + small
  per-market column pick; NFL receiving byte-identical (`_role_score` flat-reduce preserved,
  max-diff 0.0 on 14,923 rows). Corner enrolls via `role_registry_gated` so the board sweep can
  search it everywhere; nothing ships without the six gates. Coverage golden
  `tests/golden/test_role_specs.py` fails loud on any uncovered continuous cell. Serving stays
  NFL-validating + fail-loud (blob-driven generalization deferred, no live gap). Pipeline cleanup:
  the two NFL heads already share one config-driven `group_conditional_cdf` engine; collapsed
  `train_market`'s structural if/elif into `_structural_gate_inputs`, split the 226-line receiving
  support audit into six cohesive helpers, and pulled the structural branches out of
  `_step_predict_splits`/`_step_persist_artifacts` — four transient complexity tags removed, all
  honestly under CC 10. Gates: golden 3771 pass (only the pre-existing shipped-cell ship-gate
  invariant red), integration 30, Ruff + complexity clean.
- 2026-07-20 (2) · Generic board/confirm seam landed: declarative `StrategySpec` registry,
  generic `--structural-strategy`, canonical controls JSON, spec/family/control/matrix-bound corner
  fingerprint plus separate structural split fingerprint, and exact model/CSV/model_stats/
  scorecard/serving identity. The final focused regression suites passed 431 tests; real
  receiving/rushing deterministic paths both returned 6/6 with exact model/CSV identity and
  unchanged protected production hashes, but no confirm ran. Authoritative Ruff passed; the full
  golden/integration runs exposed one unchanged shipped-cell runtime invariant plus two repaired
  fixture defects, so the integration marker remains unset (details: §6). The exact integer
  PIT/settlement kernel is golden-tested but deliberately not production-wired. Research briefs
  and hashes are pinned in §6.6; owner must select push-void
  conditional versus full three-category semantics before any NBA FGA, NFL carries, or NFL
  receptions candidate pilot. No integer-candidate pilot/heldout access, registry enrollment,
  `shipped` flip, or live mutation occurred.
- 2026-07-20 (1) · NFL popular-market re-baseline: passing yards
  (SkewNormal/`ratio_meanyr`/CRPS, n=369), passing TDs (DPO canonical `none`; legacy evidence
  identity `ratio_meanyr`/NLL/`pit_ks`/**`roe_mean`**, n=373), and rushing yards
  (`rushing-qb-rb-affine-groupcdf-bookpool-v1`, cold 95-trial HPO, n=1009), plus receiving yards
  (`receiving-role-position-two-part-groupcdf-fixedlinear-v3`, cold 146-trial HPO, n=2238), each
  produced an isolated full-HPO artifact and passed all six gates on independent heldout evidence.
  Receiving also passed every frozen
  canonical validation/support guard on 2,239 rows / 382 players; its prior direct-read result
  remains retired for 1,118-row heldout overlap. Evidence and commands are pinned in §6.6; all
  artifacts remain quarantined, the generic board/confirm seam was not exercised for enrollment,
  and there was no `shipped` flip, live enrollment, push, or numerator change.
- 2026-07-19 (3) · Mixture family #6 built end-to-end + board-piloted same day (owner bumped it to the front). Ships first: NFL receptions (ratio_meanyr/nll/direct/nll) + yards (eb_meanyr_k10/crps/centered/nll) confirmed 5/5 → devel, NFL 9/20; carries g1 / qb yards g3 / targets g2 reverted. Build: pipeline branches (decode/fuse-by-location-shift/scalar-c dispersion on served PIT-KS/temperature/persist MIX_* via strategy encode), helpers kernels (_mixnorm cdf/ppf/odds + start values), scorecard gates (MIX decode + CDF/PPF), sweep FamilySpec (6 corners), validator continuous-aware (CONTINUOUS_DISTS), confirm hard-skips Mixture (serve-iff-ship). Three guardrails made it trainable (§6.6): scale clamp [0.02,20]×label-std, L2 stabilization, min_child_weight/lambda_l2 floors — root cause was σ→0 likelihood spikes + zero-hessian Newton blowup (probe-localized: fused_loc inf−inf → NaN). Pilots: receiving yards Mixture corner = rank-1, g4 0.0552 vs 0.050 (60% of SN's excess removed), g1 +0.0064 CI-hi, rest pass — serve path then full-HPO confirm; rushing g4 halved but g1-walled (bss −0.0245). Sharp-book kill NOT triggered. Scoreboard: WNBA 14/18 ✓ · NBA 16/21 ✓ · MLB 14/19 · NHL 10/15 · NFL 9/20.
- 2026-07-19 (2) · NFL breadth research + first mandatory ship. research-analyst brief (research/briefs/researcher_nfl_breadth_20260719.md): board ceiling 13/20; receiving/rushing-yards g4 = upper-tail under-dispersion (right-tail PIT 0.21–0.25 vs 0.05, positive-only KS ≈ full KS — NOT the zero atom, ZAGamma dominated); passing yards + completions + passing tds = efficient-book kills (88×/8×/5× n at unfavorable point estimates); interceptions correction — DPO/NegBin corners PASS g1, wall is g4 over-dispersion; attempts = only favorable thin-n cell (1.5× effective n via EB-shrink). Owner: yards trio = priority, interceptions lowest; endorsed a 6th dist type conditional on smoke-testing the compound count×severity architecture first → smoke test run (real PBP severities, MC compound PIT): PARTIAL on receiving (right-tail 0.206→0.180, ~13% of needed), NO MERIT on rushing (worse) — independence assumption understates the dependence-driven joint upper tail; compound rejected for yards, **Mixture head pilot = active lever**. NFL receptions SHIPPED at confirm (SN ratio_meanyr, dist_training_loss=nll persisted — S4 live) → NFL 8/20; carries + qb yards (bss +0.118, g4-pass, other-gate fail) reverted, recovery queued. Living plan file created: docs/handoffs/breadth75_plan.md.
- 2026-07-19 · Owner: NFL breadth = highest priority. NHL+MLB continuous boards closed: NHL saves +0.011 and MLB pitches thrown +0.052 both ships-but-RANKS-ONLY (dl=nll, third/second occurrence) → S4 BUILT same day (commit 4538861: dist_training_loss persists per cell, meditate resolves it per-cell, ranks-only machinery deleted; −87 LOC). MLB pitching outs REVERTED at full-HPO (g4; +0.057 ×2 corners remain for a manual walk), runs allowed re-swept to g1-only −0.040 (book-side, dead this wave), NHL skater fantasy best −0.230 g4-only on a persistable centered corner (posthoc-isotonic retry queued), goalie fantasy/hitter fantasy/pitcher fantasy dead. NBA STL's DPO ship stale-pinned the ZINB book round-trip golden → re-pinned to FTM. W4 NFL chain LIVE: receptions/targets/carries flipped ZINB (blindspot workaround) → count board+confirm → unshipped flips revert → continuous board+confirm → tds matrix regen. Post-chain queue: pitches thrown + saves re-confirms (S4-unlocked), pitching-outs walk, skater-fantasy retry. Scoreboard: WNBA 14/18 ✓ · NBA 16/21 ✓ · MLB 14/19 · NHL 10/15 · NFL 7/20.
- 2026-07-18 · W1 finished + W2/W5 executed; three leagues over target. MLB count board 8/8 confirms shipped (6 DPO, 2 NegBin) → MLB 14/19 with batter strikeouts' W5 re-earn. W2 reroutes: NHL sogBS shipped (DPO, BSS +0.220), NBA FGA all-16 kill (g4+g6 — family not binding, stays SN). W5 direct re-earns: NHL assists (DPO +0.085) + MLB batter strikeouts shipped; NBA FTM's DPO corner honestly dead — first live calibrated-retry exercise (fired, still g4); NBA STL shipped on retest (DPO +0.083) → NBA 16/21 CROSSES 75%. Found + fixed: report()'s pickle-dist writeback reverted STL's pending flip off a stale pickle — writeback removed, stat_meta dist input-only (save_distribution_config archived; forced-dist warning = flip-reached-training audit). W3 pilots both KILL (PA g6-only −0.004 knife-edge; NFL receiving-yards g1+g4 sharp-book) ⇒ K4; Mixture.py confirmed in pinned lightgbmlss (SplineFlow too) — escalation ask becomes Mixture-not-SHASH, deferred until continuous-board verdicts. NHL+MLB continuous boards in flight. Scoreboard: WNBA 14/18 ✓ · NBA 16/21 ✓ · MLB 14/19 · NHL 10/15 · NFL 7/20.
- 2026-07-17 · W1 count boards NBA + NHL closed. NBA: 0 board ships (FTM hurdle +0.02 corner reverted at full-HPO on g4 alone; STL −0.040 / TOV −0.300 kills) but BLST + OREB shipped via direct full-HPO re-earn → NBA 15/21. NHL: 3/3 confirms shipped — blocked (DPO, BSS +0.024), powerPlayPoints (hurdle-ZINB, BSS +0.449), shots (DPO, BSS +0.012) → NHL 8/15; assists −0.058 near-miss (best corner DPO). DPO ships now 4 (TOV, points, blocked, shots). S1 evidence: joint-ZINB 0/132 board corners, best −0.166. Hurdle path has no Optuna search (`_step_select_hyperparams` fixed/warm params) — calibrated HP selection can never apply to hurdle corners. Owner-ordered fallback BUILT same day: `_calibration_penalty` extended to the LSS count/Gamma families (refit → decode → per-trial dispersion-c → served PIT-KS; guard now `not use_hurdle`) and meditate auto-retries a cell once with `hpo_selection=calibrated` when ship fails on g4 alone under loss selection, pinning the knob to stat_meta on a shipped retry; confirm re-syncs subprocess-written pins from disk so later whole-file writes can't erase them. Three gates green + goldens (predicate matrix, persist, sync, count-closure live-path).
- 2026-07-17 · WS-3 pilots + breadth-75 reframe (owner: 75% every league is the mission, not WS-3 step-following). DPO first ship — WNBA TOV full-HPO 5/5 → devel (4/4 DPO corners board-pass, all NB corners die g4, BSS +0.042); NBA PF all-kill (<50% gap-close, family not binding → §6.1/§6.2); NHL points in flight; kill rule dead. Gap: NBA +3 / NFL +8 / NHL +8 / MLB +10 (WNBA 14/18 done). Five waves: count boards (non-NFL first) → integer-meta reroute → centered-SN pilots + continuous board → NFL book repair (paid backfill agent in flight, apply-at-gap) → ECE recal. Found: NBA BLST/OREB ship=True-but-withheld (free re-earns); `--dist-class` meta-dist blindspot (§6.6 gotcha); book-less MLB trio auto-passes g1–g3 (owner rule). Model-class escalation past families → §6.6 (ordinal stack / mixture / comp-PMF, §8.2-gated). NFL book audit closed same day: July repair already real for the big-7, paid backfill dead (API hole pre-2023-05-03; 882 credits spent, probe-verified ~0 recovery) — blockers rerouted code-side; tds ev-clamp bug root-caused (population zi 0.777 vs quoted-star prices, selection effect) + FIXED (gate-refuted quotes → NULL ev + shape-free quote; 20,840 poisoned evs NULLed; tds matrix purged) + repair applied (88k seed rows deleted, matrices swapped); autopass RETRACTED same day — bookless cells must beat the coin flip. Pilots 2/3 SHIP: NHL points confirm 5/5 → devel (BSS +0.012, n_val 11.8k). Simplification track S1–S5 adopted (§6.6). NHL points interim: DPO 6/6 deterministic +0.226, pit_ks 0.0153 (NB floor 0.083) — low-mean ceiling thesis validated; full-HPO confirm in flight.
- 2026-07-11 · WS-2 activation COMPLETE: 9 cells live on devel — 5 MLB (2026-07-10) + 4 NHL (goals, hits, shotsAgainst, timeOnIce; commit 3635a20), all deterministic-board → full-HPO confirm on 2-season matrices; final spend 2.39M of 5M credits. NHL powerPlayPoints failed confirm, saves ranks-only (non-persistable dist-loss corner). Post-GO grind on no-ship cells moves to the §6 operating loop (detail: ../archive/mlb-nhl-activation.md §10).
- 2026-07-10 · dfs-products lane created (decision-engine expansion: game-line combos verify-first, Ladders, alt-line hardening) · §6.11 Rivals pricer build repointed there (tail read stays); ladders + gamelines stage-0 briefs in docs/archive/; serve-time budget locked ≤15 min heavy day · next: unchanged
- 2026-07-10 · WS-2/WS-4 backfill program done (1.76M credits of 5M): MLB+NHL feature gap closed (7-11 sharp books, NHL 2023-24 refilled), `ladder` seeded 15.5M rungs all five leagues (alt keys backfill-only), MLB/NHL close-layer dual snapshots (23Z eval-only) → CLV/movement computable; §6.11 tail read + §6.5 ladder-lift re-test unblocked. MLB matrices rebuilt 19/19 at 2 seasons; NHL rebuild + both sweeps in flight.
- 2026-07-09 · WS-2 Track A (key-independent) done: MLB/NHL klepto seed purged (3.1M junk odds rows), backfill `_probe` key bugfix, per-league `trim_gamelog` windows (MLB 95k / NHL 110k rows = 2-season matrices), Savant affinity bot-block fix, activation guard emptied per GO. Gates clean. Backfill + rebuild + sweep/confirm blocked on the activated Odds API key (detail: ../archive/mlb-nhl-activation.md §10).
- 2026-07-09 · Stage-0 engine work LANDED (§6 status revised in place): version stamping train→serve→history→dashboard + `backfill_history_eras.py`; board `--resume`/per-cell upsert/`swept_at`/`code_rev`/`--dry-run`; FamilySpec registry live. Residue → WS-3: confirm queue-manifest, auto archive-snapshot, family-as-swept-axis. Owner declared D1/D2 GO with 5M-credit backfill — WS-2 activation execution starts (plan: `~/.claude/plans/review-the-model-improvement-track-md-ha-lexical-storm.md`).
- 2026-07-07 · plan reworked profit-first (owner reframe). WS-1 live-alignment = P1; MLB+NHL activation folded in (WS-2); family research DONE (WS-3: Double Poisson count + centered-SN continuous); copula stage-0 DONE (WS-4); sweep-engine Stage-0 spec written. §6 restructured to workstreams; sweep-era verdicts → refs §15. Confirm league-guard added — withheld MLB/NHL never auto-flip pre-D1/D2 (`_drop_activation_gated` + goldens).
- 2026-06-28 · WS2 book-shape gate cleared but served-gate lift decoupled at w≈0.90 (research bet); book DREB stays in-family SkewNormal (measured skew ≪ bound), ladder table empty → ladder lift deferred. detail refs §15.
- 2026-06-27 · WNBA 7→13/18 on built non-family levers (FTM hurdle+pit_ks; PR/RA/AST/OREB/FGA). built-lever lane closed; residual §6.6-bound (count family / centered-SN) + STL §6.3.
- 2026-06-26 · §6.5 book-distribution audit + Pooling-half BLP + decoupled + weight-challenge probe-v2 all NO-GO; blend not the lever at w≈0.90; residual → §6.6/§6.1. refs §15.
- 2026-06-25 · NFL hole #4 NEGATIVE — volume five feature-mature (485 cols), negative BSS = target-shape not features; `centered_additive_mean10` decisive (carries g1 flips on pure target transform) → §6.2/§6.6, not §6.3.
- 2026-06-23 · Lever 1 reworked to calibrated HP search-gate + VALIDATED (ships WNBA PR); per-cell `hpo_selection` persists so cron reproduces.
- 2026-06-22 · Gate 6 redesigned (OR of 3 one-sided legs + anchor hysteresis) + widened to all cells; `ratio_projvol` REFUTED 0/40, g6 validated; first full strategy board generated.
- 2026-06-19 · Gate 6 added (anti-shrinkage); §6.3 feature batch 1 + Playoff/series shipped via §7.2 A/B.
- 2026-06-17 · §6.0 train/live parity harness built; archive NBA Totals ×1.4427 corruption root-caused+fixed+guardrailed; QW-1 game-script A/B NO-SHIP across 15 cells.
- older lines live in git (`git log docs/handoffs/model_improvement_track.md`); durable sweep-era verdicts consolidated in [`../operation_ship_references.md`](../operation_ship_references.md) §15.
