# Plan: Mitigate GBDT Regression-Toward-the-Mean in the Training Pipeline

> **Multi-session home-of-record.** LightGBMLSS predictions compress toward the
> global mean (high-volume players under-predicted, low-volume over-predicted).
> This document is the durable plan + progress log for the fix, worked entirely on
> branch `claude/fix-gbdt-mean-regression-GcY1g` (intended rename `model-research`)
> under **PR #46 → `devel`** — every phase's code, run-log entries, and status
> updates land as commits on that one PR, not separate per-phase PRs. Status is
> updated each session. No fact in this document may be lost; citations are
> numbered `[n]` and collected in **References** at the end.
>
> **Companion context doc:** `gbdt_mean_regression_context.md` (decision history,
> prior-phase results, research verdicts, references).

## Abstract

LightGBMLSS predictions exhibit **regression-toward-the-mean / top-decile
compression** — a GBT leaf-averaging property that under-predicts high-volume
players and over-predicts low-volume ones. The **North Star is baseline breadth:
set a baseline for ≥ 75% of markets in every covered league** (NBA ≥ 16/21, WNBA ≥
14/18, NFL ≥ 15/20), not depth on a few cells. **Current state (full-HP locked,
2026-05-21): NBA 13/21, WNBA 10/18, NFL 13/20** — gaps of −3 / −4 / −2. The
operative next step toward breadth is the **Stage B1.6 feature/bias track**.

This is the **lean execution plan**: what to do, the per-market status, and the
execution-critical mechanics. History, prior-phase results, research verdicts, and
references live in the **context doc** (`gbdt_mean_regression_context.md`).

## Table of contents

- [North Star — baseline breadth (≥ 75%)](#north-star--baseline-breadth--75-of-markets-per-league)
- [Market cell status — Gate 1 / `devel` / `main`](#market-cell-status--gate-1--devel--main)
- [Phase status index](#phase-status-index)
- [Scope, testing policy](#scope--leagues-this-plan-covers)
- [Ship incrementally — per-market graduation](#ship-incrementally--per-market-graduation)
- [Diminishing returns — stop-the-track principle](#diminishing-returns--stop-the-track-principle)
- [Inference-path compatibility](#inference-path-compatibility-applies-to-every-shipped-change)
- [Path to 75% — feature/bias track (Stage B1.6)](#path-to-75--featurebias-track-stage-b16)
- [Cross-league caveats](#cross-league-caveats-read-before-running-any-cross-league-ab)
- [Critical files](#critical-files)
- [Verification](#verification-every-code-session)
- [Session handoff](#session-handoff)

## North Star — baseline breadth (≥ 75% of markets per league)

**The objective is breadth, not depth: set a baseline for at least three-quarters
of the markets in every covered league — NBA ≥ 16/21, WNBA ≥ 14/18, NFL ≥ 15/20.**
This supersedes the earlier "ship only on a ≥ 5% top-decile-MAE *improvement*"
framing. Stage B1.5 §7a showed top-decile compression is **family-invariant** (a
GBT leaf-averaging property no strategy or distribution swap removes), so gating
the *first* ship on ≥ 5% compression improvement blocks most cells from ever
shipping (the NBA sweep set a baseline on only 2 of 21 markets under that bar). The
gate is reframed to measure money-making breadth directly.

**Breadth is a 3-rung ladder, not a one-shot target — ratchet quality and breadth
alternately:**

1. **Rung 1 (now): reach 75% per league at the current bar.** Absolute bias gates at
   the wide Phase-0 value (30% of the band's empirical mean, floor 0.10). Close the
   gaps (NBA −3, WNBA −4, NFL −2) via the Stage B1.6 feature/bias track.
2. **Rung 2 (next): tighten the absolute gates 30% → 20% of band mean** (floor 0.10
   unchanged) and re-reach 75% at that stricter bar. Some Rung-1 passers will fail at
   20% and need more B1.6 work. **Dispatch a fresh research agent at that point** to
   inform the strategy.
3. **Rung 3 (after): push to >90% coverage** at the tightened (20%) bar.

### Two-tier ship model

**Tier 0 — set the first baseline (absolute gates only; this is the priority).** A
cell with no baseline gets one as soon as *any* candidate clears the **absolute**
gates — nothing relative is required:

1. **Bench-warmer (bottom-quartile) absolute gate** — bottom-quartile signed bias
   within the one-sided over-prediction bound (under-prediction tolerated).
2. **Star (top-decile) absolute gate** — top-decile signed bias within the
   bidirectional bound.
3. **BSS in tolerated range** — `brier_skill_score ≥ 0` (beats the book) on the
   full-quality retrain. This is the knob for reaching 75%: if a league can't get
   three-quarters of markets to ≥ 0, widen toward the Gate-2 −0.02 tolerance rather
   than ship a book-losing cell.

The **incumbent devel/default strategy is itself a candidate.** Of every candidate
clearing the absolute gates, the highest-`brier_skill_score` after a full
non-deterministic retrain ships as the baseline (live production A/Bs one strategy
per cell; tiebreaker in [ship_gate.md](ship_gate.md)).

**Tier 1 — supersede an established baseline (absolute + relative gates).** A
challenger must clear both the absolute gates and the full Universal decision
threshold: ≥ 5% top-decile-MAE improvement, global MAE not worse, BSS not worse,
bottom-quartile bias not more positive. The ≥ 5% bar's correct role is here.

**Gate 2 — live graduation** (unchanged): a newly-set baseline soaks ≥ 14 days,
then graduates on settled-offer book-BSS, calibration, and profit-sim.

**"Absolute gate"** = the per-band signed-**bias** (calibration) bounds built this
session — bottom-quartile one-sided on over-prediction, top-decile bidirectional —
at the wide Phase-0 values (30% of the band's empirical mean, floor 0.10). Current
numbers + BSS tiebreaker live in [ship_gate.md](ship_gate.md).
(`compression_eval.verdict()` currently encodes only the Tier-1 path; the Tier-0
absolute-only mode is the immediate next code step.)

**Universal decision threshold — the Tier-1 gate (every market, every covered
league).** Once a cell's baseline is set, a challenger ships only if it (1) reduces
**top-mean-decile MAE by ≥ 5%** vs the baseline, (2) does not worsen **global MAE by
> 1%**, (3) does not worsen `brier_skill_score`, and (4) does not buy the top-decile
win with **low-volume over-prediction**. The **Tier-0 first-baseline gate is the
absolute subset**: condition (4)'s absolute bounds (bottom-quartile + top-decile
bias) plus `brier_skill_score ≥ 0`; the relative parts are dropped (no prior baseline
to improve on).

Condition (4) is gated *asymmetrically* by betting risk. **Bottom quartile** (bench
warmers): failure mode is over-prediction only, so the bottom-mean-quartile signed
bias must (a) not become more positive than the default's (*relative*) and (b) not
over-predict beyond a fraction of that quartile's mean (*absolute, one-sided* —
currently **+30%**, floor **0.10**); **under-prediction is tolerated**. **Top decile**
(stars): **bidirectional** — |signed bias| within a fraction of that decile's mean
(currently **30%**, floor **0.10**). The lowest quartile is pooled coarsely on purpose
(bench warmers generalize more than stars); absolute bounds start **wide and tighten
as the bias-correction track lands improvements**. Condition (4) was added after Stage
B1.5 §7a found two cells whose only MAE wins came from over-predicting low-volume
players (FG3M bottom buckets predicted ~3.4× actual); the bounds are tunable but the
*direction* (a top-decile win must not be financed by bottom-quartile over-prediction)
is fixed. Quick-reference values: [docs/ship_gate.md](ship_gate.md). The threshold
must hold on **every market in every covered league** — or the routing config records
the exceptions. The harness
([compression_eval.py](../src/sportstradamus/scripts/compression_eval.py)) is the
ship/kill gate.

### Roadmap to 75%

1. **Audit (done — Step 1):** run the absolute-only Tier-0 gate over every cell ×
   candidate (incl. incumbent) in all three leagues.
2. **Fill the gap (Step 2 — the work now):** the **Stage B1.6** feature/bias track
   (post-hoc bias correction, expanding-mean encoding, opponent-defense/blowout
   features). Post-hoc bias correction pulls both bands toward zero, serving the
   absolute gates directly.
3. **Set + soak:** ship the highest-BSS passer per cell to `devel`; start the
   14-day Gate-2 soak.

**Immediate next steps.** First **code** task: add `compression_eval.verdict()`
Tier-0 absolute-only mode (today it encodes only the Tier-1 relative path — the
Stage B1.6 prerequisite). Then run **Stage B1.6** to close gaps (NBA −3, WNBA −4,
NFL −2), then set + soak. Depth tracks (A2/B2/B3) follow only once breadth is met.
The **2026-05-22 research verdict** ranks the B1.6 work (post-hoc mean-bias correction
first) and sets per-league targets — see **Path to 75% — feature/bias track** below.

| League | passes bias gates | bias + BSS ≥ 0 | 75% target | gap |
|---|---|---|---|---|
| NBA | 14/21 | 13/21 | 16 | −3 |
| WNBA | 10/18 | 10/18 | 14 | −4 |
| NFL | 17/20 | 16/20 | 15 | **MEETS (+1)** |

The Step-1 screen reports crippled-HP bias-pass counts (e.g. NFL 17/20). Three NFL
cells (rushing-yards, qb-tds, sacks-taken) failed Tier-0 on the full-HP retrain, so
**NFL lands 13/20 confirmed at full HP**, short of 15/20 by 2.

> **Per-cell audit + locked-baseline BSS detail:** see context doc → Prior-phase
> results (Step 1 breadth baseline + Step 3 baselines-locked narratives, per-cell
> BSS lists, the WNBA TOR fix, and the NFL pruning detail).

Depth work (superseding baselines via the ≥ 5% bar, Track A tail-head, Track B
family builds) is **secondary** to reaching 75% breadth.

## Market cell status — Gate 1 / `devel` / `main`

Per-cell shipping status. Sourced from committed state only — locked baselines in
[`data/ship_config.json`](../src/sportstradamus/data/ship_config.json), roster in
[`training/markets.py`](../src/sportstradamus/training/markets.py) (`ALL_MARKETS`),
families in [`data/stat_dist.json`](../src/sportstradamus/data/stat_dist.json), and
the locked 2026-05-21 breadth audit. No verdict is recomputed here.

- **Gate 1** = passes the Tier-0 absolute gates. A locked baseline ⇒ Gate 1 passed.
- **`devel`** = baseline shipped to live beta (the `ship_config.json` entry).
- **`main`** = graduated on the 14-day Gate-2 live soak. **No cell has begun a soak**
  (`data/live_metrics_per_market.parquet` not yet populated), so every `main` is ⏳.
- Unbaselined cells (no `ship_config.json` entry) are the **Step-2 feature/bias-track
  (B1.6)** work pool.

**Baselined (full-HP locked, Step 3): NBA 13/21, WNBA 10/18, NFL 13/20.** 75% target
/ gap: NBA 16 (**−3**), WNBA 14 (**−4**), NFL 15 (**−2**). The Step-1 screen reports
the crippled-HP bias-pass counts (e.g. NFL 17/20), three of which (NFL rushing-yards,
qb-tds, sacks-taken) failed Tier-0 on the full-HP retrain — Step 3 is the real state.

**NBA — 13/21 baselined**

| Market | Family | Baseline strategy | Gate 1 | `devel` | `main` | Next step |
|---|---|---|---|---|---|---|
| MIN | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| PTS | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| REB | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| AST | SkewNormal | — | ❌ | — | — | Step-2 (B1.6) |
| PRA | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| PR | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| RA | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| PA | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| FG3M | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| fantasy points prizepicks | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| FG3A | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| FTM | ZINB | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| FGM | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| FGA | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| STL | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| BLK | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| BLST | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| TOV | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| OREB | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| DREB | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| PF | ZINB | — | ❌ | — | — | Step-2 (B1.6) |

**WNBA — 10/18 baselined**

| Market | Family | Baseline strategy | Gate 1 | `devel` | `main` | Next step |
|---|---|---|---|---|---|---|
| MIN | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| AST | SkewNormal | — | ❌ | — | — | Step-2 (B1.6) |
| FG3M | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| PA | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| PR | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| PTS | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| RA | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| REB | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| OREB | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| DREB | SkewNormal | `centered_additive_mean10` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| FGA | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| BLK | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| STL | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| BLST | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| TOV | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| FTM | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| PRA | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| fantasy points prizepicks | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |

**NFL — 13/20 baselined**

| Market | Family | Baseline strategy | Gate 1 | `devel` | `main` | Next step |
|---|---|---|---|---|---|---|
| targets | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| carries | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| attempts | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| passing yards | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| rushing yards | SkewNormal | — | ❌ | — | — | Step-2 (B1.6) † |
| receiving yards | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| yards | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| qb yards | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| fantasy points prizepicks | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| fantasy points underdog | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| passing tds | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| tds | ZINB | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| rushing tds | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| receiving tds | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| qb tds | ZINB | — | ❌ | — | — | Step-2 (B1.6) † |
| completions | SkewNormal | `ratio_meanyr` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| receptions | SkewNormal | `centered_additive_mean10` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |
| interceptions | ZINB | — | ❌ | — | — | Step-2 (B1.6) |
| sacks taken | ZINB | — | ❌ | — | — | Step-2 (B1.6) † |
| passing first downs | SkewNormal | `centered_additive_eb_meanyr_k10` | ✅ | ✅ 2026-05-21 | ⏳ | Gate-2 soak |

† Passed the crippled-HP Step-1 screen but failed Tier-0 on the full-HP retrain
(Step 3). `sacks taken`'s `ratio_meanyr`+hurdle plumbing works end-to-end (pickle
`is_hurdle=True`), but the cell's star-bias still exceeds the bound — these three
join the Step-2 pool.

## Phase status index

Compact one-row-per-phase index. Full result prose, numbers, and verdicts for every
phase are in the context doc → **Status / progress log (detailed)**.

| Phase | State | Summary (≤ 15 words) |
|---|---|---|
| **P0 — offline eval harness** | ✅ done (PR #46) | `compression_eval.py` + golden test; full poetry gates need a normal-network run before merge. |
| **P0.5 — determinism gate** | ✅ done (PR #46) | Opt-in `meditate --deterministic`; RNGs pinned, fixed params, writes redirected; P1 unblocked. |
| **P1 — centered-target bridge (SkewNormal)** | ✅ done | **FGA-only SHIP, family-wide KILL**; default stays `ratio_meanyr`. |
| **P2.A — `init_score` baseline** | ✅ closed | **DEAD** — byte-identical to plain NegBin; no compression on count-branch mean. |
| **P2.B — HurdleZINB (derived-π gate)** | ✅ done | **6/8 NBA ZINB markets SHIP**; default stays `--zinb-mode=joint`. |
| **Stage 0 — live-data instrumentation** | ✅ done (PR #46) | All five deliverables shipped; graduation lookups are now a parquet read. |
| **Stage B1 — ZTNB fix + routing diagnostics** | ✅ done | **ZTNB REFUTED; routing rescope delivered** (0/23 → ztnb; 13 cmp, 10 mzinb). |
| **Stage B1.5 — §7a likelihood-vs-features pre-check** | ✅ done | **FEATURES, not likelihood**; compression family-invariant; family build DEFERRED. |
| **Stage B1.6 — feature/bias track (breadth → 75%)** | 🔜 research done, build pending | Post-hoc mean-bias correction is the rank-1 lever; Tier-0 `verdict()` mode is the precondition; per-league targets set (2026-05-22). |
| **Stage A1 — SkewNormal ICC diagnostic gate** | ✅ done | **Family clusters AMBIGUOUS** (25 ambiguous, 10 eb, 1 tail); ICC alone does not route. |
| **Stage A1.5 — factor-ICC de-risk (T5 fork gate)** | ✅ done | **T5 KILLED wholesale**; A2 pivots to T3 tail head. |
| **Stage A1.6 — NFL position-split cleanup + WNBA test fix** | ✅ done | Write-side NFL position scoping; new NFL ICCs supersede A1; T11 entry added. |
| **P3–P10** | ⬜ | Superseded by T-method replacements; P10 GPBoost prototyped and failed deterministically. |
| **Docs rework — breadth-led reorg + roadmap sync** | ✅ done (docs-only) | Breadth North Star + per-cell status lead; depth Tracks A/B labeled secondary; roadmap synced. |

## Scope — leagues this plan covers

The training pipeline ships models for three leagues. Every method below applies to
all three unless flagged league-specific. Market counts (from `data/stat_dist.json`):

| League | SkewNormal markets | ZINB markets | Games/season per player | EB K (current) |
|---|---|---|---|---|
| NBA | 13 (PTS, REB, AST, PRA, FGA, MIN, PA, PR, FG3A, FGM, fantasy points) | 8 (FG3M, FTM, OREB, PF, STL, TOV, BLK, BLST) | ~82 | 10 |
| WNBA | 11 | 7 (NBA names minus PF) | ~40 | 10 (likely fine) |
| NFL | 12 (passing/rushing/receiving yards, attempts, carries, targets) | 8 (passing tds, tds, rushing tds, receiving tds, qb tds, interceptions, sacks taken, passing first downs) | ~17 (regular season) | 10 (almost certainly wrong; see caveats) |

NBA and WNBA share ZINB market names (same stat universe, PF dropped on WNBA). NFL's
ZINB markets have different names but the same structural problem (low-mean count
stats with zero inflation). `meditate --league {NBA,NFL,WNBA}` is wired in
[training/cli.py:36-38](../src/sportstradamus/training/cli.py#L36-L38); the per-league
dispatch loop ([training/cli.py:218-261](../src/sportstradamus/training/cli.py#L218-L261))
trains each market through the same `train_market` orchestrator. The compression_eval
harness is league-agnostic.

> **Cross-league testing policy, research handoffs, context, and the architectural
> principle** moved to the context doc (`gbdt_mean_regression_context.md`). The
> testing-policy summary that gates code work is restated here:

### Cross-league testing policy (applies to every method below)

Every change goes through two test phases before shipping:

1. **Smoke phase (start of work):** 1–2 representative markets per league. Track A:
   the canonical SkewNormal market (NBA: FGA + PTS; WNBA: FGA + PTS; NFL: passing
   yards + receiving yards). Track B: a SHIP and a KILL from the P2.B verdict (NBA:
   FG3M + FTM/STL; WNBA: FG3M + STL; NFL: highest-zero-rate ZINB + lowest). Must pass
   before further development.
2. **Full-verification phase (before any default-flag flip):** the compression_eval
   A/B on **every market in every covered league** using the affected distribution
   branch. SkewNormal-only changes touch 13+11+12 = 36 markets; ZINB-only changes
   touch 8+7+8 = 23. SHIP requires the universal decision threshold on every market
   in every league, or a per-league/per-market routing config documenting exceptions.

Deterministic test sets land under `data/test_sets/deterministic/{strategy}/` keyed
by filename (`{LEAGUE}_{market}.csv`), so per-league output is already separable — no
schema change needed.

## Ship incrementally — per-market graduation

**The unit of shipping is the (league, market) cell — never the stage or track.** The
moment a cell clears Gate 1 offline, promote it to the 14-day live soak immediately;
do not hold a well-behaved market hostage to failing ones. A well-calibrated STL in
production gathering live evidence is worth more than on a branch waiting for FG3M.

- A stage is "ready to ship" the instant **any** cell clears Gate 1 — ship those,
  keep the stage open on the rest. No batch; cells promote independently.
- Struggling cells stay on the track into the next stage **in parallel** — never
  blocking a graduated cell.
- Live data on shipped cells is itself an input: a cell that looks good offline but
  regresses live (Gate 2) teaches more than another offline iteration on the laggards.
- Corollary for the Stage B1.5 pivot: ship the cells that pass the cheap feature/bias
  fixes as soon as they clear Gate 1; reserve the expensive family build for the
  specific cells that *still* kill after.

This is the constructive half of the diminishing-returns rule: **ship what is good
now, stop when a cell is good live, spend freed effort only on cells that still fail.**

## Diminishing returns — stop-the-track principle

Each stage exists because the *previous* stage's verdict suggested more lift was
available; if live data shows the deployed model is already calibrated and profitable
on a market/league, **stop that track there and redeploy effort elsewhere**. The plan
is a backlog, not a queue. (If a stop/continue residual or cost analysis is ambiguous,
dispatch the `research-analyst` agent before deciding.)

**The break — active vs speculative tail.** This is the cut line that drives the
[roadmap](sportstradamus_roadmap_v2.md) phase split:

- **Pre-break (active).** Reach 75% breadth first — the Tier-0 audit code
  (`compression_eval.verdict()` absolute-only mode) then the **Stage B1.6**
  feature/bias track — then the core depth methods: **A2** (T3 tail-head), **A3**
  (calibration polish — orthogonal, stacks on A2; active, not tail), **B2** (routing +
  orthogonal feature engineering), **B3** (MZINB / marginalized-hurdle family build).
  The roadmap's leading "Active Track" phase.
- **Post-break (deferred — roadmap Phase 6).** **Stage A4** (novel risky retries, only
  if A2/A3 leave a gap), **Stage B4** (tuning/polish — optional), and any long-shot
  method. Bodies stay in the context doc (home of record); none was ever urgent.

See the context doc → **Decisions & trade-offs** for the full deferral list.

### Lifecycle: offline gate → production test → live gate → graduation

Every cell moves through three states; two gates control transitions. Both gates
appear in the same `check_graduation` table (Stage 0 deliverable 0.4).

```
not-shipped  ─[Gate 1: offline]→  in-production-test  ─[Gate 2: live]→  graduated
   ▲                                       │                                │
   └────── re-entry / revert ←─────────────┴────[live regression]───────────┘
```

### Gate 1 — Offline ship gate

Computed on the **held-out validation + test split** `train_market` already produces
(lines 547-553 in [pipeline.py](../src/sportstradamus/training/pipeline.py) + the
deterministic test_set CSVs). Two tiers: **Tier 0** sets a cell's *first* baseline from
the **absolute** rows only; **Tier 1** supersedes an *established* baseline and adds the
**relative** rows. **Quick-reference threshold values:
[docs/ship_gate.md](ship_gate.md) is the authoritative mirror.**

| Offline metric | Threshold | Tier | Where it lives |
|---|---|---|---|
| Top-mean-decile MAE on test split | ≥ 5% better than current baseline | Tier 1 only | `compression_eval --baseline … --candidate …` |
| Global MAE on test split | not worse by > 1% | Tier 1 only | `compression_eval` global summary |
| `brier_skill_score` (book baseline) | **Tier 0:** ≥ 0; **Tier 1:** not worse than baseline | both | `model_stats.parquet` for the candidate run |
| Bottom-quartile bias — **absolute** | over-prediction ≤ +30% of quartile mean (floor 0.10); under-prediction tolerated | both | `bottom_quartile_bias` / `bottom_quartile_mean` |
| Bottom-quartile bias — **relative** | not more positive than baseline | Tier 1 only | `verdict()` condition 4a |
| Top-decile bias — **absolute** | \|bias\| ≤ 30% of decile mean (floor 0.10), bidirectional | both | `top_decile_bias` / `top_decile_mean` |
| Determinism gate (when changing the deterministic-mode pipeline) | green for every league with cached parquets | both | `tests/integration/test_determinism_gate.py` (currently NBA-only; Stage 0 extends to WNBA + NFL) |

**Tier 0** (priority): the "both" rows clear; the highest-BSS full retrain among all
candidates (incl. incumbent) sets the baseline and is promoted to the **mandatory
≥ 14-day soak**. **Tier 1**: all rows clear on every cell in every covered league (or
routing config records exceptions). On promotion the previous pickle is archived under
`data/old_models/` so revert is one cron-pull away.

### Gate 2 — Live graduation gate

Computed on the **last 30 days of settled production offers** by Stage 0's
`compute_book_brier_skill_score` and the rolling-window aggregator in `nightly.py`. A
cell graduates if all five hold:

| Live metric | Threshold | Where it lives |
|---|---|---|
| Settled book-BSS (30d, ≥ 200 offers) | ≥ 0 AND ≥ training `brier_skill_score − 0.02` | Stage 0 0.1/0.2 → `live_metrics_per_market.parquet` |
| Empirical vs predicted over-rate (settled) | within ±0.03 over ≥ 200 offers | same parquet (0.2) |
| Top-decile live MAE on settled bets | ≥ 5% better than prior-version live MAE, OR within 5% of offline test-set MAE | Stage 0 0.3 (`compression_eval --live-window 30`) |
| Bottom-quartile bias (one-sided) + top-decile bias (bidirectional) on settled bets | mirrors Gate 1 cond. 4 over ≥ 100 offers | Stage 0 0.3, `bottom_quartile_bias` column |
| Profit-sim parlay yield | non-negative on slates containing the cell | dashboard Stats Profit Sim; 0.2 aggregates per-cell |

On graduation, mark ✅ in the Status table with the graduating stage + triggering
metrics. The track continues on non-graduating cells only — the live-data analog of
P1's "FGA-only SHIP".

**Why both gates exist.** Gate 1 without Gate 2 is the failure P1 hit
(`centered_additive_eb_meanyr_k10` cleared offline on FGA under deterministic mode, but
production HP / live data could differ; the 14-day soak catches that drift). Gate 2
without Gate 1 is the inverse (shipping on live-data intuition with no offline verdict
to revert to). Every cell must clear Gate 1, soak, then clear Gate 2.

### Ship mechanism — per-cell strategy config on `devel`

The gates decide *when* a cell ships; this is *how* it goes live on the production
server (which tracks `devel`). Two invariants keep it safe:

- **Training is config-driven.** A git-tracked `data/ship_config.json` (nested
  `{league: {market: strategy}}`, mirroring `ALL_MARKETS`) assigns each cell one of
  three states:
  - a real strategy slug (one of `baselines.STRATEGY_SLUGS`) → **shipped**: train that
    cell with that strategy;
  - `"withheld"` → **under rework**: skip training **and delete** the cell's pickle
    (`data/models/{league}_{market}.mdl`) so it goes dark;
  - **absent** → **untouched**: train with the run's default strategy. An empty/missing
    `ship_config.json` is a strict no-op.
- **Inference is pickle-driven.** `model_prob` never reads `ship_config.json`; it
  decodes the strategy recorded *in the pickle it loaded* (pickles are
  self-describing). A missing pickle returns `[]` (market skipped). Training config and
  inference cannot drift — the server runs only the strategy baked into the pickle.

**Shipping a cell is a one-line PR to `ship_config.json` on `devel`,** landed by the
next weekly `meditate`:

```
absent ──(begin rework)──▶ "withheld" ──(Gate 1 pass)──▶ "<strategy>" ──(Gate 2 pass)──▶ graduated
  ▲          (default,        │ (dark,                      │ (live,                          │
  │           live)           │  pruned)                    │  new strategy)                  │
  └──(back to default)────────┘                             └◀──(Gate 2 live regression)──────┘
```

- **Withhold is deliberate and scoped.** Only an *explicitly* `"withheld"` cell is
  pruned; the prune is inline in `meditate`'s per-cell loop, so a scoped run
  (`meditate --market FG3M`) can only prune cells in its own scope — a stray dev run
  cannot dark-out the book.
- **New strategies are additive:** (1) a slug in `baselines.STRATEGY_SLUGS` with
  forward/decode functions, (2) a `model_prob` decode branch keyed on that slug, (3)
  the `ship_config.json` line. The decode branch lands *with* the strategy (Stage B1.6
  workstream 1). `load_ship_config` validates every value against
  `STRATEGY_SLUGS ∪ {"withheld"}` at startup so a typo fails `meditate` fast.

End-to-end: Gate 1 SHIP → edit `ship_config.json` → merge `devel` → next weekly
`meditate` retrains only that cell → `prophecize` scores it live → 14-day soak →
Gate 2. Revert-archiving the prior pickle under `data/old_models/` is orthogonal to the
withhold prune.

**Implementation note:** the plumbing — `training/ship_config.py`
(loader + `resolve_cell_strategy` + `WITHHELD`), `helpers/io.py` (`model_pickle_path` +
`prune_model_pickle`, deduping the inline path copied in `model_prob.py` and
`pipeline.py`), and the thin `meditate` wiring — was built in the main session (one
tightly-coupled feature). refactoring-specialist runs over every touched file before
any push.

**Shipping process.** The production server tracks `devel` and pulls the whole branch,
so a research branch is never merged wholesale. The two-phase model (one-time
foundation, then production-delta-only per-market PRs) and the keep/drop denylist live
in **CONTRIBUTING.md → "Shipping to Production (`devel`)"**. Every per-market ship PR
**must be carved by the `devel-ship-curator` agent**
(`.claude/agents/devel-ship-curator.md`), which enforces the denylist (no
`compression_eval` / `zinb-routing-diagnostics` / `icc-diagnostics` / `statsmodels`),
keeps the Gate-1 verdict as PR prose not committed code, and verifies the three gates.
The initial Phase A foundation PR is carved by hand (the one exception).

### Branches & model-promotion flow

Four branches map to the two gates:

- **`model-research`** — where candidate updates are developed and offline-tested
  (deterministic A/B, research scaffolding, cached test sets). All experimentation bloat
  lives here. *(Intended rename of `claude/fix-gbdt-mean-regression-GcY1g`; rename is
  **documented here, not performed** — it is entangled with PR #46 whose head ref is that
  branch, so renaming now would orphan the PR. Sequence after PR #46 merges, or retarget
  the PR first.)*
- **`devel-foundation`** — the trimmed clean "production shipping foundation":
  `model-research` minus the testing bloat (the denylist).
- **`devel`** — the **live beta**. Candidates passing **Gate 1** ship here and soak
  through the 14-day **Gate 2** window. The remote server tracks `devel`.
- **`main`** — the **final shipped models**. A cell graduates here once it passes
  **Gate 2** on live data.

```
model-research ─(build + offline-test)→ trim → devel-foundation
   ─(Gate-1 passers)→ devel [live beta / Gate-2 soak] ─(Gate-2 graduates)→ main
```

The `devel-foundation` / `model-research` →(Gate-1 passers)→ `devel` crossing is carved
by the **`devel-ship-curator`** agent: it branches off `devel`, brings only
production-runtime code + the single `data/ship_config.json` toggle for one cleared
cell, **hard-excludes** the research-scaffolding denylist (`compression_eval`,
`zinb_routing_diagnostics`, `icc_diagnostics`, `statsmodels`, `/tmp` harnesses), carries
the offline verdict as **PR prose not code**, verifies the three gates, and **never
pushes** (the human approves). It selects, excludes, verifies, and packages — Gate 1
already decided *whether* to ship.

### Track-wide stop condition

A whole track stops when **every cell has graduated**. Remaining staged work (e.g.
Stage B3 MZINB if Stage B1 ZTNB already graduated 8/8 ZINB cells) is filed under
"future research" — code in `src/deprecated/` if prototyped, otherwise noted in the
Status table as deprioritized with a one-line reason.

### Re-entry condition

If a graduated cell *regresses* — settled brier_skill_score below the graduation
threshold for two consecutive 7-day windows — it re-enters the track at the stage where
it graduated. Track work resumes on that cell only. Re-entry is logged in the Status
table with the regression metric.

> **Stage 0 — Live-data instrumentation** (the completed-build detail: what existed
> before Stage 0, the five deliverables in dependency order, and the Stage 0 ship gate)
> moved to the context doc → **Stage 0 — Live-data instrumentation**. Stage 0 is
> **done** (see the Phase status index / context status log).

## Inference-path compatibility (applies to every shipped change)

Every change must land its inference-side mirror in the **same PR** before promotion to
production. Gate 1 lets a change into the test-run window; the inference-path checklist
is what makes that window safe. This restates the architectural principle as a hard
requirement covering **every** change type and names the concrete seams.

### Inference path (where things live)

| Component | File / lines | Role |
|---|---|---|
| CLI entry | `prophecize` → `sportstradamus.prediction.cli` | Loads slate, dispatches per-market |
| Model load | [prediction/__init__.py](../src/sportstradamus/prediction/__init__.py) `main()` | Reads pickle from `data/models/{LEAGUE}_{market}.mdl` |
| Per-offer scoring | [model_prob.py:114](../src/sportstradamus/prediction/model_prob.py#L114) `model_prob()` | Runs once per offer |
| Per-offer features | [stats/base.py:597](../src/sportstradamus/stats/base.py#L597) `get_stats()` | Builds the `playerStats` row. **Inference mirror of `get_training_matrix`** — any new training feature computed identically here (leakage-safe). |
| Per-distribution decode | [model_prob.py:259-272](../src/sportstradamus/prediction/model_prob.py#L259-L272) (NegBin/ZINB/Gamma/ZAGamma) + [model_prob.py:276](../src/sportstradamus/prediction/model_prob.py#L276) (SkewNormal via `_decode_skewnormal`) | Turns `predict(pred_type="parameters")` into `Model EV` + `Model Gate` + `Model R/Alpha` |
| Hurdle dispatch | [model_prob.py:205](../src/sportstradamus/prediction/model_prob.py#L205) `getattr(model, "is_hurdle", False)` | The P2.B pattern for a non-default predict path. **Every new model class follows it.** |
| Distribution-specific blend | [helpers/distributions.py:69](../src/sportstradamus/helpers/distributions.py#L69) `get_ev`, [314](../src/sportstradamus/helpers/distributions.py#L314) `fused_loc`, [163](../src/sportstradamus/helpers/distributions.py#L163) `get_odds`, [425](../src/sportstradamus/helpers/distributions.py#L425) `set_model_start_values` | Every new `dist` name must round-trip through all four |
| Pickle schema | [pipeline.py:1940](../src/sportstradamus/training/pipeline.py#L1940) `_build_filedict` (writer) ↔ readers | Any new key added to writer AND read back by every consumer; legacy pickles load via `filedict.get("new_key", legacy_default)` |

### Per-change-type inference checklist

| Change type | Inference-side work | Precedent |
|---|---|---|
| **Training-only** (T6 FAGTB objective, T9 monotone constraint, B1 ZTNB likelihood, B4 per-parameter Optuna, B4 reduced regularization, B4 sample reweighting) | **None.** Output schema unchanged; pickle round-trips. | B1 ZTNB fix: only loss changes. |
| **New target/baseline strategy** (P1-style; future Track-A variants) | Inverse decode in `model_prob.py:_decode_skewnormal` via `baselines.STRATEGY_REGISTRY[strategy].decode_loc/decode_scale`; matching `*_Ratio` feature in `get_stats`; `target_strategy` + `offset_meta` keys round-trip. | P1 `centered_additive_*`. |
| **New distribution head** (T3 spliced/Pareto, T10 PGBM, B3 MZINB, B4 CMP, B4 quantile heads, T4 MEGB, T7 gbex) | (a) new decode block in [model_prob.py:259-272](../src/sportstradamus/prediction/model_prob.py#L259-L272); (b) `get_ev`/`get_odds`/`fused_loc`/`set_model_start_values` accept the new `dist`; (c) `dist` in `_build_filedict` + legacy fallback; (d) new live-path test mirroring [test_zinb_hurdle_live_path.py](../tests/integration/test_zinb_hurdle_live_path.py). | P2.B HurdleZINB. |
| **Post-hoc calibration object** (A3 isotonic on loc, T8 CQR/LCMQR, B4 isotonic on ZINB-mean) | Pickle as a new key (`isotonic`/`cqr`/`temperature` precedent); load in `model_prob`, apply after decode (before/after `fused_loc` per what's calibrated); byte-identical round-trip test. | `temperature` field ([pipeline.py:1958](../src/sportstradamus/training/pipeline.py#L1958)). |
| **New player-level feature** (B2 leakage-safe target-encoded `expanding().mean().shift(1)`) | Column added to BOTH `get_training_matrix` and [get_stats](../src/sportstradamus/stats/base.py#L597), computed identically, leakage-safe; same dtype/index; add to `feature_filter.json` whitelist. | `MeanYr`/`Mean10`/`*_Ratio` (`base.py:676-702`), `test_meanyr_mean10_leakage.py`. |
| **Multi-head factorization** (T5-basketball, T5-NFL) | `prophecize` loads N pickles/market; `model_prob` Monte Carlos (sample each factor, multiply, derive marginal); `fused_loc` may need a multi-output blend; new `factor_pickles: dict[str, Path]` on the parent pickle; new live-path test. **Expect the largest inference-side change in the plan.** | None in-repo; closest is per-market book_weights. |
| **Different model class** (T2 CatBoost ordered TS, T4 MEGB native R, B3 GPBoost) | New `is_catboost`/`is_gpboost` flag; `model_prob` + `prediction/__init__.py` load path branch; determinism gate extended; adapt if no LSS `predict(pred_type="parameters")` API. | P2.B `is_hurdle`. |

### Inference-path test as a hard ship gate

Any change requiring inference-side work must have a passing live-path integration test
under `tests/integration/` **before promotion to production** (before Gate 1 is even
checked). The test asserts:

1. `Model EV` finite for every offer.
2. For ZI-class distributions: `Model Gate ∈ [0, 1]`.
3. Two runs with `DETERMINISTIC_SEED` produce identical predictions.
4. Legacy pickles (without the new keys) still load and predict — the
   `filedict.get(key, default)` contract from P2.B.

If the inference test does not exist, the change cannot ship (the
OVERCONFIDENCE_INVESTIGATION §3.4 live-path-confound lesson: offline A/B verdicts are
meaningless if the change crashes or silently drifts in `prophecize`).

### Pickle-schema discipline (where train/predict drift hides)

The pickle dict written by
[pipeline.py:1940 `_build_filedict`](../src/sportstradamus/training/pipeline.py#L1940)
is the train↔inference contract. Every new field needs: (1) a reader site in
`model_prob.py` (or wherever consumed); (2) a legacy-default fallback
(`filedict.get("new_key", "joint")` — the P2.B `zinb_mode` pattern); (3) a round-trip
test asserting byte-identical predictions.

Fields written as of commit `77e4a41`: `model`, `step`, `stats`, `metrics`,
`diagnostics`, `params`, `distribution`, `cv`, `std`, `temperature`, `dispersion_cal`,
`weight`, `r_book`, `hist_gate`, `shape_ceiling`, `normalized`, `offset_meta`,
`target_strategy`, `zinb_mode`, `is_hurdle`, `expected_columns`. Any addition lands in
this list and gets the three-step contract.

## Path to 75% — feature/bias track (Stage B1.6)

The operative next step toward breadth. Per the Stage B1.5 §7a verdict, top-decile
compression is a **mean-head + feature** problem, not a likelihood problem, so Stage
B1.6 attacks those two layers cheapest-first and **ships per market the instant a cell
clears Gate 1** (per "Ship incrementally"). It **supersedes the family build (B2
routing + B3 fork) as the next step**; the family build is deferred behind a re-entry
condition. Target: the bias-failing low-mean count cells (NBA −3, WNBA −4, NFL −2).
Lock the eval protocol before refitting [29].

**Three workstreams, sequenced cheapest/highest-leverage first — (1) → (2) → (3):**

1. **Post-hoc mean-bias / dynamic-range correction** (NEW; the only piece targeting
   the actual mechanism). Fit `ŷ_corrected = f(ŷ)` on validation (ROE or EDM [3]),
   apply to the predicted mean **before** `fused_loc`. Symmetric: pulls bottom-decile
   over-prediction *down*, serving the new bias gates; recovers ~half the
   FG3M/rushing-tds gain by construction. New pickle key (e.g. `bias_correction`); apply
   step in `model_prob` before [distributions.py:314](../src/sportstradamus/helpers/distributions.py#L314); one round-trip test.
2. **Leakage-safe target-encoded player features** (pulled forward from B2).
   `groupby(player_id).expanding().mean().shift(1)` for stat and stat×opponent; new
   columns in [get_stats](../src/sportstradamus/stats/base.py#L597); same leakage audit
   as MeanYr/Mean10.
3. **Opponent-defense interaction + garbage-time/blowout flag** (B1.5 Finding 2).
   Player stat × opponent defensive profile (`profile_market`) + a blowout/garbage-time
   flag; addresses the FG3M low-volume overshoot. Same `get_stats` mirror, leakage-safe.

**Decision points / validation:** after each workstream re-run `compression_eval` per
cell; promote every cell clearing Gate 1's five conditions to the 14-day soak. **Validate
(1) does not worsen `brier_skill_score`** (CRPS/log-score + brier_skill, not MAE alone —
Open question #12). Build CMPμ / marginalized hurdle **only** on a cell that still kills
after B1.6 (conditional Dunn–Smyth RQR variance < 0.70 AND Poisson tracking the top
decile while NB compresses).

**Research verdict (2026-05-22, research-analyst).** The gap is a **bias-gate**
problem; workstream (1) post-hoc correction is the **rank-1 lever** — the only method
that moves both gated bands by construction — ahead of the feature work. Prefer
**isotonic-on-prediction** where the decile miscalibration is curved (over-low /
under-high); per-decile multicalibration [44] is the formal frame but a fallback for a
band *just* outside, not the lead (trained GBDTs are often near-multicalibrated [45]).
**Hard precondition (Step 0):** build the Tier-0 absolute-only
`compression_eval.verdict()` mode and run the read-only per-cell triage audit — the
+3/+4/+2 counts are unmeasurable until it exists. **Ordered:** Step 0 → (1) post-hoc →
(2) player feature → (3) opponent features → (4) gate-tuning; ship each cell to Gate-2
the instant it clears Tier-0. **Per-league targets:** NBA **TOV / BLST / OREB / PF**
(STL, FG3M backups); WNBA **TOV / BLST / OREB / STL** (re-validate — half the games make
isotonic tails + the expanding-mean feature noisier; fall back to affine ROE / strong
shrinkage); NFL **rushing-tds + receiving-tds** via **affine ROE** (not isotonic /
per-decile — too few positive events at interceptions ~0.5, TD zero-rates 0.78–0.92 [48];
keep rushing-yards / qb-tds / sacks-taken on the Track-A T11 bench). **Gate-tuning (4)** —
widen the Tier-0 BSS floor toward −0.02 only for the ~1 bias-passes/BSS-fails cell per
league (NBA PF, NFL interceptions); widen the constant, not per-cell, and flag for
Gate-2. **Note:** NBA/WNBA **AST are SkewNormal**, not count cells — post-hoc + player
feature apply, but count-family reasoning does not (treat as Track-A). New refs [43]–[48].

> **Full Stage B1.6 rationale (incl. the 2026-05-22 research verdict in detail), the §7a
> evidence behind it, and all of Track A / Track B depth work** live in the context doc →
> **Track B → Stage B1.6 — breadth research verdict** and **Two-track depth plan** (Track
> A SkewNormal: A1/A1.5/A2/A3/A4; Track B ZINB: B1/B1.5/B1.6 full body/B2/B3/B4).

## Cross-league caveats (read before running any cross-league A/B)

1. **NFL sample sizes are an order of magnitude smaller than NBA** (~17 vs ~82
   games/player/season). EB(MeanYr, K=10) is aggressive shrinkage at that size — A1 should
   re-derive `K = σ²_within / σ²_between` per league. Expect NFL K much lower (or the EB
   transform to fail on form-volatile NFL markets — file as an A1 finding).
2. **NFL stats are position-locked in a way basketball isn't.** `Player position` is
   already a categorical ([pipeline.py](../src/sportstradamus/training/pipeline.py)
   `_step_build_splits`). Track-A methods training one cross-player model per market may
   not transfer cleanly (a QB and WR don't share "passing yards"); compute the A1 ICC table
   *within position* for NFL where relevant.
3. **WNBA shares NBA's structure but has half the games/season.** EB K=10 probably fine but
   verify in A1. The per-100-poss factorization (T5-basketball) transfers exactly. **A1.5
   update:** T5-basketball is KILLED, so the transfer point is moot. WNBA's own factor-ICC
   verdict is **low-confidence**: WNBA has no `FGM` or `FG3A` *markets* (confirmed against
   `stat_dist.json` — WNBA set is MIN/AST/FG3M/PA/PR/PTS/RA/REB/OREB/DREB/FGA/BLK/STL/BLST/
   TOV/FTM/PRA/fantasy), so true FG%/3P% are uncomputable and `PTS_per_FGA` (0.103) is the
   *only* clean efficiency factor — the WNBA gap (+0.456) rests on a single point. No regen
   path for FGM/FG3A; a real verdict needs **new test cases** from WNBA's actual markets.
   **A1.6 update — chosen replacements:** `FTM_per_FGA` and `FG3M_per_FGA` (distinct ICC
   regimes — free-throw-generation rate a stable role trait → higher/ambiguous; 3P
   make-rate dominated by sampling variability per [2] → low/tail), with `PTS_per_FGA`
   (0.103) as anchor. All computable from existing WNBA markets; **no code change**
   (`icc_diagnostics.py` already factors them). The NFL half (position-split cleanup)
   shipped in A1.6. The NBA/NFL fork does **not** depend on the WNBA gap.
4. **The compression_eval A/B harness is league-agnostic but file paths are
   league-specific.** Cached parquets at `data/training_data/{LEAGUE}_{market}.parquet`;
   deterministic test sets at
   `data/test_sets/deterministic/{strategy}/{LEAGUE}_{market}.csv`. The full-verification
   phase iterates the existing league loop — no harness rewrite.
5. **Determinism gate currently covers NBA only.** The two `test_deterministic_mode_*`
   tests use NBA_FGA + NBA_FG3M
   ([tests/integration/test_determinism_gate.py:37,102](../tests/integration/test_determinism_gate.py)).
   Before a cross-league change, add parallel assertions on WNBA_FGA + WNBA_FG3M + a
   representative NFL market — else the cross-league verdict is noise (P1's lesson, see
   `CENTERED_TARGET_NEGATIVE_RESULT.md`).
6. **For low-mean NFL markets** (interceptions mean ~0.5, sacks ~1.5), the ZINB diagnostic
   formulae in B1 may need to compute on `log(1+Y)` [22]; the asymptotic Vuong degrades
   badly at very low means. Wilson-Einbeck's non-asymptotic test should be the only one
   trusted for NFL interceptions/sacks.
7. **Two-track parallelism holds across leagues.** Track A and B touch different
   distribution branches and markets; workable in parallel per league. Shared resource is
   the read-only compression_eval harness.
8. **Low-mean conditional-dispersion diagnostics need a non-Pearson estimator — trust the
   Dunn–Smyth RQR over Pearson at mean ≲ 0.11** (extends #6). In B1.5 the df-corrected
   Pearson and the RQR variance diverged at very low NFL means (rushing-tds: Pearson 0.57
   vs RQR 0.96; interceptions: Pearson 1.38 vs RQR 1.04) — Pearson `(y−μ̂)²/μ̂` is unstable
   when many μ̂ ≈ 0, and RQR randomization is coarse when P(Y=0|x) ≈ 0.97. A future low-mean
   NFL pass should bootstrap the RQR variance and/or use deviance-based dispersion, and lean
   on Wilson–Einbeck for the marginal zero-modification call. (B1.5 §7a verdict.)
9. **Post-hoc bias correctors at NFL count means must be affine ROE, not isotonic /
   per-decile.** At interceptions ~0.5 and TD zero-rates 0.78–0.92 the top bin holds a
   handful of positive events; isotonic tails and per-bin (multicalibration) correctors
   overfit, and percent-calibration error is worst in low-base-rate groups [48]. Use the
   global affine `y ~ a + b·ŷ` form there; reserve isotonic / per-decile for the
   higher-mean NBA/WNBA count cells. (B1.6 research verdict, 2026-05-22.)

## Critical files

| File | Role | Key lines |
|---|---|---|
| [training/pipeline.py](../src/sportstradamus/training/pipeline.py) | target build, dist select, training, denorm, test_set dump | 245–324 (branch/target), 328 (`lgb.Dataset` — `init_score` injection), 341/394–409 (`set_model_start_values`), 345–346 (MeanYr monotone), 348–368 (Optuna search space), 439–452 (SkewNormal denorm), ~960/981 (test_set dump) |
| [training/report.py](../src/sportstradamus/training/report.py) | diagnostics → `training_report.txt`, `model_stats.parquet` | `ev_meanyr_corr`/`result_meanyr_corr` (~850), `write_model_stats` |
| [stats/base.py](../src/sportstradamus/stats/base.py) | baseline features + target; inference-time mirror | 597 (`get_stats`), 676–702 (`MeanYr`, `Mean10`, `*_Ratio`), 1005/1011/1082 (`Result`) |
| [stats/nba.py](../src/sportstradamus/stats/nba.py) | NBA `MIN`, `USG_PCT`, per-48 stats | 127–135, 359, 366 |
| [helpers/distributions.py](../src/sportstradamus/helpers/distributions.py) | `set_model_start_values`; `fused_loc` (book blend) | 425–504 |
| [skew_normal.py](../src/sportstradamus/skew_normal.py) | custom SkewNormal (location-scale, supports negatives) | 30–199 |
| [hurdle.py](../src/sportstradamus/hurdle.py) | HurdleZINB (Stage 2 ZTNB lives here for B1) | ~201 (NegBin loss for Stage 2) |
| [scripts/compression_eval.py](../src/sportstradamus/scripts/compression_eval.py) | **P0 harness** — decile table, compression ratio, run log, diff verdict | — |
| [prediction/model_prob.py](../src/sportstradamus/prediction/model_prob.py) | **Live-path confound** — where shipped strategies must survive end-to-end | SkewNormal decode, `fused_loc` w≈0.9 blend, `temperature`≈1.37 |
| [docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md](superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md) | Source spec for the **ZINB derived-π gate** fix (P2.B precursor) | Phase B "SUPERSEDED → derived-π" |

## Verification (every code session)

**Always-on quality gates** (every commit, session, PR):
- `poetry run ruff check src/sportstradamus/`
- `poetry run pytest tests/golden/` (incl. `test_compression_eval.py`)
- `poetry run pytest -m integration` (fake-mode, no network)
- Regenerate CLI help snapshots if `meditate` flags change:
  `REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py`
- Determinism gate (P0.5):
  `poetry run pytest tests/integration/test_determinism_gate.py -v -m integration`

**Smoke phase** (start of a new experiment, before full A/B): pick 1–2 representative
markets per league (see "Cross-league testing policy"); run
`meditate --deterministic --league {NBA,WNBA,NFL} --market <smoke-market>` per league ×
smoke-market; confirm no determinism blowup, sensible `compression_eval` output, no
smoke regression vs baseline. A smoke regression is a hard stop.

**Full-verification phase** (before any default-flag flip or `--zinb-mode` config
change): run the A/B on every market in every covered league for the affected branch
(SkewNormal: 36 cells; ZINB: 23). SHIP only if it clears the universal threshold on every
cell, OR the routing config records exceptions.

- **Inference-path test** (required for any change touching the prediction-side schema —
  every "Per-change-type inference checklist" row except training-only): a live-path
  integration test under `tests/integration/` loading the new pickle, running `model_prob`
  on a cached 100-row fixture, asserting (a) `Model EV` finite; (b) `Model Gate ∈ [0,1]`
  if ZI-class; (c) two `DETERMINISTIC_SEED` runs identical; (d) legacy pickles still load.
  Exists **before promotion to test production**.
- **Pickle round-trip test** (required for any new `_build_filedict` key): save, reload,
  assert byte-identical predictions on a cached fixture. Mirrors
  [tests/test_hurdle_zinb.py](../tests/test_hurdle_zinb.py) test 3.
- **Live-path gate** (catches end-to-end-only behaviors): the promoted strategy confirmed
  through [model_prob.py](../src/sportstradamus/prediction/model_prob.py) end-to-end (no
  `Model Skew`=NaN, EV not collapsed by the book blend), one representative market per
  league per affected branch.

**Cross-league determinism additions** (needed before full-verification is meaningful on
WNBA + NFL): add `test_deterministic_mode_is_bit_reproducible_wnba` (on
`WNBA_FGA.parquet`) and `_nfl` (on `NFL_passing-yards.parquet`) to
[test_determinism_gate.py](../tests/integration/test_determinism_gate.py), plus
`test_deterministic_mode_hurdle_is_bit_reproducible_wnba` / `_nfl` (WNBA FG3M + e.g.
`NFL_interceptions.parquet`). Without them the cross-league verdict is noise on the new
leagues (P1's lesson, `CENTERED_TARGET_NEGATIVE_RESULT.md`).

## Session handoff

### Per-session rules

- One strategy/experiment per session where feasible (aligns with CLAUDE.md "one module
  per subagent"); commit + push to `claude/fix-gbdt-mean-regression-GcY1g` and update the
  harness run log so the next session sees the scorecard history.
- Keep the default strategy = current production behavior until an experiment clears the
  threshold, so `devel`-tracking production is never regressed mid-project.
- Record each experiment's ship/kill verdict in the committed run log (not a scratch doc)
  and update the **Status / progress log** table.
- Track A and Track B can be worked in separate sessions / subagents; they share no
  mutable state beyond the harness. The B3 strategic fork is the only place a single
  decision blocks multiple downstream sessions.

### Phase-to-phase handoff prompts

Each completed phase produces the next phase's handoff prompt as part of its
Definition-of-Done, via the [prompt-engineer subagent](../.claude/agents/prompt-engineer.md)
(`subagent_type: "prompt-engineer"`). The agent's addendum documents the reading list, the
standard 10-section structure (opener / reading list / scope / locked decisions /
inference-path checklist / decision threshold / verification gates / branch state /
out-of-scope / definition-of-done), and the `/tmp/{stage}_handoff_prompt.md` convention.
Handoff prompts are scratch until accepted; on acceptance they move to
`docs/handoffs/{stage}.md`. The Stage 0 handoff lives at `/tmp/stage0_handoff_prompt.md`
after this plan revision (initial production by the prompt-engineer agent in commit
`{stage0-handoff-commit}`).

> **Research handoffs (in-repo)** — how to dispatch the `research-analyst` subagent,
> its input/output convention, and the distillation rule — moved to the context doc →
> **Research handoffs that fed this plan**.

### Tooling note: `gh` is a userspace install on this workstation

`gh` is **not a system package** — it lives at `~/.local/bin/gh` (installed 2026-05-19 via
the official static tarball). Future sessions must ensure `~/.local/bin` is on `PATH`; on
this workstation it already is (`~/.profile` / `~/.bashrc`), but a sandboxed/non-login
shell may not inherit it. If `gh --version` fails:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Authentication is a one-time local setup (`gh auth login` interactive, or `export
GH_TOKEN=…` from a PAT with `repo` scope). Agent sessions lack credentials by default — if
`gh api …` returns `HTTP 401`, the user needs to re-auth.

### Branch / PR / commit refs

See **"Branches & model-promotion flow"** for the four-branch pipeline (`model-research` →
`devel-foundation` → `devel` → `main`) and why the `model-research` rename is deferred
behind PR #46.

- Branch: `claude/fix-gbdt-mean-regression-GcY1g` (the intended future `model-research`).
- PR: #46 (→ `devel`); HEAD `fbec3cc` ("feat(ship): per-cell `zinb_mode` plumbing + lock
  13 NFL baselines (full-HP confirmed)").
- Earlier plan-rewrite HEAD: `6e913b1` ("docs: add research handoff for centered-target
  negative result").
- Latest shipped: 13 NBA + 10 WNBA + 13 NFL baselines locked in `data/ship_config.json`
  (`b5d2609` / `c9fcf01` / `fbec3cc`); P2.B HurdleZINB (`cee5625` ships
  `centered_additive_eb_meanyr_k10`); P1 follow-up `1d0e65e` adds `centered_additive_mean10`
  as the path-wide A/B counterexample.
- This breadth-led docs reorg + roadmap sync landed on `claude/roadmap-rework-docs-h13so`
  (cut off `fbec3cc`; PR → `devel`, stacked on PR #46).

---

**Decision history, prior-phase results, research verdicts, and references:** see
[`gbdt_mean_regression_context.md`](gbdt_mean_regression_context.md).
