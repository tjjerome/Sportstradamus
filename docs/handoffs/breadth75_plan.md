# Breadth-75 Campaign Plan (living)

Operational plan for the breadth-75 campaign end-game. This file is the execution queue —
what runs next, in what order, and what decides each fork. Verdict history and design detail
live in [model_improvement_track.md](model_improvement_track.md) (§6.6 + §10 ledger) and the
memory notes; this file cross-references rather than restates. Update this file in place
whenever the queue changes; git holds the history.

## Objective and owner rules

**Mission: ≥75% of cells shipped in every league.** Standing owner rules:

1. NFL is the highest-priority league until it crosses.
2. NFL receptions, targets, carries are **mandatory** members of the shipped set.
3. The three yards markets (passing / rushing / receiving yards) are the priority ships beyond
   the current candidate set — owner wants **all three**. Interceptions is the lowest-priority
   NFL cell (least popular market).
4. No denominator pruning. Bookless cells must beat the coin flip (g1–g3 vs synthetic 0.5
   stand as computed).
5. ≤5 distribution types in play (ZINB, NegBin, DPO, SkewNormal direct+centered). Adding a
   6th requires an owner ask with pilot evidence. Current standing: a 6th type is endorsed
   in principle, contingent on the compositional-architecture smoke test (NFL end-game
   step 4) — compound count×severity first; generic Mixture only as the fallback if the
   compound test shows no merit. ZAGamma stays retired (dominated — fixes the wrong defect).
6. The six offline ship gates are inviolable; deterministic boards rank, full-HPO confirms ship.
7. Never push devel directly — devel-ship-curator PR, human approves. Review stat_meta diffs
   after any `--yes` confirm.

## Scoreboard

| League | Shipped | Target (75%) | Status |
|---|---|---|---|
| WNBA | 14/18 | 14 | **DONE** |
| NBA | 16/21 | 16 | **DONE** |
| MLB | 14/19 | 15 | +1 needed |
| NHL | 10/15 | 12 | +2 needed |
| NFL | 9/20 | 15 | +6 needed — priority |

## In flight right now

- **Mixture pilot RUN — verdict: PROMISING on receiving yards, g1-walled on rushing.**
  Board-only, and now out of the sweep pool entirely: `SWEEP_CAPABILITIES` requires
  `CAP_CONFIRM`, so a family with no serving path spends no trial budget. It was the
  rank-1 corner on 3 of the first 4 holdout-blind cells and skipped at nomination every
  time; see model_improvement_track.md §8.2 hole #0a for the put-it-back trigger.
  Deterministic 6-corner Mixture boards vs the cells' SN boards:
  - receiving yards: best Mixture corner (centered_additive_mean10 / crps blend) is the
    cell's **rank-1 corner overall**, g4 pit_ks 0.0552 vs bar 0.050 (SN best 0.0629 —
    60% of the g4 excess removed), g1 CI-hi +0.0064 (SN best +0.0050), g2/g3/g5/g6 all
    pass. Deterministic undersells a new family (30 rounds, no HPO) — full-HPO confirm
    is the decider once serving lands.
  - rushing yards: Mixture also halves the g4 excess (0.0703 vs SN 0.0868) but g1 is a
    real wall (CI-hi +0.0143, bss −0.0245 vs SN −0.0027) — book-bound beyond the family.
  - Pre-registered sharp-book kill (g4 clears while g1 stays positive) NOT triggered.
  - Build hardening that made it trainable: component-scale clamp to
    [0.02, 20]×label-std (Hathaway-style — kills the σ→0 likelihood spike and the
    dead-head exp-overflow), stabilization None→L2, and per-fit floors
    min_child_weight≥0.1 / lambda_l2≥1.0 (near-dead components have ~zero hessian and
    unregularized Newton leaf steps explode).
  - Next: build the Mixture serve path (model_prob decode + blend + dashboard deep-dive
    + live-path integration test), then full-HPO confirm receiving yards.
- **NFL chain complete**: receptions + yards shipped to devel (committed) → NFL 9/20.
  Revert gates now known: carries g1, qb yards g3 (not g4 — bss +0.118), targets g2.
  Count-board candidates (targets NegBin +0.149, qb tds hurdle-ZINB +0.064) still queued
  for the post-fix re-run; receptions NegBin is moot (SN ship banked).

## NFL end-game (7 → 15)

1. **Six candidates through confirm** (in flight + re-run): the five SN confirms above, plus
   the count re-runs. If a confirm reverts, its recovery levers in order: g4-only calibrated
   auto-retry (already wired into meditate), `posthoc: cdf_recal_isotonic` (rung-C precedent),
   family swap to the cell's other board-passing family (receptions and targets ship both
   ways).
2. **Count-confirm crash fix** (first post-chain code window): confirm's cross-family persist
   must write `target_normalization: "none"` when the winning family is a count family —
   ship_config validation rejects an SN slug on a non-SN cell and validates the whole file, so
   one bad entry kills every meditate. Add golden. Flip scripts set the slug too.
3. **Count re-run**: re-flip receptions/targets/carries (dist=ZINB **and**
   target_normalization=none) → `model-strategy-sweep --dist-class count --league NFL
   --confirm --yes` → revert unshipped flips. Board evidence stands; only the confirms were
   lost to the crash.
4. **Yards trio** (research verdict; brief archived at
   `research/briefs/researcher_nfl_breadth_20260719.md`, gitignored dir, dev box): one
   priority, two different problems.
   - **receiving + rushing yards**: g4 fails on **upper-tail under-dispersion** (right-tail
     PIT mass 0.21–0.25 vs nominal 0.05; boom games land far more often than a single-mode
     SkewNormal tail allows). *(Superseded 2026-08-26: under the post-fusion corrector and
     the joint `(c, s)` calibrator the incumbent's far tail is now too heavy — cov80 0.851 —
     so the residual is asymmetry + support, not raw tail weight;
     `docs/archive/researcher_studentt_head.md`.)* NOT the zero atom — positive-only KS ≈ full KS, so
     ZAGamma/hurdle would fix the wrong defect. **Compound/compositional architecture
     (count × per-event severity) SMOKE-TESTED — PARTIAL/NO-MERIT, not the lever.** With
     real play-by-play severities and the pre-registered bar (right-tail PIT ≤ 0.10 AND
     KS-beat on the same rows): receiving yards right-tail 0.206 → 0.180 (~13% of the
     needed movement, still a g4 fail), rushing yards got WORSE (0.230 → 0.243). Root
     cause: the tractable compound assumes count ⊥ severity, but boom games are
     dependence-driven (pass-fest scripts raise catch count AND per-catch yardage
     together) — the independent compound structurally understates the joint upper tail.
     Compound is complementary-at-best; do not build it for yards.
     **Active lever: Mixture head (2-component normal) pilot on receiving + rushing
     yards** — owner-endorsed 6th type (endorsement was conditioned on smoke-testing
     compound first; done). Mixture.py confirmed in pinned lightgbmlss; build follows the
     DPO precedent; research brief satisfies the research-first gate. g1 is only marginally
     negative (bss −0.004 / −0.007) so the g4 fix has a plausible-not-assured g1 crossing;
     pre-registered kill: g4 clears but g1 stays positive ⇒ sharp-book KILL — bank the
     evidence, do not chase.
   - **passing yards**: NOT reachable this cycle — ~0 zeros (different cell), most
     efficiently priced NFL prop market, g1 needs ~88× the current n with a point estimate
     worse than the book. Routes to the pooling/TabPFN lane, not Mixture; honest KILL for
     now.
5. **Remaining pool**: attempts is the only favorable-point-estimate g1 cell (needs ~1.5×
   effective n — obtainable via EB-shrink partial pooling, not history; the Odds API has no
   pre-2023-05 NFL prop history). completions (8× n) and passing tds (5× n + g4) are
   efficient-book KILLs this cycle. interceptions — **correction to earlier framing**: its
   DPO/NegBin corners already PASS g1 (bss +0.068); the real wall is g4 over-dispersion on
   every corner. Lowest priority per owner; no structural spend.

**Ceiling arithmetic (brief §1):** 7 shipped + 6 candidates = **13/20 base**; 15 requires
the Mixture ask landing ≥2 of receiving/rushing yards, or EB-shrink rescuing attempts + one
thin-n cell. Confirm-revert risk ranking: targets (+0.0065 g4 margin) and yards (+0.0069)
thinnest; carries/qb-yards/receptions healthiest. Recovery per reverter: g4-only calibrated
retry → `cdf_recal_isotonic` (continuous cells only — NOT qb tds; the monotone CDF map
degrades low-mean counts).

## NHL queue (10 → 12), after the NFL chain

1. **saves** re-confirm — S4 unlocked its +0.011 corner (ratio_meanyr / dist_training_loss=nll
   / direct / crps): single-cell `model-strategy-sweep --league NHL --market saves --confirm
   --yes`.
2. **skater fantasy** — persist the −0.230 g4-only corner (centered_additive_eb_meanyr_k10 /
   crps / centered / nll) + `posthoc: cdf_recal_isotonic` + calibrated-HPO meditate; keep on
   5/5 else revert. Backup: goalsAgainst count retry (−0.118).
3. If both fail: Mixture pilot ask to owner (6th type) with full board evidence.

## MLB queue (14 → 15), after the NFL chain

1. **pitches thrown** re-confirm — S4 unlocked its +0.052 corner
   (centered_additive_eb_meanyr_k10 / nll / centered / crps): single-cell sweep --confirm.
2. Fallback: **pitching outs** manual walk of its two remaining +0.057 corners
   (centered_additive_mean10 / crps / direct / crps, then centered_additive_eb_meanyr_k10 /
   crps / direct / crps) — persist by hand, full-HPO meditate, keep on 5/5 else revert+prune.
3. runs allowed is dead this wave (g1-only, book-side — calibration levers do not apply).

## Debt / bookkeeping queue

- refactoring-specialist over every touched `.py` (sweep, confirm, cli, + the upcoming
  norm-none fix and tests) before any push/PR/review — then re-run the three gates
  (ruff, golden, `-m integration -n0` + touch integration_green).
- devel-ship-curator PR for the session's ship batch (~20 stat_meta flips + S4 + fixes).
  Never push devel directly.
- S1: drop the joint-ZINB sweep axis (0 shipping corners across all boards, best −0.166) —
  evidence complete once MLB/WNBA boards counted; one-line registry edit + pin updates.
- Golden-suite dashboard smoke `test_lab_modifiers_pairwise_isolation_updates_stale_pair`
  is a cross-file xdist flake (passes isolated and file-scoped) — apply xfail(strict=False)
  treatment if it recurs at the pre-push gate.
- Stale `stat_meta.*.bak.json` sweep backups: delete as they appear.
- WNBA margin board at idle (target already met).
- Ledger line + this file updated per session boundary.

## Operating laws (every session)

- `pgrep -af "[m]editate|model.strategy.(sweep|confirm)"` must be empty before any
  sweep/confirm/meditate launch.
- No training-code edits while any sweep/confirm chain runs.
- `--resume` is safe on board sweeps: it reopens each cell's Optuna journal and reuses prior rows
  per row, not per cell, rejecting any whose spec/controls/matrix identity has moved.
- Deterministic board slack oversells g4 — read direction, not magnitude; confirms decide.
  Root-caused 2026-07-30 (protocol-dominated; see ../archive/matrix-provenance-and-crossfit-gates.md §B):
  rank and read nominees by `discounted_slack`, treat `confirm_risk=high` corners (continuous
  family on integer target, g4 inside measured inflation) as coin flips regardless of slack,
  and calibrate expectations from the rollup's P(ship | slack band × family) table — raw slack
  ≥ +0.19 has shipped 1/9.
- Research-first: §8.2-flagged levers and distribution-family changes need a research brief
  (`/tmp/researcher_*.md`) before build.
