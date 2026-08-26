# Lane brief — shape-bound SN triage: stop losing confirms to family-misspecified cells

**Read first:** CLAUDE.md, [docs/ARCHITECTURE.md](../ARCHITECTURE.md),
[model_improvement_track.md](model_improvement_track.md) §6.1 (scale-bound vs shape-bound
block) + §6.6 + open-Q #9. This brief kicks off a fresh session; it is self-contained but those
docs are the context spine.

## Mandate

The shape-bound SN cohort: **24 confirm-ledger rows (20 SkewNormal)** whose PIT
central-coverage (cov50/cov80) sits near nominal yet g4 fails because the PIT defect wanders —
a shape no scalar σ, HP-selection policy, or (c, s) recal can move. The calibrated search-gate
is a proven no-op here (NBA PA: calibrated arm widened σ +34%, raised book-skill, still failed
g4). This lane does three things, in order of expected value:

1. **Centered-SN rung, per cell** — the only escalation rung with a live win behind it.
2. **Data-artifact audits** — some "shape-bound" reads are fixable measurement artifacts.
3. **Sweep exclusion** — cells that stay family-limited stop burning confirm attempts.

New distribution families are explicitly OUT of this lane's scope (see prior-art table).

## Workstream 1 — centered-SN rung (cheap, proven)

`sn_param: "centered"` re-parametrizes the SkewNormal head as (mean, sd, gamma1) — the control
exists, persists per-cell in `stat_meta.json`, and **NBA PTS ships with it**. The R2
continuous-family routing assigned each SN g4-failing cell to a cohort by normal-scores z of
the gate-matched PIT; its shape/kurtosis bucket is this lane's target list.

- The routing table came from the R2 research brief (`/tmp/researcher_continuous_family.md`).
  `/tmp` does not survive; if the file is gone, re-derive the split from
  `data/training/model_stats.parquet` — shape-bound ⇔ `g4_pass == False` with cov50/cov80 near
  nominal (the coverage triple is in the gate row); scale-bound cells (coverage below nominal)
  belong to the calibrated lever, not here.
- Per cell: one confirm through the normal supersession lane (deterministic board does not see
  `sn_param`; it is a confirm-time axis like `hpo_selection`). A shipping confirm persists the
  pin; a failing one records the attempt in the ledger and moves the cell to workstream 3.
- Do not batch: one cell per confirm, board-confident ordering (positive-slack first).

## Workstream 2 — data-appropriateness audit before family blame

Precedent: NBA AST's g4 defect traced to a floor-granularity artifact in the data, not the
family (memory `two_part_data_appropriateness_gate`). Before consigning a cell to workstream 3,
spend one cheap read per cell: PIT histogram against the discrete support of the stat
(half-point lines, small-integer outcomes, platform-clamped quotes). A lattice artifact has a
concrete data fix; a genuine shape defect does not. The zinb-routing diagnostics
(`poetry run pytest -m diagnostics`) and the decoded test CSVs are enough — no retraining.

## Workstream 3 — retire the unfixable from the confirm loop

Cells that fail the centered rung and pass the artifact audit are family-limited. Confirm
attempts on them are pure waste — part of the original "systemic g4 failure" pain was paying
for these retries. Change: encode the shape-bound list so the sweep/confirm machinery skips
them —

- `_g4_only_retry_wanted` (confirm path) must not fire the calibrated retry on a listed cell;
- the board must not nominate a listed cell for a g4-rescue-shaped confirm.

Keep the mechanism boring: a small explicit list or a per-cell stat_meta field, whichever the
existing code reaches most naturally (the swept-control plumbing has a known 3-site ripple —
memory `swept_control_three_site_ripple` — prefer a read-only list over a new swept control).
This is `model_strategy/` code: one module, subagent-scoped, golden-tested.

## Prior art — tried and dead, do not retry without new evidence

| Escalation | Verdict |
|---|---|
| SHASH / StudentT families on the g4-hard pilots | NO-GO (memory `cdf_recal_nonstationary_pit`) |
| StudentT conditional-scale head (gated right-skewed cohort) | Piloted NO-GO 2026-08-26 — redescending `loc` serves −25% mean; dominated by the ±8 skew cap (`docs/archive/researcher_studentt_head.md`) |
| Whole-CDF isotonic-PIT recal (Rung C) | Built; neither pilot survives (PA g1-walled, DREB g5-bound) |
| μ-conditional / higher-DOF post-hoc defaults | Killed (R7) |
| Book-skew shape borrow (§6.5 WS2) | Refuted, reverted |
| 2-component Gaussian mixture | Diverges without guardrails; guardrails exist but no ship |
| Sequential (c,s) → Rung C on FGA | Killed (Exp-5) |

Genuinely new families (skew-t, SHASH revisit, comp-PMF ladder) are research-gated: dispatch
`research-analyst` for a cited brief first, and only after workstreams 1–3 exhaust.

## Gates and conventions

- Any `.py` touched: `refactoring-specialist` before push/review, then the single authoritative
  gate run — `poetry run ruff check src/sportstradamus/`, `poetry run pytest tests/golden/`,
  `poetry run pytest -m integration -n0`.
- No pushes to devel from the lane; a human merges. Ledger + §6.6 in
  `model_improvement_track.md` get one-line verdicts per cell, caveman style.
