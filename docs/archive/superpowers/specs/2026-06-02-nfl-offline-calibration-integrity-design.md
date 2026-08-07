# NFL Offline Win-Banking + Gate Integrity (Operation Ship 75, Route C)

- **Date:** 2026-06-02
- **Status:** Design approved; ready for implementation plan.
- **Home doc:** [`docs/operation_ship_75.md`](../../operation_ship_75.md)
- **Research briefs:** `/tmp/researcher_passer_volume_skewnormal.md`,
  `/tmp/researcher_passer_events_zinb.md`, `/tmp/researcher_receiver_rusher.md`,
  `/tmp/researcher_g1_power_leverfit.md`

## Problem

NFL is the entire remaining Ship 75 gap: NBA 18/21 and WNBA 14/18 both clear the
75% target; NFL sits at 9/20 (needs 15). Every one of the 11 failing NFL cells
fails **g1** (the paired-Brier-vs-book gate: the model's over/under probabilities
must beat the bookmaker's with a bootstrap CI upper bound below zero), and 8 of
11 fail *only* g1.

Four parallel research agents (passers/SkewNormal, passers/ZINB,
receivers+rushers, and a cross-cutting g1-power agent) reached a consistent
verdict that reshapes the plan:

1. **The doc's planned Step 2 lever (a per-player expanding-mean / EB feature) is
   dead.** Per-player level is already the top-SHAP signal (`Mean10` #1, the
   `MeanYr` family ~35% of total importance), already double-injected on
   `ratio_meanyr` cells, and already priced by the book (corr(Line, MeanYr)
   0.57–0.96). ICC ≈ 0.13–0.18 means only ~15% of variance is between-player
   level the book might miss; ~85% is within-player game noise neither side
   predicts. An expanding-mean feature is provably redundant.

2. **Blending is already on, and the gate already scores it.** The gated
   probability `P` is the `fused_loc` model+book blend (temperature- and
   prob-stage-posthoc-calibrated), not the raw model. The 11 cells fail g1
   *despite* blending.

3. **The blend weight is optimized on the wrong objective.** `fit_model_weight`
   maximizes the blended *full-distribution* log-likelihood on validation with
   `w ∈ [0.05, 0.9]`. It does not minimize the over/under Brier at the line —
   what g1 grades. Smoking gun: `receiving yards` sits at `w=0.90` yet is worse
   than the book at the line; `passing yards` / `attempts` / `completions` are
   pinned at the `0.05` floor (the optimizer wants pure book, and the forced 5%
   model makes the blend a hair worse than book).

4. **The g1 i.i.d. bootstrap is anti-conservative for repeated-player panels.**
   For high-ICC cells (Brier-diff ICC 0.06–0.18) it over-credits a *pass* by
   18–48% (design effect 1.4–2.2). It never makes a *fail* easier. Thin passes
   must be re-checked under a player-clustered bootstrap before they are trusted.

5. **The DFS reframe is infeasible offline for NFL.** Gating against the soft
   DFS line we actually bet (vs the sharp sportsbook consensus) is the genuine
   path to NFL breadth, but the archive's DFS odds (`book='Underdog'`/`'Sleeper'`)
   start 2026-03-16 and cover only in-season leagues — **zero NFL rows**, and
   there is no historical DFS feed to backfill. DFS-gating can only help NFL
   forward, from the 2026-09 season. Out of scope here.

The honest consequence: **NFL 15/20 is not reachable by offline modeling against
sharp, efficient lines.** Reweighting or post-processing cannot manufacture an
edge a sharp market lacks. The realistic offline ceiling is ~11–12. This spec
banks the offline wins that *do* exist and makes the blend and the gate honest,
so the follow-on Phase 2 (g1 loosening) and the future blending-strategy research
are justified rather than shortcuts.

## Goal & success criteria

Bank every honest offline NFL g1 win available now, with no cell deferral, and
lay the integrity + extensibility groundwork for Phase 2.

A cell is **won** when it newly passes all five Tier-0 gates on the offline
scorecard **and** survives the Component 4 player-clustered g1 recheck; it is then
promoted `withheld → devel` in `stat_meta.json` (counting toward the 75%
numerator). Expected: **+1 to +3 NFL** (passing-tds is the highest-confidence
win; interceptions and carries are marginal candidates).

Deliverables regardless of how many cells flip:
- Mean-stage post-hoc correction wired end-to-end (Component 1).
- A `blending` strategy seam, behavior-preserving (Component 2).
- Honest convolution-derived books for `qb-yards` / `qb-tds` (Component 3).
- A player-clustered g1 recheck gating promotion (Component 4).
- All behind green `ruff`, golden, integration (fake-mode), and determinism gates.

**Honest breadth accounting.** Immediate breadth rests entirely on Component 1.
Components 2–4 bank little or nothing now: Component 2 is behavior-preserving
scaffolding, and Component 3 will likely *confirm* `qb-yards`/`qb-tds` as walls
(it raises the bar). Their payoff is structural — they make Phase 2 (where the
wall cells ship as clean ties against an honest book) possible.

## Pipeline-stage symmetry

The per-cell config already split the legacy `strategy` into
`target_normalization` + `posthoc`. This spec adds `blending` as the third
orthogonal field. The three map onto three distinct pipeline stages:

| field | controls | pipeline point |
|---|---|---|
| `target_normalization` | how the GBDT target is reshaped for training | train / decode |
| `posthoc` | correction after the distribution is formed | mean-stage (new, Component 1) + prob-stage (already wired) |
| `blending` | how the model and book distributions are combined | `fit_model_weight` / `fused_loc` |

All three are validated in
[`training/ship_config.py`](../../../src/sportstradamus/training/ship_config.py)
`load_ship_config`.

## Component 1 — Wire mean-stage post-hoc (affine-ROE on μ)

*The immediate breadth lever.*

**Rationale.** Leaf-averaging compresses the model's conditional mean at the
extremes (research: `passing-tds` top-decile μ bias −0.443, `qb-tds` −0.504).
This compression is family-invariant — hurdle / NegBin / dispersion changes can't
touch it — so the only fix is a post-hoc mean correction. A two-fold affine ROE
on μ moved `passing-tds` g1 CI upper from +0.011 to −0.009 (BSS +0.060 → +0.122)
in the research replay.

**Reuse.** The corrector already exists and is dormant.
[`training/posthoc.py`](../../../src/sportstradamus/training/posthoc.py) defines
`MEAN_STAGE = {"roe_mean", "isotonic_mean"}` with full `fit_posthoc` /
`apply_posthoc` support, but
[`training/pipeline.py`](../../../src/sportstradamus/training/pipeline.py)
`train_market` only invokes the `PROB_STAGE` branch (lines 2226–2235). The work is
wiring `MEAN_STAGE`, not writing it, and it mirrors the fully-wired prob-stage
path exactly.

**Changes.**
- `train_market`: add a `MEAN_STAGE` branch at the decode → fuse seam (after
  `_step_decode_predictions`, before `_step_fuse_predictions`). Fit `roe_mean`
  on (validation decoded μ, validation `Result`); apply the fitted blob to the
  **test** decoded μ before fusion. Persist the blob through the existing
  `posthoc_blob` pickle key (pipeline.py:1280 already persists it generically) —
  no new key needed; a cell carries one `posthoc` slug, dispatched by stage.
- [`prediction/model_prob.py`](../../../src/sportstradamus/prediction/model_prob.py):
  add `_apply_mean_posthoc` mirroring the existing `_apply_prob_posthoc`
  (line 125), applied to decoded μ **before** `fused_loc`. Legacy pickles without
  the blob load unchanged (`filedict.get("posthoc_blob", None)`).
- Config: set `posthoc: "roe_mean"` in `stat_meta.json` for target cells,
  starting with `passing-tds`.

**Data flow.** decoded μ → `roe_mean` correct → `fused_loc` blend → dispersion +
temperature calibration → prob-stage posthoc (if any) → `P` → gates.

**Edge cases / guardrails.** `isotonic_mean` stays available but is *not* used for
NFL count cells (too few positive tail events for isotonic tails — research
ref 48); affine ROE only. The mean correction is fit on validation and rejected
per the Component 5 BSS guardrail if it regresses Brier skill.

**Tests.** Pickle round-trip carrying a mean blob; golden for `roe_mean`
fit/apply on a synthetic compressed cell; inference parity (legacy pickle without
blob; new pickle with blob) asserting the corrected μ reaches `fused_loc`.

## Component 2 — `blending` strategy seam (behavior-preserving)

*Integrity + extensibility now; breadth in Phase 2.*

**Rationale and the constraint it respects.** A Brier-at-the-line blend objective
is blind to the rest of the distribution: the weight that minimizes at-line Brier
can move the blended mean and spread freely, so it can sharpen g1 while regressing
g2 (star bias), g3 (bench bias), and g4 (predictive IQR), which read the mean and
spread. The current NLL objective avoids this by scoring the whole distribution.
Brier-at-line is therefore a candidate to *evaluate*, not adopt blind. This spec
builds the seam to make such evaluation possible later; it does **not** change any
objective.

**Changes.**
- Extract the current `fit_model_weight`
  ([`training/calibration.py`](../../../src/sportstradamus/training/calibration.py))
  logic into a registry keyed by a `blending` slug. Register `nll` as the sole
  entry **and the default**. Each strategy owns both its weight-fitting objective
  *and* its weight bounds — so the `_MODEL_WEIGHT_MIN = 0.05` floor becomes a
  per-strategy property rather than a global constant.
- Add the `blending` field to `stat_meta.json` and validate it in
  `ship_config.load_ship_config` alongside `target_normalization` and `posthoc`
  (default `nll` for cells without the field → **zero change to all 41 shipped
  cells**). Extend the `ship_config.py` module docstring's "strategy = combination
  of …" sentence to include `blending`.

**Explicitly not in this spec.** No `brier_line` (or any non-`nll`) strategy is
implemented, and the floor is not dropped. Those belong to the future
blending-strategy research session, which will implement and evaluate candidate
strategies against **all five gates** (the g2–g4 concern above). The existing
`supersede_verdict` (g1–g5 + paired Brier CI + paired Sharpe) is the guardrail
that prevents any future blend strategy from shipping while regressing g2–g4 on a
baselined cell. Phase 2 (g1 loosening) then consumes whichever strategy that
research selects — e.g. one with a dropped floor, which is what lets a no-signal
cell cleanly *tie* the book rather than be contaminated to slightly-worse.

**Rule-of-three note.** A strategy registry with a single entry is justified here
because it mirrors two existing sibling patterns (`baselines._STRATEGIES` for
`target_normalization`, the `posthoc` slug sets) rather than inventing
scaffolding, the user has explicitly scoped a follow-on with ≥2 further
strategies, and it maps to a real third pipeline stage.

**Tests.** Default `nll` reproduces current `fit_model_weight` output bit-for-bit
on a fixture cell (behavior-preservation); `ship_config` rejects an unknown
`blending` slug; a cell with no `blending` field resolves to `nll`.

## Component 3 — Convolution-derived honest books for `qb-yards` / `qb-tds`

*Integrity. Likely confirms walls rather than banking wins.*

**Rationale.** `qb-yards`, `qb-tds`, and `passing-first-downs` have no sportsbook
market to quote, so the archive carries a fabricated `p_book = 0.5` placeholder
(confirmed: `qb-yards` / `passing-first-downs` are 100% `p_book=0.5`; `qb-tds` is
~40% coin-flip). g1 against a coin flip is meaningless. Two of them are exact sums
of markets that *do* have books: `qb-yards = passing-yards + rushing-yards`,
`qb-tds = passing-tds + rushing-tds`. Convolving the sharp component books yields
an honest benchmark.

**Math.**
- `qb-yards`: the book side is symmetric, so the combined book is
  `Normal(μ_pass + μ_rush, σ²_pass + σ²_rush + 2 ρ σ_pass σ_rush)`, with **ρ taken
  from `NFL_corr.csv`** (negative — game-script pass/rush substitution — so an
  independence assumption would overstate variance). Closed form; the combined
  over-probability at a line is `1 − Φ((line − μ_sum)/σ_sum)`.
- `qb-tds`: numerical PMF convolution of the two count books; the game-script
  dependence is handled with the same `NFL_corr.csv` correlation via a shared
  game-script shift, or an independence approximation that is documented and
  cross-checked against Monte-Carlo.
- `passing-first-downs`: **no clean two-market decomposition** — left on its
  current book and documented as offline-unresolvable.

**Injection.** A preprocessing step sources the component book lines/EVs for the
same `(player, game_date)` from the archive (`get_line` / `get_ev`) and writes the
convolution-derived book probability — in the archive's `Odds` under-probability
convention, where book over = `1 − Odds` (per `scorecard._brier_inputs`) — for
`qb-yards` and `qb-tds`, replacing the `0.5` placeholder, before the gates run.

**Honest expectation.** This raises the bar (the model must now beat a sharp
combined book) and will likely confirm `qb-yards`/`qb-tds` as walls; it may
*lower* the NFL count before Phase 2 restores it. It is included for gate
integrity, not breadth.

**Risks / validation.** QB rushing-yards and rushing-tds book coverage may be
sparse (books quote these mainly for mobile QBs); the preprocessing must skip
offers lacking both component books rather than fabricate them, and coverage is
reported as a validation metric.

**Tests.** Convolution reproduces the component means; the Normal-sum closed form
matches Monte-Carlo within tolerance; ρ sign sanity (negative for pass/rush);
offers missing a component book are skipped, not zero-filled.

## Component 4 — Player-clustered g1 recheck

*Integrity. Only makes passing harder.*

**Rationale.** The i.i.d. paired-Brier bootstrap over-credits high-ICC
(repeated-player) cells by 18–48%. Any new g1 pass from Components 1–3 — especially
a thin one like `passing-tds` (CI upper ≈ −0.009) — must clear a clustered
bootstrap or it may be a false pass.

**Changes.**
- Persist `Player` and `Date` in `_step_persist_artifacts` (both survive in `M`;
  `Player` is currently dropped from the dumped test set, `Date` not carried
  through). This is the schema addition the home doc's §0.5 backlog calls for.
- Add a player-block (or game-week-block) bootstrap variant of
  `_gate1_brier_ci` in `training/scorecard.py`. Promotion of any newly-passing
  cell is gated on the clustered recheck in addition to the standard g1.

**Scope discipline.** This is an integrity gate on *promotion*, not a replacement
for the offline g1 and not applied retroactively to demote shipped cells in this
spec (that, like the global blend-objective question, is a separately-scoped
governance change).

**Tests.** The persisted test set carries `Player`/`Date`; the clustered CI is ≥
the i.i.d. CI on a synthetic repeated-player panel with positive within-player
Brier-diff correlation.

## Component 5 — Per-cell validation & rollout

**Guardrails.**
- BSS guardrail: reject any Component 1 mean correction that drops Brier skill
  score by more than 0.01 on validation.
- `supersede_verdict` on every change to a *baselined* (already-shipped) cell.
  First-ship of a currently-withheld cell (e.g. `passing-tds`) uses the Tier-0
  absolute gates, not supersession.
- The determinism gate
  ([`tests/integration/test_determinism_gate.py`](../../../tests/integration/test_determinism_gate.py))
  stays green; extend to NFL before any NFL cell ships.

**Rollout order.** `passing-tds` first (highest-confidence, proven in research),
then sweep `qb-tds` (after Component 3's honest book), `interceptions`, and
`carries` under the new levers. Promote `withheld → devel` in `stat_meta.json`
only after the Component 4 clustered recheck passes. `interceptions` carries a low
prior (μ nearly uncorrelated with realized INTs, slope 0.047) and may accept-stay
withheld.

## Out of scope (separate specs / sessions)

- **Phase 2 — g1 loosening revisit.** After this spec lands (honest books +
  clustered recheck in place), evaluate whether the strict `g1_ci_hi < 0` (beat
  the sharp book) can be relaxed toward "blend ≥ book" on the argument that tying
  the sharp book is still +EV on the soft DFS line we actually bet. Its own spec.
- **Blending-strategy research session.** Implement and evaluate `brier_line` and
  further blend strategies against all five gates; choose per cell. Consumes the
  Component 2 seam.
- **DFS-gate machinery and historical-DFS acquisition.** No historical NFL DFS
  data exists; a forward capability for the 2026-09 season.
- **New orthogonal-signal research on the yardage walls.** Low prior (single-game
  efficiency is largely unforecastable); not attempted here.

## Risks

| risk | mitigation |
|---|---|
| Component 1 affine-ROE overfits validation | fit on validation only; BSS guardrail; Component 4 clustered recheck |
| Component 3 independence error inflates the combined book variance | use `NFL_corr.csv` ρ; Monte-Carlo cross-check |
| Component 3 component (QB rushing) book coverage too sparse | skip offers missing a component book; report coverage |
| `qb-tds` count convolution complexity | numerical PMF convolution; document the dependence approximation |
| thin g1 passes are false (i.i.d. anti-conservative) | Component 4 clustered recheck is mandatory before promotion |
| `blending` registry seen as premature scaffolding | mirrors two existing patterns; single behavior-preserving entry; rule-of-three satisfied by the scoped follow-on |
| determinism flake under the new mean-stage step | extend + run the determinism gate before any NFL cell ships |

## Testing strategy

`poetry run ruff check src/sportstradamus/`, `poetry run pytest tests/golden/`,
and `poetry run pytest -m integration` (fake-mode, no network) must all be clean.
New golden tests per component as listed above. No new features are added, so the
leakage tests are unaffected. `refactoring-specialist` runs on every touched
`.py` before any push, per CLAUDE.md.

## References

- Home doc: [`docs/operation_ship_75.md`](../../operation_ship_75.md)
- Research verdicts + citations: [`docs/operation_ship_references.md`](../../operation_ship_references.md)
- Ship gate thresholds: [`docs/ship_gate.md`](../../ship_gate.md)
- Research briefs (this session): `/tmp/researcher_passer_volume_skewnormal.md`,
  `/tmp/researcher_passer_events_zinb.md`, `/tmp/researcher_receiver_rusher.md`,
  `/tmp/researcher_g1_power_leverfit.md`
