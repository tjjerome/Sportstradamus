# Experiment Graduation — Bootstrap (two-part + affine into the sweep pool)

> Status: READY FOR THE `experiment-graduation-specialist`. This is the atypical
> FIRST invocation of that agent: two proven methods graduate together, from a
> partially-landed state, and the shared dispatch plumbing they need does not
> exist yet. Steady-state graduations (one freshly-passed experiment) follow the
> agent's own procedure without this bootstrap. Production-neutral: nothing ships;
> the sweep + gates + human decide per cell afterward.

## 1. Mission & money logic

Two calibration methods have passed as single-cell experiments:

- **two-part** group-conditional-CDF (`role-position-two-part-groupcdf-fixedlinear-v3`)
  — pilot `NFL / receiving yards`.
- **affine** group-CDF-bookpool (`affine-groupcdf-bookpool-v1`) — pilot
  `NFL / rushing yards`.

Per the intended experiment lifecycle (`.claude/agents/experiment-graduation-specialist.md`),
a method that passes on its pilot should **graduate into the sweep option pool** so
the per-cell sweep can try it on every market and let the gates decide where it
helps. The owner's chosen shape is the **full merge**: fold both methods into the
single per-cell calibration selector (`posthoc`, renamed `calibration` — §4.2) and
**retire the parallel `--structural-strategy` / `model_strategy_registry` axis** they
ride today, so one field names at most one calibration method per cell.

**Current state (verified this lane — read the code, not the brief's earlier drafts):**
- two-part is **already** market-agnostic *and already sweep-selectable* — a live
  `model_strategy_registry` structural spec (`_yards(TWO_PART_STRATEGY, …)`,
  [model_strategy_specs.py:226](../../src/sportstradamus/training/model_strategy_specs.py#L226))
  enrolled via the `--structural-strategy` / `_YARDS_SELECTOR` axis
  ([model_strategy_specs.py:25](../../src/sportstradamus/training/model_strategy_specs.py#L25)),
  applicable to every continuous cell (`role_registry_gated`). It self-killed only on
  multi-position cells; that granularity gap is **now fixed** (Lane A role-only fallback,
  committed `ea564a7`). It needs pool-dispatch + the axis retirement — no market-agnostic code work.
- affine is the genuinely un-graduated one — same `--structural-strategy` axis, but its fit
  logic hardcodes `AFFINE_POSITIONS = {1:QB, 3:RB}`, so every off-NFL cell routes to
  `pooled_fallback` / support-kill. It needs the full market-agnostic conversion (§5 Stage C).
- The `--experiment` / `resolve_experiment_selection` lever
  ([structural_strategies.py:44,71](../../src/sportstradamus/training/structural_strategies.py#L44))
  the brief's earlier drafts named as "the cage" is **vestigial** — zero callers. The
  `--structural-strategy` axis above is the real selector; the dead functions die in cleanup.

**Do not confuse the two `structural_calibration` names.** The persisted filedict/pickle key
`structural_calibration` (written [pipeline.py:1940](../../src/sportstradamus/training/pipeline.py#L1940),
read at serve [model_prob.py:1372](../../src/sportstradamus/prediction/model_prob.py#L1372); also
the specs' `legacy_model_key`) is **live and load-bearing** — it holds the fitted structural blob
and must survive the merge unchanged (it is also a B5 rename-collision to leave alone, §4.2). The
identically-named stat_meta *cell-selector field* `cell.get("structural_calibration")` inside
`resolve_experiment_selection` is part of the vestigial lever and dies with it.

Graduating both is pure breadth.

**Codify the METHOD, not the artifact** (the governing principle — see the agent
def). Do not reuse rushing yards' fitted coefficients anywhere; codify the
*procedure* that produced them and run it fresh per cell.

## 2. Why this first run is atypical (needs this bootstrap, not just the agent)

1. **Two methods at once.** They share the new dispatch plumbing (§4). Doing them
   together builds it once.
2. **Unequal starting states.**
   - *two-part is already market-agnostic* — [cee7304](../../src/sportstradamus/training/structural_context.py)
     generalized it (position codes discovered via `league_position_codes`, labels
     via `position_label`, role spec via `role_spec_for`; it already skips codes a
     market never fields). It needs **only** pool-exposure + dispatch + a
     non-regression pin. **No market-agnostic code work.**
   - *affine is NOT market-agnostic* — `build_affine_expert_context`
     ([structural_context.py:168-204](../../src/sportstradamus/training/structural_context.py#L168))
     hardcodes `AFFINE_POSITIONS = {1:QB, 3:RB}`
     ([structural_strategies.py:27](../../src/sportstradamus/training/structural_strategies.py#L27)).
     It needs the full conversion first (§5).
3. **The unified dispatch does not exist.** `posthoc` (light correctors,
   [pipeline.py:3545](../../src/sportstradamus/training/pipeline.py#L3545)) and the
   structural stage ([pipeline.py:3501](../../src/sportstradamus/training/pipeline.py#L3501))
   are separate; two-part even *requires* `posthoc="none"`
   ([structural_strategies.py:105](../../src/sportstradamus/training/structural_strategies.py#L105)).
   Letting a `posthoc` value route to the structural stage is net-new shared
   foundation (§4).
4. **Existing pins get replaced.** Receiving/rushing yards already carry byte-identity
   golden pins from the two-part lane. Convert them into the data-driven generic-path
   non-regression table (agent step 7), asserting the pilots reproduce **through the
   pool path**, not the retiring experiment lever.

## 3. Read first (in order)

1. `.claude/agents/experiment-graduation-specialist.md` — the charter you execute;
   §"codify the METHOD, not the artifact" and step 7 (non-regression) are binding.
2. `CLAUDE.md`, `docs/STYLE_GUIDE.md`, `CONTRIBUTING.md` — repo law + the three gates.
3. **The template diff:** `git show cee7304` — it is the two-part version of this
   exact job. `structural_context.py` (−32, the hardcoding removal),
   `role_specs.py` (+51, the per-cell role-spec registry),
   `model_prob.py` (serve generalization), `cli.py` (posthoc trainability). Mirror
   its shape for affine.
4. This lane's design (§4) + the coupling worklist (§6).

## 4. The design: `posthoc` becomes the unified calibration-method selector

`posthoc` is the live per-cell lever (`POSTHOC_SLUGS`,
[posthoc.py:35](../../src/sportstradamus/training/posthoc.py#L35): `none`,
`prob_recal_isotonic`, `prob_recal_platt`, `roe_mean`, `isotonic_mean`,
`cdf_recal_isotonic`) and is single-valued / mutually-exclusive by design
([posthoc.py:31-33](../../src/sportstradamus/training/posthoc.py#L31)). Add both
method slugs to that pool. The field then names **at most one** calibration method
per cell — a light corrector *or* a structural method, never both — which is exactly
the mutual exclusivity the owner asked for.

Because a structural method is not a post-distribution corrector (it reshapes the
target/CDF earlier in the fit), the field value must **dispatch**: a structural slug
routes to the structural stage (`build_*_context` + `fit_*_groupcdf` / apply); a light
slug routes to `posthoc.fit_posthoc`. This broadens `posthoc` semantically from
"post-distribution corrector" to "the per-cell calibration-method selector" — update
the module docstring + `POSTHOC_SLUGS` validation accordingly.

Mutual exclusivity is **already enforced today** and must be preserved, not invented: the
structural specs pin `"posthoc": "none"` in `_YARDS_CONTROLS`
([model_strategy_specs.py:167-176](../../src/sportstradamus/training/model_strategy_specs.py#L167))
plus `fixed_persist`, so a structural cell already can't also carry a corrector. After the
merge the single `calibration` field carries that invariant structurally (one value = one
method), and the `posthoc="none"` fixed-control is subsumed.

### 4.1 Retire the `--structural-strategy` / registry axis (the real cage)

The live selector is **not** `--experiment`; it is the `--structural-strategy` axis backed by
`model_strategy_registry`. The full merge retires it:
- **`cli.py`** — `_resolve_structural_strategy`
  ([cli.py:123](../../src/sportstradamus/training/cli.py#L123)), `_validate_structural_controls`
  ([cli.py:147](../../src/sportstradamus/training/cli.py#L147)), the `--structural-strategy` flag
  ([cli.py:491](../../src/sportstradamus/training/cli.py#L491)) and its resolve/validate call sites
  ([cli.py:596](../../src/sportstradamus/training/cli.py#L596),
  [749](../../src/sportstradamus/training/cli.py#L749)), plus any `--deterministic`-only structural guard.
- **`model_strategy_specs.py`** — the `_YARDS_SELECTOR` axis
  ([:25](../../src/sportstradamus/training/model_strategy_specs.py#L25)); the two `_yards(...)` specs
  ([:226](../../src/sportstradamus/training/model_strategy_specs.py#L226),
  [:241](../../src/sportstradamus/training/model_strategy_specs.py#L241)) stay as the pool method
  definitions (their `fixed_controls`/`persist`/`applicability`/`legacy_model_key` still hold — the
  fitted-state key), but their `selector` axis wiring is dropped.
- **`model_strategy_registry.py`** — the structural-enrollment + `role_registry_gated`
  (`BASE_STRUCTURAL_STRATEGY`) handling now that the pool, not the registry axis, selects.
- **`structural_strategies.py`** — delete the dead `validate_experiment_selection` /
  `resolve_experiment_selection` lever; keep the live exports (`ROLE_COLUMNS`, `AFFINE_POSITIONS`
  until §5 Stage C removes it, `AFFINE_STRATEGY`/`TWO_PART_STRATEGY`, the blob-key constants).

### 4.2 Rename `posthoc` → `calibration` — INVESTIGATED, ABORTED (owner call)

The field's meaning broadened (it now selects any calibration method, not only post-hoc
correctors), which motivated a rename. Recon killed it: **`training/calibration.py` already exists**
— a 610-line *model*-calibration module (book-weight fitting, model-blend weight, distribution
selection; exports `fit_book_weights`/`BLENDING_SLUGS`/`fit_book_shape`, imported by 5+ files) — and
"calibration" is already overloaded across `structural_calibration`, `stat_calibration.json`,
`get_market_calibration`, `apply_cdf_recal`, `dispersion_cal`. `posthoc`→`calibration` would collide
on the module filename and *worsen* the ambiguity, not fix it. The owner chose to **keep `posthoc`**
(B1 already broadened its docstring to "per-cell calibration-method selector", so it is documented
and unambiguous) rather than rename to a distinct token or rename the existing module. No rename
runs; `posthoc` stays the field/module/flag name.

## 5. Stage plan

**Stage A — Foundation (shared, build once).**
- Add `two_part` + `affine` slugs to `POSTHOC_SLUGS`; broaden the docstring/validation.
- Build the dispatch: `posthoc` value → structural stage vs corrector stage, at both
  train ([pipeline.py:3501](../../src/sportstradamus/training/pipeline.py#L3501)/[3545](../../src/sportstradamus/training/pipeline.py#L3545))
  and serve ([model_prob.py:638](../../src/sportstradamus/prediction/model_prob.py#L638) affine,
  [model_prob.py:811+](../../src/sportstradamus/prediction/model_prob.py#L811) two-part,
  [261](../../src/sportstradamus/prediction/model_prob.py#L261)/[289](../../src/sportstradamus/prediction/model_prob.py#L289) posthoc).
- Verify the three control sites stay coherent: `_CONTROL_FLAGS`
  ([model_strategy_specs.py:10](../../src/sportstradamus/training/model_strategy_specs.py#L10)),
  `runtime_controls` ([pipeline.py:3739](../../src/sportstradamus/training/pipeline.py#L3739)),
  the CLI resolve `_resolve_cell_knob`
  ([cli.py:112](../../src/sportstradamus/training/cli.py#L112)/[715](../../src/sportstradamus/training/cli.py#L715)).
  The `--posthoc` `Choice` ([cli.py:339](../../src/sportstradamus/training/cli.py#L339))
  auto-picks up the new slugs; confirm the sweep enumerates them.
- **Retire the `--structural-strategy` / registry axis (§4.1)** in the same stage — the pool is
  now the only structural selector, so the parallel axis and its CLI resolve/validate come out.
  The sweep must enumerate the two structural methods **from the pool**, not the retired axis.
- Scaffold the data-driven non-regression table (agent step 7): `(league, market,
  method, expected)` rows, driven through the generic pool path.

**Stage B — Graduate two-part (cheap; already agnostic).**
- Remove its `STRUCTURAL_STRATEGIES` entry + `validate_two_part_recipe`'s
  `posthoc="none"` coupling; select it via the pool instead.
- Add its pilot row (`NFL / receiving yards`) to the non-regression table; confirm
  byte-identical reproduction through the pool path.

**Stage C — Graduate affine (the real conversion).**
- Rewrite `build_affine_expert_context` to mirror `build_two_part_context`: discover
  codes with `league_position_codes(league)` (or `discover_codes` on the position
  column), label via `position_label(league, code)`, skip codes the market never
  fields, per-cell support guards — **delete `AFFINE_POSITIONS`** and every read of it
  ([structural_context.py:187-201](../../src/sportstradamus/training/structural_context.py#L187),
  [_pipeline_steps_affine.py:29](../../src/sportstradamus/training/group_conditional_cdf/_pipeline_steps_affine.py#L29)).
- Generalize the serve path: [model_prob.py:638-649](../../src/sportstradamus/prediction/model_prob.py#L638)
  must accept per-cell discovered experts, not a fixed QB/RB pair.
- Remove the `("NFL","rushing yards")` gate.
- Add its pilot row (`NFL / rushing yards`) to the non-regression table; confirm
  reproduction through the pool path.

**Stage D — Verify + gates.**
- Both pilots green through the generic path; `ruff` + `pytest tests/golden/` +
  `pytest -m integration -n0`; `refactoring-specialist` on every touched `.py`.
- Golden pins to convert/extend, not delete: `test_two_part_groupcdf.py`,
  `test_scorecard_two_part.py`, `test_structural_conditional_gate_serving.py`,
  `test_model_strategy_sweep.py`, `test_model_strategy_confirm.py`.

## 6. Affine market-lock worklist (the cage to remove)

This is the **affine-specific** cage (Stage C); the shared `--structural-strategy` axis
retirement is §4.1. Treat the line numbers below as pointers, not gospel — they predate this
lane's recon; confirm each against current code before editing (`validate_experiment_selection`
in particular is the vestigial lever that §4.1 already deletes).

| Site | What it hardcodes |
|---|---|
| [structural_strategies.py:16](../../src/sportstradamus/training/structural_strategies.py#L16) | `AFFINE_STRATEGY: ("NFL","rushing yards")` applicability gate |
| [structural_strategies.py:27](../../src/sportstradamus/training/structural_strategies.py#L27) | `AFFINE_POSITIONS = {1:"QB", 3:"RB"}` |
| [structural_strategies.py:44-68](../../src/sportstradamus/training/structural_strategies.py#L44) | `validate_experiment_selection` single-cell rejection |
| [structural_context.py:187,195,198](../../src/sportstradamus/training/structural_context.py#L187) | `AFFINE_POSITIONS` route/support/label reads |
| [_pipeline_steps_affine.py:29](../../src/sportstradamus/training/group_conditional_cdf/_pipeline_steps_affine.py#L29) | imports/reads `AFFINE_POSITIONS` |
| [model_prob.py:638-649](../../src/sportstradamus/prediction/model_prob.py#L638) | serve requires "QB/RB expert models" |

**Already done — do not redo, and do not trust the commit subjects.** A prior
partial generalization landed the naming and the sweep-*enrollment* side: `32a78f6`
("make the affine strategy a market-agnostic sweep candidate") only dropped affine's
sweep enrollment lock (one line in `model_strategy_specs.py`), and a rename series
(`cb05fdc`, `7b93f16`, `67498d8`, `3f0de44`) made affine's modules/symbols/persisted
columns market-agnostic. What still remains — despite `32a78f6`'s subject claiming
otherwise — is the fit **logic** (`AFFINE_POSITIONS`), the `validate_experiment_selection`
applicability gate, the serve QB/RB requirement, and the posthoc-pool migration (§4).
Read the code, not the log.

Generic mechanisms to reuse (already market-agnostic, proven by two-part):
`league_position_codes(league)`, `position_label(league, code)`,
`role_spec_for(league, market)` ([role_specs.py](../../src/sportstradamus/training/role_specs.py)),
`discover_codes` / `positive_groups`
([_config.py:93-100](../../src/sportstradamus/training/group_conditional_cdf/_config.py#L93)).
The engine config itself (`AFFINE_CONFIG`,
[_config.py:78-88](../../src/sportstradamus/training/group_conditional_cdf/_config.py#L78))
is already generic — the lock lives only in the expert-context + gate + serve layers.

## 7. Locked decisions & stop conditions

- **Production-neutral.** No `stat_meta.json` ship edits, no pickles, no config that
  serves a cell. Graduation only populates the *pool*; per-cell ship is the sweep +
  gates + human, later. All iteration `--deterministic`.
- **Pilots reproduce through the generic path.** If either pilot needs an
  `if league/market == …` branch to stay green, the generalization is wrong — fix the
  method, do not special-case (agent step 7).
- **Generalization, not redesign.** If the affine conversion tempts a *mechanism*
  change (new distribution family, new dispersion model), STOP — that re-triggers the
  research-analyst gate.
- **Kills stay rare + principled.** A per-cell affine kill must be a discovered
  data/structure impossibility, never a league/market literal. If it fires on more
  than a small minority of cells, the guard is wrong.
- **One agent, whole bootstrap.** Dispatch `experiment-graduation-specialist` on this
  brief. Do not ship a cell, do not graduate a third method.
- **Retire the axis (done); the field rename was aborted.** The full merge retired the
  `--structural-strategy` / registry axis (§4.1). The `posthoc`→`calibration` rename (§4.2) was
  investigated and **abandoned** — the target name is taken by an unrelated 610-line module and the
  concept is already overloaded; the owner kept `posthoc`.

## 8. Ledger

- 2026-07-21 · bootstrap COMPLETE · Stage A (foundation + `--structural-strategy` axis retirement,
  `308a564`), Stage B (two-part pilot pinned through the pool, `c40d9bc`), Stage C (affine made
  market-agnostic — `AFFINE_POSITIONS` deleted, per-cell code discovery mirroring two-part; both
  pilots pinned, `862e106`) all landed. Both methods are now single-valued `posthoc`-pool members
  the sweep tries on any market; production-neutral (0 stat_meta cells carry a structural value).
  B5 rename `posthoc`→`calibration` ABORTED (§4.2) — collides with the existing `calibration.py` +
  overloaded concept; owner kept `posthoc`. Gates green throughout (1 pre-existing OOS ship-gate).
- 2026-07-21 · premise corrected + scope expanded (Lane B B0) · earlier drafts named the
  vestigial `--experiment` / `structural_calibration`-cell-selector lever as the cage; the live
  selector is the `--structural-strategy` / `model_strategy_registry` axis (§1, §4.1). Full-merge
  target: fold both methods into the `posthoc`→`calibration` pool **and** retire the registry axis;
  the `structural_calibration` **pickle key** (load-bearing serve state) + four look-alikes are
  rename-collisions to leave alone (§4.2). Mutual exclusivity already holds via
  `_YARDS_CONTROLS.posthoc="none"`. Lane A (two-part grouping) DONE — two-part's multi-position
  self-kill fixed, so it needs only pool-dispatch, no market-agnostic work.
- Bootstrap authored for the first `experiment-graduation-specialist` run: graduate
  two-part + affine into the `posthoc` pool as the unified calibration selector;
  affine gets the cee7304 market-agnostic treatment; pilots re-pinned via a
  data-driven generic-path non-regression table. Zero ship.
