# Two-Part Grouping / Gating Generalization

> Status: QUEUED (entry: research-analyst brief on the grouping direction — see §6 stage 0)

## 1. Mission & money logic

Make the structural role×position two-part group-conditional-CDF strategy *fit* on
continuous cells that concentrate in a subset of roster positions (NBA AST, and every
analogous multi-position cell) instead of hitting its support-audit kill. The strategy is
already market-agnostic in code and trains through `model-strategy-sweep` (landed in
`cee7304`); the remaining gap is that its per-group support floors were frozen for the
NFL 3-position pilots, so 5-position leagues fail the audit even when every position has
ample data. Filling out these cells is pure breadth: each one that fits becomes a
sweep+gate ship candidate on markets we currently cannot even attempt structurally.

This is a **calibration-mechanism change → research-gated** (CLAUDE.md §Agentic workflow
conventions). Dispatch `research-analyst` before editing the gating floors or the group
construction.

## 2. Read first (in order)

1. `CLAUDE.md` §"Writing code in this repo", §"Research-first", §"Hard rules" — repo law.
2. `docs/handoffs/model_improvement_track.md` — the parent lane; this is a sub-lane of the
   structural-calibration breadth work.
3. The saved plan `~/.claude/plans/look-at-the-uncommitted-synthetic-jellyfish.md` — the
   generalization design. It already names **grouping ∈ {role-tier, position, none}** as an
   intended sweep axis (§Design 1) — option B below is that axis.
4. Memory: `two_part_data_appropriateness_gate` and `swept_control_three_site_ripple`
   (`~/.claude/projects/-home-trevor-Sportstradamus/memory/`) — the kill taxonomy and the
   posthoc-control wiring.
5. `src/sportstradamus/training/group_conditional_cdf/_pipeline_steps_two_part_support.py`
   — the frozen floors and the audit that raises. **This is the file to change.**
6. `src/sportstradamus/training/group_conditional_cdf/_config.py` — `discover_codes()`
   (the position-code source) and `positive_groups()` (role × codes → group keys).
7. `src/sportstradamus/training/role_specs.py` — `RoleSpec.role_score` (role tiering) and
   `boundary_residual_positions`; per-cell role columns.

## 3. Verify before you trust

Rule: if a command below contradicts this brief, the output wins — fix the brief in place
(minor) or stop and ask the owner (material). All runs are `--deterministic`, which
sandboxes to `data/test_sets/deterministic/` and `data/models/deterministic/` and never
writes `model_stats.parquet` or a served pickle (verified: NBA_PTS.mdl md5 unchanged across
a run).

**Confirm the commit is present:**
```
git log --oneline -1 cee7304   # feat(two-part): finish market-agnostic generalization + posthoc trainability
```

**The pilots still pass / behave as recorded** (matrices ARE cached, hyphenated —
`NFL_receiving-yards.parquet`, `NFL_rushing-yards.parquet`):
```
# receiving yards (two-part) — PASSES end-to-end (EXIT 0, writes sandbox CSV)
poetry run meditate --deterministic --bypass-withholding --league NFL --market "receiving yards" \
  --structural-strategy role-position-two-part-groupcdf-fixedlinear-v3 \
  --dist SkewNormal --target-normalization ratio_meanyr --dist-training-loss crps \
  --sn-param direct --blending-loss-fn nll --hpo-selection loss --stabilization None --posthoc none

# rushing yards (affine) — fits + OOF-scores, then fails gate6_recent under deterministic's
# weak fixed HPs (NOT the full-HPO ship path; do not "fix" from a deterministic result)
poetry run meditate --deterministic --bypass-withholding --league NFL --market "rushing yards" \
  --structural-strategy affine-groupcdf-bookpool-v1 \
  --dist SkewNormal --target-normalization ratio_meanyr --dist-training-loss crps \
  --sn-param direct --blending-loss-fn nll --hpo-selection loss --stabilization None --posthoc none
```

**Reproduce the NBA AST kill** (swap `--league NBA --market AST`, keep two-part strategy):
```
poetry run meditate --deterministic --bypass-withholding --league NBA --market AST \
  --structural-strategy role-position-two-part-groupcdf-fixedlinear-v3 \
  --dist SkewNormal --target-normalization ratio_meanyr --dist-training-loss crps \
  --sn-param direct --blending-loss-fn nll --hpo-selection loss --stabilization None --posthoc none
# → ValueError: two-part candidate failed support guard(s): positive_map_support, positive_holds_nonempty
```

**The kill is granularity, not missing data** — at the matrix level every NBA AST position
clears the floor (100 positive rows / 20 players):
```
python3 - <<'PY'
import pandas as pd, importlib.resources as ir
from sportstradamus.training.role_specs import _POSITION_LABELS
td = ir.files("sportstradamus.data") / "training_data"
df = pd.read_parquet(str(td/"NBA_AST.parquet"), columns=["Player position","Result","Player"])
pos = pd.to_numeric(df["Player position"], errors="coerce"); res = pd.to_numeric(df["Result"], errors="coerce")
for c in sorted(pos.dropna().unique()):
    m = pos==c; pm = m & (res>0)
    print(_POSITION_LABELS["NBA"][int(c)-1], "pos>0 rows", int(pm.sum()), "players", df.loc[pm,"Player"].nunique())
PY
# All 5 positions: 1050–4696 positive rows, 39–178 players — every one PASSES at matrix level.
```

**Instrument the actual audit** to see which nested-CV cell drops below floor (monkeypatch,
no source edit; run via CliRunner so the patch survives into the in-process train):
```python
import sportstradamus.training.group_conditional_cdf._pipeline_steps_two_part_support as S
_orig = S._two_part_support_guards
def patched(*a, **k):
    g = _orig(*a, **k); pos_support, pos_hold = a[4], a[5]   # positional args of _two_part_support_guards
    for name, v in sorted(pos_support.items()):
        print(name, "min_rows", v["minimum_rows"], "@", v["minimum_rows_partition"],
              "min_players", v["minimum_players"])
    return g
S._two_part_support_guards = patched
from click.testing import CliRunner; from sportstradamus.training.cli import meditate
CliRunner().invoke(meditate, ["--deterministic","--bypass-withholding","--league","NBA","--market","AST",
  "--structural-strategy","role-position-two-part-groupcdf-fixedlinear-v3","--dist","SkewNormal",
  "--target-normalization","ratio_meanyr","--dist-training-loss","crps","--sn-param","direct",
  "--blending-loss-fn","nll","--hpo-selection","loss","--stabilization","None","--posthoc","none"])
```
Observed (validation split = 2222 rows / 338 players, 10 groups = high/low × 5 positions):
7 of 10 groups fall below the `positive_rows>=100` / `positive_players>=20` floor in their
worst nested fold (thin ones reach 14–32 rows / 3–6 players; 3 groups even hit empty holds).
NFL receiving yards clears the same floors because its matrix carries only 3 positions
(QB never enters the receiving matrix) → 6 groups, each ~2× denser.

## 4. Locked decisions (owner; do not relitigate)

- **NFL pilots stay byte-identical.** Any grouping/floor change must leave the
  receiving-yards two-part blob and the rushing-yards affine blob numerically unchanged.
  Pin: `tests/golden/test_two_part_groupcdf.py`, `test_scorecard_two_part.py`,
  `test_structural_conditional_gate_serving.py`, `test_serve_decode_parity.py`.
- **Production-neutral.** No cell serves a structural strategy today (all withheld). No
  `stat_meta.json` / model-pickle / config edits as part of this lane; the sweep+confirm+gate
  path is the only route a cell ships (a separate, gated task).
- **Deterministic ≠ ship.** Deterministic uses fixed weak HPs; gate outcomes (esp. gate5/
  gate6) differ from full-HPO. Judge *the fit + support/OOF audit passing*, not the gate
  verdict, from a deterministic run.

## 5. Module footprint & canonical paths

Import via `sportstradamus.training.group_conditional_cdf` (per CONTRIBUTING §Package Map).
Editable in this lane:
- `_pipeline_steps_two_part_support.py` — `_TWO_PART_SUPPORT_FLOORS`, `_two_part_positive_support`,
  `_two_part_support_guards`, `_two_part_nested_support_audit`. **Primary.**
- `_config.py` — `discover_codes`, `positive_groups` (group construction).
- `_pipeline_steps_two_part.py` — passes `positions`/`residual_positions` into the audit + fit.
- `_boundary.py`, `_fit_two_part.py`, `_selection.py`, `_apply.py`, `_validation.py` — the
  fit/serve kernels that consume the discovered code set (recover-from-blob idiom: every
  entry point recovers the group set the blob persisted). If the code set becomes adaptive,
  the blob must persist the *kept* codes and serve must recover them (mirror
  `_blob_positive_groups` / `_blob_residual_positions`).
- `role_specs.py` — role tiering; only touch if introducing a role-only grouping mode.
- `structural_context.py` — builds `context["routes"]` (roles) + `boundary_residual_positions`.

Out of footprint (stop condition §8): `stat_meta.json`, model pickles, `model_prob.py`
serving beyond the recover-from-blob mirror, any distribution-family file
(`.claude/research_gated.txt`).

## 6. Stage plan

### Stage 0 — Research brief (entry gate)
- **Goal:** `research-analyst` brief comparing the three directions and recommending one.
- **Directions to weigh:**
  - **A. Adaptive code set** — drop/merge position codes whose per-group support fails the
    floor; fit the supported positions, fall back (role-only or whole-CDF `posthoc`) for the
    thin ones. Mirrors how receiving works (no QB group). Question: what fallback for a
    dropped position at serve, and does the blob→serve recovery stay clean.
  - **B. Grouping axis {role×position, role-only}** — let the sweep explore a coarser
    role-only grouping (2 dense groups) when position granularity fails. This is the axis the
    original plan already names. Question: bias/variance vs position conditioning.
  - **C. Roster-size-aware floors** — scale `_TWO_PART_SUPPORT_FLOORS` by group count /
    league roster. Question: does relaxing the floor break the statistical guarantee the
    floor exists to provide (each CDF map well-estimated per nested fold).
- **Acceptance:** `/tmp/researcher_two_part_grouping.md` exists, cites the bias/variance +
  serve-fallback + byte-identity tradeoffs, recommends A/B/C (or a hybrid), and states the
  expected effect on NBA AST and on the NFL pilots.
- **Est.:** 1 session. **Kill branch:** if research says the floors are load-bearing and no
  direction preserves calibration validity, record the verdict in §10 and close the lane.

### Stage 1 — Implement the chosen direction
- **Entry:** stage-0 brief cited. **Scope:** §5 primary files (one module per subagent).
- **Acceptance:** NBA AST clears the support audit and reaches the OOF gate audit (a gate
  verdict — pass or fail — not a support kill); NFL receiving/rushing pilots reproduce
  byte-identical (the four golden pins in §4 green); ruff + golden + integration green.
- **Est.:** 1–2 sessions. **Kill branch:** if AST completes but the pilots drift, revert and
  escalate — byte-identity is a locked decision.

### Stage 2 — Breadth sweep
- **Goal:** run `model-strategy-sweep` over the multi-position continuous cells now unblocked
  and record which reach a gate verdict.
- **Acceptance:** a board with structural candidates scored (not support-killed) for NBA/WNBA
  continuous cells; ship decisions stay in the normal sweep+confirm+gate lane.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > `model_improvement_track.md`
  > this brief > roadmap v3.
- The recipe a structural candidate needs is the full `_YARDS_CONTROLS` set; the sweep emits
  it automatically (`strategy_cli_args`). A bare `meditate` needs every flag (see §3) or it
  fails the recipe check on the cell's stat_meta defaults — this is expected, not a bug.
- `--deterministic` for all iteration (sandboxed). Never run a non-deterministic `meditate`
  on a served cell in this lane — it overwrites the production pickle.

## 8. Escalation & stop conditions

- **STOP and ask the owner:** any edit to `_TWO_PART_SUPPORT_FLOORS` values or gate
  tolerances that is not backed by the stage-0 brief; NFL pilot blobs drift; gates red at
  session start through no fault of yours; two sessions with no acceptance criterion moving.
- **DISPATCH research-analyst** before touching the floors or group construction (the gated
  trigger for this lane). Use `refactoring-specialist` per the five CLAUDE.md triggers before
  any push/PR/review/done. `devel-ship-curator` carves any devel-bound ship PR.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session.
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean **except** the known pre-existing
  `test_ship_gate_invariant::test_no_served_cell_fails_ship_gate` (partial dev-box
  `model_stats.parquet`; 11 unrelated cells; do NOT "fix" — it demotes production).
- `poetry run pytest -m integration -n0` clean, then `touch .claude/.state/integration_green`.
- One ledger line appended to §10; status updated if a stage boundary was crossed.
- Durable non-obvious lesson? Offer a memory capture.

## 10. Ledger (append-only, newest first, cap ~15)

- 2026-07-21 · stage 0 pending · Lane opened from the market-agnostic generalization
  (`cee7304`). Diagnosed the NBA AST kill as a floor/granularity artifact (10 groups vs
  NFL's 6), not missing data — instrumented audit shows 7/10 groups below the
  100-row/20-player floor in their worst nested fold. NFL receiving-yards two-part pilot
  passes deterministic; rushing-yards affine fails gate6_recent under weak deterministic HPs.
  Gates ✓/✓(1 pre-existing OOS)/✓. next: research-analyst brief on grouping direction A/B/C.
