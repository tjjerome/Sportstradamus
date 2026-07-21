# Two-Part Fit-Size Floor Reduction

> Status: CLOSED — KILL. Stage-0 brief (`/tmp/researcher_two_part_fit_floor.md`) found
> `fit_players` is NON-binding; the `authentic_*`/settlement guards are the real wall and no
> floor reduction reaches the target cells. See §5 verdict. Both cells → accepted kills in the
> parent `two-part-grouping-gating` lane. Zero code.

## 1. Mission & money logic

The structural two-part group-conditional-CDF strategy carries a **fit-size support floor**
in `_TWO_PART_SUPPORT_FLOORS` — `fit_rows=1000`, `fit_players=150` per nested fit partition
(plus `authentic_rows=1000` / `authentic_players=100`). Unlike the per-group positive-map
floor (which the grouping lane addresses), this one pools across the **whole role tier**, so
no grouping change touches it. Measured on cached matrices (real production split), it is the
true gatekeeper for high-row / **low-player** cells: they have ample rows but too few unique
players to fill the 150-player floor after a 70/15/15 split + 5×5 nested CV.

This blocks otherwise-viable structural candidates. Each cell the floor reduction unblocks
(without breaking calibration validity) becomes a sweep+gate ship candidate — breadth the
grouping lane cannot reach.

This is a **calibration-mechanism change → research-gated** (CLAUDE.md §Agentic workflow
conventions). Dispatch `research-analyst` before editing `_TWO_PART_SUPPORT_FLOORS`.

## 2. Read first (in order)

1. `CLAUDE.md` §"Writing code in this repo", §"Research-first", §"Hard rules" — repo law.
2. `docs/handoffs/two-part-grouping-gating.md` — the parent lane; its §6 verdict + Stage-1
   acceptance depend on this lane for WNBA AST and NFL carries.
3. `/tmp/researcher_two_part_grouping.md` addendum **A0–A4** — the measured evidence below
   (all six cells instrumented on the real split).
4. `src/sportstradamus/training/group_conditional_cdf/_pipeline_steps_two_part_support.py`
   — `_TWO_PART_SUPPORT_FLOORS` and the `fit_size` / `authentic` guards. **The file to change.**

## 3. Measured evidence (brief addendum A0–A4; cached, real split)

The fit-size floor is what kills these; their per-group positive-map floors pass:

| Cell | val slice | fit floor result | status |
|---|---|---|---|
| WNBA AST | 2207 rows / 160 players | `fit_players_min = 102` (floor 150) | volume-close kill |
| NFL carries | 971 rows / 172 players | `fit_rows_min = 620`, `fit_players_min = 109` | volume-close kill |
| NFL passing yds | 369 rows / 59 players | fit-size **and** positive-map fail (~86 QBs total) | firm kill |
| WNBA DREB | thin league | fit floor fail | accepted kill |

WNBA AST and NFL carries sit *just* under 150 (102, 109) — the lever for them is more data or
a lower/redesigned player floor, nothing grouping does. NFL passing yds (~86-QB universe) is
firmer. The **NFL pilots clear 150 comfortably** (receiving codes 2/3/4 = 6793/3882/4248 rows;
rushing 2639/4109), so a lower player floor cannot perturb them — but that must be *verified*,
not assumed.

## 4. Locked decisions (owner; do not relitigate)

- **NFL pilots stay byte-identical.** They clear 150 comfortably, so a lower floor should not
  change their audit path — verify against the four golden pins in the parent lane §4.
- **Production-neutral.** No cell serves a structural strategy today. No `stat_meta.json` /
  pickle / config edits; the sweep+confirm+gate path is the only ship route.
- **Deterministic ≠ ship.** All iteration is `--deterministic` (sandboxed).

## 5. Stage plan

### Stage 0 — Research brief (entry gate)
- **Goal:** `research-analyst` brief answering: can the `fit_players` floor (and, if needed,
  `fit_rows` / `authentic_players`) be safely lowered — or replaced by a rows∧players criterion
  or a robust low-player-regime estimator — for high-row/low-player leagues, **without**
  breaking the "boundary intercept + temperature + non-positive stages well-estimated per
  nested fold" guarantee the floor provides?
- **Weigh:** how many unique players each pooled stage (per-tier boundary logit intercept,
  temperature, non-positive class) actually needs to be resolvable; whether the floor should be
  a fixed constant, a rows∧players conjunction, or league-volume-aware; the estimation-error
  basis for the chosen value.
- **Acceptance:** `/tmp/researcher_two_part_fit_floor.md` exists; recommends a value/criterion;
  states expected effect on **WNBA AST**, **NFL carries** (do they clear?), and **byte-identity
  on the NFL pilots** (unchanged?). **Kill branch:** if the floor is load-bearing at 150 and no
  lower value preserves validity, record the verdict and close the lane (those cells stay kills).
- **Verdict — KILL (brief `/tmp/researcher_two_part_fit_floor.md`).** Ran the full audit
  conjunction on the real split: `fit_players` is **not** the binding constraint. Both target
  cells fail a conjunction of guards, and relaxing `fit_players`/`fit_rows` leaves the
  **`authentic_*`/settlement** guards failing — WNBA AST misses the authentic-hold class floor
  by ~3× (31/100; players 11/50), the field-standard ≥100-event external-validation rule
  (Collins 2015; Van Calster 2016). The cheap fitted stages (boundary intercept, RB residual,
  temperature ≈ 1 param each) are over-floored but already clear, so their cheapness rescues
  nothing. **NFL carries** has an independent structural kill: its high role tier is zero-sparse
  (workhorse RBs almost always carry), starving `nonpositive_map_support` — a hurdle-family
  mismatch, not a floor issue. A floor edit would be byte-safe but unblocks nothing → **no
  code**. Levers if breadth needs these later (neither is this lane): **more data** — WNBA AST
  flips to reachable with no code change once its authentic base grows (re-instrument at season
  start); **a non-hurdle family** for carries (its own distribution-family gate).

### Stage 1 — Implement
- **Not executed — lane KILLED at Stage 0 (verdict above).** Original plan retained for record:
  edit the floor per the brief; re-instrument WNBA AST + NFL carries; keep the parent lane's
  golden pins green (ruff + golden + integration). Feed the result back to
  `two-part-grouping-gating.md` Stage-1 acceptance (flip WNBA AST / carries from contingent to
  reachable, or to accepted kills).

## 6. Escalation & stop conditions

- **DISPATCH research-analyst** before touching `_TWO_PART_SUPPORT_FLOORS` (the gated trigger).
- **STOP and ask the owner:** any floor edit not backed by the stage-0 brief; NFL pilot blobs
  drift; gates red at session start through no fault of yours.
- `refactoring-specialist` per the five CLAUDE.md triggers before any push/PR/review/done.
