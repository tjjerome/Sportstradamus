# Gate-driven serving control — design

**Date:** 2026-05-22
**Status:** Design, pending user review of this spec.
**Author:** Claude (brainstorming session with Trevor)

## Problem

`ship_config.json` selects each cell's *training* strategy; it does **not** gate what
`prophecize` serves. Serving is "a pickle exists on disk for `(league, market)`" —
`model_prob` never reads `ship_config`, it decodes the strategy baked into the pickle it
loads (verified: `model_prob.py:183` reads `target_strategy`, dispatches via
`baselines.get_strategy`). The existing `WITHHELD` mechanism dark-outs a cell only when a
cell is *explicitly* `"withheld"` (meditate prunes its pickle); an **absent** cell trains
with the default strategy and **is served**.

Trevor wants a two-branch serving gate:

- **Beta (`devel`):** a cell serves only if it cleared **Gate 1** (the Tier-0 offline
  lock-in). Everything else is withheld (dark).
- **Primary (`main`):** a cell serves only if it cleared **Gate 2** (live graduation —
  `ship_gate.md`: ≥ 14-day soak, then settled-offer metrics over a 30-day window).
  Everything else withheld.

This is a **deliberate behavior change**: today unbaselined cells are `absent` → served with
the default strategy; under this policy they become explicitly `"withheld"` → dark. The
result is **default-deny**: only gate-passing cells serve. On `devel` today that darkens
60 of 96 `ALL_MARKETS` cells, leaving the 36 baselined live — the same outcome as the
manual pickle-prune already done on the dev box, made durable and automatic.

## Confirmed facts (verified this session)

- **Strategy travels in the pickle**, not the config. `target_strategy` + `offset_meta` are
  baked at train time; `prophecize` decodes via the baselines registry. No new work for
  strategy storage; no train/inference drift possible.
- **Serving = pickle-exists.** Gate = control which pickles exist. The lever is
  `ship_config` `"withheld"` → meditate `prune_model_pickle` → dark.
- **`ALL_MARKETS` = 96** (NFL 20, NBA 21, WNBA 18, MLB 22, NHL 15). Gate-1 passers today = 36.
- **`check_graduation` prints only** (no state file). Classification = `_classify_lifecycle`
  + parquet readers (`model_stats.parquet` Gate 1, `live_metrics_per_market.parquet` Gate 2).
  No live-metrics parquet exists yet ⇒ **0 graduated**.
- **`main` lacks the ship system** (`ship_config.py`/`baselines.py` absent; ~108 commits
  behind `devel`) and is the **active branch for other users**. The main tier is dormant
  until a foundation lands there.

## Policy ↔ ship_config state (per branch)

The existing 3-state model is reused, made exhaustive per branch by the generator:

| Lifecycle | `ship_config` value | Served? |
|---|---|---|
| Passed the branch's gate | `"<strategy>"` (from decisions) | yes |
| Not passed (or under rework) | `"withheld"` | no (pruned/dark) |
| (no longer used in generated configs) | `absent` | — |

"Graduated" is **not** a 4th config value — it is a cell that is `"<strategy>"` on `main`.
The lifecycle `absent → "withheld" → "<strategy>" → graduated` (plan §ship-mechanism) is
realized across branches: `"<strategy>"` on `devel` = Gate-1 shipped; `"<strategy>"` on
`main` = Gate-2 graduated.

## Design (Approach A — chosen)

### 1. Canonical decisions file — `data/gate1_decisions.json` (new)
`{league: {market: strategy}}` for the Tier-0 lock-in (seed from the current 36). The
single human-curated source of truth for "passed Gate 1 + which strategy", **branch-
independent** (identical on every branch). Humans edit this when a cell clears or leaves
Tier-0. `ship_config.json` becomes a **generated artifact**.

### 2. Shared classifier — refactor into `training/graduation.py` (new module)
Extract `check_graduation`'s `_classify_lifecycle` + the two parquet readers into importable
pure functions. `check-graduation` (display) and the generator (decision) then share one
definition of "graduated". Behavior unchanged for `check-graduation`.

> **Known gap (flagged, not closed here):** the classifier's Gate-2 rule (`n_settled ≥ 30`
> + book-BSS ≥ 0) is a *simplified proxy* of `ship_gate.md`'s full Gate 2 (≥ 200 offers,
> ±0.03 over-rate, top-decile MAE, calibration bias, parlay yield, ≥ 14-day soak). `main`
> is dormant with no live data, so the proxy is acceptable now; aligning it to the full
> Gate 2 is future work tracked in the plan.

### 3. Generator CLI — `scripts/generate_ship_config.py` (new console script)
`generate-ship-config`:
- `--branch {devel|main}` (required) — *not* `--tier` (avoids colliding with the offline
  Tier-0/Tier-1 sub-modes).
- `--prune/--no-prune` (default **no-prune**), `--decisions`, `--out`, `--model-stats`,
  `--live-metrics`, `--dry-run`.
- **Active set:** `devel` = all decisions (the 36); `main` = `{cell ∈ decisions :
  graduated(cell)}` (0 today).
- **Output:** writes `ship_config.json` over **all** `ALL_MARKETS`: active → its decisions
  strategy, everyone else → `"withheld"`. Deterministic/sorted. Errors if a decisions cell
  is not in `ALL_MARKETS`. Output passes `load_ship_config` by construction.
- `--prune` additionally `prune_model_pickle`s every non-active cell (immediate dark-out on
  the current machine; matches the manual prune already done).

### 4. Monthly cron — Option X (regenerate + auto-PR)
**Regeneration cadence differs by branch.** `devel`'s active set = Gate-1 passers, which
only change when a human edits `gate1_decisions.json` (a cell clears/leaves Tier-0) — so
`devel`'s `ship_config.json` is regenerated **manually on decision change**, not on a timer.
`main`'s active set = graduated cells, which evolve with live data — so **only `main`** is
on the monthly cron.

New `scripts/run_job.sh gate-status` job (monthly, e.g. `0 2 1 * *`), wrapped like every
other job (flock + healthchecks). It runs `generate-ship-config --branch main` from live
graduation and **regenerates `main`'s `ship_config.json`, commits, and opens a PR** (not a
direct push — a human merges; `main` is the public branch). As cells graduate, they flip
`"withheld"→"<strategy>"` on `main`. The job may also update the plan's status-table `main`
column (optional extension).

### 5. `main` foundation cherry-pick (prerequisite for the main tier)
`main` has none of the ship system. Before `--branch main` output is usable, the foundation
(the `ship_config.py` loader + `baselines.py` registry + `helpers/io.py` prune + `model_prob`
decode + `pipeline`/`cli` wiring) must reach `main` — a clean delta carve via the
`devel-ship-curator`, **targeting `main`**, denylist-enforced. Because `main` is the active
public branch, the curator commits locally and **Trevor pushes/merges** (curator never
pushes). This is its own carve, sequenced before the first `--branch main` commit.

## Data flow

```
gate1_decisions.json (36, curated) ─┐
ALL_MARKETS (96, static) ───────────┤
                                    ├─ generate-ship-config --branch B ─> ship_config.json ─[commit/PR]─> meditate prune ─> prophecize serves only active
live_metrics + model_stats parquets ┘        (devel: 36 active + 60 withheld; main: graduated∩36 active + rest withheld)
        │
        └─ training/graduation.py (shared) — also powers check-graduation display
```

## Documentation updates (part of this work)
- `docs/ship_gate.md` and `docs/gbdt_mean_regression_plan.md` (§ship-mechanism / lifecycle):
  record the **default-deny** change — generated per-branch configs are exhaustive over
  `ALL_MARKETS`; non-passers are `"withheld"`, not `absent`. Update the "absent = serve
  default" prose accordingly.
- Note the `gate1_decisions.json` source-of-truth and the `generate-ship-config` workflow.

## Testing
- **Generator:** `devel` → 36 active + 60 withheld; `main` no-data → all-withheld; `main`
  with a fake-graduated cell → that cell active, rest withheld; decisions cell ∉ `ALL_MARKETS`
  → error; idempotent (re-run = same output); output passes `load_ship_config`; `--prune`
  deletes exactly the non-active pickles (temp models dir).
- **Classifier refactor:** existing `check_graduation` golden tests stay green (pure move).
- **Cron job:** a dry-run/fake-mode test of the `run_job.sh gate-status` path.
- **Gates:** `ruff` + `pytest tests/golden/` + `pytest -m integration` green;
  `refactoring-specialist` on every touched `.py`.

## Out of scope
- Closing the Gate-2 proxy → full-`ship_gate.md` metric gap (future; main dormant).
- Superseding baselines (Tier-1 ≥ 5% bar), Step-2 / B1.6 feature track.
- Auto-merging the monthly PR (human merges `main`).

## Open items for the implementation plan
- Exact home/signature of the extracted `training/graduation.py` functions.
- `run_job.sh` job mechanics for commit + PR (auth, branch naming, healthcheck slug).
- Sequencing of the `main` foundation carve relative to the generator build.
