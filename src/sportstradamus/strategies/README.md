# `sportstradamus.strategies`

Bet-sizing and contest-construction logic. Pure orchestration over the
prediction pipeline — no league-specific math lives here.

## Modules

| Module | Purpose |
|---|---|
| `kelly` | Fractional-Kelly stake sizing with CLV/training shrinkage blend. CLI `poetry run kelly`. |
| `underdog_pickem` | Pick'em (Power/Flex/Rivals) entry construction for Underdog and Sleeper. `prophecize` snapshots ranked entries to `current_pickem.parquet` for the dashboard's **Predictions — Pickem** page; CLI `poetry run pickem-build --platform {Underdog,Sleeper}` emits a YAML for offline use. |

## Pick'em entries (`underdog_pickem`)

`construct_entries` ranks and Kelly-sizes Pick'em contest entries
(Power/Flex/Rivals — Rivals is Underdog-only, see `PLATFORM_CONTEST_VARIANTS`).
Two front doors, both `platform`-parameterized (`"Underdog"` or `"Sleeper"`):

- **Dashboard (primary).** The hourly `prophecize` run calls
  `build_entries_from_scored` once per platform on the offers it already
  scored — no second scrape, no extra archive lock — and writes one combined
  `current_pickem.parquet` tagged by `platform`. The **Predictions — Pickem**
  page reads it and sizes stakes live against a user-entered bankroll via
  `fractional_kelly_stake`, replacing the offline `kelly` CLI for interactive
  use.
- **Offline CLI.** `poetry run pickem-build --platform {Underdog,Sleeper}`
  re-runs the loader itself and emits a recommendations YAML; `poetry run
  kelly` re-sizes that YAML.

## Shrinkage resolution (`kelly.resolve_shrinkage`)

Order of precedence, used by `fractional_kelly_stake` when no explicit
`model_shrinkage` is supplied:

1. Explicit kwarg — overrides everything.
2. Both live (CLV-segment) and training Brier skill scores present →
   blended on a linear ramp keyed by the per-segment leg count `n`:

   ```
   w_live = clamp((n - LIVE_BLEND_FLOOR) /
                  (LIVE_BLEND_FULL - LIVE_BLEND_FLOOR), 0.0, 1.0)
   shrinkage = w_live * live_bss + (1 - w_live) * training_bss
   ```

3. Only training BSS → use it directly.
4. Only live BSS → use it directly.
5. Neither → `NO_EVIDENCE_SHRINKAGE` (`0.0`, zero stake), logged at DEBUG.

The ramp constants (`LIVE_BLEND_FLOOR=25`, `LIVE_BLEND_FULL=100`) are
roadmap-aligned with `CLV_SEGMENT_MIN_N=20`: live signal is no longer
pure noise once the segment crosses ~25 legs but should still be smoothed
by training-time history; by ~100 legs the live signal dominates because
current-season conditions have moved past training assumptions.

## Dependencies

`joint_kelly_portfolio` uses `cvxpy` (SCS solver); the `kelly` CLI uses
`PyYAML` + `tabulate` for recommendations-YAML I/O. All three are required
base dependencies (installed by `poetry install`) and imported at module top.
