# `sportstradamus.strategies`

Bet-sizing and contest-construction logic. Pure orchestration over the
prediction pipeline — no league-specific math lives here.

## Modules

| Module | Purpose |
|---|---|
| `kelly` | Fractional-Kelly stake sizing with CLV/training shrinkage blend. CLI `sportstradamus bet kelly`. |
| `underdog_pickem` | Pick'em (Power/Flex) entry construction for Underdog and Sleeper. CLI `sportstradamus bet pickem --platform {Underdog,Sleeper}` emits a recommendations YAML. |
| `ledger` | Simulated-bettor ledger commit: draws and sizes each persona's entries and appends them to the JSONL ledger (`_ledger_*` hold selection, cross-game pricing, settlement, bankroll and storage). CLI `sportstradamus bet ledger-commit`. |
| `profit_sim` | Kelly-sized profit / ROI / Sharpe / drawdown Monte-Carlo backtest. |

## Pick'em entries (`underdog_pickem`)

`construct_entries` ranks and Kelly-sizes Pick'em contest entries (Power/Flex),
`platform`-parameterized (`"Underdog"` or `"Sleeper"`).
`sportstradamus bet pickem --platform {Underdog,Sleeper}` runs the loader itself
and emits a recommendations YAML; `sportstradamus bet kelly` re-sizes that YAML.
The dashboard reads no pick'em file: its slip rail builds and sizes entries live.

## Shrinkage resolution (`kelly.resolve_shrinkage`)

`underdog_pickem.resolve_market_shrinkage` calls `resolve_shrinkage` for each
cell, and callers pass the result to `fractional_kelly_stake` / `kelly_edge` as
`model_shrinkage` (which defaults to `1.0`, full trust, when left unset).
`resolve_shrinkage` takes the first rule that applies:

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
5. Neither → `NO_EVIDENCE_SHRINKAGE` (a small prior: unmeasured is not
   measured-worthless), logged at DEBUG.

`kelly_edge` blends toward the book's implied probability `1 / payout`, not
toward 0.5, so shrinkage `0.0` is exactly zero edge at every payout. A
coin-flip anchor instead scored any payout above 2x as +EV however little
the model was trusted, which manufactured edge on long multi-leg entries.

The ramp constants (`LIVE_BLEND_FLOOR=25`, `LIVE_BLEND_FULL=100`) are
roadmap-aligned with `CLV_SEGMENT_MIN_N=20`: live signal is no longer
pure noise once the segment crosses ~25 legs but should still be smoothed
by training-time history; by ~100 legs the live signal dominates because
current-season conditions have moved past training assumptions.

## Dependencies

`joint_kelly_portfolio` uses `cvxpy` (SCS solver); the `kelly` CLI uses
`PyYAML` + `tabulate` for recommendations-YAML I/O. All three are required
base dependencies (installed by `poetry install`) and imported at module top.
