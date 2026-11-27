# `sportstradamus.strategies`

Bet-sizing and contest-construction logic. Pure orchestration over the
prediction pipeline — no league-specific math lives here.

## Modules

| Module | Purpose |
|---|---|
| `kelly` | Fractional-Kelly stake sizing, with a trust weight its caller supplies. CLI `sportstradamus bet kelly`. |
| `underdog_pickem` | Pick'em (Power/Flex) entry construction for Underdog and Sleeper. CLI `sportstradamus bet pickem --platform {Underdog,Sleeper}` emits a recommendations YAML. |
| `ledger` | Simulated-bettor ledger commit: draws and sizes each persona's entries and appends them to the JSONL ledger (`_ledger_*` hold selection, cross-game pricing, settlement, bankroll and storage). CLI `sportstradamus bet ledger-commit`. Policy: `docs/handoffs/sim-bettor-ledger.md` §10. |
| `profit_sim` | Kelly-sized profit / ROI / Sharpe / drawdown Monte-Carlo backtest. |

## Pick'em entries (`underdog_pickem`)

`construct_entries` takes the same-game entry search's rows (Power/Flex) and ranks
them by their fitted value (`EV`), `platform`-parameterized (`"Underdog"` or
`"Sleeper"`). No entry is staked: `recommended_stake` is 0 until the entries' money
record supports a stake (`docs/handoffs/profit-engine.md` §5.3).
`sportstradamus bet pickem --platform {Underdog,Sleeper}` runs the loader itself
and emits a recommendations YAML; `sportstradamus bet kelly` sizes that YAML's
entries by Kelly on their fitted hit chance, which stakes nothing on an entry valued
under 1. The dashboard reads no pick'em file: its slip rail builds and sizes entries
live.

## Trust weight (`model_shrinkage`)

`fractional_kelly_stake`, `kelly_edge` and each `KellyCandidate` take a
`model_shrinkage` in `[0, 1]` (default `1.0`, full trust). One caller passes one:
the dashboard's slip rail, which takes the lowest `kelly_shrinkage` among its legs'
cells in `model_stats.parquet` and `NO_EVIDENCE_SHRINKAGE` when no leg has a cell
(`dashboard/components/slip_builder.py:slip_shrinkage`). `NO_EVIDENCE_SHRINKAGE` is a
small prior: unmeasured is not measured-worthless.

`kelly_edge` blends toward the book's implied probability `1 / payout`, not
toward 0.5, so shrinkage `0.0` is exactly zero edge at every payout. A
coin-flip anchor instead scored any payout above 2x as +EV however little
the model was trusted, which manufactured edge on long multi-leg entries.

## Dependencies

`joint_kelly_portfolio` uses `cvxpy` (SCS solver); the `kelly` CLI uses
`PyYAML` + `tabulate` for recommendations-YAML I/O. All three are required
base dependencies (installed by `poetry install`) and imported at module top.
