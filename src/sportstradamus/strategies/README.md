# `sportstradamus.strategies`

Bet-sizing and contest-construction logic. Pure orchestration over the
prediction pipeline — no league-specific math lives here.

## Modules

| Module | Purpose |
|---|---|
| `kelly` | Fractional-Kelly stake sizing with CLV/training shrinkage blend. CLI `poetry run kelly`. |

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
5. Neither → fallback `1.0`, logged at DEBUG.

The ramp constants (`LIVE_BLEND_FLOOR=25`, `LIVE_BLEND_FULL=100`) are
roadmap-aligned with `CLV_SEGMENT_MIN_N=20`: live signal is no longer
pure noise once the segment crosses ~25 legs but should still be smoothed
by training-time history; by ~100 legs the live signal dominates because
current-season conditions have moved past training assumptions.

## Optional dependencies

`joint_kelly_portfolio` and the `kelly` CLI lazy-import `cvxpy`,
`PyYAML`, and `tabulate`. Install via:

```bash
poetry install --with strategy
```

Importing `sportstradamus.strategies.kelly` itself does not require any
of these.
