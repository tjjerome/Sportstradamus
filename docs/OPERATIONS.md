# Operations

The daily workflow and the production cron schedule. This is the canonical
runbook home; `<repo-dir>` below stands for wherever the repository is checked
out on the box running the jobs.

## Daily Workflow

```
morning:
  poetry run sportstradamus confer          # pull today's lines
  poetry run sportstradamus prophecize      # score + write dashboard snapshots

weekly (or when model accuracy drops):
  poetry run sportstradamus meditate        # retrain stale models
```

League activity — which leagues `confer` polls and `prophecize`/`meditate` update —
is derived automatically from the Odds API events feed, so season starts and ends
need no manual edits. The hand-set `season_start` constants in the Stats classes
are only fallback seeds. To force a gamelog update in the offseason, set
`SPORTSTRADAMUS_FORCE_UPDATE=1`.

## Recommended Cron

All jobs run through [`scripts/run_job.sh`](../scripts/run_job.sh), which adds a
per-job `flock` (an overlapping run of the same job is skipped), a self-deploy
`git pull` before each job, a shared archive lock (DuckDB allows one writer),
and Healthchecks.io start/fail/success pings. Each job pings
`HEALTHCHECK_URL_<JOB>` (the job name uppercased, `-` → `_`; e.g. `close-lines`
→ `HEALTHCHECK_URL_CLOSE_LINES`), falling back to the shared `HEALTHCHECK_URL`.

```cron
50 8-20 * * *          <repo-dir>/scripts/run_job.sh prophecize
30 8,11,14,17,20 * * * <repo-dir>/scripts/run_job.sh confer
55 8 * * *             <repo-dir>/scripts/run_job.sh ledger-commit --run-slot morning
55 14 * * *            <repo-dir>/scripts/run_job.sh ledger-commit --run-slot afternoon
0 1 * * 5              <repo-dir>/scripts/run_job.sh meditate
0 23 * * *             <repo-dir>/scripts/run_job.sh reflect
*/10 11-23,0-1 * * *   <repo-dir>/scripts/run_job.sh close-lines
0 2 1 * *              <repo-dir>/scripts/run_job.sh gate-status
0 10 * * 3             <repo-dir>/scripts/run_job.sh fp-fetch
```

| Job | Cadence | Purpose |
|---|---|---|
| `prophecize` | hourly, 8am–8pm | score offers, write dashboard snapshots |
| `confer` | 5 slots/day | broad odds/props fetch; the credit governor decides which leagues each slot fetches |
| `ledger-commit` | 2×/day (morning + afternoon) | simulated-bettor ledger commit (policy_v1) |
| `meditate` | Fri 1am | retrain models |
| `reflect` | nightly 11pm | grade history, profit-sim + calibration summaries + simulated-bettor ledger |
| `close-lines` | every 10 min, game hours | closing-line capture for games starting in 5–25 min; no-op tick when nothing is due |
| `gate-status` | monthly | Gate-2 promote/demote PR against `main`; needs `gh` auth and `HEALTHCHECK_URL_GATE_STATUS` |
| `fp-fetch` | Wed 10am (NFL season) | Fantasy Points endpoint snapshots; needs a fresh session cookie and `HEALTHCHECK_URL_FP_FETCH` |

Couplings to keep in sync:

- The number of `confer` slots must match `broad_slots_per_day` in
  `data/config/odds_api_budget.json` (currently 5). Fewer real slots than the
  config claims underspends the monthly credit budget; more overspends it.
- `close-lines` is the per-game data floor and is never throttled by the
  governor — don't widen its hours without re-checking the credit budget.
- Season starts/ends never require cron edits: idle leagues cost one free
  events call per broad run, and `prophecize`/`meditate` skip them.

Dev-side collectors (`ctg-fetch` NBA, `savant-fetch` MLB) are **not** in the prod
crontab — they run on the dev box beside the manual weekly `meditate`, then
`scripts/sync_to_prod.sh` uploads their snapshots (additive, never `--delete`).
Both are `run_job.sh` cases, so scheduling either (with a
`HEALTHCHECK_URL_CTG_FETCH` / `HEALTHCHECK_URL_SAVANT_FETCH` set) is possible.
See [data_collectors.md](data_collectors.md).

## Dashboard as a service

Run the dashboard under systemd with the module-form entry point — it survives
`poetry install` re-registrations, which console-script stubs do not:

```ini
[Service]
User=<runtime-user>
WorkingDirectory=<repo-dir>
ExecStart=poetry run python -m sportstradamus dashboard
Restart=always
RestartSec=5
```

## Related runbooks

- [merge_archives.md](merge_archives.md) — reconcile the production and dev odds
  archives when they drift apart.
- [dashboard_remote_access.md](dashboard_remote_access.md) — keep the dashboard
  always-on and reachable tailnet-only, without exposing it to the internet.
