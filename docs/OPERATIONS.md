# Operations

The daily workflow and the production cron schedule. This is the canonical
runbook home; `<repo-dir>` below stands for wherever the repository is checked
out on the box running the jobs.

## Daily Workflow

```
morning:
  sportstradamus confer          # pull today's lines
  sportstradamus prophecize      # score + write dashboard snapshots

weekly, on the dev box (or when model accuracy drops):
  sportstradamus meditate        # retrain stale models, then scripts/sync_to_prod.sh
```

League activity — which leagues `confer` polls and `prophecize`/`meditate` update —
is derived automatically from the Odds API events feed, so season starts and ends
need no manual edits. The feed's opening night is captured once a season, when a
league comes back from an off-season (more than 60 days between games), and is kept
through breaks and idle spells. `Stats.update` adopts it as the season start only
when it falls more than 300 days after the hand-set `season_start` constant in the
Stats class: a nearer one was captured inside the season the constant already
names, and the constant stands. To force a gamelog update in the offseason, set
`SPORTSTRADAMUS_FORCE_UPDATE=1`.

Pre-season is never priced: a game dated before the feed's opening night for its
league is archived and not scored (`_in_season_markets` in `prediction/scoring.py`).
A new NBA season reads the newest earlier `player_data/NBA/nba_players_<season>.csv`
seed, with a warning in the log, until that season's own Cleaning the Glass export
is dropped in.

## Recommended Cron

All jobs run through [`scripts/run_job.sh`](../scripts/run_job.sh), which adds a
per-job `flock` (an overlapping run of the same job is skipped), a self-deploy
`git pull` before each job (restarting the dashboard when that pull moves its
code — see [Dashboard as a service](#dashboard-as-a-service)), a shared archive
lock (DuckDB allows one writer; `close-lines` takes it itself, below), and Healthchecks.io
start/fail/success pings. Each job pings
`HEALTHCHECK_URL_<JOB>` (the job name uppercased, `-` → `_`; e.g. `close-lines`
→ `HEALTHCHECK_URL_CLOSE_LINES`), falling back to the shared `HEALTHCHECK_URL`.

Every ping carries a `?rid=` run ID and a body opening `job=<job> host=<host>
status=…`; failures append the exit code and a tqdm-stripped log excerpt, and a
lock timeout names the jobs that were running instead. Point each
`HEALTHCHECK_URL_<JOB>` at its own check UUID if you can — with one shared UUID
any job's success still marks the check alive, so a job that fails every run
stays invisible.

```cron
50 8-20 * * *          <repo-dir>/scripts/run_job.sh prophecize
30 8,11,14,17,20 * * * <repo-dir>/scripts/run_job.sh confer
55 8 * * *             <repo-dir>/scripts/run_job.sh ledger-commit --run-slot morning
55 14 * * *            <repo-dir>/scripts/run_job.sh ledger-commit --run-slot afternoon
# 0 1 * * 5            <repo-dir>/scripts/run_job.sh meditate   # off: models train dev-side (below)
0 2 * * *              <repo-dir>/scripts/run_job.sh reflect
*/10 11-23,0-1 * * *   <repo-dir>/scripts/run_job.sh close-lines
0 4 1 * *              <repo-dir>/scripts/run_job.sh gate-status
0 10 * * 3             <repo-dir>/scripts/run_job.sh fp-fetch
0 5 1 * *              <repo-dir>/scripts/run_job.sh headshots
15 4 28-31 * *         <repo-dir>/scripts/run_job.sh odds-backfill --league NFL --markets 'yards,qb yards' --start 2024-09-05 --end 2026-10-04 --dates-from 'Moneyline,rushing yards'
```

| Job | Cadence | Purpose |
|---|---|---|
| `prophecize` | hourly, 8am–8pm | score offers, write dashboard snapshots |
| `confer` | 5 slots/day | broad odds/props fetch; the credit governor decides which leagues each slot fetches |
| `ledger-commit` | 2×/day (morning + afternoon) | simulated-bettor ledger commit (policy: [handoffs/sim-bettor-ledger.md](handoffs/sim-bettor-ledger.md) §10) |
| `meditate` | off on prod | retrain models — runs dev-side weekly; `scripts/sync_to_prod.sh` sends the results (below) |
| `reflect` | nightly 2am | grade history, profit-sim + calibration summaries + simulated-bettor ledger |
| `close-lines` | every 10 min, game hours | closing-line capture for games starting in 5–25 min; no-op tick when nothing is due or spooled |
| `gate-status` | monthly | Gate-2 promote/demote PR against `main`; needs `gh` auth and `HEALTHCHECK_URL_GATE_STATUS` |
| `fp-fetch` | Wed 10am (NFL season) | Fantasy Points endpoint snapshots; needs a fresh session cookie and `HEALTHCHECK_URL_FP_FETCH` |
| `headshots` | monthly | refresh the player-headshot cache the constellation card draws; needs `HEALTHCHECK_URL_HEADSHOTS` |
| `odds-backfill` | 4:15am, days 28–31 | buy odds history with the credit cycle's leftover: acts on the cycle's last day only, newest date first, and makes no API call once its range is held; needs `HEALTHCHECK_URL_ODDS_BACKFILL` |

Couplings to keep in sync:

- The number of `confer` slots must match `broad_slots_per_day` in
  `data/config/odds_api_budget.json` (currently 5). Fewer real slots than the
  config claims underspends the monthly credit budget; more overspends it.
- `close-lines` is the per-game data floor and is never throttled by the
  governor — don't widen its hours without re-checking the credit budget.
- `reflect` holds the shared archive lock for its whole ~1 h run, so it sits at
  2am, after `close-lines` stops at 01:50. `close-lines` alone does not hold that
  lock for its whole run: it fetches on time, spools the fetch to
  `data/runtime/close_line_captures.json`, and takes the lock only to archive the
  spool, giving up after `moneylines.CLOSE_LINES_LOCK_WAIT_S` and leaving the spool
  to a later tick. A tick that overlaps `reflect` therefore archives nothing until
  `reflect` finishes. Every other job holds the lock for its whole run, including
  ones like `gate-status` that never read the archive, which is why it sits at 4am
  rather than sharing `reflect`'s start minute.
- The headshot cache is per-box and gitignored, so `headshots` runs on the dev box
  and the prod box independently and is deliberately absent from `sync_to_prod.sh`'s
  `COLLECTOR_RELS` — don't "fix" that omission by adding it.
- Season starts/ends never require cron edits: idle leagues cost one free
  events call per broad run, and `prophecize`/`meditate` skip them.
- `odds-backfill` pays from the key `confer` and `close-lines` poll with. The
  governor hands every spendable credit to props by the reset, so no credit is
  idle on its books and the job takes only what
  `helpers/odds_budget.py:backfill_allowance` grants: on the cycle's last day,
  the balance above the governor's reserve and above a whole day of broad runs
  at their last-week rate (padded by `floor_safety_factor`). Cron cannot name a
  month's last day, so the line fires on days 28–31 and each earlier morning
  costs one free quota probe. It sits at 4:15am to clear `reflect`.
  `run_job.sh` holds the paying key and the leftover-only flags, because the
  same pull without `--surplus-only` buys its whole range at once; the cron line
  only names the pull. The job walks back from `--end`, so what is held is one
  unbroken block. `--dates-from` names two markets because a 49ers game leaves
  no `Moneyline` rows ([handoffs/honest-receipts.md](handoffs/honest-receipts.md)):
  the second market seeds the dates that team played alone, and every date is
  priced one game above the archive's count. A bought run therefore leaves
  credits over, and the job plans again on the live balance until the next date
  no longer fits. Why this pull and what follows it are in
  [handoffs/nfl-ship15-recovery.md](handoffs/nfl-ship15-recovery.md) §6. Its
  resume file, `archive/backfill_progress.json`, is per box: the same pull run
  by hand on another box buys the dates twice. The rows reach the dev box by the
  usual archive pull ([merge_archives.md](merge_archives.md)).

`export-line-movement` is **not** in the crontab either, because `prophecize`
already writes `current_line_movement.parquet` on every run — the Board's move
column tracks the hourly pipeline by default. Schedule it only to refresh that
snapshot between `prophecize` runs: it re-derives the movement from the archive
against the offers already on disk, so it costs one archive read and no scrape.

```cron
20 8-20 * * *          <repo-dir>/scripts/run_job.sh export-line-movement
```

Keep it off the `:50` minute `prophecize` and `close-lines` already share — it
takes the same archive lock.

Dev-side collectors (`ctg-fetch` NBA, `savant-fetch` MLB) are **not** in the prod
crontab — they run on the dev box beside the manual weekly `meditate`, then
`scripts/sync_to_prod.sh` uploads the models, the gitignored serving artifacts
(`book_weights.json`, `stat_calibration.json`, `model_stats.{parquet,csv}`,
per-league `corr_*` files, MLB's two Savant comp tables — see the script
header), and their snapshots
(snapshots additive, never `--delete`). The models mirror skips
`data/models/prior/` and the script never sends `data/test_sets/`: the superseded
model files and test sets a retrain keeps stay on the box that trained them
(`training/prior_models.py` in [ARCHITECTURE.md](ARCHITECTURE.md)). Training runs dev-side this way, so prod's
`meditate` cron line stays commented out — the two workflows would overwrite
each other's models and serving artifacts.
Both collectors are `run_job.sh` cases, so scheduling either (with a
`HEALTHCHECK_URL_CTG_FETCH` / `HEALTHCHECK_URL_SAVANT_FETCH` set) is possible.
See [data_collectors.md](data_collectors.md).

The dev box's own crontab holds one line: the weekly pull of production's archive,
runtime outputs and collector snapshots (`scripts/sync_from_prod.sh`), so nothing
prod prunes (feature-log partitions, history rows) is lost before dev has a copy.

```cron
30 6 * * 1 cd <repo-dir> && { date -u +"start \%FT\%TZ"; PATH=<home>/.local/bin:/usr/bin:/bin flock -n /tmp/sportstradamus-sync-from-prod.lock timeout 2h scripts/sync_from_prod.sh && ls -1 archive/archive.duckdb.bak-* | head -n -3 | xargs -r rm -fv --; echo "exit=$?"; } >> logs/sync_from_prod.log 2>&1
```

Monday 06:30 CDT falls after `reflect` and `odds-backfill` and before the first
`confer`, so the weekend's games arrive before the Monday reads. `timeout` bounds
the prod-side archive flock, which has no timeout of its own. After a green sync
the line keeps the newest 3 `archive.duckdb.bak-*` copies (`merge-archives` writes
one per sync, about 2.5 GB each) and deletes the rest. It skips `run_job.sh`, whose
self-deploy pull would move the dev checkout. A Monday with no `start` line in the
log means WSL was not running at 06:30: run the line by hand. The fold step can
leave `banned_combos.json` and `parlay_rake.json` modified in the checkout; review
them before they are committed.

That sync carries every league at once: the models mirror deletes any prod model
file the dev box lacks, and `book_weights.json`, `stat_calibration.json` and
`model_stats.{parquet,csv}` each hold every league's entries. When it would
carry a league that is not ready (its `--dry-run` lists every league's files;
a retrain may still be running), send the ready league alone:

```bash
scripts/sync_to_prod.sh --league NFL --dry-run   # proof and file list; writes nothing on prod
scripts/sync_to_prod.sh --league NFL             # send
```

`--league` copies prod's three shared files down, replaces only that league's
entries with the dev box's, and prints the proof: per league, how many entries
differ from prod's copy, which only the named league may. It then sends the
league's `models/<LEAGUE>_*` files, the merged shared files, the league's
`corr_*` files and, for MLB, the two comp tables. It deletes nothing on prod (a
model file the dev box no longer has stays until a full sync), sends no
collector snapshots, and refuses to send when the proof fails or a shared file
changed on prod after it was copied down. Send between board runs, after one
`prophecize` logs `OK` and well before the next starts, so no run reads half a
send.

Every file a send replaces is kept on prod in
`~/league-send-backup/<LEAGUE>-<UTC stamp>/`, laid out like the data folder, so
one command run on prod puts them all back:
`cp -a <that folder>/. <repo-dir>/src/sportstradamus/data/`. The send prints the
folder before it writes and the whole command as its last line. A send that
stops partway is finished by running it again. To take back several sends, run
their commands newest first. A file a send added rather than replaced (the
model of a cell prod never had; its line in the send's output reads
`<f+++++++++`) stays after the undo; delete it by hand.

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

The dashboard is the only long-lived process in the system, and every cron job
self-deploys before it runs, so prod code routinely moves under a running
dashboard. Streamlit re-execs page scripts on each rerun but never re-imports
`sys.modules`, so a pulled page meets helper modules and config loaders from
before the pull — surfacing as an `ImportError` on a symbol that exists only in
the new module, or a `KeyError` on a config key the new schema dropped.

`run_job.sh` therefore restarts the unit whenever a pull moves anything under
`src/sportstradamus/dashboard/` or `src/sportstradamus/data/config/`, logging
`PULL_RESTART`. Two things must line up:

- `DASHBOARD_UNIT` names the unit (default `sportstradamus-dashboard.service`);
  set it empty to skip the restart.
- The cron user needs a passwordless restart, since the wrapper runs unattended:

  ```
  # /etc/sudoers.d/sportstradamus-dashboard, mode 0440
  sportstradamus ALL=(root) NOPASSWD: /usr/bin/systemctl restart sportstradamus-dashboard.service
  ```

  The rule is pinned to the literal command, so the unit name here and in
  `DASHBOARD_UNIT` must match exactly. Validate with `visudo -cf` before it
  takes effect. A failed restart logs `PULL_WARN reason=dashboard_restart_failed`
  and never blocks the job.

## Incident recovery

Quick checks when collection or serving looks wrong:

- **Governor decisions**: `grep -h "budget decision" logs/*/confer.jsonl | tail -20`
  — per run: `tiers`, props `allowed`, `gameline` leagues, `reason`, `per_slot`,
  `remaining`. A live league missing from `gameline` means the close-lines floor
  itself is breached; missing from `allowed` for more than a day means the
  starvation escape isn't firing — check its ledger cost vs `per_slot × slots`.
- **Last paid props run per league** (from the ledger):
  `jq -r 'select(.kind=="broad" and .endpoint=="event_odds" and .cost>0) | [.league,.ts] | @tsv' src/sportstradamus/data/runtime/odds_api_usage.jsonl | sort -k2 | awk '{last[$1]=$2} END {for (l in last) print l, last[l]}'`
- **What the backfill bought**:
  `grep -hE "surplus:|credits spent|STOP|call failed" logs/odds-backfill.log | tail` — one
  `surplus:` line per plan (credits remaining, the leftover granted, the dates
  it covers and the most they can cost), then the credits the job spent. A date
  with a failed Odds API call fails the job and stays unbought, so the next
  cycle starts there. One that fails every cycle blocks the walk and pays for
  its earlier calls each time: read the log for the call that fails, and to give
  the date up, add it to `archive/backfill_progress.json` under the job's key.
- **Poisoned odds rows**: `sportstradamus admin sweep-runaway-odds`
  (dry-run; add `--apply` after review). Follow with `meditate --league <LG>` so
  `fit_book_weights` refits off the cleaned archive.
- **Withheld cell still serving**: the dev box's `meditate` prunes withheld
  pickles, and the next full `scripts/sync_to_prod.sh` removes them from prod
  (`--league` never deletes); a volume denominator in `VOLUME_STATS` keeps its
  pickle and serves nothing, no warning. A pickle still on disk is refused by
  `model_prob` with a "withheld but pickle on disk" warning. To prune immediately:
  `poetry run python -c "from sportstradamus.helpers.io import prune_model_pickle; print(prune_model_pickle('WNBA','PRA'))"`
- **Served cell failing fresh gates**: `meditate` only warns (SHIP-GATE
  WARNINGS table at the end of the run) — demotion is manual: flip the cell to
  `shipped: "withheld"`, or `sportstradamus ship config --branch devel --prune`.
- **Diverged dispersion fit**: serving logs "dispersion_cal … pinned at its fit
  bound" and serves the unscaled shape; retrain the cell to clear it.
- **`prophecize` pings FAIL but a board was written**: grep its log for
  `update failed` and `platforms failed`. A league whose gamelog update raised sits
  that run out (`stats.load_live_leagues`): its lines are archived, the other
  leagues are scored, and the run exits 1 once its snapshots are written. It is not
  scored on the cached gamelog, because the day's schedule arrives only with an
  update. `bet pickem` drops the league the same way and only logs it.
- **Whole league unfit to serve** (retrain in flight, schema break): prefix the
  cron line with `SPORTSTRADAMUS_HOLD_LEAGUES=NFL` (comma-separated for several).
  `league_is_live` then reports the league idle, so `prophecize` and
  `bet pickem` skip it while still archiving its DFS lines; `confer` and
  `close-lines` are unaffected. Clear it by removing the prefix — effective next
  tick, nothing to deploy. The hold also stops the league's gamelog `update()` in
  those two jobs; the nightly `reflect` job still runs it, so the gamelog stays
  current through a hold.
  Keep it off `meditate --league All` lines: that loop honors the same gate.

## Related runbooks

- [merge_archives.md](merge_archives.md) — reconcile the production and dev odds
  archives when they drift apart.
- [dashboard_remote_access.md](dashboard_remote_access.md) — keep the dashboard
  always-on and reachable tailnet-only, without exposing it to the internet.
