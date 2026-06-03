# Merging two odds archives

The odds archive (`archive/archive.duckdb`) lives on two machines — the production
server (continuously appending today's odds via `confer`) and a dev machine (where
historical backfills run). They drift apart: each holds observations the other lacks.

`merge-archives` reconciles them losslessly. The archive is append-only time-series
(the same `(league, market, game_date, entity, book)` key recurs with different
`observed_at` timestamps), so the merge is a **set-union of full rows**: every row in
either database is kept, and only bit-identical duplicates collapse. No observation is
ever dropped, and distinct timestamps for the same key both survive.

The source archive is opened read-only and never modified. The target is rebuilt in
place — with a timestamped `.bak-<epoch>` written first — unless `--output` sends the
union to a new file instead.

## One-command dev sync

From the dev machine, `scripts/sync_from_prod.sh` automates the whole pull: it snapshots
prod's archive under prod's flock, rsync's the snapshot and the `data/runtime/` outputs
down, then runs `merge-archives` to fold them in. Every ssh/rsync carries a connect
timeout, so a powered-off prod fails fast instead of hanging.

```bash
scripts/sync_from_prod.sh             # full sync
scripts/sync_from_prod.sh --dry-run   # preview the commands without running them
```

It defaults to `sportstradamus@192.168.1.84:/home/sportstradamus/Sportstradamus`;
override with the `PROD_SSH` / `PROD_DIR` environment variables. The `data/runtime/`
pull is update-only (never deletes dev-only files), and the archive merge backs dev's
archive up to `.bak-<epoch>` first, so the local backfill is never lost.

The steps below are what the script runs under the hood — and the manual path if you'd
rather drive it yourself.

## Pull production into dev

This is the usual direction: fold the server's live rows into the dev machine.

1. **Snapshot production consistently.** DuckDB holds an exclusive file lock for the
   lifetime of a connection, so a plain `cp` during a `confer`/`prophecize` run can copy a
   torn database. Take the snapshot while holding the same archive flock the cron wrapper
   uses, which guarantees no job is mid-write:

   ```bash
   flock /tmp/sportstradamus-archive.lock \
       cp archive/archive.duckdb /tmp/prod_archive_snapshot.duckdb
   ```

   A clean shutdown leaves no `archive.duckdb.wal` sidecar — the single file is
   self-contained and safe to copy. If a `.wal` exists outside the flock, a writer is
   active or crashed; wait for the flock before copying.

2. **Ship it to dev:**

   ```bash
   scp sportstradamus@192.168.1.84:/tmp/prod_archive_snapshot.duckdb /tmp/
   ```

3. **Dry-run, then merge.** The target defaults to dev's `archive/archive.duckdb`, and a
   backup is written automatically before any in-place change:

   ```bash
   poetry run merge-archives --source /tmp/prod_archive_snapshot.duckdb --dry-run
   poetry run merge-archives --source /tmp/prod_archive_snapshot.duckdb
   ```

   The dry-run prints how many rows each table would gain. The real run reports the same
   counts and rebuilds the tables sorted, so read-time zone-map pruning stays effective.

## Round-trip back to production

The merge is union-based and idempotent, so folding dev's now-complete archive back into
production loses nothing. Snapshot dev, ship it to the server, and run the same command
there with dev's snapshot as `--source`. Re-running on already-merged data is a no-op.

## Options

| Option | Effect |
|---|---|
| `--source PATH` | Archive to merge FROM (read-only, never modified). Required. |
| `--target PATH` | Archive to merge INTO, rebuilt in place. Defaults to `$SPORTSTRADAMUS_ARCHIVE_DB` or `archive/archive.duckdb`. |
| `--output PATH` | Write the union to a new file and leave both inputs untouched. |
| `--dry-run` | Report row counts only; write nothing. |
| `--no-backup` | Skip the pre-merge `.bak-<epoch>` copy of the target. |

If the target is locked by a running cron, the command fails with guidance rather than
waiting — run it during a quiet window or while holding the archive flock above.
