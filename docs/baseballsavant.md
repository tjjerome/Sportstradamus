# Baseball Savant snapshotter

`savant-fetch` snapshots Baseball Savant (Statcast) MLB leaderboards to
date-stamped parquets that `StatsMLB` folds into player/team features. It runs on
the shared collector framework — see [data_collectors.md](data_collectors.md) for
the CLI surface (`run`/`verify`/`list`), the idempotent skip-if-on-disk behavior,
and the strict-`<` leakage rule. This page covers only what is savant-specific.

## Public source — no credentials

Baseball Savant leaderboards are public, so this source has **no keys.json entry
and no `refresh-auth` command**. It fetches through the shared `Scrape`
header-rotation client — the same one that already pulls savant's game feed and
affinity CSVs, and that unblocks hosts (savant, moneypuck) which answer a default
User-Agent with an HTML bot-block page. CSV leaderboards
(`response_format="csv"`) come back as a DataFrame via `Scrape.get_csv`; JSON
endpoints return the decoded dict.

## One-time setup

### Register endpoints

The catalog ships empty (`data/config/baseballsavant_endpoints.json` = `[]`). Pin
each leaderboard or custom-CSV URL from the savant site — **do not invent column
names**. Name each spec `player_<tool>` or `team_<tool>` so it routes to
`player_data/` / `team_data/` (e.g. `player_expected_stats`,
`team_running_game`). Each entry is an `EndpointSpec` (`name`, `url`,
`response_format="csv"` for the CSV leaderboards, optional `params`), using
`{season}` where savant's URL takes the year. Savant's `player_name` column is
typically `"Last, First"`; account for that when pinning the join key below.

### Verify locally

```bash
sportstradamus fetch savant list
sportstradamus fetch savant run --season 2026 --dry-run
sportstradamus fetch savant run --season 2026 --only player_expected_stats
sportstradamus fetch savant verify --season 2026
```

`--season` is the calendar year; `--date` defaults to today, so omit it for a
live capture. Each run writes
`data/{player,team}_data/MLB/baseballsavant/{season}/{date}/{tool}.parquet`.

## Wiring the features (once real columns exist)

`StatsMLB._join_fp_player_features` / `_join_fp_team_features` already delegate to
the shared as-of loader. To light them up after pinning real leaderboards, set the
join-schema constants in `src/sportstradamus/stats/mlb.py`:

- `_SAVANT_PLAYER_KEY_COL` — the player-name column (savant emits `"Last, First"`;
  the loader accent-strips it to match the gamelog, but reorder it upstream if the
  gamelog uses `"First Last"`).
- `_SAVANT_PLAYER_META_COLS` — non-feature columns to drop (player_id, team, …).
- `_SAVANT_TEAM_KEY_COL` / `_SAVANT_TEAM_META_COLS` — same for the team tables,
  keyed to match the gamelog's team abbreviation.

Until these are set the hooks return `None` and MLB features stay gamelog-only.
Once set, the `{col}_asof` columns appear in models on the next `meditate` that
rebuilds the matrix cache.

## Sync to production

Savant collection is dev-side: run `savant-fetch` on the dev box, then
`scripts/sync_to_prod.sh` pushes the snapshots up (additive, never `--delete`).
