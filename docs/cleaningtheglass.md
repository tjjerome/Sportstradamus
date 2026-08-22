# Cleaning the Glass snapshotter

`ctg-fetch` snapshots Cleaning the Glass NBA advanced tables to date-stamped
parquets that `StatsNBA` folds into player/team features. It runs on the shared
collector framework — see [data_collectors.md](data_collectors.md) for the CLI
surface (`run`/`verify`/`list`/`refresh-auth`), the idempotent skip-if-on-disk
behavior, the mid-run token-refresh flow, and the strict-`<` leakage rule. This
page covers only what is CTG-specific.

## Legal

Cleaning the Glass is a paid subscription and its Terms of Service may restrict
automated access. Read them and decide whether to proceed. The collector is
single-process with a pause between calls and a realistic User-Agent; it does
nothing to evade detection, so account suspension is a realistic risk if their
ToS forbids this.

## One-time setup

### 1. Capture the session cookie

CTG authenticates with a session **cookie** (no bearer token). Capture it once
per session:

1. Log in to <https://cleaningtheglass.com/> in a desktop browser.
2. Open DevTools → **Network**, filter to **Fetch/XHR** (or **Doc**).
3. Load any stats page so the request fires.
4. Right-click the request row → **Copy** → **Copy as cURL (bash)**, saved to a
   temp file (`wl-paste > /tmp/ctg.curl` on Wayland Linux; `pbpaste > …` on macOS).
5. Write the cookie into `creds/keys.json`:

   ```bash
   sportstradamus fetch ctg refresh-auth /tmp/ctg.curl
   ```

   This populates `cleaningtheglass_cookie` (and `cleaningtheglass_user_agent`),
   preserving every other key. The cookie rotates on CTG's schedule; when it
   expires the run fails and you redo this step (set `HEALTHCHECK_URL_CTG_FETCH`
   so cron surfaces the expiry immediately).

### 2. Register endpoints

The catalog ships empty (`data/config/cleaningtheglass_endpoints.json` = `[]`).
Add one entry per table from a real authenticated capture — **do not invent
column names**. Name each spec `player_<tool>` or `team_<tool>` so it routes to
`player_data/` / `team_data/` (e.g. `player_usage`, `team_four_factors`). The
generic `import-curl` path is not attached to `ctg-fetch`; hand-add entries to
the JSON, each an `EndpointSpec` (`name`, `url`, optional `params`/`method`,
`response_format`), using `{season}` where CTG's URL takes the season.

### 3. Verify locally

```bash
sportstradamus fetch ctg list
sportstradamus fetch ctg run --season 2025 --dry-run
sportstradamus fetch ctg run --season 2025 --only player_usage
sportstradamus fetch ctg verify --season 2025
```

`--season` is CTG's integer label (the 2025-26 season is `2025`, flipping in
September); `--date` defaults to today, so omit it for a live capture. Each run
writes `data/{player,team}_data/NBA/cleaningtheglass/{season}/{date}/{tool}.parquet`.

## Wiring the features (once real columns exist)

`StatsNBA._join_fp_player_features` / `_join_fp_team_features` already delegate to
the shared as-of loader. To light them up after capturing real tables, set the
join-schema constants in `src/sportstradamus/stats/nba.py`:

- `_CTG_PLAYER_KEY_COL` — the player-name column in the CTG player tables.
- `_CTG_PLAYER_META_COLS` — non-feature columns to drop (team, position, …).
- `_CTG_TEAM_KEY_COL` / `_CTG_TEAM_META_COLS` — same for the team tables, keyed by
  a value that matches the gamelog's team abbreviation (add a name→abbr transform
  if CTG emits full names).

Until these are set the hooks return `None` and NBA features stay gamelog-only.
Once set, the `{col}_asof` columns appear in models on the next `meditate` that
rebuilds the matrix cache.

## Sync to production

CTG collection is dev-side: run `ctg-fetch` on the dev box, then
`scripts/sync_to_prod.sh` pushes the snapshots up (additive, never `--delete`).
