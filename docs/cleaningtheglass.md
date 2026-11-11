# Cleaning the Glass snapshotter

`sportstradamus fetch ctg` snapshots Cleaning the Glass NBA player tables to date-stamped
parquets that `StatsNBA` can fold into player features. It runs on the shared collector
framework — see [data_collectors.md](data_collectors.md) for the CLI surface
(`run`/`verify`/`list`/`refresh-auth`), the idempotent skip-if-on-disk behavior, the mid-run
token-refresh flow, and the strict-`<` leakage rule. This page covers only what is CTG-specific.

## Legal

Cleaning the Glass is a paid subscription and its Terms of Service may restrict automated
access. Read them and decide whether to proceed. The collector is single-process with a pause
between calls and a realistic User-Agent; it does nothing to evade detection, so account
suspension is a realistic risk if their ToS forbids this.

## What it captures

CTG has no table endpoint. `GET /stats/players?season=<year>&seasontype=regseason` returns an
HTML page whose inline script passes every table to `vuePlayers(...)` as JSON, and the page's
CSV button builds its file in the browser from that data. The default page embeds the data of
all 15 stat views, so two catalog tools fetch the same page and keep one table each:

| Tool | Parquet | Table |
|---|---|---|
| `player_stats` | `stats.parquet` | Offensive overview, shooting, foul drawing and defense/rebounding views: one row per player, ~110 columns (rates, shot makes and attempts, `ptile_*` percentiles) |
| `player_onoff` | `onoff.parquet` | The nine on/off views: several rows per player-team, told apart by `position` |

Rows are CTG's as served. In `onoff.parquet`, `position` 0 and 6 appear for every player (0
carries the on-court values and 6 the on-off differences) and 1-5 appear where a player logged
minutes at that position.

`_parse_page` (`collectors/cleaningtheglass/source.py`) finds each tool's table by a column only
that table has (`_TABLE_KEY_COL`), not by its place in the call, and raises if none matches, so
a layout change fails the run instead of writing the wrong table. The call's other arguments,
none captured, are a percentile-only copy of the on/off rows, the column definitions of the 15
views, the default view's name and two flags.

The season goes straight into the URL: `--season` is CTG's starting-year label (the 2025-26
season is `2025`, flipping in September) and `--date` defaults to today. Only the current
season is a valid as-of snapshot; a past season is a final table and would leak into earlier
games. Each run writes `data/player_data/NBA/cleaningtheglass/{season}/{date}/{stats,onoff}.parquet`.

Not captured: team pages (`/stats/teams`, `/stats/league/*`, `/stats/lineups`), which need their
own capture of the page, and playoffs (`seasontype=playoffs`).

## Session cookie

CTG authenticates with session cookies (`sessionid`, a WordPress `wordpress_logged_in_*` login
cookie and `csrftoken`); there is no bearer token. The client refuses to run without one, so
the `creds/keys.json` of the box that runs the job — prod, for the weekly cron — needs the
cookie. The login cookie lasts about a year. The tables do not depend on it today: a request
with an invalid cookie got the same tables, cell for cell (checked 2026-11-11), so the cookie
matters only if CTG gates the page. Capture it once:

1. Log in to <https://cleaningtheglass.com/> in a desktop browser.
2. Open DevTools → **Network**, filter to **Doc**.
3. Load <https://cleaningtheglass.com/stats/players> so the request fires.
4. Right-click the request row → **Copy** → **Copy as cURL (bash)**, saved to a
   temp file (`wl-paste > /tmp/ctg.curl` on Wayland Linux; `pbpaste > …` on macOS).
5. Write the cookie into `creds/keys.json`:

   ```bash
   sportstradamus fetch ctg refresh-auth /tmp/ctg.curl
   ```

   This populates `cleaningtheglass_cookie` (and `cleaningtheglass_user_agent`),
   preserving every other key.

### Verify locally

```bash
sportstradamus fetch ctg list
sportstradamus fetch ctg run --dry-run
sportstradamus fetch ctg run --only player_stats
sportstradamus fetch ctg verify
```

## Weekly cron

The schedule, and why it sits where it does, live in [OPERATIONS.md](OPERATIONS.md). Set
`HEALTHCHECK_URL_CTG_FETCH` so an expired cookie alerts immediately.

## When the run fails

`_dispatch` raises the framework's auth error when the page comes back without the data call,
which means CTG gated the page or changed its layout: on a TTY the run prompts for a fresh
curl, under cron it fails fast and Healthchecks.io pings `/fail`. A lapsed cookie alone does not
trip it, because the page does not need the login today. Nothing renews the cookie from a
stored login, as `fetch fp` does: with a year's life and no dependence on it, that is not
built. `GET /?ctg_check` and `GET /stats/api/ctg_check` answer `{"logged_in": true}` or
`false` for a cookie, a quick manual test.

## Wiring the features

`StatsNBA._join_fp_player_features` / `_join_fp_team_features` already delegate to the shared
as-of loader. The `_CTG_*` join constants in `stats/nba.py` are deliberately still empty, so NBA
features stay gamelog-only. Wiring them is a separate decision: it adds about 340 `{col}_asof`
columns to every NBA cell, all NaN until snapshots accumulate, and the loader needs one row per
player, which `onoff.parquet` does not give (cut it to a `position` first). Both tables carry
`player_id` and `team_id`, the join keys.

## Moving snapshots

The job runs on prod; the weekly `sync_from_prod.sh` copies its snapshots to the dev box that
trains. A snapshot captured on the dev box goes up with a full `scripts/sync_to_prod.sh`
(additive, never `--delete`; `--league` sends none) — see [OPERATIONS.md](OPERATIONS.md).
