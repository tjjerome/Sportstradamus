# Fantasy Points Data Suite snapshotter

`fp-fetch` walks a catalog of Fantasy Points Data Suite endpoints and
writes each tool's response to disk every week. The catalog and the
session token are the only two things you maintain.

This page covers FP-specific setup. The shared collector framework — the
`run`/`verify`/`list`/`refresh-auth` CLI surface, curl→keys.json auth rotation,
the skip-if-on-disk contract, and how to add a new source — lives in
[data_collectors.md](data_collectors.md); FP is its week-keyed member.

## Before you start: legal

Fantasy Points' Terms of Service may restrict automated access to the
Data Suite even for paying subscribers. Read them and decide whether to
proceed; the scraper does nothing to evade detection (single-process,
2 s pause between calls, realistic User-Agent, no parallelism), so
account suspension is a realistic risk if their ToS forbids this.

## One-time setup

### 1. Grab a fresh `Authorization` token

The Data Suite v2 API authenticates via a bearer-style `Authorization`
header (not a cookie). You capture it once per session:

1. Log in to <https://data.fantasypoints.com/> in a desktop browser.
2. Open DevTools → **Network** tab, filter to **Fetch/XHR**.
3. Click any tool (line matchups is fine) so the SPA fires its data
   call.
4. Click the request row in the Network list → **Headers** panel →
   **Request Headers**. Copy the *value* of the `Authorization:`
   header (everything after `Authorization: `).
5. (Optional but recommended) Copy the `Cookie:` header value too —
   some endpoints want analytics/session cookies alongside the token.
6. Paste into `src/sportstradamus/creds/keys.json`:

```json
{
  "fantasypoints_authorization": "Bearer eyJ...",
  "fantasypoints_cookie": "_shopify_y=...; ..."
}
```

Optionally set `fantasypoints_user_agent` to your browser's UA in the
same file — the default Firefox UA is fine for most users.

The tokens rotate on a schedule we don't control. When they expire the
weekly job alerts via Healthchecks.io and you redo this step.

### 2. Register endpoints

Two paths — pick one (or combine):

**Bulk: `fp-fetch discover`** — auto-populates from FP's tool registry.

```bash
poetry run fp-fetch discover --dry-run    # preview new entries
poetry run fp-fetch discover              # write them
```

`discover` hits `POST /v2/ds/all/tools`, walks every published tool,
and adds one catalog entry per `(tool, context)` pair (e.g.
`passingBasic` with `context: ["player","team","opponent"]` becomes
three entries):

- `player` → `POST .../tools/player/{slug}/values`
- `team` → `POST .../tools/team/{slug}/values` (team's offensive
  view — there is no `/offense/` segment, the SPA hits the bare
  `/team/{slug}` path for offense)
- `opponent` → `POST .../tools/team/defense/{slug}/values` (team's
  defensive view, one row per opponent faced)

Existing names are preserved by default — re-run any time and only
new tools land. Pass `--replace` to discard the existing catalog
and regenerate from scratch; needed once after upgrading from a
pre-v2 catalog so old bodies pick up the integer week/season
sentinels (otherwise FP keeps returning whole-season data
regardless of `--week`).

Defaults: skips `isPrivate: true` (debug / VIP-only / concept
tables); pass `--include-private` to keep them. Skips the `other`
context (no observed URL pattern). League defaults to `nfl`; override
with `--league wnba` etc. once FP launches one.

Per-tool body matches the v2 `/values` contract observed via
DevTools: `tableProperty`, `routeContextTarget`, integer
`weeks: {"REG": [N]}`, `filterMatch.game.season.eq`, plus the
per-tool `requires*` and `requiredRoles` flags pulled from the
registry entry. Week + season are stored as sentinel strings
(`__WEEK_INT__`, `__SEASON_INT__`) and rewritten to integers per
call by `body_substitute.substitute_runtime`. The `weeks` block is
also re-shaped per mode (see `--mode` below).

**Manual: `fp-fetch import-curl`** — for the per-tool path:

1. Open the tool in your browser with DevTools' **Network** tab open
   and filter by `Fetch/XHR`.
2. Click around until the tool fires its data call (usually on page
   load).
3. Right-click the XHR row → **Copy** → **Copy as cURL (bash)**.
4. Paste into a temp file:

   ```bash
   pbpaste > /tmp/line_matchups.curl   # macOS
   wl-paste > /tmp/line_matchups.curl  # Wayland Linux
   ```

5. Register:

   ```bash
   poetry run fp-fetch import-curl /tmp/line_matchups.curl \
       --name line_matchups \
       --output-subdir team/line_matchups
   ```

   The endpoint lands in
   `src/sportstradamus/data/config/fantasypoints_endpoints.json`.

`import-curl` handles both GET and POST out of the box. POST requests
with a `--data-raw '{...}'` JSON body are parsed and stored in the
catalog as `json_body`. `Authorization`, `Cookie`, and `User-Agent`
headers are stripped automatically (they come from `creds/keys.json`).

Two substitution paths coexist for hand-imported entries:

- **Legacy `{week}` / `{season}` string substitution** — recursive
  through nested dicts and lists; only strings containing the literal
  placeholders are touched. Useful for v1-style endpoints with a
  `filters.{week,season}` shape.
- **Integer sentinels** (`__WEEK_INT__`, `__SEASON_INT__`) — used by
  discover-generated entries. Rewritten to integers per call so FP's
  v2 schema (which type-checks these as ints) accepts the body.

For season-long aggregates that should not be re-fetched per week,
pass `--season-long` at `import-curl` time (or set `"weekly": false`
in the catalog entry).

#### Patching a discover-generated entry (`--replace`)

A handful of FP tools return 0 rows when called with the minimal
body discover generates, because the SPA injects per-tool filters
(position, stat-availability qualifier) that the registry doesn't
expose. Symptom: those tools land in the report with `status: "empty"`
and a `response_preview` showing `"count": 0`.

Fix: capture a working DevTools curl for the failing tool, then
overlay it onto the existing catalog entry with `--replace`:

```bash
pbpaste > /tmp/passing_advanced.curl
poetry run fp-fetch import-curl /tmp/passing_advanced.curl \
    --name player_passing_advanced \
    --replace
```

`--replace` preserves the existing `output_subdir` (so you don't
have to re-specify it) and rewrites the captured body's literal
`game.season.eq` value to the `__SEASON_INT__` sentinel — the entry
stays usable across seasons even though you captured it on one
specific week. The `weeks` block doesn't need sentinelisation: the
runtime substitutor overrides it per call based on `--mode`.

### 3. Verify locally

```bash
poetry run fp-fetch list
poetry run fp-fetch run --week 5 --season 2025 --dry-run
poetry run fp-fetch run --week 5 --season 2025 --only team_line_matchups
```

`run` fetches each endpoint, parses `content.rows.values` (v2
endpoints) or `content.table.rows.values` (legacy) into a pandas
DataFrame, and writes one parquet per (tool, week, mode), grouped
into a per-week subfolder so 45 files don't clutter the season dir:

- player-context entries →
  `src/sportstradamus/data/player_data/NFL/{season}/week_NN/{tool}{mode_suffix}.parquet`
- team-context entries →
  `src/sportstradamus/data/team_data/NFL/{season}/week_NN/{tool}{mode_suffix}.parquet`
- opponent-context entries → same `team_data/` directory with an
  `_opp` infix in the filename (e.g.
  `passing_basic_opp.parquet`, `passing_basic_opp_s2d.parquet`).

`mode_suffix` is empty for `--mode weekly` (the default, kept blank
so existing weekly parquets don't have to be renamed) and for
`--mode postseason` (postseason rounds live in their own
`week_19`..`week_22` folders so no filename collision is possible).
`_s2d` is used for `--mode season_to_date` because it shares the
regular-season `week_NN` folder with `weekly`. Modes for the same
`(tool, week)` write to different files / folders and do not
overwrite each other.

Postseason folder mapping: postseason round 1 (wildcard) → `week_19`,
2 (divisional) → `week_20`, 3 (conference championship) → `week_21`,
4 (super bowl) → `week_22`. The CLI continues to take `--week 1..4`
for postseason mode; the `+18` shift happens during path resolution.

#### `--mode` flag

```bash
# default — one regular-season week only (REG: [N])
poetry run fp-fetch run --week 5 --season 2025

# season-to-date through week N (REG: [1..N])
poetry run fp-fetch run --week 5 --season 2025 --mode season_to_date

# postseason week N (POST: [N])
poetry run fp-fetch run --week 1 --season 2025 --mode postseason
```

The mode rewrites `context.weeks` in the request body so FP returns
exactly the slice you ask for. Bodies in the catalog ship with
`weeks: {"REG": ["__WEEK_INT__"]}` as the canonical shape and
`body_substitute.substitute_runtime` rewrites both the int and the
mode shape per call.

Re-running the same (week, mode) **skips** cells that already have a
non-empty parquet — `run` and `backfill` both default to "don't
re-download what's already on disk", so re-executing after a partial
failure (or resuming a half-finished backfill) is near-instant for
the cells that already succeeded. Pass `--refetch` to force a
re-download regardless. Zero-row parquets (failed previous fetches)
are always re-fetched. Raw JSON is not persisted — the parquet is the
deliverable. If a parse fails, the diagnostic includes Content-Type /
Content-Encoding so you can spot a stale catalog Accept-Encoding or a
wholesale API change.

### 4. Spot-check the download

After a run, sanity-check what landed on disk against what you
asked for:

```bash
poetry run fp-fetch verify --week 5 --season 2025
poetry run fp-fetch verify --week 5 --season 2025 --mode season_to_date
poetry run fp-fetch verify --week 5 --season 2025 --only player_passing_basic
```

For every catalog entry the verifier:

- Confirms the expected parquet exists at the routed path.
- Loads the file and checks ``gameSeason`` matches the requested
  season exactly (no stray rows from other years).
- Checks ``gameWeek`` matches the week set implied by ``--mode``:
  ``{N}`` for ``weekly`` / ``postseason``, ``{1..N}`` for
  ``season_to_date``.
- When ``gameType`` is present, confirms ``weekly`` /
  ``season_to_date`` carries only regular-season games and
  ``postseason`` carries only playoff games.

Output is one line per spec (``OK`` / ``WARN`` / ``FAIL``) with
indented issue detail when something's off. Exit code is non-zero
if any spec hits an error so you can chain it into a script.

This is the check that catches the pre-v2 "no week filter" bug —
if you upgrade an older catalog without running ``discover
--replace``, ``verify`` will FAIL every spec with ``week_mismatch``
pointing at the same fix.

### Token expired mid-run

If the Authorization or Cookie expires partway through `fp-fetch
run`, the CLI pauses, prints a banner, and waits for you to paste
a fresh DevTools curl on stdin (end with EOF / Ctrl+D). It updates
`creds/keys.json` and the in-memory client, then retries the
failing call and resumes the batch. Non-TTY contexts (cron) skip
the prompt and fail fast so Healthchecks.io pings `/fail`.

## Historical backfill

```bash
poetry run fp-fetch backfill \
    --start-season 2021 --end-season 2024 \
    --start-week 1 --end-week 18
```

Iterates every (season, week) pair and runs the same fetch +
parse + write as `run`. Pacing is conservative by default:

- **2–8 s** random pause between endpoints in the same week
  (`--request-pause-min` / `--request-pause-max`).
- **8–28 s** random pause when transitioning to a new week
  (`--week-pause-min` / `--week-pause-max`).

With ~45 tools × 18 weeks × N seasons at the defaults plan for
several hours per season — designed for an overnight one-time
grab, not a cron job. `--only`, `--dry-run`, and `--mode` work the
same as on `run` (e.g. `--mode season_to_date` to backfill
cumulative weekly snapshots).

## Weekly cron

On the production box (see `CLAUDE.md` for the full crontab):

```cron
0 10 * * 3   /home/sportstradamus/Sportstradamus/scripts/run_job.sh fp-fetch
```

Wednesday 10:00 server time — after Monday/Tuesday stat corrections
settle, well before Sunday games. Set `HEALTHCHECK_URL_FP_FETCH` in
the environment so token-expiry failures alert via Healthchecks.io.

## When the token expires

The healthcheck alert (`/fail` ping with the last 50 log lines) quotes a `401`
message pointing here. Refresh in one paste with
`poetry run fp-fetch refresh-auth /tmp/fresh.curl` (or `pbpaste | … refresh-auth -`),
then confirm with `fp-fetch run --only line_matchups`. The extract-headers,
preserve-other-keys, redacted-preview mechanics are the shared refresh-auth flow
documented in [data_collectors.md](data_collectors.md#auth).

## Output layout

```
src/sportstradamus/data/
  player_data/NFL/{season}/week_NN/{tool}{mode_suffix}.parquet
  team_data/NFL/{season}/week_NN/
      {tool}{mode_suffix}.parquet         # team-offense view
      {tool}_opp{mode_suffix}.parquet     # opponent (team-defense) view
```

`week_NN` is `01..18` for regular-season modes and `19..22` for
postseason rounds (wildcard / divisional / conf championship /
super bowl). `mode_suffix` is `""` (weekly, default), `_s2d`
(season-to-date), or `""` (postseason — distinguished by its
`week_19..22` folder, no suffix needed). Re-running the same
`(week, mode)` overwrites; different modes coexist without
collision because they target different filenames (`_s2d`) or
different folders (postseason).

## Adding new endpoints later

Re-run `fp-fetch import-curl` whenever Fantasy Points adds a new tool
you want snapshotted. Existing catalog entries are untouched.
