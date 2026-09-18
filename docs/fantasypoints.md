# Fantasy Points Data Suite snapshotter

`fp-fetch` walks a catalog of Fantasy Points Data Suite endpoints and
writes each tool's response to disk every week. The catalog and the
session cookie are the only two things you maintain.

The API is `fantasypointsdata.com/api/nfl/{tool}` — a GET per tool, filters
in the query string, and a bare JSON array of flat rows back. The columns it
returns are **not** the ones the NFL models were trained on; the collector
translates them back to the legacy vocabulary on the way to parquet. See
[Column translation](#column-translation) for what that means when you add a
tool. What the new API makes newly *possible* is surveyed separately in
[fantasypoints_expansion.md](fantasypoints_expansion.md).

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

### 1. Store your login

The API authenticates on the `ds_session` cookie alone. It accepts nothing
else — an `Authorization` header is ignored, and a request carrying only
one gets a 401. The cookie is a **seven-day** token (its own payload
carries the expiry) and it does **not** slide: using it does not push the
expiry out, so it lapses a week after login no matter how often the
collector runs.

That would mean a DevTools paste every week, so the collector signs in for
itself instead. Put your **fantasypoints.com** account credentials — the
email and password the site's sign-in form takes — in
`src/sportstradamus/creds/keys.json`:

```json
{
  "fantasypoints_email": "<your fantasypoints.com email>",
  "fantasypoints_password": "<your password>"
}
```

Then prove they work:

```bash
sportstradamus fetch fp login
```

That makes the same two hops the sign-in form makes — Firebase trades the
email and password for an `idToken`, then `/api/auth/firebase-login` trades
that for the cookie — writes the result into `fantasypoints_cookie`, and
reports the length. From then on `run` renews the cookie by itself the first
time a call comes back 401 — under cron too, with no TTY — so nothing has to
be pasted again.

Firebase is the only way in, so the email must be the one the account was
created with. The sign-in page also shows an account-name form posting to
`/api/auth/login`, but that path is disabled site-wide and answers `404 Not
found.` for any Firebase-backed account, which is every account now.
Google and Apple sign-in are not supported here — they need an interactive
popup — so an account created that way has to be given a password on
fantasypoints.com before the collector can use it.

`fantasypoints_authorization` is no longer read; delete it from both
boxes. Optionally set `fantasypoints_user_agent` to your browser's UA —
the default Firefox UA is fine for most users.

If you would rather not store the password, paste a cookie into
`fantasypoints_cookie` by hand (DevTools → Network → any `/api/nfl/` XHR →
copy the `Cookie:` header value) and accept the weekly refresh; without
credentials there is nothing for the collector to renew from.

### 2. Register endpoints

There is no tool registry to enumerate — the SPA has no equivalent of the
old `/v2/ds/all/tools` — so every entry comes from a captured request:

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
   sportstradamus fetch fp import-curl /tmp/coverage_matrix.curl \
       --name team_coverage_matrix \
       --output-subdir team/coverage_matrix
   ```

   The endpoint lands in
   `src/sportstradamus/data/config/fantasypoints_endpoints.json`.

`Authorization`, `Cookie` and `User-Agent` are stripped automatically (they
come from `creds/keys.json`), along with the browser telemetry a "Copy as
cURL" drags in — `Referer`, `Origin`, `Accept-Language`, `Sec-Fetch-*`, `DNT`,
`Sec-GPC`, `Priority`, `TE` and the rest. `Referer` is the one that matters:
it records whichever page you had open, so keeping it would make two captures
of the same tool produce two different entries, and the client sets its own.

**Name the entry `player_` / `team_` / `opponent_` + the file kind you
want**, because the prefix picks `player_data/` vs `team_data/` and the
rest becomes the parquet basename, which is what `FILE_KINDS` in
`stats/nfl_fp_weekly.py` and `stats/nfl_fp_team_weekly.py` look for.

The captured query string is stored as `params` verbatim, minus the
period: every entry carries `"seasons": "{season}"` and
`"regWeeks": "{week}"`, templated so the dry-run listing and the run
report print a URL you can paste into a browser. The source overrides
both per call from `--season` / `--week` / `--mode`, so a captured
literal would be a lie rather than a bug — `--replace` normalises it
either way. Keep every *other* captured filter: `mode=offense|defense`,
`positions=`, and any situational slice (`down=3`, `ydsToScoreMax=10`,
`scoreDiffMin=7`) are what define the entry.

For season-long aggregates that should not be re-fetched per week,
pass `--season-long` at `import-curl` time (or set `"weekly": false`
in the catalog entry).

#### Patching an existing entry (`--replace`)

When a tool gains a filter, re-capture it and overlay the new request
onto the existing catalog entry:

```bash
pbpaste > /tmp/passing_advanced.curl
sportstradamus fetch fp import-curl /tmp/passing_advanced.curl \
    --name player_passing_advanced \
    --replace
```

`--replace` preserves the existing `output_subdir`, so you don't have to
re-specify it.

### 3. Verify locally

```bash
sportstradamus fetch fp list
sportstradamus fetch fp run --week 5 --season 2025 --dry-run
sportstradamus fetch fp run --week 5 --season 2025 --only team_coverage_matrix
```

`run` fetches each endpoint, parses the returned JSON array into a pandas
DataFrame, and writes one parquet per (tool, week, mode), grouped
into a per-week subfolder so 56 files don't clutter the season dir:

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
sportstradamus fetch fp run --week 5 --season 2025

# season-to-date through week N (REG: [1..N])
sportstradamus fetch fp run --week 5 --season 2025 --mode season_to_date

# postseason week N (POST: [N])
sportstradamus fetch fp run --week 1 --season 2025 --mode postseason
```

The mode sets the period query parameters, which are merged over the
catalog entry's own params per call (`source.period_params`):

| mode | parameters |
|---|---|
| `weekly` | `seasons=2025&regWeeks=5` |
| `season_to_date` | `seasons=2025&regWeeks=1,2,3,4,5` |
| `postseason` | `seasons=2025&regWeeks=&postWeeks=1` |

`regWeeks=` with an empty value returns zero regular-season rows, which is
how a postseason request excludes them.

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
sportstradamus fetch fp verify --week 5 --season 2025
sportstradamus fetch fp verify --week 5 --season 2025 --mode season_to_date
sportstradamus fetch fp verify --week 5 --season 2025 --only player_passing_basic
```

Each response row is one aggregate per entity over the requested weeks,
so there is no per-row week column to check. What every row does carry is
``games`` — how many of that entity's games went into the aggregate — and
that is exactly what the week filter controls. For every catalog entry the
verifier:

- Confirms the expected parquet exists at the routed path.
- Checks no row aggregates more games than ``--mode`` allows: 1 for
  ``weekly`` / ``postseason``, N for ``season_to_date`` through week N.
  This is the check that catches a week filter that didn't bind — an
  unfiltered response still lands at the right path with a healthy row
  count, and fails here on 17 games per entity.
- Confirms the identity columns survived (``playerPlayerId`` /
  ``playerFirstName`` / ``playerLastName`` on player entries,
  ``teamTeamId`` / ``teamAbbreviation`` on team ones). A dropped identity
  column is indistinguishable from an empty week downstream and silently
  removes every feature built on the file.

Output is one line per spec (``OK`` / ``WARN`` / ``FAIL``) with
indented issue detail when something's off. Exit code is non-zero
if any spec hits an error so you can chain it into a script.

### Cookie expired mid-run

The first 401 triggers a login, and the failing call is retried with the
fresh cookie — the batch continues and `keys.json` is updated in passing.
This needs no terminal, so the weekly cron recovers on its own.

Only when there are no stored credentials (or the login is rejected) does
it fall back to the old behaviour: at a TTY it pauses and waits for a
fresh DevTools curl on stdin (end with EOF / Ctrl+D); under cron it fails
fast so Healthchecks.io pings `/fail`.

## Historical backfill

```bash
sportstradamus fetch fp backfill \
    --start-season 2021 --end-season 2024 \
    --start-week 1 --end-week 18
```

Iterates every (season, week) pair and runs the same fetch +
parse + write as `run`. Pacing is conservative by default:

- **2–8 s** random pause between endpoints in the same week
  (`--request-pause-min` / `--request-pause-max`).
- **8–28 s** random pause when transitioning to a new week
  (`--week-pause-min` / `--week-pause-max`).

With ~56 tools × 18 weeks × N seasons at the defaults plan for
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

## When auth fails anyway

A `/fail` healthcheck quoting a `401` now means the *sign-in* failed, not that
the cookie lapsed — the cookie renews itself. The alert quotes the reason
directly, and Firebase's codes say which half is wrong: `EMAIL_NOT_FOUND`
means the stored email is not an account, `INVALID_PASSWORD` means the email
is right and only the password is stale. `sportstradamus fetch fp login`
reproduces it on demand.

A rejected sign-in **stops the run** rather than moving to the next endpoint.
The credential is shared by every endpoint, so finishing the walk cannot
succeed — it would only fire one more refused sign-in per endpoint, and
dozens of those in a burst is what gets an account locked. The run report is
still written, so whatever was fetched before the credential died stays
visible.

The manual paths still work if you need them:
`sportstradamus fetch fp login` to re-mint from credentials, or
`sportstradamus fetch fp refresh-auth /tmp/fresh.curl` to install a cookie
captured by hand. The extract-headers, preserve-other-keys, redacted-preview
mechanics are the shared refresh-auth flow documented in
[data_collectors.md](data_collectors.md#auth).

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

## Column translation

The API's column names are its own — `yards`, `cpoe`, `man_routes` — and none
of them is what the NFL models were trained on. Those names are frozen: the
`expected_columns` list inside every NFL model pickle is sliced strictly at
serve time, so a column that stops appearing is a `KeyError` in production,
not a degraded feature. Rather than rename 104 aggregate outputs and retrain,
the collector translates back on the way to parquet.

`config/fantasypoints_column_map.json` holds the translation, keyed by file
kind, and `collectors/fantasypoints/column_map.py` applies it. Three additive
sections:

- **`rename`** copies a column to its legacy camelCase name, applying any
  **`scale`**. Those carry the legacy conventions the API dropped: rates
  stored as fractions where the API reports percents (`scale: 0.01`), and
  sack yardage stored as a loss (`scale: -1.0`).
- **`derive`** builds a legacy rate from a numerator/denominator pair in the
  row's `__raw` block. Preferred wherever the API's displayed rate is
  pre-rounded, since the counts are exact.
- **`bucket`** re-nests flat `{bucket}_{stat}` columns into the single JSON
  cell the legacy schema carried, so the bucket-parsing aggregators read
  archived legacy snapshots and new pulls alike.

Every section is additive — the new-schema columns stay on the frame, so one
parquet serves both the frozen feature set and anything built on the wider
schema later.

`tests/test_fantasypoints.py::test_every_aggregate_output_column_survives_the_new_schema`
walks the catalog, the map and both recipe tables and fails if any aggregate
output column has become underivable. It is the gate on the silent chain that
otherwise ends at a serve-time `KeyError`: a recipe whose source columns are
missing is skipped without a word.

### What the port cost

The cutover was verified by re-pulling a week that already existed as a legacy
snapshot and diffing the production aggregates: **team 16/16 and defense 7/7
columns, zero suspect; player 154/154 columns present**, of which 150 are
numeric and comparable. Of those 150, four differences are real and the rest is
±1 charting noise (86–99% of entities exact, near-zero-mean residuals).

The four are FP redefinitions with no closer endpoint or filter, so the port
takes the substitutes rather than the retrain it exists to avoid:

| aggregate column | correlation | cause |
|---|---|---|
| `eff_XFP` | 0.898 | FP redefined expected fantasy points (~2% high) |
| `rush_bellcow_XFP_pct` | 0.948 | same xFP redefinition |
| `bellcow_score` | derived | composite of the above |
| `rush_adv_EXP_YDS` | 0.915 | explosive-run yardage threshold moved |

All four are low-weight inputs. Re-examine if one shows up in a ship-gate
regression.

### The cross-era identity split

Legacy and new snapshots do not share an identity space. Legacy
`playerPlayerId` is an FP-internal hex id, the new one is a GSIS id
(`00-0039424`), and there is **zero** overlap between them; legacy
`teamTeamId` is an integer, the new one the 3-letter abbreviation.

Within one era this is harmless — the player id is only ever a groupby key
before the result is re-keyed to the player's name, and the team abbreviation
makes `_build_team_abbreviation_map` an identity mapping. **Across eras it is
not.** A lookback window that spans the cutover splits each entity into two
aggregate rows, and the name-index projection keeps only one of them.

There is no transform that fixes this: the two id spaces carry no shared key.
The only clean resolution is re-pulling the historical seasons through the new
API, which `seasons=2022` confirms is possible. Until that happens, treat any
window straddling the cutover week as unreliable. Scoping that re-pull is the
first item in
[fantasypoints_expansion.md](fantasypoints_expansion.md#sequencing-and-the-retrain-rule).

## Adding new endpoints later

Re-run `fp-fetch import-curl` whenever Fantasy Points adds a new tool
you want snapshotted. Existing catalog entries are untouched.

If the new tool feeds an existing aggregate recipe, add its file kind to
`fantasypoints_column_map.json` too — the catalog entry alone gets the file on
disk under new-schema names, which no recipe reads.
