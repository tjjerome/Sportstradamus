# Fantasy Points Data Suite snapshotter

`fp-fetch` walks a catalog of Fantasy Points Data Suite endpoints and
writes each tool's response to disk every week. The catalog and the
session token are the only two things you maintain.

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
three entries). Existing names are preserved — re-run any time and
only new tools land.

Defaults: skips `isPrivate: true` (debug / VIP-only / concept
tables); pass `--include-private` to keep them. Skips the `other`
context (no observed URL pattern). League defaults to `nfl`; override
with `--league wnba` etc. once FP launches one.

Per-tool body uses the larger of the two observed request shapes
(`useCache`/`flatten`/`isInitial`/`withValues`/`compress` + a
`context` block). If a tool needs a `filters.{week,season}` body
instead, re-run `import-curl` on its curl to override the auto-
generated entry.

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

If you want week/season substitution into the JSON body (FP's v2
endpoints we've seen so far don't need this — the response carries
all weeks and the SPA filters client-side), open the catalog JSON and
replace literal values with `"{week}"` / `"{season}"`:

```json
{
  "name": "line_matchups",
  "url": "https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups",
  "method": "POST",
  "output_subdir": "team/line_matchups",
  "json_body": {
    "context": {"grouping": "$team.teamId", "routeContext": "team"},
    "filters": {"week": "{week}", "season": "{season}"},
    "useCache": true
  }
}
```

Substitution works recursively through nested dicts and lists; only
strings that contain `{week}` or `{season}` are touched.

For season-long aggregates that should not be re-fetched per week,
pass `--season-long` at `import-curl` time (or set `"weekly": false`
in the catalog entry).

### 3. Verify locally

```bash
poetry run fp-fetch list
poetry run fp-fetch run --week 5 --season 2025 --dry-run
poetry run fp-fetch run --week 5 --season 2025 --only line_matchups
```

The snapshot lands at
`src/sportstradamus/data/fantasypoints/2025/week_05/team/line_matchups.json`.

## Weekly cron

On the production box (see `CLAUDE.md` for the full crontab):

```cron
0 10 * * 3   /home/sportstradamus/Sportstradamus/scripts/run_job.sh fp-fetch
```

Wednesday 10:00 server time — after Monday/Tuesday stat corrections
settle, well before Sunday games. Set `HEALTHCHECK_URL_FP_FETCH` in
the environment so token-expiry failures alert via Healthchecks.io.

## When the token expires

The healthcheck alert (`/fail` ping with the last 50 log lines) will
quote a message like:

```
FP returned 401 for POST https://...  — Authorization token is
expired or missing. Refresh fantasypoints_authorization in
creds/keys.json; see docs/fantasypoints.md.
```

Refresh in one step:

1. In DevTools, copy any logged-in XHR as cURL (the same recipe as
   §1, step 4, but to a temp file):

   ```bash
   pbpaste > /tmp/fresh.curl   # macOS
   wl-paste > /tmp/fresh.curl  # Wayland Linux
   ```

2. Update `creds/keys.json`:

   ```bash
   poetry run fp-fetch refresh-auth /tmp/fresh.curl
   ```

   Or pipe directly without the temp file:

   ```bash
   pbpaste | poetry run fp-fetch refresh-auth -
   ```

   The command extracts the `Authorization`, `Cookie`, and
   `User-Agent` headers and writes them to `creds/keys.json`,
   preserving every other field. It prints a redacted preview of
   each value (first 24 chars) so you can confirm without leaking
   the full token to a shared terminal.

3. Confirm:

   ```bash
   poetry run fp-fetch run --only line_matchups
   ```

## Output layout

```
src/sportstradamus/data/fantasypoints/
  YYYY/                       # season
    week_NN/                  # 01..18
      <output_subdir>.<ext>   # one file per catalog entry
    season/                   # entries flagged "weekly": false
      <output_subdir>.<ext>
```

Re-running the same week overwrites; nothing is appended.

## Adding new endpoints later

Re-run `fp-fetch import-curl` whenever Fantasy Points adds a new tool
you want snapshotted. Existing catalog entries are untouched.
