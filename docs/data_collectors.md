# Data collectors

A **collector** snapshots an external data provider's tables to parquet on a
schedule, and the stats layer folds those snapshots into player/team features at
prediction and training time. Three sources run on this framework today:

| Source | CLI | League | Auth | Period key |
|---|---|---|---|---|
| Fantasy Points Data Suite | `fp-fetch` | NFL | bearer + cookie | game **week** |
| Cleaning the Glass | `ctg-fetch` | NBA | session cookie | capture **date** |
| Baseball Savant (Statcast) | `savant-fetch` | MLB | none (public) | capture **date** |

Everything shared lives in `src/sportstradamus/collectors/`; each source is a
thin package under it. This document is the canonical description of the
framework and the recipe for adding a fourth source. The per-source specifics —
where to get the cookie, which endpoints to register — live in
[fantasypoints.md](fantasypoints.md), [cleaningtheglass.md](cleaningtheglass.md),
and [baseballsavant.md](baseballsavant.md).

## Architecture

```
src/sportstradamus/collectors/
  auth.py            keys.json read/write + curl→keys refresh (shared)
  transport.py       CookieClient (bearer/cookie HTTP) + auth errors
  catalog.py         EndpointSpec + load/save + {season}/{week}/{date} substitution
  runner.py          generic fetch→parse→write loop, skip-if-on-disk, run report
  report.py          RunResult + report writer + build_url
  dispatch.py        per-spec error capture + interactive auth refresh
  commands.py        week-centric run/backfill/list/verify/refresh-auth builders
  commands_dated.py  date-centric run/verify builders (cumulative sources)
  cli.py             Source dataclass + build_source_cli(source) -> click.Group
  tabular.py         CSV/JSON→DataFrame parse + dated player/team path routing
  fantasypoints/     FP_SOURCE + FP-only discover / import-curl / body sentinels
  cleaningtheglass/  CTG_SOURCE (cookie, date-keyed)
  baseballsavant/    SAVANT_SOURCE (public Scrape, date-keyed)
```

A source is a `Source` dataclass (`cli.py`) — **data and callables, not a
subclass hierarchy**. Its fields:

| Field | Meaning |
|---|---|
| `name`, `help` | CLI name and group help text |
| `catalog_path` | the endpoint-catalog JSON under `data/config/` |
| `make_client(resolved_auth, sleep)` | build the HTTP client (a `CookieClient`, or a `Scrape` for public sources) |
| `default_context(season, week)` | resolve the season/week label when the operator omits it |
| `path_for(spec, **period)` | where a fetched tool's parquet lands |
| `dispatch(client, spec, **period)` | perform one request, return raw body |
| `transform(body)` | parse the body into a DataFrame |
| `report_prefix` | prefix for the run-report JSON |
| `auth_fields` | `AuthFields` naming the keys.json slots, or `None` for a public source |
| `env_prefix` | env-var prefix for auth overrides (`{PREFIX}_COOKIE`, …) |
| `period_kind` | `"week"` (FP) or `"date"` (CTG/savant) — selects the command set |
| `has_backfill`, `render_request_body`, `verify_fn` | FP-only extras (default off) |

`build_source_cli(source)` returns a real `click.Group` bound at module level
(e.g. `ctg_fetch = build_source_cli(CTG_SOURCE)`). It attaches `run`, `verify`,
`list`, plus `refresh-auth` when the source is authed and `backfill` when it opts
in. `period_kind` decides whether `run`/`verify` come from `commands.py` (week,
with `--week`/`--mode`) or `commands_dated.py` (date, with `--date`).

## The CLI surface

Every source exposes:

```bash
poetry run <name>-fetch list                      # print the registered catalog
poetry run <name>-fetch run [--dry-run] [--only …]  # fetch every endpoint, write parquets
poetry run <name>-fetch verify …                  # confirm the parquets landed with rows
poetry run <name>-fetch refresh-auth <curl>       # (authed sources) rotate the token/cookie
```

`run` is idempotent: it skips a cell whose parquet already exists non-empty, so
re-running after a partial failure only fetches the gaps. `--refetch` forces a
re-download; zero-row parquets are always retried. `--dry-run` lists what would
be fetched, touching nothing — the capture-stub-drift guard: dry-run one date
before any real (paid or rate-limited) run to confirm auth and the catalog.

**Period kinds.** Week-keyed sources (`fp-fetch`) take `--week`/`--season` and a
`--mode`; each snapshot is one game week. Date-keyed sources (`ctg-fetch`,
`savant-fetch`) take `--season`/`--date` and snapshot the provider's
already-cumulative season-to-date table, stamped by the capture date (default:
today). Date sources do not carry `--week`, `--mode`, or `backfill` — the
provider only serves "as of now".

## Auth

`AuthFields` names up to three keys.json slots — `authorization`, `cookie`,
`user_agent` — and a `None` slot means the source doesn't use it. FP uses a
bearer `authorization` + `cookie`; CTG uses `cookie` + `user_agent` only; savant
sets `auth_fields=None` (public — no keys.json entry, no `refresh-auth` command).

`read_auth` resolves each slot from the env (`{ENV_PREFIX}_COOKIE`, …) first,
then keys.json, defaulting to `""` when a key is absent — so a source works
before its slot is populated (it just fails the request with a clear "empty
cookie" message).

**Rotating a token** is a one-paste operation. Copy any logged-in request from
DevTools as cURL, then:

```bash
sportstradamus fetch ctg refresh-auth /tmp/fresh.curl   # or: pbpaste | … refresh-auth -
```

`refresh-auth` extracts the `Authorization`/`Cookie`/`User-Agent` headers, writes
them to `creds/keys.json` preserving every other field, and prints a redacted
preview (first 24 chars). If a token expires **mid-run** on a TTY, the CLI pauses,
prompts for a fresh curl, updates the client, and resumes; under cron (non-TTY)
it fails fast so Healthchecks.io pings `/fail`.

## Snapshot layout & the leakage rule

Collector snapshots live in **dedicated subdirs**, never at the league root
beside `nba_players_*.csv` / `affinity_*.csv` (so a scoped, additive sync can't
clobber siblings):

```
# week-keyed (Fantasy Points)
data/player_data/NFL/{season}/week_NN/{tool}{mode_suffix}.parquet
data/team_data/NFL/{season}/week_NN/{tool}…parquet

# date-keyed (CTG / savant)
data/player_data/NBA/cleaningtheglass/{season}/{YYYY-MM-DD}/{tool}.parquet
data/team_data/NBA/cleaningtheglass/{season}/{YYYY-MM-DD}/{tool}.parquet
data/player_data/MLB/baseballsavant/{season}/{YYYY-MM-DD}/{tool}.parquet
data/team_data/MLB/baseballsavant/{season}/{YYYY-MM-DD}/{tool}.parquet
```

`tabular.dated_path_for` routes a spec named `player_<tool>` to `player_data/` and
`team_<tool>` to `team_data/`, stripping the prefix for the filename.

**The leakage rule.** Date-keyed snapshots are cumulative season-to-date, so the
consumer must select the most recent snapshot **strictly before** the game date —
a snapshot captured the morning of a game may already fold that game in.
`DatedSnapshotStore.latest_before` enforces the strict `<`; this mirrors FP's
"week − 1" rule.

## Consumption: folding snapshots into features

The stats base class exposes two per-league hooks
(`stats/base.py`), both no-ops by default:

- `_join_fp_player_features(date) -> DataFrame | None` — feeds `playerstats`.
- `_join_fp_team_features(date) -> (team, defense) | (None, None)` — feeds
  `teamProfile` / `defenseProfile`.

`StatsNFL` overrides them for week-keyed FP. `StatsNBA` (CTG) and `StatsMLB`
(savant) override them for date-keyed sources through the shared read machinery in
`stats/collector_snapshots.py`:

- `DatedSnapshotStore(base_dir, label)` — lists capture dates/tools on disk and
  loads a tool's parquet as-of a game date (strict `<`).
- `load_asof_features(store, season, game_date, key_col, meta_cols)` — reads every
  tool in the latest pre-game snapshot from one capture date, merges them
  horizontally on `key_col`, indexes by `remove_accents(key_col)` (player name)
  or a team-abbreviation key, dedupes traded players (row with the most non-null
  cells wins), and renames each non-meta column to raw `{col}_asof`. The
  base-class join then `add_prefix("Player ")`s the player frame.
- `Stats._collector_asof_features(...)` — the cached wrapper each hook calls,
  memoizing per `(grain_dir, season, date)`.

Returns `None` — degrading to gamelog-only features, unchanged — whenever no
snapshot predates the game **or the source's join schema (`key_col`/`meta_cols`)
is unset**. Those constants live next to each league's hook (`_CTG_*` in
`stats/nba.py`, `_SAVANT_*` in `stats/mlb.py`) and are empty until a real capture
pins the provider's column names; see the per-source docs. New `{col}_asof`
columns enter the candidate feature set automatically (the no-filter rewire
removed `Filtered`) and appear in models on the next `meditate` that rebuilds the
matrix cache — no forced retrain, no `feature_filter.json` edit.

## Ops wiring

1. **`pyproject.toml`** — add the `[tool.poetry.scripts]` entry
   (`ctg-fetch = "sportstradamus.collectors.cleaningtheglass.cli:ctg_fetch"`).
2. **`scripts/run_job.sh`** — add a `<name>-fetch)` case; the
   `HEALTHCHECK_URL_<NAME>_FETCH` var auto-derives.
3. **`scripts/sync_to_prod.sh`** — the four collector dirs are pushed by an
   **additive** pass (never `--delete`, so prod-only siblings survive). A dir
   absent on dev is skipped. Only the models pass mirrors with `--delete`.
4. **Cron.** CTG/savant are **dev-side**: run them on the dev box beside the
   manual weekly `meditate`, then `scripts/sync_to_prod.sh` uploads the
   snapshots. Optionally schedule them via `run_job.sh` with a
   `HEALTHCHECK_URL_*` set.

## Recipe: add a new source

1. **Register the catalog.** Ship `data/config/<source>_endpoints.json` as `[]`.
   Populate it from a real DevTools capture (authed sources) or pinned public
   URLs — never invent columns. Name each spec `player_<tool>` or `team_<tool>`
   so `dated_path_for` routes it.
2. **Write the source package** `collectors/<source>/source.py`: a `Source` with
   `make_client`, `default_context`, `dispatch`, `path_for`
   (`partial(dated_path_for, league=…, source_dir=…)` for a date source),
   `transform=parse_tabular_response`, and either `auth_fields`/`env_prefix` (a
   cookie source, reusing `CookieClient`) or `auth_fields=None` (public, reusing
   `Scrape`). Then `cli.py`: `<name>_fetch = build_source_cli(SOURCE)`.
3. **Wire consumption.** Add the join-schema constants and override the two hooks
   on the league's `Stats` subclass, delegating to `_collector_asof_features`
   (see `stats/nba.py`). Leave the schema empty until you capture real columns.
4. **Ops + docs.** pyproject entry, `run_job.sh` case, `sync_to_prod.sh` dir,
   a `docs/<source>.md` cross-referencing this file, and a `<name>-fetch` case in
   `tests/golden/test_cli_help.py` (regenerate with `REGENERATE_SNAPSHOTS=1`).
5. **Probe before bulk.** `dry-run` one date, then a one-tool live fetch to
   confirm auth + parse before walking the whole catalog.
