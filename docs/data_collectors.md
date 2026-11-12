# Data collectors

A **collector** snapshots an external data provider's tables to parquet on a
schedule, and the stats layer folds those snapshots into player/team features at
prediction and training time. Three sources run on this framework today:

| Source | CLI | League | Auth | Period key |
|---|---|---|---|---|
| Fantasy Points Data Suite | `fetch fp` | NFL | cookie | game **week** |
| Cleaning the Glass | `fetch ctg` | NBA | session cookie | capture **date** |
| Baseball Savant (Statcast) | `fetch savant` | MLB | none (public) | capture **date** |

Everything shared lives in `src/sportstradamus/collectors/`; each source is a
thin package under it. This document is the canonical description of the
framework and the recipe for adding a fourth source. The per-source specifics —
where to get the cookie, which endpoints to register — live in
[fantasypoints.md](fantasypoints.md), [cleaningtheglass.md](cleaningtheglass.md),
and [baseballsavant.md](baseballsavant.md). Beside the framework sit a plain
command, `fetch headshots`, and the forward captures each `prophecize` run writes
of what the polls already see for the T2a / T2b reads (their own section below).

## Architecture

```
src/sportstradamus/collectors/
  _options.py        shared click options + catalog-load helper for the command builders
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
  fantasypoints/     FP_SOURCE + FP-only import-curl / login / season, column translation
  cleaningtheglass/  CTG_SOURCE (cookie, date-keyed)
  baseballsavant/    SAVANT_SOURCE (public Scrape, date-keyed)
  headshots.py       `fetch headshots` player-headshot cache (a plain command, not a Source)
  underdog_capture.py  per-poll Underdog line states + pill entry counts, written from `get_ud(capture_dir=…)`
  lineup_capture.py  per-run MLB / NHL starters from the live leagues' `Stats` + NBA daily lineups, written from `prophecize`
```

A source is a `Source` dataclass (`cli.py`) — **data and callables, not a
subclass hierarchy**. Its fields:

| Field | Meaning |
|---|---|
| `name`, `help` | log and progress label; group help text (the command name is the source's key in the `fetch` group in `src/sportstradamus/cli.py`) |
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
| `after_write(spec, df, season)` | date sources only: runs once a non-empty table's parquet has landed (never on a skip; an error ends the run). CTG refreshes the NBA roster seed with it |
| `has_backfill`, `backfill_end_week`, `modes`, `default_mode`, `render_request_body`, `verify_fn`, `renew_auth` | FP-only extras (default off) |

`build_source_cli(source)` returns a real `click.Group` bound at module level
(e.g. `ctg_fetch = build_source_cli(CTG_SOURCE)`). It attaches `run`, `verify`,
`list`, plus `refresh-auth` when the source is authed and `backfill` when it opts
in. `period_kind` decides whether `run`/`verify` come from `commands.py` (week,
with `--week`/`--mode`) or `commands_dated.py` (date, with `--date`).

## The CLI surface

Every source exposes:

```bash
sportstradamus fetch <fp|ctg|savant> list                        # print the registered catalog
sportstradamus fetch <fp|ctg|savant> run [--dry-run] [--only …]  # fetch every endpoint, write parquets
sportstradamus fetch <fp|ctg|savant> verify …                    # confirm the parquets landed with rows
sportstradamus fetch <fp|ctg> refresh-auth <curl>                # (authed sources) rotate the token/cookie
```

`run` is idempotent: it skips a cell whose parquet already exists non-empty, so
re-running after a partial failure only fetches the gaps. `--refetch` forces a
re-download; zero-row parquets are always retried. `--dry-run` lists what would
be fetched, touching nothing — the capture-stub-drift guard: dry-run one date
before any real (paid or rate-limited) run to confirm auth and the catalog.

**Period kinds.** Week-keyed sources (`fetch fp`) take `--week`/`--season` and a
`--mode`; each snapshot is one game week. Date-keyed sources (`fetch ctg`,
`fetch savant`) take `--season`/`--date` and snapshot the provider's
already-cumulative season-to-date table, stamped by the capture date (default:
today). Date sources do not carry `--week`, `--mode`, or `backfill` — the
provider only serves "as of now".

## Auth

`AuthFields` names up to three keys.json slots — `authorization`, `cookie`,
`user_agent` — and a `None` slot means the source doesn't use it. FP and CTG
both use `cookie` + `user_agent` only; savant sets `auth_fields=None` (public —
no keys.json entry, no `refresh-auth` command). No shipped source currently
uses the bearer `authorization` slot.

A source whose credential can be minted from a stored login sets
`Source.renew_auth`, a no-argument callable returning a fresh cookie. The
401 path tries it before prompting for a pasted curl, so those sources
recover under cron. FP is the one that does (its cookie lapses weekly);
see [fantasypoints.md](fantasypoints.md).

When neither path yields a credential the run **stops** — `_refresh_auth`
raises `CollectorAuthRecoveryError`, the loop writes its report and exits
non-zero naming the reason. The credential is shared by every endpoint, so
continuing cannot succeed; it would only add an identical failure per spec
and, for a source that renews from a stored login, one more refused sign-in
per spec.

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
`stats/nba.py`, `_SAVANT_*` in `stats/mlb.py`) and are empty until the source's
tables are wired into the features; see the per-source docs. New `{col}_asof`
columns enter the candidate feature set automatically (the no-filter rewire
removed `Filtered`) and appear in models on the next `meditate` that rebuilds the
matrix cache — no forced retrain, no `feature_filter.json` edit.

## Ops wiring

1. **`src/sportstradamus/cli.py`** — add the source to the `fetch` group's
   `lazy_subcommands` (`"ctg": "sportstradamus.collectors.cleaningtheglass.cli:ctg_fetch"`).
2. **`scripts/run_job.sh`** — add a `<name>-fetch)` case; the
   `HEALTHCHECK_URL_<NAME>_FETCH` var auto-derives.
3. **`scripts/sync_to_prod.sh`** — `COLLECTOR_RELS` lists the six collector dirs
   (each source's `player_data/` and `team_data/` dir), pushed by an **additive**
   pass (never `--delete`, so prod-only siblings survive). A dir absent on dev is
   skipped. Only the models pass mirrors with `--delete`.
4. **Cron.** FP and CTG run on prod from `run_job.sh` with a `HEALTHCHECK_URL_*`
   set (schedule: [OPERATIONS.md](OPERATIONS.md)), and `sync_from_prod.sh` brings
   their snapshots to the dev box. Savant is **dev-side**: run it on the dev box
   beside the manual weekly `meditate`, then a full `scripts/sync_to_prod.sh`
   uploads the snapshots (`--league` sends none).

## Forward capture collectors

Four captures store, at decision time, what the polls already see and the parse
drops: the inputs of the two pre-registered reads in
[handoffs/model_improvement_track.md](handoffs/model_improvement_track.md) §6.10,
**T2a**, the news guard (a line's suspension or a teammate's move to Out /
Doubtful before the decision), and **T2b**, popularity (the lobby's
`entry_count`). The bars, sample sizes and read dates live there and are not
repeated here. Nothing in these files changes a served number: no `Stats` hook
reads them, the dashboard never does, and only `prophecize` writes them. None is
a `Source`: the Underdog captures are a seam inside `get_ud`, the rest one call in
`prophecize` (`capture_lineups`) right after the leagues update.

| Capture | Trigger | Files under `data/runtime/` | Schema (pinned in the module) |
|---|---|---|---|
| Underdog per poll | every `prophecize` run, `get_ud(capture_dir=UNDERDOG_CAPTURE_DIR)`; no extra request | `underdog_capture/lines/date=<UTC date>/HHMMSS.parquet`, `underdog_capture/matches/…` | `collectors/underdog_capture.py` |
| Underdog pill entry counts | every `prophecize` run, the same capture, last in `get_ud`; ~70 requests | `underdog_capture/entry_counts/date=<UTC date>/HHMMSS.parquet`; `underdog_capture/lineup_statuses.json` | `collectors/underdog_capture.py` |
| NBA daily lineups | every `prophecize` run with the NBA live, `capture_lineups`; one request, only while an NBA event in `upcoming_events.json` commences within 2 h | `lineups/NBA/date=<UTC date>/HHMMSS.parquet` | `collectors/lineup_capture.py` |
| MLB / NHL starters | every `prophecize` run, `capture_lineups` right after the leagues update; no extra request | `lineups/<league>/date=<UTC date>/HHMMSS.parquet` | `collectors/lineup_capture.py` |

**Underdog per poll** (`collectors/underdog_capture.py`). Every line of the feed
and of each modeled league's lobby read, the suspended and live ones the parser
drops included: the line's `status`, the appearance's `lineup_status_id`
(Underdog's own Q / D / O flag; decode it with `lineup_statuses.json`), both
options' status and multiplier, and `over_under_id`, which survives a line move
so two polls diff on it. `matches/` holds each lobby's per-match `entry_count`, a
display string (`"75.2K"`). `source` names the payload (`feed`, `lobby:NBA`).
The ledger's `get_ud()` passes no `capture_dir`: it writes nothing and reads no
pill. Every poll's file carries the one schema the module's dtype table pins, so a
read over a date partition concatenates the files as they are. A feed that trips
the even-pick check ([underdog_api.md](underdog_api.md) §7.4) raises before the
capture, so that run stores nothing.

**Underdog pill entry counts** (`collectors/underdog_capture.py`, the reads in
`books/underdog.py`). The feed carries no per-line popularity; the lobby's
`lines?filter_type=PickemStat` pill does ([underdog_api.md](underdog_api.md) §2,
§6.4). Each poll reads one pill per `(sport, pickem_stat_id)` the feed shows for a
modeled league, about seventy requests under a time budget that a run of empty
answers ends early ([underdog_api.md](underdog_api.md) §2), then refreshes
`lineup_statuses.json`. The reads come last in `get_ud`, after the alternate-line
pass, so they can neither delay the board's own requests nor get them blocked, and
the line states are on disk before they start. A pill that answers empty is named in one warning and the
others still land; nothing in the capture fails the run or costs the board its
offers.

**NBA daily lineups** (`collectors/lineup_capture.py`). The NBA update names no
starters, so the run reads `stats.nba.com`'s `00_daily_lineups_<YYYYMMDD>.json`
for the Eastern date, flattened to one row per listed player on both teams
(`lineup_status`, `roster_status`, `position`). The JSON's own `timestamp` is
unreliable (the probes found stale values), so each row carries the run's
`Scored At` in `polled_at` and keeps the source's in `source_timestamp`. Lineups
post about half an hour before tip, so a run outside the 2 h window makes no
request. It runs only with the NBA live, and a failed read costs neither another
league's capture nor the board.

**MLB / NHL starters** (`collectors/lineup_capture.py`). The update `prophecize`
runs on each live league already reads each team's next game: MLB's probable
pitchers and the batting orders posted so far (`StatsMLB.upcoming_games`), NHL's
predicted starting goalie from dobbersports (`StatsNHL.upcoming_games`). The next
update overwrites them, so each run stores them, one row per team, role and
player (`pitcher`, `batter` with its batting `slot`, `goalie`) beside the
opponent and the home flag. A starter not yet named keeps his row with `player`
empty; an order not yet posted has no `batter` rows. `team` is the
`upcoming_games` key, so an MLB doubleheader's second game reads `NYY2`. A league
out of season, whose update failed or with no game ahead writes nothing; one
league's failed capture is logged and costs neither the others nor the board.

**The decision-time join.** A leg's decision time is its `Scored At`; join each
capture to the latest one at or before it (`polled_at`). This is `<=`, not the strict `<` of the dated snapshots above:
a capture is the state at a moment, never a table that may already fold the game
in. Every capture a `prophecize` run writes carries that run's `Scored At` in that
column, and every file is named by it, so `<=` picks the leg's own
run. Underdog files written before 2026-11-11 carry their own clock instead,
minutes after their run's `Scored At` (their `polled_at` matches no history
stamp): a reader assigns each to the run whose `Scored At` is the latest before it.

**Ops.** No capture has a cron line or `fetch` command of its own: all ride the
`prophecize` job ([OPERATIONS.md](OPERATIONS.md)). A capture that fails is an ERROR with its traceback in the
run's log; it never fails the run, so no healthcheck sees it.
`scripts/sync_from_prod.sh` pulls `data/runtime` whole, so the captures reach the
dev box with the rest of it; `sync_to_prod.sh` never sends them.

## Recipe: add a new source

1. **Register the catalog.** Ship `data/config/<source>_endpoints.json` as `[]`.
   Populate it from a real DevTools capture (authed sources) or pinned public
   URLs — never invent columns. Name each spec `player_<tool>` or `team_<tool>`
   so `dated_path_for` routes it.
2. **Write the source package** `collectors/<source>/source.py`: a `Source` with
   `make_client`, `default_context`, `dispatch`, `path_for`
   (`partial(dated_path_for, league=…, source_dir=…)` for a date source),
   `transform=parse_tabular_response` (CTG, whose data is JSON inside an HTML page,
   brings its own parser), and either `auth_fields`/`env_prefix` (a
   cookie source, reusing `CookieClient`) or `auth_fields=None` (public, reusing
   `Scrape`). Then `cli.py`: `<name>_fetch = build_source_cli(SOURCE)`.
3. **Wire consumption.** Add the join-schema constants and override the two hooks
   on the league's `Stats` subclass, delegating to `_collector_asof_features`
   (see `stats/nba.py`). Leave the schema empty until you capture real columns.
4. **Ops + docs.** `cli.py` `fetch` entry, `run_job.sh` case, `sync_to_prod.sh`
   dir and a `docs/<source>.md` cross-referencing this file, then regenerate the
   help snapshots (`REGENERATE_SNAPSHOTS=1 poetry run pytest
   tests/golden/test_cli_help.py`); the test walks the whole command tree, so
   there is no case to add.
5. **Probe before bulk.** `dry-run` one date, then a one-tool live fetch to
   confirm auth + parse before walking the whole catalog.
