# Player Headshots

> Status: COMPLETE — all four stages closed; the monthly cron row is the owner's to install

## 1. Mission & money logic

Put a face on every player star. The constellation ticket card drew an initials
disc titled "Player headshot — coming soon" — the last visible scar on the Games
surface. It now draws the face this box's cache holds and that disc when it holds
none (`build/main.js`, `shotHtml`). No money logic; the dashboard is the
product the owner reads nightly and a face is recognition at a glance across
forty stars on a phone. Scope, in the owner's words: find, download and cache
headshots for all players in all five leagues (MLB, NBA, WNBA, NHL, NFL), and a
pipeline that picks up new and rookie headshots as they appear, runnable monthly.

## 2. Read first (in order)

1. [`../art_assets.md`](../art_assets.md) — the `player_headshots` row this lane
   fills (disk-cache, never hot-link; ~34×34 circular render).
2. `src/sportstradamus/dashboard/assets.py` — `_data_uri` (mtime-keyed
   `lru_cache`, WebP downscale) is the embed precedent; Streamlit serves no
   arbitrary static files, so images travel as data URIs.
3. `dashboard/components/constellation_component/__init__.py` — the
   `_component(figure_json=…, sparks=…, moves=…)` side-channel dicts keyed by
   star key; `build/main.js` reads them next to the figure and draws the card
   (`initials()`, the `.cst-shot` disc); `build/index.html` holds the disc CSS.
4. `dashboard/components/constellation_traces.py` — `_card_fields` / `node_info`:
   what each star knows (key, player, team, league resolved server-side in
   `constellation.py`). Adding a customdata field renumbers every positional
   read in `main.js` — use the side channel instead.
5. `helpers/io.py` `_gamelog_paths` / `read_gamelog` — the gamelog parquet is the
   one enumeration that carries id + name + team for every league; the
   `players.json` sidecars lack ids for NBA/WNBA/NFL.
6. `helpers/text.py` `remove_accents` — the name normalization every league
   applies to gamelog names; the offers frame uses the same names.
7. `helpers/scraping.py` `Scrape` — header rotation + retries; `get` returns
   parsed JSON only, so a bytes path is new.
8. `scripts/run_job.sh` (the `case "$JOB"` map) + [`../OPERATIONS.md`](../OPERATIONS.md)
   cron table + `cli.py` (`LazyGroup`, the `fetch` group) — job plumbing.
9. [`../dashboard_ux_redesign.md`](../dashboard_ux_redesign.md) §6 (player assets)
   and §8 item 3 — the spec passages this lane closes.

## 3. Verify before you trust

```bash
git fetch origin && git log --oneline origin/devel -3
grep -n "shotHtml" src/sportstradamus/dashboard/components/constellation_component/build/main.js      # the render
ls src/sportstradamus/data/assets/headshots/ 2>/dev/null | head                                        # this box's cache
poetry run python - <<'EOF'
import pandas as pd
for lg, col in [("mlb","playerId"),("nba","PLAYER_ID"),("wnba","PLAYER_ID"),("nhl","playerId"),("nfl","player id")]:
    g = pd.read_parquet(f"src/sportstradamus/data/leagues/{lg}/gamelog.parquet", columns=[col])
    print(lg, g[col].nunique())
EOF
curl -sI https://cdn.nba.com/headshots/nba/latest/1040x760/2544.png | head -1                         # pattern still live?
```

CDN patterns, as `collectors/headshots.py` sends them. A `200` is not proof of a face:
NBA and WNBA answer an unknown id with a flat grey silhouette and NHL redirects to one, so
every decoded image is colour-count probed before it reaches the cache.

| League | Id column | Pattern |
|---|---|---|
| MLB | `playerId` (StatsAPI person id) | `https://img.mlbstatic.com/mlb-photos/image/upload/w_256,c_fill,ar_1:1,g_face/v1/people/{id}/headshot/67/current` — Cloudinary, so the face crop is a URL param. The only league that 404s honestly |
| NBA | `PLAYER_ID` | `https://cdn.nba.com/headshots/nba/latest/1040x760/{id}.png` |
| WNBA | `PLAYER_ID` | `https://cdn.wnba.com/headshots/wnba/latest/1040x760/{id}.png` |
| NHL | `playerId` | `https://assets.nhle.com/mugs/nhl/latest/{id}.png` — answers for traded and retired players alike (40/40 gamelog ids), so neither the season/team path nor the `landing` endpoint is needed |
| NFL | `player id` = `gsis_id` | `nflreadpy.load_rosters(...)` column `headshot_url`, over the gamelog's **whole season span** — one season covers 62 % of its ids, the span 99.8 %. Also Cloudinary: splicing a face crop into the URL cuts a transfer from 3.8 MB to ~21 KB |

### Volatile product assumptions

- CDN terms of use — the owner clears each league before the job goes on the
  production cron (stage 0); nothing in code checks it.
- The patterns are unversioned and can move. A league that starts answering a
  placeholder in a new shape would slip past the colour probe — the monthly run's
  `missing=` count is the tell.
- The NFL roster release carries rookies before they have a gamelog row; the
  other leagues' rookies appear only once they play.

## 4. Locked decisions

- 2026-09-12 — **Disk-cache, never hot-link** ([`../art_assets.md`](../art_assets.md)).
  The dashboard reads the local cache only; a missing file renders the initials
  disc exactly as today.
- 2026-09-12 — **All five leagues, every player in the gamelog, refreshed
  monthly** (owner). A refresh downloads new ids and retries recorded misses;
  it never re-downloads an existing file without `--force`.
- 2026-09-12 — **Cache is gitignored, per box.** Roughly 4,500 files at 128 px
  WebP (~20 MB) have no place in git; the dev box and the prod box each run the
  fetcher (the same rule as `stat_calibration.json`: runtime-produced, not committed).
- 2026-09-12 — **Team marks stay skipped** (owner) — this lane is faces only.
- 2026-09-12 — **Licence checking is the owner's, no code gate** — the same rule
  the ambient art follows.
- 2026-09-13 — **All five leagues cleared for this deployment** (owner). The box is
  private and single-user, no commercial licence is being sought, and the cache is
  disk-held, gitignored and never hot-linked or redistributed. Exposing the dashboard
  publicly would reopen the question; the fallback makes that a cache deletion, not a
  code change.

## 5. Module footprint & canonical paths

| Module | Role |
|---|---|
| `data/assets/headshots/{league}/{id}.webp` + `data/assets/headshots/index.parquet` (NEW, gitignored) | the cache and its index: `league, id, name, team, file, fetched_at, status` |
| `collectors/headshots.py` (NEW) + `cli.py` `fetch headshots` | enumerate ids, download misses, crop/resize, write the index; tqdm bar |
| `helpers/scraping.py` | `Scrape.get_bytes` (NEW, small) — reuses the header rotation and retry loop |
| `dashboard/assets.py` | `headshot_uris(pool)` → `{player: data URI}` for the stars on one figure, mtime-cached like `_data_uri` |
| `dashboard/components/slip_builder.py` / `constellation_component/__init__.py` / `build/main.js` / `build/index.html` | the `shots` side channel and the `<img>` render in the card |
| `scripts/run_job.sh`, `docs/OPERATIONS.md`, `.gitignore` | the monthly job |
| `tests/golden/test_headshots.py`; `test_constellation_component.py` stays green | crop + placeholder probe, index round-trip, resolver miss → no entry |

The collector never opens the archive (no DuckDB); `run_job.sh` takes the archive
flock for every job regardless, so schedule it off `reflect`'s window like `gate-status`.

## 6. Stage plan

0. **Owner clears the CDNs** — CLOSED: all five cleared for this private deployment
   (§4). A league the owner later withdraws is dropped with `--league` and keeps the
   initials disc.
1. **Fetcher + cache** — BUILT. `sportstradamus fetch headshots [--league …] [--force]`
   enumerates ids per league from the gamelog parquet (NFL unions `load_rosters` so
   rookies land pre-season), downloads through `Scrape.get_bytes`, crops square to the
   alpha bounding box, resizes to 128 px WebP, and writes the index with `status` so
   misses retry next month. No per-request sleep: these are public CDNs serving browser
   traffic and a 1–3 s pause makes the cold run four hours; a `403`/`429` raises
   `BlockedError` instead of knocking again. Cold runs: WNBA 337 ok / 4 missing (all four
   the CDN's own silhouette), MLB 1631/1631. A second run downloads nothing.
2. **Monthly job** — BUILT. `run_job.sh` case `headshots`, OPERATIONS.md cron row + table
   row (`0 5 1 * *`, after `gate-status`), `HEALTHCHECK_URL_HEADSHOTS`. Installing the
   crontab row is the owner's action, gated on stage 0.
3. **Render** — BUILT. `slip_builder` passes `shots = {player: uri}` for the stars on the
   figure; `assets.headshot_uris` reads each row's own league (the wider lens mixes them),
   breaks a name tie on team, and resolves a still-ambiguous name to nothing. `main.js`
   `shotHtml` draws `<img class="cst-shot">` when the player has a URI and the initials
   disc when not. Live playwright verdict, desktop 1600×1000 and phone 390×844, zero page
   errors both: a cached player renders the `<img>` at 34×34 from a WebP data URI, and the
   same stars against an empty cache render the initials disc at an identical iframe height.

Only the constellation card carries the scar (the art-catalog sweep found no
other person-icon stand-in); stop there unless the owner asks for more surfaces.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > home-of-record
  doc > this brief > roadmap v3.
- One module per subagent; no HEAD-moving git in subagent prompts; `git add` by
  explicit path (never the cache directory — it is gitignored anyway).
- The fetcher never runs inside the integration suite (network); goldens build
  every image in memory and commit no binary for a gitignored cache.
- `main.js` has no test runner: every edit gets the playwright verdict
  (pageerror + iframe height on lens toggles).
- DESIGN.md tokens only for the disc ring/border; no new hex.

## 8. Escalation & stop conditions

**Stop and ask the owner:** before the prod cron row lands (stage 0 answers);
a CDN answering 403/429 — back off, do not route images through the
ScrapingFish fallback; a figure payload past ~1 MB (then the fix is
`server.enableStaticServing` in `.streamlit/config.toml`, a deployment change).

**Park and pivot:** stage 3 can land on the dev cache alone; stage 2 waits on
stage 0.

**Dispatch:** `refactoring-specialist` on every touched `.py`.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session.
- `poetry run ruff check src/sportstradamus/` clean; CI also runs `ruff format`.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- Stage 3: playwright verdict recorded (desktop + phone).
- One ledger line appended below; status line updated on stage boundaries;
  `docs/art_assets.md` row and spec §6/§8 trued when the scar closes.
- Never push `devel`.

## 10. Ledger (append-only, newest first, cap ~15)

- 2026-09-13 · stage 0 · owner cleared all five leagues for this private box, no commercial licence sought (§4) · lane COMPLETE
- 2026-09-13 · stages 1-3 · `fetch headshots` + monthly job + card render built and live-verified (desktop + phone, both the face and the initials fallback); three §3 claims corrected in place
- 2026-09-12 · stage 0 · brief written; five CDN patterns verified `200` from the dev box (§3), NFL via nflverse `headshot_url`; render path chosen = `shots` side channel, not customdata · next: owner clears CDNs ∥ stage 1 fetcher
