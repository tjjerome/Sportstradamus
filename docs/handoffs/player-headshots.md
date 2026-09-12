# Player Headshots

> Status: ACTIVE — stage 0 (owner clears the CDNs) ∥ stage 1 not started (briefed 2026-09-12)

## 1. Mission & money logic

Put a face on every player star. The constellation ticket card ships an initials
disc titled "Player headshot — coming soon"
(`dashboard/components/constellation_component/build/main.js`, `.cst-shot`) — the
one visible scar left on the Games surface. No money logic; the dashboard is the
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
grep -n "coming soon" src/sportstradamus/dashboard/components/constellation_component/build/main.js   # scar present = stage 3 open
ls src/sportstradamus/data/assets/headshots/ 2>/dev/null | head                                        # cache built?
poetry run python - <<'EOF'
import pandas as pd
for lg, col in [("mlb","playerId"),("nba","PLAYER_ID"),("wnba","PLAYER_ID"),("nhl","playerId"),("nfl","player id")]:
    g = pd.read_parquet(f"src/sportstradamus/data/leagues/{lg}/gamelog.parquet", columns=[col])
    print(lg, g[col].nunique())
EOF
curl -sI https://cdn.nba.com/headshots/nba/latest/1040x760/2544.png | head -1                         # pattern still live?
```

CDN patterns, each answered `200` to a HEAD from the dev box on 2026-09-12:

| League | Id column | Pattern |
|---|---|---|
| MLB | `playerId` (StatsAPI person id) | `https://img.mlbstatic.com/mlb-photos/image/upload/w_213,q_100/v1/people/{id}/headshot/67/current` (JPEG) |
| NBA | `PLAYER_ID` | `https://cdn.nba.com/headshots/nba/latest/1040x760/{id}.png` |
| WNBA | `PLAYER_ID` | `https://cdn.wnba.com/headshots/wnba/latest/1040x760/{id}.png` |
| NHL | `playerId` | `https://assets.nhle.com/mugs/nhl/{season}/{TEAM}/{id}.png` — season-stamped and team-stamped; `https://api-web.nhle.com/v1/player/{id}/landing` returns the current `headshot` URL |
| NFL | `player id` = `gsis_id` | `nflreadpy.load_rosters([season])` column `headshot_url` (2025: 3,137 rows, 2.9 % null) |

### Volatile product assumptions

- CDN terms of use — the owner clears each league before the job goes on the
  production cron (stage 0); nothing in code checks it.
- The patterns are unversioned and can move; the NHL path changes every season
  and on every trade (the `landing` endpoint is the stable lookup).
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

## 5. Module footprint & canonical paths

| Module | Role |
|---|---|
| `data/assets/headshots/{league}/{id}.webp` + `data/assets/headshots/index.parquet` (NEW, gitignored) | the cache and its index: `league, id, name, team, file, fetched_at, status` |
| `collectors/headshots.py` (NEW) + `cli.py` `fetch headshots` | enumerate ids, download misses, crop/resize, write the index; tqdm bar |
| `helpers/scraping.py` | `Scrape.get_bytes` (NEW, small) — reuses the header rotation and retry loop |
| `dashboard/assets.py` | `headshot_uri(league, player, team)` → data URI or `None`, mtime-cached like `_data_uri` |
| `dashboard/components/slip_builder.py` / `constellation_component/__init__.py` / `build/main.js` / `build/index.html` | the `shots` side channel and the `<img>` render in the card |
| `scripts/run_job.sh`, `docs/OPERATIONS.md`, `.gitignore` | the monthly job |
| `tests/golden/test_headshots.py` (NEW); `test_constellation_component.py` stays green | index schema, resolver miss → `None`, payload cap |

The collector never opens the archive (no DuckDB); `run_job.sh` takes the archive
flock for every job regardless, so schedule it off `reflect`'s window like `gate-status`.

## 6. Stage plan

0. **Owner clears the CDNs** (owner, minutes; gates the prod cron, not the build):
   yes / no per league for the five patterns in §3. A "no" league keeps the
   initials disc and is skipped by a league list in the collector.
1. **Fetcher + cache** (1–2 sessions). `sportstradamus fetch headshots
   [--league …] [--force]`: enumerate ids per league from the gamelog parquet
   (NFL: union with `load_rosters` so rookies land pre-season; NHL: build the
   season/team URL from the gamelog, fall back to `landing`); download misses
   through `Scrape.get_bytes` with the usual jittered sleep; centre-crop square,
   resize to 128 px (renders at 34 px; 2× phone DPR needs 68), WebP; write the
   index with `status` so misses retry next month. Idempotent — a second run
   downloads nothing. Acceptance: index rows ≥ 95 % of gamelog ids per league
   (counts in the ledger); second run 0 downloads.
2. **Monthly job** (½ session). `run_job.sh` case `headshots`, OPERATIONS.md cron
   row + table row (`0 5 1 * *`, after `gate-status`), `HEALTHCHECK_URL_HEADSHOTS`.
   Acceptance: a dry run on the dev box adds only new ids; the healthcheck ping fires.
3. **Render** (1 session). `slip_builder` builds `shots = {key: uri}` for the
   stars on the figure — league and team are known server-side, the name
   matches the index through `remove_accents`, team breaks a name tie, no match
   → no entry. `_component(..., shots=shots)`; `main.js` draws `<img
   class="cst-shot">` (`object-fit: cover; border-radius: 50%`) when the key has
   a shot, initials otherwise. Payload: ~40 stars × ~4 KB. Live-verify with
   playwright on desktop and phone: zero page errors, iframe height posts on each
   lens toggle, one star with a file shows the face, one without shows initials.
   Acceptance: live verdict in the ledger; goldens green.

Only the constellation card carries the scar (the art-catalog sweep found no
other person-icon stand-in); stop there unless the owner asks for more surfaces.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > home-of-record
  doc > this brief > roadmap v3.
- One module per subagent; no HEAD-moving git in subagent prompts; `git add` by
  explicit path (never the cache directory — it is gitignored anyway).
- The fetcher never runs inside the integration suite (network); goldens use a
  two-file fixture cache.
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

- 2026-09-12 · stage 0 · brief written; five CDN patterns verified `200` from the dev box (§3), NFL via nflverse `headshot_url`; render path chosen = `shots` side channel, not customdata · next: owner clears CDNs ∥ stage 1 fetcher
