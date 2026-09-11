# Underdog Fantasy API reference

How the Underdog web app talks to its backend today, what the payloads carry, and how
to pull the most lines per request. The scraper
([books/underdog.py](../src/sportstradamus/books/underdog.py)) is built on it; §2 is
its request plan. The board-capture ToS caution stays canonical in
[underdog_edge_suite.md §2.2](underdog_edge_suite.md#22-underdog-line-capture); nothing
here changes it.

**Sources.** One Firefox HAR capture of a logged-in browse session
(`new_ud_api/lines.json`, 2026-09-10 18:38 CDT, `Client-Version` `20260907143253`,
135 entries, 59 unique URLs, 36 distinct Underdog endpoints), the repo, the web app's
JavaScript bundles, and 248 live probes run the same evening from a plain `requests`
client, 115 of them with tokens the owner pasted (§11). The HAR holds the
account's access token and personal data, so it is gitignored and never quoted here
(§12).

**Tag legend.** Every endpoint row and every claim carries one:

| Tag | Meaning |
|---|---|
| **[HAR]** | Seen in the 2026-09-10 capture |
| **[CODE]** | From the repo (file:line cited) |
| **[DISC]** | URL discovered inside a scaffold `data_source.url`, not requested by the browser |
| **[LEGACY]** | Pre-lobby endpoint the scraper uses |
| **[P-n]** | Settled (or still open) by live probe n; §11 has the log |
| **[INF]** | Inference; treat as a hypothesis |
| **[OWNER]** | Product fact stated by the account owner, not observable from the API |

**What this is not.** Not a description of boosted payloads: the capture is pre-game
and promo-free (every one of its 307 lines is `line_type: balanced`,
`live_event: false`, `boost: null`). Live lines were reached only by probes P11 and
P21, alternate rungs only by probes P12, P19, P23, P39 and P40 (§6.2, §9.4). Recapture
instructions are in §12.

## 1. Why this document

Underdog retired `beta/v5` and `beta/v6/over_under_lines` around 2026-09-08. The
retirement signature is HTTP 426 with body
`{"error": {"api_code": "upgrade_required", "detail": "A new version is required to continue", ...}}`
**[P-2]**. `Scrape.get` treats any non-200 as a retry, logs the status at DEBUG and
returns `{}` after three attempts
([scraping.py:121-128](../src/sportstradamus/helpers/scraping.py#L121-L128)), so the
Underdog leg of `prophecize` died silently for two days. Commit `6a85f74e` moved the
scraper to `v1/over_under_lines`, which answers today.

The web app talks to a different surface: lobby *scaffolds* that point at *content*
endpoints, a search endpoint, reference data on a stats host, and Pusher channels for
line updates. Those payloads also carry data the scraper has never read: per-side
prices and implied probabilities, line status, live flags, provider ids for
prediction-market contracts, an alternates flag, a boost slot. Probe P1 established
that the legacy bulk feed carries the same line object, so most of that data is one
parser change away (§8, §10).

## 2. The scraper today

`get_ud` ([books/underdog.py](../src/sportstradamus/books/underdog.py)) makes only
unauthenticated GETs **[CODE]**:

| Call | When | Purpose |
|---|---|---|
| `GET api…/v1/over_under_lines` (`UD_LINES_URL`) | once | every player prop, all sports (§6.1) |
| `GET api…/v1/lobbies/content/match_grouped_lines?sport_id=<X>` + trio (`UD_LOBBY_URL`) | once per modeled league (`UD_MODELED_LEAGUES`) with games in the feed | moneyline, spread, total; its `teams` dict also names the feed's teams (§7.3) |
| `GET api…/v1/lobbies/content/lines?sport_id=NFL&filter_id=<pill>&limit=1000…` + trio | once per pinned pill (`UD_TEAM_PILLS`: Team Totals, TD Picks, 1Q–4Q / 1H / 2H Team Picks) | team totals, team TDs, period team lines (§7.6) |
| `GET api…/v3/over_unders/<over_under_id>/alternate_projections` (`UD_ALT_LINES_URL`) | one per modeled, `stat_map`-mapped player prop flagged `has_alternates`, three workers on one session, until `UD_ALT_LINES_BUDGET_S` (120 s) runs out or `UD_ALT_LINES_EMPTY_STREAK` (25) markets in a row answer without rungs; games about to lock and deep lines first | alternate lines (§9.4) |

The parser takes both container styles (§7.3), drops suspended and live lines, combo
players and appearances without a game (season futures), and prices a suspended
option side at 0. Team lines are keyed on the team and match lines on the home team,
with the API `stat` slug as the market (`moneyline`, `spread`, `points`,
`team_total_points`, `period_1_moneyline`, …) so nothing collides with the
sportsbook team markets in the archive; a null `stat_value` becomes a 0.5 line. The
frozen contract is
[tests/golden/test_books_get_ud.py](../tests/golden/test_books_get_ud.py). An empty
feed logs at WARNING and returns nothing: a 426 from a retired path is the signature
to watch (§12). `/v1/teams` and `beta/v3/rival_lines` are no longer requested.

The web app's own bundle still knows this feed: the `entry.app` chunk maps
`regular` → `/v1/over_under_lines` and `live` → `/beta/v2/live_over_under_lines`
**[P-8]**, so the legacy path is not orphaned. The browse session never called either;
they serve the pick-slip refresh, not the lobby.

## 3. Hosts and client identity

| Host | Role | Tag |
|---|---|---|
| `api.underdogfantasy.com` | lobby content, lines, search, account | [HAR] |
| `stats.underdogfantasy.com` | reference data: sports, teams, styles, statuses | [HAR] |
| `app.underdogsports.com` | the SPA; `GET /client-version` → `{"clientVersion": "20260907143253"}`, `cache-control: public, max-age=14400` | [HAR] |
| `login.underdogsports.com` | Auth0 tenant (§4) | [HAR] |

Third parties in the capture that a scraper never needs: Radar (geo token), Intercom,
Contentful (promo copy), Algolia (search-as-you-type suggestions) **[HAR]**.

**Headers the browser sends** on every `api.` request **[HAR]**: `Authorization`
(bare JWT, §4), `Client-Type: web`, `Client-Version: <clientVersion>`,
`Client-Device-Id: <uuid4, stable per browser>`, `Client-Request-Id: <uuid4 per call>`,
`User-Latitude`, `User-Longitude`, `User-Location-Token` (§4), and empty
`Referring-Link` / `User-Geo-Comply-License-Key`, plus
`Origin` / `Referer: https://app.underdogsports.com`.

**Headers the server accepts** (`access-control-allow-headers` on the CORS preflight)
**[HAR]**: `authorization, client-device-id, client-request-id, client-type,
client-version, referring-link, user-geo-comply-license-key, user-latitude,
user-location-token, user-longitude`.

None of them is required for public reads: probe P3 fetched lobby lines with no
headers beyond a desktop `User-Agent`, and adding the `Client-*` set changed nothing
but the CDN cache state **[P-3]**.

## 4. Authentication and geo

**Access token.** `Authorization` carries the access JWT *bare*, with no `Bearer`
scheme, on 58 of the 59 captured requests; the one Bearer-scheme header in the capture
is Contentful's **[HAR]**. Claims (no secrets) **[HAR]**:

| Claim | Value |
|---|---|
| `iss` | `https://login.underdogsports.com/` |
| `aud` | `https://api.underdogfantasy.com` |
| `azp` (client id) | `cQvYz1T2BAFbix4dYR37dyD9O0Thf1s6` |
| `gty` | `password`, then `refresh_token` |
| `scope` | `offline_access` |
| `exp − iat` | 600 s |

So the login is an Auth0 Resource Owner Password grant, `POST
login.underdogsports.com/oauth/token` with a 259-byte JSON body (the HAR entry has
status 0 and no body on either side, so nothing beyond the size is captured),
renewed with `grant_type=refresh_token` at most every ten minutes **[HAR]**. The body
shape is the standard Auth0 one **[INF]**: `grant_type`, `username`, `password`,
`client_id`, `audience`, `scope`. The `Authorization` response header on `GET /v1/user` is the
Bearer scheme plus the *same* token: an echo, not a rotation **[HAR]**.

**Location token.** `User-Location-Token` is a Radar JWT (`api-verified.radar.io`),
lifetime 1200 s, with `passed`, `failureReasons`, `warningReasons`, `events` and
`user` claims; six requests sent it empty and still got 200 **[HAR]**.

**Who needs a token.** The browser sent one on every request, so the capture cannot
tell public from private. The probes can:

| Surface | Without token | Tag |
|---|---|---|
| `v1/over_under_lines`, `beta/v3/rival_lines`, `beta/v2/live_over_under_lines` | 200 | [P-1] [P-11] |
| `lobbies/content/lines`, `match_grouped_lines`, `market_price_history`, `v2/pickem_search/search_results` | 200 | [P-3] [P-4] [P-5] [P-6] |
| `lobbies/scaffolds/sports`, `live_matches`, `futures`, `players` | 200 | [P-4] [P-7] |
| everything on `stats.underdogfantasy.com` | 200 | [P-4] [P-9] |
| `lobbies/content/pickem_lobby_sections` | 200 on three sections at 21:50 (an NFL game, ATP Tennis, MLB Games); the same NFL section had answered 401 `{"error": {"detail": "Invalid email or password.", "http_status_code": 401}}` at 19:27 with no token, so keep a token fallback for sections | [P-4] [P-29] [P-30] |
| `scaffolds/home`, `scaffolds/matches`, `match_details`, `sport_navigation`, `match_navigation`, `lines_with_stats`, `lines?data_driven_line_source_id=` (Popular Picks), `sport_promo_carousel`, `promo_carousels/main`, `alternate_projections`, `/v1/features`, `/v3/user/features` | 200 | [P-12] [P-25] [P-26] [P-27] [P-29] [P-30] |
| `lobbies/content/market_filters` (sport or game target) | **400** `Unable to construct the scaffold for market filter navigation.`; 200 with a token for both targets | [P-25] [P-33] [P-37] |
| `personalized/fantasy_lineup_cards`, `packs` | 401; 200 with a token | [P-29] [P-31] |
| `/v1/user`, `/v1/user/entry_slip_limits`, `/v1/user/power_ups`, `/v3/entry_slips/estimate` (also wants `UD-User-ID`) | answered with a token; not tried bare | [P-31] [P-32] |
| `/v1/user/state_configs` | 422 `location_needed` with and without a token until `User-Latitude` / `User-Longitude` are sent; 200 with the two headers plus a token, no Radar token needed | [P-28] [P-31] [P-36] |
| `/v1/authorizations/pusher_channels`, `/v1/intercom/*` | account endpoints; assume required | [INF] |

A stale token is worse than none: the origin answers 401 `Invalid email or password.`
to an expired JWT on any endpoint it validates (P24, `pickem_lobby_sections`), and P3c
got a 200 with the dead HAR token only because the CDN served it **[P-3] [P-24]**. Send
no token on public reads and a fresh one on the rest.

**Getting a token for the authed endpoints.** Log in at `app.underdogsports.com`,
open DevTools → Network → any `api.underdogfantasy.com` request → Request Headers, and
copy the `Authorization` value. Paste it into `src/sportstradamus/creds/keys.json` as
`underdog_authorization` (same naming as `fantasypoints_authorization`, see
[data_collectors.md](data_collectors.md#auth)). It dies after ten minutes, which makes
it useless for cron; an automated client would have to run the password grant itself,
which is the ToS-adjacent step §2.2 of the edge suite warns about, and this task did
not exercise it. One pasted token served up to 71 requests inside its ten minutes
(P39–P41); copy it right after a page load, since the app refreshes it silently and a
copied value can already be minutes old (the first paste had expired before it was
used, P24).

## 5. Request conventions

**Ubiquitous query params.** Every content and search request carries **[HAR]**:

| Param | Value | Origin |
|---|---|---|
| `product` | `fantasy` | fixed |
| `product_experience_id` | `b34dfd93-d0e8-4da3-8bf4-45c15c548dec` | `GET /v1/user` → `product_experience.id` ("Fantasy and Prediction Market (Kalshi, Nadex, and UDX)") |
| `state_config_id` | `16fa6ed3-ea21-4654-bcee-fb32d2f31357` | `GET /v1/user` → `state_config_id`; `GET /v1/user/state_configs` → `state: "TX"` |

`state_config_id` is mandatory on the `api.` host: without it the server answers 400
`{"error": {"detail": "The 'state_config_id' parameter is required.", "http_status_code": 400}}`
**[P-4]**. Whether `product` and `product_experience_id` are also mandatory was not
isolated (P4e dropped all three and the server complained about `state_config_id`
first). The `stats.` host ignores all three **[P-9]**. Because the state config gates
which stats and sports are visible (`pick_em.visible_stats` has 3,348 slugs,
`visible_sports` 131 **[HAR]**), a client should send the id of the state it actually
operates from.

**Scaffold params** **[HAR]**: `ct_token=w2`, `include_prediction_markets=true`,
`market_view=compact`, and `experiments_token`, which is the zlib-compressed,
URL-safe-base64 encoding of
`{"2026-american-odds": "variant-a", "2026-event-page-redesign": "variant-a"}` and is
also served as `user.lobby_experiments_token`. Scaffolds answer without
`experiments_token` **[P-4]**; the other three were always sent.

**Caching.** Quoted verbatim **[HAR]** and confirmed by probe **[P-9]**:

| Surface | `cache-control` |
|---|---|
| lobby content (`lines`, `match_grouped_lines`, `market_filters`, …) | `max-age=30, public, stale-while-revalidate=60, stale-if-error=600, s-maxage=30, no-store` |
| `search_results`, `beta/v2/live_over_under_lines` | same, with `stale-while-revalidate=30` |
| `market_price_history` | same, with `stale-while-revalidate=300` |
| `v1/over_under_lines` | `max-age=10, public, stale-while-revalidate=10, stale-if-error=600, s-maxage=10, no-store` |
| `beta/v3/rival_lines` | as above with 15 s |
| `scaffolds/futures`, `scaffolds/players`, `scaffolds/matches`, `match_details`, `lines?data_driven_line_source_id=` | `max-age=90, public, stale-while-revalidate=60, …` (`=5` on the last) |
| `sport_navigation`, `match_navigation`, `lines_with_stats`, `/v1/features` | `max-age=30, public, …` with `stale-while-revalidate=30` or `5` |
| `v3/over_unders/<id>/alternate_projections` | `max-age=10, public, stale-while-revalidate=10, …` |
| `v3/entry_slips/estimate` | `max-age=30, public, …` on a fantasy-only slip (the URL carries a unique `estimate_request_id`), `private` once a prediction-market leg is in |
| `scaffolds/home`, `scaffolds/sports`, `sport_promo_carousel`, `/v3/user/features`, `/v1/user*` | `private` (ETag; browser got 304s) |
| `scaffolds/live_matches`, `promo_carousels/main` | `private, no-store` |
| `stats.underdogfantasy.com/*` | `max-age=900, public, stale-while-revalidate=60, stale-if-error=600, s-maxage=900, no-store` |
| `app.underdogsports.com/client-version` | `public, max-age=14400` |

`cf-cache-status` reports `HIT` inside the window, so polling a URL faster than its
`max-age` re-reads the Cloudflare copy: **30 s is the floor for lobby content, 10 s for
the legacy feed.** Every 200 carries a weak ETag; a repeat GET with `If-None-Match`
returns 304 with an empty body **[P-9]**, which saves bytes but not requests. No
rate-limit headers appear anywhere; the only statuses seen were 200, 304, 400, 401,
404, 422 and 426.

**Sizes.** Firefox's `content.size` is the brotli wire size and `len(content.text)`
the decoded size; the ratio runs 5–12×. Probe sizes below are decoded bytes (the
client decompresses). Both are given where known.

## 6. Endpoint catalog

Columns: path (query minus the ubiquitous trio), auth without a token, size
wire/decoded, cache `max-age`, what comes back, tag. "list" and "dict" say whether
entity containers are arrays or id-keyed objects (§7).

### 6.1 Legacy bulk feed

| Endpoint | Auth | Size | TTL | Returns | Tag |
|---|---|---|---|---|---|
| `GET api…/v1/over_under_lines` | none | ?/32.7 MB | 10 s | `over_under_lines` (12,272 across 14 sports on 2026-09-10), `players` (2,335), `games` (128), `solo_games` (45), `appearances` (2,502), `opened_lines_count`; all lists; the full §7 line object; **one line per (appearance, stat), so no alternates**; player props only (no `core` / `team_prop` / `misc` categories and no live lines, §7.6); `?over_under_ids=<id>,<id>` narrows the response to the named markets with their players, games and appearances (1 id → 4.6 KB, 10 ids → 30 KB; the `over_under_ids[]=` array form is ignored and returns the whole board); 7.5 s for the first transfer, 0.5 s for the repeat, both CDN hits | [LEGACY] [P-1] [P-13] [P-14] [P-21] |
| `GET api…/beta/v2/live_over_under_lines` | none | ?/639 KB | 30 s | same shape; 270 lines, all `live_event: true` with `live_event_stat` (the running stat), MLB in-progress games at probe time | [P-11] |
| `GET api…/beta/v3/rival_lines` | none | 98 B | 15 s | `rival_lines`, `players`, `games`, `solo_games`, `appearances`: all empty, with and without a token. The Rivals product is retired and Ladders replaced it **[OWNER]**; the `rivals_enabled: true` flag on `/v2/sports` is stale | [LEGACY] [P-1] [P-31] |
| `GET api…/beta/v6/over_under_lines` | — | 222 B | — | 426 `upgrade_required` with and without current `Client-Type` / `Client-Version` headers: a retirement, not a version gate | [P-2] |
| `GET stats…/v1/teams` | none | 1.22 MB/11.3 MB | 900 s | `teams[]{id, abbr, name, short_name, sport_id, icon_url, colors, sport_radar_id, …}` for every sport | [LEGACY] [HAR] |

Lines per sport in the legacy feed on 2026-09-10 **[P-1]**: NFL 5,052 · CFB 3,110 ·
FIFA 1,986 · MLB 990 · LOL 280 · TENNIS 234 · CS 233 · MMA 214 · NHL 89 · KBO 36 ·
F1SZN 22 · CFL 11 · MOTORCYCLE 10 · NPB 5. 6,233 of the 12,272 lines have a single
`higher` option (TD scorers, goals, assists); 6,039 have both sides.

The feed is a strict subset of the lobby surface: the 29-pill NFL sweep (§7.6)
returned every one of the feed's 4,337 NFL markets plus 676 it does not carry: 541
team and game markets (`core`, `partial_core`, `team_prop`, `misc`) and 135 in-play
player props (`live_event: true`) **[P-21]**.

### 6.2 Lobby lines

| Endpoint | Auth | Size | TTL | Returns | Tag |
|---|---|---|---|---|---|
| `GET api…/v2/pickem_search/search_results[?sport_id=NFL]` | none | 23 KB/232 KB | 30 s | lists: `over_under_lines` (100 without `sport_id`, 96–97 with, so treat ~100 as the cap **[INF]**), `appearances`, `players`, `games`, `teams`, `sports`, `trending_players`, `subscriptions`, `opened_lines_count`, `providers` when prediction-market lines are present; options lack the `odds` key | [HAR] [P-4] |
| `GET api…/v1/lobbies/content/lines?sport_id=NFL&include_live=true&show_mass_option_markets=false` | none | 12 KB/107 KB | 30 s | dicts keyed by id plus an ordered `lines[]{over_under_line_id, appearance_id, entry_count ("75.2K"), header}`; 20 lines by default; `channels`, `subscriptions` (§6.5) | [HAR] [P-3] |
| `… lines?sport_id=NFL&filter_id=<uuid>&filter_type=MarketGroup\|PickemStat\|Custom&limit=N&show_mass_option_markets=true` | none | 0.1–2.5 MB decoded per pill | 30 s | one market-filter pill (§6.4, §7.6). `limit` caps the pill: Rushing returned 200 of its 241 lines at `limit=200` and all 241 at `limit=1000`; every other pill came back whole at `limit=1000` (largest: 1H Picks, 750); `page=2` returns the same rows as page 1, so there is **no pagination**. Without `filter_id` the sport and match views ignore `limit` and return 20 (`market_view` and `show_mass_option_markets` change nothing) | [P-5] [P-15] [P-18] [P-21] |
| `… lines?match_id=177184&match_type=Game&include_live=true&show_mass_option_markets=false` | none | 10 KB/94 KB | 30 s | the "Popular Picks" of one match: 20 player props, `limit` ignored; pair it with the match's `market_filters` pills (§6.3) for the rest | [HAR] [P-16] |
| `GET api…/v1/lobbies/content/match_grouped_lines?sport_id=NFL&market_categories[]=core&match_limit=30&include_live=true&show_more_picks_cta=true&two_box_enabled_surface=true` | none | 42 KB/429 KB | 30 s | `match_groups[]{id (int match id), type, over_under_line_ids[], entry_count, grid{row_headers, columns[]{title: Spread\|Moneyline\|Total, rows}}, cta_info}` for 30 matches plus the 90 core lines (62 active / 28 suspended) and `games`, `teams`; the only place a client can enumerate match ids on the new surface; omitting `market_categories[]` returns the same 90 lines | [HAR] [P-4] [P-20] |
| `… match_grouped_lines?match_id=177184&match_type=Game&…` | not probed | 3.6 KB/17 KB | 30 s | one match's core grid | [HAR] |
| `GET api…/v1/lobbies/content/pickem_lobby_sections?section_id=<uuid>` | none (one 401, §4) | 45 KB/550 KB | 30 s | one curated `scaffolds/home` rail: `content_type` `player_grouped_lines` (`player_groups[]{id, appearance_id, over_under_line_ids[]}`; a game, "Heavy Hitters", …) or `match_grouped_lines` (`match_groups[]` with a grid; "MLB Games", "Soccer Matches") plus dict-keyed entities; 200 lines for 10 players in the capture, 124 for the same section at 21:50; `has_alternates` flags but one line per (appearance, stat) | [HAR] [P-4] [P-29] [P-30] |
| `GET api…/v3/over_unders/<over_under_id>/alternate_projections` | none | 4–56 KB | 10 s | the app's alternates drawer (bundle `sourceEndpoint: "GET v3/over_unders/:overUnderId/alternate_projections"`): `projections[]{id, stable_id, entry_stable_id, is_main, stat_value, options[]}`, the balanced line (`is_main: true`) plus every rung, each with the full §7.1 options; `subscriptions` names the `lines-balanced;…` and `lines-alternate;Game-<id>;PickemStat-<uuid>;<provider>` channels; the ubiquitous trio is optional; empty `projections` once the market is gone; one market per call (§9.4) | [P-8] [P-12] [P-19] [P-22] [P-23] |
| `GET api…/v1/lobbies/content/lines_with_stats?appearance_id=<uuid>` | none | 10 KB | 30 s | the "picks" tab of a player page (`view_type: market_with_stats_list`): the player's balanced lines plus `stat_values_by_over_under_id` (recent game values per market, empty before week 1), not alternates | [DISC] [P-8] [P-26] |
| `… lines_with_stats?target_type=hot_hands` | none | 75 KB | 30 s | the home "Hot Hands" rail: 10 lines with their `stat_values_by_over_under_id` | [P-29] |
| `… lines?data_driven_line_source_id=<uuid>` ("Popular Picks"; `personalized/lines?…` in the capture) | none | 136 KB | 90 s | 23 lines, same shape as `lines` | [HAR] [P-30] |
| `… personalized/fantasy_lineup_cards?section_id=<uuid>` | **token** | 35 KB | private | `cards[]` for the account's best-ball lineup players with their lines, `supported_pick_sides`, `show_line` | [P-29] [P-31] |
| `… packs?section_id=<uuid>` | **token** | 124 B | private | `pack_ids`, `pickem_packs` (empty for this account) | [P-29] [P-31] |

### 6.3 Navigation and scaffolds

A scaffold is the app's own fetch plan for a screen: `sections[]{content_type,
view_type, title, data_source{path, url, endpoint}}` plus `key`, `name`, `analytics`,
`search_metadata`. Read a scaffold, then GET each section's `data_source.url`. All
URLs below are the section targets **[HAR]** unless tagged otherwise.

| Scaffold | Auth | Sections (content_type → target) | Tag |
|---|---|---|---|
| `GET api…/v1/lobbies/scaffolds/home?ct_token=w2&include_prediction_markets=true` | none (`private`) | ~50 sections in the capture, 42 at 21:45: `sport_navigation`; two `promo_carousel`s (`/v1/promo_carousels/main\|featured`); ~40 `pickem_lobby_sections?section_id=` rails (`NFL Games`, `49ers vs Rams`, `Heavy Hitters`, `US Open Matches`, …) of both `player_grouped_lines` and `match_grouped_lines` content; futures teasers `lines?filter_type=PickemStat&limit=3`; `lines?data_driven_line_source_id=` ("Popular Picks"); `personalized/fantasy_lineup_cards`; `lines_with_stats?target_type=hot_hands`; `packs` | [HAR] [P-25] |
| `GET …/scaffolds/sports?sport_id=NFL&ct_token=w2&include_prediction_markets=true&market_view=compact` | none | `sport_navigation?selected_id=NFL&selected_type=Sport`, `match_navigation?sport_id=NFL`, `market_filters?sport_id=NFL&empty_if_single_core_pill=true&include_live=true`, `sport_promo_carousel?sport_id=NFL&nonce=`, `match_grouped_lines?sport_id=NFL&…` ("Matchups"), `lines?sport_id=NFL&…` ("Popular Picks") | [HAR] [P-4] |
| `GET …/scaffolds/matches?match_id=177184&match_type=Game&ct_token=w2&…` | none (`max-age=90`) | `match_details?match_id=&match_type=`, `market_filters?target_id=177184&target_type=Game&show_futures=false&…`, `market_price_history?chart_type=ranked_market&target_id=&target_type=Game`, `match_grouped_lines?match_id=&match_type=`, `lines?match_id=&match_type=` | [HAR] |
| `GET …/scaffolds/live_matches?ct_token=w2&…` | none | `sport_navigation?live_only=true&live_tab_enabled=true`, then one `match_grouped_lines?sport_id=<X>&live_only=true&market_categories[]=core` per sport with live play (CFB, TENNIS, MLB, FIFA, CS, LOL, CRICKET at probe time) and a `lines?sport_id=PGA&filter_type=PickemStat` section | [DISC] [P-7] |
| `GET …/scaffolds/futures?sport_id=NFL&ct_token=w2&…` | none | `market_filters?…&match_type=Series`, then one `lines?filter_id=<uuid>&filter_type=PickemStat&show_mass_option_markets=true` per futures market (division wins, MVP, champion) | [DISC] [P-7] |
| `GET …/scaffolds/players?player_id=<uuid>&tab_id=picks&ct_token=w2&…` | none | `player_appearance_header?appearance_id=&player_id=`, `scaffold_navigation?player_id=&selected_id=picks&show_futures=true&show_game_log=true`, `lines_with_stats?appearance_id=` | [DISC] [P-7] |
| `GET …/scaffolds/sport_groups?sport_group_id=` | not probed | — | [DISC] |

Navigation content endpoints **[HAR]**:

| Endpoint | Size | Returns |
|---|---|---|
| `GET …/content/sport_navigation?live_tab_enabled=true&top_sports=NFL,MLB,FIFA,CFB,CS` | 0.9 KB/7.9 KB | public, 30 s **[P-25]**; 30 tabs (§6.4), each with an `action_path.url` to `scaffolds/home`, `scaffolds/live_matches` or `scaffolds/sports?sport_id=X` |
| `GET …/content/match_navigation?sport_id=NFL` | 0.3 KB/0.5 KB | public, 30 s **[P-25]**; on a pre-game Thursday: a single "Futures" selector and empty `games`, so it does **not** enumerate matches; use `match_grouped_lines` for match ids |
| `GET …/content/market_filters?sport_id=NFL&empty_if_single_core_pill=true&include_live=true` | 1.4 KB/11 KB | **token required**: 400 `Unable to construct the scaffold for market filter navigation.` without one, for sport and game targets alike, even with the browser's exact query **[P-25] [P-33]**; 29 pills in the capture, 32 with a token at 21:52 (§6.4) with `filter_id`, `filter_type` ∈ `Custom`, `MarketGroup`, `PickemStat`, `BadgedPill`; each pill's `action_path.url` is the `lines?filter_id=…` request to make. A token-less client pins the ids from §6.4 or asks by stat with `filter_type=PickemStat&filter_id=<pickem_stat_id>` **[P-34]** |
| `… market_filters?target_id=177184&target_type=Game&show_futures=false&…` | 1.5 KB/13 KB | the same for one match (token required as above; 28 pills for a Sunday game, no 2Q/3Q/2H Picks) **[P-37]**; `selected_content_id`, `subscriptions` |
| `GET …/content/match_details?match_id=177184&match_type=Game` | 1.3 KB/4.2 KB | public, 90 s **[P-25]**; scoreboard header, venue, weather, broadcasts, both teams; `channel: cache-game-177184` |
| `GET …/content/sport_promo_carousel?sport_id=NFL&nonce=` and `GET /v1/promo_carousels/main\|featured` | ≤1.7 KB/7.6 KB | promo cards, `private`; both answer without a token (empty `content`, `power_ups`, `voucher_offers`, `airdrop_offers`, `kym_cards`) **[P-27]**; useless for lines |

### 6.4 Reference data (`stats.underdogfantasy.com`, all public, 900 s)

| Endpoint | Size | Returns | Tag |
|---|---|---|---|
| `GET /v2/sports` | 5 KB/92 KB | 141 rows: `id`, `name`, `status`, `pickem_status` (14 `active`: CFB, CFL, CS, F1SZN, FIFA, KBO, LOL, MLB, MMA, MOTORCYCLE, NFL, NHL, NPB, TENNIS), `ladder_status` (`active` for NFL and FIFA only), `rivals_enabled` (true for all 141; stale, §6.1), `over_unders_enabled`, `active_in_game_pickem` (MLB only), `game_type` Game\|SoloGame, `periods`, `period_label`, `team_sport`, `matchup_based`, `pickem_team_stacks_allowed`, `draft_status`, `news_status`, images | [HAR] [P-4] |
| `GET /v1/teams` | 1.22 MB/11.3 MB | §6.1 | [HAR] |
| `GET /v1/lineup_statuses` | 0.5 KB/1.4 KB | id → abbr/display/auto_pick; the ids that `appearances[].lineup_status_id` points at | [HAR] [P-9] |
| `GET /v2/entry_styles` (1.8 MB decoded), `/v1/contest_styles`, `/v1/scoring_types`, `/v1/slots` | — | best-ball / draft products, not pick'em | [HAR] |
| `GET /v2/pickem_pool_styles`, `/v3/pickem_pool_styles` | 13–93 KB decoded | the **Pools** product (all inactive in TX), not the Power/Flex payout tables (those are quoted per slip by `entry_slips/estimate`, §6.8) | [HAR] |

`lineup_statuses` **[HAR]**:

| id | abbr | display | auto_pick |
|---|---|---|---|
| `2d7ccaac-7432-4576-83f3-4d930906a306` | Q | QUESTIONABLE | true |
| `e25f95d4-03b1-4810-94d4-4be310ae181a` | P | PROBABLE | true |
| `7df92a4e-60d9-4193-b4b1-88f89378fd2d` | O | OUT | false |
| `ebf65a0d-a6ea-4a9c-a120-d120f3b8b815` | I | IN | true |
| `49bfe837-b2c6-4f51-937d-4423b6aa8503` | O | OUT | false |
| `628397cd-ddbf-45ed-9a71-316f78c7abc8` | N/A | N/A | false |
| `ca85597a-944f-4c75-8149-470583d2ad22` | D | DOUBTFUL | true |
| `080bf70a-87d4-49d1-89ca-7aa33fb2c559` | C | CUT | false |
| `ab95060e-d65b-4f2d-8d6b-67e6c8b17193` | WD | WD | false |

`sport_navigation` tabs **[HAR]**: `featured`, `live_matches`, NFL, MLB, FIFA (Soccer),
CFB, CS (CS2), TENNIS, MMA (UFC), LOL, VAL, RAINBOW_SIX_SIEGE, OVERWATCH_2, WNBA, NBA,
PGA, RACING (NASCAR), F1SZN, BOXING, NPB, KBO, CFL, AUSSIE (AFL), RUGBY, CBB, WCBB,
NHL, MOTORCYCLE, CRICKET, CHESS. The tab id is the `sport_id` everywhere else.

NFL `market_filters` pills on 2026-09-10 **[HAR]**. `filter_id` + `filter_type` are
the `lines` query; ids are opaque and can change between seasons, and the endpoint
needs a token (§6.3), so pin these and re-read them as described below the table:

| `filter_id` | Title | `filter_type` |
|---|---|---|
| `popular` | Popular | `Custom` |
| `dd701b83-af7e-496e-b1c1-184301556335` | Team Picks | `MarketGroup` |
| `a02dd90d-4906-436a-b25d-6742d0bc8f99` | TD Scorers | `MarketGroup` |
| `9741276e-5413-4b04-8c2a-4be10609721a` | Passing | `MarketGroup` |
| `6924be80-da5b-467d-8106-e2d66a7c4095` | Receiving | `MarketGroup` |
| `4e6d05ef-a158-4d8b-b1b6-801923c7baa2` | Rushing | `MarketGroup` |
| `a8ad0c2c-4466-4ea7-a0ef-4768da712547` | Combo Yards | `MarketGroup` |
| `069d9262-8a77-4cce-a52f-51c17a9f9e7a` | 1Q Team Picks | `MarketGroup` |
| `0d60f246-332d-4bdf-9195-966603a01311` | 1Q Picks | `MarketGroup` |
| `7b4c392d-f8fe-4293-aeee-7e1833073e5a` | 2Q Team Picks | `MarketGroup` |
| `33dc0bff-2565-4012-8e67-01c42b275422` | 1H Team Picks | `MarketGroup` |
| `32d972e5-c986-43e5-8595-7dd7708fdde0` | 1H Picks | `MarketGroup` |
| `59179c3c-176b-4c51-8858-8b03dd0b960b` | Team Totals | `MarketGroup` |
| `d173a2b6-fe91-424e-86ee-0e6cafa14e25` | Kicking | `MarketGroup` |
| `de78fd23-65bb-48dc-ad1b-ac8da6c3e3ab` | Defense | `MarketGroup` |
| `8096392f-3bee-4a9d-b06a-51ba7c3c1f9d` | Fantasy Points | `PickemStat` |
| `e51d168f-58ef-40c9-9b9b-593ec2f4537d` | Each Quarter | `MarketGroup` |
| `87b48863-b6b1-416b-bcd9-f0ff9db53cee` | Each Half | `MarketGroup` |
| `09c12426-e47f-4b74-acde-e8974bd9de8f` | TD Picks | `MarketGroup` |
| `160c4e5b-088e-4115-a481-ba5eebaba824` | Race to X | `MarketGroup` |
| `2864b0ca-a52f-4095-95f2-7ad8e816bcb9` | Both Teams to Score | `MarketGroup` |
| `ac1bb933-2947-437b-9e47-714f0be9b0b0` | Game High | `MarketGroup` |
| `623b6994-1be3-40a8-a4f6-937d494715ea` | Halftime/Fulltime | `PickemStat` |
| `14fde018-4c36-4c49-88d8-5ca1449e4683` | Winning Margin | `PickemStat` |
| `fdb40be2-5032-404a-b174-93eac7548136` | Overtime | `MarketGroup` |
| `92b50fa9-4455-4f12-94dc-176960fed6d9` | 2H Team Picks | `MarketGroup` |
| `e66445e0-2dec-4946-bc6f-b84d03496b29` | 3Q Team Picks | `MarketGroup` |
| `90dc8d4c-4ea1-40dd-aed5-c7de567bfeb0` | 4Q Team Picks | `MarketGroup` |
| `0541f59d-92b1-4b3b-942f-53e5663d5c69` | Fumbles Lost | `PickemStat` |
| `b30275cc-50f4-4f86-8ead-b01c9e9ed510` | 2Q Picks | `MarketGroup` (added by 21:52) |
| `6ca78f37-982c-46c9-8aa8-0d013983933d` | 3Q Picks | `MarketGroup` (added by 21:52) |
| `9cfe2176-d37f-4ddd-8d23-b7fc3aab7c9a` | 2H Picks | `MarketGroup` (added by 21:52) |

The last three were not in the capture; a tokened read at 21:52 listed 32 pills, the
same 29 ids plus these **[P-33]**. `market_filters` itself needs a token (§6.3), so a
scraper pins these ids and re-reads the list with a token when a pill sweep starts
returning empties. The `PickemStat` pills use the stat's own `pickem_stat_id` as
`filter_id` (Fantasy Points `8096392f…` is the same id in §7.4), and a bare
`lines?sport_id=NFL&filter_type=PickemStat&filter_id=<pickem_stat_id>&limit=1000`
returns that stat's whole board whether or not a pill exists for it (rushing yards:
81 lines, 408 KB) **[P-34]**.

### 6.5 Realtime

| Item | Detail | Tag |
|---|---|---|
| Pusher app key | `d65207c183930ff953dc`, the prefix of the auth string returned by `POST api…/v1/authorizations/pusher_channels` (form body `socket_id`, `channel_name`; the channel authorized was the account's private channel) | [HAR] |
| Cluster / websocket host | unknown: not in the three bundles the index page loads nor in the `page.pick-em` and `entry.app` chunks (the Pusher client is in a lazily loaded vendor chunk the runtime names only by hash) | [P-8] open |
| Public line channels | every lobby-content response lists `channels` (`over_under_lines-NFL-balanced`; `cache-game-177184` on match details) and `subscriptions[]{schema: "over_under_lines", channels: ["lines-balanced;Game-<id>;PickemStat-<uuid>;non_prediction_market", "…;udx", "…;kalshi"]}`; no `private-` prefix, so presumed subscribable without auth | [HAR] [INF] |
| Price-history push | `market_price_history` answers with `scope.pusher_channel: "chart-data"` and `scope.pusher_key: "over_under:<over_under_id>:option_normalized_probability"`, plus `polling_interval_seconds: "5"` for clients without push | [P-6] |
| Live legacy feed | `beta/v2/live_over_under_lines` is the polling alternative for in-play lines (§6.1) | [P-11] |

### 6.6 Price history

`GET api…/v1/lobbies/content/market_price_history?chart_type=ranked_market&target_id=177184&target_type=Game`
**[P-6]**, 5.5 KB decoded, public, `max-age=30, stale-while-revalidate=300`. Returns
the chart for the match's ranked market (here the moneyline, a prediction-market
contract): `scope{metric: "option_normalized_probability", over_under_id, sport_id,
refresh_url, news_items_url, volume_ticker_url, pusher_key, pusher_channel,
polling_interval_seconds}`, `filters[]` (`PT1H`, `P1D`, `P1W`, `all`, each with a
`filter_path`), `series[]` (one per option, keyed
`home__<appearance_id>__<over_under_id>`), `axes`, and
`data{timepoints[68], series_values{<key>: ["63.0", …]}}` covering 2026-07-07 to the
request time as percent strings. The `refresh_url` family is a second, undocumented
surface **[DISC]**: `GET /v1/over_unders/<over_under_id>/chart_data?metric=…[&time_range=P1D]`,
`…/chart_news_items`, `…/volume_ticker`.

### 6.7 Account endpoints (not needed for lines)

Token required **[HAR] [P-31]**: `GET /v1/user` (`user.id`, the value of the
`UD-User-ID` header in §6.8; `state_config_id`, `product_experience`,
`lobby_experiments_token`, `lobby_features`, wallet), `/v1/user/profile`,
`/v1/user/bonus_wallet_accounts`, `/v1/user/entry_slip_limits` (`limits[]{sport_id,
max_fee, max_appearance_fee, sport_limit, user_override}`, one row per sport: the stake
caps a pricer must respect), `/v1/user/power_ups` (a `promo_carousel` of the account's
boosts with `restrictions`), `/v1/user/referral_stats`,
`/v1/intercom/identity_verification`. `/v1/user/state_configs` also wants the geo
headers: 422 `location_needed` with a token and no `User-Latitude` / `User-Longitude`,
200 (79 KB, `private`) once those two are sent; the Radar `User-Location-Token` is not
needed **[P-28] [P-31] [P-36]**. Its body carries `state_config.pick_em`
(`maximum_selection_size: 8`, `minimum_selection_size: 2`,
`pickem_ladders_enabled: true`, `pickem_streaks_enabled: true`,
`pickem_packs_enabled: true`, `prediction_markets_enabled: true`,
`payout_modifiers: true`, `pickem_cash_out_enabled`,
`pickem_opening_lines_only_enabled`, `pickem_pools_*`, `pickem_swipe_enabled`,
`spin_to_win_enabled`, `guaranteed_pickem_pool_payout_coefficient`, `visible_stats`,
`visible_sports`) **[HAR] [P-36]**.

Public despite the name **[P-27]**: `/v1/features` (383 flags, 30 s) and
`/v3/user/features` (`general_features`, `pickem_features`, `pricing_experiment_key`;
with a token `pickem_features` lists `2026-no-stakeback`, `2026-prediction-boosts`,
`2026-reduced-friction-promo-handling`, `2026-suggested-packs`, without one it is empty
and the key null). Flags of note: `web_scaffold_lobby`,
`clients_pickem_lobby_pagination: false`, `clients_original_ladders`,
`ios_original_ladders`, `android_original_ladders_wip`, `allow_multiple_streaks`,
`clients_pusher_connection`, `algolia_search` **[HAR]**.

### 6.8 Entry-slip pricing

The slip builder prices every selection set through `GET /v3/entry_slips/estimate`
(bundle: `underdogAPIVersion: "v3"`; header `UD-User-ID: <user.id from /v1/user>`;
params `estimate_request_id=<uuid4 the client makes>`, `fee`, `fee_source` (`cash` |
`bonus`), `options[i][id]=<option id>`, `options[i][type]=OverUnderOption`, optional
`power_up_id` and `features[]`), token required **[P-8] [P-32]**. It answered
synchronously (`status: "valid"` | `"invalid"`) on every one of the 100-odd calls made
here; `async_config{
wait_seconds: 2, polling_interval_seconds: 2, timeout_seconds: 15}` and `subscriptions`
(the picked lines' Pusher channels) cover the `pending` case the bundle polls for.
Nothing is created: the entry itself is a separate POST this document never made.

Response **[P-32]**: `odds{multiplier, type: "modifier", visual{value, icon_url}}` (the
slip's payout multiplier after correlation modifiers), `flex{enabled,
min_selection_count: 3, max_selection_count: 8, error, offerings[]{max_losses,
pricing{"<losses>": {losses, odds{american, decimal, probability}, unmodified_odds}}}}`
(the Flex table for this slip; `decimal` is the payout multiplier), `errors[]{error_key,
detail, invalid_options[]{id, line_id, prop_id (= over_under_id), stable_id}}`,
`combinability_rules`, `max_fee: 5000`, `min_fee: 0.1`, `max_multiplier: 5000`,
`min_selection_count: 2`, `max_selection_count: 8`, `max_prediction_selection_count: 4`,
`eligible_fee_sources`, `min_bonus_multiplier: 2`, `pool_styles[]` (the Pools product:
`style_id`, `rake`, `min_fee`, `max_fee`, `total_fees`), `promo`, `predictions_quote`
(prediction-market legs: `average_unit_cost`, `total_quantity`, `total_cost`,
`fees[]{issuer: Kalshi | UDX, amount, type: per_contract, total}`, `winning_payout`),
`combo_fantasy_component_odds{multiplier, type, visual}` (the fantasy legs' own
multiplier inside a mixed slip), `checksum`.

Quotes on 2026-09-10 for $5 slips of even-money (`payout_multiplier` 1.0) player props
from different games **[P-32]**:

| Picks | Power `odds.multiplier` | Flex `offerings` (`decimal` by losses; `max_losses` 1 up to 5 picks, 2 from 6) | `underdog_payouts.json` today (`power`; `flex`; `insurance`) |
|---|---|---|---|
| 2 | 3.5 | not offered (needs 3) | 3.0; —; 3 / 0 |
| 3 | 6.5 | 0: 3.25 · 1: 1.09 | 6.0; 2.25 / 1.25; 6 / 0 |
| 4 | 12.0 | 0: 7.2 · 1: 1.8 | 10.0; 5.0 / 1.5 / 0.4; 10 / 0 |
| 5 | 20.0 | 0: 10.0 · 1: 2.5 | 20.0; 10 / 2.0 / 0.4 / 0.4; 10 / 2.5 |
| 6 | 35.0 | 0: 25.0 · 1: 2.6 · 2: 0.25 | 25.0; 25 / 2.0 / 0.4 / 0.4 / 0.4; 25 / 2.6 / 0.25 |
| 7 | 65.0 | 0: 40.0 · 1: 2.75 · 2: 0.5 | absent |
| 8 | 120.0 | 0: 80.0 · 1: 3.0 · 2: 1.0 | absent |

The static file matches the live Flex only through its `insurance` rows for 5 and 6
picks, is off everywhere else, and stops at 6 picks, so the quote, not the file, is the
settle-truth for a priced slip. Composition and rules seen **[P-32] [P-35]**:

- A 2.57× rung (James Cook 1.5+ TDs) with an even pick quoted 8.99 = 3.5 × 2.57: rung
  multipliers multiply into the table.
- Same-team QB + WR alone: `invalid`, `at_least_two_teams` ("Pick at least 2 players
  from different teams."); with a third pick from another game: `valid` at **5.17**
  against 6.5 uncorrelated (Flex 2.47 / 0.82 against 3.25 / 1.09): the correlation
  modifier, mapped below.
- The same player twice (rush yards + rush attempts): `at_least_two_players`.
- A team total with one player pick: `invalid_combo_selection_count` ("Remove your
  player pick or add more picks"); the team line is a prediction-market contract
  (`prediction_market: true`) and the quote carries a Kalshi `predictions_quote`
  ($0.02 per contract).
- A moneyline contract plus two player props: `valid`, 6.91 overall with
  `combo_fantasy_component_odds` 3.5 for the two props and a UDX `predictions_quote`
  (`average_unit_cost` 0.4578, `total_quantity` 9, `total_cost` 4.12,
  `winning_payout` 9).
- Priced picks: three `higher` options at 0.87×, 1.16× and 0.74× (product 0.747)
  quoted Power **4.85** = 6.5 × 0.747, Flex 0-loss **2.42** = 3.25 × 0.747 and Flex
  1-loss **1.10** = 1.09 × 0.87 × 1.16 (the two largest multipliers; formula below)
  **[P-35] [P-39]**.
- `fee` (5 or 100) and `fee_source` (`cash` or `bonus`) leave the quote unchanged;
  `odds.visual.value` read `no_impact` on every quote, taxed ones included **[P-38]**.

Which pairs carry a correlation modifier (3-pick slips: the pair plus an even filler
from another game; uncorrelated = 6.5 Power, 3.25 / 1.09 Flex) **[P-32] [P-35] [P-38]**:

| Pair (same game) | Sides | Power | Flex 0 / 1 loss |
|---|---|---|---|
| QB pass yds + same-team WR rec yds | both higher | 5.17 (one game), 5.68 (another) | 2.47 / 0.82; 2.50 / 0.83 |
| same pair | higher + lower | 6.5 | 3.25 / 1.09 |
| same pair | both lower | 5.35 | 2.47 / 0.82 |
| QB pass yds + same-team WR receptions | both higher | 5.92 | 2.66 / 0.89 |
| QB pass TDs (1.2×) + same-team WR rush+rec TDs | both higher | 7.13 (7.8 untaxed) | 3.90 / 1.30 |
| QB pass yds + opposing QB pass yds | both higher | 6.5 | 2.92 / 0.98 |
| QB pass yds + opposing WR rec yds | both higher | 6.5 | 3.25 / 1.09 |
| QB pass yds + same-team RB rush yds | both higher | 6.5 | 3.25 / 1.09 |
| WR rec yds + same-team WR rec yds | both higher | 6.5 | 3.25 / 1.09 |
| RB rush yds + opposing RB rush yds | both higher | 6.5 | 3.25 / 1.09 |

How the number arises **[P-39] [P-40] [P-41]** (74 designed quotes on CIN, NYJ and
IND pairs, every one deterministic on repeat):

- **Formula.** Power = `T_n` × Π(pick multipliers) × `m`; Flex tier with `k` losses =
  `T_n,k` × Π(the `n−k` largest pick multipliers) × `m`, so the 1-loss tier is quoted
  as if the smallest-multiplier pick missed (0.87 / 1.16 / 0.74 picks: 1.09 × 0.87 ×
  1.16 = 1.10; a 7.5× rung with two even picks: 1.09 × 7.5 × 0.56 = 4.57). `m` ≤ 1.
- **`m` is a joint-probability lift over the same-game picks**, `m` = Π p_i /
  P(all hit), clipped at 1. It ignores the other picks (three fillers, 3 to 6 picks:
  Power `m` 0.7923–0.7926, Flex 0.76), multiplies across games (two taxed pairs from
  two games: 0.6483 against 0.7925 × 0.8183 = 0.6485) and does not multiply within a
  game (QB + WR1 + WR2: 0.716 against a pairwise product of 0.620, which is what a
  trivariate joint with ρ ≈ −0.2 between the two receivers gives; the WR + WR pair
  alone comes back at exactly 1.0, as do opposite sides, because negative association
  never pays above the table).
- **Flex uses a Gaussian copula with one ρ per pair type.** QB pass yds × same-team
  receiving yds: ρ 0.48 for all 16 receivers on three teams (WR1 down to a third TE,
  lines 9.5 to 87.5), both-under equal to both-over, and a fixed ρ reproduces all nine
  rung quotes (49.5 to 119.5 rec yds, 229.5 to 309.5 pass yds, both sides: fitted ρ
  0.44–0.51 against the de-vigged ladder probabilities). Other types at the main
  lines: fantasy points × fantasy points 0.36, fantasy points × yards 0.26, pass yds
  × opposing pass yds 0.18, pass TDs × opposing pass TDs ≈ 0.2, pass TDs × rec yds
  ≈ 0.25, pass TDs × receiver TDs ≈ 0, pass yds × kicker FGs ≈ 0, pass yds × rush yds
  (own RB or opponent), WR × WR, opposing WR or RB pairs: untaxed.
- **Power uses a pair-specific, asymmetric joint.** Same QB, main lines, implied ρ:
  Chase 0.40, Higgins 0.43, RB receiving 0.31, TE 0.19, WR3 0.23, TE2 0.17; the other
  two teams 0.12–0.75 with no relation to line size; a rookie TE and the cross-team
  QB-yards pair come back untaxed; both-under differs from both-over (Chase 0.57 vs
  0.40, one WR 0 vs 0.75, one TE 0.14 vs 0.61); the rung-implied ρ drifts 0.29–0.58.
  That is an empirical joint from shared game history, independent where there is
  none **[INF]**. Fantasy-points pairs fall back to the Flex values.

So every positive-ρ pair Power prices at 1.0 is free correlation, and Flex charges a
low-target receiver the same 0.48 as the WR1; the pricer's use of this is in
[dfs-products.md](handoffs/dfs-products.md).

Cash-out on a live entry follows the same pattern **[P-8]**: `GET /v1/entry_slips/<id>/cash_out_requests`
polls a quote and `POST` accepts it (`estimate_request_id`, `payout_expected`,
`price_expected`); never called here.

### 6.9 Not seen anywhere

No ladders endpoint: `ladder_status: active`, `pickem_ladders_enabled: true` and the
flags `ios_original_ladders` / `android_original_ladders_wip` / `clients_original_ladders`
exist, the promo card deep-links to `underdogfantasy://ladders`, none of the fetched web
bundles mentions a ladder route, and `scaffolds/ladders`, `pickem_ladders` and
`content/ladders` are 404 with a token **[P-31]**: ladders look mobile-app only
**[INF]**; the owner has pinned that capture for now. No rivals anywhere: the product is
retired and Ladders replaced it **[OWNER]**, and `beta/v3/rival_lines` answers empty
with and without a token **[P-31]** (§6.1). No streaks feed, and no batch form of the
alternates fetch (§6.2). `state_configs`, `features` and `user` bodies contain no payout
multipliers **[P-10]**; slips are priced by §6.8.

## 7. Payload reference

### 7.1 The line object

Identical on the legacy feed, `lines`, `match_grouped_lines`, `pickem_lobby_sections`
and `search_results` (the last drops `options[].odds`) **[HAR] [P-1]**:

```
over_under_line {
  id, over_under_id, stable_id ("<over_under_id>|balanced"), entry_stable_id,
  line_type ("balanced"), status ("active" | "suspended"),
  stat_value (string; null on moneyline), non_discounted_stat_value,
  live_event (bool), live_event_stat (running stat while live, else null),
  expires_at, provider_id ("manual" | "swish" | "udx" | "kalshi" | "nadex" | null),
  rank, sort_by, updated_at, contract_url, contract_terms_url,
  over_under {
    id, title, category ("player_prop" | "core" | "partial_core" | "team_prop" | "misc"; §7.6),
    display_mode ("default" | "moneyline" | "spread"), has_alternates (bool),
    boost (null in the capture), multi_provider, prediction_market (bool),
    grid_display_title, option_priority, scoring_type_id, team_divider,
    appearance_stat { id, appearance_id, pickem_stat_id, stat (slug), display_stat,
                      graded_by, rank, hide_current_stat_value }
  },
  options[] {
    id, over_under_line_id, choice ("higher" | "lower" | "away" | "home" | "draw" | "yes" | "no"),
    choice_id ("over__" | "under__" | "draw__" | "yes__" | "no__" | "away__<appearance_id>" | "home__<appearance_id>"),
    choice_display, choice_display_short, choice_display_name_shorter,
    payout_multiplier (string), american_price, decimal_price,
    raw_probability (null on fantasy lines; "0.36" on prediction-market lines),
    odds { fantasy { american, decimal, probability, type: "precision", experiments }
           | prediction { american, decimal, probability } | sportsbook },
    appearance_id (the team's appearance on away / home options, else null), lookup_id, grouping_id, rfq_combinable,
    selection_header, selection_subheader, status, type ("OverUnderOption"), updated_at
  }
}
```

Trimmed example, Kyren Williams Rush Yards 54.5 on 2026-09-10 **[HAR]**:

```json
{
  "id": "f4c1dec0-c99c-423d-bd1b-a595cca76781",
  "over_under_id": "b39253a6-85a8-4c64-8e3f-408281f3580c",
  "stable_id": "b39253a6-85a8-4c64-8e3f-408281f3580c|balanced",
  "line_type": "balanced",
  "status": "active",
  "stat_value": "54.5",
  "live_event": false,
  "expires_at": null,
  "provider_id": "manual",
  "updated_at": "2026-09-10T23:37:40Z",
  "over_under": {
    "id": "b39253a6-85a8-4c64-8e3f-408281f3580c",
    "title": "Kyren Williams Rush Yards O/U",
    "category": "player_prop",
    "display_mode": "default",
    "has_alternates": true,
    "boost": null,
    "prediction_market": false,
    "appearance_stat": {
      "appearance_id": "7e076ff5-6e3f-40ef-ab21-af518bfedb06",
      "pickem_stat_id": "880784a3-e88f-4bc8-bbb2-ad44fb721a48",
      "stat": "rushing_yds",
      "display_stat": "Rush Yards",
      "graded_by": "high_score"
    }
  },
  "options": [
    {
      "choice": "higher",
      "choice_id": "over__",
      "choice_display_name_shorter": "55+",
      "payout_multiplier": "1.0",
      "american_price": "-112",
      "decimal_price": "1.9",
      "raw_probability": null,
      "odds": {
        "fantasy": {"american": "-115", "decimal": "1.87", "probability": "50", "type": "precision"},
        "prediction": null,
        "sportsbook": null
      },
      "selection_header": "Kyren Williams",
      "selection_subheader": "Higher 54.5 Rush Yards",
      "status": "active"
    },
    {
      "choice": "lower",
      "choice_id": "under__",
      "choice_display_name_shorter": "54-",
      "payout_multiplier": "1.0",
      "american_price": "-112",
      "decimal_price": "1.9",
      "raw_probability": null,
      "odds": {
        "fantasy": {"american": "-115", "decimal": "1.87", "probability": "50", "type": "precision"},
        "prediction": null,
        "sportsbook": null
      },
      "selection_header": "Kyren Williams",
      "selection_subheader": "Lower 54.5 Rush Yards",
      "status": "active"
    }
  ]
}
```

### 7.2 Entities around a line

```json
{
  "appearance": {
    "id": "7e076ff5-6e3f-40ef-ab21-af518bfedb06",
    "player_id": "df0c7db6-0a74-4343-a31e-0b64e2d6b0ea",
    "team_id": "d150534e-6a05-587b-b9e3-50ef86602e20",
    "position_id": "e2571131-7600-5bc9-878b-946a992c8203",
    "match_id": 177184,
    "match_type": "Game",
    "type": "Player",
    "lineup_status_id": null,
    "badges": [],
    "multiple_picks_allowed": true
  },
  "player": {
    "id": "df0c7db6-0a74-4343-a31e-0b64e2d6b0ea",
    "first_name": "Kyren",
    "last_name": "Williams",
    "sport_id": "NFL",
    "team_id": "d150534e-6a05-587b-b9e3-50ef86602e20",
    "position_name": "RB",
    "position_display_name": "Running Back",
    "jersey_number": "23",
    "action_path": {"url": "https://api.underdogfantasy.com/v1/lobbies/scaffolds/players?player_id=df0c7db6-0a74-4343-a31e-0b64e2d6b0ea&tab_id=picks"}
  },
  "game": {
    "id": 177184,
    "sport_id": "NFL",
    "type": "Game",
    "status": "scheduled",
    "title": "SF @ LAR",
    "short_title": "49ers @ Rams",
    "full_team_names_title": "San Francisco 49ers @ Los Angeles Rams",
    "away_team_id": "7161e62b-de20-56e2-a300-0dc23637faaa",
    "home_team_id": "d150534e-6a05-587b-b9e3-50ef86602e20",
    "scheduled_at": "2026-09-11T00:35:00Z",
    "match_progress": "Thu 08:35pm",
    "period": 0,
    "away_team_score": 0,
    "home_team_score": 0,
    "season_type": "regular",
    "year": 2026,
    "pre_game_data": {"venue": {"name": "Melbourne Cricket Ground", "type": "outdoor"}, "weather": {"condition": null, "temperature": null}, "broadcasts": ["Netflix"]}
  },
  "team": {
    "id": "d150534e-6a05-587b-b9e3-50ef86602e20",
    "abbr": "LAR",
    "name": "Los Angeles Rams",
    "short_name": "Rams",
    "sport_id": "NFL",
    "sport_radar_id": "2eff2a03-54d4-46ba-890e-2bc3925548f3"
  }
}
```

Games also carry `scoreboard`, `scoreboard_data`, `manually_created`,
`rescheduled_from`, `sport_group_id`, `matchup_color_roles`; `solo_games` (tennis,
MMA, racing) carry `away_player_id` / `home_player_id` / `competition_id` instead of
team ids; players carry image URLs and `country` **[HAR] [P-1]**.

`appearances[].type` is `Player`, `Team` (`player_id` null, `team_id` set: team props)
or `Match` (both null: moneyline, spread, total and the `misc` game props);
`lineup_status_id` and `position_id` are null on the last two **[P-21]**.

### 7.3 Id and container semantics

- `over_under_id` names the market (appearance × stat) and survives line moves: eight
  NFL markets re-read 2.5 h apart kept it while `stat_value` and the line `id` changed
  (Kyren Williams rush yards 54.5 → 55.5) **[P-13]**; `id` names the priced line;
  `stable_id` is `<over_under_id>|balanced` (`|alternate` on rungs, §9.4) and is what
  the grid and the Pusher channel names use; `appearance_stat.id` is
  `<appearance_id>-<pickem_stat_id>` **[P-21]**. `pickem_stat_id` is the stat's global id; `appearance_stat.stat`
  (`rushing_yds`, `period_1_2_passing_yds`) is the stable machine slug and
  `display_stat` the label the current parser keys on **[HAR]**.
- `match_id` is an **integer** on the new surface and in the legacy feed today; the
  scraper keys matches on `str(id)` **[CODE]**.
- `games[].abbreviated_title` is `AWAY @ HOME` in the US sports and `HOME vs AWAY` in
  soccer, but `AWAY vs HOME` in esports (every CS and LOL game checked against the
  lobby `teams` dicts and `/v1/teams`), and one CFL game contradicted its own ids
  **[P-42]**; the scraper takes abbreviations from the lobby `teams` dicts and reads
  the title only for leagues it does not model.
- Lobby content keys `over_under_lines`, `appearances`, `players`, `games`, `teams`,
  `sports`, `series`, `solo_games` by id (dicts); ordering lives in `lines[]`,
  `match_groups[]` or `player_groups[]`. `search_results` and the legacy feed return
  lists **[HAR] [P-1]**.
- `entry_count` is a display string (`"75.2K"`, `"1.3M"`), not a number **[HAR]**.
- `choice` is `higher` / `lower` on player props, team totals and game totals;
  `away` / `home` on moneylines and spreads (plus `draw` on the three-way period
  moneylines); `yes` / `no` on the `misc` game props. `stat_value` is null on
  moneylines **[HAR] [P-21]**.

### 7.4 Pricing fields

`payout_multiplier` is what the scraper reads per side (`_ud_boosts` in
[books/underdog.py](../src/sportstradamus/books/underdog.py)) and what the app shows;
treat it as the settle value. The other fields are representations of the same price
whose exact relationship is **[INF]** until a settled entry confirms it:

- In the capture 109 of the 234 two-sided lines carry unequal multipliers (Rush + Rec
  TDs 0.5: higher `0.84` / `-150` / `60`, lower `1.10` / `+122` / `45`) **[HAR]**.
- `american_price` and `odds.fantasy.american` disagree on 201 of 206 fantasy options
  (`-112` vs `-115` at 1.0×); `decimal_price` ≈ `payout_multiplier` × 1.9 while
  `odds.fantasy.decimal` ≈ × 1.87 **[HAR]**. One is the standard-entry price and the
  other an experiment variant (`odds.fantasy.experiments`, `type: "precision"`, and the
  `2026-american-odds` experiment in `experiments_token`) **[INF]**.
- `odds.fantasy.probability` is a percent string; the two sides of a line sum to
  100–106, so it is the implied probability with vig, not a fair probability **[HAR]**.
- Prediction-market lines (`prediction_market: true`, `provider_id` udx / kalshi /
  nadex, 90 core lines in the capture) fill `odds.prediction` and `raw_probability`
  instead and carry a `lookup_id` and `contract_url`; `providers{}` on the response
  gives each provider's `license_type` (`UDX`, `FCM`) **[HAR]**.
- `odds.sportsbook` is null on every lobby and legacy option seen but filled on the
  player-prop options `alternate_projections` returns (main line and rungs; its
  `american` equals `american_price`), and null again on team-market rungs
  **[P-12] [P-19] [P-23]**.
- A slip's Power multiplier is the per-option `payout_multiplier`s multiplied into the
  table `entry_slips/estimate` quotes (§6.8): a 2.57× rung with an even pick = 8.99 =
  3.5 × 2.57; a Flex tier with `k` losses multiplies in the `n−k` largest of them
  **[P-32] [P-35] [P-39]**.

### 7.5 NFL stat vocabulary

50 distinct `(pickem_stat_id, stat, display_stat)` triples across the capture's 307
lines; 17 of the 50 `display_stat` values have a `stat_map.json["Underdog"]` entry
**[HAR] [CODE]**. The unmapped ones are period props, core markets, kicking, defense
and TD-scorer markets.

| `stat` | `display_stat` | category | `pickem_stat_id` | in `stat_map` |
|---|---|---|---|---|
| `moneyline` | Moneyline | core | `0251dd94-773d-47ec-878d-8a7349b8b967` | no |
| `points` | Total Points | core | `8f654930-4852-4510-babc-58ba0ff9840f` | no |
| `spread` | Spread | core | `42ae12ae-89ce-49f3-80fa-2dc1ab9338f8` | no |
| `100_pass_yds_each_half` | 100+ Pass Yards in Each Half | player_prop | `2ec58719-538d-4c3d-82fd-f069498d235d` | no |
| `10_rec_yds_each_quarter` | 10+ Rec Yards in Each Quarter | player_prop | `ba344a33-d946-433c-9409-754823ae54de` | no |
| `10_rush_yds_each_quarter` | 10+ Rush Yards in Each Quarter | player_prop | `4a081599-d495-4199-b14d-915da31d2683` | no |
| `150_pass_yds_each_half` | 150+ Pass Yards in Each Half | player_prop | `0a67a375-7362-4a2f-860d-3ad284649097` | no |
| `25_pass_yds_each_quarter` | 25+ Pass Yards in Each Quarter | player_prop | `40d5b61c-9f4e-4195-9042-3f904011df6d` | no |
| `25_rec_yds_each_half` | 25+ Rec Yards in Each Half | player_prop | `7f1b362d-a636-48f8-97bf-5d1ca0a095df` | no |
| `25_rush_yds_each_half` | 25+ Rush Yards in Each Half | player_prop | `42511380-1370-4d7c-9d50-56c9428926de` | no |
| `50_pass_yds_each_quarter` | 50+ Pass Yards in Each Quarter | player_prop | `8e8b8908-597c-479b-b4fc-20109b72dd64` | no |
| `50_rec_yds_each_half` | 50+ Rec Yards in Each Half | player_prop | `836d6b2b-2d1b-49de-97a5-dbf6eb09323c` | no |
| `50_rush_yds_each_half` | 50+ Rush Yards in Each Half | player_prop | `423ec329-8f16-49cd-9445-44415b2d2baf` | no |
| `5_rec_yds_each_quarter` | 5+ Rec Yards in Each Quarter | player_prop | `4b4e3c48-b05a-4b1f-8b99-3421df41c41d` | no |
| `5_rush_yds_each_quarter` | 5+ Rush Yards in Each Quarter | player_prop | `9512703d-55bf-44da-a3a2-d384f990ed4a` | no |
| `fantasy_points` | Fantasy Points | player_prop | `8096392f-3bee-4a9d-b06a-51ba7c3c1f9d` | yes |
| `field_goals_made` | FG Made | player_prop | `dfe9ec6e-5c5e-4db0-895b-905ca5634c5a` | no |
| `fumbles_lost` | Fumbles Lost | player_prop | `0541f59d-92b1-4b3b-942f-53e5663d5c69` | yes |
| `last_touchdown_scored` | Last TD Scorer | player_prop | `ca2e0c24-41ea-4422-92cd-52ee612eab17` | no |
| `passing_and_rushing_yds` | Pass + Rush Yards | player_prop | `bb9cb80b-b2cb-45ab-830d-4f4d30f3795d` | yes |
| `passing_att` | Pass Attempts | player_prop | `de868934-c920-405c-b827-693c15aa47a1` | yes |
| `passing_comps` | Completions | player_prop | `9ef21f52-a282-4ad3-ba87-3dc4c6bbdf6e` | yes |
| `passing_ints` | INTs Thrown | player_prop | `2941dd67-044a-49b0-b6c9-4ed2d89dc3cb` | no |
| `passing_long` | Longest Completion | player_prop | `58c855d4-714b-46de-bf89-07125370f825` | yes |
| `passing_tds` | Pass TDs | player_prop | `860b5aaa-97df-49fb-9ce2-84d56ddaa0ec` | yes |
| `passing_yds` | Pass Yards | player_prop | `55d1a168-e9bf-4a40-8440-fb8a7a07713a` | yes |
| `period_1_2_passing_tds` | 1H Pass TDs | player_prop | `11ea7230-d0d9-48c3-a037-fee6e8d15ee6` | no |
| `period_1_2_passing_yds` | 1H Pass Yards | player_prop | `33e16474-d406-411c-b2ae-0999991b64a2` | no |
| `period_1_2_receiving_rec` | 1H Receptions | player_prop | `8a47ab16-d106-4a13-93c4-ef55de26b6f4` | no |
| `period_1_2_receiving_yds` | 1H Rec Yards | player_prop | `e9a5a389-7854-4f61-a131-74730c69b45e` | no |
| `period_1_2_rush_rec_tds` | 1H Rush + Rec TDs | player_prop | `8c315de8-5438-41b1-9ae9-2f26e4c51221` | no |
| `period_1_2_rushing_yds` | 1H Rush Yards | player_prop | `d7101678-7e4d-43b1-985d-071d8fb1b7d6` | no |
| `period_1_passing_tds` | 1Q Pass TDs | player_prop | `fe426ad9-5ba2-4eb0-9a01-236248bdf3e0` | no |
| `period_1_passing_yds` | 1Q Pass Yards | player_prop | `73299d7b-05b7-40e6-a74e-29ffd9396a05` | no |
| `period_1_receiving_rec` | 1Q Receptions | player_prop | `6af2b921-1ab6-4ddb-a7b6-acc75ffa22a9` | no |
| `period_1_receiving_yds` | 1Q Rec Yards | player_prop | `8a3f2ef8-fd81-4ec4-b071-4d5722d956b1` | no |
| `period_1_rush_rec_tds` | 1Q Rush + Rec TDs | player_prop | `44ba935c-696f-4706-b1fa-0dcc1946a005` | no |
| `period_1_rushing_yds` | 1Q Rush Yards | player_prop | `f56be848-b69a-4822-80fc-433a774e6c91` | no |
| `period_first_touchdown_scored` | First TD Scorer | player_prop | `713f60c4-283b-4444-9660-f421d431add5` | no |
| `receiving_long` | Longest Reception | player_prop | `0f5a7732-8c25-4847-8b9f-1ed9b8acac16` | yes |
| `receiving_rec` | Receptions | player_prop | `b1e29153-4109-461d-937d-66ca0790019b` | yes |
| `receiving_tgts` | Targets | player_prop | `166ff68d-3656-46fd-b80a-c87bc921b0c1` | yes |
| `receiving_yds` | Receiving Yards | player_prop | `8eba18b7-a2c8-47ec-95c6-e682494a51c1` | yes |
| `rush_rec_tds` | Rush + Rec TDs | player_prop | `7b5ec002-905d-4ae7-bf44-d18d2fe1c322` | yes |
| `rush_rec_yds` | Rush + Rec Yards | player_prop | `e112ee2c-0e0b-48d4-8ca5-f960f8d51a3d` | yes |
| `rushing_att` | Rush Attempts | player_prop | `906e8c60-8c5f-4f3c-acba-67ad274ad8a7` | yes |
| `rushing_long` | Longest Rush | player_prop | `bccc919e-d113-4a7e-81e5-db4468d1e9f5` | yes |
| `rushing_yds` | Rush Yards | player_prop | `880784a3-e88f-4bc8-bbb2-ad44fb721a48` | yes |
| `sacks` | Sacks | player_prop | `e0403d79-a3f7-4f15-9130-d20507353d91` | no |
| `tackles_and_assists` | Tackles + Assists | player_prop | `4fb708ae-66a2-4045-9fec-0d02dd179863` | no |

The full slug list for every sport is `state_config.pick_em.visible_stats` (3,348
slugs) on `GET /v1/user/state_configs` **[HAR]**; the legacy feed carried 14 sports'
worth of lines on the same day, so the slug set is far larger than this table.

### 7.6 Team and game markets

The legacy feed and the player pills are `category: player_prop` only. Team and game
markets sit behind the team pills of `market_filters` (§6.4) and in
`match_grouped_lines`, in four more categories **[P-17] [P-21]**. The 29-pill NFL sweep
on the Thursday of week 1 (`limit=1000`, 24 pills non-empty, 5,013 distinct markets,
19.5 MB decoded, 29 requests):

| Pill | Category | Lines | With rungs | `stat` (`display_stat`) |
|---|---|---|---|---|
| Team Picks | `core` | 93 | 32 | `moneyline`, `spread`, `points` (Total Points) for 31 games; `match_grouped_lines` returns the same lines with a grid |
| 1Q / 2Q / 3Q / 4Q / 1H Team Picks | `partial_core` | 45 each | 30 each | `period_N_moneyline` (Moneyline 3-Way, `away` / `draw` / `home`), `period_N_spread`, `period_N_points`; `period_1_2_*` for the first half; 2H Team Picks was empty |
| Team Totals | `team_prop` | 79 | 78 | `team_total_points`, `nfl_team_total_1h` (1H Team Total), `nfl_team_total_yards` |
| Defense (team rows) | `team_prop` | 19 | 18 | `nfl_team_sacks` (single `higher` option) beside 517 player defense props |
| TD Picks | `misc` | 35 | 17 | `nfl_team_total_tds`, `nfl_dst_td` (Will There Be a D/ST Touchdown?, `yes` / `no`) |
| Both Teams to Score | `misc` | 75 | 0 | `nfl_1q_btts`, `nfl_2q_btts`, `nfl_btts_3q`, `nfl_btts_4q`, `nfl_btts_every_quarter` (`yes` / `no`) |
| Overtime | `misc` | 15 | 0 | `overtime_yes_no` |
| Race to X, Game High, Halftime/Fulltime, Winning Margin | — | 0 | — | pills present, no lines on a Thursday; recapture on a game day |

The player pills for scale: 1H Picks 750, Receiving 722, 1Q Picks 675, TD Scorers 605,
Defense 517 players, Each Half 270, Rushing 241, Passing 176, Fantasy Points 140,
Fumbles Lost 109, Each Quarter 97, Kicking 90, Combo Yards 80, Popular 20; Popular
overlaps the stat pills (5,033 rows for 5,013 markets). Every team market keys off a
`Team` or `Match` appearance (§7.2), so the team is `appearances[].team_id` or, for
`Match` lines, `games[].away_team_id` / `home_team_id` with the side in
`options[].choice`. None of the team stats has a `stat_map.json["Underdog"]` entry
**[CODE]**.

`state_configs.pick_em.visible_stats` lists 3,348 stat slugs the account may see across
all sports (`1q_moneyline`, `2h_total_points`, `both_teams_to_score_in_first_quarter`,
win totals, division winners, …) **[HAR]**: the full vocabulary, most of it futures and
other sports; the table above is what NFL priced that day.

## 8. Differences from the legacy feed and what breaks

Probe P1 answered the field-diff question: **`v1/over_under_lines` returns the §7.1
object verbatim** (`status`, `expires_at`, `provider_id`, `live_event`,
`has_alternates`, `american_price`, `decimal_price`, `odds`, `raw_probability`,
`stable_id`), with `players`, `games`, `solo_games`, `appearances` as lists. There is
no field the new surface exposes on a line that the legacy feed lacks; what the feed
lacks are whole markets (team and game, in-play) and the rungs. The differences are
structural:

| Aspect | Legacy `v1/over_under_lines` | Lobby content | `books/underdog.py` |
|---|---|---|---|
| Containers | lists | dicts keyed by id, order in `lines[]` | `_by_id` indexes either style **[CODE]** |
| Scope | all sports, one request | one sport / pill / match / section per request | feed for player props, lobby for team and game markets (§2) |
| Alternates | absent inline; `has_alternates` flags 5,447 lines; rungs come from one `v3/over_unders/<id>/alternate_projections` call per market (§6.2, §9.4) | same flag, same fetch | fetched in-run under a wall-clock budget (§9.4); ladders still have no endpoint |
| Live lines | separate `beta/v2/live_over_under_lines` | `include_live=true` inline, `live_event: true` | `live_event` lines dropped |
| Team and game markets | **absent**: every legacy line is `player_prop` **[P-1] [P-21]** | `core`, `partial_core`, `team_prop`, `misc` behind the team pills (§7.6) | `higher` / `yes` / `home` price the over slot, `lower` / `no` / `away` the under, `draw` leaves the under at 0; null `stat_value` → 0.5; keyed on the team (home team for `Match`) with the `stat` slug as market **[CODE]** |
| Suspended lines | `status: "suspended"` present | present | dropped; a suspended option side prices at 0 |
| `match_id` | int | int | `str()` on both sides |
| Rivals | `beta/v3/rival_lines` answers with empty lists; the product is retired (§6.1) | not seen | gone: the call, the `H2H` parsing, the `rivals` payout table and the pickem variant |
| Stat coverage | 14 sports; NFL period props, kicking, defense, TD scorers, each-quarter / each-half props | same plus the team stats of §7.6 | `INTs Thrown` mapped; the other unmapped NFL `display_stat` values (§7.5) and every team stat archive under their raw names |
| Combo players | still one player per line | — | dropped |

## 9. Efficiency playbook

Requests are the scarce thing; bytes matter less because every JSON body is
brotli-compressed 5–12× on the wire. Each row states the tag its numbers rest on.

### 9.1 Full board once an hour (`prophecize`, 13 runs/day per [OPERATIONS.md](OPERATIONS.md))

| Strategy | Requests per run | Bytes (decoded) | Auth | Notes | Tag |
|---|---|---|---|---|---|
| **Legacy feed + lobby (implemented, §2)** | 1 feed + 1 `match_grouped_lines` per modeled league in play + 8 NFL pills + rungs under a budget: 11 + 2,013 rung reads on 2026-09-11 (MLB and NFL in play) | 32.7 MB + ~0.4 MB per league + ≤2.4 MB + 4–56 KB per rung set | none | the feed alone returns every player prop across 14 sports with the full §7.1 object; the core call per league adds moneyline / spread / total and the `teams` dict that names the feed's teams; the eight pills add the team totals, TD and period markets; `/v1/teams` (11.3 MB, yearly) and the dead `rival_lines` call (§6.1) are gone | [P-1] [P-21] [P-44] |
| Lobby `lines` per market pill | 29 for NFL (one per `market_filters` pill, `limit=1000`) | 19.5 MB for 5,013 NFL markets (≈3.9 KB per line) | none with the pill ids pinned from §6.4 (`market_filters` itself needs a token) | measured: every legacy NFL market plus the 676 team, game and in-play markets the feed lacks (§7.6); pills overlap by 20 rows; no single-request sport sweep exists (`limit` needs a pill); 14 active sports ≈ 150–300 requests | [P-15] [P-21] [P-33] |
| **Team pills only** (Team Picks, Team Totals, 1Q–4Q / 1H / 2H Team Picks, TD Picks, Both Teams to Score, Overtime, Race to X, Game High, Halftime/Fulltime, Winning Margin) | 15 for NFL (10 non-empty on a Thursday) | 2.4 MB for 522 markets | none | the cheapest way to add the team and game markets on top of the legacy feed; the 19 team-sacks rows ride inside the Defense pill; `match_grouped_lines` alone (1 request, 429 KB) covers just moneyline / spread / total | [P-17] [P-21] |
| Alternate rungs for the whole board | 3,188 flagged, modeled, mapped player props on 2026-09-11 (one `alternate_projections` each) | 4–56 KB each | none | ~2,000 fit the 120 s budget at three workers; the rest, the latest kickoffs, wait for a later hourly run (§9.4) | [P-23] [P-44] |
| `pickem_lobby_sections` via `scaffolds/home` | 1 + ~40 sections | 550 KB per section | none (one 401 seen, §4) | curated rails, not the whole board | [HAR] [P-4] [P-30] |
| `search_results?sport_id=X` | 1 per sport | 240 KB | none | capped near 100 lines, options lack `odds` | [HAR] [P-4] |

The scraper (§2) keeps `v1/over_under_lines` as the primary feed for player props (it
is public, complete for that category, and the web app's own slip flow still depends
on it), adds `match_grouped_lines` per modeled league and the eight NFL team pills
that carry the team and game markets it wants, and fetches rungs in-run under a
budget (§9.4); an empty feed logs at WARNING. Still open: `If-None-Match` so an
unchanged board costs a 304, the other sports' team pills (their ids need a tokened
`market_filters` read, §6.4). The full pill sweep is the fallback if the feed retires,
priced above.

### 9.2 Line movement inside the hour

| Need | Cheapest path | Floor | Tag |
|---|---|---|---|
| Whole board every N minutes | poll the legacy feed with `If-None-Match` | 10 s CDN window; 304 costs ~0 bytes | [P-9] |
| One sport or one market | `lines?sport_id=…&filter_id=…&limit=1000` | 30 s window | [P-5] [P-18] |
| A watch-list of markets | `v1/over_under_lines?over_under_ids=<id>,…` returns just those lines with their entities (10 ids → 30 KB) | 10 s window | [P-13] [P-14] |
| One market's rungs | `v3/over_unders/<id>/alternate_projections` (§9.4) | 10 s window | [P-12] |
| One market's price path (prediction markets) | `market_price_history` gives 68 timepoints back to July in one request; `refresh_url` polls at 5 s | 30 s window, `stale-while-revalidate=300` | [P-6] |
| Push instead of polling | Pusher public channels `over_under_lines-<SPORT>-balanced` and the `lines-balanced;Game-<id>;PickemStat-<uuid>;…` subscriptions | needs the cluster (§6.5) and a websocket client; not in the repo today | [P-8] open |
| In-play lines | `beta/v2/live_over_under_lines` (270 lines, one request) or `include_live=true` on lobby content | 30 s | [P-11] |

### 9.3 Reference data (daily)

| Endpoint | Why | Tag |
|---|---|---|
| `stats…/v2/sports` (92 KB) | `pickem_status` tells a poller which of the 141 sport ids have a board today, so it never requests dead sports; `ladder_status` flags the two ladder sports | [P-4] |
| `stats…/v1/teams` (11 MB) | abbreviations; 900 s TTL, ETag | [HAR] |
| `stats…/v1/lineup_statuses` (1.4 KB) | decode `appearances[].lineup_status_id` | [P-9] |
| `app…/client-version` (34 B) | only if a client ever sends `Client-Version`; public reads do not need it | [P-3] |

Payout tables are not a reference download; `entry_slips/estimate` quotes them per slip
with a token (§6.8), and the quotes disagree with the static `underdog_payouts.json` on
most cells. The file stays the offline table (canonical:
[hygiene-closeout.md](handoffs/hygiene-closeout.md)) until a tokened refresh rewrites
it from quotes.

### 9.4 Alternate rungs

Rungs are never inline. The app's alternates drawer calls
`GET /v3/over_unders/<over_under_id>/alternate_projections` (bundle
`sourceEndpoint: "GET v3/over_unders/:overUnderId/alternate_projections"`, client-side
re-fetch after 10 s); the endpoint is public, one market per call, with no batch form
(`?over_under_ids=` → 404, comma-joined ids in the path → empty) **[P-8] [P-12]
[P-22]**. `projections[]` is the balanced line (`is_main: true`, `stable_id`
`…|balanced`) plus every rung (`stable_id` `…|alternate`), sorted by `stat_value`,
each with its own line `id` and the full §7.1 options **[P-19] [P-23]**. Measured on
2026-09-10:

| Market | Rungs (incl. main) | Range | Bytes |
|---|---|---|---|
| NFL spread | 25 | −20.5 … +14.5, both sides priced (0.52× … 6.7×) | 56 KB |
| Pass yards | 16 | 189.5 … 339.5 | 25 KB |
| Team total points | 14 | 3.5 … 42.5 | 26 KB |
| Receiving yards | 12 | 49.5 … 159.5 | 21 KB |
| Rush yards | 11 | 54.5 … 149.5 (`higher` 0.66× … 7.54×) | 18 KB |
| Rush + Rec TDs | 3 | 0.5 / 1.5 / 2.5 at 1.0× / 2.57× / 11.56×, `higher` only | 4 KB |
| Fantasy points | 3 | main two-sided, rungs `higher` only | 4 KB |

Cost of a full catalog: 2,928 NFL markets carried `has_alternates` in the sweep (2,633
player props, 150 period team lines, 96 team props, 32 core, 17 misc) **[P-21]**, so
one pass is ~2,900 requests and, at the sizes above, roughly 50 MB decoded; at the
1.5 s spacing used here that is over an hour. Measured on 2026-09-11 from one
keep-alive session **[P-44]**: three workers read 2,013 markets in 120 s (~17/s) and
every answer carried rungs; four workers ran at ~36/s and, after about 1,850 requests
inside a minute, the origin kept answering 200 but with empty `projections` for the
rest of the pass (no 429, no rate-limit header, `cf-cache-status` MISS / EXPIRED),
while an ordered 1,200-request pass at ~23/s stayed clean. The scraper therefore runs
three workers and stops a pass after 25 consecutive empty answers
(`UD_ALT_LINES_EMPTY_STREAK`). The cheapest correct scheme:

1. Take `over_under_id` and `has_alternates` from the legacy feed (player props) and
   the team pills (team and game markets); the id is stable across moves (§7.3), so it
   is the cache key.
2. Fetch rungs only for markets the model will act on: the candidate list after
   scoring, or the stat families where the ladder carries the edge (spreads, totals,
   yardage). TD and fantasy-point markets have three rungs and rarely need a fetch.
3. Re-fetch a market's rungs when its balanced line moves (`stat_value` or
   `options[].updated_at` changed in the cheap feed), not on a timer; the CDN serves the
   same body for 10 s anyway.
4. For a watch-list, `v1/over_under_lines?over_under_ids=` refreshes the balanced
   lines of many markets in one request (§9.2) and decides which rung sets to re-read.
5. Push: every `alternate_projections` response names a
   `lines-alternate;Game-<id>;PickemStat-<uuid>;<provider>` channel beside the
   `lines-balanced;…` one, so once the Pusher cluster is known (§11) rung updates can
   arrive without polling.

As implemented (§2): the scraper fetches in-run, every modeled and `stat_map`-mapped
player prop flagged `has_alternates`, ordered by kickoff and then by line descending,
three workers on one session, a 120 s budget and 10 s per request; on 2026-09-11 that
returned 2,378 NFL and 4,127 MLB rungs **[P-44]**. Items 3–5 (move-triggered
re-reads, watch-lists, push) are still open.

## 10. Levers for edge, ranked

Each lever names where the data sits and the gate before it can be trusted.

1. **Per-side prices and implied probabilities.** `payout_multiplier` is already
   parsed; `american_price`, `decimal_price` and `odds.fantasy.probability` are on
   every legacy line **[P-1]**. Today `prediction/cli.py` divides a flat
   `UNDERDOG_BOOST_BASELINE` back out of the boost
   ([prediction/cli.py:260-266](../src/sportstradamus/prediction/cli.py#L260-L266))
   **[CODE]**; the explicit per-side price could replace the flat assumption. Gate:
   confirm which field settles an entry (§7.4) with one placed-and-settled pick.
2. **Alternate lines as real ladder rungs.** `has_alternates` is true on 5,447 legacy
   lines and `v3/over_unders/<id>/alternate_projections` returns the whole ladder
   with both sides priced (§9.4) **[P-19] [P-23]**. Gate: request volume (§9.4) and
   the price-field question of lever 1; the fetch itself is settled.
3. **Team and game markets.** 541 NFL markets the legacy feed never carried: team
   totals (points, first half, yards, sacks, touchdowns), period moneylines, spreads
   and totals, both-teams-to-score, overtime (§7.6), most with ladders. Gate:
   `stat_map` entries and a team-level model
   ([dfs-products.md](handoffs/dfs-products.md)).
4. **Slip quotes.** `entry_slips/estimate` returns the Power multiplier, the Flex
   table and the correlation-adjusted multiplier for any candidate slip (same-team
   same-direction QB-to-receiver pairs are taxed, opposite sides and other pairs are
   not, §6.8), and `entry_slip_limits` the stake caps (§6.7) **[P-31] [P-32] [P-38]**.
   It is the oracle for `underdog_payouts.json` (off on most cells today), and the Flex
   modifier it applies is reproducible offline from the ρ table in §6.8. Gate: a
   token, ten minutes at a time, so a manual refresh session rather than cron.
5. **`status`, `expires_at`, `lineup_status_id`.** All on the legacy feed **[P-1]**;
   filter `suspended` lines and decode injury status with `lineup_statuses`. Gate:
   none, parser change only.
6. **Prediction-market contracts.** `provider_id` udx / kalshi / nadex,
   `raw_probability`, `odds.prediction`, `contract_url` on core and team lines, and the
   estimate's `predictions_quote` (unit cost and fees per leg, §6.8); display-only
   per [dfs-products.md](handoffs/dfs-products.md) (CFTC surface). Gate: the owner's
   ToS call; `raw_probability` semantics.
7. **Price history without self-archiving.** `market_price_history` and the
   `/v1/over_unders/<id>/chart_data` family return the full price path of a market in
   one request **[P-6]**; only the match's ranked market was probed. Gate: whether
   `chart_type` / `target_type` accept a player-prop `over_under_id`.
8. **Push.** Pusher channels replace polling once the cluster is known **[P-8]**.
9. **Live lines.** `beta/v2/live_over_under_lines` + `live_event_stat` give the
   running stat next to the live line **[P-11]**. Gate: a live-market model.
10. **Popularity and movement stamps.** `entry_count` per line and per match,
    `updated_at` per line and per option **[HAR]**; cheap features for a
    line-movement archive.
11. **Context.** `game.pre_game_data.weather` / `venue` / `broadcasts`, `scoreboard`
    **[HAR]**; low value, already available elsewhere.
12. **Boosts and power-ups.** `over_under.boost` (null in the capture),
    `/v1/user/power_ups` and the estimate's `power_up_id` are account-scoped promos;
    read-only, never automated.

## 11. Open questions and probe log

248 requests on 2026-09-10 from a bare `requests` client, 1–1.5 s apart, desktop
User-Agent. P1–P30 and P34 ran without a token (the HAR token had expired); P31–P33
used a fresh token the owner pasted into `creds/keys.json` at 21:41 CDT (20 requests
in five minutes; an earlier paste had already expired, P24), P35–P38 a third token
later that evening (24 requests in four minutes) and P39–P41 a fourth at 22:02 (71
requests in eight minutes). P12–P23 followed once the bundle gave up the alternates
path. P42–P44 ran on 2026-09-11 from the scraper's own client (about 6,500 requests
across two full runs and two pace probes).

| # | Question | Probe | Result |
|---|---|---|---|
| P1 | Does the legacy feed still answer, and does it carry the new fields and alternates? | `v1/over_under_lines` bare and with `Client-*`; `beta/v3/rival_lines` | 200 both ways, 12,272 lines, full §7.1 object, one line per (appearance, stat) so **no alternates inline**; rivals 200 but empty |
| P2 | Is 426 a `Client-Version` gate? | `beta/v6/over_under_lines` bare, then with `Client-Type: web` + current `Client-Version` | 426 `upgrade_required` both times: retired |
| P3 | Is lobby content public? | `content/lines?sport_id=NFL` with nothing, `Client-*`, and the dead JWT | 200, 200, 200 (CDN HIT on the repeats) |
| P4 | Per-endpoint auth; are the ubiquitous params required? | `search_results`, `pickem_lobby_sections`, `match_grouped_lines`, `scaffolds/sports`, `stats/v2/sports`; `lines` without the trio | 200, **401**, 200, 200, 200; `lines` without the trio → 400 "state_config_id parameter is required" |
| P5 | Pagination and the cost of a by-pill sweep | Rushing pill `limit=200`, `limit=3`, `limit=3&page=2`; Fantasy Points pill `limit=200` | 200 lines / 3 / same 3 (page ignored) / 150 (all of them); P18 settled the cap |
| P6 | Price-history shape | `market_price_history?chart_type=ranked_market&target_id=177184&target_type=Game` | 200 public; §6.6 |
| P7 | Live, futures and player scaffolds | `scaffolds/live_matches`, `scaffolds/futures?sport_id=NFL`, `scaffolds/players?player_id=…&tab_id=picks` | 200 all; §6.3; player page points at `lines_with_stats?appearance_id=` |
| P8 | Pusher cluster; endpoint paths the app knows | `app.underdogsports.com/`, `runtime`, `critical`, `main`, `page.pick-em`, `entry.app` bundles | cluster **not found**; `entry.app` maps `regular` → `/v1/over_under_lines`, `live` → `/beta/v2/live_over_under_lines`, names the alternates call (`GET v3/over_unders/:overUnderId/alternate_projections`), the `lines_with_stats` adapter and the scaffold path map (`home`, `matches`, `players`, `rewards`, `sport_groups`, `sports`) |
| P9 | Conditional GET from a non-browser client | `stats/v1/lineup_statuses` twice, second with `If-None-Match` | 200 then 304, empty body, `cf-cache-status: HIT` |
| P10 | Payout / ladder exposure in captured bodies | grep `state_configs`, `features`, `user` for `payout`, `multiplier`, `ladder` | flags only (`pickem_ladders_enabled`, `payout_modifiers`); no tables |
| P11 | The live endpoint the bundle names | `beta/v2/live_over_under_lines` bare | 200, 270 live MLB lines, legacy shape, `max-age=30` |
| P12 | The alternates path the bundle names | `v3/over_unders/<id>/alternate_projections` for a rush-yards market from the 19:27 lobby body, then a fresh Rush + Rec TDs market, with and without the ubiquitous trio | 200 public, `max-age=10`; empty `projections` for the first (its game had kicked off), 3 rungs for the second; the trio is optional |
| P13 | Is `over_under_id` stable, and does the feed filter by id? | `v1/over_under_lines?over_under_ids=<id>` with stale, then fresh ids; eight markets compared 2.5 h apart | filter honored (0 lines for the stale ids, 1 for the fresh one, 4.6 KB); ids unchanged across line moves |
| P14 | Batch forms | `v3/over_unders/alternate_projections?over_under_ids=`, `v3/over_unders/<id>`, `over_under_ids=` × 10, `line_type=alternate` on the feed | 404, 404, 10 lines in 30 KB, `line_type` ignored |
| P15 | Do lobby params unlock more per request? | `content/lines?sport_id=NFL&limit=200` with `market_view=expanded` / `full`, `show_mass_option_markets=true` | 20 lines every time: `limit` needs a `filter_id` |
| P16 | Team props on a match page | `content/lines?match_id=178861&match_type=Game&limit=200` | 20 player props |
| P17 | Team pills | Team Picks, Team Totals, Winning Margin at `limit=200` | 93 `core`, 79 `team_prop`, 0 (§7.6) |
| P18 | `limit` cap | Rushing pill `limit=1000` | all 241 lines (200 had been a cap) |
| P19 | Rungs on a team market | `alternate_projections` for the SF @ LAR spread | 25 rungs, −20.5 … +14.5, 56 KB |
| P20 | `market_categories[]` on `match_grouped_lines` | omitted | same 90 core lines |
| P21 | The whole NFL board on the new surface | all 29 pills at `limit=1000`, compared with a fresh legacy pull by `over_under_id` | 5,013 markets ⊇ the feed's 4,337; 676 lobby-only (541 team and game, 135 in-play); five categories; appearance types `Player` / `Team` / `Match` |
| P22 | Comma-joined ids in the alternates path | `v3/over_unders/<id>,<id>/alternate_projections` | 200 with empty `projections`: ignored |
| P23 | Rung counts by stat family | `alternate_projections` for rush, receiving and pass yards, team total points, fantasy points | 11 / 12 / 16 / 14 / 3 rungs (§9.4) |
| P24 | Does an expired token hurt? | `pickem_lobby_sections` with the first pasted token (27 min old) | 401 `Invalid email or password.` |
| P25 | Auth and caching of scaffolds and navigation | `scaffolds/home`, `scaffolds/matches`, `market_filters`, `match_details`, `sport_navigation`, `match_navigation` bare | 200 (`private`), 200 (90 s), **400**, 200 (90 s), 200, 200 |
| P26 | `lines_with_stats` shape | `?appearance_id=<Player appearance>` bare | 200: 2 lines, empty `stat_values_by_over_under_id` |
| P27 | Promo and feature endpoints | `sport_promo_carousel`, `promo_carousels/main`, `/v1/features`, `/v3/user/features`, `/v1/user/state_configs` bare | 200, 200, 200, 200, **422** `location_needed` |
| P28 | The 400 and 422 bodies | `market_filters` with the HAR's exact query; `state_configs` | "Unable to construct the scaffold for market filter navigation."; "We could not detect your location." |
| P29 | Home rails without a token | `fantasy_lineup_cards`, `lines_with_stats?target_type=hot_hands`, `pickem_lobby_sections` ("MLB Games"), `packs` | 401, 200, 200, 401 |
| P30 | Is the section 401 per section? | the HAR's NFL section and an ATP Tennis section bare; Popular Picks (`data_driven_line_source_id`) bare | 200 (was 401 at 19:27), 200, 200 (90 s) |
| P31 | Authenticated reach | `/v1/user`, `pickem_lobby_sections`, `fantasy_lineup_cards`, `packs`, `state_configs` (no geo), `rival_lines`, three ladders paths, `entry_slip_limits`, `power_ups`, all with a fresh token | 200, 200 (124 lines), 200, 200 (empty), 422, 200 (empty), 404 × 3, 200, 200 |
| P32 | Slip pricing | `entry_slips/estimate` for 2–6 even picks, same-team QB + WR with and without a filler, the same player twice, a rung, a team total, a moneyline contract | §6.8 |
| P33 | `market_filters` with a token | sport target with the probe's and the HAR's exact query; game target bare | 200 (32 pills: the 29 captured + 2Q, 3Q, 2H Picks) both; game target bare 400 |
| P34 | Can a bare client ask by stat without a pill? | `lines?sport_id=NFL&filter_type=PickemStat&filter_id=<rushing_yds pickem_stat_id>&limit=1000` | 200: 81 lines, all `rushing_yds`, 408 KB |
| P35 | Tables past 6 picks; priced picks | `entry_slips/estimate` for 7 and 8 even picks; 3 picks at 0.87 / 1.16 / 0.74× | Power 65 / 120, Flex 40 / 2.75 / 0.5 and 80 / 3 / 1; 4.85 and 2.42 / 1.10 (§6.8) |
| P36 | `state_configs` geo headers | token + `User-Latitude` / `User-Longitude`, no Radar token | 200, 79 KB |
| P37 | `market_filters` for a game with a token | `target_id=<Sunday game>&target_type=Game` | 200, 28 pills |
| P38 | Which pairs are taxed | eight 3-pick quotes: same-team QB + WR on each side combination, QB + WR receptions, QB + WR TDs, opposing QB + QB, opposing QB + WR, same-team QB + RB, WR + WR, opposing RB + RB; `fee=100`; `fee_source=bonus` | §6.8 table; fee and source change nothing |
| P39 | Modifier structure | 57 quotes: one pair with three fillers and at 3–6 picks, four side combinations, QB × each of 16 receivers on three teams, seven rungs, six stat pairs, twelve position pairs, QB + WR1 + WR2, two pairs in one slip, a repeat; two ladders | §6.8 "How the number arises" |
| P40 | Lower tail and combined rungs | both-under for four more pairs, two lower-side rungs, a QB rung with a WR rung, fantasy points both-under; two ladders re-read | Power asymmetric per pair, Flex symmetric; one rung suspended mid-run |
| P41 | Cross-team leg inside a taxed pair | QB + WR1 + opposing QB (+ filler); QB + opposing QB with two fillers | 0.80 / 0.789 against 0.7925 / 0.76 and 1.0 / 0.90: one joint over the game, not a product |
| P42 | Title order against the ids | `abbreviated_title` vs `home_team_id` / `away_team_id` for every game in the feed and the lobby `teams` dicts | US sports `AWAY @ HOME`; soccer `HOME vs AWAY` (FIFA 69/69); esports `AWAY vs HOME` (CS 42/42, LOL 24/24 swapped); one CFL title contradicts its own ids; the scraper seeds abbreviations from the lobby `teams` dicts and falls back to titles (§7.3) |
| P43 | Do the lobby reads answer the scraper's bare client? | `match_grouped_lines?sport_id=MLB` and an NFL pill with the python-requests default User-Agent | 200 both (60 core lines / 30 games / 30 teams; the full pill page), so `Scrape.get`'s header-less first attempt suffices |
| P44 | Full run and rung pace | `get_ud` end to end at four and at three workers, a 40-request burst, a 1,200-request ordered pass | four workers: 3,156 rung reads in 88 s (~36/s), 200 with empty `projections` after ~1,850 in a minute; 1,200 at ~23/s clean; three workers: 2,013 in 120 s (~17/s) clean, 21,066 offers (6,533 rungs) in 130 s wall (§9.4) |

Still open, in priority order:

| Question | Cheapest next step |
|---|---|
| Pusher cluster and whether the line channels are public | find the Pusher vendor chunk (search the `critical` bundle's chunk loader for `pusher`), then open `wss://ws-<cluster>.pusher.com/app/d65207c183930ff953dc?protocol=7` and subscribe to `over_under_lines-NFL-balanced` and a `lines-alternate;…` channel |
| Whether a rung re-price moves the balanced line's `updated_at` (the trigger in §9.4) | poll one market's feed row and its rungs through a line move |
| Where the `alternate_projections` ceiling sits: 200 with empty `projections` at ~36/s after ~1,850 reads in a minute, clean at ~23/s, never a 429 **[P-44]** | a four-worker pass that pauses 2 s every 500 reads, watching when the empties start and how long they last |
| The pills empty on a Thursday (Race to X, Game High, Halftime/Fulltime, Winning Margin, 2H Team Picks) and the other 13 active sports' pills | rerun the pill sweep on a Sunday morning and per sport |
| Ladders (pinned by the owner) | a proxy capture from the iOS or Android app (§6.9); nothing on the web surface answers |
| `market_filters` without a token | the ids in §6.4 come from tokened reads; re-read them with a token each season (three pills were new by 21:52) or ask by `pickem_stat_id` |
| The section 401 seen once at 19:27 | repeat the bare section read across a game day; keep the token fallback until it never recurs |
| Power's joint input (shared-game history or a simulation) and whether the Flex ρ table changes by sport or season | quote the same pairs after week 1 (history grows) and on another sport's board |
| Whether `entry_slips/estimate` ever answers `pending` (the polling path the bundle implements) | not seen in 100-odd quotes; watch for it under load |
| Which price field settles; `odds.fantasy.experiments` | one settled entry |
| Whether `market_price_history` accepts player-prop markets | `chart_data?metric=option_normalized_probability` on a player-prop `over_under_id` |

## 12. Recapturing a HAR and maintenance

1. Log in at `app.underdogsports.com` in Firefox, open DevTools → Network, tick
   "Persist Logs", and browse the surface you need. To cover what this capture missed:
   open a sport lobby and a match page on a game day. **Ladders** need the mobile app
   (§6.9). Rivals no longer exist. The slip builder's quotes are reproducible with a
   token (§6.8), so no capture is needed for payout tables; never submit an entry.
2. Network → gear icon → **Save All As HAR**, once per session. Firefox's right-click
   **Copy All As HAR** also exports the whole log regardless of which request is
   selected, so per-request copies are identical duplicates (the 2026-09-10 set was
   eight copies of one 135-entry session that differed only in the page id). Large
   bodies are not safe in a HAR (the 11 MB `/v1/teams` body came back corrupted in
   this capture), so keep a note of which requests matter and re-fetch big ones
   directly.
3. Store it under `new_ud_api/` (gitignored). A HAR holds the live access token, the
   Radar token, the Pusher auth signature, the account email and user id: never commit
   one, never paste one into docs or tests, and delete it when the doc is updated.
4. Update this document in place (one canonical home per fact,
   [STYLE_GUIDE.md §16](STYLE_GUIDE.md)): change the capture line at the top, refresh
   the counts, and move settled rows out of §11.

Retirement watch: the feed's `cache-control` header and body shape are the canary. A
426 with `api_code: upgrade_required` means the path is gone; the replacement is
whatever the current `entry.app` chunk maps under `regular` (§2).

## Changelog

- 2026-09-11 scraper rebuilt on this doc (`books/underdog.py`): feed + core + NFL team pills + budgeted rungs (three workers, 120 s, empty-streak stop; the origin answers empty past ~36/s, §9.4); `/v1/teams` and `rival_lines` gone; team codes from lobby `teams` dicts (esports titles list the away side first, §7.3)
- 2026-09-10 modifier mapped: slip = table × Π picks × m; Flex m = Gaussian copula, ρ 0.48 per pair type; Power m pair-specific + asymmetric; untaxed pairs = free correlation. Rivals retired, Ladders replaced them (owner)
- 2026-09-10 authed batch: `entry_slips/estimate` quotes payout tables (2–8 picks) + which pairs are taxed; scaffolds/sections public, `market_filters` needs token; `state_configs` needs lat/long; ladders not on web
- 2026-09-10 alternates path found (`v3/over_unders/<id>/alternate_projections`); team/game markets catalogued (5 categories, 29-pill sweep); feed filters by `over_under_ids`; `limit` needs a pill
- 2026-09-10 first version from HAR + 39 probes; legacy feed = full new schema, no alternates; lobby lines public, sections 401
