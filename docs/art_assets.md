# Art asset catalog

Every place the dashboard renders art — real or a stand-in — as one table. A slot with no
file today renders an honest token-gradient or generated-SVG scar (DESIGN.md §3); nothing
here changes what a user sees until a file lands in the slot. The loader that renders a
present file is `dashboard/assets.py`; ambient slot state lives in
`data/assets/ambient/ambient_manifest.json`. The licence column is the owner's own checklist
— nothing in code enforces it.

| slot | surface / component | today | want | source | licence requirement | format + size | priority |
|---|---|---|---|---|---|---|---|
| `logo_wordmark` | Sidebar via `st.logo()` (`dashboard/app.py`) | none — plain nav, no wordmark | Guru/genie mystic-sports mark + "SPORTSTRADAMUS" wordmark, horizontal lockup | **commissioned** ([`art_briefs/logo_guru.md`](art_briefs/logo_guru.md)) | full exclusive commercial rights assigned; artist credited in this row once delivered | SVG master + PNG @1x/@2x, ~600×160 | P1 |
| `logo_mark` | Collapsed-sidebar icon, `st.logo(icon_image=...)` (`dashboard/app.py`) | none | Square mark alone; also the eventual favicon source, reads at 16px | **commissioned** (same brief) | same rights as `logo_wordmark`; artist credited in this row once delivered | SVG master + PNG @1x/@2x, 512×512 | P1 |
| `favicon` | Browser tab (`st.set_page_config(page_icon=...)`, `dashboard/app.py`) | Streamlit default icon | Logo-derived mark | generated now (hand-drawn ◈ comet SVG) → replaced by `logo_mark` later, zero code change | hand-drawn vector is licence-exempt (generated-SVG-keep) | SVG (64×64 PNG fallback if Streamlit won't serve SVG), legible at 16×16 | P1 |
| `ambient_tonight` | Tonight card deck — `.tonight-card` background (`dashboard/theme.py`) | `night_sky.jpg` at 0.14, scaled to the column and tiled down it; each card shows the next slice (`surfaces/tonight.py` script) | — | free-stock (pickpik) | owner-checked; `source_url` in `ambient_manifest.json` | 6016×4016 JPEG source, downscaled to 1600 px WebP at import | done |
| `ambient_receipts_hero` | Receipts verdict hero — `_HERO_BG` (`dashboard/surfaces/receipts.py`) | `nebula.jpg` (Horsehead, Wikimedia Commons) at 0.16, cropped to cover | — | free-stock | owner-checked; `source_url` in the manifest | 7300×4867 JPEG source (18 MB in git), downscaled to 1600 px WebP at import | done |
| `ambient_games_hero` | Games hero card — `_HERO_BG` (`dashboard/surfaces/games.py`) | the same `nebula.jpg` as the Receipts hero, by owner request | — | same | same | same | done |
| `ambient_gutters` (starfield) | App-level starfield — `theme._starfield_background()` + `STARFIELD_HTML` | Generated CSS dust dots + twinkle divs, seeded PRNG | keep — generated is the design | generated-SVG-keep | n/a | n/a | P3 |
| `page_hero_wash` | Every surface's `.page-hero` header band (`dashboard/theme.py`) | Generated CSS two-stop nebula wash | keep — generated is the design | generated-SVG-keep | n/a | n/a | P3 |
| `glyphs_game_shape` | Tonight/Games game-shape glyphs (`dashboard/components/glyphs.py`) | Generated inline SVG (comet, supernova, scales, nebula, hourglass) | keep; optional licensed-texture upgrade later | generated-SVG-keep | n/a | n/a | P3 |
| `astrolabe_engraving` | Slip-builder astrolabe bezel (`dashboard/components/astrolabe_component/build/index.html`) | Generated SVG bezel/orbitals/dials | keep; optional licensed engraving texture | generated-SVG-keep | n/a | n/a | P3 |
| `constellation_silhouettes` | Games constellation decoration layer (`data/config/constellation_shapes.json`, rendered by `constellation_component`) | Generated SVG template paths (100-template bank) | keep; optional artist pass | catalog-only here — Phase D owns these files | n/a | n/a | P3 |
| `team_marks` | Anywhere a team renders — constellation star fills, badges — via `theme.team_colors()` | Hex colors + full names only (`data/config/team_assets.json`); no logo art | none — **skipped** by the owner; colors carry the constellation grammar (DESIGN §4a) | n/a | n/a | n/a | skipped |
| `player_headshots` | Constellation ticket-card headshot disc — `.cst-shot` (`dashboard/components/constellation_component/build/main.js:350`, title "Player headshot — coming soon") | Initials-disc scar (`initials()` fallback) | League-CDN player headshots | **wanted** by the owner — next lane | owner verifies each league CDN's terms; disk-cache locally rather than hot-link | circular crop, ~34×34 render (`.cst-shot`) | next lane |

The `player_headshots` scar lives in the constellation component's ticket card, not in
`components/deep_dive.py` — there is no separate person-icon stand-in on the offer-detail
dialog or the Board's mobile cards (`components/offer_cards.py`); the sweep for
`:material/person:`-style icons and other `avatar`/`disc` stand-ins turned up nothing else.
The `radial-gradient` sweep found exactly the four hero/card washes above (`theme.py`
twice, `games.py`, `receipts.py`) — no other surface builds one.

## The scar mechanism

`ambient_tonight`, `ambient_receipts_hero` and `ambient_games_hero` are wired through
`dashboard.assets.ambient_css(slot, fallback_gradient)`: it returns `fallback_gradient`
byte-identical unless the manifest slot names a `file` that exists on disk. A present file
is embedded as a base64 data URI (Streamlit serves no arbitrary static files) under a solid
overlay of the surface color, so it never exceeds its manifest `opacity`; anything wider
than 1600 px is downscaled to WebP at import, so the page never ships an original. The
slot's `placement` picks the geometry: `hero-background` crops to cover the box;
`card-background` scales the image to the column width and tiles it downward, and the
Tonight page's script offsets each card by the cards above it, so the column reads as one
sky sliced card by card, looping once it runs out of image. Every other row above is
either `generated-SVG-keep` (no sourcing needed — the generated form *is* the design) or
`catalog-only` (owned by another phase).

## What the owner does next

**Swapping an ambient file** — no code:

1. Check the licence yourself — skip editorial-only, NC, or unclear terms; nothing in code
   checks it. Mostly-dark images work best: the Tonight deck renders at 0.14 under body
   text, the heroes at 0.16.
2. Drop the file into `src/sportstradamus/data/assets/ambient/` (JPEG, PNG or WebP, any
   size — the loader downscales anything wider than 1600 px at import, but git keeps every
   committed version forever, so a web-sized file is kinder to the repo).
3. Point that slot's `"file"` at it in `ambient_manifest.json`, plus `attribution` /
   `source_url` if you want the note. Leave `opacity` alone: 0.14 / 0.16 are tuned, and the
   loader refuses anything above 0.20.
4. Restart the dashboard service — the callers read the manifest at import.

**Team marks**: skipped by the owner — colors-only stays; the constellation grammar never
needed logos.

**Player headshots**: the owner wants them — a lane of its own, not a file drop:
per-league CDN URL patterns, a disk cache, and the `main.js` "coming soon" disc replaced.

**Commissioned logo**: the brief is [`art_briefs/logo_guru.md`](art_briefs/logo_guru.md).
`app.py` already carries the reserved, existence-guarded slot — dropping
`dashboard/static/logo_wordmark.png` and `dashboard/static/logo_mark.png` in place renders
the logo with zero code change; neither file is committed today. Logos aren't part of
`ambient_manifest.json`'s schema (that manifest is ambient-image only), so this table is
their attribution record instead: once the commission is delivered, note the artist's name
and the rights grant directly in the `logo_wordmark`/`logo_mark` rows above.
