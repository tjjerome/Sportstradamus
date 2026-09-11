# Art asset catalog

Every place the dashboard renders art — real or a stand-in — as one table. A slot with no
file today renders an honest token-gradient or generated-SVG scar (DESIGN.md §3); nothing
here changes what a user sees until a file lands in the slot. The license-gated loader that
enforces this is `dashboard/assets.py`; ambient slot state lives in
`data/assets/ambient/ambient_manifest.json`.

| slot | surface / component | today | want | source | licence requirement | format + size | priority |
|---|---|---|---|---|---|---|---|
| `logo_wordmark` | Sidebar via `st.logo()` (`dashboard/app.py`) | none — plain nav, no wordmark | Guru/genie mystic-sports mark + "SPORTSTRADAMUS" wordmark, horizontal lockup | **commissioned** ([`art_briefs/logo_guru.md`](art_briefs/logo_guru.md)) | full exclusive commercial rights assigned; artist credited in this row once delivered | SVG master + PNG @1x/@2x, ~600×160 | P1 |
| `logo_mark` | Collapsed-sidebar icon, `st.logo(icon_image=...)` (`dashboard/app.py`) | none | Square mark alone; also the eventual favicon source, reads at 16px | **commissioned** (same brief) | same rights as `logo_wordmark`; artist credited in this row once delivered | SVG master + PNG @1x/@2x, 512×512 | P1 |
| `favicon` | Browser tab (`st.set_page_config(page_icon=...)`, `dashboard/app.py`) | Streamlit default icon | Logo-derived mark | generated now (hand-drawn ◈ comet SVG) → replaced by `logo_mark` later, zero code change | hand-drawn vector is licence-exempt (generated-SVG-keep) | SVG (64×64 PNG fallback if Streamlit won't serve SVG), legible at 16×16 | P1 |
| `ambient_tonight` | Tonight card wash — `.tonight-card` background (`dashboard/theme.py`) | CSS radial-gradient nebula wash | Faint night-sky field behind the card | free-stock (NASA/ESA or Unsplash/Pexels class) | commercial-clean licence recorded in `ambient_manifest.json` (`license`/`attribution`/`source_url`) | JPEG/WebP < 300 KB; manifest opacity 0.14 | P2 |
| `ambient_receipts_hero` | Receipts verdict hero — `_HERO_BG` (`dashboard/surfaces/receipts.py`) | CSS radial-gradient nebula wash | Licensed nebula/night-sky texture | free-stock (NASA/ESA class) | same as `ambient_tonight` | JPEG/WebP < 300 KB; manifest opacity 0.16 | P2 |
| `ambient_gutters` (starfield) | App-level starfield — `theme._starfield_background()` + `STARFIELD_HTML` | Generated CSS dust dots + twinkle divs, seeded PRNG | keep — generated is the design | generated-SVG-keep | n/a | n/a | P3 |
| `page_hero_wash` | Every surface's `.page-hero` header band (`dashboard/theme.py`) | Generated CSS two-stop nebula wash | keep — generated is the design | generated-SVG-keep | n/a | n/a | P3 |
| `games_hero_wash` | Games hero card — `_HERO_BG` (`dashboard/surfaces/games.py`) | CSS radial-gradient nebula wash | Same ambient upgrade as the other two hero washes | catalog-only here — `games.py` is Phase D territory, not wired by the E2 loader | n/a until D wires it | n/a | P3 |
| `glyphs_game_shape` | Tonight/Games game-shape glyphs (`dashboard/components/glyphs.py`) | Generated inline SVG (comet, supernova, scales, nebula, hourglass) | keep; optional licensed-texture upgrade later | generated-SVG-keep | n/a | n/a | P3 |
| `astrolabe_engraving` | Slip-builder astrolabe bezel (`dashboard/components/astrolabe_component/build/index.html`) | Generated SVG bezel/orbitals/dials | keep; optional licensed engraving texture | generated-SVG-keep | n/a | n/a | P3 |
| `constellation_silhouettes` | Games constellation decoration layer (`data/config/constellation_shapes.json`, rendered by `constellation_component`) | Generated SVG template paths (49-template bank) | keep; optional artist pass | catalog-only here — Phase D owns these files | n/a | n/a | P3 |
| `team_marks` | Anywhere a team renders — constellation star fills, badges — via `theme.team_colors()` | Hex colors + full names only (`data/config/team_assets.json`); no logo art | Official team logo marks | **owner decision** — licensed or skip; colors already carry the constellation grammar (DESIGN §4a) without them | league merchandising/IP licence required if pursued | SVG per team, recolor-safe via `currentColor` | owner decision |
| `player_headshots` | Constellation ticket-card headshot disc — `.cst-shot` (`dashboard/components/constellation_component/build/main.js:350`, title "Player headshot — coming soon") | Initials-disc scar (`initials()` fallback) | League-CDN player headshots | **owner decision** — CDN terms of use unverified | verify each league CDN's ToS before hot-linking; if cleared, disk-cache locally rather than hot-link (`dashboard_ux_redesign.md` §6 asset-layer plan) | circular crop, ~34×34 render (`.cst-shot`) | owner decision |

The `player_headshots` scar lives in the constellation component's ticket card, not in
`components/deep_dive.py` — there is no separate person-icon stand-in on the offer-detail
dialog or the Board's mobile cards (`components/offer_cards.py`); the sweep for
`:material/person:`-style icons and other `avatar`/`disc` stand-ins turned up nothing else.
The `radial-gradient` sweep found exactly the four hero/card washes above (`theme.py`
twice, `games.py`, `receipts.py`) — no other surface builds one.

## The scar mechanism

`ambient_tonight` and `ambient_receipts_hero` are wired through
`dashboard.assets.ambient_css(slot, fallback_gradient)`: it returns `fallback_gradient`
byte-identical unless the manifest slot has both a `file` that exists on disk *and* a
non-null `license` — an unlicensed file renders the fallback exactly as if the file were
absent. A licensed file is embedded as a base64 data URI (Streamlit serves no arbitrary
static files) under a solid overlay of the surface color, so it never exceeds its manifest
`opacity`. Every other row above is either `generated-SVG-keep` (no sourcing needed — the
generated form *is* the design) or `catalog-only` (owned by another phase).

## What the owner does next

**Free-stock shortlist** (E5, parked — no files downloaded, just the search target per slot):

- `ambient_tonight` — a NASA/ESA-class night-sky or star-field photograph (public-domain or
  CC-attribution), or Unsplash/Pexels-class night-sky photography: subtle, mostly dark, no
  bright foreground subject (renders at 0.14 opacity under body text).
- `ambient_receipts_hero` — a NASA/ESA-class nebula image (Hubble/JWST public-domain
  releases): a data-forward "verdict" card can carry a slightly richer wash (0.16 opacity).

Skip anything editorial-only, NC, or unclear-terms — `ambient_css`'s license gate would
refuse to render it regardless, so an unclear license is wasted download effort.

**Two owner decisions**, both above: `team_marks` (license official team marks, or keep
colors-only — the constellation grammar doesn't require logos) and `player_headshots`
(each league CDN's terms of use need checking before any headshot is hot-linked or cached).

**Commissioned logo**: the brief is [`art_briefs/logo_guru.md`](art_briefs/logo_guru.md).
`app.py` already carries the reserved, existence-guarded slot — dropping
`dashboard/static/logo_wordmark.png` and `dashboard/static/logo_mark.png` in place renders
the logo with zero code change; neither file is committed today. Logos aren't part of
`ambient_manifest.json`'s schema (that manifest is ambient-image only), so this table is
their license/attribution record instead: once the commission is delivered, note the
artist's name and the rights grant directly in the `logo_wordmark`/`logo_mark` rows above,
the same way an ambient asset's `license`/`attribution`/`source_url` triplet works in the
manifest.
