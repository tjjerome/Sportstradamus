# Constellation Art

> Status: ACTIVE — stage 0 done (proof of concept), stage 1 not started (briefed 2026-09-12)

## 1. Mission & money logic

Replace the generated constellation silhouettes with real imagery. Today each of
the 100 shape templates carries a hand-authored SVG `silhouette` path, rendered
as a filled Plotly shape at 13 % alpha under the stars; the owner rates them the
weakest visual in the app. The replacement: licence-free clip art, turned into a
line drawing where it is not one already, recoloured to the theme's light blue,
gaussian-blurred and made translucent, so it reads as the faint figure the stars
trace. No money logic — the Games surface is the owner's nightly product. The
owner prefers automated sourcing and will hunt images by hand only where
automation fails; template vertices may be re-fit to the images.

Proof of concept (2026-09-12, session scratchpad): the owner's openclipart
baseball (Gerald_G, public domain) rasterized at 900 px, ink mask, blur, tint,
alpha → a soft blue line drawing over the starfield that already looks right; a
coloured Commons SVG (`File:Baseball bat.svg`, CC0) through a Sobel edge mask
works too, with a noisier interior. Recipe in §6 stage 1.

## 2. Read first (in order)

1. `src/sportstradamus/dashboard/components/constellation_shapes.py` — the
   docstring rules S1–S9, the `silhouette` path grammar (`M L Q C T S Z` only,
   `scale_path` walks strict x/y alternation), `eligible_templates`,
   `assign_templates` (one deck per night, no repeats, league wall, md5-seeded).
2. `data/config/constellation_shapes.json` — `{version, tuning, templates}`;
   per template `label`, `leagues`, `topology`, `min_nodes`, `vertices` in
   `[-1, 1]²`, `outline`, `silhouette`. Labels are the search vocabulary ("The
   Hoop", "The Plate", "The Goalie Mask", "The Catcher's Mask", "The Goal Line",
   "The Trophy", …).
3. `constellation_slate.py` — `add_decoration` (the `layout.shapes` path entry,
   `layer="below"`, `_SILHOUETTE_FILL`), `scale_path`, `SHAPE_SCALE` /
   `SHAPE_SCALE_MOBILE` (desktop/mobile aspect inversion), the render knobs,
   `template_positions`.
4. `constellation_layout.py` `assign_stars` — the star fit reads `vertices`,
   never the path, so the image can change without touching the fitter, but
   `vertices`/`outline` must survive.
5. `constellation_component/__init__.py` + `build/main.js` — `Plotly.react` on
   every render wipes any DOM node injected into the plot; blur must be baked
   into the image, and the figure JSON is the only Python→JS channel.
6. `dashboard/assets.py` — `_data_uri`: the base64 embed + downscale precedent.
7. [`../../DESIGN.md`](../../DESIGN.md) §3 palette tokens and §4a star grammar
   (FIXED; the decoration layer is not the grammar, and its sentence about "the
   dealt template's filled silhouette" is trued at stage 3).
8. [`../art_assets.md`](../art_assets.md) `constellation_silhouettes` row.
9. Goldens: `tests/golden/test_constellation.py` (decoration block: one shape,
   `layer == "below"`, alpha string, `scale_path` exactness, `focus_scale`
   coupling, mobile inversion, decoration inert and never gold);
   `test_constellation_shapes.py` (schema, S-rules, hot reload, dealer
   determinism, bank depth ≥ 100 / ≥ 60 eligible per league).

## 3. Verify before you trust

```bash
git fetch origin && git log --oneline origin/devel -3
poetry run python -c "import json; d=json.load(open('src/sportstradamus/data/config/constellation_shapes.json')); t=d['templates']; print(len(t), sum('silhouette' in v for v in t.values()), sum('image' in v for v in t.values()))"
grep -n "layout.images\|add_layout_image\|images=" src/sportstradamus/dashboard/components/constellation*.py   # empty = stage 3 open
ls src/sportstradamus/data/assets/constellations/ 2>/dev/null | wc -l                                          # processed layers so far
poetry run python -c "import cairosvg" 2>&1 | tail -1     # not on the dev box; playwright chromium rasterizes SVG instead
curl -s -m 20 -A "Sportstradamus/0.1 (dev)" 'https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=baseball%20bat%20svg&gsrnamespace=6&gsrlimit=3&prop=imageinfo&iiprop=url|extmetadata&iiextmetadatafilter=LicenseShortName|Artist&format=json' | head -c 600
```

### Volatile product assumptions

- Commons licence metadata is user-entered; the owner spot-checks every image
  before it is committed. openclipart's search API answered nothing from the dev
  box (HTTP 000) on 2026-09-12 — Commons is the automated source until that changes.
- Dev-box tooling: PIL (WebP yes), numpy, scipy, playwright chromium; no
  cairosvg, rsvg-convert, inkscape, cv2 or skimage. Rasterize SVG through
  chromium; edge-detect with `scipy.ndimage`.
- Plotly `layout.images` sizing (`sizex`/`sizey`, `xanchor`/`yanchor`) under the
  pinned plotly version — never used in this repo yet.

## 4. Locked decisions

- 2026-09-12 — **Real images replace the generated silhouettes** (owner). The
  processing recipe is line drawing → light-blue palette → gaussian blur →
  transparency, composited below the stars.
- 2026-09-12 — **Automated sourcing first** (owner): search licence-free /
  public-domain clip art programmatically and put candidates in front of the
  owner as one review sheet; hand-hunting is the fallback, batched, never the plan.
- 2026-09-12 — **Licence-free or licensable only, checked by the owner, no code
  gate** — the ambient-art rule. Source URL, artist and licence per image live in
  a manifest next to the files.
- 2026-09-12 — **Template vertices may be re-authored to fit the images**
  (owner); the star grammar (DESIGN.md §4a), the dealer's guarantees (no repeats
  per night, league wall, deterministic deal) and the bank-depth floors do not move.
- **No AI-generated imagery** (CLAUDE.md, DESIGN.md NEVER list): human-drawn clip
  art through filters is fine; nothing model-made.
- **Palette**: the tint is the theme's light blue `#7FAAE8`
  (`theme.SEQUENTIAL_COLORS`); no new hex. Opacity stays a named knob beside
  `SILHOUETTE_ALPHA`.
- **Incremental fill**: a template with `image: null` renders outline-only
  (engraved outline + fillers, no silhouette), so the bank never waits on the
  last image.

## 5. Module footprint & canonical paths

| Module | Role |
|---|---|
| `data/assets/constellations/{slug}.png` (NEW, committed, ≤ 600 px RGBA) + `manifest.json` | one processed layer per template; `slug → {file, source_url, artist, licence, recipe}` |
| `scripts/constellation_art.py` (NEW, dev-side; never a prod job) | `search` (Commons API → candidates per template), `render` (SVG → PNG via playwright; PNG passthrough), `process` (mask → blur → tint → alpha), `sheet` (contact sheet over the page ground for owner review) |
| `data/config/constellation_shapes.json` | `image` per template replaces `silhouette`; vertices re-fit where needed |
| `components/constellation_shapes.py` | schema: `image` (nullable) validated, file must exist; path rules retire with the last silhouette |
| `components/constellation_slate.py` | `add_decoration`: a `layout.images` entry (data URI, `xref`/`yref` data, `sizex`/`sizey` from the same `sx, sy`, `layer="below"`, opacity knob) replaces the path shape; `scale_path` retires |
| `dashboard/assets.py` | `_data_uri` reused for the embed |
| `tests/golden/test_constellation.py`, `test_constellation_shapes.py` | decoration pins re-targeted to `layout.images`; every referenced image exists; manifest licence non-empty |

## 6. Stage plan

0. **Proof of concept** — done 2026-09-12 (§1). The owner holds three sample
   files (openclipart baseball by Gerald_G, a crossed-bats line-art PNG, a
   football gridiron SVG); ask for them at stage 1 and keep them under
   `data/assets/constellations/sources/`.
1. **Processing tool** (1 session). `scripts/constellation_art.py process <src>
   --mode ink|edges --slug <slug>`: render SVG at 900 px through chromium;
   `ink` mask = pixels darker than 75 % grey (line art on a light ground),
   `edges` mask = Sobel magnitude normalised at its 99.5th percentile (filled
   colour art); gaussian σ ≈ 2.2 px at 900 px; tint `#7FAAE8`; alpha peak ≈ 0.55
   before the render-time opacity knob; crop to the ink bounding box with a
   margin; save ≤ 600 px RGBA PNG. Every number a named constant with a why.
   `sheet` tiles every processed layer over `#0E1117` with dust dots. Acceptance:
   the two POC inputs reproduce; the sheet renders.
2. **Sourcing** (1–2 sessions + one owner sitting). For each template, query
   Commons (`generator=search`, namespace 6, `iiextmetadatafilter=LicenseShortName|Artist`)
   with the label's nouns plus "svg" / "line art" / "clip art"; keep CC0 and
   public-domain hits (the owner says whether CC-BY, which needs an attribution
   line somewhere on Games, is acceptable); write candidates + thumbnails into
   the review sheet; the owner picks or rejects per template in one pass;
   rejects go to the owner's hand-hunt list. League-specific templates first
   (12 each for MLB / NFL / NHL / NBA+WNBA), then the 52 general ones.
   Acceptance: manifest rows for every league-specific template; the general
   set may fill over later visits.
3. **Render path** (1 session). `add_decoration` emits `layout.images` instead
   of the path shape; the desktop/mobile `SHAPE_SCALE` inversion and the
   `focus_scale` coupling carry through `sizex`/`sizey`; opacity knob; goldens
   re-pinned; DESIGN.md's decoration sentence trued. Live-verify on Games
   (playwright, desktop + phone, main/deeper/wider, iframe height on every
   toggle, zero page errors). Acceptance: verdict in the ledger.
4. **Star re-fit** (1–2 sessions). Templates whose vertex graph no longer sits
   on the image get `vertices`/`outline` re-authored against an overlay sheet at
   the real figure aspect (the offline authoring sheet, never string surgery);
   dealer goldens stay green. Acceptance: every filled template's stars land on
   the drawing in the sheet.
5. **Retire the SVG silhouettes.** Drop `silhouette` keys, `scale_path` and the
   path-grammar rules once every template has an image or a deliberate `null`;
   `art_assets.md` row → done.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > home-of-record
  doc > this brief > roadmap v3.
- Dev-side tooling only; the images are committed, so prod needs no job.
- One module per subagent; no HEAD-moving git in subagent prompts; `git add` by
  explicit path.
- Commit web-sized files only (≤ 600 px) — git keeps every version of a binary.
- DESIGN.md tokens only; nothing model-generated; the decoration layer stays
  inert (no hover, no click, never gold).
- `main.js` and the figure get the playwright verdict on every render change.

## 8. Escalation & stop conditions

**Stop and ask the owner:** any image that is not CC0 / public domain (CC-BY
means an attribution line on the surface — a design call); the committed image
set passing ~5 MB; `layout.images` unable to honour the desktop/mobile aspect
contract; a re-fit that would breach the bank-depth floors.

**Park and pivot:** stages 2 and 4 can pause per template (`image: null` keeps
the outline-only fallback); stage 3 can land on the POC images alone.

**Dispatch:** `refactoring-specialist` on every touched `.py`.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session.
- `poetry run ruff check src/sportstradamus/` clean; CI also runs `ruff format`.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- Render changes: playwright verdict recorded (desktop + phone, three lens modes).
- Manifest row (source, artist, licence) for every image committed this session.
- One ledger line appended below; status line updated on stage boundaries.
- Never push `devel`.

## 10. Ledger (append-only, newest first, cap ~15)

- 2026-09-12 · stage 0 · brief written; POC on the owner's baseball SVG + a CC0 Commons bat (ink and edge masks, blur, tint, alpha) reads right over the starfield; Commons API confirmed as the automated source, openclipart API dead from the dev box · next: stage 1 processing tool
