# Constellation Art

> Status: ACTIVE — stage 2 in progress (sourcing, 2026-09-13): 88 of 101 templates have a layer; MLB + NHL complete; 10 league templates + 3 general on the owner's hand-hunt list (§6)

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
works too, with a noisier interior. The recipe lives as named constants in
`src/sportstradamus/scripts/constellation_art.py` (stage 1).

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
ls src/sportstradamus/data/assets/constellations/*.png 2>/dev/null | wc -l                                     # processed layers so far
poetry run python -c "import cairosvg" 2>&1 | tail -1     # not on the dev box; playwright chromium rasterizes SVG instead
curl -s -m 20 -A "Sportstradamus/0.1 (dev)" 'https://openclipart.org/search/?query=hockey%20stick' | grep -o 'href="/detail/[0-9]*/[^"]*"' | head -3   # openclipart HTML search; empty = the source is down
curl -s -m 20 -A "Sportstradamus/0.1 (dev)" 'https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=baseball%20bat%20svg&gsrnamespace=6&gsrlimit=3&prop=imageinfo&iiprop=url|extmetadata&iiextmetadatafilter=LicenseShortName|Artist&format=json' | head -c 600   # the Commons fallback
```

### Volatile product assumptions

- openclipart is the automated source (2026-09-13): its JSON search API is dead,
  but the HTML search page (`/search/?query=`, ~30 hits a page), the
  `/image/250px/<id>` thumbnails, `/download/<id>` (302 to the named SVG; an
  unknown id lands on the site logo, not a 404) and `/detail/<id>` (artist) all
  answer from the dev box, and every upload is CC0 by the site's terms. Commons
  is the fallback: licence metadata is user-entered (the owner spot-checks), its
  API 429s after about ten quick calls, and its CC0/PD yield for sports nouns is
  thin — icon sets, logos and maps; game-icons.net's near-complete sports set
  there is CC BY 3.0 (an owner call, §8).
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
- 2026-09-13 — **Templates may be remade to fit the artwork** (owner): vertices,
  outline, topology, and new templates for art the bank has no shape for (the
  crossed bats), inside the dealer guarantees and the bank-depth floors.
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
| `data/assets/constellations/{slug}.png` (committed, ≤ 600 px RGBA) + `manifest.json` + `sources/` | one processed layer per template; `slug → {file, source, source_url, artist, licence, mode}`; every input lands under `sources/`, but the directory's `.gitignore` commits only what no `pick` or URL can re-fetch (compositions, the owner's files) — openclipart and Commons downloads run to megabytes |
| `src/sportstradamus/scripts/constellation_art.py` (dev-side; never a prod job) | `search` (openclipart HTML search per template → `candidates.json`, `thumbs/`, one review sheet per league group and page), `pick` (one openclipart id → download under `sources/`, artist off the detail page, CC0 → the same write path as `process`), `process` (an owner-supplied SVG through chromium or PNG through PIL → mask → blur → tint → alpha → crop; writes the layer, its manifest row and the source copy), `sheet` (contact sheet of the processed layers over the page ground) |
| `data/config/constellation_shapes.json` | `image` per template replaces `silhouette`; vertices re-fit where needed |
| `components/constellation_shapes.py` | schema: `image` (nullable) validated, file must exist; path rules retire with the last silhouette |
| `components/constellation_slate.py` | `add_decoration`: a `layout.images` entry (data URI, `xref`/`yref` data, `sizex`/`sizey` from the same `sx, sy`, `layer="below"`, opacity knob) replaces the path shape; `scale_path` retires |
| `dashboard/assets.py` | `_data_uri` reused for the embed |
| `tests/golden/test_constellation.py`, `test_constellation_shapes.py` | decoration pins re-targeted to `layout.images`; every referenced image exists; manifest licence non-empty |

## 6. Stage plan

0. **Proof of concept** — done 2026-09-12 (§1). The owner's three sample files:
   the openclipart baseball (Gerald_G, public domain) is `the-baseball` (ink);
   the football gridiron SVG is byte-for-byte Commons' CC0 copy already under
   `sources/`; the crossed-bats line-art PNG was shown but never handed over as
   a file, so `the-crossed-bats` (MLB, twin, the first template made under the §4
   remake authority, 2026-09-13) is composed instead: two copies of Gerald_G's
   public-domain openclipart bat (8300), the second mirrored, as
   `sources/crossed-bats.svg` (ink). The owner's own PNG can replace it through
   the same `process` call.
1. **Processing tool** — done 2026-09-13.
   `poetry run python -m sportstradamus.scripts.constellation_art process <src>
   --slug <slug> --mode ink|edges --source-url … --artist … --licence …` writes
   the layer, its manifest row and the source copy; `sheet --out <png>` tiles
   every layer over the page ground with dust. Every recipe number is a named
   constant in the module; the alpha band is stored in 16 steps
   (`_ALPHA_LEVELS`, 2026-09-13) — invisible at the layer's on-screen opacity,
   about a third of the bytes, which is what fits a hundred layers under the §8
   line. Pins: `tests/golden/test_constellation_art.py`.
   First layers: `the-bat` and `the-gridiron` (Commons CC0, `edges`), `the-baseball`
   (the owner's openclipart file, `ink`).
2. **Sourcing** — in progress (first pass 2026-09-13: 88 of 101 templates).
   `search --out <folder>` asks openclipart per template with its label
   (sport-prefixed for a league template; `--query "…" <slug>` re-asks one) and
   writes `candidates.json`, `thumbs/` and `sheet-<group>-<n>.png` review pages;
   a choice is `pick <slug> <id> --mode ink|edges`. Everything on openclipart is
   CC0; a Commons hit goes through `process` with the licence and artist read
   off its file page (eight landed that way: route tree, field with hashmarks,
   backboard, half court, hockey goalie, field overview, goalkeeper glove,
   bleachers glyph). MLB 13/13 and NHL 12/12 complete (`the-crossed-sticks` is
   two mirrored copies of J_Alves's stick, like the bats); NFL 8/12, NBA+WNBA
   6/12, general 49/52.
   **Hand-hunt list** (neither source has a CC0/PD drawing; the owner finds a
   file or remakes the template around art that exists, §4): NFL
   `the-goalposts`, `the-kicking-tee`, `the-chain-gang`, `the-goal-line`; NBA
   `the-shot-clock`, `the-arc`, `the-crossover`, `the-alley-oop`,
   `the-wristbands`, `the-elbows` (`the-arc` and `the-elbows` could share
   `the-key`'s half-court drawing if the three merge); general `the-firework`,
   `the-bracket`, `the-bowtie` (openclipart offered a starburst blob, a curly
   brace and a portrait). A found file lands with one call:
   `process <file> --slug <slug> --mode ink|edges --source-url … --artist … --licence …`.
   **Owner call:** game-icons.net's complete sports set on Commons is CC BY 3.0
   (an attribution line on Games); it would fill most of the list.
   Acceptance unchanged: manifest rows for every league-specific template.
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

- 2026-09-13 · stage 2 · openclipart `search`/`pick` (every upload CC0; HTML search, 400 px thumbs — the 250 px ones are placeholders for recent ids — `/download` + `/detail`) + per-group review sheets; 80 layers picked in one sitting, 8 more from Commons by hand through `process`, `the-crossed-sticks` composed; 88/101 with MLB and NHL complete, 13 on the hand-hunt list; downloads gitignored (one huddle SVG is 6.8 MB), only compositions and the owner's files committed · next: owner hand-hunt + the CC BY 3.0 call; stage 3 render path
- 2026-09-13 · crossed bats · `the-crossed-bats` template (MLB, twin/chain, 9 stars: tips, knobs, barrel and handle mids, the crossing) + its layer composed from two mirrored copies of Gerald_G's public-domain openclipart bat (8300, ink); stars checked on an overlay of the layer at its own aspect; catalog 100 → 101, MLB eligible 65 · next: stage 2 sourcing
- 2026-09-13 · stage 1 · `src/sportstradamus/scripts/constellation_art.py` (`process`, `sheet`) + golden pins; first three layers committed (the-bat + the-gridiron from Commons CC0 art, edges; the-baseball from the owner's openclipart file, ink) with manifest rows + sources; `render` folded into `process`, `search` deferred to stage 2; chromium synthesises a viewBox for width/height-only SVGs, so `object-fit: contain` scales every Commons file seen so far · next: stage 2 sourcing; the crossed-bats PNG still needs a file path
- 2026-09-12 · stage 0 · brief written; POC on the owner's baseball SVG + a CC0 Commons bat (ink and edge masks, blur, tint, alpha) reads right over the starfield; Commons API confirmed as the automated source, openclipart API dead from the dev box · next: stage 1 processing tool
