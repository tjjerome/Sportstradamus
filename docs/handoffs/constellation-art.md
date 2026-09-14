# Constellation Art

> Status: ACTIVE — stage 4 done 2026-09-13: every drawn template's stars sit on its drawing; 99 of 101 templates have a layer, `the-chain-gang` (NFL) and `the-crossover` (NBA) on the owner's hand-hunt list; stage 5 (retire the silhouettes) next (§6)

## 1. Mission & money logic

Replace the generated constellation silhouettes with real imagery. Each shape
template carried a hand-authored SVG `silhouette` path, rendered as a filled
Plotly shape at 13 % alpha under the stars; the owner rated them the weakest
visual in the app. The replacement: licence-free clip art, turned into a
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
   docstring rules S1–S9 (S5: `image`, a layer file under `ART_DIR` or `null`),
   `eligible_templates`, `assign_templates` (one deck per night, no repeats,
   league wall, md5-seeded).
2. `data/config/constellation_shapes.json` — `{version, tuning, templates}`;
   per template `label`, `leagues`, `topology`, `min_nodes`, `vertices` in
   `[-1, 1]²`, `outline`, `image` (and the unrendered `silhouette`). Labels are
   the search vocabulary ("The
   Hoop", "The Plate", "The Goalie Mask", "The Catcher's Mask", "The Goal Line",
   "The Trophy", …).
3. `constellation_slate.py` — `add_decoration` (one `layout.images` entry,
   `layer="below"`, `ART_OPACITY`), `SHAPE_SCALE` / `SHAPE_SCALE_MOBILE`
   (desktop/mobile aspect inversion), the render knobs, `template_positions`.
4. `constellation_layout.py` `assign_stars` — the star fit reads `vertices`,
   never the path, so the image can change without touching the fitter, but
   `vertices`/`outline` must survive.
5. `constellation_component/__init__.py` + `build/main.js` — `Plotly.react` on
   every render wipes any DOM node injected into the plot; blur must be baked
   into the image, and the figure JSON is the only Python→JS channel.
6. `dashboard/assets.py` — `constellation_layer` (the base64 embed plus the
   layer's sides as shares of its longer one) and `constellation_credit` (the CC BY
   line under the Games map).
7. [`../../DESIGN.md`](../../DESIGN.md) §3 palette tokens and ambient ceiling,
   §4a star grammar (FIXED; the decoration layer is not the grammar).
8. [`../art_assets.md`](../art_assets.md) `constellation_silhouettes` row.
9. Goldens: `tests/golden/test_constellation.py` (decoration block: one layout
   image below everything at `ART_OPACITY`, the layer's aspect under both frame
   scales, outline alone without art, `focus_scale` coupling, decoration inert and
   never gold); `test_constellation_shapes.py` (schema, S-rules, every star on its drawing, hot reload,
   dealer determinism, bank depth ≥ 100 / ≥ 60 eligible per league);
   `test_constellation_art.py` (the tool, the shipped manifest, every drawn image
   recorded, the §3 ceiling, the layer loader, the credit).

## 3. Verify before you trust

```bash
git fetch origin && git log --oneline origin/devel -3
poetry run python -c "import json; d=json.load(open('src/sportstradamus/data/config/constellation_shapes.json')); t=d['templates']; print(len(t), 'with art:', sum(v['image'] is not None for v in t.values()), 'silhouettes left:', sum('silhouette' in v for v in t.values()))"
grep -n "add_layout_image" src/sportstradamus/dashboard/components/constellation_slate.py   # the stage-3 render path
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
  thin — icon sets, logos and maps. game-icons.net (CC BY 3.0, in per §4)
  ships its whole set as one zip,
  `https://game-icons.net/archives/svg/zip/ffffff/000000/game-icons.net.svg.zip`
  (4181 icons as `1x1/<author>/<name>.svg`, a white glyph on a black square);
  the credit page is `https://game-icons.net/1x1/<author>/<name>.html`.
- Dev-box tooling: PIL (WebP yes), numpy, scipy, playwright chromium; no
  cairosvg, rsvg-convert, inkscape, cv2 or skimage. Rasterize SVG through
  chromium; edge-detect with `scipy.ndimage`.
- Plotly `layout.images` (plotly 6.9.0, live-verified 2026-09-13): data
  `xref`/`yref` with `sizing="stretch"` fills `sizex` × `sizey` exactly, and
  `layer="below"` draws under every trace on the transparent plot; the image rides
  `Plotly.react` with no `main.js` change.

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
- 2026-09-13 — **game-icons.net art is in** (owner): its icons are CC BY 3.0, so
  stage 3 renders an attribution line on Games naming the authors (Delapouite,
  Lorc, Skoll …) and game-icons.net. Rows carry `licence: "CC BY 3.0"`; the
  archive is in §3, the recolour recipe in §6.
- 2026-09-13 — **Templates may be remade to fit the artwork** (owner): vertices,
  outline, topology, and new templates for art the bank has no shape for (the
  crossed bats), inside the dealer guarantees and the bank-depth floors.
- **No AI-generated imagery** (CLAUDE.md, DESIGN.md NEVER list): human-drawn clip
  art through filters is fine; nothing model-made.
- **Palette**: the tint is the theme's light blue `#7FAAE8`
  (`theme.SEQUENTIAL_COLORS`); no new hex. Opacity is the named knob
  `ART_OPACITY` (`constellation_slate.py`), sized so the layers' baked 0.55 alpha
  peak renders under the DESIGN §3 ceiling of 0.20.
- **Incremental fill**: a template with `image: null` renders outline-only
  (engraved outline + fillers, no silhouette), so the bank never waits on the
  last image.

## 5. Module footprint & canonical paths

| Module | Role |
|---|---|
| `data/assets/constellations/{slug}.png` (committed, ≤ 600 px RGBA) + `manifest.json` + `sources/` | one processed layer per template; `slug → {file, source, source_url, artist, licence, mode}`; every input lands under `sources/`, but the directory's `.gitignore` commits only what no `pick` or URL can re-fetch (compositions, pose wrappers, the owner's files, the court crops) — openclipart, Commons and game-icons downloads run to megabytes |
| `src/sportstradamus/scripts/constellation_art.py` (dev-side; never a prod job) | `search` (openclipart HTML search per template → `candidates.json`, `thumbs/`, one review sheet per league group and page), `pick` (one openclipart id → download under `sources/`, artist off the detail page, CC0 → the same write path as `process`), `process` (an owner-supplied SVG through chromium or PNG through PIL → mask → blur → tint → alpha → crop; writes the layer, its manifest row and the source copy), `sheet` (contact sheet of the processed layers over the page ground) |
| `data/config/constellation_shapes.json` | `image` per template: a layer file (two templates may share one) or `null`; `silhouette` still in the file, unrendered, until stage 5; `vertices` fit to the drawing (stage 4) |
| `components/constellation_shapes.py` | `ART_DIR`; S5: a non-null `image` must be a file there, and every star sits on its drawing; the silhouette grammar check retires with the key |
| `components/constellation_layout.py` | `_field_box`: a side's field stars run out to the edge of its half, so a narrow drawing never pens them into a column |
| `components/constellation_slate.py` | `add_decoration`: one `layout.images` entry (data URI, data `xref`/`yref`, centred on the origin, `sizex`/`sizey` = 2 × the layer's side shares × the frame's `sx, sy`, `sizing="stretch"`, `layer="below"`, `ART_OPACITY`); `image: null` draws the outline alone |
| `dashboard/assets.py` | `constellation_layer` (data URI through `_data_uri`, sides as shares of the longer), `constellation_credit` (markdown credit for CC BY rows: artists, site, licence link) |
| `surfaces/games.py` | the credit caption under the map and cockpit |
| `tests/golden/test_constellation.py`, `test_constellation_shapes.py`, `test_constellation_art.py` | decoration pins on `layout.images`; every catalog `image` a recorded layer and every star on its drawing; every layer under the §3 ceiling at `ART_OPACITY`; the credit names every CC BY artist |

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
2. **Sourcing** — in progress (2026-09-13: 99 of 101 templates).
   `search --out <folder>` asks openclipart per template with its label
   (sport-prefixed for a league template; `--query "…" <slug>` re-asks one) and
   writes `candidates.json`, `thumbs/` and `sheet-<group>-<n>.png` review pages;
   a choice is `pick <slug> <id> --mode ink|edges`. Everything on openclipart is
   CC0; a Commons hit goes through `process` with the licence and artist read
   off its file page (eight landed that way: route tree, field with hashmarks,
   backboard, half court, hockey goalie, field overview, goalkeeper glove,
   bleachers glyph). A game-icons.net icon (§4) goes the same way: unzip the
   archive (§3), delete the icon's `<path d="M0 0h512v512H0z"/>` background,
   turn `fill="#fff"` into `fill="#000"`, then `process <svg> --slug <slug>
   --mode ink --licence "CC BY 3.0" --artist <Author> --source-url <credit page>`
   (nine landed that way: goalposts, kicking tee, goal line, shot clock,
   alley-oop, wristbands, firework, bracket, bowtie). MLB 13/13, NHL 12/12 and
   general 52/52 complete (`the-crossed-sticks` is two mirrored copies of
   J_Alves's stick, like the bats); NFL 11/12, NBA+WNBA 11/12. The court trio
   share one public-domain Commons drawing (`File:Basketball_positions.svg`,
   GateKeeperX): `the-key` is the full half court, `the-arc` and `the-elbows`
   are crops of it (`sources/half-court.svg`, `three-point-arc.svg`,
   `the-paint.svg`; the tan floor fill is removed so the crop borders don't
   edge), and all three had their stars re-authored on the overlay sheet under
   the §4 remake authority — folding them into one template is blocked by the
   dealer's deck-depth pin (NBA keeps three templates per topology class).
   Weak fits to swap when better art turns up: `the-kicking-tee` (a golf tee),
   `the-wristbands` (a headband knot).
   **Hand-hunt list** (no drawing on any of the three sources; the owner finds
   a file or remakes the template around art that exists, §4): NFL
   `the-chain-gang`, NBA `the-crossover`. A found file lands with one call,
   `process <file> --slug <slug> --mode ink|edges --source-url … --artist … --licence …`,
   plus `"image": "<slug>.png"` on its template in the catalog (a swapped layer
   keeps its file name, so a swap needs no catalog edit).
   Acceptance unchanged: manifest rows for every league-specific template.
3. **Render path** — done 2026-09-13. `add_decoration` draws the template's
   layer as one `layout.images` entry instead of the path shape (§5); the
   desktop/mobile `SHAPE_SCALE` inversion and the `focus_scale` coupling carry
   through `sizex`/`sizey`; `ART_OPACITY` 0.36 renders the layers' stored 0.53
   alpha peak at 0.19; `image: null` draws the outline alone; the CC BY credit
   sits under the map; DESIGN.md §4a trued. The render keeps each layer's
   aspect, so the stars authored on non-square layers moved with it: the arc's y
   and the paint's x scaled by the layer's side share. Verdict in the ledger.
4. **Star re-fit** — done 2026-09-13. Every drawn template's `vertices` were
   re-authored on an overlay of its layer at the render's aspect (vertex `(x, y)`
   at layer pixel `(W/2 + x·L/2, H/2 − y·L/2)`, `L` the longer side) and snapped
   onto the drawn line, never by string surgery. `test_constellation_shapes.py`
   pins the result: every star within 0.035 of a line the layer stores at alpha
   34 or more, none above |y| 0.90 or nearer than 0.18 to another (`settle`
   would nudge it off its line), and each axis spanning 1.0, or 80 % of a
   drawing narrower than that. A centre vertex stays on the axis, because both
   teams draw from it. Five layers were re-posed to carry their shape, each a
   committed SVG wrapper around the unchanged download: the ladder leans, the
   arrow and the comet climb, the clipboard stands clip-up, and the medal is a
   darkened ink silhouette instead of edges. Under the §4 authority `the-laurels`
   and `the-clipboard` lost a centre star their art has no line for, `the-key`
   gained one, `the-banner` was re-meshed, `the-bolt` and `the-staircase` were
   re-sided and `the-stick` mirrored to follow their art. A drawing narrower than
   the old ±0.8 reach penned a busy side's field stars into a column, so
   `constellation_layout._field_box` runs each side's field out to the edge of
   its half. Verdict in the ledger.
5. **Retire the SVG silhouettes.** Drop the `silhouette` keys and the loader's
   path-grammar check — unblocked since stage 3 gave every template an `image` or
   a deliberate `null` (`scale_path` went with the path shape); `art_assets.md`
   row → done.

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

**Stop and ask the owner:** any image that is not CC0 / public domain and not
game-icons.net (§4; CC-BY means an attribution line on the surface); the
committed image set passing ~5 MB; `layout.images` unable to honour the
desktop/mobile aspect contract; a re-fit that would breach the bank-depth floors.

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

- 2026-09-13 · stage 4 · every drawn template's stars re-fit on an overlay of its layer at the render's aspect and snapped onto the line; `test_constellation_shapes.py` pins it (each star within 0.035 of alpha ≥ 34, |y| ≤ 0.90, stars ≥ 0.18 apart, each axis spanning 1.0 or 0.8 of a narrower drawing); five layers re-posed through committed SVG wrappers (ladder, arrow, comet and clipboard rotated, medal darkened to ink); laurels and clipboard lost a centre star, key gained one, banner re-meshed, bolt and staircase re-sided, stick mirrored; narrow drawings penned the field stars into a column and turned the hourglass clearance and caption pins red, so `_field_box` runs each side's field to its half's edge · bank stress check (101 templates, ladder pools, old → new catalog, both with the field fix): stars drifting past their radius under wider desktop 4 → 7, phone 45 → 58; slip star uncaptioned desktop 9 → 11, phone 78 → 80; a y-band floor (±0.7 or full height) cut phone drift to 46 or 35 and desktop misses to 4 but raised phone misses to 87, left for the owner · playwright verdict from a worktree at HEAD + this stage's files (a peer session had the shared tree mid-edit): desktop 1440×900 + phone 390×844 through main → deeper → deeper+wider → wider → main on an NFL night dealt `the-gridiron`; one image per mode, ×0.8 under wider, stars on the sidelines and yard lines, iframe heights as in stage 3, credit present, zero page errors; offline `constellation_figure` renders of the hourglass, torch, scoreboard, ladder, medal, key and banner put every vertex star on its drawing, the short ones (scoreboard, banner) crowded on the phone · next: stage 5 retire the silhouettes; owner call on the y band; owner hunts the last two
- 2026-09-13 · stage 3 · Games draws the art: `image` key per template (99 files, 2 null), one `layout.images` entry at `ART_OPACITY` 0.36 (peak ≈ 0.19, under the §3 ceiling), layer aspect kept and scaled like the vertices, outline alone without art, CC BY credit caption from the manifest; `scale_path` and the silhouette fill removed, keys left for stage 5 · playwright verdict: desktop 1440×900 + phone 390×844 through main → deeper → deeper+wider → wider → main on an NFL night dealt `the-gridiron`; one image in every mode, shrunk ×0.8 with the stars under wider, iframe = figure + 8 (desktop) and + 208 (phone, 380 → 739 px sky), credit with both links, zero page errors (two Streamlit `/games/_stcore` 404s from the deep link itself) · next: stage 4 star re-fit; owner hunts the last two
- 2026-09-13 · stage 2, second round · game-icons.net in (CC BY 3.0, §4): nine hand-hunt layers off its archive zip (background dropped, fill flipped, ink); `the-key` re-authored on the full half court, `the-arc` and `the-elbows` on crops of the same public-domain Commons court (merging the three blocked by the NBA deck-depth pin), stars checked on overlays; 99/101 — `the-chain-gang` and `the-crossover` left; weak fits: kicking tee (golf tee), wristbands (headband knot) · next: owner hunts the last two; stage 3 render path + the attribution line
- 2026-09-13 · stage 2 · openclipart `search`/`pick` (every upload CC0; HTML search, 400 px thumbs — the 250 px ones are placeholders for recent ids — `/download` + `/detail`) + per-group review sheets; 80 layers picked in one sitting, 8 more from Commons by hand through `process`, `the-crossed-sticks` composed; 88/101 with MLB and NHL complete, 13 on the hand-hunt list; downloads gitignored (one huddle SVG is 6.8 MB), only compositions and the owner's files committed · next: owner hand-hunt + the CC BY 3.0 call; stage 3 render path
- 2026-09-13 · crossed bats · `the-crossed-bats` template (MLB, twin/chain, 9 stars: tips, knobs, barrel and handle mids, the crossing) + its layer composed from two mirrored copies of Gerald_G's public-domain openclipart bat (8300, ink); stars checked on an overlay of the layer at its own aspect; catalog 100 → 101, MLB eligible 65 · next: stage 2 sourcing
- 2026-09-13 · stage 1 · `src/sportstradamus/scripts/constellation_art.py` (`process`, `sheet`) + golden pins; first three layers committed (the-bat + the-gridiron from Commons CC0 art, edges; the-baseball from the owner's openclipart file, ink) with manifest rows + sources; `render` folded into `process`, `search` deferred to stage 2; chromium synthesises a viewBox for width/height-only SVGs, so `object-fit: contain` scales every Commons file seen so far · next: stage 2 sourcing; the crossed-bats PNG still needs a file path
- 2026-09-12 · stage 0 · brief written; POC on the owner's baseball SVG + a CC0 Commons bat (ink and edge masks, blur, tint, alpha) reads right over the starfield; Commons API confirmed as the automated source, openclipart API dead from the dev box · next: stage 1 processing tool
