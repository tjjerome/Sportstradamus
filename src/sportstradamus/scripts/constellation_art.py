"""Turn clip art into the soft light-blue layer a constellation renders beneath its stars.

One piece of licence-free clip art in; one web-sized RGBA PNG out under
``data/assets/constellations/``, its provenance row in ``manifest.json``, and a copy of
the input under ``sources/`` so the run reproduces offline. The recipe is the stage-0
proof of concept in ``docs/handoffs/constellation-art.md``: rasterize at 900 px, mask
the ink (line art) or the edges (filled colour art), gaussian-blur, tint the theme's
light blue, bake the alpha, crop. Dev-side only — the layers are committed, so
production never runs this.

Sourcing goes through openclipart, whose every upload is CC0: ``search`` asks its
HTML search page per template (the JSON API is dead) and lays the hits out on review
sheets, ``pick`` fetches one drawing by id and processes it with provenance filled in.

Usage
-----
    poetry run python -m sportstradamus.scripts.constellation_art search --out /tmp/review
    poetry run python -m sportstradamus.scripts.constellation_art search --out /tmp/review \\
        --query "goal post" the-goalposts
    poetry run python -m sportstradamus.scripts.constellation_art pick the-stick 35545 --mode ink
    poetry run python -m sportstradamus.scripts.constellation_art process SRC \\
        --slug the-bat --mode edges --source-url URL --artist "Gerald G" --licence CC0
    poetry run python -m sportstradamus.scripts.constellation_art sheet --out /tmp/sheet.png
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import shutil
import time
from pathlib import Path

import click
import numpy as np
import requests
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from scipy import ndimage
from tqdm import tqdm

from sportstradamus.dashboard.components.constellation_shapes import ART_DIR, shape_catalog
from sportstradamus.dashboard.theme import GRAY, SEQUENTIAL_COLORS

# Every mask and blur number below was tuned at this size in the stage-0 proof of
# concept, so a raster input is brought to it too before anything is measured.
_RENDER_PX = 900

# Luminance under three-quarters grey counts as ink: line art on a light ground with its
# anti-aliased edges, and none of the paper.
_INK_LUMA_MAX = 0.75

# Sobel magnitude is normalised at this percentile of the edge pixels, so a few hard
# seams saturate instead of washing the rest of the drawing out.
_EDGE_NORM_PERCENTILE = 99.5

# At 900 px: soft enough to read as a faint figure, sharp enough to keep the drawing.
_BLUR_SIGMA_PX = 2.2

# Peak alpha baked into the file. The render-time opacity knob (stage 3) scales down from
# here to the DESIGN.md §3 ceiling, so the layer keeps headroom.
_ALPHA_PEAK = 0.55

# Alpha at or under this is ground; the crop box is measured on what remains.
_ALPHA_FLOOR = 2 / 255

# Margin around the ink box as a fraction of its longer side, so the blur tail and the
# figure's breathing room survive the crop.
_CROP_MARGIN = 0.06

# Brief §7: commit web-sized files only — git keeps every version of a binary.
_LAYER_MAX_PX = 600

# Sixteen alpha levels band invisibly at the layer's on-screen opacity and shrink the PNGs
# by about two thirds — what keeps a hundred layers under the brief's §8 line.
_ALPHA_LEVELS = 16

# The theme's light blue (#7FAAE8) is the locked tint; no new hex.
_TINT = ImageColor.getrgb(SEQUENTIAL_COLORS[3])

# Sheet ground mirrors .streamlit/config.toml backgroundColor; theme.py names no token.
_GROUND = "#0E1117"
_SHEET_COLUMNS = 5  # one row the owner can scan on a single screen
_SHEET_TILE_PX = 640  # the layer cap plus room for the label
_SHEET_DUST_PER_TILE = 60  # enough scattered stars to judge a layer against a sky
_SHEET_DUST_SEED = 0  # the same sky every run, so two sheets differ on the layers alone
_SHEET_LABEL_PT = 14  # the size-less default font is an unreadable bitmap

_OPENCLIPART = "https://openclipart.org"

# openclipart's upload terms put every drawing under CC0, so no licence is ever read.
_OPENCLIPART_LICENCE = "CC0"

# An identifying agent and one request a second: the site is volunteer-run.
_USER_AGENT = "Sportstradamus/0.1 (dev-side constellation-art tool)"
_REQUEST_PAUSE_S = 1.0
_REQUEST_TIMEOUT_S = 30

# Enough to choose from per template while a review row stays one screen wide.
_CANDIDATES_PER_TEMPLATE = 6

# The review sheet tiles thumbnails at 250 px, but openclipart's 250 px render is a
# placeholder for most recent uploads; 400 px is the smallest size that answers.
_THUMB_PX = 250
_THUMB_FETCH_PX = 400

# Review-sheet geometry: label column, thumb plus gutter, thumb plus caption, and one
# league's dozen templates per page (the general set pages the same way).
_CAND_LABEL_PX = 170
_CAND_TILE_PX = 260
_CAND_ROW_PX = 300
_CAND_ROWS_PER_PAGE = 12
_CAND_CAPTION_CHARS = 28  # of the title, after the id: what a tile holds at the label size

# A sport word in front of a league template's label: "hoop" alone finds hula hoops.
_SPORT_WORD = {
    "MLB": "baseball",
    "NFL": "football",
    "NHL": "hockey",
    "NBA": "basketball",
    "WNBA": "basketball",
}

# openclipart's search gallery and detail page, as served on 2026-09-13.
_ARTWORK = re.compile(r'<a href="/detail/(\d+)/[^"]*">\s*<img src="[^"]*" alt="([^"]*)"')
_ARTIST = re.compile(r'by <a href="/artist/[^"]*">([^<]*)<')


def rasterize(src: Path) -> Image.Image:
    """``src`` as RGB on white with its longest side at ``_RENDER_PX``.

    An SVG goes through chromium as an ``<img>`` in a square box with ``object-fit:
    contain``, which normalises every sizing an SVG can carry — bar one with neither
    width/height nor a viewBox, which chromium draws at 300×150 unscaled. A raster is
    flattened onto white and fitted to the same box.
    """
    if src.suffix.lower() == ".svg":
        # A dev-group dependency: importing it here keeps the module importable where
        # chromium is absent (the golden tests, CI).
        from playwright.sync_api import sync_playwright

        uri = "data:image/svg+xml;base64," + base64.b64encode(src.read_bytes()).decode("ascii")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.set_content(
                '<body style="margin:0;background:white">'
                f'<img src="{uri}" style="display:block;width:{_RENDER_PX}px;'
                f'height:{_RENDER_PX}px;object-fit:contain"></body>'
            )
            shot = page.locator("img").screenshot()
            browser.close()
        return Image.open(io.BytesIO(shot)).convert("RGB")
    with Image.open(src) as source:
        image = source.convert("RGBA")
        flat = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
    return ImageOps.contain(flat, (_RENDER_PX, _RENDER_PX))


def mask(rgb: Image.Image, mode: str) -> np.ndarray:
    """The drawing as a float mask in [0, 1] — ``ink`` for line art, ``edges`` for colour art."""
    if mode == "ink":
        return (np.asarray(rgb.convert("L")) / 255 < _INK_LUMA_MAX).astype(float)
    channels = np.asarray(rgb, dtype=float).transpose(2, 0, 1)
    # Per channel, then the strongest: colour art has seams luminance cannot see.
    magnitude = np.max(
        [np.hypot(ndimage.sobel(ch, axis=0), ndimage.sobel(ch, axis=1)) for ch in channels],
        axis=0,
    )
    scale = np.percentile(magnitude[magnitude > 0], _EDGE_NORM_PERCENTILE)
    return np.clip(magnitude / scale, 0, 1)


def layer(rgb: Image.Image, mode: str) -> Image.Image:
    """The finished layer: the blurred mask as alpha over the tint, cropped, capped, stepped."""
    soft = ndimage.gaussian_filter(mask(rgb, mode), _BLUR_SIGMA_PX)
    alpha = soft / soft.max() * _ALPHA_PEAK
    rows, cols = np.nonzero(alpha > _ALPHA_FLOOR)
    margin = round(_CROP_MARGIN * max(rows.max() - rows.min(), cols.max() - cols.min()))
    box = (
        max(cols.min() - margin, 0),
        max(rows.min() - margin, 0),
        min(cols.max() + margin + 1, alpha.shape[1]),
        min(rows.max() + margin + 1, alpha.shape[0]),
    )
    # Only the alpha is resampled; the colour is laid on afterwards so every visible
    # pixel stays exactly the tint instead of drifting at the translucent fringe.
    alpha_band = Image.fromarray(np.round(alpha * 255).astype(np.uint8), "L").crop(box)
    alpha_band.thumbnail((_LAYER_MAX_PX, _LAYER_MAX_PX), Image.Resampling.LANCZOS)
    step = 255 // (_ALPHA_LEVELS - 1)
    alpha_band = alpha_band.point(lambda pixel: round(pixel / step) * step)
    out = Image.new("RGBA", alpha_band.size, _TINT)
    out.putalpha(alpha_band)
    return out


def _get(url: str, params: dict | None = None) -> requests.Response:
    """One paced, identified GET — the single seam the golden tests stub."""
    response = requests.get(
        url, params=params, headers={"User-Agent": _USER_AGENT}, timeout=_REQUEST_TIMEOUT_S
    )
    response.raise_for_status()
    time.sleep(_REQUEST_PAUSE_S)
    return response


def _query(slug: str) -> str:
    """The default search text for a template: its label, sport-prefixed when league-bound."""
    template = shape_catalog()["templates"][slug]
    words = template["label"].removeprefix("The ").lower()
    if template["leagues"] == "all":
        return words
    sport = _SPORT_WORD[template["leagues"][0]]
    return words if sport in words else f"{sport} {words}"


def search_openclipart(query: str) -> list[dict]:
    """The first few hits for ``query`` as ``{"id", "title"}``, in openclipart's order."""
    page = _get(f"{_OPENCLIPART}/search/", params={"query": query}).text
    return [
        {"id": int(oca_id), "title": html.unescape(title)}
        for oca_id, title in _ARTWORK.findall(page)[:_CANDIDATES_PER_TEMPLATE]
    ]


def _review_sheets(out: Path, rows: dict) -> None:
    """One PNG per league group and page: a template per row, its candidates across."""
    templates = shape_catalog()["templates"]
    groups: dict[str, list[str]] = {}
    for slug in templates:
        if slug in rows:
            leagues = templates[slug]["leagues"]
            group = "all" if leagues == "all" else "+".join(leagues)
            groups.setdefault(group, []).append(slug)
    for stale in out.glob("sheet-*.png"):
        stale.unlink()
    font = ImageFont.load_default(size=_SHEET_LABEL_PT)
    width = _CAND_LABEL_PX + _CANDIDATES_PER_TEMPLATE * _CAND_TILE_PX
    for group, slugs in groups.items():
        for page, start in enumerate(range(0, len(slugs), _CAND_ROWS_PER_PAGE), 1):
            chunk = slugs[start : start + _CAND_ROWS_PER_PAGE]
            canvas = Image.new("RGB", (width, len(chunk) * _CAND_ROW_PX), "white")
            draw = ImageDraw.Draw(canvas)
            for r, slug in enumerate(chunk):
                top = r * _CAND_ROW_PX
                draw.text((8, top + 8), slug, fill="black", font=font)
                draw.text((8, top + 28), rows[slug]["query"], fill=GRAY, font=font)
                for c, hit in enumerate(rows[slug]["candidates"]):
                    left = _CAND_LABEL_PX + c * _CAND_TILE_PX
                    try:
                        with Image.open(out / "thumbs" / f"{hit['id']}.png") as thumb:
                            thumb = thumb.convert("RGBA")
                            thumb.thumbnail((_THUMB_PX, _THUMB_PX))
                            canvas.paste(thumb, (left, top), thumb)
                    except UnidentifiedImageError:
                        # A few old ids answer 200 with an empty body instead of a 404.
                        draw.text((left, top + 8), "no preview", fill=GRAY, font=font)
                    caption = f"{hit['id']}  {hit['title'][:_CAND_CAPTION_CHARS]}"
                    draw.text((left, top + _THUMB_PX + 6), caption, fill="black", font=font)
            canvas.save(out / f"sheet-{group}-{page}.png")


def _write_layer(
    kept: Path, slug: str, mode: str, source_url: str, artist: str, licence: str
) -> None:
    """Process the source already under ``sources/`` into ``<slug>.png``; record its row."""
    out = ART_DIR / f"{slug}.png"
    layer(rasterize(kept), mode).save(out, optimize=True)

    manifest_path = ART_DIR / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        manifest = {"version": 1, "templates": {}}
    manifest["templates"][slug] = {
        "file": out.name,
        "source": f"sources/{kept.name}",
        "source_url": source_url,
        "artist": artist,
        "licence": licence,
        "mode": mode,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with Image.open(out) as saved:
        click.echo(
            f"{slug}: {saved.width}x{saved.height} {mode} {out.stat().st_size / 1024:.0f} KiB"
        )


_SLUG_CHOICE = click.Choice(sorted(shape_catalog()["templates"]))


@click.group()
def constellation_art() -> None:
    """Clip art into constellation layers (docs/handoffs/constellation-art.md)."""


@constellation_art.command()
@click.argument("slugs", nargs=-1, type=_SLUG_CHOICE, metavar="[SLUG]...")
@click.option(
    "--out",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Review folder for candidates.json, thumbs/ and the sheets — keep it out of the repo.",
)
@click.option("--query", help="Search text for one SLUG in place of its label.")
def search(slugs: tuple[str, ...], out: Path, query: str | None) -> None:
    """Ask openclipart for candidates per template (all of them by default)."""
    if query and len(slugs) != 1:
        raise click.UsageError("--query goes with exactly one SLUG.")
    slugs = slugs or tuple(shape_catalog()["templates"])
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    path = out / "candidates.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))["templates"]
    except FileNotFoundError:
        rows = {}
    for slug in tqdm(slugs, desc="openclipart", unit="template"):
        text = query or _query(slug)
        hits = search_openclipart(text)
        for hit in hits:
            thumb = _get(f"{_OPENCLIPART}/image/{_THUMB_FETCH_PX}px/{hit['id']}").content
            (out / "thumbs" / f"{hit['id']}.png").write_bytes(thumb)
        rows[slug] = {"query": text, "candidates": hits}
        rows = {name: rows[name] for name in shape_catalog()["templates"] if name in rows}
        path.write_text(
            json.dumps({"version": 1, "templates": rows}, indent=2) + "\n", encoding="utf-8"
        )
    _review_sheets(out, rows)
    click.echo(f"{path}: {len(rows)} templates")


@constellation_art.command()
@click.argument("slug", type=_SLUG_CHOICE, metavar="SLUG")
@click.argument("oca_id", type=int, metavar="ID")
@click.option(
    "--mode",
    required=True,
    type=click.Choice(["ink", "edges"]),
    help="ink: line art on a light ground; edges: filled colour art.",
)
def pick(slug: str, oca_id: int, mode: str) -> None:
    """Fetch openclipart drawing ID and process it into ``<slug>.png``."""
    download = _get(f"{_OPENCLIPART}/download/{oca_id}")
    if "/download/" not in download.url:
        # An unknown id is answered with the site's logo, not a 404.
        raise click.BadParameter(f"openclipart has no drawing {oca_id}", param_hint="ID")
    kept = ART_DIR / "sources" / Path(download.url).name
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(download.content)
    detail = _get(f"{_OPENCLIPART}/detail/{oca_id}")
    artist = _ARTIST.search(detail.text).group(1)
    _write_layer(kept, slug, mode, detail.url, artist, _OPENCLIPART_LICENCE)


@constellation_art.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--slug",
    required=True,
    type=_SLUG_CHOICE,
    metavar="SLUG",
    help="The template this layer belongs to.",
)
@click.option(
    "--mode",
    required=True,
    type=click.Choice(["ink", "edges"]),
    help="ink: line art on a light ground; edges: filled colour art.",
)
@click.option("--source-url", required=True, help="Where the art came from (its file page).")
@click.option("--artist", required=True, help="As the source credits it.")
@click.option("--licence", required=True, help="As the source states it, e.g. CC0.")
def process(src: Path, slug: str, mode: str, source_url: str, artist: str, licence: str) -> None:
    """Process one source image into ``<slug>.png`` and record where it came from."""
    kept = ART_DIR / "sources" / src.name
    kept.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != kept.resolve():
        shutil.copy2(src, kept)
    _write_layer(kept, slug, mode, source_url, artist, licence)


@constellation_art.command()
@click.option(
    "--out",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the sheet — keep it out of the repo.",
)
def sheet(out: Path) -> None:
    """Tile every processed layer over the page ground for the owner's review."""
    rows = json.loads((ART_DIR / "manifest.json").read_text(encoding="utf-8"))["templates"]
    n_cols, n_rows = min(len(rows), _SHEET_COLUMNS), -(-len(rows) // _SHEET_COLUMNS)
    canvas = Image.new("RGB", (n_cols * _SHEET_TILE_PX, n_rows * _SHEET_TILE_PX), _GROUND)
    draw = ImageDraw.Draw(canvas)
    dust = np.random.default_rng(_SHEET_DUST_SEED)
    for x, y in dust.integers(canvas.size, size=(_SHEET_DUST_PER_TILE * len(rows), 2)).tolist():
        draw.point((x, y), fill=GRAY)
    font = ImageFont.load_default(size=_SHEET_LABEL_PT)
    for i, (slug, row) in enumerate(rows.items()):
        left = (i % _SHEET_COLUMNS) * _SHEET_TILE_PX
        top = (i // _SHEET_COLUMNS) * _SHEET_TILE_PX
        with Image.open(ART_DIR / row["file"]) as image:
            offset = (
                left + (_SHEET_TILE_PX - image.width) // 2,
                top + (_SHEET_TILE_PX - image.height) // 2,
            )
            canvas.paste(image, offset, image)
        draw.text((left + 8, top + 6), slug, fill=GRAY, font=font)
    canvas.save(out)
    click.echo(f"{out}: {len(rows)} layers")


if __name__ == "__main__":
    constellation_art()
