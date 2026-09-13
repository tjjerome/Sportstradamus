"""Turn clip art into the soft light-blue layer a constellation renders beneath its stars.

One piece of licence-free clip art in; one web-sized RGBA PNG out under
``data/assets/constellations/``, its provenance row in ``manifest.json``, and a copy of
the input under ``sources/`` so the run reproduces offline. The recipe is the stage-0
proof of concept in ``docs/handoffs/constellation-art.md``: rasterize at 900 px, mask
the ink (line art) or the edges (filled colour art), gaussian-blur, tint the theme's
light blue, bake the alpha, crop. Dev-side only — the layers are committed, so
production never runs this.

Usage
-----
    poetry run python -m sportstradamus.scripts.constellation_art process SRC \\
        --slug the-bat --mode edges --source-url URL --artist "Gerald G" --licence CC0
    poetry run python -m sportstradamus.scripts.constellation_art sheet --out /tmp/sheet.png
"""

from __future__ import annotations

import base64
import io
import json
import shutil
from importlib import resources
from pathlib import Path

import click
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

from sportstradamus import data
from sportstradamus.dashboard.components.constellation_shapes import shape_catalog
from sportstradamus.dashboard.theme import GRAY, SEQUENTIAL_COLORS

ASSETS_DIR = Path(str(resources.files(data) / "assets" / "constellations"))

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

# The theme's light blue (#7FAAE8) is the locked tint; no new hex.
_TINT = ImageColor.getrgb(SEQUENTIAL_COLORS[3])

# Sheet ground mirrors .streamlit/config.toml backgroundColor; theme.py names no token.
_GROUND = "#0E1117"
_SHEET_COLUMNS = 5  # one row the owner can scan on a single screen
_SHEET_TILE_PX = 640  # the layer cap plus room for the label
_SHEET_DUST_PER_TILE = 60  # enough scattered stars to judge a layer against a sky
_SHEET_DUST_SEED = 0  # the same sky every run, so two sheets differ on the layers alone
_SHEET_LABEL_PT = 14  # the size-less default font is an unreadable bitmap


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
    """The finished layer: the blurred mask as alpha over the constant tint, cropped, capped."""
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
    out = Image.new("RGBA", alpha_band.size, _TINT)
    out.putalpha(alpha_band)
    return out


@click.group()
def constellation_art() -> None:
    """Clip art into constellation layers (docs/handoffs/constellation-art.md)."""


@constellation_art.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--slug",
    required=True,
    type=click.Choice(sorted(shape_catalog()["templates"])),
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
    kept = ASSETS_DIR / "sources" / src.name
    kept.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != kept.resolve():
        shutil.copy2(src, kept)
    out = ASSETS_DIR / f"{slug}.png"
    layer(rasterize(kept), mode).save(out, optimize=True)

    manifest_path = ASSETS_DIR / "manifest.json"
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


@constellation_art.command()
@click.option(
    "--out",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the sheet — keep it out of the repo.",
)
def sheet(out: Path) -> None:
    """Tile every processed layer over the page ground for the owner's review."""
    rows = json.loads((ASSETS_DIR / "manifest.json").read_text(encoding="utf-8"))["templates"]
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
        with Image.open(ASSETS_DIR / row["file"]) as image:
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
