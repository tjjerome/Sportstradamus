"""Golden pins for the constellation-art processing tool.

``scripts/constellation_art.py`` turns one piece of clip art into the soft light-blue
layer a template will render beneath its stars (constellation-art lane, stage 1). The
pins cover the two masks, the layer contract (tint, alpha peak, crop, size cap), the
``process`` command's three outputs (layer, manifest row, source copy) and the ``sheet``
command. Every fixture is drawn in memory at the render size; nothing here launches
chromium, so the SVG path is accepted by eye on the contact sheet instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image, ImageDraw

from sportstradamus.dashboard.components.constellation_shapes import shape_catalog
from sportstradamus.scripts import constellation_art as art

_RUNNER = CliRunner()

_REPO = Path(__file__).resolve().parents[2]
_SHIPPED = _REPO / "src" / "sportstradamus" / "data" / "assets" / "constellations"
_SHIPPED_ROWS = json.loads((_SHIPPED / "manifest.json").read_text(encoding="utf-8"))["templates"]

# Brief §8: the committed image set passing this is a stop-and-ask, so the suite says so.
_SHIPPED_BUDGET_BYTES = 5 * 1024 * 1024


def _ring(box: tuple[int, int, int, int] = (200, 200, 700, 700)) -> Image.Image:
    """A black ring on white at the render size — line art in the ink sense."""
    image = Image.new("RGB", (art._RENDER_PX, art._RENDER_PX), "white")
    ImageDraw.Draw(image).ellipse(box, outline="black", width=10)
    return image


def _halves() -> Image.Image:
    """Two flat colours of near-equal luminance split down the middle."""
    image = Image.new("RGB", (art._RENDER_PX, art._RENDER_PX), (200, 40, 40))
    ImageDraw.Draw(image).rectangle((450, 0, 899, 899), fill=(40, 120, 40))
    return image


def _process(src: Path, slug: str, mode: str = "ink"):
    return _RUNNER.invoke(
        art.constellation_art,
        [
            "process",
            str(src),
            "--slug",
            slug,
            "--mode",
            mode,
            "--source-url",
            f"https://example.com/{slug}",
            "--artist",
            "Nobody",
            "--licence",
            "CC0",
        ],
        catch_exceptions=False,
    )


def test_ink_mask_keeps_only_dark_pixels():
    mask = art.mask(_ring(), "ink")
    assert mask.shape == (art._RENDER_PX, art._RENDER_PX)
    assert mask[450, 205] == 1.0, "the stroke is ink"
    assert mask[450, 450] == 0.0, "the hollow centre is ground"
    assert mask[10, 10] == 0.0


def test_edge_mask_peaks_on_the_seam_and_is_flat_inside():
    mask = art.mask(_halves(), "edges")
    assert mask[450, 449:451].max() == 1.0, "a seam luminance can barely see"
    assert mask[450, 100] == 0.0 and mask[450, 800] == 0.0, "flat colour is not an edge"


def test_layer_is_tinted_translucent_and_cropped_with_a_margin():
    out = art.layer(_ring(), "ink")
    assert out.mode == "RGBA"
    pixels = np.asarray(out)
    visible = pixels[pixels[..., 3] > 0]
    assert (visible[:, :3] == art._TINT).all(), "every visible pixel is the theme blue"
    # 0.55 * 255 = 140.25, and the LANCZOS resample wobbles the peak by a level or so.
    assert 135 <= pixels[..., 3].max() <= 141
    # The ring's ink box is 500 px inside a 900 px canvas: cropped, with the margin kept.
    assert 510 < max(out.size) < 600


def test_layer_caps_the_longest_side():
    out = art.layer(_ring((50, 50, 850, 850)), "ink")
    assert max(out.size) == art._LAYER_MAX_PX


def test_process_writes_layer_manifest_row_and_source_copy(tmp_path, monkeypatch):
    assets = tmp_path / "constellations"
    monkeypatch.setattr(art, "ASSETS_DIR", assets)
    src = tmp_path / "ring.png"
    _ring().save(src)

    result = _process(src, "the-ring")

    assert result.exit_code == 0, result.output
    with Image.open(assets / "the-ring.png") as layer:
        assert layer.mode == "RGBA"
    assert (assets / "sources" / "ring.png").read_bytes() == src.read_bytes()
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "version": 1,
        "templates": {
            "the-ring": {
                "file": "the-ring.png",
                "source": "sources/ring.png",
                "source_url": "https://example.com/the-ring",
                "artist": "Nobody",
                "licence": "CC0",
                "mode": "ink",
            }
        },
    }


def test_process_merges_rows_sorted_by_slug(tmp_path, monkeypatch):
    assets = tmp_path / "constellations"
    monkeypatch.setattr(art, "ASSETS_DIR", assets)
    src = tmp_path / "ring.png"
    _ring().save(src)

    _process(src, "the-ring")
    _process(src, "the-bat", mode="edges")

    rows = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))["templates"]
    assert list(rows) == ["the-bat", "the-ring"]
    assert rows["the-ring"]["mode"] == "ink" and rows["the-bat"]["mode"] == "edges"


def test_process_reruns_from_the_sources_folder(tmp_path, monkeypatch):
    assets = tmp_path / "constellations"
    monkeypatch.setattr(art, "ASSETS_DIR", assets)
    src = assets / "sources" / "ring.png"
    src.parent.mkdir(parents=True)
    _ring().save(src)

    result = _process(src, "the-ring")

    assert result.exit_code == 0, result.output
    assert (assets / "the-ring.png").is_file()


def test_process_rejects_a_slug_off_the_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "ASSETS_DIR", tmp_path / "constellations")
    src = tmp_path / "ring.png"
    _ring().save(src)

    result = _process(src, "the-unicorn")

    assert result.exit_code == 2
    assert "the-unicorn" in result.output


def test_sheet_tiles_every_layer(tmp_path, monkeypatch):
    assets = tmp_path / "constellations"
    monkeypatch.setattr(art, "ASSETS_DIR", assets)
    src = tmp_path / "ring.png"
    _ring().save(src)
    _process(src, "the-ring")
    _process(src, "the-bat")
    out = tmp_path / "sheet.png"

    result = _RUNNER.invoke(
        art.constellation_art, ["sheet", "--out", str(out)], catch_exceptions=False
    )

    assert result.exit_code == 0, result.output
    with Image.open(out) as sheet:
        assert sheet.size == (2 * art._SHEET_TILE_PX, art._SHEET_TILE_PX), "two tiles, one row"


@pytest.mark.parametrize(("slug", "row"), sorted(_SHIPPED_ROWS.items()))
def test_shipped_row_is_complete_and_its_layer_is_web_sized(slug, row):
    assert set(row) == {"file", "source", "source_url", "artist", "licence", "mode"}
    assert all(row.values()), f"{slug} has an empty provenance field"
    assert slug in shape_catalog()["templates"], f"{slug} is not a template"
    assert row["mode"] in ("ink", "edges")
    assert (_SHIPPED / row["source"]).is_file(), f"{slug} names a missing source"
    with Image.open(_SHIPPED / row["file"]) as layer:
        assert layer.mode == "RGBA"
        assert max(layer.size) <= art._LAYER_MAX_PX
        pixels = np.asarray(layer)
        assert (pixels[pixels[..., 3] > 0][:, :3] == art._TINT).all(), f"{slug} is off-tint"


def test_shipped_set_stays_under_the_budget():
    total = sum(path.stat().st_size for path in _SHIPPED.rglob("*") if path.is_file())
    assert total <= _SHIPPED_BUDGET_BYTES
