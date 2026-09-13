"""Golden pins for the constellation-art processing tool.

``scripts/constellation_art.py`` turns one piece of clip art into the soft light-blue
layer a template will render beneath its stars (constellation-art lane, stage 1). The
pins cover the two masks, the layer contract (tint, alpha peak, crop, size cap), the
``process`` command's three outputs (layer, manifest row, source copy), the ``sheet`` command,
and the stage-2 openclipart pair: ``search`` (query → candidates, thumbnails, review
sheet) and ``pick`` (one openclipart id → layer + provenance). Every fixture is drawn in
memory at the render size and every HTTP call goes through the stubbed ``_get`` seam;
nothing here launches chromium, so the SVG path is accepted by eye on the contact sheet.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

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


def _ring_source(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Patch ``ASSETS_DIR`` under ``tmp_path`` and save the ring fixture as its source.

    Returns ``(assets, src)``.
    """
    assets = tmp_path / "constellations"
    monkeypatch.setattr(art, "ASSETS_DIR", assets)
    src = tmp_path / "ring.png"
    _ring().save(src)
    return assets, src


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


def test_layer_stores_alpha_in_a_few_even_steps():
    levels = np.unique(np.asarray(art.layer(_ring(), "ink"))[..., 3])
    assert len(levels) <= art._ALPHA_LEVELS
    assert (levels % (255 // (art._ALPHA_LEVELS - 1)) == 0).all(), "0, 17, 34 … 255"


def test_process_writes_layer_manifest_row_and_source_copy(tmp_path, monkeypatch):
    assets, src = _ring_source(tmp_path, monkeypatch)

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
    assets, src = _ring_source(tmp_path, monkeypatch)

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
    _, src = _ring_source(tmp_path, monkeypatch)

    result = _process(src, "the-unicorn")

    assert result.exit_code == 2
    assert "the-unicorn" in result.output


def test_sheet_tiles_every_layer(tmp_path, monkeypatch):
    _, src = _ring_source(tmp_path, monkeypatch)
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
    # Downloads are gitignored (megabytes); a missing one must be one URL away.
    if not (_SHIPPED / row["source"]).is_file():
        assert row["source_url"].startswith("https://"), f"{slug}: source missing, no https URL"
    with Image.open(_SHIPPED / row["file"]) as layer:
        assert layer.mode == "RGBA"
        assert max(layer.size) <= art._LAYER_MAX_PX
        pixels = np.asarray(layer)
        assert (pixels[pixels[..., 3] > 0][:, :3] == art._TINT).all(), f"{slug} is off-tint"


def test_shipped_set_stays_under_the_budget():
    # The committed set is the layers and the manifest; sources/ holds gitignored downloads
    # beside a few KB-scale compositions.
    total = sum(path.stat().st_size for path in _SHIPPED.iterdir() if path.is_file())
    assert total <= _SHIPPED_BUDGET_BYTES


_SEARCH_HTML = """<h2 class="text-center"> 2 clipart for "hockey stick" <small> (Page 1 of 1) </small></h2>
<div class="gallery"> <div class="artwork"> <a href="/detail/74437/hockey-stick-ball">
<img src="/image/800px/74437" alt="Hockey Stick &amp; Ball" /> </a> </div>
<div class="artwork"> <a href="/detail/35545/hockey-stick"> <img src="/image/800px/35545" alt="Hockey Stick" /> </a> </div> </div>"""

_DETAIL_HTML = (
    """<title>Ring - Openclipart</title> <p>by <a href="/artist/Nobody">Nobody</a> </p>"""
)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _fake_openclipart(monkeypatch, hits: int = 2):
    """Route every ``_get`` to canned openclipart pages; return the URLs asked for."""
    asked = []
    gallery = "".join(
        f'<div class="artwork"> <a href="/detail/{1000 + i}/x"> '
        f'<img src="/image/800px/{1000 + i}" alt="Item {i}" /> </a> </div>'
        for i in range(hits)
    )
    thumb = _png_bytes(Image.new("RGBA", (art._THUMB_PX, 180), (0, 0, 0, 0)))

    def fake_get(url, params=None):
        asked.append((url, params))
        if "/search/" in url:
            return SimpleNamespace(text=_SEARCH_HTML if hits == 2 else gallery, url=url)
        if "/image/" in url:
            return SimpleNamespace(content=thumb, url=url)
        if "/download/" in url:
            return SimpleNamespace(content=_png_bytes(_ring()), url=f"{url}/ring.png")
        if "/detail/" in url:
            return SimpleNamespace(text=_DETAIL_HTML, url=url)
        raise AssertionError(url)

    monkeypatch.setattr(art, "_get", fake_get)
    return asked


@pytest.mark.parametrize(
    ("slug", "query"),
    [
        ("the-hoop", "basketball hoop"),
        ("the-goalposts", "football goalposts"),
        ("the-catchers-mask", "baseball catcher's mask"),
        ("the-crossed-sticks", "hockey crossed sticks"),
        ("the-hourglass", "hourglass"),
        ("the-basketball", "basketball"),
    ],
)
def test_default_query_is_the_label_prefixed_by_the_sport(slug, query):
    assert art._query(slug) == query


def test_search_parses_openclipart_hits_in_page_order(monkeypatch):
    _fake_openclipart(monkeypatch)

    assert art.search_openclipart("hockey stick") == [
        {"id": 74437, "title": "Hockey Stick & Ball"},
        {"id": 35545, "title": "Hockey Stick"},
    ]


def test_search_keeps_only_the_first_few_hits(monkeypatch):
    _fake_openclipart(monkeypatch, hits=art._CANDIDATES_PER_TEMPLATE + 3)

    hits = art.search_openclipart("anything")

    assert [hit["id"] for hit in hits] == list(range(1000, 1000 + art._CANDIDATES_PER_TEMPLATE))


def test_search_command_writes_candidates_thumbs_and_one_sheet_per_group(tmp_path, monkeypatch):
    asked = _fake_openclipart(monkeypatch)
    out = tmp_path / "review"

    result = _RUNNER.invoke(
        art.constellation_art,
        ["search", "--out", str(out), "the-bat", "the-hoop"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    queries = [params["query"] for url, params in asked if "/search/" in url]
    assert queries == ["baseball bat", "basketball hoop"]
    thumbs = [url for url, _ in asked if "/image/" in url]
    assert thumbs and all(f"/image/{art._THUMB_FETCH_PX}px/" in url for url in thumbs)
    assert art._THUMB_FETCH_PX > art._THUMB_PX, "fetched above the tile size, never below"
    rows = json.loads((out / "candidates.json").read_text(encoding="utf-8"))["templates"]
    assert rows["the-hoop"] == {
        "query": "basketball hoop",
        "candidates": [
            {"id": 74437, "title": "Hockey Stick & Ball"},
            {"id": 35545, "title": "Hockey Stick"},
        ],
    }
    assert (out / "thumbs" / "74437.png").is_file()
    assert sorted(path.name for path in out.glob("sheet-*.png")) == [
        "sheet-MLB-1.png",
        "sheet-NBA+WNBA-1.png",
    ]
    with Image.open(out / "sheet-MLB-1.png") as sheet:
        assert sheet.size == (
            art._CAND_LABEL_PX + art._CANDIDATES_PER_TEMPLATE * art._CAND_TILE_PX,
            art._CAND_ROW_PX,
        )


def test_search_sheet_survives_a_thumbnail_openclipart_did_not_render(tmp_path, monkeypatch):
    real = _fake_openclipart(monkeypatch)
    routed = art._get

    def with_one_bad_thumb(url, params=None):
        response = routed(url, params)
        if url.endswith("/74437"):
            return SimpleNamespace(content=b"<html>not an image</html>", url=url)
        return response

    monkeypatch.setattr(art, "_get", with_one_bad_thumb)

    result = _RUNNER.invoke(
        art.constellation_art, ["search", "--out", str(tmp_path), "the-bat"], catch_exceptions=False
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "sheet-MLB-1.png").is_file()
    assert real, "the stub was consulted"


def test_search_command_takes_a_query_override_for_one_slug(tmp_path, monkeypatch):
    asked = _fake_openclipart(monkeypatch)

    result = _RUNNER.invoke(
        art.constellation_art,
        ["search", "--out", str(tmp_path), "--query", "goal post", "the-goalposts"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert [params["query"] for url, params in asked if "/search/" in url] == ["goal post"]
    rows = json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8"))["templates"]
    assert rows["the-goalposts"]["query"] == "goal post"


def test_search_command_merges_into_an_existing_candidates_file(tmp_path, monkeypatch):
    _fake_openclipart(monkeypatch)
    for slug in ("the-bat", "the-hoop"):
        _RUNNER.invoke(
            art.constellation_art, ["search", "--out", str(tmp_path), slug], catch_exceptions=False
        )

    rows = json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8"))["templates"]

    assert list(rows) == ["the-hoop", "the-bat"], "both rows kept, in catalog order"


def test_pick_downloads_the_file_and_records_openclipart_provenance(tmp_path, monkeypatch):
    assets = tmp_path / "constellations"
    monkeypatch.setattr(art, "ASSETS_DIR", assets)
    _fake_openclipart(monkeypatch)

    result = _RUNNER.invoke(
        art.constellation_art, ["pick", "the-bat", "77", "--mode", "ink"], catch_exceptions=False
    )

    assert result.exit_code == 0, result.output
    with Image.open(assets / "the-bat.png") as layer:
        assert layer.mode == "RGBA"
    assert (assets / "sources" / "ring.png").read_bytes() == _png_bytes(_ring())
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["templates"]["the-bat"] == {
        "file": "the-bat.png",
        "source": "sources/ring.png",
        "source_url": "https://openclipart.org/detail/77",
        "artist": "Nobody",
        "licence": "CC0",
        "mode": "ink",
    }


def test_pick_refuses_an_id_openclipart_answers_with_its_logo(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "ASSETS_DIR", tmp_path / "constellations")
    monkeypatch.setattr(
        art,
        "_get",
        lambda url, params=None: SimpleNamespace(
            content=b"<svg/>", url="https://openclipart.org/assets/images/openclipart-logo-2019.svg"
        ),
    )

    result = _RUNNER.invoke(art.constellation_art, ["pick", "the-bat", "0", "--mode", "ink"])

    assert result.exit_code == 2
    assert "0" in result.output and not (tmp_path / "constellations").exists()
