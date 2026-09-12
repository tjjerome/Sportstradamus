"""Golden pins for the ambient-art manifest + its loader.

``dashboard.assets.ambient_css`` is the scar mechanism DESIGN.md §3 describes: a slot
renders its caller's fallback gradient unless it names a file that exists on disk. These
pins cover the file gate, the opacity ceiling, the placement → CSS geometry map, the
import-time downscale of oversized originals, the malformed-manifest fail-loud contract,
and that the shipped manifest's files exist and reach the three wired surfaces (Tonight
cards, Receipts hero, Games hero).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from sportstradamus.dashboard import assets, theme

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = (
    _REPO / "src" / "sportstradamus" / "data" / "assets" / "ambient" / "ambient_manifest.json"
)
_SURFACES = _REPO / "src" / "sportstradamus" / "dashboard" / "surfaces"

# Smallest legal PNG (1x1, transparent) — stands in for a real ambient asset without
# committing a binary fixture to the repo.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

_FALLBACK = "radial-gradient(fallback)"

_BASE_SLOT = {
    "file": None,
    "opacity": 0.14,
    "placement": "card-background",
    "attribution": None,
    "source_url": None,
}


def _write_manifest(path: Path, slots: dict) -> None:
    path.write_text(json.dumps({"version": 1, "slots": slots}))


def test_null_file_slot_returns_fallback_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "ambient_manifest.json"
    _write_manifest(path, {"demo": dict(_BASE_SLOT)})
    monkeypatch.setattr(assets, "MANIFEST_PATH", path)
    assert assets.ambient_css("demo", _FALLBACK) == _FALLBACK


def test_shipped_manifest_slots_stay_under_the_opacity_ceiling():
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    for slot, entry in manifest["slots"].items():
        assert 0 <= entry["opacity"] <= 0.20, f"{slot} exceeds the DESIGN.md §3 ceiling"


def test_named_file_missing_on_disk_returns_fallback(tmp_path, monkeypatch):
    path = tmp_path / "ambient_manifest.json"
    _write_manifest(path, {"demo": {**_BASE_SLOT, "file": "night.png"}})
    monkeypatch.setattr(assets, "MANIFEST_PATH", path)
    assert assets.ambient_css("demo", _FALLBACK) == _FALLBACK


@pytest.mark.parametrize(
    ("placement", "geometry"),
    [
        ("card-background", "0 0/100% auto repeat-y"),
        ("hero-background", "center/cover no-repeat"),
    ],
)
def test_present_file_yields_data_uri_under_surface_overlay_per_placement(
    tmp_path, monkeypatch, placement, geometry
):
    (tmp_path / "night.png").write_bytes(_TINY_PNG)
    path = tmp_path / "ambient_manifest.json"
    _write_manifest(
        path,
        {
            "demo": {
                **_BASE_SLOT,
                "file": "night.png",
                "placement": placement,
                "attribution": "NASA",
                "source_url": "https://example.com/night.png",
            }
        },
    )
    monkeypatch.setattr(assets, "MANIFEST_PATH", path)
    css = assets.ambient_css("demo", _FALLBACK)
    # opacity 0.14 -> overlay alpha 1 - 0.14 = 0.86, over the DESIGN §2 surface tone; a
    # file under the embed ceiling ships byte-for-byte in its own format.
    assert css.startswith("linear-gradient(rgba(26,29,36,0.86),rgba(26,29,36,0.86)),url(")
    assert "data:image/png;base64," in css
    assert css.endswith(f") {geometry}")


def test_wide_file_is_downscaled_to_webp_before_embedding(tmp_path, monkeypatch):
    Image.new("RGB", (3200, 200), "navy").save(tmp_path / "wide.png")
    path = tmp_path / "ambient_manifest.json"
    _write_manifest(path, {"demo": {**_BASE_SLOT, "file": "wide.png"}})
    monkeypatch.setattr(assets, "MANIFEST_PATH", path)
    css = assets.ambient_css("demo", _FALLBACK)
    uri = css.split("url(", 1)[1].split(")", 1)[0]
    assert uri.startswith("data:image/webp;base64,")
    embedded = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert embedded.size == (1600, 100)


@pytest.mark.parametrize(
    ("break_", "needle"),
    [
        ("bad_version", "version"),
        ("missing_key", "missing keys"),
        ("opacity_too_high", "ceiling"),
        ("unknown_placement", "placement"),
    ],
)
def test_malformed_manifest_fails_loud(tmp_path, monkeypatch, break_, needle):
    manifest = {"version": 1, "slots": {"demo": dict(_BASE_SLOT)}}
    if break_ == "bad_version":
        manifest["version"] = 2
    elif break_ == "missing_key":
        del manifest["slots"]["demo"]["placement"]
    elif break_ == "opacity_too_high":
        manifest["slots"]["demo"]["opacity"] = 0.9
    elif break_ == "unknown_placement":
        manifest["slots"]["demo"]["placement"] = "sidebar"

    path = tmp_path / "ambient_manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(assets, "MANIFEST_PATH", path)
    with pytest.raises(ValueError, match=needle):
        assets.ambient_css("demo", _FALLBACK)


def test_shipped_manifest_names_files_that_exist():
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    for slot, entry in manifest["slots"].items():
        assert entry["file"], f"{slot} has no file — the slot renders its fallback"
        assert (_MANIFEST_PATH.parent / entry["file"]).is_file(), f"{slot} names a missing file"


def test_tonight_cards_embed_the_shipped_night_sky():
    """theme.py has no Streamlit import, so its build result is safe to import directly."""
    assert theme.TONIGHT_CARD_HAS_ART
    assert "__TONIGHT_CARD_BG__" not in theme.APP_CSS
    assert "__TONIGHT_CARD_MUTED_BG__" not in theme.APP_CSS
    # Lit and muted cards share the same tiled image so the column reads as one sky.
    assert theme.APP_CSS.count(theme._TONIGHT_CARD_BG) == 2
    assert theme._TONIGHT_CARD_BG.endswith(") 0 0/100% auto repeat-y")


@pytest.mark.parametrize(
    ("surface", "slot"),
    [("receipts", "ambient_receipts_hero"), ("games", "ambient_games_hero")],
)
def test_heroes_embed_the_shipped_nebula(surface, slot):
    """The page scripts don't import in bare mode (games.py is dropped from sys.modules
    when its body raises), so pin the wiring in source and the slot's render separately.
    """
    source = (_SURFACES / f"{surface}.py").read_text(encoding="utf-8")
    assert f'_HERO_BG = ambient_css("{slot}", _HERO_BG_FALLBACK)' in source
    css = assets.ambient_css(slot, _FALLBACK)
    assert css.startswith("linear-gradient(rgba(26,29,36,0.84),rgba(26,29,36,0.84)),url(")
    assert "data:image/webp;base64," in css
    assert css.endswith(") center/cover no-repeat")
