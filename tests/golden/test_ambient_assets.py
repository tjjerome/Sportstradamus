"""Golden pins for the ambient-art manifest + its loader.

``dashboard.assets.ambient_css`` is the scar mechanism DESIGN.md §3 describes: a slot
renders its caller's fallback gradient unless it names a file that exists on disk. These
pins cover the file gate, the opacity ceiling, the malformed-manifest fail-loud contract,
and that the two wired surfaces (Tonight card, Receipts hero) still emit their original
background strings against the shipped (empty) manifest.
"""

from __future__ import annotations

import base64
import contextlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from sportstradamus.dashboard import assets, theme

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = (
    _REPO / "src" / "sportstradamus" / "data" / "assets" / "ambient" / "ambient_manifest.json"
)

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


def test_present_file_yields_data_uri_and_surface_overlay(tmp_path, monkeypatch):
    (tmp_path / "night.png").write_bytes(_TINY_PNG)
    path = tmp_path / "ambient_manifest.json"
    _write_manifest(
        path,
        {
            "demo": {
                **_BASE_SLOT,
                "file": "night.png",
                "attribution": "NASA",
                "source_url": "https://example.com/night.png",
            }
        },
    )
    monkeypatch.setattr(assets, "MANIFEST_PATH", path)
    css = assets.ambient_css("demo", _FALLBACK)
    assert css != _FALLBACK
    assert "data:image/png;base64," in css
    # opacity 0.14 -> overlay alpha 1 - 0.14 = 0.86, over the DESIGN §2 surface tone.
    assert "rgba(26,29,36,0.86)" in css


@pytest.mark.parametrize(
    ("break_", "needle"),
    [
        ("bad_version", "version"),
        ("missing_key", "missing keys"),
        ("opacity_too_high", "ceiling"),
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

    path = tmp_path / "ambient_manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(assets, "MANIFEST_PATH", path)
    with pytest.raises(ValueError, match=needle):
        assets.ambient_css("demo", _FALLBACK)


def test_tonight_card_wash_unchanged_with_shipped_manifest():
    """theme.py has no Streamlit import, so its build result is safe to import directly."""
    assert (
        assets.ambient_css("ambient_tonight", theme._TONIGHT_CARD_BG_FALLBACK)
        == theme._TONIGHT_CARD_BG_FALLBACK
    )
    assert theme._TONIGHT_CARD_BG_FALLBACK in theme.APP_CSS
    assert "__TONIGHT_CARD_BG__" not in theme.APP_CSS


def test_receipts_hero_unchanged_with_shipped_manifest():
    """Importing a surfaces/*.py script can raise past its early module-level constants
    (no live ScriptRunContext here) — sys.modules still holds whatever bound before that,
    same partial-import contract test_dashboard_no_archive_lock.py already relies on.
    """
    module_name = "sportstradamus.dashboard.surfaces.receipts"
    with contextlib.suppress(Exception):
        importlib.import_module(module_name)
    receipts = sys.modules[module_name]
    assert (
        assets.ambient_css("ambient_receipts_hero", receipts._HERO_BG_FALLBACK)
        == receipts._HERO_BG_FALLBACK
    )
    assert receipts._HERO_BG == receipts._HERO_BG_FALLBACK
