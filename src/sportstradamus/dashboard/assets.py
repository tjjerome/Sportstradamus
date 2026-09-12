"""Ambient-art loader (DESIGN.md §3) — the scar mechanism.

Ambient art is optional dressing over a token gradient: a slot with no file renders its
caller's fallback gradient byte-identical to today. A slot that names a real file on
disk renders the image, layered under a solid overlay of the surface color so it never
shows above the slot's manifest opacity. The manifest
(``data/assets/ambient/ambient_manifest.json``) is the owner's tuning surface, not a
build artifact — same mtime-cached, validate-on-load contract as
``constellation_shapes.py``. Licensing is the owner's call, made before a file lands;
``attribution`` / ``source_url`` are free notes the loader never reads.
"""

from __future__ import annotations

import base64
import functools
import importlib.resources as pkg_resources
import json
from pathlib import Path

from sportstradamus import data

MANIFEST_PATH = Path(
    str(pkg_resources.files(data) / "assets" / "ambient" / "ambient_manifest.json")
)

# DESIGN.md §2 secondaryBackgroundColor #1A1D24, as r,g,b for the overlay below — the
# same surface tone every wired hero/card background already ends its gradient on.
_SURFACE_RGB = "26,29,36"

# DESIGN.md §3 ambient opacity ceiling — static ambient art never exceeds this.
_OPACITY_CEILING = 0.20

_REQUIRED_SLOT_KEYS = frozenset({"file", "opacity", "placement"})

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _validate(manifest: dict) -> dict:
    """Raise on a manifest whose shape or values a hand-edit could break."""
    if manifest.get("version") != 1:
        raise ValueError(f"ambient manifest: unsupported version {manifest.get('version')!r}")
    for slot, entry in manifest["slots"].items():
        missing = _REQUIRED_SLOT_KEYS - entry.keys()
        if missing:
            raise ValueError(f"ambient manifest slot {slot!r}: missing keys {sorted(missing)}")
        if not 0 <= entry["opacity"] <= _OPACITY_CEILING:
            raise ValueError(
                f"ambient manifest slot {slot!r}: opacity {entry['opacity']} exceeds the "
                f"DESIGN.md §3 ceiling of {_OPACITY_CEILING}"
            )
    return manifest


@functools.lru_cache(maxsize=8)
def _load(path: Path, mtime_ns: int) -> dict:
    """Parse + validate the manifest at ``path``.

    ``mtime_ns`` is not read in the body — it is half the cache key, so an owner edit
    invalidates the entry (same contract as ``constellation_shapes.py:_load``).
    """
    return _validate(json.loads(path.read_text(encoding="utf-8")))


def _manifest() -> dict:
    return _load(MANIFEST_PATH, MANIFEST_PATH.stat().st_mtime_ns)


def _data_uri(path: Path) -> str:
    mime = _MIME_BY_SUFFIX[path.suffix.lower()]
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def ambient_css(slot: str, fallback_gradient: str) -> str:
    """CSS ``background`` value for ``slot``.

    Returns ``fallback_gradient`` unchanged unless the manifest slot names a ``file``
    that exists under ``data/assets/ambient/``. A present file is embedded as a base64
    data URI (Streamlit serves no arbitrary static files) under a solid surface-color
    overlay, so the image never shows above its manifest ``opacity``.

    Args:
        slot: Key into ``ambient_manifest.json``'s ``"slots"`` object.
        fallback_gradient: The token-gradient CSS this slot renders when empty.

    Returns:
        A CSS ``background`` property value.
    """
    entry = _manifest()["slots"][slot]
    if not entry["file"]:
        return fallback_gradient
    image_path = MANIFEST_PATH.parent / entry["file"]
    if not image_path.exists():
        return fallback_gradient
    overlay_alpha = round(1 - entry["opacity"], 2)
    overlay = f"rgba({_SURFACE_RGB},{overlay_alpha})"
    return (
        f"linear-gradient({overlay},{overlay}),url({_data_uri(image_path)}) center/cover no-repeat"
    )
