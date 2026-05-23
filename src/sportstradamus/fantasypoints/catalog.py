"""Endpoint catalog for the Fantasy Points Data Suite snapshotter.

The catalog is a JSON list at
``src/sportstradamus/data/config/fantasypoints_endpoints.json``. Each
entry describes one tool's XHR endpoint and how its response should be
written. ``params`` and ``json_body`` values may contain ``{week}`` /
``{season}`` placeholders that the runner substitutes at fetch time.

Adding an endpoint: capture a ``curl`` command in DevTools (Network →
right-click → Copy as cURL), then run
``poetry run fp-fetch import-curl <curl_file> --name <slug> --output-subdir <path>``.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sportstradamus import data

# Source of truth for the bundled catalog. Committed; humans extend it
# via the ``fp-fetch import-curl`` CLI rather than by hand-editing.
CATALOG_PATH = Path(str(pkg_resources.files(data) / "config" / "fantasypoints_endpoints.json"))

# File extension per supported response format.
_FORMAT_EXTENSIONS: dict[str, str] = {"json": ".json", "csv": ".csv", "html": ".html"}

# Template placeholders the runner substitutes at fetch time. We only
# touch a string if it actually contains one of these tokens so JSON
# values with stray ``{`` / ``}`` are left alone.
_TEMPLATE_TOKENS = ("{week}", "{season}")

ResponseFormat = Literal["json", "csv", "html"]


@dataclass
class EndpointSpec:
    """One Fantasy Points tool endpoint, plus where its snapshot lives."""

    name: str
    url: str
    output_subdir: str
    method: str = "GET"
    params: dict[str, str] | None = None
    json_body: dict | list | None = None
    extra_headers: dict[str, str] | None = None
    response_format: ResponseFormat = "json"
    weekly: bool = True
    notes: str = ""

    def render_params(self, *, season: int, week: int) -> dict[str, str]:
        """Resolve ``{season}`` / ``{week}`` placeholders in ``params``."""
        out: dict[str, str] = {}
        for k, v in (self.params or {}).items():
            out[k] = _substitute(v, season=season, week=week) if isinstance(v, str) else v
        return out

    def render_json_body(self, *, season: int, week: int) -> dict | list | None:
        """Resolve ``{season}`` / ``{week}`` placeholders inside the JSON body.

        Args:
            season: NFL season year substituted for ``{season}`` tokens.
            week: NFL week number substituted for ``{week}`` tokens.

        Returns:
            A deep copy of ``json_body`` with all template tokens replaced,
            or ``None`` if the spec has no body.
        """
        if self.json_body is None:
            return None
        return _substitute_in_tree(self.json_body, season=season, week=week)

    def output_path(self, *, base: Path, season: int, week: int) -> Path:
        """Compose the on-disk path for one snapshot.

        Weekly endpoints land under ``<base>/<season>/week_NN/``;
        season-long ones under ``<base>/<season>/season/``.
        """
        ext = _FORMAT_EXTENSIONS[self.response_format]
        leaf = f"{self.output_subdir}{ext}"
        if self.weekly:
            return base / str(season) / f"week_{week:02d}" / leaf
        return base / str(season) / "season" / leaf


# Field defaults that should not be persisted when they are empty —
# keeps the on-disk catalog compact and reviewable.
_OMIT_IF_EMPTY = ("params", "json_body", "extra_headers", "notes")


def load_catalog(path: Path | None = None) -> list[EndpointSpec]:
    """Read the catalog JSON into a list of :class:`EndpointSpec`."""
    path = path or CATALOG_PATH
    if not path.is_file():
        return []
    with path.open() as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of endpoint specs")
    return [EndpointSpec(**entry) for entry in raw]


def save_catalog(catalog: list[EndpointSpec], path: Path | None = None) -> None:
    """Write the catalog JSON, dropping empty optional fields."""
    path = path or CATALOG_PATH
    serialised = [_spec_to_dict(spec) for spec in catalog]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(serialised, f, indent=2)
        f.write("\n")


def _spec_to_dict(spec: EndpointSpec) -> dict:
    payload = dict(spec.__dict__)
    for key in _OMIT_IF_EMPTY:
        if not payload.get(key):
            payload.pop(key, None)
    return payload


def _substitute(value: str, *, season: int, week: int) -> str:
    if any(token in value for token in _TEMPLATE_TOKENS):
        return value.format(season=season, week=week)
    return value


def _substitute_in_tree(node, *, season: int, week: int):
    if isinstance(node, str):
        return _substitute(node, season=season, week=week)
    if isinstance(node, dict):
        return {k: _substitute_in_tree(v, season=season, week=week) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute_in_tree(v, season=season, week=week) for v in node]
    return node
