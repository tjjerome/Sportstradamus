"""Source-neutral endpoint catalog for the collector framework.

A catalog is a JSON list of endpoint specs. Each entry describes one
upstream XHR endpoint and how its response should be written. ``params``
and ``json_body`` values may contain ``{season}`` / ``{week}`` / ``{date}``
placeholders that a source substitutes at fetch time via
:meth:`EndpointSpec.render_params` / :meth:`EndpointSpec.render_json_body`.

The catalog file path is a per-source concern — pass it in explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Template placeholders a source may substitute at fetch time. We only
# touch a string if it actually contains one of these tokens so JSON
# values with stray ``{`` / ``}`` are left alone.
_TEMPLATE_TOKENS = ("{season}", "{week}", "{date}")

ResponseFormat = Literal["json", "csv", "html"]


@dataclass
class EndpointSpec:
    """One upstream tool endpoint, plus where its snapshot lives."""

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

    def render_params(self, **context: object) -> dict[str, str]:
        """Resolve template placeholders in ``params`` from ``context``."""
        out: dict[str, str] = {}
        for k, v in (self.params or {}).items():
            out[k] = _substitute(v, context) if isinstance(v, str) else v
        return out

    def render_json_body(self, **context: object) -> dict | list | None:
        """Resolve template placeholders inside the JSON body from ``context``.

        Returns a deep copy of ``json_body`` with all present template tokens
        replaced, or ``None`` if the spec has no body.
        """
        if self.json_body is None:
            return None
        return _substitute_in_tree(self.json_body, context)


# Field defaults that should not be persisted when they are empty —
# keeps the on-disk catalog compact and reviewable.
_OMIT_IF_EMPTY = ("params", "json_body", "extra_headers", "notes")


def load_catalog(path: Path) -> list[EndpointSpec]:
    """Read the catalog JSON at ``path`` into a list of :class:`EndpointSpec`."""
    if not path.is_file():
        return []
    with path.open() as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of endpoint specs")
    return [EndpointSpec(**entry) for entry in raw]


def save_catalog(catalog: list[EndpointSpec], path: Path) -> None:
    """Write the catalog JSON to ``path``, dropping empty optional fields."""
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


def _substitute(value: str, context: dict[str, object]) -> str:
    if any(token in value for token in _TEMPLATE_TOKENS):
        return value.format(**context)
    return value


def _substitute_in_tree(node, context: dict[str, object]):
    if isinstance(node, str):
        return _substitute(node, context)
    if isinstance(node, dict):
        return {k: _substitute_in_tree(v, context) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute_in_tree(v, context) for v in node]
    return node
