"""Auto-discover Fantasy Points Data Suite tool endpoints from the registry.

FP's SPA fetches a tool registry at ``POST /v2/ds/all/tools`` on login.
The response shape is ``{content: {tables: {values: [{property,
context, isPublished, isPrivate, roles, ...}]}}}`` — enough to build
:class:`EndpointSpec` entries for every published tool without the
user copying a curl per tool.

For each registry entry we expand ``context[]`` (player / team /
opponent) into one URL each — the SPA picks the URL prefix from
``routeContext``, so ``passingBasic`` with context ``["player",
"team", "opponent"]`` becomes three endpoints. ``other`` and any
unknown context are skipped (no URL pattern observed).

The body template is taken from the larger of the two observed
shapes (player/passing-advanced); the smaller (team/line-matchups)
shape is a subset, and the server tolerates the extra keys.
"""

from __future__ import annotations

import re

from sportstradamus.fantasypoints.catalog import EndpointSpec

# Registry endpoint — same across leagues; the per-tool URLs use the
# ``league`` segment (``nfl`` here). The body filters out tools that
# aren't published yet so we don't catalog drafts.
REGISTRY_URL = "https://data.fantasypoints.com/v2/ds/all/tools"
REGISTRY_BODY: dict = {"filters": {"filter": {"tableIsPublished": {"eq": True}}}}

# URL template for the per-tool calls. The ``context`` segment is the
# routeContext (player/team/opponent); ``slug`` is the property in
# kebab-case (``lineMatchups`` -> ``line-matchups``).
_TOOL_URL_TEMPLATE = "https://data.fantasypoints.com/v2/ds/{league}/tools/{context}/{slug}"

# Contexts FP exposes through the routeContext URL segment. ``other``
# appears in the registry but maps to no observed URL — skip it so
# discover doesn't generate 404s. Users can add ``other`` endpoints
# manually via import-curl if they find one.
_KNOWN_CONTEXTS: tuple[str, ...] = ("player", "team", "opponent")

# Default sub-directory layout: snapshots land at
# ``{context}/{slug}.json`` — keeps player/team/opponent versions of
# the same tool side by side and matches the on-disk hierarchy already
# used for the hand-imported entries.
_OUTPUT_SUBDIR_TEMPLATE = "{context}/{slug_underscore}"


def expand_registry(
    registry: dict,
    *,
    league: str = "nfl",
    existing_names: set[str] | None = None,
    include_private: bool = False,
) -> list[EndpointSpec]:
    """Walk the registry payload, yield one :class:`EndpointSpec` per (tool, context).

    Args:
        registry: The decoded JSON body returned by ``POST /v2/ds/all/tools``.
        league: League segment in the per-tool URL (``"nfl"``).
        existing_names: Names already in the user's catalog; entries
            that would collide are skipped so user customisation is
            preserved across discover runs.
        include_private: When ``False`` (default), skip tools flagged
            ``isPrivate: true`` (debug / VIP-only / under-construction
            tables that 4xx for most accounts).

    Returns:
        A list of fresh :class:`EndpointSpec` ready to append to the
        catalog. Order matches the registry's ``values`` order, with
        each tool's contexts in :data:`_KNOWN_CONTEXTS` order.
    """
    existing_names = existing_names or set()
    tables = (
        registry.get("content", {}).get("tables", {}).get("values", [])
        if isinstance(registry, dict)
        else []
    )
    out: list[EndpointSpec] = []
    for table in tables:
        if not _should_include(table, include_private=include_private):
            continue
        prop = table.get("property")
        contexts = table.get("context") or []
        if not prop or not contexts:
            continue
        for ctx in contexts:
            if ctx not in _KNOWN_CONTEXTS:
                continue
            spec = _make_spec(prop, ctx, league=league)
            if spec.name in existing_names:
                continue
            out.append(spec)
            existing_names.add(spec.name)
    return out


def _should_include(table: dict, *, include_private: bool) -> bool:
    if not table.get("isPublished", False):
        return False
    return include_private or not table.get("isPrivate", False)


def _make_spec(prop: str, context: str, *, league: str) -> EndpointSpec:
    slug = _camel_to_kebab(prop)
    slug_underscore = slug.replace("-", "_")
    return EndpointSpec(
        name=f"{context}_{slug_underscore}",
        url=_TOOL_URL_TEMPLATE.format(league=league, context=context, slug=slug),
        output_subdir=_OUTPUT_SUBDIR_TEMPLATE.format(
            context=context, slug_underscore=slug_underscore
        ),
        method="POST",
        json_body=_default_body(context),
        response_format="json",
        weekly=True,
    )


def _default_body(context: str) -> dict:
    """Per-tool request body inferred from observed FP calls.

    Two observed shapes:

    - ``team/line-matchups``: minimal — ``context.grouping``,
      ``context.routeContext``, ``filters.{week,season}``, ``useCache``.
    - ``player/passing-advanced``: fuller — adds ``dualContext``,
      ``modelContext``, ``flatten``, ``isInitial``, ``withValues``,
      ``compress``; no ``filters`` block (server returns all weeks).

    The fuller shape is a strict superset of the minimal one for the
    fields they share, so we send the fuller shape for all tools. If
    a particular tool needs ``filters``, re-import its curl to
    override the auto-generated entry.
    """
    return {
        "context": {
            "grouping": f"${context}.{context}Id",
            "dualContext": False,
            "modelContext": context,
            "routeContext": context,
        },
        "useCache": True,
        "flatten": True,
        "isInitial": True,
        "withValues": True,
        "compress": False,
    }


# Inserts a ``-`` at every lowercase-then-uppercase boundary
# (``passingAdvanced`` -> ``passing-Advanced``) before lowercasing.
# The negative-lookbehind on start-of-string keeps a leading capital
# from gaining a stray leading ``-``.
_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_kebab(name: str) -> str:
    return _CAMEL_BOUNDARY.sub("-", name).lower()
