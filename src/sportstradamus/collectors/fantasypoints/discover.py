"""Auto-discover Fantasy Points Data Suite tool endpoints from the registry.

FP's SPA fetches a tool registry at ``POST /v2/ds/all/tools`` on login.
The response shape is ``{content: {tables: {values: [{property,
context, isPublished, isPrivate, roles, requiresCharting, ...}]}}}``
— enough to build :class:`EndpointSpec` entries for every published
tool without the user copying a curl per tool.

For each registry entry we expand ``context[]`` (player / team /
opponent) into one URL each — the SPA picks the URL prefix from
``routeContext`` + ``routeContextTarget``:

- ``player`` → ``/v2/ds/{league}/tools/player/{slug}/values``
- ``team``   → ``/v2/ds/{league}/tools/team/{slug}/values``
  (team-side offensive view — no ``/offense/`` segment; that's the
  bare path the SPA hits for team-context tools)
- ``opponent`` → ``/v2/ds/{league}/tools/team/defense/{slug}/values``
  (team-side defensive view; one team's row per opponent it faced)

``other`` and any unknown context are skipped (no observed URL
pattern). The body shape follows the v2 ``/values`` endpoint
contract observed via DevTools — ``tableProperty``,
``routeContextTarget``, integer ``weeks: {"REG": [N]}`` block,
``filterMatch.game.season.eq``, and the per-tool ``requires*`` /
``requiredRoles`` flags pulled straight from the registry entry.
Per-call substitution (week → int, season → int, weeks block per
mode) is handled by :mod:`body_substitute` at request time.
"""

from __future__ import annotations

import re

from sportstradamus.collectors.catalog import EndpointSpec

# Registry endpoint — same across leagues; the per-tool URLs use the
# ``league`` segment (``nfl`` here). The body filters out tools that
# aren't published yet so we don't catalog drafts.
REGISTRY_URL = "https://data.fantasypoints.com/v2/ds/all/tools"
REGISTRY_BODY: dict = {"filters": {"filter": {"tableIsPublished": {"eq": True}}}}

# Per-tool URL templates — observed from real DevTools curls.
# All FP v2 tool data is fetched from a ``/values`` sub-endpoint.
# Team-OFFENSE tools use the bare ``/team/{slug}/values`` path — there
# is no ``/team/offense/`` segment; the SPA's URL bar shows
# ``/team/{slug}`` for offense views and ``/team/defense/{slug}`` for
# the defensive (opponent) view of the same tool. Sending requests to
# the non-existent ``/team/offense/{slug}/values`` path returns empty.
_BASE_URL = "https://data.fantasypoints.com/v2/ds/{league}/tools"
_PLAYER_URL_TEMPLATE = _BASE_URL + "/player/{slug}/values"
_TEAM_URL_TEMPLATE = _BASE_URL + "/team/{slug}/values"
_OPPONENT_URL_TEMPLATE = _BASE_URL + "/team/defense/{slug}/values"

# Contexts FP exposes through the routeContext URL segment. ``other``
# appears in the registry but maps to no observed URL — skip it so
# discover doesn't generate 404s.
_KNOWN_CONTEXTS: tuple[str, ...] = ("player", "team", "opponent")

# Default sub-directory layout: snapshots land at
# ``{context}/{slug}.json`` — kept for backward compatibility with
# the legacy raw-snapshot path; the parquet routing in
# :mod:`transform` reads name+url+output_subdir.
_OUTPUT_SUBDIR_TEMPLATE = "{context}/{slug_underscore}"

# Sentinel strings the runtime body builder
# (:func:`body_substitute.substitute_runtime`) recognises and
# replaces with ``int`` values at call time. FP rejects string-typed
# week / season here, but the catalog file is JSON — so we store
# sentinels and substitute at call time.
_WEEK_INT_SENTINEL = "__WEEK_INT__"
_SEASON_INT_SENTINEL = "__SEASON_INT__"


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
        out.extend(
            _specs_for_table(prop, contexts, table, league=league, existing_names=existing_names)
        )
    return out


def _specs_for_table(
    prop: str, contexts: list, table: dict, *, league: str, existing_names: set[str]
) -> list[EndpointSpec]:
    """Mutates ``existing_names`` so later tables in the same walk dedupe against
    names this one just claimed.
    """
    specs: list[EndpointSpec] = []
    for ctx in contexts:
        if ctx not in _KNOWN_CONTEXTS:
            continue
        spec = _make_spec(prop, ctx, league=league, table=table)
        if spec.name in existing_names:
            continue
        specs.append(spec)
        existing_names.add(spec.name)
    return specs


def _should_include(table: dict, *, include_private: bool) -> bool:
    if not table.get("isPublished", False):
        return False
    return include_private or not table.get("isPrivate", False)


def _make_spec(prop: str, context: str, *, league: str, table: dict) -> EndpointSpec:
    slug = _camel_to_kebab(prop)
    slug_underscore = slug.replace("-", "_")
    return EndpointSpec(
        name=f"{context}_{slug_underscore}",
        url=_url_for(context, league=league, slug=slug),
        output_subdir=_OUTPUT_SUBDIR_TEMPLATE.format(
            context=context, slug_underscore=slug_underscore
        ),
        method="POST",
        json_body=_default_body(prop, context, table=table),
        response_format="json",
        weekly=True,
    )


def _url_for(context: str, *, league: str, slug: str) -> str:
    if context == "player":
        return _PLAYER_URL_TEMPLATE.format(league=league, slug=slug)
    if context == "team":
        return _TEAM_URL_TEMPLATE.format(league=league, slug=slug)
    if context == "opponent":
        return _OPPONENT_URL_TEMPLATE.format(league=league, slug=slug)
    raise ValueError(f"Unknown context: {context}")


def _default_body(prop: str, context: str, *, table: dict) -> dict:
    """Build the v2 ``/values`` request body for one (tool, context) pair.

    The body shape was reverse-engineered from real DevTools curls,
    one per context. Per-tool flags (``requiresCharting``,
    ``requiresPlayByPlay``, ``requiredRoles``, etc.) come from the
    registry entry. The ``weeks`` block uses a single-int sentinel
    that :mod:`body_substitute` rewrites per call based on mode.

    Filter shape varies by context — the SPA injects different
    defaults per context and getting them wrong returns 0 rows:

    - **player**: ``filterMatch.isGamePlayed=true`` (drops scheduled
      games) plus a ``filterPlay.teamRoles`` selector that scopes to
      offensive plays. Tool-specific position / qualifier filters
      narrow further but aren't required.
    - **team** (offense view): empty ``filterMatch`` (apart from
      season) and empty ``filterPlay``. Sending the player-style
      filters here makes FP return 0 rows.
    - **opponent** (defense view): empty ``filterPlay`` here too.
      Tool-specific filters (e.g. ``playType``) are sent by the SPA
      per tool, but no universal default exists.

    Season is the one universal ``filterMatch`` key across all
    contexts (sentinelised here, rewritten per call).
    """
    route_context, route_target, model_context, grouping = _context_routing(context)
    return {
        "context": {
            "tableProperty": prop,
            "grouping": grouping,
            "dualContext": False,
            "modelContext": model_context,
            "routeContext": route_context,
            "routeContextTarget": route_target,
            "weeks": {"REG": [_WEEK_INT_SENTINEL]},
            "filterMatch": _filter_match(context),
            "filterPlay": _filter_play(context),
            "filterResult": {},
            "qualifiers": {},
            "splits": {},
            "disabled": {"filterMatch": {}, "filterPlay": {}, "filterResult": {}},
            "requiresSchedule": None,
            "requiresCharting": bool(table.get("requiresCharting", False)),
            "requiresPlayByPlay": bool(table.get("requiresPlayByPlay", False)),
            "requiresPreTotals": bool(table.get("requiresPreTotals", False)),
            "requiresPostTotals": bool(table.get("requiresPostTotals", False)),
            "requiresMultiplePipelines": table.get("requiresMultiplePipelines"),
            "requiredRoles": list(table.get("roles") or []),
            "isFreePreview": bool(table.get("isFreePreview", False)),
        },
        "useCache": True,
        "flatten": True,
        "debug": False,
    }


# Universal play-context filter for player-context tools only — the
# SPA omits this for team and opponent tools, and sending it on those
# returns 0 rows. ``$$play.team.roles`` is FP's literal placeholder
# for the team's role on each play and must appear in the array.
_PLAY_ROLE_PLACEHOLDER = {"$ifNull": ["$$play.team.roles", []]}


def _filter_match(context: str) -> dict:
    """Return the ``filterMatch`` defaults for one context.

    Season is the one universal key. Player-context adds an
    ``isGamePlayed`` guard so scheduled-but-not-played games don't
    inflate aggregates; team / opponent tools rely on FP to apply
    that filter server-side.
    """
    base = {"game.season": {"eq": _SEASON_INT_SENTINEL}}
    if context == "player":
        return {"isGamePlayed": {"eq": True}, **base}
    return base


def _filter_play(context: str) -> dict:
    """Return the ``filterPlay`` defaults for one context.

    Only player-context tools need the ``teamRoles`` selector; team
    and opponent tools either have no play-level default or use a
    per-tool filter that we can't generalise here.
    """
    if context == "player":
        return {"teamRoles": {"in": ["offense", _PLAY_ROLE_PLACEHOLDER]}}
    return {}


def _context_routing(context: str) -> tuple[str, str, str, str]:
    """Return ``(routeContext, routeContextTarget, modelContext, grouping)``.

    Mirrors the per-context URL choice in :func:`_url_for`:

    - player → routes to ``/player/...`` with grouping by playerId.
    - team   → routes to ``/team/{slug}/values`` (team's offensive view —
      no ``/offense/`` segment), grouping by teamId.
    - opponent → routes to ``/team/defense/...`` (team's defensive
      view), grouping by teamId, ``modelContext=opponent`` so FP
      returns defensive aggregates.
    """
    if context == "player":
        return "player", "player", "player", "$player.playerId"
    if context == "team":
        return "team", "offense", "team", "$team.teamId"
    if context == "opponent":
        return "team", "defense", "opponent", "$team.teamId"
    raise ValueError(f"Unknown context: {context}")


# Inserts a ``-`` at every lowercase-then-uppercase boundary
# (``passingAdvanced`` -> ``passing-Advanced``) before lowercasing.
# The negative-lookbehind on start-of-string keeps a leading capital
# from gaining a stray leading ``-``.
_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_kebab(name: str) -> str:
    return _CAMEL_BOUNDARY.sub("-", name).lower()
