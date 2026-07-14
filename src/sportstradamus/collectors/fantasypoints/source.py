"""Fantasy Points source wiring for the generic collector framework.

Binds the FP-specific pieces — cookie/bearer auth against
``data.fantasypoints.com``, the v2 ``/values`` body substitution, and the
NFL season/week parquet routing — to the source-neutral runner and CLI
builder in :mod:`sportstradamus.collectors`.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from datetime import date, timedelta
from pathlib import Path

from sportstradamus import data
from sportstradamus.collectors.auth import AuthFields, ResolvedAuth
from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.cli import Source
from sportstradamus.collectors.fantasypoints.body_substitute import (
    DEFAULT_MODE,
    Mode,
    substitute_runtime,
)
from sportstradamus.collectors.fantasypoints.transform import (
    parquet_path_for_spec,
    parse_table_response,
)
from sportstradamus.collectors.fantasypoints.verify import verify_catalog
from sportstradamus.collectors.transport import CookieClient

# Source of truth for the bundled catalog. Committed; humans extend it via
# the ``fp-fetch import-curl`` CLI rather than by hand-editing.
CATALOG_PATH = Path(str(pkg_resources.files(data) / "config" / "fantasypoints_endpoints.json"))

# The FP data app's own origin — sent as Referer/Origin so requests look
# like they came from the logged-in SPA.
_FP_REFERER = "https://data.fantasypoints.com/"
_FP_ORIGIN = "https://data.fantasypoints.com"

# NFL regular season is 18 weeks. The default-week inference clamps to this
# range; the user can always pass --week explicitly.
NFL_REGULAR_SEASON_WEEKS = 18

# A "NFL season" labelled by year Y starts in September of year Y. Before
# July we're still in the prior year's playoff/offseason tail.
_SEASON_FLIP_MONTH = 7

# NFL Week 1 starts Thursday after Labor Day; a Tuesday cutoff is a
# close-enough boundary for the default. Tuesdays have weekday() == 1.
_TUESDAY = 1


def _make_client(resolved: ResolvedAuth, inter_request_sleep_s: float | None) -> CookieClient:
    return CookieClient(
        authorization=resolved.authorization,
        cookie=resolved.cookie,
        user_agent=resolved.user_agent,
        referer=_FP_REFERER,
        origin=_FP_ORIGIN,
        inter_request_sleep_s=inter_request_sleep_s,
    )


def _default_context(season: int | None, week: int | None) -> dict[str, int]:
    resolved_season = season or _default_season()
    return {"season": resolved_season, "week": week or _default_week(resolved_season)}


def _path_for(spec: EndpointSpec, *, season: int, week: int, mode: str) -> Path:
    return parquet_path_for_spec(spec, season=season, week=week, mode=mode)


def _render_request_body(
    spec: EndpointSpec, *, season: int, week: int, mode: Mode, use_cache: bool
):
    """Render the body captured in the run report (mirrors :func:`_dispatch`).

    The legacy ``{week}`` / ``{season}`` string substitution runs first
    (hand-imported entries), then the v2 runtime substitution
    (discover-generated entries). Kept separate from :func:`_dispatch` so
    the report shows exactly what was (or would be) sent even when routing
    or the fetch fails before the wire call.
    """
    legacy_body = spec.render_json_body(season=season, week=week)
    if legacy_body is None:
        return None
    rendered = substitute_runtime(legacy_body, season=season, week=week, mode=mode)
    if not use_cache and isinstance(rendered, dict) and "useCache" in rendered:
        rendered["useCache"] = False
    return rendered


def _dispatch(
    client: CookieClient,
    spec: EndpointSpec,
    *,
    season: int,
    week: int,
    mode: Mode = DEFAULT_MODE,
    use_cache: bool = True,
) -> dict | list | str | bytes:
    """Call the right verb on the client for one endpoint spec.

    For POST requests we apply the legacy ``{week}`` / ``{season}`` string
    substitution first (covers hand-imported entries), then the v2 runtime
    substitution from :mod:`body_substitute` (covers discover-generated
    entries — int sentinels and the per-mode ``context.weeks`` rewrite).
    The two are independent: bodies that use one shape are pass-through
    under the other.

    When ``use_cache=False``, the top-level ``useCache`` field on a v2 body
    is flipped to ``False`` so FP recomputes instead of returning a cached
    response.
    """
    # style: allow-complexity — per-spec verb dispatcher: GET/POST selection,
    # accept-token mapping, and POST body substitution. Residual CC is the
    # request-shape branching, not nested logic.
    params = spec.render_params(season=season, week=week)
    if spec.response_format == "json":
        accept = "json"
    elif spec.response_format in ("csv", "html"):
        accept = "text"
    else:
        accept = "bytes"
    method = spec.method.upper()
    if method == "POST":
        json_body = spec.render_json_body(season=season, week=week)
        if json_body is not None:
            json_body = substitute_runtime(json_body, season=season, week=week, mode=mode)
            if not use_cache and isinstance(json_body, dict) and "useCache" in json_body:
                json_body["useCache"] = False
        return client.post(
            spec.url,
            json_body=json_body,
            params=params or None,
            headers=spec.extra_headers,
            accept=accept,
        )
    if method == "GET":
        return client.get(
            spec.url,
            params=params or None,
            headers=spec.extra_headers,
            accept=accept,
        )
    raise ValueError(f"Unsupported method {method!r} for endpoint {spec.name!r}")


def _default_season() -> int:
    today = date.today()
    return today.year if today.month >= _SEASON_FLIP_MONTH else today.year - 1


def _default_week(season: int) -> int:
    today = date.today()
    season_start = date(season, 9, 1)
    while season_start.weekday() != _TUESDAY:
        season_start += timedelta(days=1)
    delta_days = (today - season_start).days
    return max(1, min(NFL_REGULAR_SEASON_WEEKS, (delta_days // 7) + 1))


FP_SOURCE = Source(
    name="fp-fetch",
    help="Snapshot Fantasy Points Data Suite tools to disk.",
    catalog_path=CATALOG_PATH,
    make_client=_make_client,
    default_context=_default_context,
    path_for=_path_for,
    dispatch=_dispatch,
    render_request_body=_render_request_body,
    transform=parse_table_response,
    verify_fn=verify_catalog,
    report_prefix="fp_fetch",
    auth_fields=AuthFields(
        authorization="fantasypoints_authorization",
        cookie="fantasypoints_cookie",
        user_agent="fantasypoints_user_agent",
    ),
    env_prefix="FANTASYPOINTS",
    modes=("weekly", "season_to_date", "postseason"),
    default_mode=DEFAULT_MODE,
    has_backfill=True,
    backfill_end_week=NFL_REGULAR_SEASON_WEEKS,
)
