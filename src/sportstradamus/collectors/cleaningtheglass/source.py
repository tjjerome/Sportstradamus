"""Cleaning the Glass source wiring for the collector framework.

Binds CTG's cookie-authenticated GET endpoints and NBA dated-snapshot routing
to the generic runner / date-based CLI builder. CTG authenticates by session
``Cookie`` only (no bearer), so :class:`AuthFields` names just the cookie +
user-agent keys.json slots. See ``docs/cleaningtheglass.md`` for capturing the
cookie and registering endpoints.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from datetime import date
from functools import partial
from pathlib import Path

from sportstradamus import data
from sportstradamus.collectors.auth import AuthFields, ResolvedAuth
from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.cli import Source
from sportstradamus.collectors.tabular import dated_path_for, parse_tabular_response
from sportstradamus.collectors.transport import CookieClient

CATALOG_PATH = Path(str(pkg_resources.files(data) / "config" / "cleaningtheglass_endpoints.json"))

_CTG_REFERER = "https://cleaningtheglass.com/"
_CTG_ORIGIN = "https://cleaningtheglass.com"

# NBA seasons are labelled by their starting year (the 2025-26 season is
# ``2025``). The label flips in September, before the October tip-off, so a
# preseason capture still stamps the upcoming season.
_NBA_SEASON_FLIP_MONTH = 9


def _make_client(resolved: ResolvedAuth, inter_request_sleep_s: float | None) -> CookieClient:
    return CookieClient(
        authorization=resolved.authorization,
        cookie=resolved.cookie,
        user_agent=resolved.user_agent,
        referer=_CTG_REFERER,
        origin=_CTG_ORIGIN,
        inter_request_sleep_s=inter_request_sleep_s,
    )


def _default_season() -> int:
    today = date.today()
    return today.year if today.month >= _NBA_SEASON_FLIP_MONTH else today.year - 1


def _default_context(season: int | None, _week: int | None) -> dict[str, int]:
    return {"season": season or _default_season(), "week": 0}


def _dispatch(client: CookieClient, spec: EndpointSpec, *, season: int, **_ignored):
    """GET (or POST) one CTG endpoint. Week / mode / use_cache are unused here."""
    params = spec.render_params(season=season)
    accept = "json" if spec.response_format == "json" else "text"
    if spec.method.upper() == "POST":
        return client.post(
            spec.url,
            json_body=spec.render_json_body(season=season),
            params=params or None,
            headers=spec.extra_headers,
            accept=accept,
        )
    return client.get(spec.url, params=params or None, headers=spec.extra_headers, accept=accept)


CTG_SOURCE = Source(
    name="ctg-fetch",
    help="Snapshot Cleaning the Glass NBA advanced tables to disk.",
    catalog_path=CATALOG_PATH,
    make_client=_make_client,
    default_context=_default_context,
    path_for=partial(dated_path_for, league="NBA", source_dir="cleaningtheglass"),
    dispatch=_dispatch,
    transform=parse_tabular_response,
    report_prefix="ctg_fetch",
    auth_fields=AuthFields(
        cookie="cleaningtheglass_cookie",
        user_agent="cleaningtheglass_user_agent",
    ),
    env_prefix="CLEANINGTHEGLASS",
    period_kind="date",
)
