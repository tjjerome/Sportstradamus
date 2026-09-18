"""Fantasy Points source wiring for the generic collector framework.

Binds the FP-specific pieces — cookie auth against
``fantasypointsdata.com``, the season/week query parameters, and the NFL
season/week parquet routing — to the source-neutral runner and CLI
builder in :mod:`sportstradamus.collectors`.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from sportstradamus import data
from sportstradamus.collectors.auth import AuthFields, ResolvedAuth
from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.cli import Source
from sportstradamus.collectors.fantasypoints.session import renew_session
from sportstradamus.collectors.fantasypoints.transform import (
    parquet_path_for_spec,
    parse_table_response,
)
from sportstradamus.collectors.fantasypoints.verify import verify_catalog
from sportstradamus.collectors.transport import CookieClient

Mode = Literal["weekly", "season_to_date", "postseason"]
DEFAULT_MODE: Mode = "weekly"

# Source of truth for the bundled catalog. Committed; humans extend it via
# the ``fp-fetch import-curl`` CLI rather than by hand-editing.
CATALOG_PATH = Path(str(pkg_resources.files(data) / "config" / "fantasypoints_endpoints.json"))

# The FP data app's own origin — sent as Referer/Origin so requests look
# like they came from the logged-in SPA.
_FP_REFERER = "https://fantasypointsdata.com/"
_FP_ORIGIN = "https://fantasypointsdata.com"

# NFL regular season is 18 weeks. The default-week inference clamps to this
# range; the user can always pass --week explicitly.
NFL_REGULAR_SEASON_WEEKS = 18

# A "NFL season" labelled by year Y starts in September of year Y. Before
# July we're still in the prior year's playoff/offseason tail.
_SEASON_FLIP_MONTH = 7

# NFL Week 1 opens the Thursday after Labor Day — the first Monday in
# September, which has weekday() == 0. Anchoring on Sep 1's first Tuesday
# instead drifts a week whenever Sep 1 is itself a Tuesday.
_MONDAY = 0


def _default_season() -> int:
    today = date.today()
    return today.year if today.month >= _SEASON_FLIP_MONTH else today.year - 1


def _default_week(season: int) -> int:
    """Return the most recently completed NFL week of ``season``.

    Each week's window opens on the Tuesday two days before its Thursday
    kickoff, so flooring the elapsed weeks since Week 1's Tuesday names the
    week whose games have all finished — the one worth snapshotting. Before
    Week 1 completes (and after Week 18) the result clamps into range; pass
    ``--week`` to reach a week outside it.
    """
    labor_day = date(season, 9, 1)
    while labor_day.weekday() != _MONDAY:
        labor_day += timedelta(days=1)
    delta_days = (date.today() - (labor_day + timedelta(days=1))).days
    return max(1, min(NFL_REGULAR_SEASON_WEEKS, delta_days // 7))


def period_params(*, season: int, week: int, mode: Mode = DEFAULT_MODE) -> dict[str, str]:
    """Return the season/week query parameters for one (season, week, mode).

    The API filters on comma-separated week lists, with regular-season and
    postseason weeks in separate parameters. ``regWeeks=`` with an empty
    value returns zero rows, which is how a postseason request excludes the
    regular season.

    Kept public because :mod:`verify` needs the same mapping to state what
    a parquet should contain.
    """
    if mode == "postseason":
        return {"seasons": str(season), "regWeeks": "", "postWeeks": str(week)}
    weeks = range(1, week + 1) if mode == "season_to_date" else (week,)
    return {"seasons": str(season), "regWeeks": ",".join(str(w) for w in weeks)}


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


def _dispatch(
    client: CookieClient,
    spec: EndpointSpec,
    *,
    season: int,
    week: int,
    mode: Mode = DEFAULT_MODE,
    use_cache: bool = True,
) -> dict | list | str | bytes:
    """GET one endpoint spec with its catalog filters plus the period window.

    The catalog entry carries only the filters that define *what* the tool
    is — position set, offence/defence grain, and any situational slice.
    The season and week parameters are layered on here so a mode change
    doesn't require touching 52 catalog entries.

    ``use_cache`` is accepted for CLI symmetry; the new API has no cache
    control, so it is ignored.
    """
    params = {
        **spec.render_params(season=season, week=week),
        **period_params(season=season, week=week, mode=mode),
    }
    return client.get(spec.url, params=params, headers=spec.extra_headers, accept="json")


def _render_request_body(spec: EndpointSpec, **_context) -> None:
    """No request bodies on this API; the run report records the URL instead.

    Required by the week-centric command builders, which call it
    unconditionally.
    """


FP_SOURCE = Source(
    name="fp-fetch",
    help="Snapshot Fantasy Points Data Suite tools to disk.",
    catalog_path=CATALOG_PATH,
    make_client=_make_client,
    default_context=_default_context,
    path_for=parquet_path_for_spec,
    dispatch=_dispatch,
    render_request_body=_render_request_body,
    transform=parse_table_response,
    verify_fn=verify_catalog,
    renew_auth=renew_session,
    report_prefix="fp_fetch",
    auth_fields=AuthFields(
        cookie="fantasypoints_cookie",
        user_agent="fantasypoints_user_agent",
    ),
    env_prefix="FANTASYPOINTS",
    modes=("weekly", "season_to_date", "postseason"),
    default_mode=DEFAULT_MODE,
    has_backfill=True,
    backfill_end_week=NFL_REGULAR_SEASON_WEEKS,
)
