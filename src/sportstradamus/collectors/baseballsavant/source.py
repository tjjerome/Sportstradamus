"""Baseball Savant source wiring for the collector framework.

Savant / Statcast leaderboards are public, so this source has NO keys.json
entry and no ``refresh-auth`` command. It reuses the shared :class:`Scrape`
header-rotation client — the same one that already fetches savant's game feed
and affinity CSVs, and that unblocks hosts (moneypuck, savant) which answer a
default user-agent with an HTML bot-block page. See ``docs/baseballsavant.md``
for pinning the leaderboard URLs.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from datetime import date
from functools import partial
from pathlib import Path

from sportstradamus import data
from sportstradamus.collectors.auth import ResolvedAuth
from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.cli import Source
from sportstradamus.collectors.report import build_url
from sportstradamus.collectors.tabular import dated_path_for, parse_tabular_response
from sportstradamus.helpers.scraping import Scrape

CATALOG_PATH = Path(str(pkg_resources.files(data) / "config" / "baseballsavant_endpoints.json"))


def _make_client(_resolved: ResolvedAuth, _inter_request_sleep_s: float | None) -> Scrape:
    """Public source — a fresh header-rotating Scrape (auth args unused)."""
    return Scrape()


def _default_context(season: int | None, _week: int | None) -> dict[str, int]:
    # MLB seasons sit within one calendar year, so the year is the label.
    return {"season": season or date.today().year, "week": 0}


def _dispatch(scrape: Scrape, spec: EndpointSpec, *, season: int, **_ignored):
    """Fetch one savant endpoint via header-rotated GET. Week / mode unused.

    CSV leaderboards (``response_format="csv"``) come back as a DataFrame from
    :meth:`Scrape.get_csv`; JSON endpoints return the decoded dict.
    """
    params = spec.render_params(season=season)
    if spec.response_format == "csv":
        return scrape.get_csv(build_url(spec.url, params))
    return scrape.get(spec.url, params=params or None)


SAVANT_SOURCE = Source(
    name="savant-fetch",
    help="Snapshot Baseball Savant (Statcast) MLB leaderboards to disk.",
    catalog_path=CATALOG_PATH,
    make_client=_make_client,
    default_context=_default_context,
    path_for=partial(dated_path_for, league="MLB", source_dir="baseballsavant"),
    dispatch=_dispatch,
    transform=parse_tabular_response,
    report_prefix="savant_fetch",
    period_kind="date",
)
