"""Cleaning the Glass source wiring for the collector framework.

Binds CTG's cookie-authenticated stats pages and NBA dated-snapshot routing to the generic
runner / date-based CLI builder. CTG authenticates by session ``Cookie`` only (no bearer), so
:class:`AuthFields` names just the cookie + user-agent keys.json slots.

CTG has no table endpoint: ``/stats/players`` is an HTML page whose inline script passes every
table to ``vuePlayers(...)`` as JSON, and the page's CSV button builds its file in the browser
from that data. See ``docs/cleaningtheglass.md`` for capturing the cookie and what each tool
holds.

After the player table is written, ``after_write`` also rebuilds the season's NBA roster seed,
``player_data/NBA/nba_players_<season>.csv``, in the layout of that CSV export. ``StatsNBA``
reads the file as its player seed.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from functools import partial
from math import floor
from pathlib import Path

import click
import pandas as pd

from sportstradamus import data
from sportstradamus.collectors import tabular
from sportstradamus.collectors.auth import AuthFields, ResolvedAuth
from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.cli import Source
from sportstradamus.collectors.transport import CollectorAuthError, CookieClient
from sportstradamus.helpers.io import _atomic_write_csv

CATALOG_PATH = Path(str(pkg_resources.files(data) / "config" / "cleaningtheglass_endpoints.json"))

_CTG_REFERER = "https://cleaningtheglass.com/"
_CTG_ORIGIN = "https://cleaningtheglass.com"

# NBA seasons are labelled by their starting year (the 2025-26 season is
# ``2025``). The label flips in September, before the October tip-off, so a
# preseason capture still stamps the upcoming season.
_NBA_SEASON_FLIP_MONTH = 9

# The inline script call that carries the page's tables. A gated or restyled page would lack it.
_PAGE_CALL = "vuePlayers("

# A column only that tool's table has. Tables are told apart by it rather than by their order
# in the call, which CTG is free to change.
_TABLE_KEY_COL = {
    "player_stats": "ptile_usage",
    "player_onoff": "ptile_diff_oer",
}

# Whitespace and commas up to the start of the next call argument; no match at the closing paren.
_NEXT_ARG = re.compile(r"[\s,]*(?=[^\s,)])")

# The stats in CTG's own player CSV export, after Player/Age/Team/Pos/MIN/MPG, as
# (label, table column, scale, decimals, suffix). Each is preceded by its "<label> Rank",
# the position percentile on a 0-100 scale.
_SEED_STATS = (
    ("Usage", "usage", 100, 1, "%"),
    ("PSA", "psa", 1, 1, ""),
    ("AST%", "ast_perc", 100, 1, "%"),
    ("AST:Usg", "ast_usage_ratio", 1, 2, ""),
    ("TOV%", "tov_perc", 100, 1, "%"),
)


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


def _dispatch(client: CookieClient, spec: EndpointSpec, *, season: int, **_ignored) -> str:
    """GET one CTG stats page. Week / mode / use_cache are unused here.

    Raises:
        CollectorAuthError: The page lacks the data call, so CTG gated it or changed its
            layout and the framework's auth-refresh path takes over.
    """
    page = client.get(
        spec.url,
        params=spec.render_params(season=season),
        headers=spec.extra_headers,
        accept="text",
    )
    if _PAGE_CALL not in page:
        raise CollectorAuthError(
            f"{spec.url} came back without the player data: CTG gated the page or changed "
            "its layout. If gated, refresh the session cookie with `fetch ctg refresh-auth`."
        )
    return page


def _parse_page(html: str, *, spec: EndpointSpec, **_context) -> pd.DataFrame:
    """Return the table the page's data call carries for this tool, rows exactly as served.

    Raises:
        ValueError: No call argument is a table with the tool's key column, so CTG changed
            its page.
    """
    key_col = _TABLE_KEY_COL[spec.name]
    decoder = json.JSONDecoder()
    pos = html.index(_PAGE_CALL) + len(_PAGE_CALL)
    while arg_start := _NEXT_ARG.match(html, pos):
        arg, pos = decoder.raw_decode(html, arg_start.end())
        if isinstance(arg, list) and arg and key_col in arg[0]:
            return pd.DataFrame.from_records(arg)
    raise ValueError(
        f"{spec.name}: no table with column {key_col!r} in the CTG page; its layout changed."
    )


def _fixed(value: float, decimals: int) -> str:
    """Round half up, as CTG's export does (26.25 reads 26.3; Python's format() gives 26.2)."""
    return str(Decimal(value).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def _roster_seed(stats: pd.DataFrame) -> pd.DataFrame:
    """The player table in the format of CTG's own CSV export, which StatsNBA reads as its seed.

    The export floors minutes and rounds every other figure half up, ranks included.
    """
    seed = pd.DataFrame(
        {
            "Player": stats["name"],
            "Age": [_fixed(age, 1) for age in stats["age"]],
            "Team": stats["team_name"],
            "Pos": stats["pos_category"].str.capitalize(),
            "MIN": (stats["seconds_played"] // 60).astype(int),
            "MPG": [_fixed(mpg, 1) for mpg in stats["mpg"]],
        }
    )
    for label, column, scale, decimals, suffix in _SEED_STATS:
        seed[f"{label} Rank"] = [floor(100 * ptile + 0.5) for ptile in stats[f"ptile_{column}"]]
        seed[label] = [_fixed(value * scale, decimals) + suffix for value in stats[column]]
    return seed


def _write_roster_seed(spec: EndpointSpec, table: pd.DataFrame, season: int) -> None:
    """Rewrite the season's NBA roster seed from the player table, where StatsNBA reads it."""
    if spec.name != "player_stats":
        return
    season_label = f"{season}-{(season + 1) % 100:02d}"
    path = tabular.PLAYER_DATA_BASE / "NBA" / f"nba_players_{season_label}.csv"
    seed = _roster_seed(table)
    _atomic_write_csv(seed, path)
    click.echo(f"  roster seed: wrote {len(seed)} rows -> {path}", err=True)


CTG_SOURCE = Source(
    name="ctg-fetch",
    help="Snapshot Cleaning the Glass NBA advanced tables to disk.",
    catalog_path=CATALOG_PATH,
    make_client=_make_client,
    default_context=_default_context,
    path_for=partial(tabular.dated_path_for, league="NBA", source_dir="cleaningtheglass"),
    dispatch=_dispatch,
    transform=_parse_page,
    after_write=_write_roster_seed,
    report_prefix="ctg_fetch",
    auth_fields=AuthFields(
        cookie="cleaningtheglass_cookie",
        user_agent="cleaningtheglass_user_agent",
    ),
    env_prefix="CLEANINGTHEGLASS",
    period_kind="date",
)
