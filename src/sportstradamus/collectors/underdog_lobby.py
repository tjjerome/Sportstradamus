"""``fetch underdog-lobby``: per-line entry counts by PickemStat pill, and the lineup-status decode.

The feed carries no per-line popularity; the lobby's ``lines?filter_type=PickemStat`` pill does
(``lines[]{over_under_line_id, appearance_id, entry_count, header}``, docs/underdog_api.md §6.4).
One request per (sport, pickem_stat_id) the feed shows today, about a hundred a day, once a day:
the T2b read wants a count per leg, not a time series. ``/v1/lineup_statuses`` decodes the
``lineup_status_id`` the per-poll capture stores and is refreshed beside it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import click
import pandas as pd
from tqdm import tqdm

from sportstradamus.books.underdog import (
    _UD_QUERY,
    UD_LINES_URL,
    UD_LOBBY_URL,
    UD_MODELED_LEAGUES,
    _ud_league,
)
from sportstradamus.collectors.underdog_capture import by_id
from sportstradamus.helpers.io import (
    UNDERDOG_CAPTURE_DIR,
    _atomic_write_json,
    _atomic_write_parquet,
)
from sportstradamus.helpers.scraping import Scrape

UD_PILL_URL = f"{UD_LOBBY_URL}/lines?sport_id={{sport}}&filter_type=PickemStat&filter_id={{stat_id}}&limit=1000&{_UD_QUERY}"
UD_LINEUP_STATUSES_URL = "https://stats.underdogfantasy.com/v1/lineup_statuses"
ENTRY_DTYPES = {
    "polled_at": "datetime64[ns]",
    "sport_id": "string",
    "pickem_stat_id": "string",
    "over_under_line_id": "string",
    "appearance_id": "string",
    "entry_count": "string",
    "header": "string",
}


def feed_pills(feed: dict) -> set[tuple[str, str]]:
    """The ``(sport_id, pickem_stat_id)`` pairs the feed's modeled-league lines point at."""
    appearances = by_id(feed.get("appearances", []))
    games = by_id(feed.get("games", []))
    pills = set()
    for line in feed.get("over_under_lines", []):
        stat = line["over_under"]["appearance_stat"]
        game = games.get(str(appearances.get(stat["appearance_id"], {}).get("match_id")), {})
        sport = _ud_league(str(game.get("sport_id", "")))
        if sport in UD_MODELED_LEAGUES and stat.get("pickem_stat_id"):
            pills.add((sport, stat["pickem_stat_id"]))
    return pills


@click.command("underdog-lobby")
def underdog_lobby() -> None:
    """Snapshot per-line Underdog entry counts for today's modeled-league stats, once a day."""
    scraper = Scrape()
    polled_at = datetime.now(UTC)
    stamp = pd.Timestamp(polled_at).tz_localize(None)
    feed = scraper.get(UD_LINES_URL)
    rows = []
    for sport, stat_id in tqdm(sorted(feed_pills(feed)), desc="Underdog pills", unit="pill"):
        payload = scraper.get(UD_PILL_URL.format(sport=sport, stat_id=stat_id))
        rows += [
            {
                "polled_at": stamp,
                "sport_id": sport,
                "pickem_stat_id": stat_id,
                "over_under_line_id": line.get("over_under_line_id"),
                "appearance_id": line.get("appearance_id"),
                "entry_count": line.get("entry_count"),
                "header": line.get("header"),
            }
            for line in payload.get("lines", [])
        ]
    entries = pd.DataFrame(rows, columns=list(ENTRY_DTYPES)).astype(ENTRY_DTYPES)
    day, clock = polled_at.strftime("%Y-%m-%d"), polled_at.strftime("%H%M%S")
    _atomic_write_parquet(
        entries, UNDERDOG_CAPTURE_DIR / "entry_counts" / f"date={day}" / f"{clock}.parquet"
    )
    statuses = scraper.get(UD_LINEUP_STATUSES_URL)
    if statuses:
        _atomic_write_json(statuses, UNDERDOG_CAPTURE_DIR / "lineup_statuses.json")
    click.echo(
        f"underdog-lobby: {len(entries)} lines over {entries['pickem_stat_id'].nunique()} pills; lineup statuses {'refreshed' if statuses else 'unchanged'}"
    )
