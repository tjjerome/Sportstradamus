"""Per-poll capture of what Underdog's feed and lobbies already show, beside the parse.

``get_ud`` fetches the feed and one lobby per modeled league on every ``prophecize`` run and keeps
only the active, priced lines. The information-audit reads need what it drops: a line's ``status``
(suspended lines are a news signal), the appearance's ``lineup_status_id`` (Underdog's own Q / D / O
flag), each option's status and multiplier, and the lobby's per-match ``entry_count``. One parquet
per poll under ``UNDERDOG_CAPTURE_DIR/{lines,matches}/date=YYYY-MM-DD/HHMMSS.parquet`` (UTC);
``over_under_id`` survives line moves, so two polls diff on it. ``books.underdog`` imports this
module, so the payload indexer ``by_id`` lives here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from sportstradamus.helpers.io import _atomic_write_parquet

# Column order and dtype of each capture table, pinned so every poll's parquet carries one
# schema: an all-null column left to pyarrow is typed ``null`` and a date's files stop reading
# together.
CAPTURE_DTYPES = {
    "polled_at": "datetime64[ns]",
    "source": "string",
    "line_id": "string",
    "over_under_id": "string",
    "status": "string",
    "live_event": "bool",
    "updated_at": "string",
    "expires_at": "string",
    "provider_id": "string",
    "stat_value": "float64",
    "stat": "string",
    "display_stat": "string",
    "pickem_stat_id": "string",
    "category": "string",
    "has_alternates": "boolean",
    "appearance_id": "string",
    "appearance_type": "string",
    "lineup_status_id": "string",
    "player_id": "string",
    "player": "string",
    "team_id": "string",
    "match_id": "Int64",
    "sport_id": "string",
    "scheduled_at": "string",
    "over_multiplier": "string",
    "over_status": "string",
    "under_multiplier": "string",
    "under_status": "string",
}
MATCH_DTYPES = {
    "polled_at": "datetime64[ns]",
    "source": "string",
    "match_id": "Int64",
    "entry_count": "string",
}
# Option choices on the over slot and the under slot (the parser's _CHOICE_SLOT, by name).
_OVER_CHOICES = ("higher", "yes", "home")
_UNDER_CHOICES = ("lower", "no", "away")


def by_id(container: list[dict] | dict[str, dict]) -> dict[str, dict]:
    """Index a feed list or a lobby dict by ``str(id)``: JSON keys are strings, match ids ints."""
    items = container.values() if isinstance(container, dict) else container
    return {str(item["id"]): item for item in items}


def _option(options: list[dict], choices: tuple[str, ...]) -> dict:
    return next((option for option in options if option["choice"] in choices), {})


def poll_rows(payload: dict, source: str, polled_at: datetime) -> pd.DataFrame:
    """One row per over/under line in a feed or lobby payload, suspended and live lines included."""
    players = by_id(payload.get("players", []))
    appearances = by_id(payload.get("appearances", []))
    games = {**by_id(payload.get("games", [])), **by_id(payload.get("solo_games", []))}
    lines = payload.get("over_under_lines", [])
    stamp = pd.Timestamp(polled_at).tz_convert("UTC").tz_localize(None)
    rows = []
    for line in lines.values() if isinstance(lines, dict) else lines:
        stat = line["over_under"]["appearance_stat"]
        appearance = appearances.get(stat["appearance_id"], {})
        player = players.get(str(appearance.get("player_id")), {})
        game = games.get(str(appearance.get("match_id")), {})
        over = _option(line["options"], _OVER_CHOICES)
        under = _option(line["options"], _UNDER_CHOICES)
        name = f"{player.get('first_name') or ''} {player.get('last_name') or ''}".strip()
        rows.append(
            {
                "polled_at": stamp,
                "source": source,
                "line_id": line["id"],
                "over_under_id": line["over_under_id"],
                "status": line["status"],
                "live_event": bool(line["live_event"]),
                "updated_at": line.get("updated_at"),
                "expires_at": line.get("expires_at"),
                "provider_id": line.get("provider_id"),
                "stat_value": line["stat_value"],
                "stat": stat["stat"],
                "display_stat": stat["display_stat"],
                "pickem_stat_id": stat.get("pickem_stat_id"),
                "category": line["over_under"].get("category"),
                "has_alternates": line["over_under"].get("has_alternates"),
                "appearance_id": stat["appearance_id"],
                "appearance_type": appearance.get("type"),
                "lineup_status_id": appearance.get("lineup_status_id"),
                "player_id": appearance.get("player_id"),
                "player": name or None,
                "team_id": appearance.get("team_id"),
                "match_id": appearance.get("match_id"),
                "sport_id": player.get("sport_id") or game.get("sport_id"),
                "scheduled_at": game.get("scheduled_at"),
                "over_multiplier": over.get("payout_multiplier"),
                "over_status": over.get("status"),
                "under_multiplier": under.get("payout_multiplier"),
                "under_status": under.get("status"),
            }
        )
    return pd.DataFrame(rows, columns=list(CAPTURE_DTYPES)).astype(CAPTURE_DTYPES)


def match_rows(lobbies: dict[str, dict], polled_at: datetime) -> pd.DataFrame:
    """The lobbies' per-match entry counts (display strings such as ``"75.2K"``)."""
    stamp = pd.Timestamp(polled_at).tz_convert("UTC").tz_localize(None)
    rows = [
        {
            "polled_at": stamp,
            "source": f"lobby:{sport}",
            "match_id": group["id"],
            "entry_count": group.get("entry_count"),
        }
        for sport, payload in lobbies.items()
        for group in payload.get("match_groups", [])
    ]
    return pd.DataFrame(rows, columns=list(MATCH_DTYPES)).astype(MATCH_DTYPES)


def write_poll_parquet(frame: pd.DataFrame, table_dir: Path, polled_at: datetime) -> None:
    """Write ``frame`` to ``table_dir/date=YYYY-MM-DD/HHMMSS.parquet``.

    The date and clock are ``polled_at``'s in UTC.
    """
    stamp = pd.Timestamp(polled_at).tz_convert("UTC")
    day, clock = stamp.strftime("%Y-%m-%d"), stamp.strftime("%H%M%S")
    _atomic_write_parquet(frame, table_dir / f"date={day}" / f"{clock}.parquet")


def write_poll_capture(
    feed: dict, lobbies: dict[str, dict], capture_dir: Path, polled_at: datetime
) -> None:
    """Write this poll's line rows and match rows, one parquet each, under ``capture_dir``."""
    lines = [poll_rows(feed, "feed", polled_at)]
    lines += [poll_rows(payload, f"lobby:{sport}", polled_at) for sport, payload in lobbies.items()]
    write_poll_parquet(pd.concat(lines, ignore_index=True), capture_dir / "lines", polled_at)
    write_poll_parquet(match_rows(lobbies, polled_at), capture_dir / "matches", polled_at)
