"""Per-poll capture of what Underdog's feed, lobbies and pills show, beside the parse.

``get_ud`` fetches the feed and one lobby per modeled league on every ``prophecize`` run and keeps
only the active, priced lines. The information-audit reads need what it drops: a line's ``status``
(suspended lines are a news signal), the appearance's ``lineup_status_id`` (Underdog's own Q / D / O
flag), each option's status and multiplier, and the lobby's per-match ``entry_count``. Once the
board is built, ``get_ud`` also reads each PickemStat pill the feed points at for the per-line
``entry_count`` and the ``lineup_status_id`` decode. One parquet per poll and table under
``UNDERDOG_CAPTURE_DIR/{lines,matches,entry_counts}/date=YYYY-MM-DD/HHMMSS.parquet``, named by the
run's ``Scored At`` (UTC); ``over_under_id`` survives line moves, so two polls diff on it.
``books.underdog`` imports this module, so the payload indexer ``by_id`` lives here and the
requests stay there.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sportstradamus.helpers.io import _atomic_write_json, write_capture_parquet
from sportstradamus.spiderLogger import logger

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
ENTRY_DTYPES = {
    "polled_at": "datetime64[ns]",
    "sport_id": "string",
    "pickem_stat_id": "string",
    "over_under_line_id": "string",
    "appearance_id": "string",
    "entry_count": "string",
    "header": "string",
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


def poll_rows(payload: dict, source: str, polled_at: pd.Timestamp) -> pd.DataFrame:
    """One row per over/under line in a feed or lobby payload, suspended and live lines included."""
    players = by_id(payload.get("players", []))
    appearances = by_id(payload.get("appearances", []))
    games = {**by_id(payload.get("games", [])), **by_id(payload.get("solo_games", []))}
    lines = payload.get("over_under_lines", [])
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
                "polled_at": polled_at,
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


def match_rows(lobbies: dict[str, dict], polled_at: pd.Timestamp) -> pd.DataFrame:
    """The lobbies' per-match entry counts (display strings such as ``"75.2K"``)."""
    rows = [
        {
            "polled_at": polled_at,
            "source": f"lobby:{sport}",
            "match_id": group["id"],
            "entry_count": group.get("entry_count"),
        }
        for sport, payload in lobbies.items()
        for group in payload.get("match_groups", [])
    ]
    return pd.DataFrame(rows, columns=list(MATCH_DTYPES)).astype(MATCH_DTYPES)


def entry_rows(pills: dict[tuple[str, str], dict], polled_at: pd.Timestamp) -> pd.DataFrame:
    """The per-line entry counts of each ``(sport, pickem_stat_id)`` pill's ``lines[]``."""
    rows = [
        {
            "polled_at": polled_at,
            "sport_id": sport,
            "pickem_stat_id": stat_id,
            "over_under_line_id": line.get("over_under_line_id"),
            "appearance_id": line.get("appearance_id"),
            "entry_count": line.get("entry_count"),
            "header": line.get("header"),
        }
        for (sport, stat_id), payload in pills.items()
        for line in payload.get("lines", [])
    ]
    return pd.DataFrame(rows, columns=list(ENTRY_DTYPES)).astype(ENTRY_DTYPES)


def write_poll_capture(
    feed: dict, lobbies: dict[str, dict], capture_dir: Path, polled_at: pd.Timestamp
) -> None:
    """Write this poll's line rows and match rows, one parquet each, under ``capture_dir``."""
    lines = [poll_rows(feed, "feed", polled_at)]
    lines += [poll_rows(payload, f"lobby:{sport}", polled_at) for sport, payload in lobbies.items()]
    write_capture_parquet(pd.concat(lines, ignore_index=True), capture_dir / "lines", polled_at)
    write_capture_parquet(match_rows(lobbies, polled_at), capture_dir / "matches", polled_at)


def write_entry_capture(
    pills: dict[tuple[str, str], dict], statuses: dict, capture_dir: Path, polled_at: pd.Timestamp
) -> None:
    """Write the pills' entry counts and, when it answered, the ``lineup_status_id`` decode.

    A pill that answered no lines costs only its own rows; the warning names it. An empty decode
    leaves the last one on disk.
    """
    empty = [
        f"{sport}/{stat_id}"
        for (sport, stat_id), payload in pills.items()
        if not payload.get("lines")
    ]
    if empty:
        logger.warning(f"{len(empty)} Underdog pills answered empty: {empty}")
    write_capture_parquet(entry_rows(pills, polled_at), capture_dir / "entry_counts", polled_at)
    if statuses:
        _atomic_write_json(statuses, capture_dir / "lineup_statuses.json")
