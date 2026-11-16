"""NBA availability features: who sits out each game, weighted by what those players do.

Training reads the official per-game lists (:mod:`sportstradamus.collectors.game_absences`),
keeping only the absences a decision before tip could know; serving reads the newest ESPN injury
capture (:mod:`sportstradamus.collectors.injury_capture`). An absence keeps its source's team,
mapped to the gamelog's codes, never the player's last gamelog team: a traded player's is stale.

An absent player weighs in at his means per game over his latest games strictly before the game
date, a game an official list names him in counting 0: two-way and end-of-bench players, inactive
most regular-season nights and on no playoff list, weigh near 0 in either phase, as does a player
out for weeks. :func:`availability_features` sums those per team game and counts the row player's
own absences over his team's previous games.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from sportstradamus.collectors.tabular import PLAYER_DATA_BASE
from sportstradamus.helpers import remove_accents
from sportstradamus.helpers.io import INJURY_CAPTURE_DIR

AVAILABILITY_COLUMNS: tuple[str, ...] = (
    "avail team out min",
    "avail team out pts",
    "avail team out ast",
    "avail team out reb",
    "avail team out fga",
    "avail team out top min",
    "avail opp out min",
    "avail opp out pts",
    "avail opp out top min",
    "avail own missed10",
    "avail own back",
)
# A player's weights are his means over this many of his latest games, about a month of them.
_WEIGHT_WINDOW = 15
# Fewer games than this say too little about a player's role: he weighs 0.
_MIN_WEIGHT_GAMES = 3
# How many of the team's previous games "avail own missed10" looks back over.
_OWN_WINDOW = 10
_WEIGHT_STATS = ("MIN", "PTS", "AST", "REB", "FGA")
# Decided during the game, so no decision before tip could know them. Every other status
# ("Inactive", "DND - Injury/Illness", "NWT - ...") was settled before tip.
_GAME_TIME_STATUSES = (
    "DNP - Coach's Decision",
    "DNP - Rest",
    "DNP - Injury/Illness",
    "DND - Coach's Decision",
)
# ESPN's statuses for a player who sits; a "Day-To-Day" player may still play.
_ESPN_OUT_STATUSES = ("Out", "Suspension")
# Typed alike on every path, so an empty frame concatenates and joins like a full one. An ESPN
# list has no date: the caller gives it its upcoming game's.
_ABSENCE_DTYPES = {"team": "object", "date": "datetime64[ns]", "player_id": "int64"}
_ESPN_DTYPES = {"team": "object", "player_id": "int64"}
_LISTED_DTYPES = {"player_id": "int64", "date": "datetime64[ns]"}


def official_absences(
    league: str, team_codes: Mapping[str, str], base: Path = PLAYER_DATA_BASE
) -> pd.DataFrame:
    """Every pregame-known absence in ``league``'s stored official lists, all seasons.

    Returns:
        ``team`` (through ``team_codes``), ``date`` (datetime64[ns], midnight) and ``player_id``
        (int64); no rows when no list is stored.
    """
    lists = _stored_lists(league, base, ["game_date", "team", "player_id", "status"])
    known = lists.loc[~lists["status"].isin(_GAME_TIME_STATUSES)]
    absences = pd.DataFrame(
        {
            "team": known["team"].replace(team_codes),
            "date": known["game_date"],
            "player_id": known["player_id"],
        }
    )
    return absences.astype(_ABSENCE_DTYPES).reset_index(drop=True)


def listed_games(league: str, base: Path = PLAYER_DATA_BASE) -> pd.DataFrame:
    """Every game a stored official list of ``league`` names a player in, whatever his status.

    Returns:
        ``player_id`` (int64) and ``date`` (datetime64[ns], midnight); no rows when no list is
        stored.
    """
    lists = _stored_lists(league, base, ["game_date", "player_id"])
    return pd.DataFrame({"player_id": lists["player_id"], "date": lists["game_date"]}).astype(
        _LISTED_DTYPES
    )


def _stored_lists(league: str, base: Path, columns: list[str]) -> pd.DataFrame:
    paths = sorted((base / league / "injuries").glob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=columns)
    return pd.concat((pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True)


def espn_absences(
    league: str,
    team_codes: Mapping[str, str],
    player_ids: Mapping[str, int],
    capture_dir: Path = INJURY_CAPTURE_DIR,
) -> pd.DataFrame:
    """The players the newest ESPN injury capture lists Out or Suspension.

    A player resolves to his gamelog id by ``player_ids[remove_accents(name)]``. One missing
    from ``player_ids`` is left out: with no games he has no weight.

    Returns:
        ``team`` (through ``team_codes``) and ``player_id`` (int64); no rows when no capture is
        stored.
    """
    captures = sorted((capture_dir / league).glob("date=*/*.parquet"))
    if not captures:
        return pd.DataFrame(columns=list(_ESPN_DTYPES)).astype(_ESPN_DTYPES)
    report = pd.read_parquet(captures[-1], columns=["team", "name", "status"])
    out = report.loc[report["status"].isin(_ESPN_OUT_STATUSES)]
    listed = pd.DataFrame(
        {
            "team": out["team"].replace(team_codes),
            "player_id": out["name"].map(remove_accents).map(player_ids),
        }
    )
    return listed.dropna(subset=["player_id"]).astype(_ESPN_DTYPES).reset_index(drop=True)


def player_weights(gamelog: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    """Each player's means per game over his last games up to and including each game.

    His games are those he played and those ``listed`` names him in, which count 0 for every
    stat. A mean covers his latest ``_WEIGHT_WINDOW`` games; a game with fewer than
    ``_MIN_WEIGHT_GAMES`` games behind it, itself included, has no row.

    Args:
        gamelog: The NBA gamelog, rows in any order; ``GAME_DATE`` may be text.
        listed: :func:`listed_games`; a game the gamelog also holds counts as played.

    Returns:
        ``PLAYER_ID``, ``GAME_DATE`` (datetime64) and the means of ``MIN``, ``PTS``, ``AST``,
        ``REB`` and ``FGA``, sorted by ``GAME_DATE`` for an as-of lookup.
    """
    played = gamelog[["PLAYER_ID", "GAME_DATE", *_WEIGHT_STATS]].assign(
        GAME_DATE=lambda log: pd.to_datetime(log["GAME_DATE"])
    )
    sat = listed.rename(columns={"player_id": "PLAYER_ID", "date": "GAME_DATE"}).assign(
        **dict.fromkeys(_WEIGHT_STATS, 0.0)
    )
    games = (
        pd.concat([played, sat])
        .drop_duplicates(["PLAYER_ID", "GAME_DATE"])
        .sort_values(["PLAYER_ID", "GAME_DATE"], ignore_index=True)
    )
    means = (
        games.groupby("PLAYER_ID")[list(_WEIGHT_STATS)]
        .rolling(_WEIGHT_WINDOW, min_periods=_MIN_WEIGHT_GAMES)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return (
        games[["PLAYER_ID", "GAME_DATE"]]
        .join(means)
        .dropna()
        .sort_values("GAME_DATE", kind="stable", ignore_index=True)
    )


def availability_features(
    rows: pd.DataFrame,
    absences: pd.DataFrame,
    weights: pd.DataFrame,
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """The :data:`AVAILABILITY_COLUMNS` of each row; 0.0 where nobody is listed.

    An absent player's weights are his :func:`player_weights` row with the latest
    ``GAME_DATE`` strictly before the game date; with none he weighs 0.

    Args:
        rows: One row per player: ``team``, ``opponent``, ``date`` (datetime64, midnight) and
            ``player_id`` (NaN for a player the gamelog does not know). Any index.
        absences: ``team``, ``date`` (datetime64[ns]) and ``player_id`` of every absence known
            for the rows' dates and earlier games.
        weights: :func:`player_weights` of the gamelog.
        team_games: ``team`` and ``date`` of the gamelog's team games. The rows' own games
            (team and opponent) join them, so an upcoming game comes last in its team's order.

    Returns:
        Floats indexed like ``rows``, with exactly :data:`AVAILABILITY_COLUMNS`.
    """
    games = rows[["team", "opponent", "date", "player_id"]].reset_index(drop=True)
    listed = absences.loc[absences["date"].isin(games["date"].unique())]
    weighted = pd.merge_asof(
        listed.sort_values("date"),
        weights,
        left_on="date",
        right_on="GAME_DATE",
        left_by="player_id",
        right_by="PLAYER_ID",
        allow_exact_matches=False,
    ).fillna(dict.fromkeys(_WEIGHT_STATS, 0.0))
    team_out = weighted.groupby(["team", "date"]).agg(
        **{
            "avail team out min": ("MIN", "sum"),
            "avail team out pts": ("PTS", "sum"),
            "avail team out ast": ("AST", "sum"),
            "avail team out reb": ("REB", "sum"),
            "avail team out fga": ("FGA", "sum"),
            "avail team out top min": ("MIN", "max"),
        }
    )
    team = team_out.reindex(pd.MultiIndex.from_frame(games[["team", "date"]]))
    opp = team_out.reindex(pd.MultiIndex.from_frame(games[["opponent", "date"]])).rename(
        columns=lambda column: column.replace("avail team", "avail opp")
    )
    features = pd.concat(
        [
            team.reset_index(drop=True),
            opp.reset_index(drop=True),
            _own_absences(games, absences, team_games),
        ],
        axis=1,
    )
    return features[list(AVAILABILITY_COLUMNS)].fillna(0.0).set_axis(rows.index)


def _own_absences(
    games: pd.DataFrame, absences: pd.DataFrame, team_games: pd.DataFrame
) -> pd.DataFrame:
    """The row players' absences over their team's previous games, indexed by row position.

    A row whose player has no such absence, or no id, is left out.
    """
    schedule = (
        pd.concat(
            [
                team_games[["team", "date"]],
                games[["team", "date"]],
                games[["opponent", "date"]].rename(columns={"opponent": "team"}),
            ]
        )
        .drop_duplicates()
        .sort_values("date")
    )
    schedule["seq"] = schedule.groupby("team").cumcount()
    players = (
        games.merge(schedule, on=["team", "date"], how="left")
        .dropna(subset=["player_id"])
        .astype({"player_id": "int64"})
        .reset_index(names="row")
    )
    missed = (
        absences.loc[absences["player_id"].isin(players["player_id"])]
        .merge(schedule, on=["team", "date"])
        .merge(
            players[["row", "team", "player_id", "seq"]],
            on=["team", "player_id"],
            suffixes=("_absent", ""),
        )
    )
    games_back = missed["seq"] - missed["seq_absent"]
    return pd.DataFrame(
        {
            "avail own missed10": games_back.between(1, _OWN_WINDOW).groupby(missed["row"]).sum(),
            "avail own back": games_back.eq(1).groupby(missed["row"]).any(),
        }
    ).astype(float)
