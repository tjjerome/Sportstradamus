"""MLB batting order cannot reach the feature frame ``Stats.get_stats`` returns.

``_join_defense_and_parks`` overwrites ``Player depth`` with ``Player position``
for MLB, and ``get_stats`` calls it *after* ``_join_profiles`` has joined the
profile in -- so whatever ``playerProfile["depth"]`` holds during a gameday build
is discarded before the frame is returned. That is why the MLB model pickles carry
no ``Player depth`` input, and why both live readers of the column
(``StatsMLB._project_plate_appearances``,
``prediction.correlation._resolve_player_positions``) re-resolve the slot through
``get_depth`` immediately before reading it.

The pin drives the real ``StatsMLB.get_stats`` offline on one real gameday, once
with the posted ``Batting Order`` in ``upcoming_games`` and once with it emptied.
A frozen today forces ``_game_context`` onto its upcoming branch -- the only branch
that consults the posted lineup at all -- and the stubbed archive keeps the run off
DuckDB. The lineup is then the single input that differs between the two arms, so
identical frames are proof that it is not a feature.

The MLB gamelog bundle is gitignored, so this skips in CI and runs on any box that
has pulled league data.
"""

from __future__ import annotations

import datetime as dt
import importlib.resources as pkg_resources

import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.stats import base
from sportstradamus.stats.mlb import StatsMLB

MARKET = "total bases"

# Distinct from any real gamelog "totals" / "moneyline" so the frame itself shows
# which _game_context branch produced it.
_STUB_TOTAL = 7.25
_STUB_MONEYLINE = -137.0

pytestmark = pytest.mark.skipif(
    not (pkg_resources.files(data) / "leagues" / "mlb" / "gamelog.parquet").is_file(),
    reason="MLB gamelog bundle is gitignored; absent in CI",
)


class _StubArchive:
    """The two archive reads ``_game_context``'s upcoming branch makes, off DuckDB."""

    def get_moneyline(self, league, game_date, team, **_):
        return _STUB_MONEYLINE

    def get_total(self, league, game_date, team, **_):
        return _STUB_TOTAL


def _fullest_logged_game(gamelog: pd.DataFrame) -> pd.DataFrame:
    """Rows of the most completely logged game on the last gameday in ``gamelog``."""
    gamedays = pd.to_datetime(gamelog["gameDate"]).dt.date
    day = gamelog[gamedays == gamedays.max()]
    return day[day["gameId"] == day.groupby("gameId").size().idxmax()]


@pytest.fixture
def upcoming_mlb_game(monkeypatch) -> tuple[StatsMLB, dt.date, list[dict]]:
    """A loaded ``StatsMLB`` posed as if one real game were tonight's slate.

    Rebuilds ``upcoming_games`` and the offers from that game's own gamelog rows,
    seeds ``playerProfile["depth"]`` with its real lineup slots (the ``get_depth``
    contract, injected here so the arms do not depend on that method's own date
    branch), and freezes today onto the gameday. ``profile_market`` is primed at the
    gameday so its cache guard lets both arms reuse one profile.
    """
    stats = StatsMLB(load_live_pitchers=False)
    stats.load()

    game = _fullest_logged_game(stats.gamelog)
    gameday = pd.Timestamp(game["gameDate"].iloc[0]).date()

    offers, slots = [], {}
    for team, rows in game.groupby("team"):
        lineup = rows[rows["battingOrder"] > 0].sort_values("battingOrder")
        opponent = rows["opponent"].iloc[0]
        stats.upcoming_games[team] = {
            "Opponent": opponent,
            "Home": bool(rows["home"].iloc[0]),
            "Opponent Pitcher": rows["opponent pitcher"].iloc[0],
            "Batting Order": lineup["playerName"].tolist(),
        }
        offers.extend(
            {"Player": p, "Team": team, "Opponent": opponent, "Date": gameday.isoformat()}
            for p in lineup["playerName"]
        )
        slots.update(zip(lineup["playerName"], lineup["battingOrder"], strict=True))

    class _FrozenToday(dt.datetime):
        @classmethod
        def today(cls):
            return dt.datetime(gameday.year, gameday.month, gameday.day)

    monkeypatch.setattr(base, "archive", _StubArchive())
    monkeypatch.setattr(base, "datetime", _FrozenToday)

    stats.profile_market(MARKET, gameday)
    stats.playerProfile["depth"] = pd.Series(slots, dtype=float)
    return stats, gameday, offers


def test_posted_lineup_does_not_change_the_returned_features(upcoming_mlb_game):
    stats, gameday, offers = upcoming_mlb_game
    seeded_depth = stats.playerProfile["depth"].copy()

    posted = stats.get_stats(MARKET, offers, gameday)
    assert not posted.empty, "the upcoming arm produced no rows"
    assert posted["Total"].eq(_STUB_TOTAL).all(), "_game_context took the historical branch"

    lineup_slots = seeded_depth.reindex(posted.index)
    assert lineup_slots.nunique() > 1, "the seeded lineup carries no slot variation to lose"
    assert posted["Player depth"].tolist() == posted["Player position"].tolist(), (
        "_join_defense_and_parks no longer overwrites MLB Player depth"
    )
    assert posted["Player depth"].tolist() != lineup_slots.tolist(), (
        "lineup slots survived into the returned frame"
    )

    # The emptied lineup is the only input that differs between the arms, so restore
    # the seeded depth: a gameday build is free to mutate the profile it read.
    for lineup in stats.upcoming_games.values():
        lineup["Batting Order"] = []
    stats.playerProfile["depth"] = seeded_depth
    blank = stats.get_stats(MARKET, offers, gameday)

    pd.testing.assert_frame_equal(posted, blank)
