"""Pin that a historical gameday's depth chart is keyed on who played, not on today's roster.

``Stats.get_depth`` writes the per-(team, position) usage rank that ``_join_profiles`` gates
training rows on (``Player depth > 0``). On a past date the gamelog row proves the player took
the field, so the roster snapshot may only contribute attributes; an inner merge dropped every
player absent from the roster *on the day the matrix was built* from every historical gameday,
so an NFL cold rebuild's population depended on its run date. The upcoming-date branch keeps
roster membership as the depth chart.

Two neighbours of the same roster table are pinned alongside: the NFL loader must pass unmapped
team codes through (``Series.map`` with a dict turns them into NaN), and the NFL fantasy spec
must read the position ``get_depth`` wrote to ``playerProfile`` -- the gamelog's on a past date,
the roster's on an upcoming one -- rather than the roster alone.
"""

from datetime import date, datetime

import pandas as pd

from sportstradamus.stats import StatsNFL
from sportstradamus.stats import nfl as nfl_module
from sportstradamus.stats.nfl import NFL_FANTASY_COMPONENTS
from sportstradamus.training.component_cells import _fantasy_weights

GAMEDAY = date(2023, 10, 29)
ROSTERED = "Tony Pollard"
UNROSTERED = "Brandin Cooks"
OFFERS = [{"Player": ROSTERED}, {"Player": UNROSTERED}]


def _stats_with_one_unrostered_player() -> StatsNFL:
    stats = StatsNFL()
    stats.gamelog = pd.DataFrame(
        {
            "player display name": [ROSTERED, UNROSTERED],
            "team": ["DAL", "DAL"],
            "position": ["RB", "WR"],
            "gameday": [GAMEDAY, GAMEDAY],
        }
    )
    stats.players = pd.DataFrame(
        {"age": [27.0], "height": [72.0], "weight": [209.0], "team": ["DAL"], "position": ["RB"]},
        index=pd.Index([ROSTERED], name="name"),
    )
    stats.playerProfile = pd.DataFrame(
        {"snap pct short": [0.8, 0.6], "route participation short": [0.7, 0.9]},
        index=[ROSTERED, UNROSTERED],
    )
    stats.profile_market = lambda market, date=None: None
    return stats


def test_historical_gameday_ranks_every_player_who_played():
    stats = _stats_with_one_unrostered_player()

    stats.get_depth(OFFERS, GAMEDAY)

    profile = stats.playerProfile
    assert profile.loc[UNROSTERED, "depth"] == 1
    assert profile.loc[UNROSTERED, "position"] == stats.positions.index("WR") + 1
    assert profile.loc[UNROSTERED, "team"] == "DAL"
    assert profile.loc[ROSTERED, "depth"] == 1
    assert profile.loc[ROSTERED, "position"] == stats.positions.index("RB") + 1


def test_upcoming_gameday_still_needs_the_roster():
    stats = _stats_with_one_unrostered_player()

    stats.get_depth(OFFERS, datetime.today().date())

    assert stats.playerProfile.loc[ROSTERED, "depth"] == 1
    assert stats.playerProfile.loc[UNROSTERED, "depth"] == 0


def test_nfl_roster_loader_keeps_unmapped_team_codes(monkeypatch):
    raw = pd.DataFrame(
        {
            "name": ["Josh Allen", "Patrick Mahomes", "Tyreek Hill", "Some Kicker", "Drake Maye"],
            "team": ["BUF", "KCC", "FA", "BUF", "NEP"],
            "position": ["QB", "QB", "WR", "PK", "QB"],
            "gsis_id": ["00-0034857", "00-0033873", "00-0033040", "00-0000001", "00-0039918"],
            "age": [30.3, 31.0, 32.6, 25.0, 24.1],
            "height": [77.0, 74.0, 70.0, 72.0, 76.0],
            "weight": [237.0, 225.0, 185.0, 200.0, 225.0],
        }
    )
    monkeypatch.setattr(nfl_module.nfl, "import_ids", lambda: raw)
    stats = StatsNFL()

    stats._load_player_ids()

    assert stats.players.loc["Josh Allen", "team"] == "BUF"
    assert stats.players.loc["Patrick Mahomes", "team"] == "KC"
    assert stats.players.loc["Drake Maye", "team"] == "NE"
    assert "Tyreek Hill" not in stats.players.index
    assert "Some Kicker" not in stats.players.index
    assert stats.ids["Josh Allen"] == "00-0034857"


def test_nfl_fantasy_spec_reads_the_position_get_depth_wrote():
    stats = StatsNFL()
    stats.playerProfile = pd.DataFrame(
        {"position": [stats.positions.index("WR") + 1, 0]},
        index=["Some Receiver", "Unknown Position"],
    )

    spec = stats._fantasy_combo_spec("fantasy points underdog", "Some Receiver")

    assert tuple(sub for sub, _ in spec.marginals) == NFL_FANTASY_COMPONENTS["WR"]
    assert stats._fantasy_combo_spec("fantasy points underdog", "Unknown Position") is None
    assert stats._fantasy_combo_spec("fantasy points underdog", "Never Profiled") is None


def test_component_cells_seed_the_nfl_spec_from_the_combo_matrix_position():
    combo = pd.DataFrame({"Player": ["Some QB", "Some RB"], "Player position": [1, 3]})

    specs, labels = _fantasy_weights(
        "NFL", "fantasy points underdog", ["Some QB", "Some RB"], combo
    )

    assert tuple(sub for sub, _ in specs["Some RB"]) == NFL_FANTASY_COMPONENTS["RB"]
    assert ("rushing tds", 6.0) in specs["Some QB"]
    assert "qb_tds_via_rushing_tds" in labels
