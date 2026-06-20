"""Behavior pins for the v2 thesis engine: leg enrichment + archetype routing.

These are deterministic, bank-free assertions — they prove the engine selects
the right archetype, subject, direction, and category from a leg-set + game
context, independent of the phrase bank's exact prose (pinned separately once
the bank lands). The driving requirement: a slip with no standout player must
route to ``game-script`` (the game itself is the story), never an arbitrary
alphabetical "star".
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.prediction.stories.context import GameCtx, Leg
from sportstradamus.prediction.stories.engine import route
from sportstradamus.prediction.stories.legs import enrich_legs


def test_enrich_legs_joins_offer_context():
    offers = pd.DataFrame(
        [
            {
                "Player": "Jokic",
                "Bet": "Over",
                "Line": 9.5,
                "Market": "AST",
                "Game": "DEN/MIA",
                "Team": "DEN",
                "Position": "C1",
            },
            {
                "Player": "Jokic",
                "Bet": "Over",
                "Line": 12.5,
                "Market": "REB",
                "Game": "DEN/MIA",
                "Team": "DEN",
                "Position": "C1",
            },
        ]
    )
    parsed = [{"Player": "Jokic", "Bet": "Over", "Line": 9.5, "Market": "Assists"}]
    legs = enrich_legs(parsed, offers)
    assert len(legs) == 1
    leg = legs[0]
    assert leg.market == "AST"  # canonical code from the joined offer, not "Assists"
    assert leg.game == "DEN/MIA" and leg.team == "DEN" and leg.position == "C1"
    assert leg.category == "playmaking"


def test_enrich_legs_unmatched_falls_back_to_parsed():
    legs = enrich_legs(
        [{"Player": "Ghost", "Bet": "Under", "Line": 3.5, "Market": "Points"}],
        pd.DataFrame(columns=["Player", "Bet", "Line"]),
    )
    leg = legs[0]
    assert leg.market == "Points" and leg.game is None and leg.team is None
    assert leg.category == "scoring"


def _legs(*specs: tuple) -> list[Leg]:
    return [
        Leg(player=p, bet=b, line=ln, market=m, game=g, team=t, position=pos, category=c)
        for p, b, ln, m, g, t, pos, c in specs
    ]


def test_route_player_archetype_on_clear_star():
    legs = _legs(
        ("Tatum", "Over", 28.5, "PTS", "BOS/PHI", "BOS", "F1", "scoring"),
        ("Tatum", "Over", 8.5, "REB", "BOS/PHI", "BOS", "F1", "boards"),
        ("Embiid", "Under", 30.5, "PTS", "BOS/PHI", "PHI", "C1", "scoring"),
    )
    ctx = GameCtx(league="NBA", game="BOS/PHI", shape="blowout")
    archetype, subject = route(legs, {"BOS/PHI": ctx})
    assert archetype == "player"
    assert subject["p"] == "Tatum"


def test_route_game_script_when_no_standout():
    """Four legs, two players two each — no unique majority ⇒ game-script, not a coin-toss star."""
    legs = _legs(
        ("Alpha", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring"),
        ("Alpha", "Over", 5.5, "AST", "X/Y", "X", "G1", "playmaking"),
        ("Zeta", "Over", 22.5, "PTS", "X/Y", "Y", "G1", "scoring"),
        ("Zeta", "Over", 6.5, "REB", "X/Y", "Y", "G1", "boards"),
    )
    ctx = GameCtx(league="NBA", game="X/Y", shape="shootout")
    archetype, subject = route(legs, {"X/Y": ctx})
    assert archetype == "game-script"
    assert "p" not in subject


def test_route_stack_on_correlated_same_game_bundle():
    legs = _legs(
        ("A", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring"),
        ("B", "Over", 8.5, "REB", "X/Y", "X", "F1", "boards"),
        ("C", "Over", 6.5, "AST", "X/Y", "X", "G2", "playmaking"),
    )
    rho = {
        frozenset({"A|PTS|Over", "B|REB|Over"}): 0.4,
        frozenset({"A|PTS|Over", "C|AST|Over"}): 0.3,
        frozenset({"B|REB|Over", "C|AST|Over"}): 0.2,
    }
    ctx = GameCtx(league="NBA", game="X/Y", shape="even", rho=rho)
    archetype, subject = route(legs, {"X/Y": ctx})
    assert archetype == "stack"
    assert subject["n"] == 3 and "p" in subject


def test_route_unit_on_shared_position_group_edge():
    legs = _legs(
        ("A", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring"),
        ("B", "Over", 18.5, "PTS", "X/Y", "X", "G2", "scoring"),
    )
    ctx = GameCtx(
        league="NBA",
        game="X/Y",
        shape="even",
        fav_team="X",
        pos_edges={"X": {"G": {"dvpoa": 0.12, "n": 2}}},
    )
    archetype, subject = route(legs, {"X/Y": ctx})
    assert archetype == "unit"
    assert subject["team"] == "X" and subject["grp"] == "G" and subject["opp"] == "Y"


def test_route_empty_legs_is_game_script():
    archetype, _subject = route([], {})
    assert archetype == "game-script"
