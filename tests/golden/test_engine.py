"""Behavior pins for the v2 thesis engine: leg enrichment + archetype routing.

These are deterministic, bank-free assertions — they prove the engine selects
the right archetype, subject, direction, and category from a leg-set + game
context, independent of the phrase bank's exact prose (pinned separately once
the bank lands). The driving requirement: a slip with no standout player must
route to ``game-script`` (the game itself is the story), never an arbitrary
alphabetical "star".
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from sportstradamus.prediction.stories import engine
from sportstradamus.prediction.stories.context import GameCtx, Leg
from sportstradamus.prediction.stories.engine import (
    _city_from_name,
    _direction,
    _home_city,
    _localize_g,
    _stack_focus,
    _subject_legs,
    route,
    thesis_variants,
)
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
    parsed = [{"player": "Jokic", "bet": "Over", "line": 9.5, "market": "Assists"}]
    legs = enrich_legs(parsed, offers)
    assert len(legs) == 1
    leg = legs[0]
    assert leg.market == "AST"  # canonical code from the joined offer, not "Assists"
    assert leg.game == "DEN/MIA" and leg.team == "DEN" and leg.position == "C1"
    assert leg.category == "playmaking"


def test_enrich_legs_unmatched_falls_back_to_parsed():
    legs = enrich_legs(
        [{"player": "Ghost", "bet": "Under", "line": 3.5, "market": "Points"}],
        pd.DataFrame(columns=["Player", "Bet", "Line"]),
    )
    leg = legs[0]
    assert leg.market == "Points" and leg.game is None and leg.team is None
    assert leg.category == "scoring"


def _legs(*specs: tuple) -> list[Leg]:
    out = []
    for spec in specs:
        p, b, ln, m, g, t, pos, c, *extra = spec
        out.append(
            Leg(
                player=p,
                bet=b,
                line=ln,
                market=m,
                game=g,
                team=t,
                position=pos,
                category=c,
                negative=extra[0] if extra else False,
                win_prob=extra[1] if len(extra) > 1 else None,
            )
        )
    return out


def _stack_ctxs(legs: list[Leg], rho: float = 0.3) -> dict[str, GameCtx]:
    ids = [f"{leg.player}|{leg.market}|{leg.bet}" for leg in legs]
    pairs = {frozenset({a, b}): rho for a, b in combinations(ids, 2)}
    game = legs[0].game
    return {game: GameCtx(league="NBA", game=game, shape="even", rho=pairs)}


def test_city_from_name_strips_one_or_two_word_nicknames():
    assert _city_from_name("Chicago Bulls") == "Chicago"
    assert _city_from_name("Oklahoma City Thunder") == "Oklahoma City"
    assert _city_from_name("Golden State Warriors") == "Golden State"
    assert _city_from_name("LA Clippers") == "LA"
    # Two-word nicknames strip both words — not "Portland Trail" / "Boston Red".
    assert _city_from_name("Portland Trail Blazers") == "Portland"
    assert _city_from_name("Boston Red Sox") == "Boston"
    assert _city_from_name("Toronto Maple Leafs") == "Toronto"


def test_localize_g_reads_prepositions_and_place_nouns_as_location():
    # A locative preposition before {g}, or a venue/crowd noun after it, means the home city.
    assert _localize_g("the points pile up in {g}") == "the points pile up in {home}"
    assert _localize_g("defense travels to {g}") == "defense travels to {home}"
    assert _localize_g("the {g} faithful roar") == "the {home} faithful roar"
    # Anything else stays the matchup.
    assert _localize_g("the {g} shootout lifts lines") == "the {g} shootout lifts lines"
    assert _localize_g("{g} goes the distance") == "{g} goes the distance"


def test_localize_g_resolves_each_occurrence_independently():
    out = _localize_g("{g} tips off in {g}").format(g="CHI/SEA", home="Chicago")
    assert out == "CHI/SEA tips off in Chicago"


def test_home_city_falls_back_to_matchup_without_home_team():
    assert _home_city(GameCtx(league="NBA", game="CHI/SEA"), "CHI/SEA") == "CHI/SEA"
    assert _home_city(None, "CHI/SEA") == "CHI/SEA"


def test_home_city_resolves_home_abbrev_to_city_via_team_assets():
    ctx = GameCtx(league="NBA", game="CHI/SEA", home_team="CHI")
    assert _home_city(ctx, "CHI/SEA") == "Chicago"


def test_thesis_variants_leaves_no_unsubstituted_token_with_home_city():
    legs = _legs(
        ("Tatum", "Over", 28.5, "PTS", "BOS/PHI", "BOS", "F1", "scoring"),
        ("Tatum", "Over", 8.5, "REB", "BOS/PHI", "BOS", "F1", "boards"),
        ("Embiid", "Under", 30.5, "PTS", "BOS/PHI", "PHI", "C1", "scoring"),
    )
    ctx = GameCtx(league="NBA", game="BOS/PHI", shape="blowout", home_team="BOS")
    rendered, idx, _ = thesis_variants(legs, {"BOS/PHI": ctx})
    assert rendered and 0 <= idx < len(rendered)
    # Every slot resolved: the localize + format pass leaves no raw {g}/{home}/{p} braces.
    assert all("{" not in variant for variant in rendered)


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


def test_enrich_legs_carries_valence_and_win_prob():
    offers = pd.DataFrame(
        [
            {
                "Player": "Guarded Guy",
                "Bet": "Under",
                "Line": 2.5,
                "Market": "TOV",
                "Game": "X/Y",
                "Team": "X",
                "Position": "G1",
                "Win Prob": 0.62,
            }
        ]
    )
    leg = enrich_legs(
        [{"player": "Guarded Guy", "bet": "Under", "line": 2.5, "market": "Turnovers"}], offers
    )[0]
    assert leg.negative is True
    assert leg.win_prob == 0.62
    assert leg.category == "mistakes"
    ghost = enrich_legs(
        [{"player": "Ghost", "bet": "Over", "line": 20.5, "market": "PTS"}], offers
    )[0]
    assert ghost.negative is False and ghost.win_prob is None


def test_player_direction_is_valence_unanimous():
    """Overs plus a TOV Under all thrive together — Over, not Mixed."""
    legs = _legs(
        ("Star", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring"),
        ("Star", "Over", 6.5, "AST", "X/Y", "X", "G1", "playmaking"),
        ("Star", "Under", 2.5, "TOV", "X/Y", "X", "G1", "mistakes", True),
    )
    archetype, subject = route(legs, {"X/Y": GameCtx(league="NBA", game="X/Y")})
    assert archetype == "player" and subject["p"] == "Star"
    assert _direction(legs) == "Over"


def test_player_direction_mixed_when_own_legs_split():
    legs = _legs(
        ("Star", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring"),
        ("Star", "Over", 2.5, "TOV", "X/Y", "X", "G1", "mistakes", True),
    )
    archetype, _subject = route(legs, {"X/Y": GameCtx(league="NBA", game="X/Y")})
    assert archetype == "player"
    assert _direction(legs) == "Mixed"


def test_route_stack_contrast_over_names_the_lone_thriver():
    """One player leans Over against an Under field — that player fronts the prose."""
    legs = _legs(
        ("Star", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring", False, 0.55),
        ("Field A", "Under", 8.5, "REB", "X/Y", "Y", "F1", "boards", False, 0.70),
        ("Field B", "Under", 4.5, "AST", "X/Y", "Y", "G2", "playmaking", False, 0.72),
    )
    archetype, subject = route(legs, _stack_ctxs(legs))
    assert archetype == "stack"
    assert subject["dir"] == "ContrastOver"
    assert subject["p"] == "Star"


def test_stack_focus_contrast_under_inverse():
    legs = _legs(
        ("Fader", "Under", 25.5, "PTS", "X/Y", "X", "G1", "scoring", False, 0.60),
        ("Field A", "Over", 8.5, "REB", "X/Y", "Y", "F1", "boards", False, 0.70),
        ("Field B", "Over", 4.5, "AST", "X/Y", "Y", "G2", "playmaking", False, 0.72),
    )
    assert _stack_focus(legs) == ("ContrastUnder", "Fader")


def test_stack_focus_head_to_head_anchor_is_conviction_not_alphabet():
    """Both sides single-player: conviction picks the anchor even when the
    alphabet and the leg count both point at the other player (anti-Fox)."""
    legs = _legs(
        ("Aaron Early", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring", False, 0.55),
        ("Aaron Early", "Over", 8.5, "REB", "X/Y", "X", "G1", "boards", False, 0.58),
        ("Zed Late", "Under", 30.5, "PTS", "X/Y", "Y", "C1", "scoring", False, 0.80),
    )
    assert _stack_focus(legs) == ("ContrastUnder", "Zed Late")


def test_stack_focus_two_v_two_is_mixed():
    legs = _legs(
        ("A", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring", False, 0.55),
        ("B", "Over", 8.5, "REB", "X/Y", "X", "F1", "boards", False, 0.60),
        ("C", "Under", 30.5, "PTS", "X/Y", "Y", "C1", "scoring", False, 0.75),
        ("D", "Under", 6.5, "AST", "X/Y", "Y", "G1", "playmaking", False, 0.65),
    )
    direction, anchor = _stack_focus(legs)
    assert direction == "Mixed"
    assert anchor == "C"  # counts tie, so conviction picks the anchor


def test_stack_focus_straddler_is_mixed():
    """A player on both narrative sides muddies the contrast — Mixed, not Contrast."""
    legs = _legs(
        ("Star", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring", False, 0.60),
        ("Star", "Under", 8.5, "REB", "X/Y", "X", "G1", "boards", False, 0.58),
        ("Field", "Under", 30.5, "PTS", "X/Y", "Y", "C1", "scoring", False, 0.70),
    )
    direction, _anchor_name = _stack_focus(legs)
    assert direction == "Mixed"


def test_anchor_conviction_breaks_leg_count_ties():
    """Equal leg counts: the higher-conviction player anchors, not the first name."""
    legs = _legs(
        ("Aaron Early", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring", False, 0.55),
        ("Zed Late", "Over", 8.5, "REB", "X/Y", "Y", "F1", "boards", False, 0.80),
    )
    assert engine._anchor(legs) == "Zed Late"


def test_unit_subject_filters_group_and_maps_display():
    legs = _legs(
        ("A", "Over", 25.5, "PTS", "X/Y", "X", "P1", "scoring"),
        ("B", "Over", 18.5, "PTS", "X/Y", "X", "P2", "scoring"),
        ("C", "Under", 8.5, "REB", "X/Y", "Y", "C1", "boards"),
    )
    ctx = GameCtx(league="NBA", game="X/Y", pos_edges={"X": {"P": {"dvpoa": 0.12, "n": 2}}})
    archetype, subject = route(legs, {"X/Y": ctx})
    assert archetype == "unit"
    assert subject["grp"] == "point guard"  # NBA "P" mapped for display
    assert subject["dir"] == "Over"  # the group's narrative side, not the bystander's
    sub = _subject_legs(legs, "X/Y", "unit", subject)
    assert [leg.player for leg in sub] == ["A", "B"]


def test_thesis_variants_uses_stack_focus_direction(monkeypatch):
    """The routed subject's direction reaches the bank — anchor and direction can't split."""
    calls = {}

    def fake_bank_cell(voice, archetype, shape, direction, category):
        calls.update(archetype=archetype, direction=direction)
        return ["{p} against the grain in {g}"]

    monkeypatch.setattr(engine, "bank_cell", fake_bank_cell)
    legs = _legs(
        ("Star", "Over", 25.5, "PTS", "X/Y", "X", "G1", "scoring", False, 0.55),
        ("Field A", "Under", 8.5, "REB", "X/Y", "Y", "F1", "boards", False, 0.70),
        ("Field B", "Under", 4.5, "AST", "X/Y", "Y", "G2", "playmaking", False, 0.72),
    )
    variants, vi, _subject = engine.thesis_variants(legs, _stack_ctxs(legs))
    assert calls == {"archetype": "stack", "direction": "ContrastOver"}
    assert variants[vi] == "Star against the grain in X/Y"
