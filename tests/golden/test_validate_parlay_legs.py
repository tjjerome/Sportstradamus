"""Pins for ``stories.legs.validate_parlay_legs`` — same-game parlay validity rules.

A valid same-game entry has distinct players and covers both teams. The rule lives
in the prediction layer so both sides share one copy: the constellation builder
gates lock-in on it, and the story-menu preset generator enumerates only valid
subsets (so a seeded preset is valid by construction).
"""

from __future__ import annotations

from sportstradamus.prediction.stories.legs import validate_parlay_legs


def _leg(player: str, team: str | None) -> dict:
    return {"Player": player, "Team": team}


def test_both_teams_distinct_players_is_valid():
    ok, reason = validate_parlay_legs([_leg("A", "NYK"), _leg("B", "SAS")])
    assert ok
    assert reason == ""


def test_duplicate_player_is_invalid():
    ok, reason = validate_parlay_legs([_leg("A", "NYK"), _leg("A", "SAS")])
    assert not ok
    assert "player" in reason.lower()


def test_single_team_is_invalid():
    ok, reason = validate_parlay_legs([_leg("A", "NYK"), _leg("B", "NYK")])
    assert not ok
    assert "team" in reason.lower()


def test_legs_without_team_dont_count_toward_both_sides():
    # A combo leg carrying no Team can't satisfy the both-teams rule on its own.
    ok, _ = validate_parlay_legs([_leg("A", "NYK"), _leg("B", None)])
    assert not ok


def test_single_team_valid_when_both_teams_not_required():
    # The story menu relaxes the both-teams rule for a game whose model edge is
    # one-sided — a single-team preset is then a legitimate starting point.
    ok, reason = validate_parlay_legs(
        [_leg("A", "NYK"), _leg("B", "NYK")], require_both_teams=False
    )
    assert ok
    assert reason == ""


def test_duplicate_player_invalid_even_without_both_teams_rule():
    # Relaxing the both-teams rule never relaxes the distinct-player rule.
    ok, reason = validate_parlay_legs(
        [_leg("A", "NYK"), _leg("A", "NYK")], require_both_teams=False
    )
    assert not ok
    assert "player" in reason.lower()
