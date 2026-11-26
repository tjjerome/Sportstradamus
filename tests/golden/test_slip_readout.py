"""Pins for how an entry's numbers read (``dashboard/components/slip_readout.py``).

``ev_text`` is the one EV-beside-Model-EV format the rail, the shelf and the story
preloader print; ``rail_lines`` is what both slip rails print under a priced, an unpriced
and a refused slip. The scores are built by hand so each number is exact.
"""

from __future__ import annotations

from decimal import Decimal

from sportstradamus.dashboard.components.slip_readout import ev_text, rail_lines
from sportstradamus.dashboard.slip_engine import SlipScore
from sportstradamus.leg_schema import leg_label


def _leg(player: str, fit_prob: float | None) -> dict:
    return {
        "player": player,
        "market": "PTS",
        "bet": "Over",
        "line": 25.5,
        "league": "NBA",
        "game": "NYK/BOS",
        "fit_prob": fit_prob,
    }


def _score(ev: float | None, joint_p: float | None, *, pair_mods: tuple = ()) -> SlipScore:
    return SlipScore(
        ev=ev,
        indep_p=joint_p,
        joint_p=joint_p,
        payout=3.5,
        model_ev=1.204,
        bet_size=2,
        play_type="Power",
        stake=Decimal("0"),
        payout_approximate=False,
        pair_mods=pair_mods,
    )


def test_ev_text_leads_with_ev_and_shows_model_ev_alone_when_there_is_none():
    assert ev_text(0.951, 1.124) == "EV 0.95 · Model EV 1.12"
    assert ev_text(None, 1.124) == "Model EV 1.12"
    assert ev_text(float("nan"), 1.124) == "Model EV 1.12"


def test_a_priced_slip_leads_with_ev_then_checks_the_payout():
    legs = [_leg("Jalen Brunson", 0.52), _leg("Jayson Tatum", 0.5)]
    assert rail_lines(_score(0.946, 0.27), legs) == [
        "EV 0.95 · Model EV 1.20 · Kelly stake $0 · joint 27%",
        "Worth a bet if the app pays at least 3.70x (we assume 3.50x).",
        "Legs here share a game, so this value is unproven until you check the app's payout "
        "against that number.",
    ]


def test_an_unpriced_slip_names_the_leg_and_shows_only_its_model_ev():
    legs = [_leg("Jalen Brunson", 0.52), _leg("Jayson Tatum", None)]
    assert rail_lines(_score(None, None), legs) == [
        "Model EV 1.20 · Kelly stake $0",
        f"Not priced: only one side is posted for {leg_label(legs[1])}.",
    ]


def test_a_refused_slip_leaves_the_words_to_the_pair_note():
    legs = [_leg("Jalen Brunson", 0.52), _leg("Jayson Tatum", 0.5)]
    assert rail_lines(_score(0.0, 0.26, pair_mods=((0, 1, 0.0),)), legs) == []
