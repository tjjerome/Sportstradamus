"""Golden pins for the canonical structured-leg schema."""

import pandas as pd

from sportstradamus.leg_schema import LEG_FIELDS, build_leg, leg_label

OFFER_ROW = {
    "Player": "Luka Doncic",
    "Team": "LAL",
    "Market": "PTS",
    "Stat": "PTS",
    "Bet": "Over",
    "Line": 26.5,
    "League": "NBA",
    "Game": "LAL/GSW",
    "Date": "2026-07-03",
    "Platform": "Underdog",
    "Win Prob": 0.623,
    "Boost": 1.5,
    "Push Prob": 0.0,
    "Kelly": 0.12,
    "Extra Col": "ignored",
}


def test_build_leg_maps_offer_row_to_schema():
    leg = build_leg(OFFER_ROW)
    assert set(leg) == set(LEG_FIELDS)
    assert leg["player"] == "Luka Doncic"
    assert leg["market"] == "PTS"
    assert leg["stat"] == "PTS"
    assert leg["bet"] == "Over"
    assert leg["line"] == 26.5
    assert leg["date"] == "2026-07-03"
    assert leg["win_prob"] == 0.623


def test_build_leg_accepts_series():
    leg = build_leg(pd.Series(OFFER_ROW))
    assert leg["game"] == "LAL/GSW"


def test_leg_label_renders_without_stored_string():
    leg = build_leg(OFFER_ROW)
    assert leg_label(leg) == "Luka Doncic Over 26.5 PTS"


def test_legs_round_trip_parquet(tmp_path):
    legs = [build_leg(OFFER_ROW)]
    df = pd.DataFrame({"legs": [legs]})
    p = tmp_path / "t.parquet"
    df.to_parquet(p)
    back = pd.read_parquet(p)["legs"].iloc[0]
    assert set(back[0].keys()) == set(LEG_FIELDS)
    assert back[0]["line"] == 26.5
