"""Golden pins for the canonical structured-leg schema."""

import json

import pandas as pd

from sportstradamus.leg_schema import LEG_FIELDS, build_leg, is_priced, leg_field, leg_label

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
    assert leg_label(leg) == "Luka Doncic Over 26.5 Points"


def test_legs_round_trip_parquet(tmp_path):
    legs = [build_leg(OFFER_ROW)]
    df = pd.DataFrame({"legs": [legs]})
    p = tmp_path / "t.parquet"
    df.to_parquet(p)
    back = pd.read_parquet(p)["legs"].iloc[0]
    assert set(back[0].keys()) == set(LEG_FIELDS)
    assert back[0]["line"] == 26.5


def test_build_leg_defaults_nan_and_missing_numeric():
    leg = build_leg({**OFFER_ROW, "Boost": float("nan")})
    assert leg["boost"] == 1.0

    row = {k: v for k, v in OFFER_ROW.items() if k not in ("Stat", "Kelly")}
    leg = build_leg(row)
    assert leg["stat"] == "PTS"
    assert leg["kelly"] == 0.0


def test_build_leg_carries_the_fit_and_the_pick_flag():
    leg = build_leg({**OFFER_ROW, "Fit Prob": 0.52, "EV": 0.93, "Pick": True})
    assert (leg["fit_prob"], leg["ev"], leg["pick"]) == (0.52, 0.93, True)


def test_a_leg_the_fit_did_not_price_reads_none_and_is_no_pick():
    # One side posted: the stamp leaves NaN. A snapshot older than the stamp has no column.
    one_sided = {**OFFER_ROW, "Fit Prob": float("nan"), "EV": float("nan"), "Pick": False}
    for row in (one_sided, OFFER_ROW):
        leg = build_leg(row)
        assert leg["fit_prob"] is None and leg["ev"] is None and leg["pick"] is False


def test_a_leg_built_from_a_frame_row_is_json():
    # The dashboard's components and the ledger serialise legs, and numpy's bool is not JSON.
    board = pd.DataFrame([{**OFFER_ROW, "Fit Prob": 0.52, "EV": 0.93, "Pick": True}])
    leg = json.loads(json.dumps(build_leg(board.iloc[0])))
    assert (leg["fit_prob"], leg["ev"], leg["pick"]) == (0.52, 0.93, True)


def test_leg_field_reads_canonical_lowercase_leg():
    leg = build_leg(OFFER_ROW)
    assert leg_field(leg, "player") == "Luka Doncic"
    assert leg_field(leg, "team") == "LAL"
    assert leg_field(leg, "line") == 26.5


def test_leg_field_reads_raw_offer_row():
    assert leg_field(OFFER_ROW, "player") == "Luka Doncic"
    assert leg_field(OFFER_ROW, "market") == "PTS"
    assert leg_field(OFFER_ROW, "bet") == "Over"
    assert leg_field(OFFER_ROW, "line") == 26.5
    assert leg_field({"League": "NBA"}, "league") == "NBA"  # the constellation's display name


def test_leg_field_defaults_when_missing():
    assert leg_field({"Player": "Luka Doncic"}, "team") is None
    assert leg_field({"Player": "Luka Doncic"}, "team", "N/A") == "N/A"


def test_leg_field_reads_the_fit_off_a_raw_offer_row():
    row = {**OFFER_ROW, "Fit Prob": 0.52, "EV": 0.95, "Pick": True}
    assert leg_field(row, "fit_prob") == 0.52
    assert leg_field(row, "ev") == 0.95
    assert leg_field(row, "platform") == "Underdog"


def test_a_leg_is_priced_when_the_fit_gave_it_an_ev():
    # Neither the model's Kelly nor the board's pick flag decides: a leg under break-even
    # that is no slate pick is still priced, and a one-sided rung is not.
    priced = {**OFFER_ROW, "Kelly": 0.0, "Fit Prob": 0.52, "EV": 0.93, "Pick": False}
    one_sided = {**OFFER_ROW, "Kelly": 0.12, "Fit Prob": float("nan"), "EV": float("nan")}
    for leg in (priced, build_leg(priced)):
        assert is_priced(leg) is True
    for leg in (one_sided, build_leg(one_sided)):
        assert is_priced(leg) is False


def test_a_leg_stored_before_the_fit_is_not_priced():
    stored = {k: v for k, v in build_leg(OFFER_ROW).items() if k not in ("fit_prob", "ev", "pick")}
    assert is_priced(stored) is False
    assert is_priced({**stored, "ev": None}) is False
