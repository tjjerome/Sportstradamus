"""Hash-stable pins for ``prediction.stories`` — the P2 prophecy prose generator.

These freeze the EXACT ``Thesis`` and ``Why`` strings for fixed fixtures so the
md5 seeding, star-carousel ranking, game-shape classification, phrase-bank
selection, and the slate-uniqueness pass are all proven behavior-preserving.
The property tests are the dashboard phrase-bank's old guarantees (a clear star
headlines, a lone benchwarmer never does, deterministic, empty-safe) reframed
for the new per-family theses. ``Date`` is fixed in every fixture because it
seeds the variant pick.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.prediction.stories import (
    STORIES_VERSION,
    attach_offer_why,
    attach_parlay_theses,
    parse_leg,
)

_DATE = "2026-06-13"

# Two games' worth of offer context the theses read: a high-total lopsided
# matchup (BOS favored, total 238 → blowout) and a tight mid-total one
# (DEN/MIA, ML ~ coin flip). ``Moneyline`` is the implied win probability.
_OFFERS = pd.DataFrame(
    [
        {"League": "NBA", "Game": "BOS/PHI", "Player": "Jayson Tatum", "Market": "PTS",
         "Bet": "Over", "Line": 28.5, "O/U": 238.0, "Moneyline": 0.74,
         "Avg 5": 3.5, "Avg H2H": 2.0, "DVPOA": 0.10, "Model P": 0.59, "Books P": 0.52,
         "Model": 1.18, "Books": 1.00},
        {"League": "NBA", "Game": "BOS/PHI", "Player": "Jayson Tatum", "Market": "REB",
         "Bet": "Over", "Line": 8.5, "O/U": 238.0, "Moneyline": 0.74,
         "Avg 5": 1.0, "Avg H2H": 0.0, "DVPOA": 0.02, "Model P": 0.56, "Books P": 0.50,
         "Model": 1.10, "Books": 1.00},
        {"League": "NBA", "Game": "BOS/PHI", "Player": "Joel Embiid", "Market": "PTS",
         "Bet": "Under", "Line": 30.5, "O/U": 238.0, "Moneyline": 0.26,
         "Avg 5": -2.0, "Avg H2H": -1.5, "DVPOA": -0.08, "Model P": 0.57, "Books P": 0.51,
         "Model": 1.12, "Books": 1.00},
        {"League": "NBA", "Game": "DEN/MIA", "Player": "Nikola Jokic", "Market": "AST",
         "Bet": "Over", "Line": 9.5, "O/U": 224.0, "Moneyline": 0.55,
         "Avg 5": 1.5, "Avg H2H": 0.0, "DVPOA": 0.06, "Model P": 0.60, "Books P": 0.53,
         "Model": 1.20, "Books": 1.00},
        {"League": "NBA", "Game": "DEN/MIA", "Player": "Nikola Jokic", "Market": "REB",
         "Bet": "Over", "Line": 12.5, "O/U": 224.0, "Moneyline": 0.55,
         "Avg 5": 2.0, "Avg H2H": 1.0, "DVPOA": 0.03, "Model P": 0.58, "Books P": 0.52,
         "Model": 1.15, "Books": 1.00},
    ]
)

# Three families on one (NBA, _DATE) slate. Tatum drives BOS family 1 (two
# markets), Embiid the Under-only BOS family 2; Jokic drives DEN/MIA.
_PARLAYS = pd.DataFrame(
    [
        {"League": "NBA", "Game": "BOS/PHI", "Date": _DATE, "Family": 1.0,
         "Leg 1": "Jayson Tatum Over 28.5 PTS - 59.0%, 1.0x",
         "Leg 2": "Jayson Tatum Over 8.5 REB - 56.0%, 1.0x",
         "Leg 3": "Joel Embiid Under 30.5 PTS - 57.0%, 1.0x"},
        {"League": "NBA", "Game": "BOS/PHI", "Date": _DATE, "Family": 2.0,
         "Leg 1": "Joel Embiid Under 30.5 PTS - 57.0%, 1.0x",
         "Leg 2": "Jayson Tatum Over 28.5 PTS - 59.0%, 1.0x",
         "Leg 3": None},
        {"League": "NBA", "Game": "DEN/MIA", "Date": _DATE, "Family": 1.0,
         "Leg 1": "Nikola Jokic Over 9.5 AST - 60.0%, 1.0x",
         "Leg 2": "Nikola Jokic Over 12.5 REB - 58.0%, 1.0x",
         "Leg 3": None},
    ]
)


def _theses_by_family(parlays: pd.DataFrame, offers: pd.DataFrame) -> dict[tuple, str]:
    out = attach_parlay_theses(parlays.copy(), offers.copy())
    heads = out.drop_duplicates(["Game", "Family"])
    return {(r["Game"], r["Family"]): r["Thesis"] for _, r in heads.iterrows()}


def test_version_present():
    assert STORIES_VERSION == "p2-1"


def test_thesis_exact_strings():
    assert _theses_by_family(_PARLAYS, _OFFERS) == {
        ("BOS/PHI", 1.0): "The BOS/PHI blowout lets Jayson Tatum hunt points unopposed",
        ("BOS/PHI", 2.0): "Up big early, Joel Embiid sits as BOS/PHI gets called off the gas",
        ("DEN/MIA", 1.0): (
            "Nikola Jokic pulls the strings down the stretch of a razor-thin DEN/MIA"
        ),
    }


def test_thesis_written_on_every_family_row():
    out = attach_parlay_theses(_PARLAYS.copy(), _OFFERS.copy())
    bos1 = out[(out["Game"] == "BOS/PHI") & (out["Family"] == 1.0)]
    assert bos1["Thesis"].nunique() == 1
    assert (out["Thesis"] != "").all()


def test_why_exact_strings():
    out = attach_offer_why(_OFFERS.copy())
    whys = {(r["Player"], r["Market"], r["Bet"]): r["Why"] for _, r in out.iterrows()}
    assert whys[("Jayson Tatum", "PTS", "Over")] == (
        "3.5 above a 28.5 line over his last 5, and 2 above it head-to-head, "
        "a favorable matchup on paper, model 59% vs book 52% on the over, a 7-pt edge."
    )
    assert whys[("Joel Embiid", "PTS", "Under")] == (
        "2 below a 30.5 line over his last 5, and 1.5 below it head-to-head, "
        "a favorable matchup on paper, model 57% vs book 51% on the under, a 6-pt edge."
    )
    assert whys[("Jayson Tatum", "REB", "Over")] == (
        "1 above a 8.5 line over his last 5, model 56% vs book 50% on the over, a 6-pt edge."
    )


def test_why_ev_fallback_when_no_book_prob():
    """No ``Books P`` column ⇒ edge clause comes from the EV multiples."""
    row = pd.DataFrame(
        [{"Player": "Solo Star", "Market": "REB", "Bet": "Under", "Line": 6.5,
          "Avg 5": None, "DVPOA": None, "Model P": 0.72, "Model": 1.41, "Books": 1.41}]
    )
    assert attach_offer_why(row)["Why"].iloc[0] == "The model prices it at 1.41x."


def test_slate_uniqueness_guaranteed():
    """No two families in the same (League, Date) share a thesis."""
    date = _DATE
    offers = pd.DataFrame(
        [
            {"League": "NBA", "Game": "AAA/BBB", "Player": "Star One", "Market": "PTS",
             "Bet": "Over", "Line": 25.5, "O/U": 225.0, "Moneyline": 0.5},
            {"League": "NBA", "Game": "CCC/DDD", "Player": "Star Two", "Market": "PTS",
             "Bet": "Over", "Line": 25.5, "O/U": 225.0, "Moneyline": 0.5},
            {"League": "NBA", "Game": "EEE/FFF", "Player": "Star Three", "Market": "PTS",
             "Bet": "Over", "Line": 25.5, "O/U": 225.0, "Moneyline": 0.5},
        ]
    )
    legs = lambda game, p: {  # noqa: E731
        "Leg 1": f"{p} Over 25.5 PTS - 60.0%, 1.0x",
        "Leg 2": f"{p} Over 5.5 REB - 55.0%, 1.0x",
    }
    parlays = pd.DataFrame(
        [
            {"League": "NBA", "Game": "AAA/BBB", "Date": date, "Family": 1.0, **legs("AAA/BBB", "Star One")},
            {"League": "NBA", "Game": "CCC/DDD", "Date": date, "Family": 1.0, **legs("CCC/DDD", "Star Two")},
            {"League": "NBA", "Game": "EEE/FFF", "Date": date, "Family": 1.0, **legs("EEE/FFF", "Star Three")},
        ]
    )
    heads = list(_theses_by_family(parlays, offers).values())
    assert len(set(heads)) == len(heads)
    assert all(heads)


def test_clear_star_headlines_not_the_benchwarmer():
    """The multi-market star drives the family; a lone tiny prop never headlines."""
    offers = pd.DataFrame(
        [
            {"League": "NBA", "Game": "X/Y", "Player": "Star Wing", "Market": "PTS",
             "Bet": "Over", "Line": 28.5, "O/U": 224.0, "Moneyline": 0.5},
            {"League": "NBA", "Game": "X/Y", "Player": "Bench Guy", "Market": "REB",
             "Bet": "Under", "Line": 3.5, "O/U": 224.0, "Moneyline": 0.5},
        ]
    )
    parlays = pd.DataFrame(
        [
            {"League": "NBA", "Game": "X/Y", "Date": _DATE, "Family": 1.0,
             "Leg 1": "Star Wing Over 28.5 PTS - 60.0%, 1.0x",
             "Leg 2": "Star Wing Over 9.5 REB - 58.0%, 1.0x",
             "Leg 3": "Star Wing Over 6.5 AST - 57.0%, 1.0x",
             "Leg 4": "Bench Guy Under 3.5 REB - 70.0%, 1.0x"},
        ]
    )
    thesis = _theses_by_family(parlays, offers)[("X/Y", 1.0)]
    assert "Star Wing" in thesis
    assert "Bench Guy" not in thesis


def test_deterministic_across_calls():
    a = attach_parlay_theses(_PARLAYS.copy(), _OFFERS.copy())["Thesis"].tolist()
    b = attach_parlay_theses(_PARLAYS.copy(), _OFFERS.copy())["Thesis"].tolist()
    assert a == b


def test_empty_inputs_get_columns():
    empty_parlays = pd.DataFrame(columns=["League", "Game", "Family", "Date", "Leg 1"])
    empty_offers = pd.DataFrame(columns=["League", "Game", "Player", "Market", "Bet", "Line"])
    assert "Thesis" in attach_parlay_theses(empty_parlays, empty_offers).columns
    assert "Why" in attach_offer_why(empty_offers).columns


def test_no_parseable_legs_yields_blank_thesis():
    parlays = pd.DataFrame(
        [{"League": "NBA", "Game": "X/Y", "Date": _DATE, "Family": 1.0, "Leg 1": None}]
    )
    out = attach_parlay_theses(parlays, pd.DataFrame())
    assert out["Thesis"].iloc[0] == ""


def test_thesis_without_game_context_falls_back_to_even_shape():
    """Offers lacking the ``Game`` key (early-P2 snapshot) ⇒ league-agnostic shape."""
    offers_no_game = pd.DataFrame(
        [{"League": "NBA", "Player": "Lone Star", "Market": "PTS", "Bet": "Over", "Line": 25.5}]
    )
    parlays = pd.DataFrame(
        [{"League": "NBA", "Game": "L/M", "Date": _DATE, "Family": 1.0,
          "Leg 1": "Lone Star Over 25.5 PTS - 60.0%, 1.0x"}]
    )
    thesis = _theses_by_family(parlays, offers_no_game)[("L/M", 1.0)]
    assert "Lone Star" in thesis
    assert thesis  # an "even"-shape scoring-over headline still renders


def test_parse_leg_round_trip():
    assert parse_leg("Jayson Tatum Over 28.5 PTS - 59.0%, 1.03x") == {
        "Player": "Jayson Tatum", "Bet": "Over", "Line": 28.5, "Market": "PTS",
    }
    assert parse_leg("Pts + Rebs Guy Under 17.5 Pts + Rebs + Asts - 58.9%, 1.17x") == {
        "Player": "Pts + Rebs Guy", "Bet": "Under", "Line": 17.5, "Market": "Pts + Rebs + Asts",
    }
    assert parse_leg("") is None
    assert parse_leg("no bet token here") is None
