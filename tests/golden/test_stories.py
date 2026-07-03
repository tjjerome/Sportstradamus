"""Hash-stable pins for ``prediction.stories`` — the v2 prophecy generator.

These freeze the exact ``Thesis`` / ``Why`` strings for fixed fixtures so the
archetype routing, game-context classification, voice-bank selection, md5
seeding, and slate-uniqueness pass are all proven behavior-preserving. The
routing property tests are the heart: a clear star headlines its family, a
no-standout leg-set routes to the *game-script* archetype (never an arbitrary
alphabetical star), and the per-offer "why" reads form + matchup + edge.
``Date`` is fixed because it seeds the variant pick; ``O/U`` is the team
half-total (the two sides sum to the game total) per the v2 context contract.
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


def _legs(*specs: tuple[str, str, float, str]) -> list[dict]:
    """A ``legs`` column value from ``(player, bet, line, market)`` tuples.

    ``enrich_legs`` only reads the four lowercase keys, so fixtures don't need
    full ``build_leg`` structs — this mirrors the old ``"{Player} {Bet} {Line}
    {Market} - ..."`` ``Leg N`` strings these tests used to encode.
    """
    return [
        {"player": p, "bet": b, "line": ln, "market": m} for p, b, ln, m in specs
    ]

# Two games. BOS/PHI: half-totals 120+118 = 238, BOS favored 0.74 → blowout.
# DEN/MIA: 112+112 = 224, ML ~ coin flip (0.55) → coinflip. ``Moneyline`` is the
# team's implied win probability; ``Position`` is the resolved depth-chart label.
_OFFERS = pd.DataFrame(
    [
        {
            "League": "NBA",
            "Game": "BOS/PHI",
            "Date": _DATE,
            "Team": "BOS",
            "Opponent": "PHI",
            "Player": "Jayson Tatum",
            "Market": "PTS",
            "Bet": "Over",
            "Line": 28.5,
            "O/U": 120.0,
            "Moneyline": 0.74,
            "DVPOA": 0.10,
            "Position": "F1",
            "Avg 5": 3.5,
            "Avg H2H": 2.0,
            "Win Prob": 0.59,
            "Market Prob": 0.52,
            "Model EV": 1.18,
            "Market EV": 1.00,
        },
        {
            "League": "NBA",
            "Game": "BOS/PHI",
            "Date": _DATE,
            "Team": "BOS",
            "Opponent": "PHI",
            "Player": "Jayson Tatum",
            "Market": "REB",
            "Bet": "Over",
            "Line": 8.5,
            "O/U": 120.0,
            "Moneyline": 0.74,
            "DVPOA": 0.02,
            "Position": "F1",
            "Avg 5": 1.0,
            "Avg H2H": 0.0,
            "Win Prob": 0.56,
            "Market Prob": 0.50,
            "Model EV": 1.10,
            "Market EV": 1.00,
        },
        {
            "League": "NBA",
            "Game": "BOS/PHI",
            "Date": _DATE,
            "Team": "PHI",
            "Opponent": "BOS",
            "Player": "Joel Embiid",
            "Market": "PTS",
            "Bet": "Under",
            "Line": 30.5,
            "O/U": 118.0,
            "Moneyline": 0.26,
            "DVPOA": -0.08,
            "Position": "C1",
            "Avg 5": -2.0,
            "Avg H2H": -1.5,
            "Win Prob": 0.57,
            "Market Prob": 0.51,
            "Model EV": 1.12,
            "Market EV": 1.00,
        },
        {
            "League": "NBA",
            "Game": "DEN/MIA",
            "Date": _DATE,
            "Team": "DEN",
            "Opponent": "MIA",
            "Player": "Nikola Jokic",
            "Market": "AST",
            "Bet": "Over",
            "Line": 9.5,
            "O/U": 112.0,
            "Moneyline": 0.55,
            "DVPOA": 0.06,
            "Position": "C1",
            "Avg 5": 1.5,
            "Avg H2H": 0.0,
            "Win Prob": 0.60,
            "Market Prob": 0.53,
            "Model EV": 1.20,
            "Market EV": 1.00,
        },
        {
            "League": "NBA",
            "Game": "DEN/MIA",
            "Date": _DATE,
            "Team": "DEN",
            "Opponent": "MIA",
            "Player": "Nikola Jokic",
            "Market": "REB",
            "Bet": "Over",
            "Line": 12.5,
            "O/U": 112.0,
            "Moneyline": 0.55,
            "DVPOA": 0.03,
            "Position": "C1",
            "Avg 5": 2.0,
            "Avg H2H": 1.0,
            "Win Prob": 0.58,
            "Market Prob": 0.52,
            "Model EV": 1.15,
            "Market EV": 1.00,
        },
    ]
)

# Three families: Tatum drives BOS family 1 (two markets, blowout); BOS family 2
# is an even Embiid-Under / Tatum-Over split with no leg-majority → game-script;
# Jokic drives DEN/MIA.
_PARLAYS = pd.DataFrame(
    [
        {
            "League": "NBA",
            "Game": "BOS/PHI",
            "Date": _DATE,
            "Family": 1.0,
            "legs": _legs(
                ("Jayson Tatum", "Over", 28.5, "PTS"),
                ("Jayson Tatum", "Over", 8.5, "REB"),
                ("Joel Embiid", "Under", 30.5, "PTS"),
            ),
        },
        {
            "League": "NBA",
            "Game": "BOS/PHI",
            "Date": _DATE,
            "Family": 2.0,
            "legs": _legs(
                ("Joel Embiid", "Under", 30.5, "PTS"),
                ("Jayson Tatum", "Over", 28.5, "PTS"),
            ),
        },
        {
            "League": "NBA",
            "Game": "DEN/MIA",
            "Date": _DATE,
            "Family": 1.0,
            "legs": _legs(
                ("Nikola Jokic", "Over", 9.5, "AST"),
                ("Nikola Jokic", "Over", 12.5, "REB"),
            ),
        },
    ]
)


def _theses_by_family(parlays: pd.DataFrame, offers: pd.DataFrame) -> dict[tuple, str]:
    out = attach_parlay_theses(parlays.copy(), offers.copy())
    heads = out.drop_duplicates(["Game", "Family"])
    return {(r["Game"], r["Family"]): r["Thesis"] for _, r in heads.iterrows()}


def test_version_present():
    assert STORIES_VERSION == "p3b"


def test_thesis_exact_strings():
    assert _theses_by_family(_PARLAYS, _OFFERS) == {
        ("BOS/PHI", 1.0): "Jayson Tatum piles on before the BOS/PHI bench empties",
        ("BOS/PHI", 2.0): "edges on both sides of the ledger in BOS/PHI",
        ("DEN/MIA", 1.0): ("The closer DEN/MIA gets, the better Nikola Jokic's passes become"),
    }


def test_no_standout_family_routes_to_game_script_not_a_star():
    """The even Embiid-Under / Tatum-Over split has no leg-majority ⇒ the headline
    is about the game, naming neither player (the v1 alphabetical-star bug)."""
    thesis = _theses_by_family(_PARLAYS, _OFFERS)[("BOS/PHI", 2.0)]
    assert "Tatum" not in thesis and "Embiid" not in thesis
    assert thesis


def test_thesis_written_on_every_row():
    out = attach_parlay_theses(_PARLAYS.copy(), _OFFERS.copy())
    assert (out["Thesis"] != "").all()
    bos1 = out[(out["Game"] == "BOS/PHI") & (out["Family"] == 1.0)]
    assert bos1["Thesis"].nunique() == 1


def test_why_exact_strings():
    out = attach_offer_why(_OFFERS.copy())
    whys = {(r["Player"], r["Market"], r["Bet"]): r["Why"] for _, r in out.iterrows()}
    assert whys[("Jayson Tatum", "PTS", "Over")] == (
        "Running 3.5 past a 28.5 line over his last 5, with 2 of daylight above it "
        "head-to-head, the matchup grades out friendly, a 7-pt gap on the over: "
        "model 59%, book 52%."
    )
    assert whys[("Joel Embiid", "PTS", "Under")] == (
        "2 below a 30.5 line over his last 5, and 1.5 short of it when these two meet, "
        "a matchup that leans the right way, model 57%, book 51% — the under carries "
        "a 6-pt edge."
    )
    assert whys[("Jayson Tatum", "REB", "Over")] == (
        "1 above a 8.5 line over his last 5, a 6-pt gap on the over: model 56%, book 50%."
    )


def test_why_ev_fallback_when_no_book_prob():
    """No ``Books P`` column ⇒ edge clause comes from the EV multiples."""
    row = pd.DataFrame(
        [
            {
                "Player": "Solo Star",
                "Market": "REB",
                "Bet": "Under",
                "Line": 6.5,
                "Avg 5": None,
                "DVPOA": None,
                "Win Prob": 0.72,
                "Model EV": 1.41,
                "Market EV": 1.41,
            }
        ]
    )
    assert attach_offer_why(row)["Why"].iloc[0] == "The model prices it at 1.41x."


def _star_offer(game: str, team: str, opp: str, player: str, pos: str) -> dict:
    return {
        "League": "NBA",
        "Game": game,
        "Date": _DATE,
        "Team": team,
        "Opponent": opp,
        "Player": player,
        "Market": "PTS",
        "Bet": "Over",
        "Line": 25.5,
        "O/U": 112.0,
        "Moneyline": 0.5,
        "DVPOA": 0.0,
        "Position": pos,
    }


def test_slate_uniqueness_guaranteed():
    """No two leg-sets in the same (League, Date) share a thesis."""
    offers = pd.DataFrame(
        [
            _star_offer("AAA/BBB", "AAA", "BBB", "Star One", "G1"),
            _star_offer("CCC/DDD", "CCC", "DDD", "Star Two", "G1"),
            _star_offer("EEE/FFF", "EEE", "FFF", "Star Three", "G1"),
        ]
    )
    def star_legs(p: str) -> list[dict]:
        return _legs((p, "Over", 25.5, "PTS"), (p, "Over", 5.5, "REB"))

    parlays = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "AAA/BBB",
                "Date": _DATE,
                "Family": 1.0,
                "legs": star_legs("Star One"),
            },
            {
                "League": "NBA",
                "Game": "CCC/DDD",
                "Date": _DATE,
                "Family": 1.0,
                "legs": star_legs("Star Two"),
            },
            {
                "League": "NBA",
                "Game": "EEE/FFF",
                "Date": _DATE,
                "Family": 1.0,
                "legs": star_legs("Star Three"),
            },
        ]
    )
    heads = list(_theses_by_family(parlays, offers).values())
    assert len(set(heads)) == len(heads)
    assert all(heads)


def test_clear_star_headlines_not_the_benchwarmer():
    """The multi-leg star drives the family; a lone tiny prop never headlines."""
    offers = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "X/Y",
                "Date": _DATE,
                "Team": "X",
                "Opponent": "Y",
                "Player": "Star Wing",
                "Market": "PTS",
                "Bet": "Over",
                "Line": 28.5,
                "O/U": 112.0,
                "Moneyline": 0.5,
                "DVPOA": 0.0,
                "Position": "F1",
            },
            {
                "League": "NBA",
                "Game": "X/Y",
                "Date": _DATE,
                "Team": "X",
                "Opponent": "Y",
                "Player": "Star Wing",
                "Market": "REB",
                "Bet": "Over",
                "Line": 9.5,
                "O/U": 112.0,
                "Moneyline": 0.5,
                "DVPOA": 0.0,
                "Position": "F1",
            },
            {
                "League": "NBA",
                "Game": "X/Y",
                "Date": _DATE,
                "Team": "X",
                "Opponent": "Y",
                "Player": "Star Wing",
                "Market": "AST",
                "Bet": "Over",
                "Line": 6.5,
                "O/U": 112.0,
                "Moneyline": 0.5,
                "DVPOA": 0.0,
                "Position": "F1",
            },
            {
                "League": "NBA",
                "Game": "X/Y",
                "Date": _DATE,
                "Team": "Y",
                "Opponent": "X",
                "Player": "Bench Guy",
                "Market": "REB",
                "Bet": "Under",
                "Line": 3.5,
                "O/U": 112.0,
                "Moneyline": 0.5,
                "DVPOA": 0.0,
                "Position": "C2",
            },
        ]
    )
    parlays = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "X/Y",
                "Date": _DATE,
                "Family": 1.0,
                "legs": _legs(
                    ("Star Wing", "Over", 28.5, "PTS"),
                    ("Star Wing", "Over", 9.5, "REB"),
                    ("Star Wing", "Over", 6.5, "AST"),
                    ("Bench Guy", "Under", 3.5, "REB"),
                ),
            },
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
    empty_parlays = pd.DataFrame(columns=["League", "Game", "Family", "Date", "legs"])
    empty_offers = pd.DataFrame(columns=["League", "Game", "Player", "Market", "Bet", "Line"])
    assert "Thesis" in attach_parlay_theses(empty_parlays, empty_offers).columns
    assert "Why" in attach_offer_why(empty_offers).columns


def test_no_parseable_legs_yields_blank_thesis():
    parlays = pd.DataFrame(
        [{"League": "NBA", "Game": "X/Y", "Date": _DATE, "Family": 1.0, "legs": None}]
    )
    out = attach_parlay_theses(parlays, pd.DataFrame())
    assert out["Thesis"].iloc[0] == ""


def test_thesis_without_game_context_degrades_gracefully():
    """Offers lacking the ``Game`` key (no context) ⇒ a non-empty fallback headline."""
    offers_no_game = pd.DataFrame(
        [{"League": "NBA", "Player": "Lone Star", "Market": "PTS", "Bet": "Over", "Line": 25.5}]
    )
    parlays = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "L/M",
                "Date": _DATE,
                "Family": 1.0,
                "legs": _legs(("Lone Star", "Over", 25.5, "PTS")),
            }
        ]
    )
    thesis = _theses_by_family(parlays, offers_no_game)[("L/M", 1.0)]
    assert thesis  # a game-script "even"-shape headline still renders, no crash


def _mini_offer(game: str, team: str, player: str, market: str, bet: str, line: float) -> dict:
    return {
        "League": "NBA",
        "Game": game,
        "Date": _DATE,
        "Team": team,
        "Player": player,
        "Market": market,
        "Bet": bet,
        "Line": line,
        "O/U": 112.0,
        "Moneyline": 0.5,
    }


def test_contrast_slip_headline_names_the_star(monkeypatch):
    """Correlated star-Over / field-Under slip: the stack anchors on the lone thriver."""
    from sportstradamus.prediction.stories import engine as engine_mod

    monkeypatch.setattr(engine_mod, "bank_cell", lambda *_k: ["{p} against the grain in {g}"])
    offers = pd.DataFrame(
        [
            _mini_offer("X/Y", "X", "Star Q", "PTS", "Over", 25.5),
            _mini_offer("X/Y", "Y", "Field A", "REB", "Under", 8.5),
            _mini_offer("X/Y", "Y", "Field B", "AST", "Under", 4.5),
        ]
    )
    corr = [
        {"Game": "X/Y", "leg_a": "Star Q|PTS|Over", "leg_b": "Field A|REB|Under", "rho": 0.3},
        {"Game": "X/Y", "leg_a": "Star Q|PTS|Over", "leg_b": "Field B|AST|Under", "rho": 0.3},
        {"Game": "X/Y", "leg_a": "Field A|REB|Under", "leg_b": "Field B|AST|Under", "rho": 0.3},
    ]
    parlays = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "X/Y",
                "Date": _DATE,
                "Family": 1.0,
                "legs": _legs(
                    ("Star Q", "Over", 25.5, "PTS"),
                    ("Field A", "Under", 8.5, "REB"),
                    ("Field B", "Under", 4.5, "AST"),
                ),
            }
        ]
    )
    out = attach_parlay_theses(parlays, offers, corr=corr)
    thesis = out["Thesis"].iloc[0]
    assert "Star Q" in thesis
    assert "Field A" not in thesis and "Field B" not in thesis


def test_muddled_mixed_slip_names_nobody(monkeypatch):
    """Uncorrelated two-a-side split with mixed directions: game-script, no name."""
    from sportstradamus.prediction.stories import engine as engine_mod

    monkeypatch.setattr(engine_mod, "bank_cell", lambda *_k: ["{g} refuses to pick a lane"])
    offers = pd.DataFrame(
        [
            _mini_offer("X/Y", "X", "Alpha", "PTS", "Over", 25.5),
            _mini_offer("X/Y", "X", "Alpha", "REB", "Under", 8.5),
            _mini_offer("X/Y", "Y", "Zeta", "PTS", "Under", 22.5),
            _mini_offer("X/Y", "Y", "Zeta", "AST", "Over", 6.5),
        ]
    )
    parlays = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "X/Y",
                "Date": _DATE,
                "Family": 1.0,
                "legs": _legs(
                    ("Alpha", "Over", 25.5, "PTS"),
                    ("Alpha", "Under", 8.5, "REB"),
                    ("Zeta", "Under", 22.5, "PTS"),
                    ("Zeta", "Over", 6.5, "AST"),
                ),
            }
        ]
    )
    thesis = attach_parlay_theses(parlays, offers)["Thesis"].iloc[0]
    assert thesis
    assert "Alpha" not in thesis and "Zeta" not in thesis


def test_wnba_why_uses_her():
    row = pd.DataFrame(
        [
            {
                "League": "WNBA",
                "Player": "A. Wilson",
                "Market": "PTS",
                "Bet": "Over",
                "Line": 22.5,
                "Avg 5": 3.0,
                "Date": _DATE,
            }
        ]
    )
    why = attach_offer_why(row)["Why"].iloc[0]
    assert "her last 5" in why


def test_why_rotation_deterministic():
    """The md5 variant rotation is a pure function of the row — reruns are identical."""
    a = attach_offer_why(_OFFERS.copy())["Why"].tolist()
    b = attach_offer_why(_OFFERS.copy())["Why"].tolist()
    assert a == b


def test_parse_leg_round_trip():
    assert parse_leg("Jayson Tatum Over 28.5 PTS - 59.0%, 1.03x") == {
        "Player": "Jayson Tatum",
        "Bet": "Over",
        "Line": 28.5,
        "Market": "PTS",
    }
    assert parse_leg("Pts + Rebs Guy Under 17.5 Pts + Rebs + Asts - 58.9%, 1.17x") == {
        "Player": "Pts + Rebs Guy",
        "Bet": "Under",
        "Line": 17.5,
        "Market": "Pts + Rebs + Asts",
    }
    assert parse_leg("") is None
    assert parse_leg("no bet token here") is None
