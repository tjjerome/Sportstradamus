"""Behavior guards for ``prediction/correlation.py`` — the parlay-core half-2
decomposition of :func:`find_correlation`.

Two complementary fixtures, because the function has two output halves with
different determinism:

* **offer_df correlation columns** are a pure function of the c_map + each leg's
  Model P / Boost — fully deterministic for a *fixed* c_map. The real NBA offer
  slate (``NYK/SAS``, 2026-06-03 from ``current_offers.parquet``) is the input,
  but the correlation source is a FIXED synthetic c_map (:func:`_synthetic_offer_cmap`),
  not the gitignored, periodically-regenerated corr parquets — pinning against those
  made the old ``_real_nba`` test a chronic false positive (boundary-sensitive 2-dp
  drift whenever the parquets were rebuilt). Three snapshot-dropped input columns
  (``Books P``, ``K``, ``Player position``) are reconstructed in :func:`_nba_offers`.

* **parlay_df** depends on ``beam_search_parlays`` → SciPy's multivariate-normal
  CDF, which uses a randomized (unseeded) Genz–Bretz estimator whenever the leg
  correlation submatrix is non-diagonal — so exact parlay EVs are *not*
  reproducible run to run. We therefore only assert parlay-assembly *structure*
  (non-empty, well-formed columns, plausible bet sizes) on a synthetic WNBA
  slate tuned to clear the beam gates. The parlay engine's numeric output is
  pinned separately, on identity correlation, by
  ``tests/golden/test_parlay_search.py::test_beam_search_characterization``.

The kernel unit tests below (`_leg_pair_corr_boost`, `_build_correlation_matrices`)
drive a *real* ``c_map`` built from the NBA parquet via ``_build_game_corr_map``,
so the correlation arithmetic is pinned on real values with no SciPy randomness.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from itertools import permutations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from sportstradamus import data
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE
from sportstradamus.leg_schema import LEG_FIELDS
from sportstradamus.prediction import correlation
from sportstradamus.prediction.correlation import (
    _annotate_correlation_columns,
    _build_cmarket,
    _build_correlation_matrices,
    _build_game_corr_map,
    _collect_game_corr,
    _leg_pair_corr_boost,
    _leg_shrinkage,
    _resolve_player_positions,
    find_correlation,
)
from sportstradamus.prediction.parlay import GameArrays
from tests.golden.test_underdog_pickem import stub_cell_trust


def _has_corr_parquets(league: str) -> bool:
    league_dir = pkg_resources.files(data) / "leagues" / league
    return (league_dir / "corr_same_team.parquet").is_file()


# The per-league correlation parquets are gitignored, so they are absent in CI.
# The tests that read them skip when the data is missing; the synthetic-c_map
# kernel tests below run regardless.
_needs_nba_corr = pytest.mark.skipif(
    not _has_corr_parquets("nba"),
    reason="NBA correlation parquets are gitignored; absent in CI",
)
_needs_wnba_corr = pytest.mark.skipif(
    not _has_corr_parquets("wnba"),
    reason="WNBA correlation parquets are gitignored; absent in CI",
)
_needs_mlb_corr = pytest.mark.skipif(
    not _has_corr_parquets("mlb"),
    reason="MLB correlation parquets are gitignored; absent in CI",
)


class _FakeStats:
    """Minimal stand-in for a league ``Stats`` object.

    ``find_correlation``'s non-MLB branch calls ``profile_market`` (side-effect
    only here) and reads ``playerProfile`` for the usage + tiebreaker columns
    that drive position ranking.
    """

    def __init__(self, profile: pd.DataFrame) -> None:
        self.playerProfile = profile

    def profile_market(self, _usage_col: str) -> None:
        pass


def _stats_for(players: list[str], usage_col: str, tiebreaker_col: str) -> _FakeStats:
    uniq = list(dict.fromkeys(players))
    profile = pd.DataFrame(
        {
            usage_col: np.linspace(36.0, 18.0, len(uniq)),
            tiebreaker_col: np.linspace(0.34, 0.14, len(uniq)),
        },
        index=pd.Index(uniq, name="PLAYER_NAME"),
    )
    return _FakeStats(profile)


def test_find_correlation_empty_slate_returns_column_stable_frames() -> None:
    """Every league skipped ⇒ no offers, and the old unguarded ``df["Player"]`` raised
    KeyError there — which cli.py's bare except swallowed, dropping the whole platform's
    slate. The empty frame must still carry the columns cli.py reads before ``.empty``."""
    offer_df, parlay_df = find_correlation([], {}, "Underdog")

    assert offer_df.empty
    assert parlay_df.empty
    assert "Market" in offer_df.columns
    assert "Boost" in offer_df.columns


# --- real NBA slate (offer-correlation characterization) --------------------


# Real NYK/SAS offers (2026-06-03) from data/runtime/current_offers.parquet.
# League/Date/Team/Opponent/Player/Market/Line/Boost/Bet/Model P/Model/Books are
# verbatim. Books P = clip(Model P - 0.08, .05, .95) (book-prob proxy; snapshot
# drops it), K = round(Model P, 4) (Kelly sort key; snapshot drops it),
# Player position = first-seen-player index mod 5 + 1 → positions["NBA"]
# (snapshot drops it). Processed as Underdog, which divides UNDERDOG_BOOST_BASELINE out
# of Boost: the frame carries boost ÷ 1.78 x the baseline, so the multipliers searched
# stay the real boost ÷ 1.78 the snapshot below was pinned at, whatever the baseline is.
def _nba_offers() -> list[dict]:
    raw = [
        # team, opp, player, market, line, boost, bet, model_p, model, books
        ("NYK", "SAS", "Jalen Brunson", "AST", 6.5, 1.05, "Under", 0.596523, 1.114901, 0.890181),
        ("NYK", "SAS", "Jalen Brunson", "FTM", 4.5, 0.86, "Over", 0.824506, 1.262153, 0.841373),
        ("NYK", "SAS", "Jose Alvarado", "PTS", 2.5, 1.02, "Over", 0.681707, 1.237706, 0.824925),
        ("NYK", "SAS", "Jose Alvarado", "STL", 0.5, 0.69, "Under", 0.540721, 0.664114, 0.795823),
        ("NYK", "SAS", "Mitchell Robinson", "PRA", 9.5, 1.82, "Over", 0.683758, 1.244439, 0.837713),
        ("NYK", "SAS", "Mitchell Robinson", "REB", 5.5, 2.08, "Over", 0.640494, 1.332227, 0.810926),
        ("SAS", "NYK", "De'Aaron Fox", "AST", 5.5, 1.98, "Under", 0.584958, 1.158216, 0.982567),
        ("SAS", "NYK", "De'Aaron Fox", "STL", 1.5, 2.52, "Over", 0.580061, 1.461753, 1.294296),
        ("SAS", "NYK", "Luke Kornet", "PR", 5.5, 1.73, "Over", 0.739604, 1.279515, 0.855582),
        ("SAS", "NYK", "Luke Kornet", "REB", 3.5, 2.39, "Over", 0.577842, 1.381042, 0.494779),
        ("SAS", "NYK", "Victor Wembanyama", "BLK", 3.5, 1.96, "Over", 0.702637, 1.377169, 0.735718),
        (
            "SAS",
            "NYK",
            "Victor Wembanyama",
            "BLST",
            4.5,
            1.87,
            "Under",
            0.735725,
            1.375805,
            1.194457,
        ),
    ]
    players = [r[2] for r in raw]
    pos_of = {p: (i % 5) + 1 for i, p in enumerate(dict.fromkeys(players))}
    return [
        {
            "League": "NBA",
            "Date": "2026-06-03",
            "Team": team,
            "Opponent": opp,
            "Player": player,
            "Market": market,
            "Line": line,
            "Boost": boost / 1.78 * UNDERDOG_BOOST_BASELINE,
            "Bet": bet,
            "Win Prob": mp,
            "Market Prob": round(min(max(mp - 0.08, 0.05), 0.95), 4),
            "Model EV": model,
            "Market EV": books,
            "Kelly": round(mp, 4),
            "Player position": pos_of[player],
        }
        for team, opp, player, market, line, boost, bet, mp, model, books in raw
    ]


# The 12 cMarket tokens the NBA offer slate resolves to (position prefix from the fake usage
# profile in _stats_for + the market map — a stable code path, independent of any data file). A
# FIXED synthetic correlation per token-pair makes the offer-corr annotation deterministic,
# decoupled from the gitignored corr parquets whose periodic regeneration drifted the old pin.
_OFFER_CORR_TOKENS = (
    "C1.PTS",
    "C1.STL",
    "F1.PRA",
    "F1.REB",
    "P1.AST",
    "P1.FTM",
    "_OPP_B1.PR",
    "_OPP_B1.REB",
    "_OPP_P1.BLK",
    "_OPP_P1.BLST",
    "_OPP_W1.AST",
    "_OPP_W1.STL",
)


def _synthetic_offer_cmap() -> dict[tuple[str, str], float]:
    """A fixed symmetric correlation for every ordered token-pair (spans -0.45..+0.45)."""

    def value(a: str, b: str) -> float:
        lo, hi = sorted((_OFFER_CORR_TOKENS.index(a), _OFFER_CORR_TOKENS.index(b)))
        return round(((lo * 31 + hi * 17) % 19 - 9) / 20, 4)

    return {(a, b): value(a, b) for a in _OFFER_CORR_TOKENS for b in _OFFER_CORR_TOKENS if a != b}


# Characterization snapshot of the offer_df correlation annotations under the synthetic c_map
# above. Deterministic (pure-numpy EV grid, no SciPy randomness) and stable run-to-run because
# the c_map is fixed — 10/12 legs annotate, with the Corr Same and Corr Opp columns both
# exercised. Each partner is the structured record ``_annotate_correlation_columns`` writes
# (player/market/bet/line/mult, plus a ``league`` the test adds: NBA on every leg here) —
# ``leg_schema.leg_label`` renders it to a string on demand, so the test pins the structured
# fields directly rather than a formatted label.
_EXPECTED_OFFER_CORR = [
    (
        "De'Aaron Fox",
        "AST",
        [
            {
                "player": "Victor Wembanyama",
                "market": "BLK",
                "bet": "Over",
                "line": 3.5,
                "mult": 1.11,
            },
            {"player": "Luke Kornet", "market": "REB", "bet": "Over", "line": 3.5, "mult": 1.02},
        ],
        [
            {
                "player": "Mitchell Robinson",
                "market": "PRA",
                "bet": "Over",
                "line": 9.5,
                "mult": 1.06,
            }
        ],
    ),
    (
        "De'Aaron Fox",
        "STL",
        [
            {
                "player": "Victor Wembanyama",
                "market": "BLK",
                "bet": "Over",
                "line": 3.5,
                "mult": 1.09,
            },
            {"player": "Luke Kornet", "market": "PR", "bet": "Over", "line": 5.5, "mult": 1.03},
        ],
        [
            {"player": "Jose Alvarado", "market": "PTS", "bet": "Over", "line": 2.5, "mult": 1.08},
            {
                "player": "Mitchell Robinson",
                "market": "REB",
                "bet": "Over",
                "line": 5.5,
                "mult": 1.06,
            },
        ],
    ),
    ("Jalen Brunson", "AST", [], []),
    (
        "Jalen Brunson",
        "FTM",
        [],
        [
            {
                "player": "Victor Wembanyama",
                "market": "BLST",
                "bet": "Under",
                "line": 4.5,
                "mult": 1.04,
            }
        ],
    ),
    (
        "Jose Alvarado",
        "PTS",
        [],
        [
            {
                "player": "Victor Wembanyama",
                "market": "BLST",
                "bet": "Under",
                "line": 4.5,
                "mult": 1.09,
            },
            {"player": "De'Aaron Fox", "market": "STL", "bet": "Over", "line": 1.5, "mult": 1.08},
        ],
    ),
    ("Jose Alvarado", "STL", [], []),
    (
        "Luke Kornet",
        "PR",
        [
            {
                "player": "Victor Wembanyama",
                "market": "BLK",
                "bet": "Over",
                "line": 3.5,
                "mult": 1.09,
            },
            {"player": "De'Aaron Fox", "market": "STL", "bet": "Over", "line": 1.5, "mult": 1.03},
        ],
        [
            {
                "player": "Mitchell Robinson",
                "market": "PRA",
                "bet": "Over",
                "line": 9.5,
                "mult": 1.03,
            }
        ],
    ),
    (
        "Luke Kornet",
        "REB",
        [
            {"player": "De'Aaron Fox", "market": "AST", "bet": "Under", "line": 5.5, "mult": 1.02},
            {
                "player": "Victor Wembanyama",
                "market": "BLK",
                "bet": "Over",
                "line": 3.5,
                "mult": 1.02,
            },
        ],
        [],
    ),
    (
        "Mitchell Robinson",
        "PRA",
        [],
        [
            {"player": "De'Aaron Fox", "market": "AST", "bet": "Under", "line": 5.5, "mult": 1.06},
            {
                "player": "Victor Wembanyama",
                "market": "BLST",
                "bet": "Under",
                "line": 4.5,
                "mult": 1.03,
            },
            {"player": "Luke Kornet", "market": "PR", "bet": "Over", "line": 5.5, "mult": 1.03},
        ],
    ),
    (
        "Mitchell Robinson",
        "REB",
        [],
        [{"player": "De'Aaron Fox", "market": "STL", "bet": "Over", "line": 1.5, "mult": 1.06}],
    ),
    (
        "Victor Wembanyama",
        "BLK",
        [
            {"player": "De'Aaron Fox", "market": "AST", "bet": "Under", "line": 5.5, "mult": 1.11},
            {"player": "Luke Kornet", "market": "PR", "bet": "Over", "line": 5.5, "mult": 1.09},
        ],
        [],
    ),
    (
        "Victor Wembanyama",
        "BLST",
        [{"player": "De'Aaron Fox", "market": "AST", "bet": "Under", "line": 5.5, "mult": 1.03}],
        [
            {"player": "Jose Alvarado", "market": "PTS", "bet": "Over", "line": 2.5, "mult": 1.09},
            {"player": "Jalen Brunson", "market": "FTM", "bet": "Over", "line": 4.5, "mult": 1.04},
            {
                "player": "Mitchell Robinson",
                "market": "PRA",
                "bet": "Over",
                "line": 9.5,
                "mult": 1.03,
            },
        ],
    ),
]


def _stub_corr_sources(monkeypatch, c_map: dict[tuple[str, str], float]) -> None:
    """Stand a fixed ``c_map`` in for every game's, and an empty frame for the gitignored
    corr parquets it is built from, so a slate scores the same on any box."""
    monkeypatch.setattr(correlation, "_build_game_corr_map", lambda *a, **k: c_map)
    real_read = pd.read_parquet

    def _stub_corr_read(path, *args, **kwargs):
        if str(path).endswith(("corr_same_team.parquet", "corr_opposing.parquet")):
            return pd.DataFrame({0: []}, index=pd.MultiIndex.from_arrays([[], [], []]))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _stub_corr_read)


def test_find_correlation_offer_correlations_synthetic(monkeypatch) -> None:
    """Pin the deterministic offer_df correlation annotation: the real NBA offer slate driven by
    a FIXED synthetic c_map (and a stubbed corr-parquet read), so it no longer false-positives on
    the drifting, gitignored correlation parquets and runs everywhere (no real data needed)."""
    _stub_corr_sources(monkeypatch, _synthetic_offer_cmap())

    offers = _nba_offers()
    stats = {"NBA": _stats_for([o["Player"] for o in offers], "MIN short", "USG_PCT short")}

    offer_df, _ = find_correlation(offers, stats, "Underdog", contest_variant="power")

    actual = sorted(
        (r["Player"], r["Market"], r["Corr Same"] or [], r["Corr Opp"] or [])
        for _, r in offer_df[["Player", "Market", "Corr Same", "Corr Opp"]].iterrows()
    )
    expected = sorted(
        (
            player,
            market,
            [{**partner, "league": "NBA"} for partner in same],
            [{**partner, "league": "NBA"} for partner in opp],
        )
        for player, market, same, opp in _EXPECTED_OFFER_CORR
    )
    assert actual == expected


# --- synthetic WNBA slate (parlay-assembly structure) -----------------------


# Uniform-ish high win probs + Boost == UNDERDOG_BOOST_BASELINE (post-normalize
# 1.0) clear the beam-search EV gates so the assembly path actually runs.
def _wnba_offers() -> list[dict]:
    raw = [
        ("Aja Wilson", "LVA", "NYL", "PTS", 22.5, 3, 0.82),
        ("Jackie Young", "LVA", "NYL", "PRA", 27.5, 1, 0.80),
        ("Kelsey Plum", "LVA", "NYL", "AST", 4.5, 1, 0.81),
        ("Sabrina Ionescu", "NYL", "LVA", "PTS", 19.5, 1, 0.83),
        ("Breanna Stewart", "NYL", "LVA", "REB", 8.5, 2, 0.79),
        ("Jonquel Jones", "NYL", "LVA", "PTS", 14.5, 3, 0.84),
    ]
    return [
        {
            "League": "WNBA",
            "Date": "2026-05-08",
            "Team": team,
            "Opponent": opp,
            "Player": player,
            "Market": market,
            "Line": line,
            "Boost": UNDERDOG_BOOST_BASELINE,  # post-normalization 1.0
            "Bet": "Over",
            "Win Prob": mp,
            "Market Prob": round(mp - 0.10, 4),
            "Model EV": 1.0 + (mp - 0.5),
            "Market EV": 1.0,
            "Kelly": 1.0,
            "Player position": pos,
        }
        for player, team, opp, market, line, pos, mp in raw
    ]


@_needs_wnba_corr
def test_find_correlation_builds_parlays_wnba() -> None:
    """find_correlation wires beam_search + assembles a well-formed parlay_df.

    Structural only (parlay EVs are non-deterministic on correlated legs); the
    guard is that the assembly path runs and produces sane rows. ``legs`` (the
    P8 structured-leg schema) replaces the retired ``Leg 1..6`` / ``Legs`` /
    ``Desc`` / ``Family`` columns; ``Fun`` survives as a sort key the parlay
    history stores: ``persist._PARLAY_DROP_COLS`` strips it from
    ``current_parlays.parquet`` only.
    """
    offers = _wnba_offers()
    stats = {"WNBA": _stats_for([o["Player"] for o in offers], "MIN short", "USG_PCT short")}

    _, parlay_df = find_correlation(offers, stats, "Underdog", contest_variant="power")

    assert not parlay_df.empty
    for col in ("Game", "League", "Platform", "Model EV", "Market EV", "legs", "Bet Size"):
        assert col in parlay_df.columns
    assert "Family" not in parlay_df.columns
    for col in ("Leg 1", "Legs", "Desc"):
        assert col not in parlay_df.columns
    assert set(parlay_df["Bet Size"].astype(int)).issubset({2, 3, 4, 5, 6})
    assert (parlay_df["League"] == "WNBA").all()
    assert (parlay_df["Game"] == "LVA/NYL").all()

    first_legs = parlay_df["legs"].iloc[0]
    assert isinstance(first_legs, list) and len(first_legs) >= 2
    assert all(set(leg.keys()) == set(LEG_FIELDS) for leg in first_legs)
    # PTS/PRA/AST/REB aren't in stat_map["Underdog"], so identity-fallback (stat
    # == market) is the *correct* answer for this fixture's markets — the
    # remap-actually-fires case (WNBA "Fantasy Points") is pinned directly on
    # resolve_leg_stat in test_parlay_search.py, not fixture-dependent here.
    assert all(leg["stat"] == leg["market"] for row in parlay_df["legs"] for leg in row)


# The cMarket tokens the WNBA slate resolves to: LVA is listed first, so NYL is the
# ``_OPP_`` side.
_WNBA_TOKENS = ("C1.PTS", "G1.PRA", "G2.AST", "_OPP_G1.PTS", "_OPP_F1.REB", "_OPP_C1.PTS")


def test_find_correlation_builds_one_game_frame_per_game_date(monkeypatch) -> None:
    """A team with offers on two dates plays two games. Grouped by team alone, a
    back-to-back made one frame of both nights: parlays mixing the two, one story
    context under one date, partner lists naming the other night's lines, and a line
    posted at the same number both nights kept once. The earlier game is walked first,
    because the stored correlation slice keeps the first of two rows for a game and
    leg pair."""
    _stub_corr_sources(monkeypatch, dict.fromkeys(permutations(_WNBA_TOKENS, 2), 0.2))
    first = _wnba_offers()
    later = [{**leg, "Date": "2026-05-09", "Line": leg["Line"] + 1.0} for leg in first]
    later[0]["Line"] = first[0]["Line"]
    offers = later + first
    stats = {"WNBA": _stats_for([o["Player"] for o in offers], "MIN short", "USG_PCT short")}
    story_sink: list = []

    offer_df, parlay_df = find_correlation(
        offers, stats, "Underdog", contest_variant="power", story_sink=story_sink
    )

    assert [(ctx.game, ctx.date) for ctx in story_sink] == [
        ("LVA/NYL", "2026-05-08"),
        ("LVA/NYL", "2026-05-09"),
    ]
    for ctx in story_sink:
        assert {leg["Date"] for leg in ctx.bet_df.values()} == {ctx.date}
        assert len(ctx.bet_df) == len(first)

    assert set(parlay_df["Date"]) == {"2026-05-08", "2026-05-09"}
    for legs, date in zip(parlay_df["legs"], parlay_df["Date"], strict=True):
        assert {leg["date"] for leg in legs} == {date}

    # All but one of the later night's lines are a point higher, so a partner's line
    # tells its date.
    lines_on = {
        date: set(zip(legs["Player"], legs["Line"], strict=True))
        for date, legs in offer_df.groupby("Date")
    }
    for _, leg in offer_df.iterrows():
        partners = leg["Corr Same"] + leg["Corr Opp"]
        assert partners
        assert {(p["player"], p["line"]) for p in partners} <= lines_on[leg["Date"]]


@_needs_wnba_corr
def test_find_correlation_writes_position_labels_wnba() -> None:
    """Resolved depth-chart labels survive onto the returned offers frame.

    ``_resolve_player_positions`` builds ``G1``/``F1``/... labels on a league
    slice that the function discards; v2 writes them back as a ``Position``
    column so ``build_game_context`` can aggregate positional matchup edges.
    """
    offers = _wnba_offers()
    stats = {"WNBA": _stats_for([o["Player"] for o in offers], "MIN short", "USG_PCT short")}

    offer_df, _ = find_correlation(offers, stats, "Underdog", contest_variant="power")

    assert "Position" in offer_df.columns
    labels = offer_df["Position"].astype(str).str.strip()
    assert (labels != "").all()  # no combo legs in this fixture ⇒ every leg resolves
    # WNBA depth labels are a position letter (G/F/C) plus a usage rank digit.
    assert labels.str.match(r"^[GFC]\d+$").all()


# --- MLB batting slots (correlation-matrix keys) ----------------------------


class _FakeMLBStats:
    """Stand-in for ``StatsMLB`` in the MLB branch of ``_resolve_player_positions``.

    ``get_depth`` records the arguments it was handed and writes the batting
    slots onto ``playerProfile`` the way the real method does — a dict pandas
    aligns on the profile index, so a player the profile does not carry stays
    absent from the ``depth`` column. A ``None`` slot puts a player in the
    profile with no entry in that dict, i.e. NaN depth, which is what the real
    method's ``continue`` leaves behind for a hitter it cannot place.
    """

    def __init__(self, slots: dict[str, int | None]) -> None:
        self.playerProfile = pd.DataFrame(index=pd.Index(list(slots), name="playerName"))
        self._slots = {k: v for k, v in slots.items() if v is not None}
        self.depth_calls: list[tuple[list[dict], object]] = []

    def get_depth(self, offers, date=None):
        self.depth_calls.append((offers, date))
        self.playerProfile["depth"] = self._slots


# "Player position" is 0 on every real MLB leg (playerProfile["position"] is only
# filled by the non-MLB get_depth), so the fixture pins it there: the branch has to
# resolve the batting slot from the stats object, not from the column.
def _mlb_legs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Player": player,
                "Team": team,
                "Date": "2026-09-04",
                "Market": "hits",
                "Player position": 0,
            }
            for player, team in [
                ("Kyle Tucker", "CHC"),
                ("Nico Hoerner", "CHC"),
                ("Shota Imanaga", "CHC"),
                ("Kyle Tucker + Shota Imanaga", "CHC"),
                ("Ian Happ", "CHC"),
                ("Oneil Cruz", "PIT"),
            ]
        ]
    )


def _mlb_stats() -> _FakeMLBStats:
    return _FakeMLBStats(
        {"Kyle Tucker": 3, "Nico Hoerner": 9, "Shota Imanaga": 0, "Ian Happ": None}
    )


def test_resolve_player_positions_mlb_labels_batting_slots() -> None:
    """MLB slots come from ``get_depth``, not the always-zero offer column.

    Reading the offer column labeled every MLB leg ``P``, so every batter pair
    looked up a missing correlation row and scored rho 0. Slots 1-9 become
    ``B1``..``B9``; the three ways a leg misses — a pitcher's slot 0, a NaN depth
    (Ian Happ, in the profile but unplaced), and a player the profile never
    carried at all (Oneil Cruz) — all fall through to ``P``. Combo legs keep the
    one-label-per-component list shape ``_build_cmarket`` expects.
    """
    stat_data = _mlb_stats()

    resolved = _resolve_player_positions(_mlb_legs(), "MLB", stat_data)

    assert list(resolved["Player position"]) == ["B3", "B9", "P", ["B3", "P"], "P", "P"]
    assert len(stat_data.depth_calls) == 1  # once for the league, not once per leg
    records, call_date = stat_data.depth_calls[0]
    # Dated: a hitter offered for a later game gets that game's slot, as his features do.
    assert records == [
        {"Player": player, "Team": team, "Date": "2026-09-04"}
        for player, team in [
            ("Kyle Tucker", "CHC"),
            ("Nico Hoerner", "CHC"),
            ("Shota Imanaga", "CHC"),
            ("Kyle Tucker + Shota Imanaga", "CHC"),
            ("Ian Happ", "CHC"),
            ("Oneil Cruz", "PIT"),
        ]
    ]
    assert call_date is None  # the default, today, exactly as scoring.py calls it


def test_build_cmarket_mlb_keys_by_batting_slot() -> None:
    """cMarket tokens must key the way the MLB corr parquets do: ``B{n}``/``P``."""
    resolved = _resolve_player_positions(_mlb_legs(), "MLB", _mlb_stats())

    cmarket = _build_cmarket(resolved, "MLB", "Underdog")["cMarket"]

    assert list(cmarket) == [
        ["B3.hits"],
        ["B9.hits"],
        ["P.hits"],
        ["B3.hits", "P.hits"],
        ["P.hits"],
        ["P.hits"],
    ]


# No live product posts an MLB "A vs. B" leg any more, but the shape survives the
# correlation stage where other leagues' combo legs don't, so the slate keeps one as
# a pin against it reaching persist unhandled.
def _mlb_offers() -> list[dict]:
    raw = [
        # player, team, opp, market, line
        ("Kyle Tucker", "CHC", "PIT", "hits", 0.5),
        ("Nico Hoerner", "CHC", "PIT", "total bases", 1.5),
        ("Shota Imanaga", "CHC", "PIT", "pitcher strikeouts", 5.5),
        ("Oneil Cruz", "PIT", "CHC", "hits", 0.5),
        ("Bryan Reynolds", "PIT", "CHC", "runs", 0.5),
        ("Kyle Tucker vs. Oneil Cruz", "CHC/PIT", "PIT/CHC", "hits", 0.5),
    ]
    return [
        {
            "League": "MLB",
            "Date": "2026-09-04",
            "Team": team,
            "Opponent": opp,
            "Player": player,
            "Market": market,
            "Line": line,
            "Boost": UNDERDOG_BOOST_BASELINE,  # post-normalization 1.0
            "Bet": "Over",
            "Win Prob": 0.81,
            "Market Prob": 0.71,
            "Model EV": 1.31,
            "Market EV": 1.0,
            "Kelly": 1.0,
            "Player position": 0,
        }
        for player, team, opp, market, line in raw
    ]


@_needs_mlb_corr
def test_find_correlation_mlb_position_stays_a_string_column() -> None:
    """The ``Position`` writeback must not leak a combo leg's list of labels.

    ``persist`` writes the offers frame to parquet, and pyarrow refuses a column
    mixing lists with strings, so a single "A vs. B" leg would take prophecize down.
    Combo legs get ``""`` — what ``stories/context._pos_edges`` already expects —
    while ``Player position`` keeps the list ``_build_cmarket`` splits on.
    """
    stat_data = _FakeMLBStats(
        {
            "Kyle Tucker": 3,
            "Nico Hoerner": 9,
            "Shota Imanaga": 0,
            "Oneil Cruz": 2,
            "Bryan Reynolds": 5,
        }
    )

    offer_df, _ = find_correlation(_mlb_offers(), {"MLB": stat_data}, "Underdog")

    assert all(isinstance(p, str) for p in offer_df["Position"])
    labels = dict(zip(offer_df["Player"], offer_df["Position"], strict=True))
    assert labels["Kyle Tucker"] == "B3"
    assert labels["Shota Imanaga"] == "P"
    assert labels["Kyle Tucker vs. Oneil Cruz"] == ""
    pa.Table.from_pandas(offer_df[["Position"]])  # the snapshot write, which a mix aborts


# --- cMarket names (the league's own, not the platform map's) ---------------


def _cmarket_tokens(league: str, platform: str, position: str, *labels: str) -> list[list[str]]:
    legs = pd.DataFrame({"Player position": position, "Market": list(labels)})
    return list(_build_cmarket(legs, league, platform)["cMarket"])


@pytest.mark.parametrize(
    ("league", "platform", "position", "label", "token"),
    [
        ("NHL", "Sleeper", "C1", "assists", "C1.assists"),
        ("NHL", "Underdog", "C1", "Assists", "C1.assists"),
        ("NBA", "Underdog", "P1", "Fantasy Points", "P1.fantasy points prizepicks"),
    ],
)
def test_build_cmarket_names_a_market_the_way_its_leagues_matrices_do(
    league, platform, position, label, token
) -> None:
    """A platform's map names a market the same for every league and the correlation
    matrices go by the league's own name. Sleeper's ``assists`` is ``AST`` in the map and
    ``assists`` in NHL's matrices: keyed ``C1.AST`` the leg read no correlation."""
    assert _cmarket_tokens(league, platform, position, label) == [[token]]


def test_build_cmarket_keeps_one_leagues_names_out_of_the_next() -> None:
    """A platform's leagues are built one after the other and each names a market its own
    way. One map shared by them all once carried NBA's ``fantasy points prizepicks`` into
    every league built after it."""
    assert _cmarket_tokens("NHL", "Underdog", "C1", "Points") == [["C1.points"]]
    assert _cmarket_tokens("NBA", "Underdog", "P1", "Points", "Fantasy Points") == [
        ["P1.PTS"],
        ["P1.fantasy points prizepicks"],
    ]
    assert _cmarket_tokens("NFL", "Underdog", "QB1", "Fantasy Points") == [
        ["QB1.fantasy points underdog"]
    ]


# --- corr-slice collector (dashboard rail / constellation) ------------------


def test_collect_game_corr_keys_canonical_and_symmetric() -> None:
    """Per-game corr slice: canonical leg keys, sorted pair order, rho from C."""
    game_df = pd.DataFrame(
        {
            "Player": ["A. Wilson", "S. Ionescu", "A. Wilson"],
            "Market": ["Points", "Points", "Rebounds"],
            "Bet": ["Over", "Over", "Under"],
        }
    )
    C = np.array([[1.0, 0.3, 0.5], [0.3, 1.0, -0.2], [0.5, -0.2, 1.0]])
    M = np.array([[0.0, 1.1, 0.9], [1.1, 0.0, 1.05], [0.9, 1.05, 0.0]])
    market_map = {"Points": "PTS", "Rebounds": "REB"}

    rows = _collect_game_corr(game_df, C, M, "Underdog", "WNBA", "LVA/NYL", market_map)

    assert len(rows) == 3  # 3 distinct legs → 3 unique pairs
    for r in rows:
        assert r["Platform"] == "Underdog"
        assert r["League"] == "WNBA"
        assert r["Game"] == "LVA/NYL"
        assert r["leg_a"] < r["leg_b"]  # canonical pair ordering
        for key in (r["leg_a"], r["leg_b"]):
            _player, market, bet = key.split("|")
            assert market in {"PTS", "REB"}  # canonical code, not display name
            assert bet in {"Over", "Under"}

    by_pair = {(r["leg_a"], r["leg_b"]): r["rho"] for r in rows}
    assert by_pair[tuple(sorted(["A. Wilson|PTS|Over", "S. Ionescu|PTS|Over"]))] == pytest.approx(
        0.3
    )
    assert by_pair[tuple(sorted(["A. Wilson|PTS|Over", "A. Wilson|REB|Under"]))] == pytest.approx(
        0.5
    )
    assert by_pair[tuple(sorted(["S. Ionescu|PTS|Over", "A. Wilson|REB|Under"]))] == pytest.approx(
        -0.2
    )


def test_collect_game_corr_skips_identical_leg_keys() -> None:
    """Two offers with the same Player|Market|Bet (different lines) emit no self-pair."""
    game_df = pd.DataFrame(
        {
            "Player": ["A. Wilson", "A. Wilson"],
            "Market": ["Points", "Points"],
            "Bet": ["Over", "Over"],
        }
    )
    C = np.array([[1.0, 0.9], [0.9, 1.0]])
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    rows = _collect_game_corr(game_df, C, M, "Underdog", "WNBA", "LVA/NYL", {"Points": "PTS"})
    assert rows == []


def test_collect_game_corr_carries_platform_and_modifier() -> None:
    """Every row carries the passed Platform and modifier == M[i, j], including a
    same-player pair whose M is 0.0 — the dashboard needs that row to mark a
    slip player's other markets as mutually exclusive, so it must not be dropped
    or overridden."""
    game_df = pd.DataFrame(
        {
            "Player": ["A. Wilson", "A. Wilson", "S. Ionescu"],
            "Market": ["Points", "Rebounds", "Points"],
            "Bet": ["Over", "Over", "Over"],
        }
    )
    C = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.1], [0.4, 0.1, 1.0]])
    # Same-player pair (0, 1) is uncombinable -> M is 0.0, same as
    # _leg_pair_corr_boost would compute; must survive into the row untouched.
    M = np.array([[0.0, 0.0, 1.2], [0.0, 0.0, 0.83], [1.2, 0.83, 0.0]])
    market_map = {"Points": "PTS", "Rebounds": "REB"}

    rows = _collect_game_corr(game_df, C, M, "Sleeper", "WNBA", "LVA/NYL", market_map)

    assert all(r["Platform"] == "Sleeper" for r in rows)
    by_pair = {(r["leg_a"], r["leg_b"]): r["modifier"] for r in rows}
    same_player_pair = tuple(sorted(["A. Wilson|PTS|Over", "A. Wilson|REB|Over"]))
    assert by_pair[same_player_pair] == pytest.approx(0.0)
    assert by_pair[tuple(sorted(["A. Wilson|PTS|Over", "S. Ionescu|PTS|Over"]))] == pytest.approx(
        1.2
    )
    assert by_pair[tuple(sorted(["A. Wilson|REB|Over", "S. Ionescu|PTS|Over"]))] == pytest.approx(
        0.83
    )


# --- pure-kernel unit tests -------------------------------------------------


# Two single-position legs whose cMarket tokens key into the c_map below.
def _leg(player: str, bet: str, cmarket: str) -> dict:
    return {"Player": player, "Bet": bet, "cMarket": [cmarket]}


def test_leg_pair_corr_boost_reads_cmap_value() -> None:
    """Same-bet pair on different players: rho is the c_map entry, boost 1."""
    c_map = {("G1.PTS", "G2.AST"): 0.4}
    rho, boost = _leg_pair_corr_boost(
        _leg("A", "Over", "G1.PTS"), _leg("B", "Over", "G2.AST"), c_map, {}, {}
    )
    assert rho == pytest.approx(0.4)
    assert boost == 1


def test_leg_pair_corr_boost_opposite_bet_flips_sign() -> None:
    """Over vs Under negates the correlation increment."""
    c_map = {("G1.PTS", "G2.AST"): 0.4}
    rho, _ = _leg_pair_corr_boost(
        _leg("A", "Over", "G1.PTS"), _leg("B", "Under", "G2.AST"), c_map, {}, {}
    )
    assert rho == pytest.approx(-0.4)


def test_leg_pair_corr_boost_reversed_key_lookup() -> None:
    """The (y, x) fallback lookup finds a pair stored in the other order."""
    c_map = {("G2.AST", "G1.PTS"): 0.25}
    rho, _ = _leg_pair_corr_boost(
        _leg("A", "Over", "G1.PTS"), _leg("B", "Over", "G2.AST"), c_map, {}, {}
    )
    assert rho == pytest.approx(0.25)


def test_leg_pair_corr_boost_two_players_in_one_slot_read_no_correlation() -> None:
    """Different players sharing a position key get rho 0, not the slot's own block.

    MLB's modal batting-slot fallback (no lineup posted yet) can put two hitters in
    one ``B{n}`` key; the matrix's ``B7.x ~ B7.y`` entries are within-player
    correlations and say nothing about that pair. The banned-combo modifier still
    applies — it is keyed by market, not by who bats where.
    """
    c_map = {("B7.hits", "B7.hits"): 1.0, ("B7.hits", "B7.total bases"): 0.9}
    team_mod = {frozenset(["B.hits", "B.total bases"]): [0.8, 1.1]}
    rho, boost = _leg_pair_corr_boost(
        _leg("Church", "Under", "B7.hits"), _leg("Fermin", "Under", "B7.hits"), c_map, {}, {}
    )
    assert rho == 0
    assert boost == 1
    rho, boost = _leg_pair_corr_boost(
        _leg("Church", "Under", "B7.hits"),
        _leg("Fermin", "Under", "B7.total bases"),
        c_map,
        team_mod,
        {},
    )
    assert rho == 0
    assert boost == pytest.approx(0.8)
    rho, _ = _leg_pair_corr_boost(
        _leg("Church", "Under", "B7.hits"), _leg("Church", "Under", "B7.total bases"), c_map, {}, {}
    )
    assert rho == pytest.approx(0.9)


def test_leg_pair_corr_boost_same_player_zeroes_boost() -> None:
    """A leg pair on the same player collapses the boost to 0 (uncombinable)."""
    c_map = {("G1.PTS", "G1.AST"): 0.4}
    _, boost = _leg_pair_corr_boost(
        _leg("Luka", "Over", "G1.PTS"), _leg("Luka", "Over", "G1.AST"), c_map, {}, {}
    )
    assert boost == 0


def test_leg_pair_corr_boost_applies_banned_modifier() -> None:
    """The boost modifier is keyed by the digit-stripped, _OPP_-stripped tokens.

    Same bet uses ``modifier[0]``; opposite bet uses ``modifier[1]``.
    """
    team_mod = {frozenset(["G.PTS", "G.AST"]): [2.0, 0.5]}
    _, boost_same = _leg_pair_corr_boost(
        _leg("A", "Over", "G1.PTS"), _leg("B", "Over", "G1.AST"), {}, team_mod, {}
    )
    _, boost_opp = _leg_pair_corr_boost(
        _leg("A", "Over", "G1.PTS"), _leg("B", "Under", "G1.AST"), {}, team_mod, {}
    )
    assert boost_same == pytest.approx(2.0)
    assert boost_opp == pytest.approx(0.5)


def test_committed_underdog_pair_modifiers_never_exceed_one() -> None:
    """Underdog pays table x pick multipliers x m with m <= 1 (docs/underdog_api.md §6.8),
    so a value above 1.0 is a bonus the app does not pay.

    Reads the committed file, not ``helpers.banned``: that one also carries the runtime
    overlay of owner-captured quotes, which stay authoritative.
    """
    combos = json.loads((pkg_resources.files(data) / "config" / "banned_combos.json").read_text())
    above_one = {
        f"{league} {relation} {pair}": modifiers
        for league, relations in combos["Underdog"].items()
        for relation, pairs in relations.items()
        for pair, modifiers in pairs.items()
        if max(modifiers) > 1.0
    }
    assert not above_one


def _matrix_game() -> tuple[pd.DataFrame, dict]:
    game_df = pd.DataFrame(
        {
            "Win Prob": [0.6, 0.5, 0.7],
            "Market Prob": [0.55, 0.48, 0.66],
            "Boost": [1.0, 1.0, 1.0],
            "Player": ["A", "B", "C"],
            "Bet": ["Over", "Over", "Over"],
            "cMarket": [["G1.PTS"], ["G2.AST"], ["C1.REB"]],
            # Deliberately not real stat_map["Underdog"] keys — keeps
            # _leg_shrinkage on its no-I/O 1.0 fallback path in these tests.
            "Market": ["Test1", "Test2", "Test3"],
        }
    )
    return game_df, game_df.to_dict("index")


def test_build_correlation_matrices_structure_and_values() -> None:
    """C/M are symmetric leg matrices; EV follows the documented closed form."""
    game_df, game_dict = _matrix_game()
    c_map = {("G1.PTS", "G2.AST"): 0.4, ("G2.AST", "C1.REB"): 0.2}

    g = _build_correlation_matrices(game_df, game_dict, c_map, {}, {}, [3.0], "Underdog", "NBA", {})
    C, M, EV, EVb, p_push = g.C, g.M, g.EV, g.EVb, g.p_push

    assert C.shape == (3, 3)
    assert np.allclose(np.diag(C), 1.0)  # np.eye base
    assert C[0, 1] == pytest.approx(0.4)
    assert C[1, 2] == pytest.approx(0.2)
    assert C[0, 2] == pytest.approx(0.0)  # no c_map entry
    assert np.allclose(C, C.T)

    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)  # only off-diagonals are set
    assert M[0, 1] == pytest.approx(1.0)  # different players, no banned mod

    assert np.allclose(p_push, 0.0)  # no Push P column → zeros
    # EV[0,1] = exp(C·V) · P · (boost·M·boost) · payout[0], hand-checked.
    v01 = np.sqrt(0.6 * 0.4 * 0.5 * 0.5)
    assert EV[0, 1] == pytest.approx(np.exp(0.4 * v01) * 0.3 * 1.0 * 3.0)
    assert np.allclose(EV, EV.T)
    assert EVb.shape == (3, 3)


def test_build_correlation_matrices_honors_push_column() -> None:
    """A present Push P column is carried through (NaNs filled with 0)."""
    game_df, game_dict = _matrix_game()
    game_df["Push Prob"] = [0.1, np.nan, 0.2]

    p_push = _build_correlation_matrices(
        game_df, game_dict, {}, {}, {}, [3.0], "Underdog", "NBA", {}
    ).p_push
    assert p_push.tolist() == [0.1, 0.0, 0.2]


def _annotated_game() -> tuple[pd.DataFrame, GameArrays]:
    """The matrix game as one night's frame, C correlated with both A and B."""
    game_df, game_dict = _matrix_game()
    game_df = game_df.assign(
        League="NBA", Team=["NYK", "SAS", "NYK"], Line=[24.5, 6.5, 9.5], Date="2026-06-03"
    )
    c_map = {("G1.PTS", "C1.REB"): 0.4, ("G2.AST", "C1.REB"): 0.2}
    g = _build_correlation_matrices(game_df, game_dict, c_map, {}, {}, [3.0], "Underdog", "NBA", {})
    return game_df, g


def test_annotate_names_a_partners_market_the_way_the_board_does() -> None:
    """The board run renames ``Market`` through the platform's map once scoring is done,
    and the deep dive looks each partner up on that board: stored under the platform's
    own label, no Underdog partner was ever found there. A label the map lacks stays as
    it is, the fall-through ``_collect_game_corr`` has."""
    game_df, g = _annotated_game()
    df = game_df.assign(**{"Corr Same": None, "Corr Opp": None})

    # Of the fixture's Test1..Test3 labels the map holds only A's.
    _annotate_correlation_columns(df, game_df, g, {"Test1": "PTS"})

    same, opp = df.loc[2, ["Corr Same", "Corr Opp"]]
    assert [(partner["player"], partner["market"]) for partner in same] == [("A", "PTS")]
    assert [(partner["player"], partner["market"]) for partner in opp] == [("B", "Test2")]


def test_annotate_leaves_a_players_row_of_another_date_alone() -> None:
    """A game's partner lists belong to that game's rows. Matched on player and market
    alone, they also landed on the same player's row of the series' next game."""
    game_df, g = _annotated_game()
    next_game = game_df.assign(Date="2026-06-05")
    df = pd.concat([game_df, next_game], ignore_index=True).assign(
        **{"Corr Same": None, "Corr Opp": None}
    )

    _annotate_correlation_columns(df, game_df, g, {})

    assert df.loc[df["Date"] == "2026-06-03", "Corr Same"].notna().all()
    assert df.loc[df["Date"] == "2026-06-05", ["Corr Same", "Corr Opp"]].isna().all(axis=None)


@pytest.mark.parametrize(
    ("league", "markets", "expected"),
    [
        ("NBA", ["Fantasy Points", "Rebounds", "H2H Points"], [0.163, 0.40, 1.0]),
        ("NHL", ["Assists", "Goals"], [0.30, 0.45]),
    ],
)
def test_leg_shrinkage_asks_a_cells_trust_under_the_leagues_own_market_name(
    monkeypatch, league, markets, expected
) -> None:
    """``stat_map`` names a market the same for every league and a cell goes by its
    league's own name: Underdog's NBA Fantasy Points is the ``fantasy points
    prizepicks`` cell, its NHL Assists the ``assists`` cell. Asked for under the
    mapped name no cell answers, and the no-evidence rung sizes every parlay holding
    the leg to nothing. REB and goals are already their league's own names; a market
    the platform map lacks is never asked for."""
    stub_cell_trust(
        monkeypatch,
        {
            ("NBA", "fantasy points prizepicks"): 0.163,
            ("NBA", "REB"): 0.40,
            ("NHL", "assists"): 0.30,
            ("NHL", "goals"): 0.45,
        },
    )

    shrinkage = _leg_shrinkage(pd.DataFrame({"Market": markets}), "Underdog", league, {})

    assert shrinkage.tolist() == expected


@_needs_nba_corr
def test_kernel_reads_real_nba_cmap() -> None:
    """End-to-end on real data: a real c_map entry flows through the kernel.

    ``_build_game_corr_map`` reads the packaged NBA parquets; the kernel must
    reproduce one of its (weighted) entries as the leg-pair correlation.
    """
    league_dir = pkg_resources.files(data) / "leagues" / "nba"
    c_same = pd.read_parquet(league_dir / "corr_same_team.parquet")
    c_same.rename_axis(["team", "market", "correlation"], inplace=True)
    c_same.columns = ["R"]
    c_opp = pd.read_parquet(league_dir / "corr_opposing.parquet")
    c_opp.rename_axis(["team", "market", "correlation"], inplace=True)
    c_opp.columns = ["R"]

    c_map = _build_game_corr_map("NYK", "SAS", c_same, c_opp)
    assert c_map, "real NBA c_map is empty — parquet vocabulary changed?"

    (x, y), expected = next(
        (k, v) for k, v in c_map.items() if k[0].split(".")[0] != k[1].split(".")[0]
    )
    rho, boost = _leg_pair_corr_boost(_leg("A", "Over", x), _leg("B", "Over", y), c_map, {}, {})
    assert rho == pytest.approx(expected)
    assert boost == 1
