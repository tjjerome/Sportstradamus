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

import datetime
import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from sportstradamus import data
from sportstradamus.leg_schema import LEG_FIELDS
from sportstradamus.prediction import correlation
from sportstradamus.prediction.correlation import (
    _build_cmarket,
    _build_correlation_matrices,
    _build_game_corr_map,
    _collect_game_corr,
    _leg_pair_corr_boost,
    _resolve_player_positions,
    find_correlation,
)


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


# --- real NBA slate (offer-correlation characterization) --------------------


# Real NYK/SAS offers (2026-06-03) from data/runtime/current_offers.parquet.
# League/Date/Team/Opponent/Player/Market/Line/Boost/Bet/Model P/Model/Books are
# verbatim. Books P = clip(Model P - 0.08, .05, .95) (book-prob proxy; snapshot
# drops it), K = round(Model P, 4) (Kelly sort key; snapshot drops it),
# Player position = first-seen-player index mod 5 + 1 → positions["NBA"]
# (snapshot drops it). Processed as Underdog (Boost is /UNDERDOG_BOOST_BASELINE
# normalized, so displayed multipliers are the real boost ÷ 1.78).
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
            "Boost": boost,
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
# the c_map is fixed — 8/12 legs annotate, with the Corr Same and Corr Opp columns both
# exercised. Each partner is the structured record ``_annotate_correlation_columns`` writes
# (player/market/bet/line/mult) — ``leg_schema.leg_label`` renders it to a string on demand,
# so the test pins the structured fields directly rather than a formatted label.
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
            {
                "player": "Mitchell Robinson",
                "market": "REB",
                "bet": "Over",
                "line": 5.5,
                "mult": 1.06,
            }
        ],
    ),
    ("Jalen Brunson", "AST", [], []),
    ("Jalen Brunson", "FTM", [], []),
    ("Jose Alvarado", "PTS", [], []),
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
            {
                "player": "Mitchell Robinson",
                "market": "PRA",
                "bet": "Over",
                "line": 9.5,
                "mult": 1.03,
            }
        ],
    ),
]


def test_find_correlation_offer_correlations_synthetic(monkeypatch) -> None:
    """Pin the deterministic offer_df correlation annotation: the real NBA offer slate driven by
    a FIXED synthetic c_map (and a stubbed corr-parquet read), so it no longer false-positives on
    the drifting, gitignored correlation parquets and runs everywhere (no real data needed)."""
    monkeypatch.setattr(
        correlation, "_build_game_corr_map", lambda *a, **k: _synthetic_offer_cmap()
    )
    real_read = pd.read_parquet

    def _stub_corr_read(path, *args, **kwargs):
        if str(path).endswith(("corr_same_team.parquet", "corr_opposing.parquet")):
            return pd.DataFrame({0: []}, index=pd.MultiIndex.from_arrays([[], [], []]))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _stub_corr_read)

    offers = _nba_offers()
    stats = {"NBA": _stats_for([o["Player"] for o in offers], "MIN short", "USG_PCT short")}

    offer_df, _ = find_correlation(offers, stats, "Underdog", contest_variant="power")

    actual = sorted(
        (r["Player"], r["Market"], r["Corr Same"] or [], r["Corr Opp"] or [])
        for _, r in offer_df[["Player", "Market", "Corr Same", "Corr Opp"]].iterrows()
    )
    assert actual == sorted(_EXPECTED_OFFER_CORR)


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
            "Boost": 1.78,  # == UNDERDOG_BOOST_BASELINE → post-normalization 1.0
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
    ``Desc`` / ``Family`` columns; ``Fun`` survives as a transient sort key that
    ``persist._PARLAY_DROP_COLS`` strips before ``current_parlays.parquet``.
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
    absent from the ``depth`` column.
    """

    def __init__(self, slots: dict[str, int]) -> None:
        self.playerProfile = pd.DataFrame(index=pd.Index(list(slots), name="playerName"))
        self._slots = slots
        self.depth_calls: list[tuple[list[dict], object]] = []

    def get_depth(self, offers, date):
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
                ("Oneil Cruz", "PIT"),
            ]
        ]
    )


def test_resolve_player_positions_mlb_labels_batting_slots() -> None:
    """MLB slots come from ``get_depth``, not the always-zero offer column.

    Reading the offer column labeled every MLB leg ``P``, so every batter pair
    looked up a missing correlation row and scored rho 0. Slots 1-9 become
    ``B1``..``B9``; a pitcher (slot 0) and a hitter the profile never resolved
    both fall through to ``P``. Combo legs keep the one-label-per-component list
    shape ``_build_cmarket`` expects.
    """
    stat_data = _FakeMLBStats({"Kyle Tucker": 3, "Nico Hoerner": 9, "Shota Imanaga": 0})

    resolved = _resolve_player_positions(_mlb_legs(), "MLB", stat_data)

    assert list(resolved["Player position"]) == ["B3", "B9", "P", ["B3", "P"], "P"]
    assert len(stat_data.depth_calls) == 1  # once per game date, not once per leg
    records, call_date = stat_data.depth_calls[0]
    assert records == [
        {"Player": "Kyle Tucker", "Team": "CHC"},
        {"Player": "Nico Hoerner", "Team": "CHC"},
        {"Player": "Shota Imanaga", "Team": "CHC"},
        {"Player": "Kyle Tucker + Shota Imanaga", "Team": "CHC"},
        {"Player": "Oneil Cruz", "Team": "PIT"},
    ]
    # get_depth compares `date < datetime.today().date()`, which raises on a datetime.
    assert type(call_date) is datetime.date
    assert call_date == datetime.date(2026, 9, 4)


def test_build_cmarket_mlb_keys_by_batting_slot() -> None:
    """cMarket tokens must key the way the MLB corr parquets do: ``B{n}``/``P``."""
    stat_data = _FakeMLBStats({"Kyle Tucker": 3, "Nico Hoerner": 9, "Shota Imanaga": 0})
    resolved = _resolve_player_positions(_mlb_legs(), "MLB", stat_data)

    cmarket = _build_cmarket(resolved, "MLB", {})["cMarket"]

    assert list(cmarket) == [
        ["B3.hits"],
        ["B9.hits"],
        ["P.hits"],
        ["B3.hits", "P.hits"],
        ["P.hits"],
    ]


# Underdog Rivals builds MLB "A vs. B" legs, and unlike the other leagues' combo
# legs those survive into the correlation stage, so the slate carries one.
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
            "Boost": 1.78,  # == UNDERDOG_BOOST_BASELINE -> post-normalization 1.0
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
    mixing lists with strings, so a single Rivals leg would take prophecize down.
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
    market_map = {"Points": "PTS", "Rebounds": "REB"}

    rows = _collect_game_corr(game_df, C, "WNBA", "LVA/NYL", market_map)

    assert len(rows) == 3  # 3 distinct legs → 3 unique pairs
    for r in rows:
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
    rows = _collect_game_corr(game_df, C, "WNBA", "LVA/NYL", {"Points": "PTS"})
    assert rows == []


# --- pure-kernel unit tests -------------------------------------------------


# Two single-position legs whose cMarket tokens key into the c_map below.
def _leg(player: str, bet: str, cmarket: str) -> dict:
    return {"Player": player, "Bet": bet, "cMarket": [cmarket]}


def test_leg_pair_corr_boost_reads_cmap_value() -> None:
    """Same-bet pair on different players: rho is the c_map entry, boost 1."""
    c_map = {("G1.PTS", "G1.AST"): 0.4}
    rho, boost = _leg_pair_corr_boost(
        _leg("A", "Over", "G1.PTS"), _leg("B", "Over", "G1.AST"), c_map, {}, {}
    )
    assert rho == pytest.approx(0.4)
    assert boost == 1


def test_leg_pair_corr_boost_opposite_bet_flips_sign() -> None:
    """Over vs Under negates the correlation increment."""
    c_map = {("G1.PTS", "G1.AST"): 0.4}
    rho, _ = _leg_pair_corr_boost(
        _leg("A", "Over", "G1.PTS"), _leg("B", "Under", "G1.AST"), c_map, {}, {}
    )
    assert rho == pytest.approx(-0.4)


def test_leg_pair_corr_boost_reversed_key_lookup() -> None:
    """The (y, x) fallback lookup finds a pair stored in the other order."""
    c_map = {("G1.AST", "G1.PTS"): 0.25}
    rho, _ = _leg_pair_corr_boost(
        _leg("A", "Over", "G1.PTS"), _leg("B", "Over", "G1.AST"), c_map, {}, {}
    )
    assert rho == pytest.approx(0.25)


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


def _matrix_game() -> tuple[pd.DataFrame, dict]:
    game_df = pd.DataFrame(
        {
            "Win Prob": [0.6, 0.5, 0.7],
            "Market Prob": [0.55, 0.48, 0.66],
            "Boost": [1.0, 1.0, 1.0],
            "Player": ["A", "B", "C"],
            "Bet": ["Over", "Over", "Over"],
            "cMarket": [["G1.PTS"], ["G1.AST"], ["C1.REB"]],
            # Deliberately not real stat_map["Underdog"] keys — keeps
            # _leg_shrinkage on its no-I/O 1.0 fallback path in these tests.
            "Market": ["Test1", "Test2", "Test3"],
        }
    )
    return game_df, game_df.to_dict("index")


def test_build_correlation_matrices_structure_and_values() -> None:
    """C/M are symmetric leg matrices; EV follows the documented closed form."""
    game_df, game_dict = _matrix_game()
    c_map = {("G1.PTS", "G1.AST"): 0.4, ("G1.AST", "C1.REB"): 0.2}

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

    (x, y), expected = next((k, v) for k, v in c_map.items() if k[0] != k[1])
    rho, boost = _leg_pair_corr_boost(_leg("A", "Over", x), _leg("B", "Over", y), c_map, {}, {})
    assert rho == pytest.approx(expected)
    assert boost == 1
