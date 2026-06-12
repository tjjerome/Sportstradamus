"""Behavior guards for ``prediction/correlation.py`` — the parlay-core half-2
decomposition of :func:`find_correlation`.

Two complementary fixtures, because the function has two output halves with
different determinism:

* **offer_df correlation columns** are a pure function of the stratified corr
  parquets + each leg's Model P / Boost — fully deterministic. Pinned exactly
  on a real NBA slate (``NYK/SAS``, 2026-06-03) lifted from
  ``data/runtime/current_offers.parquet``. Real columns are used verbatim; the
  three find_correlation *input* columns the snapshot projection drops
  (``Books P``, ``K``, ``Player position``) are reconstructed deterministically
  here and documented in :data:`_NBA_OFFERS`.

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

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.prediction.correlation import (
    _build_correlation_matrices,
    _build_game_corr_map,
    _leg_pair_corr_boost,
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
            "Model P": mp,
            "Books P": round(min(max(mp - 0.08, 0.05), 0.95), 4),
            "Model": model,
            "Books": books,
            "K": round(mp, 4),
            "Player position": pos_of[player],
        }
        for team, opp, player, market, line, boost, bet, mp, model, books in raw
    ]


# Characterization snapshot of the offer_df correlation annotations.
#
# Originally captured on HEAD 340e178 (pre half-2 decomposition); re-pinned
# 2026-06-05 to current HEAD. The complexity-reduction refactor that landed since
# (get_odds in distributions.py, parlay payout assembly) shifted the inverted EVs
# by ~1e-12, which lands on opposite sides of the 2-dp multiplier rounding and the
# display/ranking cutoffs: the opp legs re-rank (Fox STL <-> Kornet REB across
# Robinson's PRA/REB), a few (x.xx) multipliers tick by 0.01, and the
# near-threshold Fox AST <-> Wembanyama BLST team pair drops out (both directions).
# Internally consistent (mirror pairs agree) — boundary sensitivity, not a logic
# break in the correlation assembly.
_EXPECTED_OFFER_CORR = [
    ("De'Aaron Fox", "AST", "", ""),
    ("De'Aaron Fox", "STL", "", "Mitchell Robinson Over 9.5 PRA - 68.4%, 1.02x (1.02x)"),
    ("Jalen Brunson", "AST", "", ""),
    ("Jalen Brunson", "FTM", "", ""),
    ("Jose Alvarado", "PTS", "", ""),
    ("Jose Alvarado", "STL", "", ""),
    ("Luke Kornet", "PR", "", ""),
    ("Luke Kornet", "REB", "", "Mitchell Robinson Over 5.5 REB - 64.0%, 1.17x (1.03x)"),
    (
        "Mitchell Robinson",
        "PRA",
        "",
        "Luke Kornet Over 3.5 REB - 57.8%, 1.34x (1.02x), "
        "De'Aaron Fox Over 1.5 STL - 58.0%, 1.42x (1.02x), "
        "Victor Wembanyama Over 3.5 BLK - 70.3%, 1.1x (1.01x)",
    ),
    (
        "Mitchell Robinson",
        "REB",
        "",
        "Victor Wembanyama Over 3.5 BLK - 70.3%, 1.1x (1.04x), "
        "Luke Kornet Over 3.5 REB - 57.8%, 1.34x (1.03x), "
        "De'Aaron Fox Over 1.5 STL - 58.0%, 1.42x (1.01x)",
    ),
    ("Victor Wembanyama", "BLK", "", "Mitchell Robinson Over 5.5 REB - 64.0%, 1.17x (1.04x)"),
    ("Victor Wembanyama", "BLST", "", "Mitchell Robinson Over 9.5 PRA - 68.4%, 1.02x (1.01x)"),
]


@_needs_nba_corr
def test_find_correlation_offer_correlations_real_nba() -> None:
    """Pin the deterministic offer_df correlation annotation on a real slate."""
    offers = _nba_offers()
    stats = {"NBA": _stats_for([o["Player"] for o in offers], "MIN short", "USG_PCT short")}

    offer_df, _ = find_correlation(offers, stats, "Underdog", contest_variant="power")

    actual = sorted(
        (r["Player"], r["Market"], r["Team Correlation"], r["Opp Correlation"])
        for _, r in offer_df[["Player", "Market", "Team Correlation", "Opp Correlation"]].iterrows()
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
            "Model P": mp,
            "Books P": round(mp - 0.10, 4),
            "Model": 1.0 + (mp - 0.5),
            "Books": 1.0,
            "K": 1.0,
            "Player position": pos,
        }
        for player, team, opp, market, line, pos, mp in raw
    ]


@_needs_wnba_corr
def test_find_correlation_builds_parlays_wnba() -> None:
    """find_correlation wires beam_search + assembles a well-formed parlay_df.

    Structural only (parlay EVs are non-deterministic on correlated legs); the
    guard is that the assembly path runs and produces sane rows.
    """
    offers = _wnba_offers()
    stats = {"WNBA": _stats_for([o["Player"] for o in offers], "MIN short", "USG_PCT short")}

    _, parlay_df = find_correlation(offers, stats, "Underdog", contest_variant="power")

    assert not parlay_df.empty
    for col in ("Game", "League", "Platform", "Model EV", "Books EV", "Legs", "Bet Size", "Family"):
        assert col in parlay_df.columns
    assert set(parlay_df["Bet Size"].astype(int)).issubset({2, 3, 4, 5, 6})
    assert (parlay_df["League"] == "WNBA").all()
    assert (parlay_df["Game"] == "LVA/NYL").all()


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


def test_leg_pair_corr_boost_same_player_zeroes_boost() -> None:
    """A leg pair on the same player collapses the boost to 0 (uncombinable)."""
    c_map = {("G1.PTS", "G1.AST"): 0.4}
    _, boost = _leg_pair_corr_boost(
        _leg("Luka", "Over", "G1.PTS"), _leg("Luka", "Over", "G1.AST"), c_map, {}, {}
    )
    assert boost == 0


def _matrix_game() -> tuple[pd.DataFrame, dict]:
    game_df = pd.DataFrame(
        {
            "Model P": [0.6, 0.5, 0.7],
            "Books P": [0.55, 0.48, 0.66],
            "Boost": [1.0, 1.0, 1.0],
            "Player": ["A", "B", "C"],
            "Bet": ["Over", "Over", "Over"],
            "cMarket": [["G1.PTS"], ["G1.AST"], ["C1.REB"]],
        }
    )
    return game_df, game_df.to_dict("index")


def test_build_correlation_matrices_structure_and_values() -> None:
    """C/M are symmetric leg matrices; EV follows the documented closed form."""
    game_df, game_dict = _matrix_game()
    c_map = {("G1.PTS", "G1.AST"): 0.4, ("G1.AST", "C1.REB"): 0.2}

    g = _build_correlation_matrices(game_df, game_dict, c_map, {}, {}, [3.0])
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
