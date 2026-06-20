"""Hash-stable pins for ``prediction.stories.context`` — the P2 game-context layer.

``build_game_context`` aggregates the finished offers frame into one row per
``(League, Game, Date)``: the league-relative game shape, derived spread,
favorite, and positional DVPOA edges the v2 thesis engine routes on.
``classify_shape`` is the league-agnostic ratio-band classifier that replaces
v1's per-league ``_TOTAL_BANDS`` (which mis-fired because the snapshot ``O/U``
is the team half-total, not the raw game total).
"""

from __future__ import annotations

import io
import json

import pandas as pd

from sportstradamus.prediction.stories.context import (
    GameCtx,
    Leg,
    build_game_context,
    classify_shape,
    ctx_from_row,
    ctxs_from_frame,
)


def test_classify_shape_ratio_bands():
    assert classify_shape(1.10, 0.05) == "shootout"
    assert classify_shape(0.90, 0.05) == "grind"
    assert classify_shape(1.00, 0.05) == "coinflip"
    assert classify_shape(1.00, None) == "even"
    assert classify_shape(None, None) == "even"
    # A lopsided win probability beats the total band either way.
    assert classify_shape(1.00, 0.25) == "blowout"
    assert classify_shape(1.10, 0.25) == "blowout"


def test_build_game_context_two_team_game():
    """One game: total = sum of the two team half-totals, spread = their diff,
    favorite = higher win prob, positional DVPOA edges aggregated by group."""
    offers = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "BOS/PHI",
                "Date": "2026-06-13",
                "Team": "BOS",
                "Opponent": "PHI",
                "O/U": 120.0,
                "Moneyline": 0.74,
                "DVPOA": 0.10,
                "Position": "F1",
                "Player": "A",
                "Market": "PTS",
            },
            {
                "League": "NBA",
                "Game": "BOS/PHI",
                "Date": "2026-06-13",
                "Team": "BOS",
                "Opponent": "PHI",
                "O/U": 120.0,
                "Moneyline": 0.74,
                "DVPOA": 0.20,
                "Position": "F1",
                "Player": "B",
                "Market": "REB",
            },
            {
                "League": "NBA",
                "Game": "BOS/PHI",
                "Date": "2026-06-13",
                "Team": "PHI",
                "Opponent": "BOS",
                "O/U": 118.0,
                "Moneyline": 0.26,
                "DVPOA": -0.05,
                "Position": "C1",
                "Player": "C",
                "Market": "PTS",
            },
        ]
    )
    ctx = build_game_context(offers, {"NBA": 111.667})
    assert len(ctx) == 1
    row = ctx.iloc[0]
    assert row["game_total"] == 238.0
    assert row["spread"] == 2.0
    assert row["fav_team"] == "BOS"
    assert row["ml_fav_prob"] == 0.74
    assert round(row["ml_margin"], 2) == 0.24
    assert row["n_offers"] == 3
    # 1 game (< 4) ⇒ baseline = 2 × 111.667 ≈ 223.33; ratio ≈ 1.066 ⇒ shootout band,
    # but the 0.24 win-prob margin is lopsided ⇒ blowout wins.
    assert row["shape"] == "blowout"
    edges = json.loads(row["pos_edges"])
    assert edges["BOS"]["F"]["n"] == 2
    assert round(edges["BOS"]["F"]["dvpoa"], 3) == 0.15
    assert edges["PHI"]["C"]["n"] == 1


def test_build_game_context_duplicate_index_favorite_is_scalar():
    """Regression: ``groupby`` keeps offers' index, which can repeat a label across
    two teams (a real slate hit ``23 SAS`` / ``23 NYK``). The favorite lookup must
    stay a scalar so the context row serializes to parquet instead of carrying a
    two-row ``Team`` Series."""
    offers = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "SAS/NYK",
                "Date": "2026-06-13",
                "Team": "SAS",
                "Opponent": "NYK",
                "O/U": 112.0,
                "Moneyline": 0.40,
                "Player": "A",
                "Market": "PTS",
            },
            {
                "League": "NBA",
                "Game": "SAS/NYK",
                "Date": "2026-06-13",
                "Team": "NYK",
                "Opponent": "SAS",
                "O/U": 116.0,
                "Moneyline": 0.60,
                "Player": "B",
                "Market": "PTS",
            },
        ],
        index=[23, 23],  # the non-unique label that produced the Series-not-scalar crash
    )
    ctx = build_game_context(offers, {"NBA": 111.667})
    row = ctx.iloc[0]
    assert row["fav_team"] == "NYK" and isinstance(row["fav_team"], str)
    ctx.to_parquet(io.BytesIO(), engine="pyarrow", index=False)  # must not raise ArrowInvalid


def test_build_game_context_slate_median_baseline():
    """Four games on a league slate ⇒ baseline is the slate-median game total,
    and the shape rides the league-relative ratio when the game is tight."""
    rows = []
    for game, half in [
        ("AAA/BBB", 100.0),
        ("CCC/DDD", 110.0),
        ("EEE/FFF", 120.0),
        ("GGG/HHH", 130.0),
    ]:
        a, b = game.split("/")
        rows += [
            {
                "League": "NBA",
                "Game": game,
                "Date": "2026-06-13",
                "Team": a,
                "Opponent": b,
                "O/U": half,
                "Moneyline": 0.5,
                "Player": "x",
                "Market": "PTS",
            },
            {
                "League": "NBA",
                "Game": game,
                "Date": "2026-06-13",
                "Team": b,
                "Opponent": a,
                "O/U": half,
                "Moneyline": 0.5,
                "Player": "y",
                "Market": "PTS",
            },
        ]
    ctx = build_game_context(pd.DataFrame(rows), {"NBA": 111.667}).set_index("Game")
    assert (ctx["baseline_total"] == 230.0).all()  # median of 200/220/240/260
    assert ctx.loc["AAA/BBB", "shape"] == "grind"  # 200/230 = 0.87
    assert ctx.loc["GGG/HHH", "shape"] == "shootout"  # 260/230 = 1.13


def test_ctx_from_row_parses_pos_edges():
    row = {
        "League": "NBA",
        "Game": "BOS/PHI",
        "Date": "2026-06-13",
        "shape": "blowout",
        "game_total": 238.0,
        "total_ratio": 1.07,
        "fav_team": "BOS",
        "ml_margin": 0.24,
        "spread": 2.0,
        "pos_edges": '{"BOS": {"F": {"dvpoa": 0.15, "n": 2}}}',
    }
    ctx = ctx_from_row(row)
    assert isinstance(ctx, GameCtx)
    assert ctx.league == "NBA" and ctx.game == "BOS/PHI" and ctx.shape == "blowout"
    assert ctx.fav_team == "BOS" and ctx.spread == 2.0
    assert ctx.pos_edges["BOS"]["F"]["dvpoa"] == 0.15
    assert ctx.rho == {}


def test_ctxs_from_frame_keys_by_game_and_attaches_rho():
    offers = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "BOS/PHI",
                "Date": "2026-06-13",
                "Team": "BOS",
                "Opponent": "PHI",
                "O/U": 120.0,
                "Moneyline": 0.55,
                "Player": "A",
                "Market": "PTS",
            },
            {
                "League": "NBA",
                "Game": "BOS/PHI",
                "Date": "2026-06-13",
                "Team": "PHI",
                "Opponent": "BOS",
                "O/U": 118.0,
                "Moneyline": 0.45,
                "Player": "B",
                "Market": "PTS",
            },
        ]
    )
    ctx_df = build_game_context(offers, {"NBA": 111.667})
    corr = [
        {
            "League": "NBA",
            "Game": "BOS/PHI",
            "leg_a": "A|PTS|Over",
            "leg_b": "B|REB|Over",
            "rho": 0.4,
        }
    ]
    ctxs = ctxs_from_frame(ctx_df, corr)
    assert set(ctxs) == {"BOS/PHI"}
    assert ctxs["BOS/PHI"].rho[frozenset({"A|PTS|Over", "B|REB|Over"})] == 0.4


def test_leg_dataclass_is_frozen():
    leg = Leg(player="A", bet="Over", line=5.5, market="PTS", category="scoring")
    assert leg.player == "A" and leg.category == "scoring"
    assert leg.game is None  # unenriched default


def test_build_game_context_empty_safe():
    out = build_game_context(pd.DataFrame(), {"NBA": 111.667})
    assert list(out.columns) == [
        "League",
        "Game",
        "Date",
        "game_total",
        "spread",
        "fav_team",
        "ml_fav_prob",
        "ml_margin",
        "total_ratio",
        "baseline_total",
        "shape",
        "pos_edges",
        "n_offers",
    ]
    assert out.empty
