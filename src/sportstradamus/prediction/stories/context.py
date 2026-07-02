"""Per-game context for the prophecy engine: shape, spread, favorite, edges.

``build_game_context`` aggregates the finished offers frame into one row per
``(League, Game, Date)`` — the data the v2 thesis archetype router needs that a
single offer row can't carry. It is pure (no I/O, no archive) and runs over the
post-scoring ``snapshot_offers`` at prophecize time; the row set is persisted to
``current_game_context.parquet`` and the dashboard rebuilds :class:`GameCtx`
objects from it.

``classify_shape`` names the game script from a *league-relative* total ratio
plus the win-probability margin. It replaces v1's per-league ``_TOTAL_BANDS``,
which mis-fired in production: the snapshot ``O/U`` column is the team-implied
half-total (~108 NBA, ~82 WNBA — ``archive.get_total`` stores ``(total ±
spread)/2`` per team), not the raw game total the absolute bands expected, so
every non-lopsided game classified as a grind. Working in ratio space makes one
classifier serve every league.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import pandas as pd

_CTX_COLS = [
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

# League-relative game-shape bands over ``game_total / baseline``. A slate plays
# as a shootout at/above the high band, a rockfight at/below the low band. NBA's
# hand-written v1 points bands imply ≈1.05 / ≈0.96 against a ~223 baseline, so
# these reproduce v1's intent without any per-league table.
_SHOOTOUT_RATIO: float = 1.05
_GRIND_RATIO: float = 0.95

# How far a side's implied win probability must sit from a coin flip (0.5) for
# the game to read as lopsided (blowout) rather than tight. 0.18 ⇒ ~68% favorite.
# Lives in probability space, so it is already league-agnostic (kept from v1).
_ML_LOPSIDED_MARGIN: float = 0.18

# Minimum games on a league's slate before the slate-median game total is a
# trustworthy baseline; below it, fall back to twice the archive's half-total.
_BASELINE_MIN_GAMES: int = 4


@dataclass(frozen=True)
class Leg:
    """One enriched parlay leg the archetype router reads (no DataFrame access).

    ``negative`` marks a market whose Under is the thriving side (narrative
    valence flips); ``win_prob`` is the offer's model hit probability, the
    anchor-conviction tiebreaker.
    """

    player: str
    bet: str
    line: float
    market: str
    game: str | None = None
    team: str | None = None
    position: str | None = None
    category: str = "production"
    negative: bool = False
    win_prob: float | None = None


@dataclass(frozen=True)
class GameCtx:
    """Per-game context the engine routes on, rebuilt from the snapshot parquets.

    ``pos_edges`` is ``{team: {pos_group: {"dvpoa": mean, "n": count}}}`` and
    ``rho`` maps a leg pair (``frozenset`` of two ``"Player|Market|Bet"`` keys)
    to its correlation, both keyed to this one game.
    """

    league: str
    game: str
    date: str = ""
    shape: str = "even"
    game_total: float | None = None
    total_ratio: float | None = None
    fav_team: str | None = None
    ml_margin: float | None = None
    spread: float | None = None
    pos_edges: Mapping[str, Mapping[str, dict]] = field(default_factory=dict)
    rho: Mapping[frozenset, float] = field(default_factory=dict)


def ctx_from_row(row: Mapping) -> GameCtx:
    """Rebuild a :class:`GameCtx` (sans ``rho``) from one context-parquet row."""
    pe = row.get("pos_edges")
    pos_edges = json.loads(pe) if isinstance(pe, str) and pe.strip() else {}
    fav = row.get("fav_team")
    if isinstance(fav, float) and math.isnan(fav):
        fav = None
    return GameCtx(
        league=str(row.get("League", "")),
        game=str(row.get("Game", "")),
        date=str(row.get("Date", "")),
        shape=str(row.get("shape") or "even"),
        game_total=_none_if_nan(row.get("game_total")),
        total_ratio=_none_if_nan(row.get("total_ratio")),
        fav_team=fav,
        ml_margin=_none_if_nan(row.get("ml_margin")),
        spread=_none_if_nan(row.get("spread")),
        pos_edges=pos_edges,
    )


def ctxs_from_frame(context: pd.DataFrame, corr=None) -> dict[str, GameCtx]:
    """Map ``Game`` → :class:`GameCtx` from the context frame + corr slices.

    The correlation slices (``current_game_corr`` rows) attach per-game ``rho``;
    a game present only in the corr slice still gets a minimal context.
    """
    rho_by_game = _rho_by_game(corr)
    out: dict[str, GameCtx] = {}
    if context is not None and not context.empty:
        for _, row in context.iterrows():
            base = ctx_from_row(row)
            out[base.game] = replace(base, rho=rho_by_game.get(base.game, {}))
    for game, rho in rho_by_game.items():
        if game not in out:
            out[game] = GameCtx(league="", game=game, rho=rho)
    return out


def _rho_by_game(corr) -> dict[str, dict[frozenset, float]]:
    out: dict[str, dict] = defaultdict(dict)
    if corr is None:
        return out
    rows = corr.to_dict("records") if isinstance(corr, pd.DataFrame) else corr
    for r in rows:
        out[r["Game"]][frozenset({r["leg_a"], r["leg_b"]})] = r["rho"]
    return out


def classify_shape(total_ratio: float | None, ml_margin: float | None) -> str:
    """Name the game script from the league-relative total ratio and win margin.

    A lopsided win probability reads as a blowout regardless of total (checked
    first, matching v1); otherwise a high/low total ratio reads shootout/grind,
    a tight game with a middling total reads coinflip, and anything left is even.
    """
    if ml_margin is not None and ml_margin >= _ML_LOPSIDED_MARGIN:
        return "blowout"
    if total_ratio is not None and total_ratio >= _SHOOTOUT_RATIO:
        return "shootout"
    if total_ratio is not None and total_ratio <= _GRIND_RATIO:
        return "grind"
    if ml_margin is not None and ml_margin < _ML_LOPSIDED_MARGIN:
        return "coinflip"
    return "even"


def build_game_context(offers: pd.DataFrame, default_totals: dict[str, float]) -> pd.DataFrame:
    """One context row per ``(League, Game, Date)`` over the scored offers frame.

    Pure aggregation, no archive: ``game_total`` is the sum of the two teams'
    median ``O/U`` half-totals, ``spread`` their difference, the favorite the
    side with the higher ``Moneyline`` win probability, and ``pos_edges`` the
    mean ``DVPOA`` per ``(team, position-group)``. The league baseline is the
    slate-median game total once a league has ``_BASELINE_MIN_GAMES`` games, else
    twice the archive's half-total, and ``shape`` rides the resulting ratio.
    """
    if offers is None or offers.empty or "Game" not in offers.columns:
        return pd.DataFrame(columns=_CTX_COLS)

    rows = [
        _game_row(key, grp) for key, grp in offers.groupby(["League", "Game", "Date"], dropna=False)
    ]
    df = pd.DataFrame(rows)
    baselines = _league_baselines(df, default_totals)
    df["baseline_total"] = df["League"].map(baselines)
    df["total_ratio"] = df["game_total"] / df["baseline_total"]
    df["shape"] = [
        classify_shape(_none_if_nan(tr), _none_if_nan(mm))
        for tr, mm in zip(df["total_ratio"], df["ml_margin"], strict=True)
    ]
    return df[_CTX_COLS]


def _game_row(key: tuple, grp: pd.DataFrame) -> dict:
    league, game, date = key
    # groupby keeps offers' index, which can repeat a label across two teams; reset
    # it so the favorite lookup (grp.loc[mls.idxmax()]) returns a scalar, not a Series.
    grp = grp.reset_index(drop=True)
    team_totals = (
        grp.groupby("Team")["O/U"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").median())
        .dropna()
        .to_dict()
    )
    if len(team_totals) >= 2:
        vals = list(team_totals.values())
        game_total = float(sum(vals))
        spread = float(abs(max(vals) - min(vals)))
    else:
        game_total = spread = float("nan")

    mls = (
        pd.to_numeric(grp["Moneyline"], errors="coerce").dropna()
        if "Moneyline" in grp.columns
        else pd.Series(dtype=float)
    )
    if not mls.empty:
        ml_fav_prob = float(mls.max())
        ml_margin = float((mls - 0.5).abs().max())
        fav_team = grp.loc[mls.idxmax(), "Team"]
    else:
        ml_fav_prob = ml_margin = float("nan")
        fav_team = None

    return {
        "League": league,
        "Game": game,
        "Date": date,
        "game_total": game_total,
        "spread": spread,
        "fav_team": fav_team,
        "ml_fav_prob": ml_fav_prob,
        "ml_margin": ml_margin,
        "pos_edges": _pos_edges(grp),
        "n_offers": len(grp),
    }


def _pos_edges(grp: pd.DataFrame) -> str:
    """JSON ``{team: {pos_group: {dvpoa, n}}}`` of mean DVPOA by position group.

    Position group is the depth-chart label minus its rank digit (``G1``→``G``,
    ``QB1``→``QB``). Combo legs carry an empty ``Position`` and drop out.
    """
    if "Position" not in grp.columns or "DVPOA" not in grp.columns:
        return "{}"
    sub = grp[["Team", "Position", "DVPOA"]].copy()
    sub["Position"] = sub["Position"].astype("string").fillna("").str.strip()
    sub = sub[(sub["Position"] != "") & (sub["Position"].str.lower() != "nan")]
    sub["DVPOA"] = pd.to_numeric(sub["DVPOA"], errors="coerce")
    sub = sub.dropna(subset=["DVPOA"])
    if sub.empty:
        return "{}"
    sub["grp"] = sub["Position"].str.replace(r"\d+$", "", regex=True)
    out: dict[str, dict] = {}
    for (team, group), gg in sub.groupby(["Team", "grp"]):
        out.setdefault(team, {})[group] = {
            "dvpoa": round(float(gg["DVPOA"].mean()), 4),
            "n": len(gg),
        }
    return json.dumps(out, sort_keys=True)


def _league_baselines(df: pd.DataFrame, default_totals: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for league, g in df.groupby("League"):
        totals = g["game_total"].dropna()
        base = default_totals.get(league)
        if len(g) >= _BASELINE_MIN_GAMES and not totals.empty:
            out[league] = float(totals.median())
        elif base is not None:
            out[league] = 2.0 * float(base)
        else:
            out[league] = float(totals.median()) if not totals.empty else float("nan")
    return out


def _none_if_nan(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
