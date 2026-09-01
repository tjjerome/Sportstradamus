"""The components behind a combo cell: which ones, at what weight, decoded per row.

The input side of Lane B's component sum (see :mod:`training.component_sum`).
Weights come from the same tables the serving path uses — ``helpers.config.combo_props``
and each league's own ``Stats._fantasy_combo_spec`` — never a copy, which would drift
from ``tests/golden/test_combo_weights_settlement.py``. The per-row predictives come
from each component cell's ``data/test_sets/{LEAGUE}_{slug}.csv``: its test split
only, so every row is out-of-sample for that cell's own model.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from sportstradamus.helpers.config import combo_props, stat_cv, stat_meta
from sportstradamus.helpers.distributions import _dp_mean
from sportstradamus.stats import Stats, StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA
from sportstradamus.stats.mlb import FANTASY_HBP_WEIGHTS, LEAGUE_HBP_PER_GAME
from sportstradamus.training.scorecard import (
    _decode_sn_loc_scale,
    _decode_strategy_for_frame,
    _infer_dist_from_columns,
    _pred_midpit,
    load_test_set,
)

# Unloaded league instances, used only for `_fantasy_combo_spec`'s pure per-league
# weight tables. MLB's constructor otherwise fetches today's probable pitchers.
_LEAGUE_STATS: dict[str, Callable[[], Stats]] = {
    "MLB": partial(StatsMLB, load_live_pitchers=False),
    "NBA": StatsNBA,
    "NFL": StatsNFL,
    "NHL": StatsNHL,
    "WNBA": StatsWNBA,
}

# NFL's `tds` cell trains on {WR, RB, TE} only (NFL_MARKET_POSITIONS), so a QB has
# zero rows in it and the model-fed join collapses to nothing at that term. Its
# `rushing tds` cell covers {QB, RB}, and a QB's receiving TDs are ~0, so the two
# settle the same for that position at the same weight. Lane A's book path needs no
# such swap — sportsbooks quote QB anytime-TD directly.
_QB_COMPONENT_CELL = {"tds": "rushing tds"}

# `ComboComponent`'s optional per-family shape fields, in the order they are decoded
# out of a component CSV and read back per row.
SHAPE_FIELDS = ("sigma", "skew", "gate", "phi", "r")

# Keeps norm.ppf finite where a mid-PIT lands on the 0/1 boundary of a count lattice.
_PIT_CLIP = 1e-9

Weights = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ComponentCell:
    """One component cell's served predictive, decoded into kernel parameters.

    ``params`` is indexed by ``(Player, Date)`` and holds ``mean`` (the BASE,
    pre-gate mean the kernel deflates itself), the five optional shape fields
    (NaN where the family has none), and ``z``, the normal score of the row's own
    mid-PIT that the model-rho estimator correlates across components.
    """

    market: str
    dist: str
    cv: float
    params: pd.DataFrame


def _fantasy_weights(
    league: str, market: str, players: list[str], combo: pd.DataFrame
) -> tuple[dict[str, Weights], list[str]]:
    """Per-player component weights from the league's own ``_fantasy_combo_spec``.

    The spec tables read no instance state except NFL's, which selects components off
    ``self.players["position"]`` — seeded here from the combo matrix's own numeric
    position column. A spec carrying sampled, Bernoulli or post-hook terms has no
    model-fed equivalent (MLB pitcher win, the quality-start indicator, NHL goalie
    win), so the cell is refused rather than priced from a partial spec.
    """
    stats = _LEAGUE_STATS[league]()
    position_of: dict[str, str] = {}
    if league == "NFL":
        codes = pd.to_numeric(combo["Player position"], errors="coerce")
        named = codes.map(dict(enumerate(stats.positions, start=1)))
        position_of = dict(zip(combo["Player"], named, strict=True))
        stats.players = pd.DataFrame({"position": pd.Series(position_of)})
    specs: dict[str, Weights] = {}
    labels = ["fantasy_combo_spec"]
    for player in players:
        spec = stats._fantasy_combo_spec(market, player)
        if spec is None:
            continue
        if spec.sampled or spec.bernoulli or spec.post_builder:
            return {}, [f"{market} spec needs terms no component cell models"]
        marginals = spec.marginals
        if position_of.get(player) == "QB":
            marginals = tuple((_QB_COMPONENT_CELL.get(sub, sub), w) for sub, w in marginals)
            if marginals != spec.marginals and labels[-1] != "qb_tds_via_rushing_tds":
                labels.append("qb_tds_via_rushing_tds")
        specs[player] = marginals
    return specs, labels


def _modeled_only(league: str, specs: dict[str, Weights]) -> tuple[dict[str, Weights], list[str]]:
    """Drop spec components the system models nowhere, naming them in provenance.

    MLB ``triples`` is the only one: no ``stat_meta`` cell, no pickle, no test set.
    Its term is weight 8 on a ~0.005/game rate, ~0.04 of a 9.6-point fantasy mean.
    """
    modeled = set(stat_meta[league])
    unmodeled = sorted({sub for weights in specs.values() for sub, _ in weights} - modeled)
    if not unmodeled:
        return specs, []
    trimmed = {
        player: tuple((sub, w) for sub, w in weights if sub in modeled)
        for player, weights in specs.items()
    }
    return trimmed, ["omitted_unmodeled:" + ",".join(unmodeled)]


def spec_weights(
    league: str, market: str, combo: pd.DataFrame
) -> tuple[dict[str, Weights], float, list[str]]:
    """Per-player component weights, a deterministic mean offset, and provenance.

    Simple ``combo_props`` cells take unit weights. MLB hitter fantasy is the one
    deliberate divergence from the Lane A spec: it prices the hit types from their
    own model cells instead of Lane A's compound-multinomial split of a sampled
    ``hits`` — books quote no hit types, models do — while keeping Lane A's
    deterministic hit-by-pitch mean offset. Everything else goes through
    :func:`_fantasy_weights`. An empty result means the cell has no expressible
    model-fed spec, and ``provenance`` says why.
    """
    players = list(dict.fromkeys(combo["Player"]))
    offset = 0.0
    if market in combo_props:
        specs = dict.fromkeys(players, tuple((sub, 1.0) for sub in combo_props[market]))
        labels = ["combo_props"]
    elif league == "MLB" and market in FANTASY_HBP_WEIGHTS:
        weights = tuple((sub, float(w)) for sub, w in StatsMLB._mlb_fantasy_props(market))
        specs = dict.fromkeys(players, weights)
        offset = FANTASY_HBP_WEIGHTS[market] * LEAGUE_HBP_PER_GAME
        labels = ["hit_types_direct", "hbp_offset"]
    else:
        specs, labels = _fantasy_weights(league, market, players, combo)
    specs, omitted = _modeled_only(league, specs)
    return specs, offset, labels + omitted


def load_component_cell(league: str, market: str, path: Path) -> ComponentCell:
    """Decode one component cell's test-set CSV into per-row kernel parameters.

    The mean handed to the kernel is the BASE distribution mean — ``_effective_gate``
    deflates it there — matching how ``get_odds`` and the scorecard both read
    ``(R, NB_P)`` plus a separate ``Gate`` atom. SkewNormal locs and scales are
    decoded out of the cell's normalized target space first, then converted to that
    family's mean, which the kernel re-inverts back to ``loc``. The family is read off
    the persisted columns rather than ``stat_meta`` so a config that moved after the
    CSV was written cannot silently mis-decode it.
    """
    df = load_test_set(path, "Blended_EV").set_index(["Player", "Date"])
    dist = _infer_dist_from_columns(df)
    cv = stat_cv[league][market]
    if dist in ("NegBin", "ZINB") and cv == 1:
        raise ValueError(f"{league} {market}: cv == 1 makes the kernel price a bare Poisson")
    strategy = _decode_strategy_for_frame(df, league, market)
    params = pd.DataFrame(index=df.index, columns=SHAPE_FIELDS, dtype=float)
    if dist == "SkewNormal":
        loc, scale = _decode_sn_loc_scale(df, strategy)
        skew = df["SN_Alpha"].to_numpy(dtype=float)
        delta = skew / np.sqrt(1.0 + skew**2)
        params["mean"] = loc + scale * delta * np.sqrt(2.0 / np.pi)
        params["sigma"] = scale
        params["skew"] = skew
    elif dist in ("NegBin", "ZINB"):
        r = df["R"].to_numpy(dtype=float)
        p = df["NB_P"].to_numpy(dtype=float)
        params["mean"] = r * p / (1.0 - p)
        params["r"] = r
    elif dist == "DPO":
        phi = df["DP_PHI"].to_numpy(dtype=float)
        params["mean"] = _dp_mean(df["DP_MU"].to_numpy(dtype=float), phi)
        params["phi"] = phi
    else:
        raise ValueError(f"{league} {market}: the combo kernel cannot price a {dist!r} cell")
    if "Gate" in df.columns:
        params["gate"] = df["Gate"].to_numpy(dtype=float)
    midpit = _pred_midpit(df, dist, df["Result"].to_numpy(dtype=float), strategy=strategy)
    params["z"] = norm.ppf(np.clip(midpit, _PIT_CLIP, 1.0 - _PIT_CLIP))
    return ComponentCell(market, dist, cv, params)
