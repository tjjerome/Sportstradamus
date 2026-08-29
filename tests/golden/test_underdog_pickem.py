"""Tests for Phase 3 Step 6: Underdog Pick'em orchestrator."""

from __future__ import annotations

import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from sportstradamus.strategies._pickem_emit import emit_yaml, entries_to_frame
from sportstradamus.strategies.underdog_pickem import (
    PLATFORM_CONTEST_VARIANTS,
    PickemConfig,
    _filter_parlays,
    _validate_rivals_coverage,
    construct_entries,
    filter_legs,
)


def _offers(rows):
    cols = ["Player", "Team", "Market", "Bet", "Win Prob", "Market Prob"]
    return pd.DataFrame(rows, columns=cols)


def _canonical_leg(player, *, market, league, game, platform="Underdog"):
    return {
        "player": player,
        "team": "",
        "market": market,
        "stat": "PTS",
        "bet": "Over",
        "line": 4.5,
        "league": league,
        "game": game,
        "date": "2026-05-08",
        "platform": platform,
        "win_prob": 0.6,
        "boost": 1.0,
        "push_prob": 0.0,
        "kelly": 0.0,
    }


def _parlay(
    legs,
    *,
    bet_size=None,
    model_ev=2.5,
    payout=3.0,
    league="WNBA",
    game="A/B",
    platform="Underdog",
):
    """Parlay row shaped like ``prediction.parlay``'s beam-search output.

    ``legs``: player names, or ``(player, raw platform market)`` pairs; bare
    names get the platform's raw Points market so shrinkage resolution has a
    real ``stat_map`` key to map.
    """
    default_market = "Points" if platform == "Underdog" else "points"
    pairs = [(leg, default_market) if isinstance(leg, str) else leg for leg in legs]
    return {
        "Game": game,
        "Date": "2026-05-08",
        "League": league,
        "Platform": platform,
        "Model EV": model_ev,
        "Market EV": model_ev * 0.95,
        "Boost": payout,
        "Rec Bet": 1.0,
        "Bet Size": bet_size or len(legs),
        "legs": [
            _canonical_leg(p, market=m, league=league, game=game, platform=platform)
            for p, m in pairs
        ],
    }


# --- leg filter ---------------------------------------------------------------


def test_filter_legs_drops_below_model_edge():
    cfg = PickemConfig(min_model_edge=0.05)
    df = _offers(
        [
            ("Caitlin Clark", "IND", "PTS", "Over", 0.56, 0.55),
            ("A'ja Wilson", "LV", "PTS", "Over", 0.52, 0.51),
        ]
    )
    out = filter_legs(df, cfg)
    assert list(out["Player"]) == ["Caitlin Clark"]


def test_filter_legs_drops_below_sharp_edge():
    cfg = PickemConfig(min_model_edge=0.0, min_sharp_edge=0.10)
    df = _offers([("X", "T", "M", "Over", 0.60, 0.55)])
    assert filter_legs(df, cfg).empty


def test_filter_legs_disagreement():
    cfg = PickemConfig(min_model_edge=0.0, min_sharp_edge=0.0, disagreement_threshold=0.02)
    df = _offers(
        [
            ("Agree", "T", "M", "Over", 0.60, 0.59),
            ("Diverge", "T", "M", "Over", 0.70, 0.55),
        ]
    )
    out = filter_legs(df, cfg)
    assert list(out["Player"]) == ["Agree"]


# --- parlay candidate pool ----------------------------------------------------


def test_parlays_per_variant_routes_filtered_pool(monkeypatch):
    from sportstradamus.prediction import correlation
    from sportstradamus.strategies import underdog_pickem as up

    seen = {}

    def fake_find_correlation(offers, stats, platform, *, contest_variant):
        seen[contest_variant] = [o["Player"] for o in offers]
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(correlation, "find_correlation", fake_find_correlation)
    scored = pd.DataFrame(
        {
            "Player": ["Edge", "NoEdge"],
            "Win Prob": [0.56, 0.51],
            "Market Prob": [0.54, 0.505],
        }
    )
    cfg = PickemConfig(contest_variants=("power", "flex"))
    out = up._parlays_per_variant(scored, {}, cfg, "Underdog")
    assert seen == {"power": ["Edge"], "flex": ["Edge"]}
    assert set(out) == {"power", "flex"}
    assert all(df.empty for df in out.values())


def test_parlays_per_variant_empty_pool_skips_search(monkeypatch):
    from sportstradamus.prediction import correlation
    from sportstradamus.strategies import underdog_pickem as up

    def boom(*args, **kwargs):
        raise AssertionError("find_correlation must not run on an empty pool")

    monkeypatch.setattr(correlation, "find_correlation", boom)
    cfg = PickemConfig(contest_variants=("power", "flex", "rivals"))
    no_edge = pd.DataFrame({"Player": ["NoEdge"], "Win Prob": [0.50], "Market Prob": [0.50]})
    for scored in (no_edge, pd.DataFrame()):
        out = up._parlays_per_variant(scored, {}, cfg, "Underdog")
        assert set(out) == {"power", "flex", "rivals"}
        assert all(df.empty for df in out.values())


# --- parlay-level filter ------------------------------------------------------


def test_filter_parlays_size_and_ev():
    cfg = PickemConfig(min_ev=0.10, entry_sizes=(3,))
    df = pd.DataFrame(
        [
            _parlay(["L1", "L2", "L3"], model_ev=1.50),
            _parlay(["L1", "L2"], model_ev=2.0),  # wrong size
            _parlay(["L1", "L2", "L3"], model_ev=1.05),  # below min_ev
        ]
    )
    out = _filter_parlays(df, "power", cfg.entry_sizes, cfg)
    assert len(out) == 1
    assert out["Bet Size"].iloc[0] == 3


def test_filter_parlays_rivals_overrides_entry_sizes():
    cfg = PickemConfig(min_ev=0.0, entry_sizes=(5, 6))
    df = pd.DataFrame(
        [
            _parlay(["A vs. B", "C vs. D"], model_ev=1.5),
            _parlay(["A vs. B", "C vs. D", "E vs. F", "G vs. H", "I vs. J"], model_ev=2.0),
        ]
    )
    out = _filter_parlays(df, "rivals", cfg.entry_sizes, cfg)
    assert list(out["Bet Size"]) == [2]


# --- rivals coverage ---------------------------------------------------------


def test_rivals_one_sided_dropped(caplog):
    import logging

    target = logging.getLogger("sportstradamus.cli.pickem-build")
    target.addHandler(caplog.handler)
    target.setLevel(logging.WARNING)
    try:
        offers = _offers([("Caitlin Clark", "IND", "PTS", "Over", 0.55, 0.54)])
        df = pd.DataFrame(
            [_parlay(["Caitlin Clark vs. Sabrina Ionescu", "A'ja Wilson vs. Breanna Stewart"])]
        )
        out = _validate_rivals_coverage(df, offers)
    finally:
        target.removeHandler(caplog.handler)
    assert out.empty
    assert any("rivals candidate dropped" in r.getMessage() for r in caplog.records)


def test_rivals_both_sides_kept():
    offers = _offers(
        [
            ("Caitlin Clark", "IND", "PTS", "Over", 0.55, 0.54),
            ("Sabrina Ionescu", "NY", "PTS", "Under", 0.55, 0.54),
        ]
    )
    df = pd.DataFrame([_parlay(["Caitlin Clark vs. Sabrina Ionescu"], model_ev=1.5)])
    out = _validate_rivals_coverage(df, offers)
    assert len(out) == 1


# --- end-to-end with injection -----------------------------------------------


@pytest.fixture
def fake_market_calibration(monkeypatch):
    monkeypatch.setattr(
        "sportstradamus.strategies.underdog_pickem.resolve_market_shrinkage",
        lambda league, market: (0.5, "training"),
    )


def test_construct_entries_all_three_variants(fake_market_calibration):
    cfg = PickemConfig(
        min_model_edge=0.0,
        min_sharp_edge=0.0,
        disagreement_threshold=1.0,
        min_ev=0.0,
        entry_sizes=(3,),
        contest_variants=("power", "flex", "rivals"),
        top_k=10,
        max_overlap=2,
    )
    offers = _offers(
        [
            ("Clark", "IND", "PTS", "Over", 0.55, 0.54),
            ("Wilson", "LV", "PTS", "Over", 0.55, 0.54),
            ("Ionescu", "NY", "PTS", "Over", 0.55, 0.54),
            ("Stewart", "NY", "PTS", "Over", 0.55, 0.54),
        ]
    )
    parlay_dfs = {
        "power": pd.DataFrame([_parlay(["P1", "P2", "P3"], model_ev=1.4)]),
        "flex": pd.DataFrame([_parlay(["F1", "F2", "F3"], model_ev=1.3)]),
        "rivals": pd.DataFrame(
            [_parlay(["Clark vs. Ionescu", "Wilson vs. Stewart"], model_ev=1.5)]
        ),
    }
    out = construct_entries(
        datetime.date(2026, 5, 8),
        Decimal("500"),
        cfg,
        parlay_dfs=parlay_dfs,
        offers_df=offers,
    )
    variants = {e.contest_variant for e in out}
    assert variants == {"power", "flex", "rivals"}
    assert all(leg for e in out for leg in e.legs)


def test_construct_entries_yaml_roundtrip(tmp_path: Path, fake_market_calibration):
    cfg = PickemConfig(
        min_model_edge=0.0,
        min_sharp_edge=0.0,
        disagreement_threshold=1.0,
        min_ev=0.0,
        entry_sizes=(3,),
        contest_variants=("power",),
    )
    offers = _offers([("Clark", "IND", "PTS", "Over", 0.6, 0.58)])
    parlay_dfs = {"power": pd.DataFrame([_parlay(["L1", "L2", "L3"], model_ev=1.5)])}
    entries = construct_entries(
        datetime.date(2026, 5, 8),
        Decimal("500"),
        cfg,
        parlay_dfs=parlay_dfs,
        offers_df=offers,
    )
    assert len(entries) == 1

    yaml = pytest.importorskip("yaml")
    out_path = tmp_path / "rec.yaml"
    emit_yaml(entries, datetime.date(2026, 5, 8), Decimal("500"), cfg, out_path)
    payload = yaml.safe_load(out_path.read_text())
    assert payload["date"] == "2026-05-08"
    assert payload["bankroll"] == "500"
    assert payload["entries"][0]["contest_variant"] == "power"
    assert payload["entries"][0]["entry_size"] == 3
    # Display legs are rendered from the canonical ``legs`` records via
    # leg_label — non-empty, one per leg record.
    assert payload["entries"][0]["legs"] == [
        "L1 Over 4.5 Points",
        "L2 Over 4.5 Points",
        "L3 Over 4.5 Points",
    ]
    assert "joint_prob" in payload["entries"][0]
    assert "payout_multiplier" in payload["entries"][0]


def test_dedupe_max_overlap(fake_market_calibration):
    cfg = PickemConfig(
        min_model_edge=0.0,
        min_sharp_edge=0.0,
        disagreement_threshold=1.0,
        min_ev=0.0,
        entry_sizes=(3,),
        contest_variants=("power",),
        top_k=10,
        max_overlap=2,
    )
    parlay_dfs = {
        "power": pd.DataFrame(
            [
                _parlay(["L1", "L2", "L3"], model_ev=1.5),
                _parlay(["L1", "L2", "L3"], model_ev=1.5),  # exact dup
            ]
        )
    }
    out = construct_entries(
        datetime.date(2026, 5, 8),
        Decimal("500"),
        cfg,
        parlay_dfs=parlay_dfs,
    )
    assert len(out) == 1


# --- shrinkage resolution -----------------------------------------------------


def test_parlay_shrinkage_min_over_canonical_leg_markets(monkeypatch):
    from sportstradamus.strategies import underdog_pickem as up

    # Raw Underdog Market names map via stat_map: Rebounds->REB, Steals->STL.
    # Indexing (not .get) pins that only canonical cell keys reach resolution.
    shrink = {"REB": (0.80, "training"), "STL": (0.20, "training")}
    monkeypatch.setattr(up, "resolve_market_shrinkage", lambda league, market: shrink[market])
    row = pd.Series(_parlay([("A. Player", "Rebounds"), ("B. Player", "Steals")]))
    shrinkage, source = up._parlay_shrinkage(row, "WNBA", "Underdog")
    assert shrinkage == 0.20  # min(REB=0.80, STL=0.20) — least-calibrated leg
    assert source == "training"


def test_parlay_shrinkage_unmapped_markets_fall_back(monkeypatch):
    from sportstradamus.strategies import underdog_pickem as up

    monkeypatch.setattr(
        up,
        "resolve_market_shrinkage",
        lambda league, market: pytest.fail(f"unmapped market resolved: {market}"),
    )
    row = pd.Series(_parlay([("A vs. B", "H2H Points")]))
    assert up._parlay_shrinkage(row, "WNBA", "Underdog") == (1.0, "fallback")


def test_parlay_shrinkage_sleeper_uses_sleeper_stat_map(monkeypatch):
    from sportstradamus.strategies import underdog_pickem as up

    calls = []
    monkeypatch.setattr(
        up,
        "resolve_market_shrinkage",
        lambda league, market: calls.append(market) or (0.5, "training"),
    )
    # Real Sleeper stat_map raw key (disjoint namespace from Underdog's
    # "Rebounds"): rebounds->REB.
    row = pd.Series(_parlay([("A. Player", "rebounds")], platform="Sleeper"))
    assert up._parlay_shrinkage(row, "WNBA", "Sleeper") == (0.5, "training")
    assert calls == ["REB"]
    # The same raw name under the Underdog map resolves to nothing -> fallback.
    assert up._parlay_shrinkage(row, "WNBA", "Underdog") == (1.0, "fallback")


# --- Sleeper platform threading ------------------------------------------------


def test_construct_entries_sleeper_excludes_rivals(fake_market_calibration):
    cfg = PickemConfig(
        min_model_edge=0.0,
        min_sharp_edge=0.0,
        disagreement_threshold=1.0,
        min_ev=0.0,
        entry_sizes=(3,),
        contest_variants=PLATFORM_CONTEST_VARIANTS["Sleeper"],
        top_k=10,
        max_overlap=2,
    )
    offers = _offers(
        [
            ("Clark", "IND", "PTS", "Over", 0.55, 0.54),
            ("Wilson", "LV", "PTS", "Over", 0.55, 0.54),
        ]
    )
    parlay_dfs = {
        "power": pd.DataFrame([_parlay(["P1", "P2", "P3"], model_ev=1.4, platform="Sleeper")]),
        "flex": pd.DataFrame([_parlay(["F1", "F2", "F3"], model_ev=1.3, platform="Sleeper")]),
        # If a rivals frame ever leaked in it must never surface in the output
        # because "rivals" isn't in Sleeper's contest_variants.
        "rivals": pd.DataFrame([_parlay(["Clark vs. Wilson"], model_ev=1.5, platform="Sleeper")]),
    }
    out = construct_entries(
        datetime.date(2026, 5, 8),
        Decimal("500"),
        cfg,
        parlay_dfs=parlay_dfs,
        offers_df=offers,
        platform="Sleeper",
    )
    variants = {e.contest_variant for e in out}
    assert variants == {"power", "flex"}
    assert "rivals" not in variants
    assert all(e.platform == "Sleeper" for e in out)


def test_construct_entries_sleeper_push_case_sizes(fake_market_calibration):
    """Size-2 Sleeper Max (full-refund push) and size-6 Sleeper Flex.

    The PickemConfig default entry_sizes=(3, 5) never reaches either edge;
    this exercises both explicitly per the stage acceptance criteria.
    """
    cfg = PickemConfig(
        min_model_edge=0.0,
        min_sharp_edge=0.0,
        disagreement_threshold=1.0,
        min_ev=0.0,
        entry_sizes=(2, 3, 4, 5, 6),
        contest_variants=PLATFORM_CONTEST_VARIANTS["Sleeper"],
        top_k=10,
        max_overlap=5,
    )
    parlay_dfs = {
        "power": pd.DataFrame([_parlay(["P1", "P2"], model_ev=1.5, platform="Sleeper")]),
        "flex": pd.DataFrame(
            [
                _parlay(
                    ["F1", "F2", "F3", "F4", "F5", "F6"],
                    model_ev=1.5,
                    platform="Sleeper",
                )
            ]
        ),
    }
    out = construct_entries(
        datetime.date(2026, 5, 8),
        Decimal("500"),
        cfg,
        parlay_dfs=parlay_dfs,
        platform="Sleeper",
    )
    sizes_by_variant = {e.contest_variant: e.entry_size for e in out}
    assert sizes_by_variant.get("power") == 2
    assert sizes_by_variant.get("flex") == 6
    assert all(e.platform == "Sleeper" for e in out)


@pytest.mark.parametrize("platform", ["Underdog", "Sleeper"])
def test_recommended_entry_platform_roundtrips_yaml_and_frame(
    tmp_path: Path, fake_market_calibration, platform
):
    cfg = PickemConfig(
        min_model_edge=0.0,
        min_sharp_edge=0.0,
        disagreement_threshold=1.0,
        min_ev=0.0,
        entry_sizes=(3,),
        contest_variants=("power",),
    )
    offers = _offers([("Clark", "IND", "PTS", "Over", 0.6, 0.58)])
    parlay_dfs = {
        "power": pd.DataFrame([_parlay(["L1", "L2", "L3"], model_ev=1.5, platform=platform)])
    }
    entries = construct_entries(
        datetime.date(2026, 5, 8),
        Decimal("500"),
        cfg,
        parlay_dfs=parlay_dfs,
        offers_df=offers,
        platform=platform,
    )
    assert len(entries) == 1
    assert entries[0].platform == platform

    yaml = pytest.importorskip("yaml")
    out_path = tmp_path / "rec.yaml"
    emit_yaml(entries, datetime.date(2026, 5, 8), Decimal("500"), cfg, out_path)
    payload = yaml.safe_load(out_path.read_text())
    assert payload["entries"][0]["platform"] == platform

    frame = entries_to_frame(entries)
    assert frame["platform"].iloc[0] == platform
    assert all(frame["legs"].iloc[0])
