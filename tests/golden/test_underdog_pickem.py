"""Tests for Phase 3 Step 6: Underdog Pick'em orchestrator."""

from __future__ import annotations

import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from sportstradamus.strategies._pickem_emit import emit_yaml
from sportstradamus.strategies.underdog_pickem import (
    PickemConfig,
    _filter_legs,
    _filter_parlays,
    _validate_rivals_coverage,
    construct_entries,
)


def _offers(rows):
    cols = ["Player", "Team", "Market", "Bet", "Model P", "Books P"]
    return pd.DataFrame(rows, columns=cols)


def _parlay(legs, *, bet_size=None, model_ev=2.5, payout=3.0, league="WNBA", game="A/B"):
    bet_size = bet_size or len(legs)
    row = {
        "Game": game,
        "Date": "2026-05-08",
        "League": league,
        "Platform": "Underdog",
        "Model EV": model_ev,
        "Books EV": model_ev * 0.95,
        "Boost": payout,
        "Rec Bet": 1.0,
        "Bet Size": bet_size,
    }
    for i, leg in enumerate(legs, 1):
        row[f"Leg {i}"] = leg
    for i in range(len(legs) + 1, 7):
        row[f"Leg {i}"] = ""
    return row


# --- leg filter ---------------------------------------------------------------


def test_filter_legs_drops_below_model_edge():
    cfg = PickemConfig(min_model_edge=0.05)
    df = _offers(
        [
            ("Caitlin Clark", "IND", "PTS", "Over", 0.56, 0.55),
            ("A'ja Wilson", "LV", "PTS", "Over", 0.52, 0.51),
        ]
    )
    out = _filter_legs(df, cfg)
    assert list(out["Player"]) == ["Caitlin Clark"]


def test_filter_legs_drops_below_sharp_edge():
    cfg = PickemConfig(min_model_edge=0.0, min_sharp_edge=0.10)
    df = _offers([("X", "T", "M", "Over", 0.60, 0.55)])
    assert _filter_legs(df, cfg).empty


def test_filter_legs_disagreement():
    cfg = PickemConfig(min_model_edge=0.0, min_sharp_edge=0.0, disagreement_threshold=0.02)
    df = _offers(
        [
            ("Agree", "T", "M", "Over", 0.60, 0.59),
            ("Diverge", "T", "M", "Over", 0.70, 0.55),
        ]
    )
    out = _filter_legs(df, cfg)
    assert list(out["Player"]) == ["Agree"]


# --- parlay-level filter ------------------------------------------------------


def test_filter_parlays_size_and_ev():
    cfg = PickemConfig(min_ev=0.10, entry_sizes=(3,))
    df = pd.DataFrame(
        [
            _parlay(["L1", "L2", "L3"], model_ev=1.50),
            _parlay(["L1", "L2"], model_ev=2.0, bet_size=2),  # wrong size
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
            _parlay(["A vs. B"], bet_size=2, model_ev=1.5),
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
            [_parlay(["Caitlin Clark vs. Sabrina Ionescu"], bet_size=2, model_ev=1.5)]
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
    df = pd.DataFrame(
        [_parlay(["Caitlin Clark vs. Sabrina Ionescu"], bet_size=2, model_ev=1.5)]
    )
    out = _validate_rivals_coverage(df, offers)
    assert len(out) == 1


# --- end-to-end with injection -----------------------------------------------


@pytest.fixture
def fake_market_calibration(monkeypatch):
    monkeypatch.setattr(
        "sportstradamus.strategies.underdog_pickem._resolve_market_shrinkage",
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
            [_parlay(["Clark vs. Ionescu"], bet_size=2, model_ev=1.5)]
        ),
    }
    out = construct_entries(
        datetime.date(2026, 5, 8), Decimal("500"), cfg,
        parlay_dfs=parlay_dfs, offers_df=offers,
    )
    variants = {e.contest_variant for e in out}
    assert variants == {"power", "flex", "rivals"}


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
        datetime.date(2026, 5, 8), Decimal("500"), cfg,
        parlay_dfs=parlay_dfs, offers_df=offers,
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
    assert payload["entries"][0]["legs"] == ["L1", "L2", "L3"]
    assert "joint_prob" in payload["entries"][0]
    assert "payout_multiplier" in payload["entries"][0]


def test_dedupe_max_overlap(fake_market_calibration):
    cfg = PickemConfig(
        min_model_edge=0.0, min_sharp_edge=0.0, disagreement_threshold=1.0,
        min_ev=0.0, entry_sizes=(3,), contest_variants=("power",),
        top_k=10, max_overlap=2,
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
        datetime.date(2026, 5, 8), Decimal("500"), cfg, parlay_dfs=parlay_dfs,
    )
    assert len(out) == 1
