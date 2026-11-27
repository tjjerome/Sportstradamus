"""Tests for Phase 3 Step 6: Underdog Pick'em orchestrator."""

from __future__ import annotations

import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.strategies._pickem_emit import emit_yaml, rank_and_dedupe
from sportstradamus.strategies.kelly import kelly
from sportstradamus.strategies.underdog_pickem import (
    PickemConfig,
    RecommendedEntry,
    _filter_parlays,
    construct_entries,
)


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
    ev=0.9,
    model_ev=2.5,
    payout=3.0,
    league="WNBA",
    game="A/B",
    platform="Underdog",
    pair_modifiers=None,
):
    """Parlay row shaped like ``prediction.parlay._parlay_row``.

    ``legs``: player names, or ``(player, raw platform market)`` pairs; bare
    names get the platform's raw Points market. ``ev`` is the fitted value of a
    unit staked, ``model_ev`` the model's claim beside it. ``pair_modifiers``: one
    per leg pair, every pair unmodified when omitted.
    """
    default_market = "Points" if platform == "Underdog" else "points"
    pairs = [(leg, default_market) if isinstance(leg, str) else leg for leg in legs]
    return {
        "Game": game,
        "Date": "2026-05-08",
        "League": league,
        "Platform": platform,
        "EV": ev,
        "Model EV": model_ev,
        "Market EV": model_ev * 0.95,
        "Boost": payout,
        "Boost Pairs": pair_modifiers or (1.0,) * (len(legs) * (len(legs) - 1) // 2),
        "Rec Bet": 0.0,
        "Bet Size": bet_size or len(legs),
        "legs": [
            _canonical_leg(p, market=m, league=league, game=game, platform=platform)
            for p, m in pairs
        ],
    }


# --- parlay candidate pool ----------------------------------------------------


def test_every_scored_record_reaches_each_variants_search(monkeypatch):
    """No leg gate: a leg the old gate dropped for its read, its book or their gap still
    seeds entries, as on the board, and the rank chooses among them."""
    from sportstradamus.prediction import correlation
    from sportstradamus.strategies import underdog_pickem as up

    seen = {}

    def fake_find_correlation(offers, stats, platform, *, contest_variant):
        seen[contest_variant] = [o["Player"] for o in offers]
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(correlation, "find_correlation", fake_find_correlation)
    scored = pd.DataFrame(
        {
            "Player": ["Edge", "Weak Read", "No Book", "Far From Book"],
            "Win Prob": [0.56, 0.51, 0.58, 0.70],
            "Market Prob": [0.54, 0.505, float("nan"), 0.55],
        }
    )
    cfg = PickemConfig(contest_variants=("power", "flex"))
    out = up._parlays_per_variant(scored, {}, cfg, "Underdog")
    everyone = ["Edge", "Weak Read", "No Book", "Far From Book"]
    assert seen == {"power": everyone, "flex": everyone}
    assert set(out) == {"power", "flex"}


def test_an_empty_board_gets_an_empty_parlay_frame_per_variant():
    from sportstradamus.strategies import underdog_pickem as up

    cfg = PickemConfig(contest_variants=("power", "flex"))
    out = up._parlays_per_variant(pd.DataFrame(), {}, cfg, "Underdog")
    assert set(out) == {"power", "flex"}
    assert all(df.empty for df in out.values())
    assert construct_entries(cfg, parlay_dfs=out) == []


# --- parlay-level filter ------------------------------------------------------


def test_filter_parlays_keeps_every_entry_of_a_kept_size():
    """No value floor: the rank decides. The last entry sat under the old 5 % floor on
    the model's claim and is kept."""
    cfg = PickemConfig(entry_sizes=(3,))
    df = pd.DataFrame(
        [
            _parlay(["L1", "L2", "L3"], ev=0.95, model_ev=1.50),
            _parlay(["L1", "L2"], ev=0.95, model_ev=2.0),  # wrong size
            _parlay(["L1", "L2", "L3"], ev=0.80, model_ev=1.03),
        ]
    )
    out = _filter_parlays(df, cfg)
    assert out["Bet Size"].tolist() == [3, 3]


# --- end-to-end with injection -----------------------------------------------


def test_entry_hit_chance_and_value_come_from_the_rows_fitted_ev():
    """``EV`` is the fitted value of a unit staked; ``Model EV``, the model's claim beside
    it, prices nothing."""
    cfg = PickemConfig(entry_sizes=(3,), contest_variants=("power",))
    parlay_dfs = {
        "power": pd.DataFrame([_parlay(["L1", "L2", "L3"], ev=0.90, model_ev=1.50, payout=6.0)])
    }

    (entry,) = construct_entries(cfg, parlay_dfs=parlay_dfs)

    assert entry.payout_multiplier == 6.0
    assert entry.joint_prob == pytest.approx(0.15)
    assert entry.ev == pytest.approx(-0.10)


def test_a_board_with_every_entry_under_one_still_yields_entries_and_stakes_none():
    """On the live board no entry's fitted value clears 1, and the model's claim here does
    not either. Every entry is still recommended, by rank, and none is staked until the
    entries' money record supports one."""
    parlay_dfs = {
        "power": pd.DataFrame(
            [
                _parlay(["P1", "P2"], ev=0.93, model_ev=0.93),
                _parlay(["Q1", "Q2", "Q3"], ev=0.88, model_ev=0.88),
            ]
        ),
        "flex": pd.DataFrame([_parlay(["F1", "F2", "F3", "F4", "F5"], ev=0.79, model_ev=0.79)]),
    }

    out = construct_entries(PickemConfig(entry_sizes=(2, 3, 5)), parlay_dfs=parlay_dfs)

    assert [round(e.ev, 2) for e in out] == [-0.07, -0.12, -0.21]
    assert all(e.recommended_stake == Decimal("0") for e in out)


def test_construct_entries_both_variants():
    cfg = PickemConfig(
        entry_sizes=(3,),
        contest_variants=("power", "flex"),
        top_k=10,
        max_overlap=2,
    )
    parlay_dfs = {
        "power": pd.DataFrame([_parlay(["P1", "P2", "P3"], model_ev=1.4)]),
        "flex": pd.DataFrame([_parlay(["F1", "F2", "F3"], model_ev=1.3)]),
    }
    out = construct_entries(
        cfg,
        parlay_dfs=parlay_dfs,
    )
    variants = {e.contest_variant for e in out}
    assert variants == {"power", "flex"}
    assert all(leg for e in out for leg in e.legs)


def test_construct_entries_yaml_roundtrip(tmp_path: Path):
    cfg = PickemConfig(
        entry_sizes=(3,),
        contest_variants=("power",),
    )
    parlay_dfs = {"power": pd.DataFrame([_parlay(["L1", "L2", "L3"], model_ev=1.5)])}
    entries = construct_entries(
        cfg,
        parlay_dfs=parlay_dfs,
    )
    assert len(entries) == 1

    yaml = pytest.importorskip("yaml")
    out_path = tmp_path / "rec.yaml"
    emit_yaml(entries, datetime.date(2026, 5, 8), cfg, out_path)
    payload = yaml.safe_load(out_path.read_text())
    assert payload["date"] == "2026-05-08"
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


def test_dedupe_max_overlap():
    cfg = PickemConfig(
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
        cfg,
        parlay_dfs=parlay_dfs,
    )
    assert len(out) == 1


def _ranked_entry(entry_id: str, *, ev: float, joint_prob: float) -> RecommendedEntry:
    return RecommendedEntry(
        id=entry_id,
        contest_variant="power",
        entry_size=2,
        legs=(f"{entry_id} A Over 4.5 Points", f"{entry_id} B Over 4.5 Points"),
        joint_prob=joint_prob,
        payout_multiplier=3.0,
        ev=ev,
        recommended_stake=Decimal("0"),
    )


def test_entries_rank_by_their_fitted_value_highest_first():
    """With every entry under 1, EV per dollar times the hit chance ranked the long shot
    above the likelier entry worth more; the value itself is the rank."""
    entries = [
        _ranked_entry("long_shot", ev=-0.20, joint_prob=0.10),
        _ranked_entry("likely", ev=-0.05, joint_prob=0.50),
        _ranked_entry("best", ev=-0.02, joint_prob=0.05),
    ]

    ranked = rank_and_dedupe(entries, PickemConfig(top_k=10))

    assert [e.id for e in ranked] == ["best", "likely", "long_shot"]


def test_entry_carries_the_product_of_its_pair_modifiers():
    """The search prices an entry on its legs' multipliers times the product of its
    same-game pair modifiers; the ledger settles a winner on that same product."""
    cfg = PickemConfig(entry_sizes=(3,), contest_variants=("power",))
    parlay_dfs = {
        "power": pd.DataFrame(
            [_parlay(["L1", "L2", "L3"], model_ev=1.5, pair_modifiers=(1.0, 0.9, 0.8))]
        )
    }

    (entry,) = construct_entries(cfg, parlay_dfs=parlay_dfs)

    assert entry.pair_modifier == pytest.approx(0.72)


# --- Sleeper platform threading ------------------------------------------------


def test_construct_entries_sleeper_platform():
    cfg = PickemConfig(
        entry_sizes=(3,),
        contest_variants=("power", "flex"),
        top_k=10,
        max_overlap=2,
    )
    parlay_dfs = {
        "power": pd.DataFrame([_parlay(["P1", "P2", "P3"], model_ev=1.4, platform="Sleeper")]),
        "flex": pd.DataFrame([_parlay(["F1", "F2", "F3"], model_ev=1.3, platform="Sleeper")]),
    }
    out = construct_entries(
        cfg,
        parlay_dfs=parlay_dfs,
        platform="Sleeper",
    )
    variants = {e.contest_variant for e in out}
    assert variants == {"power", "flex"}
    assert all(e.platform == "Sleeper" for e in out)


def test_construct_entries_sleeper_push_case_sizes():
    """Size-2 Sleeper Max (full-refund push) and size-6 Sleeper Flex.

    The PickemConfig default entry_sizes=(3, 5) never reaches either edge;
    this exercises both explicitly per the stage acceptance criteria.
    """
    cfg = PickemConfig(
        entry_sizes=(2, 3, 4, 5, 6),
        contest_variants=("power", "flex"),
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
        cfg,
        parlay_dfs=parlay_dfs,
        platform="Sleeper",
    )
    sizes_by_variant = {e.contest_variant: e.entry_size for e in out}
    assert sizes_by_variant.get("power") == 2
    assert sizes_by_variant.get("flex") == 6
    assert all(e.platform == "Sleeper" for e in out)


@pytest.mark.parametrize("platform", ["Underdog", "Sleeper"])
def test_recommended_entry_platform_roundtrips_yaml(tmp_path: Path, platform):
    cfg = PickemConfig(
        entry_sizes=(3,),
        contest_variants=("power",),
    )
    parlay_dfs = {
        "power": pd.DataFrame([_parlay(["L1", "L2", "L3"], model_ev=1.5, platform=platform)])
    }
    entries = construct_entries(
        cfg,
        parlay_dfs=parlay_dfs,
        platform=platform,
    )
    assert len(entries) == 1
    assert entries[0].platform == platform

    yaml = pytest.importorskip("yaml")
    out_path = tmp_path / "rec.yaml"
    emit_yaml(entries, datetime.date(2026, 5, 8), cfg, out_path)
    payload = yaml.safe_load(out_path.read_text())
    assert payload["entries"][0]["platform"] == platform
    assert all(payload["entries"][0]["legs"])


def test_bet_kelly_sizes_the_yaml_bet_pickem_emits_on_each_entrys_fitted_chance(tmp_path: Path):
    """An entry valued above 1 gets a Kelly stake (here the 0.5 % cap of a 1,000 bankroll);
    one valued under 1 gets none, though the model claims 2.5 for both."""
    cfg = PickemConfig(entry_sizes=(3,), contest_variants=("power",))
    parlays = pd.DataFrame(
        [_parlay(["W1", "W2", "W3"], ev=1.2), _parlay(["L1", "L2", "L3"], ev=0.9)]
    )
    out_path = tmp_path / "rec.yaml"
    emit_yaml(
        construct_entries(cfg, parlay_dfs={"power": parlays}),
        datetime.date(2026, 5, 8),
        cfg,
        out_path,
    )

    result = CliRunner().invoke(kelly, ["--bankroll", "1000", "--from", str(out_path)])

    assert result.exit_code == 0, result.output
    rows = [line.split("|") for line in result.output.splitlines()[2:]]
    stake_by_ev = {float(cells[6]): float(cells[7]) for cells in rows}
    assert stake_by_ev == {0.2: 5.0, -0.1: 0.0}
