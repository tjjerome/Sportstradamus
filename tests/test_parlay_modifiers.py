"""Unit tests for the expected-vs-actual modifier reconciliation library."""

import json
import math

import pytest

from sportstradamus.helpers.config import apply_modifier_overrides
from sportstradamus.helpers.parlay_modifiers import (
    Leg,
    fold_overlay,
    reconcile,
    slip_pairs,
)

MODS = {"MLB": {"team": {"B.hits & B.runs": [0.9, 1.05]}, "opponent": {}}}
RAKE_3 = 0.962


def leg(game, team, key, higher=True, mult=1.5):
    return Leg(game, team, "MLB", key, higher, mult)


def expected_quote(legs, pairs, rake):
    product = math.prod(lg.mult for lg in legs)
    known = math.prod(p.value for p in pairs if p.value is not None)
    return (rake if rake is not None else 1.0) * product * known


def test_slip_pairs_classifies_relation_slot_and_lookup():
    legs = [
        leg("1", "SD", "B.hits", True, 1.5),
        leg("1", "SD", "B.runs", False, 1.6),
        leg("1", "TOR", "B.hits", True, 1.7),
        leg("2", "LAD", "B.runs", True, 1.8),
    ]
    pairs = slip_pairs(legs, MODS)
    assert [(p.relation, p.slot) for p in pairs] == [
        ("team", 1),
        ("opponent", 0),
        ("opponent", 1),
    ]
    assert pairs[0].disk_key == "B.hits & B.runs"
    assert pairs[0].value == 1.05
    assert pairs[0].group_key() == ("MLB", "team", "B.hits & B.runs", 1)
    assert [p.legs for p in pairs] == [(0, 1), (0, 2), (1, 2)]
    assert pairs[1].value is None


def test_solves_unknown_pair_from_actual_quote():
    legs = [
        leg("1", "SD", "TEAM.moneyline", True, 1.79),
        leg("1", "SD", "P.pitcher strikeouts", True, 1.62),
        leg("2", "LAD", "B.hits+runs+rbi", True, 2.33),
    ]
    pairs = slip_pairs(legs, MODS)
    expected = expected_quote(legs, pairs, RAKE_3)
    assert expected == pytest.approx(6.500, abs=0.001)
    kind, target, value = reconcile(legs, pairs, RAKE_3, 6.11, expected)
    assert kind == "solve"
    assert target == ("MLB", "team", "TEAM.moneyline & P.pitcher strikeouts", 0)
    assert value == pytest.approx(0.94, abs=0.001)


def test_all_cross_game_slip_calibrates_rake():
    legs = [
        leg("1", "SD", "B.hits", True, 1.5),
        leg("2", "LAD", "B.runs", True, 1.6),
        leg("3", "NYY", "B.hits", True, 1.7),
        leg("4", "BOS", "B.runs", True, 1.4),
    ]
    pairs = slip_pairs(legs, MODS)
    assert not pairs
    product = 1.5 * 1.6 * 1.7 * 1.4
    kind, target, value = reconcile(legs, pairs, None, product * 0.95, product)
    assert kind == "rake"
    assert target == 4
    assert value == pytest.approx(0.95, abs=0.001)


def test_known_pair_drift_updates_modifier():
    legs = [
        leg("1", "SD", "B.hits", True, 1.5),
        leg("1", "SD", "B.runs", True, 1.6),
        leg("2", "LAD", "B.hits", True, 1.7),
    ]
    pairs = slip_pairs(legs, MODS)
    expected = expected_quote(legs, pairs, RAKE_3)
    kind, target, value = reconcile(legs, pairs, RAKE_3, expected * 0.95, expected)
    assert kind == "drift"
    assert target == ("MLB", "team", "B.hits & B.runs", 0)
    assert value == pytest.approx(0.855, abs=0.001)


def test_matching_quote_teaches_nothing():
    legs = [
        leg("1", "SD", "TEAM.moneyline", True, 1.79),
        leg("2", "LAD", "B.hits", True, 1.62),
        leg("3", "NYY", "B.runs", True, 2.33),
    ]
    pairs = slip_pairs(legs, MODS)
    expected = expected_quote(legs, pairs, RAKE_3)
    assert reconcile(legs, pairs, RAKE_3, expected * 1.001, expected) == ("match", None, None)


def test_two_unknown_groups_is_ambiguous():
    legs = [
        leg("1", "SD", "TEAM.moneyline", True, 1.79),
        leg("1", "SD", "P.pitcher strikeouts", True, 1.62),
        leg("1", "SD", "B.home runs", True, 2.0),
    ]
    pairs = slip_pairs(legs, MODS)
    expected = expected_quote(legs, pairs, RAKE_3)
    kind, target, value = reconcile(legs, pairs, RAKE_3, 5.0, expected)
    assert kind == "ambiguous"
    assert len(target) == 3
    assert value is None


def test_unknown_rake_blocks_pair_solving():
    legs = [
        leg("1", "SD", "TEAM.moneyline", True, 1.79),
        leg("1", "SD", "P.pitcher strikeouts", True, 1.62),
    ]
    pairs = slip_pairs(legs, MODS)
    expected = expected_quote(legs, pairs, None)
    kind, target, _value = reconcile(legs, pairs, None, expected * 0.9, expected)
    assert kind == "uncalibrated"
    assert target == 2


@pytest.fixture()
def config_paths(tmp_path):
    combos = tmp_path / "banned_combos.json"
    combos.write_text(json.dumps({"Underdog": MODS}))
    rake = tmp_path / "parlay_rake.json"
    rake.write_text(json.dumps({"Underdog": {"3": RAKE_3}}))
    return combos, rake


NEW_PAIR = {"TEAM.moneyline & P.pitcher strikeouts": [0.94, 1.0]}


def test_fold_overlay_applies_and_keeps_overlay(config_paths, tmp_path):
    combos, rake = config_paths
    overlay = tmp_path / "modifier_overrides.json"
    payload = {
        "modifiers": {"Underdog": {"MLB": {"team": dict(NEW_PAIR)}}},
        "rake": {"Underdog": {"4": 0.95}},
    }
    overlay.write_text(json.dumps(payload))
    counts = fold_overlay(overlay, combos, rake)
    assert counts == {"pairs": 1, "rakes": 1, "pruned": 0}
    combos_d = json.loads(combos.read_text())
    assert combos_d["Underdog"]["MLB"]["team"]["TEAM.moneyline & P.pitcher strikeouts"] == [
        0.94,
        1.0,
    ]
    assert json.loads(rake.read_text())["Underdog"] == {"3": RAKE_3, "4": 0.95}
    assert json.loads(overlay.read_text()) == payload


def test_fold_overlay_prune_drops_synced_entries(config_paths, tmp_path):
    combos, rake = config_paths
    overlay = tmp_path / "modifier_overrides.json"
    overlay.write_text(
        json.dumps(
            {
                "modifiers": {
                    "Underdog": {"MLB": {"team": {"B.hits & B.runs": [0.9, 1.05], **NEW_PAIR}}}
                },
                "rake": {"Underdog": {"3": RAKE_3, "4": 0.95}},
            }
        )
    )
    counts = fold_overlay(overlay, combos, rake, prune=True)
    assert counts == {"pairs": 1, "rakes": 1, "pruned": 2}
    remaining = json.loads(overlay.read_text())
    assert remaining["modifiers"]["Underdog"]["MLB"]["team"] == NEW_PAIR
    assert remaining["rake"]["Underdog"] == {"4": 0.95}
    combos_d = json.loads(combos.read_text())
    assert combos_d["Underdog"]["MLB"]["team"]["B.hits & B.runs"] == [0.9, 1.05]
    assert "TEAM.moneyline & P.pitcher strikeouts" in combos_d["Underdog"]["MLB"]["team"]


def test_fold_overlay_prune_empties_fully_synced_overlay(config_paths, tmp_path):
    combos, rake = config_paths
    overlay = tmp_path / "modifier_overrides.json"
    overlay.write_text(
        json.dumps(
            {
                "modifiers": {"Underdog": {"MLB": {"team": {"B.hits & B.runs": [0.9, 1.05]}}}},
                "rake": {"Underdog": {"3": RAKE_3}},
            }
        )
    )
    counts = fold_overlay(overlay, combos, rake, prune=True)
    assert counts == {"pairs": 0, "rakes": 0, "pruned": 2}
    assert json.loads(overlay.read_text()) == {"modifiers": {}, "rake": {}}


def test_fold_overlay_missing_file_is_noop(config_paths, tmp_path):
    combos, rake = config_paths
    before = combos.read_text()
    counts = fold_overlay(tmp_path / "absent.json", combos, rake)
    assert counts == {"pairs": 0, "rakes": 0, "pruned": 0}
    assert combos.read_text() == before


def test_apply_modifier_overrides_merges_in_place():
    base = {"Underdog": {"MLB": {"team": {"B.hits & B.runs": [0.9, 1.05]}, "opponent": {}}}}
    apply_modifier_overrides(
        base,
        {
            "Underdog": {
                "MLB": {"team": {"B.hits & B.runs": [0.85, 1.05], "P.walks & B.hits": [0.7, 1.2]}},
                "NBA": {"opponent": {"G.points & G.assists": [0.8, 1.1]}},
            }
        },
    )
    assert base["Underdog"]["MLB"]["team"]["B.hits & B.runs"] == [0.85, 1.05]
    assert base["Underdog"]["MLB"]["team"]["P.walks & B.hits"] == [0.7, 1.2]
    assert base["Underdog"]["NBA"]["opponent"]["G.points & G.assists"] == [0.8, 1.1]
    assert base["Underdog"]["NBA"]["team"] == {}
