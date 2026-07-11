"""Unit tests for the expected-vs-actual quote reconciler."""

import json

import pytest
from click.testing import CliRunner

from sportstradamus.scripts.calibrate_parlay_modifiers import (
    Leg,
    calibrate_parlay_modifiers,
    parse_slip,
    slip_pairs,
)

PADRES_SLIP = (
    '1:SD:TEAM.moneyline:h:1.79 "1:SD:P.pitcher strikeouts:h:1.62" 2:LAD:B.hits+runs+rbi:h:2.33'
)


@pytest.fixture()
def config_paths(tmp_path):
    combos = tmp_path / "banned_combos.json"
    combos.write_text(
        json.dumps(
            {
                "Underdog": {
                    "MLB": {
                        "team": {"B.hits & B.runs": [0.9, 1.05]},
                        "opponent": {},
                    }
                }
            }
        )
    )
    rake = tmp_path / "parlay_rake.json"
    rake.write_text(json.dumps({"Underdog": {"3": 0.962}}))
    return combos, rake


def run_cli(config_paths, cli_input):
    combos, rake = config_paths
    result = CliRunner().invoke(
        calibrate_parlay_modifiers,
        ["--league", "MLB", "--combos-path", str(combos), "--rake-path", str(rake)],
        input=cli_input,
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_parse_slip_rejects_bad_specs():
    with pytest.raises(ValueError, match="needs game:team"):
        parse_slip("1:SD:B.hits:h")
    with pytest.raises(ValueError, match=r"POS\.market"):
        parse_slip("1:SD:hits:h:1.5 1:SD:B.runs:h:1.5")
    with pytest.raises(ValueError, match="direction"):
        parse_slip("1:SD:B.hits:x:1.5 1:SD:B.runs:h:1.5")
    with pytest.raises(ValueError, match="at least 2"):
        parse_slip("1:SD:B.hits:h:1.5")


def test_slip_pairs_classifies_relation_slot_and_lookup():
    legs = parse_slip("1:SD:B.hits:h:1.5 1:SD:B.runs:l:1.6 1:TOR:B.hits:h:1.7 2:LAD:B.runs:h:1.8")
    mods = {"team": {"B.hits & B.runs": [0.9, 1.05]}, "opponent": {}}
    pairs = slip_pairs(legs, mods)
    assert [(p.relation, p.slot) for p in pairs] == [
        ("team", 1),
        ("opponent", 0),
        ("opponent", 1),
    ]
    assert pairs[0].disk_key == "B.hits & B.runs"
    assert pairs[0].value == 1.05
    assert pairs[1].value is None


def test_solves_unknown_pair_from_actual_quote(config_paths):
    output = run_cli(config_paths, f"{PADRES_SLIP}\n6.11\ny\n\n")
    assert "UNKNOWN" in output
    assert "expected: 6.500x" in output
    combos = json.loads(config_paths[0].read_text())
    written = combos["Underdog"]["MLB"]["team"]["TEAM.moneyline & P.pitcher strikeouts"]
    assert written == [0.94, 1.0]


def test_all_cross_game_slip_calibrates_rake(config_paths):
    slip = "1:SD:B.hits:h:1.5 2:LAD:B.runs:h:1.6 3:NYY:B.hits:h:1.7 4:BOS:B.runs:h:1.4"
    product = 1.5 * 1.6 * 1.7 * 1.4
    output = run_cli(config_paths, f"{slip}\n{product * 0.95:.4f}\ny\n\n")
    assert "write rake[4]" in output
    rake = json.loads(config_paths[1].read_text())
    assert rake["Underdog"]["4"] == pytest.approx(0.95, abs=0.001)


def test_known_pair_drift_offers_update(config_paths):
    slip = "1:SD:B.hits:h:1.5 1:SD:B.runs:h:1.6 2:LAD:B.hits:h:1.7"
    expected = 0.962 * 1.5 * 1.6 * 1.7 * 0.9
    output = run_cli(config_paths, f"{slip}\n{expected * 0.95:.4f}\ny\n\n")
    assert "drift-update" in output
    combos = json.loads(config_paths[0].read_text())
    assert combos["Underdog"]["MLB"]["team"]["B.hits & B.runs"][0] == pytest.approx(
        0.855, abs=0.001
    )


def test_matching_quote_writes_nothing(config_paths):
    output = run_cli(config_paths, f"{PADRES_SLIP}\n6.50\n\n")
    assert "nothing to learn" in output
    combos = json.loads(config_paths[0].read_text())
    assert "TEAM.moneyline & P.pitcher strikeouts" not in combos["Underdog"]["MLB"]["team"]


def test_two_unknown_groups_is_ambiguous(config_paths):
    slip = '1:SD:TEAM.moneyline:h:1.79 "1:SD:P.pitcher strikeouts:h:1.62" "1:SD:B.home runs:h:2.0"'
    output = run_cli(config_paths, f"{slip}\n5.0\n\n")
    assert "cannot attribute" in output


def test_leg_namedtuple_direction_parsing():
    legs = parse_slip("1:SD:B.hits:over:1.5 1:SD:B.runs:U:1.6")
    assert legs[0] == Leg("1", "SD", "B.hits", True, 1.5)
    assert legs[1].higher is False
