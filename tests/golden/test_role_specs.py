"""Coverage + parity pins for the role×position registry (`training/role_specs.py`).

Guards two invariants the generic two-part strategy relies on:

* Every continuous (SkewNormal/Mixture) cell outside MLB resolves to a
  ``RoleSpec`` so the sweep can search the method league-wide; MLB stays
  uncovered by design.
* ``_POSITION_LABELS`` mirrors each league's ``Stats.positions`` list, since the
  registry duplicates those rosters to stay import-light.
"""

from __future__ import annotations

import json
from importlib.resources import files

import pytest

from sportstradamus.stats import StatsNBA, StatsNFL, StatsNHL, StatsWNBA
from sportstradamus.training.role_specs import (
    _POSITION_LABELS,
    RoleSpec,
    league_position_codes,
    position_label,
    role_spec_for,
)

_CONTINUOUS = {"SkewNormal", "Mixture"}
_COVERED = ("NFL", "NBA", "WNBA", "NHL")


def _stat_meta() -> dict:
    return json.loads((files("sportstradamus.data") / "config" / "stat_meta.json").read_text())


def _continuous_cells(league: str) -> list[str]:
    return [m for m, c in _stat_meta()[league].items() if c.get("dist") in _CONTINUOUS]


@pytest.mark.parametrize(
    ("league", "stats_cls"),
    [("NFL", StatsNFL), ("NBA", StatsNBA), ("WNBA", StatsWNBA), ("NHL", StatsNHL)],
)
def test_position_labels_mirror_stats_roster(league, stats_cls):
    assert _POSITION_LABELS[league] == tuple(stats_cls().positions)
    assert league_position_codes(league) == tuple(range(1, len(stats_cls().positions) + 1))
    assert position_label(league, 1) == stats_cls().positions[0]


@pytest.mark.parametrize("league", _COVERED)
def test_every_continuous_cell_is_covered(league):
    uncovered = [m for m in _continuous_cells(league) if role_spec_for(league, m) is None]
    assert uncovered == [], f"{league} continuous cells without a role spec: {uncovered}"


def test_mlb_continuous_cells_stay_uncovered():
    covered = [m for m in _continuous_cells("MLB") if role_spec_for("MLB", m) is not None]
    assert covered == [], f"MLB should be excluded from the role method: {covered}"


@pytest.mark.parametrize("league", _COVERED)
def test_specs_are_well_formed(league):
    for market in _continuous_cells(league):
        spec = role_spec_for(league, market)
        assert isinstance(spec, RoleSpec)
        assert spec.all_columns[-1] == "Player position"
        assert spec.role_columns, f"{league} {market} has no role columns"
        assert not spec.efficiency_cols or len(spec.efficiency_cols) == len(spec.usage_share_cols)
