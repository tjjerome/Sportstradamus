"""Unit tests for the per-cell ship config (``training/ship_config.py``).

``ship_config.json`` assigns each ``(league, market)`` cell one of three states:
a real strategy slug (shipped), the reserved ``"withheld"`` sentinel (skip +
prune), or absent (train with the run's default strategy). ``meditate`` reads it
to decide, per cell, what to train and what to dark-out on the production server.
"""

from __future__ import annotations

import json

import pytest

from sportstradamus.training.ship_config import (
    WITHHELD,
    load_ship_config,
    resolve_cell_strategy,
    resolve_cell_zinb_mode,
)

# A real non-default slug, so a "shipped" cell is behaviorally distinct from an
# "absent" cell (both must not collapse to the same resolved strategy).
_SHIPPED_SLUG = "centered_additive_mean10"

# An object-form cell: explicit strategy + per-cell zinb_mode. This is the
# sacks-taken case — a count market that ships as ratio_meanyr under the hurdle
# ZINB architecture, which a bare-string cell cannot express.
_HURDLE_CELL = {"strategy": "ratio_meanyr", "zinb_mode": "hurdle"}


def test_load_missing_file_returns_empty(tmp_path) -> None:
    """A missing config means no cell is shipped or withheld — a strict no-op."""
    assert load_ship_config(tmp_path / "does_not_exist.json") == {}


def test_load_valid_config_round_trips(tmp_path) -> None:
    """A config with a real slug and the withheld sentinel loads unchanged."""
    cfg = {"NBA": {"FG3M": _SHIPPED_SLUG, "STL": WITHHELD}}
    path = tmp_path / "ship_config.json"
    path.write_text(json.dumps(cfg))
    assert load_ship_config(path) == cfg


def test_load_rejects_unknown_strategy(tmp_path) -> None:
    """A value that is neither a known slug nor 'withheld' fails fast at load."""
    path = tmp_path / "ship_config.json"
    path.write_text(json.dumps({"NBA": {"FG3M": "not_a_real_strategy"}}))
    with pytest.raises(ValueError):
        load_ship_config(path)


def test_resolve_absent_cell_uses_flag_strategy() -> None:
    """A cell absent from the map trains with the run's default flag strategy."""
    assert resolve_cell_strategy("NBA", "PTS", "ratio_meanyr", {}) == "ratio_meanyr"


def test_resolve_shipped_cell_uses_mapped_strategy() -> None:
    """A shipped cell's mapped strategy wins over the flag (the map is authoritative)."""
    cfg = {"NBA": {"FG3M": _SHIPPED_SLUG}}
    assert resolve_cell_strategy("NBA", "FG3M", "ratio_meanyr", cfg) == _SHIPPED_SLUG


def test_resolve_withheld_cell_returns_sentinel() -> None:
    """A withheld cell resolves to WITHHELD so meditate prunes + skips it."""
    cfg = {"NBA": {"STL": WITHHELD}}
    assert resolve_cell_strategy("NBA", "STL", "ratio_meanyr", cfg) == WITHHELD


def test_load_object_cell_round_trips(tmp_path) -> None:
    """An object-form cell {strategy, zinb_mode} loads unchanged (no normalizing)."""
    cfg = {"NFL": {"sacks taken": _HURDLE_CELL}}
    path = tmp_path / "ship_config.json"
    path.write_text(json.dumps(cfg))
    assert load_ship_config(path) == cfg


def test_load_rejects_unknown_zinb_mode(tmp_path) -> None:
    """An object cell whose zinb_mode is outside {joint, hurdle} fails fast."""
    path = tmp_path / "ship_config.json"
    path.write_text(json.dumps({"NFL": {"sacks taken": {"strategy": "ratio_meanyr", "zinb_mode": "bogus"}}}))
    with pytest.raises(ValueError):
        load_ship_config(path)


def test_load_rejects_object_unknown_strategy(tmp_path) -> None:
    """An object cell with an unknown strategy fails fast, like the string form."""
    path = tmp_path / "ship_config.json"
    path.write_text(json.dumps({"NFL": {"sacks taken": {"strategy": "nope", "zinb_mode": "hurdle"}}}))
    with pytest.raises(ValueError):
        load_ship_config(path)


def test_load_rejects_object_unexpected_key(tmp_path) -> None:
    """An object cell with an unexpected key fails fast (typo guard, e.g. 'znib_mode')."""
    path = tmp_path / "ship_config.json"
    path.write_text(json.dumps({"NFL": {"sacks taken": {"strategy": "ratio_meanyr", "znib_mode": "hurdle"}}}))
    with pytest.raises(ValueError):
        load_ship_config(path)


def test_resolve_strategy_from_object_cell() -> None:
    """resolve_cell_strategy unwraps the strategy from an object-form cell."""
    cfg = {"NFL": {"sacks taken": _HURDLE_CELL}}
    assert resolve_cell_strategy("NFL", "sacks taken", "ratio_meanyr", cfg) == "ratio_meanyr"


def test_resolve_zinb_mode_from_object_cell() -> None:
    """resolve_cell_zinb_mode returns the cell's explicit zinb_mode over the flag."""
    cfg = {"NFL": {"sacks taken": _HURDLE_CELL}}
    assert resolve_cell_zinb_mode("NFL", "sacks taken", "joint", cfg) == "hurdle"


def test_resolve_zinb_mode_bare_string_uses_flag() -> None:
    """A bare-string (strategy-only) cell falls back to the run's flag zinb_mode."""
    cfg = {"NFL": {"receptions": "centered_additive_mean10"}}
    assert resolve_cell_zinb_mode("NFL", "receptions", "joint", cfg) == "joint"


def test_resolve_zinb_mode_absent_cell_uses_flag() -> None:
    """A cell absent from the map falls back to the run's flag zinb_mode."""
    assert resolve_cell_zinb_mode("NFL", "passing tds", "joint", {}) == "joint"


def test_resolve_zinb_mode_object_without_mode_uses_flag() -> None:
    """An object cell that omits zinb_mode falls back to the flag (strategy still wins)."""
    cfg = {"NFL": {"sacks taken": {"strategy": "ratio_meanyr"}}}
    assert resolve_cell_zinb_mode("NFL", "sacks taken", "joint", cfg) == "joint"
