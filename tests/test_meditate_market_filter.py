"""Unit tests for ``select_markets``, the ``meditate --market`` filter helper.

The helper narrows the per-league market list down to a comma-separated set of
requested stems. These tests use the real ``ALL_MARKETS`` registry so the
filter stays honest about which stems actually exist per league.
"""

from __future__ import annotations

import pytest
from click import UsageError

from sportstradamus.training.markets import ALL_MARKETS, select_markets


def test_none_returns_mapping_unchanged() -> None:
    """``None`` means no filtering: the mapping is returned untouched."""
    active = {"NBA": ALL_MARKETS["NBA"]}
    assert select_markets(active, None) == active


def test_filters_to_requested_stems_in_registry_order() -> None:
    """``"FTM,STL"`` keeps only those stems, in the registry's original order."""
    active = {"NBA": ALL_MARKETS["NBA"]}
    result = select_markets(active, "FTM,STL")
    # FTM precedes STL in ALL_MARKETS["NBA"], so registry order is preserved.
    assert result == {"NBA": ["FTM", "STL"]}


def test_tolerates_whitespace_and_empty_tokens() -> None:
    """Surrounding whitespace and empty tokens are stripped before matching."""
    active = {"NBA": ALL_MARKETS["NBA"]}
    result = select_markets(active, " FTM , ,STL ")
    assert result == {"NBA": ["FTM", "STL"]}


def test_unknown_stem_raises_usage_error() -> None:
    """A stem absent from every active league is a hard error."""
    active = {"NBA": ALL_MARKETS["NBA"]}
    with pytest.raises(UsageError):
        select_markets(active, "FTM,NOT_A_REAL_MARKET")


def test_stem_in_one_league_filters_per_league() -> None:
    """A stem present in one active league but not another filters per league.

    ``FTM`` exists in NBA but not NFL. Requesting it leaves NBA with the single
    matching stem and NFL with an empty list — no error, since the stem is
    present in at least one active league.
    """
    active = {"NBA": ALL_MARKETS["NBA"], "NFL": ALL_MARKETS["NFL"]}
    result = select_markets(active, "FTM")
    assert result == {"NBA": ["FTM"], "NFL": []}
