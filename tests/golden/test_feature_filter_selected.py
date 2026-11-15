"""A ``Selected`` list in ``feature_filter.json`` is its cell's whole training feature set.

The pipeline trains on ``M.reindex(columns=get_stat_columns(market))`` and then reads some of
those columns by name: the seeding columns, ``Home``, and whatever the cell's target
normalization baseline takes. Part B's first pruned NFL arms died on ``KeyError: 'GamesPlayed'``
when a drop set took a baseline input. Each list is the column set a confirm-level run trained
on, checked against the cached matrices when written; CI holds no matrices, so these pins cover
the lookup and the columns training reads by name.
"""

import pytest

from sportstradamus.helpers import feature_filter, stat_meta
from sportstradamus.stats import StatsNFL
from sportstradamus.training.baselines import TARGET_NORMALIZATION_SLUGS, get_target_normalization
from sportstradamus.training.pipeline import _SEEDING_REQUIRED_COLUMNS

SELECTED = [
    (league, market, columns)
    for league, buckets in feature_filter.items()
    if isinstance(buckets, dict)
    for market, columns in buckets.get("Selected", {}).items()
]


@pytest.mark.parametrize(
    ("league", "market", "columns"), SELECTED, ids=[f"{lg} {mk}" for lg, mk, _ in SELECTED]
)
def test_a_selected_list_holds_every_column_training_reads_by_name(league, market, columns):
    slug = stat_meta[league][market]["target_normalization"]
    read_by_name = {*_SEEDING_REQUIRED_COLUMNS, "Home"}
    # ratio_projvol's denominator is carried in from the matrix whatever the list holds.
    if slug in TARGET_NORMALIZATION_SLUGS and slug != "ratio_projvol":
        meta = get_target_normalization(slug).offset_meta(0.0)
        read_by_name |= {value for key, value in meta.items() if key.endswith("_col")}

    assert read_by_name <= set(columns)
    assert len(columns) == len(set(columns))


def test_a_cell_with_a_selected_list_trains_on_it_and_the_others_on_the_full_set(monkeypatch):
    stats = StatsNFL()
    full = stats.get_stat_columns("targets")
    monkeypatch.setitem(feature_filter["NFL"], "Selected", {"receptions": ["MeanYr", "Home"]})

    assert stats.get_stat_columns("receptions") == ["Home", "MeanYr"]
    assert stats.get_stat_columns("targets") == full
