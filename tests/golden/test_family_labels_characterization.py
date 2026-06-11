"""Exact-value characterization pin for ``dashboard_detail.family_labels_for_game``.

The existing ``tests/test_dashboard_detail.py`` test asserts only PROPERTIES
(distinct labels, deterministic, star-not-bench). This pins the EXACT label
strings so the CC-23 decomposition of the star-carousel logic (the iterrows
accumulation + the two family-assignment loops) is proven behavior-preserving:
the ``_phrase`` md5 seeding, the stardom ranking, and the #1-star-to-most-driven-
family assignment all resolve to the same headlines.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.components.stories import family_labels_for_game

# Star Wing: 3 markets (Points/Rebounds/Assists), biggest lines -> #1 star.
# Other Star: 2 markets -> #2. Bench Guy: 1 tiny prop -> never headlines.
_GAME = pd.DataFrame(
    {
        "Family": [1.0, 1.0, 2.0, 2.0],
        "Leg 1": [
            "Star Wing Over 28.5 Points - 60%, 1.0x",
            "Star Wing Over 9.5 Rebounds - 58%, 1.0x",
            "Star Wing Over 28.5 Points - 60%, 1.0x",
            "Star Wing Over 6.5 Assists - 57%, 1.0x",
        ],
        "Leg 2": [
            "Other Star Over 22.5 Points - 59%, 1.0x",
            "Other Star Over 7.5 Assists - 55%, 1.0x",
            "Other Star Over 22.5 Points - 59%, 1.0x",
            "Other Star Over 7.5 Assists - 55%, 1.0x",
        ],
        "Leg 3": [
            "Bench Guy Under 3.5 Rebounds - 70%, 1.0x",
            "Bench Guy Under 3.5 Rebounds - 70%, 1.0x",
            None,
            None,
        ],
    }
)


def test_family_labels_exact_headlines():
    assert family_labels_for_game(_GAME) == {
        1.0: "Star Wing fills it up",
        2.0: "Other Star fills it up",
    }


def test_family_labels_empty_when_no_families():
    empty = pd.DataFrame({"Family": [], "Leg 1": []})
    assert family_labels_for_game(empty) == {}


def test_family_labels_mixed_bag_when_no_parseable_legs():
    no_legs = pd.DataFrame({"Family": [1.0, 2.0], "Leg 1": [None, None]})
    assert family_labels_for_game(no_legs) == {1.0: "Mixed bag", 2.0: "Mixed bag"}
