"""Canonical schema for the normalized prediction-history frame.

`prophecize` (``prediction/cli.py``) writes player predictions to the history
parquet as one row per ``(Player, League, Date, Market)`` group, with the
per-offer rows collapsed into an ``Offers`` list of fixed-arity tuples.
`reflect` (``clv.py`` and ``analysis.py``) reads that frame back to fold in
closing-line value and to explode offers for scoring. The writer and both
readers must agree on the exact column set and offer-tuple shape; this module
is the single source of truth so a schema change lands in one place instead of
three.
"""

PREDICTION_LEVEL_COLS = [
    "Team",
    "Model EV",
    "Books EV",
    "Dist",
    "CV",
    "Model Param",
    "Gate",
    "Temperature",
    "Disp Cal",
    "Step",
]

# ``Team`` is hoisted next to the identity key; the rest of the prediction-level
# block follows the (Player, League, Team, Date, Market) prefix.
HISTORY_COLS = (
    ["Player", "League", "Team", "Date", "Market"]
    + [c for c in PREDICTION_LEVEL_COLS if c != "Team"]
    + ["Offers", "Actual"]
)

# The trailing three (the closing trio) are NaN at prediction time and filled by
# `reflect` once the line closes.
OFFER_FIELDS = [
    "Line",
    "Boost",
    "Platform",
    "Bet",
    "Model P",
    "Books P",
    "Close Books P",
    "Market CLV",
    "Model CLV",
]
OFFER_ARITY = len(OFFER_FIELDS)
# Offers written before the CLV migration carried only the first six fields.
LEGACY_OFFER_ARITY = 6
# Tuple positions of the closing trio (Close Books P, Market CLV, Model CLV).
CLOSING_FIELD_INDICES = tuple(range(LEGACY_OFFER_ARITY, OFFER_ARITY))
