"""Canonical schema for the flat prediction-history frame.

`prophecize` (``prediction/cli.py``) writes player predictions to the history
parquet as one row per ``(Player, League, Date, Market)`` x book offer —
prediction-level columns (Team, Projection, Dist, ...) are duplicated across
every offer row for the same prediction. `reflect` (``clv.py`` and
``analysis.py``) reads that frame back to fold in closing-line value and to
compute per-offer outcomes. The writer and both readers must agree on the
exact column set; this module is the single source of truth so a schema
change lands in one place instead of three.

A single ``(Player, League, Date, Market)`` never spans more than one
``Team`` in production history (verified against 14,776 real groups), so
``Team`` is a prediction-level column, not part of the identity key.
"""

PREDICTION_KEY = ["Player", "League", "Date", "Market"]

PREDICTION_LEVEL_COLS = [
    "Team",
    "Projection",
    "Market Projection",
    "Dist",
    "CV",
    "Model Param",
    "Gate",
    "Temperature",
    "Disp Cal",
    "Step",
    # Identity of the model that produced this prediction (train-time stamp, or a
    # synthesized ``legacy.<sha>`` for pre-stamp pickles). WS-1 era-aware live reads
    # join on it; pre-stamp history rows carry NaN and fall back to the era backfill.
    "Model Version",
]

OFFER_LEVEL_COLS = [
    "Line",
    "Boost",
    "Platform",
    "Bet",
    "Win Prob",
    "Market Prob",
    "Close Market Prob",
    "Market CLV",
    "Model CLV",
    # |Line - Consensus Line| > tolerance, stamped at write time (cli.py).
    "Alt Line",
]

HISTORY_COLS = PREDICTION_KEY + PREDICTION_LEVEL_COLS + OFFER_LEVEL_COLS + ["Actual"]
