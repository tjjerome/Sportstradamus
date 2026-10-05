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
    # |Line - reference line| > tolerance, stamped at write time (cli.py). The reference
    # is Consensus Line, else the entry's line of record, which is not stored.
    "Alt Line",
    # Full decimal payout of each side on the platform (0 = side not posted); named
    # "Payout" because `Boost` on this frame is Underdog's raw multiplier. Underdog's are at
    # the UNDERDOG_BOOST_BASELINE in force when the row was scored, so read a side's raw
    # multiplier as `Payout {side} x Boost / Payout {Bet}` (where the chosen side was posted).
    "Payout Over",
    "Payout Under",
]

# Decision-time context the research and trust-layer work reads back from history:
# what the scorer saw when it chose the side. Offer-level, never backfilled, NaN on
# rows written before the column existed.
DECISION_COLS = [
    # One UTC stamp per prophecize run, so a row joins the archive state that run saw.
    "Scored At",
    # The sportsbook consensus line, NaN when no sportsbook posts one.
    "Consensus Line",
    "Commence",
    "Opponent",
    "Home",
    "Player position",
    "Moneyline",
    "O/U",
    "DVPOA",
    "Avg 5",
    "Avg H2H",
    "Push Prob",
    "Projection STD",
    "Books STD",
    "Model EV",
    "Kelly",
    # The pickle's model/book blend weight (0.0 on the book-fallback path).
    "Model Weight",
    # Provenance of the book quote behind the serve; None/NaN when the leg was unquoted.
    "Quote Source",
    "Quote Authenticity",
    "Quote Books",
    "Quote Line",
    "Quote Observed At",
]

HISTORY_COLS = (
    PREDICTION_KEY + PREDICTION_LEVEL_COLS + OFFER_LEVEL_COLS + DECISION_COLS + ["Actual"]
)
