"""Scoring-column semantics — the one home for what the snapshot's score columns mean.

The prediction snapshot already ships clear scoring names (``Win Prob``, ``Model EV``,
``Market EV``, ``Kelly``) after the scoring-column rename, so this module only relabels
``Win Prob`` to the shorter **Win %** header, attaches the header tooltips, and derives
one **Edge** column (``Model EV`` minus ``Market EV`` — the number a bettor actually
compares). Display-only: the raw snapshot columns are unchanged.
"""

from __future__ import annotations

import pandas as pd

# Snapshot column -> user-facing header. Only ``Win Prob`` is relabeled; ``Model EV`` /
# ``Market EV`` / ``Kelly`` / ``Edge`` already read clearly and pass through unchanged.
LABELS = {"Win Prob": "Win %"}

# One plain sentence per user-facing header, shown as the grid header tooltip.
HELP = {
    "Win %": "Model's probability the pick hits.",
    "Model EV": "Model's expected return per $1 staked (Win % × payout); 1.00 is break-even.",
    "Market EV": "Consensus market's expected return per $1 at the same payout; 1.00 is break-even.",
    "Edge": "Model EV minus Market EV: how much more the model thinks this pays than the market.",
    "Kelly": "Kelly edge — the bankroll fraction full-Kelly would stake on this leg.",
}


def edge_series(df: pd.DataFrame) -> pd.Series:
    """Edge = Model EV − Market EV as a numeric Series (NaN where either EV is missing)."""
    return pd.to_numeric(df["Model EV"], errors="coerce") - pd.to_numeric(
        df["Market EV"], errors="coerce"
    )


def add_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Append the derived ``Edge`` column (``Model EV − Market EV``) when both are present.

    Returns the frame unchanged when either EV column is missing.
    """
    if "Model EV" not in df.columns or "Market EV" not in df.columns:
        return df
    df = df.copy()
    df["Edge"] = edge_series(df)
    return df
