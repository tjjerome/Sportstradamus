"""Per-offer "the case" string: ``attach_offer_why`` and its clause helpers.

Builds a deterministic 1-2 clause sentence per offer row from columns already on
``current_offers``. ``Avg 5`` / ``Avg H2H`` are stored as average-minus-line
deviations; the edge clause prefers explicit ``Model P`` vs ``Books P`` and falls
back to the model/book EV multiples (payout x, > 1.0 is +EV).
"""

import pandas as pd

# Implied-probability gap (model vs book hit prob) below which the two sides
# effectively agree and the edge clause is dropped.
_EDGE_CLAUSE_FLOOR: float = 0.02

# Model-vs-book EV-multiple gap below which the two prices agree (or the book
# EV is just the model EV with no consensus book) and the price clause is
# dropped. The columns are payout multiples; >1.0 is +EV.
_EV_GAP_FLOOR: float = 0.03

# A rolling-average-minus-line deviation this small reads as "right at the
# number" and is phrased as such instead of above/below.
_FORM_AT_LINE: float = 0.25

# DVPOA (defense-vs-position-over-average) magnitude that separates a matchup
# worth calling out from neutral noise. The live column rides ~[-0.30, 0.60]
# with most mass inside ±0.05, so 0.05 keeps the clause for genuine edges.
_DVPOA_NOTE_FLOOR: float = 0.05


def attach_offer_why(offers: pd.DataFrame) -> pd.DataFrame:
    """Add a ``Why`` column: a deterministic 1-2 clause case per offer row.

    Each clause is guarded on its source column being present and non-null, so
    rows missing a feature simply contribute fewer clauses; a row with no usable
    columns gets ``""``. Built only from columns already on ``current_offers``
    (``Avg 5``, ``Avg H2H``, ``Line``, ``DVPOA``, ``Model P`` / ``Books`` /
    ``Model``, ``Bet``). The frame is mutated in place and returned.
    """
    if offers.empty:
        offers["Why"] = pd.Series(dtype=str)
        return offers
    offers["Why"] = [_offer_case(row) for _, row in offers.iterrows()]
    return offers


def _offer_case(row: pd.Series) -> str:
    clauses = [c for c in (_form_clause(row), _matchup_clause(row), _edge_clause(row)) if c]
    if not clauses:
        return ""
    sentence = ", ".join(clauses)
    return sentence[0].upper() + sentence[1:] + "."


def _val(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    v = row[col]
    return None if pd.isna(v) else float(v)


def _form_clause(row: pd.Series) -> str:
    """Form vs the line. ``Avg 5`` / ``Avg H2H`` are stored as average-minus-line.

    ``Avg H2H == 0`` is the no-head-to-head-history sentinel (set in
    ``model_prob`` when ``H2HPlayed == 0``), so an exact 0.0 H2H is dropped
    rather than read as "even with the line".
    """
    avg5, line = _val(row, "Avg 5"), _val(row, "Line")
    if avg5 is None or line is None:
        return ""
    pronoun = "their" if " vs. " in str(row.get("Player", "")) else "his"
    clause = f"{_above_below(avg5)} a {line:g} line over {pronoun} last 5"
    h2h = _val(row, "Avg H2H")
    if h2h is not None and abs(h2h) >= _FORM_AT_LINE:
        clause += f", and {_above_below(h2h)} it head-to-head"
    return clause


def _above_below(deviation: float) -> str:
    if abs(deviation) < _FORM_AT_LINE:
        return "sitting right on"
    return f"{abs(deviation):g} {'above' if deviation > 0 else 'below'}"


def _matchup_clause(row: pd.Series) -> str:
    dvpoa = _val(row, "DVPOA")
    if dvpoa is None or abs(dvpoa) < _DVPOA_NOTE_FLOOR:
        return ""
    over = row.get("Bet") == "Over"
    favorable = (dvpoa > 0) == over
    return "a favorable matchup on paper" if favorable else "into a tough matchup"


def _edge_clause(row: pd.Series) -> str:
    """Model-vs-book disagreement.

    Prefers explicit hit probabilities (``Model P`` vs ``Books P``) when the
    book column is present. The live snapshot carries no book hit-prob column,
    so it falls back to the EV multiples (``Model`` / ``Books``, payout x where
    > 1.0 is +EV); when no consensus book exists those two are equal and the
    clause is dropped.
    """
    model_p, book_p = _val(row, "Model P"), _val(row, "Books P")
    if book_p is not None and model_p is not None:
        gap = model_p - book_p
        if abs(gap) < _EDGE_CLAUSE_FLOOR:
            return ""
        side = (row.get("Bet") or "the model's side").lower()
        lead = "edge" if gap > 0 else "the book pushing back"
        return (
            f"model {round(model_p * 100)}% vs book {round(book_p * 100)}% on the {side}, "
            f"a {round(abs(gap) * 100)}-pt {lead}"
        )
    return _ev_clause(row)


def _ev_clause(row: pd.Series) -> str:
    model_ev, book_ev = _val(row, "Model"), _val(row, "Books")
    if model_ev is None:
        return ""
    if book_ev is None or abs(model_ev - book_ev) < _EV_GAP_FLOOR:
        return f"the model prices it at {model_ev:.2f}x" if model_ev >= 1.0 else ""
    return f"the model prices it at {model_ev:.2f}x against the book's {book_ev:.2f}x"
