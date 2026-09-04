"""Per-offer "the case" string and the story-menu dek: facts in code, prose in data.

Builds deterministic clause sentences from columns already on
``current_offers``. Every template string lives in ``why_bank.json`` (clause →
branch → variants); this module computes the facts, picks the branch, formats
the numbers, and rotates the variant by an md5 of the row key — repeat users
see varied phrasing that never changes for the same row and date. ``Avg 5`` /
``Avg H2H`` are stored as average-minus-line deviations; the edge clause
prefers explicit ``Win Prob`` vs ``Market Prob`` and falls back to the
model/book EV multiples (``Model EV`` / ``Market EV``, payout x where > 1.0 is +EV);
the MLB lineup clauses read the columns ``stories.lineup`` attaches.
"""

import hashlib
from collections.abc import Mapping, Sequence
from itertools import combinations

import pandas as pd

from sportstradamus.leg_schema import leg_label
from sportstradamus.prediction.stories.bank import why_bank
from sportstradamus.prediction.stories.legs import lower_leg, offer_index
from sportstradamus.prediction.stories.lineup import batting_slot

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

# A core whose mean pairwise rho sits below the menu's cluster floor (or is
# negative) would contradict the "move together" cluster prose — drop the clause.
_DEK_CLUSTER_RHO_FLOOR: float = 0.05

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
    ``Model``, ``Bet``, ``Position`` / ``Bats`` / ``Opp Hand`` / ``Lineup``).
    The frame is mutated in place and returned.
    """
    if offers.empty:
        offers["Why"] = pd.Series(dtype=str)
        return offers
    offers["Why"] = [_offer_case(row) for _, row in offers.iterrows()]
    return offers


def _offer_case(row: pd.Series) -> str:
    clauses = [
        c
        for c in (
            _form_clause(row),
            _matchup_clause(row),
            _lineup_clause(row),
            _edge_clause(row),
        )
        if c
    ]
    if not clauses:
        return ""
    sentence = ", ".join(clauses)
    return sentence[0].upper() + sentence[1:] + "."


def _val(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    v = row[col]
    return None if pd.isna(v) else float(v)


def _text(row: Mapping, col: str) -> str:
    """A string column read where missing and NaN both mean "no fact"."""
    v = row.get(col)
    return "" if v is None or pd.isna(v) else str(v)


def _pick(variants: Sequence[str], seed: str) -> str:
    """The md5-rotated variant for a stable seed — deterministic, never random."""
    return variants[int(hashlib.md5(seed.encode()).hexdigest(), 16) % len(variants)]


def _why_variant(row: Mapping, clause: str, branch: str) -> str:
    key = "|".join(str(row.get(c) or "") for c in ("League", "Player", "Market", "Bet", "Date"))
    return _pick(why_bank()["why"][clause][branch], f"{clause}|{key}")


def _form_branch(deviation: float) -> str:
    if abs(deviation) < _FORM_AT_LINE:
        return "at_line"
    return "above" if deviation > 0 else "below"


def _pronoun(row: pd.Series) -> str:
    pronouns = why_bank()["pronouns"]
    if " vs. " in str(row.get("Player", "")):
        return pronouns["combo"]
    return pronouns.get(str(row.get("League", "")), pronouns["default"])


def _form_clause(row: pd.Series) -> str:
    """Form vs the line. ``Avg 5`` / ``Avg H2H`` are stored as average-minus-line.

    ``Avg H2H == 0`` is the no-head-to-head-history sentinel (set in
    ``model_prob`` when ``H2HPlayed == 0``), so an exact 0.0 H2H is dropped
    rather than read as "even with the line".
    """
    avg5, line = _val(row, "Avg 5"), _val(row, "Line")
    if avg5 is None or line is None:
        return ""
    clause = _why_variant(row, "form", _form_branch(avg5)).format(
        dev=f"{abs(avg5):g}", line=f"{line:g}", pronoun=_pronoun(row)
    )
    h2h = _val(row, "Avg H2H")
    if h2h is not None and abs(h2h) >= _FORM_AT_LINE:
        tail = _why_variant(row, "form_h2h", _form_branch(h2h)).format(dev=f"{abs(h2h):g}")
        clause += f", {tail}"
    return clause


def _matchup_clause(row: pd.Series) -> str:
    dvpoa = _val(row, "DVPOA")
    if dvpoa is None or abs(dvpoa) < _DVPOA_NOTE_FLOOR:
        return ""
    favorable = (dvpoa > 0) == (row.get("Bet") == "Over")
    return _why_variant(row, "matchup", "favorable" if favorable else "tough")


def _lineup_clause(row: pd.Series) -> str:
    """Batting slot, plus the platoon read against tonight's probable starter.

    A ``"usual"`` slot is the modal one ``get_depth`` fell back to before the card
    is posted, so it reads as habit ("usually 3rd") from the ``lineup_usual``
    family rather than asserting tonight's order. Guarded on ``Lineup`` as well as
    the slot label: ``attach_lineup_columns`` fills it only for MLB hitters, which
    keeps an NBA bench label (``B1`` is a bench rank there) out of baseball prose.
    """
    slot, lineup = batting_slot(row.get("Position")), _text(row, "Lineup")
    if slot is None or not lineup:
        return ""
    throws = _text(row, "Opp Hand")
    clause = "lineup" if lineup == "posted" else "lineup_usual"
    return _why_variant(row, clause, _lineup_branch(_text(row, "Bats"), throws)).format(
        slot=slot, throws=why_bank()["hands"].get(throws, "")
    )


def _lineup_branch(bats: str, throws: str) -> str:
    if not bats or not throws:
        return "slot_only"
    if bats == "S":
        return "switch"
    return "same_side" if bats == throws else "platoon_edge"


def _edge_clause(row: pd.Series) -> str:
    """Model-vs-book disagreement.

    Prefers explicit hit probabilities (``Win Prob`` vs ``Market Prob``) when the
    book column is present. The live snapshot carries no book hit-prob column,
    so it falls back to the EV multiples (``Model EV`` / ``Market EV``, payout x where
    > 1.0 is +EV); when no consensus book exists those two are equal and the
    clause is dropped.
    """
    model_p, book_p = _val(row, "Win Prob"), _val(row, "Market Prob")
    if book_p is not None and model_p is not None:
        gap = model_p - book_p
        if abs(gap) < _EDGE_CLAUSE_FLOOR:
            return ""
        return _why_variant(row, "edge", "model_ahead" if gap > 0 else "book_back").format(
            model_pct=round(model_p * 100),
            book_pct=round(book_p * 100),
            side=(row.get("Bet") or "the model's side").lower(),
            gap=round(abs(gap) * 100),
        )
    return _ev_clause(row)


def _ev_clause(row: pd.Series) -> str:
    model_ev, book_ev = _val(row, "Model EV"), _val(row, "Market EV")
    if model_ev is None:
        return ""
    if book_ev is None or abs(model_ev - book_ev) < _EV_GAP_FLOOR:
        if model_ev < 1.0:
            return ""
        return _why_variant(row, "ev", "solo").format(ev=f"{model_ev:.2f}")
    return _why_variant(row, "ev", "vs_book").format(ev=f"{model_ev:.2f}", book_ev=f"{book_ev:.2f}")


def story_dek(core_bet_ids: Sequence[int], sctx, offers: pd.DataFrame) -> str:
    """A story card's one-line dek: cluster strength, anchor form, matchup.

    Built from the legs shared by both of the story's presets, so it stays
    valid whichever mode chip is active. Clause-guarded and empty-safe — a
    sub-2-leg core carries no cluster clause and missing offer facts drop
    their clauses; everything deterministic via the md5 rotation.
    """
    descs = [leg_label(lower_leg(sctx.bet_df[i])) for i in core_bet_ids]
    seed_tail = "|".join(sorted(descs)) + f"|{sctx.date}"
    clauses = _cluster_clause(core_bet_ids, sctx, seed_tail)
    anchor = _dek_anchor(core_bet_ids, sctx, offers)
    if anchor is not None:
        clauses += _anchor_clauses(*anchor, seed_tail)
    return " · ".join(clauses)


def _cluster_clause(core_bet_ids: Sequence[int], sctx, seed_tail: str) -> list[str]:
    if len(core_bet_ids) < 2:
        return []
    pairs = list(combinations(core_bet_ids, 2))
    rho = sum(float(sctx.g.C[a, b]) for a, b in pairs) / len(pairs)
    if rho < _DEK_CLUSTER_RHO_FLOOR:
        return []
    template = _pick(why_bank()["dek"]["cluster"], f"dek.cluster|{seed_tail}")
    return [template.format(n=len(core_bet_ids), rho=f"{rho:.2f}")]


def _anchor_clauses(player: str, match: Mapping, seed_tail: str) -> list[str]:
    """Form, matchup, and lineup dek clauses for the core's anchor leg, each fact-guarded."""
    bank = why_bank()["dek"]
    clauses = []
    avg5, line = match.get("Avg 5"), match.get("Line")
    if avg5 is not None and line is not None and not pd.isna(avg5) and not pd.isna(line):
        clauses.append(
            _pick(bank["form"][_form_branch(float(avg5))], f"dek.form|{seed_tail}").format(
                p=player, dev=f"{abs(float(avg5)):g}", line=f"{float(line):g}"
            )
        )
    dvpoa = match.get("DVPOA")
    if dvpoa is not None and not pd.isna(dvpoa) and abs(float(dvpoa)) >= _DVPOA_NOTE_FLOOR:
        favorable = (float(dvpoa) > 0) == (match.get("Bet") == "Over")
        clauses.append(
            _pick(
                bank["matchup"]["favorable" if favorable else "tough"],
                f"dek.matchup|{seed_tail}",
            ).format(p=player)
        )
    lineup_clause = _dek_lineup_clause(player, match, seed_tail)
    if lineup_clause:
        clauses.append(lineup_clause)
    return clauses


def _dek_lineup_clause(player: str, match: Mapping, seed_tail: str) -> str:
    """The anchor's batting slot, said as posted or usual and named with the starter's hand."""
    slot, lineup = batting_slot(match.get("Position")), _text(match, "Lineup")
    if slot is None or not lineup:
        return ""
    throws = why_bank()["hands"].get(_text(match, "Opp Hand"), "")
    branch = f"{lineup}_hand" if throws else lineup
    template = _pick(why_bank()["dek"]["lineup"][branch], f"dek.lineup|{seed_tail}")
    return template.format(p=player, slot=slot, throws=throws)


def _dek_anchor(
    core_bet_ids: Sequence[int], sctx, offers: pd.DataFrame
) -> tuple[str, Mapping] | None:
    """The core's highest-conviction leg and its offer row — the dek's subject."""
    if not core_bet_ids or offers.empty:
        return None
    anchor_id = max(core_bet_ids, key=lambda i: float(sctx.g.p_model[i]))
    row = sctx.bet_df[anchor_id]
    match = offer_index(offers).get((row["Player"], row["Bet"], row["Line"]))
    if match is None:
        return None
    return row["Player"], match
