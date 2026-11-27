# ARCHIVED 2026-11-26 from src/sportstradamus/strategies/_ledger_selection.py
# Reason: Every ledger persona but safe ranks by an entry's fitted value (policy_v5), as
#         pick'em's rank_and_dedupe does, so this EV-per-dollar-times-hit-chance key lost
#         its callers.
# Last live SHA: 3eacab96
# Original imports (now unresolved here):
#   from decimal import Decimal
#   from sportstradamus.strategies._ledger_selection import LedgerCandidate


def ev_per_dollar(candidate: LedgerCandidate) -> float:
    """Mirrors _pickem_emit.rank_and_dedupe's ranking key exactly (that file
    is untouched -- this is an independent, intentional re-derivation so the
    personas below share one formula instead of a copy each).
    """
    return (candidate.ev * candidate.joint_prob) / float(max(candidate.stake, Decimal("0.01")))
