# ARCHIVED 2026-11-26 from src/sportstradamus/strategies/kelly.py
# Reason: Its one caller, underdog_pickem.resolve_market_shrinkage, was archived when
#         every ledger pool came to price a leg at its fitted chance (policy_v5); the
#         blend ramp, its two window constants and the module logger only it used go
#         with it. Their tests were deleted from tests/golden/test_kelly.py (git keeps
#         them).
# Last live SHA: d67670cf
# Original imports (now unresolved here):
#   import math
#   from sportstradamus.helpers.logging import get_logger
#   from sportstradamus.strategies.kelly import NO_EVIDENCE_SHRINKAGE, _clip01

_logger = get_logger("kelly")

# CLV-segment leg count below which live BSS is ignored; the segment is
# still mostly noise at this size even if it cleared CLV_SEGMENT_MIN_N.
LIVE_BLEND_FLOOR: int = 25

# At this many legs the live signal fully dominates training-time BSS;
# current-season conditions have moved enough that training is stale.
LIVE_BLEND_FULL: int = 100


def resolve_shrinkage(
    *,
    explicit: float | None = None,
    training_bss: float | None = None,
    live_bss: float | None = None,
    live_n: int = 0,
) -> float:
    """Resolve a shrinkage weight in ``[0, 1]`` from the available sources.

    Resolution order (matches ``strategies/README.md``):

    1. ``explicit`` — overrides the blend.
    2. Both ``training_bss`` and ``live_bss`` present and
       ``live_n >= LIVE_BLEND_FLOOR`` → blended per the ``w_live`` ramp.
    3. Only ``training_bss`` (or live below the floor) → training_bss.
    4. Only ``live_bss`` (training missing) → live_bss.
    5. Neither → :data:`NO_EVIDENCE_SHRINKAGE`, logged at DEBUG.

    NaNs in ``training_bss`` / ``live_bss`` are treated as missing.
    Result is clipped to ``[SHRINKAGE_FLOOR, 1.0]``.
    """
    if explicit is not None:
        return _clip01(explicit)

    has_train = training_bss is not None and not math.isnan(training_bss)
    has_live = live_bss is not None and not math.isnan(live_bss) and live_n > 0

    if has_train and has_live:
        return _blend_shrinkage(training_bss, live_bss, live_n)
    if has_train:
        return _clip01(training_bss)
    if has_live:
        return _clip01(live_bss)

    _logger.debug("kelly shrinkage fallback: no training or live BSS available; staking near zero")
    return NO_EVIDENCE_SHRINKAGE


def _blend_shrinkage(training_bss: float, live_bss: float, live_n: int) -> float:
    """Ramp from training BSS to live BSS as live_n crosses the blend window."""
    if live_n <= LIVE_BLEND_FLOOR:
        return _clip01(training_bss)
    if live_n >= LIVE_BLEND_FULL:
        return _clip01(live_bss)
    w_live = (live_n - LIVE_BLEND_FLOOR) / (LIVE_BLEND_FULL - LIVE_BLEND_FLOOR)
    blended = w_live * float(live_bss) + (1.0 - w_live) * float(training_bss)
    return _clip01(blended)
