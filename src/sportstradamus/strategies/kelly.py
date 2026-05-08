"""Fractional-Kelly stake sizing for Underdog Pick'em recommendations.

Phase 3 §3.1. Public API:

- :func:`fractional_kelly_stake` — single-bet quarter-Kelly with shrinkage.
- :func:`joint_kelly_portfolio` — multi-bet portfolio via cvxpy SCS.
- :func:`resolve_shrinkage` — combine live (CLV) and training Brier
  skill scores into a single ``[0, 1]`` shrinkage weight; rules
  documented in ``strategies/README.md``.

``effective_p = 0.5 + (win_prob - 0.5) * shrinkage`` — shrinkage at
``1.0`` is a no-op, ``0.0`` collapses to a coin flip and forces zero
stake. Shrinkage is plumbed through every entrypoint so the dashboard
can audit which source (CLV vs. training) sized any given bet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

from sportstradamus.helpers.logging import get_logger

_logger = get_logger("kelly")

# Quarter-Kelly hedge against win-probability estimate variance (edge-suite §5.1).
DEFAULT_KELLY_FRACTION: float = 0.25

# 0.5 %-per-entry hard cap from edge-suite §5.1; bounds tail-risk on a
# misestimated edge regardless of what fractional Kelly recommends.
MAX_FRACTION_OF_BANKROLL: float = 0.005

# Below this shrinkage value the win probability collapses to 0.5;
# treat the leg as no-information and stake zero rather than negative.
SHRINKAGE_FLOOR: float = 0.0

# CLV-segment leg count below which live BSS is ignored; the segment is
# still mostly noise at this size even if it cleared CLV_SEGMENT_MIN_N.
LIVE_BLEND_FLOOR: int = 25

# At this many legs the live signal fully dominates training-time BSS;
# current-season conditions have moved enough that training is stale.
LIVE_BLEND_FULL: int = 100

# Smallest representable stake; sub-cent rounds flow into bankroll, not bets.
_STAKE_QUANTUM: Decimal = Decimal("0.01")


@dataclass(frozen=True)
class KellyCandidate:
    """One bet considered by :func:`joint_kelly_portfolio`."""

    bet_id: str
    win_prob: float
    payout_multiplier: Decimal
    model_shrinkage: float = 1.0


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
    5. Neither → ``1.0``, logged at DEBUG.

    NaNs in ``training_bss`` / ``live_bss`` are treated as missing.
    Result is clipped to ``[SHRINKAGE_FLOOR, 1.0]``.
    """
    if explicit is not None:
        return _clip01(explicit)

    has_train = training_bss is not None and not _isnan(training_bss)
    has_live = live_bss is not None and not _isnan(live_bss) and live_n > 0

    if has_train and has_live:
        if live_n <= LIVE_BLEND_FLOOR:
            return _clip01(training_bss)
        if live_n >= LIVE_BLEND_FULL:
            return _clip01(live_bss)
        w_live = (live_n - LIVE_BLEND_FLOOR) / (LIVE_BLEND_FULL - LIVE_BLEND_FLOOR)
        blended = w_live * float(live_bss) + (1.0 - w_live) * float(training_bss)
        return _clip01(blended)

    if has_train:
        return _clip01(training_bss)
    if has_live:
        return _clip01(live_bss)

    _logger.debug("kelly shrinkage fallback: no training or live BSS available; using 1.0")
    return 1.0


def fractional_kelly_stake(
    bankroll: Decimal,
    win_prob: float,
    payout_multiplier: Decimal,
    fraction: float = DEFAULT_KELLY_FRACTION,
    model_shrinkage: float = 1.0,
    max_fraction_of_bankroll: float = MAX_FRACTION_OF_BANKROLL,
) -> Decimal:
    """Return the recommended stake for a single bet.

    Args:
        bankroll: Total bankroll in dollars.
        win_prob: Model probability of winning, in ``[0, 1]``.
        payout_multiplier: Total payout per dollar staked on a win
            (e.g. ``Decimal("3")`` for a 3× return — Underdog Power
            two-leg). Net odds ``b = payout_multiplier - 1``.
        fraction: Kelly fraction multiplier (default quarter-Kelly).
        model_shrinkage: ``[0, 1]`` weight on the model edge; ``1.0``
            means full trust, ``0.0`` collapses to no-information.
        max_fraction_of_bankroll: Hard cap as a fraction of bankroll;
            see :data:`MAX_FRACTION_OF_BANKROLL` for the source.

    Returns:
        Stake as a :class:`~decimal.Decimal` quantized to the cent.
        Returns ``Decimal("0")`` for −EV bets or shrinkage at/below
        :data:`SHRINKAGE_FLOOR`.
    """
    if model_shrinkage <= SHRINKAGE_FLOOR:
        return Decimal("0")

    bankroll = Decimal(bankroll)
    payout_multiplier = Decimal(payout_multiplier)

    p = 0.5 + (float(win_prob) - 0.5) * _clip01(model_shrinkage)
    b = float(payout_multiplier) - 1.0
    if b <= 0.0:
        return Decimal("0")

    q = 1.0 - p
    raw_kelly = (b * p - q) / b  # f* = (bp - q) / b
    if raw_kelly <= 0.0:
        return Decimal("0")

    target_fraction = min(raw_kelly * float(fraction), float(max_fraction_of_bankroll))
    stake = bankroll * Decimal(repr(target_fraction))
    return _quantize_stake(stake)


def joint_kelly_portfolio(
    bankroll: Decimal,
    candidates: list[KellyCandidate],
    fraction: float = DEFAULT_KELLY_FRACTION,
) -> dict[str, Decimal]:
    """Allocate stakes across independent candidates by maximising
    expected log-bankroll under a Kelly-fraction budget.

    Solves with cvxpy's SCS solver; cvxpy is imported lazily so the core
    package install does not require it. Bets assumed independent —
    callers should de-overlap legs before invocation.

    Args:
        bankroll: Total bankroll in dollars.
        candidates: Bets to consider. ``model_shrinkage`` per candidate
            is applied to that candidate's ``win_prob`` only.
        fraction: Kelly fraction multiplier; total allocation is bounded
            by ``fraction * bankroll`` and each leg additionally by the
            single-bet hard cap :data:`MAX_FRACTION_OF_BANKROLL`.

    Returns:
        Mapping of ``bet_id`` to staked dollars. Bets that solve to
        below the cent quantum are dropped from the dict.
    """
    if not candidates:
        return {}

    cvxpy = _lazy_import("cvxpy")

    bankroll = Decimal(bankroll)
    sized_p: list[float] = []
    sized_b: list[float] = []
    bet_ids: list[str] = []
    for c in candidates:
        s = _clip01(c.model_shrinkage)
        if s <= SHRINKAGE_FLOOR:
            continue
        p = 0.5 + (float(c.win_prob) - 0.5) * s
        b = float(c.payout_multiplier) - 1.0
        if b <= 0.0 or b * p - (1.0 - p) <= 0.0:
            continue
        sized_p.append(p)
        sized_b.append(b)
        bet_ids.append(c.bet_id)

    if not bet_ids:
        return {}

    f = cvxpy.Variable(len(bet_ids), nonneg=True)
    log_terms = []
    for i, (p, b) in enumerate(zip(sized_p, sized_b, strict=True)):
        log_terms.append(p * cvxpy.log(1.0 + b * f[i]) + (1.0 - p) * cvxpy.log(1.0 - f[i]))
    objective = cvxpy.Maximize(sum(log_terms))
    constraints = [
        cvxpy.sum(f) <= float(fraction),
        f <= float(MAX_FRACTION_OF_BANKROLL),
    ]
    cvxpy.Problem(objective, constraints).solve(solver=cvxpy.SCS)

    if f.value is None:
        return {}

    out: dict[str, Decimal] = {}
    for bet_id, frac_i in zip(bet_ids, f.value, strict=True):
        stake = _quantize_stake(bankroll * Decimal(repr(float(max(frac_i, 0.0)))))
        if stake > Decimal("0"):
            out[bet_id] = stake
    return out


# --------------------------------------------------------------------------- #
# CLI


def _build_cli() -> Any:
    import click

    @click.command()
    @click.option("--bankroll", type=str, required=True, help="Bankroll in dollars.")
    @click.option(
        "--from",
        "yaml_path",
        type=click.Path(exists=True, dir_okay=False),
        required=True,
        help="Path to a recommendations YAML produced by pickem-build.",
    )
    @click.option(
        "--fraction",
        type=float,
        default=DEFAULT_KELLY_FRACTION,
        show_default=True,
        help="Kelly fraction multiplier.",
    )
    def kelly(bankroll: str, yaml_path: str, fraction: float) -> None:
        """Re-size stakes from a recommendations YAML and print a table."""
        yaml = _lazy_import("yaml")
        tabulate = _lazy_import("tabulate")

        with open(yaml_path) as fh:
            doc = yaml.safe_load(fh)

        entries = doc.get("entries", []) if isinstance(doc, dict) else []
        bankroll_dec = Decimal(bankroll)
        rows = []
        for entry in entries:
            stake = fractional_kelly_stake(
                bankroll=bankroll_dec,
                win_prob=float(entry.get("joint_prob", 0.0)),
                payout_multiplier=Decimal(str(entry.get("payout_multiplier", 0))),
                fraction=fraction,
                model_shrinkage=float(entry.get("shrinkage", 1.0)),
            )
            rows.append(
                [
                    entry.get("id", ""),
                    entry.get("contest_variant", ""),
                    entry.get("entry_size", ""),
                    f"{float(entry.get('joint_prob', 0.0)):.3f}",
                    f"{float(entry.get('payout_multiplier', 0.0)):.2f}",
                    f"{float(entry.get('ev', 0.0)):+.3f}",
                    f"{stake:.2f}",
                ]
            )
        click.echo(
            tabulate.tabulate(
                rows,
                headers=["id", "variant", "size", "p", "payout", "ev", "stake"],
                tablefmt="github",
            )
        )

    return kelly


def main() -> None:
    _build_cli()()


# --------------------------------------------------------------------------- #
# Internal helpers


def _clip01(x: float) -> float:
    return max(SHRINKAGE_FLOOR, min(1.0, float(x)))


def _isnan(x: float | None) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _quantize_stake(stake: Decimal) -> Decimal:
    if stake <= Decimal("0"):
        return Decimal("0")
    return stake.quantize(_STAKE_QUANTUM, rounding=ROUND_DOWN)


def _lazy_import(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:
        raise ImportError(
            f"`{name}` is required for this code path. Install with "
            f"`poetry install --with strategy`."
        ) from exc


if __name__ == "__main__":
    main()
