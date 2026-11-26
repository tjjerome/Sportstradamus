"""How an entry's numbers read on the slip rail, the locked shelf and the story preloader.

An entry's fitted value leads as EV with the model's claim beside it as Model EV, in the
format a leg prints them. An entry with no fitted value (the fit did not price one of its
legs, or it was stored before the fit existed) shows its Model EV alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from sportstradamus.dashboard.slip_engine import SlipScore, payout_check
from sportstradamus.leg_schema import leg_label


def ev_text(ev: float | None, model_ev: float) -> str:
    """``"EV 0.95 · Model EV 1.12"``, or ``"Model EV 1.12"`` when ``ev`` is None or NaN."""
    claim = f"Model EV {model_ev:.2f}"
    return claim if pd.isna(ev) else f"EV {ev:.2f} · {claim}"


def rail_lines(score: SlipScore, legs: Sequence[Mapping]) -> list[str]:
    """The rail's lines for a slip of two or more legs.

    A priced slip leads with EV beside Model EV, then the Kelly stake and the fitted joint,
    and the pays-at-least lines follow. A slip the fit did not price shows its Model EV and
    its stake, and names each leg it could not price. A slip the platform refuses has no
    lines: the pair note says why.
    """
    if score.banned:
        return []
    numbers = f"{ev_text(score.ev, score.model_ev)} · Kelly stake ${score.stake}"
    if score.ev is None:
        unpriced = ", ".join(leg_label(leg) for leg in legs if leg["fit_prob"] is None)
        return [numbers, f"Not priced: only one side is posted for {unpriced}."]
    return [
        f"{numbers} · joint {score.joint_p:.0%}",
        *payout_check(legs, score.play_type, score.joint_p, score.payout),
    ]
