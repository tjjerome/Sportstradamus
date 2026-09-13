"""The slip's pair note: which of its pairs the platform refuses or reprices.

Drawn under the slip's price by both builders and the phone dock, from the pair
modifiers ``score_slip`` already priced the slip with.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import streamlit as st

from sportstradamus.dashboard.slip_engine import REFUSED_MODIFIER, SlipScore
from sportstradamus.leg_schema import leg_label


def render_pair_note(score: SlipScore, legs: Sequence[Mapping], platform: str) -> None:
    """Name the pairs ``platform`` refuses or, on a slip it will take, the pairs it reprices.

    A same-player refusal gets no warning here, since ``validate_parlay_legs`` already
    says two legs share a player. A refused slip prices at $0, so it gets no reprice
    caption either.
    """
    refused = [
        f"{leg_label(legs[i])} with {leg_label(legs[j])}"
        for i, j, modifier in score.pair_mods
        if modifier == REFUSED_MODIFIER and legs[i]["player"] != legs[j]["player"]
    ]
    if refused:
        st.warning(
            f"{platform} won't pair {'; '.join(refused)}. Remove one to lock it in.",
            icon=":material/block:",
        )
    if score.banned or not score.pair_mods:
        return
    repriced = [
        f"{leg_label(legs[i])} with {leg_label(legs[j])} ×{modifier:g}"
        for i, j, modifier in score.pair_mods
    ]
    pronoun = "it" if len(repriced) == 1 else "them"
    st.caption(f"{platform} reprices {'; '.join(repriced)}; the payout includes {pronoun}.")
