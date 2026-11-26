"""Lab EV Fit — the fit behind the board's EV, and every play's record.

The nightly job fits EV's win chance per league, pick kind and class of sportsbook coverage
(``live_fit.fit``) and grades each play's settled legs over its trailing window
(``plays.play_record``). This page shows both tables as the job wrote them.
"""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.hero import desk_only_notice, page_hero
from sportstradamus.dashboard.data import load_live_fit, load_play_record, sport_filtered

# The fit's coefficients (``live_fit.TERMS``), named for what each one weighs.
_TERMS = {
    "c": "Intercept",
    "a1": "App price",
    "w_book": "Books",
    "w_model": "Model",
    "w_fb": "Book stand-in",
}


def _three_places(frame: pd.DataFrame) -> dict:
    return {
        col: st.column_config.NumberColumn(format="%.3f")
        for col in frame.columns
        if pd.api.types.is_float_dtype(frame[col])
    }


page_hero("MODEL LAB · EV FIT", "EV Fit & Play Record")
desk_only_notice()

st.subheader("Fit weights")
st.caption(
    "One fit per league, pick kind and class of book coverage (A: a book prices the line; "
    "B: books price only other lines; C: no book quotes the player). Each weight is zero where "
    "EV's win chance is the app's own price, and blank where its fit has no such term."
)
fit = sport_filtered(load_live_fit())
if fit.empty:
    st.info("No fit yet: the nightly job writes it.")
else:
    names = {"league": "League", "kind": "Pick kind", "book_class": "Class", "n_legs": "Legs"}
    for term, name in _TERMS.items():
        names |= {term: name, f"se_{term}": f"{name} SE"}
    weights = fit[list(names)].rename(columns=names)
    st.dataframe(weights, hide_index=True, width="stretch", column_config=_three_places(weights))

st.subheader("Play record")
st.caption(
    "Each play's settled legs by league and pick kind over the trailing window, every leg at "
    "its own payout: ev_pick is the board's picks, model_edge the old 5% edge rule. roi is "
    "the return per $1 a leg, roi_lo to roi_hi its 95% range."
)
record = sport_filtered(load_play_record())
if record.empty:
    st.info("No play record yet: the nightly job writes it.")
else:
    st.dataframe(record, hide_index=True, width="stretch", column_config=_three_places(record))
