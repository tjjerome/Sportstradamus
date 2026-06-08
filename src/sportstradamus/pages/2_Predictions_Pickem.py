"""Today's Underdog Pick'em entries from the latest `prophecize` run.

Stakes are sized live: the snapshot stores bankroll-independent fields
(joint probability, payout, EV, shrinkage) and this page sizes each entry
against a user-entered bankroll via `fractional_kelly_stake` (pure math, no
archive/network). The bankroll slider is the interactive replacement for the
offline `poetry run kelly` re-sizer.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from decimal import Decimal

import pandas as pd
import streamlit as st

from sportstradamus.dashboard_data import (
    format_ts,
    load_current_meta,
    load_current_pickem,
    render_banner,
)
from sportstradamus.strategies.kelly import fractional_kelly_stake

# Iteration order controls section order in the rendered page.
VARIANT_LABELS = {"power": "Power", "flex": "Flex", "rivals": "Rivals"}

st.set_page_config(page_title="Predictions — Pickem", layout="wide")
st.title("Today's Pick'em Entries")

meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
render_banner("predictions", f"generated {generated}")

entries = load_current_pickem()
if entries.empty:
    st.info(
        "No current Pick'em entries found. Run `poetry run prophecize` to "
        "generate `current_pickem.parquet`."
    )
    st.stop()

bankroll = st.number_input(
    "Bankroll ($)",
    min_value=0.0,
    value=1000.0,
    step=50.0,
    help=(
        "Stakes are quarter-Kelly × the per-cell shrinkage. Adjusting bankroll "
        "rescales every entry — this replaces the offline `kelly` CLI."
    ),
)


def _stake(row: pd.Series) -> float:
    return float(
        fractional_kelly_stake(
            bankroll=Decimal(str(bankroll)),
            win_prob=float(row["joint_prob"]),
            payout_multiplier=Decimal(str(row["payout_multiplier"])),
            model_shrinkage=float(row["shrinkage"]),
        )
    )


view = entries.copy()
view["Stake"] = view.apply(_stake, axis=1)
view = view.sort_values("ev", ascending=False)

st.caption(f"{len(view)} entries across {view['contest_variant'].nunique()} contest variants.")

for variant, label in VARIANT_LABELS.items():
    group = view.loc[view["contest_variant"] == variant]
    if group.empty:
        continue
    st.subheader(f"{label} — {len(group)} entries, ${group['Stake'].sum():,.2f} staked")
    st.dataframe(
        pd.DataFrame(
            {
                "Legs": group["legs"].apply("  +  ".join),
                "Size": group["entry_size"],
                "Joint P": group["joint_prob"].map("{:.1%}".format),
                "Payout": group["payout_multiplier"].map("{:.2f}x".format),
                "EV": group["ev"].map("{:+.1%}".format),
                "Shrinkage": group["shrinkage_source"],
                "Stake": group["Stake"].map("${:,.2f}".format),
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("Snapshot info"):
    st.json(meta)
