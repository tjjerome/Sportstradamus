"""Build a slip — story-seeded constellation builder, plus Pick'em entries."""

from decimal import Decimal

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.slip_builder import (
    render_constellation_builder,
    seed_from_legs,
    seed_from_story,
)
from sportstradamus.dashboard.data import (
    format_ts,
    load_current_game_context,
    load_current_game_corr,
    load_current_game_stories,
    load_current_meta,
    load_current_offers,
    load_current_pickem,
    render_banner,
    sport_filtered,
)
from sportstradamus.dashboard.legs import find_offer_idx, parse_leg
from sportstradamus.prediction.stories.context import ctxs_from_frame
from sportstradamus.strategies.kelly import fractional_kelly_stake

# Iteration order controls section order in the Pick'em tab.
VARIANT_LABELS = {"power": "Power", "flex": "Flex", "rivals": "Rivals"}
# Default bankroll if the global slip-state key hasn't been touched yet.
_DEFAULT_BANKROLL = 1000.0


def _compute_stake(row: pd.Series, bankroll: float) -> float:
    """Quarter-Kelly stake for a Pick'em entry scaled to ``bankroll``."""
    return float(
        fractional_kelly_stake(
            bankroll=Decimal(str(bankroll)),
            win_prob=float(row["joint_prob"]),
            payout_multiplier=Decimal(str(row["payout_multiplier"])),
            model_shrinkage=float(row["shrinkage"]),
        )
    )


def _render_story_cascade(stories: pd.DataFrame, offers: pd.DataFrame) -> None:
    """Platform → game → story → objective; seed the constellation builder."""
    platform = st.selectbox(
        "Platform", sorted(stories["platform"].dropna().unique()), key="cascade_platform"
    )
    sub = stories.loc[stories["platform"] == platform]
    games = sorted(sub["Game"].dropna().unique())
    if not games:
        st.info("No stories for this platform.")
        return
    game = st.selectbox("Game", games, key="cascade_game")
    gsub = sub.loc[sub["Game"] == game]
    story_ids = sorted(
        gsub["story_id"].dropna().unique(), key=lambda s: int(str(s).rsplit("#", 1)[-1])
    )

    def _story_label(sid: str) -> str:
        rows = gsub.loc[gsub["story_id"] == sid]
        return str(rows.iloc[0]["headline"]) or sid

    story_id = st.selectbox("Story", story_ids, format_func=_story_label, key="cascade_story")
    rows = gsub.loc[gsub["story_id"] == story_id]
    objective = st.radio(
        "Start from",
        ["builder", "moon"],
        format_func=lambda o: "Bankroll Builder" if o == "builder" else "Shoot the Moon",
        horizontal=True,
        key="cascade_objective",
    )
    orow = rows.loc[rows["objective"] == objective]
    if orow.empty:
        return
    o = orow.iloc[0]
    st.caption(f"{o['headline']} · {int(o['bet_size'])} legs · EV {o['model_ev'] - 1:+.1%}")
    if st.button("Seed builder", key="cascade_seed", type="primary"):
        seed_from_story(o["legs"], platform, offers)
        st.rerun()


def _render_pickem(entries: pd.DataFrame, offers: pd.DataFrame) -> None:
    bankroll = float(st.session_state.get("slip_bankroll", _DEFAULT_BANKROLL))
    view = entries.copy()
    view["Stake"] = view.apply(lambda r: _compute_stake(r, bankroll), axis=1)
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
                    "Stake": group["Stake"].map("${:,.2f}".format),
                }
            ),
            hide_index=True,
            width="stretch",
        )
    _render_pickem_loader(view, offers)


def _render_pickem_loader(entries: pd.DataFrame, offers: pd.DataFrame) -> None:
    """Load any Pick'em entry into the builder to edit as your own."""
    labels = {
        f"{VARIANT_LABELS.get(e['contest_variant'], e['contest_variant'])}: {'  +  '.join(e['legs'])}": e
        for e in entries.to_dict("records")
    }
    pick = st.selectbox("Edit a Pick'em entry as your own", ["—", *labels], key="pickem_load")
    if pick == "—" or not st.button("Load into builder", key="pickem_load_btn"):
        return
    rows = []
    for leg in labels[pick]["legs"]:
        parsed = parse_leg(leg)
        idx = find_offer_idx(parsed, offers, "Underdog") if parsed else None
        if idx is not None:
            rows.append(offers.loc[idx].to_dict())
    if len(rows) < 2:
        st.toast("Couldn't resolve this entry's legs against current offers.")
        return
    same_game = len({r["Game"] for r in rows}) == 1
    seed_from_legs(rows, "Underdog", "constellation" if same_game else "simple")
    if same_game:
        st.rerun()
    else:
        st.switch_page("surfaces/board.py")


tab_build, tab_pickem = st.tabs(["Build a slip", "Pick'em"])

with tab_build:
    st.title("Build a slip")
    meta = load_current_meta()
    render_banner(
        "predictions", f"generated {format_ts(meta.get('generated_at', 'no run on record'))}"
    )

    offers = load_current_offers().reset_index(drop=True)
    corr = load_current_game_corr()
    ctxs = ctxs_from_frame(load_current_game_context(), corr)
    stories = sport_filtered(load_current_game_stories())

    editing = (
        st.session_state.get("edit_slip_id") is not None
        and st.session_state.get("slip_builder") == "constellation"
    )
    if not editing:
        if stories.empty:
            st.info("No story menu found. Run `poetry run prophecize` to generate stories.")
        else:
            _render_story_cascade(stories, offers)

    st.divider()
    render_constellation_builder(offers, corr, ctxs)

with tab_pickem:
    st.title("Pick'em entries")
    meta = load_current_meta()
    render_banner(
        "predictions", f"generated {format_ts(meta.get('generated_at', 'no run on record'))}"
    )

    entries = sport_filtered(load_current_pickem())
    if entries.empty:
        st.info("No current Pick'em entries. Run `poetry run prophecize` to generate them.")
    else:
        _render_pickem(entries, load_current_offers().reset_index(drop=True))
