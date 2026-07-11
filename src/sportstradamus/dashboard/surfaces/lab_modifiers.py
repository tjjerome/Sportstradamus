"""Lab Modifiers — expected-vs-actual quote reconciler over dashboard slips.

Runs the ``calibrate_parlay_modifiers`` pricing on the slip already built in
the session (the rail) or a locked slip, plus optional typed legs (game lines
are not board offers yet). The operator enters the app's actual quoted
payout; the solved pair modifier or parlay rake lands in
``data/runtime/modifier_overrides.json`` — never the committed configs, so
the production checkout stays clean for cron pulls. ``helpers.config`` merges
the overlay at load; fold it into ``banned_combos.json`` on a dev box with
``calibrate_parlay_modifiers --fold-overlay``.
"""

import json
import math
from importlib import resources

import streamlit as st

from sportstradamus import data
from sportstradamus.dashboard.data import load_current_offers, render_banner
from sportstradamus.dashboard.legs import find_offer_idx
from sportstradamus.helpers.config import apply_modifier_overrides
from sportstradamus.helpers.io import (
    read_modifier_overrides,
    read_user_slips,
    write_modifier_overrides,
)
from sportstradamus.leg_schema import leg_label
from sportstradamus.scripts.calibrate_parlay_modifiers import (
    MODIFIER_DECIMALS,
    Leg,
    parse_slip,
    reconcile,
    slip_pairs,
)

_CONFIG_DIR = resources.files(data) / "config"


def _effective_tables(platform: str) -> tuple[dict, dict]:
    """Committed configs with the runtime overlay merged, string-keyed.

    ``helpers.config.banned`` is frozenset-keyed for parlay scoring; the
    reconciler needs the on-disk string keys, so it re-reads the json.
    """
    combos = json.loads((_CONFIG_DIR / "banned_combos.json").read_text())
    overlay = read_modifier_overrides()
    apply_modifier_overrides(combos, overlay.get("modifiers", {}))
    rake_file = _CONFIG_DIR / "parlay_rake.json"
    rakes = json.loads(rake_file.read_text()).get(platform, {}) if rake_file.is_file() else {}
    rakes.update(overlay.get("rake", {}).get(platform, {}))
    return combos.get(platform, {}), rakes


def _pos_prefix(league: str, position) -> str:
    """Banned-combos key prefix from a snapshot Position value."""
    text = "" if position is None else str(position)
    if text.lower() == "nan":
        text = ""
    if league == "MLB":
        return "P" if text.upper() in {"P", "SP", "RP"} else "B"
    return text


def _default_key(leg: dict, offers, platform: str) -> str:
    idx = find_offer_idx(leg, offers, platform)
    position = offers.at[idx, "Position"] if idx is not None else None
    prefix = _pos_prefix(leg["league"], position)
    return f"{prefix}.{leg['market']}" if prefix else str(leg["market"])


st.title("Modifier Reconciler")
render_banner("stats", "expected vs actual quote — solves platform correlation modifiers")

slips = read_user_slips()
source_options = ["Current slip"] + ([] if slips.empty else ["Locked slip"])
source = st.radio("Slip source", source_options, horizontal=True)

if source == "Current slip":
    raw_legs = list(st.session_state.get("slip_legs", []))
    platform = st.session_state.get("slip_platform", "Underdog")
    source_id = "rail"
else:
    slips = slips.sort_values("saved_at", ascending=False)
    labels = {
        f"{row.headline} ({str(row.saved_at)[:10]})": row.slip_id for row in slips.itertuples()
    }
    picked = st.selectbox("Locked slip", list(labels))
    slip_row = slips.loc[slips["slip_id"] == labels[picked]].iloc[0]
    raw_legs = [dict(leg) for leg in slip_row["legs"]]
    platform = str(slip_row["platform"])
    source_id = str(slip_row["slip_id"])[:8]

offers = load_current_offers()
combos_platform, rakes = _effective_tables(platform)
leagues = sorted(combos_platform) or ["MLB"]
league_default = raw_legs[0]["league"] if raw_legs else "MLB"
league = st.selectbox(
    "League (same-game pairs must share it)",
    leagues,
    index=leagues.index(league_default) if league_default in leagues else 0,
)

legs_input: list[Leg] = []
if raw_legs:
    st.caption("Leg | pair key (edit to match banned_combos convention) | bet | app multiplier")
for i, leg in enumerate(raw_legs):
    c_label, c_key, c_bet, c_mult = st.columns([3, 3, 2, 2])
    c_label.markdown(leg_label(leg))
    key_val = c_key.text_input(
        "pair key",
        value=_default_key(leg, offers, platform),
        key=f"mod_key_{source_id}_{i}",
        label_visibility="collapsed",
    )
    bet_val = c_bet.selectbox(
        "bet",
        ["Over", "Under"],
        index=0 if str(leg["bet"]) in {"Over", "Higher"} else 1,
        key=f"mod_bet_{source_id}_{i}",
        label_visibility="collapsed",
    )
    mult_val = c_mult.number_input(
        "multiplier",
        min_value=1.0,
        value=float(leg.get("boost") or 1.0),
        step=0.01,
        key=f"mod_mult_{source_id}_{i}",
        label_visibility="collapsed",
    )
    legs_input.append(
        Leg(str(leg["game"]), str(leg["team"]), key_val, bet_val == "Over", float(mult_val))
    )

extra_spec = st.text_input(
    "Extra legs — game:team:POS.market:h|l:multiplier (game lines / prediction picks)",
    key=f"mod_extra_{source_id}",
    placeholder='g1:SD:TEAM.moneyline:h:1.79 "g1:SD:P.pitcher strikeouts:h:1.62"',
)
if extra_spec:
    try:
        legs_input.extend(parse_slip(extra_spec, min_legs=1))
    except ValueError as exc:
        st.error(str(exc))

if len(legs_input) < 2:
    st.info("Add at least two legs — build a slip on the Board or type extra legs above.")
    st.stop()

league_mods = combos_platform.get(league, {"team": {}, "opponent": {}})
pairs = slip_pairs(legs_input, league_mods)
rake = rakes.get(str(len(legs_input)))
product = math.prod(leg.mult for leg in legs_input)
known = math.prod(p.value for p in pairs if p.value is not None)
expected = (rake if rake is not None else 1.0) * product * known

for p in pairs:
    slot_name = "same-dir" if p.slot == 0 else "opp-dir"
    shown = "UNKNOWN" if p.value is None else f"{p.value}"
    st.caption(f"pair [{p.relation}/{slot_name}] {p.display_key}: {shown}")
rake_note = (
    f"rake[{len(legs_input)}]={rake}"
    if rake is not None
    else f"rake[{len(legs_input)}] uncalibrated"
)
st.metric(
    "Expected quote",
    f"{expected:.3f}x",
    help=f"product {product:.3f} x {rake_note} x known {known:.3f}",
)

actual = st.number_input("Actual quoted payout (x)", min_value=0.0, value=0.0, step=0.01)
if st.button("Solve & save to overlay", type="primary", disabled=actual <= 0):
    kind, target, value = reconcile(legs_input, pairs, rake, actual, expected)
    if kind == "match":
        st.info("Matches expected — nothing to learn.")
    elif kind == "uncalibrated":
        st.warning(f"rake[{target}] unknown — reconcile an all-cross-game {target}-leg slip first.")
    elif kind == "ambiguous":
        st.warning(f"Cannot attribute the residual across multiple pair groups: {target}")
    elif kind == "rake":
        value = round(value, MODIFIER_DECIMALS)
        overlay = read_modifier_overrides()
        overlay.setdefault("rake", {}).setdefault(platform, {})[str(target)] = value
        write_modifier_overrides(overlay)
        st.success(f"Saved rake[{target}] = {value} for {platform} to the overlay.")
    else:
        relation, key_str, slot = target
        value = round(value, MODIFIER_DECIMALS)
        entry = list(league_mods.get(relation, {}).get(key_str, [1.0, 1.0]))
        entry[slot] = value
        overlay = read_modifier_overrides()
        overlay.setdefault("modifiers", {}).setdefault(platform, {}).setdefault(
            league, {}
        ).setdefault(relation, {})[key_str] = entry
        write_modifier_overrides(overlay)
        slot_name = "same-dir" if slot == 0 else "opp-dir"
        st.success(f"Saved [{relation}/{slot_name}] {key_str} = {value} to the overlay.")

overlay_now = read_modifier_overrides()
n_pending = sum(
    len(entries)
    for leagues_d in overlay_now.get("modifiers", {}).values()
    for relations in leagues_d.values()
    for entries in relations.values()
) + sum(len(t) for t in overlay_now.get("rake", {}).values())
if n_pending:
    st.caption(
        f"{n_pending} override(s) pending fold-in — run "
        "`calibrate_parlay_modifiers --fold-overlay` on the dev box to commit them."
    )
