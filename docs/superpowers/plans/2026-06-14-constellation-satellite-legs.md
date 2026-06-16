# Constellation Satellite Legs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a slip pull edge legs from other games into a separate "satellite" section so a single-team game can still form a valid parlay, and show captions on candidate stars.

**Architecture:** The constellation builder gains a *focus game* (the oldest leg's game) that the star map anchors on, unchanged and same-game. Any slip leg from another game is a *satellite*, added/removed through a new grouped-by-game picker section under the map. A new `satellite_picker.py` holds the pure candidate query plus a thin render that returns an add/remove action; `slip_builder.py` applies it. Validity, scoring, and lock-in already operate over the whole slip and are untouched.

**Tech Stack:** Python 3.11, pandas, Streamlit, plotly (existing dashboard stack). Tests are pytest golden tests over pure functions (Streamlit render paths are verified manually, per the lane precedent — no AppTest pattern in-tree).

---

## Spec

`docs/superpowers/specs/2026-06-14-constellation-satellite-legs-design.md` (gitignored per repo convention; on disk).

## Repo-state preconditions & commit policy

- The `feature/dashboard-ux` working tree already carries **uncommitted P4.5 work** in `components/slip_builder.py`, `components/constellation.py`, and several untracked files. This is the lane's normal mid-round state.
- **Commit policy for this plan:** the lane batches commits at milestones, not per task. Because `constellation.py` and `slip_builder.py` are already dirty with P4.5, do **not** add per-task commits that would entangle that work. Each task ends at a green test / clean gate; the owner commits the dashboard-ux round (P4.5 + satellites) as one milestone when ready. If a commit is wanted, stage explicit paths — never `git add -A`.
- Do **not** push (the lane branch is unpushed by design).

## File structure

| Path | Responsibility |
|---|---|
| `src/sportstradamus/dashboard/components/satellite_picker.py` | **New.** Pure `satellite_groups` (other-game edge legs grouped by game) + `render_satellites` (the expander UI; returns an add/remove action). No Archive, no parquet. |
| `src/sportstradamus/dashboard/components/constellation.py` | Promote `_star_label` → public `star_label` (reused by the picker chips); candidate stars carry captions. |
| `src/sportstradamus/dashboard/components/slip_builder.py` | Focus-game generalization (`_game_pool` → `_focus_pool`), focus/satellite leg-list split, call `render_satellites` + apply its action, headline over focus legs. |
| `tests/golden/test_satellite_picker.py` | **New.** Pins `satellite_groups` filtering / exclusion / cap / ranking. |
| `tests/golden/test_constellation.py` | Update the candidate-label pin + module docstring. |

---

## Task 1: `satellite_groups` pure query

**Files:**
- Create: `src/sportstradamus/dashboard/components/satellite_picker.py`
- Test: `tests/golden/test_satellite_picker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/golden/test_satellite_picker.py`:

```python
"""Satellite picker — the pure other-game candidate query.

Render paths (the expander, chips, auto-open) are Streamlit-runtime and verified
manually; only the grouping query is unit-pinned here.
"""

import pandas as pd

from sportstradamus.dashboard.components.satellite_picker import (
    _PER_GAME_CAP,
    satellite_groups,
)
from sportstradamus.dashboard.legs import corr_key


def _offer(player, market, bet, line, game, team, platform, k):
    return {
        "Player": player, "Market": market, "Bet": bet, "Line": line,
        "Game": game, "Team": team, "Platform": platform, "K": k,
        "League": "NBA", "Date": "2026-06-14", "Model P": 0.6, "Boost": 2.0,
    }


def test_keeps_only_other_game_positive_edge_on_platform():
    offers = pd.DataFrame([
        _offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.30),   # focus game -> out
        _offer("B", "AST", "Over", 6.5, "BOS/MIA", "BOS", "Underdog", 0.20),    # in
        _offer("C", "REB", "Over", 10.5, "BOS/MIA", "MIA", "Underdog", 0.0),    # zero edge -> out
        _offer("D", "PTS", "Over", 30.5, "DEN/LAL", "DEN", "Sleeper", 0.40),    # off platform -> out
    ])
    groups = satellite_groups(offers, focus_game="NYK/SAS", platform="Underdog", exclude_keys=set())
    assert [g[0] for g in groups] == ["BOS/MIA"]
    assert [r["Player"] for r in groups[0][1]] == ["B"]


def test_drops_already_slipped_keys():
    offers = pd.DataFrame([
        _offer("B", "AST", "Over", 6.5, "BOS/MIA", "BOS", "Underdog", 0.20),
        _offer("E", "PTS", "Over", 22.5, "BOS/MIA", "MIA", "Underdog", 0.10),
    ])
    exclude = {corr_key({"Player": "B", "Market": "AST", "Bet": "Over"})}
    groups = satellite_groups(offers, focus_game="NYK/SAS", platform="Underdog", exclude_keys=exclude)
    assert [r["Player"] for r in groups[0][1]] == ["E"]


def test_caps_per_game_and_ranks_games_by_best_edge():
    rows = [
        _offer(f"P{i}", "PTS", "Over", 20 + i, "BOS/MIA", "BOS", "Underdog", 0.10 + i * 0.001)
        for i in range(_PER_GAME_CAP + 3)
    ]
    rows.append(_offer("Top", "PTS", "Over", 40.5, "DEN/LAL", "DEN", "Underdog", 0.50))
    groups = satellite_groups(pd.DataFrame(rows), focus_game="NYK/SAS", platform="Underdog", exclude_keys=set())
    assert groups[0][0] == "DEN/LAL"                       # strongest single leg leads
    bos = dict(groups)["BOS/MIA"]
    assert len(bos) == _PER_GAME_CAP                       # capped per game
    assert [r["K"] for r in bos] == sorted((r["K"] for r in bos), reverse=True)  # sorted by edge


def test_empty_when_no_other_game_qualifies():
    offers = pd.DataFrame([_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.30)])
    assert satellite_groups(offers, focus_game="NYK/SAS", platform="Underdog", exclude_keys=set()) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/golden/test_satellite_picker.py -n0 -q`
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'satellite_groups'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/sportstradamus/dashboard/components/satellite_picker.py` with the module docstring and the pure query (the render function lands in Task 3):

```python
"""Satellite legs — edge legs from other games, to complete a single-team slip.

The constellation builder is same-game (DESIGN §4a): a game whose model-liked legs
(Kelly ``K`` > 0) sit on only one of its two teams can't form a valid parlay on its
own (``validate_parlay_legs`` needs two distinct teams). This picker offers the game's
complement — ``K`` > 0 legs from *other* games on the slip's platform, grouped by
game — so a user can add one validating leg or a whole second cluster. The
constellation stays untouched; these ride along as satellites.

Pure query plus a thin Streamlit render that returns an add/remove action for the
builder to apply (the builder owns the slip state). No Archive, no parquet — it slices
the in-memory ``current_offers`` the builder already holds.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.legs import corr_key

# Top legs offered per other game — enough to grab a small second cluster, few enough
# to keep the section uncluttered.
_PER_GAME_CAP = 6


def satellite_groups(
    offers: pd.DataFrame,
    *,
    focus_game: str,
    platform: str,
    exclude_keys: set[str],
) -> list[tuple[str, list[dict]]]:
    """Other-game edge legs to offer, grouped by game, games ranked by best edge.

    Filters ``offers`` to model-liked legs (``K`` > 0) on ``platform`` in games other
    than ``focus_game``, drops anything already in the slip (``exclude_keys`` of
    ``corr_key``), keeps each game's top ``_PER_GAME_CAP`` by ``K``, and orders the
    games by their single strongest leg so the best options lead.
    """
    if offers.empty:
        return []
    pool = offers[
        (offers["Platform"] == platform)
        & (offers["Game"] != focus_game)
        & (offers["K"] > 0)
    ]
    groups: list[tuple[str, list[dict]]] = []
    for game, block in pool.groupby("Game", sort=False):
        rows = [
            row
            for row in block.sort_values("K", ascending=False).to_dict("records")
            if corr_key(row) not in exclude_keys
        ]
        if rows:
            groups.append((str(game), rows[:_PER_GAME_CAP]))
    groups.sort(key=lambda g: float(g[1][0]["K"]), reverse=True)
    return groups
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `poetry run pytest tests/golden/test_satellite_picker.py -n0 -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint the new module**

Run: `poetry run ruff check src/sportstradamus/dashboard/components/satellite_picker.py`
Expected: clean (no findings).

---

## Task 2: Candidate-star labels

**Files:**
- Modify: `src/sportstradamus/dashboard/components/constellation.py` (`_star_label` → `star_label`; `_node_info`; `_add_node_trace`)
- Test: `tests/golden/test_constellation.py:115-120` (the candidate-label pin) + module docstring `:6-7`

- [ ] **Step 1: Update the failing test to the new behavior**

In `tests/golden/test_constellation.py`, replace `test_only_active_stars_carry_labels` (lines 115-120) with:

```python
def test_active_and_candidate_stars_both_carry_labels():
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2))
    fig = constellation_figure(legs, _corr(("A|PTS|Over", "B|REB|Under", 0.3)), pool)
    assert list(_trace(fig, "active").text) == ["A PTS o10.5"]
    assert list(_trace(fig, "candidate").text) == ["B REB u10.5"]  # candidates now labelled too
```

Also fix the stale module-docstring line (lines 6-7) — change "a candidate renders desaturated + dim + label-on-hover" to "a candidate renders desaturated + dim, labelled like the rest".

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest "tests/golden/test_constellation.py::test_active_and_candidate_stars_both_carry_labels" -n0 -q`
Expected: FAIL — candidate `text` is currently `None`, so `list(None)` raises `TypeError`.

- [ ] **Step 3: Promote `_star_label` to public**

In `src/sportstradamus/dashboard/components/constellation.py`, rename the function and its one internal caller (no forwarder — a pure pass-through is banned):

- Line 77: `def _star_label(leg: Mapping) -> str:` → `def star_label(leg: Mapping) -> str:`
- Line 106 (inside `_node_info`): `"label": _star_label(leg),` → `"label": star_label(leg),`

- [ ] **Step 4: Give candidate stars their captions**

In `_add_node_trace` (lines 354-372), the trace currently switches text off for candidates. Change these three fields so both traces are labelled:

```python
        go.Scatter(
            x=[pos[k][0] for k in keys],
            y=[pos[k][1] for k in keys],
            mode="markers+text",
            name="active" if active else "candidate",
            marker={
                "symbol": "star",
                "size": [sizes[k] for k in keys],
                "color": colors,
                "opacity": 1.0 if active else _INACTIVE_ALPHA,
            },
            text=[info[k]["label"] for k in keys],
            textposition="top center",
            textfont={"color": GRAY, "size": _LABEL_FONT_SIZE},
            customdata=[[k, *info[k]["card"]] for k in keys],
            hovertext=[info[k]["hover"] for k in keys],
            hoverinfo="none",  # the component draws the hover card; suppress the native tooltip
        )
```

(i.e. `mode` is always `"markers+text"`; `text` is always the labels — drop both `if active else …` ternaries.) Then update the function's own docstring (lines 345-349): the current "candidates are bare stars (label on hover)" is now false — change it to note both active and candidate stars carry their caption, active rendering on top.

- [ ] **Step 5: Run the constellation goldens**

Run: `poetry run pytest tests/golden/test_constellation.py -n0 -q`
Expected: PASS (all node/edge/customdata pins green, including the rewritten candidate-label test).

- [ ] **Step 6: Lint**

Run: `poetry run ruff check src/sportstradamus/dashboard/components/constellation.py`
Expected: clean.

---

## Task 3: `render_satellites` + builder wiring

This task is the Streamlit render path; the lane verifies render paths manually, so it has no unit test. The pure query it depends on is already pinned (Task 1). Verification is `ruff` + the manual walkthrough in Task 4.

**Files:**
- Modify: `src/sportstradamus/dashboard/components/satellite_picker.py` (add `render_satellites` + `_render_added`)
- Modify: `src/sportstradamus/dashboard/components/slip_builder.py` (`_game_pool` → `_focus_pool`; `render_constellation_builder`; `_render_leg_list`; new `_apply_satellite_action`; import)

- [ ] **Step 1: Add the render function to `satellite_picker.py`**

Append to `src/sportstradamus/dashboard/components/satellite_picker.py`, and extend the imports at the top:

```python
from collections.abc import Mapping, Sequence

import streamlit as st

from sportstradamus.dashboard.components.constellation import star_label
from sportstradamus.prediction.stories.legs import validate_parlay_legs
```

(Keep the existing `import pandas as pd` and `from sportstradamus.dashboard.legs import corr_key`.)

```python
def render_satellites(
    offers: pd.DataFrame,
    *,
    focus_game: str,
    platform: str,
    legs: Sequence[Mapping],
    key_prefix: str,
) -> dict | None:
    """Render the 'add a leg from another game' section; return an add/remove action.

    ``{"add": offer_row}`` when a candidate chip is clicked, ``{"remove": leg_index}``
    when an added satellite's remove is clicked, else ``None``. The builder applies the
    action — it owns the slip state. The expander auto-opens while the slip can't yet be
    locked in (single-team / one leg) and stays collapsed once it is valid.
    """
    satellites = [(i, leg) for i, leg in enumerate(legs) if leg["Game"] != focus_game]
    exclude = {corr_key(leg) for leg in legs}
    groups = satellite_groups(offers, focus_game=focus_game, platform=platform, exclude_keys=exclude)
    with st.expander("Add a leg from another game", expanded=not validate_parlay_legs(legs)[0]):
        action = _render_added(satellites, key_prefix)
        if not groups:
            st.caption(f"No other-game edge legs on {platform}.")
            return action
        for game, rows in groups:
            st.caption(game)
            cols = st.columns(3)
            for j, row in enumerate(rows):
                if cols[j % 3].button(
                    f":material/add: {star_label(row)}",
                    key=f"{key_prefix}_sat_add_{corr_key(row)}",
                ):
                    action = {"add": row}
    return action


def _render_added(satellites: list[tuple[int, Mapping]], key_prefix: str) -> dict | None:
    """List the slip's current satellites with a remove control; return a remove action."""
    if not satellites:
        return None
    st.caption("From other games")
    action: dict | None = None
    for i, leg in satellites:
        text_col, rm_col = st.columns([8, 1])
        text_col.write(leg["Desc"])
        if rm_col.button(":material/close:", key=f"{key_prefix}_sat_rm_{i}", help="Remove leg"):
            action = {"remove": i}
    return action
```

- [ ] **Step 2: Lint the picker**

Run: `poetry run ruff check src/sportstradamus/dashboard/components/satellite_picker.py`
Expected: clean.

- [ ] **Step 3: Generalize `_game_pool` → `_focus_pool` in `slip_builder.py`**

Replace `_game_pool` (lines 369-380) with:

```python
def _focus_pool(legs: Sequence[Mapping], offers: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """The slip's focus game (the oldest leg's game) and its candidate offers on the platform.

    The constellation anchors on this one game; legs from other games are satellites,
    rendered outside the map. ``legs`` is non-empty (the caller guards).
    """
    focus = legs[0]["Game"]
    platform = st.session_state[_PLATFORM]
    return focus, offers.loc[(offers["Game"] == focus) & (offers["Platform"] == platform)]
```

- [ ] **Step 4: Add the import and the action applier to `slip_builder.py`**

Add the import next to the other component imports (after line 31, the `render_constellation` import):

```python
from sportstradamus.dashboard.components.satellite_picker import render_satellites
```

Add this helper (next to `_apply_constellation_action`):

```python
def _apply_satellite_action(action: Mapping | None, legs: list[dict]) -> bool:
    """Apply the satellite picker's add/remove to the slip; return True if it changed."""
    if not action:
        return False
    if "add" in action:
        legs.append(_leg_from_offer(action["add"]))
    else:
        remove_leg(action["remove"])
    return True
```

- [ ] **Step 5: Split `_render_leg_list` by focus game**

Replace `_render_leg_list` (lines 276-290) with a version that can show only the focus game's legs (satellites are listed in the picker section instead). The enumerate index stays the true `legs` index so removal is correct:

```python
def _render_leg_list(key_prefix: str, *, focus_game: str | None = None, removable: bool = True) -> None:
    """List slip legs. ``focus_game`` shows only that game's legs (satellites are listed
    in the picker); ``removable=False`` drops the button column because a leg is removed
    by clicking its star.
    """
    legs = st.session_state[_LEGS]
    for i, leg in enumerate(legs):
        if focus_game is not None and leg["Game"] != focus_game:
            continue
        line = f"{leg['Desc']}  ·  {leg['League']}"
        if not removable:
            st.write(line)
            continue
        text_col, rm_col = st.columns([8, 1])
        text_col.write(line)
        if rm_col.button(":material/close:", key=f"{key_prefix}_rm_{i}", help="Remove leg"):
            remove_leg(i)
            st.rerun()
```

- [ ] **Step 6: Wire the focus model + picker into `render_constellation_builder`**

Replace the body of `render_constellation_builder` (lines 226-246, from `legs = …` through the lock-in call) with:

```python
    legs = st.session_state[_LEGS]
    if not legs:
        st.info("Pick a story above to start a slip, or load one from the rail.")
        return
    focus, pool = _focus_pool(legs, offers)
    focus_legs = [leg for leg in legs if leg["Game"] == focus]
    _render_constellation(focus_legs, corr, pool, key_prefix)
    _render_leg_list(key_prefix, focus_game=focus, removable=False)
    act = render_satellites(
        offers,
        focus_game=focus,
        platform=st.session_state[_PLATFORM],
        legs=legs,
        key_prefix=key_prefix,
    )
    if _apply_satellite_action(act, legs):
        st.rerun()
    valid, reason = validate_parlay_legs(legs)
    if len(legs) < 2:
        st.caption(reason or "Tap a gray star to add a leg.")
        return
    if not valid:
        st.warning(reason)
    shrink = _slip_shrinkage(legs)
    score = _score(legs, corr, shrink)
    headline = slip_headline(focus_legs, offers, ctxs)
    if headline:
        st.markdown(f"#### {headline}")
    _render_metrics(score, correlated=True)
    st.caption("Pairing-block risk arrives with the correlation-block model.")
    _render_lock_in(score, headline, shrink, key_prefix, can_lock=valid)
```

The two changes from today: `_render_constellation` and `slip_headline` now take `focus_legs` (so satellites don't become stray stars or muddy the thesis), and the `render_satellites` call + `_apply_satellite_action` sit between the leg list and the validity branch (so the picker renders even in the `< 2 legs` single-team case).

- [ ] **Step 7: Lint both modified modules**

Run: `poetry run ruff check src/sportstradamus/dashboard/components/slip_builder.py src/sportstradamus/dashboard/components/satellite_picker.py`
Expected: clean.

- [ ] **Step 8: Import-safety + no-archive-lock check**

Run: `poetry run pytest tests/golden/test_dashboard_no_archive_lock.py -n0 -q`
Expected: PASS — the new `satellite_picker` module is auto-discovered and carries no module-level `Archive()`.

---

## Task 4: Gates, refactoring-specialist, docs & memory

**Files:**
- `DESIGN.md` (§4a — one sentence on satellites living outside the map)
- `docs/handoffs/dashboard-ux.md` (§10 ledger line + status bump)
- memory: `project_dashboard_ux_lane.md` + `MEMORY.md` index line

- [ ] **Step 1: Run the refactoring-specialist on every touched `.py`**

Dispatch the `refactoring-specialist` subagent (CLAUDE.md mandatory pre-review gate) with exactly this scope:
- `src/sportstradamus/dashboard/components/satellite_picker.py`
- `src/sportstradamus/dashboard/components/constellation.py`
- `src/sportstradamus/dashboard/components/slip_builder.py`

Wait for its report; address anything it flags (re-invoke if it edits). Then run `git status` to confirm it touched nothing out of scope (it has edited unprompted before).

- [ ] **Step 2: Run the three authoritative gates**

```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/
poetry run pytest -m integration -n0 && touch /home/trevor/Sportstradamus/.claude/.state/integration_green
```

Expected: ruff clean; golden green except the **three documented pre-existing** `test_correlate.py:210` corr-pollution reds (`test_find_correlation_offer_correlations_real_nba`, `test_kernel_reads_real_nba_cmap`, `test_nba_get_training_matrix` — signature "real NBA c_map is empty"), which are out of this footprint; integration 14 passed.

- [ ] **Step 3: Update DESIGN.md §4a**

In the §4a paragraph, after the both-teams/empty-half sentence, add one sentence making the satellite boundary explicit, e.g.: "Legs from *other* games never appear as stars — the map is one game; cross-game legs ride along in a separate satellite section so a single-sided game can still reach two teams." Keep it within the FIXED-grammar description (the map itself is unchanged).

- [ ] **Step 4: Append the handoff ledger line + bump status**

In `docs/handoffs/dashboard-ux.md`, bump the status line (note satellites landed on top of P4.5) and add one newest-first §10 ledger entry dated 2026-06-14 summarizing: focus-game + satellite picker (grouped-by-game, auto-open), candidate-star labels; the constellation stays same-game.

- [ ] **Step 5: Update memory**

- `project_dashboard_ux_lane.md`: revise the constellation/builder section — note the focus-game model, the `satellite_picker` module (pure `satellite_groups` + grouped render returning an add/remove action), the leg-list focus/satellite split, and that candidate stars are now labelled (the old "label-on-hover only" is reversed). Convert any relative date to absolute (2026-06-14).
- `MEMORY.md`: update the Dashboard-ux lane index line with the satellite-legs + candidate-label facts.

- [ ] **Step 6: Manual walkthrough (owner-run; describe, don't run unprompted)**

On a fresh `poetry run dashboard` (imported modules don't hot-reload):
1. Seed a story on a game whose edge legs are one-sided → the constellation shows one populated half, the satellite expander is **auto-open**, and the validity caption nudges to add another team.
2. Open a game → pick a satellite chip → the slip reaches two teams and **locks in**; the candidate stars all show their `Lastname MKT o/uLine` caption now.
3. Combine two single-team clusters: seed game 1, add several of game 2's chips → both clusters in one valid slip; the map stays anchored on game 1, game 2's legs listed as removable satellites.
4. Remove a satellite via its `x` → slip updates; remove the map's legs via star clicks as before.

---

## Self-review notes

- **Spec coverage:** focus game (Task 3 §3,6) · satellite query (Task 1) · grouped render + auto-open (Task 3 §1) · leg-list split (Task 3 §5) · reuse validity/scorer/headline (Task 3 §6 + unchanged) · candidate labels (Task 2) · tests (Tasks 1–2) · edge cases ("no other game" caption Task 3 §1; empty slip unchanged) — all mapped.
- **Type consistency:** `satellite_groups(offers, *, focus_game, platform, exclude_keys) -> list[tuple[str, list[dict]]]` is defined in Task 1 and called identically in Task 3. `render_satellites(...) -> dict | None` returns `{"add": row}` / `{"remove": i}`, consumed by `_apply_satellite_action(action, legs)` in Task 3 §4. `star_label` (Task 2) is the name imported by the picker (Task 3 §1). `_focus_pool` returns `(str, DataFrame)`, unpacked as `focus, pool` (Task 3 §6).
- **No placeholders:** every code step shows complete code; commands carry expected output.
