# Dashboard Mobile (Phase M) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dashboard's core money loop (Tonight → Games/constellation → slip build → stakes) work well on a phone, with the Board readable, per the approved spec `docs/archive/superpowers/specs/2026-07-16-dashboard-mobile-design.md`.

**Architecture:** A User-Agent branch (`viewport.is_mobile()`, session-cached, test/query-param overridable) gates render-level differences — top nav, a fixed bottom slip dock, a Board card list, and a touch-mode constellation — while one `@media (max-width: 767px)` block in `theme.APP_CSS` handles style-only reshaping. Desktop paths stay byte-identical.

**Tech Stack:** Streamlit 1.58 (`st.context.headers`, `st.navigation(position=)`, AppTest), Plotly + hand-authored ES6 custom component (no build step), pytest golden suite.

**Branch:** `feature/dashboard-ux` (continue in place — Phase M of the dashboard-ux lane).

**Binding project rules (read before any task):**
- CLAUDE.md §Hard rules: dashboard never touches DuckDB; no file >~300 lines; no back-compat shims; comments explain *why* only.
- DESIGN.md: FIXED tokens are inviolable. The dock/cards introduce **no new colors, fonts, or radii** — only token values restated where an injected `<style>` needs literals (same convention as the component's `index.html`).
- Every task that changes a `.py` runs `poetry run ruff check src/sportstradamus/` before its commit.
- UI tasks end with a **recorded live-browser verdict** (the Phase R process rule). Run `poetry run dashboard`, open Chrome devtools device mode (iPhone 12/13, 390×844 — device mode sends a mobile UA, which exercises the real branch), and note pass/fail in the commit message body.
- **MANDATORY before any push or "done":** dispatch the `refactoring-specialist` subagent on every `.py` touched this session (list the files explicitly), then run all three gates: `poetry run ruff check src/sportstradamus/`, `poetry run pytest tests/golden/`, `poetry run pytest -m integration -n0`.

**File map (whole phase):**

| File | Action | Owns |
|---|---|---|
| `src/sportstradamus/dashboard/viewport.py` | Create | `is_mobile()` detection |
| `tests/golden/test_viewport.py` | Create | detection precedence tests |
| `src/sportstradamus/dashboard/theme.py` | Modify | `MOBILE_MAX_PX`, mobile media block |
| `tests/golden/test_design_tokens.py` | Modify | pin media block + sentinel hygiene |
| `src/sportstradamus/dashboard/components/slip_state.py` | Modify | `bankroll_input(key=)` |
| `src/sportstradamus/dashboard/components/slip_builder.py` | Modify | `slip_shrinkage` made public; mobile figure/component pass-through |
| `src/sportstradamus/dashboard/components/slip_dock.py` | Create | bottom bar + sheet |
| `tests/golden/test_slip_dock.py` | Create | dock AppTest |
| `src/sportstradamus/dashboard/app.py` | Modify | nav position, dock mount |
| `tests/golden/test_dashboard_render_smoke.py` | Modify | mobile boot smoke |
| `src/sportstradamus/dashboard/components/offer_cards.py` | Create | Board mobile card list |
| `src/sportstradamus/dashboard/surfaces/board.py` | Modify | filters expander + grid/cards branch |
| `tests/golden/test_offer_cards.py` | Create | card list AppTest |
| `src/sportstradamus/dashboard/components/constellation.py` | Modify | `mobile=` figure param |
| `tests/golden/test_constellation.py` | Modify | mobile-figure units + desktop pin |
| `src/sportstradamus/dashboard/components/constellation_component/__init__.py` | Modify | `mobile` kwarg |
| `src/sportstradamus/dashboard/components/constellation_component/build/main.js` | Modify | touch flow |
| `src/sportstradamus/dashboard/components/constellation_component/build/index.html` | Modify | docked-card CSS |
| `DESIGN.md`, `docs/handoffs/dashboard-ux.md` | Modify | §Mobile note; Phase M stage entry + ledger |

---

### Task 1: `viewport.py` — mobile detection

**Files:**
- Create: `src/sportstradamus/dashboard/viewport.py`
- Test: `tests/golden/test_viewport.py`

- [x] **Step 1: Write the failing tests**

Create `tests/golden/test_viewport.py`:

```python
"""Pin viewport.is_mobile()'s resolution order: force-key > query param > UA sniff.

Runs each check through AppTest (st.session_state / st.context need a script
runtime). AppTest requests carry no User-Agent header, so the UA fallback
resolves False — which is also why every existing desktop-path test stays on
the desktop branch untouched.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

_SCRIPT = """
import streamlit as st
from sportstradamus.dashboard.viewport import is_mobile

st.write(f"mobile={is_mobile()}")
"""


def _run(force: bool | None = None, query_m: str | None = None) -> str:
    at = AppTest.from_string(_SCRIPT, default_timeout=10)
    if force is not None:
        at.session_state["_force_mobile"] = force
    if query_m is not None:
        at.query_params["m"] = query_m
    at.run()
    assert not at.exception
    return at.markdown[0].value


def test_defaults_to_desktop_without_headers():
    assert _run() == "mobile=False"


def test_force_key_wins():
    assert _run(force=True) == "mobile=True"


def test_query_param_overrides_ua():
    assert _run(query_m="1") == "mobile=True"
    assert _run(query_m="0") == "mobile=False"


def test_force_key_beats_query_param():
    assert _run(force=False, query_m="1") == "mobile=False"
```

- [x] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/golden/test_viewport.py -n0 -v`
Expected: 4 FAIL / ERROR with `ModuleNotFoundError: No module named 'sportstradamus.dashboard.viewport'`

- [x] **Step 3: Implement `viewport.py`**

Create `src/sportstradamus/dashboard/viewport.py`:

```python
"""Mobile detection — one User-Agent sniff per session (Phase M spec §2.1).

Resolution order: the ``_force_mobile`` session key (tests), a ``?m=1`` /
``?m=0`` query param (manual override, made sticky in session state), then the
request's User-Agent — the ``"Mobi"`` substring every phone browser carries
(iPad reports a desktop UA and deliberately takes the desktop path). The
verdict is cached in session state: a phone never becomes a desktop
mid-session, and a desktop window dragged narrow staying on the desktop path
is by design. The 767px breakpoint twin for the CSS layer lives in
``theme.MOBILE_MAX_PX``; this module never measures pixels.
"""

from __future__ import annotations

import streamlit as st

_CACHED = "_is_mobile"
FORCE_KEY = "_force_mobile"


def is_mobile() -> bool:
    """True when this session should render the phone experience."""
    ss = st.session_state
    if FORCE_KEY in ss:
        return bool(ss[FORCE_KEY])
    param = st.query_params.get("m")
    if param in ("0", "1"):
        ss[_CACHED] = param == "1"
    if _CACHED not in ss:
        agent = (st.context.headers or {}).get("User-Agent", "")
        ss[_CACHED] = "Mobi" in agent
    return ss[_CACHED]
```

- [x] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/golden/test_viewport.py -n0 -v`
Expected: 4 PASS

- [x] **Step 5: Ruff + commit**

```bash
poetry run ruff check src/sportstradamus/
git add src/sportstradamus/dashboard/viewport.py tests/golden/test_viewport.py
git commit -m "feat(p-m): viewport.is_mobile — UA sniff with test/query overrides"
```

---

### Task 2: `theme.py` mobile media block + token-golden extension

**Files:**
- Modify: `src/sportstradamus/dashboard/theme.py` (constants ~line 128, template ~line 293)
- Test: `tests/golden/test_design_tokens.py`

- [x] **Step 1: Write the failing test**

Append to `tests/golden/test_design_tokens.py`:

```python
def test_mobile_media_block_present_and_resolved() -> None:
    """Phase M: theme.APP_CSS carries exactly one mobile media block, keyed to
    MOBILE_MAX_PX, with every template sentinel resolved."""
    from sportstradamus.dashboard import theme

    assert theme.MOBILE_MAX_PX == 767
    assert f"@media (max-width: {theme.MOBILE_MAX_PX}px)" in theme.APP_CSS
    assert "__MOBILE_MAX__" not in theme.APP_CSS
    assert "__STARFIELD_DUST__" not in theme.APP_CSS
```

- [x] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_design_tokens.py::test_mobile_media_block_present_and_resolved -n0 -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'MOBILE_MAX_PX'`

- [x] **Step 3: Add the constant and media block**

In `src/sportstradamus/dashboard/theme.py`, after the `_STARFIELD_MAX_ALPHA` constant (line ~130), add:

```python
# Phase M breakpoint — the CSS twin of viewport.is_mobile()'s UA branch. Style-only
# rules reshape below this width; render-level differences go through is_mobile().
MOBILE_MAX_PX = 767
```

In `_APP_CSS_TEMPLATE`, insert directly before the closing `</style>` (after the `prefers-reduced-motion: reduce` line):

```css
/* Phase M (spec §2.3): style-only phone reshaping. Render-level differences (nav,
   dock, Board cards, constellation touch) branch on viewport.is_mobile() instead. */
@media (max-width: __MOBILE_MAX__px){
  .page-hero{padding:10px 12px 3px}
  .page-hero .hero-title{font-size:19px}
  .celestial-kicker{font-size:9.5px;letter-spacing:.2em}
  .tonight-card{flex-direction:column;gap:8px;padding:12px 14px}
  .tonight-card .tc-side{flex-direction:row;flex:0 0 auto;justify-content:flex-start}
  .tc-matchup{font-size:19px}
  .tc-foot{gap:10px;margin-top:8px}
  .st-key-constellation_legpanel{padding:6px 10px}
}
```

Change the `APP_CSS` build line (currently `APP_CSS = _APP_CSS_TEMPLATE.replace(...)`) to resolve both sentinels:

```python
APP_CSS = _APP_CSS_TEMPLATE.replace("__STARFIELD_DUST__", _starfield_background()).replace(
    "__MOBILE_MAX__", str(MOBILE_MAX_PX)
)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/golden/test_design_tokens.py -n0 -v`
Expected: all PASS (the new test plus every existing token test — the media block adds no banned pattern).

- [x] **Step 5: Live check + commit**

Run `poetry run dashboard`, open devtools device mode (390px): Tonight cards stack single-column, hero shrinks, no horizontal scroll on Tonight. Record verdict.

```bash
poetry run ruff check src/sportstradamus/
git add src/sportstradamus/dashboard/theme.py tests/golden/test_design_tokens.py
git commit -m "feat(p-m): mobile media layer in APP_CSS behind MOBILE_MAX_PX

Live check: Tonight 390px stacks single-column, no h-scroll — pass."
```

---

### Task 3: prep — public `slip_shrinkage`, keyed `bankroll_input`

The dock (Task 4) needs both; neither is reachable cleanly today.

**Files:**
- Modify: `src/sportstradamus/dashboard/components/slip_builder.py:61` (`_slip_shrinkage` → `slip_shrinkage`, plus its two callers at ~lines 152 and 179)
- Modify: `src/sportstradamus/dashboard/components/slip_state.py:44` (`bankroll_input` key param)

- [x] **Step 1: Find every `_slip_shrinkage` reference**

Run: `grep -rn "_slip_shrinkage" src/ tests/`
Expected: definition + two callers in `slip_builder.py`; possibly monkeypatch references in `tests/golden/test_slip_builder.py`. Every hit found must be renamed in Step 2.

- [x] **Step 2: Rename to public**

In `slip_builder.py`, rename the function and both call sites:

```python
def slip_shrinkage(legs: Sequence[Mapping]) -> float:
    """Worst-cell kelly_shrinkage across the slip's legs (1.0 when stats are absent)."""
    stats = load_model_stats()
    if stats.empty or "kelly_shrinkage" not in stats.columns:
        return 1.0
    vals = []
    for leg in legs:
        cell = stats.loc[(stats["league"] == leg["league"]) & (stats["market"] == leg["market"])]
        if not cell.empty and pd.notna(cell.iloc[0]["kelly_shrinkage"]):
            vals.append(float(cell.iloc[0]["kelly_shrinkage"]))
    return min(vals) if vals else 1.0
```

(Callers: `shrink = slip_shrinkage(legs)` in both `render_constellation_builder` and `render_simple_builder`; update any test references found in Step 1 the same way.)

- [x] **Step 3: Key param on `bankroll_input`**

In `slip_state.py`, change the signature and widget key (the docstring's convention note stays):

```python
def bankroll_input(key: str = "_bankroll_widget") -> None:
    """Render the one global bankroll control; mirror it onto the plain slip key.

    Follows the app's widget-key→plain-key convention so the builders read a
    stable non-widget key (``slip_bankroll``) that every Kelly stake scales to.
    ``key`` exists because the mobile slip dock renders a second instance in the
    same run as the sidebar shelf's — two widgets can't share one key.
    """
    value = st.number_input(
        "Bankroll ($)",
        min_value=0.0,
        value=float(st.session_state[_BANKROLL]),
        step=50.0,
        key=key,
    )
    st.session_state[_BANKROLL] = value
```

- [x] **Step 4: Run the touched suites**

Run: `poetry run pytest tests/golden/test_slip_builder.py tests/golden/test_slip_engine.py tests/golden/test_dashboard_render_smoke.py -n0 -v`
Expected: all PASS (pure rename + default-preserving param).

- [x] **Step 5: Ruff + commit**

```bash
poetry run ruff check src/sportstradamus/
git add src/sportstradamus/dashboard/components/slip_builder.py src/sportstradamus/dashboard/components/slip_state.py tests/
git commit -m "refactor(p-m): public slip_shrinkage + keyed bankroll_input for the dock"
```

---

### Task 4: `slip_dock.py` — bottom bar + sheet

**Files:**
- Create: `src/sportstradamus/dashboard/components/slip_dock.py`
- Test: `tests/golden/test_slip_dock.py`

- [x] **Step 1: Write the failing test**

Create `tests/golden/test_slip_dock.py`:

```python
"""AppTest smoke for the mobile slip dock: hidden when empty, bar when filled,
sheet contents + remove-leg behavior when expanded.

Renders the dock in a minimal script (not the full app) so the test exercises
the component alone; legs are seeded as canonical structured legs the way
slip_state stores them. Offers/corr parquets aren't needed — score_slip prices
off the leg snapshots and an empty corr frame.
"""

from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

_SCRIPT = """
import streamlit as st
from sportstradamus.dashboard.components.slip_dock import render_slip_dock
from sportstradamus.dashboard.components.slip_state import init_slip_state

init_slip_state()
render_slip_dock()
"""


def _leg(player: str, market: str) -> dict:
    return {
        "player": player,
        "market": market,
        "stat": market,
        "bet": "Over",
        "line": 25.5,
        "league": "NBA",
        "game": "NYK/BOS",
        "team": "NYK",
        "platform": "Underdog",
        "win_prob": 0.60,
        "boost": 1.0,
        "kelly": 0.05,
        "ev": 1.10,
    }


def _dock_test(legs: list[dict], *, open_sheet: bool = False) -> AppTest:
    at = AppTest.from_string(_SCRIPT, default_timeout=15)
    at.session_state["slip_legs"] = legs
    if open_sheet:
        at.session_state["slip_dock_open"] = True
    at.run()
    assert not at.exception
    return at


def test_empty_slip_renders_nothing():
    at = _dock_test([])
    assert not any("slip_dock" in (b.key or "") for b in at.button)


def test_bar_summarizes_slip():
    at = _dock_test([_leg("Jalen Brunson", "PTS"), _leg("Josh Hart", "REB")])
    assert any(b.key == "slip_dock_toggle" for b in at.button)
    summary = " ".join(m.value for m in at.markdown)
    assert "2 legs" in summary


def test_sheet_lists_legs_and_remove_works():
    at = _dock_test([_leg("Jalen Brunson", "PTS"), _leg("Josh Hart", "REB")], open_sheet=True)
    body = " ".join(m.value for m in at.markdown)
    assert "Brunson" in body
    at.button(key="slip_dock_rm_0").click().run()
    assert len(at.session_state["slip_legs"]) == 1
```

- [x] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_slip_dock.py -n0 -v`
Expected: FAIL with `ModuleNotFoundError ... slip_dock`

- [x] **Step 3: Implement the dock**

Create `src/sportstradamus/dashboard/components/slip_dock.py`:

```python
"""Mobile slip dock — a fixed bottom bar that expands into the slip sheet.

The phone replacement for the sidebar rail's ambient visibility (Phase M spec
§3): collapsed, a one-line summary of the active slip rides the bottom of the
viewport on every surface; expanded, the sheet lists the legs with remove
controls, prices the slip through the same ``score_slip`` path the builders
use, and carries the bankroll + Lock it in! controls. Renders nothing without
legs, so desktop (which never mounts it) and an empty phone session are
unaffected. Money is ``Decimal``; scoring reuses slip_state/slip_builder
helpers — no duplicated math.
"""

from __future__ import annotations

import streamlit as st

from sportstradamus.dashboard.components.slip_builder import slip_shrinkage
from sportstradamus.dashboard.components.slip_state import (
    _BANKROLL,
    _LEGS,
    _PLATFORM,
    bankroll_input,
    clear_slip,
    lock_in,
    remove_leg,
)
from sportstradamus.dashboard.data import load_current_game_corr
from sportstradamus.dashboard.slip_engine import score_slip
from sportstradamus.leg_schema import leg_label
from sportstradamus.prediction.stories.legs import validate_parlay_legs

_OPEN = "slip_dock_open"

# Fixed-bottom chrome for the .st-key-slip_dock container. Token literals restated
# because an injected <style> can't read config.toml (same convention as the
# constellation component's index.html): surface #1A1D24, text #E6E9EF, gold #C9A227.
# The main-container bottom padding keeps the last page row tappable above the bar.
_DOCK_CSS = """
<style>
.st-key-slip_dock{position:fixed;left:0;right:0;bottom:0;z-index:400;
  background:#1A1D24;border-top:1px solid rgba(201,162,39,.42);border-radius:4px 4px 0 0;
  box-shadow:0 -4px 18px rgba(0,0,0,.45);padding:8px 12px;
  max-height:70vh;overflow-y:auto}
.st-key-slip_dock [data-testid="stVerticalBlock"]{gap:.35rem}
.slip-dock-line{font-family:'IBM Plex Mono',monospace;font-size:13px;color:#E6E9EF;margin:2px 0}
[data-testid="stMainBlockContainer"]{padding-bottom:120px}
</style>
"""


def render_slip_dock() -> None:
    """Render the bar (and, when toggled open, the sheet) for a non-empty slip."""
    legs = st.session_state[_LEGS]
    if not legs:
        return
    st.markdown(_DOCK_CSS, unsafe_allow_html=True)
    score = _price(legs)
    with st.container(key="slip_dock"):
        bar_col, toggle_col = st.columns([5, 1])
        bar_col.markdown(f'<div class="slip-dock-line">{_summary(legs, score)}</div>', unsafe_allow_html=True)
        opened = st.session_state.get(_OPEN, False)
        if toggle_col.button(
            ":material/expand_less:" if not opened else ":material/expand_more:",
            key="slip_dock_toggle",
            help="Open the slip",
        ):
            st.session_state[_OPEN] = not opened
            st.rerun()
        if opened:
            _render_sheet(legs, score)


def _price(legs):
    """SlipScore for ≥2 legs, else None — the same gate the builders apply."""
    if len(legs) < 2:
        return None
    from decimal import Decimal

    return score_slip(
        legs,
        load_current_game_corr(),
        platform=st.session_state[_PLATFORM],
        bankroll=Decimal(str(st.session_state[_BANKROLL])),
        shrinkage=slip_shrinkage(legs),
    )


def _summary(legs, score) -> str:
    if score is None:
        return f"{len(legs)} leg · add another to price"
    return (
        f"{len(legs)} legs · {float(score.payout):.2f}x · "
        f"EV {float(score.model_ev) - 1:+.0%} · ${score.stake}"
    )


def _render_sheet(legs, score) -> None:
    for i, leg in enumerate(legs):
        text_col, rm_col = st.columns([8, 1])
        text_col.markdown(
            f'<div class="slip-dock-line">{leg_label(leg)} · {leg["league"]}</div>',
            unsafe_allow_html=True,
        )
        if rm_col.button(":material/close:", key=f"slip_dock_rm_{i}", help="Remove leg"):
            remove_leg(i)
            st.rerun()
    bankroll_input(key="_bankroll_dock")
    if score is None:
        st.caption("Select at least two legs to price the slip.")
        return
    valid, reason = validate_parlay_legs(legs)
    if not valid:
        st.warning(reason)
    st.markdown(
        f'<div class="slip-dock-line">Kelly stake ${score.stake} · '
        f"joint {float(score.joint_p):.0%}</div>",
        unsafe_allow_html=True,
    )
    lock_col, clear_col = st.columns(2)
    if lock_col.button("Lock it in!", key="slip_dock_lock", type="primary", disabled=not valid):
        lock_in(score, "", slip_shrinkage(legs))
    if clear_col.button("Clear", key="slip_dock_clear"):
        clear_slip()
        st.rerun()
```

Note the local `from decimal import Decimal` is wrong style — hoist it to the module imports (shown here inline only to flag it; the committed file imports `Decimal` at the top with the rest).

- [x] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/golden/test_slip_dock.py -n0 -v`
Expected: 3 PASS. If `score_slip` needs a non-empty corr frame, `load_current_game_corr()` already degrades to empty; the two-leg fixture prices independent.

- [x] **Step 5: Ruff + commit**

```bash
poetry run ruff check src/sportstradamus/
git add src/sportstradamus/dashboard/components/slip_dock.py tests/golden/test_slip_dock.py
git commit -m "feat(p-m): slip dock — fixed bottom bar + expandable sheet"
```

---

### Task 5: `app.py` wiring — top nav + dock mount

**Files:**
- Modify: `src/sportstradamus/dashboard/app.py` (imports; `st.navigation` call ~line 49; after `pg.run()` line 114)
- Test: `tests/golden/test_dashboard_render_smoke.py`

- [x] **Step 1: Write the failing test**

Append to `tests/golden/test_dashboard_render_smoke.py` (reuse the module's existing `_WRAPPER`):

```python
def test_app_boots_mobile_with_dock():
    """Forced-mobile boot: app renders, and a seeded slip mounts the dock bar."""
    st.cache_data.clear()
    at = AppTest.from_string(_WRAPPER, default_timeout=30)
    at.session_state["_force_mobile"] = True
    at.session_state["slip_legs"] = [
        {
            "player": "Jalen Brunson", "market": "PTS", "stat": "PTS", "bet": "Over",
            "line": 25.5, "league": "NBA", "game": "NYK/BOS", "team": "NYK",
            "platform": "Underdog", "win_prob": 0.6, "boost": 1.0, "kelly": 0.05, "ev": 1.1,
        },
        {
            "player": "Josh Hart", "market": "REB", "stat": "REB", "bet": "Over",
            "line": 8.5, "league": "NBA", "game": "NYK/BOS", "team": "NYK",
            "platform": "Underdog", "win_prob": 0.58, "boost": 1.0, "kelly": 0.04, "ev": 1.08,
        },
    ]
    at.run()
    assert not at.exception
    assert any(b.key == "slip_dock_toggle" for b in at.button)
```

- [x] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_dashboard_render_smoke.py::test_app_boots_mobile_with_dock -n0 -v`
Expected: FAIL on the `slip_dock_toggle` assertion (app boots, dock never mounted).

- [x] **Step 3: Wire `app.py`**

Add imports:

```python
from sportstradamus.dashboard.components.slip_dock import render_slip_dock
from sportstradamus.dashboard.viewport import is_mobile
```

After `init_slip_state()` add:

```python
_mobile = is_mobile()
```

Change the `st.navigation(` call to take the position (the dict argument stays exactly as-is):

```python
pg = st.navigation(
    {
        ...  # unchanged page dict
    },
    position="top" if _mobile else "sidebar",
)
```

After `pg.run()` (end of file) add:

```python
if _mobile:
    render_slip_dock()
```

- [x] **Step 4: Run the smoke suite**

Run: `poetry run pytest tests/golden/test_dashboard_render_smoke.py -n0 -v`
Expected: all PASS — the new mobile test plus every existing desktop test (AppTest sends no UA, so they stay on the desktop branch; this is the desktop-unchanged pin).

- [x] **Step 5: Live check + commit**

`poetry run dashboard` in device mode: top nav bar shows the pages; add two legs from Games, then confirm the dock bar sits fixed at the viewport bottom on Tonight/Board/Games, expands, removes a leg, and Lock it in! saves (check the sidebar shelf afterward). Also confirm desktop (normal window) is pixel-identical: sidebar nav, no dock. Record verdict.

```bash
poetry run ruff check src/sportstradamus/
git add src/sportstradamus/dashboard/app.py tests/golden/test_dashboard_render_smoke.py
git commit -m "feat(p-m): mobile app chrome — top nav + slip dock mount

Live check: dock fixed-bottom on all surfaces, expands/locks; desktop unchanged — pass."
```

---

### Task 6: Board mobile — `offer_cards.py` + filters expander

**Files:**
- Create: `src/sportstradamus/dashboard/components/offer_cards.py`
- Modify: `src/sportstradamus/dashboard/surfaces/board.py` (filters block lines ~55–116; grid/select block lines ~156–221)
- Test: `tests/golden/test_offer_cards.py`

- [x] **Step 1: Write the failing test**

Create `tests/golden/test_offer_cards.py`:

```python
"""AppTest for the Board's mobile card list: cards render from a scored-offers
frame, paging extends, Add seeds the simple slip, Detail pushes the dialog stack."""

from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

_SCRIPT = """
import pandas as pd
import streamlit as st
from sportstradamus.dashboard.components.deep_dive import init_detail_state
from sportstradamus.dashboard.components.offer_cards import render_offer_cards
from sportstradamus.dashboard.components.slip_state import init_slip_state

init_slip_state()
init_detail_state()
df = pd.DataFrame(st.session_state["_fixture_rows"])
render_offer_cards(df)
"""


def _rows(n: int) -> list[dict]:
    return [
        {
            "League": "NBA", "Match": "NYK vs BOS", "Player": f"Player {i}",
            "Market": "PTS", "Market Display": "Points", "Bet": "Over", "Line": 20.5 + i,
            "Boost": 1.0, "Win Prob": 0.55, "Model Edge": 4.0, "Consensus Edge": 2.0,
            "Platform": "Underdog", "Game": "NYK/BOS", "Team": "NYK", "Date": "2026-07-16",
            "Model EV": 1.05,
        }
        for i in range(n)
    ]


def _card_test(n: int) -> AppTest:
    at = AppTest.from_string(_SCRIPT, default_timeout=15)
    at.session_state["_fixture_rows"] = _rows(n)
    at.run()
    assert not at.exception
    return at


def test_cards_render_and_page():
    at = _card_test(35)
    assert any(b.key == "offer_cards_more" for b in at.button)
    body = " ".join(m.value for m in at.markdown)
    assert "Player 0" in body and "Player 34" not in body
    at.button(key="offer_cards_more").click().run()
    body = " ".join(m.value for m in at.markdown)
    assert "Player 34" in body


def test_add_seeds_simple_slip():
    at = _card_test(3)
    at.button(key="offer_card_add_0").click().run()
    assert len(at.session_state["slip_legs"]) == 1
    assert at.session_state["slip_builder"] == "simple"


def test_detail_pushes_stack():
    at = _card_test(3)
    at.button(key="offer_card_detail_1").click().run()
    assert at.session_state["detail_stack"] == [1]
```

- [x] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_offer_cards.py -n0 -v`
Expected: FAIL with `ModuleNotFoundError ... offer_cards`

- [x] **Step 3: Implement `offer_cards.py`**

Create `src/sportstradamus/dashboard/components/offer_cards.py`:

```python
"""The Board's mobile card list — one bordered card per offer (Phase M spec §4.3).

The phone replacement for the 10-column AG-Grid, which can't fit a 390px
viewport. Each card carries the row's read (player, market, line/side, Win %,
Model Edge, platform) with numerals in the mono code face, plus the grid's two
actions: Detail (the deep-dive dialog, via the shared ``detail_stack``) and Add
to slip (the same ``add_to_simple_slip`` path the grid selection uses). Pages
``_PAGE_SIZE`` at a time — a slate can run hundreds of offers and phone DOM is
the constraint.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.slip_state import add_to_simple_slip

# Cards rendered before the "Show more" button extends the list; sized so a full
# slate stays scrollable without flooding the phone DOM.
_PAGE_SIZE = 30
_SHOWN = "offer_cards_shown"


def render_offer_cards(offers: pd.DataFrame) -> None:
    """Render the paged card list over a filtered, scored offers frame."""
    shown = st.session_state.setdefault(_SHOWN, _PAGE_SIZE)
    for idx, row in offers.head(shown).iterrows():
        _render_card(idx, row)
    if len(offers) > shown and st.button(
        f"Show more ({len(offers) - shown} left)", key="offer_cards_more"
    ):
        st.session_state[_SHOWN] = shown + _PAGE_SIZE
        st.rerun()


def _render_card(idx, row: pd.Series) -> None:
    arrow = "▲" if str(row.get("Bet", "")).lower().startswith("o") else "▼"
    win = float(row.get("Win Prob") or 0.0)
    edge = float(row.get("Model Edge") or 0.0)
    market = row.get("Market Display") or row.get("Market", "")
    with st.container(border=True):
        st.markdown(
            f"**{row['Player']}** · {market} {arrow} `{row['Line']:.10g}`  \n"
            f"Win `{win:.0%}` · Edge `{edge:+.1f}%` · {row.get('Platform', '')} · "
            f"{row.get('League', '')}"
        )
        detail_col, add_col = st.columns(2)
        if detail_col.button("Detail", key=f"offer_card_detail_{idx}"):
            st.session_state.detail_stack = [idx]
        if add_col.button("Add to slip", key=f"offer_card_add_{idx}"):
            add_to_simple_slip(row.to_dict())
            st.rerun()
```

(`Model Edge` arrives already ×100 on the Board's mobile path — see the Step 4 board wiring, which reuses the grid's own percent conversion block.)

- [x] **Step 4: Wire `board.py`**

Add imports at the top of `surfaces/board.py`:

```python
from sportstradamus.dashboard.components.offer_cards import render_offer_cards
from sportstradamus.dashboard.viewport import is_mobile
```

After `offers = sport_filtered(load_current_offers())` (line ~26) add:

```python
mobile = is_mobile()
```

Wrap the filter widgets (the `col1, col2, col3 = st.columns(3)` block through `player_query = st.text_input(...)` AND the "Numeric range filters" slider block) in a host container — the lens/side segmented controls above stay outside:

```python
filter_host = st.expander("Filters") if mobile else st.container()
with filter_host:
    col1, col2, col3 = st.columns(3)
    ...  # the three multiselects, unchanged
    player_query = st.text_input("Player search", placeholder="e.g. Jokic")
```

The lens/side filtering logic between the two widget groups is pure dataframe code and stays where it is; move only the *widgets* inside the host. The range-slider block moves inside the same `with filter_host:` (dedent-sensitive — the sliders currently render after `add_edges`; keep the data-flow order by computing `filtered = columns.add_edges(filtered)` before opening the slider section, exactly as today, with only the `st.caption`/`st.columns`/`slot.slider` calls inside the host).

Replace the grid render + selection block (from `selected_rows = render_themed_grid(` through the `st.subheader("Build a cross-game slip")` section's `else:` caption) with a branch:

```python
if mobile:
    selected_rows = []
    render_offer_cards(grid_df.rename(columns={"Win %": "Win Prob"}).assign(
        **{"Market": filtered["Market"], "Bet": filtered["Bet"]}
    ) if False else filtered)
else:
    selected_rows = render_themed_grid(
        grid_df,
        numeric_cols=numeric_cols,
        heatmap_col=columns.MODEL_EDGE,
        heatmap_center=0.0,
        header_help=columns.HELP,
        percent_cols=["Win %"],
        signed_percent_cols=[columns.MODEL_EDGE, "Cons Edge"],
        arrow_col="Line",
        hidden_cols=["Bet", "Market Slug"],
    )
    st.caption("Trend sparklines arrive with the L1 line-movement export.")
```

**Correction to the above (write it this way):** the cards read the *pre-rename* `filtered` frame (it has `Player`, `Market Display`, `Bet`, `Line`, `Win Prob` 0–1, `Model Edge` as a fraction). Convert the two percent fields the same way the grid path does, on a copy:

```python
if mobile:
    selected_rows = []
    cards_df = filtered.copy()
    cards_df["Win Prob"] = pd.to_numeric(cards_df["Win Prob"], errors="coerce")
    cards_df[columns.MODEL_EDGE] = (
        pd.to_numeric(cards_df[columns.MODEL_EDGE], errors="coerce") * 100
    ).round(1)
    render_offer_cards(cards_df)
else:
    ...  # grid branch exactly as above
```

(`render_offer_cards` formats `Win Prob` itself via `{win:.0%}`, so it takes the raw 0–1 value; only `Model Edge` needs the ×100.)

The `if st.session_state.corr_nav: / elif selected_rows: / else:` selection-tracking block stays as-is (mobile's empty `selected_rows` routes to the `else` branch only when the detail stack should reset — but the cards set `detail_stack` directly *after* that block runs, so card-driven details survive; confirm test_detail_pushes_stack covers the standalone component and the live check covers the page). Guard the desktop-only "Build a cross-game slip" selected-row section:

```python
if not mobile:
    st.subheader("Build a cross-game slip")
    ...  # existing selected_rows add-button block, unchanged
render_simple_builder(filtered, load_current_game_corr())
```

**Ordering caveat:** on mobile the existing `else:` branch of the selection tracker (`detail_stack = []`) runs every rerun *before* the cards render. A card's Detail click sets `detail_stack` on its rerun **after** that reset, so the dialog opens; but make the reset desktop-only anyway to avoid the reset racing the card click on the next unrelated rerun:

```python
elif selected_rows:
    ...  # unchanged
elif not mobile:
    st.session_state.detail_stack = []
    st.session_state.last_grid_key = None
```

- [x] **Step 5: Run tests**

Run: `poetry run pytest tests/golden/test_offer_cards.py tests/golden/test_dashboard_render_smoke.py -n0 -v`
Expected: all PASS (desktop board smoke test unchanged — it never sets the force key).

- [x] **Step 6: Live check + commit**

Device mode → Board: filters collapsed in an expander, cards readable, no horizontal scroll, Detail opens the dialog near-fullscreen, Add feeds the dock bar, Show more pages. Desktop Board: identical to before (grid, filters open). Record verdict.

```bash
poetry run ruff check src/sportstradamus/
git add src/sportstradamus/dashboard/components/offer_cards.py src/sportstradamus/dashboard/surfaces/board.py tests/golden/test_offer_cards.py
git commit -m "feat(p-m): Board mobile — offer card list + collapsed filters

Live check: 390px cards + expander + detail/add; desktop grid unchanged — pass."
```

---

### Task 7: constellation figure `mobile=` param

**Files:**
- Modify: `src/sportstradamus/dashboard/components/constellation.py` (constants ~line 71; `constellation_figure` ~line 174; `_star_sizes` ~line 365; `_add_node_trace` ~line 587; `_add_deep_trace` ~line 266; `_add_wider_trace` ~line 309; `_card_fields`/customdata assembly)
- Test: `tests/golden/test_constellation.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/golden/test_constellation.py` (reuse the module's existing fixture frames — it already builds a pool/corr; follow its local naming):

```python
def test_mobile_figure_raises_size_floor_and_flags_slip_membership():
    """mobile=True lifts every star to the touch floor and appends the in-slip
    flag to customdata; mobile=False stays byte-identical to the no-arg figure."""
    from sportstradamus.dashboard.components import constellation as c

    legs, corr, pool = _figure_fixture()  # or the module's existing equivalent
    baseline = c.constellation_figure(legs, corr, pool)
    desktop = c.constellation_figure(legs, corr, pool, mobile=False)
    assert desktop.to_json() == baseline.to_json()

    fig = c.constellation_figure(legs, corr, pool, mobile=True)
    node_traces = [t for t in fig.data if t.name in ("active", "candidate")]
    assert node_traces
    for t in node_traces:
        assert min(t.marker.size) >= c._SIZE_MIN_MOBILE
        assert t.textfont.size == c._LABEL_FONT_SIZE_MOBILE
        for cd in t.customdata:
            assert len(cd) == 9
            assert cd[8] in (0, 1)
    active = [t for t in node_traces if t.name == "active"]
    if active:
        assert all(cd[8] == 1 for cd in active[0].customdata)
```

If `test_constellation.py` has no reusable fixture builder, add one mirroring its existing figure tests' pool/corr construction (two teams, ≥3 model-liked rows with `K` > 0, one leg active).

- [x] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/golden/test_constellation.py -n0 -v -k mobile`
Expected: FAIL with `TypeError: constellation_figure() got an unexpected keyword argument 'mobile'`

- [x] **Step 3: Implement**

In `constellation.py`, next to `_SIZE_MIN`/`_LABEL_FONT_SIZE` add:

```python
# Phase M touch floors: a fingertip needs ~22px; the label lifts with it. The Kelly
# ordering (size = edge) survives — the floor compresses the range, never reorders it.
_SIZE_MIN_MOBILE = 22
_LABEL_FONT_SIZE_MOBILE = 13
```

Thread the flag:

```python
def constellation_figure(
    slip_legs: Sequence[Mapping],
    corr: pd.DataFrame | None,
    pool: pd.DataFrame | None = None,
    *,
    deep_pool: pd.DataFrame | None = None,
    wider_groups: list[tuple[str, list[dict]]] | None = None,
    mobile: bool = False,
) -> go.Figure:
```

Inside, derive the two knobs and pass them down (everything else unchanged):

```python
    floor = _SIZE_MIN_MOBILE if mobile else _SIZE_MIN
    label_size = _LABEL_FONT_SIZE_MOBILE if mobile else _LABEL_FONT_SIZE
    ...
    sizes = _star_sizes(keys, info, floor=floor)
    ...
    if deep_pool is not None:
        _add_deep_trace(fig, deep_pool, slip_legs, radius=_DEEP_RADIUS * focus_scale, floor=floor)
    ...
    _add_node_trace(
        fig, [k for k in keys if k not in active], pos, info, sizes, team_color,
        active=False, label_size=label_size,
    )
    _add_node_trace(
        fig, [k for k in keys if k in active], pos, info, sizes, team_color,
        active=True, label_size=label_size,
    )
    if wider_groups is not None:
        _add_wider_trace(fig, wider_groups, floor=floor)
```

Update the helpers' signatures — each keeps its current default so every other caller is untouched:

```python
def _star_sizes(keys, info, *, floor: float = _SIZE_MIN) -> dict[str, float]:
    top = max((info[k]["edge"] for k in keys), default=0.0)
    if top <= 0:
        return dict.fromkeys(keys, float(floor))
    span = _SIZE_MAX - floor
    return {k: floor + max(info[k]["edge"], 0.0) / top * span for k in keys}
```

`_add_deep_trace(..., *, radius, floor: float = _SIZE_MIN)` — its marker size list becomes `[floor] * len(keys)`.
`_add_wider_trace(fig, wider_groups, *, floor: float = _SIZE_MIN)` — same substitution.
`_add_node_trace(..., *, active: bool, label_size: int = _LABEL_FONT_SIZE)` — `textfont` size becomes `label_size`, and the customdata row gains the in-slip flag:

```python
            customdata=[[k, *info[k]["card"], 1 if active else 0] for k in keys],
```

Also append the flag (always `0` — neither trace can hold an in-slip leg) to the deep and wider traces' customdata rows:

```python
            customdata=[[k, *info[k]["card"], 0] for k in keys],       # deep
            customdata.append([corr_key(row), *info["card"], 0])       # wider
```

- [x] **Step 4: Run the constellation suite**

Run: `poetry run pytest tests/golden/test_constellation.py -n0 -v`
Expected: all PASS — the new mobile test AND every existing figure pin (desktop customdata grew one trailing element; if an existing golden pins customdata length/content exactly, update that pin in the same commit and say so in the message — the flag is additive, index 0–7 unchanged, and the component reads by index so desktop JS behavior is unaffected).

- [x] **Step 5: Ruff + commit**

```bash
poetry run ruff check src/sportstradamus/
git add src/sportstradamus/dashboard/components/constellation.py tests/golden/test_constellation.py
git commit -m "feat(p-m): constellation mobile floors + in-slip customdata flag"
```

---

### Task 8: constellation component touch mode (Python kwarg + JS + CSS)

**Files:**
- Modify: `src/sportstradamus/dashboard/components/constellation_component/__init__.py`
- Modify: `src/sportstradamus/dashboard/components/constellation_component/build/main.js`
- Modify: `src/sportstradamus/dashboard/components/constellation_component/build/index.html`
- Modify: `src/sportstradamus/dashboard/components/slip_builder.py` (`_render_constellation` ~line 236)

No JS test harness exists — the verification for this task is the recorded live-browser check (per DESIGN §4b: goldens can't see inside the iframe).

- [x] **Step 1: Python pass-through**

`constellation_component/__init__.py` — extend the render signature:

```python
def render_constellation(fig: go.Figure, *, key: str, mobile: bool = False) -> dict | None:
    """Render the star map; return the last ``{action, key, nonce}`` the user fired.

    ``action`` is ``"click"`` (toggle the star's leg) or ``"detail"`` (open the offer
    dialog); ``key`` is the star's ``Player|Market|Bet``. ``None`` until the user acts.
    The caller dedups by ``nonce`` — a repeat click re-sends the same value.
    ``mobile`` switches the frontend to its touch flow (docked tap card, no hover).
    """
    return _component(figure_json=fig.to_json(), mobile=mobile, key=key, default=None)
```

`slip_builder.py` — import the flag once and pass it through both calls in `_render_constellation`:

```python
from sportstradamus.dashboard.viewport import is_mobile
...
def _render_constellation(...):
    ...
    mobile = is_mobile()
    action = render_constellation(
        constellation_figure(
            legs, corr, pool, deep_pool=deep_pool, wider_groups=wider_groups, mobile=mobile
        ),
        key=f"{key_prefix}_constellation",
        mobile=mobile,
    )
```

- [x] **Step 2: `index.html` docked-card CSS**

Add to the `<style>` block (token literals per the file's existing header comment):

```css
      /* Phase M docked tap card: on touch the hover card becomes a full-width sheet
         pinned to the iframe's bottom edge (thumb zone). Same tokens, new geometry. */
      .cst-card.cst-docked {
        position: fixed;
        left: 8px;
        right: 8px;
        bottom: 8px;
        top: auto;
        width: auto;
        z-index: 30;
      }
      .cst-actions {
        display: flex;
        gap: 8px;
      }
      .cst-actions .cst-btn {
        flex: 1;
      }
      .cst-btn.cst-toggle {
        background: transparent;
        color: var(--gold);
        border: 1px solid var(--gold);
      }
```

- [x] **Step 3: `main.js` touch flow**

Apply these changes (complete replacements for each touched function):

At the state block (after `let renderSeq = 0;`):

```javascript
  let MOBILE = false; // set per render from Python's mobile prop (viewport.is_mobile)
  const COARSE_POINTER =
    window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  const MOBILE_CARD_PAD = 150; // extra frame height so the docked card clears the map
```

In `render(args)`, right after `const fig = JSON.parse(args.figure_json);`:

```javascript
    MOBILE = !!args.mobile || COARSE_POINTER;
```

And change the frame-height line at the end of `render` to reserve the dock strip:

```javascript
    const height =
      (fig.layout && fig.layout.height ? fig.layout.height : 380) +
      FRAME_PAD +
      (MOBILE ? MOBILE_CARD_PAD : 0);
    setFrameHeight(height);
```

Replace `attachHandlers` — the touch branch turns the first tap into select-and-card, the second tap (same star, or the card's toggle button) into the actual toggle; hover stays desktop-only:

```javascript
  function attachHandlers() {
    chartDiv.on("plotly_click", function (data) {
      const pt = pointFrom(data);
      if (!pt) return;
      if (!MOBILE) {
        emit("click", pt.customdata[0]);
        return;
      }
      skyTap = false; // a star tap is never a dismiss
      if (activeKey === pt.customdata[0]) {
        emit("click", activeKey); // second tap on the focused star = toggle
        hideCard();
      } else {
        previewEdges(pt.customdata[0]);
        showCard(pt, data.event);
      }
    });
    chartDiv.on("plotly_hover", function (data) {
      if (MOBILE) return;
      const pt = pointFrom(data);
      if (!pt) return;
      previewEdges(pt.customdata[0]);
      showCard(pt, data.event);
    });
    chartDiv.on("plotly_unhover", function () {
      if (MOBILE) return;
      restoreEdges();
      scheduleHide();
    });
    card.addEventListener("mouseenter", function () {
      clearTimeout(hideTimer);
    });
    card.addEventListener("mouseleave", function () {
      if (!MOBILE) hideCard();
    });
    // Empty-sky tap dismisses the docked card: plotly_click marks star taps via
    // skyTap=false in the same event turn; any other tap on the map falls through here.
    chartDiv.addEventListener("click", function () {
      if (!MOBILE) return;
      if (skyTap) {
        restoreEdges();
        hideCard();
      }
      skyTap = true;
    });
  }
```

Add `let skyTap = true;` beside the other state lets.

Replace `showCard` and `cardHtml` — the card gains the toggle button on mobile, reading the Task 7 in-slip flag at `customdata[8]`:

```javascript
  function showCard(pt, mouseEvent) {
    clearTimeout(hideTimer);
    const cd = pt.customdata; // [key, player, market, bet, line, win, boost, kelly, inSlip]
    activeKey = cd[0];
    card.innerHTML = cardHtml(cd);
    card.querySelector(".cst-btn.cst-detail").addEventListener("click", function () {
      emit("detail", activeKey);
    });
    const toggle = card.querySelector(".cst-btn.cst-toggle");
    if (toggle) {
      toggle.addEventListener("click", function () {
        emit("click", activeKey);
        hideCard();
      });
    }
    card.classList.toggle("cst-docked", MOBILE);
    card.classList.remove("cst-hidden");
    card.setAttribute("aria-hidden", "false");
    if (!MOBILE) positionCard(mouseEvent);
  }
```

```javascript
  function cardHtml(cd) {
    const player = cd[1];
    const market = cd[2];
    const bet = cd[3];
    const line = cd[4];
    const win = cd[5];
    const boost = cd[6];
    const kelly = cd[7];
    const inSlip = !!cd[8];
    const toggleBtn = MOBILE
      ? '<button class="cst-btn cst-toggle" type="button">' +
        (inSlip ? "Remove from slip" : "Add to slip") +
        "</button>"
      : "";
    return [
      '<div class="cst-head">',
      '<div class="cst-shot" title="Player headshot — coming soon">',
      esc(initials(player)),
      "</div>",
      '<div class="cst-id"><div class="cst-name">',
      esc(player),
      '</div><div class="cst-leg">',
      esc(market) + " · " + esc(bet) + " " + esc(line),
      "</div></div></div>",
      '<div class="cst-stats">Win ',
      pct(win),
      " · ",
      (Number(boost) || 1).toFixed(2),
      'x · <span class="cst-kelly">Kelly ',
      pct(kelly),
      "</span></div>",
      '<div class="cst-scar">Last 5 — coming soon</div>',
      '<div class="cst-actions">',
      toggleBtn,
      '<button class="cst-btn cst-detail" type="button">Full detail →</button>',
      "</div>",
    ].join("");
  }
```

(The desktop card markup changes only in wrapping the one button in `.cst-actions` and adding the `cst-detail` class — update the selector in `showCard` as shown so desktop keeps working.)

- [x] **Step 4: Python-side sanity run**

Run: `poetry run pytest tests/golden/test_constellation.py tests/golden/test_slip_builder.py tests/golden/test_dashboard_render_smoke.py -n0 -v`
Expected: all PASS (JS/HTML aren't exercised; this catches the Python signature threading).

- [x] **Step 5: Live check (the real gate) + commit**

Device mode → Games:
1. Tap a candidate star → docked card slides up with **Add to slip** + **Full detail**; incident edges faint-preview.
2. Tap the same star again → leg joins the slip (star burns full color), card hides, dock bar updates.
3. Tap a slip star → card shows **Remove from slip**; the button removes it.
4. Tap empty sky → card dismisses.
5. Full detail → deep-dive dialog opens, slip intact.
6. Desktop (normal window): hover card + click-toggle behave exactly as before this task.
Record all six verdicts.

```bash
poetry run ruff check src/sportstradamus/
git add src/sportstradamus/dashboard/components/constellation_component/ src/sportstradamus/dashboard/components/slip_builder.py
git commit -m "feat(p-m): constellation touch mode — docked tap card, second-tap toggle

Live check: 6-point touch flow + desktop hover parity — pass."
```

---

### Task 9: Lab captions + docs (DESIGN.md, handoff, spec tick)

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/lab_diagnostics.py`, `lab_correlations.py`, `lab_training.py`, `lab_modifiers.py`, `receipts.py` (one caption each)
- Modify: `DESIGN.md` (§Mobile note), `docs/handoffs/dashboard-ux.md` (§6 Phase M entry + §10 ledger line)

- [x] **Step 1: Desk captions**

In each of the five files, directly after its `page_hero(...)` call, add:

```python
from sportstradamus.dashboard.viewport import is_mobile  # with the other imports

if is_mobile():
    st.caption("Best at a desk — this page keeps its desktop layout.")
```

- [x] **Step 2: DESIGN.md §Mobile note**

Add a short section after §8 Accessibility:

```markdown
## 8a. Mobile (Phase M)

The money loop (Tonight → Games → slip → stakes) renders a phone experience behind
`viewport.is_mobile()` (User-Agent; `?m=1` override) plus one `@media (max-width: 767px)`
block in `theme.APP_CSS` — `theme.MOBILE_MAX_PX` is the single breakpoint. Mobile chrome:
top nav, the slip dock (fixed bottom bar + sheet, gold hairline, surface tokens), Board
offer cards in place of the AG-Grid, and the constellation touch flow (tap → docked card,
second tap / card button toggles — selection stays alpha + saturation, §4a grammar
unchanged; the touch size floor scales stars, never reorders them). Receipts/Lab keep
desktop layouts. Desktop rendering is pixel-unchanged; every mobile difference gates on
`is_mobile()` or the media block.
```

- [x] **Step 3: Handoff updates**

In `docs/handoffs/dashboard-ux.md` §6, after the Phase R bullet, add:

```markdown
- **Phase M — mobile money loop** (spec
  [specs/2026-07-16-dashboard-mobile-design.md](../superpowers/specs/2026-07-16-dashboard-mobile-design.md),
  plan [plans/2026-07-16-dashboard-mobile-phase-m.md](../superpowers/plans/2026-07-16-dashboard-mobile-phase-m.md)):
  UA-branch (`viewport.is_mobile`) + one media block; top nav + slip dock (bottom
  bar/sheet) on phone; Board card list; constellation touch mode (docked tap card,
  second-tap toggle, in-slip customdata flag); Lab/Receipts desk-first captions.
  Desktop paths byte-identical; every task carries a recorded live-browser verdict.
  Phase D seam: D owns positions, M owns sizes/events; D's knot explode radius must
  respect the mobile floor.
```

Update the §3 status line (`> Status: ...`) to name Phase M as the active step, and append a one-line §10 ledger entry (newest-first, caveman-short, cap respected):

```markdown
- 2026-07-16 · **Phase M started** · mobile money-loop lane opened: spec+plan committed; viewport/theme/dock/board-cards/constellation-touch tasks queued.
```

- [x] **Step 4: Gates + commit**

Run: `poetry run pytest tests/golden/ && poetry run ruff check src/sportstradamus/`
Expected: clean (the captions are render-only; smoke tests don't set the force key).

```bash
git add DESIGN.md docs/handoffs/dashboard-ux.md src/sportstradamus/dashboard/surfaces/
git commit -m "docs(p-m): DESIGN §8a mobile, handoff Phase M entry, desk captions"
```

---

### Task 10: close-out — specialist, full gates, phone pass

- [x] **Step 1: refactoring-specialist (MANDATORY)**

Dispatch the `refactoring-specialist` subagent listing every `.py` touched this phase:
`viewport.py`, `theme.py`, `slip_state.py`, `slip_builder.py`, `slip_dock.py`, `app.py`, `offer_cards.py`, `board.py`, `constellation.py`, `constellation_component/__init__.py`, the five caption surfaces, plus the new/modified test files. Address anything it raises before proceeding.

- [x] **Step 2: Full gates**

```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/
poetry run pytest -m integration -n0 && touch "$CLAUDE_PROJECT_DIR/.claude/.state/integration_green"
```
Expected: all clean.

- [ ] **Step 3: Real-phone pass (owner, over tailscale)**

Acceptance walk on an actual phone (spec §6): Tonight cards → tap through to Games → build a slip by tapping stars (docked card confirm) → dock expands → Lock it in! → shelf shows it; Board cards readable, Detail + Add work; no horizontal scroll on any money-loop surface; `?m=0` flips the session back to desktop layout. Record the verdict in the handoff ledger line (amend "Phase M started" → shipped once green).

- [ ] **Step 4: Final commit + ledger tick**

Tick this plan's checkboxes, amend the §10 ledger entry to record the phone-pass verdict, and commit:

```bash
git add docs/
git commit -m "docs(p-m): close out Phase M — plan ticked, phone-pass verdict recorded"
```

Do **not** push — the branch already holds unpushed commits; the owner decides when to push `feature/dashboard-ux`.

---

## Plan self-review (done at write time)

- **Spec coverage:** §2.1 detection → Task 1; §2.2 chrome → Task 5 (+ Task 9 captions); §2.3 CSS → Task 2; §3 dock → Tasks 3–5; §4.1 Tonight → Task 2 (CSS-only, by design); §4.2 constellation → Tasks 7–8; §4.3 Board → Task 6; §4.4 → Task 9; §5 Phase D seam → recorded in Task 9 handoff text (no code by design); §6 testing → per-task tests + Task 10 gates/phone pass; §7 rollout → Tasks 9–10.
- **Placeholders:** none — every code step carries the full code; the one intentionally-flagged inline import (Task 4) instructs hoisting.
- **Type consistency:** `is_mobile()` (Tasks 1/5/6/8/9), `slip_shrinkage` (3/4), `bankroll_input(key=)` (3/4), `render_offer_cards(offers)` (6), `constellation_figure(..., mobile=)` + `_SIZE_MIN_MOBILE`/`_LABEL_FONT_SIZE_MOBILE` (7/8), `render_constellation(..., mobile=)` (8), customdata index 8 flag (7/8) — all aligned.
