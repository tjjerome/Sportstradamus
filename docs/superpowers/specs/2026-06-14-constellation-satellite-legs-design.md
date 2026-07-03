# Constellation satellite legs — design

Dashboard-UX lane, round 5 addendum. The constellation slip builder is same-game
only; this adds a way to pull legs from *other* games into a slip so a single-team
game can still be made into a valid parlay, plus a small constellation polish
(labels on candidate stars). Approved verbally before this doc.

## Problem

The constellation builder is strictly same-game: `_game_pool` returns an empty pool
the moment a slip spans more than one game, so the star map blanks out. A valid DFS
entry needs at least two distinct teams (`validate_parlay_legs`: distinct players +
≥2 teams). But a game's model-liked legs (Kelly `K` > 0) frequently land on **only
one of its two teams** — the constellation then shows one populated half and one
empty half, and there is no second team to satisfy the both-teams rule from within
that game. The owner reports this single-sided case is more common than expected.

Two user intents follow:

1. **Repair a single-team game** — add one leg from another team so the slip locks in.
2. **Combine two single-team parlays across games** — e.g. a NYK-only cluster from
   one game plus a BOS-only cluster from another, neither valid alone, valid together.

Separately, a small polish: candidate (not-in-slip) stars currently show no caption —
the owner wants the compact `Lastname MKT o/uLine` label on candidates too, not only
on active stars.

## Goals / non-goals

**Goals**
- From the slip page, pick legs with an edge (`K` > 0) from teams **not** in the
  current game, grouped by their game, and add one or several to the slip.
- Keep the constellation itself pure and same-game (DESIGN §4a grammar is FIXED):
  the cross-game legs live in a separate "satellite" section, never as stars on the
  map.
- Reuse the existing validity rule, slip scorer, and lock-in path unchanged.
- Show candidate-star labels.

**Non-goals**
- No correlation modelling between games (cross-game legs are independent; the
  block-diagonal scorer already treats them that way).
- No second constellation, no outer-orbit stars, no page switch to the Board.
- No change to the both-teams validity rule itself.
- No new player/headshot/last-5 data (that scar is a separate lane).

## Design

### 1. Focus game + satellites

Introduce one concept: the **focus game** — the single game the constellation
anchors on — distinct from the full multi-game slip.

- **Focus game = the game of the first (oldest) slip leg**, `legs[0]["Game"]`.
  Deterministic and stable: seeding a story sets every leg to that game, so it
  becomes the focus and stays the focus as you add other-game legs (they append
  after `legs[0]`). Removing `legs[0]` re-anchors the map to the new oldest leg —
  intuitive (you removed your anchor).
- A **satellite** is any slip leg whose `Game != focus`.
- `_game_pool` is generalized to `_focus_pool(legs, offers) -> (focus_game, pool)`:
  `focus = legs[0]["Game"]`; `pool = offers[(Game == focus) & (Platform == platform)]`.
  It no longer returns empty on a multi-game slip.
- **The constellation is drawn over the focus-game legs only.**
  `render_constellation_builder` passes `focus_legs = [l for l in legs if l["Game"]
  == focus]` to `constellation_figure`, not the whole slip — otherwise `_universe`
  would add satellite legs as stray stars in the focus map. The map's node set stays
  exactly the focus game's `K` > 0 pool (plus any active focus leg), unchanged from
  today.

### 2. Satellite picker — pure query

New module `dashboard/components/satellite_picker.py` (keeps `slip_builder.py`, already
~414 lines, from growing toward a monolith; the pure query is the testable unit).

```python
def satellite_groups(
    offers: pd.DataFrame,
    *,
    focus_game: str,
    platform: str,
    exclude_keys: set[str],
) -> list[tuple[str, list[dict]]]:
    """Other-game edge legs to offer, grouped by game, games ranked by best edge."""
```

Semantics:
- Filter `offers` to `K > 0`, `Platform == platform`, `Game != focus_game`.
- Drop any leg already in the slip (`corr_key(row) in exclude_keys`).
- Group by `Game`; within a game keep the top `_SATELLITE_PER_GAME_CAP` legs by `K`;
  order games by their single best `K` (descending) so the strongest options lead.
- Each leg dict carries the fields `_leg_from_offer` needs (it is a raw offer row),
  plus `corr_key` and a compact label for the chip.

`_SATELLITE_PER_GAME_CAP` is a named module constant (start at 6) with a one-line
rationale: enough to grab a small second cluster, few enough to stay uncluttered.

### 3. Satellite picker — render

```python
def render_satellites(
    offers: pd.DataFrame,
    *,
    focus_game: str,
    platform: str,
    legs: list[dict],
    key_prefix: str,
) -> dict | None:
    """Render the section; return an add/remove action for the caller to apply."""
```

- Wraps an `st.expander("Add a leg from another game")`. **Auto-open** (`expanded=`)
  when the slip is not yet valid (`not validate_parlay_legs(legs)[0]`) — i.e. the
  single-team / one-leg case — and collapsed when the slip is already valid. So it is
  always reachable for proactive cross-slate combining but quiet on games that do not
  need it.
- Inside, for each `(game, legs)` from `satellite_groups(..., exclude_keys=
  {corr_key(l) for l in legs})`: a small game header (e.g. `BOS / MIA`) and a row of
  add chips `+ Lastname o27.5 PTS`, each an `st.button`. A click returns
  `{"add": row}`.
- Above the picker, the **currently added satellites** (slip legs where `Game !=
  focus_game`) render as removable chips; a remove click returns
  `{"remove": index}` (the leg's index in `legs`).
- Returns `None` when nothing was clicked this run.

The render owns no slip state: it returns an action union, mirroring the constellation
component's `_apply_constellation_action` pattern already in `slip_builder.py`.

### 4. Builder wiring — `slip_builder.py`

In `render_constellation_builder`:

```python
focus, pool = _focus_pool(legs, offers)
focus_legs = [l for l in legs if l["Game"] == focus]
_render_constellation(focus_legs, corr, pool, key_prefix)
_render_leg_list(key_prefix, focus_game=focus)         # focus legs only
act = render_satellites(offers, focus_game=focus, platform=st.session_state[_PLATFORM], legs=legs, key_prefix=key_prefix)
if _apply_satellite_action(act, legs):
    st.rerun()
valid, reason = validate_parlay_legs(legs)
...
headline = slip_headline(focus_legs, offers, ctxs)     # focus legs only
```

- `_render_leg_list` gains a `focus_game` argument and renders **only focus-game legs**
  (plain text; the star toggles them). Satellite legs are listed/removed inside the
  satellite section, so each leg keeps one obvious removal control.
- `_apply_satellite_action(act, legs)`: `{"add": row}` → `legs.append(_leg_from_offer
  (row))`; `{"remove": i}` → `remove_leg(i)`; returns whether the slip changed (so the
  caller reruns). Mutation and `_leg_from_offer` stay in `slip_builder.py`, their owner.
- Validity, scoring, lock-in run over the **whole** `legs` list as today; only the
  constellation and the thesis headline are scoped to `focus_legs`.

### 5. Reuse, unchanged

- `validate_parlay_legs(legs)` — a single-team focus plus one satellite is two distinct
  teams, hence valid. The existing lock-in gate already runs over the full slip.
- `score_slip(legs, corr, ...)` — cross-game pairs have no `current_game_corr` entry,
  so the block-diagonal covariance treats them as independent. No scorer change; the
  payout (Sleeper = ∏ boosts) and Kelly are leg-count based and game-agnostic.
- `slip_headline` — computed over `focus_legs`: satellites add no same-game narrative,
  and feeding cross-game legs to the same-game thesis would muddy it.

### 6. Candidate-star labels — `constellation.py`

In `_add_node_trace`, candidates switch from `mode="markers"` to `mode="markers+text"`,
carrying the same compact `_star_label` caption as active stars, in `GRAY`. The
active/candidate distinction is still carried by star **fill color and opacity** (full
team color vs desaturated + dimmed), so the labels identify without flattening the
hierarchy. The one golden pin that asserts candidate traces use `markers` (no text) is
updated to expect `markers+text`.

## Data flow

`current_offers` (already passed into `render_constellation_builder` as `offers`) holds
every game's legs. `_focus_pool` slices the focus game for the map; `satellite_groups`
slices the complement (`Game != focus`, `K` > 0, same platform) for the picker. Both
read the same in-memory frame — no new load, no archive access. Added legs are
snapshotted via the existing `_leg_from_offer` and stored in `st.session_state
["slip_legs"]`, exactly like every other leg, so scoring and lock-in see no new leg
shape.

## Edge cases

- **Empty slip** — unchanged: the builder shows its "pick a story" info and returns
  before any focus/satellite logic.
- **Focus game has no other games on the slate** — `satellite_groups` returns `[]`;
  the expander shows a one-line "No other-game edge legs on {platform}." and the slip
  stays single-team (cannot lock in, as today).
- **Removing the first leg** — focus re-anchors to the new `legs[0]`; former focus legs
  become satellites and former satellites of the new focus stay satellites. Deterministic.
- **Combine two single-team clusters** — focus = the seeded game (its legs are oldest);
  the other game's legs are satellites. The map anchors the seeded cluster; the second
  cluster shows as satellite chips. Both validate together.
- **A satellite leg's offer moved/disappeared** — same as any leg: it was snapshotted at
  add time, so it survives in the slip; re-resolution on load already drops moved legs.

## Testing

Pure-function goldens (lane precedent — render paths are manual):

- `tests/golden/test_satellite_picker.py` (new) covering `satellite_groups`:
  - excludes the focus game and any already-slipped key;
  - keeps only `K` > 0 and the slip's platform;
  - caps to `_SATELLITE_PER_GAME_CAP` per game and ranks games by best `K`;
  - empty result when no other game qualifies.
- `tests/golden/test_constellation.py` — update the candidate-trace pin to expect
  `mode="markers+text"` with the label text present; keep all other node/edge pins.
- `tests/golden/test_dashboard_no_archive_lock.py` picks up the new module via its
  pkgutil auto-discovery; confirm `satellite_picker` has no module-level `Archive()`.

The expander, chips, and auto-open behavior are Streamlit-runtime and verified in the
manual walkthrough, not pytest (no AppTest pattern in-tree).

## Out of scope

- Headshots / last-5 / richer satellite cards (separate data lane).
- Modelling or displaying cross-game correlation.
- Any change to the both-teams validity rule, the scorer, or the lock-in/grading path.
- The Board's simple builder (untouched).

## Open items for the implementation plan

- Exact home of `_apply_satellite_action` and the `_render_leg_list` focus split inside
  `slip_builder.py`; confirm the file stays coherent (and whether the refactoring-
  specialist wants any extraction once the section lands).
- The candidate-label golden's current assertion location in `test_constellation.py`.
- Manual walkthrough script for the satellite flow (single-team repair + two-cluster
  combine) on a fresh `poetry run dashboard`.
