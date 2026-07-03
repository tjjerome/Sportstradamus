# P8 Phase 0 — Legacy Data-Model Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire every Google-Sheets-era string/tuple encoding from runtime: legs become one
structured record everywhere (pipeline → snapshots → grading → dashboard), the history snapshot
goes flat, the CLV probability slot stores an actual probability, and the parsers die.

**Architecture:** A single canonical leg schema (`leg_schema.py`, arrow `list<struct>` on disk /
`list[dict]` in memory — zero codec) replaces the `Desc` string round-trip, the `Leg 1..6` wide
columns, and the `"Player|MKT|Over (1.23x)"` correlation display strings. `history.parquet`
flattens to one row per offer with named columns, deleting the positional 9-tuple codec. A
one-shot idempotent migration script converts all historical parquets; `parse_leg`/`LEG_PATTERN`
survive only inside that script.

**Tech stack:** pandas + pyarrow parquet (existing), click + tqdm for the migration CLI.

**Branch:** work on `feature/dashboard-ux`.

---

## Context for a clean-context implementer

Read once before Task 0.1: `CLAUDE.md` (gates, refactoring-specialist, no monoliths),
`docs/STYLE_GUIDE.md`, `docs/handoffs/dashboard-ux.md` §5/§7. Rules that bind every task here:

- **The dashboard never opens the DuckDB archive.** Migration/ops scripts may (they run server-side).
- **Gates before "done"** (run after each task's commit, must be green):
  `poetry run ruff check src/sportstradamus/` · `poetry run pytest tests/golden/` ·
  `poetry run pytest -m integration -n0`.
- **refactoring-specialist subagent** runs on every touched `.py` before any push/PR/review.
- Copula internals in `prediction/parlay.py` are out of bounds (future lanes rebuild them); this
  plan only reshapes its **output rows**.
- Snapshot schema changes here are **breaking by design** — sanctioned via the migration script +
  coordinated deploy (see the runbook at the end). Do not add dual-format readers.

Why this exists: legs are currently strings built at `prediction/correlation.py:353-363`
(`"{Player} {Bet} {Line} {Market} - {WinProb}%, {Boost}x"`), rebuilt independently at
`dashboard/components/slip_state.py:77-79`, and parsed back in three places
(`prediction/stories/legs.py:178` `parse_leg`, `analysis.py:30` `LEG_PATTERN` for grading,
`slip_state.py:83-91` for slip restore). Parlay rows carry legs as six wide text columns.
`history.parquet` offers are positional 9-tuples. All of it dates from fitting data into Google
Sheets cells; the Sheets surface is gone.

### The canonical leg record

One schema for parlay legs, story legs, and user-slip legs. On disk: parquet `list<struct>`
(pandas: a column whose cells are `list[dict]` — this already round-trips natively; the history
`Offers` structs prove it). Never JSON strings.

```python
LEG_FIELDS = (
    "player",     # str  - display name, combo legs keep " + " / " vs. "
    "team",       # str
    "market",     # str  - CANONICAL market code (PTS, FG3M, "rushing yards"), never a book label
    "stat",       # str  - gamelog column key, resolved at WRITE time (kills grading-time stat_map lookups)
    "bet",        # str  - "Over" | "Under"
    "line",       # float
    "league",     # str
    "game",       # str  - canonical "AAA/BBB" matchup key
    "date",       # str  - ISO date of the game
    "platform",   # str  - "Underdog" | "Sleeper"
    "win_prob",   # float - model probability for the side
    "boost",      # float - per-leg payout multiplier
    "push_prob",  # float
    "kelly",      # float
)
```

`stat` note: `current_offers` already carries a `Stat` column (cli resolves it); parlay `bet_df`
rows carry `Market` + the league; where `Stat` is absent on a source row, resolve it at build time
via the same `stat_map` lookup grading uses today (`analysis._leg_market_map`) — the point is the
lookup happens **once at write**, never at grade/render.

---

### Task 0.1: `leg_schema.py` — the one leg codec

**Files:**
- Create: `src/sportstradamus/leg_schema.py`
- Test: `tests/golden/test_leg_schema.py`

- [ ] **Step 1: Write the failing test**

```python
"""Golden pins for the canonical structured-leg schema."""

import pandas as pd

from sportstradamus.leg_schema import LEG_FIELDS, build_leg, leg_label

OFFER_ROW = {
    "Player": "Luka Doncic",
    "Team": "LAL",
    "Market": "PTS",
    "Stat": "PTS",
    "Bet": "Over",
    "Line": 26.5,
    "League": "NBA",
    "Game": "LAL/GSW",
    "Date": "2026-07-03",
    "Platform": "Underdog",
    "Win Prob": 0.623,
    "Boost": 1.5,
    "Push Prob": 0.0,
    "Kelly": 0.12,
    "Extra Col": "ignored",
}


def test_build_leg_maps_offer_row_to_schema():
    leg = build_leg(OFFER_ROW)
    assert set(leg) == set(LEG_FIELDS)
    assert leg["player"] == "Luka Doncic"
    assert leg["market"] == "PTS"
    assert leg["stat"] == "PTS"
    assert leg["bet"] == "Over"
    assert leg["line"] == 26.5
    assert leg["date"] == "2026-07-03"
    assert leg["win_prob"] == 0.623


def test_build_leg_accepts_series():
    leg = build_leg(pd.Series(OFFER_ROW))
    assert leg["game"] == "LAL/GSW"


def test_leg_label_renders_without_stored_string():
    leg = build_leg(OFFER_ROW)
    assert leg_label(leg) == "Luka Doncic Over 26.5 PTS"


def test_legs_round_trip_parquet(tmp_path):
    legs = [build_leg(OFFER_ROW)]
    df = pd.DataFrame({"legs": [legs]})
    p = tmp_path / "t.parquet"
    df.to_parquet(p)
    back = pd.read_parquet(p)["legs"].iloc[0]
    assert list(back[0].keys()) == list(LEG_FIELDS)
    assert back[0]["line"] == 26.5
```

- [ ] **Step 2: Run it — expect FAIL (module not found)**

Run: `poetry run pytest tests/golden/test_leg_schema.py -v`

- [ ] **Step 3: Implement `src/sportstradamus/leg_schema.py`**

```python
"""Canonical structured-leg schema shared by prediction, grading, and dashboard.

One leg = one dict with LEG_FIELDS keys; stored in parquet as list<struct>
columns (pandas cells of list[dict]). Replaces the retired Desc string
round-trip — display strings are rendered on demand via leg_label, never stored.
"""

from collections.abc import Mapping

LEG_FIELDS = (
    "player",
    "team",
    "market",
    "stat",
    "bet",
    "line",
    "league",
    "game",
    "date",
    "platform",
    "win_prob",
    "boost",
    "push_prob",
    "kelly",
)

_OFFER_KEYS = {
    "player": "Player",
    "team": "Team",
    "market": "Market",
    "stat": "Stat",
    "bet": "Bet",
    "league": "League",
    "game": "Game",
    "platform": "Platform",
}


def build_leg(row: Mapping) -> dict:
    """Snapshot an offer row (dict or Series) into one canonical leg record."""
    leg = {field: str(row[col]) for field, col in _OFFER_KEYS.items()}
    leg["stat"] = str(row.get("Stat") or row["Market"])
    leg["date"] = str(row["Date"])[:10]
    leg["line"] = float(row["Line"])
    leg["win_prob"] = float(row["Win Prob"])
    leg["boost"] = float(row.get("Boost", 1.0) or 1.0)
    leg["push_prob"] = float(row.get("Push Prob", 0.0) or 0.0)
    leg["kelly"] = float(row.get("Kelly", 0.0) or 0.0)
    return leg


def leg_label(leg: Mapping) -> str:
    """Human display string for a leg. Render-only — never parsed back."""
    return f"{leg['player']} {leg['bet']} {leg['line']:.10g} {leg['market']}"
```

(Phase A upgrades `leg_label` to route `market` through `market_display_name`; keep the plain
slug here.)

- [ ] **Step 4: Run tests — expect PASS.** `poetry run pytest tests/golden/test_leg_schema.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/leg_schema.py tests/golden/test_leg_schema.py
git commit -m "feat(p8-0): canonical structured-leg schema (leg_schema.py)"
```

---

### Task 0.2: Pipeline writers emit structured legs; Desc build + dead columns die

**Files:**
- Modify: `src/sportstradamus/prediction/parlay.py` (output-row block ~509-543 only)
- Modify: `src/sportstradamus/prediction/correlation.py` (Desc build 353-363; dedup 393; `Fun`
  ranking site ~501; `Family` write ~507)
- Modify: `src/sportstradamus/prediction/cli.py` (~262 `Legs` reset)
- Modify: `src/sportstradamus/prediction/persist.py` (`_PARLAY_DROP_COLS` / keep-cols)
- Tests: existing parlay/persist characterization goldens + integration

Key edits (verbatim targets):

1. `correlation.py:353-363` — delete the whole `league_df["Desc"] = ...` block. The frame keeps
   its real columns; nothing downstream needs the string.
2. `correlation.py:393` — `game_df.drop_duplicates(subset="Desc")` becomes
   `game_df.drop_duplicates(subset=["Player", "Market", "Bet", "Line"])` (the same identity the
   Desc string encoded).
3. `parlay.py:513-543` — replace the `Leg 1..6` / string-`Legs` / `Leg Probs` emission:

```python
    bet = itemgetter(*bet_id)(bet_df)
    display_boost = boost if legacy else payout
    parlay_dict = info | {
        "Model EV": p,
        "Market EV": pb,
        "Boost": display_boost,
        "Rec Bet": units,
        "legs": [build_leg(leg) for leg in (bet if isinstance(bet, tuple) else (bet,))],
        "Bet ID": bet_id,
        "P": prev_p,
        "PB": prev_pb,
        "Bet Size": bet_size,
        "Corr Pairs": tuple(SIG[np.triu_indices(bet_size, 1)]),
        "Boost Pairs": tuple(M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size, 1)]),
        "Indep P": float(np.prod(p_model[np.ix_(bet_id)]) * payout),
        "Indep PB": float(np.prod(p_books[np.ix_(bet_id)]) * payout),
    }
    return parlay_dict
```

   `from sportstradamus.leg_schema import build_leg` at module top. `bet_df` rows are offer
   dicts — verify they carry `Stat`/`Kelly` (they come off the scored offers frame; if `Stat` is
   absent add it to the frame the caller builds in `correlation.py`, do not reach into stat_map
   here). Per-leg `win_prob`/`boost` now live inside `legs`, so `Leg Probs` dies. `Fun` leaves
   the emitted row; `_parlay_fun` stays only if `correlation.py:501` still ranks with it — check:
   if that ranking reads the in-memory column, compute it there locally; if not, move
   `_parlay_fun` to `src/deprecated/`.
4. `correlation.py:507` — delete the `Family` assignment; grep `Family` repo-wide first
   (`grep -rn '"Family"' src/ tests/`) and clean any dead references (verified zero consumers at
   plan time).
5. `cli.py:262` — `parlay_df[["Legs", "Misses", "Profit"]] = np.nan` → rename the count column:
   the frame gains `"Legs Resolved"` + `"Misses"` + `"Profit"` as the nightly-owned columns;
   nothing writes a column literally named `Legs` anymore.
6. `persist.py` — update `_PARLAY_DROP_COLS`/keep logic: `current_parlays` keeps `legs`
   (list<struct>), drops `Bet ID`; `parlay_hist` continues to persist `Corr Pairs`/`Boost Pairs`
   (raw material for the parlay-dependence lane). Update the persist characterization pins.

- [ ] Write/adjust failing pins first: parlay characterization test asserting the new row shape
  (has `legs` list of dicts with `LEG_FIELDS`, no `Leg 1`, no string `Legs`, no `Fun`/`Family`).
- [ ] Implement edits 1-6.
- [ ] `poetry run pytest tests/golden/ -k "parlay or persist or correlation" -v` green.
- [ ] `poetry run pytest -m integration -n0` — inspect that `current_parlays.parquet` in the fake
  run carries `legs` structs.
- [ ] Commit: `feat(p8-0): parlays emit structured legs; Desc/Leg1-6/Fun/Family retired`

---

### Task 0.3: Stories path consumes leg dicts

**Files:**
- Modify: `src/sportstradamus/prediction/stories/thesis.py` (~52-58)
- Modify: `src/sportstradamus/prediction/stories/menu.py` (~325-345 `_story_prose`, and the
  `legs` JSON written to `current_game_stories`)
- Modify: `src/sportstradamus/prediction/stories/legs.py` (`enrich_legs` input adapter;
  `parse_leg` stays until Task 0.8 removes it)

Edits:

1. `thesis.py` — replace the `leg_cols` scan + `parse_leg` with the structured column:

```python
    for _, row in parlays.iterrows():
        legs = enrich_legs(
            [
                {"Player": lg["player"], "Bet": lg["bet"], "Line": lg["line"], "Market": lg["market"]}
                for lg in (row.get("legs") or [])
            ],
            offers,
        )
```

   Better: change `enrich_legs` to accept the canonical lowercase keys directly and update its
   other caller in the same commit — do NOT keep two key spellings alive. Pick the canonical
   lowercase form everywhere inside `stories/`.
2. `menu.py` — `parse_leg(sctx.bet_df[i]["Desc"])` becomes `build_leg(sctx.bet_df[i])`-shaped
   access (the bet_df rows now have no Desc; construct the enrich input from the row fields
   directly). `current_game_stories.legs` becomes the same `list<struct>` (persist writes it
   unchanged through `_atomic_write_parquet`).
3. `dashboard/components/slip_state.py:seed_from_story` currently `json.loads(legs_json)` — flag
   here, fixed in Task 0.5 (stories now hand it a list of dicts, not JSON).

- [ ] Repoint `test_story_menu` / thesis determinism goldens to struct fixtures (headline output
  must be byte-identical for the same leg set — the md5 seed keys on player/market/bet tuples,
  which are unchanged).
- [ ] Gates green; commit: `feat(p8-0): stories consume structured legs`

---

### Task 0.4: Grading on structs — the regex dies from the hot path

**Files:**
- Modify: `src/sportstradamus/analysis.py` (`_resolve_leg` 124-143, `check_bet` 146-175;
  `LEG_PATTERN` deleted in Task 0.8)
- Modify: `src/sportstradamus/nightly.py` (`_resolve_parlays` ~436-453, `_resolve_user_slips`
  ~495-505)
- Test: `tests/golden/` grading characterization

New `_resolve_leg` (drop `new_map` entirely — `leg["stat"]` was resolved at write time):

```python
def _resolve_leg(game, ls, leg: Mapping):
    """Resolve one structured leg to 1 (miss), 0 (hit), or None (skip/push)."""
    stat = leg.get("stat") or leg.get("market")
    result_val = _leg_result_value(game, ls, leg["player"], stat)
    if result_val is None or result_val == leg["line"]:
        return None
    over = leg["bet"] == "Over"
    missed = (over and result_val < leg["line"]) or (not over and result_val > leg["line"])
    return 1 if missed else 0
```

`check_bet` iterates `bet.legs` (the struct column) instead of scanning `Leg N` attributes; its
`(legs, misses)` return contract is unchanged but the caller in `nightly._resolve_parlays` writes
`Legs Resolved` (Task 0.2 rename). `_leg_market_map` loses its grading caller — if
`build_leg`-time resolution (Task 0.2) is its only remaining use, relocate it beside the writer;
zero-caller → `src/deprecated/`.

`nightly._resolve_user_slips` passes `leg` dicts straight through (user_slips legs are structs
after Task 0.5 + migration).

- [ ] Rewrite grading characterization pins on struct fixtures first (hit/miss/push/combo-player
  cases carried over verbatim from the old string fixtures).
- [ ] Gates green; commit: `feat(p8-0): grading resolves structured legs, no regex`

---

### Task 0.5: Dashboard slip state, shelf, offer lookup, headline

**Files:**
- Modify: `src/sportstradamus/dashboard/components/slip_state.py`
- Modify: `src/sportstradamus/dashboard/components/locked_shelf.py` (line 75)
- Modify: `src/sportstradamus/dashboard/legs.py` (`find_offer_idx`, delete `_candidate_markets`)
- Modify: `src/sportstradamus/dashboard/slip_engine.py` (`slip_headline` ~149-156)
- Modify: `src/sportstradamus/dashboard/surfaces/lab_correlations.py` (its `Leg Probs` fallback
  read repoints to `legs[].win_prob`)
- Tests: `test_user_slips_io`, slip-engine + headline goldens

Edits:

1. `slip_state._leg_from_offer` → `return build_leg(row)` plus the constellation's extra needs:
   keep `_SNAPSHOT_COLS` semantics by confirming every consumer of the old uppercase keys
   (`leg["Game"]`, `leg["Kelly"]`, `leg["Player"]`…) across `slip_builder.py`,
   `constellation.py`, `satellite_picker.py`, `slip_engine.py`, `locked_shelf.py` is repointed to
   the lowercase schema keys in this same task (grep `leg\[["']` across `dashboard/`). One
   spelling, everywhere.
2. `lock_in` stores `"legs": legs` (the list of structs — no `json.dumps`; parquet handles it).
3. `_legs_from_descs` → `_legs_from_records(legs, platform, offers)`: for each stored leg,
   `find_offer_idx(leg, offers, platform)` re-snapshots the live row (`build_leg`) so prices
   refresh; unmatched legs drop (unchanged behavior).
4. `find_offer_idx(leg, offers, platform)` masks on
   `Player == leg["player"] / Bet == leg["bet"] / Market == leg["market"] / Line == leg["line"]`
   — canonical market equality, `_candidate_markets` and the stat_map import die.
5. `locked_shelf.py:75` → `st.write(f"- {leg_label(leg)}  ·  {leg.get('league', '')}")`.
6. `slip_engine.slip_headline` sorts/consumes leg dicts directly (no parse).

- [ ] Update goldens first (user_slips round-trip with struct legs; headline determinism on
  dict legs; find_offer_idx platform pin stays — the +2530% EV regression test must keep passing).
- [ ] Gates green; commit: `feat(p8-0): dashboard slip path on structured legs`

---

### Task 0.6: Correlation display structs replace the `"Desc (1.23x)"` strings

**Files:**
- Modify: `src/sportstradamus/prediction/correlation.py` (`_annotate_correlation_columns` 425-455)
- Modify: `src/sportstradamus/prediction/persist.py` (`_OFFER_KEEP_COLS`: swap
  `Team Correlation`/`Opp Correlation` → `Corr Same`/`Corr Opp`)
- Modify: `src/sportstradamus/dashboard/components/deep_dive.py` (`_parse_corr` dies;
  `_render_corr_tab` 347-353 reads structs)
- Tests: annotate golden + deep-dive corr-tab unit

`_annotate_correlation_columns` emits, per offer row:

```python
        df.at[mask_index, "Corr Same"] = [
            {
                "player": r["Player"],
                "market": r["Market"],
                "bet": r["Bet"],
                "line": float(r["Line"]),
                "mult": round(float(m), 2),
            }
            for (_, r), m in zip(same.iterrows(), same["Corr Mult"])
        ]
```

(and likewise `Corr Opp` from `other`). Implementation detail: `.at` with list values needs the
column pre-created with `object` dtype — mirror how the string columns are initialized today at
`correlation.py:691-692` (`df["Corr Same"] = None`). The `(EV/C/EVb)` display gates and
top-1-per-player logic stay identical.

`deep_dive._render_corr_tab` consumes the dicts (player/market/bet/line/mult) directly;
`_parse_corr` and `_find_corr_row_idx`'s string sniffing are deleted. The Correlated tab's row
rendering keeps its current card layout — the P8 Phase C plan rebuilds the tab visuals; here we
only swap the data source.

- [ ] Goldens first (annotate emits structs w/ gates honored; corr tab renders struct fixtures).
- [ ] Gates green; commit: `feat(p8-0): correlation partners as structs, string parser retired`

---

### Task 0.7: history.parquet goes flat (one row per offer)

**Files:**
- Modify: `src/sportstradamus/history_schema.py` (new flat schema + `Alt Line`)
- Modify: `src/sportstradamus/helpers/io.py` (delete the offers codec:
  `_OFFER_FIELDS` tuple mapping, `_offer_dict_to_tuple`/`_offer_tuple_to_dict`,
  `_offers_for_parquet`/`_offers_from_parquet`, legacy padding)
- Modify: `src/sportstradamus/analysis.py` (delete `explode_offers`, `_migrate_flat_history`,
  `_prep_legacy_columns`, `_offer_tuple`, `_group_offers`, `_dedup_offers`, `_merge_offers`
  tuple surgery — replaced by frame-level upsert)
- Modify: `src/sportstradamus/clv.py` (`_fill_offer` tuple surgery → vectorized column assign)
- Modify: `src/sportstradamus/prediction/cli.py` (history append path writes flat rows + stamps
  `Alt Line`)
- Modify: `src/sportstradamus/dashboard/data.py` (`get_filtered_history` drops the explode;
  `get_prediction_history` = `drop_duplicates` on the prediction key)
- Tests: history round-trip + upsert goldens, `test_clv_close_sanity`

Flat schema (one row per (prediction × book offer)):

```python
# history_schema.py
PREDICTION_KEY = ["Player", "League", "Team", "Date", "Market"]
PREDICTION_LEVEL_COLS = [...existing list unchanged...]
OFFER_LEVEL_COLS = [
    "Line", "Boost", "Platform", "Bet",
    "Win Prob", "Market Prob",
    "Close Market Prob", "Market CLV", "Model CLV",
    "Alt Line",           # bool: |Line − Consensus Line| > tolerance, stamped at write
]
```

Upsert semantics preserved: today `_merge_offers` dedups by `(Line, Platform)` within a
prediction; flat equivalent = `drop_duplicates(subset=[*PREDICTION_KEY, "Line", "Platform"],
keep="last")` after concat, **except** rows whose closing trio is already filled are protected
(mirror the existing "new offers overwrite except closing fields" rule — port the rule, then pin
it in a golden).

`Alt Line` stamping in `cli.py`'s history-append: with `Consensus Line` already on the offers
frame, `alt = (offers["Line"] - offers["Consensus Line"]).abs() > _ALT_LINE_TOL(market)`.
Tolerance constants (named, module-level, one-line reasons — STYLE_GUIDE §9):

```python
# cli.py — flag ladder/alt rungs vs the standard line; count stats move in 0.5 steps,
# continuous (yardage) lines drift ±1-2 without being a different rung.
_ALT_LINE_TOL_COUNT = 0.75
_ALT_LINE_TOL_CONTINUOUS = 2.5
```

**Flag these two values for owner review in the PR description** — they gate the Phase B
calibration split.

`clv.fill_from_archive` becomes a groupby over `PREDICTION_KEY` on rows where
`Close Market Prob` is NaN: one archive read per prediction, assigned to that group's rows via
plain column ops (semantic fix is Task 0.9; this task only removes the tuple mechanics).

- [ ] Goldens first: flat round-trip, upsert-protects-closing-fields, `get_prediction_history`
  dedup parity vs a grouped fixture.
- [ ] Gates green; commit: `feat(p8-0): history.parquet flat long-format, offers codec deleted`

---

### Task 0.8: Migration script + parser deletion + klepto removal

**Files:**
- Create: `scripts/migrate_leg_schema.py`
- Modify: `src/sportstradamus/prediction/stories/legs.py` (delete `parse_leg`),
  `src/sportstradamus/analysis.py` (delete `LEG_PATTERN`), `src/sportstradamus/dashboard/legs.py`
  (delete the `parse_leg` re-export)
- Modify: `pyproject.toml` (drop `klepto` line 34); move `scripts/migrate_archive_to_duckdb.py`
  → `src/deprecated/` with the archive header
- Test: `tests/golden/test_migrate_leg_schema.py` (synthetic old-schema fixtures, all four frames,
  idempotency: run twice → byte-equal)

Script skeleton (click + tqdm, `--dry-run`, atomic writes via the io helpers; the ONLY place the
old parsers survive, as frozen private copies):

```python
"""One-shot migration: Sheets-era leg strings/tuples -> structured schema.

Converts parlay_hist (Leg 1..6 -> legs structs), user_slips (desc JSON ->
legs structs), history (nested Offers tuples -> flat rows, 'nan' platform
normalized, optional Alt Line / closing-prob backfill from the archive),
and the current_* snapshots. Idempotent: each frame is gated on an
unambiguous old-schema marker and skipped when already migrated.
"""

_LEG_PATTERN = re.compile(r"^(.+?)\s+(Over|Under)\s+([\d.]+)\s+(.+?)\s+-\s+[\d.]+%")  # frozen copy


def _parse_desc(desc, league, game, date, platform):
    m = _LEG_PATTERN.match(desc or "")
    if not m:
        return None
    player, bet, line, market = m.groups()
    return {
        "player": player, "team": "", "market": market.replace("H2H ", ""),
        "stat": _stat_for(league, platform, market), "bet": bet, "line": float(line),
        "league": league, "game": game, "date": date, "platform": platform,
        "win_prob": float("nan"), "boost": float("nan"),
        "push_prob": float("nan"), "kelly": float("nan"),
    }
```

Order of operations (each step prints counts; `--dry-run` prints and writes nothing):

1. `parlay_hist.parquet` — gate: `"Leg 1" in columns`. Build `legs` from `Leg 1..6` +
   `Leg Probs[i]` → `win_prob`; drop `Leg 1..6`, string remnants of `Legs`, `Fun`, `Family`,
   `Leg Probs`; rename count `Legs` → `Legs Resolved`.
2. `history.parquet` — gate: `"Offers" in columns`. Explode to flat (inline the old
   `explode_offers` one last time), normalize `platform == "nan"` → `None`,
   quarantine `Close Market Prob > 1` → closing trio NaN (print count; ~35k expected),
   `--backfill-close` re-derives closing probs from the archive (Task 0.9 helper),
   `--backfill-alt-lines` stamps `Alt Line` from `archive.get_line` history where recoverable.
3. `user_slips.parquet` — gate: first leg record has a `"desc"` key. `_parse_desc` each.
4. `current_offers/current_parlays/current_game_stories` — same conversions so the dashboard
   never dual-reads (`Team Correlation`→`Corr Same` structs via one inlined `_parse_corr` copy).
5. Print a summary table; exit nonzero if any frame was left partially migrated.

Deletion sweep after the script lands: `grep -rn "parse_leg\|LEG_PATTERN" src/` must return only
`scripts/migrate_leg_schema.py`. Then drop klepto (`poetry lock` note for devel-ship-curator).

- [ ] Fixtures + idempotency golden first; script; deletion sweep; gates green.
- [ ] Commit: `feat(p8-0): one-shot leg-schema migration; parsers + klepto retired from runtime`

---

### Task 0.9: CLV semantic fix — a probability in the probability slot

**Files:**
- Modify: `src/sportstradamus/clv.py`
- Test: extend `tests/golden/test_clv*.py`

**The bug (verified 2026-07-03):** `fill_from_archive` writes `archive.get_ev(...)` — the book's
projected **stat mean** — into `Close Market Prob`; 35,263/35,923 close-valued offers in
production history have "probability" > 1.5, and `Market CLV`/`Model CLV` subtract probabilities
from means. Receipts CLV tiles are noise today.

**Fix contract:** `Close Market Prob` = the book-consensus probability of the **offer's side at
the offer's line** at commence time. Two existing helpers make this line-exact with no new
plumbing — reuse, don't build:

```python
close_ev = _safe_get_ev(archive, league, market, date_str, player, at=commence_at)
# get_odds(line, ev, dist, cv) is the SAME conversion the pipeline uses at open:
# it returns P(under) for a book quote with mean `ev` under the row's distribution.
close_under = get_odds(offer_line, close_ev, row["Dist"], cv=row["CV"])
close_p = close_under if bet == "Under" else 1.0 - close_under
```

(`Dist`/`CV` are prediction-level history columns; `get_odds` lives in `sportstradamus.helpers`.
Alternative considered: `archive.get_composite_under_prob(..., at=)` reads the WS1 shape-free
`under_prob` directly — line-INEXACT when books moved off the offer's line. Prefer the
`get_odds` conversion; note the composite as fallback when `Dist`/`CV` are NaN.)

CLV definitions become coherent automatically (`_signed_clv` unchanged): Market CLV =
sign · (close_p − Market Prob), Model CLV = sign · (close_p − Win Prob). Migration `--backfill-close`
(Task 0.8) reuses this helper to repopulate historical rows where the archive still holds the
close snapshot; unrecoverable rows stay NaN-quarantined.

- [ ] Golden first: synthetic archive stub → close_p ∈ [0,1]; an Over and an Under case
  hand-computed; a `close_p > 1` write must be impossible (assert raises/clamps NaN).
- [ ] Gates green; commit: `fix(p8-0): CLV stores closing probability, not book mean`

---

## Deploy runbook (owner-run, server-side)

1. Merge Phase 0 to `devel`; server `git pull` in the cron dead zone (between the 20:50
   prophecize and the 23:00 reflect, or disable cron for the window).
2. `poetry install` (klepto drop), then
   `poetry run python scripts/migrate_leg_schema.py --backfill-close --backfill-alt-lines`
   (add `--dry-run` first; the script refuses to run if a writer lock/temp file is present).
3. Restart the dashboard service. Next `prophecize` writes new-schema snapshots natively;
   `STORIES_VERSION` bump reshuffles headlines once (established precedent).

## Exit criteria (whole plan)

- `grep -rn "parse_leg\|LEG_PATTERN\|\"Desc\"\|'Desc'" src/sportstradamus/` → only the migration
  script (and archived files under `src/deprecated/`).
- No runtime module indexes an offer or leg positionally.
- Three gates green; refactoring-specialist run on every touched `.py`.
- Migration idempotency golden green; dry-run output pasted into the PR.
