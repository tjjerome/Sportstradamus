# Odds Archive Time-Series Rework

## Context

The archive on `devel` stores book EVs and lines keyed by `(league, market, game_date, entity, book)` in DuckDB with **no timestamp column** — every poll overwrites the previous EV per book, and the `lines` table only retains distinct line values. CLV in `clv.py` reads "whatever was in the archive at `reflect` time" as the closing probability and computes `Market CLV = sign * (close_p − open_p)` against the open probability stored on each offer at placement.

This is naive in two specific ways:

1. The "close" we read is whatever the Odds API last surfaced — possibly 30 min before kickoff, possibly several hours. The `clv.py` module docstring already concedes this. With no `observed_at`, we can't validate it.
2. We capture **no line movement information at all**. We can't tell whether the market moved **toward** the model's prediction (a signal of edge) or **away** from it (a signal of overconfidence) — neither is computable from the current schema. Roadmap §1.3 originally proposed a closing-line freeze; Phase 3 deferred it as monitor-only. This rework picks that thread back up with proper time-series.

The rework adds `observed_at` to the DuckDB schema, switches writes from per-book overwrite to append-only, exposes new APIs for history / movement / point-in-time queries, makes training pipeline reads use a configurable lookback (default 8 h before commence_time) instead of "latest", and migrates existing single-point rows to a sentinel timestamp so every current reader keeps working unchanged.

## Storage Choice — DuckDB Stays

Edge Suite §2.5 recommends TimescaleDB for ~500K rows/day across NFL/NBA/MLB at 60-second polling. Our actual cadence (`confer` once daily plus `confer --close-lines` every 5 min during a 30-min pre-kickoff window) projects ~50K rows/day — about 1/10th the Edge Suite scenario. Devel is also fully invested in DuckDB: `helpers/archive.py`, `scripts/migrate_archive_to_duckdb.py`, `tests/integration/test_end_to_end.py`, and the Phase 3 plan all assume it. Postgres would add a service to operate, network hop, and auth concerns for no measurable benefit at this scale. If volume grows past ~5 GB or we need multi-writer concurrency from a hosted dashboard later, the schema below ports to TimescaleDB cleanly via Parquet export. Retention: keep everything; revisit at 5 GB.

## Branch Setup

The current branch was cut off `origin/main` by mistake. Reset onto `devel` before any code work:

```
git fetch origin devel
git reset --hard origin/devel
git push --force-with-lease origin claude/odds-history-tracking-ezeOy
```

`--force-with-lease` is safe — the branch had no unique commits.

## Schema Changes

**File:** `src/sportstradamus/helpers/archive.py:38-55` (`_SCHEMA_DDL`)

Add `observed_at TIMESTAMP NOT NULL` to both tables:

```sql
CREATE TABLE IF NOT EXISTS odds (
    league       TEXT NOT NULL,
    market       TEXT NOT NULL,
    game_date    DATE NOT NULL,
    entity       TEXT NOT NULL,
    book         TEXT NOT NULL,
    ev           DOUBLE,
    observed_at  TIMESTAMP NOT NULL    -- NEW
);
CREATE TABLE IF NOT EXISTS lines (
    league       TEXT NOT NULL,
    market       TEXT NOT NULL,
    game_date    DATE NOT NULL,
    entity       TEXT NOT NULL,
    line         DOUBLE NOT NULL,
    observed_at  TIMESTAMP NOT NULL    -- NEW
);
```

Disk sort order becomes `(league, market, game_date, entity, book, observed_at)` for `odds` (analogous for `lines`) — keeps existing zone-map pruning effective and lets recent-time predicates scan contiguous blocks.

## Write-Path Changes

**File:** `src/sportstradamus/helpers/archive.py:265-470`

- `_stage_book_ev`, `_stage_line` — append `observed_at = datetime.utcnow()` to each staged tuple.
- `write()` — drop the DELETE-then-INSERT per-book overwrite for `odds`; pure INSERT. Drop the anti-join dedup for `lines`; pure INSERT. Drop `_replace_keys` and the `set_team_books` replace path (append semantics make it redundant).
- `add_dfs` — no signature change; just routes through the modified stagers.
- `merge_player_books`, `set_team_books` — no signature change; same.

Result: every `confer` poll inserts new rows; nothing is overwritten. `confer --close-lines` running every 5 min during the closing window naturally produces a 5-min-resolution movement series.

## Read-Path Changes (Backwards Compatible)

**File:** `src/sportstradamus/helpers/archive.py:130-260`

All existing read methods gain an optional `at: datetime | None = None` keyword. Default behavior preserves current semantics:

```python
def get_ev(self, league, market, date, player, *, at=None) -> float
def get_line(self, league, market, date, player, *, at=None) -> float
def get_team_market(self, league, market, date, team, *, at=None) -> float
def get_moneyline(self, league, date, team, *, at=None) -> float
def get_total(self, league, date, team, *, at=None) -> float
```

- `at=None` for odds: most recent observation per book at-or-before `now()`. Reuses `_weighted_book_ev` on the latest-per-book result.
- `at=ts` for odds: most recent observation per book at-or-before `ts`, via `QUALIFY ROW_NUMBER() OVER (PARTITION BY book ORDER BY observed_at DESC) = 1`.
- `at=None` for lines: existing consensus median over distinct lines for the day (preserved).
- `at=ts` for lines: median over distinct lines observed at-or-before `ts` (new).

`to_pandas` and `archived_players_by_date` need no signature change — but `to_pandas` should select the latest EV per book (matches current "one EV per book" semantics) so book-weight calibration in `training/calibration.py:fit_book_weights` keeps producing the same numbers it would have without the rework.

## New History APIs (Additive)

**File:** `src/sportstradamus/helpers/archive.py` (new methods)

```python
def get_line_history(self, league, market, date, entity, *,
                     books=None, since=None, until=None) -> pd.DataFrame
    # Returns: [observed_at, book, line] sorted by observed_at.

def get_ev_history(self, league, market, date, entity, *,
                   books=None, since=None, until=None) -> pd.DataFrame
    # Returns: [observed_at, book, ev] sorted by observed_at.

def get_movement(self, league, market, date, entity, *,
                 books=None, until=None) -> dict
    # Returns: {open_line, open_ev, close_line, close_ev,
    #           n_obs, n_moves, peak_line, trough_line, time_span_minutes}.
    # Used by the closing-line bias diagnostic in clv.py.
```

No callers required to use these for the system to keep working; they exist for new diagnostics.

## Training Pipeline Change — 8 h Pre-Kickoff Lookup

**Files:** `src/sportstradamus/stats/{base,mlb,nba,nfl,nhl,wnba}.py`, `src/sportstradamus/training/pipeline.py`

Each `Stats` subclass already loads start times in its gamelog (MLB `gameDate`, NBA `GAME_DATE_EST` + start time, NFL `gametime`, NHL `startTimeUTC`). Add a `game_time` column to the matrix returned by `Stats.get_training_matrix(market)`, sourced from those columns. Where unknown (very old rows), fall back to `game_date` midnight in the league's local timezone.

In the loop in `stats/base.py:1042-1063` that calls `archive.get_ev(...)` and `archive.get_line(...)` to populate training features, switch to:

```python
# In a module-level constants block:
TRAINING_LOOKBACK_HOURS: int = 8        # offset before commence_time for "the books' line"
TRAINING_LOOKBACK = timedelta(hours=TRAINING_LOOKBACK_HOURS)

# In the loop:
target_at = row["game_time"] - TRAINING_LOOKBACK
ev   = archive.get_ev(league, market, date, player, at=target_at)
line = archive.get_line(league, market, date, player, at=target_at)
```

Legacy single-point rows (one observation at `game_date` midnight) are always ≤ `target_at` for any game starting after 08:00 local, so they keep returning the same value — no NaN regressions.

## Prediction Pipeline — No Change

**File:** `src/sportstradamus/prediction/model_prob.py:140-141`

Live `prophecize` keeps reading "the line right now" via `at=None`. No code change.

## CLV Update — Pinned Closing Read + Movement Diagnostic

**File:** `src/sportstradamus/clv.py`

Two changes, both small:

1. `_safe_get_ev(archive, league, market, date, player)` becomes `_safe_get_ev(archive, league, market, date, player, *, at)` and `fill_from_archive` looks up `commence_time` per row (from the same source the training pipeline uses) and passes `at=commence_time`. The closing read is now reproducible regardless of when `reflect` runs.
2. Add a small per-segment diagnostic in `summarize`: for each (league, market, platform), call `archive.get_movement(...)` per offer and report `frac_lines_moved_toward_model` (sign of `close_line − open_line` matches sign of `model_ev − open_line`). Directly answers "did the market move toward me or away from me".

## Migration

**New file:** `src/sportstradamus/scripts/add_observed_at_to_archive.py`

Idempotent in-place ALTER on the existing `archive/archive.duckdb`:

```sql
ALTER TABLE odds  ADD COLUMN IF NOT EXISTS observed_at TIMESTAMP;
ALTER TABLE lines ADD COLUMN IF NOT EXISTS observed_at TIMESTAMP;
UPDATE odds  SET observed_at = CAST(game_date AS TIMESTAMP) WHERE observed_at IS NULL;
UPDATE lines SET observed_at = CAST(game_date AS TIMESTAMP) WHERE observed_at IS NULL;
-- Recreate sorted (per the existing migrate_archive_to_duckdb.py pattern)
CREATE TABLE odds_sorted  AS SELECT * FROM odds  ORDER BY league, market, game_date, entity, book, observed_at;
DROP TABLE odds;  ALTER TABLE odds_sorted  RENAME TO odds;
-- same for lines
ANALYZE; CHECKPOINT;
```

Backs up `archive.duckdb` to `archive.duckdb.bak-<unix-ts>` before mutating. Refuses to run twice (checks for the column).

Also extend `src/sportstradamus/scripts/migrate_archive_to_duckdb.py:_iter_league` to emit `observed_at = game_date midnight` so a fresh klepto→duckdb migration produces a valid time-series schema from day one.

## Critical Files

| File | Change |
|---|---|
| `src/sportstradamus/helpers/archive.py` | Schema, write path, read path, new history/movement APIs, `TRAINING_LOOKBACK_HOURS` |
| `src/sportstradamus/stats/base.py:990-1090` | Add `game_time` to training matrix; pass `at=game_time − 8h` to archive reads |
| `src/sportstradamus/stats/{mlb,nba,nfl,nhl,wnba}.py` | Surface league-specific start time into the matrix |
| `src/sportstradamus/clv.py` | Pass `at=commence_time` to `get_ev`; add movement diagnostic in `summarize` |
| `src/sportstradamus/scripts/add_observed_at_to_archive.py` | NEW — one-shot ALTER + backfill |
| `src/sportstradamus/scripts/migrate_archive_to_duckdb.py` | Emit `observed_at = game_date midnight` from legacy klepto flow |
| `tests/test_archive_history.py` | NEW — unit tests for all three time modes |
| `tests/integration/test_end_to_end.py:77-89` | Tighten to verify `observed_at` increments across two confer runs |

No changes needed in `moneylines.py`, `prediction/model_prob.py`, `prediction/scoring.py`, or `training/calibration.py` — they all route through `Archive` and the default `at=None` preserves their behavior.

## Constants (per STYLE_GUIDE §8)

In `src/sportstradamus/helpers/archive.py`:

```python
# Hours before commence_time treated as "the books' line" during training.
# Aligned with the typical Vegas closing-window inflection (~8h pre-game).
TRAINING_LOOKBACK_HOURS: int = 8

# Sharp books that anchor the movement-direction diagnostic in CLV.
SHARP_BOOKS: tuple[str, ...] = ("pinnacle", "circa", "bookmaker")
```

## Backwards Compatibility — Three Guarantees

1. **Existing `archive.duckdb`** — ALTER + UPDATE in place. Old rows backfill to `game_date` midnight; no data lost.
2. **Existing callers** — every `get_ev/get_line/get_team_market/get_moneyline/get_total` call site works unchanged. New `at` kwarg is optional and defaults to "latest", which matches today's behavior.
3. **Single-point legacy entries** — any `at=ts` query against a row with one observation at midnight returns that observation (it's always ≤ target). No NaN regressions in training or prediction.

## Verification

1. **Unit tests** — `tests/test_archive_history.py`: round-trip writes with explicit `observed_at`; point-in-time reads at exact / before-first / after-last timestamps; history range scans; legacy single-point fallback (insert one row, query with `at=`, expect that row); `get_movement` against a synthetic 5-observation series.
2. **Integration smoke** — extend `tests/integration/test_end_to_end.py` to run `confer` twice with a sleep between, then assert `archive.get_line_history(...)` returns ≥2 rows with monotonically increasing `observed_at`.
3. **Manual end-to-end**:
   - `poetry run python -m sportstradamus.scripts.add_observed_at_to_archive`
   - `poetry run confer && sleep 300 && poetry run confer --close-lines`
   - In `duckdb archive/archive.duckdb`: `SELECT entity, book, observed_at, ev FROM odds WHERE league='WNBA' AND market='PTS' ORDER BY entity, observed_at LIMIT 20;` → expect ≥2 rows per (entity, book) with distinct timestamps.
   - `poetry run meditate --league WNBA` → confirm `training_report.txt` `mean_line` matches a SQL query against the 8 h-prior archive snapshot for the same date set.
   - `poetry run reflect` → verify CLV summary shape unchanged and the new `frac_lines_moved_toward_model` column appears.
4. **Quality gates** (per CLAUDE.md hard rule, both must pass before commit):
   - `poetry run ruff check src/sportstradamus/`
   - `poetry run pytest tests/golden/`

## Out of Scope

- Storing raw `over_odds`/`under_odds` (would let us re-derive EV with new `cv`/`dist`/`gate` later — useful but bigger change; track as follow-up).
- Per-book reliability calibration based on movement direction (would feed `book_weights.json` from movement data — requires a season of history first).
- Postgres/TimescaleDB migration (port the schema later if scale demands).
- Compaction / retention job (revisit at 5 GB).
