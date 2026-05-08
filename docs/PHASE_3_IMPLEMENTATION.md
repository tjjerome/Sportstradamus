# Phase 3 Implementation Plan — Underdog-Specific Decision Engine

**Status target:** turn the existing parlay candidate stream into ranked,
Kelly-sized, contest-variant-aware Underdog recommendations.

**Source documents:**
- `docs/sportstradamus_roadmap_v2.md` §Phase 3 (3.1–3.5) and §Updated Critical
  Path (post-2026-05-08).
- `docs/underdog_edge_suite.md` §4 (Decision Engines) and §5 (Execution & Risk).
- Audit notes in §1.3 (closing-line freeze — dropped, workaround sufficient)
  and §2.4 (CLV reporting — monitor the implicit-close assumption).

**Estimated time:** 4–5 weekends, sequenced so each step ships independently.

---

## 0. Scope and Non-Goals

**In scope (this plan delivers):**

1. CLV-side bring-up needed before Kelly can lean on `model_calib` numbers
   (Step 1).
2. Kelly sizing module (§3.1).
3. Underdog-native `pickem-build` orchestrator (§3.3).
4. Pick'em Champions strategy (§3.4).
5. Streaks recommender + smaller Ladders / Rivals follow-ups (§3.5).

**Explicitly out of scope:**

- Placed-bet logger (`tracking/`) and per-bet CLV. Roadmap §2.2/2.3 dropped.
- Phase 4.1 alerts (deferred).
- Best Ball / Battle Royale (Phase 5).
- Bayesian / conformal modeling (Phase 6).

**Roadmap §3.2 (contest payouts) is already DONE.** This plan consumes
`data/underdog_payouts.json` and the `contest_variant` arg on
`beam_search_parlays`; it does not redo that work.

---

## 1. CLV Bring-Up Required Before Kelly (≈½ weekend)

Roadmap §1.3 dropped the explicit `freeze_close` pipeline because the only
consumer was the placed-bet logger. Phase 3 has two new consumers:

- Kelly's `model_calibration` shrink factor wants per-(league, market) CLV
  segment quality, not just the global mean.
- The recommendation YAML wants a `closing_devig_prob` field so that, if a
  user later imports the YAML into a personal tracking sheet, CLV can be
  back-filled per-leg without rerunning prediction.

We do **not** rebuild the dropped `freeze_close` system. Instead:

### 1.a Close-timestamp sanity check in `clv.fill_from_archive`

Per §2.4 monitoring note: warn when the archive snapshot used as "close"
is more than `N` minutes from `commence_time`. Cheap insurance against a
silent Odds API behavior change.

**Touchpoints:** `src/sportstradamus/clv.py:fill_from_archive`,
`src/sportstradamus/clv.py:_safe_get_ev`.

**Spec:**

- Add a constant `CLOSE_LOOKBACK_WARN_MINUTES = 90` at module scope with a
  one-line reason comment per STYLE_GUIDE.md §8.
- Have `_safe_get_ev` (or its caller) record the timestamp of the archive
  row it pulled. Compare to `commence_time` if available on the row.
- Log a single WARNING per (league, market, date) segment via
  `helpers.logging.get_logger`. Do not warn per leg — that floods.
- Test: synthetic archive where the latest pre-kickoff sample is 4h old
  triggers the warning; one within 30m does not.

### 1.b Per-segment CLV getter for Kelly

Add `clv.get_segment_calibration(league: str, market: str) -> float`. Reads
the most recent `reflect`-produced CLV summary (already logged; persist
the segment table to `data/clv_segments.parquet` on each `reflect` run as
a side effect). Returns `frac_beat_close` clamped to `[0.45, 0.65]` and
remapped to `[0.0, 1.0]` calibration weight, with a documented fallback
of `1.0` when the segment has fewer than `CLV_SEGMENT_MIN_N=20` legs.

This is the **only** CLV-side change that is genuinely required for
Phase 3. Step 2 (Kelly) consumes it.

**Touchpoints:** `src/sportstradamus/clv.py`, `src/sportstradamus/nightly.py`
(persist segments parquet at end of `reflect`).

**Tests:** `tests/golden/test_clv_segment_calibration.py` — small, large,
and missing-segment cases.

### 1.c Lightweight `Archive.get_closing_line` shim

Not the freeze pipeline. Just a read-only helper that wraps
`Archive.get_line(...)` + `Archive.get_ev(...)` for the latest pre-kickoff
sample, returning a small dataclass `(line, devig_over, sample_ts, book_set)`.
Useful inside the strategy module (Step 3) when writing recommendation
YAML, and inside Step 1.a's sanity check.

**Touchpoints:** `src/sportstradamus/helpers/archive.py`.

---

## 2. Kelly Sizing Module — §3.1 (1 weekend)

Revive `src/deprecated/opt_kelley_bet.py` into
`src/sportstradamus/strategies/kelly.py`.

### 2.a Module layout

```
src/sportstradamus/strategies/
├── __init__.py        # re-exports kelly + future pickem
├── kelly.py
└── README.md          # one-paragraph orientation
```

### 2.b Public API (matches roadmap §3.1 prompt verbatim)

```python
def fractional_kelly_stake(
    bankroll: Decimal,
    win_prob: float,
    payout_multiplier: Decimal,
    fraction: float = 0.25,
    model_calibration: float = 1.0,
    max_fraction_of_bankroll: float = 0.005,
) -> Decimal: ...

def joint_kelly_portfolio(
    bankroll: Decimal,
    candidates: list[KellyCandidate],
    fraction: float = 0.25,
) -> dict[str, Decimal]: ...
```

### 2.c Constants (no magic numbers — STYLE_GUIDE §8)

| Name | Value | Reason |
|---|---|---|
| `DEFAULT_KELLY_FRACTION` | `0.25` | Quarter-Kelly; standard recreational hedge against estimate variance. |
| `MAX_FRACTION_OF_BANKROLL` | `0.005` | Hard 0.5%-per-entry cap from edge-suite §5.1. |
| `CALIBRATION_FLOOR` | `0.5` | Below this we treat the model as uncalibrated and degenerate to `effective_p = 0.5`. |

### 2.d Effective probability shrinkage

```
effective_p = 0.5 + (win_prob - 0.5) * model_calibration
```

`model_calibration` resolved in this order:

1. Explicit kwarg.
2. Per-(league, market) value from
   `clv.get_segment_calibration` (Step 1.b).
3. Per-market `model_calib` from `training/report.py` via the new thin
   getter `training.report.get_market_calibration(league, market)`.
4. Fallback `1.0` (no shrinkage) if all of the above return None.

Resolution is intentional: CLV-derived calibration is more recent than
training-time calibration and should win when both exist.

### 2.e Portfolio variant

Convex problem solved with cvxpy SCS. Add cvxpy under
`[tool.poetry.group.strategy]`, keep core deps untouched. Module imports
cvxpy lazily so that `kelly.fractional_kelly_stake` works without it.

### 2.f CLI

`poetry run kelly --bankroll 500 --from data/recommendations/{date}.yaml`

Reads the YAML produced in Step 3, prints a Rich/tabulate table of
`(candidate_id, EV, win_prob, eff_p, recommended_stake)`. Register in
`pyproject.toml`.

### 2.g Tests — `tests/golden/test_kelly.py`

- −EV bet returns `Decimal("0")`.
- +EV closed-form expected stake matches analytical fractional Kelly.
- Hard cap kicks in when uncapped Kelly exceeds 0.5%.
- Two-bet portfolio sums to ≤ `fraction × bankroll`.
- Calibration shrinkage: `(win_prob=0.6, calib=0.5)` ≡ `(win_prob=0.55, calib=1.0)`.

### 2.h Deprecated cleanup

After tests green, move `src/deprecated/opt_kelley_bet.py` to
`src/deprecated/.archived/` per the deprecated README protocol. Update
the README's TODO list to remove the entry.

---

## 3. Underdog-Native Strategy — §3.3 (1 weekend)

`src/sportstradamus/strategies/underdog_pickem.py` is the orchestrator. No
math lives here — it composes `prediction/correlation.beam_search_parlays`,
`strategies/kelly`, and `clv` helpers.

### 3.a Public API

```python
@dataclass
class PickemConfig:
    min_model_edge: float = 0.020
    min_sharp_edge: float = 0.015
    disagreement_threshold: float = 0.04
    min_correlation: float = 0.10
    min_ev: float = 0.05
    entry_sizes: tuple[int, ...] = (3, 5)
    contest_variants: tuple[str, ...] = ('power', 'flex')
    top_k: int = 20
    max_overlap: int = 2
    kelly_fraction: float = 0.25
    max_stake_pct_bankroll: float = 0.005

def construct_entries(
    date: datetime.date,
    bankroll: Decimal,
    config: PickemConfig | None = None,
) -> list[RecommendedEntry]: ...
```

### 3.b Pipeline

1. **Load offers** — reuse the loader `prophecize` already calls; do not
   duplicate.
2. **Filter legs** — Underdog markets only, model coverage present, sharp
   devig present, |model − sharp| ≤ `disagreement_threshold`, both edges
   above their thresholds.
3. **Search** — call `beam_search_parlays(..., contest_variant=v)` once
   per `(entry_size, variant)` cross. Power and Flex by default.
4. **Size** — call `fractional_kelly_stake` per candidate. Resolve
   `model_calibration` via Step 1.b.
5. **Rank** — sort by EV-per-dollar, dedupe by leg overlap (≤
   `max_overlap` shared legs across recommendations), cut to `top_k`.
6. **Emit two sinks:**
   - Append "Pickem Recommendations" tab to the existing Sheets export
     (`prediction/sheets.py`). Add a `Variant` column.
   - Write `data/recommendations/{date}.yaml`. Schema below.

### 3.c YAML schema

Stable enough for downstream consumers (CLV back-fill, future tracking).

```yaml
generated_at: 2026-09-12T15:30:00-04:00
date: 2026-09-12
bankroll: 500
config: {model_calibration_source: clv_segment, ...}
entries:
  - id: <hash of legs>
    contest_variant: power
    entry_size: 3
    legs:
      - sport: NFL
        player_id: "00-0036442"
        player_name: "Josh Allen"
        market: player_pass_yds
        line: 248.5
        side: over
        model_prob: 0.582
        sharp_devig: 0.561
        closing_devig: null    # populated post-lock by reflect
    joint_prob: 0.276
    payout_multiplier: 6.0
    ev: 0.656
    recommended_stake: 1.25
    notes: []
```

### 3.d CLI

`poetry run pickem-build --date today --bankroll 500`. Defaults pulled
from `data/pickem_config.json`. **Bankroll is a CLI flag only** — no
`data/bankroll.json` per roadmap §Operational note "one less piece of
state to drift".

### 3.e Tests — `tests/golden/test_underdog_pickem.py`

- Filtering excludes legs failing each threshold.
- Disagreement threshold catches a divergence and skips the leg.
- Both Power and Flex appear when both enabled.
- End-to-end on the existing WNBA fixture
  (`tests/integration/fixtures/`) produces a non-empty YAML.
- YAML round-trips via `yaml.safe_load`.

### 3.f CLI snapshot

Regenerate `tests/golden/test_cli_help.py` snapshots after pin.

---

## 4. Pick'em Champions — §3.4 (1 weekend)

`src/sportstradamus/strategies/underdog_champions.py`. Different
optimization target than `underdog_pickem.py` — pari-mutuel against the
field, not arb against a static line.

### 4.a Field-pick-rate input

Two-tier sourcing:

1. **Preferred:** extend `books.get_ud()` to capture the per-leg
   popularity indicator the Champions board exposes. Stored on the offer
   dict as `field_pick_rate_higher ∈ [0, 1]`.
2. **Fallback (when missing):** ADP-style heuristic — bias toward
   Higher when line < season average, Lower when line > season average,
   with a documented base rate of 0.58 for Higher. Comment the fallback
   with a TODO referencing 4.a.1.

### 4.b Optimization

For each candidate entry:

1. Sample 100 000 opponent rosters by drawing each leg according to
   `field_pick_rate`. Reuse the existing sim infra; no new RNG.
2. Score every roster (yours + opponents') via the model's joint sample
   counts of legs hit.
3. Compute your candidate's percentile rank distribution.
4. Map percentile to expected prize equity using `champions` payout
   curve in `data/underdog_payouts.json` (extend the JSON if missing —
   keep §3.2 schema).

EV is `prize_equity - entry_cost`.

### 4.c CLI

`poetry run champions-build --date today --bankroll 500`.

### 4.d Tests

- Contrarian entries score higher than chalky entries when both have
  equal model EV.
- Fallback `field_pick_rate` produces non-empty output.
- Fixture round-trip → YAML, schema matches `pickem-build`'s with an
  extra `prize_equity` column.

Constraint: do **not** merge with `underdog_pickem.py`.

---

## 5. Streaks, Ladders, Rivals — §3.5 (½–1 weekend)

### 5.a Streaks (`underdog_streaks.py`)

Sequential decision: at streak length `n`, continue or cash out.
Threshold derived from the geometric payout structure.

```python
def recommend_streak_action(
    current_streak: int,
    used_team_ids: list[str],
    available_offers: list[Offer],
    config: StreaksConfig,
) -> StreakRecommendation: ...
```

CLI: `poetry run streaks-recommend --streak 4 --used-teams KC,SF`.

Tests:
- Streak 1, 60% available pick → `continue`.
- Streak 8, 52% best available → `cash_out`.
- Two-different-teams rule enforced for picks 1 and 2.

### 5.b Ladders (`underdog_ladders.py`)

Lowest-shared-rung expected-payout calc. Smaller; one weekend day. Reuse
joint sampling from `prediction/correlation.py` — no new copula code.

### 5.c Rivals (`underdog_rivals.py`)

Essentially 2-pick / 3-pick Power on player-vs-player matchups.
Critical-path check: model must cover **both** players for the same
market; if not, skip the matchup with a logged WARNING.

These three are independent — each ships its own PR.

---

## 6. Sequencing and PR Plan

| PR | Scope | Branch off |
|---|---|---|
| 3-pre | Step 1 (CLV bring-up: sanity warn, segment getter, archive shim) | `devel` |
| 3-1 | Step 2 (Kelly module + CLI + cvxpy group) | merges after 3-pre |
| 3-3 | Step 3 (`pickem-build`) | merges after 3-1 |
| 3-4 | Step 4 (`champions-build`) | merges after 3-3 |
| 3-5a | Step 5.a (Streaks) | parallel with 3-5b/c after 3-3 |
| 3-5b | Step 5.b (Ladders) | parallel |
| 3-5c | Step 5.c (Rivals) | parallel |

Each PR ships with: ruff clean, golden tests pass, regenerated CLI
snapshot if the CLI changed.

## 7. Quality Gates

Per PR, in order:

1. `poetry run ruff check src/sportstradamus/`
2. `poetry run pytest tests/golden/`
3. `poetry run pytest tests/integration -m integration` if the change
   crosses module boundaries (Steps 3, 4, 5).
4. `REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py`
   on intentional CLI flag changes only.

## 8. Risks and Open Questions

- **Calibration source priority (Step 2.d).** If CLV-segment counts are
  thin early in a season, the shrink will fall through to training-time
  `model_calib`. Confirm this is desired before implementation; an
  alternative is to weight-average the two sources by sample size.
- **Champions popularity scrape (Step 4.a).** Depends on `books.get_ud()`
  surface; if the field isn't reliably exposed, Step 4 ships with the
  fallback only and 4.a.1 becomes a follow-up.
- **Magic-number purge in `prediction/correlation.py`** (roadmap §1.2
  follow-up) is not formally part of Phase 3 but blocks confidence in
  the Step 3 EV ranking. Recommend bundling that purge into PR 3-pre or
  a sibling PR before PR 3-3 lands.
- **`prediction/parlay.py` split** is advertised in `CONTRIBUTING.md` but
  the live functions live in `prediction/correlation.py`. Decide before
  PR 3-3: split the module, or update `CONTRIBUTING.md`. Don't carry the
  inconsistency into Phase 3 imports.

## 9. Done Criteria

Phase 3 is complete when:

1. `poetry run pickem-build --date today --bankroll 500` produces a
   ranked, sized YAML and updates the Sheets tab.
2. `poetry run kelly --from <yaml>` re-sizes from the same YAML offline.
3. `champions-build`, `streaks-recommend`, plus Ladders and Rivals CLIs
   all run end-to-end on fixtures.
4. `clv.summarize` continues to surface segments with no regressions,
   and the new close-timestamp sanity warning fires in tests.
5. `src/deprecated/opt_kelley_bet.py` has moved to `.archived/` and the
   deprecated README's TODO list is shorter by one entry.
