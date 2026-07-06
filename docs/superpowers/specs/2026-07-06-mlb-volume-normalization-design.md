# MLB volume normalization — design spec

**Status:** approved (design); pending implementation plan
**Scope:** `stats/mlb.py`, `stats/base.py` (read only), `data/config/stat_meta.json`,
`training/markets.py`, plus golden unit tests. No training, no `shipped:` flip, no
distribution-family change.

## Problem

Every other league runs a team-volume sanity check that redistributes a game's
opportunity budget across the players actually in the lineup: NBA and NHL through
`Stats.scale_team_volume_to_budget`, NFL through `_rescale_team_volume`. The intent
(owner's words) is to "lower overconfident volume estimations or raise volume
projections in case some key players that usually have a large volume share are
missing." MLB calls none of them — its `get_volume_stats` just loads a
`plateAppearances` model pickle and joins raw per-player projections, with no
team-level normalization. This spec closes that gap in the way baseball's game
structure actually demands, which turns out to be *simpler* than the other leagues,
not a copy of them.

## Key insight: the batting order is the normalization

In basketball and hockey the volume driver is a continuous, coach-discretion minutes
budget, so a missing starter frees minutes that must be redistributed to whoever else
is on the floor — hence `scale_team_volume_to_budget`. Baseball is different. Plate
appearances are governed by a fixed nine-slot batting order that turns over in strict
sequence. The leadoff slot gets ~4.5 PA, the nine-hole ~3.4 PA, and **those counts
belong to the slot, not the player**. If a star sits, his replacement bats in the same
slot and inherits the same PA count; no PAs are "freed" for teammates.

So MLB does **not** need `scale_team_volume_to_budget`. The slot curve *is* the
per-player normalization. The only team-level scalar left is how much the whole lineup
turns over on a given day — a good offense reaching base more pushes a bit of extra PA
to the bottom of the order. That is a single bounded team multiplier
(`offense_adjustment`), not a per-player variance-weighted redistribution.

This also means the hitter PA projection is **structural** — a lookup table times a
scalar, no model fit — which has a useful side effect: it needs no training, so hitter
volume features populate the training matrices immediately (the `--matrix-only`
bootstrap gap that blocks a *trained* projector does not apply here).

## Two tracks

`get_volume_stats(offers, date, pitcher)` splits into two independent tracks by the
`pitcher` flag that already drives it.

### Track 1 — pitchers: model, no normalization

Unchanged in spirit from today. `pitcher=True` loads the `pitches thrown` volume
model via the shared `load_volume_model_params` head and joins its per-player
distribution params. **No team normalization** — the owner's decision, because books
don't price bullpen arms so there is no team pitch budget to redistribute against, and
the starter's own workload model already carries the signal. `pitches thrown` stays in
`self.volume_stats` and stays a trained cell. This mirrors the NHL goalie track
(`shotsAgainst`), which also skips `scale_team_volume_to_budget`.

### Track 2 — hitters: structural PA projector

`pitcher=False` builds a structural `plateAppearances` projection with no model:

```
proj plateAppearances mean = SLOT_PA[home|away][slot] × offense_adjustment
proj plateAppearances std  = SLOT_STD[slot]
```

`plateAppearances` is **retired as a trained cell** entirely (owner: "pull plate
appearances as a modeled cell entirely; we don't need it") — removed from
`ALL_MARKETS["MLB"]` and its `stat_meta.json` entry deleted. It survives only as this
structural projector feeding `proj plateAppearances mean/std` into every hitter cell's
`playerProfile`, exactly as a trained volume model would have.

The `std` is left unadjusted by offense — it is the within-slot game-to-game PA spread
(extra innings, blowouts, early removal), which is genuinely unpredictable and does not
scale with lineup quality.

#### Slot resolution — three paths

The batting slot for each hitter is resolved in priority order:

1. **Order known** (lineup posted): use the actual slot → `SLOT_PA[h|a][slot]`.
2. **Order unknown** (lineup not yet announced when bets post — common): use the
   player's own historical slot distribution,
   `mean = Σ_s P(player bats slot s) × SLOT_PA[h|a][s]`, with `P(bats s)` from the
   player's game log. `std` widens to reflect the extra slot uncertainty (root-sum the
   slot-mix variance into `SLOT_STD`).
3. **No history** (rookie / first start, no slot prior): fall back to the league-average
   slot PA with a deliberately wide `std`.

Fewer than nine hitters in the offer set is handled by construction — project whoever is
present by their own slot; unmodeled slots simply get no projection.

## The offense adjustment (bounded, deliberately light)

`offense_adjustment` is a single team-level scalar, nominal `1.0`, blending two
estimates of "how much offense today" with the mechanistic OBP term leading and the
market as a sanity anchor (owner: option 3, blend). It is bounded so it stays a nudge —
the predictable offense signal is only ~±1–2 PA on a ~36 PA base; the rest of the
team-PA spread is unpredictable and correctly lives in `std`, not the mean.

```
OBP_exp  = team_recent_OBP × (opp_starter_OBP_allowed / lg_avg_OBP) × park_OBP_factor
OBP_factor    = (1 − lg_avg_OBP) / (1 − OBP_exp)          # PAs ∝ 1/(1−OBP); =1 at league avg
market_factor = implied_team_runs / lg_avg_team_runs       # =1 at league avg
offense_adjustment = clip(0.70 · OBP_factor + 0.30 · market_factor, 0.92, 1.08)
```

- **`team_recent_OBP`** — the team's rolling OBP. **`opp_starter_OBP_allowed`** — the
  opposing starter's rolling OBP-allowed. Both come from the `OBP` gamelog column, which
  is populated on both batter and pitcher rows (verified: 100% non-null on starter-pitcher
  rows, so a pitcher's `OBP` is his OBP-allowed).
- **`park_OBP_factor`** — the per-team `OBP` multiplier already in
  `data/config/park_factor.json`.
- **`implied_team_runs`** — the Vegas game total split by the moneyline, from the
  archive (`get_total` + `get_moneyline`); the same odds already consulted elsewhere.
- **`lg_avg_OBP`**, **`lg_avg_team_runs`** — league denominators, module constants
  computed by the committed measurement script as **pooled ratios**
  (`Σ times-on-base / Σ PA`, `Σ runs / team-games`), not means of per-game ratios.
  `lg_avg_team_runs ≈ 4.44` (measured); `lg_avg_OBP ≈ 0.318` (pooled).

The `0.70/0.30` weights and the `±8%` clip are measured-tunable constants, not magic.
The adjustment is applied as a flat multiplier across all nine slots (a better offense
technically tail-loads the extra PA toward the bottom of the order; that second-order
refinement is deliberately out of scope for v1).

### Graceful degradation

Each input can be missing when bets post; the adjustment degrades term by term rather
than failing:

- No archive total/moneyline → drop the market term, use `OBP_factor` alone.
- Missing OBP inputs (early season, thin history) → `offense_adjustment = 1.0` (bare
  home/away slot curve).

## Integration

- **`get_volume_stats`** splits along the existing `pitcher` flag: `True` → unchanged
  model path; `False` → structural projector. Two small private helpers keep it
  readable: one for slot resolution (the three paths), one for the offense adjustment.
- **`self.volume_stats = ["pitches thrown"]`** — `plateAppearances` drops out. Hitter
  markets are not in the list, so `_dispatch_volume_stats` routes them to
  `get_volume_stats(pitcher=False)`; `pitches thrown` stays in the list so it trains as
  a volume target; other pitcher markets route to `get_volume_stats(pitcher=True)`.
- **Remove `plateAppearances`** from `ALL_MARKETS["MLB"]` and delete its
  `stat_meta.json` cell.
- **Constants** — `SLOT_PA_HOME`, `SLOT_PA_AWAY`, `SLOT_STD`, `LG_AVG_OBP`,
  `LG_AVG_TEAM_RUNS`, the blend weights, and the clip band are module-level named
  constants in `mlb.py` with a one-line comment citing the gamelog measurement
  (STYLE_GUIDE §9). MLB does **not** call `scale_team_volume_to_budget`; the divergence
  from NBA/NHL is intentional and documented at the call site.

## Testing

The structural projector is fully unit-testable **without any training**, which is the
main testing win:

- Golden unit tests for each slot-resolution path (order known, order unknown via
  historical mix, no-history league fallback), home vs away, and the `std` widening
  when the order is unknown.
- Golden unit tests for the offense adjustment: OBP-only, OBP+market blend, both
  degradation fallbacks, and the clip band edges.
- An assembly check: a hitter-market matrix shows `proj plateAppearances mean/std`
  populated with **no `.mdl` present**.
- A committed measurement script (evolved from the scratchpad probe) reproduces the
  slot curve, `std` table, and league denominators from the gamelog so the constants
  are auditable, not hand-waved.

## Measured constants (backfilled ~2-season gamelog)

```
SLOT_PA_HOME = [4.404, 4.291, 4.208, 4.110, 3.971, 3.831, 3.688, 3.533, 3.358]  # Σ 35.39
SLOT_PA_AWAY = [4.584, 4.488, 4.377, 4.284, 4.149, 4.008, 3.862, 3.705, 3.543]  # Σ 37.00
SLOT_STD     = [0.712, 0.698, 0.685, 0.662, 0.693, 0.731, 0.764, 0.784, 0.799]
LG_AVG_TEAM_RUNS = 4.44          # pooled team runs / team-game
LG_AVG_OBP       ≈ 0.318         # pooled Σ on-base / Σ PA (measurement script finalizes)
```

Full-team PA budget is 36.84 (home) / 38.52 (away); the nine modeled slots sum to
35.39 / 37.00. The ~1.5 PA difference is bench / pinch-hit reserve, intentionally
unmodeled (books don't price it, and the slot means already bake in average removal).
The projector anchors on the slot sums, not the full-team budget.

## Non-goals

- No pitcher normalization (owner decision).
- No `scale_team_volume_to_budget` for MLB (the slot structure replaces it).
- No slot-dependent (tail-loaded) offense adjustment in v1.
- No training, no `shipped:` flip, no distribution-family change.

## Done criteria

1. `ruff` / `pytest tests/golden/` / `pytest -m integration -n0` all clean.
2. A hitter-market MLB matrix has populated `proj plateAppearances mean/std` columns
   with no model pickle.
3. `plateAppearances` absent from `ALL_MARKETS["MLB"]` and `stat_meta.json`;
   `pitches thrown` still trains as the pitcher volume cell.
4. Golden unit tests cover all three slot paths + the adjustment blend/degradation/clip.
