# Expanding the NFL feature set on the new Fantasy Points API

The rebuilt Fantasy Points product exposes far more than the collector currently
takes. This page is the survey: what the new API offers beyond parity, which of
it maps to markets we train, and in what order it is worth building.

The port itself — the transport swap, the column translation back to the legacy
vocabulary — is in [fantasypoints.md](fantasypoints.md). This page assumes it has
landed and asks what to do next.

## Why the new API is a different kind of source

The old catalog was 44 narrow tool endpoints, each returning one fixed view. The
new one is ~25 wide endpoints with two capabilities the old one lacked, and they
compound:

* **A filter grid.** Any endpoint accepts a situational filter — down, distance,
  field position, score differential, personnel, pressure, motion. A request is
  a slice, not a table.
* **Denominators.** Every row carries a `__raw` block holding the counts behind
  the displayed rates. Slices are therefore poolable into correct season-to-date
  aggregates instead of averages-of-averages.

Together they mean a feature no longer has to exist as a column upstream. If the
concept can be expressed as a filter, it can be collected.

**History is available.** `seasons=2022` returns full rows on the new endpoints,
so anything added here has training signal, not just serving signal. This was the
open question that gated the whole expansion; it is settled.

## The filter grid

Confirmed binding on `playcaller-stats`, and the `split_*` columns on `passing` /
`rushing` / `receiving` indicate the same grid applies there:

| Axis | Parameter | Values |
|---|---|---|
| Down | `down` | `1`, `3`, `3,4`, `1,2,3,4` |
| Distance | `ydsToGoMin` (and `…Max`) | integer yards |
| Field position | `ydsToScoreMax` | `20`, `10`, `5` |
| Score state | `scoreDiffMin`, `scoreDiffMax` | signed points |
| Drive number | `drive` | `1`…`19` |
| Half | `half` | `first`, `second` |
| Motion | `motion`, `motionType` | `1`; `DS`, `PS`, `SH`, `B` |
| Personnel | `personnel` | `00`…`41` (18 groupings) |
| Dropback type | `dropbackType` | `SD`, `RL`, `RR`, `SL`, `SR` |
| Play action | `playAction` | `1` |
| Screen | `screenPass` | `1` |
| Pressure | `qbPressured` | `1` |
| Scramble | `scramble` | `0`, `1` |
| Rhythm | `inRhythm` | `1` |
| Position | `positions` | `QB,RB,FB,WR,TE` |
| Grain / side | `mode` | `offense`, `defense` |
| Period | `regWeeks`, `postWeeks`, `seasons` | comma lists |

`mode` deserves emphasis: it switches **grain**, not merely side.
`rushing?positions=RB` returns players; `rushing?mode=defense` returns the 32
team rows. Every player endpoint therefore has a team-grain twin for free.

Further axes are implied by the `split_*` column vocabulary but not yet
exercised: `split_opp_head_coach`, `split_opp_defensive_coordinator`,
`split_team_playcaller`, `split_rush_concept`, `split_injury_designation`,
`split_practice_participation`, `split_indoor_outdoor`, `split_location`,
`split_conference`, `split_division`, `split_quarter`. Each is one probe.

**Budget the fan-out explicitly.** Every filtered slice is one more request per
week per endpoint, against a single session cookie, and the weekly run is already
44 calls. Expansion means a fixed, named list of catalog entries — never a
cross-product.

## Tier 1 — new endpoints

Whole endpoints the old catalog never had. Highest value per unit of work: each
needs a catalog entry and a recipe, no filter design.

### `passing-situation` — 311 columns

The largest single addition. The QB slate pre-split into 19 defensive contexts —
`pressured`, `clean`, `blitz`, `no_blitz`, `pa`, `no_pa`, `man`, `zone`,
`single_high`, `two_high`, `press`, `no_press`, `motion`, `no_motion`,
`disguised`, `static_look`, `base_d`, `sub_d`, `overall` — each with `dropbacks`,
`attempts`, `completions`, `yards`, `touchdowns`, `interceptions`, `ypa`,
`cmp_pct`, `epa_per_db`, `fp_per_db`, `passer_rating`. Counts as well as rates, so
recipes pool correctly.

Consumers: `passing yards`, `passing tds`, `completions`, `attempts`,
`interceptions`, `sacks taken`, `qb yards`, `qb tds` — 8 of the 20 NFL cells.

Collected as `player_passing_situation`; 17 `ps_*` recipes (context shares,
per-context YPA / EPA / INT rates, scramble yards per dropback) pool the
`__raw` counts.

The mechanism is straightforward. A quarterback's yardage distribution against a
heavy-blitz defense is a different distribution, and the model currently sees only
the marginal. Pairing the player-grain split with the opposing defense's
`coverage-matrix` and `team-stats` blitz/pressure rates — both already collected —
produces a real matchup interaction rather than two independent main effects.

### `pace` — 21 columns

`pace_neutral`, `pace_leading`, `pace_trailing`, `pace_h1`, `pace_h2`,
`pace_differential`, `pace_forced`, `plays_per_drive`, `sec_per_drive`, `drives`,
plus `__raw` counts.

Team-grain and small, but it supplies the volume-driver term the counting markets
lack. Pace times pass rate is play volume, and play volume is most of the variance
in `attempts`, `carries` and `targets`.

Collected as `team_pace` / `opponent_pace`; 8 `pace_*` team recipes and 4
`def_pace_*` / `def_plays_*` defense recipes, all pooled from the `__raw` counts.

### `lineup-combos/ol` — 27 columns

Offensive-line unit performance keyed by the actual five-man combination:
`unit_player_ids`, `snaps`, `pressure_rate`, `sack_rate`, `ybc_a`, `ypc`,
`epa_per_rush`, `epa_per_db`, `expl_rate`.

`sacks taken` and `rushing yards` are the obvious consumers, and this is the rare
feature with a clean causal story: an injury changes the unit, and the unit's
historical pressure rate is forward-looking information the market prices slowly.

Collected as `team_ol_combos`; the endpoint publishes rates only (no `__raw`), so
the 7 `ol_*` recipes are snap-weighted means across the units a team fielded, and
`ol_top_unit_share` (snaps of the most-used five ÷ all snaps) is the continuity
term.

### `cornerback-stats` + `rec-points-allowed`

`cornerback-stats` (41 columns) gives per-defender coverage production — `tgts`,
`recs`, `cmp_pct`, `ypt`, `epa_per_tgt`, `fp_per_tgt`, `blanketed_pct`,
`step_plus_pct`, `finc_pct` — plus alignment shares `lcb_pct` / `rcb_pct` /
`scb_pct`. `rec-points-allowed` (53 columns) gives fantasy points allowed split by
receiver alignment (`outer_*`, `slot_*`, `lwr_*`, `rwr_*`).

Together they enable genuine WR-vs-CB matchup features for `receiving yards`,
`receptions`, `targets` and `receiving tds`. The existing `wr_coverage_matchup` /
`qb_coverage_matchup` kinds are a thin version of the same idea.

### `recsep-roles` — 60 columns

Separation and route production split by alignment role — X, Z, F, condensed —
each with `routes`, `rte_pct`, `tprr`, `yprr`, `win_rate`, `sep_score`, plus
`sep_market_share` and `target_market_share`.

Role is a usage-stability signal: an X receiver's target share behaves differently
from a slot's under the same team pass volume.

### `injury-reports` — 36 columns

`status`, `practice`, `injury`, `body_group` joined to that player's `snaps`,
`snap_pct`, `routes`, `route_pct`, `carry_pct`, `tgt_pct` — availability and usage
in one row, keyed by `gsis_id` and `week`.

**This is a leak hazard and must be treated like `line_matchups` was.** The week-N
row reflects week-N practice reports, which are published before the game, so the
data is legitimately available at prediction time — but only if the collector
snapshots it *before* kickoff and the training matrix joins the pre-game version.
A snapshot taken afterwards is week-N results in week-N features, which is exactly
why `line_matchups` was retired. Design the join before writing the fetch.

### `coach-records` — 24 columns

Team results attributed to the coach, alongside `playcaller-stats`' coach grain.
`split_team_playcaller` and `split_opp_head_coach` exist as split axes, so
tendencies can be attributed to the playcaller rather than the franchise. That is
the right unit — a new coordinator moves pass rate more than a roster change does,
and franchise-level history mis-attributes it.

## Tier 2 — filter fan-out on endpoints already collected

Cheap in code, moderate in request budget. Ranked:

1. **`ydsToScoreMax=10` and `=5`.** Red-zone and goal-line usage. `rushing tds`,
   `receiving tds`, `tds` and `passing tds` are ZINB/NegBin cells whose
   zero-inflation is driven almost entirely by whether the player gets goal-line
   work. The current features carry `rush_basic_INSIDE5_ATT` and
   `rec_basic_INSIDE10_TGT` as bare counts; the filtered endpoints give the whole
   rate and efficiency panel inside those zones.
2. **`down=3`.** Third-down usage separates the every-down back from the
   early-down back, which is most of the residual variance in `carries` and
   `targets`.
3. **`qbPressured=1`, `playAction=1`, `screenPass=1`.** Play-type mix is a
   coaching choice and therefore stable week to week.
4. **`personnel=11|12|13|21`.** Personnel usage exists today as team rates
   (`personnel_11_rate`) but not as per-player conditional production.

**Defense mirrors.** `defenseProfile` receives 12 FP columns (the 7 legacy ones,
the 4 `pace_opp` recipes and the revived `def_rush_ybc_allowed_per_att`) against
the offense side's 33. That asymmetry is not principled — it is an artifact of
which tools have been pulled with `mode=defense`, and any of the offense panels
can be mirrored the same way `pace` was.

## Tier 3 — structural

**Server-side season-to-date.** `regWeeks=1,2,3,…` returns the pooled aggregate
directly, where `_lookback_windows` currently pools week parquets client-side.
Fewer requests and one less class of blending bug, but it changes snapshot
semantics — a redesign, not a config change. Probably not worth it.

**History lives on the new API.** `seasons=` accepts prior years, so every
season from 2021 (postseason included) is pulled through the current API. That is
what makes Tier-1 endpoints usable for training rather than only serving, and it
closed the cross-era identity split the legacy pulls had opened — see
[One identity space](fantasypoints.md#one-identity-space).

## Sequencing and the retrain rule

Anything on this page changes `expected_columns`, so it requires a retrain.
Sequence the work as **one deliberate feature-set bump** with a full `meditate`
and a ship-gate comparison against the current board — not as incremental column
creep. The gate machinery compares cells against their own history, and a moving
feature set makes that comparison meaningless (see
[MODEL_LIFECYCLE.md](MODEL_LIFECYCLE.md)).

Order, with the state of each step:

1. History on the new API — 2021 pulled; 2022–2025 plus every postseason pull
   season by season under the account's daily request budget
   ([fantasypoints.md](fantasypoints.md) § Historical backfill), most recent season
   first. Until a season is complete it is new-era-or-missing on disk, never mixed.
2. Tier 1 in one batch — code landed: `passing-situation`, `pace`,
   `lineup-combos/ol` plus the raw-count free wins below, 47 new recipe outputs.
   The rebuild of every NFL matrix and the retrain follow the pull;
   `docs/handoffs/model_improvement_track.md` §10 carries the verdict.
3. Tier 2 filter fan-out, as a fixed slice list — open.
4. `injury-reports` last, once the pre-game snapshot discipline is designed — open.

Each later batch repeats step 2's shape: catalog entries, `FILE_KINDS`, recipes
reading the raw names, then the full-rebuild → `volume-v1` → retrain sequence in
[MODEL_LIFECYCLE.md](MODEL_LIFECYCLE.md); appending to the cached matrices leaves
every new column NaN on history and silently inert.

## Free wins that shipped with Tier 1

The `__raw` blocks of tools already collected carried features no legacy recipe
read: `rush_ybc_per_att` / `def_rush_ybc_allowed_per_att` (one `ybc_total` map
entry each), expected-TD rates for passing / rushing / receiving, RYOE per
attempt, red-zone carry shares (`inside_10_rushes / team_inside_10_active`,
inside-5 likewise), the endzone-attempt rate and expected receiving yards per
target. Anything else still unread in a `__raw` block is the same kind of win:
a recipe row and a retrain.
