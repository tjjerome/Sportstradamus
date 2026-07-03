# Thesis-engine v2 — design

**Date:** 2026-06-11
**Status:** Design, pending owner review of this spec.
**Author:** Claude (fable) — thesis-engine v2

## Problem

The P2 thesis generator (`src/sportstradamus/prediction/stories/`) sells every slip with the
same story: *a featured player does a thing in a game shape*. That was the right v1 — it beat
the render-time phrase bank it replaced — but it has four structural weaknesses and one
outright bug, all verified against the code this session:

1. **Single archetype.** `_family_thesis` (`thesis.py:167-187`) always elects a player
   (`player = max(legs.totals, key=...)`, line 175) and renders a player-template cell. There
   is no story for a slip whose evidence is the game script itself, a positional-unit
   mismatch, or a correlated same-game stack. Every one of the bank's 107 variants
   (`bank.py:113`, 44 cells — counted by AST this session) takes `{p}` as its subject.

2. **Arbitrary "standout" on no-standout slips.** `_stardom` (`thesis.py:154-164`) ranks by
   (market breadth, line-vs-market percentile, leg volume) and then **breaks ties
   alphabetically by player name** — the sort tuple ends in `player` (line 164). A slip of
   four near-equal legs still gets a forced star, picked by the alphabet. The owner's bar:
   such a slip should instead read "shootout game script lifts all player stats" or "NYK
   guards feast against a weak SAS perimeter."

3. **Family-coupled.** Theses are keyed one-per-`(League, Game, Family)` (`thesis.py:58`).
   `Family` is the output of `assign_parlay_families` (`parlay.py:243`) — RKHS-cosine
   hierarchical clustering the owner has explicitly demoted: *"Don't get too attached to the
   family clustering. That was an artifact of a less sophisticated system."* The thesis must
   be a function of the actual legs in a slip; family survives only as a Slips
   grouping/ordering device.

4. **Goes stale on edit.** The thesis is computed once at prophecize time (`cli.py:203`) and
   frozen into `current_parlays.parquet`. The redesign gives the user full agency to edit a
   slip in the rail (redesign spec §4); remove the named player and the frozen headline still
   names them. The engine must regenerate live on slip edit.

5. **The game-shape classifier is broken in production (found this session).**
   `_TOTAL_BANDS` (`thesis.py:25-31`) expects raw *game* totals (NBA 235/215) and the module
   docstring claims "NBA/WNBA points in the 200s" — but the snapshot's `O/U` column actually
   carries the **team-implied total**: `cli.py:193-198` overwrites `O/U` with
   `archive.get_total(...)`, and the archive's `Totals` bucket stores `(game_total ±
   spread)/2` per team (`moneylines.py:290-303`; league fallbacks NBA 111.667 / WNBA 81.667,
   `archive.py:286-292`). The live snapshot confirms: NBA `O/U` spans 105.8–111.0, WNBA
   75.1–91.7. Since `total.median() ≈ 108 ≤ 215` always, `_classify_shape`
   (`thesis.py:105-118`) returns **"grind" for every non-lopsided game in every league** —
   the shootout, coinflip, and even cells are dead inventory. This is precisely the class of
   per-league magic-number bug the league-general requirement exists to kill.

v2 fixes all five: an archetype taxonomy with a leg-driven classifier, a pure
`thesis(legs, ctx)` core the dashboard can call live, a precomputed per-game context
artifact, and league-relative normalization with per-sport voice banks.

## Confirmed facts (verified this session)

- **Package inventory.** `stories/` = `thesis.py` (229 lines), `bank.py` (314), `why.py`
  (123), `legs.py` (31), `__init__.py` (`STORIES_VERSION = "p2-1"`, line 22). All imports are
  pure (hashlib/collections/dataclasses/pandas); no `Archive` anywhere in the package. The
  dashboard already imports it (`dashboard/legs.py:11` re-exports `parse_leg`), and
  `tests/golden/test_dashboard_no_archive_lock.py:84-95` auto-discovers every dashboard
  module, so the import-safety hard rule is already pinned for this path.
- **Leg strings.** Parlay rows carry up to six `"Leg N"` strings in the format
  `"{Player} {Bet} {Line} {Market} - {Model P}%, {Boost}x"` built by `_build_cmarket_desc`
  (`correlation.py:352-362`); `parse_leg` (`legs.py:8-31`) recovers Player/Bet/Line/Market.
  Leg markets are platform display names ("Pts + Rebs + Asts"), canonicalized via
  `stat_map[platform]` when joining offers (`dashboard/legs.py:16-32`).
- **`Moneyline`** in `current_offers` is the team's implied win probability in (0,1) — not
  American odds. `to_american` (`dashboard/components/deep_dive.py:29-35`) converts it for
  display; `archive.get_moneyline` falls back to 0.5 (`archive.py:415-424`). Live range
  0.284–0.716. Each offer row carries *its own team's* probability.
- **`O/U`** is the team-implied total in raw points (see Problem #5). Therefore **both the
  game total and the spread are derivable with no new plumbing**: for a game's two teams,
  `game_total = implied_A + implied_B` and `spread = |implied_A − implied_B|` — exact by
  construction of the archive write (`moneylines.py:295-302`). `Archive` has `get_moneyline`
  / `get_total` / `get_team_market` but **no `get_spread`** (grep of `helpers/archive.py`);
  none is needed.
- **`Avg 5` / `Avg H2H`** are average-MINUS-line deviations (`model_prob.py:328-329`);
  `Avg H2H == 0` is the no-history sentinel (`model_prob.py:330`, honored in `why.py:62-76`).
- **`DVPOA`** = the `Defense position` feature (`model_prob.py:332`), defense-vs-position
  over average; live range ≈ [−0.30, +0.60] with most mass inside ±0.05 (`why.py:24-27`).
- **`Player position`** is computed twice and persisted never: `model_prob` exports it as an
  int depth-slot in [−1, 4] (`model_prob.py:333-339`); `_resolve_player_positions`
  (`correlation.py:264-318`) resolves it to usage-ranked labels (`G1`, `QB1`, `B3`) **on the
  league slice only** — the labels are never written back to the returned offers frame — and
  `persist._OFFER_KEEP_COLS` (`persist.py:39-74`) drops the int (comment, lines 36-38).
  Per-league position vocab + usage stats live at `correlation.py:73-85`.
- **`current_game_corr.parquet` exists** (P2): columns `League, Game, leg_a, leg_b, rho`,
  leg key `"{Player}|{Market}|{Bet}"` with `Market` canonicalized via `stat_map[platform]`
  so it joins offers (`correlation.py:456-485`, writer `persist.py:125-137`, dashboard
  loader `dashboard/data.py:202-213`). `rho` is bet-signed: the C matrix entries already
  flip sign for opposing bet directions (`correlation.py:185-187`).
- **`Game`** is the canonical `"/".join(sorted([team, opp]))` matchup key on offers
  (`correlation.py:645-649`), parlays, and the corr slice.
- **Pipeline order** (`prediction/cli.py`): `process_offers` → market remap to canonical
  codes (`cli.py:144,172`) → `O/U` overwrite (`193-198`) → `attach_offer_why` (202) →
  `attach_parlay_theses(parlay_df, snapshot_offers)` (203) → `write_current_offers` /
  `write_current_game_corr` (205-212). Context can therefore be built pipeline-side from the
  finished offers frame with zero archive reads beyond what already ran.
- **Pace is not available league-generally.** Only NBA gamelogs carry `PACE` / `E_PACE`
  (`stats/nba.py:119,201-202`); WNBA/NFL/NHL/MLB have no equivalent column. Spread is
  derivable (above); pace is not — it stays an open item, with `total_ratio` (below) as the
  tempo proxy.
- **Slips surface** reads the precomputed `Thesis` by taking the first non-null within a
  family (`dashboard/surfaces/slips.py:38-47`), so switching theses from per-family to
  per-leg-set requires no Slips change to stay rendering.
- **Determinism contract** (lane brief §4, locked): deterministic templates only, no LLM, no
  paid APIs; diversity = bank size + context keying + the md5 date seed
  (`thesis.py:183-184`). Free-LLM rewriter is registered as L6, never a dependency.
- **Live snapshot evidence**: `current_offers.parquet` on disk (875 rows, meta
  `generated_at 2026-06-11T21:53Z`) predates P2 (no `Game`/`K`/`Why`, no `stories_version`
  in meta) — written by pre-P2 code; the column ranges quoted above come from it and from
  the code path that regenerates it.

## Design

### 1. Archetype taxonomy + classifier

Four archetypes. The classifier is a pure, deterministic function of the leg-set plus game
context — no Family input anywhere.

| Archetype | Story | Fires when (league-agnostic gates, named constants) |
|---|---|---|
| **player** | The v1 case: a featured player drives the slip | One player holds ≥ `_PLAYER_LEG_SHARE` (0.5) of legs **and** ≥ `_PLAYER_MIN_LEGS` (2) legs. The strict gate is the fix for weakness #2 — no share majority, no star. |
| **stack** | Correlated same-game bundle that rises together | ≥ `_STACK_MIN_LEGS` (3) legs in the primary game, ≥ 2 distinct players, and mean pairwise bet-signed ρ over the slip's leg pairs ≥ `_STACK_MEAN_RHO` (0.10), read from the corr slice. |
| **unit** | Positional/team unit vs a soft matchup ("NYK guards feast on a weak SAS perimeter") | ≥ `_UNIT_MIN_LEGS` (2) legs share (team, position group, direction) and that group's aggregated DVPOA edge ≥ `_UNIT_EDGE_FLOOR` (0.05 — same floor `why.py:27` uses per offer). Position group = `Position` label minus its rank digit (`G1`→`G`). |
| **game-script** | The game shape is the story; no player named ("shootout lifts every stat line", "grind → unders") | Context shape ∈ {shootout, grind, blowout, coinflip}. Always available as the no-standout answer for mixed/level slips — and for game-line-heavy slips once L3 lands. |

**Precedence:** player → stack → unit → game-script. Rationale: when a true star clears the
strict share gate, the star sells the slip (and the player cells are already shape-keyed, so
the script still colors the prose); a correlated stack is a stronger specific claim than a
unit edge; game-script is the universal floor. **Fallback:** if nothing fires and shape is
"even", render from the game-script "even" cells ("edges scattered across {g}" voice) —
never a forced star. This replaces v1's alphabetical election outright.

**Multi-game slips** (the rail allows them; cross-game pairs are ρ=0 by design): the
classifier picks the **primary game** — most legs, ties broken by sorted game key — and
tells that game's story; the player archetype may still fire on a cross-game star since a
player belongs to one game. Richer "multi-game ladder" phrasing is an open item.

### 2. `thesis(legs, ctxs)` — the pure core

```python
@dataclass(frozen=True)
class Leg:        # enrich_legs(parsed, offers) builds these; engine never touches a frame
    player: str; bet: str; line: float; market: str
    game: str | None; team: str | None; position: str | None; category: str

@dataclass(frozen=True)
class GameCtx:    # one per game; built from current_game_context + corr-slice rows
    league: str; game: str; date: str; shape: str
    game_total: float | None; total_ratio: float | None
    fav_team: str | None; ml_margin: float | None; spread: float | None
    pos_edges: Mapping[str, Mapping[str, float]]      # {team: {pos_group: mean DVPOA}}
    rho: Mapping[frozenset[str], float]               # {{leg_key_a, leg_key_b}: rho}

def thesis(legs: Sequence[Leg], ctxs: Mapping[str, GameCtx]) -> str
```

Pure and cheap: no I/O, no archive, no randomness, no pandas required at call time. Variant
selection keeps the md5 scheme with the family term replaced by the leg-set:
`md5(f"{game}|{sorted leg keys}|{date}|{shape}|{archetype}")` — deterministic per snapshot,
rotating day to day, identical between prophecize and the rail because `date` is the slip's
snapshot date, not wall clock.

`attach_parlay_theses(parlays, offers, *, corr=None, context=None)` becomes a thin pipeline
adapter: parse each parlay row's legs, enrich, build `GameCtx`s once per game, call
`thesis()` per **distinct leg-set**, then run the existing slate-uniqueness pass
(`_dedupe_slate` machinery, `thesis.py:195-229`, moves intact) over distinct leg-sets within
`(League, Date)`. Rows with identical leg-sets share one headline. The rail's live recompute
skips the slate pass — a single slip has nothing to collide with, and faithfulness beats
slate uniqueness there.

Family is gone from the thesis path entirely. The `Family` column stays on
`current_parlays.parquet` (append-only schema) and Slips keeps using it to group — that and
ordering are its whole remaining job.

### 3. `current_game_context.parquet` — the precomputed game context

New snapshot artifact, written by `persist.py` beside the corr slice, path constant in
`helpers/io.py` (`CURRENT_GAME_CONTEXT_PATH`). One row per `(League, Game, Date)`, built by
a pure `build_game_context(offers, baseline_totals)` aggregation over the finished
`snapshot_offers` frame — no archive reads (cli passes `archive.default_totals` in; the
stories package itself never imports the archive, preserving import-safety):

| Column | Type | Source |
|---|---|---|
| `League`, `Game`, `Date` | str | offer group keys |
| `game_total` | float | sum of the two per-team median `O/U` values (NaN if one side absent) |
| `spread` | float | abs difference of the two team-implied totals — the de-vigged expected margin, exact per `moneylines.py:295-302` |
| `fav_team` | str | team with max `Moneyline` |
| `ml_fav_prob`, `ml_margin` | float | max implied win prob; `max|p − 0.5|` |
| `total_ratio` | float | `game_total / baseline` (see §4 — the league-relative tempo signal) |
| `baseline_total` | float | the denominator used (transparency/debug) |
| `shape` | str | shootout / grind / blowout / coinflip / even, from the shared classifier in §4 |
| `pos_edges` | str (JSON) | `{team: {pos_group: {"dvpoa": mean, "n": count}}}` aggregated from per-offer `DVPOA` × the kept `Position` labels |
| `n_offers` | int | rows aggregated |

Two prerequisites, both in-footprint (`prediction/correlation.py`, `persist.py`):

- **Keep the position labels.** `_resolve_player_positions` writes its `G1`/`QB1`/`B3`
  labels back onto the main offers frame as a new string column **`Position`** (the int
  `Player position` stays internal and dropped). Combo/`vs.` legs resolve to empty — they
  are excluded from `pos_edges`. `Position` is appended to `_OFFER_KEEP_COLS` (append-only:
  new column, nothing renamed or removed).
- **Stack evidence is already precomputed** — the engine consumes the existing
  `current_game_corr.parquet` rows; the context artifact does not duplicate them.

The dashboard reads the parquet (new `load_current_game_context` in `dashboard/data.py`,
mtime-cached like its siblings) and reconstructs `GameCtx` objects via a shared pure codec
in the stories package — the dashboard never recomputes context from offers.

### 4. League-general normalization + per-sport voice banks

**Normalization (kills the #5 bug class).** Shape classification moves off raw-point bands
onto one league-agnostic pair of ratio bands over `total_ratio`:
`_SHOOTOUT_RATIO = 1.05`, `_GRIND_RATIO = 0.95`, with `_ML_LOPSIDED_MARGIN = 0.18` kept from
v1 (already league-agnostic — it lives in probability space). The baseline denominator is
the **slate median game total for the league** when the slate has ≥ `_BASELINE_MIN_GAMES`
(4) games, else `2 × default_totals[league]` (the league constant the archive already
carries). Sanity check against v1's intended NBA bands: 235/223.3 ≈ 1.05 and 215/223.3 ≈
0.96 — the ratio bands reproduce the intent of the hand-written points bands without any
per-league table. `_TOTAL_BANDS` and `_DEFAULT_TOTAL_BAND` are deleted; one classifier
serves NBA/WNBA/NFL/NHL/MLB and any future league.

**Voice banks.** The bank grows an archetype axis and a sport axis. Cell key becomes
`(sport_voice, archetype, shape, direction, category)` with a deterministic fallback chain:

```
(voice, archetype, shape, dir, cat) → (shared, archetype, shape, dir, cat)
  → (shared, archetype, "even", dir, cat) → (shared, archetype, "even", dir, "production")
```

guaranteeing a hit for every input (same philosophy as `_bank_cell`, `bank.py:96-103`).
Sport voices: `basketball` (NBA + WNBA — same vocabulary, one module per the
consolidation rule), `football` (NFL: ground game / pocket / secondary / front),
`hockey` (NHL: forecheck / power play / crease), `baseball` (MLB: lineup / bullpen /
strike zone). `shared` holds the league-neutral structural cells that make the fallback
total. The v1 107-variant bank becomes the basketball + shared content of the **player**
archetype, so no prose is thrown away. Template slots per archetype: player `{p}`/`{g}`;
unit `{team}`/`{grp}`/`{opp}`; game-script `{g}`; stack `{n}`/`{g}`/`{p}` (anchor player).

**Adding a league** = adding (or pointing at) one sport-voice module and nothing else: the
classifier, normalization, categories, and fallback chain are league-blind. A
bank-coverage golden test parametrized over all five leagues × archetypes × shapes ×
directions asserts the chain resolves and that each sport voice defines at least its
player- and game-script-archetype cells.

`_STAT_CATEGORY` (`bank.py:13-85`) survives but matches on canonical market codes after
`enrich_legs` canonicalizes via `stat_map[platform]` (today it substring-matches raw
display names — the canonicalization tightens needle collisions like `"pa"`/`"pr"`).

### 5. Live regeneration

- **Precompute (unchanged surface):** prophecize keeps writing `Thesis` onto every
  `current_parlays.parquet` row — Tonight/Slips render instantly from the snapshot, exactly
  as today. `stories_version` in `current_meta.json` (`persist.py:119`) bumps per stage.
- **Rail recompute (new, P3):** the slip rail calls `thesis(legs, ctxs)` on every slip edit
  — add/remove/swap — via a thin `dashboard/slip/story.py` (lane §5 footprint). Legs come
  from the rail's slip state enriched against the loaded offers frame; `GameCtx`s come from
  `load_current_game_context` + `load_current_game_corr`. An **unedited** loaded slip shows
  the precomputed `Thesis` verbatim (it may carry a slate-uniqueness bump the pure function
  can't see); the first edit switches the headline to the live recompute. A slip whose game
  is missing a context row degrades to `ctx=None` → shape "even" routing — never a crash.
- This satisfies precompute-first (lane §4/§7): the only live computations remain the slip
  joint-prob and this pure, frame-free template render.

### 6. File layout (300-line cap)

```
prediction/stories/
  __init__.py      re-exports: thesis, attach_parlay_theses, attach_offer_why, parse_leg,
                   enrich_legs, build_game_context, GameCtx, Leg; STORIES_VERSION
  legs.py          parse_leg (as-is) + enrich_legs (leg ← offer-row enrichment, pure)
  why.py           unchanged (per-offer case is out of v2 scope)
  context.py       NEW — GameCtx/Leg dataclasses, build_game_context, parquet↔ctx codec,
                   shape classifier + ratio-band constants
  engine.py        NEW — thesis(legs, ctxs), archetype classifier, variant seed,
                   slate-uniqueness machinery (moved from thesis.py)
  thesis.py        slims to the pipeline adapter attach_parlay_theses (public name kept)
  bank/
    __init__.py    cell lookup + fallback chain + _STAT_CATEGORY
    shared.py      league-neutral cells (all archetypes)
    basketball.py  NBA+WNBA voice    football.py  NFL voice
    hockey.py      NHL voice         baseball.py  MLB voice
```

Each file stays under 300 lines; the bank subpackage exists precisely because four
archetypes × five shapes × two directions × six categories cannot fit one file at v1's
variant density. Dashboard side: `dashboard/slip/story.py` (P3) and a Tonight/Game consumer
of the context strip (P4) — both thin callers, keeping `test_dashboard_no_archive_lock.py`'s
auto-discovery green.

## Data flow

```
prophecize (cli.py)
  process_offers → find_correlation ──────────────┐
    └ Position labels written back (corr.py)      │ corr_sink rows
  O/U overwrite → snapshot_offers                 ▼
    ├─ build_game_context(offers, default_totals) ──> current_game_context.parquet  (NEW)
    ├─ attach_offer_why(offers)                      current_game_corr.parquet      (P2)
    └─ attach_parlay_theses(parlays, offers,
         corr=sink, context=ctx_rows)                current_offers.parquet (+Position)
           └ thesis(legs, ctxs) per leg-set          current_parlays.parquet (+Thesis)
                       ▲                                        │
                       │ same pure function                     ▼ snapshots only
            dashboard rail (P3): slip edit ──> thesis(legs, ctxs) live; Tonight/Game (P4)
            read context strip + headlines from the parquets
```

## Staged rollout

Five stages, T1–T5, each independently shippable and ending green on the lane §9 gates
(refactoring-specialist, ruff, golden, integration `-n0`). T1–T3 are P2-style precompute
extensions that slot before/alongside P3; T4 rides P3; T5 rides P4.

- **T1 — context precompute + position keep (no UX change).** `stories/context.py`,
  `current_game_context.parquet` writer + `io.py` path + cli wiring; `Position` write-back
  in `correlation.py` + keep-col; v1's `_game_shapes` delegates to the shared ratio-band
  classifier (deleting `_TOTAL_BANDS` — this alone fixes Problem #5 while keeping every v1
  template). Acceptance: context-builder golden (synthetic two-team offers → exact row,
  game_total/spread/ratio hand-computed); persist characterization extended; shape
  distribution on a synthetic slate hits all five shapes (the all-grind repro now fails);
  integration `-n0` shows the new file + column flowing; dashboard untouched and green.
- **T2 — engine refactor: archetypes + classifier.** `stories/engine.py`,
  `thesis(legs, ctxs)`, per-leg-set theses, family demoted, bank gains the archetype axis
  (player cells = v1 content; minimal shared-voice game-script/stack/unit cells).
  Acceptance: routing unit tests per archetype gate (incl. the no-standout fixture routing
  to game-script, *not* an alphabetical star); determinism test (two calls byte-equal);
  slate-uniqueness test stays green; stories goldens re-pinned; Slips still renders
  (first-non-null lookup unchanged).
- **T3 — per-sport voice banks.** `bank/` subpackage split, five-league coverage seeded
  (basketball voice inherits v1; football/hockey/baseball seeded with ≥ 2 variants per
  reachable player + game-script cell), canonical-code category matching. Acceptance:
  bank-coverage golden (every league × archetype × shape × direction resolves; no
  KeyError); per-league fixture slates render sport-correct vocabulary; headline-uniqueness
  bar (lane P2) still holds on a dense synthetic slate.
- **T4 — rail live-regen (P3).** `dashboard/slip/story.py` + `load_current_game_context`;
  recompute on every slip edit; unedited slips show the precomputed string. Acceptance:
  unit/AppTest — removing the named player changes the headline and the old name never
  renders; missing-context slip degrades to "even" routing; archive-lock golden green.
- **T5 — narrative surfaces (P4).** Tonight cards + Game context strip read
  `current_game_context.parquet` (total, spread, favorite — replacing the per-row
  `O/U`/`Moneyline` peek at `surfaces/game.py:51-56`); top prophecy headline per game from
  per-leg-set theses. Acceptance: render pins; View-game param fix per the P4 brief; strip
  shows derived spread for a fixture game.

## Testing

Per-stage acceptance above, plus: the golden suite stays macro-level (engine + `attach_*`
inputs→exact strings; no per-helper characterization pins, per the test-cull philosophy);
determinism is asserted at the `thesis()` boundary (same inputs, same date → byte-equal);
`pytest -m integration -n0` proves the new artifact and columns flow end to end in fake
mode; `test_dashboard_no_archive_lock.py` remains the import-safety gate for every new
module the dashboard touches. Refactoring-specialist runs on every touched `.py` before any
push (CLAUDE.md five-trigger rule).

## Out of scope

- The L6 free-LLM rewriter (later optional seam; templates stay the contract).
- Game-line legs as story subjects (waits for L3 — game lines into the correlation engine).
- Pace plumbing (no league-general source; see Open items).
- `why.py` per-offer case improvements (v1 behavior kept).
- Archive schema, `get_spread` (unneeded — derived), crontab, `training/` / `stats/` /
  `strategies/` internals, `stat_meta.json`.
- Renaming or removing any existing snapshot column (append-only, lane §7).

## Open items for the implementation plan

1. **Pace source — DECIDED (owner, 2026-06-12): skip.** No pace plumbing; `total_ratio`
   is the tempo proxy. NBA-only `PACE` in gamelogs is not worth a per-league build.
2. **Baseline refinement.** T1 ships slate-median with the `2 × default_totals` fallback.
   A rolling 30-day league median from the archive would be steadier for 1–3-game slates
   (finals weeks); also note NHL's `default_totals` (2.674 ⇒ 5.35 game total) reads low vs.
   the modern ~6.1 — owner to verify/refresh that constant before NHL season.
3. **Stack threshold.** `_STACK_MEAN_RHO = 0.10` is a reasoned seed; tune against real
   slates once T2 lands (print a routing board for a week of snapshots).
4. **Multi-game phrasing.** Primary-game rule ships; a true "three-game ladder" voice is a
   later bank addition.
5. **WNBA voice — DECIDED (owner, 2026-06-12): shares the basketball voice.** No distinct
   WNBA module; the bank key still supports adding one later if it is ever wanted.
6. **Display grain on Slips.** Theses become per-leg-set; Slips keeps rendering one
   headline per family group until the P5 board upgrade decides the final grain.
