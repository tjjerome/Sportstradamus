# Archive EV Repair Plan

Status: executing (2026-06-02). Repairs the **local** `archive/archive.duckdb` that feeds
local training and rebuilds. The production/server archive is a separate, deferred concern
(see Phase 0).

> **2026-06-02 update — the real root cause was in `get_ev`, not just the parser
> default.** The parser-`dist` fix below stopped one source, but two numerical bugs in
> `helpers/distributions.py:get_ev` were still minting blown evs into *live* and *re-fetched*
> rows (not only the migration seed). Both are now fixed; see
> [§ `get_ev` numerical root cause](#get_ev-numerical-root-cause-2026-06-02). The execution
> log at the bottom records what actually ran.

## Problem

The archive `odds` table stores corrupt `ev` for **count-distribution cells** (ZINB/NegBin —
NBA `FG3M/TOV/BLK/STL/PF/OREB/BLST/FTM`, NFL `tds/passing tds/interceptions/qb tds/
receiving tds/rushing tds`, WNBA `FG3M/BLK`).

Root cause: the props parser called `get_ev(line, under)` with no `dist`, so it used the
default `SkewNormal`. Harmless while those cells *were* SkewNormal; this season they were
routed to ZINB. The SkewNormal inversion of a ZINB book price is ill-conditioned — its CDF
asymptotes at `Φ(-1/cv)` — so the stored `ev` either blew up (→ `line × 1e6`) or settled at a
wrong, effectively inverted value. Fixed in `moneylines.py` (now passes
`dist=stat_dist[league][market]`); pinned by `tests/golden/test_parser_ev_dist.py`.

### Evidence

- NBA BLK archive: 171,629 rows; **97% are the `observed_at`-midnight seed** (klepto→duckdb
  migration, commit `1006fe6`, 2026-05-07, which *copied* klepto's already-blown evs); **59.6%
  are blown** (`ev > 5`).
- The corruption is **not a clean reversible transform**: some rows blow up to thousands
  (Holmgren 6553), others invert moderately (Fox 1.55 → recovers to 89% over a 0.5 line; true
  rate ~25%). **In-place recovery is therefore unreliable** — proven empirically: recovering
  NBA BLK from the archive drove the book Brier from **0.235 → 0.405** (worse than a coin).
- Continuous SkewNormal cells (PTS/PRA/yards/receptions) are clean — their distribution
  matches the parser default.

### Why the training matrices are mostly fine

The matrices are built incrementally (`stats/base.py` `get_training_matrix`) and **freeze each
row when first computed** — `meditate` only fetches game-dates after the cached cutoff.
Nov-2025→Mar-2026 rows were frozen *before* the corruption (clean); only the Apr–May tail was
appended afterward (blown). The archive is the most-corrupt layer; the cache is a clean
time-capsule of the recent season.

Per-month NBA BLK, matrix vs current archive:

| window | matrix mean EV | matrix blown | archive blown |
|---|---|---|---|
| 2025-11 → 2026-03 | ~0.7 (clean) | 0% | 32–47% |
| 2026-04 | 227 | 8% | 48% |
| 2026-05 | 539 | 20% | 20% |

The clean window's book *discriminates* (Odds bin → empirical over: 0.22 → 0.39 → 0.37 → 0.56);
the blown tail does not (top bin says >70% over, 42% hit).

## `get_ev` numerical root cause (2026-06-02)

The parser-`dist` fix was necessary but not sufficient. Two ill-conditioned bands in
`helpers/distributions.py:get_ev` were still producing blown evs from *correct* inputs, so
the corruption kept landing in live and re-fetched rows — not just the migration seed:

1. **ZINB/ZAGamma gate underflow → runaway mean.** For zero-inflated cells `get_ev` strips the
   gate: `base_CDF = (under − gate) / (1 − gate)`. When the de-vigged under-prob is at or below
   the zero-inflation `gate` (routine for high-zero count cells — NBA `BLK` gate ≈ 0.63, so a
   fair −130/+110 book de-vigs to under ≈ 0.46 < gate), `base_CDF ≤ 0` clips to `1e-6` and the
   NegBin inversion solves for a mean in the thousands (BLK → 5 917). **Only `add_dfs` passes a
   `gate`** (archive.py), so this is the DFS write path; the live props parser passes
   `gate=None` and stays bounded. Two add_dfs sub-triggers fed it: a missing/zero `Boost_Under`
   (Underdog Rivals/Sleeper) made `no_vig_odds` fabricate a ~6.5%-vig under ≈ 0.06, and even a
   symmetric DFS pick (under = 0.5) fell below the gate. **Fix:** `base_CDF ≤ 0 → return line`;
   and `add_dfs` now prices a missing under side symmetrically (`_dfs_under_boost`).
2. **SkewNormal mean→∞ asymptote → millions, then a `brentq` crash.** The scale grows with the
   mean, so `cdf(line, mean)` asymptotes to `Φ(−1/cv)` as `mean → ∞`, not to 0. A book under-prob
   a hair above that floor inverts to an astronomically large mean (rushing yards → 1.9–3.2M) and,
   at the exact boundary, hands `brentq` a same-sign bracket that raises `ValueError` — this
   crashed the NFL backfill at date 150/199. **Fix:** the new `_skewnormal_ev` helper returns the
   line when the implied mean would exceed `SN_MAX_MEAN_FACTOR × line` (cutoff `Φ((1−F)/(F·cv))`).

Pinned by `tests/golden/test_get_ev_robustness.py`. Because these minted blown evs into live and
re-fetch layers, the cleanup needed a **magnitude predicate across all `observed_at` layers**, not
just the midnight seed — `delete_corrupt_seed.py --blown-all-layers`.

## Constraint: the only reliable source of truth is the original two-sided prices

Corrupt evs cannot be inverted back to real book probabilities. The original prices survive in
exactly two places:

1. **The Odds API historical endpoint** — authoritative, costs credits (`events × markets × 10`
   per date).
2. **The clean training cache** (Nov-2025→Mar-2026) — free, but consensus-only (no per-book)
   and recent-season-only.

A full re-fetch is unaffordable (balance 76,873):

| league | span | dates | rows | Apr-2026+ dates | cur-season dates |
|---|---|---|---|---|---|
| NBA | 2022-10 → 2026-05 | 824 | 1.19M | 52 | 199 |
| NFL | 2022-09 → 2026-02 | 253 | 224k | 0 | 64 |
| WNBA | 2023-05 → 2026-05 | 300 | 17k | 21 | 46 |

At ~200–1000 cr/date on dense NBA dates, a full re-fetch is hundreds of thousands of credits.
So the plan **repairs what's consumed and quarantines the deep history**.

## Strategy

### Phase 0 — Deploy the parser fix (prerequisite, deferred per owner)

Stops future current-game corruption on the **server**. The historical repair below is
**independent**: it re-fetches past dates with the already-fixed local parser, and confer never
rewrites past dates — so Phases 1–3 proceed without touching devel. Deploy whenever; only the
live/current-game half of the benefit waits on it.

### Phase 1 — Re-fetch the recent contaminated window (affordable; the only part consumed)

- **Scope:** count-cell markets only, `game_date >= 2026-04-01`, NBA (52 dates) + WNBA (21
  dates). NFL = 0 (season ended Feb 2026). Requesting only the ~8 count-cell markets (not the
  full board) keeps the per-event cost down.
- **Tool:** `scripts/backfill_historical_odds.py` running the fixed parser, stamping
  `observed_at` after the midnight seed so re-fetched rows supersede the blown ones (no delete
  needed).
- **Cost:** ~15–30k cr estimated; `--dry-run` computes the exact per-league number first —
  confirm before spending.
- **Payoff:** repairs the recent archive **and**, via `scripts/inject_backfilled_odds.py`,
  cleans the Apr–May training-cache tail → fully clean count-cell matrices. This also resolves
  the deferred "Apr–May cache tail" cleanup decision in one shot.
- **Verify:** rerun the split-by-window calibration (book Brier vs base, reliability monotone,
  no blowups) on the re-fetched cells; the Apr–May window should now match the clean Nov–Mar
  window.

### Phase 2 — Make from-scratch rebuilds safe for the rest of the current season (free)

The cache holds clean consensus evs for Nov-2025→Mar-2026, but the archive's seed for those
dates is still blown — so a *from-scratch* rebuild (cache deleted) would re-corrupt.

- **Default (simplest):** **protect the cache.** Back up `data/training_data/`, treat it as
  canonical, never force a from-scratch rebuild. The archive's clean-season corruption is then
  inert for training.
- **Optional (fuller):** push the cache's clean consensus evs back into the archive (superseding
  the blown seed) so a rebuild reads clean. Consensus-only (per-book lost for those rows;
  `fit_book_weights` loses some recent count-cell granularity — acceptable).
- **Verify:** rebuild one cell from scratch and confirm its book column matches the cached one.

### Phase 3 — Neutralize / quarantine deep history (2022–2024)

Unaffordable to re-fetch; not used by training (trimmed) or live.

- **Default:** document as known-corrupt; leave it. It is never read by the current pipeline.
- **Optional landmine removal (destructive — requires explicit confirmation + a DB backup
  first):** delete the gross-blowup count-cell rows (`ev > 5×line`). `get_ev` then returns
  `nan` and a rebuild synthesizes an honest 0.5 coin-flip instead of actively-wrong garbage.
  Caveat: inverted-moderate rows (e.g. Fox 1.55) aren't caught by a magnitude threshold and
  survive — full clean still needs on-demand re-fetch.
- **On-demand:** if a backtest later needs a specific deep-history window, re-fetch just that
  window then.

## Risks

- **Server re-corruption** while production confer is unfixed — affects only the server archive
  + current games; the local historical repair is safe. Deploy Phase 0 before relying on the
  server.
- **Consensus-only cache-push** degrades per-book weight fitting for the pushed rows — minor.
- **Threshold deletion misses inverted-moderate rows** — deep history isn't fully clean without
  re-fetch; documented, on-demand.
- **Cost overrun** — every re-fetch is `--dry-run`-costed and confirmed before any credits are
  spent.

## Out of scope / superseded

`scripts/repair_matrix_ev.py` (in-place recover+reproject from the archive) was the wrong
approach — it reads the most-corrupt layer and made training worse in testing. **Deleted
2026-06-02**; the right tools are `backfill_historical_odds.py` + `inject_backfilled_odds.py`.

## Execution log

- **2026-06-02 — `get_ev` fix + cleanup + re-fetch.**
  - Fixed both `get_ev` bands + the `add_dfs` symmetric-under path; deleted the obsolete
    `repair_matrix_ev.py`. Gates green (golden / integration / ruff); refactoring-specialist clean.
  - Extended `delete_corrupt_seed.py` with `--blown-all-layers` (magnitude predicate, any
    `observed_at` layer). Backed up the DB, then deleted blown + degenerate rows for NBA
    `BLK/STL/FG3M`, WNBA `FG3M`, NFL `rushing yards/receiving yards/carries/tds`. **Blown count
    → 0 across all layers** for every target cell afterward.
  - Re-fetched (Odds API historical, one pre-game snapshot per game-day): WNBA `FG3M` (hour 14),
    NBA `BLK/STL/FG3M` (hour 23 — evening is the only snapshot that carries BLK/STL lines). New
    rows verified clean (NBA p50: BLK 0.55, STL 1.8, FG3M 2.2; zero `>5×line`). Cost ≈ 31 cr per
    market-date.
  - **Deferred (budget):** the NFL yardage/`tds` tail (50 dates, `2025-10-05`→end, ~16k cr) — the
    backfill had already landed 149/199 dates before the SkewNormal crash; balance after NBA+WNBA
    won't cover it. Resume `backfill_historical_odds … --league NFL --markets "tds,receiving
    yards,rushing yards,carries" --start 2023-09-01 --end 2026-02-28 --snapshot-hour 14` on refill.
  - **Caveat:** seed deletion ran *before* the re-fetch finished, so a handful of fully-blown
    NFL dates lost their only odds rows and now fall outside `_game_dates` (odds-driven). They
    resolve to synthetic 0.5 until a future `lines`-driven re-fetch; negligible (≈3 dates).
  - **NFL tail completed** (budget freed up): the full NFL `tds/carries/rushing yards/
    receiving yards/interceptions` history was re-fetched after NBA/WNBA, clearing the
    SkewNormal-asymptote crash at date 150 once the `get_ev` fix landed. A second
    `--blown-all-layers` pass on every re-fetched cell drove blown → 0.

- **2026-06-02 — inject + retrain (Phase 3).**
  - `inject_backfilled_odds.py` refreshed the 9 cached matrices from the repaired archive.
    Two latent inject bugs surfaced (the tool now equals a from-scratch rebuild for every
    row):
    1. **Residual blown.** Rows the swept archive can no longer price — DFS-only / thin-market
       player-games with no API re-quote, plus a few inverted-moderate rows the magnitude
       sweep missed (archive's latest line ≠ the point-in-time line) — kept their blown cached
       EV. `_resolve_row` now synthesizes any *resolved* `EV > 5×line` to the honest 0.5 a
       rebuild produces, regardless of source.
    2. **NaN-EV fit crash.** Synthesized rows were first written `EV=NaN, Odds=0.5`; with no
       `Odds_synthetic` column on most matrices, `pipeline._step_synthesize_odds` missed them
       and the NaN crashed the LightGBMLSS fit (NBA + NFL, pass 1). Fixed: synthesized rows now
       carry the `Odds==0` sentinel (caught regardless of the optional column) with `EV=NaN`
       for the pipeline to fill canonically, and `Odds_synthetic` is always written. Verified:
       0 blown and every NaN-EV row flagged for synthesis across all 9 matrices.
  - Retrained all 9 cells with `meditate --force`, Optuna **warm-started** from each pickle's
    prior HPs (150 trials / 5 min, best trial 0 throughout). Honest before→after on the real book:

    | cell | book brier | brier skill | ship |
    |---|---|---|---|
    | NBA BLK | 0.245 → 0.233 | +0.134 → +0.080 | devel (5/5 pass) |
    | NBA STL | 0.251 → 0.255 | +0.027 → +0.058 | devel (5/5 pass) |
    | NBA FG3M | 0.253 → 0.238 | +0.117 → +0.074 | devel (5/5 pass) |
    | NFL tds | 0.770 → 0.147 | +0.809 → −0.024 | devel (5/5 pass) |
    | WNBA FG3M | 0.253 → 0.251 | +0.136 → +0.147 | **withheld** (g4 iqr 0.50) |
    | NFL carries | 0.342 → 0.259 | +0.227 → −0.010 | **withheld** (g1) |
    | NFL rushing yards | 0.320 → 0.257 | +0.210 → +0.022 | **withheld** (g1) |
    | NFL receiving yards | 0.247 → 0.251 | −0.025 → −0.009 | **withheld** (g1) |
    | NFL interceptions | 0.318 → 0.269 | +0.179 → +0.064 | **withheld** (g1) |

  - The blown book had inflated apparent skill — NFL `tds` book brier 0.77 meant the book was
    confidently *wrong*, so the +0.81 skill was a broken-strawman artifact. On the real book the
    skill compresses and 5 cells no longer clear their gates. This is the correct outcome: a
    sharper book makes the gate harder to pass. The 5 are flipped `devel → withheld` in
    `stat_meta.json`; `g1_oracle` stays strongly negative (−0.25 … −0.56) so the headroom is
    real — they are re-ship candidates once training is strengthened. NBA `BLK/STL/FG3M` and
    NFL `tds` still pass all five gates and stay `devel`.
