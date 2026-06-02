# Archive EV Repair Plan

Status: proposed (2026-06-01). Repairs the **local** `archive/archive.duckdb` that feeds
local training and rebuilds. The production/server archive is a separate, deferred concern
(see Phase 0).

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

`scripts/repair_matrix_ev.py` (in-place recover+reproject from the archive) is the wrong
approach — it reads the most-corrupt layer and made training worse in testing. Delete it; the
right tools are `backfill_historical_odds.py` + `inject_backfilled_odds.py`.
