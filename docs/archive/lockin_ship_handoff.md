# GBDT mean-regression — baseline lock-in results & devel ship handoff

**Date:** 2026-05-21
**Research branch:** `claude/fix-gbdt-mean-regression-GcY1g`
**Status:** Lock-in complete for NBA + WNBA + NFL. **36 cells locked** in
`data/ship_config.json` (NBA 13, WNBA 10, NFL 13), all full-HP Tier-0 confirmed.
Nothing has shipped to production yet — this document is the handoff for the
`devel-ship-curator` to carry it there.

---

## 1. What must reach `devel` first — foundation-update scope

Production tracks `devel`. `devel-foundation` (cut 2026-05-21 12:58, the one-time
production-runtime foundation) is **8 commits ahead of `devel`, not yet merged**,
and is **missing two production-runtime deltas** added on the research branch
*after* it was cut. Both are foundation-grade (production code, not per-market
data) and **must be folded into `devel-foundation` before any cell ships**:

| Delta | Files | Commit | Why it is foundation-grade |
|---|---|---|---|
| **TOR expansion-team fix** | `src/sportstradamus/stats/base.py` (`_profile_rows_for_teams`) + `tests/golden/test_profile_team_lookup.py` | `c9fcf01` | Without it, production WNBA `meditate`/`prophecize` crashes with `KeyError: ['TOR']` (Toronto Tempo, 2026 expansion). Live bug for every WNBA cell. |
| **Per-cell `zinb_mode` plumbing** | `training/baselines.py` (`ZINB_MODES`), `training/ship_config.py` (object-form cells + `resolve_cell_zinb_mode` + `_validate_cell`), `training/cli.py` (wiring), `training/pipeline.py` (validation) + `tests/test_ship_config.py` | (this branch HEAD) | Lets a `ship_config` cell pin `zinb_mode` via the object form `{"strategy", "zinb_mode"}`. If an object-form cell reaches a `devel` whose `ship_config.py` is still string-only, `load_ship_config` raises `ValueError` and the `meditate` cron dies at startup. |

**Order of operations (all carved by `devel-ship-curator`, which enforces the
diagnostic denylist — no `compression_eval` / `zinb-routing-diagnostics` /
`icc-diagnostics` / `statsmodels`):**

1. Update `devel-foundation` with the two deltas above (a "further foundation layer").
2. Merge `devel-foundation` → `devel` (the first foundation merge; production then
   runs the ship system for the first time).
3. Layer the per-market ship deltas (the `ship_config.json` blocks below + trained
   pickles) on top.

> **Note on the object form:** after NFL pruning (§4), **no `ship_config` cell
> currently uses the object form** — sacks-taken, the plumbing's motivating cell,
> failed full-HP Tier-0 and was pruned. The `zinb_mode` plumbing therefore ships
> as **tested infrastructure with no live consumer yet**, ready for the first
> hurdle cell that passes. It is still required on `devel` because the string-only
> loader and the object-aware loader are the same file; shipping the new
> `ship_config.py` is what makes a future hurdle cell a one-line edit.

---

## 2. Locked `ship_config.json` (the per-market ship payload)

All 36 cells passed full-HP Tier-0 (bench bias one-sided ≤ bound, star bias
bidirectional ≤ bound, `brier_skill_score` ≥ 0). `ratio_meanyr` is the incumbent.

### NBA — 13/13 (commit `b5d2609`)
All `ratio_meanyr`: DREB, FG3A, FGA, FGM, FTM, MIN, PA, PR, PRA, PTS, RA, REB,
fantasy points prizepicks. (Crippled-HP "centered wins" were 0–2 pp noise;
incumbent confirmed best-or-equal.)

### WNBA — 10/10 (commit `c9fcf01`)
`DREB` → `centered_additive_mean10`; the other 9 → `ratio_meanyr` (FGA, MIN, PA,
PR, PRA, PTS, RA, REB, fantasy points prizepicks). Unblocked by the TOR fix.

### NFL — 13/16 baseline-able (this branch HEAD)
`passing first downs` → `centered_additive_eb_meanyr_k10`; `receptions` →
`centered_additive_mean10`; the other 11 → `ratio_meanyr` (attempts, carries,
completions, fantasy points prizepicks, fantasy points underdog, passing yards,
qb yards, receiving yards, targets, tds, yards).

---

## 3. Breadth vs the 75% North Star

| League | Locked | All markets | North Star (75%) | Gap |
|---|---|---|---|---|
| NBA | 13 | 13 baseline-able / (full set) | met for baseline-able | — |
| WNBA | 10 | 18 | 14 | **−4** (Step-2 count markets: FG3M/FTM/STL …) |
| NFL | 13 | 20 | 15 | **−2** (see §4) |

---

## 4. NFL full-HP Tier-0 verdict (the analysis)

The deterministic screen (crippled HP) flagged 16 NFL cells as baseline-able.
A fresh `--force` full-HP retrain (real Optuna), scored on the regenerated
test sets, **confirmed only 13**. The screen was optimistic for 3 cells:

### Shipped (13) — `brier_skill_score`
| Market | Strategy | BSS |
|---|---|---|
| tds | ratio_meanyr | +0.756 |
| qb yards | ratio_meanyr | +0.198 |
| passing first downs | centered_additive_eb_meanyr_k10 | +0.139 |
| targets | ratio_meanyr | +0.135 |
| fantasy points underdog | ratio_meanyr | +0.115 |
| fantasy points prizepicks | ratio_meanyr | +0.093 |
| passing yards | ratio_meanyr | +0.085 |
| receptions | centered_additive_mean10 | +0.074 |
| yards | ratio_meanyr | +0.060 |
| attempts | ratio_meanyr | +0.037 |
| carries | ratio_meanyr | +0.032 |
| completions | ratio_meanyr | +0.027 |
| receiving yards | ratio_meanyr | +0.026 |

### Pruned (3) — failed full-HP Tier-0
| Market | Strategy tried | Failure | BSS |
|---|---|---|---|
| qb tds | ratio_meanyr | bench-bias +0.39 > 0.34 **and** star-bias +0.68 > 0.35 (over-predicts at full HP; screen under-predicted) | +0.076 |
| rushing yards | centered_additive_mean10 | BSS −0.006 < 0 (incumbent `ratio_meanyr` also failed it on the screen) | −0.006 |
| sacks taken | ratio_meanyr + **hurdle** | star-bias +0.67 > 0.50 | +0.003 |

### Never baseline-able on the screen (4) → Step-2 feature/bias track
interceptions, passing tds, rushing tds, receiving tds.

**NFL Step-2 pool = 7 cells** (the 3 pruned + the 4 above). Closing +2 of them
reaches the 15/20 North Star.

### Plumbing verification
The sacks-taken model pickle baked `is_hurdle=True`, `zinb_mode="hurdle"`,
`target_strategy="ratio_meanyr"` — the per-cell `zinb_mode` plumbing flows
end-to-end into a production hurdle model. The plumbing is **correct**; the cell
it was built for simply does not clear Tier-0 at full HP.

---

## 5. Gates & process

- All three gates green on the plumbing: `ruff` clean; `pytest tests/golden/
  tests/test_ship_config.py` = 166 passed / 6 skipped (15 are the new
  `ship_config` tests); `pytest -m integration` = 11 passed (determinism intact).
- `refactoring-specialist` run on every touched `src/` Python file (base.py;
  baselines/ship_config/cli/pipeline) — clean, no behavior change.
- TOR fix and `zinb_mode` plumbing both built TDD (RED→GREEN).

### Open follow-ups (not done here — for the curator / next session)
- Carve the foundation update (§1) and the per-market ship deltas (§2).
- NFL Step-2 feature/bias track (§4) to reach 15/20.
- WNBA Step-2 count markets to reach 14/18.
- `ZINB_MODES` lives in `baselines.py` (the per-cell-decision constants home);
  refactoring-specialist flagged a *future* move to a dedicated training-config
  module — deferred (5-file touch, zero behavior change).
