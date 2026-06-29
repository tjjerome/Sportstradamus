# Book-shape WS2/WS3 — session hand-off

Living brief for the book-odds shape-free lane — current state + remaining work. Read this, then the
plan (`~/.claude/plans/review-src-sportstradamus-data-runtime-c-calm-hamster.md`) for the full design,
file seams, and the safety/validation checklist.

## Goal (one line)

Fix the book's implied distribution for count stats (it is rebuilt as a symmetric constant-CV
Gaussian, wrong for integer counts — e.g. Kamilla Cardoso AST u2.5 reads 84.5%, ~22pt too high at
low mean) **within the SkewNormal family** (no model-family switch, no model retrain), and make the
archive **shape-free** (store the devigged `(under_prob, line)` the book actually quoted; apply
distribution shape only at read time).

## Branch / commit map

| Branch | Has | State |
|---|---|---|
| `devel` (prod-tracking) | WS1 (`aaa0431` storage+migration, `298236f` confer pin, `391c5ee` research verdict) + `4c7d393` merge_archives fix + `37ff4c5` correlation synthetic-c_map fix | **pushed** |
| `model-research` | the same 6 book-shape commits cherry-picked + `3d00d7a` keystone doc polish | **pushed** |
| `ws2-book-shape` (off `devel`@391c5ee) | WS2 keystone `0c4aefd` + `f85069e` doc polish | **local only — WS2 continues here** |

WS2 lands on `devel` **only after** it is validated as one unit (gates + retrain). The keystone is
deliberately not on `devel` yet.

## Done

- **WS1 — shape-free storage** (additive, behavior-preserving; reads untouched). `odds` table now
  carries `under_prob`/`line` beside `ev`; local archive migrated (13.87M player-prop rows, ev
  byte-identical). Confer write-path keeps writing the shape-free quote. Details: [[ws1-archive-shapefree]].
- **merge_archives fix** — the sync rebuild dropped `under_prob`/`line` on every merge; now dedups on
  the observation identity and preserves the quotes (`_odds_select`). Same trap applies to the prod
  backfill (below).
- **WS2 keystone** — `skewnormal_params_from_moments(var, skew) -> (sigma, alpha)` in
  [helpers/distributions.py](../../src/sportstradamus/helpers/distributions.py) (Azzalini inversion,
  skew clamped just inside the 0.99527 bound). Validated exact against the research brief's hand-solved
  DREB bins. **This is the result that keeps the cell set in-family** — no COM-Poisson, no model retrain.
  Pinned by `tests/golden/test_distributions_characterization.py`.

## Next (ordered — validate each before the next)

### 1. WS2 foundation (additive, safe — no behavior change yet)
- `training/calibration.py` `fit_book_shape(league, market, results, lines) -> dict | None`: bin raw
  `Result` by `Line` (min ~120 rows/bin); per bin compute (mean, var, skew); fit `var = a·μ^b` (Taylor,
  sqrt(n)-weighted in log space, **`b` FREE to cross `var=μ`** — DREB is sub-Poisson b≈0.34, do NOT force
  b≥1); fit `γ(μ)` linear then **clamp** to the feasibility band (never extrapolate the raw linear — that
  produced the spurious DREB −1.55). Sparse-cell fallback → constant-cv, logged. Return `{a, b, skew_c,
  skew_d, n_bins}`; `None` when too few bins. Do **not** wire it into the `meditate` pipeline yet.
- `helpers/config.py` `book_skewnormal_shape(league, market, μ) -> (sigma, skew_alpha)`: eval the curve,
  call the keystone; **strict no-op fallback `(μ·cv, 0.0)` for unfitted cells** so the build stays
  behavior-preserving until step 2. Surface per-cell coeffs from `stat_calibration.json` beside `cv/std/zi`.
- Golden tests: planted-coeff recovery on a synthetic cell; round-trip through `get_odds`; unfitted-cell
  fallback. (A `--zinb-mode`/`2am` autonomous build was scoped for exactly this step.)

### 2. WS2 thread (behavior-changing — the delicate part; per-module subagents)
Feeds the **training read path**, so each site gets its own equivalence check. Sites:
- `prediction/model_prob.py` `_book_over_prob` / `book_fallback_prob` → use `book_skewnormal_shape`.
- `helpers/distributions.py` `fused_loc` book side (~L597-610) → book `(sigma, skew_alpha)` from the helper
  (model side + pickle decode untouched).
- archive read-boundary re-encode — `get_ev` / `_book_rows` / `to_pandas` rebuild `ev` from
  `(under_prob, line)` at the **fitted** shape (this is the behavior change WS1 deliberately deferred).
- `stats/base.py:2009` training feature → prob-space `1 - composite_under_prob` (mandatory in WS2;
  ~10-30bp Jensen change, shape-invariant thereafter).

### 3. WS2 consensus
`helpers/archive.py` `_weighted_book_ev` → WLS-on-CDF one-point fit with inverse-binomial-variance weights
`n/(p(1−p))` + isotonic monotonicity (Breeden-Litzenberger). **Ladder lift DEFERRED** — the `ladder` table
is empty (0 rows vs 19.2M odds), so WS2 consensus ≈ today's + the shape shift.

### 4. WS2 validate (the load-bearing gate)
- Round-trip invertibility across the `(line, under)` domain + real archived samples (fail loud, never
  write a garbage EV).
- **Open-Q1 settling experiment — do this BEFORE the full build:** co-fit `(shape, w)` over ≥8 folds via the
  reusable invert-blend A/B harness; **require the shaped book to beat the symmetric baseline OOS at the
  refit `w`, else collapse to constant-cv.** The served-gate lift is a research bet decoupled at `w≈0.90`
  ([[book_distribution_audit_nogo]], [[pooling_half_blp_nogo]]); the book-leg/Cardoso fix is CERTAIN.
- Gate revalidation (g4/g5/PIT-KS) via `training/scorecard.py` A/B on the SkewNormal cohort; retrain affected
  cells; live spot-check **Cardoso AST u2.5 84.5% → ~79%**, Kelly shrinks, no new "implausible book EV" warns.
- Then `refactoring-specialist` on every touched `.py`, the three gates, and merge to `devel`.

### 5. WS3 — shape-free team markets (queued)
Back **spread + game total** out to `(line, under_prob)` (moneyline already shape-free). Drop them from
`_TEAM_ONLY_MARKETS` skip; extend `set_team_books` write + the migration; handle the totals total/2-per-team
blend. Same archive-safety discipline as WS1.

### 6. Prod backfill (deferred, user-gated)
Migrate the production `archive.duckdb` on the server (manual, quiet ~3-6am window) **after WS2 validates**.
**CRITICAL:** back it up first, and never rebuild the `odds` table with a fixed column list — that silently
drops `under_prob`/`line` (the bug `4c7d393` fixed in `merge_archives`). See [[ws1-archive-shapefree]].

## Functional forms (from the brief / §10 verdict)

- `var = a·μ^b` (Taylor's power law), `b` free. REJECT NegBin/Gamma `var = μ + kμ²` (k<0 inadmissible on DREB).
- `γ(μ)` clamped/splined in `[max(δ−1/δ, −0.99), 0.99]`; the real SN violation is the **low-mean right-skew**
  (AST line 0.5 skew ≈ 1.26, Poisson(1) floor) — clamp α + the existing integer PMF correction
  ([calibration.py:453](../../src/sportstradamus/training/calibration.py#L453)); cost ~1.7pt P(under) vs the
  variance fix's ~22pt.
- Line-binning is correct (the line is the book's posterior mean; within/between inflation 5-7% < 15% Mundlak).

## References

- Plan: `~/.claude/plans/review-src-sportstradamus-data-runtime-c-calm-hamster.md`.
- Research verdict + forms: `docs/handoffs/model_improvement_track.md` §10 (the WS2 book-shape verdict entry). Fuller brief
  `/tmp/researcher_ws2_book_shape.md` (may be gone after a reboot — §10 holds the load-bearing conclusions).
- Refs [82]-[85] (Azzalini/Taylor/Cobain/Busetti) in `docs/operation_ship_references.md`.

## Commands

```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/                    # parallel
poetry run pytest -m integration -n0               # fake-mode, no network
# A/B a candidate cell without touching production:
poetry run python -m sportstradamus.training.scorecard --baseline data/test_sets/NBA_AST.csv --candidate /tmp/cand.csv
```
