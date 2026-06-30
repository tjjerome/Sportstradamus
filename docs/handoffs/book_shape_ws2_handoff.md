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
| `ws2-book-shape` (off `devel`@391c5ee) | WS2 keystone `0c4aefd` + `f85069e` doc polish; foundation `a1acda6` (fit + eval); read-path contract `af25e90` (get_ev sigma, fused_loc book shape); archive re-encode `fa385dd` (gated); model_prob book paths `e2967ad` (gated, cv-param) | **local only — wiring phase done; Unit B next** |

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
- **WS2 foundation** (`a1acda6`, additive, no behavior change). `fit_book_shape(league, market,
  results, lines)` in [training/calibration.py](../../src/sportstradamus/training/calibration.py) bins
  realized results by line (≥120 rows/bin), fits `var = a·μ^b` (sqrt(n)-weighted log-space, `b` free)
  + linear skew, returns the coeff dict or `None` on sparse. `book_skewnormal_shape(league, market,
  mean)` in [helpers/config.py](../../src/sportstradamus/helpers/config.py) evals the curve + keystone,
  with the strict no-op `(μ·cv, 0.0)` fallback for unfitted cells. `book_shape` surfaced in the
  `stat_meta` union. Not wired into `meditate`. Pinned by `tests/golden/test_book_shape.py`.
- **WS2 archive re-encode** (`fa385dd`, **gated → behavior-preserving until a cell is fitted**). Archive
  reads (`_book_rows` → `get_ev`/`get_team_market`/`get_moneyline`/`get_total`, and `to_pandas`) rebuild
  each book's mean from its stored `(under_prob, line)` quote at the per-cell fitted SkewNormal shape, via
  `book_skewnormal_shape(lg, mkt, line)` → `get_ev(sigma=, skew_alpha=)`. New `Archive._reencode_ev` +
  `Archive._book_shape_fitted`. **Decision (locked, deviation from plan):** the re-encode is **gated on a
  fitted `book_shape`**, not "always on" — every cell today (none fitted) returns the stored `ev`
  bit-identically, so the wiring lands green and the behavior turns on per-cell only when a retrain
  populates coeffs. (The plan wanted always-on for cv-drift immunity; that becomes a follow-on once cells
  are fitted, e.g. by dropping the gate.) `to_pandas` skips its whole-cell loop for unfitted cells.
  Pinned in `tests/golden/test_archive_shapefree_storage.py` (WS2 read section).
- **WS2 model_prob book paths** (`e2967ad`, **gated → behavior-preserving until a cell is fitted**).
  `_book_over_prob` (the `Market EV` feature + book-fallback pricing) and `_blend_with_book`'s
  SkewNormal `fused_loc` now read the per-cell shape via `book_skewnormal_shape(lg, mkt, mean)` — the
  book mean (not the line), since both legacy sites already used `mean·cv`. Both thread `league`/`market`;
  `_blend_with_book` swapped its `cell=""` log param for `league`/`market`. `book_skewnormal_shape`
  exported from the helpers package and gained an optional `cv` so the **unfitted fallback honors the
  caller-held CV** — keeps `(mean·cv, 0)` bit-identical to the legacy read AND decoupled from the config
  CV copy, so the train/serve parity invariant (book-fallback == weight-0 blend) holds at an arbitrary
  passed CV, not just the config value. No explicit gate needed: the unfitted no-op reproduces today's
  symmetric read, the fitted curve turns on per-cell with the retrain. Existing blend-guard / parity
  callers updated for the signature; parity test pins the symmetric contract with a synthetic never-fitted
  cell. Pinned by the `book_over_prob` tests in `tests/golden/test_book_shape.py`.
- **WS2 read-path contract** (`af25e90`, additive, behavior-preserving defaults). The per-cell shape is
  **injected by callers**, never pulled inside `distributions.py` (config imports distributions → the
  reverse cycles). `get_ev` gains a fixed `sigma`; `fused_loc` gains `book_sigma`/`book_skew_alpha` for
  the SkewNormal book side (blend now carries the book's skew). `sigma=None`/`book_sigma=None` reproduce
  today's `ev*cv`/symmetric shape bit-for-bit. **Decision (locked):** the book shape is evaluated once at
  the **quoted line** (`book_skewnormal_shape(lg, mkt, line)`), not re-evaluated at the solved mean — so
  the inversion bracket stays monotone (no new fail-loud hazard) and round-trips exactly at the line; the
  line ≈ the book's posterior mean (the 5-7% gap §10 measured), so shape-at-mean is only a 2nd-order
  refinement. This supersedes the plan's literal "`get_odds`/`_skewnormal_odds` use `book_skewnormal_shape`"
  (impossible — import cycle).

## Next (ordered — validate each before the next)

### 1. WS2 foundation + read-path contract — **DONE** (`a1acda6`, `af25e90`; see Done above)
`fit_book_shape` + `book_skewnormal_shape` built and pinned; `get_ev` (fixed `sigma`) + `fused_loc`
(`book_sigma`/`book_skew_alpha`) extended with behavior-preserving defaults. Neither is wired into a
caller yet — step 2 does that.

### 2. WS2 thread (behavior-changing — per-module subagents; **gate each on fitted `book_shape`**)
The contract + the archive re-encode exist. Remaining sites wire the callers to inject
`book_skewnormal_shape(lg, mkt, line)` (shape at the quoted line). Only **SkewNormal** cells change. Follow
2b's pattern: **gate each site on a fitted `book_shape`** so it lands behavior-preserving (every cell
unfitted today) and turns on per-cell with the retrain.
- **archive re-encode — DONE** (`fa385dd`). `_book_rows` (→ `get_ev`/team readers) + `to_pandas` re-encode
  via `Archive._reencode_ev`, gated by `Archive._book_shape_fitted`. `_weighted_book_ev` unchanged — it
  averages the already-re-encoded per-book means (the CDF-fit consensus is the deferred step 3). Other ev
  read sites (`get_ev_history`/`get_movement`/`get_closing_line`) are **not yet** re-encoded — fine while
  unfitted; revisit for CLV consistency once a cell is fitted.
- **`prediction/model_prob.py` — DONE** (`e2967ad`). `_book_over_prob` + `_blend_with_book` read the
  per-cell shape via `book_skewnormal_shape(lg, mkt, mean)` (at the book **mean**, not the line — both
  legacy sites used `mean·cv`, so the unfitted no-op is bit-identical without an explicit gate). The
  optional `cv` on the helper carries the caller's per-cell CV through the unfitted fallback. See the
  WS2 model_prob book-paths entry under **Done**.
- **`stats/base.py:2009` training feature (2d) — still to do; lands with Unit B.** Replace the mean→prob
  round-trip `1 - get_odds(line, ev, dist, cv)` (where `ev = archive.get_ev`, the re-encoded consensus
  **mean**) with the prob-space `1 - composite_under_prob` read **directly** from the stored quotes —
  shape-invariant + Jensen-free (~10-30bp). **Discovery:** the archive has no composite-under-prob
  accessor yet (`get_ev`/`_weighted_book_ev` average per-book *means*); 2d needs a new prob-space
  consensus reader (weighted mean of per-book `under_prob`, mirroring `_weighted_book_ev`), which is the
  shallow end of the deferred step-3 consensus (CDF-fit lift stays deferred — ladder empty). 2d is
  **mandatory + kept regardless of the settling-experiment fork** (it is shape-invariant; only the
  blend-shape ship is the research bet). Only takes effect on the next `meditate`, so it validates with
  the Unit B retrain.

`fused_loc`'s `book_sigma`/`book_skew_alpha` contract stays (`af25e90`, harmless behavior-preserving
default), but per the SPLIT verdict the **blend collapses-to-cv** — `_blend_with_book` should stop passing
the shaped book (Unit B reverts the 2c pass-through). The API is just unused by the blend, not removed.

### 2.5 Unit B — turn it on (`meditate` wiring + retrain), **re-scoped by the SPLIT verdict**
Code wiring DONE on `ws2-book-shape` (local checkpoints, gated, not pushed). Retrain is the one remaining step
and is **user-gated**. Until a cell is retrained `book_shape` stays unfitted and every path is bit-identical
to today.
- **Step A — served blend collapse-to-cv (`2fd92a2`).** `_blend_with_book` reverted the 2c
  `book_sigma`/`book_skew_alpha` pass-through; the SkewNormal blend keeps the symmetric book leg. The shaped
  book is NOT the served lever ([[ws2-settling-split-verdict]]). Invariant pinned: the blend is independent of
  a fitted `book_shape`.
- **Step B — reverted the 2b `get_ev` re-encode (`ef9ea72`), NOT a mean-split.** Q1 validated the book-only
  fix as the shaped SHAPE at the **symmetric** mean, and `_book_over_prob` (2c) already applies that shape at
  `get_ev`'s mean — so both paths want `get_ev` symmetric; the shaped-mean re-encode was unnecessary. Reverted
  `fa385dd` (`_reencode_ev` / `_book_shape_fitted` + the `_book_rows`/`to_pandas` re-encode); kept the WS1
  `(under_prob,line)` storage. The shaped treatment lives entirely in `_book_over_prob`.
- **Step C — 2d shape-free training feature (`16742bb`).** `Archive.get_composite_under_prob` returns the
  book-weighted consensus de-vigged `under_prob` (a `value_col` param on `_book_rows`, reusing
  `_weighted_book_ev`); `stats/base._resolve_player_market_odds` reads `1 - get_composite_under_prob`, falling
  back to the symmetric `get_odds(line, ev)` only when no book quoted. Shape-invariant + Jensen-free.
- **Step D — meditate fit + persist (`2079cdf`).** `train_market` fits `calibration.fit_book_shape` for each
  SkewNormal cell (matrix `Result` binned by `Line`) and persists via `save_book_shape_config` into
  `stat_calibration.json`; config.py already merges `book_shape` into the in-memory `stat_meta` cell, so
  `book_skewnormal_shape` reads it on the book-only leg. Deterministic runs skip it.
- **Remaining (user-gated): retrain** affected SkewNormal cells, then validate per step 4 (gates g4/g5/PIT-KS +
  Cardoso 84.5% → ~79% on the fallback path).

### 3. WS2 consensus
`helpers/archive.py` `_weighted_book_ev` → WLS-on-CDF one-point fit with inverse-binomial-variance weights
`n/(p(1−p))` + isotonic monotonicity (Breeden-Litzenberger). **Ladder lift DEFERRED** — the `ladder` table
is empty (0 rows vs 19.2M odds), so WS2 consensus ≈ today's + the shape shift.

### 4. WS2 validate (the load-bearing gate)
- Round-trip invertibility across the `(line, under)` domain + real archived samples (fail loud, never
  write a garbage EV).
- **Open-Q1 settling experiment — DONE; verdict = SPLIT.** Brief
  `/tmp/researcher_ws2_settling.md`; durable in [[ws2-settling-split-verdict]]. The shaped book wins the
  **book-only leg** (book-PIT-KS 5/7, large on skewed low-mean counts; the Cardoso fix — GO) but **loses the
  served blend OOS 6/7** two ways (co-fit drives `w_shp`→floor and still loses; hurts at the served weight) —
  so **collapse-to-cv in the blend.** Confirms the `w≈0.90` decoupling ([[book_distribution_audit_nogo]],
  [[pooling_half_blp_nogo]]) for WS2's conditional fit: the model's per-row SN already carries the shape.
- Gate revalidation (g4/g5/PIT-KS) via `training/scorecard.py` A/B on the SkewNormal cohort; retrain affected
  cells; live spot-check **Cardoso AST u2.5 84.5% → ~79%** (the book-only fallback path), Kelly shrinks, no
  new "implausible book EV" warns.
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
