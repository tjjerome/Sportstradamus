# Model Lifecycle

How a model goes from "I want to improve this market" to fully live: the release
surfaces, the strategy sweep, the confirming training run, the ship-gate scorecard,
and graduation. This is the canonical home for that workflow and for `meditate`'s
research/sandbox flags.

Each model covers one **cell** — a single league-and-market pair, such as *NBA points*. Before a
cell's model is allowed to serve real recommendations it has to clear the **ship gates**: a fixed
battery of automatic quality checks (Is it as accurate as the sportsbook? Are its predictions
unbiased and well-calibrated?). A cell's release status is the `shipped` field in `stat_meta.json`,
and it moves through three stages:

- `withheld` — the model is not served (the board prices its props at the book consensus, and drops a prop no sportsbook quotes); still being worked on.
- `devel` — served on the tracking server and watched on real settled bets.
- `main` — fully live.

The path from "I want to improve a cell" to "it's live" is five steps.

## 1. Sweep the cell's strategy options

`sportstradamus ship sweep` trains a quick throwaway model per candidate recipe — the knobs that
move the cell's distribution family (target shape × training loss × blend loss × post-hoc
calibrator for SkewNormal; count mode × dispersion objective × blend loss × calibrator for the
count families) — and ranks them by how comfortably each would clear the ship gates, the margin it
calls **slack**.

The recipe space is far larger than a budget can enumerate, so each cell gets one Optuna study that
proposes recipes rather than walking a grid, capped at `--max-trials` (48 by default). The cell's
current recipe and any corner already proven under full HPO are always evaluated, so a search can
only improve on what you have.

```bash
sportstradamus ship sweep --league NBA --market FGM   # one cell
sportstradamus ship sweep                             # every withheld cell with cached data
sportstradamus ship sweep --league WNBA               # just one league's withheld cells
sportstradamus ship sweep --include-shipped           # also re-check already-shipped cells
sportstradamus ship sweep --resume                    # continue a crashed multi-hour run
sportstradamus ship sweep --dry-run --resume          # what it would do, without training
sportstradamus ship sweep --league WNBA -v            # one line per recipe (-q for less)
sportstradamus ship sweep --jobs 4                    # 4 cells at once (default: cores/2, max 8)
sportstradamus ship sweep --confirm --min-model-weight 0.3   # ship only recipes whose fitted blend weight lands >= 0.3
```

Naming a single `--league` **and** `--market` sweeps just that cell. Omit `--market` and it sweeps
every **withheld** cell in `stat_meta.json` — every family — that has a cached training matrix (a
"board" run), printing an up-front count of how many cells (and trainings) that is;
`--league` alone narrows the board to one league. A cell with no cached matrix is **skipped with a
yellow warning** rather than swept — the throwaway trainings reuse the cached matrix and never
rebuild one. The weekly `meditate` keeps every active cell's matrix warm (withheld cells included —
they skip only the fit), so this bites only a market that has never seen a weekly run. Add
`--include-shipped` to also rank already-shipped (devel/main) cells when hunting a better strategy for
a live cell; that path is judged by the supersession test, and `--confirm` never auto-re-ships a live
cell.

A board run sweeps several cells at once — `--jobs`, defaulting to half the cores capped at 8. Each
throwaway training is a single-threaded subprocess that never opens the DuckDB archive, so cells
parallelize cleanly; the recipes *within* a cell stay sequential, because the search proposes each
one from what the last one scored. Expect roughly a 5x wall-clock saving, floored by the slowest
single cell. `--jobs 1` restores one-at-a-time.

Each running cell gets a progress bar carrying its recipes done, the best slack so far, and a ceiling
on the time left, under a summary line counting cells running and queued. A recipe prints a line only
when it is a new best, ships, or fails, and each finished cell prints its verdict and the board's
remaining time. `-v` restores one line per recipe (cache hits included) and `-q` drops to the bars
and the per-cell verdicts. Redirected output — the overnight driver — gets those same lines plus a
periodic heartbeat naming the recipe currently training, and no bars. The short ranked table per
cell, marked `SHIP` (green) or `KILL` (red), prints as the cell finishes when sweeping one at a time
and at the end of the run in board order otherwise; the full ranked results, every recipe and every
column, are saved to `data/research/strategy_research_board.csv`. **Nothing ships from this step** —
the throwaway models only *rank* the options so you know which one to train for real.

A recipe that runs far past what the rest of its cell needed is killed and recorded as a failure, so
one wedged training cannot hold up a whole cell; the ceiling is derived from that cell's own recorded
timings, and a killed recipe is retried on `--resume` rather than cached. Once a family has had a few
recipes and none of them is anywhere near the cell's best, the search stops proposing it and spends
the rest of the budget on the families still in contention.

`--dry-run` resolves the scope without training anything: per cell, its families, its training
ceiling, what `--resume` would reuse, and — once a board has run once and recorded per-recipe
timings — how long it should take.

Each trial is scored **holdout-blind**: the rows the ship gate will use are dropped from the run
entirely, and the calibration head is fit out-of-fold across the rest. Ranking therefore cannot
adapt against the evidence the ship decision rests on. `--resume` reopens the cell's saved study and
reuses every prior row still valid for the current recipe registry and training matrix, so a crashed
run continues instead of retraining what it already scored.

Prefer to skip the manual walkthrough below? Add `--confirm`: for each **withheld** cell it nominates
its top-ranked recipes plus its current one, retrains each for real until one passes the official
gates, and keeps that one — steps 2–4 automated, with a prompt before it touches anything (`--yes` to
skip it). A cell whose board has no outright winner still gets confirmed: fixed-hyperparameter
ranking cannot recognize a recipe that only passes under a real hyperparameter search, so the gates
decide after the retrain rather than before it. Each retrain is roughly an hour, so it announces how
long this cell's own past confirms took and, on a terminal, ticks elapsed against that while it runs.
For an already-shipped cell (only present under `--include-shipped`), `--confirm` runs the
supersession test: it snapshots the incumbent, retrains the candidate in place, and scores S1/S2/S3
(candidate clears the six gates standalone, is paired-Brier sharper, and paired-Sharpe sharper). It
prints the comparison and swaps the live cell only when all three pass **and** you confirm the
promotion; a loss (or a declined prompt) restores the incumbent byte-identical and it keeps serving.
`--min-model-weight T` makes the fitted blend weight a ship criterion in both lanes; the three
clauses (nomination, ship-time revert, S1-only waiver) live in [ship_gate.md](ship_gate.md). The
board's `model_weight` and `n_authentic_validation` columns show each corner's fitted weight and
the sportsbook-priced (`authentic`) validation rows behind it; `pickem` rows (DFS-only cohorts)
count for neither.

## 2. Confirm the winner with a real training run

The board's top row names the winner in its axis columns. Each column the winner's family saves maps
to one field of that cell's entry in `src/sportstradamus/data/config/stat_meta.json` (the per-family
persist maps are in `training/model_strategy/specs.py`):

| Board column | What it is | `stat_meta.json` field |
|---|---|---|
| `dist` | the distribution family | `dist` |
| `normalization` | the target shape (SkewNormal) | `target_normalization`; a count-family winner saves `none` |
| `dist_training_loss` | the training loss (`crps` or `nll`; the count families always train `nll`) | `dist_training_loss` |
| `sn_param` | the SkewNormal head parametrization (`direct` or `centered`) | `sn_param` |
| `zinb_mode` | the ZINB architecture (`joint` or `hurdle`) | `zinb_mode` (ZINB only) |
| `count_dispersion_objective` | the count-branch dispersion objective | `count_dispersion_objective` (count families only) |
| `blending_loss_fn` | how the model and book are combined | `blending` (leaving it out means the `nll` default) |
| `posthoc` | the post-hoc calibrator | `posthoc` |

No base family saves the board's `hpo_selection` or `stabilization` column; only the two structural
recipes (two-part and affine) save `hpo_selection`. You don't have to map columns by hand either
way — for each shipping cell the sweep prints the exact `field=value` line to copy
(`→ stat_meta.json: …`).

So a SkewNormal board winner of `centered_additive_mean10` with a `crps` training loss, `direct`
heads and `crps` blending makes NBA FGM read:

```json
"FGM": { "dist": "SkewNormal", "shipped": "withheld",
         "target_normalization": "centered_additive_mean10", "dist_training_loss": "crps",
         "sn_param": "direct", "blending": "crps", "posthoc": "none" }
```

Set those fields *before* you train — `meditate` reads them:

```bash
sportstradamus meditate --league NBA --market FGM --bypass-withholding
```

A real training run (unlike the sweep) is the one whose result actually counts.

## 3. Read the ship-gate scorecard

`meditate` writes each cell's gate results into `model_stats.csv` (and its `.parquet` twin). To see
the verdict for one cell on its own:

```bash
sportstradamus ship scorecard --league NBA --market FGM
```

It prints a **SHIP SUMMARY** naming any gate the cell fails. A cell that passes every gate is ready
to ship. In plain terms the gates check:

- **Accuracy** — the model is at least as good as the sportsbook.
- **Star / bench bias** — predictions aren't systematically too high or too low for the best or the
  lowest-usage players.
- **Calibration** — the whole predicted distribution matches what actually happens.
- **Confidence** — the model is neither over- nor under-confident.
- **No shrinkage** — it doesn't quietly pull star players back toward the average.

The exact thresholds behind each gate are in [ship_gate.md](ship_gate.md).

## 4. Ship it to devel

Change the cell's `shipped` field from `"withheld"` to `"devel"` in `stat_meta.json`, then
sanity-check the config:

```bash
sportstradamus ship config --branch devel   # validates and summarizes; changes nothing
```

On `devel` the server serves the model and records how it does on real settled bets.

The weekly `meditate` retrains each served cell once with its persisted `stat_meta.json` config.
A retrain that fails the fresh gates only **warns** (a SHIP-GATE WARNINGS table at the end of the
run, echoed by `ship config --branch devel`) — the cell keeps serving until you demote it: flip
`shipped` to `"withheld"` (the next meditate prunes the pickle) or run
`sportstradamus ship config --branch devel --prune` to prune the pickle now.

## 5. Graduate to main

Once a cell has proven itself on live bets, `sportstradamus ship graduation` shows where every cell
stands and `sportstradamus ship config --branch main` promotes the ones that passed:

```bash
sportstradamus ship graduation              # status of every cell
sportstradamus ship config --branch main    # promote graduated cells to fully live
```

## Research flags

Beyond the production surface `meditate --help` lists (`--league`, `--market`,
`--force`, `--branch`, `--rebuild-correlations`, `--matrix-only`, `--log-level`),
`meditate` carries hidden research, debug, and sandbox flags that `--help` never
shows. This list is their canonical doc home; each flag's full text is its
`help=` string in `training/cli.py`.

**Ship-policy override:**

- `--bypass-withholding` — train every registered market whatever its `shipped` state (confirming a withheld cell, or the NFL volume cells whose pickles feed other cells' matrices).

**Search axes** — each overrides the per-cell `stat_meta.json` value for every
selected cell; the `auto` default honors what the cell has persisted:

- `--target-normalization` — target-shape transform for SkewNormal markets.
- `--posthoc` — post-hoc calibration method (light corrector or structural).
- `--zinb-mode` — ZINB architecture: legacy `joint` fit vs two-stage `hurdle`.
- `--dist-training-loss` — training loss (`nll`/`crps`) instead of the per-family default.
- `--dist` — distribution family (DPO / Mixture / NegBin / ZINB / SkewNormal) over the data-driven pick.
  Mixture never serves ([ARCHITECTURE](ARCHITECTURE.md#adding-a-new-market)).
- `--blending-loss-fn` — loss minimized when fitting the model↔book blend weight.
- `--stabilization` — per-parameter gradient damping (`MAD`/`L2`) on the scale head.
- `--hpo-selection` — Optuna trial-selection rule: lowest CV loss vs PIT-KS-aware `calibrated`.
  The confirm walk reruns a nominee whose blind run misses on Gate 4 alone once under
  `calibrated`, still blind, and persists the knob only if that retry is the cell's pick; the
  weekly `meditate` just trains with the persisted value.
- `--count-dispersion-objective` — objective for the count-branch dispersion fit (`crps` vs `pit_ks`).
- `--sn-param` — SkewNormal parametrization: `direct` vs `centered` boosting heads.

**Debug / eval:**

- `--deterministic` — pinned RNGs and fixed fast hyperparameters for bit-identical eval runs; never publish a model trained with it.
- `--holdout-blind` — drop the ship-gate rows and score on player-disjoint cross-fit folds (requires `--deterministic`, or `--artifact-output` for a full-HPO sandbox run).
- `--params-from` — refit one cell (`--league` plus one `--market`, not `--deterministic`) with a model pickle's LightGBM params, round count, `cv` and shape ceiling instead of searching, so a canonical run ships the booster a blind sandbox run ranked; a pickle of another distribution family is refused.

**Sandbox matrix and artifact plumbing:**

- `--full-rebuild` — rebuild matrices from cached raw inputs without touching the canonical matrix (requires `--matrix-only` and `--matrix-output`).
- `--matrix-output` — quarantine directory for full-rebuild parquet and manifest outputs.
- `--dependency-root` — root containing a versioned model-dependency namespace.
- `--frozen-matrix-dir` — train from lineage-validated matrices without rebuilding or rewriting them.
- `--artifact-output` — isolated output directory for models and test sets from frozen-matrix training.
- `--dependency-namespace` — dependency identity stamped onto isolated artifacts, or selected from `--dependency-root` during a full rebuild.

### Rebuilding a league's matrices from scratch

Appending to a cached matrix leaves any new feature column NaN on every
historical row, so a feature-set bump rebuilds the league's matrices. The cold
start (delete the cache, plain `meditate --force`) crashes for leagues whose
cells depend on volume models (NFL: attempts, carries, targets); the rebuild
resolves those from the versioned registry
`data/model_dependencies/<namespace>/{LEAGUE}_{market}.mdl` (default namespace
`volume-v1`), which the same run family bootstraps:

```bash
Q=/path/quarantine; A=/path/artifacts
# 1. volume cells first: they need no dependency, and they must not be mixed
#    with dependent cells in one invocation (the preflight would raise on the empty registry)
sportstradamus meditate --league NFL --market 'attempts,carries,targets' --full-rebuild --matrix-only --matrix-output $Q
python -m sportstradamus.training.matrix_audit $Q/NFL_*.parquet
# 2. stamped volume pickles from the frozen quarantine (its manifests satisfy --frozen-matrix-dir)
sportstradamus meditate --league NFL --market 'attempts,carries,targets' --frozen-matrix-dir $Q \
    --artifact-output $A --dependency-namespace volume-v1 --bypass-withholding
mkdir -p src/sportstradamus/data/model_dependencies/volume-v1 && cp $A/NFL_*.mdl src/sportstradamus/data/model_dependencies/volume-v1/
# 3. every other cell, in ONE process so the per-week snapshot caches amortize
sportstradamus meditate --league NFL --market '<the remaining markets>' --full-rebuild --matrix-only --matrix-output $Q \
    --dependency-namespace volume-v1 --dependency-root src/sportstradamus/data/model_dependencies
# 4. promote by hand (there is no promotion command), then the ordinary retrain
cp -p $Q/NFL_*.parquet src/sportstradamus/data/training_data/ && sportstradamus meditate --league NFL --force
```

Back up `training_data/`, `models/`, `test_sets/`, `training/model_stats.*`,
the SHAP CSVs, `stat_calibration.json` and `book_weights.json` first; every
ship verdict re-rolls because `strategy_matrix_hash` moves. A `--deterministic`
A/B of the old and new caches isolates the feature effect from HPO noise, but
deterministic mode resolves `auto` target normalization to `ratio_meanyr`, so
pass `--target-normalization` explicitly per group of cells.
