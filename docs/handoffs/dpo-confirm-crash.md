# Fix brief: a forced-DPO confirm dies silently ~6 minutes in

NFL carries lost **both** confirm nominees to `REVERTED … failed retrain error` — a non-zero
`meditate` exit, not a gate verdict. No `model_stats` row is written, so the cell produces no
evidence at all and the nominee ledger stays empty for it. Diagnosis below is deliberately split
into what is established and what is ruled out; the cause is **not** identified.

## Established

**It is a silent death, not an exception.** The last line in
`research/logs/confirm/NFL_carries.log` is the dist-selection warning:

```
WARNING …pipeline: NFL carries: training forced dist=DPO
  (data-driven pick is SkewNormal: global_mean=7.47, zero_rate=0.06)
```

`_run_meditate_with_lock_retry` runs the subprocess with
`stderr=subprocess.STDOUT` ([sweep.py:386](../../src/sportstradamus/training/model_strategy/sweep.py#L386)),
so a Python traceback *would* be in that log. Its absence means the process died without raising —
a signal or a native abort.

**It dies early, inside the first minutes of the fit.** The cell ran 16:59:42→17:25:04. Its Optuna
board-sweep journal (`research/optuna/NFL_carries.8758d634.log`) stops at 17:12, and the confirm log's
final write is 17:24:39 — so the deterministic board sweep took ~13 min and **both** nominees fit into
the remaining ~12.6 min, about **6 min each**. A cold full-HPO search is capped at 60 min / 300
trials, so neither nominee got near the end of its search.

**Both nominees were forced DPO** on a cell whose data-driven pick is SkewNormal:

```
1. dist=DPO, count_dispersion_objective=pit_ks, blending=crps, posthoc=prob_recal_isotonic
2. dist=DPO, count_dispersion_objective=crps,   blending=crps, posthoc=prob_recal_platt
```

**Forced DPO is not sufficient to trigger it.** Two other cells ran the same forced
`SkewNormal`→`DPO` override through the same production confirm path without dying: MLB pitcher
fantasy points underdog produced real gate rows for both its DPO nominees, and NBA DREB passed
37 minutes on nominee 1. Matrix width is the one axis that separates them —

| cell | matrix | forced DPO | outcome |
|---|---|---|---|
| NFL carries | 6266 × **483** | yes | died ~6 min, both nominees |
| NBA DREB | 13915 × 317 | yes | survived past the window |
| MLB pitcher fp | 2436 × 141 | yes | confirmed, real gate rows |

— but three points is a correlation, not a cause, and DREB's 317 columns sit roughly midway, which is
weak support for a width threshold. Row count argues against size altogether: DREB has more than
twice NFL carries' rows and is fine.

## Ruled out, with the evidence

| Hypothesis | Why it is out |
|---|---|
| Out of memory | Matrix is 6266 × 483 (26 MB); 31 GB available; no OOM lines |
| Confirm timeout | `_CONFIRM_TIMEOUT_S` is 4 h; actual ~6 min |
| Warm-start Optuna enqueue (the `lambda=0` crash) | No pickle for the cell, so cold start and no `enqueue_trial` |
| DPO itself | `meditate --deterministic --dist DPO` on the same cell and matrix exits **0** |
| DPO under full HPO, in isolation | A `--frozen-matrix-dir` + `--artifact-output` run reached 98/300 trials over 29 min without dying — it survived five times the failure window |
| SHAP / `compute_market_importance` | Runs only *after* the fit; at ~6 min nothing reached it. (Plausible on path-difference grounds, refuted on timing — do not re-chase without new evidence.) |

## Where that leaves it

Two independent contrasts bracket it, and they point different ways:

*Same cell, different path.* The isolated probe survived on NFL carries where production died. It
differs only in passing `matrix_input` and `artifact_output`, which set `isolated_run = True`
([pipeline.py:4242](../../src/sportstradamus/training/pipeline.py#L4242)) and suppress the
matrix/comps persist, `report()`, and SHAP. Two of those three are post-fit, so on this reading the
suspect is the matrix and comps persist path — the one thing `isolated_run` skips *before* the fit.

*Same path, different cell.* DREB and MLB pitcher fp both survive the production path under forced
DPO, so the production path is not broken in general and something about NFL carries participates.

Neither contrast is decisive alone. Deciding between them is cheap and should come first: re-run the
isolated probe once more to confirm it reliably survives (it was run once), then run the production
recipe below with `PYTHONFAULTHANDLER=1` and read the fault address.

## Affected cells

**Lost:** NFL carries (both nominees, no gate row).

**At risk — DPO among the top-2 board corners, and still queued:** WNBA BLST, WNBA STL, WNBA DREB,
NFL completions. NBA DREB was on this list and has since cleared the window, so the exposure is
4 cells rather than 5. Expect any `failed retrain error` among them to be this bug, not the cell;
check the log's last line for the forced-dist warning before concluding anything about the corner.

## Reproducing it

Run the **production** path — the isolated one does not reproduce. It calls `report()`, so stop the
sweep driver first or it will race the run's `model_stats.parquet`.

**Configure through stat_meta, not flags.** Every control the DPO corner varies is in that spec's
`persist` map (`dist`, `count_dispersion_objective`, `blending`, `posthoc`), so
`strategy_full_hpo_cli_args` emits nothing and the confirm's real command is the bare one below —
which is what `pgrep` shows for the live cells. Passing the same values as CLI flags exercises a
different resolution path and is not the same test.

```jsonc
// src/sportstradamus/data/config/stat_meta.json → NFL.carries
{"dist": "DPO", "shipped": "withheld", "target_normalization": "none",
 "posthoc": "prob_recal_isotonic", "blending": "crps", "count_dispersion_objective": "pit_ks"}
```

```bash
PYTHONFAULTHANDLER=1 poetry run meditate --league NFL --market carries --force
echo "exit code: $?"
```

Restore the cell's original entry afterwards — it is
`{"dist": "SkewNormal", "shipped": "withheld", "target_normalization": "centered_additive_mean10",
"posthoc": "none"}`.

`PYTHONFAULTHANDLER=1` is the point of the recipe: it costs nothing and dumps a Python-level
traceback on `SIGSEGV` / `SIGABRT`, which is the one piece of evidence this investigation never had.
The exit code alone already narrows the families — `-11` segfault, `-9` external kill, `-6` native
abort.
