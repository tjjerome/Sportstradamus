# The confirm walk's full-HPO cross-validation aborts inside LightGBM

NFL carries lost **both** confirm nominees to `REVERTED … failed retrain error` — a non-zero
`meditate` exit, not a gate verdict, with no `model_stats` row and an empty nominee ledger. The
cause is an upstream LightGBM memory-safety defect, reproduced and worked around; this brief is the
record of what it is, what it is not, and what to check if the symptom returns.

## What kills it

A glibc heap-corruption `abort()` — SIGABRT, exit `134` (`-6` as a return code). The signature was
in two files from the start:

* `research/logs/confirm/NFL_carries.log`, last line, glued to the tail of a multi-kilobyte tqdm
  progress line: `0%| | 0/300 [00:00<?, ?it/s]malloc_consolidate(): invalid chunk size`
* `research/overnight/NFL_carries.confirm.log`, both nominees: `malloc(): invalid size (unsorted)`
  and `malloc_consolidate(): invalid chunk size`

Two different glibc detection sites across two nominees is the tell: real heap corruption moves its
trip point with allocation history. A third and fourth message (`unsorted double linked list
corrupted`, `free(): invalid next size`) appeared under reproduction, and LightGBM 4.7.0 produced a
plain SIGSEGV.

The abort lands in the **first Optuna trial** — the progress bar reads `0%| | 0/300 [00:00` — inside
`LGBM_BoosterCreate`:

```
File ".../lightgbm/basic.py", line 3661 in __init__        # LGBM_BoosterCreate
File ".../lightgbm/engine.py", line 572 in _make_n_folds   # Booster(tparam, train_set)
File ".../lightgbm/engine.py", line 777 in cv
File ".../lightgbmlss/model.py", line 237 in cv
File "src/sportstradamus/training/hyperparams.py", line 143 in objective
```

The ~6 minutes each nominee burned is `Stats.update()` — a 1693-item tqdm that takes `04:59` — plus
the gamelog load. The fit itself got well under a minute.

## Root cause

`lgb.cv` slices its folds with `Dataset.subset`, and **building a Booster over a subset Dataset
corrupts the heap** on this frame. Reproducible with `lightgbm`, `numpy` and `pandas` alone — no
torch, no LightGBMLSS, a trivial numpy objective — so it is LightGBM's own defect, present in both
4.6.0 and 4.7.0.

| variant | aborts |
|---|---|
| `lgb.cv`, `nfold=4` — the production configuration | 5/5 |
| `lgb.cv`, `nfold=3` / `nfold=5` | 0/10 each |
| `Dataset.subset` construction alone, no Booster | 0/5 |
| a single `lgb.train` | 0/3 |
| per-fold `lgb.train` on materialised slices | 0/15 |
| synthetic same-shape frame (4386 × 435), `nfold=4` | 0/3 |
| LightGBM 4.7.0, isolated venv, same frame | 5/5 |

It is data-dependent (a synthetic frame of identical shape survives) and probabilistic: `nfold=4`
trips it near-deterministically on this cell, while `num_threads=1` still aborted 1/3. Treat any
surviving configuration as luck, not safety.

## Ruled out, with the evidence

| Hypothesis | Why it is out |
|---|---|
| The DPO family | The same abort class hit NFL rushing-yards under SkewNormal/crps in July — `data/research/stage5-recovery-20260725-v1/RUSHING_NATIVE_DIAGNOSIS.md`. The pure-LightGBM reproduction has no distribution at all |
| Two OpenMP runtimes in the process | Real (torch's vendored `libgomp.so.1` satisfies LightGBM's, sklearn ships a third under a mangled SONAME) but not causal: a process with neither torch nor sklearn aborts identically, and import order changes nothing |
| The pre-fit matrix/comps persist | The pure reproduction reads a parquet and aborts; nothing writes |
| `hist_pool_size`, `max_bin`, `monotone_constraints`, `num_class`, categorical dtypes, `init_score`, `free_raw_data` | Each toggled independently; all still abort |
| Out of memory | 39 GB box, matrix 6266 × 483, no OOM lines, and the message is corruption rather than exhaustion |
| Confirm timeout | `_CONFIRM_TIMEOUT_S` is 4 h; actual ~6 min |
| SHAP / `compute_market_importance` | Runs after the fit; the abort precedes the first boosting round |

## Why the board sweep never saw it

`--deterministic` takes the `DETERMINISTIC_FIXED_PARAMS` branch in
`training/pipeline.py:_step_select_hyperparams` and never calls `run_hyper_opt`, so it never calls
`model.cv`. Only the confirm's full-HPO path is exposed. That asymmetry is why the same cell passed
77 board trials and then died on both confirm nominees, and why "three other cells ran the same
forced override" was weaker evidence than it looked.

## The fix

`lgb.cv` takes an `fpreproc` hook that is handed each fold's two Datasets before the Booster is
built, and LightGBMLSS's `cv` forwards it. `hyperparams._materialise_fold_datasets` uses that hook
to rebuild both folds as plain Datasets referencing the full frame, so `Dataset.subset` never
reaches a Booster while cv's own loop, early stopping and aggregation stay untouched. Referencing
the full Dataset preserves its bin mappers: on a cell where the upstream path survives, the two
produce **bit-identical** cv losses. `run_hyper_opt` sets `free_raw_data=False` on the training
Dataset because the hook reads the frame back off it.

Two changes make the next one legible rather than mysterious:

* `training/cli.py` enables `faulthandler`, so a native abort dumps a Python-level stack into the
  same log instead of ending mid-line.
* `model_strategy/sweep.py:_failure_reason` classifies a signal death separately from a non-zero
  exit, and the failed-run echo pulls the glibc message out of the progress line by pattern. The
  confirm report now reads `retrain native abort (SIGABRT)` rather than `retrain error`.

## What it cost, and what was recovered

Exposure is **not** "the DPO cells" — the pure-LightGBM reproduction has no distribution at all, and
deriving the list from `dist` is what produced the earlier, wrong exposure list. The signal is a
confirm nominee that died `retrain error`. Six cells did:

| cell | nominees lost | cause | where it landed |
|---|---|---|---|
| NFL carries | 2/2 | this bug — the glibc signature is in its logs | see below |
| NBA DREB | 3/3 | not recoverable from the 07-27 logs | re-walked 07-29, all six gates pass, `devel` |
| NFL passing tds | 4/4 | not recoverable from the 07-27 logs | re-walked 07-29, all six gates pass, `devel` |
| NBA FGA | 2/3 | not recoverable from the 07-27 logs | re-walked 07-30, all six gates pass, `devel` |
| NFL interceptions | 1 | `LightGBMError: monotone_constraints.size()` mismatch, with a traceback | re-walked 07-29, genuinely fails g4/g6 |
| MLB runs allowed | 3 | `ValueError: … quote authenticity is missing`, with a traceback | re-walked 07-29, shipped `devel` |

Only NFL carries carries the glibc signature. The two tracebacked failures are unrelated bugs. The
three middle rows lost every nominee to a silent `retrain error` whose cause the 07-27 driver did
not record, so they cannot be attributed either way — but all three later completed real gate
verdicts, so nothing is owed on them.

## If the symptom returns

Read `research/overnight/<cell>.confirm.log` first — the driver echoes the failing run's reason and
tail there. A `native abort (SIGABRT)` line means a signal, so exception-hunting is wasted; the
faulthandler stack names the frame.

To reproduce a walk run by hand, configure the cell through `stat_meta.json` rather than flags —
every control the DPO corner varies is in its spec's `persist` map, so `strategy_full_hpo_cli_args`
emits nothing and the confirm's real command is the bare one below. **Set `shipped` to `devel`**, as
`_confirm_one` does before it retrains; a `withheld` cell prunes its pickle and skips training.

```jsonc
// src/sportstradamus/data/config/stat_meta.json → NFL.carries
{"dist": "DPO", "shipped": "devel", "target_normalization": "none",
 "posthoc": "prob_recal_isotonic", "blending": "crps", "count_dispersion_objective": "pit_ks"}
```

```bash
poetry run meditate --league NFL --market carries --force
echo "exit code: $?"
```

Stop the sweep driver first — the production path calls `report()` and will race
`model_stats.parquet`. Restore the cell afterwards; its own entry is
`{"dist": "SkewNormal", "shipped": "withheld", "target_normalization": "centered_additive_mean10",
"posthoc": "none"}`.
