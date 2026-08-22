# Centered SkewNormal — Hessian Overflow

> Status: ACTIVE — stage 0 (characterize; nothing diagnosed as damage yet)

## 1. Mission & money logic

The centered-parametrization SkewNormal head produces non-finite hessians on a
handful of training rows. LightGBM casts hessians to `float32` before boosting,
those rows arrive as `inf`, and tree growth stops early on some fixtures. Seven
production cells train with `sn_param: "centered"` — six of them shipped to
`devel` — so whatever this costs in fit quality, it is already priced into live
recommendations for those markets.

Two concrete things are blocked on understanding it:

* **`numpy` is pinned below 2.0.** numpy 2 perturbs the arithmetic enough to
  move where boosting stops, which is why the pin outlived the rest of the
  2026-08 dependency sweep (`pyproject.toml`, the `numpy` block).
* **The boosting-round regression guard is loose.** `tests/test_skew_normal_centered.py`
  asserts `current_iteration() >= 20` against a catastrophe that used to strike
  at ~4 rounds. It was `>= 50` until this fixture started landing at 41.

The honest open question comes first, before any fix: **stage 0 must decide
whether the early stop is damage or benign convergence.** Section 3's evidence
points both ways and the lane should not skip past it.

## 2. Read first (in order)

1. [`../../src/sportstradamus/skew_normal_centered.py`](../../src/sportstradamus/skew_normal_centered.py)
   — 171 lines, the whole subject. `_centered_to_direct` (the Azzalini CP→DP
   inversion) and `_GAMMA1_MAX` / `_ALPHA_RADICAND_FLOOR` are where the
   arithmetic lives.
2. [`../../tests/test_skew_normal_centered.py`](../../tests/test_skew_normal_centered.py)
   `test_wrapper_contract_and_predict_reemits_direct_params` — the reproducing
   fixture and the loosened guard.
3. `lightgbmlss/distributions/distribution_utils.py`
   `compute_gradients_and_hessians` (~line 433) and `stabilize_derivative`
   (~line 511) — the `nan_to_num` + L2 rescale the hessians pass through.
4. `lightgbm/basic.py` `Booster.__boost` (~line 4200) → `_list_to_1d_numpy`
   (~365) → `_cast_numpy_array_to_dtype` (344) — the `float32` cast that
   emits `RuntimeWarning: overflow encountered in cast`.
5. [`model_improvement_track.md`](model_improvement_track.md) §8.2 — the
   research-first policy that gates every edit in this lane.
6. `/tmp/researcher_continuous_family.md` §1a, if it still exists — the brief
   that motivated the centered parametrization. Regenerate via
   `research-analyst` if gone.

## 3. Verify before you trust

Everything below was measured on `devel` at `cee582c`, numpy 1.26.4,
torch 2.9.1+cpu, lightgbm 4.6.0. Re-measure before building on it.

```bash
git fetch origin && git log --oneline origin/devel -3
grep -n "numpy = " pyproject.toml                     # still <2.0?
grep -n "current_iteration() >=" tests/test_skew_normal_centered.py   # guard still 20?
python -c "import json;m=json.load(open('src/sportstradamus/data/config/stat_meta.json'));print([(l,k,v.get('shipped')) for l,mm in m.items() for k,v in mm.items() if isinstance(v,dict) and v.get('sn_param')=='centered'])"
poetry run pytest tests/test_skew_normal_centered.py -q
```

**Where it breaks.** On the fixture's `default_rng(5)` draw, exactly one row of
400 (index 394) produces a non-finite raw hessian on **all three heads**, and it
does so at **boosting round 2** — not at round 41. Round 41 is only where
LightGBM stops adding trees; the arithmetic has been broken for 39 rounds by
then. Anyone who reads the `iters=41` number as the failure point will look in
the wrong place.

At that row the gamma1 head has railed: raw score `atanh_g1 = -64.74`, which
`tanh` saturates to `gamma1 = -0.99` exactly (the `_GAMMA1_MAX` bound), mapping
to `alpha = -27.85`. The log-density there is *fine* — `alpha*z = -9.42`,
`log_ndtr = -47.5859` in both float32 and float64, agreeing to nine digits. **The
overflow is in the second derivative through the CP→DP map, not in the density.**
`radicand = 1 - (pi/2 - 1) r^2` is 3.53e-3 at the bound (giving exactly the
observed `alpha = 27.85`), and `alpha` differentiates it as `radicand^(-3/2)`
once and `radicand^(-5/2)` twice — a 1.35e6 amplification before anything else
multiplies in. The bound is doing real work here: at the attainable maximum
`gamma1 = 0.99527` the radicand is 3.1e-8 and that factor is 5.9e18.

**Why the stabilizer does not catch it.** `stabilize_derivative`'s `nan_to_num`
replaces `inf` with the float64 max (1.798e308, not zero), and the L2 divisor is
clamped to at most 10000. So the offending entry leaves stabilization at
**1.798e304** while its healthy neighbours sit at ~1e1–1e3. The clamp is what
turns a survivable outlier into a poisoned one; it is worth treating as a
suspect in its own right, not just as a passthrough.

**Measured mitigations** (fixture only, single seed — leads, not results):

| Change | iters | first bad round | mean recovery err |
|---|---|---|---|
| baseline | 41 | 2 | 0.0035 |
| build the torch dist in float64 | 100 | 2 | 0.0035 |
| `_GAMMA1_MAX` 0.99 → 0.95 | 100 | 89 | 0.4020 |
| `_GAMMA1_MAX` 0.99 → 0.90 | 100 | none | 0.4159 |
| `_ALPHA_RADICAND_FLOOR` 1e-6 → 1e-2 | 100 | 48 | 0.2046 |

Read that table carefully, because it is more ambiguous than it looks:

* The float64 build reaches 100 rounds **with an identical mean-recovery error**
  and still goes non-finite at round 2. Identical error across 59 extra rounds
  means those rounds were no-ops. That is the strongest single piece of evidence
  that the early stop is convergence, not damage — and the reason stage 0 exists.
* Both `_GAMMA1_MAX` reductions make the error ~100x worse, but the fixture's
  start value is `atanh(g1_seed / _GAMMA1_MAX)`, so changing the bound also moves
  the starting point. **That comparison is confounded and cannot be used as-is.**

**Seed sensitivity** (same fixture, `default_rng(1..8)`): 100, 100, 100, 84,
**41**, 100, 74, 100. The fixture's seed 5 is near-worst-case, which is why the
`>= 50` guard broke rather than the code.

### Volatile product assumptions

- Hessian and cast line numbers move with `lightgbm` / `lightgbmlss` upgrades;
  re-grep rather than trusting §2's numbers.
- The seven `sn_param: "centered"` cells drift as cells retrain and ship — the
  §3 one-liner is the live list, not this brief.

## 4. Locked decisions

- 2026-08-22 — `numpy` stays `<2.0` until this is understood. The bump is not a
  security fix and the suite's tolerance for the wobble is not the same as the
  wobble being safe (`pyproject.toml` numpy comment; CHANGELOG `[Unreleased]`).
- 2026-08-22 — The `>= 20` guard is a deliberate floor over the ~4-round
  catastrophe, not a fit-quality bar. The finiteness and mean-recovery asserts
  in the same test carry model quality. Do not re-tighten it before stage 2.
- `skew_normal_centered.py` is research-gated (`.claude/research_gated.txt`).
  Every code change in this lane needs a `research-analyst` brief cited first.

## 5. Module footprint & canonical paths

`src/sportstradamus/skew_normal_centered.py` (the subject),
`src/sportstradamus/skew_normal.py` (base density — also research-gated),
`tests/test_skew_normal_centered.py`, `pyproject.toml` (the numpy pin).
`helpers/distributions.py` `_skewnormal_start_values` owns the centered start
values and is where a bound change lands second. Read-only: the `sn_param`
routing in `training/pipeline.py` and `training/model_strategy/specs.py`.

## 6. Stage plan

0. **Damage or convergence?** Instrument the fixture to record per-round
   train NLL and tree counts, and re-run the float64 variant. If rounds 42-100
   add no trees and no NLL improvement, the stall is benign and this lane's
   scope collapses to the numpy pin plus a comment correcting the "stall"
   framing. If they do improve, it is damage and stages 1-3 follow. Acceptance:
   a one-paragraph verdict with the NLL trace in the ledger. Half a session.
   *This gate decides whether the rest of the lane is worth running.*
1. **`research-analyst` brief** (mandatory before any code edit, §8.2). Ask
   specifically: is the right fix a reparametrization that avoids
   `radicand^(-5/2)` near the bound, a tighter `_GAMMA1_MAX` with the start
   value decoupled from it, float64 evaluation, or a hessian clip? Include the
   §3 table and its confound. Acceptance: `/tmp/researcher_*.md` exists and
   names a preferred option with a reason.
2. **Implement + re-tighten the guard.** One change, cited to the brief. The
   fixture must reach 100 rounds across `default_rng(1..8)` with mean-recovery
   error no worse than 0.0035, and the guard goes back up to a real bar.
   Acceptance: seed sweep in the ledger; `>= 20` replaced.
3. **Lift the numpy pin.** Only after stage 2. Widen to `<3.0.0`, relock, run
   all three gates, then a `meditate` smoke pass on one centered cell on the dev
   box — green unit tests are necessary but not sufficient for a promotion
   change under the whole torch/pyro/lightgbmlss stack. Acceptance: gates green
   plus the smoke result; Dependabot #92 closes.

Stages are sequential; 0 can retire 1-2 outright.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > home-of-record
  doc > this brief > roadmap v3.
- Any `.py` edit here trips the research gate. Cite the brief, or write the
  one-line justification to `.claude/.state/research_waiver` and say why in the
  ledger.
- Do not "fix" the overflow by widening the guard further or by silencing the
  `RuntimeWarning`. Both hide the only signal this lane has.
- The seven centered cells are live on `devel`. A parametrization change means
  those cells retrain before their numbers mean anything.

## 8. Escalation & stop conditions

**Stop and ask the owner:** any change to `_GAMMA1_MAX` or the CP→DP map that
would require retraining shipped cells; lifting the numpy pin if stage 2 did not
land; a stage-0 verdict of "benign" (the owner decides whether the pin still
comes off).

**Park and pivot:** if stage 0 says benign and the owner leaves the pin, close
the lane with a ledger line — do not carry stages 1-3 forward as busywork.

**Dispatch:** `research-analyst` before stage 2 (mandatory);
`refactoring-specialist` on any touched `.py`; `devel-ship-curator` for the PR.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session (if any).
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- One ledger line appended below; status line updated on stage boundaries.
- Never push `devel` directly — the curator carves ship PRs.

## 10. Ledger (append-only, newest first, cap ~15)

- 2026-08-22 · stage 0 opened · lane split out of the dependabot sweep; first-failure forensics put the non-finite hessian at round 2 (row 394, gamma1 railed to the bound), not at the round-41 stall; float64 variant reaches 100 rounds with identical error, so "damage vs convergence" is unresolved · next: stage 0 NLL trace
