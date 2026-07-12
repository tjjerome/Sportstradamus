# In-repo research brief — Simulated-bettor ledger: replicate-ensemble sizing & architecture

> This is an in-repo research brief (read-only w.r.t. production). It answers the
> single open question in `docs/handoffs/sim-bettor-ledger.md` §10: whether to run
> **100 independent replicate ledgers per persona (300 total)** with mean/range
> reporting, versus a single realized path or a modified design. Everything else in
> §10 Policy v1 is locked and out of scope. Date: 2026-07-12.

---

## TL;DR

- **The right frame is ensemble-forecast verification, not "more Monte Carlo is
  safer."** Each replicate is a member of a finite ensemble of a *stochastic
  forecasting policy*; the mean-across-replicates is the ensemble-mean forecast.
  The meteorology literature has solved "how many members" exactly: proper scores
  converge to the infinite-ensemble limit as **`1 + 1/M`** (Leutbecher 2019,
  doi:10.1002/qj.3387). Going **50→100 members buys ~1% score improvement**; the
  marginal value of each replicate falls as `1/M²`.
- **For the persona MEAN, ~30–50 replicates is statistically ample; 100 is a
  defensible round number with a small safety margin, and anything past ~100 is
  waste.** The mean of a bounded daily-ROI statistic stabilizes at
  `SE = σ/√M` — the classic square-root law (Q1).
- **For the persona RANGE reported as 5th/95th percentiles, 100 is far too few.**
  Tail-quantile estimates need **2,000+** replicates to stabilize their endpoints
  (percentile-bootstrap rule of thumb); the noise per replicate is larger in the
  tail and scales as `1/√M`. **Reporting a 5/95 band off 100 replicates would be
  reporting a number noisier than the thing it's meant to characterize.**
- **Full brute-force 300-logged-and-settled ledgers is the wrong architecture** —
  not because the mean needs fewer draws, but because settlement volume (~3,000
  rows/day) is ~100× the single-path design for a mean you could get at ~30–50, and
  because the day-to-day *reported* series would still be a single noisy draw of the
  ensemble unless you also fix the reporting statistic. **Verified feasible** (nightly
  settlement cost is bounded by *distinct legs*, not entries — §Q3), but feasible ≠
  warranted under CLAUDE.md "avoid over-engineering."
- **Verdict: adopt a MODIFIED design.** Keep a real forward ensemble, but (a) **cut
  to M = 40 replicates per persona (120 total)**, (b) make **replicate 0 the
  canonical pre-registered path** (seeded exactly as §10 already specifies) and treat
  replicates 1…39 as the *variance envelope* around it, (c) **report the mean and a
  bootstrap-corrected ±1 SE / IQR band, NOT raw 5/95 percentiles**, and (d) offload
  the "full outcome distribution / percentile-tail" question to a **retrospective
  `profit_sim.py`-style characterization** run on the *accumulated real ledger*, which
  is what that sibling exists for. This gives D7 a clean, pre-registered skill estimate
  with an honest noise band at ~40% of the proposed settlement volume, and reserves
  the expensive tail-distribution claim for the tool built to make it.

---

## Key findings

### 1. The correct statistical object: a finite ensemble of a stochastic policy

The 300 replicates are not "300 backtests." Each persona's selection algorithm is a
*randomized decision rule* — a distribution over daily entry sets induced by the
Jaccard-decay weighted draw (§10). Running it `M` times and averaging the settled ROI
is exactly constructing an **`M`-member ensemble forecast** of that persona's daily
return and reading its **ensemble mean**. This is the object the numerical-weather-
prediction (NWP) community has studied for two decades, and it is a much tighter frame
than generic "Monte Carlo variance reduction" because it directly answers *how the
verification score of the ensemble mean depends on `M`*.

The load-bearing result (Leutbecher 2019, *QJRMS* doi:10.1002/qj.3387; ECMWF
Newsletter 157): for a reliable ensemble, proper scores (CRPS and its relatives)
converge to the infinite-member limit as

```
Score(M) ≈ Score(∞) · (1 + 1/M)          [the "1 + 1/M" law]
```

with the fractional penalty of a size-`M` ensemble versus a size-`M₁` one given by
`100·(M₁ − M)/[M·(M₁ + 1)] %`. The concrete numbers ECMWF publishes:

- **50 → 100 members: ~1% score improvement.**
- **50 → 20 members: ~3% penalty.**

The marginal value of the `M`-th member is `∝ 1/M²`. By `M ≈ 30–50` you have captured
essentially all of the recoverable ensemble-mean skill; past `M ≈ 100` you are paying
linearly in compute for sub-percent, un-observable gains. This is the single strongest
argument against `M = 100` *and* against anything larger: **the owner's instinct to
average over replicates is correct and valuable; the specific count of 100 is above the
knee of a curve whose knee is near 40.**

### 2. `profit_sim.py`'s `N_MONTE_CARLO_DEFAULT = 100` does NOT transfer — it is a display-latency budget, and I verified this in git

I pickaxed the constant's origin. It entered at commit `2c28fac` ("ship75 step0.5:
land gate code on devel") with the verbatim comment that survives today
(`strategies/profit_sim.py:36-38`):

```python
# 100 MC runs is the dashboard default — enough to compute a stable 10-90 band
# without making the Streamlit page sluggish. Gate callers override to 1.
N_MONTE_CARLO_DEFAULT: int = 100
```

Three facts from this:

1. **The rationale is Streamlit page latency, not a statistical derivation.** "Without
   making the page sluggish" is the binding constraint; "stable 10-90 band" is asserted,
   not derived. There is no CI-half-width target, no variance-stabilization argument, no
   convergence check anywhere in the file or its history.
2. **The gate callers — the load-bearing users — override to `N = 1`** (`simulate_kelly_all`
   sets `n_mc=1`, line 197). The number that actually feeds the S3 supersede gate and the
   Gate-2 Kelly yield is **one deterministic run**, precisely because those are *paired*
   comparisons where common random numbers make a single run well-defined (see Q2). The
   `100` only ever drives a *dashboard display band* over historical years.
3. **The backtest reason for a band ≠ the forward reason.** In `profit_sim.py` the MC
   samples *which subset of an oversupplied historical slate you'd have bet* when eligible
   bets exceed `max_bets_day` (`_pick_day_bets`, lines 300-319) — it is exploring
   selection variance over *years* of resolved data with hindsight and survivorship. The
   ledger's replicate draw explores selection variance *forward, one day at a time, with no
   hindsight*. These are different populations. The `100` was never sized for a live daily
   cadence and carries no authority here.

**Do not inherit `100` by analogy.** It is a coincidence that the two numbers match.

### 3. Sizing the MEAN: the square-root law puts the knee at ~30–50, consistent with (1)

Standard MC error (Ocean Optics Book, error-estimation; any Law & Kelton edition): the
SE of the mean across `M` replicates is `σ/√M`, and to hit a confidence-interval
half-width `ε` at level `1−α` you need

```
M = (z_{1-α/2} · σ / ε)²          [95%: M = (1.96 σ / ε)²]
```

The daily-ROI statistic per persona is **bounded and low-variance by construction**: at
most 5 entries/day, quarter-Kelly-sized, capped at 0.5% of a fixed $5,000 bankroll per
leg (`kelly.py:37` `MAX_FRACTION_OF_BANKROLL`), so a persona's *daily* staked exposure is
≤ ~2.5% of bankroll and daily ROI lives in a narrow band. The between-replicate σ of the
*windowed* (7-/30-day, matching `nightly.LIVE_METRICS_WINDOWS`) ROI is what the ensemble
mean averages down, and windowing already shrinks it. Plugging even a pessimistic
per-replicate daily-ROI σ of ~2% and asking for a ±0.3% 95% half-width on the *mean* gives
`M ≈ (1.96·2/0.3)² ≈ 170` — but that is the half-width on a *single day's* mean; over the
30-day graduation window the effective σ on the reported mean falls by ~√30, dropping the
required `M` for the same window-level precision to **the tens**. The ensemble-verification
result in (1) is the cleaner statement of the same thing and is not sensitive to my σ guess:
**the ensemble mean is ~99% of the way to its infinite-`M` value by M ≈ 50.**

Convergence should be *checked, not assumed* (SAS "monitoring convergence of Monte Carlo,"
Welford running-SE): a one-time offline study on the first ~2 weeks of live data — plot
ensemble-mean ROI vs `M` for `M ∈ {10,20,40,80,160}` and read where it flattens — would
empirically pin the knee for *these* personas and retire the guesswork. That study belongs
in stage 3 analytics, run once, not in the daily job.

### 4. Sizing the RANGE is a different, much harsher problem — and it breaks `M = 100`

The owner wants "mean **and range**." The range is where the design goes wrong, because
**tail quantiles converge far slower than the mean.** The percentile-bootstrap literature
(metricgate; garstats "the percentile bootstrap") is explicit:

- A **standard error / center statistic tolerates small `M`** (hundreds).
- A **5th/95th-percentile band needs ≥ 2,000** resamples; a 1st/99th or heavier tail wants
  **5,000+**, "to stabilise the endpoint estimates."
- Reason: the endpoint sits in the thin part of the distribution where noise per replicate
  is larger, and it still only falls as `1/√M`, so *halving the wobble on the band edge
  needs 4× the replicates.*

The tail-sensitivity point is independently corroborated in the ensemble literature:
Leutbecher notes tail-distribution scores are "considerably larger" in their sensitivity to
`M` than central scores. **So `M = 100` is simultaneously (a) more than enough for the mean
and (b) an order of magnitude too few for a 5/95 range.** A 5/95 band read off 100 draws
would jitter day to day *from the replicate RNG itself*, not from any change in the engine —
the exact "noisy realized path" pathology the owner is trying to escape, reintroduced one
level up. **This is the finding that most changes the recommendation:** if you report a
range at all from a live forward ensemble, it must be an SE/IQR-type *central-spread* band,
not extreme percentiles.

### 5. `ladders_stage0` Sobol'-QMC / CRN reasoning: partially transferable — CRN yes, QMC no

The Ladders brief (`docs/archive/researcher_ladders_stage0.md`, A1/A2) uses **N=8,192
scrambled Sobol' points with a per-game reused draw pool as Common Random Numbers across
candidates**, achieving ≤±2% EV error where naive IID would need ~5–10× more points. Two
distinct techniques there; they transfer asymmetrically:

- **Common Random Numbers (CRN): transfers, and is the key to Q2.** Ladders reuses *one*
  draw block across candidates so that candidate-to-candidate EV *differences* are far
  lower-variance than their *levels* (l'Ecuyer RQMC; Law & Kelton). The direct ledger analogue:
  the three personas should be driven by **synchronized randomness** so that cross-persona
  *differences* (Safe vs High-EV vs Kelly, which is the whole point of running three) are
  measured with the shared-slate noise cancelled. Concretely — draw all three personas' entry
  sets on a given (date, run_slot) from **one shared stream of uniforms** (e.g. one RNG seeded
  on `(date, run_slot)`, consumed in a fixed persona order), rather than three independent
  seeds. This is the same variance-reduction principle `profit_sim.py`'s gate path already
  exploits by running `N=1` on a *paired* universe. It costs nothing and makes the persona
  contrast — the lane's actual deliverable — sharper at any `M`.
- **Sobol'/QMC itself: does NOT transfer.** QMC's advantage is integrating a *smooth,
  moderate-dimension* integrand (the orthant probability over d≤5 latent Gaussians). The
  ledger "integrand" is the expected settled ROI of a **sequential, without-replacement,
  Jaccard-path-dependent discrete draw** over a *variable-length, high-dimensional*
  candidate set — non-smooth and combinatorial. Sobol' has no low-discrepancy advantage
  there and its equidistribution guarantees don't hold for sequential rejection-style
  sampling. **Plain PRNG replicates are correct here; QMC would be cargo-culting a technique
  out of its regime** (house rule: never claim an out-of-domain result transfers). Antithetic
  variates are similarly awkward against a without-replacement sequential draw and I would not
  bother.

### 6. Operational feasibility: 3,000 entries/day is affordable, because settlement cost scales with *distinct legs*, not entries — verified against the code

I traced the nightly settlement + CLV path. **Settlement volume is bounded by the number of
distinct (player, market, date, line) legs, which is a property of the daily slate — not by
how many replicate entries reference those legs.**

- `resolve_history` (`analysis.py:254`) fills `Actual` by a **per-(player, date) gamelog
  lookup**, and history is keyed one row per (prediction × book offer). Ledger legs draw from
  that same already-scored offer universe. Across 300 replicates a persona re-selects the
  *same small pool* of that day's candidate legs; the number of *distinct legs to resolve* is
  the daily offer count (order 10²–10³ across all live leagues), regardless of how many
  replicate entries cite each leg. Resolution is memoizable per distinct leg.
- CLV fill (`clv.fill_from_archive`, `clv.py:219`) is explicitly **group-invariant**: it
  groups by `PREDICTION_KEY` and does *one* archive read per (player, market, date) group,
  reusing it across every offer row in the group (`_fill_one_group` docstring: "runs once per
  group, not once per line"). 3,000 replicate entries over the same ~hundreds of distinct legs
  = the *same* number of archive reads as the single-path design. **CLV cost is flat in
  replicate count.**
- What *does* grow linearly with entries: (a) the append to `data/ledger/entries/{date}.jsonl`
  (3,000 vs 30 JSONL lines/day — trivial I/O, ~MB/day), (b) the per-entry *payout settlement*
  (push/void rules + payout curve) in the stage-2 extension, which is arithmetic over already-
  resolved legs — vectorizable, microseconds each, so ~3,000/day is milliseconds total, and
  (c) parquet size of `bankroll.parquet` / settled-entry parquet (300× rows). None of these is
  close to a bottleneck; the nightly job's real cost is `stats.update()` + `resolve_history`'s
  gamelog joins, both **independent of replicate count**.

**Estimate:** at 300 replicates the *incremental* nightly cost over a single-path ledger is
dominated by the extra ~3,000 payout-arithmetic ops and the larger parquet write — sub-second
added to a job already dominated by API `update()` calls. **Feasibility is not the constraint;
warrant is.** (This is *why* the recommendation to cut to 40 is about parsimony and honest
reporting, not a performance wall.)

### 7. Pre-registration / hindsight consistency: the owner's read is correct, with one real subtlety

**Confirmed:** all replicates are drawn and committed at decision time from that day's live
data only; none uses hindsight; averaging over replicates reports the *policy's* outcome
distribution, not a cherry-picked path. This is *more* hindsight-safe than a single path,
because it removes the "we happened to draw a lucky/unlucky realization" degree of freedom.
Nothing about `M > 1` violates §4's locked pre-registration principle **provided each
replicate's seed is fixed at commit time** (§10's `(date, persona, run_slot)` seeding already
guarantees reproducibility; extend it to `(date, persona, run_slot, replicate_id)` and the
whole ensemble is as immutable and re-runnable as the single path).

**The subtlety — and it is real:** reporting "mean + range" can be *read* as more
authoritative than one pre-registered path, and there are two ways that misleads if you are not
careful:

1. **The range must not be mistaken for forecast uncertainty about the world.** The
   between-replicate spread is **selection-algorithm noise** (which entries the draw happened
   to pick), *not* the uncertainty in whether the system makes money. Two personas could have
   identical true edge and different replicate spreads purely from how peaky their priority
   scores are. If the dashboard labels the band "range of outcomes," a reader infers risk that
   isn't there. It must be labelled as what it is: *"spread across equivalent draws of the
   selection policy."* The genuine forward evidence for D7 is the **mean's** trajectory and its
   *sampling* CI, plus realized CLV — not the width of the replicate band.
2. **Averaging can launder a fragile policy.** The ensemble mean of a persona that
   *occasionally* makes a catastrophic selection looks smooth, hiding left-tail fragility that a
   real single-bankroll bettor would feel. This is an *argument for* keeping a canonical single
   path (replicate 0) whose *actual* drawdown drives the stage-3 circuit breaker — the breaker
   should fire on a lived path, not on an ensemble average that never has a bad week. **Do not
   let the ensemble mean drive the circuit breaker;** let replicate 0 (the pre-registered path)
   drive it, and use the ensemble only for the skill-vs-variance decomposition D7 reads.

Net: the ensemble *strengthens* D7's evidence by **separating skill (the mean) from selection
variance (the spread)** — which is precisely the decomposition the spread-skill literature
formalizes — *as long as* the labelling is honest and the breaker runs on a real path.

---

## Recommendation / policy-spec revision (concrete, foldable)

Adopt a **hybrid ensemble** — not the 300-brute-force design, and not a bare single path:

**R1 — Replicate count: `M = 40` per persona (120 total), not 100 (300).**
`M = 40` sits at the knee of the `1 + 1/M` curve (within ~1% of the infinite-`M` ensemble
mean; the 40→100 step buys <1% and is invisible against real-world ROI noise). Expose it as a
single named constant `LEDGER_REPLICATES = 40` (round, easy to reason about, honestly labelled
as "at the ensemble-mean knee, not a tail-resolving count"). This is a `policy_v1` parameter;
changing it later is a `policy_v2` per §4.

**R2 — Canonical path: replicate 0 is the pre-registered ledger; 1…39 are the envelope.**
Replicate 0 uses the exact `(date, persona, run_slot)` seed §10 already specifies. It is *the*
committed ledger for every existing purpose: the circuit breaker (§6 stage 3) reads replicate
0's lived bankroll/drawdown; hindsight-proofing golden tests (§6 stage 4) assert on replicate
0's `committed_at`. Replicates 1…39 seed on `(date, persona, run_slot, replicate_id)` and exist
only to quantify selection-variance around replicate 0.

**R3 — Common Random Numbers across personas (from the Ladders CRN reasoning).**
For a given `(date, run_slot, replicate_id)`, draw all three personas from one shared uniform
stream consumed in fixed persona order, so the persona *contrast* (the lane's real output) has
the shared-slate noise cancelled. Free; sharpens every cross-persona comparison at any `M`.

**R4 — Report the MEAN and a CENTRAL-SPREAD band, never raw 5/95 percentiles.**
Report per persona: (i) **ensemble-mean** ROI/CLV/bankroll (the headline, D7's skill estimate);
(ii) a **±1 SE** band on that mean *and* the **inter-quartile range (25th–75th)** of the
replicate distribution as the selection-variance envelope. IQR is robust at `M = 40` (25/75 are
central enough to be stable at tens of replicates, unlike 5/95). **Explicitly do not plot 5/95
or min/max** off the live ensemble — at `M = 40` those are noise, and even at `M = 100` they
would be. Label the band "spread across equivalent policy draws," not "range of outcomes."

**R5 — Offload the full outcome distribution / tail question to a retrospective
`profit_sim.py`-style study on the accumulated real ledger.**
The owner's underlying want — "what's the full distribution of outcomes, including tails" — is
a **retrospective** question best answered the way this repo already answers it: once enough
real forward ledger has accrued, run a large-`N` (2,000+) resampling characterization *over the
settled real entries* to get stable tail percentiles. That is exactly the role
`docs/handoffs/sim-bettor-ledger.md` §1 reserves for `profit_sim.py` ("retrospective … strategy
exploration"). Keep that machinery for the tail claim; keep the live ledger lean (M=40, central
bands). This respects the "do not modify `profit_sim.py`" rule (you *use* its pattern
retrospectively; you don't fold 300 forward replicates into the live job).

**R6 — Settlement/CLV: memoize per distinct leg (already the CLV design; extend to resolution).**
Because CLV fill is already group-invariant and resolution is a per-(player, date) lookup,
settling 40 (or 100) replicates must reuse one resolution per distinct leg. The stage-2 extension
should resolve the *union of distinct legs* once, then map outcomes onto replicate entries —
never resolve per entry. This keeps nightly cost flat in `M` (verified feasible in §Q3).

---

## Reality checks

- **Effect size of the whole change vs a single path:** the ensemble mean at `M=40` cuts the
  between-replicate sampling noise on the reported daily statistic by **√40 ≈ 6.3×** versus one
  path — a genuine, worth-having reduction in *reported* jitter. Going to `M=100` improves that
  to √100 = 10× on the *daily* mean, but on the **30-day window** that D7 actually reads, both
  are already noise-limited by real-world ROI variance, not replicate count — the marginal
  real-world value of 100 over 40 is unmeasurable. **Regime where this holds:** the low-variance
  bounded-daily-ROI regime this policy is in (≤5 entries, quarter-Kelly, 0.5% cap, fixed
  bankroll). If a `policy_v2` ever removed the stake cap or the entry-count cap, per-replicate σ
  would rise and the knee would move right — re-derive then.
- **Build cost:** this is an **engineering task, not a research bet.** M=40 vs M=100 is one
  constant; the CRN change is one shared RNG instead of three seeds; the reporting change is
  choosing IQR/SE over percentiles in the stage-3 page; the memoized settlement is the natural
  implementation anyway. No new abstraction, no new module beyond the already-planned
  `strategies/ledger.py`. Fully consistent with CLAUDE.md "avoid over-engineering."
- **What could make this wrong:** (a) If the personas' priority scores turn out *extremely*
  peaky (one candidate dominates the draw), the effective selection entropy is low, the replicate
  spread collapses, and even M=40 is overkill — the stage-3 convergence study (§Q3) will reveal
  this and could justify M=20. (b) If, conversely, the candidate pool is large and flat and the
  Jaccard decay spreads mass widely, per-day selection variance is higher and the *daily* band is
  wide — but the *windowed mean* D7 reads is still fine at M=40; only the daily plot looks noisy.
  (c) The whole ensemble-mean framing assumes replicates are *exchangeable* draws of the same
  policy — true here by construction, but it would break if the seed derivation accidentally
  correlated replicates; the `(…, replicate_id)` seed must produce independent streams (use
  `np.random.default_rng` spawned children, not seed+id arithmetic that can collide).

---

## Open questions / caveats to carry into the plan

1. **Empirical knee check (stage 3, run once).** After ~2 weeks live, plot ensemble-mean
   ROI vs `M ∈ {10,20,40,80,160}` per persona and confirm the knee is ≤ 40; adjust the
   constant via `policy_v2` if the data says otherwise. Do **not** ship this as a recurring
   job — it is a one-time calibration.
2. **Circuit-breaker source of truth.** Confirm with the owner that the stage-3 breaker
   fires on **replicate 0's lived drawdown**, not the ensemble mean (§7 subtlety 2). This is a
   design decision the breaker-threshold escalation (§8) should ratify.
3. **Band labelling.** The stage-3 dashboard must label the replicate band as
   *selection-policy spread*, not outcome uncertainty (§7 subtlety 1) — a DESIGN.md / copy
   decision, flagged so it isn't read as risk.
4. **CRN persona ordering.** Fixing a persona consumption order over a shared stream (R3)
   induces a deterministic coupling; confirm this is acceptable vs fully-independent persona
   draws (it trades a tiny bias-free coupling for a large variance reduction on the contrast —
   standard CRN, but worth an explicit owner nod since "three *independent* personas" is
   §10 language).
5. **Retrospective tail study trigger (R5).** Define the minimum accrued real-ledger size
   before the `profit_sim.py`-style large-`N` tail characterization is meaningful (echoes the
   `CLV_SEGMENT_MIN_N`/`_MIN_BETS_FOR_PRECISION` thresholds already in the repo).

---

## Bibliography

| # | Source | Identifier / URL | Bears on |
|---|---|---|---|
| 1 | Leutbecher, M. (2019). *Ensemble size: How suboptimal is less than infinity?* Quarterly Journal of the Royal Meteorological Society, 145(S1). | doi:10.1002/qj.3387 | The `1 + 1/M` score-convergence law; 50→100 members ≈ 1%; tail scores more `M`-sensitive (Q1, K1, K4) |
| 2 | ECMWF Newsletter 157 — *How many ensemble members are desirable?* | ecmwf.int/en/newsletter/157/news/how-many-ensemble-members-are-desirable | Concrete numbers: `100(M₁−M₂)/[M₂(M₁+1)]%`; 50→100 = 1%, 50→20 = 3% (Q1, K1) |
| 3 | Ferro, C. A. T. (2014). *Fair scores for ensemble forecasts.* QJRMS 140(683). | doi:10.1002/qj.2270 | Fair scores make small-`M` ensembles usable; expected score independent of `M` under exchangeability (K1 framing) |
| 4 | Leutbecher, M. & Palmer, T. N. (2008). *Ensemble forecasting.* J. Comput. Phys. 227(7). | doi:10.1016/j.jcp.2007.02.014 | Spread–skill: ensemble-mean error vs member spread equal in expectation for reliable ∞-ensemble (K1, Q7 decomposition) |
| 5 | Ocean Optics Web Book — *Monte Carlo Simulation: Error Estimation.* | oceanopticsbook.info/view/monte-carlo-simulation/level-2/error-estimation | `SE = σ/√N`; `N = (z·σ/ε)²` CI-half-width sizing; √-law (Q1, K3) |
| 6 | Law, A. M. & Kelton, W. D. *Simulation Modeling and Analysis* (common random numbers chapter). | (standard text; ed. 2000/2015) | CRN as variance reduction for comparing systems; paired-estimator covariance term (Q2, K5, R3) |
| 7 | l'Ecuyer, P. *Randomized Quasi-Monte Carlo.* | link.springer.com/chapter/10.1007/0-306-48102-2_20 | RQMC/CRN across candidates lowers *difference* variance; QMC regime limits (K5) |
| 8 | Metricgate — *How Many Bootstrap Replicates Do You Actually Need?* | metricgate.com/blogs/choosing-number-of-bootstrap-replicates/ | ≥2,000 for percentile bands, 5,000+ for extreme tails; center tolerates small B (Q1/Q5, K4) |
| 9 | Wilcox / garstats — *The percentile bootstrap.* | garstats.wordpress.com/2016/05/27/the-percentile-bootstrap/ | Tail endpoints noisier per replicate; `1/√B` on band edges (K4, R4) |
| 10 | SAS — *On monitoring the convergence of Monte Carlo simulations* (Welford running SE). | blogs.sas.com/content/iml/2026/04/13/monitor-convergence-monte-carlo.html | Convergence should be checked via running SE, not assumed (K3, OQ1) |
| 11 | Repo: `strategies/profit_sim.py` + git `2c28fac` | profit_sim.py:36-38, 197 | `N=100` is a page-latency default; gate callers use `N=1` on paired universe (Q1/Q2, K2) |
| 12 | Repo: `strategies/kelly.py` | kelly.py:33,37; joint_kelly_portfolio 162-214 | Stake caps bound per-replicate daily-ROI σ (Q1/K3); portfolio solver assumes de-overlapped candidates |
| 13 | Repo: `clv.py` + `analysis.py` + `nightly.py` | clv.py:219-250 (group-invariant fill); analysis.py:254 (per-player resolution); nightly.py run() | Settlement/CLV cost flat in replicate count → 3,000 entries/day affordable (Q3, K6, R6) |
| 14 | Repo: `docs/archive/researcher_ladders_stage0.md` | A1/A2 (Sobol' N=8,192, reused-pool CRN) | CRN transfers, Sobol'/QMC does not (Q2, K5) |
| 15 | Repo: `docs/handoffs/sim-bettor-ledger.md` §1, §4, §10 | (file) | Locked pre-registration principle; `profit_sim.py` reserved for retrospective use (Q4, R2/R5) |
