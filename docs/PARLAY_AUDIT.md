# Parlay Audit

Audit of `sportstradamus.prediction.correlation.find_correlation` and
`sportstradamus.prediction.correlation.beam_search_parlays`. Scope: read-only
analysis. No production code is changed in this pass; remediation is gated on
the findings below and lands in the next session.

> **Layout note.** `CONTRIBUTING.md` advertises a separate
> `prediction/parlay.py` module. It does not exist on disk:
> `beam_search_parlays` is defined in `prediction/correlation.py` (line 504)
> alongside `find_correlation`. The audit therefore reads both functions out
> of `correlation.py`. Splitting them is a refactor candidate but is out of
> scope here.

---

## 1. `find_correlation`

### 1.1 Joint-probability formula

`find_correlation` does **not** itself compute a parlay joint probability.
It builds two pairwise matrices and a per-leg "correlation multiplier" hint,
then hands the matrices to `beam_search_parlays`. The two pieces of
parlay-relevant arithmetic inside `find_correlation` are:

**Pairwise EV matrix** (`correlation.py:389-402`):

```python
EV = (
    np.multiply(
        np.multiply(np.exp(np.multiply(C, V)), P),
        boosts.reshape(len(boosts), 1) * M * boosts,
    )
    * payout_table[platform][0]
)
```

For two legs `i, j` this is

```
EV[i, j] = exp(C[i,j] * V[i,j]) * P[i,j] * boosts[i] * M[i,j] * boosts[j] * payouts[0]
```

with `V[i,j] = sqrt(p_i*(1-p_i) * p_j*(1-p_j))`, `P[i,j] = p_i * p_j`, and
`payouts[0]` the 2-leg payout multiplier. So the implicit pairwise-joint
estimator inside `find_correlation` is

```
P_joint(i, j) ≈ exp(C[i,j] * V[i,j]) * p_i * p_j        (line 391)
```

This is **neither** a Gaussian copula nor a Pearson product. It is a
log-linear / exponential boost on the independent product:

```
log P_joint = log p_i + log p_j + C[i,j] * sqrt(p_i(1-p_i)*p_j(1-p_j))
```

i.e. an additive-log-odds-style perturbation where the perturbation size is
scaled by the geometric mean of the per-leg standard deviations. It is fast
and produces sensible signs, but it is not a probability — values can exceed
`1` for high `C` and intermediate `p`, which is then clipped only implicitly
when the EV column is filtered (`EV[:, i] > 0.95`, line 405).

The actual joint probability used to rank a complete parlay lives in
`beam_search_parlays` (§2.1) via a Gaussian copula on `C`.

**Per-pair correlation** (`correlation.py:367-386`):

```python
rho = 0
boost = 0 if (n1 in n2 or n2 in n1) else 1
for xi, x in enumerate(cm1):
    for yi, y in enumerate(cm2):
        increment = c_map.get((x, y), c_map.get((y, x), 0))
        if b1[xi] != b2[yi]:
            increment = -increment
        rho += increment
        ...
        modifier = mod_map.get(frozenset([x_key, y_key]), [1, 1])
        boost *= modifier[0] if b1[xi] == b2[yi] else modifier[1]

C[i, j] = C[j, i] = rho / len(cm1) / len(cm2)
M[i, j] = M[j, i] = boost
```

`c_map` is the per-game pair lookup built by `_build_game_corr_map`
(`correlation.py:56-93`). The pair score is summed across every
`(market_a, market_b)` pair from the two combo entries, sign-flipped when
the bet sides disagree, then averaged over the cartesian-product size. The
result lives in `[-1, 1]` for normal single-market legs but is **not
guaranteed** to be a valid correlation — there is no positive-semidefinite
projection.

### 1.2 Same-player guard

`find_correlation` does not look at player identity when populating `C`. A
weak guard exists on the boost matrix only:

```python
boost = 0 if (n1 in n2 or n2 in n1) else 1   # correlation.py:368
```

This zeroes the **boost**, not the correlation, when one player name is a
substring of the other. The intent is to neutralize duplicate-player pairs,
but:

- It catches strict prefixes/suffixes, missing e.g. "James Harden" vs
  "Tyrese Haliburton" (both contain "ar") — false positives are avoided by
  the `in` check, but exact-name self-pairs are caught.
- It does **not** affect `C[i, j]`, so the Gaussian-copula path in
  `beam_search_parlays` will still receive a non-zero correlation for two
  legs on the same player.
- The deduplication that actually keeps same-player pairs out of the parlay
  search lives in `beam_search_parlays` (`leg_players` check, lines 552-556),
  not here.

So the same-player invariant is enforced **by the beam search**, not by
`find_correlation`. Combo legs (`" + "`, `" vs. "`) sidestep the substring
check by construction.

### 1.3 Banned-combos usage

Yes. Loaded once at import time in `helpers/config.py:67`:

```python
with open(pkg_resources.files(data) / "banned_combos.json") as infile:
    banned = json.load(infile)
```

Used in `find_correlation` to pull two modifier maps per `(platform, league)`:

```python
team_mod_map = banned[platform][league]["team"]
opp_mod_map = banned[platform][league]["opponent"]   # correlation.py:196-197
```

Inspecting `data/banned_combos.json` shows the file is **not** a hard-banlist
of pairs — every entry is a tuple `[same_side_multiplier,
opposite_side_multiplier]`, e.g.
`"QB.completions & WR.receptions": [0.73, 1.05]`. So the file functions as a
soft-modifier table feeding into `M[i, j]` (line 384):

```python
modifier = mod_map.get(frozenset([x_key, y_key]), [1, 1])
boost *= modifier[0] if b1[xi] == b2[yi] else modifier[1]
```

A real hard-ban would set the modifier to `0`, which the rest of the
pipeline interprets via the `boost <= 0.7` filter in `beam_search_parlays`
(line 585). Pairs that should never appear together would need an explicit
zero entry; today nothing in the file is below `0.7`, so the file behaves
purely as a calibration nudge. Pair selection is `frozenset`-keyed on the
position-stripped market names, so order-independence is preserved but
position-specific bans (e.g. `QB1` vs `QB2`) are not expressible.

---

## 2. `beam_search_parlays`

### 2.1 Beam width and configurability

```python
K = 1000                                # correlation.py:531
max_bet_size = len(payouts) + 1         # correlation.py:532
```

`K` is hardcoded. There is no CLI flag, no environment variable, no kwarg.
Changing the beam width requires editing the source. `max_bet_size` is
implicitly configurable per-platform via the length of the
`payout_table[platform]` list (5 entries → max 6-leg parlays for Underdog
and PrizePicks).

### 2.2 Per-step scoring

Partial parlays are scored by the **geometric mean of pairwise EV** over
the upper triangle (`correlation.py:560-563`):

```python
n_pairs = target_size * (target_size - 1) // 2
ev_prod = np.prod(EV[np.ix_(extended, extended)][np.triu_indices(target_size, 1)])
geo_mean = ev_prod ** (1 / n_pairs)
if geo_mean < 1.05:
    continue
```

The score therefore uses `EV` from `find_correlation` (the log-linear
approximation in §1.1), not the multivariate-normal joint probability. The
final scoring on survivors swaps to the copula model:

```python
SIG = C[np.ix_(bet_id, bet_id)]
if any(np.linalg.eigvalsh(SIG) < 0.0001):    # correlation.py:599
    continue
payout = np.clip(payout_base * boost, 1, 100)
p = payout * multivariate_normal.cdf(norm.ppf(p), np.zeros(bet_size), SIG)
```

So the joint probability surfaced as `Model EV` is

```
P_joint(parlay) = Φ_n( Φ⁻¹(p_1), …, Φ⁻¹(p_n); Σ = C[bet_id, bet_id] )
```

— a **Gaussian copula** on the single-leg model probabilities, with the
correlation matrix taken directly from `C` (after a small PSD check). Two
notes on the copula:

1. `C` is built pair-by-pair without any PSD projection. The eigenvalue
   guard at line 599 only **rejects** non-PSD subsets; nothing in the
   pipeline repairs them, so candidate parlays whose `Σ` is singular or
   non-PSD are silently dropped.
2. `multivariate_normal.cdf` calls SciPy's mvn integration, which gets
   numerically noisy for `n >= 5` and tight correlations. There is no
   sample-count or tolerance argument passed.

### 2.3 Constraints enforced

| Constraint | Where | Threshold |
|---|---|---|
| Strictly increasing leg index | line 550 | `new_leg <= last_idx → skip` |
| Same-player dedup (exact + substring) | lines 552-556 | `new_player in used_players` or substring overlap |
| Per-step EV floor | line 563 | geo-mean pairwise EV `< 1.05` → drop |
| Beam width | line 569 | top `K=1000` survive |
| Both teams must appear | lines 577-580 | parlay must touch `team` and `opp` |
| Boost band | line 585 | `0.7 < boost ≤ max_boost` (`2.5` Underdog, `60` else) |
| Books-EV floor (independent) | line 590 | `prev_pb < 0.9 → drop` |
| Model-EV floor (independent) | line 595 | `prev_p < 1.5 → drop` |
| Σ positive-definite | line 599 | min eigenvalue `< 1e-4 → drop` |
| Final EV floors | line 607 | `units < 0.5` or `p < 2` or `pb < 0.9 → drop` |

Constants worth flagging — none are module-level named constants, all are
inline magic numbers (CLAUDE.md §"No magic numbers"):

- `K = 1000` (beam width)
- `1.05` per-step EV cutoff
- `0.7`, `2.5`, `60` boost bands
- `0.9`, `1.5`, `2`, `0.5` final filters
- `1e-4` PSD tolerance
- `np.clip(..., 1, 100)` payout cap

### 2.4 Payout source

```python
payout_table = {                                            # correlation.py:171-177
    "Underdog":   [3.5, 6.5, 10.9, 20.2, 39.9],
    "PrizePicks": [3,   5.3, 10,   20.8, 38.8],
    "Sleeper":    [1, 1, 1, 1, 1],
    "ParlayPlay": [1, 1, 1, 1, 1],
    "Chalkboard": [1, 1, 1, 1, 1],
}
```

Hardcoded in `find_correlation` and passed to `beam_search_parlays` as
`payouts`. **Not parametrized** by contest variant. The Underdog table
encodes a single payout curve — the comment says "equivalent payouts when
insured picks are better", so it is the insured-pick line. Power vs. Flex
vs. Standard are **not** distinguished. Sleeper / ParlayPlay / Chalkboard
are stubbed with all-ones, which means EV is essentially the joint
probability for those platforms (the `payout * mvn.cdf` term collapses to
`mvn.cdf`).

A second, contradictory payout table appears at the very end of
`find_correlation` (`correlation.py:497-499`), used only when
`platform == "Underdog"` to overwrite the `Boost` column post-search:

```python
payouts = [0, 0, 3.5, 6.5, 6, 10, 25]
parlay_df["Boost"] = parlay_df["Bet Size"].apply(lambda x: payouts[x]) * parlay_df["Boost"]
```

The numbers (`6, 10, 25` for 4/5/6-leg) match the standard-pick line, not
the insured-pick line that drove the search. The downstream `Boost` column
therefore reflects a **different** payout regime than the one used to rank
parlays. This is a likely scoring-vs-display mismatch worth checking in the
remediation pass.

### 2.5 Output ranking and dedup

Within `beam_search_parlays`, results are returned in the order they were
generated (one block per target size, ordered by the geometric-mean EV that
won the beam). **No final sort** happens inside the function.

Sorting and deduplication happen back in `find_correlation`
(`correlation.py:455-465`):

```python
df5 = (
    pd.concat(
        [
            bets.sort_values("Model EV",  ascending=False).head(300),
            bets.sort_values("Rec Bet",   ascending=False).head(300),
            bets.sort_values("Fun",       ascending=False).head(300),
        ]
    )
    .drop_duplicates()
    .sort_values("Model EV", ascending=False)
)
```

- Three rankings: Model EV (correlated), Rec Bet (Kelly units), Fun (a
  hand-rolled novelty score).
- `drop_duplicates()` runs on the full row, but `Bet ID` is a tuple of leg
  indices in lexicographic order, so order-permuted duplicates are already
  collapsed by the strictly-increasing index constraint inside the beam.
- **Overlapping** parlays — i.e. parlays that share legs — are not
  deduplicated. The `Family` column (lines 467-491) clusters parlays into
  three families using Ward linkage on a normalized cross-correlation
  distance, but every member is kept; the consumer is expected to pick one
  per family.

Dedup against the Sheets layer happens in `prediction/sheets.py:save_data`
(`sheets.py:60-64`), which adds `(Platform, Game, Family) → Rank` columns
but again does not drop overlaps.

---

## 3. Empirical calibration

The audit script
[`src/sportstradamus/scripts/audit_parlay_calibration.py`](../src/sportstradamus/scripts/audit_parlay_calibration.py)
inverts `Model EV → joint probability` via

```
joint_p = Model EV / clip(payout_table[Platform][Bet Size - 2] * Boost, 1, 100)
```

then buckets resolved parlays from `data/parlay_hist.dat` (last 90 days) by
decile of `joint_p`, computes hit rate from the `(Legs == Bet Size,
Misses == 0)` indicator filled in by `reflect`, and writes a reliability
diagram plus a CSV of the per-decile counts.

**Run** (2026-05-05):

```text
No resolved parlays found in window [2026-02-04 .. 2026-05-05].
Wrote placeholder artifacts to /home/user/Sportstradamus/docs.
```

`data/parlay_hist.dat` is not present in this checkout — only `*_res.json`
result fixtures live under `data/`. The script wrote
[`PARLAY_CALIBRATION_2026-05-05.png`](PARLAY_CALIBRATION_2026-05-05.png)
(placeholder) and [`PARLAY_CALIBRATION_2026-05-05.csv`](PARLAY_CALIBRATION_2026-05-05.csv)
(empty header row). On a host where `prophecize` and `reflect` have been
running for ≥ 90 days, re-running the script produces a populated
diagram without code changes:

```bash
poetry run python -m sportstradamus.scripts.audit_parlay_calibration
```

The empirical calibration question therefore remains **open** in this
session and must be re-run on a host with production parlay history.

---

## 4. Findings (5–10 bullets)

- `find_correlation` is a feature-engineering pass plus a per-game scratch
  EV matrix; the actual parlay joint probability is a Gaussian copula on
  `C` evaluated inside `beam_search_parlays`
  (`correlation.py:603`).
- `find_correlation`'s pairwise EV approximation
  `exp(C * sqrt(V_i V_j)) * p_i p_j` (line 391) is not a probability and
  is unbounded above; downstream filters mask the issue rather than fix
  it.
- The Σ matrix passed to `multivariate_normal.cdf` is built pair-by-pair
  with no PSD projection. Singular submatrices are dropped (line 599) but
  not repaired, biasing the surviving population towards loosely
  correlated parlays.
- Same-player guarding lives only in the beam search (`leg_players` set,
  lines 552-556). `find_correlation` would happily put a non-zero
  correlation between two legs on the same player; relying on substring
  matching for the dedup is fragile (e.g. "Mike Williams" vs "Mike
  Williams Jr.") but is currently the only line of defense.
- `data/banned_combos.json` is a soft-modifier file, not a banlist. No
  entry zeros a pair, so nothing in production is hard-banned today; the
  `boost <= 0.7` cutoff (line 585) gives a back-door hard-ban via cumulative
  product but no individual pair triggers it.
- Beam width `K=1000`, per-step EV cutoff `1.05`, boost bands, and final
  EV floors are all inline magic numbers. They violate CLAUDE.md §"No
  magic numbers" and make tuning impossible without source edits.
- `payout_table` is per-platform but not per-contest-variant. Power, Flex,
  and Standard Underdog payouts collapse into one table, and Sleeper /
  ParlayPlay / Chalkboard are stubbed with `1`s so their Model EV is
  effectively just the copula joint probability.
- `find_correlation` overwrites the `Boost` column at the end with a
  *different* payout table (`[0, 0, 3.5, 6.5, 6, 10, 25]`,
  correlation.py:498) than the one used to rank parlays. The displayed
  `Boost` does not correspond to the EV that drove selection.
- Parlays are deduplicated only by exact bet-id; overlap-aware dedup is
  delegated to a 3-cluster Ward linkage (`Family` column) without
  enforcing one-per-family selection.
- Empirical calibration is **not measurable** in this checkout —
  `parlay_hist.dat` is absent. The audit script runs end-to-end and
  produces a placeholder PNG/CSV; re-run on a production host to populate.
