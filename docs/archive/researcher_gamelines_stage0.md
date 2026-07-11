# In-repo research brief — game-line (spread/total/moneyline) decision engine for DFS apps

Statistician's Stage-0 brief for a new `dfs-products` lane (Sportstradamus). Answers the
pricing / combo-EV / player×game-line-ρ design question for Underdog Prediction Picks +
Combo Entries and Sleeper Markets. Owner decision is **VERIFY-FIRST**: every
product-mechanic assumption is tagged **CONDITIONAL** and mapped to the Stage-0 capture
fact that de-conditionalizes it (§B7). Date: 2026-07-10. Read-only w.r.t. production;
zero edits to `parlay.py` / `correlation.py` / `correlate.py` at v1.

Repo facts verified by reading source; product facts web-verified against
help.underdogsports.com (some articles 403 to plain fetch — mechanics recovered from the
search-index rendering of the same help articles + secondary DFS references, flagged
inline). All product mechanics remain CONDITIONAL on Stage-0 payload capture regardless of
web confirmation, because terms vary by state, account cohort (Kalshi vs CDNA), and change
without notice.

---

## Verdict summary (up front)

**PROCEED to Stage-0 payload capture; do NOT build the engine yet.** The design is sound
and composes cleanly with the incumbent machinery import-only, but its **entire edge
thesis rests on one unverified fact**: whether Underdog's Combo Entry reprices the
*event × fantasy-pick* correlation. Web evidence shows UD *does* reprice *player × player*
correlation ("Correlated Projections": positive corr → lower multiplier, MM may decline a
pair) but the combo's event leg is a **separately-priced CFTC contract at market price**
that then rolls its payout into an independently-priced fantasy entry — strongly
suggesting the event×pick correlation is **untaxed**. If Stage-0 confirms that gap, the
lane has a real, structural edge (a correlated event leg levers the fantasy stake at a
price that ignores the correlation). If Stage-0 shows the combo *does* tax event×pick
correlation, the primary edge source collapses and the lane narrows to divergence-alerting
only (§B8 kill-adjacent).

- **B1 (combo EV):** derived closed-form. `EV(combo) = 0.1F·V_fan(no roll) + [contract
  branch]`; the coupling enters through `P(event AND all picks) = P(event)·P(picks) +
  ρ-driven excess`, and the rolled stake makes the excess **multiply the fantasy payout
  curve**, not just add. Every mechanic CONDITIONAL-mapped.
- **B2 (edge decomposition):** four sources; the dominant one is **(i) fantasy side not
  pricing event×pick correlation** IF §B7-P4 confirms it. (ii) leverage of a real pick
  edge and (iv) contract-vs-consensus divergence are secondary and measurable now;
  (iii) fee asymmetry is small and computable. Decision rule + numeric thresholds given.
- **B3 (player×game-line ρ):** estimand = corr of residualized player stat with
  game-outcome residual vs the **de-vigged close** (close = conditional mean, so the
  residual is the correct dependence target — **justified**, see §B3). Standalone script
  over gamelogs + archived closes; hierarchical Fisher-z EB per the copula brief's R3;
  viability gate ≥8 pair-types at pooled N≥300/league; migration path into `correlate.py`
  after `sleeper-parity`.
- **B4 (event marginal):** trust the **Odds-API de-vigged consensus** as truth; treat the
  **contract market price** as the *tradeable* price and the divergence `|m − p̂_devig|`
  as the edge/alert signal. Fee-adjust the contract price. Power de-vig for lopsided
  moneylines. Numeric trigger given.
- **B5 (mixed-slip pricing):** the event leg composes as **one more binary cut at
  `norm.ppf(p_event)`** with a ρ column — imports `parlay.py`'s analytical MVN + push-MC
  **with zero edits**. The one real gap: **spread/total legs CAN push** (margin/total
  lands exactly on an integer line) but `get_push_prob` returns 0 for `dist="Normal"`, so
  the incumbent would silently treat event pushes as impossible. Cheapest correct
  treatment specified (feed a nonzero `p_push` for integer event lines into the existing
  3-band MC classifier).
- **B6 (cash-out):** **ignore option value at v1** (treat as European; value = intrinsic
  only). Quantified what's left on the table (small; the option is mostly a
  variance-management tool, not an EV source, given near-sharp pricing).
- **B7:** exact payload-capture checklist, each field keyed to the answer it flips.
- **B8:** four measurable kill conditions; the binding one is "combo prices event×pick
  correlation" (§B7-P4 returns 'taxed').

Two-line compute budget verdict: game-line scoring is `O(games × markets × 2)` ≈ a few
hundred CDF evals; combo enumeration is the cost driver and is bounded to
`O(scored_player_offers × game_line_legs)` after a two-stage filter — **well inside the
15-min budget**; full worst-case wall-time in the compute-budget section.

---

## B1 — Combo-entry EV algebra

### Mechanics as verified (all CONDITIONAL, §B7 keys in brackets)

From the UD help articles (Prediction Picks Overview; Pick'em Classic Combo Rules), as
rendered in search and corroborated by secondary DFS references:

- A Combo Entry = **1 prediction pick** (event contract, CFTC/NFA, priced $0-$1) **+ 2 to
  8 fantasy picks**. [P1]
- **Fee split:** "the majority of the entry fee is used to purchase the event contract(s),
  with **10% allocated to the Fantasy pick**." $10 combo → **$1 fantasy, ~$9 contract**.
  The prompt's ~0.9F/~0.1F is confirmed as the *nominal* split. [P2] — exactness (rounding,
  whether the 90% buys an integer number of $-priced contracts) is CONDITIONAL.
- **Roll mechanic (the crux):** "If your Prediction wins (**or you cash out early**), and
  your Fantasy entry wins → your combined payout is determined based on **normal payout
  rules for a fantasy entry, with your entry fee being the amount of your fantasy
  reservation + the amount paid out from your prediction entry**." So the prediction
  payout becomes **additional effective stake** in the fantasy payout calc. [P3]
- **Partial-payout floor (not in the prompt — material):** "if your prediction pick(s)
  lose but all your fantasy picks are correct, you still receive a partial payout based on
  standard Pick'em rules, **using your fantasy reservation fee as the stake**." So the
  fantasy side is NOT zeroed when the event loses — it pays on the 0.1F reservation. [P3]
- **Combo fantasy side does NOT support Flex** — all fantasy picks must win (power /
  all-or-nothing schedule on the fantasy legs). [P5]
- Contract settles $1 if event wins, $0 if not; net per contract = $1 − m − fees. [P6]

### Notation

- `F` — total entry fee (dollars).
- `s_fan = 0.1·F` — fantasy reservation (the ~10%). [P2]
- `s_ct = F − s_fan ≈ 0.9·F` — dollars spent on the event contract(s). [P2]
- `m` — event contract price ∈ (0,1) (dollars per contract; implied prob ≈ m). [P6]
- `c_fee` — per-contract fee (Sleeper: $0.02; UD: CONDITIONAL, likely bundled). [P6]
- `n_ct = s_ct / (m + c_fee)` — contracts bought with the 90% (integer-rounded; CONDITIONAL). [P2,P6]
- `A_ct = n_ct · $1 = s_ct/(m + c_fee)` — **gross contract payout if the event wins** (dollars). [P6]
- `p_e` — true P(event wins) (our de-vigged consensus estimate). [feeds B4]
- `q(k)` — fantasy power-schedule multiplier for `k` fantasy picks (Combo = power, no Flex):
  from `underdog_payouts.json` `power`: `q(2)=3, q(3)=6, q(4)=10, q(5)=20, q(6)=25`. [P5]
- `k` — number of fantasy picks; `p_π = P(all k fantasy picks win)` (joint, ρ-coupled). [feeds B5]
- `ρ_eπ` — dependence between the event outcome and the joint fantasy-pick outcome. [feeds B3]

### Effective stake into the fantasy payout, by branch

The fantasy entry pays `q(k) × (effective stake)` **iff all k fantasy picks win**, where
the effective stake depends on the event branch [P3]:

- Event **wins** (or is cashed out to `A_ct`): effective stake `= s_fan + A_ct`.
- Event **loses**: effective stake `= s_fan` (partial-payout floor). [P3]

So define the fantasy multiplier random variable on the *event* outcome:

```
stake_eff = s_fan + A_ct·1{event wins}
```

### EV(combo) — exact, with the coupling term explicit

Payout is nonzero only in the two "all fantasy picks win" cells. Writing the joint over
(event, all-picks):

```
Payout(combo) =
    q(k) · (s_fan + A_ct) ,  if event wins AND all k picks win
    q(k) · s_fan          ,  if event loses AND all k picks win
    0                     ,  otherwise
```

Take expectation. Let:

- `P11 = P(event wins AND all picks win)`
- `P01 = P(event loses AND all picks win) = p_π − P11`   (marginal of picks minus the joint)

```
EV_gross(combo) = q(k)·(s_fan + A_ct)·P11 + q(k)·s_fan·P01
                = q(k)·s_fan·p_π  +  q(k)·A_ct·P11
```

**Net EV** subtracts the fee `F` (the contract fee `c_fee·n_ct` is already inside `A_ct`
via the `m + c_fee` denominator; the fantasy 0.1F has no separate rake on UD pick'em —
CONDITIONAL [P2]):

```
┌─────────────────────────────────────────────────────────────────────┐
│  EV_net(combo) = q(k)·s_fan·p_π  +  q(k)·A_ct·P11  −  F               │
│                                                                       │
│  with A_ct = 0.9F / (m + c_fee),  s_fan = 0.1F,                       │
│       P11 = p_e·p_π + Cov_eπ,   Cov_eπ = P(event ∧ picks) − p_e·p_π   │
└─────────────────────────────────────────────────────────────────────┘
```

### The coupling term, made explicit

`P11` is where event×picks correlation enters, and it enters the **A_ct-levered** term —
the big one, because `A_ct ≈ 0.9F/m` dwarfs `s_fan = 0.1F` whenever `m < 0.9`.
Decompose:

```
P11 = p_e · p_π + Cov_eπ
```

- **Independence (naïve app assumption):** `Cov_eπ = 0` ⟹ `P11 = p_e·p_π`.
- **Correlated reality:** for a positively-correlated event+picks stack (e.g. team-Over
  total + its QB Over passing yards + WR Over receiving yards), `Cov_eπ > 0`, so
  `P11 > p_e·p_π`. **If the app prices the contract at `m ≈ p_e` and the fantasy multiplier
  `q(k)` at the independent joint, the combo's realized `P11` exceeds the priced joint —
  that surplus, levered by `A_ct`, is the edge.** [core thesis; gated on B7-P4]

Model the coupling through the **same Gaussian-copula machinery the parlay engine already
uses** (§B5): treat the event as one more binary leg with marginal `p_e` and a ρ column
`ρ_{e,j}` to each fantasy leg `j` (from B3), then

```
P11 = P(Z_e > z_e, Z_1 > z_1, …, Z_k > z_k)  under MVN(0, Σ),   z_· = norm.ppf(1 − p_·-side)
```

i.e. `P11 = multivariate_normal.cdf` over the (k+1)-leg correlation matrix — the **exact
call at parlay.py:352**. `p_π` is the same integral dropping the event row. So both terms
in `EV_net` come from one MVN CDF evaluation of the augmented leg set, plus one of the
un-augmented set. No new probability machinery.

### CONDITIONAL → capture-fact map for B1 (which fact flips which term)

| Term / assumption | Depends on capture | Flip if capture shows… |
|---|---|---|
| `s_fan = 0.1F`, `s_ct = 0.9F` | **P2** split exactness | different % → rescale `s_fan`, `A_ct`; a *fixed* $ reservation → `s_fan` constant not `0.1F` |
| `A_ct = 0.9F/(m+c_fee)`, integer `n_ct` | **P2, P6** rounding & tick | contracts are integer-lot → `A_ct` step-quantized; fee bundled → drop `c_fee` |
| roll into stake (`s_fan + A_ct`) | **P3** roll mechanic | if payout uses `max(s_fan, A_ct)` or a capped roll → replace the additive stake |
| partial floor `q(k)·s_fan·P01` | **P3** partial rule | if event-loss zeroes fantasy → drop the `s_fan·p_π` term entirely (all-or-nothing) |
| `q(k)` = power, no Flex | **P5** | if Flex allowed on combo fantasy → replace `q(k)·1{all win}` with the Flex miss-tier curve (use `_expected_payout_with_pushes` path) |
| cash-out = `A_ct` at exit | **P3, P7** cash-out timing | if cash-out disallowed once fantasy legs live → the "or cash out" branch drops; European treatment (B6) |

---

## B2 — Edge decomposition & decision rule

Where combo edge can come from, ranked by expected magnitude and by whether it is
measurable **today** vs **gated on B7**:

**(i) Fantasy side does not price event×pick correlation** — *the primary thesis; gated on
B7-P4.* Web evidence: UD reprices *player×player* correlation via "Correlated Projections"
(positive corr → lower multiplier; MM may decline a pair). But the combo's **event leg is a
separately-priced CFTC contract**, and its payout rolls into an **independently-priced
fantasy entry** — the two pricers are architecturally distinct, so the event×pick `Cov_eπ`
plausibly escapes the correlation tax that a player×player pair inside the fantasy entry
would pay. **Edge magnitude** ≈ `q(k)·A_ct·Cov_eπ` per §B1. At `F=$10, m=0.55, k=3
(q=6), Cov_eπ=0.03`: `A_ct≈16.4`, edge term `≈ 6·16.4·0.03 ≈ $2.95` gross uplift over the
independence price — i.e. the correlation surplus alone is ~30% of stake. **This is the
whole ballgame and it is CONDITIONAL.**

**(ii) Player-pick edge levered by the rolled stake** — *measurable now.* Our fantasy legs
already carry a certified marginal edge (the six ship gates). In a plain fantasy slip that
edge multiplies `s_fan`; in a combo it multiplies `s_fan + A_ct`. A pick with per-leg edge
`δ` sees its dollar contribution scaled by `(s_fan+A_ct)/s_fan ≈ 1 + 9/m` — a **~17×
leverage** at `m=0.55`. But leverage cuts both ways: it also levers *negative* pick edge
and adds the event's own price drag `(m − p_e)`. Net only when the pick edge is real AND
`p_e ≥ m` (event not overpriced). Measurable from `model_stats.parquet` (`brier_skill_score`,
`mean_ev_diff`) today.

**(iii) Fee asymmetries** — *small, computable now.* The 10% fantasy reservation carries no
separate rake on UD pick'em (the payout curve is the rake); the 90% contract carries
`c_fee` (Sleeper $0.02/contract ⟹ ~2% at `m≈1`, larger at low `m`). A combo pays the
contract fee that a plain fantasy slip avoids, so fees are a **drag on combos**, not a
source — quantify and subtract, don't chase.

**(iv) Contract price vs our de-vigged sharp consensus** — *measurable now, real, bounded.*
If the CFTC contract trades at `m` while our Odds-API de-vigged consensus says `p̂_e`, the
divergence `p̂_e − m` is a direct edge on the event leg itself (and levers into (i)/(ii)).
This is a **prediction-market-vs-book arbitrage-flavored signal**, exploitable *even
standalone* (buy the contract when `p̂_e − m > threshold`), and is the fallback edge if (i)
is taxed away. See B4 for the trigger.

### Decision rule (when to attach a game-line leg; when a combo dominates)

Let the plain fantasy slip have EV `EV_fan = q(k)·s_fan·p_π − s_fan` (staking only the
reservation-scale — i.e. the same k picks as a standalone entry at fee `s_fan`; use the
real standalone fee in practice). Attach an event leg / prefer the combo iff **all**:

1. **Event marginal not overpriced:** `p̂_e ≥ m − τ_price`, with `τ_price = 0.01` (1¢, ~1
   Sleeper tick). If the contract is richer than that beyond our consensus, the event drag
   eats the leverage — skip. [uses B4]
2. **Positive coupling:** `Cov_eπ ≥ τ_cov`, `τ_cov = 0.01` (post-shrinkage; below this the
   copula surplus is within estimation noise per B3). Sign matters — a *negatively*
   coupled event leg (e.g. own-team Under total with player Overs) makes `P11 < p_e·p_π`
   and should be **rejected**, not attached. [uses B3]
3. **Combo EV clears the incumbent floors:** `EV_net(combo)/F ≥ EV_net(plain)/s_fan` AND
   the combo's `EV_net(combo) ≥ _MODEL_EV_FINAL_FLOOR`-equivalent on a per-dollar basis
   (`parlay.py:78`, `_MODEL_EV_FINAL_FLOOR=2.0` is a payout-multiple floor; the combo
   analogue is `EV_net(combo)/F + 1 ≥ 2.0`, i.e. ≥ +100% expected). [reuses incumbent gate]
4. **B7-P4 says event×pick correlation is untaxed** (or only partially taxed with residual
   `Cov_eπ` above `τ_cov` after the app's adjustment). If fully taxed → combos never
   dominate on source (i); fall back to source (iv) standalone contracts only.

**Combo dominates the plain fantasy slip** precisely when the levered coupling+pick-edge
surplus `q(k)·A_ct·(Cov_eπ + p_e·Δπ_edge)` exceeds the added event price drag+fee
`q(k)·A_ct·(m − p̂_e) + c_fee·n_ct`. Numeric crossover at `F=$10, k=3, m=0.55`: combo wins
if `Cov_eπ + coupling ≳ (m − p̂_e) + 0.001`, i.e. **any positive coupling ≥ ~1% beats a
fairly-priced event leg.** Threshold table lives in the future `strategies/` module,
locked, not tuned to fit.

---

## B3 — Player × game-line ρ estimation design

### Estimand & why the de-vigged close is the right residual target (justified)

**Estimand:** the correlation between a player's residualized stat and the game-outcome
residual, both taken against the **de-vigged closing line as conditional mean**:

- ATS margin residual: `r_margin = (home_pts − away_pts) − (−spread_close)` — i.e. actual
  margin minus the closing spread (the market's conditional-mean margin).
- Total residual: `r_total = (home_pts + away_pts) − total_close`.
- Own-team implied-total residual: `r_ownIT = team_pts − team_implied_total_close`, where
  `team_implied_total = (total_close + team_spread_close)/2` (**exactly the archive's
  stored `Totals` team-entity convention**, moneylines.py:451-458 / archive team-total =
  `(game_total + team_spread)/2`).

**Justification (not refutation).** The sharp de-vigged close is, by construction, very
close to the true conditional expectation of the game outcome given all public information
at close (§1.1 of `model_improvement_track.md`: "weighted, closing, de-vigged consensus …
sits very close to the true probability"). Subtracting it yields a **mean-zero,
information-orthogonal residual** — the market has already priced schedule, injuries,
pace, weather. What remains is the unforecastable game noise, which is exactly the
component that co-moves with a player's *own* unforecastable residual (both driven by the
same realized game script: blowout → starters rest → unders; shootout → both QBs' yards +
the total co-rise). Correlating **raw** points against a player stat would instead be
dominated by the shared *predictable* level (good offense → high team total AND high player
yards), inflating ρ with a component the app **also** sees and prices. Residualizing
against the close isolates the **residual dependence the app cannot see** — the only part
that is an edge. This mirrors the incumbent's own choice to residualize player stats
against a rolling mean before correlating (`_residualize_gamelog`, correlate.py:481); we
extend the same leak-free discipline to the game side, swapping the rolling-mean anchor for
the **closing-line anchor** (which is a *better* conditional mean than an 8-game rolling
mean, and available per game from the archive). **Verdict: the close is the correct
residual target; use it.**

Caveat to carry: the player-stat side should stay on its existing 8-game rolling-mean
residual (correlate.py:481) for symmetry with the incumbent and because per-player closing
props are not archived for every stat/date; the **cross-correlation** of a rolling-mean
player residual with a closing-line game residual is still a valid dependence estimate (two
mean-zero series), it just mixes two anchor conventions — flag it, don't block on it.

### Pair-type taxonomy

`pair_type = (league, player_market, game_target, scope)` with:
- `player_market` ∈ shipped cells only (PIT/marginal certified),
- `game_target` ∈ {own-team margin (ATS), game total, own-team implied total},
- `scope` = same-game (the player is in that game) — **the only grain that carries
  dependence**; cross-game player×game-line pairs are independent by construction (drop).

Sign conventions must be explicit (an Over player leg vs an Over-total leg is same-sign;
an Over player leg vs a favorite-cover leg depends on which side the player is on) — reuse
the incumbent's Over/Under sign-flip logic (`correlation.py:160` `"vs."` flip precedent).

### Estimator (reuse the incumbent pipeline verbatim, swap the anchor)

Standalone script; **imports** correlate.py helpers, copies nothing:

1. Build per-(team, game) records over the `LOOKBACK_DAYS=300` window
   (`_build_team_game_records`, correlate.py:532 — import).
2. Player residual: `_residualize_gamelog` (correlate.py:481, rolling-8, min-3) — import.
3. Game residual: **new, small** — join each (team, game) to the archived closing line
   (`Archive.get_line`/`get_total`/team-total via `get_team_market`) and subtract, per the
   three `game_target` definitions above.
4. Rank-correlate player-residual × game-residual with overlap counts via the
   `M.T@M` / `X.T@M` / `X.T@X` trick (`_pairwise_spearman_with_overlap`, correlate.py:637)
   — import; it already returns the overlap matrix that **is** the census raw material.
5. Spearman→Gaussian-copula ρ remap `2·sin(π·ρ_S/6)` (correlate.py:867) — the map is
   already the copula parameter (for a Gaussian copula `ρ_S = (6/π)·asin(ρ/2)`), so no
   change of estimand.
6. **Hierarchical Fisher-z EB shrinkage per the copula brief's R3** (not the incumbent's
   shrink-to-zero `_shrink_correlations`): work in `z = atanh(ρ)`, `Var(z) ≈ 1/(n−3)`;
   Level-1 shrink each team toward the pair-type mean `μ_g` with DerSimonian–Laird `τ²_g`;
   Level-2 shrink `μ_g` toward 0 with `N_g/(N_g+N₀)`, `N₀≈200`. Reuse the R3 design
   verbatim — this is the CorShrink (Dey & Stephens 2018) lineage. **Do not** reuse the
   incumbent's linear `n/30` shrink-to-zero: for a lane whose thesis is "the app
   under-taxes correlation," shrinking to zero is a systematic bias *against* the edge (the
   copula brief §Q2 makes this argument at length — same argument applies here).

### Minimum-sample viability / kill gates (mirror R3)

- Per-team ρ usable at any `n_t ≥ 10` (shrinks hard toward `μ_g`); `n_t ≥ 30` = "mostly own
  data" (matches `MIN_OVERLAP_FOR_FULL_WEIGHT=30`).
- Pair-type prior trustworthy at pooled `N_g ≥ 300` (SE(ρ̄) ≈ 0.06 — tight enough to beat a
  zero target).
- **Stage-0 kill gate for the ρ sub-component:** a league is player×game-line-viable iff
  **≥ 8 fit-eligible pair-types reach `N_g ≥ 300`** (lower than the copula brief's ≥15
  because there are only 3 game-targets × a handful of shipped player markets, so the
  pair-type universe is smaller). If **no league** qualifies, ship the engine with
  **ρ_eπ ≡ 0** (independent event leg) — the combo still prices via the marginal edge
  (source ii/iv), just without the correlation surplus (source i). This is a *degrade*, not
  a lane kill.

### Standalone-script design & migration path

- `scripts/estimate_game_line_corr.py` (read-only; click CLI; tqdm over leagues) →
  writes `data/leagues/{LEAGUE}/player_gameline_corr.parquet`
  (columns: `league, scope, player_market, game_target, side_sign, rho_shrunk, n_pair_obs,
  n_teams_ge_10, N_g, window_start, window_end`).
- **Census reuse (do not build a second census):** the overlap/`N_g` counting is the
  **same computation** the copula brief's `scripts/census_parlay_pairs.py` performs (design
  in `/tmp/researcher_copula_stage0.md` §Q3). Add game-target columns as a `--source
  gameline` mode of that census rather than a new tool — one census, two pair universes.
  (Reuse pointer per the dispatch.)
- **Migration into `correlate.py`:** `correlate.py` is file-conflict-gated (roadmap §5.1:
  `sleeper-parity` **before** `parlay-dependence`, both rebuild `correlate.py`/
  `correlation.py`; roadmap D3). So v1 stays standalone. Post-`sleeper-parity`, fold the
  game-residual anchor and the game-target columns into `_build_team_game_records` /
  `_TRACKED_STATS` as a new "game" role, and let the produced parquet feed
  `prediction/correlation.py`'s `_build_game_corr_map` as additional `_OPP_`-free
  team-level rows. The parquet schema above is chosen to slot into the existing
  `corr_same_team`/`corr_opposing` stratification without a re-fit. **No rework** because
  the estimator IS the incumbent estimator (steps 1-6 import correlate.py); migration is
  moving the call site, not rewriting the math.

---

## B4 — Marginals for the event leg

**Two prices, two roles:**

- **Truth estimate** `p̂_e` = the **Odds-API de-vigged consensus** already in the archive
  (`get_moneyline`/`get_total`; spread recoverable from team-totals). By §1.1 this is our
  best truth proxy; use it as `p_event` in the EV algebra and in ρ residualization.
- **Tradeable price** `m` = the **CFTC contract market price** on UD Predict / Sleeper
  Markets (`$0-$1`, ≈ implied prob). This is what you actually pay; use it in `A_ct`.

**When to trust which:** always price EV with `p̂_e` (truth) but stake at `m` (cost). The
gap drives both the fee-adjusted contract edge (B2-iv) and the alert (below).

**Fee / spread adjustment on the exchange price.** The contract has a bid-ask spread and a
per-contract fee. Effective buy cost = `m_ask + c_fee` (Sleeper: `+$0.02` total = $0.01
Sleeper + $0.01 Kalshi; UD: CONDITIONAL — likely bundled into price, capture P6). Compare
`p̂_e` against the **fee-inflated ask**, not the mid: buy signal iff `p̂_e > m_ask + c_fee +
τ_edge`. Selling/cash-out uses `m_bid − c_fee`.

**De-vig method.** Use the repo's `no_vig_odds` (distributions.py:95). For **lopsided
moneylines** (heavy favorite, `|p_over − 0.5| > 0.3` = `_DEVIG_LOPSIDED_FLAG`) prefer the
**power (logarithmic) de-vig** (`method="power"`, Clarke, Kovalchik & Ingram 2017 — corrects
favourite-longshot bias) over the proportional default. **Repo caveat to flag:** the
current game-line archive path (`_parse_market_books`, moneylines.py:405/414/419) calls
`no_vig_odds` with the **proportional default** — so archived favorite moneylines carry
favourite-longshot bias. For the event-marginal comparison, re-de-vig from raw two-sided
prices with `method="power"` rather than trusting the archived proportional `ev` for
lopsided lines. Totals/spreads near pick'em are unaffected (identical at even money).

**3-way vs 2-way de-vig (soccer / draw-able markets).** Moneyline/spread/total for the US
big-4 (NFL/NBA/MLB/NHL — all Sleeper Markets + UD leagues found) are **2-way** (no draw;
spread/total have a push, not a third traded outcome, and the push is handled in B5, not in
the de-vig). **Only soccer 3-way moneylines** need 3-way de-vig (`p_home + p_draw + p_away`
normalized). Since the current UD/Sleeper team-pick universe is US big-4, **2-way de-vig
suffices at v1**; add 3-way only if soccer team picks appear (capture P8/state). The
archive's `no_vig_odds` is 2-way only — a 3-way helper would be new, deferred.

**Divergence as edge/alert signal — numeric trigger.** Define `d = p̂_e − (m_ask + c_fee)`.

- `d ≥ +0.03` (3¢): **contract underpriced vs sharp** → standalone event-contract buy
  candidate AND a strong combo event leg. (3¢ ≈ well outside typical de-vig noise + a
  Sleeper tick + fee; below it the "edge" is inside consensus/fee slop.)
- `|d| < 0.03`: fairly priced; attach only if coupling (B3) justifies (B2 rule).
- `d ≤ −0.03`: contract **richer** than sharp → reject event leg; do not chase.

This trigger doubles as the dashboard "game line whose book-implied prob diverges from our
weighted consensus is a candidate strong leg" surface already sketched in
`dashboard-ux.md` L3 (:354-357).

---

## B5 — Mixed-slip joint pricing through the incumbent machinery

**Composition (confirmed import-only).** The parlay scorer prices `P(all legs hit)` as
`multivariate_normal.cdf(norm.ppf(p), 0, SIG)` (parlay.py:352, analytical) or via the
push-aware MC `_expected_payout_with_pushes` (parlay.py:219). The scoring struct
(`GameArrays`/`GameScoringContext`, parlay.py:19/38) carries `C` (corr matrix), `M`
(boosts), `p_model`, `p_books`, `p_push`, `boosts`. **An event leg is one more entry in
each:** `p_model[i] = p_e` (chosen side), a ρ column `C[i, j] = ρ_{e,j}` from B3
(0 for cross-game / unshipped), `p_push[i]` per below, `boosts[i] = 1` (contracts carry no
UD boost multiplier). The augmented `SIG` goes through `_psd_or_none`/`_nearest_psd`
(parlay.py:330/198) unchanged. **Zero edits to parlay.py/correlation.py**: a new module
builds the augmented arrays and *calls* `beam_search_parlays` / the `_parlay_payout_prob`
math, exactly as `pickem-build` does today.

**What breaks: event legs CAN push, and the incumbent silently says they can't.** A spread
leg pushes when the margin lands exactly on an integer spread; a total leg pushes when the
total lands exactly on an integer. But `get_push_prob` (distributions.py:411) **returns 0
for `dist="Normal"`** (continuous families have zero point mass) — and game lines are
priced as `_GAME_LINE_DIST="Normal"` (moneylines.py:102). So if you naïvely feed a Normal
event leg into the scorer, `p_push=0` and the analytical `mvn.cdf` path runs, **treating a
push as a loss** — which mis-scores the ~2-9% of NFL/NBA games that land on a key number.

**Cheapest correct treatment (no parlay.py edit).** The push-MC path
(`_expected_payout_with_pushes`, parlay.py:219) *already* implements the exact 3-band
LOSS/PUSH/WIN classifier a spread/total push needs — it splits the standard normal into
`(p_lose, p_push, p_win)` bands via inverse-CDF cuts and drops a pushed leg per UD rules. It
just needs a **nonzero `p_push` for the event leg**, which the incumbent won't compute for
Normal. So the new module computes the event-leg push probability itself and passes it in:

- **Preferred:** derive `p_push` from the **book**: `p_push ≈ p_over(line−0.5) −
  p_over(line+0.5)` using the archived de-vigged prices at adjacent integer/half lines (the
  mass the book itself assigns to the key number). Requires half-point alt prices — often
  archived; if not, capture at Stage-0. This is model-free and matches the book's own push
  belief.
- **Fallback:** a discrete lattice — model the margin/total as a discrete distribution
  (e.g. Normal-rounded, or Skellam for margin in low-scoring leagues) and take the point
  mass at the integer line. `get_push_prob` already supports `Poisson`/`NegBin` point
  masses; a small local helper for a rounded-Normal or Skellam point mass is the only new
  math, and it lives in the new module, not in `distributions.py`.
- Then call the existing MC path: `_parlay_payout_prob` (parlay.py:339) auto-routes to
  `_expected_payout_with_pushes` when `any(push_legs > _PUSH_PROB_FLOOR=1e-6)` — so simply
  supplying a nonzero event `p_push` flips it onto the correct path with **no code change**.

**Moneyline legs don't push** (win/loss only) → `p_push=0`, analytical path, no issue.
**The 3-way outcome for spread/total is (win, push, loss)** — the incumbent MC already
models exactly this; there is no need for a separate 3-way machinery. Estimated cost of
getting push wrong (for the ledger): ~0.02-0.09 probability mass on the wrong side of a
single event leg near a key number → a few % EV error on that slip; the fix is cheap and
already-built, so do it at v1.

---

## B6 — Cash-out: value or ignore at v1

**Recommendation: IGNORE option value at v1; treat cash-out as European (intrinsic value
only).** Rationale:

- Cash-out value = current market price (verified: UD "cash-out value is based on the
  current market price, may be higher or lower"; Sleeper "sell at any time"). With
  near-sharp contract pricing, the market price ≈ the updated true probability, so the
  cash-out is (approximately) a **fair** exit — its *expected* value equals holding, minus
  the round-trip spread+fee. There is **no systematic EV** in the option itself; its value
  is variance management (lock a win, cap a loss), which is a bankroll/Kelly concern, not a
  pricing concern.
- Pricing the American-style early-exercise option correctly requires modeling the
  contract's price *path* (a stochastic-process / optimal-stopping problem) — large build,
  no EV upside given fair exits. **Not worth it at v1.**
- **Combo interaction (the one real subtlety):** the roll mechanic pays the combo if you
  *cash out* the prediction and the fantasy legs win [P3]. So cash-out on the event leg is a
  *feature* that de-risks the event contribution while keeping fantasy upside — but that is
  a **staking/exit policy**, priced as: value at exit = realized `A_ct'` (cash-out proceeds)
  fed into the same B1 stake roll. The engine does not need to *optimize* the exit at v1; it
  prices the *entry* EV assuming hold-to-settle, which is the conservative (lower) bound
  since a fair cash-out doesn't add EV.

**What's left on the table (quantified):** for a near-sharp market, the value of optimal
early exercise over European is bounded by the expected favorable price excursion net of
spread+fee. Empirically for short-horizon sports contracts this is small (< a few % of
stake) and is pure variance reduction, not EV. **Defensible to ignore at v1; revisit only
if Stage-0 shows wide bid-ask (a fat spread makes the timing of exit matter for realized
cost, not for EV).**

---

## Compute-budget section (HARD CONSTRAINT: prophecize few-min typical / 15-min MAX)

**Slate scale (heavy day):** ~10-15 games × 3 markets (ML/spread/total) × 2 sides ≈
**60-90 event contracts** to score; player offers already scored ≈ **hundreds**
(existing prophecize load).

**Costs, per stage:**

1. **Event-leg scoring** (book-implied `p_e` per offered contract + divergence `d`):
   `O(N_event)` = 60-90 `get_ev`(Normal) inversions + 60-90 archive lookups. Each is a
   scalar CDF/inverse — **vectorizable** into one `numpy` call over the 90-row frame.
   Wall-time: **milliseconds.** Archive lookups: batch via `get_team_market_map`
   (archive.py:484, bulk `{(date,entity): ev}` map) — one query per (league, market), not
   per contract. **~tens of ms total.**
2. **ρ lookups** (B3): a **precomputed parquet** keyed by `(league, player_market,
   game_target, team)`; at prophecize time it is a dict/`merge` lookup — `O(N_pairs)`,
   no fitting on the serve path. The EB fit runs **offline** in the standalone script
   (weekly, alongside `meditate`/`correlate`), never in prophecize. **~ms.**
3. **Combo candidate enumeration** — *the cost driver.* Naïve: every subset of
   {event legs} × {player offers} is combinatorial. **Bound it with a two-stage filter
   (specified below).** After filtering, each surviving combo needs one augmented
   `mvn.cdf` (≤ ~9 dims: 1 event + ≤8 fantasy). An `mvn.cdf` at dim ≤9 is ~0.1-1 ms
   (or the 50k-sample MC path ~a few ms if pushes present). With the filter capping
   survivors at `_BEAM_WIDTH=1000`-scale per game (the incumbent's own beam budget), total
   combos ≈ `O(games × BEAM_WIDTH)` ≈ 10-15k `mvn.cdf` calls worst case ⟹
   **~1-15 s wall** (analytical) or **~30-60 s** if a large fraction hit the push-MC path.
   **Inside the 15-min budget with wide margin;** even the MC-heavy worst case is < 1 min.

**Two-stage combo filter (pruning to fit budget):**

- **Stage-1 (cheap, marginal-only, no copula):** for each event leg passing B4
  (`d ≥ −τ_price`), pair it only with **same-game** player offers whose marginal edge is
  positive AND whose B3 coupling sign with that event target is **positive** (reject
  negative-coupling pairs outright — they can't help a combo). This reuses the incumbent
  `_parlay_admissible` team-coverage gate (parlay.py:317) and a sign check on the
  precomputed ρ — `O(N_event × player_offers_same_game)`, a few thousand cheap ops.
- **Stage-2 (copula, only on survivors):** run the augmented `mvn.cdf`/push-MC EV
  (§B1/B5) only on Stage-1 survivors that also clear the marginal geometric-mean floor
  (`_PARLAY_GEO_MEAN_FLOOR=1.05`, parlay.py:69) and the `_MODEL_EV_PRECHECK_FLOOR=1.5`
  pre-check (parlay.py:77) — i.e. **reuse the incumbent beam-search pre-checks verbatim** so
  the expensive copula eval only fires on candidates already likely to ship. This is
  exactly how `beam_search_parlays` already prunes (geo-mean beam, EV pre-check before the
  full copula) — the combo path inherits it by construction.

**Caching:** the ρ parquet (weekly), the `get_team_market_map` bulk EV pull (once per
run), and the event `p_e`/`d` frame (once per run) are all computed once and reused across
every combo candidate. No per-candidate archive or model call.

---

## Stage-acceptance gates + kill rule

**Stage-0 (this brief → payload capture):** GO to Stage-1 (build the standalone ρ script +
event ingestion) iff the B7 checklist is captured AND B7-P4 (event×pick correlation
repricing) returns "untaxed" or "partially taxed with residual coupling ≥ τ_cov". If P4
returns "fully taxed," skip the combo engine; build only the **standalone
contract-divergence alerter** (B2-iv, B4) and mark the combo sub-lane DONE(no-ship).

**Stage-1 (ρ estimation viability):** GO iff ≥1 league has ≥8 fit-eligible pair-types at
`N_g ≥ 300` (B3 gate). Else ship the engine with `ρ_eπ ≡ 0` (independent-leg degrade;
combos still price on marginal edge) — not a kill.

**Stage-2 (offline combo calibration):** before serving combos, an offline A/B on archived
closes must show the augmented-`mvn.cdf` combo `P11` is calibrated (predicted vs empirical
joint hit rate within ±2% absolute on a held-out slate sample; reuse the
`audit_parlay_calibration.py` `gap_copula`/`gap_indep` split precedent from the copula
brief). Combo EV must beat the plain-fantasy-slip EV on the same legs with a
block-bootstrap 95% CI excluding 0.

### Kill rule (B8) — measurable conditions to close the sub-lane no-ship

Close the game-line-combo sub-lane (no combo product; keep at most the divergence alerter)
if **any**:

1. **App prices the event×pick correlation** (B7-P4 = "fully taxed"): the combo repricer
   applies a Correlated-Projections-style multiplier adjustment to the event leg such that
   the realized `Cov_eπ` surplus is removed. *Measure:* capture the combo's quoted payout
   with vs without a known-correlated event leg; if the multiplier moves to neutralize the
   coupling (within τ_cov), killed.
2. **Pairing rules block correlated combos** (B7-P4/P7): if UD/Sleeper forbid same-game
   event+player pairing (the Correlated-Projections "Market Maker may decline" discretion
   extended to combos), the correlated stack can't be built. *Measure:* attempt to
   construct a same-game event+player combo in the payload; if systematically rejected,
   killed for source (i).
3. **Fees eat the edge:** if `c_fee` (+ any combo-specific rake) exceeds the median
   coupling+divergence edge across viable pair-types. *Measure:* `median(q(k)·A_ct·(Cov_eπ +
   d⁺)) < median fee drag` → killed.
4. **Contract prices strictly sharper than our consensus:** if the divergence `d = p̂_e −
   (m_ask + c_fee)` has mean ≤ 0 AND its positive tail never clears +0.03 across a full
   captured slate (the CFTC contract is a *better* truth proxy than our Odds-API consensus).
   *Measure:* full-slate `P(d ≥ 0.03) ≈ 0` and `E[d] ≤ 0` → no standalone contract edge; if
   (1) also holds, the whole sub-lane is dead → DONE(no-ship).

If all four are false on the captured payload, the lane has a defensible edge and proceeds
to build.

---

## CONDITIONAL / VERIFY register (keyed to B7) — the de-conditionalization map

Every product-mechanic assumption above is CONDITIONAL until the matching capture. This is
the map from "what we assumed" → "what to capture" → "which answer it flips."

| Key | CONDITIONAL assumption (as used above) | Capture at Stage-0 | Flips |
|---|---|---|---|
| **P1** | Combo = 1 prediction + 2-8 fantasy picks | UD combo builder: min/max fantasy legs, whether >1 prediction leg allowed in a combo | B1 `k` range; B5 event-leg count |
| **P2** | Fee split 0.9/0.1; contracts integer-lot at `m+c_fee` | Place a combo; read the itemized fee allocation + how many contracts the 90% buys + rounding | B1 `s_fan`,`A_ct`,`n_ct`; all EV magnitudes |
| **P3** | Roll = `s_fan + A_ct` into fantasy stake; partial floor on event-loss; cash-out still pays combo | UD Combo Rules payout worked example; settle a combo in each branch; cash-out-then-fantasy-win outcome | B1 stake definition, the `s_fan·p_π` floor term, B6 combo/cash-out branch |
| **P4** | **Event×pick correlation is UNTAXED** (thesis) | Quote a combo with a known-correlated event leg vs uncorrelated; compare the fantasy multiplier / whether MM declines | **B2 source (i)** entirely; B8 kill-1/2; whole lane GO/NO-GO |
| **P5** | Combo fantasy side = power (no Flex) | UD Combo Rules: Flex availability on combos | B1 `q(k)` vs Flex miss-tier curve |
| **P6** | Contract `$0-$1`, `c_fee` (Sleeper $0.02; UD bundled?), tick size | Read a contract's price format, tick, and explicit fee on UD Predict + Sleeper | B4 fee-adjust, `A_ct` denom, B5 book-derived `p_push` from adjacent ticks |
| **P7** | Cash-out = market price, timing rules (UD multi-team: only pre-start) | UD/Sleeper cash-out terms + timing per product | B6 European treatment; B1 cash-out branch |
| **P8** | US big-4 only ⟹ 2-way de-vig; multi-team predictions same-sport/cross-sport, **not same game** | Enumerate available leagues + multi-team same-game rule + any soccer 3-way | B4 2-way vs 3-way de-vig; multi-team event-leg ρ (same-game blocked ⟹ ρ=0 across event legs) |
| **P9** | State availability (CFTC contracts geo-gated) | Which states have Predict / Sleeper Markets live | whether the product is reachable at all for the operator |

**B7 capture checklist (exact fields/rules to pull from the payloads):**

- **UD Prediction Picks payload:** contract `id`, `market_type` (moneyline/spread/total),
  `team`/`game` ids, **contract price (`m`)**, tick, bid/ask if exposed, fee line,
  settle-value rule, cash-out availability + timing, state gating. (Note: `books.py`
  `_ud_match_ids` at :70 already ingests "team games and solo (combo) games"; `_parse_ud_line`
  at :157 already returns `(None, market)` for combo/`"+"`-name legs — **the team/combo
  structures are already arriving in the UD payload and being dropped**; capture = stop
  dropping them and log the raw fields.)
- **UD Combo Entries:** the **fee-split line** (P2), the **payout worked example** (P3),
  Flex availability (P5), min fantasy legs (P1), **whether a correlated event+player combo
  reprices or is declined** (P4 — the load-bearing capture), same-game event+player
  allowed?
- **UD Multi-team Prediction Picks:** per-position prices, product-of-prices payout,
  same-game prohibition, DNP position-adjustment rule, cash-out pre-start-only.
- **Sleeper Markets:** contract price, **$0.02 fee** (confirmed; re-verify), bid/ask,
  sell-early terms, league coverage, whether Sleeper offers any *combo-with-fantasy*
  analogue (web shows Sleeper Markets as standalone team-pick trading — capture whether a
  fantasy pairing exists at all; if not, Sleeper is a **standalone contract / divergence**
  play only, no combo engine).
- **Classic fixed-multiplier pick'em (both apps):** **confirmed cannot contain team picks**
  ("Multileg entries with team moneyline, spread, and total options aren't available on
  Underdog Fantasy … limited to player prop-based picks only") — capture whether any state
  differs; default assumption: team outcomes enter **only** via Prediction Picks/Combos
  (UD) and standalone Markets (Sleeper).

---

## Reality checks

- **The edge is one capture away from zero.** Everything hinges on B7-P4. Web evidence that
  UD reprices player×player correlation (Correlated Projections) is a *warning shot*: an app
  sophisticated enough to tax player×player correlation may extend it to event×pick in
  combos. The architectural separation (CFTC contract priced by Kalshi/CDNA; fantasy priced
  by UD's market maker) is the reason to *hope* it doesn't — but hope is not evidence. **Do
  not build the combo engine before P4 is captured.**
- **This is an engineering project on the mechanics, a research project on the edge.** The
  pricing algebra (B1), the MVN composition (B5), and the ρ estimator (B3) are all *known
  methods with quantified build cost* (reuse correlate.py + parlay.py, one standalone
  script, one ingestion path). The *edge* (B2-i) is *unproven and may not transfer* — it is
  a bet that a specific product seam is unpriced. Separate them in the ledger.
- **ρ magnitudes will be modest.** Residualizing against a sharp close removes the shared
  predictable component, so `Cov_eπ` post-shrinkage will likely sit in the 0.02-0.15 band
  (comparable to the copula brief's 0.05-0.4 player×player range, but *lower* because the
  game side is anchored to a better conditional mean). The edge per combo is real but
  **small per entry**; it compounds across volume, like every edge in this repo.
- **Two anchor conventions mixed.** The player residual uses an 8-game rolling mean; the
  game residual uses the closing line. Both are mean-zero, so the cross-correlation is
  valid, but it is not a pure single-anchor estimate — flagged, not blocking.
- **Push mis-scoring is a silent trap.** The single most likely v1 bug is feeding a Normal
  event leg into the incumbent scorer and getting `p_push=0` → key-number games mis-priced.
  Called out in B5 with the cheap fix; a weaker implementer must not skip it.

## Open questions / caveats (carry into the lane brief's Open-questions)

1. **B7-P4 (event×pick repricing) is unresolved and load-bearing.** No web source states
   whether a Combo Entry taxes the event×fantasy-pick correlation. Must be captured
   empirically. Until then the entire edge is CONDITIONAL.
2. **UD contract fee structure (P6)** unknown from web (Sleeper's $0.02 is confirmed; UD
   Predict fee is described as "net profit = $1 − price − applicable fees" without a
   number). Flip `A_ct` and B4.
3. **Sleeper combo analogue.** Web shows Sleeper Markets as standalone team-pick trading;
   whether Sleeper offers a fantasy-pairing combo at all is unconfirmed. If not, Sleeper is
   a divergence/standalone play only.
4. **Closing spread not stored as its own archive market.** Only Moneyline + Totals
   (team-total = `(total+spread)/2`) are in the `odds`/`lines` tables (archive.py schema,
   moneylines.py:451-458). The ATS margin residual must reconstruct the closing spread from
   the two team-totals (`spread = 2·team_total − total`) or re-pull from the Odds API.
   Verify the reconstruction is exact per league before trusting `r_margin`.
5. **Book-derived `p_push` needs adjacent-line prices** (half-point alt lines). If not
   archived, either capture them or fall back to the lattice (Skellam/rounded-Normal) point
   mass. Confirm alt-line availability per league at Stage-0.
6. **Power-de-vig for archived favorites.** The archive stored proportional de-vig for
   lopsided moneylines; the event-marginal comparison should re-de-vig with `method="power"`.
   Confirm this doesn't require an archive backfill (it doesn't — re-de-vig from raw prices
   at compare time).

## Sources

| Source | Identifier / URL | Used for |
|---|---|---|
| Underdog — Prediction Picks Overview | help.underdogsports.com/en/articles/14127552 ; help.underdogfantasy.com/en/articles/12312909 (301→sports) | contract pricing $0-$1, CFTC/NFA (Kalshi/CDNA), cash-out = market price, B1/B4/B6 |
| Underdog — Pick'em Classic Combo Rules | help.underdogsports.com/en/articles/12141877 | combo fee split 90/10, roll-into-stake payout, partial floor, no-Flex, B1 |
| Underdog — Multi-team Prediction Picks | help.underdogsports.com/en/articles/14127564 ; …/13248008 | product-of-prices payout, up-to-4, not-same-game, cash-out pre-start-only, B4/P8 |
| Underdog — Correlated Projections | help.underdogfantasy.com/en/articles/11010091 | player×player correlation IS repriced (pos→lower mult, MM may decline), B2-i/B8-kill |
| Underdog — Pick'em rules (team picks excluded) | app.underdogsports.com/rules/pick-em ; oddsassist.com/dfs/how-underdog-works/ | classic pick'em is player-prop only, no team ML/spread/total; ≥2 teams per entry |
| Sleeper Markets — reviews/overview | bettingusa.com/prediction-markets/reviews/sleeper-markets/ ; fantasylabs.com/articles/sleeper-markets/ ; defirate.com | CFTC/NFA (9 Jan 2026, Kalshi), ML/spread/total NFL-NBA-MLB-NHL, $0.02 fee, sell-early, B4 |
| Clarke, Kovalchik & Ingram (2017) | power/logarithmic de-vig, favourite-longshot bias | referenced in `no_vig_odds`; B4 lopsided-moneyline de-vig |
| Dey & Stephens (2018), *CorShrink* | EB adaptive Fisher-z correlation shrinkage | B3 hierarchical shrinkage (via copula brief R3) |
| `/tmp/researcher_copula_stage0.md` | in-repo (R3 verdict) | B3 EB design, census reuse, viability-gate conventions |
| DerSimonian & Laird (1986) | method-of-moments between-group variance | B3 level-1 `τ²_g` |

### Repo file:line references (verified this session)

- `moneylines.py:392-426` `_parse_market_books` — de-vig via `no_vig_odds`, per-book ML/total/spread; `:405/:414/:419` proportional de-vig default (favorite bias flag, B4).
- `moneylines.py:451-458` team-total = `(game_total + team_spread)/2` (B3 own-team implied-total anchor).
- `moneylines.py:102` `_GAME_LINE_DIST = "Normal"` (B5 zero-push trap).
- `helpers/archive.py:96-114` `odds`/`lines` schema `(league,market,game_date,entity,book,ev/line)` + `under_prob` (B3 residual targets).
- `helpers/archive.py:449-482` `get_team_market`/`get_moneyline`/`get_total` (event marginals, B4).
- `helpers/archive.py:484` `get_team_market_map` bulk EV map (compute budget — batch lookup).
- `helpers/distributions.py:95-121` `no_vig_odds` (proportional/power de-vig, `_DEVIG_LOPSIDED_FLAG=0.3`, B4).
- `helpers/distributions.py:274` `get_ev(..., dist="Normal")` (event marginal inversion, B4).
- `helpers/distributions.py:411-459` `get_push_prob` — returns 0 for Normal (B5 fix target).
- `training/correlate.py:40/45/49/54/58` window/rolling/shrink constants; `:481` `_residualize_gamelog`; `:532` `_build_team_game_records`; `:637` `_pairwise_spearman_with_overlap`; `:698` `_shrink_correlations` (shrink-to-zero — replace with EB); `:867` `2·sin(π·ρ_S/6)` remap (B3 reuse targets).
- `prediction/parlay.py:19/38` `GameArrays`/`GameScoringContext` (augment with event leg); `:198` `_nearest_psd`; `:219-290` `_expected_payout_with_pushes` (3-band push MC, B5); `:317` `_parlay_admissible` (team coverage); `:339-352` `_parlay_payout_prob` (auto-routes analytical vs push-MC); `:352` analytical `mvn.cdf` (B5 import-only).
- `books.py:70` `_ud_match_ids` ("team games and solo (combo) games"); `:157` `_parse_ud_line` returns `(None, market)` for combo legs — **combo/team structures already arrive, currently dropped** (B7 capture = log them).
- `data/config/stat_map.json` — no team-market entries (all "Total*" are player totals) → new ingestion needed.
- `data/config/underdog_payouts.json` — `power {2:3,3:6,4:10,5:20,6:25}` (B1 `q(k)`, combo = power/no-Flex).
- `docs/handoffs/model_improvement_track.md §1.1` — sharp de-vigged consensus ≈ truth; LOCKED: book-implied marginals for game lines, no team-market model (B3/B4 premise).
- `docs/sportstradamus_roadmap_v3.md §5.1` — `sleeper-parity` before `parlay-dependence`, both rebuild `correlate.py`/`correlation.py`; D3 gate (B3 file-conflict constraint → standalone v1).
- `docs/handoffs/dashboard-ux.md:354-357` L3 — game lines as book-implied correlation anchors + divergence = strong-leg candidate (B4 alert surface).

---

## 10-line verdict summary (print this on reply)

1. PROCEED to Stage-0 payload capture; do NOT build the combo engine yet — the entire edge hinges on one unverified fact (B7-P4).
2. B7-P4 = "does the Combo Entry reprice event×fantasy-pick correlation?" UD reprices player×player (Correlated Projections); the combo's separately-priced CFTC event leg plausibly escapes that tax — capture empirically.
3. B1 combo EV (closed form): EV_net = q(k)·s_fan·p_π + q(k)·A_ct·P11 − F, with A_ct=0.9F/(m+fee), P11=p_e·p_π+Cov_eπ; the ρ surplus is levered by A_ct (~17× at m=0.55).
4. B2 edge = mostly (i) untaxed event×pick correlation [gated] + (iv) contract-vs-sharp divergence [measurable now]; combo dominates plain slip when any positive coupling ≥ ~1% beats a fairly-priced event leg.
5. B3 ρ estimand = residualized player stat × game residual vs the de-vigged close (close = conditional mean → correct target, justified); standalone script reusing correlate.py verbatim + hierarchical Fisher-z EB per R3; gate ≥8 pair-types at N≥300/league; migrate into correlate.py after sleeper-parity.
6. B4 = trust Odds-API de-vigged consensus as truth, stake at CFTC contract price; power-de-vig lopsided moneylines; divergence d=p̂−(ask+fee)≥+0.03 = strong-leg/alert; 2-way de-vig suffices (US big-4).
7. B5 = event leg composes as one more norm.ppf(p) cut with a ρ column, import-only through parlay.py's MVN + push-MC — ZERO edits; the one trap: spread/total legs CAN push but get_push_prob returns 0 for Normal → feed a book-derived p_push into the existing 3-band MC.
8. B6 = ignore cash-out option value at v1 (European/intrinsic); near-sharp pricing makes exits fair → no EV, only variance management; small residual, revisit only on wide bid-ask.
9. Compute budget: event scoring O(90) vectorized ~ms; ρ from precomputed parquet ~ms; combo enumeration is the driver — bounded by a two-stage filter (marginal+sign prune, then copula only on beam survivors) to ~1-60s worst case, well inside 15-min.
10. Kill rule (B8): close no-ship if the app prices event×pick correlation OR pairing rules block correlated combos OR fees exceed median edge OR contract prices are strictly sharper than our consensus (P(d≥0.03)≈0 and E[d]≤0). Reuse census_parlay_pairs.py (add a --source gameline mode); do NOT build a second census.
