# Lane brief — story balance: an honest ledger by bet side, Over-led and Under-led star stories

**Read first:** CLAUDE.md, [docs/ARCHITECTURE.md](../ARCHITECTURE.md),
[docs/story_voice.md](../story_voice.md), [docs/dashboard_ux_redesign.md](../dashboard_ux_redesign.md)
(Tonight and Games rows), [docs/ship_gate.md](../ship_gate.md) (Gate 2), and the research brief
[docs/archive/researcher_selection_shrink.md](../archive/researcher_selection_shrink.md).
Status: OPEN — every workstream landed on `devel` 2026-10-03, unpushed; the first live read waits on
the next `prophecize` and `reflect` after the push (§7).

## 1. Mission & money logic

The Games and Tonight tabs read as "this game will be exciting, but take the unders anyway". The
owner asked, in order: are the under-heavy recommendations actually +EV; if not, fix the models;
either way give Overs and Unders near-equal representation with the stars in front, in every
league, without giving up profitability.

The first question has an answer, and it is not the one the dashboard was showing. Priced at the
platforms' real payouts, **neither side of the recommended cohort is realized +EV**:

| recommended legs (Model EV ≥ 1.05) | n | hit | model read | book read | flat ROI |
|---|---|---|---|---|---|
| 90d Over (2026-07-04 → 10-02) | 5,656 | 0.470 | 0.663 | 0.483 | −14.8% |
| 90d Under | 11,237 | 0.504 | 0.642 | 0.492 | −8.8% |
| 30d Over | — | 0.443 | 0.609 | 0.483 | — |
| 30d Under | — | 0.495 | 0.609 | 0.492 | — |

Unconditional calibration is exact (mean model P(Over) 0.442, realized 0.442, book 0.445). The loss
is overconfidence conditional on selection: on the legs the rule picks, the sportsbook consensus
beats the model by 2–8 pp, and the hit rate falls as the claimed edge over the market grows. Legs
paying above 2.5× hit 0.15–0.35. The under skew itself is structural (DFS x.5 lines on low-count
stats, argmax-probability side selection, right-skewed distributions, Kelly ranking that favours
high-probability low-payout legs, Over stories needing unanimous Over legs), not a bias of the
distributions.

Four reporting bugs had hidden all of that from every nightly number the owner reads: Kelly went
positive on −EV legs whose decimal payout sat below 1; the profit sims priced Unders as `boost/(1 − p)`
although `Market Prob` is the chosen side's probability (MLB home runs showed a +615% Kelly yield);
CLV carried an extra Under sign flip (390k stored rows wrong-signed); Gate 2 demoted on Over
precision only while 52% of Under history rows were never posted on the platform yet counted. All
four are fixed (§4). The model lever the brief would have justified was researched and **killed** (§6).
So the lane's money logic is: stop lying to the owner, stop ranking −EV and deep-alt legs first, and
let the stories balance by construction at the unchanged 0.05 edge floor rather than by pretending
the Overs are better than they are.

## 2. Locked decisions (owner)

- Model scope: diagnostic + guardrails now; the selection lever research-gated (it was, and it died).
- Balance: slate-level Over/Under alternation of the lead story at the unchanged `_MENU_EDGE_FLOOR`
  (0.05); the top-EV story stays in the menu, often behind the led ones (median cost ≈ 0.05 EV on the
  first slate; accepted).
- Star: computed prominence from the snapshot alone (`Star` = rank-1 depth label + line percentile +
  market breadth, in [0, 3]); no new data; **no MLB depth term** (B1..B9 is the batting slot, not
  usage), so MLB stars top out at 2.0 on line percentile + breadth.
- Voice: the "live analyst" register (`docs/story_voice.md`), pinned by density tests, not strings.
- Gate 2 symmetric: either recommended side's live precision below 0.50 demotes (see §8 before you
  push; two mandatory NFL cells demote on the dev-box read).
- Orphan lead seeds pair with the top-edge strong leg from the other team (any other player in a
  one-sided game); the dek drops its correlation line below the ρ floor, so no false "move together".
- Both sides' full payouts persist to history (`Payout Over`, `Payout Under`) for the argmax-EV
  backtest the research brief specifies.

## 3. Verify before you trust

```bash
git log --oneline devel -24 | grep -E "stories|realized|nightly|clv|training|history|prediction|dashboard"
ls -la src/sportstradamus/data/runtime/ | grep -E "realized_by_side|current_game_stories|current_offers"
poetry run python - <<'EOF'
import pandas as pd
o = pd.read_parquet("src/sportstradamus/data/runtime/current_offers.parquet")
s = pd.read_parquet("src/sportstradamus/data/runtime/current_game_stories.parquet")
print("Star column:", "Star" in o.columns, "| lead columns:", {"lead", "lead_side", "lead_player", "star"} <= set(s.columns))
print("-EV legs with Kelly > 0:", int(((o["Model EV"] < 1) & (o["Kelly"] > 0)).sum()))
EOF
```

Until the first post-push `prophecize`, `current_offers` has no `Star` and the stories no `lead`
columns; the dashboard loaders backfill both (`Star` 0.0, `lead` False), so Tonight and Games keep
rendering on the old snapshots. A `KeyError` on `Star` in Tonight means the loader backfill in
`dashboard/data.py` is not deployed, not that the snapshot is stale.

## 4. What landed (all on `devel`)

| commit | change | where it shows |
|---|---|---|
| `1d0d7ba9` | `Kelly` = `(Model EV − 1)/(Boost − 1)` only for `1 < payout ≤ MAX_FAVORED_PAYOUT` (2.5), else 0; `Model EV` untouched | Board Kelly sort, "favored" counts, constellation, satellite picker, story legs |
| `ad599623` | `Payout Over` / `Payout Under` on every history row (0 when the side is not posted; NaN before 2026-10-03) | `history.parquet` |
| `ed7a32b9` | `realized.py`: `compute_realized_by_side` — long ledger (window × cohort × split × key × side) at platform payouts, `Boost > 0` only | `data/runtime/realized_by_side.parquet` |
| `bd6548d5` | nightly prices every row as `boost / p` (chosen side), counts posted sides only in precision and both profit sims, writes the ledger + one INFO ROI line | `live_metrics_per_market.parquet`, `reflect` log |
| `82c0adb2` | CLV = close − open on bet-side probabilities, no Under flip; `fill_from_archive` recomputes every row with a close (repairs the 390k rows on the next `reflect`) | Receipts beat-close rate, pick'em CLV shrinkage |
| `a6d96fd8` | Gate 2 demotes on either side's precision < `MIN_PRECISION_SIDE` (0.50) | `lifecycle_table`, `gate-status`, Lab › Training |
| `cc2f8942` | Receipts panel "Realized by side — platform payouts" (`components/by_side.py`) | Receipts, after Skeptic checks |
| `6a4b7dc9`, `c9c303a8` | `stories/lead.py`: `attach_prominence` (`Star`), `lead_seeds`, `assign_leads`; `prophecize` scores and persists `Star` | `current_offers.Star`, story `lead*` columns |
| `d37ef588`, `db6e834c`, `0806d11a` | engine takes a `lead`; the menu grows an Over-led and an Under-led story per game (`Kelly > 0` and edge ≥ 0.05 required of every leg, lead in both presets, led stories first, slate-wide headline dedup); pricing helpers moved to `stories/pricing.py` | `current_game_stories` |
| `a3ca2684`, `c12b90af`, `624a45a8` | `game_headline` prefers the lead story; Tonight sorts `(urgent, −star, −favored, minutes)` behind the lead side's arrow; Games opens on the lead story with "Over-led · " / "Under-led · " labels | Tonight, Games |
| `31a70732` | why/dek clauses read the defense (`gives`/`takes`) and the bet (`above_for` … `below_against`); `bank_cell` resolves `mistakes` before any `production` fallback | offer Why text, story deks |
| `6fc2dbd4` … `c1354dfe`, `0450274f` | all five voice banks rewritten to the live-analyst register; `STORIES_VERSION` p5 (every persisted headline reshuffles once; `user_slips` headlines stay frozen); density pins in `test_bank_coverage.py` | every headline |

Untouched on purpose: `strategies/profit_sim.py`, `strategies/_ledger_*.py` (sim-bettor lane),
`analysis._add_kelly_columns` and the flat −110 grading in Receipts — both since fixed by the
honest-receipts lane ([honest-receipts.md](honest-receipts.md): Kelly at the platform payout, the
`record_grid` and flat hero retired).

## 5. How the balance works

`attach_prominence` scores every offer row once per `prophecize`. In the menu, `_strong_legs` keeps
legs with `Model EV − 1 ≥ 0.05` **and** `Kelly > 0` (so no −EV leg and no payout above 2.5× can seed
or join a story). `lead_seeds` picks, per side, the strong leg with the highest `(Star, Model EV)`.
The Over-led cluster grows first, then the Under-led one, then the old greedy clusters; a lead whose
cluster prices nothing borrows one leg and is priced once more. `_best_subsets` only enumerates
subsets containing the lead, so Builder and Moon stay true argmaxes with the lead inside. Led stories
rank first. `assign_leads` then walks each date's games in tip order and alternates the wanted side
(Over first on an even date ordinal), taking the other side only when the wanted one has no led
story; both platforms show the same side whenever both offer it. The Tonight arrow and the Games
default story read that `lead` flag.

## 6. Research verdict: the selection-shrink lever is KILLED

Question put to `research-analyst`: should the served probability of a recommended leg be shrunk
toward the sportsbook consensus by a factor fitted on the realized selected subset (grain, estimator,
placement), against the status quo, a higher floor, `kelly_shrinkage` as λ, unconditional book CITL,
and argmax-EV side selection? Walk-forward on fit < 2026-09-01, eval ≥ 2026-09-01, cohort re-derived
after the shrink. The full brief is archived at
[docs/archive/researcher_selection_shrink.md](../archive/researcher_selection_shrink.md).

- **Every option fails.** Best shrink +1.7 pp ROI (per-cell λ, day-clustered CI [−0.3, +3.4]) against
  a +3 pp bar; Over hit among survivors ≤ 0.475 against 0.50. A hindsight oracle fitted on the eval
  window itself passes nothing.
- **Nothing to tune.** On selected legs the served `Win Prob` carries no information beyond the
  market: λ_sel = −0.10 ± 0.10, logistic slope 0.01 ± 0.15; hit rates track the book in every
  disagreement bin. λ = 0 is "replace the model with the market".
- **64% of recommended legs have no sportsbook quote** (`Market Prob` is the payout-implied fill), so
  a book shrink cannot reach most of the overconfidence. Those legs hit 0.466 / 0.514 against 0.610.
- A **higher floor hurts** at every level: claimed edge is anti-correlated with outcome because a big
  edge is a big payout, which is the side the platform priced as unlikely.
- **MLB is selection on noise** (calibrated, selected hit == book). **NFL is selection plus a model
  bias** (Unders +3.6 pp overconfident on 10,004 legs, λ_all −0.14 ± 0.03): a training problem for the
  NFL lanes, not a serving patch.
- **Payout cap 2.0** is the most consistent single lever (+2.3 pp, CI [+0.3, +3.8]) but its Over hit
  0.498 fails the bar. Its trigger is pre-registered below; the cap stays at 2.5.

If the lever is ever reopened (about 2026-11-01, one book regime, NBA in sample): post-argmax only,
in `finalize_records` after the confidence clip, on real-quote rows,
`Win Prob' = Market Prob + λ_L · (Win Prob − Market Prob)` with λ_L per league keyed on
`Model Version`, through-origin least squares on trailing recommended quoted legs, at least 1,000
legs and a CI lower bound above 0, else λ = 0, never 1.

## 7. First live read (run once after the first post-push `prophecize` and `reflect`)

Runtime parquets only; nothing here touches DuckDB. On the dev box run it after the `sync`
from prod, or on prod as the `sportstradamus` user.

```bash
poetry run python - <<'EOF'
import pandas as pd
from sportstradamus.helpers import platform_payout
R = "src/sportstradamus/data/runtime/"
o = pd.read_parquet(R + "current_offers.parquet")
s = pd.read_parquet(R + "current_game_stories.parquet")
led = s[s["lead"] & (s["objective"] == "moon")]
both = s[s["lead_side"] != ""].groupby(["platform", "Date", "Game"])["lead_side"].nunique()
menus_with_both = both[both == 2].index
over_share = led.set_index(["platform", "Date", "Game"]).loc[led.set_index(["platform", "Date", "Game"]).index.isin(menus_with_both), "lead_side"].eq("Over").mean()
print(f"Over share of lead headlines where both sides exist: {over_share:.2f}  (want 0.40–0.60)")
legs = pd.DataFrame([dict(l, story=r.story_id) for r in s.itertuples() for l in r.legs])
print("story legs below the floor or with Kelly <= 0:", int(((legs["Model EV"] - 1 < 0.05) | (legs["Kelly"] <= 0)).sum()), "(want 0)")
print("-EV offers with Kelly > 0:", int(((o["Model EV"] < 1) & (o["Kelly"] > 0)).sum()), "| payout > 2.5 with Kelly > 0:",
      int(((platform_payout(o["Boost"], o["Platform"]) > 2.5) & (o["Kelly"] > 0)).sum()), "(want 0, 0)")
r = pd.read_parquet(R + "realized_by_side.parquet")
print(r[(r["window_days"] == 30) & (r["cohort"] == "recommended") & (r["split"] == "side")][["side", "n", "hit_rate", "pred_rate", "book_rate", "roi"]].to_string(index=False))
h = pd.read_parquet(R + "history.parquet")
u = h[(h["Bet"] == "Under") & h["Close Market Prob"].notna()]
print("Under rows whose Market CLV == close − open:", f"{((u['Market CLV'] - (u['Close Market Prob'] - u['Market Prob'])).abs() < 1e-9).mean():.3f}", "(want 1.000 after the first reflect)")
EOF
```

Then eyeball Tonight: at least one Over arrow whenever an Over-led story exists, star games above
non-star games within a tier, and the Games tab opening on the led story.

## 8. Behaviour changes you will see (read before pushing)

- **Gate 2 will demote cells on Under precision.** On the dev-box history the symmetric
  rule demotes five cells that pass on Over precision: NFL `rushing yards` (Under precision 0.39,
  Over has too few bets to read), NFL `receiving yards` (0.46), WNBA `PR` (0.43), `RA` (0.44),
  `PA` (0.47). Two of those are mandatory cells of
  [nfl-ship15-recovery.md](nfl-ship15-recovery.md). Demotion changes `lifecycle_state`, the monthly
  `gate-status` PR and Lab › Training; it does not stop serving (serving is pickle-exists). If you want
  the old asymmetric rule back, revert `a6d96fd8`; the precision inputs stay posted-only either way.
- **Board Kelly reads 0** on every leg paying ≤ 1× or > 2.5× (61 MLB Underdog Unders on the first
  slate were −EV with Kelly 3.5–40 and ranked first everywhere). Tooltip updated.
- **Every persisted headline reshuffles once** (`STORIES_VERSION` p5); `user_slips.parquet` keeps its
  frozen text.
- **CLV flips sign on every Under segment** at the next `reflect` (NFL Under beat-close reads ≈ 68%,
  not 31%); pick'em CLV shrinkage follows.
- **Live metrics move**: MLB home runs' Kelly yield leaves the hundreds of percent; flat and Kelly
  yields share a sign on cells with n ≥ 100; `precision_under_live` drops where unposted Unders were
  propping it up.
- `voice_bank.json` lives under `data/config/`, so the production pull restarts the dashboard by
  itself (`run_job.sh`); nothing manual.

## 9. Records, not tasks (left open on purpose)

Monitoring the research brief asked for; nothing is wired, nothing is yours to do now:

- A `quote` split (quoted / unquoted / fallback) in `realized._offers_by_split`, so `book_rate` stops
  pooling sportsbook and payout-implied anchors (that pooling is why the 90d "book" read 0.586
  against 0.504 realized).
- The selection residual per league × side: `hit_rate − book_rate` and a trailing-30d λ_sel with its
  day-clustered CI. Reopen the shrink only when the CI lower bound clears 0.
- Gate 2 cannot see this failure (its precision uses all posted `Bet` rows, ≈ 0.59 / 0.64); the
  realized ledger is the only monitor for it.
- Pre-registered trigger for lowering `MAX_FAVORED_PAYOUT` to 2.0: two consecutive monthly 30d reads
  in which the recommended 2.0–2.5 band's ROI sits ≥ 5 pp below the 1.5–2.0 band on **both** sides
  (today's 30d gap: 6.6 pp Over, 9.0 pp Under; 90d: 5.1 pp Over, 0.2 pp Under). Owner packet, not a
  session flip.

Open questions the data could not settle:

- Honest, market-anchored probabilities recommend about 1% of today's legs. Should the `recommended`
  label and the 0.05 menu floor keep firing on unproven claimed edge? Product call.
- Unquoted legs (64% of the cohort) get full trust (λ = 1); requiring a quote makes ROI worse (−1.6 pp)
  because quoted disagreements are the most adversely selected. Unresolved, and the larger lever by
  count.
- Is a sportsbook alt-line quote at the rung in the archive? It would beat the shape decode as
  `Market Prob` for rung payouts (a WS-4 question).
- Star weights: NBA/WNBA depth labels rank minutes within team × position (A'ja Wilson posts as F3);
  line percentile + breadth carry her anyway. If a true star loses a seed to a role player on the
  first slates, re-weight `depth` in `stories/lead.py` (owner call; not pre-built).
- `legs._stat_category` maps the "Goals Against" display name to `scoring` while the `goalsAgainst`
  slug reads `mistakes`; the leg's `stat` carries the slug, so the slug path fires. Worth a glance
  because mistakes cells are the valence-sensitive ones.
- Re-run the walk-forward around 2026-11-01 within one book regime and with NBA in sample; NFL had no
  fit-window rows in the prescribed split.

## 10. Owner asks (one each)

1. Push `devel` (`git push origin devel`); prod self-deploys on the next job and restarts the
   dashboard.
2. After the first `prophecize` + `reflect`, paste the §7 block into a shell on the box and read the
   six lines; the targets are in the parentheses.
3. Decide on the Gate-2 demotions in §8 (keep the symmetric rule, or revert `a6d96fd8`).
