# Consensus sanity check — is the sportsbook consensus profitable on DFS legs with a real edge?

Record of one measurement for the [honest-receipts lane](../handoffs/honest-receipts.md) (§7), run
2026-10-04 at the owner's request. Verdict: **no**. No model output is used anywhere in it.

## Method

Window: game dates 2026-08-30 to 2026-10-03, the 33 days the archive `ladder` has polled Underdog and
Sleeper. Consensus = the plain mean of each sportsbook's own de-vigged price at the same line, two or
more books (DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Fanatics, BetOnline, Bovada, MyBookie; no
DFS platform), latest quote at or before the platform's last poll of the rung. No shape decode.
Payout = the platform's own rule. An Underdog entry pays table × the legs' multipliers (2-pick 3.5×,
3-pick 6.5×), so a leg is worth multiplier × √3.5 = multiplier × 1.871 inside a 2-pick; entries are
also graded as entries (every same-day pair, every triple). A Sleeper entry pays the product of the
posted multipliers. Edge = consensus × the leg's per-pick value − 1. One leg per player per day, the
side the consensus likes most. CIs are day-clustered bootstraps (2,000 resamples).

## Known answers

Betting every side of every Underdog 1.00× leg returns −6.5 % per pick, which is −12.5 % per 2-pick
entry (3.5 × ¼ − 1). The consensus is calibrated in bulk: on 39,332 lines it says .403 and .397
happen (+0.65 pp [−0.39, +1.78]); within ±1 pp from .30 to .60, about 2 pp too high below .30.

## Result

| kind of leg | platform's cut per pick (every offered side) | consensus-edge legs (edge ≥ 0) | consensus | hit | needs | return per pick | as real 2-pick entries |
|---|---|---|---|---|---|---|---|
| Underdog 1.00× both ways | 6.5 % | 88 | .548 | .545 | .535 | +2.0 % [−19.4, +29.3] | −14.6 % [−45.7, +90.2] |
| Underdog priced, both sides offered | 10.8 % | 205 | .407 | .380 | .386 | −2.6 % [−23.9, +18.0] | with the row above: −2.6 % [−28.3, +32.2] |
| Underdog, one side offered (longshot ladder rungs) | 19.9 % | 1,382 | .134 | .098 | .118 | −21.1 % [−29.0, −13.6] | −43.4 % [−52.0, −36.5] |
| Sleeper, both sides offered | 12.3 % | 66 | .524 | .455 | .504 | −5.6 % [−30.9, +23.5] | −31.6 % [−67.3, +47.7] |
| all kinds, both platforms | 11.5–12.5 % | 1,760 | .204 | .168 | .187 | −17.5 % [−23.7, −11.0] | −37.0 % [−48.1, −24.7] |

A higher floor does not help: edge ≥ 5 % is 1,082 legs at −14.2 % [−22.3, −6.1]; edge ≥ 10 % is 726 at
−18.8 % [−32.0, +1.3]. Under the other Underdog conventions the edge ≥ 0 tier reads −18.1 % (3-pick
root), −19.0 % (the code's 1.78) and −25.1 % (the old file's 3.0×). Three measured reasons:

- **The platforms already price like the books.** On two-sided rungs the platform's price and the
  consensus differ by 1.0 pp (Sleeper) and 1.3 pp (Underdog) on average, correlation .99; they are 5 pp
  or more apart at the same moment on about 5 Sleeper and 7 Underdog rungs a day. Where they differ,
  Sleeper's own price is the informative one (logit weight 1.12 (se .27) against 0.01 (.26) for the
  consensus); on Underdog the consensus is (0.80 (.33) against 0.28 (.32)).
- **Most apparent edges are our own longshot bias.** 79 % of the edge legs are Underdog one-sided
  ladder rungs, where the consensus overstates by 3.6 pp [2.5, 4.8]: sportsbooks quote those alt lines
  one-way, `no_vig_odds` removes a flat 6.52 % from a one-way price, and the consensus there is often
  two books (two books −23.4 %; four or more −7.8 % [−23.7, +9.7]).
- **Stale quotes.** Sportsbooks are polled five times a day, the DFS boards hourly. Edge legs whose
  quote is at most an hour old return −7.9 % [−21.4, +9.9]; one to three hours old, −32.1 %.

## The model's recommended legs, same weeks, by how the platform priced the leg

Rule as it ran, graded at the real table; history's own side and multiplier, nothing reconstructed.

| kind of leg | legs | model read | book read | hit | needs | return per pick | every posted model-scored leg |
|---|---|---|---|---|---|---|---|
| Underdog 1.00× | 1,075 | .622 | .522 | .527 | .535 | −1.5 % [−9.8, +6.4] | −6.1 % |
| Underdog discounted (< 1.00×) | 430 | .735 | .627 | .579 | .636 | −8.9 % [−20.1, +0.8] | −9.9 % |
| Underdog boosted (> 1.00×) | 1,528 | .553 | .422 | .401 | .464 | −13.8 % [−21.5, −4.3] | −10.0 % |
| Sleeper even (1.70–1.85×) | 993 | .622 | .515 | .510 | .563 | −9.5 % [−17.4, −3.0] | −10.8 % |
| Sleeper favorite (< 1.70×) | 835 | .711 | .627 | .590 | .654 | −10.0 % [−15.1, −4.9] | −10.3 % |
| Sleeper longshot (> 1.85×) | 1,239 | .553 | .444 | .436 | .490 | −11.1 % [−16.7, −5.1] | −9.5 % |

On every kind of leg but one the rule returns the platform's cut. On the Underdog 1.00× leg, where the
cut is smallest, it sits at break-even: at the real-table rule 2,153 recommended legs hit .533 against
.490 for the 5,430 it passed (+4.4 pp [−0.4, 7.6]; MLB .514, NFL .542; above break-even in three of
five weeks; Under .554, Over .477). Before the post-fix era the same split shows nothing (read ≥ .57:
hit .511). With a sportsbook quote on the leg: model bets and the book agrees, 158 legs hit .551;
model bets and the book sees no edge, 393 legs hit .545; model passes and the book sees an edge on the
model's side, 217 legs hit .535. A lead, not a proven edge.

## Method trap

The first pass read +17 % on "both sides offered" legs. It derived the unscored side's multiplier from
history's boost (the last snapshot in which the leg survived trimming) and the ladder's price (the
last poll), two different moments, and the profit sat entirely in payouts for sides the platform
never offered. Keep a side only when its multiplier is provable: a 1.00× leg (archived price exactly
0.5); a one-sided rung whose `1 / (1.78 · p)` is a two-decimal number; a two-sided rung whose pair
hold is inside the platform's band (Underdog .80–.895, Sleeper .84–.93 on `boost × per-pick × price`)
and whose derived other-side multiplier is itself a two-decimal number, the proof that both numbers
come from one poll. 15,182 of 71,930 quoted rungs fail that test and are dropped.

Scripts, dev box only: `~/backups/sportstradamus/2026-10-04-honest-receipts/main/consensus/`
(`build_base.py`, `strict.py`, `entries.py`, `by_kind.py`, `standard_robust.py`; `analyze.py`,
`analyze2.py` and `robust.py` are the superseded first pass and the check that caught it).
