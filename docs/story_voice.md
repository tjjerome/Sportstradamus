# Story voice

The authoring contract for `data/config/voice_bank.json`, `data/config/stat_words.json`, and
the `dek` branches of `data/config/why_bank.json`. Prose lives in those files, never in code
(`tests/golden/test_bank_coverage.py::test_no_prose_literals_in_stories_source`); the goldens
in that file enforce everything mechanical below. Register: **calm analyst** — an analyst's
causal spine in a monk's still cadence, chosen after a survey of how the DFS/parlay press
writes a slip (environment first, correlation as cause and named effects, confidence stated
once, no contentless instructions).

## Rules

1. **One thought per headline**: subject, verb, consequence. Six to fourteen rendered words.
   Present tense. No em-dash, no semicolon, at most one comma or one colon. Every variant
   leads with a capital letter or a slot, in every direction (Mixed and Contrast* included).
   A headline is a card title: one sentence carries no closing period; a two-sentence
   headline keeps both.
2. **Cause first when the shape is known**, then the effect on a named stat family or player.
   Cause words by shape: *shootout* — the total, the pace, a loud number; *grind* — two aces,
   a low number, the pitching, the trenches; *blowout* — the spread, the margin, garbage time,
   the bench; *coinflip* — the clock, a tight game, one possession; *even* — the card, the
   board. Never a venue name: the bank cannot know the park or the arena.
3. **Mixed = one cause, two named effects.** `{up}` and `{down}` are full clauses the engine
   supplies from `stat_words.json` ("the strikeouts pile up", "Joel Embiid comes up short on
   points"). A Mixed template contains each exactly once, never leads with either, and joins
   them with "and", "while", ":" or ",". Never "ride some, fade the rest".
4. **Calm, non-attached diction**: sits, settles, stays, lets, comes, rests, the quiet side,
   nothing to force, no need to chase. At most one nature image per headline (tide, wind,
   still water), none in a dek. No hype (smash, hammer, cash, lock), no emoji, no imperative
   aimed at the reader beyond a gentle "take the quiet side"-class lean.
5. **Analyst spine**: the read names a mechanism — usage, minutes, pace, platoon, pitch count,
   bench depth, tempo, clock, the spread.
6. **Sport vocabulary** stays inside the keyword floors (below).
7. **Bet words** (over, overs, under, unders) stay banned outside `mistakes` cells, and inside
   them only the flipped word may appear (an `Over` mistakes cell may say "under", never
   "over").
8. **`{g}` is a matchup label, never an opponent** ("against {g}" is banned). A locative
   `{g}` renders as the home city: the engine rewrites it when one of *in, at, to, into,
   through, from, across, around, inside, onto, within, near* precedes it or one of
   *faithful, crowd, fans, building, barn, arena, ice, floor, court, field, park, gym, house*
   follows it. Use only those forms for the place reading; any other construction renders the
   matchup label.

## Slots

| Archetype | Slots | Mixed adds |
|---|---|---|
| player | `{p}` player, `{g}` matchup | `{up}` `{down}` |
| stack | `{n}` leg count, `{p}` anchor, `{g}` | `{up}` `{down}` |
| unit | `{team}`, `{grp}` position-group word, `{opp}`, `{g}` | — (unit has no Mixed) |
| game-script | `{g}` | `{up}` `{down}` |

`{home}` is synthesized from a locative `{g}` (rule 8); never write it yourself.

### `{up}` / `{down}` clause contract (`stat_words.json`)

`voice → family → {noun, thrive, fade, owned_thrive, owned_fade}` for the seven families
(`scoring`, `boards`, `playmaking`, `stops`, `k's`, `production`, `mistakes`) in `shared` and
the four sport voices. `noun` is plural and starts with "the" ("the strikeouts"); `thrive` and
`fade` are plural verb phrases ("pile up", "stay down"). Valence for negative markets lives
here: the `mistakes` thrive clause reads "the turnovers stay down". When both sides of a split
share a family, the engine names the players instead: `owned_thrive` / `owned_fade` are
player-subject templates with exactly `{who}` and `{what}` ("{who} clears the {what} number",
"{who} comes up short on {what}"), so verb agreement never depends on the market name.

## Keyword floors (from `test_bank_coverage.py`)

- basketball: player/even/Mixed/production has one of floor, glass, bucket, rim, possession;
  stack/even/ContrastOver has one of gym, floor, bucket, shot, rim; player/even/Under/mistakes
  has one of turnover, handle, giveaway, pocket.
- football: player/even/Over/scoring has one of yard, chain, end zone, drive, snap;
  game-script/grind/Under/production has one of punt, three-and-out, slugfest, trench;
  player/even/Mixed/production has one of snap, drive, yard, down, huddle;
  stack/even/ContrastUnder has one of field, drive, chain, boundary, clock;
  player/even/Under/mistakes has one of sack, pick, fumble, pocket.
- hockey: player/even/Over/scoring has one of net, lamp, puck, twine, ice; player/even/Over/k's
  has one of crease, door, puck, save; player/even/Mixed/production has one of shift, ice,
  puck, period, bench; stack/even/ContrastOver has one of lamp, ice, skate, line, net;
  player/even/Under/mistakes has one of crease, net, lamp, goal, rubber.
- baseball: player/even/Over/k's has one of strikeout, punchout, whiff, swing, carve;
  player/even/Over/scoring has one of bases, line drive, bat, plate, square;
  player/even/Mixed/production has one of plate, inning, bat, swing, zone;
  stack/even/ContrastOver has one of lineup, order, bat, barrel, plate;
  player/even/Under/mistakes has one of walk, free pass, zone, wild.

## Structure the goldens hold fixed

Every cell that exists today must survive with at least six distinct variants; adding a
category cell under an existing node is welcome; removing a shaped node breaks
`test_shape_never_falls_through_to_even`. Every shaped node carries a `production` cell. Each
sport voice authors its own `even/production` cells for player, stack, and game-script in
every direction, plus `player/even/{Over,Under}/mistakes`, plus the four contrarian cells
(`game-script/shootout/Under`, `game-script/grind/Over`, `stack/shootout/Under`,
`stack/grind/Over`). Templates stay at most sixteen words. Edit the JSON with `json.load` and
`json.dump(indent=2, ensure_ascii=False)` plus a trailing newline, never string surgery.

## Samples

- baseball / game-script / shootout / Under — "The {g} total says slugfest. The model reads
  these bats quieter than the number."
- baseball / game-script / grind / Mixed — "Two aces shorten {g}: {up} and {down}" →
  "Two aces shorten DET/CLE: the strikeouts climb and the hits stay down"
- basketball / player / shootout / Over / scoring — "The {g} pace runs through {p}, and the
  points follow"
- shared / stack / even / ContrastOver — "{p} carries the {g} card while the rest of it sits
  still"
- dek — "these {n} legs move together, {rho} average correlation · {p} runs {dev} past a
  {line} line over his last 5"
