# Story voice

The authoring contract for `data/config/voice_bank.json`, `data/config/stat_words.json`, and
the `dek` branches of `data/config/why_bank.json`. Prose lives in those files, never in code
(`tests/golden/test_bank_coverage.py::test_no_prose_literals_in_stories_source`); the goldens
in that file enforce everything mechanical below. Register: **live analyst** — the analyst's
causal spine (environment first, correlation as cause and named effects, confidence stated
once, no contentless instructions) delivered from the booth. The mechanics stay: one
mechanism, one read, six to fourteen rendered words, present tense, no hype, deterministic
rotation. On top of them: concrete action verbs, booth imagery (the scoreboard, the chains,
the lamp, the barrel, the crowd noise, never a venue name), and the star as the sentence's
agent, so every game on a slate reads as its own game rather than one calm sentence with the
names swapped.

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
4. **Active diction**: the verb carries the headline, drawn from the live list the goldens
   score (`_LIVE_VERB_STEMS` and `_LIVE_VERB_PHRASES` in `test_bank_coverage.py`: drives,
   pours, piles, carries, floods, buries, swarms, smothers, erupts, takes over, goes cold,
   dries up, racks up, …). Booth imagery is welcome (the scoreboard, the chains, the lamp,
   the barrel, the crowd noise), at most one image per headline, none in a dek. Still words
   (sits, settles, rests, still, quiet, calm, gentle, nothing to force, no need to chase) are
   rationed to one variant in ten per voice. No hype (smash, hammer, cash, lock), no emoji,
   no imperative aimed at the reader.
5. **Analyst spine**: the read names a mechanism — usage, minutes, pace, platoon, pitch count,
   bench depth, tempo, clock, the spread.
6. **Sport vocabulary** stays inside the keyword floors (below).
7. **Bet words** (over, overs, under, unders) stay banned outside `mistakes` cells, and inside
   them only the flipped word may appear (an `Over` mistakes cell may say "under", never
   "over"). The live phrase "takes over" carries the bet word, so it fits only an `Under`
   mistakes cell; everywhere else pick another live phrase. The `dek.form` branches of `why_bank.json` (`above_for` … `below_against`) are
   bet-relative, so each may name only its own side ("a trend the under rides"); the
   `dek.matchup` branches (`gives`, `takes`) read the defense, true on either side, and name
   neither.
8. **`{g}` is a matchup label, never an opponent** ("against {g}" is banned). A locative
   `{g}` renders as the home city: the engine rewrites it when one of *in, at, to, into,
   through, from, across, around, inside, onto, within, near* precedes it or one of
   *faithful, crowd, fans, building, barn, arena, ice, floor, court, field, park, gym, house*
   follows it. Use only those forms for the place reading; any other construction renders the
   matchup label.
9. **Star-led cells**: in every `player/<shape>/Over/<category>` cell at least three variants
   open on `{p}` (within the first three tokens; `{p}'s` counts). The Over-led story each
   game headlines is seeded on the game's star, so the star must be the agent of the
   sentence, not the object the board happens to favor.
10. **Valence outranks shape for `mistakes`**: a negative market (turnovers, sacks taken,
    walks) thrives on the Under, so `bank_cell` resolves a `mistakes` lookup through every
    mistakes cell (voiced shape, shared shape, voiced even, shared even) before any
    `production` fallback. Mistakes cells are authored under `even` only, and `shared`
    carries `even` mistakes cells for player, stack, and game-script in both directions, so
    a TOV stack or a mistakes-led game script never inherits production copy.

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

Register floors, scored per voice (`shared` and each sport voice independently, every
archetype and cell counted):

- live-verb share ≥ 0.60 of the voice's variants (`_LIVE_VERB_RE`);
- still-word share ≤ 0.10 (`_STILL_WORD_RE`);
- star-led: every `player/<shape>/Over/<category>` cell has at least 3 variants opening on
  `{p}` (rule 9);
- daily cells: every `player/<shape>/Over/scoring` and `player/<shape>/Over/production` cell
  carries at least 8 variants, because they fire daily as Over-led headlines and slate-wide
  dedup needs the room;
- `shared` authors `even` mistakes cells for player, stack, and game-script in both Over and
  Under (rule 10).

Vocabulary floors:

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

Every cell that exists today must survive with at least six distinct variants (eight for the daily
cells, see the floors above); adding a category cell under an existing node is welcome; removing a
shaped node breaks `test_shape_never_falls_through_to_even`. Every shaped node carries a
`production` cell. The football voice authors no `stops` cell: no NFL market maps there (`sacks
taken` and `interceptions` are negative markets and read as `mistakes`). The basketball voice
authors no `k's` cell for the same reason: no NBA or WNBA market maps there (`blocks` reads as
`stops`, because `_STAT_CATEGORY` checks `block` before the `ks` needle). Each sport voice authors
its own `even/production` cells for player, stack, and game-script in every direction, plus
`player/even/{Over,Under}/mistakes`, plus the four contrarian cells (`game-script/shootout/Under`,
`game-script/grind/Over`, `stack/shootout/Under`, `stack/grind/Over`). Templates stay at most
sixteen words. Edit the JSON with `json.load` and `json.dump(indent=2, ensure_ascii=False)` plus a
trailing newline, never string surgery.

## Samples

- baseball / player / even / Over / scoring — "{p} pours line drives into the gaps in {g}"
- baseball / game-script / grind / Mixed / production — "Two aces choke {g}: {up} and {down}" →
  "Two aces choke DET/CLE: the strikeouts climb and the hits stay down"
- basketball / player / shootout / Over / scoring — "{p} pours in points while the {g} pace runs
  wide open"
- football / player / grind / Over / scoring — "{p} grinds the chains forward while {g} bogs down"
- hockey / player / shootout / Under / scoring — "Goals rain from every stick in {g} except {p}'s"
  → "Goals rain from every stick in Edmonton except Connor McDavid's"
- shared / stack / even / ContrastOver / production — "{p} carries the {g} card while the rest of
  it stalls"
- shared / stack / even / Under / mistakes — "The pressure hunts {p} in {g}, and {n} legs go over
  with it" (the flipped bet word is the only one an `Under` mistakes cell may carry)
- dek — "these {n} legs move together, {rho} average correlation"
