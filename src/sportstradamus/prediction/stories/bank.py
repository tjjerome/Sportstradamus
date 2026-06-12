"""Phrase bank for prophecy theses: stat-category vocabulary + the headline cells.

The ``_PHRASES`` literal is the deterministic template bank a thesis renders
from; ``{p}`` is the driving player and ``{g}`` the game label. A stable md5 of
(game, family, player, date, shape) picks the variant — diversity comes from
bank size + context keying, never randomness.
"""

# Map a leg market to a coarse stat category so the bank can pick imagery that
# fits the stat. Needles cover every leg vocabulary in play: canonical codes
# ("PRA", "FG3M"), Underdog display names ("Pts + Rebs + Asts"), and Sleeper
# snake keys ("pts_reb_ast"), across NBA/WNBA/NFL/NHL/MLB.
_STAT_CATEGORY = {
    "scoring": (
        "point",
        "pts",
        "pra",
        "pr",
        "pa",
        "p+",
        "3-p",
        "3pt",
        "three",
        "threes",
        "fg3",
        "fgm",
        "fga",
        "fg_",
        "ftm",
        "free throw",
        "pass yd",
        "passing yards",
        "pass_yds",
        "pass td",
        "passing td",
        "rush yd",
        "rushing yards",
        "rush_yds",
        "rec yd",
        "receiving yards",
        "rec_yds",
        "kicking points",
        "goal",
        "shots on goal",
        "sog",
        "total bases",
        "hits",
        "rbi",
        "runs",
    ),
    "boards": ("rebound", "reb", "board", "ra", "pr"),
    "playmaking": (
        "assist",
        "ast",
        "pa",
        "playmak",
        "dish",
        "completions",
        "pass att",
        "receptions",
        "targets",
    ),
    "stops": (
        "steal",
        "stl",
        "block",
        "blk",
        "blst",
        "stocks",
        "tackle",
        "sack",
        "interception",
        "blocked",
    ),
    "k's": (
        "strikeout",
        "pitcher strikeouts",
        "pitcher_strikeouts",
        "ks",
        "_k",
        "strikeouts",
        "saves",
        "outs",
    ),
}


def _stat_category(market: str) -> str:
    m = (market or "").lower()
    for cat, needles in _STAT_CATEGORY.items():
        if any(n in m for n in needles):
            return cat
    return "production"


def _bank_cell(shape: str, direction: str, category: str) -> list[str]:
    cell = _PHRASES.get((shape, direction, category))
    if cell:
        return cell
    cell = _PHRASES.get(("even", direction, category))
    if cell:
        return cell
    return _PHRASES[("even", direction, "production")]


# Prophecy-voiced phrase bank, indexed by (shape, direction, stat_category).
# ``{p}`` = driving player, ``{g}`` = game label. Mystic but concrete: each
# variant names the player and the game script. A stable md5 of
# (game, family, player, date, shape) picks the variant; the date term rotates
# the headline day to day on the same matchup-shape, still deterministic per
# snapshot. Cells deliberately carry many variants so repeat users don't see
# the same headline two days running.
_PHRASES: dict[tuple[str, str, str], list[str]] = {
    # ---- shootout (high total) -------------------------------------------- #
    ("shootout", "Over", "scoring"): [
        "In {g}'s track meet, {p} pours in the points",
        "The {g} shootout writes itself and {p} is the author",
        "Scoreboard ablaze in {g}, {p} answers every run",
        "{g} turns into a fireworks show; {p} lights the fuse",
        "No brakes in {g} — {p} rides the avalanche of points",
    ],
    ("shootout", "Over", "boards"): [
        "Misses fly in the {g} shootout and {p} hoards the rebounds",
        "All that {g} firing means caroms — {p} owns the glass",
        "{p} feasts on the wreckage of {g}'s shootout",
    ],
    ("shootout", "Over", "playmaking"): [
        "{g} runs end to end and {p} keeps feeding the break",
        "In the {g} shootout {p} conducts the orchestra of assists",
        "Every {g} possession a chance — {p} sets the table all night",
    ],
    ("shootout", "Over", "stops"): [
        "Even in the {g} shootout {p} keeps stuffing the box score",
        "Chaos in {g} plays into {p}'s hands — stocks pile up",
    ],
    ("shootout", "Over", "k's"): [
        "{p} answers the {g} firefight with strikeout after strikeout",
        "Bats hot all over {g}, yet {p} keeps missing them",
    ],
    ("shootout", "Over", "production"): [
        "The {g} shootout belongs to {p}, who stuffs the sheet",
        "{p} rides the {g} track meet to a monster line",
    ],
    ("shootout", "Under", "scoring"): [
        "Even amid the {g} shootout, {p} gets held under",
        "{g} runs wild but {p} is the one shooter who goes quiet",
    ],
    ("shootout", "Under", "production"): [
        "The {g} shootout passes {p} by",
        "Points everywhere in {g} except on {p}'s ledger",
    ],
    # ---- grind (low total) ------------------------------------------------ #
    ("grind", "Over", "scoring"): [
        "In the {g} rockfight, {p} still finds the bucket",
        "{g} is a grind, but {p} drags points out of the mud",
        "When {g} bogs down, {p} is the lone source of offense",
        "Low and slow in {g} — {p} is where the points hide",
    ],
    ("grind", "Over", "boards"): [
        "The {g} rockfight is won on the glass and {p} crashes it",
        "Every {g} possession a war; {p} cleans up the misses",
        "Grind games reward the boards — {p} dominates the {g} glass",
    ],
    ("grind", "Over", "playmaking"): [
        "In the {g} grind, {p} is the one who makes the pass that matters",
        "{g} slows to a crawl and {p} orchestrates every good look",
    ],
    ("grind", "Over", "stops"): [
        "The {g} rockfight is {p}'s element — stops and takeaways pile up",
        "Defense decides {g} and {p} is everywhere on the stat sheet",
        "{p} thrives in the {g} mud, racking up stocks",
    ],
    ("grind", "Over", "k's"): [
        "{p} turns the {g} pitchers' duel into a strikeout clinic",
        "Bats go cold in {g} and {p} buries them",
    ],
    ("grind", "Over", "production"): [
        "The {g} grind bends to {p}, who fills the box score anyway",
        "{p} is the heartbeat of a rugged {g} night",
    ],
    ("grind", "Under", "scoring"): [
        "The {g} rockfight swallows {p}'s scoring whole",
        "Nothing comes easy in {g}; {p} stays bottled up",
    ],
    ("grind", "Under", "production"): [
        "The {g} grind smothers {p}",
        "A quiet {p} fits a quiet {g} script",
    ],
    # ---- blowout (lopsided ML) -------------------------------------------- #
    ("blowout", "Over", "scoring"): [
        "{p} pads the lead as {g} turns into a runaway",
        "The {g} blowout lets {p} hunt points unopposed",
        "Garbage time looms in {g} and {p} cashes in early",
        "{p} steps on the gas before {g} is even decided",
    ],
    ("blowout", "Over", "boards"): [
        "The {g} runaway sends {p} to the glass with room to roam",
        "{p} mops up the boards as {g} gets out of hand",
    ],
    ("blowout", "Over", "playmaking"): [
        "{p} carves up a {g} mismatch with assists to spare",
        "The {g} blowout is a playground and {p} runs it",
    ],
    ("blowout", "Over", "stops"): [
        "A {g} mismatch frees {p} to gamble — stocks everywhere",
        "{p} feasts on a careless {g} blowout for stops",
    ],
    ("blowout", "Over", "k's"): [
        "{p} cruises through a one-sided {g}, strikeouts mounting",
    ],
    ("blowout", "Over", "production"): [
        "The {g} runaway is {p}'s stat-sheet stuffing showcase",
        "{p} feasts before {g}'s outcome is ever in doubt",
    ],
    ("blowout", "Under", "scoring"): [
        "Up big early, {p} sits as {g} gets called off the gas",
        "The {g} blowout benches {p} before the points pile up",
        "{p}'s night ends early in a lopsided {g}",
    ],
    ("blowout", "Under", "production"): [
        "Garbage time steals {p}'s counting stats in {g}",
        "A {g} runaway quietly caps {p}'s line",
    ],
    # ---- coinflip (tight ML, mid total) ----------------------------------- #
    ("coinflip", "Over", "scoring"): [
        "{g} comes down to the wire and {p} keeps scoring to the end",
        "In a {g} coin flip, {p} is the hand on the scale — buckets",
        "Every possession matters in {g}; {p} answers with points",
    ],
    ("coinflip", "Over", "boards"): [
        "A tight {g} is decided on the glass and {p} owns it",
        "{p} crashes the boards as {g} stays knotted to the finish",
    ],
    ("coinflip", "Over", "playmaking"): [
        "{p} pulls the strings down the stretch of a razor-thin {g}",
        "In a {g} toss-up, {p} keeps creating the next good look",
    ],
    ("coinflip", "Over", "stops"): [
        "A clutch {g} swings on stops — {p} makes them",
        "{p} loads the box score as {g} stays deadlocked",
    ],
    ("coinflip", "Over", "k's"): [
        "{p} wins the {g} chess match a strikeout at a time",
    ],
    ("coinflip", "Over", "production"): [
        "In a coin-flip {g}, {p} tilts it with a full stat line",
        "{p} carries the load through a nervy {g}",
    ],
    ("coinflip", "Under", "scoring"): [
        "A clenched {g} chokes off {p}'s looks",
        "{p} gets squeezed quiet in a tense {g}",
    ],
    ("coinflip", "Under", "production"): [
        "The tight {g} keeps {p} in check",
        "{p} fades as {g} grinds to the wire",
    ],
    # ---- even (no strong signal) ------------------------------------------ #
    ("even", "Over", "scoring"): [
        "{p} is primed to fill it up against {g}",
        "The number's too low for {p} in {g} — over",
        "{p} has the matchup to score early and often in {g}",
        "Lean into {p}'s scoring touch in {g}",
    ],
    ("even", "Over", "boards"): [
        "{p} cleans the glass in {g}",
        "Rebounds come to {p} in {g} — over the number",
        "{p} owns the paint against {g}",
    ],
    ("even", "Over", "playmaking"): [
        "{p} sets the table all night in {g}",
        "Assists flow through {p} in {g}",
        "{p} runs the show against {g}",
    ],
    ("even", "Over", "stops"): [
        "{p} stuffs the box score in {g}",
        "Stocks pile up for {p} against {g}",
        "{p} is a menace on defense in {g}",
    ],
    ("even", "Over", "k's"): [
        "{p} racks up the strikeouts against {g}",
        "{p} is unhittable in {g}",
    ],
    ("even", "Over", "production"): [
        "{p} stuffs the stat sheet in {g}",
        "{p} takes over against {g}",
        "All signs point to a big {p} night in {g}",
    ],
    ("even", "Under", "scoring"): [
        "{p} runs cold against {g}",
        "The matchup holds {p} in check in {g}",
        "{p} can't buy a bucket against {g}",
    ],
    ("even", "Under", "boards"): [
        "{p} gets boxed out in {g}",
        "{p} stays off the glass against {g}",
    ],
    ("even", "Under", "playmaking"): [
        "{p} can't find anyone against {g}",
        "{g} bottles {p} up",
    ],
    ("even", "Under", "stops"): [
        "{p} stays out of the box score in {g}",
        "Quiet defensive night for {p} in {g}",
    ],
    ("even", "Under", "k's"): [
        "{p} gets hit around in {g}",
        "No swing-and-miss for {p} against {g}",
    ],
    ("even", "Under", "production"): [
        "{p} has a quiet night in {g}",
        "{p} no-shows against {g}",
        "The under is the call on {p} in {g}",
    ],
}
