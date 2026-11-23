"""MLB profile and matchup inputs: pitch-level records, leash, lineup spot and handedness.

Each reads the profile's window, the games before the row's date
(``Stats.window_short_logs``), so a value never depends on whether the player appeared that
day (docs/gbdt_feature_study.md §7.1-§7.3, §12). Handedness comes from the player registry
(``StatsMLB.players``), which the update keeps from MLB's people feed.

A rate pools its counts over the window: numerator and denominator from the same source, the
box score or the game feed (stats/mlb_pitch.py), so a row the feed has not filled adds to
neither and never dilutes a rate.
"""

import pandas as pd

from sportstradamus.stats.base import PROFILE_SHORT_GAMES
from sportstradamus.stats.mlb_pitch import PITCH_COLUMNS

# Each pooled rate: its (numerator, denominator) columns, summed over a player's games.
_PITCHER_RATES = {
    "pitcher K rate": ("pitcher strikeouts", "batters faced"),
    "pitcher BB rate": ("walks allowed", "batters faced"),
    "pitcher whiff rate": ("pitcher whiffs", "pitcher swings"),
    "pitcher chase rate": ("pitcher chases", "pitcher out-of-zone pitches"),
    "pitcher zone rate": ("pitcher zone pitches", "pitcher pitches located"),
    "pitcher zone contact rate": ("pitcher zone contacts", "pitcher zone swings"),
    "fastball mph": ("pitcher fastball mph total", "pitcher fastballs"),
}
# The leash: per-start means.
_PITCHER_MEANS = {
    "pitches per start": "pitches thrown",
    "batters faced per start": "batters faced",
    "times through order": "pitcher times through order",
}
_BATTER_RATES = {
    "batter whiff rate": ("batter whiffs", "batter swings"),
    "batter chase rate": ("batter chases", "batter out-of-zone pitches"),
    "exit velocity": ("batter exit velocity total", "batter batted balls"),
    "hard-hit rate": ("batter hard-hit balls", "batter batted balls"),
    "barrel rate": ("batter barrels", "batter batted balls"),
    "sweet-spot rate": ("batter sweet-spot balls", "batter batted balls"),
    "ground-ball rate": ("batter ground balls", "batter batted balls"),
}
_BATTER_MEANS = {"lineup spot": "spot when started", "start rate": "started"}
# The opposing lineup against a pitcher's hand, pooled over a team's games against that hand.
_LINEUP_RATES = {
    "lineup K rate vs hand": ("batter strikeouts", "plateAppearances"),
    "lineup BB rate vs hand": ("walks", "plateAppearances"),
    "lineup whiff rate vs hand": ("batter whiffs", "batter swings"),
    "lineup chase rate vs hand": ("batter chases", "batter out-of-zone pitches"),
}
# The lineup's recent form: its last 10 games against the hand (the study's 10-game half-life).
_LINEUP_SHORT_GAMES = 10
# Prior weights, in plate appearances, shrinking a batter's split toward his overall rate
# and that toward the league's (the study's x10 build, §7.1).
_SPLIT_PRIOR_PA = 150
_OVERALL_PRIOR_PA = 50
# A flag whose side or hand is not on file reads halfway between its two values.
_UNKNOWN = 0.5
_OPPOSITE = {"L": "R", "R": "L"}


def _pooled(frame: pd.DataFrame, player_col: str, rates: dict, means: dict) -> pd.DataFrame:
    sums = frame.groupby(player_col)[sorted({c for pair in rates.values() for c in pair})].sum()
    columns = {
        name: sums[num] / sums[den].where(sums[den] > 0) for name, (num, den) in rates.items()
    }
    by_player = frame.groupby(player_col)
    columns |= {name: by_player[column].mean() for name, column in means.items()}
    return pd.DataFrame(columns)


def _with_short(frame: pd.DataFrame, player_col: str, rates: dict, means: dict) -> pd.DataFrame:
    recent = frame.groupby(player_col).tail(PROFILE_SHORT_GAMES)
    return _pooled(frame, player_col, rates, means).join(
        _pooled(recent, player_col, rates, means).add_suffix(" short")
    )


def player_rates(logs: pd.DataFrame, player_col: str) -> pd.DataFrame:
    """Each player's pitch-level record, leash and lineup-spot history over ``logs``.

    A pitcher's come from his starts, a batter's from the games he batted in; each column's
    `` short`` twin reads his last PROFILE_SHORT_GAMES of them.
    """
    logs = logs.reindex(columns=logs.columns.union(PITCH_COLUMNS, sort=False))
    starts = logs[logs["starting pitcher"]].assign(
        **{
            "pitcher pitches located": logs["pitcher zone pitches"]
            + logs["pitcher out-of-zone pitches"]
        }
    )
    batted = logs[logs["battingOrder"] > 0]
    batted = batted.assign(
        **{
            "spot when started": batted["battingOrder"].where(batted["starting batter"]),
            "started": batted["starting batter"].astype(float),
        }
    )
    return _with_short(starts, player_col, _PITCHER_RATES, _PITCHER_MEANS).join(
        _with_short(batted, player_col, _BATTER_RATES, _BATTER_MEANS), how="outer"
    )


def _flag(condition: pd.Series, known: pd.Series) -> pd.Series:
    return condition.astype(float).where(known, _UNKNOWN)


def _shrunk(count, plate_appearances, baseline, prior_pa):
    """Rate of ``count`` in ``plate_appearances`` after adding ``prior_pa`` more at ``baseline``."""
    return (count + prior_pa * baseline) / (plate_appearances + prior_pa)


def platoon(
    players: pd.Index,
    logs: pd.DataFrame,
    market: str,
    starters: dict,
    bats: dict,
    throws: dict,
    player_col: str,
) -> pd.DataFrame:
    """The handedness matchup of each batter in ``players`` against his opposing starter.

    Flags for his side, the starter's hand and the platoon advantage (a switch hitter always
    holds it), then two split histories in ``market`` per plate appearance: his own against
    the starter's hand and the starter's against his side, each shrunk toward its baseline and
    read as the gap from it, beside the plate appearances behind it.
    """
    side = players.to_series().map(bats)
    starter = players.to_series().map(starters)
    hand = starter.map(throws)
    # A switch hitter bats from the side opposite the pitcher's hand.
    effective = side.where(side != "S", hand.map(_OPPOSITE))

    batting = logs[logs["battingOrder"] > 0]
    counts = [market, "plateAppearances"]
    totals = batting.groupby(player_col)[counts].sum()
    league = totals[market].sum() / totals["plateAppearances"].sum()
    overall = _shrunk(totals[market], totals["plateAppearances"], league, _OVERALL_PRIOR_PA)
    base = overall.reindex(players).to_numpy()
    own = batting.groupby([player_col, "opponent pitcher hand"])[counts].sum()
    own = own.reindex(pd.MultiIndex.from_arrays([players, hand])).fillna(0).to_numpy()
    own_rate = _shrunk(own[:, 0], own[:, 1], base, _SPLIT_PRIOR_PA)

    past_side = batting[player_col].map(bats)
    past_side = past_side.where(past_side != "S", batting["opponent pitcher hand"].map(_OPPOSITE))
    allowed = batting.groupby(["opponent pitcher", past_side.rename("side")])[counts].sum()
    allowed = allowed.reindex(pd.MultiIndex.from_arrays([starter, effective])).fillna(0).to_numpy()
    allowed_rate = _shrunk(allowed[:, 0], allowed[:, 1], league, _SPLIT_PRIOR_PA)
    return pd.DataFrame(
        {
            "platoon advantage": _flag(
                (side == "S") | (side != hand), side.notna() & (hand.notna() | (side == "S"))
            ),
            "facing lefty": _flag(hand == "L", hand.notna()),
            "bats left": _flag(side == "L", side.notna()),
            "switch hitter": (side == "S").astype(float),
            "split gap": own_rate - base,
            "PA vs hand": own[:, 1],
            "pitcher split gap": allowed_rate - league,
            "pitcher PA vs side": allowed[:, 1],
        },
        index=players,
    )


def lineup_vs_hand(
    players: pd.Index, logs: pd.DataFrame, opponents: dict, throws: dict
) -> pd.DataFrame:
    """The opposing lineup's rates against each pitcher's hand, over the window and recently.

    A lineup is the opponent's batters in its games against a starter of that hand.
    """
    logs = logs.reindex(columns=logs.columns.union(PITCH_COLUMNS, sort=False))
    batting = logs[logs["battingOrder"] > 0]
    counts = sorted({c for pair in _LINEUP_RATES.values() for c in pair})
    games = batting.groupby(["team", "opponent pitcher hand", "gameId"], sort=False)[counts].sum()
    keys = pd.MultiIndex.from_arrays([players.map(opponents), players.map(throws)])
    frames = []
    for suffix, window in (
        ("", games),
        (" short", games.groupby(level=[0, 1]).tail(_LINEUP_SHORT_GAMES)),
    ):
        sums = window.groupby(level=[0, 1]).sum().reindex(keys)
        frames.append(
            pd.DataFrame(
                {
                    name + suffix: (sums[num] / sums[den].where(sums[den] > 0)).to_numpy()
                    for name, (num, den) in _LINEUP_RATES.items()
                },
                index=players,
            )
        )
    return pd.concat(frames, axis=1)
