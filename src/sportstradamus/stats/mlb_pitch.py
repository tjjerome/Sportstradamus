"""Per player-game pitch counts from Baseball Savant's game feed (``gf?game_pk=``).

``StatsMLB.parse_game`` reads the feed for its box score, which keeps none of the pitches;
:func:`pitch_columns` keeps their counts, and :func:`fill_pitch_columns` re-reads the feed for
the games logged before it did. They are their own column set, which the hourly update never
refills, so no update run downloads a past game for them.

Counts follow the search export's definitions (docs/gbdt_feature_study.md §7.3) and stay sums,
so a rate pools them: whiffs over swings, chases over out-of-zone pitches, the fastball mph
total over fastballs. Pitches thrown, batters faced, strikeouts, walks and plate appearances
are the box score's. History comes from the feed, not the export: on 24 games of 2024-26 the
export, re-processed since, differs on the zone of 253 of 7,043 pitches, on pitch types and
barrels, and lists automatic balls as pitches.
"""

from time import sleep

import pandas as pd
from tqdm import tqdm

from sportstradamus.spiderLogger import logger
from sportstradamus.stats.base import scraper

GAME_FEED_URL = "https://baseballsavant.mlb.com/gf?game_pk={game_id}"
# A feed runs 2-3 MB and takes about 2.3 s to fetch; with a second between them a game
# takes 3.3 s, a season 2.3 hours (measured 2026-11-23, 40 games of 2024-26).
_REQUEST_GAP_S = 1.0
# Games in a row the feed fails before a fill stops and leaves the rest for a later call.
_MAX_FAILURES_IN_A_ROW = 5

_FIELDS = [
    "type",
    "pitcher",
    "batter",
    "pitch_call",
    "zone",
    "pitch_type",
    "start_speed",
    "launch_speed",
    "launch_angle",
    "is_barrel",
    "pitcher_time_thru_order",
]
_SWINGS = frozenset(
    {
        "swinging_strike",
        "swinging_strike_blocked",
        "foul",
        "foul_tip",
        "hit_into_play",
        "foul_bunt",
        "missed_bunt",
        "bunt_foul_tip",
    }
)
_WHIFFS = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})
_FASTBALLS = frozenset({"FF", "SI", "FC"})
# Statcast numbers the strike zone 1-9 and the squares around it 11-14.
_ZONE = (1, 9)
_OUT_OF_ZONE = (11, 14)
# Statcast's hard-hit ball leaves the bat at 95 mph or more.
_HARD_HIT_MPH = 95
# Statcast's sweet spot: launch angles 8 to 32 degrees.
_SWEET_SPOT_DEGREES = (8, 32)
# Statcast's ground ball leaves the bat under 10 degrees.
_GROUND_BALL_DEGREES = 10

_PITCHER_COUNTS = [
    "fastballs",
    "fastball mph total",
    "swings",
    "whiffs",
    "chases",
    "zone pitches",
    "out-of-zone pitches",
    "zone swings",
    "zone contacts",
]
_BATTER_COUNTS = [
    "pitches",
    "swings",
    "whiffs",
    "chases",
    "out-of-zone pitches",
    "batted balls",
    "exit velocity total",
    "hard-hit balls",
    "barrels",
    "sweet-spot balls",
    "ground balls",
]
PITCH_COLUMNS = [
    "pitcher times through order",
    *(f"pitcher {count}" for count in _PITCHER_COUNTS),
    *(f"batter {count}" for count in _BATTER_COUNTS),
]


def pitch_columns(game: dict, player_ids: pd.Series) -> pd.DataFrame:
    """PITCH_COLUMNS of each of ``player_ids`` in ``game``, one row each, in their order.

    The pitcher columns count the pitches he threw, the batter columns those he saw; the feed
    holds every pitch of the game, so a player it shows none for reads 0. An intentional walk
    or automatic ball or strike is no pitch and counts nowhere.
    """
    pitches = pd.DataFrame.from_records(game["team_home"] + game["team_away"], columns=_FIELDS)
    pitches = pitches[pitches["type"] == "pitch"]
    call = pitches["pitch_call"]
    swing = call.isin(_SWINGS)
    whiff = call.isin(_WHIFFS)
    in_zone = pitches["zone"].between(*_ZONE)
    out_of_zone = pitches["zone"].between(*_OUT_OF_ZONE)
    speed = pd.to_numeric(pitches["start_speed"], errors="coerce")
    fastball = pitches["pitch_type"].isin(_FASTBALLS) & speed.notna()
    # The feed sends batted-ball measurements as text.
    exit_velocity = pd.to_numeric(pitches["launch_speed"], errors="coerce")
    batted = call.eq("hit_into_play") & exit_velocity.notna()
    # The feed leaves out a launch angle of exactly 0, and is_barrel with it.
    angle = pd.to_numeric(pitches["launch_angle"], errors="coerce").fillna(0)
    counts = pd.DataFrame(
        {
            "pitches": 1,
            "fastballs": fastball,
            "fastball mph total": speed.where(fastball, 0),
            "swings": swing,
            "whiffs": whiff,
            "chases": swing & out_of_zone,
            "zone pitches": in_zone,
            "out-of-zone pitches": out_of_zone,
            "zone swings": swing & in_zone,
            "zone contacts": swing & in_zone & ~whiff,
            "batted balls": batted,
            "exit velocity total": exit_velocity.where(batted, 0),
            "hard-hit balls": batted & (exit_velocity >= _HARD_HIT_MPH),
            "barrels": batted & pitches["is_barrel"].eq(1),
            "sweet-spot balls": batted & angle.between(*_SWEET_SPOT_DEGREES),
            "ground balls": batted & (angle < _GROUND_BALL_DEGREES),
        }
    ).astype(float)
    thrown = counts.groupby(pitches["pitcher"])[_PITCHER_COUNTS].sum().add_prefix("pitcher ")
    thrown["pitcher times through order"] = pitches.groupby("pitcher")[
        "pitcher_time_thru_order"
    ].max()
    seen = counts.groupby(pitches["batter"])[_BATTER_COUNTS].sum().add_prefix("batter ")
    return thrown.join(seen, how="outer").reindex(player_ids, columns=PITCH_COLUMNS).fillna(0)


def fill_pitch_columns(
    gamelog: pd.DataFrame, seasons: set[int] | None = None, max_attempts: int = 3
) -> pd.DataFrame:
    """``gamelog`` with PITCH_COLUMNS filled on the rows of every game the feed answers for.

    Only games holding a row that lacks the columns are read, so a call resumes where an
    interrupted one stopped. The rows of a game the feed fails keep NaN, as do the rest once it
    fails ``_MAX_FAILURES_IN_A_ROW`` games running.

    Args:
        gamelog: The MLB gamelog.
        seasons: Years to fill, e.g. ``{2025}``; ``None`` is every year.
        max_attempts: Tries per feed request.

    Returns:
        The gamelog, holding every PITCH_COLUMNS column.
    """
    log = gamelog.reindex(columns=gamelog.columns.union(PITCH_COLUMNS, sort=False))
    lacking = log[PITCH_COLUMNS].isna().any(axis=1)
    if seasons is not None:
        lacking &= pd.to_datetime(log["gameDate"]).dt.year.isin(seasons)
    failures = 0
    for game_id in tqdm(log.loc[lacking, "gameId"].unique(), desc="Getting MLB pitch counts"):
        sleep(_REQUEST_GAP_S)
        game = scraper.get(GAME_FEED_URL.format(game_id=game_id), max_attempts=max_attempts)
        if not game:
            failures += 1
            if failures == _MAX_FAILURES_IN_A_ROW:
                logger.warning("Savant's game feed stopped answering; the unfilled rows wait")
                break
            continue
        failures = 0
        rows = lacking & (log["gameId"] == game_id)
        log.loc[rows, PITCH_COLUMNS] = pitch_columns(game, log.loc[rows, "playerId"]).to_numpy()
    return log
