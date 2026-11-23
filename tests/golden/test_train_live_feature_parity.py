"""Train/live feature-parity gate (model track §6.0.4, the build-first blocker).

``get_training_matrix`` (``stats/base.py`` ``get_training_matrix``) and the live
serving path both compute features through the **same** ``Stats.get_stats``:
training calls it per gameday inside the matrix loop, serving calls it from
``prediction/scoring.py`` ``match_offers`` / ``prediction/model_prob.py``. With the
``date`` argument fixed to a gameday ``D`` every feature helper is keyed on ``date``
and is identical across the two paths -- except ``_game_context``, which branches on
``date < datetime.today().date()``:

* historical (training) branch -- reads ``Home`` / ``Moneyline`` / ``Total`` from the
  realized gamelog row for ``D``, and (StatsNBA's ``_availability_context``, at its end)
  who sits out from the official per-game absence lists;
* upcoming (serving) branch -- reads ``Home`` from ``self.upcoming_games``,
  ``Moneyline`` / ``Total`` from ``archive.get_moneyline`` / ``archive.get_total``, and
  who sits out ``D`` from the newest ESPN injury report (the official lists for the
  games before it).

The gamelog's ``moneyline`` / ``totals`` are baked from that same archive as of 15:00
UTC on the game date, while serving reads the newest quote for today's game. With the
two stubbed equal, the invariant under test is *archive-baked-at-train ==
archive-read-at-serve*; likewise the official list and ESPN's report name the same
players out on ``D``, so the absence columns must agree. This is the "edit both
branches" risk that every game-context feature carries
(``docs/handoffs/model_improvement_track.md`` §6.3.1, §7.2 step 2); the doc gates all
later feature work behind this harness existing.

The test drives the real ``StatsNBA.get_stats`` twice on one synthetic gameday,
flipping **only** that branch (real today ``> D`` vs a frozen today ``== D``), and
asserts the shared feature surface (``get_stat_columns``) matches. Synthetic gamelog
data, a stubbed archive, absence sources written under ``tmp_path``, pre-seeded comps,
and an injected depth keep it fast, deterministic, and free of network / DuckDB -- each
of those is branch-*independent* state shared by both arms, so none can mask the branch
divergence under test. The ``inf`` handling is covered by the no-inf assertion:
training (``get_training_matrix``) and the model's serving path (``match_offers``) both
end ``fillna(0).replace([inf,-inf],0)``, but the book fallback (``book_fallback_prob``)
joins its features after ``fillna(0)`` only, so a shared column that can be ``inf``
would diverge there, and the gate forbids one.

``test_mlb_train_and_live_feature_paths_agree`` holds ``StatsMLB`` to the same gate on synthetic
logs, for a pitcher and a hitter market, and requires each input ``stats/mlb_profile.py`` adds to
reach ``get_stat_columns`` and read non-zero somewhere.
"""

from __future__ import annotations

import datetime as dt
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import StatsMLB, StatsNBA, availability, base, mlb, nba
from sportstradamus.stats.mlb_pitch import PITCH_COLUMNS
from sportstradamus.stats.weather_features import WEATHER_COLUMNS

MARKET = "PTS"
GAMEDAY = date(2025, 12, 1)  # after the NBA season_start; far in the past vs real today
TEAMS = ("AAA", "BBB")
# Player 6's gamelog name keeps its accent, as some real ones do: ESPN's report reaches his
# gamelog id only through the accent-free spelling.
PLAYERS = {
    ("Plàyer 6" if i == 6 else f"Player {i}"): (TEAMS[0] if i < 5 else TEAMS[1]) for i in range(10)
}
PLAYER_IDS = {player: 1000 + i for i, player in enumerate(PLAYERS)}
POSITIONS = ("P", "C", "F", "W", "B")
GAMES_BACK = 15

# Who sits, as (days before GAMEDAY, team, player, official list status). Training reads the
# official list; serving reads it for the earlier days and ESPN's report for GAMEDAY, which
# names the same players. Each source also lists one player its reader drops: the official
# list a game-time DNP, ESPN a Day-To-Day.
OFFICIAL_LIST = [
    (0, "AAA", "Player 1", "Inactive - Injury/Illness"),
    (0, "BBB", "Plàyer 6", "Inactive"),
    (0, "AAA", "Player 3", "DNP - Coach's Decision"),
    (1, "AAA", "Player 2", "Inactive"),
    (4, "BBB", "Player 7", "Inactive"),
]
ESPN_REPORT = [
    ("AAA", "Player 1", "Out"),
    ("BBB", "Plàyer 6", "Suspension"),
    ("AAA", "Player 3", "Day-To-Day"),
]

# Per-team game context baked into the synthetic gamelog. The stubbed archive and
# upcoming_games must echo these exactly, so the two _game_context branches agree.
TEAM_MONEYLINE = {"AAA": -150.0, "BBB": 130.0}
# 14.5-point implied gap so |Spread| clears the NBA blowout threshold -- exercises
# the Blowout==1 (.astype(int) on True) path through both _game_context branches.
TEAM_TOTAL = {"AAA": 112.5, "BBB": 98.0}
TEAM_HOME = {"AAA": True, "BBB": False}


def _opponent(team: str) -> str:
    return TEAMS[1] if team == TEAMS[0] else TEAMS[0]


def _build_logs(stat_types: list[str], team_stat_types: list[str]):
    """Synthetic NBA gamelog + teamlog covering ``GAMES_BACK`` days up to ``GAMEDAY``.

    Columns follow ``StatsNBA.log_strings`` plus the per-cell stat-type panels, the
    archive-baked ``moneyline`` / ``totals``, the ``PLAYER_ID`` and ``AST`` / ``REB`` /
    ``FGA`` the absence weights read, and the other counts the profile pools per 48 minutes.
    Values are deterministic (seeded RNG); only their *parity across the two branches*
    matters, never their realism.
    """
    rng = np.random.default_rng(0)
    game_dates = [GAMEDAY - dt.timedelta(days=k) for k in range(GAMES_BACK, -1, -1)]
    player_pos = {p: POSITIONS[i % len(POSITIONS)] for i, p in enumerate(PLAYERS)}

    rows = []
    for gd in game_dates:
        for player, team in PLAYERS.items():
            row = {
                "GAME_ID": f"{gd.isoformat()}_{team}",
                "GAME_DATE": gd.isoformat(),
                "PLAYER_ID": PLAYER_IDS[player],
                "PLAYER_NAME": player,
                "MIN": float(rng.uniform(15, 35)),
                "USG_PCT": float(rng.uniform(0.1, 0.3)),
                "POS": player_pos[player],
                "TEAM_ABBREVIATION": team,
                "OPP": _opponent(team),
                "HOME": TEAM_HOME[team],
                "WL": "W",
                "PTS": float(rng.uniform(5, 30)),
                "AST": float(rng.uniform(0, 10)),
                "REB": float(rng.uniform(0, 12)),
                "FGA": float(rng.uniform(3, 20)),
                "FG3A": float(rng.uniform(1, 8)),
                "OREB": float(rng.uniform(0, 4)),
                "DREB": float(rng.uniform(1, 8)),
                "TOV": float(rng.uniform(0, 4)),
                "BLKA": float(rng.uniform(0, 2)),
                "STL": float(rng.uniform(0, 3)),
                "AGE": 25.0,
                "moneyline": TEAM_MONEYLINE[team],
                "totals": TEAM_TOTAL[team],
            }
            for st in stat_types:
                row.setdefault(st, float(rng.uniform(0, 1)))
            rows.append(row)

    team_rows = []
    for gd in game_dates:
        for team in TEAMS:
            row = {
                "GAME_DATE": gd.isoformat(),
                "GAME_ID": f"{gd.isoformat()}_{team}",
                "TEAM_ABBREVIATION": team,
                "OPP": _opponent(team),
                "HOME": TEAM_HOME[team],
                "moneyline": TEAM_MONEYLINE[team],
                "totals": TEAM_TOTAL[team],
                "PTS": float(rng.uniform(95, 120)),
            }
            for st in team_stat_types:
                row.setdefault(st, float(rng.uniform(0, 1)))
            team_rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(team_rows)


class _StubArchive:
    """Archive that returns the gamelog-baked context -- never opens DuckDB.

    Only the upcoming ``_game_context`` branch consults the archive; it must return
    exactly what the historical branch reads from the gamelog row, which is the
    parity invariant being asserted.
    """

    def get_moneyline(self, league, game_date, team, **_):
        return TEAM_MONEYLINE[team]

    def get_total(self, league, game_date, team, **_):
        return TEAM_TOTAL[team]

    def get_ev(self, *_, **__):
        return np.nan

    def get_line(self, *_, **__):
        return 0.0


def _seed_comps(stats: StatsNBA) -> None:
    """Pre-seed the per-week comp cache so ``_ensure_comps`` short-circuits.

    Avoids the BallTree build (which needs ``self.players`` + ``playerCompStats.json``).
    Comps are date-keyed and branch-independent, so a fixed pool is shared identically
    by both arms.
    """
    names = list(PLAYERS)
    comps: dict = {pos: {} for pos in POSITIONS}
    for i, player in enumerate(names):
        partner = names[(i + 5) % len(names)]
        comps[POSITIONS[i % len(POSITIONS)]][player] = {"comps": [partner], "distances": [0.5]}
    stats.comps = comps
    key = stats._comp_target_key(GAMEDAY)
    stats._current_comps_key = key
    stats._comps_by_week[key] = comps


def _write_absence_sources(lists_base: Path, capture_dir: Path) -> None:
    """Store ``OFFICIAL_LIST`` and ``ESPN_REPORT`` as their readers find them on disk."""
    lists = pd.DataFrame(
        [
            (pd.Timestamp(GAMEDAY - dt.timedelta(days=days)), team, PLAYER_IDS[player], status)
            for days, team, player, status in OFFICIAL_LIST
        ],
        columns=["game_date", "team", "player_id", "status"],
    )
    # Lists are filed by season start year, and GAMEDAY falls in 2025-26.
    lists_path = lists_base / "NBA" / "injuries" / "2025.parquet"
    lists_path.parent.mkdir(parents=True)
    lists.to_parquet(lists_path, index=False)
    report_path = capture_dir / "NBA" / f"date={GAMEDAY.isoformat()}" / "120000.parquet"
    report_path.parent.mkdir(parents=True)
    pd.DataFrame(ESPN_REPORT, columns=["team", "name", "status"]).to_parquet(
        report_path, index=False
    )


@pytest.fixture
def primed_stats(monkeypatch, tmp_path) -> tuple[StatsNBA, list[str]]:
    """A ``StatsNBA`` (with synthetic logs) and its ``get_stat_columns``, both arms ready.

    Points the absence readers at ``OFFICIAL_LIST`` and ``ESPN_REPORT`` under ``tmp_path``.
    Fetches ``get_stat_columns`` first (it refreshes the base profile at *today*),
    then establishes the profile at ``GAMEDAY`` and injects ``depth`` -- normally set
    by the external ``get_depth`` volume step that gates ``_join_profiles``. The
    ``profile_market`` cache guard then lets both arms reuse this profile without a
    rebuild, so callers must not touch ``get_stat_columns`` / ``base_profile`` between
    the arms -- which is why the column list is resolved here, not in the test.
    """
    monkeypatch.setattr(base, "archive", _StubArchive())
    lists_base, capture_dir = tmp_path / "player_data", tmp_path / "injury_capture"
    _write_absence_sources(lists_base, capture_dir)
    monkeypatch.setattr(
        availability, "official_absences", partial(availability.official_absences, base=lists_base)
    )
    monkeypatch.setattr(
        availability, "espn_absences", partial(availability.espn_absences, capture_dir=capture_dir)
    )
    stats = StatsNBA()
    player_stat_types, team_stat_types = stats._profile_stat_types()
    stats.gamelog, stats.teamlog = _build_logs(player_stat_types, team_stat_types)
    _seed_comps(stats)
    stats.upcoming_games = {
        team: {"Opponent": _opponent(team), "Home": TEAM_HOME[team]} for team in TEAMS
    }

    columns = stats.get_stat_columns(MARKET)
    stats.profile_market(MARKET, GAMEDAY)
    stats.playerProfile["depth"] = 1.0
    return stats, columns


def _offers() -> list[dict]:
    return [
        {"Player": p, "Team": t, "Opponent": _opponent(t), "Date": GAMEDAY.isoformat()}
        for p, t in PLAYERS.items()
    ]


class _FrozenDatetime(dt.datetime):
    @classmethod
    def today(cls):
        return dt.datetime(GAMEDAY.year, GAMEDAY.month, GAMEDAY.day)


def test_train_and_live_feature_paths_agree_on_a_frozen_gameday(primed_stats, monkeypatch):
    stats, columns = primed_stats
    offers = _offers()

    # Training arm: real today >> GAMEDAY -> _game_context historical branch.
    train = stats.get_stats(MARKET, offers, GAMEDAY)
    assert not train.empty, "synthetic historical arm produced no rows"

    # Serving arm: freeze today == GAMEDAY -> _game_context upcoming branch, and the same
    # branch in StatsNBA's absence hook, which reads today in stats/nba.py.
    for module in (base, nba):
        monkeypatch.setattr(module, "datetime", _FrozenDatetime)
    live = stats.get_stats(MARKET, offers, GAMEDAY)
    assert not live.empty, "synthetic upcoming arm produced no rows"

    assert set(train.columns) == set(live.columns), (
        "training and serving get_stats emit different columns: "
        f"{set(train.columns) ^ set(live.columns)}"
    )

    shared = [c for c in columns if c in train.columns and c in live.columns]
    assert shared, "no get_stat_columns survived into get_stats output"

    train, live = train.sort_index(), live.sort_index()
    mismatched = []
    for col in shared:
        a, b = train[col], live[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(
                a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=0, atol=1e-9, equal_nan=True
            ):
                mismatched.append(col)
        elif not a.equals(b):
            mismatched.append(col)
    assert not mismatched, f"train/live feature values diverge on the shared surface: {mismatched}"

    # inf guard: training and match_offers zero inf, the book fallback (model_prob
    # fillna only) keeps it. Forbid inf on the shared surface so every path agrees.
    inf_cols = [
        c
        for c in shared
        if pd.api.types.is_numeric_dtype(train[c])
        and (
            np.isinf(train[c].to_numpy(dtype=float)).any()
            or np.isinf(live[c].to_numpy(dtype=float)).any()
        )
    ]
    assert not inf_cols, (
        f"shared feature columns contain inf (the book fallback keeps it): {inf_cols}"
    )

    # The _game_context invariant, named explicitly so a regression points here.
    # OppTotal + the derived Spread/GameTotal/Blowout (QW-1) ride the same
    # historical-gamelog vs upcoming-archive divergence as Total, so they pin here too;
    # Playoff (game-ID season-type decode) and the series-context columns
    # (SeriesWins/SeriesLosses/FacingElimination/CanClinch) are the same _game_context
    # surface -- regular-season synthetic logs short-circuit them to 0 in both branches.
    for col in (
        "Home",
        "Moneyline",
        "Total",
        "OppTotal",
        "Spread",
        "GameTotal",
        "Blowout",
        "Playoff",
        "SeriesWins",
        "SeriesLosses",
        "FacingElimination",
        "CanClinch",
    ):
        assert train[col].tolist() == live[col].tolist(), (
            f"_game_context branch divergence on {col}"
        )

    # M-1 player-level build: the career expanding mean (full-gamelog, strict-< date),
    # its EB-shrunk + vs-opponent variants, and the opp-defense × player interaction
    # products must agree across the historical/upcoming branches. Pinned explicitly
    # so a regression points here, not just at the shared-surface loop above.
    for col in (
        "MeanYr_expanding_shifted",
        "MeanYr_expanding_eb",
        "MeanYr_expanding_vsopp",
        "PlayerExp_x_DefAvg",
        "PlayerZ_x_DefPos",
    ):
        assert np.allclose(
            train[col].to_numpy(dtype=float),
            live[col].to_numpy(dtype=float),
            rtol=0,
            atol=1e-9,
            equal_nan=True,
        ), f"M-1 feature branch divergence on {col}"

    # The absence columns (StatsNBA's hook at the end of _game_context): the official list
    # (training) and ESPN's report (serving) name the same players out on GAMEDAY. Every
    # column is non-zero somewhere, so none can agree by being empty in both arms.
    absence = list(availability.AVAILABILITY_COLUMNS)
    assert set(absence) <= set(shared), "absence columns missing from the shared surface"
    pd.testing.assert_frame_equal(train[absence], live[absence])
    assert (train[absence] != 0).any().all(), "an absence column is zero on every row"

    # StatsNBA's profile batch: minutes history and every *_48 rate pooled over the profile's
    # two windows, assists also adjusted for the defenses faced. Non-zero on every row, so
    # none can agree by being blank in both arms.
    rates = [*(stat for stat in stats.stat_types if stat.endswith("_48")), "AST_48 opp adj"]
    batch = ["Player MIN last game", "Player MIN last 3"]
    batch += [f"Player {rate} pooled{window}" for rate in rates for window in ("", " short")]
    assert set(batch) <= set(shared), "profile batch columns missing from the shared surface"
    pd.testing.assert_frame_equal(train[batch], live[batch])
    assert (train[batch] != 0).all().all(), "a profile batch column is zero on some row"

    # parity-surface extension point: a future feature batch that adds a per-gameday
    # column (or a new _game_context lookup / volume-dispatch surface) extends the
    # synthetic logs above and is automatically covered by the shared-surface assertion.


# MLB: the same gate on StatsMLB, whose serving branch takes the opposing starter from
# upcoming_games where training takes it from the gamelog row. One game a day; the starters
# throw from opposite hands and each lineup bats left, right, both and right, so every
# handedness flag and split is live in both arms.
MLB_GAMEDAY = date(2026, 6, 15)  # after the 2026 season_start, before real today
MLB_TEAMS = {"AAA": ("Arm Right", "R"), "BBB": ("Arm Left", "L")}
MLB_SIDES = ("L", "R", "S", "R")
MLB_MONEYLINE = {"AAA": 0.6, "BBB": 0.4}
MLB_TOTAL = {"AAA": 4.5, "BBB": 4.0}
# Title case, as remove_accents leaves a gamelog name.
MLB_LINEUPS = {team: [f"{team.title()} Batter {slot}" for slot in range(4)] for team in MLB_TEAMS}
MLB_BATS = {
    batter: side
    for lineup in MLB_LINEUPS.values()
    for batter, side in zip(lineup, MLB_SIDES, strict=True)
}
MLB_IDS = {
    name: 600000 + i for i, name in enumerate([*(p for p, _ in MLB_TEAMS.values()), *MLB_BATS])
}
# The counts every row draws: the box score's and the game feed's.
MLB_COUNTS = [
    *PITCH_COLUMNS,
    "pitcher strikeouts",
    "batters faced",
    "walks allowed",
    "pitches thrown",
    "plateAppearances",
    "batter strikeouts",
    "walks",
    "hits",
]
# What stats/mlb_profile.py adds for the players a market prices. The profile's and the
# lineup's each have a " short" twin.
_PITCHER_PROFILE = (
    "pitcher K rate",
    "pitcher BB rate",
    "pitcher whiff rate",
    "pitcher chase rate",
    "pitcher zone rate",
    "pitcher zone contact rate",
    "fastball mph",
    "pitches per start",
    "batters faced per start",
    "times through order",
)
_BATTER_PROFILE = (
    "batter whiff rate",
    "batter chase rate",
    "exit velocity",
    "hard-hit rate",
    "barrel rate",
    "sweet-spot rate",
    "ground-ball rate",
    "lineup spot",
    "start rate",
)
MLB_NEW_INPUTS = {
    "pitcher strikeouts": [
        *(f"Player {name}{w}" for name in _PITCHER_PROFILE for w in ("", " short")),
        *(
            f"lineup {rate} rate vs hand{w}"
            for rate in ("K", "BB", "whiff", "chase")
            for w in ("", " short")
        ),
    ],
    "hits": [
        *(f"Player {name}{w}" for name in _BATTER_PROFILE for w in ("", " short")),
        "platoon advantage",
        "facing lefty",
        "bats left",
        "switch hitter",
        "split gap",
        "PA vs hand",
        "pitcher split gap",
        "pitcher PA vs side",
    ],
}


def _mlb_opponent(team: str) -> str:
    return "BBB" if team == "AAA" else "AAA"


def _build_mlb_logs(stats: StatsMLB) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic MLB gamelog + teamlog, one AAA-BBB game a day up to ``MLB_GAMEDAY``."""
    rng = np.random.default_rng(1)
    player_stat_types, team_stat_types = stats._profile_stat_types()
    rows, team_rows = [], []
    for day in (MLB_GAMEDAY - dt.timedelta(days=k) for k in range(GAMES_BACK, -1, -1)):
        for team, (pitcher, _) in MLB_TEAMS.items():
            opponent = _mlb_opponent(team)
            opp_pitcher, opp_hand = MLB_TEAMS[opponent]
            game = {
                "gameId": day.toordinal(),
                "gameDate": day.isoformat(),
                "team": team,
                "opponent": opponent,
                "opponent pitcher": opp_pitcher,
                "opponent pitcher id": MLB_IDS[opp_pitcher],
                "opponent pitcher hand": opp_hand,
                "home": team == "AAA",
                "moneyline": MLB_MONEYLINE[team],
                "totals": MLB_TOTAL[team],
                "game type": "R",
            }
            for slot, player in enumerate([pitcher, *MLB_LINEUPS[team]]):
                rows.append(
                    game
                    | {
                        "playerId": MLB_IDS[player],
                        "playerName": player,
                        "position": "OF" if slot else "P",
                        "starting pitcher": not slot,
                        "starting batter": bool(slot),
                        "battingOrder": slot,
                    }
                    | {col: float(rng.integers(1, 30)) for col in MLB_COUNTS}
                    | {st: float(rng.uniform(0, 1)) for st in player_stat_types}
                )
            team_rows.append(
                {"gameId": day.toordinal(), "gameDate": day.isoformat(), "team": team}
                | {st: float(rng.uniform(0, 1)) for st in team_stat_types}
            )
    return pd.DataFrame(rows), pd.DataFrame(team_rows)


class _MlbStubArchive:
    """The serving branch's archive reads, echoing what ``_build_mlb_logs`` baked in."""

    def get_moneyline(self, league, game_date, team, **_):
        return MLB_MONEYLINE[team]

    def get_total(self, league, game_date, team, **_):
        return MLB_TOTAL[team]


class _FrozenMlbDatetime(dt.datetime):
    @classmethod
    def today(cls):
        return dt.datetime(MLB_GAMEDAY.year, MLB_GAMEDAY.month, MLB_GAMEDAY.day)


@pytest.mark.parametrize("market", ["pitcher strikeouts", "hits"])
def test_mlb_train_and_live_feature_paths_agree(monkeypatch, market):
    monkeypatch.setattr(base, "archive", _MlbStubArchive())
    stats = StatsMLB()
    stats.gamelog, stats.teamlog = _build_mlb_logs(stats)
    stats.comps = {"pitchers": {}, "hitters": {}}
    stats.park_factors = {team: {"R": 1.0, "K": 1.0} for team in MLB_TEAMS}
    # The stored forecasts are on disk only where data was pulled; both arms read none.
    stats._weather_history = stats._weather_forecast = pd.DataFrame(
        columns=list(WEATHER_COLUMNS["MLB"]), dtype=float
    )
    stats.players = {MLB_IDS[p]: {"name": p, "throws": hand} for p, hand in MLB_TEAMS.values()}
    stats.players |= {MLB_IDS[b]: {"name": b, "bats": side} for b, side in MLB_BATS.items()}
    stats.upcoming_games = {
        team: {
            "Pitcher": pitcher,
            "Home": team == "AAA",
            "Opponent": _mlb_opponent(team),
            "Opponent Pitcher": MLB_TEAMS[_mlb_opponent(team)][0],
            "Batting Order": MLB_LINEUPS[team],
            "Playoff": False,
            "game type": "R",
            "Game ID": str(MLB_GAMEDAY.toordinal()),
        }
        for team, (pitcher, _) in MLB_TEAMS.items()
    }
    offers = [
        {"Player": p, "Team": t, "Opponent": _mlb_opponent(t), "Date": MLB_GAMEDAY.isoformat()}
        for t, (pitcher, _) in MLB_TEAMS.items()
        for p in ([pitcher] if market == "pitcher strikeouts" else MLB_LINEUPS[t])
    ]
    columns = stats.get_stat_columns(market)
    new = MLB_NEW_INPUTS[market]
    assert set(new) <= set(columns), "an MLB input is missing from get_stat_columns"

    train = stats.get_stats(market, offers, MLB_GAMEDAY)
    # StatsMLB reads today for its playoff flag in stats/mlb.py.
    for module in (base, mlb):
        monkeypatch.setattr(module, "datetime", _FrozenMlbDatetime)
    live = stats.get_stats(market, offers, MLB_GAMEDAY)

    assert len(train) == len(offers) and len(live) == len(offers)
    shared = [c for c in columns if c in train.columns]
    pd.testing.assert_frame_equal(train.sort_index()[shared], live.sort_index()[shared])
    assert (train[new] != 0).any().all(), "an MLB input is zero on every row"
