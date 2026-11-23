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

The test drives the real ``get_stats`` of ``StatsNBA`` and of ``StatsWNBA`` twice on one
synthetic gameday, flipping **only** that branch (real today ``> D`` vs a frozen today
``== D``), and asserts the shared feature surface (``get_stat_columns``) matches. Synthetic gamelog
data, a stubbed archive, absence sources written under ``tmp_path``, pre-seeded comps,
and an injected depth keep it fast, deterministic, and free of network / DuckDB -- each
of those is branch-*independent* state shared by both arms, so none can mask the branch
divergence under test. The ``inf`` handling is covered by the no-inf assertion:
training (``get_training_matrix``) and the model's serving path (``match_offers``) both
end ``fillna(0).replace([inf,-inf],0)``, but the book fallback (``book_fallback_prob``)
joins its features after ``fillna(0)`` only, so a shared column that can be ``inf``
would diverge there, and the gate forbids one.
"""

from __future__ import annotations

import datetime as dt
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import StatsNBA, StatsWNBA, availability, base, nba

MARKET = "PTS"
GAMEDAY = date(2025, 12, 1)  # after the NBA season_start; far in the past vs real today
TEAMS = ("AAA", "BBB")
# Player 6's gamelog name keeps its accent, as some real ones do: ESPN's report reaches his
# gamelog id only through the accent-free spelling.
PLAYERS = {
    ("Plàyer 6" if i == 6 else f"Player {i}"): (TEAMS[0] if i < 5 else TEAMS[1]) for i in range(10)
}
PLAYER_IDS = {player: 1000 + i for i, player in enumerate(PLAYERS)}
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


def _build_logs(stat_types: list[str], team_stat_types: list[str], positions: list[str]):
    """Synthetic NBA or WNBA gamelog + teamlog covering ``GAMES_BACK`` days up to ``GAMEDAY``.

    Columns follow ``StatsNBA.log_strings`` plus the per-cell stat-type panels, the
    archive-baked ``moneyline`` / ``totals``, the ``PLAYER_ID`` and ``AST`` / ``REB`` /
    ``FGA`` the absence weights read, and the other counts the profile pools per regulation
    game. Values are deterministic (seeded RNG); only their *parity across the two branches*
    matters, never their realism.
    """
    rng = np.random.default_rng(0)
    game_dates = [GAMEDAY - dt.timedelta(days=k) for k in range(GAMES_BACK, -1, -1)]
    player_pos = {p: positions[i % len(positions)] for i, p in enumerate(PLAYERS)}

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
    comps: dict = {pos: {} for pos in stats.positions}
    for i, player in enumerate(names):
        partner = names[(i + 5) % len(names)]
        position = stats.positions[i % len(stats.positions)]
        comps[position][player] = {"comps": [partner], "distances": [0.5]}
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


@pytest.fixture(params=[StatsNBA, StatsWNBA], ids=["NBA", "WNBA"])
def primed_stats(request, monkeypatch, tmp_path) -> tuple[StatsNBA, list[str]]:
    """A ``StatsNBA`` or ``StatsWNBA`` (with synthetic logs) and its ``get_stat_columns``.

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
    stats = request.param()
    player_stat_types, team_stat_types = stats._profile_stat_types()
    stats.gamelog, stats.teamlog = _build_logs(player_stat_types, team_stat_types, stats.positions)
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
    # column is non-zero somewhere, so none can agree by being empty in both arms. The WNBA's
    # absences are unscreened.
    absence = list(availability.AVAILABILITY_COLUMNS)
    if stats.league == "NBA":
        assert set(absence) <= set(shared), "absence columns missing from the shared surface"
        pd.testing.assert_frame_equal(train[absence], live[absence])
        assert (train[absence] != 0).any().all(), "an absence column is zero on every row"

    # StatsNBA's profile batch: minutes history and every per-minute rate (per 48 in the NBA,
    # per 40 in the WNBA) pooled over the profile's two windows, assists also adjusted for the
    # defenses faced. Non-zero on every row, so none can agree by being blank in both arms.
    suffix = f"_{stats.regulation_minutes}"
    rates = [*(stat for stat in stats.stat_types if stat.endswith(suffix)), f"AST{suffix} opp adj"]
    batch = ["Player MIN last game", "Player MIN last 3"]
    batch += [f"Player {rate} pooled{window}" for rate in rates for window in ("", " short")]
    assert set(batch) <= set(shared), "profile batch columns missing from the shared surface"
    pd.testing.assert_frame_equal(train[batch], live[batch])
    assert (train[batch] != 0).all().all(), "a profile batch column is zero on some row"

    # parity-surface extension point: a future feature batch that adds a per-gameday
    # column (or a new _game_context lookup / volume-dispatch surface) extends the
    # synthetic logs above and is automatically covered by the shared-surface assertion.
