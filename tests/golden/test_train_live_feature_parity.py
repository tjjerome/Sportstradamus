"""Train/live feature-parity gate (model track §6.0.4, the build-first blocker).

``get_training_matrix`` (``stats/base.py`` ``get_training_matrix``) and the live
serving path both compute features through the **same** ``Stats.get_stats``:
training calls it per gameday inside the matrix loop, serving calls it from
``prediction/scoring.py`` ``match_offers`` / ``prediction/model_prob.py``. With the
``date`` argument fixed to a gameday ``D`` every feature helper is keyed on ``date``
and is identical across the two paths -- except ``_game_context``, which branches on
``date < datetime.today().date()``:

* historical (training) branch -- reads ``Home`` / ``Moneyline`` / ``Total`` from the
  realized gamelog row for ``D``;
* upcoming (serving) branch -- reads ``Home`` from ``self.upcoming_games`` and
  ``Moneyline`` / ``Total`` from ``archive.get_moneyline`` / ``archive.get_total``.

The gamelog's ``moneyline`` / ``totals`` were themselves baked from that same archive
at load time, so the invariant under test is *archive-baked-at-train ==
archive-read-at-serve*. This is the "edit both branches" risk that every game-context
feature carries (``docs/handoffs/model_improvement_track.md`` §6.3.1, §7.2 step 2);
the doc gates all later feature work behind this harness existing.

The test drives the real ``StatsNBA.get_stats`` twice on one synthetic gameday,
flipping **only** that branch (real today ``> D`` vs a frozen today ``== D``), and
asserts the shared feature surface (``get_stat_columns``) matches. Synthetic gamelog
data, a stubbed archive, pre-seeded comps, and an injected depth keep it fast,
deterministic, and free of network / DuckDB -- each of those is branch-*independent*
state shared by both arms, so none can mask the branch divergence under test. The
``fillna`` asymmetry between the two real paths (training does
``fillna(0).replace([inf,-inf],0)``, serving does ``fillna(0)`` only) is covered by
the no-inf assertion: a shared column that can be ``inf`` would diverge, so the gate
forbids one.
"""

from __future__ import annotations

import datetime as dt
from datetime import date

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import StatsNBA, base

MARKET = "PTS"
GAMEDAY = date(2025, 12, 1)  # after the NBA season_start; far in the past vs real today
TEAMS = ("AAA", "BBB")
PLAYERS = {f"Player {i}": (TEAMS[0] if i < 5 else TEAMS[1]) for i in range(10)}
POSITIONS = ("P", "C", "F", "W", "B")
GAMES_BACK = 15

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

    Columns follow ``StatsNBA.log_strings`` plus the per-cell stat-type panels and the
    archive-baked ``moneyline`` / ``totals``. Values are deterministic (seeded RNG);
    only their *parity across the two branches* matters, never their realism.
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
                "PLAYER_NAME": player,
                "MIN": float(rng.uniform(15, 35)),
                "USG_PCT": float(rng.uniform(0.1, 0.3)),
                "POS": player_pos[player],
                "TEAM_ABBREVIATION": team,
                "OPP": _opponent(team),
                "HOME": TEAM_HOME[team],
                "WL": "W",
                "PTS": float(rng.uniform(5, 30)),
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


@pytest.fixture
def primed_stats(monkeypatch) -> tuple[StatsNBA, list[str]]:
    """A ``StatsNBA`` (with synthetic logs) and its ``get_stat_columns``, both arms ready.

    Fetches ``get_stat_columns`` first (it refreshes the base profile at *today*),
    then establishes the profile at ``GAMEDAY`` and injects ``depth`` -- normally set
    by the external ``get_depth`` volume step that gates ``_join_profiles``. The
    ``profile_market`` cache guard then lets both arms reuse this profile without a
    rebuild, so callers must not touch ``get_stat_columns`` / ``base_profile`` between
    the arms -- which is why the column list is resolved here, not in the test.
    """
    monkeypatch.setattr(base, "archive", _StubArchive())
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

    # Serving arm: freeze today == GAMEDAY -> _game_context upcoming branch.
    monkeypatch.setattr(base, "datetime", _FrozenDatetime)
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

    # fillna asymmetry guard: training zeros inf (base.py replace), serving keeps it
    # (model_prob fillna only). Forbid inf on the shared surface so the two agree.
    inf_cols = [
        c
        for c in shared
        if pd.api.types.is_numeric_dtype(train[c])
        and (
            np.isinf(train[c].to_numpy(dtype=float)).any()
            or np.isinf(live[c].to_numpy(dtype=float)).any()
        )
    ]
    assert not inf_cols, f"shared feature columns contain inf (train zeros it, serve does not): {inf_cols}"

    # The _game_context invariant, named explicitly so a regression points here.
    # OppTotal + the derived Spread/GameTotal/Blowout (QW-1) ride the same
    # historical-gamelog vs upcoming-archive divergence as Total, so they pin here too;
    # Playoff (game-ID season-type decode) and the series-context columns
    # (SeriesWins/SeriesLosses/FacingElimination/CanClinch) are the same _game_context
    # surface -- regular-season synthetic logs short-circuit them to 0 in both branches.
    for col in (
        "Home", "Moneyline", "Total", "OppTotal", "Spread", "GameTotal", "Blowout",
        "Playoff", "SeriesWins", "SeriesLosses", "FacingElimination", "CanClinch",
    ):
        assert train[col].tolist() == live[col].tolist(), f"_game_context branch divergence on {col}"

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

    # parity-surface extension point: a future feature batch that adds a per-gameday
    # column (or a new _game_context lookup / volume-dispatch surface) extends the
    # synthetic logs above and is automatically covered by the shared-surface assertion.
