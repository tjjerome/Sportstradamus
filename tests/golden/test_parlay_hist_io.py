"""Pins for the day-partitioned parlay history store.

``parlay_hist`` moved from one ever-growing parquet (2M rows, rewritten whole
every prophecize slot — 43% of a warm run's wall clock) to one parquet per game
date. These tests pin the partition round-trip, the upsert dedup semantics, the
legacy single-file self-migration, the retention sweep, and the partial
``days=`` write reflect uses.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.helpers import io


def _patch_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "PARLAY_HIST_PATH", tmp_path / "parlay_hist.parquet")
    monkeypatch.setattr(io, "PARLAY_HIST_DIR", tmp_path / "parlay_hist")


def _parlays(rows):
    return pd.DataFrame(
        [
            {
                "Game": game,
                "Date": day,
                "League": "MLB",
                "Platform": "Underdog",
                "Model EV": model_ev,
                "Market EV": market_ev,
                "legs": [{"Player": "Player A", "Market": "PTS"}],
                "Leg Probs": (0.55, 0.60),
                "Legs Resolved": resolved,
                "Misses": None,
            }
            for game, day, model_ev, market_ev, resolved in rows
        ]
    )


def test_read_empty_when_nothing_exists(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    assert io.read_parlay_hist().empty


def test_full_write_partitions_by_date_and_round_trips(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    df = _parlays(
        [
            ("BOS/NYY", "2026-08-01", 1.10, 1.00, 3.0),
            ("LAD/SD", "2026-08-02", 1.20, 1.05, None),
        ]
    )
    io.write_parlay_hist(df)

    assert (tmp_path / "parlay_hist" / "2026-08-01.parquet").is_file()
    assert (tmp_path / "parlay_hist" / "2026-08-02.parquet").is_file()
    back = io.read_parlay_hist()
    assert len(back) == 2
    assert tuple(back["Leg Probs"].iloc[0]) == (0.55, 0.60)
    assert back.loc[back["Date"] == "2026-08-01", "Game"].iloc[0] == "BOS/NYY"


def test_full_write_drops_days_absent_from_frame(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    io.write_parlay_hist(_parlays([("BOS/NYY", "2026-08-01", 1.10, 1.00, None)]))
    io.write_parlay_hist(_parlays([("LAD/SD", "2026-08-02", 1.20, 1.05, None)]))
    assert not (tmp_path / "parlay_hist" / "2026-08-01.parquet").exists()
    assert io.read_parlay_hist()["Date"].tolist() == ["2026-08-02"]


def test_upsert_merges_day_and_new_row_wins_dedup_collision(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    io.write_parlay_hist(
        _parlays(
            [
                ("BOS/NYY", "2026-08-01", 1.10, 1.00, 3.0),
                ("LAD/SD", "2026-08-01", 1.20, 1.05, 2.0),
            ]
        )
    )
    io.upsert_parlay_hist(
        _parlays(
            [
                ("CHC/STL", "2026-08-01", 1.10, 1.00, None),
                ("MIN/DET", "2026-08-02", 1.30, 1.15, None),
            ]
        ),
        dedup_subset=["Model EV", "Market EV"],
    )

    back = io.read_parlay_hist()
    day1 = back.loc[back["Date"] == "2026-08-01"]
    assert len(day1) == 2
    # (1.10, 1.00) collided: the fresh CHC/STL row replaced the resolved BOS/NYY one.
    assert set(day1["Game"]) == {"CHC/STL", "LAD/SD"}
    assert back.loc[back["Date"] == "2026-08-02", "Game"].tolist() == ["MIN/DET"]


def test_upsert_migrates_legacy_single_file(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    legacy = _parlays([("BOS/NYY", "2026-08-01", 1.10, 1.00, 3.0)])
    legacy["Leg Probs"] = legacy["Leg Probs"].apply(list)
    legacy.to_parquet(tmp_path / "parlay_hist.parquet", index=False)

    io.upsert_parlay_hist(
        _parlays([("LAD/SD", "2026-08-02", 1.20, 1.05, None)]),
        dedup_subset=["Model EV", "Market EV"],
    )

    assert not (tmp_path / "parlay_hist.parquet").exists()
    back = io.read_parlay_hist()
    assert sorted(back["Date"]) == ["2026-08-01", "2026-08-02"]


def test_upsert_retention_prunes_old_day_files(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    today = pd.Timestamp.today().date()
    old_day = str(today - pd.Timedelta(days=400))
    recent_day = str(today)
    io.write_parlay_hist(
        _parlays(
            [
                ("BOS/NYY", old_day, 1.10, 1.00, 3.0),
                ("LAD/SD", recent_day, 1.20, 1.05, None),
            ]
        )
    )
    io.upsert_parlay_hist(
        _parlays([("CHC/STL", recent_day, 1.30, 1.15, None)]),
        dedup_subset=["Model EV", "Market EV"],
        retention_days=365,
    )
    assert not (tmp_path / "parlay_hist" / f"{old_day}.parquet").exists()
    assert set(io.read_parlay_hist()["Date"]) == {recent_day}


def test_days_write_touches_only_listed_partitions(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    df = _parlays(
        [
            ("BOS/NYY", "2026-08-01", 1.10, 1.00, None),
            ("LAD/SD", "2026-08-02", 1.20, 1.05, None),
        ]
    )
    io.write_parlay_hist(df)

    df.loc[df["Date"] == "2026-08-02", "Legs Resolved"] = 2.0
    df.loc[df["Date"] == "2026-08-01", "Legs Resolved"] = 99.0  # must NOT persist
    io.write_parlay_hist(df, days=["2026-08-02"])

    back = io.read_parlay_hist()
    assert back.loc[back["Date"] == "2026-08-02", "Legs Resolved"].iloc[0] == 2.0
    assert back.loc[back["Date"] == "2026-08-01", "Legs Resolved"].isna().all()


def test_read_column_projection_spans_partitions_and_legacy(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    legacy = _parlays([("BOS/NYY", "2026-08-01", 1.10, 1.00, 3.0)])
    legacy["Leg Probs"] = legacy["Leg Probs"].apply(list)
    legacy.to_parquet(tmp_path / "parlay_hist.parquet", index=False)
    io.write_parlay_hist(
        _parlays([("LAD/SD", "2026-08-02", 1.20, 1.05, None)]), days=["2026-08-02"]
    )

    back = io.read_parlay_hist(columns=["Date", "Model EV"])
    assert list(back.columns) == ["Date", "Model EV"]
    assert sorted(back["Date"]) == ["2026-08-01", "2026-08-02"]
