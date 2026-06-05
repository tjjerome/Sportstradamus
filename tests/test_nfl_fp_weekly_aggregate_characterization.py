"""Characterization pins for the over-CC functions in ``nfl_fp_weekly_aggregate``:
the four bucket aggregators (``_aggregate_separation_by_routes`` /
``_aggregate_per_coverage_yprr`` / ``_aggregate_separation_by_coverage`` /
``_aggregate_man_vs_zone_bucket``) and the ``load_multi_window_one_year`` orchestrator.

The aggregators are pinned on synthetic input with ``_load_kind_multi`` monkeypatched
(no disk), freezing the weighted-mean-per-bucket output so the shared-helper DRY
extraction is provably behavior-preserving. ``load_multi_window_one_year`` is pinned
nightly.run-style: every sub-helper is a recording seam returning a known tiny frame,
so the test fixes the combine-first order + finalize wiring (which the decomposition
into ``_combine_bucket_aggregates`` + ``_finalize_player_frame`` must reproduce).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import sportstradamus.stats.nfl_fp_weekly_aggregate as m

PGC = m.PLAYER_GROUP_COL
_W = [(2024, 1, 18)]


def _bucket_df(payload_by_player: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            PGC: list(payload_by_player),
            "bucket": [json.dumps(p) for p in payload_by_player.values()],
        }
    )


def _patch_loader(monkeypatch, kind_map: dict):
    monkeypatch.setattr(
        m, "_load_kind_multi", lambda windows, kind: kind_map.get(kind, pd.DataFrame())
    )


def test_aggregate_separation_by_routes(monkeypatch):
    routes = list(m._ROUTE_BUCKET_ORDER[:3])
    df = _bucket_df(
        {
            "p1": {
                routes[0]: {m._SEP_BY_ROUTES_VALUE_KEY: 0.40, m._SEP_BY_ROUTES_WEIGHT_KEY: 10},
                routes[1]: {m._SEP_BY_ROUTES_VALUE_KEY: 0.55, m._SEP_BY_ROUTES_WEIGHT_KEY: 5},
            },
            "p2": {
                routes[0]: {m._SEP_BY_ROUTES_VALUE_KEY: 0.30, m._SEP_BY_ROUTES_WEIGHT_KEY: 8},
                "UnknownRoute": {m._SEP_BY_ROUTES_VALUE_KEY: 0.9, m._SEP_BY_ROUTES_WEIGHT_KEY: 2},
            },
        }
    )
    _patch_loader(monkeypatch, {m._SEP_BY_ROUTES_KIND: df})
    out = m._aggregate_separation_by_routes(_W)

    assert list(out.columns) == [
        "rec_sep_route_SEP_SCORE",
        "rec_sep_route_SEP_SCORE.1",
        "rec_sep_route_SEP_SCORE.2",
    ]
    assert out.loc["p1", "rec_sep_route_SEP_SCORE"] == pytest.approx(0.40)
    assert out.loc["p1", "rec_sep_route_SEP_SCORE.1"] == pytest.approx(0.55)
    assert out.loc["p2", "rec_sep_route_SEP_SCORE"] == pytest.approx(0.30)
    assert out.loc["p2", "rec_sep_route_SEP_SCORE.2"] == pytest.approx(
        0.90
    )  # unknown route appended
    assert pd.isna(out.loc["p1", "rec_sep_route_SEP_SCORE.2"])


def test_aggregate_per_coverage_yprr(monkeypatch):
    cols = {PGC: ["p1", "p2"]}
    for schemes in m._COVERAGE_YPRR_SCHEMES.values():
        for scheme in schemes:
            cols[f"playerStatsCoverageScheme{scheme}ReceivingYardsPerRoute"] = [1.2, 0.8]
            cols[f"playerStatsCoverageScheme{scheme}ReceivingRoutesTotal"] = [10, 5]
    _patch_loader(monkeypatch, {"wr_coverage_matchup": pd.DataFrame(cols)})
    out = m._aggregate_per_coverage_yprr(_W)

    assert set(out.columns) == {f"rec_yprr_vs_{b}" for b in m._COVERAGE_YPRR_SCHEMES}
    assert out.loc["p1", "rec_yprr_vs_man"] == pytest.approx(1.2)
    assert out.loc["p2", "rec_yprr_vs_zone"] == pytest.approx(0.8)


def test_aggregate_separation_by_coverage(monkeypatch):
    schemes = list(m._SEP_BY_COVERAGE_SCHEMES.items())
    df = _bucket_df(
        {
            "p1": {
                schemes[0][1]: {
                    m._SEP_BY_COVERAGE_VALUE_KEY: 0.5,
                    m._SEP_BY_COVERAGE_WEIGHT_KEY: 20,
                },
                schemes[1][1]: {
                    m._SEP_BY_COVERAGE_VALUE_KEY: 0.6,
                    m._SEP_BY_COVERAGE_WEIGHT_KEY: 10,
                },
            },
            "p2": {
                schemes[0][1]: {
                    m._SEP_BY_COVERAGE_VALUE_KEY: 0.45,
                    m._SEP_BY_COVERAGE_WEIGHT_KEY: 15,
                }
            },
        }
    )
    _patch_loader(monkeypatch, {m._SEP_BY_COVERAGE_KIND: df})
    out = m._aggregate_separation_by_coverage(_W)

    first, second = schemes[0][0], schemes[1][0]
    assert out.loc["p1", f"rec_sep_vs_{first}"] == pytest.approx(0.5)
    assert out.loc["p1", f"rec_sep_vs_{second}"] == pytest.approx(0.6)
    assert out.loc["p2", f"rec_sep_vs_{first}"] == pytest.approx(0.45)
    assert pd.isna(out.loc["p2", f"rec_sep_vs_{second}"])


def test_aggregate_man_vs_zone_bucket(monkeypatch):
    shells = list(m._MZ_BUCKETS.items())
    df = _bucket_df(
        {
            "p1": {
                shells[0][1]: {m._MZ_VALUE_KEY: 1.5, m._MZ_WEIGHT_KEY: 30},
                shells[1][1]: {m._MZ_VALUE_KEY: 2.0, m._MZ_WEIGHT_KEY: 20},
            },
        }
    )
    _patch_loader(monkeypatch, {m._MZ_BUCKET_KIND: df})
    out = m._aggregate_man_vs_zone_bucket(_W)

    assert out.loc["p1", f"rec_mz_YPRR_{shells[0][0]}"] == pytest.approx(1.5)
    assert out.loc["p1", f"rec_mz_YPRR_{shells[1][0]}"] == pytest.approx(2.0)


def test_load_multi_window_one_year_wiring(monkeypatch):
    """Pin the orchestration: base recipe + 5 bucket aggregates combine-first'd in
    order, then the metadata/games/derive/POS-filter/pbp finalize chain."""
    base = pd.DataFrame({"recipe_col": [1.0, 2.0]}, index=["p1", "p2"])
    monkeypatch.setattr(m, "_aggregate_recipes", lambda w: base.copy())
    # five bucket aggregates, each contributing one distinct column
    monkeypatch.setattr(
        m,
        "_aggregate_separation_by_routes",
        lambda w: pd.DataFrame({"rt": [0.1, 0.2]}, index=["p1", "p2"]),
    )
    monkeypatch.setattr(
        m,
        "_aggregate_per_coverage_yprr",
        lambda w: pd.DataFrame({"yprr": [0.3, 0.4]}, index=["p1", "p2"]),
    )
    monkeypatch.setattr(
        m,
        "_aggregate_separation_by_coverage",
        lambda w: pd.DataFrame({"cov": [0.5, 0.6]}, index=["p1", "p2"]),
    )
    monkeypatch.setattr(
        m,
        "_aggregate_separation_by_alignment_bucket",
        lambda w: pd.DataFrame({"align": [0.7, 0.8]}, index=["p1", "p2"]),
    )
    monkeypatch.setattr(
        m,
        "_aggregate_man_vs_zone_bucket",
        lambda w: pd.DataFrame({"mz": [0.9, 1.0]}, index=["p1", "p2"]),
    )
    monkeypatch.setattr(
        m, "_player_metadata", lambda w: pd.DataFrame({"POS": ["WR", "RB"]}, index=["p1", "p2"])
    )
    monkeypatch.setattr(m, "_player_game_counts", lambda w: pd.Series([3, 4], index=["p1", "p2"]))
    monkeypatch.setattr(m, "_project_to_name_index", lambda df: df)
    monkeypatch.setattr(m, "_derive_aggregate_metrics", lambda df: df)
    monkeypatch.setattr(m, "_broadcast_team_coverage", lambda df, w: df)
    monkeypatch.setattr(m, "derive_comp_metrics", lambda df: df)
    monkeypatch.setattr(m, "_load_pbp_named_by_player", lambda season: None)
    monkeypatch.setattr(m, "_join_nfl_players", lambda df: df)

    out = m.load_multi_window_one_year(_W)

    for col in ("recipe_col", "rt", "yprr", "cov", "align", "mz"):
        assert col in out.columns, f"missing {col}"
    assert "position" in out.columns and "player_game_count" in out.columns
    assert out.loc["p1", "player_game_count"] == 3
    assert set(out["position"]) == {"WR", "RB"}


def test_load_multi_window_empty_windows():
    assert m.load_multi_window_one_year([(2024, 18, 1)]).empty  # start > end -> filtered
