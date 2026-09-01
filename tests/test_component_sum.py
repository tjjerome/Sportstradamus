"""Contract tests for Lane B's component-sum assembler.

Covers the three things that can silently go wrong: the spec a combo cell resolves
to, the per-row decode of each component family into the kernel's parameterization,
and the endpoint columns the ship scorecard reads back off the sampled sum.
"""

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.combined_markets import ComboComponent, combo_sum_quote
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training import scorecard
from sportstradamus.training.component_cells import (
    SHAPE_FIELDS,
    ComponentCell,
    load_component_cell,
    spec_weights,
)
from sportstradamus.training.component_sum import _model_rho, component_sum_frame

# Sobol at 8192 draws resolves a marginal CDF far tighter than the 0.006 tail bound
# tests/test_combined_markets.py uses for a multi-component sum.
_MARGINAL_TOL = 2e-3

_DATES = [f"2026-01-{day:02d}" for day in range(1, 6)]


def _rho_zero(_a, _b):
    return 0.0


def _base_frame(players, dates, results):
    """The ride-along columns every dumped test-set CSV carries."""
    return pd.DataFrame(
        {
            "Player": players,
            "Date": dates,
            "Result": results,
            "Line": np.full(len(results), 2.5),
            "Odds": np.full(len(results), -110.0),
            "MeanYr": np.full(len(results), 2.0),
            "Mean10": np.full(len(results), 2.0),
            "EV": np.full(len(results), 2.0),
            "Blended_EV": np.full(len(results), 2.0),
            "P": np.full(len(results), 0.5),
        }
    )


def _write(root, league, market, frame):
    frame.to_csv(root / f"{market_file_slug(league, market)}.csv", index=False)


def _repeat(values, times):
    return np.tile(np.asarray(values, dtype=float), times)


# --- spec resolution -------------------------------------------------------


def test_combo_props_spec_resolves_without_a_loaded_stats():
    combo = pd.DataFrame({"Player": ["A", "B"]})
    specs, offset, provenance = spec_weights("NBA", "BLST", combo)

    assert offset == 0.0
    assert provenance == ["combo_props"]
    assert specs == {"A": (("BLK", 1.0), ("STL", 1.0)), "B": (("BLK", 1.0), ("STL", 1.0))}


def test_nba_fantasy_spec_resolves_through_the_league_table():
    combo = pd.DataFrame({"Player": ["A"]})
    specs, _, provenance = spec_weights("NBA", "fantasy points prizepicks", combo)

    assert provenance == ["fantasy_combo_spec"]
    assert specs["A"] == (
        ("PTS", 1),
        ("REB", 1.2),
        ("AST", 1.5),
        ("BLK", 3),
        ("STL", 3),
        ("TOV", -1),
    )


def test_nfl_fantasy_spec_is_per_position_and_swaps_qb_tds():
    combo = pd.DataFrame({"Player": ["Q", "W", "R", "T"], "Player position": [1.0, 2.0, 3.0, 4.0]})
    specs, _, provenance = spec_weights("NFL", "fantasy points underdog", combo)

    assert provenance == ["fantasy_combo_spec", "qb_tds_via_rushing_tds"]
    assert dict(specs["Q"])["rushing tds"] == 6.0
    assert "tds" not in dict(specs["Q"])
    assert dict(specs["W"])["tds"] == 6.0
    assert dict(specs["W"])["receptions"] == 0.5
    assert set(dict(specs["R"])) == {"rushing yards", "receiving yards", "receptions", "tds"}
    assert set(dict(specs["T"])) == {"receiving yards", "receptions", "tds"}


def test_mlb_hitter_fantasy_prices_hit_types_directly_and_drops_triples():
    combo = pd.DataFrame({"Player": ["A"]})
    specs, offset, provenance = spec_weights("MLB", "hitter fantasy points underdog", combo)

    assert provenance == ["hit_types_direct", "hbp_offset", "omitted_unmodeled:triples"]
    assert offset == pytest.approx(3.0 * 0.045)
    assert dict(specs["A"]) == {
        "singles": 3.0,
        "doubles": 6.0,
        "home runs": 10.0,
        "walks": 3.0,
        "rbi": 2.0,
        "runs": 2.0,
        "stolen bases": 4.0,
    }


def test_spec_with_bernoulli_or_post_terms_is_refused():
    combo = pd.DataFrame({"Player": ["A"]})
    specs, _, provenance = spec_weights("MLB", "pitcher fantasy points underdog", combo)

    assert specs == {}
    assert "needs terms no component cell models" in provenance[0]


# --- per-row decode --------------------------------------------------------


def _decode_case(tmp_path, league, market, params):
    """Round-trip one component cell: CSV -> ComponentCell -> kernel marginal."""
    frame = _base_frame(["A"] * 5, _DATES, [1.0] * 5).assign(**params)
    _write(tmp_path, league, market, frame)
    cell = load_component_cell(league, market, tmp_path / f"{market_file_slug(league, market)}.csv")
    row = cell.params.iloc[0]
    component = ComboComponent(
        market,
        1.0,
        float(row["mean"]),
        cell.dist,
        cell.cv,
        **{f: None if np.isnan(row[f]) else float(row[f]) for f in SHAPE_FIELDS},
    )
    draws = combo_sum_quote([component], _rho_zero).draws_sorted
    loaded = scorecard.load_test_set(
        tmp_path / f"{market_file_slug(league, market)}.csv", "Blended_EV"
    )
    strategy = scorecard._decode_strategy_for_frame(loaded, league, market)
    return cell, draws, loaded.iloc[[0]], strategy


def _assert_matches_scorecard_cdf(draws, one_row, dist, strategy, grid):
    for y in grid:
        cdf, _ = scorecard._pred_cdf_pmf(one_row, dist, np.array([y]), strategy=strategy)
        assert np.mean(draws <= y) == pytest.approx(float(cdf[0]), abs=_MARGINAL_TOL)


def test_zinb_component_decode_matches_the_scorecard_cdf(tmp_path):
    cell, draws, one_row, strategy = _decode_case(
        tmp_path,
        "NBA",
        "BLK",
        {"R": 2.5, "NB_P": 0.35, "Gate": 0.3},
    )

    assert cell.dist == "ZINB"
    assert cell.params["mean"].iloc[0] == pytest.approx(2.5 * 0.35 / 0.65)
    assert cell.params["gate"].iloc[0] == pytest.approx(0.3)
    _assert_matches_scorecard_cdf(draws, one_row, "ZINB", strategy, [0, 1, 2, 3, 5, 8])


def test_negbin_component_decode_matches_the_scorecard_cdf(tmp_path):
    cell, draws, one_row, strategy = _decode_case(tmp_path, "NBA", "REB", {"R": 4.0, "NB_P": 0.5})

    assert cell.dist == "NegBin"
    assert cell.params["mean"].iloc[0] == pytest.approx(4.0)
    assert np.isnan(cell.params["gate"].iloc[0])
    _assert_matches_scorecard_cdf(draws, one_row, "NegBin", strategy, [0, 1, 3, 5, 9, 14])


def test_dpo_component_decode_matches_the_scorecard_cdf(tmp_path):
    cell, draws, one_row, strategy = _decode_case(
        tmp_path, "NBA", "STL", {"DP_MU": 1.4, "DP_PHI": 0.9}
    )

    assert cell.dist == "DPO"
    # DP_MU is the natural parameter, not the mean; the exact series mean differs.
    assert cell.params["mean"].iloc[0] != pytest.approx(1.4, abs=1e-6)
    _assert_matches_scorecard_cdf(draws, one_row, "DPO", strategy, [0, 1, 2, 3, 5, 7])


def test_skewnormal_component_decode_matches_the_scorecard_cdf(tmp_path):
    cell, draws, one_row, strategy = _decode_case(
        tmp_path,
        "NBA",
        "PTS",
        {"SN_Loc": 4.0, "SN_Scale": 6.0, "SN_Alpha": 1.5, "Gate": 0.08},
    )

    assert cell.dist == "SkewNormal"
    loc, scale = scorecard._decode_sn_loc_scale(one_row, strategy)
    delta = 1.5 / np.sqrt(1 + 1.5**2)
    assert cell.params["mean"].iloc[0] == pytest.approx(
        loc[0] + scale[0] * delta * np.sqrt(2 / np.pi)
    )
    _assert_matches_scorecard_cdf(
        draws, one_row, "SkewNormal", strategy, [0.0, 2.0, 6.0, 12.0, 20.0]
    )


def test_a_gamma_component_is_refused_rather_than_mispriced(tmp_path):
    frame = _base_frame(["A"] * 5, _DATES, [1.0] * 5).assign(Alpha=3.0)
    _write(tmp_path, "NBA", "MIN", frame)

    with pytest.raises(ValueError, match="cannot price a 'Gamma' cell"):
        load_component_cell("NBA", "MIN", tmp_path / "NBA_MIN.csv")


# --- rho estimation --------------------------------------------------------


def _score_cell(market, values, index):
    return ComponentCell(market, "DPO", 0.5, pd.DataFrame({"z": values}, index=index))


def test_model_rho_recovers_a_planted_correlation():
    rng = np.random.default_rng(7)
    n, planted = 4000, 0.55
    z1 = rng.standard_normal(n)
    z2 = planted * z1 + np.sqrt(1 - planted**2) * rng.standard_normal(n)
    index = pd.MultiIndex.from_arrays([["A"] * n, [f"d{i}" for i in range(n)]])
    cells = {"a": _score_cell("a", z1, index), "b": _score_cell("b", z2, index)}

    rho, reported = _model_rho(cells, pd.MultiIndex.from_tuples([("A", "d0")]), 200)

    assert rho("a", "b") == pytest.approx(planted, abs=0.05)
    assert rho("b", "a") == rho("a", "b")
    assert rho("a", "a") == 1.0
    assert reported == {"a|b": pytest.approx(planted, abs=0.05)}


def test_model_rho_falls_back_to_independence_below_the_pair_floor():
    rng = np.random.default_rng(7)
    n = 50
    z1 = rng.standard_normal(n)
    index = pd.MultiIndex.from_arrays([["A"] * n, [f"d{i}" for i in range(n)]])
    cells = {"a": _score_cell("a", z1, index), "b": _score_cell("b", z1, index)}

    rho, _ = _model_rho(cells, pd.MultiIndex.from_tuples([("A", "d0")]), 200)

    assert rho("a", "b") == 0.0


def test_model_rho_excludes_the_graded_rows():
    rng = np.random.default_rng(11)
    n = 3000
    z1 = rng.standard_normal(n)
    index = pd.MultiIndex.from_arrays([["A"] * n, [f"d{i}" for i in range(n)]])
    # Perfectly anti-correlated inside the graded block, independent outside it.
    z2 = rng.standard_normal(n)
    z2[:1000] = -z1[:1000]
    cells = {"a": _score_cell("a", z1, index), "b": _score_cell("b", z2, index)}

    rho, _ = _model_rho(cells, index[:1000], 200)

    assert rho("a", "b") == pytest.approx(0.0, abs=0.05)


# --- assembled frame -------------------------------------------------------


@pytest.fixture(name="blst_dir")
def _blst_dir(tmp_path):
    """A synthetic NBA BLST cell: five outcomes per player over identical params."""
    players = np.repeat(["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"], 5)
    dates = _repeat(range(5), 8)
    dates = [f"2026-01-{int(d) + 1:02d}" for d in dates]
    results = _repeat([0, 1, 2, 3, 4], 8)
    _write(
        tmp_path, "NBA", "BLST", _base_frame(players, dates, results).assign(DP_MU=1.2, DP_PHI=0.8)
    )
    component_rows = _base_frame(players, dates, results)
    _write(
        tmp_path,
        "NBA",
        "BLK",
        component_rows.assign(R=np.repeat(np.linspace(1.5, 4.0, 8), 5), NB_P=0.3, Gate=0.2),
    )
    _write(
        tmp_path,
        "NBA",
        "STL",
        component_rows.assign(DP_MU=np.repeat(np.linspace(0.6, 1.6, 8), 5), DP_PHI=0.9),
    )
    return tmp_path


def test_emitted_endpoints_are_internally_consistent(blst_dir):
    candidate, baseline, diagnostics = component_sum_frame("NBA", "BLST", test_sets_dir=blst_dir)

    assert diagnostics["reason"] is None
    assert diagnostics["graded_rows"] == 40
    assert len(candidate) == len(baseline) == 40
    assert scorecard._infer_dist_from_columns(candidate) == scorecard.COMPONENT_SUM_DIST

    assert (candidate["SUM_Q10"] <= candidate["SUM_Q25"]).all()
    assert (candidate["SUM_Q25"] <= candidate["SUM_Q75"]).all()
    assert (candidate["SUM_Q75"] <= candidate["SUM_Q90"]).all()
    assert (candidate["SUM_CDF"] - candidate["SUM_PMF"] >= 0).all()
    assert candidate["SUM_CDF"].between(0.0, 1.0).all()
    for _, block in candidate.groupby("Player"):
        ordered = block.sort_values("Result")["SUM_CDF"].to_numpy()
        assert np.all(np.diff(ordered) >= 0)


def test_candidate_drops_the_incumbent_predictive_and_both_frames_drop_identity(blst_dir):
    candidate, baseline, _ = component_sum_frame("NBA", "BLST", test_sets_dir=blst_dir)

    assert not {"DP_MU", "DP_PHI"} & set(candidate.columns)
    assert {"DP_MU", "DP_PHI"} <= set(baseline.columns)
    assert not any(column.startswith("Strategy") for column in baseline.columns)
    assert not np.allclose(candidate["Blended_EV"], baseline["Blended_EV"])
    assert list(candidate["Player"]) == list(baseline["Player"])
    assert list(candidate["Date"]) == list(baseline["Date"])


def test_model_rho_arm_reports_its_pair_estimates(blst_dir):
    _, _, diagnostics = component_sum_frame(
        "NBA", "BLST", test_sets_dir=blst_dir, rho_source="model"
    )

    # Every row is graded here, so no out-of-sample pairs remain to estimate from.
    assert diagnostics["rho"] == {"BLK|STL": 0.0}


def test_missing_component_csv_yields_an_empty_result_with_a_reason(tmp_path):
    _write(
        tmp_path,
        "NBA",
        "BLST",
        _base_frame(["A"] * 5, _DATES, [1.0] * 5).assign(DP_MU=1.2, DP_PHI=0.8),
    )

    candidate, baseline, diagnostics = component_sum_frame("NBA", "BLST", test_sets_dir=tmp_path)

    assert candidate.empty and baseline.empty
    assert diagnostics["reason"] == "no test set for component(s): BLK, STL"


def test_a_thin_join_is_reported_rather_than_graded(blst_dir):
    candidate, _, diagnostics = component_sum_frame(
        "NBA", "BLST", test_sets_dir=blst_dir, min_rows=100
    )

    assert candidate.empty
    assert diagnostics["reason"] == "only 40 of 40 rows join every component"
