"""Pins the shared volume-model load head every league's ``get_volume_stats`` uses.

``Stats.load_volume_model_params`` flattens offers -> profile_market/get_depth/
get_stats -> loads the ``models/{league}_{market}.mdl`` pickle (cached) -> slices
to ``expected_columns`` -> casts ``Home``/``Player position`` to category ->
``set_model_start_values`` -> ``model.predict`` -> decodes -> writes
``proj {market} mean`` / ``proj {market} std`` into ``playerProfile``.

Both the seeding and the decode read the artifact's own ``distribution`` /
``normalized`` / ``target_normalization`` / ``offset_meta``, which is what these
tests exist to pin: this path previously hard-coded a SkewNormal ``loc``/
``scale``/``alpha`` rename, so the DPO ``carries`` model produced no ``loc`` at
all and served nothing, and no normalization was ever undone.

The trained pickles are not needed and are not on CI boxes: a stub ``filedict``
goes straight into ``_volume_model_cache``, so any family or normalization can be
exercised with no pickle and no network.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.distributions import _dp_mean
from sportstradamus.prediction.model_prob import _decode_skewnormal
from sportstradamus.stats import base
from sportstradamus.stats.base import Stats
from sportstradamus.stats.nba import StatsNBA

_DATE = "2024-01-01"
# Offers shaped like the real call: a per-team mapping plus a per-market bucket,
# both folded into one flat dict by the head's offer-flatten preamble.
_OFFERS = {"BOS": {"player a": {"Line": 1.5}}, "timeOnIce": {"player b": {"Line": 2.5}}}
_SEED_COLUMNS = ["MeanYr", "STDYr", "ZeroYr"]


@pytest.fixture
def _stub_start_values(monkeypatch):
    # The seeding is exercised for real by the seeding tests; elsewhere a no-op
    # removes the MeanYr/STDYr/ZeroYr coupling so only the head is under test.
    monkeypatch.setattr(base, "set_model_start_values", lambda *a, **k: None)


class _StubModel:
    def __init__(self, params):
        self._params = params
        self.seen = None
        self.start_values = None

    def predict(self, player_stats, pred_type=None):
        self.seen = player_stats.copy()
        return self._params.loc[player_stats.index].copy()


class _Stub:
    """Minimal Stats stand-in: records the head's profile/depth/stats wiring and
    serves a fixed feature frame so output depends only on the scaffolding."""

    # The pickle/dependency resolution is part of what is under test, so the
    # stub borrows the real one rather than reimplementing the cache protocol.
    _load_volume_model = Stats._load_volume_model

    def __init__(self, league, player_stats, player_profile, cache):
        self.league = league
        self._player_stats = player_stats
        self.playerProfile = player_profile
        self._volume_model_cache = cache
        self.calls = []

    def profile_market(self, market, date):
        self.calls.append(("profile_market", market, date))

    def get_depth(self, flat_offers, date):
        self.calls.append(("get_depth", dict(flat_offers), date))

    def get_stats(self, market, flat_offers, date):
        self.calls.append(("get_stats", market, dict(flat_offers), date))
        return self._player_stats.copy()


def _run(league, market, player_stats, player_profile, filedict, **kwargs):
    filedict.setdefault("expected_columns", player_stats.columns.tolist())
    stub = _Stub(
        league,
        player_stats,
        player_profile,
        {f"{league}_{market}".replace(" ", "-"): filedict},
    )
    assert Stats.load_volume_model_params(stub, _OFFERS, market, _DATE, **kwargs) is True
    return stub


def _params(index, columns):
    # Distinct, deterministic values so a misrouted decode is visible.
    return pd.DataFrame(
        {
            c: [round(0.1 * (j + 1) + i, 3) for i in range(len(index))]
            for j, c in enumerate(columns)
        },
        index=index,
    )


def _seed_frame(index, mean_yr):
    return pd.DataFrame(
        {
            "Home": [1, 0, 1][: len(index)],
            "MeanYr": [float(m) for m in mean_yr],
            "STDYr": [2.0] * len(index),
            "ZeroYr": [0.0] * len(index),
        },
        index=index,
    )


def test_dpo_model_emits_mean_and_std_with_no_loc_anywhere(_stub_start_values):
    # NFL_carries is DPO: it emits mu/phi and never a loc. The head must decode
    # by family rather than renaming SkewNormal columns that do not exist.
    idx = ["A", "B"]
    params = pd.DataFrame({"mu": [3.0, 8.0], "phi": [1.5, 0.8]}, index=idx)
    stub = _run(
        "NFL",
        "carries",
        pd.DataFrame({"Home": [1, 0], "f1": [0.4, 0.5]}, index=idx),
        pd.DataFrame({"keep": [1, 2]}, index=idx),
        {"model": _StubModel(params), "distribution": "DPO", "target_normalization": "none"},
    )

    expected_mean = _dp_mean(params["mu"].to_numpy(), params["phi"].to_numpy())
    np.testing.assert_allclose(stub.playerProfile["proj carries mean"], expected_mean)
    np.testing.assert_allclose(
        stub.playerProfile["proj carries std"], np.sqrt(expected_mean / params["phi"])
    )


def test_ratio_normalization_is_decoded_back_onto_the_stat_scale(_stub_start_values):
    # loc ~= 1.0 is a ratio of MeanYr, not a projection of one carry.
    idx = ["A", "B", "C"]
    mean_yr = [10.0, 20.0, 30.0]
    params = pd.DataFrame(
        {"loc": [1.0, 1.0, 1.0], "scale": [0.2, 0.2, 0.2], "alpha": [0.0, 0.0, 0.0]}, index=idx
    )
    stub = _run(
        "NFL",
        "attempts",
        _seed_frame(idx, mean_yr),
        pd.DataFrame({"keep": [1, 2, 3]}, index=idx),
        {
            "model": _StubModel(params),
            "distribution": "SkewNormal",
            "normalized": True,
            "target_normalization": "ratio_meanyr",
            "offset_meta": {"method": "ratio", "denom_col": "MeanYr"},
        },
    )

    np.testing.assert_allclose(stub.playerProfile["proj attempts mean"], mean_yr)
    np.testing.assert_allclose(stub.playerProfile["proj attempts std"], np.array(mean_yr) * 0.2)


def test_centered_normalization_adds_the_mean10_baseline_back(_stub_start_values):
    # NHL/MLB volume cells learn y - Mean10; loc is a residual, not a projection.
    idx = ["A", "B", "C"]
    player_stats = _seed_frame(idx, [10.0, 20.0, 30.0])
    player_stats["Mean10"] = [12.0, 18.0, 33.0]
    params = pd.DataFrame(
        {"loc": [1.0, -2.0, 0.5], "scale": [3.0, 3.0, 3.0], "alpha": [0.0, 0.0, 0.0]}, index=idx
    )
    stub = _run(
        "NHL",
        "timeOnIce",
        player_stats,
        pd.DataFrame({"keep": [1, 2, 3]}, index=idx),
        {
            "model": _StubModel(params),
            "distribution": "SkewNormal",
            "target_normalization": "centered_additive_mean10",
            "offset_meta": {"method": "mean10_additive", "prior_fallback_col": "MeanYr"},
        },
    )

    np.testing.assert_allclose(stub.playerProfile["proj timeOnIce mean"], [13.0, 16.0, 33.5])
    # The centered decode leaves scale absolute -- no baseline multiplier.
    np.testing.assert_allclose(stub.playerProfile["proj timeOnIce std"], 3.0)


def test_seeding_uses_the_artifacts_normalization_not_the_raw_player_mean():
    # start_values is added at predict time, so seeding a normalized model at
    # MeanYr is a straight additive offset on every row's loc.
    idx = ["A", "B", "C"]
    mean_yr = [10.0, 20.0, 30.0]
    model = _StubModel(_params(idx, ["loc", "scale", "alpha"]))
    _run(
        "NFL",
        "attempts",
        _seed_frame(idx, mean_yr),
        pd.DataFrame({"keep": [1, 2, 3]}, index=idx),
        {
            "model": model,
            "distribution": "SkewNormal",
            "normalized": True,
            "target_normalization": "ratio_meanyr",
            "offset_meta": {"method": "ratio", "denom_col": "MeanYr"},
        },
    )

    np.testing.assert_allclose(model.start_values[:, 0], 1.0)


def test_centered_seeding_starts_the_residual_at_zero():
    idx = ["A", "B", "C"]
    player_stats = _seed_frame(idx, [10.0, 20.0, 30.0])
    player_stats["Mean10"] = [12.0, 18.0, 33.0]
    model = _StubModel(_params(idx, ["loc", "scale", "alpha"]))
    _run(
        "NHL",
        "timeOnIce",
        player_stats,
        pd.DataFrame({"keep": [1, 2, 3]}, index=idx),
        {
            "model": model,
            "distribution": "SkewNormal",
            "target_normalization": "centered_additive_mean10",
            "offset_meta": {"method": "mean10_additive", "prior_fallback_col": "MeanYr"},
        },
    )

    np.testing.assert_allclose(model.start_values[:, 0], 0.0)


@pytest.mark.parametrize(
    ("dist", "columns"),
    [("SkewNormal", ["loc", "scale", "alpha"]), ("DPO", ["mu", "phi"])],
)
def test_only_mean_and_std_reach_player_profile(dist, columns, _stub_start_values):
    idx = ["A", "B"]
    stub = _run(
        "NFL",
        "carries",
        pd.DataFrame({"Home": [1, 0], "MeanYr": [5.0, 9.0]}, index=idx),
        pd.DataFrame({"keep": [1, 2]}, index=idx),
        {
            "model": _StubModel(_params(idx, columns)),
            "distribution": dist,
            "target_normalization": "ratio_meanyr",
        },
    )

    assert set(stub.playerProfile.columns) == {"keep", "proj carries mean", "proj carries std"}
    assert stub.playerProfile["proj carries mean"].dtype == np.float64
    assert stub.playerProfile["proj carries std"].dtype == np.float64


def test_volume_decode_matches_the_market_path_decode(_stub_start_values):
    # The whole point of routing both through training.baselines: a volume
    # projection and a market projection off the same artifact must agree.
    idx = ["A", "B", "C"]
    player_stats = _seed_frame(idx, [10.0, 20.0, 30.0])
    params = _params(idx, ["loc", "scale", "alpha"])
    meta = {"method": "ratio", "denom_col": "MeanYr"}
    stub = _run(
        "NFL",
        "attempts",
        player_stats.copy(),
        pd.DataFrame({"keep": [1, 2, 3]}, index=idx),
        {
            "model": _StubModel(params),
            "distribution": "SkewNormal",
            "normalized": True,
            "target_normalization": "ratio_meanyr",
            "offset_meta": meta,
        },
    )

    market_path = _decode_skewnormal(params.copy(), player_stats, 0.0, meta, "ratio_meanyr")
    np.testing.assert_array_equal(
        stub.playerProfile["proj attempts mean"].to_numpy(), market_path["Projection"].to_numpy()
    )
    # The volume path publishes the predictive SD, not the SkewNormal scale the
    # market path carries as "Model Sigma" -- they differ by the skew factor.
    delta = market_path["Model Skew"] / np.sqrt(1 + market_path["Model Skew"] ** 2)
    np.testing.assert_allclose(
        stub.playerProfile["proj attempts std"].to_numpy(),
        market_path["Model Sigma"] * np.sqrt(1 - 2 * delta**2 / np.pi),
    )


def test_position_filter_drops_out_of_position_players(_stub_start_values):
    # NFL restricts each market to its depth-chart positions (carries -> [1, 3]);
    # an out-of-position player gets no projection columns.
    idx = ["A", "B", "C"]
    stub = _run(
        "NFL",
        "carries",
        pd.DataFrame(
            {"Home": [1, 0, 1], "Player position": [1, 2, 3], "MeanYr": [5.0, 9.0, 7.0]},
            index=idx,
        ),
        pd.DataFrame({"keep": [1, 2, 3]}, index=idx),
        {
            "model": _StubModel(_params(idx, ["mu", "phi"])),
            "distribution": "DPO",
            "target_normalization": "none",
        },
        position_filter=[1, 3],
    )

    assert (
        stub.playerProfile.loc["B", "proj carries mean"]
        != stub.playerProfile.loc["B", "proj carries mean"]
    )  # NaN: B is position 2
    assert stub.playerProfile.loc[["A", "C"], "proj carries mean"].notna().all()


def test_join_collision_keeps_the_fresh_projection(_stub_start_values):
    idx = ["A", "B"]
    stub = _run(
        "NFL",
        "carries",
        pd.DataFrame({"Home": [1, 0], "MeanYr": [5.0, 9.0]}, index=idx),
        pd.DataFrame({"proj carries mean": [99.0, 99.0], "keep": [1, 2]}, index=idx),
        {
            "model": _StubModel(pd.DataFrame({"mu": [3.0, 8.0], "phi": [1.5, 0.8]}, index=idx)),
            "distribution": "DPO",
            "target_normalization": "none",
        },
    )

    assert "proj carries mean_obs" not in stub.playerProfile.columns
    assert (stub.playerProfile["proj carries mean"] != 99.0).all()


def test_stats_without_player_position_skip_the_category_cast(_stub_start_values):
    idx = ["A", "B", "C"]
    stub = _run(
        "MLB",
        "pitches thrown",
        pd.DataFrame({"Home": [1, 0, 1], "MeanYr": [80.0, 90.0, 70.0]}, index=idx),
        pd.DataFrame({"keep": [1, 2, 3]}, index=idx),
        {
            "model": _StubModel(_params(idx, ["loc", "scale", "alpha"])),
            "distribution": "SkewNormal",
            "target_normalization": "ratio_meanyr",
        },
    )

    assert stub.playerProfile["proj pitches thrown mean"].notna().all()


def test_missing_pickle_returns_false_and_leaves_profile_untouched():
    # No cache entry and no ZZZ_absent.mdl on disk -> head must abort, return
    # False, and leave playerProfile untouched (NHL relies on this to skip its
    # budget scaling).
    player_stats = pd.DataFrame({"Home": [1, 0], "f1": [0.1, 0.2]}, index=["A", "B"])
    player_profile = pd.DataFrame({"keep": [1, 2]}, index=["A", "B"])
    stub = _Stub("ZZZ", player_stats, player_profile.copy(), {})

    result = Stats.load_volume_model_params(stub, _OFFERS, "absent", _DATE)

    assert result is False
    pd.testing.assert_frame_equal(stub.playerProfile, player_profile)


def test_nba_minutes_uses_shared_dependency_loader(monkeypatch):
    stats = StatsNBA.__new__(StatsNBA)
    calls = []
    offers = {"BOS": {"player a": {"Line": 30.5}}}
    target_date = date(2024, 1, 1)

    def load_volume_model_params(received_offers, market, received_date):
        calls.append((received_offers, market, received_date))
        return False

    monkeypatch.setattr(stats, "load_volume_model_params", load_volume_model_params)

    assert stats.get_volume_stats(offers, target_date) == []
    assert calls == [(offers, "MIN", target_date)]


def test_snapshot_only_dependency_inference_zero_aligns_absent_historical_features(
    _stub_start_values,
):
    idx = ["A", "B"]
    model = _StubModel(_params(idx, ["loc", "scale", "alpha"]))
    stub = _Stub(
        "NFL",
        pd.DataFrame({"Home": [1, 0], "MeanYr": [5.0, 9.0]}, index=idx),
        pd.DataFrame({"keep": [1, 2]}, index=idx),
        {
            "NFL_attempts": {
                "expected_columns": ["Home", "MeanYr", "Player age_asof", "Team proe"],
                "model": model,
                "distribution": "SkewNormal",
            }
        },
    )
    stub.snapshot_only_rebuild = True

    assert Stats.load_volume_model_params(stub, _OFFERS, "attempts", _DATE) is True
    assert model.seen.columns.tolist() == ["Home", "MeanYr", "Player age_asof", "Team proe"]
    assert model.seen[["Player age_asof", "Team proe"]].eq(0).all().all()


def test_live_dependency_inference_keeps_missing_feature_failure_strict(_stub_start_values):
    idx = ["A", "B"]
    stub = _Stub(
        "NFL",
        pd.DataFrame({"Home": [1, 0]}, index=idx),
        pd.DataFrame({"keep": [1, 2]}, index=idx),
        {
            "NFL_attempts": {
                "expected_columns": ["Home", "Player age_asof"],
                "model": _StubModel(_params(idx, ["loc", "scale", "alpha"])),
                "distribution": "SkewNormal",
            }
        },
    )

    with pytest.raises(KeyError, match="Player age_asof"):
        Stats.load_volume_model_params(stub, _OFFERS, "attempts", _DATE)
