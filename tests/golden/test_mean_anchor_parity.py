"""Pins design (b)'s mean anchor: every decode path serves one re-centred distribution.

A cell whose stat_meta sets ``mean_anchor: "plain"`` carries the plain squared-error booster
(``training/plain_baseline.py``) in its pickle, and ``baselines.decode_model_params`` swaps the
decoded mean for the booster's prediction while every shape field stays as decoded. The knob sits
outside the strategy identity, so either pickle signs the corner it always did. Four paths
decode a pickle: the market path (``model_prob._decode_model_params``), the feature-parity
monitor (``feature_parity.score``), the volume-projection head
(``Stats.load_volume_model_params``) and training (``pipeline._step_decode_predictions``). For a
gated SkewNormal, a NegBin and a Double Poisson cell these pins hold that:

- serving resolves the pickle's identity, and the four paths decode the same mean and shape,
  training with the live booster and serving with the one the pickle round-trips;
- the served zero-inclusive mean is the anchor's prediction, so ``ev = anchor / (1 - gate)``,
  with a count family's anchor floored above zero;
- the shape fields are the unanchored decode's;
- the scorecard's decode of the dump's shape columns recovers ``ev``;
- a pickle without the key, signed as every deployed pickle is, still serves and decodes
  exactly as the shared kernel does.

The distributional model is a stand-in that returns fixed raw parameters; the anchor is a real
``fit_plain_baseline`` booster with a spaced, categorical feature set.
"""

from __future__ import annotations

import importlib
import pickle
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, skewnorm

from sportstradamus.helpers.distributions import (
    DecodedParams,
    _dp_mean,
    decode_predictive_mean,
    predictive_std,
)
from sportstradamus.scripts import feature_parity
from sportstradamus.stats import base
from sportstradamus.stats.base import Stats
from sportstradamus.training import baselines
from sportstradamus.training import pipeline as pipe
from sportstradamus.training.model_strategy import (
    MODEL_STRATEGY_MODEL_KEY,
    build_artifact_identity,
)
from sportstradamus.training.plain_baseline import fit_plain_baseline
from sportstradamus.training.scorecard import _decode_sn_loc_scale

# The package __init__ re-exports the model_prob *function*, shadowing the submodule.
mp = importlib.import_module("sportstradamus.prediction.model_prob")

LEAGUE, MARKET = "NFL", "anchor parity"
PLAYERS = ["A", "B", "C", "D", "E", "F"]
# NFL receptions' zero rate: above GATE_PUBLISH_THRESHOLD, so the SkewNormal publishes a gate.
SN_HIST_GATE = 0.11
SN_GLOBAL_MEAN = 4.2
# Training's column order; "Defense avg" carries the space LightGBM stores as an underscore.
COLUMNS = ["MeanYr", "MeanYr_nonzero", "GamesPlayed", "Defense avg", "Home"]
FRAME = pd.DataFrame(
    {
        "MeanYr": [2.5, 3.0, 4.5, 6.0, 7.5, 5.0],
        "MeanYr_nonzero": [2.8, 3.4, 4.9, 6.5, 8.1, 5.6],
        "GamesPlayed": [3.0, 12.0, 30.0, 8.0, 25.0, 16.0],
        "Defense avg": [-1.2, 0.4, 0.9, -0.3, 1.5, 0.0],
        "Home": [0, 1, 0, 1, 0, 1],
    },
    index=PLAYERS,
)
# ``version`` is the implementation_version every deployed pickle of the family signs: a bump
# would unserve them all until each cell retrains.
CASES = {
    "SkewNormal": {
        "params": {
            "loc": [-0.4, 0.2, 0.6, -0.1, 0.3, 0.0],
            "scale": [1.5, 1.8, 2.2, 2.6, 3.0, 2.0],
            "alpha": [1.5, -0.5, 2.0, 0.8, -1.2, 0.3],
        },
        "slug": "centered_additive_eb_meanyr_k10",
        "version": 3,
        "denom_col": "MeanYr_nonzero",
        "hist_gate": SN_HIST_GATE,
        "controls": {
            "dist": "SkewNormal",
            "normalization": "centered_additive_eb_meanyr_k10",
            "dist_training_loss": "crps",
            "sn_param": "direct",
            "blending_loss_fn": "crps",
            "posthoc": "none",
        },
    },
    "NegBin": {
        "params": {
            "total_count": [4.0, 9.0, 15.0, 6.0, 20.0, 11.0],
            "probs": [0.35, 0.30, 0.25, 0.50, 0.28, 0.33],
        },
        "slug": "none",
        "version": 1,
        "denom_col": "MeanYr",
        "hist_gate": 0.0,
        "controls": {
            "dist": "NegBin",
            "dist_training_loss": "nll",
            "count_dispersion_objective": "crps",
            "blending_loss_fn": "crps",
            "posthoc": "none",
        },
    },
    "DPO": {
        "params": {
            "mu": [2.6, 3.1, 4.4, 5.9, 7.2, 4.8],
            "phi": [0.8, 1.2, 1.0, 0.6, 1.4, 0.9],
        },
        "slug": "none",
        "version": 1,
        "denom_col": "MeanYr",
        "hist_gate": 0.0,
        "controls": {
            "dist": "DPO",
            "dist_training_loss": "nll",
            "count_dispersion_objective": "crps",
            "blending_loss_fn": "crps",
            "posthoc": "none",
        },
    },
}


class _StubLSS:
    """Stand-in for the LightGBMLSS booster: fixed raw parameters per player."""

    def __init__(self, params: pd.DataFrame):
        self.params = params

    def predict(self, rows, pred_type=None):
        return self.params.loc[rows.index].copy()


class _VolumeStub:
    """The slice of ``Stats`` the volume head reads, serving one cached pickle."""

    _load_volume_model = Stats._load_volume_model

    def __init__(self, filedict: dict):
        self.league = LEAGUE
        self.playerProfile = pd.DataFrame(index=PLAYERS)
        self._volume_model_cache = {f"{LEAGUE}_{MARKET}".replace(" ", "-"): filedict}

    def profile_market(self, market, date):
        pass

    def get_depth(self, offers, date):
        pass

    def get_stats(self, market, offers, date):
        return FRAME.copy()


@pytest.fixture(scope="module")
def anchor():
    """A real plain booster, fit as training fits it on categorically cast features."""
    rng = np.random.default_rng(5)
    n_rows = 600
    index = pd.Index([f"row {i}" for i in range(n_rows)])
    mean_yr = rng.uniform(2.0, 8.0, n_rows)
    features = pd.DataFrame(
        {
            "MeanYr": mean_yr,
            "MeanYr_nonzero": mean_yr * 1.1,
            "GamesPlayed": rng.integers(1, 40, n_rows).astype(float),
            "Defense avg": rng.normal(0.0, 1.0, n_rows),
            "Home": pd.Categorical(rng.integers(0, 2, n_rows)),
        },
        index=index,
    )
    result = (
        features["MeanYr"]
        + 0.5 * features["Defense avg"]
        + features["Home"].astype(int)
        + rng.normal(0.0, 1.0, n_rows)
    )
    splits = {"X": features, "y_train": result.to_frame("Result"), "X_test": features.iloc[:1]}
    dates = pd.Series(pd.date_range("2025-09-01", periods=n_rows, freq="h"), index=index)
    return fit_plain_baseline(splits, dates, {}).booster


def _served_frame() -> pd.DataFrame:
    """The rows as decode sees them: categoricals cast, as every serving path casts them."""
    return FRAME.astype({"Home": "category"})


def _strategy(dist: str):
    return baselines.get_target_normalization(CASES[dist]["slug"]) if dist == "SkewNormal" else None


def _pickle(dist: str, booster) -> dict:
    """A tiny signed model file, round-tripped through pickle as serving loads it."""
    case = CASES[dist]
    strategy = _strategy(dist)
    filedict = {
        "model": _StubLSS(pd.DataFrame(case["params"], index=PLAYERS)),
        "distribution": dist,
        "target_normalization": case["slug"],
        "normalized": False,
        "offset_meta": (
            None if strategy is None else strategy.offset_meta(SN_GLOBAL_MEAN, case["denom_col"])
        ),
        "hist_gate": case["hist_gate"],
        "sn_param": "direct",
        "posthoc": "none",
        "expected_columns": COLUMNS,
        MODEL_STRATEGY_MODEL_KEY: build_artifact_identity(
            dist, LEAGUE, MARKET, case["controls"], matrix_hash="c" * 64
        ).as_model_blob(),
    }
    if booster is not None:
        filedict["mean_anchor"] = booster
    return pickle.loads(pickle.dumps(filedict))


def _from_frame(frame: pd.DataFrame) -> DecodedParams:
    return DecodedParams(
        ev=frame["Projection"],
        r=frame.get("Model R"),
        sigma=frame.get("Model Sigma"),
        skew=frame.get("Model Skew"),
        gate=frame.get("Model Gate"),
        phi=frame.get("Model Phi"),
    )


def _fields(decoded: DecodedParams) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(values, dtype=float)
        for name, values in asdict(decoded).items()
        if values is not None
    }


def _assert_same(decoded: DecodedParams, reference: DecodedParams) -> None:
    actual, expected = _fields(decoded), _fields(reference)
    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)


def _decode_four_ways(dist: str, booster, monkeypatch) -> dict[str, DecodedParams]:
    """One frame decoded by every path: training with ``booster``, serving with its pickle."""
    monkeypatch.setattr(mp, "set_model_start_values", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(base, "set_model_start_values", lambda *_args, **_kwargs: None)
    case = CASES[dist]
    filedict = _pickle(dist, booster)
    # Serving refuses a pickle whose signed identity does not resolve, before it decodes.
    mp._resolve_serving_strategy(filedict, LEAGUE, MARKET)
    params = pd.DataFrame(case["params"], index=PLAYERS)
    served = _served_frame()

    trained = pipe._step_decode_predictions(
        params,
        params,
        served,
        served,
        dist,
        _strategy(dist),
        SN_GLOBAL_MEAN,
        case["denom_col"],
        case["hist_gate"],
        booster,
    )
    # model_prob's frame carries more than the model's columns, in its own order.
    market_rows = served[served.columns[::-1]].assign(**{"Market Projection": 1.0})
    market = params.copy()
    mp._decode_model_params(
        market,
        dist,
        market_rows,
        case["hist_gate"],
        filedict["offset_meta"],
        filedict["target_normalization"],
        filedict.get("mean_anchor"),
    )
    volume = _VolumeStub(filedict)
    assert Stats.load_volume_model_params(volume, {}, MARKET, "2025-10-05") is True
    return {
        "training": DecodedParams(
            ev=trained["ev"],
            r=trained["r"],
            alpha=trained["alpha"],
            sigma=trained["sn_sigma_test"],
            skew=trained["sn_alpha_test"],
            gate=trained["gate_test"],
            phi=trained["phi"],
        ),
        "model_prob": _from_frame(market),
        "feature_parity": _from_frame(
            feature_parity.score(filedict, LEAGUE, MARKET, FRAME, case["hist_gate"])
        ),
        # The volume head publishes only the predictive mean and SD.
        "volume": DecodedParams(
            ev=volume.playerProfile[f"proj {MARKET} mean"],
            sigma=volume.playerProfile[f"proj {MARKET} std"],
        ),
    }


def _assert_paths_agree(dist: str, decoded: dict[str, DecodedParams]) -> None:
    trained = decoded["training"]
    _assert_same(decoded["model_prob"], trained)
    _assert_same(decoded["feature_parity"], trained)
    np.testing.assert_array_equal(decoded["volume"].ev, trained.ev)
    np.testing.assert_array_equal(decoded["volume"].sigma, predictive_std(dist, trained))


@pytest.mark.parametrize("dist", list(CASES))
def test_every_path_serves_the_anchor_mean_with_the_unanchored_shape(dist, anchor, monkeypatch):
    anchored = _decode_four_ways(dist, anchor, monkeypatch)
    unanchored = _decode_four_ways(dist, None, monkeypatch)

    _assert_paths_agree(dist, anchored)
    # By position, in training's column order: independent of the decode's lookup by name.
    raw = anchor.predict(_served_frame()[COLUMNS])
    served = anchored["training"]
    if dist == "SkewNormal":
        np.testing.assert_array_equal(served.ev, raw / (1 - served.gate))
        np.testing.assert_allclose((1 - served.gate) * served.ev, raw, rtol=1e-12)
    else:
        assert served.gate is None
        np.testing.assert_array_equal(served.ev, np.maximum(raw, baselines._ANCHOR_MEAN_FLOOR))
    assert not np.allclose(served.ev, unanchored["training"].ev)
    _assert_same(replace(served, ev=unanchored["training"].ev), unanchored["training"])


@pytest.mark.parametrize("dist", list(CASES))
def test_scorecard_decode_of_the_dump_recovers_the_anchored_mean(dist, anchor, monkeypatch):
    served = _decode_four_ways(dist, anchor, monkeypatch)["training"]
    case = CASES[dist]
    dump = _served_frame()

    pipe._stage_family_shape_columns(
        dump,
        dist=dist,
        prob_params=pd.DataFrame(case["params"], index=PLAYERS),
        weighted_mean=served.ev,
        sn_scale_test=served.sigma,
        sn_skew_test=served.skew,
        mix_test=None,
        r_test=served.r,
        gate_blend_test=served.gate,
        phi_test=served.phi,
        target_normalization=case["slug"],
        global_mean=SN_GLOBAL_MEAN,
        denom_col=case["denom_col"],
        hist_gate=case["hist_gate"],
    )

    if dist == "SkewNormal":
        loc, scale = _decode_sn_loc_scale(dump, case["slug"])
        np.testing.assert_allclose(skewnorm.mean(dump["SN_Alpha"], loc, scale), served.ev)
    elif dist == "NegBin":
        np.testing.assert_allclose(nbinom.mean(dump["R"], 1.0 - dump["NB_P"]), served.ev)
    else:
        recovered = _dp_mean(dump["DP_MU"].to_numpy(), dump["DP_PHI"].to_numpy())
        np.testing.assert_allclose(recovered, served.ev, rtol=1e-5)


@pytest.mark.parametrize("dist", list(CASES))
def test_a_pickle_without_the_key_still_serves_and_decodes_byte_identical(dist, monkeypatch):
    """Pins that the knob never unserves an existing pickle or moves what it decodes."""
    case = CASES[dist]
    filedict = _pickle(dist, None)
    assert "mean_anchor" not in filedict
    identity = mp._resolve_serving_strategy(filedict, LEAGUE, MARKET)
    assert identity.implementation_version == case["version"]

    params = pd.DataFrame(case["params"], index=PLAYERS)
    if dist == "SkewNormal":
        strategy, served = _strategy(dist), _served_frame()
        kernel = decode_predictive_mean(
            params,
            dist,
            sn_loc=strategy.decode_loc(
                params["loc"].to_numpy(), served, SN_GLOBAL_MEAN, case["denom_col"]
            ),
            sn_scale=strategy.decode_scale(params["scale"].to_numpy(), served, case["denom_col"]),
            hist_gate=case["hist_gate"],
        )
    else:
        kernel = decode_predictive_mean(params, dist)

    decoded = _decode_four_ways(dist, None, monkeypatch)

    _assert_same(decoded["training"], kernel)
    _assert_paths_agree(dist, decoded)


class _BelowZeroAnchor:
    """A plain model predicting below zero, as squared error can, on one spaced column."""

    def feature_name(self):
        return ["Defense_avg"]

    def predict(self, rows):
        return -1.0 - rows["Defense avg"].abs().to_numpy()


@pytest.mark.parametrize("dist", ["NegBin", "DPO"])
def test_count_family_floors_the_anchor_above_zero(dist):
    decoded = baselines.decode_model_params(
        pd.DataFrame(CASES[dist]["params"], index=PLAYERS),
        dist,
        FRAME,
        hist_gate=0.0,
        offset_meta=None,
        target_normalization="none",
        mean_anchor=_BelowZeroAnchor(),
    )

    np.testing.assert_array_equal(decoded.ev, baselines._ANCHOR_MEAN_FLOOR)
