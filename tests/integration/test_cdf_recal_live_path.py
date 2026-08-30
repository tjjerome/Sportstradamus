"""Live-path verification for §6.1 Rung C (whole-CDF PIT recalibration).

Drives the real ``model_prob`` inference seam end-to-end on the cached NBA_DREB
SkewNormal fixture: fits a deterministic model, packages a pickle carrying a
``pit_recal_blob``, and scores a small offer batch through the production decode →
blend → over-prob path. Asserts the served ``Model Over`` (and thus ``Win Prob``,
which the parlay beam search reads at ``correlation.py``'s ``game_df["Win Prob"]``)
equals the un-recalibrated run warped through ``1 − g(1 − over)`` — the same map the
offline Gate-4 scored. A pickle with no blob serves byte-identically.

Marked @pytest.mark.integration, no network — relies on the cached NBA_DREB.parquet
fixture. Skips cleanly when the parquet is absent.
"""

from __future__ import annotations

import importlib
import importlib.resources as pkg_resources
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.helpers.training_quotes import ArchivedBookQuote
from sportstradamus.skew_normal import SkewNormal as SkewNormalDist
from sportstradamus.stats import StatsNBA
from sportstradamus.training import posthoc
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    DETERMINISTIC_SEED,
    _step_build_splits,
    fit_predict_params,
)

# ``sportstradamus.prediction`` re-exports the ``model_prob`` function, shadowing the
# submodule under attribute access — fetch the module so monkeypatch hits its globals.
mp = importlib.import_module("sportstradamus.prediction.model_prob")

_LEAGUE, _MARKET, _PLATFORM = "NBA", "DREB", "Underdog"
_N_TRAIN_ROWS = 4000
# 12 only left ~4 rows unclamped against the current cached fixture (needs >=5 below); 25
# gives comfortable headroom against future NBA_DREB.parquet regens shifting the fit.
_N_OFFERS = 25
_BOOK_EV = 6.0


class _StubArchive:
    default_totals = {_LEAGUE: 220.0}

    def get_training_quote_inputs(self, league, market, date, entities):
        row = ArchivedBookQuote(
            book="fanduel", ev=_BOOK_EV, under_probability=0.55, line=5.5, observed_at=None
        )
        return {entity: ([row], 5.5) for entity in entities}


class _StubStats:
    league = _LEAGUE

    def check_combo_markets(self, market, player, date):
        return float("nan")


def _fit_real_skewnormal_params():
    parquet = pkg_resources.files(data) / f"training_data/{_LEAGUE}_{_MARKET}.parquet"
    if not parquet.is_file():
        pytest.skip(f"cached {_LEAGUE}_{_MARKET}.parquet not present")
    m = pd.read_parquet(parquet).sort_values(["Date"]).head(_N_TRAIN_ROWS).reset_index(drop=True)
    splits = _step_build_splits(m, StatsNBA(), _MARKET, "ratio_meanyr")
    params = {
        **DETERMINISTIC_FIXED_PARAMS,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "monotone_constraints": [0] * splits["X_train"].shape[1],
    }
    prob_params = fit_predict_params(
        SkewNormalDist(stabilization="None", loss_fn="crps"),
        "SkewNormal",
        splits["X_train"],
        splits["y_train_labels"].astype(float),
        splits["X_test"].reset_index(drop=True),
        params=params,
        normalized=True,
        shape_ceiling=100.0,
        seed=DETERMINISTIC_SEED,
    )
    return prob_params, splits["X_test"].reset_index(drop=True)


def _filedict(pit_recal_blob):
    # Only the keys model_prob reads — _build_prob_params is monkeypatched, so no model
    # object or expected_columns are needed. temperature=1.0 makes the warp the only
    # transform between the raw over-prob and Model Over, so the parity check is exact.
    return {
        "cv": 1.0,
        "weight": 1.0,
        "temperature": 1.0,
        "dispersion_cal": 1.0,
        "skew_cal": 0.0,
        "shape_ceiling": None,
        "distribution": "SkewNormal",
        "step": 1,
        "normalized": True,
        "offset_meta": None,
        "target_normalization": "ratio_meanyr",
        "posthoc": "none",
        "posthoc_blob": None,
        "pit_recal_blob": pit_recal_blob,
    }


def _score(monkeypatch, tmp_path, prob_params, player_stats, offers, pit_recal_blob):
    pickle_path = tmp_path / "model.pkl"
    with open(pickle_path, "wb") as fh:
        pickle.dump(_filedict(pit_recal_blob), fh)
    monkeypatch.setattr(mp, "model_pickle_path", lambda _lg, _mkt: str(pickle_path))
    monkeypatch.setattr(mp, "archive", _StubArchive())
    monkeypatch.setattr(mp, "stat_cv", {_LEAGUE: {_MARKET: 1.0}})
    monkeypatch.setattr(mp, "stat_dist", {_LEAGUE: {_MARKET: "SkewNormal"}})
    monkeypatch.setattr(mp, "stat_zi", {})
    monkeypatch.setattr(mp, "_build_prob_params", lambda *a, **k: prob_params.copy())
    return pd.DataFrame(
        mp.model_prob(offers, _LEAGUE, _MARKET, _PLATFORM, _StubStats(), player_stats)
    )


@pytest.mark.integration
def test_cdf_recal_warps_served_win_prob_end_to_end(tmp_path, monkeypatch):
    raw_params, x_test = _fit_real_skewnormal_params()
    players = [f"Player {i}" for i in range(_N_OFFERS)]
    prob_params = raw_params.head(_N_OFFERS).copy()
    prob_params.index = players
    player_stats = x_test.head(_N_OFFERS).copy()
    player_stats.index = players
    offers = [
        {
            "Player": p,
            "League": _LEAGUE,
            "Team": "LAL",
            "Opponent": "BOS",
            "Date": "2026-06-03",
            "Market": _MARKET,
            "Line": 5.5,
            "Boost": 1.0,
            "Boost_Over": 1.0,
            "Boost_Under": 1.0,
        }
        for p in players
    ]

    # A strongly non-identity map (U-shaped PIT) so the warp is unmistakable on the slate.
    blob = posthoc.fit_isotonic_pit(np.random.default_rng(0).beta(0.5, 0.5, size=4000))

    base = _score(monkeypatch, tmp_path, prob_params, player_stats, offers, None)
    warped = _score(monkeypatch, tmp_path, prob_params, player_stats, offers, blob)

    assert not base.empty and len(warped) == len(base)
    merged = base.merge(warped, on="Player", suffixes=("_base", "_warp"))
    assert len(merged) == _N_OFFERS

    # model_prob outputs Win Prob = max(over, under) + the Bet side, so recover the served
    # over-prob from the pair the parlay beam search reads (correlation.py game_df["Win Prob"]).
    def _over(suffix):
        wp = merged[f"Win Prob_{suffix}"].to_numpy()
        return np.where(merged[f"Bet_{suffix}"].to_numpy() == "Over", wp, 1.0 - wp)

    base_over, warp_over = _over("base"), _over("warp")

    # _finalize_records caps confidence at _MAX_CONFIDENCE (0.9) on both runs, which masks
    # the exact warp on rows it pushes past the cap; verify parity on the interior rows the
    # shared clamp leaves untouched (base == raw over there).
    interior = (base_over > 0.101) & (base_over < 0.899) & (warp_over > 0.101) & (warp_over < 0.899)
    assert interior.sum() >= 5, f"too few unclamped rows to verify ({interior.sum()})"

    # End-to-end: the served over equals the un-recalibrated over warped through
    # 1 − g(1 − over). temperature=1.0 and no posthoc make this the only transform, so the
    # warp the offline Gate-4 scored is exactly what reaches every parlay leg's Win Prob.
    np.testing.assert_allclose(
        warp_over[interior], mp._apply_cdf_recal_over(base_over, blob)[interior], atol=1e-9
    )
    # Non-trivial on this slate (guards against a silently-skipped warp).
    assert not np.allclose(warp_over[interior], base_over[interior], atol=1e-3)

    # The map itself is persisted on every offer so the dashboard can redraw the served
    # g∘F (deep_dive_charts renders the recalibrated distribution, not the raw parametric
    # one). A no-blob cell carries null — the raw curve is exact there.
    assert warped["Model PIT Recal"].notna().all()
    assert json.loads(warped["Model PIT Recal"].iloc[0]) == blob
    assert base["Model PIT Recal"].isna().all()
