"""Characterization test for the shared MLB/NHL volume-model load head.

StatsMLB.get_volume_stats and StatsNHL.get_volume_stats shared a near-identical
opening block: flatten offers -> profile_market/get_depth/get_stats -> load the
``models/{league}_{market}.mdl`` pickle (cached) -> slice to ``expected_columns``
-> cast ``Home`` / ``Player position`` to category -> ``set_model_start_values``
-> ``model.predict`` -> drop the ``gate`` column -> join the renamed distribution
parameters into ``playerProfile`` (early-returning when the pickle is absent).
That block was extracted into ``Stats.load_volume_model_params``, parameterized
by ``market`` and the per-league ``rename_map``.

This pins the extracted method against a verbatim reference copy of the old block
across the NHL (loc/scale/alpha) and MLB (rate/loc -> mean, scale -> std) rename
maps, the with/without ``Player position`` category branch, the ``_obs``
join-collision drop, and the missing-pickle early return.

The trained MLB/NHL pickles are not needed and are not on CI boxes: the per-cell
model is a shared pass-through -- loaded, start-valued, and predicted from
identically in both paths -- so a deterministic stub model plus a no-op
``set_model_start_values`` isolate the duplicated scaffolding under refactor. The
model's own math is covered by its training tests, not here.
"""

import importlib.resources as pkg_resources
import os

import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.stats import base
from sportstradamus.stats.base import Stats

_DATE = "2024-01-01"
# Offers shaped like the real call: a per-team mapping plus a per-market bucket,
# both folded into one flat dict by the head's offer-flatten preamble.
_OFFERS = {"BOS": {"player a": {"Line": 1.5}}, "timeOnIce": {"player b": {"Line": 2.5}}}


@pytest.fixture(autouse=True)
def _stub_start_values(monkeypatch):
    # Shared pass-through, identical in both paths and tested elsewhere; a no-op
    # here removes the MeanYr/STDYr/ZeroYr + torch coupling so the test pins only
    # the duplicated scaffolding.
    monkeypatch.setattr(base, "set_model_start_values", lambda *a, **k: None)


class _StubModel:
    def __init__(self, params):
        self._params = params

    def predict(self, player_stats, pred_type=None):
        return self._params.loc[player_stats.index].copy()


class _Stub:
    """Minimal Stats stand-in: records the head's profile/depth/stats wiring and
    serves a fixed feature frame so output depends only on the scaffolding."""

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


def _reference_load_head(stub, offers, market, date, rename_map, position_filter=None):
    """Verbatim copy of the pre-consolidation MLB/NHL/NFL opening block."""
    flat_offers = {}
    if isinstance(offers, dict):
        for players in offers.values():
            flat_offers.update(players)
    else:
        flat_offers = offers

    if isinstance(offers, dict):
        flat_offers.update(offers.get(market, {}))
    stub.profile_market(market, date)
    stub.get_depth(flat_offers, date)
    playerStats = stub.get_stats(market, flat_offers, date)

    filename = "_".join([stub.league, market]).replace(" ", "-")
    filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
    if stub._volume_model_cache is None:
        stub._volume_model_cache = {}
    if filename not in stub._volume_model_cache:
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                stub._volume_model_cache[filename] = base.pickle.load(infile)
        else:
            return

    if filename in stub._volume_model_cache:
        filedict = stub._volume_model_cache[filename]
        playerStats = playerStats[filedict["expected_columns"]]
        model = filedict["model"]
        dist = filedict["distribution"]

        categories = ["Home", "Player position"]
        if "Player position" not in playerStats.columns:
            categories.remove("Player position")
        for c in categories:
            playerStats[c] = playerStats[c].astype("category")

        base.set_model_start_values(model, dist, playerStats)

        prob_params = pd.DataFrame()
        preds = model.predict(playerStats, pred_type="parameters")
        preds.index = playerStats.index
        prob_params = pd.concat([prob_params, preds])

        prob_params.sort_index(inplace=True)
        playerStats.sort_index(inplace=True)
        if position_filter is not None:
            prob_params = prob_params.loc[playerStats["Player position"].isin(position_filter)]
    else:
        return

    prob_params.drop(columns=["gate"], inplace=True, errors="ignore")
    stub.playerProfile = stub.playerProfile.join(
        prob_params.rename(columns=rename_map), lsuffix="_obs"
    )
    stub.playerProfile.drop(
        columns=[col for col in stub.playerProfile.columns if "_obs" in col], inplace=True
    )


def _assert_match(
    league,
    market,
    player_stats,
    player_profile,
    expected_columns,
    params,
    rename_map,
    position_filter=None,
):
    filename = f"{league}_{market}".replace(" ", "-")

    def _cache():
        return {
            filename: {
                "expected_columns": expected_columns,
                "model": _StubModel(params),
                "distribution": "SkewNormal",
            }
        }

    ref = _Stub(league, player_stats.copy(), player_profile.copy(), _cache())
    new = _Stub(league, player_stats.copy(), player_profile.copy(), _cache())

    _reference_load_head(ref, _OFFERS, market, _DATE, rename_map, position_filter)
    kwargs = {} if position_filter is None else {"position_filter": position_filter}
    result = Stats.load_volume_model_params(new, _OFFERS, market, _DATE, rename_map, **kwargs)

    assert result is True
    pd.testing.assert_frame_equal(ref.playerProfile, new.playerProfile)
    assert ref.calls == new.calls


def _params(index, columns):
    # Distinct, deterministic values so a misrouted rename/drop is visible.
    return pd.DataFrame(
        {
            c: [round(0.1 * (j + 1) + i, 3) for i in range(len(index))]
            for j, c in enumerate(columns)
        },
        index=index,
    )


def test_nhl_skewnormal_map():
    idx = ["C", "A", "B"]  # unsorted: exercises the prob_params/playerStats sort
    player_stats = pd.DataFrame(
        {
            "Home": [1, 0, 1],
            "Player position": ["C", "G", "F"],
            "f1": [0.4, 0.5, 0.6],
            "drop_me": [9, 9, 9],
        },
        index=idx,
    )
    player_profile = pd.DataFrame(
        {"proj timeOnIce loc": [99.0, 99.0, 99.0], "keep": [1, 2, 3]},  # collision -> _obs drop
        index=["A", "B", "C"],
    )
    params = _params(idx, ["loc", "scale", "alpha", "gate"])
    rename_map = {
        "loc": "proj timeOnIce loc",
        "scale": "proj timeOnIce scale",
        "alpha": "proj timeOnIce alpha",
    }
    _assert_match(
        "NHL",
        "timeOnIce",
        player_stats,
        player_profile,
        ["Home", "Player position", "f1"],
        params,
        rename_map,
    )


def test_mlb_rate_map_without_player_position():
    idx = ["A", "B", "C"]
    player_stats = pd.DataFrame(  # no "Player position" -> categories.remove branch
        {"Home": [1, 0, 1], "f1": [0.4, 0.5, 0.6], "drop_me": [9, 9, 9]}, index=idx
    )
    player_profile = pd.DataFrame({"keep": [1, 2, 3]}, index=idx)
    params = _params(idx, ["rate", "scale", "gate"])
    rename_map = {
        "loc": "proj plateAppearances mean",
        "rate": "proj plateAppearances mean",
        "scale": "proj plateAppearances std",
    }
    _assert_match(
        "MLB",
        "plateAppearances",
        player_stats,
        player_profile,
        ["Home", "f1"],
        params,
        rename_map,
    )


def test_mlb_loc_map_without_player_position():
    idx = ["A", "B", "C"]
    player_stats = pd.DataFrame({"Home": [1, 0, 1], "f1": [0.4, 0.5, 0.6]}, index=idx)
    player_profile = pd.DataFrame({"keep": [1, 2, 3]}, index=idx)
    params = _params(idx, ["loc", "scale", "gate"])
    rename_map = {
        "loc": "proj plateAppearances mean",
        "rate": "proj plateAppearances mean",
        "scale": "proj plateAppearances std",
    }
    _assert_match(
        "MLB",
        "plateAppearances",
        player_stats,
        player_profile,
        ["Home", "f1"],
        params,
        rename_map,
    )


def test_nfl_position_filter_drops_out_of_position_players():
    # NFL filters prob_params to the positions relevant for the market before the
    # join (carries -> positions [1, 3]); an out-of-position player gets no
    # projection columns. Other leagues pass position_filter=None and skip this.
    idx = ["A", "B", "C"]
    player_stats = pd.DataFrame(
        {
            "Home": [1, 0, 1],
            "Player position": [1, 2, 3],  # B (pos 2) is filtered out for carries
            "f1": [0.4, 0.5, 0.6],
        },
        index=idx,
    )
    player_profile = pd.DataFrame({"keep": [1, 2, 3]}, index=idx)
    params = _params(idx, ["loc", "scale", "alpha", "gate"])
    rename_map = {
        "loc": "proj carries loc",
        "scale": "proj carries scale",
        "alpha": "proj carries alpha",
    }
    _assert_match(
        "NFL",
        "carries",
        player_stats,
        player_profile,
        ["Home", "Player position", "f1"],
        params,
        rename_map,
        position_filter=[1, 3],
    )


def test_missing_pickle_returns_false_and_leaves_profile_untouched():
    # No cache entry and no ZZZ_absent.mdl on disk -> head must abort, return
    # False, and leave playerProfile untouched (NHL relies on this to skip its
    # budget scaling).
    player_stats = pd.DataFrame({"Home": [1, 0], "f1": [0.1, 0.2]}, index=["A", "B"])
    player_profile = pd.DataFrame({"keep": [1, 2]}, index=["A", "B"])
    stub = _Stub("ZZZ", player_stats, player_profile.copy(), {})

    result = Stats.load_volume_model_params(stub, _OFFERS, "absent", _DATE, {})

    assert result is False
    pd.testing.assert_frame_equal(stub.playerProfile, player_profile)
