"""Live-path verification for the post-fusion mean-stage corrector.

Drives the real ``model_prob`` inference seam end-to-end on the cached NBA_BLK
fixture with a pickle carrying a :data:`posthoc.MEAN_STAGE` slug, and asserts the
served ``Projection`` is the corrector applied to the *pooled* mean rather than to
the model leg. ``fused_loc`` pools a count location geometrically, so the two
arrangements differ by the pool's ``rho^(1-w)`` haircut whenever the book disagrees
with the model — which is the defect this lane fixed, and the thing a unit test on
:func:`posthoc.correct_fused_mean` alone cannot see.

The same parity also pins the zero-inflation contract: the served mean is
``(1 - pi_blend) * base``, so a ZINB pickle that applied the gate twice would fail
the exact equality below.

Marked @pytest.mark.integration, no network. Skips cleanly when the cached parquet
is absent.
"""

from __future__ import annotations

import datetime
import importlib
import importlib.resources as pkg_resources
import pickle

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.helpers.config import book_count_dispersion
from sportstradamus.helpers.distributions import get_odds
from sportstradamus.helpers.training_quotes import ArchivedBookQuote
from sportstradamus.prediction import book_quotes, offer_records
from sportstradamus.training import posthoc

mp = importlib.import_module("sportstradamus.prediction.model_prob")

pytestmark = pytest.mark.integration

_LEAGUE, _MARKET, _PLATFORM = "NBA", "BLK", "Underdog"
_N_OFFERS = 30
_LINE = 1.5
# Well below _MODEL_WEIGHT_MAX so the pool is a real pool; at w = 1 the two stage
# orders coincide and the test would pass vacuously.
_MODEL_WEIGHT = 0.6
# A book that disagrees with the model on level — rho != 1 is what makes the
# geometric pool's AM-GM haircut observable.
_BOOK_EV = 0.55
_MODEL_R = 3.0
_ZI_GATE = 0.25
_OBSERVED_AT = datetime.datetime(2026, 6, 3, 12, 0, 0)
# Identity columns the offer frame owns; the feature frame must not duplicate them.
_OFFER_COLUMNS = ("Player", "League", "Team", "Opponent", "Date", "Market", "Line", "Boost")


class _StubArchive:
    """The two archive surfaces model_prob reads, and nothing else."""

    default_totals = {_LEAGUE: 220.0}

    def __init__(self, dist, gate):
        # One modal cohort of real sportsbooks whose consensus under-probability
        # inverts to exactly _BOOK_EV. Built through get_odds under the same shape
        # quote_pricing_params will invert with, rather than asserted as a mean the
        # book leg no longer accepts directly.
        _phi, r = book_count_dispersion(_LEAGUE, _MARKET, _LINE, 1.0)
        under = float(get_odds(_LINE, _BOOK_EV, dist, cv=1.0, gate=gate, r=r))
        self._rows = [
            ArchivedBookQuote(book, None, under, _LINE, _OBSERVED_AT)
            for book in ("fanduel", "draftkings")
        ]

    def get_training_quote_inputs(self, league, market, date, entities, at=None):
        return {e: (list(self._rows), _LINE) for e in entities}


class _StubStats:
    league = _LEAGUE


def _fixture_rows():
    parquet = pkg_resources.files(data) / f"training_data/{_LEAGUE}_{_MARKET}.parquet"
    if not parquet.is_file():
        pytest.skip(f"cached {_LEAGUE}_{_MARKET}.parquet not present")
    m = pd.read_parquet(parquet).sort_values(["Date"]).tail(_N_OFFERS).reset_index(drop=True)
    # model_prob joins the feature frame onto the offer frame by player index, so the
    # offer-side identity columns must not also arrive from the features.
    m = m.drop(columns=[c for c in _OFFER_COLUMNS if c in m.columns])
    players = [f"Player {i}" for i in range(len(m))]
    m.index = players
    return m, players


def _prob_params(dist, players):
    """A NegBin/ZINB parameter frame spanning a real spread of means."""
    mean = np.linspace(0.4, 2.8, len(players))
    frame = pd.DataFrame(
        {"total_count": _MODEL_R, "probs": mean / (_MODEL_R + mean)}, index=players
    )
    if dist == "ZINB":
        frame["gate"] = _ZI_GATE
    return frame


def _filedict(dist, slug, blob):
    # Only the keys model_prob reads — _build_prob_params is monkeypatched. temperature
    # and dispersion_cal are neutral so the corrector is the only mean transform.
    return {
        "cv": 1.0,
        "weight": _MODEL_WEIGHT,
        "temperature": 1.0,
        "dispersion_cal": 1.0,
        "skew_cal": 0.0,
        "shape_ceiling": None,
        "distribution": dist,
        "step": 1,
        "normalized": False,
        "offset_meta": None,
        "target_normalization": "ratio_meanyr",
        "posthoc": slug,
        "posthoc_blob": blob,
        "pit_recal_blob": None,
    }


def _score(monkeypatch, tmp_path, dist, slug, blob, player_stats, offers, players):
    pickle_path = tmp_path / f"{dist}_{slug}.pkl"
    with open(pickle_path, "wb") as fh:
        pickle.dump(_filedict(dist, slug, blob), fh)
    monkeypatch.setattr(mp, "model_pickle_path", lambda _lg, _mkt: str(pickle_path))
    # Two bindings since the model_prob split: book_quotes resolves the quotes,
    # offer_records reads default_totals when finalizing.
    stub_archive = _StubArchive(dist, _ZI_GATE if dist == "ZINB" else None)
    monkeypatch.setattr(book_quotes, "archive", stub_archive)
    monkeypatch.setattr(offer_records, "archive", stub_archive)
    monkeypatch.setattr(mp, "stat_cv", {_LEAGUE: {_MARKET: 1.0}})
    monkeypatch.setattr(mp, "stat_dist", {_LEAGUE: {_MARKET: dist}})
    monkeypatch.setattr(mp, "stat_zi", {_LEAGUE: {_MARKET: _ZI_GATE}} if dist == "ZINB" else {})
    monkeypatch.setattr(mp, "_build_prob_params", lambda *a, **k: _prob_params(dist, players))
    return pd.DataFrame(
        mp.model_prob(offers, _LEAGUE, _MARKET, _PLATFORM, _StubStats(), player_stats)
    )


def _offers(players):
    return [
        {
            "Player": p,
            "League": _LEAGUE,
            "Team": "LAL",
            "Opponent": "BOS",
            "Date": "2026-06-03",
            "Market": _MARKET,
            "Line": _LINE,
            "Boost": 1.0,
            "Boost_Over": 1.0,
            "Boost_Under": 1.0,
        }
        for p in players
    ]


def _served_over(frame):
    """model_prob publishes max(over, under) as Win Prob plus the chosen side."""
    wp = frame["Win Prob"].to_numpy(dtype=float)
    return np.where(frame["Bet"].to_numpy() == "Over", wp, 1.0 - wp)


@pytest.mark.parametrize("dist", ["NegBin", "ZINB"])
@pytest.mark.parametrize("slug", sorted(posthoc.MEAN_STAGE))
def test_live_path_corrects_the_pooled_mean_not_the_model_leg(dist, slug, tmp_path, monkeypatch):
    player_stats, players = _fixture_rows()
    offers = _offers(players)
    rng = np.random.default_rng(0)
    # A corrector fitted on a cohort that under-predicts, so it lifts the mean.
    served_fit = rng.uniform(0.2, 3.0, 3000)
    blob = posthoc.fit_posthoc(slug, served_fit, served_fit * 1.3 + 0.2)

    base = _score(monkeypatch, tmp_path, dist, "none", None, player_stats, offers, players)
    corrected = _score(monkeypatch, tmp_path, dist, slug, blob, player_stats, offers, players)
    assert not base.empty and len(corrected) == len(base)

    merged = base.merge(corrected, on="Player", suffixes=("_base", "_corr"))
    assert len(merged) == len(base)
    served_base = merged["Projection_base"].to_numpy(dtype=float)
    served_corr = merged["Projection_corr"].to_numpy(dtype=float)

    # The published Projection is the SERVED mean (base deflated by the blended gate),
    # so the corrector must show up on it exactly once, with no gate left over.
    np.testing.assert_allclose(
        served_corr, posthoc.apply_posthoc(slug, blob, served_base), rtol=1e-9, atol=1e-12
    )

    # ... and the pre-fusion arrangement this lane replaced is materially different:
    # the log pool multiplies a correction made on the model leg back down by rho^(1-w).
    # Gate deflation cancels in the comparison, so the base means are enough.
    model_frame = _prob_params(dist, players)
    model_mean = (
        model_frame["total_count"] * model_frame["probs"] / (1 - model_frame["probs"])
    ).to_numpy()
    book = np.full_like(model_mean, _BOOK_EV)

    def _pool(mean):
        return np.exp(_MODEL_WEIGHT * np.log(mean) + (1 - _MODEL_WEIGHT) * np.log(book))

    post = posthoc.apply_posthoc(slug, blob, _pool(model_mean))
    pre = _pool(posthoc.apply_posthoc(slug, blob, model_mean))
    assert not np.allclose(post, pre, rtol=1e-3), (
        "the two stage orders coincide on this fixture — the assertion above is vacuous"
    )

    # The correction has to reach the priced probability, not just the display column.
    assert (served_corr > served_base).all()
    over_base = pd.Series(_served_over(base), index=base["Player"])
    over_corr = pd.Series(_served_over(corrected), index=corrected["Player"])
    assert (over_corr.loc[over_base.index].to_numpy() > over_base.to_numpy()).all()
