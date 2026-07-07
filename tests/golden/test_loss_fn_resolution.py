"""Unit tests for the loss- and distribution-resolution helpers in ``training.pipeline``.

The Operation Ship 75 search sweeps the distribution training loss (nll ↔ crps) as a search
axis via ``--dist-training-loss``. The flag carries an ``"auto"`` sentinel that means
"per-family default" — crps for the SkewNormal continuous branch, nll for the count
branch — so the default run reproduces production behavior byte-for-byte while an
explicit nll/crps overrides every family.

Distribution selection is likewise config-first: a cell's stat_meta ``dist`` is authoritative
(``_resolve_dist``), with the data-driven ``global_mean`` / zero-rate rule (``_data_driven_dist``)
as the fallback for an unset / ``"auto"`` cell.
"""

import logging

import pytest

from sportstradamus.training.pipeline import (
    LOSS_AUTO,
    _data_driven_dist,
    _resolve_dist,
    _resolve_loss_fn,
)


def test_auto_sentinel_yields_the_per_family_default():
    assert _resolve_loss_fn("crps", LOSS_AUTO) == "crps"
    assert _resolve_loss_fn("nll", LOSS_AUTO) == "nll"


def test_explicit_override_wins_over_the_family_default():
    assert _resolve_loss_fn("crps", "nll") == "nll"
    assert _resolve_loss_fn("nll", "crps") == "crps"


def test_data_driven_dist_by_mean_then_zero_rate():
    assert _data_driven_dist(5.0, 0.0) == "SkewNormal"  # mean >= 2 → continuous
    assert _data_driven_dist(0.7, 0.57) == "ZINB"  # mean < 2, zeros clear the ZINB gate
    assert _data_driven_dist(0.1, 0.01) == "NegBin"  # mean < 2, too few zeros for ZINB


def test_resolve_dist_falls_back_to_data_when_unset_or_auto():
    assert _resolve_dist(None, "ZINB", 0.7, 0.57, "NHL", "blocked") == "ZINB"
    assert _resolve_dist("auto", "SkewNormal", 5.0, 0.0, "NBA", "PTS") == "SkewNormal"


def test_resolve_dist_configured_is_authoritative_and_warns_only_on_disagreement(caplog):
    with caplog.at_level(logging.WARNING):
        assert _resolve_dist("ZINB", "ZINB", 0.7, 0.57, "NHL", "blocked") == "ZINB"
    assert not caplog.records  # agreement: forced == data-driven, silent

    with caplog.at_level(logging.WARNING):
        assert _resolve_dist("SkewNormal", "ZINB", 0.7, 0.57, "NHL", "blocked") == "SkewNormal"
    assert any("forced dist=SkewNormal" in r.getMessage() for r in caplog.records)  # trains + warns


def test_resolve_dist_unknown_family_raises():
    with pytest.raises(ValueError, match="not a forceable family"):
        _resolve_dist("Gamma", "ZINB", 0.7, 0.57, "NHL", "blocked")
