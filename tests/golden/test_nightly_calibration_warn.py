"""Nightly calibration-divergence WARN fires on the poisoned-consensus signature.

The 2026-08 WNBA PRA book-consensus poisoning (predicted over rate 0.743 vs
empirical 0.640 over 3292 settled offers) is the motivating case: it must fire,
while small-gap and below-N cells stay quiet.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from sportstradamus import nightly
from sportstradamus.nightly import _empty_live_metrics_frame, _warn_calibration_divergence


def _row(league, market, *, n_settled, predicted, empirical, window_days=30):
    return {
        "league": league,
        "market": market,
        "window_days": window_days,
        "n_settled": n_settled,
        "predicted_over_rate": predicted,
        "empirical_over_rate": empirical,
    }


@pytest.fixture
def plain_logger(monkeypatch):
    # The structured logger disables propagation; swap in a plain stdlib
    # logger for the test so caplog can capture the WARN records.
    plain = logging.getLogger("nightly-cal-test")
    plain.setLevel(logging.WARNING)
    monkeypatch.setattr(nightly, "logger", plain)
    return plain


def test_warn_fires_only_on_divergent_30d_cell(caplog, plain_logger):
    metrics = pd.DataFrame(
        [
            _row("WNBA", "PRA", n_settled=3292, predicted=0.743, empirical=0.640),
            _row("WNBA", "PRA", n_settled=700, predicted=0.743, empirical=0.640, window_days=7),
            _row("NBA", "PTS", n_settled=2000, predicted=0.520, empirical=0.500),
            _row("NHL", "shots", n_settled=40, predicted=0.900, empirical=0.500),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="nightly-cal-test"):
        _warn_calibration_divergence(metrics)

    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    msg = warns[0].getMessage()
    assert "WNBA" in msg
    assert "PRA" in msg
    assert "predicted_over_rate=0.743" in msg
    assert "empirical_over_rate=0.640" in msg
    assert "n_settled=3292" in msg


def test_warn_silent_on_empty_metrics(caplog, plain_logger):
    with caplog.at_level(logging.WARNING, logger="nightly-cal-test"):
        _warn_calibration_divergence(_empty_live_metrics_frame())
    assert not caplog.records
