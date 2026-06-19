"""Derivation pins for the QW-5 comp-pool harvest features (model track §6.3).

QW-5 adds, reusing the cached ``_comp_pairs`` (weight = 1/(1+dist)) and the per-player
recent-slope machinery (``_player_recent_slope``):
  * ``comps std`` / ``comps trend`` (profile-level, in ``_apply_comp_features``;
    ``comps trend`` distance-weights the comps' recent market slope);
  * ``Player comps raw`` (raw-scale opponent-conditional mean) and
    ``Player comps z recent`` (recency-weighted z), in ``_nonmlb_comp_features``.

These pin the arithmetic on seeded comp pairs + a synthetic gamelog. The recency
machinery is exercised two ways: with all comp games sharing a date the recency
weight is uniform, so ``Player comps z recent`` must collapse onto the distance-only
``Player comps z``; with a recent/old split the raw mean must be pulled toward the
recent game.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from sportstradamus.stats import StatsNBA

GAMEDAY = date(2025, 3, 15)
# C1 is the close comp (dist 0.25 -> weight 0.8), C2 the far one (dist 1.5 -> 0.4).
_PAIRS = pd.DataFrame(
    {
        "player": ["T1", "T1"],
        "comp": ["C1", "C2"],
        "position": ["P", "P"],
        "dist": [0.25, 1.5],
        "weight": [0.8, 0.4],
    }
)


def _seeded_nba(monkeypatch) -> StatsNBA:
    s = StatsNBA()
    monkeypatch.setattr(s, "_ensure_comps", lambda date=None: None)
    s._comp_pairs_cache[None] = _PAIRS  # _current_comps_key defaults to None
    return s


def test_comps_std_and_trend_are_distance_weighted(monkeypatch):
    s = _seeded_nba(monkeypatch)
    s.playerProfile = pd.DataFrame(index=["T1", "C1", "C2"])
    all_mean = pd.Series({"T1": 15.0, "C1": 20.0, "C2": 10.0})
    # comps trend distance-weights the comps' per-player recent market slope.
    all_trend = pd.Series({"T1": 0.0, "C1": 2.0, "C2": -1.0})

    s._apply_comp_features(GAMEDAY, "PTS", all_mean, all_trend)

    wmean = (0.8 * 20 + 0.4 * 10) / 1.2
    wvar = (0.8 * (20 - wmean) ** 2 + 0.4 * (10 - wmean) ** 2) / 1.2
    assert s.playerProfile.loc["T1", "comps std"] == pytest.approx(wvar**0.5)
    # trend = distance-weighted mean of the comps' growth slopes.
    assert s.playerProfile.loc["T1", "comps trend"] == pytest.approx(
        (0.8 * 2.0 + 0.4 * -1.0) / 1.2
    )


def _run_nonmlb(monkeypatch, rows) -> pd.DataFrame:
    s = _seeded_nba(monkeypatch)
    ls = s.log_strings
    s.short_gamelog = pd.DataFrame(
        [{ls["player"]: p, ls["opponent"]: o, ls["date"]: d, "PTS": v} for p, o, d, v in rows]
    )
    stats = pd.DataFrame({"Player position": [1]}, index=["T1"])  # position 1 -> "P"
    defstats = pd.DataFrame(index=["T1"])
    s._nonmlb_comp_features(stats, defstats, "PTS", {"T1": "DET"}, GAMEDAY)
    return stats


def test_comps_raw_is_distance_weighted_opp_mean_when_recency_is_flat(monkeypatch):
    d5 = (GAMEDAY - timedelta(days=5)).isoformat()
    # All comp games share one date -> recency weight is a common factor that cancels.
    rows = [
        ("C1", "DET", d5, 20.0),
        ("C1", "CHI", d5, 10.0),  # off-opponent game: only feeds the within-comp z
        ("C2", "DET", d5, 8.0),
        ("C2", "CHI", d5, 4.0),
    ]
    out = _run_nonmlb(monkeypatch, rows)
    # raw = (0.8*20 + 0.4*8) / 1.2 -- only the DET games, distance-weighted.
    assert out.loc["T1", "Player comps raw"] == pytest.approx((0.8 * 20 + 0.4 * 8) / 1.2)
    # With a uniform date the recency-weighted z collapses onto the distance-only z.
    assert out.loc["T1", "Player comps z recent"] == pytest.approx(out.loc["T1", "Player comps z"])


def test_recency_weighting_pulls_raw_toward_the_recent_game(monkeypatch):
    recent = (GAMEDAY - timedelta(days=1)).isoformat()
    old = (GAMEDAY - timedelta(days=200)).isoformat()
    rows = [
        ("C1", "DET", recent, 20.0),  # recent high
        ("C1", "DET", old, 0.0),  # old low
        ("C1", "CHI", recent, 5.0),
        ("C2", "DET", recent, 8.0),
        ("C2", "CHI", recent, 4.0),
    ]
    out = _run_nonmlb(monkeypatch, rows)
    # Flat (distance-only) raw would be (0.8*mean(20,0) + 0.4*8)/1.2 = 9.33; recency
    # weights the recent 20 far above the 200-day-old 0, pulling raw well past it.
    assert out.loc["T1", "Player comps raw"] > 12.0
