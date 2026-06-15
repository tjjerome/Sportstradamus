"""Characterization of :meth:`Stats.get_training_matrix` — the per-gameday training
matrix builder (CC 14).

``get_training_matrix`` had no behavioral test (its only mention elsewhere is a
docstring). It is decomposed into a flat orchestrator plus ``_dispatch_volume_stats``
(the volume-stat get_depth/get_volume_stats dispatch) and ``_resolve_player_market_odds``
(the per-player archive EV / line / odds loop); this pin proves that is
behavior-preserving.

Strategy mirrors ``test_get_stats_characterization``: load the **real** NBA gamelog,
skip when absent. To keep the matrix deterministic and pickle-free:

* market is ``"MIN"`` — the only entry in NBA ``volume_stats`` — so the dispatch takes
  the ``get_depth`` branch and never loads the (gitignored) volume-model pickle.
* ``base.datetime.today()`` is frozen so the ``< today`` gameday upper bound is stable
  and the window stays at two immutable past gamedays (2025-03-10 / -11).
* ``base.archive`` is replaced with a deterministic md5-keyed stub for the
  ``get_ev`` / ``get_line`` reads — its values deliberately drive both the
  ``line <= 0`` (→ ``Avg10`` fallback, ``Archived=False``) and ``ev <= 0``
  (→ ``get_ev`` recompute) branches of the odds loop.
* ``base.stat_cv`` is pinned to ``{"NBA": {"MIN": _NBA_MIN_CV}}``. It is otherwise
  derived from ``stat_calibration.json`` — gitignored and rewritten by every
  ``meditate`` run — and feeds the ``Odds`` column (all rows, via ``get_odds``) and
  the recomputed ``EV`` (via ``get_ev``), so without the pin the digest drifts with
  the live calibration. ``stat_dist`` is left live: it is tracked config, so a change
  there is intentional and should legitimately re-pin this test.

The exact frame digest is the master pin; shape / column-hash / archived-count and a
few column sums make a regression diagnosable. Because the matrix embeds every
``get_stats`` feature column, this digest also transitively guards ``base_profile``,
``profile_market``, ``_comp_pairs`` and ``get_depth`` on the historical path.
"""

from __future__ import annotations

import hashlib
from datetime import date as _date
from datetime import datetime as _datetime

import pandas as pd
import pytest

from sportstradamus.stats import StatsNBA, base

_DIGEST_MASK = (1 << 63) - 1


def _digest(frame: pd.DataFrame) -> int:
    numeric = frame.select_dtypes("number").round(6)
    return int(pd.util.hash_pandas_object(numeric, index=True).sum()) & _DIGEST_MASK


def _col_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256("|".join(map(str, frame.columns)).encode()).hexdigest()[:16]


_SLATE_DATE = _date(2025, 3, 10)
_CUTOFF = _date(2025, 3, 9)
_FROZEN_TODAY = _datetime(2025, 3, 12, 12, 0, 0)
# stat_cv is derived from the gitignored, meditate-recomputed stat_calibration.json;
# pinning NBA MIN's coefficient of variation keeps the Odds/EV columns (and the digest)
# deterministic instead of drifting with the live calibration.
_NBA_MIN_CV = 0.33

_GTM_SHAPE = (321, 286)
_GTM_COL_HASH = "d268da7b83ad9449"
_GTM_DIGEST = 1449650711683417532
_GTM_ARCHIVED = 266
_GTM_DATES = ["2025-03-10", "2025-03-11"]
_GTM_EV_SUM = 1368.791
_GTM_LINE_SUM = 7058.5
_GTM_RESULT_SUM = 7605.0


class _FrozenDatetime(_datetime):
    @classmethod
    def today(cls):
        return _FROZEN_TODAY


class _FakeArchive:
    """Deterministic stand-in for the get_ev / get_line reads in the odds loop."""

    @staticmethod
    def _h(key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def get_ev(self, league, market, date, player, at=None):
        if self._h(f"ev|{date}|{player}") % 100 < 15:
            return 0.0
        return round(0.3 + (self._h(f"ev|{date}|{player}") % 600) / 1000.0, 4)

    def get_line(self, league, market, date, player, at=None):
        if self._h(f"ln|{date}|{player}") % 100 < 20:
            return 0.0
        return round(5.0 + (self._h(f"ln|{date}|{player}") % 300) / 10.0, 1)


def _load_or_skip() -> StatsNBA:
    stats = StatsNBA()
    stats.load()
    if stats.gamelog.empty:
        pytest.skip("StatsNBA gamelog bundle not present in this environment")
    day = pd.to_datetime(stats.gamelog[stats.log_strings["date"]]).dt.date
    if not (day == _SLATE_DATE).any():
        pytest.skip(f"StatsNBA gamelog does not cover {_SLATE_DATE}")
    return stats


def test_nba_get_training_matrix(monkeypatch):
    stats = _load_or_skip()
    monkeypatch.setattr(base, "datetime", _FrozenDatetime)
    monkeypatch.setattr(base, "archive", _FakeArchive())
    monkeypatch.setattr(base, "stat_cv", {"NBA": {"MIN": _NBA_MIN_CV}})

    out = stats.get_training_matrix("MIN", cutoff_date=_CUTOFF)

    assert out.shape == _GTM_SHAPE
    assert _col_hash(out) == _GTM_COL_HASH
    assert sorted(out["Date"].unique().tolist()) == _GTM_DATES
    assert int(out["Archived"].sum()) == _GTM_ARCHIVED
    assert round(float(out["EV"].sum()), 3) == _GTM_EV_SUM
    assert round(float(out["Line"].sum()), 4) == _GTM_LINE_SUM
    assert round(float(out["Result"].sum()), 4) == _GTM_RESULT_SUM
    assert _digest(out) == _GTM_DIGEST
