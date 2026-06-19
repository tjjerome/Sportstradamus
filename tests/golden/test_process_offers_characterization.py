"""Characterization pin for ``prediction.scoring.process_offers`` routing.

``process_offers`` is stubbed out by the integration suite (test_end_to_end
replaces it), so its own per-league routing is otherwise unpinned. The leaf
collaborators are module globals (``archive``, ``_score_market``,
``find_correlation``) so they monkeypatch cleanly and the same patches drive
both the pre- and post-refactor (nesting-flatten) code. This pins:

- a league IN ``stats`` whose season has started: dedupe players, call
  ``get_depth`` / ``get_volume_stats``, archive each market, score each market;
- a league IN ``stats`` whose season has NOT started: skipped entirely (no
  archive, no depth/volume, no scoring);
- a league NOT in ``stats``: DFS lines archived per market, but no scoring;
- MLB: the extra ``get_volume_stats(..., pitcher=True)`` call;
- the scored offers handed to ``find_correlation`` (order + membership).
"""

from __future__ import annotations

import datetime

import pandas as pd

from sportstradamus.prediction import scoring


class _FakeStats:
    def __init__(self, season_start):
        self.season_start = season_start
        self.depth_calls: list[int] = []
        self.volume_calls: list[bool] = []

    def get_depth(self, offers):
        self.depth_calls.append(len(offers))

    def get_volume_stats(self, offers, pitcher=False):
        self.volume_calls.append(pitcher)


class _FakeArchive:
    def __init__(self):
        self.add_dfs_calls: list[tuple[str, int]] = []

    def add_dfs(self, offers, book, mapping):
        self.add_dfs_calls.append((book, len(offers)))


def test_process_offers_routing(monkeypatch):
    today = datetime.date.today()
    started = today - datetime.timedelta(days=30)
    not_started = today + datetime.timedelta(days=30)

    stats = {
        "NBA": _FakeStats(started),
        "NFL": _FakeStats(not_started),
        "MLB": _FakeStats(started),
    }
    offer_dict = {
        "NBA": {"PTS": [{"Player": "LeBron"}, {"Player": "Curry"}], "REB": [{"Player": "Durant"}]},
        "NFL": {"PASS": [{"Player": "Mahomes"}]},
        "MLB": {"K": [{"Player": "Ohtani"}]},
        "XFL": {"YDS": [{"Player": "X"}]},
    }

    fake_archive = _FakeArchive()
    captured: dict[str, object] = {}

    def fake_score_market(offers, league, market, book, stat_data):
        return [f"scored:{league}:{market}"]

    def fake_find_correlation(
        new_offers, stats_arg, book, *, contest_variant, legacy, corr_sink=None, story_sink=None
    ):
        captured["offers"] = list(new_offers)
        captured["book"] = book
        captured["corr_sink"] = corr_sink
        captured["story_sink"] = story_sink
        return pd.DataFrame({"n": [len(new_offers)]}), pd.DataFrame({"p": []})

    monkeypatch.setattr(scoring, "archive", fake_archive)
    monkeypatch.setattr(scoring, "_score_market", fake_score_market)
    monkeypatch.setattr(scoring, "find_correlation", fake_find_correlation)

    offer_df, _parlays = scoring.process_offers(offer_dict, "Underdog", stats)

    # Only started-season, in-stats leagues are scored; NFL (season not started)
    # and XFL (not in stats) contribute nothing to the scored set.
    assert captured["offers"] == ["scored:NBA:PTS", "scored:NBA:REB", "scored:MLB:K"]
    assert captured["book"] == "Underdog"
    assert offer_df["n"].iloc[0] == 3

    # DFS lines archived for the scored leagues (per market) and the not-in-stats
    # league; never for the season-not-started league.
    assert fake_archive.add_dfs_calls == [
        ("Underdog", 2),  # NBA PTS
        ("Underdog", 1),  # NBA REB
        ("Underdog", 1),  # MLB K
        ("Underdog", 1),  # XFL YDS (not in stats)
    ]

    # Depth/volume only for started, in-stats leagues; players deduped across markets.
    assert stats["NBA"].depth_calls == [3]
    assert stats["NBA"].volume_calls == [False]
    assert stats["MLB"].depth_calls == [1]
    assert stats["MLB"].volume_calls == [False, True]  # MLB adds pitcher=True
    assert stats["NFL"].depth_calls == []
    assert stats["NFL"].volume_calls == []
