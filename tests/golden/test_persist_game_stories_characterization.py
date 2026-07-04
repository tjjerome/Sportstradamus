"""Characterization pins for ``prediction.persist.write_current_game_stories``.

The writer is a passthrough to the atomic parquet writer (no normalize step), so
the pins are: an empty slate still writes a header-only, column-stable frame (the
dashboard reads it without a torn schema), and a real frame reaches the writer
untouched at the canonical path.
"""

import pandas as pd

from sportstradamus.helpers.io import CURRENT_GAME_STORIES_PATH
from sportstradamus.prediction import persist
from sportstradamus.prediction.stories import build_game_stories
from sportstradamus.prediction.stories.menu import _STORY_COLS


def _capture(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        persist, "_atomic_write_parquet", lambda df, path: captured.update(df=df, path=path)
    )
    return captured


def test_empty_slate_writes_header_only_frame(monkeypatch):
    captured = _capture(monkeypatch)
    empty = build_game_stories([], pd.DataFrame(), pd.DataFrame(), None)
    persist.write_current_game_stories(empty)
    assert captured["path"] == CURRENT_GAME_STORIES_PATH
    assert captured["df"].empty
    assert list(captured["df"].columns) == _STORY_COLS


def test_writer_passes_frame_through_untouched(monkeypatch):
    captured = _capture(monkeypatch)
    frame = pd.DataFrame(
        [
            {
                "platform": "Underdog",
                "League": "NBA",
                "Game": "AAA/BBB",
                "story_id": "AAA/BBB#0",
                "objective": "builder",
                "headline": "AAA/BBB tilts toward points",
                "legs": [
                    {
                        "player": "Player0", "team": "AAA", "market": "Points", "stat": "PTS",
                        "bet": "Over", "line": 5.5, "league": "NBA", "game": "AAA/BBB",
                        "date": "2026-06-12", "platform": "Underdog", "win_prob": 0.60,
                        "boost": 1.0, "push_prob": 0.0, "kelly": 0.13,
                    }
                ],
                "joint_p": 0.42,
                "model_ev": 1.26,
                "kelly_stake": 0.13,
                "bet_size": 2,
                "Date": "2026-06-12",
                "dek": "These 2 legs move together",
            }
        ]
    )
    persist.write_current_game_stories(frame)
    assert captured["df"] is frame  # no copy, no normalize
    assert list(captured["df"].columns) == _STORY_COLS
