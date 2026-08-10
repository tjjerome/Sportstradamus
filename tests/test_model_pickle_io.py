"""Unit tests for the model-pickle path/prune helpers in ``helpers/io.py``.

The production pickle path (``data/models/{league}_{market}.mdl``) is the single
live-gate at inference: ``model_prob`` loads it or skips the market.
``prune_model_pickle`` deletes it so a withheld cell goes dark, and
``model_pickle_path`` is the shared builder both training and prediction use to
agree on the filename.
"""

from __future__ import annotations

from sportstradamus.helpers import io


def test_model_pickle_path_joins_league_market_and_hyphenates_spaces() -> None:
    """Path is ``models/{league}_{market}.mdl`` with spaces turned into hyphens."""
    path = io.model_pickle_path("NFL", "rushing tds")
    assert path.name == "NFL_rushing-tds.mdl"
    assert path.parent.name == "models"


def test_prune_removes_existing_pickle(tmp_path, monkeypatch) -> None:
    """Pruning an existing pickle deletes it and reports that it existed."""
    target = tmp_path / "NBA_STL.mdl"
    target.write_bytes(b"stale")
    monkeypatch.setattr(io, "model_pickle_path", lambda _lg, _mkt: target)
    assert io.prune_model_pickle("NBA", "STL") is True
    assert not target.exists()


def test_prune_missing_pickle_is_noop(tmp_path, monkeypatch) -> None:
    """Pruning a cell with no pickle reports False and raises nothing."""
    target = tmp_path / "NBA_PTS.mdl"
    monkeypatch.setattr(io, "model_pickle_path", lambda _lg, _mkt: target)
    assert io.prune_model_pickle("NBA", "PTS") is False
