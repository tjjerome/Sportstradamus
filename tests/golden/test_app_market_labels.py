"""Pin the app labels that reach a model cell: an unmapped label is archived raw and never scored."""

import pytest

from sportstradamus.helpers.archive import cell_market


@pytest.mark.parametrize(
    ("league", "platform", "label", "cell"),
    [
        ("NHL", "Underdog", "Shots on Goal", "shots"),
        ("NHL", "Underdog", "Faceoffs Won", "faceOffWins"),
        ("NHL", "Sleeper", "faceoffs_won", "faceOffWins"),
        # Underdog renamed these four on 2026-09-29; "Strikeouts" kept its label.
        ("MLB", "Underdog", "Pitching Outs Recorded", "pitching outs"),
        ("MLB", "Underdog", "Pitcher Hits Allowed", "hits allowed"),
        ("MLB", "Underdog", "Pitcher Walks Allowed", "walks allowed"),
        ("MLB", "Underdog", "Pitcher Earned Runs Allowed", "runs allowed"),
    ],
)
def test_app_label_reaches_its_cell(league, platform, label, cell):
    assert cell_market(league, platform, label) == cell
