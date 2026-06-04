"""Characterization pin for ``fantasypoints.cli._backfill_week``.

Extracted from ``backfill``'s (season × week × spec) loop in the nesting sweep.
The per-spec collaborators are module globals, so they monkeypatch cleanly.
Pins: every spec is fetched and ticks the bar; the pacing pause fires only for
specs that are NOT skipped; ``prev_week_key`` advances to this week once any
spec in it is fetched-not-skipped.
"""

from __future__ import annotations

from sportstradamus.fantasypoints import cli as fp_cli


class _Bar:
    def __init__(self):
        self.updates = 0

    def update(self, n):
        self.updates += n


def test_backfill_week_paces_unskipped_and_threads_prev_key(monkeypatch):
    pauses: list[tuple] = []
    monkeypatch.setattr(fp_cli, "_would_skip", lambda spec, **k: spec == "s2")
    monkeypatch.setattr(fp_cli, "_backfill_pause", lambda prev, cur, **k: pauses.append((prev, cur)))
    monkeypatch.setattr(fp_cli, "_fetch_and_write_one", lambda spec, client, **k: f"result:{spec}")

    bar = _Bar()
    results: list = []
    prev = fp_cli._backfill_week(
        ["s1", "s2"],
        None,
        2024,
        1,
        None,
        results,
        bar,
        mode="weekly",
        log=None,
        use_cache=True,
        refetch=False,
        request_range=(2, 8),
        week_range=(8, 28),
    )

    assert results == ["result:s1", "result:s2"]
    assert bar.updates == 2
    assert pauses == [(None, (2024, 1))]  # only s1 (not skipped) triggers a pause
    assert prev == (2024, 1)              # s1 fetched-not-skipped -> prev_week_key advances
