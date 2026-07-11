"""Characterization pin for ``collectors.runner._backfill_week``.

Extracted from ``backfill``'s (season × week × spec) loop in the nesting sweep.
The generic driver takes its per-cell collaborators (``would_skip``,
``fetch_and_write_one``, the pacing pause) as arguments; the pause and fetch
are module globals so they monkeypatch cleanly. Pins: every spec is fetched
and ticks the bar; the pacing pause fires only for specs that are NOT skipped;
``prev_week_key`` advances to this week once any spec in it is
fetched-not-skipped.
"""

from __future__ import annotations

from sportstradamus.collectors import runner


class _Bar:
    def __init__(self):
        self.updates = 0

    def update(self, n):
        self.updates += n


def test_backfill_week_paces_unskipped_and_threads_prev_key(monkeypatch):
    pauses: list[tuple] = []
    monkeypatch.setattr(
        runner, "_backfill_pause", lambda prev, cur, **k: pauses.append((prev, cur))
    )
    monkeypatch.setattr(
        runner, "fetch_and_write_one", lambda spec, **k: f"result:{spec}"
    )

    bar = _Bar()
    results: list = []
    prev = runner._backfill_week(
        ["s1", "s2"],
        2024,
        1,
        None,
        results,
        bar,
        make_fetch_one=lambda spec, s, w: (None, None),
        request_body_for=lambda spec, s, w: None,
        path_for_cell=lambda spec, s, w: None,
        would_skip=lambda spec, s, w: spec == "s2",
        transform=lambda body: None,
        log=None,
        refetch=False,
        request_range=(2, 8),
        week_range=(8, 28),
    )

    assert results == ["result:s1", "result:s2"]
    assert bar.updates == 2
    assert pauses == [(None, (2024, 1))]  # only s1 (not skipped) triggers a pause
    assert prev == (2024, 1)  # s1 fetched-not-skipped -> prev_week_key advances
