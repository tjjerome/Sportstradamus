"""Pin the Phase-1.5/2 postseason-clean contract on ``StatsNFL._lookback_windows``.

``_lookback_windows`` is the single place the per-game-date comp pool
(Phase 1.5) AND the team-grain FP feature pool (Phase 2) resolve their
``(season, start_week, end_week)`` windows from. The 2026-05-25 in-repo
research diagnostic validated the operator's intuition that postseason
rows should NOT blend into priors for a regular-season target -- the
distribution shift on Phase-2 features (PROE +8.4 pp,
def_two_high_pct +7.7 pp on 2022 W11-18 vs W19-22) is large enough to
bias the prior in the wrong direction.

The current implementation already encodes the rule via the
``NFL_REGULAR_SEASON_WEEKS = 18`` cap on the prior-season window and the
``target_week - 1`` cap on the current-season window. This test pins
that behavior: a future "simplification" of ``_lookback_windows`` (e.g.
extending the prior-season window to ``COMP_LOOKBACK_PRIOR_SEASON_TO_WEEK
= 22`` to "use more data") would silently regress the rule, and this
test exists to catch that. The symmetric-rule postseason-target case
(week >= 19 includes its own season's postseason in the prior) is also
pinned so a future tightening doesn't accidentally kill that path
before the operator decides to price postseason props.
"""

import pytest

from sportstradamus.stats import StatsNFL
from sportstradamus.stats.nfl import NFL_REGULAR_SEASON_WEEKS


@pytest.mark.parametrize(
    ("target_season", "target_week", "expected"),
    [
        # Post-cutoff regular-season target -- current season only, ends at W-1.
        # Postseason cannot be reached because target_week - 1 <= 17.
        (2025, 7, [(2025, 1, 6)]),
        # Pre-cutoff regular-season target -- prior-season back half BLENDED
        # with current partial. Prior window clips at 18, so 2024 postseason
        # rows (weeks 19-22) cannot leak into the 2025-W3 prior pool.
        (2025, 3, [(2024, 11, 18), (2025, 1, 2)]),
        # Pre-cutoff regular-season opener -- prior only, no current partial.
        (2025, 1, [(2024, 11, 18)]),
        # Regular-season finale -- last regular-season week. End cap is 17.
        (2025, 18, [(2025, 1, 17)]),
        # Postseason target W19 (wildcard) -- the symmetric rule's INCLUDE
        # branch fires as a side effect of target_week - 1: the prior is
        # weeks 1..18, which is regular-season only because no postseason
        # has yet been played in target_season. This is correct, not lucky.
        (2025, 19, [(2025, 1, 18)]),
        # Postseason target W22 (Super Bowl) -- the prior pool is weeks 1..21,
        # which DOES include postseason rows (weeks 19, 20, 21). Per the
        # symmetric rule a postseason target sees prior postseason rows;
        # this case documents that behavior in the absence of an explicit
        # ``target_is_postseason`` branch.
        (2025, 22, [(2025, 1, 21)]),
    ],
)
def test_lookback_windows_is_postseason_clean_for_reg_targets(
    target_season: int, target_week: int, expected: list[tuple[int, int, int]]
) -> None:
    assert StatsNFL._lookback_windows(target_season, target_week) == expected


def test_lookback_windows_legacy_fallback_caps_at_regular_season() -> None:
    """``target_week is None`` -> legacy per-season full-season aggregates.

    Every returned window's ``end_week`` must equal ``NFL_REGULAR_SEASON_WEEKS``
    (18). This is the path ``optimize_comp_weights`` still uses; allowing
    weeks 19-22 in here would silently shift the comp weights toward the
    playoff-team subset (selection bias on top of the distribution shift).
    """
    windows = StatsNFL._lookback_windows(2025, None)
    assert len(windows) == 4  # prior 3 + current
    assert {season for season, _, _ in windows} == {2022, 2023, 2024, 2025}
    for _, start_week, end_week in windows:
        assert start_week == 1
        assert end_week == NFL_REGULAR_SEASON_WEEKS


def test_lookback_windows_never_leaks_postseason_into_reg_season_target() -> None:
    """Property test: for every regular-season target week (1..18), every
    returned window's ``end_week`` is <= ``NFL_REGULAR_SEASON_WEEKS``.

    This is the structural guarantee the research brief's recommendation
    asks for. Encoding it as a property test (not just the parametrized
    point cases above) prevents a refactor that, say, switches the
    prior-season window to "previous season's complete data" from
    silently violating the rule.
    """
    for target_week in range(1, NFL_REGULAR_SEASON_WEEKS + 1):
        windows = StatsNFL._lookback_windows(2025, target_week)
        for season, start_week, end_week in windows:
            assert end_week <= NFL_REGULAR_SEASON_WEEKS, (
                f"target_week={target_week} leaked postseason via "
                f"window=({season}, {start_week}, {end_week})"
            )
