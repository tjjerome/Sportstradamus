"""Rule PR-1 (``training.model_strategy.priced_select``): tiers, margin, ties and the harm guard."""

import pandas as pd
import pytest

from sportstradamus.helpers.training_quotes import AUTHENTIC, PRICED_AUTHENTICITY
from sportstradamus.training.model_strategy import confirm, priced_select, sweep
from sportstradamus.training.model_strategy.confirm_runs import BlindRun


def _rows(n_dates: int, quote: str = AUTHENTIC) -> pd.DataFrame:
    """One quoted selection-half row per game date."""
    dates = [f"2026-01-{day:02d}" for day in range(1, n_dates + 1)]
    return pd.DataFrame({"Date": dates, "QuoteAuthenticity": quote, "Line": 10.5})


@pytest.mark.parametrize(
    ("rows", "classes", "p2_dates", "tier"),
    [
        (_rows(19), PRICED_AUTHENTICITY, 0, "U"),
        (_rows(20), PRICED_AUTHENTICITY, 19, "P1"),
        (_rows(5), PRICED_AUTHENTICITY, 20, "P2"),
        (_rows(25, "pickem"), PRICED_AUTHENTICITY, 0, "P1"),
        # A withheld cell meets the book, so only its authentic rows count.
        (_rows(25, "pickem"), (AUTHENTIC,), 0, "U"),
    ],
)
def test_a_tier_needs_twenty_game_dates(rows, classes, p2_dates, tier):
    assert priced_select.tier(rows, classes, p2_dates) == tier


def _nominees(**shrunk: float) -> pd.DataFrame:
    """Eligible nominees in the order given: s 1, one axis change and equal slack each."""
    return pd.DataFrame({"eligible": True, "shrunk": shrunk, "s": 1.0, "axes": 1, "slack": 0.1})


def test_a_switch_must_beat_the_incumbent_by_half_its_standard_error():
    assert priced_select.pick(_nominees(a=-0.51))[0] == "a"
    # At the margin exactly the incumbent stays: a tie goes to the incumbent.
    assert priced_select.pick(_nominees(a=-0.5))[0] is None
    # An ineligible nominee never switches, whatever its contrast.
    assert priced_select.pick(_nominees(a=-3.0, b=-1.0).assign(eligible=[False, True]))[0] == "b"


def test_nominees_within_a_quarter_s_tie_and_the_fewest_axis_changes_win():
    nominees = _nominees(a=-2.0, b=-1.8, c=-1.7).assign(axes=[3, 1, 0])
    # c lies 0.3 s behind the best, outside the band; of a and b, b changes fewer axes.
    assert priced_select.pick(nominees)[0] == "b"
    assert priced_select.pick(nominees.assign(axes=1, slack=[0.1, 0.3, 0.5]))[0] == "b"


def _test_half(**reads: tuple[float, float]) -> pd.DataFrame:
    """A test-half contrast: per statistic its (estimate, interval low), s 1 over 25 dates."""
    return pd.DataFrame(
        [
            {"estimate": est, "lo": lo, "hi": est + 1.0, "s": 1.0, "n_dates": 25}
            for est, lo in reads.values()
        ],
        index=pd.Index(list(reads), name="statistic"),
    )


def test_the_harm_guard_reverts_a_pick_reading_worse_than_one_standard_error():
    assert priced_select.harm(_test_half(p1_dmse=(0.99, -0.5)), "P1", 0.3, 0.3) == []
    assert priced_select.harm(_test_half(p1_dmse=(1.01, -0.5)), "P1", 0.3, 0.3) != []
    # The KILL leg: an interval wholly above zero reverts under +1s too, on tier P2's guard as well.
    assert priced_select.harm(_test_half(p1_dmse=(0.5, 0.1)), "P1", 0.3, 0.3) != []
    both = _test_half(p2_dbrier=(-3.0, -5.0), p1_dmse=(0.5, 0.1))
    assert priced_select.harm(both, "P2", 0.3, 0.3) == ["p1_dmse interval above 0 [+0.100, +1.500]"]


def test_a_refit_whose_blend_weight_drifted_must_read_better_than_the_incumbent():
    worse = _test_half(p1_dmse=(0.5, -0.5))
    assert priced_select.harm(worse, "P1", 0.30, 0.44) == []
    assert priced_select.harm(worse, "P1", 0.30, 0.46) != []
    assert priced_select.harm(_test_half(p1_dmse=(-0.01, -0.5)), "P1", 0.30, 0.46) == []


def test_a_tier_u_cell_keeps_the_slack_ranking(monkeypatch):
    """No priced baseline: TPE maximizes raw slack and confirm refits the best-slack blind pass."""
    row = {"slack": 0.2, "ships": True, "priced_shrunk": -1.0, "priced_mean_ratio": 1.0}
    assert sweep._objective_value(row, None) == -0.2
    priced = priced_select.Baseline("P1", None, 10.0)
    assert sweep._objective_value(row, priced) == -1.0
    assert sweep._objective_value({**row, "priced_mean_ratio": 2.5}, priced) > 1e8

    monkeypatch.setattr(sweep, "_cell_context", lambda league, market: None)
    monkeypatch.setattr(sweep, "_priced_baseline", lambda context, dump: None)
    cand = {"league": "WNBA", "market": "AST"}
    measured = [
        (label, [BlindRun(cand, label, label, failed, {"slack": slack, "model_weight": 0.5})])
        for label, slack, failed in (("a", 0.3, ["g4"]), ("b", 0.2, []), ("c", 0.1, []))
    ]
    assert confirm._top_blind_passer(measured, None)[::2] == ("b", {"priced_tier": "U"})
