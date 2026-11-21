"""``ship priced``: P1/P2 selection, the push drop, date-clustered intervals, the verdict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.helpers.training_quotes import AUTHENTIC, PRICED_AUTHENTICITY
from sportstradamus.scripts import priced_score as ps

THIN = f"NEUTRAL (fewer than {ps.MIN_DATES} dates)"


def _arm(p: list[float], mean: list[float], lines: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Player": ["a", "b", "c", "d", "e"],
            "Date": "2026-11-01",
            "Result": [3.0, 1.0, 2.0, 0.0, 5.0],
            "Line": lines,
            "QuoteAuthenticity": ["pickem", "authentic", "synthetic", "derived", "authentic"],
            "Odds": [0.5, 0.45, 0.5, 0.5, 0.4],
            "p": p,
            "mean": mean,
        }
    )


def test_p1_keeps_priced_rows_both_arms_scored_at_one_line():
    lines = [2.5, 1.5, 1.5, 0.5, 4.5]
    candidate = _arm([0.6, 0.4, 0.5, 0.3, 0.7], [3.1, 1.2, 1.9, 0.4, 5.2], lines)
    # e's line moved in the baseline's dump; f is in the baseline's alone.
    baseline = pd.concat(
        [
            _arm([0.5] * 5, [2.0] * 5, [*lines[:4], 5.5]),
            _arm([0.5] * 5, [2.0] * 5, lines).head(1).assign(Player="f"),
        ]
    )

    rows, split = ps.p1_rows(candidate, baseline, PRICED_AUTHENTICITY)

    assert rows["Player"].tolist() == ["a", "b"]
    assert split == 1
    assert rows["p_cand"].tolist() == [0.6, 0.4] and rows["p_base"].tolist() == [0.5, 0.5]
    assert rows["mean_cand"].tolist() == [3.1, 1.2] and rows["mean_base"].tolist() == [2.0, 2.0]
    book_rows, _ = ps.p1_rows(candidate, baseline, (AUTHENTIC,))
    assert book_rows["Player"].tolist() == ["b"]


def _p1(**columns) -> pd.DataFrame:
    defaults = {
        "League": "NBA",
        "cell": "NBA_PTS",
        "Date": "2026-11-01",
        "Line": 2.5,
        "QuoteAuthenticity": "pickem",
        "Odds": 0.5,
        "p_cand": 0.5,
        "p_base": 0.5,
        "mean_cand": 2.5,
        "mean_base": 2.5,
    }
    return pd.DataFrame({**defaults, **columns})


def test_pushes_leave_the_brier_statistics_and_stay_in_dmse():
    # The push (Result == Line) carries a P that would dominate either Brier mean.
    p1 = _p1(Result=[3.0, 2.5, 1.0], p_cand=[0.7, 0.99, 0.2], mean_cand=[3.0, 2.0, 1.0])
    p2 = pd.DataFrame(
        {
            "League": "NBA",
            "cell": "NBA_PTS",
            "Date": "2026-11-01",
            "Result": [3.0, 3.5],
            "Line": 3.5,
            "p_cand": [0.3, 0.99],
            "p_base": 0.5,
        }
    )

    changes = ps.row_changes(p1, p2)

    by_statistic = changes.groupby("statistic", observed=True)["change"]
    assert by_statistic.size().to_dict() == {"p1_dmse": 3, "p1_dbrier": 2, "p2_dbrier": 1}
    # (0.7 - 1)^2 - (0.5 - 1)^2 and (0.2 - 0)^2 - (0.5 - 0)^2, in 1e-4.
    assert by_statistic.get_group("p1_dbrier").tolist() == pytest.approx([-1600, -2100])
    assert by_statistic.get_group("p2_dbrier").tolist() == pytest.approx([-1600])


def test_p2_keeps_the_rungs_still_posted_that_both_arms_priced():
    last = pd.Timestamp("2026-11-01 23:30")
    candidate = pd.DataFrame(
        {
            "League": "NBA",
            "Market": "PTS",
            "cell": "NBA_PTS",
            "Player": ["a", "a", "a", "b"],
            "Date": "2026-11-01",
            "Platform": "Underdog",
            "Line": [20.5, 21.5, 19.5, 10.5],
            "last_poll": [last, last, last - pd.Timedelta(hours=3), last],
            "Actual": [22.0, 22.0, 22.0, 9.0],
            "p_serving": [0.6, 0.55, 0.7, 0.4],
        }
    )
    baseline = candidate.head(3).assign(p_serving=[0.5, 0.45, 0.65])

    rungs = ps.p2_rungs(candidate, baseline)

    # 19.5 is a line the app had moved off before the prop's last poll; b's rung is the
    # candidate's alone.
    assert rungs["Line"].tolist() == [20.5, 21.5]
    assert rungs["p_cand"].tolist() == [0.6, 0.55]
    assert rungs["p_base"].tolist() == [0.5, 0.45]
    assert rungs["Result"].tolist() == [22.0, 22.0]
    assert ps.p2_rungs(candidate, pd.DataFrame()).empty


def test_pooled_dmse_scores_each_row_against_its_own_cells_baseline_mse():
    # The candidate cuts a small-scale cell's squared error 75 % and lifts a large-scale
    # cell's 21 %. Pooled on raw squared error the large cell would swamp it (about +21 %).
    dates = [f"2026-10-{day:02d}" for day in range(1, 11)]
    small = _p1(cell="NBA_FG3M", Date=dates, Result=1.0, mean_base=1.1, mean_cand=1.05)
    large = _p1(cell="NBA_PRA", Date=dates, Result=100.0, mean_base=110.0, mean_cand=111.0)

    no_rungs = ps.p2_rungs(pd.DataFrame(), pd.DataFrame())
    changes = ps.row_changes(pd.concat([small, large]), no_rungs)
    table = ps.scope_table(changes, ps.DEFAULT_SEED).set_index(["cell", "statistic"])

    assert table.loc[("NBA_FG3M", "p1_dmse"), "estimate"] == pytest.approx(-75)
    assert table.loc[("NBA_PRA", "p1_dmse"), "estimate"] == pytest.approx(21)
    assert table.loc[(ps.POOLED, "p1_dmse"), "estimate"] == pytest.approx(-27)


def test_values_both_arms_serve_alike_read_no_change():
    # The candidate serves the baseline's mean and P a rounding error away on every row, all
    # on the worse side: float noise whose interval lies wholly above zero must read zero.
    dates = [f"2026-10-{day:02d}" for day in range(1, 26)]
    mean = 2.5 + np.arange(25) / 10
    p1 = _p1(
        Date=dates,
        Result=1.0,
        mean_base=mean,
        mean_cand=mean * (1 + 4e-16),
        p_base=0.3,
        p_cand=0.3 * (1 + 4e-16),
    )
    no_rungs = ps.p2_rungs(pd.DataFrame(), pd.DataFrame())

    table = ps.judge(ps.scope_table(ps.row_changes(p1, no_rungs), ps.DEFAULT_SEED))

    assert table[["estimate", "lo", "hi"]].eq(0.0).all().all()
    assert set(table["verdict"]) == {"NEUTRAL"}


def _dated(statistic: str, n_dates: int, change: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range("2026-10-01", periods=n_dates).strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "League": "NBA",
            "cell": "NBA_PTS",
            "Date": np.repeat(dates, 2),
            "statistic": statistic,
            "change": change,
        }
    )


def test_intervals_resample_whole_game_dates_under_the_seed():
    levels = np.repeat(np.random.default_rng(0).normal(size=30), 2)
    # Each date's two rows sit 1 either side of the date's level.
    spread = _dated("p1_dmse", 30, levels + np.tile([1.0, -1.0], 30))

    table = ps.scope_table(spread, seed=7)

    pd.testing.assert_frame_equal(table, ps.scope_table(spread, seed=7))
    assert not np.allclose(table["lo"], ps.scope_table(spread, seed=8)["lo"])
    # Whole dates carry their rows together, so the spread inside a date never reaches the
    # interval: it matches the levels' alone. A row bootstrap would widen it.
    level_only = ps.scope_table(_dated("p1_dmse", 30, levels), seed=7)
    pd.testing.assert_frame_equal(table, level_only)
    assert (table["lo"] < table["estimate"]).all() and (table["estimate"] < table["hi"]).all()


@pytest.mark.parametrize(
    ("p2_dates", "primary"), [(ps.MIN_DATES - 1, "p1_dmse"), (ps.MIN_DATES, "p2_dbrier")]
)
def test_the_primary_is_p2_dbrier_once_its_rungs_span_min_dates(p2_dates, primary):
    changes = pd.concat(
        [
            _dated("p1_dmse", 30, np.linspace(-1, 1, 60)),
            _dated("p2_dbrier", p2_dates, np.linspace(-1, 1, 2 * p2_dates)),
        ]
    )

    table = ps.judge(ps.scope_table(changes, ps.DEFAULT_SEED))

    assert set(table.loc[table["primary"], "statistic"]) == {primary}


def _table(*scopes: tuple[str, str, int, float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "league": "NHL",
                "cell": cell,
                "statistic": statistic,
                "n_rows": 100,
                "n_dates": n_dates,
                "estimate": (lo + hi) / 2,
                "lo": lo,
                "hi": hi,
            }
            for cell, statistic, n_dates, lo, hi in scopes
        ]
    )


@pytest.mark.parametrize(
    ("scopes", "verdict"),
    [
        ([("p1_dmse", 25, -2.0, -0.5), ("p1_dbrier", 25, -3.0, 1.0)], "KEEP"),
        ([("p1_dmse", 25, -2.0, -0.5), ("p1_dbrier", 25, 0.5, 3.0)], "KILL"),
        ([("p1_dmse", 25, 0.1, 2.0)], "KILL"),
        ([("p1_dmse", 25, -1.0, 1.0), ("p1_dbrier", 25, -3.0, 1.0)], "NEUTRAL"),
        # toward_book is information: wholly above zero, it still kills nothing.
        ([("p1_dmse", 25, -2.0, -0.5), ("toward_book", 25, 0.5, 1.0)], "KEEP"),
        # Too few dates in the primary's rows decide nothing, a would-be KILL included.
        ([("p1_dmse", ps.MIN_DATES - 1, 0.5, 2.0)], THIN),
        # Nor does a thin kill statistic beside a primary that spans enough dates.
        ([("p1_dmse", 25, -2.0, -0.5), ("p2_dbrier", ps.MIN_DATES - 1, 0.5, 3.0)], "KEEP"),
    ],
)
def test_league_verdict_truth_table(scopes, verdict):
    table = ps.judge(_table(*((ps.POOLED, *scope) for scope in scopes)))

    assert set(table["verdict"]) == {verdict}


def test_a_cell_reads_its_leagues_primary_on_its_own_rows():
    table = ps.judge(
        _table(
            (ps.POOLED, "p1_dmse", 30, -1.0, 1.0),
            (ps.POOLED, "p2_dbrier", 25, -20.0, -5.0),
            ("NHL_shots", "p1_dmse", 30, -1.0, 1.0),
            ("NHL_shots", "p2_dbrier", 25, -30.0, -2.0),
            # Wins on P1 but holds no posted rung, so its primary has no dates.
            ("NHL_goals", "p1_dmse", 30, -3.0, -1.0),
            ("NHL_hits", "p2_dbrier", 25, 1.0, 9.0),
        )
    )

    assert set(table.loc[table["primary"], "statistic"]) == {"p2_dbrier"}
    assert table.groupby("cell")["verdict"].first().to_dict() == {
        ps.POOLED: "KEEP",
        "NHL_goals": THIN,
        "NHL_hits": "KILL",
        "NHL_shots": "KEEP",
    }


@pytest.mark.parametrize(
    ("baseline_args", "message"),
    [
        ([], "--baseline-book alone"),
        (["--baseline-book", "--baseline-models", "{dir}"], "--baseline-book alone"),
        (["--baseline-test-sets", "{dir}"], "--baseline-book alone"),
        (["--baseline-test-sets", "{dir}", "--baseline-models", "{dir}"], "--archive is required"),
    ],
)
def test_contradictory_baseline_options_are_usage_errors(tmp_path, baseline_args, message):
    args = [
        "--candidate-test-sets",
        str(tmp_path),
        "--candidate-models",
        str(tmp_path),
        "--start",
        "2026-11-01",
        *(arg.format(dir=tmp_path) for arg in baseline_args),
    ]

    result = CliRunner().invoke(ps.main, args)

    assert result.exit_code == 2, result.output
    assert message in result.output
