"""Star lean: NFL P(Under) on the top quarter of lines against what happened.

Reads gitignored dev files: each served NFL cell's model file
(``src/sportstradamus/data/models/NFL_<slug>.mdl``, its presence only) and the held-out test
set the same retrain wrote (``src/sportstradamus/data/test_sets/NFL_<slug>.csv``).

Pins, over the test rows dated 2026-06-01 or later that did not push, in the top quarter of
each cell's lines on each date (``Line`` ``rank(pct=True, method="average") >= 0.75`` within
cell and date, ranked after that filter): the served P(Under), ``1 - P``, runs at most 1.5
points above the realized Under rate, and the fused mean ``Blended_EV`` reaches at least 98 %
of the outcome. The 2026-10-15 retrain reads gap +0.037 and ratio 0.965 on 1,253 rows over 16
dates, so the test is marked an expected failure; an XPASS means the lean is fixed and the
marker comes off.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.diagnostics

DATA = Path(__file__).resolve().parents[2] / "src" / "sportstradamus" / "data"
START = "2026-06-01"  # this season's held-out rows; earlier rows are last season's
TOP_TIER = 0.75  # the top quarter of a cell's lines on a date stands in for its stars
MAX_GAP = 0.015  # served P(Under) may run 1.5 points over the Under rate, no more
MIN_RATIO = 0.98  # the fused mean must reach 98 % of the outcome on the top tier
MIN_ROWS = 1000  # below this many top-tier rows the pooled read is too thin to judge
MIN_DATES = 12  # and below this many game dates the day-block CI is not an interval
N_BOOT = 2000  # day-block bootstrap draws for the CIs in the failure message


def _top_tier() -> pd.DataFrame:
    """Top-tier rows of every served NFL cell: a model file with a matching test set."""
    frames = []
    for model in sorted((DATA / "models").glob("NFL_*.mdl")):
        path = DATA / "test_sets" / f"{model.stem}.csv"
        if not path.exists():
            continue
        rows = pd.read_csv(path, usecols=["Date", "Line", "Result", "P", "Blended_EV"])
        rows = rows[rows["Date"].ge(START) & rows["Result"].ne(rows["Line"])]
        pct = rows.groupby("Date")["Line"].rank(pct=True, method="average")
        frames.append(rows[pct >= TOP_TIER].assign(cell=model.stem))
    return pd.concat(frames, ignore_index=True)


# raises=AssertionError keeps the sample guard and a broken loader as real failures.
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="star lean: NFL top-line-quartile P(Under) runs high; flips at the retrain that fixes it",
)
def test_nfl_top_line_quartile_under_read_matches_outcomes():
    if not any((DATA / "models").glob("NFL_*.mdl")):
        pytest.skip("no NFL model files on this box")
    top = _top_tier()
    top["resid"] = (1 - top["P"]) - (top["Result"] < top["Line"])
    top["one"] = 1.0
    dates = top["Date"].nunique()
    if len(top) < MIN_ROWS or dates < MIN_DATES:
        pytest.fail(f"sample too thin: {len(top)} top-tier rows over {dates} dates")

    gap = top["resid"].mean()
    ratio = top["Blended_EV"].sum() / top["Result"].sum()
    # Message only: today's gap CI holds 0, so a CI clause could never fail.
    days = top.groupby("Date")[["resid", "one", "Blended_EV", "Result"]].sum().to_numpy()
    picks = np.random.default_rng(7).integers(0, len(days), (N_BOOT, len(days)))
    boot = days[picks].sum(axis=1)
    gap_lo, gap_hi = np.percentile(boot[:, 0] / boot[:, 1], [2.5, 97.5])
    ratio_lo, ratio_hi = np.percentile(boot[:, 2] / boot[:, 3], [2.5, 97.5])
    cells = top.groupby("cell").agg(
        n=("resid", "size"), gap=("resid", "mean"), ev=("Blended_EV", "sum"), out=("Result", "sum")
    )
    cells["ratio"] = cells["ev"] / cells["out"]
    message = (
        f"top-tier gap {gap:+.3f} [{gap_lo:+.3f}, {gap_hi:+.3f}] (max {MAX_GAP:+.3f}); "
        f"ratio {ratio:.3f} [{ratio_lo:.3f}, {ratio_hi:.3f}] (min {MIN_RATIO:.2f}); "
        f"{len(top)} rows over {dates} dates; day-block 95 % CIs\n"
        + cells[["n", "gap", "ratio"]].round(3).to_string()
    )
    assert gap <= MAX_GAP and ratio >= MIN_RATIO, message
