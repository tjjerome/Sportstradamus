"""PR-1: a priced cell's recipe is chosen on the rows we price, shrunk against the winner's curse.

The rule is pre-registered in the live-gap-2 brief ``researcher_priced_recipe_selection.md``.
A cell's tier is fixed from the selection half's counts before any contrast: ``P2`` when its
posted Underdog/Sleeper rungs span ``MIN_DATES`` game dates, else ``P1`` when its pick'em and
authentic rows do, else ``U`` (unpriced: gate slack ranks it, as before). A candidate's contrast
with the incumbent is ``ship priced``'s own (:mod:`sportstradamus.scripts.priced_score`): its
standard error is inflated by the player design effect and the estimate shrunk toward zero by it,
a switch must clear a margin, a tie keeps the incumbent, and the one test-half read guards against
harm. A withheld cell's incumbent is the book on its authentic rows.

The selection half is a blind cross-fit dump. Its ``P`` is out of fold while the run's pickle
holds the whole-fit calibration, so re-serving it at a ladder rung would score each row against
state fitted on it. That half therefore ranks every priced tier on its own-line squared error
(``SELECTION_STATISTIC``), read off the dump as training served it; ``p2_dbrier`` is read on the
canonical refit's test half only.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from sportstradamus.helpers.training_quotes import AUTHENTIC, PRICED_AUTHENTICITY
from sportstradamus.scripts.priced_score import (
    DEFAULT_SEED,
    MIN_DATES,
    p1_rows,
    p2_rungs,
    row_changes,
    scope_table,
)
from sportstradamus.scripts.realized_replay import still_posted
from sportstradamus.training.scorecard import _zero_inflated_mean

# PR-1.5's prior spread of a recipe's true effect: P1 dMSE 1.0 % of the incumbent's MSE (the lane
# measured 1.07 %), P2 dBrier 10 x1e-4 (the forward slope .77 implies 11-16; 10 leans
# conservative). A statistic without one is reported, never shrunk.
TAU0 = {"p1_dmse": 1.0, "p2_dbrier": 10.0}
# PR-1.6: a switch must beat the incumbent by half its own standard error after shrinkage, so a
# tie goes to the incumbent.
MARGIN = 0.5
# PR-1.6: nominees within a quarter of the pick's standard error tie; fewer axis changes win.
TIE_BAND = 0.25
# PR-1.7b: the test half reverts a pick whose primary reads worse than one standard error.
HARM_GUARD = 1.0
# PR-1.7c: a refit whose blend weight moved this far from its nominee's is not the object that
# was ranked, so its test half must read better than the incumbent outright.
W_DRIFT = 0.15
# PR-1.4: a served mean past twice the matrix's largest outcome is a blown-up fit (NHL goals
# served 35.6 against a matrix maximum of 4), not a recipe.
MEAN_SANITY = 2.0
# The selection half cannot be re-served at a rung (module docstring).
SELECTION_STATISTIC = "p1_dmse"
# PR-1.3: each priced tier's primary statistic on the test half; tier P2's guard is p1_dmse.
PRIMARY = {"P2": "p2_dbrier", "P1": "p1_dmse"}
# The board columns a priced cell's corners carry (and confirm's nominees, read on their blind
# runs): the tier, the selection statistic, its estimate, its inflated and date-only standard
# errors, the shrunk estimate that ranks, its game dates, and the mean-sanity ratio.
COLUMNS = (
    "priced_tier",
    "priced_stat",
    "priced_est",
    "priced_se",
    "priced_se_date",
    "priced_shrunk",
    "priced_dates",
    "priced_mean_ratio",
)
# What :func:`pick` reads of each nominee, beside the label that indexes it.
NOMINEE_COLUMNS = ("eligible", "shrunk", "s", "axes", "slack")


class Baseline(NamedTuple):
    """What a priced cell's recipes are measured against on the selection half."""

    tier: str
    # The incumbent's served selection-half rows (:func:`served`); ``None`` prices against the
    # book on authentic rows, for a withheld cell.
    rows: pd.DataFrame | None
    # The largest outcome in the cell's matrix: the mean-sanity scan's yardstick.
    max_result: float


def served(dump: pd.DataFrame) -> pd.DataFrame:
    """A dump's rows as training served them: ``p`` is its ``P``, ``mean`` the gated ``Blended_EV``.

    The training convention: an authentic row blended at the fitted weight, every other row at 1.
    """
    mean = _zero_inflated_mean(dump, dump["Blended_EV"].to_numpy(float))
    return dump.assign(p=dump["P"], mean=mean)


def tier(rows: pd.DataFrame, classes: tuple[str, ...], n_p2_dates: int) -> str:
    """PR-1.3's tier from the selection half's counts: ``P2``, ``P1`` or ``U``.

    ``classes`` are the quote classes a P1 row may carry (a withheld cell counts authentic rows
    only); ``n_p2_dates`` is :func:`p2_dates` of the half's rungs. P2 is checked first, as E0 did.
    """
    if n_p2_dates >= MIN_DATES:
        return "P2"
    priced = rows["QuoteAuthenticity"].isin(classes) & rows["Line"].notna()
    return "P1" if rows.loc[priced, "Date"].nunique() >= MIN_DATES else "U"


def p2_dates(rungs: pd.DataFrame) -> int:
    """Game dates with a settled rung still posted at its prop's last poll (PR-1.1's P2)."""
    posted = rungs[still_posted(rungs)]
    return posted.loc[posted["Result"] != posted["Line"], "Date"].nunique()


def contrast(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame | None,
    cell: str,
    p2: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One cell's priced contrast, candidate minus baseline, per ``ship priced`` statistic.

    Args:
        candidate: The candidate's rows with ``p`` and ``mean``.
        baseline: The incumbent's rows likewise, or ``None`` for the book on authentic rows
            (P(over) ``1 - Odds``, mean ``Book_EV``), as ``ship priced --baseline-book``.
        cell: The ``{LEAGUE}_{market}`` dump stem, which keys the bootstrap streams as
            ``ship priced`` keys them.
        p2: Both arms' P2 rungs (``priced_score.p2_rungs``); none on the selection half.

    Returns:
        ``scope_table``'s rows for the cell indexed by statistic, plus ``s``, the date-bootstrap
        SE times sqrt(``deff_player``) (the design effect is at least 1, its ICC floored at 0),
        and ``shrunk``, the estimate times tau0^2 / (tau0^2 + s^2), NaN without a prior. Empty
        when the arms share no priced row or rung.
    """
    classes = PRICED_AUTHENTICITY
    if baseline is None:
        baseline = candidate.assign(p=1 - candidate["Odds"], mean=candidate["Book_EV"])
        classes = (AUTHENTIC,)
    p1, _ = p1_rows(candidate, baseline, classes)
    p2 = p2_rungs(pd.DataFrame(), pd.DataFrame()) if p2 is None else p2
    if p1.empty and p2.empty:
        return pd.DataFrame()
    p1 = p1.assign(League=cell.split("_", 1)[0], cell=cell)
    table = scope_table(row_changes(p1, p2), DEFAULT_SEED)
    table = table[table["cell"].eq(cell)].set_index("statistic")
    s = table["se"] * np.sqrt(table["deff_player"])
    prior = pd.Series(TAU0).reindex(table.index.astype(str)).to_numpy()
    return table.assign(s=s, shrunk=prior**2 / (prior**2 + s**2) * table["estimate"])


def selection_columns(dump: pd.DataFrame, baseline: Baseline, cell: str) -> dict[str, object]:
    """A candidate's :data:`COLUMNS`: its selection-half contrast with ``baseline``.

    ``dump`` is the candidate's blind dump as ``scorecard.load_test_set`` reads it. The estimate
    columns are NaN when the two share no priced row.
    """
    rows = served(dump)
    read = contrast(rows, baseline.rows, cell)
    lead = read.loc[SELECTION_STATISTIC] if SELECTION_STATISTIC in read.index else pd.Series()
    return {
        "priced_tier": baseline.tier,
        "priced_stat": SELECTION_STATISTIC,
        "priced_est": lead.get("estimate", np.nan),
        "priced_se": lead.get("s", np.nan),
        "priced_se_date": lead.get("se", np.nan),
        "priced_shrunk": lead.get("shrunk", np.nan),
        "priced_dates": lead.get("n_dates", np.nan),
        "priced_mean_ratio": float(rows["mean"].max()) / baseline.max_result,
    }


def pick(nominees: pd.DataFrame) -> tuple[str | None, str]:
    """PR-1.6's switch among a cell's blind nominees, or the incumbent.

    ``nominees`` is indexed by label in nomination order, with :data:`NOMINEE_COLUMNS`:
    ``eligible`` (PR-1.4), ``shrunk``, ``s``, ``axes`` (axis changes from the incumbent's recipe)
    and ``slack``. The best eligible
    shrunk contrast must clear ``MARGIN`` of its own standard error, else the incumbent stays.
    Eligible nominees that clear their own margin and lie within ``TIE_BAND`` of the best's
    standard error tie; fewer axis changes win, then the higher slack, then nomination order.

    Returns:
        The pick's label, ``None`` when the incumbent stays, and why.
    """
    eligible = nominees[nominees["eligible"] & nominees["shrunk"].notna()]
    if eligible.empty:
        return None, "no eligible nominee"
    best = eligible.loc[eligible["shrunk"].idxmin()]
    if not best["shrunk"] < -MARGIN * best["s"]:
        return None, (
            f"{best.name} short of the margin: shrunk {best['shrunk']:+.3f} vs "
            f"{-MARGIN * best['s']:+.3f}"
        )
    clears = eligible["shrunk"] < -MARGIN * eligible["s"]
    ties = eligible[clears & (eligible["shrunk"] <= best["shrunk"] + TIE_BAND * best["s"])]
    chosen = ties.sort_values(["axes", "slack"], ascending=[True, False], kind="stable").index[0]
    return chosen, f"switch to {chosen} (tie set: {', '.join(map(str, ties.index))})"


def harm(table: pd.DataFrame, cell_tier: str, w_nominee: float, w_refit: float) -> list[str]:
    """PR-1.7's guard on a pick's canonical refit: why it reverts, empty to keep it.

    ``table`` is the refit's :func:`contrast` with the incumbent on the test half. The pick
    reverts when its primary reads above ``HARM_GUARD`` standard errors, or at or above zero once
    the refit's blend weight drifted past ``W_DRIFT`` from its nominee's, or when the primary's or
    tier P2's guard's interval lies wholly above zero over ``MIN_DATES`` dates (``ship priced``'s
    KILL leg on those two).
    """
    primary = PRIMARY[cell_tier]
    if primary not in table.index:
        return [f"no {primary} on the test half"]
    lead = table.loc[primary]
    failed = []
    if abs(w_refit - w_nominee) > W_DRIFT:
        if not lead["estimate"] < 0:
            failed.append(
                f"{primary} {lead['estimate']:+.3f} not below 0 after w drift "
                f"{w_nominee:.3f}->{w_refit:.3f}"
            )
    elif lead["estimate"] > HARM_GUARD * lead["s"]:
        failed.append(f"{primary} {lead['estimate']:+.3f} above +{HARM_GUARD:g}s ({lead['s']:.3f})")
    for statistic in dict.fromkeys((primary, "p1_dmse")):
        if statistic not in table.index:
            continue
        read = table.loc[statistic]
        if read["n_dates"] >= MIN_DATES and read["lo"] > 0:
            failed.append(f"{statistic} interval above 0 [{read['lo']:+.3f}, {read['hi']:+.3f}]")
    return failed
