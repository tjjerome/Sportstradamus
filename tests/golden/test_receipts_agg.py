"""Pure shaping behind the Receipts hero and its by-side grid.

``window_offers`` cuts the page window where the nightly ledger cuts its own;
``cohort_figures`` formats one ``realized.cohort_summary`` for the hero and its context row;
``by_side_grid`` adds ``Breakeven%`` and ``Edge captured`` to a ``realized.by_split``
result. All over the reconciliation fixture's history. No Streamlit, no I/O.
"""

from __future__ import annotations

import pytest

from sportstradamus.dashboard.components.by_side import by_side_grid
from sportstradamus.dashboard.components.receipts_hero import cohort_figures, window_offers
from sportstradamus.realized import by_split, cohort_summary, settled_offers, window
from tests.golden.test_receipts_reconciles_ledger import HISTORY, NOW


def test_the_all_window_keeps_every_offer():
    offers = settled_offers(HISTORY)
    assert window_offers(offers, "All", NOW).equals(offers)


@pytest.mark.parametrize(("label", "days"), [("7d", 7), ("30d", 30), ("3m", 91), ("1y", 365)])
def test_a_dated_window_cuts_where_the_ledger_cuts(label, days):
    offers = settled_offers(HISTORY)
    assert window_offers(offers, label, NOW).equals(window(offers, days, NOW))


def test_cohort_figures_carry_explicit_signs():
    gain = cohort_figures(
        {
            "n": 1234,
            "hit_rate": 0.496,
            "pred_rate": 0.65,
            "book_rate": 0.56,
            "payout": 1.83,
            "breakeven_rate": 0.5464,
            "units": 38.4,
            "roi": 0.0311,
        }
    )
    assert gain == {
        "n": "1,234",
        "roi": "+3.1%",
        "hit_rate": "49.6%",
        "pred_rate": "65.0%",
        "breakeven_rate": "54.6%",
        "payout": "1.83x",
        "record": "612–622",
        "units": "+38",
    }
    # A losing cohort reads "-1,296", never the "+-1296" a hard-coded plus produced.
    loss = cohort_figures(
        {
            "n": 12345,
            "hit_rate": 0.5,
            "pred_rate": 0.6,
            "book_rate": 0.55,
            "payout": 1.8,
            "breakeven_rate": 0.5556,
            "units": -1296.2,
            "roi": -0.105,
        }
    )
    assert (loss["units"], loss["roi"]) == ("-1,296", "-10.5%")


def test_cohort_figures_of_the_fixtures_recommended_legs():
    offers = window(settled_offers(HISTORY), 90, NOW)
    figures = cohort_figures(cohort_summary(offers[offers["Recommended"]]))
    # B and I hit; both A offers, C, D and J miss: 0.75 + 0.5 - 5 = -3.75 units over 7.
    assert (figures["n"], figures["record"], figures["units"], figures["roi"]) == (
        "7",
        "2–5",
        "-4",
        "-53.6%",
    )


def test_cohort_figures_of_an_empty_cohort():
    figures = cohort_figures(cohort_summary(settled_offers(HISTORY.iloc[:0])))
    assert figures == {
        "n": "0",
        "record": "0–0",
        "units": "+0",
        "roi": "—",
        "hit_rate": "—",
        "pred_rate": "—",
        "breakeven_rate": "—",
        "payout": "—",
    }


def test_breakeven_is_the_hit_rate_the_payouts_need_in_points():
    grid = by_side_grid(by_split(settled_offers(HISTORY)), "platform")
    row = grid[(grid["Platform"] == "Sleeper") & (grid["Side"] == "Under")].iloc[0]
    # B at 1.75 and F at 1.875: two legs need 2 / 3.625 of a hit each to break even.
    assert row["Breakeven%"] == pytest.approx(100 * 2 / (1.75 + 1.875))


def test_edge_captured_is_the_realized_share_of_the_claimed_edge():
    grid = by_side_grid(by_split(settled_offers(HISTORY)), "cell")
    row = grid[(grid["Cell"] == "NBA/REB") & (grid["Side"] == "Under")].iloc[0]
    # B alone hit on a 0.64 read over a 0.52 book: (1 - 0.52) / (0.64 - 0.52).
    assert row["Edge captured"] == pytest.approx(4.0)


def test_edge_captured_blanks_a_claim_under_the_floor():
    thin = HISTORY.assign(**{"Market Prob": HISTORY["Win Prob"] - 0.015})
    grid = by_side_grid(by_split(settled_offers(thin)), "cell")
    assert grid["Edge captured"].isna().all()
    assert grid["Hit%"].notna().all()
