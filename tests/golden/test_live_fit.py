"""The EV fit: the weights it recovers, what its prior holds, and the price it gives."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from sportstradamus import live_fit
from sportstradamus.helpers.io import read_parquet_safe


def _settled(
    rng,
    n,
    *,
    league="NHL",
    platform="Underdog",
    even=True,
    book_class="A",
    fallback_share=0.0,
    c=0.0,
    a1=0.0,
    w_book=0.0,
    w_model=0.0,
    w_fb=0.0,
    dates=40,
):
    """Settled offers whose Over hits with the chance the given coefficients imply."""
    if even:
        payout_over = payout_under = np.full(n, 1.83)
    else:
        payout_over = rng.uniform(1.35, 2.6, n)
        payout_under = 1 / (1.08 - 1 / payout_over)
    fair = (1 / payout_over) / (1 / payout_over + 1 / payout_under)
    x0 = logit(fair)
    model_over = expit(x0 + rng.normal(0, 0.35, n))
    book_over = expit(x0 + rng.normal(0, 0.15, n)) if book_class == "A" else np.full(n, np.nan)
    fallback = rng.random(n) < fallback_share
    d_model = logit(model_over) - x0
    eta = x0 + c + a1 * x0 + np.where(fallback, w_fb, w_model) * d_model
    if book_class == "A":
        eta = eta + w_book * (logit(book_over) - x0)
    over_hit = rng.random(n) < expit(eta)
    over = model_over >= 0.5
    return pd.DataFrame(
        {
            "League": league,
            "Platform": platform,
            "Date": (
                pd.Timestamp("2026-10-09") + pd.to_timedelta(rng.integers(0, dates, n), unit="D")
            ).strftime("%Y-%m-%d"),
            "Payout Over": payout_over,
            "Payout Under": payout_under,
            "Bet": np.where(over, "Over", "Under"),
            "Win Prob": np.where(over, model_over, 1 - model_over),
            "Model Version": np.where(fallback, "book_fallback", "v1"),
            "Book Prob": np.where(over, book_over, 1 - book_over),
            "Book Class": book_class,
            "Result": np.where(over_hit, "Over", "Under"),
        }
    )


def _stratum(table, league="NHL", kind="underdog_even", book_class="A"):
    row = table[
        (table["league"] == league) & (table["kind"] == kind) & (table["book_class"] == book_class)
    ]
    assert len(row) == 1
    return row.iloc[0]


def test_fit_recovers_the_weights_that_made_the_outcomes():
    rng = np.random.default_rng(0)
    offers = _settled(rng, 60_000, even=False, c=0.10, a1=0.25, w_book=0.8, w_model=0.3)

    row = _stratum(live_fit.fit(offers, prior_legs=1), kind="underdog_modified")

    for term, truth in (("c", 0.10), ("a1", 0.25), ("w_book", 0.8), ("w_model", 0.3)):
        assert abs(row[term] - truth) < 4 * row[f"se_{term}"], term
        assert row[f"se_{term}"] < 0.08, term
    assert row["n_legs"] == 60_000
    assert row["n_dates"] == 40
    assert row["first_date"] == "2026-10-09"
    assert row["last_date"] == "2026-11-17"


def test_fit_gives_withheld_cells_their_own_model_weight():
    rng = np.random.default_rng(1)
    offers = _settled(rng, 60_000, book_class="C", fallback_share=0.4, w_model=0.6, w_fb=-0.5)

    row = _stratum(live_fit.fit(offers, prior_legs=1), book_class="C")

    assert row["w_model"] == pytest.approx(0.6, abs=0.1)
    assert row["w_fb"] == pytest.approx(-0.5, abs=0.1)


def test_fit_leaves_out_a_term_its_rows_cannot_show():
    rng = np.random.default_rng(2)
    offers = _settled(rng, 2_000, book_class="C", w_model=0.5)

    row = _stratum(live_fit.fit(offers), book_class="C")

    # An even pick's fair price is one half on every row, no book priced the line and no cell
    # was withheld: the slope, the book weight and the withheld weight have nothing to read.
    absent = ["a1", "w_book", "w_fb", "se_a1", "se_w_book", "se_w_fb"]
    assert row[absent].astype(float).isna().all()
    assert np.isfinite(row[["c", "w_model", "se_c", "se_w_model"]].astype(float)).all()
    assert np.array(row["cov"]).shape == (4,)


def test_fit_skips_a_stratum_under_the_floor():
    rng = np.random.default_rng(3)
    offers = pd.concat(
        [
            _settled(rng, live_fit.MIN_FIT_LEGS, league="NHL"),
            _settled(rng, live_fit.MIN_FIT_LEGS - 1, league="NBA"),
        ]
    )

    table = live_fit.fit(offers)

    assert table["league"].tolist() == ["NHL"]
    assert table.columns.tolist() == live_fit.FIT_COLS


def test_fit_reads_only_offers_with_both_sides_posted():
    rng = np.random.default_rng(4)
    offers = _settled(rng, 1_000)
    offers.loc[offers.index[:300], "Payout Under"] = 0.0

    assert _stratum(live_fit.fit(offers))["n_legs"] == 700


def test_the_prior_holds_a_small_sample_at_the_apps_price():
    rng = np.random.default_rng(5)
    offers = _settled(rng, 500, book_class="C", w_model=1.0)

    loose = _stratum(live_fit.fit(offers, prior_legs=1), book_class="C")
    held = _stratum(live_fit.fit(offers), book_class="C")

    assert loose["w_model"] > 0.5
    assert abs(held["w_model"]) < 0.25 * loose["w_model"]


def test_the_prior_holds_a_term_by_the_legs_that_carry_it():
    rng = np.random.default_rng(10)
    offers = _settled(rng, 12_000, book_class="C", fallback_share=0.04, w_fb=1.0)

    row = _stratum(live_fit.fit(offers), book_class="C")

    # About 480 withheld legs carry the term, against a prior worth 4,000 of them, so the
    # weight stays near a tenth of the truth. Scaled over all 12,000 legs the prior would let
    # those 480 move it to about three quarters.
    assert (offers["Model Version"] == "book_fallback").sum() < 600
    assert 0 < row["w_fb"] < 0.35


def test_score_with_no_fit_is_the_apps_fair_price():
    offers = pd.DataFrame(
        {
            "League": "NHL",
            "Platform": ["Underdog", "Underdog", "Sleeper"],
            "Payout Over": [1.83, 1.50, 2.00],
            "Payout Under": [1.83, 2.40, 1.60],
            "Bet": ["Over", "Under", "Over"],
            "Win Prob": [0.62, 0.55, 0.58],
            "Model Version": "v1",
            "Book Prob": np.nan,
            "Book Class": "C",
        }
    )

    scored = live_fit.score(offers, pd.DataFrame(columns=live_fit.FIT_COLS))

    fair_over = np.array([0.5, (1 / 1.5) / (1 / 1.5 + 1 / 2.4), (1 / 2.0) / (1 / 2.0 + 1 / 1.6)])
    fair = np.array([fair_over[0], 1 - fair_over[1], fair_over[2]])
    np.testing.assert_allclose(scored["Fit Prob"], fair)
    np.testing.assert_allclose(scored["Fit Prob Low"], fair)
    np.testing.assert_allclose(scored["EV"], fair * np.array([1.83, 2.40, 2.00]))
    assert (scored["EV"] < 1).all()
    assert scored.index.equals(offers.index)


def test_score_prices_the_side_the_model_took():
    rng = np.random.default_rng(6)
    offers = _settled(rng, 30_000, c=0.0, w_book=0.9, w_model=0.2)
    table = live_fit.fit(offers, prior_legs=1)
    row = _stratum(table)
    leg = offers.iloc[[0, 1]].assign(
        Bet=["Over", "Under"], **{"Win Prob": [0.60, 0.60], "Book Prob": [0.56, 0.56]}
    )

    scored = live_fit.score(leg, table)

    eta_over = row["c"] + row["w_book"] * logit(0.56) + row["w_model"] * logit(0.60)
    eta_under = row["c"] + row["w_book"] * logit(0.44) + row["w_model"] * logit(0.40)
    np.testing.assert_allclose(scored["Fit Prob"], [expit(eta_over), expit(-eta_under)])
    np.testing.assert_allclose(scored["EV"], scored["Fit Prob"] * 1.83)
    assert (scored["Fit Prob Low"] < scored["Fit Prob"]).all()
    # The bound is the same distance below the point on the logit scale for either side.
    drop = logit(scored["Fit Prob"]) - logit(scored["Fit Prob Low"])
    assert drop.iat[0] > 0
    cov = np.array(row["cov"]).reshape(3, 3)
    x = np.array([1.0, logit(0.56), logit(0.60)])
    assert drop.iat[0] == pytest.approx(live_fit.LOWER_BOUND_Z * np.sqrt(x @ cov @ x))


def test_score_gives_no_price_to_an_offer_with_one_side_posted():
    rng = np.random.default_rng(7)
    offers = _settled(rng, 1_000)
    table = live_fit.fit(offers)
    one_sided = offers.iloc[[0]].assign(**{"Payout Under": 0.0})

    scored = live_fit.score(one_sided, table)

    assert scored[["Fit Prob", "Fit Prob Low", "EV"]].isna().all(axis=None)


def test_score_reads_the_table_as_parquet_stores_it(tmp_path):
    rng = np.random.default_rng(9)
    offers = _settled(rng, 3_000, w_book=0.8, w_model=0.3)
    table = live_fit.fit(offers)
    table.to_parquet(tmp_path / "live_fit.parquet")

    stored = pd.read_parquet(tmp_path / "live_fit.parquet")

    assert _stratum(table)["w_book"] != 0
    pd.testing.assert_frame_equal(live_fit.score(offers, stored), live_fit.score(offers, table))


def test_score_before_the_first_table_is_the_apps_fair_price(tmp_path):
    rng = np.random.default_rng(10)
    offers = _settled(rng, 50)

    scored = live_fit.score(offers, read_parquet_safe(tmp_path / "live_fit.parquet"))

    np.testing.assert_allclose(scored["Fit Prob"], 0.5)
    np.testing.assert_allclose(scored["Fit Prob Low"], 0.5)


def test_walk_forward_scores_a_date_from_earlier_dates_only():
    rng = np.random.default_rng(8)
    offers = _settled(rng, 8_000, book_class="C", w_model=0.8, dates=20).sort_values("Date")
    dates = np.sort(offers["Date"].unique())
    flipped = offers.copy()
    last = flipped["Date"] == dates[-1]
    flipped.loc[last, "Result"] = np.where(flipped.loc[last, "Result"] == "Over", "Under", "Over")

    walked = live_fit.walk_forward(offers)
    walked_flipped = live_fit.walk_forward(flipped)

    assert sorted(walked["Date"].unique()) == list(dates[live_fit.MIN_TRAIN_DATES :])
    pd.testing.assert_frame_equal(
        walked[["Fit Prob", "Fit Prob Low", "EV"]],
        walked_flipped[["Fit Prob", "Fit Prob Low", "EV"]],
    )
    # A fit that had read the date it scores would lean every leg toward its own outcome.
    first = walked[walked["Date"] == dates[live_fit.MIN_TRAIN_DATES]]
    alone = live_fit.score(
        first, live_fit.fit(offers[offers["Date"] < dates[live_fit.MIN_TRAIN_DATES]])
    )
    np.testing.assert_allclose(first["Fit Prob"], alone["Fit Prob"])


def test_walk_forward_trains_on_the_trailing_window_only():
    rng = np.random.default_rng(9)
    recent = _settled(rng, 6_000, book_class="C", w_model=0.0, dates=20)
    recent["Date"] = (
        pd.to_datetime(recent["Date"]) + pd.Timedelta(days=live_fit.FIT_WINDOW_DAYS + 30)
    ).dt.strftime("%Y-%m-%d")
    old = _settled(rng, 60_000, book_class="C", w_model=1.5, dates=14)

    walked = live_fit.walk_forward(pd.concat([old, recent], ignore_index=True))

    last_date = walked["Date"].max()
    last = walked[walked["Date"] == last_date]
    in_window = recent[recent["Date"] < last_date]
    expected = live_fit.score(last, live_fit.fit(in_window))
    np.testing.assert_allclose(last["Fit Prob"], expected["Fit Prob"])


def _priced(**fields) -> dict:
    """One settled offer the fit priced: an Underdog even pick the model took Over, and won."""
    return {
        "League": "NHL",
        "Platform": "Underdog",
        "Date": "2026-11-01",
        "Payout Over": 1.83,
        "Payout Under": 1.83,
        "Bet": "Over",
        "Win Prob": 0.6,
        "Model Version": "v1",
        "Book Prob": np.nan,
        "Book Class": "C",
        "Result": "Over",
        "Fit Prob": 0.56,
        "Fit Prob Low": 0.53,
        "EV": 0.56 * 1.83,
    } | fields


def test_fit_gain_is_the_apps_log_loss_less_the_fits_on_the_overs_outcome():
    modified = {"Payout Over": 1.5, "Payout Under": 2.4}
    fair_over = (1 / 1.5) / (1 / 1.5 + 1 / 2.4)
    offers = pd.DataFrame(
        [
            _priced(),  # the Over hit: the fit said .56, the app one half
            _priced(Bet="Under", Date="2026-11-02"),  # the fit said .56 Under, so .44 Over
            _priced(Result="Under", Date="2026-11-02"),
            _priced(**modified, **{"Fit Prob": 0.65}),
            # An Under that hit at a modified price: the fit said .56, the app its fair Under.
            _priced(League="NBA", Platform="Sleeper", Bet="Under", Result="Under", **modified),
        ]
    )

    table = live_fit.fit_gain(offers, np.random.default_rng(0))

    assert table.columns.tolist() == live_fit.GAIN_COLS
    gain = table.set_index(["league", "kind"])
    even = [np.log(0.56 / 0.5), np.log(0.44 / 0.5), np.log(0.44 / 0.5)]
    modified_gain = np.log(0.65 / fair_over)
    sleeper_gain = np.log(0.56 / (1 - fair_over))
    assert gain.at[("NHL", "underdog_even"), "gain"] == pytest.approx(np.mean(even))
    assert gain.at[("NHL", "underdog_modified"), "gain"] == pytest.approx(modified_gain)
    assert gain.at[("NBA", "sleeper"), "gain"] == pytest.approx(sleeper_gain)
    assert gain.at[("all", "all"), "gain"] == pytest.approx(
        np.mean([*even, modified_gain, sleeper_gain])
    )
    assert gain["n_legs"].to_dict() == {
        ("NBA", "sleeper"): 1,
        ("NHL", "underdog_even"): 3,
        ("NHL", "underdog_modified"): 1,
        ("all", "all"): 5,
    }
    assert gain.at[("NHL", "underdog_even"), "n_dates"] == 2
    assert gain.at[("all", "all"), "n_dates"] == 2


def test_fit_gain_leaves_out_an_offer_the_fit_did_not_price():
    unpriced = {"Fit Prob": np.nan, "Fit Prob Low": np.nan, "EV": np.nan}
    offers = pd.DataFrame(
        [
            _priced(),
            _priced(Date="2026-11-02", **unpriced),  # scored before the fit
            _priced(Date="2026-11-03", **unpriced, **{"Payout Under": 0.0}),  # one side posted
        ]
    )

    (row,) = live_fit.fit_gain(offers, np.random.default_rng(0)).to_dict("records")[-1:]

    assert (row["league"], row["n_legs"], row["n_dates"]) == ("all", 1, 1)
    assert row["gain"] == pytest.approx(np.log(0.56 / 0.5))


def test_fit_gain_bounds_resample_whole_game_dates():
    rng = np.random.default_rng(11)
    chances = rng.uniform(0.4, 0.6, 60)
    one_date = pd.DataFrame([_priced(**{"Fit Prob": chance}) for chance in chances])
    spread = one_date.assign(Date=[f"2026-11-{1 + leg % 20:02d}" for leg in range(60)])

    alone = live_fit.fit_gain(one_date, np.random.default_rng(0)).iloc[-1]
    across = live_fit.fit_gain(spread, np.random.default_rng(0)).iloc[-1]

    # One date is one cluster, so every resample is the sample; legs drawn one by one would vary.
    assert alone["gain_lo"] == alone["gain_hi"] == pytest.approx(alone["gain"])
    assert across["gain"] == pytest.approx(alone["gain"])
    assert across["gain_lo"] < across["gain"] < across["gain_hi"]


def test_fit_gain_credits_a_walk_forward_whose_model_knows_more_than_the_app():
    rng = np.random.default_rng(12)
    offers = _settled(rng, 20_000, book_class="C", w_model=0.8, dates=30)
    walked = live_fit.walk_forward(offers)

    (row,) = live_fit.fit_gain(walked, np.random.default_rng(0)).to_dict("records")[-1:]

    assert (row["n_legs"], row["n_dates"]) == (len(walked), 30 - live_fit.MIN_TRAIN_DATES)
    assert row["gain_lo"] > 0


def test_ev_bands_set_what_the_fit_said_beside_what_the_models_side_paid():
    under_at_2_4 = {"Bet": "Under", "Payout Over": 1.5, "Payout Under": 2.4}
    offers = pd.DataFrame(
        [
            _priced(EV=0.85),
            _priced(EV=0.90, Result="Under"),  # a band holds its lower edge
            _priced(EV=0.92, Result="Under", **under_at_2_4),  # the Under hit, at its own payout
            _priced(EV=0.97),
            _priced(EV=1.00),
            _priced(EV=1.30, Result="Under"),
            _priced(EV=np.nan),  # the fit gave it no price
        ]
    )

    table = live_fit.ev_bands(offers, np.random.default_rng(0))

    assert table.columns.tolist() == live_fit.BAND_COLS
    assert table["band"].tolist() == [
        "[0.0, 0.9)",
        "[0.9, 0.94)",
        "[0.94, 0.97)",
        "[0.97, 1.0)",
        "[1.0, inf)",
    ]
    assert table["n_legs"].tolist() == [1, 2, 0, 1, 2]
    np.testing.assert_allclose(table["says"], [0.85, 0.91, np.nan, 0.97, 1.15])
    np.testing.assert_allclose(table["paid"], [1.83, 1.2, np.nan, 1.83, 0.915])
    assert table.loc[2, ["paid_lo", "paid_hi"]].isna().all()


def test_ev_bands_bounds_resample_whole_game_dates():
    rng = np.random.default_rng(13)
    results = rng.choice(["Over", "Under"], 60)
    one_date = pd.DataFrame([_priced(EV=0.95, Result=result) for result in results])
    spread = one_date.assign(Date=[f"2026-11-{1 + leg % 20:02d}" for leg in range(60)])

    alone = live_fit.ev_bands(one_date, np.random.default_rng(0)).set_index("band")
    across = live_fit.ev_bands(spread, np.random.default_rng(0)).set_index("band")

    band = "[0.94, 0.97)"
    assert alone.at[band, "paid_lo"] == alone.at[band, "paid_hi"] == alone.at[band, "paid"]
    assert across.at[band, "paid_lo"] < across.at[band, "paid"] < across.at[band, "paid_hi"]
