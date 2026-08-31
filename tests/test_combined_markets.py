"""Oracle and convention tests for the scrambled-Sobol NORTA combo kernel.

The retired analytic helpers return here as ground truths: the closed-form
bivariate-Normal sum checks the continuous path and a brute-force PMF
convolution checks the count path at rho = 0, per the research brief's
"keep the convolution and the closed-form Normal as test oracles" verdict.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, norm, poisson

from sportstradamus.helpers import combined_markets as cm
from sportstradamus.helpers.combined_markets import ComboComponent, combo_sum_quote
from sportstradamus.helpers.distributions import get_odds, get_push_prob

# Brief-measured max Sobol tail error at n = 8192 is 0.0053; 0.006 bounds it.
QMC_TOL = 0.006


def rho_zero(_a, _b):
    return 0.0


def make_rho(value):
    return lambda _a, _b: value


def test_bivariate_normal_oracle():
    w1, w2, rho = 1.0, 1.0, 0.4
    c1 = ComboComponent("pass yds", w1, 250.0, "Normal", 0.28, sigma=70.0)
    c2 = ComboComponent("rush yds", w2, 30.0, "Normal", 0.6, sigma=20.0)
    quote = combo_sum_quote([c1, c2], make_rho(rho))
    mu = w1 * 250.0 + w2 * 30.0
    sd = np.sqrt(w1**2 * 70.0**2 + w2**2 * 20.0**2 + 2 * w1 * w2 * rho * 70.0 * 20.0)
    assert quote.mean == pytest.approx(mu)
    assert quote.sd == pytest.approx(sd, rel=0.02)
    for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        line = norm.ppf(q, mu, sd)
        assert abs(quote.under_prob(line) - q) < QMC_TOL


def test_bivariate_normal_oracle_fractional_weights():
    w1, w2, rho = 1.2, 0.5, -0.3
    c1 = ComboComponent("a", w1, 20.0, "Normal", 0.3, sigma=6.0)
    c2 = ComboComponent("b", w2, 8.0, "Normal", 0.5, sigma=4.0)
    quote = combo_sum_quote([c1, c2], make_rho(rho))
    mu = w1 * 20.0 + w2 * 8.0
    sd = np.sqrt(w1**2 * 36.0 + w2**2 * 16.0 + 2 * w1 * w2 * rho * 6.0 * 4.0)
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        line = norm.ppf(q, mu, sd)
        assert abs(quote.under_prob(line) - q) < QMC_TOL


def test_count_convolution_oracle():
    """Independent integer-weight count sum vs brute-force PMF enumeration."""
    r2 = 1 / 0.5
    p2 = r2 / (r2 + 2.4)
    gate3, mean3 = 0.25, 1.1
    r3 = 1 / 0.4
    p3 = r3 / (r3 + mean3)
    comps = [
        ComboComponent("pois", 1.0, 1.3, "Poisson", 1.0),
        ComboComponent("nb", 2.0, 2.4, "NegBin", 0.5),
        ComboComponent("zinb", 1.0, mean3, "ZINB", 0.4, gate=gate3),
    ]
    quote = combo_sum_quote(comps, rho_zero)

    ks = np.arange(81)
    pmf1 = poisson.pmf(ks, 1.3)
    pmf2 = nbinom.pmf(ks, r2, p2)
    pmf3 = (1 - gate3) * nbinom.pmf(ks, r3, p3)
    pmf3[0] += gate3
    probs = pmf1[:, None, None] * pmf2[None, :, None] * pmf3[None, None, :]
    vals = ks[:, None, None] + 2 * ks[None, :, None] + ks[None, None, :]
    for line in [0.5, 1.5, 2.5, 3.0, 4.5, 6.0, 7.5, 10.5, 13.0]:
        oracle = probs[vals < line].sum() + probs[vals == line].sum() / 2
        assert abs(quote.under_prob(line) - oracle) < QMC_TOL
    marginal_mean = 1.3 + 2 * 2.4 + (1 - gate3) * mean3
    assert quote.mean == pytest.approx(marginal_mean)


def test_determinism():
    comps = [
        ComboComponent("a", 1.0, 12.0, "SkewNormal", 0.4, sigma=5.0, skew=1.5),
        ComboComponent("b", -1.0, 3.1, "NegBin", 0.45),
        ComboComponent("c", 2.0, 1.4, "DPO", 0.55),
    ]
    q1 = combo_sum_quote(comps, make_rho(0.2))
    q2 = combo_sum_quote(comps, make_rho(0.2))
    assert np.array_equal(q1.draws_sorted, q2.draws_sorted)
    assert q1.mean == q2.mean and q1.sd == q2.sd
    for line in [2.5, 8.0, 11.5, 15.0]:
        assert q1.under_prob(line) == q2.under_prob(line)


def test_monotone_alt_lines():
    comps = [
        ComboComponent("cont", 1.0, 10.0, "Gamma", 0.35),
        ComboComponent("count", 3.0, 0.8, "ZINB", 0.5, gate=0.3),
        ComboComponent("dpo", -1.0, 2.5, "DPO", 0.55),
    ]
    quote = combo_sum_quote(comps, make_rho(0.15))
    grid = np.arange(-4.0, 25.0, 0.25)
    probs = [quote.under_prob(x) for x in grid]
    assert all(b >= a for a, b in zip(probs, probs[1:]))


def test_achieved_correlation():
    s1, s2 = 70.0, 20.0
    comps = [
        ComboComponent("x", 1.0, 250.0, "Normal", 0.28, sigma=s1),
        ComboComponent("y", 1.0, 30.0, "Normal", 0.6, sigma=s2),
    ]
    quote = combo_sum_quote(comps, make_rho(0.4))
    rho_hat = (quote.sd**2 - s1**2 - s2**2) / (2 * s1 * s2)
    assert abs(rho_hat - 0.4) < 0.02


def test_negative_weight_component():
    comps = [
        ComboComponent("x", 1.0, 250.0, "Normal", 0.28, sigma=70.0),
        ComboComponent("y", -1.0, 30.0, "Normal", 0.6, sigma=20.0),
    ]
    quote = combo_sum_quote(comps, rho_zero)
    assert quote.mean == pytest.approx(220.0)
    assert quote.under_prob(220.0) == pytest.approx(0.5, abs=0.01)


def test_missing_rho_pair_reads_independent():
    table = {("a", "c"): 0.9}  # the (a, b) pair the combo needs is absent

    def rho_lookup(x, y):
        return table.get((x, y), table.get((y, x), 0.0))

    comps = [
        ComboComponent("a", 1.0, 5.0, "Gamma", 0.4),
        ComboComponent("b", 1.0, 2.0, "NegBin", 0.5),
    ]
    q_missing = combo_sum_quote(comps, rho_lookup)
    q_indep = combo_sum_quote(comps, rho_zero)
    assert np.array_equal(q_missing.draws_sorted, q_indep.draws_sorted)


def test_zinb_ppf_matches_distributions_cdf():
    c = ComboComponent("zinb", 1.0, 1.1, "ZINB", 0.4, gate=0.25)
    r = 1 / 0.4
    p = r / (r + 1.1)

    def zi_cdf(k):
        return 0.25 + 0.75 * nbinom.cdf(k, r, p)

    u = np.array([0.01, 0.13, 0.24, 0.2500001, 0.31, 0.5, 0.777, 0.9, 0.99, 0.9999])
    x = cm._component_ppf(u, c)
    assert np.all(zi_cdf(x) >= u)
    positive = x >= 1
    assert np.all(zi_cdf(x[positive] - 1) < u[positive])
    for k in [1, 2, 3, 5]:
        push = get_push_prob(k, 1.1, "ZINB", cv=0.4, gate=0.25)
        assert zi_cdf(k) == pytest.approx(
            get_odds(k, 1.1, "ZINB", cv=0.4, gate=0.25, step=1) + push / 2
        )


def test_dpo_ppf_matches_distributions_cdf():
    c = ComboComponent("dpo", 1.0, 2.49, "DPO", 0.547)
    ks = np.arange(40)
    pmf = np.array([get_push_prob(float(k), 2.49, "DPO", cv=0.547) for k in ks])
    cdf = np.cumsum(pmf)
    assert cdf[-1] == pytest.approx(1.0, abs=1e-9)
    u = np.array([0.02, 0.1, 0.31, 0.5, 0.68, 0.9, 0.99, 0.9995])
    x = cm._component_ppf(u, c).astype(int)
    grid_quantile = np.searchsorted(cdf, u)
    assert np.array_equal(x, grid_quantile)
    assert np.all(cdf[x] >= u - 1e-12)
    for k in [0, 1, 2, 4, 6]:
        assert cdf[k] == pytest.approx(
            get_odds(float(k), 2.49, "DPO", cv=0.547) + pmf[k] / 2, abs=1e-9
        )


def test_post_hook_quality_start_functional():
    comps = [
        ComboComponent("pitching outs", 1.0, 16.0, "Normal", 0.25, sigma=4.0),
        ComboComponent("runs allowed", -3.0, 3.2, "NegBin", 0.45),
    ]
    seen = {}

    def post(draws):
        seen.update(draws)
        return 5.0 * ((draws["pitching outs"] >= 18) & (draws["runs allowed"] <= 3))

    quote = combo_sum_quote(comps, make_rho(-0.35), post=post)
    assert set(seen) == {"pitching outs", "runs allowed"}
    extra = 5.0 * ((seen["pitching outs"] >= 18) & (seen["runs allowed"] <= 3))
    expected = np.sort(seen["pitching outs"] - 3.0 * seen["runs allowed"] + extra)
    assert np.array_equal(quote.draws_sorted, expected)
    assert quote.mean == pytest.approx(16.0 - 3.0 * 3.2 + extra.mean())

    again = combo_sum_quote(comps, make_rho(-0.35), post=post)
    assert np.array_equal(quote.draws_sorted, again.draws_sorted)


def test_bernoulli_component():
    p_win = 0.29
    quote = combo_sum_quote([ComboComponent("win", 6.0, p_win, "Bernoulli", 0.0)], rho_zero)
    assert quote.mean == pytest.approx(6.0 * p_win)
    assert set(np.unique(quote.draws_sorted)) <= {0.0, 6.0}
    assert quote.draws_sorted.mean() == pytest.approx(6.0 * p_win, abs=0.01)
    # under_prob just below 6 is P(loss) = 1 - p (success sits at high copula u)
    assert quote.under_prob(5.9) == pytest.approx(1 - p_win, abs=0.01)


def test_weight_zero_component_contributes_only_through_post():
    comps = [
        ComboComponent("a", 1.0, 20.0, "Normal", 0.3, sigma=6.0),
        ComboComponent("hits", 0.0, 1.6, "NegBin", 0.5),
    ]
    seen = {}

    def post(draws):
        seen.update(draws)
        return 0.5 * draws["hits"]

    quote = combo_sum_quote(comps, rho_zero, post=post)
    assert "hits" in seen and seen["hits"].mean() > 0
    expected = np.sort(seen["a"] + 0.5 * seen["hits"])
    assert np.array_equal(quote.draws_sorted, expected)
    assert quote.mean == pytest.approx(20.0 + 0.5 * seen["hits"].mean())

    no_post = combo_sum_quote(comps, rho_zero)
    assert no_post.mean == pytest.approx(20.0)
    assert np.array_equal(no_post.draws_sorted, np.sort(seen["a"]))


def test_component_cap_and_unknown_family():
    many = [ComboComponent(f"m{i}", 1.0, 2.0, "Poisson", 1.0) for i in range(10)]
    with pytest.raises(ValueError, match="1..9 components"):
        combo_sum_quote(many, rho_zero)
    with pytest.raises(ValueError, match="Mixture"):
        combo_sum_quote([ComboComponent("m", 1.0, 2.0, "Mixture", 1.0)], rho_zero)


def test_load_same_player_rho(monkeypatch):
    frame = pd.DataFrame(
        {
            "market_a": ["PTS", "PTS", "BLK"],
            "market_b": ["REB", "AST", "STL"],
            "rho_mean": [0.312, 0.201, 0.053],
            "n_teams": [30, 30, 30],
            "scope": ["same_player", "same_player", "same_team"],
        }
    )
    monkeypatch.setattr(cm.pd, "read_parquet", lambda _path: frame)
    rho = cm.load_same_player_rho("fakeleague")
    assert rho("PTS", "REB") == pytest.approx(0.312)
    assert rho("REB", "PTS") == pytest.approx(0.312)  # symmetric
    assert rho("BLK", "STL") == 0.0  # same_team scope is not same_player
    assert rho("PTS", "TOV") == 0.0  # missing pair
    assert rho("PTS", "PTS") == 1.0

    def boom(_path):
        raise FileNotFoundError

    monkeypatch.setattr(cm.pd, "read_parquet", boom)
    rho_none = cm.load_same_player_rho("nosuchleague")
    assert rho_none("a", "b") == 0.0
    assert rho_none("a", "a") == 1.0
