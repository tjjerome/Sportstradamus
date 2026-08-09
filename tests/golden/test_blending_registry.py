import numpy as np
import pytest
from scipy.stats import nbinom

from sportstradamus.training import calibration


def test_nll_is_default_and_registered():
    assert calibration.DEFAULT_BLENDING == "nll"
    assert "nll" in calibration.BLENDING_SLUGS


def test_crps_registered_default_still_nll():
    assert "crps" in calibration.BLENDING_SLUGS
    assert calibration.DEFAULT_BLENDING == "nll"


def test_fit_blend_weight_crps_returns_bounded_weight_each_family():
    rng = np.random.default_rng(3)
    n = 400
    model_ev = rng.uniform(1.0, 6.0, n)
    odds_ev = model_ev + rng.normal(0, 0.4, n)
    result = np.maximum(0, np.round(rng.normal(model_ev, 1.5))).astype(float)
    families = (
        {"dist": "NegBin", "model_r": np.full(n, 5.0), "cv": 0.5},
        {"dist": "Gamma", "model_alpha": np.full(n, 4.0), "cv": 0.5},
        {
            "dist": "SkewNormal",
            "model_sigma": np.full(n, 1.5),
            "model_skew_alpha": np.zeros(n),
            "cv": 0.5,
        },
    )
    for fam in families:
        dist = fam.pop("dist")
        w = calibration.fit_blend_weight("crps", model_ev, odds_ev, result, dist, **fam)
        assert calibration._MODEL_WEIGHT_MIN - 1e-9 <= w <= calibration._MODEL_WEIGHT_MAX + 1e-9


def test_crps_weight_favors_the_correct_side():
    # Data drawn from the model's NegBin: with the book badly biased the crps-fit weight leans to
    # the model (high w); flip which side is right and the weight drops.
    rng = np.random.default_rng(7)
    n = 1500
    true_mean = rng.uniform(3.0, 9.0, n)
    r = np.full(n, 6.0)
    p = r / (r + true_mean)
    result = nbinom.rvs(r, p, random_state=rng).astype(float)

    w_model_right = calibration.fit_blend_weight(
        "crps", true_mean, true_mean + 4.0, result, "NegBin", model_r=r, cv=0.5
    )
    w_model_wrong = calibration.fit_blend_weight(
        "crps", true_mean + 4.0, true_mean, result, "NegBin", model_r=r, cv=0.5
    )
    assert w_model_right > w_model_wrong


def test_nll_strategy_reproduces_fit_model_weight():
    # The registry's nll entry must equal the legacy fit_model_weight output.
    rng = np.random.default_rng(1)
    n = 300
    model_ev = rng.uniform(0.5, 3.0, n)
    odds_ev = model_ev + rng.normal(0, 0.3, n)
    result = np.maximum(0, np.round(rng.normal(model_ev, 1.0)))
    legacy = calibration.fit_model_weight(
        model_ev, odds_ev, result, "NegBin", model_r=np.full(n, 5.0), cv=0.5
    )
    via_registry = calibration.fit_blend_weight(
        "nll", model_ev, odds_ev, result, "NegBin", model_r=np.full(n, 5.0), cv=0.5
    )
    assert abs(legacy - via_registry) < 1e-9


def test_unknown_blending_slug_rejected():
    from sportstradamus.training import ship_config

    with pytest.raises(ValueError, match="unknown blending"):
        ship_config._validate_cell(
            "NFL",
            "passing-tds",
            {
                "shipped": "withheld",
                "dist": "ZINB",
                "target_normalization": "none",
                "posthoc": "none",
                "blending": "bogus",
            },
        )


def test_fit_blend_weight_rejects_unknown_slug():
    with pytest.raises(ValueError, match="Unknown blending slug"):
        calibration.fit_blend_weight("bogus", [1.0], [1.0], [1.0], "NegBin")


def test_validate_cell_mixture_is_continuous():
    """Mixture cells follow the continuous-family norm rules: a real slug is valid
    (and required to ship); a count cell still cannot carry one.
    """
    from sportstradamus.training import ship_config

    mix_cell = {
        "shipped": "devel",
        "dist": "Mixture",
        "target_normalization": "ratio_meanyr",
        "posthoc": "none",
    }
    ship_config._validate_cell("NFL", "yards", mix_cell)
    with pytest.raises(ValueError, match="cannot ship"):
        ship_config._validate_cell("NFL", "yards", {**mix_cell, "target_normalization": "none"})
    with pytest.raises(ValueError, match="cannot carry"):
        ship_config._validate_cell(
            "NFL", "yards", {**mix_cell, "dist": "NegBin", "shipped": "withheld"}
        )
