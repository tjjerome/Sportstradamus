import numpy as np
import pytest

from sportstradamus.training import calibration


def test_nll_is_default_and_registered():
    assert calibration.DEFAULT_BLENDING == "nll"
    assert "nll" in calibration.BLENDING_SLUGS


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
