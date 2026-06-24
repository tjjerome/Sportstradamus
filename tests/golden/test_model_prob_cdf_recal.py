"""§6.1 Rung C — the inference over-prob warp matches the gate's whole-CDF recalibration.

Inference serves ``over = 1 − g(F(line))``. Since the raw model over-prob is
``over_raw = 1 − F(line)``, the seam applies ``1 − g(1 − over_raw)`` — the same map ``g``
the offline Gate-4 scored on the PIT. A legacy pickle (no ``pit_recal_blob``) must be an
exact no-op so a pre-Rung-C cell serves byte-identically.
"""

import numpy as np

from sportstradamus.prediction.model_prob import _apply_cdf_recal_over
from sportstradamus.training import posthoc


def test_apply_cdf_recal_over_matches_one_minus_g_of_cdf():
    rng = np.random.default_rng(7)
    blob = posthoc.fit_isotonic_pit(rng.beta(0.5, 0.5, size=2000))
    cdf = rng.uniform(0.0, 1.0, size=50)  # F(line) for a batch of offers
    over_raw = 1.0 - cdf
    warped = _apply_cdf_recal_over(over_raw, blob)
    # The served over-prob must equal 1 − g(F(line)) — the recalibrated CDF the gate used.
    np.testing.assert_allclose(warped, 1.0 - posthoc.apply_cdf_recal(blob, cdf))


def test_apply_cdf_recal_over_none_is_identity():
    over = np.array([0.1, 0.4, 0.9])
    np.testing.assert_array_equal(_apply_cdf_recal_over(over, None), over)
