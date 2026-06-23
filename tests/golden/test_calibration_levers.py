"""Calibration-aware HP levers (research brief /tmp/researcher_calibration_hp.md).

Lever 2 — count-branch dispersion calibration re-targeted from CRPS to the Gate-4
randomized-PIT KS (`_dispersion_pit_ks_loss`).
Lever 1 — calibration-constrained HP selection policy (`_pick_calibrated_candidate`):
pick the lowest-CRPS trial whose CV PIT-KS clears the gate threshold, else fall back to
the best-calibrated trial.

Both are opt-in; production defaults (CRPS dispersion, loss-only selection) are unchanged.
"""

from __future__ import annotations

import numpy as np


def test_dispersion_pit_ks_loss_negbin_minimized_at_calibrating_scale():
    """The count PIT-KS loss is lower at the ``c`` that restores the true NegBin
    shape than at the over-dispersed blended shape."""
    from sportstradamus.training.pipeline import _dispersion_pit_ks_loss

    rng = np.random.default_rng(0)
    n, r_true, mu = 6000, 12.0, 5.0
    p_success = r_true / (r_true + mu)
    y = rng.negative_binomial(r_true, p_success, size=n).astype(float)

    r_blend = np.full(n, r_true / 2.0)  # half the true shape → too over-dispersed
    mean = np.full(n, mu)

    def loss(c):
        return _dispersion_pit_ks_loss(
            c,
            dist="NegBin",
            y_val_arr=y,
            val_weighted_mean=mean,
            gate_blend_val=None,
            r_blend_val=r_blend,
        )

    assert 0.0 <= loss(2.0) <= 1.5
    assert loss(2.0) < loss(1.0)


def test_dispersion_pit_ks_loss_gamma_holds_mean_fixed():
    """Gamma branch scales the shape while holding the mean (EV) fixed, so a
    mis-shaped blended alpha is corrected toward the calibrating ``c``."""
    from sportstradamus.training.pipeline import _dispersion_pit_ks_loss

    rng = np.random.default_rng(1)
    n, a_true, mu = 5000, 8.0, 4.0
    y = rng.gamma(shape=a_true, scale=mu / a_true, size=n)

    a_blend = np.full(n, a_true / 2.0)  # too few shape → too wide; c=2 restores it
    mean = np.full(n, mu)

    def loss(c):
        return _dispersion_pit_ks_loss(
            c,
            dist="Gamma",
            y_val_arr=y,
            val_weighted_mean=mean,
            gate_blend_val=None,
            alpha_blend_val=a_blend,
        )

    assert loss(2.0) < loss(1.0)


def test_pick_calibrated_candidate_prefers_lowest_loss_within_threshold():
    """Lever 1 policy: among trials clearing the PIT-KS threshold, the lowest CRPS wins."""
    from sportstradamus.training.pipeline import _pick_calibrated_candidate

    candidates = [
        {"name": "sharp_miscalibrated", "cv_loss": 1.0, "pit_ks": 0.20},
        {"name": "calibrated_best", "cv_loss": 1.2, "pit_ks": 0.03},
        {"name": "calibrated_worse", "cv_loss": 1.5, "pit_ks": 0.04},
    ]
    picked = _pick_calibrated_candidate(candidates, threshold=0.05)
    assert picked["name"] == "calibrated_best"


def test_pick_calibrated_candidate_falls_back_to_best_pit_ks():
    """When no trial clears the threshold, fall back to the best-calibrated (not the sharpest)."""
    from sportstradamus.training.pipeline import _pick_calibrated_candidate

    candidates = [
        {"name": "sharp", "cv_loss": 1.0, "pit_ks": 0.20},
        {"name": "least_miscalibrated", "cv_loss": 1.4, "pit_ks": 0.11},
    ]
    picked = _pick_calibrated_candidate(candidates, threshold=0.05)
    assert picked["name"] == "least_miscalibrated"


def test_stabilization_arg_reaches_distribution_attribute():
    """Lever 4 guard: the --stabilization value must land on a real LightGBMLSS attribute,
    so a renamed constructor arg can't make the flag a silent no-op."""
    from sportstradamus.skew_normal import SkewNormal

    assert SkewNormal(stabilization="MAD", loss_fn="crps").stabilization == "MAD"
    assert SkewNormal(stabilization="None", loss_fn="crps").stabilization == "None"


def test_resolve_cell_knob_persists_per_cell_selection_and_honors_flag_override():
    """Lever-1 durability: a cell's stat_meta hpo_selection is what the production cron applies
    under the default 'auto' flag, so a calibrated-selected ship reproduces on retrain instead of
    darking out; an explicit flag forces every cell for a one-shot A/B."""
    from sportstradamus.training.cli import LOSS_AUTO, _resolve_cell_knob

    sm = {"WNBA": {"PR": {"hpo_selection": "calibrated"}}}
    assert _resolve_cell_knob(sm, "WNBA", "PR", "hpo_selection", "loss", LOSS_AUTO) == "calibrated"
    assert _resolve_cell_knob(sm, "WNBA", "PA", "hpo_selection", "loss", LOSS_AUTO) == "loss"
    assert _resolve_cell_knob(sm, "WNBA", "PR", "hpo_selection", "loss", "loss") == "loss"
