"""Calibration-aware HP levers (research brief /tmp/researcher_calibration_hp.md).

Lever 2 — count-branch dispersion calibration re-targeted from CRPS to the Gate-4
randomized-PIT KS (`_dispersion_pit_ks_loss`).
Lever 1 — calibration-constrained HP *search*: the Optuna objective is the one-sided
search-gating score `_penalized_objective` (CRPS plus a hinge on the validation PIT-KS over
the Gate-4 threshold), so the sampler explores the wider-sigma region; the final pick is
`_pick_calibrated_candidate` (lowest-CRPS trial clearing the threshold, else best-calibrated).

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


def test_pick_calibrated_candidate_logs_headroom(caplog):
    """R2 step 1 (Experiment-B probe): one INFO line reports the feasible set the
    epsilon-constraint saw — min PIT-KS, feasible count, threshold, and the CRPS price
    of feasibility (best feasible minus global best); price is n/a when no trial is
    feasible. Diagnostic only — the pick itself is unchanged."""
    import logging

    from sportstradamus.training.pipeline import _pick_calibrated_candidate

    # _pick_calibrated_candidate's logger sets propagate=False (helpers.get_logger),
    # so attach caplog's handler directly to it by its real name.
    target = logging.getLogger("sportstradamus.cli.sportstradamus.training.pipeline")
    target.addHandler(caplog.handler)
    try:
        _pick_calibrated_candidate(
            [
                {"cv_loss": 1.0, "pit_ks": 0.20},
                {"cv_loss": 1.2, "pit_ks": 0.03},
                {"cv_loss": 1.5, "pit_ks": 0.04},
            ],
            threshold=0.05,
        )
        _pick_calibrated_candidate(
            [
                {"cv_loss": 1.0, "pit_ks": 0.20},
                {"cv_loss": 1.4, "pit_ks": 0.11},
            ],
            threshold=0.05,
        )
    finally:
        target.removeHandler(caplog.handler)

    headroom = [r.getMessage() for r in caplog.records if "headroom" in r.getMessage()]
    assert len(headroom) == 2
    assert "min_pit_ks=0.0300" in headroom[0]
    assert "feasible=2/3" in headroom[0]
    assert "threshold=0.0500" in headroom[0]
    assert "crps_price=0.20000" in headroom[0]  # cv_loss(best feasible 1.2) - cv_loss(best 1.0)
    assert "min_pit_ks=0.1100" in headroom[1]
    assert "feasible=0/2" in headroom[1]
    assert "crps_price=n/a" in headroom[1]


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


def test_penalized_objective_is_pure_crps_in_feasible_region():
    """Lever 1 search-gate: the hinge is one-sided, so a calibrated trial (PIT-KS at or below the
    threshold) is scored by pure CRPS — the feasible region keeps sharpest-wins, with no pull toward
    the over-dispersed marginal predictor (the GBR degeneracy guard)."""
    from sportstradamus.training.hyperparams import _penalized_objective

    assert _penalized_objective(3.0, 0.04, 0.05) == 3.0
    assert _penalized_objective(3.0, 0.05, 0.05) == 3.0


def test_penalized_objective_steers_search_away_from_uncalibrated():
    """Lever 1 search-gate: a sharp-but-uncalibrated trial (low CRPS, PIT-KS over the gate) must
    score WORSE than a blunter calibrated one, so the TPE sampler is steered into the wider-sigma
    feasible region instead of only re-ranking the sharpest cluster post hoc."""
    from sportstradamus.training.hyperparams import _penalized_objective

    sharp_uncalibrated = _penalized_objective(3.00, 0.10, 0.05)
    blunt_calibrated = _penalized_objective(3.20, 0.04, 0.05)
    assert sharp_uncalibrated > blunt_calibrated


def test_clamp_seed_params_rescues_sub_floor_log_param():
    """Warm-start regression: a pickle storing lambda_l1=0.0 (LightGBM's default, below the
    1e-6 log-scale floor) must be clamped up to the floor, not enqueued as-is — seeding 0.0 into
    a log=True distribution makes Optuna take log(0) and abort the whole study. In-bounds seeds
    pass through unchanged and 'none'-type (fixed) params are not seeded."""
    from sportstradamus.training.hyperparams import _clamp_seed_params

    hp_dict = {
        "num_threads": ["none", [8]],
        "lambda_l1": ["float", {"low": 1e-6, "high": 10, "log": True}],
        "num_leaves": ["int", {"low": 8, "high": 127, "log": False}],
    }
    seed = _clamp_seed_params(hp_dict, {"lambda_l1": 0.0, "num_leaves": 31, "num_threads": 8})

    assert seed["lambda_l1"] == 1e-6  # sub-floor log param clamped up — no log(0) abort
    assert seed["num_leaves"] == 31  # in-bounds seed unchanged
    assert "num_threads" not in seed  # 'none'-type fixed params are not seeded


def test_resolve_cell_knob_persists_per_cell_selection_and_honors_flag_override():
    """Durability: a cell's stat_meta training knob (hpo_selection, blending,
    count_dispersion_objective, zinb_mode) is what the production cron applies under the default 'auto' flag,
    so a tuned ship reproduces on retrain instead of darking out; an explicit flag forces every
    cell for a one-shot A/B."""
    from sportstradamus.training.cli import LOSS_AUTO, _resolve_cell_knob

    sm = {
        "WNBA": {
            "PR": {"hpo_selection": "calibrated"},
            "OREB": {"count_dispersion_objective": "pit_ks"},
            "FTM": {"zinb_mode": "hurdle"},
        }
    }
    assert _resolve_cell_knob(sm, "WNBA", "PR", "hpo_selection", "loss", LOSS_AUTO) == "calibrated"
    assert _resolve_cell_knob(sm, "WNBA", "PA", "hpo_selection", "loss", LOSS_AUTO) == "loss"
    assert _resolve_cell_knob(sm, "WNBA", "PR", "hpo_selection", "loss", "loss") == "loss"
    # count_dispersion_objective persists identically, so OREB's pit_ks ship reproduces on the cron
    assert _resolve_cell_knob(sm, "WNBA", "OREB", "count_dispersion_objective", "crps", LOSS_AUTO) == "pit_ks"
    assert _resolve_cell_knob(sm, "WNBA", "BLST", "count_dispersion_objective", "crps", LOSS_AUTO) == "crps"
    assert _resolve_cell_knob(sm, "WNBA", "OREB", "count_dispersion_objective", "crps", "crps") == "crps"
    # zinb_mode persists so a hurdle ship reproduces on the cron instead of darking back to joint
    assert _resolve_cell_knob(sm, "WNBA", "FTM", "zinb_mode", "joint", LOSS_AUTO) == "hurdle"
    assert _resolve_cell_knob(sm, "WNBA", "PA", "zinb_mode", "joint", LOSS_AUTO) == "joint"
    assert _resolve_cell_knob(sm, "WNBA", "FTM", "zinb_mode", "joint", "joint") == "joint"
