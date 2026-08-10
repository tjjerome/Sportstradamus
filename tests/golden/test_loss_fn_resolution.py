"""Unit tests for the loss- and distribution-resolution helpers in ``training.pipeline``.

The Operation Ship 75 search sweeps the distribution training loss (nll ↔ crps) as a search
axis via ``--dist-training-loss``. The flag carries an ``"auto"`` sentinel that means
"per-family default" — crps for the SkewNormal continuous branch, nll for the count
branch — so the default run reproduces production behavior byte-for-byte while an
explicit nll/crps overrides every family.

Distribution selection is likewise config-first: a cell's stat_meta ``dist`` is authoritative
(``_resolve_dist``), with the data-driven ``global_mean`` / zero-rate rule (``_data_driven_dist``)
as the fallback for an unset / ``"auto"`` cell. The WS-3 ``meditate --dist`` axis threads a
family override into that seam: ``"auto"`` defers to the cell config; an explicit family becomes
``_resolve_dist``'s authoritative ``configured`` for every cell.
"""

import logging

import pytest

from sportstradamus.training.pipeline import (
    LOSS_AUTO,
    _data_driven_dist,
    _deterministic_dump_suffix,
    _resolve_dist,
    _resolve_loss_fn,
)


def test_auto_sentinel_yields_the_per_family_default():
    assert _resolve_loss_fn("crps", LOSS_AUTO) == "crps"
    assert _resolve_loss_fn("nll", LOSS_AUTO) == "nll"


def test_explicit_override_wins_over_the_family_default():
    assert _resolve_loss_fn("crps", "nll") == "nll"
    assert _resolve_loss_fn("nll", "crps") == "crps"


def test_data_driven_dist_by_mean_then_zero_rate():
    assert _data_driven_dist(5.0, 0.0) == "SkewNormal"  # mean >= 2 → continuous
    assert _data_driven_dist(0.7, 0.57) == "ZINB"  # mean < 2, zeros clear the ZINB gate
    assert _data_driven_dist(0.1, 0.01) == "NegBin"  # mean < 2, too few zeros for ZINB


def test_resolve_dist_falls_back_to_data_when_unset_or_auto():
    assert _resolve_dist(None, "ZINB", 0.7, 0.57, "NHL", "blocked") == "ZINB"
    assert _resolve_dist("auto", "SkewNormal", 5.0, 0.0, "NBA", "PTS") == "SkewNormal"


def test_resolve_dist_configured_is_authoritative_and_warns_only_on_disagreement(caplog):
    with caplog.at_level(logging.WARNING):
        assert _resolve_dist("ZINB", "ZINB", 0.7, 0.57, "NHL", "blocked") == "ZINB"
    assert not caplog.records  # agreement: forced == data-driven, silent

    with caplog.at_level(logging.WARNING):
        assert _resolve_dist("SkewNormal", "ZINB", 0.7, 0.57, "NHL", "blocked") == "SkewNormal"
    assert any("forced dist=SkewNormal" in r.getMessage() for r in caplog.records)  # trains + warns


def test_resolve_dist_unknown_family_raises():
    with pytest.raises(ValueError, match="not a forceable family"):
        _resolve_dist("Gamma", "ZINB", 0.7, 0.57, "NHL", "blocked")


def _select_configured(dist_override: str, stat_meta_dist: str | None) -> str | None:
    """The ``configured`` selection ``_step_select_distribution`` feeds ``_resolve_dist``.

    Mirrors the seam expression so the --dist precedence is asserted without a full training run:
    the override wins unless ``"auto"``, in which case the cell's stat_meta dist stands.
    """
    return dist_override if dist_override != LOSS_AUTO else stat_meta_dist


def test_dist_override_forces_family_over_stat_meta_pin():
    # --dist NegBin on a cell pinned ZINB in stat_meta trains NegBin (emits the disagree warning).
    configured = _select_configured("NegBin", "ZINB")
    assert _resolve_dist(configured, "ZINB", 0.7, 0.57, "WNBA", "TOV") == "NegBin"


def test_dist_override_auto_defers_to_stat_meta_then_data():
    # auto → cell config stands; an unset cell then falls through to the data-driven dist.
    assert _select_configured(LOSS_AUTO, "ZINB") == "ZINB"
    configured = _select_configured(LOSS_AUTO, None)
    assert _resolve_dist(configured, "SkewNormal", 5.0, 0.0, "NBA", "PTS") == "SkewNormal"


def test_dist_override_invalid_family_fails_loud():
    configured = _select_configured("Gamma", "ZINB")
    with pytest.raises(ValueError, match="not a forceable family"):
        _resolve_dist(configured, "ZINB", 0.7, 0.57, "WNBA", "TOV")


def test_deterministic_dump_suffix_keys_on_trained_arch_not_raw_zinb_mode():
    # The bug: a --dist NegBin corner on a hurdle-pinned cell (e.g. WNBA TOV) trains plain NegBin
    # and must dump to the non-hurdle path.
    assert _deterministic_dump_suffix("NegBin", "hurdle") == ""
    assert _deterministic_dump_suffix("ZINB", "hurdle") == "_hurdle"
    assert _deterministic_dump_suffix("ZINB", "joint") == ""
    assert _deterministic_dump_suffix("SkewNormal", "joint") == ""
