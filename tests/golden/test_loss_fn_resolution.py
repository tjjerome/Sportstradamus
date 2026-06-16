"""Unit tests for the loss-resolution helper in ``training.pipeline``.

The Operation Ship 75 search sweeps the distribution training loss (nll ↔ crps) as a search
axis via ``--dist-training-loss``. The flag carries an ``"auto"`` sentinel that means
"per-family default" — crps for the SkewNormal continuous branch, nll for the count
branch — so the default run reproduces production behavior byte-for-byte while an
explicit nll/crps overrides every family.
"""

from sportstradamus.training.pipeline import LOSS_AUTO, _resolve_loss_fn


def test_auto_sentinel_yields_the_per_family_default():
    assert _resolve_loss_fn("crps", LOSS_AUTO) == "crps"
    assert _resolve_loss_fn("nll", LOSS_AUTO) == "nll"


def test_explicit_override_wins_over_the_family_default():
    assert _resolve_loss_fn("crps", "nll") == "nll"
    assert _resolve_loss_fn("nll", "crps") == "crps"
