"""Pin the book-anchored calibration-in-the-large corrector (``prob_recal_book_citl``).

A slope-1 Platt map whose single intercept equates the mean served over-probability to
the mean book over-probability on the rows that carry a quote. Outcomes are never
consulted, so the fit cannot learn the validation fold's over-rate; unquoted rows stay
out of the fit, and too few quoted rows degrade to identity.
"""

import numpy as np
import pytest
from scipy.special import expit, logit

from sportstradamus.training import posthoc
from sportstradamus.training.model_strategy import registered_strategies

_SLUG = "prob_recal_book_citl"


def _synthetic(n: int, seed: int, offset: float = 0.3):
    """Served probs sitting ``offset`` logits above the book's quotes on the same rows."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    book = expit(z + rng.normal(0.0, 0.2, n))
    x = expit(z + offset)
    y = (rng.random(n) < book).astype(float)
    return x, y, book


def _feat(x: np.ndarray) -> np.ndarray:
    return logit(np.clip(x, posthoc._PROB_CLIP, 1 - posthoc._PROB_CLIP))


def test_intercept_equates_mean_served_to_mean_book():
    x, y, book = _synthetic(2000, seed=7, offset=0.3)
    blob = posthoc.fit_posthoc(_SLUG, x, y, book=book)
    assert blob["a"] == 1.0
    corrected = posthoc.apply_posthoc(_SLUG, blob, x)
    assert corrected.mean() == pytest.approx(book.mean(), abs=1e-6)
    # The planted logit shift is what the intercept undoes.
    assert blob["b"] == pytest.approx(-0.3, abs=0.05)


def test_unquoted_rows_are_excluded_from_the_fit():
    x, y, book = _synthetic(600, seed=3)
    quoted = np.ones(x.size, dtype=bool)
    quoted[::3] = False
    mixed = np.where(quoted, book, np.nan)
    assert posthoc.fit_posthoc(_SLUG, x, y, book=mixed) == posthoc.fit_posthoc(
        _SLUG, x[quoted], y[quoted], book=book[quoted]
    )


def test_too_few_quoted_rows_is_identity():
    x, y, book = _synthetic(60, seed=5)
    sparse = np.full(x.size, np.nan)
    sparse[: posthoc._MIN_FIT_ROWS - 1] = book[: posthoc._MIN_FIT_ROWS - 1]
    assert posthoc.fit_posthoc(_SLUG, x, y, book=sparse) is None
    sparse[posthoc._MIN_FIT_ROWS - 1] = book[posthoc._MIN_FIT_ROWS - 1]
    assert posthoc.fit_posthoc(_SLUG, x, y, book=sparse) is not None


def test_blob_is_a_plain_platt_map():
    x, y, book = _synthetic(300, seed=11)
    blob = posthoc.fit_posthoc(_SLUG, x, y, book=book)
    assert isinstance(blob, dict)
    assert all(isinstance(v, (str, float)) for v in blob.values())
    probe = np.linspace(1e-6, 1 - 1e-6, 500)
    expected = np.clip(expit(_feat(probe) + blob["b"]), posthoc._PROB_CLIP, 1 - posthoc._PROB_CLIP)
    np.testing.assert_allclose(posthoc.apply_posthoc(_SLUG, blob, probe), expected)


def test_outcomes_are_never_consulted():
    x, y, book = _synthetic(300, seed=13)
    blob = posthoc.fit_posthoc(_SLUG, x, y, book=book)
    rng = np.random.default_rng(0)
    for other_y in (rng.permutation(y), 1.0 - y, np.zeros_like(y)):
        assert posthoc.fit_posthoc(_SLUG, x, other_y, book=book) == blob


def test_registry_exposes_the_new_slug():
    assert _SLUG in posthoc.PROB_STAGE
    assert _SLUG in posthoc.POSTHOC_SLUGS


@pytest.mark.parametrize(
    "spec", [s for s in registered_strategies() if s.axes], ids=lambda spec: spec.slug
)
def test_sweep_axis_offers_the_slug_on_every_searchable_family(spec):
    assert _SLUG in spec.axes["posthoc"]
