"""Golden pins for exact nonnegative-integer PIT and line settlement."""

import numpy as np
import pytest

from sportstradamus.helpers import integer_distribution as integer


def test_observation_points_are_the_two_sides_of_each_cdf_jump():
    lower, upper = integer.observation_cdf_points(np.array([0.0, 1.0, 4.0]))

    np.testing.assert_array_equal(lower, [-1, 0, 3])
    np.testing.assert_array_equal(upper, [0, 1, 4])


def test_randomized_pit_is_seeded_and_stays_inside_observed_atoms():
    outcomes = np.array([0.0, 2.0, 5.0])
    lower = np.array([0.0, 0.2, 0.8])
    upper = np.array([0.3, 0.55, 0.8])
    expected_uniforms = np.random.default_rng(integer.RANDOMIZED_PIT_SEED).random(
        (integer.RANDOMIZED_PIT_DRAWS, 3)
    )
    expected = lower[None, :] + expected_uniforms * (upper - lower)[None, :]

    first = integer.randomized_pit(outcomes, lower, upper)
    second = integer.randomized_pit(outcomes, lower, upper)

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, first)
    assert np.all((first[:, 0] >= 0.0) & (first[:, 0] <= 0.3))
    assert np.all((first[:, 1] >= 0.2) & (first[:, 1] <= 0.55))
    np.testing.assert_array_equal(first[:, 2], np.full(integer.RANDOMIZED_PIT_DRAWS, 0.8))


def test_whole_and_half_lines_use_exact_integer_settlement_points():
    lines = np.array([0.0, 0.5, 2.0, 2.5])
    lower_points, upper_points = integer.line_cdf_points(lines)

    np.testing.assert_array_equal(lower_points, [-1, 0, 1, 2])
    np.testing.assert_array_equal(upper_points, [0, 0, 2, 2])

    settled = integer.settlement_probabilities(
        lines,
        lower_cdf=np.array([0.0, 0.2, 0.6, 0.85]),
        upper_cdf=np.array([0.2, 0.2, 0.85, 0.85]),
    )

    np.testing.assert_allclose(settled.under, [0.0, 0.2, 0.6, 0.85])
    np.testing.assert_allclose(settled.push, [0.2, 0.0, 0.25, 0.0])
    np.testing.assert_allclose(settled.over, [0.8, 0.8, 0.15, 0.15])
    np.testing.assert_allclose(settled.under + settled.push + settled.over, 1.0)


def test_lines_below_zero_respect_nonnegative_support():
    settled = integer.settlement_probabilities(
        np.array([-0.5]),
        lower_cdf=np.array([0.0]),
        upper_cdf=np.array([0.0]),
    )

    np.testing.assert_array_equal(settled.under, [0.0])
    np.testing.assert_array_equal(settled.push, [0.0])
    np.testing.assert_array_equal(settled.over, [1.0])


@pytest.mark.parametrize(
    ("outcomes", "message"),
    [
        (np.array([np.nan]), "finite"),
        (np.array([-1.0]), "nonnegative"),
        (np.array([1.5]), "integers"),
    ],
)
def test_observation_points_reject_invalid_outcomes(outcomes, message):
    with pytest.raises(ValueError, match=message):
        integer.observation_cdf_points(outcomes)


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (np.array([np.nan]), np.array([0.2]), "finite"),
        (np.array([-0.1]), np.array([0.2]), r"\[0, 1\]"),
        (np.array([0.1]), np.array([1.1]), r"\[0, 1\]"),
        (np.array([0.7]), np.array([0.6]), "monotone"),
        (np.array([0.1]), np.array([0.2]), "nonnegative support"),
    ],
)
def test_randomized_pit_rejects_invalid_cdf_endpoints(lower, upper, message):
    with pytest.raises(ValueError, match=message):
        integer.randomized_pit(np.array([0.0]), lower, upper)


def test_line_settlement_rejects_non_lattice_and_inconsistent_endpoints():
    with pytest.raises(ValueError, match="whole or half"):
        integer.line_cdf_points(np.array([1.25]))
    with pytest.raises(ValueError, match="finite"):
        integer.line_cdf_points(np.array([np.inf]))
    with pytest.raises(ValueError, match="half-line"):
        integer.settlement_probabilities(
            np.array([1.5]),
            lower_cdf=np.array([0.4]),
            upper_cdf=np.array([0.5]),
        )
    with pytest.raises(ValueError, match="nonnegative support"):
        integer.settlement_probabilities(
            np.array([0.0]),
            lower_cdf=np.array([0.1]),
            upper_cdf=np.array([0.4]),
        )


def test_endpoint_arrays_must_be_row_aligned():
    with pytest.raises(ValueError, match="match the observation or line shape"):
        integer.randomized_pit(
            np.array([1.0, 2.0]),
            lower_cdf=np.array([0.2]),
            upper_cdf=np.array([0.4, 0.7]),
        )
