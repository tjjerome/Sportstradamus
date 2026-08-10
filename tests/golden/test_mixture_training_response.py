"""Golden pins for deterministic Gaussian-mixture training and prediction responses."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from lightgbmlss.model import LightGBMLSS

from sportstradamus.helpers import set_model_start_values
from sportstradamus.training.pipeline import _continuous_dist_obj


class _IndexedRawBooster:
    def __init__(self, raw_scores: pd.DataFrame):
        self.raw_scores = raw_scores

    def predict(self, frame: pd.DataFrame, *, raw_score: bool) -> np.ndarray:
        assert raw_score
        return self.raw_scores.loc[frame.index].to_numpy()


@pytest.fixture
def mixture_distribution():
    return _continuous_dist_obj(
        "Mixture",
        "direct",
        "None",
        "auto",
        np.array([0.0, 1.0, 2.0, 3.0]),
    )


def _loss_and_gradients(distribution, raw_heads, targets, start_values):
    """Evaluate six raw heads in LightGBMLSS's Fortran-major objective convention."""
    raw_tensors, loss = distribution.get_params_loss(
        raw_heads.flatten(order="F").copy(),
        targets,
        start_values,
        requires_grad=True,
    )
    gradients = torch.cat(torch.autograd.grad(loss, raw_tensors), dim=1)
    return loss.item(), gradients.detach().numpy()


def test_softmax_response_is_deterministic_and_permutation_equivariant(
    mixture_distribution,
):
    response = mixture_distribution.param_dict["mix_prob"]
    logits = torch.tensor(
        [
            [np.log(0.75), np.log(0.25)],
            [1.2, -0.4],
            [-1.0, 0.7],
            [0.2, 0.1],
        ],
        dtype=torch.float64,
    )
    row_order = torch.tensor([2, 0, 3, 1])
    inverse_row_order = torch.argsort(row_order)
    component_order = torch.tensor([1, 0])

    with torch.random.fork_rng():
        torch.manual_seed(947)
        rng_state = torch.random.get_rng_state().clone()
        probabilities = response(logits)
        reordered_rows = response(logits[row_order])[inverse_row_order]
        reordered_components = response(logits[:, component_order])[:, component_order]
        assert torch.equal(torch.random.get_rng_state(), rng_state)

    torch.testing.assert_close(probabilities, torch.softmax(logits, dim=1))
    torch.testing.assert_close(probabilities[0], torch.tensor([0.75, 0.25], dtype=torch.float64))
    torch.testing.assert_close(reordered_rows, probabilities)
    torch.testing.assert_close(reordered_components, probabilities)
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(probabilities.shape[0], dtype=probabilities.dtype),
    )
    assert torch.all(probabilities > 0.0)
    assert torch.all(probabilities < 1.0)


def test_mixture_loss_and_gradients_are_permutation_equivariant(mixture_distribution):
    raw_heads = np.array(
        [
            [0.1, 1.8, -0.2, 0.1, 0.4, -0.1],
            [-0.4, 2.2, -0.1, 0.2, -0.2, 0.5],
            [0.7, 1.5, 0.0, -0.1, 0.8, 0.1],
            [0.2, 2.6, 0.1, 0.0, -0.5, 0.3],
        ],
        dtype=np.float64,
    )
    targets = torch.tensor([0.2, 1.1, -0.4, 2.3], dtype=torch.float64)
    start_values = [0.0, 2.0, -0.2, 0.1, np.log(0.75), np.log(0.25)]
    row_order = np.array([2, 0, 3, 1])
    inverse_row_order = np.argsort(row_order)
    component_order = np.array([1, 0, 3, 2, 5, 4])
    inverse_component_order = np.argsort(component_order)

    base_loss, base_gradients = _loss_and_gradients(
        mixture_distribution,
        raw_heads,
        targets,
        start_values,
    )
    row_loss, row_gradients = _loss_and_gradients(
        mixture_distribution,
        raw_heads[row_order],
        targets[row_order],
        start_values,
    )
    component_loss, component_gradients = _loss_and_gradients(
        mixture_distribution,
        raw_heads[:, component_order],
        targets,
        np.asarray(start_values)[component_order].tolist(),
    )

    assert row_loss == pytest.approx(base_loss, rel=1e-12, abs=1e-12)
    assert component_loss == pytest.approx(base_loss, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(
        row_gradients[inverse_row_order],
        base_gradients,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        component_gradients[:, inverse_component_order],
        base_gradients,
        rtol=1e-10,
        atol=1e-10,
    )


def test_parameter_prediction_is_order_invariant_after_pickle(mixture_distribution):
    features = pd.DataFrame(
        {
            "MeanYr": [10.0, 6.0, 14.0],
            "STDYr": [3.0, 2.0, 4.0],
            "ZeroYr": [0.1, 0.2, 0.05],
            "feature": [0.5, -0.2, 1.1],
        },
        index=["alpha", "bravo", "charlie"],
    )
    raw_scores = pd.DataFrame(
        [
            [0.1, 0.2, 0.0, 0.0, 1.0, -0.5],
            [0.3, 0.1, 0.1, -0.1, -0.2, 0.4],
            [-0.2, 0.4, -0.1, 0.2, 0.7, 0.1],
        ],
        index=features.index,
        dtype=np.float32,
    )
    model = LightGBMLSS(mixture_distribution)
    model.booster = _IndexedRawBooster(raw_scores)
    set_model_start_values(model, "Mixture", features)
    parameters = model.predict(features, pred_type="parameters", n_samples=1)
    parameters.index = features.index

    start_logits = torch.log(torch.tensor([0.75, 0.25], dtype=torch.float32))
    expected_weights = torch.softmax(
        torch.tensor(raw_scores.iloc[:, 4:6].to_numpy()) + start_logits,
        dim=1,
    ).numpy()
    np.testing.assert_allclose(
        parameters[["mix_prob_1", "mix_prob_2"]].to_numpy(),
        expected_weights,
        rtol=1e-6,
        atol=1e-7,
    )

    restored = pickle.loads(pickle.dumps(model))
    row_order = ["charlie", "alpha", "bravo"]
    reordered_features = features.loc[row_order]
    set_model_start_values(restored, "Mixture", reordered_features)
    reordered_parameters = restored.predict(
        reordered_features,
        pred_type="parameters",
        n_samples=1,
    )
    reordered_parameters.index = reordered_features.index

    pd.testing.assert_frame_equal(
        reordered_parameters.loc[features.index],
        parameters,
        check_exact=True,
    )
