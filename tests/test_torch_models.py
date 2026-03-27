"""Tests for the lightweight PyTorch model wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from mltsa.models import CNN1D, LSTM, MLP, get_model


@pytest.fixture()
def tiny_sequence_data() -> tuple[np.ndarray, np.ndarray]:
    """Provide a tiny deterministic binary time-series dataset."""

    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 5, 3)).astype(np.float32)
    signal = X[:, :, 0].mean(axis=1) + 0.5 * X[:, -1, 1]
    y = (signal > 0.0).astype(np.int64)
    return X, y


@pytest.mark.parametrize(
    ("factory_name", "kwargs", "wrapper_type"),
    [
        ("mlp", {"hidden_sizes": (16,), "epochs": 4, "batch_size": 8, "learning_rate": 0.01}, MLP),
        ("lstm", {"hidden_size": 12, "epochs": 4, "batch_size": 8, "learning_rate": 0.01}, LSTM),
        ("cnn1d", {"channels": 8, "epochs": 4, "batch_size": 8, "learning_rate": 0.01}, CNN1D),
    ],
)
def test_torch_wrappers_fit_predict_and_save_load(
    workspace_tmp_dir,
    tiny_sequence_data,
    factory_name: str,
    kwargs: dict[str, object],
    wrapper_type: type[MLP | LSTM | CNN1D],
) -> None:
    """Torch wrappers should fit, predict, and reload on tiny data."""

    X, y = tiny_sequence_data
    model = get_model(factory_name, random_state=0, **kwargs)

    assert isinstance(model, wrapper_type)

    fitted = model.fit(X, y)
    predictions = fitted.predict(X)
    probabilities = fitted.predict_proba(X)

    assert predictions.shape == y.shape
    assert probabilities.shape == (X.shape[0], 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(X.shape[0]), atol=1e-5)

    save_path = workspace_tmp_dir / f"{factory_name}.pt"
    fitted.save(save_path)
    reloaded = wrapper_type.load(save_path)
    reloaded_probabilities = reloaded.predict_proba(X)

    assert reloaded.predict(X).shape == y.shape
    np.testing.assert_allclose(reloaded_probabilities.sum(axis=1), np.ones(X.shape[0]), atol=1e-5)
