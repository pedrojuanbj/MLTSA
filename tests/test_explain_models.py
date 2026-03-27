"""Tests for sklearn wrappers and explainability helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from mltsa.explain import analyze, global_mean_importance, native_importance, permutation_importance
from mltsa.io.h5 import open_h5, read_utf8_array
from mltsa.io.schema import results_experiment_path
from mltsa.models import get_model


@pytest.fixture()
def classification_data() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Provide a deterministic classification dataset for model tests."""

    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=7,
    )
    feature_names = tuple(f"feat_{index}" for index in range(X.shape[1]))
    return X.astype(np.float64), y.astype(np.int64), feature_names


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("random_forest", {"n_estimators": 20, "random_state": 0}),
        ("gradient_boosting", {"n_estimators": 20, "random_state": 0}),
        ("hist_gradient_boosting", {"max_iter": 30, "random_state": 0}),
        ("extra_trees", {"n_estimators": 20, "random_state": 0}),
    ],
)
def test_model_wrappers_fit_and_predict(name: str, kwargs: dict[str, int], classification_data) -> None:
    """All supported wrapper models should fit and predict."""

    X, y, _ = classification_data
    model = get_model(name, **kwargs)

    fitted = model.fit(X, y)
    predictions = fitted.predict(X)

    assert predictions.shape == y.shape
    assert set(np.unique(predictions)).issubset({0, 1})
    assert fitted.score(X, y) > 0.75


def test_native_importance_works_without_xy(classification_data) -> None:
    """Native importance should work for supported models without X or y."""

    X, y, feature_names = classification_data
    model = get_model("random_forest", n_estimators=40, random_state=0).fit(X, y)

    result = native_importance(model, feature_names=feature_names)

    assert result.method == "native"
    assert result.feature_names == feature_names
    assert result.baseline_score is None
    assert result.importances.shape == (X.shape[1],)
    assert np.all(result.importances >= 0.0)


def test_permutation_and_global_mean_with_external_model(classification_data) -> None:
    """Permutation and global-mean importance should work with raw sklearn models."""

    X, y, feature_names = classification_data
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)

    permutation = permutation_importance(
        model,
        X,
        y,
        feature_names=feature_names,
        n_repeats=5,
        random_state=0,
    )
    global_mean = global_mean_importance(model, X, y, feature_names=feature_names)
    dispatched = analyze(model, method="permutation", X=X, y=y, feature_names=feature_names, n_repeats=5)

    assert permutation.method == "permutation"
    assert permutation.feature_names == feature_names
    assert permutation.baseline_score is not None
    assert permutation.std is not None
    assert permutation.perturbed_scores is not None
    assert permutation.importances.shape == (X.shape[1],)

    assert global_mean.method == "global_mean"
    assert global_mean.baseline_score is not None
    assert global_mean.perturbed_scores is not None
    assert global_mean.importances.shape == (X.shape[1],)

    np.testing.assert_allclose(dispatched.importances, permutation.importances)


def test_explanation_results_save_and_append(workspace_tmp_dir, classification_data) -> None:
    """Explanation results should append cleanly into a results HDF5 file."""

    X, y, feature_names = classification_data
    model = get_model("extra_trees", n_estimators=30, random_state=0).fit(X, y)
    native = native_importance(model, feature_names=feature_names)
    global_mean = global_mean_importance(model, X, y, feature_names=feature_names)

    results_path = workspace_tmp_dir / "explanations.h5"
    first_path = native.save(results_path, experiment_id="benchmark")
    second_path = global_mean.save(results_path, experiment_id="benchmark")

    assert first_path.endswith("/explanation_0000")
    assert second_path.endswith("/explanation_0001")

    with open_h5(results_path, "r") as handle:
        experiment = handle[f"{results_experiment_path('benchmark')}/explanations"]

        assert tuple(sorted(experiment.keys())) == ("explanation_0000", "explanation_0001")
        first = experiment["explanation_0000"]
        second = experiment["explanation_0001"]

        assert first.attrs["method"] == "native"
        assert second.attrs["method"] == "global_mean"
        assert "plot_path" not in first.attrs
        assert read_utf8_array(first, "feature_names") == list(feature_names)
        np.testing.assert_allclose(first["importances"][...], native.importances)
        np.testing.assert_allclose(second["score_deltas"][...], global_mean.score_deltas)
