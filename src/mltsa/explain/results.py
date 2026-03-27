"""Shared result type and persistence for explainability analyses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from sklearn.metrics import get_scorer

from mltsa.io.h5 import create_appendable_group, ensure_group, open_h5, replace_dataset, write_utf8_array
from mltsa.io.schema import ensure_results_layout, results_experiment_path
from mltsa.synthetic.base import JSONValue, as_float_array


@dataclass(slots=True)
class ExplanationResult:
    """Common result object returned by all explainability methods."""

    method: str
    importances: np.ndarray
    feature_names: tuple[str, ...]
    baseline_score: float | None = None
    perturbed_scores: np.ndarray | None = None
    score_deltas: np.ndarray | None = None
    std: np.ndarray | None = None
    model_name: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize result arrays and validate feature-level shapes."""

        self.importances = as_float_array(self.importances)
        self.feature_names = tuple(self.feature_names)
        self.metadata = dict(self.metadata)

        if self.importances.ndim != 1:
            raise ValueError("importances must be one-dimensional.")
        if len(self.feature_names) != self.importances.shape[0]:
            raise ValueError("feature_names length must match importances length.")

        if self.perturbed_scores is not None:
            self.perturbed_scores = as_float_array(self.perturbed_scores)
            if self.perturbed_scores.shape != self.importances.shape:
                raise ValueError("perturbed_scores must match importances shape.")

        if self.score_deltas is not None:
            self.score_deltas = as_float_array(self.score_deltas)
            if self.score_deltas.shape != self.importances.shape:
                raise ValueError("score_deltas must match importances shape.")

        if self.std is not None:
            self.std = as_float_array(self.std)
            if self.std.shape != self.importances.shape:
                raise ValueError("std must match importances shape.")

        if self.baseline_score is not None:
            self.baseline_score = float(self.baseline_score)

    @property
    def n_features(self) -> int:
        """Return the number of analyzed features."""

        return int(self.importances.shape[0])

    @property
    def ranked_indices(self) -> np.ndarray:
        """Return feature indices sorted by descending importance."""

        return np.argsort(self.importances)[::-1]

    def save(self, path: str | Path, *, experiment_id: str = "default") -> str:
        """Append this explanation result to an HDF5 results file."""

        file_path = Path(path)
        with open_h5(file_path, "a") as handle:
            ensure_results_layout(handle)
            experiment = ensure_group(handle, results_experiment_path(experiment_id), attrs={"kind": "explain"})
            explanation_group = create_appendable_group(experiment, "explanations", prefix="explanation", width=4)
            explanation_group.attrs["method"] = self.method
            explanation_group.attrs["model_name"] = self.model_name or ""
            explanation_group.attrs["metadata_json"] = json.dumps(self.metadata, sort_keys=True)
            if self.baseline_score is not None:
                explanation_group.attrs["baseline_score"] = self.baseline_score

            replace_dataset(explanation_group, "importances", self.importances)
            write_utf8_array(explanation_group, "feature_names", self.feature_names, overwrite=True)

            if self.perturbed_scores is not None:
                replace_dataset(explanation_group, "perturbed_scores", self.perturbed_scores)
            if self.score_deltas is not None:
                replace_dataset(explanation_group, "score_deltas", self.score_deltas)
            if self.std is not None:
                replace_dataset(explanation_group, "std", self.std)

            return explanation_group.name


def coerce_feature_names(feature_names: list[str] | tuple[str, ...] | None, n_features: int) -> tuple[str, ...]:
    """Return user-provided feature names or generate fallback names."""

    if feature_names is None:
        return tuple(f"feature_{index:03d}" for index in range(n_features))

    normalized = tuple(str(name) for name in feature_names)
    if len(normalized) != n_features:
        raise ValueError("feature_names length must match the number of features.")
    return normalized


def resolve_estimator(model: Any) -> Any:
    """Unwrap mltsa model wrappers while accepting raw sklearn estimators."""

    if hasattr(model, "model_name") and hasattr(model, "estimator"):
        return model.estimator
    return model


def infer_model_name(model: Any) -> str:
    """Infer a user-facing model name from a wrapper or sklearn estimator."""

    if hasattr(model, "model_name"):
        return str(model.model_name)
    estimator = resolve_estimator(model)
    return type(estimator).__name__


def ensure_feature_matrix(X: Any) -> np.ndarray:
    """Convert input data to a 2D float feature matrix."""

    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features).")
    return matrix


def ensure_target_vector(y: Any, n_samples: int) -> np.ndarray:
    """Convert targets to a 1D array matching the sample count."""

    targets = np.asarray(y)
    if targets.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if targets.shape[0] != n_samples:
        raise ValueError("y length must match the number of samples in X.")
    return targets


def score_estimator(model: Any, X: np.ndarray, y: np.ndarray, scoring: str | Callable[[Any, np.ndarray, np.ndarray], float] | None) -> float:
    """Score an estimator using its own score method or a custom scoring rule."""

    estimator = resolve_estimator(model)
    if scoring is None:
        return float(estimator.score(X, y))
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
        return float(scorer(estimator, X, y))
    return float(scoring(estimator, X, y))


def scoring_name(scoring: str | Callable[[Any, np.ndarray, np.ndarray], float] | None) -> str:
    """Return a stable scoring description for result metadata."""

    if scoring is None:
        return "estimator.score"
    if isinstance(scoring, str):
        return scoring
    return getattr(scoring, "__name__", "callable")


__all__ = [
    "ExplanationResult",
    "coerce_feature_names",
    "ensure_feature_matrix",
    "ensure_target_vector",
    "infer_model_name",
    "resolve_estimator",
    "score_estimator",
    "scoring_name",
]
