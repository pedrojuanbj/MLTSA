"""High-level MD analysis workflow built on stored MD feature sets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from mltsa.explain import ExplanationResult, analyze as analyze_explanation, plot_importances
from mltsa.io.h5 import create_appendable_group, ensure_group, open_h5, replace_dataset, write_utf8_array
from mltsa.io.schema import ensure_results_layout, results_experiment_path
from mltsa.models import get_model
from mltsa.synthetic.base import JSONValue

from .featurize import MDFeatureDataset, load_dataset


@dataclass(slots=True)
class MDAnalysisResult:
    """Result bundle produced by :func:`run_mltsa`."""

    dataset: MDFeatureDataset
    X: np.ndarray
    y: np.ndarray
    analysis_feature_names: tuple[str, ...]
    model: object
    model_name: str
    predictions: np.ndarray
    probabilities: np.ndarray | None
    training_score: float
    explanation: ExplanationResult
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    results_h5_path: Path | None = None
    analysis_path: str | None = None
    explanation_path: str | None = None

    def save(self, path: str | Path, *, experiment_id: str = "md_analysis") -> tuple[str, str]:
        """Append analysis outputs to a results HDF5 file."""

        results_path = Path(path)
        with open_h5(results_path, "a") as handle:
            ensure_results_layout(handle)
            experiment = ensure_group(handle, results_experiment_path(experiment_id), attrs={"workflow": "md_analysis"})
            analysis_group = create_appendable_group(experiment, "analyses", prefix="analysis", width=4)
            analysis_group.attrs["model_name"] = self.model_name
            analysis_group.attrs["feature_set"] = self.dataset.feature_set
            analysis_group.attrs["feature_type"] = self.dataset.feature_type
            analysis_group.attrs["metadata_json"] = json.dumps(self.metadata, sort_keys=True)
            analysis_group.attrs["training_score"] = float(self.training_score)
            analysis_group.attrs["n_samples"] = int(self.X.shape[0])
            analysis_group.attrs["n_features"] = int(self.X.shape[1])

            replace_dataset(analysis_group, "y_true", np.asarray(self.y))
            replace_dataset(analysis_group, "y_pred", np.asarray(self.predictions))
            write_utf8_array(analysis_group, "replica_ids", self.dataset.replica_ids, overwrite=True)
            write_utf8_array(analysis_group, "state_labels", self.dataset.state_labels, overwrite=True)
            write_utf8_array(analysis_group, "feature_names", self.analysis_feature_names, overwrite=True)
            if self.probabilities is not None:
                replace_dataset(analysis_group, "predict_proba", np.asarray(self.probabilities, dtype=np.float64))

            analysis_path = analysis_group.name

        explanation_path = self.explanation.save(results_path, experiment_id=experiment_id)
        self.results_h5_path = results_path
        self.analysis_path = analysis_path
        self.explanation_path = explanation_path
        return analysis_path, explanation_path


def run_mltsa(
    dataset_h5_path: str | Path,
    feature_set: str,
    *,
    model: object | str | None = None,
    model_kwargs: Mapping[str, object] | None = None,
    explanation_method: str = "native",
    explanation_kwargs: Mapping[str, object] | None = None,
    results_h5_path: str | Path | None = None,
    experiment_id: str | None = None,
) -> MDAnalysisResult:
    """Load one MD feature set, train a model, and compute feature importance."""

    dataset = load_dataset(dataset_h5_path, feature_set)
    matrix, flattened_feature_names = _flatten_dataset(dataset)
    estimator = _resolve_model(model, model_kwargs=model_kwargs)
    estimator.fit(matrix, dataset.y)
    predictions = np.asarray(estimator.predict(matrix))

    probabilities: np.ndarray | None
    try:
        probabilities = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    except (AttributeError, NotImplementedError):
        probabilities = None

    if hasattr(estimator, "score"):
        training_score = float(estimator.score(matrix, dataset.y))
    else:
        training_score = float(np.mean(predictions == dataset.y))

    explanation = analyze_explanation(
        estimator,
        method=explanation_method,
        X=matrix,
        y=dataset.y,
        feature_names=flattened_feature_names,
        **dict(explanation_kwargs or {}),
    )

    model_name = getattr(estimator, "model_name", type(estimator).__name__)
    result = MDAnalysisResult(
        dataset=dataset,
        X=matrix,
        y=np.asarray(dataset.y),
        analysis_feature_names=flattened_feature_names,
        model=estimator,
        model_name=str(model_name),
        predictions=predictions,
        probabilities=probabilities,
        training_score=training_score,
        explanation=explanation,
        metadata={
            "dataset_h5_path": str(Path(dataset_h5_path)),
            "feature_set": dataset.feature_set,
            "feature_type": dataset.feature_type,
            "explanation_method": explanation.method,
        },
    )

    if results_h5_path is not None:
        resolved_experiment_id = experiment_id or _default_experiment_id(
            feature_set=dataset.feature_set,
            model_name=str(model_name),
            explanation_method=explanation.method,
        )
        result.save(results_h5_path, experiment_id=resolved_experiment_id)

    return result


def plot_top_features(
    result: MDAnalysisResult | ExplanationResult,
    *,
    top_n: int = 10,
    ax: Any | None = None,
) -> Any:
    """Plot the most important features from an MD analysis result."""

    explanation = result.explanation if isinstance(result, MDAnalysisResult) else result
    return plot_importances(explanation, top_n=top_n, ax=ax)


def plot_feature_traces(
    result: MDAnalysisResult | MDFeatureDataset,
    *,
    feature_indices: list[int] | tuple[int, ...] | None = None,
    top_n: int = 3,
    ax: Any | None = None,
) -> Any:
    """Plot mean feature traces by state for selected base MD features."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for MD plotting helpers.") from exc

    dataset = result.dataset if isinstance(result, MDAnalysisResult) else result
    selected = _select_trace_features(result, dataset=dataset, feature_indices=feature_indices, top_n=top_n)
    if not selected:
        raise ValueError("No feature indices were selected for plotting.")

    axes: list[Any]
    if ax is None:
        figure, created_axes = plt.subplots(len(selected), 1, figsize=(7, 3 * len(selected)), squeeze=False)
        axes = created_axes[:, 0].tolist()
        figure.tight_layout()
    else:
        axes = [ax]
        if len(selected) > 1:
            raise ValueError("A single Axes object can only be used for one feature trace.")

    x_values = np.arange(dataset.n_frames)
    state_colors = {"IN": "#4c956c", "OUT": "#bc4749", "TS": "#f4a259"}

    for axis, feature_index in zip(axes, selected):
        for state_label in ("IN", "OUT", "TS"):
            mask = np.asarray([label == state_label for label in dataset.state_labels], dtype=bool)
            if not mask.any():
                continue
            mean_trace = dataset.X[mask, :, feature_index].mean(axis=0)
            axis.plot(x_values, mean_trace, label=state_label, color=state_colors[state_label])
        axis.set_title(dataset.feature_names[feature_index])
        axis.set_xlabel("frame")
        axis.set_ylabel("feature value")
        axis.legend()

    return axes if len(axes) > 1 else axes[0]


def _flatten_dataset(dataset: MDFeatureDataset) -> tuple[np.ndarray, tuple[str, ...]]:
    """Flatten one MD feature set into a 2D matrix suitable for model fitting."""

    matrix = np.asarray(dataset.X, dtype=np.float64).reshape(dataset.n_replicas, dataset.n_frames * dataset.n_features)
    feature_names = tuple(
        f"{name}@t{frame_index:03d}"
        for frame_index in range(dataset.n_frames)
        for name in dataset.feature_names
    )
    return matrix, feature_names


def _resolve_model(model: object | str | None, *, model_kwargs: Mapping[str, object] | None) -> object:
    """Resolve a user-provided model object or a named mltsa model."""

    kwargs = dict(model_kwargs or {})
    if model is None:
        return get_model("random_forest", **kwargs)
    if isinstance(model, str):
        return get_model(model, **kwargs)
    if kwargs:
        raise ValueError("model_kwargs can only be used when model is provided by name or omitted.")
    return model


def _select_trace_features(
    result: MDAnalysisResult | MDFeatureDataset,
    *,
    dataset: MDFeatureDataset,
    feature_indices: list[int] | tuple[int, ...] | None,
    top_n: int,
) -> list[int]:
    """Resolve which base feature traces should be plotted."""

    if feature_indices is not None:
        return [int(index) for index in feature_indices]

    if isinstance(result, MDAnalysisResult):
        importances = np.asarray(result.explanation.importances, dtype=np.float64)
        if importances.size == dataset.n_frames * dataset.n_features:
            aggregated = importances.reshape(dataset.n_frames, dataset.n_features).mean(axis=0)
            return np.argsort(aggregated)[::-1][:top_n].astype(int).tolist()

    return list(range(min(top_n, dataset.n_features)))


def _default_experiment_id(*, feature_set: str, model_name: str, explanation_method: str) -> str:
    """Build a stable default experiment identifier for saved results."""

    parts = [feature_set, model_name, explanation_method]
    return "_".join(
        str(part).strip().lower().replace("-", "_").replace(" ", "_")
        for part in parts
        if str(part).strip()
    )


__all__ = ["MDAnalysisResult", "plot_feature_traces", "plot_top_features", "run_mltsa"]
