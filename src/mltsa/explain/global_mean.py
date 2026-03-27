"""Global-mean feature perturbation importance."""

from __future__ import annotations

import numpy as np

from .results import (
    ExplanationResult,
    coerce_feature_names,
    ensure_feature_matrix,
    ensure_target_vector,
    infer_model_name,
    score_estimator,
    scoring_name,
)


def global_mean_importance(
    model: object,
    X: object,
    y: object,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
    scoring: str | None = None,
) -> ExplanationResult:
    """Measure importance by replacing each feature with its global mean."""

    matrix = ensure_feature_matrix(X)
    targets = ensure_target_vector(y, matrix.shape[0])
    baseline = score_estimator(model, matrix, targets, scoring)
    feature_means = matrix.mean(axis=0)

    perturbed_scores = np.empty(matrix.shape[1], dtype=np.float64)
    for feature_index, feature_mean in enumerate(feature_means):
        perturbed = matrix.copy()
        perturbed[:, feature_index] = feature_mean
        perturbed_scores[feature_index] = score_estimator(model, perturbed, targets, scoring)

    importances = baseline - perturbed_scores
    names = coerce_feature_names(feature_names, int(importances.shape[0]))
    return ExplanationResult(
        method="global_mean",
        importances=importances,
        feature_names=names,
        baseline_score=baseline,
        perturbed_scores=perturbed_scores,
        score_deltas=importances.copy(),
        model_name=infer_model_name(model),
        metadata={"scoring": scoring_name(scoring), "replacement": "global_mean"},
    )


__all__ = ["global_mean_importance"]
