"""Permutation importance helpers."""

from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance as sklearn_permutation_importance

from .results import (
    ExplanationResult,
    coerce_feature_names,
    ensure_feature_matrix,
    ensure_target_vector,
    infer_model_name,
    resolve_estimator,
    score_estimator,
    scoring_name,
)


def permutation_importance(
    model: object,
    X: object,
    y: object,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> ExplanationResult:
    """Compute permutation importance for a fitted model."""

    matrix = ensure_feature_matrix(X)
    targets = ensure_target_vector(y, matrix.shape[0])
    estimator = resolve_estimator(model)
    baseline = score_estimator(estimator, matrix, targets, scoring)
    permutation = sklearn_permutation_importance(
        estimator,
        matrix,
        targets,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importances = np.asarray(permutation.importances_mean, dtype=np.float64)
    std = np.asarray(permutation.importances_std, dtype=np.float64)
    perturbed_scores = baseline - importances
    names = coerce_feature_names(feature_names, int(importances.shape[0]))

    return ExplanationResult(
        method="permutation",
        importances=importances,
        feature_names=names,
        baseline_score=baseline,
        perturbed_scores=perturbed_scores,
        score_deltas=importances.copy(),
        std=std,
        model_name=infer_model_name(model),
        metadata={
            "scoring": scoring_name(scoring),
            "n_repeats": int(n_repeats),
            "random_state": int(random_state),
        },
    )


__all__ = ["permutation_importance"]
