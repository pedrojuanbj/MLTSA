"""Native feature importance helpers."""

from __future__ import annotations

import numpy as np

from .results import ExplanationResult, coerce_feature_names, infer_model_name, resolve_estimator


def native_importance(model: object, *, feature_names: list[str] | tuple[str, ...] | None = None) -> ExplanationResult:
    """Extract native feature importance from a fitted model when available."""

    estimator = resolve_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=np.float64)
        source = "feature_importances_"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=np.float64)
        importances = np.abs(coefficients)
        if importances.ndim > 1:
            importances = importances.mean(axis=0)
        source = "coef_"
    else:
        raise ValueError("The provided model does not expose native feature importances.")

    names = coerce_feature_names(feature_names, int(importances.shape[0]))
    return ExplanationResult(
        method="native",
        importances=importances,
        feature_names=names,
        score_deltas=importances.copy(),
        model_name=infer_model_name(model),
        metadata={"source": source},
    )


__all__ = ["native_importance"]
