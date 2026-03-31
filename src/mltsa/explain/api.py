"""Public dispatcher for explainability methods."""

from __future__ import annotations

from .global_mean import global_mean_importance
from .native import native_importance
from .permutation import permutation_importance
from .results import ExplanationResult


def analyze(
    model: object,
    *,
    method: str = "native",
    X: object | None = None,
    y: object | None = None,
    feature_names: list[str] | tuple[str, ...] | None = None,
    **kwargs: object,
) -> ExplanationResult:
    """Dispatch to a supported explainability method."""

    normalized = method.strip().lower().replace("-", "_").replace(" ", "_")

    if normalized == "native":
        return native_importance(model, feature_names=feature_names)
    if normalized == "permutation":
        if X is None or y is None:
            raise ValueError("permutation analysis requires both X and y.")
        return permutation_importance(model, X, y, feature_names=feature_names, **kwargs)
    if normalized in {"global_mean", "globalmean"}:
        if X is None or y is None:
            raise ValueError("global mean analysis requires both X and y.")
        return global_mean_importance(model, X, y, feature_names=feature_names, **kwargs)

    raise ValueError(f"Unsupported analysis method {method!r}.")


__all__ = ["analyze"]
