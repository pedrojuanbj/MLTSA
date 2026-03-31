"""Explainability, attribution, and diagnostics tools for mltsa."""

from .api import analyze
from .global_mean import global_mean_importance
from .native import native_importance
from .permutation import permutation_importance
from .plotting import plot_importances
from .results import ExplanationResult

__all__ = [
    "ExplanationResult",
    "analyze",
    "global_mean_importance",
    "native_importance",
    "permutation_importance",
    "plot_importances",
]
