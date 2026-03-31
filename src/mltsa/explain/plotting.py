"""Plotting helpers for explanation results."""

from __future__ import annotations

from typing import Any

import numpy as np

from .results import ExplanationResult


def plot_importances(
    result: ExplanationResult,
    *,
    top_n: int | None = None,
    sort: bool = True,
    ax: Any | None = None,
) -> Any:
    """Plot feature importances for an explanation result."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for explanation plotting helpers.") from exc

    axis = ax if ax is not None else plt.subplots()[1]
    indices = np.arange(result.n_features)
    if sort:
        indices = result.ranked_indices
    if top_n is not None:
        indices = indices[:top_n]

    labels = [result.feature_names[index] for index in indices]
    values = result.importances[indices]
    axis.bar(np.arange(len(indices)), values)
    axis.set_xticks(np.arange(len(indices)), labels, rotation=45, ha="right")
    axis.set_ylabel("importance")
    axis.set_title(f"{result.method.replace('_', ' ').title()} Importance")
    return axis


__all__ = ["plot_importances"]
