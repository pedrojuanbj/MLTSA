"""Simple plotting helpers for synthetic datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .dataset import SyntheticDataset


def plot_ground_truth_relevance(dataset: SyntheticDataset, *, ax: Any | None = None) -> Any:
    """Plot the aggregate ground-truth feature relevance."""

    plt = _get_pyplot()
    axis = ax if ax is not None else plt.subplots()[1]

    relevance = dataset.ground_truth_relevance
    axis.bar(np.arange(dataset.n_features), relevance)
    axis.set_title("Ground Truth Relevance")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Relevance")
    axis.set_xticks(np.arange(dataset.n_features), dataset.feature_names, rotation=45, ha="right")
    return axis


def plot_example_trajectories(
    dataset: SyntheticDataset,
    *,
    trajectory_indices: Sequence[int] = (0,),
    feature_indices: Sequence[int] = (0,),
    ax: Any | None = None,
) -> Any:
    """Plot example latent or observed trajectories from a synthetic dataset."""

    plt = _get_pyplot()
    axis = ax if ax is not None else plt.subplots()[1]
    indices = [int(index) for index in trajectory_indices]

    if dataset.latent_trajectories is not None and dataset.latent_trajectories.shape[2] == 2:
        for index in indices:
            latent = dataset.latent_trajectories[index]
            axis.plot(latent[:, 0], latent[:, 1], label=f"traj {index}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title("Example Latent Trajectories")
    else:
        feature_id = int(feature_indices[0])
        time = np.arange(dataset.n_steps)
        for index in indices:
            axis.plot(time, dataset.X[index, :, feature_id], label=f"traj {index}")
        axis.set_xlabel("time step")
        axis.set_ylabel(dataset.feature_names[feature_id])
        axis.set_title("Example Feature Trajectories")

    axis.legend(loc="best")
    return axis


def plot_relevance_over_time(dataset: SyntheticDataset, *, ax: Any | None = None) -> Any:
    """Plot time-dependent ground-truth relevance as a heatmap."""

    if dataset.time_relevance is None:
        raise ValueError("This dataset does not provide time-dependent relevance.")

    plt = _get_pyplot()
    figure = None
    if ax is None:
        figure, axis = plt.subplots()
    else:
        axis = ax

    image = axis.imshow(dataset.time_relevance.T, aspect="auto", origin="lower")
    axis.set_title("Ground Truth Relevance Over Time")
    axis.set_xlabel("time step")
    axis.set_ylabel("feature")
    axis.set_yticks(np.arange(dataset.n_features), dataset.feature_names)

    if figure is None:
        figure = axis.figure
    figure.colorbar(image, ax=axis, label="relevance")
    return axis


def _get_pyplot() -> Any:
    """Import matplotlib lazily so plotting stays optional."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for synthetic plotting helpers.") from exc
    return plt


__all__ = [
    "plot_example_trajectories",
    "plot_ground_truth_relevance",
    "plot_relevance_over_time",
]
