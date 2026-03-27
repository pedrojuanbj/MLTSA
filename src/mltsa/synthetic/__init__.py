"""Synthetic data generators and benchmark helpers for mltsa."""

from .dataset import SyntheticDataset, load_dataset, make_1d_dataset, make_2d_dataset
from .plotting import plot_example_trajectories, plot_ground_truth_relevance, plot_relevance_over_time

__all__ = [
    "SyntheticDataset",
    "load_dataset",
    "make_1d_dataset",
    "make_2d_dataset",
    "plot_example_trajectories",
    "plot_ground_truth_relevance",
    "plot_relevance_over_time",
]
