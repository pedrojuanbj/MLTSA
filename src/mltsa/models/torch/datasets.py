"""Dataset helpers for PyTorch time-series models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def ensure_feature_array(X: Any) -> np.ndarray:
    """Normalize feature input to ``(n_samples, n_steps, n_features)``."""

    features = np.asarray(X, dtype=np.float32)
    if features.ndim == 2:
        features = features[:, None, :]
    elif features.ndim != 3:
        raise ValueError("X must have shape (n_samples, n_features) or (n_samples, n_steps, n_features).")

    if features.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    return np.ascontiguousarray(features, dtype=np.float32)


def encode_targets(y: Any, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique classes and encoded integer targets."""

    targets = np.asarray(y)
    if targets.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if targets.shape[0] != n_samples:
        raise ValueError("y length must match the number of samples in X.")

    classes, encoded = np.unique(targets, return_inverse=True)
    if classes.size < 2:
        raise ValueError("At least two classes are required for classification.")
    return classes, encoded.astype(np.int64, copy=False)


class TimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Minimal dataset backed by in-memory numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.features = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.targets = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        """Return the sample count."""

        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single feature-target pair."""

        return self.features[index], self.targets[index]


__all__ = ["TimeSeriesDataset", "encode_targets", "ensure_feature_array"]
