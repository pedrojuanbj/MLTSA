"""PCA-based coordinate features for MD trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import FeatureComputation, non_water_atom_indices, select_atom_indices


def pca_xyz(
    topology: Any,
    xyz: np.ndarray,
    *,
    atom_selection: str | None = None,
    n_components: int = 3,
) -> FeatureComputation:
    """Project selected Cartesian coordinates onto principal components."""

    if n_components < 1:
        raise ValueError("n_components must be at least 1.")

    if atom_selection is None:
        atom_indices = non_water_atom_indices(topology)
        selection_label = "non_water"
    else:
        atom_indices = select_atom_indices(topology, atom_selection)
        selection_label = atom_selection

    matrix = np.asarray(xyz[:, atom_indices, :], dtype=np.float64).reshape(int(xyz.shape[0]), -1)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, right_t = np.linalg.svd(centered, full_matrices=False)

    component_count = min(int(n_components), int(right_t.shape[0]))
    components = centered @ right_t[:component_count].T

    total_variance = float(np.square(singular_values).sum())
    if total_variance > 0.0:
        explained_variance_ratio = (np.square(singular_values[:component_count]) / total_variance).tolist()
    else:
        explained_variance_ratio = [0.0] * component_count

    return FeatureComputation(
        feature_type="pca_xyz",
        values=components,
        feature_names=tuple(f"pc_{index:03d}" for index in range(component_count)),
        metadata={
            "atom_selection": selection_label,
            "atom_indices": [int(index) for index in atom_indices.tolist()],
            "explained_variance_ratio": [float(value) for value in explained_variance_ratio],
        },
    )


__all__ = ["pca_xyz"]
