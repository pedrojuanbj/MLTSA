"""Contact-map feature helpers for MD trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import FeatureComputation
from .distances import all_ligand_protein_distances


def contact_map(
    topology: Any,
    xyz: np.ndarray,
    *,
    ligand_selection: str = "resname LIG",
    protein_selection: str = "protein",
    contact_threshold: float = 0.45,
    water_atom_indices: tuple[int, ...] = (),
) -> FeatureComputation:
    """Compute a binary ligand contact map using an atom-distance threshold."""

    if contact_threshold <= 0.0:
        raise ValueError("contact_threshold must be greater than 0.")

    distances = all_ligand_protein_distances(
        topology,
        xyz,
        ligand_selection=ligand_selection,
        protein_selection=protein_selection,
        water_atom_indices=water_atom_indices,
    )
    contacts = (distances.values <= float(contact_threshold)).astype(np.float64)

    return FeatureComputation(
        feature_type="contact_map",
        values=contacts,
        feature_names=tuple(name.replace("distance:", "contact:", 1) for name in distances.feature_names),
        metadata={
            **distances.metadata,
            "contact_threshold": float(contact_threshold),
        },
    )


__all__ = ["contact_map"]
