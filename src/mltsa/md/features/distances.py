"""Distance-based MD feature families."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import (
    FeatureComputation,
    atom_label,
    group_atom_indices_by_residue,
    min_group_distance_series,
    pairwise_distances,
    residue_label,
    select_atom_indices,
    unique_indices,
    without_indices,
)


def closest_residue_distances(
    topology: Any,
    xyz: np.ndarray,
    *,
    ligand_selection: str = "resname LIG",
    protein_selection: str = "protein",
    water_atom_indices: tuple[int, ...] = (),
) -> FeatureComputation:
    """Compute the closest ligand-to-residue distance for each selected residue."""

    ligand_indices = select_atom_indices(topology, ligand_selection)
    protein_indices = select_atom_indices(topology, protein_selection)
    residue_groups = group_atom_indices_by_residue(topology, protein_indices)

    water_indices = np.asarray(water_atom_indices, dtype=np.int64)
    water_group_count = 0
    if water_indices.size:
        water_groups = group_atom_indices_by_residue(topology, water_indices)
        water_group_count = len(water_groups)
        residue_groups.extend(water_groups)

    if not residue_groups:
        raise ValueError("closest_residue_distances did not find any partner residues.")

    values = np.empty((int(xyz.shape[0]), len(residue_groups)), dtype=np.float64)
    feature_names: list[str] = []

    for column, (residue, residue_indices) in enumerate(residue_groups):
        values[:, column] = min_group_distance_series(xyz, ligand_indices, residue_indices)
        feature_names.append(f"closest_residue:{residue_label(residue)}")

    return FeatureComputation(
        feature_type="closest_residue_distances",
        values=values,
        feature_names=tuple(feature_names),
        metadata={
            "ligand_selection": ligand_selection,
            "protein_selection": protein_selection,
            "residue_count": len(residue_groups),
            "water_feature_count": water_group_count,
        },
    )


def all_ligand_protein_distances(
    topology: Any,
    xyz: np.ndarray,
    *,
    ligand_selection: str = "resname LIG",
    protein_selection: str = "protein",
    water_atom_indices: tuple[int, ...] = (),
) -> FeatureComputation:
    """Compute all ligand-to-partner atom distances across the trajectory."""

    ligand_indices = select_atom_indices(topology, ligand_selection)
    partner_indices = _resolve_partner_indices(
        topology=topology,
        ligand_indices=ligand_indices,
        protein_selection=protein_selection,
        water_atom_indices=water_atom_indices,
    )
    values = pairwise_distances(xyz, ligand_indices, partner_indices)

    return FeatureComputation(
        feature_type="all_ligand_protein_distances",
        values=values,
        feature_names=_pair_feature_names(topology, ligand_indices, partner_indices, prefix="distance"),
        metadata={
            "ligand_selection": ligand_selection,
            "protein_selection": protein_selection,
            "ligand_atom_count": int(ligand_indices.size),
            "partner_atom_count": int(partner_indices.size),
            "water_atom_count": int(np.asarray(water_atom_indices, dtype=np.int64).size),
        },
    )


def bubble_distances(
    topology: Any,
    xyz: np.ndarray,
    *,
    ligand_selection: str = "resname LIG",
    protein_selection: str = "protein",
    bubble_cutoff: float = 0.6,
    water_atom_indices: tuple[int, ...] = (),
) -> FeatureComputation:
    """Compute ligand distances to nearby atoms in a first-frame distance bubble."""

    if bubble_cutoff <= 0.0:
        raise ValueError("bubble_cutoff must be greater than 0.")

    ligand_indices = select_atom_indices(topology, ligand_selection)
    protein_indices = without_indices(select_atom_indices(topology, protein_selection), ligand_indices)
    reference = np.asarray(xyz[0], dtype=np.float64)

    bubble_partners: list[int] = []
    for atom_index in protein_indices.tolist():
        deltas = reference[ligand_indices, :] - reference[int(atom_index), :]
        if float(np.linalg.norm(deltas, axis=1).min()) <= float(bubble_cutoff):
            bubble_partners.append(int(atom_index))

    partner_indices = unique_indices(
        np.asarray(bubble_partners, dtype=np.int64),
        np.asarray(water_atom_indices, dtype=np.int64),
    )
    partner_indices = without_indices(partner_indices, ligand_indices)
    if partner_indices.size == 0:
        raise ValueError("bubble_distances did not find any nearby partner atoms.")

    values = pairwise_distances(xyz, ligand_indices, partner_indices)
    return FeatureComputation(
        feature_type="bubble_distances",
        values=values,
        feature_names=_pair_feature_names(topology, ligand_indices, partner_indices, prefix="bubble"),
        metadata={
            "ligand_selection": ligand_selection,
            "protein_selection": protein_selection,
            "bubble_cutoff": float(bubble_cutoff),
            "ligand_atom_count": int(ligand_indices.size),
            "partner_atom_count": int(partner_indices.size),
            "water_atom_count": int(np.asarray(water_atom_indices, dtype=np.int64).size),
        },
    )


def _resolve_partner_indices(
    *,
    topology: Any,
    ligand_indices: np.ndarray,
    protein_selection: str,
    water_atom_indices: tuple[int, ...],
) -> np.ndarray:
    """Resolve the partner atom indices for pairwise feature families."""

    protein_indices = select_atom_indices(topology, protein_selection)
    merged = unique_indices(protein_indices, np.asarray(water_atom_indices, dtype=np.int64))
    partner_indices = without_indices(merged, ligand_indices)
    if partner_indices.size == 0:
        raise ValueError("No partner atoms were selected for the requested feature family.")
    return partner_indices


def _pair_feature_names(
    topology: Any,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    prefix: str,
) -> tuple[str, ...]:
    """Build stable feature names for all left/right atom pairs."""

    names: list[str] = []
    for left_index in np.asarray(left_indices, dtype=np.int64).tolist():
        left_atom = topology.atoms[int(left_index)]
        for right_index in np.asarray(right_indices, dtype=np.int64).tolist():
            right_atom = topology.atoms[int(right_index)]
            names.append(f"{prefix}:{atom_label(left_atom)}__{atom_label(right_atom)}")
    return tuple(names)


__all__ = [
    "all_ligand_protein_distances",
    "bubble_distances",
    "closest_residue_distances",
]
