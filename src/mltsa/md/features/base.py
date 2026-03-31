"""Shared types and helpers for MD feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from mltsa.synthetic.base import JSONValue, as_float_array, as_int_array, clone_json

from .rules import WATER_RESIDUE_NAMES

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

FEATURE_SET_SCHEMA = "mltsa.md.features"
FEATURE_SET_SCHEMA_VERSION = 1
STATE_ORDER = ("IN", "OUT", "TS")
STATE_TO_INT = {label: index for index, label in enumerate(STATE_ORDER)}


@dataclass(frozen=True, slots=True)
class FeatureComputation:
    """Feature values computed for one trajectory."""

    feature_type: str
    values: FloatArray
    feature_names: tuple[str, ...]
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize computed arrays and validate feature dimensions."""

        object.__setattr__(self, "values", as_float_array(self.values))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "metadata", clone_json(self.metadata))

        if self.values.ndim != 2:
            raise ValueError("Feature values must have shape (n_frames, n_features).")
        if self.values.shape[1] != len(self.feature_names):
            raise ValueError("feature_names length must match the feature dimension of values.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique within one feature computation.")


@dataclass(slots=True)
class MDFeatureDataset:
    """Loaded MD feature dataset for one selected feature set."""

    feature_set: str
    feature_type: str
    X: FloatArray
    y: IntArray
    state_labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    replica_ids: tuple[str, ...]
    trajectory_paths: tuple[str, ...]
    topology_paths: tuple[str, ...]
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    processed_replica_ids: tuple[str, ...] = ()
    skipped_replica_ids: tuple[str, ...] = ()
    overwritten_replica_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize arrays and validate dataset consistency."""

        self.X = as_float_array(self.X)
        self.y = as_int_array(self.y)
        self.state_labels = tuple(self.state_labels)
        self.feature_names = tuple(self.feature_names)
        self.replica_ids = tuple(self.replica_ids)
        self.trajectory_paths = tuple(self.trajectory_paths)
        self.topology_paths = tuple(self.topology_paths)
        self.metadata = clone_json(self.metadata)
        self.processed_replica_ids = tuple(self.processed_replica_ids)
        self.skipped_replica_ids = tuple(self.skipped_replica_ids)
        self.overwritten_replica_ids = tuple(self.overwritten_replica_ids)

        if self.X.ndim != 3:
            raise ValueError("X must have shape (n_replicas, n_frames, n_features).")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("y length must match the number of replicas in X.")
        if self.X.shape[0] != len(self.state_labels):
            raise ValueError("state_labels length must match the number of replicas in X.")
        if self.X.shape[0] != len(self.replica_ids):
            raise ValueError("replica_ids length must match the number of replicas in X.")
        if self.X.shape[0] != len(self.trajectory_paths):
            raise ValueError("trajectory_paths length must match the number of replicas in X.")
        if self.X.shape[0] != len(self.topology_paths):
            raise ValueError("topology_paths length must match the number of replicas in X.")
        if self.X.shape[2] != len(self.feature_names):
            raise ValueError("feature_names length must match the feature dimension of X.")

    @property
    def n_replicas(self) -> int:
        """Number of stored MD replicas."""

        return int(self.X.shape[0])

    @property
    def n_frames(self) -> int:
        """Number of time steps per replica."""

        return int(self.X.shape[1])

    @property
    def n_features(self) -> int:
        """Number of features per time step."""

        return int(self.X.shape[2])


def normalize_feature_type(value: str) -> str:
    """Canonicalize a user-provided feature-family name."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    supported = {
        "closest_residue_distances",
        "all_ligand_protein_distances",
        "bubble_distances",
        "contact_map",
        "pca_xyz",
    }
    if normalized not in supported:
        raise ValueError(f"Unsupported MD feature type {value!r}.")
    return normalized


def validate_replica_id(value: str) -> str:
    """Reject empty or nested replica identifiers."""

    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("replica ids must not be empty.")
    if "/" in cleaned:
        raise ValueError("replica ids must not contain '/'.")
    return cleaned


def normalize_path(path: str | Path) -> str:
    """Normalize a filesystem path for metadata identity checks."""

    return Path(path).resolve(strict=False).as_posix().casefold()


def state_to_int(label: str) -> int:
    """Map a state label to a stable integer code."""

    normalized = str(label).strip().upper()
    if normalized not in STATE_TO_INT:
        raise ValueError(f"Unsupported MD state label {label!r}.")
    return int(STATE_TO_INT[normalized])


def select_atom_indices(topology: Any, selection: str) -> IntArray:
    """Resolve a topology selection string into atom indices."""

    indices = np.asarray(topology.select(selection), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"Selection {selection!r} did not resolve to any atoms.")
    return indices


def non_water_atom_indices(topology: Any) -> IntArray:
    """Return all atom indices that do not belong to water residues."""

    indices = [
        int(atom.index)
        for atom in topology.atoms
        if not is_water_residue(atom.residue)
    ]
    if not indices:
        raise ValueError("Topology does not contain any non-water atoms.")
    return np.asarray(indices, dtype=np.int64)


def unique_indices(*arrays: np.ndarray | tuple[int, ...] | list[int]) -> IntArray:
    """Merge one or more index collections into a sorted unique array."""

    merged: list[int] = []
    for values in arrays:
        merged.extend(int(value) for value in np.asarray(values, dtype=np.int64).tolist())
    if not merged:
        return np.empty(0, dtype=np.int64)
    return np.asarray(sorted(set(merged)), dtype=np.int64)


def without_indices(indices: np.ndarray, excluded: np.ndarray) -> IntArray:
    """Return ``indices`` with any ``excluded`` values removed."""

    if indices.size == 0:
        return np.empty(0, dtype=np.int64)
    if excluded.size == 0:
        return np.asarray(indices, dtype=np.int64)
    mask = ~np.isin(indices, excluded)
    return np.asarray(indices[mask], dtype=np.int64)


def group_atom_indices_by_residue(topology: Any, atom_indices: np.ndarray) -> list[tuple[Any, IntArray]]:
    """Group selected atom indices by residue while preserving topology order."""

    groups: dict[int, tuple[Any, list[int]]] = {}
    order: list[int] = []
    selected = {int(index) for index in np.asarray(atom_indices, dtype=np.int64).tolist()}

    for atom in topology.atoms:
        atom_index = int(atom.index)
        if atom_index not in selected:
            continue
        residue = atom.residue
        key = id(residue)
        if key not in groups:
            groups[key] = (residue, [])
            order.append(key)
        groups[key][1].append(atom_index)

    return [
        (groups[key][0], np.asarray(groups[key][1], dtype=np.int64))
        for key in order
    ]


def pairwise_distances(xyz: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> FloatArray:
    """Compute frame-wise distances for all left/right atom-index combinations."""

    left = np.asarray(left_indices, dtype=np.int64)
    right = np.asarray(right_indices, dtype=np.int64)
    if left.size == 0 or right.size == 0:
        raise ValueError("Distance features require at least one atom on both sides.")

    left_xyz = np.asarray(xyz[:, left, :], dtype=np.float64)[:, :, None, :]
    right_xyz = np.asarray(xyz[:, right, :], dtype=np.float64)[:, None, :, :]
    return np.linalg.norm(left_xyz - right_xyz, axis=3).reshape(int(xyz.shape[0]), left.size * right.size)


def min_group_distance_series(xyz: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> FloatArray:
    """Compute the minimum distance between two atom groups per frame."""

    left = np.asarray(left_indices, dtype=np.int64)
    right = np.asarray(right_indices, dtype=np.int64)
    if left.size == 0 or right.size == 0:
        raise ValueError("Minimum group distances require non-empty atom groups.")

    left_xyz = np.asarray(xyz[:, left, :], dtype=np.float64)[:, :, None, :]
    right_xyz = np.asarray(xyz[:, right, :], dtype=np.float64)[:, None, :, :]
    return np.linalg.norm(left_xyz - right_xyz, axis=3).min(axis=(1, 2))


def atom_label(atom: Any) -> str:
    """Build a readable stable atom label for feature names."""

    residue = atom.residue
    return f"{residue_label(residue)}:{atom_name(atom)}{int(atom.index):03d}"


def residue_label(residue: Any) -> str:
    """Build a readable stable residue label for feature names."""

    return f"{residue_name(residue)}{residue_id(residue):03d}"


def atom_name(atom: Any) -> str:
    """Return a normalized atom name."""

    return str(getattr(atom, "name", "X")).strip() or "X"


def residue_name(residue: Any) -> str:
    """Return a normalized residue name."""

    return str(getattr(residue, "name", "UNK")).strip() or "UNK"


def residue_id(residue: Any) -> int:
    """Return a stable residue identifier when available."""

    if hasattr(residue, "resSeq"):
        return int(residue.resSeq)
    if hasattr(residue, "index"):
        return int(residue.index)
    return 0


def is_water_residue(residue: Any) -> bool:
    """Return ``True`` when a residue looks like water."""

    if bool(getattr(residue, "is_water", False)):
        return True
    return residue_name(residue).upper() in WATER_RESIDUE_NAMES


__all__ = [
    "FEATURE_SET_SCHEMA",
    "FEATURE_SET_SCHEMA_VERSION",
    "FeatureComputation",
    "FloatArray",
    "IntArray",
    "MDFeatureDataset",
    "STATE_ORDER",
    "STATE_TO_INT",
    "atom_label",
    "atom_name",
    "group_atom_indices_by_residue",
    "is_water_residue",
    "min_group_distance_series",
    "non_water_atom_indices",
    "normalize_feature_type",
    "normalize_path",
    "pairwise_distances",
    "residue_id",
    "residue_label",
    "residue_name",
    "select_atom_indices",
    "state_to_int",
    "unique_indices",
    "validate_replica_id",
    "without_indices",
]
