"""Rule evaluation helpers for MD labeling workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SelectionPair = tuple[str, str]

WATER_RESIDUE_NAMES = {"HOH", "WAT", "SOL", "TIP3", "TIP3P", "TIP4", "TIP4P"}


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    """Per-frame values produced by a labeling rule on a final trajectory window."""

    rule: str
    frame_values: np.ndarray
    component_values: np.ndarray
    component_labels: tuple[str, ...]
    reference_atom_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WaterMetadata:
    """Nearby-water summary for a trajectory window."""

    atom_indices: tuple[int, ...]
    residue_ids: tuple[int, ...]
    residue_names: tuple[str, ...]
    mean_distances: np.ndarray


def evaluate_rule(
    *,
    topology: Any,
    window_xyz: np.ndarray,
    rule: str,
    selection_pairs: tuple[SelectionPair, ...],
) -> RuleEvaluation:
    """Evaluate one supported labeling rule on a final trajectory window."""

    normalized_rule = rule.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_rule == "sum_distances":
        return _evaluate_sum_distances(topology=topology, window_xyz=window_xyz, selection_pairs=selection_pairs)
    if normalized_rule == "com_distance":
        return _evaluate_com_distance(topology=topology, window_xyz=window_xyz, selection_pairs=selection_pairs)
    raise ValueError(f"Unsupported labeling rule {rule!r}.")


def gather_nearby_waters(
    *,
    topology: Any,
    window_xyz: np.ndarray,
    reference_atom_indices: tuple[int, ...],
    n_waters: int = 50,
) -> WaterMetadata:
    """Collect mean distances for the nearest water residues around a reference selection."""

    if n_waters < 1:
        raise ValueError("n_waters must be at least 1.")

    reference_indices = np.asarray(reference_atom_indices, dtype=np.int64)
    if reference_indices.size == 0:
        return WaterMetadata(atom_indices=(), residue_ids=(), residue_names=(), mean_distances=np.empty(0, dtype=np.float64))

    reference_com = _compute_center_of_mass(window_xyz, topology, reference_indices)
    candidates: list[tuple[float, int, int, str]] = []

    for residue, residue_atom_indices in _water_residue_groups(topology):
        water_indices = np.asarray(_preferred_water_atom_indices(residue_atom_indices, topology), dtype=np.int64)
        water_com = _compute_center_of_mass(window_xyz, topology, water_indices)
        mean_distance = float(np.linalg.norm(reference_com - water_com, axis=1).mean())
        candidates.append((mean_distance, int(water_indices[0]), _residue_id(residue), _residue_name(residue)))

    candidates.sort(key=lambda item: item[0])
    selected = candidates[:n_waters]

    return WaterMetadata(
        atom_indices=tuple(item[1] for item in selected),
        residue_ids=tuple(item[2] for item in selected),
        residue_names=tuple(item[3] for item in selected),
        mean_distances=np.asarray([item[0] for item in selected], dtype=np.float64),
    )


def _evaluate_sum_distances(
    *,
    topology: Any,
    window_xyz: np.ndarray,
    selection_pairs: tuple[SelectionPair, ...],
) -> RuleEvaluation:
    """Evaluate the frame-wise sum of atom-pair distances."""

    if not selection_pairs:
        raise ValueError("sum_distances requires at least one selection pair.")

    resolved_pairs: list[tuple[int, int]] = []
    component_labels: list[str] = []
    reference_indices: set[int] = set()

    for selection_a, selection_b in selection_pairs:
        indices_a = _select_indices(topology, selection_a)
        indices_b = _select_indices(topology, selection_b)
        if indices_a.size != 1 or indices_b.size != 1:
            raise ValueError(
                "sum_distances expects each selection to resolve to exactly one atom. "
                "Use com_distance when selections contain multiple atoms."
            )
        atom_a = int(indices_a[0])
        atom_b = int(indices_b[0])
        resolved_pairs.append((atom_a, atom_b))
        component_labels.append(f"{selection_a} -> {selection_b}")
        reference_indices.update((atom_a, atom_b))

    component_values = _pair_distances(window_xyz, resolved_pairs)
    frame_values = component_values.sum(axis=1)
    return RuleEvaluation(
        rule="sum_distances",
        frame_values=frame_values,
        component_values=component_values,
        component_labels=tuple(component_labels),
        reference_atom_indices=tuple(sorted(reference_indices)),
    )


def _evaluate_com_distance(
    *,
    topology: Any,
    window_xyz: np.ndarray,
    selection_pairs: tuple[SelectionPair, ...],
) -> RuleEvaluation:
    """Evaluate the frame-wise center-of-mass distance between two selections."""

    if len(selection_pairs) != 1:
        raise ValueError("com_distance requires exactly one selection pair.")

    selection_a, selection_b = selection_pairs[0]
    indices_a = _select_indices(topology, selection_a)
    indices_b = _select_indices(topology, selection_b)

    com_a = _compute_center_of_mass(window_xyz, topology, indices_a)
    com_b = _compute_center_of_mass(window_xyz, topology, indices_b)
    frame_values = np.linalg.norm(com_a - com_b, axis=1)
    component_values = frame_values[:, None]
    return RuleEvaluation(
        rule="com_distance",
        frame_values=frame_values,
        component_values=component_values,
        component_labels=(f"COM({selection_a}) -> COM({selection_b})",),
        reference_atom_indices=tuple(sorted({*(int(index) for index in indices_a), *(int(index) for index in indices_b)})),
    )


def _select_indices(topology: Any, selection: str) -> np.ndarray:
    """Resolve an mdtraj selection string into atom indices."""

    indices = np.asarray(topology.select(selection), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"Selection {selection!r} did not resolve to any atoms.")
    return indices


def _pair_distances(window_xyz: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Compute Euclidean distances for atom pairs across all frames in a window."""

    component_values = np.empty((window_xyz.shape[0], len(pairs)), dtype=np.float64)
    for column, (atom_a, atom_b) in enumerate(pairs):
        delta = window_xyz[:, atom_a, :] - window_xyz[:, atom_b, :]
        component_values[:, column] = np.linalg.norm(delta, axis=1)
    return component_values


def _compute_center_of_mass(window_xyz: np.ndarray, topology: Any, atom_indices: np.ndarray) -> np.ndarray:
    """Compute a center-of-mass series across frames for a selection."""

    masses = np.asarray([_atom_mass(_atom_by_index(topology, int(index))) for index in atom_indices], dtype=np.float64)
    total_mass = float(masses.sum())
    if total_mass <= 0.0:
        masses = np.ones(atom_indices.shape[0], dtype=np.float64)
        total_mass = float(masses.sum())

    coords = window_xyz[:, atom_indices, :]
    return (coords * masses[None, :, None]).sum(axis=1) / total_mass


def _water_residue_groups(topology: Any) -> list[tuple[Any, list[int]]]:
    """Group water atoms by residue."""

    groups: dict[int, tuple[Any, list[int]]] = {}
    for atom in topology.atoms:
        residue = atom.residue
        if not _is_water_residue(residue):
            continue
        key = id(residue)
        if key not in groups:
            groups[key] = (residue, [])
        groups[key][1].append(int(atom.index))
    return list(groups.values())


def _preferred_water_atom_indices(atom_indices: list[int], topology: Any) -> list[int]:
    """Prefer water oxygen atoms when available, otherwise use the full residue."""

    oxygen_indices = [index for index in atom_indices if _atom_name(_atom_by_index(topology, index)).upper().startswith("O")]
    return oxygen_indices or atom_indices


def _is_water_residue(residue: Any) -> bool:
    """Return ``True`` when a residue looks like water."""

    if bool(getattr(residue, "is_water", False)):
        return True
    return _residue_name(residue).upper() in WATER_RESIDUE_NAMES


def _atom_by_index(topology: Any, atom_index: int) -> Any:
    """Fetch an atom object by index from a topology."""

    return topology.atoms[atom_index]


def _atom_name(atom: Any) -> str:
    """Return a normalized atom name."""

    return str(getattr(atom, "name", getattr(atom, "element", "X")))


def _atom_mass(atom: Any) -> float:
    """Return an atomic mass, defaulting to 1.0 when unavailable."""

    element = getattr(atom, "element", None)
    if element is not None:
        mass = getattr(element, "mass", None)
        if mass is not None:
            return float(mass)
    mass = getattr(atom, "mass", None)
    if mass is not None:
        return float(mass)
    return 1.0


def _residue_name(residue: Any) -> str:
    """Return a residue name."""

    return str(getattr(residue, "name", "UNK"))


def _residue_id(residue: Any) -> int:
    """Return a stable residue identifier when possible."""

    if hasattr(residue, "resSeq"):
        return int(residue.resSeq)
    if hasattr(residue, "index"):
        return int(residue.index)
    return 0


__all__ = ["RuleEvaluation", "SelectionPair", "WaterMetadata", "evaluate_rule", "gather_nearby_waters"]
