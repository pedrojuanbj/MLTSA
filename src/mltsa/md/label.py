"""Trajectory labeling helpers for molecular dynamics workflows."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from mltsa.io.h5 import ensure_group, open_h5, replace_dataset, write_utf8_array
from mltsa.io.schema import ensure_results_layout, results_experiment_path
from mltsa.md.features.rules import SelectionPair, WaterMetadata, evaluate_rule, gather_nearby_waters
from mltsa.synthetic.base import JSONValue, canonical_json

LABEL_SCHEMA_VERSION = 1
LABELS_GROUP_NAME = "trajectory_labels"
ENTRIES_GROUP_NAME = "entries"
STATE_ORDER = ("IN", "OUT", "TS")


@dataclass(frozen=True, slots=True)
class SnapshotModel:
    """Single-frame snapshot used for optional PDB export."""

    label: str
    xyz: np.ndarray
    topology: Any
    trajectory_path: str
    replica_id: str


@dataclass(frozen=True, slots=True)
class TrajectoryLabelEntry:
    """Labeled result for one trajectory."""

    entry_key: str
    trajectory_path: str
    normalized_trajectory_path: str
    topology_path: str
    replica_id: str
    label: str
    mean_value: float
    frame_values: np.ndarray
    component_values: np.ndarray
    component_labels: tuple[str, ...]
    rule: str
    n_frames_total: int
    final_window_start: int
    nearby_waters: WaterMetadata | None


@dataclass(frozen=True, slots=True)
class LabelingResult:
    """Summary returned by :func:`label_trajectories`."""

    experiment_path: str
    h5_path: Path
    processed: tuple[TrajectoryLabelEntry, ...]
    skipped_entry_keys: tuple[str, ...]
    overwritten_entry_keys: tuple[str, ...]
    state_counts: dict[str, int]
    summary_path: Path
    pie_chart_path: Path
    snapshot_paths: dict[str, Path]


def label_trajectories(
    *,
    trajectory_paths: Sequence[str | Path],
    topology: str | Path | Sequence[str | Path],
    h5_path: str | Path,
    experiment_id: str,
    rule: str,
    selection_pairs: Sequence[SelectionPair],
    lower_threshold: float,
    upper_threshold: float,
    window_size: int,
    replica_ids: Sequence[str] | None = None,
    append: bool = False,
    overwrite: bool = False,
    store_waters: bool = True,
    n_waters: int = 50,
    export_snapshots: bool = False,
    snapshot_dir: str | Path | None = None,
    center_snapshots: bool = False,
    align_protein: bool = False,
    report_dir: str | Path | None = None,
) -> LabelingResult:
    """Label trajectories from the mean rule value on the final ``window_size`` frames.

    ``sum_distances`` expects one or more atom-pair selections. ``com_distance``
    requires exactly one selection pair and uses the center of mass of each
    selection. Thresholds are interpreted in the same units as the underlying
    mdtraj coordinates, which are typically nanometers.
    """

    normalized_rule = _normalize_rule(rule)
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if upper_threshold < lower_threshold:
        raise ValueError("upper_threshold must be greater than or equal to lower_threshold.")

    trajectory_list = [Path(path) for path in trajectory_paths]
    if not trajectory_list:
        raise ValueError("trajectory_paths must not be empty.")

    topology_list = _normalize_topology_paths(topology, len(trajectory_list))
    replica_list = _normalize_replica_ids(replica_ids, trajectory_list)
    selection_tuple = tuple((str(left), str(right)) for left, right in selection_pairs)
    if not selection_tuple:
        raise ValueError("selection_pairs must not be empty.")

    md = _import_mdtraj()
    result_file = Path(h5_path)
    report_root = Path(report_dir) if report_dir is not None else result_file.parent
    snapshot_root = Path(snapshot_dir) if snapshot_dir is not None else report_root / "snapshots" / _safe_file_stem(experiment_id)

    store_config = _store_config(
        rule=normalized_rule,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        window_size=window_size,
        selection_pairs=selection_tuple,
        store_waters=store_waters,
        n_waters=n_waters,
    )

    processed: list[TrajectoryLabelEntry] = []
    skipped_entry_keys: list[str] = []
    overwritten_entry_keys: list[str] = []
    snapshots: list[SnapshotModel] = []

    experiment_path = f"{results_experiment_path(experiment_id)}/{LABELS_GROUP_NAME}"
    with open_h5(result_file, "a") as handle:
        ensure_results_layout(handle)
        experiment_group = ensure_group(handle, results_experiment_path(experiment_id), attrs={"kind": "md_labeling"})

        if LABELS_GROUP_NAME in experiment_group and not append:
            if not overwrite:
                raise ValueError(
                    f"Label store already exists at {experiment_path!r}. Use append=True to add entries "
                    "or overwrite=True to replace the store."
                )
            del experiment_group[LABELS_GROUP_NAME]

        label_group = ensure_group(
            experiment_group,
            LABELS_GROUP_NAME,
            attrs={
                "schema": "mltsa.md.labels",
                "schema_version": LABEL_SCHEMA_VERSION,
                **store_config,
            },
        )
        entries_group = ensure_group(label_group, ENTRIES_GROUP_NAME)
        write_utf8_array(
            label_group,
            "selection_pairs",
            [json.dumps({"left": left, "right": right}, sort_keys=True) for left, right in selection_tuple],
            overwrite=True,
        )

        if len(entries_group) > 0:
            _validate_store_config(label_group, store_config, append=append, overwrite=overwrite)

        for trajectory_path, topology_path, replica_id in zip(trajectory_list, topology_list, replica_list):
            normalized_trajectory_path = _normalize_path(trajectory_path)
            entry_key = _entry_key(replica_id, normalized_trajectory_path)
            existing = entries_group.get(entry_key)

            if existing is not None and append and not overwrite:
                skipped_entry_keys.append(entry_key)
                continue
            if existing is not None and overwrite:
                del entries_group[entry_key]
                overwritten_entry_keys.append(entry_key)

            trajectory = _load_trajectory(md, trajectory_path, topology_path)
            topology_obj = getattr(trajectory, "topology", getattr(trajectory, "top", None))
            if topology_obj is None:
                raise TypeError("Loaded trajectory does not expose a topology object.")

            n_frames_total = int(trajectory.xyz.shape[0])
            if n_frames_total < 1:
                raise ValueError(f"Trajectory {trajectory_path} does not contain any frames.")
            final_window_start = max(0, n_frames_total - window_size)
            window_xyz = np.asarray(trajectory.xyz[final_window_start:], dtype=np.float64)
            evaluation = evaluate_rule(
                topology=topology_obj,
                window_xyz=window_xyz,
                rule=normalized_rule,
                selection_pairs=selection_tuple,
            )
            mean_value = float(evaluation.frame_values.mean())
            label = _classify_state(mean_value, lower_threshold=lower_threshold, upper_threshold=upper_threshold)

            nearby_waters = (
                gather_nearby_waters(
                    topology=topology_obj,
                    window_xyz=window_xyz,
                    reference_atom_indices=evaluation.reference_atom_indices,
                    n_waters=n_waters,
                )
                if store_waters
                else None
            )

            entry = TrajectoryLabelEntry(
                entry_key=entry_key,
                trajectory_path=str(trajectory_path),
                normalized_trajectory_path=normalized_trajectory_path,
                topology_path=str(topology_path),
                replica_id=replica_id,
                label=label,
                mean_value=mean_value,
                frame_values=np.asarray(evaluation.frame_values, dtype=np.float64),
                component_values=np.asarray(evaluation.component_values, dtype=np.float64),
                component_labels=tuple(evaluation.component_labels),
                rule=normalized_rule,
                n_frames_total=n_frames_total,
                final_window_start=final_window_start,
                nearby_waters=nearby_waters,
            )
            _write_entry(
                entries_group,
                entry,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                window_size=window_size,
            )
            processed.append(entry)

            if export_snapshots:
                snapshots.append(
                    SnapshotModel(
                        label=label,
                        xyz=np.asarray(trajectory.xyz[-1], dtype=np.float64),
                        topology=topology_obj,
                        trajectory_path=str(trajectory_path),
                        replica_id=replica_id,
                    )
                )

        state_counts = _state_counts_from_store(entries_group)

    summary_path = report_root / f"{_safe_file_stem(experiment_id)}_label_summary.json"
    pie_chart_path = report_root / f"{_safe_file_stem(experiment_id)}_state_counts.png"
    _write_summary(
        summary_path,
        experiment_id=experiment_id,
        h5_path=result_file,
        config=store_config,
        state_counts=state_counts,
        processed=len(processed),
        skipped=len(skipped_entry_keys),
        overwritten=len(overwritten_entry_keys),
    )
    _write_state_pie_chart(pie_chart_path, state_counts=state_counts, experiment_id=experiment_id)

    snapshot_paths = {state: snapshot_root / f"{state}.pdb" for state in STATE_ORDER}
    if export_snapshots:
        _export_snapshots(
            snapshots=snapshots,
            snapshot_paths=snapshot_paths,
            center=center_snapshots,
            align_protein=align_protein,
        )

    return LabelingResult(
        experiment_path=experiment_path,
        h5_path=result_file,
        processed=tuple(processed),
        skipped_entry_keys=tuple(skipped_entry_keys),
        overwritten_entry_keys=tuple(overwritten_entry_keys),
        state_counts=state_counts,
        summary_path=summary_path,
        pie_chart_path=pie_chart_path,
        snapshot_paths=snapshot_paths,
    )


def _import_mdtraj() -> Any:
    """Import mdtraj lazily so package import remains lightweight."""

    try:
        import mdtraj as md
    except ImportError as exc:  # pragma: no cover - exercised indirectly in user environments.
        raise ImportError(
            "label_trajectories requires mdtraj at runtime. Install the optional "
            "dependency set with `pip install mltsa[md]` or provide mdtraj in the active environment."
        ) from exc
    return md


def _load_trajectory(md: Any, trajectory_path: Path, topology_path: Path) -> Any:
    """Load a trajectory through mdtraj using a single shared topology."""

    return md.load(str(trajectory_path), top=str(topology_path))


def _normalize_topology_paths(topology: str | Path | Sequence[str | Path], count: int) -> list[Path]:
    """Broadcast one topology path or validate a per-trajectory topology list."""

    if isinstance(topology, (str, Path)):
        return [Path(topology)] * count

    topology_list = [Path(path) for path in topology]
    if len(topology_list) != count:
        raise ValueError("When topology is a sequence, it must match the number of trajectories.")
    return topology_list


def _normalize_replica_ids(replica_ids: Sequence[str] | None, trajectory_paths: Sequence[Path]) -> list[str]:
    """Normalize replica ids, defaulting to the trajectory stem."""

    if replica_ids is None:
        return [_validate_replica_id(path.stem or f"replica_{index:04d}") for index, path in enumerate(trajectory_paths)]

    if len(replica_ids) != len(trajectory_paths):
        raise ValueError("replica_ids must match the number of trajectories.")
    return [_validate_replica_id(replica_id) for replica_id in replica_ids]


def _validate_replica_id(value: str) -> str:
    """Reject empty or nested replica ids."""

    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("replica ids must not be empty.")
    if "/" in cleaned:
        raise ValueError("replica ids must not contain '/'.")
    return cleaned


def _normalize_rule(rule: str) -> str:
    """Canonicalize a user-facing rule name."""

    normalized = rule.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {"sum_distances", "com_distance"}:
        raise ValueError(f"Unsupported labeling rule {rule!r}.")
    return normalized


def _normalize_path(path: str | Path) -> str:
    """Normalize a filesystem path for HDF5 entry identity checks."""

    resolved = Path(path).resolve(strict=False)
    return resolved.as_posix().casefold()


def _entry_key(replica_id: str, normalized_trajectory_path: str) -> str:
    """Build a deterministic HDF5-safe entry key."""

    digest = hashlib.sha1(f"{replica_id}\n{normalized_trajectory_path}".encode("utf-8")).hexdigest()[:16]
    replica_fragment = re.sub(r"[^0-9A-Za-z]+", "_", replica_id).strip("_") or "replica"
    return f"{replica_fragment}_{digest}"


def _classify_state(mean_value: float, *, lower_threshold: float, upper_threshold: float) -> str:
    """Map a window mean value to one of the supported MD states."""

    if mean_value <= lower_threshold:
        return "IN"
    if mean_value >= upper_threshold:
        return "OUT"
    return "TS"


def _store_config(
    *,
    rule: str,
    lower_threshold: float,
    upper_threshold: float,
    window_size: int,
    selection_pairs: tuple[SelectionPair, ...],
    store_waters: bool,
    n_waters: int,
) -> dict[str, JSONValue]:
    """Build canonical store-level metadata."""

    return {
        "rule": rule,
        "lower_threshold": float(lower_threshold),
        "upper_threshold": float(upper_threshold),
        "window_size": int(window_size),
        "selection_pairs_json": canonical_json(
            {
                "pairs": [{"left": left, "right": right} for left, right in selection_pairs],
            }
        ),
        "store_waters": bool(store_waters),
        "n_waters": int(n_waters),
    }


def _validate_store_config(label_group: h5py.Group, expected: dict[str, JSONValue], *, append: bool, overwrite: bool) -> None:
    """Prevent accidental mixing of incompatible labeling configurations."""

    current = {name: _normalize_attr_value(label_group.attrs.get(name)) for name in expected}
    if current == expected:
        return

    if append and not overwrite:
        raise ValueError(
            "Existing label store configuration does not match the requested labeling setup. "
            "Use overwrite=True with append=False to rebuild the store from scratch."
        )
    if append and overwrite:
        raise ValueError(
            "Appending with overwrite=True can replace matching entries, but it cannot safely change the store-wide "
            "labeling configuration while keeping old entries. Re-run with append=False and overwrite=True."
        )


def _normalize_attr_value(value: object) -> JSONValue:
    """Convert HDF5 attribute values into plain JSON-compatible Python values."""

    if isinstance(value, np.generic):
        return value.item()
    return value  # type: ignore[return-value]


def _write_entry(
    entries_group: h5py.Group,
    entry: TrajectoryLabelEntry,
    *,
    lower_threshold: float,
    upper_threshold: float,
    window_size: int,
) -> None:
    """Write one trajectory label entry into HDF5."""

    group = entries_group.create_group(entry.entry_key, track_order=True)
    group.attrs.update(
        {
            "kind": "md_trajectory_label",
            "entry_key": entry.entry_key,
            "trajectory_path": entry.trajectory_path,
            "normalized_trajectory_path": entry.normalized_trajectory_path,
            "topology_path": entry.topology_path,
            "replica_id": entry.replica_id,
            "state_label": entry.label,
            "rule": entry.rule,
            "mean_value": float(entry.mean_value),
            "n_frames_total": int(entry.n_frames_total),
            "final_window_start": int(entry.final_window_start),
            "window_size": int(window_size),
            "lower_threshold": float(lower_threshold),
            "upper_threshold": float(upper_threshold),
            "waters_stored": bool(entry.nearby_waters is not None),
        }
    )
    replace_dataset(group, "frame_values", entry.frame_values)
    replace_dataset(group, "component_values", entry.component_values)
    write_utf8_array(group, "component_labels", entry.component_labels, overwrite=True)

    if entry.nearby_waters is not None:
        waters_group = ensure_group(group, "nearby_waters")
        replace_dataset(waters_group, "atom_indices", np.asarray(entry.nearby_waters.atom_indices, dtype=np.int64))
        replace_dataset(waters_group, "residue_ids", np.asarray(entry.nearby_waters.residue_ids, dtype=np.int64))
        replace_dataset(waters_group, "mean_distances", np.asarray(entry.nearby_waters.mean_distances, dtype=np.float64))
        write_utf8_array(waters_group, "residue_names", entry.nearby_waters.residue_names, overwrite=True)


def _state_counts_from_store(entries_group: h5py.Group) -> dict[str, int]:
    """Count stored state labels using only entry metadata."""

    counts: Counter[str] = Counter()
    for child in entries_group.values():
        if not isinstance(child, h5py.Group):
            continue
        counts[str(child.attrs.get("state_label", ""))] += 1
    return {state: int(counts.get(state, 0)) for state in STATE_ORDER}


def _write_summary(
    path: Path,
    *,
    experiment_id: str,
    h5_path: Path,
    config: dict[str, JSONValue],
    state_counts: dict[str, int],
    processed: int,
    skipped: int,
    overwritten: int,
) -> None:
    """Write a JSON summary for a labeling run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment_id,
        "h5_path": str(h5_path),
        "config": config,
        "state_counts": state_counts,
        "total_entries": int(sum(state_counts.values())),
        "processed_entries": int(processed),
        "skipped_entries": int(skipped),
        "overwritten_entries": int(overwritten),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_state_pie_chart(path: Path, *, state_counts: dict[str, int], experiment_id: str) -> None:
    """Write a simple pie chart summarizing state counts."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    values = [state_counts.get(state, 0) for state in STATE_ORDER]
    colors = ["#4c956c", "#bc4749", "#f4a259"]

    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    total = sum(values)
    if total == 0:
        axis.text(0.5, 0.5, "No labels stored", ha="center", va="center")
        axis.axis("off")
    else:
        axis.pie(
            values,
            labels=list(STATE_ORDER),
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            startangle=90,
        )
    axis.set_title(f"State counts: {experiment_id}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _export_snapshots(
    *,
    snapshots: Sequence[SnapshotModel],
    snapshot_paths: dict[str, Path],
    center: bool,
    align_protein: bool,
) -> None:
    """Export multi-model PDBs for the currently processed snapshots."""

    grouped = {state: [snapshot for snapshot in snapshots if snapshot.label == state] for state in STATE_ORDER}
    for state, path in snapshot_paths.items():
        prepared = _prepare_snapshots(grouped[state], center=center, align_protein=align_protein)
        _write_multi_model_pdb(path, prepared)


def _prepare_snapshots(snapshots: Sequence[SnapshotModel], *, center: bool, align_protein: bool) -> list[SnapshotModel]:
    """Apply optional centering and alignment to snapshots before export."""

    prepared = [
        SnapshotModel(
            label=snapshot.label,
            xyz=np.asarray(snapshot.xyz, dtype=np.float64).copy(),
            topology=snapshot.topology,
            trajectory_path=snapshot.trajectory_path,
            replica_id=snapshot.replica_id,
        )
        for snapshot in snapshots
    ]

    if not prepared:
        return []

    if center:
        for index, snapshot in enumerate(prepared):
            atom_indices = _protein_atom_indices(snapshot.topology)
            if atom_indices.size == 0:
                atom_indices = np.arange(snapshot.xyz.shape[0], dtype=np.int64)
            centroid = snapshot.xyz[atom_indices].mean(axis=0)
            prepared[index] = SnapshotModel(
                label=snapshot.label,
                xyz=snapshot.xyz - centroid,
                topology=snapshot.topology,
                trajectory_path=snapshot.trajectory_path,
                replica_id=snapshot.replica_id,
            )

    if align_protein and len(prepared) > 1:
        reference = prepared[0]
        atom_indices = _protein_atom_indices(reference.topology)
        if atom_indices.size == 0:
            atom_indices = np.arange(reference.xyz.shape[0], dtype=np.int64)
        reference_subset = reference.xyz[atom_indices]
        ref_center = reference_subset.mean(axis=0)
        reference_subset = reference_subset - ref_center

        aligned: list[SnapshotModel] = [reference]
        for snapshot in prepared[1:]:
            moving_subset = snapshot.xyz[atom_indices]
            moving_center = moving_subset.mean(axis=0)
            centered_moving = moving_subset - moving_center
            rotation = _kabsch(centered_moving, reference_subset)
            aligned_xyz = (snapshot.xyz - moving_center) @ rotation + ref_center
            aligned.append(
                SnapshotModel(
                    label=snapshot.label,
                    xyz=aligned_xyz,
                    topology=snapshot.topology,
                    trajectory_path=snapshot.trajectory_path,
                    replica_id=snapshot.replica_id,
                )
            )
        prepared = aligned

    return prepared


def _protein_atom_indices(topology: Any) -> np.ndarray:
    """Return protein atom indices when the topology can identify them."""

    indices = [
        int(atom.index)
        for atom in topology.atoms
        if bool(getattr(atom.residue, "is_protein", False))
    ]
    return np.asarray(indices, dtype=np.int64)


def _kabsch(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute an optimal rotation from ``moving`` to ``reference``."""

    if moving.shape[0] < 2:
        return np.eye(3, dtype=np.float64)

    covariance = moving.T @ reference
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0.0:
        correction[-1, -1] = -1.0
    return left @ correction @ right_t


def _write_multi_model_pdb(path: Path, snapshots: Sequence[SnapshotModel]) -> None:
    """Write a lightweight multi-model PDB file from single-frame snapshots."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("REMARK Generated by mltsa.md.label\n")
        if not snapshots:
            handle.write("END\n")
            return

        for model_index, snapshot in enumerate(snapshots, start=1):
            handle.write(f"MODEL     {model_index}\n")
            for atom in snapshot.topology.atoms:
                residue = atom.residue
                serial = int(getattr(atom, "serial", atom.index + 1))
                name = str(getattr(atom, "name", "X"))[:4]
                resname = str(getattr(residue, "name", "UNK"))[:3]
                chain_id = _chain_id(residue)
                residue_id = int(getattr(residue, "resSeq", getattr(residue, "index", 1)))
                x, y, z = np.asarray(snapshot.xyz[int(atom.index)], dtype=np.float64) * 10.0
                element = str(getattr(getattr(atom, "element", None), "symbol", name[:1])).strip()[:2]
                handle.write(
                    f"ATOM  {serial:5d} {name:<4s} {resname:>3s} {chain_id}{residue_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
                )
            handle.write("ENDMDL\n")
        handle.write("END\n")


def _chain_id(residue: Any) -> str:
    """Return a PDB-compatible chain identifier."""

    chain = getattr(residue, "chain", None)
    index = int(getattr(chain, "index", 0))
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index % 26]


def _safe_file_stem(value: str) -> str:
    """Return a filesystem-safe identifier fragment."""

    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return cleaned or "labels"


__all__ = ["LabelingResult", "TrajectoryLabelEntry", "label_trajectories"]
