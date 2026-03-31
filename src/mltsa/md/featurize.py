"""MD featurization helpers built on the appendable mltsa HDF5 schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from mltsa.io.h5 import ensure_group, open_h5, read_utf8_array, replace_dataset, write_utf8_array
from mltsa.io.schema import ensure_md_layout, feature_set_path, replica_path, results_experiment_path
from mltsa.synthetic.base import JSONValue, canonical_json

from .features.base import (
    FEATURE_SET_SCHEMA,
    FEATURE_SET_SCHEMA_VERSION,
    FeatureComputation,
    MDFeatureDataset,
    normalize_feature_type,
    normalize_path,
    state_to_int,
    validate_replica_id,
)
from .features.contacts import contact_map
from .features.distances import all_ligand_protein_distances, bubble_distances, closest_residue_distances
from .features.pca import pca_xyz
from .features.waters import read_stored_water_atom_indices

FEATURE_REPLICAS_GROUP = "replicas"


@dataclass(frozen=True, slots=True)
class LabelSourceEntry:
    """Replica metadata read from a stored MD labeling result."""

    entry_key: str
    replica_id: str
    trajectory_path: str
    normalized_trajectory_path: str
    topology_path: str
    state_label: str
    y: int
    water_atom_indices: tuple[int, ...]


def featurize_dataset(
    *,
    h5_path: str | Path,
    feature_set: str,
    feature_type: str,
    label_experiment_id: str,
    replica_ids: Sequence[str] | None = None,
    append: bool = False,
    overwrite: bool = False,
    use_waters: bool = False,
    ligand_selection: str = "resname LIG",
    protein_selection: str = "protein",
    bubble_cutoff: float = 0.6,
    contact_threshold: float = 0.45,
    pca_components: int = 3,
    pca_selection: str | None = None,
) -> MDFeatureDataset:
    """Compute one named MD feature set from trajectories already labeled in the HDF5 file."""

    normalized_feature_type = normalize_feature_type(feature_type)
    feature_set_id = validate_replica_id(feature_set)
    requested_replica_ids = None if replica_ids is None else tuple(validate_replica_id(value) for value in replica_ids)
    file_path = Path(h5_path)
    config = _feature_config(
        feature_type=normalized_feature_type,
        ligand_selection=ligand_selection,
        protein_selection=protein_selection,
        bubble_cutoff=bubble_cutoff,
        contact_threshold=contact_threshold,
        pca_components=pca_components,
        pca_selection=pca_selection,
        use_waters=use_waters,
    )

    processed: list[str] = []
    skipped: list[str] = []
    overwritten: list[str] = []

    with open_h5(file_path, "a") as handle:
        ensure_md_layout(handle)
        label_entries = _load_label_entries(
            handle,
            label_experiment_id=label_experiment_id,
            replica_ids=requested_replica_ids,
            use_waters=use_waters,
        )
        feature_group = _prepare_feature_group(
            handle,
            feature_set=feature_set_id,
            feature_type=normalized_feature_type,
            label_experiment_id=label_experiment_id,
            config=config,
            append=append,
            overwrite=overwrite,
        )
        replicas_group = ensure_group(feature_group, FEATURE_REPLICAS_GROUP)
        existing = _scan_existing_replicas(replicas_group)
        expected_feature_names = tuple(read_utf8_array(feature_group, "feature_names")) if "feature_names" in feature_group else None

        md = _import_mdtraj()
        for entry in label_entries:
            stored = existing.get(entry.replica_id)
            if stored is not None:
                if stored.normalized_trajectory_path != entry.normalized_trajectory_path:
                    raise ValueError(
                        f"Feature set {feature_set_id!r} already contains replica {entry.replica_id!r} "
                        "for a different trajectory path. Rebuild the feature set with overwrite=True "
                        "and append=False if you want to replace it."
                    )
                if append and not overwrite:
                    skipped.append(entry.replica_id)
                    continue
                del replicas_group[entry.replica_id]
                overwritten.append(entry.replica_id)

            trajectory = _load_trajectory(md, Path(entry.trajectory_path), Path(entry.topology_path))
            topology = getattr(trajectory, "topology", getattr(trajectory, "top", None))
            if topology is None:
                raise TypeError("Loaded trajectory does not expose a topology object.")
            xyz = np.asarray(trajectory.xyz, dtype=np.float64)
            if xyz.ndim != 3 or xyz.shape[0] < 1:
                raise ValueError(f"Trajectory {entry.trajectory_path} does not contain usable coordinates.")

            computation = _compute_features(
                topology=topology,
                xyz=xyz,
                feature_type=normalized_feature_type,
                ligand_selection=ligand_selection,
                protein_selection=protein_selection,
                bubble_cutoff=bubble_cutoff,
                contact_threshold=contact_threshold,
                pca_components=pca_components,
                pca_selection=pca_selection,
                water_atom_indices=entry.water_atom_indices if use_waters else (),
            )

            if expected_feature_names is None:
                write_utf8_array(feature_group, "feature_names", computation.feature_names, overwrite=True)
                expected_feature_names = computation.feature_names
            elif expected_feature_names != computation.feature_names:
                raise ValueError(
                    f"Replica {entry.replica_id!r} produced feature names that do not match the "
                    f"existing feature set {feature_set_id!r}."
                )

            _write_replica_metadata(handle, entry, label_experiment_id=label_experiment_id)
            _write_replica_features(replicas_group, entry, computation)
            processed.append(entry.replica_id)

        feature_group.attrs["n_replicas"] = int(len(replicas_group))
        feature_group.attrs["feature_dim"] = int(len(expected_feature_names or ()))

        return _load_dataset_from_handle(
            handle,
            feature_set=feature_set_id,
            processed_replica_ids=tuple(processed),
            skipped_replica_ids=tuple(skipped),
            overwritten_replica_ids=tuple(overwritten),
        )


def load_dataset(path: str | Path, feature_set: str) -> MDFeatureDataset:
    """Load one selected MD feature set from an HDF5 file."""

    with open_h5(path, "r") as handle:
        return _load_dataset_from_handle(handle, feature_set=validate_replica_id(feature_set))


def _prepare_feature_group(
    handle: h5py.File,
    *,
    feature_set: str,
    feature_type: str,
    label_experiment_id: str,
    config: dict[str, JSONValue],
    append: bool,
    overwrite: bool,
) -> h5py.Group:
    """Create or validate the on-disk feature-set group."""

    feature_path = feature_set_path(feature_set)
    if feature_path in handle and not append:
        if not overwrite:
            raise ValueError(
                f"Feature set already exists at {feature_path!r}. Use append=True to add missing replicas "
                "or overwrite=True to rebuild the feature set."
            )
        del handle[feature_path]

    feature_group = ensure_group(
        handle,
        feature_path,
        attrs={
            "schema": FEATURE_SET_SCHEMA,
            "schema_version": FEATURE_SET_SCHEMA_VERSION,
            "feature_type": feature_type,
            "label_experiment_id": label_experiment_id,
            "feature_config_json": canonical_json(config),
            "kind": "md_feature_set",
            "use_waters": bool(config["use_waters"]),
        },
    )

    replicas_group = ensure_group(feature_group, FEATURE_REPLICAS_GROUP)
    if len(replicas_group) > 0:
        _validate_feature_group_config(
            feature_group,
            feature_type=feature_type,
            label_experiment_id=label_experiment_id,
            config=config,
            append=append,
            overwrite=overwrite,
        )
    return feature_group


def _validate_feature_group_config(
    feature_group: h5py.Group,
    *,
    feature_type: str,
    label_experiment_id: str,
    config: dict[str, JSONValue],
    append: bool,
    overwrite: bool,
) -> None:
    """Prevent accidental mixing of incompatible feature-set configurations."""

    expected = {
        "feature_type": feature_type,
        "label_experiment_id": label_experiment_id,
        "feature_config_json": canonical_json(config),
        "use_waters": bool(config["use_waters"]),
    }
    current = {name: _normalize_attr(feature_group.attrs.get(name)) for name in expected}
    if current == expected:
        return

    if append and not overwrite:
        raise ValueError(
            "Existing feature-set configuration does not match the requested featurization setup. "
            "Use overwrite=True with append=False to rebuild the feature set."
        )
    if append and overwrite:
        raise ValueError(
            "Appending with overwrite=True can replace matching replicas, but it cannot safely change the "
            "feature-set configuration while keeping other stored replicas. Re-run with append=False and overwrite=True."
        )


def _load_label_entries(
    handle: h5py.File,
    *,
    label_experiment_id: str,
    replica_ids: tuple[str, ...] | None,
    use_waters: bool,
) -> list[LabelSourceEntry]:
    """Read replica metadata from a stored trajectory-labeling result."""

    entries_path = f"{results_experiment_path(label_experiment_id)}/trajectory_labels/entries"
    entries_group = handle.get(entries_path, default=None)
    if not isinstance(entries_group, h5py.Group):
        raise ValueError(
            f"Could not find stored MD labels at {entries_path!r}. Run label_trajectories(...) first."
        )

    requested = None if replica_ids is None else set(replica_ids)
    entries: list[LabelSourceEntry] = []

    for entry_key, child in entries_group.items():
        if not isinstance(child, h5py.Group):
            continue
        replica_id = validate_replica_id(str(child.attrs["replica_id"]))
        if requested is not None and replica_id not in requested:
            continue

        state_label = str(child.attrs["state_label"]).upper()
        trajectory_path = str(child.attrs["trajectory_path"])
        entries.append(
            LabelSourceEntry(
                entry_key=str(entry_key),
                replica_id=replica_id,
                trajectory_path=trajectory_path,
                normalized_trajectory_path=str(child.attrs.get("normalized_trajectory_path", normalize_path(trajectory_path))),
                topology_path=str(child.attrs["topology_path"]),
                state_label=state_label,
                y=state_to_int(state_label),
                water_atom_indices=read_stored_water_atom_indices(child) if use_waters else (),
            )
        )

    if requested is not None:
        found = {entry.replica_id for entry in entries}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"Requested replica ids are missing from the label store: {missing}")

    if not entries:
        raise ValueError("No labeled replicas matched the requested featurization inputs.")

    entries.sort(key=lambda entry: entry.replica_id)
    return entries


def _scan_existing_replicas(replicas_group: h5py.Group) -> dict[str, LabelSourceEntry]:
    """Collect existing featurized replicas using only child groups and attributes."""

    existing: dict[str, LabelSourceEntry] = {}
    for replica_id, child in replicas_group.items():
        if not isinstance(child, h5py.Group):
            continue
        normalized_trajectory_path = str(child.attrs.get("normalized_trajectory_path", ""))
        state_label = str(child.attrs.get("state_label", "IN")).upper()
        existing[str(replica_id)] = LabelSourceEntry(
            entry_key=str(replica_id),
            replica_id=str(replica_id),
            trajectory_path=str(child.attrs.get("trajectory_path", "")),
            normalized_trajectory_path=normalized_trajectory_path,
            topology_path=str(child.attrs.get("topology_path", "")),
            state_label=state_label,
            y=state_to_int(state_label),
            water_atom_indices=(),
        )
    return existing


def _write_replica_metadata(handle: h5py.File, entry: LabelSourceEntry, *, label_experiment_id: str) -> None:
    """Persist lightweight per-replica metadata under the canonical MD schema root."""

    replica_group = ensure_group(
        handle,
        replica_path(entry.replica_id),
        attrs={
            "kind": "md_replica",
            "trajectory_path": entry.trajectory_path,
            "normalized_trajectory_path": entry.normalized_trajectory_path,
            "topology_path": entry.topology_path,
        },
    )
    ensure_group(
        replica_group,
        f"labels/{label_experiment_id}",
        attrs={
            "kind": "md_replica_label",
            "state_label": entry.state_label,
            "label_code": int(entry.y),
            "label_entry_path": f"{results_experiment_path(label_experiment_id)}/trajectory_labels/entries/{entry.entry_key}",
        },
    )


def _write_replica_features(replicas_group: h5py.Group, entry: LabelSourceEntry, computation: FeatureComputation) -> None:
    """Write one featurized replica to HDF5."""

    replica_group = ensure_group(
        replicas_group,
        entry.replica_id,
        attrs={
            "kind": "md_featurized_replica",
            "replica_id": entry.replica_id,
            "trajectory_path": entry.trajectory_path,
            "normalized_trajectory_path": entry.normalized_trajectory_path,
            "topology_path": entry.topology_path,
            "state_label": entry.state_label,
            "label_code": int(entry.y),
            "metadata_json": canonical_json(computation.metadata),
            "n_frames": int(computation.values.shape[0]),
            "n_features": int(computation.values.shape[1]),
        },
    )
    replace_dataset(replica_group, "X", computation.values)


def _compute_features(
    *,
    topology: Any,
    xyz: np.ndarray,
    feature_type: str,
    ligand_selection: str,
    protein_selection: str,
    bubble_cutoff: float,
    contact_threshold: float,
    pca_components: int,
    pca_selection: str | None,
    water_atom_indices: tuple[int, ...],
) -> FeatureComputation:
    """Dispatch one supported feature family."""

    if feature_type == "closest_residue_distances":
        return closest_residue_distances(
            topology,
            xyz,
            ligand_selection=ligand_selection,
            protein_selection=protein_selection,
            water_atom_indices=water_atom_indices,
        )
    if feature_type == "all_ligand_protein_distances":
        return all_ligand_protein_distances(
            topology,
            xyz,
            ligand_selection=ligand_selection,
            protein_selection=protein_selection,
            water_atom_indices=water_atom_indices,
        )
    if feature_type == "bubble_distances":
        return bubble_distances(
            topology,
            xyz,
            ligand_selection=ligand_selection,
            protein_selection=protein_selection,
            bubble_cutoff=bubble_cutoff,
            water_atom_indices=water_atom_indices,
        )
    if feature_type == "contact_map":
        return contact_map(
            topology,
            xyz,
            ligand_selection=ligand_selection,
            protein_selection=protein_selection,
            contact_threshold=contact_threshold,
            water_atom_indices=water_atom_indices,
        )
    if feature_type == "pca_xyz":
        return pca_xyz(
            topology,
            xyz,
            atom_selection=pca_selection,
            n_components=pca_components,
        )
    raise ValueError(f"Unsupported MD feature type {feature_type!r}.")


def _load_dataset_from_handle(
    handle: h5py.File,
    *,
    feature_set: str,
    processed_replica_ids: tuple[str, ...] = (),
    skipped_replica_ids: tuple[str, ...] = (),
    overwritten_replica_ids: tuple[str, ...] = (),
) -> MDFeatureDataset:
    """Load one feature set from an already-open HDF5 handle."""

    feature_group = handle.get(feature_set_path(feature_set), default=None)
    if not isinstance(feature_group, h5py.Group):
        raise ValueError(f"Could not find feature set {feature_set!r} in the HDF5 file.")

    replicas_group = feature_group.get(FEATURE_REPLICAS_GROUP, default=None)
    if not isinstance(replicas_group, h5py.Group):
        raise ValueError(f"Feature set {feature_set!r} does not contain any replica groups.")

    feature_names = tuple(read_utf8_array(feature_group, "feature_names"))
    arrays: list[np.ndarray] = []
    y_values: list[int] = []
    state_labels: list[str] = []
    replica_ids: list[str] = []
    trajectory_paths: list[str] = []
    topology_paths: list[str] = []

    expected_shape: tuple[int, int] | None = None
    for replica_id in sorted(replicas_group.keys()):
        child = replicas_group[replica_id]
        if not isinstance(child, h5py.Group):
            continue
        values = np.asarray(child["X"][...], dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Replica {replica_id!r} does not contain a 2D feature matrix.")
        if values.shape[1] != len(feature_names):
            raise ValueError(f"Replica {replica_id!r} has a feature dimension that does not match feature_names.")
        if expected_shape is None:
            expected_shape = (int(values.shape[0]), int(values.shape[1]))
        elif expected_shape != (int(values.shape[0]), int(values.shape[1])):
            raise ValueError(
                "Stored replicas do not share a common frame/feature shape, so they cannot be loaded "
                "as one stacked dataset."
            )

        arrays.append(values)
        replica_ids.append(str(replica_id))
        state_labels.append(str(child.attrs["state_label"]))
        y_values.append(int(child.attrs["label_code"]))
        trajectory_paths.append(str(child.attrs["trajectory_path"]))
        topology_paths.append(str(child.attrs["topology_path"]))

    if not arrays:
        raise ValueError(f"Feature set {feature_set!r} does not contain any stored replica matrices.")

    metadata = json.loads(str(feature_group.attrs["feature_config_json"]))
    metadata["label_experiment_id"] = str(feature_group.attrs["label_experiment_id"])

    return MDFeatureDataset(
        feature_set=feature_set,
        feature_type=str(feature_group.attrs["feature_type"]),
        X=np.stack(arrays, axis=0),
        y=np.asarray(y_values, dtype=np.int64),
        state_labels=tuple(state_labels),
        feature_names=feature_names,
        replica_ids=tuple(replica_ids),
        trajectory_paths=tuple(trajectory_paths),
        topology_paths=tuple(topology_paths),
        metadata=metadata,
        processed_replica_ids=processed_replica_ids,
        skipped_replica_ids=skipped_replica_ids,
        overwritten_replica_ids=overwritten_replica_ids,
    )


def _feature_config(
    *,
    feature_type: str,
    ligand_selection: str,
    protein_selection: str,
    bubble_cutoff: float,
    contact_threshold: float,
    pca_components: int,
    pca_selection: str | None,
    use_waters: bool,
) -> dict[str, JSONValue]:
    """Build canonical feature-set configuration metadata."""

    config: dict[str, JSONValue] = {
        "feature_type": feature_type,
        "ligand_selection": ligand_selection,
        "protein_selection": protein_selection,
        "use_waters": bool(use_waters),
    }
    if feature_type == "bubble_distances":
        config["bubble_cutoff"] = float(bubble_cutoff)
    if feature_type == "contact_map":
        config["contact_threshold"] = float(contact_threshold)
    if feature_type == "pca_xyz":
        config["pca_components"] = int(pca_components)
        config["pca_selection"] = pca_selection if pca_selection is not None else "non_water"
    return config


def _import_mdtraj() -> Any:
    """Import mdtraj lazily so package import stays lightweight."""

    try:
        import mdtraj as md
    except ImportError as exc:  # pragma: no cover - exercised only in user environments.
        raise ImportError(
            "featurize_dataset requires mdtraj at runtime. Install the optional dependency set "
            "with `pip install mltsa[md]` or provide mdtraj in the active environment."
        ) from exc
    return md


def _load_trajectory(md: Any, trajectory_path: Path, topology_path: Path) -> Any:
    """Load one trajectory through mdtraj using a shared topology."""

    return md.load(str(trajectory_path), top=str(topology_path))


def _normalize_attr(value: object) -> JSONValue:
    """Convert HDF5 attribute values to plain JSON-compatible Python values."""

    if isinstance(value, np.generic):
        return value.item()
    return value  # type: ignore[return-value]


__all__ = ["MDFeatureDataset", "featurize_dataset", "load_dataset"]
