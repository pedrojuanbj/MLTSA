"""Export helpers for labeled MD state structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from mltsa.io.h5 import open_h5
from mltsa.io.schema import results_experiment_path

from .label import STATE_ORDER, SnapshotModel, _export_snapshots, _import_mdtraj, _load_trajectory


def export_state_structures(
    *,
    h5_path: str | Path,
    experiment_id: str,
    output_dir: str | Path,
    replica_ids: Sequence[str] | None = None,
    center: bool = False,
    align_protein: bool = False,
) -> dict[str, Path]:
    """Export multi-model PDB snapshots for previously labeled MD states."""

    target_dir = Path(output_dir)
    selected_replica_ids = None if replica_ids is None else {str(value) for value in replica_ids}
    snapshots: list[SnapshotModel] = []

    with open_h5(h5_path, "r") as handle:
        entries_group = handle.get(f"{results_experiment_path(experiment_id)}/trajectory_labels/entries", default=None)
        if not isinstance(entries_group, h5py.Group):
            raise ValueError(
                f"Could not find stored MD labels for experiment {experiment_id!r}. "
                "Run label_trajectories(...) first."
            )

        md = _import_mdtraj()
        for child in entries_group.values():
            if not isinstance(child, h5py.Group):
                continue
            replica_id = str(child.attrs["replica_id"])
            if selected_replica_ids is not None and replica_id not in selected_replica_ids:
                continue

            trajectory = _load_trajectory(
                md,
                Path(str(child.attrs["trajectory_path"])),
                Path(str(child.attrs["topology_path"])),
            )
            topology = getattr(trajectory, "topology", getattr(trajectory, "top", None))
            if topology is None:
                raise TypeError("Loaded trajectory does not expose a topology object.")

            snapshots.append(
                SnapshotModel(
                    label=str(child.attrs["state_label"]),
                    xyz=np.asarray(trajectory.xyz[-1], dtype=np.float64),
                    topology=topology,
                    trajectory_path=str(child.attrs["trajectory_path"]),
                    replica_id=replica_id,
                )
            )

    snapshot_paths = {state: target_dir / f"{state}.pdb" for state in STATE_ORDER}
    _export_snapshots(snapshots=snapshots, snapshot_paths=snapshot_paths, center=center, align_protein=align_protein)
    return snapshot_paths


__all__ = ["export_state_structures"]
