"""Tests for the MD trajectory labeling pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from mltsa.io.h5 import open_h5, read_utf8_array
from mltsa.io.schema import results_experiment_path
from mltsa.md import label_trajectories


@dataclass(frozen=True, slots=True)
class FakeElement:
    symbol: str
    mass: float


@dataclass(frozen=True, slots=True)
class FakeChain:
    index: int


@dataclass(frozen=True, slots=True)
class FakeResidue:
    name: str
    index: int
    resSeq: int
    chain: FakeChain
    is_water: bool = False
    is_protein: bool = False


@dataclass(frozen=True, slots=True)
class FakeAtom:
    index: int
    name: str
    residue: FakeResidue
    element: FakeElement
    serial: int


class FakeTopology:
    """Very small subset of the mdtraj topology interface used by the labeler."""

    def __init__(self, atoms: list[FakeAtom]) -> None:
        self.atoms = atoms

    def select(self, query: str) -> np.ndarray:
        normalized = " ".join(query.strip().split()).lower()
        if normalized.startswith("index "):
            return np.asarray([int(normalized.split()[1])], dtype=np.int64)
        if normalized == "protein":
            return np.asarray([atom.index for atom in self.atoms if atom.residue.is_protein], dtype=np.int64)
        if normalized == "resname lig":
            return np.asarray([atom.index for atom in self.atoms if atom.residue.name.upper() == "LIG"], dtype=np.int64)
        if normalized in {"water", "resname hoh", "water and name o", "name o and water"}:
            return np.asarray(
                [
                    atom.index
                    for atom in self.atoms
                    if atom.residue.is_water and (normalized in {"water", "resname hoh"} or atom.name.upper().startswith("O"))
                ],
                dtype=np.int64,
            )
        raise KeyError(f"Unsupported fake selection: {query!r}")


class FakeTrajectory:
    """Small trajectory container with the attributes needed by the labeler."""

    def __init__(self, xyz: np.ndarray, topology: FakeTopology) -> None:
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.topology = topology
        self.top = topology


class FakeMdtraj:
    """Registry-backed fake mdtraj module used for tests."""

    def __init__(self) -> None:
        self._registry: dict[str, FakeTrajectory] = {}

    def register(self, path: Path, trajectory: FakeTrajectory) -> None:
        self._registry[path.resolve().as_posix().casefold()] = trajectory

    def load(self, path: str, top: str | None = None) -> FakeTrajectory:
        key = Path(path).resolve().as_posix().casefold()
        trajectory = self._registry[key]
        if top is None:
            return trajectory
        topology_key = Path(top).resolve().as_posix().casefold()
        topology = self._registry[topology_key].topology
        return FakeTrajectory(np.asarray(trajectory.xyz, dtype=np.float64), topology)


@pytest.fixture()
def fake_md_module() -> FakeMdtraj:
    """Provide a fake mdtraj module with a reusable topology."""

    chain = FakeChain(index=0)
    residues = [
        FakeResidue(name="ALA", index=0, resSeq=1, chain=chain, is_protein=True),
        FakeResidue(name="ALA", index=0, resSeq=1, chain=chain, is_protein=True),
        FakeResidue(name="LIG", index=1, resSeq=2, chain=chain),
        FakeResidue(name="HOH", index=2, resSeq=10, chain=chain, is_water=True),
        FakeResidue(name="HOH", index=3, resSeq=11, chain=chain, is_water=True),
    ]
    atoms = [
        FakeAtom(index=0, name="CA", residue=residues[0], element=FakeElement("C", 12.0), serial=1),
        FakeAtom(index=1, name="CB", residue=residues[1], element=FakeElement("C", 12.0), serial=2),
        FakeAtom(index=2, name="C1", residue=residues[2], element=FakeElement("C", 12.0), serial=3),
        FakeAtom(index=3, name="O", residue=residues[3], element=FakeElement("O", 16.0), serial=4),
        FakeAtom(index=4, name="O", residue=residues[4], element=FakeElement("O", 16.0), serial=5),
    ]
    topology = FakeTopology(atoms)
    topology_traj = FakeTrajectory(np.zeros((1, len(atoms), 3), dtype=np.float64), topology)

    fake_md = FakeMdtraj()
    fake_md.topology = topology_traj
    return fake_md


def _make_xyz(ligand_x: list[float], *, water_near_x: float = 0.7, water_far_x: float = 8.0) -> np.ndarray:
    """Build a tiny deterministic trajectory geometry."""

    xyz = np.zeros((len(ligand_x), 5, 3), dtype=np.float64)
    xyz[:, 0, :] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    xyz[:, 1, :] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    xyz[:, 2, 0] = np.asarray(ligand_x, dtype=np.float64)
    xyz[:, 3, 0] = water_near_x
    xyz[:, 4, 0] = water_far_x
    return xyz


def _register_fake_inputs(workspace_tmp_dir: Path, fake_md_module: FakeMdtraj, names_to_xyz: dict[str, np.ndarray]) -> tuple[Path, dict[str, Path]]:
    """Register topology and trajectory paths in the fake mdtraj registry."""

    topology_path = workspace_tmp_dir / "topology.pdb"
    topology_path.write_text("fake topology\n", encoding="utf-8")
    fake_md_module.register(topology_path, fake_md_module.topology)

    trajectory_paths: dict[str, Path] = {}
    for name, xyz in names_to_xyz.items():
        path = workspace_tmp_dir / f"{name}.dcd"
        path.write_text("fake trajectory\n", encoding="utf-8")
        fake_md_module.register(path, FakeTrajectory(xyz, fake_md_module.topology.topology))
        trajectory_paths[name] = path
    return topology_path, trajectory_paths


def test_final_window_logic_and_water_storage(monkeypatch, workspace_tmp_dir: Path, fake_md_module: FakeMdtraj) -> None:
    """Labeling should depend only on the final fixed frame window and store water metadata by default."""

    topology_path, trajectory_paths = _register_fake_inputs(
        workspace_tmp_dir,
        fake_md_module,
        {"traj_window": _make_xyz([8.0, 8.0, 8.0, 0.5, 0.5])},
    )
    monkeypatch.setattr("mltsa.md.label._import_mdtraj", lambda: fake_md_module)

    result = label_trajectories(
        trajectory_paths=[trajectory_paths["traj_window"]],
        topology=topology_path,
        h5_path=workspace_tmp_dir / "labels.h5",
        experiment_id="md_labels",
        rule="sum_distances",
        selection_pairs=[("index 0", "index 2"), ("index 1", "index 2")],
        lower_threshold=2.0,
        upper_threshold=5.0,
        window_size=2,
    )

    assert len(result.processed) == 1
    entry = result.processed[0]
    assert entry.label == "IN"
    assert entry.final_window_start == 3
    np.testing.assert_allclose(entry.frame_values, np.array([1.0, 1.0]))
    assert entry.nearby_waters is not None
    assert entry.nearby_waters.residue_ids[0] == 10
    assert result.state_counts == {"IN": 1, "OUT": 0, "TS": 0}

    with open_h5(workspace_tmp_dir / "labels.h5", "r") as handle:
        store = handle[f"{results_experiment_path('md_labels')}/trajectory_labels"]
        entries = store["entries"]
        stored_entry = entries[entry.entry_key]
        waters = stored_entry["nearby_waters"]

        assert bool(stored_entry.attrs["waters_stored"]) is True
        assert read_utf8_array(stored_entry, "component_labels") == ["index 0 -> index 2", "index 1 -> index 2"]
        np.testing.assert_allclose(stored_entry["frame_values"][...], np.array([1.0, 1.0]))
        np.testing.assert_allclose(waters["mean_distances"][...], entry.nearby_waters.mean_distances)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["state_counts"] == {"IN": 1, "OUT": 0, "TS": 0}
    assert result.pie_chart_path.exists()


def test_append_uses_normalized_paths_and_skips_existing(monkeypatch, workspace_tmp_dir: Path, fake_md_module: FakeMdtraj) -> None:
    """Appending should identify existing entries from normalized paths plus replica ids."""

    topology_path, trajectory_paths = _register_fake_inputs(
        workspace_tmp_dir,
        fake_md_module,
        {
            "traj_same": _make_xyz([8.0, 8.0, 8.0, 0.5, 0.5]),
            "traj_new": _make_xyz([8.0, 8.0, 8.0, 7.5, 7.5]),
        },
    )
    monkeypatch.setattr("mltsa.md.label._import_mdtraj", lambda: fake_md_module)

    h5_path = workspace_tmp_dir / "append_labels.h5"
    label_trajectories(
        trajectory_paths=[trajectory_paths["traj_same"]],
        topology=topology_path,
        h5_path=h5_path,
        experiment_id="append_case",
        rule="sum_distances",
        selection_pairs=[("index 0", "index 2")],
        lower_threshold=1.0,
        upper_threshold=5.0,
        window_size=2,
        replica_ids=["replica_a"],
    )

    relative_same = trajectory_paths["traj_same"].relative_to(Path.cwd())
    result = label_trajectories(
        trajectory_paths=[relative_same, trajectory_paths["traj_new"]],
        topology=topology_path,
        h5_path=h5_path,
        experiment_id="append_case",
        rule="sum_distances",
        selection_pairs=[("index 0", "index 2")],
        lower_threshold=1.0,
        upper_threshold=5.0,
        window_size=2,
        replica_ids=["replica_a", "replica_b"],
        append=True,
    )

    assert len(result.processed) == 1
    assert len(result.skipped_entry_keys) == 1
    assert result.state_counts == {"IN": 1, "OUT": 1, "TS": 0}

    with open_h5(h5_path, "r") as handle:
        entries = handle[f"{results_experiment_path('append_case')}/trajectory_labels/entries"]
        assert len(entries.keys()) == 2


def test_overwrite_relabels_existing_entry(monkeypatch, workspace_tmp_dir: Path, fake_md_module: FakeMdtraj) -> None:
    """overwrite=True should replace an existing entry instead of silently skipping it."""

    topology_path, trajectory_paths = _register_fake_inputs(
        workspace_tmp_dir,
        fake_md_module,
        {"traj_replace": _make_xyz([8.0, 8.0, 8.0, 0.5, 0.5])},
    )
    monkeypatch.setattr("mltsa.md.label._import_mdtraj", lambda: fake_md_module)

    h5_path = workspace_tmp_dir / "overwrite_labels.h5"
    label_trajectories(
        trajectory_paths=[trajectory_paths["traj_replace"]],
        topology=topology_path,
        h5_path=h5_path,
        experiment_id="overwrite_case",
        rule="sum_distances",
        selection_pairs=[("index 0", "index 2")],
        lower_threshold=1.0,
        upper_threshold=5.0,
        window_size=2,
        replica_ids=["replica_x"],
    )

    fake_md_module.register(trajectory_paths["traj_replace"], FakeTrajectory(_make_xyz([8.0, 8.0, 8.0, 7.5, 7.5]), fake_md_module.topology.topology))
    result = label_trajectories(
        trajectory_paths=[trajectory_paths["traj_replace"]],
        topology=topology_path,
        h5_path=h5_path,
        experiment_id="overwrite_case",
        rule="sum_distances",
        selection_pairs=[("index 0", "index 2")],
        lower_threshold=1.0,
        upper_threshold=5.0,
        window_size=2,
        replica_ids=["replica_x"],
        append=True,
        overwrite=True,
    )

    assert len(result.processed) == 1
    assert len(result.overwritten_entry_keys) == 1
    assert result.processed[0].label == "OUT"

    with open_h5(h5_path, "r") as handle:
        entry = next(iter(handle[f"{results_experiment_path('overwrite_case')}/trajectory_labels/entries"].values()))
        assert entry.attrs["state_label"] == "OUT"


def test_com_distance_rule_and_snapshot_export_do_not_crash(monkeypatch, workspace_tmp_dir: Path, fake_md_module: FakeMdtraj) -> None:
    """COM-distance labeling and optional snapshot export should work on a tiny mocked case."""

    topology_path, trajectory_paths = _register_fake_inputs(
        workspace_tmp_dir,
        fake_md_module,
        {
            "traj_in": _make_xyz([8.0, 8.0, 8.0, 0.2, 0.2]),
            "traj_ts": _make_xyz([8.0, 8.0, 8.0, 3.0, 3.0]),
            "traj_out": _make_xyz([8.0, 8.0, 8.0, 7.5, 7.5]),
        },
    )
    monkeypatch.setattr("mltsa.md.label._import_mdtraj", lambda: fake_md_module)

    result = label_trajectories(
        trajectory_paths=[trajectory_paths["traj_in"], trajectory_paths["traj_ts"], trajectory_paths["traj_out"]],
        topology=topology_path,
        h5_path=workspace_tmp_dir / "snapshot_labels.h5",
        experiment_id="snapshot_case",
        rule="com_distance",
        selection_pairs=[("protein", "resname LIG")],
        lower_threshold=1.0,
        upper_threshold=5.0,
        window_size=2,
        replica_ids=["rep_in", "rep_ts", "rep_out"],
        export_snapshots=True,
        snapshot_dir=workspace_tmp_dir / "snapshots",
        center_snapshots=True,
        align_protein=True,
    )

    assert result.state_counts == {"IN": 1, "OUT": 1, "TS": 1}
    for state in ("IN", "OUT", "TS"):
        path = result.snapshot_paths[state]
        assert path.exists()
        assert "MODEL" in path.read_text(encoding="utf-8")
