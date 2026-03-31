"""Shared fake mdtraj helpers for MD unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    """Small subset of the mdtraj topology interface used by mltsa tests."""

    def __init__(self, atoms: list[FakeAtom]) -> None:
        self.atoms = atoms
        seen_residues: list[FakeResidue] = []
        seen_ids: set[int] = set()
        for atom in atoms:
            residue = atom.residue
            marker = id(residue)
            if marker not in seen_ids:
                seen_residues.append(residue)
                seen_ids.add(marker)
        self.residues = tuple(seen_residues)

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
    """Small trajectory container with the attributes needed by the MD modules."""

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


def build_fake_md_module() -> FakeMdtraj:
    """Construct a reusable fake mdtraj module and topology."""

    chain = FakeChain(index=0)
    residues = [
        FakeResidue(name="ALA", index=0, resSeq=1, chain=chain, is_protein=True),
        FakeResidue(name="GLY", index=1, resSeq=2, chain=chain, is_protein=True),
        FakeResidue(name="LIG", index=2, resSeq=3, chain=chain),
        FakeResidue(name="HOH", index=3, resSeq=10, chain=chain, is_water=True),
        FakeResidue(name="HOH", index=4, resSeq=11, chain=chain, is_water=True),
    ]
    atoms = [
        FakeAtom(index=0, name="CA", residue=residues[0], element=FakeElement("C", 12.0), serial=1),
        FakeAtom(index=1, name="CB", residue=residues[1], element=FakeElement("C", 12.0), serial=2),
        FakeAtom(index=2, name="C1", residue=residues[2], element=FakeElement("C", 12.0), serial=3),
        FakeAtom(index=3, name="O", residue=residues[3], element=FakeElement("O", 16.0), serial=4),
        FakeAtom(index=4, name="O", residue=residues[4], element=FakeElement("O", 16.0), serial=5),
    ]
    topology = FakeTopology(atoms)
    topology_trajectory = FakeTrajectory(np.zeros((1, len(atoms), 3), dtype=np.float64), topology)

    fake_md = FakeMdtraj()
    fake_md.topology = topology_trajectory
    return fake_md


def make_xyz(
    ligand_x: list[float],
    *,
    protein_a_x: float = 0.0,
    protein_b_x: float = 1.0,
    water_near_x: float = 0.7,
    water_far_x: float = 4.0,
) -> np.ndarray:
    """Build a tiny deterministic trajectory geometry for tests."""

    xyz = np.zeros((len(ligand_x), 5, 3), dtype=np.float64)
    xyz[:, 0, 0] = protein_a_x
    xyz[:, 1, 0] = protein_b_x
    xyz[:, 2, 0] = np.asarray(ligand_x, dtype=np.float64)
    xyz[:, 3, 0] = water_near_x
    xyz[:, 4, 0] = water_far_x
    return xyz


def register_fake_inputs(
    workspace_tmp_dir: Path,
    fake_md_module: FakeMdtraj,
    names_to_xyz: dict[str, np.ndarray],
) -> tuple[Path, dict[str, Path]]:
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


__all__ = [
    "FakeMdtraj",
    "build_fake_md_module",
    "make_xyz",
    "register_fake_inputs",
]
