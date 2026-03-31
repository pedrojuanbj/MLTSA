"""Helpers for reusing stored nearby-water metadata during featurization."""

from __future__ import annotations

from typing import Final

import h5py

WATERS_GROUP_NAME: Final[str] = "nearby_waters"


def read_stored_water_atom_indices(label_entry: h5py.Group) -> tuple[int, ...]:
    """Read stored nearby-water atom indices from a label entry when available."""

    waters = label_entry.get(WATERS_GROUP_NAME, default=None)
    if not isinstance(waters, h5py.Group):
        return ()

    atom_indices = waters.get("atom_indices", default=None)
    if not isinstance(atom_indices, h5py.Dataset):
        return ()

    return tuple(int(value) for value in atom_indices[...].tolist())


__all__ = ["WATERS_GROUP_NAME", "read_stored_water_atom_indices"]
