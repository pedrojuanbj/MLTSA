"""Reusable HDF5 helpers for the mltsa package.

The helpers in this module are intentionally small and metadata-focused. They
support safe file opening, dataset replacement, UTF-8 string storage, and
lightweight file scans that inspect shapes, dtypes, groups, and attributes
without reading full numeric arrays into memory.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TypeAlias

import h5py
import numpy as np

StrPath: TypeAlias = str | Path
H5Location: TypeAlias = h5py.File | h5py.Group
Attrs: TypeAlias = Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class H5GroupScan:
    """Metadata collected for a group during a lightweight scan."""

    path: str
    groups: tuple[str, ...]
    datasets: tuple[str, ...]
    attrs: dict[str, object]


@dataclass(frozen=True, slots=True)
class H5DatasetScan:
    """Metadata collected for a dataset during a lightweight scan."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    attrs: dict[str, object]


@dataclass(frozen=True, slots=True)
class H5Scan:
    """Flat metadata view of an HDF5 tree."""

    groups: dict[str, H5GroupScan]
    datasets: dict[str, H5DatasetScan]


@contextmanager
def open_h5(path: StrPath, mode: str = "r", *, track_order: bool = True) -> Iterator[h5py.File]:
    """Open an HDF5 file with sensible defaults for package workflows.

    Parent directories are created automatically for write-capable modes. The
    returned file handle is always closed when the context exits.
    """

    file_path = Path(path)
    if file_path.exists() and file_path.is_dir():
        raise IsADirectoryError(f"Expected an HDF5 file path, got directory: {file_path}")

    if any(flag in mode for flag in ("a", "w", "x", "+")):
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, mode, track_order=track_order) as handle:
        yield handle


def ensure_group(location: H5Location, path: str, *, attrs: Attrs = None) -> h5py.Group:
    """Create a group if needed and optionally update its attributes."""

    if path in {"", "."}:
        group = location
    else:
        group = location.require_group(path)

    _update_attrs(group.attrs, attrs)
    return group


def write_dataset(
    location: H5Location,
    path: str,
    data: Any,
    *,
    attrs: Attrs = None,
    overwrite: bool = False,
    **create_dataset_kwargs: Any,
) -> h5py.Dataset:
    """Create a dataset at ``path``.

    Parameters
    ----------
    location:
        File or group where the dataset should live.
    path:
        Absolute or relative HDF5 path.
    data:
        Dataset payload written directly to HDF5.
    attrs:
        Optional attributes attached after dataset creation.
    overwrite:
        When ``True``, replace an existing dataset or conflicting object.
    create_dataset_kwargs:
        Extra keyword arguments forwarded to ``create_dataset``.
    """

    parent, name = _resolve_parent(location, path)
    existing = parent.get(name)
    if existing is not None:
        if not overwrite:
            raise ValueError(f"Dataset already exists at {existing.name!r}.")
        del parent[name]

    dataset = parent.create_dataset(name, data=data, **create_dataset_kwargs)
    _update_attrs(dataset.attrs, attrs)
    return dataset


def replace_dataset(
    location: H5Location,
    path: str,
    data: Any,
    *,
    attrs: Attrs = None,
    **create_dataset_kwargs: Any,
) -> h5py.Dataset:
    """Replace a dataset in-place, creating parent groups as needed."""

    return write_dataset(
        location,
        path,
        data,
        attrs=attrs,
        overwrite=True,
        **create_dataset_kwargs,
    )


def create_appendable_group(
    location: H5Location,
    parent_path: str,
    *,
    prefix: str = "item",
    width: int = 4,
    attrs: Attrs = None,
) -> h5py.Group:
    """Create the next numbered child group under ``parent_path``.

    The function scans only child names, not dataset payloads. New groups follow
    the pattern ``{prefix}_{index}`` with zero-padded numeric suffixes.
    """

    if not prefix:
        raise ValueError("prefix must not be empty.")
    if "/" in prefix:
        raise ValueError("prefix must not contain '/'.")
    if width < 1:
        raise ValueError("width must be at least 1.")

    parent = ensure_group(location, parent_path)
    index = 0
    marker = f"{prefix}_"

    for name in parent.keys():
        if not name.startswith(marker):
            continue
        suffix = name[len(marker) :]
        if suffix.isdigit():
            index = max(index, int(suffix) + 1)

    name = f"{prefix}_{index:0{width}d}"
    group = parent.create_group(name, track_order=True)
    _update_attrs(group.attrs, attrs)
    return group


def write_utf8_array(
    location: H5Location,
    path: str,
    values: Sequence[str],
    *,
    attrs: Attrs = None,
    overwrite: bool = False,
) -> h5py.Dataset:
    """Store a one-dimensional UTF-8 string array."""

    dtype = h5py.string_dtype(encoding="utf-8")
    data = np.asarray(list(values), dtype=dtype)
    return write_dataset(location, path, data, attrs=attrs, overwrite=overwrite)


def read_utf8_array(location: H5Location, path: str) -> list[str]:
    """Read a UTF-8 string dataset back into Python strings."""

    dataset = location[path]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"Expected a dataset at {path!r}.")
    return [str(item) for item in dataset.asstr()[...].tolist()]


def group_exists(location: H5Location, path: str) -> bool:
    """Return ``True`` when ``path`` exists and is a group."""

    return isinstance(location.get(path, default=None), h5py.Group)


def dataset_exists(location: H5Location, path: str) -> bool:
    """Return ``True`` when ``path`` exists and is a dataset."""

    return isinstance(location.get(path, default=None), h5py.Dataset)


def scan_h5(location: H5Location, *, root: str | None = None) -> H5Scan:
    """Collect group, dataset, and attribute metadata without reading arrays."""

    target = _resolve_object(location, root)
    if isinstance(target, h5py.Dataset):
        return H5Scan(groups={}, datasets={target.name: _scan_dataset(target)})

    groups: dict[str, H5GroupScan] = {}
    datasets: dict[str, H5DatasetScan] = {}
    queue: list[h5py.Group] = [target]

    while queue:
        group = queue.pop()
        child_groups: list[str] = []
        child_datasets: list[str] = []

        for name, child in group.items():
            if isinstance(child, h5py.Group):
                child_groups.append(name)
                queue.append(child)
            elif isinstance(child, h5py.Dataset):
                child_datasets.append(name)
                datasets[child.name] = _scan_dataset(child)

        groups[group.name] = H5GroupScan(
            path=group.name,
            groups=tuple(sorted(child_groups)),
            datasets=tuple(sorted(child_datasets)),
            attrs=_scan_attrs(group.attrs),
        )

    return H5Scan(groups=groups, datasets=datasets)


def _resolve_parent(location: H5Location, path: str) -> tuple[h5py.Group, str]:
    """Resolve the parent group and final element name for an HDF5 path."""

    cleaned = path.rstrip("/")
    if cleaned in {"", "/"}:
        raise ValueError("path must refer to a child object, not the root group.")

    parts = cleaned.strip("/").split("/")
    name = parts[-1]
    parent_parts = parts[:-1]
    absolute = cleaned.startswith("/")

    if not parent_parts:
        parent = location.file["/"] if absolute else location
        return parent, name

    parent_path = ("/" if absolute else "") + "/".join(parent_parts)
    return ensure_group(location, parent_path), name


def _resolve_object(location: H5Location, root: str | None) -> h5py.Group | h5py.Dataset:
    """Resolve the object that should be scanned."""

    if root is None or root in {"", "."}:
        return location
    return location[root]


def _scan_dataset(dataset: h5py.Dataset) -> H5DatasetScan:
    """Collect dataset metadata without reading its payload."""

    return H5DatasetScan(
        path=dataset.name,
        shape=tuple(int(size) for size in dataset.shape),
        dtype=str(dataset.dtype),
        attrs=_scan_attrs(dataset.attrs),
    )


def _scan_attrs(attrs: h5py.AttributeManager) -> dict[str, object]:
    """Normalize HDF5 attributes into plain Python values."""

    return {name: _normalize_attr_value(value) for name, value in attrs.items()}


def _normalize_attr_value(value: Any) -> object:
    """Convert HDF5 and NumPy attribute values into plain Python objects."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "O", "U"}:
            return [_normalize_attr_value(item) for item in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _update_attrs(attrs: h5py.AttributeManager, values: Attrs) -> None:
    """Apply attribute updates when values are provided."""

    if not values:
        return

    for name, value in values.items():
        attrs[name] = value


__all__ = [
    "H5DatasetScan",
    "H5GroupScan",
    "H5Scan",
    "create_appendable_group",
    "dataset_exists",
    "ensure_group",
    "group_exists",
    "open_h5",
    "read_utf8_array",
    "replace_dataset",
    "scan_h5",
    "write_dataset",
    "write_utf8_array",
]
