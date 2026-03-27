"""Schema helpers for planned md and results HDF5 layouts.

The layout is intentionally simple for the migration phase:

- ``/md/replicas/<replica_id>`` stores per-replica data and metadata
- ``/md/feature_sets/<feature_set_id>`` stores feature definitions and labels
- ``/results/experiments/<experiment_id>`` stores append-only experiment outputs
"""

from __future__ import annotations

from typing import Final

import h5py

from .h5 import H5Location, create_appendable_group, ensure_group, group_exists

MD_ROOT: Final[str] = "/md"
MD_REPLICAS_ROOT: Final[str] = "/md/replicas"
MD_FEATURE_SETS_ROOT: Final[str] = "/md/feature_sets"

RESULTS_ROOT: Final[str] = "/results"
RESULTS_EXPERIMENTS_ROOT: Final[str] = "/results/experiments"

SCHEMA_VERSION: Final[int] = 1


def ensure_md_layout(location: H5Location) -> h5py.Group:
    """Create the standard md layout if it does not already exist."""

    root = ensure_group(
        location,
        MD_ROOT,
        attrs={"schema": "mltsa.md", "schema_version": SCHEMA_VERSION},
    )
    ensure_group(location, MD_REPLICAS_ROOT)
    ensure_group(location, MD_FEATURE_SETS_ROOT)
    return root


def ensure_results_layout(location: H5Location) -> h5py.Group:
    """Create the standard results layout if it does not already exist."""

    root = ensure_group(
        location,
        RESULTS_ROOT,
        attrs={"schema": "mltsa.results", "schema_version": SCHEMA_VERSION},
    )
    ensure_group(location, RESULTS_EXPERIMENTS_ROOT)
    return root


def replica_path(replica_id: str) -> str:
    """Return the canonical group path for an MD replica."""

    return f"{MD_REPLICAS_ROOT}/{_validate_name(replica_id, kind='replica')}"


def feature_set_path(feature_set_id: str) -> str:
    """Return the canonical group path for an MD feature set."""

    return f"{MD_FEATURE_SETS_ROOT}/{_validate_name(feature_set_id, kind='feature set')}"


def results_experiment_path(experiment_id: str) -> str:
    """Return the canonical group path for a results experiment."""

    return f"{RESULTS_EXPERIMENTS_ROOT}/{_validate_name(experiment_id, kind='experiment')}"


def list_replicas(location: H5Location) -> tuple[str, ...]:
    """List known replica ids without reading any datasets."""

    return _list_children(location, MD_REPLICAS_ROOT)


def list_feature_sets(location: H5Location) -> tuple[str, ...]:
    """List known feature-set ids without reading any datasets."""

    return _list_children(location, MD_FEATURE_SETS_ROOT)


def list_experiments(location: H5Location) -> tuple[str, ...]:
    """List stored experiment groups without reading any datasets."""

    return _list_children(location, RESULTS_EXPERIMENTS_ROOT)


def replica_exists(location: H5Location, replica_id: str) -> bool:
    """Return ``True`` if a replica group already exists."""

    return group_exists(location, replica_path(replica_id))


def feature_set_exists(location: H5Location, feature_set_id: str) -> bool:
    """Return ``True`` if a feature-set group already exists."""

    return group_exists(location, feature_set_path(feature_set_id))


def append_experiment_group(
    location: H5Location,
    *,
    prefix: str = "experiment",
    width: int = 4,
    attrs: dict[str, object] | None = None,
) -> h5py.Group:
    """Append a new numbered experiment group under ``/results/experiments``."""

    ensure_results_layout(location)
    return create_appendable_group(
        location,
        RESULTS_EXPERIMENTS_ROOT,
        prefix=prefix,
        width=width,
        attrs=attrs,
    )


def _list_children(location: H5Location, path: str) -> tuple[str, ...]:
    """Return sorted child group names for a schema root."""

    node = location.get(path, default=None)
    if not isinstance(node, h5py.Group):
        return ()
    return tuple(sorted(node.keys()))


def _validate_name(value: str, *, kind: str) -> str:
    """Reject empty or nested names that would create ambiguous schema paths."""

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{kind} id must not be empty.")
    if "/" in cleaned:
        raise ValueError(f"{kind} id must not contain '/'.")
    return cleaned


__all__ = [
    "MD_FEATURE_SETS_ROOT",
    "MD_REPLICAS_ROOT",
    "MD_ROOT",
    "RESULTS_EXPERIMENTS_ROOT",
    "RESULTS_ROOT",
    "SCHEMA_VERSION",
    "append_experiment_group",
    "ensure_md_layout",
    "ensure_results_layout",
    "feature_set_exists",
    "feature_set_path",
    "list_experiments",
    "list_feature_sets",
    "list_replicas",
    "replica_exists",
    "replica_path",
    "results_experiment_path",
]
