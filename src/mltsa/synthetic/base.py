"""Shared types and metadata helpers for synthetic datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

SYNTHETIC_GROUP = "/synthetic"
SYNTHETIC_SCHEMA = "mltsa.synthetic"
SYNTHETIC_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SyntheticBlueprint:
    """System-level metadata needed to generate a synthetic dataset."""

    dataset_type: str
    generation_params: dict[str, JSONValue]
    system_definition: dict[str, JSONValue]
    feature_names: tuple[str, ...]
    relevant_features: tuple[int, ...]
    time_relevance: FloatArray | None


@dataclass(frozen=True, slots=True)
class GeneratedSyntheticData:
    """Generated synthetic observations for a fixed system definition."""

    X: FloatArray
    y: IntArray
    trajectory_seeds: IntArray
    latent_trajectories: FloatArray | None = None


def canonical_json(data: dict[str, JSONValue]) -> str:
    """Serialize JSON-compatible metadata deterministically."""

    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def clone_json(data: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Create a fully JSON-normalized copy of metadata."""

    return json.loads(canonical_json(data))


def as_float_array(data: Any) -> FloatArray:
    """Convert input data to a float64 NumPy array."""

    return np.asarray(data, dtype=np.float64)


def as_int_array(data: Any) -> IntArray:
    """Convert input data to an int64 NumPy array."""

    return np.asarray(data, dtype=np.int64)


def make_feature_names(prefix: str, n_features: int) -> tuple[str, ...]:
    """Build canonical feature names for a synthetic dataset."""

    return tuple(f"{prefix}_{index:03d}" for index in range(n_features))


def make_trajectory_seeds(base_seed: int, count: int, *, offset: int = 0) -> IntArray:
    """Build deterministic per-trajectory seeds from a base seed."""

    start = int(base_seed) + int(offset)
    return np.arange(start, start + int(count), dtype=np.int64)


__all__ = [
    "FloatArray",
    "GeneratedSyntheticData",
    "IntArray",
    "JSONValue",
    "SYNTHETIC_GROUP",
    "SYNTHETIC_SCHEMA",
    "SYNTHETIC_SCHEMA_VERSION",
    "SyntheticBlueprint",
    "as_float_array",
    "as_int_array",
    "canonical_json",
    "clone_json",
    "make_feature_names",
    "make_trajectory_seeds",
]
