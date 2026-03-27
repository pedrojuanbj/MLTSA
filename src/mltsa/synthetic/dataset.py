"""Public synthetic dataset API for the mltsa package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from mltsa.io.h5 import ensure_group, open_h5, read_utf8_array, replace_dataset, write_utf8_array

from .base import (
    FloatArray,
    GeneratedSyntheticData,
    IntArray,
    JSONValue,
    SYNTHETIC_GROUP,
    SYNTHETIC_SCHEMA,
    SYNTHETIC_SCHEMA_VERSION,
    as_float_array,
    as_int_array,
    canonical_json,
    clone_json,
    make_trajectory_seeds,
)
from .oned import create_1d_blueprint, generate_1d_data
from .twod import TwoDPattern, create_2d_blueprint, generate_2d_data

DatasetGenerator = Callable[[dict[str, Any], np.ndarray], GeneratedSyntheticData]


@dataclass(slots=True)
class SyntheticDataset:
    """Synthetic dataset plus all metadata required for lifecycle operations."""

    dataset_type: str
    X: FloatArray
    y: IntArray
    feature_names: tuple[str, ...]
    generation_params: dict[str, JSONValue]
    system_definition: dict[str, JSONValue]
    relevant_features: tuple[int, ...]
    time_relevance: FloatArray | None
    trajectory_seeds: IntArray
    latent_trajectories: FloatArray | None = None

    def __post_init__(self) -> None:
        """Normalize arrays and validate dataset consistency."""

        self.X = as_float_array(self.X)
        self.y = as_int_array(self.y)
        self.trajectory_seeds = as_int_array(self.trajectory_seeds)
        self.generation_params = clone_json(self.generation_params)
        self.system_definition = clone_json(self.system_definition)
        self.feature_names = tuple(self.feature_names)
        self.relevant_features = tuple(int(index) for index in self.relevant_features)

        if self.X.ndim != 3:
            raise ValueError("X must have shape (n_trajectories, n_steps, n_features).")
        if self.y.shape != (self.X.shape[0],):
            raise ValueError("y must have shape (n_trajectories,).")
        if self.trajectory_seeds.shape != (self.X.shape[0],):
            raise ValueError("trajectory_seeds must have shape (n_trajectories,).")
        if len(self.feature_names) != self.X.shape[2]:
            raise ValueError("feature_names must match the feature dimension of X.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique.")

        if self.time_relevance is not None:
            self.time_relevance = as_float_array(self.time_relevance)
            expected_shape = (self.X.shape[1], self.X.shape[2])
            if self.time_relevance.shape != expected_shape:
                raise ValueError(
                    f"time_relevance must have shape {expected_shape}, got {self.time_relevance.shape}."
                )

        if self.latent_trajectories is not None:
            self.latent_trajectories = as_float_array(self.latent_trajectories)
            if self.latent_trajectories.shape[:2] != self.X.shape[:2]:
                raise ValueError("latent_trajectories must match the trajectory and time dimensions of X.")

        invalid_indices = [index for index in self.relevant_features if index < 0 or index >= self.X.shape[2]]
        if invalid_indices:
            raise ValueError(f"Invalid relevant feature indices: {invalid_indices}")

    @property
    def n_trajectories(self) -> int:
        """Number of trajectories in the dataset."""

        return int(self.X.shape[0])

    @property
    def n_steps(self) -> int:
        """Number of time steps per trajectory."""

        return int(self.X.shape[1])

    @property
    def n_features(self) -> int:
        """Number of observed features."""

        return int(self.X.shape[2])

    @property
    def relevant_feature_names(self) -> tuple[str, ...]:
        """Feature names corresponding to the ground-truth relevant indices."""

        return tuple(self.feature_names[index] for index in self.relevant_features)

    @property
    def ground_truth_relevance(self) -> FloatArray:
        """Feature-level relevance summary for the dataset."""

        if self.time_relevance is not None:
            return as_float_array(self.time_relevance.mean(axis=0))

        relevance = np.zeros(self.n_features, dtype=np.float64)
        relevance[list(self.relevant_features)] = 1.0
        return relevance

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Persist the dataset to an HDF5 file."""

        file_path = Path(path)
        mode = "w" if overwrite else "x"
        with open_h5(file_path, mode) as handle:
            self._write_to_handle(handle)
        return file_path

    @classmethod
    def load(cls, path: str | Path) -> SyntheticDataset:
        """Load a synthetic dataset from HDF5."""

        with open_h5(path, "r") as handle:
            root = _require_synthetic_group(handle)
            generation_params = json.loads(str(root.attrs["generation_params_json"]))
            system_definition = json.loads(str(root.attrs["system_definition_json"]))
            time_relevance = root["time_relevance"][...] if "time_relevance" in root else None
            latent = root["latent_trajectories"][...] if "latent_trajectories" in root else None

            return cls(
                dataset_type=str(root.attrs["dataset_type"]),
                X=root["X"][...],
                y=root["y"][...],
                feature_names=tuple(read_utf8_array(root, "feature_names")),
                generation_params=generation_params,
                system_definition=system_definition,
                relevant_features=tuple(int(value) for value in root["relevant_features"][...].tolist()),
                time_relevance=time_relevance,
                trajectory_seeds=root["trajectory_seeds"][...],
                latent_trajectories=latent,
            )

    def rebuild_exact(self) -> SyntheticDataset:
        """Regenerate the exact dataset using the stored system definition and seeds."""

        generated = _generator_for_type(self.dataset_type)(self.system_definition, self.trajectory_seeds)
        return self._from_generated(generated, n_trajectories=self.n_trajectories)

    def generate_more(self, n_trajectories: int) -> SyntheticDataset:
        """Generate more trajectories from the same synthetic system definition."""

        if n_trajectories < 1:
            raise ValueError("n_trajectories must be at least 1.")

        if self.trajectory_seeds.size:
            seed_start = int(self.trajectory_seeds.max()) + 1
        else:
            seed_start = int(self.system_definition["base_seed"])

        seeds = np.arange(seed_start, seed_start + n_trajectories, dtype=np.int64)
        generated = _generator_for_type(self.dataset_type)(self.system_definition, seeds)
        return self._from_generated(generated, n_trajectories=n_trajectories)

    def append(self, other: SyntheticDataset) -> SyntheticDataset:
        """Append another compatible synthetic dataset in memory."""

        self._assert_compatible(other)
        _assert_disjoint_seeds(self.trajectory_seeds, other.trajectory_seeds)

        latent = None
        if self.latent_trajectories is not None and other.latent_trajectories is not None:
            latent = np.concatenate([self.latent_trajectories, other.latent_trajectories], axis=0)

        return SyntheticDataset(
            dataset_type=self.dataset_type,
            X=np.concatenate([self.X, other.X], axis=0),
            y=np.concatenate([self.y, other.y], axis=0),
            feature_names=self.feature_names,
            generation_params=self._generation_params_for_count(self.n_trajectories + other.n_trajectories),
            system_definition=self.system_definition,
            relevant_features=self.relevant_features,
            time_relevance=self.time_relevance,
            trajectory_seeds=np.concatenate([self.trajectory_seeds, other.trajectory_seeds], axis=0),
            latent_trajectories=latent,
        )

    def append_to_file(self, path: str | Path) -> Path:
        """Append the dataset to an existing on-disk synthetic dataset, or create it."""

        file_path = Path(path)
        if not file_path.exists():
            return self.save(file_path)

        with open_h5(file_path, "a") as handle:
            if SYNTHETIC_GROUP not in handle:
                self._write_to_handle(handle)
                return file_path

            root = _require_synthetic_group(handle)
            self._assert_file_compatible(root)
            stored_seeds = as_int_array(root["trajectory_seeds"][...])
            _assert_disjoint_seeds(stored_seeds, self.trajectory_seeds)

            start = int(root["X"].shape[0])
            stop = start + self.n_trajectories

            _append_array_dataset(root["X"], self.X, start)
            _append_array_dataset(root["y"], self.y, start)
            _append_array_dataset(root["trajectory_seeds"], self.trajectory_seeds, start)

            if self.latent_trajectories is not None:
                _append_array_dataset(root["latent_trajectories"], self.latent_trajectories, start)

            root.attrs["n_trajectories"] = stop
            root.attrs["generation_params_json"] = canonical_json(self._generation_params_for_count(stop))

        return file_path

    def _from_generated(self, generated: GeneratedSyntheticData, *, n_trajectories: int) -> SyntheticDataset:
        """Create a new dataset from generated arrays while preserving system metadata."""

        return SyntheticDataset(
            dataset_type=self.dataset_type,
            X=generated.X,
            y=generated.y,
            feature_names=self.feature_names,
            generation_params=self._generation_params_for_count(n_trajectories),
            system_definition=self.system_definition,
            relevant_features=self.relevant_features,
            time_relevance=self.time_relevance,
            trajectory_seeds=generated.trajectory_seeds,
            latent_trajectories=generated.latent_trajectories,
        )

    def _generation_params_for_count(self, n_trajectories: int) -> dict[str, JSONValue]:
        """Return generation params updated for a new trajectory count."""

        params = clone_json(self.generation_params)
        params["n_trajectories"] = int(n_trajectories)
        return params

    def _write_to_handle(self, handle: h5py.File) -> None:
        """Write the dataset into an open HDF5 file."""

        root = ensure_group(
            handle,
            SYNTHETIC_GROUP,
            attrs={
                "schema": SYNTHETIC_SCHEMA,
                "schema_version": SYNTHETIC_SCHEMA_VERSION,
                "dataset_type": self.dataset_type,
                "generation_params_json": canonical_json(self.generation_params),
                "system_definition_json": canonical_json(self.system_definition),
                "n_trajectories": self.n_trajectories,
            },
        )

        replace_dataset(root, "X", self.X, maxshape=(None, self.n_steps, self.n_features), chunks=True)
        replace_dataset(root, "y", self.y, maxshape=(None,), chunks=True)
        replace_dataset(root, "trajectory_seeds", self.trajectory_seeds, maxshape=(None,), chunks=True)
        replace_dataset(root, "relevant_features", np.asarray(self.relevant_features, dtype=np.int64))
        write_utf8_array(root, "feature_names", self.feature_names, overwrite=True)

        if self.time_relevance is not None:
            replace_dataset(root, "time_relevance", self.time_relevance)
        if self.latent_trajectories is not None:
            replace_dataset(
                root,
                "latent_trajectories",
                self.latent_trajectories,
                maxshape=(None,) + self.latent_trajectories.shape[1:],
                chunks=True,
            )

    def _assert_compatible(self, other: SyntheticDataset) -> None:
        """Validate that two datasets share the same system definition."""

        if self.dataset_type != other.dataset_type:
            raise ValueError("Datasets have different synthetic types and cannot be appended.")
        if self.feature_names != other.feature_names:
            raise ValueError("Datasets have different feature names and cannot be appended.")
        if self.relevant_features != other.relevant_features:
            raise ValueError("Datasets have different relevant feature definitions.")
        if canonical_json(self.system_definition) != canonical_json(other.system_definition):
            raise ValueError("Datasets come from different system definitions and cannot be appended.")
        if (self.time_relevance is None) != (other.time_relevance is None):
            raise ValueError("Datasets disagree on time-dependent relevance availability.")
        if self.time_relevance is not None and not np.allclose(self.time_relevance, other.time_relevance):
            raise ValueError("Datasets have different time-dependent relevance metadata.")
        if (self.latent_trajectories is None) != (other.latent_trajectories is None):
            raise ValueError("Datasets disagree on latent trajectory availability.")

    def _assert_file_compatible(self, root: h5py.Group) -> None:
        """Validate that an on-disk dataset matches this dataset's system."""

        if str(root.attrs.get("schema", "")) != SYNTHETIC_SCHEMA:
            raise ValueError("The target file does not contain an mltsa synthetic dataset.")
        if str(root.attrs["dataset_type"]) != self.dataset_type:
            raise ValueError("The target file contains a different synthetic dataset type.")
        if tuple(read_utf8_array(root, "feature_names")) != self.feature_names:
            raise ValueError("The target file contains different feature names.")
        if tuple(int(value) for value in root["relevant_features"][...].tolist()) != self.relevant_features:
            raise ValueError("The target file contains different relevant feature metadata.")

        stored_definition = json.loads(str(root.attrs["system_definition_json"]))
        if canonical_json(stored_definition) != canonical_json(self.system_definition):
            raise ValueError("The target file comes from a different system definition.")

        has_time_relevance = "time_relevance" in root
        if has_time_relevance != (self.time_relevance is not None):
            raise ValueError("The target file disagrees on time-dependent relevance availability.")
        if has_time_relevance and not np.allclose(root["time_relevance"][...], self.time_relevance):
            raise ValueError("The target file has different time-dependent relevance metadata.")

        has_latent = "latent_trajectories" in root
        if has_latent != (self.latent_trajectories is not None):
            raise ValueError("The target file disagrees on latent trajectory availability.")


def make_1d_dataset(
    n_trajectories: int,
    *,
    n_steps: int = 64,
    n_features: int = 12,
    n_relevant: int = 3,
    base_seed: int = 1234,
    ar_coefficient: float = 0.92,
    latent_noise: float = 0.10,
    relevant_noise: float = 0.07,
    background_noise: float = 0.25,
) -> SyntheticDataset:
    """Create a deterministic one-dimensional synthetic dataset."""

    blueprint = create_1d_blueprint(
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        n_features=n_features,
        n_relevant=n_relevant,
        base_seed=base_seed,
        ar_coefficient=ar_coefficient,
        latent_noise=latent_noise,
        relevant_noise=relevant_noise,
        background_noise=background_noise,
    )
    seeds = make_trajectory_seeds(base_seed, n_trajectories)
    generated = generate_1d_data(blueprint.system_definition, seeds)
    return SyntheticDataset(
        dataset_type=blueprint.dataset_type,
        X=generated.X,
        y=generated.y,
        feature_names=blueprint.feature_names,
        generation_params=blueprint.generation_params,
        system_definition=blueprint.system_definition,
        relevant_features=blueprint.relevant_features,
        time_relevance=blueprint.time_relevance,
        trajectory_seeds=generated.trajectory_seeds,
        latent_trajectories=generated.latent_trajectories,
    )


def make_2d_dataset(
    n_trajectories: int,
    *,
    n_steps: int = 80,
    n_features: int = 18,
    base_seed: int = 4321,
    pattern: TwoDPattern = "spiral",
    latent_noise: float = 0.08,
    feature_noise: float = 0.05,
) -> SyntheticDataset:
    """Create a deterministic two-dimensional projected synthetic dataset."""

    blueprint = create_2d_blueprint(
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        n_features=n_features,
        base_seed=base_seed,
        pattern=pattern,
        latent_noise=latent_noise,
        feature_noise=feature_noise,
    )
    seeds = make_trajectory_seeds(base_seed, n_trajectories)
    generated = generate_2d_data(blueprint.system_definition, seeds)
    return SyntheticDataset(
        dataset_type=blueprint.dataset_type,
        X=generated.X,
        y=generated.y,
        feature_names=blueprint.feature_names,
        generation_params=blueprint.generation_params,
        system_definition=blueprint.system_definition,
        relevant_features=blueprint.relevant_features,
        time_relevance=blueprint.time_relevance,
        trajectory_seeds=generated.trajectory_seeds,
        latent_trajectories=generated.latent_trajectories,
    )


def load_dataset(path: str | Path) -> SyntheticDataset:
    """Load a saved synthetic dataset from disk."""

    return SyntheticDataset.load(path)


def _generator_for_type(dataset_type: str) -> DatasetGenerator:
    """Resolve the generator function for a synthetic dataset type."""

    if dataset_type == "1d":
        return generate_1d_data
    if dataset_type == "2d":
        return generate_2d_data
    raise ValueError(f"Unsupported synthetic dataset type: {dataset_type!r}")


def _require_synthetic_group(handle: h5py.File) -> h5py.Group:
    """Validate and return the synthetic dataset root group."""

    root = handle.get(SYNTHETIC_GROUP, default=None)
    if not isinstance(root, h5py.Group):
        raise ValueError("The file does not contain a saved mltsa synthetic dataset.")
    return root


def _assert_disjoint_seeds(first: np.ndarray, second: np.ndarray) -> None:
    """Reject dataset appends that would duplicate trajectory seeds."""

    overlap = set(as_int_array(first).tolist()).intersection(as_int_array(second).tolist())
    if overlap:
        raise ValueError("Synthetic datasets contain overlapping trajectory seeds and cannot be appended.")


def _append_array_dataset(dataset: h5py.Dataset, values: np.ndarray, start: int) -> None:
    """Resize and append values to the first dimension of an HDF5 dataset."""

    stop = start + int(values.shape[0])
    dataset.resize((stop,) + dataset.shape[1:])
    dataset[start:stop, ...] = values


__all__ = ["SyntheticDataset", "load_dataset", "make_1d_dataset", "make_2d_dataset"]
