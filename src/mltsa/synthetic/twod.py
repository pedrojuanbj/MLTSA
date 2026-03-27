"""Deterministic two-dimensional synthetic trajectory generation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .base import (
    GeneratedSyntheticData,
    SyntheticBlueprint,
    as_float_array,
    as_int_array,
    clone_json,
    make_feature_names,
)

TwoDPattern = Literal["spiral", "zshape"]


def create_2d_blueprint(
    *,
    n_trajectories: int,
    n_steps: int = 80,
    n_features: int = 18,
    base_seed: int = 4321,
    pattern: TwoDPattern = "spiral",
    latent_noise: float = 0.08,
    feature_noise: float = 0.05,
) -> SyntheticBlueprint:
    """Create the system definition for a 2D projected synthetic dataset."""

    if n_trajectories < 1:
        raise ValueError("n_trajectories must be at least 1.")
    if n_steps < 3:
        raise ValueError("n_steps must be at least 3.")
    if n_features < 2:
        raise ValueError("n_features must be at least 2.")
    if pattern not in {"spiral", "zshape"}:
        raise ValueError("pattern must be 'spiral' or 'zshape'.")

    rng = np.random.default_rng(base_seed)
    projection_angles = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    feature_biases = rng.normal(0.0, 0.04, size=n_features)

    time_axis = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    x_profile = 0.25 + 1.35 * (time_axis**1.2)
    y_profile = _y_profile(pattern, time_axis)

    discriminative_strength = np.abs(np.cos(projection_angles))[None, :] * x_profile[:, None]
    row_sums = discriminative_strength.sum(axis=1, keepdims=True)
    time_relevance = np.divide(
        discriminative_strength,
        row_sums,
        out=np.zeros_like(discriminative_strength),
        where=row_sums > 0,
    )

    n_relevant = max(1, min(n_features, max(2, n_features // 4)))
    ranking = np.argsort(np.abs(np.cos(projection_angles)))[::-1]
    relevant_features = tuple(sorted(int(index) for index in ranking[:n_relevant]))

    generation_params = clone_json(
        {
            "n_trajectories": n_trajectories,
            "n_steps": n_steps,
            "n_features": n_features,
            "base_seed": base_seed,
            "pattern": pattern,
            "latent_noise": latent_noise,
            "feature_noise": feature_noise,
        }
    )
    system_definition = clone_json(
        {
            "kind": "2d",
            "base_seed": base_seed,
            "n_steps": n_steps,
            "n_features": n_features,
            "pattern": pattern,
            "projection_angles": projection_angles.tolist(),
            "feature_biases": feature_biases.tolist(),
            "x_profile": x_profile.tolist(),
            "y_profile": y_profile.tolist(),
            "latent_noise": latent_noise,
            "feature_noise": feature_noise,
            "relevant_features": list(relevant_features),
        }
    )
    return SyntheticBlueprint(
        dataset_type="2d",
        generation_params=generation_params,
        system_definition=system_definition,
        feature_names=make_feature_names("twod_feature", n_features),
        relevant_features=relevant_features,
        time_relevance=as_float_array(time_relevance),
    )


def generate_2d_data(system_definition: dict[str, Any], trajectory_seeds: np.ndarray) -> GeneratedSyntheticData:
    """Generate deterministic 2D-projected synthetic data for the given seeds."""

    n_steps = int(system_definition["n_steps"])
    n_features = int(system_definition["n_features"])
    pattern = str(system_definition["pattern"])
    angles = as_float_array(system_definition["projection_angles"])
    feature_biases = as_float_array(system_definition["feature_biases"])
    x_profile = as_float_array(system_definition["x_profile"])
    y_profile = as_float_array(system_definition["y_profile"])
    latent_noise = float(system_definition["latent_noise"])
    feature_noise = float(system_definition["feature_noise"])

    seeds = as_int_array(trajectory_seeds)
    X = np.empty((len(seeds), n_steps, n_features), dtype=np.float64)
    y = np.empty(len(seeds), dtype=np.int64)
    latent = np.empty((len(seeds), n_steps, 2), dtype=np.float64)

    time_axis = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)

    for trajectory_index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        label = int(rng.integers(0, 2))
        y[trajectory_index] = label

        class_sign = 1.0 if label == 1 else -1.0
        phase = rng.uniform(0.0, 2.0 * np.pi)
        x_path = class_sign * x_profile + rng.normal(scale=latent_noise, size=n_steps)
        y_path = _nuisance_path(pattern, time_axis, phase) * y_profile + rng.normal(scale=latent_noise, size=n_steps)

        latent_path = np.stack([x_path, y_path], axis=1)
        latent[trajectory_index] = latent_path

        projected = np.empty((n_steps, n_features), dtype=np.float64)
        for feature_index in range(n_features):
            signal = np.cos(angles[feature_index]) * x_path + np.sin(angles[feature_index]) * y_path
            projected[:, feature_index] = (
                feature_biases[feature_index]
                + signal
                + rng.normal(scale=feature_noise, size=n_steps)
            )

        X[trajectory_index] = projected

    return GeneratedSyntheticData(
        X=as_float_array(X),
        y=as_int_array(y),
        trajectory_seeds=seeds,
        latent_trajectories=latent,
    )


def _y_profile(pattern: str, time_axis: np.ndarray) -> np.ndarray:
    """Build the time-varying nuisance scale for the selected 2D pattern."""

    if pattern == "spiral":
        return 0.7 + 0.3 * np.sin(2.0 * np.pi * time_axis) ** 2

    rising = np.where(time_axis < 0.5, 1.0 - 2.0 * time_axis, 2.0 * time_axis - 1.0)
    return 0.4 + 0.8 * np.abs(rising)


def _nuisance_path(pattern: str, time_axis: np.ndarray, phase: float) -> np.ndarray:
    """Build the class-independent nuisance coordinate for a trajectory."""

    if pattern == "spiral":
        return np.sin(2.0 * np.pi * time_axis + phase)

    shifted = (time_axis + phase / (2.0 * np.pi)) % 1.0
    return np.where(
        shifted < 1.0 / 3.0,
        1.0 - 3.0 * shifted,
        np.where(
            shifted < 2.0 / 3.0,
            -1.0 + 3.0 * (shifted - 1.0 / 3.0),
            1.0 - 3.0 * (shifted - 2.0 / 3.0),
        ),
    )


__all__ = ["TwoDPattern", "create_2d_blueprint", "generate_2d_data"]
