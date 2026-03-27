"""Deterministic one-dimensional synthetic trajectory generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import (
    GeneratedSyntheticData,
    SyntheticBlueprint,
    as_float_array,
    as_int_array,
    clone_json,
    make_feature_names,
)


def create_1d_blueprint(
    *,
    n_trajectories: int,
    n_steps: int = 64,
    n_features: int = 12,
    n_relevant: int = 3,
    base_seed: int = 1234,
    ar_coefficient: float = 0.92,
    latent_noise: float = 0.10,
    relevant_noise: float = 0.07,
    background_noise: float = 0.25,
) -> SyntheticBlueprint:
    """Create the system definition for a 1D synthetic dataset."""

    if n_trajectories < 1:
        raise ValueError("n_trajectories must be at least 1.")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2.")
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")
    if not 1 <= n_relevant <= n_features:
        raise ValueError("n_relevant must be between 1 and n_features.")

    rng = np.random.default_rng(base_seed)
    relevant_features = tuple(sorted(rng.choice(n_features, size=n_relevant, replace=False).tolist()))

    feature_weights = np.zeros(n_features, dtype=np.float64)
    feature_weights[list(relevant_features)] = rng.uniform(0.9, 1.4, size=n_relevant)
    feature_biases = rng.normal(0.0, 0.05, size=n_features)

    time_axis = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    relevance_profile = np.clip(time_axis**1.3 + 0.1, 0.1, None)

    time_relevance = relevance_profile[:, None] * np.abs(feature_weights)[None, :]
    row_sums = time_relevance.sum(axis=1, keepdims=True)
    time_relevance = np.divide(
        time_relevance,
        row_sums,
        out=np.zeros_like(time_relevance),
        where=row_sums > 0,
    )

    generation_params = clone_json(
        {
            "n_trajectories": n_trajectories,
            "n_steps": n_steps,
            "n_features": n_features,
            "n_relevant": n_relevant,
            "base_seed": base_seed,
            "ar_coefficient": ar_coefficient,
            "latent_noise": latent_noise,
            "relevant_noise": relevant_noise,
            "background_noise": background_noise,
        }
    )
    system_definition = clone_json(
        {
            "kind": "1d",
            "base_seed": base_seed,
            "n_steps": n_steps,
            "n_features": n_features,
            "relevant_features": list(relevant_features),
            "feature_weights": feature_weights.tolist(),
            "feature_biases": feature_biases.tolist(),
            "relevance_profile": relevance_profile.tolist(),
            "ar_coefficient": ar_coefficient,
            "latent_noise": latent_noise,
            "relevant_noise": relevant_noise,
            "background_noise": background_noise,
            "class_centers": [-1.0, 1.0],
        }
    )
    return SyntheticBlueprint(
        dataset_type="1d",
        generation_params=generation_params,
        system_definition=system_definition,
        feature_names=make_feature_names("oned_feature", n_features),
        relevant_features=relevant_features,
        time_relevance=as_float_array(time_relevance),
    )


def generate_1d_data(system_definition: dict[str, Any], trajectory_seeds: np.ndarray) -> GeneratedSyntheticData:
    """Generate deterministic 1D synthetic data for the given seeds."""

    n_steps = int(system_definition["n_steps"])
    n_features = int(system_definition["n_features"])
    relevant_features = set(int(index) for index in system_definition["relevant_features"])
    feature_weights = as_float_array(system_definition["feature_weights"])
    feature_biases = as_float_array(system_definition["feature_biases"])
    relevance_profile = as_float_array(system_definition["relevance_profile"])
    ar_coefficient = float(system_definition["ar_coefficient"])
    latent_noise = float(system_definition["latent_noise"])
    relevant_noise = float(system_definition["relevant_noise"])
    background_noise = float(system_definition["background_noise"])
    class_centers = as_float_array(system_definition["class_centers"])

    seeds = as_int_array(trajectory_seeds)
    X = np.empty((len(seeds), n_steps, n_features), dtype=np.float64)
    y = np.empty(len(seeds), dtype=np.int64)
    latent = np.empty((len(seeds), n_steps, 1), dtype=np.float64)

    for trajectory_index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        label = int(rng.integers(0, 2))
        target = float(class_centers[label])
        y[trajectory_index] = label

        latent_path = np.empty(n_steps, dtype=np.float64)
        latent_path[0] = rng.normal(loc=0.15 * target, scale=latent_noise)
        for step in range(1, n_steps):
            drift_target = target * relevance_profile[step]
            latent_path[step] = (
                ar_coefficient * latent_path[step - 1]
                + (1.0 - ar_coefficient) * drift_target
                + rng.normal(scale=latent_noise)
            )

        features = np.empty((n_steps, n_features), dtype=np.float64)
        for feature_index in range(n_features):
            if feature_index in relevant_features:
                signal = feature_weights[feature_index] * relevance_profile * latent_path
                noise = rng.normal(scale=relevant_noise, size=n_steps)
                features[:, feature_index] = feature_biases[feature_index] + signal + noise
            else:
                noise = rng.normal(scale=background_noise, size=n_steps)
                smooth = np.empty(n_steps, dtype=np.float64)
                smooth[0] = noise[0]
                for step in range(1, n_steps):
                    smooth[step] = 0.85 * smooth[step - 1] + noise[step]
                features[:, feature_index] = feature_biases[feature_index] + smooth

        X[trajectory_index] = features
        latent[trajectory_index, :, 0] = latent_path

    return GeneratedSyntheticData(
        X=as_float_array(X),
        y=as_int_array(y),
        trajectory_seeds=seeds,
        latent_trajectories=latent,
    )


__all__ = ["create_1d_blueprint", "generate_1d_data"]
