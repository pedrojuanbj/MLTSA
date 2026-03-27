"""Tests for the synthetic dataset lifecycle."""

from __future__ import annotations

import numpy as np

from mltsa.synthetic import SyntheticDataset, load_dataset, make_1d_dataset, make_2d_dataset


def test_synthetic_save_load_roundtrip(workspace_tmp_dir) -> None:
    """Synthetic datasets should survive an HDF5 save/load roundtrip."""

    dataset = make_1d_dataset(n_trajectories=4, n_steps=24, n_features=6, n_relevant=2, base_seed=101)
    path = workspace_tmp_dir / "oned_roundtrip.h5"

    dataset.save(path)
    loaded = load_dataset(path)

    assert isinstance(loaded, SyntheticDataset)
    assert loaded.dataset_type == "1d"
    assert loaded.feature_names == dataset.feature_names
    assert loaded.relevant_features == dataset.relevant_features
    assert loaded.generation_params == dataset.generation_params
    assert loaded.system_definition == dataset.system_definition
    np.testing.assert_allclose(loaded.X, dataset.X)
    np.testing.assert_array_equal(loaded.y, dataset.y)
    np.testing.assert_array_equal(loaded.trajectory_seeds, dataset.trajectory_seeds)
    np.testing.assert_allclose(loaded.latent_trajectories, dataset.latent_trajectories)
    np.testing.assert_allclose(loaded.time_relevance, dataset.time_relevance)


def test_synthetic_rebuild_exact_consistency() -> None:
    """Rebuilding a dataset exactly should reproduce all saved arrays."""

    dataset = make_2d_dataset(n_trajectories=5, n_steps=30, n_features=8, base_seed=303, pattern="spiral")
    rebuilt = dataset.rebuild_exact()

    assert rebuilt.dataset_type == dataset.dataset_type
    assert rebuilt.system_definition == dataset.system_definition
    np.testing.assert_allclose(rebuilt.X, dataset.X)
    np.testing.assert_array_equal(rebuilt.y, dataset.y)
    np.testing.assert_array_equal(rebuilt.trajectory_seeds, dataset.trajectory_seeds)
    np.testing.assert_allclose(rebuilt.latent_trajectories, dataset.latent_trajectories)


def test_synthetic_generate_more_keeps_system_and_changes_seeds() -> None:
    """Generating more data should preserve the system definition with fresh trajectories."""

    dataset = make_1d_dataset(n_trajectories=3, n_steps=20, n_features=5, n_relevant=2, base_seed=707)
    more = dataset.generate_more(2)

    assert more.dataset_type == dataset.dataset_type
    assert more.system_definition == dataset.system_definition
    assert more.feature_names == dataset.feature_names
    assert more.n_trajectories == 2
    assert more.trajectory_seeds.min() > dataset.trajectory_seeds.max()
    assert not np.allclose(more.X, dataset.X[:2])


def test_synthetic_append_and_append_to_file(workspace_tmp_dir) -> None:
    """Synthetic datasets should append in memory and on disk."""

    dataset = make_2d_dataset(n_trajectories=4, n_steps=28, n_features=7, base_seed=909, pattern="zshape")
    more = dataset.generate_more(3)

    combined = dataset.append(more)
    assert combined.n_trajectories == 7
    np.testing.assert_allclose(combined.X[: dataset.n_trajectories], dataset.X)
    np.testing.assert_allclose(combined.X[dataset.n_trajectories :], more.X)
    np.testing.assert_array_equal(combined.trajectory_seeds[: dataset.n_trajectories], dataset.trajectory_seeds)
    np.testing.assert_array_equal(combined.trajectory_seeds[dataset.n_trajectories :], more.trajectory_seeds)

    path = workspace_tmp_dir / "appendable.h5"
    dataset.save(path)
    more.append_to_file(path)
    loaded = SyntheticDataset.load(path)

    assert loaded.n_trajectories == 7
    np.testing.assert_allclose(loaded.X, combined.X)
    np.testing.assert_array_equal(loaded.y, combined.y)
    np.testing.assert_array_equal(loaded.trajectory_seeds, combined.trajectory_seeds)


def test_synthetic_ground_truth_metadata_is_available() -> None:
    """Ground-truth relevance metadata should always be inspectable."""

    dataset = make_2d_dataset(n_trajectories=3, n_steps=18, n_features=6, base_seed=111)

    assert dataset.relevant_features
    assert dataset.relevant_feature_names
    assert dataset.ground_truth_relevance.shape == (dataset.n_features,)
    assert dataset.time_relevance is not None
    assert dataset.time_relevance.shape == (dataset.n_steps, dataset.n_features)
    assert np.all(dataset.ground_truth_relevance >= 0.0)
    assert float(dataset.ground_truth_relevance.sum()) > 0.0
