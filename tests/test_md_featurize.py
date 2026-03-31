"""Tests for MD featurization and feature-set storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mltsa.io.h5 import open_h5, scan_h5
from mltsa.io.schema import feature_set_path, list_feature_sets
from mltsa.md import featurize_dataset, label_trajectories, load_dataset

from ._md_fakes import FakeMdtraj, build_fake_md_module, make_xyz, register_fake_inputs


@pytest.fixture()
def fake_md_module() -> FakeMdtraj:
    """Provide a fake mdtraj module with a reusable topology."""

    return build_fake_md_module()


def test_feature_set_creation_and_selected_load(monkeypatch, workspace_tmp_dir: Path, fake_md_module: FakeMdtraj) -> None:
    """Featurization should create one named feature set and load it back cleanly."""

    topology_path, trajectory_paths = register_fake_inputs(
        workspace_tmp_dir,
        fake_md_module,
        {
            "traj_a": make_xyz([0.2, 0.3, 0.4, 0.5]),
            "traj_b": make_xyz([2.0, 2.1, 2.2, 2.3]),
        },
    )
    monkeypatch.setattr("mltsa.md.label._import_mdtraj", lambda: fake_md_module)
    monkeypatch.setattr("mltsa.md.featurize._import_mdtraj", lambda: fake_md_module)

    h5_path = workspace_tmp_dir / "md_features.h5"
    label_trajectories(
        trajectory_paths=[trajectory_paths["traj_a"], trajectory_paths["traj_b"]],
        topology=topology_path,
        h5_path=h5_path,
        experiment_id="labels",
        rule="sum_distances",
        selection_pairs=[("index 0", "index 2")],
        lower_threshold=0.8,
        upper_threshold=1.5,
        window_size=2,
        replica_ids=["rep_a", "rep_b"],
    )

    dataset = featurize_dataset(
        h5_path=h5_path,
        feature_set="closest",
        feature_type="closest_residue_distances",
        label_experiment_id="labels",
    )

    assert dataset.feature_set == "closest"
    assert dataset.feature_type == "closest_residue_distances"
    assert dataset.replica_ids == ("rep_a", "rep_b")
    assert dataset.state_labels == ("IN", "OUT")
    assert dataset.X.shape == (2, 4, 2)
    assert dataset.feature_names == ("closest_residue:ALA001", "closest_residue:GLY002")

    with open_h5(h5_path, "r") as handle:
        scan = scan_h5(handle, root=feature_set_path("closest"))
        assert scan.groups[feature_set_path("closest")].groups == ("replicas",)
        assert scan.datasets[f"{feature_set_path('closest')}/feature_names"].shape == (2,)
        assert scan.datasets[f"{feature_set_path('closest')}/replicas/rep_a/X"].shape == (4, 2)

    reloaded = load_dataset(h5_path, "closest")
    np.testing.assert_allclose(reloaded.X, dataset.X)
    assert reloaded.feature_names == dataset.feature_names


def test_append_skips_existing_and_only_processes_new_replicas(
    monkeypatch,
    workspace_tmp_dir: Path,
    fake_md_module: FakeMdtraj,
) -> None:
    """Appending should skip already-featurized replicas before loading trajectory arrays."""

    topology_path, trajectory_paths = register_fake_inputs(
        workspace_tmp_dir,
        fake_md_module,
        {
            "traj_a": make_xyz([0.2, 0.3, 0.4, 0.5]),
            "traj_b": make_xyz([2.0, 2.1, 2.2, 2.3]),
        },
    )
    monkeypatch.setattr("mltsa.md.label._import_mdtraj", lambda: fake_md_module)
    monkeypatch.setattr("mltsa.md.featurize._import_mdtraj", lambda: fake_md_module)

    h5_path = workspace_tmp_dir / "append_features.h5"
    label_trajectories(
        trajectory_paths=[trajectory_paths["traj_a"]],
        topology=topology_path,
        h5_path=h5_path,
        experiment_id="labels",
        rule="sum_distances",
        selection_pairs=[("index 0", "index 2")],
        lower_threshold=0.8,
        upper_threshold=1.5,
        window_size=2,
        replica_ids=["rep_a"],
    )
    featurize_dataset(
        h5_path=h5_path,
        feature_set="closest",
        feature_type="closest_residue_distances",
        label_experiment_id="labels",
    )

    label_trajectories(
        trajectory_paths=[trajectory_paths["traj_b"]],
        topology=topology_path,
        h5_path=h5_path,
        experiment_id="labels",
        rule="sum_distances",
        selection_pairs=[("index 0", "index 2")],
        lower_threshold=0.8,
        upper_threshold=1.5,
        window_size=2,
        replica_ids=["rep_b"],
        append=True,
    )

    load_calls: list[str] = []
    original_load = __import__("mltsa.md.featurize", fromlist=["_load_trajectory"])._load_trajectory

    def counting_load(md_module, trajectory_path: Path, topology_path_arg: Path):
        load_calls.append(Path(trajectory_path).name)
        return original_load(md_module, trajectory_path, topology_path_arg)

    monkeypatch.setattr("mltsa.md.featurize._load_trajectory", counting_load)
    result = featurize_dataset(
        h5_path=h5_path,
        feature_set="closest",
        feature_type="closest_residue_distances",
        label_experiment_id="labels",
        append=True,
    )

    assert result.processed_replica_ids == ("rep_b",)
    assert result.skipped_replica_ids == ("rep_a",)
    assert load_calls == ["traj_b.dcd"]


def test_second_feature_family_in_same_file_and_water_reuse(
    monkeypatch,
    workspace_tmp_dir: Path,
    fake_md_module: FakeMdtraj,
) -> None:
    """The same HDF5 should allow a second feature family while reusing stored water metadata."""

    topology_path, trajectory_paths = register_fake_inputs(
        workspace_tmp_dir,
        fake_md_module,
        {
            "traj_a": make_xyz([0.2, 0.3, 0.4, 0.5]),
            "traj_b": make_xyz([0.1, 0.2, 0.3, 0.4], water_near_x=0.35),
        },
    )
    monkeypatch.setattr("mltsa.md.label._import_mdtraj", lambda: fake_md_module)
    monkeypatch.setattr("mltsa.md.featurize._import_mdtraj", lambda: fake_md_module)

    h5_path = workspace_tmp_dir / "two_feature_sets.h5"
    label_trajectories(
        trajectory_paths=[trajectory_paths["traj_a"], trajectory_paths["traj_b"]],
        topology=topology_path,
        h5_path=h5_path,
        experiment_id="labels",
        rule="sum_distances",
        selection_pairs=[("index 0", "index 2")],
        lower_threshold=0.8,
        upper_threshold=1.5,
        window_size=2,
        replica_ids=["rep_a", "rep_b"],
    )

    featurize_dataset(
        h5_path=h5_path,
        feature_set="closest",
        feature_type="closest_residue_distances",
        label_experiment_id="labels",
    )
    bubble_dataset = featurize_dataset(
        h5_path=h5_path,
        feature_set="bubble",
        feature_type="bubble_distances",
        label_experiment_id="labels",
        use_waters=True,
        bubble_cutoff=0.25,
    )

    assert bubble_dataset.feature_set == "bubble"
    assert bubble_dataset.X.shape[0] == 2
    assert any("HOH010" in name or "HOH011" in name for name in bubble_dataset.feature_names)

    with open_h5(h5_path, "r") as handle:
        assert list_feature_sets(handle) == ("bubble", "closest")
        assert bool(handle[feature_set_path("bubble")].attrs["use_waters"]) is True

    closest_only = load_dataset(h5_path, "closest")
    assert closest_only.feature_set == "closest"
    assert closest_only.feature_type == "closest_residue_distances"
