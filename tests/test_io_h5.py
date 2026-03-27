"""Focused tests for the reusable HDF5 IO layer."""

from __future__ import annotations

import numpy as np

from mltsa.io.h5 import (
    open_h5,
    read_utf8_array,
    replace_dataset,
    scan_h5,
    write_dataset,
    write_utf8_array,
)
from mltsa.io.schema import (
    append_experiment_group,
    ensure_md_layout,
    ensure_results_layout,
    feature_set_exists,
    feature_set_path,
    list_experiments,
    list_feature_sets,
    list_replicas,
    replica_exists,
    replica_path,
)


def test_dataset_create_and_replace(workspace_tmp_dir) -> None:
    """Replacing a dataset should update shape, dtype, and attrs."""

    h5_path = workspace_tmp_dir / "replace.h5"

    with open_h5(h5_path, "a") as handle:
        replica_group = handle.require_group(replica_path("replica_0001"))
        write_dataset(replica_group, "frames", np.arange(3), attrs={"units": "frame"})

        replace_dataset(
            replica_group,
            "frames",
            np.linspace(0.0, 1.0, num=5),
            attrs={"units": "ps"},
        )

        dataset = replica_group["frames"]
        assert dataset.shape == (5,)
        assert str(dataset.dtype) == "float64"
        assert dataset.attrs["units"] == "ps"
        np.testing.assert_allclose(dataset[...], np.linspace(0.0, 1.0, num=5))


def test_utf8_string_roundtrip(workspace_tmp_dir) -> None:
    """UTF-8 string datasets should round-trip through HDF5 cleanly."""

    h5_path = workspace_tmp_dir / "strings.h5"
    names = ["Ca+", "Glycine", "torsion_phi"]

    with open_h5(h5_path, "a") as handle:
        write_utf8_array(handle, "/metadata/labels", names)
        assert read_utf8_array(handle, "/metadata/labels") == names


def test_lightweight_scan_and_schema_existence(workspace_tmp_dir) -> None:
    """Schema existence checks should work without loading dataset arrays."""

    h5_path = workspace_tmp_dir / "md_scan.h5"

    with open_h5(h5_path, "a") as handle:
        ensure_md_layout(handle)

        replica_group = handle.require_group(replica_path("replica_0001"))
        replica_group.attrs["temperature_k"] = 300

        write_dataset(
            handle,
            f"{replica_path('replica_0001')}/coordinates",
            np.arange(12).reshape(3, 4),
            attrs={"units": "nm"},
        )
        write_utf8_array(
            handle,
            f"{feature_set_path('backbone_angles')}/names",
            ["phi", "psi"],
            attrs={"kind": "angle_labels"},
        )

        scan = scan_h5(handle, root="/md")

        assert replica_exists(handle, "replica_0001")
        assert not replica_exists(handle, "replica_9999")
        assert feature_set_exists(handle, "backbone_angles")
        assert not feature_set_exists(handle, "distances")

        assert list_replicas(handle) == ("replica_0001",)
        assert list_feature_sets(handle) == ("backbone_angles",)
        assert scan.groups["/md"].groups == ("feature_sets", "replicas")
        assert scan.groups[replica_path("replica_0001")].attrs["temperature_k"] == 300
        assert scan.datasets[f"{replica_path('replica_0001')}/coordinates"].shape == (3, 4)
        assert scan.datasets[f"{feature_set_path('backbone_angles')}/names"].attrs["kind"] == "angle_labels"


def test_append_experiment_groups_in_results_file(workspace_tmp_dir) -> None:
    """Appending experiment groups should produce stable sequential names."""

    h5_path = workspace_tmp_dir / "results.h5"

    with open_h5(h5_path, "a") as handle:
        ensure_results_layout(handle)

        first = append_experiment_group(handle, attrs={"model": "mlp"})
        second = append_experiment_group(handle, attrs={"model": "gbdt"})
        write_dataset(first, "scores", np.asarray([0.7, 0.8, 0.9]))

        assert first.name == "/results/experiments/experiment_0000"
        assert second.name == "/results/experiments/experiment_0001"
        assert first.attrs["model"] == "mlp"
        assert second.attrs["model"] == "gbdt"
        assert list_experiments(handle) == ("experiment_0000", "experiment_0001")

    with open_h5(h5_path, "a") as handle:
        third = append_experiment_group(handle)
        assert third.name == "/results/experiments/experiment_0002"
        assert list_experiments(handle) == (
            "experiment_0000",
            "experiment_0001",
            "experiment_0002",
        )
