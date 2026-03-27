"""Smoke tests for the package skeleton."""

from __future__ import annotations

import numpy as np

import mltsa
from mltsa.io.h5 import open_h5, read_utf8_array, scan_h5, write_dataset, write_utf8_array
from mltsa.io.schema import append_experiment_group, ensure_results_layout


def test_imports_package() -> None:
    """The new public package should be importable."""
    assert mltsa.__version__ == "0.1.0"


def test_h5py_roundtrip_smoke(workspace_tmp_dir) -> None:
    """A minimal end-to-end HDF5 roundtrip should work through the public helpers."""

    h5_path = workspace_tmp_dir / "smoke_roundtrip.h5"

    with open_h5(h5_path, "a") as handle:
        ensure_results_layout(handle)
        experiment = append_experiment_group(handle, attrs={"kind": "smoke"})
        write_dataset(experiment, "metrics", np.asarray([[0.1, 0.2], [0.3, 0.4]]))
        write_utf8_array(experiment, "labels", ["train", "test"])

    with open_h5(h5_path, "r") as handle:
        scan = scan_h5(handle, root="/results")

        assert read_utf8_array(handle, "/results/experiments/experiment_0000/labels") == [
            "train",
            "test",
        ]
        assert scan.datasets["/results/experiments/experiment_0000/metrics"].shape == (2, 2)
        assert scan.groups["/results/experiments/experiment_0000"].attrs["kind"] == "smoke"
