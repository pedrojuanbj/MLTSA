"""Tests for the MD analysis workflow and thin CLI wrappers."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mltsa.io.h5 import open_h5
from mltsa.io.schema import results_experiment_path
from mltsa.md import featurize_dataset, label_trajectories, run_mltsa
from mltsa.cli.main import main as root_cli_main
from mltsa.cli.md import main as md_cli_main

from ._md_fakes import FakeMdtraj, build_fake_md_module, make_xyz, register_fake_inputs


@pytest.fixture()
def fake_md_module() -> FakeMdtraj:
    """Provide a fake mdtraj module with a reusable topology."""

    return build_fake_md_module()


def test_run_mltsa_uses_selected_feature_set(monkeypatch, workspace_tmp_dir: Path, fake_md_module: FakeMdtraj) -> None:
    """run_mltsa should load and analyze only the requested feature set."""

    dataset_h5 = _build_md_feature_file(monkeypatch, workspace_tmp_dir, fake_md_module)
    featurize_dataset(
        h5_path=dataset_h5,
        feature_set="pca_set",
        feature_type="pca_xyz",
        label_experiment_id="labels",
        pca_components=2,
    )

    result = run_mltsa(
        dataset_h5,
        "pca_set",
        model="random_forest",
        model_kwargs={"n_estimators": 8, "max_depth": 2},
    )

    assert result.dataset.feature_set == "pca_set"
    assert result.dataset.feature_type == "pca_xyz"
    assert result.analysis_feature_names[:2] == ("pc_000@t000", "pc_001@t000")
    assert result.predictions.shape == result.y.shape


def test_run_mltsa_end_to_end_and_save_results(monkeypatch, workspace_tmp_dir: Path, fake_md_module: FakeMdtraj) -> None:
    """A small end-to-end MD analysis should train, explain, and persist results."""

    dataset_h5 = _build_md_feature_file(monkeypatch, workspace_tmp_dir, fake_md_module)
    results_h5 = workspace_tmp_dir / "analysis_results.h5"

    result = run_mltsa(
        dataset_h5,
        "closest",
        model="random_forest",
        model_kwargs={"n_estimators": 12, "max_depth": 3},
        explanation_method="native",
        results_h5_path=results_h5,
        experiment_id="md_demo",
    )

    assert result.training_score >= 0.5
    assert result.explanation.method == "native"
    assert result.analysis_path is not None
    assert result.explanation_path is not None

    with open_h5(results_h5, "r") as handle:
        analysis_group = handle[f"{results_experiment_path('md_demo')}/analyses/analysis_0000"]
        explanation_group = handle[f"{results_experiment_path('md_demo')}/explanations/explanation_0000"]
        np.testing.assert_array_equal(analysis_group["y_true"][...], result.y)
        np.testing.assert_array_equal(analysis_group["y_pred"][...], result.predictions)
        assert explanation_group.attrs["method"] == "native"


def test_cli_argument_parsing_and_dispatch(monkeypatch, workspace_tmp_dir: Path, capsys) -> None:
    """The MD CLI should stay a thin wrapper over the Python API."""

    calls: dict[str, dict[str, object]] = {}

    def fake_label(**kwargs):
        calls["label"] = kwargs
        return SimpleNamespace(
            processed=(object(),),
            skipped_entry_keys=(),
            overwritten_entry_keys=(),
        )

    def fake_build(**kwargs):
        calls["build"] = kwargs
        return SimpleNamespace(
            feature_set="feat",
            n_replicas=2,
            n_frames=4,
            n_features=3,
        )

    def fake_analyze(*args, **kwargs):
        calls["analyze"] = {"args": args, "kwargs": kwargs}
        return SimpleNamespace(
            model_name="random_forest",
            training_score=0.75,
            explanation=SimpleNamespace(n_features=8),
        )

    monkeypatch.setattr("mltsa.cli.md.label_trajectories", fake_label)
    monkeypatch.setattr("mltsa.cli.md.featurize_dataset", fake_build)
    monkeypatch.setattr("mltsa.cli.md.run_mltsa", fake_analyze)

    assert (
        md_cli_main(
            [
                "label",
                "--h5",
                str(workspace_tmp_dir / "labels.h5"),
                "--experiment",
                "exp",
                "--topology",
                str(workspace_tmp_dir / "topology.pdb"),
                "--trajectory",
                str(workspace_tmp_dir / "traj_a.dcd"),
                "--rule",
                "sum_distances",
                "--selection-pair",
                "index 0",
                "index 2",
                "--lower-threshold",
                "1.0",
                "--upper-threshold",
                "2.0",
                "--window-size",
                "3",
            ]
        )
        == 0
    )
    assert calls["label"]["experiment_id"] == "exp"
    captured = capsys.readouterr()
    assert "Labeled" in captured.out

    assert (
        md_cli_main(
            [
                "build",
                "--h5",
                str(workspace_tmp_dir / "dataset.h5"),
                "--feature-set",
                "feat",
                "--feature-type",
                "pca_xyz",
                "--label-experiment",
                "labels",
            ]
        )
        == 0
    )
    assert calls["build"]["feature_type"] == "pca_xyz"

    assert (
        md_cli_main(
            [
                "analyze",
                "--h5",
                str(workspace_tmp_dir / "dataset.h5"),
                "--feature-set",
                "feat",
                "--model",
                "rf",
                "--model-kwarg",
                "n_estimators=5",
                "--explain",
                "native",
                "--results-h5",
                str(workspace_tmp_dir / "results.h5"),
            ]
        )
        == 0
    )
    assert calls["analyze"]["kwargs"]["model"] == "rf"
    assert calls["analyze"]["kwargs"]["model_kwargs"] == {"n_estimators": 5}

    delegated: list[list[str]] = []
    root_cli_module = importlib.import_module("mltsa.cli.main")
    monkeypatch.setattr(root_cli_module.md_cli, "main", lambda argv=None: delegated.append(list(argv or [])) or 0)
    assert root_cli_main(["md", "analyze", "--h5", "dataset.h5", "--feature-set", "feat"]) == 0
    assert delegated == [["analyze", "--h5", "dataset.h5", "--feature-set", "feat"]]


def _build_md_feature_file(monkeypatch, workspace_tmp_dir: Path, fake_md_module: FakeMdtraj) -> Path:
    """Create a small labeled and featurized MD HDF5 for analysis tests."""

    topology_path, trajectory_paths = register_fake_inputs(
        workspace_tmp_dir,
        fake_md_module,
        {
            "traj_in_a": make_xyz([0.15, 0.20, 0.25, 0.30]),
            "traj_in_b": make_xyz([0.25, 0.30, 0.35, 0.40]),
            "traj_out_a": make_xyz([2.00, 2.10, 2.20, 2.30]),
            "traj_out_b": make_xyz([2.20, 2.30, 2.40, 2.50]),
        },
    )
    monkeypatch.setattr("mltsa.md.label._import_mdtraj", lambda: fake_md_module)
    monkeypatch.setattr("mltsa.md.featurize._import_mdtraj", lambda: fake_md_module)

    h5_path = workspace_tmp_dir / "md_dataset.h5"
    label_trajectories(
        trajectory_paths=[
            trajectory_paths["traj_in_a"],
            trajectory_paths["traj_in_b"],
            trajectory_paths["traj_out_a"],
            trajectory_paths["traj_out_b"],
        ],
        topology=topology_path,
        h5_path=h5_path,
        experiment_id="labels",
        rule="sum_distances",
        selection_pairs=[("index 0", "index 2")],
        lower_threshold=0.8,
        upper_threshold=1.5,
        window_size=2,
        replica_ids=["rep_in_a", "rep_in_b", "rep_out_a", "rep_out_b"],
    )
    featurize_dataset(
        h5_path=h5_path,
        feature_set="closest",
        feature_type="closest_residue_distances",
        label_experiment_id="labels",
    )
    return h5_path
