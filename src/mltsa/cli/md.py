"""Thin command-line wrapper for MD workflows."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

from mltsa import __version__
from mltsa.md import featurize_dataset, label_trajectories, run_mltsa


def build_parser() -> ArgumentParser:
    """Create the ``mltsa-md`` argument parser."""

    parser = ArgumentParser(
        prog="mltsa-md",
        description="Label, featurize, and analyze MD datasets stored in mltsa HDF5 files.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    label_parser = subparsers.add_parser("label", help="Label trajectories into IN/OUT/TS states.")
    label_parser.add_argument("--h5", required=True, help="Target HDF5 path for the stored labels.")
    label_parser.add_argument("--experiment", required=True, help="Results experiment id used for labeling output.")
    label_parser.add_argument("--topology", required=True, help="Topology file shared by the trajectories.")
    label_parser.add_argument("--trajectory", action="append", required=True, help="Trajectory path. Repeat for multiple inputs.")
    label_parser.add_argument("--replica-id", action="append", default=None, help="Replica identifier. Repeat in trajectory order.")
    label_parser.add_argument("--rule", choices=("sum_distances", "com_distance"), required=True)
    label_parser.add_argument(
        "--selection-pair",
        action="append",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        required=True,
        help="Selection pair used by the labeling rule. Repeat as needed.",
    )
    label_parser.add_argument("--lower-threshold", type=float, required=True)
    label_parser.add_argument("--upper-threshold", type=float, required=True)
    label_parser.add_argument("--window-size", type=int, required=True)
    label_parser.add_argument("--append", action="store_true", help="Append new entries instead of requiring an empty store.")
    label_parser.add_argument("--overwrite", action="store_true", help="Overwrite matching entries or stores when allowed.")
    label_parser.add_argument("--no-waters", action="store_true", help="Disable nearby-water metadata capture.")
    label_parser.add_argument("--n-waters", type=int, default=50, help="Number of nearby waters to store when enabled.")
    label_parser.add_argument("--export-snapshots", action="store_true", help="Write IN/OUT/TS multi-model PDB snapshots.")
    label_parser.add_argument("--snapshot-dir", default=None, help="Output directory for snapshot PDB files.")
    label_parser.add_argument("--center-snapshots", action="store_true")
    label_parser.add_argument("--align-protein", action="store_true")
    label_parser.set_defaults(handler=_handle_label)

    build_parser_ = subparsers.add_parser("build", help="Build one named feature set inside an MD HDF5 dataset.")
    build_parser_.add_argument("--h5", required=True, help="MD dataset HDF5 path.")
    build_parser_.add_argument("--feature-set", required=True, help="Name of the feature set to create or append.")
    build_parser_.add_argument(
        "--feature-type",
        required=True,
        choices=(
            "closest_residue_distances",
            "all_ligand_protein_distances",
            "bubble_distances",
            "contact_map",
            "pca_xyz",
        ),
    )
    build_parser_.add_argument("--label-experiment", required=True, help="Label experiment id used as the source labels.")
    build_parser_.add_argument("--replica-id", action="append", default=None, help="Replica identifier to featurize. Repeat as needed.")
    build_parser_.add_argument("--append", action="store_true")
    build_parser_.add_argument("--overwrite", action="store_true")
    build_parser_.add_argument("--use-waters", action="store_true", help="Reuse stored nearby-water metadata when available.")
    build_parser_.add_argument("--ligand-selection", default="resname LIG")
    build_parser_.add_argument("--protein-selection", default="protein")
    build_parser_.add_argument("--bubble-cutoff", type=float, default=0.6)
    build_parser_.add_argument("--contact-threshold", type=float, default=0.45)
    build_parser_.add_argument("--pca-components", type=int, default=3)
    build_parser_.add_argument("--pca-selection", default=None)
    build_parser_.set_defaults(handler=_handle_build)

    analyze_parser = subparsers.add_parser("analyze", help="Train a model and compute feature importances for one feature set.")
    analyze_parser.add_argument("--h5", required=True, help="MD dataset HDF5 path.")
    analyze_parser.add_argument("--feature-set", required=True, help="Feature set name to analyze.")
    analyze_parser.add_argument("--model", default="random_forest", help="Model name understood by mltsa.models.get_model(...).")
    analyze_parser.add_argument("--model-kwarg", action="append", default=None, help="Model parameter as KEY=VALUE. Repeat as needed.")
    analyze_parser.add_argument("--explain", default="native", help="Explainability method to use.")
    analyze_parser.add_argument("--explain-kwarg", action="append", default=None, help="Explain parameter as KEY=VALUE. Repeat as needed.")
    analyze_parser.add_argument("--results-h5", default=None, help="Optional results HDF5 path for persisted analysis output.")
    analyze_parser.add_argument("--experiment", default=None, help="Optional results experiment id.")
    analyze_parser.set_defaults(handler=_handle_analyze)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``mltsa-md`` CLI and return a process exit code."""

    parser = build_parser()
    namespace = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(namespace, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(namespace))


def _handle_label(namespace: Namespace) -> int:
    """Dispatch the ``label`` subcommand to the Python API."""

    result = label_trajectories(
        trajectory_paths=[Path(path) for path in namespace.trajectory],
        topology=Path(namespace.topology),
        h5_path=Path(namespace.h5),
        experiment_id=namespace.experiment,
        rule=namespace.rule,
        selection_pairs=[(str(left), str(right)) for left, right in namespace.selection_pair],
        lower_threshold=float(namespace.lower_threshold),
        upper_threshold=float(namespace.upper_threshold),
        window_size=int(namespace.window_size),
        replica_ids=None if namespace.replica_id is None else list(namespace.replica_id),
        append=bool(namespace.append),
        overwrite=bool(namespace.overwrite),
        store_waters=not bool(namespace.no_waters),
        n_waters=int(namespace.n_waters),
        export_snapshots=bool(namespace.export_snapshots),
        snapshot_dir=None if namespace.snapshot_dir is None else Path(namespace.snapshot_dir),
        center_snapshots=bool(namespace.center_snapshots),
        align_protein=bool(namespace.align_protein),
    )
    print(
        f"Labeled {len(result.processed)} trajectories "
        f"(skipped={len(result.skipped_entry_keys)}, overwritten={len(result.overwritten_entry_keys)})."
    )
    return 0


def _handle_build(namespace: Namespace) -> int:
    """Dispatch the ``build`` subcommand to the Python API."""

    dataset = featurize_dataset(
        h5_path=Path(namespace.h5),
        feature_set=namespace.feature_set,
        feature_type=namespace.feature_type,
        label_experiment_id=namespace.label_experiment,
        replica_ids=None if namespace.replica_id is None else list(namespace.replica_id),
        append=bool(namespace.append),
        overwrite=bool(namespace.overwrite),
        use_waters=bool(namespace.use_waters),
        ligand_selection=namespace.ligand_selection,
        protein_selection=namespace.protein_selection,
        bubble_cutoff=float(namespace.bubble_cutoff),
        contact_threshold=float(namespace.contact_threshold),
        pca_components=int(namespace.pca_components),
        pca_selection=namespace.pca_selection,
    )
    print(
        f"Feature set {dataset.feature_set!r}: "
        f"{dataset.n_replicas} replicas, {dataset.n_frames} frames, {dataset.n_features} features."
    )
    return 0


def _handle_analyze(namespace: Namespace) -> int:
    """Dispatch the ``analyze`` subcommand to the Python API."""

    model_kwargs = _parse_key_value_options(namespace.model_kwarg)
    explanation_kwargs = _parse_key_value_options(namespace.explain_kwarg)
    result = run_mltsa(
        Path(namespace.h5),
        namespace.feature_set,
        model=namespace.model,
        model_kwargs=model_kwargs,
        explanation_method=namespace.explain,
        explanation_kwargs=explanation_kwargs,
        results_h5_path=None if namespace.results_h5 is None else Path(namespace.results_h5),
        experiment_id=namespace.experiment,
    )
    print(
        f"Analysis complete with {result.model_name}: "
        f"score={result.training_score:.3f}, features={result.explanation.n_features}."
    )
    return 0


def _parse_key_value_options(values: Sequence[str] | None) -> dict[str, object]:
    """Parse repeated ``KEY=VALUE`` options with JSON-like value coercion."""

    parsed: dict[str, object] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, received {item!r}.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Option keys must not be empty.")
        parsed[key] = _coerce_cli_value(raw_value.strip())
    return parsed


def _coerce_cli_value(value: str) -> object:
    """Best-effort coercion for CLI keyword values."""

    if value == "":
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "none":
            return None
        return value


__all__ = ["build_parser", "main"]
