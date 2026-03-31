"""Generate the current tutorial notebooks for the mltsa package."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def lines(text: str) -> list[str]:
    """Return notebook-ready source lines."""

    return dedent(text).strip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict[str, object]:
    """Build a markdown notebook cell."""

    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code(text: str) -> dict[str, object]:
    """Build a code notebook cell."""

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    """Build a notebook document."""

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_CELL = code(
    """
    from pathlib import Path
    import sys

    try:
        import mltsa  # noqa: F401
    except ImportError:
        for parent in (Path.cwd(), *Path.cwd().parents):
            src_dir = parent / "src"
            if (src_dir / "mltsa").exists():
                sys.path.insert(0, str(src_dir))
                break
        import mltsa  # noqa: F401

    import numpy as np
    import matplotlib.pyplot as plt

    for parent in (Path.cwd(), *Path.cwd().parents):
        notebooks_dir = parent / "notebooks"
        if notebooks_dir.exists():
            DATA_DIR = notebooks_dir / "_generated"
            break
    else:
        DATA_DIR = Path.cwd() / "_generated"

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    np.set_printoptions(precision=3, suppress=True)
    """
)
SETUP_CELL["metadata"] = {"jupyter": {"source_hidden": True}, "tags": ["hide-input", "remove-input"]}


def write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    """Serialize one notebook to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")


def synthetic_1d_notebook() -> list[dict[str, object]]:
    """Build the 1D synthetic tutorial notebook."""

    return [
        markdown(
            """
            # 1 - Synthetic 1D Data Generation (The Basics)

            This is the smallest synthetic system in `mltsa`, and it exists for one reason:
            **to give us a clean toy problem where only a few observed features really matter**.

            The intuition is simple. We imagine a one-dimensional transition coordinate that starts
            near the top of a barrier and then falls toward one of two outcomes: `IN` or `OUT`.
            In real data we usually do not observe that clean coordinate directly. Instead, we see a
            set of noisy features, and only some of them carry that latent signal.
            """
        ),
        SETUP_CELL,
        markdown(
            """
            ## Step 1: draw the intuition first

            The current lightweight generator does **not** numerically integrate a physical
            double-well potential. Instead, it creates a latent coordinate that gradually separates
            into two classes over time. Still, it helps to keep the classical double-well picture in
            mind because it explains why this toy problem is useful.
            """
        ),
        code(
            """
            x = np.linspace(-1.8, 1.8, 400)
            potential = x**4 - x**2

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(x, potential, lw=3, color="#2a6f97")
            ax.axvline(-0.707, ls="--", color="#4c956c", label="IN basin")
            ax.axvline(0.707, ls="--", color="#bc4749", label="OUT basin")
            ax.axvline(0.0, ls=":", color="#666666", label="transition region")
            ax.set_xlabel("reaction coordinate")
            ax.set_ylabel("cartoon potential")
            ax.set_title("Double-well intuition for the 1D toy system")
            ax.legend(loc="upper center", ncol=3)
            plt.show()
            """
        ),
        markdown(
            """
            ## Step 2: generate a small dataset

            We keep the example intentionally small. The generator gives us:

            - `X`: observed time-series features
            - `y`: final class labels
            - `feature_names`
            - `relevant_features`
            - enough metadata to save, reload, rebuild, and extend the dataset later
            """
        ),
        code(
            """
            from mltsa.synthetic import make_1d_dataset

            dataset = make_1d_dataset(
                n_trajectories=120,  # Number of trajectories to generate
                n_steps=64,  # Number of time steps per trajectory
                n_features=12,  # Total observed features seen by the model
                n_relevant=3,  # Hidden features that truly carry the signal
                base_seed=1234,  # Reproducible random seed
            )

            print("X shape:", dataset.X.shape)
            print("y shape:", dataset.y.shape)
            print("Relevant feature ids:", dataset.relevant_features)
            print("Relevant feature names:", dataset.relevant_feature_names)
            print("Generation parameters:", dataset.generation_params)
            """
        ),
        markdown(
            """
            ## Step 3: compare the clean latent coordinate with the observed features

            Below, **before mixing** means the hidden latent coordinate that drives the outcome.
            **After mixing** means the observed feature space that a model will actually see.
            """
        ),
        code(
            """
            relevant_feature = dataset.relevant_features[0]
            irrelevant_feature = next(
                index for index in range(dataset.n_features) if index not in dataset.relevant_features
            )
            time = np.arange(dataset.n_steps)

            in_indices = np.where(dataset.y == 0)[0][:4]
            out_indices = np.where(dataset.y == 1)[0][:4]

            fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

            for index in in_indices:
                axes[0].plot(time, dataset.latent_trajectories[index, :, 0], color="#4c956c", alpha=0.8)
                axes[1].plot(time, dataset.X[index, :, relevant_feature], color="#4c956c", alpha=0.8)
                axes[2].plot(time, dataset.X[index, :, irrelevant_feature], color="#4c956c", alpha=0.8)

            for index in out_indices:
                axes[0].plot(time, dataset.latent_trajectories[index, :, 0], color="#bc4749", alpha=0.8)
                axes[1].plot(time, dataset.X[index, :, relevant_feature], color="#bc4749", alpha=0.8)
                axes[2].plot(time, dataset.X[index, :, irrelevant_feature], color="#bc4749", alpha=0.8)

            axes[0].set_title("Before mixing: latent coordinate")
            axes[1].set_title(f"After mixing: {dataset.feature_names[relevant_feature]}")
            axes[2].set_title(f"Mostly noise: {dataset.feature_names[irrelevant_feature]}")

            for axis in axes:
                axis.set_xlabel("time step")
            axes[0].set_ylabel("value")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            from mltsa.synthetic import (
                plot_ground_truth_relevance,
                plot_relevance_over_time,
            )

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            plot_ground_truth_relevance(dataset, ax=axes[0])
            plot_relevance_over_time(dataset, ax=axes[1])
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## Step 4: save the dataset to HDF5

            The dataset object knows how to persist itself cleanly.
            """
        ),
        code(
            """
            dataset_path = DATA_DIR / "synthetic_1d_basics.h5"
            dataset.save(dataset_path, overwrite=True)
            dataset_path
            """
        ),
        markdown(
            """
            ## Step 5: reload it and generate more trajectories later

            This is handy when you want to keep the same system definition but expand the dataset.
            """
        ),
        code(
            """
            from mltsa.synthetic import load_dataset

            reloaded = load_dataset(dataset_path)
            extra = reloaded.generate_more(24)
            extra.append_to_file(dataset_path)

            expanded = load_dataset(dataset_path)
            print("Trajectories before append:", reloaded.n_trajectories)
            print("New trajectories generated:", extra.n_trajectories)
            print("Trajectories after append:", expanded.n_trajectories)
            print("Relevant features are unchanged:", expanded.relevant_feature_names)
            """
        ),
    ]


def synthetic_2d_notebook() -> list[dict[str, object]]:
    """Build the 2D synthetic tutorial notebook."""

    return [
        markdown(
            """
            # 2 - Synthetic 2D Data Generation (Added Complexity through Ice-Cream)

            The 2D synthetic system adds two things that are useful for MLTSA:

            - more geometric complexity than the 1D toy example
            - a **time-dependent change in importance**, because the discriminative direction grows
              as the trajectory evolves

            The silly mental image is an ice-cream: a simple cone-like direction that matters for the
            class, plus a spiral nuisance motion that makes the path look richer than it really is.
            """
        ),
        SETUP_CELL,
        markdown(
            """
            ## Step 1: serve the ice-cream

            First we create the actual dataset. In this 2D toy system, the `x` direction is the
            informative one, while the `y` direction acts as the nuisance spiral.
            """
        ),
        code(
            """
            from mltsa.synthetic import make_2d_dataset

            dataset = make_2d_dataset(
                n_trajectories=160,  # Number of trajectories to generate
                n_steps=80,  # Number of time steps per trajectory
                n_features=16,  # Number of observed projected features
                pattern="spiral",  # The nuisance path we want to use
                base_seed=4321,  # Reproducible dataset definition
            )

            print("X shape:", dataset.X.shape)
            print("Latent shape:", dataset.latent_trajectories.shape)
            print("Relevant feature names:", dataset.relevant_feature_names[:5], "...")
            """
        ),
        markdown(
            """
            ## Step 2: plot the ice-cream

            The left and middle panels are the cartoon. The right panel is the actual latent data
            produced by `mltsa`.
            """
        ),
        code(
            """
            theta = np.linspace(0, 4 * np.pi, 400)
            radius = np.linspace(0.15, 1.2, 400)
            spiral_x = radius * np.cos(theta)
            spiral_y = radius * np.sin(theta)

            grid = np.linspace(-2.0, 2.0, 160)
            xx, yy = np.meshgrid(grid, grid)
            cartoon_surface = 0.18 * (xx**2 + yy**2) - 0.4 * xx + 0.1 * yy**2

            fig, axes = plt.subplots(1, 3, figsize=(16, 4))

            cone_x = np.array([-0.7, 0.0, 0.7, -0.7])
            cone_y = np.array([-1.5, -0.1, -1.5, -1.5])
            axes[0].fill(cone_x, cone_y, color="#d4a373", alpha=0.9)
            axes[0].plot(spiral_x * 0.55, spiral_y * 0.55 + 0.7, color="#e76f51", lw=4)
            axes[0].set_title("Ice-cream cartoon")
            axes[0].set_aspect("equal")
            axes[0].axis("off")

            contour = axes[1].contourf(xx, yy, cartoon_surface, levels=18, cmap="cividis")
            axes[1].plot(spiral_x, spiral_y, color="white", lw=2)
            axes[1].set_title("Cartoon 2D landscape + spiral")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("y")
            fig.colorbar(contour, ax=axes[1], shrink=0.8)

            for index in range(6):
                color = "#bc4749" if dataset.y[index] else "#4c956c"
                axes[2].plot(
                    dataset.latent_trajectories[index, :, 0],
                    dataset.latent_trajectories[index, :, 1],
                    color=color,
                    alpha=0.75,
                )
            axes[2].set_title("Latent trajectories generated by mltsa")
            axes[2].set_xlabel("latent x")
            axes[2].set_ylabel("latent y")

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## Step 3: analysis of the ice-cream

            Now we inspect both the observed features and the built-in ground-truth relevance.
            """
        ),
        code(
            """
            relevant_feature = dataset.relevant_features[0]
            less_relevant_feature = next(
                index for index in range(dataset.n_features) if index not in dataset.relevant_features
            )
            time = np.arange(dataset.n_steps)

            fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
            for index in range(4):
                axes[0].plot(time, dataset.X[index, :, relevant_feature], alpha=0.75)
                axes[1].plot(time, dataset.X[index, :, less_relevant_feature], alpha=0.75)

            axes[0].set_title(f"More informative: {dataset.feature_names[relevant_feature]}")
            axes[1].set_title(f"Mostly nuisance: {dataset.feature_names[less_relevant_feature]}")
            for axis in axes:
                axis.set_xlabel("time step")
                axis.set_ylabel("feature value")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            from mltsa.synthetic import (
                plot_ground_truth_relevance,
                plot_relevance_over_time,
            )

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            plot_ground_truth_relevance(dataset, ax=axes[0])
            plot_relevance_over_time(dataset, ax=axes[1])
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## Step 4: save the ice-cream for later
            """
        ),
        code(
            """
            from mltsa.synthetic import load_dataset

            dataset_path = DATA_DIR / "synthetic_2d_icecream.h5"
            dataset.save(dataset_path, overwrite=True)
            print("Saved to:", dataset_path)
            """
        ),
        markdown(
            """
            ## Step 5: reconstruct the ice-cream

            Because the seeds and system definition are stored, we can rebuild the same dataset
            exactly.
            """
        ),
        code(
            """
            from mltsa.synthetic import load_dataset

            reloaded = load_dataset(dataset_path)
            rebuilt = reloaded.rebuild_exact()

            print("Reloaded shape:", reloaded.X.shape)
            print("Exact reconstruction:", np.allclose(reloaded.X, rebuilt.X))
            print("Time relevance available:", reloaded.time_relevance is not None)
            """
        ),
    ]


def mltsa_for_dummies_notebook() -> list[dict[str, object]]:
    """Build the basic end-to-end MLTSA notebook."""

    return [
        markdown(
            """
            # 3 - MLTSA for dummies

            This notebook is the shortest end-to-end introduction to the basic MLTSA loop:

            1. get a dataset
            2. decide which part of the trajectory to use
            3. train a model
            4. check validation and truly unseen test accuracy
            5. inspect which features the model relied on

            We keep one trajectory as one sample. That makes the train/validation/test split easy to
            reason about and avoids leaking frames from the same trajectory across splits.
            """
        ),
        SETUP_CELL,
        markdown(
            """
            ## Step 1: prepare the trajectory splits

            We either reuse the 1D dataset from notebook 1 or create a small one on the fly. Then we
            define an early-time window, split the available trajectories into train and validation,
            and generate a separate set of unseen trajectories for the final test.
            """
        ),
        code(
            """
            from sklearn.model_selection import train_test_split

            from mltsa.explain import analyze
            from mltsa.models import get_model
            from mltsa.synthetic import load_dataset, make_1d_dataset

            dataset_path = DATA_DIR / "synthetic_1d_basics.h5"
            if dataset_path.exists():
                dataset = load_dataset(dataset_path)
                print("Loaded existing dataset from notebook 1.")
            else:
                dataset = make_1d_dataset(
                    n_trajectories=144,  # Enough trajectories for train/validation
                    n_steps=64,  # Total steps available in each trajectory
                    n_features=12,  # Total observed features
                    n_relevant=3,  # Hidden ground-truth relevant features
                    base_seed=1234,  # Reproducible dataset definition
                )
                dataset.save(dataset_path, overwrite=True)
                print("Generated a fresh dataset because no local HDF5 was available.")

            window = 24  # Early-time window used for the prediction problem
            X_all = dataset.X[:, :window, :].reshape(dataset.n_trajectories, -1)
            y_all = dataset.y
            feature_names = tuple(
                f"{name}@t{step:03d}"
                for step in range(window)
                for name in dataset.feature_names
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_all,
                y_all,
                test_size=0.25,
                random_state=0,
                stratify=y_all,
            )

            test_dataset = dataset.generate_more(60)
            X_test = test_dataset.X[:, :window, :].reshape(test_dataset.n_trajectories, -1)
            y_test = test_dataset.y

            print("Train shape:", X_train.shape)
            print("Validation shape:", X_val.shape)
            print("Unseen test shape:", X_test.shape)
            """
        ),
        markdown(
            """
            The shapes above are the key check: train and validation come from the same dataset, while
            the test set comes from brand-new trajectories generated later from the same system.
            """
        ),
        markdown(
            """
            ## Step 2: fit a simple model

            A random forest is a nice starting point because it is fast, robust, and exposes native
            feature importances.
            """
        ),
        code(
            """
            model = get_model(
                "random_forest",
                n_estimators=200,  # Forest size
                min_samples_leaf=2,  # Mild regularization
                random_state=0,  # Reproducible training
            )
            model.fit(X_train, y_train)

            validation_accuracy = model.score(X_val, y_val)
            test_accuracy = model.score(X_test, y_test)

            print(f"Validation accuracy: {validation_accuracy:.3f}")
            print(f"Unseen test accuracy: {test_accuracy:.3f}")
            """
        ),
        markdown(
            """
            A test score in roughly the `0.8-0.9` range is a good sign here: the model is learning
            real signal, but the problem is still non-trivial.
            """
        ),
        markdown(
            """
            ## Step 3: ask the model which features mattered

            Native importance works for tree models without needing `X` or `y` again.
            """
        ),
        code(
            """
            explanation = analyze(
                model,
                method="native",
                feature_names=feature_names,
            )

            importance_by_feature = explanation.importances.reshape(window, dataset.n_features).mean(axis=0)
            ranked_feature_ids = np.argsort(importance_by_feature)[::-1]

            print("Ground-truth relevant features:", dataset.relevant_feature_names)
            print(
                "Top recovered features:",
                [dataset.feature_names[index] for index in ranked_feature_ids[:5]],
            )
            """
        ),
        code(
            """
            colors = [
                "#e76f51" if index in dataset.relevant_features else "#b8c0c8"
                for index in range(dataset.n_features)
            ]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(np.arange(dataset.n_features), importance_by_feature, color=colors)
            ax.set_xticks(np.arange(dataset.n_features), dataset.feature_names, rotation=45, ha="right")
            ax.set_ylabel("mean importance across time")
            ax.set_title("Recovered feature importance")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            The orange bars are the true relevant features. If the model and the explanation are doing
            their job, those bars should stand out from the rest.
            """
        ),
    ]


def real_data_placeholder_notebook() -> list[dict[str, object]]:
    """Build the placeholder real-data notebook."""

    return [
        markdown(
            """
            # 4 - MLTSA on real data

            **Placeholder notebook.**

            For now we use the public data from the workshop repository below:

            - <https://github.com/rostaresearch/enhanced-sampling-workshop-2022/tree/main/Day1/4.MLTSA/data>

            These are already-calculated collective variables, so this notebook focuses on:

            1. downloading the files
            2. loading them cleanly
            3. splitting trajectories into train, validation, and unseen test sets
            4. concatenating frames within each split
            5. running a small MLTSA-style analysis on top of them

            This is intentionally not the final polished MD workflow notebook. It is a compact bridge
            until the full end-to-end real-data tutorial is ready.
            """
        ),
        SETUP_CELL,
        markdown(
            """
            ## Step 1: download the placeholder workshop files
            """
        ),
        code(
            """
            import json
            from urllib.request import Request, urlopen

            WORKSHOP_API = (
                "https://api.github.com/repos/rostaresearch/"
                "enhanced-sampling-workshop-2022/contents/Day1/4.MLTSA/data?ref=main"
            )
            WORKSHOP_HEADERS = {"User-Agent": "mltsa-tutorial"}

            workshop_dir = DATA_DIR / "workshop_placeholder"
            workshop_dir.mkdir(exist_ok=True)


            def download_workshop_data(target_dir: Path) -> list[Path]:
                request = Request(WORKSHOP_API, headers=WORKSHOP_HEADERS)
                with urlopen(request, timeout=60) as response:
                    listing = json.load(response)

                downloaded: list[Path] = []
                for item in listing:
                    name = item["name"]
                    if not name.endswith((".csv", ".npy", ".txt")):
                        continue

                    destination = target_dir / name
                    if not destination.exists():
                        file_request = Request(item["download_url"], headers=WORKSHOP_HEADERS)
                        with urlopen(file_request, timeout=120) as response:
                            destination.write_bytes(response.read())
                    downloaded.append(destination)

                return sorted(downloaded)


            downloaded_files = download_workshop_data(workshop_dir)
            for path in downloaded_files:
                print(path.name)
            """
        ),
        markdown(
            """
            ## Step 2: load the CVs and labels

            We download everything in the folder, including:

            - `allres_features.csv`
            - `allres_features.npy`
            - the precomputed CV arrays in `downhill_*.npy`
            - the labels in `downhill_labels.txt`

            For this placeholder analysis we only need the CSV names, the `downhill_*` arrays, and
            the labels. The pickled `allres_features.npy` file is preserved locally as well, but we do
            not depend on it here.
            """
        ),
        code(
            """
            import numpy as np

            feature_name_path = workshop_dir / "allres_features.csv"
            downhill_1_path = workshop_dir / "downhill_allres1.npy"
            downhill_2_path = workshop_dir / "downhill_allres2.npy"
            labels_path = workshop_dir / "downhill_labels.txt"
            metadata_path = workshop_dir / "allres_features.npy"

            feature_names = [
                line.strip()
                for line in feature_name_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            downhill_1 = np.load(downhill_1_path, allow_pickle=True)
            downhill_2 = np.load(downhill_2_path, allow_pickle=True)
            label_text = np.array(
                [
                    line.strip()
                    for line in labels_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            )

            try:
                optional_metadata = np.load(metadata_path, allow_pickle=True)
                print("Optional metadata loaded:", optional_metadata.shape)
            except Exception as exc:
                print("Optional metadata was downloaded but is not needed here:", type(exc).__name__)

            raw_cv = np.concatenate([downhill_1, downhill_2], axis=0)
            classes, y = np.unique(label_text, return_inverse=True)

            print("Raw CV shape:", raw_cv.shape)
            print("Class labels:", classes.tolist())
            print("Feature count:", len(feature_names))
            """
        ),
        markdown(
            """
            ## Step 3, 4 and 5: split trajectories, concatenate frames, and run MLTSA

            We keep the full feature dimension. The only slicing we do is in time: frames `100:200`.

            The split is done at the **trajectory** level first. Only after that do we concatenate
            frames inside each split. That way the test set still contains unseen trajectories.
            """
        ),
        code(
            """
            from sklearn.model_selection import train_test_split

            from mltsa.explain import analyze, plot_importances
            from mltsa.models import get_model

            frame_window = slice(100, 200)  # Keep only the middle part of each trajectory
            X_window = raw_cv[:, frame_window, :]

            X_train_traj, X_test_traj, y_train_traj, y_test_traj = train_test_split(
                X_window,
                y,
                test_size=0.25,
                random_state=0,
                stratify=y,
            )
            X_train_traj, X_val_traj, y_train_traj, y_val_traj = train_test_split(
                X_train_traj,
                y_train_traj,
                test_size=0.25,
                random_state=0,
                stratify=y_train_traj,
            )


            def concatenate_frames(X_traj: np.ndarray, y_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                n_traj, n_frames, n_features = X_traj.shape
                X_frames = X_traj.reshape(n_traj * n_frames, n_features)
                y_frames = np.repeat(y_traj, n_frames)
                return X_frames, y_frames


            X_train, y_train = concatenate_frames(X_train_traj, y_train_traj)
            X_val, y_val = concatenate_frames(X_val_traj, y_val_traj)
            X_test, y_test = concatenate_frames(X_test_traj, y_test_traj)

            print("Trajectory split shapes:")
            print("  train:", X_train_traj.shape)
            print("  validation:", X_val_traj.shape)
            print("  test:", X_test_traj.shape)
            print("Frame-concatenated shapes:")
            print("  train:", X_train.shape)
            print("  validation:", X_val.shape)
            print("  test:", X_test.shape)

            model = get_model(
                "random_forest",
                n_estimators=200,  # Forest size
                min_samples_leaf=2,  # Mild regularization
            )
            model.fit(X_train, y_train)
            validation_accuracy = model.score(X_val, y_val)
            test_accuracy = model.score(X_test, y_test)

            result = analyze(
                model,
                method="native",
                feature_names=tuple(feature_names),
            )

            print(f"Placeholder validation accuracy: {validation_accuracy:.3f}")
            print(f"Placeholder test accuracy: {test_accuracy:.3f}")
            print("Top workshop features:")
            for index in result.ranked_indices[:10]:
                print(f"  {result.feature_names[index]} -> {result.importances[index]:.4f}")
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_importances(result, top_n=15, ax=ax)
            plt.tight_layout()
            plt.show()
            """
        ),
    ]


def exploring_models_notebook() -> list[dict[str, object]]:
    """Build the model comparison notebook."""

    return [
        markdown(
            """
            # 5 - Exploring models

            Here the goal is not to crown a universal best model. The goal is to show that:

            - different model families can all be used inside the same `mltsa` workflow
            - some models expose **native** importance directly
            - others need **perturbation-based** explanations such as permutation or global mean

            To keep the comparison fair and fast, we use the same flattened early-time window for
            every model.
            """
        ),
        SETUP_CELL,
        code(
            """
            from sklearn.model_selection import train_test_split

            from mltsa.explain import analyze
            from mltsa.models import get_model
            from mltsa.synthetic import make_1d_dataset

            dataset = make_1d_dataset(
                n_trajectories=200,  # Quick comparison dataset
                n_steps=64,  # Total steps per trajectory
                n_features=10,  # Total observed features
                n_relevant=3,  # Hidden ground-truth relevant features
                base_seed=2024,  # Reproducible dataset definition
            )

            window = 28  # Shared early-time window for all models
            X = dataset.X[:, :window, :].reshape(dataset.n_trajectories, -1)
            y = dataset.y
            flat_feature_names = tuple(
                f"{name}@t{step:03d}"
                for step in range(window)
                for name in dataset.feature_names
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=0,
                stratify=y,
            )
            """
        ),
        markdown(
            """
            We compare four models:

            - `RandomForest` and `GradientBoosting` from sklearn
            - `MLP` and `CNN1D` from the lightweight torch wrappers

            The tree models expose native importances. The neural models do not, so we switch to
            `global_mean` and `permutation`.
            """
        ),
        code(
            """
            model_specs = [
                ("RandomForest", "random_forest", {"n_estimators": 120}, "native", {}),
                ("GradientBoosting", "gradient_boosting", {"n_estimators": 120}, "native", {}),
                ("Torch MLP", "mlp", {"epochs": 10, "hidden_sizes": (128, 64), "batch_size": 32}, "global_mean", {}),
                ("Torch CNN1D", "cnn1d", {"epochs": 10, "channels": 24, "batch_size": 32}, "permutation", {"n_repeats": 3, "n_jobs": 1}),
            ]

            summaries = []
            explanations = {}

            for label, model_name, kwargs, method, extra_kwargs in model_specs:
                model = get_model(model_name, **kwargs)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                explanation = analyze(
                    model,
                    method=method,
                    X=X_test,
                    y=y_test,
                    feature_names=flat_feature_names,
                    **extra_kwargs,
                )

                base_importance = explanation.importances.reshape(window, dataset.n_features).mean(axis=0)
                explanations[label] = base_importance
                summaries.append((label, method, score))

            for label, method, score in summaries:
                print(f"{label:18s} | method={method:11s} | test_accuracy={score:.3f}")
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
            axes = axes.ravel()

            for axis, (label, base_importance) in zip(axes, explanations.items()):
                colors = [
                    "#e76f51" if index in dataset.relevant_features else "#c7ced6"
                    for index in range(dataset.n_features)
                ]
                axis.bar(np.arange(dataset.n_features), base_importance, color=colors)
                axis.set_title(label)
                axis.set_xticks(np.arange(dataset.n_features), dataset.feature_names, rotation=45, ha="right")
                axis.set_ylabel("mean importance")

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            A nice rule of thumb:

            - **tree models**: try native importance first
            - **neural models**: expect to use perturbation methods more often
            """
        ),
    ]


def shap_compatible_notebook() -> list[dict[str, object]]:
    """Build the SHAP-compatible model notebook."""

    return [
        markdown(
            """
            # 6 - SHAP compatible

            This notebook uses a tree ensemble because tree models are both:

            - easy to analyze directly with the built-in `mltsa` tools
            - good candidates for external SHAP workflows later on

            The notebook itself stays inside `mltsa`. An optional final cell shows how you could hook
            the same fitted model into SHAP if that package is installed locally.
            """
        ),
        SETUP_CELL,
        code(
            """
            from sklearn.model_selection import train_test_split

            from mltsa.explain import analyze, plot_importances
            from mltsa.models import get_model
            from mltsa.synthetic import make_1d_dataset

            dataset = make_1d_dataset(
                n_trajectories=180,  # Small but stable synthetic dataset
                n_steps=64,  # Total steps per trajectory
                n_features=12,  # Total observed features
                n_relevant=3,  # Hidden ground-truth relevant features
                base_seed=77,  # Reproducible dataset definition
            )
            window = 28  # Early-time window used for the analysis

            X = dataset.X[:, :window, :].reshape(dataset.n_trajectories, -1)
            y = dataset.y
            feature_names = tuple(
                f"{name}@t{step:03d}"
                for step in range(window)
                for name in dataset.feature_names
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=0,
                stratify=y,
            )

            model = get_model(
                "extra_trees",
                n_estimators=240,  # A larger tree ensemble for a stable ranking
                min_samples_leaf=2,  # Mild regularization
            )
            model.fit(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)

            explanation = analyze(
                model,
                method="native",
                feature_names=feature_names,
            )

            base_importance = explanation.importances.reshape(window, dataset.n_features).mean(axis=0)
            print(f"Test accuracy: {test_accuracy:.3f}")
            print("Ground-truth relevant features:", dataset.relevant_feature_names)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            plot_importances(explanation, top_n=12, ax=axes[0])

            colors = [
                "#e76f51" if index in dataset.relevant_features else "#c7ced6"
                for index in range(dataset.n_features)
            ]
            axes[1].bar(np.arange(dataset.n_features), base_importance, color=colors)
            axes[1].set_xticks(np.arange(dataset.n_features), dataset.feature_names, rotation=45, ha="right")
            axes[1].set_title("Base-feature view")
            axes[1].set_ylabel("mean importance")

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            We can also persist the explanation result to an HDF5 results file.
            """
        ),
        code(
            """
            results_path = DATA_DIR / "shap_compatible_results.h5"
            explanation_path = explanation.save(results_path, experiment_id="shap_compatible")

            print("Results file:", results_path)
            print("Saved explanation path:", explanation_path)
            """
        ),
        markdown(
            """
            ## Optional: use the same fitted model with SHAP

            This cell is optional on purpose. The tutorial still works without `shap`.
            """
        ),
        code(
            """
            try:
                import shap
            except ImportError:
                print("Optional dependency not installed: pip install shap")
            else:
                explainer = shap.TreeExplainer(model.estimator)
                shap_values = explainer.shap_values(X_test[:20])
                summary = np.asarray(shap_values)
                print("SHAP values computed for 20 samples.")
                print("SHAP array shape:", summary.shape)
            """
        ),
    ]


def main() -> None:
    """Write all current tutorial notebooks."""

    tutorials = {
        "01_synthetic_1d_data_generation_the_basics.ipynb": synthetic_1d_notebook(),
        "02_synthetic_2d_data_generation_added_complexity_through_ice_cream.ipynb": synthetic_2d_notebook(),
        "03_mltsa_for_dummies.ipynb": mltsa_for_dummies_notebook(),
        "04_mltsa_on_real_data_placeholder.ipynb": real_data_placeholder_notebook(),
        "05_exploring_models.ipynb": exploring_models_notebook(),
        "06_shap_compatible.ipynb": shap_compatible_notebook(),
    }

    for filename, cells in tutorials.items():
        write_notebook(NOTEBOOKS / filename, cells)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
