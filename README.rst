##########################################
mltsa: Modern Package Skeleton for MLTSA
##########################################

``mltsa`` is the new public package name for the next breaking release of
Machine Learning Transition State Analysis. This milestone introduces a modern
``src/``-layout foundation so the project can grow into a cleaner, better
structured library without removing the historical research code immediately.

********
Overview
********

The current repository now has a dedicated package root at ``src/mltsa`` with
room for the next-generation API:

- ``mltsa.io`` for data loading, serialization, and external data interfaces
- ``mltsa.synthetic`` for synthetic datasets and benchmark generators
- ``mltsa.models`` for model abstractions and training entry points
- ``mltsa.explain`` for explainability, attribution, and diagnostics
- ``mltsa.md`` for molecular dynamics-specific workflows
- ``mltsa.cli`` for command-line interfaces
- ``mltsa.utils`` for small shared helpers

The scientific implementation is being migrated into this structure in
incremental milestones. The repository now includes reusable HDF5 IO helpers,
synthetic dataset lifecycles, model wrappers, explainability tools, MD
labeling and featurization, MD analysis helpers, and thin CLI commands.

************
Installation
************

For local development:

.. code-block:: console

   pip install -e .

For development with tests:

.. code-block:: console

   pip install -e ".[test]"

For the MD workflow:

.. code-block:: console

   pip install -e ".[test,md]"

*****
Usage
*****

1. Synthetic time-series data generation for relevant feature identification
===========================================================================

.. code-block:: python

   from mltsa.synthetic import make_1d_dataset

   dataset = make_1d_dataset(
       n_trajectories=64,  # number of trajectories to generate
       n_steps=64,  # time steps per trajectory
       n_features=12,  # total observed features
       n_relevant=3,  # hidden ground-truth relevant features
   )
   dataset.save("synthetic_1d.h5", overwrite=True)

2. Running MLTSA on the saved synthetic HDF5 dataset
====================================================

.. code-block:: python

   from mltsa.explain import analyze
   from mltsa.models import get_model
   from mltsa.synthetic import load_dataset

   dataset = load_dataset("synthetic_1d.h5")
   X = dataset.X.reshape(dataset.n_trajectories, -1)  # flatten time for a simple baseline model
   feature_names = [
       f"{name}@t{step:03d}"
       for step in range(dataset.n_steps)
       for name in dataset.feature_names
   ]

   model = get_model("random_forest", n_estimators=200)  # tree model with native importance
   model.fit(X, dataset.y)

   result = analyze(
       model,
       method="native",  # no X/y needed for native tree importance
       feature_names=feature_names,
   )
   print(result.feature_names[result.ranked_indices[0]])

3. Molecular Dynamics CV generation, loading, and MLTSA analysis
================================================================

.. code-block:: python

   from mltsa.md import (
       featurize_dataset,
       label_trajectories,
       load_dataset as load_md_dataset,
       run_mltsa,
   )

   trajectory_paths = ["traj1.dcd", "traj2.dcd"]  # trajectories to process

   label_trajectories(
       trajectory_paths=trajectory_paths,
       topology="topology.pdb",  # shared topology file
       h5_path="md_dataset.h5",  # output HDF5 database
       experiment_id="labels",
       rule="sum_distances",
       selection_pairs=[("index 10", "index 220")],  # minimal labeling rule
       lower_threshold=0.4,  # IN threshold
       upper_threshold=0.8,  # OUT threshold
       window_size=25,  # only the final 25 frames are used for labeling
   )

   featurize_dataset(
       h5_path="md_dataset.h5",
       feature_set="closest",
       feature_type="closest_residue_distances",  # generate one named CV family
       label_experiment_id="labels",
   )

   md_dataset = load_md_dataset("md_dataset.h5", "closest")
   result = run_mltsa(
       "md_dataset.h5",  # MD feature database
       "closest",  # feature set to analyze
       model="random_forest",
   )

   print(md_dataset.X.shape)
   print(result.training_score)  # quick training score on the loaded feature set

CLI usage
=========

The current CLI directly supports the MD HDF5 workflow and is useful once you
already have an HDF5 CV dataset prepared.

.. code-block:: console

   mltsa --version
   mltsa-md --help
   mltsa-md analyze --h5 md_dataset.h5 --feature-set closest --model random_forest --explain native --results-h5 md_results.h5

*************
Documentation
*************

The initial documentation baseline lives under ``docs/source`` and includes:

- installation notes
- package overview
- synthetic dataset usage
- models and explainability
- MD workflow documentation
- CLI usage
- an upgrade guide from the historical repository structure

Example notebooks now live under ``notebooks/`` as a numbered tutorial series,
with older material preserved under ``notebooks/legacy/``.

**************
Migration Note
**************

Legacy packages such as ``MLTSA_datasets``, ``MLTSA_sklearn``, and
``MLTSA_tensorflow`` remain in the repository during the migration. They are no
longer the intended main public API for new work. New development should target
the ``mltsa`` package under ``src/``.
