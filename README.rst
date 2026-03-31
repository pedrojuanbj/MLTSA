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

.. code-block:: python

   from mltsa.models import get_model
   from mltsa.synthetic import make_1d_dataset

   dataset = make_1d_dataset(n_trajectories=16)
   model = get_model("random_forest", n_estimators=100)

   model.fit(dataset.X.reshape(dataset.n_trajectories, -1), dataset.y)

.. code-block:: python

   from mltsa.md import run_mltsa

   result = run_mltsa("md_dataset.h5", "closest", model="random_forest")
   print(result.training_score)

.. code-block:: console

   mltsa --version
   mltsa-md --help

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

Example notebooks now live under ``notebooks/`` with topic folders for
synthetic, MD, and legacy material.

**************
Migration Note
**************

Legacy packages such as ``MLTSA_datasets``, ``MLTSA_sklearn``, and
``MLTSA_tensorflow`` remain in the repository during the migration. They are no
longer the intended main public API for new work. New development should target
the ``mltsa`` package under ``src/``.
