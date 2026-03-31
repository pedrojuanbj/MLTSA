Upgrade Guide
=============

This repository is moving from several legacy top-level modules to a single
package-first API under ``mltsa``.

Concept map
-----------

.. list-table::
   :header-rows: 1

   * - Legacy module or concept
     - New home
     - Notes
   * - ``MLTSA_datasets.OneD_pot`` and ``MLTSA_datasets.TwoD_pot``
     - ``mltsa.synthetic``
     - Use ``make_1d_dataset(...)``, ``make_2d_dataset(...)``, and ``SyntheticDataset``.
   * - ``MLTSA_sklearn``
     - ``mltsa.models`` and ``mltsa.explain``
     - Use ``get_model(...)`` plus ``analyze(...)``.
   * - ``MLTSA_tensorflow``
     - no direct v1 replacement
     - Prefer the PyTorch wrappers in ``mltsa.models`` for new work.
   * - ``CV_from_MD`` and related MD labeling scripts
     - ``mltsa.md.label_trajectories``
     - Labeling now uses only the final frame window instead of assuming a fixed trajectory length.
   * - ad hoc MD CV generation
     - ``mltsa.md.featurize_dataset``
     - Feature sets are stored appendably under ``/md/feature_sets/<name>``.
   * - notebook-driven MD analysis
     - ``mltsa.md.run_mltsa``
     - The full load, fit, explain, and save workflow is now available as a Python API.
   * - manual feature importance outputs
     - ``mltsa.explain.ExplanationResult`` and results HDF5 storage
     - Explanation outputs can be appended to a separate results file.
   * - script-specific CLI entry points
     - ``mltsa-md``
     - The CLI mirrors the labeling, feature building, and analysis workflow.

Typical migration patterns
--------------------------

Synthetic data
^^^^^^^^^^^^^^

Old approach:

.. code-block:: python

   from MLTSA_datasets.OneD_pot.OneD_pot_data import dataset, potentials

New approach:

.. code-block:: python

   from mltsa.synthetic import make_1d_dataset

   synthetic = make_1d_dataset(n_trajectories=64)

Models and feature importance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Old approach:

.. code-block:: python

   # Historical sklearn and notebook-specific helpers
   from MLTSA_sklearn import MLTSA_sk

New approach:

.. code-block:: python

   from mltsa.explain import analyze
   from mltsa.models import get_model

   model = get_model("random_forest", n_estimators=200)
   model.fit(X, y)
   explanation = analyze(model, method="native", feature_names=feature_names)

MD workflow
^^^^^^^^^^^

Old approach:

- label trajectories with legacy MD helpers
- generate CV arrays separately
- train and interpret models in notebooks

New approach:

.. code-block:: python

   from mltsa.md import featurize_dataset, label_trajectories, run_mltsa

   label_trajectories(...)
   featurize_dataset(...)
   result = run_mltsa("md_dataset.h5", "closest")

What stays legacy for now
-------------------------

- TensorFlow-specific training code
- older notebooks that still import historical packages
- historical helper modules that remain in the repository for reference during
  the migration
