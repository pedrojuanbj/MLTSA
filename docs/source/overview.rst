Package Overview
================

The repository is now organized around a modern ``src/`` layout with the
package root at ``src/mltsa``.

Package areas
-------------

- ``mltsa.io`` provides reusable HDF5 helpers and schema utilities.
- ``mltsa.synthetic`` provides deterministic benchmark datasets with save,
  load, rebuild, and append support.
- ``mltsa.models`` provides sklearn and PyTorch model wrappers through a small
  shared factory.
- ``mltsa.explain`` provides native, permutation, and global-mean importance
  methods with HDF5 persistence.
- ``mltsa.md`` provides labeling, featurization, analysis, and export helpers
  for MD workflows.
- ``mltsa.cli`` provides thin command-line wrappers around the Python API.

Repository state
----------------

The historical research modules remain in the repository during the migration,
but they are no longer the preferred public API for new work.

New development should target ``mltsa`` directly.

Storage model
-------------

The package uses a small HDF5 schema designed for incremental workflows:

- ``/md/replicas/<replica_id>`` for lightweight per-replica metadata
- ``/md/feature_sets/<feature_set_id>`` for appendable MD feature sets
- ``/results/experiments/<experiment_id>`` for saved analyses and diagnostics

This keeps metadata scans fast and avoids loading large numeric arrays just to
check whether a result already exists.
