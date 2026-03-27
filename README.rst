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

The scientific implementation will be migrated into this structure over future
milestones. For now, the package provides a stable import target, version
metadata, a minimal CLI entry point, and a clean testing and packaging setup.

************
Installation
************

For local development:

.. code-block:: console

   pip install -e ".[test]"

*****
Usage
*****

.. code-block:: python

   import mltsa

   print(mltsa.__version__)

.. code-block:: console

   mltsa --version

**************
Migration Note
**************

Legacy packages such as ``MLTSA_datasets``, ``MLTSA_sklearn``, and
``MLTSA_tensorflow`` remain in the repository during the migration. They are no
longer the intended main public API for new work. New development should target
the ``mltsa`` package under ``src/``.
