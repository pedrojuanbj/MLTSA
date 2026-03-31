Installation
============

``mltsa`` targets Python 3.10 and newer.

Base install
------------

For local development inside this repository:

.. code-block:: console

   pip install -e .

For a test-ready environment:

.. code-block:: console

   pip install -e ".[test]"

MD extras
---------

The MD workflow relies on optional dependencies such as ``mdtraj`` and
``matplotlib``.

.. code-block:: console

   pip install -e ".[test,md]"

Notes
-----

- ``scikit-learn`` and ``torch`` are regular package dependencies.
- ``mdtraj`` can require platform-specific binary wheels or local build tools.
- The package CLI is available through ``mltsa`` and ``mltsa-md`` after
  installation.
