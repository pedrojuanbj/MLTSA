Notebooks
=========

The repository notebooks now live under the top-level ``notebooks/`` directory
with a small topic-based structure.

Layout
------

- ``notebooks/synthetic/`` contains synthetic-data examples
- ``notebooks/md/`` contains MD-focused examples
- ``notebooks/legacy/`` contains older exploratory notebooks kept for reference

Migration note
--------------

Most of the existing notebooks still use historical imports such as
``MLTSA_datasets`` or ``MLTSA_tensorflow``. They are preserved as reference
material during the package migration and will be rewritten incrementally to
use ``mltsa`` directly.

For new examples, prefer the Python APIs described in the package guides and
keep notebooks small and task-focused.
