Notebooks
=========

The active tutorials now live directly under the top-level ``notebooks/``
directory.

Current tutorials
-----------------

- ``01_synthetic_1d_data_generation_the_basics.ipynb``
- ``02_synthetic_2d_data_generation_added_complexity_through_ice_cream.ipynb``
- ``03_mltsa_for_dummies.ipynb``
- ``04_mltsa_on_real_data_placeholder.ipynb`` (placeholder notebook using
  public precomputed workshop CVs)
- ``05_exploring_models.ipynb``
- ``06_shap_compatible.ipynb``

Generated datasets and result files created while running the notebooks are
written to ``notebooks/_generated/``.

Legacy material
---------------

- ``notebooks/legacy/`` contains older exploratory notebooks kept for reference

Migration note
--------------

Most notebooks in ``notebooks/legacy/`` still use historical imports such as
``MLTSA_datasets`` or ``MLTSA_tensorflow``. They are preserved as reference
material during the package migration, but new examples should use ``mltsa``
directly.

For new examples, prefer the Python APIs described in the package guides and
keep notebooks small and task-focused.
