# Notebooks

The active tutorials now live directly under the repository-level `notebooks/`
directory.

## Current tutorials

- `01_synthetic_1d_data_generation_the_basics.ipynb`
- `02_synthetic_2d_data_generation_added_complexity_through_ice_cream.ipynb`
- `03_mltsa_for_dummies.ipynb`
- `04_mltsa_on_real_data_placeholder.ipynb`
- `05_exploring_models.ipynb`
- `06_shap_compatible.ipynb`

Generated files created while running the notebooks are written to
`notebooks/_generated/`.

## Legacy material

- `notebooks/legacy/` contains older exploratory notebooks kept for reference
  during the migration.

Most notebooks in `legacy/` still use the historical repository imports such as
`MLTSA_datasets`, `MLTSA_sklearn`, or `MLTSA_tensorflow`. They are preserved
for continuity, but new examples should prefer the modern `mltsa` package API.
