# Notebooks

The notebooks are now organized by topic under the repository-level
`notebooks/` directory.

## Structure

- `notebooks/synthetic/` holds synthetic-data examples.
- `notebooks/md/` holds molecular-dynamics examples.
- `notebooks/legacy/` holds older exploratory notebooks that are still useful
  for reference during the migration.

## Migration note

Most existing notebooks still use the historical repository imports such as
`MLTSA_datasets`, `MLTSA_sklearn`, or `MLTSA_tensorflow`. They are preserved
for continuity, but new examples should prefer the modern `mltsa` package API.
