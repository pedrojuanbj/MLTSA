Synthetic Datasets
==================

The synthetic module provides deterministic benchmark data for method
development and regression testing.

Main entry points
-----------------

- ``mltsa.synthetic.make_1d_dataset(...)``
- ``mltsa.synthetic.make_2d_dataset(...)``
- ``mltsa.synthetic.load_dataset(path)``
- ``mltsa.synthetic.SyntheticDataset``

Typical workflow
----------------

.. code-block:: python

   from mltsa.synthetic import make_1d_dataset, load_dataset

   dataset = make_1d_dataset(n_trajectories=32, n_steps=64, n_features=12)
   dataset.save("synthetic.h5", overwrite=True)

   restored = load_dataset("synthetic.h5")
   rebuilt = restored.rebuild_exact()
   more = restored.generate_more(8)
   combined = restored.append(more)

What is stored
--------------

Synthetic datasets persist:

- ``X`` and ``y``
- feature names
- generation parameters
- system definition metadata
- relevant feature indices
- time-dependent relevance when available
- per-trajectory seeds
- latent trajectories when available

That metadata is enough to rebuild the dataset exactly and to generate more
trajectories from the same system definition.
