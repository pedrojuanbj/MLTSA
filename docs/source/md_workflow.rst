MD Workflow
===========

The MD workflow is organized as a small sequence of reusable steps.

1. Label trajectories
---------------------

Use ``mltsa.md.label_trajectories(...)`` to classify each trajectory as
``IN``, ``OUT``, or ``TS`` based on the mean rule value over the final
``window_size`` frames.

2. Build feature sets
---------------------

Use ``mltsa.md.featurize_dataset(...)`` to create one named feature set inside
the same MD HDF5 file.

Supported v1 feature families:

- ``closest_residue_distances``
- ``all_ligand_protein_distances``
- ``bubble_distances``
- ``contact_map``
- ``pca_xyz``

3. Train and explain
--------------------

Use ``mltsa.md.run_mltsa(...)`` to load a selected feature set, fit a model,
compute feature importances, and optionally save results to a separate results
HDF5.

4. Export structures
--------------------

Use ``mltsa.md.export_state_structures(...)`` to write multi-model ``IN.pdb``,
``OUT.pdb``, and ``TS.pdb`` files from an existing label experiment.

Minimal example
---------------

.. code-block:: python

   from mltsa.md import featurize_dataset, label_trajectories, run_mltsa

   label_trajectories(
       trajectory_paths=["traj_0001.dcd", "traj_0002.dcd"],
       topology="topology.pdb",
       h5_path="md_dataset.h5",
       experiment_id="labels",
       rule="sum_distances",
       selection_pairs=[("index 10", "index 220")],
       lower_threshold=0.4,
       upper_threshold=0.8,
       window_size=25,
   )

   featurize_dataset(
       h5_path="md_dataset.h5",
       feature_set="closest",
       feature_type="closest_residue_distances",
       label_experiment_id="labels",
   )

   result = run_mltsa(
       "md_dataset.h5",
       "closest",
       model="random_forest",
       explanation_method="native",
       results_h5_path="md_results.h5",
       experiment_id="rf_native",
   )

   print(result.training_score)
