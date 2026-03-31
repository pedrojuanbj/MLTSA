CLI Usage
=========

The command line intentionally stays thin and delegates directly to the Python
API.

Root command
------------

.. code-block:: console

   mltsa --version

MD commands
-----------

The dedicated MD CLI is available as ``mltsa-md``. The root command also
supports ``mltsa md ...`` as a shortcut.

Label trajectories
------------------

.. code-block:: console

   mltsa-md label ^
     --h5 md_dataset.h5 ^
     --experiment labels ^
     --topology topology.pdb ^
     --trajectory traj_0001.dcd ^
     --trajectory traj_0002.dcd ^
     --rule sum_distances ^
     --selection-pair "index 10" "index 220" ^
     --lower-threshold 0.4 ^
     --upper-threshold 0.8 ^
     --window-size 25

Build a feature set
-------------------

.. code-block:: console

   mltsa-md build ^
     --h5 md_dataset.h5 ^
     --feature-set closest ^
     --feature-type closest_residue_distances ^
     --label-experiment labels

Run analysis
------------

.. code-block:: console

   mltsa-md analyze ^
     --h5 md_dataset.h5 ^
     --feature-set closest ^
     --model random_forest ^
     --explain native ^
     --results-h5 md_results.h5 ^
     --experiment rf_native

Keyword arguments
-----------------

The ``build`` and ``analyze`` commands support repeated ``KEY=VALUE`` options
for model or explanation parameters where appropriate.
