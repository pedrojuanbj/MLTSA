Models and Explainability
=========================

Model factory
-------------

The simplest way to create a supported model is through
``mltsa.models.get_model(...)``.

.. code-block:: python

   from mltsa.models import get_model

   model = get_model("random_forest", n_estimators=200, max_depth=6)

Supported wrappers
------------------

- sklearn: random forest, gradient boosting, histogram gradient boosting, and
  extra trees
- torch: MLP, LSTM, and 1D CNN

Explainability
--------------

The explainability layer works with the ``mltsa`` wrappers and with compatible
external fitted estimators.

.. code-block:: python

   from mltsa.explain import analyze

   result = analyze(
       model,
       method="permutation",
       X=X,
       y=y,
       feature_names=feature_names,
       n_repeats=10,
   )

Available methods
-----------------

- ``native`` for built-in model importances
- ``permutation`` for sklearn permutation importance
- ``global_mean`` for feature replacement by the global mean

The returned ``ExplanationResult`` can be saved to a results HDF5 file through
``result.save(...)``.
