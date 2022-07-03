#############################################################
MLTSA: Machine Learning Transition State Analysis repository
#############################################################

************
Introduction
************

This is a Python package to apply the MLTSA approach for relevant CV identification on Molecular Dynamics data using both Sklearn and TensorFlow modules.It also includes both a suite of 1D Potential Analytical model feature generation module for light testing and a suite of different 2D potential shapes (Spiral, Z-shaped) generation as well as the posterior feature generation by 1D projections of the 2D data. In this package you will find: 

- Data Generation Module (**MLTSA_datasets**) : Contains files with the easy to call 1D/2D/MD examples to generate data or play around with it as tests for the approach.

- Scikit-Learn-based ML models and Feature Reduction module (**MLTSA_sklearn**) : Contains the Scikit-Learn integrated functions to apply MLTSA on data.

- TensorFlow-based ML models and Feature Reduction module (**MLTSA_tensorflow**): Contains the set of functions and different models built on TensorFlow to apply MLTSA on data.

*****
Usage
*****

- Example OneD
- Example TwoD
- Example Train
- Example MLTSA

************
Installation
************

To use MLTSA, first install it using pip:

.. code-block:: console

    (.venv) $ pip install MLTSA
