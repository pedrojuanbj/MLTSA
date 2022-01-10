.. MLTSA documentation master file, created by
   sphinx-quickstart on Tue Dec 28 19:09:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MLTSA's documentation!
=================================

MLTSA is a python-based package which enables users to apply the Machine Learning Transition State Analysis
from https://doi.org/10.1101/2021.09.08.459492 to any given data with an array of ML models, as well as generating an
analytical model on demand for detecting relevant features in a dataset.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   onedfunctions

   sklearnfunctions

.. include:: ../../README.rst

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   MLTSA_examples


.. note::
   This project is under active development, do not expect a stable version, code is provided as is.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
