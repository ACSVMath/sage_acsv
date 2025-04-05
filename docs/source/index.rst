sage-acsv: a SageMath package for Analytic Combinatorics in Several Variables
=============================================================================

.. toctree::
   :maxdepth: 1
   :hidden:

   reference

This package provides tools for the study of the asymptotic behavior of
the sequence of coefficients of rational generating functions in
several variables along a given direction.

The public interface of our toolbox is provided by the following
functions and classes:

- :func:`.diagonal_asymptotics_combinatorial` -- the central function of the package,
- :func:`.get_expansion_terms` -- helper function for extracting
  the terms in the output expansion with more structure,
- :func:`.contributing_points_combinatorial` -- computes all contributing
  points of a combinatorial multivariate rational function,
- :func:`.minimal_critical_points_combinatorial` -- computes all non-zero minimal
  critical points of a combinatorial multivariate rational function,
- :func:`.critical_points` -- compute all critical points of a combinatorial
  multivariate rational function,
- :class:`.ACSVSettings` -- a class for managing several package-global
  settings (like the default output format for :func:`.diagonal_asymptotics_combinatorial`,
  or the backend used for Gröbner basis computations).


Quickstart
==========

The easiest way to install the latest released version of the package
is via PyPI simply by running

.. code-block:: bash

   sage -pip install sage-acsv

or, alternatively, executing a cell containing

::

   %pip install sage-acsv

in a SageMath Jupyter notebook.

The package can be run in an interactive environment in the browser
using `Binder <https://mybinder.org/v2/gh/ACSVMath/sage_acsv/HEAD>`__.
The :mod:`.asymptotics` module includes a collection of examples illustrating
how this package is used to extract coefficient asymptotics from multivariate
rational combinatorial generating functions.

An article serving as an introduction to version 0.1.0 of the package
and its internals can be found at [SageACSV23]_.


Reference Manual
================

All public-facing functions and classes in our package are documented
in our :doc:`reference manual <reference>`.


Bibliography
============

.. [SageACSV23] Benjamin Hackl, Andrew Luo, Stephen Melczer, Jesse Selover,
   and Elaine Wong,
   `Rigorous Analytic Combinatorics in Several Variables in SageMath <https://www.mat.univie.ac.at/~slc/wpapers/FPSAC2023/90.pdf>`__,
   Sém. Lothar. Combin., vol. 89B, no. 90, pp. 1–12, 2023;
   `arXiv:2303.09603 [math.CO] <https://arxiv.org/abs/2303.09603>`__.


