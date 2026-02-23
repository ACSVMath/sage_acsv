# sage_acsv

![GitHub Release](https://img.shields.io/github/v/release/ACSVMath/sage_acsv)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ACSVMath/sage_acsv/docs-main.yml?label=docs)](https://acsvmath.github.io/sage_acsv/)
![GitHub branch check runs](https://img.shields.io/github/check-runs/ACSVMath/sage_acsv/main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ACSVMath/sage_acsv/HEAD)


This repository hosts the implementation of a SageMath package
containing algorithms for analytic combinatorics in several variables.

The package works with any reasonably recent version of SageMath, we
recommend to have SageMath 9.4 (released in August 2021) or newer.
Documentation is available at <https://acsvmath.github.io/sage_acsv/>.


## Quickstart

The easiest way to install the latest released version of the package
is via PyPI simply by running

```sh
sage -pip install sage-acsv
```

The package can be run in an interactive environment in the browser
using [Binder](https://mybinder.org/v2/gh/ACSVMath/sage_acsv/HEAD).

An article serving as an introduction to version 0.1.0 of the package
and its internals can be found on the [arXiv](https://arxiv.org/abs/2303.09603).


## Installation from source

To install the package from the source code, either clone the git repository
and run the command
```sh
sage -pip install .
```
from the root directory, i.e., the directory containing the `pyproject.toml`
file.

For development, use `sage -pip install -e .` for an editable installation.

Alternatively, to install the latest version of the main branch directly from
the GitHub repository, run
```sh
sage -pip install git+https://github.com/ACSVMath/sage_acsv.git
```


## Running package tests 

The doctests that are added in the package can be run by executing
```sh
sage -t sage_acsv
```
from the root of the cloned repository. The tests are run automatically
on every push and for any PR to the `main` branch, and compatibility with
several different SageMath releases (see [recent workflows to see all tested versions](https://github.com/ACSVMath/sage_acsv/actions/workflows/ci.yml))
is checked.

