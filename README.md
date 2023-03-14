# sage_acsv

This repository hosts the implementation of a SageMath package
containing algorithms for analytic combinatorics in several variables.

The package works with any reasonably recent version of SageMath, we
recommend to have SageMath 9.4 (released in August 2021) or newer.

## Quickstart

The easiest way to install the latest released version of the package
is via PyPI simply by running

```sh
sage -pip install sage-acsv
```

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
several different SageMath releases (the oldest currently being SageMath 9.4)
is checked.

