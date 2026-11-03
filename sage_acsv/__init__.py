r"""A SageMath package for Analytic Combinatorics in Several Variables"""

import importlib.metadata
__version__ = importlib.metadata.version(__name__)

from sage_acsv.asymptotics import (
    central_limit_theorem_combinatorial,
    compute_asymptotics_at_points,
    diagonal_asy,
    diagonal_asymptotics_combinatorial,
    diagonal_asymptotics_hyperplane,
)
from sage_acsv.critical_points import (
    MinimalCriticalCombinatorial,
    contributing_points_combinatorial,
    contributing_points_combinatorial_smooth,
    contributing_points_hyperplane,
    critical_points,
    minimal_critical_points_combinatorial,
)
from sage_acsv.kronecker import kronecker_representation
from sage_acsv.helpers import get_expansion_terms, get_limit_theorem_terms, LimitTheoremTerm
from sage_acsv.settings import ACSVSettings
from sage_acsv.algebraic import algebraic_diagonal

__all__ = [
    "diagonal_asy",
    "diagonal_asymptotics_combinatorial",
    "diagonal_asymptotics_hyperplane",
    "compute_asymptotics_at_points",
    "central_limit_theorem_combinatorial",
    "contributing_points_combinatorial",
    "contributing_points_combinatorial_smooth",
    "minimal_critical_points_combinatorial",
    "MinimalCriticalCombinatorial",
    "critical_points",
    "contributing_points_hyperplane",
    "kronecker_representation",
    "get_expansion_terms",
    "get_limit_theorem_terms",
    "LimitTheoremTerm",
    "ACSVSettings",
    "algebraic_diagonal",
]
