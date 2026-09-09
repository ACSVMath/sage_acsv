r"""A SageMath package for Analytic Combinatorics in Several Variables"""

import importlib.metadata
__version__ = importlib.metadata.version(__name__)

from sage_acsv.asymptotics import (
    diagonal_asy,
    diagonal_asymptotics_combinatorial,
    diagonal_asymptotics_hyperplane
)
from sage_acsv.asymptotic_terms import compute_asymptotics_at_points
from sage_acsv.central_limit_theorem import central_limit_theorem_combinatorial
from sage_acsv.critical_points import (
    contributing_points_combinatorial,
    contributing_points_combinatorial_smooth,
    minimal_critical_points_combinatorial,
    MinimalCriticalCombinatorial,
    critical_points
)
from sage_acsv.kronecker import kronecker_representation
from sage_acsv.helpers import get_expansion_terms, get_limit_theorem_terms, LimitTheoremTerm
from sage_acsv.settings import ACSVSettings
from sage_acsv.algebraic import algebraic_diagonal
