r"""Compatibility facade for the moved asymptotic-contribution algorithms.

New code should import from :mod:`sage_acsv.asymptotics` or its implementation
modules. This module preserves the pre-refactor import paths.
"""

from sage_acsv.asymptotics.contributions import (
    PointsStrategy,
    _compute_asymptotic_contributions_at_points,
    _compute_asymptotics_at_points_hyperplane,
    _compute_asymptotics_at_points_smooth,
    _compute_asymptotics_at_points_transverse,
    _format_output,
    compute_asymptotics_at_points,
)
from sage_acsv.asymptotics.data import ClassifiedPoint, LocalPointStrategy
from sage_acsv.asymptotics.general_terms import (
    _general_term_asymptotics,
    _general_term_asymptotics_complete_intersection_hyperplane,
    _general_term_asymptotics_smooth,
)

# Preserve the former private name for internal and downstream callers.
_compute_asymptotics_at_points = _compute_asymptotic_contributions_at_points

__all__ = [
    "PointsStrategy",
    "compute_asymptotics_at_points",
    "LocalPointStrategy",
    "ClassifiedPoint",
]
