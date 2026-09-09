

from sage.algebras.weyl_algebra import DifferentialWeylAlgebra
from sage.arith.misc import gcd
from sage.arith.functions import lcm
from sage.arith.srange import srange
from sage.functions.log import log, exp
from sage.functions.other import factorial
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.misc.prandom import shuffle
from sage.modules.free_module_element import vector
from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing
from sage.rings.complex_interval_field import ComplexIntervalField
from sage.rings.ideal import Ideal
from sage.rings.imaginary_unit import I
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.qqbar import AA, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.constants import pi
from sage.symbolic.ring import SR

from sage_acsv.kronecker import _kronecker_representation
from sage_acsv.helpers import (
    ACSVException,
    Term,
    is_contributing,
    compute_newton_series,
    compute_newton_series_general,
    rational_function_reduce,
    compute_hessian,
    compute_implicit_hessian,
    compute_square_root_determinant_of_hessian,
    collapse_zero_part,
    transverse_leading_normalization,
)
from sage_acsv.debug import Timer, acsv_logger
from sage_acsv.settings import ACSVSettings, OutputFormat
from sage_acsv.whitney import whitney_stratification
from sage_acsv.groebner import compute_primary_decomposition, compute_saturation

def _subs(F, u, k):
    """
    Monkey patch for Sage's algebraic numbers implementation being slow.
    """
    CIF = ComplexIntervalField()
    a = F.subs({u:k}); CIF(a)
    return a


def _prepare_symbolic_fraction(F):
    r"""Extract polynomial numerators and denomiators from a symbolic fraction,
    including the replacement of variable names.

    INPUT:

    * ``F`` -- A symbolic fraction
    """
    G, H = F.numerator(), F.denominator()
    original_variables = H.variables()
    if any(v not in original_variables for v in G.variables()):
        raise ValueError(f"Numerator {G} has variables not in denominator {H}")
    new_variables = list(SR.var("acsvvar", len(original_variables)))
    variable_map = {v: new_v for v, new_v in zip(original_variables, new_variables)}
    G = G.subs(variable_map)
    H = H.subs(variable_map)
    frac_gcd = gcd(G, H)
    return G / frac_gcd, H / frac_gcd, variable_map


def _dict_to_variable_order(F, d, default=1):
    r"""Converts a direction dictionary `r` to a direction vector in the variable order
    given by `F.variables()`

    INPUT:

    * ``F`` -- A symbolic fraction
    * ``r`` -- A direction dictionary
    """

    vs = F.variables()
    if default is None and any(v not in d for v in vs):
        raise ValueError(f"Provided dictionary {d} is missing entries for some variables {vs}.")
    return [d.get(v, default) for v in vs]


def _prepare_expanded_polynomial_ring(variables, direction=None, include_t=True):
    r"""Prepare an auxiliary polynomial ring for computing diagonal asymptotics.

    INPUT:

    * ``variables`` -- variables in the rational function `F`
    * ``direction`` -- (Optional) direction vector `r` for the asymptotics,
      defaults to the diagonal (all ones).
    * ``include_t`` -- (Optional) whether to include the auxiliary variable `t`
      in the expanded ring.
    """
    # if direction r is not given, default to the diagonal
    if direction is None:
        direction = [1 for _ in variables]

    replaced_direction = copy(direction)

    # in case there are irrational numbers in the direction vector,
    # replace them with polynomial variables
    direction_variable_values = {}
    for idx, dir_entry in enumerate(replaced_direction):
        if AA(dir_entry).minpoly().degree() > 1:
            dir_var = SR.var(f"r{idx}")
            direction_variable_values[dir_var] = AA(dir_entry)
            replaced_direction[idx] = dir_var

    # auxiliary variables
    auxiliary_variables = []
    if include_t:
        auxiliary_variables.append(SR.var("t"))
    auxiliary_variables.append(SR.var("lambda_"))
    auxiliary_variables.append(SR.var("u_"))

    # create the expanded polynomial ring
    expanded_ring = PolynomialRing(
        QQ, list(variables) + list(direction_variable_values) + auxiliary_variables
    )
    variables = [expanded_ring(v) for v in variables]
    auxiliary_variables = [expanded_ring(v) for v in auxiliary_variables]
    replaced_direction = [expanded_ring(ri) for ri in replaced_direction]
    direction_variable_values = {
        expanded_ring(ri): val for (ri, val) in direction_variable_values.items()
    }
    return (
        expanded_ring,
        variables,
        auxiliary_variables,
        replaced_direction,
        direction_variable_values,
    )
