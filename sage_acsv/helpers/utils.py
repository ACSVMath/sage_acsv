from copy import copy

from sage.arith.misc import gcd
from sage.functions.other import ceil
from sage.misc.prandom import randint
from sage.rings.complex_interval_field import ComplexIntervalField
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import AA, AlgebraicNumber, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.ring import SR


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

def _collapse_zero_part(algebraic_number: AlgebraicNumber) -> AlgebraicNumber:
    if algebraic_number.real().is_zero():
        algebraic_number = QQbar(algebraic_number.imag()) * QQbar(-1).sqrt()
    if algebraic_number.imag().is_zero():
        algebraic_number = QQbar(algebraic_number.real())
    algebraic_number.simplify()
    return algebraic_number


def _rational_function_reduce(G, H):
    r"""Reduction of the rational function `G/H` by dividing `G` and `H` by their GCD.

    INPUT:

    * ``G``, ``H`` -- polynomials

    OUTPUT:

    A tuple ``(G/g, H/g)``, where ``g`` is the GCD of ``G`` and ``H``.
    """
    g = gcd(G, H)
    return G / g, H / g


def _generate_linear_form(system, vsT, u_, linear_form=None):
    r"""Generate a linear form for the input system.

    This is an integer linear combination of the variables that,
    with high probability, takes unique values on the solutions of
    the system.

    INPUT:

    * ``system`` -- A polynomial system of equations
    * ``vsT`` -- A list of variables in the system
    * ``u_`` -- A variable not in the system
    * ``linear_form`` -- (Optional) A precomputed linear form in the
      variables of the system. If passed, the returned form is
      based on the given linear form and not randomly generated.

    OUTPUT:

    A linear form in ``u_`` and the variables of ``vsT``.
    """
    if linear_form is not None:
        return u_ - linear_form

    maxcoeff = ceil(
        max([max([abs(x) for x in f.coefficients()]) for f in system if f != 0])
    )
    maxdegree = max([f.degree() for f in system])
    return u_ - sum(
        [
            randint(-maxcoeff * maxdegree - 31, maxcoeff * maxdegree + 31) * z
            for z in vsT
        ]
    )