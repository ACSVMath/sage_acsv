from sage.functions.other import factorial
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.modules.free_module_element import vector
from sage.rings.integer_ring import ZZ
from sage.symbolic.ring import SR

from .data import ClassifiedPoint
from .general_terms import (
    _general_term_asymptotics,
    _general_term_asymptotics_complete_intersection_hyperplane,
    _general_term_asymptotics_smooth,
)
from sage_acsv.data.exceptions import ACSVException
from sage_acsv.helpers import (
    compute_hessian,
    compute_implicit_hessian,
    compute_square_root_determinant_of_hessian,
    transverse_leading_normalization,
)


def _compute_contribution_of_simple_transverse_point(
    G, data: ClassifiedPoint, expansion_precision
):
    n = SR.var("n")
    s, r, vs, cp, subs_dict = (
        data.codimension,
        data.direction,
        data.variables,
        data.point,
        data.substitutions,
    )
    unit, factors, extra_factors = data.unit, data.factors, data.extra_factors
    d = len(vs)

    Qw = compute_implicit_hessian(factors, vs, r, subs_dict)
    expansion = sum(
        term / n**term_order
        for term_order, term in enumerate(
            _general_term_asymptotics(
                G, factors, extra_factors, r, vs, cp, expansion_precision
            )
        )
    ) / unit
    constant = SR(
        1
        / compute_square_root_determinant_of_hessian(Qw)
        / (ZZ(2) ** (d - s)).sqrt()
    )

    return constant, expansion


def _compute_contribution_of_high_order_transverse_point(
    G, data: ClassifiedPoint, expansion_precision
):
    if expansion_precision > 1:
        raise ACSVException(
            "Cannot compute higher order expansions for non-simple transverse poles."
        )

    s, r, vs, cp, subs_dict = (
        data.codimension,
        data.direction,
        data.variables,
        data.point,
        data.substitutions,
    )
    unit, factors, multiplicities, extra_factors = (
        data.unit,
        data.factors,
        data.multiplicities,
        data.extra_factors,
    )
    d = len(vs)
    R = vs[0].parent()

    # Compute the gamma matrix before either branch needs its determinant.
    Gamma = matrix(
        [[(v * Q.derivative(v)).subs(subs_dict) for v in vs] for Q in factors]
        + [
            [v.subs(subs_dict) if vs.index(v) == i else 0 for i in range(d)]
            for v in vs[: d - s]
        ]
    )

    if s != d:
        Qw = compute_implicit_hessian(factors, vs, r, subs=subs_dict)
        expansion = SR(
            G.subs(subs_dict)
            / transverse_leading_normalization(factors, vs, r, cp)
            / unit
            / R(prod(extra_factors)).subs(subs_dict)
        )
        constant = SR(
            1
            / compute_square_root_determinant_of_hessian(Qw)
            / (ZZ(2) ** (d - s)).sqrt()
        )
    else:
        expansion = SR(
            G.subs(subs_dict)
            / unit
            / abs(Gamma.determinant())
            / R(prod(extra_factors)).subs(subs_dict)
        )
        constant = ZZ.one()

    mult_fac = prod(factorial(m - 1) for m in multiplicities)
    r_gamma_inv = prod(
        x ** (multiplicities[i] - 1)
        for i, x in enumerate(list(vector(r) * Gamma.inverse())[:s])
    )
    expansion *= (
        (-1) ** sum(m - 1 for m in multiplicities) * r_gamma_inv / mult_fac
    )

    return constant, expansion


def _compute_contribution_of_smooth_point(
    G, H, data: ClassifiedPoint, expansion_precision
):
    n = SR.var("n")
    r, vs, cp, subs_dict = (
        data.direction,
        data.variables,
        data.point,
        data.substitutions,
    )
    d = len(vs)

    expansion = sum(
        term / (r[-1] * n) ** term_order
        for term_order, term in enumerate(
            _general_term_asymptotics_smooth(
                G, H, r, vs, cp, expansion_precision
            )
        )
    )
    Hess = compute_hessian(H, vs, r, subs_dict)
    constant = SR(
        1
        / compute_square_root_determinant_of_hessian(Hess)
        / (r[-1] ** (d - 1) * ZZ(2) ** (d - 1)).sqrt()
    )

    return constant, expansion


def _compute_contribution_of_complete_hyperplane_point(
    G, data: ClassifiedPoint, expansion_precision
):
    n = SR.var("n")
    r, vs, cp, subs_dict = (
        data.direction,
        data.variables,
        data.point,
        data.substitutions,
    )
    unit, factors, multiplicities = (
        data.unit,
        data.factors,
        data.multiplicities,
    )

    expansion = sum(
        term / n**term_order
        for term_order, term in enumerate(
            _general_term_asymptotics_complete_intersection_hyperplane(
                G, factors, multiplicities, r, vs, cp, expansion_precision
            )
        )
    ) / unit.subs(subs_dict)

    return ZZ.one(), expansion
