from sage.arith.misc import gcd, xgcd
from sage.rings.qqbar import AA, QQbar

from sage_acsv.critical_points.filter import _filter_direction_solutions, _filter_non_combinatorial_kronecker_roots_by_conjugate_pairs
from sage_acsv.helpers.utils import (
    _subs,
    _collapse_zero_part,
)

def solve_kronecker_system(
    vs,
    r_variable_values,
    kronecker_representation
):
    r"""Solves a critical point system given its kronecker representation.

    INPUT:

    * ``vs`` -- A list of the original variables of the system.
    * ``r_variable_values`` -- A list of numbers representing values of direction variables.
    * ``kronecker_representation`` -- A tuple ``(P, Qs)`` where ``P`` is univariate polynomial
      and ``Qs`` is a list of univariate polynomials.

    OUTPUT:

    A list of solutions to the system.
    """
    P, Qs = kronecker_representation
    Pd = P.derivative()

    # Keep auxiliary direction coordinates until filtering, then return only
    # the original critical-point coordinates.
    solutions = []
    for u in P.roots(QQbar, multiplicities=False):
        solutions.append(
            [
                _collapse_zero_part(QQbar((q / Pd).subs(u_=u)))
                for q in Qs
            ]
        )

    solutions = _filter_direction_solutions(
        solutions, r_variable_values, len(vs)
    )
    return [solution[: len(vs)] for solution in solutions]


def solve_and_filter_positive_minimal_solutions_combinatorial(
    vs,
    r_variable_values,
    kronecker_representation,
    Qt
):
    r"""Solves a critical point system and filters for minimal positive solutions

    INPUT:

    * ``vs`` -- A list of the original variables of the system.
    * ``r_variable_values`` -- A list of numbers representing values of direction variables.
    * ``kronecker_representation`` -- A tuple ``(P, Qs)`` where ``P`` is univariate polynomial
        and ``Qs`` is a list of univariate polynomials. \
    * ``Qt`` -- A univariate polynomial corresponding to the auxiliary variable `t`.

    OUTPUT:

    A reduced kronecker representation excluding `t \ne 1` and a list of positive
    minimal solutions.
    """
    P, Qs = kronecker_representation
    u_ = P.parent().gen()
    Pd = P.derivative()

    # Solutions to Pt are solutions to the system where t is not 1
    one_minus_t = gcd(Pd - Qt, P)
    Pt, _ = P.quo_rem(one_minus_t)
    rts_t_zo = list(
        filter(
            lambda k: _subs(Qt, u_, k) / _subs(Pd, u_, k) > 0 and _subs(Qt, u_, k) / _subs(Pd, u_, k) < 1,
            Pt.roots(AA, multiplicities=False),
        )
    )
    non_min = [[(q / Pd).subs(u_=u) for q in Qs] for u in rts_t_zo]

    # Filter the real roots for minimal points with positive coords
    pos_minimals = list(
        [(q / Pd).subs(u_=u) for q in Qs]
        for u in one_minus_t.roots(AA, multiplicities=False)
    )
    pos_minimals = _filter_direction_solutions(
        pos_minimals, r_variable_values, len(vs)
    )
    # Positivity applies only to the original variables. Direction variables
    # are matched above, and the Lagrange multiplier need only be nonzero.
    lambda_index = len(vs) + len(r_variable_values)
    pos_minimals = [
        v for v in pos_minimals if all(
            w > 0 for w in v[: len(vs)]
        ) and v[lambda_index] != 0 and not any(
            all(a == b for a, b in list(zip(v, pt))[:len(vs)])
            for pt in non_min
        )
    ]

    return (
        one_minus_t,
        [q * one_minus_t.derivative() * xgcd(Pd, one_minus_t)[1] for q in Qs],
        [
            [
                _collapse_zero_part(v)
                for v in pt[:len(vs)]
            ]
            for pt in pos_minimals
        ]
    )

def solve_and_filter_minimal_solutions_general(
    vs,
    r_variable_values,
    kronecker_representation,
    Qt
):
    r"""Solves a critical point system and filters for minimal positive solutions

    INPUT:

    * ``vs`` -- A list of the original variables of the system.
    * ``r_variable_values`` -- A list of numbers representing values of direction variables.
    * ``kronecker_representation`` -- A tuple ``(P, Qs)`` where ``P`` is univariate polynomial
        and ``Qs`` is a list of univariate polynomials. \
    * ``Qt`` -- A univariate polynomial corresponding to the auxiliary variable `t`.

    OUTPUT:

    A reduced kronecker representation excluding `t \ne 1` and a list of positive
    minimal solutions.
    """
    P, Qs = kronecker_representation
    u_ = P.parent().gen()
    Pd = P.derivative()

    d = len(vs)

    Qas = Qs[:d]
    Qbs = Qs[d:2*d]
    Qxs = Qs[2*d:3*d]
    Qys = Qs[3*d:4*d]

    # Solutions to Pt are solutions to the system where t is not 1
    one_minus_t = gcd(Pd - Qt, P)
    Pt, _ = P.quo_rem(one_minus_t)
    Pt_roots = _filter_non_combinatorial_kronecker_roots_by_conjugate_pairs(
        Pt.roots(QQbar, multiplicities=False),
        Pd, Qas, Qbs, Qxs, Qys
    )
    rts_t_zo = list(
        filter(
            lambda k: (_subs(Qt, u_, k) / _subs(Pd, u_, k)) in AA and 
                      (_subs(Qt, u_, k) / _subs(Pd, u_, k)) > 0 and 
                      (_subs(Qt, u_, k) / _subs(Pd, u_, k)) < 1,
            Pt_roots,
        )
    )
    non_min = [[(q / Pd).subs(u_=u) for q in Qas] for u in rts_t_zo]

    # TODO: Reduce kronecker representation to original non-auxiliary system
    P_reduced = gcd(one_minus_t, )
    Qsa_reduced = []
    Qsb_reduced = []

    one_minus_t_roots = _filter_non_combinatorial_kronecker_roots_by_conjugate_pairs(
        one_minus_t.roots(QQbar, multiplicities=False),
        Pd, Qas, Qbs, Qxs, Qys
    )
    minimal_candidates = list(
        [(q / Pd).subs(u_=u) for q in Qas]
        for u in one_minus_t_roots
    )
    minimal_candidates = _filter_direction_solutions(
        minimal_candidates, r_variable_values, 4*d
    )

    # Positivity applies only to the original variables. Direction variables
    # are matched above, and the Lagrange multiplier need only be nonzero.
    lambda_index = 4 * d + len(r_variable_values)
    minimals = [
        v for v in minimal_candidates if 
        v[lambda_index] != 0 and not any(
            all(a == b for a, b in list(zip(v, pt))[:len(vs)])
            for pt in non_min
        )
    ]

    return (
        one_minus_t,
        [q * one_minus_t.derivative() * xgcd(Pd, one_minus_t)[1] for q in Qs],
        [
            [
                _collapse_zero_part(v)
                for v in pt[:len(vs)]
            ]
            for pt in minimals
        ]
    )