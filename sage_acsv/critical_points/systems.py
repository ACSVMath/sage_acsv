from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.rings.ideal import Ideal
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ

from sage_acsv.critical_points.data import CriticalPointProblem, CriticalPointSystem
from sage_acsv.helpers import (
    ACSVException,
)
from sage_acsv.debug import Timer
from sage_acsv.whitney import whitney_stratification
from sage_acsv.groebner.functions import compute_primary_decomposition, compute_saturation

def build_stratified_critical_point_systems(
    problem: CriticalPointProblem, whitney_strat=None
) -> dict[int, list[CriticalPointSystem]]:
    r"""Builds all the critical point systems necessary to a particular instance of a
    ``CriticalPointProblem``.

    INPUT:

    * ``problem`` -- A ``CriticalPointProblem`` object.
    * ``whitney_strat`` -- (Optional) If known / precomputed, a
      Whitney Stratification of `V(H)`. The program will not check if
      this stratification is correct. Should be a list of length `d`, where
      the `k`-th entry is a list of tuples of ideal generators representing
      a component of the `k`-dimensional stratum.

    OUTPUT:

    A dictionary mapping dimension to all critical point systems in that dimension.
    """

    timer = Timer()

    if problem.require_minimal and problem.t_variable is None:
        raise ACSVException("Auxiliary t variable must be provided for minimality checking.")

    H = problem.H_polynomial
    vs, r, t, lambda_ = problem.variables, problem.direction, problem.t_variable, problem.lambda_variable
    r_polys = problem.r_polynomials
    R = vs[0].parent()

    timer.checkpoint()
    pure_H = PolynomialRing(QQ, vs)
    if whitney_strat is None:
        whitney_strat = whitney_stratification(Ideal(pure_H(H)), pure_H)
    else:
        # Cast symbolic generators for provided stratification into the correct ring
        whitney_strat = [
            prod([Ideal([pure_H(f) for f in comp]) for comp in stratum])
            for stratum in whitney_strat
        ]
    timer.checkpoint("Whitney Stratification")

    critical_point_ideals = {}
    for d, stratum in enumerate(whitney_strat):
        critical_point_ideals[d] = []
        for id in compute_primary_decomposition(stratum):
            id = id.change_ring(R)
            c = len(vs) - d
            M = matrix([[v * f.derivative(v) for v in vs] for f in id.gens()] + [r])
            # Create ideal of R containing extended critical point equations
            cpid = id + Ideal(
                M.minors(c + 1)
                + r_polys +  [prod(vs) * lambda_ - 1]
            )

            if problem.require_minimal:
                cpid += Ideal([H.subs({v: v * t for v in vs})])

            # Saturate cpid by lower dimension stratum, if d > 0
            if d > 0:
                cpid = compute_saturation(
                    cpid, whitney_strat[d - 1].change_ring(R)
                )

            critical_point_ideals[d].append(
                CriticalPointSystem(
                    generators=cpid.gens(),
                    parent_ideal=id,
                    parent_dimension=d,
                    parent_stratum=(
                        stratum.change_ring(R),
                        whitney_strat[d-1].change_ring(R) if d > 0 else Ideal(R.one())
                    )
                )
            )
    timer.checkpoint("Critical Point Ideals")

    return critical_point_ideals

def build_smooth_critical_point_system(
        problem: CriticalPointProblem
) -> list[object]:
    H = problem.H_polynomial
    vs, r, t, lambda_ = problem.variables, problem.direction, problem.t_variable, problem.lambda_variable
    r_polys = problem.r_polynomials
    R = vs[0].parent()

    vsH = H.variables()
    system = [
        H_var * H.derivative(H_var) - r_var * lambda_
        for H_var, r_var in zip(H.variables(), r)
    ]
    system.extend([H, H.subs({z: z * t for z in vsH})])
    system.extend(r_polys)

    return system