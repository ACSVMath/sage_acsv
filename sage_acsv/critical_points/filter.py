
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod

from sage_acsv.helpers import (
    ACSVException,
    is_contributing,
)

def _filter_direction_solutions(solutions, r_variable_values, start_index):
    r"""Filters critical point solutions for ones whose `r` values match the original
    parameters given.

    The function is assumed to have a combinatorial expansion.

    INPUT:

    * ``solutions`` -- A list of lists (coordinates), with the solutions for `r`
        variables appearing contiguously.
    * ``r_variable_values`` -- A list representing the original `r` value given.
    * ``start_index`` -- The coordinate index where solutions of `r` begin.

    OUTPUT:

    A filtered sub-list of ``solutions``.
    """
    return [
        sol for sol in solutions if
        all(
            rsol == r for rsol, r in zip(sol[start_index: ], list(r_variable_values.values()))
        )
    ]

def _filter_positive_real_minimal_contributing_point(solutions_by_dimension, H, vs, r):
    r"""Filters critical point solutions for a unique real positive minimal contributing point.

    INPUT:

    * ``solutions_by_dimension`` -- A ``dict`` mapping dimension to a list of all critical points
      lying in a stratum of that dimension, where each critical point is represented as a list of numbers.
    * ``H`` -- A polynomial for which minimality is tested against.
    * ``vs`` -- A list of variables.
    * ``r`` -- A list of numbers.

    OUTPUT:

    A list of critical points.
    """
    all_factors = list(factor[0] for factor in H.factor())
    contributing_pos_minimals = []
    for d, pos_minimals in sorted(solutions_by_dimension.items(), reverse=True):
        if len(contributing_pos_minimals) > 0:
            break

        for x in pos_minimals:
            if is_contributing(vs, x, r, all_factors, len(vs) - d):
                contributing_pos_minimals.append(x)

    if len(contributing_pos_minimals) == 0:
        raise ACSVException("No contributing points found.")
    if len(contributing_pos_minimals) > 1:
        raise ACSVException(
            f"More than one minimal contributing point with positive real coordinates found: {contributing_pos_minimals}"
        )
    return contributing_pos_minimals[0]

def _filter_contributing_points(solutions_by_dimension, H, vs, r, allow_boundary=False):
    r"""Filters critical point solutions for a unique real positive minimal contributing point.

    INPUT:

    * ``solutions_by_dimension`` -- A ``dict`` mapping dimension to a list of all critical points
        lying in a stratum of that dimension, where each critical point is represented as a list of numbers.
    * ``H`` -- A polynomial for which contributing is tested against.
    * ``vs`` -- A list of variables.
    * ``r`` -- A list of numbers.

    OUTPUT:

    A list of contributing points.
    """
    all_factors = list(factor[0] for factor in H.factor())
    contributing_points = []
    for d, critical_points in sorted(solutions_by_dimension.items(), reverse=True):
        for w in critical_points:
            if is_contributing(vs, w, r, all_factors, len(vs) - d, allow_boundary):
                contributing_points.append(w)

    return contributing_points

def _filter_minimal_points(solutions, minimal_points):
    r"""Filters critical point solutions for ones equal in modulus to a candidate minimal point.

    INPUT:

    * ``solutions`` -- A list of critical points.
    * ``minimal_points`` -- A list of candidate minimal points.

    OUTPUT:

    A list of minimal critical points.
    """
    return [
        w for w in solutions if any(
            all(
                abs(w_i) == abs(min_i) for w_i, min_i in zip(w, pt)
            ) for pt in minimal_points
        ) 
    ]

def _filter_contributing_points_by_height(points, H, vs, r):
    r"""Filters critical point solutions for ones that are contributing and separates
    ones with largest height.

    INPUT:

    * ``points`` -- A list critical points.
    * ``H`` -- A polynomial for which contributing is tested against.
    * ``vs`` -- A list of variables.
    * ``r`` -- A list of numbers.

    OUTPUT:

    A list of contributing points of largest height, and a list of all other contributing points.
    """
    Hs = [f for f, _ in H.factor()]
    # Sort all critical points by height
    cps_by_height = [(pt, prod([abs(vi)**ri for (vi, ri) in zip(pt, r)])) for pt in points]
    cps_by_height.sort(key=lambda x: x[1])

    minimal_contributing_points = []
    next_contributing_points = []

    contributing_height = None
    for cp, h in cps_by_height:
        subs_dict = {vs[i]: cp[i] for i in range(len(vs))}

        factors = [f for f in Hs if f.subs(subs_dict) == 0]
        s = len(factors)
        normals = matrix(
            [[f.derivative(v).subs(subs_dict) for v in vs] for f in factors]
        )
        if normals.rank() < s:
            raise ACSVException(
                "Not a transverse intersection. Cannot deal with this case."
            )

        # If contributing_height != None, then a contributing point has been found. Any other contributing points
        # of larger height are just for the error bound, so they don't need to be generic.
        if is_contributing(vs, cp, r, factors, s, contributing_height is not None and h > contributing_height):
            if contributing_height is None or h == contributing_height:
                contributing_height = h
                minimal_contributing_points.append(cp)
            else:
                next_contributing_points.append(cp)

    return minimal_contributing_points, next_contributing_points