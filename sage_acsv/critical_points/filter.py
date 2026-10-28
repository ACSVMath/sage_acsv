
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

    * ``solutions`` -- A list of lists representing critical points.
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