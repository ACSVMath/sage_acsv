from sage.arith.misc import gcd
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.rings.ideal import Ideal
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import AA, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.ring import SR

from sage_acsv.kronecker import _kronecker_representation
from sage_acsv.helpers import (
    ACSVException,
    is_contributing,
    collapse_zero_part,
)
from sage_acsv.debug import Timer, acsv_logger
from sage_acsv.whitney import whitney_stratification
from sage_acsv.groebner import compute_primary_decomposition, compute_saturation
from sage_acsv.utils import (
    _prepare_expanded_polynomial_ring, 
    _prepare_symbolic_fraction, 
    _dict_to_variable_order, 
    _subs
)

def contributing_points_combinatorial(
    F,
    r=None,
    linear_form=None,
    whitney_strat=None,
):
    r"""Compute contributing points of a multivariate
    rational function `F=G/H` admitting a finite number of critical points.

    The function is assumed to have a combinatorial expansion.

    INPUT:

    * ``F`` -- Symbolic fraction, the rational function assumed to have
      a finite number of critical points.
    * ``r`` -- (Optional) Length `d` vector or dictionary of positive integers.
      If a vector is given, assumes the variable order is given by ``F.variables()``.
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions.
    * ``whitney_strat`` -- (Optional) If known / precomputed, a
      Whitney Stratification of `V(H)`. The program will not check if
      this stratification is correct. Should be a list of length `d`, where
      the `k`-th entry is a list of tuples of ideal generators representing
      a component of the `k`-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming `F` has a finite number of critical points)
    the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import contributing_points_combinatorial
        sage: var('x y')
        (x, y)
        sage: pts = contributing_points_combinatorial(1/((1-(2*x+y)/3)*(1-(3*x+y)/4)))
        sage: sorted(pts)
        [[3/4, 3/2]]

    """
    if isinstance(r, dict):
        r = _dict_to_variable_order(F, r)

    G, H, variable_map = _prepare_symbolic_fraction(F)
    variables = list(variable_map.values())
    if whitney_strat is not None:
        whitney_strat = [
            [
                tuple(SR(gen).subs(variable_map) for gen in component)
                for component in stratum
            ]
            for stratum in whitney_strat
        ]
    if linear_form is not None:
        linear_form = SR(linear_form).subs(variable_map)

    return _find_contributing_points_combinatorial(
        G,
        H,
        variables,
        r=r,
        linear_form=linear_form,
        whitney_strat=whitney_strat,
    )


def _find_contributing_points_combinatorial(
    G,
    H,
    variables,
    r=None,
    linear_form=None,
    whitney_strat=None,
):
    r"""Compute contributing points of a combinatorial multivariate
    rational function `F=G/H` admitting a finite number of critical points where the singular variety is the transverse union of smooth varieties.

    Typically, this function is called as a subroutine of :func:`.diagonal_asymptotics_combinatorial`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with ``F = G/H``
    * ``variables`` -- List of variables of ``G`` and ``H``
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions
    * ``whitney_strat`` -- (Optional) If known / precomputed, a
      Whitney Stratification of `V(H)`. The program will not check if
      this stratification is correct. Should be a list of length ``d``, where
      the ``k``-th entry is a list of tuples of ideal generators representing
      a component of the ``k``-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.
    """
    (
        expanded_R,
        vs,
        (t, lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r)
    H = expanded_R(H)
    vsT = vs + list(r_variable_values.keys()) + [t, lambda_]

    # Compute the critical point system for each stratum
    pure_H = PolynomialRing(QQ, vs)

    if whitney_strat is None:
        whitney_strat = whitney_stratification(Ideal(pure_H(H)), pure_H)
    else:
        # Cast symbolic generators for provided stratification into the correct ring
        whitney_strat = [
            prod([Ideal([pure_H(f) for f in comp]) for comp in stratum])
            for stratum in whitney_strat
        ]

    critical_point_ideals = []
    for d, stratum in enumerate(whitney_strat):
        critical_point_ideals.append([])
        for P in compute_primary_decomposition(stratum):
            c = len(vs) - d
            P_ext = P.change_ring(expanded_R)
            M = matrix([[v * f.derivative(v) for v in vs] for f in P_ext.gens()] + [r])
            # Add in min polys for the direction variables
            r_polys = [
                direction_value.minpoly().subs(direction_var)
                for direction_var, direction_value in r_variable_values.items()
            ]
            # Create ideal of expanded_R containing extended critical point equations
            cpid = P_ext + Ideal(
                M.minors(c + 1)
                + [H.subs({v: v * t for v in vs}), (prod(vs) * lambda_ - 1)]
                + r_polys
            )
            # Saturate cpid by lower dimension stratum, if d > 0
            if d > 0:
                cpid = compute_saturation(
                    cpid, whitney_strat[d - 1].change_ring(expanded_R)
                )

            critical_point_ideals[-1].append((P, cpid))

    # Final minimal critical points with positive coordinates on each stratum
    critical_points_by_stratum = {}
    pos_minimals_by_stratum = {}
    for d in reversed(range(len(critical_point_ideals))):
        ideals = critical_point_ideals[d]
        critical_points_by_stratum[d] = []
        pos_minimals_by_stratum[d] = []

        for _, ideal in ideals:
            if ideal.dimension() < 0:
                continue
            P, Qs = _kronecker_representation(ideal.gens(), u_, vsT, linear_form)

            Qt = Qs[-2]  # Qs ordering is H.variables() + rvars + [t, lambda_]
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
            non_min = [[(q / Pd).subs(u_=u) for q in Qs[0:-2]] for u in rts_t_zo]

            # Filter the real roots for minimal points with positive coords
            pos_minimals = []
            for u in one_minus_t.roots(AA, multiplicities=False):
                is_min = True
                v = [(q / Pd).subs(u_=u) for q in Qs[: len(vs)]]
                rv = {
                    ri: (q / Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
                }
                if any(
                    rv[ri] != ri_value for ri, ri_value in r_variable_values.items()
                ):
                    continue
                if any(value <= 0 for value in v[: len(vs)]):
                    continue
                for pt in non_min:
                    if all(a == b for (a, b) in zip(v, pt)):
                        is_min = False
                        break
                if is_min:
                    pos_minimals.append(u)

            pos_minimals_by_stratum[d].extend(
                [
                    [
                        collapse_zero_part(QQbar((q / Pd).subs(u_=u)))
                        for q in Qs[: len(vs)]
                    ]
                    for u in pos_minimals
                ]
            )

            # Characterize all complex critical points in each stratum
            for u in one_minus_t.roots(QQbar, multiplicities=False):
                rv = {
                    ri: (q / Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
                }
                if any(
                    rv[r_var] != r_value for r_var, r_value in r_variable_values.items()
                ):
                    continue

                w = [
                    collapse_zero_part(QQbar((q / Pd).subs(u_=u)))
                    for q in Qs[: len(vs)]
                ]
                critical_points_by_stratum[d].append(w)

    # Refine positive minimal critical points to those that are contributing
    contributing_pos_minimals = []
    all_factors = list(factor[0] for factor in H.factor())
    r = [QQbar(r_variable_values.get(ri, ri)) for ri in r]
    for d in reversed(range(len(critical_point_ideals))):
        pos_minimals = pos_minimals_by_stratum[d]
        if len(contributing_pos_minimals) > 0:
            break

        for x in pos_minimals:
            if is_contributing(vs, x, r, all_factors, len(vs) - d):
                contributing_pos_minimals.append(x)
                for i in range(d):
                    stratum = whitney_strat[i]
                    if stratum.subs(
                        {pure_H(wi): val for wi, val in zip(vs, x)}
                    ) == Ideal(pure_H.zero()):
                        raise ACSVException(
                            "Non-generic direction detected - critical point {w} is contained in {dim}-dimensional stratum".format(
                                w=str(x), dim=i
                            )
                        )

    if len(contributing_pos_minimals) == 0:
        raise ACSVException("No contributing points found.")
    if len(contributing_pos_minimals) > 1:
        raise ACSVException(
            f"More than one minimal contributing point with positive real coordinates found: {contributing_pos_minimals}"
        )
    minimal = contributing_pos_minimals[0]

    # Characterize all complex contributing points
    contributing_points = []
    for d in reversed(range(len(critical_point_ideals))):
        for w in critical_points_by_stratum[d]:
            if all(
                    abs(w_i) == abs(min_i) for w_i, min_i in zip(w, minimal)
            ) and is_contributing(vs, w, r, all_factors, len(vs) - d):
                contributing_points.append(w)

    return contributing_points


def contributing_points_hyperplane(G, H, vs, r=None, linear_form=None):
    r"""Compute contributing points of a multivariate
    rational function `F=G/H` admitting a finite number of critical points.
    Assumes that the singular variety of `F` is a union of transversely intersecting hyperplanes.

    Typically, this function is called as a subroutine of :func:`._diagonal_asymptotics_combinatorial_smooth`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with `F = G/H`.
    * ``vs`` -- List of variables of ``G`` and ``H``.
    * ``r`` -- (Optional) the direction, a vector or dictionary of positive algebraic numbers (usually integers).
        If a vector is given, assumes the variable order is given by ``(G/H).variables()``.
    * ``linear_form`` -- (Optional) A linear combination of the input
        variables that separates the critical point solutions.

    OUTPUT:

    List of minimal critical points of `F` in the direction `r`, as a list of tuples of algebraic numbers.
    List of tuples of non-minimal contributing points of `F` in the direction `r`, along with their height
    contribution and multiplicity.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import contributing_points_hyperplane
        sage: R.<x, y> = QQ[]
        sage: min_pts, other = contributing_points_hyperplane(
        ....:     1,
        ....:     (3-2*x-y)*(3-x-2*y),
        ....:     [x, y],
        ....: )
        sage: sorted(min_pts)
        [[1, 1]]
        sage: sorted(other)
        [([3/4, 3/2], 9/8, 1), ([3/2, 3/4], 9/8, 1)]
    """
    d = len(vs)
    if r is None:
        r = [ZZ.one() for _ in range(d)]

    Hs = [f for f, _ in H.factor()]
    if H.subs({v: 0 for v in H.variables()}) == 0:
        raise ValueError("Denominator vanishes at 0.")
    if any(f.degree() > 1 for f in Hs):
        raise ValueError("H does not define a hyperplane arrangement.")

    # Find critical points in Kronecker Representation
    cps = critical_points(
        G/H, r, linear_form
    )

    minimal_contributing_points = []
    next_contrib_vals = []

    # Sort all critical points by height
    cps_by_height = [(cp, prod([abs(vi)**ri for (vi, ri) in zip(cp, r)])) for cp in cps]
    cps_by_height.sort(key=lambda x: x[1])

    # Determine which critical points are contributing
    contributing_height = None
    for cp, h in cps_by_height:
        subs_dict = {vs[i]: cp[i] for i in range(d)}

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
                next_contrib_vals.append((cp, h, s))

    if not minimal_contributing_points:
        raise ACSVException("No contributing points found.")

    return minimal_contributing_points, next_contrib_vals


def contributing_points_combinatorial_smooth(G, H, variables, r=None, linear_form=None):
    r"""Compute contributing points of a multivariate
    rational function `F=G/H` admitting a finite number of critical points.
    Assumes that the singular variety of `F` is smooth and the function has a combinatorial expansion.

    Typically, this function is called as a subroutine of :func:`._diagonal_asymptotics_combinatorial_smooth`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with `F = G/H`.
    * ``variables`` -- List of variables of ``G`` and ``H``.
    * ``r`` -- (Optional) the direction, a vector or dictionary of positive algebraic numbers (usually integers).
      If a vector is given, assumes the variable order is given by ``(G/H).variables()``.
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions.

    OUTPUT:

    List of minimal critical points of `F` in the direction `r`,
    as a list of tuples of algebraic numbers.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import contributing_points_combinatorial_smooth
        sage: R.<x, y, w, lambda_, t, u_> = QQ[]
        sage: min_pts = contributing_points_combinatorial_smooth(
        ....:     1,
        ....:     1 - w*(y + x + x^2*y + x*y^2),
        ....:     [w, x, y],
        ....: )
        sage: sorted(min_pts)
        [[-1/4, -1, -1], [1/4, 1, 1]]
    """

    if isinstance(r, dict):
        r = _dict_to_variable_order(G/H, r)

    timer = Timer()
    (
        expanded_R,
        vs,
        (t, lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r)

    G, H = expanded_R(G), expanded_R(H)
    vsT = vs + list(r_variable_values.keys()) + [t, lambda_]

    # Create the critical point equations system
    vsH = H.variables()
    system = [
        H_var * H.derivative(H_var) - r_var * lambda_
        for H_var, r_var in zip(H.variables(), r)
    ]
    system.extend([H, H.subs({z: z * t for z in vsH})])
    system.extend(
        [
            direction_value.minpoly().subs(direction_var)
            for direction_var, direction_value in r_variable_values.items()
        ]
    )

    # Compute the Kronecker representation of our system
    timer.checkpoint()

    P, Qs = _kronecker_representation(system, u_, vsT, linear_form)
    timer.checkpoint("Kronecker")

    Qt = Qs[-2]  # Qs ordering is H.variables() + rvars + [t, lambda_]
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
    non_min = [[(q / Pd).subs(u_=u) for q in Qs[0:-2]] for u in rts_t_zo]

    # Filter the real roots for minimal points with positive coords
    pos_minimals = []
    for u in one_minus_t.roots(AA, multiplicities=False):
        is_min = True
        v = [(q / Pd).subs(u_=u) for q in Qs[: len(vs)]]
        rv = {
            ri: (q / Pd).subs(u_=u)
            for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
        }
        if any(rv[ri] != ri_value for ri, ri_value in r_variable_values.items()):
            continue
        if any(value <= 0 for value in v[: len(vs)]):
            continue
        for pt in non_min:
            if all(a == b for a, b in zip(v, pt)):
                is_min = False
                break
        if is_min:
            pos_minimals.append(u)

    # Remove non-smooth points and points with zero coordinates (where lambda=0)
    pos_minimals_new = []
    for pos_minimal in pos_minimals:
        x = (Qs[-1] / Pd).subs(u_=pos_minimal)
        if x == 0:
            acsv_logger.warning(
                f"Removing critical point {pos_minimal} because it either "
                "has a zero coordinate or is not smooth."
            )
        else:
            pos_minimals_new.append(pos_minimal)
    pos_minimals = pos_minimals_new

    # Verify necessary assumptions
    if not pos_minimals:
        raise ACSVException("No smooth minimal critical points found.")
    elif len(pos_minimals) > 1:
        raise ACSVException(
            f"More than one minimal point with positive real coordinates found: {pos_minimals}"
        )

    # Find all minimal critical points
    minCP = [(q / Pd).subs(u_=pos_minimals[0]) for q in Qs[0:-2]]
    minimals = []

    for u in one_minus_t.roots(QQbar, multiplicities=False):
        v = [(q / Pd).subs(u_=u) for q in Qs[: len(vs)]]
        rv = {
            ri: (q / Pd).subs(u_=u)
            for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
        }
        if any(rv[r_var] != r_value for r_var, r_value in r_variable_values.items()):
            continue
        if all(a.abs() == b.abs() for a, b in zip(minCP, v)):
            minimals.append(u)

    # Get minimal point coords, and make exact if possible
    minimal_coords = [[(q / Pd).subs(u_=u) for q in Qs[: len(vs)]] for u in minimals]
    [[a.exactify() for a in b] for b in minimal_coords]

    timer.checkpoint("Minimal Points")

    return [[(q / Pd).subs(u_=u) for q in Qs[: len(vs)]] for u in minimals]


def MinimalCriticalCombinatorial(F, r=None, linear_form=None, whitney_strat=None):
    acsv_logger.warning(
        "MinimalCriticalCombinatorial is deprecated and will be removed "
        "in a future release. Please use minimal_critical_points_combinatorial "
        "(same signature) instead.",
    )
    return minimal_critical_points_combinatorial(
        F, r=r, linear_form=linear_form, whitney_strat=whitney_strat
    )


def minimal_critical_points_combinatorial(
    F, r=None, linear_form=None, whitney_strat=None
):
    r"""Compute nonzero minimal critical points of a combinatorial multivariate
    rational function `F=G/H` admitting a finite number of critical points.

    The function is assumed to have a combinatorial expansion.

    INPUT:

    * ``F`` -- Symbolic fraction, the rational function of interest.
    * ``r`` -- (Optional) Length `d` vector or dictionary of positive integers.
      If a vector is given, assumes the variable order is given by ``F.variables()``.
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions
    * ``whitney_strat`` -- (Optional) If known / precomputed, a
      Whitney Stratification of `V(H)`. The program will not check if
      this stratification is correct. Should be a list of length ``d``, where
      the ``k``-th entry is a list of tuples of ideal generators representing
      a component of the ``k``-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import minimal_critical_points_combinatorial
        sage: var('x y')
        (x, y)
        sage: pts = minimal_critical_points_combinatorial(1/((1-(2*x+y)/3)*(1-(3*x+y)/4)))
        sage: sorted(pts)
        [[3/4, 3/2], [1, 1]]

    """

    if isinstance(r, dict):
        r = _dict_to_variable_order(F, r)

    _, H, variable_map = _prepare_symbolic_fraction(F)
    variables = list(variable_map.values())
    if whitney_strat is not None:
        whitney_strat = [
            [
                tuple(SR(gen).subs(variable_map) for gen in component)
                for component in stratum
            ]
            for stratum in whitney_strat
        ]
    if linear_form is not None:
        linear_form = SR(linear_form).subs(variable_map)
    (
        expanded_R,
        vs,
        (t, lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r)
    H = expanded_R(H)

    vsT = vs + list(r_variable_values) + [t, lambda_]

    # Compute the critical point system for each stratum
    pure_H = PolynomialRing(QQ, vs)

    if whitney_strat is None:
        whitney_strat = whitney_stratification(Ideal(pure_H(H)), pure_H)
    else:
        # Cast symbolic generators for provided stratification into the correct ring
        whitney_strat = [
            prod([Ideal([pure_H(f) for f in comp]) for comp in stratum])
            for stratum in whitney_strat
        ]

    critical_points = []
    pos_minimals = []
    for d, stratum in enumerate(whitney_strat):
        for P_comp in compute_primary_decomposition(stratum):
            c = len(vs) - d
            P_ext = P_comp.change_ring(expanded_R)
            M = matrix([[v * f.derivative(v) for v in vs] for f in P_ext.gens()] + [r])
            # Add in min polys for the direction variables
            r_polys = [
                direction_value.minpoly().subs(direction_var)
                for direction_var, direction_value in r_variable_values.items()
            ]
            # Create ideal of expanded_R containing extended critical point equations
            ideal = P_ext + Ideal(
                M.minors(c + 1)
                + [H.subs({v: v * t for v in vs}), (prod(vs) * lambda_ - 1)]
                + r_polys
            )
            # Saturate cpid by lower dimension stratum, if d > 0
            if d > 0:
                ideal = compute_saturation(
                    ideal, whitney_strat[d - 1].change_ring(expanded_R)
                )

            if ideal.dimension() < 0:
                continue
            P, Qs = _kronecker_representation(ideal.gens(), u_, vsT, linear_form)

            Qt = Qs[-2]  # Qs ordering is H.variables() + rvars + [t, lambda_]
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
            non_min = [[(q / Pd).subs(u_=u) for q in Qs[0:-2]] for u in rts_t_zo]

            # Filter the real roots for minimal points with positive coords
            for u in one_minus_t.roots(AA, multiplicities=False):
                is_min = True
                v = [(q / Pd).subs(u_=u) for q in Qs[: len(vs)]]
                rv = {
                    ri: (q / Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
                }
                if any(
                    rv[ri] != ri_value for ri, ri_value in r_variable_values.items()
                ):
                    continue
                if any(value <= 0 for value in v[: len(vs)]):
                    continue
                for pt in non_min:
                    if all(a == b for a, b in zip(v, pt)):
                        is_min = False
                        break
                if is_min:
                    pos_minimals.append(v)

            # Characterize all complex critical points in each stratum
            for u in one_minus_t.roots(QQbar, multiplicities=False):
                rv = {
                    ri: (q / Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
                }
                if any(
                    rv[r_var] != r_value for r_var, r_value in r_variable_values.items()
                ):
                    continue

                w = [QQbar((q / Pd).subs(u_=u)) for q in Qs[: len(vs)]]
                critical_points.append(w)

    if not pos_minimals:
        raise ACSVException("No critical points found.")

    # Characterize all complex contributing points
    minimal_criticals = (
        w for w in critical_points
        if any(all(abs(w_i) == abs(min_i) for w_i, min_i in zip(w, minimal))
               for minimal in pos_minimals)
    )

    return [[collapse_zero_part(w_i) for w_i in w] for w in minimal_criticals]


def critical_points(F, r=None, linear_form=None, whitney_strat=None):
    r"""Compute critical points of a multivariate
    rational function `F=G/H` admitting a finite number of critical points.

    Typically, this function is called as a subroutine of :func:`.diagonal_asymptotics_combinatorial`.

    INPUT:

    * ``F`` -- Symbolic fraction, the rational function of interest.
    * ``r`` -- (Optional) Length `d` vector or dictionary of positive integers.
      If a vector is given, assumes the variable order is given by ``F.variables()``.
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions
    * ``whitney_strat`` -- (Optional) If known / precomputed, a
      Whitney Stratification of `V(H)`. The program will not check if
      this stratification is correct. Should be a list of length ``d``, where
      the ``k``-th entry is a list of tuples of ideal generators representing
      a component of the ``k``-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`,

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import critical_points
        sage: var('x y')
        (x, y)
        sage: pts = critical_points(1/((1-(2*x+y)/3)*(1-(3*x+y)/4)))
        sage: sorted(pts)
        [[2/3, 2], [3/4, 3/2], [1, 1]]

    """
    if isinstance(r, dict):
        r = _dict_to_variable_order(F, r)

    _, H, variable_map = _prepare_symbolic_fraction(F)
    variables = list(variable_map.values())
    if whitney_strat is not None:
        whitney_strat = [
            [
                tuple(SR(gen).subs(variable_map) for gen in component)
                for component in stratum
            ]
            for stratum in whitney_strat
        ]
    if linear_form is not None:
        linear_form = SR(linear_form).subs(variable_map)
    (
        expanded_R,
        vs,
        (lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r, include_t=False)
    H = expanded_R(H)

    vsT = vs + list(r_variable_values.keys()) + [lambda_]

    # Compute the critical point system for each stratum
    pure_H = PolynomialRing(QQ, vs)

    if whitney_strat is None:
        whitney_strat = whitney_stratification(Ideal(pure_H(H)), pure_H)
    else:
        # Cast symbolic generators for provided stratification into the correct ring
        whitney_strat = [
            prod([Ideal([pure_H(f) for f in comp]) for comp in stratum])
            for stratum in whitney_strat
        ]

    critical_points = []
    for d, stratum in enumerate(whitney_strat):
        for P_comp in compute_primary_decomposition(stratum):
            c = len(vs) - d
            P_ext = P_comp.change_ring(expanded_R)
            M = matrix([[v * f.derivative(v) for v in vs] for f in P_ext.gens()] + [r])
            # Add in min polys for the direction variables
            r_polys = [
                direction_value.minpoly().subs(direction_var)
                for direction_var, direction_value in r_variable_values.items()
            ]
            # Create ideal of expanded_R containing extended critical point equations
            ideal = P_ext + Ideal(
                M.minors(c + 1) + [(prod(vs) * lambda_ - 1)] + r_polys
            )
            # Saturate cpid by lower dimension stratum, if d > 0
            if d > 0:
                ideal = compute_saturation(
                    ideal, whitney_strat[d - 1].change_ring(expanded_R)
                )

            if ideal.dimension() < 0:
                continue
            P, Qs = _kronecker_representation(ideal.gens(), u_, vsT, linear_form)

            Pd = P.derivative()

            # Characterize all complex critical points in each stratum
            for u in P.roots(QQbar, multiplicities=False):
                rv = {
                    ri: (q / Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs):-1])
                }
                if any(
                    rv[r_var] != r_value for r_var, r_value in r_variable_values.items()
                ):
                    continue

                w = [
                    collapse_zero_part(QQbar((q / Pd).subs(u_=u)))
                    for q in Qs[: len(vs)]
                ]
                critical_points.append(w)

    return critical_points
