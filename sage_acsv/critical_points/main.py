from sage.arith.misc import gcd
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.rings.ideal import Ideal
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import AA, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.ring import SR

from sage_acsv.critical_points.data import CriticalPointProblem
from sage_acsv.critical_points.filter import _filter_contributing_points, _filter_contributing_points_by_height, _filter_minimal_points, _filter_positive_real_minimal_contributing_point
from sage_acsv.critical_points.solver import solve_and_filter_positive_minimal_solutions, solve_kronecker_system
from sage_acsv.critical_points.systems import build_smooth_critical_point_system, build_stratified_critical_point_systems
from sage_acsv.kronecker.kronecker_representation import _kronecker_representation
from sage_acsv.helpers import (
    ACSVException,
    is_contributing,
)
from sage_acsv.debug import Timer, acsv_logger
from sage_acsv.whitney import whitney_stratification
from sage_acsv.groebner.functions import compute_primary_decomposition, compute_saturation
from sage_acsv.helpers.utils import (
    _prepare_expanded_polynomial_ring, 
    _prepare_symbolic_fraction, 
    _dict_to_variable_order,
    _process_linear_form_input,
    _process_whitney_stratification_input, 
    _subs,
    _collapse_zero_part,
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
    whitney_strat = _process_whitney_stratification_input(whitney_strat, variable_map)
    linear_form = _process_linear_form_input(linear_form, variable_map)

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
    timer = Timer()

    (
        expanded_R,
        vs,
        (t, lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r)
    H = expanded_R(H)
    vsT = vs + list(r_variable_values.keys()) + [lambda_, t]

    # Add in min polys for the direction variables
    r_polys = [
        direction_value.minpoly().subs(direction_var)
        for direction_var, direction_value in r_variable_values.items()
    ]

    problem = CriticalPointProblem(
        require_contributing=True,
        require_minimal=True,
        H_polynomial=H,
        variables=vs,
        direction=r,
        lambda_variable=lambda_,
        t_variable=t,
        r_polynomials=r_polys,
        r_variable_values=r_variable_values
    )

    critical_point_systems = build_stratified_critical_point_systems(
        problem, whitney_strat
    )

    # Final minimal critical points with positive coordinates on each stratum
    critical_points_by_stratum = {}
    pos_minimals_by_stratum = {}
    for d, systems in reversed(critical_point_systems.items()):
        critical_points_by_stratum[d] = []
        pos_minimals_by_stratum[d] = []

        for system in systems:
            if Ideal(system.generators).dimension() < 0:
                continue

            P, Qs = _kronecker_representation(system.generators, u_, vsT, linear_form)
            Qt = Qs[-1]
            Qs = Qs[:-1]

            P, Qs, pos_minimals = solve_and_filter_positive_minimal_solutions(
                vs, r_variable_values, (P, Qs), Qt
            )

            pos_minimals_by_stratum[d].extend(
                pos_minimals
            )

            critical_points_by_stratum[d].extend(
                solve_kronecker_system(
                    vs, r_variable_values, (P, Qs)
                )
            )

    timer.checkpoint("Critical Points")

    # Refine positive minimal critical points to those that are contributing
    r = [QQbar(r_variable_values.get(ri, ri)) for ri in r]
    minimal = _filter_positive_real_minimal_contributing_point(
        pos_minimals_by_stratum, H, vs, r
    )

    # Discard non-minimal points before testing contribution. Testing contributing
    # on non-minimal points can raise exceptions.
    minimal_points_by_stratum = {
        dimension: _filter_minimal_points(points, [minimal])
        for dimension, points in critical_points_by_stratum.items()
    }
    minimal_contributing_points = _filter_contributing_points(
        minimal_points_by_stratum, H, vs, r
    )

    timer.checkpoint("Contributing Points")

    return minimal_contributing_points


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
    List of non-minimal contributing points of `F` in the direction `r` as a list of tuples of algebraic numbers.
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
        [[3/4, 3/2], [3/2, 3/4]]
    """
    timer = Timer()

    d = len(vs)
    if r is None:
        r = [ZZ.one() for _ in range(d)]

    Hs = [f for f, _ in H.factor()]
    if H.subs({v: 0 for v in H.variables()}) == 0:
        raise ValueError("Denominator vanishes at 0.")
    if any(f.degree() > 1 for f in Hs):
        raise ValueError("H does not define a hyperplane arrangement.")

    # Find critical points in Kronecker Representation
    timer.checkpoint()
    cps = critical_points(
        G/H, r, linear_form
    )
    timer.checkpoint("Critical Points")

    minimal_contributing_points, next_contributing_points = _filter_contributing_points_by_height(
        cps,
        H,
        vs,
        r
    )

    if not minimal_contributing_points:
        raise ACSVException("No contributing points found.")

    timer.checkpoint("Contributing Points")

    return minimal_contributing_points, next_contributing_points


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
    vsT = vs + list(r_variable_values.keys()) + [lambda_, t]

    r_polys = [
        direction_value.minpoly().subs(direction_var)
        for direction_var, direction_value in r_variable_values.items()
    ]

    # Create the critical point equations system
    system = build_smooth_critical_point_system(
        CriticalPointProblem(
            require_minimal=True,
            require_contributing=True,
            H_polynomial=H,
            variables=vs,
            lambda_variable=lambda_,
            direction=r,
            t_variable=t,
            r_polynomials=r_polys,
            r_variable_values=r_variable_values
        )
    )

    # Compute the Kronecker representation of our system
    timer.checkpoint()
    P, Qs = _kronecker_representation(system, u_, vsT, linear_form)
    Qt = Qs[-1]
    Qs = Qs[:-1]
    timer.checkpoint("Kronecker")

    P, Qs, pos_minimals = solve_and_filter_positive_minimal_solutions(
        vs, r_variable_values, (P, Qs), Qt
    )

    # Verify necessary assumptions
    if not pos_minimals:
        raise ACSVException("No smooth minimal critical points found.")
    elif len(pos_minimals) > 1:
        raise ACSVException(
            f"More than one minimal point with positive real coordinates found: {pos_minimals}"
        )

    critical_points = solve_kronecker_system(
        vs, r_variable_values, (P, Qs)
    )

    minimal_points = _filter_minimal_points(
        critical_points, pos_minimals
    )

    timer.checkpoint("Minimal Points")

    return minimal_points


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
    whitney_strat = _process_whitney_stratification_input(whitney_strat, variable_map)
    linear_form = _process_linear_form_input(linear_form, variable_map)
    (
        expanded_R,
        vs,
        (t, lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r)
    H = expanded_R(H)
    vsT = vs + list(r_variable_values.keys()) + [lambda_, t]

    # Add in min polys for the direction variables
    r_polys = [
        direction_value.minpoly().subs(direction_var)
        for direction_var, direction_value in r_variable_values.items()
    ]

    problem = CriticalPointProblem(
        require_contributing=False,
        require_minimal=True,
        H_polynomial=H,
        variables=vs,
        direction=r,
        lambda_variable=lambda_,
        t_variable=t,
        r_polynomials=r_polys,
        r_variable_values=r_variable_values
    )

    critical_point_systems = build_stratified_critical_point_systems(
        problem, whitney_strat
    )

    critical_points = []
    pos_minimals = []
    for _, systems in critical_point_systems.items():
        for system in systems:
            if Ideal(system.generators).dimension() < 0:
                continue

            P, Qs = _kronecker_representation(system.generators, u_, vsT, linear_form)
            Qt = Qs[-1]
            Qs = Qs[:-1]

            P, Qs, new_pos_minimals = solve_and_filter_positive_minimal_solutions(
                vs, r_variable_values, (P, Qs), Qt
            )

            pos_minimals.extend(new_pos_minimals)

            critical_points.extend(
                solve_kronecker_system(
                    vs, r_variable_values, (P, Qs)
                )
            )

    if not pos_minimals:
        raise ACSVException("No critical points found.")

    # Characterize all complex contributing points
    minimal_criticals = _filter_minimal_points(
        critical_points, pos_minimals
    )

    return [[_collapse_zero_part(w_i) for w_i in w] for w in minimal_criticals]


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
    whitney_strat = _process_whitney_stratification_input(whitney_strat, variable_map)
    linear_form = _process_linear_form_input(linear_form, variable_map)
    (
        expanded_R,
        vs,
        (lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r, include_t=False)
    H = expanded_R(H)

    vsT = vs + list(r_variable_values.keys()) + [lambda_]

    # Add in min polys for the direction variables
    r_polys = [
        direction_value.minpoly().subs(direction_var)
        for direction_var, direction_value in r_variable_values.items()
    ]

    problem = CriticalPointProblem(
        require_contributing=False,
        require_minimal=False,
        H_polynomial=H,
        variables=vs,
        direction=r,
        lambda_variable=lambda_,
        r_polynomials=r_polys,
        r_variable_values=r_variable_values
    )

    critical_point_systems = build_stratified_critical_point_systems(
        problem, whitney_strat
    )

    # Final minimal critical points with positive coordinates on each stratum
    critical_points = []
    for _, systems in reversed(critical_point_systems.items()):
        for system in systems:
            if Ideal(system.generators).dimension() < 0:
                continue

            P, Qs = _kronecker_representation(system.generators, u_, vsT, linear_form)

            critical_points.extend(
                solve_kronecker_system(
                    vs, r_variable_values, (P, Qs)
                )
            )

    return critical_points
