from copy import copy
from enum import Enum

from sage.misc.misc_c import prod
from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import AA, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.constants import pi
from sage.symbolic.ring import SR

from .data import ClassifiedPoint, LocalPointStrategy
from .point_contribution import (
    _compute_contribution_of_complete_hyperplane_point,
    _compute_contribution_of_high_order_transverse_point,
    _compute_contribution_of_simple_transverse_point,
    _compute_contribution_of_smooth_point,
)
from sage_acsv.data.terms import Term
from sage_acsv.debug import Timer, acsv_logger
from sage_acsv.helpers.factorization import (
    _compute_transverse_factorization_at_point,
    _find_parametrizing_order,
)
from sage_acsv.helpers.utils import (
    _collapse_zero_part,
    _dict_to_variable_order,
    _prepare_symbolic_fraction,
    _rational_function_reduce,
)
from sage_acsv.settings import ACSVSettings, OutputFormat

class PointsStrategy(Enum):
    """Options for computing asymptotic contributions of points
    based on local geometry. 

    """

    TRANSVERSE = "transverse"
    HYPERPLANE = "hyperplane"
    SMOOTH = "smooth"

def compute_asymptotics_at_points(
    F,
    contributing_points,
    r=None,
    expansion_precision=1,
    output_format=None,
    *,
    next_contributing_points=[],
    strategy=None,
):
    r"""Compute asymptotic contribution of points of a multivariate rational function `F=G/H`
    admitting a finite number of critical points. Currently, this function function only supports
    the case where the singular variety is the transverse union of smooth varieties, with
    specialized strategies for hyperplane intersections and smooth varieties.

    This function assumes ``contributing_points`` are minimal contributing points for `F` in
    the direction `r` provided, and will not verify this for the user.

    INPUT:

    * ``F`` -- The rational function `G/H` in `d` variables.
    * ``contributing_points`` -- A list of ``d``-tuples of algebraic numbers
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``expansion_precision`` -- (Optional) A positive integer value. This is the number
      of terms to compute in the asymptotic expansion. Defaults to 1, which
      only computes the leading term.
    * ``output_format`` -- (Optional) A string or :class:`.ACSVSettings.Output`
      specifying the way the asymptotic growth is returned. Allowed values
      currently are:

      - ``"tuple"``: the growth is returned as a list of
        tuples of the form ``(a, n^b, pi^c, d)`` such that the `r`-diagonal of `F`
        is the sum of ``a^n n^b pi^c d + O(a^n n^{b-1})`` over these tuples.
      - ``"symbolic"``: the growth is returned as an expression from the symbolic
        ring ``SR`` in the variable ``n``.
      - ``"asymptotic"``: the growth is returned as an expression from an appropriate
        ``AsymptoticRing`` in the variable ``n``.
      - ``None``: the default, which uses the default set for
        :class:`.ACSVSettings.Output` itself via
        :meth:`.ACSVSettings.set_default_output_format`. The default behavior
        is asymptotic output.
    * ``next_contributing_points`` -- (Optional) A list of ``d``-tuples of algebraic numbers
    * ``strategy`` -- A string or :class:`.PointsStrategy` value specifying the computation
      stratgy to use. Allowed values currently are

      - ``"transverse"``: The contributing points lie on one or more transversely intersecting
        hypersurfaces (default)
      - ``"hyperplane"``: The contributing points lie on one or more transversely intersecting 
        hyperplanes
      - ``"smooth"``: The contributing points lie on a smooth surface

    OUTPUT:

    A representation of the asymptotic contributions from the ``contributing_points``
    of the coefficient array of `F` along the specified direction.

    EXAMPLES::

        sage: from sage_acsv import compute_asymptotics_at_points
        sage: var('x, y')
        (x, y)
        sage: compute_asymptotics_at_points(1/(1-x-y), [(1/2, 1/2)])
        1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))

    Note that we do not check if the input points are actually contributing::

        sage: compute_asymptotics_at_points(1/(1-x-y), [(2/3, 1/3)])
        3/2/sqrt(pi)*(9/2)^n*n^(-1/2) + O((9/2)^n*n^(-3/2))

    An example with complex coordinates for the contributing point.
    
        sage: R.<x1, y1, x2, y2, x3, y3> = QQ[]
        sage: blk = lambda u, v: (1 - u - v + 3*u*v)^2 + (u*v)^2
        sage: F = 1/(blk(x1, y1)*blk(x2, y2)*blk(x3, y3))
        sage: T.<t> = QQ[]
        sage: rts = (10*t^4 - 12*t^3 + 10*t^2 - 4*t + 1).roots(QQbar, multiplicities=False)
        sage: p = sorted(sorted(rts, key=lambda rt: abs(rt))[:2], key=lambda rt: CDF(rt).imag())[0]
        sage: res1 = compute_asymptotics_at_points(1/blk(x1, y1), [(p, p)],  # long time
        ....:                                      r=[1, 1], output_format="tuple")
        sage: res3 = compute_asymptotics_at_points(F, [(p, p, p, p, p, p)],  # long time
        ....:                                      r=[1]*6, output_format="tuple")
        sage: c1, c3 = QQbar(res1[0][3]), QQbar(res3[0][3])  # long time
        sage: c3  # long time
        3.048436400357616? + 2.604341598696821?*I
        sage: c3 == c1^3  # long time
        True

    """
    if isinstance(r, dict):
        r = _dict_to_variable_order(F, r)

    contributing_points = [
        _dict_to_variable_order(F, cp, None) if isinstance(cp, dict) else cp for cp in contributing_points
    ]

    G, H, variable_map = _prepare_symbolic_fraction(F)
    vs = list(variable_map.values())

    if r is None:
        n = len(H.variables())
        r = [1 for _ in range(n)]

    try:
        r = [QQ(ri) for ri in r]
    except (ValueError, TypeError):
        r = [AA(ri) for ri in r]

    R, vs = PolynomialRing(QQ, vs, len(vs)).objgens()

    # Make sure G and H are coprime, and that H does not vanish at 0
    G, H = _rational_function_reduce(G, H)
    G, H = R(G), R(H)

    return _compute_asymptotic_contributions_at_points(
        G, H,
        vs,
        r,
        contributing_points,
        next_contributing_points,
        expansion_precision,
        output_format,
        strategy
    )


def _compute_asymptotic_contributions_at_points(
    G, H,
    vs,
    r,
    minimal_contributing_points,
    next_contributing_points,
    expansion_precision,
    output_format,
    strategy=None
):
    r"""Internal function for computing asymptotic contribution of points of a multivariate rational
    F = `G/H` admitting a finite number of critical points. Currently, this function function only
    supports the case where the singular variety is the transverse union of smooth varieties, with
    specialized strategies for hyperplane intersections and smooth varieties.

    INPUT:

    * ``G, H`` -- Polynomials `d` variables.
    * ``vs`` -- a list of `d` variables
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``minimal_contributing_points`` -- A list of ``d``-tuples of algebraic numbers
    * ``next_contributing_points`` -- A list of ``d``-tuples of algebraic numbers
    * ``expansion_precision`` -- (Optional) A positive integer value. This is the number
      of terms to compute in the asymptotic expansion. Defaults to 1, which
      only computes the leading term.
    * ``output_format`` -- (Optional) A string or :class:`.ACSVSettings.Output`
      specifying the way the asymptotic growth is returned. Allowed values
      currently are:

      - ``"tuple"``: the growth is returned as a list of
        tuples of the form ``(a, n^b, pi^c, d)`` such that the `r`-diagonal of `F`
        is the sum of ``a^n n^b pi^c d + O(a^n n^{b-1})`` over these tuples.
      - ``"symbolic"``: the growth is returned as an expression from the symbolic
        ring ``SR`` in the variable ``n``.
      - ``"asymptotic"``: the growth is returned as an expression from an appropriate
        ``AsymptoticRing`` in the variable ``n``.
      - ``None``: the default, which uses the default set for
        :class:`.ACSVSettings.Output` itself via
        :meth:`.ACSVSettings.set_default_output_format`. The default behavior
        is asymptotic output.
    * ``strategy`` -- A string or :class:`.PointsStrategy` value specifying the computation
      stratgy to use. Allowed values currently are
      - ``"transverse"``: The contributing points lie on one or more transversely intersecting
        hypersurfaces (default)
      - ``"hyperplane"``: The contributing points lie on one or more transversely intersecting 
        hyperplanes
      - ``"smooth"``: The contributing points lie on a smooth surface

    OUTPUT:

    A representation of the asymptotic contributions from the ``contributing_points``
    of the coefficient array of `F` along the specified direction.
    """
    strategy = (
        PointsStrategy.TRANSVERSE
        if strategy is None
        else PointsStrategy(strategy)
    )

    match strategy:
        case PointsStrategy.SMOOTH:
            return _compute_asymptotics_at_points_smooth(
                G, H,
                vs,
                r,
                minimal_contributing_points,
                expansion_precision,
                output_format
            )
        case PointsStrategy.HYPERPLANE:
            return _compute_asymptotics_at_points_hyperplane(
                G, H,
                vs,
                r,
                minimal_contributing_points,
                next_contributing_points,
                expansion_precision,
                output_format
            )
        case _:
            return _compute_asymptotics_at_points_transverse(
                G, H,
                vs,
                r,
                minimal_contributing_points,
                expansion_precision,
                output_format
            )

def _compute_asymptotics_at_points_transverse(
    G, H,
    vs,
    r,
    contributing_points,
    expansion_precision,
    output_format,
):
    r"""Compute contributing points of a combinatorial multivariate
    rational function `F=G/H` admitting a finite number of critical points where the singular variety is the transverse union of smooth varieties.

    Typically, this function is called as a subroutine of :func:`.diagonal_asymptotics_combinatorial`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with ``F = G/H``
    * ``vs`` -- List of variables of ``G`` and ``H``
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``contributing_points`` -- A list of ``d``-tuples of algebraic numbers
    * ``expansion_precision`` -- (Optional) A positive integer value. This is the number
      of terms to compute in the asymptotic expansion. Defaults to 1, which
      only computes the leading term.
    * ``output_format`` -- (Optional) A string or :class:`.ACSVSettings.Output`
      specifying the way the asymptotic growth is returned. Allowed values
      currently are:
      - ``"tuple"``: the growth is returned as a list of
        tuples of the form ``(a, n^b, pi^c, d)`` such that the `r`-diagonal of `F`
        is the sum of ``a^n n^b pi^c d + O(a^n n^{b-1})`` over these tuples.
      - ``"symbolic"``: the growth is returned as an expression from the symbolic
        ring ``SR`` in the variable ``n``.
      - ``"asymptotic"``: the growth is returned as an expression from an appropriate
        ``AsymptoticRing`` in the variable ``n``.
      - ``None``: the default, which uses the default set for
        :class:`.ACSVSettings.Output` itself via
        :meth:`.ACSVSettings.set_default_output_format`. The default behavior
        is asymptotic output.
    * ``as_symbolic`` -- (Optional) deprecated in favor of the equivalent
      ``output_format="symbolic"``. Will be removed in a future release.

    OUTPUT:

    A representation of the asymptotic contributions from the ``contributing_points``
    of the coefficient array of `F` along the specified direction.
    """
    timer = Timer()
    timer.checkpoint()
    d = len(vs)

    strip_error = False
    cps_data = []
    vs_copy, r_copy = copy(vs), copy(r)
    for cp in contributing_points:
        vs, r = copy(vs_copy), copy(r_copy)

        # Save extra H factors that the contrib point does not lie on
        # They can keep their multiplicities.
        extra_factors = []

        # Determine if pt is a transverse multiple point of H and compute the factorization
        R = PolynomialRing(QQbar, len(vs), vs)
        G = R(SR(G))
        H = R(SR(H))
        vs = [R(SR(v)) for v in vs]
        subs_dict = {vs[i]: cp[i] for i in range(d)}

        # Compute factorization around the critical point
        unit, factors, multiplicities, extra_factors  = _compute_transverse_factorization_at_point(
            H, vs, subs_dict
        )
        s = len(factors)

        # Reorder so first d-s coordinates parametrize the remaining s
        vs, r, cp = _find_parametrizing_order(factors, vs, r, cp)

        local_strategy = None
        if s == 1 and sum(multiplicities) == 1:
            local_strategy = LocalPointStrategy.SMOOTH
        elif all(p == 1 for p in multiplicities) and s != d:
            local_strategy = LocalPointStrategy.TRANSVERSE_SIMPLE
        elif s == d and all(f.degree() == 1 for f in factors):
            local_strategy = LocalPointStrategy.HYPERPLANE_COMPLETE
        else:
            local_strategy = LocalPointStrategy.TRANSVERSE_HIGHER_ORDER

        cps_data.append(
            ClassifiedPoint(
                strategy=local_strategy,
                codimension=s,
                direction=r,
                variables=vs,
                point=cp,
                substitutions=subs_dict,
                unit=unit,
                factors=factors,
                multiplicities=multiplicities,
                extra_factors=extra_factors
            )
        )
    
    if expansion_precision > 1 and any(
        data.strategy == LocalPointStrategy.TRANSVERSE_HIGHER_ORDER for data in cps_data
    ):
        acsv_logger.warning(
            "Higher order expansions are not supported for non-simple poles. Defaulting to expansion_precision 1."
        )
        expansion_precision = 1

    if all(
        data.strategy == LocalPointStrategy.HYPERPLANE_COMPLETE and
        expansion_precision > sum(data.multiplicities) - d
        for data in cps_data
    ):
        strip_error = True

    terms = []
    n = SR.var("n")
    for data in cps_data:
        s, r, vs, cp, subs_dict = (
            data.codimension, data.direction, data.variables, data.point, data.substitutions
        )
        unit, factors, multiplicities, extra_factors = (
            data.unit, data.factors, data.multiplicities, data.extra_factors
        )

        # If critical point lies on a single smooth component, we can compute asymptotics
        # like in the smooth case
        if data.strategy == LocalPointStrategy.SMOOTH:
            constant, expansion = _compute_contribution_of_smooth_point(
                G, H, data, expansion_precision
            )
        # We now support higher order expansions for non-smooth non-complete intersections
        elif data.strategy == LocalPointStrategy.TRANSVERSE_SIMPLE:
            constant, expansion = _compute_contribution_of_simple_transverse_point(
                G, data, expansion_precision
            )
        # When we have a complete intersection of hyperplanes, we know the degree of the polynomial expansion.
        elif data.strategy == LocalPointStrategy.HYPERPLANE_COMPLETE:
            constant, expansion = _compute_contribution_of_complete_hyperplane_point(
                G, data, expansion_precision
            )
        else:
            constant, expansion = _compute_contribution_of_high_order_transverse_point(
                G, data, expansion_precision
            )

        T = prod(SR(vs[i].subs(subs_dict)) ** r[i] for i in range(d))
        base = SR(1 / T)
        exponent = QQ((s - d) / 2 + sum(multiplicities) - s)
        try:
            constant = QQbar(constant)
            if constant in QQ:
                constant = QQ(constant)
            base = QQbar(base)
        except (ValueError, TypeError):
            pass

        if constant * expansion != 0:
            terms.append(
                Term(constant*expansion, (pi ** (s - d)).sqrt(), base, exponent) 
            )

    timer.checkpoint("Final Asymptotics")

    return _format_output(
        terms, 
        output_format, 
        expansion_precision, 
        strip_error
    )

def _compute_asymptotics_at_points_hyperplane(
    G, H,
    vs,
    r,
    minimal_contributing_points,
    next_contributing_points,
    expansion_precision,
    output_format,
):
    r"""Compute contributing points of a combinatorial multivariate
    rational function `F=G/H` admitting a finite number of critical points where the singular variety is a
    hyperplane arrangement.

    Typically, this function is called as a subroutine of :func:`.diagonal_asymptotics_hyperplane`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with ``F = G/H``
    * ``vs`` -- List of variables of ``G`` and ``H``
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``contributing_points`` -- A list of ``d``-tuples of algebraic numbers
    * ``next_contributing_points`` -- A list of ``d``-tuples of algebraic numbers
    * ``expansion_precision`` -- (Optional) A positive integer value. This is the number
        of terms to compute in the asymptotic expansion. Defaults to 1, which
        only computes the leading term.
    * ``output_format`` -- (Optional) A string or :class:`.ACSVSettings.Output`
        specifying the way the asymptotic growth is returned. Allowed values
        currently are:
        - ``"tuple"``: the growth is returned as a list of
        tuples of the form ``(a, n^b, pi^c, d)`` such that the `r`-diagonal of `F`
        is the sum of ``a^n n^b pi^c d + O(a^n n^{b-1})`` over these tuples.
        - ``"symbolic"``: the growth is returned as an expression from the symbolic
        ring ``SR`` in the variable ``n``.
        - ``"asymptotic"``: the growth is returned as an expression from an appropriate
        ``AsymptoticRing`` in the variable ``n``.
        - ``None``: the default, which uses the default set for
        :class:`.ACSVSettings.Output` itself via
        :meth:`.ACSVSettings.set_default_output_format`. The default behavior
        is asymptotic output.
    * ``as_symbolic`` -- (Optional) deprecated in favor of the equivalent
        ``output_format="symbolic"``. Will be removed in a future release.

    OUTPUT:

    A representation of the asymptotic contributions from the ``contributing_points``
    of the coefficient array of `F` along the specified direction.
    """
    timer = Timer()
    timer.checkpoint()
    d = len(vs)
    
    result = _compute_asymptotics_at_points_transverse(
        G, H, vs, r, minimal_contributing_points, expansion_precision, output_format
    )

    output_format = ACSVSettings.get_default_output_format() if output_format is None else ACSVSettings.Output(output_format)
    if output_format == OutputFormat.ASYMPTOTIC:
        n = result.parent().gen()
        for next_cp in next_contributing_points:
            subs_dict = {vs[i]: next_cp[i] for i in range(d)}
            height = prod([abs(vi)**ri for (vi, ri) in zip(next_cp, r)])
            s = sum([f.subs(subs_dict) == 0 for f, _ in H.factor()])
            multiplicities = [p for f, p in H.factor() if f.subs(subs_dict) == 0]
            result = result + (((1 / abs(height)) ** n) * (n ** (QQ((-s - d) / 2 + sum(multiplicities))))).O()

    timer.checkpoint("Final Asymptotics")

    return result

def _compute_asymptotics_at_points_smooth(
    G, H,
    vs,
    r,
    contributing_points,
    expansion_precision,
    output_format,
):
    r"""Compute contributing points of a combinatorial multivariate
    rational function `F=G/H` admitting a finite number of critical points where the singular variety is a smooth.

    Typically, this function is called as a subroutine of :func:`.diagonal_asymptotics_combinatorial_smooth`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with ``F = G/H``
    * ``vs`` -- List of variables of ``G`` and ``H``
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``contributing_points`` -- A list of ``d``-tuples of algebraic numbers
    * ``expansion_precision`` -- (Optional) A positive integer value. This is the number
      of terms to compute in the asymptotic expansion. Defaults to 1, which
      only computes the leading term.
    * ``output_format`` -- (Optional) A string or :class:`.ACSVSettings.Output`
      specifying the way the asymptotic growth is returned. Allowed values
      currently are:
      - ``"tuple"``: the growth is returned as a list of
        tuples of the form ``(a, n^b, pi^c, d)`` such that the `r`-diagonal of `F`
        is the sum of ``a^n n^b pi^c d + O(a^n n^{b-1})`` over these tuples.
      - ``"symbolic"``: the growth is returned as an expression from the symbolic
        ring ``SR`` in the variable ``n``.
      - ``"asymptotic"``: the growth is returned as an expression from an appropriate
        ``AsymptoticRing`` in the variable ``n``.
      - ``None``: the default, which uses the default set for
        :class:`.ACSVSettings.Output` itself via
        :meth:`.ACSVSettings.set_default_output_format`. The default behavior
        is asymptotic output.
    * ``as_symbolic`` -- (Optional) deprecated in favor of the equivalent
      ``output_format="symbolic"``. Will be removed in a future release.

    OUTPUT:

    A representation of the asymptotic contributions from the ``contributing_points``
    of the coefficient array of `F` along the specified direction.
    """

    timer = Timer()
    timer.checkpoint()

    rd = r[-1]
    d = len(vs)

     # Find exponential growth
    T = prod([SR(vs[i]) ** r[i] for i in range(d)])
    # Find constants appearing in asymptotics in terms of original variables
    C = SR(1 / T)

    # Compute constants at contributing singularities
    n = SR.var("n")
    terms = []
    for cp in contributing_points:
        subs_dict = {v: value for v, value in zip(vs, cp)}
        constant, expansion = _compute_contribution_of_smooth_point(
            G,
            H,
            ClassifiedPoint(
                strategy=LocalPointStrategy.SMOOTH,
                codimension=1,
                direction=r,
                variables=vs,
                point=cp,
                substitutions=subs_dict,
                unit=1,
                factors=[H],
                multiplicities=[1],
                extra_factors=[],
            ),
            expansion_precision,
        )
        base = C.subs(subs_dict)
        try:
            constant = QQbar(constant)
            if constant in QQ:
                constant = QQ(constant)
            base = QQbar(base)
        except (ValueError, TypeError):
            pass

        if constant * expansion != 0:
            terms.append(
                Term(constant*expansion, (pi ** (ZZ.one() - d)).sqrt(), base, QQ(1-d)/2) 
            )

    timer.checkpoint("Final Asymptotics")

    return _format_output(terms, output_format, expansion_precision)


def _format_output(terms, output_format, expansion_precision, strip_error=False):
    r"""
    Formats an asymptotic ``Term`` to the specified ``output_format``.

    INPUT:
    
        * ``terms`` -- List of ``Term``.
        * ``outlut_format`` -- A string or :class:`.ACSVSettings.Output`.
        * ``expansion_precision`` -- A positive integer value.
        * ``strip_error`` -- A boolean. Indicates if return value should have an error bound, in the case
        the output is in asymptotic form.
    """
    if output_format is None:
        output_format = ACSVSettings.get_default_output_format()
    else:
        output_format = ACSVSettings.Output(output_format)

    if output_format in (ACSVSettings.Output.TUPLE, ACSVSettings.Output.SYMBOLIC):
        n = SR.var("n")
        result = [
            (term.base, n**term.power, term.pi_factor, term.coefficient)
            for term in terms
        ]
        if output_format == ACSVSettings.Output.SYMBOLIC:
            result = sum([a**n * b * c * d for (a, b, c, d) in result])

    elif output_format == ACSVSettings.Output.TERMS:
        result = terms

    elif output_format == ACSVSettings.Output.ASYMPTOTIC:
        AR = AsymptoticRing("QQbar^n * n^QQ", QQbar)
        n = AR.gen()
        try:
            result = sum(
                [  # bug in AsymptoticRing requires splitting out modulus manually
                    AR(term.coefficient)
                    * term.pi_factor
                    * abs(term.base) ** n
                    * _collapse_zero_part(term.base / abs(term.base)) ** n
                    * n**term.power
                    + (abs(term.base) ** n * n ** (term.power - expansion_precision)).O()
                    for term in terms
                ], start = AR.zero()
            )
        except ValueError:
            # Issue with Sage algebraic numbers equality checking
            for term in terms:
                term.base.simplify()
                term.coefficient.simplify()
            result = sum(
                [  # bug in AsymptoticRing requires splitting out modulus manually
                    term.coefficient
                    * term.pi_factor
                    * abs(term.base) ** n
                    * _collapse_zero_part(term.base / abs(term.base)) ** n
                    * n**term.power
                    + (abs(term.base) ** n * n ** (term.power - expansion_precision)).O()
                    for term in terms
                ], start = AR.zero()
            )
        if strip_error:
            result = result.exact_part()
    else:
        raise NotImplementedError(f"Missing implementation for {output_format}")

    return result
