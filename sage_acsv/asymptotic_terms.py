from copy import copy
from itertools import combinations

from sage.algebras.weyl_algebra import DifferentialWeylAlgebra
from sage.arith.srange import srange
from sage.functions.log import log, exp
from sage.functions.other import factorial
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.misc.prandom import shuffle
from sage.modules.free_module_element import vector
from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing
from sage.rings.imaginary_unit import I
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.qqbar import AA, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.constants import pi
from sage.symbolic.ring import SR

from sage_acsv.helpers import (
    ACSVException,
    Term,
    compute_newton_series,
    compute_newton_series_general,
    rational_function_reduce,
    compute_hessian,
    compute_implicit_hessian,
    compute_square_root_determinant_of_hessian,
    collapse_zero_part,
    transverse_leading_normalization,
)
from sage_acsv.debug import acsv_logger
from sage_acsv.settings import ACSVSettings, OutputFormat
from sage_acsv.utils import ( 
    _prepare_symbolic_fraction, 
    _dict_to_variable_order, 
)

def compute_asymptotics_at_points(
    F,
    contributing_points,
    r=None,
    expansion_precision=1,
    output_format=None
):
    r"""Compute asymptotic contribution of points of a multivariate rational function `F=G/H`
    admitting a finite number of critical points where the singular variety is the transverse union of smooth varieties.

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

    * ``as_symbolic`` -- (Optional) deprecated in favor of the equivalent
      ``output_format="symbolic"``. Will be removed in a future release.

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

    R = PolynomialRing(QQ, vs, len(vs))

    # Make sure G and H are coprime, and that H does not vanish at 0
    G, H = rational_function_reduce(G, H)
    G, H = R(G), R(H)
    return _compute_asymptotics_at_points(
        G, H,
        vs,
        r,
        contributing_points,
        expansion_precision,
        output_format
    )


def _compute_asymptotics_at_points(
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
    d = len(vs)

    asm_quantities = []
    # Store copy of vs and r in case order changes due to parametrization
    vs_copy, r_copy = copy(vs), copy(r)
    for cp in contributing_points:
        vs, r = copy(vs_copy), copy(r_copy)

        # Save extra H factors that the contrib point does not lie on
        # They can keep their multiplicities.
        extra_factors = []

        # Step 1: Determine if pt is a transverse multiple point of H,
        # and compute the factorization
        R = PolynomialRing(QQbar, len(vs), vs)
        G = R(SR(G))
        H = R(SR(H))
        vs = [R(SR(v)) for v in vs]
        subs_dict = {vs[i]: cp[i] for i in range(d)}
        poly_factors = H.factor()
        unit = poly_factors.unit()
        factors = []
        multiplicities = []
        for factor, multiplicity in poly_factors:
            const = factor.coefficients()[-1]
            unit *= const**multiplicity
            factor /= const
            if factor.subs(subs_dict) != 0:
                extra_factors.append(factor**multiplicity)
                continue
            factors.append(factor)
            multiplicities.append(multiplicity)
        s = len(factors)
        normals = matrix(
            [[f.derivative(v).subs(subs_dict) for v in vs] for f in factors]
        )
        if normals.rank() < s:
            raise ACSVException(
                "Not a transverse intersection. Cannot deal with this case."
            )

        # Step 2: Find the locally parametrizing coordinates of the point pt
        # Since we have d variables and s factors, there should be d-s of these
        # parametrizing coordinates
        # We will try to parametrize with the first d-s coordinates, shuffling
        # the vs and r if it doesn't work
        last_block = tuple(range(d-s, d))
        subsets = [last_block] + [
            c for c in combinations(range(d), s) if c != last_block
        ]
        for Rset in subsets:
            Jac = matrix(
                [
                    [(vs[j] * Q.derivative(vs[j])).subs(subs_dict) for j in Rset]
                    for Q in factors
                ]
            )
            if Jac.determinant() != 0:
                if Rset != last_block:
                    perm = [j for j in range(d) if j not in Rset] + list(Rset)
                    vs = tuple(vs[j] for j in perm)
                    r = tuple(r[j] for j in perm)
                    cp = tuple(cp[j] for j in perm)
                break

            acsv_logger.info("Variables do not parametrize, shuffling")
            vs_r_cp = list(zip(vs, r, cp))
            shuffle(vs_r_cp)  # shuffle mutates the list
            vs, r, cp = zip(*vs_r_cp)
        else:
            raise ACSVException("Cannot find parametrizing set.")

        # Step 3: Compute the gamma matrix as defined in 9.10
        Gamma = matrix(
            [[(v * Q.derivative(v)).subs(subs_dict) for v in vs] for Q in factors]
            + [
                [v.subs(subs_dict) if vs.index(v) == i else 0 for i in range(d)]
                for v in vs[: d - s]
            ]
        )

        # If critical point lies on a single smooth component, we can compute asymptotics
        # like in the smooth case
        if s == 1 and sum(multiplicities) == 1:
            n = SR.var("n")
            expansion = sum(
                term / (r[-1] * n) ** (term_order)
                for term_order, term in enumerate(
                    _general_term_asymptotics_smooth(G, H, r, vs, cp, expansion_precision)
                )
            )
            Hess = compute_hessian(H, vs, r, subs_dict)
            B = SR(1 / compute_square_root_determinant_of_hessian(Hess) / (r[-1] ** (d - 1) * ZZ(2) ** (d - 1)).sqrt())
        # We now support higher order expansions for non-smooth non-complete intersections
        elif all(p == 1 for p in multiplicities) and s != d:
            Qw = compute_implicit_hessian(factors, vs, r, subs=subs_dict)
            n = SR.var("n")
            expansion = sum(
                term / n ** term_order
                for term_order, term in enumerate(
                    _general_term_asymptotics(G, factors, extra_factors, r, vs, cp, expansion_precision)
                )
            ) / unit
            B = SR(
                1
                / compute_square_root_determinant_of_hessian(Qw)
                / (ZZ(2) ** (d - s)).sqrt()
            )
        # When we have a complete intersection of hyperplanes, we know the degree of the polynomial expansion.
        elif s == d and all(f.degree() == 1 for f in factors):
            n = SR.var("n")
            expansion = sum(
                term / n**term_order
                for term_order, term in enumerate(
                    _general_term_asymptotics_complete_intersection_hyplerplane(G, factors, multiplicities, r, vs, cp, expansion_precision)
                )
            ) / unit.subs(subs_dict)
            B = ZZ.one()
        # Higher order expansions not currently supported for higher-order poles
        # In complete intersection case, error bound is actually exponentially lower, but we don't currently have
        # a way to represent it using the asymptotic ring.
        else:
            if expansion_precision > 1:
                acsv_logger.warning(
                    "Higher order expansions are not supported for non-simple poles. Defaulting to expansion_precision 1."
                )
                expansion_precision = 1
            # For non-complete intersections, we must compute the parametrized Hessian matrix
            if s != d:
                Qw = compute_implicit_hessian(factors, vs, r, subs=subs_dict)
                expansion = SR(
                   G.subs(subs_dict)
                    / transverse_leading_normalization(factors, vs, r, cp)
                    / unit / R(prod(extra_factors)).subs(subs_dict)
                )
                B = SR(
                    1
                    / compute_square_root_determinant_of_hessian(Qw)
                    / (ZZ(2) ** (d - s)).sqrt()
                )
            else:
                expansion = SR(
                    G.subs(subs_dict)
                    / unit / abs(Gamma.determinant())
                    / R(prod(extra_factors)).subs(subs_dict)
                )
                B = ZZ.one()

            # Some constants appearing for higher order singularities
            mult_fac = prod([factorial(m - 1) for m in multiplicities])
            r_gamma_inv = prod(
                x ** (multiplicities[i] - 1)
                for i, x in enumerate(list(vector(r) * Gamma.inverse())[:s])
            )

            expansion *= (
                (-1) ** sum([m - 1 for m in multiplicities]) * r_gamma_inv / mult_fac
            )

        T = prod(SR(vs[i].subs(subs_dict)) ** r[i] for i in range(d))
        C = SR(1 / T)
        D = QQ((s - d) / 2 + sum(multiplicities) - s)
        try:
            B = QQbar(B)
            if B in QQ:
                B = QQ(B)
            C = QQbar(C)
        except (ValueError, TypeError):
            pass

        asm_quantities.append([expansion, B, C, D, s])

    asm_vals = [(c, d, b, a, s) for a, b, c, d, s in asm_quantities]

    if output_format is None:
        output_format = ACSVSettings.get_default_output_format()
    else:
        output_format = ACSVSettings.Output(output_format)

    if output_format in (ACSVSettings.Output.TUPLE, ACSVSettings.Output.SYMBOLIC):
        n = SR.var("n")
        result = [
            (base, n**exponent, (pi ** (s - d)).sqrt(), constant * expansion)
            for (base, exponent, constant, expansion, s) in asm_vals
        ]
        if output_format == ACSVSettings.Output.SYMBOLIC:
            result = sum([a**n * b * c * d for (a, b, c, d) in result])

    elif output_format == ACSVSettings.Output.TERMS:
        result = [
            Term(constant*expansion, (pi ** (s - d)).sqrt(), base, exponent) 
            for (base, exponent, constant, expansion, s) in asm_vals
            if constant*expansion != 0
        ]

    elif output_format == ACSVSettings.Output.ASYMPTOTIC:
        AR = AsymptoticRing("QQbar^n * n^QQ", QQbar)
        n = AR.gen()
        try:
            result = sum(
                [  # bug in AsymptoticRing requires splitting out modulus manually
                    constant
                    * (pi ** (s - d)).sqrt()
                    * abs(base) ** n
                    * collapse_zero_part(base / abs(base)) ** n
                    * n**exponent
                    * AR(expansion)
                    + (abs(base) ** n * n ** (exponent - expansion_precision)).O()
                    for (base, exponent, constant, expansion, s) in asm_vals
                ]
            )
        except ValueError:
            # Issue with Sage algebraic numbers equality checking
            for a, _, c, _, _ in asm_vals:
                a.simplify()
                c.simplify()
            result = sum(
                [  # bug in AsymptoticRing requires splitting out modulus manually
                    constant
                    * (pi ** (s - d)).sqrt()
                    * abs(base) ** n
                    * collapse_zero_part(base / abs(base)) ** n
                    * n**exponent
                    * AR(expansion)
                    + (abs(base) ** n * n ** (exponent - expansion_precision)).O()
                    for (base, exponent, constant, expansion, s) in asm_vals
                ]
            )

        # For complete intersections, the error bound is actually exponentially smaller after a certain precision
        # But we can currently only represent this for hyplerplane intersections
        if all(asm_val[-1] == d for asm_val in asm_vals) and all(f.degree() == 1 for f in factors) and expansion_precision > sum(multiplicities) - d:
            result = result.exact_part()
    else:
        raise NotImplementedError(f"Missing implementation for {output_format}")

    return result

def _compute_asymptotics_at_points_hyperplane(
    G, H,
    vs,
    r,
    contributing_points,
    next_contributing_vals,
    expansion_precision,
    output_format,
):
    r"""Compute contributing points of a combinatorial multivariate
    rational function `F=G/H` admitting a finite number of critical points where the singular variety is a smooth.

    Typically, this function is called as a subroutine of :func:`.diagonal_asymptotics_hyperplane`.

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
    d = len(vs)
    
    result = _compute_asymptotics_at_points(
        G, H, vs, r, contributing_points, expansion_precision, output_format
    )

    output_format = ACSVSettings.get_default_output_format() if output_format is None else ACSVSettings.Output(output_format)
    if output_format == OutputFormat.ASYMPTOTIC:
        n = result.parent().gen()
        for next_cp, next_height, s in next_contributing_vals:
            subs_dict = {vs[i]: next_cp[i] for i in range(d)}
            multiplicities = [p for f, p in H.factor() if f.subs(subs_dict) == 0]
            result = result + (((1 / abs(next_height)) ** n) * (n ** (QQ((-s - d) / 2 + sum(multiplicities))))).O()

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

    rd = r[-1]
    d = len(vs)

     # Find exponential growth
    T = prod([SR(vs[i]) ** r[i] for i in range(d)])

    # Find constants appearing in asymptotics in terms of original variables
    B = SR(1 / (rd ** (d - 1) * ZZ(2) ** (d - 1)).sqrt())
    C = SR(1 / T)

    # Compute constants at contributing singularities
    n = SR.var("n")
    asm_quantities = []
    for cp in contributing_points:
        subs_dict = {SR(v): V for (v, V) in zip(vs, cp)}
        expansion = sum(
            [
                term / (rd * n) ** (term_order)
                for term_order, term in enumerate(
                    _general_term_asymptotics_smooth(G, H, r, vs, cp, expansion_precision)
                )
            ]
        )
        # Find det(zH_z Hess) where Hess is the Hessian of z_1...z_n * log(g(z_1, ..., z_n))
        Hess = compute_hessian(H, vs, r, {v: V for (v, V) in zip(vs, cp)})
        B_sub = B.subs(subs_dict)/compute_square_root_determinant_of_hessian(Hess)
        C_sub = C.subs(subs_dict)
        try:
            B_sub = QQbar(B_sub)
            if B_sub in QQ:
                B_sub = QQ(B_sub)
            C_sub = QQbar(C_sub)
        except (ValueError, TypeError):
            pass

        asm_quantities.append([expansion, B_sub, C_sub])

    n = SR.var("n")
    asm_vals = [(c, QQ(1 - d) / 2, b, a) for (a, b, c) in asm_quantities]

    if output_format is None:
            output_format = ACSVSettings.get_default_output_format()
    else:
        output_format = ACSVSettings.Output(output_format)

    if output_format in (ACSVSettings.Output.TUPLE, ACSVSettings.Output.SYMBOLIC):
        n = SR.var("n")
        result = [
            (base, n**exponent, pi**exponent, constant * expansion)
            for (base, exponent, constant, expansion) in asm_vals
        ]
        if output_format == ACSVSettings.Output.SYMBOLIC:
            result = sum([a**n * b * c * d for (a, b, c, d) in result])

    elif output_format == ACSVSettings.Output.TERMS:
        result = [
            Term(constant*expansion, pi ** exponent, base, exponent) 
            for (base, exponent, constant, expansion) in asm_vals
            if constant*expansion != 0
        ]

    elif output_format == ACSVSettings.Output.ASYMPTOTIC:
        AR = AsymptoticRing("QQbar^n * n^QQ", QQbar)
        n = AR.gen()
        try:
            result = sum(
                [  # bug in AsymptoticRing requires splitting out modulus manually
                    constant
                    * pi**exponent
                    * abs(base) ** n
                    * collapse_zero_part(base / abs(base)) ** n
                    * n**exponent
                    * AR(expansion)
                    + (abs(base) ** n * n ** (exponent - expansion_precision)).O()
                    for (base, exponent, constant, expansion) in asm_vals
                ]
            )
        except ValueError:
            # Issue with Sage algebraic numbers equality checking
            for a, _, c, _ in asm_vals:
                a.simplify()
                c.simplify()
            result = sum(
                [  # bug in AsymptoticRing requires splitting out modulus manually
                    constant
                    * pi**exponent
                    * abs(base) ** n
                    * collapse_zero_part(base / abs(base)) ** n
                    * n**exponent
                    * AR(expansion)
                    + (abs(base) ** n * n ** (exponent - expansion_precision)).O()
                    for (base, exponent, constant, expansion) in asm_vals
                ]
            )

    else:
        raise NotImplementedError(f"Missing implementation for {output_format}")

    return result

def _general_term_asymptotics(G, Hs, Hs_ext, r, vs, cp, expansion_precision):
    r"""
    Compute coefficients of general (not necessarily leading) terms of
    the asymptotic expansion for a given critical
    point of a rational combinatorial multivariate rational function.

    Typically, this function is called as a subroutine of :func:`._compute_asymptotics_at_points`.

    INPUT:

    * ``G`` -- A polynomial in `vs`.
    * ``Hs`` -- A list of polynomials in `vs` such that `[G,Hs]` have no pairwise common factors.
    * ``Hs_ext`` -- A list of polynomials in `vs` such that that do not vanish at `cp`.
    * ``r`` -- The direction. A length `d` vector of positive algebraic numbers (usually
      integers).
    * ``vs`` -- Tuple of variables occurring in `G` and `Hs`.
    * ``cp`` -- A minimal critical point of `F` with coordinates specified in the
      same order as in ``vs``.
    * ``expansion_precision`` -- A positive integer value. This is the number of terms
      for which to compute coefficients in the asymptotic expansion.

    OUTPUT:

    List of coefficients of the asymptotic expansion.

    EXAMPLES::

        sage: from sage_acsv.asymptotic_terms import _general_term_asymptotics
        sage: R.<x, y, z> = QQ[]
        sage: _general_term_asymptotics(1, [1 - x - y], [], [1, 1], [x, y], [1/2, 1/2], 5)
        [2, -1/4, 1/64, 5/512, -21/16384]
        sage: _general_term_asymptotics(1, [1 - x - y], [1-y], [1, 1], [x, y], [1/2, 1/2], 5)
        [4, -5/2, 73/32, -575/256, 18459/8192]
        sage: _general_term_asymptotics(1, [1-x-2*y-z, 1-2*x-y-z], [], [1, 1, 1], [x, y, z], [2/9, 2/9, 1/3], 5)
        [27/2, -57/16, 1081/768, -106795/165888, 15808177/47775744]
        sage: _general_term_asymptotics(1, [1-x-2*y-z, 1-2*x-y-z], [1-x/2-y/2-z/2], [1, 1, 1], [x, y,z], [2/9, 2/9, 1/3], 3)
        [243/11, -70497/10648, 57642411/20614528]
    """

    d = len(vs)
    s = len(Hs)
    R = PolynomialRing(QQbar, vs)
    vs = R.gens()
    tvars = SR.var("t", d - s)
    G, Hs, Hs_ext = R(SR(G)), [R(SR(H)) for H in Hs], [R(SR(H)) for H in Hs_ext]
    subs_dict = {vs[i]: cp[i] for i in range(d)}

    # Step 3: Compute the gamma matrix as defined in 9.10
    Gamma = matrix(
        [[(v * Q.derivative(v)) for v in vs] for Q in Hs]
        + [
            [v if vs.index(v) == i else 0 for i in range(d)]
            for v in vs[: d - s]
        ]
    )

    leading_term = (
        _general_term_asymptotics_smooth(G, prod(Hs + Hs_ext), r, vs, cp, 1)[0]
        if s == 1 else
        SR(G.subs(subs_dict)
           / transverse_leading_normalization(Hs, vs, r, cp))
        / prod(Hs_ext).subs(subs_dict)
    )

    # Directly compute leading term
    if expansion_precision == 1:
        return [leading_term]

    # P and PsiTilde only need to be computed to order 2M
    N = 2 * expansion_precision + 1

    W = DifferentialWeylAlgebra(PolynomialRing(QQbar, tvars))
    TR = PowerSeriesRing(QQbar, tvars, default_prec=N)
    T = TR.gens()
    tvars = T
    D = list(W.differentials())

    # Function to apply differential operator dop on function f
    def eval_op(dop, f):
        if len(f.parent().gens()) == 1:
            return sum(
                prod([factorial(k) for k in E[0][1]]) * E[1] * f[E[0][1][0]]
                for E in dop
            )
        else:
            return sum(
                [prod([factorial(k) for k in E[0][1]]) * E[1] * f[E[0][1]] for E in dop]
            )

    Hess = compute_implicit_hessian(Hs, vs, r, subs_dict)
    Hessinv = Hess.inverse()
    v = matrix(W, [D[: d - s]])
    Epsilon = -(v * Hessinv.change_ring(W) * v.transpose())[0, 0]

    # Find series expansion of function g given implicitly by
    # H(w_1, ..., w_{d-s}, g_{d-s+1}, ..., g_{d}) = 0 up to needed order
    Hs_shift = [H.subs({v: v + subs_dict[v] for v in vs}) for H in Hs]
    gs = [g.subs({v: v-subs_dict[v] for v in vs}) + subs_dict[vs[d-s+i]] for i, g in enumerate(compute_newton_series_general(Hs_shift, vs, N))]

    # Polar change of coordinates
    tsubs = {v: subs_dict[v] * exp(I * t).add_bigoh(N) for v, t in zip(vs, tvars)}
    for i in range(s):
        tsubs[vs[d-s+i]] = gs[i].subs(tsubs)

    psi = sum([r[d-s+i]*log(g.subs(tsubs) / g.subs(subs_dict)) for i, g in enumerate(gs)]).add_bigoh(N)
    psi += I * sum([r[k] * tvars[k] for k in range(d - s)])
    v = matrix(TR, [tvars[k] for k in range(d - s)])
    psiTilde = psi - (v * Hess.change_ring(TR) * v.transpose())[0, 0] / 2
    PsiSeries = psiTilde.truncate(N)

    # Compute series expansion of P = G*z_1*...*z_{d-s}/Gamma up to needed order
    P_num = (G.subs(tsubs)*prod([tsubs[v] for v in vs[:d-s]])).add_bigoh(N)
    P_denom = (prod(Hs_ext).subs(tsubs)*Gamma.determinant().subs(tsubs)).add_bigoh(N)
    PSeries = (P_num / P_denom).truncate(N)

    if len(tvars) > 1:
        PsiSeries = PsiSeries.polynomial()
        PSeries = PSeries.polynomial()

    # Precompute products used for asymptotics
    EE = [Epsilon**k for k in range(3 * expansion_precision - 2)]
    PP = [PSeries]
    for k in range(1, 2 * expansion_precision - 1):
        PP.append(PP[k - 1] * PsiSeries)

    # Function to compute constants appearing in asymptotic expansion
    def constants_clj(ell, j):
        extra_contrib = (-ZZ.one()) ** j / (
            2 ** (ell + j) * factorial(ell) * factorial(ell + j)
        )
        return extra_contrib * eval_op(EE[ell + j], PP[ell])

    res = [
        sum([constants_clj(ell, j) for ell in srange(2 * j + 1)])
        for j in srange(expansion_precision)
    ]
    try:
        for i in range(len(res)):
            if res[i].imag() == 0:
                res[i] = AA(res[i])
    except (TypeError, ValueError, NotImplementedError):
        pass

    if res[0] == leading_term:
        return res
    if res[0] == -leading_term:
        return [-term for term in res]
    raise ACSVException("Issue with computing general terms - this should never happen.")


def _general_term_asymptotics_complete_intersection_hyplerplane(G, Hs, exps, r, vs, cp, expansion_precision):
    r"""
    Compute coefficients of general (not necessarily leading) terms of the asymptotic expansion for a given critical
    point of a rational combinatorial multivariate rational function lying on a complete intersection of hyperplanes.

    Typically, this function is called as a subroutine of :func:`._compute_asymptotics_at_points`.

    INPUT:

    * ``G`` -- A polynomial in `vs`.
    * ``Hs`` -- A list of polynomials in `vs` such that `[G,Hs]` have no pairwise common factors.
    * ``exps`` -- A list of integers representing the multiplicity of each of the polynomials in ``Hs``.
    * ``r`` -- The direction. A length `d` vector of positive algebraic numbers (usually
      integers).
    * ``vs`` -- Tuple of variables occurring in `G` and `Hs`.
    * ``cp`` -- A minimal critical point of `F` with coordinates specified in the
      same order as in ``vs``.
    * ``expansion_precision`` -- A positive integer value. This is the number of terms
      for which to compute coefficients in the asymptotic expansion.

    OUTPUT:

    List of coefficients of the asymptotic expansion.

    EXAMPLES::

        sage: from sage_acsv.asymptotic_terms import _general_term_asymptotics_complete_intersection_hyplerplane
        sage: R.<x, y> = QQ[]
        sage: _general_term_asymptotics_complete_intersection_hyplerplane(1, [3-2*x-y, 3-x-2*y], [2, 3], [1, 1], [x, y], [1, 1], 2)
        [1/162, 0]
        sage: _general_term_asymptotics_complete_intersection_hyplerplane(1, [3-2*x-y, 3-x-2*y], [2, 3], [1, 1], [x, y], [1, 1], 5)
        [1/162, 0, -7/162, -1/27]
    """
    M = matrix(
        [
            [
                QQ(H.derivative(v)) for v in vs
            ] for H in Hs
        ]
    )

    N = sum(exps) + expansion_precision + 1

    # Perform the change of variables z = cp + M^(-1)*t
    subs_dict = {vs[i]: cp[i] for i in range(len(vs))}
    tvars = SR.var("t", len(vs))
    Rt = PowerSeriesRing(QQbar, list(tvars) + [SR.var("n")], default_prec=N)
    tvars = Rt.gens()[:-1]
    n = Rt.gens()[-1]

    # Compute a power series expansion for G(z)/z^{nr} up to needed order
    # Take exp(log) for efficiency
    tsubs = {v: subs_dict[v] - (M.inverse() * vector(tvars))[i] for i, v in enumerate(vs)}
    GSeries = G.subs(tsubs) / prod([tsubs[v] for v in vs]).add_bigoh(N)
    log_series = sum([
        r[i]*n*log(tsubs[v]/cp[i]) for i, v in enumerate(vs)
    ]).add_bigoh(N)
    Pseries = GSeries / exp(log_series).truncate(N)

    # Differentiate given power series to the necessary order
    # Then, substitute t=0 to get polynomial part
    for i in range(len(exps)):
        Pseries = Pseries.derivative(tvars[i], exps[i]-1) / factorial(exps[i]-1)

    R = PolynomialRing(QQbar, 1, [n])
    n = R.gens()[0]
    Pseries = R(Pseries.subs({t: 0 for t in tvars}).polynomial())

    # helper function since getting coefficients is unreliable for sage versions <= 10.4
    def get_coefficient(P, n, deg):
        if deg == 0:
            return P.subs({n:0})
        return P.coefficient(n**deg)

    # Return the coefficients of the resulting power series in n
    return [(-1)**(sum(exps) + len(vs))*get_coefficient(Pseries, n, k)/M.determinant().abs() for k in range(sum(exps)-len(vs)+1)][::-1][:expansion_precision]


def _general_term_asymptotics_smooth(G, H, r, vs, cp, expansion_precision):
    r"""
    Compute coefficients of general (not necessarily leading) terms of
    the asymptotic expansion for a given critical
    point of a rational combinatorial multivariate rational function lying on a smooth point of `V(H)`.

    Typically, this function is called as a subroutine of :func:`._compute_asymptotics_at_points`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with `F = G/H`.
    * ``r`` -- The direction. A length `d` vector of positive algebraic numbers (usually
      integers).
    * ``vs`` -- Tuple of variables occurring in `G` and `H`.
    * ``cp`` -- A minimal critical point of `F` with coordinates specified in the
      same order as in ``vs``.
    * ``expansion_precision`` -- A positive integer value. This is the number of terms
      for which to compute coefficients in the asymptotic expansion.

    OUTPUT:

    List of coefficients of the asymptotic expansion.

    EXAMPLES::

        sage: from sage_acsv.asymptotic_terms import _general_term_asymptotics_smooth
        sage: R.<x, y, z> = QQ[]
        sage: _general_term_asymptotics_smooth(1, 1 - x - y, [1, 1], [x, y], [1/2, 1/2], 5)
        [2, -1/4, 1/64, 5/512, -21/16384]
        sage: _general_term_asymptotics_smooth(1, 1 - x - y - z, [1, 1, 1], [x, y, z], [1/3, 1/3, 1/3], 4)
        [3, -2/3, 2/27, 14/729]

        sage: R.<x, y> = QQ[]
        sage: _general_term_asymptotics_smooth(1, 1 - x - y, [1, 1], [x, y], [1/2, 1/2], 11)
        [2, -1/4, 1/64, 5/512, -21/16384, -399/131072, 869/2097152, 39325/16777216, -334477/1073741824, -28717403/8589934592, 59697183/137438953472]
    """

    if expansion_precision == 1:
        A = SR(-G / vs[-1] / H.derivative(vs[-1]))
        subs_dict = {SR(v): V for (v, V) in zip(vs, cp)}
        return [A.subs(subs_dict)]

    # Convert everything to field of algebraic numbers
    d = len(vs)
    R = PolynomialRing(QQbar, vs)
    vs = R.gens()
    vd = vs[-1]
    tvars = SR.var("t", d - 1)
    G, H = R(SR(G)), R(SR(H))

    cp = {v: V for (v, V) in zip(vs, cp)}

    # P and PsiTilde only need to be computed to order 2M
    N = 2 * expansion_precision + 1

    W = DifferentialWeylAlgebra(PolynomialRing(QQbar, tvars))
    TR = PowerSeriesRing(QQbar, tvars, default_prec=N)
    T = TR.gens()
    tvars = T
    D = list(W.differentials())

    # Function to apply differential operator dop on function f
    def eval_op(dop, f):
        if len(f.parent().gens()) == 1:
            return sum(
                prod([factorial(k) for k in E[0][1]]) * E[1] * f[E[0][1][0]]
                for E in dop
            )
        else:
            return sum(
                [prod([factorial(k) for k in E[0][1]]) * E[1] * f[E[0][1]] for E in dop]
            )

    Hess = compute_hessian(H, vs, r, cp)
    Hessinv = Hess.inverse()
    v = matrix(W, [D[: d - 1]])
    Epsilon = -(v * Hessinv.change_ring(W) * v.transpose())[0, 0]

    # Find series expansion of function g given implicitly by
    # H(w_1, ..., w_{d-1}, g(w_1, ..., w_{d-1})) = 0 up to needed order
    g = compute_newton_series(H.subs({v: v + cp[v]for v in vs}), vs, N)
    g = g.subs({v: v - cp[v] for v in vs}) + cp[vd]

    # Polar change of coordinates
    tsubs = {v: cp[v] * exp(I * t).add_bigoh(N) for v, t in zip(vs, tvars)}
    tsubs[vd] = g.subs(tsubs)

    # Compute PsiTilde up to needed order
    psi = log(g.subs(tsubs) / g.subs(cp)).add_bigoh(N)
    psi += I * sum([r[k] * tvars[k] for k in range(d - 1)]) / r[-1]
    v = matrix(TR, [tvars[k] for k in range(d - 1)])
    psiTilde = psi - (v * Hess * v.transpose())[0, 0] / 2
    PsiSeries = psiTilde.truncate(N)

    # Compute series expansion of P = -G/(g*H_{z_d}) up to needed order
    P_num = -G.subs(tsubs).add_bigoh(N)
    P_denom = (g * H.derivative(vd)).subs(tsubs).add_bigoh(N)
    PSeries = (P_num / P_denom).truncate(N)

    if len(tvars) > 1:
        PsiSeries = PsiSeries.polynomial()
        PSeries = PSeries.polynomial()

    # Precompute products used for asymptotics
    EE = [Epsilon**k for k in range(3 * expansion_precision - 2)]
    PP = [PSeries]
    for k in range(1, 2 * expansion_precision - 1):
        PP.append(PP[k - 1] * PsiSeries)

    # Function to compute constants appearing in asymptotic expansion
    def constants_clj(ell, j):
        extra_contrib = (-ZZ.one()) ** j / (
            2 ** (ell + j) * factorial(ell) * factorial(ell + j)
        )
        return extra_contrib * eval_op(EE[ell + j], PP[ell])

    res = [
        sum([constants_clj(ell, j) for ell in srange(2 * j + 1)])
        for j in srange(expansion_precision)
    ]
    try:
        for i in range(len(res)):
            if res[i].imag() == 0:
                res[i] = AA(res[i])
    except (TypeError, ValueError, NotImplementedError):
        pass

    return res


