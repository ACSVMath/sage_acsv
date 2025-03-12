"""Functions for determining asymptotics of the coefficients
of multivariate rational functions.
"""
from copy import copy

from sage.all import AA, PolynomialRing, QQ, QQbar, SR, DifferentialWeylAlgebra, Ideal, Polyhedron
from sage.all import gcd, prod, pi, matrix, exp, log, I, factorial, srange, shuffle, vector

from sage_acsv.kronecker import _kronecker_representation, _msolve_kronecker_representation
from sage_acsv.helpers import ACSVException, IsContributing, NewtonSeries, RationalFunctionReduce, OutputFormat, GetHessian, ImplicitHessian
from sage_acsv.debug import Timer, acsv_logger
from sage_acsv.whitney import WhitneyStrat
from sage_acsv.macaulay2 import PrimaryDecomposition, Saturate


MAX_MIN_CRIT_RETRIES = 3

# we need to monkeypatch a function from the asymptotics module such that creating
# asymptotic expansions over QQbar is possible. this should be removed once the
# upstream issue is resolved.

import sage.rings.asymptotic.misc as asy_misc

from sage.rings.integer_ring import ZZ

strip_symbolic_original = asy_misc.strip_symbolic
def strip_symbolic(expression):
    expression = strip_symbolic_original(expression)
    if expression in ZZ:
        expression = ZZ(expression)
    return expression

asy_misc.strip_symbolic = strip_symbolic



def diagonal_asy_smooth(
    F,
    r=None,
    linear_form=None,
    expansion_precision=1,
    return_points=False,
    output_format=None,
    as_symbolic=False
):
    r"""Asymptotics in a given direction `r` of the multivariate rational function `F`.
        Assumes the singular variety of F is smooth.

    INPUT:

    * ``F`` -- The rational function ``G/H`` in ``d`` variables. This function is
      assumed to have a combinatorial expansion.
    * ``r`` -- A vector of length d of positive algebraic numbers (generally integers).
      Defaults to the appropriate vector of all 1's if not specified.
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions.
    * ``expansion_precision`` -- A positive integer value. This is the number of terms to
      compute in the asymptotic expansion. Defaults to 1, which only computes the leading
      term.
    * ``return_points`` -- If ``True``, also returns the coordinates of
      minimal critical points. By default ``False``.
    * ``output_format`` -- (Optional) A string or :class:`.OutputFormat` specifying
      the way the asymptotic growth is returned. Allowed values currently are:
      - ``"tuple"`` or ``None``, the default: the growth is returned as a list of
        tuples of the form ``(a, n^b, pi^c, d)`` such that the `r`-diagonal of `F`
        is the sum of ``a^n n^b pi^c d + O(a^n n^{b-1})`` over these tuples.
      - ``"symbolic"``: the growth is returned as an expression from the symbolic
        ring ``SR`` in the variable ``n``.
      - ``"asymptotic"``: the growth is returned as an expression from an appropriate
        ``AsymptoticRing`` in the variable ``n``.
    * ``as_symbolic`` -- deprecated in favor of the equivalent
      ``output_format="symbolic"``. Will be removed in a future release.

    OUTPUT:

    A representation of the asymptotic main term, either as a list of tuples,
    or as a symbolic expression.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming `F` has a finite number of critical
    points) the code can be rerun until a separating form is found.

    Examples::

        sage: from sage_acsv import diagonal_asy_smooth
        sage: var('x,y,z,w')
        (x, y, z, w)
        sage: diagonal_asy_smooth(1/(1-x-y))
        1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))
        sage: diagonal_asy_smooth(1/(1-(1+x)*y), r = [1,2], return_points=True)
        (1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2)), [[1, 1/2]])
        sage: diagonal_asy_smooth(1/(1-(x+y+z)+(3/4)*x*y*z), output_format="symbolic")
        0.840484893481498?*24.68093482214177?^n/(pi*n)
        sage: diagonal_asy_smooth(1/(1-(x+y+z)+(3/4)*x*y*z))
        0.840484893481498?/pi*24.68093482214177?^n*n^(-1) + O(24.68093482214177?^n*n^(-2))
        sage: var('n')
        n
        sage: asy = diagonal_asy_smooth(
        ....:     1/(1 - w*(1 + x)*(1 + y)*(1 + z)*(x*y*z + y*z + y + z + 1)),
        ....:     output_format="tuple",
        ....: )
        sage: sum([
        ....:      a.radical_expression()^n * b * c * QQbar(d).radical_expression()
        ....:      for (a, b, c, d) in asy
        ....: ])
        1/4*(12*sqrt(2) + 17)^n*sqrt(17/2*sqrt(2) + 12)/(pi^(3/2)*n^(3/2))

    Not specifying any ``output_format`` falls back to the default tuple
    representation::

        sage: from sage_acsv import diagonal_asy_smooth, OutputFormat
        sage: var('x')
        x
        sage: diagonal_asy_smooth(1/(1 - 2*x))
        2^n + O(2^n*n^(-1))
        sage: diagonal_asy_smooth(1/(1 - 2*x), output_format="tuple")
        [(2, 1, 1, 1)]

    Passing ``"symbolic"`` lets the function return an element of the
    symbolic ring in the variable ``n`` that describes the asymptotic growth::

        sage: growth = diagonal_asy_smooth(1/(1 - 2*x), output_format="symbolic"); growth
        2^n
        sage: growth.parent()
        Symbolic Ring

    The argument ``"asymptotic"`` constructs an asymptotic expansion over
    an appropriate ``AsymptoticRing`` in the variable ``n``, including the
    appropriate error term::

        sage: assume(SR.an_element() > 0)  # required to make coercions involving SR work properly
        sage: growth = diagonal_asy_smooth(1/(1 - x - y), output_format="asymptotic"); growth
        1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))
        sage: growth.parent()
        Asymptotic Ring <(Algebraic Real Field)^n * n^QQ * (Arg_(Algebraic Field))^n> over Symbolic Ring

    Increasing the precision of the expansion returns an expansion with more terms
    (works for all available output formats)::

        sage: diagonal_asy_smooth(1/(1 - x - y), expansion_precision=3, output_format="asymptotic")
        1/sqrt(pi)*4^n*n^(-1/2) - 1/8/sqrt(pi)*4^n*n^(-3/2) + 1/128/sqrt(pi)*4^n*n^(-5/2)
        + O(4^n*n^(-7/2))

    The direction of the diagonal, `r`, defaults to the standard diagonal (i.e., the
    vector of all 1's) if not specified. It also supports passing non-integer values,
    notably rational numbers::

        sage: diagonal_asy_smooth(1/(1 - x - y), r=(1, 17/42), output_format="symbolic")
        1.317305628032865?*2.324541507270374?^n/(sqrt(pi)*sqrt(n))
    
    and even algebraic numbers (note, however, that the performance for complicated
    algebraic numbers is significantly degraded)::

        sage: diagonal_asy_smooth(1/(1 - x - y), r=(sqrt(2), 1))
        0.9238795325112868?/sqrt(pi)*(2.414213562373095?/0.5857864376269049?^1.414213562373095?)^n*n^(-1/2) + O((2.414213562373095?/0.5857864376269049?^1.414213562373095?)^n*n^(-3/2))

    ::

        sage: diagonal_asy_smooth(1/(1 - x - y*x^2), r=(1, 1/2 - 1/2*sqrt(1/5)), output_format="asymptotic")
        1.710862642974252?/sqrt(pi)*1.618033988749895?^n*n^(-1/2)
        + O(1.618033988749895?^n*n^(-3/2))

    The function times individual steps of the algorithm, timings can
    be displayed by increasing the printed verbosity level of our debug logger::

        sage: import logging
        sage: from sage_acsv.debug import acsv_logger
        sage: acsv_logger.setLevel(logging.INFO)
        sage: diagonal_asy_smooth(1/(1 - x - y))
        INFO:sage_acsv:... Executed Kronecker in ... seconds.
        INFO:sage_acsv:... Executed Minimal Points in ... seconds.
        INFO:sage_acsv:... Executed Final Asymptotics in ... seconds.
        1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))
        sage: acsv_logger.setLevel(logging.WARNING)


    Tests:

    Check that passing a non-supported ``output_format`` errors out::

        sage: diagonal_asy_smooth(1/(1 - x - y), output_format='hello world')
        Traceback (most recent call last):
        ...
        ValueError: 'hello world' is not a valid OutputFormat
        sage: diagonal_asy_smooth(1/(1 - x - y), output_format=42)
        Traceback (most recent call last):
        ...
        ValueError: 42 is not a valid OutputFormat

    """
    G, H = F.numerator(), F.denominator()
    if r is None:
        n = len(H.variables())
        r = [1 for _ in range(n)]

    try:
        r = [QQ(ri) for ri in r]
    except (ValueError, TypeError):
        r = [AA(ri) for ri in r]

    # Initialize variables
    vs = list(H.variables())

    t, lambda_, u_ = PolynomialRing(QQ, 't, lambda_, u_').gens()
    expanded_R = PolynomialRing(QQ, len(vs)+3, vs + [t, lambda_, u_])

    vs = [expanded_R(v) for v in vs]
    t, lambda_, u_ = expanded_R(t), expanded_R(lambda_), expanded_R(u_)
    vsT = vs + [t, lambda_]

    all_variables = (vs, lambda_, t, u_)
    d = len(vs)
    rd = r[-1]
    vd = vs[-1]

    # Make sure G and H are coprime, and that H does not vanish at 0
    G, H = RationalFunctionReduce(G, H)
    G, H = expanded_R(G), expanded_R(H)
    if H.subs({v: 0 for v in H.variables()}) == 0:
        raise ValueError("Denominator vanishes at 0.")

    # In case form doesn't separate, we want to try again
    for _ in range(MAX_MIN_CRIT_RETRIES):
        try:
            # Find minimal critical points in Kronecker Representation
            min_crit_pts = ContributingCombinatorialSmooth(
                G, H, all_variables,
                r=r,
                linear_form=linear_form
            )
            break
        except Exception as e:
            if isinstance(e, ACSVException) and e.retry:
                acsv_logger.warning(
                    "Randomly generated linear form was not suitable, "
                    f"encountered error: {e}\nRetrying..."
                )
                continue
            else:
                raise e
    else:
        return

    timer = Timer()

    # Find det(zH_z Hess) where Hess is the Hessian of z_1...z_n * log(g(z_1, ..., z_n))
    Det = GetHessian(H, vsT[0:-2], r).determinant()

    # Find exponential growth
    T = prod([SR(vs[i])**r[i] for i in range(d)])


    # Find constants appearing in asymptotics in terms of original variables
    B = SR(1 / Det / rd**(d-1) / 2**(d-1))
    C = SR(1 / T)

    # Compute constants at contributing singularities
    n = SR.var('n')
    asm_quantities = []
    for cp in min_crit_pts:
        subs_dict = {SR(v): V for (v, V) in zip(vs, cp)}
        expansion = sum([
            term / (rd * n)**(term_order)
            for term_order, term in enumerate(
                GeneralTermAsymptotics(G, H, r, vs, cp, expansion_precision)
            )
        ])
        B_sub = B.subs(subs_dict)
        C_sub = C.subs(subs_dict)
        try:
            B_sub = QQbar(B_sub)
            C_sub = QQbar(C_sub)
        except (ValueError, TypeError):
            pass
        asm_quantities.append([expansion, B_sub, C_sub])

    n = SR.var('n')
    asm_vals = [(c, QQ(1 - d)/2, b.sqrt(), a) for (a, b, c) in asm_quantities]
    timer.checkpoint("Final Asymptotics")

    if as_symbolic:
        from warnings import warn

        warn(
            "The as_symbolic argument has been deprecated in favor of output_format='symbolic' "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if output_format is None:
            output_format = OutputFormat.SYMBOLIC

    if output_format is None:
        output_format = OutputFormat.ASYMPTOTIC
    else:
        output_format = OutputFormat(output_format)

    if output_format in (OutputFormat.TUPLE, OutputFormat.SYMBOLIC):
        n = SR.var('n')
        result = [
            (base, n**exponent, pi**exponent, constant * expansion)
            for (base, exponent, constant, expansion) in asm_vals
        ]
        if output_format == OutputFormat.SYMBOLIC:
            result = sum([a**n * b * c * d for (a, b, c, d) in result])

    elif output_format == OutputFormat.ASYMPTOTIC:
        from sage.all import AsymptoticRing
        AR = AsymptoticRing('QQbar^n * n^QQ', QQbar)
        n = AR.gen()
        result = sum([
            constant * pi**exponent * base**n * n**exponent * AR(expansion) 
            + (abs(base)**n * n**(exponent - expansion_precision)).O()
            for (base, exponent, constant, expansion) in asm_vals
        ])

    else:
        raise NotImplementedError(f"Missing implementation for {output_format}")
    
    if return_points:
        return result, min_crit_pts

    return result

def diagonal_asy(
    F,
    r=None,
    linear_form=None,
    expansion_precision=1,
    return_points=False,
    output_format=None,
    whitney_strat=None,
    as_symbolic=False
):
    r"""Asymptotics in a given direction `r` of the multivariate rational function `F`.

    INPUT:

    * ``F`` -- The rational function ``G/H`` in ``d`` variables. This function is
      assumed to have a combinatorial expansion.
    * ``r`` -- A vector of length d of positive algebraic numbers (generally integers).
      Defaults to the appropriate vector of all 1's if not specified.
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions.
    * ``expansion_precision`` -- A positive integer value. This is the number of terms to
      compute in the asymptotic expansion. Defaults to 1, which only computes the leading
      term.
    * ``return_points`` -- If ``True``, also returns the coordinates of
      minimal critical points. By default ``False``.
    * ``output_format`` -- (Optional) A string or :class:`.OutputFormat` specifying
      the way the asymptotic growth is returned. Allowed values currently are:
      - ``"tuple"`` or ``None``, the default: the growth is returned as a list of
        tuples of the form ``(a, n^b, pi^c, d)`` such that the `r`-diagonal of `F`
        is the sum of ``a^n n^b pi^c d + O(a^n n^{b-1})`` over these tuples.
      - ``"symbolic"``: the growth is returned as an expression from the symbolic
        ring ``SR`` in the variable ``n``.
      - ``"asymptotic"``: the growth is returned as an expression from an appropriate
        ``AsymptoticRing`` in the variable ``n``.
    * ``as_symbolic`` -- deprecated in favor of the equivalent
      ``output_format="symbolic"``. Will be removed in a future release.
    * ``whitney_strat`` -- (Optional) The user can pass in a Whitney Stratification of V(H)
        to save computation time. The program will not check if this stratification is correct.
        The whitney_strat should be an array of length ``d``, with the ``k``-th entry a list of
        tuples of ideal generators representing a component of the ``k``-dimensional stratum.

    OUTPUT:

    A representation of the asymptotic main term, either as a list of tuples,
    or as a symbolic expression.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming `F` has a finite number of critical
    points) the code can be rerun until a separating form is found.

    Examples::

        sage: from sage_acsv import diagonal_asy
        sage: var('x,y')
        (x, y)
        sage: diagonal_asy(1/((1-(2*x+y)/3)*(1-(3*x+y)/4)), r = [17/24, 7/24], output_format = 'asymptotic')
        12 + O(n^(-1))

        sage: from sage_acsv import diagonal_asy
        sage: var('x,y,z')
        (x, y, z)
        sage: G = (1+x)*(1-x*y^2+x^2)
        sage: H = (1-z*(1+x^2+x*y^2))*(1-y)*(1+x^2)
        sage: strat = [
        ....:     [(1-z*(1+x^2+x*y^2), 1-y, 1+x^2)],
        ....:     [(1-z*(1+x^2+x*y^2), 1-y),(1-z*(1+x^2+x*y^2), 1+x^2),(1-y,1+x^2)],
        ....:     [(H,)],
        ....: ]
        sage: diagonal_asy(G/H, r = [1,1,1], output_format = 'asymptotic', whitney_strat = strat)
        0.866025403784439?/sqrt(pi)*3^n*n^(-1/2) + O(3^n*n^(-3/2))

    """
    G, H = F.numerator(), F.denominator()
    # Initialize variables
    vs = list(H.variables())
    R = PolynomialRing(QQ, vs, len(vs))
    H_sing = Ideal([R(H)] + [R(H.derivative(v)) for v in vs])
    if H_sing.dimension() < 0:
        return diagonal_asy_smooth(
            F,
            r = r,
            linear_form = linear_form,
            expansion_precision=expansion_precision,
            return_points = return_points,
            output_format = output_format,
            as_symbolic=as_symbolic
        )

    if r is None:
        n = len(H.variables())
        r = [1 for _ in range(n)]

    try:
        r = [QQ(ri) for ri in r]
    except (ValueError, TypeError):
        r = [AA(ri) for ri in r]

    t, lambda_, u_ = PolynomialRing(QQ, 't, lambda_, u_').gens()
    expanded_R = PolynomialRing(QQ, len(vs)+3, vs + [t, lambda_, u_])

    vs = [expanded_R(v) for v in vs]
    t, lambda_, u_ = expanded_R(t), expanded_R(lambda_), expanded_R(u_)

    all_variables = (vs, lambda_, t, u_)
    d = len(vs)

    # Make sure G and H are coprime, and that H does not vanish at 0
    G, H = RationalFunctionReduce(G, H)
    G, H = expanded_R(G), expanded_R(H)
    if H.subs({v: 0 for v in H.variables()}) == 0:
        raise ValueError("Denominator vanishes at 0.")

    H_sf = prod([f for f,_ in H.factor()])
    # In case form doesn't separate, we want to try again
    for _ in range(MAX_MIN_CRIT_RETRIES):
        try:
            # Find minimal critical points in Kronecker Representation
            min_crit_pts = ContributingCombinatorial(
                G, H_sf, all_variables,
                r=r,
                linear_form=linear_form,
                whitney_strat=whitney_strat
            )
            break
        except Exception as e:
            if isinstance(e, ACSVException) and e.retry:
                acsv_logger.warning(
                    "Randomly generated linear form was not suitable, "
                    f"encountered error: {e}\nRetrying..."
                )
                continue
            else:
                raise e
    else:
        return

    timer = Timer()

    asm_quantities = []
    for cp in min_crit_pts:
        # Step 1: Determine if pt is a transverse multiple point of H, and compute the factorization
        # for now, we'll just try to factor it in the polynomial ring
        R = PolynomialRing(QQbar, len(vs), vs)
        G = R(SR(G))
        H = R(SR(H))
        vs = [R(SR(v)) for v in vs]
        subs_dict = {vs[i]:cp[i] for i in range(d)}
        poly_factors = H.factor()
        unit = poly_factors.unit()
        factors = []
        multiplicities = []
        for factor, multiplicity in poly_factors:
            const = factor.coefficients()[-1]
            unit *= const ** multiplicity
            factor /= const
            if factor.subs(subs_dict) != 0:
                unit *= factor.subs(subs_dict)
                continue
            factors.append(factor)
            multiplicities.append(multiplicity)
        s = len(factors)
        normals = matrix(
            [
                [
                    f.derivative(v).subs(subs_dict) for v in vs
                ] for f in factors
            ]
        )
        if normals.rank() < s:
            raise ACSVException("Not a transverse intersection. Cannot deal with this case.")

        # Step 2: Find the locally parametrizing coordinates of the point pt
        # Since we have d variables and s factors, there should be d-s of these parametrizing coordinates
        # We will try to parametrize with the first d-s coordinates, shuffling the vs and r if it doesn't work
        for _ in range(s**2):
            Jac = matrix(
                [
                    [
                        ((v * Q.derivative(v))).subs(subs_dict) for v in vs[d-s:]
                    ] for Q in factors
                ]
            )
            if Jac.determinant() != 0:
                break

            acsv_logger.info("Variables do not parametrize, shuffling")
            vs_r_cp = list(zip(vs,r, cp))
            shuffle(vs_r_cp) # shuffle mutates the list
            vs, r, cp = zip(*vs_r_cp)
        else:
            raise ACSVException("Cannot find parametrizing set.")

        # Step 3: Compute the gamma matrix as defined in 9.10
        Gamma = matrix(
            [
                [
                    (v * Q.derivative(v)).subs(subs_dict) for v in vs
                ] for Q in factors
            ] + [
                [v.subs(subs_dict) if vs.index(v) == i else 0 for i in range(d)]
                for v in vs[:d-s]
            ]
        )

        # Some constants appearing for higher order singularities
        mult_fac = prod([factorial(m-1) for m in multiplicities])
        r_gamma_inv = prod([x**(multiplicities[i]-1) for i,x in  enumerate(list(vector(r)*Gamma.inverse())[:s])])
        # If cp lies on a single smooth component, we can compute asymptotics like in the smooth case
        if s == 1 and sum(multiplicities) == 1:
            n = SR.var('n')
            expansion = sum([
                term / (r[-1] * n)**(term_order)
                for term_order, term in enumerate(
                    GeneralTermAsymptotics(G, H, r, vs, cp, expansion_precision)
                )
            ])
            Det = GetHessian(H, vs, r).determinant()
            B = SR(1 / Det.subs(subs_dict) / r[-1]**(d-1) / 2**(d-1))
        else:
            # Higher order expansions not currently supported for non-smooth critical points
            if expansion_precision > 1:
                acsv_logger.warn("Higher order expansions are not supported in the non-smooth case. Defaulting to expansion_precision 1.")
            # For non-complete intersections, we must compute the parametrized Hessian matrix
            if s != d:
                Qw = ImplicitHessian(factors, vs, r, subs=subs_dict)
                expansion = SR(G.subs(subs_dict)/abs(Gamma.determinant())/unit)
                B = SR(prod([v for v in vs[:d-s]]).subs(subs_dict)/(r[-1] * Qw).determinant()/2**(d-s))
            else:
                expansion = SR(G.subs(subs_dict)/unit/abs(Gamma.determinant()))
                B = 1

        expansion *= (-1)**sum([m-1 for m in multiplicities]) * r_gamma_inv / mult_fac

        T = prod(SR(vs[i].subs(subs_dict))**r[i] for i in range(d))
        C = SR(1/T)
        D = QQ((s-d)/2 + sum(multiplicities) - s)
        try:
            B = QQbar(B)
            C = QQbar(C)
        except (ValueError, TypeError):
            pass
        
        asm_quantities.append([expansion,B,C,D,s])

    asm_vals = [(c, d, b.sqrt(), a, s) for a,b,c,d,s in asm_quantities]

    if as_symbolic:
        acsv_logger.warn(
            "The as_symbolic argument has been deprecated in favor of output_format='symbolic' "
        )
    
    if output_format is None:
        output_format = OutputFormat.ASYMPTOTIC
    else:
        output_format = OutputFormat(output_format)

    if output_format in (OutputFormat.TUPLE, OutputFormat.SYMBOLIC):
        n = SR.var('n')
        result = [
            (base, n**exponent, (pi**(s-d)).sqrt(), constant * expansion)
            for (base, exponent, constant, expansion, s) in asm_vals
        ]
        if output_format == OutputFormat.SYMBOLIC:
            result = sum([a**n * b * c * d for (a, b, c, d) in result])

    elif output_format == OutputFormat.ASYMPTOTIC:
        from sage.all import AsymptoticRing
        AR = AsymptoticRing('QQbar^n * n^QQ', QQbar)
        n = AR.gen()
        result = sum([
            constant * (pi**(s-d)).sqrt() * base**n * n**exponent * AR(expansion)
            + (abs(base)**n * n**(exponent - expansion_precision)).O()
            for (base, exponent, constant, expansion, s) in asm_vals
        ])

    else:
        raise NotImplementedError(f"Missing implementation for {output_format}")
    
    if return_points:
        return result, min_crit_pts

    return result

def GeneralTermAsymptotics(G, H, r, vs, cp, expansion_precision):
    r"""
    Compute coefficients of general (not necessarily leading) terms of
    the asymptotic expansion for a given critical
    point of a rational combinatorial multivariate rational function.

    Typically, this function is called as a subroutine of :func:`.diagonal_asy`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with `F = G/H`.
    * ``vs`` -- Tuple of variables occurring in `G` and `H`.
    * ``r`` -- The direction. A length `d` vector of positive algebraic numbers (usually
      integers).
    * ``cp`` -- A minimal critical point of `F` with coordinates specified in the
      same order as in ``vs``.
    * ``expansion_precision`` -- A positive integer value. This is the number of terms
      for which to compute coefficients in the asymptotic expansion.

    OUTPUT:

    List of coefficients of the asymptotic expansion.

    EXAMPLES::

        sage: from sage_acsv import GeneralTermAsymptotics
        sage: R.<x, y, z> = QQ[]
        sage: GeneralTermAsymptotics(1, 1 - x - y, [1, 1], [x, y], [1/2, 1/2], 5)
        [2, -1/4, 1/64, 5/512, -21/16384]
        sage: GeneralTermAsymptotics(1, 1 - x - y - z, [1, 1, 1], [x, y, z], [1/3, 1/3, 1/3], 4)
        [3, -2/3, 2/27, 14/729]
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
    tvars = SR.var('t', d - 1)
    G, H = R(SR(G)), R(SR(H))

    cp = {v: V for (v, V) in zip(vs, cp)}

    W = DifferentialWeylAlgebra(PolynomialRing(QQbar, tvars))
    TR = QQbar[[tvars]]
    T = TR.gens()
    tvars = T
    D = list(W.differentials())
    
    # Function to apply differential operator dop on function f
    def eval_op(dop, f):
        if len(f.parent().gens()) == 1:
            return sum([
                prod([factorial(k) for k in E[0][1]]) * E[1] * f[E[0][1][0]]
                for E in dop
            ])
        else:
            return sum([
                prod([factorial(k) for k in E[0][1]]) * E[1] * f[E[0][1]]
                for E in dop
            ])

    Hess = GetHessian(H, vs, r, cp)
    Hessinv = Hess.inverse()
    v = matrix(W, [D[:d-1]])
    Epsilon = -(v * Hessinv.change_ring(W) * v.transpose())[0,0]

    # P and PsiTilde only need to be computed to order 2M
    N = 2 * expansion_precision + 1
    
    # Find series expansion of function g given implicitly by 
    # H(w_1, ..., w_{d-1}, g(w_1, ..., w_{d-1})) = 0 up to needed order
    g = NewtonSeries(H.subs({v: v + cp[v] for v in vs}), vs, N)
    g = g.subs({v: v - cp[v] for v in vs}) + cp[vd]

    # Polar change of coordinates
    tsubs = {v: cp[v] * exp(I*t).add_bigoh(N) for v, t in zip(vs, tvars)}
    tsubs[vd] = g.subs(tsubs)

    # Compute PsiTilde up to needed order
    psi = log(g.subs(tsubs) / g.subs(cp)).add_bigoh(N)
    psi += I * sum([r[k]*tvars[k] for k in range(d-1)])/r[-1]
    v = matrix(TR, [tvars[k] for k in range(d-1)])
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
    EE = [Epsilon**k for k in range(3*expansion_precision - 2)]
    PP = [PSeries]
    for k in range(1,2*expansion_precision-1):
        PP.append(PP[k-1] * PsiSeries)

    # Function to compute constants appearing in asymptotic expansion
    def constants_clj(ell, j):
        extra_contrib = (-1)**j / (2**(ell + j) * factorial(ell) * factorial(ell + j))
        return extra_contrib * eval_op(EE[ell + j], PP[ell])

    res = [
        sum([constants_clj(ell, j) for ell in srange(2*j + 1)])
        for j in srange(expansion_precision)
    ]
    try:
        for i in range(len(res)):
            if res[i].imag() == 0:
                res[i] = AA(res[i])
    except:
        pass

    return res

def ContributingCombinatorialSmooth(G, H, variables, r=None, linear_form=None):
    r"""Compute contributing points of a combinatorial multivariate
    rational function F=G/H admitting a finite number of critical points.
    Assumes the singular variety of F is smooth.

    Typically, this function is called as a subroutine of :func:`.diagonal_asy_smooth`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with `F = G/H`.
    * ``variables`` -- Tuple of variables of ``G`` and ``H``, followed
      by ``lambda_, t, u_``.
    * ``r`` -- (Optional) the direction, a vector of positive algebraic
      numbers (usually integers).
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

    Examples::

        sage: from sage_acsv import ContributingCombinatorialSmooth
        sage: R.<x, y, w, lambda_, t, u_> = QQ[]
        sage: pts = ContributingCombinatorialSmooth(
        ....:     1,
        ....:     1 - w*(y + x + x^2*y + x*y^2),
        ....:     ([w, x, y], lambda_, t, u_)
        ....: )
        sage: sorted(pts)
        [[-1/4, -1, -1], [1/4, 1, 1]]
    """

    timer = Timer()

    # Fetch the variables we need
    vs, lambda_, t, u_ = variables
    vsT = vs + [t, lambda_]

    # If direction r is not given, default to the diagonal
    if r is None:
        r = [1 for _ in vs]

    # Make copy of r so they don't get mutated outside of function
    r = copy(r)

    # Replace irrational r with variable ri, add min poly to system
    r_variable_values = {}
    r_subs = []
    for idx, direction in enumerate(r):
        r_i = direction
        if AA(direction).minpoly().degree() > 1:
            r_i = SR.var(f"r{idx}")
            r_variable_values[r_i] = AA(direction)
        r_subs.append(r_i)
    
    r = r_subs

    # variables need to be ordered in this particular way
    expanded_R_variables = vs + list(r_variable_values.keys()) + [t, lambda_, u_]
    expanded_R = PolynomialRing(QQ, expanded_R_variables)

    vs = [expanded_R(v) for v in vs]
    t, lambda_, u_ = expanded_R(t), expanded_R(lambda_), expanded_R(u_)
    G, H = expanded_R(G), expanded_R(H)

    r_variable_values = {
        expanded_R(ri): val for (ri, val) in r_variable_values.items()
    }
    r = [expanded_R(ri) for ri in r]

    vsT = vs + list(r_variable_values.keys()) + [t, lambda_]

    # Create the critical point equations system
    vsH = H.variables()
    system = [
        H_var * H.derivative(H_var) - r_var * lambda_
        for H_var, r_var in zip(H.variables(), r)
    ]
    system.extend([H, H.subs({z: z*t for z in vsH})])
    system.extend([
        direction_value.minpoly().subs(direction_var)
        for direction_var, direction_value in r_variable_values.items()
    ])

    # Compute the Kronecker representation of our system
    timer.checkpoint()

    P, Qs = _kronecker_representation(system, u_, vsT, lambda_, linear_form)
    timer.checkpoint("Kronecker")

    Qt = Qs[-2]  # Qs ordering is H.variables() + rvars + [t, lambda_]
    Pd = P.derivative()

    # Solutions to Pt are solutions to the system where t is not 1
    one_minus_t = gcd(Pd - Qt, P)
    Pt, _ = P.quo_rem(one_minus_t)
    rts_t_zo = list(
        filter(
            lambda k: (Qt/Pd).subs(u_=k) > 0 and (Qt/Pd).subs(u_=k) < 1,
            Pt.roots(AA, multiplicities=False)
        )
    )
    non_min = [[(q/Pd).subs(u_=u) for q in Qs[0:-2]] for u in rts_t_zo]

    # Filter the real roots for minimal points with positive coords
    pos_minimals = []
    for u in one_minus_t.roots(AA, multiplicities=False):
        is_min = True
        v = [(q/Pd).subs(u_=u) for q in Qs[:len(vs)]]
        rv = {
            ri: (q/Pd).subs(u_=u)
            for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
        }
        if any([rv[ri] != ri_value for ri, ri_value in r_variable_values.items()]):
            continue
        if any([value <= 0 for value in v[:len(vs)]]):
            continue
        for pt in non_min:
            if all([a == b for (a, b) in zip(v, pt)]):
                is_min = False
                break
        if is_min:
            pos_minimals.append(u)

    # Remove non-smooth points and points with zero coordinates (where lambda=0)
    for i in range(len(pos_minimals)):
        x = (Qs[-1]/Pd).subs(u_=pos_minimals[i])
        if x == 0:
            acsv_logger.warning(
                f"Removing critical point {pos_minimals[i]} because it either "
                "has a zero coordinate or is not smooth."
            )
            pos_minimals.pop(i)

    # Verify necessary assumptions
    if len(pos_minimals) == 0:
        raise ACSVException("No smooth minimal critical points found.")
    elif len(pos_minimals) > 1:
        raise ACSVException(
            f"More than one minimal point with positive real coordinates found: {pos_minimals}"
        )

    # Find all minimal critical points
    minCP = [(q/Pd).subs(u_=pos_minimals[0]) for q in Qs[0:-2]]
    minimals = []

    for u in one_minus_t.roots(QQbar, multiplicities=False):
        v = [(q/Pd).subs(u_=u) for q in Qs[:len(vs)]]
        rv = {
            ri: (q/Pd).subs(u_=u)
            for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
        }
        if any([rv[r_var] != r_value for r_var, r_value in r_variable_values.items()]):
            continue
        if all([a.abs() == b.abs() for (a, b) in zip(minCP, v)]):
            minimals.append(u)

    # Get minimal point coords, and make exact if possible
    minimal_coords = [[(q/Pd).subs(u_=u) for q in Qs[:len(vs)]] for u in minimals]
    [[a.exactify() for a in b] for b in minimal_coords]

    timer.checkpoint("Minimal Points")

    return [[(q/Pd).subs(u_=u) for q in Qs[:len(vs)]] for u in minimals]

def ContributingCombinatorial(
    G,
    H,
    variables,
    r=None,
    linear_form=None,
    m2=None,
    whitney_strat=None,
):
    r"""Compute contributing points of a combinatorial multivariate
    rational function F=G/H admitting a finite number of critical points.

    Typically, this function is called as a subroutine of :func:`.diagonal_asy`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with ``F = G/H``
    * ``variables`` -- Tuple of variables of ``G`` and ``H``, followed
      by ``lambda_, t, u_``
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions
    * ``m2`` -- (Optional) The option to pass in a SageMath Macaulay2 interface for
        computing primary decompositions. Macaulay2 must be installed by the user
    * ``whitney_strat`` -- (Optional) The user can pass in a Whitney Stratification of V(H)
        to save computation time. The program will not check if this stratification is correct.
        The whitney_strat should be an array of length ``d``, with the ``k``-th entry a list of
        tuples of ideal generators representing a component of the ``k``-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`,

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    Examples::

        sage: from sage_acsv import ContributingCombinatorial
        sage: R.<x, y, lambda_, t, u_> = QQ[]
        sage: pts = ContributingCombinatorial(
        ....:     1,
        ....:     (1-(2*x+y)/3)*(1-(3*x+y)/4),
        ....:     ([x, y], lambda_, t, u_)
        ....: )
        sage: sorted(pts)
        [[3/4, 3/2]]

    """

    #####
    timer = Timer()

    # Fetch the variables we need
    vs, lambda_, t, u_ = variables
    vsT = vs + [t, lambda_]

    # If direction r is not given, default to the diagonal
    if r is None:
        r = [1 for _ in range(len(vs))]

    # Make copies of r so they don't get mutated outside of function
    r_copy = copy(r)

    # Replace irrational r with variable ri, add min poly to system
    r_variable_values = {}
    r_subs = []
    for idx, direction in enumerate(r):
        r_i = direction
        if AA(direction).minpoly().degree() > 1:
            r_i = SR.var(f"r{idx}")
            r_variable_values[r_i] = AA(direction)
        r_subs.append(r_i)

    r = r_subs

    expanded_R_variables = vs + list(r_variable_values.keys()) + [t, lambda_, u_]
    expanded_R = PolynomialRing(QQ, expanded_R_variables)

    vs = [expanded_R(v) for v in vs]
    t, lambda_, u_ = expanded_R(t), expanded_R(lambda_), expanded_R(u_)
    G, H = expanded_R(G), expanded_R(H)

    r_variable_values = {
        expanded_R(ri):val for (ri, val) in r_variable_values.items()
    }
    r = [expanded_R(ri) for ri in r]

    vsT = vs + list(r_variable_values.keys())+ [t, lambda_]

    # Compute the critical point system for each stratum
    pure_H = PolynomialRing(QQ, vs)

    if whitney_strat is None:
        whitney_strat = WhitneyStrat(Ideal(pure_H(H)), pure_H, m2)
    else:
        # Cast symbolic generators for provided stratification into the correct ring
        whitney_strat = [prod([Ideal([pure_H(f) for f in comp]) for comp in stratum]) for stratum in whitney_strat]

    critical_point_ideals = []
    for d, stratum in enumerate(whitney_strat):
        critical_point_ideals.append([])
        for P in PrimaryDecomposition(stratum):
            c = len(vs) - d
            P_ext = P.change_ring(expanded_R)
            M = matrix(
                [
                    [v * f.derivative(v) for v in vs] for f in P_ext.gens()
                ] + [r]
            )
            # Add in min polys for the direction variables
            r_polys = [
                direction_value.minpoly().subs(direction_var) 
                for direction_var, direction_value in r_variable_values.items()
            ]
            # Create ideal of expanded_R containing extended critical point equations
            cpid = P_ext + Ideal(M.minors(c+1) + [H.subs({v:v*t for v in vs}), (prod(vs)*lambda_ - 1)] + r_polys)
            # Saturate cpid by lower dimension stratum, if d > 0
            if d > 0:
                cpid = Saturate(cpid, whitney_strat[d-1].change_ring(expanded_R))
            
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
            P, Qs = _kronecker_representation(ideal.gens(), u_, vsT, lambda_, linear_form)

            Qt = Qs[-2]  # Qs ordering is H.variables() + rvars + [t, lambda_]
            Pd = P.derivative()

            # Solutions to Pt are solutions to the system where t is not 1
            one_minus_t = gcd(Pd - Qt, P)
            Pt, _ = P.quo_rem(one_minus_t)
            rts_t_zo = list(
                filter(
                    lambda k: (Qt/Pd).subs(u_=k) > 0 and (Qt/Pd).subs(u_=k) < 1,
                    Pt.roots(AA, multiplicities=False)
                )
            )
            non_min = [[(q/Pd).subs(u_=u) for q in Qs[0:-2]] for u in rts_t_zo]

            # Filter the real roots for minimal points with positive coords
            pos_minimals = []
            for u in one_minus_t.roots(AA, multiplicities=False):
                is_min = True
                v = [(q/Pd).subs(u_=u) for q in Qs[:len(vs)]]
                rv = {
                    ri: (q/Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
                }
                if any([rv[ri] != ri_value for ri, ri_value in r_variable_values.items()]):
                    continue
                if any([value <= 0 for value in v[:len(vs)]]):
                    continue
                for pt in non_min:
                    if all([a == b for (a, b) in zip(v, pt)]):
                        is_min = False
                        break
                if is_min:
                    pos_minimals.append(u)

            pos_minimals_by_stratum[d].extend(
                [[QQbar((q/Pd).subs(u_=u)) for q in Qs[:len(vs)]] for u in pos_minimals]
            )

            # Characterize all complex critical points in each stratum
            for u in one_minus_t.roots(QQbar, multiplicities=False):
                rv = {
                    ri : (q/Pd).subs(u_=u) 
                    for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
                }
                if (any([rv[r_var] != r_value for r_var, r_value in r_variable_values.items()])):
                    continue

                w = [QQbar((q/Pd).subs(u_=u)) for q in Qs[:len(vs)]]
                critical_points_by_stratum[d].append(w)

    # Refine positive minimal critical points to those that are contributing
    contributing_pos_minimals = []
    all_factors = list(factor[0] for factor in H.factor())
    r = r_copy
    for d in reversed(range(len(critical_point_ideals))):
        pos_minimals = pos_minimals_by_stratum[d]
        if len(contributing_pos_minimals) > 0:
            break 

        for x in pos_minimals:
            if IsContributing(vs, x, r, all_factors, len(vs)-d):
                contributing_pos_minimals.append(x)
                for i in range(d):
                    stratum = whitney_strat[i]
                    if stratum.subs({pure_H(wi):val for wi, val in zip(vs, x)}) == Ideal(pure_H(0)):
                        raise ACSVException(
                            "Non-generic critical point found - {w} is contained in {dim}-dimensional stratum".format(w = str(x), dim = i)
                        )

    if len(contributing_pos_minimals) == 0:
        raise ACSVException("No smooth minimal critical points found.")
    if len(contributing_pos_minimals) > 1:
        raise ACSVException(
            f"More than one minimal point with positive real coordinates found: {contributing_pos_minimals}"
        )
    minimal = contributing_pos_minimals[0]

    # Characterize all complex contributing points
    contributing_points = []
    for d in reversed(range(len(critical_point_ideals))):
        for w in critical_points_by_stratum[d]:
            if all([abs(w_i)==abs(min_i) for w_i, min_i in zip(w, minimal)]) and IsContributing(vs, w, r, all_factors, len(vs)-d):
                contributing_points.append(w)

    return contributing_points

def MinimalCriticalCombinatorial(G, H, variables, r=None, linear_form=None, m2=None, whitney_strat=None):
    r"""Compute nonzero minimal critical points of a combinatorial multivariate
    rational function F=G/H admitting a finite number of critical points.

    Typically, this function is called as a subroutine of :func:`.diagonal_asy`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with ``F = G/H``
    * ``variables`` -- Tuple of variables of ``G`` and ``H``, followed
      by ``lambda_, t, u_``
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions
    * ``m2`` -- (Optional) The option to pass in a SageMath Macaulay2 interface for
        computing primary decompositions. Macaulay2 must be installed by the user
    * ``whitney_strat`` -- (Optional) The user can pass in a Whitney Stratification of V(H)
        to save computation time. The program will not check if this stratification is correct.
        The whitney_strat should be an array of length ``d``, with the ``k``-th entry a list of
        tuples of ideal generators representing a component of the ``k``-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`,

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    Examples::

    """

    #####
    # Fetch the variables we need
    vs, lambda_, t, u_ = variables
    vsT = vs + [t, lambda_]

    # If direction r is not given, default to the diagonal
    if r is None:
        r = [1 for _ in range(len(vs))]

    # Replace irrational r with variable ri, add min poly to system
    r_variable_values = {}
    r_subs = []
    for idx, direction in enumerate(r):
        r_i = direction
        if AA(direction).minpoly().degree() > 1:
            r_i = SR.var(f"r{idx}")
            r_variable_values[r_i] = AA(direction)
        r_subs.append(r_i)

    r = r_subs

    expanded_R_variables = vs + list(r_variable_values.keys()) + [t, lambda_, u_]
    expanded_R = PolynomialRing(QQ, expanded_R_variables)

    vs = [expanded_R(v) for v in vs]
    t, lambda_, u_ = expanded_R(t), expanded_R(lambda_), expanded_R(u_)
    G, H = expanded_R(G), expanded_R(H)

    r_variable_values = {
        expanded_R(ri):val for (ri, val) in r_variable_values.items()
    }
    r = [expanded_R(ri) for ri in r]

    vsT = vs + list(r_variable_values.keys())+ [t, lambda_]

    # Compute the critical point system for each stratum
    pure_H = PolynomialRing(QQ, vs)

    if whitney_strat is None:
        whitney_strat = WhitneyStrat(Ideal(pure_H(H)), pure_H, m2)
    else:
        # Cast symbolic generators for provided stratification into the correct ring
        whitney_strat = [prod([Ideal([pure_H(f) for f in comp]) for comp in stratum]) for stratum in whitney_strat]

    critical_points = []
    pos_minimals = []
    for d, stratum in enumerate(whitney_strat):
        for P_comp in PrimaryDecomposition(stratum):
            c = len(vs) - d
            P_ext = P_comp.change_ring(expanded_R)
            M = matrix(
                [
                    [v * f.derivative(v) for v in vs] for f in P_ext.gens()
                ] + [r]
            )
            # Add in min polys for the direction variables
            r_polys = [
                direction_value.minpoly().subs(direction_var) 
                for direction_var, direction_value in r_variable_values.items()
            ]
            # Create ideal of expanded_R containing extended critical point equations
            ideal = P_ext + Ideal(M.minors(c+1) + [H.subs({v:v*t for v in vs}), (prod(vs)*lambda_ - 1)] + r_polys)
            # Saturate cpid by lower dimension stratum, if d > 0
            if d > 0:
                ideal = Saturate(ideal, whitney_strat[d-1].change_ring(expanded_R))
        
            if ideal.dimension() < 0:
                continue
            P, Qs = _kronecker_representation(ideal.gens(), u_, vsT, lambda_, linear_form)

            Qt = Qs[-2]  # Qs ordering is H.variables() + rvars + [t, lambda_]
            Pd = P.derivative()

            # Solutions to Pt are solutions to the system where t is not 1
            one_minus_t = gcd(Pd - Qt, P)
            Pt, _ = P.quo_rem(one_minus_t)
            rts_t_zo = list(
                filter(
                    lambda k: (Qt/Pd).subs(u_=k) > 0 and (Qt/Pd).subs(u_=k) < 1,
                    Pt.roots(AA, multiplicities=False)
                )
            )
            non_min = [[(q/Pd).subs(u_=u) for q in Qs[0:-2]] for u in rts_t_zo]

            # Filter the real roots for minimal points with positive coords
            for u in one_minus_t.roots(AA, multiplicities=False):
                is_min = True
                v = [(q/Pd).subs(u_=u) for q in Qs[:len(vs)]]
                rv = {
                    ri: (q/Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
                }
                if any([rv[ri] != ri_value for ri, ri_value in r_variable_values.items()]):
                    continue
                if any([value <= 0 for value in v[:len(vs)]]):
                    continue
                for pt in non_min:
                    if all([a == b for (a, b) in zip(v, pt)]):
                        is_min = False
                        break
                if is_min:
                    pos_minimals.append(v)

            # Characterize all complex critical points in each stratum
            for u in one_minus_t.roots(QQbar, multiplicities=False):
                rv = {
                    ri : (q/Pd).subs(u_=u) 
                    for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
                }
                if (any([rv[r_var] != r_value for r_var, r_value in r_variable_values.items()])):
                    continue

                w = [QQbar((q/Pd).subs(u_=u)) for q in Qs[:len(vs)]]
                critical_points.append(w)

    if len(pos_minimals) == 0:
        raise ACSVException("No critical points found.")

    # Characterize all complex contributing points
    minimal_criticals = []
    for w in critical_points:
        if any([all([abs(w_i)==abs(min_i) for w_i, min_i in zip(w, minimal)]) for minimal in pos_minimals]):
            minimal_criticals.append(w)

    return minimal_criticals

def CriticalPoints(G, H, variables, r=None, linear_form=None, m2=None, whitney_strat=None):
    r"""Compute critical points of a multivariate
    rational function F=G/H admitting a finite number of critical points.

    Typically, this function is called as a subroutine of :func:`.diagonal_asy`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with ``F = G/H``
    * ``variables`` -- Tuple of variables of ``G`` and ``H``, followed
      by ``t`` and ``u_``
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions
    * ``m2`` -- (Optional) The option to pass in a SageMath Macaulay2 interface for
        computing primary decompositions. Macaulay2 must be installed by the user
    * ``whitney_strat`` -- (Optional) The user can pass in a Whitney Stratification of V(H)
        to save computation time. The program will not check if this stratification is correct.
        The whitney_strat should be an array of length ``d``, with the ``k``-th entry a list of
        tuples of ideal generators representing a component of the ``k``-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`,

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    Examples::

    """

    #####

    # Fetch the variables we need
    vs, lambda_, u_ = variables
    vsT = vs + [lambda_]

    # If direction r is not given, default to the diagonal
    if r is None:
        r = [1 for _ in range(len(vs))]

    # Replace irrational r with variable ri, add min poly to system
    r_variable_values = {}
    r_subs = []
    for idx, direction in enumerate(r):
        r_i = direction
        if AA(direction).minpoly().degree() > 1:
            r_i = SR.var(f"r{idx}")
            r_variable_values[r_i] = AA(direction)
        r_subs.append(r_i)

    r = r_subs

    expanded_R_variables = vs + list(r_variable_values.keys()) + [lambda_, u_]
    expanded_R = PolynomialRing(QQ, expanded_R_variables)

    vs = [expanded_R(v) for v in vs]
    lambda_, u_ = expanded_R(lambda_), expanded_R(u_)
    G, H = expanded_R(G), expanded_R(H)

    r_variable_values = {
        expanded_R(ri):val for (ri, val) in r_variable_values.items()
    }
    r = [expanded_R(ri) for ri in r]

    vsT = vs + list(r_variable_values.keys())+ [lambda_]

    # Compute the critical point system for each stratum
    pure_H = PolynomialRing(QQ, vs)

    if whitney_strat is None:
        whitney_strat = WhitneyStrat(Ideal(pure_H(H)), pure_H, m2)
    else:
        # Cast symbolic generators for provided stratification into the correct ring
        whitney_strat = [prod([Ideal([pure_H(f) for f in comp]) for comp in stratum]) for stratum in whitney_strat]

    critical_points = []
    for d, stratum in enumerate(whitney_strat):
        for P_comp in PrimaryDecomposition(stratum):
            c = len(vs) - d
            P_ext = P_comp.change_ring(expanded_R)
            M = matrix(
                [
                    [v * f.derivative(v) for v in vs] for f in P_ext.gens()
                ] + [r]
            )
            # Add in min polys for the direction variables
            r_polys = [
                direction_value.minpoly().subs(direction_var) 
                for direction_var, direction_value in r_variable_values.items()
            ]
            # Create ideal of expanded_R containing extended critical point equations
            ideal = P_ext + Ideal(M.minors(c+1) + [(prod(vs)*lambda_ - 1)] + r_polys)
            # Saturate cpid by lower dimension stratum, if d > 0
            if d > 0:
                ideal = Saturate(ideal, whitney_strat[d-1].change_ring(expanded_R))
        
            if ideal.dimension() < 0:
                continue
            P, Qs = _kronecker_representation(ideal.gens(), u_, vsT, lambda_, linear_form)

            Qt = Qs[-2]  # Qs ordering is H.variables() + rvars + [t, lambda_]
            Pd = P.derivative()

            # Characterize all complex critical points in each stratum
            for u in P.roots(QQbar, multiplicities=False):
                rv = {
                    ri : (q/Pd).subs(u_=u) 
                    for (ri, q) in zip(r_variable_values, Qs[len(vs):-2])
                }
                if (any([rv[r_var] != r_value for r_var, r_value in r_variable_values.items()])):
                    continue

                w = [QQbar((q/Pd).subs(u_=u)) for q in Qs[:len(vs)]]
                critical_points.append(w)

    return critical_points