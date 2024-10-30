"""Functions for determining asymptotics of the coefficients
of multivariate rational functions.
"""

from sage.all import AA, PolynomialRing, QQ, QQbar, SR, DifferentialWeylAlgebra, Ideal
from sage.all import gcd, prod, pi, matrix, exp, log, add, I, factorial

from sage_acsv.kronecker import _kronecker_representation
from sage_acsv.helpers import ACSVException, NewtonSeries, RationalFunctionReduce, DetHessianWithLog, OutputFormat, GetHessian
from sage_acsv.debug import Timer, acsv_logger


MAX_MIN_CRIT_RETRIES = 3


def diagonal_asy(
    F,
    r=None,
    linear_form=None,
    expansion_precision=1,
    return_points=False,
    output_format=None,
    as_symbolic=False
):
    r"""Asymptotics in a given direction r of the multivariate rational function F.

    INPUT:

    * ``F`` -- The rational function ``G/H`` in ``d`` variables. This function is
      assumed to have a combinatorial expansion.
    * ``r`` -- A vector of length d of positive integers.
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

        sage: from sage_acsv import diagonal_asy
        sage: var('x,y,z,w')
        (x, y, z, w)
        sage: diagonal_asy(1/(1-x-y))
        [(4, 1/sqrt(n), 1/sqrt(pi), 1)]
        sage: diagonal_asy(1/(1-(1+x)*y), r = [1,2], return_points=True)
        ([(4, 1/sqrt(n), 1/sqrt(pi), 1)], [[1, 1/2]])
        sage: diagonal_asy(1/(1-(x+y+z)+(3/4)*x*y*z), output_format="symbolic")
        0.840484893481498?*24.68093482214177?^n/(pi*n)
        sage: diagonal_asy(1/(1-(x+y+z)+(3/4)*x*y*z))
        [(24.68093482214177?, 1/n, 1/pi, 0.840484893481498?)]
        sage: var('n')
        n
        sage: asy = diagonal_asy(
        ....:     1/(1 - w*(1 + x)*(1 + y)*(1 + z)*(x*y*z + y*z + y + z + 1))
        ....: )
        sage: sum([
        ....:      a.radical_expression()^n * b * c * d.radical_expression()
        ....:      for (a, b, c, d) in asy
        ....: ])
        1/4*(12*sqrt(2) + 17)^n*sqrt(17/2*sqrt(2) + 12)/(pi^(3/2)*n^(3/2))

    Not specifying any ``output_format`` falls back to the default tuple
    representation::

        sage: from sage_acsv import diagonal_asy, OutputFormat
        sage: var('x')
        x
        sage: diagonal_asy(1/(1 - 2*x))
        [(2, 1, 1, 1)]
        sage: diagonal_asy(1/(1 - 2*x), output_format="tuple")
        [(2, 1, 1, 1)]

    Passing ``"symbolic"`` lets the function return an element of the
    symbolic ring in the variable ``n`` that describes the asymptotic growth::

        sage: growth = diagonal_asy(1/(1 - 2*x), output_format="symbolic"); growth
        2^n
        sage: growth.parent()
        Symbolic Ring

    The argument ``"asymptotic"`` constructs an asymptotic expansion over
    an appropriate ``AsymptoticRing`` in the variable ``n``, including the
    appropriate error term::

        sage: assume(SR.an_element() > 0)  # required to make coercions involving SR work properly
        sage: growth = diagonal_asy(1/(1 - x - y), output_format="asymptotic"); growth
        1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))
        sage: growth.parent()
        Asymptotic Ring <SR^n * n^QQ * Arg_SR^n> over Symbolic Ring

    Increasing the precision of the expansion returns an expansion with more terms
    (works for all available output formats)::

        sage: diagonal_asy(1/(1 - x - y), expansion_precision=3, output_format="asymptotic")
        1/sqrt(pi)*4^n*n^(-1/2) - 1/8/sqrt(pi)*4^n*n^(-3/2) + 1/128/sqrt(pi)*4^n*n^(-5/2)
        + O(4^n*n^(-7/2))

    The function times individual steps of the algorithm, timings can
    be displayed by increasing the printed verbosity level of our debug logger::

        sage: import logging
        sage: from sage_acsv.debug import acsv_logger
        sage: acsv_logger.setLevel(logging.INFO)
        sage: diagonal_asy(1/(1 - x - y))
        INFO:sage_acsv:... Executed Kronecker in ... seconds.
        INFO:sage_acsv:... Executed Minimal Points in ... seconds.
        INFO:sage_acsv:... Executed Final Asymptotics in ... seconds.
        [(4, 1/sqrt(n), 1/sqrt(pi), 1)]
        sage: acsv_logger.setLevel(logging.WARNING)


    Tests:

    Check that passing a non-supported ``output_format`` errors out::

        sage: diagonal_asy(1/(1 - x - y), output_format='hello world')
        Traceback (most recent call last):
        ...
        ValueError: 'hello world' is not a valid OutputFormat
        sage: diagonal_asy(1/(1 - x - y), output_format=42)
        Traceback (most recent call last):
        ...
        ValueError: 42 is not a valid OutputFormat

    """
    G, H = F.numerator(), F.denominator()
    if r is None:
        n = len(H.variables())
        r = [1 for _ in range(n)]

    # Initialize variables
    vs = list(H.variables())

    RR, (t, lambda_, u_) = PolynomialRing(QQ, 't, lambda_, u_').objgens()
    expanded_R, _ = PolynomialRing(QQ, len(vs)+3, vs + [t, lambda_, u_]).objgens()

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
            min_crit_pts = MinimalCriticalCombinatorial(
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
    Det = DetHessianWithLog(H, vsT[0:-2], r)

    # Find exponential growth
    T = prod([vs[i]**r[i] for i in range(d)])


    # Find constants appearing in asymptotics in terms of original variables
    A = SR(-G / vd / H.derivative(vd))
    B = SR(1 / Det / rd**(d-1) / 2**(d-1))
    C = SR(1 / T)

    # Compute constants at contributing singularities
    n = SR.var('n')
    asm_quantities = []
    for cp in min_crit_pts:
        subs_dict = {SR(v): V for (v, V) in zip(vs, cp)}
        if expansion_precision == 1: 
            # TODO: GeneralTermAsymptotics should return the expansion
            # when passing a precision of 1
            expansion = QQbar(A.subs(subs_dict))
        else:
            expansion = sum([
                term / (rd * n)**(term_order)
                for term_order, term in enumerate(
                    GeneralTermAsymptotics(G, H, r, vs, cp, expansion_precision)
                )
            ])
        asm_quantities.append([expansion, QQbar(B.subs(subs_dict)), QQbar(C.subs(subs_dict))])

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
        output_format = OutputFormat.TUPLE
    else:
        output_format = OutputFormat(output_format)

    if output_format in (OutputFormat.TUPLE, OutputFormat.SYMBOLIC):
        n = SR.var('n')
        result = [
            (base, n**exponent, pi**exponent, constant*expansion)
            for (base, exponent, constant, expansion) in asm_vals
        ]
        if output_format == OutputFormat.SYMBOLIC:
            result = sum([a**n * b * c * d for (a, b, c, d) in result])

    elif output_format == OutputFormat.ASYMPTOTIC:
        from sage.all import AsymptoticRing
        SR_without_n = SR.subring(rejecting_variables=('n',))
        AR = AsymptoticRing('SR^n * n^QQ', SR_without_n)
        n = AR.gen()
        result = sum([
            constant * pi**exponent * base**n * n**exponent * (AR(expansion) + (n**(-expansion_precision)).O())
            for (base, exponent, constant, expansion) in asm_vals
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
    * ``r`` -- The direction. A length `d` vector of positive integers.
    * ``cp`` -- A minimal critical point of `F` with coordinates specified in the
      same order as in ``vs``.
    * ``expansion_precision`` -- A positive integer value. This is the number of terms
      for which to compute coefficients in the asymptotic expansion.

    OUTPUT:

    List of coefficients of the asymptotic expansion.
    """
    
    # Convert everything to field of algebraic numbers
    d = len(vs)
    R = PolynomialRing(QQbar, vs)
    vs = R.gens()
    vd = vs[-1]
    tvars = tuple(SR.var('t%d'%i) for i in range(d-1))
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
            return add([prod([factorial(k) for k in E[0][1]])*E[1]*f[E[0][1][0]] for E in dop])
        else:
            return add([prod([factorial(k) for k in E[0][1]])*E[1]*f[(v for v in E[0][1])] for E in dop])

    Hess = GetHessian(H, vs, r, cp)
    Hessinv = Hess.inverse()
    v = matrix(W,[[D[k] for k in range(d-1)]])
    Epsilon = -(v * Hessinv.change_ring(W) * v.transpose())[0,0]

    # P and PsiTilde only need to be computed to order 2M
    N = 2 * expansion_precision + 1
    
    # Find series expansion of function g given implicitly by 
    # H(w_1, ..., w_{d-1}, g(w_1, ..., w_{d-1})) = 0 up to needed order
    g = NewtonSeries(H.subs({v:v+v.subs(cp) for v in vs}), vs, N)
    g = g.subs({v:v-v.subs(cp) for v in vs}) + vd.subs(cp)

    # Polar change of coordinates
    tsubs = {v : v.subs(cp)*exp(I*t).add_bigoh(N) for [v,t] in zip(vs,tvars)}
    tsubs[vd] = g.subs(tsubs)

    # Compute PsiTilde up to needed order
    psi = log(g.subs(tsubs)/g.subs(cp)).add_bigoh(N)
    psi += I * add([r[k]*tvars[k] for k in range(d-1)])/r[-1]
    v = matrix(TR,[tvars[k] for k in range(d-1)])
    psiTilde = psi - (v * Hess * v.transpose())[0,0]/2
    PsiSeries = psiTilde.truncate(N)

    # Compute series expansion of P = -G/(g*H_{z_d}) up to needed order
    P_num = -G.subs(tsubs).add_bigoh(N)
    P_denom = (g*H.derivative(vd)).subs(tsubs).add_bigoh(N)
    PSeries = (P_num/P_denom).truncate(N)

    if len(tvars) > 1:
        PsiSeries = PsiSeries.polynomial()
        PSeries = PSeries.polynomial()

    # Precompute products used for asymptotics
    EE = [Epsilon**k for k in range(3*expansion_precision-2)]
    PP = [PSeries] + [0 for k in range(2*expansion_precision-2)]
    for k in range(1,2*expansion_precision-1):
        PP[k] = PP[k-1]*PsiSeries

    # Function to compute constants appearing in asymptotic expansion
    def Clj(l,j):
        return (-1)**j*SR(eval_op(EE[l+j],PP[l]))/(2**(l+j)*factorial(l)*factorial(l+j))

    return [sum([Clj(l,j) for l in range(2 * j + 1)]) for j in range(expansion_precision)]

def MinimalCriticalCombinatorial(G, H, variables, r=None, linear_form=None):
    r"""Compute minimal critical points of a combinatorial multivariate
    rational function F=G/H admitting a finite number of critical points.

    Typically, this function is called as a subroutine of :func:`.diagonal_asy`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with `F = G/H`
    * ``variables`` -- Tuple of variables of ``G`` and ``H``, followed
      by ``lambda_, t, u_``
    * ``r`` -- (Optional) Length `d` vector of positive integers
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions

    OUTPUT:

    List of minimal critical points of `F` in the direction `r`,
    as a list of tuples of algebraic numbers.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    Examples::

        sage: from sage_acsv import MinimalCriticalCombinatorial
        sage: R.<x, y, w, lambda_, t, u_> = QQ[]
        sage: pts = MinimalCriticalCombinatorial(
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
        r = [1 for i in range(len(vs))]

    # Create the critical point equations system
    vsH = H.variables()
    system = [
        vsH[i]*H.derivative(vsH[i]) - r[i]*lambda_
        for i in range(len(vsH))
    ] + [H, H.subs({z: z*t for z in vsH})]

    # Compute the Kronecker representation of our system
    timer.checkpoint()
    P, Qs = _kronecker_representation(system, u_, vsT, lambda_, linear_form)
    timer.checkpoint("Kronecker")

    Qt = Qs[-2]  # Qs ordering is H.variables() + [t, lambda_]
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
        v = [(q/Pd).subs(u_=u) for q in Qs[0:-2]]
        if any([k <= 0 for k in v]):
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
            "More than one minimal point with positive real coordinates found."
        )

    # Find all minimal critical points
    minCP = [(q/Pd).subs(u_=pos_minimals[0]) for q in Qs[0:-2]]
    minimals = []

    for u in one_minus_t.roots(QQbar, multiplicities=False):
        v = [(q/Pd).subs(u_=u) for q in Qs[0:-2]]
        if all([a.abs() == b.abs() for (a, b) in zip(minCP, v)]):
            minimals.append(u)

    # Get minimal point coords, and make exact if possible
    minimal_coords = [[(q/Pd).subs(u_=u) for q in Qs[0:-2]] for u in minimals]
    [[a.exactify() for a in b] for b in minimal_coords]

    timer.checkpoint("Minimal Points")
    return [[(q/Pd).subs(u_=u) for q in Qs[0:-2]] for u in minimals]
