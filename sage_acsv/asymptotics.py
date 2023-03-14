"""Functions for determining asymptotics of the coefficients
of multivariate rational functions.
"""

from sage.all import AA, PolynomialRing, QQ, QQbar, SR, gcd, prod, pi

from sage_acsv.kronecker import _kronecker_representation
from sage_acsv.helpers import ACSVException, RationalFunctionReduce, DetHessianWithLog
from sage_acsv.debug import Timer, acsv_logger


MAX_MIN_CRIT_RETRIES = 3


def diagonal_asy(F, r=None, linear_form=None, return_points=False, as_symbolic=False):
    r"""Asymptotics in a given direction r of the multivariate rational function F.

    INPUT:

    * ``F`` -- The rational function ``G/H`` in ``d`` variables. This function is
      assumed to have a combinatorial expansion.
    * ``r`` -- A vector of length d of positive integers.
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions.
    * ``return_points`` -- If ``True``, also returns the coordinates of
      minimal critical points. By default ``False``.
    * ``as_symbolic`` -- If ``False`` (the default value), a list of tuples
      of the form ``(a, n^b, pi^c, d)`` is returned, such that the `r`-diagonal
      of `F` is the sum of `a^n n^b \pi^c d + O(a^n n^{b-1})` over these tuples.
      Otherwise, if ``True``, the symbolic expression corresponding to the sum
      from the other case without the error terms (i.e., a symbolic representation
      of the asympttoic main term) is returned.

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
        sage: diagonal_asy(1/(1-(x+y+z)+(3/4)*x*y*z), as_symbolic=True)
        0.840484893481498?*24.68093482214177?^n/(pi^1.0*n^1.0)
        sage: diagonal_asy(1/(1-(x+y+z)+(3/4)*x*y*z))
        [(24.68093482214177?, n^(-1.0), pi^(-1.0), 0.840484893481498?)]
        sage: var('n')
        n
        sage: asy = diagonal_asy(
        ....:     1/(1 - w*(1 + x)*(1 + y)*(1 + z)*(x*y*z + y*z + y + z + 1))
        ....: )
        sage: sum([
        ....:      a.radical_expression()^n * b * c * d.radical_expression()
        ....:      for (a, b, c, d) in asy
        ....: ])
        1/4*(12*sqrt(2) + 17)^n*sqrt(17/2*sqrt(2) + 12)/(pi^1.5*n^1.5)

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
    A = SR(-G / vs[-1] / H.derivative(vs[-1]))
    B = SR(1 / Det / rd**(d-1) / 2**(d-1))
    C = SR(1 / T)

    # Compute constants at contributing singularities
    asm_quantities = [
        [QQbar(q.subs([SR(v) == V for (v, V) in zip(vs, cp)])) for q in [A, B, C]]
        for cp in min_crit_pts
    ]
    n = SR.var('n')
    asm_vals = [
        (c, n**((1-d)/2), pi**((1-d)/2), a * b.sqrt())
        for (a, b, c) in asm_quantities
    ]
    timer.checkpoint("Final Asymptotics")

    result = asm_vals
    if as_symbolic:
        result = sum([a**n * b * c * d for (a, b, c, d) in result])

    if return_points:
        return result, min_crit_pts
    return result


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
