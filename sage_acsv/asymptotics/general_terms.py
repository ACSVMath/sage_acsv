
from sage.algebras.weyl_algebra import DifferentialWeylAlgebra
from sage.arith.srange import srange
from sage.functions.log import log, exp
from sage.functions.other import factorial
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.modules.free_module_element import vector
from sage.rings.imaginary_unit import I
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.qqbar import AA, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.ring import SR

from sage_acsv.helpers import (
    ACSVException,
    compute_newton_series,
    compute_newton_series_general,
    compute_hessian,
    compute_implicit_hessian,
    transverse_leading_normalization,
)

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

        sage: from sage_acsv.asymptotics.general_terms import _general_term_asymptotics
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


def _general_term_asymptotics_complete_intersection_hyperplane(G, Hs, exps, r, vs, cp, expansion_precision):
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

        sage: from sage_acsv.asymptotics.general_terms import _general_term_asymptotics_complete_intersection_hyperplane
        sage: R.<x, y> = QQ[]
        sage: _general_term_asymptotics_complete_intersection_hyperplane(1, [3-2*x-y, 3-x-2*y], [2, 3], [1, 1], [x, y], [1, 1], 2)
        [1/162, 0]
        sage: _general_term_asymptotics_complete_intersection_hyperplane(1, [3-2*x-y, 3-x-2*y], [2, 3], [1, 1], [x, y], [1, 1], 5)
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

        sage: from sage_acsv.asymptotics.general_terms import _general_term_asymptotics_smooth
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