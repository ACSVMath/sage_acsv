

from sage.arith.functions import lcm
from sage.functions.log import exp
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import AA
from sage.rings.rational_field import QQ
from sage.symbolic.constants import pi
from sage.symbolic.ring import SR

from sage_acsv.critical_points import minimal_critical_points_combinatorial
from sage_acsv.debug import acsv_logger
from sage_acsv.helpers import (
    ACSVException,
    rational_function_reduce,
    compute_hessian,
    compute_square_root_determinant_of_hessian,
)
from sage_acsv.settings import ACSVSettings
from sage_acsv.helpers.utils import (
    _prepare_symbolic_fraction, 
    _dict_to_variable_order
)


def central_limit_theorem_combinatorial(F, main_var, as_symbolic=False, r=None):
    r"""Take a multivariate rational generating function, check if it admits a
    minimal critical point of a form implying a local central limit theorem, and
    (if so) return the local central limit theorem.

    INPUT:

    * ``F`` -- The rational function ``G/H`` in ``d`` variables. This function is
        assumed to have a combinatorial expansion.
    * ``main_var`` -- The variable that marks the ``size`` of the objects (so that the limit
        theorem holds as the exponent of ``main_var`` goes to infinity).
    * ``as_symbolic`` -- If ``True``, returns the limit theorem as an expression from the symbolic
        ring ``SR`` in the variable ``n``. If ``False``, the default, returns a tuple
        (a, n^b, pi^b, C, D, v) such that the local central limit theorem is specified by the
        function ``f(s) = a^n * n^b * pi^b * C * exp(-((s-n*v)*D*(s-n*v).transpose())/2/n)``
    * ``r`` -- (Optional) A dictionary specifying the variables in the direction vector. If not provided, the direction is
        determined by ``H.variables()``.

    OUTPUT:

    A representation of the local central limit theorem, either as a list of tuples,
    or as a symbolic expression.

    EXAMPLES::

        sage: from sage_acsv import central_limit_theorem_combinatorial
        sage: var('z t')
        (z, t)
        sage: central_limit_theorem_combinatorial(1/(1-t-z*t^2), t, as_symbolic = True)
        1.710862642974252?*(1/2*sqrt(5) + 1/2)^n*e^(-1/2*(-0.2763932022500211?*n + s0)*(-3.090169943749475?*n + 11.18033988749895?*s0)/n)/(sqrt(pi)*sqrt(n))
    """

    if isinstance(r, dict):
        r = _dict_to_variable_order(F, r)

    G, H, variable_map = _prepare_symbolic_fraction(F)

    main_var = variable_map[main_var]
    zvariables = [v for v in variable_map.values() if v != main_var]

    R = PolynomialRing(QQ, zvariables + [main_var])
    vs = R.gens()

    # Make sure G and H are coprime, and that H does not vanish at 0
    G, H = rational_function_reduce(G, H)
    G, H = R(G), R(H)
    if H.subs({v: 0 for v in vs}) == 0:
        raise ValueError("Denominator vanishes at 0.")

    # Find rho
    P = H.subs({v: 1 for v in vs[0:-1]})
    rts = [rt for rt in QQ[vs[-1]](P).roots(AA, multiplicities=False) if rt > 0]
    if not rts:
        raise ValueError("H(1,rho)=0 has no positive solution.")
    rho = min(rts)
    sbs = {v: 1 for v in vs[0:-1]} | {vs[-1]: rho}

    # Check numerator and denominator requirements are met
    if H.derivative(vs[-1]).subs(sbs) == 0:
        raise ValueError("The partial derivative of the denominator at (1, rho) is 0.")

    if G.subs(sbs) == 0:
        raise ValueError("The numerator at (1, rho) is 0.")

    # Get direction for LCLT
    m = [H.derivative(v).subs(sbs) / (rho * H.derivative(vs[-1]).subs(sbs)) for v in vs[0:-1]] + [ZZ.one()]

    # Determine direction type and set r accordingly
    if max([k.degree() for k in m[0:-1]]) == 1:
        multiple = lcm([QQ(k).denom() for k in m])
        r = [ZZ(multiple * k) for k in m]
    else:
        r = m

    q, lambda_, u_ = PolynomialRing(QQ, 'q, lambda_, u_').gens()
    expanded_R = PolynomialRing(QQ, len(vs) + 3, vs + (q, lambda_, u_))

    vs = [expanded_R(v) for v in vs]
    q, lambda_, u_ = expanded_R(q), expanded_R(lambda_), expanded_R(u_)
    vsT = vs + [q, lambda_]
    G, H = expanded_R(G), expanded_R(H)

    d = len(vs)

    # Reorder direction to match F's variable ordering for minimal_critical_points_combinatorial
    original_vars = [variable_map[v] for v in F.denominator().variables()]
    var_to_r_idx = {v: i for i, v in enumerate(zvariables + [main_var])}
    r_reordered = [r[var_to_r_idx[v]] for v in original_vars]
    expected_point = [rho if v == main_var else 1 for v in original_vars]

    for _ in range(ACSVSettings.MAX_MIN_CRIT_RETRIES):
        try:
            # Find minimal critical points
            min_crit_pts = minimal_critical_points_combinatorial(F, r=r_reordered)

            if len(min_crit_pts) != 1 or min_crit_pts != [expected_point]:
                raise ValueError("The point (1,rho) is not the only critical point with this coordinate-wise modulus.")
            break
        except Exception as e:
            # In case form doesn't separate, we want to try again
            if isinstance(e, ACSVException) and e.retry:
                acsv_logger.warning(
                    "Randomly generated linear form was not suitable, "
                    f"encountered error: {e}\nRetrying..."
                )
                continue
            else:
                raise e

    sbs = {v: 1 for v in vsT[0:-3]} | {vsT[-3]: rho}
    Hess = compute_hessian(H, vsT[0:-2], r)
    Hess = Hess.subs({v: 1 for v in Hess.base_ring().gens()[0:-4]} | {Hess.base_ring().gens()[-4]: rho})

    if Hess.determinant() == 0:
        raise ValueError("Hessian determinant is 0.")

    # Values appearing in asymptotics
    base = 1 / rho
    constant = - AA(G.subs(sbs) / rho / H.derivative(vs[-1]).subs(sbs) / (ZZ(2)**(d - 1)).sqrt() / compute_square_root_determinant_of_hessian(Hess))
    exponent = (1 - d) / 2

    s = matrix((SR.var('s', n=d - 1)))
    invHess = Hess.inverse()

    n = SR.var('n')
    result = (base, n**exponent, pi**exponent, constant, invHess, matrix(m[:-1]))

    if as_symbolic:
        a, b, c, d, e, f = result

        if a.degree() <= 2:
            a = base.radical_expression()
        if d.degree() <= 2:
            d = constant.radical_expression()

        sfactor = exp(-(((s - n * f) * e * (s - n * f).transpose())[0, 0]) / 2 / n)

        result = a**n * b * c * d * sfactor

    return result
