"""Functions related to computing the Kronecker representation of
a system of polynomials.
"""

from sage.arith.misc import gcd
from sage.rings.polynomial.multi_polynomial_ideal import MPolynomialIdeal
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ

from sage_acsv.helpers import ACSVException, generate_linear_form
from sage_acsv.debug import acsv_logger
from sage_acsv.backends.msolve import get_parametrization
from sage_acsv.groebner import compute_groebner_basis, compute_radical
from sage_acsv.settings import ACSVSettings


def _kronecker_representation_sage(system, u_, vs, linear_form=None):
    r"""Computes the Kronecker Representation of a system of polynomials.

    This method is intended for internal use and requires a consistent
    setup of parameters. Use the :func:`.kronecker_representation` wrapper function
    to avoid doing the setup yourself.

    INPUT:

    * ``system`` -- A system of polynomials in ``d`` variables
    * ``u_`` -- Variable not contained in the variables in system
    * ``vs`` -- Variables of the system
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions

    OUTPUT:

    A polynomial ``P`` and ``d`` polynomials ``Q1, ..., Q_d`` such
    that ``z_i = Q_i(u)/P'(u)`` for ``u`` ranging over the roots of ``P``

    EXAMPLES::

        sage: from sage_acsv.kronecker import kronecker_representation
        sage: var('x, y')
        (x, y)
        sage: kronecker_representation(
        ....:     [x**3+y**3-10, y**2-2],
        ....:     [x, y],
        ....:     linear_form=x + y
        ....: )  # indirect doctest
        (u_^6 - 6*u_^4 - 20*u_^3 + 36*u_^2 - 120*u_ + 100,
         [60*u_^3 - 72*u_^2 + 360*u_ - 600, 12*u_^4 - 72*u_^2 + 240*u_])
    """

    # Generate a linear form
    linear_form = generate_linear_form(system, vs, u_, linear_form)

    expanded_R = u_.parent()

    rabinowitsch_R = PolynomialRing(
        QQ, list(expanded_R.gens()), len(expanded_R.gens()), order="degrevlex"
    )
    u_ = rabinowitsch_R(u_)

    rabinowitsch_system = [rabinowitsch_R(f) for f in system]
    rabinowitsch_system.append(rabinowitsch_R(linear_form))

    # Compute Grobner basis for ordered system of polynomials
    ideal = MPolynomialIdeal(rabinowitsch_R, rabinowitsch_system)
    try:
        ideal = MPolynomialIdeal(rabinowitsch_R, compute_groebner_basis(ideal))
    except Exception:
        raise ACSVException(
            "Trouble computing Groebner basis. System may be too large."
        )

    if ideal.dimension() != 0:
        raise ACSVException(
            f"Ideal {ideal} is not 0-dimensional. Cannot compute Kronecker representation."
        )

    ideal = compute_radical(ideal)
    gb = ideal.transformed_basis("fglm")

    rabinowitsch_R = rabinowitsch_R.change_ring(order="lex")
    u_ = rabinowitsch_R(u_)

    Ps = [
        p
        for p in gb
        if len(p.variables()) != 0 and not any([z in vs for z in p.variables()])
    ]
    if len(Ps) != 1:
        acsv_logger.debug(
            f"Rabinowitsch system: {rabinowitsch_system}\nPs: {Ps}\nbasis: {gb}"
        )
        raise ACSVException(
            "No P polynomial found for Kronecker Representation.", retry=True
        )
    u_ = Ps[0].variables()[0]
    R = PolynomialRing(QQ, u_)
    P = R(Ps[0])
    P, _ = P.quo_rem(gcd(P, P.derivative(u_)))
    Pd = P.derivative(u_)

    # Find Q_i for each variable
    Qs = []
    for z in vs:
        z = rabinowitsch_R(z)
        eqns = [f for f in gb if z in f.variables()]
        if len(eqns) != 1:
            acsv_logger.debug(f"equations: {eqns}\nz: {z}\nvs: {vs}")
            raise ACSVException("Linear form does not separate the roots.", retry=True)

        eq = eqns[0].polynomial(z)

        if eq.degree() != 1:
            acsv_logger.debug(f"eq: {eq}\nz: {z}")
            raise ACSVException("Linear form does not separate the roots.", retry=True)
        _, rem = (Pd * eq.roots()[0][0]).quo_rem(P)
        Qs.append(rem)

    # Forget base ring, move to univariate polynomial ring in u over a field
    Qs = [R(Q) for Q in Qs]

    return P, Qs


def _kronecker_representation_msolve(system, u_, vs):
    result = get_parametrization(vs, system)
    _, nvars, _, msvars, _, param = result[1]

    # msolve may reorder the variables, so order them back
    Qparams = param[1][2]
    vsExt = [str(v) for v in vs]
    # Check if no new variable was created by msolve
    # If so, the linear_form used is just zd
    # i.e. u_ = zd and Qd = zd * P'(zd)
    if nvars == len(vs):
        Pdz = [0] + [-c for c in param[1][1][1]]
        Qparams.append([[1, Pdz], 1])
    pidx = [msvars.index(v) for v in vsExt]
    Qparams = [Qparams[i] for i in pidx]

    R = PolynomialRing(QQ, u_)
    u_ = R(u_)

    P_coeffs = param[1][0][1]
    P = sum([c * u_**i for (i, c) in enumerate(P_coeffs)])

    Qs = []
    for Q_param in Qparams:
        Q_coeffs = Q_param[0][1]
        c_div = Q_param[1]
        Q = -sum([c * u_**i for (i, c) in enumerate(Q_coeffs)]) / c_div
        Qs.append(Q)

    return P, Qs


def _kronecker_representation(system, u_, vs, linear_form=None):
    r"""Computes the Kronecker Representation of a system of polynomials

    Internal intermediate function for choosing the Kronecker backend implementation

    INPUT:

    * ``system`` -- A system of polynomials in ``d`` variables
    * ``u_`` -- Variable not contained in the variables in system
    * ``vs`` -- Variables of the system
    * ``linear_form`` -- (Optional) A linear combination of the
      input variables that separates the critical point solutions

    OUTPUT:

    A polynomial ``P`` and ``d`` polynomials ``Q1, ..., Q_d`` such that
    ``z_i = Q_i(u)/P'(u)`` for ``u`` ranging over the roots of ``P``.
    """
    if ACSVSettings.get_default_kronecker_backend() == ACSVSettings.Kronecker.MSOLVE:
        return _kronecker_representation_msolve(system, u_, vs)
    return _kronecker_representation_sage(system, u_, vs, linear_form=linear_form)

def kronecker(system, vs, linear_form=None):
    acsv_logger.warning(
        "The kronecker function has been deprecated and will "
        "be removed in a future version. Please use the "
        "kronecker_representation function (same signature) instead.",
    )
    return kronecker_representation(system, vs, linear_form=linear_form)


def kronecker_representation(system, vs, linear_form=None):
    r"""Computes the Kronecker Representation of a system of polynomials.

    INPUT:

    * ``system`` -- A system of polynomials in `d` variables defining a zero-dimensional (finite) variety
    * ``vs`` -- Variables of the system
    * ``linear_form`` -- (Optional) A linear combination of the
      input variables that separates the critical point solutions

    OUTPUT:

    A polynomial `P` and `d` polynomials `Q1, ..., Q_d` such that
    `z_i = Q_i(u)/P'(u)` for `u` ranging over the roots of `P`.

    EXAMPLES::

        sage: from sage_acsv.kronecker import kronecker_representation
        sage: var('x,y')
        (x, y)
        sage: kronecker_representation([x**3+y**3-10, y**2-2], [x,y], x+y)
        (u_^6 - 6*u_^4 - 20*u_^3 + 36*u_^2 - 120*u_ + 100,
         [60*u_^3 - 72*u_^2 + 360*u_ - 600, 12*u_^4 - 72*u_^2 + 240*u_])
    """
    R, u_ = PolynomialRing(QQ, "u_").objgen()
    R = PolynomialRing(QQ, len(vs) + 1, vs + [u_])
    system = [R(f) for f in system]
    vs = [R(v) for v in vs]
    u_ = R(u_)
    return _kronecker_representation(system, u_, vs, linear_form)
