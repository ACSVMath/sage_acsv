"""Functions related to computing the Kronecker representation of
a system of polynomials.
"""

from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ

from sage_acsv.backends.msolve import get_parametrization


def _kronecker_representation_msolve(system, u_, vs, return_linear_form=False):
    result = get_parametrization(vs, system)
    _, nvars, _, msvars, form, param = result[1]

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

    if return_linear_form:
        linear_form = sum(form[pidx[i]] * vs[i] for i in range(len(vs)))
        if nvars != len(vs):
            # msolve introduces an auxiliary variable when no input variable
            # separates the solutions. Its form defines a homogeneous equation,
            # so solve that equation for the auxiliary parameter.
            auxiliary_indices = [i for i in range(nvars) if i not in pidx]
            linear_form = -linear_form / form[auxiliary_indices[0]]
        return P, Qs, linear_form
    return P, Qs