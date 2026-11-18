"""Functions related to computing the Kronecker representation of
a system of polynomials.
"""

from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ

from sage_acsv.backends.julia import get_parametrization


def _kronecker_representation_julia(system, u_, vs, return_linear_form=False):
    rur, form = get_parametrization(vs, system)

    R = PolynomialRing(QQ, u_)
    u_ = R(u_)

    P = sum(
        [QQ(c) * u_**i for (i, c) in enumerate(rur[0])]
    )

    Qs = [
        sum([QQ(c) * u_**i for (i, c) in enumerate(q)])
        for q in rur[1:]
    ]

    if return_linear_form:
        linear_form = -sum(
            [QQ(c) * vs[i] for (i, c) in enumerate(form)]
        )
        return P, Qs, linear_form
    return P, Qs