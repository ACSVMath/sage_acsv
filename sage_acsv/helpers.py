from enum import Enum


from sage.all import QQ, AA, RIF, CIF, PolynomialRing, RealIntervalField, ComplexIntervalField
from sage.all import ceil, gcd, matrix, randint, sqrt, log, factorial


class OutputFormat(Enum):
    """Output options for displaying the asymptotic behavior determined
    by :func:`.diagonal_asy`.

    See also:

    - :func:`.diagonal_asy`
    """
    ASYMPTOTIC = "asymptotic"
    SYMBOLIC = "symbolic"
    TUPLE = "tuple"
    


def RationalFunctionReduce(G, H):
    r"""Reduction of G and H by dividing out their GCD.

    INPUT:

    * ``G``, ``H`` -- polynomials

    OUTPUT:

    A tuple ``(G/d, H/d)``, where ``d`` is the GCD of ``G`` and ``H``.
    """
    g = gcd(G, H)
    return G/g, H/g


def GenerateLinearForm(system, vsT, u_, linear_form=None):
    r"""Generate a linear form of the input system.

    This is an integer linear combination of the variables that
    take unique values on the solutions of the system. The
    generated linear form is with high probability separating.

    INPUT:

    * ``system`` -- A polynomial system of equations
    * ``vsT`` -- A list of variables in the system
    * ``u_`` -- A variable not in the system
    * ``linear_form`` -- (Optional) A precomputed linear form in the
      variables of the system. If passed, the returned form is
      based on the given linear form and not randomly generated.

    OUTPUT:

    A linear form.
    """
    if linear_form is not None:
        return u_ - linear_form

    maxcoeff = ceil(max([
        max([abs(x) for x in f.coefficients()]) for f in system
    ]))
    maxdegree = max([f.degree() for f in system])
    return u_ - sum([
        randint(-maxcoeff*maxdegree-31, maxcoeff*maxdegree+31)*z
        for z in vsT
    ])


def DetHessianWithLog(H, vs, r):
    r"""Computes the determinant of `z_d H_{z_d} Hess`, where `Hess` is
    the Hessian of a given map.

    The map underlying `Hess` is defined as
    `(z_1, \ldots, z_{d-1}) \mapsto z_1 \cdots z_{d-1} \log(g(z_1, \ldots, z_{d-1}))`,
    with `g` defined from IFT via `H(z_a1,...,z_{d-1},g(z_1,...,z_{d-1}))` at
    a critical point in direction `r`.

    INPUT:

    * ``H`` -- a polynomail (the denominator of the rational GF `F` in ACSV)
    * ``vs`` -- list of variables ``z_1, ..., z_d``
    * ``r`` -- direction vector of length `d` with positive integers

    OUTPUT:

    The determinant as a rational function in the variables ``vs``.
    """
    z_d = vs[-1]
    d = len(vs)

    # Build d x d matrix of U[i,j] = z_i * z_j * H'_{z_i * z_j}
    U = matrix(
        [
            [
                v1 * v2 * H.derivative(v1, v2)/(z_d * H.derivative(z_d))
                for v2 in vs
            ] for v1 in vs
        ]
    )
    V = [QQ(r[k] / r[-1]) for k in range(d)]

    # Build (d-1) x (d-1) Matrix for Hessian
    Hess = [
        [
            V[i] * V[j] + U[i][j] - V[j] * U[i][-1] - V[i]*U[j][-1]
            + V[i] * V[j] * U[-1][-1]
            for j in range(d-1)
        ] for i in range(d-1)
    ]
    for i in range(d-1):
        Hess[i][i] = Hess[i][i] + V[i]

    # Return determinant
    return matrix(Hess).determinant()

def CountRootsInInterval(F, t_, start, end):
    R = PolynomialRing(AA, t_)
    F_prev = R(F.radical())
    F_cur = F_prev.derivative(t_)

    ss = [F_prev.subs({t_:start})]
    es = [F_prev.subs({t_:end})]
    while F_cur != 0:
        ss.append(F_cur.subs({t_:start}))
        es.append(F_cur.subs({t_:end}))

        _, rem = F_prev.quo_rem(F_cur)
        F_prev, F_cur = F_cur, -rem

    ss = list(filter(lambda x: x != 0, ss))
    es = list(filter(lambda x: x != 0, es))
    alt_s = len([_ for i in range(1, len(ss)) if ss[i] * ss[i-1] < 0 ])
    alt_e = len([_ for i in range(1, len(es)) if es[i] * es[i-1] < 0 ])

    return alt_s - alt_e

class ACSVException(Exception):
    def __init__(self, message, retry=False):
        super().__init__(message)
        self._message = message
        self._retry = retry

    def __str__(self):
        return self._message

    @property
    def retry(self):
        return self._retry
