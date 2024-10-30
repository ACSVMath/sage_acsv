from enum import Enum

from sage.all import QQ, Ideal, ceil, gcd, matrix, randint


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

def GetHessian(H, variables, r, critical_point=None):
    r"""Computes the Hessian of a given map.

    The map underlying `Hess` is defined as

    .. math::

        (z_1, \ldots, z_{d-1}) \mapsto z_1 \cdots z_{d-1} \log(g(z_1, \ldots, z_{d-1})),
    
    with `g` determined via the Implicit Function Theorem for the equation
    
    .. math::
    
        H(z_1,...,z_{d-1}, g(z_1,...,z_{d-1})) = 0`
    
    at a critical point in direction `r`.

    INPUT:

    * ``H`` -- A polynomial; the denominator of the rational generating function
      `F = G/H`.
    * ``vs`` -- A list of variables ``z_1, ..., z_d``
    * ``r`` -- The direction. A vector of length `d` with positive integer coordinates.
    * ``critical_point`` -- An optional critical point where the Hessian should be evaluated at.
      If not specified, the symbolic Hessian is returned.

    OUTPUT:

    A matrix representing the Hessian of the given map.
    """
    z_d = variables[-1]
    d = len(variables)

    # Build d x d matrix of U[i,j] = z_i * z_j * H'_{z_i * z_j}
    U = matrix(
        [
            [
                v1 * v2 * H.derivative(v1, v2)/(z_d * H.derivative(z_d))
                for v2 in variables
            ] for v1 in variables
        ]
    )
    V = [QQ(r[k] / r[-1]) for k in range(d)]

    # Build (d-1) x (d-1) Matrix for Hessian
    hessian = [
        [
            V[i] * V[j] + U[i][j] - V[j] * U[i][-1] - V[i]*U[j][-1]
            + V[i] * V[j] * U[-1][-1]
            for j in range(d-1)
        ] for i in range(d-1)
    ]
    for i in range(d-1):
        hessian[i][i] = hessian[i][i] + V[i]

    hessian = matrix(hessian)
    if critical_point is not None:
        hessian = hessian.subs(critical_point)
    return hessian

    r"""Computes the determinant of `z_d H_{z_d} Hess`, where `Hess` is
    the Hessian of a given map.

    The map underlying `Hess` is defined as
    `(z_1, \ldots, z_{d-1}) \mapsto z_1 \cdots z_{d-1} \log(g(z_1, \ldots, z_{d-1}))`,
    with `g` defined from IFT via `H(z_a1,...,z_{d-1},g(z_1,...,z_{d-1}))` at
    a critical point in direction `r`.

    INPUT:

    * ``H`` -- a polynomial (the denominator of the rational GF `F` in ACSV)
    * ``vs`` -- list of variables ``z_1, ..., z_d``
    * ``r`` -- direction vector of length `d` with positive integers

    OUTPUT:

    The determinant as a rational function in the variables ``vs``.
    """
    # Return determinant
    Hess = GetHessian(H, vs, r)
    return matrix(Hess).determinant()

def NewtonSeries(Phi, vs, N):
    """
    Computes series expansion approximation of g defined implicitly by Phi(x,g(x)) = 0
    """

    X = vs[:-1]
    Y = vs[-1]

    def ModX(F, N):
        return F.mod(Ideal(X)**N)

    def ModY(F, N):
        return F.mod(Y**N)

    def Mod(F, N):
        return ModX(ModY(F, N), N)

    def NewtonRecur(H, N):
        if N == 1:
            return 0, 1/H.derivative(Y).subs({v:0 for v in vs})
        F, G = NewtonRecur(H, ceil(N/2))
        G = G + (1-G*H.derivative(Y).subs({Y:F}))*G
        F = F - G*H.subs({Y:F})
        return ModX(F, N), ModX(G, ceil(N/2))
    
    return NewtonRecur(Mod(Phi, N), N)[0]

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
