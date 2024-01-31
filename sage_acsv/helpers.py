from enum import Enum


from sage.all import QQ, RIF, CIF, PolynomialRing, RealIntervalField, ComplexIntervalField
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

class IntervalOperator():
    def __init__(self, P, Qs, u_):
        self.P = P
        self.Qs = Qs
        deg = P.degree()
        degQMax = max([Q.degree() for Q in Qs])
        norm = self._getNorm(self.P)
        precision = ((deg+2)/2)*log(deg) + log(norm)*(deg-1)

        Pd = P.derivative(u_)
        precisionByQ = []
        precisionModByQ = []
        for Q in Qs:
            # Precision for Q depends on degree and height of annihilating polynomial
            # In this case, the annihilating polynomial has the same degree as P
            PdQDegree = max(Pd.degree(), Q.degree())
            PdQHeight = max(self._getHeight(Pd), self._getHeight(Q))
            h = PdQDegree + PdQHeight * deg + self._safeLog(factorial(deg + PdQDegree)) + PdQDegree + deg
            precisionByQ.append(((deg+2)/2) * self._safeLog(deg) + (deg-1) * (h * self._safeLog(sqrt(deg))) + 5)

            # Bound the degree and height of the minimal polynomial of each coordinate
            # Recall that Res(P, Pd*x-Q) is an annhilating polynomial
            degMax = deg
            heightMax = self._getHeight(P) * degQMax + self._getHeight(Q) * deg + \
                        log(factorial(deg + degQMax)) + log(2) * deg
            
            # Determine precision needed for 
            mod_bound = (degMax**2-1)*log(degMax+1) + (2 * heightMax + log(degMax+1))*(degMax-1) + \
                        2 * heightMax * degMax + 2 * degMax * log(degMax) + degMax * log(sqrt(degMax**2+1))
            precisionModByQ.append(mod_bound)

        # Number of binary digits needed in approximation of Q_j/P'
        self.coord_separation = max(precisionByQ+[precision])
        self.coord_separation = self._safeInt(self.coord_separation) + 1

        # Number of binary digits needed in approximation of Q_j/P'
        self.modulus_separation = max(precisionModByQ)
        self.modulus_separation = self._safeInt(self.modulus_separation) + 1
        
    def equals(self, x, y, separation = None, real=False):
        separation = separation or self.coord_separation
        Field = RIF if real else CIF
        prec = Field.precision()
        Field(x), Field(y)
        while prec < separation:
            if not (Field(x)-Field(y)).contains_zero():
                return False
            prec = min(prec * 2, separation)
            Field = RealIntervalField(prec) if real else ComplexIntervalField(prec)
            Field(x), Field(y)
        return True

    def modulus_equals(self, x, y):
        return self.equals(x.abs(),y.abs(),separation=self.modulus_separation, real=True)

    def _getNorm(self, F):
        return sqrt(sum([abs(x**2) for x in F.coefficients()]))
    
    def _getHeight(self, F):
        return self._safeLog(max([abs(x) for x in F.coefficients()]))
    
    def _safeLog(self, x, base=2):
        return log(max(x, 1), base)

    def _safeInt(self, x):
        try:
            return int(x)
        except:
            if type(x) == type(RIF(0)):
                return int(x.center())

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
