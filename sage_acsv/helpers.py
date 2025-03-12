from enum import Enum

from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing
from sage.symbolic.operators import add_vararg

from sage.all import AA, QQ, SR, Ideal, Polyhedron, ceil, gcd, matrix, randint, vector, kronecker_delta, prod

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
        max([abs(x) for x in f.coefficients()]) for f in system if f != 0
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
    * ``r`` -- The direction. A vector of length `d` with positive algebraic numbers
      (usually integers) as coordinates.
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

    try:
        V = [QQ(r[k] / r[-1]) for k in range(d)]
    except (ValueError, TypeError):    
        V = [AA(r[k] / r[-1]) for k in range(d)]

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

def NewtonSeries(phi, variables, series_precision):
    r"""Computes the series expansion of an implicitly defined function.

    The function `g(x)` for which a series expansion is computed satisfies

    .. math::

        \Phi(x, g(x)) = 0
    
    INPUT:

    * ``phi`` -- A polynomial; the equation defining the function that is expanded.
    * ``variables`` -- A list of variables in the equation. The last variable in this
      list is the variable corresponding to `g(x)`.
    * ``series_precision`` -- A positive integer, the precision of the series expansion.

    OUTPUT:

    A series expansion of the function `g(x)`.

    EXAMPLES::

        sage: from sage_acsv.helpers import NewtonSeries
        sage: R.<x, T> = QQ[]
        sage: NewtonSeries(x*T^2 - T + 1, [x, T], 7)
        132*x^6 + 42*x^5 + 14*x^4 + 5*x^3 + 2*x^2 + x + 1

    """

    X = variables[:-1]
    Y = variables[-1]

    def ModX(F, N):
        return F.mod(Ideal(X)**N)

    def ModY(F, N):
        return F.mod(Y**N)

    def Mod(F, N):
        return ModX(ModY(F, N), N)

    def NewtonRecur(H, N):
        if N == 1:
            return 0, 1 / H.derivative(Y).subs({v: 0 for v in variables})
        F, G = NewtonRecur(H, ceil(N/2))
        G = G + (1 - G * H.derivative(Y).subs({Y: F})) * G
        F = F - G*H.subs({Y: F})
        return ModX(F, N), ModX(G, ceil(N/2))
    
    return NewtonRecur(Mod(phi, series_precision), series_precision)[0]

def ImplicitHessian(Hs, vs, r, subs):
    r"""Compute the Hessian of an implicitly defined function.
    
    Given a transverse intersection point `w` in `H_1(w),\dots,H_s(w)=0`, we can parametrize `V(H_1,\dots,H_s)`
    near `w` by writing `z_{d-s+j} = g_j(z_1,\dots,z_{d-s})`. 
    
    Let `h(\theta_1,\dots,\theta_{d-s}) = \sum_{j=1}^s r_{d-s+j}\log g_j({w_1 exp(i\theta_1) \dots w_{d-s} exp(i\theta_{d-s})})`.
    This function returns the Hessian of h.
    
    INPUT:

    * ``Hs`` -- A list of polynomials `H`
    * ``vs`` -- A list of variables in the equation
    * ``r`` -- A direction vector
    * ``subs`` -- a dic `{v_i:w_i}` for point w

    OUTPUT:

    The Hessian of the implicitly defined function `h` defined above.

    EXAMPLES::

        sage: from sage_acsv.helpers import ImplicitHessian
        sage: R.<x,y,z,w> = PolynomialRing(QQ,4)
        sage: Hs = [
        ....:     z^2+z*w+x*y-4,
        ....:     w^3+z*x-y
        ....: ]
        sage: ImplicitHessian(Hs, [x,y,z,w], [1,1,1,1], {x:1,y:1,z:1,w:1})
        [21/32     0]
        [    0   7/8]
    """

    d = len(vs)
    s = len(Hs)
    Hs, vs = [SR(H) for H in Hs], [SR(v) for v in vs]
    if subs:
        subs = {SR(v):val for v, val in subs.items()}
    dHdg = matrix(
        [
            [
                H.derivative(v) for v in vs[d-s:]
            ] for H in Hs
        ]
    )
    dHdv = matrix(
        [
            [
                H.derivative(v) for v in vs[:d-s]
            ] for H in Hs
        ]
    )
    dgdv = -dHdg.inverse() * dHdv

    d2gdv2 = [
        [ -dHdg.inverse() * (
            vector([H.derivative(vs[i]).derivative(vs[j]) for H in Hs]) + \
                matrix(
                    [
                        [
                            H.derivative(vs[i]).derivative(g) for g in vs[d-s:]
                        ] for H in Hs
                    ]
                ) * dgdv.column(j)+ \
                matrix(
                    [
                        [
                            H.derivative(g).derivative(vs[j]) for g in vs[d-s:]
                        ] for H in Hs
                    ]
                ) * dgdv.column(i) + \
                vector(
                    [
                        matrix(
                            [
                                [
                                    H.derivative(g1).derivative(g2) for g1 in vs[d-s:]
                                ]
                                for g2 in vs[d-s:]
                            ]
                        ) * dgdv.column(i) * dgdv.column(j)
                     for H in Hs
                    ]
                )
            ) for i in range(d-s)
        ] for j in range(d-s)
    ]

    Hess = matrix(
        [
            [
                sum(
                    [
                        r[k] * (
                            -vs[i] * vs[j] * d2gdv2[i][j][k-(d-s)]*vs[k] - kronecker_delta(i,j)*dgdv[k-(d-s),j]*vs[k]*vs[i] \
                                + vs[i]*vs[j]*dgdv[k-(d-s),i]*dgdv[k-(d-s),j]
                        )/vs[k]**2 for k in range(d-s, d)
                    ]
                ) for i in range(d-s)
            ] for j in range(d-s)
        ] 
    )
    if subs:
        return Hess.subs(subs)

    return Hess

def IsContributing(vs, pt, r, factors, c):
    r"""Determines if critical point `pt` is contributing; that is, `r` is in the interior
    of the scaled log-normal cone of `factors` at `pt`
    
    INPUT:

    * ``vs`` -- A list of variables
    * ``pt`` -- A point
    * ``r`` -- A direction vector
    * ``factors`` -- A list of factors of `H` for which `pt` vanishes
    * ``c`` -- The co-dimension of the intersection of factors

    OUTPUT:

    If vs is contributing

    EXAMPLES::

        sage: from sage_acsv.helpers import IsContributing
        sage: R.<x,y> = PolynomialRing(QQ, 2)
        sage: IsContributing([x,y], [1,1], [17/24, 7/24], [1-(2*x+y)/3,1-(3*x+y)/4], 2)
        True
        sage: IsContributing([x,y], [1,1], [1, 1], [1-(2*x+y)/3,1-(3*x+y)/4], 2)
        False

    """
    critical_subs = {v:point for v, point in zip(vs, pt)}
    # Compute irreducible components of H that contain the point
    vanishing_factors = list(
        f for f in factors
        if f.subs(critical_subs) == 0
    )
    for f in vanishing_factors:
        if all([f.derivative(v).subs(critical_subs)==0 for v in vs]):
            raise ACSVException(f"Critical point {pt} lies in non-smooth part of component {f}")

    vkjs = []
    for f in vanishing_factors:
        for v in vs:
            if f.derivative(v).subs(critical_subs)!=0:
                vkjs.append(v)
                break
        else:
            # In theory this shouldn't ever happen, now that we check the condition before
            raise ACSVException(f"All partials of component {vanishing_factors} vanish at point {pt}")

    normals = matrix(
        list(
            [AA(
                f.derivative(v).subs(critical_subs) * critical_subs[v] / (
                    critical_subs[vkj] * f.derivative(vkj).subs(critical_subs)
                )
            ) for v in vs] for vkj, f in zip(vkjs, vanishing_factors)
        )
    )
    
    polytope = Polyhedron(rays=normals)
    if r not in polytope:
        return False
    elif any([r in f for f in polytope.faces(c-1)]):
        # If r is in the boundary of the log normal cone, point is non-generic
        raise ACSVException(
            f"Non-generic critical point found - {pt} is contained in {len(vs)-c}-dimensional stratum"
        )
    return True

def get_coefficients(expr):
    r"""Determines coefficients for each n^k that appears in the asymptotic expression.
    
    INPUT:

    * ``expr`` -- An asymptotic expression, symbolic expression, ACSV tuple, or list of ACSV tuples

    OUTPUT:

    A dictionary `{d: [(c, e)]}` where `c*e*n^d` is a term appearing in the fully expanded expr,
    `c` is a constant, and `e` an exponential in `n`

    EXAMPLES::

        sage: from sage_acsv import diagonal_asy, get_coefficients
        sage: var('x,y')
        (x, y)
        sage: res = diagonal_asy(1/(1 - x - y), r=[1,1], expansion_precision=2)
        sage: coefs = get_coefficients(res) 
        sage: sorted(coefs.items())
        [(-3/2, [(-1/8/sqrt(pi), 4^n)]), (-1/2, [(1/sqrt(pi), 4^n)])]
        sage: res = diagonal_asy(1/(1 - x - y), r=[1,1], expansion_precision=2, output_format="tuple")
        sage: get_coefficients(res) == coefs
        True
        sage: res = diagonal_asy(1/(1 - x - y), r=[1,1], expansion_precision=2, output_format="symbolic")
        sage: get_coefficients(res) == coefs
        True

    ::

        sage: res = diagonal_asy(1/(1 - x^7))
        sage: get_coefficients(res)
        {0: [(1/7, (e^(I*pi - I*arctan(4.381286267534823?)))^n),
          (1/7, (e^(I*pi - I*arctan(0.4815746188075287?)))^n),
          (1/7, (e^(-I*pi + I*arctan(4.381286267534823?)))^n),
          (1/7, (e^(-I*pi + I*arctan(0.4815746188075287?)))^n),
          (1/7, (e^(I*arctan(1.253960337662704?)))^n),
          (1/7, (e^(-I*arctan(1.253960337662704?)))^n),
          (1, 1)]}

    """
    if isinstance(expr, tuple):
        expr = prod(expr)
    elif isinstance(expr, list):
        expr = sum([prod(tup) for tup in expr])
    elif isinstance(expr.parent(), AsymptoticRing):
        expr = SR(expr.exact_part())

    if not isinstance(expr.parent(), type(SR)):
        raise ACSVException(f"Cannot deal with expression of type {expr.parent()}")
    
    if len(expr.args()) > 1:
        raise ACSVException("Cannot process multivariate symbolic expression.")
    n = expr.args()[0]

    # If expression is the sum of a bunch of terms, handle each one separately
    expr = expr.expand()
    terms = [expr]
    if expr.operator() == add_vararg:
        terms = expr.operands()

    terms_by_degree = {}
    for term in terms:
        deg = 0
        const = 1
        exponent = 1
        for v in term.operands():
            v = SR(v)
            if n in v.args():
                if v.degree(n) != 0:
                    deg += v.degree(n)
                else:
                    exponent *= v
            else:
                const *= v
        if deg not in terms_by_degree:
            terms_by_degree[deg] = []
        
        terms_by_degree[deg].append((const, exponent))

    return terms_by_degree




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
