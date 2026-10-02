from sage.arith.misc import binomial
from sage.functions.generalized import kronecker_delta
from sage.functions.other import factorial
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.modules.free_module_element import vector
from sage.rings.big_oh import O
from sage.rings.fraction_field import FractionField
from sage.rings.ideal import Ideal
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.qqbar import AA, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.ring import SR
from sage_acsv.helpers.exceptions import ACSVException

def compute_implicit_hessian(Hs, vs, r, subs):
    r"""Compute the Hessian of an implicitly defined function.

    Given a transverse intersection point `w` in `H_1(w),\dots,H_s(w)=0`, we can parametrize `V(H_1,\dots,H_s)`
    near `w` by writing `z_{d-s+j} = g_j(z_1,\dots,z_{d-s})`.

    Let `h(\theta_1,\dots,\theta_{d-s}) = \sum_{j=1}^s r_{d-s+j}\log g_j({w_1 \exp(i\theta_1) \dots w_{d-s} \exp(i\theta_{d-s})})`.
    This function returns the Hessian of `h`.

    INPUT:

    * ``Hs`` -- A list of polynomials `H`
    * ``vs`` -- A list of variables in the equation
    * ``r`` -- A direction vector
    * ``subs`` -- a dictionary ``{v_i: w_i}`` defining the point w

    OUTPUT:

    The Hessian of the implicitly defined function `h` defined above.

    EXAMPLES::

        sage: from sage_acsv.helpers import compute_implicit_hessian
        sage: R.<x,y,z,w> = PolynomialRing(QQ,4)
        sage: Hs = [
        ....:     z^2+z*w+x*y-4,
        ....:     w^3+z*x-y
        ....: ]
        sage: compute_implicit_hessian(Hs, [x,y,z,w], [1,1,1,1], {x:1,y:1,z:1,w:1})
        [21/32     0]
        [    0   7/8]
    """

    d = len(vs)
    s = len(Hs)
    Hs, vs = [SR(H) for H in Hs], [SR(v) for v in vs]
    if subs:
        subs = {SR(v): val for v, val in subs.items()}
    dHdg = matrix([[H.derivative(v) for v in vs[d - s:]] for H in Hs])
    dHdv = matrix([[H.derivative(v) for v in vs[:d - s]] for H in Hs])
    dgdv = -dHdg.inverse() * dHdv

    d2gdv2 = [
        [
            -dHdg.inverse()
            * (
                vector([H.derivative(vs[i]).derivative(vs[j]) for H in Hs])
                + matrix(
                    [
                        [H.derivative(vs[i]).derivative(g) for g in vs[d - s:]]
                        for H in Hs
                    ]
                )
                * dgdv.column(j)
                + matrix(
                    [
                        [H.derivative(g).derivative(vs[j]) for g in vs[d - s:]]
                        for H in Hs
                    ]
                )
                * dgdv.column(i)
                + vector(
                    [
                        matrix(
                            [
                                [H.derivative(g1).derivative(g2) for g1 in vs[d - s:]]
                                for g2 in vs[d - s:]
                            ]
                        )
                        * dgdv.column(i)
                        * dgdv.column(j)
                        for H in Hs
                    ]
                )
            )
            for i in range(d - s)
        ]
        for j in range(d - s)
    ]

    Hess = matrix(
        [
            [
                sum(
                    r[k]
                    * (
                        -vs[i] * vs[j] * d2gdv2[i][j][k - (d - s)] * vs[k]
                        - kronecker_delta(i, j) * dgdv[k - (d - s), j] * vs[k] * vs[i]
                        + vs[i] * vs[j] * dgdv[k - (d - s), i] * dgdv[k - (d - s), j]
                    )
                    / vs[k] ** 2
                    for k in range(d - s, d)
                )
                for i in range(d - s)
            ]
            for j in range(d - s)
        ]
    )
    if subs:
        return Hess.subs(subs)

    return Hess


def compute_square_root_determinant_of_hessian(hessian):
    r"""Square root of the determinant of a complex Hessian defined as the product of
    the principle branch square root of all its eigenvalues.

    INPUT:

    * ``hessian`` -- a matrix `M` of elements in ``QQbar``

    OUTPUT:

    The square root of ``hessian`` on the branch described above.

    EXAMPLES:

    For real positive definite Hessians (and any matrix of dimension at most
    two) the principal square root is returned::

        sage: from sage_acsv.helpers import compute_square_root_determinant_of_hessian
        sage: compute_square_root_determinant_of_hessian(matrix(QQbar, [[2, 1], [1, 2]]))
        1.732050807568878?

    In dimension three and higher the principal square root of the assembled
    scalar can lie on the wrong branch. For `M = i I_3` we have `1/\det(M) = i`, whose
    principal square root `e^{i\pi/4}` differs by a sign from the
    continuation value `(i^{-1/2})^3 = e^{-3i\pi/4}`::

        sage: M = QQbar(I)*matrix.identity(QQbar, 3)
        sage: compute_square_root_determinant_of_hessian(M)
        -0.7071067811865475? + 0.7071067811865475?*I
        sage: compute_square_root_determinant_of_hessian(M) == -QQbar(M.determinant()).sqrt()
        True
    """

    # Compute naive square root first for simplicity (pretty printing)
    sqrt_det = hessian.determinant().sqrt()
    if sqrt_det == 0:
        raise ACSVException("Hessian is not full rank. We cannot handle this case.")

    eigenvalues_with_multiplicity = matrix(QQbar, hessian).charpoly().roots(QQbar)

    # The sqrt function in Sage comptues the principle branch by default
    sqrt_prod = prod(ev.sqrt() ** m for (ev, m) in eigenvalues_with_multiplicity)

    if sqrt_det == sqrt_prod:
        return sqrt_det
    elif sqrt_det == -sqrt_prod:
        return -sqrt_det
    else:
        raise ACSVException("Determinant of Hessian does not match product of eigenvalues. This should never happen")


def transverse_leading_normalization(Hs, vs, r, cp):
    r"""Phase-correct normalization of the leading amplitude at a transverse point.

    INPUT:

    * ``Hs`` -- list of the local factors (polynomials in ``vs``) vanishing at ``cp``
    * ``vs`` -- list of variables, ordered so that the first `d-s` parametrize
    * ``r`` -- the direction vector
    * ``cp`` -- the contributing point, in the same order as ``vs``

    OUTPUT:

    An algebraic number, real positive when ``cp`` has positive real coordinates.

    EXAMPLES:

    At a positive real point the normalization is a positive rational::

        sage: from sage_acsv.helpers import transverse_leading_normalization
        sage: R.<x, y, z> = QQ[]
        sage: transverse_leading_normalization([1-x-2*y-z, 1-2*x-y-z], [x, y, z], [1, 1, 1], (2/9, 2/9, 1/3))
        2/27

    Here we have the two smooth critical points `(\pm 1/\sqrt{3}, 3/2, 3/2)` of
    `1/((1 - y(1+x^2)/2)(1 - z(1+x^2)/2))`. Their contributions double at even 
    `n` and cancel at odd `n` matching the exact diagonal `2^{-2n}\binom{2n}{n/2}`::

        sage: from sage_acsv.helpers import transverse_leading_normalization
        sage: R.<x, y, z> = QQ[]
        sage: Hs = [1 - y*(1 + x^2)/2, 1 - z*(1 + x^2)/2]
        sage: x0 = QQbar(sqrt(1/3))
        sage: transverse_leading_normalization(Hs, [x, y, z], [1, 1, 1], (x0, 3/2, 3/2))
        1
        sage: transverse_leading_normalization(Hs, [x, y, z], [1, 1, 1], (-x0, 3/2, 3/2))
        1

    At contributing points with complex coordinates the normalization
    carries a phase::

        sage: from sage_acsv.helpers import transverse_leading_normalization
        sage: R.<x, y> = QQ[]
        sage: H = (1 - x - y + 3*x*y)^2 + (x*y)^2
        sage: T.<t> = QQ[]
        sage: rts = (10*t^4 - 12*t^3 + 10*t^2 - 4*t + 1).roots(QQbar, multiplicities=False)
        sage: p = sorted(sorted(rts, key=lambda rt: abs(rt))[:2], key=lambda rt: CDF(rt).imag())[0]
        sage: transverse_leading_normalization([H], [x, y], [1, 1], (p, p))
        0.3492177218735412? + 0.1617762850074898?*I

    Here two smooth sheets in disjoint variable pairs cross
    transversely along a two-dimensional stratum::

        sage: from sage_acsv.helpers import transverse_leading_normalization
        sage: R.<x1, x2, y1, y2> = QQ[]
        sage: blk1 = (1 - x1 - y1 + 3*x1*y1)^2 + (x1*y1)^2
        sage: blk2 = (1 - x2 - y2 + 3*x2*y2)^2 + (x2*y2)^2
        sage: T.<t> = QQ[]
        sage: rts = (10*t^4 - 12*t^3 + 10*t^2 - 4*t + 1).roots(QQbar, multiplicities=False)
        sage: p = sorted(sorted(rts, key=lambda rt: abs(rt))[:2], key=lambda rt: CDF(rt).imag())[0]
        sage: transverse_leading_normalization([blk1, blk2], [x1, x2, y1, y2], [1, 1, 1, 1], (p, p, p, p))
        0.09578145087972136? + 0.11299029140696055?*I
    """
    d = len(vs)
    s = len(Hs)
    R, vs = PolynomialRing(QQbar, vs).objgens()
    Hs = [R(SR(Hi)) for Hi in Hs]
    subs_dict = {v:p for v, p in zip(vs, cp)}
    taus = []
    vrows = []
    for Hi in Hs:
        g = [(v * Hi.derivative(v)).subs(subs_dict) for v in vs]
        j0 = max(range(d), key=lambda j: abs(g[j]))
        tau = QQbar(g[j0])
        row = []
        for gi in g:
            c = QQbar(gi) / tau
            if not c.imag().is_zero():
                raise ACSVException(
                    "Log-gradient at contributing point is not a complex "
                    "multiple of a real vector."
                )
            row.append(AA(c.real()))
        taus.append(tau)
        vrows.append(row)
    V = matrix(AA, vrows)
    rvec = vector([AA(SR(ri)) for ri in r])
    a = (V * V.transpose()).solve_right(V * rvec)
    if V.transpose() * a != rvec:
        raise ACSVException(
            "Direction r is not in the span of the log-normal directions."
        )
    for i in range(s):
        if a[i] < 0:
            taus[i] = -taus[i]
            vrows[i] = [-c for c in vrows[i]]
        elif a[i] == 0:
            raise ACSVException("Point is not contributing: some a_i vanishes.")
    Gamma = matrix(
        AA,
        vrows + [[1 if j == k else 0 for j in range(d)] for k in range(d - s)],
    )
    return prod(-tau for tau in taus) * Gamma.determinant().abs()


def is_contributing(vs, pt, r, factors, c, allow_boundary=False) -> bool:
    r"""Determines if a minimal critical point ``pt`` such that the singular
    variety has transverse square-free factorization
    is contributing; that is, whether `r` is in the interior
    of the scaled log-normal cone of ``factors`` at ``pt``

    INPUT:

    * ``vs`` -- A list of variables
    * ``pt`` -- A point
    * ``r`` -- A direction vector
    * ``factors`` -- A list of factors of `H` for which `pt` vanishes
    * ``c`` -- The co-dimension of the intersection of factors
    * ``allow_boundary`` -- Whether or not to allow 'non-generic' directions, where the critical
      point lies on the boundary of the normal cone.

    OUTPUT:

    ``True`` or ``False`` verifying if ``vs`` is contributing

    EXAMPLES::

        sage: from sage_acsv.helpers import is_contributing
        sage: R.<x,y> = PolynomialRing(QQ, 2)
        sage: is_contributing([x, y], [1, 1], [17/24, 7/24], [1-(2*x+y)/3,1-(3*x+y)/4], 2)
        True
        sage: is_contributing([x, y], [1, 1], [1, 1], [1-(2*x+y)/3,1-(3*x+y)/4], 2)
        False

    """
    critical_subs = {v: point for v, point in zip(vs, pt)}
    # Compute irreducible components of H that contain the point
    vanishing_factors = list(f for f in factors if f.subs(critical_subs) == 0)
    for f in vanishing_factors:
        if all(f.derivative(v).subs(critical_subs) == 0 for v in vs):
            raise ACSVException(
                f"Critical point {pt} lies in non-smooth part of component {f}"
            )

    vkjs = []
    for f in vanishing_factors:
        for v in vs:
            if f.derivative(v).subs(critical_subs) != 0:
                vkjs.append(v)
                break
        else:
            # In theory this shouldn't ever happen, now that we check the condition before
            raise ACSVException(
                f"All partials of component {vanishing_factors} vanish at point {pt}"
            )

    normals = matrix(
        list(
            [
                AA(
                    f.derivative(v).subs(critical_subs)
                    * critical_subs[v]
                    / (critical_subs[vkj] * f.derivative(vkj).subs(critical_subs))
                )
                for v in vs
            ]
            for vkj, f in zip(vkjs, vanishing_factors)
        )
    )

    polytope = Polyhedron(rays=normals)
    if r not in polytope:
        return False
    elif not allow_boundary and any(r in f for f in polytope.faces(c - 1)):
        # If r is in the boundary of the log normal cone, point is non-generic
        raise ACSVException(
            f"Non-generic direction detected - critical point {pt} is contained in {len(vs) - c}-dimensional stratum"
        )
    return True


def is_transverse_at_point(H, vs, pt):
    r"""Determines if point ``pt`` lies at a transverse multiple point of ``V(H)``.

    INPUT:

    * ``H`` -- A polynomial
    * ``vs`` -- A list of variables
    * ``pt`` -- A point

    OUTPUT:

    ``True`` or ``False`` verifying if ``V(H)`` is transverse at ``pt``

    EXAMPLES::

        sage: from sage_acsv.helpers import is_transverse_at_point
        sage: R.<x,y> = PolynomialRing(QQ, 2)
        sage: is_transverse_at_point(y^2-x^2-x^3-x^4, [x, y], [0, 0])
        True
        sage: is_transverse_at_point(y^2-x^3-x^2-x^4, [x, y], [1, 1]) # Not a point in V(H)
        False
        sage: is_transverse_at_point((1-x)^2*(1-y)^2+(2-x-y)*(3-2*x-y), [x, y], [1, 1])
        True
        sage: is_transverse_at_point(x^2-y^3, [x, y], [0, 0])
        False
        sage: R.<x,y,z> = PolynomialRing(QQ, 3)
        sage: is_transverse_at_point((x-y+z^3)^3 * (x+y), [x, y, z], [0, 0, 0])
        True
        sage: is_transverse_at_point(x*y-z^3, [x, y, z], [0, 0, 0])
        False

    """
    R = PolynomialRing(QQbar, len(vs), vs)
    H = R(H)
    vs = [R(v) for v in vs]

    if H.subs({v: w for v, w in zip(vs, pt)}) != 0:
        return False
    
    def hom(f):
        if f == 0:
            return 0
        f = R(f)
        return f.homogeneous_components().get(min(f.homogeneous_components().keys()))

    # Step 1: Check that the leading homogeneous part of H factors nicely
    # Shift H so pt is at the origin
    H = H.subs({v: v + w for v, w in zip(vs, pt)})
    factors, multiplicities = zip(*hom(H).factor())
    if any(f.degree() > 1 for f in factors):
        return False

    # Step 2. Compute hom of singular set at pt
    Id = Ideal([H, *H.gradient()])

    # Convert to reverse graded ring in one additional variable and homogenize
    z0 = SR.var('z0')
    R_ordered, new_vars = PolynomialRing(QQbar, [z0] + vs, order='deglex').objgens()
    z0 = new_vars[0]
    Id = Id.change_ring(R_ordered)
    Id_hom = Id.homogenize(z0)
    gb = Id_hom.groebner_basis()
    tangent_cone = Ideal([hom(f.subs({z0:1})) for f in gb]).change_ring(R)

    # Step 3: Compute the ideal generated by removing one factor from the leading hom of H
    J = Ideal([
        factors[i]**(multiplicities[i]-1) * prod(factors[j]**multiplicities[j] for j in range(len(factors)) if j != i)
        for i in range(len(factors))
    ])

    return tangent_cone == J


def compute_hessian(H, variables, r, critical_point=None):
    r"""Computes the Hessian of an implicitly defined function.

    The computed matrix is the Hessian of the map

    .. math::

        (t_1,...t_{d-1}) \mapsto \log(g(z_1t_1,...,z_{d-1}t_{d-1}))/g(z_1,...,z_{d-1})
        + I\cdot (r_1t_1+...+r_{d-1}t_{d-1})/r_d

    at a critical point where the partial derivative of `H` with respect to `z_d` is non-zero, and
    `g` determined implicitly by

    .. math::

        H(z_1,...,z_{d-1}, g(z_1,...,z_{d-1})) = 0.

    INPUT:

    * ``H`` -- A polynomial; the denominator of the rational generating function
      `F = G/H`.
    * ``variables`` -- A list of variables ``z_1, ..., z_d``
    * ``r`` -- The direction. A vector of length `d` with positive algebraic numbers
      (usually integers) as coordinates.
    * ``critical_point`` -- (Optional) A critical point of the map at which to evaluate
      the Hessian. If not specified, the symbolic Hessian is returned.

    OUTPUT:

    A matrix representing the specified Hessian.
    """
    z_d = variables[-1]
    d = len(variables)

    zdHz = z_d * H.derivative(z_d)
    v2dH2 = [
        [
            v1 * v2 * H.derivative(v1, v2) for v2 in variables
        ] for v1 in variables
    ]
    if critical_point:
        zdHz = zdHz.subs(critical_point)
        v2dH2 = [
            [
                f.subs(critical_point) for f in row
            ]
            for row in v2dH2
        ]
    U = matrix(
        [
            [
                f / zdHz
                for f in row
            ]
            for row in v2dH2
        ]
    )

    try:
        V = [QQ(r[k] / r[-1]) for k in range(d)]
    except (ValueError, TypeError):
        V = [AA(r[k] / r[-1]) for k in range(d)]

    # Build (d-1) x (d-1) Matrix for Hessian
    hessian = [
        [
            V[i] * V[j]
            + U[i][j]
            - V[j] * U[i][-1]
            - V[i] * U[j][-1]
            + V[i] * V[j] * U[-1][-1]
            for j in range(d - 1)
        ]
        for i in range(d - 1)
    ]
    for i in range(d - 1):
        hessian[i][i] = hessian[i][i] + V[i]

    hessian = matrix(hessian)
    return hessian