"""Miscellaneous mathematical helper functions."""

from __future__ import annotations

from dataclasses import dataclass

from sage.arith.misc import gcd
from sage.functions.generalized import kronecker_delta
from sage.functions.other import ceil
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.groups.misc_gps.argument_groups import ArgumentByElementGroup
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.misc.prandom import randint
from sage.modules.free_module_element import vector
from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing, AsymptoticExpansion
from sage.rings.asymptotic.growth_group import (
    ExponentialGrowthGroup,
    MonomialGrowthGroup,
)
from sage.rings.ideal import Ideal
from sage.rings.qqbar import AA, AlgebraicNumber, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.expression import Expression
from sage.symbolic.ring import SymbolicRing, SR
from sage.symbolic.operators import add_vararg
from sage.symbolic.constants import pi


@dataclass
class Term:
    r"""A dataclass for storing the decomposed terms of an asymptotic expression.

    INPUT:

    * ``coefficient`` -- The coefficient of the term
    * ``pi_factor`` -- The factor of pi in the term
    * ``base`` -- The base of the term
    * ``power`` -- The power of the term

    OUTPUT:

    A dataclass with the given attributes.

    EXAMPLES::

        sage: from sage_acsv.helpers import Term
        sage: Term(1, 1/sqrt(pi), 4, -1/2)
        Term(coefficient=1, pi_factor=1/sqrt(pi), base=4, power=-1/2)
    """

    coefficient: Expression | AlgebraicNumber
    pi_factor: Expression
    base: Expression | AlgebraicNumber
    power: Expression | AlgebraicNumber

    def __lt__(self, other):
        return (self.base, self.power) < (other.base, other.power)


def collapse_zero_part(algebraic_number: AlgebraicNumber) -> AlgebraicNumber:
    if algebraic_number.real().is_zero():
        algebraic_number = QQbar(algebraic_number.imag()) * QQbar(-1).sqrt()
    if algebraic_number.imag().is_zero():
        algebraic_number = QQbar(algebraic_number.real())
    algebraic_number.simplify()
    return algebraic_number


def rational_function_reduce(G, H):
    r"""Reduction of the rational function `G/H` by dividing `G` and `H` by their GCD.

    INPUT:

    * ``G``, ``H`` -- polynomials

    OUTPUT:

    A tuple ``(G/g, H/g)``, where ``g`` is the GCD of ``G`` and ``H``.
    """
    g = gcd(G, H)
    return G / g, H / g


def generate_linear_form(system, vsT, u_, linear_form=None):
    r"""Generate a linear form for the input system.

    This is an integer linear combination of the variables that,
    with high probability, takes unique values on the solutions of 
    the system. 

    INPUT:

    * ``system`` -- A polynomial system of equations
    * ``vsT`` -- A list of variables in the system
    * ``u_`` -- A variable not in the system
    * ``linear_form`` -- (Optional) A precomputed linear form in the
      variables of the system. If passed, the returned form is
      based on the given linear form and not randomly generated.

    OUTPUT:

    A linear form in ``u_`` and the variables of ``vsT``.
    """
    if linear_form is not None:
        return u_ - linear_form

    maxcoeff = ceil(
        max([max([abs(x) for x in f.coefficients()]) for f in system if f != 0])
    )
    maxdegree = max([f.degree() for f in system])
    return u_ - sum(
        [
            randint(-maxcoeff * maxdegree - 31, maxcoeff * maxdegree + 31) * z
            for z in vsT
        ]
    )


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
    * ``vs`` -- A list of variables ``z_1, ..., z_d``
    * ``r`` -- The direction. A vector of length `d` with positive algebraic numbers
      (usually integers) as coordinates.
    * ``critical_point`` -- (Optional) A critical point of the map at which to evaluate
      the Hessian. If not specified, the symbolic Hessian is returned.

    OUTPUT:

    A matrix representing the specified Hessian.
    """
    z_d = variables[-1]
    d = len(variables)

    # Build d x d matrix of U[i,j] = z_i * z_j * H'_{z_i * z_j}
    U = matrix(
        [
            [
                v1 * v2 * H.derivative(v1, v2) / (z_d * H.derivative(z_d))
                for v2 in variables
            ]
            for v1 in variables
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
    if critical_point is not None:
        hessian = hessian.subs(critical_point)
    return hessian


def compute_newton_series(phi, variables, series_precision):
    r"""Computes the series expansion of an implicitly defined function.

    The function `g(x)` for which a series expansion is computed is a simple root of the expression

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

        sage: from sage_acsv.helpers import compute_newton_series
        sage: R.<x, T> = QQ[]
        sage: compute_newton_series(x*T^2 - T + 1, [x, T], 7)
        132*x^6 + 42*x^5 + 14*x^4 + 5*x^3 + 2*x^2 + x + 1

    """

    X = variables[:-1]
    Y = variables[-1]

    def ModX(F, N):
        return F.mod(Ideal(X) ** N)

    def ModY(F, N):
        return F.mod(Y**N)

    def Mod(F, N):
        return ModX(ModY(F, N), N)

    def NewtonRecur(H, N):
        if N == 1:
            return 0, 1 / H.derivative(Y).subs({v: 0 for v in variables})
        F, G = NewtonRecur(H, ceil(N / 2))
        G = G + (1 - G * H.derivative(Y).subs({Y: F})) * G
        F = F - G * H.subs({Y: F})
        return ModX(F, N), ModX(G, ceil(N / 2))

    return NewtonRecur(Mod(phi, series_precision), series_precision)[0]


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
    dHdg = matrix([[H.derivative(v) for v in vs[d - s :]] for H in Hs])
    dHdv = matrix([[H.derivative(v) for v in vs[: d - s]] for H in Hs])
    dgdv = -dHdg.inverse() * dHdv

    d2gdv2 = [
        [
            -dHdg.inverse()
            * (
                vector([H.derivative(vs[i]).derivative(vs[j]) for H in Hs])
                + matrix(
                    [
                        [H.derivative(vs[i]).derivative(g) for g in vs[d - s :]]
                        for H in Hs
                    ]
                )
                * dgdv.column(j)
                + matrix(
                    [
                        [H.derivative(g).derivative(vs[j]) for g in vs[d - s :]]
                        for H in Hs
                    ]
                )
                * dgdv.column(i)
                + vector(
                    [
                        matrix(
                            [
                                [H.derivative(g1).derivative(g2) for g1 in vs[d - s :]]
                                for g2 in vs[d - s :]
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


def is_contributing(vs, pt, r, factors, c):
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

    OUTPUT:

    ``True`` or ``False`` verifying if ``vs`` is contributing

    EXAMPLES::

        sage: from sage_acsv.helpers import is_contributing
        sage: R.<x,y> = PolynomialRing(QQ, 2)
        sage: is_contributing([x,y], [1,1], [17/24, 7/24], [1-(2*x+y)/3,1-(3*x+y)/4], 2)
        True
        sage: is_contributing([x,y], [1,1], [1, 1], [1-(2*x+y)/3,1-(3*x+y)/4], 2)
        False

    """
    critical_subs = {v: point for v, point in zip(vs, pt)}
    # Compute irreducible components of H that contain the point
    vanishing_factors = list(f for f in factors if f.subs(critical_subs) == 0)
    for f in vanishing_factors:
        if all([f.derivative(v).subs(critical_subs) == 0 for v in vs]):
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
    elif any([r in f for f in polytope.faces(c - 1)]):
        # If r is in the boundary of the log normal cone, point is non-generic
        raise ACSVException(
            f"Non-generic direction detected - critical point {pt} is contained in {len(vs) - c}-dimensional stratum"
        )
    return True


def get_expansion_terms(
    expr: tuple | list[tuple] | Expression | AsymptoticExpansion,
) -> list[Term]:
    r"""Determines coefficients for each n^k that appears in the asymptotic expressions
    returned by :func:`.diagonal_asymptotics_combinatorial`.

    INPUT:

    * ``expr`` -- An asymptotic expression, symbolic expression, ACSV tuple, or list of ACSV tuples

    OUTPUT:

    A list of :class:`.Term` objects (with attributes ``coefficient``, ``pi_factor``,
    ``base`` and ``power``), each representing a summand in the fully expanded expression.

    EXAMPLES::

        sage: from sage_acsv import diagonal_asymptotics_combinatorial, get_expansion_terms
        sage: var('x y z')
        (x, y, z)
        sage: res = diagonal_asymptotics_combinatorial(1/(1 - x - y), r=[1,1], expansion_precision=2)
        sage: coefs = sorted(get_expansion_terms(res), reverse=True)
        sage: coefs
        [Term(coefficient=1, pi_factor=1/sqrt(pi), base=4, power=-1/2),
         Term(coefficient=-1/8, pi_factor=1/sqrt(pi), base=4, power=-3/2)]
        sage: res = diagonal_asymptotics_combinatorial(1/(1 - x - y), r=[1,1], expansion_precision=2, output_format="tuple")
        sage: sorted(get_expansion_terms(res)) == sorted(coefs)
        True
        sage: res = diagonal_asymptotics_combinatorial(1/(1 - x - y), r=[1,1], expansion_precision=2, output_format="symbolic")
        sage: sorted(get_expansion_terms(res)) == sorted(coefs)
        True

    ::

        sage: res = diagonal_asymptotics_combinatorial(1/(1 - x^7))
        sage: get_expansion_terms(res)
        [Term(coefficient=1/7, pi_factor=1, base=0.6234898018587335? + 0.7818314824680299?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=0.6234898018587335? - 0.7818314824680299?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=-0.2225209339563144? + 0.9749279121818236?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=-0.2225209339563144? - 0.9749279121818236?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=-0.9009688679024191? + 0.4338837391175582?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=-0.9009688679024191? - 0.4338837391175582?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=1, power=0)]

    ::

        sage: res = diagonal_asymptotics_combinatorial(1/(1 - x - y^2))
        sage: coefs = get_expansion_terms(res); coefs
        [Term(coefficient=0.6123724356957945?, pi_factor=1/sqrt(pi), base=-2.598076211353316?, power=-1/2),
         Term(coefficient=0.6123724356957945?, pi_factor=1/sqrt(pi), base=2.598076211353316?, power=-1/2)]
        sage: coefs[0].coefficient.parent()
        Algebraic Field
        sage: coefs[0].coefficient.radical_expression()
        1/2*sqrt(3/2)

    ::

        sage: F2 = (1+x)*(1+y)/(1-z*x*y*(x+y+1/x+1/y))
        sage: res = diagonal_asymptotics_combinatorial(F2, expansion_precision=3)
        sage: coefs = get_expansion_terms(res); coefs
        [Term(coefficient=4, pi_factor=1/pi, base=4, power=-1),
         Term(coefficient=1, pi_factor=1/pi, base=-4, power=-3),
         Term(coefficient=-6, pi_factor=1/pi, base=4, power=-2),
         Term(coefficient=19/2, pi_factor=1/pi, base=4, power=-3)]

    ::

        sage: res = diagonal_asymptotics_combinatorial(3/(1 - x))
        sage: get_expansion_terms(res)
        [Term(coefficient=3, pi_factor=1, base=1, power=0)]

    ::

        sage: res = diagonal_asymptotics_combinatorial((x - y)/(1 - x - y))
        sage: get_expansion_terms(res)
        []

    """
    n = SR.var("n")
    if isinstance(expr, tuple):
        expr = SR(expr[0] ** n * prod(expr[1:]))
    elif isinstance(expr, list):
        expr = SR(sum([tup[0] ** n * prod(tup[1:]) for tup in expr]))
    elif isinstance(expr.parent(), AsymptoticRing):
        expr = expr.exact_part()
        symbolic_expr = SR.zero()
        for summand in expr.summands:
            symbolic_summand = summand.coefficient
            for factor in summand.growth.value:
                if isinstance(factor.parent(), MonomialGrowthGroup):
                    symbolic_summand *= n**factor.exponent
                elif isinstance(factor.parent(), ExponentialGrowthGroup):
                    if isinstance(factor.base.parent(), ArgumentByElementGroup):
                        symbolic_summand *= factor.base._element_**n
                    else:
                        symbolic_summand *= factor.base**n
            symbolic_expr += symbolic_summand
        expr = symbolic_expr

    if not isinstance(expr.parent(), type(SR)):
        raise ACSVException(f"Cannot deal with expression of type {expr.parent()}")

    if len(expr.args()) > 1:
        raise ACSVException("Cannot process multivariate symbolic expression.")

    # If expression is the sum of a bunch of terms, handle each one separately
    expr = expr.expand()
    if expr.is_zero():
        return []

    terms = [expr]
    if expr.operator() == add_vararg:
        terms = expr.operands()

    decomposed_terms = []
    for summand in terms:
        term = Term(
            coefficient=QQ.one(), pi_factor=QQ.one(), base=QQ.one(), power=QQ.zero()
        )
        if summand in QQbar:
            term.coefficient = QQbar(summand)
            decomposed_terms.append(term)
            continue

        for v in summand.operands():
            if n in v.args():
                if v.degree(n) != 0:
                    term.power += v.degree(n)
                else:
                    term.base *= v.operands()[0]
            elif v.degree(pi) != 0:
                pi_deg = v.degree(pi)
                term.pi_factor *= pi**pi_deg
                term.coefficient *= v.coefficient(pi**pi_deg)
            else:
                term.coefficient *= v

        for attr in ("coefficient", "base", "power"):
            elem = getattr(term, attr)
            if isinstance(elem.parent(), SymbolicRing) and elem in QQbar:
                setattr(term, attr, QQbar(elem))

        decomposed_terms.append(term)

    return decomposed_terms


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
