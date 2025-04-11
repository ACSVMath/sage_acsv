"""Functions related to computing Whitney stratifications."""

from sage.combinat.combination import Combinations
from sage.matrix.constructor import matrix
from sage.rings.ideal import Ideal
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ
from sage.schemes.projective.projective_space import ProjectiveSpace

from sage_acsv.groebner import compute_primary_decomposition, compute_groebner_basis, compute_saturation, compute_radical


def conormal_ideal(X, P, RZ):
    r"""Compute the ideal associated with the map sending `X` to its conormal space.

    The map is also denoted as `\operatorname{Con}(X)`.
    """
    vs = P.gens()
    vzs = tuple(v for v in RZ.gens() if v not in vs)
    c = X.codimension()

    X = P.subscheme(compute_radical(X.defining_ideal()))
    M = X.Jacobian_matrix()
    Jac = X.Jacobian()
    JacZ = Ideal(matrix([list(vzs)] + list(M)).minors(c + 1))

    return compute_saturation(X.defining_ideal().change_ring(RZ) + JacZ, Jac.change_ring(RZ))


def decompose_variety(Y, X, P, R, RZ):
    r"""Given varieties `X` and `Y`, return the points in `Y` that fail
    Whitney's condition `B` with respect to `X`.
    """
    d = Y.dimension()
    Ys = [P.subscheme(1) for _ in range(d + 1)]
    Ys[-1] = Y

    J = conormal_ideal(X, P, RZ) + Y.defining_ideal().change_ring(RZ)
    J = Ideal(compute_groebner_basis(J))
    for IQ in compute_primary_decomposition(J):
        K = IQ.elimination_ideal(
            [v for v in RZ.gens() if v not in R.gens()]
        ).change_ring(R)
        W = P.subscheme(K)
        if W.dimension() < Y.dimension():
            i = W.dimension()
            Ys[i] = Ys[i].union(W)
    return Ys


def merge_stratifications(Xs, Ys):
    r"""Merge two stratifications."""
    # Ensures Xs >= Ys
    if len(Ys) > len(Xs):
        return merge_stratifications(Ys, Xs)

    res = []
    for i in range(len(Ys)):
        res.append(Xs[i].union(Ys[i]))
    for i in range(len(Ys), len(Xs)):
        res.append(Xs[i].union(Ys[-1]))

    return res


def whitney_stratification_projective(X, P):
    r"""Computes a Whitney stratification of projective variety `X` in
    the ring `P`.
    """
    X = P.subscheme(compute_radical(X.defining_ideal()))
    vs = P.gens()
    k = X.dimension()

    R = P.gens()[0].parent()
    RZ = PolynomialRing(R, ["t%d" % i for i in range(len(vs))])
    RZ = RZ.flattening_morphism()(vs[0]).parent()

    Xs = [P.subscheme(1) for _ in range(k + 1)]
    Xs[-1] = X
    X_sing = X.intersection(P.subscheme(X.Jacobian()))
    mu = X_sing.dimension()

    for IZ in compute_primary_decomposition(X_sing.defining_ideal()):
        Z = P.subscheme(IZ)
        i = Z.dimension()
        Xs[i] = Xs[i].union(Z)

    for d in range(mu + 1)[::-1]:
        Xd = Xs[d]
        if Xd.dimension() < d:
            continue
        Xs = merge_stratifications(Xs, decompose_variety(Xd, X, P, R, RZ))
        Xs = merge_stratifications(Xs, whitney_stratification_projective(Xd, P))

    return Xs


def whitney_stratification(IX, R):
    r"""Computes the Whitney Stratification of a pure-dimensional algebraic variety.

    Uses an algorithm developed by Helmer and Nanda (2022).

    INPUT:

    * ``IX`` -- A `k`-dimensional polynomial ideal representation of the algebraic variety `X`
    * ``R`` -- Base ring of the ideal. Should be a PolynomialRing object

    OUTPUT:

    A list ``[IX_0, IX_1, ..., IX_k]`` of polynomial ideals representing the Whitney stratification of `X`.
    ``IX_j`` reprensents the `j`-dimensional stratum.

    EXAMPLES::

        sage: from sage_acsv.whitney import whitney_stratification
        sage: R.<x,y,z> = PolynomialRing(QQ, 3)
        sage: whitney_stratification(Ideal(y^2+x^3-y^2*z^2), R)
        [Ideal (y, x, z^2 - 1) of Multivariate Polynomial Ring in x, y, z over Rational Field,
         Ideal (y, x) of Multivariate Polynomial Ring in x, y, z over Rational Field,
         Ideal (y^2*z^2 - x^3 - y^2) of Multivariate Polynomial Ring in x, y, z over Rational Field]
    """
    vs = R.gens()
    d = len(vs)
    # Check if IX = V(H) and H factors smoothly
    # if so, the whitney stratification is just the intersection
    # of subsets of the components
    if len(IX.gens()) == 1:
        factors = [fm[0] for fm in IX.gens()[0].factor()]
        if all(
            Ideal(
                fs + matrix([[f.derivative(v) for v in vs] for f in fs]).minors(len(fs))
            ).dimension()
            < 0
            for fs in Combinations(factors)
            if len(fs) >= 1
        ):
            strat = [Ideal(R.one()) for _ in range(d)]
            strat[-1] = compute_radical(IX)
            for k in reversed(range(1, d)):
                Jac = matrix([[f.derivative(v) for v in vs] for f in strat[k].gens()])
                sing = compute_radical(Ideal(Jac.minors(d - k) + strat[k].gens()))
                for i in range(max(sing.dimension(), 0), k):
                    strat[i] = compute_radical(sing.intersection(strat[i]))
            return strat

    P, vsP = ProjectiveSpace(QQ, len(vs), list(vs) + ["z0"]).objgens()

    z0 = vsP[-1]
    R_hom = z0.parent()
    proj_strat = whitney_stratification_projective(
        P.subscheme(IX.change_ring(R_hom).homogenize(z0)), P
    )

    strat = [Ideal(R.one()) for _ in range(len(proj_strat))]
    for stratum in proj_strat:
        for Id in compute_primary_decomposition(stratum.defining_ideal()):
            newId = Id.subs({z0: 1}).change_ring(R)
            for k in range(newId.dimension(), len(strat)):
                strat[k] = compute_radical(strat[k].intersection(newId))

    return strat
