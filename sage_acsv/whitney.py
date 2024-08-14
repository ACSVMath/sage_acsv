from sage.all import Ideal, PolynomialRing, ProjectiveSpace, QQ
from sage.all import  matrix

def PrimaryDecompositionM2(Id):
    """
    Uses Macaulay2 to compute the primary decomposition of an ideal
    """
    import macaulay2

    Id = macaulay2.ideal(Id)
    for J in Id.decompose():
        yield J.sage()

def PrimaryDecomposition(Id, m2=False):
    if m2:
        return PrimaryDecompositionM2(Id)
    return Id.primary_decomposition()

def Con(X, P, RZ):
    r"""
    Computes the ideal associated with Con(X), which is the map that sends X to its conormal space
    """
    vs = P.gens()
    vzs = tuple(v for v in RZ.gens() if v not in vs)
    c = X.codimension()

    X = P.subscheme(X.defining_ideal().radical())
    M = X.Jacobian_matrix()
    Jac = X.Jacobian()
    JacZ = Ideal(matrix([list(vzs)] + list(M)).minors(c+1))
    
    return (X.defining_ideal().change_ring(RZ)+JacZ).saturation(Jac.change_ring(RZ))[0]

def Decompose(Y,X,P,R,RZ):
    r"""
    Given varieties X and Y, returns the points in Y that fail Whitney's condition B
    with respect to X
    """
    vs = P.gens()
    vzs = tuple(v for v in RZ.gens() if v not in vs)
    CX = Ideal(vs)

    d = Y.dimension()
    Ys = [P.subscheme(1) for _ in range(d+1)]
    Ys[-1] = Y
    
    J = Con(X, P, RZ) + Y.defining_ideal().change_ring(RZ)
    J = Ideal(J.groebner_basis())
    for IQ in PrimaryDecomposition(J):
        K = IQ.elimination_ideal([v for v in RZ.gens() if v not in R.gens()]).change_ring(R)
        W = P.subscheme(K)
        if W.dimension() < Y.dimension():
            i = W.dimension()
            Ys[i] = Ys[i].union(W)
    return Ys

def Merge(Xs, Ys):
    r"""
    Simple subroutine to merge two stratifications
    """
    # Ensures Xs > Ys
    if len(Ys) > len(Xs):
        return Merge(Ys, Xs)
    
    res = []
    for i in range(len(Ys)):
        res.append(Xs[i].union(Ys[i]))
    for i in range(len(Ys), len(Xs)):
        res.append(Xs[i].union(Ys[-1]))
        
    return res
    
def WhitneyStratProjective(X, P):
    r"""
    Computes a WhitneyStratification of projective variety X in the ring P
    """
    X = P.subscheme(X.defining_ideal().radical())
    vs = P.gens()
    n = len(vs)-1
    k = X.dimension()
    ck = X.codimension()

    R = P.gens()[0].parent()
    RZ = PolynomialRing(R, ['t%d'%i for i in range(len(vs))])
    RZ = RZ.flattening_morphism()(vs[0]).parent()
    
    Xs = [P.subscheme(1) for _ in range(k+1)]
    Xs[-1] = X
    X_sing = X.intersection(P.subscheme(X.Jacobian()))
    mu = X_sing.dimension()
    if (k == mu):
        print("mu same dim as X")
    
    for IZ in PrimaryDecomposition(X_sing.defining_ideal()):
        Z = P.subscheme(IZ)
        i = Z.dimension()
        Xs[i] = Xs[i].union(Z)

    for d in range(mu+1)[::-1]:
        Xd = Xs[d]
        if Xd.dimension() < d:
            continue
        Xs = Merge(Xs, Decompose(Xd, X, P, R, RZ))
        Xs = Merge(Xs, WhitneyStratProjective(Xd, P))
        
    return Xs

def WhitneyStrat(IX, R):
    vs = R.gens()
    P, vsP = ProjectiveSpace(QQ, len(vs), list(vs)+['z0']).objgens()
    
    z0 = vsP[-1]
    R_hom = z0.parent()
    proj_strat = WhitneyStratProjective(P.subscheme(IX.change_ring(R_hom).homogenize(z0)), P)
    
    
    strat = [Ideal(R(1)) for _ in range(len(proj_strat))]
    for stratum in proj_strat:
        for Id in PrimaryDecomposition(stratum.defining_ideal()):
            newId = Id.subs({z0:1}).change_ring(R)
            strat[newId.dimension()] = strat[newId.dimension()].intersection(newId)
            
    return strat