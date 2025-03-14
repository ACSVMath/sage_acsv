# TODO - this should be removed once we have the global configuration settings
from sage.all import Macaulay2, PolynomialRing, QQ, Hom
from sage_acsv.settings import ACSVSettings

def _construct_m2_morphims(Id):
    R = Id.gens()[0].parent()
    n = len(R.gens())
    RM2 = PolynomialRing(QQ, n, [f'x{i}' for i in range(n)])
    vs = list(RM2.gens())
    mor = Hom(R, RM2)(vs)
    inv = mor.inverse()
    return mor, inv

def PrimaryDecomposition(Id):
    """Return the primary decomposition of an ideal.

    If a Macaulay2 interface is provided, it will be used instead
    of Sage's default implementation.
    
    INPUT:

    * ``Id`` - A polynomial ideal
    """
    if ACSVSettings.get_default_groebner_backend() == ACSVSettings.Groebner.MACAULAY2:
        mor, inv = _construct_m2_morphims(Id)
        Id = Id.apply_morphism(mor)
        m2 = ACSVSettings.get_macaulay2()
        return iter(J.sage().apply_morphism(inv) for J in m2.decompose(Id))
    return Id.primary_decomposition()

def Saturate(I, J):
    """Return the saturation of ideal I with respect to ideal J
    
    INPUT:

    * ``I`` - A polynomial ideal
    * ``J`` - A polynomial ideal
    """
    return I.saturation(J)[0]

def GroebnerBasis(I):
    """Return a Groebner Basis of ideal I

    If a Macaulay2 interface is provided, it will be used instead
    of Sage's default implementation.
    
    INPUT:

    * ``I`` - A polynomial ideal
    """
    if ACSVSettings.get_default_groebner_backend() == ACSVSettings.Groebner.MACAULAY2:
        mor, inv = _construct_m2_morphims(I)
        I = I.apply_morphism(mor)
        m2 = ACSVSettings.get_macaulay2()
        # This is really roundabout but for some reason returning the generators from M2 to sage
        # puts it in the wrong ring
        return m2.ideal(m2.gb(I).generators()).sage().apply_morphism(inv).gens()
    return I.groebner_basis()

def Radical(I):
    """Return the radical of ideal I

    If a Macaulay2 interface is provided, it will be used instead
    of Sage's default implementation.
    
    INPUT:

    * ``I`` - A polynomial ideal
    """
    if ACSVSettings.get_default_groebner_backend() == ACSVSettings.Groebner.MACAULAY2:
        mor, inv = _construct_m2_morphims(I)
        I = I.apply_morphism(mor)
        m2 = ACSVSettings.get_macaulay2()
        return m2.radical(I).sage().apply_morphism(inv)
    return I.radical()