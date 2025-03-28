# TODO - this should be removed once we have the global configuration settings
from sage.all import PolynomialRing, QQ, Hom
from sage_acsv.settings import ACSVSettings


def _construct_m2_morphims(ideal):
    R = ideal.gens()[0].parent()
    n = len(R.gens())
    RM2 = PolynomialRing(QQ, n, [f"x{i}" for i in range(n)])
    vs = list(RM2.gens())
    mor = Hom(R, RM2)(vs)
    inv = mor.inverse()
    return mor, inv


def compute_primary_decomposition(ideal):
    """Return the primary decomposition of an ideal.

    If a Macaulay2 interface is provided, it will be used instead
    of Sage's default implementation.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    if ACSVSettings.get_default_groebner_backend() == ACSVSettings.Groebner.MACAULAY2:
        mor, inv = _construct_m2_morphims(ideal)
        ideal = ideal.apply_morphism(mor)
        m2 = ACSVSettings.get_macaulay2()
        return iter(J.sage().apply_morphism(inv) for J in m2.decompose(ideal))
    return ideal.primary_decomposition()


def compute_saturation(ideal_I, ideal_J):
    """Return the saturation of ideal I with respect to ideal J.

    INPUT:

    * ``ideal_I`` - A polynomial ideal
    * ``ideal_J`` - A polynomial ideal
    """
    return ideal_I.saturation(ideal_J)[0]


def compute_groebner_basis(ideal):
    """Return a Groebner Basis of an ideal.

    If a Macaulay2 interface is provided, it will be used instead
    of Sage's default implementation.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    if ACSVSettings.get_default_groebner_backend() == ACSVSettings.Groebner.MACAULAY2:
        mor, inv = _construct_m2_morphims(ideal)
        ideal = ideal.apply_morphism(mor)
        m2 = ACSVSettings.get_macaulay2()
        # This is really roundabout but for some reason returning the generators from M2 to sage
        # puts it in the wrong ring
        return m2.ideal(m2.gb(ideal).generators()).sage().apply_morphism(inv).gens()
    return ideal.groebner_basis()


def compute_radical(ideal):
    """Return the radical of an ideal.

    If a Macaulay2 interface is provided, it will be used instead
    of Sage's default implementation.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    if ACSVSettings.get_default_groebner_backend() == ACSVSettings.Groebner.MACAULAY2:
        mor, inv = _construct_m2_morphims(ideal)
        ideal = ideal.apply_morphism(mor)
        m2 = ACSVSettings.get_macaulay2()
        return m2.radical(ideal).sage().apply_morphism(inv)
    return ideal.radical()
