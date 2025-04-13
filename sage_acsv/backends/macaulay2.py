"""Interface to macaulay2."""

from sage.categories.homset import Hom
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ

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
    """Return the primary decomposition of an ideal, computed via
    Macaulay2.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    mor, inv = _construct_m2_morphims(ideal)
    ideal = ideal.apply_morphism(mor)
    m2 = ACSVSettings.get_macaulay2()
    return iter(J.sage().apply_morphism(inv) for J in m2.decompose(ideal))


def compute_groebner_basis(ideal):
    """Return a Groebner basis of an ideal, computed via Macaulay2.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    mor, inv = _construct_m2_morphims(ideal)
    ideal = ideal.apply_morphism(mor)
    m2 = ACSVSettings.get_macaulay2()
    # This is really roundabout but for some reason returning the generators
    # from M2 to sage puts them in the wrong ring
    return m2.ideal(m2.gb(ideal).generators()).sage().apply_morphism(inv).gens()


def compute_radical(ideal):
    """Return the radical of an ideal, computed via Macaulay2.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    mor, inv = _construct_m2_morphims(ideal)
    ideal = ideal.apply_morphism(mor)
    m2 = ACSVSettings.get_macaulay2()
    return m2.radical(ideal).sage().apply_morphism(inv)
