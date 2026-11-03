"""Central interface for accessing and using different backends
for Groebner basis computations.

Depending on the chosen backend for Groebner basis computations
(via :meth:`.ACSVSettings.set_default_groebner_backend`, allowed
values are given in :class:`.ACSVSettings.Groebner`) the members
of this module delegate their computations to the available
computational backends:

- :mod:`.backends.default`
- :mod:`.backends.macaulay2`
"""


import sage_acsv.backends.default as default_backend
import sage_acsv.backends.macaulay2 as macaulay2_backend
from sage_acsv.settings import ACSVSettings


def compute_primary_decomposition(ideal):
    """Return the primary decomposition of an ideal.

    Delegates the computation to the backend set in
    :class:`.ACSVSettings`.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    match ACSVSettings.get_default_groebner_backend():
        case ACSVSettings.Groebner.MACAULAY2:
            return macaulay2_backend.compute_primary_decomposition(ideal)
        case _:
            return default_backend.compute_primary_decomposition(ideal)


def compute_saturation(ideal_I, ideal_J):
    """Return the saturation of ideal I with respect to ideal J.

    INPUT:

    * ``ideal_I`` - A polynomial ideal
    * ``ideal_J`` - A polynomial ideal
    """
    return ideal_I.saturation(ideal_J)[0]


def compute_groebner_basis(ideal):
    """Return a Groebner basis of an ideal.

    Delegates the computation to the backend set in
    :class:`.ACSVSettings`.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    match ACSVSettings.get_default_groebner_backend():
        case ACSVSettings.Groebner.MACAULAY2:
            return macaulay2_backend.compute_groebner_basis(ideal)
        case _:
            return default_backend.compute_groebner_basis(ideal)


def compute_radical(ideal):
    """Return the radical of an ideal.

    Delegates the computation to the backend set in
    :class:`.ACSVSettings`.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    match ACSVSettings.get_default_groebner_backend():
        case ACSVSettings.Groebner.MACAULAY2:
            return macaulay2_backend.compute_radical(ideal)
        case _:
            return default_backend.compute_radical(ideal)
