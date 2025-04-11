"""Default implementations of Groebner basis computations
using SageMath."""

def compute_primary_decomposition(ideal):
    """Return the primary decomposition of an ideal.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    return ideal.primary_decomposition()

def compute_groebner_basis(ideal):
    """Return a Groebner basis of an ideal.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    return ideal.groebner_basis()

def compute_radical(ideal):
    """Return the radical of an ideal.

    INPUT:

    * ``ideal`` - A polynomial ideal
    """
    return ideal.radical()
