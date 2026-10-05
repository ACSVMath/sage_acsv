"""Functions related to computing the Kronecker representation of
a system of polynomials.
"""

from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ

from sage_acsv.kronecker.kronecker_msolve import _kronecker_representation_msolve
from sage_acsv.kronecker.kronecker_sage import _kronecker_representation_sage
from sage_acsv.debug import acsv_logger
from sage_acsv.settings import ACSVSettings

def _kronecker_representation(system, u_, vs, linear_form=None, return_linear_form=False):
    r"""Computes the Kronecker Representation of a system of polynomials

    Internal intermediate function for choosing the Kronecker backend implementation

    INPUT:

    * ``system`` -- A system of polynomials in ``d`` variables
    * ``u_`` -- Variable not contained in the variables in system
    * ``vs`` -- Variables of the system
    * ``linear_form`` -- (Optional) A linear combination of the
      input variables that separates the critical point solutions

    OUTPUT:

    A polynomial ``P`` and ``d`` polynomials ``Q1, ..., Q_d`` such that
    ``z_i = Q_i(u)/P'(u)`` for ``u`` ranging over the roots of ``P``.
    """
    if ACSVSettings.get_default_kronecker_backend() == ACSVSettings.Kronecker.MSOLVE:
        if linear_form is not None:
            acsv_logger.warning(
                "msolve chooses its own linear form by default. The provided linear form will be dropped."
            )
        return _kronecker_representation_msolve(system, u_, vs, return_linear_form=return_linear_form)
    return _kronecker_representation_sage(system, u_, vs, linear_form=linear_form, return_linear_form=return_linear_form)

def kronecker(system, vs, linear_form=None):
    acsv_logger.warning(
        "The kronecker function has been deprecated and will "
        "be removed in a future version. Please use the "
        "kronecker_representation function (same signature) instead.",
    )
    return kronecker_representation(system, vs, linear_form=linear_form)


def kronecker_representation(system, vs, linear_form=None, return_linear_form=False):
    r"""Computes the Kronecker Representation of a system of polynomials.

    INPUT:

    * ``system`` -- A system of polynomials in `d` variables defining a zero-dimensional (finite) variety
    * ``vs`` -- Variables of the system
    * ``linear_form`` -- (Optional) A linear combination of the
      input variables that separates the critical point solutions

    OUTPUT:

    A polynomial `P` and `d` polynomials `Q1, ..., Q_d` such that
    `z_i = Q_i(u)/P'(u)` for `u` ranging over the roots of `P`.

    EXAMPLES::

        sage: from sage_acsv.kronecker import kronecker_representation
        sage: var('x,y')
        (x, y)
        sage: kronecker_representation([x**3+y**3-10, y**2-2], [x,y], x+y)
        (u_^6 - 6*u_^4 - 20*u_^3 + 36*u_^2 - 120*u_ + 100,
         [60*u_^3 - 72*u_^2 + 360*u_ - 600, 12*u_^4 - 72*u_^2 + 240*u_])
    """
    R, u_ = PolynomialRing(QQ, "u_").objgen()
    R = PolynomialRing(QQ, len(vs) + 1, vs + [u_])
    system = [R(f) for f in system]
    vs = [R(v) for v in vs]
    u_ = R(u_)
    return _kronecker_representation(system, u_, vs, linear_form, return_linear_form)
