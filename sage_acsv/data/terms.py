from __future__ import annotations

from dataclasses import dataclass

from sage.rings.integer import Integer
from sage.rings.qqbar import AlgebraicNumber
from sage.symbolic.expression import Expression

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


@dataclass
class LimitTheoremTerm(Term):
    r"""A dataclass for storing the decomposed terms of a local central limit theorem expression.

    Extends :class:`.Term` with additional fields specific to the density function
    returned by :func:`~sage_acsv.asymptotics.central_limit_theorem_combinatorial`.

    INPUT:

    * ``coefficient`` -- The coefficient of the term
    * ``pi_factor`` -- The factor of pi in the term
    * ``base`` -- The base of the term
    * ``power`` -- The power of the term
    * ``density_symbolic`` -- The symbolic density expression (the exponential factor from the LCLT)
    * ``density_covariance_matrix`` -- The inverse Hessian matrix `D` appearing in the density
    * ``density_mean_vector`` -- The mean vector `v` appearing in the density

    OUTPUT:

    A dataclass with the given attributes.

    EXAMPLES::

        sage: from sage_acsv.helpers import LimitTheoremTerm
        sage: LimitTheoremTerm(1, 1/sqrt(pi), 4, -1/2, None, None, None)
        LimitTheoremTerm(coefficient=1, pi_factor=1/sqrt(pi), base=4, power=-1/2, density_symbolic=None, density_covariance_matrix=None, density_mean_vector=None)
    """

    density_symbolic: Expression | None = None
    density_covariance_matrix: object = None
    density_mean_vector: object = None