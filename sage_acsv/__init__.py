r"""A SageMath package for Analytic Combinatorics in Several Variables

This package provides tools for the study of the asymptotic behavior of
the sequence of coefficients of rational generating functions in
several variables along a given direction.

The public interface of our toolbox is provided by the following
functions and classes:

- :func:`.diagonal_asy` -- the central function of the package,
- :func:`.get_expansion_terms` --
- :func:`.ContributingCombinatorial` --
- :func:`.MinimalCriticalCombinatorial` --
- :func:`.CriticalPoints` -- 
- :class:`.ACSVSettings` -- 

The following exmples illustrate some typical use cases. We
first import relevant functions and define the required
symbolic variables::

    sage: from sage_acsv import diagonal_asy, get_expansion_terms
    sage: var('w x y z')
    (w, x, y, z)

The asymptotic expansion for the sequence of central binomial
coefficients, `\binom{2n}{n}`, generated along the `(1, 1)`-diagonal
of `F(x, y) = \frac{1}{1 - x - y}` is simply computed as
::

    sage: diagonal_asy(1 / (1 - x - y))
    1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))

The precision of the expansion can be controlled by the
``expansion_precision`` keyword argument::

    sage: diagonal_asy(1 / (1 - x - y), expansion_precision=4)
    1/sqrt(pi)*4^n*n^(-1/2) - 1/8/sqrt(pi)*4^n*n^(-3/2) + 1/128/sqrt(pi)*4^n*n^(-5/2) + 5/1024/sqrt(pi)*4^n*n^(-7/2) + O(4^n*n^(-9/2))

::

    sage: F1 = (2*y^2 - x)/(x + y - 1)
    sage: diagonal_asy(F1, expansion_precision=3)
    1/4/sqrt(pi)*4^n*n^(-3/2) + 3/32/sqrt(pi)*4^n*n^(-5/2) + O(4^n*n^(-7/2))

::

    sage: F2 = (1+x)*(1+y)/(1-w*x*y*(x+y+1/x+1/y))
    sage: diagonal_asy(F2, expansion_precision=3)
    4/pi*4^n*n^(-1) - 6/pi*4^n*n^(-2) + 1/pi*4^n*n^(-3)*(e^(I*arg(-1)))^n + 19/2/pi*4^n*n^(-3) + O(4^n*n^(-4))

The following example is from Apéry's proof concerning the
irrationality of `\zeta(3)`::

    sage: F3 = 1/(1 - w*(1 + x)*(1 + y)*(1 + z)*(x*y*z + y*z + y + z + 1))
    sage: apery_expansion = diagonal_asy(F3, expansion_precision=2); apery_expansion
    1.225275868941647?/pi^(3/2)*33.97056274847714?^n*n^(-3/2) - 0.5128314911970734?/pi^(3/2)*33.97056274847714?^n*n^(-5/2) + O(33.97056274847714?^n*n^(-7/2))

While the representation might suggest otherwise, the numerical
constants in the expansion are not approximations, but in fact
explicitly known algebraic numbers. We can use the
:func:`.get_expansion_terms` function to inspect them closer::

    sage: coefs = get_expansion_terms(apery_expansion); coefs
    [Term(coefficient=1.225275868941647?, pi_factor=pi^(-3/2), base=33.97056274847714?, power=-3/2),
     Term(coefficient=-0.5128314911970734?, pi_factor=pi^(-3/2), base=33.97056274847714?, power=-5/2)]
    sage: coefs[0].coefficient.radical_expression()
    1/4*sqrt(17/2*sqrt(2) + 12)
    sage: coefs[0].base.radical_expression()
    12*sqrt(2) + 17

::

    sage: F4 = -1/(1 - (1 - x - y)*(20 - x - 40*y))
    sage: diagonal_asy(F4, expansion_precision=2)
    0.09677555757474702?/sqrt(pi)*5.884442204019508?^n*n^(-1/2) + 0.002581950724843528?/sqrt(pi)*5.884442204019508?^n*n^(-3/2) + O(5.884442204019508?^n*n^(-5/2))

The package raises an exception if it detects that some of the
requirements are not met::

    sage: F5 = 1/(x^4*y + x^3*y + x^2*y + x*y - 1)
    sage: diagonal_asy(F5)
    Traceback (most recent call last):
    ...
    ACSVException: No smooth minimal critical points found.

::

    sage: F6 = 1/((-x + 1)^4 - x*y*(x^3 + x^2*y - x^2 - x + 1))
    sage: diagonal_asy(F6)  # long time
    Traceback (most recent call last):
    ...
    ACSVException: No contributing points found.

Here is the asymptotic growth of the Delannoy numbers::

    sage: F7 = 1/(1 - x - y - x*y)
    sage: diagonal_asy(F7)
    1.015051765128218?/sqrt(pi)*5.828427124746190?^n*n^(-1/2) + O(5.828427124746190?^n*n^(-3/2))

::

    sage: F8 = 1/(1 - x^7)
    sage: diagonal_asy(F8)
    1/7 + 1/7*(e^(I*arg(-0.2225209339563144? + 0.9749279121818236?*I)))^n + 1/7*(e^(I*arg(-0.2225209339563144? - 0.9749279121818236?*I)))^n + 1/7*(e^(I*arg(-0.9009688679024191? + 0.4338837391175581?*I)))^n + 1/7*(e^(I*arg(-0.9009688679024191? - 0.4338837391175581?*I)))^n + 1/7*(e^(I*arg(0.6234898018587335? + 0.7818314824680299?*I)))^n + 1/7*(e^(I*arg(0.6234898018587335? - 0.7818314824680299?*I)))^n + O(n^(-1))

This example is for a generating function whose singularities have
very close moduli::

    sage: F9 = 1/(8 - 17*x^3 - 9*x^2 + 7*x)
    sage: diagonal_asy(F9, return_points=True)
    (0.03396226416457560?*1.285654384750451?^n + O(1.285654384750451?^n*n^(-1)),
     [[0.7778140158516262?]])

"""

from sage_acsv.asymptotics import *
from sage_acsv.kronecker import *
from sage_acsv.helpers import get_expansion_terms
from sage_acsv.settings import ACSVSettings
