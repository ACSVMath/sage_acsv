r"""Functions for determining asymptotics of the coefficients
of multivariate rational functions.


The following examples illustrate some typical use cases. We
first import relevant functions and define the required
symbolic variables:

::

    sage: from sage_acsv import diagonal_asymptotics_combinatorial, get_expansion_terms
    sage: var('w x y z')
    (w, x, y, z)

The asymptotic expansion for the sequence of central binomial
coefficients, `\binom{2n}{n}`, generated along the `(1, 1)`-diagonal
of `F(x, y) = \frac{1}{1 - x - y}` is computed by

::

    sage: diagonal_asymptotics_combinatorial(1 / (1 - x - y))
    1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))

The precision of the expansion can be controlled by the
``expansion_precision`` keyword argument::

    sage: diagonal_asymptotics_combinatorial(1 / (1 - x - y), expansion_precision=4)
    1/sqrt(pi)*4^n*n^(-1/2) - 1/8/sqrt(pi)*4^n*n^(-3/2) + 1/128/sqrt(pi)*4^n*n^(-5/2) + 5/1024/sqrt(pi)*4^n*n^(-7/2) + O(4^n*n^(-9/2))

::

    sage: F1 = (2*y^2 - x)/(x + y - 1)
    sage: diagonal_asymptotics_combinatorial(F1, expansion_precision=3)
    1/4/sqrt(pi)*4^n*n^(-3/2) + 3/32/sqrt(pi)*4^n*n^(-5/2) + O(4^n*n^(-7/2))

::

    sage: F2 = (1+x)*(1+y)/(1-w*x*y*(x+y+1/x+1/y))
    sage: diagonal_asymptotics_combinatorial(F2, expansion_precision=3)
    4/pi*4^n*n^(-1) - 6/pi*4^n*n^(-2) + 1/pi*4^n*n^(-3)*(e^(I*arg(-1)))^n + 19/2/pi*4^n*n^(-3) + O(4^n*n^(-4))

The following example comes from Apéry's proof concerning the
irrationality of `\zeta(3)`.

::

    sage: F3 = 1/(1 - w*(1 + x)*(1 + y)*(1 + z)*(x*y*z + y*z + y + z + 1))
    sage: apery_expansion = diagonal_asymptotics_combinatorial(F3, expansion_precision=2); apery_expansion
    1.225275868941647?/pi^(3/2)*33.97056274847714?^n*n^(-3/2) - 0.5128314911970734?/pi^(3/2)*33.97056274847714?^n*n^(-5/2) + O(33.97056274847714?^n*n^(-7/2))

While the representation might suggest otherwise, the numerical
constants in the expansion are not approximations, but in fact
explicitly known algebraic numbers. We can use the
:func:`.get_expansion_terms` function to inspect them closer:

::

    sage: coefs = get_expansion_terms(apery_expansion); coefs
    [Term(coefficient=1.225275868941647?, pi_factor=pi^(-3/2), base=33.97056274847714?, power=-3/2),
     Term(coefficient=-0.5128314911970734?, pi_factor=pi^(-3/2), base=33.97056274847714?, power=-5/2)]
    sage: coefs[0].coefficient.radical_expression()
    1/4*sqrt(17/2*sqrt(2) + 12)
    sage: coefs[0].base.radical_expression()
    12*sqrt(2) + 17

::

    sage: F4 = -1/(1 - (1 - x - y)*(20 - x - 40*y))
    sage: diagonal_asymptotics_combinatorial(F4, expansion_precision=2)
    0.09677555757474702?/sqrt(pi)*5.884442204019508?^n*n^(-1/2) + 0.002581950724843528?/sqrt(pi)*5.884442204019508?^n*n^(-3/2) + O(5.884442204019508?^n*n^(-5/2))

The package raises an exception if it detects that some of the
requirements are not met:

::

    sage: F5 = 1/(x^4*y + x^3*y + x^2*y + x*y - 1)
    sage: diagonal_asymptotics_combinatorial(F5)
    Traceback (most recent call last):
    ...
    ACSVException: No smooth minimal critical points found.

::

    sage: F6 = 1/((-x + 1)^4 - x*y*(x^3 + x^2*y - x^2 - x + 1))
    sage: diagonal_asymptotics_combinatorial(F6)  # long time
    Traceback (most recent call last):
    ...
    ACSVException: No contributing points found.

Here is the asymptotic growth of the Delannoy numbers:

::

    sage: F7 = 1/(1 - x - y - x*y)
    sage: diagonal_asymptotics_combinatorial(F7)
    1.015051765128218?/sqrt(pi)*5.828427124746190?^n*n^(-1/2) + O(5.828427124746190?^n*n^(-3/2))

::

    sage: F8 = 1/(1 - x^7)
    sage: diagonal_asymptotics_combinatorial(F8)
    1/7 + 1/7*(e^(I*arg(-0.2225209339563144? + 0.9749279121818236?*I)))^n + 1/7*(e^(I*arg(-0.2225209339563144? - 0.9749279121818236?*I)))^n + 1/7*(e^(I*arg(-0.9009688679024191? + 0.4338837391175581?*I)))^n + 1/7*(e^(I*arg(-0.9009688679024191? - 0.4338837391175581?*I)))^n + 1/7*(e^(I*arg(0.6234898018587335? + 0.7818314824680299?*I)))^n + 1/7*(e^(I*arg(0.6234898018587335? - 0.7818314824680299?*I)))^n + O(n^(-1))

This example is for a generating function whose singularities have
very close moduli:

::

    sage: F9 = 1/(8 - 17*x^3 - 9*x^2 + 7*x)
    sage: diagonal_asymptotics_combinatorial(F9, return_points=True)
    (0.03396226416457560?*1.285654384750451?^n + O(1.285654384750451?^n*n^(-1)),
     [[0.7778140158516262?]])

"""

from copy import copy

from sage.algebras.weyl_algebra import DifferentialWeylAlgebra
from sage.arith.misc import gcd
from sage.arith.srange import srange
from sage.functions.log import log, exp
from sage.functions.other import factorial
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.misc.prandom import shuffle
from sage.modules.free_module_element import vector
from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing
from sage.rings.ideal import Ideal
from sage.rings.imaginary_unit import I
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import AA, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.constants import pi
from sage.symbolic.ring import SR

from sage_acsv.kronecker import _kronecker_representation
from sage_acsv.helpers import (
    ACSVException,
    is_contributing,
    compute_newton_series,
    rational_function_reduce,
    compute_hessian,
    compute_implicit_hessian,
    collapse_zero_part,
)
from sage_acsv.debug import Timer, acsv_logger
from sage_acsv.settings import ACSVSettings
from sage_acsv.whitney import whitney_stratification
from sage_acsv.groebner import compute_primary_decomposition, compute_saturation


# we need to monkeypatch a function from the asymptotics module such that creating
# asymptotic expansions over QQbar is possible. this should be removed once the
# upstream issue is resolved.

import sage.rings.asymptotic.misc as asy_misc

from sage.rings.integer_ring import ZZ

strip_symbolic_original = asy_misc.strip_symbolic


def strip_symbolic(expression):
    expression = strip_symbolic_original(expression)
    if expression in ZZ:
        expression = ZZ(expression)
    return expression


asy_misc.strip_symbolic = strip_symbolic


def _diagonal_asymptotics_combinatorial_smooth(
    G,
    H,
    r=None,
    linear_form=None,
    expansion_precision=1,
    return_points=False,
    output_format=None,
    as_symbolic=False,
):
    r"""Asymptotics in a given direction `r` of the multivariate rational
    function `F = G/H` when the singular variety of `F` is smooth.

    The function is assumed to have a combinatorial expansion.

    INPUT:

    * ``G`` -- The numerator of the rational function ``F``.
    * ``H`` -- The denominator of the rational function ``F``.
    * ``r`` -- (Optional) A vector of positive algebraic numbers (generally integers),
      one entry per variable of `F`. Defaults to the appropriate vector of
      all 1's if not specified.
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions.
    * ``expansion_precision`` -- (Optional) A positive integer value. This is the number
      of terms to compute in the asymptotic expansion. Defaults to 1, which
      only computes the leading term.
    * ``return_points`` -- (Optional) If ``True``, also returns the coordinates of
      minimal critical points. By default ``False``.
    * ``output_format`` -- (Optional) A string or :class:`.ACSVSettings.Output`
      specifying the way the asymptotic growth is returned. Allowed values
      currently are:
      - ``"tuple"``: the growth is returned as a list of
        tuples of the form ``(a, n^b, pi^c, d)`` such that the `r`-diagonal of `F`
        is the sum of ``a^n n^b pi^c d + O(a^n n^{b-1})`` over these tuples.
      - ``"symbolic"``: the growth is returned as an expression from the symbolic
        ring ``SR`` in the variable ``n``.
      - ``"asymptotic"``: the growth is returned as an expression from an appropriate
        ``AsymptoticRing`` in the variable ``n``.
      - ``None``: the default, which uses the default set for
        :class:`.ACSVSettings.Output` itself via
        :meth:`.ACSVSettings.set_default_output_format`. The default behavior
        is asymptotic output.
    * ``as_symbolic`` -- (Optional) deprecated in favor of the equivalent
      ``output_format="symbolic"``. Will be removed in a future release.

    OUTPUT:

    A representation of the asymptotic behavior of the coefficient sequence,
    either as a list of tuples, or as a symbolic expression.

    See also:

    - :func:`.diagonal_asymptotics_combinatorial`

    TESTS:

    Check that passing a non-supported ``output_format`` errors out::

        sage: from sage_acsv import diagonal_asymptotics_combinatorial
        sage: var('x y')
        (x, y)
        sage: diagonal_asymptotics_combinatorial(1/(1 - x - y), output_format='hello world')  # indirect doctest
        Traceback (most recent call last):
        ...
        ValueError: 'hello world' is not a valid OutputFormat
        sage: diagonal_asymptotics_combinatorial(1/(1 - x - y), output_format=42)  # indirect doctest
        Traceback (most recent call last):
        ...
        ValueError: 42 is not a valid OutputFormat

    """
    # Initialize variables
    vs = list(H.variables())

    t, lambda_, u_ = PolynomialRing(QQ, "t, lambda_, u_").gens()
    expanded_R = PolynomialRing(QQ, len(vs) + 3, vs + [t, lambda_, u_])

    vs = [expanded_R(v) for v in vs]
    t, lambda_, u_ = expanded_R(t), expanded_R(lambda_), expanded_R(u_)
    vsT = vs + [t, lambda_]

    d = len(vs)
    rd = r[-1]

    # Make sure G and H are coprime, and that H does not vanish at 0
    G, H = expanded_R(G), expanded_R(H)
    if H.subs({v: 0 for v in H.variables()}) == 0:
        raise ValueError("Denominator vanishes at 0.")

    # In case form doesn't separate, we want to try again
    for _ in range(ACSVSettings.MAX_MIN_CRIT_RETRIES):
        try:
            # Find minimal critical points in Kronecker Representation
            min_crit_pts = contributing_points_combinatorial_smooth(
                G, H, vs, r=r, linear_form=linear_form
            )
            break
        except Exception as e:
            if isinstance(e, ACSVException) and e.retry:
                acsv_logger.info(
                    "Randomly generated linear form was not suitable, "
                    f"encountered error: {e}\nRetrying..."
                )
                continue
            else:
                raise e
    else:
        return

    timer = Timer()

    # Find det(zH_z Hess) where Hess is the Hessian of z_1...z_n * log(g(z_1, ..., z_n))
    Det = compute_hessian(H, vsT[0:-2], r).determinant()

    # Find exponential growth
    T = prod([SR(vs[i]) ** r[i] for i in range(d)])

    # Find constants appearing in asymptotics in terms of original variables
    B = SR(1 / Det / rd ** (d - 1) / 2 ** (d - 1))
    C = SR(1 / T)

    # Compute constants at contributing singularities
    n = SR.var("n")
    asm_quantities = []
    for cp in min_crit_pts:
        subs_dict = {SR(v): V for (v, V) in zip(vs, cp)}
        expansion = sum(
            [
                term / (rd * n) ** (term_order)
                for term_order, term in enumerate(
                    _general_term_asymptotics(G, H, r, vs, cp, expansion_precision)
                )
            ]
        )
        B_sub = B.subs(subs_dict)
        C_sub = C.subs(subs_dict)
        try:
            B_sub = QQbar(B_sub)
            C_sub = QQbar(C_sub)
        except (ValueError, TypeError):
            pass
        asm_quantities.append([expansion, B_sub, C_sub])

    n = SR.var("n")
    asm_vals = [(c, QQ(1 - d) / 2, b.sqrt(), a) for (a, b, c) in asm_quantities]
    timer.checkpoint("Final Asymptotics")

    if as_symbolic:
        acsv_logger.warning(
            "The as_symbolic argument has been deprecated in favor of output_format='symbolic' "
        )
        output_format = ACSVSettings.Output.SYMBOLIC

    if output_format is None:
        output_format = ACSVSettings.get_default_output_format()
    else:
        output_format = ACSVSettings.Output(output_format)

    if output_format in (ACSVSettings.Output.TUPLE, ACSVSettings.Output.SYMBOLIC):
        n = SR.var("n")
        result = [
            (base, n**exponent, pi**exponent, constant * expansion)
            for (base, exponent, constant, expansion) in asm_vals
        ]
        if output_format == ACSVSettings.Output.SYMBOLIC:
            result = sum([a**n * b * c * d for (a, b, c, d) in result])

    elif output_format == ACSVSettings.Output.ASYMPTOTIC:
        AR = AsymptoticRing("QQbar^n * n^QQ", QQbar)
        n = AR.gen()
        result = sum(
            [  # bug in AsymptoticRing requires splitting out modulus manually
                constant
                * pi**exponent
                * abs(base) ** n
                * collapse_zero_part(base / abs(base)) ** n
                * n**exponent
                * AR(expansion)
                + (abs(base) ** n * n ** (exponent - expansion_precision)).O()
                for (base, exponent, constant, expansion) in asm_vals
            ]
        )

    else:
        raise NotImplementedError(f"Missing implementation for {output_format}")

    if return_points:
        return result, min_crit_pts

    return result


def diagonal_asy(
    F,
    r=None,
    linear_form=None,
    expansion_precision=1,
    return_points=False,
    output_format=None,
    whitney_strat=None,
    as_symbolic=False,
):
    acsv_logger.warning(
        "diagonal_asy is deprecated and will be removed in a future release. "
        "Please use diagonal_asymptotics_combinatorial (same signature) instead.",
    )
    return diagonal_asymptotics_combinatorial(
        F,
        r=r,
        linear_form=linear_form,
        expansion_precision=expansion_precision,
        return_points=return_points,
        output_format=output_format,
        whitney_strat=whitney_strat,
        as_symbolic=as_symbolic,
    )


def diagonal_asymptotics_combinatorial(
    F,
    r=None,
    linear_form=None,
    expansion_precision=1,
    return_points=False,
    output_format=None,
    whitney_strat=None,
    as_symbolic=False,
):
    r"""Asymptotic behavior of the coefficient array of a multivariate rational
    function `F` along a given direction `r`. The function is assumed to have a combinatorial expansion.

    INPUT:

    * ``F`` -- The rational function `G/H` in `d` variables. This function is
      assumed to have a combinatorial expansion.
    * ``r`` -- (Optional) A vector of length `d` of positive algebraic numbers.
      Defaults to the appropriate vector of all 1's if not specified.
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions. Is generated
      randomly if not specified.
    * ``expansion_precision`` -- (Optional) A positive integer, the number of terms to
      compute in the asymptotic expansion. Defaults to 1, which only computes
      the leading term.
    * ``return_points`` -- (Optional) If ``True``, also returns the coordinates of
      minimal critical points. By default ``False``.
    * ``output_format`` -- (Optional) A string or
      :class:`.ACSVSettings.Output` specifying the way the asymptotic growth
      is returned. Allowed values currently are:

      - ``"tuple"``: the growth is returned as a list of
        tuples of the form ``(a, n^b, pi^c, d)`` such that the `r`-diagonal of `F`
        behaves like the sum of ``a^n n^b pi^c d + O(a^n n^{b-1})`` over these tuples.
      - ``"symbolic"``: the growth is returned as an expression from the symbolic
        ring ``SR`` in the variable ``n``.
      - ``"asymptotic"``: the growth is returned as an expression from an appropriate
        ``AsymptoticRing`` in the variable ``n``.
      - ``None``: the default, which uses the default set for
        :class:`.ACSVSettings.Output` itself via
        :meth:`.ACSVSettings.set_default_output_format`. The default behavior
        is asymptotic output.

    * ``as_symbolic`` -- deprecated in favor of the equivalent
      ``output_format="symbolic"``. Will be removed in a future release.
    * ``whitney_strat`` -- (Optional) If known / precomputed, a
      Whitney Stratification of `V(H)`. The program will not check if
      this stratification is correct. Should be a list of length ``d``, where
      the ``k``-th entry is a list of tuples of ideas generators representing
      a component of the ``k``-dimensional stratum.

    OUTPUT:

    A representation of the asymptotic behavior of the coefficient array of `F` along
    the specified direction.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming `F` has a finite number of critical
    points) the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import diagonal_asymptotics_combinatorial
        sage: var('x, y, z, w')
        (x, y, z, w)
        sage: diagonal_asymptotics_combinatorial(1/(1-x-y))
        1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))
        sage: diagonal_asymptotics_combinatorial(1/(1-(1+x)*y), r = [1,2], return_points=True)
        (1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2)), [[1, 1/2]])
        sage: diagonal_asymptotics_combinatorial(1/(1-(x+y+z)+(3/4)*x*y*z), output_format="symbolic")
        0.840484893481498?*24.68093482214177?^n/(pi*n)
        sage: diagonal_asymptotics_combinatorial(1/(1-(x+y+z)+(3/4)*x*y*z))
        0.840484893481498?/pi*24.68093482214177?^n*n^(-1) + O(24.68093482214177?^n*n^(-2))
        sage: var('n')
        n
        sage: asy = diagonal_asymptotics_combinatorial(
        ....:     1/(1 - w*(1 + x)*(1 + y)*(1 + z)*(x*y*z + y*z + y + z + 1)),
        ....:     output_format="tuple",
        ....: )
        sage: sum([
        ....:      a.radical_expression()^n * b * c * QQbar(d).radical_expression()
        ....:      for (a, b, c, d) in asy
        ....: ])
        1/4*(12*sqrt(2) + 17)^n*sqrt(17/2*sqrt(2) + 12)/(pi^(3/2)*n^(3/2))

    Not specifying any ``output_format`` falls back to the default asymptotic
    representation::

        sage: diagonal_asymptotics_combinatorial(1/(1 - 2*x))
        2^n + O(2^n*n^(-1))
        sage: diagonal_asymptotics_combinatorial(1/(1 - 2*x), output_format="tuple")
        [(2, 1, 1, 1)]

    Passing ``"symbolic"`` lets the function return an element of the
    symbolic ring in the variable ``n`` that describes the asymptotic growth::

        sage: growth = diagonal_asymptotics_combinatorial(1/(1 - 2*x), output_format="symbolic"); growth
        2^n
        sage: growth.parent()
        Symbolic Ring

    The argument ``"asymptotic"`` constructs an asymptotic expansion over
    an appropriate ``AsymptoticRing`` in the variable ``n``, including the
    appropriate error term::

        sage: assume(SR.an_element() > 0)  # required to make coercions involving SR work properly
        sage: growth = diagonal_asymptotics_combinatorial(1/(1 - x - y), output_format="asymptotic"); growth
        1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))
        sage: growth.parent()
        Asymptotic Ring <(Algebraic Real Field)^n * n^QQ * (Arg_(Algebraic Field))^n> over Symbolic Ring

    Increasing the precision of the expansion returns an expansion with more terms
    (works for all available output formats)::

        sage: diagonal_asymptotics_combinatorial(1/(1 - x - y), expansion_precision=3, output_format="asymptotic")
        1/sqrt(pi)*4^n*n^(-1/2) - 1/8/sqrt(pi)*4^n*n^(-3/2) + 1/128/sqrt(pi)*4^n*n^(-5/2)
        + O(4^n*n^(-7/2))

    The direction of the diagonal, `r`, defaults to the standard diagonal (i.e., the
    vector of all 1's) if not specified. It also supports passing non-integer values,
    notably rational numbers::

        sage: diagonal_asymptotics_combinatorial(1/(1 - x - y), r=(1, 17/42), output_format="symbolic")
        1.317305628032865?*2.324541507270374?^n/(sqrt(pi)*sqrt(n))

    and even algebraic numbers (note, however, that the performance for complicated
    algebraic numbers is significantly degraded)::

        sage: diagonal_asymptotics_combinatorial(1/(1 - x - y), r=(sqrt(2), 1))
        0.9238795325112868?/sqrt(pi)*(2.414213562373095?/0.5857864376269049?^1.414213562373095?)^n*n^(-1/2) + O((2.414213562373095?/0.5857864376269049?^1.414213562373095?)^n*n^(-3/2))

    ::

        sage: diagonal_asymptotics_combinatorial(1/(1 - x - y*x^2), r=(1, 1/2 - 1/2*sqrt(1/5)), output_format="asymptotic")
        1.710862642974252?/sqrt(pi)*1.618033988749895?^n*n^(-1/2)
        + O(1.618033988749895?^n*n^(-3/2))

    The function times individual steps of the algorithm, timings can
    be displayed by increasing the printed verbosity level of our debug logger::

        sage: import logging
        sage: from sage_acsv import ACSVSettings
        sage: ACSVSettings.set_logging_level(logging.INFO)
        sage: diagonal_asymptotics_combinatorial(1/(1 - x - y))
        INFO:sage_acsv:... Executed Kronecker in ... seconds.
        INFO:sage_acsv:... Executed Minimal Points in ... seconds.
        INFO:sage_acsv:... Executed Final Asymptotics in ... seconds.
        1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))
        sage: ACSVSettings.set_logging_level(logging.WARNING)

    Extraction of coefficient asymptotics even works in cases where the singular variety of `F`
    is not smooth but is the transverse union of smooth varieties::

        sage: diagonal_asymptotics_combinatorial(1/((1-(2*x+y)/3)*(1-(3*x+y)/4)), r = [17/24, 7/24], output_format = 'asymptotic')
        12 + O(n^(-1))

        sage: diagonal_asymptotics_combinatorial(1/((1-(2*x+y)/3)*(1-(3*x+y)/4)), r = [17/24, 7/24], output_format = 'asymptotic')
        12 + O(n^(-1))
        sage: G = (1+x)*(1-x*y^2+x^2)
        sage: H = (1-z*(1+x^2+x*y^2))*(1-y)*(1+x^2)
        sage: strat = [
        ....:     [(1-z*(1+x^2+x*y^2), 1-y, 1+x^2)],
        ....:     [(1-z*(1+x^2+x*y^2), 1-y),(1-z*(1+x^2+x*y^2), 1+x^2),(1-y,1+x^2)],
        ....:     [(H,)],
        ....: ]
        sage: diagonal_asymptotics_combinatorial(G/H, r = [1,1,1], output_format = 'asymptotic', whitney_strat = strat)
        0.866025403784439?/sqrt(pi)*3^n*n^(-1/2) + O(3^n*n^(-3/2))

    TESTS:

    Check that the workaround for the AsymptoticRing swallowing
    the modulus works as intended::

        sage: diagonal_asymptotics_combinatorial(1/(1 - x^4 - y^4))  # long time
        1/2/sqrt(pi)*1.414213562373095?^n*n^(-1/2) + 1/2/sqrt(pi)*1.414213562373095?^n*n^(-1/2)*(e^(I*arg(-1)))^n + 1/2/sqrt(pi)*1.414213562373095?^n*n^(-1/2)*(e^(I*arg(-I)))^n + 1/2/sqrt(pi)*1.414213562373095?^n*n^(-1/2)*(e^(I*arg(I)))^n + O(1.414213562373095?^n*n^(-3/2))

    Check that there are no prohibited variable names::

        sage: var('n t u_')
        (n, t, u_)
        sage: diagonal_asymptotics_combinatorial(1/(1 - n - t - u_))
        0.866025403784439?/pi*27^n*n^(-1) + O(27^n*n^(-2))

    """
    G, H, variable_map = _prepare_symbolic_fraction(F)
    vs = list(variable_map.values())

    if whitney_strat is not None:
        whitney_strat = [
            [
                tuple(SR(gen).subs(variable_map) for gen in component)
                for component in stratum
            ]
            for stratum in whitney_strat
        ]
    if linear_form is not None:
        linear_form = SR(linear_form).subs(variable_map)

    if r is None:
        n = len(H.variables())
        r = [1 for _ in range(n)]

    try:
        r = [QQ(ri) for ri in r]
    except (ValueError, TypeError):
        r = [AA(ri) for ri in r]

    R = PolynomialRing(QQ, vs, len(vs))
    H_sing = Ideal([R(H)] + [R(H.derivative(v)) for v in vs])
    if H_sing.dimension() < 0:
        return _diagonal_asymptotics_combinatorial_smooth(
            G,
            H,
            r=r,
            linear_form=linear_form,
            expansion_precision=expansion_precision,
            return_points=return_points,
            output_format=output_format,
            as_symbolic=as_symbolic,
        )

    t, lambda_, u_ = PolynomialRing(QQ, "t, lambda_, u_").gens()
    expanded_R = PolynomialRing(QQ, len(vs) + 3, vs + [t, lambda_, u_])

    vs = [expanded_R(v) for v in vs]
    t, lambda_, u_ = expanded_R(t), expanded_R(lambda_), expanded_R(u_)

    d = len(vs)

    # Make sure G and H are coprime, and that H does not vanish at 0
    G, H = rational_function_reduce(G, H)
    G, H = expanded_R(G), expanded_R(H)
    if H.subs({v: 0 for v in H.variables()}) == 0:
        raise ValueError("Denominator vanishes at 0.")

    H_sf = prod([f for f, _ in H.factor()])
    # In case form doesn't separate, we want to try again
    for _ in range(ACSVSettings.MAX_MIN_CRIT_RETRIES):
        try:
            # Find minimal critical points in Kronecker Representation
            min_crit_pts = _find_contributing_points_combinatorial(
                G, H_sf, vs, r=r, linear_form=linear_form, whitney_strat=whitney_strat
            )
            break
        except Exception as e:
            if isinstance(e, ACSVException) and e.retry:
                acsv_logger.info(
                    "Randomly generated linear form was not suitable, "
                    f"encountered error: {e}\nRetrying..."
                )
                continue
            else:
                raise e
    else:
        return

    asm_quantities = []
    # Store copy of vs and r in case order changes due to parametrization
    vs_copy, r_copy = copy(vs), copy(r)
    for cp in min_crit_pts:
        vs, r = copy(vs_copy), copy(r_copy)
        # Step 1: Determine if pt is a transverse multiple point of H,
        # and compute the factorization
        # for now, we'll just try to factor it in the polynomial ring
        R = PolynomialRing(QQbar, len(vs), vs)
        G = R(SR(G))
        H = R(SR(H))
        vs = [R(SR(v)) for v in vs]
        subs_dict = {vs[i]: cp[i] for i in range(d)}
        poly_factors = H.factor()
        unit = poly_factors.unit()
        factors = []
        multiplicities = []
        for factor, multiplicity in poly_factors:
            const = factor.coefficients()[-1]
            unit *= const**multiplicity
            factor /= const
            if factor.subs(subs_dict) != 0:
                unit *= factor.subs(subs_dict)
                continue
            factors.append(factor)
            multiplicities.append(multiplicity)
        s = len(factors)
        normals = matrix(
            [[f.derivative(v).subs(subs_dict) for v in vs] for f in factors]
        )
        if normals.rank() < s:
            raise ACSVException(
                "Not a transverse intersection. Cannot deal with this case."
            )

        # Step 2: Find the locally parametrizing coordinates of the point pt
        # Since we have d variables and s factors, there should be d-s of these
        # parametrizing coordinates
        # We will try to parametrize with the first d-s coordinates, shuffling
        # the vs and r if it doesn't work
        for _ in range(s**2):
            Jac = matrix(
                [
                    [(v * Q.derivative(v)).subs(subs_dict) for v in vs[d - s :]]
                    for Q in factors
                ]
            )
            if Jac.determinant() != 0:
                break

            acsv_logger.info("Variables do not parametrize, shuffling")
            vs_r_cp = list(zip(vs, r, cp))
            shuffle(vs_r_cp)  # shuffle mutates the list
            vs, r, cp = zip(*vs_r_cp)
        else:
            raise ACSVException("Cannot find parametrizing set.")

        # Step 3: Compute the gamma matrix as defined in 9.10
        Gamma = matrix(
            [[(v * Q.derivative(v)).subs(subs_dict) for v in vs] for Q in factors]
            + [
                [v.subs(subs_dict) if vs.index(v) == i else 0 for i in range(d)]
                for v in vs[: d - s]
            ]
        )

        # Some constants appearing for higher order singularities
        mult_fac = prod([factorial(m - 1) for m in multiplicities])
        r_gamma_inv = prod(
            x ** (multiplicities[i] - 1)
            for i, x in enumerate(list(vector(r) * Gamma.inverse())[:s])
        )
        # If cp lies on a single smooth component, we can compute asymptotics
        # like in the smooth case
        if s == 1 and sum(multiplicities) == 1:
            n = SR.var("n")
            expansion = sum(
                term / (r[-1] * n) ** (term_order)
                for term_order, term in enumerate(
                    _general_term_asymptotics(G, H, r, vs, cp, expansion_precision)
                )
            )
            Det = compute_hessian(H, vs, r).determinant()
            B = SR(1 / Det.subs(subs_dict) / r[-1] ** (d - 1) / 2 ** (d - 1))
        else:
            # Higher order expansions not currently supported for non-smooth critical points
            if expansion_precision > 1:
                acsv_logger.warning(
                    "Higher order expansions are not supported in the non-smooth case. Defaulting to expansion_precision 1."
                )
            # For non-complete intersections, we must compute the parametrized Hessian matrix
            if s != d:
                Qw = compute_implicit_hessian(factors, vs, r, subs=subs_dict)
                expansion = SR(abs(prod([v for v in vs[: d - s]]).subs(subs_dict)) * G.subs(subs_dict) / abs(Gamma.determinant()) / unit)
                B = SR(
                    1
                    / Qw.determinant()
                    / 2 ** (d - s)
                )
            else:
                expansion = SR(G.subs(subs_dict) / unit / abs(Gamma.determinant()))
                B = 1

        expansion *= (
            (-1) ** sum([m - 1 for m in multiplicities]) * r_gamma_inv / mult_fac
        )

        T = prod(SR(vs[i].subs(subs_dict)) ** r[i] for i in range(d))
        C = SR(1 / T)
        D = QQ((s - d) / 2 + sum(multiplicities) - s)
        try:
            B = QQbar(B)
            C = QQbar(C)
        except (ValueError, TypeError):
            pass

        asm_quantities.append([expansion, B, C, D, s])

    asm_vals = [(c, d, b.sqrt(), a, s) for a, b, c, d, s in asm_quantities]

    if as_symbolic:
        acsv_logger.warning(
            "The as_symbolic argument has been deprecated in favor of output_format='symbolic' "
        )
        output_format = ACSVSettings.Output.SYMBOLIC

    if output_format is None:
        output_format = ACSVSettings.get_default_output_format()
    else:
        output_format = ACSVSettings.Output(output_format)

    if output_format in (ACSVSettings.Output.TUPLE, ACSVSettings.Output.SYMBOLIC):
        n = SR.var("n")
        result = [
            (base, n**exponent, (pi ** (s - d)).sqrt(), constant * expansion)
            for (base, exponent, constant, expansion, s) in asm_vals
        ]
        if output_format == ACSVSettings.Output.SYMBOLIC:
            result = sum([a**n * b * c * d for (a, b, c, d) in result])

    elif output_format == ACSVSettings.Output.ASYMPTOTIC:
        AR = AsymptoticRing("QQbar^n * n^QQ", QQbar)
        n = AR.gen()
        result = sum(
            [  # bug in AsymptoticRing requires splitting out modulus manually
                constant
                * (pi ** (s - d)).sqrt()
                * abs(base) ** n
                * collapse_zero_part(base / abs(base)) ** n
                * n**exponent
                * AR(expansion)
                + (abs(base) ** n * n ** (exponent - expansion_precision)).O()
                for (base, exponent, constant, expansion, s) in asm_vals
            ]
        )

    else:
        raise NotImplementedError(f"Missing implementation for {output_format}")

    if return_points:
        return result, min_crit_pts

    return result


def _general_term_asymptotics(G, H, r, vs, cp, expansion_precision):
    r"""
    Compute coefficients of general (not necessarily leading) terms of
    the asymptotic expansion for a given critical
    point of a rational combinatorial multivariate rational function.

    Typically, this function is called as a subroutine of :func:`.diagonal_asymptotics_combinatorial`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with `F = G/H`.
    * ``vs`` -- Tuple of variables occurring in `G` and `H`.
    * ``r`` -- The direction. A length `d` vector of positive algebraic numbers (usually
      integers).
    * ``cp`` -- A minimal critical point of `F` with coordinates specified in the
      same order as in ``vs``.
    * ``expansion_precision`` -- A positive integer value. This is the number of terms
      for which to compute coefficients in the asymptotic expansion.

    OUTPUT:

    List of coefficients of the asymptotic expansion.

    EXAMPLES::

        sage: from sage_acsv.asymptotics import _general_term_asymptotics
        sage: R.<x, y, z> = QQ[]
        sage: _general_term_asymptotics(1, 1 - x - y, [1, 1], [x, y], [1/2, 1/2], 5)
        [2, -1/4, 1/64, 5/512, -21/16384]
        sage: _general_term_asymptotics(1, 1 - x - y - z, [1, 1, 1], [x, y, z], [1/3, 1/3, 1/3], 4)
        [3, -2/3, 2/27, 14/729]
    """

    if expansion_precision == 1:
        A = SR(-G / vs[-1] / H.derivative(vs[-1]))
        subs_dict = {SR(v): V for (v, V) in zip(vs, cp)}
        return [A.subs(subs_dict)]

    # Convert everything to field of algebraic numbers
    d = len(vs)
    R = PolynomialRing(QQbar, vs)
    vs = R.gens()
    vd = vs[-1]
    tvars = SR.var("t", d - 1)
    G, H = R(SR(G)), R(SR(H))

    cp = {v: V for (v, V) in zip(vs, cp)}

    W = DifferentialWeylAlgebra(PolynomialRing(QQbar, tvars))
    TR = QQbar[[tvars]]
    T = TR.gens()
    tvars = T
    D = list(W.differentials())

    # Function to apply differential operator dop on function f
    def eval_op(dop, f):
        if len(f.parent().gens()) == 1:
            return sum(
                prod([factorial(k) for k in E[0][1]]) * E[1] * f[E[0][1][0]]
                for E in dop
            )
        else:
            return sum(
                [prod([factorial(k) for k in E[0][1]]) * E[1] * f[E[0][1]] for E in dop]
            )

    Hess = compute_hessian(H, vs, r, cp)
    Hessinv = Hess.inverse()
    v = matrix(W, [D[: d - 1]])
    Epsilon = -(v * Hessinv.change_ring(W) * v.transpose())[0, 0]

    # P and PsiTilde only need to be computed to order 2M
    N = 2 * expansion_precision + 1

    # Find series expansion of function g given implicitly by
    # H(w_1, ..., w_{d-1}, g(w_1, ..., w_{d-1})) = 0 up to needed order
    g = compute_newton_series(H.subs({v: v + cp[v] for v in vs}), vs, N)
    g = g.subs({v: v - cp[v] for v in vs}) + cp[vd]

    # Polar change of coordinates
    tsubs = {v: cp[v] * exp(I * t).add_bigoh(N) for v, t in zip(vs, tvars)}
    tsubs[vd] = g.subs(tsubs)

    # Compute PsiTilde up to needed order
    psi = log(g.subs(tsubs) / g.subs(cp)).add_bigoh(N)
    psi += I * sum([r[k] * tvars[k] for k in range(d - 1)]) / r[-1]
    v = matrix(TR, [tvars[k] for k in range(d - 1)])
    psiTilde = psi - (v * Hess * v.transpose())[0, 0] / 2
    PsiSeries = psiTilde.truncate(N)

    # Compute series expansion of P = -G/(g*H_{z_d}) up to needed order
    P_num = -G.subs(tsubs).add_bigoh(N)
    P_denom = (g * H.derivative(vd)).subs(tsubs).add_bigoh(N)
    PSeries = (P_num / P_denom).truncate(N)

    if len(tvars) > 1:
        PsiSeries = PsiSeries.polynomial()
        PSeries = PSeries.polynomial()

    # Precompute products used for asymptotics
    EE = [Epsilon**k for k in range(3 * expansion_precision - 2)]
    PP = [PSeries]
    for k in range(1, 2 * expansion_precision - 1):
        PP.append(PP[k - 1] * PsiSeries)

    # Function to compute constants appearing in asymptotic expansion
    def constants_clj(ell, j):
        extra_contrib = (-1) ** j / (
            2 ** (ell + j) * factorial(ell) * factorial(ell + j)
        )
        return extra_contrib * eval_op(EE[ell + j], PP[ell])

    res = [
        sum([constants_clj(ell, j) for ell in srange(2 * j + 1)])
        for j in srange(expansion_precision)
    ]
    try:
        for i in range(len(res)):
            if res[i].imag() == 0:
                res[i] = AA(res[i])
    except (TypeError, ValueError, NotImplementedError):
        pass

    return res


def contributing_points_combinatorial_smooth(G, H, variables, r=None, linear_form=None):
    r"""Compute contributing points of a multivariate
    rational function `F=G/H` admitting a finite number of critical points.
    Assumes that the singular variety of `F` is smooth and the function has a combinatorial expansion.

    Typically, this function is called as a subroutine of :func:`._diagonal_asymptotics_combinatorial_smooth`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with `F = G/H`.
    * ``variables`` -- List of variables of ``G`` and ``H``.
    * ``r`` -- (Optional) the direction, a vector of positive algebraic
      numbers (usually integers).
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions.

    OUTPUT:

    List of minimal critical points of `F` in the direction `r`,
    as a list of tuples of algebraic numbers.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import contributing_points_combinatorial_smooth
        sage: R.<x, y, w, lambda_, t, u_> = QQ[]
        sage: pts = contributing_points_combinatorial_smooth(
        ....:     1,
        ....:     1 - w*(y + x + x^2*y + x*y^2),
        ....:     [w, x, y],
        ....: )
        sage: sorted(pts)
        [[-1/4, -1, -1], [1/4, 1, 1]]
    """

    timer = Timer()
    (
        expanded_R,
        vs,
        (t, lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r)

    G, H = expanded_R(G), expanded_R(H)
    vsT = vs + list(r_variable_values.keys()) + [t, lambda_]

    # Create the critical point equations system
    vsH = H.variables()
    system = [
        H_var * H.derivative(H_var) - r_var * lambda_
        for H_var, r_var in zip(H.variables(), r)
    ]
    system.extend([H, H.subs({z: z * t for z in vsH})])
    system.extend(
        [
            direction_value.minpoly().subs(direction_var)
            for direction_var, direction_value in r_variable_values.items()
        ]
    )

    # Compute the Kronecker representation of our system
    timer.checkpoint()

    P, Qs = _kronecker_representation(system, u_, vsT, linear_form)
    timer.checkpoint("Kronecker")

    Qt = Qs[-2]  # Qs ordering is H.variables() + rvars + [t, lambda_]
    Pd = P.derivative()

    # Solutions to Pt are solutions to the system where t is not 1
    one_minus_t = gcd(Pd - Qt, P)
    Pt, _ = P.quo_rem(one_minus_t)
    rts_t_zo = list(
        filter(
            lambda k: (Qt / Pd).subs(u_=k) > 0 and (Qt / Pd).subs(u_=k) < 1,
            Pt.roots(AA, multiplicities=False),
        )
    )
    non_min = [[(q / Pd).subs(u_=u) for q in Qs[0:-2]] for u in rts_t_zo]

    # Filter the real roots for minimal points with positive coords
    pos_minimals = []
    for u in one_minus_t.roots(AA, multiplicities=False):
        is_min = True
        v = [(q / Pd).subs(u_=u) for q in Qs[: len(vs)]]
        rv = {
            ri: (q / Pd).subs(u_=u)
            for (ri, q) in zip(r_variable_values, Qs[len(vs) : -2])
        }
        if any([rv[ri] != ri_value for ri, ri_value in r_variable_values.items()]):
            continue
        if any([value <= 0 for value in v[: len(vs)]]):
            continue
        for pt in non_min:
            if all([a == b for (a, b) in zip(v, pt)]):
                is_min = False
                break
        if is_min:
            pos_minimals.append(u)

    # Remove non-smooth points and points with zero coordinates (where lambda=0)
    for i in range(len(pos_minimals)):
        x = (Qs[-1] / Pd).subs(u_=pos_minimals[i])
        if x == 0:
            acsv_logger.warning(
                f"Removing critical point {pos_minimals[i]} because it either "
                "has a zero coordinate or is not smooth."
            )
            pos_minimals.pop(i)

    # Verify necessary assumptions
    if len(pos_minimals) == 0:
        raise ACSVException("No smooth minimal critical points found.")
    elif len(pos_minimals) > 1:
        raise ACSVException(
            f"More than one minimal point with positive real coordinates found: {pos_minimals}"
        )

    # Find all minimal critical points
    minCP = [(q / Pd).subs(u_=pos_minimals[0]) for q in Qs[0:-2]]
    minimals = []

    for u in one_minus_t.roots(QQbar, multiplicities=False):
        v = [(q / Pd).subs(u_=u) for q in Qs[: len(vs)]]
        rv = {
            ri: (q / Pd).subs(u_=u)
            for (ri, q) in zip(r_variable_values, Qs[len(vs) : -2])
        }
        if any([rv[r_var] != r_value for r_var, r_value in r_variable_values.items()]):
            continue
        if all([a.abs() == b.abs() for (a, b) in zip(minCP, v)]):
            minimals.append(u)

    # Get minimal point coords, and make exact if possible
    minimal_coords = [[(q / Pd).subs(u_=u) for q in Qs[: len(vs)]] for u in minimals]
    [[a.exactify() for a in b] for b in minimal_coords]

    timer.checkpoint("Minimal Points")

    return [[(q / Pd).subs(u_=u) for q in Qs[: len(vs)]] for u in minimals]


def _find_contributing_points_combinatorial(
    G,
    H,
    variables,
    r=None,
    linear_form=None,
    whitney_strat=None,
):
    r"""Compute contributing points of a combinatorial multivariate
    rational function `F=G/H` admitting a finite number of critical points where the singular variety is the transverse union of smooth varieties.

    Typically, this function is called as a subroutine of :func:`.diagonal_asymptotics_combinatorial`.

    INPUT:

    * ``G, H`` -- Coprime polynomials with ``F = G/H``
    * ``variables`` -- List of variables of ``G`` and ``H``
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions
    * ``whitney_strat`` -- (Optional) If known / precomputed, a
      Whitney Stratification of `V(H)`. The program will not check if
      this stratification is correct. Should be a list of length ``d``, where
      the ``k``-th entry is a list of tuples of ideas generators representing
      a component of the ``k``-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.
    """
    (
        expanded_R,
        vs,
        (t, lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r)
    H = expanded_R(H)
    vsT = vs + list(r_variable_values.keys()) + [t, lambda_]

    # Compute the critical point system for each stratum
    pure_H = PolynomialRing(QQ, vs)

    if whitney_strat is None:
        whitney_strat = whitney_stratification(Ideal(pure_H(H)), pure_H)
    else:
        # Cast symbolic generators for provided stratification into the correct ring
        whitney_strat = [
            prod([Ideal([pure_H(f) for f in comp]) for comp in stratum])
            for stratum in whitney_strat
        ]

    critical_point_ideals = []
    for d, stratum in enumerate(whitney_strat):
        critical_point_ideals.append([])
        for P in compute_primary_decomposition(stratum):
            c = len(vs) - d
            P_ext = P.change_ring(expanded_R)
            M = matrix([[v * f.derivative(v) for v in vs] for f in P_ext.gens()] + [r])
            # Add in min polys for the direction variables
            r_polys = [
                direction_value.minpoly().subs(direction_var)
                for direction_var, direction_value in r_variable_values.items()
            ]
            # Create ideal of expanded_R containing extended critical point equations
            cpid = P_ext + Ideal(
                M.minors(c + 1)
                + [H.subs({v: v * t for v in vs}), (prod(vs) * lambda_ - 1)]
                + r_polys
            )
            # Saturate cpid by lower dimension stratum, if d > 0
            if d > 0:
                cpid = compute_saturation(
                    cpid, whitney_strat[d - 1].change_ring(expanded_R)
                )

            critical_point_ideals[-1].append((P, cpid))

    # Final minimal critical points with positive coordinates on each stratum
    critical_points_by_stratum = {}
    pos_minimals_by_stratum = {}
    for d in reversed(range(len(critical_point_ideals))):
        ideals = critical_point_ideals[d]
        critical_points_by_stratum[d] = []
        pos_minimals_by_stratum[d] = []

        for _, ideal in ideals:
            if ideal.dimension() < 0:
                continue
            P, Qs = _kronecker_representation(ideal.gens(), u_, vsT, linear_form)

            Qt = Qs[-2]  # Qs ordering is H.variables() + rvars + [t, lambda_]
            Pd = P.derivative()

            # Solutions to Pt are solutions to the system where t is not 1
            one_minus_t = gcd(Pd - Qt, P)
            Pt, _ = P.quo_rem(one_minus_t)
            rts_t_zo = list(
                filter(
                    lambda k: (Qt / Pd).subs(u_=k) > 0 and (Qt / Pd).subs(u_=k) < 1,
                    Pt.roots(AA, multiplicities=False),
                )
            )
            non_min = [[(q / Pd).subs(u_=u) for q in Qs[0:-2]] for u in rts_t_zo]

            # Filter the real roots for minimal points with positive coords
            pos_minimals = []
            for u in one_minus_t.roots(AA, multiplicities=False):
                is_min = True
                v = [(q / Pd).subs(u_=u) for q in Qs[: len(vs)]]
                rv = {
                    ri: (q / Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs) : -2])
                }
                if any(
                    [rv[ri] != ri_value for ri, ri_value in r_variable_values.items()]
                ):
                    continue
                if any([value <= 0 for value in v[: len(vs)]]):
                    continue
                for pt in non_min:
                    if all([a == b for (a, b) in zip(v, pt)]):
                        is_min = False
                        break
                if is_min:
                    pos_minimals.append(u)

            pos_minimals_by_stratum[d].extend(
                [
                    [
                        collapse_zero_part(QQbar((q / Pd).subs(u_=u)))
                        for q in Qs[: len(vs)]
                    ]
                    for u in pos_minimals
                ]
            )

            # Characterize all complex critical points in each stratum
            for u in one_minus_t.roots(QQbar, multiplicities=False):
                rv = {
                    ri: (q / Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs) : -2])
                }
                if any(
                    rv[r_var] != r_value for r_var, r_value in r_variable_values.items()
                ):
                    continue

                w = [
                    collapse_zero_part(QQbar((q / Pd).subs(u_=u)))
                    for q in Qs[: len(vs)]
                ]
                critical_points_by_stratum[d].append(w)

    # Refine positive minimal critical points to those that are contributing
    contributing_pos_minimals = []
    all_factors = list(factor[0] for factor in H.factor())
    r = [QQbar(r_variable_values.get(ri, ri)) for ri in r]
    for d in reversed(range(len(critical_point_ideals))):
        pos_minimals = pos_minimals_by_stratum[d]
        if len(contributing_pos_minimals) > 0:
            break

        for x in pos_minimals:
            if is_contributing(vs, x, r, all_factors, len(vs) - d):
                contributing_pos_minimals.append(x)
                for i in range(d):
                    stratum = whitney_strat[i]
                    if stratum.subs(
                        {pure_H(wi): val for wi, val in zip(vs, x)}
                    ) == Ideal(pure_H.zero()):
                        raise ACSVException(
                            "Non-generic direction detected - critical point {w} is contained in {dim}-dimensional stratum".format(
                                w=str(x), dim=i
                            )
                        )

    if len(contributing_pos_minimals) == 0:
        raise ACSVException("No contributing points found.")
    if len(contributing_pos_minimals) > 1:
        raise ACSVException(
            f"More than one minimal contributing point with positive real coordinates found: {contributing_pos_minimals}"
        )
    minimal = contributing_pos_minimals[0]

    # Characterize all complex contributing points
    contributing_points = []
    for d in reversed(range(len(critical_point_ideals))):
        for w in critical_points_by_stratum[d]:
            if all(
                [abs(w_i) == abs(min_i) for w_i, min_i in zip(w, minimal)]
            ) and is_contributing(vs, w, r, all_factors, len(vs) - d):
                contributing_points.append(w)

    return contributing_points


def contributing_points_combinatorial(
    F,
    r=None,
    linear_form=None,
    whitney_strat=None,
):
    r"""Compute contributing points of a multivariate
    rational function `F=G/H` admitting a finite number of critical points. 

    The function is assumed to have a combinatorial expansion. 

    INPUT:

    * ``F`` -- Symbolic fraction, the rational function assumed to have
      a finite number of critical points.
    * ``r`` -- (Optional) Length `d` vector of positive integers.
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions.
    * ``whitney_strat`` -- (Optional) If known / precomputed, a
      Whitney Stratification of `V(H)`. The program will not check if
      this stratification is correct. Should be a list of length `d`, where
      the `k`-th entry is a list of tuples of ideas generators representing
      a component of the `k`-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming `F` has a finite number of critical points)
    the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import contributing_points_combinatorial
        sage: var('x y')
        (x, y)
        sage: pts = contributing_points_combinatorial(1/((1-(2*x+y)/3)*(1-(3*x+y)/4)))
        sage: sorted(pts)
        [[3/4, 3/2]]

    """
    G, H, variable_map = _prepare_symbolic_fraction(F)
    variables = list(variable_map.values())
    if whitney_strat is not None:
        whitney_strat = [
            [
                tuple(SR(gen).subs(variable_map) for gen in component)
                for component in stratum
            ]
            for stratum in whitney_strat
        ]
    if linear_form is not None:
        linear_form = SR(linear_form).subs(variable_map)

    return _find_contributing_points_combinatorial(
        G,
        H,
        variables,
        r=r,
        linear_form=linear_form,
        whitney_strat=whitney_strat,
    )


def MinimalCriticalCombinatorial(F, r=None, linear_form=None, whitney_strat=None):
    acsv_logger.warning(
        "MinimalCriticalCombinatorial is deprecated and will be removed "
        "in a future release. Please use minimal_critical_points_combinatorial "
        "(same signature) instead.",
    )
    return minimal_critical_points_combinatorial(
        F, r=r, linear_form=linear_form, whitney_strat=whitney_strat
    )


def minimal_critical_points_combinatorial(
    F, r=None, linear_form=None, whitney_strat=None
):
    r"""Compute nonzero minimal critical points of a combinatorial multivariate
    rational function `F=G/H` admitting a finite number of critical points.

    The function is assumed to have a combinatorial expansion.

    INPUT:

    * ``F`` -- Symbolic fraction, the rational function of interest.
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions
    * ``whitney_strat`` -- (Optional) If known / precomputed, a
      Whitney Stratification of `V(H)`. The program will not check if
      this stratification is correct. Should be a list of length ``d``, where
      the ``k``-th entry is a list of tuples of ideas generators representing
      a component of the ``k``-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import minimal_critical_points_combinatorial
        sage: var('x y')
        (x, y)
        sage: pts = minimal_critical_points_combinatorial(1/((1-(2*x+y)/3)*(1-(3*x+y)/4)))
        sage: sorted(pts)
        [[3/4, 3/2], [1, 1]]

    """
    _, H, variable_map = _prepare_symbolic_fraction(F)
    variables = list(variable_map.values())
    if whitney_strat is not None:
        whitney_strat = [
            [
                tuple(SR(gen).subs(variable_map) for gen in component)
                for component in stratum
            ]
            for stratum in whitney_strat
        ]
    if linear_form is not None:
        linear_form = SR(linear_form).subs(variable_map)
    (
        expanded_R,
        vs,
        (t, lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r)
    H = expanded_R(H)

    vsT = vs + list(r_variable_values) + [t, lambda_]

    # Compute the critical point system for each stratum
    pure_H = PolynomialRing(QQ, vs)

    if whitney_strat is None:
        whitney_strat = whitney_stratification(Ideal(pure_H(H)), pure_H)
    else:
        # Cast symbolic generators for provided stratification into the correct ring
        whitney_strat = [
            prod([Ideal([pure_H(f) for f in comp]) for comp in stratum])
            for stratum in whitney_strat
        ]

    critical_points = []
    pos_minimals = []
    for d, stratum in enumerate(whitney_strat):
        for P_comp in compute_primary_decomposition(stratum):
            c = len(vs) - d
            P_ext = P_comp.change_ring(expanded_R)
            M = matrix([[v * f.derivative(v) for v in vs] for f in P_ext.gens()] + [r])
            # Add in min polys for the direction variables
            r_polys = [
                direction_value.minpoly().subs(direction_var)
                for direction_var, direction_value in r_variable_values.items()
            ]
            # Create ideal of expanded_R containing extended critical point equations
            ideal = P_ext + Ideal(
                M.minors(c + 1)
                + [H.subs({v: v * t for v in vs}), (prod(vs) * lambda_ - 1)]
                + r_polys
            )
            # Saturate cpid by lower dimension stratum, if d > 0
            if d > 0:
                ideal = compute_saturation(
                    ideal, whitney_strat[d - 1].change_ring(expanded_R)
                )

            if ideal.dimension() < 0:
                continue
            P, Qs = _kronecker_representation(ideal.gens(), u_, vsT, linear_form)

            Qt = Qs[-2]  # Qs ordering is H.variables() + rvars + [t, lambda_]
            Pd = P.derivative()

            # Solutions to Pt are solutions to the system where t is not 1
            one_minus_t = gcd(Pd - Qt, P)
            Pt, _ = P.quo_rem(one_minus_t)
            rts_t_zo = list(
                filter(
                    lambda k: (Qt / Pd).subs(u_=k) > 0 and (Qt / Pd).subs(u_=k) < 1,
                    Pt.roots(AA, multiplicities=False),
                )
            )
            non_min = [[(q / Pd).subs(u_=u) for q in Qs[0:-2]] for u in rts_t_zo]

            # Filter the real roots for minimal points with positive coords
            for u in one_minus_t.roots(AA, multiplicities=False):
                is_min = True
                v = [(q / Pd).subs(u_=u) for q in Qs[: len(vs)]]
                rv = {
                    ri: (q / Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs) : -2])
                }
                if any(
                    [rv[ri] != ri_value for ri, ri_value in r_variable_values.items()]
                ):
                    continue
                if any([value <= 0 for value in v[: len(vs)]]):
                    continue
                for pt in non_min:
                    if all([a == b for (a, b) in zip(v, pt)]):
                        is_min = False
                        break
                if is_min:
                    pos_minimals.append(v)

            # Characterize all complex critical points in each stratum
            for u in one_minus_t.roots(QQbar, multiplicities=False):
                rv = {
                    ri: (q / Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs) : -2])
                }
                if any(
                    rv[r_var] != r_value for r_var, r_value in r_variable_values.items()
                ):
                    continue

                w = [QQbar((q / Pd).subs(u_=u)) for q in Qs[: len(vs)]]
                critical_points.append(w)

    if len(pos_minimals) == 0:
        raise ACSVException("No critical points found.")

    # Characterize all complex contributing points
    minimal_criticals = []
    for w in critical_points:
        if any(
            all([abs(w_i) == abs(min_i) for w_i, min_i in zip(w, minimal)])
            for minimal in pos_minimals
        ):
            minimal_criticals.append(w)

    minimal_criticals = [
        [collapse_zero_part(w_i) for w_i in w] for w in minimal_criticals
    ]
    return minimal_criticals


def critical_points(F, r=None, linear_form=None, whitney_strat=None):
    r"""Compute critical points of a multivariate
    rational function `F=G/H` admitting a finite number of critical points.

    Typically, this function is called as a subroutine of :func:`.diagonal_asymptotics_combinatorial`.

    INPUT:

    * ``F`` -- Symbolic fraction, the rational function of interest.
    * ``r`` -- (Optional) Length ``d`` vector of positive integers
    * ``linear_form`` -- (Optional) A linear combination of the input
      variables that separates the critical point solutions
    * ``whitney_strat`` -- (Optional) If known / precomputed, a
      Whitney Stratification of `V(H)`. The program will not check if
      this stratification is correct. Should be a list of length ``d``, where
      the ``k``-th entry is a list of tuples of ideas generators representing
      a component of the ``k``-dimensional stratum.

    OUTPUT:

    A list of minimal contributing points of `F` in the direction `r`,

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming F has a finite number of critical points)
    the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import critical_points
        sage: var('x y')
        (x, y)
        sage: pts = critical_points(1/((1-(2*x+y)/3)*(1-(3*x+y)/4)))
        sage: sorted(pts)
        [[2/3, 2], [3/4, 3/2], [1, 1]]

    """
    _, H, variable_map = _prepare_symbolic_fraction(F)
    variables = list(variable_map.values())
    if whitney_strat is not None:
        whitney_strat = [
            [
                tuple(SR(gen).subs(variable_map) for gen in component)
                for component in stratum
            ]
            for stratum in whitney_strat
        ]
    if linear_form is not None:
        linear_form = SR(linear_form).subs(variable_map)
    (
        expanded_R,
        vs,
        (lambda_, u_),
        r,
        r_variable_values,
    ) = _prepare_expanded_polynomial_ring(variables, direction=r, include_t=False)
    H = expanded_R(H)

    vsT = vs + list(r_variable_values.keys()) + [lambda_]

    # Compute the critical point system for each stratum
    pure_H = PolynomialRing(QQ, vs)

    if whitney_strat is None:
        whitney_strat = whitney_stratification(Ideal(pure_H(H)), pure_H)
    else:
        # Cast symbolic generators for provided stratification into the correct ring
        whitney_strat = [
            prod([Ideal([pure_H(f) for f in comp]) for comp in stratum])
            for stratum in whitney_strat
        ]

    critical_points = []
    for d, stratum in enumerate(whitney_strat):
        for P_comp in compute_primary_decomposition(stratum):
            c = len(vs) - d
            P_ext = P_comp.change_ring(expanded_R)
            M = matrix([[v * f.derivative(v) for v in vs] for f in P_ext.gens()] + [r])
            # Add in min polys for the direction variables
            r_polys = [
                direction_value.minpoly().subs(direction_var)
                for direction_var, direction_value in r_variable_values.items()
            ]
            # Create ideal of expanded_R containing extended critical point equations
            ideal = P_ext + Ideal(
                M.minors(c + 1) + [(prod(vs) * lambda_ - 1)] + r_polys
            )
            # Saturate cpid by lower dimension stratum, if d > 0
            if d > 0:
                ideal = compute_saturation(
                    ideal, whitney_strat[d - 1].change_ring(expanded_R)
                )

            if ideal.dimension() < 0:
                continue
            P, Qs = _kronecker_representation(ideal.gens(), u_, vsT, linear_form)

            Pd = P.derivative()

            # Characterize all complex critical points in each stratum
            for u in P.roots(QQbar, multiplicities=False):
                rv = {
                    ri: (q / Pd).subs(u_=u)
                    for (ri, q) in zip(r_variable_values, Qs[len(vs) : -2])
                }
                if any(
                    rv[r_var] != r_value for r_var, r_value in r_variable_values.items()
                ):
                    continue

                w = [
                    collapse_zero_part(QQbar((q / Pd).subs(u_=u)))
                    for q in Qs[: len(vs)]
                ]
                critical_points.append(w)

    return critical_points


def _prepare_symbolic_fraction(F):
    r"""Extract polynomial numerators and denomiators from a symbolic fraction,
    including the replacement of variable names.

    INPUT:

    * ``F`` -- A symbolic fraction
    """
    G, H = F.numerator(), F.denominator()
    original_variables = H.variables()
    if any(v not in original_variables for v in G.variables()):
        raise ValueError(f"Numerator {G} has variables not in denominator {H}")
    new_variables = list(SR.var("acsvvar", len(original_variables)))
    variable_map = {v: new_v for v, new_v in zip(original_variables, new_variables)}
    G = G.subs(variable_map)
    H = H.subs(variable_map)
    frac_gcd = gcd(G, H)
    return G / frac_gcd, H / frac_gcd, variable_map


def _prepare_expanded_polynomial_ring(variables, direction=None, include_t=True):
    r"""Prepare an auxiliary polynomial ring for computing diagonal asymptotics.

    INPUT:

    * ``variables`` -- variables in the rational function `F`
    * ``direction`` -- (Optional) direction vector `r` for the asymptotics,
      defaults to the diagonal (all ones).
    * ``include_t`` -- (Optional) whether to include the auxiliary variable `t`
      in the expanded ring.
    """
    # if direction r is not given, default to the diagonal
    if direction is None:
        direction = [1 for _ in variables]

    replaced_direction = copy(direction)

    # in case there are irrational numbers in the direction vector,
    # replace them with polynomial variables
    direction_variable_values = {}
    for idx, dir_entry in enumerate(replaced_direction):
        if AA(dir_entry).minpoly().degree() > 1:
            dir_var = SR.var(f"r{idx}")
            direction_variable_values[dir_var] = AA(dir_entry)
            replaced_direction[idx] = dir_var

    # auxiliary variables
    auxiliary_variables = []
    if include_t:
        auxiliary_variables.append(SR.var("t"))
    auxiliary_variables.append(SR.var("lambda_"))
    auxiliary_variables.append(SR.var("u_"))

    # create the expanded polynomial ring
    expanded_ring = PolynomialRing(
        QQ, list(variables) + list(direction_variable_values) + auxiliary_variables
    )
    variables = [expanded_ring(v) for v in variables]
    auxiliary_variables = [expanded_ring(v) for v in auxiliary_variables]
    replaced_direction = [expanded_ring(ri) for ri in replaced_direction]
    direction_variable_values = {
        expanded_ring(ri): val for (ri, val) in direction_variable_values.items()
    }
    return (
        expanded_ring,
        variables,
        auxiliary_variables,
        replaced_direction,
        direction_variable_values,
    )
