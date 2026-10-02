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


from sage.misc.misc_c import prod
from sage.rings.ideal import Ideal
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import AA
from sage.rings.rational_field import QQ
from sage.symbolic.ring import SR

from sage_acsv.asymptotic_terms import (
    _compute_asymptotics_at_points,
    _compute_asymptotics_at_points_hyperplane,
    _compute_asymptotics_at_points_smooth
)
from sage_acsv.critical_points import (
    contributing_points_combinatorial_smooth,
    _find_contributing_points_combinatorial,
    contributing_points_hyperplane
)
from sage_acsv.debug import acsv_logger
from sage_acsv.helpers import (
    ACSVException,
    rational_function_reduce,
)
from sage_acsv.settings import ACSVSettings
from sage_acsv.helpers.utils import (
    _prepare_symbolic_fraction, 
    _dict_to_variable_order
)

# we need to monkeypatch a function from the asymptotics module such that creating
# asymptotic expansions over QQbar is possible. this should be removed once the
# upstream issue is resolved.

import sage.rings.asymptotic.misc as asy_misc

if asy_misc.strip_symbolic("acsv_test_defined") != "defined":
    asy_misc.strip_symbolic_original = asy_misc.strip_symbolic
    def strip_symbolic(expression):
        if expression == "acsv_test_defined":
            return "defined"
        expression = asy_misc.strip_symbolic_original(expression)
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
        raise ACSVException(f"Could not find suitable linear form after {ACSVSettings.MAX_MIN_CRIT_RETRIES} attempts.")

    result = _compute_asymptotics_at_points_smooth(
        G, H, vs, r, min_crit_pts, expansion_precision, output_format
    )

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
    * ``r`` -- (Optional) A vector or dictionary of length `d` of positive algebraic numbers.
      Defaults to the appropriate vector of all 1's if not specified.
      If a vector is given, assumes the variable order is given by ``F.variables()``.
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

    The argument ``"terms"`` lets the function return a list of Term objects. See ``sage_acsv.helpers.Term`` for
    a type definition. The output given here should match that of using :func:`.get_expansion_terms`::

        sage: from sage_acsv import get_expansion_terms
        sage: growth_symbolic = diagonal_asymptotics_combinatorial(1/(1 - x - y), output_format="asymptotic")
        sage: growth_terms = diagonal_asymptotics_combinatorial(1/(1 - x - y), output_format="terms"); growth_terms
        [Term(coefficient=1, pi_factor=1/sqrt(pi), base=4, power=-1/2)]
        sage: get_expansion_terms(growth_symbolic) == growth_terms
        True

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
        12 + O(0.9960121882524521?^n*n^(-1/2))

        sage: diagonal_asymptotics_combinatorial(1/((1-(2*x+y)/3)*(1-(3*x+y)/4)), r = [17/24, 7/24], output_format = 'asymptotic')
        12 + O(0.9960121882524521?^n*n^(-1/2))
        sage: G = (1+x)*(1-x*y^2+x^2)
        sage: H = (1-z*(1+x^2+x*y^2))*(1-y)*(1+x^2)
        sage: strat = [
        ....:     [(1-z*(1+x^2+x*y^2), 1-y, 1+x^2)],
        ....:     [(1-z*(1+x^2+x*y^2), 1-y),(1-z*(1+x^2+x*y^2), 1+x^2),(1-y,1+x^2)],
        ....:     [(H,)],
        ....: ]
        sage: diagonal_asymptotics_combinatorial(G/H, r = [1,1,1], output_format = 'asymptotic', whitney_strat = strat)
        0.866025403784439?/sqrt(pi)*3^n*n^(-1/2) + O(3^n*n^(-3/2))
        sage: diagonal_asymptotics_combinatorial(G/H, r = [1,1,1], output_format = 'asymptotic', whitney_strat = strat, expansion_precision = 2)
        0.866025403784439?/sqrt(pi)*3^n*n^(-1/2) - 1.136658342467076?/sqrt(pi)*3^n*n^(-3/2) + O(3^n*n^(-5/2))

    An example containing a complex contributing point carrying a phase.
    
        sage: from sage_acsv import get_expansion_terms
        sage: P = 2 - 2*x - 2*y + 6*x*y + x^2 + y^2
        sage: F = (1/(1 - x - y) + 1/(2*P))/(1 - z)
        sage: terms = get_expansion_terms(diagonal_asymptotics_combinatorial(F))
        sage: c = [t.coefficient for t in terms if t.base == 4*QQbar.zeta(3)^2][0]; c
        0.1834862281267339? + 0.04916498664879108?*I
        sage: c == QQbar.zeta(24)/(4*QQbar(3)^(1/4))
        True

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

    Check that direction can be passed as dictionary::

        sage: var('y T x')
        (y, T, x)
        sage: diagonal_asymptotics_combinatorial(1/(1 - x - 2*y - 3*T), r = {T:2, x:1, y:3})
        1/2/pi*31104^n*n^(-1) + O(31104^n*n^(-2))

    """
    if isinstance(r, dict):
        r = _dict_to_variable_order(F, r)

    if as_symbolic:
        acsv_logger.warning(
            "The as_symbolic argument has been deprecated in favor of output_format='symbolic'"
        )
        output_format = ACSVSettings.Output.SYMBOLIC

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
    vs = [R(v) for v in vs]

    # Make sure G and H are coprime, and that H does not vanish at 0
    G, H = rational_function_reduce(G, H)
    G, H = R(G), R(H)
    if H.subs({v: 0 for v in H.variables()}) == 0:
        raise ValueError("Denominator vanishes at 0.")

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
    if all(f.degree() == 1 for f, _ in R(H).factor()):
        return diagonal_asymptotics_hyperplane(
            F,
            r=r,
            linear_form=linear_form,
            expansion_precision=expansion_precision,
            return_points=return_points,
            output_format=output_format,
        )

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

    result = _compute_asymptotics_at_points(
        G, H, vs, r, min_crit_pts, expansion_precision, output_format
    )

    if return_points:
        return result, min_crit_pts

    return result


def diagonal_asymptotics_hyperplane(
    F,
    r=None,
    linear_form=None,
    expansion_precision=1,
    return_points=False,
    output_format=None,
):
    r"""Asymptotic behavior of the coefficient array of a multivariate rational
    function `F` whose denominator `H` can be factored into linear factors. Note that this function
    does not require `F` to be combinatorial.

    INPUT:

    * ``F`` -- The rational function `G/H` in `d` variables.
    * ``r`` -- (Optional) A vector or dictionary of length `d` of positive algebraic numbers.
      Defaults to the appropriate vector of all 1's if not specified.
      If a vector is given, assumes the variable order is given by ``F.variables()``.
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

    OUTPUT:

    A representation of the asymptotic behavior of the coefficient array of `F` along
    the specified direction.

    NOTE:

    The code randomly generates a linear form, which for generic rational functions
    separates the solutions of an intermediate polynomial system with high probability.
    This separation step can fail, but (assuming `F` has a finite number of critical
    points) the code can be rerun until a separating form is found.

    EXAMPLES::

        sage: from sage_acsv import diagonal_asymptotics_hyperplane
        sage: var('x, y')
        (x, y)
        sage: diagonal_asymptotics_hyperplane(1/(1-x-y))
        1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))

    Non-smooth combinatorial example::

        sage: diagonal_asymptotics_hyperplane(1/((1-x/3-2*y/3)*(1-2*x/3-y/3)))
        3 + O((8/9)^n*n^(-1/2))
        sage: diagonal_asymptotics_hyperplane(1/((1-x/3-2*y/3)*(1-2*x/3-y/3)), r=[3,1])
        6.531972647421808?/sqrt(pi)*(2048/2187)^n*n^(-1/2) + O((2048/2187)^n*n^(-3/2))
        sage: diagonal_asymptotics_hyperplane(SR(1/((3-2*x-y)*(3-x-2*y)*(1-x/4-y/4))))
        1/3 + O((8/9)^n*n^(-1/2))

    Non-combinatorial example::

        sage: diagonal_asymptotics_hyperplane(1/(1+x+y))
        1/sqrt(pi)*4^n*n^(-1/2) + O(4^n*n^(-3/2))
        sage: diagonal_asymptotics_hyperplane(1/(1+x+y), r=[2,1])
        0.866025403784439?/sqrt(pi)*(27/4)^n*n^(-1/2)*(e^(I*arg(-1)))^n + O((27/4)^n*n^(-3/2))

    Non-smooth non-combinatorial non-smooth example::

        sage: diagonal_asymptotics_hyperplane(1/((1+x/3+2*y/3)*(1-2*x/3-y/3)))
        8/9/sqrt(pi)*(8/9)^n*n^(-1/2) + O((8/9)^n*n^(-3/2))

    Should fail if H is not a hyperplane arrangement::

        sage: diagonal_asymptotics_hyperplane(1/(1-x^2-y^2))
        Traceback (most recent call last):
        ...
        ValueError: H does not define a hyperplane arrangement.

    Should fail if H is not transverse::

        sage: diagonal_asymptotics_hyperplane(SR(1/((3-2*x-y)*(3-x-2*y)*(2-x-y))))
        Traceback (most recent call last):
        ...
        ACSVException: Not a transverse intersection. Cannot deal with this case.

    """
    if isinstance(r, dict):
        r = _dict_to_variable_order(F, r)

    if r is None:
        n = len(F.variables())
        r = [1 for _ in range(n)]

    try:
        r = [QQ(ri) for ri in r]
    except (ValueError, TypeError):
        r = [AA(ri) for ri in r]

    G, H, variable_map = _prepare_symbolic_fraction(F)
    vs = list(variable_map.values())
    R = PolynomialRing(QQ, vs)
    vs = [R(v) for v in vs]


    # Make sure G and H are coprime, and that H does not vanish at 0
    G, H = rational_function_reduce(G, H)
    G, H = R(G), R(H)
    Hs = [f for f, _ in H.factor()]
    if H.subs({v: 0 for v in H.variables()}) == 0:
        raise ValueError("Denominator vanishes at 0.")
    if any(f.degree() > 1 for f in Hs):
        raise ValueError("H does not define a hyperplane arrangement.")

    for _ in range(ACSVSettings.MAX_MIN_CRIT_RETRIES):
        try:
            minimal_contributing_points, next_contrib_vals = contributing_points_hyperplane(
                G, H, vs, r, linear_form=linear_form
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

    result = _compute_asymptotics_at_points_hyperplane(
        G, H, vs, r, minimal_contributing_points, next_contrib_vals, expansion_precision, output_format
    )

    if return_points:
        return result, minimal_contributing_points

    return result

