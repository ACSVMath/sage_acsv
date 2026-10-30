from __future__ import annotations


from sage.functions.log import exp
from sage.groups.misc_gps.argument_groups import ArgumentByElementGroup
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing, AsymptoticExpansion
from sage.rings.asymptotic.growth_group import (
    ExponentialGrowthGroup,
    MonomialGrowthGroup,
)
from sage.rings.qqbar import QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.expression import Expression
from sage.symbolic.ring import SymbolicRing, SR
from sage.symbolic.operators import add_vararg
from sage.symbolic.constants import pi
from sage.symbolic.relation import solve
from sage_acsv.data.terms import Term, LimitTheoremTerm
from sage_acsv.data.exceptions import ACSVException

def get_expansion_terms(
    expr: tuple | list[tuple] | Expression | AsymptoticExpansion,
) -> list[Term]:
    r"""Determines coefficients for each n^k that appears in the asymptotic expressions
    returned by :func:`.diagonal_asymptotics_combinatorial`.

    INPUT:

    * ``expr`` -- An asymptotic expression, symbolic expression, ACSV tuple, or list of ACSV tuples

    OUTPUT:

    A list of :class:`.Term` objects (with attributes ``coefficient``, ``pi_factor``,
    ``base`` and ``power``), each representing a summand in the fully expanded expression.

    EXAMPLES::

        sage: from sage_acsv import diagonal_asymptotics_combinatorial, get_expansion_terms
        sage: var('x y z')
        (x, y, z)
        sage: res = diagonal_asymptotics_combinatorial(1/(1 - x - y), r=[1,1], expansion_precision=2)
        sage: coefs = sorted(get_expansion_terms(res), reverse=True)
        sage: coefs
        [Term(coefficient=1, pi_factor=1/sqrt(pi), base=4, power=-1/2),
         Term(coefficient=-1/8, pi_factor=1/sqrt(pi), base=4, power=-3/2)]
        sage: res = diagonal_asymptotics_combinatorial(1/(1 - x - y), r=[1,1], expansion_precision=2, output_format="tuple")
        sage: sorted(get_expansion_terms(res)) == sorted(coefs)
        True
        sage: res = diagonal_asymptotics_combinatorial(1/(1 - x - y), r=[1,1], expansion_precision=2, output_format="symbolic")
        sage: sorted(get_expansion_terms(res)) == sorted(coefs)
        True

    ::

        sage: res = diagonal_asymptotics_combinatorial(1/(1 - x^7))
        sage: get_expansion_terms(res)
        [Term(coefficient=1/7, pi_factor=1, base=0.6234898018587335? + 0.7818314824680299?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=0.6234898018587335? - 0.7818314824680299?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=-0.2225209339563144? + 0.9749279121818236?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=-0.2225209339563144? - 0.9749279121818236?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=-0.9009688679024191? + 0.4338837391175582?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=-0.9009688679024191? - 0.4338837391175582?*I, power=0),
         Term(coefficient=1/7, pi_factor=1, base=1, power=0)]

    ::

        sage: res = diagonal_asymptotics_combinatorial(1/(1 - x - y^2))
        sage: coefs = get_expansion_terms(res); coefs
        [Term(coefficient=0.6123724356957945?, pi_factor=1/sqrt(pi), base=-2.598076211353316?, power=-1/2),
         Term(coefficient=0.6123724356957945?, pi_factor=1/sqrt(pi), base=2.598076211353316?, power=-1/2)]
        sage: coefs[0].coefficient.parent()
        Algebraic Field
        sage: coefs[0].coefficient.radical_expression()
        1/2*sqrt(3/2)

    ::

        sage: F2 = (1+x)*(1+y)/(1-z*x*y*(x+y+1/x+1/y))
        sage: res = diagonal_asymptotics_combinatorial(F2, expansion_precision=3)
        sage: coefs = get_expansion_terms(res); coefs
        [Term(coefficient=4, pi_factor=1/pi, base=4, power=-1),
         Term(coefficient=1, pi_factor=1/pi, base=-4, power=-3),
         Term(coefficient=-6, pi_factor=1/pi, base=4, power=-2),
         Term(coefficient=19/2, pi_factor=1/pi, base=4, power=-3)]

    ::

        sage: res = diagonal_asymptotics_combinatorial(3/(1 - x))
        sage: get_expansion_terms(res)
        [Term(coefficient=3, pi_factor=1, base=1, power=0)]

    ::

        sage: res = diagonal_asymptotics_combinatorial((x - y)/(1 - x - y))
        sage: get_expansion_terms(res)
        []

    """
    n = SR.var("n")
    if isinstance(expr, tuple):
        expr = SR(expr[0] ** n * prod(expr[1:]))
    elif isinstance(expr, list):
        expr = SR(sum([tup[0] ** n * prod(tup[1:]) for tup in expr]))
    elif isinstance(expr.parent(), AsymptoticRing):
        expr = expr.exact_part()
        symbolic_expr = SR.zero()
        for summand in expr.summands:
            symbolic_summand = summand.coefficient
            for factor in summand.growth.value:
                if isinstance(factor.parent(), MonomialGrowthGroup):
                    symbolic_summand *= n**factor.exponent
                elif isinstance(factor.parent(), ExponentialGrowthGroup):
                    if isinstance(factor.base.parent(), ArgumentByElementGroup):
                        symbolic_summand *= factor.base._element_**n
                    else:
                        symbolic_summand *= factor.base**n
            symbolic_expr += symbolic_summand
        expr = symbolic_expr

    if not isinstance(expr.parent(), type(SR)):
        raise ACSVException(f"Cannot deal with expression of type {expr.parent()}")

    if len(expr.args()) > 1:
        raise ACSVException("Cannot process multivariate symbolic expression.")

    # If expression is the sum of a bunch of terms, handle each one separately
    expr = expr.expand()
    if expr.is_zero():
        return []

    terms = [expr]
    if expr.operator() == add_vararg:
        terms = expr.operands()

    decomposed_terms = []
    for summand in terms:
        term = Term(
            coefficient=QQ.one(), pi_factor=QQ.one(), base=QQ.one(), power=QQ.zero()
        )
        if summand in QQbar:
            term.coefficient = QQbar(summand)
            decomposed_terms.append(term)
            continue

        for v in summand.operands():
            if n in v.args():
                if v.degree(n) != 0:
                    term.power += v.degree(n)
                else:
                    term.base *= v.operands()[0]
            elif v.degree(pi) != 0:
                pi_deg = v.degree(pi)
                term.pi_factor *= pi**pi_deg
                term.coefficient *= v.coefficient(pi**pi_deg)
            else:
                term.coefficient *= v

        for attr in ("coefficient", "base", "power"):
            elem = getattr(term, attr)
            if isinstance(elem.parent(), SymbolicRing) and elem in QQbar:
                setattr(term, attr, QQbar(elem))

        decomposed_terms.append(term)

    return decomposed_terms

def get_limit_theorem_terms(
    expr,
) -> list[LimitTheoremTerm]:
    r"""Determines coefficients for each term in the asymptotic expressions
    returned by :func:`~sage_acsv.asymptotics.central_limit_theorem_combinatorial`,
    including density-related information.

    INPUT:

    * ``expr`` -- The return value of :func:`~sage_acsv.asymptotics.central_limit_theorem_combinatorial`,
      either a 6-tuple ``(base, n^b, pi^b, constant, invHess, mean_vector)`` or a symbolic expression
      (when ``as_symbolic=True`` was used).

    OUTPUT:

    A list of :class:`.LimitTheoremTerm` objects (with attributes ``coefficient``, ``pi_factor``,
    ``base``, ``power``, ``density_symbolic``, ``density_covariance_matrix``, and
    ``density_mean_vector``), each representing a summand in the fully expanded
    asymptotic expression with associated density information.

    EXAMPLES::

        sage: from sage_acsv import central_limit_theorem_combinatorial
        sage: from sage_acsv.helpers import get_limit_theorem_terms
        sage: var('z t')
        (z, t)
        sage: expr = central_limit_theorem_combinatorial(1/(1-t-z*t^2), t, as_symbolic=True)
        sage: get_limit_theorem_terms(expr)[0]
        LimitTheoremTerm(coefficient=1.710862642974252?, pi_factor=1/sqrt(pi), base=1.618033988749895?, power=-0.5, density_symbolic=e^(-1/2*(-0.2763932022500211?*n + s0)*(-3.090169943749475?*n + 11.18033988749895?*s0)/n), density_covariance_matrix=[11.18033988749895?], density_mean_vector=[0.2763932022500211?])
    """
    n = SR.var("n")

    if isinstance(expr, tuple):
        base_val, n_power, pi_power, constant, inv_hess, mean_vec = expr
        growth_expr = SR(base_val ** n * n_power * pi_power * constant)
        d_minus_1 = inv_hess.nrows()
        s = matrix((SR.var('s', n=d_minus_1)))
        density_sym = exp(-(((s - n * mean_vec) * inv_hess * (s - n * mean_vec).transpose())[0, 0]) / 2 / n)
        base_terms = get_expansion_terms(growth_expr)

        result = []
        for term in base_terms:
            lterm = LimitTheoremTerm(
                coefficient=term.coefficient,
                pi_factor=term.pi_factor,
                base=term.base,
                power=term.power,
                density_symbolic=density_sym,
                density_covariance_matrix=inv_hess,
                density_mean_vector=mean_vec,
            )
            result.append(lterm)

        return result

    elif isinstance(expr, Expression):
        density_sym = None
        growth_operands = []

        if expr.operator() is not None:
            for op in expr.operands():
                op_str = str(op)
                if 'e^' in op_str or op_str.startswith('exp('):
                    density_sym = op
                else:
                    growth_operands.append(op)

        if density_sym is not None:
            exponent_expr = density_sym.operands()[0] * n
            zero_subbed_exp = exponent_expr.subs(n=0)
            s_vars = zero_subbed_exp.variables()
            cov_matrix = -matrix([
                [zero_subbed_exp.derivative(v1, v2) for v2 in s_vars] for v1 in s_vars
            ])

            if len(s_vars) > 0:
                try:
                    grad_n = matrix([[exponent_expr.derivative(v).derivative(n) for v in s_vars]])
                    mean_vec = grad_n * cov_matrix.inverse()
                except Exception:
                    one_subbed_exp = exponent_expr.subs(n=1)
                    eqs = [one_subbed_exp.derivative(v) == 0 for v in s_vars]
                    sols = solve(eqs, list(s_vars), solution_dict=True)
                    if sols:
                        mean_vec = matrix([[sols[0][v] for v in s_vars]])
                    else:
                        mean_vec = matrix([[0] * len(s_vars)])
            else:
                mean_vec = matrix([[0] * len(s_vars)])
        else:
            cov_matrix = matrix([])
            mean_vec = matrix([])

        if density_sym is not None and growth_operands:
            growth_expr = prod(growth_operands)
        elif density_sym is None:
            growth_expr = expr
        else:
            growth_expr = SR.one()

        base_terms = get_expansion_terms(growth_expr)

        result = []
        for term in base_terms:
            lterm = LimitTheoremTerm(
                coefficient=term.coefficient,
                pi_factor=term.pi_factor,
                base=term.base,
                power=term.power,
                density_symbolic=density_sym,
                density_covariance_matrix=cov_matrix,
                density_mean_vector=mean_vec,
            )
            result.append(lterm)

        return result
    else:
        raise ACSVException(f"Cannot process expression of type {type(expr)}")