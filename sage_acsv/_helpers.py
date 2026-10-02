"""Miscellaneous mathematical helper functions."""

from __future__ import annotations

from dataclasses import dataclass

from sage.arith.misc import binomial, gcd
from sage.functions.generalized import kronecker_delta
from sage.functions.log import exp
from sage.functions.other import ceil, factorial
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.groups.misc_gps.argument_groups import ArgumentByElementGroup
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.misc.prandom import randint
from sage.modules.free_module_element import vector
from sage.rings.asymptotic.asymptotic_ring import AsymptoticRing, AsymptoticExpansion
from sage.rings.big_oh import O
from sage.rings.asymptotic.growth_group import (
    ExponentialGrowthGroup,
    MonomialGrowthGroup,
)
from sage.rings.fraction_field import FractionField
from sage.rings.ideal import Ideal
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.qqbar import AA, AlgebraicNumber, QQbar
from sage.rings.rational_field import QQ
from sage.symbolic.expression import Expression
from sage.symbolic.ring import SymbolicRing, SR
from sage.symbolic.operators import add_vararg
from sage.symbolic.constants import pi
from sage.symbolic.relation import solve















