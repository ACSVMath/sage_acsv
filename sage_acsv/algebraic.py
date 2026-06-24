from functools import cmp_to_key

from sage.geometry.newton_polygon import NewtonPolygon
from sage.all import AA, PolynomialRing, PuiseuxSeriesRing, QQ, QQbar, ZZ, lcm

# Wrapper that calls main function below and converts output to list of Puiseux series prefixes
def compute_separating_puiseux_terms(P, x_var, y_var, max_depth=50, fractional_power_series=False):
    # Error checking
    if set(P.variables()) != {x_var, y_var}: 
        raise ValueError("Only variables in P should be the variables given in the arguments.")
    
    # Compute each Puiseux series solution prefix
    dictionary_branches = _compute_separating_puiseux_terms(P, x_var, y_var, max_depth, fractional_power_series)

    # Build the Puiseux series ring
    PR = PuiseuxSeriesRing(QQbar, str(x_var))
    x_P = PR.gen()
    expansions = []

    # Convert each dictionary into an element of the Puiseux series ring
    for branch in dictionary_branches:
        expr = PR.zero()
        for coeff, exp in branch:
            expr += coeff * x_P^exp
            
        # We don't append O(...) here so that these finite prefixes can be 
        # easily evaluated numerically to sort the branches.
        expansions.append(expr)
        
    return expansions


# Compute only up to the necessary terms for the branches to separate
def _compute_separating_puiseux_terms(P, x_var, y_var, max_depth, fractional_power_series=False):
    if max_depth <= 0:
        raise ValueError("Maximum recursion depth reached. The polynomial might not be square-free.")
        
    # Build polynomial rings and variables to be used below
    Rx = PolynomialRing(QQbar, str(x_var))
    Rxy = PolynomialRing(Rx, str(y_var))
    y_gen = Rxy.gen()

    # Express input polynomial as a polynomial in \overline{Q}[x][y] 
    pol = Rxy(P)
    if pol == 0: return []

    # Compute valuations of coefficients
    vals = {i: pol[i].valuation() for i in range(pol.degree() + 1) if pol[i] != 0}
    if not vals: return []
    
    # Build Newton polygon of P and extract negated slopes 
    pts = [(i, v) for i, v in vals.items()]
    negated_slopes = sorted({-QQ(s) for s in NewtonPolygon(pts).slopes()})
    if fractional_power_series:
        negated_slopes = [a for a in negated_slopes if a >= 0]

    # Iterate through each negated slope
    sols = []
    for alpha in negated_slopes:
        r, num = alpha.denominator(), alpha.numerator()
        
        # Find the lowest coefficient valuation m
        m = min(v * r + num * i for i, v in vals.items())
        
        # Build the characteristic polynomial using only terms on the Newton polygon edge
        char_poly_dict = {i: pol[i][v] for i, v in vals.items() if v * r + num * i == m}
        v_min = min(char_poly_dict.keys())
        Ry = PolynomialRing(QQbar, str(y_var))
        char_poly = Ry({i - v_min: c for i, c in char_poly_dict.items()})
        if char_poly == 0: continue

        # Build transformed polynomial p after mapping x^k -> x^(k*r + num*i - m)
        p_dict = {}
        for i in vals:
            new_x_poly_dict = {k * r + num * i - m: c for k, c in pol[i].dict().items()}
            p_dict[i] = Rx(new_x_poly_dict)
        p = Rxy(p_dict)

        for lc, mult in char_poly.roots():
            if mult == 1:
                current_sol = [(lc, alpha)]
                sols.append(current_sol)
            else:
                # Shift and scale for the next level of recursion
                p_shift = p(y_gen + lc)
                p_final_dict = {}
                for i in range(p_shift.degree() + 1):
                    if p_shift[i] != 0:
                        p_final_dict[i] = Rx({int(k * mult + i): c for k, c in p_shift[i].dict().items()})
                
                # Further shift y -> x*y to resolve singularity
                p_final = Rxy(p_final_dict)

                # Pass max_depth - 1 to prevent infinite loops on non-square-free inputs
                sub_branches = _compute_separating_puiseux_terms(p_final, x_var, y_var, max_depth - 1, True)
                for branch in sub_branches:
                    adj = [(lc, alpha)]
                    for val, exp in branch:
                        if val in QQ: val = QQ(val)
                        adj.append((val, alpha + (exp + 1) / (mult * r)))
                    sols.append(adj)
                    
    return sols


def sort_puiseux_branches(branches, side='auto', real_only=False):
    r"""Sort Puiseux branches in decreasing lexicographic order.

    This follows the real branch sorting described by Chabaud's ``SuiviReel``:
    branches are compared lexicographically by increasing exponent, with larger
    coefficients appearing first.  To sort on the left of a point, use
    ``side='left'``; this applies the substitution `x -> -x` before sorting.

    The default ``side='auto'`` applies that substitution only to branches with
    negative real leading coefficient, matching the convention used by this
    package for selecting a leading branch.

    INPUT:

    * ``branches`` -- A list of Puiseux series elements, as returned by
      :func:`compute_separating_puiseux_terms`.
    * ``side`` -- (default: ``'auto'``) one of ``'auto'``, ``'right'``, or
      ``'left'``.  ``'right'`` sorts the expansions as written, while
      ``'left'`` first applies `x -> -x` to every branch.
    * ``real_only`` -- (default: ``False``) if ``True``, discard branches whose
      transformed coefficients are not all real, as in the real branch tracing
      algorithm.

    OUTPUT:

    The list ``branches`` sorted in place and returned.

    EXAMPLES::

        sage: from sage_acsv.algebraic import compute_separating_puiseux_terms, sort_puiseux_branches
        sage: R.<x, y> = PolynomialRing(QQ, 2)
        sage: P = y^2 - (1 + x)
        sage: branches = compute_separating_puiseux_terms(P, x, y)
        sage: sorted_branches = sort_puiseux_branches(branches)
        sage: sorted_branches
        ...

    """
    if side not in {'auto', 'right', 'left'}:
        raise ValueError("side must be one of 'auto', 'right', or 'left'.")

    def coefficient_cmp(left, right):
        left = QQbar(left)
        right = QQbar(right)
        if left == right:
            return 0

        left_real = bool(left.imag().is_zero())
        right_real = bool(right.imag().is_zero())
        if left_real and right_real:
            return -1 if AA(left.real()) > AA(right.real()) else 1

        left_tuple = (AA(left.real()), AA(left.imag()))
        right_tuple = (AA(right.real()), AA(right.imag()))
        return -1 if left_tuple > right_tuple else 1

    def leading_sign(branch):
        coeffs = branch.coefficients()
        if not coeffs:
            return 0

        lc = QQbar(coeffs[0])
        if not lc.imag().is_zero():
            return 0
        if AA(lc.real()) > 0:
            return 1
        if AA(lc.real()) < 0:
            return -1
        return 0

    def use_left_substitution(branch):
        if side == 'left':
            return True
        if side == 'right':
            return False
        return leading_sign(branch) < 0

    def transformed_terms(branch):
        coeffs = branch.coefficients()
        exponents = [QQ(exp) for exp in branch.exponents()]
        if not coeffs:
            return {}

        if not use_left_substitution(branch):
            return {
                exp: QQbar(coeff)
                for exp, coeff in zip(exponents, coeffs)
            }

        r = ZZ(1)
        for exp in exponents:
            r = lcm(r, exp.denominator())
        w = QQbar.zeta(2 * r)

        terms = {}
        for coeff, expnt in zip(coeffs, exponents):
            coeff = QQbar(coeff)
            if expnt.denominator() == 1:
                coeff *= QQbar((-1) ** ZZ(expnt))
            else:
                coeff *= w ** ZZ(r * expnt)
            terms[expnt] = coeff
        return terms

    transformed_cache = {}

    def cached_terms(branch):
        cache_key = id(branch)
        if cache_key not in transformed_cache:
            transformed_cache[cache_key] = transformed_terms(branch)
        return transformed_cache[cache_key]

    def has_real_terms(branch):
        return all(coeff.imag().is_zero() for coeff in cached_terms(branch).values())

    def branch_cmp(left, right):
        left_terms = cached_terms(left)
        right_terms = cached_terms(right)

        for expnt in sorted(set(left_terms) | set(right_terms)):
            cmp = coefficient_cmp(
                left_terms.get(expnt, QQbar(0)),
                right_terms.get(expnt, QQbar(0)),
            )
            if cmp != 0:
                return cmp

        return 0

    if real_only:
        branches[:] = [branch for branch in branches if has_real_terms(branch)]
    branches.sort(key=cmp_to_key(branch_cmp))
    return branches