from sage.geometry.newton_polygon import NewtonPolygon

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
            # =========================================================
            # MODIFICATION 1: Simple root case
            # Branches have separated! We just record this term and stop.
            # =========================================================
            if mult == 1:
                current_sol = [(lc, alpha)]
                sols.append(current_sol)
                
            # =========================================================
            # MODIFICATION 2: Multiple root case
            # Branches share this prefix. We must recurse until mult == 1.
            # =========================================================
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


def sort_puiseux_branches(branches, test_point=QQ(1)/1000):
    r"""Sort Puiseux branches by evaluation at a small positive real point.

    Branches are ordered so that complex-valued branches (those with nonzero
    imaginary part at the test point) come first, followed by real-valued
    branches sorted in ascending order by their real part.  The **dominating
    branch** (i.e. the largest real branch) therefore appears last.

    INPUT:

    * ``branches`` -- A list of Puiseux series elements, as returned by
      :func:`compute_separating_puiseux_terms`.
    * ``test_point`` -- (default: ``1/1000``) A small positive rational number
      at which each branch is evaluated numerically.

    OUTPUT:

    The list ``branches`` sorted in place and returned.

    EXAMPLES::

        sage: from sage_acsv.algebraic import compute_separating_puiseux_terms, sort_puiseux_branches
        sage: R.<x, y> = PolynomialRing(QQ, 2)
        sage: P = y^2 - (1 + x)
        sage: branches = compute_separating_puiseux_terms(P, x, y)
        sage: sorted_branches = sort_puiseux_branches(branches)
        sage: sorted_branches[-1]  # dominating branch is last
        ...

    """
    def sort_key(branch):
        # Evaluate the Puiseux prefix at the test point.
        # Use AA(test_point) so that nth_root works for fractional exponents.
        val = QQbar(branch(AA(test_point)))

        # Determine whether the branch is real-valued at the test point
        is_real = bool(val.imag().is_zero())

        # Real branches are sorted after complex ones; among real branches
        # sort by real value ascending so the largest is last.
        return (is_real, AA(val.real()) if is_real else AA(0))

    branches.sort(key=sort_key)
    return branches


def newton_series_dominating(P, x_var, y_var, sorted_branches, series_precision):
    r"""Compute the power series expansion of the dominating branch via Newton's method.

    The dominating branch is the last element of ``sorted_branches`` (the
    largest real branch, as returned by :func:`sort_puiseux_branches`).
    The full Puiseux prefix is used to shift `P` so the branch passes
    through the origin, fractional exponents are rationalized via
    `x = t^e`, and :func:`~sage_acsv.helpers.compute_newton_series`
    extends the expansion.

    INPUT:

    * ``P`` -- A bivariate polynomial `P(x, y) = 0` defining the algebraic curve.
    * ``x_var`` -- The independent variable `x`.
    * ``y_var`` -- The dependent variable `y`.
    * ``sorted_branches`` -- A list of Puiseux series prefixes sorted by
      :func:`sort_puiseux_branches`.  The last element is taken as the
      dominating branch.
    * ``series_precision`` -- A positive integer giving the number of terms
      to compute.

    OUTPUT:

    A polynomial representing the power series expansion of the dominating
    branch.  When the ramification index is 1 the result is a polynomial
    in ``x_var``; otherwise it is a polynomial in an auxiliary variable
    ``t`` where `x = t^e`.

    EXAMPLES::

        sage: from sage_acsv.algebraic import compute_separating_puiseux_terms, sort_puiseux_branches, newton_series_dominating
        sage: R.<x, y> = PolynomialRing(QQ, 2)
        sage: P = x*y^2 - y + 1
        sage: branches = sort_puiseux_branches(compute_separating_puiseux_terms(P, x, y))
        sage: newton_series_dominating(P, x, y, branches, 7)
        132*x^6 + 42*x^5 + 14*x^4 + 5*x^3 + 2*x^2 + x + 1

    """
    from sage_acsv.helpers import compute_newton_series

    # The dominating branch is the last (largest real) branch
    dominating = sorted_branches[-1]

    # Extract coefficients and exponents from the Puiseux prefix
    coeffs = dominating.coefficients()
    exponents = dominating.exponents()

    # Compute the ramification index e (LCD of exponent denominators).
    # When e > 1 the branch has fractional exponents; we substitute
    # x = t^e to obtain integer exponents.
    e = ZZ(1)
    for exp in exponents:
        e = lcm(e, QQ(exp).denominator())

    # Use the original variable name when e == 1, otherwise introduce t
    t_name = str(x_var) if e == 1 else 't'

    # Build a flat multivariate polynomial ring over QQbar.
    # compute_newton_series uses Ideal().mod() which requires a true
    # multivariate ring, not a nested univariate ring like QQbar[x][y].
    Rty = PolynomialRing(QQbar, [t_name, str(y_var)])
    t_gen, y_gen = Rty.gens()

    # Convert P(x, y) -> P(t^e, y) by rewriting each monomial
    P_xy = PolynomialRing(QQbar, [str(x_var), str(y_var)])(P)
    P_ty = Rty.zero()
    for monom, c in P_xy.dict().items():
        P_ty += QQbar(c) * t_gen ** ZZ(monom[0] * e) * y_gen ** ZZ(monom[1])

    # Build the known Puiseux prefix as a polynomial in t
    prefix = Rty.zero()
    for c, exp in zip(coeffs, exponents):
        prefix += QQbar(c) * t_gen ** ZZ(exp * e)

    # Shift: P(t^e, y + prefix(t)) so the dominating branch goes through origin
    P_shifted = P_ty.subs({y_gen: y_gen + prefix})

    # We need enough terms in t to cover series_precision terms in x
    t_precision = series_precision * e

    # Run Newton's method on the shifted polynomial
    series_t = compute_newton_series(P_shifted, [t_gen, y_gen], t_precision)

    # Shift back: add the known prefix
    return series_t + prefix