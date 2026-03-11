from sage.misc.misc_c import prod
from sage.rings.fraction_field import FractionField
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ

from sage_acsv.helpers import algebraic_residues, pure_composed_sum


def algebraic_diagonal(f, params=[], output_vars=None):
    r"""Compute annihilating polynomial for the (main) diagonal of a bivariate rational function.
    Taken from Algorithm 3 of Bostan, Dumont, and Salvy (2017).

    INPUT:

    * ``f``            -- A symbolic rational function with two variables (can have additional symbolic parameters)
    * ``params``       -- (optional) List of variables in f that should be considered parameters
    * ``output_vars``  -- (optional) Variables for the resulting polynomial (if None then output is polynomial in t and y)

    OUTPUT:

    A polynomial P(t,y) in K[t][y] such that the diagonal d(t) of f satisfies P(t,d(t)) = 0,
    where output_vars = (t,y). Note, in particular, that the dependent variable for the diagonal
    is the first element of output_vars.
    """

    # TODO: Verify local variable names don't clash with input variables
    # TODO: Take more care with variable names in final section of code
    # TODO: Fix case with parameters (need to use proof.WithProof('polynomial', False) when factoring) 
    
    # Extract variables and define ring structure
    f_variables = [v for v in f.variables() if v not in params]

    if len(params) == 0:
        K = QQ
    else:
        raise NotImplementedError("The case with parameters is still in progress")
        
    R = PolynomialRing(K, names = f_variables)

    # Verify f has two (non-parameter) variables
    if len(f_variables) != 2:
        raise ValueError("Input f needs to have exactly 2 non-parameter variables")
        
    local_x, local_y = R.gens()

    # Verify f is a bivariate rational function (possibly with parameters)
    try:
        A, B = R(f.numerator()), R(f.denominator())
    except:
        raise ValueError("Input f not a rational function in all variables and parameters")

    # Compute needed quantities
    dA = max(i - j for (i,j) in A.dict())
    dB = max(i - j for (i,j) in B.dict())
    a = dB - dA - 1

    # Compute number of branches that go to 0 as the dependent variable goes to 0
    pol = R(local_y**dB * B(local_x/local_y, local_y))
    pol_star = prod([p for (p,_) in pol.factor()])
    pol_star_lowest = pol_star.polynomial(local_x)[pol_star.polynomial(local_x).valuation()]
    c = pol_star_lowest.polynomial(local_y).valuation()

    # Define quantities we use to compute residues
    Kt = FractionField(PolynomialRing(K, 't'))
    t = Kt.gen()
    Kt_Y = PolynomialRing(Kt, 'Y')
    Y = Kt_Y.gen()
    P = Y**dA * A(t/Y, Y)
    Q = Y**dB * B(t/Y, Y)
    
    Pt = Kt_Y(P)
    Qt = Kt_Y(Q)

    # Compute residues -- note formula in Bostan, Dumont, and Salvy (2017) has typo when a < 0
    if a < 0:
        R = algebraic_residues(Pt, Y**(-a) * Qt, Qt)
    else:
        R = algebraic_residues(Y**a * Pt, Qt, Qt)
    
    T = PolynomialRing(R.parent().base_ring(), 'Z')
    Z = T.gen()
    R = R(Z)
    
    Phi = (pure_composed_sum(R,c,'Z')).numerator()
    
    if a < 0:
        r = algebraic_residues(Pt, Y**(-a)*Qt, Y**(-a))
        r = Phi.parent()(r)
    
    # Determine desired output variables
    y = Phi.parent()(local_y)
    if output_vars == None:
        output_t, output_y  = (t, y)
    elif len(output_vars) == 2:
        output_t, output_y  = output_vars
    else:
        raise ValueError("output_vars needs to specify exactly 2 variables")

    Output_Ring = PolynomialRing(PolynomialRing(QQ, output_t), output_y)

    # Convert to the desired output variables and return result
    if a < 0:
        return Output_Ring(Phi(r).numerator().subs(t=output_t, y=output_y))
    else:
        return Output_Ring(Phi.subs(t=output_t, y=output_y))
