from sage.arith.misc import binomial
from sage.functions.other import factorial
from sage.misc.misc_c import prod
from sage.rings.big_oh import O
from sage.rings.fraction_field import FractionField
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing


def algebraic_residues(P, Q, S):
    r"""Compute annihilating polynomial of the residues of a rational function at the zeroes of another polynomial.

    Taken from Algorithm 1 of Bostan, Dumont, and Salvy (2017).

    INPUT:

    * ``P``      -- Polynomial in K[y].
    * ``Q``      -- Polynomial in K[y] with non-zero constant.
    * ``S``      -- Polynomial in K[y] that is a divisor of Q such that S and Q/S are coprime (S can be Q).

    OUTPUT:

    A polynomial R(z) in K[z] that annihilates the residues of P/Q at the roots of S.
    """
    # Build needed Sage objects
    R = P.parent()                                 # R = K[y]
    y = R.gen()
    B = PowerSeriesRing(FractionField(R), 't')  # B = K(y)[[t]]
    t = B.gen()

    # Compute square-free decomposition of S
    decomp = list(S.squarefree_decomposition())
    m = len(decomp)
    Result = [1] * m

    # Iterate over factors in square-free decomposition
    for i in range(m):
        # decomp[i] = (Q_{i+1}, d_{i+1})
        Qi = decomp[i][0]
        di = decomp[i][1]

        if Qi.degree() == 0:
            Result[i] = 1                                # Ri = 1
        else:
            Ui = Q // Qi**di                        # Ui = Q/Q_i^i
            Vi = (Qi(y + t) - Qi(y)).quo_rem(t)[0]    # Vi = (Qi(y+t)-Qi(y))/t

            # Take coefficient Si = [t^(di-1)] P(y+t) / (Ui(y+t) * Vi^di)
            # by expanding the rational function as a power series to O(t^di)
            Di = Ui(y + t) * Vi**di
            Yi = P(y + t) / (Di + O(t**di))
            Si = Yi[di - 1]

            # Extract numerator and denominator of Si, and make sure they are coprime
            Ai = Si.numerator()
            Bi = Si.denominator()
            G = Ai.gcd(Bi)
            Ai = Ai // G
            Bi = Bi // G

            # Create polynomial rings to perform resultant computation
            Lz = PolynomialRing(R.base_ring(), 'z')     # Lz = K[z]
            z = Lz.gen()
            Ly = PolynomialRing(Lz, 'y')                # Ly = K[z][y]
            y = Ly.gen()

            # Compute desired resultant for the term in the square-free decomposition
            H = Ly(Ai) - z * Ly(Bi)
            Result[i] = H.resultant(Ly(Qi))  # Resultant taken over y, so result is in K[z]

    return prod(Result[i] for i in range(m)).monic()  # Other formulas require our output to be monic


def pure_composed_sum(P, c, var):
    r"""Compute annihilating polynomial for all sums of c roots of a polynomial.
    Taken from Algorithm 2 of Bostan, Dumont, and Salvy (2017).

    INPUT:

    * ``P``      -- A univariate or multivariate polynomial WITHOUT y and z among its variables
    * ``c``      -- An integer between 1 and deg(P) (inclusive)
    * ``var``    -- A string or variable specifying a variable that appears in P

    OUTPUT:

    A polynomial (Σ_cP)(y) in K[y] whose roots are the sums of all c-subsets of the roots of P.
    """

    # TODO: Add tests that var is a variable of P and that z and y do not appear in P
    # TODO: Fix overloading of Y and y in univariate and multivariate rings

    # Convert P to a polynomial in var
    # K is the polynomial ring in the remaining variables
    P_var = P.polynomial(P.parent()(var))
    K = P_var.base_ring()
    d = P_var.degree()
    D = binomial(d, c)

    # Create rings needed for our computations
    R1 = PolynomialRing(K, names=(var, 'z'))
    R2 = PowerSeriesRing(K, 'Y')
    Y = R2.gen()
    y, z = R1.gens()

    # Cast P into K[[Y]]
    P_Y = P_var(Y)
    P_Y_der = P_Y.derivative()

    # Compute reciprocal polynomials of P and P' (polynomials with reversed coefficient sequences)
    recP = Y**d * P_Y(1 / Y)
    recP_der = Y**(d - 1) * P_Y_der(1 / Y)

    # Introduce bivariate power series ring
    R3 = PowerSeriesRing(K, names=('Y', 'Z'))
    Y, Z = R3.gens()

    # Compute terms in generating function for Newton Sum of P and take Hadamard product with exp(Y)
    # Note: for now S lies in K[[Y,Z]] even though it depends only on Y
    NP = recP_der / (recP + recP.parent()(O(Y**(D + 1)))) # convert everything to same ring for sage versions <= 10.4
    S = sum((NP[n] / factorial(n)) * Y**n for n in range(D + 1))

    # Compute intermediate sum, take its exponential, and extract desired term
    Sum = sum((-1)**(n - 1) * (S(n * Y, 0) / n) * Z**n
              for n in range(1, c + 1))
    F = (Sum + O(Y**(c + D + 1))).exp()
    Fc = sum(C * y**e[0] for e, C in F.polynomial().dict().items()
             if e[0] < D + 1 and e[1] == c)

    # Compute the Newton Sum of Σ_cP and cast into K[[Y]]
    NSumP = sum(Fc[n, 0] * factorial(n) * Y**n for n in range(D + 1))
    NSumP_Y = R2(NSumP)

    # Recover Σ_cP by integrating and taking an exponential
    integrand = (D - NSumP_Y) / R2(Y)
    IntegralEXP = (integrand.integral() + O(R2(Y)**(D + 1))).exp()
    U = IntegralEXP.truncate(D + 1)(y)
    recU = y**(U.degree()) * U(1 / y, 0)

    R4 = PolynomialRing(K, 'y')
    y = R4.gen()
    return R4(recU(y, 0))
