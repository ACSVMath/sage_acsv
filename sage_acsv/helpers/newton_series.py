from sage.functions.other import ceil
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector

def compute_newton_series(phi, variables, series_precision):
    r"""Computes the series expansion of an implicitly defined function.

    The function `g(x)` for which a series expansion is computed is a simple root of the expression

    .. math::

        \Phi(x, g(x)) = 0

    INPUT:

    * ``phi`` -- A polynomial; the equation defining the function that is expanded.
    * ``variables`` -- A list of variables in the equation. The last variable in this
      list is the variable corresponding to `g(x)`.
    * ``series_precision`` -- A positive integer, the precision of the series expansion.

    OUTPUT:

    A series expansion of the function `g(x)`.

    EXAMPLES::

        sage: from sage_acsv.helpers import compute_newton_series
        sage: R.<x, T> = PolynomialRing(QQ, ['x', 'T'])
        sage: compute_newton_series(x*T^2 - T + 1, [x, T], 7)
        132*x^6 + 42*x^5 + 14*x^4 + 5*x^3 + 2*x^2 + x + 1

    """

    return compute_newton_series_general([phi], variables, series_precision)[0]


def compute_newton_series_general(phis, variables, series_precision):
    r"""Computes the series expansion of implicitly defined functions.

    The functions `g_1(x), ..., g_s(x)` for which a series expansion is computed is a simple root of the expression

    .. math::

        \Phi(x, g_1(x), ..., g_s(x)) = 0

    INPUT:

    * ``phis`` -- A list of polynomials; the equations defining the functions that are expanded.
    * ``variables`` -- A list of variables in the equation. The last variable in this
      list is the variable corresponding to `g(x)`.
    * ``series_precision`` -- A positive integer, the precision of the series expansion.

    OUTPUT:

    A tuple of series expansion of the functions `g_1(x), ..., g_s(x)`.

    EXAMPLES::

        sage: from sage_acsv.helpers import compute_newton_series_general
        sage: R.<x, T> = PolynomialRing(QQ, ['x', 'T'])
        sage: compute_newton_series_general([x*T^2 - T + 1], [x, T], 7)[0]
        132*x^6 + 42*x^5 + 14*x^4 + 5*x^3 + 2*x^2 + x + 1

        sage: R.<x, Y, Z> = PolynomialRing(QQ, ['x', 'Y', 'Z'])
        sage: compute_newton_series_general([x^3 - Y - Z^2, x - Y - Z], [x, Y, Z], 7)
        (-23*x^6 - 8*x^5 - 3*x^4 - x^3 - x^2, 23*x^6 + 8*x^5 + 3*x^4 + x^3 + x^2 + x)

    """
    s = len(phis)
    X = variables[:-s]
    Y = variables[-s:]

    def ModX(Fs, N):
        return vector([sum([c*f for c,f in F if sum([f.degree(x) for x in X]) < N]) for F in Fs])

    def ModY(Fs, N):
        return vector([sum([c*f for c,f in F if sum([f.degree(y) for y in Y]) < N]) for F in Fs])

    def Mod(Fs, N):
        return ModX(ModY(Fs, N), N)

    def Jacobian(Fs):
        return matrix([
            [F.derivative(v) for v in Y]
            for F in Fs
        ])

    def NewtonRecur(Hs, N):
        if N == 1:
            return vector([0 for _ in range(s)]), Jacobian(Hs).inverse().subs({v: 0 for v in variables})
        Fs, Gs = NewtonRecur(Hs, ceil(N / 2))
        Gs = Gs + (matrix.identity(s) - Gs * Jacobian(Hs).subs({Y[j]: Fs[j] for j in range(s)})) * Gs
        Fs = Fs - Gs * Hs.subs({Y[j]: Fs[j] for j in range(s)})
        return ModX(Fs, N), matrix([ModX(G, ceil(N / 2)) for G in Gs])

    return NewtonRecur(Mod(phis, series_precision), series_precision)[0]