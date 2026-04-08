import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d(0)
    base_functions = []

    # TODO: Generate Lagrange base polynomials and interpolation polynomial
    n = x.size
    for i in range(n):
        li = np.poly1d(1)  # start with 1 for product
        for j in range(n):
            if i != j:
                li *= np.poly1d([1.0, -x[j]]) / (x[i] - x[j])
        base_functions.append(li)
        polynomial += y[i] * li

    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    # TODO compute piecewise interpolating cubic polynomials
    
    spline = []
    anzahl_intervalle = x.size - 1
    
    for k in range(anzahl_intervalle):
        # Aktuelle Stützstellen und Werte für das Intervall [k, k+1]
        xk = x[k]
        xk1 = x[k+1]
        
        # Aufstellen der Interpolationsmatrix (höchste Potenz links)
        # Zeilen 1-2: p(xk) = yk und p(xk1) = yk1
        # Zeilen 3-4: p'(xk) = ypk und p'(xk1) = ypk1
        A_hermite = np.array([
            [xk**3,   xk**2,   xk, 1],
            [xk1**3,  xk1**2,  xk1, 1],
            [3*xk**2, 2*xk,    1,   0],
            [3*xk1**2, 2*xk1,  1,   0]
        ])
        
        # Zielvektor mit den bekannten Werten
        b_vektor = np.array([y[k], y[k+1], yp[k], yp[k+1]])
        
        # Berechne Koeffizientenvektor [a, b, c, d]
        koeffs = np.linalg.solve(A_hermite, b_vektor)
        
        # Erzeuge das Teilpolynom und füge es der Liste hinzu
        teil_polynom = np.poly1d(koeffs)
        spline.append(teil_polynom)
        
    return  spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    n = x.size
    # TODO construct linear system with natural boundary conditions
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Randbedingungen (2. Ableitung an den Enden = 0)
    # Diese Formeln für das LGS berechnen die Ableitungen (Steigungen) yp
    A[0, 0] = 2
    A[0, 1] = 1
    b[0] = 3 * (y[1] - y[0]) / (x[1] - x[0])
    
    A[n - 1, n - 2] = 1
    A[n - 1, n - 1] = 2
    b[n - 1] = 3 * (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])

    # Innere Gleichungen
    for i in range(1, n - 1):
        h_prev = x[i] - x[i - 1]
        h_next = x[i + 1] - x[i]
        A[i, i - 1] = h_next
        A[i, i]     = 2 * (h_prev + h_next)
        A[i, i + 1] = h_prev
        b[i] = 3 * (h_next / h_prev * (y[i] - y[i - 1]) + h_prev / h_next * (y[i + 1] - y[i]))

    # TODO solve linear system for the coefficients of the spline
    yp = np.linalg.solve(A, b)
    
    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(n - 1):
        x0, x1 = x[i], x[i+1]
        
        # Lokales LGS für die 4 Koeffizienten [a, b, c, d]
        # p(x0)=y0, p(x1)=y1, p'(x0)=yp0, p'(x1)=yp1
        # Entspricht der Vorgabe: höchste Potenz (x^3) links
        M_lokal = np.array([
            [x0**3,   x0**2,   x0, 1],
            [x1**3,   x1**2,   x1, 1],
            [3*x0**2, 2*x0,    1,  0],
            [3*x1**2, 2*x1,    1,  0]
        ])
        
        ziel_werte = np.array([y[i], y[i+1], yp[i], yp[i+1]])
        koeffizienten = np.linalg.solve(M_lokal, ziel_werte)
        
        # Erstellt das poly1d Objekt mit globalen x-Koeffizienten
        spline.append(np.poly1d(koeffizienten))

    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO: construct linear system with periodic boundary conditions

    # TODO solve linear system for the coefficients of the spline

    spline = []
    # TODO extract local interpolation coefficients from solution


    return spline


if __name__ == '__main__':

    x = np.array( [1.0, 2.0, 3.0, 4.0])
    y = np.array( [3.0, 2.0, 4.0, 1.0])

    splines = natural_cubic_interpolation( x, y)

    # # x-values to be interpolated
    # keytimes = np.linspace(0, 200, 11)
    # # y-values to be interpolated
    # keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
    #              np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
    # keyframes.append(keyframes[0])
    # splines = []
    # for i in range(11):  # Iterate over all animated parts
    #     x = keytimes
    #     y = np.array([keyframes[k][i] for k in range(11)])
    #     spline = natural_cubic_interpolation(x, y)
    #     if len(spline) == 0:
    #         animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
    #         self.fail("Natural cubic interpolation not implemented.")
    #     splines.append(spline)

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
