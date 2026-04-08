
import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A is not square")
    if b.shape[0] != A.shape[0]:
        raise ValueError("Matrix A and vector b have incompatible sizes")

    # Wir machen sicher, dass wir mit Kommazahlen rechnen
    A = A.astype(float)
    b = b.astype(float)

    m = A.shape[0]

    # TODO: Perform gaussian elimination

    # Gauß-Elimination
    for i in range(m - 1):  # wir gehen Spalte für Spalte
        # Pivoting: größtes Element in Spalte finden
        if use_pivoting:
            max_row = np.argmax(np.abs(A[i:, i])) + i
            if max_row != i:
                # Zeilen tauschen
                A[[i, max_row]] = A[[max_row, i]]
                b[[i, max_row]] = b[[max_row, i]]

        # Wenn die Diagonale 0 ist -> nicht lösbar
        if A[i, i] == 0:
            raise ValueError("Matrix is singular")

        # Jetzt alle Zeilen unterhalb zu 0 machen
        for j in range(i + 1, m):
            faktor = A[j, i] / A[i, i]

            # komplette Zeile abziehen
            for k in range(i, m):
                A[j, k] = A[j, k] - faktor * A[i, k]

            # rechte Seite anpassen
            b[j] = b[j] - faktor * b[i]

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix A is not square")
    if b.shape[0] != m:
        raise ValueError("Matrix and vector sizes are incompatible")
    # TODO: Initialize solution vector with proper size
    x = np.zeros(m)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    for i in range(m - 1, -1, -1):  # von unten nach oben
        if A[i, i] == 0:
            raise ValueError("No or infinite solutions exist")

        # b[i] minus Summe von A[i,j] * x[j] für j > i
        sum_term = 0.0
        for j in range(i + 1, m):
            sum_term += A[i, j] * x[j]

        x[i] = (b[i] - sum_term) / A[i, i]
    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    if n != m:
        raise ValueError("Matrix is not square")
    if not np.allclose(M ,M.T):
        raise ValueError("Matrix is not symmetric")
    

    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))
    for i in range(n):
        sum_diag = 0.0
        for k in range(i):
            sum_diag += L[i, k] * L[i, k]

        diag_value = M[i, i] - sum_diag
        if diag_value <= 0:
            raise ValueError("Matrix is not positive definite")

        L[i, i] = np.sqrt(diag_value)

        for j in range(i + 1, n):
            sum_sub = 0.0
            for k in range(i):
                sum_sub += L[j, k] * L[i, k]

            L[j, i] = (M[j, i] - sum_sub) / L[i, i]

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    # Hier prüfen wir, ob die Matrix L eine richtige Form hat.
    # Sie muss quadratisch sein, also gleich viele Zeilen wie Spalten haben.
    (n, m) = L.shape
    if n != m:
        raise ValueError("Matrix is not square")
    
    # Hier prüfen wir, ob der Vektor b die richtige Länge hat.
    # b muss genauso viele Werte haben wie L Zeilen hat.
    if b.shape[0] != n:
        raise ValueError("Matrix and vector sizes do not match")

    # Hier schauen wir, ob L wirklich eine untere Dreiecksmatrix ist.
    # Das bedeutet: Alles über der Diagonalen muss 0 sein.
    for i in range(n):
        for j in range(i+1, n):
            if L[i, j] != 0:
                raise ValueError("L is not lower triangular")

    # TODO Solve the system by forward- and backsubstitution

    # Jetzt lösen wir zuerst L * y = b (Vorwärtseinsetzen).
    # Wir rechnen uns von oben nach unten vor.
    y = np.zeros(n)
    for i in range(n):
        if L[i, i] == 0:  # Wenn auf der Diagonale eine 0 ist, kann man nicht teilen.
            raise ValueError("Matrix is singular")
        
        # Wir rechnen die Summe aus L[i,k] * y[k] für alle bekannten y[k].
        s = 0.0
        for k in range(i):
            s += L[i, k] * y[k]

        y[i] = (b[i] - s) / L[i, i]

    x = back_substitution(L.T, y)

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """
    # TODO: Initialize system matrix with proper size
    n_cells = n_grid * n_grid   # wie viele Kästchen hat unser Bild?
    N = n_shots * n_rays        # wie viele Strahlen sind in unserem Bild?    
    L = np.zeros((N, n_cells))  # eine große Tabelle, die sagt, wie jeder Strahl durch jedes Kästchen läuft

    # TODO: Initialize intensity vector
    g = np.zeros(N)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    for s in range(n_shots):

        # Take a measurement with the tomograph from direction r_theta.
        # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
        # ray_indices: indices of rays that intersect a cell
        # isect_indices: indices of intersected cells
        # lengths: lengths of segments in intersected cells
        # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.

        # Take a measurement with the tomograph from direction r_theta.
        theta = s * (np.pi / n_shots)
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(
            n_grid, n_rays, theta
        )

        # Hier tragen wir ein, durch welche Zelle der Strahl gegangen ist.
        # Und wie lang der Strahl in dieser Zelle war.
        moh=s * n_rays
        g[moh: moh + n_rays] = intensities

        # Jede Intersektion direkt eintragen
        # Hier tragen wir ein, durch welche Zelle der Strahl gegangen ist.
        # Und wie lang der Strahl in dieser Zelle war.
        for k in range(len(ray_indices)):
            ray = ray_indices[k]            # Welcher Strahl wurde getroffen?
            cell = isect_indices[k]         # Welche Zelle wurde getroffen?
            soh = moh + ray
            length = lengths[k]             # Wie lang war der Weg in dieser Zelle?
            L[soh,cell] = length    # Wir tragen die Länge in die Matrix ein


    return [L, g]


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)

    # Wir bauen erst das Gleichungssystem in Cholesky-Form um:
    # A = L^T L  und  b = L^T g
    A = L.T @ L
    b = L.T @ g
     # Jetzt machen wir Cholesky-Faktorisierung von A
    C = compute_cholesky(A)

    # Und lösen das System C C^T x = b
    x = solve_cholesky(C, b)

    # TODO: Convert solution of linear system to 2D image
    tim = x.reshape((n_grid, n_grid))

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")

# TODO: Replace this with a justification of why your code is correct