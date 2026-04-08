import numpy as np
import lib
import matplotlib as mpl
import matplotlib.image as mpimg   # ← WICHTIG, damit imread funktioniert!



####################################################################################################
# Exercise 1: Power Iteration

def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    """
    Compute largest eigenvector of matrix M using power iteration. It is assumed that the
    largest eigenvalue of M, in magnitude, is well separated.

    Arguments:
    M: matrix, assumed to have a well separated largest eigenvalue
    epsilon: epsilon used for convergence (default: 10 * machine precision)

    Return:
    vector: eigenvector associated with largest eigenvalue
    residuals : residual for each iteration step

    Raised Exceptions:
    ValueError: if matrix is not square

    Forbidden:
    numpy.linalg.eig, numpy.linalg.eigh, numpy.linalg.svd
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix not nxn")

    # TODO: set epsilon to default value if not set by user
    if epsilon < 0:
        epsilon = 10 * np.finfo(M.dtype).eps

    # TODO: normalized random vector of proper size to initialize iteration
    vector = np.random.rand(M.shape[0])
    vector = vector / np.linalg.norm(vector)

    # Initialize residual list and residual of current eigenvector estimate
    residuals = []
    residual = 2.0 * epsilon

    while residual > epsilon:
        # TODO: implement power iteration
        w = M @ vector  # Indent this line
        # Schritt 2: Normieren
        new_vector = w / np.linalg.norm(w)

        # Schritt 3: Rayleigh-Quotient
        eigenvalue = np.dot(new_vector, M @ new_vector)

        # Schritt 4: Residual
        residual = np.linalg.norm(M @ new_vector - eigenvalue * new_vector)
        residuals.append(residual)

     # Schritt 5: Update
        vector = new_vector

    return vector, residuals


####################################################################################################
# Exercise 2: Eigenfaces

def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []

    # TODO read each image in path as numpy.ndarray and append to images
    # Useful functions: lib.list_directory(), matplotlib.image.imread(), numpy.asarray()
    file_names = lib.list_directory(path)

    # nur passende Dateien sammeln
    picture_list = []  # hier kommen nur die Bilder rein
    for name in file_names:
        # Wenn die Datei auf .png endet, wollen wir sie benutzen
        if name.endswith(file_ending):
            picture_list.append(name)

    # einfache Sortierung
    # Wir sortieren die Namen, damit die Bilder in der richtigen Reihenfolge geladen werden
    picture_list.sort()

    # Bilder laden
    for name in picture_list:
        # Bild aus der Datei lesen
        img = mpimg.imread(path + "/" + name)

        # Bild in ein numpy-Array umwandeln (als Zahlentabelle)
        img = np.asarray(img, float)

        # Bild in unsere Liste speichern
        images.append(img)

    # TODO set dimensions according to first image in images
    # Falls wir mindestens ein Bild haben, bestimmen wir die Größe
    if len(images) > 0:
        # Höhe des Bildes (Pixel oben nach unten)
        dimension_y = images[0].shape[0]

        # Breite des Bildes (Pixel links nach rechts)
        dimension_x = images[0].shape[1]
    else:
        # Wenn keine Bilder da sind, ist die Größe 0
        dimension_y = 0
        dimension_x = 0

    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """
    # TODO: initialize data matrix with proper size and data type
    if len(images) == 0:
            return np.zeros((0, 0))

        # Größe des ersten Bildes bestimmen
    h = images[0].shape[0]
    w = images[0].shape[1]
     # Datenmatrix initialisieren: m Bilder, jedes flachgemacht zu h*w
    D = np.zeros((len(images), h * w), dtype=np.float64)


    # TODO: add flattened images to data matrix
    for i in range(len(images)):

        D[i, :] = images[i].flatten()

    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principal component analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    pcs: matrix containing principal components as rows
    svals: singular values associated with principle components
    mean_data: mean that was subtracted from data
    """

    # TODO: subtract mean from data / center data at origin
    # Durchschnitt von allen Bildern berechnen
    mean_data = np.mean(D, axis=0) 
    # Von jedem Bild den Durchschnitt abziehen
    D_centered = D - mean_data


    # TODO: compute left and right singular vectors and singular values
    # Useful functions: numpy.linalg.svd(..., full_matrices=False)
    # Wichtige Richtungen und Zahlen ausrechnen (SVD)
    U, svals, Vt = np.linalg.svd(D_centered, full_matrices=False)


    # pcs = right singular vectors = rows of Vt
    pcs = Vt

    return pcs, svals, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """

       # TODO: Normalize singular value magnitudes
    # Wir rechnen zuerst die Summe von allen Singularwerten aus.
    total = np.sum(singular_values)
    # So können wir einfach sehen, wie viel Prozent jeder Wert ausmacht.
    normalized = singular_values / total

    k = 0
    # TODO: Determine k that first k singular values make up threshold percent of magnitude
    # Wir starten bei 0 Prozent und addieren die Werte nacheinander.
    current_sum = 0.0

    # Wir gehen jeden Wert vorne nach hinten durch.
    for i in range(len(normalized)):

        # Den aktuellen Wert dazurechnen.
        current_sum += normalized[i]

        # Wenn wir genug gesammelt haben (z.B. 80 %),
        # dann geben wir zurück, wie viele Werte wir gebraucht haben.
        if current_sum >= threshold:
            return i + 1   # +1 weil wir bei 0 anfangen zu zählen

    # Falls der Wert nie erreicht wird, geben wir einfach die volle Länge zurück.
    return len(normalized)


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs:  matrix containing principal components / eigenfunctions as rows
    images: original input images from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """

    # TODO: initialize coefficients array with proper size
    # Anzahl Bilder herausfinden
    num_images = len(images)

    # Anzahl der Hauptkomponenten = Zeilenanzahl von pcs
    k = pcs.shape[0]

    # Tabelle für Ergebnisse erstellen
    # Jede Zeile = ein Bild
    # Jede Spalte = ein Wert für eine Hauptkomponente
    coefficients = np.zeros((num_images, k))

    # TODO: iterate over images and project each normalized image into principal component basis
    # Wir gehen Bild für Bild durch
    for i in range(num_images):

        # Bild flach machen (von 2D in 1D)
        flat = images[i].flatten()

        # Mittelwert abziehen (zentrieren)
        centered = flat - mean_data

        # Projektion: Wie stark passt das Bild zu jedem Eigenface?
        for j in range(k):
            coefficients[i, j] = np.dot(pcs[j], centered)

    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """


    # TODO: load test data set
    # Wir laden die Testbilder aus dem Test-Ordner.
    imgs_test, dx, dy = load_images(path_test)

    # TODO: project test data set into eigenbasis
    # Wir berechnen die Koeffizienten der Testbilder.
    coeffs_test = project_faces(pcs, imgs_test, mean_data)

    # TODO: Initialize scores matrix with proper size
    # Anzahl der Trainingsbilder und Testbilder bestimmen.
    num_train = coeffs_train.shape[0]
    num_test = coeffs_test.shape[0]

    # Die Score-Tabelle hat Zeilen = Training, Spalten = Test.
    scores = np.zeros((num_train, num_test))

    # TODO: Iterate over all images and calculate pairwise correlation
    # Wir vergleichen jedes Trainingsbild mit jedem Testbild.
    for i in range(num_train):
        for j in range(num_test):

            # Die Werte vom Trainingsbild holen
            a = coeffs_train[i]

            # Die Werte vom Testbild holen
            b = coeffs_test[j]

            # Punktprodukt ausrechnen (zeigt, wie ähnlich die Richtungen sind)
            dot = np.dot(a, b)

            # Länge der beiden Vektoren ausrechnen
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            # Wenn einer der Vektoren Länge 0 hat, ist der Winkel maximal
            if norm_a == 0 or norm_b == 0:
                theta = np.pi   # größter möglicher Winkel
            else:
                # cos(θ) ausrechnen (wie ähnlich die Richtungen sind)
                cos_theta = dot / (norm_a * norm_b)
                # sicherstellen, dass der Wert nicht über 1 oder unter -1 geht
                if cos_theta > 1:
                    cos_theta = 1
                elif cos_theta < -1:
                    cos_theta = -1
                # Winkel berechnen (kleiner Winkel = ähnlicher)
                theta = np.arccos(cos_theta)
            # Den berechneten Winkel in die Scores-Tabelle schreiben
            scores[i, j] = theta
    return scores, imgs_test, coeffs_test


if __name__ == '__main__':

    A = np.random.randn( 7, 7)
    A = A.transpose().dot(A)
    L,U = np.linalg.eig( A)
    L[1] = L[0] - 10**-3
    A = U.dot(np.diag(L)).dot(U.transpose())
    print( )
    np.set_printoptions(precision=16)
    print( A.flatten())

    A = np.array( [ 18.2112344794043359,   0.7559886314903312,  7.2437569750169502,
                    -13.8991061752623271,   4.8768689715057691,  -1.318055436971276,
                    -6.7829844205260148,   0.7559886314903312,   7.9204801042364448,
                     1.5378938590357767,   7.1775560914639325,   2.8536549530686015,
                     1.9998683983340397,  -5.9532930598376685,   7.2437569750169502,
                     1.5378938590357767,   9.841906218619128,   0.5841092845624152,
                     6.7510103134860797,   4.6111951240722888,  -8.9825300821798191,
                    -13.8991061752623271,   7.1775560914639334,   0.5841092845624152,
                     24.2028041177043818,   0.8180957104689988,   6.6087248591945729,
                    -4.1573996873552073,   4.8768689715057691,   2.8536549530686015,
                     6.7510103134860806,   0.8180957104689979,   7.0366782892027206,
                     5.4944303652858073,  -9.0773671527609796,  -1.318055436971276,
                     1.9998683983340397,   4.6111951240722888,   6.608724859194572,
                     5.4944303652858073,   8.1889694453300805,  -7.1176432086570651,
                    -6.7829844205260148,  -5.9532930598376685,  -8.9825300821798191,
                    -4.1573996873552046,  -9.0773671527609796,  -7.1176432086570633,
                    13.664209790087753 ])
    A = A.reshape( (7,7))

    ev, res = power_iteration( A)



    print( 'ev = ' + str(ev))

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
