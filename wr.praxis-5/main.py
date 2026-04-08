import numpy as np

####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # TODO: initialize matrix with proper size
    F = np.zeros((n, n), dtype='complex128')
    # TODO: create principal term for DFT matrix
    #Berechne die n-te Einheitswurzel (Hauptfaktor der DFT)
    principal_term = np.exp(-1j * 2 * np.pi / n)

    # TODO: fill matrix with values
    # Befülle die DFT-Matrix gemäß Definition: F[i,j] = ω^(i·j)
    for i in range(n):
        for j in range(n):
            F[i, j] = principal_term ** (i * j)

    # TODO: normalize dft matrix
    F /= np.sqrt(n)


    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    unitary = True
    # TODO: check that F is unitary, if not return false
    #prüfen, ob es wirklich eine quadratische 2D-Matrix ist
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    n = matrix.shape[0]
    ident = np.eye(n, dtype='complex128')
    # U*U berechnen (konjugiert-transponiert mal Matrix)
    prod = matrix.conj().T @ matrix

    # Numerischer Vergleich mit Toleranz (wegen Rundungsfehlern)
    return np.allclose(prod, ident, atol=1e-10, rtol=1e-10)

    


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    # TODO: create signals and extract harmonics out of DFT matrix
    F = dft_matrix(n)     # Build DFT matrix once

    # Create delta impulse signals e_i and obtain their DFTs.
    # For a DFT matrix F, the transform is X = F @ x.
    for i in range(n):
        e = np.zeros(n, dtype='float64')
        e[i] = 1.0
        sigs.append(e)
         # Copy to avoid accidental aliasing
        fsigs.append(F[:, i].copy())


    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT

def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """

   
    # TODO: implement shuffling by reversing index bits
    # Länge des Arrays (muss eine Zweierpotenz sein, z.B. 128, 256, ...)
    n = data.size
    # Anzahl der Bits, die man für die Indizes braucht (log2(n))
    num_bits = int(np.log2(n))
    # Neues Array für die bit-reversed Reihenfolge
    rev_data = np.empty_like(data)

    # Funktion zum Umdrehen der Bits eines Index
    def reverse_bits(x, bits):
        r = 0
        for _ in range(bits):
           # Schiebe r nach links und füge das letzte Bit von x an
            r = (r << 1) | (x & 1)
    # Schiebe x nach rechts, um das nächste Bit zu holen
            x >>= 1
        return r
    # Ordne die Daten entsprechend der bit-reversed Indizes neu
    for i in range(n):
        rev_data[reverse_bits(i, num_bits)] = data[i]

    # Rückgabe des umsortierten Arrays
    return rev_data
    
   


def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """



    # TODO: first step of FFT: shuffle data
    fdata = np.asarray(data, dtype='complex128')     # Eingabedaten in komplexes Array umwandeln
    n = fdata.size
    if not n > 0 or (n & (n - 1)) != 0:  # Eingabedaten in komplexes Array umwandeln
        raise ValueError
    fdata = shuffle_bit_reversed_order(fdata) # Erste FFT-Stufe: Daten in bit-reversed Reihenfolge bringen


    # TODO: second step, recursively merge transforms
    # Cooley-Tukey (iterativ): "Butterfly"-Merges in log2(n) Stufen
    levels = int(np.log2(n))
    for m in range(levels):
        step = 2 ** (m + 1)  # aktuelle Blocklaenge (2, 4, 8, ...)
        half = step // 2

        for k in range(half):
        # Twiddle-Faktor: exp(-2pi i k / step)
            omega = np.exp(-2j * np.pi * k / step)
            # Butterfly-Operationen auf allen passenden Positionen
            for i in range(k, n, step):
                j = i + half
                # Zwischenergebnis berechnen
                p = omega * fdata[j]
                #Addition und Subtraktion
                fdata[j] = fdata[i] - p
                fdata[i] = fdata[i] + p

    # 3) Normierung (unitary): wie bei der DFT-Matrix 1/sqrt(n)
    fdata = fdata / np.sqrt(n)

    # TODO: normalize fft signal
    return fdata


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0

    data = np.zeros(num_samples)

    # TODO: Generate sine wave with proper frequency
    # Zeitpunkte (endpoint=False -> [0,1) 
    x = np.linspace(x_min, x_max, num_samples, endpoint=False)
    # Sinus: sin(2*pi*f*t)
    data = np.sin(2 * np.pi * f * x)
    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """
    
   
   # TODO: compute Fourier transform of input data

    # Umrechnung des Bandlimits von Hz in einen Index im Frequenzspektrum
    bandlimit_index = int(bandlimit * adata.size / sampling_rate)
    # Originale Länge des Signals merken
    n_orig = adata.size
    # FFT benötigt eine Länge, die eine Zweierpotenz ist
    # Deshalb suchen wir die nächste Zweierpotenz >= n_orig
    n_fft = 1
    while n_fft < n_orig:
        n_fft *= 2
    # Wenn die Länge schon passt, kein Padding nötig
    if n_fft == n_orig:
        padded = np.asarray(adata, dtype='float64')
    else:
        # Ansonsten: neues Array mit Nullen erzeugen (Zero-Padding)
        padded = np.zeros(n_fft, dtype='float64')
        padded[:n_orig] = adata  # Originaldaten am Anfang kopieren

    # Fourier-Transformation des (ggf. gepaddeten) Signals
    fdata = fft(padded)

 
    # TODO: set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.
    # Bandlimit erneut berechnen, jetzt bezogen auf die gepaddete FFT-Laenge
    bandlimit_index = int(bandlimit * n_fft / sampling_rate)
    # Sicherstellen, dass der Index im gueltigen Bereich liegt
    bandlimit_index = max(0, min(bandlimit_index, n_fft // 2))
    # Alles zwischen +bandlimit und -bandlimit wird auf 0 gesetzt
    fdata[bandlimit_index + 1 : n_fft - bandlimit_index] = 0.0 + 0.0j

    
    # TODO: compute inverse transform and extract real component
    # Ruecktransformieren ins Zeit-Signal (Inverse FFT ohne numpy.fft):
    # Bei unitary FFT gilt: ifft(X) = conj( fft(conj(X)) )
    adata_rec = np.conj(fft(np.conj(fdata))).real

    # Padding abschneiden -> wieder genau die Originallaenge
    adata_filtered = adata_rec[:n_orig].copy()

    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
