import numpy as np
import scipy.linalg
from scipy.linalg import solve_toeplitz


def solve_yule_walker(r: np.ndarray, order: int) -> np.ndarray:
    """Resolve equações de Yule-Walker para análise LPC."""
    if len(r) < order + 1:
        return np.zeros(order)

    try:
        # Matriz de Toeplitz (apenas primeira coluna é necessária para solve_toeplitz se simétrica)
        # R = scipy.linalg.toeplitz(r[:order]) # Não precisamos da matriz
        # completa

        # Vetor de correlação (lado direito)
        p = r[1:order + 1]

        # Primeira coluna da matriz de Toeplitz (autocorrelação R(0)...R(p-1))
        c = r[:order]

        # Resolver sistema R * a = p
        # solve_toeplitz aceita a primeira coluna c (para matriz simétrica)
        a = solve_toeplitz(c, p)

        return a
    except BaseException:
        return np.zeros(order)


def lpc_to_cepstral(lpc_coeffs: np.ndarray, n_mfcc: int) -> np.ndarray:
    """Converte coeficientes LPC para cepstrais."""
    # Garantir que lpc_coeffs seja 2D (order, n_frames)
    if lpc_coeffs.ndim == 1:
        lpc_coeffs = lpc_coeffs[:, np.newaxis]

    order, n_frames = lpc_coeffs.shape
    cepstral = np.zeros((n_mfcc, n_frames))

    for i in range(n_frames):
        a = lpc_coeffs[:, i]
        c = np.zeros(n_mfcc)

        # Algoritmo de recursão para LPC -> Cepstrum
        # c_1 = a_1
        # c_n = a_n + sum_{k=1}^{n-1} (k/n) c_k a_{n-k}  (1 <= n <= p)
        # c_n = sum_{k=n-p}^{n-1} (k/n) c_k a_{n-k}      (n > p)

        # Mapeamento: c[n] -> c_{n+1}, a[n] -> a_{n+1}

        if order > 0 and n_mfcc > 0:
            c[0] = a[0]

        for n in range(1, n_mfcc):
            sum_val = 0.0
            if n < order:
                sum_val = a[n]

            # k vai de 1 até n (correspondendo a c_1 ... c_n)
            # No array c, índices 0 ... n-1
            # Termo a: a_{n+1-k} -> a[n-k]

            for k in range(1, n + 1):
                # c_k -> c[k-1]
                # a_{m-k} onde m=n+1 -> a[n+1-k-1] -> a[n-k]

                if (n - k) < order:
                    sum_val += (k / (n + 1)) * c[k - 1] * a[n - k]

            c[n] = sum_val

        cepstral[:, i] = c

    return cepstral


def compute_lpc(frame: np.ndarray, order: int) -> np.ndarray:
    """Computa coeficientes LPC para um frame."""
    try:
        # Calcular autocorrelação
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        # Normalizar
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Resolver Yule-Walker
        if len(autocorr) > order:
            lpc_coeffs = solve_yule_walker(autocorr, order)
        else:
            lpc_coeffs = np.zeros(order)

        return lpc_coeffs
    except BaseException:
        return np.zeros(order)


def apply_preemphasis(y: np.ndarray, preemphasis: float) -> np.ndarray:
    """Aplica pré-ênfase ao sinal."""
    if preemphasis == 0:
        return y

    preemphasized = np.zeros_like(y)
    preemphasized[0] = y[0]
    for i in range(1, len(y)):
        preemphasized[i] = y[i] - preemphasis * y[i - 1]

    return preemphasized
