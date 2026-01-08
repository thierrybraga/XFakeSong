import numpy as np


def preemphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """Aplica filtro de pré-ênfase."""
    if len(signal) == 0:
        return signal

    preemphasized = np.zeros_like(signal)
    preemphasized[0] = signal[0]

    for i in range(1, len(signal)):
        preemphasized[i] = signal[i] - alpha * signal[i - 1]

    return preemphasized


def levinson_durbin_autocorr(signal: np.ndarray, order: int) -> np.ndarray:
    """Calcula coeficientes LPC usando Levinson-Durbin."""
    # Calcular autocorrelação
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    if len(autocorr) < order + 1:
        autocorr = np.pad(autocorr, (0, order + 1 - len(autocorr)))
    else:
        autocorr = autocorr[:order + 1]

    if autocorr[0] == 0:
        return np.zeros(order)

    # Algoritmo Levinson-Durbin
    lpc = np.zeros(order)
    error = autocorr[0]

    for i in range(order):
        # Calcular coeficiente de reflexão
        reflection = 0
        for j in range(i):
            reflection += lpc[j] * autocorr[i - j]

        if error != 0:
            reflection = -(autocorr[i + 1] + reflection) / error

        lpc[i] = reflection

        # Atualizar coeficientes anteriores
        for j in range(i):
            lpc[j] = lpc[j] + reflection * lpc[i - 1 - j]

        # Atualizar erro
        error *= (1 - reflection ** 2)

        if error <= 0:
            break

    return lpc
