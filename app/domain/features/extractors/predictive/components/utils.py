"""
Utilitários para extração de características preditivas.
"""

import numpy as np
import scipy.signal
import scipy.linalg
from typing import Tuple, Optional


def frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> list:
    """Divide o sinal em frames."""
    frames = []

    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]

        # Aplicar janela
        window = scipy.signal.windows.hann(len(frame))
        windowed_frame = frame * window

        frames.append(windowed_frame)

    return frames


def compute_lpc(frame: np.ndarray,
                order: int) -> Tuple[Optional[np.ndarray], float]:
    """Calcula coeficientes LPC usando autocorrelação."""
    try:
        # Calcular autocorrelação
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        if len(autocorr) <= order:
            return None, 0

        # Resolver equações de Yule-Walker
        R = scipy.linalg.toeplitz(autocorr[:order])
        r = autocorr[1:order + 1]

        # Resolver sistema linear
        lpc_coeffs = scipy.linalg.solve(R, r)

        # Calcular ganho
        gain = autocorr[0] - np.dot(lpc_coeffs, autocorr[1:order + 1])

        return lpc_coeffs, gain

    except BaseException:
        return None, 0


def lpc_to_lsf(lpc_coeffs: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """Converte coeficientes LPC para Line Spectral Frequencies."""
    try:
        order = len(lpc_coeffs)

        # Construir polinômios P e Q
        # P(z) = A(z) + z^(-m-1) * A(z^(-1))
        # Q(z) = A(z) - z^(-m-1) * A(z^(-1))

        # Coeficientes do polinômio A(z) = 1 + a1*z^(-1) + ... + am*z^(-m)
        A = np.concatenate(([1], lpc_coeffs))

        # Construir P(z) e Q(z)
        if order % 2 == 0:
            # Ordem par
            P = A + np.concatenate((A[::-1], [0]))
            Q = A - np.concatenate((A[::-1], [0]))
        else:
            # Ordem ímpar
            P = A + A[::-1]
            Q = A - A[::-1]

        # Encontrar raízes
        P_roots = np.roots(P)
        Q_roots = np.roots(Q)

        # Extrair LSF das raízes no círculo unitário
        lsf = []

        # Processar raízes de P
        for root in P_roots:
            if np.abs(np.abs(root) - 1) < 0.1:  # Próximo ao círculo unitário
                angle = np.angle(root)
                if 0 <= angle <= np.pi:
                    lsf.append(angle)

        # Processar raízes de Q
        for root in Q_roots:
            if np.abs(np.abs(root) - 1) < 0.1:  # Próximo ao círculo unitário
                angle = np.angle(root)
                if 0 <= angle <= np.pi:
                    lsf.append(angle)

        # Ordenar e normalizar
        lsf = sorted(lsf)

        # Converter para frequência (Hz)
        lsf_hz = np.array(lsf) * sr / (2 * np.pi)

        # Garantir que temos o número correto de LSF
        if len(lsf_hz) != order:
            # Preencher ou truncar conforme necessário
            if len(lsf_hz) < order:
                # Preencher com valores interpolados
                lsf_hz = np.pad(lsf_hz, (0, order - len(lsf_hz)), mode='edge')
            else:
                # Truncar
                lsf_hz = lsf_hz[:order]

        return lsf_hz

    except BaseException:
        return None


def compute_reflection_coefficients(
        frame: np.ndarray, order: int) -> Optional[np.ndarray]:
    """Calcula coeficientes de reflexão usando algoritmo de Levinson-Durbin."""
    try:
        # Calcular autocorrelação
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        if len(autocorr) <= order or autocorr[0] == 0:
            return None

        # Algoritmo de Levinson-Durbin
        reflection_coeffs = np.zeros(order)
        a = np.zeros((order + 1, order + 1))
        E = autocorr[0]

        for i in range(1, order + 1):
            # Calcular coeficiente de reflexão
            k = autocorr[i]

            for j in range(1, i):
                k -= a[i - 1, j] * autocorr[i - j]

            if E == 0:
                break

            k = k / E
            reflection_coeffs[i - 1] = k

            # Atualizar coeficientes
            a[i, i] = k

            for j in range(1, i):
                a[i, j] = a[i - 1, j] - k * a[i - 1, i - j]

            # Atualizar energia do erro
            E = E * (1 - k * k)

            if E <= 0:
                break

        return reflection_coeffs

    except BaseException:
        return None


def compute_prediction_error(
        frame: np.ndarray, order: int) -> Tuple[Optional[np.ndarray], float]:
    """Calcula erro de predição."""
    try:
        # Calcular coeficientes LPC
        lpc_coeffs, gain = compute_lpc(frame, order)

        if lpc_coeffs is None:
            return None, 0

        # Calcular sinal predito
        predicted = np.zeros_like(frame)

        for i in range(order, len(frame)):
            prediction = 0

            for j in range(order):
                prediction += lpc_coeffs[j] * frame[i - j - 1]

            predicted[i] = prediction

        # Calcular erro
        error = frame - predicted

        # Retornar apenas a parte com predição válida
        valid_error = error[order:]

        return valid_error, gain

    except BaseException:
        return None, 0


def check_stability(
        lpc_coeffs: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """Verifica estabilidade do filtro LPC através dos polos."""
    try:
        # Construir polinômio característico
        # A(z) = 1 + a1*z^(-1) + ... + am*z^(-m)
        # Converter para z^m + a1*z^(m-1) + ... + am

        poly_coeffs = np.concatenate(([1], lpc_coeffs))

        # Encontrar raízes (polos)
        poles = np.roots(poly_coeffs)

        # Calcular magnitudes dos polos
        pole_magnitudes = np.abs(poles)

        # Sistema é estável se todos os polos estão dentro do círculo unitário
        is_stable = np.all(pole_magnitudes < 1.0)

        return is_stable, pole_magnitudes

    except BaseException:
        return False, None


def lpc_to_frequency_response(
        lpc_coeffs: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """Calcula resposta em frequência do filtro LPC."""
    try:
        # Construir função de transferência
        # H(z) = 1 / A(z) onde A(z) = 1 + a1*z^(-1) + ... + am*z^(-m)

        # Coeficientes do denominador
        denominator = np.concatenate(([1], lpc_coeffs))
        numerator = [1]

        # Calcular resposta em frequência
        w, h = scipy.signal.freqz(numerator, denominator, worN=512, fs=sr)

        # Retornar magnitude
        magnitude = np.abs(h)

        return magnitude

    except BaseException:
        return None


def compute_spectral_tilt(magnitude: np.ndarray, sr: int) -> float:
    """Calcula inclinação espectral."""
    try:
        # Converter para dB
        magnitude_db = 20 * np.log10(magnitude + 1e-12)

        # Ajustar linha
        freqs = np.linspace(0, sr / 2, len(magnitude_db))
        slope, _, _, _, _ = scipy.stats.linregress(freqs, magnitude_db)

        return slope

    except BaseException:
        return 0


def compute_spectral_flatness(magnitude: np.ndarray) -> float:
    """Calcula flatness espectral."""
    try:
        # Evitar log de zero
        magnitude = magnitude + 1e-12

        # Média geométrica / média aritmética
        geometric_mean = np.exp(np.mean(np.log(magnitude)))
        arithmetic_mean = np.mean(magnitude)

        if arithmetic_mean > 0:
            flatness = geometric_mean / arithmetic_mean
        else:
            flatness = 0

        return flatness

    except BaseException:
        return 0
