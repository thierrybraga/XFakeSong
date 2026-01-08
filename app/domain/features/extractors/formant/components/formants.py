import numpy as np
from typing import List, Tuple


def find_formants_from_lpc(lpc_coeffs: np.ndarray,
                           sr: int) -> List[Tuple[float, float, float]]:
    """Encontra formantes a partir dos coeficientes LPC."""
    if len(lpc_coeffs) == 0:
        return []

    # Criar polinômio LPC
    # A(z) = 1 - sum(a_k * z^(-k))
    poly_coeffs = np.concatenate(([1], -lpc_coeffs))

    # Encontrar raízes do polinômio
    roots = np.roots(poly_coeffs)

    # Filtrar raízes para formantes
    formants = []

    for root in roots:
        # Considerar apenas raízes complexas conjugadas
        if np.imag(root) > 0:  # Considerar apenas metade superior do plano complexo
            # Converter para frequência e largura de banda
            angle = np.angle(root)
            magnitude = np.abs(root)

            # Frequência em Hz
            freq = angle * sr / (2 * np.pi)

            # Largura de banda
            if magnitude > 0:
                bandwidth = -np.log(magnitude) * sr / np.pi
            else:
                bandwidth = 0

            # Filtrar frequências válidas (50 Hz a Nyquist)
            if 50 <= freq <= sr / 2 and bandwidth > 0:
                # Amplitude (aproximação baseada na magnitude)
                amplitude = magnitude
                formants.append((freq, bandwidth, amplitude))

    # Ordenar por frequência
    formants.sort(key=lambda x: x[0])

    return formants
