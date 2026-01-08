"""
Extração de características de contraste espectral.
"""
import numpy as np
from typing import Dict


def compute_subband_energy_ratios(S: np.ndarray, freqs: np.ndarray) -> Dict:
    """Computa razões de energia entre sub-bandas."""
    # Definir bandas de frequência
    bands = {
        'low': (0, 500),
        'mid_low': (500, 1000),
        'mid': (1000, 2000),
        'mid_high': (2000, 4000),
        'high': (4000, 8000)
    }

    ratios = {}

    # Calcular energia de cada banda
    band_energies = {}
    for band_name, (f_low, f_high) in bands.items():
        band_mask = (freqs >= f_low) & (freqs <= f_high)
        if np.any(band_mask):
            band_energy = np.sum(S[band_mask, :], axis=0)
            band_energies[band_name] = band_energy

    # Calcular razões
    if 'low' in band_energies and 'high' in band_energies:
        ratios['low_high_ratio'] = (band_energies['low'] /
                                    (band_energies['high'] + 1e-10))

    if 'mid' in band_energies and 'high' in band_energies:
        ratios['mid_high_ratio'] = (band_energies['mid'] /
                                    (band_energies['high'] + 1e-10))

    if 'low' in band_energies and 'mid' in band_energies:
        ratios['low_mid_ratio'] = (band_energies['low'] /
                                   (band_energies['mid'] + 1e-10))

    return ratios


def compute_high_freq_content(S: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Computa concentração de energia em altas frequências."""
    # Definir alta frequência como > 4kHz
    high_freq_mask = freqs > 4000

    if np.any(high_freq_mask):
        high_freq_energy = np.sum(S[high_freq_mask, :], axis=0)
        total_energy = np.sum(S, axis=0)

        # Razão de energia em alta frequência
        hfc = high_freq_energy / (total_energy + 1e-10)
    else:
        hfc = np.zeros(S.shape[1])

    return hfc
