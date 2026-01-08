import numpy as np
import pywt
import warnings
from typing import Dict, Any


def extract_wavelet_features(
        y: np.ndarray, sr: int, wavelet: str = 'db4') -> Dict[str, Any]:
    """Extrai características de transformadas wavelet (DWT e CWT)."""
    features = {}

    try:
        # DWT usando Daubechies
        coeffs = pywt.wavedec(y, wavelet, level=4)

        # Energia dos coeficientes
        features['dwt_energy'] = [np.sum(c**2) for c in coeffs]

        # Entropia dos coeficientes
        features['dwt_entropy'] = []
        for c in coeffs:
            if len(c) > 0:
                c_norm = np.abs(c) / (np.sum(np.abs(c)) + 1e-10)
                ent = -np.sum(c_norm * np.log2(c_norm + 1e-10))
                features['dwt_entropy'].append(ent)
            else:
                features['dwt_entropy'].append(0)

        # CWT features
        cwt_features = extract_cwt_features(y, sr, wavelet)
        features.update(cwt_features)

    except Exception as e:
        warnings.warn(f"Erro na extração wavelet: {str(e)}")
        features['dwt_energy'] = [0]
        features['dwt_entropy'] = [0]
        # Adicionar defaults CWT também
        features['cwt_energy'] = np.array([0])
        features['cwt_centroid'] = np.array([0])

    return features


def extract_cwt_features(y: np.ndarray, sr: int,
                         wavelet: str) -> Dict[str, Any]:
    """Extrai características da Continuous Wavelet Transform."""
    features = {}

    try:
        # Escalas para CWT
        scales = np.arange(1, 32)

        # Calcular CWT
        # Limitar tamanho para evitar estouro de memória/tempo em CWT
        y_cwt = y[:min(len(y), 8192)]

        try:
            cwt_coeffs, freqs = pywt.cwt(
                y_cwt, scales, wavelet, sampling_period=1 / sr)
        except AttributeError:
            # Fallback para 'mexh' se a wavelet escolhida não suportar CWT (ex:
            # db4 em algumas versões)
            cwt_coeffs, freqs = pywt.cwt(
                y_cwt, scales, 'mexh', sampling_period=1 / sr)
        except Exception:
            # Tentar com 'gaus1' como último recurso
            cwt_coeffs, freqs = pywt.cwt(
                y_cwt, scales, 'gaus1', sampling_period=1 / sr)

        # Energia por escala
        features['cwt_energy'] = np.sum(np.abs(cwt_coeffs)**2, axis=1)

        # Centroide CWT
        features['cwt_centroid'] = np.sum(scales[:,
                                                 np.newaxis] * np.abs(cwt_coeffs),
                                          axis=0) / (np.sum(np.abs(cwt_coeffs),
                                                            axis=0) + 1e-10)

    except Exception as e:
        warnings.warn(f"Erro na CWT: {str(e)}")
        features['cwt_energy'] = np.array([0])
        features['cwt_centroid'] = np.array([0])

    return features
