import numpy as np
import librosa
import scipy.signal
import scipy.fftpack
import warnings
from typing import Dict


def hz_to_erb(hz: float) -> float:
    """Converte Hz para escala ERB."""
    return 21.4 * np.log10(1 + 0.00437 * hz)


def erb_to_hz(erb: float) -> float:
    """Converte escala ERB para Hz."""
    return (10**(erb / 21.4) - 1) / 0.00437


def apply_gammatone_filterbank(y: np.ndarray, sr: int, frame_length: int, hop_length: int,
                               fmin: float, fmax: float, n_gammatone: int, gammatone_order: int) -> np.ndarray:
    """Aplica banco de filtros Gammatone."""
    # Implementação simplificada do banco de filtros Gammatone

    # Frequências centrais em escala ERB
    erb_low = hz_to_erb(fmin)
    erb_high = hz_to_erb(fmax)
    erb_centers = np.linspace(erb_low, erb_high, n_gammatone)
    center_freqs = erb_to_hz(erb_centers)

    # Dividir em frames
    frames = librosa.util.frame(y, frame_length=frame_length,
                                hop_length=hop_length, axis=0)

    gammatone_spectrum = np.zeros((n_gammatone, frames.shape[1]))

    for i, fc in enumerate(center_freqs):
        # Largura de banda ERB
        erb = 24.7 * (4.37 * fc / 1000 + 1)

        # Filtro Gammatone simplificado usando filtro Butterworth
        try:
            sos = scipy.signal.butter(gammatone_order,
                                      [max(fc - erb / 2, 1),
                                       min(fc + erb / 2, sr / 2)],
                                      btype='band', fs=sr, output='sos')

            for j in range(frames.shape[1]):
                frame = frames[:, j]
                filtered = scipy.signal.sosfilt(sos, frame)
                gammatone_spectrum[i, j] = np.sum(filtered**2)
        except BaseException:
            gammatone_spectrum[i, :] = 0

    return gammatone_spectrum


def extract_gtcc_features(y: np.ndarray, sr: int, frame_length: int, hop_length: int,
                          n_mfcc: int, fmin: float, fmax: float) -> Dict:
    """Extrai Gammatone Cepstral Coefficients."""
    features = {}

    try:
        # Parâmetros internos
        n_gammatone = 64
        gammatone_order = 4

        # Aplicar banco de filtros Gammatone
        gammatone_spectrum = apply_gammatone_filterbank(y, sr, frame_length, hop_length,
                                                        fmin, fmax, n_gammatone, gammatone_order)

        # Aplicar logaritmo
        log_spectrum = np.log(gammatone_spectrum + 1e-10)

        # Aplicar DCT
        gtcc = scipy.fftpack.dct(log_spectrum, axis=0, norm='ortho')[:n_mfcc]

        features['gtcc'] = gtcc
        features['gtcc_delta'] = librosa.feature.delta(gtcc)
        features['gtcc_delta2'] = librosa.feature.delta(gtcc, order=2)

    except Exception as e:
        warnings.warn(f"Erro na extração GTCC: {str(e)}")
        features['gtcc'] = np.zeros((n_mfcc, 1))
        features['gtcc_delta'] = np.zeros((n_mfcc, 1))
        features['gtcc_delta2'] = np.zeros((n_mfcc, 1))

    return features
