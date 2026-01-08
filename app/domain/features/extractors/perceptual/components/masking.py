"""
Extração de características de mascaramento auditivo.
"""
import numpy as np
import librosa
from typing import Dict
# from .utils import hz_to_bark


def extract_masking_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de mascaramento auditivo (versão otimizada)."""
    features = {}

    try:
        # Calcular espectrograma
        stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        power = magnitude ** 2

        # Frequências dos bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

        # Converter todas as frequências para Bark uma vez
        # bark_freqs = hz_to_bark(freqs) # Não utilizado no loop principal mas
        # mantido se necessário no futuro

        # Calcular características simplificadas de mascaramento
        # Usar apenas algumas amostras para evitar travamento
        n_samples = min(10, power.shape[1])  # Máximo 10 frames
        sample_indices = np.linspace(
            0, power.shape[1] - 1, n_samples, dtype=int)

        masking_values = []

        for idx in sample_indices:
            frame_power = power[:, idx]

            # Encontrar picos significativos
            threshold = np.percentile(frame_power, 90)  # Top 10%
            significant_bins = frame_power > threshold

            if np.any(significant_bins):
                # Calcular mascaramento apenas para picos significativos
                masking_total = 0

                for i in np.where(significant_bins)[0]:
                    # Mascaramento local (vizinhança)
                    start_idx = max(0, i - 5)
                    end_idx = min(len(frame_power), i + 6)

                    local_masking = np.sum(frame_power[start_idx:end_idx])
                    masking_total += local_masking

                masking_values.append(masking_total)
            else:
                masking_values.append(0.0)

        masking_values = np.array(masking_values)

        # Características de mascaramento
        if len(masking_values) > 0:
            features['masking_threshold_mean'] = np.mean(masking_values)
            features['masking_threshold_std'] = np.std(masking_values)
            features['masking_threshold_max'] = np.max(masking_values)
        else:
            features['masking_threshold_mean'] = 0.0
            features['masking_threshold_std'] = 0.0
            features['masking_threshold_max'] = 0.0

    except Exception as e:
        print(f"Erro no masking: {str(e)}")
        features['masking_threshold_mean'] = 0.0
        features['masking_threshold_std'] = 0.0
        features['masking_threshold_max'] = 0.0

    # Centroide espectral perceptual (considerando mascaramento)
    try:
        perceptual_centroids = []

        for i, frame_power in enumerate(power.T):
            if i < len(masking_values) and masking_values[i] > 0:
                # Ponderar por limiar de mascaramento
                weights = frame_power / (masking_values[i] + 1e-10)

                if np.sum(weights) > 0:
                    centroid = np.sum(freqs * weights) / np.sum(weights)
                    perceptual_centroids.append(centroid)

        if perceptual_centroids:
            features['perceptual_centroid_mean'] = np.mean(
                perceptual_centroids)
            features['perceptual_centroid_std'] = np.std(perceptual_centroids)
        else:
            features['perceptual_centroid_mean'] = 0
            features['perceptual_centroid_std'] = 0
    except BaseException:
        features['perceptual_centroid_mean'] = 0
        features['perceptual_centroid_std'] = 0

    return features
