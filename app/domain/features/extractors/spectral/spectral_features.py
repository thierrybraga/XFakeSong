"""
Extração de Características Espectrais
=====================================

Implementa extração de características do espectro de magnitude e contraste.
"""

import numpy as np
import librosa
import warnings
from typing import Dict, Optional, Tuple, List, Any

from app.domain.features.interfaces import IFeatureExtractor
from .....core.interfaces.audio import AudioData, AudioFeatures, FeatureType
from .....core.interfaces.base import ProcessingResult, ProcessingStatus

from .components.magnitude import (
    compute_spectral_slope,
    compute_spectral_kurtosis,
    compute_spectral_skewness,
    compute_spectral_spread,
    compute_spectral_entropy
)
from .components.contrast import (
    compute_subband_energy_ratios,
    compute_high_freq_content
)
from .components.advanced import (
    compute_spectral_flux,
    compute_spectral_decrease,
    compute_spectral_crest,
    compute_spectral_irregularity,
    compute_spectral_roughness,
    compute_spectral_inharmonicity
)
from .components.utils import (
    apply_preemphasis,
    get_default_spectral_features
)


class SpectralFeatureExtractorCore:
    """Implementação core do extrator espectral."""

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512, window: str = 'hamming',
                 fmin: float = 80.0, fmax: float = 8000.0,
                 preemphasis: float = 0.97):
        """Inicializa o extrator core."""
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window = window
        self.fmin = fmin
        self.fmax = min(fmax, sr // 2)
        self.preemphasis = preemphasis

        if sr <= 0:
            raise ValueError("Taxa de amostragem deve ser positiva")
        if frame_length <= 0 or hop_length <= 0:
            raise ValueError("Comprimentos de janela devem ser positivos")
        if hop_length > frame_length:
            warnings.warn(
                "hop_length maior que frame_length pode causar aliasing")


class SpectralFeatureExtractor(IFeatureExtractor):
    """Extrator de características espectrais."""

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512, window: str = 'hamming',
                 fmin: float = 80.0, fmax: float = 8000.0, preemphasis: float = 0.97):
        """
        Inicializa o extrator de características espectrais.

        Args:
            sr: Taxa de amostragem
            frame_length: Comprimento da janela
            hop_length: Salto entre janelas
            window: Tipo de janela para STFT
            fmin: Frequência mínima (padrão: 80Hz)
            fmax: Frequência máxima (padrão: 8000Hz)
            preemphasis: Coeficiente de pré-ênfase (padrão: 0.97)
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window = window
        self.fmin = fmin
        self.fmax = min(fmax, sr // 2)  # Garantir que não exceda Nyquist
        self.preemphasis = preemphasis

        # Validação de parâmetros
        if sr <= 0:
            raise ValueError("Taxa de amostragem deve ser positiva")
        if frame_length <= 0 or hop_length <= 0:
            raise ValueError("Comprimentos de janela devem ser positivos")
        if hop_length > frame_length:
            warnings.warn(
                "hop_length maior que frame_length pode causar aliasing")

        # Inicializar extrator core
        self.core = SpectralFeatureExtractorCore(
            sr=sr, frame_length=frame_length, hop_length=hop_length,
            window=window, fmin=fmin, fmax=fmax, preemphasis=preemphasis
        )

    def extract(
            self, audio_data: AudioData) -> ProcessingResult[AudioFeatures]:
        """Extrai características espectrais do áudio."""
        try:
            if audio_data.samples is None or len(audio_data.samples) == 0:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Dados de áudio vazios"]
                )

            # Extrair características usando o método original
            features_dict = self.extract_features(audio_data.samples)

            # Converter para formato AudioFeatures com características nomeadas
            features_arrays = {}
            for name, values in features_dict.items():
                if isinstance(values, np.ndarray):
                    if values.ndim > 1:
                        # Achatar arrays multidimensionais
                        values = values.flatten()
                    features_arrays[name] = values
                else:
                    features_arrays[name] = np.array([float(values)])

            audio_features = AudioFeatures(
                features=features_arrays,
                feature_type=FeatureType.SPECTRAL,
                extraction_params=self.get_extraction_params(),
                audio_metadata={'sample_rate': audio_data.sample_rate}
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=audio_features
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def get_feature_type(self) -> FeatureType:
        """Retorna tipo de característica extraída."""
        return FeatureType.SPECTRAL

    def get_feature_names(self) -> List[str]:
        """Retorna nomes das características."""
        return [
            'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
            'spectral_contrast', 'zero_crossing_rate', 'spectral_flatness',
            'spectral_slope', 'spectral_kurtosis', 'spectral_skewness',
            'spectral_spread', 'spectral_entropy'
        ]

    def get_extraction_params(self) -> Dict[str, Any]:
        """Retorna parâmetros de extração."""
        return {
            'sr': self.sr,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'window': self.window,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'preemphasis': self.preemphasis
        }

    def extract_features(self, y: np.ndarray) -> Dict:
        """
        Extrai todas as características espectrais do sinal.

        Args:
            y: Sinal de áudio

        Returns:
            Dicionário com características espectrais
        """
        # Aplicar pré-ênfase
        y = apply_preemphasis(y, self.preemphasis)
        # Validação de entrada
        if not isinstance(y, np.ndarray):
            raise TypeError("Entrada deve ser um array numpy")

        if len(y) == 0:
            raise ValueError("Sinal de áudio não pode estar vazio")

        if len(y) < self.frame_length:
            warnings.warn(
                f"Sinal muito curto ({
                    len(y)} samples), padding será aplicado")

        # Normalizar sinal para evitar overflow
        y_norm = y / (np.max(np.abs(y)) + 1e-10)

        features = {}

        try:
            # Calcular espectrograma
            S = np.abs(librosa.stft(y_norm, n_fft=self.frame_length,
                                    hop_length=self.hop_length))

            # Frequências correspondentes
            freqs = librosa.fft_frequencies(
                sr=self.sr, n_fft=self.frame_length)

            # === CARACTERÍSTICAS DO ESPECTRO DE MAGNITUDE ===

            # Spectral Centroid
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                S=S, sr=self.sr, hop_length=self.hop_length)[0]

            # Spectral Rolloff (85%, 95%)
            features['spectral_rolloff_85'] = librosa.feature.spectral_rolloff(
                S=S, sr=self.sr, roll_percent=0.85, hop_length=self.hop_length)[0]
            features['spectral_rolloff_95'] = librosa.feature.spectral_rolloff(
                S=S, sr=self.sr, roll_percent=0.95, hop_length=self.hop_length)[0]

            # Spectral Bandwidth
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
                S=S, sr=self.sr, hop_length=self.hop_length)[0]

            # Spectral Flatness
            features['spectral_flatness'] = librosa.feature.spectral_flatness(
                S=S, hop_length=self.hop_length)[0]

            # Spectral Slope
            features['spectral_slope'] = compute_spectral_slope(S, freqs)

            # Spectral Kurtosis
            features['spectral_kurtosis'] = compute_spectral_kurtosis(S, freqs)

            # Spectral Skewness
            features['spectral_skewness'] = compute_spectral_skewness(S, freqs)

            # Spectral Spread (Variance)
            centroid = features['spectral_centroid']
            features['spectral_spread'] = compute_spectral_spread(
                S, freqs, centroid)

            # Spectral Entropy
            features['spectral_entropy'] = compute_spectral_entropy(S)

            # === CARACTERÍSTICAS DE CONTRASTE ===

            # Spectral Contrast
            features['spectral_contrast'] = librosa.feature.spectral_contrast(
                S=S, sr=self.sr, hop_length=self.hop_length)

            # Sub-band Energy Ratios
            subbands = compute_subband_energy_ratios(S, freqs)
            features.update(subbands)

            # High-Frequency Content
            features['high_freq_content'] = compute_high_freq_content(S, freqs)

            # === CARACTERÍSTICAS ESPECTRAIS AVANÇADAS ===

            # Spectral Flux
            features['spectral_flux'] = compute_spectral_flux(S)

            # Spectral Decrease
            features['spectral_decrease'] = compute_spectral_decrease(S, freqs)

            # Spectral Crest Factor
            features['spectral_crest'] = compute_spectral_crest(S)

            # Spectral Irregularity
            features['spectral_irregularity'] = compute_spectral_irregularity(
                S)

            # Spectral Roughness
            features['spectral_roughness'] = compute_spectral_roughness(
                S, freqs)

            # Spectral Inharmonicity
            features['spectral_inharmonicity'] = compute_spectral_inharmonicity(
                S, freqs)

            # === MAPEAMENTO PARA NOMES ESPERADOS ===
            # Mapear spectral_rolloff_85 para spectral_rolloff (nome esperado)
            if 'spectral_rolloff_85' in features:
                features['spectral_rolloff'] = features['spectral_rolloff_85']

        except Exception as e:
            warnings.warn(f"Erro na extração de características: {str(e)}")
            # Retornar características padrão em caso de erro
            features = self._get_default_features()

        return features

    def _get_default_features(self) -> Dict:
        """
        Retorna características padrão em caso de erro.
        Mantido para compatibilidade, mas delega para função utilitária.

        Returns:
            Dicionário com valores padrão
        """
        return get_default_spectral_features()
