"""
Extração de Características Prosódicas
=====================================

Implementa extração de características de pitch/F0 e qualidade vocal.
Refatorado para usar sub-módulos.
"""

import warnings
from typing import Any, Dict

import librosa
import numpy as np
from scipy.stats import kurtosis, skew

from app.core.interfaces.audio import AudioData, AudioFeatures
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.domain.features.interfaces import IFeatureExtractor
from app.domain.features.types import FeatureType

from .advanced import (
    compute_apq,
    compute_dfa,
    compute_nhr,
    compute_ppq,
    compute_rap,
    compute_shdb,
    compute_spi,
    compute_vf0,
    compute_vti,
)

# Importar sub-módulos refatorados
from .pitch import (
    compute_pitch_contour_features,
    compute_pitch_slope,
    compute_pitch_strength,
    compute_voicing_probability,
)
from .quality import (
    compute_cpp,
    compute_hnr,
    compute_jitter,
    compute_shimmer,
    compute_snr,
)


class ProsodicFeatureExtractor(IFeatureExtractor):
    """Extrator de características prosódicas."""

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512, f0_min: float = 80.0,
                 f0_max: float = 400.0, window: str = 'hamming',
                 preemphasis: float = 0.97):
        """
        Inicializa o extrator de características prosódicas.

        Args:
            sr: Taxa de amostragem
            frame_length: Comprimento da janela
            hop_length: Salto entre janelas
            f0_min: Frequência fundamental mínima (Hz)
            f0_max: Frequência fundamental máxima (Hz)
            window: Tipo de janela
            preemphasis: Coeficiente de pré-ênfase (padrão: 0.97)
        """
        # Validação de parâmetros
        if sr <= 0:
            raise ValueError("Taxa de amostragem deve ser positiva")
        if frame_length <= 0:
            raise ValueError("Tamanho da janela deve ser positivo")
        if hop_length <= 0:
            raise ValueError("Tamanho do salto deve ser positivo")
        if f0_min <= 0 or f0_max <= f0_min:
            raise ValueError(
                "Frequências f0 devem ser positivas e f0_max > f0_min")

        if hop_length > frame_length:
            warnings.warn(
                "hop_length maior que frame_length pode causar "
                "perda de informação"
            )

        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.window = window
        self.preemphasis = preemphasis
        self.fmin = f0_min  # Compatibilidade com código existente
        self.fmax = f0_max  # Compatibilidade com código existente

    def extract(
            self, audio_data: AudioData) -> ProcessingResult[AudioFeatures]:
        """
        Extrai características prosódicas do áudio.

        Args:
            audio_data: Dados de áudio

        Returns:
            ProcessingResult com características prosódicas
        """
        try:
            if audio_data.samples is None or len(audio_data.samples) == 0:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Dados de áudio vazios"]
                )

            samples = audio_data.samples
            if audio_data.sample_rate != self.sr:
                # Reamostrar se necessário
                samples = librosa.resample(
                    samples, orig_sr=audio_data.sample_rate, target_sr=self.sr)

            features_dict = self.extract_features(samples)

            # Converter características para formato AudioFeatures
            features_arrays = {}
            for name, value in features_dict.items():
                if isinstance(value, (list, np.ndarray)):
                    if len(np.array(value).shape) == 0:
                        features_arrays[name] = np.array([float(value)])
                    else:
                        features_arrays[name] = np.array(value).flatten()
                else:
                    features_arrays[name] = np.array([float(value)])

            audio_features = AudioFeatures(
                features=features_arrays,
                feature_type=FeatureType.PROSODIC,
                extraction_params=self.get_extraction_params(),
                audio_metadata={'sample_rate': audio_data.sample_rate}
            )

            return ProcessingResult.success(
                data=audio_features
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def get_feature_type(self) -> FeatureType:
        """Retorna o tipo de características extraídas."""
        return FeatureType.PROSODIC

    def get_feature_names(self) -> list[str]:
        """Retorna os nomes das características extraídas."""
        return [
            'f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_median',
            'f0_q25', 'f0_q75', 'f0_iqr', 'f0_skewness', 'f0_kurtosis',
            'f0_slope', 'voicing_rate', 'voicing_prob', 'jitter', 'shimmer',
            'hnr', 'snr', 'cpp', 'rap', 'ppq', 'apq', 'vf0', 'shdb', 'nhr',
            'vti', 'spi', 'dfa'
        ]

    def get_extraction_params(self) -> Dict[str, Any]:
        """Retorna os parâmetros de extração."""
        return {
            'sr': self.sr,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'f0_min': self.f0_min,
            'f0_max': self.f0_max,
            'window': self.window,
            'preemphasis': self.preemphasis
        }

    def extract_features(self, y: np.ndarray, context: Any = None) -> Dict:
        """
        Extrai todas as características prosódicas.

        Args:
            y: Sinal de áudio
            context: Contexto de processamento (opcional)

        Returns:
            Dicionário com características prosódicas
        """
        # Validação de entrada
        if not isinstance(y, np.ndarray):
            raise TypeError("Entrada deve ser um array numpy")

        if len(y) == 0:
            warnings.warn("Sinal de áudio vazio")
            return self._get_default_prosodic_features()

        # Verificar se o sinal é muito curto
        min_length = self.frame_length
        if len(y) < min_length:
            warnings.warn(
                f"Sinal muito curto ({len(y)} samples), fazendo padding para {min_length}")
            y = np.pad(y, (0, min_length - len(y)), mode='constant')

        # Normalizar sinal
        y_max = np.max(np.abs(y))
        if y_max > 0:
            y = y / y_max

        try:
            features = {}

            # === PITCH/F0 ===

            # Fundamental Frequency usando YIN
            f0 = librosa.yin(y, fmin=self.fmin, fmax=self.fmax, sr=self.sr,
                             frame_length=self.frame_length, hop_length=self.hop_length)

            # Filtrar valores válidos de F0
            valid_f0 = f0[~np.isnan(f0)]
            valid_f0 = valid_f0[valid_f0 > 0]

            features['f0'] = f0

            if len(valid_f0) > 0:
                # Pitch Range
                features['pitch_range'] = np.max(valid_f0) - np.min(valid_f0)

                # Pitch Variance
                features['pitch_variance'] = np.var(valid_f0)

                # Pitch Mean
                features['pitch_mean'] = np.mean(valid_f0)

                # Pitch Median
                features['pitch_median'] = np.median(valid_f0)

                # Pitch Standard Deviation
                features['pitch_std'] = np.std(valid_f0)

                # Pitch Slope (tendência ao longo do tempo)
                features['pitch_slope'] = compute_pitch_slope(valid_f0)

                # Pitch Contour Features
                contour_features = compute_pitch_contour_features(f0)
                features.update(contour_features)
            else:
                # Valores padrão quando não há pitch válido
                features['pitch_range'] = 0
                features['pitch_variance'] = 0
                features['pitch_mean'] = 0
                features['pitch_median'] = 0
                features['pitch_std'] = 0
                features['pitch_slope'] = 0

            # Voicing Probability
            features['voicing_probability'] = compute_voicing_probability(f0)

            # Pitch Strength/Confidence
            features['pitch_strength'] = compute_pitch_strength(
                y, f0, self.sr, self.frame_length, self.hop_length)

            # === QUALIDADE VOCAL BÁSICA ===

            # Jitter (variabilidade de período)
            features['jitter'] = compute_jitter(y, f0, self.sr)

            # Shimmer (variabilidade de amplitude)
            features['shimmer'] = compute_shimmer(
                y, f0, self.frame_length, self.hop_length)

            # Harmonic-to-Noise Ratio
            features['hnr'] = compute_hnr(
                y, f0, self.sr, self.frame_length, self.hop_length)

            # Signal-to-Noise Ratio (aproximação)
            features['snr'] = compute_snr(
                y, self.frame_length, self.hop_length)

            # Cepstral Peak Prominence
            features['cpp'] = compute_cpp(
                y, self.sr, self.frame_length, self.hop_length)

            # === CARACTERÍSTICAS PROSÓDICAS AVANÇADAS ===

            # Medidas de Perturbação Avançadas
            features['rap'] = compute_rap(
                y, f0, self.sr)  # Relative Average Perturbation
            # Pitch Period Perturbation Quotient
            features['ppq'] = compute_ppq(y, f0, self.sr)
            # Amplitude Perturbation Quotient
            features['apq'] = compute_apq(
                y, f0, self.frame_length, self.hop_length)
            # Fundamental Frequency Variation
            features['vf0'] = compute_vf0(f0)
            features['shdb'] = compute_shdb(
                y, f0, self.frame_length, self.hop_length)  # Shimmer in dB

            # Medidas de Ruído Avançadas
            features['nhr'] = compute_nhr(
                y,
                f0,
                self.sr,
                self.frame_length,
                self.hop_length)  # Noise-to-Harmonics Ratio
            features['vti'] = compute_vti(
                y, self.frame_length, self.hop_length)      # Voice Turbulence Index
            features['spi'] = compute_spi(
                y,
                self.sr,
                self.frame_length,
                self.hop_length)      # Soft Phonation Index
            # Detrended Fluctuation Analysis
            features['dfa'] = compute_dfa(y)

            # === MAPEAMENTO PARA NOMES ESPERADOS ===
            # Mapear características já calculadas para os nomes esperados
            features['f0_mean'] = features.get('pitch_mean', 0)
            features['f0_std'] = features.get('pitch_std', 0)
            features['f0_min'] = np.min(valid_f0) if len(valid_f0) > 0 else 0
            features['f0_max'] = np.max(valid_f0) if len(valid_f0) > 0 else 0
            features['f0_range'] = features.get('pitch_range', 0)
            features['f0_median'] = features.get('pitch_median', 0)
            features['f0_q25'] = np.percentile(
                valid_f0, 25) if len(valid_f0) > 0 else 0
            features['f0_q75'] = np.percentile(
                valid_f0, 75) if len(valid_f0) > 0 else 0
            features['f0_iqr'] = features['f0_q75'] - features['f0_q25']
            features['f0_skewness'] = skew(
                valid_f0) if len(valid_f0) > 2 else 0
            features['f0_kurtosis'] = kurtosis(
                valid_f0) if len(valid_f0) > 2 else 0
            features['f0_slope'] = features.get('pitch_slope', 0)
            features['voicing_rate'] = features.get('voicing_probability', 0)
            features['voicing_prob'] = features.get('voicing_probability', 0)

            return features

        except Exception as e:
            warnings.warn(
                f"Erro na extração de características prosódicas: {e}")
            return self._get_default_prosodic_features()

    def _get_default_prosodic_features(self) -> Dict:
        """Retorna características prosódicas padrão em caso de erro."""
        return {
            'f0_mean': 0, 'f0_std': 0, 'f0_min': 0, 'f0_max': 0,
            'f0_range': 0, 'f0_median': 0, 'f0_q25': 0, 'f0_q75': 0,
            'f0_iqr': 0, 'f0_skewness': 0, 'f0_kurtosis': 0, 'f0_slope': 0,
            'voicing_rate': 0, 'voicing_prob': 0, 'jitter': 0, 'shimmer': 0,
            'hnr': 0, 'snr': 0, 'cpp': 0, 'rap': 0, 'ppq': 0, 'apq': 0,
            'vf0': 0, 'shdb': 0, 'nhr': 0, 'vti': 0, 'spi': 0, 'dfa': 0
        }


def test_prosodic_features():
    """Testa as características prosódicas."""
    # Gerar sinal de teste com pitch variável
    duration = 2.0
    sr = 22050
    t = np.linspace(0, duration, int(duration * sr))

    # Pitch modulado (vibrato)
    f0_base = 220  # A3
    vibrato_freq = 5  # Hz
    vibrato_depth = 10  # Hz

    f0_t = f0_base + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)

    # Gerar sinal com pitch variável
    y = np.sin(2 * np.pi * f0_t * t)

    # Adicionar alguns harmônicos
    y += 0.5 * np.sin(2 * np.pi * 2 * f0_t * t)
    y += 0.3 * np.sin(2 * np.pi * 3 * f0_t * t)

    # Adicionar envelope e ruído
    envelope = np.exp(-t)  # Decaimento exponencial
    y = y * envelope
    y += 0.05 * np.random.randn(len(y))

    # Extrair características
    extractor = ProsodicFeatureExtractor(sr=sr)
    features = extractor.extract_features(y)

    print("🎵 Características Prosódicas Extraídas:")
    for name, values in features.items():
        if isinstance(values, np.ndarray):
            if values.ndim == 1:
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    print(f"  {name}: média={np.mean(valid_values):.3f}, "
                          f"std={np.std(valid_values):.3f}")
                else:
                    print(f"  {name}: sem valores válidos")
            else:
                print(f"  {name}: shape={values.shape}")
        else:
            print(f"  {name}: {values:.3f}")


if __name__ == "__main__":
    test_prosodic_features()
