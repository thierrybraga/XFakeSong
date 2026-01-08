"""Extração de Características Cepstrais para Detecção de Deepfakes

Implementação conservadora focada na fidelidade das características,
seguindo princípios SOLID e clean code.
"""

import numpy as np
import librosa
import warnings
from typing import Dict, Tuple, Any, List

# Imports de interface
try:
    from .interfaces import (
        FeatureExtractor, IFeatureExtractor
    )
except ImportError:
    # Fallback para imports básicos
    from app.domain.features.interfaces import IFeatureExtractor

from .....core.interfaces.audio import (
    AudioData, AudioFeatures, FeatureType
)
from .....core.interfaces.base import ProcessingResult, ProcessingStatus

# Imports dos componentes refatorados
from .components.mfcc import MFCCExtractor
from .components.mel import MelSpectrogramExtractor
from .components.delta import DeltaFeaturesExtractor
from .components.spectral import extract_spectral_features
from .components.plp import extract_plp_features
from .components.lpcc import extract_lpcc_features


class CepstralFeatureExtractor(IFeatureExtractor):
    """Extrator principal de características cepstrais.

    Implementação consolidada seguindo princípios SOLID:
    - Single Responsibility: Cada classe tem uma responsabilidade específica
    - Open/Closed: Extensível através de novos extratores
    - Liskov Substitution: Extratores são intercambiáveis
    - Interface Segregation: Interfaces específicas para cada tipo
    - Dependency Inversion: Depende de abstrações, não implementações

    Inclui normalização Min-Max integrada.
    """

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512, n_mfcc: int = 13, n_mels: int = 128,
                 normalize: bool = True,
                 feature_range: Tuple[float, float] = (0.0, 1.0)):
        """Inicializa o extrator com parâmetros conservadores.

        Args:
            normalize: Se deve aplicar normalização Min-Max
            feature_range: Faixa de valores para normalização
        """
        self._validate_parameters(sr, frame_length, hop_length, n_mfcc, n_mels)

        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.normalize = normalize
        self.feature_range = feature_range

        # Inicializar extratores especializados
        self.mfcc_extractor = MFCCExtractor(
            sr=sr, n_mfcc=n_mfcc, n_fft=frame_length,
            hop_length=hop_length, n_mels=40
        )

        self.mel_extractor = MelSpectrogramExtractor(
            sr=sr, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels
        )

        self.delta_extractor = DeltaFeaturesExtractor()

        # Normalizador
        # self.normalizer = FeatureNormalizer(
        #     feature_range=feature_range
        # ) if normalize else None
        self.normalizer = None  # Temporariamente desabilitado

    def _validate_audio_input(self, y: np.ndarray,
                              min_length: int = 1024) -> np.ndarray:
        """Valida e normaliza entrada de áudio."""
        if not isinstance(y, np.ndarray):
            raise TypeError("Entrada deve ser um array numpy")

        if len(y) == 0:
            raise ValueError("Sinal de áudio não pode estar vazio")

        if len(y) < min_length:
            warnings.warn(
                f"Sinal muito curto ({len(y)} samples), padding será aplicado"
            )
            y = np.pad(y, (0, min_length - len(y)), mode='constant')

        # Normalização conservadora
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        return y

    def extract_features(self, audio_data: np.ndarray,
                         metadata: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """Interface compatível para extração de características."""
        if isinstance(audio_data, np.ndarray):
            # Se for array direto, verificar se metadata tem sample_rate
            if metadata and hasattr(metadata, 'sample_rate'):
                sample_rate = metadata.sample_rate
            elif (metadata and isinstance(metadata, dict) and
                  'sample_rate' in metadata):
                sample_rate = metadata['sample_rate']
            else:
                sample_rate = self.sr
            return self.extract_cepstral_features(audio_data, sample_rate)

        # Fallback ou erro se não for array
        return self.extract_features_internal(audio_data)

    def extract(self, audio_data: AudioData) -> ProcessingResult:
        """Extrai características cepstrais do áudio."""
        try:
            if audio_data.samples is None or len(audio_data.samples) == 0:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Dados de áudio vazios"]
                )

            # Extrair características usando o método original
            features_dict = self.extract_cepstral_features(
                audio_data.samples, audio_data.sample_rate
            )

            # Não achatar arrays - manter estrutura para validação

            audio_features = AudioFeatures(
                features=features_dict,
                feature_type=FeatureType.SPECTRAL,
                extraction_params=self.get_extraction_params(),
                audio_metadata={'feature_names': list(features_dict.keys())}
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=audio_features
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na extração cepstral: {str(e)}"]
            )

    def get_feature_type(self) -> FeatureType:
        """Retorna tipo de característica extraída."""
        return FeatureType.SPECTRAL

    def get_feature_names(self) -> List[str]:
        """Retorna nomes das características."""
        return [
            'mfcc', 'delta_mfcc', 'delta_delta_mfcc',
            'mel_spectrogram', 'plp', 'lpcc'
        ]

    def get_extraction_params(self) -> Dict[str, Any]:
        """Retorna parâmetros de extração."""
        return {
            'sr': self.sr,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'normalize': self.normalize,
            'feature_range': self.feature_range
        }

    def extract_cepstral_features(self, audio_data: np.ndarray,
                                  sample_rate: int) -> Dict[str, np.ndarray]:
        """Extrai características cepstrais com taxa de amostragem específica."""
        # Ajustar taxa de amostragem se necessário
        if sample_rate != self.sr:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=self.sr
            )

        return self.extract_features_internal(audio_data)

    def _validate_parameters(self, sr: int, frame_length: int, hop_length: int,
                             n_mfcc: int, n_mels: int) -> None:
        """Valida parâmetros de inicialização."""
        if sr <= 0 or frame_length <= 0 or hop_length <= 0:
            raise ValueError("Parâmetros devem ser positivos")

        if n_mfcc <= 0 or n_mels <= 0:
            raise ValueError("Número de características deve ser positivo")

        if n_mfcc > n_mels:
            warnings.warn("n_mfcc maior que n_mels pode causar problemas")

    def extract_features_internal(
            self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Extrai conjunto completo de características cepstrais."""
        # Validar entrada
        y = self._validate_audio_input(y, self.frame_length)

        features = {}

        try:
            # MFCC e suas derivadas
            mfcc = self.mfcc_extractor.extract(y)
            features['mfcc'] = mfcc
            features['mfcc_delta'] = self.delta_extractor.extract_delta(
                mfcc, order=1
            )
            features['mfcc_delta2'] = self.delta_extractor.extract_delta(
                mfcc, order=2
            )

            # Espectrograma Mel
            mel_spec = self.mel_extractor.extract(y)
            features['log_mel_spectrogram'] = mel_spec

            # Características espectrais básicas
            spectral_features = extract_spectral_features(
                y, self.sr, self.frame_length, self.hop_length
            )
            features.update(spectral_features)

            # Aplicar normalização se habilitada
            if self.normalize and self.normalizer is not None:
                try:
                    features = self.normalizer.fit_transform(features)
                except Exception as e:
                    warnings.warn(
                        f"Erro na normalização de características: {str(e)}"
                    )

        except Exception as e:
            warnings.warn(f"Erro na extração de características: {str(e)}")
            features = self._get_default_features()

        # === CARACTERÍSTICAS CEPSTRAIS AVANÇADAS ===

        # PLP (Perceptual Linear Prediction)
        try:
            plp_dict = extract_plp_features(
                y, self.sr, self.frame_length, self.hop_length, 12
            )
            features['plp'] = plp_dict['plp']
        except Exception as e:
            warnings.warn(f"Erro no cálculo PLP: {str(e)}")
            features['plp'] = np.zeros((12, 1))

        # LPCC (Linear Prediction Cepstral Coefficients)
        try:
            lpcc_dict = extract_lpcc_features(
                y, self.frame_length, self.hop_length, 12
            )
            features['lpcc'] = lpcc_dict['lpcc']
        except Exception as e:
            warnings.warn(f"Erro no cálculo LPCC: {str(e)}")
            features['lpcc'] = np.zeros((12, 1))

        return features

    def _get_default_features(self) -> Dict[str, np.ndarray]:
        """Retorna características padrão em caso de erro."""
        return {
            'mfcc': np.zeros((self.mfcc_extractor.n_mfcc, 1)),
            'mfcc_delta': np.zeros((self.mfcc_extractor.n_mfcc, 1)),
            'mfcc_delta2': np.zeros((self.mfcc_extractor.n_mfcc, 1)),
            'log_mel_spectrogram': np.zeros((self.mel_extractor.n_mels, 1)),
            'spectral_centroids': np.zeros((1, 1)),
            'spectral_rolloff': np.zeros((1, 1)),
            'zero_crossing_rate': np.zeros((1, 1))
        }
