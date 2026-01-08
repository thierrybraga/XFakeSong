from typing import Dict, List, Any
import numpy as np
from app.core.interfaces.audio import (
    AudioData, AudioFeatures, FeatureType, IFeatureExtractor
)
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.domain.features.extractors.prosodic.prosodic_features import (
    ProsodicFeatureExtractor
)


class ProsodicExtractorWrapper(IFeatureExtractor):
    def __init__(self):
        self.extractor = ProsodicFeatureExtractor()

    def extract(self, audio_data: AudioData) -> ProcessingResult:
        try:
            features_dict = self.extractor.extract_features(audio_data.samples)

            # Normalizar features para arrays 1D
            normalized_features = {}
            for key, value in features_dict.items():
                if isinstance(value, (int, float)):
                    # Converter escalares para arrays 1D
                    normalized_features[key] = np.array([value])
                elif isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        # Converter arrays 0D para 1D
                        normalized_features[key] = np.array([value.item()])
                    elif value.ndim == 1:
                        # Manter arrays 1D
                        normalized_features[key] = value
                    else:
                        # Achatar arrays multidimensionais
                        normalized_features[key] = value.flatten()
                else:
                    # Converter outros tipos para arrays
                    normalized_features[key] = np.array([float(value)])

            # Converter para AudioFeatures
            audio_features = AudioFeatures(
                features=normalized_features,
                feature_type=FeatureType.PROSODIC,
                extraction_params={'sample_rate': audio_data.sample_rate},
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

    def extract_features(self, audio_data: AudioData) -> ProcessingResult:
        return self.extract(audio_data)

    def get_feature_type(self) -> FeatureType:
        return FeatureType.PROSODIC

    def get_feature_names(self) -> List[str]:
        return [
            'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope',
            'intensity_mean', 'intensity_std', 'intensity_range',
            'speaking_rate', 'pause_rate', 'rhythm_score',
            'stress_pattern', 'intonation_contour'
        ]

    def get_extraction_params(self) -> Dict[str, Any]:
        return {
            'f0_min': self.extractor.f0_min,
            'f0_max': self.extractor.f0_max,
            'frame_length': self.extractor.frame_length,
            'hop_length': self.extractor.hop_length,
            'sr': self.extractor.sr
        }
