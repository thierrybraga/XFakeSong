import numpy as np
from typing import List, Dict, Any

from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.core.interfaces.audio import (
    AudioData, AudioFeatures, FeatureType, IFeatureExtractor
)
from app.domain.features.extractors.temporal.temporal_features import (
    TemporalFeatureExtractor
)


class TemporalExtractorWrapper(IFeatureExtractor):
    def __init__(self):
        self.extractor = TemporalFeatureExtractor()

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
                feature_type=FeatureType.TEMPORAL,
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
        return FeatureType.TEMPORAL

    def get_feature_names(self) -> List[str]:
        return [
            'energy', 'rms_energy', 'zero_crossing_rate', 'autocorr_max',
            'temporal_centroid', 'temporal_rolloff', 'temporal_flux',
            'envelope_mean', 'envelope_std', 'envelope_max', 'envelope_min'
        ]

    def get_extraction_params(self) -> Dict[str, Any]:
        return {
            'sr': self.extractor.sr,
            'frame_length': self.extractor.frame_length,
            'hop_length': self.extractor.hop_length,
            'window': self.extractor.window
        }
