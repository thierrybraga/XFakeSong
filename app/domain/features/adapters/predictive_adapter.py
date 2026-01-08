from typing import Dict, List, Any
import numpy as np
from app.core.interfaces.audio import (
    AudioData, AudioFeatures, FeatureType, IFeatureExtractor
)
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.domain.features.extractors.predictive.predictive_features import (
    PredictiveFeatureExtractor
)


class PredictiveExtractorWrapper(IFeatureExtractor):
    def __init__(self):
        self.extractor = PredictiveFeatureExtractor(sr=22050)

    def extract(
            self, audio_data: AudioData) -> ProcessingResult[AudioFeatures]:
        try:
            features_dict = self.extractor.extract_features(audio_data.samples)
            # Normalizar features para array 1D
            features_list = []
            for key, value in features_dict.items():
                if isinstance(value, np.ndarray):
                    if value.ndim > 1:
                        features_list.extend(value.flatten())
                    else:
                        features_list.extend(value)
                else:
                    features_list.append(float(value))

            features_array = np.array(features_list)
            audio_features = AudioFeatures(
                features={"predictive_features": features_array},
                feature_type=FeatureType.ADVANCED,
                extraction_params={"sr": 22050},
                audio_metadata={"extractor": "predictive"}
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

    def extract_features(
        self, audio_data: AudioData
    ) -> ProcessingResult[AudioFeatures]:
        return self.extract(audio_data)

    def get_feature_type(self) -> FeatureType:
        return FeatureType.ADVANCED

    def get_feature_names(self) -> List[str]:
        return getattr(self.extractor, 'get_feature_names', lambda: ["lpc"])()

    def get_extraction_params(self) -> Dict[str, Any]:
        return {
            "sr": getattr(self.extractor, 'sr', 22050),
            "order": getattr(self.extractor, 'order', 12)
        }
