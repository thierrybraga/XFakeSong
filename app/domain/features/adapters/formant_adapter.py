from typing import Dict, List, Any
import numpy as np
from app.core.interfaces.audio import (
    AudioData, AudioFeatures, FeatureType, IFeatureExtractor
)
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.domain.features.extractors.formant.formant_features import (
    FormantFeatureExtractor
)


class FormantExtractorWrapper(IFeatureExtractor):
    def __init__(self):
        self.extractor = FormantFeatureExtractor(sr=22050)

    def extract(self, audio_data: AudioData) -> ProcessingResult:
        try:
            features_dict = self.extractor.extract_features(audio_data.samples)

            # Normalizar todas as features para arrays 1D
            normalized_features = {}
            for key, value in features_dict.items():
                if isinstance(value, (int, float)):
                    normalized_features[key] = np.array([value])
                elif isinstance(value, np.ndarray):
                    normalized_features[key] = value.flatten()
                else:
                    normalized_features[key] = np.array([float(value)])

            audio_features = AudioFeatures(
                features=normalized_features,
                feature_type=FeatureType.PROSODIC,
                extraction_params=self.get_extraction_params(),
                audio_metadata={"extractor": "formant"}
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=audio_features
            )
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na extração de formantes: {str(e)}"]
            )

    def extract_features(self, audio_data: AudioData) -> ProcessingResult:
        return self.extract(audio_data)

    def get_feature_type(self) -> FeatureType:
        return FeatureType.PROSODIC

    def get_feature_names(self) -> List[str]:
        return self.extractor.get_feature_names()

    def get_extraction_params(self) -> Dict[str, Any]:
        return {
            "sr": self.extractor.sr,
            "n_formants": self.extractor.n_formants,
            "frame_length": self.extractor.frame_length,
            "hop_length": self.extractor.hop_length
        }
