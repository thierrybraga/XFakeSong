from typing import Dict, List, Any
import numpy as np
from app.core.interfaces.audio import (
    AudioData, AudioFeatures, FeatureType, IFeatureExtractor
)
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.domain.features.extractors.timefreq.timefreq_features import (
    TimeFrequencyFeatureExtractor
)


class TimeFreqExtractorWrapper(IFeatureExtractor):
    def __init__(self):
        self.extractor = TimeFrequencyFeatureExtractor(sr=22050)

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
                features={"timefreq_features": features_array},
                feature_type=FeatureType.PERCEPTUAL,
                extraction_params=self.get_extraction_params(),
                audio_metadata={"extractor": "timefreq"}
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
        return FeatureType.PERCEPTUAL

    def get_feature_names(self) -> List[str]:
        # Tenta obter nomes do extrator ou retorna genérico
        return getattr(
            self.extractor,
            'get_feature_names',
            lambda: ["spectral_centroid", "spectral_bandwidth"]
        )()

    def get_extraction_params(self) -> Dict[str, Any]:
        return {
            "sr": getattr(self.extractor, 'sr', 22050),
            "n_fft": getattr(self.extractor, 'n_fft', 2048),
            "hop_length": getattr(self.extractor, 'hop_length', 512)
        }
