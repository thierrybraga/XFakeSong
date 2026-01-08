from typing import Dict, List, Any
import numpy as np
from app.core.interfaces.audio import (
    AudioData, AudioFeatures, FeatureType, IFeatureExtractor
)
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.domain.features.extractors.cepstral.cepstral_features import (
    CepstralFeatureExtractor
)


class CepstralExtractorWrapper(IFeatureExtractor):
    def __init__(self):
        self.extractor = CepstralFeatureExtractor(sr=22050)

    def extract(
            self, audio_data: AudioData) -> ProcessingResult[AudioFeatures]:
        try:
            # CepstralFeatureExtractor já retorna ProcessingResult, mas precisamos adaptar
            # para o formato esperado pelo pipeline se necessário, ou retornar direto se compatível.
            # Assumindo que o extrator retorna ProcessingResult com dicionário
            # de features.

            # Verificando a implementação do extrator, ele parece ser independente.
            # Vamos assumir que ele retorna um dict de features como os outros.
            features_dict = self.extractor.extract_features(audio_data.samples)

            # Normalizar features
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
                feature_type=FeatureType.CEPSTRAL,
                extraction_params=self.get_extraction_params(),
                audio_metadata={"extractor": "cepstral"}
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

    def extract_features(
        self, audio_data: AudioData
    ) -> ProcessingResult[AudioFeatures]:
        return self.extract(audio_data)

    def get_feature_type(self) -> FeatureType:
        return FeatureType.CEPSTRAL

    def get_feature_names(self) -> List[str]:
        # Retorna lista genérica ou específica se disponível
        return getattr(
            self.extractor,
            'get_feature_names',
            lambda: ["mfcc", "delta", "delta2"]
        )()

    def get_extraction_params(self) -> Dict[str, Any]:
        return {
            "sr": getattr(self.extractor, 'sr', 22050),
            "n_mfcc": getattr(self.extractor, 'n_mfcc', 13)
        }
