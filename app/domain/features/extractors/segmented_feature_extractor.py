#!/usr/bin/env python3
"""
Extrator de Features Segmentadas
===============================

Este módulo implementa a extração de características de áudio usando segmentação.
O áudio é dividido em segmentos menores para análise mais detalhada.
"""

import numpy as np
from typing import List, Any, Optional
from pathlib import Path
import librosa
from dataclasses import dataclass

# Importações do projeto
from ....core.interfaces.audio import AudioData
from ....core.interfaces.base import ProcessingResult, ProcessingStatus
from ..models.segmented_features import SegmentedFeatures

# Importar extratores específicos
from .spectral.spectral_features import SpectralFeatureExtractor
from .cepstral.cepstral_features import CepstralFeatureExtractor
from .temporal.temporal_features import TemporalFeatureExtractor
from .prosodic.prosodic_features import ProsodicFeatureExtractor
from .formant.formant_features import FormantFeatureExtractor
from .voice_quality.voice_quality_features import VoiceQualityFeatureExtractor
from .perceptual.perceptual_features import PerceptualFeatureExtractor
from .complexity.complexity_features import ComplexityFeatureExtractor
from .transform.transform_features import TransformFeatureExtractor
from .timefreq.timefreq_features import TimeFrequencyFeatureExtractor
from .predictive.predictive_features import PredictiveFeatureExtractor
from .speech.speech_features import SpeechFeatureExtractor


@dataclass
class SegmentedExtractionConfig:
    """Configuração para extração segmentada."""
    segment_duration: float = 1.0
    overlap_ratio: float = 0.0
    target_sample_rate: int = 16000
    normalize_segments: bool = True
    remove_silence: bool = False

    # Flags para categorias de features
    extract_spectral: bool = True
    extract_cepstral: bool = True
    extract_temporal: bool = True
    extract_prosodic: bool = True
    extract_formant: bool = True
    extract_voice_quality: bool = True
    extract_perceptual: bool = True
    extract_complexity: bool = True
    extract_transform: bool = True
    extract_timefreq: bool = True
    extract_predictive: bool = True
    extract_speech: bool = True

    # Configuração de exportação CSV
    export_csv: bool = False
    csv_config: Optional[Any] = None


class SegmentedFeatureExtractor:
    """Extrator de características usando segmentação de áudio."""

    def __init__(self, config: SegmentedExtractionConfig):
        self.config = config
        self._init_services()

    def _init_services(self):
        """Inicializa os serviços de extração."""
        self.extractors = {}

        # Inicializar extratores baseados na configuração
        if self.config.extract_spectral:
            self.extractors['spectral'] = SpectralFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_cepstral:
            self.extractors['cepstral'] = CepstralFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_temporal:
            self.extractors['temporal'] = TemporalFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_prosodic:
            self.extractors['prosodic'] = ProsodicFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_formant:
            self.extractors['formant'] = FormantFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_voice_quality:
            self.extractors['voice_quality'] = VoiceQualityFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_perceptual:
            self.extractors['perceptual'] = PerceptualFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_complexity:
            self.extractors['complexity'] = ComplexityFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_transform:
            self.extractors['transform'] = TransformFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_timefreq:
            self.extractors['timefreq'] = TimeFrequencyFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_predictive:
            self.extractors['predictive'] = PredictiveFeatureExtractor(
                sr=self.config.target_sample_rate
            )

        if self.config.extract_speech:
            self.extractors['speech'] = SpeechFeatureExtractor(
                sr=self.config.target_sample_rate
            )

    def extract(self, audio_data: AudioData) -> ProcessingResult[List[SegmentedFeatures]]:
        """Extrai características de um objeto AudioData."""
        try:
            y = audio_data.samples
            sr = audio_data.sample_rate

            # Converter para float32 se necessário
            y = y.astype(np.float32)

            # Resample se necessário
            if sr != self.config.target_sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.config.target_sample_rate)
                sr = self.config.target_sample_rate

            # Segmentar áudio
            segments = self._segment_audio(y, sr)

            # Extrair features de cada segmento
            segmented_features = []
            for i, segment in enumerate(segments):
                features = self._extract_segment_features(segment, sr, i)
                segmented_features.append(features)

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=segmented_features,
                metadata={"segments_count": len(segmented_features)}
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                data=[],
                errors=[f"Erro na extração: {str(e)}"]
            )

    def extract_from_file(
            self, file_path: str, label: int = None) -> ProcessingResult[List[SegmentedFeatures]]:
        """Extrai características de um arquivo de áudio e opcionalmente exporta para CSV."""
        try:
            # Carregar áudio
            y, sr = librosa.load(file_path, sr=self.config.target_sample_rate)

            # Segmentar áudio
            segments = self._segment_audio(y, sr)

            # Extrair features de cada segmento
            segmented_features = []
            for i, segment in enumerate(segments):
                features = self._extract_segment_features(segment, sr, i)
                segmented_features.append(features)

            # Exportar para CSV se configurado
            if self.config.export_csv and hasattr(
                    self.config, 'csv_config') and self.config.csv_config:
                self._export_individual_csv(
                    segmented_features, file_path, label)

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=segmented_features,
                metadata={"segments_count": len(segmented_features)}
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                data=[],
                errors=[f"Erro na extração: {str(e)}"]
            )

    def _segment_audio(self, y: np.ndarray, sr: int) -> List[np.ndarray]:
        """Segmenta o áudio em intervalos menores."""
        segment_samples = int(self.config.segment_duration * sr)
        overlap_samples = int(segment_samples * self.config.overlap_ratio)
        step_samples = segment_samples - overlap_samples

        segments = []
        start = 0

        while start + segment_samples <= len(y):
            segment = y[start:start + segment_samples]

            if self.config.normalize_segments:
                segment = librosa.util.normalize(segment)

            segments.append(segment)
            start += step_samples

        return segments

    def _extract_segment_features(
            self, segment: np.ndarray, sr: int, segment_idx: int) -> SegmentedFeatures:
        """Extrai características de um segmento específico."""
        all_features = []
        feature_names = []
        category_features = {}

        # Extrair features de cada categoria ativa
        print(f"DEBUG: Extratores disponíveis: {list(self.extractors.keys())}")
        for category, extractor in self.extractors.items():
            print(f"DEBUG: Processando categoria {category}")
            try:
                # Verificar primeiro categorias especiais que usam
                # extract_features
                if category in ['speech', 'temporal', 'perceptual']:
                    print(
                        f"DEBUG: Extraindo features de {category} para segmento {segment_idx}")
                    print(f"DEBUG: Tamanho do segmento: {len(segment)}")
                    features = extractor.extract_features(segment)
                    print(
                        f"DEBUG: Features {category} extraídas: {
                            len(features) if features else 0}")
                    if features:
                        print(
                            f"DEBUG: Primeiras 5 features {category}: {
                                list(
                                    features.keys())[
                                    :5]}")
                elif category == 'cepstral':
                    print(
                        f"DEBUG: Extraindo features de cepstral para segmento {segment_idx}")
                    features = extractor.extract_features_internal(segment)
                    print(
                        f"DEBUG: Features cepstral extraídas: {
                            len(features) if features else 0}")
                    if features:
                        print(
                            f"DEBUG: Primeiras 5 features cepstral: {
                                list(
                                    features.keys())[
                                    :5]}")
                elif hasattr(extractor, 'extract'):
                    # Verificar se o extrator precisa de metadados
                    if category == 'transform':
                        metadata = AudioData(
                            samples=segment, sample_rate=sr, duration=len(segment) / sr)
                        features = extractor.extract(segment, metadata)
                    elif category in ['spectral', 'cepstral', 'prosodic', 'formant', 'voice_quality', 'perceptual', 'complexity']:
                        # Estes extratores esperam AudioData
                        print(
                            f"DEBUG: Extraindo features de {category} para segmento {segment_idx}")
                        print(f"DEBUG: Tamanho do segmento: {len(segment)}")
                        audio_data = AudioData(
                            samples=segment, sample_rate=sr, duration=len(segment) / sr)
                        result = extractor.extract(audio_data)
                        if result.status == ProcessingStatus.SUCCESS:
                            features = result.data.features
                            print(
                                f"DEBUG: Features {category} extraídas: {
                                    len(features) if features else 0}")
                            if features:
                                print(
                                    f"DEBUG: Primeiras 5 features {category}: {
                                        list(
                                            features.keys())[
                                            :5]}")
                        else:
                            print(
                                f"Erro no extrator {category}: {
                                    result.errors}")
                            features = {}
                    elif category in ['temporal']:
                        features = extractor.extract(segment, sr)
                    else:
                        features = extractor.extract(segment)
                else:
                    print(
                        f"ERRO: Extrator {category} não tem método extract nem extract_features")
                    features = {}

                # Se retornou ProcessingResult, extrair os dados
                if hasattr(features, 'data') and hasattr(features, 'status'):
                    if features.status.name in ['SUCCESS', 'PARTIAL_SUCCESS']:
                        features = features.data
                    else:
                        continue

                # Converter para array 1D se necessário
                if isinstance(features, dict):
                    for name, values in features.items():
                        if isinstance(values, np.ndarray):
                            if values.ndim > 1:
                                values = values.flatten()
                            all_features.extend(values)
                            feature_names.extend(
                                [f"{category}_{name}_{i}" for i in range(len(values))])
                        else:
                            all_features.append(float(values))
                            feature_names.append(f"{category}_{name}")

                    category_features[category] = features
                else:
                    # Features como array direto
                    if isinstance(features, np.ndarray):
                        if features.ndim > 1:
                            features = features.flatten()
                        all_features.extend(features)
                        feature_names.extend(
                            [f"{category}_{i}" for i in range(len(features))])
                        category_features[category] = features
                    else:
                        all_features.append(float(features))
                        feature_names.append(category)
                        category_features[category] = [features]

            except Exception as e:
                print(f"Erro ao extrair features {category}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return SegmentedFeatures(
            segment_index=segment_idx,
            start_time=segment_idx * self.config.segment_duration,
            end_time=(segment_idx + 1) * self.config.segment_duration,
            combined_features=np.array(all_features),
            feature_names=feature_names,
            spectral_features=category_features.get('spectral', {}),
            cepstral_features=category_features.get('cepstral', {}),
            temporal_features=category_features.get('temporal', {}),
            prosodic_features=category_features.get('prosodic', {}),
            formant_features=category_features.get('formant', {}),
            perceptual_features=category_features.get('perceptual', {}),
            speech_features=category_features.get('speech', {}),
            metadata={
                'complexity_features': category_features.get('complexity', {}),
                'transform_features': category_features.get('transform', {}),
                'timefreq_features': category_features.get('timefreq', {}),
                'predictive_features': category_features.get('predictive', {}),
                'voice_quality_features': category_features.get('voice_quality', {})
            }
        )

    def export_features_to_csv(
            self, features: List[SegmentedFeatures], output_path: str, label: str = None) -> ProcessingResult:
        """Exporta features para CSV."""
        try:
            import pandas as pd

            # Converter features para lista de dicionários
            data = []
            for feature in features:
                row = {
                    'segment_index': feature.segment_index,
                    'start_time': feature.start_time,
                    'end_time': feature.end_time
                }

                # Adicionar features de cada categoria
                if feature.spectral_features is not None:
                    for i, val in enumerate(feature.spectral_features):
                        row[f'spectral_{i}'] = val

                if feature.temporal_features is not None:
                    for i, val in enumerate(feature.temporal_features):
                        row[f'temporal_{i}'] = val

                if feature.cepstral_features is not None:
                    for i, val in enumerate(feature.cepstral_features):
                        row[f'cepstral_{i}'] = val

                if feature.prosodic_features is not None:
                    for i, val in enumerate(feature.prosodic_features):
                        row[f'prosodic_{i}'] = val

                if feature.perceptual_features is not None:
                    for i, val in enumerate(feature.perceptual_features):
                        row[f'perceptual_{i}'] = val

                if feature.formant_features is not None:
                    for i, val in enumerate(feature.formant_features):
                        row[f'formant_{i}'] = val

                if feature.speech_features is not None:
                    for i, val in enumerate(feature.speech_features):
                        row[f'speech_{i}'] = val

                # Adicionar metadata features
                if hasattr(feature, 'metadata') and feature.metadata:
                    for key, value in feature.metadata.items():
                        if isinstance(value, (list, tuple)):
                            for i, val in enumerate(value):
                                row[f'{key}_{i}'] = val
                        else:
                            row[key] = value

                data.append(row)

            # Criar DataFrame e salvar
            df = pd.DataFrame(data)
            csv_path = f"{output_path}_features.csv"
            df.to_csv(csv_path, index=False)

            return ProcessingResult.success(
                data={"csv_path": csv_path, "rows": len(df)},
                metadata={"label": label, "features_count": len(features)}
            )

        except Exception as e:
            return ProcessingResult.error(
                errors=[f"Erro ao exportar CSV: {e}"]
            )

    def _export_individual_csv(
            self, features: List[SegmentedFeatures], file_path: str, label: int = None):
        """Exporta features para CSV individual por arquivo de áudio."""
        try:
            from ..exporters.csv_feature_exporter import CSVFeatureExporter

            # Determinar label string
            label_str = "real" if label == 0 else "fake" if label == 1 else "unknown"

            # Obter nome do arquivo sem extensão
            filename = Path(file_path).stem

            # Criar exportador CSV
            csv_exporter = CSVFeatureExporter(self.config.csv_config)

            # Exportar features segmentadas por categoria
            result = csv_exporter.export_segmented_features_by_category(
                features_list=features,
                filename=filename,
                label=label_str
            )

            if result.status == ProcessingStatus.SUCCESS:
                print(
                    f"✅ CSV exportado para {filename}: {len(result.data)} arquivos criados")
            else:
                print(
                    f"⚠️ Erro ao exportar CSV para {filename}: {
                        result.errors}")

        except Exception as e:
            print(f"⚠️ Erro na exportação individual de CSV: {e}")
