from typing import Any, Dict, Optional

import numpy as np

from app.core.interfaces.audio import AudioData, FeatureType
from app.core.interfaces.base import ProcessingStatus
from app.domain.features.extractors.mel.mel_spectrogram import MelSpectrogramExtractor
from app.domain.models.architectures.registry import load_hyperparameters_json
from app.domain.services.feature_extraction_service import (
    AudioFeatureExtractionService,
    ExtractionConfig,
)

from .model_loader import ModelInfo


class FeaturePreparer:
    """Prepara features para entrada nos modelos."""

    def __init__(self, feature_service: AudioFeatureExtractionService):
        self.feature_service = feature_service

    @staticmethod
    def _resolve_input_requirements(
        model_info: 'ModelInfo', arch_info: Optional[Any]
    ) -> Dict[str, Any]:
        """Resolve requisitos de input priorizando input_contract do treinamento.

        Prioridade:
        1. input_contract (salvo pelo trainer — fonte de verdade)
        2. arch_info.input_requirements (registry — fallback genérico)
        3. Dict vazio (nenhuma informação disponível)
        """
        # Base: registry info
        base_req = {}
        if arch_info and isinstance(arch_info.input_requirements, dict):
            base_req = dict(arch_info.input_requirements)

        # Override com input_contract se presente
        contract = getattr(model_info, 'input_contract', None)
        if contract and isinstance(contract, dict):
            # O contrato tem prioridade sobre o registry
            for key in ('type', 'format', 'input_shape', 'feature_types',
                        'sample_rate', 'scaler_applied'):
                if key in contract and contract[key] is not None:
                    base_req[key] = contract[key]

        return base_req

    def prepare_input(self, audio_data: AudioData, model_info: ModelInfo,
                      arch_info: Optional[Any]) -> Dict[str, Any]:
        """
        Prepara a entrada de acordo com o tipo de arquitetura.

        Prioridade de decisão:
        1. input_contract salvo pelo trainer (fonte de verdade do treinamento)
        2. arch_info.input_requirements do registry (fallback para modelos sem contrato)

        Retorna dict com keys: status, features, metadata
        """
        try:
            # Resolver requisitos de input: contrato do treinamento > registry
            input_req = self._resolve_input_requirements(model_info, arch_info)

            # Casos de arquiteturas que operam em áudio bruto
            if input_req.get('type') == 'audio':
                target_shape = model_info.input_shape

                # Determinar comprimento da sequência alvo
                seq_len = None
                if target_shape and len(target_shape) >= 1 and target_shape[0]:
                    seq_len = target_shape[0]

                # Fallback: calcular via requisitos de duração
                if not seq_len:
                    max_dur = input_req.get('max_duration')
                    req_sr = input_req.get('sample_rate', 16000)
                    if max_dur:
                        seq_len = int(max_dur * req_sr)

                # Se ainda não definido, usar comprimento atual (dinâmico)
                if not seq_len:
                    seq_len = len(audio_data.samples)

                feat_dim = target_shape[1] if target_shape and len(
                    target_shape
                ) >= 2 else 1
                if feat_dim and int(feat_dim) > 1:
                    extractor = MelSpectrogramExtractor(
                        sr=input_req.get("sample_rate", audio_data.sample_rate or 16000),
                        n_mels=int(feat_dim),
                    )
                    features_result = extractor.extract(audio_data)
                    if features_result.status != ProcessingStatus.SUCCESS:
                        return {
                            "status": "error",
                            "error": (
                                features_result.errors[0]
                                if features_result.errors
                                else "Erro ao extrair mel spectrogram"
                            ),
                        }

                    mel_feats = features_result.data.features.get("mel_spectrogram")
                    if mel_feats is None:
                        return {
                            "status": "error",
                            "error": "Mel spectrogram não encontrado no resultado",
                        }

                    features = np.asarray(mel_feats, dtype=np.float32)
                    metadata = {
                        "feature_type": "mel_spectrogram",
                        "feature_names": ["mel_spectrogram"],
                        "feature_shapes": {"mel_spectrogram": features.shape},
                        "feature_count_total": int(features.size),
                        "features_shape": features.shape,
                        "sample_rate": audio_data.sample_rate,
                        "duration_s": audio_data.duration,
                        "channels": audio_data.channels,
                        "config_feature_types": ["mel_spectrogram"],
                    }
                    try:
                        from pathlib import Path

                        results_dir = (
                            Path(__file__).resolve().parent.parent.parent.parent
                            / "results"
                        )
                        metadata["recommended_hyperparameters"] = load_hyperparameters_json(
                            model_info.architecture, str(results_dir)
                        )
                    except Exception:
                        metadata["recommended_hyperparameters"] = {}

                    return {"status": "ok", "features": features, "metadata": metadata}
                samples = np.asarray(audio_data.samples, dtype=np.float32)

                if samples.ndim > 1:
                    samples = samples[:, 0]

                # Normalização simples se requerido
                if input_req.get('preprocessing') == 'normalize':
                    max_abs = np.max(np.abs(samples)) or 1.0
                    samples = samples / max_abs

                # Ajuste de tamanho
                if samples.shape[0] > seq_len:
                    mid = samples.shape[0] // 2
                    start = max(0, mid - seq_len // 2)
                    samples = samples[start:start + seq_len]
                elif samples.shape[0] < seq_len:
                    pad = seq_len - samples.shape[0]
                    samples = np.pad(samples, (0, pad), mode='constant')

                features = samples.reshape(seq_len, 1)
                metadata = {
                    'feature_type': 'raw',
                    'feature_names': ['waveform'],
                    'feature_shapes': {'waveform': (seq_len,)},
                    'feature_count_total': int(seq_len),
                    'features_shape': (seq_len, feat_dim),
                    'sample_rate': audio_data.sample_rate,
                    'duration_s': audio_data.duration,
                    'channels': audio_data.channels,
                    'config_feature_types': ['raw']
                }
                try:
                    from pathlib import Path
                    results_dir = Path(__file__).resolve(
                    ).parent.parent.parent.parent / "results"
                    metadata['recommended_hyperparameters'] = (
                        load_hyperparameters_json(
                            model_info.architecture, str(results_dir)
                        )
                    )
                except Exception:
                    metadata['recommended_hyperparameters'] = {}
                return {'status': 'ok', 'features': features,
                        'metadata': metadata}

            # Verificar requisitos da arquitetura
            req_format = input_req.get('format')

            # Verificar se o modelo tem requisitos específicos de features
            # Prioridade: input_contract > atributo do modelo > fallback
            feature_types_used = None
            if model_info.input_contract and model_info.input_contract.get('feature_types'):
                feature_types_used = model_info.input_contract['feature_types']
            if not feature_types_used:
                feature_types_used = getattr(
                    model_info.model, 'feature_types_used', None)
            aggregate_method = getattr(
                model_info.model, 'aggregate_method', 'mean')

            # Se a arquitetura exige tabular mas o modelo não define features
            # (ex: carregado sem metadados), usar conjunto padrão robusto
            if req_format == 'tabular' and not feature_types_used:
                feature_types_used = [
                    'spectral', 'cepstral', 'temporal', 'prosodic'
                ]

            if feature_types_used:
                # Converter strings para FeatureType enums
                feature_enums = []
                for ft in feature_types_used:
                    try:
                        feature_enums.append(FeatureType(ft))
                    except ValueError:
                        pass

                if feature_enums:
                    config = ExtractionConfig(
                        feature_types=feature_enums,
                        normalize=True,
                        aggregate_method=aggregate_method
                    )

                    try:
                        # Usar extração segmentada
                        audio_features = self.feature_service.extract_segmented_features(
                            audio_data, config)

                        # Obter features combinadas
                        if hasattr(audio_features, 'features') and isinstance(
                            audio_features.features, dict
                        ):
                            features = audio_features.features.get(
                                'combined_features', np.array([])
                            )
                            feature_names = audio_features.features.get(
                                'feature_names', []
                            )
                        else:
                            features = np.array([])
                            feature_names = []

                        # Metadados
                        metadata = {
                            'feature_type': 'segmented_aggregated',
                            'feature_names': feature_names,
                            'feature_shapes': {
                                'combined_features': features.shape
                            },
                            'feature_count_total': features.size,
                            'features_shape': features.shape,
                            'sample_rate': audio_data.sample_rate,
                            'duration_s': audio_data.duration,
                            'channels': audio_data.channels,
                            'config_feature_types': [
                                ft.value for ft in feature_enums
                            ],
                            'aggregate_method': aggregate_method
                        }

                        try:
                            from pathlib import Path
                            results_dir = Path(__file__).resolve(
                            ).parent.parent.parent.parent / "results"
                            metadata['recommended_hyperparameters'] = (
                                load_hyperparameters_json(
                                    model_info.architecture, str(results_dir)
                                )
                            )
                        except Exception:
                            metadata['recommended_hyperparameters'] = {}

                        return {
                            'status': 'ok',
                            'features': features.reshape(1, -1),
                            'metadata': metadata
                        }

                    except Exception as e:
                        return {
                            'status': 'error',
                            'error': f"Erro na extração segmentada: {str(e)}"
                        }

            # Demais arquiteturas: extrair features conforme necessidade
            selected_feature_type = FeatureType.SPECTRAL
            features_result = None

            if input_req:
                req_type = input_req.get('type')
                req_format = input_req.get('format')
                feature_dim = input_req.get('feature_dim')

                if req_type == 'features' and req_format == 'spectrogram':
                    selected_feature_type = FeatureType.MEL_SPECTROGRAM
                    n_mels = feature_dim if feature_dim else 128
                    # Usar extrator específico para garantir dimensão correta
                    extractor = MelSpectrogramExtractor(n_mels=n_mels)
                    features_result = extractor.extract(audio_data)

            if features_result is None:
                config = ExtractionConfig(
                    feature_types=[selected_feature_type], normalize=True)
                features_result = self.feature_service.extract_features(
                    audio_data, config)

            if features_result.status != ProcessingStatus.SUCCESS:
                return {
                    'status': 'error',
                    'error': features_result.errors[0] if features_result.errors else 'Erro desconhecido'  # noqa: E501
                }

            extraction_result = features_result.data
            audio_features = extraction_result
            feature_shapes_map = {}
            feature_names_list = []
            # Preparar estrutura apropriada
            if selected_feature_type == FeatureType.MEL_SPECTROGRAM:
                if hasattr(audio_features, 'features') and isinstance(
                    audio_features.features, dict
                ):
                    if 'mel_spectrogram' in audio_features.features:
                        features = audio_features.features['mel_spectrogram']
                        feature_shapes_map['mel_spectrogram'] = features.shape
                        feature_names_list = ['mel_spectrogram']
                        # Ajustar canais conforme input_shape
                        target_shape = model_info.input_shape
                        if len(target_shape) == 3 and features.ndim == 2:
                            features = np.expand_dims(features, axis=-1)
            else:
                feature_arrays = []
                if hasattr(audio_features, 'features') and isinstance(
                        audio_features.features, dict):
                    for feature_name, feature_array in audio_features.features.items():
                        if isinstance(feature_array, np.ndarray):
                            feature_arrays.append(feature_array.flatten())
                            feature_shapes_map[feature_name] = tuple(
                                feature_array.shape)
                            feature_names_list.append(feature_name)
                features = np.concatenate(
                    feature_arrays) if feature_arrays else np.array([])

            metadata = {
                'feature_type': audio_features.feature_type.value if hasattr(audio_features, 'feature_type') else None,
                'feature_names': feature_names_list,
                'feature_shapes': feature_shapes_map,
                'feature_count_total': int(features.size) if isinstance(features, np.ndarray) and features.ndim == 1 else (features.shape[0] * features.shape[1] if isinstance(features, np.ndarray) and features.ndim >= 2 else 0),
                'features_shape': extraction_result.feature_shape if hasattr(extraction_result, 'feature_shape') else (features.shape if hasattr(features, 'shape') else None),
                'sample_rate': audio_data.sample_rate,
                'duration_s': audio_data.duration,
                'channels': audio_data.channels,
                'config_feature_types': [selected_feature_type.value]
            }
            try:
                from pathlib import Path
                results_dir = Path(__file__).resolve(
                ).parent.parent.parent.parent / "results"
                metadata['recommended_hyperparameters'] = (
                    load_hyperparameters_json(
                        model_info.architecture, str(results_dir)
                    )
                )
            except Exception:
                metadata['recommended_hyperparameters'] = {}
            return {'status': 'ok', 'features': features, 'metadata': metadata}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
