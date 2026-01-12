"""Serviço de Detecção de Deepfake

Este módulo implementa o serviço principal de detecção de deepfake,
integrando todas as arquiteturas disponíveis ao pipeline.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from ...core.interfaces.services import IDetectionService
from ...core.interfaces.audio import AudioData, DeepfakeDetectionResult, FeatureType
from ...core.interfaces.base import ProcessingResult, ProcessingStatus
from ..models.architectures.registry import get_architecture_info
from .feature_extraction_service import AudioFeatureExtractionService
from .detection.model_loader import ModelLoader, ModelInfo
from .detection.feature_preparer import FeaturePreparer
from .detection.predictor import Predictor

logger = logging.getLogger(__name__)


class DetectionService(IDetectionService):
    """Implementação do serviço de detecção de deepfake."""

    def __init__(self, models_dir: Union[str, Path] = "models"):
        self.models_dir = Path(models_dir)
        self.feature_service = AudioFeatureExtractionService()

        # Inicializar sub-serviços
        self.model_loader = ModelLoader(self.models_dir)
        self.feature_preparer = FeaturePreparer(self.feature_service)
        self.predictor = Predictor()

        # Carregar modelos
        self.model_loader.load_available_models()

    @property
    def loaded_models(self) -> Dict[str, ModelInfo]:
        return self.model_loader.loaded_models

    @property
    def default_model(self) -> Optional[str]:
        return self.model_loader.default_model

    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis."""
        return self.model_loader.get_available_models()

    def get_available_architectures(self) -> List[str]:
        """Retorna lista de arquiteturas disponíveis."""
        return self.model_loader.get_available_architectures()

    def find_model(self, architecture: str,
                   variant: str = None) -> Optional[str]:
        """Encontra modelo treinado por arquitetura e variante."""
        return self.model_loader.find_model(architecture, variant)

    def detect_single(self, audio_data: AudioData,
                      model_name: str = None) -> ProcessingResult[DeepfakeDetectionResult]:
        """Detecta deepfake em um único áudio."""
        try:
            # Usar modelo padrão se não especificado
            if model_name is None:
                model_name = self.default_model

            if model_name not in self.loaded_models:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[
                        f"Modelo '{model_name}' não encontrado. Disponíveis: {list(self.loaded_models.keys())}"]
                )

            model_info = self.loaded_models[model_name]
            try:
                arch_info = get_architecture_info(model_info.architecture)
            except Exception:
                arch_info = None

            prepared = self.feature_preparer.prepare_input(
                audio_data, model_info, arch_info)
            if prepared['status'] != 'ok':
                return ProcessingResult(status=ProcessingStatus.ERROR, errors=[
                                        prepared.get('error', 'Falha ao preparar entrada')])

            features = prepared['features']
            extraction_info = prepared['metadata']

            prediction_result = self.predictor.predict(model_info, features)
            if prediction_result.status != ProcessingStatus.SUCCESS:
                return prediction_result
            prediction = prediction_result.data

            result = DeepfakeDetectionResult(
                is_fake=prediction['is_deepfake'],
                confidence=prediction['confidence'],
                probabilities={
                    'fake': prediction['confidence'],
                    'real': 1.0 - prediction['confidence']},
                model_name=model_name,
                features_used=extraction_info.get('config_feature_types', []),
                metadata=extraction_info
            )
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS, data=result)

        except Exception as e:
            logger.error(f"Erro na detecção: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na detecção: {str(e)}"]
            )

    def detect_batch(self, audio_list: List[AudioData],
                     model_name: str = None) -> ProcessingResult[List[DeepfakeDetectionResult]]:
        """Detecta deepfake em lote de forma otimizada."""
        try:
            if not audio_list:
                return ProcessingResult(status=ProcessingStatus.SUCCESS, data=[])

            if model_name is None:
                model_name = self.default_model

            # Usar get_model para carregamento Lazy
            model_info = self.model_loader.get_model(model_name)
            
            if not model_info:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Modelo '{model_name}' não encontrado ou falha ao carregar."]
                )

            try:
                arch_info = get_architecture_info(model_info.architecture)
            except Exception:
                arch_info = None

            # 1. Preparar features para todos os áudios
            prepared_features = []
            metadatas = []
            indices_with_error = []
            errors_map = {}

            for idx, audio_data in enumerate(audio_list):
                try:
                    prepared = self.feature_preparer.prepare_input(audio_data, model_info, arch_info)
                    if prepared['status'] == 'ok':
                        prepared_features.append(prepared['features'])
                        metadatas.append(prepared['metadata'])
                    else:
                        indices_with_error.append(idx)
                        errors_map[idx] = prepared.get('error', 'Falha na preparação')
                except Exception as e:
                    indices_with_error.append(idx)
                    errors_map[idx] = str(e)

            if not prepared_features and indices_with_error:
                # Se todos falharam
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Falha ao preparar features para todos os {len(audio_list)} áudios"]
                )

            # 2. Executar predição em lote
            batch_result = self.predictor.predict_batch(model_info, prepared_features)
            
            if batch_result.status != ProcessingStatus.SUCCESS:
                return ProcessingResult(status=ProcessingStatus.ERROR, errors=batch_result.errors)

            predictions = batch_result.data

            # 3. Reconstruir resultados
            final_results = []
            pred_idx = 0
            
            for original_idx in range(len(audio_list)):
                if original_idx in indices_with_error:
                    # Adicionar resultado de erro
                    final_results.append(DeepfakeDetectionResult(
                        is_fake=False, confidence=0.0,
                        probabilities={'fake': 0.0, 'real': 1.0},
                        model_name=model_name,
                        features_used=[],
                        metadata={'error': errors_map[original_idx]}
                    ))
                else:
                    # Adicionar resultado de sucesso
                    pred = predictions[pred_idx]
                    meta = metadatas[pred_idx]
                    final_results.append(DeepfakeDetectionResult(
                        is_fake=pred['is_deepfake'],
                        confidence=pred['confidence'],
                        probabilities={'fake': pred['confidence'], 'real': 1.0 - pred['confidence']},
                        model_name=model_name,
                        features_used=meta.get('config_feature_types', []),
                        metadata=meta
                    ))
                    pred_idx += 1

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=final_results
            )

        except Exception as e:
            logger.error(f"Erro na detecção em lote: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro geral na detecção em lote: {str(e)}"]
            )

    def detect_from_file(self, file_path: Union[str, Path], model_name: str = None,
                         feature_types: Optional[List[Union[str,
                                                            'FeatureType']]] = None,
                         normalize: bool = True,
                         segmented: bool = False) -> ProcessingResult[DeepfakeDetectionResult]:
        """Detecta deepfake de arquivo."""
        try:
            from ...core.utils.helpers import validate_audio_file
            fp = Path(file_path)
            if not validate_audio_file(fp):
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Arquivo inválido ou formato não suportado: {fp}"]
                )

            audio_data = AudioData.from_file(fp)
            if audio_data.duration <= 0 or audio_data.sample_rate <= 0:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Duração/taxa de amostragem inválida"]
                )
            # Validação de duração mínima (ex.: 0.3s)
            if audio_data.duration < 0.3:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Áudio muito curto: {audio_data.duration:.3f}s"]
                )
            return self._detect_with_config(
                audio_data, model_name, feature_types or [], normalize, segmented)

        except Exception as e:
            logger.error(f"Erro ao carregar arquivo {file_path}: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro ao carregar arquivo: {str(e)}"]
            )

    def _detect_with_config(self, audio_data: AudioData, model_name: Optional[str],
                            feature_types: List[Union[str, 'FeatureType']], normalize: bool = True,
                            segmented: bool = False) -> ProcessingResult[DeepfakeDetectionResult]:
        try:
            import time
            if model_name is None:
                model_name = self.default_model
            
            # Usar get_model para carregamento Lazy
            model_info = self.model_loader.get_model(model_name)
            
            if not model_info:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[
                        f"Modelo '{model_name}' não encontrado ou falha ao carregar. Disponíveis: {self.model_loader.get_available_models()}"]
                )

            try:
                arch_info = get_architecture_info(model_info.architecture)
            except Exception:
                arch_info = None

            # Se for segmentado, usamos o método dedicado
            if segmented:
                return self._predict_segmented(
                    audio_data, model_info, model_name, feature_types, normalize)

            # Fallback para não segmentado ou raw normal
            prepared = self.feature_preparer.prepare_input(
                audio_data, model_info, arch_info)
            if prepared['status'] != 'ok':
                return ProcessingResult(status=ProcessingStatus.ERROR, errors=[
                                        prepared.get('error')])

            features = prepared['features']
            
            # Usar predict single para compatibilidade simples aqui
            prediction_result = self.predictor.predict(
                model_info, features)
            
            if prediction_result.status != ProcessingStatus.SUCCESS:
                return prediction_result

            result = DeepfakeDetectionResult(
                is_fake=prediction_result.data['is_deepfake'],
                confidence=prediction_result.data['confidence'],
                probabilities={
                    'fake': prediction_result.data['confidence'],
                    'real': 1.0 - prediction_result.data['confidence']},
                model_name=model_name,
                features_used=['raw'],
                metadata=prepared['metadata']
            )
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS, data=result)

        except Exception as e:
            logger.error(f"Erro na detecção com config: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR, errors=[str(e)])

    def _predict_segmented(self, audio_data: AudioData, model_info: ModelInfo,
                           model_name: str, feature_types: List, normalize: bool) -> ProcessingResult[DeepfakeDetectionResult]:
        """Realiza predição segmentada (janelamento)."""
        try:
            target_shape = model_info.input_shape
            seq_len = target_shape[0] if len(target_shape) >= 1 else 0
            
            # Se seq_len for 0 ou muito pequeno, não faz sentido segmentar por janelas de input
            if seq_len < 100:
                # Tentar usar duração fixa de 3s se não tiver seq_len definido no input shape
                seq_len = 16000 * 3

            samples = np.asarray(audio_data.samples, dtype=np.float32)
            if samples.ndim > 1:
                samples = samples[:, 0]
            
            if normalize:
                max_abs = np.max(np.abs(samples)) or 1.0
                samples = samples / max_abs

            # Janelamento com sobreposição de 50%
            hop = max(seq_len // 2, 1)
            start = 0
            
            windows = []
            
            # Se o áudio for menor que uma janela, processa como único
            if samples.shape[0] < seq_len:
                # Pad
                window = np.pad(samples, (0, seq_len - samples.shape[0]), mode='constant')
                features_win = window.reshape(seq_len, 1)
                windows.append(features_win)
            else:
                while start < samples.shape[0]:
                    end = start + seq_len
                    window = samples[start:end]
                    
                    # Ignorar janelas muito pequenas no final (< 10%) ou fazer pad? Vamos fazer pad.
                    if window.shape[0] < seq_len:
                         window = np.pad(window, (0, seq_len - window.shape[0]), mode='constant')
                    
                    features_win = window.reshape(seq_len, 1)
                    windows.append(features_win)
                    
                    start += hop

            if not windows:
                 return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Falha ao gerar janelas de predição"]
                )

            # Processar janelas em lote
            batch_result = self.predictor.predict_batch(model_info, windows)
            if batch_result.status != ProcessingStatus.SUCCESS:
                return ProcessingResult(status=ProcessingStatus.ERROR, errors=batch_result.errors)
            
            window_preds = [d['confidence'] for d in batch_result.data]

            # Média das confianças (Soft Voting)
            avg_conf = float(np.mean(window_preds))
            is_fake = avg_conf > 0.5
            
            extraction_info = {
                'feature_type': 'raw_segmented',
                'windows_used': len(window_preds),
                'duration_s': audio_data.duration
            }

            result = DeepfakeDetectionResult(
                is_fake=is_fake,
                confidence=avg_conf,
                probabilities={
                    'fake': avg_conf,
                    'real': 1.0 - avg_conf},
                model_name=model_name,
                features_used=['raw'],
                metadata=extraction_info
            )
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS, data=result)
        
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na segmentação: {str(e)}"]
            )

    def save_analysis_result(self, result: DeepfakeDetectionResult, filename: str) -> bool:
        """Persiste o resultado da análise no banco de dados."""
        try:
            from ...core.db_setup import get_flask_app
            from ...domain.models import AnalysisResult
            
            flask_app = get_flask_app()
            with flask_app.app_context():
                analysis = AnalysisResult(
                    filename=filename,
                    is_fake=result.is_fake,
                    confidence=result.confidence,
                    model_name=result.model_name,
                    duration_seconds=result.metadata.get('duration_s', 0.0),
                    sample_rate=16000, # Padronizado
                    details={
                        "probabilities": result.probabilities,
                        "metadata": result.metadata,
                        "features_used": result.features_used
                    }
                )
                analysis.save()
                logger.info(f"Análise salva com sucesso: ID {analysis.id}")
                return True
        except Exception as e:
            logger.error(f"Falha ao salvar análise: {e}")
            return False
