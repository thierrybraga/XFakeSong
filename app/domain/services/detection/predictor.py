import logging
import numpy as np
from typing import Dict, Any, List

from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from .model_loader import ModelInfo
from .utils import pad_or_truncate, prepare_batch_for_model

logger = logging.getLogger(__name__)


class Predictor:
    """Responsável por realizar predições com modelos."""

    def predict(self, model_info: ModelInfo,
                features: np.ndarray) -> ProcessingResult[Dict[str, Any]]:
        """Faz predição com um modelo específico (wrapper para single input)."""
        # Encapsula em uma lista para processamento em lote de tamanho 1
        batch_result = self.predict_batch(model_info, [features])
        
        if batch_result.status != ProcessingStatus.SUCCESS:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=batch_result.errors
            )
            
        # Retorna o primeiro (e único) resultado
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=batch_result.data[0]
        )

    def predict_batch(self, model_info: ModelInfo,
                      features_list: List[np.ndarray]) -> ProcessingResult[List[Dict[str, Any]]]:
        """Faz predição em lote."""
        try:
            if not features_list:
                return ProcessingResult(status=ProcessingStatus.SUCCESS, data=[])

            if model_info.model_type == 'tensorflow':
                return self._predict_tensorflow_batch(model_info, features_list)
            elif model_info.model_type == 'sklearn':
                return self._predict_sklearn_batch(model_info, features_list)
            else:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Tipo de modelo não suportado: {model_info.model_type}"]
                )

        except Exception as e:
            logger.error(f"Erro na predição em lote com modelo {model_info.name}: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na predição: {str(e)}"]
            )

    def _predict_tensorflow_batch(self, model_info: ModelInfo,
                                  features_list: List[np.ndarray]) -> ProcessingResult[List[Dict[str, Any]]]:
        """Predição em lote com modelo TensorFlow."""
        try:
            # Prepara o lote usando o utilitário
            batch_features = prepare_batch_for_model(features_list, model_info.input_shape)
            
            # Predição
            predictions = model_info.model.predict(batch_features, verbose=0)
            
            results = []
            for i in range(len(predictions)):
                pred = predictions[i]
                
                # Interpretar resultado
                if pred.shape[-1] == 1:
                    # Saída binária
                    confidence = float(pred[0] if pred.ndim > 0 else pred)
                    is_deepfake = confidence > 0.5
                else:
                    # Saída categórica
                    confidence = float(np.max(pred))
                    is_deepfake = np.argmax(pred) == 1
                    
                results.append({
                    'is_deepfake': is_deepfake,
                    'confidence': confidence
                })

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=results
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na predição TensorFlow: {str(e)}"]
            )

    def _predict_sklearn_batch(self, model_info: ModelInfo,
                               features_list: List[np.ndarray]) -> ProcessingResult[List[Dict[str, Any]]]:
        """Predição em lote com modelo sklearn."""
        try:
            results = []
            # Sklearn geralmente espera (n_samples, n_features)
            # Precisamos achatar cada feature vector
            
            flattened_features = []
            for f in features_list:
                if model_info.scaler:
                     # Scaler espera 2D array (1, n_features) para transform
                     f_reshaped = f.reshape(1, -1)
                     f_scaled = model_info.scaler.transform(f_reshaped)
                     flattened_features.append(f_scaled.flatten())
                else:
                     flattened_features.append(f.flatten())
            
            X = np.array(flattened_features)
            
            # Predição em lote
            predictions = model_info.model.predict(X)
            
            # Probabilidade
            if hasattr(model_info.model, 'predict_proba'):
                probas = model_info.model.predict_proba(X)
                for i in range(len(probas)):
                    confidence = float(np.max(probas[i]))
                    is_deepfake = np.argmax(probas[i]) == 1
                    results.append({
                        'is_deepfake': is_deepfake,
                        'confidence': confidence
                    })
            else:
                for i in range(len(predictions)):
                    is_deepfake = int(predictions[i]) == 1
                    results.append({
                        'is_deepfake': is_deepfake,
                        'confidence': 1.0
                    })

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=results
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na predição sklearn: {str(e)}"]
            )
