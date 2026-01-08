#!/usr/bin/env python3
"""
Inferência para modelo SVM de detecção de deepfake.
"""

import joblib
import numpy as np
from typing import Dict, Any, List
import logging

from .base_inference import BaseInference

logger = logging.getLogger(__name__)


class SVMInference(BaseInference):
    """
    Classe de inferência para modelo SVM.
    """

    def __init__(
            self, model_path: str = "models/svm_segmented_features.joblib"):
        """
        Inicializa a inferência SVM.

        Args:
            model_path: Caminho para o modelo SVM treinado
        """
        super().__init__(model_path)
        self.scaler = None

    def load_model(self) -> None:
        """
        Carrega o modelo SVM treinado.
        """
        try:
            logger.info(f"Carregando modelo SVM de: {self.model_path}")

            # Carregar modelo
            model_data = joblib.load(self.model_path)

            if isinstance(model_data, dict):
                # Se o modelo foi salvo com metadados
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                logger.info(
                    f"Modelo carregado com metadados: {
                        list(
                            model_data.keys())}")
            else:
                # Se apenas o modelo foi salvo
                self.model = model_data
                logger.info("Modelo carregado sem scaler")

            self.is_loaded = True
            logger.info("Modelo SVM carregado com sucesso")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo SVM: {e}")
            raise

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Realiza predição com o modelo SVM.

        Args:
            features: Array numpy com as features extraídas

        Returns:
            Dict com resultado da predição
        """
        if not self.is_loaded:
            self.load_model()

        try:
            # Garantir que features é 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Aplicar normalização se disponível
            if self.scaler is not None:
                features = self.scaler.transform(features)

            # Realizar predição
            prediction = self.model.predict(features)[0]

            # Obter probabilidades se disponível
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = np.max(probabilities)

                # Mapear classes para labels
                classes = getattr(self.model, 'classes_', [0, 1])
                prob_dict = {}
                for i, cls in enumerate(classes):
                    label = 'real' if cls == 0 else 'fake'
                    prob_dict[label] = float(probabilities[i])
            else:
                # Se não há probabilidades, usar confiança baseada na distância
                decision_score = self.model.decision_function(features)[0]
                confidence = float(1.0 / (1.0 + np.exp(-abs(decision_score))))

                prob_dict = {
                    'real': confidence if prediction == 0 else 1 - confidence,
                    'fake': confidence if prediction == 1 else 1 - confidence
                }

            # Mapear predição para label
            prediction_label = 'real' if prediction == 0 else 'fake'

            result = {
                'prediction': prediction_label,
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'model_type': 'SVM',
                'raw_prediction': int(prediction)
            }

            logger.info(
                f"Predição SVM: {prediction_label} (confiança: {
                    confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Erro na predição SVM: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo carregado.

        Returns:
            Dict com informações do modelo
        """
        if not self.is_loaded:
            return {'loaded': False}

        info = {
            'loaded': True,
            'model_type': 'SVM',
            'model_path': str(self.model_path),
            'has_scaler': self.scaler is not None
        }

        if hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()

        if hasattr(self.model, 'classes_'):
            info['classes'] = self.model.classes_.tolist()

        if hasattr(self.model, 'support_vectors_'):
            info['n_support_vectors'] = len(self.model.support_vectors_)

        return info
