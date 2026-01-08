#!/usr/bin/env python3
"""
Inferência para modelo Random Forest de detecção de deepfake.
"""

import joblib
import numpy as np
from typing import Dict, Any, List
import logging

from .base_inference import BaseInference

logger = logging.getLogger(__name__)


class RandomForestInference(BaseInference):
    """
    Classe de inferência para modelo Random Forest.
    """

    def __init__(
            self, model_path: str = "models/random_forest_segmented_features.joblib"):
        """
        Inicializa a inferência Random Forest.

        Args:
            model_path: Caminho para o modelo Random Forest treinado
        """
        super().__init__(model_path)
        self.feature_names = None
        self.scaler = None

    def load_model(self) -> None:
        """
        Carrega o modelo Random Forest treinado.
        """
        try:
            logger.info(
                f"Carregando modelo Random Forest de: {
                    self.model_path}")

            # Carregar modelo
            model_data = joblib.load(self.model_path)

            if isinstance(model_data, dict):
                # Se o modelo foi salvo com metadados
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names')
                self.scaler = model_data.get('scaler')
                logger.info(
                    f"Modelo carregado com metadados: {
                        list(
                            model_data.keys())}")
            else:
                # Se apenas o modelo foi salvo
                self.model = model_data
                logger.info("Modelo carregado sem metadados")

            self.is_loaded = True
            logger.info("Modelo Random Forest carregado com sucesso")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo Random Forest: {e}")
            raise

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Realiza predição com o modelo Random Forest.

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

            # Obter probabilidades
            probabilities = self.model.predict_proba(features)[0]
            confidence = np.max(probabilities)

            # Mapear classes para labels
            classes = getattr(self.model, 'classes_', [0, 1])
            prob_dict = {}
            for i, cls in enumerate(classes):
                label = 'real' if cls == 0 else 'fake'
                prob_dict[label] = float(probabilities[i])

            # Mapear predição para label
            prediction_label = 'real' if prediction == 0 else 'fake'

            # Obter importância das features se disponível
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_.tolist()

            result = {
                'prediction': prediction_label,
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'model_type': 'Random Forest',
                'raw_prediction': int(prediction),
                'feature_importance': feature_importance
            }

            logger.info(
                f"Predição Random Forest: {prediction_label} (confiança: {
                    confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Erro na predição Random Forest: {e}")
            raise

    def get_feature_importance(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna as features mais importantes do modelo.

        Args:
            top_n: Número de features mais importantes a retornar

        Returns:
            Lista com features e suas importâncias
        """
        if not self.is_loaded:
            self.load_model()

        if not hasattr(self.model, 'feature_importances_'):
            return []

        importances = self.model.feature_importances_

        # Criar lista de features com importância
        feature_list = []
        for i, importance in enumerate(importances):
            feature_name = f"feature_{i}"
            if self.feature_names and i < len(self.feature_names):
                feature_name = self.feature_names[i]

            feature_list.append({
                'name': feature_name,
                'importance': float(importance),
                'index': i
            })

        # Ordenar por importância
        feature_list.sort(key=lambda x: x['importance'], reverse=True)

        return feature_list[:top_n]

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
            'model_type': 'Random Forest',
            'model_path': str(self.model_path),
            'has_scaler': self.scaler is not None,
            'has_feature_names': self.feature_names is not None
        }

        if hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()

        if hasattr(self.model, 'classes_'):
            info['classes'] = self.model.classes_.tolist()

        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators

        if hasattr(self.model, 'feature_importances_'):
            info['n_features'] = len(self.model.feature_importances_)
            info['top_features'] = self.get_feature_importance(5)

        return info
