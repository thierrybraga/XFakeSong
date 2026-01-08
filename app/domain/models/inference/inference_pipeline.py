#!/usr/bin/env python3
"""
Pipeline de inferência para detecção de deepfake.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np

from .svm_inference import SVMInference
from .random_forest_inference import RandomForestInference

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Pipeline completo de inferência para detecção de deepfake.
    """

    def __init__(self,
                 svm_model_path: str = "models/svm_segmented_features.joblib",
                 rf_model_path: str = "models/random_forest_segmented_features.joblib"):
        """
        Inicializa o pipeline de inferência.

        Args:
            svm_model_path: Caminho para o modelo SVM
            rf_model_path: Caminho para o modelo Random Forest
        """
        self.svm_model_path = svm_model_path
        self.rf_model_path = rf_model_path

        self.svm_inference = None
        self.rf_inference = None

        self.models_loaded = False

    def load_models(self, models: List[str] = None) -> None:
        """
        Carrega os modelos especificados.

        Args:
            models: Lista de modelos a carregar ['svm', 'rf']. Se None, carrega todos.
        """
        if models is None:
            models = ['svm', 'rf']

        logger.info(f"Carregando modelos: {models}")

        if 'svm' in models:
            if os.path.exists(self.svm_model_path):
                self.svm_inference = SVMInference(self.svm_model_path)
                self.svm_inference.load_model()
                logger.info("Modelo SVM carregado")
            else:
                logger.warning(
                    f"Modelo SVM não encontrado: {
                        self.svm_model_path}")

        if 'rf' in models:
            if os.path.exists(self.rf_model_path):
                self.rf_inference = RandomForestInference(self.rf_model_path)
                self.rf_inference.load_model()
                logger.info("Modelo Random Forest carregado")
            else:
                logger.warning(
                    f"Modelo Random Forest não encontrado: {
                        self.rf_model_path}")

        self.models_loaded = True

    def predict_single(self,
                       audio_path: str,
                       model_type: str = 'both',
                       feature_types: List[str] = None) -> Dict[str, Any]:
        """
        Realiza predição em um único arquivo de áudio.

        Args:
            audio_path: Caminho para o arquivo de áudio
            model_type: Tipo de modelo ('svm', 'rf', 'both')
            feature_types: Lista de tipos de features a extrair

        Returns:
            Dict com resultados da predição
        """
        if not self.models_loaded:
            self.load_models()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Arquivo de áudio não encontrado: {audio_path}")

        logger.info(f"Realizando predição para: {audio_path}")

        results = {
            'audio_path': audio_path,
            'predictions': {},
            'consensus': None,
            'confidence_avg': 0.0
        }

        # Predição com SVM
        if model_type in ['svm', 'both'] and self.svm_inference:
            try:
                svm_result = self.svm_inference.predict_audio(
                    audio_path, feature_types)
                results['predictions']['svm'] = svm_result
                logger.info(
                    f"SVM: {
                        svm_result['prediction']} ({
                        svm_result['confidence']:.3f})")
            except Exception as e:
                logger.error(f"Erro na predição SVM: {e}")
                results['predictions']['svm'] = {'error': str(e)}

        # Predição com Random Forest
        if model_type in ['rf', 'both'] and self.rf_inference:
            try:
                rf_result = self.rf_inference.predict_audio(
                    audio_path, feature_types)
                results['predictions']['rf'] = rf_result
                logger.info(
                    f"Random Forest: {
                        rf_result['prediction']} ({
                        rf_result['confidence']:.3f})")
            except Exception as e:
                logger.error(f"Erro na predição Random Forest: {e}")
                results['predictions']['rf'] = {'error': str(e)}

        # Calcular consenso se ambos os modelos foram usados
        if len(results['predictions']) > 1:
            results['consensus'] = self._calculate_consensus(
                results['predictions'])

        return results

    def predict_batch(self,
                      audio_paths: List[str],
                      model_type: str = 'both',
                      feature_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Realiza predição em lote para múltiplos arquivos.

        Args:
            audio_paths: Lista de caminhos para arquivos de áudio
            model_type: Tipo de modelo ('svm', 'rf', 'both')
            feature_types: Lista de tipos de features a extrair

        Returns:
            Lista com resultados das predições
        """
        results = []

        for i, audio_path in enumerate(audio_paths):
            logger.info(
                f"Processando {i + 1}/{len(audio_paths)}: {audio_path}")

            try:
                result = self.predict_single(
                    audio_path, model_type, feature_types)
                results.append(result)
            except Exception as e:
                logger.error(f"Erro ao processar {audio_path}: {e}")
                results.append({
                    'audio_path': audio_path,
                    'error': str(e),
                    'predictions': {},
                    'consensus': None
                })

        return results

    def predict_directory(self,
                          directory_path: str,
                          model_type: str = 'both',
                          feature_types: List[str] = None,
                          audio_extensions: List[str] = None) -> Dict[str, Any]:
        """
        Realiza predição em todos os arquivos de áudio de um diretório.

        Args:
            directory_path: Caminho para o diretório
            model_type: Tipo de modelo ('svm', 'rf', 'both')
            feature_types: Lista de tipos de features a extrair
            audio_extensions: Extensões de áudio aceitas

        Returns:
            Dict com resultados e estatísticas
        """
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']

        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(
                f"Diretório não encontrado: {directory_path}")

        # Encontrar arquivos de áudio
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(directory.glob(f"*{ext}"))
            audio_files.extend(directory.glob(f"*{ext.upper()}"))

        if not audio_files:
            logger.warning(
                f"Nenhum arquivo de áudio encontrado em: {directory_path}")
            return {
                'directory': directory_path,
                'total_files': 0,
                'results': [],
                'statistics': {}
            }

        logger.info(f"Encontrados {len(audio_files)} arquivos de áudio")

        # Realizar predições
        audio_paths = [str(f) for f in audio_files]
        results = self.predict_batch(audio_paths, model_type, feature_types)

        # Calcular estatísticas
        statistics = self._calculate_statistics(results)

        return {
            'directory': directory_path,
            'total_files': len(audio_files),
            'results': results,
            'statistics': statistics
        }

    def _calculate_consensus(
            self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calcula consenso entre múltiplos modelos.

        Args:
            predictions: Dict com predições de diferentes modelos

        Returns:
            Dict com consenso calculado
        """
        valid_predictions = {}

        # Filtrar predições válidas
        for model, pred in predictions.items():
            if 'error' not in pred and 'prediction' in pred:
                valid_predictions[model] = pred

        if not valid_predictions:
            return {'prediction': 'unknown',
                    'confidence': 0.0, 'agreement': False}

        # Coletar predições e confianças
        pred_labels = [p['prediction'] for p in valid_predictions.values()]
        confidences = [p['confidence'] for p in valid_predictions.values()]

        # Verificar acordo
        agreement = len(set(pred_labels)) == 1

        if agreement:
            # Se todos concordam
            consensus_pred = pred_labels[0]
            consensus_conf = np.mean(confidences)
        else:
            # Se há discordância, usar modelo com maior confiança
            max_conf_idx = np.argmax(confidences)
            consensus_pred = pred_labels[max_conf_idx]
            # Reduzir confiança devido à discordância
            consensus_conf = confidences[max_conf_idx] * 0.8

        return {
            'prediction': consensus_pred,
            'confidence': float(consensus_conf),
            'agreement': agreement,
            'models_used': list(valid_predictions.keys()),
            'individual_confidences': {model: pred['confidence']
                                       for model, pred in valid_predictions.items()}
        }

    def _calculate_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Calcula estatísticas dos resultados.

        Args:
            results: Lista de resultados de predição

        Returns:
            Dict com estatísticas
        """
        stats = {
            'total_processed': len(results),
            'successful': 0,
            'errors': 0,
            'predictions': {'real': 0, 'fake': 0, 'unknown': 0},
            'average_confidence': 0.0,
            'model_agreement': 0
        }

        confidences = []
        agreements = 0

        for result in results:
            if 'error' in result:
                stats['errors'] += 1
                continue

            stats['successful'] += 1

            # Usar consenso se disponível, senão primeira predição válida
            if result.get('consensus'):
                pred = result['consensus']['prediction']
                conf = result['consensus']['confidence']
                if result['consensus']['agreement']:
                    agreements += 1
            else:
                # Pegar primeira predição válida
                for model_pred in result['predictions'].values():
                    if 'error' not in model_pred:
                        pred = model_pred['prediction']
                        conf = model_pred['confidence']
                        break
                else:
                    pred = 'unknown'
                    conf = 0.0

            stats['predictions'][pred] += 1
            confidences.append(conf)

        if confidences:
            stats['average_confidence'] = float(np.mean(confidences))

        if stats['successful'] > 0:
            stats['model_agreement'] = agreements / stats['successful']

        return stats

    def get_models_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre os modelos carregados.

        Returns:
            Dict com informações dos modelos
        """
        info = {
            'models_loaded': self.models_loaded,
            'svm': None,
            'rf': None
        }

        if self.svm_inference:
            info['svm'] = self.svm_inference.get_model_info()

        if self.rf_inference:
            info['rf'] = self.rf_inference.get_model_info()

        return info
