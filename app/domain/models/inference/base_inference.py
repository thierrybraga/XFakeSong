#!/usr/bin/env python3
"""
Interface base para inferência de modelos de detecção de deepfake.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseInference(ABC):
    """
    Interface base para inferência de modelos de detecção de deepfake.
    """

    def __init__(self, model_path: str):
        """
        Inicializa o sistema de inferência.

        Args:
            model_path: Caminho para o modelo treinado
        """
        self.model_path = Path(model_path)
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """
        Carrega o modelo treinado.
        """
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Realiza predição com base nas features extraídas.

        Args:
            features: Array numpy com as features extraídas do áudio

        Returns:
            Dict contendo:
                - prediction: 'real' ou 'fake'
                - confidence: Confiança da predição (0-1)
                - probabilities: Probabilidades para cada classe
        """
        pass

    def predict_audio(self, audio_path: str,
                      feature_types: List[str] = None) -> Dict[str, Any]:
        """
        Realiza predição completa a partir de um arquivo de áudio.

        Args:
            audio_path: Caminho para o arquivo de áudio
            feature_types: Lista de tipos de features a extrair (None = todos)

        Returns:
            Dict contendo resultado da predição
        """
        if not self.is_loaded:
            self.load_model()

        # Carregar áudio diretamente
        import librosa
        from app.core.interfaces.audio import AudioData

        samples, sr = librosa.load(audio_path, sr=16000)
        audio_data = AudioData(
            samples=samples,
            sample_rate=sr,
            duration=len(samples) / sr,
            channels=1
        )

        # Extrair features do áudio
        features = self._extract_features_for_inference(audio_data)

        # Realizar predição
        result = self.predict(features)
        result['audio_path'] = audio_path

        return result

    def _extract_features_for_inference(self, audio_data) -> np.ndarray:
        """
        Extrai features do áudio usando exatamente o mesmo método do treinamento.
        Salva temporariamente o áudio e usa o carregador segmentado.

        Args:
            audio_path: Caminho para o arquivo de áudio
            feature_types: Lista de tipos de features a extrair (ignorado, usa todos)

        Returns:
            Array numpy com features extraídas (7180 dimensões)
        """
        import tempfile
        import soundfile as sf
        from pathlib import Path
        from app.domain.features.segmented_feature_loader import create_feature_loader

        logger.info(f"=== INICIANDO EXTRAÇÃO DE FEATURES PARA INFERÊNCIA ===")
        logger.info(
            f"Áudio: {len(audio_data.samples)} amostras, {audio_data.sample_rate}Hz")

        # Criar arquivo temporário
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Salvar áudio temporariamente
            sf.write(temp_path, audio_data.samples, audio_data.sample_rate)

            # Criar estrutura temporária para simular pasta segmented
            temp_dir = Path(tempfile.mkdtemp())
            segmented_dir = temp_dir / "segmented" / "test"

            # Extrair features usando o extrator segmentado
            from app.domain.features.extractors.segmented_feature_extractor import (
                SegmentedFeatureExtractor, SegmentedExtractionConfig
            )

            logger.info("Criando configuração de extração...")
            # Configuração idêntica ao treinamento
            config = SegmentedExtractionConfig(
                segment_duration=2.0,  # Usar mesmo tamanho do treinamento
                overlap_ratio=0.5,     # Usar mesmo overlap do treinamento
                target_sample_rate=16000,  # Padronizar sample rate
                extract_spectral=True,
                extract_cepstral=True,
                extract_temporal=True,
                extract_prosodic=True,
                extract_formant=True,
                extract_voice_quality=True,
                extract_perceptual=True,
                extract_complexity=True,
                extract_transform=True,
                extract_timefreq=False,
                extract_predictive=False,
                extract_speech=True,
                export_csv=True
            )

            logger.info("Iniciando extração com SegmentedFeatureExtractor...")
            # Extrair features
            try:
                extractor = SegmentedFeatureExtractor(config)
                logger.info(f"Extraindo features do arquivo: {temp_path}")
                result = extractor.extract_from_file(temp_path)
                logger.info("Extração concluída!")
            except Exception as extraction_error:
                logger.error(f"Erro durante a extração: {extraction_error}")
                logger.error(f"Tipo do erro: {type(extraction_error)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise extraction_error

            logger.info(
                f"Resultado da extração: status={
                    result.status.name if result.status else 'None'}")
            logger.info(
                f"Dados disponíveis: {
                    result.data is not None if hasattr(
                        result,
                        'data') else 'No data attr'}")

            # Debug detalhado do resultado
            if hasattr(result, 'status'):
                logger.info(f"Status detalhado: {result.status}")
                logger.info(
                    f"Status é SUCCESS? {
                        result.status.name == 'SUCCESS' if result.status else False}")
            if hasattr(result, 'data'):
                logger.info(f"Tipo de dados: {type(result.data)}")
                if result.data:
                    logger.info(f"Quantidade de dados: {len(result.data)}")
                    logger.info(
                        f"Primeiro item: {
                            type(
                                result.data[0]) if len(
                                result.data) > 0 else 'Lista vazia'}")
                else:
                    logger.warning("Dados estão vazios ou None")

            # Verificar se há erros no resultado
            if hasattr(result, 'errors') and result.errors:
                logger.error(f"Erros na extração: {result.errors}")

            if result.status.name == 'SUCCESS' and result.data:
                segments = result.data
                logger.info(f"Segmentos extraídos: {len(segments)}")

                if segments:
                    # Criar estrutura de pastas esperada pelo carregador:
                    # segmented/test/
                    test_class_dir = temp_dir / "segmented" / "test"
                    test_class_dir.mkdir(parents=True, exist_ok=True)

                    # Salvar features em CSV para usar o carregador
                    self._save_features_as_csv(segments, test_class_dir)

                    logger.info(f"Features salvas em: {test_class_dir}")

                    # Usar o carregador de features segmentadas (mesmo do
                    # treinamento)
                    feature_loader = create_feature_loader(
                        segmented_path=str(temp_dir / "segmented"),
                        feature_types=None,  # Todas as features
                        normalize=False,  # Não normalizar aqui
                        aggregate_method='mean'
                    )

                    # Carregar features usando o mesmo método do treinamento
                    X, y, feature_names = feature_loader.load_multiple_samples_per_class(
                        classes=['test'],
                        max_samples_per_class=1
                    )

                    # Limpar arquivos temporários após carregamento
                    try:
                        import os
                        import shutil
                        os.unlink(temp_path)
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except BaseException:
                        pass

                    if len(X) > 0:
                        # Primeira (e única) amostra
                        aggregated_features = X[0]
                        logger.info(
                            f"Features carregadas via loader: {
                                aggregated_features.shape}")
                        return aggregated_features.reshape(1, -1)
                    else:
                        logger.warning("Nenhuma feature carregada pelo loader")
                        return np.array([[]])
                else:
                    logger.warning("Nenhum segmento extraído")
                    # Limpar arquivos temporários
                    try:
                        import os
                        import shutil
                        os.unlink(temp_path)
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except BaseException:
                        pass
                    return np.array([[]])
            else:
                logger.error(
                    f"Erro na extração: {
                        result.errors if hasattr(
                            result,
                            'errors') else 'Erro desconhecido'}")
                # Limpar arquivos temporários
                try:
                    import os
                    import shutil
                    os.unlink(temp_path)
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except BaseException:
                    pass
                return np.array([[]])

        except Exception as e:
            logger.error(f"Erro na extração de features: {e}")
            # Limpar arquivos temporários em caso de erro
            try:
                import os
                import shutil
                if 'temp_path' in locals():
                    os.unlink(temp_path)
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except BaseException:
                pass
            return np.array([])

    def _save_features_as_csv(self, segments, output_dir):
        """Salva features dos segmentos em CSV para uso pelo carregador"""
        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Salvando {len(segments)} segmentos em {output_dir}")

        saved_count = 0
        for i, segment in enumerate(segments):
            logger.debug(f"Processando segmento {i}: {type(segment)}")
            if hasattr(
                    segment, 'combined_features') and segment.combined_features is not None:
                logger.debug(
                    f"Segmento {i} tem features: {len(segment.combined_features)} features")
                logger.debug(
                    f"Feature names: {len(segment.feature_names)} nomes")

                # Criar dicionário com features nomeadas
                features_dict = {}
                for j, feature_name in enumerate(segment.feature_names):
                    if j < len(segment.combined_features):
                        features_dict[feature_name] = segment.combined_features[j]

                # Converter features para DataFrame
                features_df = pd.DataFrame([features_dict])

                # Salvar como CSV no formato esperado pelo carregador:
                # {class}_sample_{id}_{feature_type}.csv
                csv_path = output_dir / f"test_sample_{i:03d}_all.csv"
                features_df.to_csv(csv_path, index=False)
                saved_count += 1
                logger.info(
                    f"Features do segmento {i} salvas em {csv_path} ({
                        len(features_dict)} features)")
            else:
                logger.warning(f"Segmento {i} não tem features válidas")
                if hasattr(segment, 'combined_features'):
                    logger.warning(
                        f"  combined_features: {
                            segment.combined_features}")
                if hasattr(segment, 'feature_names'):
                    logger.warning(
                        f"  feature_names: {len(segment.feature_names) if segment.feature_names else 0} nomes")

        logger.info(f"Total de arquivos CSV salvos: {saved_count}")

    def batch_predict(
            self, audio_paths: List[str], feature_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Realiza predição em lote para múltiplos arquivos de áudio.

        Args:
            audio_paths: Lista de caminhos para arquivos de áudio
            feature_types: Lista de tipos de features a extrair

        Returns:
            Lista com resultados das predições
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict_audio(audio_path, feature_types)
                results.append(result)
            except Exception as e:
                results.append({
                    'audio_path': audio_path,
                    'error': str(e),
                    'prediction': None,
                    'confidence': 0.0
                })

        return results
