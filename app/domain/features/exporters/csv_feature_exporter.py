#!/usr/bin/env python3
"""
Exportador de Features CSV
=========================

Módulo responsável por exportar features segmentadas para formato CSV,
onde cada linha representa um segmento de 1 segundo e cada coluna uma característica.
"""

import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

from app.domain.features.models.segmented_features import SegmentedFeatures
from app.core.interfaces.base import ProcessingResult, ProcessingStatus

logger = logging.getLogger(__name__)


@dataclass
class CSVExportConfig:
    """Configuração para exportação CSV"""
    output_base_dir: str = "datasets/features"
    segmented_dir: str = "segmented"
    original_dir: str = "original"
    include_metadata: bool = True
    decimal_places: int = 6
    delimiter: str = ","
    include_headers: bool = True


class CSVFeatureExporter:
    """Exportador de features para formato CSV"""

    def __init__(self, config: CSVExportConfig = None):
        self.config = config or CSVExportConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def export_segmented_features(
        self,
        features_list: List[SegmentedFeatures],
        filename: str,
        label: str = "real"
    ) -> ProcessingResult[str]:
        """
        Exporta features segmentadas para CSV.

        Args:
            features_list: Lista de features segmentadas
            filename: Nome do arquivo (sem extensão)
            label: Label do arquivo (real/fake)

        Returns:
            ProcessingResult com caminho do arquivo criado
        """
        try:
            if not features_list:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    data=None,
                    errors=["Lista de features vazia"]
                )

            # Criar diretório de saída
            output_dir = Path(self.config.output_base_dir) / \
                self.config.segmented_dir / label
            output_dir.mkdir(parents=True, exist_ok=True)

            # Caminho do arquivo CSV
            csv_path = output_dir / f"{filename}.csv"

            # Preparar dados para CSV
            csv_data = self._prepare_csv_data(features_list)

            # Escrever CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=self.config.delimiter)

                # Escrever cabeçalho se configurado
                if self.config.include_headers:
                    headers = self._generate_headers(features_list[0])
                    writer.writerow(headers)

                # Escrever dados
                for row in csv_data:
                    writer.writerow(row)

            self.logger.info(
                f"Features segmentadas exportadas para: {csv_path}")

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=str(csv_path),
                metadata={
                    'segments_count': len(features_list),
                    'features_dimension': len(features_list[0].combined_features),
                    'file_size': csv_path.stat().st_size
                }
            )

        except Exception as e:
            self.logger.error(f"Erro ao exportar features segmentadas: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                data=None,
                errors=[str(e)]
            )

    def export_original_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        filename: str,
        label: str = "real"
    ) -> ProcessingResult[str]:
        """
        Exporta features originais (não segmentadas) para CSV.

        Args:
            features: Array de features
            feature_names: Nomes das features
            filename: Nome do arquivo
            label: Label do arquivo (real/fake)

        Returns:
            ProcessingResult com caminho do arquivo criado
        """
        try:
            # Criar diretório de saída
            output_dir = Path(self.config.output_base_dir) / \
                self.config.original_dir / label
            output_dir.mkdir(parents=True, exist_ok=True)

            # Caminho do arquivo CSV
            csv_path = output_dir / f"{filename}.csv"

            # Escrever CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=self.config.delimiter)

                # Escrever cabeçalho
                if self.config.include_headers:
                    writer.writerow(feature_names)

                # Escrever features (uma linha)
                formatted_features = [
                    round(
                        float(f),
                        self.config.decimal_places) for f in features]
                writer.writerow(formatted_features)

            self.logger.info(f"Features originais exportadas para: {csv_path}")

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=str(csv_path),
                metadata={
                    'features_dimension': len(features),
                    'file_size': csv_path.stat().st_size
                }
            )

        except Exception as e:
            self.logger.error(f"Erro ao exportar features originais: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                data=None,
                errors=[str(e)]
            )

    def export_original_features_by_category(
        self,
        spectral_features: Dict[str, np.ndarray],
        temporal_features: Dict[str, np.ndarray],
        filename: str,
        label: str = "real"
    ) -> ProcessingResult[List[str]]:
        """
        Exporta features originais organizadas por categoria.
        Cada linha representa um arquivo de áudio, cada coluna uma característica.

        Args:
            spectral_features: Dicionário de features espectrais
            temporal_features: Dicionário de features temporais
            filename: Nome do arquivo base
            label: Label do arquivo (real/fake)

        Returns:
            ProcessingResult com lista de caminhos dos arquivos criados
        """
        try:
            exported_files = []

            # Definir categorias e suas features
            categories = {
                'spectral': spectral_features,
                'temporal': temporal_features
            }

            for category_name, category_features in categories.items():
                if not category_features:
                    continue

                # Criar diretório de saída para a categoria
                output_dir = Path(self.config.output_base_dir) / \
                    self.config.original_dir / label / category_name
                output_dir.mkdir(parents=True, exist_ok=True)

                # Caminho do arquivo CSV
                csv_path = output_dir / f"{filename}_{category_name}.csv"

                # Preparar dados para CSV
                feature_names = []
                feature_values = []

                for feature_name, feature_array in category_features.items():
                    if isinstance(feature_array, np.ndarray):
                        if feature_array.ndim == 1 and feature_array.size > 1:
                            # Feature com múltiplos valores (ex: MFCC)
                            for i, value in enumerate(feature_array):
                                feature_names.append(
                                    f"{category_name}_{feature_name}_{i}")
                                feature_values.append(float(value))
                        else:
                            # Feature escalar (ndim == 0 ou array com 1
                            # elemento)
                            if feature_array.size == 1:
                                feature_names.append(
                                    f"{category_name}_{feature_name}")
                                feature_values.append(
                                    float(feature_array.item()))
                            else:
                                # Array vazio ou inválido
                                feature_names.append(
                                    f"{category_name}_{feature_name}")
                                feature_values.append(0.0)
                    elif isinstance(feature_array, (int, float)):
                        # Feature escalar simples
                        feature_names.append(f"{category_name}_{feature_name}")
                        feature_values.append(float(feature_array))

                # Escrever CSV
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(
                        csvfile, delimiter=self.config.delimiter)

                    # Escrever cabeçalho
                    if self.config.include_headers:
                        writer.writerow(feature_names)

                    # Escrever features (uma linha por arquivo)
                    formatted_features = [
                        round(f, self.config.decimal_places) for f in feature_values]
                    writer.writerow(formatted_features)

                exported_files.append(str(csv_path))
                self.logger.info(
                    f"Features {category_name} exportadas para: {csv_path}")

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=exported_files,
                metadata={
                    'categories_exported': len(exported_files),
                    'total_files': len(exported_files)
                }
            )

        except Exception as e:
            self.logger.error(
                f"Erro ao exportar features originais por categoria: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                data=None,
                errors=[str(e)]
            )

    def export_segmented_features_by_category(
        self,
        features_list: List[SegmentedFeatures],
        filename: str,
        label: str = "real"
    ) -> ProcessingResult[List[str]]:
        """
        Exporta features segmentadas organizadas por categoria.
        Cada linha representa um segmento de 1s, cada coluna uma característica.
        Gera um arquivo por categoria (spectral, temporal, cepstral, etc.).

        Args:
            features_list: Lista de features segmentadas
            filename: Nome do arquivo base
            label: Label do arquivo (real/fake)

        Returns:
            ProcessingResult com lista de caminhos dos arquivos criados
        """
        try:
            if not features_list:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    data=None,
                    errors=["Lista de features vazia"]
                )

            exported_files = []

            # Definir categorias de features baseadas no organize_features.py
            feature_categories = {
                'spectral': ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
                             'spectral_flatness', 'spectral_slope', 'spectral_kurtosis',
                             'spectral_skewness', 'spectral_spread', 'spectral_entropy',
                             'spectral_contrast', 'spectral_flux'],
                'cepstral': ['mfcc', 'lpcc', 'delta_mfcc', 'delta2_mfcc'],
                'temporal': ['zcr', 'energy', 'rms', 'tempo', 'onset_strength',
                             'autocorr', 'envelope'],
                'prosodic': ['f0', 'pitch', 'jitter', 'shimmer', 'hnr', 'intensity'],
                'formant': ['formant_f1', 'formant_f2', 'formant_f3'],
                'voice_quality': ['rap', 'ppq', 'apq', 'vf0', 'shimmer_db', 'nhr', 'vti',
                                  'spi', 'dfa_alpha', 'spectral_tilt', 'breathiness_index',
                                  'roughness_index', 'voice_breaks', 'shdb', 'breathiness',
                                  'roughness', 'hoarseness'],
                'perceptual': ['tonnetz', 'chroma', 'spectral_rolloff_perceptual'],
                'speech': ['vot', 'vowel_duration', 'speech_rate', 'pause'],
                'transform': ['stft', 'cqt', 'mel_spectrogram', 'wavelet', 'hilbert'],
                'complexity': ['lempel_ziv', 'sample_entropy', 'permutation_entropy',
                               'fractal_dimension']
            }

            # Organizar features por categoria
            for category_name, category_keywords in feature_categories.items():
                category_features = self._extract_category_features(
                    features_list, category_name, category_keywords
                )

                if not category_features['feature_names']:
                    continue  # Pular categorias sem features

                # Criar diretório de saída para a categoria
                output_dir = Path(self.config.output_base_dir) / \
                    self.config.segmented_dir / label / category_name
                output_dir.mkdir(parents=True, exist_ok=True)

                # Caminho do arquivo CSV
                csv_path = output_dir / f"{filename}_{category_name}.csv"

                # Escrever CSV
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(
                        csvfile, delimiter=self.config.delimiter)

                    # Escrever cabeçalho
                    if self.config.include_headers:
                        headers = self._generate_category_headers(
                            category_features)
                        writer.writerow(headers)

                    # Escrever dados (uma linha por segmento)
                    for row_data in category_features['data']:
                        formatted_row = [
                            round(
                                float(f),
                                self.config.decimal_places) for f in row_data]
                        writer.writerow(formatted_row)

                exported_files.append(str(csv_path))
                self.logger.info(
                    f"Features {category_name} exportadas para: {csv_path} ({
                        len(
                            category_features['data'])} segmentos)")

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=exported_files,
                metadata={
                    'categories_exported': len(exported_files),
                    'segments_count': len(features_list),
                    'total_files': len(exported_files)
                }
            )

        except Exception as e:
            self.logger.error(f"Erro ao exportar features por categoria: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                data=None,
                errors=[str(e)]
            )

    def _extract_category_features(self, features_list: List[SegmentedFeatures],
                                   category_name: str, category_keywords: List[str]) -> Dict[str, Any]:
        """
        Extrai features de uma categoria específica dos segmentos.

        Args:
            features_list: Lista de features segmentadas
            category_name: Nome da categoria
            category_keywords: Palavras-chave para identificar features da categoria

        Returns:
            Dicionário com feature_names e data organizados
        """
        category_features = {
            'feature_names': [],
            'data': []
        }

        if not features_list:
            return category_features

        # Identificar índices das features da categoria
        feature_indices = []
        feature_names = features_list[0].feature_names

        for i, feature_name in enumerate(feature_names):
            feature_name_lower = feature_name.lower()

            # Verificar se a feature pertence à categoria
            belongs_to_category = False

            # Verificar APENAS por prefixo da categoria (mais restritivo)
            if feature_name_lower.startswith(f"{category_name}_"):
                belongs_to_category = True

            if belongs_to_category:
                feature_indices.append(i)
                category_features['feature_names'].append(feature_name)

        # Extrair dados das features da categoria
        for segment_features in features_list:
            row_data = []

            # Adicionar metadados se configurado
            if self.config.include_metadata:
                row_data.extend([
                    segment_features.segment_index,
                    round(segment_features.start_time, 3),
                    round(segment_features.end_time, 3),
                    round(
                        segment_features.end_time -
                        segment_features.start_time,
                        3)  # duration
                ])

            # Adicionar features da categoria
            for idx in feature_indices:
                if idx < len(segment_features.combined_features):
                    row_data.append(segment_features.combined_features[idx])
                else:
                    row_data.append(0.0)  # valor padrão se índice inválido

            category_features['data'].append(row_data)

        return category_features

    def _generate_category_headers(
            self, category_features: Dict[str, Any]) -> List[str]:
        """
        Gera cabeçalhos para CSV de categoria específica.

        Args:
            category_features: Dados das features da categoria

        Returns:
            Lista de nomes dos cabeçalhos
        """
        headers = []

        # Adicionar cabeçalhos de metadados se configurado
        if self.config.include_metadata:
            headers.extend(
                ['segment_index', 'start_time', 'end_time', 'duration'])

        # Adicionar nomes das features
        headers.extend(category_features['feature_names'])

        return headers

    def _prepare_csv_data(
            self, features_list: List[SegmentedFeatures]) -> List[List[float]]:
        """
        Prepara dados para exportação CSV.
        Cada linha = um segmento, cada coluna = uma característica.
        """
        csv_data = []

        for segment_features in features_list:
            row = []

            # Adicionar metadados se configurado
            if self.config.include_metadata:
                row.extend([
                    segment_features.segment_index,
                    round(segment_features.start_time, 3),
                    round(segment_features.end_time, 3),
                    round(segment_features.duration, 3)
                ])

            # Adicionar features combinadas
            formatted_features = [
                round(float(f), self.config.decimal_places)
                for f in segment_features.combined_features
            ]
            row.extend(formatted_features)

            csv_data.append(row)

        return csv_data

    def _generate_headers(
            self, sample_features: SegmentedFeatures) -> List[str]:
        """
        Gera cabeçalhos para o CSV baseado nas features.
        """
        headers = []

        # Adicionar cabeçalhos de metadados se configurado
        if self.config.include_metadata:
            headers.extend([
                'segment_id',
                'start_time',
                'end_time',
                'duration'
            ])

        # Adicionar cabeçalhos de features
        if hasattr(sample_features,
                   'feature_names') and sample_features.feature_names:
            headers.extend(sample_features.feature_names)
        else:
            # Gerar nomes genéricos se não disponíveis
            feature_count = len(sample_features.combined_features)
            headers.extend([f'feature_{i + 1}' for i in range(feature_count)])

        return headers

    def export_batch(
        self,
        features_batch: Dict[str, List[SegmentedFeatures]],
        label: str = "real"
    ) -> ProcessingResult[List[str]]:
        """
        Exporta um lote de features segmentadas.

        Args:
            features_batch: Dicionário {filename: features_list}
            label: Label dos arquivos (real/fake)

        Returns:
            ProcessingResult com lista de caminhos criados
        """
        exported_files = []
        errors = []

        for filename, features_list in features_batch.items():
            result = self.export_segmented_features(
                features_list, filename, label)

            if result.status == ProcessingStatus.SUCCESS:
                exported_files.append(result.data)
            else:
                errors.extend(result.errors or [])

        if errors:
            return ProcessingResult(
                status=ProcessingStatus.PARTIAL_SUCCESS if exported_files else ProcessingStatus.ERROR,
                data=exported_files,
                errors=errors
            )

        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=exported_files,
            metadata={'exported_count': len(exported_files)}
        )


def main():
    """Função principal para teste"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("🔧 Testando CSVFeatureExporter...")

    # Criar exportador
    exporter = CSVFeatureExporter()

    print("✅ CSVFeatureExporter criado com sucesso!")
    print(f"📁 Diretório base: {exporter.config.output_base_dir}")
    print(f"📁 Diretório segmentado: {exporter.config.segmented_dir}")
    print(f"📁 Diretório original: {exporter.config.original_dir}")


if __name__ == "__main__":
    main()
