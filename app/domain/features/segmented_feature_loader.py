"""Segmented Feature Loader

Módulo para carregar e processar features extraídas da pasta segmented.
Este módulo é responsável por:
- Carregar features de diferentes tipos (spectral, cepstral, etc.)
- Combinar features de múltiplos tipos
- Preparar dados para treinamento de modelos clássicos (SVM, Random Forest)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [SegmentedFeatureLoader] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class SegmentedFeatureLoader:
    """
    Carregador de features extraídas da pasta segmented.
    Suporta diferentes tipos de features e combinações.
    """

    FEATURE_TYPES = [
        'spectral', 'cepstral', 'formant', 'perceptual',
        'prosodic', 'speech', 'temporal', 'transform',
        'voice_quality', 'complexity', 'rhythm', 'energy',
        'pitch', 'harmonics', 'noise', 'dynamics'
    ]

    def __init__(self,
                 segmented_path: str = "datasets/features/segmented",
                 feature_types: Optional[List[str]] = None,
                 normalize: bool = True,
                 aggregate_method: str = 'mean'):
        """
        Inicializa o carregador de features.

        Args:
            segmented_path: Caminho para a pasta segmented
            feature_types: Lista de tipos de features a carregar (None = todos)
            normalize: Se deve normalizar as features
            aggregate_method: Método de agregação ('mean', 'median', 'std', 'all')
        """
        self.segmented_path = Path(segmented_path)
        self.feature_types = feature_types or self.FEATURE_TYPES
        self.normalize = normalize
        self.aggregate_method = aggregate_method

        self.scaler = StandardScaler() if normalize else None
        self.label_encoder = LabelEncoder()

        # Validar tipos de features
        invalid_types = set(self.feature_types) - set(self.FEATURE_TYPES)
        if invalid_types:
            raise ValueError(f"Invalid feature types: {invalid_types}")

        logger.info(f"SegmentedFeatureLoader initialized")
        logger.info(f"Path: {self.segmented_path}")
        logger.info(f"Feature types: {self.feature_types}")
        logger.info(f"Normalize: {normalize}, Aggregate: {aggregate_method}")

    def load_single_feature_file(self, filepath: Path) -> pd.DataFrame:
        """
        Carrega um único arquivo de features CSV.

        Args:
            filepath: Caminho para o arquivo CSV

        Returns:
            DataFrame com as features
        """
        try:
            df = pd.read_csv(filepath)

            # Remover colunas de metadados se existirem
            metadata_cols = [
                'segment_index',
                'start_time',
                'end_time',
                'duration']
            feature_cols = [
                col for col in df.columns if col not in metadata_cols]

            return df[feature_cols]

        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")
            return pd.DataFrame()

    def aggregate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Agrega features de múltiplos segmentos.

        Args:
            df: DataFrame com features de segmentos

        Returns:
            Array numpy com features agregadas
        """
        if df.empty:
            return np.array([])

        # Remover colunas que são completamente NaN
        df_clean = df.dropna(axis=1, how='all')

        if df_clean.empty:
            logger.warning("All columns contain only NaN values")
            return np.array([])

        if self.aggregate_method == 'mean':
            result = df_clean.mean().fillna(0).values
        elif self.aggregate_method == 'median':
            result = df_clean.median().fillna(0).values
        elif self.aggregate_method == 'std':
            result = df_clean.std().fillna(0).values
        elif self.aggregate_method == 'all':
            # Combina mean, std, min, max
            mean_vals = df_clean.mean().fillna(0).values
            std_vals = df_clean.std().fillna(0).values
            min_vals = df_clean.min().fillna(0).values
            max_vals = df_clean.max().fillna(0).values
            result = np.concatenate([mean_vals, std_vals, min_vals, max_vals])
        else:
            raise ValueError(
                f"Unknown aggregate method: {
                    self.aggregate_method}")

        # Verificar se ainda há NaN e substituir por 0
        if np.isnan(result).any():
            logger.warning(
                f"Found {
                    np.isnan(result).sum()} NaN values, replacing with 0")
            result = np.nan_to_num(result, nan=0.0)

        return result

    def load_sample_features(self,
                             sample_path: Path,
                             feature_type: str) -> np.ndarray:
        """
        Carrega features de um sample específico para um tipo de feature.

        Args:
            sample_path: Caminho para a pasta do sample (real ou fake)
            feature_type: Tipo de feature a carregar

        Returns:
            Array numpy com features agregadas do sample
        """
        feature_dir = sample_path / feature_type

        if not feature_dir.exists():
            logger.warning(f"Feature directory not found: {feature_dir}")
            return np.array([])

        # Carregar todos os arquivos CSV do tipo de feature
        csv_files = list(feature_dir.glob("*.csv"))

        if not csv_files:
            logger.warning(f"No CSV files found in {feature_dir}")
            return np.array([])

        # Combinar features de todos os arquivos
        all_features = []
        for csv_file in csv_files:
            df = self.load_single_feature_file(csv_file)
            if not df.empty:
                all_features.append(df)

        if not all_features:
            return np.array([])

        # Concatenar e agregar
        combined_df = pd.concat(all_features, ignore_index=True)
        return self.aggregate_features(combined_df)

    def load_dataset(self,
                     classes: List[str] = ['real', 'fake']) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Carrega o dataset completo com features de todas as classes.

        Args:
            classes: Lista de classes a carregar

        Returns:
            Tuple com (features, labels, feature_names)
        """
        logger.info(f"Loading dataset for classes: {classes}")
        logger.info(f"Feature types: {self.feature_types}")

        all_features = []
        all_labels = []
        feature_names = []

        for class_name in classes:
            class_path = self.segmented_path / class_name

            if not class_path.exists():
                logger.warning(f"Class directory not found: {class_path}")
                continue

            logger.info(f"Loading features for class: {class_name}")

            # Para cada tipo de feature
            class_features_by_type = []

            for feature_type in self.feature_types:
                # Carregar features deste tipo para esta classe
                sample_features = self.load_sample_features(
                    class_path, feature_type)

                if sample_features.size > 0:
                    class_features_by_type.append(sample_features)

                    # Gerar nomes das features na primeira iteração
                    if not feature_names:
                        if self.aggregate_method == 'all':
                            base_size = len(sample_features) // 4
                            suffixes = ['_mean', '_std', '_min', '_max']
                            for i in range(base_size):
                                for suffix in suffixes:
                                    feature_names.append(
                                        f"{feature_type}_{i}{suffix}")
                        else:
                            for i in range(len(sample_features)):
                                feature_names.append(
                                    f"{feature_type}_{i}_{
                                        self.aggregate_method}")

            # Combinar features de todos os tipos para esta classe
            if class_features_by_type:
                # Verificar se todas as features têm a mesma dimensão
                feature_shapes = [f.shape for f in class_features_by_type]
                logger.debug(
                    f"Feature shapes for {class_name}: {feature_shapes}")

                combined_features = np.concatenate(class_features_by_type)
                all_features.append(combined_features)
                all_labels.append(class_name)

                logger.info(
                    f"Loaded {
                        len(combined_features)} features for class {class_name}")

        if not all_features:
            raise ValueError(
                "No features loaded. Check paths and data availability.")

        # Verificar se todas as features têm a mesma dimensão
        if all_features:
            feature_lengths = [len(f) for f in all_features]
            if len(set(feature_lengths)) > 1:
                logger.warning(
                    f"Features have different lengths: {
                        set(feature_lengths)}")
                # Usar o tamanho mínimo para padronizar
                min_length = min(feature_lengths)
                all_features = [f[:min_length] for f in all_features]
                logger.info(f"Truncated all features to length: {min_length}")

        # Converter para arrays numpy
        X = np.array(all_features)
        y = np.array(all_labels)

        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)

        logger.info(f"Dataset loaded successfully")
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y_encoded.shape}")
        logger.info(f"Classes: {self.label_encoder.classes_}")
        logger.info(f"Feature names: {len(feature_names)} features")

        return X, y_encoded, feature_names

    def load_multiple_samples_per_class(self,
                                        classes: List[str] = ['real', 'fake'],
                                        max_samples_per_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Carrega múltiplas amostras por classe, tratando cada arquivo como uma amostra separada.

        Args:
            classes: Lista de classes a carregar
            max_samples_per_class: Máximo de amostras por classe (None = todas)

        Returns:
            Tuple com (features, labels, feature_names)
        """
        logger.info(f"Loading multiple samples per class: {classes}")
        logger.info(f"Max samples per class: {max_samples_per_class}")
        logger.info(f"Feature types: {self.feature_types}")

        all_features = []
        all_labels = []
        feature_names = []

        # Primeiro, determinar a estrutura das features carregando uma amostra
        reference_features = None

        for class_name in classes:
            class_path = self.segmented_path / class_name

            if not class_path.exists():
                logger.warning(f"Class directory not found: {class_path}")
                continue

            logger.info(f"Processing class: {class_name}")

            # Obter lista de amostras disponíveis
            sample_ids = set()
            for feature_type in self.feature_types:
                feature_dir = class_path / feature_type
                if feature_dir.exists():
                    for csv_file in feature_dir.glob("*.csv"):
                        # Extrair ID da amostra do nome do arquivo
                        # Formato esperado:
                        # {class}_sample_{id}_{feature_type}.csv
                        parts = csv_file.stem.split('_')
                        if len(parts) >= 3:
                            sample_id = parts[2]  # sample_XXX
                            sample_ids.add(sample_id)

            sample_ids = sorted(list(sample_ids))
            logger.info(f"Found {len(sample_ids)} samples for {class_name}")

            # Limitar número de amostras se especificado
            if max_samples_per_class:
                sample_ids = sample_ids[:max_samples_per_class]
                logger.info(
                    f"Limited to {
                        len(sample_ids)} samples for {class_name}")

            # Carregar features para cada amostra
            for sample_id in sample_ids:
                sample_features_by_type = []

                for feature_type in self.feature_types:
                    feature_dir = class_path / feature_type

                    if not feature_dir.exists():
                        logger.warning(
                            f"Feature directory not found: {feature_dir}")
                        continue

                    # Procurar arquivo específico da amostra
                    pattern = f"{class_name}_sample_{sample_id}_{feature_type}.csv"
                    csv_files = list(feature_dir.glob(pattern))

                    if not csv_files:
                        logger.warning(
                            f"No CSV file found for pattern: {pattern}")
                        continue

                    # Carregar e agregar features do arquivo
                    df = self.load_single_feature_file(csv_files[0])
                    if not df.empty:
                        aggregated = self.aggregate_features(df)
                        if aggregated.size > 0:
                            sample_features_by_type.append(aggregated)

                # Combinar features de todos os tipos para esta amostra
                if sample_features_by_type:
                    try:
                        combined_features = np.concatenate(
                            sample_features_by_type)

                        # Usar como referência para padronizar outras amostras
                        if reference_features is None:
                            reference_features = combined_features
                            reference_length = len(combined_features)
                            logger.info(
                                f"Reference feature length set to: {reference_length}")

                            # Gerar nomes das features
                            feature_idx = 0
                            for i, feature_type in enumerate(
                                    self.feature_types):
                                if i < len(sample_features_by_type):
                                    feature_array = sample_features_by_type[i]
                                    if self.aggregate_method == 'all':
                                        base_size = len(feature_array) // 4
                                        suffixes = [
                                            '_mean', '_std', '_min', '_max']
                                        for j in range(base_size):
                                            for suffix in suffixes:
                                                feature_names.append(
                                                    f"{feature_type}_{j}{suffix}")
                                    else:
                                        for j in range(len(feature_array)):
                                            feature_names.append(
                                                f"{feature_type}_{j}_{self.aggregate_method}")

                        # Padronizar tamanho se necessário
                        if len(combined_features) != reference_length:
                            logger.warning(
                                f"Feature length mismatch: {
                                    len(combined_features)} vs {reference_length}")
                            min_length = min(
                                len(combined_features), reference_length)
                            combined_features = combined_features[:min_length]
                            if len(reference_features) > min_length:
                                reference_features = reference_features[:min_length]
                                reference_length = min_length

                        all_features.append(combined_features)
                        all_labels.append(class_name)

                    except Exception as e:
                        logger.error(
                            f"Error combining features for sample {sample_id}: {e}")
                        continue

        if not all_features:
            raise ValueError(
                "No features loaded. Check paths and data availability.")

        # Verificar se todas as features têm a mesma dimensão
        if all_features:
            feature_lengths = [len(f) for f in all_features]
            if len(set(feature_lengths)) > 1:
                logger.warning(
                    f"Features have different lengths: {
                        set(feature_lengths)}")
                # Usar o tamanho mínimo para padronizar
                min_length = min(feature_lengths)
                all_features = [f[:min_length] for f in all_features]
                feature_names = feature_names[:min_length]
                logger.info(f"Truncated all features to length: {min_length}")

        # Converter para arrays numpy
        X = np.array(all_features)
        y = np.array(all_labels)

        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)

        logger.info(f"Multiple samples dataset loaded successfully")
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y_encoded.shape}")
        logger.info(f"Classes: {self.label_encoder.classes_}")
        logger.info(f"Total samples: {len(all_features)}")

        return X, y_encoded, feature_names

    def prepare_train_test_split(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 test_size: float = 0.2,
                                 random_state: int = 42,
                                 stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara divisão treino/teste com normalização opcional.

        Args:
            X: Features
            y: Labels
            test_size: Proporção do conjunto de teste
            random_state: Seed para reprodutibilidade
            stratify: Se deve estratificar a divisão

        Returns:
            Tuple com (X_train, X_test, y_train, y_test)
        """
        # Divisão treino/teste
        stratify_param = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

        # Normalização se habilitada
        if self.normalize and self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        logger.info(f"Train/test split completed")
        logger.info(
            f"Train shape: {
                X_train.shape}, Test shape: {
                X_test.shape}")

        return X_train, X_test, y_train, y_test

    def get_feature_importance_names(self) -> List[str]:
        """
        Retorna nomes das features para análise de importância.

        Returns:
            Lista com nomes das features
        """
        feature_names = []

        for feature_type in self.feature_types:
            # Estimativa baseada nos tipos de features conhecidos
            if feature_type == 'spectral':
                base_count = 100  # Estimativa
            elif feature_type == 'cepstral':
                base_count = 50
            else:
                base_count = 30

            if self.aggregate_method == 'all':
                suffixes = ['_mean', '_std', '_min', '_max']
                for i in range(base_count):
                    for suffix in suffixes:
                        feature_names.append(f"{feature_type}_{i}{suffix}")
            else:
                for i in range(base_count):
                    feature_names.append(
                        f"{feature_type}_{i}_{
                            self.aggregate_method}")

        return feature_names


def create_feature_loader(
    segmented_path: str = "datasets/features/segmented",
    feature_types: Optional[List[str]] = None,
    normalize: bool = True,
    aggregate_method: str = 'mean'
) -> SegmentedFeatureLoader:
    """
    Função de conveniência para criar um carregador de features.

    Args:
        segmented_path: Caminho para a pasta segmented
        feature_types: Lista de tipos de features
        normalize: Se deve normalizar
        aggregate_method: Método de agregação

    Returns:
        Instância do carregador
    """
    return SegmentedFeatureLoader(
        segmented_path=segmented_path,
        feature_types=feature_types,
        normalize=normalize,
        aggregate_method=aggregate_method
    )
