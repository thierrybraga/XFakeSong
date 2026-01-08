"""Pipeline de Treinamento Seguro para Prevenção de Data Leakage

Este módulo implementa um pipeline de treinamento que previne vazamento de dados
através de práticas seguras de divisão de dados e normalização.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
import tensorflow as tf
import joblib

from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.core.interfaces.audio import AudioData

logger = logging.getLogger(__name__)


@dataclass
class SecureTrainingConfig:
    """Configuração para treinamento seguro."""
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    use_temporal_split: bool = True
    scaler_type: str = "standard"  # "standard" ou "minmax"
    save_scaler: bool = True
    validate_no_leakage: bool = True


class SecureDataSplitter:
    """Divisor de dados seguro que previne data leakage."""

    def __init__(self, config: SecureTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def split_data(self, X: np.ndarray, y: np.ndarray,
                   metadata: Optional[Dict] = None) -> Tuple[np.ndarray, ...]:
        """Divide dados de forma segura.

        Args:
            X: Features
            y: Labels
            metadata: Metadados opcionais (ex: timestamps)

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        self.logger.info("Iniciando divisão segura dos dados")

        if self.config.use_temporal_split and metadata and 'timestamps' in metadata:
            return self._temporal_split(X, y, metadata['timestamps'])
        else:
            return self._stratified_split(X, y)

    def _temporal_split(self, X: np.ndarray, y: np.ndarray,
                        timestamps: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Divisão temporal para dados sequenciais."""
        # Ordenar por timestamp
        sorted_indices = np.argsort(timestamps)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]

        n_samples = len(X_sorted)
        test_split_idx = int(n_samples * (1 - self.config.test_size))
        val_split_idx = int(test_split_idx * (1 - self.config.validation_size))

        # Divisão temporal: treino -> validação -> teste
        X_train = X_sorted[:val_split_idx]
        y_train = y_sorted[:val_split_idx]

        X_val = X_sorted[val_split_idx:test_split_idx]
        y_val = y_sorted[val_split_idx:test_split_idx]

        X_test = X_sorted[test_split_idx:]
        y_test = y_sorted[test_split_idx:]

        self.logger.info(
            f"Divisão temporal: Train={
                len(X_train)}, Val={
                len(X_val)}, Test={
                len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _stratified_split(self, X: np.ndarray,
                          y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Divisão estratificada para dados não-temporais."""
        # Primeira divisão: treino+val vs teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state, stratify=y
        )

        # Segunda divisão: treino vs validação
        val_size_adjusted = self.config.validation_size / \
            (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=self.config.random_state, stratify=y_temp
        )

        self.logger.info(
            f"Divisão estratificada: Train={
                len(X_train)}, Val={
                len(X_val)}, Test={
                len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test


class SecureFeatureScaler:
    """Scaler seguro que previne data leakage."""

    def __init__(self, config: SecureTrainingConfig):
        self.config = config
        self.scaler = None
        self.logger = logging.getLogger(__name__)

        # Inicializar scaler baseado na configuração
        if config.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif config.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"Tipo de scaler não suportado: {
                    config.scaler_type}")

    def fit_transform_train(self, X_train: np.ndarray) -> np.ndarray:
        """Ajusta o scaler APENAS nos dados de treino e transforma.

        CRÍTICO: Este método deve ser chamado APENAS com dados de treino!
        """
        self.logger.info("Ajustando scaler apenas nos dados de treino")

        # Verificar se os dados não estão vazios
        if len(X_train) == 0:
            raise ValueError("Dados de treino não podem estar vazios")

        # Ajustar e transformar dados de treino
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.logger.info(
            f"Scaler ajustado. Média: {
                self.scaler.mean_[
                    :5] if hasattr(
                    self.scaler,
                    'mean_') else 'N/A'}")
        return X_train_scaled

    def transform_validation(self, X_val: np.ndarray) -> np.ndarray:
        """Transforma dados de validação usando scaler já ajustado."""
        if self.scaler is None:
            raise ValueError(
                "Scaler deve ser ajustado primeiro com dados de treino")

        self.logger.info("Transformando dados de validação")
        return self.scaler.transform(X_val)

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        """Transforma dados de teste usando scaler já ajustado."""
        if self.scaler is None:
            raise ValueError(
                "Scaler deve ser ajustado primeiro com dados de treino")

        self.logger.info("Transformando dados de teste")
        return self.scaler.transform(X_test)

    def save_scaler(self, path: Union[str, Path]) -> None:
        """Salva o scaler para uso futuro."""
        if self.scaler is None:
            raise ValueError("Scaler deve ser ajustado primeiro")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, path)
        self.logger.info(f"Scaler salvo em: {path}")

    def load_scaler(self, path: Union[str, Path]) -> None:
        """Carrega scaler salvo."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler não encontrado: {path}")

        self.scaler = joblib.load(path)
        self.logger.info(f"Scaler carregado de: {path}")


class DataLeakageValidator:
    """Validador para detectar possíveis vazamentos de dados."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_data_splits(self, X_train: np.ndarray, X_val: np.ndarray,
                             X_test: np.ndarray) -> Dict[str, Any]:
        """Valida se há sobreposição entre conjuntos de dados."""
        validation_results = {
            "no_overlap": True,
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "warnings": []
        }

        # Verificar sobreposições (simplificado - em produção seria mais
        # sofisticado)
        if np.array_equal(X_train, X_val):
            validation_results["train_val_overlap"] = 1
            validation_results["no_overlap"] = False
            validation_results["warnings"].append(
                "Sobreposição detectada entre treino e validação")

        if np.array_equal(X_train, X_test):
            validation_results["train_test_overlap"] = 1
            validation_results["no_overlap"] = False
            validation_results["warnings"].append(
                "Sobreposição detectada entre treino e teste")

        if np.array_equal(X_val, X_test):
            validation_results["val_test_overlap"] = 1
            validation_results["no_overlap"] = False
            validation_results["warnings"].append(
                "Sobreposição detectada entre validação e teste")

        return validation_results

    def validate_scaler_usage(self, scaler: SecureFeatureScaler,
                              X_train: np.ndarray, X_val: np.ndarray) -> Dict[str, Any]:
        """Valida se o scaler foi usado corretamente."""
        validation_results = {
            "scaler_fitted": scaler.scaler is not None,
            "proper_usage": True,
            "warnings": []
        }

        if not validation_results["scaler_fitted"]:
            validation_results["proper_usage"] = False
            validation_results["warnings"].append("Scaler não foi ajustado")

        # Verificar se as estatísticas do scaler fazem sentido
        if hasattr(scaler.scaler, 'mean_') and hasattr(
                scaler.scaler, 'scale_'):
            train_mean = np.mean(X_train, axis=0)
            scaler_mean = scaler.scaler.mean_

            # Verificar se as médias são similares (tolerância de 1%)
            if not np.allclose(train_mean, scaler_mean, rtol=0.01):
                validation_results["warnings"].append(
                    "Estatísticas do scaler não correspondem aos dados de treino"
                )

        return validation_results


class SecureTrainingPipeline:
    """Pipeline completo de treinamento seguro."""

    def __init__(self, config: SecureTrainingConfig):
        self.config = config
        self.splitter = SecureDataSplitter(config)
        self.scaler = SecureFeatureScaler(config)
        self.validator = DataLeakageValidator()
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                     metadata: Optional[Dict] = None) -> ProcessingResult[Dict[str, np.ndarray]]:
        """Prepara dados de forma segura para treinamento.

        Returns:
            Dicionário com dados divididos e normalizados
        """
        try:
            self.logger.info("Iniciando preparação segura dos dados")

            # 1. Dividir dados
            X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.split_data(
                X, y, metadata)

            # 2. Validar divisão se habilitado
            if self.config.validate_no_leakage:
                split_validation = self.validator.validate_data_splits(
                    X_train, X_val, X_test)
                if not split_validation["no_overlap"]:
                    self.logger.warning(
                        "Possível vazamento detectado na divisão dos dados")
                    for warning in split_validation["warnings"]:
                        self.logger.warning(warning)

            # 3. Normalizar dados (ORDEM CRÍTICA: ajustar apenas no treino!)
            X_train_scaled = self.scaler.fit_transform_train(X_train)
            X_val_scaled = self.scaler.transform_validation(X_val)
            X_test_scaled = self.scaler.transform_test(X_test)

            # 4. Validar uso do scaler se habilitado
            if self.config.validate_no_leakage:
                scaler_validation = self.validator.validate_scaler_usage(
                    self.scaler, X_train, X_val)
                if not scaler_validation["proper_usage"]:
                    self.logger.warning("Possível problema no uso do scaler")
                    for warning in scaler_validation["warnings"]:
                        self.logger.warning(warning)

            # 5. Salvar scaler se habilitado
            if self.config.save_scaler:
                scaler_path = Path("models/scalers/secure_scaler.pkl")
                self.scaler.save_scaler(scaler_path)

            prepared_data = {
                "X_train": X_train_scaled,
                "X_val": X_val_scaled,
                "X_test": X_test_scaled,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test
            }

            self.logger.info("Preparação dos dados concluída com sucesso")
            return ProcessingResult(
                success=True,
                data=prepared_data,
                status=ProcessingStatus.COMPLETED
            )

        except Exception as e:
            self.logger.error(f"Erro na preparação dos dados: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                status=ProcessingStatus.FAILED
            )

    def get_scaler(self) -> SecureFeatureScaler:
        """Retorna o scaler para uso em predições futuras."""
        return self.scaler

# Função de conveniência para uso rápido


def create_secure_pipeline(use_temporal_split: bool = True,
                           scaler_type: str = "standard") -> SecureTrainingPipeline:
    """Cria pipeline seguro com configurações padrão."""
    config = SecureTrainingConfig(
        use_temporal_split=use_temporal_split,
        scaler_type=scaler_type
    )
    return SecureTrainingPipeline(config)
