"""Pipeline de Treinamento Seguro para Prevenção de Data Leakage

Este módulo implementa um pipeline de treinamento que previne vazamento de dados
através de práticas seguras de divisão de dados e normalização.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.core.interfaces.base import ProcessingResult, ProcessingStatus

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
            f"Divisão temporal: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
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
            f"Divisão estratificada: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test


class SecureFeatureScaler:
    """Scaler seguro que previne data leakage."""

    def __init__(self, config: SecureTrainingConfig):
        self.config = config
        self.scaler = None
        self.logger = logging.getLogger(__name__)
        # True quando o input NÃO é tabular (espectrograma/raw-audio, ndim != 2):
        # StandardScaler/MinMaxScaler exigem 2D e modelos profundos normalizam
        # internamente, então pulamos a escala. Ver fit_transform_train.
        self._skip_scaling = False

        # Inicializar scaler baseado na configuração
        if config.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif config.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"Tipo de scaler não suportado: {config.scaler_type}")

    @staticmethod
    def _sanitize(X: np.ndarray, tag: str, logger) -> np.ndarray:
        """Substitui NaN/Inf por 0.0 e loga se encontrado.

        BUG FIX: StandardScaler propaga NaN/Inf diretamente para os pesos do
        modelo, causando loss: nan na 1ª época. Fontes comuns: arquivos de
        áudio corrompidos, silêncio puro, clipping extremo, ou spectrogramas
        com valores indefinidos (log de 0 não protegido por epsilon).
        """
        nan_count = int(np.sum(np.isnan(X)))
        inf_count = int(np.sum(np.isinf(X)))
        if nan_count > 0 or inf_count > 0:
            logger.warning(
                f"[sanitize/{tag}] {nan_count} NaN e {inf_count} Inf "
                f"encontrados em {X.shape} — substituídos por 0.0. "
                "Verifique os arquivos de áudio do dataset."
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit_transform_train(self, X_train: np.ndarray) -> np.ndarray:
        """Ajusta o scaler APENAS nos dados de treino e transforma.

        CRÍTICO: Este método deve ser chamado APENAS com dados de treino!
        """
        self.logger.info("Ajustando scaler apenas nos dados de treino")

        if len(X_train) == 0:
            raise ValueError("Dados de treino não podem estar vazios")

        # BUG FIX: sanitizar antes de fit — NaN/Inf no input corrompem o scaler
        X_train = self._sanitize(X_train, "train", self.logger)

        # Inputs NÃO-tabulares (espectrograma 3D, raw-audio 2D/3D com canal):
        # StandardScaler/MinMaxScaler exigem 2D (n_samples, n_features) e
        # levantariam "Found array with dim 3, while dim <= 2 is required".
        # Modelos profundos normalizam internamente (AudioNormalizationLayer/
        # BatchNorm); além disso, aplicar um scaler no treino SEM reaplicá-lo na
        # inferência (o FeaturePreparer não usa este scaler) criaria mismatch
        # train/inference. Então PULAMOS a escala — só sanitizamos NaN/Inf.
        if X_train.ndim != 2:
            self._skip_scaling = True
            self.scaler = None
            self.logger.info(
                f"Scaler pulado (input ndim={X_train.ndim}, não-tabular). "
                "Normalização fica a cargo das camadas internas do modelo."
            )
            return X_train

        X_train_scaled = self.scaler.fit_transform(X_train)

        # Sanitizar pós-escala (StandardScaler pode produzir NaN se std=0)
        X_train_scaled = self._sanitize(X_train_scaled, "train_scaled", self.logger)

        self.logger.info(
            f"Scaler ajustado. Média: {self.scaler.mean_[:5] if hasattr(self.scaler, 'mean_') else 'N/A'}")
        return X_train_scaled

    def transform_validation(self, X_val: np.ndarray) -> np.ndarray:
        """Transforma dados de validação usando scaler já ajustado."""
        # Input não-tabular: passthrough (só sanitiza), consistente com o train.
        if self._skip_scaling:
            return self._sanitize(X_val, "val", self.logger)
        if self.scaler is None:
            raise ValueError(
                "Scaler deve ser ajustado primeiro com dados de treino")

        self.logger.info("Transformando dados de validação")
        X_val = self._sanitize(X_val, "val", self.logger)
        result = self.scaler.transform(X_val)
        return self._sanitize(result, "val_scaled", self.logger)

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        """Transforma dados de teste usando scaler já ajustado."""
        if self._skip_scaling:
            return self._sanitize(X_test, "test", self.logger)
        if self.scaler is None:
            raise ValueError(
                "Scaler deve ser ajustado primeiro com dados de treino")

        self.logger.info("Transformando dados de teste")
        X_test = self._sanitize(X_test, "test", self.logger)
        result = self.scaler.transform(X_test)
        return self._sanitize(result, "test_scaled", self.logger)

    def save_scaler(self, path: Union[str, Path]) -> None:
        """Salva o scaler para uso futuro."""
        # Input não-tabular: não há scaler a salvar (no-op, não é erro).
        if self._skip_scaling or self.scaler is None:
            self.logger.info("Nenhum scaler a salvar (input não-tabular).")
            return

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

            # 4. Validar uso do scaler se habilitado (só faz sentido p/ tabular)
            if self.config.validate_no_leakage and self.scaler.scaler is not None:
                scaler_validation = self.validator.validate_scaler_usage(
                    self.scaler, X_train, X_val)
                if not scaler_validation["proper_usage"]:
                    self.logger.warning("Possível problema no uso do scaler")
                    for warning in scaler_validation["warnings"]:
                        self.logger.warning(warning)

            # 5. Salvar scaler se habilitado (no-op se não-tabular)
            if self.config.save_scaler and self.scaler.scaler is not None:
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
                status=ProcessingStatus.SUCCESS,
                data=prepared_data,
            )

        except Exception as e:
            self.logger.error(f"Erro na preparação dos dados: {str(e)}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
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
