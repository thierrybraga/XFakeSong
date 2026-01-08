"""Configuração Otimizada de Treinamento para Redução de Overfitting

Este módulo centraliza todas as configurações e callbacks otimizados
para treinar modelos com máxima generalização e mínimo overfitting.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LearningRateScheduler, TensorBoard
)
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

# Importar nossos callbacks customizados
from ..callbacks.anti_overfitting import (
    AdvancedEarlyStopping,
    GradientClippingCallback,
    LossMonitoringCallback,
    OverfittingDetectionCallback
)
from ..augmentation.advanced_audio_augmentation import (
    AdvancedAudioAugmentation,
    create_augmentation_pipeline,
    create_adaptive_augmentation_layer,
    create_robust_dataset
)

logger = logging.getLogger(__name__)


class OptimizedTrainingConfig:
    """Configuração otimizada para treinamento anti-overfitting."""

    def __init__(self,
                 model_name: str = "default",
                 patience: int = 15,
                 min_delta: float = 0.001,
                 lr_reduction_factor: float = 0.5,
                 lr_patience: int = 8,
                 gradient_clip_value: float = 1.0,
                 augmentation_strength: float = 0.3):

        self.model_name = model_name
        self.patience = patience
        self.min_delta = min_delta
        self.lr_reduction_factor = lr_reduction_factor
        self.lr_patience = lr_patience
        self.gradient_clip_value = gradient_clip_value
        self.augmentation_strength = augmentation_strength

        self._apply_architecture_config()

    def _apply_architecture_config(self):
        """Aplica configurações específicas da arquitetura a partir do banco de dados/registry."""
        try:
            # Importação tardia para evitar ciclos
            from ..architectures.registry import ArchitectureRegistry
            registry = ArchitectureRegistry()
            
            # Tenta obter config do banco (via registry) ou defaults do registry
            config = registry.get_active_config(self.model_name)
            
            # Atualiza apenas se as chaves existirem no config retornado
            if "patience" in config:
                self.patience = config["patience"]
            if "lr_patience" in config:
                self.lr_patience = config["lr_patience"]
            if "gradient_clip" in config:
                self.gradient_clip_value = config["gradient_clip"]
            if "augmentation_strength" in config:
                self.augmentation_strength = config["augmentation_strength"]
                
        except Exception as e:
            logger.warning(f"Erro ao carregar config da arquitetura {self.model_name}: {e}. Usando defaults fornecidos no init.")

    def get_callbacks(self,
                      model_checkpoint_path: str,
                      tensorboard_log_dir: str = None) -> List[tf.keras.callbacks.Callback]:
        """Retorna lista de callbacks otimizados."""

        callbacks = []

        # Early Stopping Avançado
        callbacks.append(
            AdvancedEarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                min_delta=self.min_delta,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            )
        )

        # Redução de Learning Rate
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.lr_reduction_factor,
                patience=self.lr_patience,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            )
        )

        # Model Checkpoint
        callbacks.append(
            ModelCheckpoint(
                filepath=model_checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            )
        )

        # Gradient Clipping
        callbacks.append(
            GradientClippingCallback(
                clip_value=self.gradient_clip_value
            )
        )

        # Monitoramento de Loss
        callbacks.append(
            LossMonitoringCallback(
                patience=self.patience // 2
            )
        )

        # Detecção de Overfitting
        callbacks.append(
            OverfittingDetectionCallback(
                threshold=0.1,
                patience=self.patience // 3
            )
        )

        # TensorBoard (opcional)
        if tensorboard_log_dir:
            callbacks.append(
                TensorBoard(
                    log_dir=tensorboard_log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch'
                )
            )

        return callbacks

    def get_optimizer(
            self, learning_rate: float = 0.001) -> tf.keras.optimizers.Optimizer:
        """Retorna otimizador otimizado (AdamW com weight decay)."""
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )

    def get_augmentation_pipeline(self,
                                  sample_rate: int = 16000,
                                  max_duration: float = 5.0) -> AdvancedAudioAugmentation:
        """Retorna pipeline de augmentation otimizado."""
        return create_augmentation_pipeline(
            sample_rate=sample_rate,
            max_duration=max_duration,
            augmentation_strength=self.augmentation_strength
        )

    def create_augmented_dataset(self,
                                 dataset: tf.data.Dataset,
                                 sample_rate: int = 16000,
                                 max_duration: float = 5.0) -> tf.data.Dataset:
        """Cria dataset com augmentation robusta."""
        return create_robust_dataset(
            dataset=dataset,
            sample_rate=sample_rate,
            max_duration=max_duration,
            augmentation_strength=self.augmentation_strength,
            batch_size=32
        )

    def get_training_config(self) -> Dict[str, Any]:
        """Retorna configuração completa de treinamento."""
        return {
            "model_name": self.model_name,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "lr_reduction_factor": self.lr_reduction_factor,
            "lr_patience": self.lr_patience,
            "gradient_clip_value": self.gradient_clip_value,
            "augmentation_strength": self.augmentation_strength,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "early_stopping": True,
            "gradient_clipping": True,
            "advanced_augmentation": True,
            "overfitting_detection": True
        }


def create_optimized_training_setup(model_name: str,
                                    model_checkpoint_path: str,
                                    tensorboard_log_dir: str = None,
                                    learning_rate: float = 0.001) -> Tuple[List[tf.keras.callbacks.Callback],
                                                                           tf.keras.optimizers.Optimizer,
                                                                           Dict[str, Any]]:
    """Cria setup completo de treinamento otimizado."""

    config = OptimizedTrainingConfig(model_name=model_name)

    callbacks = config.get_callbacks(
        model_checkpoint_path=model_checkpoint_path,
        tensorboard_log_dir=tensorboard_log_dir
    )

    optimizer = config.get_optimizer(learning_rate=learning_rate)

    training_config = config.get_training_config()

    logger.info(f"Setup de treinamento otimizado criado para {model_name}")
    logger.info(f"Configuração: {training_config}")

    return callbacks, optimizer, training_config


def get_recommended_hyperparameters(model_name: str) -> Dict[str, Any]:
    """Retorna hiperparâmetros recomendados para cada arquitetura."""

    recommendations = {
        "AASIST": {
            "batch_size": 32,
            "learning_rate": 0.0005,
            "epochs": 100,
            "validation_split": 0.2,
            "dropout_rate": 0.2,
            "l2_reg_strength": 0.0005
        },
        "RawGAT-ST": {
            "batch_size": 24,
            "learning_rate": 0.0008,
            "epochs": 80,
            "validation_split": 0.2,
            "dropout_rate": 0.2,
            "l2_reg_strength": 0.0005
        },
        "MultiscaleCNN": {
            "batch_size": 40,
            "learning_rate": 0.001,
            "epochs": 60,
            "validation_split": 0.2,
            "dropout_rate": 0.2,
            "l2_reg_strength": 0.0003
        },
        "SpectrogramTransformer": {
            "batch_size": 16,
            "learning_rate": 0.0003,
            "epochs": 120,
            "validation_split": 0.2,
            "dropout_rate": 0.1,
            "l2_reg_strength": 0.0001
        },
        "Conformer": {
            "batch_size": 20,
            "learning_rate": 0.0004,
            "epochs": 100,
            "validation_split": 0.2,
            "dropout_rate": 0.1,
            "l2_reg_strength": 0.0002
        },
        # Novos hiperparâmetros recomendados para Hybrid CNN-Transformer
        "Hybrid-CNN-Transformer": {
            "batch_size": 32,
            "learning_rate": 0.0005,
            "epochs": 100,
            "validation_split": 0.2,
            "dropout_rate": 0.2,
            "l2_reg_strength": 0.0003,
            "base_filters": 64,
            "num_residual_blocks": 3,
            "num_transformer_layers": 2,
            "attention_heads": 8
        },
        # Novos hiperparâmetros recomendados para RawNet2
        "RawNet2": {
            "batch_size": 24,
            "learning_rate": 0.0008,
            "epochs": 80,
            "validation_split": 0.2,
            "dropout_rate": 0.3,
            "l2_reg_strength": 0.0005,
            "conv_filters": [64, 128, 256],
            "gru_units": 128,
            "dense_units": 64
        }
    }

    return recommendations.get(model_name, {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 80,
        "validation_split": 0.2,
        "dropout_rate": 0.2,
        "l2_reg_strength": 0.0005
    })


# Exportar principais classes e funções
__all__ = [
    "OptimizedTrainingConfig",
    "create_optimized_training_setup",
    "get_recommended_hyperparameters"
]



def save_default_hyperparameters_json(
        model_name: str, output_dir: str, custom_params: Optional[Dict[str, Any]] = None) -> str:
    """Salva hiperparâmetros (recomendados ou customizados) no Banco de Dados.

    Mantém a assinatura para compatibilidade, mas `output_dir` é ignorado.
    """
    from app.core.db_setup import get_flask_app
    from app.domain.models.architecture_config import ArchitectureConfig
    from app.extensions import db

    # Carrega defaults
    params = get_recommended_hyperparameters(model_name)

    # Atualiza com customizados se houver
    if custom_params:
        params.update(custom_params)
        source = "custom_user_defined"
    else:
        source = "recommended_default"

    try:
        app = get_flask_app()
        with app.app_context():
            # Tenta encontrar existente
            arch_config = ArchitectureConfig.query.filter_by(
                architecture_name=model_name,
                variant_name="default"
            ).first()

            if not arch_config:
                arch_config = ArchitectureConfig(
                    architecture_name=model_name,
                    variant_name="default",
                    parameters=params,
                    description=f"Configuração {source} para {model_name}"
                )
                db.session.add(arch_config)
            else:
                arch_config.parameters = params
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(arch_config, "parameters")

            db.session.commit()
            logger.info(f"Hiperparâmetros para {model_name} salvos no banco de dados.")
            return "database"
    except Exception as e:
        logger.error(f"Erro ao salvar hiperparâmetros no banco: {e}")
        # Fallback para arquivo se banco falhar (opcional, mas solicitado para remover JSON)
        return "error_db"


def load_hyperparameters_json(
        model_name: str, search_dir: str) -> Dict[str, Any]:
    """Carrega hiperparâmetros do Banco de Dados.

    Mantém assinatura para compatibilidade, mas `search_dir` é ignorado.
    """
    from app.core.db_setup import get_flask_app
    from app.domain.models.architecture_config import ArchitectureConfig

    default = get_recommended_hyperparameters(model_name)

    try:
        app = get_flask_app()
        with app.app_context():
            arch_config = ArchitectureConfig.query.filter_by(
                architecture_name=model_name,
                variant_name="default"
            ).first()

            if arch_config and arch_config.parameters:
                logger.info(f"Hiperparâmetros carregados do banco para {model_name}")
                return arch_config.parameters
    except Exception as e:
        logger.error(f"Erro ao carregar hiperparâmetros do banco: {e}")

    logger.info(f"Usando hiperparâmetros padrão (hardcoded) para {model_name}")
    return default

