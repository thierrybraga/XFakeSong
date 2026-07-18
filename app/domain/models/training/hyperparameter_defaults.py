"""Hiperparâmetros recomendados por arquitetura, persistidos no banco.

Extraído de ``optimized_training_config.py`` (consolidação — o resto daquele
módulo, ``OptimizedTrainingConfig``/``create_optimized_training_setup``, não
tinha nenhum chamador real e duplicava callbacks/augmentation já usados pelo
``trainer.py``; ver ``app/domain/models/training/trainer.py``,
``app/domain/models/training/augmentation.py``,
``app/domain/models/training/spec_augment.py`` e
``app/domain/models/training/rawboost.py``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_recommended_hyperparameters(model_name: str) -> Dict[str, Any]:
    """Retorna hiperparâmetros recomendados para cada arquitetura."""

    # Hiperparametros alinhados com TCC (Secao 6.1 + Tabela 10):
    #   - epochs      = 100 (early stopping patience=10)
    #   - batch_size  = 32
    #   - l2          = 0.0001 (lambda do TCC)
    #   - dropout     = conforme Tabela 10 do TCC
    #   - lr          = 0.001 (Adam — Secao 5.2.3)
    #   - validation_split removido: splits vem dos arquivos (70/15/15)
    recommendations = {
        "AASIST": {
            "batch_size": 16,
            "learning_rate": 0.0008,
            "epochs": 100,
            "dropout_rate": 0.2,
            "l2_reg_strength": 0.0001,
            "attention_heads": 12,
            "hidden_units": 512,
        },
        "RawGAT-ST": {
            "batch_size": 24,
            "learning_rate": 0.0008,
            "epochs": 100,
            "dropout_rate": 0.2,
            "l2_reg_strength": 0.0001,
        },
        "MultiscaleCNN": {
            "batch_size": 64,
            "learning_rate": 0.002,
            "epochs": 100,
            "dropout_rate": 0.5,
            "l2_reg_strength": 0.0005,
            "hidden_units": "128/256",
        },
        "SpectrogramTransformer": {
            "batch_size": 16,
            "learning_rate": 0.0003,
            "epochs": 100,
            "dropout_rate": 0.1,
            "l2_reg_strength": 0.0001,
        },
        "Conformer": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 100,
            "dropout_rate": 0.3,
            "l2_reg_strength": 0.0001,
            "attention_heads": 8,
            "hidden_units": 256,
        },
        "EfficientNet-LSTM": {
            "batch_size": 32,
            "learning_rate": 0.0005,
            "epochs": 100,
            "dropout_rate": 0.4,
            "l2_reg_strength": 0.0002,
            "hidden_units": "128/64",
        },
        "Hybrid-CNN-Transformer": {
            "batch_size": 32,
            "learning_rate": 0.0005,
            "epochs": 100,
            "dropout_rate": 0.2,
            "l2_reg_strength": 0.0001,
            "base_filters": 64,
            "num_residual_blocks": 3,
            "num_transformer_layers": 2,
            "attention_heads": 8,
        },
        "RawNet2": {
            "batch_size": 24,
            "learning_rate": 0.0008,
            "epochs": 100,
            "dropout_rate": 0.3,
            "l2_reg_strength": 0.0001,
            "conv_filters": [64, 128, 256],
            "gru_units": 128,
            "dense_units": 64,
        },
        # Ensemble Adaptativo (TCC Eq. 27-28) — menos epocas pois parte de modelos pre-treinados
        "ensemble_adaptive": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 50,
            "dropout_rate": 0.3,
            "l2_reg_strength": 0.0001,
        },
    }

    return recommendations.get(model_name, {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        "dropout_rate": 0.3,
        "l2_reg_strength": 0.0001,
    })


def save_default_hyperparameters_json(
        model_name: str, output_dir: str, custom_params: Optional[Dict[str, Any]] = None) -> str:
    """Salva hiperparâmetros (recomendados ou customizados) no Banco de Dados.

    Mantém a assinatura para compatibilidade, mas `output_dir` é ignorado.
    """
    from sqlalchemy.orm.attributes import flag_modified

    from app.core.db.session import SessionLocal
    from app.domain.models.architecture_config import ArchitectureConfig

    # Carrega defaults
    params = get_recommended_hyperparameters(model_name)

    # Atualiza com customizados se houver
    if custom_params:
        params.update(custom_params)
        source = "custom_user_defined"
    else:
        source = "recommended_default"

    try:
        db = SessionLocal()
        try:
            arch_config = (
                db.query(ArchitectureConfig)
                .filter_by(architecture_name=model_name, variant_name="default")
                .first()
            )

            if not arch_config:
                arch_config = ArchitectureConfig(
                    architecture_name=model_name,
                    variant_name="default",
                    parameters=params,
                    description=f"Configuração {source} para {model_name}",
                )
                db.add(arch_config)
            else:
                arch_config.parameters = params
                flag_modified(arch_config, "parameters")

            db.commit()
            logger.info(
                f"Hiperparâmetros para {model_name} salvos no banco de dados."
            )
            return "database"
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Erro ao salvar hiperparâmetros no banco: {e}")
        # Fallback para arquivo se banco falhar (opcional, mas solicitado para remover JSON)
        return "error_db"


def load_hyperparameters_json(
        model_name: str, search_dir: str) -> Dict[str, Any]:
    """Carrega hiperparâmetros do Banco de Dados.

    Mantém assinatura para compatibilidade, mas `search_dir` é ignorado.
    """
    from app.core.db.session import SessionLocal
    from app.domain.models.architecture_config import ArchitectureConfig

    default = get_recommended_hyperparameters(model_name)

    try:
        db = SessionLocal()
        try:
            arch_config = (
                db.query(ArchitectureConfig)
                .filter_by(architecture_name=model_name, variant_name="default")
                .first()
            )
            if arch_config and arch_config.parameters:
                logger.info(
                    f"Hiperparâmetros carregados do banco para {model_name}"
                )
                return arch_config.parameters
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Erro ao carregar hiperparâmetros do banco: {e}")

    logger.info(f"Usando hiperparâmetros padrão (hardcoded) para {model_name}")
    return default


__all__ = [
    "get_recommended_hyperparameters",
    "save_default_hyperparameters_json",
    "load_hyperparameters_json",
]
