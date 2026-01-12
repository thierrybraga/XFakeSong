import logging
import numpy as np
import importlib
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import json

from app.core.interfaces.services import (
    ITrainingService, ProcessingResult, ProcessingStatus, ModelMetadata
)
from app.domain.models.training.trainer import ModelTrainer
from app.core.config.settings import TrainingConfig
from app.domain.models.architectures.registry import get_architecture_info

logger = logging.getLogger(__name__)


class TrainingService(ITrainingService):
    """
    Serviço responsável por gerenciar processos de treinamento.
    Integra ModelTrainer com o restante do sistema.
    """

    def __init__(self, models_dir: str = "app/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_model(self, architecture: str, dataset_path: str,
                    config: Dict[str, Any]) -> ProcessingResult[ModelMetadata]:
        """
        Inicia o treinamento de um modelo.
        """
        try:
            logger.info(
                f"Iniciando serviço de treinamento para {architecture}")

            # 1. Validar Arquitetura
            arch_info = get_architecture_info(architecture)
            if not arch_info:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Arquitetura '{architecture}' não encontrada."]
                )

            # 2. Carregar Dados
            # Suporte inicial para arquivos .npz (padrão numpy)
            data_path = Path(dataset_path)
            if not data_path.exists():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Dataset não encontrado: {dataset_path}"]
                )

            try:
                if data_path.suffix == '.npz':
                    data = np.load(data_path)
                    if 'X_train' not in data or 'y_train' not in data:
                        return ProcessingResult(
                            status=ProcessingStatus.ERROR,
                            errors=[
                                "Arquivo .npz deve conter chaves "
                                "'X_train' e 'y_train'"
                            ]
                        )

                    X_train = data['X_train']
                    y_train = data['y_train']

                    validation_data = None
                    if 'X_val' in data and 'y_val' in data:
                        validation_data = (data['X_val'], data['y_val'])

                else:
                    # TODO: Integrar com SegmentedFeatureLoader para CSVs
                    return ProcessingResult(
                        status=ProcessingStatus.ERROR,
                        errors=[
                            f"Formato não suportado: {data_path.suffix}. "
                            "Use .npz"
                        ]
                    )
            except Exception as e:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Erro ao carregar dataset: {str(e)}"]
                )

            # 3. Instanciar Modelo
            try:
                module = importlib.import_module(arch_info.module_path)
                create_model_fn = getattr(module, arch_info.function_name)

                # Mesclar parâmetros padrão com os fornecidos
                model_params = arch_info.default_params.copy()
                if 'parameters' in config:
                    model_params.update(config['parameters'])

                # Determinar input shape
                # Assumindo (batch, time, feats) ou (batch, feats)
                input_shape = X_train.shape[1:]

                model = create_model_fn(
                    input_shape=input_shape, **model_params)
            except Exception as e:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Erro ao instanciar modelo: {str(e)}"]
                )

            # 4. Configurar e Executar Treinamento
            # Converter dicionário para objeto TrainingConfig
            train_conf_dict = config.copy()
            # Remover campos que não pertencem ao config direto se necessário
            train_conf_dict.pop('parameters', None)
            train_conf_dict.pop('model_name', None)

            training_config = TrainingConfig(**train_conf_dict)
            trainer = ModelTrainer(training_config)

            # kwargs para callbacks podem ser passados via config
            train_result = trainer.train(
                model,
                (X_train, y_train),
                validation_data=validation_data
            )

            if train_result.status != ProcessingStatus.SUCCESS:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=train_result.errors
                )

            # 5. Salvar Modelo e Metadados
            model_name = config.get(
                'model_name',
                f"{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            save_path = self.models_dir / f"{model_name}.h5"

            save_result = trainer.save_model(model, save_path)
            if save_result.status != ProcessingStatus.SUCCESS:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=save_result.errors
                )

            # Salvar config do modelo para carregamento futuro
            config_path = self.models_dir / f"{model_name}_config.json"
            model_metadata = {
                "architecture": architecture,
                "input_shape": list(input_shape),
                # Hardcoded por enquanto, mas pode vir do arch_info
                "model_type": "tensorflow",
                "created_at": str(datetime.now()),
                "metrics": train_result.data
            }
            with open(config_path, 'w') as f:
                json.dump(model_metadata, f, indent=4)

            # Extract metrics
            metrics = train_result.data or {}

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=ModelMetadata(
                    name=model_name,
                    version="1.0.0",
                    architecture=architecture,
                    created_at=datetime.now(),
                    metrics=metrics,
                    file_path=Path(save_path),
                    accuracy=metrics.get("accuracy", 0.0),
                    precision=metrics.get("precision", 0.0),
                    recall=metrics.get("recall", 0.0),
                    f1_score=metrics.get("f1", 0.0),
                    training_dataset=str(dataset_path),
                    file_size=(
                        save_path.stat().st_size if save_path.exists() else 0
                    )
                )
            )

        except Exception as e:
            logger.error(
                f"Erro não tratado no TrainingService: {e}", exc_info=True)
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro interno: {str(e)}"]
            )

    def evaluate_model(
        self, model_name: str, test_dataset: str
    ) -> ProcessingResult[Dict[str, float]]:
        # Implementação futura
        return ProcessingResult(
            status=ProcessingStatus.ERROR, errors=["Not implemented"])

    def fine_tune_model(
        self, base_model: str, dataset_name: str, config: Dict[str, Any]
    ) -> ProcessingResult[ModelMetadata]:
        # Implementação futura
        return ProcessingResult(
            status=ProcessingStatus.ERROR, errors=["Not implemented"])

    def get_training_progress(
            self, training_id: str) -> ProcessingResult[Dict[str, Any]]:
        # Implementação futura
        return ProcessingResult(
            status=ProcessingStatus.ERROR, errors=["Not implemented"])

