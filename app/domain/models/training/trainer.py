"""Módulo de Treinamento de Modelos

Este módulo implementa o treinador principal para modelos de detecção de deepfake.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

from app.core.interfaces.audio import IModelTrainer
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.core.config.settings import TrainingConfig
from app.core.training.secure_training_pipeline import SecureTrainingPipeline, SecureTrainingConfig
from .metrics import MetricsCalculator
from .optimization import OptimizerFactory
from .augmentation import AudioAugmenter


class ModelTrainer(IModelTrainer):
    """Implementação do treinador de modelos com prevenção de data leakage."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_calculator = MetricsCalculator()
        self.optimizer_factory = OptimizerFactory()
        self.augmenter = AudioAugmenter(config.augmentation_config)

        # Configurar pipeline seguro para prevenção de data leakage
        secure_config = SecureTrainingConfig(
            test_size=getattr(config, 'test_size', 0.2),
            validation_size=getattr(config, 'validation_split', 0.2),
            random_state=42,
            use_temporal_split=getattr(config, 'use_temporal_split', True),
            scaler_type=getattr(config, 'scaler_type', 'standard'),
            save_scaler=True,
            validate_no_leakage=True
        )
        self.secure_pipeline = SecureTrainingPipeline(secure_config)

    def train(
        self,
        model: tf.keras.Model,
        train_data: Tuple[np.ndarray, np.ndarray],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> ProcessingResult[Dict[str, Any]]:
        """Treina modelo com pipeline seguro para prevenção de data leakage."""
        try:
            self.logger.info("Iniciando treinamento seguro do modelo")

            X_train, y_train = train_data

            # Usar pipeline seguro para preparar dados se validation_data não
            # fornecida
            if validation_data is None:
                self.logger.info(
                    "Aplicando pipeline seguro para divisão e normalização dos dados")

                # Preparar dados usando pipeline seguro
                preparation_result = self.secure_pipeline.prepare_data(
                    X_train, y_train, metadata)

                if not preparation_result.success:
                    raise ValueError(
                        f"Erro na preparação segura dos dados: {
                            preparation_result.error}")

                prepared_data = preparation_result.data
                X_train = prepared_data["X_train"]
                X_val = prepared_data["X_val"]
                y_train = prepared_data["y_train"]
                y_val = prepared_data["y_val"]

                # Armazenar dados de teste para avaliação posterior
                self._test_data = (
                    prepared_data["X_test"],
                    prepared_data["y_test"])

                validation_data = (X_val, y_val)

                self.logger.info(
                    f"Dados preparados com segurança - Train: {
                        len(X_train)}, Val: {
                        len(X_val)}, Test: {
                        len(
                            self._test_data[0])}")
            else:
                self.logger.warning(
                    "Dados de validação fornecidos externamente - pipeline seguro não aplicado")

            # Configurar otimizador
            optimizer = self.optimizer_factory.create_optimizer(
                self.config.optimizer,
                learning_rate=self.config.learning_rate
            )

            # Compilar modelo
            model.compile(
                optimizer=optimizer,
                loss=self.config.loss_function,
                metrics=self.config.metrics
            )

            # Preparar callbacks
            callbacks = self._prepare_callbacks(**kwargs)

            # Aplicar data augmentation se habilitado
            if self.config.use_augmentation:
                train_dataset = self.augmenter.create_augmented_dataset(
                    X_train, y_train, self.config.batch_size
                )
                steps_per_epoch = len(X_train) // self.config.batch_size
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (X_train, y_train))
                train_dataset = train_dataset.batch(self.config.batch_size)
                steps_per_epoch = None

            # Preparar dados de validação
            val_dataset = tf.data.Dataset.from_tensor_slices(validation_data)
            val_dataset = val_dataset.batch(self.config.batch_size)

            # Treinar modelo
            history = model.fit(
                train_dataset,
                epochs=self.config.epochs,
                validation_data=val_dataset,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                verbose=1
            )

            # Calcular métricas finais
            final_metrics = self._calculate_final_metrics(
                model, validation_data)

            result = {
                "history": history.history,
                "final_metrics": final_metrics,
                "model_summary": self._get_model_summary(model),
                "training_config": self.config.__dict__
            }

            self.logger.info("Treinamento concluído com sucesso")
            return ProcessingResult(
                success=True,
                data=result,
                status=ProcessingStatus.COMPLETED
            )

        except Exception as e:
            self.logger.error(f"Erro durante treinamento: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                status=ProcessingStatus.FAILED
            )

    def get_scaler(self):
        """Retorna o scaler treinado para uso em predições."""
        if hasattr(self, 'secure_pipeline'):
            return self.secure_pipeline.get_scaler()
        else:
            self.logger.warning("Pipeline seguro não inicializado")
            return None

    def predict_with_scaler(self, model: tf.keras.Model,
                            X: np.ndarray) -> np.ndarray:
        """Faz predição aplicando o mesmo scaler usado no treinamento."""
        scaler = self.get_scaler()
        if scaler is None or scaler.scaler is None:
            self.logger.warning(
                "Scaler não disponível - usando dados sem normalização")
            return model.predict(X)

        # Aplicar mesma normalização usada no treinamento
        X_scaled = scaler.transform_test(X)
        return model.predict(X_scaled)

    def save_training_artifacts(self, model: tf.keras.Model,
                                save_dir: Union[str, Path]) -> ProcessingResult[Dict[str, str]]:
        """Salva modelo e artefatos de treinamento (incluindo scaler)."""
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Salvar modelo
            model_path = save_dir / "model.h5"
            model.save(str(model_path))

            # Salvar scaler se disponível
            scaler_path = None
            scaler = self.get_scaler()
            if scaler is not None and scaler.scaler is not None:
                scaler_path = save_dir / "scaler.pkl"
                scaler.save_scaler(scaler_path)

            # Salvar configuração
            config_path = save_dir / "training_config.json"
            import json
            config_dict = {
                "model_config": self.config.__dict__,
                "secure_pipeline_config": self.secure_pipeline.config.__dict__ if hasattr(self, 'secure_pipeline') else None,
                "training_timestamp": datetime.now().isoformat()
            }
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)

            artifacts = {
                "model_path": str(model_path),
                "config_path": str(config_path)
            }

            if scaler_path:
                artifacts["scaler_path"] = str(scaler_path)

            self.logger.info(f"Artefatos de treinamento salvos em: {save_dir}")
            return ProcessingResult(
                success=True,
                data=artifacts,
                status=ProcessingStatus.COMPLETED
            )

        except Exception as e:
            self.logger.error(f"Erro ao salvar artefatos: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                status=ProcessingStatus.FAILED
            )

    def evaluate(
        self,
        model: tf.keras.Model,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> ProcessingResult[Dict[str, float]]:
        """Avalia modelo usando dados de teste seguros."""
        try:
            self.logger.info("Iniciando avaliação segura do modelo")

            # Usar dados de teste do pipeline seguro se disponíveis
            if test_data is None and hasattr(self, '_test_data'):
                X_test, y_test = self._test_data
                self.logger.info("Usando dados de teste do pipeline seguro")
            elif test_data is not None:
                X_test, y_test = test_data
                self.logger.warning(
                    "Usando dados de teste fornecidos externamente")
            else:
                raise ValueError(
                    "Nenhum dado de teste disponível. Execute o treinamento primeiro ou forneça test_data.")

            # Avaliação básica
            test_loss, *test_metrics = model.evaluate(
                X_test, y_test,
                batch_size=self.config.batch_size,
                verbose=0
            )

            # Predições para métricas detalhadas
            y_pred = model.predict(X_test, batch_size=self.config.batch_size)
            y_pred_classes = np.argmax(
                y_pred, axis=1) if y_pred.shape[1] > 1 else (
                y_pred > 0.5).astype(int)

            # Calcular métricas detalhadas
            detailed_metrics = self.metrics_calculator.calculate_all_metrics(
                y_test, y_pred_classes, y_pred
            )

            # Combinar métricas
            metrics = {
                "test_loss": float(test_loss),
                **{f"test_{metric}": float(value) for metric, value in zip(self.config.metrics, test_metrics)},
                **detailed_metrics
            }

            self.logger.info("Avaliação segura concluída com sucesso")
            return ProcessingResult(
                success=True,
                data=metrics,
                status=ProcessingStatus.COMPLETED
            )

        except Exception as e:
            self.logger.error(f"Erro durante avaliação: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                status=ProcessingStatus.FAILED
            )

    def save_model(
        self,
        model: tf.keras.Model,
        save_path: Union[str, Path]
    ) -> ProcessingResult[str]:
        """Salva modelo treinado."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Salvar modelo
            model.save(str(save_path))

            # Salvar configuração de treinamento
            config_path = save_path.parent / f"{save_path.stem}_config.json"
            import json
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)

            self.logger.info(f"Modelo salvo em: {save_path}")
            return ProcessingResult(
                success=True,
                data=str(save_path),
                status=ProcessingStatus.COMPLETED
            )

        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                status=ProcessingStatus.FAILED
            )

    def load_model(
        self,
        model_path: Union[str, Path]
    ) -> ProcessingResult[tf.keras.Model]:
        """Carrega modelo salvo."""
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

            # Carregar modelo
            model = tf.keras.models.load_model(str(model_path))

            self.logger.info(f"Modelo carregado de: {model_path}")
            return ProcessingResult(
                success=True,
                data=model,
                status=ProcessingStatus.COMPLETED
            )

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                status=ProcessingStatus.FAILED
            )

    def _prepare_callbacks(
            self, **kwargs) -> List[tf.keras.callbacks.Callback]:
        """Prepara callbacks para treinamento."""
        callbacks = []

        # Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))

        # Reduce learning rate
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ))

        # Model checkpoint
        if 'checkpoint_path' in kwargs:
            callbacks.append(ModelCheckpoint(
                filepath=kwargs['checkpoint_path'],
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ))

        # TensorBoard
        if 'tensorboard_dir' in kwargs:
            callbacks.append(TensorBoard(
                log_dir=kwargs['tensorboard_dir'],
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ))

        # CSV Logger
        if 'csv_log_path' in kwargs:
            callbacks.append(CSVLogger(
                filename=kwargs['csv_log_path'],
                append=True
            ))

        return callbacks

    def _calculate_final_metrics(
        self,
        model: tf.keras.Model,
        validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Calcula métricas finais do modelo."""
        X_val, y_val = validation_data

        # Predições
        y_pred = model.predict(X_val, batch_size=self.config.batch_size)
        y_pred_classes = np.argmax(
            y_pred, axis=1) if y_pred.shape[1] > 1 else (
            y_pred > 0.5).astype(int)

        # Calcular métricas
        return self.metrics_calculator.calculate_all_metrics(
            y_val, y_pred_classes, y_pred
        )

    def _get_model_summary(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Retorna resumo do modelo."""
        return {
            "total_params": model.count_params(),
            "trainable_params": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            "non_trainable_params": sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
            "layers_count": len(model.layers),
            "input_shape": model.input_shape,
            "output_shape": model.output_shape
        }
