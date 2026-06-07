"""Módulo de Treinamento de Modelos

Este módulo implementa o treinador principal para modelos de detecção de deepfake.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from app.core.config.settings import TrainingConfig
from app.core.interfaces.audio import IModelTrainer
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.core.performance import optimize_tf_dataset
from app.core.training.secure_training_pipeline import (
    SecureTrainingConfig,
    SecureTrainingPipeline,
)

from .augmentation import AudioAugmenter
from .metrics import MetricsCalculator
from .optimization import OptimizerFactory

_save_logger = logging.getLogger(__name__)


def save_inference_keras(model: "tf.keras.Model", path) -> None:
    """Salva um artefato de INFERÊNCIA (.keras) SEM o estado do otimizador.

    No Keras 3, `include_optimizer=False` é IGNORADO para o formato `.keras`: o
    estado do Adam (2 momentos por peso, ~2× os pesos) é sempre serializado,
    deixando o arquivo ~3× maior que o necessário para inferência.

    Removemos o otimizador ANTES de salvar e o restauramos DEPOIS. O grafo e os
    pesos ficam idênticos — a saída do modelo NÃO muda (neutro em acurácia) —,
    apenas o estado de treino deixa de ser gravado. Ganho típico: 561 MB → 188 MB
    (MultiscaleCNN), com load proporcionalmente mais rápido.

    Obs.: reconstruir via from_config+set_weights foi descartado por alterar a
    saída (NaN em modelos com BatchNormalization/camadas custom).
    """
    path = str(path)
    saved_opt = getattr(model, "optimizer", None)
    try:
        try:
            model.optimizer = None
        except Exception as e:
            _save_logger.debug(f"Não foi possível remover o otimizador: {e}")
        model.save(path)
    finally:
        # Restaura o otimizador para não afetar usos posteriores (ex.: continuar
        # o treino, avaliar com o mesmo objeto de modelo).
        if saved_opt is not None:
            try:
                model.optimizer = saved_opt
            except Exception:
                pass


class ModelTrainer(IModelTrainer):
    """Implementação do treinador de modelos com prevenção de data leakage."""

    def __init__(
        self, config: TrainingConfig, use_mixed_precision: Optional[bool] = None
    ):
        """
        Args:
            config: configuração de treinamento
            use_mixed_precision: Sprint 3.2 — Se None (default), auto-detecta:
                habilita mixed_float16 se houver GPU com Compute Capability >= 7.0
                (Volta+, RTX 20xx+). Setar True/False para forçar.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_calculator = MetricsCalculator()
        self.optimizer_factory = OptimizerFactory()
        self.augmenter = AudioAugmenter(config.augmentation_config)

        # Sprint 3.2: Mixed precision (float16) auto-detect em GPU
        # 2× speedup + metade da VRAM em GPUs Tensor Core (CC >= 7.0).
        if use_mixed_precision is None:
            use_mixed_precision = self._should_enable_mixed_precision()

        if use_mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                self.logger.info(
                    "Mixed precision training habilitado (mixed_float16): "
                    "~2× speedup + ~50% menos VRAM"
                )
            except Exception as e:
                self.logger.warning(f"Mixed precision indisponível: {e}")
        else:
            try:
                tf.keras.mixed_precision.set_global_policy("float32")
                self.logger.info("Mixed precision desabilitado para este treino.")
            except Exception as e:
                self.logger.warning(f"Falha ao definir política float32: {e}")

        # Configurar pipeline seguro para prevenção de data leakage
        secure_config = SecureTrainingConfig(
            test_size=getattr(config, "test_size", 0.2),
            validation_size=getattr(config, "validation_split", 0.2),
            random_state=42,
            use_temporal_split=getattr(config, "use_temporal_split", True),
            scaler_type=getattr(config, "scaler_type", "standard"),
            save_scaler=True,
            validate_no_leakage=True,
        )
        self.secure_pipeline = SecureTrainingPipeline(secure_config)

    def train(
        self,
        model: tf.keras.Model,
        train_data: Tuple[np.ndarray, np.ndarray],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> ProcessingResult[Dict[str, Any]]:
        """Treina modelo com pipeline seguro para prevenção de data leakage."""
        try:
            self.logger.info("Iniciando treinamento seguro do modelo")

            X_train, y_train = train_data

            # Usar pipeline seguro para preparar dados se validation_data não
            # fornecida
            if validation_data is None:
                self.logger.info(
                    "Aplicando pipeline seguro para divisão e normalização dos dados"
                )

                # Preparar dados usando pipeline seguro
                preparation_result = self.secure_pipeline.prepare_data(
                    X_train, y_train, metadata
                )

                if preparation_result.status != ProcessingStatus.SUCCESS:
                    raise ValueError(
                        f"Erro na preparação segura dos dados: {preparation_result.errors}"
                    )

                prepared_data = preparation_result.data
                X_train = prepared_data["X_train"]
                X_val = prepared_data["X_val"]
                y_train = prepared_data["y_train"]
                y_val = prepared_data["y_val"]

                # Armazenar dados de teste para avaliação posterior
                self._test_data = (prepared_data["X_test"], prepared_data["y_test"])

                validation_data = (X_val, y_val)

                self.logger.info(
                    f"Dados preparados com segurança - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(self._test_data[0])}"
                )
            else:
                self.logger.warning(
                    "Dados de validação fornecidos externamente - pipeline seguro não aplicado"
                )

            # Configurar otimizador
            optimizer = self.optimizer_factory.create_optimizer(
                self.config.optimizer, learning_rate=self.config.learning_rate
            )

            # Compilar modelo — a loss é resolvida conforme a SAÍDA real do
            # modelo + o formato dos labels, evitando o rank mismatch
            # "target and output must have the same rank" (ex.: softmax de 2
            # unidades + labels esparsos com binary_crossentropy default).
            resolved_loss = self._resolve_loss(model, y_train)
            model.compile(
                optimizer=optimizer,
                loss=resolved_loss,
                metrics=self._resolve_metrics(),
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
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                train_dataset = train_dataset.batch(self.config.batch_size)
                steps_per_epoch = None
            train_dataset = optimize_tf_dataset(
                train_dataset, cache=False, prefetch=True
            )

            # Sprint 2.4: Mixup data augmentation (opt-in).
            # IMPORTANTE: Mixup produz soft labels, então é incompatível com
            # losses sparse (sparse_categorical_crossentropy). Quando habilitado,
            # converte y para one-hot e usa categorical_crossentropy automaticamente.
            if getattr(self.config, "use_mixup", False):
                try:
                    num_classes = int(self._infer_num_classes(y_train))
                    alpha = float(getattr(self.config, "mixup_alpha", 0.2))
                    train_dataset = self.augmenter.apply_mixup_to_dataset(
                        train_dataset, alpha=alpha, num_classes=num_classes
                    )
                    # Class weighting é incompatível com soft labels do mixup
                    # (Keras espera int classes em class_weight dict)
                    self._mixup_enabled = True
                    self.logger.info(
                        f"Mixup habilitado: α={alpha}, num_classes={num_classes}"
                    )
                except Exception as e:
                    self.logger.warning(f"Falha ao habilitar Mixup: {e}")
                    self._mixup_enabled = False
            else:
                self._mixup_enabled = False

            # Preparar dados de validação
            val_dataset = tf.data.Dataset.from_tensor_slices(validation_data)
            val_dataset = val_dataset.batch(self.config.batch_size)
            val_dataset = optimize_tf_dataset(val_dataset, cache=False, prefetch=True)

            # Class weighting automático para datasets desbalanceados
            # Desabilita quando Mixup está ativo (soft labels não são compatíveis
            # com class_weight dict de Keras)
            if self._mixup_enabled:
                class_weight = None
                self.logger.info(
                    "Class weighting pulado (incompatível com Mixup soft labels)"
                )
            else:
                class_weight = self._compute_class_weights(y_train)

            # Treinar modelo
            history = model.fit(
                train_dataset,
                epochs=self.config.epochs,
                validation_data=val_dataset,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                class_weight=class_weight,
                verbose=1,
            )

            # Calibração automática de temperatura (post-hoc)
            self._calibrated_temperature = self._auto_calibrate_temperature(
                model, validation_data
            )

            # Sprint 2.5: Calibra threshold de OOD detection no val set
            self._ood_threshold = self._compute_ood_threshold(
                model, validation_data, self._calibrated_temperature
            )

            # Sprint 4.5: Calibra threshold EER (Equal Error Rate) no val set
            # Habilita threshold adaptativo por modelo na inferência (em vez de 0.5)
            self._eer_threshold, self._eer_value = self._compute_eer_threshold(
                model, validation_data, self._calibrated_temperature
            )

            # Calcular métricas finais
            final_metrics = self._calculate_final_metrics(model, validation_data)

            result = {
                "history": history.history,
                "final_metrics": final_metrics,
                "model_summary": self._get_model_summary(model),
                "training_config": self.config.__dict__,
            }

            self.logger.info("Treinamento concluído com sucesso")
            return ProcessingResult(status=ProcessingStatus.SUCCESS, data=result)

        except Exception as e:
            self.logger.error(f"Erro durante treinamento: {str(e)}")
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def get_scaler(self):
        """Retorna o scaler treinado para uso em predições."""
        if hasattr(self, "secure_pipeline"):
            return self.secure_pipeline.get_scaler()
        else:
            self.logger.warning("Pipeline seguro não inicializado")
            return None

    def predict_with_scaler(self, model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
        """Faz predição aplicando o mesmo scaler usado no treinamento."""
        scaler = self.get_scaler()
        if scaler is None or scaler.scaler is None:
            self.logger.warning("Scaler não disponível - usando dados sem normalização")
            return model.predict(X)

        # Aplicar mesma normalização usada no treinamento
        X_scaled = scaler.transform_test(X)
        return model.predict(X_scaled)

    def save_training_artifacts(
        self, model: tf.keras.Model, save_dir: Union[str, Path]
    ) -> ProcessingResult[Dict[str, str]]:
        """Salva modelo e artefatos de treinamento (incluindo scaler)."""
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Salvar modelo no formato nativo Keras 3 (.keras) como artefato de
            # INFERÊNCIA — sem o estado do otimizador (~3× menor, load mais
            # rápido, saída idêntica). Ver save_inference_keras().
            model_path = save_dir / "model.keras"
            save_inference_keras(model, model_path)

            # Salvar scaler se disponível
            scaler_path = None
            scaler = self.get_scaler()
            if scaler is not None and scaler.scaler is not None:
                scaler_path = save_dir / "scaler.pkl"
                scaler.save_scaler(scaler_path)

            # Salvar configuração com input_contract para consistência train/inference
            config_path = save_dir / "training_config.json"
            import json

            # Construir input_contract a partir do modelo treinado
            input_contract = self._build_input_contract(model)

            config_dict = {
                "model_config": self.config.__dict__,
                "secure_pipeline_config": self.secure_pipeline.config.__dict__
                if hasattr(self, "secure_pipeline")
                else None,
                "training_timestamp": datetime.now().isoformat(),
                "input_contract": input_contract,
            }
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

            artifacts = {"model_path": str(model_path), "config_path": str(config_path)}

            if scaler_path:
                artifacts["scaler_path"] = str(scaler_path)

            # Sprint 3.4: ONNX export opcional
            if getattr(self.config, "export_onnx", False):
                onnx_paths = self._export_onnx_artifacts(model, save_dir)
                artifacts.update(onnx_paths)

            self.logger.info(f"Artefatos de treinamento salvos em: {save_dir}")
            return ProcessingResult(status=ProcessingStatus.SUCCESS, data=artifacts)

        except Exception as e:
            self.logger.error(f"Erro ao salvar artefatos: {str(e)}")
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def evaluate(
        self,
        model: tf.keras.Model,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> ProcessingResult[Dict[str, float]]:
        """Avalia modelo usando dados de teste seguros."""
        try:
            self.logger.info("Iniciando avaliação segura do modelo")

            # Usar dados de teste do pipeline seguro se disponíveis
            if test_data is None and hasattr(self, "_test_data"):
                X_test, y_test = self._test_data
                self.logger.info("Usando dados de teste do pipeline seguro")
            elif test_data is not None:
                X_test, y_test = test_data
                self.logger.warning("Usando dados de teste fornecidos externamente")
            else:
                raise ValueError(
                    "Nenhum dado de teste disponível. Execute o treinamento primeiro ou forneça test_data."
                )

            # Avaliação básica
            test_loss, *test_metrics = model.evaluate(
                X_test, y_test, batch_size=self.config.batch_size, verbose=0
            )

            # Predições para métricas detalhadas
            y_pred = model.predict(X_test, batch_size=self.config.batch_size)
            # Suporta saídas (N,1) sigmoid e (N,K) softmax
            y_pred_classes = (
                np.argmax(y_pred, axis=1)
                if (y_pred.ndim > 1 and y_pred.shape[-1] > 1)
                else (y_pred.ravel() > 0.5).astype(int)
            )

            # Calcular métricas detalhadas
            detailed_metrics = self.metrics_calculator.calculate_all_metrics(
                y_test, y_pred_classes, y_pred
            )

            # Combinar métricas
            metrics = {
                "test_loss": float(test_loss),
                **{
                    f"test_{metric}": float(value)
                    for metric, value in zip(self.config.metrics, test_metrics)
                },
                **detailed_metrics,
            }

            self.logger.info("Avaliação segura concluída com sucesso")
            return ProcessingResult(status=ProcessingStatus.SUCCESS, data=metrics)

        except Exception as e:
            self.logger.error(f"Erro durante avaliação: {str(e)}")
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def save_model(
        self, model: tf.keras.Model, save_path: Union[str, Path]
    ) -> ProcessingResult[str]:
        """Salva modelo treinado."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Salvar modelo como artefato de inferência (sem estado do otimizador;
            # ~3× menor e load mais rápido, saída idêntica). Ver save_inference_keras.
            save_inference_keras(model, save_path)

            # Salvar configuração de treinamento com input_contract
            config_path = save_path.parent / f"{save_path.stem}_config.json"
            import json

            config_data = {
                **self.config.__dict__,
                "input_contract": self._build_input_contract(model),
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2, default=str)

            self.logger.info(f"Modelo salvo em: {save_path}")
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS, data=str(save_path)
            )

        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {str(e)}")
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def load_model(
        self, model_path: Union[str, Path]
    ) -> ProcessingResult[tf.keras.Model]:
        """Carrega modelo salvo."""
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

            # Carregar modelo
            model = tf.keras.models.load_model(str(model_path))

            self.logger.info(f"Modelo carregado de: {model_path}")
            return ProcessingResult(status=ProcessingStatus.SUCCESS, data=model)

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def _prepare_callbacks(self, **kwargs) -> List[tf.keras.callbacks.Callback]:
        """Prepara callbacks para treinamento."""
        callbacks = []

        # Termina o treinamento imediatamente se NaN/Inf aparecer na loss
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())

        # Sprint 2.3: Stochastic Weight Averaging (opt-in via TrainingConfig)
        if getattr(self.config, "use_swa", False):
            try:
                from app.domain.models.training.swa_callback import SWACallback

                swa_cb = SWACallback(
                    start_epoch=getattr(self.config, "swa_start_epoch", -1),
                    swa_freq=getattr(self.config, "swa_freq", 1),
                    bn_update_data=kwargs.get("bn_update_data", None),
                    verbose=1,
                )
                callbacks.append(swa_cb)
                self._swa_callback = swa_cb  # acessível depois do treino
                self.logger.info("SWA habilitado")
            except Exception as e:
                self.logger.warning(f"Falha ao adicionar SWA callback: {e}")

        # Early stopping
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

        # Reduce learning rate
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1,
            )
        )

        # Model checkpoint
        if "checkpoint_path" in kwargs:
            callbacks.append(
                ModelCheckpoint(
                    filepath=kwargs["checkpoint_path"],
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                )
            )

        # TensorBoard
        if "tensorboard_dir" in kwargs:
            callbacks.append(
                TensorBoard(
                    log_dir=kwargs["tensorboard_dir"],
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                )
            )

        # CSV Logger
        if "csv_log_path" in kwargs:
            callbacks.append(CSVLogger(filename=kwargs["csv_log_path"], append=True))

        return callbacks

    @staticmethod
    def _should_enable_mixed_precision() -> bool:
        """Sprint 3.2: auto-detecta se mixed precision deve ser habilitado.

        Critério: existe ao menos uma GPU com Compute Capability >= 7.0
        (Volta+, ou seja: V100, T4, RTX 20xx, RTX 30xx, RTX 40xx, A100, H100).
        GPUs anteriores (Pascal/Maxwell) não têm Tensor Cores e mixed precision
        pode até reduzir performance ou causar overflow numérico.

        Retorna False se não houver GPU, ou se a detecção falhar.
        """
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return False
            # Inspeciona compute capability via details (TF 2.6+)
            for gpu in gpus:
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    cc = details.get("compute_capability")
                    if cc is not None:
                        major = (
                            cc[0]
                            if isinstance(cc, (list, tuple))
                            else int(str(cc).split(".")[0])
                        )
                        if major >= 7:
                            return True
                except Exception:
                    # Detalhes não disponíveis — assume seguro habilitar se há GPU
                    return True
            return False
        except Exception:
            return False

    def _resolve_metrics(self) -> List[Any]:
        """Métricas seguras para o `compile` (só as nativas do Keras).

        O default do TrainingConfig inclui 'f1' — que NÃO é métrica nativa do
        Keras (`Could not interpret metric identifier: f1`). Além disso,
        Precision/Recall/AUC como classes podem quebrar com saída softmax de 2
        unidades + labels esparsos (esperam binário). Como F1, EER, precision e
        recall já são calculados post-hoc pelo MetricsCalculator
        (`calculate_all_metrics`), no `compile` usamos APENAS 'accuracy', que é
        robusta para binário/multiclasse e labels esparsos/one-hot.
        """
        return ["accuracy"]

    def _resolve_loss(self, model: tf.keras.Model, y_train: np.ndarray):
        """Escolhe a loss compatível com a saída do modelo e o formato dos labels.

        O `loss_function` default do TrainingConfig é `binary_crossentropy`
        (assume sigmoid de 1 unidade). Mas o TrainingService instancia modelos
        com `num_classes` detectado (2 para binário) → saída softmax de 2
        unidades. Compilar com binary_crossentropy nesse caso quebra o fit com
        "target and output must have the same rank". Aqui auto-corrigimos:

        - saída 1 unidade  → binary_crossentropy (labels esparsos ou (N,1))
        - saída K>1 + labels esparsos (N,)   → sparse_categorical_crossentropy
        - saída K>1 + labels one-hot (N,K)   → categorical_crossentropy

        Respeita a loss configurada quando ela já é compatível.
        """
        configured = self.config.loss_function
        try:
            out_units = int(model.output_shape[-1])
        except Exception:
            return configured

        y = np.asarray(y_train)
        labels_one_hot = y.ndim > 1 and y.shape[-1] > 1

        if out_units == 1:
            chosen = "binary_crossentropy"
        elif labels_one_hot:
            chosen = "categorical_crossentropy"
        else:
            chosen = "sparse_categorical_crossentropy"

        # Se a loss configurada já é compatível, mantém (evita sobrescrever
        # escolhas legítimas como focal loss customizada via string).
        compatible = {
            1: {"binary_crossentropy", "bce", "mse", "mae"},
        }.get(out_units, (
            {"categorical_crossentropy", "kl_divergence"}
            if labels_one_hot
            else {"sparse_categorical_crossentropy"}
        ))
        if configured in compatible:
            return configured

        if chosen != configured:
            self.logger.warning(
                f"Loss '{configured}' incompatível com saída de {out_units} "
                f"unidade(s) + labels "
                f"{'one-hot' if labels_one_hot else 'esparsos'}; "
                f"usando '{chosen}'."
            )
        return chosen

    def _infer_num_classes(self, y: np.ndarray) -> int:
        """Inferência de num_classes a partir de y (suporta sparse e one-hot)."""
        y_arr = np.asarray(y)
        if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
            return int(y_arr.shape[-1])
        return max(int(np.unique(y_arr).size), 2)

    def _compute_class_weights(self, y_train: np.ndarray) -> Optional[Dict[int, float]]:
        """Calcula pesos por classe para compensar desbalanceamento.

        Usa `sklearn.utils.class_weight.compute_class_weight('balanced')`,
        equivalente a `n_samples / (n_classes * np.bincount(y))`.

        Returns:
            Dict {class_idx: weight} ou None se desabilitado/inválido.
        """
        if not getattr(self.config, "use_class_weighting", True):
            return None

        try:
            from sklearn.utils.class_weight import compute_class_weight

            # Suporta y one-hot (N, C) e categórico (N,) ou (N, 1)
            y_arr = np.asarray(y_train)
            if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
                y_for_weights = np.argmax(y_arr, axis=-1)
            else:
                y_for_weights = y_arr.ravel().astype(int)

            unique_classes = np.unique(y_for_weights)
            if len(unique_classes) < 2:
                self.logger.warning(
                    "Class weighting pulado: apenas 1 classe presente em y_train"
                )
                return None

            weights_array = compute_class_weight(
                "balanced", classes=unique_classes, y=y_for_weights
            )
            class_weight = {
                int(c): float(w) for c, w in zip(unique_classes, weights_array)
            }

            # Log: contagem por classe + pesos
            counts = {int(c): int((y_for_weights == c).sum()) for c in unique_classes}
            self.logger.info(
                f"Class weighting habilitado | counts={counts} | weights={class_weight}"
            )
            return class_weight

        except Exception as e:
            self.logger.warning(f"Erro ao calcular class weights: {e}")
            return None

    def _auto_calibrate_temperature(
        self, model: tf.keras.Model, validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        """Calibração post-hoc de temperatura via grid search no conjunto de validação.

        Implementa Temperature Scaling (Guo et al., ICML 2017): busca o T que
        minimiza NLL nas predições do val set. O valor é salvo no input_contract
        e aplicado automaticamente pelo Predictor na inferência.

        Returns:
            Temperatura calibrada (1.0 se desabilitado/falhar).
        """
        default_t = 1.0
        if not getattr(self.config, "auto_calibrate_temperature", True):
            return default_t

        try:
            X_val, y_val = validation_data
            min_samples = getattr(self.config, "calibration_min_samples", 50)
            if len(X_val) < min_samples:
                self.logger.info(
                    f"Calibração de temperatura pulada: val set tem "
                    f"{len(X_val)} amostras (mínimo {min_samples})"
                )
                return default_t

            # Converte y para índices de classe (compatível com calibrate())
            y_for_calib = np.asarray(y_val)
            if y_for_calib.ndim > 1 and y_for_calib.shape[-1] > 1:
                y_for_calib = np.argmax(y_for_calib, axis=-1)
            else:
                y_for_calib = y_for_calib.ravel().astype(int)

            # Importação tardia para evitar ciclo (predictor importa do trainer)
            from app.domain.services.detection.predictor import TemperatureScaler

            scaler = TemperatureScaler()
            scaler.calibrate(model, X_val, y_for_calib)
            temperature = float(scaler.temperature)
            self.logger.info(
                f"Temperatura calibrada: T={temperature:.3f} "
                f"({len(X_val)} amostras de val)"
            )
            return temperature

        except Exception as e:
            self.logger.warning(f"Erro na calibração de temperatura: {e}")
            return default_t

    def _compute_ood_threshold(
        self,
        model: tf.keras.Model,
        validation_data: Tuple[np.ndarray, np.ndarray],
        temperature: float = 1.0,
    ) -> Optional[float]:
        """Sprint 2.5: Calibra threshold de OOD detection no val set.

        Computa energy scores para todas as amostras de validação (que são
        consideradas in-distribution por construção) e usa o quantil
        (1 - ood_quantile) como threshold. Amostras com energy score abaixo
        desse threshold serão flagged como OOD na inferência.

        Args:
            model: modelo treinado
            validation_data: (X_val, y_val)
            temperature: T calibrado (para consistência com energia)

        Returns:
            Threshold (float) ou None se desabilitado/falhar.
        """
        if not getattr(self.config, "compute_ood_threshold", True):
            return None

        try:
            from app.domain.services.detection.predictor import (
                apply_temperature_scaling,
                compute_energy_score,
            )

            X_val, _ = validation_data
            predictions = model.predict(
                X_val, batch_size=self.config.batch_size, verbose=0
            )
            # Aplica mesma temperatura que será usada em inferência
            predictions = apply_temperature_scaling(predictions, temperature)
            energy_scores = compute_energy_score(predictions, temperature=temperature)

            # Threshold = quantil inferior. ood_quantile=0.95 → 5% das amostras
            # in-distribution com menores energy scores serão falsos positivos OOD.
            q = 1.0 - float(getattr(self.config, "ood_quantile", 0.95))
            threshold = float(np.quantile(energy_scores, q))
            self.logger.info(
                f"OOD threshold calibrado: {threshold:.4f} "
                f"(quantile {q:.2f} de {len(energy_scores)} amostras val | "
                f"range=[{energy_scores.min():.3f}, {energy_scores.max():.3f}])"
            )
            return threshold

        except Exception as e:
            self.logger.warning(f"Erro ao calibrar OOD threshold: {e}")
            return None

    def _calculate_final_metrics(
        self, model: tf.keras.Model, validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Calcula métricas finais do modelo."""
        X_val, y_val = validation_data

        # Predições
        y_pred = model.predict(X_val, batch_size=self.config.batch_size)
        # Suporta saídas (N,1) sigmoid e (N,K) softmax
        y_pred_classes = (
            np.argmax(y_pred, axis=1)
            if (y_pred.ndim > 1 and y_pred.shape[-1] > 1)
            else (y_pred.ravel() > 0.5).astype(int)
        )

        # Calcular métricas
        return self.metrics_calculator.calculate_all_metrics(
            y_val, y_pred_classes, y_pred
        )

    def _compute_eer_threshold(
        self,
        model: tf.keras.Model,
        validation_data: Tuple[np.ndarray, np.ndarray],
        temperature: float = 1.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Sprint 4.5: Calibra Equal Error Rate threshold no val set.

        EER é o ponto onde FPR == FNR — métrica padrão em anti-spoofing
        (ASVspoof). O threshold associado é frequentemente mais informativo
        que o 0.5 fixo, pois balanceia falsos positivos e falsos negativos.

        Args:
            model: modelo treinado
            validation_data: (X_val, y_val)
            temperature: T calibrado (Sprint 1.4) para consistência

        Returns:
            (eer_threshold, eer_value), ambos float ou None se falhar.
            - eer_threshold: score acima do qual classificar como fake
            - eer_value: valor de EER (taxa de erro no ponto FPR=FNR)
        """
        try:
            from app.domain.services.detection.predictor import (
                apply_temperature_scaling,
            )

            X_val, y_val = validation_data
            predictions = model.predict(
                X_val, batch_size=self.config.batch_size, verbose=0
            )
            predictions = apply_temperature_scaling(predictions, temperature)

            # Extrai score de probabilidade da classe "fake" (índice 1)
            if predictions.ndim > 1 and predictions.shape[-1] > 1:
                scores = predictions[:, 1]
            elif predictions.ndim > 1 and predictions.shape[-1] == 1:
                scores = predictions[:, 0]
            else:
                scores = predictions.ravel()

            # y_val pode ser sparse ou one-hot
            y_arr = np.asarray(y_val)
            if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
                y_true = np.argmax(y_arr, axis=-1)
            else:
                y_true = y_arr.ravel().astype(int)

            # Usa MetricsCalculator.calculate_eer (já existente)
            eer_value, eer_threshold = self.metrics_calculator.calculate_eer(
                y_true, scores
            )
            self.logger.info(
                f"EER threshold calibrado: T={eer_threshold:.4f}, EER={eer_value:.4f} "
                f"({len(scores)} amostras val) — alternativa ao threshold 0.5"
            )
            return float(eer_threshold), float(eer_value)

        except Exception as e:
            self.logger.warning(f"Erro ao calibrar EER threshold: {e}")
            return None, None

    def _export_onnx_artifacts(
        self,
        model: tf.keras.Model,
        save_dir: Path,
    ) -> Dict[str, str]:
        """Sprint 3.4: Export ONNX FP32 e INT8 (opcional).

        Degrada graciosamente se tf2onnx/onnxruntime não estão instalados —
        retorna dict vazio sem levantar exceção (NÃO bloqueia save do .keras).
        """
        artifacts: Dict[str, str] = {}
        try:
            from app.domain.models.inference.onnx_export import (
                export_to_onnx,
                is_onnx_available,
                quantize_int8,
            )

            if not is_onnx_available():
                self.logger.info(
                    "ONNX export pulado: tf2onnx/onnxruntime não instalados. "
                    "Instale com: pip install tf2onnx onnxruntime"
                )
                return artifacts

            onnx_path = save_dir / "model.onnx"
            result = export_to_onnx(model, onnx_path)
            if result is not None:
                artifacts["onnx_path"] = str(result)

                # INT8 quantization opcional
                if getattr(self.config, "export_onnx_int8", False):
                    int8_path = save_dir / "model_int8.onnx"
                    # Usa val set como calibração se disponível
                    calib_data = None
                    if hasattr(self, "_test_data"):
                        X_test, _ = self._test_data
                        # Pega até 100 amostras para calibração estática
                        calib_data = X_test[:100].astype(np.float32)
                    int8_result = quantize_int8(result, int8_path, calib_data)
                    if int8_result is not None:
                        artifacts["onnx_int8_path"] = str(int8_result)
        except Exception as e:
            self.logger.warning(f"ONNX export falhou (não-crítico): {e}")
        return artifacts

    def _build_input_contract(
        self, model: tf.keras.Model, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Constrói contrato de entrada para garantir consistência train/inference.

        O input_contract é salvo junto ao modelo e lido pelo FeaturePreparer
        na inferência para garantir que as mesmas features/formato sejam usados.
        """
        metadata = metadata or {}
        model_input_shape = list(model.input_shape[1:]) if model.input_shape else None

        # Determinar tipo de input a partir da arquitetura ou shape
        architecture = metadata.get("architecture", "")
        input_type = metadata.get("input_type", "features")
        input_format = metadata.get("input_format", "tabular")

        # Heurística: modelos com input (N, 1) provavelmente recebem áudio raw
        if (
            model_input_shape
            and len(model_input_shape) == 2
            and model_input_shape[-1] == 1
        ):
            input_type = "audio"
            input_format = "raw"
        # Modelos com input (H, W) ou (H, W, C) provavelmente usam spectrogram
        elif (
            model_input_shape
            and len(model_input_shape) in (2, 3)
            and (model_input_shape[-1] != 1 if len(model_input_shape) == 2 else True)
        ):
            if any(dim and dim > 10 for dim in model_input_shape[:2]):
                input_type = "features"
                input_format = "spectrogram"

        # Feature types se disponíveis
        feature_types = metadata.get("feature_types", None)
        if feature_types is None:
            feature_types_attr = getattr(model, "feature_types_used", None)
            if feature_types_attr:
                feature_types = list(feature_types_attr)

        contract = {
            "type": input_type,
            "format": input_format,
            "input_shape": model_input_shape,
            "architecture": architecture,
            "feature_types": feature_types,
            "sample_rate": metadata.get("sample_rate", 16000),
            "scaler_applied": self.get_scaler() is not None
            and self.get_scaler().scaler is not None,
            # Temperatura calibrada (Sprint 1.4) — aplicada na inferência pelo Predictor
            "temperature": float(getattr(self, "_calibrated_temperature", 1.0)),
        }
        # Sprint 2.5: OOD threshold (energy-based). None se desabilitado/falhou.
        ood_t = getattr(self, "_ood_threshold", None)
        if ood_t is not None:
            contract["ood_threshold"] = float(ood_t)

        # Sprint 4.5: EER threshold (Equal Error Rate) — alternativa adaptativa
        # ao threshold 0.5 fixo. Predictor pode usar via flag use_eer_threshold.
        eer_t = getattr(self, "_eer_threshold", None)
        eer_v = getattr(self, "_eer_value", None)
        if eer_t is not None:
            contract["eer_threshold"] = float(eer_t)
        if eer_v is not None:
            contract["eer_value"] = float(eer_v)
        return contract

    def predict_with_tta(
        self,
        model: tf.keras.Model,
        X: np.ndarray,
        n_augmentations: int = 5,
        noise_std: float = 0.005,
        shift_factor: float = 0.02,
        volume_range: Tuple = (0.95, 1.05),
    ) -> np.ndarray:
        """Test-Time Augmentation: run multiple augmented copies and average predictions.

        Typically improves accuracy by 1-3% without retraining.
        Uses 5 versions: original + noise + neg_noise + time_shift + volume_change.
        """
        predictions = []

        # Original prediction
        predictions.append(model.predict(X, batch_size=self.config.batch_size))

        if n_augmentations >= 2:
            # Positive noise
            X_noise = X + np.random.normal(0, noise_std, X.shape).astype(np.float32)
            predictions.append(
                model.predict(X_noise, batch_size=self.config.batch_size)
            )

        if n_augmentations >= 3:
            # Negative noise
            X_noise_neg = X - np.random.normal(0, noise_std, X.shape).astype(np.float32)
            predictions.append(
                model.predict(X_noise_neg, batch_size=self.config.batch_size)
            )

        if n_augmentations >= 4:
            # Time shift (small circular shift along first feature axis)
            shift_amount = (
                max(1, int(X.shape[1] * shift_factor)) if len(X.shape) > 1 else 0
            )
            if shift_amount > 0:
                X_shifted = np.roll(X, shift_amount, axis=1)
                predictions.append(
                    model.predict(X_shifted, batch_size=self.config.batch_size)
                )

        if n_augmentations >= 5:
            # Volume change
            vol_factor = np.random.uniform(volume_range[0], volume_range[1])
            X_vol = X * vol_factor
            predictions.append(model.predict(X_vol, batch_size=self.config.batch_size))

        # Average all predictions
        avg_prediction = np.mean(predictions, axis=0)
        self.logger.info(f"TTA applied with {len(predictions)} augmentations")
        return avg_prediction

    def _get_model_summary(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Retorna resumo do modelo."""
        return {
            "total_params": model.count_params(),
            "trainable_params": sum(
                [tf.keras.backend.count_params(w) for w in model.trainable_weights]
            ),
            "non_trainable_params": sum(
                [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
            ),
            "layers_count": len(model.layers),
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
        }
