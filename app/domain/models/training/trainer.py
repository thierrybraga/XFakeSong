"""Módulo de Treinamento de Modelos

Este módulo implementa o treinador principal para modelos de detecção de deepfake.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    BackupAndRestore,
    CSVLogger,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

from app.core.config.settings import TrainingConfig
from app.core.contracts.audio import IModelTrainer
from app.core.contracts.base import ProcessingResult, ProcessingStatus
from app.core.performance import optimize_tf_dataset

from .artifacts import build_input_contract, save_inference_keras
from .augmentation import AudioAugmenter
from .calibration import (
    build_calibration_set,
    calibrate_temperature,
    compute_eer_threshold,
    compute_ood_threshold,
    final_metrics,
)
from .callbacks import (
    CollapseAbort,
    EpochProgressLogger,
    PersistentEpochHistory,
    ResumableModelCheckpoint,
    ValidationEER,
)
from .metrics import MetricsCalculator
from .optimization import OptimizerFactory
from .secure_training_pipeline import SecureTrainingConfig, SecureTrainingPipeline


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

    def _array_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        batch_size: int,
        purpose: str,
        shuffle: bool = False,
    ) -> tuple[tf.data.Dataset, bool]:
        """Cria dataset sem materializar arrays grandes como Tensor constante.

        AJUSTE 2026-07-14 (`shuffle`): o Keras IGNORA `shuffle=True` do fit()
        quando `x` é um tf.data.Dataset — o caminho sem augmentation treinava
        com ordem de batches FIXA em todas as épocas. No benchmark isso é
        agravado pelo protocolo AWGN: o treino é [bloco limpo | bloco ruidoso]
        concatenados, então cada época via primeiro só amostras limpas e
        depois só ruidosas. `shuffle=True` (usar apenas no treino) embaralha
        por época nos dois caminhos (tensor_slices e generator).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        bytes_total = int(X.nbytes + y.nbytes)
        large_threshold = 256 * 1024 * 1024

        if bytes_total <= large_threshold:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            if shuffle:
                dataset = dataset.shuffle(
                    buffer_size=len(y), seed=42, reshuffle_each_iteration=True
                )
            # prefetch(AUTOTUNE): sem isso, a GPU fica ociosa esperando o
            # próximo lote em vez de sobrepor preparo de dado (CPU) com
            # computo (GPU) — o padrão clássico de baixa utilização de GPU
            # (~30% observado no Conformer) mesmo com o modelo saudável.
            return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), False

        self.logger.info(
            "Dataset %s grande (%.1f MB) — usando generator em batches para "
            "reduzir cópias de RAM.",
            purpose,
            bytes_total / (1024 * 1024),
        )

        x_shape = (None,) + tuple(X.shape[1:])
        y_shape = (None,) + tuple(y.shape[1:])
        x_dtype = tf.as_dtype(X.dtype)
        y_dtype = tf.as_dtype(y.dtype)
        rng = np.random.default_rng(42)

        def batch_generator():
            n = len(y)
            # Permutação nova a cada passagem (época): o estado do rng
            # persiste no closure entre reinvocações do generator (repeat()).
            order = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                yield X[idx], y[idx]

        n_batches = int(np.ceil(len(y) / max(1, batch_size)))
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            output_signature=(
                tf.TensorSpec(shape=x_shape, dtype=x_dtype),
                tf.TensorSpec(shape=y_shape, dtype=y_dtype),
            ),
        )
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(n_batches))
        # Idem ao caminho from_tensor_slices: sem prefetch, o generator
        # Python (single-threaded, GIL) monta cada lote de forma síncrona e
        # bloqueia a GPU entre lotes. Com AUTOTUNE, o próximo lote é montado
        # numa thread em segundo plano enquanto a GPU processa o atual —
        # este é justamente o caminho usado pelos datasets grandes
        # (>256 MB: RawNet2/AASIST/RawGAT-ST/Conformer/etc. com a cópia
        # AWGN), onde o ganho de sobreposição CPU/GPU é maior.
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset, True

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

            # ── Compile-respect ─────────────────────────────────────────────
            # Cada arquitetura compila a si mesma com loss/otimizador corretos
            # (ex.: AASIST = AM-Softmax logits + loss from_logits=True + AdamW
            # clipnorm; Conformer/CCT/AST = WarmupCosineDecay; WavLM/HuBERT =
            # LR baixo p/ fine-tune SSL). Recompilar aqui DESCARTAVA tudo isso e
            # — pior — aplicava a string "sparse_categorical_crossentropy"
            # (from_logits=False) sobre logits crus do AASIST, quebrando o
            # treino. Agora: modelo já compilado é RESPEITADO; só ajustamos o
            # LR quando o chamador o definiu explicitamente (lr_is_explicit).
            already_compiled = getattr(model, "optimizer", None) is not None
            if already_compiled:
                self.logger.info(
                    "Modelo já compilado pela arquitetura — preservando "
                    "loss/otimizador originais."
                )
                if getattr(self.config, "lr_is_explicit", False):
                    try:
                        model.optimizer.learning_rate = self.config.learning_rate
                        self.logger.info(
                            f"LR explícito aplicado: {self.config.learning_rate}"
                        )
                    except Exception as e:
                        # LR pode ser um schedule (WarmupCosineDecay) — nesse
                        # caso o schedule da arquitetura prevalece.
                        self.logger.warning(
                            f"LR explícito ignorado (schedule da arquitetura "
                            f"prevalece): {e}"
                        )
            else:
                optimizer = self.optimizer_factory.create_optimizer(
                    self.config.optimizer,
                    learning_rate=self.config.learning_rate,
                )
                # A loss é resolvida conforme a SAÍDA real do modelo (inclusive
                # ativação linear → from_logits=True) + o formato dos labels.
                resolved_loss = self._resolve_loss(model, y_train)
                model.compile(
                    optimizer=optimizer,
                    loss=resolved_loss,
                    metrics=self._resolve_metrics(),
                )

            # Preparar callbacks
            callbacks = self._prepare_callbacks(**kwargs)

            # `val_eer` só existe nos logs se este callback estiver ativo, e o
            # ModelCheckpoint pode estar configurado para monitorá-lo
            # (config.checkpoint_monitor). Registrar apenas quando pedido evita
            # pagar uma inferência extra por época nos runs que usam val_loss.
            if str(getattr(self.config, "checkpoint_monitor", "")) == "val_eer":
                if validation_data is not None:
                    # INSERE NA FRENTE, não no fim. O Keras percorre os
                    # callbacks em ORDEM dentro de `on_epoch_end`, passando o
                    # MESMO dicionário `logs` a todos. Anexado ao fim, o
                    # ValidationEER publicaria `val_eer` depois de o
                    # ModelCheckpoint e o PersistentEpochHistory já terem lido
                    # o dicionário — e ambos veriam a métrica ausente.
                    #
                    # Foi o que o smoke `_smoke_eer2` (2026-08-17) mostrou:
                    # com o encanamento do monitor já corrigido, o Keras
                    # avisou "Can save best model only with val_eer available"
                    # nas duas épocas, NENHUM `best.json` foi gravado e o
                    # histórico saiu sem a coluna. Um retreino de 27 h
                    # terminaria sem artefato selecionado.
                    callbacks.insert(
                        0,
                        ValidationEER(
                            validation_data=validation_data,
                            label=getattr(self.config, "progress_label", "")
                            or "training",
                            # Mesmo lote do treino: se cabe treinar com ele,
                            # cabe inferir com ele.
                            batch_size=int(
                                getattr(self.config, "batch_size", 32) or 32
                            ),
                        ),
                    )
                else:
                    # Falha ALTO: seguir com o monitor apontando para uma
                    # métrica que ninguém publica faria o ModelCheckpoint nunca
                    # salvar — 27 h de treino sem artefato.
                    raise ValueError(
                        "checkpoint_monitor='val_eer' exige validation_data "
                        "para o callback ValidationEER; sem ela nenhum "
                        "checkpoint seria salvo."
                    )

            # Aplicar data augmentation se habilitado
            if self.config.use_augmentation:
                train_dataset = self.augmenter.create_augmented_dataset(
                    X_train, y_train, self.config.batch_size
                )
                # O dataset aumentado é FINITO (~2N amostras). Com
                # steps_per_epoch fixo e SEM repeat(), o iterator esgotava na
                # ~2ª época e o Keras interrompia o treino ("ran out of data").
                # repeat() + steps_per_epoch garante épocas completas sempre.
                train_dataset = train_dataset.repeat()
                steps_per_epoch = max(
                    1, len(X_train) // self.config.batch_size
                )
            else:
                train_dataset, train_streaming = self._array_dataset(
                    X_train,
                    y_train,
                    batch_size=self.config.batch_size,
                    purpose="train",
                    shuffle=True,
                )
                if train_streaming:
                    train_dataset = train_dataset.repeat()
                    steps_per_epoch = int(
                        np.ceil(len(y_train) / max(1, self.config.batch_size))
                    )
                else:
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
            X_val, y_val = validation_data
            val_dataset, _val_streaming = self._array_dataset(
                X_val,
                y_val,
                batch_size=self.config.batch_size,
                purpose="validation",
            )
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

            verbose = int(getattr(self.config, "verbose", 1))

            # Treinar modelo
            history = model.fit(
                train_dataset,
                epochs=self.config.epochs,
                validation_data=val_dataset,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                class_weight=class_weight,
                verbose=verbose,
            )

            # Conjunto de calibração: por padrão inclui cópias com ruído (AWGN
            # nos SNRs do teste) para que temperatura/threshold reflitam o ponto
            # de operação sob ruído (não só em áudio limpo).
            calibration_data = self._build_calibration_set(validation_data)

            # Calibração automática de temperatura (post-hoc)
            self._calibrated_temperature = self._auto_calibrate_temperature(
                model, calibration_data
            )

            # Sprint 2.5: Calibra threshold de OOD detection no val set
            self._ood_threshold = self._compute_ood_threshold(
                model, calibration_data, self._calibrated_temperature
            )

            # Sprint 4.5: Calibra threshold EER (Equal Error Rate) no val set
            # Habilita threshold adaptativo por modelo na inferência (em vez de 0.5)
            self._eer_threshold, self._eer_value = self._compute_eer_threshold(
                model, calibration_data, self._calibrated_temperature
            )

            # Calcular métricas finais
            final_metrics = self._calculate_final_metrics(model, validation_data)

            # `history.history` cobre só as épocas DESTA execução; numa
            # retomada isso truncaria a série (RawNet2 saiu com 17 de 100 no
            # clean_benchmark_15k). O callback persistente devolve o treino
            # inteiro. Só substitui se for pelo menos tão completo quanto.
            full_history = None
            history_cb = getattr(self, "_history_callback", None)
            if history_cb is not None:
                merged = history_cb.merged()
                longest = max((len(v) for v in merged.values()), default=0)
                current = max(
                    (len(v) for v in (history.history or {}).values()), default=0
                )
                if longest >= current:
                    full_history = merged

            result = {
                "history": full_history or history.history,
                "final_metrics": final_metrics,
                "model_summary": self._get_model_summary(model),
                "training_config": self.config.__dict__,
            }

            # Um treino abortado por colapso NÃO pode passar por treino curto
            # normal: sem isso o artefato registraria só "menos épocas".
            collapse_cb = getattr(self, "_collapse_callback", None)
            if collapse_cb is not None and collapse_cb.triggered:
                result["collapsed"] = True
                result["collapse_reason"] = collapse_cb.reason
                self.logger.warning(
                    "Treinamento ABORTADO por colapso: %s", collapse_cb.reason
                )
                return ProcessingResult(status=ProcessingStatus.SUCCESS, data=result)

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
            y_pred = model.predict(
                X_test, batch_size=self.config.batch_size, verbose=0
            )
            from app.domain.services.detection.predictor import (
                normalize_logits_to_probs,
            )
            y_pred = normalize_logits_to_probs(y_pred)
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
            from app.utils.file_storage import atomic_write_text

            atomic_write_text(
                config_path,
                json.dumps(config_data, indent=2, ensure_ascii=False, default=str),
            )

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

        progress_interval = int(getattr(self.config, "progress_log_interval", 0) or 0)
        if progress_interval > 0:
            callbacks.append(
                EpochProgressLogger(
                    interval=progress_interval,
                    label=getattr(self.config, "progress_label", "") or "training",
                )
            )

        # Sprint 2.3: Stochastic Weight Averaging (opt-in via TrainingConfig)
        if getattr(self.config, "use_swa", False):
            try:
                from app.domain.models.training.swa_callback import SWACallback

                swa_cb = SWACallback(
                    start_epoch=getattr(self.config, "swa_start_epoch", -1),
                    swa_freq=getattr(self.config, "swa_freq", 1),
                    bn_update_data=kwargs.get("bn_update_data", None),
                    verbose=int(getattr(self.config, "verbose", 1)),
                )
                callbacks.append(swa_cb)
                self._swa_callback = swa_cb  # acessível depois do treino
                self.logger.info("SWA habilitado")
            except Exception as e:
                self.logger.warning(f"Falha ao adicionar SWA callback: {e}")

        # Histórico resistente a retomadas. Ancorado no diretório do
        # checkpoint, NÃO no `backup_dir` — este último é apagado ao fim do
        # treino (delete_checkpoint=True) e levaria o histórico junto.
        history_anchor = kwargs.get("checkpoint_path") or kwargs.get("backup_dir")
        if history_anchor:
            anchor = Path(str(history_anchor))
            history_dir = anchor.parent if anchor.suffix else anchor
            history_cb = PersistentEpochHistory(
                history_dir / "epoch_history.jsonl",
                label=getattr(self.config, "progress_label", "") or "training",
            )
            callbacks.append(history_cb)
            self._history_callback = history_cb

        # Guarda de colapso — independente do early stopping (ver docstring de
        # CollapseAbort: uma coisa é parar um modelo que ainda melhora, outra é
        # abortar um que virou palpite constante e não volta).
        if getattr(self.config, "abort_on_collapse", True):
            collapse_cb = CollapseAbort(
                patience=int(getattr(self.config, "collapse_patience", 15)),
                nan_patience=int(getattr(self.config, "collapse_nan_patience", 3)),
                label=getattr(self.config, "progress_label", "") or "training",
            )
            callbacks.append(collapse_cb)
            self._collapse_callback = collapse_cb

        # Early stopping
        if getattr(self.config, "early_stopping", True):
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=int(getattr(self.config, "verbose", 1)),
                )
            )

        # Reduce learning rate. Arquiteturas com LearningRateSchedule próprio
        # não aceitam setar optimizer.learning_rate em runtime.
        if getattr(self.config, "reduce_lr_on_plateau", True):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=self.config.reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=int(getattr(self.config, "verbose", 1)),
                )
            )

        # Model checkpoint
        if "checkpoint_path" in kwargs:
            # AJUSTE 2026-07-14: quando o caminho termina em `.weights.h5`,
            # salva SÓ os pesos — o save de modelo completo serializava grafo
            # + estado do otimizador (~3× os pesos; ~1 GB por melhoria de
            # época no AST) e dependia da desserialização de camadas custom.
            # A restauração já usa load_weights, que aceita ambos os formatos.
            ckpt_path = str(kwargs["checkpoint_path"])
            from app.utils.file_storage import break_hardlink

            break_hardlink(Path(ckpt_path))
            # Monitor CONFIGURÁVEL, com `val_loss` como padrão.
            #
            # O padrão preserva a reprodutibilidade dos artefatos publicados,
            # todos selecionados por menor perda de validação. Mas o critério
            # tem um descompasso conhecido com a avaliação (ver `ValidationEER`):
            # seleciona por calibração, avalia por ordenação. Para runs novos
            # de anti-spoofing, `val_eer` alinha os dois — e exige que o
            # callback `ValidationEER` esteja ativo para publicar a métrica.
            monitor = str(getattr(self.config, "checkpoint_monitor", "")
                          or "val_loss")
            modo = "min" if monitor in ("val_loss", "val_eer") else "max"
            callbacks.append(
                ResumableModelCheckpoint(
                    filepath=ckpt_path,
                    monitor=monitor,
                    mode=modo,
                    save_best_only=True,
                    save_weights_only=ckpt_path.endswith(".weights.h5"),
                    verbose=int(getattr(self.config, "verbose", 1)),
                )
            )

        # Recuperacao de falhas de infraestrutura (reinicio do host/Docker).
        # Diferentemente do melhor checkpoint, o backup preserva tambem o
        # estado do otimizador e a epoca concluida, permitindo que fit()
        # retome sem transformar a continuacao em um novo experimento.
        #
        # AJUSTE 2026-08-04 (queda de energia): double_checkpoint=True. O
        # save escreve os pesos POR CIMA do backup unico; uma queda no meio
        # dessa gravacao deixa um HDF5 truncado e sem fallback — perdendo o
        # treino inteiro, nao so a epoca corrente. Com a opcao ligada o Keras
        # mantem o estado anterior em `.bkp` e cai nele quando o atual falha
        # ao carregar. Custa o dobro de disco no diretorio de backup.
        if "backup_dir" in kwargs:
            from app.utils.file_storage import break_hardlinks_in_tree

            break_hardlinks_in_tree(Path(kwargs["backup_dir"]))
            callbacks.append(
                BackupAndRestore(
                    backup_dir=str(kwargs["backup_dir"]),
                    save_freq="epoch",
                    double_checkpoint=True,
                    delete_checkpoint=True,
                )
            )

        # TensorBoard
        if "tensorboard_dir" in kwargs:
            from app.utils.file_storage import break_hardlinks_in_tree

            break_hardlinks_in_tree(Path(kwargs["tensorboard_dir"]))
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
            from app.utils.file_storage import break_hardlink

            break_hardlink(Path(kwargs["csv_log_path"]))
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

        # Saída LINEAR (sem sigmoid/softmax) emite LOGITS CRUS — ex.: AASIST
        # com AMSoftmaxLayer. As strings de loss do Keras assumem
        # from_logits=False e fariam log/normalização de valores negativos
        # (gradiente sem sentido). Detecta e usa objetos com from_logits=True.
        from_logits = False
        try:
            last_act = getattr(model.layers[-1], "activation", None)
            from_logits = last_act is None or last_act is tf.keras.activations.linear
        except Exception:
            pass

        if from_logits:
            if out_units == 1:
                chosen = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            elif labels_one_hot:
                chosen = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            else:
                chosen = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                )
            self.logger.info(
                "Saída linear detectada (logits crus) — loss com from_logits=True."
            )
            return chosen

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

    def _build_calibration_set(self, validation_data):
        """Val limpo + cópias com AWGN; ver ``calibration.build_calibration_set``."""
        return build_calibration_set(validation_data, self.config)

    def _auto_calibrate_temperature(
        self, model: tf.keras.Model, validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        """Temperature scaling no val; ver ``calibration.calibrate_temperature``."""
        return calibrate_temperature(model, validation_data, self.config)

    def _compute_ood_threshold(
        self,
        model: tf.keras.Model,
        validation_data: Tuple[np.ndarray, np.ndarray],
        temperature: float = 1.0,
    ) -> Optional[float]:
        """Limiar de OOD por energia; ver ``calibration.compute_ood_threshold``."""
        return compute_ood_threshold(model, validation_data, self.config, temperature)

    def _calculate_final_metrics(
        self, model: tf.keras.Model, validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Métricas finais no val; ver ``calibration.final_metrics``."""
        return final_metrics(
            model, validation_data, self.config, self.metrics_calculator
        )

    def _compute_eer_threshold(
        self,
        model: tf.keras.Model,
        validation_data: Tuple[np.ndarray, np.ndarray],
        temperature: float = 1.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """(limiar, valor) do EER no val; ver ``calibration.compute_eer_threshold``."""
        return compute_eer_threshold(
            model, validation_data, self.config, self.metrics_calculator, temperature
        )

    def _build_input_contract(
        self, model: tf.keras.Model, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Contrato de entrada com a calibração deste treino; ver
        ``artifacts.build_input_contract``."""
        return build_input_contract(
            model,
            metadata,
            scaler_applied=self.get_scaler() is not None
            and self.get_scaler().scaler is not None,
            temperature=getattr(self, "_calibrated_temperature", 1.0),
            ood_threshold=getattr(self, "_ood_threshold", None),
            eer_threshold=getattr(self, "_eer_threshold", None),
            eer_value=getattr(self, "_eer_value", None),
        )

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
