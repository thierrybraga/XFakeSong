"""Anti-Overfitting Callbacks and Utilities"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Any

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger, Callback
)
import numpy as np

logger = logging.getLogger(__name__)


class GradientClippingCallback(Callback):
    """Callback para clipagem de gradientes durante o treinamento."""

    def __init__(self, clip_norm: float = 1.0, clip_value: float = None):
        super().__init__()
        self.clip_norm = clip_norm
        self.clip_value = clip_value or clip_norm

    def on_train_batch_begin(self, batch, logs=None):
        # Aplicar clipagem de gradientes
        if hasattr(self.model.optimizer, 'clipnorm'):
            self.model.optimizer.clipnorm = self.clip_norm


class AdvancedEarlyStopping(EarlyStopping):
    """Early stopping avançado com funcionalidades extras."""

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0,
                 mode='auto', baseline=None, restore_best_weights=False, **kwargs):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience,
                         verbose=verbose, mode=mode, baseline=baseline,
                         restore_best_weights=restore_best_weights, **kwargs)
        self.overfitting_threshold = 0.1
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        # Armazenar histórico de losses
        if logs:
            train_loss = logs.get('loss')
            val_loss = logs.get('val_loss')
            if train_loss is not None:
                self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)

        # Chamar early stopping padrão
        super().on_epoch_end(epoch, logs)

        # Detectar overfitting
        if len(self.train_losses) >= 5 and len(self.val_losses) >= 5:
            recent_train = np.mean(self.train_losses[-5:])
            recent_val = np.mean(self.val_losses[-5:])
            if recent_val - recent_train > self.overfitting_threshold:
                logger.warning(
                    f"Overfitting detectado: train_loss={
                        recent_train:.4f}, val_loss={
                        recent_val:.4f}")


class LossMonitoringCallback(Callback):
    """Callback para monitoramento avançado de loss."""

    def __init__(self, patience: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.patience = patience
        self.train_losses = []
        self.val_losses = []
        self.overfitting_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            train_loss = logs.get('loss')
            val_loss = logs.get('val_loss')

            if train_loss is not None:
                self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)

            # Detectar divergência entre train e val loss
            if len(self.train_losses) >= 3 and len(self.val_losses) >= 3:
                train_trend = np.mean(self.train_losses[-3:])
                val_trend = np.mean(self.val_losses[-3:])

                if val_trend > train_trend * 1.2:  # Val loss 20% maior que train loss
                    self.overfitting_count += 1
                    if self.overfitting_count >= self.patience:
                        logger.warning(
                            "Possível overfitting detectado - considere parar o treinamento")
                else:
                    self.overfitting_count = 0


class OverfittingDetectionCallback(Callback):
    """Callback especializado para detecção de overfitting."""

    def __init__(self, threshold: float = 0.1, patience: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.patience = patience
        self.overfitting_epochs = 0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            train_loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)

            # Calcular diferença entre val_loss e train_loss
            loss_diff = val_loss - train_loss

            if loss_diff > self.threshold:
                self.overfitting_epochs += 1
                logger.info(
                    f"Epoch {
                        epoch +
                        1}: Possível overfitting detectado (diff: {
                        loss_diff:.4f})")

                if self.overfitting_epochs >= self.patience:
                    logger.warning(
                        f"Overfitting persistente por {
                            self.overfitting_epochs} epochs")
                    self.model.stop_training = True
            else:
                self.overfitting_epochs = 0

            # Atualizar melhor val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss


class ValidationLossMonitor(Callback):
    """Monitor personalizado para perda de validação."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss is None:
            return

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            logger.info(
                f"Epoch {
                    epoch +
                    1}: Validation loss improved to {
                    current_loss:.6f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.warning(
                    f"Epoch {
                        epoch +
                        1}: Early stopping triggered. No improvement for {
                        self.patience} epochs.")
                self.model.stop_training = True


def create_anti_overfitting_callbacks(
    model_name: str,
    save_dir: str = "./models/checkpoints",
    patience: int = 15,
    min_delta: float = 0.001,
    factor: float = 0.5,
    lr_patience: int = 8,
    min_lr: float = 1e-7,
    monitor: str = 'val_loss',
    mode: str = 'min',
    verbose: int = 1
) -> List[Callback]:
    """Cria uma lista de callbacks para prevenir overfitting.

    Args:
        model_name: Nome do modelo para salvar checkpoints
        save_dir: Diretório para salvar checkpoints
        patience: Paciência para early stopping
        min_delta: Mínima mudança para considerar melhoria
        factor: Fator de redução da learning rate
        lr_patience: Paciência para redução da learning rate
        min_lr: Learning rate mínima
        monitor: Métrica a ser monitorada
        mode: Modo de monitoramento ('min' ou 'max')
        verbose: Nível de verbosidade

    Returns:
        Lista de callbacks configurados
    """

    # Criar diretório se não existir
    os.makedirs(save_dir, exist_ok=True)

    callbacks = []

    # Early Stopping - mais agressivo para prevenir overfitting
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        verbose=verbose,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)

    # Model Checkpoint - salvar melhor modelo
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.h5")
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=False,
        verbose=verbose
    )
    callbacks.append(model_checkpoint)

    # Reduce Learning Rate on Plateau
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=lr_patience,
        min_lr=min_lr,
        mode=mode,
        verbose=verbose,
        cooldown=3
    )
    callbacks.append(reduce_lr)

    # Gradient Clipping
    gradient_clipping = GradientClippingCallback(clip_norm=1.0)
    callbacks.append(gradient_clipping)

    # Validation Loss Monitor personalizado
    val_monitor = ValidationLossMonitor(
        patience=patience + 5,  # Mais paciência que early stopping
        min_delta=min_delta
    )
    callbacks.append(val_monitor)

    # TensorBoard para monitoramento
    tensorboard_dir = os.path.join(save_dir, "tensorboard", model_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    tensorboard = TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch'
    )
    callbacks.append(tensorboard)

    # CSV Logger para histórico de treinamento
    csv_path = os.path.join(save_dir, f"{model_name}_training_log.csv")
    csv_logger = CSVLogger(csv_path, append=True)
    callbacks.append(csv_logger)

    logger.info(
        f"Criados {
            len(callbacks)} callbacks anti-overfitting para {model_name}")
    return callbacks


def create_data_augmentation_pipeline(
    noise_factor: float = 0.01,
    time_shift_factor: float = 0.1,
    speed_factor: float = 0.1,
    pitch_factor: float = 0.1
) -> tf.data.Dataset:
    """Cria pipeline de data augmentation para áudio.

    Args:
        noise_factor: Fator de ruído a ser adicionado
        time_shift_factor: Fator de deslocamento temporal
        speed_factor: Fator de mudança de velocidade
        pitch_factor: Fator de mudança de pitch

    Returns:
        Função de augmentation para usar com tf.data.Dataset.map
    """

    def augment_audio(audio_features, label):
        """Aplica augmentations aleatórias ao áudio."""

        # Adicionar ruído gaussiano
        if tf.random.uniform([]) < 0.5:
            noise = tf.random.normal(
                shape=tf.shape(audio_features),
                mean=0.0,
                stddev=noise_factor,
                dtype=tf.float32
            )
            audio_features = audio_features + noise

        # Time shifting (deslocamento temporal)
        if tf.random.uniform([]) < 0.3:
            shift_amount = tf.random.uniform(
                [],
                -int(tf.shape(audio_features)[0] * time_shift_factor),
                int(tf.shape(audio_features)[0] * time_shift_factor),
                dtype=tf.int32
            )
            audio_features = tf.roll(audio_features, shift_amount, axis=0)

        # Normalização adaptativa
        audio_features = tf.nn.l2_normalize(audio_features, axis=-1)

        # Clipping para evitar valores extremos
        audio_features = tf.clip_by_value(audio_features, -3.0, 3.0)

        return audio_features, label

    return augment_audio


def apply_mixup_augmentation(
    dataset: tf.data.Dataset,
    alpha: float = 0.2,
    num_classes: int = 2
) -> tf.data.Dataset:
    """Aplica augmentation Mixup ao dataset.

    Args:
        dataset: Dataset original
        alpha: Parâmetro alpha para distribuição Beta
        num_classes: Número de classes

    Returns:
        Dataset com Mixup aplicado
    """

    def mixup(batch_x, batch_y):
        """Aplica Mixup a um batch."""
        batch_size = tf.shape(batch_x)[0]

        # Gerar lambda da distribuição Beta
        lam = tf.random.gamma([batch_size], alpha, alpha)
        lam = tf.maximum(lam, 1 - lam)

        # Embaralhar índices
        indices = tf.random.shuffle(tf.range(batch_size))

        # Aplicar Mixup
        mixed_x = lam[:, None, None] * batch_x + \
            (1 - lam[:, None, None]) * tf.gather(batch_x, indices)

        # Para labels categóricas
        if len(batch_y.shape) == 1:
            batch_y = tf.one_hot(batch_y, num_classes)
            shuffled_y = tf.gather(
                tf.one_hot(
                    tf.gather(
                        tf.argmax(
                            batch_y,
                            axis=1),
                        indices),
                    num_classes),
                indices)
        else:
            shuffled_y = tf.gather(batch_y, indices)

        mixed_y = lam[:, None] * batch_y + (1 - lam[:, None]) * shuffled_y

        return mixed_x, mixed_y

    return dataset.batch(32).map(
        mixup, num_parallel_calls=tf.data.AUTOTUNE).unbatch()


class OverfittingDetector:
    """Detector de overfitting baseado em métricas de treinamento."""

    def __init__(self, threshold: float = 0.1, window_size: int = 5):
        self.threshold = threshold
        self.window_size = window_size
        self.train_losses = []
        self.val_losses = []

    def update(self, train_loss: float, val_loss: float):
        """Atualiza as métricas de loss."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Manter apenas a janela mais recente
        if len(self.train_losses) > self.window_size:
            self.train_losses.pop(0)
            self.val_losses.pop(0)

    def is_overfitting(self) -> bool:
        """Detecta se o modelo está em overfitting."""
        if len(self.train_losses) < self.window_size:
            return False

        # Calcular diferença média entre train e validation loss
        avg_train_loss = np.mean(self.train_losses)
        avg_val_loss = np.mean(self.val_losses)

        gap = avg_val_loss - avg_train_loss

        # Detectar overfitting se a diferença for muito grande
        return gap > self.threshold

    def get_overfitting_score(self) -> float:
        """Retorna um score de overfitting (0-1)."""
        if len(self.train_losses) < 2:
            return 0.0

        avg_train_loss = np.mean(self.train_losses)
        avg_val_loss = np.mean(self.val_losses)

        if avg_train_loss == 0:
            return 1.0

        gap = max(0, avg_val_loss - avg_train_loss)
        return min(1.0, gap / avg_train_loss)
