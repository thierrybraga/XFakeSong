"""Knowledge Distillation Pipeline (Sprint 5.1).

Implementa knowledge distillation (Hinton et al., "Distilling the Knowledge in
a Neural Network", NIPS 2015) para transferir conhecimento de modelos grandes
(teacher, ex: Ensemble, WavLM) para modelos leves (student, ex: Sonic Sleuth).

Vantagens:
- Modelos 5–10× menores com 90–95% da accuracy original
- Inferência muito mais rápida para deploy edge/mobile
- Combinável com ONNX export + INT8 quantization (Sprint 3.4)

Loss de distillation:
    L = α * CE(student_logits, hard_labels)              # supervisão direta
      + (1 - α) * T² * KL(student/T, teacher/T)          # soft targets do teacher

O termo T² compensa o fato de que gradientes do KL escalam com 1/T².

Uso típico:
    from app.domain.models.training.knowledge_distillation import (
        DistillationConfig, KnowledgeDistillationTrainer
    )

    config = DistillationConfig(temperature=4.0, alpha=0.3, epochs=50)
    distiller = KnowledgeDistillationTrainer(teacher_model, student_model, config)
    history = distiller.train(X_train, y_train, validation_data=(X_val, y_val))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuração de Knowledge Distillation."""
    # Temperatura para soft targets — valores típicos: 2-10. Maior = mais "soft".
    temperature: float = 4.0
    # Peso da loss supervisionada (vs soft targets): α em [0, 1].
    # Hinton recomenda α=0.1 (mais peso no teacher). Valor maior se teacher é fraco.
    alpha: float = 0.3
    # Hyperparams de treinamento
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = 'adamw'
    weight_decay: float = 1e-4
    # Early stopping no val_distillation_loss
    early_stopping_patience: int = 10
    # Logging
    verbose: int = 1


class _DistillationLoss(tf.keras.losses.Loss):
    """Composite loss = α * CE(hard) + (1-α) * T² * KL(soft).

    Espera y_true contendo (hard_labels, teacher_soft_targets) concatenados
    via DistillationDataAdapter; o student produz logits sem softmax.
    """

    def __init__(self, alpha: float, temperature: float, num_classes: int,
                 from_logits: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.num_classes = num_classes
        self.from_logits = from_logits

    def call(self, y_true_combined, y_pred_student):
        """
        Args:
            y_true_combined: tensor com [hard_label_onehot | teacher_soft_targets]
                shape: (batch, 2K) — primeiras K cols são labels one-hot,
                últimas K são teacher_soft_targets (softmax do teacher com T=1).
            y_pred_student: logits do student (batch, K) se from_logits=True,
                ou probabilities se from_logits=False.
        """
        K = self.num_classes
        hard_labels = y_true_combined[:, :K]
        teacher_soft = y_true_combined[:, K:]

        # Student outputs em formato softmax
        if self.from_logits:
            student_soft = tf.nn.softmax(y_pred_student / self.temperature, axis=-1)
            student_hard = tf.nn.softmax(y_pred_student, axis=-1)
        else:
            # Se já é probabilidade, simulamos temperature via log
            eps = 1e-7
            logits_approx = tf.math.log(tf.clip_by_value(y_pred_student, eps, 1 - eps))
            student_soft = tf.nn.softmax(logits_approx / self.temperature, axis=-1)
            student_hard = y_pred_student

        # Teacher já vem em softmax — reescala com temperatura
        eps = 1e-7
        teacher_soft_clipped = tf.clip_by_value(teacher_soft, eps, 1 - eps)
        teacher_logits_approx = tf.math.log(teacher_soft_clipped)
        teacher_soft_T = tf.nn.softmax(teacher_logits_approx / self.temperature, axis=-1)

        # Hard loss: CE entre hard_labels e student (sem temperatura)
        ce_hard = tf.keras.losses.categorical_crossentropy(
            hard_labels, student_hard, from_logits=False
        )

        # Soft loss: KL(student_soft || teacher_soft_T)
        # KL = Σ teacher * log(teacher / student)
        teacher_soft_T_safe = tf.clip_by_value(teacher_soft_T, eps, 1.0)
        student_soft_safe = tf.clip_by_value(student_soft, eps, 1.0)
        kl = tf.reduce_sum(
            teacher_soft_T_safe * tf.math.log(teacher_soft_T_safe / student_soft_safe),
            axis=-1,
        )

        # Combine: T² compensa gradient scaling
        loss = self.alpha * ce_hard + (1.0 - self.alpha) * (self.temperature ** 2) * kl
        return loss


class KnowledgeDistillationTrainer:
    """Treinador para distillation teacher → student.

    Args:
        teacher_model: tf.keras.Model pré-treinado (frozen durante distillation)
        student_model: tf.keras.Model não-compilado a ser treinado
        config: DistillationConfig
    """

    def __init__(
        self,
        teacher_model: tf.keras.Model,
        student_model: tf.keras.Model,
        config: DistillationConfig,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Congela teacher (não treina, só fornece soft targets)
        self.teacher.trainable = False

    def _precompute_teacher_targets(
        self,
        X: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Computa soft targets do teacher 1× para todo o dataset.

        Muito mais eficiente que recomputar a cada epoch. Retorna shape (N, K)
        de probabilidades (já em softmax sem temperatura — temperatura é
        aplicada dentro do DistillationLoss).
        """
        self.logger.info(
            f"Pré-computando soft targets do teacher ({len(X)} amostras)..."
        )
        teacher_outputs = self.teacher.predict(X, batch_size=batch_size, verbose=0)
        # Garante shape (N, K) — se sigmoid (N, 1), converte para 2-class
        if teacher_outputs.ndim == 1:
            teacher_outputs = teacher_outputs.reshape(-1, 1)
        if teacher_outputs.shape[-1] == 1:
            # Sigmoid binário: converte para softmax 2-class
            p_fake = teacher_outputs[:, 0]
            teacher_outputs = np.stack([1 - p_fake, p_fake], axis=-1)
        # Normaliza (caso teacher saída não some 1)
        teacher_outputs = teacher_outputs / np.clip(
            teacher_outputs.sum(axis=-1, keepdims=True), 1e-7, None
        )
        return teacher_outputs.astype(np.float32)

    def _build_distillation_targets(
        self,
        y: np.ndarray,
        teacher_soft: np.ndarray,
    ) -> np.ndarray:
        """Constrói y_combined = [hard_onehot | teacher_soft] (N, 2K)."""
        K = teacher_soft.shape[-1]
        y_arr = np.asarray(y)
        if y_arr.ndim > 1 and y_arr.shape[-1] == K:
            hard_onehot = y_arr.astype(np.float32)
        elif y_arr.ndim > 1 and y_arr.shape[-1] == 1:
            # Sigmoid (N, 1) → one-hot 2-class
            y_int = y_arr.ravel().astype(int)
            hard_onehot = np.eye(K, dtype=np.float32)[y_int]
        else:
            y_int = y_arr.ravel().astype(int)
            hard_onehot = np.eye(K, dtype=np.float32)[y_int]
        return np.concatenate([hard_onehot, teacher_soft], axis=-1)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Treina student via knowledge distillation.

        Returns:
            Dict com 'history' (Keras history.history) + 'config' + 'best_val_loss'.
        """
        cfg = self.config

        # 1. Pré-computa teacher soft targets
        teacher_soft_train = self._precompute_teacher_targets(
            X_train, batch_size=cfg.batch_size
        )
        K = teacher_soft_train.shape[-1]

        # 2. Compila student com distillation loss
        # IMPORTANTE: student deve ter output em logits para from_logits=True
        # funcionar bem. Se o student tem softmax/sigmoid no final, usamos
        # from_logits=False (com aproximação).
        last_layer = self.student.layers[-1]
        from_logits = True
        if hasattr(last_layer, 'activation'):
            act = last_layer.activation
            if act is not None and act.__name__ in ('softmax', 'sigmoid'):
                from_logits = False
                self.logger.info(
                    f"Student tem ativação '{act.__name__}' na última layer — "
                    f"usando from_logits=False"
                )

        loss = _DistillationLoss(
            alpha=cfg.alpha,
            temperature=cfg.temperature,
            num_classes=K,
            from_logits=from_logits,
        )

        # Build optimizer
        from app.domain.models.training.optimization import OptimizerFactory
        opt = OptimizerFactory().create_optimizer(
            cfg.optimizer, learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.student.compile(
            optimizer=opt,
            loss=loss,
            # Metrica accuracy do student (vs hard labels apenas — slice das K primeiras cols)
            metrics=[_DistillAccuracy(num_classes=K, from_logits=from_logits)],
        )

        # 3. Constrói targets combinados
        y_train_combined = self._build_distillation_targets(y_train, teacher_soft_train)

        val_combined = None
        if validation_data is not None:
            X_val, y_val = validation_data
            teacher_soft_val = self._precompute_teacher_targets(
                X_val, batch_size=cfg.batch_size
            )
            y_val_combined = self._build_distillation_targets(y_val, teacher_soft_val)
            val_combined = (X_val, y_val_combined)

        # 4. Callbacks
        callbacks = [tf.keras.callbacks.TerminateOnNaN()]
        if val_combined is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
                verbose=cfg.verbose,
            ))

        # 5. Treina
        history = self.student.fit(
            X_train, y_train_combined,
            validation_data=val_combined,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            verbose=cfg.verbose,
        )

        self.logger.info(
            f"Distillation concluído: {cfg.epochs} épocas, "
            f"T={cfg.temperature}, α={cfg.alpha}"
        )
        return {
            'history': history.history,
            'config': cfg.__dict__,
            'best_val_loss': float(min(history.history.get('val_loss', [float('inf')]))),
            'teacher_params': int(self.teacher.count_params()),
            'student_params': int(self.student.count_params()),
            'compression_ratio': self.teacher.count_params() / max(self.student.count_params(), 1),
        }


class _DistillAccuracy(tf.keras.metrics.Metric):
    """Accuracy custom para distillation — usa só hard labels (primeiras K cols)."""

    def __init__(self, num_classes: int, from_logits: bool = True, name='accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true_combined, y_pred, sample_weight=None):
        K = self.num_classes
        hard = y_true_combined[:, :K]
        true_class = tf.argmax(hard, axis=-1)
        if self.from_logits:
            pred_class = tf.argmax(y_pred, axis=-1)
        else:
            pred_class = tf.argmax(y_pred, axis=-1)
        match = tf.cast(tf.equal(true_class, pred_class), tf.float32)
        self.correct.assign_add(tf.reduce_sum(match))
        self.total.assign_add(tf.cast(tf.size(match), tf.float32))

    def result(self):
        return self.correct / tf.maximum(self.total, 1.0)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


def distill_from_teacher(
    teacher_model_path: str,
    student_model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    config: Optional[DistillationConfig] = None,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Helper conveniente para distillation a partir de um teacher salvo.

    Args:
        teacher_model_path: caminho do .keras ou .h5 do teacher
        student_model: modelo student não-compilado
        X_train, y_train: dataset de treinamento
        validation_data: opcional (X_val, y_val)
        config: DistillationConfig (default: temperature=4, alpha=0.3, epochs=50)

    Returns:
        (student_treinado, result_dict)
    """
    teacher = tf.keras.models.load_model(teacher_model_path, compile=False)
    cfg = config or DistillationConfig()
    trainer = KnowledgeDistillationTrainer(teacher, student_model, cfg)
    result = trainer.train(X_train, y_train, validation_data=validation_data)
    return trainer.student, result
