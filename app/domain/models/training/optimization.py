"""Módulo de Otimização

Este módulo implementa factory para criação de otimizadores e configurações de otimização.
"""

from typing import Dict, Any, Optional
import tensorflow as tf
import logging


class OptimizerFactory:
    """Factory para criação de otimizadores."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._optimizers = {
            'adam': self._create_adam,
            'adamw': self._create_adamw,
            'sgd': self._create_sgd,
            'rmsprop': self._create_rmsprop,
            'adagrad': self._create_adagrad,
            'adadelta': self._create_adadelta,
            'adamax': self._create_adamax,
            'nadam': self._create_nadam
        }

    def create_optimizer(
        self,
        optimizer_name: str,
        learning_rate: float = 0.001,
        **kwargs
    ) -> tf.keras.optimizers.Optimizer:
        """Cria otimizador baseado no nome e parâmetros."""
        optimizer_name = optimizer_name.lower()

        if optimizer_name not in self._optimizers:
            self.logger.warning(
                f"Otimizador '{optimizer_name}' não encontrado. Usando Adam.")
            optimizer_name = 'adam'

        try:
            optimizer = self._optimizers[optimizer_name](
                learning_rate, **kwargs)
            self.logger.info(
                f"Otimizador '{optimizer_name}' criado com lr={learning_rate}")
            return optimizer
        except Exception as e:
            self.logger.error(f"Erro ao criar otimizador: {str(e)}")
            # Fallback para Adam
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _create_adam(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        **kwargs
    ) -> tf.keras.optimizers.Adam:
        """Cria otimizador Adam."""
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad
        )

    def _create_adamw(
        self,
        learning_rate: float,
        weight_decay: float = 0.004,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        **kwargs
    ) -> tf.keras.optimizers.AdamW:
        """Cria otimizador AdamW."""
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad
        )

    def _create_sgd(
        self,
        learning_rate: float,
        momentum: float = 0.0,
        nesterov: bool = False,
        **kwargs
    ) -> tf.keras.optimizers.SGD:
        """Cria otimizador SGD."""
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov
        )

    def _create_rmsprop(
        self,
        learning_rate: float,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        centered: bool = False,
        **kwargs
    ) -> tf.keras.optimizers.RMSprop:
        """Cria otimizador RMSprop."""
        return tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered
        )

    def _create_adagrad(
        self,
        learning_rate: float,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-7,
        **kwargs
    ) -> tf.keras.optimizers.Adagrad:
        """Cria otimizador Adagrad."""
        return tf.keras.optimizers.Adagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            epsilon=epsilon
        )

    def _create_adadelta(
        self,
        learning_rate: float,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        **kwargs
    ) -> tf.keras.optimizers.Adadelta:
        """Cria otimizador Adadelta."""
        return tf.keras.optimizers.Adadelta(
            learning_rate=learning_rate,
            rho=rho,
            epsilon=epsilon
        )

    def _create_adamax(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        **kwargs
    ) -> tf.keras.optimizers.Adamax:
        """Cria otimizador Adamax."""
        return tf.keras.optimizers.Adamax(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon
        )

    def _create_nadam(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        **kwargs
    ) -> tf.keras.optimizers.Nadam:
        """Cria otimizador Nadam."""
        return tf.keras.optimizers.Nadam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon
        )

    def get_available_optimizers(self) -> list:
        """Retorna lista de otimizadores disponíveis."""
        return list(self._optimizers.keys())

    def get_optimizer_config(
            self, optimizer: tf.keras.optimizers.Optimizer) -> Dict[str, Any]:
        """Retorna configuração do otimizador."""
        try:
            return optimizer.get_config()
        except Exception as e:
            self.logger.error(
                f"Erro ao obter configuração do otimizador: {
                    str(e)}")
            return {}


class LearningRateScheduler:
    """Agendador de taxa de aprendizado."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_exponential_decay(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False
    ) -> tf.keras.optimizers.schedules.ExponentialDecay:
        """Cria scheduler de decaimento exponencial."""
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase
        )

    def create_cosine_decay(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        alpha: float = 0.0
    ) -> tf.keras.optimizers.schedules.CosineDecay:
        """Cria scheduler de decaimento cosseno."""
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=alpha
        )

    def create_polynomial_decay(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False
    ) -> tf.keras.optimizers.schedules.PolynomialDecay:
        """Cria scheduler de decaimento polinomial."""
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power,
            cycle=cycle
        )

    def create_piecewise_constant(
        self,
        boundaries: list,
        values: list
    ) -> tf.keras.optimizers.schedules.PiecewiseConstantDecay:
        """Cria scheduler de decaimento por partes."""
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=values
        )

    def create_inverse_time_decay(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False
    ) -> tf.keras.optimizers.schedules.InverseTimeDecay:
        """Cria scheduler de decaimento por tempo inverso."""
        return tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase
        )


class LossFactory:
    """Factory para criação de funções de perda."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._losses = {
            'binary_crossentropy': tf.keras.losses.BinaryCrossentropy,
            'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy,
            'sparse_categorical_crossentropy': tf.keras.losses.SparseCategoricalCrossentropy,
            'mse': tf.keras.losses.MeanSquaredError,
            'mae': tf.keras.losses.MeanAbsoluteError,
            'huber': tf.keras.losses.Huber,
            'focal': self._create_focal_loss,
            'weighted_binary_crossentropy': self._create_weighted_binary_crossentropy
        }

    def create_loss(
        self,
        loss_name: str,
        **kwargs
    ) -> tf.keras.losses.Loss:
        """Cria função de perda baseada no nome."""
        loss_name = loss_name.lower()

        if loss_name not in self._losses:
            self.logger.warning(
                f"Função de perda '{loss_name}' não encontrada. Usando binary_crossentropy.")
            loss_name = 'binary_crossentropy'

        try:
            if callable(self._losses[loss_name]):
                if loss_name in ['focal', 'weighted_binary_crossentropy']:
                    return self._losses[loss_name](**kwargs)
                else:
                    return self._losses[loss_name](**kwargs)
            else:
                return self._losses[loss_name]
        except Exception as e:
            self.logger.error(f"Erro ao criar função de perda: {str(e)}")
            return tf.keras.losses.BinaryCrossentropy()

    def _create_focal_loss(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        **kwargs
    ):
        """Cria Focal Loss para lidar com desbalanceamento de classes."""
        def focal_loss(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

            # Calcular cross entropy
            ce = -y_true * tf.math.log(y_pred)

            # Calcular peso focal
            weight = alpha * y_true * tf.pow((1 - y_pred), gamma)

            # Focal loss
            fl = weight * ce

            return tf.reduce_mean(fl)

        return focal_loss

    def _create_weighted_binary_crossentropy(
        self,
        pos_weight: float = 1.0,
        **kwargs
    ):
        """Cria Binary Cross Entropy com peso para classes positivas."""
        def weighted_binary_crossentropy(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

            # Calcular weighted binary crossentropy
            loss = -(pos_weight * y_true * tf.math.log(y_pred) +
                     (1 - y_true) * tf.math.log(1 - y_pred))

            return tf.reduce_mean(loss)

        return weighted_binary_crossentropy

    def get_available_losses(self) -> list:
        """Retorna lista de funções de perda disponíveis."""
        return list(self._losses.keys())
