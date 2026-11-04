"""Módulo de Otimização

Este módulo implementa factory para criação de otimizadores e configurações de otimização.
"""

import logging

import tensorflow as tf


class OptimizerFactory:
    """Factory para criação de otimizadores."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._optimizers = {
            "adam": self._create_adam,
            "adamw": self._create_adamw,
            "sgd": self._create_sgd,
            "rmsprop": self._create_rmsprop,
            "adagrad": self._create_adagrad,
            "adadelta": self._create_adadelta,
            "adamax": self._create_adamax,
            "nadam": self._create_nadam,
        }

    def create_optimizer(
        self,
        optimizer_name: str,
        learning_rate: float = 0.001,
        global_clipnorm: float = 1.0,
        use_loss_scaling: bool = True,
        **kwargs,
    ) -> tf.keras.optimizers.Optimizer:
        """Cria otimizador com gradient clipping e, opcionalmente, loss scaling.

        Args:
            optimizer_name: nome do otimizador (adam, adamw, sgd, ...).
            learning_rate: taxa de aprendizado.
            global_clipnorm: clipping global da norma dos gradientes (padrão 1.0).
                Essencial para Transformers e mixed_float16 — previne explosão de
                gradiente que causa loss: nan na 1ª época.
            use_loss_scaling: envolve o otimizador em LossScaleOptimizer quando
                a política global for mixed_float16. Obrigatório para estabilidade
                numérica — sem isso, gradientes float16 fazem underflow para 0 ou
                overflow para inf/NaN.
        """
        optimizer_name = optimizer_name.lower()

        if optimizer_name not in self._optimizers:
            self.logger.warning(
                f"Otimizador '{optimizer_name}' não encontrado. Usando Adam."
            )
            optimizer_name = "adam"

        try:
            optimizer = self._optimizers[optimizer_name](
                learning_rate, global_clipnorm=global_clipnorm, **kwargs
            )

            # BUG FIX: mixed_float16 sem LossScaleOptimizer causa NaN na loss.
            # Com a política global ativa, gradientes float16 são tão pequenos
            # que fazem underflow para 0 (vanishing) ou overflow para inf/NaN.
            # LossScaleOptimizer escala a loss para cima antes do backward pass
            # e divide os gradientes de volta (dynamic loss scaling automático).
            if use_loss_scaling:
                try:
                    policy = tf.keras.mixed_precision.global_policy()
                    if policy.name == "mixed_float16":
                        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                            optimizer
                        )
                        self.logger.info(
                            "LossScaleOptimizer habilitado (mixed_float16 ativo)"
                        )
                except Exception as e:
                    self.logger.debug(f"LossScaleOptimizer skip: {e}")

            self.logger.info(
                f"Otimizador '{optimizer_name}' criado com "
                f"lr={learning_rate}, clipnorm={global_clipnorm}"
            )
            return optimizer
        except Exception as e:
            self.logger.error(f"Erro ao criar otimizador: {str(e)}")
            return tf.keras.optimizers.Adam(
                learning_rate=learning_rate, global_clipnorm=global_clipnorm
            )

    def _create_adam(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        global_clipnorm: float = 1.0,
        **kwargs,
    ) -> tf.keras.optimizers.Adam:
        """Cria otimizador Adam."""
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            global_clipnorm=global_clipnorm,
        )

    def _create_adamw(
        self,
        learning_rate: float,
        weight_decay: float = 0.004,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        global_clipnorm: float = 1.0,
        **kwargs,
    ) -> tf.keras.optimizers.AdamW:
        """Cria otimizador AdamW."""
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            global_clipnorm=global_clipnorm,
        )

    def _create_sgd(
        self,
        learning_rate: float,
        momentum: float = 0.0,
        nesterov: bool = False,
        global_clipnorm: float = 1.0,
        **kwargs,
    ) -> tf.keras.optimizers.SGD:
        """Cria otimizador SGD."""
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            global_clipnorm=global_clipnorm,
        )

    def _create_rmsprop(
        self,
        learning_rate: float,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        centered: bool = False,
        global_clipnorm: float = 1.0,
        **kwargs,
    ) -> tf.keras.optimizers.RMSprop:
        """Cria otimizador RMSprop."""
        return tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered,
            global_clipnorm=global_clipnorm,
        )

    def _create_adagrad(
        self,
        learning_rate: float,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-7,
        global_clipnorm: float = 1.0,
        **kwargs,
    ) -> tf.keras.optimizers.Adagrad:
        """Cria otimizador Adagrad."""
        return tf.keras.optimizers.Adagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            epsilon=epsilon,
            global_clipnorm=global_clipnorm,
        )

    def _create_adadelta(
        self,
        learning_rate: float,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        global_clipnorm: float = 1.0,
        **kwargs,
    ) -> tf.keras.optimizers.Adadelta:
        """Cria otimizador Adadelta."""
        return tf.keras.optimizers.Adadelta(
            learning_rate=learning_rate,
            rho=rho,
            epsilon=epsilon,
            global_clipnorm=global_clipnorm,
        )

    def _create_adamax(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        global_clipnorm: float = 1.0,
        **kwargs,
    ) -> tf.keras.optimizers.Adamax:
        """Cria otimizador Adamax."""
        return tf.keras.optimizers.Adamax(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            global_clipnorm=global_clipnorm,
        )

    def _create_nadam(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        **kwargs,
    ) -> tf.keras.optimizers.Nadam:
        """Cria otimizador Nadam."""
        return tf.keras.optimizers.Nadam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )


def create_warmup_cosine_optimizer(
    initial_learning_rate: float = 1e-3,
    warmup_steps: int = 1000,
    decay_steps: int = 50000,
    weight_decay: float = 0.01,
    alpha: float = 1e-7,
    clipnorm: float = 1.0,
) -> tf.keras.optimizers.Optimizer:
    """Helper Sprint 2.2: cria AdamW + WarmupCosineDecaySchedule.

    Ideal para Transformers (SpectrogramTransformer, Conformer, Hybrid CNN-T).
    Warmup linear nos primeiros `warmup_steps` (estabiliza gradientes da atenção)
    seguido de decaimento cosseno até `alpha * initial_lr`.

    Args:
        initial_learning_rate: LR pico (após warmup)
        warmup_steps: passos de warmup linear (típico: 1k-4k)
        decay_steps: total de passos até atingir lr=alpha (típico: epochs * steps/epoch)
        weight_decay: weight decay do AdamW
        alpha: LR mínimo como fração do inicial
        clipnorm: P1 — clipping GLOBAL da norma dos gradientes (default 1.0).
            Sem ele, um pico de gradiente na atenção fazia o SpectrogramTransformer
            divergir após o pico do warmup (val colapsava para 0.5 = uma classe;
            o teste só sobrevivia via `restore_best_weights`). Espelha o
            `clipnorm=1.0` que já estabiliza o Conformer. `None` desativa.

    Returns:
        AdamW com schedule integrado.
    """
    schedule = WarmupCosineDecaySchedule(
        initial_learning_rate=initial_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        alpha=alpha,
    )
    return tf.keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        global_clipnorm=clipnorm,
    )


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class WarmupCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up followed by cosine decay schedule."""

    def __init__(self, initial_learning_rate, warmup_steps, decay_steps, alpha=0.0):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Linear warm-up
        warmup_lr = self.initial_learning_rate * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay after warm-up
        progress = (step - warmup_steps) / tf.maximum(decay_steps - warmup_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.alpha + (self.initial_learning_rate - self.alpha) * 0.5 * (
            1.0 + tf.cos(tf.constant(3.14159265358979) * progress)
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_learning_rate": float(self.initial_learning_rate),
            "warmup_steps": int(self.warmup_steps),
            "decay_steps": int(self.decay_steps),
            "alpha": float(self.alpha),
        }


tf.keras.utils.get_custom_objects().update(
    {
        "WarmupCosineDecaySchedule": WarmupCosineDecaySchedule,
        "XFakeSong>WarmupCosineDecaySchedule": WarmupCosineDecaySchedule,
    }
)
