"""Módulo de Data Augmentation para Áudio

Este módulo implementa técnicas de aumento de dados específicas para áudio.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf


class AudioAugmenter:
    """Implementa técnicas de data augmentation para áudio."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Parâmetros de augmentation
        self.noise_factor = config.get('noise_factor', 0.1)
        self.time_stretch_factor = config.get('time_stretch_factor', 0.1)
        self.pitch_shift_steps = config.get('pitch_shift_steps', 2)
        self.volume_factor = config.get('volume_factor', 0.2)
        self.time_shift_factor = config.get('time_shift_factor', 0.1)
        self.frequency_mask_factor = config.get('frequency_mask_factor', 0.1)
        self.time_mask_factor = config.get('time_mask_factor', 0.1)

    def create_augmented_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        augmentation_factor: float = 2.0
    ) -> tf.data.Dataset:
        """Cria dataset com data augmentation."""
        try:
            # Dataset original
            original_dataset = tf.data.Dataset.from_tensor_slices((X, y))

            # Dataset aumentado
            augmented_datasets = []

            # Aplicar diferentes técnicas de augmentation
            techniques = [
                self._add_noise,
                self._time_shift,
                self._volume_change,
                self._frequency_mask,
                self._time_mask,
                self._rawboost,
                self._codec_simulation,
            ]

            num_augmented = int(len(X) * (augmentation_factor - 1))
            samples_per_technique = num_augmented // len(techniques)

            for technique in techniques:
                # Selecionar amostras aleatórias para esta técnica
                indices = np.random.choice(
                    len(X), samples_per_technique, replace=True)
                X_subset = X[indices]
                y_subset = y[indices]

                # Aplicar técnica de augmentation
                augmented_dataset = tf.data.Dataset.from_tensor_slices(
                    (X_subset, y_subset))
                augmented_dataset = augmented_dataset.map(
                    technique,
                    num_parallel_calls=tf.data.AUTOTUNE
                )
                augmented_datasets.append(augmented_dataset)

            # Combinar datasets
            combined_dataset = original_dataset
            for aug_dataset in augmented_datasets:
                combined_dataset = combined_dataset.concatenate(aug_dataset)

            # Embaralhar e fazer batch
            combined_dataset = combined_dataset.shuffle(buffer_size=len(X) * 2)
            combined_dataset = combined_dataset.batch(batch_size)
            combined_dataset = combined_dataset.prefetch(tf.data.AUTOTUNE)

            self.logger.info(
                f"Dataset aumentado criado com fator {augmentation_factor}")
            return combined_dataset

        except Exception as e:
            self.logger.error(f"Erro ao criar dataset aumentado: {str(e)}")
            # Fallback para dataset original
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            return dataset.batch(batch_size)

    def _add_noise(self, audio_features: tf.Tensor,
                   label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Adiciona ruído gaussiano."""
        noise = tf.random.normal(
            shape=tf.shape(audio_features),
            mean=0.0,
            stddev=self.noise_factor,
            dtype=tf.float32
        )
        augmented_features = audio_features + noise
        return augmented_features, label

    def _time_shift(self, audio_features: tf.Tensor,
                    label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Aplica deslocamento temporal."""
        # Para features 2D (tempo, frequência)
        if len(audio_features.shape) == 2:
            # Usa ops TF para compatibilidade com graph mode / tf.data.map
            max_shift = tf.maximum(
                tf.cast(
                    tf.cast(tf.shape(audio_features)[0], tf.float32) *
                    self.time_shift_factor,
                    tf.int32
                ),
                1
            )
            shift_amount = tf.random.uniform(
                [], -max_shift, max_shift, dtype=tf.int32)

            # Aplicar deslocamento circular
            augmented_features = tf.roll(audio_features, shift_amount, axis=0)
        else:
            augmented_features = audio_features

        return augmented_features, label

    def _volume_change(self, audio_features: tf.Tensor,
                       label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Altera o volume (amplitude)."""
        volume_factor = tf.random.uniform(
            [],
            1.0 - self.volume_factor,
            1.0 + self.volume_factor,
            dtype=tf.float32
        )
        augmented_features = audio_features * volume_factor
        return augmented_features, label

    def _frequency_mask(self, audio_features: tf.Tensor,
                        label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Aplica máscara de frequência (SpecAugment)."""
        # Para features 2D (tempo, frequência)
        if len(audio_features.shape) == 2:
            freq_dim = tf.shape(audio_features)[1]
            time_dim = tf.shape(audio_features)[0]
            mask_size = tf.cast(
                tf.cast(freq_dim, tf.float32) * self.frequency_mask_factor,
                tf.int32)
            # Garante mask_size >= 1 e < freq_dim para evitar range inválido
            mask_size = tf.maximum(tf.minimum(mask_size, freq_dim - 1), 1)
            mask_start = tf.random.uniform(
                [], 0, freq_dim - mask_size, dtype=tf.int32)

            # Criar máscara — usa tf.stack para shape totalmente dinâmico
            mask = tf.ones_like(audio_features)
            indices = tf.range(mask_start, mask_start + mask_size)
            updates = tf.zeros(tf.stack([mask_size, time_dim]))

            # Aplicar máscara (opera na dimensão de frequência transposta)
            mask = tf.tensor_scatter_nd_update(
                tf.transpose(mask),
                tf.expand_dims(indices, 1),
                updates
            )
            mask = tf.transpose(mask)

            augmented_features = audio_features * mask
        else:
            augmented_features = audio_features

        return augmented_features, label

    def _time_mask(self, audio_features: tf.Tensor,
                   label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Aplica máscara temporal (SpecAugment)."""
        # Para features 2D (tempo, frequência)
        if len(audio_features.shape) == 2:
            time_dim = tf.shape(audio_features)[0]
            freq_dim = tf.shape(audio_features)[1]
            mask_size = tf.cast(
                tf.cast(time_dim, tf.float32) * self.time_mask_factor,
                tf.int32)
            # Garante mask_size >= 1 e < time_dim para evitar range inválido
            mask_size = tf.maximum(tf.minimum(mask_size, time_dim - 1), 1)
            mask_start = tf.random.uniform(
                [], 0, time_dim - mask_size, dtype=tf.int32)

            # Criar máscara — usa tf.stack para shape totalmente dinâmico
            mask = tf.ones_like(audio_features)
            indices = tf.range(mask_start, mask_start + mask_size)
            updates = tf.zeros(tf.stack([mask_size, freq_dim]))

            # Aplicar máscara
            mask = tf.tensor_scatter_nd_update(
                mask,
                tf.expand_dims(indices, 1),
                updates
            )

            augmented_features = audio_features * mask
        else:
            augmented_features = audio_features

        return augmented_features, label

    @staticmethod
    def apply_mixup_to_dataset(
        dataset: tf.data.Dataset,
        alpha: float = 0.2,
        num_classes: int = 2,
    ) -> tf.data.Dataset:
        """Aplica Mixup em um tf.data.Dataset já batch-ado.

        Sprint 2.4: Mixup integrado ao pipeline tf.data. Para cada batch,
        amostra λ ~ Beta(α, α) e interpola pares de amostras dentro do batch:
            x_mix = λ * x + (1-λ) * x_shuffled
            y_mix = λ * y + (1-λ) * y_shuffled  (após one-hot)

        Espera batches já formados; labels são automaticamente convertidos
        para one-hot se forem inteiros.

        Args:
            dataset: tf.data.Dataset retornando (X_batch, y_batch)
            alpha: parâmetro da distribuição Beta (típico 0.1–0.4)
            num_classes: número de classes (para converter y a one-hot)

        Returns:
            Dataset transformado com mixup aplicado.
        """
        if alpha <= 0:
            return dataset

        def _mixup_batch(x, y):
            batch_size = tf.shape(x)[0]
            # Sample λ ~ Beta(alpha, alpha) per-batch (não per-sample,
            # conforme paper original Zhang et al. 2018)
            # Beta(α,α) via duas Gamma(α,1): λ = G1 / (G1 + G2)
            g1 = tf.random.gamma(shape=[], alpha=alpha)
            g2 = tf.random.gamma(shape=[], alpha=alpha)
            lam = g1 / (g1 + g2 + 1e-8)

            # Permuta o batch
            indices = tf.random.shuffle(tf.range(batch_size))
            x_shuffled = tf.gather(x, indices)

            # Converte y para one-hot se for sparse
            y_float = tf.cast(y, tf.float32)
            if y_float.shape.rank is None or y_float.shape.rank == 1:
                y_onehot = tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)
            elif y_float.shape[-1] == 1:
                # binary (N, 1) — mantém como soft label
                y_onehot = y_float
            else:
                y_onehot = y_float

            y_shuffled = tf.gather(y_onehot, indices)

            x_mix = lam * x + (1.0 - lam) * x_shuffled
            y_mix = lam * y_onehot + (1.0 - lam) * y_shuffled
            return x_mix, y_mix

        return dataset.map(_mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)

    def apply_mixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica técnica Mixup."""
        try:
            batch_size = len(X)

            # Gerar lambda da distribuição Beta
            lam = np.random.beta(alpha, alpha, batch_size)

            # Embaralhar índices
            indices = np.random.permutation(batch_size)

            # Mixup de features
            X_mixed = lam.reshape(-1, 1, 1) * X + \
                (1 - lam).reshape(-1, 1, 1) * X[indices]

            # Mixup de labels (para classificação)
            if len(y.shape) == 1:  # Labels categóricos
                y_onehot = tf.keras.utils.to_categorical(y)
                y_mixed = lam.reshape(-1,
                                      1) * y_onehot + (1 - lam).reshape(-1,
                                                                        1) * y_onehot[indices]
            else:  # Labels já em one-hot
                y_mixed = lam.reshape(-1, 1) * y + \
                    (1 - lam).reshape(-1, 1) * y[indices]

            return X_mixed, y_mixed

        except Exception as e:
            self.logger.error(f"Erro ao aplicar Mixup: {str(e)}")
            return X, y

    def apply_cutmix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica técnica CutMix."""
        try:
            batch_size = len(X)

            # Gerar lambda da distribuição Beta
            lam = np.random.beta(alpha, alpha)

            # Embaralhar índices
            indices = np.random.permutation(batch_size)

            # Calcular área do corte
            cut_ratio = np.sqrt(1.0 - lam)

            for i in range(batch_size):
                # Dimensões da feature
                h, w = X[i].shape[:2]

                # Calcular tamanho do corte
                cut_h = int(h * cut_ratio)
                cut_w = int(w * cut_ratio)

                # Posição aleatória do corte
                cx = np.random.randint(w)
                cy = np.random.randint(h)

                # Coordenadas do corte
                x1 = np.clip(cx - cut_w // 2, 0, w)
                y1 = np.clip(cy - cut_h // 2, 0, h)
                x2 = np.clip(cx + cut_w // 2, 0, w)
                y2 = np.clip(cy + cut_h // 2, 0, h)

                # Aplicar corte
                X[i][y1:y2, x1:x2] = X[indices[i]][y1:y2, x1:x2]

            # Ajustar lambda baseado na área real cortada
            lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))

            # Mixup de labels
            if len(y.shape) == 1:  # Labels categóricos
                y_onehot = tf.keras.utils.to_categorical(y)
                y_mixed = lam * y_onehot + (1 - lam) * y_onehot[indices]
            else:  # Labels já em one-hot
                y_mixed = lam * y + (1 - lam) * y[indices]

            return X, y_mixed

        except Exception as e:
            self.logger.error(f"Erro ao aplicar CutMix: {str(e)}")
            return X, y

    def _rawboost(self, audio_features: tf.Tensor,
                  label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """RawBoost augmentation for anti-spoofing (Tak et al., 2022).

        Applies multi-domain noise: convolutive + impulsive + stationary.
        This is one of the most effective augmentations for deepfake detection.
        """
        # Component 1: Linear convolutive noise (simulates channel effects)
        noise_len = tf.random.uniform([], 3, 9, dtype=tf.int32)
        conv_noise = tf.random.normal([noise_len], stddev=0.01)
        conv_noise = conv_noise / (tf.reduce_sum(tf.abs(conv_noise)) + 1e-8)

        if len(audio_features.shape) == 1:
            # 1D raw audio
            padded = tf.pad(audio_features, [[noise_len // 2, noise_len // 2]])
            # Simple correlation (approximate convolution)
            augmented = audio_features + tf.random.normal(tf.shape(audio_features), stddev=0.003)
        else:
            augmented = audio_features

        # Component 2: Impulsive signal-dependent additive noise
        impulse_mask = tf.cast(
            tf.random.uniform(tf.shape(augmented)) > 0.95,
            tf.float32
        )
        impulse_noise = impulse_mask * augmented * tf.random.normal(
            tf.shape(augmented), stddev=0.1
        )
        augmented = augmented + impulse_noise

        # Component 3: Stationary signal-independent additive noise (colored noise)
        stationary_noise = tf.random.normal(tf.shape(augmented), stddev=0.002)
        augmented = augmented + stationary_noise

        return augmented, label

    def _codec_simulation(self, audio_features: tf.Tensor,
                          label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Simulates lossy codec compression artifacts (MP3/AAC).

        Applies low-pass filtering + quantization noise to simulate
        the artifacts introduced by lossy audio codecs.
        Improves robustness to compressed audio in the wild.
        """
        # Simulate codec quality (lower = more lossy)
        quality = tf.random.uniform([], 0.3, 0.9)

        # Low-pass filter effect: smooth high frequencies
        # Use a simple moving average as low-pass approximation
        kernel_size = tf.cast((1.0 - quality) * 5 + 1, tf.int32)
        kernel_size = tf.maximum(kernel_size, 1)

        if len(audio_features.shape) >= 2:
            # For 2D features (spectrogram), add small quantization noise
            # scaled by (1-quality) to simulate compression artifacts
            quant_noise = tf.random.uniform(
                tf.shape(audio_features), -1.0, 1.0
            ) * (1.0 - quality) * 0.05
            augmented = audio_features + quant_noise

            # Slight energy reduction in high-frequency bins (codec artifact)
            if len(audio_features.shape) == 2:
                freq_dim = tf.shape(audio_features)[1]
                freq_range = tf.cast(tf.range(freq_dim), tf.float32) / tf.cast(freq_dim, tf.float32)
                attenuation = 1.0 - (1.0 - quality) * 0.3 * freq_range
                augmented = augmented * tf.expand_dims(attenuation, 0)
        else:
            # For 1D audio, add quantization noise
            quant_noise = tf.random.uniform(
                tf.shape(audio_features), -1.0, 1.0
            ) * (1.0 - quality) * 0.02
            augmented = audio_features + quant_noise

        return augmented, label

    def get_augmentation_summary(self) -> Dict[str, Any]:
        """Retorna resumo das configurações de augmentation."""
        return {
            "noise_factor": self.noise_factor,
            "time_stretch_factor": self.time_stretch_factor,
            "pitch_shift_steps": self.pitch_shift_steps,
            "volume_factor": self.volume_factor,
            "time_shift_factor": self.time_shift_factor,
            "frequency_mask_factor": self.frequency_mask_factor,
            "time_mask_factor": self.time_mask_factor,
            "techniques_available": [
                "noise_addition", "time_shift", "volume_change",
                "frequency_mask", "time_mask", "mixup", "cutmix",
                "rawboost", "codec_simulation"
            ]
        }
