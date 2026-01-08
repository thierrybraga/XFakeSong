"""Módulo de Data Augmentation para Áudio

Este módulo implementa técnicas de aumento de dados específicas para áudio.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import tensorflow as tf
import logging
from scipy import signal
from scipy.ndimage import shift


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
                self._time_mask
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
            max_shift = int(
                tf.shape(audio_features)[0] *
                self.time_shift_factor)
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
            mask_size = tf.cast(
                tf.cast(
                    freq_dim,
                    tf.float32) *
                self.frequency_mask_factor,
                tf.int32)
            mask_start = tf.random.uniform(
                [], 0, freq_dim - mask_size, dtype=tf.int32)

            # Criar máscara
            mask = tf.ones_like(audio_features)
            indices = tf.range(mask_start, mask_start + mask_size)
            updates = tf.zeros((mask_size, tf.shape(audio_features)[0]))

            # Aplicar máscara
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
            mask_size = tf.cast(
                tf.cast(
                    time_dim,
                    tf.float32) *
                self.time_mask_factor,
                tf.int32)
            mask_start = tf.random.uniform(
                [], 0, time_dim - mask_size, dtype=tf.int32)

            # Criar máscara
            mask = tf.ones_like(audio_features)
            indices = tf.range(mask_start, mask_start + mask_size)
            updates = tf.zeros((mask_size, tf.shape(audio_features)[1]))

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
                "frequency_mask", "time_mask", "mixup", "cutmix"
            ]
        }
