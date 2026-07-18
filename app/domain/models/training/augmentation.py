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
        # Faixa de SNR (dB) para o ruído aditivo calibrado. O ruído de treino
        # passa a ser parametrizado por SNR alvo (potência do sinal / SNR), a
        # MESMA definição usada na avaliação de robustez (benchmarks/data.py
        # add_awgn). Antes, _add_noise usava stddev fixo (noise_factor), não
        # calibrado por SNR e descasado dos SNRs de teste (10/20/30 dB) —
        # origem do colapso de robustez. Faixa default cobre 10/20/30 dB.
        snr_range = config.get('snr_range_db', (5.0, 40.0))
        self.snr_min_db = float(min(snr_range))
        self.snr_max_db = float(max(snr_range))
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

            # Técnicas selecionadas pelo DOMÍNIO da entrada: RawBoost/codec só
            # fazem sentido em forma de onda; SpecAugment (freq/time mask) só
            # em espectrograma. Antes, todas eram aplicadas a qualquer input
            # (ex.: máscara de "frequência" em waveform = zerar amostras).
            techniques = self._select_techniques(X.shape)

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

    def _select_techniques(self, x_shape) -> list:
        """Seleciona técnicas de augmentation pelo formato do batch X.

        - raw-audio  — (N, T) ou (N, T, 1): ruído, shift, volume, RawBoost,
          simulação de codec (distorções de forma de onda).
        - espectrograma — (N, T, F>1) ou (N, T, F, C): ruído, shift, volume,
          SpecAugment (máscaras de frequência/tempo).
        """
        rank = len(x_shape)
        is_raw = rank == 2 or (rank == 3 and int(x_shape[-1]) == 1)
        common = [self._add_noise, self._time_shift, self._volume_change]
        if is_raw:
            return [
                self._add_noise,
                self._time_shift,
                self._volume_change,
                self._rawboost,
                self._codec_simulation,
                self._room_impulse_response,
                self._dynamic_range_compression,
            ]
        return common + [self._frequency_mask, self._time_mask]

    def _add_noise(self, audio_features: tf.Tensor,
                   label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Adiciona ruído gaussiano calibrado a um SNR alvo (AWGN).

        Amostra um SNR alvo ~ U(snr_min_db, snr_max_db) e deriva o desvio-padrão
        do ruído a partir da potência do sinal da própria amostra:
            sig_power = mean(x^2);  noise_std = sqrt(sig_power / 10^(SNR/10))
        Esta é a MESMA definição usada na avaliação de robustez
        (benchmarks/data.py:add_awgn), eliminando o descasamento treino↔teste:
        o modelo passa a ver, no treino, ruído na mesma escala física em que é
        testado (cobrindo 10/20/30 dB).
        """
        x = tf.cast(audio_features, tf.float32)
        snr_db = tf.random.uniform([], self.snr_min_db, self.snr_max_db)
        snr_lin = tf.pow(10.0, snr_db / 10.0)
        sig_power = tf.reduce_mean(tf.square(x))
        noise_std = tf.sqrt(sig_power / tf.maximum(snr_lin, 1e-12))
        noise = tf.random.normal(
            shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32
        ) * noise_std
        return x + noise, label

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
            # Garante range válido inclusive para tensores estreitos (freq_dim <= 1).
            mask_size = tf.where(
                freq_dim > 1,
                tf.maximum(tf.minimum(mask_size, freq_dim - 1), 1),
                0,
            )
            max_start = tf.maximum(freq_dim - mask_size, 1)
            mask_start = tf.random.uniform([], 0, max_start, dtype=tf.int32)

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
            # Garante range válido inclusive para tensores curtos (time_dim <= 1).
            mask_size = tf.where(
                time_dim > 1,
                tf.maximum(tf.minimum(mask_size, time_dim - 1), 1),
                0,
            )
            max_start = tf.maximum(time_dim - mask_size, 1)
            mask_start = tf.random.uniform([], 0, max_start, dtype=tf.int32)

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
        from app.domain.models.training.rawboost import rawboost_tf

        rank = audio_features.shape.rank
        waveform = tf.cast(audio_features, tf.float32)
        if rank == 2 and audio_features.shape[-1] == 1:
            waveform = tf.squeeze(waveform, axis=-1)
        augmented = rawboost_tf(waveform, sr=16000, algo=4, p=1.0)
        if rank == 2 and audio_features.shape[-1] == 1:
            augmented = tf.expand_dims(augmented, axis=-1)
        return augmented, label

    @staticmethod
    def _waveform_view(audio_features: tf.Tensor):
        waveform = tf.cast(audio_features, tf.float32)
        restore_channel = (
            audio_features.shape.rank == 2
            and audio_features.shape[-1] == 1
        )
        if restore_channel:
            waveform = tf.squeeze(waveform, axis=-1)
        return waveform, restore_channel

    @staticmethod
    def _restore_waveform(waveform: tf.Tensor, restore_channel: bool) -> tf.Tensor:
        if restore_channel:
            return tf.expand_dims(waveform, axis=-1)
        return waveform

    def _codec_simulation(self, audio_features: tf.Tensor,
                          label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Codec differentiable: band-limit, decimation and mu-law quantization."""
        waveform, restore_channel = self._waveform_view(audio_features)
        framed = waveform[tf.newaxis, :, tf.newaxis]

        def _resample(factor: int) -> tf.Tensor:
            reduced = tf.nn.avg_pool1d(
                framed,
                ksize=factor,
                strides=factor,
                padding="SAME",
            )
            reconstructed = tf.repeat(reduced, repeats=factor, axis=1)
            return reconstructed[:, :tf.shape(waveform)[0], 0][0]

        reconstructed = tf.switch_case(
            tf.random.uniform([], 0, 3, dtype=tf.int32),
            branch_fns={
                0: lambda: _resample(2),
                1: lambda: _resample(3),
                2: lambda: _resample(4),
            },
        )

        mu = tf.cast(
            tf.random.uniform([], 63, 256, dtype=tf.int32),
            tf.float32,
        )
        peak = tf.maximum(tf.reduce_max(tf.abs(reconstructed)), 1e-6)
        normalized = tf.clip_by_value(reconstructed / peak, -1.0, 1.0)
        companded = (
            tf.sign(normalized)
            * tf.math.log1p(mu * tf.abs(normalized))
            / tf.math.log1p(mu)
        )
        quantized = (
            tf.round((companded + 1.0) * mu / 2.0) * 2.0 / mu - 1.0
        )
        decoded = (
            tf.sign(quantized)
            * tf.math.expm1(tf.abs(quantized) * tf.math.log1p(mu))
            / mu
        ) * peak
        decoded = tf.where(tf.math.is_finite(decoded), decoded, waveform)
        return self._restore_waveform(decoded, restore_channel), label

    def _room_impulse_response(self, audio_features: tf.Tensor,
                               label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Convolução com RIR sintética causal de decaimento aleatório."""
        waveform, restore_channel = self._waveform_view(audio_features)
        taps = 257
        time = tf.range(taps, dtype=tf.float32) / 16000.0
        decay_seconds = tf.random.uniform([], 0.04, 0.35)
        envelope = tf.exp(-time / decay_seconds)
        rir = tf.random.normal([taps], stddev=0.12) * envelope
        rir = tf.tensor_scatter_nd_add(rir, [[0]], [1.0])
        rir = rir / tf.maximum(tf.reduce_sum(tf.abs(rir)), 1e-6)
        reverberant = tf.nn.conv1d(
            waveform[tf.newaxis, :, tf.newaxis],
            rir[:, tf.newaxis, tf.newaxis],
            stride=1,
            padding="SAME",
        )[0, :, 0]
        dry_wet = tf.random.uniform([], 0.25, 0.75)
        augmented = (1.0 - dry_wet) * waveform + dry_wet * reverberant
        return self._restore_waveform(augmented, restore_channel), label

    def _dynamic_range_compression(
        self,
        audio_features: tf.Tensor,
        label: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compressão suave que não é anulada pelo z-score de amplitude."""
        waveform, restore_channel = self._waveform_view(audio_features)
        drive = tf.random.uniform([], 1.5, 8.0)
        peak = tf.maximum(tf.reduce_max(tf.abs(waveform)), 1e-6)
        normalized = waveform / peak
        compressed = (
            tf.sign(normalized)
            * tf.math.log1p(drive * tf.abs(normalized))
            / tf.math.log1p(drive)
        ) * peak
        return self._restore_waveform(compressed, restore_channel), label

    def get_augmentation_summary(self) -> Dict[str, Any]:
        """Retorna resumo das configurações de augmentation."""
        return {
            "noise_factor": self.noise_factor,
            "snr_range_db": (self.snr_min_db, self.snr_max_db),
            "time_stretch_factor": self.time_stretch_factor,
            "pitch_shift_steps": self.pitch_shift_steps,
            "volume_factor": self.volume_factor,
            "time_shift_factor": self.time_shift_factor,
            "frequency_mask_factor": self.frequency_mask_factor,
            "time_mask_factor": self.time_mask_factor,
            "techniques_available": [
                "noise_addition", "time_shift", "volume_change",
                "frequency_mask", "time_mask", "mixup", "cutmix",
                "rawboost", "codec_simulation", "room_impulse_response",
                "dynamic_range_compression"
            ]
        }
