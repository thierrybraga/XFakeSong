"""Advanced Audio Data Augmentation for DeepFake Detection"""

from __future__ import annotations

import logging
from typing import Tuple, Optional, List, Callable, Union, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers

from .components.time_domain import (
    add_gaussian_noise, time_shift, time_stretch,
    volume_perturbation, apply_smoothing
)
from .components.spectral import spectral_masking
from .components.mixing import mixup_augmentation, cutmix_augmentation

logger = logging.getLogger(__name__)


class AdvancedAudioAugmentation:
    """Classe para augmentations avançadas de áudio específicas para detecção de deepfake."""

    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration: float = 5.0,
        noise_factor: float = 0.005,
        time_shift_factor: float = 0.1,
        speed_factor: float = 0.05,
        pitch_factor: float = 0.05,
        volume_factor: float = 0.1,
        spectral_masking_freq: int = 10,
        spectral_masking_time: int = 5,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        apply_probability: float = 0.8
    ):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.noise_factor = noise_factor
        self.time_shift_factor = time_shift_factor
        self.speed_factor = speed_factor
        self.pitch_factor = pitch_factor
        self.volume_factor = volume_factor
        self.spectral_masking_freq = spectral_masking_freq
        self.spectral_masking_time = spectral_masking_time
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.apply_probability = apply_probability

    def add_gaussian_noise(self, audio: tf.Tensor) -> tf.Tensor:
        """Adiciona ruído gaussiano adaptativo."""
        return add_gaussian_noise(
            audio, self.noise_factor, self.apply_probability)

    def time_shift(self, audio: tf.Tensor) -> tf.Tensor:
        """Aplica deslocamento temporal aleatório."""
        return time_shift(audio, self.time_shift_factor,
                          self.apply_probability)

    def time_stretch(self, audio: tf.Tensor) -> tf.Tensor:
        """Aplica time stretching (mudança de velocidade)."""
        return time_stretch(audio, self.speed_factor, self.apply_probability)

    def volume_perturbation(self, audio: tf.Tensor) -> tf.Tensor:
        """Aplica perturbação de volume."""
        return volume_perturbation(
            audio, self.volume_factor, self.apply_probability)

    def spectral_masking(self, spectrogram: tf.Tensor) -> tf.Tensor:
        """Aplica masking espectral (SpecAugment)."""
        return spectral_masking(spectrogram, self.spectral_masking_freq,
                                self.spectral_masking_time, self.apply_probability)

    def mixup_augmentation(self, batch_x: tf.Tensor,
                           batch_y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Aplica Mixup augmentation."""
        return mixup_augmentation(batch_x, batch_y, self.mixup_alpha)

    def cutmix_augmentation(self, batch_x: tf.Tensor,
                            batch_y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Aplica CutMix augmentation."""
        return cutmix_augmentation(
            batch_x, batch_y, self.cutmix_alpha, self.mixup_alpha)

    def apply_smoothing(self, audio: tf.Tensor) -> tf.Tensor:
        """Aplica suavização (smoothing) para simular perda de detalhes de alta frequência."""
        return apply_smoothing(audio, self.apply_probability)

    def apply_augmentations(self, audio: tf.Tensor,
                            is_spectrogram: bool = False) -> tf.Tensor:
        """Aplica todas as augmentations de áudio."""

        # Converter para tensor bool se necessário
        is_spectrogram = tf.cast(is_spectrogram, tf.bool)

        def apply_spec_augment():
            # Proteção para build-time/tracing: garantir rank >= 2
            rank = tf.rank(audio)
            # Se for espectrograma, aplicamos spectral masking
            return self.spectral_masking(audio)

        def apply_wav_augment():
            # Augmentations no domínio do tempo
            x = audio
            x = self.add_gaussian_noise(x)
            x = self.time_shift(x)
            x = self.volume_perturbation(x)
            x = self.apply_smoothing(x)
            # Time stretch é mais complexo em tensores puros, 
            # muitas vezes feito em CPU ou pré-processamento
            # Aqui omitimos para manter compatibilidade com grafo TF puro se necessário
            return x

        return tf.cond(is_spectrogram, apply_spec_augment, apply_wav_augment)


# --- Helper Functions (Factory) ---

def create_augmentation_pipeline(config: Optional[Dict[str, Any]] = None) -> AdvancedAudioAugmentation:
    """Cria uma instância de AdvancedAudioAugmentation com configurações opcionais."""
    if config is None:
        config = {}
    return AdvancedAudioAugmentation(**config)


class AugmentationLayer(layers.Layer):
    """Keras Layer wrapper for AdvancedAudioAugmentation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.aug = create_augmentation_pipeline(config)
        
    def call(self, inputs, training=None):
        if training:
            return self.aug.apply_augmentations(inputs, is_spectrogram=False)
        return inputs
        
    def get_config(self):
        config = super().get_config()
        # Aqui idealmente salvaríamos os parametros do self.aug
        return config


def create_adaptive_augmentation_layer(
    input_shape: Tuple[int, ...], 
    strength: float = 0.3
) -> layers.Layer:
    """
    Cria uma camada Keras que aplica augmentations durante o treinamento.
    
    Args:
        input_shape: Shape da entrada.
        strength: Intensidade das augmentations (ajusta apply_probability).
    """
    config = {
        'apply_probability': 0.5 + (strength * 0.5), # Scale prob based on strength
        'noise_factor': 0.005 * (1 + strength),
        'time_shift_factor': 0.1 * (1 + strength)
    }
    
    return AugmentationLayer(config=config, name="adaptive_augmentation")


def create_robust_dataset(
    dataset: tf.data.Dataset,
    batch_size: int,
    augment: bool = True,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Prepara um dataset robusto com batching, shuffling e augmentations opcionais.
    """
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    if augment:
        # Augmentation on CPU/GPU via map
        aug = create_augmentation_pipeline()
        
        def augment_map(x, y):
            # Apply mixup/cutmix randomly could be done here or in batch
            # For simplicity, apply per-sample augmentations
            x_aug = aug.apply_augmentations(x)
            return x_aug, y
            
        dataset = dataset.map(augment_map, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
