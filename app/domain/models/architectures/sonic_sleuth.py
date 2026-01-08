"""Sonic Sleuth Architecture Implementation

Arquitetura CNN especializada para detecção de deepfake usando espectrogramas de mel.
Desenvolvida para processar áudios de até 3 segundos com taxa de amostragem de 16 kHz.
"""

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import logging
import librosa
from app.domain.models.architectures.layers import is_raw_audio, ensure_flat_input, create_classification_head

logger = logging.getLogger(__name__)

# Global preprocess function for model loading compatibility


def preprocess(x):
    """Global preprocessing function for Sonic Sleuth compatibility."""
    # Se entrada é 1D (áudio raw), converter para espectrograma de mel
    if len(x.shape) == 2 and x.shape[-1] == 1:  # (batch, samples, 1)
        x = tf.squeeze(x, axis=-1)  # Remove channel dimension

    if len(x.shape) == 2:  # (batch, samples)
        # Converter para espectrograma de mel usando TensorFlow
        mel_spectrograms = []
        for i in range(tf.shape(x)[0]):
            audio_sample = x[i]

            # Calcular STFT
            stft = tf.signal.stft(
                audio_sample,
                frame_length=1024,
                frame_step=256,
                fft_length=1024
            )

            # Calcular magnitude
            magnitude = tf.abs(stft)

            # Converter para escala de mel
            mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=128,
                num_spectrogram_bins=513,  # (fft_length // 2) + 1
                sample_rate=16000,
                lower_edge_hertz=0.0,
                upper_edge_hertz=8000.0
            )

            mel_spectrogram = tf.tensordot(magnitude, mel_weight_matrix, 1)
            mel_spectrograms.append(mel_spectrogram)

        x = tf.stack(mel_spectrograms)

    # Adicionar dimensão de canal se necessário
    if len(x.shape) == 3:  # (batch, time, mel_bins)
        x = tf.expand_dims(x, axis=-1)  # Add channel dimension

    # Normalizar para [0, 1] range
    x = tf.nn.sigmoid(x)

    return x


class MelSpectrogramLayer(layers.Layer):
    """Custom layer para converter áudio em espectrograma de mel."""

    def __init__(self,
                 sample_rate=16000,
                 n_fft=1024,
                 hop_length=256,
                 n_mels=128,
                 **kwargs):
        super(MelSpectrogramLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def call(self, inputs):
        # Calcular STFT
        stft = tf.signal.stft(
            inputs,
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft
        )

        # Calcular magnitude
        magnitude = tf.abs(stft)

        # Converter para escala de mel
        mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=(self.n_fft // 2) + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=self.sample_rate / 2.0
        )

        mel_spectrogram = tf.tensordot(magnitude, mel_weight_matrix, 1)

        # Adicionar dimensão de canal
        mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=-1)

        return mel_spectrogram

    def get_config(self):
        config = super().get_config()
        config.update({
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels
        })
        return config


class ConvBlock(layers.Layer):
    """Bloco convolucional com Conv2D, MaxPool2D e Dropout."""

    def __init__(self, filters, kernel_size=(
            3, 3), dropout_rate=0.25, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same'
        )
        self.maxpool = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.maxpool(x)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate
        })
        return config


def create_sonic_sleuth_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    dropout_rate: float = 0.25,
    architecture: str = 'sonic_sleuth'
) -> models.Model:
    """Cria o modelo Sonic Sleuth.

    Args:
        input_shape: Formato da entrada (samples,) para áudio raw ou (time, mels) para espectrograma
        num_classes: Número de classes (2 para binário: real/fake)
        sample_rate: Taxa de amostragem do áudio
        n_fft: Tamanho da janela FFT
        hop_length: Tamanho do passo da janela
        n_mels: Número de filtros mel
        dropout_rate: Taxa de dropout
        architecture: Nome da arquitetura

    Returns:
        Modelo Keras compilado
    """

    logger.info(
        f"Creating Sonic Sleuth model with input_shape={input_shape}, num_classes={num_classes}")

    # Entrada
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # Se entrada é áudio raw (1D), converter para espectrograma de mel
    if is_raw_audio(input_shape):
        input_tensor = ensure_flat_input(inputs, input_shape)

        x = MelSpectrogramLayer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )(input_tensor)
    else:  # Já é espectrograma
        x = inputs
        if len(input_shape) == 2:  # Adicionar dimensão de canal
            x = layers.Reshape((*input_shape, 1))(x)

    # Camadas Convolucionais com expansão de filtros: 32 -> 64 -> 128

    # Primeiro bloco: 32 filtros
    x = ConvBlock(filters=32, dropout_rate=dropout_rate)(x)

    # Segundo bloco: 64 filtros
    x = ConvBlock(filters=64, dropout_rate=dropout_rate)(x)

    # Terceiro bloco: 128 filtros
    x = ConvBlock(filters=128, dropout_rate=dropout_rate)(x)

    # Achatar para camadas densas
    x = layers.Flatten()(x)

    # Classification head using shared logic
    outputs, loss = create_classification_head(
        x,
        num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=[256, 128]
    )

    # Criar modelo
    model = models.Model(inputs=inputs, outputs=outputs, name='sonic_sleuth')

    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )

    # Log do modelo criado
    total_params = model.count_params()
    logger.info(
        f"Sonic Sleuth model created successfully with {total_params} parameters")

    return model


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'sonic_sleuth') -> models.Model:
    """Interface padrão para criação do modelo.

    Args:
        input_shape: Formato da entrada
        num_classes: Número de classes
        architecture: Nome da arquitetura

    Returns:
        Modelo Keras
    """
    return create_sonic_sleuth_model(
        input_shape=input_shape,
        num_classes=num_classes,
        architecture=architecture
    )


# Registrar objetos customizados para compatibilidade com
# salvamento/carregamento
tf.keras.utils.get_custom_objects().update({
    'MelSpectrogramLayer': MelSpectrogramLayer,
    'ConvBlock': ConvBlock,
    'preprocess': preprocess
})
