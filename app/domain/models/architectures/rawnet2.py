"""RawNet2 Architecture Implementation

Paper-faithful implementation of RawNet2 for audio deepfake detection.
Operates directly on raw audio waveforms.

Reference: Jung et al., "Improved RawNet with Feature Map Scaling for
Text-Independent Speaker Verification using Raw Waveforms", 2020

Architecture:
1. SincNet front-end (learnable bandpass filters)
2. Residual blocks with Feature Map Scaling (FMS)
3. GRU for temporal modeling
4. Dense classifier
"""

# Third-party imports
import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from app.core.utils.audio_utils import preprocess_legacy as preprocess
from app.domain.models.architectures.layers import (
    AudioNormalizationLayer,
    FeatureMapScalingLayer,
    MultiScaleConv1DBlock,
    PreEmphasisLayer,
    ResidualBlock1D,
    SincNetLayer,
    create_classification_head,
)

logger = logging.getLogger(__name__)


class AudioResamplingLayer(layers.Layer):
    def __init__(
        self,
        source_sample_rate: int = 16000,
        target_sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_sample_rate = int(source_sample_rate)
        self.target_sample_rate = int(target_sample_rate)

    def call(self, inputs):
        x = inputs
        squeezed = False
        if x.shape.rank == 3 and x.shape[-1] == 1:
            x = tf.squeeze(x, axis=-1)
            squeezed = True

        if self.source_sample_rate == self.target_sample_rate:
            y = x
        else:
            in_len = tf.shape(x)[-1]
            ratio = tf.cast(self.target_sample_rate, tf.float32) / tf.cast(
                self.source_sample_rate, tf.float32
            )
            target_len = tf.cast(tf.round(tf.cast(in_len, tf.float32) * ratio), tf.int32)
            y = tf.signal.resample(x, target_len, axis=-1)

        if squeezed:
            y = tf.expand_dims(y, axis=-1)
        return y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "source_sample_rate": self.source_sample_rate,
                "target_sample_rate": self.target_sample_rate,
            }
        )
        return config


def _create_rawnet2_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    sinc_filters: int = 128,
    sinc_kernel_size: int = 1024,
    res_filters: list = None,
    gru_units: int = 512,
    dense_units: int = 1024,
    dropout_rate: float = 0.3,
    architecture: str = 'rawnet2'
) -> models.Model:
    """Criar modelo RawNet2 fiel ao paper 'Improved RawNet'.

    Args:
        input_shape: Formato da entrada (samples,)
        num_classes: Número de classes (1 para detecção binária)
        sinc_filters: Filtros na camada SincNet
        sinc_kernel_size: Tamanho do kernel na SincNet
        res_filters: Lista com número de filtros para cada bloco residual
        gru_units: Número de unidades na camada GRU
        dense_units: Número de unidades na camada densa
        dropout_rate: Taxa de dropout
        architecture: Nome da arquitetura

    Returns:
        Modelo Keras compilado
    """
    if res_filters is None:
        res_filters = [128, 128, 256, 256, 256, 256]

    # Input layer
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # 1. Pré-processamento
    x = PreEmphasisLayer(name='pre_emphasis')(inputs)
    x = AudioNormalizationLayer(name='audio_normalization')(x)
    if len(x.shape) == 2:
        x = layers.Reshape((-1, 1))(x)

    # 2. SincNet Front-end
    x = SincNetLayer(filters=sinc_filters, kernel_size=sinc_kernel_size, name='sincnet')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)

    # 3. Residual Blocks + FMS
    for i, filters in enumerate(res_filters):
        x = ResidualBlock1D(out_channels=filters, name=f'res_block_{i + 1}')(x)
        x = FeatureMapScalingLayer(name=f'fms_{i + 1}')(x)

        # Max pooling after some blocks to reduce temporal dimension
        if i in [1, 3]:
            x = layers.MaxPooling1D(pool_size=3)(x)

    # 4. Temporal Modeling (GRU)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)

    x = layers.GRU(
        units=gru_units,
        return_sequences=True,
        name='gru_1'
    )(x)

    x = layers.GRU(
        units=gru_units,
        return_sequences=False,
        name='gru_2'
    )(x)

    # 5. Classification Head
    outputs, loss = create_classification_head(
        x,
        num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=[dense_units]
    )

    # Criar modelo
    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    # Compilar modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(f"Modelo {architecture} criado com sucesso (Fidelidade ao paper)")
    return model


def create_lightweight_rawnet2(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    architecture: str = 'rawnet2_lite'
) -> models.Model:
    """Criar versão leve do RawNet2."""
    return _create_rawnet2_model(
        input_shape=input_shape,
        num_classes=num_classes,
        sinc_filters=16,
        sinc_kernel_size=512,
        res_filters=[16, 16, 64, 64],
        gru_units=64,
        dense_units=32,
        dropout_rate=0.3,
        architecture=architecture
    )


def create_model(input_shape: Tuple[int, ...], num_classes: int = 1,
                 architecture: str = 'rawnet2', **kwargs) -> models.Model:
    """Função principal para criar modelos RawNet2.

    Args:
        input_shape: Formato da entrada (samples,)
        num_classes: Número de classes
        architecture: Tipo de arquitetura ('rawnet2' ou 'rawnet2_lite')
        **kwargs: Parâmetros adicionais para _create_rawnet2_model

    Returns:
        Modelo Keras compilado
    """
    if architecture == 'rawnet2_lite':
        return create_lightweight_rawnet2(input_shape, num_classes, architecture)
    else:
        return _create_rawnet2_model(
            input_shape, num_classes, architecture=architecture, **kwargs)


# Registrar objetos personalizados no Keras
tf.keras.utils.get_custom_objects().update({
    'AudioResamplingLayer': AudioResamplingLayer,
    'AudioNormalizationLayer': AudioNormalizationLayer,
    'MultiScaleConv1DBlock': MultiScaleConv1DBlock,
    'PreEmphasisLayer': PreEmphasisLayer,
    'SincNetLayer': SincNetLayer,
    'ResidualBlock1D': ResidualBlock1D,
    'FeatureMapScalingLayer': FeatureMapScalingLayer,
    'preprocess': preprocess
})
