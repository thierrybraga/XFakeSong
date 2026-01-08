"""RawNet2 Architecture Implementation

Arquitetura de rede neural que opera diretamente no áudio bruto (raw audio)
para detecção de áudio deepfake. Baseada na arquitetura RawNet2 original.
"""

# Third-party imports
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
import logging
from tensorflow.keras import layers, models, regularizers
from app.core.utils.audio_utils import preprocess_legacy as preprocess
from app.domain.models.architectures.layers import create_classification_head

logger = logging.getLogger(__name__)


class AudioResamplingLayer(layers.Layer):
    """Custom layer para reamostrar áudio para 16kHz."""

    def __init__(self, target_sample_rate=16000, **kwargs):
        super(AudioResamplingLayer, self).__init__(**kwargs)
        self.target_sample_rate = target_sample_rate

    def call(self, inputs):
        # Para simplificação, assumimos que o áudio já está em 16kHz
        # Em uma implementação real, seria necessário reamostrar
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'target_sample_rate': self.target_sample_rate
        })
        return config


class AudioNormalizationLayer(layers.Layer):
    """Custom layer para normalizar áudio com média zero e variância unitária."""

    def __init__(self, **kwargs):
        super(AudioNormalizationLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Calcular média e desvio padrão ao longo do eixo temporal
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=-1, keepdims=True)

        # Evitar divisão por zero
        std = tf.maximum(std, 1e-8)

        # Normalizar
        normalized = (inputs - mean) / std

        return normalized

    def get_config(self):
        config = super().get_config()
        return config


class MultiScaleConv1DBlock(layers.Layer):
    """Bloco convolucional multi-escala para capturar características em diferentes escalas."""

    def __init__(self, filters, kernel_sizes=[3, 5, 7], **kwargs):
        super(MultiScaleConv1DBlock, self).__init__(**kwargs)
        self.filters = int(filters)  # Garantir que seja inteiro
        self.kernel_sizes = kernel_sizes

    def build(self, input_shape):
        super().build(input_shape)

        # Criar camadas convolucionais para cada kernel size
        self.conv_layers = []
        self.bn_layers = []

        for kernel_size in self.kernel_sizes:
            conv = layers.Conv1D(
                filters=self.filters,
                kernel_size=kernel_size,
                padding='same',
                activation=None
            )
            bn = layers.BatchNormalization()

            self.conv_layers.append(conv)
            self.bn_layers.append(bn)

        # Camada de concatenação
        self.concat = layers.Concatenate(axis=-1)

        # Camada de redução dimensional
        self.reduction_conv = layers.Conv1D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            activation='relu'
        )

        self.final_bn = layers.BatchNormalization()

    def call(self, inputs, training=None):
        # Aplicar convoluções multi-escala
        conv_outputs = []

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(inputs)
            x = bn(x, training=training)
            x = tf.nn.relu(x)
            conv_outputs.append(x)

        # Concatenar saídas
        concatenated = self.concat(conv_outputs)

        # Reduzir dimensionalidade
        output = self.reduction_conv(concatenated)
        output = self.final_bn(output, training=training)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes
        })
        return config


def create_rawnet2_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    conv_filters: list = [64, 128, 256],
    gru_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.25,
    architecture: str = 'rawnet2'
) -> models.Model:
    """Criar modelo RawNet2.

    Args:
        input_shape: Formato da entrada (samples,)
        num_classes: Número de classes (1 para detecção binária)
        conv_filters: Lista com número de filtros para cada bloco convolucional
        gru_units: Número de unidades na camada GRU
        dense_units: Número de unidades na camada densa
        dropout_rate: Taxa de dropout
        architecture: Nome da arquitetura

    Returns:
        Modelo Keras compilado
    """

    # Input layer
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # Pré-processamento
    x = AudioResamplingLayer(name='audio_resampling')(inputs)
    x = AudioNormalizationLayer(name='audio_normalization')(x)

    # Expandir dimensões para Conv1D
    x = layers.Reshape((-1, 1))(x)

    # Blocos convolucionais multi-escala
    for i, filters in enumerate(conv_filters):
        x = MultiScaleConv1DBlock(
            filters=filters,
            name=f'multiscale_conv_block_{i + 1}'
        )(x)

        # Max pooling para reduzir dimensionalidade temporal
        x = layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i + 1}')(x)

    # Camadas GRU para capturar dependências temporais
    x = layers.GRU(
        units=gru_units,
        return_sequences=True,
        name='gru_1'
    )(x)

    x = layers.GRU(
        units=gru_units // 2,
        return_sequences=False,
        name='gru_2'
    )(x)

    # Classification head using shared logic
    outputs, loss = create_classification_head(
        x,
        num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=[dense_units]
    )

    # Criar modelo
    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )

    return model


def create_lightweight_rawnet2(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    architecture: str = 'rawnet2_lite'
) -> models.Model:
    """Criar versão leve do RawNet2."""

    return create_rawnet2_model(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_filters=[32, 64, 128],
        gru_units=64,
        dense_units=32,
        dropout_rate=0.3,
        architecture=architecture
    )


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'rawnet2') -> models.Model:
    """Função principal para criar modelos RawNet2."""

    if architecture == 'rawnet2_lite':
        return create_lightweight_rawnet2(
            input_shape, num_classes, architecture)
    else:
        return create_rawnet2_model(
            input_shape, num_classes, architecture=architecture)


# Registrar objetos personalizados no Keras
tf.keras.utils.get_custom_objects().update({
    'AudioResamplingLayer': AudioResamplingLayer,
    'AudioNormalizationLayer': AudioNormalizationLayer,
    'MultiScaleConv1DBlock': MultiScaleConv1DBlock,
    'preprocess': preprocess
})
