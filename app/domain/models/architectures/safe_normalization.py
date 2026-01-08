"""Safe Normalization Layers for Preventing Data Leakage"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable(package="DeepFake")
class SafeInstanceNormalization(layers.Layer):
    """
    Instance Normalization que normaliza cada amostra independentemente.
    Evita data leakage pois não usa estatísticas de outras amostras.
    """

    def __init__(self, axis: int = -1, epsilon: float = 1e-6, **kwargs):
        super(SafeInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        # Parâmetros treináveis para escala e deslocamento
        param_shape = [1] * len(input_shape)
        if self.axis != -1:
            param_shape[self.axis] = input_shape[self.axis]
        else:
            param_shape[-1] = input_shape[-1]

        self.gamma = self.add_weight(
            name='gamma',
            shape=param_shape[1:],  # Remove batch dimension
            initializer='ones',
            trainable=True
        )

        self.beta = self.add_weight(
            name='beta',
            shape=param_shape[1:],  # Remove batch dimension
            initializer='zeros',
            trainable=True
        )

        super(SafeInstanceNormalization, self).build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Calcula média e variância apenas para cada instância
        axes_to_reduce = [i for i in range(1, len(inputs.shape))
                          if i != self.axis or self.axis == -1]

        if self.axis == -1:
            axes_to_reduce = list(range(1, len(inputs.shape) - 1))

        mean = tf.reduce_mean(inputs, axis=axes_to_reduce, keepdims=True)
        variance = tf.math.reduce_variance(
            inputs, axis=axes_to_reduce, keepdims=True)

        # Normaliza
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        # Aplica escala e deslocamento
        return normalized * self.gamma + self.beta

    def get_config(self) -> Dict[str, Any]:
        config = super(SafeInstanceNormalization, self).get_config()
        config.update({
            'axis': self.axis,
            'epsilon': self.epsilon
        })
        return config


@tf.keras.utils.register_keras_serializable(package="DeepFake")
class SafeLayerNormalization(layers.Layer):
    """
    Layer Normalization segura que normaliza ao longo das features.
    Mais segura que BatchNormalization pois não depende de estatísticas do batch.
    """

    def __init__(self, axis: int = -1, epsilon: float = 1e-6, **kwargs):
        super(SafeLayerNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        # Parâmetros treináveis
        if self.axis == -1:
            param_shape = input_shape[-1]
        else:
            param_shape = input_shape[self.axis]

        self.gamma = self.add_weight(
            name='gamma',
            shape=(param_shape,),
            initializer='ones',
            trainable=True
        )

        self.beta = self.add_weight(
            name='beta',
            shape=(param_shape,),
            initializer='zeros',
            trainable=True
        )

        super(SafeLayerNormalization, self).build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Normaliza ao longo do eixo especificado
        mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        variance = tf.math.reduce_variance(
            inputs, axis=self.axis, keepdims=True)

        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        return normalized * self.gamma + self.beta

    def get_config(self) -> Dict[str, Any]:
        config = super(SafeLayerNormalization, self).get_config()
        config.update({
            'axis': self.axis,
            'epsilon': self.epsilon
        })
        return config


@tf.keras.utils.register_keras_serializable(package="DeepFake")
class SafeGroupNormalization(layers.Layer):
    """
    Group Normalization que divide os canais em grupos e normaliza dentro de cada grupo.
    Evita dependência de estatísticas do batch.
    """

    def __init__(self, groups: int = 32, axis: int = -
                 1, epsilon: float = 1e-6, **kwargs):
        super(SafeGroupNormalization, self).__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        if self.axis == -1:
            channels = input_shape[-1]
        else:
            channels = input_shape[self.axis]

        if channels % self.groups != 0:
            raise ValueError(
                f"Number of channels ({channels}) must be divisible by groups ({
                    self.groups})"
            )

        self.gamma = self.add_weight(
            name='gamma',
            shape=(channels,),
            initializer='ones',
            trainable=True
        )

        self.beta = self.add_weight(
            name='beta',
            shape=(channels,),
            initializer='zeros',
            trainable=True
        )

        super(SafeGroupNormalization, self).build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        input_shape = tf.shape(inputs)

        if self.axis == -1:
            channels = input_shape[-1]
            # Reshape para separar grupos
            reshaped = tf.reshape(inputs,
                                  [input_shape[0], -1, self.groups, channels // self.groups])

            # Normaliza dentro de cada grupo
            mean = tf.reduce_mean(reshaped, axis=[1, 3], keepdims=True)
            variance = tf.math.reduce_variance(
                reshaped, axis=[1, 3], keepdims=True)

            normalized = (reshaped - mean) / tf.sqrt(variance + self.epsilon)

            # Reshape de volta
            normalized = tf.reshape(normalized, input_shape)
        else:
            raise NotImplementedError(
                "Group normalization only supports axis=-1 currently")

        return normalized * self.gamma + self.beta

    def get_config(self) -> Dict[str, Any]:
        config = super(SafeGroupNormalization, self).get_config()
        config.update({
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon
        })
        return config


def replace_batch_normalization(layer_name: str, **kwargs) -> layers.Layer:
    """
    Substitui BatchNormalization por uma alternativa segura.

    Args:
        layer_name: Nome da camada de normalização desejada
        **kwargs: Argumentos para a camada

    Returns:
        Camada de normalização segura
    """
    normalization_map = {
        'instance': SafeInstanceNormalization,
        'layer': SafeLayerNormalization,
        'group': SafeGroupNormalization
    }

    if layer_name.lower() not in normalization_map:
        logger.warning(
            f"Tipo de normalização '{layer_name}' não reconhecido. Usando LayerNormalization.")
        return SafeLayerNormalization(**kwargs)

    return normalization_map[layer_name.lower()](**kwargs)


def get_safe_normalization_layer(
        normalization_type: str = "layer", **kwargs) -> layers.Layer:
    """
    Retorna uma camada de normalização segura que previne data leakage.

    Args:
        normalization_type: Tipo de normalização ('instance', 'layer', 'group')
        **kwargs: Argumentos para a camada de normalização

    Returns:
        Camada de normalização segura
    """
    return replace_batch_normalization(normalization_type, **kwargs)


# Registrar camadas customizadas
tf.keras.utils.get_custom_objects().update({
    'SafeInstanceNormalization': SafeInstanceNormalization,
    'SafeLayerNormalization': SafeLayerNormalization,
    'SafeGroupNormalization': SafeGroupNormalization
})
