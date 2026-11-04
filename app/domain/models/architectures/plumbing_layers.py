"""Forma e encanamento: reshape, canais, redimensionamento e pequenos helpers
que as arquiteturas compartilham. Saiu de ``layers.py`` (P2-15), que continua
sendo a porta de entrada e o ponto de registro no Keras.
"""

import logging

import tensorflow as tf
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ChannelMeanLayer(layers.Layer):
    """mean(x, axis=-1, keepdims=True) como camada serializável.

    Substitui `layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, ...))` em
    `ensure_flat_input` — mesma razão de `AxisMaxAbsLayer` (Lambda de função
    Python não sobrevive ao load em safe_mode, o default do Keras 3).
    """

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


def is_raw_audio(input_shape):
    """
    Check if input shape corresponds to raw audio.
    Assumes input_shape does not include batch dimension.
    (Time,) -> Raw
    (Time, 1) -> Raw
    (Time, >1) -> Features (e.g. Spectrogram)
    """
    if len(input_shape) == 1:
        return True
    if len(input_shape) == 2:
        return input_shape[-1] == 1
    if len(input_shape) == 3:
        return input_shape[-1] == 1 and input_shape[1] == 1  # Very specific case
    return False


def ensure_flat_input(x):
    """Ensure input is (batch, time, 1) or (batch, time).

    (O parâmetro `input_shape` foi REMOVIDO: era declarado, todos os callers o
    passavam e a função nunca o lia — a decisão sai do `x.shape` real. Uma
    assinatura que mente sobre o que usa é a mesma classe de problema do
    "config morto" no registry.)
    """
    if len(x.shape) == 3 and x.shape[-1] > 1:
        # If we have channels, we might want to take the first one or mean?
        # For now assume it's mono or we take mean.
        # ChannelMeanLayer (não layers.Lambda com lambda Python crua): mesma
        # razão de AxisMaxAbsLayer — Lambda de função Python não é
        # recarregável em safe_mode (default do Keras 3).
        return ChannelMeanLayer(name="ensure_flat_mean")(x)
    return x


def apply_gru_block(
    x, units, return_sequences=True, go_backwards=False, dropout_rate=0.0, name=None
):
    """
    Apply GRU block compatible with both CPU and GPU.
    Standard GRU automatically uses the cuDNN kernel when conditions are met
    (default activations, no recurrent_dropout, unroll=False).
    """
    return layers.GRU(
        units,
        return_sequences=return_sequences,
        go_backwards=go_backwards,
        dropout=dropout_rate,
        name=name,
    )(x)


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class FlattenTimeFeaturesLayer(layers.Layer):
    """(B, T, F, C) -> (B, T, F*C) para formas dinâmicas, como camada serializável.

    Substitui `layers.Lambda(lambda t: tf.reshape(...))` — Lambda de função
    Python não é reconstruível pelo carregador safe_mode do Keras 3.
    """

    def call(self, inputs):
        shape = tf.shape(inputs)
        return tf.reshape(inputs, [shape[0], shape[1], -1])

    def compute_output_shape(self, input_shape):
        tail = input_shape[2:]
        flat = None
        if all(dim is not None for dim in tail):
            flat = 1
            for dim in tail:
                flat *= dim
        return (input_shape[0], input_shape[1], flat)

    def get_config(self):
        return super().get_config()


def flatten_features_for_gru(x, name=None):
    """
    Flatten feature dimensions (freq * channel) for GRU input (batch, time, features).
    Used in AASIST and RawGAT-ST architectures.
    """
    shape_before_gru = x.shape
    # If shape is fully defined
    if (
        shape_before_gru[1] is not None
        and shape_before_gru[2] is not None
        and shape_before_gru[3] is not None
    ):
        x = layers.Reshape(
            (shape_before_gru[1], shape_before_gru[2] * shape_before_gru[3]), name=name
        )(x)
    else:
        # Forma dinâmica: camada registrada (não Lambda) para que o modelo
        # salvo volte a carregar em safe_mode.
        x = FlattenTimeFeaturesLayer(name=name)(x)
    return x


def apply_reshape_for_cnn(tensor, target_shape):
    """
    Helper function for Reshape for CNN using SliceLayer to avoid Lambda.
    """
    # Logic:
    # If 3D -> Reshape to 4D with 1 channel
    # If 4D with > 1 channel -> Slice to 1 channel
    # If 4D with 1 channel -> Keep

    shape = tensor.shape

    if len(shape) == 3:
        return layers.Reshape((target_shape[0], target_shape[1], 1))(tensor)
    elif len(shape) == 4:
        if shape[-1] != 1:
            logger.warning(
                f"Input tensor shape {shape} for CNN expects last dim to be 1. Slicing to 1 channel."
            )
            return SliceLayer(slice_idx=0)(tensor)
        return tensor
    else:
        # Fallback: assume 3D if not 4D
        return layers.Reshape((target_shape[0], target_shape[1], 1))(tensor)


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ExpandDimsLayer(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ResizeLayer(layers.Layer):
    def __init__(self, target_height, target_width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, [self.target_height, self.target_width])

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update(
            {"target_height": self.target_height, "target_width": self.target_width}
        )
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class TimeResizeLayer(layers.Layer):
    """Reamostra o eixo temporal de (B, T, C) para (B, target_length, C).

    Camada serializável no lugar de `layers.Lambda(_resize_time)` (back-end de
    grafo SSL): o grafo do GAT espectral precisa de dimensão temporal ESTÁTICA,
    mas a saída do backbone SSL tem T dinâmico.
    """

    def __init__(self, target_length: int, **kwargs):
        super().__init__(**kwargs)
        self.target_length = int(target_length)

    def call(self, inputs):
        z4 = tf.expand_dims(inputs, axis=1)  # (B, 1, T, C)
        z4 = tf.image.resize(z4, [1, self.target_length])  # (B, 1, T_fix, C)
        return tf.cast(tf.squeeze(z4, axis=1), self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_length, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"target_length": self.target_length})
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AxisMaxAbsLayer(layers.Layer):
    """max(|x|, axis=axis) como camada serializável.

    Substitui `layers.Lambda(lambda v: tf.reduce_max(tf.abs(v), axis=N))`
    usado em AASIST/RawGAT-ST para extrair os nós espectral/temporal do
    encoder 2D. Uma Lambda com função Python crua não é reconstruível pelo
    carregador SAFE MODE do Keras 3 (`ValueError: ... Lambda layer whose
    function is a Python lambda ... disallowed by default`) — o modelo
    treina e salva normalmente, mas falha ao ser recarregado sem
    `safe_mode=False`. Mesma computação, sem o risco de execução de código
    arbitrário na desserialização.
    """

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = int(axis)

    def call(self, inputs):
        return tf.reduce_max(tf.abs(inputs), axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RepeatChannelLayer(layers.Layer):
    def __init__(self, repeats=3, axis=-1, **kwargs):
        super(RepeatChannelLayer, self).__init__(**kwargs)
        self.repeats = repeats
        self.axis = axis

    def call(self, inputs):
        return tf.repeat(inputs, self.repeats, axis=self.axis)

    def get_config(self):
        config = super(RepeatChannelLayer, self).get_config()
        config.update({"repeats": self.repeats, "axis": self.axis})
        return config


class SliceLayer(layers.Layer):
    """Custom layer to slice the last dimension."""

    def __init__(self, slice_idx=0, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.slice_idx = slice_idx

    def call(self, inputs):
        return inputs[..., : self.slice_idx + 1]

    def get_config(self):
        config = super(SliceLayer, self).get_config()
        config.update({"slice_idx": self.slice_idx})
        return config
