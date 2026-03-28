import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from app.domain.models.architectures.safe_normalization import SafeInstanceNormalization

logger = logging.getLogger(__name__)

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
        return input_shape[-1] == 1 and input_shape[1] == 1 # Very specific case
    return False

def ensure_flat_input(x, input_shape=None):
    """Ensure input is (batch, time, 1) or (batch, time)."""
    if len(x.shape) == 3 and x.shape[-1] > 1:
        # If we have channels, we might want to take the first one or mean?
        # For now assume it's mono or we take mean
        return layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name="ensure_flat_mean")(x)
    return x

def apply_gru_block(x, units, return_sequences=True, go_backwards=False, name=None):
    """
    Apply GRU block compatible with both CPU and GPU.
    Uses CuDNNGRU if available (via standard GRU with default activation),
    otherwise standard GRU.
    """
    # In TensorFlow 2.x, layers.GRU will use CuDNN implementation automatically
    # if activation='tanh' and recurrent_activation='sigmoid' (defaults)
    # and unroll=False, use_bias=True.
    return layers.GRU(
        units,
        return_sequences=return_sequences,
        go_backwards=go_backwards,
        name=name
    )(x)

def flatten_features_for_gru(x, name=None):
    """
    Flatten feature dimensions (freq * channel) for GRU input (batch, time, features).
    Used in AASIST and RawGAT-ST architectures.
    """
    shape_before_gru = x.shape
    # If shape is fully defined
    if shape_before_gru[1] is not None and shape_before_gru[2] is not None and shape_before_gru[3] is not None:
         x = layers.Reshape(
            (shape_before_gru[1],
             shape_before_gru[2] *
             shape_before_gru[3]), name=name)(x)
    else:
        # Dynamic shape
        shape_tensor = tf.shape(x)
        # We want to keep time dimension (1) and flatten the rest (2 and 3)
        # Reshape to (batch, time, -1)
        x = layers.Reshape((shape_tensor[1], -1), name=name)(x)
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
        return layers.Reshape(
            (target_shape[0], target_shape[1], 1))(tensor)
    elif len(shape) == 4:
        if shape[-1] != 1:
            logger.warning(
                f"Input tensor shape {shape} for CNN expects last dim to be 1. Slicing to 1 channel.")
            return SliceLayer(slice_idx=0)(tensor)
        return tensor
    else:
        # Fallback: assume 3D if not 4D
        return layers.Reshape(
            (target_shape[0], target_shape[1], 1))(tensor)

def residual_block(x_in, filters, kernel_size, stage):
    """
    Helper function for Residual Block.
    """
    shortcut = x_in

    x = layers.Conv2D(
        filters,
        kernel_size,
        activation='relu',
        padding='same',
        name=f"res{stage}_conv1")(x_in)
    x = layers.BatchNormalization(name=f"res{stage}_bn1")(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='same',
        name=f"res{stage}_conv2")(x)
    x = layers.BatchNormalization(name=f"res{stage}_bn2")(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, (1, 1), padding='same', name=f"res{stage}_shortcut")(shortcut)

    x = layers.add([x, shortcut], name=f"res{stage}_add")
    x = layers.Activation('relu', name=f"res{stage}_relu")(x)
    return x

class STFTLayer(layers.Layer):
    """
    Compute Short-Time Fourier Transform (STFT) of audio signals.
    """
    def __init__(self, frame_length=2048, frame_step=512, fft_length=2048, add_channel_dim=True, **kwargs):
        super(STFTLayer, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.add_channel_dim = add_channel_dim

    def call(self, inputs):
        # inputs: (batch, time) or (batch, time, 1)
        if len(inputs.shape) == 3:
            inputs = tf.squeeze(inputs, axis=-1)

        # Calculate STFT
        stft = tf.signal.stft(
            inputs,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length
        )

        # Calculate magnitude
        spectrogram = tf.abs(stft)

        if self.add_channel_dim:
            spectrogram = tf.expand_dims(spectrogram, axis=-1)

        return spectrogram

    def get_config(self):
        config = super(STFTLayer, self).get_config()
        config.update({
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'fft_length': self.fft_length,
            'add_channel_dim': self.add_channel_dim
        })
        return config

class ExpandDimsLayer(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({'axis': self.axis})
        return config

class ResizeLayer(layers.Layer):
    def __init__(self, target_height, target_width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, [self.target_height, self.target_width])

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({
            'target_height': self.target_height,
            'target_width': self.target_width
        })
        return config

class MagnitudeLayer(layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)

    def get_config(self):
        return super().get_config()

class RepeatChannelLayer(layers.Layer):
    def __init__(self, repeats=3, axis=-1, **kwargs):
        super(RepeatChannelLayer, self).__init__(**kwargs)
        self.repeats = repeats
        self.axis = axis

    def call(self, inputs):
        return tf.repeat(inputs, self.repeats, axis=self.axis)

    def get_config(self):
        config = super(RepeatChannelLayer, self).get_config()
        config.update({'repeats': self.repeats, 'axis': self.axis})
        return config

class SafeEfficientNetInputLayer(layers.Layer):
    """Custom layer to safely reshape/preprocess input for EfficientNet."""

    def __init__(self, target_height=224, target_width=224, **kwargs):
        super(SafeEfficientNetInputLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        x = inputs
        # Ensure 4D for resize: (batch, height, width, channels)
        # If 3D (batch, time, feat), expand to (batch, time, feat, 1)
        if len(x.shape) == 3:
            x = tf.expand_dims(x, axis=-1)
        # If 2D (batch, feat), expand to (batch, feat, 1, 1)
        elif len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)
            x = tf.expand_dims(x, axis=-1)

        # Resize
        x = tf.image.resize(x, [self.target_height, self.target_width])

        # Handle channels
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        elif x.shape[-1] != 3:
            # If > 3, slice. If 2, this is edge case, but we handle > 3.
            if x.shape[-1] > 3:
                x = x[..., :3]
            # If 2, we might leave it or handle it, but for now 1->3 is main case.

        return x

    def get_config(self):
        config = super(SafeEfficientNetInputLayer, self).get_config()
        config.update({
            'target_height': self.target_height,
            'target_width': self.target_width
        })
        return config

def create_classification_head(x, num_classes, dropout_rate=0.3, hidden_dims=None):
    """
    Creates a standard classification head with Dense -> BN -> Dropout blocks.
    """
    if hidden_dims is None:
        hidden_dims = [512, 256, 128]
    for i, dim in enumerate(hidden_dims):
        x = layers.Dense(dim, activation='relu', name=f'classifier_dense{i+1}')(x)
        x = layers.BatchNormalization(name=f'classifier_bn{i+1}')(x)
        # Reduce dropout for deeper layers
        current_dropout = dropout_rate * (0.5 ** (i > 0)) # 0.3, 0.15, 0.15...
        x = layers.Dropout(current_dropout, name=f'classifier_dropout{i+1}')(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'

    return outputs, loss


class AudioFeatureNormalization(SafeInstanceNormalization):
    """
    DEPRECATED: Esta classe foi substituída por SafeAudioNormalization/SafeInstanceNormalization.
    Mantida apenas para compatibilidade com modelos existentes.
    Agora usa SafeInstanceNormalization internamente para garantir segurança.
    """

    def __init__(self, axis=-1, **kwargs):
        # Remove 'epsilon' from kwargs if present, as SafeInstanceNormalization handles it via super or default
        # Actually SafeInstanceNormalization might accept epsilon.
        # But let's check SafeInstanceNormalization definition if needed.
        # Assuming it accepts axis and standard layer kwargs.
        if 'epsilon' in kwargs:
            kwargs.pop('epsilon')

        super().__init__(axis=axis, **kwargs)
        logger.warning(
            "AudioFeatureNormalization está DEPRECATED. "
            "Use SafeInstanceNormalization em vez disso."
        )

    # adapt method is not needed as SafeInstanceNormalization doesn't use it in the same way (stateless)
    # or if it does, it's inherited.
    # The previous implementation had 'adapt' which stored global mean/var.
    # We explicitly want to remove that behavior.


class AttentionLayer(layers.Layer):
    """
    Camada de atenção personalizada (Bahdanau).
    Permite que o modelo foque nas partes mais relevantes da sequência de entrada.
    """

    def __init__(self, return_attention=False, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.return_attention = return_attention
        self.W = None
        self.b = None
        self.u = None

    def build(self, input_shape: tf.TensorShape):
        if len(input_shape) != 3:
            raise ValueError(
                f"AttentionLayer espera entrada 3D (batch, seq_len, features_dim), mas recebeu {input_shape}")

        features_dim = input_shape[-1]
        self.W = self.add_weight(name="att_weight", shape=(features_dim, features_dim),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(features_dim,),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_context", shape=(features_dim,),
                                 initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        uit = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        ait = tf.matmul(uit, tf.expand_dims(self.u, axis=-1))
        ait = tf.squeeze(ait, axis=-1)
        alphas = tf.nn.softmax(ait)
        output = inputs * tf.expand_dims(alphas, axis=-1)
        output = tf.reduce_sum(output, axis=1)

        if self.return_attention:
            return output, alphas
        return output

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({"return_attention": self.return_attention})
        return config

    def compute_output_shape(
            self, input_shape: tf.TensorShape):
        output_shape = tf.TensorShape((input_shape[0], input_shape[-1]))
        if self.return_attention:
            # alphas shape: (batch, seq_len)
            alphas_shape = tf.TensorShape((input_shape[0], input_shape[1]))
            return [output_shape, alphas_shape]
        return output_shape


class SqueezeExciteBlock(layers.Layer):
    """Squeeze-and-Excitation block for feature recalibration."""

    def __init__(self, filters: int, ratio: int = 16, **kwargs):
        super(SqueezeExciteBlock, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(filters // ratio, activation='relu')
        self.dense2 = layers.Dense(filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, filters))

    def call(self, inputs):
        # Squeeze
        se = self.global_avg_pool(inputs)

        # Excitation
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)

        # Scale
        return inputs * se

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio
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

    def __init__(self, filters, kernel_sizes=None, **kwargs):
        super(MultiScaleConv1DBlock, self).__init__(**kwargs)
        self.filters = int(filters)  # Garantir que seja inteiro
        self.kernel_sizes = (
            list(kernel_sizes) if kernel_sizes is not None else [3, 5, 7]
        )

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


class FeatureMapScalingLayer(layers.Layer):
    """
    Feature Map Scaling (FMS) block from RawNet2 paper.
    Similar to SE-block but specific to RawNet2.
    """

    def __init__(self, **kwargs):
        super(FeatureMapScalingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.dense = layers.Dense(self.channels, activation='sigmoid')
        super(FeatureMapScalingLayer, self).build(input_shape)

    def call(self, inputs):
        # Global Average Pooling
        y = tf.reduce_mean(inputs, axis=1)
        # Scale vector
        y = self.dense(y)
        # Reshape for broadcasting
        y = tf.expand_dims(y, axis=1)
        # Scale input
        return inputs * y + y

    def get_config(self):
        return super(FeatureMapScalingLayer, self).get_config()


class SincNetLayer(layers.Layer):
    """
    SincNet layer for raw waveform processing.
    Implementation of the Sinc-convolution from Ravanelli & Bengio (2018).
    """

    def __init__(self, filters, kernel_size, sample_rate=16000, min_low_hz=30, min_band_hz=50, **kwargs):
        super(SincNetLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Kernel size must be odd
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1

    def build(self, input_shape):
        # Filter initialization
        # Band frequencies
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        # Initialize filters
        mel = np.linspace(self._to_mel(low_hz), self._to_mel(high_hz), self.filters + 1)
        hz = self._to_hz(mel)

        # Learnable parameters: low and high frequencies
        self.low_hz_ = self.add_weight(
            name='low_hz',
            shape=(self.filters,),
            initializer=tf.constant_initializer(hz[:-1]),
            trainable=True
        )
        self.band_hz_ = self.add_weight(
            name='band_hz',
            shape=(self.filters,),
            initializer=tf.constant_initializer(np.diff(hz)),
            trainable=True
        )

        # Time axis for sinc function
        n = np.linspace(0, self.kernel_size - 1, self.kernel_size)
        n = (n - (self.kernel_size - 1) / 2) / self.sample_rate
        self.n_ = tf.constant(n, dtype=tf.float32)

        # Window function (Hamming)
        window = 0.54 - 0.46 * tf.cos(2 * np.pi * tf.range(self.kernel_size, dtype=tf.float32) / self.kernel_size)
        self.window_ = tf.constant(window, dtype=tf.float32)

        super(SincNetLayer, self).build(input_shape)

    def _to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def _to_hz(self, mel):
        return 700 * (10**(mel / 2595) - 1)

    def call(self, inputs):
        # Constraints
        low = self.min_low_hz + tf.abs(self.low_hz_)
        high = tf.clip_by_value(low + self.min_band_hz + tf.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = high - low

        # Sinc function components
        f_times_t_low = tf.matmul(tf.expand_dims(low, 1), tf.expand_dims(self.n_, 0))
        f_times_t_high = tf.matmul(tf.expand_dims(high, 1), tf.expand_dims(self.n_, 0))

        # Band-pass sinc filters
        # sinc(x) = sin(pi*x) / (pi*x)
        # filters = 2*f_high*sinc(2*pi*f_high*t) - 2*f_low*sinc(2*pi*f_low*t)

        # Center of sinc
        # Handle t=0 case by adding small epsilon or using sinc implementation
        # For simplicity in TF:
        def sinc(x):
            return tf.where(tf.equal(x, 0), tf.ones_like(x), tf.sin(np.pi * x) / (np.pi * x))

        filters_low = 2 * tf.expand_dims(low, 1) * sinc(2 * tf.expand_dims(low, 1) * tf.expand_dims(self.n_, 0))
        filters_high = 2 * tf.expand_dims(high, 1) * sinc(2 * tf.expand_dims(high, 1) * tf.expand_dims(self.n_, 0))

        filters = filters_high - filters_low
        filters = filters * self.window_

        # Normalize filters
        filters = filters / (tf.reduce_max(tf.abs(filters), axis=1, keepdims=True) + 1e-8)

        # Reshape for Conv1D: (kernel_size, in_channels, out_channels)
        # SincNet expects (kernel_size, 1, filters)
        filters = tf.transpose(filters)
        filters = tf.expand_dims(filters, 1)

        # Apply convolution
        return tf.nn.conv1d(inputs, filters, stride=1, padding='SAME')

    def get_config(self):
        config = super(SincNetLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'sample_rate': self.sample_rate
        })
        return config


# NOTE: ResidualBlock1D is defined below (after AASIST layers section)
# with out_channels parameter and LeakyReLU activation.
# Used by RawNet2, AASIST, and RawGAT-ST.


class WeightedSumLayer(layers.Layer):
    """
    Weighted sum of hidden states for self-supervised models (WavLM, HuBERT, Wav2Vec2).
    Learns a weight for each layer and computes the weighted average.
    """

    def __init__(self, num_layers: int, **kwargs):
        super(WeightedSumLayer, self).__init__(**kwargs)
        self.num_layers = num_layers

    def build(self, input_shape):
        # input_shape should be a list of tensors [layer1, layer2, ..., layerN]
        # each with shape (batch, time, feature_dim)
        self.weights = self.add_weight(
            name='layer_weights',
            shape=(self.num_layers,),
            initializer='zeros',
            trainable=True
        )
        super(WeightedSumLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs is a list of tensors
        if not isinstance(inputs, list):
            return inputs

        # Softmax to ensure weights sum to 1
        normalized_weights = tf.nn.softmax(self.weights)

        # Weighted sum
        weighted_inputs = []
        for i in range(len(inputs)):
            weighted_inputs.append(inputs[i] * normalized_weights[i])

        return tf.add_n(weighted_inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'num_layers': self.num_layers})
        return config


class GraphAttentionLayer(layers.Layer):
    """
    Camada de Atenção em Grafos (GAT) simplificada (Self-Attention).
    Combina implementações de AASIST e RawGAT-ST.
    """

    def __init__(self, output_dim: int, num_heads: int = 4, dropout_rate: float = 0.1,
                 use_residual: bool = True, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.kernel = None
        self.bias = None
        self.residual_projection = None

    def build(self, input_shape):
        # Handle both single shape and list of shapes
        if isinstance(input_shape, list):
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape

        input_dim = feature_shape[-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.output_dim * self.num_heads),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.output_dim * self.num_heads,),
            initializer="zeros",
            trainable=True
        )

        # Residual connection projection if needed
        if self.use_residual and input_dim != self.output_dim * self.num_heads:
            self.residual_projection = self.add_weight(
                name="residual_projection",
                shape=(input_dim, self.output_dim * self.num_heads),
                initializer="glorot_uniform",
                trainable=True
            )

        super(GraphAttentionLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # Handle both single tensor and list inputs
        if isinstance(inputs, list):
            features = inputs[0]
            # adjacency_matrix = inputs[1] # Ignored
        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        num_nodes = tf.shape(features)[1]

        # Linear transformation
        # (batch, nodes, output_dim * num_heads)
        transformed_features = tf.matmul(features, self.kernel)

        # Reshape for multi-head
        # (batch, nodes, heads, output_dim)
        transformed_features = tf.reshape(
            transformed_features, (batch_size, num_nodes, self.num_heads, self.output_dim))

        # Transpose for attention calculation: (batch, heads, nodes, output_dim)
        transformed_features_t = tf.transpose(transformed_features, [0, 2, 1, 3])

        # Scaled Dot-Product Attention
        # (batch, heads, nodes, nodes)
        logits = tf.matmul(transformed_features_t, tf.transpose(transformed_features_t, [0, 1, 3, 2]))
        logits = logits / tf.sqrt(tf.cast(self.output_dim, tf.float32))

        attention_coefs = tf.nn.softmax(logits, axis=-1)

        if training:
            attention_coefs = tf.nn.dropout(attention_coefs, rate=self.dropout_rate)

        # Aggregate
        # (batch, heads, nodes, output_dim)
        outputs = tf.matmul(attention_coefs, transformed_features_t)

        # Reshape back
        # (batch, nodes, heads, output_dim)
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        # (batch, nodes, heads * output_dim)
        outputs = tf.reshape(outputs, (batch_size, num_nodes, self.output_dim * self.num_heads))

        outputs = outputs + self.bias

        # Residual connection
        if self.use_residual:
            if self.residual_projection is not None:
                residual = tf.matmul(features, self.residual_projection)
            else:
                residual = features
            outputs = outputs + residual

        return tf.nn.relu(outputs)

    def get_config(self):
        config = super(GraphAttentionLayer, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_residual": self.use_residual
        })
        return config

class SliceLayer(layers.Layer):
    """Custom layer to slice the last dimension."""

    def __init__(self, slice_idx=0, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.slice_idx = slice_idx

    def call(self, inputs):
        return inputs[..., :self.slice_idx + 1]

    def get_config(self):
        config = super(SliceLayer, self).get_config()
        config.update({'slice_idx': self.slice_idx})
        return config


# ============================================================================
# AASIST Paper-Faithful Layers (Jung et al., ICASSP 2022)
# ============================================================================

class SincConvLayer(layers.Layer):
    """Learnable sinc-based bandpass filter convolution (from SincNet/RawNet2).

    Implements parameterized sinc filters where low and high cutoff frequencies
    are learnable parameters initialized on the mel scale. Each filter is a
    bandpass filter: sinc(2*pi*f_high*t) - sinc(2*pi*f_low*t), windowed by Hamming.

    Reference: Ravanelli & Bengio, "Speaker Recognition from Raw Waveform with SincNet", 2018
    """

    def __init__(self, n_filters=70, kernel_size=129, sample_rate=16000,
                 min_low_hz=50.0, min_band_hz=50.0, **kwargs):
        super(SincConvLayer, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

    def _hz_to_mel(self, hz):
        return 2595.0 * tf.math.log(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def build(self, input_shape):
        # Initialize filter frequencies on mel scale
        low_hz = self.min_low_hz
        high_hz = self.sample_rate / 2.0

        mel_low = self._hz_to_mel(tf.constant(low_hz, dtype=tf.float32))
        mel_high = self._hz_to_mel(tf.constant(high_hz, dtype=tf.float32))

        # n_filters + 1 points on mel scale, then convert back to Hz
        mel_points = tf.linspace(mel_low, mel_high, self.n_filters + 1)
        hz_points = self._mel_to_hz(mel_points)

        # Low frequencies and bandwidths
        init_low = hz_points[:-1]  # (n_filters,)
        init_band = hz_points[1:] - hz_points[:-1]  # (n_filters,)

        self.low_hz_ = self.add_weight(
            name="low_hz",
            shape=(self.n_filters,),
            initializer=tf.keras.initializers.Constant(init_low.numpy()),
            trainable=True
        )
        self.band_hz_ = self.add_weight(
            name="band_hz",
            shape=(self.n_filters,),
            initializer=tf.keras.initializers.Constant(init_band.numpy()),
            trainable=True
        )

        # Hamming window (not trainable)
        n = tf.cast(tf.range(0, self.kernel_size), tf.float32)
        self.window_ = 0.54 - 0.46 * tf.cos(2.0 * 3.14159265 * n / (self.kernel_size - 1))

        super(SincConvLayer, self).build(input_shape)

    def _sinc(self, x):
        """Normalized sinc function: sin(x) / x, with sinc(0) = 1."""
        safe_x = tf.where(tf.abs(x) < 1e-7, tf.ones_like(x) * 1e-7, x)
        return tf.sin(safe_x) / safe_x

    def call(self, inputs):
        # inputs: (batch, time, 1)
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)

        # Ensure positive frequencies
        low = self.min_low_hz + tf.abs(self.low_hz_)
        high = tf.clip_by_value(
            low + self.min_band_hz + tf.abs(self.band_hz_),
            clip_value_min=self.min_low_hz,
            clip_value_max=self.sample_rate / 2.0
        )

        # Time vector centered at 0
        n = tf.cast(tf.range(0, self.kernel_size), tf.float32)
        n = n - (self.kernel_size - 1.0) / 2.0  # Center at 0

        # Build bandpass filters: (kernel_size, n_filters)
        # low and high are (n_filters,), n is (kernel_size,)
        # We need outer product-like computation
        low_expanded = tf.expand_dims(low, 0)      # (1, n_filters)
        high_expanded = tf.expand_dims(high, 0)     # (1, n_filters)
        n_expanded = tf.expand_dims(n, 1)           # (kernel_size, 1)

        # Band-pass filter = high_pass - low_pass
        # h(n) = 2*f_high*sinc(2*pi*f_high*n) - 2*f_low*sinc(2*pi*f_low*n)
        f_low = 2.0 * low_expanded / self.sample_rate
        f_high = 2.0 * high_expanded / self.sample_rate

        band_pass_low = f_low * self._sinc(f_low * n_expanded * self.sample_rate)
        band_pass_high = f_high * self._sinc(f_high * n_expanded * self.sample_rate)

        band_pass = band_pass_high - band_pass_low  # (kernel_size, n_filters)

        # Apply Hamming window
        window = tf.expand_dims(self.window_, 1)  # (kernel_size, 1)
        band_pass = band_pass * window

        # Normalize each filter to unit energy
        band_pass = band_pass / (tf.reduce_sum(tf.abs(band_pass), axis=0, keepdims=True) + 1e-7)

        # Reshape for conv1d: (kernel_size, 1, n_filters)
        filters = tf.expand_dims(band_pass, 1)

        # Apply convolution
        output = tf.nn.conv1d(inputs, filters, stride=1, padding='SAME')

        return output

    def get_config(self):
        config = super(SincConvLayer, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'sample_rate': self.sample_rate,
            'min_low_hz': self.min_low_hz,
            'min_band_hz': self.min_band_hz
        })
        return config


class ResidualBlock1D(layers.Layer):
    """Pre-activation residual block for 1D convolutions.

    Structure: BN -> LeakyReLU -> Conv1D -> BN -> LeakyReLU -> Conv1D + skip.
    Uses 1x1 convolution for skip connection if channel mismatch.

    Reference: RawNet2 (Tak et al., 2021)
    """

    def __init__(self, out_channels, kernel_size=3, **kwargs):
        super(ResidualBlock1D, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def build(self, input_shape):
        in_channels = input_shape[-1]

        self.bn1 = layers.BatchNormalization(name=self.name + "_bn1")
        self.conv1 = layers.Conv1D(
            self.out_channels, self.kernel_size, padding='same',
            name=self.name + "_conv1")
        self.bn2 = layers.BatchNormalization(name=self.name + "_bn2")
        self.conv2 = layers.Conv1D(
            self.out_channels, self.kernel_size, padding='same',
            name=self.name + "_conv2")

        self.skip_conv = None
        if in_channels != self.out_channels:
            self.skip_conv = layers.Conv1D(
                self.out_channels, 1, padding='same',
                name=self.name + "_skip")

        super(ResidualBlock1D, self).build(input_shape)

    def call(self, inputs, training=None):
        # Pre-activation residual
        x = self.bn1(inputs, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.3)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.3)
        x = self.conv2(x)

        # Skip connection
        shortcut = inputs
        if self.skip_conv is not None:
            shortcut = self.skip_conv(inputs)

        return x + shortcut

    def get_config(self):
        config = super(ResidualBlock1D, self).get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size
        })
        return config


class GATConvLayer(layers.Layer):
    """Graph Attention Network layer with ADDITIVE attention (Velickovic et al., 2018).

    Uses the original GAT attention mechanism:
        e_ij = LeakyReLU(a_src^T * W*h_i + a_dst^T * W*h_j)
        alpha_ij = softmax_j(e_ij)
        h'_i = sum_j(alpha_ij * W*h_j)

    This is distinct from the existing GraphAttentionLayer which uses
    scaled dot-product (Transformer-style) attention.

    Reference: Velickovic et al., "Graph Attention Networks", ICLR 2018
    """

    def __init__(self, out_features, num_heads=1, dropout_rate=0.1,
                 concat_heads=True, negative_slope=0.2, **kwargs):
        super(GATConvLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.concat_heads = concat_heads
        self.negative_slope = negative_slope

    def build(self, input_shape):
        in_features = input_shape[-1]

        # Linear transformation weight: shared across heads but output is out_features * num_heads
        self.W = self.add_weight(
            name="W",
            shape=(in_features, self.out_features * self.num_heads),
            initializer="glorot_uniform",
            trainable=True
        )

        # Attention vectors: one pair (a_src, a_dst) per head
        self.a_src = self.add_weight(
            name="a_src",
            shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.a_dst = self.add_weight(
            name="a_dst",
            shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.out_features * self.num_heads if self.concat_heads else self.out_features,),
            initializer="zeros",
            trainable=True
        )

        super(GATConvLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (batch, nodes, in_features)
        batch_size = tf.shape(inputs)[0]
        num_nodes = tf.shape(inputs)[1]

        # Linear transform: (batch, nodes, out_features * num_heads)
        h = tf.matmul(inputs, self.W)

        # Reshape to (batch, nodes, num_heads, out_features)
        h = tf.reshape(h, (batch_size, num_nodes, self.num_heads, self.out_features))

        # Transpose to (batch, num_heads, nodes, out_features)
        h = tf.transpose(h, [0, 2, 1, 3])

        # Compute attention scores using additive mechanism
        # e_src: (batch, heads, nodes, 1)
        e_src = tf.einsum('bhni,hio->bhno', h, self.a_src)
        # e_dst: (batch, heads, nodes, 1)
        e_dst = tf.einsum('bhni,hio->bhno', h, self.a_dst)

        # e_ij = LeakyReLU(e_src_i + e_dst_j)
        # Broadcasting: (batch, heads, nodes, 1) + (batch, heads, 1, nodes)
        e = e_src + tf.transpose(e_dst, [0, 1, 3, 2])
        e = tf.nn.leaky_relu(e, alpha=self.negative_slope)

        # Attention coefficients
        alpha = tf.nn.softmax(e, axis=-1)  # (batch, heads, nodes, nodes)

        if training:
            alpha = tf.nn.dropout(alpha, rate=self.dropout_rate)

        # Weighted aggregation: (batch, heads, nodes, out_features)
        out = tf.matmul(alpha, h)

        # Reshape back
        # (batch, nodes, heads, out_features)
        out = tf.transpose(out, [0, 2, 1, 3])

        if self.concat_heads:
            # (batch, nodes, heads * out_features)
            out = tf.reshape(out, (batch_size, num_nodes, self.num_heads * self.out_features))
        else:
            # Average heads: (batch, nodes, out_features)
            out = tf.reduce_mean(out, axis=2)

        out = out + self.bias
        return tf.nn.elu(out)

    def get_config(self):
        config = super(GATConvLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'concat_heads': self.concat_heads,
            'negative_slope': self.negative_slope
        })
        return config


class GraphPoolLayer(layers.Layer):
    """Learnable graph pooling via top-k node selection.

    Computes a learned score per node, selects the top-k nodes (k = ratio * N),
    and gates the selected node features by their sigmoid scores.

    Reference: Graph U-Nets (Gao & Ji, 2019)
    """

    def __init__(self, ratio=0.5, **kwargs):
        super(GraphPoolLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        in_features = input_shape[-1]

        self.score_proj = self.add_weight(
            name="score_proj",
            shape=(in_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super(GraphPoolLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, nodes, features)
        num_nodes = tf.shape(inputs)[1]
        k = tf.maximum(tf.cast(tf.cast(num_nodes, tf.float32) * self.ratio, tf.int32), 1)

        # Compute scores: (batch, nodes, 1)
        scores = tf.matmul(inputs, self.score_proj)
        scores = tf.squeeze(scores, axis=-1)  # (batch, nodes)

        # Top-k selection
        _, top_indices = tf.math.top_k(scores, k=k, sorted=False)

        # Gather selected nodes
        batch_size = tf.shape(inputs)[0]

        # Create batch indices for gather_nd
        batch_indices = tf.repeat(
            tf.expand_dims(tf.range(batch_size), 1), k, axis=1
        )  # (batch, k)

        indices = tf.stack([batch_indices, top_indices], axis=-1)  # (batch, k, 2)
        selected_features = tf.gather_nd(inputs, indices)  # (batch, k, features)
        selected_scores = tf.gather_nd(scores, indices)    # (batch, k)

        # Gate by sigmoid of scores
        gate = tf.nn.sigmoid(selected_scores)  # (batch, k)
        gate = tf.expand_dims(gate, -1)        # (batch, k, 1)

        return selected_features * gate

    def get_config(self):
        config = super(GraphPoolLayer, self).get_config()
        config.update({'ratio': self.ratio})
        return config


class GraphReadoutLayer(layers.Layer):
    """Graph readout combining max readout and attention-weighted readout.

    Produces a fixed-size graph-level representation from variable-size node features
    by concatenating max-pooled and attention-weighted node features.

    Output shape: (batch, 2 * in_features)

    Reference: AASIST (Jung et al., ICASSP 2022)
    """

    def __init__(self, **kwargs):
        super(GraphReadoutLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        in_features = input_shape[-1]

        self.att_w = self.add_weight(
            name="att_w",
            shape=(in_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super(GraphReadoutLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, nodes, features)

        # Max readout
        h_max = tf.reduce_max(inputs, axis=1)  # (batch, features)

        # Attention readout
        scores = tf.matmul(inputs, self.att_w)   # (batch, nodes, 1)
        alpha = tf.nn.softmax(scores, axis=1)    # (batch, nodes, 1)
        h_att = tf.reduce_sum(inputs * alpha, axis=1)  # (batch, features)

        return tf.concat([h_max, h_att], axis=-1)  # (batch, 2*features)

    def get_config(self):
        return super(GraphReadoutLayer, self).get_config()


class HSGALLayer(layers.Layer):
    """Heterogeneous Stacking Graph Attention Layer.

    The key contribution of AASIST: cross-domain attention between spectral
    and temporal graph nodes, enabling integrated spectro-temporal analysis.

    Takes (spectral_nodes, temporal_nodes) as input and produces updated
    representations for both via cross-attention followed by self-attention
    on the combined heterogeneous graph.

    Reference: Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
    Spectro-Temporal Graph Attention Networks", ICASSP 2022
    """

    def __init__(self, out_features, num_heads=2, dropout_rate=0.1, **kwargs):
        super(HSGALLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # input_shape is a list of two shapes: [spectral_shape, temporal_shape]
        spec_features = input_shape[0][-1]
        temp_features = input_shape[1][-1]

        # Cross-attention: spectral queries attend to temporal keys/values
        self.W_q_s2t = self.add_weight(
            name="W_q_s2t", shape=(spec_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_k_s2t = self.add_weight(
            name="W_k_s2t", shape=(temp_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_v_s2t = self.add_weight(
            name="W_v_s2t", shape=(temp_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)

        # Cross-attention: temporal queries attend to spectral keys/values
        self.W_q_t2s = self.add_weight(
            name="W_q_t2s", shape=(temp_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_k_t2s = self.add_weight(
            name="W_k_t2s", shape=(spec_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_v_t2s = self.add_weight(
            name="W_v_t2s", shape=(spec_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)

        # Attention vectors for additive GAT (per cross-attention direction)
        self.a_s2t = self.add_weight(
            name="a_s2t", shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform", trainable=True)
        self.a_t2s = self.add_weight(
            name="a_t2s", shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform", trainable=True)

        # Self-attention GAT on combined graph
        combined_features = self.out_features * self.num_heads
        self.gat_self = GATConvLayer(
            out_features=self.out_features, num_heads=self.num_heads,
            dropout_rate=self.dropout_rate, concat_heads=True,
            name=self.name + "_gat_self")
        self.gat_self.build(tf.TensorShape([None, None, combined_features]))

        # Layer norms
        self.ln_spec = layers.LayerNormalization(name=self.name + "_ln_spec")
        self.ln_temp = layers.LayerNormalization(name=self.name + "_ln_temp")

        super(HSGALLayer, self).build(input_shape)

    def _cross_attention(self, queries, keys_values, W_q, W_k, W_v, a_vec, training=None):
        """Additive cross-attention between two sets of nodes."""
        batch_size = tf.shape(queries)[0]
        n_q = tf.shape(queries)[1]
        n_kv = tf.shape(keys_values)[1]

        Q = tf.matmul(queries, W_q)       # (batch, n_q, out*heads)
        K = tf.matmul(keys_values, W_k)   # (batch, n_kv, out*heads)
        V = tf.matmul(keys_values, W_v)   # (batch, n_kv, out*heads)

        # Reshape to (batch, heads, nodes, out_features)
        Q = tf.reshape(Q, (batch_size, n_q, self.num_heads, self.out_features))
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.reshape(K, (batch_size, n_kv, self.num_heads, self.out_features))
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.reshape(V, (batch_size, n_kv, self.num_heads, self.out_features))
        V = tf.transpose(V, [0, 2, 1, 3])

        # Additive attention: e_ij = LeakyReLU(a^T * (Q_i + K_j))
        # a_vec: (heads, out_features, 1)
        e_q = tf.einsum('bhni,hio->bhno', Q, a_vec)   # (batch, heads, n_q, 1)
        e_k = tf.einsum('bhni,hio->bhno', K, a_vec)   # (batch, heads, n_kv, 1)

        e = e_q + tf.transpose(e_k, [0, 1, 3, 2])  # (batch, heads, n_q, n_kv)
        e = tf.nn.leaky_relu(e, alpha=0.2)

        alpha = tf.nn.softmax(e, axis=-1)

        if training:
            alpha = tf.nn.dropout(alpha, rate=self.dropout_rate)

        # Aggregate: (batch, heads, n_q, out_features)
        out = tf.matmul(alpha, V)
        # (batch, n_q, heads, out_features)
        out = tf.transpose(out, [0, 2, 1, 3])
        # (batch, n_q, heads * out_features)
        out = tf.reshape(out, (batch_size, n_q, self.num_heads * self.out_features))

        return out

    def call(self, inputs, training=None):
        spectral_nodes, temporal_nodes = inputs

        # Cross-attention: spectral attends to temporal
        spec_cross = self._cross_attention(
            spectral_nodes, temporal_nodes,
            self.W_q_s2t, self.W_k_s2t, self.W_v_s2t, self.a_s2t,
            training=training)

        # Cross-attention: temporal attends to spectral
        temp_cross = self._cross_attention(
            temporal_nodes, spectral_nodes,
            self.W_q_t2s, self.W_k_t2s, self.W_v_t2s, self.a_t2s,
            training=training)

        # Layer norm
        spec_cross = self.ln_spec(spec_cross)
        temp_cross = self.ln_temp(temp_cross)

        # Combine into heterogeneous graph for self-attention
        combined = tf.concat([spec_cross, temp_cross], axis=1)  # (batch, n_spec+n_temp, features)
        combined = self.gat_self(combined, training=training)

        # Split back
        n_spec = tf.shape(spec_cross)[1]
        spectral_out = combined[:, :n_spec, :]
        temporal_out = combined[:, n_spec:, :]

        return spectral_out, temporal_out

    def get_config(self):
        config = super(HSGALLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


# ====================== IMPROVED LAYERS FOR ACCURACY ======================


class AMSoftmaxLayer(layers.Layer):
    """Additive Margin Softmax (AM-Softmax / CosFace).

    Replaces Dense+Softmax for more discriminative embeddings in deepfake detection.
    Reference: Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face Recognition", CVPR 2018

    Applies angular margin penalty: cos(theta) - m, then scales by s.
    During inference (training=False), returns standard cosine similarity logits.
    """

    def __init__(self, num_classes, scale=30.0, margin=0.35, **kwargs):
        super(AMSoftmaxLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

    def build(self, input_shape):
        self.W = self.add_weight(
            name='am_softmax_weights',
            shape=(input_shape[-1], self.num_classes),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AMSoftmaxLayer, self).build(input_shape)

    def call(self, inputs, labels=None, training=None):
        # L2-normalize embeddings and weights
        x_norm = tf.nn.l2_normalize(inputs, axis=-1)
        w_norm = tf.nn.l2_normalize(self.W, axis=0)

        # Cosine similarity
        cosine = tf.matmul(x_norm, w_norm)

        if training and labels is not None:
            # One-hot encode labels
            one_hot = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
            # Subtract margin from target class
            cosine = cosine - one_hot * self.margin

        # Scale logits
        logits = self.scale * cosine
        return logits

    def get_config(self):
        config = super(AMSoftmaxLayer, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'scale': self.scale,
            'margin': self.margin,
        })
        return config


class AttentionPoolingLayer(layers.Layer):
    """Attention-weighted temporal pooling.

    Replaces GlobalAveragePooling1D with learned attention weights.
    Input: (batch, time_steps, features)
    Output: (batch, features)
    """

    def __init__(self, **kwargs):
        super(AttentionPoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = layers.Dense(1, use_bias=True)
        super(AttentionPoolingLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (batch, T, D)
        # Compute attention scores
        scores = self.attention_dense(inputs)  # (batch, T, 1)
        weights = tf.nn.softmax(scores, axis=1)  # (batch, T, 1)
        # Weighted sum
        output = tf.reduce_sum(inputs * weights, axis=1)  # (batch, D)
        return output

    def get_config(self):
        return super(AttentionPoolingLayer, self).get_config()


class ConvolutionStemLayer(layers.Layer):
    """Convolution stem for Vision Transformers.

    Replaces aggressive patch embedding with gradual downsampling via
    3 stride-2 convolutions. Produces smoother feature maps.
    Reference: Xiao et al., "Early Convolutions Help Transformers See Better", NeurIPS 2021
    """

    def __init__(self, filters=None, kernel_size=3, **kwargs):
        super(ConvolutionStemLayer, self).__init__(**kwargs)
        self.filters_list = filters or [64, 128, 256]
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv_layers = []
        self.bn_layers = []
        for i, f in enumerate(self.filters_list):
            self.conv_layers.append(
                layers.Conv2D(f, self.kernel_size, strides=2, padding='same',
                              use_bias=False, name=f'conv_stem_{i}')
            )
            self.bn_layers.append(
                layers.BatchNormalization(name=f'bn_stem_{i}')
            )
        super(ConvolutionStemLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x, training=training)
            x = tf.nn.gelu(x)
        return x

    def get_config(self):
        config = super(ConvolutionStemLayer, self).get_config()
        config.update({
            'filters': self.filters_list,
            'kernel_size': self.kernel_size,
        })
        return config


class PreEmphasisLayer(layers.Layer):
    """Pre-emphasis high-pass filter for raw audio.

    Applies y[n] = x[n] - coeff * x[n-1] to sharpen high-frequency content.
    Standard in speech processing before feature extraction.
    """

    def __init__(self, coeff=0.97, **kwargs):
        super(PreEmphasisLayer, self).__init__(**kwargs)
        self.coeff = coeff

    def call(self, inputs):
        # inputs: (batch, time) or (batch, time, 1)
        squeeze = False
        if len(inputs.shape) == 3:
            squeeze = True
            inputs = tf.squeeze(inputs, axis=-1)

        # y[n] = x[n] - coeff * x[n-1]
        emphasized = inputs[:, 1:] - self.coeff * inputs[:, :-1]
        # Prepend first sample to maintain length
        first = tf.expand_dims(inputs[:, 0], axis=-1)
        emphasized = tf.concat([first, emphasized], axis=1)

        if squeeze:
            emphasized = tf.expand_dims(emphasized, axis=-1)
        return emphasized

    def get_config(self):
        config = super(PreEmphasisLayer, self).get_config()
        config.update({'coeff': self.coeff})
        return config


class DeltaFeatureLayer(layers.Layer):
    """Computes delta (velocity) and delta-delta (acceleration) features.

    Takes a spectrogram (batch, time, freq) and returns 3-channel output
    (batch, time, freq, 3) with [static, delta, delta-delta].
    Used by EfficientNet-LSTM to replace naive channel repetition.
    """

    def __init__(self, order=2, width=2, **kwargs):
        super(DeltaFeatureLayer, self).__init__(**kwargs)
        self.order = order
        self.width = width

    def call(self, inputs):
        # inputs: (batch, time, freq) or (batch, time, freq, 1)
        if len(inputs.shape) == 4:
            x = inputs[..., 0]
        else:
            x = inputs

        channels = [x]  # static

        # Delta: finite difference approximation
        for _ in range(self.order):
            prev = x
            # Pad edges by replicating boundary frames
            padded = tf.pad(prev, [[0, 0], [self.width, self.width], [0, 0]], mode='REFLECT')
            # Weighted sum of differences
            denominator = 2.0 * sum(n * n for n in range(1, self.width + 1))
            delta = tf.zeros_like(prev)
            for n in range(1, self.width + 1):
                delta += n * (padded[:, self.width + n:self.width + n + tf.shape(prev)[1], :]
                              - padded[:, self.width - n:self.width - n + tf.shape(prev)[1], :])
            delta = delta / denominator
            channels.append(delta)
            x = delta

        # Stack: (batch, time, freq, num_channels)
        return tf.stack(channels, axis=-1)

    def get_config(self):
        config = super(DeltaFeatureLayer, self).get_config()
        config.update({
            'order': self.order,
            'width': self.width,
        })
        return config


class SqueezeExcitationBlock2D(layers.Layer):
    """Squeeze-and-Excitation block for 2D feature maps (CNN models).

    Input: (batch, H, W, C) -> Output: (batch, H, W, C), channel-recalibrated.
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """

    def __init__(self, reduction=16, **kwargs):
        super(SqueezeExcitationBlock2D, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(channels // self.reduction, 4)
        self.dense1 = layers.Dense(reduced, activation='relu', name='se2d_reduce')
        self.dense2 = layers.Dense(channels, activation='sigmoid', name='se2d_expand')
        super(SqueezeExcitationBlock2D, self).build(input_shape)

    def call(self, inputs):
        # Squeeze: global average pooling over spatial dims
        se = tf.reduce_mean(inputs, axis=[1, 2])  # (batch, C)
        # Excitation: two FC layers
        se = self.dense1(se)
        se = self.dense2(se)
        # Reshape for broadcasting: (batch, 1, 1, C)
        se = tf.reshape(se, [-1, 1, 1, tf.shape(inputs)[-1]])
        return inputs * se

    def get_config(self):
        config = super(SqueezeExcitationBlock2D, self).get_config()
        config.update({'reduction': self.reduction})
        return config


class CrossAttentionFusionLayer(layers.Layer):
    """Cross-attention fusion for multi-branch ensemble models.

    Each branch embedding attends to all other branches, enabling
    information exchange between different feature representations.
    """

    def __init__(self, embed_dim=128, num_heads=4, **kwargs):
        super(CrossAttentionFusionLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        # input_shape is a list of shapes from each branch
        self.projection_layers = []
        for i in range(len(input_shape)):
            self.projection_layers.append(
                layers.Dense(self.embed_dim, name=f'proj_{i}')
            )
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            name='cross_attn'
        )
        self.layer_norm = layers.LayerNormalization()
        super(CrossAttentionFusionLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # inputs: list of (batch, D_i) tensors from each branch
        projected = []
        for i, x in enumerate(inputs):
            p = self.projection_layers[i](x)
            projected.append(p)

        # Stack as sequence: (batch, num_branches, embed_dim)
        sequence = tf.stack(projected, axis=1)

        # Self-attention across branches
        attn_out = self.mha(sequence, sequence, training=training)
        attn_out = self.layer_norm(attn_out + sequence)

        # Flatten: (batch, num_branches * embed_dim)
        static_size = attn_out.shape[1] * attn_out.shape[2]
        if static_size is not None:
            fused = tf.reshape(attn_out, [tf.shape(attn_out)[0], int(static_size)])
        else:
            fused = tf.reshape(attn_out, [tf.shape(attn_out)[0], -1])
        return fused

    def get_config(self):
        config = super(CrossAttentionFusionLayer, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        })
        return config


class GatedFusionLayer(layers.Layer):
    """Gated fusion mechanism for multi-branch models.

    Learns per-branch sigmoid gates to weight each branch's contribution.
    """

    def __init__(self, **kwargs):
        super(GatedFusionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is a list of shapes
        self.gate_layers = []
        for i in range(len(input_shape)):
            self.gate_layers.append(
                layers.Dense(1, activation='sigmoid', name=f'gate_{i}')
            )
        super(GatedFusionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: list of (batch, D_i) tensors
        gated = []
        for i, x in enumerate(inputs):
            gate = self.gate_layers[i](x)  # (batch, 1)
            gated.append(x * gate)
        return tf.concat(gated, axis=-1)

    def get_config(self):
        return super(GatedFusionLayer, self).get_config()


# Register all custom objects
tf.keras.utils.get_custom_objects().update({
    'SincConvLayer': SincConvLayer,
    'ResidualBlock1D': ResidualBlock1D,
    'GATConvLayer': GATConvLayer,
    'GraphPoolLayer': GraphPoolLayer,
    'GraphReadoutLayer': GraphReadoutLayer,
    'HSGALLayer': HSGALLayer,
    'AMSoftmaxLayer': AMSoftmaxLayer,
    'AttentionPoolingLayer': AttentionPoolingLayer,
    'ConvolutionStemLayer': ConvolutionStemLayer,
    'PreEmphasisLayer': PreEmphasisLayer,
    'DeltaFeatureLayer': DeltaFeatureLayer,
    'SqueezeExcitationBlock2D': SqueezeExcitationBlock2D,
    'CrossAttentionFusionLayer': CrossAttentionFusionLayer,
    'GatedFusionLayer': GatedFusionLayer,
})
