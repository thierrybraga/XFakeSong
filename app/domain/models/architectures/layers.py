import tensorflow as tf
from tensorflow.keras import layers
import logging
from app.domain.models.architectures.safe_normalization import SafeInstanceNormalization

logger = logging.getLogger(__name__)

def is_raw_audio(input_shape):
    return len(input_shape) == 2 or (len(input_shape) == 3 and input_shape[-1] == 1)

def ensure_flat_input(x):
    """Ensure input is (batch, time, 1) or (batch, time)."""
    if len(x.shape) == 3 and x.shape[-1] > 1:
        # If we have channels, we might want to take the first one or mean?
        # For now assume it's mono or we take mean
        return tf.reduce_mean(x, axis=-1, keepdims=True)
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

def create_classification_head(x, num_classes, dropout_rate=0.3, hidden_dims=[512, 256, 128]):
    """
    Creates a standard classification head with Dense -> BN -> Dropout blocks.
    """
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
            self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape((input_shape[0], input_shape[-1]))


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
