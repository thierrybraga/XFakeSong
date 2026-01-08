"""Multiscale CNN Architecture Implementation"""

# Third-party imports
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, List
import logging
from app.domain.models.architectures.layers import (
    STFTLayer, ResizeLayer, is_raw_audio, ensure_flat_input,
    SqueezeExciteBlock, create_classification_head
)
from app.core.utils.audio_utils import preprocess_legacy as preprocess

logger = logging.getLogger(__name__)


class SafeInputReshapeLayer(layers.Layer):
    """Layer to safely reshape inputs."""

    def __init__(self, input_shape_tuple, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_tuple = input_shape_tuple

    def call(self, x):
        input_shape = self.input_shape_tuple
        if len(input_shape) == 2:
            # Convert 2D to 3D by adding channel dimension
            x = tf.expand_dims(x, axis=-1)

            # Ensure minimum size
            target_height = tf.maximum(64, input_shape[0])
            target_width = tf.maximum(64, input_shape[1])
            x = tf.image.resize(x, (target_height, target_width))

        elif len(input_shape) == 3:
            if input_shape[-1] == 1:
                # Already has channel dimension
                target_height = tf.maximum(64, input_shape[0])
                target_width = tf.maximum(64, input_shape[1])
                x = tf.image.resize(x, (target_height, target_width))
            else:
                # Multiple channels
                target_height = tf.maximum(64, input_shape[0])
                target_width = tf.maximum(64, input_shape[1])
                x = tf.image.resize(x, (target_height, target_width))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'input_shape_tuple': self.input_shape_tuple})
        return config


def create_safe_input_layer(input_shape: Tuple[int, ...]) -> layers.Layer:
    """Creates a SafeInputReshapeLayer."""
    return SafeInputReshapeLayer(input_shape, name='safe_input_reshape')


class MultiScaleConvBlock(layers.Layer):
    """Multi-scale convolution block with different kernel sizes."""

    def __init__(self, filters: int, kernel_sizes: List[Tuple[int, int]],
                 dropout_rate: float = 0.1, **kwargs):
        super(MultiScaleConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate

        # Create convolution branches for different scales
        self.conv_branches = []
        for i, kernel_size in enumerate(kernel_sizes):
            branch = tf.keras.Sequential([
                layers.Conv2D(filters // len(kernel_sizes), kernel_size,
                              padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate)
            ], name=f'conv_branch_{i}')
            self.conv_branches.append(branch)

        # 1x1 convolution for channel reduction/expansion
        self.pointwise_conv = layers.Conv2D(
            filters, (1, 1), padding='same', activation='relu')
        self.batch_norm = layers.BatchNormalization()

        # Squeeze-and-Excitation
        self.se_block = SqueezeExciteBlock(filters)

        # Residual connection adjustment
        self.residual_conv = layers.Conv2D(filters, (1, 1), padding='same')

    def call(self, inputs, training=None):
        # Apply multi-scale convolutions
        branch_outputs = []
        for branch in self.conv_branches:
            branch_output = branch(inputs, training=training)
            branch_outputs.append(branch_output)

        # Concatenate multi-scale features
        x = layers.Concatenate(
            axis=-1, name='multiscale_concat')(branch_outputs)

        # Pointwise convolution to combine features
        x = self.pointwise_conv(x)
        x = self.batch_norm(x, training=training)

        # Apply Squeeze-and-Excitation
        x = self.se_block(x)

        # Residual connection
        residual = self.residual_conv(inputs)
        x = x + residual

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes,
            'dropout_rate': self.dropout_rate
        })
        return config


class TemporalAttentionBlock(layers.Layer):
    """Temporal attention mechanism for sequential features."""

    def __init__(self, units: int, **kwargs):
        super(TemporalAttentionBlock, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        super().build(input_shape)
        self.query_dense = layers.Dense(self.units)
        self.key_dense = layers.Dense(self.units)
        self.value_dense = layers.Dense(self.units)
        self.output_dense = layers.Dense(self.units)

    def call(self, inputs):
        # inputs shape: (batch, time, features)
        # batch_size = tf.shape(inputs)[0]
        # seq_len = tf.shape(inputs)[1]

        # Compute query, key, value
        Q = self.query_dense(inputs)  # (batch, time, units)
        K = self.key_dense(inputs)    # (batch, time, units)
        V = self.value_dense(inputs)  # (batch, time, units)

        # Compute attention scores
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch, time, time)
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))

        # Apply softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Apply attention to values
        attended = tf.matmul(attention_weights, V)  # (batch, time, units)

        # Output projection
        output = self.output_dense(attended)

        return output + inputs  # Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class SpectralAttentionBlock(layers.Layer):
    """Spectral attention mechanism for frequency features."""

    def __init__(self, filters: int, **kwargs):
        super(SpectralAttentionBlock, self).__init__(**kwargs)
        self.filters = filters

        # Attention mechanism for spectral features
        self.spectral_conv = layers.Conv2D(
            filters, (1, 3), padding='same', activation='relu')
        self.attention_conv = layers.Conv2D(
            1, (1, 1), padding='same', activation='sigmoid')

    def call(self, inputs):
        # Apply spectral convolution
        spectral_features = self.spectral_conv(inputs)

        # Compute attention weights
        attention_weights = self.attention_conv(spectral_features)

        # Apply attention
        attended_features = inputs * attention_weights

        return attended_features + inputs  # Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


def create_multiscale_cnn_model(
    input_shape: Tuple[int, ...],
    num_classes: int,
    base_filters: int = 64,
    num_blocks: int = 4,
    dropout_rate: float = 0.2,
    architecture: str = 'multiscale_cnn'
) -> models.Model:
    """
    Create Multi-Scale CNN model for audio deepfake detection.

    Simplified version without custom layers to avoid serialization issues.

    Args:
        input_shape: Shape of input features - supports 1D (samples,), 2D (height, width), or 3D (height, width, channels)
        num_classes: Number of output classes
        base_filters: Base number of filters
        num_blocks: Number of multi-scale blocks
        dropout_rate: Dropout rate
        architecture: Architecture name (for compatibility)

    Returns:
        Compiled Keras model
    """
    logger.info(
        f"Creating Multi-Scale CNN model with input_shape={input_shape}, num_classes={num_classes}")

    # Input layer
    inputs = layers.Input(shape=input_shape, name='multiscale_cnn_input')

    # Handle 1D input (raw audio) by converting to spectrogram
    if is_raw_audio(input_shape):
        input_tensor = ensure_flat_input(inputs, input_shape)

        # Convert 1D audio to spectrogram using STFT
        x = STFTLayer(name='stft_layer', add_channel_dim=True)(input_tensor)
        
        # Ensure minimum dimensions for pooling operations
        x = ResizeLayer(
            target_height=64,
            target_width=64,
            name='resize_layer')(x)
    else:
        # Safe input reshaping for 2D/3D inputs (no normalization)
        x = create_safe_input_layer(input_shape)(inputs)

    # Initial convolution with adaptive kernel size
    kernel_size = (3, 3)  # Use smaller kernel for potentially small inputs
    x = layers.Conv2D(
        base_filters,
        kernel_size,
        strides=(
            1,
            1),
        padding='same',
        activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Use standard MaxPooling2D since we ensure minimum dimensions
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Multi-scale convolution blocks (simplified)
    filters = base_filters
    for i in range(num_blocks):
        # Branch 1: 3x3 convolutions
        branch1 = layers.Conv2D(
            filters // 3,
            (3,
             3),
            activation='relu',
            padding='same',
            name=f'branch1_conv_{i}')(x)
        branch1 = layers.BatchNormalization(name=f'branch1_bn_{i}')(branch1)

        # Branch 2: 5x5 convolutions
        branch2 = layers.Conv2D(
            filters // 3,
            (5,
             5),
            activation='relu',
            padding='same',
            name=f'branch2_conv_{i}')(x)
        branch2 = layers.BatchNormalization(name=f'branch2_bn_{i}')(branch2)

        # Branch 3: 1x1 convolutions
        branch3 = layers.Conv2D(filters - 2 * (filters // 3),
                                (1,
                                 1),
                                activation='relu',
                                padding='same',
                                name=f'branch3_conv_{i}')(x)
        branch3 = layers.BatchNormalization(name=f'branch3_bn_{i}')(branch3)

        # Concatenate branches
        multiscale_out = layers.Concatenate(
            axis=-1, name=f'multiscale_concat_{i}')([branch1, branch2, branch3])
        multiscale_out = layers.Dropout(
            dropout_rate * 0.5,
            name=f'multiscale_dropout_{i}')(multiscale_out)

        # Residual connection - ensure input and output have same shape
        if x.shape[-1] != filters:
            x = layers.Conv2D(filters, (1, 1), padding='same',
                              name=f'input_projection_{i}')(x)

        x = layers.Add(name=f'residual_add_{i}')([x, multiscale_out])

        # Downsample and increase filters
        if i < num_blocks - 1:
            # Use adaptive pooling with padding='same' to avoid dimension
            # issues
            x = layers.MaxPooling2D((2, 2), padding='same')(x)
            filters *= 2

    # Global feature extraction
    # Average pooling
    avg_pool = layers.GlobalAveragePooling2D()(x)

    # Max pooling
    max_pool = layers.GlobalMaxPooling2D()(x)

    # Combine pooling strategies
    global_features = layers.Concatenate(
        axis=-1, name='global_features_concat')([avg_pool, max_pool])

    # Classification head with multiple dense layers
    outputs, loss = create_classification_head(
        global_features,
        num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=[1024, 512, 256, 128]
    )

    # Create model
    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=architecture)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999
        ),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(
        f"Multi-Scale CNN model created successfully with {model.count_params()} parameters")
    return model


def create_lightweight_multiscale_cnn(
    input_shape: Tuple[int, ...],
    num_classes: int,
    architecture: str = 'multiscale_cnn_lite'
) -> models.Model:
    """
    Create a lightweight version of Multi-Scale CNN for faster inference.

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    return create_multiscale_cnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        base_filters=32,      # Fewer base filters
        num_blocks=3,         # Fewer blocks
        dropout_rate=0.15,    # Lower dropout
        architecture=architecture
    )


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'multiscale_cnn') -> models.Model:
    """
    Factory function to create Multi-Scale CNN models (for compatibility with existing code).

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    if architecture == 'multiscale_cnn':
        return create_multiscale_cnn_model(
            input_shape, num_classes, architecture=architecture)
    elif architecture == 'multiscale_cnn_lite':
        return create_lightweight_multiscale_cnn(
            input_shape, num_classes, architecture=architecture)
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. Use 'multiscale_cnn' or 'multiscale_cnn_lite'")


# Register custom layers and functions for model loading
tf.keras.utils.get_custom_objects().update({
    'SqueezeExciteBlock': SqueezeExciteBlock,
    'MultiScaleConvBlock': MultiScaleConvBlock,
    'TemporalAttentionBlock': TemporalAttentionBlock,
    'SpectralAttentionBlock': SpectralAttentionBlock,
    'preprocess': preprocess
})
