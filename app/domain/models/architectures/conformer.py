"""Conformer Architecture Implementation"""

import tensorflow as tf
from tensorflow.keras import layers
from app.domain.models.architectures.layers import create_classification_head
from typing import Tuple


class ConvolutionModule(layers.Layer):
    """Convolution module for Conformer architecture."""

    def __init__(self, channels, kernel_size=31, dropout_rate=0.1, **kwargs):
        super(ConvolutionModule, self).__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Layer normalization
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

        # Pointwise convolution 1
        self.pointwise_conv1 = layers.Conv1D(
            filters=2 * channels,
            kernel_size=1,
            padding='same'
        )

        # Depthwise convolution
        self.depthwise_conv = layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            padding='same'
        )

        # Batch normalization
        self.batch_norm = layers.BatchNormalization()

        # Pointwise convolution 2
        self.pointwise_conv2 = layers.Conv1D(
            filters=channels,
            kernel_size=1,
            padding='same'
        )

        # Dropout
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.layer_norm(inputs)
        x = self.pointwise_conv1(x)
        # Apply GLU activation manually
        x1, x2 = tf.split(x, 2, axis=-1)
        x = x1 * tf.nn.sigmoid(x2)
        x = self.depthwise_conv(x)
        x = tf.nn.swish(x)  # Use tf.nn.swish instead of layers.Activation
        x = self.batch_norm(x, training=training)
        x = self.pointwise_conv2(x)
        x = self.dropout(x, training=training)
        return x + inputs  # Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate
        })
        return config


class FeedForwardModule(layers.Layer):
    """Feed-forward module for Conformer architecture."""

    def __init__(self, d_model, d_ff, dropout_rate=0.1, **kwargs):
        super(FeedForwardModule, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = layers.Dense(d_ff)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(d_model)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.layer_norm(inputs)
        x = self.dense1(x)
        x = tf.nn.swish(x)  # Use tf.nn.swish instead of layers.Activation
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return 0.5 * x + inputs  # Residual connection with scaling

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate
        })
        return config


class MultiHeadSelfAttentionModule(layers.Layer):
    """Multi-head self-attention module for Conformer architecture."""

    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super(MultiHeadSelfAttentionModule, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.layer_norm(inputs)
        attn_output = self.multi_head_attention(x, x, training=training)
        attn_output = self.dropout(attn_output, training=training)
        return attn_output + inputs  # Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


class ConformerBlock(layers.Layer):
    """Conformer block combining feed-forward, self-attention, convolution, and feed-forward modules."""

    def __init__(self, d_model, d_ff, num_heads,
                 conv_kernel_size=31, dropout_rate=0.1, **kwargs):
        super(ConformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.conv_kernel_size = conv_kernel_size
        self.dropout_rate = dropout_rate

        # First feed-forward module
        self.ff_module1 = FeedForwardModule(d_model, d_ff, dropout_rate)

        # Multi-head self-attention module
        self.mhsa_module = MultiHeadSelfAttentionModule(
            d_model, num_heads, dropout_rate)

        # Convolution module
        self.conv_module = ConvolutionModule(
            d_model, conv_kernel_size, dropout_rate)

        # Second feed-forward module
        self.ff_module2 = FeedForwardModule(d_model, d_ff, dropout_rate)

        # Final layer normalization
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        # First feed-forward module
        x = self.ff_module1(inputs, training=training)

        # Multi-head self-attention module
        x = self.mhsa_module(x, training=training)

        # Convolution module
        x = self.conv_module(x, training=training)

        # Second feed-forward module
        x = self.ff_module2(x, training=training)

        # Final layer normalization
        x = self.layer_norm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'num_heads': self.num_heads,
            'conv_kernel_size': self.conv_kernel_size,
            'dropout_rate': self.dropout_rate
        })
        return config


def create_conformer_model(input_shape, num_classes=2, d_model=128, d_ff=256, num_heads=8,
                           num_blocks=4, conv_kernel_size=31, dropout_rate=0.1):
    """Create a Conformer model for audio classification."""

    inputs = layers.Input(shape=input_shape)

    # Handle different input shapes
    if len(input_shape) == 3 and input_shape[-1] == 1:
        # Input is (time, freq, 1) - reshape to (time, freq)
        x = layers.Reshape((input_shape[0], input_shape[1]))(inputs)
    elif len(input_shape) == 2:
        # Input is already (time, freq)
        x = inputs
    else:
        # For other shapes, flatten to 2D
        x = layers.Reshape((input_shape[0], -1))(inputs)

    # Initial projection to d_model dimensions
    x = layers.Dense(d_model)(x)

    # Add conformer blocks
    for i in range(num_blocks):
        x = ConformerBlock(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            conv_kernel_size=conv_kernel_size,
            dropout_rate=dropout_rate,
            name=f'conformer_block_{i}'
        )(x)

    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Final classification layer
    outputs, loss = create_classification_head(
        x,
        num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=[d_model // 2]
    )

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='conformer')
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'conformer') -> tf.keras.Model:
    """Factory function to create Conformer models (for compatibility with existing code).

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    if architecture == 'conformer':
        return create_conformer_model(
            input_shape=input_shape,
            num_classes=num_classes,
            d_model=512,
            d_ff=1024,
            num_heads=16,
            num_blocks=12,
            dropout_rate=0.1
        )
    elif architecture == 'conformer_lite':
        return create_conformer_model(
            input_shape=input_shape,
            num_classes=num_classes,
            d_model=256,
            d_ff=512,
            num_heads=8,
            num_blocks=6,
            dropout_rate=0.1
        )
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. Use 'conformer' or 'conformer_lite'")


# Register custom layers for model loading
tf.keras.utils.get_custom_objects().update({
    'ConvolutionModule': ConvolutionModule,
    'FeedForwardModule': FeedForwardModule,
    'MultiHeadSelfAttentionModule': MultiHeadSelfAttentionModule,
    'ConformerBlock': ConformerBlock
})
