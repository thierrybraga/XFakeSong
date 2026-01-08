"""Spectrogram Transformer Architecture Implementation"""

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import logging
import math
from app.domain.models.architectures.layers import (
    STFTLayer, ExpandDimsLayer, ResizeLayer, is_raw_audio, ensure_flat_input
)

logger = logging.getLogger(__name__)


class SafeSpectrogramReshapeLayer(layers.Layer):
    """Layer to safely reshape spectrogram inputs."""

    def __init__(self, input_shape_tuple, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_tuple = input_shape_tuple

    def call(self, x):
        input_shape = self.input_shape_tuple

        if len(input_shape) == 2:
            # (time, features) -> (time, features, 1)
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
                # Multiple channels, keep as is but ensure minimum size
                target_height = tf.maximum(64, input_shape[0])
                target_width = tf.maximum(64, input_shape[1])
                x = tf.image.resize(x, (target_height, target_width))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'input_shape_tuple': self.input_shape_tuple})
        return config


def create_safe_spectrogram_layer(input_shape):
    """Creates a SafeSpectrogramReshapeLayer."""
    return SafeSpectrogramReshapeLayer(
        input_shape, name='safe_spectrogram_reshape')


class ClassTokenLayer(layers.Layer):
    """Custom layer to add class token to patch embeddings."""

    def __init__(self, embed_dim, **kwargs):
        super(ClassTokenLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, self.embed_dim),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_tokens = tf.tile(self.class_token, [batch_size, 1, 1])
        return tf.concat([class_tokens, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim})
        return config


class PatchEmbedding(layers.Layer):
    """Patch embedding layer for Spectrogram Transformer."""

    def __init__(self, patch_size: Tuple[int, int], embed_dim: int, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Convolutional layer to create patches
        self.conv = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            name='patch_conv'
        )

    def call(self, inputs):
        # inputs shape: (batch, height, width, channels)
        x = self.conv(inputs)
        # Reshape to (batch, num_patches, embed_dim)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embed_dim])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim
        })
        return config


class PositionalEncoding(layers.Layer):
    """Learnable positional encoding for patches."""

    def __init__(self, max_patches: int, embed_dim: int, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_patches = max_patches
        self.embed_dim = embed_dim

        # Learnable positional embeddings
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, max_patches, embed_dim),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Take only the needed positional embeddings
        pos_emb = self.pos_embedding[:, :seq_len, :]

        return inputs + pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_patches': self.max_patches,
            'embed_dim': self.embed_dim
        })
        return config


class SpectrogramTransformerBlock(layers.Layer):
    """Transformer block optimized for spectrogram analysis."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout_rate: float = 0.1, **kwargs):
        super(SpectrogramTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )

        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])

        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        # Self-attention with residual connection
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class SpectralAttentionPooling(layers.Layer):
    """Attention-based pooling specifically designed for spectral features."""

    def __init__(self, embed_dim: int, **kwargs):
        super(SpectralAttentionPooling, self).__init__(**kwargs)
        self.embed_dim = embed_dim

        # Attention mechanism for pooling
        self.attention_weights = layers.Dense(1, activation='tanh')

    def call(self, inputs):
        # inputs shape: (batch, num_patches, embed_dim)

        # Calculate attention scores
        attention_scores = self.attention_weights(
            inputs)  # (batch, num_patches, 1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)

        # Apply attention weights
        attended_features = tf.reduce_sum(inputs * attention_weights, axis=1)

        return attended_features

    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim})
        return config


def create_spectrogram_transformer_model(
    input_shape: Tuple[int, ...],
    num_classes: int,
    patch_size: Tuple[int, int] = (8, 8),
    embed_dim: int = 256,
    num_blocks: int = 8,
    num_heads: int = 8,
    ff_dim: int = 512,
    dropout_rate: float = 0.1,
    architecture: str = 'spectrogram_transformer'
) -> models.Model:
    """
    Create Spectrogram Transformer model for audio deepfake detection.

    Args:
        input_shape: Shape of input features (samples,), (height, width, channels) or (time_steps, features)
        num_classes: Number of output classes
        patch_size: Size of patches for patch embedding
        embed_dim: Embedding dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout_rate: Dropout rate
        architecture: Architecture name (for compatibility)

    Returns:
        Compiled Keras model
    """
    logger.info(
        f"Creating Spectrogram Transformer model with input_shape={input_shape}, num_classes={num_classes}")

    # Input layer
    inputs = layers.Input(shape=input_shape, name='spect_transformer_input')

    # Preprocessing based on input type
    if is_raw_audio(input_shape):
        input_tensor = ensure_flat_input(inputs, input_shape)

        # Convert to spectrogram using custom STFT layer
        x = STFTLayer(name='stft_layer', add_channel_dim=True)(input_tensor)
        # Ensure minimum dimensions for patch embedding
        x = ResizeLayer(
            target_height=64,
            target_width=64,
            name='resize_layer')(x)
    else:
        # Preprocessing
        x = create_safe_spectrogram_layer(input_shape)(inputs)

    # Get the processed shape for patch calculation
    if is_raw_audio(input_shape):
        # After STFT and resize, we have a fixed 64x64 spectrogram
        processed_height = 64
        processed_width = 64
        channels = 1
    elif len(input_shape) == 2:
        processed_height = max(64, input_shape[0])
        processed_width = max(64, input_shape[1])
        channels = 1
    else:  # len(input_shape) == 3
        processed_height = max(64, input_shape[0])
        processed_width = max(64, input_shape[1])
        channels = input_shape[2] if len(input_shape) == 3 else 1

    # Calculate number of patches
    num_patches_h = processed_height // patch_size[0]
    num_patches_w = processed_width // patch_size[1]
    num_patches = num_patches_h * num_patches_w

    logger.info(
        f"Using {num_patches} patches ({num_patches_h}x{num_patches_w}) with patch size {patch_size}")

    # Patch embedding
    x = PatchEmbedding(patch_size, embed_dim, name='patch_embedding')(x)

    # Add class token
    x = ClassTokenLayer(embed_dim, name='class_token_layer')(x)

    # Positional encoding
    x = PositionalEncoding(num_patches + 1, embed_dim, name='pos_encoding')(x)

    # Transformer blocks
    for i in range(num_blocks):
        x = SpectrogramTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f'transformer_block_{i}'
        )(x)

    # Extract class token for classification
    class_token_output = x[:, 0, :]  # First token is class token

    # Alternative: Use spectral attention pooling on all tokens
    # pooled_output = SpectralAttentionPooling(embed_dim, name='spectral_pooling')(x[:, 1:, :])
    # combined_output = tf.concat([class_token_output, pooled_output], axis=-1)

    # Classification head
    x = layers.LayerNormalization(
        epsilon=1e-6, name='final_norm')(class_token_output)

    x = layers.Dense(512, activation='gelu', name='classifier_dense1')(x)
    x = layers.Dropout(dropout_rate, name='classifier_dropout1')(x)

    x = layers.Dense(256, activation='gelu', name='classifier_dense2')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='classifier_dropout2')(x)

    # Output layer
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(
            num_classes,
            activation='softmax',
            name='output')(x)
        loss = 'sparse_categorical_crossentropy'

    # Create model
    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name='spectrogram_transformer_model')

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=1e-5,
            beta_1=0.9,
            beta_2=0.999
        ),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(
        f"Spectrogram Transformer model created successfully with {
            model.count_params()} parameters")
    return model


def create_lightweight_spectrogram_transformer(
    input_shape: Tuple[int, ...],
    num_classes: int,
    architecture: str = 'spectrogram_transformer_lite'
) -> models.Model:
    """
    Create a lightweight version of Spectrogram Transformer for faster inference.

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    return create_spectrogram_transformer_model(
        input_shape=input_shape,
        num_classes=num_classes,
        patch_size=(16, 16),  # Larger patches
        embed_dim=128,        # Smaller embedding
        num_blocks=4,         # Fewer blocks
        num_heads=4,          # Fewer heads
        ff_dim=256,           # Smaller FF dimension
        dropout_rate=0.1,
        architecture=architecture
    )


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'spectrogram_transformer') -> models.Model:
    """
    Factory function to create Spectrogram Transformer models (for compatibility with existing code).

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    if architecture == 'spectrogram_transformer':
        return create_spectrogram_transformer_model(
            input_shape, num_classes, architecture=architecture)
    elif architecture == 'spectrogram_transformer_lite':
        return create_lightweight_spectrogram_transformer(
            input_shape, num_classes, architecture=architecture)
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. Use 'spectrogram_transformer' or 'spectrogram_transformer_lite'")


# Register custom layers and functions for model loading
tf.keras.utils.get_custom_objects().update({
    'PatchEmbedding': PatchEmbedding,
    'PositionalEncoding': PositionalEncoding,
    'SpectrogramTransformerBlock': SpectrogramTransformerBlock,
    'SpectralAttentionPooling': SpectralAttentionPooling,
    'ResizeLayer': ResizeLayer,
    'SafeSpectrogramReshapeLayer': SafeSpectrogramReshapeLayer,
    'STFTLayer': STFTLayer
})
