"""Spectrogram Transformer Architecture Implementation"""

# Third-party imports
import logging
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from app.domain.models.architectures.layers import (
    ConvolutionStemLayer,
    ResizeLayer,
    STFTLayer,
    ensure_flat_input,
    is_raw_audio,
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
    """
    Patch embedding layer for Spectrogram Transformer.
    Supports overlapping patches as per the AST paper.
    """

    def __init__(self, patch_size: Tuple[int, int], embed_dim: int, stride: Optional[Tuple[int, int]] = None, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride if stride is not None else patch_size

        # Convolutional layer to create patches (with optional overlap)
        self.conv = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=self.stride,
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
            'embed_dim': self.embed_dim,
            'stride': self.stride
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
        attn_output = tf.cast(attn_output, inputs.dtype)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = tf.cast(ffn_output, out1.dtype)
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
    patch_size: Tuple[int, int] = (16, 16),
    stride: Tuple[int, int] = (10, 10),
    embed_dim: int = 768,
    num_blocks: int = 12,
    num_heads: int = 12,
    ff_dim: int = 3072,
    # P1 — retreino obrigatório: o modelo colapsava val→teste (~25 pp) por
    # sobreajuste. Mais regularização (dropout 0.1→0.3, weight_decay 1e-5→1e-4)
    # e LR de pico menor (1e-4→5e-5) reduzem o gap de generalização. Combinado
    # com restauração obrigatória do melhor checkpoint e augmentation SNR.
    dropout_rate: float = 0.3,
    learning_rate: float = 5e-5,
    warmup_steps: int = 2000,
    decay_steps: int = 50000,
    weight_decay: float = 1e-4,
    alpha: float = 1e-7,
    clipnorm: float = 1.0,
    pretrained: bool = False,
    architecture: str = 'spectrogram_transformer'
) -> models.Model:
    """
    Create Audio Spectrogram Transformer (AST) model for audio deepfake detection.
    Topologically aligned with AST (Gong et al., 2021), trained from scratch.

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        patch_size: Size of patches for patch embedding (default: 16x16)
        stride: Stride for overlapping patches (default: 10x10)
        embed_dim: Embedding dimension (ViT-Base: 768)
        num_blocks: Number of transformer blocks (ViT-Base: 12)
        num_heads: Number of attention heads (ViT-Base: 12)
        ff_dim: Feed-forward dimension (ViT-Base: 3072)
        dropout_rate: Dropout rate
        learning_rate: Peak learning rate after warmup
        warmup_steps: Number of linear warmup optimizer steps
        decay_steps: Number of cosine decay optimizer steps
        weight_decay: AdamW weight decay
        alpha: Minimum learning rate used by the warmup/cosine schedule
        pretrained: Reserved metadata flag. This Keras implementation trains
            AST from scratch; passing True logs a warning and does not load
            AudioSet/ImageNet weights.
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    logger.info(
        f"Creating AST model with input_shape={input_shape}, num_classes={num_classes}")
    if pretrained:
        logger.warning(
            "SpectrogramTransformer/AST: pretrained=True solicitado, mas este "
            "caminho Keras nao carrega pesos AudioSet/ImageNet; treinando do zero."
        )

    # Input layer
    inputs = layers.Input(shape=input_shape, name='ast_input')

    # Preprocessing based on input type
    if is_raw_audio(input_shape):
        input_tensor = ensure_flat_input(inputs, input_shape)

        # AST typically uses 128 mel bins, 25ms window, 10ms hop
        # We use STFTLayer with defaults and resize to 128x128 for consistency
        x = STFTLayer(name='stft_layer', add_channel_dim=True)(input_tensor)
        x = ResizeLayer(
            target_height=128,
            target_width=128,
            name='resize_layer')(x)
        processed_height, processed_width = 128, 128
    else:
        # Preprocessing spectrogram input
        x = create_safe_spectrogram_layer(input_shape)(inputs)
        processed_height = max(64, input_shape[0])
        processed_width = max(64, input_shape[1])

    # Patches DIRETO no espectrograma, como no paper AST (Gong et al., 2021:
    # patches 16×16 com stride 10 sobre o espectrograma, SEM conv stem).
    # Antes havia um ConvolutionStemLayer (÷8 espacial) que reduzia (100, 80)
    # a ~12×10 → com patch 8×8/stride 6 sobravam 1×1 = **1 patch** e o
    # Transformer de 12 blocos atendia sobre 2 tokens (CLS+1) — degenerado.
    grid_h = x.shape[1] if x.shape[1] is not None else processed_height
    grid_w = x.shape[2] if x.shape[2] is not None else processed_width

    # Adapta patch/stride para entradas pequenas (garante uma grade real de
    # patches): encolhe à metade enquanto não couberem ≥2 patches por eixo.
    ph, pw = patch_size
    sh, sw = stride
    while ph > 2 and (grid_h - ph) // sh + 1 < 2:
        ph, sh = max(2, ph // 2), max(1, sh // 2)
    while pw > 2 and (grid_w - pw) // sw + 1 < 2:
        pw, sw = max(2, pw // 2), max(1, sw // 2)
    patch_size = (ph, pw)
    stride = (sh, sw)

    # num_patches = floor((input - patch) / stride) + 1 (padding='valid')
    num_patches_h = (grid_h - patch_size[0]) // stride[0] + 1
    num_patches_w = (grid_w - patch_size[1]) // stride[1] + 1
    num_patches = num_patches_h * num_patches_w
    if num_patches < 1:
        raise ValueError(
            f"AST: entrada {grid_h}x{grid_w} pequena demais para patches "
            f"{patch_size} com stride {stride}."
        )

    logger.info(
        f"Using {num_patches} patches ({num_patches_h}x{num_patches_w}) "
        f"patch={patch_size} stride={stride} (direto no espectrograma, sem stem)")

    # Patch embedding (with overlap) on stem output
    x = PatchEmbedding(patch_size, embed_dim, stride=stride, name='patch_embedding')(x)

    # Add class token (Standard ViT/AST)
    x = ClassTokenLayer(embed_dim, name='class_token_layer')(x)

    # Positional encoding (learned)
    x = PositionalEncoding(num_patches + 1, embed_dim, name='pos_encoding')(x)

    # Transformer blocks (Standard ViT-Base)
    for i in range(num_blocks):
        x = SpectrogramTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f'ast_block_{i}'
        )(x)

    # Final LayerNorm before head
    x = layers.LayerNormalization(epsilon=1e-6, name='final_norm')(x)

    # Extract class token for classification (Standard ViT/AST)
    class_token_output = x[:, 0, :]

    # Classification head with residual connections
    # First dense block with skip connection
    skip1 = layers.Dense(1024, name='ast_head_skip1')(class_token_output)
    x = layers.Dense(1024, activation='gelu', name='ast_head_dense')(class_token_output)
    x = layers.Dropout(dropout_rate, name='ast_head_dropout')(x)
    x = layers.Add(name='ast_head_residual1')([x, skip1])

    # Second dense block with skip connection
    skip2 = layers.Dense(256, name='ast_head_skip2')(x)
    x = layers.Dense(256, activation='gelu', name='classifier_dense2')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='classifier_dropout2')(x)
    x = layers.Add(name='ast_head_residual2')([x, skip2])

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

    # Sprint 2.2: WarmupCosineDecay default para Transformers.
    # Warmup linear estabiliza Self-Attention nas primeiras épocas
    # (gradientes grandes) e cosine decay melhora convergência final.
    # P1 — clipnorm=1.0 + warmup maior estabilizam a atenção e evitam o colapso
    # val→0.5 visto no benchmark (treino divergia após o pico do warmup).
    from app.domain.models.training.optimization import create_warmup_cosine_optimizer
    optimizer = create_warmup_cosine_optimizer(
        initial_learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        weight_decay=weight_decay,
        alpha=alpha,
        clipnorm=clipnorm,
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    logger.info(
        f"Spectrogram Transformer model created successfully with {model.count_params()} parameters "
        f"(WarmupCosineDecay: lr={learning_rate}, warmup={warmup_steps}, "
        f"decay={decay_steps}, weight_decay={weight_decay}, alpha={alpha}, "
        f"clipnorm={clipnorm})")
    return model


def create_lightweight_spectrogram_transformer(
    input_shape: Tuple[int, ...],
    num_classes: int,
    architecture: str = 'spectrogram_transformer_lite',
    **kwargs
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
    params = {
        "patch_size": (16, 16),  # Larger patches
        "embed_dim": 128,        # Smaller embedding
        "num_blocks": 4,         # Fewer blocks
        "num_heads": 4,          # Fewer heads
        "ff_dim": 256,           # Smaller FF dimension
        "dropout_rate": 0.1,
    }
    params.update(kwargs)
    return create_spectrogram_transformer_model(
        input_shape=input_shape,
        num_classes=num_classes,
        architecture=architecture,
        **params
    )


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'spectrogram_transformer',
                 **kwargs) -> models.Model:
    """
    Factory function to create Spectrogram Transformer models (for compatibility with existing code).

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    # Alias "default" -> AST topologicamente alinhado ao artigo, treinado do zero.
    if architecture == 'default':
        architecture = 'spectrogram_transformer'

    if architecture == 'spectrogram_transformer':
        return create_spectrogram_transformer_model(
            input_shape, num_classes, architecture=architecture, **kwargs)
    elif architecture == 'spectrogram_transformer_lite':
        return create_lightweight_spectrogram_transformer(
            input_shape, num_classes, architecture=architecture, **kwargs)
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. Use 'spectrogram_transformer' or 'spectrogram_transformer_lite'")


# Register custom layers and functions for model loading
tf.keras.utils.get_custom_objects().update({
    'PatchEmbedding': PatchEmbedding,
    'ClassTokenLayer': ClassTokenLayer,
    'PositionalEncoding': PositionalEncoding,
    'SpectrogramTransformerBlock': SpectrogramTransformerBlock,
    'SpectralAttentionPooling': SpectralAttentionPooling,
    'ResizeLayer': ResizeLayer,
    'SafeSpectrogramReshapeLayer': SafeSpectrogramReshapeLayer,
    'STFTLayer': STFTLayer
})
