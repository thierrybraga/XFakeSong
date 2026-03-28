"""Hybrid CNN-Transformer Architecture — Compact Convolutional Transformer (CCT)

Literature-based implementation combining:

1. Bartusiak & Delp (2022) — "Synthesized Speech Detection Using Convolutional
   Transformer-Based Spectrogram Analysis" (arXiv:2205.01800, IEEE Asilomar 2021)
   - CCT applied to spectrograms for synthesized speech detection
   - Outperforms CNN, LSTM, and conventional approaches on ASVspoof 2019
   - Convolutional tokenizer + Transformer encoder on spectrogram images

2. Hassani et al. (2021) — "Escaping the Big Data Paradigm with Compact Transformers"
   (arXiv:2104.05704)
   - CCT architecture: Conv tokenizer replaces patch embedding
   - Sequence pooling replaces CLS token
   - Stochastic depth regularization
   - 12x more parameter-efficient than ViT

3. Kadam et al. (2025) — "Deepfake Audio Detection Using CNN-Transformer Hybrid
   Model with Data Augmentation" (J. Propulsion Technology, Vol. 46 No. 3)
   - CNN-Transformer hybrid on mel spectrogram (128 bins)
   - 91.47% accuracy on ASVspoof 2019

Architecture (CCT-2/3x2 adapted for audio):
- Input: raw audio -> mel spectrogram (n_fft=1024, hop=160, n_mels=128)
- Conv Tokenizer: 2x [Conv2D + ReLU + MaxPool(3, stride 2)]
  - Layer 1: 64 filters, 3x3, stride 1
  - Layer 2: 128 filters, 3x3, stride 1
- Flatten spatial dims -> sequence of tokens
- Learned positional embeddings (optional per CCT)
- Transformer Encoder: 4 layers, 4 heads, 256 proj dim
  - Pre-norm (LayerNorm before attention)
  - FFN: Dense(128, GELU) -> Dense(128)
  - Stochastic depth regularization
- Sequence Pooling: attention-weighted sum (replaces CLS token)
- Dense(num_classes) classification
- Optimizer: AdamW(lr=0.001, weight_decay=0.0001)
"""

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from app.domain.models.architectures.layers import (
    SqueezeExcitationBlock2D,
    ensure_flat_input,
    is_raw_audio,
)

logger = logging.getLogger(__name__)


# ============================ CUSTOM LAYERS ============================

class MelSpectrogramLayer(layers.Layer):
    """Mel spectrogram extraction for CNN-Transformer input.

    Params per Kadam et al. (2025): 128 mel bins, 16kHz.
    STFT params: 1024 FFT, 160 hop (10ms at 16kHz).
    """

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=160,
                 n_mels=128, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def call(self, inputs):
        if len(inputs.shape) == 3 and inputs.shape[-1] == 1:
            inputs = tf.squeeze(inputs, axis=-1)
        stft = tf.signal.stft(
            inputs,
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft
        )
        magnitude = tf.abs(stft)
        mel_weight = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=self.sample_rate / 2.0
        )
        mel_spec = tf.matmul(tf.square(magnitude), mel_weight)
        log_mel = tf.math.log(mel_spec + 1e-6)
        return log_mel

    def get_config(self):
        config = super().get_config()
        config.update({
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
        })
        return config


class CCTTokenizer(layers.Layer):
    """Convolutional Tokenizer per Hassani et al. (2021).

    Replaces ViT patch embedding with conv layers.
    2x [Conv2D(k=3, s=1, ReLU) + MaxPool(3, stride 2)]
    Filters: [64, 128]

    Output: (batch, seq_len, channels) where seq_len = flattened spatial dims.
    """

    def __init__(self, num_output_channels=None, kernel_size=3, stride=1,
                 pooling_kernel_size=3, pooling_stride=2, **kwargs):
        super().__init__(**kwargs)
        if num_output_channels is None:
            num_output_channels = [64, 128]
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_stride = pooling_stride

        self.conv_layers = []
        for i, out_ch in enumerate(num_output_channels):
            self.conv_layers.append(
                layers.Conv2D(
                    out_ch, kernel_size, strides=stride,
                    padding='same', use_bias=False,
                    activation='relu',
                    kernel_initializer='he_normal',
                    name=f'conv_{i}'
                )
            )
            self.conv_layers.append(
                SqueezeExcitationBlock2D(
                    reduction=16,
                    name=f'se_block_{i}'
                )
            )
            self.conv_layers.append(
                layers.MaxPooling2D(
                    pool_size=pooling_kernel_size,
                    strides=pooling_stride,
                    padding='same',
                    name=f'pool_{i}'
                )
            )

    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        # Flatten spatial dims -> (batch, seq_len, channels)
        batch_size = tf.shape(x)[0]
        seq_len = x.shape[1] * x.shape[2] if x.shape[1] is not None and x.shape[2] is not None else tf.shape(x)[1] * tf.shape(x)[2]
        channels = x.shape[-1]
        x = tf.reshape(x, [batch_size, -1, channels])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_output_channels': self.num_output_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'pooling_kernel_size': self.pooling_kernel_size,
            'pooling_stride': self.pooling_stride,
        })
        return config


class StochasticDepth(layers.Layer):
    """Stochastic Depth regularization per Huang et al. (2016).

    During training, randomly drops the entire residual branch
    with probability drop_prob.
    """

    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, inputs, training=None):
        if training and self.drop_prob > 0:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(inputs)[0],) + (1,) * (len(inputs.shape) - 1)
            random_tensor = tf.random.uniform(shape, dtype=inputs.dtype)
            random_tensor = tf.floor(random_tensor + keep_prob)
            return (inputs / keep_prob) * random_tensor
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'drop_prob': self.drop_prob})
        return config


class TransformerBlock(layers.Layer):
    """Pre-norm Transformer encoder block per CCT.

    Architecture:
    - LayerNorm -> MultiHeadAttention -> StochasticDepth -> Add
    - LayerNorm -> FFN(GELU) -> StochasticDepth -> Add

    FFN: Dense(proj_dim, GELU) -> Dropout -> Dense(proj_dim) -> Dropout
    """

    def __init__(self, projection_dim, num_heads=2, dropout_rate=0.1,
                 stochastic_depth_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=dropout_rate
        )
        self.stochastic_depth1 = StochasticDepth(stochastic_depth_rate)

        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn_dense1 = layers.Dense(projection_dim, activation='gelu')
        self.ffn_dropout1 = layers.Dropout(dropout_rate)
        self.ffn_dense2 = layers.Dense(projection_dim)
        self.ffn_dropout2 = layers.Dropout(dropout_rate)
        self.stochastic_depth2 = StochasticDepth(stochastic_depth_rate)

    def call(self, inputs, training=None):
        # Pre-norm attention
        x1 = self.norm1(inputs)
        attn_out = self.attn(x1, x1, training=training)
        attn_out = self.stochastic_depth1(attn_out, training=training)
        x2 = inputs + attn_out

        # Pre-norm FFN
        x3 = self.norm2(x2)
        x3 = self.ffn_dense1(x3)
        x3 = self.ffn_dropout1(x3, training=training)
        x3 = self.ffn_dense2(x3)
        x3 = self.ffn_dropout2(x3, training=training)
        x3 = self.stochastic_depth2(x3, training=training)
        return x2 + x3

    def get_config(self):
        config = super().get_config()
        config.update({
            'projection_dim': self.projection_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'stochastic_depth_rate': self.stochastic_depth_rate,
        })
        return config


class PositionalEmbeddingLayer(layers.Layer):
    """Learned positional embeddings per CCT (Hassani et al., 2021).

    Adds trainable positional embeddings (Glorot uniform init)
    to the token sequence. Handles dynamic sequence lengths.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        seq_len = input_shape[1]
        feature_dim = input_shape[2]
        if seq_len is not None:
            self.pos_emb = self.add_weight(
                name='pos_emb',
                shape=(1, seq_len, feature_dim),
                initializer='glorot_uniform',
                trainable=True
            )
            self.static_seq_len = True
        else:
            self.static_seq_len = False

    def call(self, inputs):
        if self.static_seq_len:
            return inputs + self.pos_emb
        return inputs

    def get_config(self):
        return super().get_config()


class SequencePooling(layers.Layer):
    """Attention-weighted sequence pooling per Hassani et al. (2021).

    Replaces CLS token with learned weighted sum:
    - attention_weights = softmax(Dense(1)(x))
    - output = sum(attention_weights * x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_dense = layers.Dense(1)

    def call(self, inputs):
        attention_weights = tf.nn.softmax(self.attention_dense(inputs), axis=1)
        # (batch, seq, 1)^T @ (batch, seq, dim) -> (batch, 1, dim)
        weighted = tf.matmul(
            tf.transpose(attention_weights, perm=[0, 2, 1]),
            inputs
        )
        # (batch, dim)
        return tf.squeeze(weighted, axis=1)

    def get_config(self):
        return super().get_config()


# ============================ MODEL BUILDERS ============================

def _create_cct_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    projection_dim: int = 256,
    num_heads: int = 4,
    transformer_layers: int = 4,
    conv_channels: list = None,
    dropout_rate: float = 0.1,
    stochastic_depth_rate: float = 0.1,
    use_positional_emb: bool = True,
    architecture: str = 'hybrid_cnn_transformer'
) -> models.Model:
    """Create CCT model for audio deepfake detection.

    Architecture per Hassani et al. (2021) adapted for audio per
    Bartusiak & Delp (2022):
    1. Mel spectrogram front-end (if raw audio)
    2. CCT Tokenizer: 2x Conv2D([64,128], 3x3, ReLU) + MaxPool(3, stride 2)
    3. Learned positional embeddings
    4. Transformer Encoder: pre-norm, stochastic depth
    5. Sequence Pooling (attention-weighted)
    6. Dense classification
    """
    if conv_channels is None:
        conv_channels = [64, 128]

    inputs = layers.Input(shape=input_shape, name='audio_input')

    # ---------- Front-end: audio -> spectrogram ----------
    if is_raw_audio(input_shape):
        audio = ensure_flat_input(inputs, input_shape)
        # Mel spectrogram (128 bins per Kadam et al.)
        x = MelSpectrogramLayer(
            sample_rate=16000, n_fft=1024, hop_length=160, n_mels=128,
            name='mel_spectrogram'
        )(audio)
        # Add channel dim: (batch, time, 128) -> (batch, time, 128, 1)
        x = layers.Reshape(
            target_shape=(x.shape[1], 128, 1) if x.shape[1] is not None
            else (-1, 128, 1),
            name='add_channel'
        )(x) if x.shape[1] is not None else layers.Lambda(
            lambda t: tf.expand_dims(t, axis=-1), name='add_channel'
        )(x)
    else:
        x = inputs
        if len(input_shape) == 2:
            # (time, features) -> (time, features, 1)
            x = layers.Reshape((*input_shape, 1), name='add_channel')(x)
        # Already 3D (time, features, channels) -> OK

    # ---------- CCT Tokenizer ----------
    # 2x Conv2D + MaxPool per Hassani et al.
    tokens = CCTTokenizer(
        num_output_channels=conv_channels,
        kernel_size=3, stride=1,
        pooling_kernel_size=3, pooling_stride=2,
        name='cct_tokenizer'
    )(x)

    # ---------- Projection to transformer dim ----------
    tokens = layers.Dense(projection_dim, name='token_projection')(tokens)

    # ---------- Positional embeddings ----------
    if use_positional_emb:
        tokens = PositionalEmbeddingLayer(name='positional_embedding')(tokens)

    # ---------- Transformer Encoder ----------
    # Stochastic depth: linear schedule from 0 to stochastic_depth_rate
    dpr = np.linspace(0, stochastic_depth_rate, transformer_layers).tolist()

    for i in range(transformer_layers):
        tokens = TransformerBlock(
            projection_dim=projection_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            stochastic_depth_rate=dpr[i],
            name=f'transformer_block_{i}'
        )(tokens)

    # Final LayerNorm
    tokens = layers.LayerNormalization(epsilon=1e-5, name='final_norm')(tokens)

    # ---------- Sequence Pooling ----------
    representation = SequencePooling(name='sequence_pooling')(tokens)

    # ---------- Classification ----------
    representation = layers.Dropout(dropout_rate, name='head_dropout')(representation)

    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(representation)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(representation)
        loss = 'sparse_categorical_crossentropy'

    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    # AdamW per CCT paper
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4
        ),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(f"CCT model created: transformer_layers={transformer_layers}, heads={num_heads}, dim={projection_dim}, params={model.count_params()}")
    return model


def _create_cct_lite(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    architecture: str = 'hybrid_cnn_transformer_lite'
) -> models.Model:
    """Lightweight CCT variant.

    Fewer conv filters, single transformer layer, smaller dim.
    """
    return _create_cct_model(
        input_shape=input_shape,
        num_classes=num_classes,
        projection_dim=64,
        num_heads=2,
        transformer_layers=1,
        conv_channels=[32, 64],
        dropout_rate=0.1,
        stochastic_depth_rate=0.0,
        use_positional_emb=True,
        architecture=architecture
    )


# ============================ FACTORY ============================

def create_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    architecture: str = 'hybrid_cnn_transformer',
    **kwargs
) -> models.Model:
    """Factory function for CCT-based hybrid CNN-Transformer models.

    Variants:
        'hybrid_cnn_transformer': Full CCT (2x conv [64,128] + 2x Transformer)
        'hybrid_cnn_transformer_lite': Lightweight (2x conv [32,64] + 1x Transformer)
    """
    if architecture == 'hybrid_cnn_transformer_lite':
        return _create_cct_lite(input_shape, num_classes, architecture)
    else:
        return _create_cct_model(
            input_shape=input_shape,
            num_classes=num_classes,
            architecture=architecture,
            **kwargs
        )


# Register custom layers
tf.keras.utils.get_custom_objects().update({
    'MelSpectrogramLayer': MelSpectrogramLayer,
    'CCTTokenizer': CCTTokenizer,
    'StochasticDepth': StochasticDepth,
    'TransformerBlock': TransformerBlock,
    'PositionalEmbeddingLayer': PositionalEmbeddingLayer,
    'SequencePooling': SequencePooling,
})
