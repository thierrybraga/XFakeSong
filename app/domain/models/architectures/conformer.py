"""Conformer Architecture Implementation

Paper-faithful implementation of:
"Conformer: Convolution-augmented Transformer for Speech Processing"
(Gulati et al., Interspeech 2020)

Key components from the paper:
- Convolutional Subsampling (4x reduction before Conformer blocks)
- Relative Positional Encoding (sinusoidal, used in attention)
- Conformer Block: FF(0.5) → MHSA(relative) → Conv → FF(0.5) → LayerNorm
- ConvModule order: LN → PW → GLU → DW → BN → Swish → PW → Dropout

Configuração ÚNICA (Conformer-M da Tabela 1 do paper): d_model=256,
16 blocos, 4 cabeças, d_ff=1024, kernel depthwise 31, P_drop=0.1.
"""

import logging
import math
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from app.domain.models.architectures.layers import (
    LogMelSpectrogramLayer,
    SparseLabelSmoothingCrossEntropy,
    create_classification_head,
)

logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RelativePositionalEncoding(layers.Layer):
    """Relative sinusoidal positional encoding (Transformer-XL style).

    Generates position encodings for relative attention, where the encoding
    depends on the distance between positions rather than absolute position.

    Reference: Dai et al., "Transformer-XL: Attentive Language Models Beyond
    a Fixed-Length Context", ACL 2019
    """

    def __init__(self, d_model, max_len=5000, **kwargs):
        super(RelativePositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

    def build(self, input_shape):
        # Pre-compute positional encodings for efficiency
        # We need encodings for positions from -(max_len-1) to +(max_len-1)
        # But we compute for 0 to max_len and use indexing tricks in attention
        pe = np.zeros((self.max_len, self.d_model), dtype=np.float32)
        position = np.arange(0, self.max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32) *
            -(math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Shape: (1, max_len, d_model) - registered as non-trainable weight
        # Em Keras 3, tensores criados em build() via tf.constant ficam "out of
        # scope" no call() do modelo funcional. Usar add_weight com
        # trainable=False resolve, pois o tensor passa a ser uma Variable
        # rastreável pelo grafo.
        self.pe = self.add_weight(
            name="pe",
            shape=(1, self.max_len, self.d_model),
            initializer=tf.keras.initializers.Constant(pe[np.newaxis, :, :]),
            trainable=False,
            dtype="float32",
        )

        super(RelativePositionalEncoding, self).build(input_shape)

    def call(self, inputs):
        """Returns positional encoding for the input sequence length.

        Args:
            inputs: (batch, seq_len, d_model)

        Returns:
            pos_enc: (1, seq_len, d_model) — posições em ordem DECRESCENTE
            (distância relativa seq_len-1 → 0).
        """
        seq_len = tf.shape(inputs)[1]
        # CORREÇÃO: a docstring sempre afirmou "reversed (from seq_len-1 to 0)",
        # mas o código devolvia `pe[:, :seq_len, :]` em ordem CRESCENTE. Na
        # atenção relativa do Transformer-XL (Dai et al., 2019), adotada pelo
        # Conformer (§2.2), R é indexado por DISTÂNCIA relativa e a sequência
        # posicional é construída de forma decrescente; combinada com o
        # `_relative_shift`, a ordem crescente mapeava o viés posicional para
        # distâncias erradas — o termo conteúdo↔posição, que é justamente o
        # diferencial do Conformer, ficava desalinhado.
        return tf.cast(self.pe[:, :seq_len, :][:, ::-1, :], inputs.dtype)

    def get_config(self):
        config = super(RelativePositionalEncoding, self).get_config()
        config.update({
            'd_model': self.d_model,
            'max_len': self.max_len
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RelativeMultiHeadAttention(layers.Layer):
    """Multi-head attention with relative positional encoding.

    Implements the attention mechanism from Conformer which uses relative
    positional bias in the attention score computation:

        Attention = softmax((QK^T + QR^T + content_bias + pos_bias) / √d_k) · V

    Where:
        - QK^T: content-to-content attention
        - QR^T: content-to-position attention
        - content_bias (u): global content bias
        - pos_bias (v): global position bias

    Reference: Gulati et al., Section 2.2 + Shaw et al., "Self-Attention with
    Relative Position Representations", NAACL 2018
    """

    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super(RelativeMultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate
        self.scale = 1.0 / math.sqrt(self.d_k)

    def build(self, input_shape):
        # Linear projections for Q, K, V, O
        self.W_q = self.add_weight(
            name="W_q", shape=(self.d_model, self.d_model),
            initializer="glorot_uniform", trainable=True)
        self.W_k = self.add_weight(
            name="W_k", shape=(self.d_model, self.d_model),
            initializer="glorot_uniform", trainable=True)
        self.W_v = self.add_weight(
            name="W_v", shape=(self.d_model, self.d_model),
            initializer="glorot_uniform", trainable=True)
        self.W_o = self.add_weight(
            name="W_o", shape=(self.d_model, self.d_model),
            initializer="glorot_uniform", trainable=True)

        # Projection for positional encoding
        self.W_pos = self.add_weight(
            name="W_pos", shape=(self.d_model, self.d_model),
            initializer="glorot_uniform", trainable=True)

        # Global content bias (u) and position bias (v) per head
        # Shape: (num_heads, d_k)
        self.u_bias = self.add_weight(
            name="u_bias", shape=(self.num_heads, self.d_k),
            initializer="zeros", trainable=True)
        self.v_bias = self.add_weight(
            name="v_bias", shape=(self.num_heads, self.d_k),
            initializer="zeros", trainable=True)

        self.dropout = layers.Dropout(self.dropout_rate)

        super(RelativeMultiHeadAttention, self).build(input_shape)

    def _relative_shift(self, x):
        """Compute relative position attention scores via efficient shifting.

        Converts the (batch, heads, seq_len, seq_len) position attention matrix
        computed from (Q @ R^T) into proper relative position indices.
        """
        batch_size = tf.shape(x)[0]
        num_heads = tf.shape(x)[1]
        seq_len = tf.shape(x)[2]
        pos_len = tf.shape(x)[3]

        # Pad left column with zeros
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])

        # Reshape and slice to get relative positions
        x = tf.reshape(x, (batch_size, num_heads, pos_len + 1, seq_len))
        x = x[:, :, 1:, :]  # Remove first row (the padding)

        # Reshape back
        x = tf.reshape(x, (batch_size, num_heads, seq_len, pos_len))

        # Take only valid positions (up to seq_len)
        x = x[:, :, :, :seq_len]

        return x

    def call(self, inputs, pos_enc, training=None):
        """
        Args:
            inputs: (batch, seq_len, d_model) - input features
            pos_enc: (1, seq_len, d_model) - positional encoding
            training: bool

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Linear projections
        Q = tf.matmul(inputs, self.W_q)   # (batch, seq_len, d_model)
        K = tf.matmul(inputs, self.W_k)
        V = tf.matmul(inputs, self.W_v)
        pos_enc = tf.cast(pos_enc, inputs.dtype)
        P = tf.matmul(pos_enc, self.W_pos)  # (1, seq_len, d_model)

        # Reshape to (batch, seq_len, num_heads, d_k) then transpose to (batch, heads, seq_len, d_k)
        Q = tf.transpose(tf.reshape(Q, (batch_size, seq_len, self.num_heads, self.d_k)), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, (batch_size, seq_len, self.num_heads, self.d_k)), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, (batch_size, seq_len, self.num_heads, self.d_k)), [0, 2, 1, 3])

        # Position encoding: (1, seq_len, num_heads, d_k) -> (1, heads, seq_len, d_k)
        P = tf.transpose(tf.reshape(P, (1, seq_len, self.num_heads, self.d_k)), [0, 2, 1, 3])

        # Content-to-content: (Q + u) @ K^T
        # u_bias: (heads, d_k) -> (1, heads, 1, d_k)
        content_score = tf.matmul(
            Q + self.u_bias[tf.newaxis, :, tf.newaxis, :],
            tf.transpose(K, [0, 1, 3, 2])
        )  # (batch, heads, seq_len, seq_len)

        # Content-to-position: (Q + v) @ P^T
        pos_score = tf.matmul(
            Q + self.v_bias[tf.newaxis, :, tf.newaxis, :],
            tf.transpose(P, [0, 1, 3, 2])
        )  # (batch, heads, seq_len, seq_len)

        # Apply relative shift to position scores
        pos_score = self._relative_shift(pos_score)

        # Combined attention scores
        scores = (content_score + pos_score) * self.scale
        attn_weights = tf.nn.softmax(scores, axis=-1)

        if training:
            attn_weights = self.dropout(attn_weights, training=training)

        # Apply attention to values
        context = tf.matmul(attn_weights, V)  # (batch, heads, seq_len, d_k)

        # Reshape back: (batch, seq_len, d_model)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, seq_len, self.d_model))

        # Output projection
        output = tf.matmul(context, self.W_o)

        return tf.cast(output, inputs.dtype)

    def get_config(self):
        config = super(RelativeMultiHeadAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ConvSubsampling(layers.Layer):
    """Convolutional subsampling layer (4x temporal reduction).

    Applies two Conv2D layers with stride 2 each, reducing the input
    sequence length by a factor of 4 before the Conformer blocks.
    This is critical for computational efficiency on long sequences.

    Reference: Gulati et al., Section 2.1
    """

    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        super(ConvSubsampling, self).__init__(**kwargs)
        self.d_model = d_model
        # Antes o dropout do subsampling era fixo em 0.1 e ignorava o
        # `dropout_rate` do modelo — o P_drop do paper (Gulati et al., §2.1)
        # vale para o encoder inteiro, subsampling incluído.
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        import math
        # Two Conv2D layers with stride 2 for 4x subsampling
        self.conv1 = layers.Conv2D(
            filters=self.d_model, kernel_size=(3, 3), strides=(2, 2),
            padding='same', activation='relu', name=self.name + "_conv1")
        self.conv2 = layers.Conv2D(
            filters=self.d_model, kernel_size=(3, 3), strides=(2, 2),
            padding='same', activation='relu', name=self.name + "_conv2")

        # Linear projection to d_model after flattening freq * channels
        self.linear = layers.Dense(self.d_model, name=self.name + "_linear")
        self.dropout = layers.Dropout(self.dropout_rate)

        # Pre-build Dense with static shape so Keras can trace output dimensions
        freq_dim = input_shape[-1] if len(input_shape) >= 3 else None
        if freq_dim is not None:
            freq_out = math.ceil(math.ceil(freq_dim / 2) / 2)
            flat_dim = freq_out * self.d_model
            self.linear.build((None, None, flat_dim))

        super(ConvSubsampling, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch, time, freq) - spectrogram features

        Returns:
            output: (batch, time//4, d_model)
        """
        # Add channel dim: (batch, time, freq, 1)
        x = tf.expand_dims(inputs, axis=-1)

        # Conv2D subsampling: each reduces by 2x
        x = self.conv1(x)   # (batch, time//2, freq//2, d_model)
        x = self.conv2(x)   # (batch, time//4, freq//4, d_model)

        # Flatten freq and channel dims using static shape where possible
        static = x.shape.as_list()  # [None, None, freq//4, d_model]
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        if static[2] is not None and static[3] is not None:
            flat_dim = static[2] * static[3]
            x = tf.reshape(x, [batch_size, time_steps, flat_dim])
        else:
            x = tf.reshape(x, [batch_size, time_steps, -1])

        # Project to d_model
        x = self.linear(x)
        x = self.dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        import math
        batch = input_shape[0]
        time = input_shape[1] if len(input_shape) > 1 else None
        if time is not None:
            time_out = math.ceil(math.ceil(time / 2) / 2)
        else:
            time_out = None
        return (batch, time_out, self.d_model)

    def get_config(self):
        config = super(ConvSubsampling, self).get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ConvolutionModule(layers.Layer):
    """Convolution module for Conformer architecture.

    Order (paper-faithful):
        LayerNorm → Pointwise Conv (2×channels) → GLU
        → Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout + Residual

    Reference: Gulati et al., Section 2.3
    """

    def __init__(self, channels, kernel_size=31, dropout_rate=0.1, **kwargs):
        super(ConvolutionModule, self).__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Layer normalization
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

        # Pointwise convolution 1 (expansion to 2×channels for GLU)
        self.pointwise_conv1 = layers.Conv1D(
            filters=2 * channels, kernel_size=1, padding='same')

        # Depthwise convolution
        self.depthwise_conv = layers.DepthwiseConv1D(
            kernel_size=kernel_size, padding='same')

        # Batch normalization (BEFORE Swish, per paper)
        self.batch_norm = layers.BatchNormalization()

        # Pointwise convolution 2 (projection back to channels)
        self.pointwise_conv2 = layers.Conv1D(
            filters=channels, kernel_size=1, padding='same')

        # Dropout
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.layer_norm(inputs)
        x = self.pointwise_conv1(x)

        # GLU activation: split and gate
        x1, x2 = tf.split(x, 2, axis=-1)
        x = x1 * tf.nn.sigmoid(x2)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x, training=training)  # BatchNorm BEFORE Swish (paper order)
        x = tf.nn.swish(x)                         # Swish AFTER BatchNorm
        x = self.pointwise_conv2(x)
        x = self.dropout(x, training=training)

        return tf.cast(x, inputs.dtype) + inputs  # Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class FeedForwardModule(layers.Layer):
    """Feed-forward module for Conformer architecture.

    Structure: LayerNorm → Dense(d_ff) → Swish → Dropout → Dense(d_model) → Dropout
    Output is scaled by 0.5 before adding residual (Macaron-style).

    Reference: Gulati et al., Section 2.4
    """

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
        x = tf.nn.swish(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return tf.cast(0.5 * x, inputs.dtype) + inputs  # Half-step residual

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class MultiHeadSelfAttentionModule(layers.Layer):
    """Multi-head self-attention module with relative positional encoding.

    Uses RelativeMultiHeadAttention instead of standard attention to
    incorporate relative position bias in the attention computation.

    Reference: Gulati et al., Section 2.2
    """

    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super(MultiHeadSelfAttentionModule, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.rel_attention = RelativeMultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout_rate=dropout_rate)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, pos_enc, training=None):
        """
        Args:
            inputs: (batch, seq_len, d_model)
            pos_enc: (1, seq_len, d_model) - relative positional encoding
            training: bool

        Returns:
            output: (batch, seq_len, d_model) with residual connection
        """
        x = self.layer_norm(inputs)
        attn_output = self.rel_attention(x, pos_enc, training=training)
        attn_output = self.dropout(attn_output, training=training)
        return tf.cast(attn_output, inputs.dtype) + inputs  # Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ConformerBlock(layers.Layer):
    """Conformer block combining all sub-modules.

    Structure (Macaron-style):
        y = x + 0.5 * FFN(x)          # First feed-forward (half-step)
        y = y + MHSA(y, pos_enc)       # Multi-head self-attention with relative pos
        y = y + Conv(y)                # Convolution module
        y = y + 0.5 * FFN(y)          # Second feed-forward (half-step)
        y = LayerNorm(y)               # Final layer normalization

    Reference: Gulati et al., Section 2.5, Figure 1
    """

    def __init__(self, d_model, d_ff, num_heads,
                 conv_kernel_size=31, dropout_rate=0.1,
                 ff_dropout_rate=None, attn_dropout_rate=None,
                 conv_dropout_rate=None, **kwargs):
        super(ConformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.conv_kernel_size = conv_kernel_size
        self.dropout_rate = dropout_rate
        self.ff_dropout_rate = ff_dropout_rate if ff_dropout_rate is not None else dropout_rate
        self.attn_dropout_rate = attn_dropout_rate if attn_dropout_rate is not None else dropout_rate
        self.conv_dropout_rate = conv_dropout_rate if conv_dropout_rate is not None else dropout_rate

        # First feed-forward module (half-step residual)
        self.ff_module1 = FeedForwardModule(d_model, d_ff, self.ff_dropout_rate)

        # Multi-head self-attention with relative positional encoding
        self.mhsa_module = MultiHeadSelfAttentionModule(
            d_model, num_heads, self.attn_dropout_rate)

        # Convolution module
        self.conv_module = ConvolutionModule(
            d_model, conv_kernel_size, self.conv_dropout_rate)

        # Second feed-forward module (half-step residual)
        self.ff_module2 = FeedForwardModule(d_model, d_ff, self.ff_dropout_rate)

        # Final layer normalization
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, pos_enc, training=None):
        """
        Args:
            inputs: (batch, seq_len, d_model)
            pos_enc: (1, seq_len, d_model) - positional encoding
            training: bool

        Returns:
            output: (batch, seq_len, d_model)
        """
        # First feed-forward module (half-step residual is inside the module)
        x = self.ff_module1(inputs, training=training)

        # Multi-head self-attention with relative positional encoding
        x = self.mhsa_module(x, pos_enc, training=training)

        # Convolution module
        x = self.conv_module(x, training=training)

        # Second feed-forward module (half-step residual is inside the module)
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
            'dropout_rate': self.dropout_rate,
            'ff_dropout_rate': self.ff_dropout_rate,
            'attn_dropout_rate': self.attn_dropout_rate,
            'conv_dropout_rate': self.conv_dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ConformerEncoder(layers.Layer):
    """Full Conformer encoder combining subsampling, positional encoding,
    and stacked Conformer blocks.

    This layer encapsulates the entire encoder pipeline so that the
    positional encoding is passed correctly through all blocks via
    a single call() method (required for Keras Functional API).

    Reference: Gulati et al., Figure 1
    """

    def __init__(self, d_model, d_ff, num_heads, num_blocks,
                 conv_kernel_size=31, dropout_rate=0.1,
                 ff_dropout_rate=None, attn_dropout_rate=None,
                 conv_dropout_rate=None, **kwargs):
        super(ConformerEncoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.conv_kernel_size = conv_kernel_size
        self.dropout_rate = dropout_rate
        self.ff_dropout_rate = ff_dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_dropout_rate = conv_dropout_rate

    def build(self, input_shape):
        # Convolutional subsampling (4x reduction)
        self.conv_subsample = ConvSubsampling(
            self.d_model, dropout_rate=self.dropout_rate, name="conv_subsample"
        )

        # Relative positional encoding
        self.pos_encoding = RelativePositionalEncoding(
            self.d_model, max_len=5000, name="rel_pos_enc")

        # Stacked Conformer blocks
        self.conformer_blocks = [
            ConformerBlock(
                d_model=self.d_model,
                d_ff=self.d_ff,
                num_heads=self.num_heads,
                conv_kernel_size=self.conv_kernel_size,
                dropout_rate=self.dropout_rate,
                ff_dropout_rate=self.ff_dropout_rate,
                attn_dropout_rate=self.attn_dropout_rate,
                conv_dropout_rate=self.conv_dropout_rate,
                name=f'conformer_block_{i}'
            )
            for i in range(self.num_blocks)
        ]

        super(ConformerEncoder, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch, time, freq) - spectrogram features

        Returns:
            output: (batch, time//4, d_model) - encoded features
        """
        # Convolutional subsampling: (batch, time, freq) → (batch, time//4, d_model)
        x = self.conv_subsample(inputs, training=training)

        # Generate positional encoding for the subsampled sequence
        pos_enc = self.pos_encoding(x)

        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, pos_enc, training=training)

        return x

    def get_config(self):
        config = super(ConformerEncoder, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'num_heads': self.num_heads,
            'num_blocks': self.num_blocks,
            'conv_kernel_size': self.conv_kernel_size,
            'dropout_rate': self.dropout_rate,
            'ff_dropout_rate': self.ff_dropout_rate,
            'attn_dropout_rate': self.attn_dropout_rate,
            'conv_dropout_rate': self.conv_dropout_rate
        })
        return config


def create_conformer_model(input_shape, num_classes=2, d_model=256, d_ff=1024,
                           num_heads=4, num_blocks=16, conv_kernel_size=31,
                           dropout_rate=0.1, ff_dropout_rate=None,
                           attn_dropout_rate=None, conv_dropout_rate=None,
                           learning_rate=1e-4, weight_decay=1e-4,
                           warmup_steps=1500, decay_steps=50000,
                           alpha=1e-7, label_smoothing=0.05,
                           clipnorm=1.0):
    """Create a paper-faithful Conformer model for audio classification.

    Pipeline:
        Input (time, freq) → ConvSubsampling (4x reduction) → Linear(d_model) → Dropout
        → + RelativePositionalEncoding
        → N × ConformerBlock (with relative attention)
        → GlobalAveragePooling1D
        → Classification Head

    Args:
        input_shape: Shape of input features (time_steps, freq_bins)
        num_classes: Number of output classes (mínimo 2 — ver nota abaixo)
        d_model: Model dimension
        d_ff: Feed-forward inner dimension (paper: 4 × d_model)
        num_heads: Number of attention heads
        num_blocks: Number of Conformer blocks
        conv_kernel_size: Depthwise convolution kernel size
        dropout_rate: Base dropout rate (P_drop do paper — vale para
            subsampling, FFN, atenção e módulo convolucional)
        ff_dropout_rate: Override do dropout do FeedForwardModule (default: dropout_rate)
        attn_dropout_rate: Override do dropout da atenção (default: dropout_rate)
        conv_dropout_rate: Override do dropout do ConvModule (default: dropout_rate)

    Returns:
        Compiled Keras model
    """
    # A loss usada aqui é entropia cruzada categórica com label smoothing sobre
    # a saída softmax. Com num_classes=1 a cabeça vira sigmoid de 1 unidade, o
    # Keras renormaliza y_pred para 1.0 e a loss fica IDENTICAMENTE ZERO (o
    # modelo não aprende). Promovemos para 2 classes (real/fake), como AASIST e
    # RawGAT-ST já faziam — a factory ainda usa num_classes=1 como default.
    if num_classes < 2:
        logger.info(
            "Conformer: num_classes=%s promovido para 2 (a CE categórica com "
            "1 classe degeneraria a loss para 0).", num_classes
        )
        num_classes = 2

    inputs = layers.Input(shape=input_shape)

    # Handle different input shapes
    if len(input_shape) == 1 or (len(input_shape) == 2 and input_shape[-1] == 1):
        # Raw audio input: convert to log-mel spectrogram
        # STFT params: 32ms janela / 8ms hop a 16 kHz, 80 mel bins.
        if len(input_shape) == 2:
            raw = layers.Reshape((input_shape[0],))(inputs)
        else:
            raw = inputs

        # LogMelSpectrogramLayer (camada registrada) no lugar do
        # `layers.Lambda(compute_log_mel)`: uma Lambda com função Python local
        # não é reconstruível pelo carregador safe_mode do Keras 3 — o modelo
        # treinava e salvava, mas falhava no load. Mesmos parâmetros de STFT.
        x = LogMelSpectrogramLayer(
            sample_rate=16000, n_fft=512, hop_length=128, n_mels=80,
            pad_end=True, name="log_mel_spectrogram",
        )(raw)
        # x shape: (batch, ~T/128, 80)

    elif len(input_shape) == 3 and input_shape[-1] == 1:
        # Input is (time, freq, 1) - reshape to (time, freq)
        x = layers.Reshape((input_shape[0], input_shape[1]))(inputs)
    elif len(input_shape) == 2:
        # Input is already (time, freq) - use directly
        x = inputs
    else:
        # For other shapes, flatten to 2D
        x = layers.Reshape((input_shape[0], -1))(inputs)

    # Conformer encoder: subsampling + pos encoding + N blocks
    x = ConformerEncoder(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_blocks=num_blocks,
        conv_kernel_size=conv_kernel_size,
        dropout_rate=dropout_rate,
        ff_dropout_rate=ff_dropout_rate,
        attn_dropout_rate=attn_dropout_rate,
        conv_dropout_rate=conv_dropout_rate,
        name='conformer_encoder'
    )(x)

    # Global average pooling over time
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    outputs, _ = create_classification_head(
        x, num_classes, dropout_rate=dropout_rate, hidden_dims=[d_model // 2])

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='conformer')

    # Label smoothing cross-entropy. Classe registrada (não uma closure local):
    # uma função definida dentro do builder não é resolvível por
    # `load_model(..., compile=True)`.
    loss = SparseLabelSmoothingCrossEntropy(
        label_smoothing=label_smoothing, from_logits=False
    )

    # Warmup + cosine decay learning rate schedule with AdamW optimizer
    from app.domain.models.training.optimization import WarmupCosineDecaySchedule
    lr_schedule = WarmupCosineDecaySchedule(
        initial_learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        alpha=alpha,
    )
    optimizer_kwargs = {}
    if clipnorm is not None and clipnorm > 0:
        optimizer_kwargs["clipnorm"] = clipnorm
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    logger.info(
        "Conformer-M criado: blocks=%d, d_model=%d, d_ff=%d, heads=%d, "
        "kernel=%d, P_drop=%.3f, params=%d",
        num_blocks, d_model, d_ff, num_heads, conv_kernel_size, dropout_rate,
        model.count_params(),
    )
    return model


def _merge_variant_params(defaults, overrides):
    params = dict(defaults)
    overrides = dict(overrides)

    alias_map = {
        "attention_heads": "num_heads",
        "hidden_units": "d_model",
        "l2_reg_strength": "weight_decay",
    }
    for source, target in alias_map.items():
        if source in overrides:
            value = overrides.pop(source)
            if target not in overrides:
                overrides[target] = value

    for key in list(params):
        if key in overrides:
            params[key] = overrides.pop(key)
    return params, overrides


# Configuração ÚNICA do Conformer — Conformer-M (Medium) de Gulati et al.,
# Interspeech 2020, Tabela 1: d_model=256, 16 blocos, 4 cabeças, d_ff=1024
# (= 4 × d_model), kernel depthwise 31 e P_drop=0.1 uniforme.
#
# CONSOLIDAÇÃO 2026-07-27: existiam duas variantes cujos nomes estavam
# INVERTIDOS — 'conformer' (dita "Large") tinha 8 blocos e 'conformer_lite'
# (dita "Medium") tinha 16, ou seja, a "lite" era ~2× maior. Além disso a
# variante de 8 blocos sobrescrevia o dropout por módulo (ff=0.2/attn=0.1/
# conv=0.1), o que anulava o `dropout_rate` do plano de benchmark em todo o
# encoder. Agora há UMA configuração, fiel ao paper, e `dropout_rate` vale
# para o encoder inteiro. 'conformer_lite' continua sendo aceito como ALIAS
# (compatibilidade de checkpoints/config) e resolve para a mesma topologia.
_CONFORMER_M_PARAMS = {
    "d_model": 256,
    "d_ff": 1024,          # 4 × d_model (paper)
    "num_heads": 4,        # paper Conformer-M
    "num_blocks": 16,      # paper Conformer-M
    "conv_kernel_size": 31,  # paper §2.3
    "dropout_rate": 0.1,   # P_drop do paper (uniforme no encoder)
}

_CONFORMER_ALIASES = {"default", "conformer", "conformer_lite", "conformer_m"}


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'conformer', **kwargs) -> tf.keras.Model:
    """Factory function to create the Conformer model.

    Configuração única (Conformer-M, Gulati et al., Interspeech 2020):
    d_model=256, d_ff=1024, heads=4, blocks=16, kernel 31, P_drop=0.1.

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: 'conformer' (aliases aceitos: 'default', 'conformer_m',
            'conformer_lite' — este último mantido só por compatibilidade)

    Returns:
        Compiled Keras model
    """
    nested_parameters = kwargs.pop("parameters", None)
    if nested_parameters:
        parameters = dict(nested_parameters)
        parameters.update(kwargs)
        kwargs = parameters

    if architecture not in _CONFORMER_ALIASES:
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            f"Use uma destas: {sorted(_CONFORMER_ALIASES)} "
            "(todas resolvem para a mesma configuração Conformer-M)."
        )
    if architecture == 'conformer_lite':
        logger.warning(
            "Conformer: 'conformer_lite' é apenas um ALIAS legado — a "
            "configuração é única (Conformer-M, 16 blocos). Use 'conformer'."
        )

    params, kwargs = _merge_variant_params(dict(_CONFORMER_M_PARAMS), kwargs)
    return create_conformer_model(
        input_shape=input_shape,
        num_classes=num_classes,
        **params,
        **kwargs,
    )


# Register custom layers for model loading
tf.keras.utils.get_custom_objects().update({
    'RelativePositionalEncoding': RelativePositionalEncoding,
    'RelativeMultiHeadAttention': RelativeMultiHeadAttention,
    'ConvSubsampling': ConvSubsampling,
    'ConvolutionModule': ConvolutionModule,
    'FeedForwardModule': FeedForwardModule,
    'MultiHeadSelfAttentionModule': MultiHeadSelfAttentionModule,
    'ConformerBlock': ConformerBlock,
    'ConformerEncoder': ConformerEncoder
})
