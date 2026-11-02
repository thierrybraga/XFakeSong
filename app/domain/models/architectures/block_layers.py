"""Blocos convolucionais, de atenção e de fusão usados na composição dos modelos.
Saiu de ``layers.py`` (P2-15), que continua sendo a porta de entrada e o ponto de
registro no Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers


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


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class WeightedScoreFusionLayer(layers.Layer):
    """sum_i w_i * score_i sobre o eixo dos ramos, como camada serializável.

    Substitui o `layers.Lambda` da fusão adaptativa do Ensemble (Eq. 28 do
    TCC). Entradas: `[scores (B, N, U), weights (B, N, 1)]`.
    """

    def call(self, inputs):
        scores, weights = inputs
        return tf.reduce_sum(scores * tf.cast(weights, scores.dtype), axis=1)

    def compute_output_shape(self, input_shape):
        scores_shape = input_shape[0]
        return (scores_shape[0], scores_shape[-1])

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
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

    scale_mode:
        'mul_add' — forma do paper (Tak et al., 2021 / implementação oficial
                    ASVspoof): y = sigmoid(FC(GAP(x))); out = x*y + y. O termo
                    aditivo evita que o empilhamento de FMS encolha a evidência
                    (motivo pelo qual o modo 'mul2' foi criado como paliativo).
        'mul2'    — comportamento legado (out = x * 2*sigmoid, kernel zeros).
                    Mantido como DEFAULT apenas para desserializar modelos
                    .keras antigos sem alterar sua saída; novos builds do
                    RawNet2 passam 'mul_add'.
    """

    def __init__(self, scale_mode: str = 'mul2', **kwargs):
        super(FeatureMapScalingLayer, self).__init__(**kwargs)
        if scale_mode not in ('mul2', 'mul_add'):
            raise ValueError(f"scale_mode inválido: {scale_mode!r}")
        self.scale_mode = scale_mode

    def build(self, input_shape):
        self.channels = input_shape[-1]
        # 'mul2' zera o kernel para o gate nascer em 1.0 (2*sigmoid(0)).
        # 'mul_add' usa a inicialização padrão, como na implementação oficial.
        initializer = 'zeros' if self.scale_mode == 'mul2' else 'glorot_uniform'
        self.dense = layers.Dense(
            self.channels,
            activation='sigmoid',
            kernel_initializer=initializer,
            bias_initializer='zeros',
        )
        super(FeatureMapScalingLayer, self).build(input_shape)

    def call(self, inputs):
        # Global Average Pooling
        y = tf.reduce_mean(inputs, axis=1)
        # Scale vector
        y = self.dense(y)
        # Reshape for broadcasting
        y = tf.expand_dims(y, axis=1)
        if self.scale_mode == 'mul_add':
            # Paper: escala multiplicativa + deslocamento aditivo.
            return inputs * y + y
        # Legado: multiplicativo puro com gate centrado em 1.0 na inicialização.
        return inputs * (2.0 * y)

    def get_config(self):
        config = super(FeatureMapScalingLayer, self).get_config()
        config.update({'scale_mode': self.scale_mode})
        return config


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
        self._layer_weights = self.add_weight(
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
        normalized_weights = tf.nn.softmax(self._layer_weights)

        # Weighted sum
        weighted_inputs = []
        for i in range(len(inputs)):
            weighted_inputs.append(inputs[i] * normalized_weights[i])

        return tf.add_n(weighted_inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'num_layers': self.num_layers})
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

        # Ver ResidualBlock2D: mesmo risco de dtype divergente sob mixed
        # precision na reconstrução simbólica do modelo salvo.
        shortcut = tf.cast(shortcut, x.dtype)
        return x + shortcut

    def get_config(self):
        config = super(ResidualBlock1D, self).get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size
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


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
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
        se = tf.cast(se, inputs.dtype)
        return inputs * se

    def get_config(self):
        config = super(SqueezeExcitationBlock2D, self).get_config()
        config.update({'reduction': self.reduction})
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
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
        for i, shape in enumerate(input_shape):
            projection = layers.Dense(self.embed_dim, name=f'proj_{i}')
            projection.build(shape)
            self.projection_layers.append(projection)
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            name='cross_attn'
        )
        self.layer_norm = layers.LayerNormalization()
        sequence_shape = (input_shape[0][0], len(input_shape), self.embed_dim)
        self.mha.build(sequence_shape, sequence_shape)
        self.layer_norm.build(sequence_shape)
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
        batch_size = tf.shape(attn_out)[0]
        num_branches = len(inputs)
        fused = tf.reshape(attn_out, [batch_size, num_branches * self.embed_dim])
        return fused

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], len(input_shape) * self.embed_dim)

    def get_config(self):
        config = super(CrossAttentionFusionLayer, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class GatedFusionLayer(layers.Layer):
    """Gated fusion mechanism for multi-branch models.

    Learns per-branch sigmoid gates to weight each branch's contribution.
    """

    def __init__(self, **kwargs):
        super(GatedFusionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is a list of shapes
        self.gate_layers = []
        for i, shape in enumerate(input_shape):
            gate = layers.Dense(1, activation='sigmoid', name=f'gate_{i}')
            gate.build(shape)
            self.gate_layers.append(gate)
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


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ResidualBlock2D(layers.Layer):
    """Bloco residual 2D para os encoders AASIST/RawGAT-ST fiéis."""

    def __init__(self, out_channels, kernel_size=(2, 3), pool_size=(1, 3),
                 first=False, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = int(out_channels)
        # KERNEL (2, 3), como o `Residual_block` da familia RawGAT-ST/AASIST.
        # O default era (3, 3): 50% mais parametros por convolucao no eixo de
        # frequencia e um campo receptivo diferente do baseline publicado.
        self.kernel_size = tuple(kernel_size)
        self.pool_size = tuple(pool_size)
        # `first=True` PULA a BN + ativacao iniciais, como na referencia. Sem
        # isto, o primeiro bloco da pilha aplicava BN+SELU sobre um tensor que
        # ja vinha normalizado e ativado do front-end sinc — uma segunda
        # saturacao do SELU que o paper nao tem.
        self.first = bool(first)

    def build(self, input_shape):
        # Com `first=True` a pre-ativacao nao roda, entao `bn1` seria um peso
        # orfao (sem gradiente, e o Keras avisa a cada passo).
        self.bn1 = (
            None if self.first
            else layers.BatchNormalization(name=f"{self.name}_bn1")
        )
        self.conv1 = layers.Conv2D(
            self.out_channels, self.kernel_size, padding="same",
            use_bias=False, name=f"{self.name}_conv1",
        )
        self.bn2 = layers.BatchNormalization(name=f"{self.name}_bn2")
        self.conv2 = layers.Conv2D(
            self.out_channels, self.kernel_size, padding="same",
            use_bias=False, name=f"{self.name}_conv2",
        )
        self.skip_conv = None
        if input_shape[-1] != self.out_channels:
            self.skip_conv = layers.Conv2D(
                self.out_channels, 1, padding="same", use_bias=False,
                name=f"{self.name}_skip",
            )
        self.pool = layers.MaxPooling2D(
            pool_size=self.pool_size, strides=self.pool_size,
            padding="same", name=f"{self.name}_pool",
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.first:
            x = inputs
        else:
            x = self.bn1(inputs, training=training)
            x = tf.nn.selu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.selu(x)
        x = self.conv2(x)
        shortcut = self.skip_conv(inputs) if self.skip_conv is not None else inputs
        # Sob mixed_float16, a reconstrução simbólica do modelo salvo (Keras 3
        # traça call() com uma policy de dtype diferente da usada no treino)
        # pode entregar `x` e `shortcut` em dtypes distintos, quebrando o Add
        # (float32 x float16). Cast explícito evita depender da policy global.
        shortcut = tf.cast(shortcut, x.dtype)
        return self.pool(x + shortcut)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "first": self.first,
        })
        return config
