"""Camadas exclusivas do RawNet2.

As classes deste modulo pertencem A ARQUITETURA RawNet2 e a mais nenhuma. Elas
foram extraidas em 2026-08-20 de
``app/domain/models/architectures/layers.py``, onde conviviam com as camadas de
todas as outras arquiteturas do projeto.

Por que a separacao existe: enquanto as definicoes moravam num modulo
compartilhado, uma correcao de fidelidade a um paper mudava mais de uma
arquitetura em silencio -- foi o que aconteceu quando um ajuste no
``GraphReadoutLayer`` e no ``SincConvLayer`` alterou AASIST e RawGAT-ST de uma
vez, duas entradas oficiais do benchmark. Cada arquitetura passa a ter
implementacao propria para que uma mudanca dirigida a ela nao possa vazar para
as demais.

O RawNet2 reforca esse argumento: o front-end sinc dele
(``RawNet2SincConv``, ex-``SincNetLayer``) sempre foi DIFERENTE do
``SincConvLayer`` usado por AASIST e RawGAT-ST -- normalizacao por pico,
execucao do front-end em CPU para conter o workspace do cuDNN -- e ainda assim
os dois dividiam arquivo.

Os nomes das classes recebem o prefixo ``RawNet2`` de proposito:
``register_keras_serializable`` registra pelo NOME DA CLASSE, entao uma copia
homonima sobrescreveria o registro original e um modelo salvo desserializaria
com a classe errada.

Helpers de DSP generico continuam compartilhados de proposito e sao IMPORTADOS
de ``layers.py`` (ver :func:`build_sinc_bandpass_filters`), nao copiados.
"""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from app.domain.models.architectures.layers import build_sinc_bandpass_filters

logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawNet2MultiScaleConv1D(layers.Layer):
    """Bloco convolucional multi-escala para capturar características em diferentes escalas."""

    def __init__(self, filters, kernel_sizes=None, **kwargs):
        super().__init__(**kwargs)
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


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawNet2FeatureMapScaling(layers.Layer):
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
        super().__init__(**kwargs)
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
        super().build(input_shape)

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
        config = super().get_config()
        config.update({'scale_mode': self.scale_mode})
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawNet2SincConv(layers.Layer):
    """
    SincNet layer for raw waveform processing.
    Implementation of the Sinc-convolution from Ravanelli & Bengio (2018).

    Usada pelo RawNet2. Compartilha a construção dos filtros com
    :class:`SincConvLayer` via :func:`build_sinc_bandpass_filters`; o que
    permanece próprio desta classe é o ``memory_efficient_gpu`` (o kernel de
    1024 amostras do RawNet2 faz o cuDNN pedir workspace de dezenas de GB).
    """

    def __init__(
        self,
        filters,
        kernel_size,
        sample_rate=16000,
        min_low_hz=30,
        min_band_hz=50,
        memory_efficient_gpu=True,
        trainable_filters=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # BANCO FIXO por padrao (2026-08-20), como a referencia.
        #
        # No RawNet2 anti-spoofing de Tak et al. — que e o que o escopo oficial
        # passou a construir — o banco sinc e montado a partir de pontos mel
        # calculados uma vez e guardado SEM gradiente. Deixar as frequencias de
        # corte treinaveis dava ao modelo um grau de liberdade a mais
        # exatamente na camada que define o que ele enxerga, ausente no
        # baseline publicado. Mesma decisao ja aplicada ao SincConv de AASIST e
        # RawGAT-ST. `trainable_filters=True` mantem o antigo para ablacao.
        self.trainable_filters = bool(trainable_filters)
        self.filters = filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.memory_efficient_gpu = memory_efficient_gpu

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
            trainable=self.trainable_filters
        )
        self.band_hz_ = self.add_weight(
            name='band_hz',
            shape=(self.filters,),
            initializer=tf.constant_initializer(np.diff(hz)),
            trainable=self.trainable_filters
        )

        # Time axis for sinc function
        n = np.linspace(0, self.kernel_size - 1, self.kernel_size)
        n = (n - (self.kernel_size - 1) / 2) / self.sample_rate
        self.n_ = tf.constant(n, dtype=tf.float32)

        # Window function (Hamming)
        window = 0.54 - 0.46 * tf.cos(2 * np.pi * tf.range(self.kernel_size, dtype=tf.float32) / self.kernel_size)
        self.window_ = tf.constant(window, dtype=tf.float32)

        super().build(input_shape)

    def _to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def _to_hz(self, mel):
        return 700 * (10**(mel / 2595) - 1)

    def call(self, inputs):
        # Sinc filters are numerically sensitive and their trainable
        # frequency parameters stay in float32 under mixed precision.
        # Keep the whole filter construction/convolution in float32, then
        # cast back to the layer policy dtype for downstream layers.
        target_dtype = self.compute_dtype
        inputs = tf.cast(inputs, tf.float32)

        # Constraints
        min_low_hz = tf.cast(self.min_low_hz, tf.float32)
        min_band_hz = tf.cast(self.min_band_hz, tf.float32)
        nyquist = tf.cast(self.sample_rate / 2, tf.float32)
        low = min_low_hz + tf.abs(tf.cast(self.low_hz_, tf.float32))
        high = tf.clip_by_value(
            low + min_band_hz + tf.abs(tf.cast(self.band_hz_, tf.float32)),
            min_low_hz,
            nyquist,
        )
        # Construção dos filtros por `build_sinc_bandpass_filters` — a mesma
        # função usada pelo SincConvLayer (AASIST/RawGAT-ST). Antes cada classe
        # tinha sua própria cópia da matemática, com normalizações divergentes.
        # Aqui a normalização é por PICO ("max"), preservando o comportamento
        # histórico dos checkpoints do RawNet2.
        filters = build_sinc_bandpass_filters(
            low=low,
            high=high,
            n_time=tf.cast(self.n_, tf.float32),
            window=tf.cast(self.window_, tf.float32),
            normalize="max",
        )  # (kernel_size, n_filters)

        # Reshape for Conv1D: (kernel_size, in_channels, out_channels)
        # SincNet expects (kernel_size, 1, filters)
        filters = tf.expand_dims(filters, 1)

        # cuDNN may select a >35 GiB backward workspace for RawNet2's
        # 1025-sample Sinc kernel. Run only this one-input-channel front-end on
        # CPU, where TensorFlow uses bounded memory; tensors and gradients move
        # automatically, while all residual blocks and the GRU remain on GPU.
        if self.memory_efficient_gpu:
            with tf.device("/CPU:0"):
                output = tf.nn.conv1d(inputs, filters, stride=1, padding='SAME')
        else:
            output = tf.nn.conv1d(inputs, filters, stride=1, padding='SAME')
        return tf.cast(output, target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'sample_rate': self.sample_rate,
            'min_low_hz': self.min_low_hz,
            'min_band_hz': self.min_band_hz,
            'memory_efficient_gpu': self.memory_efficient_gpu,
            # Sem isto, um artefato salvo com banco treinavel recarregaria com
            # banco fixo (ou vice-versa) em silencio.
            'trainable_filters': self.trainable_filters,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawNet2ResidualBlock1D(layers.Layer):
    """Pre-activation residual block for 1D convolutions.

    Structure: BN -> LeakyReLU -> Conv1D -> BN -> LeakyReLU -> Conv1D + skip.
    Uses 1x1 convolution for skip connection if channel mismatch.

    Reference: RawNet2 (Tak et al., 2021)
    """

    def __init__(self, out_channels, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
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

        super().build(input_shape)

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
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size
        })
        return config
