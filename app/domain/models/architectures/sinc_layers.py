"""Banco sinc: a matemática compartilhada (``build_sinc_bandpass_filters``) e as
classes legadas. As camadas em uso por arquitetura estão em ``aasist_layers``,
``rawgat_layers`` e ``rawnet2_layers``; ``SincNetLayer`` e ``SincConvLayer`` ficam
para desserializar artefatos antigos. Saiu de ``layers.py`` (P2-15).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def build_sinc_bandpass_filters(low, high, n_time, window, normalize="l1"):
    """Banco de filtros passa-banda sinc — implementação ÚNICA e compartilhada.

    ``filters_k(t) = 2·f_high·sinc(2π·f_high·t) − 2·f_low·sinc(2π·f_low·t)``,
    janelado por Hamming. Usada por :class:`SincConvLayer` (AASIST/RawGAT-ST) e
    :class:`SincNetLayer` (RawNet2), que antes mantinham cópias divergentes da
    mesma matemática — inclusive com normalizações diferentes e sem contrato
    explícito sobre qual era qual.

    Args:
        low: (n_filters,) frequências de corte inferiores em Hz.
        high: (n_filters,) frequências de corte superiores em Hz.
        n_time: (kernel_size,) eixo temporal já centrado e dividido por ``sr``.
        window: (kernel_size,) janela (Hamming).
        normalize: ``"l1"`` (energia unitária) ou ``"max"`` (pico unitário).

    Returns:
        Tensor ``(kernel_size, n_filters)``.

    Nota numérica: ``sinc`` é avaliada com um epsilon SUBSTITUINDO x≈0 ANTES da
    divisão. Um ``tf.where`` sobre o resultado de ``sin(x)/x`` avaliaria 0/0 no
    ramo descartado e propagaria NaN pelo gradiente — armadilha clássica que já
    zerou o treino do RawNet2 no primeiro passo.
    """

    def _sinc(x):
        safe_x = tf.where(tf.abs(x) < 1e-7, tf.ones_like(x) * 1e-7, x)
        return tf.sin(np.pi * safe_x) / (np.pi * safe_x)

    low = tf.expand_dims(low, 1)  # (n_filters, 1)
    high = tf.expand_dims(high, 1)
    n_row = tf.expand_dims(n_time, 0)  # (1, kernel_size)

    band_pass = 2 * high * _sinc(2 * high * n_row) - 2 * low * _sinc(
        2 * low * n_row
    )  # (n_filters, kernel_size)
    band_pass = band_pass * tf.expand_dims(window, 0)

    if normalize == "max":
        denom = tf.reduce_max(tf.abs(band_pass), axis=1, keepdims=True) + 1e-8
    else:
        denom = tf.reduce_sum(tf.abs(band_pass), axis=1, keepdims=True) + 1e-7
    band_pass = band_pass / denom

    # (kernel_size, n_filters)
    return tf.transpose(band_pass)


class SincNetLayer(layers.Layer):
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
        **kwargs,
    ):
        super(SincNetLayer, self).__init__(**kwargs)
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
            name="low_hz",
            shape=(self.filters,),
            initializer=tf.constant_initializer(hz[:-1]),
            trainable=True,
        )
        self.band_hz_ = self.add_weight(
            name="band_hz",
            shape=(self.filters,),
            initializer=tf.constant_initializer(np.diff(hz)),
            trainable=True,
        )

        # Time axis for sinc function
        n = np.linspace(0, self.kernel_size - 1, self.kernel_size)
        n = (n - (self.kernel_size - 1) / 2) / self.sample_rate
        self.n_ = tf.constant(n, dtype=tf.float32)

        # Window function (Hamming)
        window = 0.54 - 0.46 * tf.cos(
            2 * np.pi * tf.range(self.kernel_size, dtype=tf.float32) / self.kernel_size
        )
        self.window_ = tf.constant(window, dtype=tf.float32)

        super(SincNetLayer, self).build(input_shape)

    def _to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def _to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)

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
                output = tf.nn.conv1d(inputs, filters, stride=1, padding="SAME")
        else:
            output = tf.nn.conv1d(inputs, filters, stride=1, padding="SAME")
        return tf.cast(output, target_dtype)

    def get_config(self):
        config = super(SincNetLayer, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "sample_rate": self.sample_rate,
                "min_low_hz": self.min_low_hz,
                "min_band_hz": self.min_band_hz,
                "memory_efficient_gpu": self.memory_efficient_gpu,
            }
        )
        return config


class SincConvLayer(layers.Layer):
    """Learnable sinc-based bandpass filter convolution (from SincNet/RawNet2).

    Implements parameterized sinc filters where low and high cutoff frequencies
    are learnable parameters initialized on the mel scale. Each filter is a
    bandpass filter: sinc(2*pi*f_high*t) - sinc(2*pi*f_low*t), windowed by Hamming.

    Reference: Ravanelli & Bengio, "Speaker Recognition from Raw Waveform with SincNet", 2018
    """

    def __init__(
        self,
        n_filters=70,
        kernel_size=129,
        sample_rate=16000,
        min_low_hz=50.0,
        min_band_hz=50.0,
        trainable_filters=False,
        **kwargs,
    ):
        super(SincConvLayer, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        # BANCO FIXO por padrao (2026-08-20), como a referencia.
        #
        # No codigo oficial do RawGAT-ST e do AASIST (classe `CONV`), o banco
        # sinc e montado no forward a partir de pontos mel calculados uma vez
        # no `__init__` e guardado num `torch.Tensor` — NAO um `nn.Parameter`,
        # portanto sem gradiente. Aqui as frequencias de corte eram pesos
        # treinaveis: um grau de liberdade a mais na camada que define o que o
        # modelo enxerga, ausente no baseline com que a tabela do TCC compara.
        # `trainable_filters=True` mantem o comportamento antigo para ablacao.
        self.trainable_filters = bool(trainable_filters)

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
            trainable=self.trainable_filters,
        )
        self.band_hz_ = self.add_weight(
            name="band_hz",
            shape=(self.n_filters,),
            initializer=tf.keras.initializers.Constant(init_band.numpy()),
            trainable=self.trainable_filters,
        )

        # Hamming window (not trainable)
        n = tf.cast(tf.range(0, self.kernel_size), tf.float32)
        self.window_ = 0.54 - 0.46 * tf.cos(
            2.0 * 3.14159265 * n / (self.kernel_size - 1)
        )

        super(SincConvLayer, self).build(input_shape)

    def _sinc(self, x):
        """Normalized sinc function: sin(x) / x, with sinc(0) = 1."""
        safe_x = tf.where(tf.abs(x) < 1e-7, tf.ones_like(x) * 1e-7, x)
        return tf.sin(safe_x) / safe_x

    def call(self, inputs):
        # inputs: (batch, time, 1)
        target_dtype = self.compute_dtype
        inputs = tf.cast(inputs, tf.float32)
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)

        # Ensure positive frequencies
        min_low_hz = tf.cast(self.min_low_hz, tf.float32)
        min_band_hz = tf.cast(self.min_band_hz, tf.float32)
        nyquist = tf.cast(self.sample_rate / 2.0, tf.float32)
        low = min_low_hz + tf.abs(tf.cast(self.low_hz_, tf.float32))
        high = tf.clip_by_value(
            low + min_band_hz + tf.abs(tf.cast(self.band_hz_, tf.float32)),
            clip_value_min=min_low_hz,
            clip_value_max=nyquist,
        )

        # Eixo temporal centrado, em SEGUNDOS (n − centro)/fs.
        n = tf.cast(tf.range(0, self.kernel_size), tf.float32)
        n = (n - (self.kernel_size - 1.0) / 2.0) / self.sample_rate

        # Construção dos filtros por `build_sinc_bandpass_filters` — a MESMA
        # função usada pelo SincNetLayer (RawNet2). Antes cada camada mantinha
        # sua própria cópia da matemática do sinc; a daqui já teve um bug de
        # argumento (faltava π e sobrava ×fs, degenerando os passa-banda
        # mel-inicializados num banco pseudo-aleatório) que a outra nunca teve.
        # Uma implementação só elimina a classe inteira de divergência.
        # Verificado numericamente: saída idêntica à versão anterior (1e-5).
        band_pass = build_sinc_bandpass_filters(
            low=low,
            high=high,
            n_time=n,
            window=tf.cast(self.window_, tf.float32),
            normalize="l1",  # energia unitária (convenção desta camada)
        )  # (kernel_size, n_filters)

        # Reshape for conv1d: (kernel_size, 1, n_filters)
        filters = tf.expand_dims(band_pass, 1)

        # Apply convolution
        output = tf.nn.conv1d(inputs, filters, stride=1, padding="SAME")

        return tf.cast(output, target_dtype)

    def get_config(self):
        config = super(SincConvLayer, self).get_config()
        config.update(
            {
                "n_filters": self.n_filters,
                "kernel_size": self.kernel_size,
                "sample_rate": self.sample_rate,
                "min_low_hz": self.min_low_hz,
                "min_band_hz": self.min_band_hz,
                # Sem isto, um artefato salvo com banco treinavel recarregaria com
                # banco fixo (ou vice-versa) em silencio.
                "trainable_filters": self.trainable_filters,
            }
        )
        return config
