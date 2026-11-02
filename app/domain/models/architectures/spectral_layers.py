"""Front-end de sinal: STFT, Mel, bancos de filtros e normalizações de entrada.
Saiu de ``layers.py`` (P2-15), que continua sendo a porta de entrada e o ponto
de registro no Keras.
"""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from app.domain.models.architectures.safe_normalization import (
    SafeInstanceNormalization,
)

logger = logging.getLogger(__name__)


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

        # `tf.signal.stft` chama RFFT, que aceita SÓ float32/float64. Sob a
        # política `mixed_float16` a entrada chega em float16 e a construção do
        # modelo falha com "RFFT requires tf.float32 or tf.float64 inputs"
        # (2026-08-02: reproduzido ao criar o MultiscaleCNN com entrada de áudio
        # bruto em GPU). O benchmark não via isso porque alimenta log-mel já
        # pronto, mas qualquer caminho raw-audio + mixed precision quebrava.
        # A análise espectral é feita em float32 e o resultado volta ao dtype
        # da camada, então a política de precisão do resto do grafo é
        # preservada — só a FFT fica fora dela, que é o exigido.
        entrada_fp32 = tf.cast(inputs, tf.float32)

        stft = tf.signal.stft(
            entrada_fp32,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length
        )

        # Calculate magnitude
        spectrogram = tf.abs(stft)
        spectrogram = tf.cast(spectrogram, self.compute_dtype)

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


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class MagnitudeLayer(layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class LogMelFromMagnitudeLayer(layers.Layer):
    """Espectro de magnitude -> log-mel, como camada serializável.

    Substitui closures locais (`def apply_log_mel(mag): ...`) passadas a
    `layers.Lambda` no branch de áudio bruto do Res2Net/MultiscaleCNN — mesma
    razão de `AxisMaxAbsLayer` (closures/funções locais não são localizáveis
    pelo desserializador do Keras 3 em safe_mode).
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        num_spectrogram_bins: int = 1025,
        sample_rate: int = 16000,
        lower_edge_hertz: float = 0.0,
        upper_edge_hertz: float = 8000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = int(num_mel_bins)
        self.num_spectrogram_bins = int(num_spectrogram_bins)
        self.sample_rate = int(sample_rate)
        self.lower_edge_hertz = float(lower_edge_hertz)
        self.upper_edge_hertz = float(upper_edge_hertz)

    def call(self, mag):
        mel_w = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=self.num_spectrogram_bins,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz,
        )
        # `linear_to_mel_weight_matrix` devolve float32 SEMPRE. Sob
        # `mixed_float16` a magnitude chega em float16 e o matmul falha com
        # "Input 'y' of 'BatchMatMulV2' Op has type float32 that does not match
        # type float16" (2026-08-02, mesmo caminho raw-audio que quebrava o
        # STFTLayer). O log fica em float32 de propósito: `log(mel + 1e-6)` com
        # mel em float16 satura — 1e-6 é menor que o menor subnormal útil da
        # meia precisão, então o épsilon somiria e valores nulos virariam -inf.
        mag = tf.cast(mag, tf.float32)
        mel = tf.matmul(mag, mel_w)
        log_mel = tf.math.log(mel + 1e-6)
        log_mel = tf.cast(log_mel, self.compute_dtype)
        return tf.expand_dims(log_mel, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_mel_bins": self.num_mel_bins,
            "num_spectrogram_bins": self.num_spectrogram_bins,
            "sample_rate": self.sample_rate,
            "lower_edge_hertz": self.lower_edge_hertz,
            "upper_edge_hertz": self.upper_edge_hertz,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class LogMelSpectrogramLayer(layers.Layer):
    """Áudio bruto -> log-mel, como camada serializável (STFT + filtros mel).

    Front-end único para as arquiteturas que precisam converter forma de onda
    em log-mel dentro do grafo. Substitui closures locais passadas a
    `layers.Lambda` (não recarregáveis em safe_mode, o default do Keras 3).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        n_mels: int = 80,
        lower_edge_hertz: float = 0.0,
        upper_edge_hertz: float = None,
        pad_end: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.lower_edge_hertz = float(lower_edge_hertz)
        self.upper_edge_hertz = float(
            upper_edge_hertz if upper_edge_hertz is not None else sample_rate / 2.0
        )
        self.pad_end = bool(pad_end)

    def build(self, input_shape):
        self.mel_weight = tf.constant(
            tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=self.n_mels,
                num_spectrogram_bins=self.n_fft // 2 + 1,
                sample_rate=self.sample_rate,
                lower_edge_hertz=self.lower_edge_hertz,
                upper_edge_hertz=self.upper_edge_hertz,
            ),
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, inputs):
        x = inputs
        if x.shape.rank == 3 and x.shape[-1] == 1:
            x = tf.squeeze(x, axis=-1)
        stft = tf.signal.stft(
            tf.cast(x, tf.float32),
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            pad_end=self.pad_end,
        )
        magnitude = tf.abs(stft)
        mel = tf.matmul(magnitude, self.mel_weight)
        return tf.cast(tf.math.log(mel + 1e-6), self.compute_dtype)

    def compute_output_shape(self, input_shape):
        # Nº de quadros ESTÁTICO quando o comprimento da entrada é conhecido.
        # Sem isto o eixo temporal saía como None e quebrava consumidores que
        # precisam da forma em tempo de construção — o AST, por exemplo, calcula
        # a grade de patches a partir dela ("unsupported operand ... NoneType").
        samples = input_shape[1] if len(input_shape) > 1 else None
        frames = None
        if samples is not None:
            if self.pad_end:
                # tf.signal.stft com pad_end=True: ceil(samples / frame_step)
                frames = -(-int(samples) // self.hop_length)
            elif int(samples) >= self.n_fft:
                frames = (int(samples) - self.n_fft) // self.hop_length + 1
            else:
                frames = 0
        return (input_shape[0], frames, self.n_mels)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "lower_edge_hertz": self.lower_edge_hertz,
            "upper_edge_hertz": self.upper_edge_hertz,
            "pad_end": self.pad_end,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ASTInputNormalization(layers.Layer):
    """Normalização de entrada do AST (Gong et al., 2021, §2.1).

    O artigo normaliza o espectrograma de entrada para **média 0 e desvio
    padrão 0,5** ("we normalize the input audio spectrogram so that the dataset
    mean and standard deviation are 0 and 0.5"). Sem isso, a escala do log-mel
    (tipicamente média ≈ −5, desvio ≈ 3) desloca a distribuição de entrada para
    longe do regime em que a inicialização do Transformer foi calibrada.

    Aqui a estatística é calculada POR AMOSTRA, e não sobre o dataset: é uma
    aproximação deliberada que evita vazamento de estatística global entre
    treino/validação/teste — a mesma política do resto do projeto
    (``SafeInstanceNormalization``).
    """

    def __init__(self, target_std: float = 0.5, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.target_std = float(target_std)
        self.epsilon = float(epsilon)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        axes = list(range(1, x.shape.rank))
        mean = tf.reduce_mean(x, axis=axes, keepdims=True)
        std = tf.math.reduce_std(x, axis=axes, keepdims=True)
        normalized = (x - mean) / (std + self.epsilon) * self.target_std
        return tf.cast(normalized, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_std": self.target_std,
            "epsilon": self.epsilon,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ImageNetRangeScalingLayer(layers.Layer):
    """Escala cada amostra para a faixa [0, 255] esperada pelos backbones Keras.

    `tf.keras.applications.EfficientNet*` embute Rescaling+Normalization e
    espera entrada em [0, 255] (o `preprocess_input` da família é no-op). Um
    log-mel cru (~[-14, +5]) atravessa essa normalização como ruído próximo de
    zero e torna os pesos ImageNet praticamente inertes. A escala é por amostra
    (min-max no próprio exemplo) — não usa estatística do dataset, portanto não
    introduz vazamento.
    """

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        axes = list(range(1, x.shape.rank))
        min_v = tf.reduce_min(x, axis=axes, keepdims=True)
        max_v = tf.reduce_max(x, axis=axes, keepdims=True)
        scaled = (x - min_v) / tf.maximum(max_v - min_v, 1e-6) * 255.0
        return tf.cast(scaled, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


# ─── Filterbanks/DCT compartilhados (numpy puro) ───────────────────────────
# LFCC/MFCC/CQT eram construídos com código duplicado em sonic_sleuth.py e
# ensemble.py — e a versão do Sonic Sleuth dependia de `tf...numpy()` dentro do
# `build()`, com um fallback SILENCIOSO para um filterbank diferente quando o
# tensor não era eager. Estas funções são a fonte única, em numpy puro.

def linear_triangular_filterbank(
    n_fft: int, sample_rate: int, n_filters: int
) -> np.ndarray:
    """Filterbank triangular LINEARMENTE espaçado (base do LFCC)."""
    num_bins = n_fft // 2 + 1
    linear_points = np.linspace(0.0, sample_rate / 2.0, n_filters + 2)
    bin_points = np.round(linear_points * n_fft / sample_rate).astype(np.int32)
    bin_points = np.clip(bin_points, 0, num_bins - 1)

    filters = np.zeros((num_bins, n_filters), dtype=np.float32)
    for i in range(n_filters):
        left, center, right = (
            int(bin_points[i]),
            int(bin_points[i + 1]),
            int(bin_points[i + 2]),
        )
        for j in range(left, center):
            if center > left:
                filters[j, i] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filters[j, i] = (right - j) / (right - center)
    return filters


def dct_matrix(n_filters: int, n_coeffs: int) -> np.ndarray:
    """Matriz DCT-II ortonormal (filtros -> coeficientes cepstrais)."""
    matrix = np.zeros((n_filters, n_coeffs), dtype=np.float32)
    for k in range(n_coeffs):
        for n in range(n_filters):
            matrix[n, k] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_filters))
    matrix[:, 0] *= 1.0 / np.sqrt(n_filters)
    matrix[:, 1:] *= np.sqrt(2.0 / n_filters)
    return matrix


def cqt_triangular_filterbank(
    n_fft: int, sample_rate: int, n_bins: int, bins_per_octave: int,
    fmin: float = 32.70,
) -> np.ndarray:
    """Aproximação do CQT por filtros triangulares log-espaçados sobre a STFT."""
    num_stft_bins = n_fft // 2 + 1
    freqs = fmin * (2.0 ** (np.arange(n_bins) / bins_per_octave))
    stft_freqs = np.linspace(0, sample_rate / 2, num_stft_bins)

    cqt_filters = np.zeros((num_stft_bins, n_bins), dtype=np.float32)
    for i, fc in enumerate(freqs):
        bandwidth = fc * (2.0 ** (1.0 / bins_per_octave) - 1)
        low, high = fc - bandwidth / 2, fc + bandwidth / 2
        for j, sf in enumerate(stft_freqs):
            if low <= sf <= high:
                if sf <= fc and fc > low:
                    cqt_filters[j, i] = (sf - low) / (fc - low)
                elif sf > fc and high > fc:
                    cqt_filters[j, i] = (high - sf) / (high - fc)
    norms = np.sum(cqt_filters, axis=0, keepdims=True) + 1e-8
    return (cqt_filters / norms).astype(np.float32)


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
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


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AudioFeatureNormalization(SafeInstanceNormalization):
    """Normalização de áudio POR AMOSTRA (alias histórico).

    Hoje é apenas uma subclasse de :class:`SafeInstanceNormalization` — a
    versão antiga, que fazia ``adapt()`` sobre o dataset inteiro (vazamento de
    estatística), não existe mais. O NOME é mantido porque AASIST e RawGAT-ST
    usam esta camada no grafo (``audio_norm_layer``): trocar a classe
    invalidaria os checkpoints já treinados.

    Antes esta classe registrava um WARNING de "DEPRECATED" a cada
    instanciação, ou seja, em toda construção de AASIST/RawGAT-ST — ruído para
    uma troca que não pode ser feita sem retreinar. Em código NOVO prefira
    ``SafeInstanceNormalization`` diretamente.
    """

    def __init__(self, axis=-1, **kwargs):
        # `epsilon` é tratado pela superclasse; descartado aqui por compat com
        # configs antigas que o serializavam.
        if 'epsilon' in kwargs:
            kwargs.pop('epsilon')

        super().__init__(axis=axis, **kwargs)
        logger.debug(
            "AudioFeatureNormalization: alias de SafeInstanceNormalization "
            "(mantido para compatibilidade de checkpoints)."
        )

    # adapt method is not needed as SafeInstanceNormalization doesn't use it in the same way (stateless)
    # or if it does, it's inherited.
    # The previous implementation had 'adapt' which stored global mean/var.
    # We explicitly want to remove that behavior.


class AudioNormalizationLayer(layers.Layer):
    """Custom layer para normalizar áudio com média zero e variância unitária."""

    def __init__(self, **kwargs):
        super(AudioNormalizationLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Calcular média e desvio padrão ao longo do eixo temporal. Para áudio
        # bruto em shape (batch, time, 1), axis=-1 é o canal mono e zera o
        # sinal inteiro; o eixo temporal é 1. Para shape (batch, time), também
        # usamos axis=1.
        temporal_axis = 1 if inputs.shape.rank in (2, 3) else -1
        mean = tf.reduce_mean(inputs, axis=temporal_axis, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=temporal_axis, keepdims=True)

        # Evitar divisão por zero
        std = tf.maximum(std, 1e-8)

        # Normalizar
        normalized = (inputs - mean) / std

        return normalized

    def get_config(self):
        config = super().get_config()
        return config


class PreEmphasisLayer(layers.Layer):
    """Pre-emphasis high-pass filter for raw audio.

    Applies y[n] = x[n] - coeff * x[n-1] to sharpen high-frequency content.
    Standard in speech processing before feature extraction.
    """

    def __init__(self, coeff=0.97, **kwargs):
        super(PreEmphasisLayer, self).__init__(**kwargs)
        self.coeff = coeff

    def call(self, inputs):
        # inputs: (batch, time) or (batch, time, 1)
        squeeze = False
        if len(inputs.shape) == 3:
            squeeze = True
            inputs = tf.squeeze(inputs, axis=-1)

        # y[n] = x[n] - coeff * x[n-1]
        emphasized = inputs[:, 1:] - self.coeff * inputs[:, :-1]
        # Prepend first sample to maintain length
        first = tf.expand_dims(inputs[:, 0], axis=-1)
        emphasized = tf.concat([first, emphasized], axis=1)

        if squeeze:
            emphasized = tf.expand_dims(emphasized, axis=-1)
        return emphasized

    def get_config(self):
        config = super(PreEmphasisLayer, self).get_config()
        config.update({'coeff': self.coeff})
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class DeltaFeatureLayer(layers.Layer):
    """Computes delta (velocity) and delta-delta (acceleration) features.

    Takes a spectrogram (batch, time, freq) and returns 3-channel output
    (batch, time, freq, 3) with [static, delta, delta-delta].
    Used by EfficientNet-LSTM to replace naive channel repetition.
    """

    def __init__(self, order=2, width=2, **kwargs):
        super(DeltaFeatureLayer, self).__init__(**kwargs)
        self.order = order
        self.width = width

    def call(self, inputs):
        # inputs: (batch, time, freq) or (batch, time, freq, 1)
        if len(inputs.shape) == 4:
            x = inputs[..., 0]
        else:
            x = inputs

        channels = [x]  # static

        # Delta: finite difference approximation
        for _ in range(self.order):
            prev = x
            # Pad edges by replicating boundary frames
            padded = tf.pad(prev, [[0, 0], [self.width, self.width], [0, 0]], mode='REFLECT')
            # Weighted sum of differences
            denominator = 2.0 * sum(n * n for n in range(1, self.width + 1))
            delta = tf.zeros_like(prev)
            for n in range(1, self.width + 1):
                delta += n * (padded[:, self.width + n:self.width + n + tf.shape(prev)[1], :]
                              - padded[:, self.width - n:self.width - n + tf.shape(prev)[1], :])
            delta = delta / denominator
            channels.append(delta)
            x = delta

        # Stack: (batch, time, freq, num_channels)
        return tf.stack(channels, axis=-1)

    def get_config(self):
        config = super(DeltaFeatureLayer, self).get_config()
        config.update({
            'order': self.order,
            'width': self.width,
        })
        return config
