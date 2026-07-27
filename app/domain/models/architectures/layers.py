import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from app.domain.models.architectures.safe_normalization import SafeInstanceNormalization

logger = logging.getLogger(__name__)

@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ChannelMeanLayer(layers.Layer):
    """mean(x, axis=-1, keepdims=True) como camada serializável.

    Substitui `layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, ...))` em
    `ensure_flat_input` — mesma razão de `AxisMaxAbsLayer` (Lambda de função
    Python não sobrevive ao load em safe_mode, o default do Keras 3).
    """

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


def is_raw_audio(input_shape):
    """
    Check if input shape corresponds to raw audio.
    Assumes input_shape does not include batch dimension.
    (Time,) -> Raw
    (Time, 1) -> Raw
    (Time, >1) -> Features (e.g. Spectrogram)
    """
    if len(input_shape) == 1:
        return True
    if len(input_shape) == 2:
        return input_shape[-1] == 1
    if len(input_shape) == 3:
        return input_shape[-1] == 1 and input_shape[1] == 1 # Very specific case
    return False

def ensure_flat_input(x):
    """Ensure input is (batch, time, 1) or (batch, time).

    (O parâmetro `input_shape` foi REMOVIDO: era declarado, todos os callers o
    passavam e a função nunca o lia — a decisão sai do `x.shape` real. Uma
    assinatura que mente sobre o que usa é a mesma classe de problema do
    "config morto" no registry.)
    """
    if len(x.shape) == 3 and x.shape[-1] > 1:
        # If we have channels, we might want to take the first one or mean?
        # For now assume it's mono or we take mean.
        # ChannelMeanLayer (não layers.Lambda com lambda Python crua): mesma
        # razão de AxisMaxAbsLayer — Lambda de função Python não é
        # recarregável em safe_mode (default do Keras 3).
        return ChannelMeanLayer(name="ensure_flat_mean")(x)
    return x

def apply_gru_block(x, units, return_sequences=True, go_backwards=False, dropout_rate=0.0, name=None):
    """
    Apply GRU block compatible with both CPU and GPU.
    Standard GRU automatically uses the cuDNN kernel when conditions are met
    (default activations, no recurrent_dropout, unroll=False).
    """
    return layers.GRU(
        units,
        return_sequences=return_sequences,
        go_backwards=go_backwards,
        dropout=dropout_rate,
        name=name
    )(x)

@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class FlattenTimeFeaturesLayer(layers.Layer):
    """(B, T, F, C) -> (B, T, F*C) para formas dinâmicas, como camada serializável.

    Substitui `layers.Lambda(lambda t: tf.reshape(...))` — Lambda de função
    Python não é reconstruível pelo carregador safe_mode do Keras 3.
    """

    def call(self, inputs):
        shape = tf.shape(inputs)
        return tf.reshape(inputs, [shape[0], shape[1], -1])

    def compute_output_shape(self, input_shape):
        tail = input_shape[2:]
        flat = None
        if all(dim is not None for dim in tail):
            flat = 1
            for dim in tail:
                flat *= dim
        return (input_shape[0], input_shape[1], flat)

    def get_config(self):
        return super().get_config()


def flatten_features_for_gru(x, name=None):
    """
    Flatten feature dimensions (freq * channel) for GRU input (batch, time, features).
    Used in AASIST and RawGAT-ST architectures.
    """
    shape_before_gru = x.shape
    # If shape is fully defined
    if shape_before_gru[1] is not None and shape_before_gru[2] is not None and shape_before_gru[3] is not None:
         x = layers.Reshape(
            (shape_before_gru[1],
             shape_before_gru[2] *
             shape_before_gru[3]), name=name)(x)
    else:
        # Forma dinâmica: camada registrada (não Lambda) para que o modelo
        # salvo volte a carregar em safe_mode.
        x = FlattenTimeFeaturesLayer(name=name)(x)
    return x

def apply_reshape_for_cnn(tensor, target_shape):
    """
    Helper function for Reshape for CNN using SliceLayer to avoid Lambda.
    """
    # Logic:
    # If 3D -> Reshape to 4D with 1 channel
    # If 4D with > 1 channel -> Slice to 1 channel
    # If 4D with 1 channel -> Keep

    shape = tensor.shape

    if len(shape) == 3:
        return layers.Reshape(
            (target_shape[0], target_shape[1], 1))(tensor)
    elif len(shape) == 4:
        if shape[-1] != 1:
            logger.warning(
                f"Input tensor shape {shape} for CNN expects last dim to be 1. Slicing to 1 channel.")
            return SliceLayer(slice_idx=0)(tensor)
        return tensor
    else:
        # Fallback: assume 3D if not 4D
        return layers.Reshape(
            (target_shape[0], target_shape[1], 1))(tensor)

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

        # Calculate STFT
        stft = tf.signal.stft(
            inputs,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length
        )

        # Calculate magnitude
        spectrogram = tf.abs(stft)

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
class ExpandDimsLayer(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({'axis': self.axis})
        return config

@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ResizeLayer(layers.Layer):
    def __init__(self, target_height, target_width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, [self.target_height, self.target_width])

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({
            'target_height': self.target_height,
            'target_width': self.target_width
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
        mel = tf.matmul(mag, mel_w)
        log_mel = tf.math.log(mel + 1e-6)
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
class TimeResizeLayer(layers.Layer):
    """Reamostra o eixo temporal de (B, T, C) para (B, target_length, C).

    Camada serializável no lugar de `layers.Lambda(_resize_time)` (back-end de
    grafo SSL): o grafo do GAT espectral precisa de dimensão temporal ESTÁTICA,
    mas a saída do backbone SSL tem T dinâmico.
    """

    def __init__(self, target_length: int, **kwargs):
        super().__init__(**kwargs)
        self.target_length = int(target_length)

    def call(self, inputs):
        z4 = tf.expand_dims(inputs, axis=1)               # (B, 1, T, C)
        z4 = tf.image.resize(z4, [1, self.target_length])  # (B, 1, T_fix, C)
        return tf.cast(tf.squeeze(z4, axis=1), self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_length, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"target_length": self.target_length})
        return config


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
class AxisMaxAbsLayer(layers.Layer):
    """max(|x|, axis=axis) como camada serializável.

    Substitui `layers.Lambda(lambda v: tf.reduce_max(tf.abs(v), axis=N))`
    usado em AASIST/RawGAT-ST para extrair os nós espectral/temporal do
    encoder 2D. Uma Lambda com função Python crua não é reconstruível pelo
    carregador SAFE MODE do Keras 3 (`ValueError: ... Lambda layer whose
    function is a Python lambda ... disallowed by default`) — o modelo
    treina e salva normalmente, mas falha ao ser recarregado sem
    `safe_mode=False`. Mesma computação, sem o risco de execução de código
    arbitrário na desserialização.
    """

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = int(axis)

    def call(self, inputs):
        return tf.reduce_max(tf.abs(inputs), axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RepeatChannelLayer(layers.Layer):
    def __init__(self, repeats=3, axis=-1, **kwargs):
        super(RepeatChannelLayer, self).__init__(**kwargs)
        self.repeats = repeats
        self.axis = axis

    def call(self, inputs):
        return tf.repeat(inputs, self.repeats, axis=self.axis)

    def get_config(self):
        config = super(RepeatChannelLayer, self).get_config()
        config.update({'repeats': self.repeats, 'axis': self.axis})
        return config

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

def create_classification_head(x, num_classes, dropout_rate=0.3, hidden_dims=None):
    """
    Creates a standard classification head with Dense -> BN -> Dropout blocks.
    """
    if hidden_dims is None:
        hidden_dims = [512, 256, 128]
    for i, dim in enumerate(hidden_dims):
        x = layers.Dense(dim, activation='relu', name=f'classifier_dense{i+1}')(x)
        x = layers.BatchNormalization(name=f'classifier_bn{i+1}')(x)
        # Reduce dropout for deeper layers
        current_dropout = dropout_rate * (0.5 ** (i > 0)) # 0.3, 0.15, 0.15...
        x = layers.Dropout(current_dropout, name=f'classifier_dropout{i+1}')(x)

    # dtype='float32': sob mixed_float16, softmax+crossentropy em float16
    # satura/perde precisão e pode colapsar o treino (rede "morta" após a
    # 1a epoca). Mesma correção já aplicada em AASIST/RawGAT-ST.
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output', dtype='float32')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output', dtype='float32')(x)
        loss = 'sparse_categorical_crossentropy'

    return outputs, loss


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

    low = tf.expand_dims(low, 1)    # (n_filters, 1)
    high = tf.expand_dims(high, 1)
    n_row = tf.expand_dims(n_time, 0)  # (1, kernel_size)

    band_pass = (
        2 * high * _sinc(2 * high * n_row) - 2 * low * _sinc(2 * low * n_row)
    )                                # (n_filters, kernel_size)
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
            name='low_hz',
            shape=(self.filters,),
            initializer=tf.constant_initializer(hz[:-1]),
            trainable=True
        )
        self.band_hz_ = self.add_weight(
            name='band_hz',
            shape=(self.filters,),
            initializer=tf.constant_initializer(np.diff(hz)),
            trainable=True
        )

        # Time axis for sinc function
        n = np.linspace(0, self.kernel_size - 1, self.kernel_size)
        n = (n - (self.kernel_size - 1) / 2) / self.sample_rate
        self.n_ = tf.constant(n, dtype=tf.float32)

        # Window function (Hamming)
        window = 0.54 - 0.46 * tf.cos(2 * np.pi * tf.range(self.kernel_size, dtype=tf.float32) / self.kernel_size)
        self.window_ = tf.constant(window, dtype=tf.float32)

        super(SincNetLayer, self).build(input_shape)

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
        config = super(SincNetLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'sample_rate': self.sample_rate,
            'min_low_hz': self.min_low_hz,
            'min_band_hz': self.min_band_hz,
            'memory_efficient_gpu': self.memory_efficient_gpu,
        })
        return config


# NOTE: ResidualBlock1D is defined below (after AASIST layers section)
# with out_channels parameter and LeakyReLU activation.
# Used by RawNet2, AASIST, and RawGAT-ST.


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


class GraphAttentionLayer(layers.Layer):
    """
    Camada de Atenção em Grafos (GAT) simplificada (Self-Attention).
    Combina implementações de AASIST e RawGAT-ST.
    """

    def __init__(self, output_dim: int, num_heads: int = 4, dropout_rate: float = 0.1,
                 use_residual: bool = True, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.kernel = None
        self.bias = None
        self.residual_projection = None

    def build(self, input_shape):
        # Handle both single shape and list of shapes
        if isinstance(input_shape, list):
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape

        input_dim = feature_shape[-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.output_dim * self.num_heads),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.output_dim * self.num_heads,),
            initializer="zeros",
            trainable=True
        )

        # Residual connection projection if needed
        if self.use_residual and input_dim != self.output_dim * self.num_heads:
            self.residual_projection = self.add_weight(
                name="residual_projection",
                shape=(input_dim, self.output_dim * self.num_heads),
                initializer="glorot_uniform",
                trainable=True
            )

        super(GraphAttentionLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # Handle both single tensor and list inputs
        if isinstance(inputs, list):
            features = inputs[0]
            # adjacency_matrix = inputs[1] # Ignored
        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        num_nodes = tf.shape(features)[1]

        # Linear transformation
        # (batch, nodes, output_dim * num_heads)
        transformed_features = tf.matmul(features, self.kernel)

        # Reshape for multi-head
        # (batch, nodes, heads, output_dim)
        transformed_features = tf.reshape(
            transformed_features, (batch_size, num_nodes, self.num_heads, self.output_dim))

        # Transpose for attention calculation: (batch, heads, nodes, output_dim)
        transformed_features_t = tf.transpose(transformed_features, [0, 2, 1, 3])

        # Scaled Dot-Product Attention
        # (batch, heads, nodes, nodes)
        logits = tf.matmul(transformed_features_t, tf.transpose(transformed_features_t, [0, 1, 3, 2]))
        logits = logits / tf.sqrt(tf.cast(self.output_dim, tf.float32))

        attention_coefs = tf.nn.softmax(logits, axis=-1)

        if training:
            attention_coefs = tf.nn.dropout(attention_coefs, rate=self.dropout_rate)

        # Aggregate
        # (batch, heads, nodes, output_dim)
        outputs = tf.matmul(attention_coefs, transformed_features_t)

        # Reshape back
        # (batch, nodes, heads, output_dim)
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        # (batch, nodes, heads * output_dim)
        outputs = tf.reshape(outputs, (batch_size, num_nodes, self.output_dim * self.num_heads))

        outputs = outputs + self.bias

        # Residual connection
        if self.use_residual:
            if self.residual_projection is not None:
                residual = tf.matmul(features, self.residual_projection)
            else:
                residual = features
            outputs = outputs + residual

        return tf.nn.relu(outputs)

    def get_config(self):
        config = super(GraphAttentionLayer, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_residual": self.use_residual
        })
        return config

class SliceLayer(layers.Layer):
    """Custom layer to slice the last dimension."""

    def __init__(self, slice_idx=0, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.slice_idx = slice_idx

    def call(self, inputs):
        return inputs[..., :self.slice_idx + 1]

    def get_config(self):
        config = super(SliceLayer, self).get_config()
        config.update({'slice_idx': self.slice_idx})
        return config


# ============================================================================
# AASIST Paper-Faithful Layers (Jung et al., ICASSP 2022)
# ============================================================================

class SincConvLayer(layers.Layer):
    """Learnable sinc-based bandpass filter convolution (from SincNet/RawNet2).

    Implements parameterized sinc filters where low and high cutoff frequencies
    are learnable parameters initialized on the mel scale. Each filter is a
    bandpass filter: sinc(2*pi*f_high*t) - sinc(2*pi*f_low*t), windowed by Hamming.

    Reference: Ravanelli & Bengio, "Speaker Recognition from Raw Waveform with SincNet", 2018
    """

    def __init__(self, n_filters=70, kernel_size=129, sample_rate=16000,
                 min_low_hz=50.0, min_band_hz=50.0, **kwargs):
        super(SincConvLayer, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

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
            trainable=True
        )
        self.band_hz_ = self.add_weight(
            name="band_hz",
            shape=(self.n_filters,),
            initializer=tf.keras.initializers.Constant(init_band.numpy()),
            trainable=True
        )

        # Hamming window (not trainable)
        n = tf.cast(tf.range(0, self.kernel_size), tf.float32)
        self.window_ = 0.54 - 0.46 * tf.cos(2.0 * 3.14159265 * n / (self.kernel_size - 1))

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
            normalize="l1",   # energia unitária (convenção desta camada)
        )  # (kernel_size, n_filters)

        # Reshape for conv1d: (kernel_size, 1, n_filters)
        filters = tf.expand_dims(band_pass, 1)

        # Apply convolution
        output = tf.nn.conv1d(inputs, filters, stride=1, padding='SAME')

        return tf.cast(output, target_dtype)

    def get_config(self):
        config = super(SincConvLayer, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'sample_rate': self.sample_rate,
            'min_low_hz': self.min_low_hz,
            'min_band_hz': self.min_band_hz
        })
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


class GATConvLayer(layers.Layer):
    """Graph Attention Network layer with ADDITIVE attention (Velickovic et al., 2018).

    Uses the original GAT attention mechanism:
        e_ij = LeakyReLU(a_src^T * W*h_i + a_dst^T * W*h_j)
        alpha_ij = softmax_j(e_ij)
        h'_i = sum_j(alpha_ij * W*h_j)

    This is distinct from the existing GraphAttentionLayer which uses
    scaled dot-product (Transformer-style) attention.

    Reference: Velickovic et al., "Graph Attention Networks", ICLR 2018
    """

    def __init__(self, out_features, num_heads=1, dropout_rate=0.1,
                 concat_heads=True, negative_slope=0.2, **kwargs):
        super(GATConvLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.concat_heads = concat_heads
        self.negative_slope = negative_slope

    def build(self, input_shape):
        in_features = input_shape[-1]

        # Linear transformation weight: shared across heads but output is out_features * num_heads
        self.W = self.add_weight(
            name="W",
            shape=(in_features, self.out_features * self.num_heads),
            initializer="glorot_uniform",
            trainable=True
        )

        # Attention vectors: one pair (a_src, a_dst) per head
        self.a_src = self.add_weight(
            name="a_src",
            shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.a_dst = self.add_weight(
            name="a_dst",
            shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.out_features * self.num_heads if self.concat_heads else self.out_features,),
            initializer="zeros",
            trainable=True
        )

        super(GATConvLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (batch, nodes, in_features)
        batch_size = tf.shape(inputs)[0]
        num_nodes = tf.shape(inputs)[1]

        # Linear transform: (batch, nodes, out_features * num_heads)
        h = tf.matmul(inputs, self.W)

        # Reshape to (batch, nodes, num_heads, out_features)
        h = tf.reshape(h, (batch_size, num_nodes, self.num_heads, self.out_features))

        # Transpose to (batch, num_heads, nodes, out_features)
        h = tf.transpose(h, [0, 2, 1, 3])

        # Compute attention scores using additive mechanism
        # e_src: (batch, heads, nodes, 1)
        e_src = tf.einsum('bhni,hio->bhno', h, self.a_src)
        # e_dst: (batch, heads, nodes, 1)
        e_dst = tf.einsum('bhni,hio->bhno', h, self.a_dst)

        # e_ij = LeakyReLU(e_src_i + e_dst_j)
        # Broadcasting: (batch, heads, nodes, 1) + (batch, heads, 1, nodes)
        e = e_src + tf.transpose(e_dst, [0, 1, 3, 2])
        e = tf.nn.leaky_relu(e, alpha=self.negative_slope)

        # Attention coefficients
        alpha = tf.nn.softmax(e, axis=-1)  # (batch, heads, nodes, nodes)

        if training:
            alpha = tf.nn.dropout(alpha, rate=self.dropout_rate)

        # Weighted aggregation: (batch, heads, nodes, out_features)
        out = tf.matmul(alpha, h)

        # Reshape back
        # (batch, nodes, heads, out_features)
        out = tf.transpose(out, [0, 2, 1, 3])

        if self.concat_heads:
            # (batch, nodes, heads * out_features)
            out = tf.reshape(out, (batch_size, num_nodes, self.num_heads * self.out_features))
        else:
            # Average heads: (batch, nodes, out_features)
            out = tf.reduce_mean(out, axis=2)

        out = out + self.bias
        return tf.nn.elu(out)

    def get_config(self):
        config = super(GATConvLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'concat_heads': self.concat_heads,
            'negative_slope': self.negative_slope
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AASISTGraphAttentionLayer(layers.Layer):
    """Atenção de grafo FIEL a RawGAT-ST/AASIST (Tak 2021; Jung ICASSP 2022).

    Diferente do GAT aditivo de Velickovic (``GATConvLayer``), estes artigos
    derivam o mapa de atenção do **produto elemento a elemento entre pares de
    nós**, projetado e comprimido por ``tanh``, reduzido a um escalar por uma
    direção aprendível e **escalado por uma temperatura** antes do softmax::

        A_ij = softmax_i( w^T · tanh(W_att (h_i ⊙ h_j)) / τ )
        h'   = W_att_proj (A · h) + W_res h          (projeção com e sem atenção)
        h'   = SELU(BN(h'))

    A temperatura é um hiperparâmetro por camada no AASIST (2.0 nos GATs
    espectral/temporal, 100.0 nas HS-GAL) e controla o quanto a atenção se
    aproxima de uma média uniforme.

    NOTA: o softmax é normalizado sobre ``axis=-2`` e a agregação soma sobre o
    último eixo — exatamente como no código de referência dos autores.
    """

    def __init__(self, out_features, temperature=1.0, dropout_rate=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_features = int(out_features)
        self.temperature = float(temperature)
        self.dropout_rate = float(dropout_rate)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.att_proj = layers.Dense(self.out_features, name="att_proj")
        self.att_weight = self.add_weight(
            name="att_weight", shape=(self.out_features, 1),
            initializer="glorot_uniform", trainable=True,
        )
        self.proj_with_att = layers.Dense(self.out_features, name="proj_with_att")
        self.proj_without_att = layers.Dense(
            self.out_features, name="proj_without_att"
        )
        self.bn = layers.BatchNormalization(name="bn")
        self.input_drop = layers.Dropout(self.dropout_rate)

        # Build EXPLÍCITO das sub-camadas: criadas aqui, elas ficariam com
        # `built=False` na reconstrução do modelo salvo e o Keras 3 aborta o
        # load ("objects could not be loaded ... Dense name=att_proj").
        self.att_proj.build((None, None, None, in_dim))   # tensor par-a-par
        self.proj_with_att.build((None, None, in_dim))
        self.proj_without_att.build((None, None, in_dim))
        self.bn.build((None, None, self.out_features))
        super().build(input_shape)

    def _derive_att_map(self, x):
        # Produto par-a-par: (B, N, 1, C) * (B, 1, N, C) -> (B, N, N, C)
        pairwise = tf.expand_dims(x, 2) * tf.expand_dims(x, 1)
        att = tf.tanh(self.att_proj(pairwise))          # (B, N, N, out)
        att = tf.matmul(att, tf.cast(self.att_weight, att.dtype))  # (B, N, N, 1)
        att = att / tf.cast(self.temperature, att.dtype)
        return tf.nn.softmax(att, axis=-2)

    def call(self, inputs, training=None):
        x = self.input_drop(inputs, training=training)
        att_map = tf.squeeze(self._derive_att_map(x), axis=-1)  # (B, N, N)
        out = self.proj_with_att(tf.matmul(att_map, x)) + self.proj_without_att(x)
        out = self.bn(out, training=training)
        return tf.nn.selu(out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.out_features)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_features": self.out_features,
            "temperature": self.temperature,
            "dropout_rate": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class MasterNodeSeed(layers.Layer):
    """Master node treinável ``(1, 1, C)`` replicado para o batch.

    O AASIST declara ``master1``/``master2`` como ``nn.Parameter`` e os injeta
    na primeira HS-GAL de cada ramo. Recebe um tensor apenas para herdar o
    tamanho do batch; o conteúdo dele é ignorado.
    """

    def __init__(self, feature_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = int(feature_dim)

    def build(self, input_shape):
        self.seed = self.add_weight(
            name="master_seed", shape=(1, 1, self.feature_dim),
            initializer="random_normal", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        return tf.tile(tf.cast(self.seed, self.compute_dtype), [batch, 1, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, self.feature_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"feature_dim": self.feature_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AASISTHtrgGraphAttentionLayer(layers.Layer):
    """HS-GAL heterogênea FIEL ao AASIST (Jung et al., ICASSP 2022, §2.3).

    A contribuição que dá nome à camada são **três conjuntos distintos de
    parâmetros de atenção por TIPO DE ARESTA**: nó-tipo1↔nó-tipo1 (``w11``),
    nó-tipo2↔nó-tipo2 (``w22``) e o par cruzado tipo1↔tipo2 (``w12``,
    compartilhado nas duas direções). Uma atenção homogênea com *type
    embeddings* — como fazia a implementação anterior — não reproduz isso.

    O **master node** agrega o grafo inteiro por uma atenção própria
    (``att_projM``/``att_weightM``) e é devolvido para alimentar a HS-GAL
    seguinte (o "stack"). Quando nenhum master é fornecido, ele é inicializado
    como a MÉDIA de todos os nós, como no código de referência; o AASIST, na
    prática, injeta um master treinável por ramo.

    Entrada: ``[x1, x2]`` ou ``[x1, x2, master]``.
    Saída:   ``(x1', x2', master')``.
    """

    def __init__(self, out_features=32, temperature=100.0, dropout_rate=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_features = int(out_features)
        self.temperature = float(temperature)
        self.dropout_rate = float(dropout_rate)

    def build(self, input_shape):
        in_dim = int(input_shape[0][-1])
        self.proj_type1 = layers.Dense(in_dim, name="proj_type1")
        self.proj_type2 = layers.Dense(in_dim, name="proj_type2")

        self.att_proj = layers.Dense(self.out_features, name="att_proj")
        self.att_projM = layers.Dense(self.out_features, name="att_projM")

        def _att_vec(name):
            return self.add_weight(
                name=name, shape=(self.out_features, 1),
                initializer="glorot_uniform", trainable=True,
            )

        # Três conjuntos por tipo de aresta + o do master (paper §2.3).
        self.att_weight11 = _att_vec("att_weight11")
        self.att_weight22 = _att_vec("att_weight22")
        self.att_weight12 = _att_vec("att_weight12")
        self.att_weightM = _att_vec("att_weightM")

        self.proj_with_att = layers.Dense(self.out_features, name="proj_with_att")
        self.proj_without_att = layers.Dense(
            self.out_features, name="proj_without_att"
        )
        self.proj_with_attM = layers.Dense(
            self.out_features, name="proj_with_attM"
        )
        self.proj_without_attM = layers.Dense(
            self.out_features, name="proj_without_attM"
        )
        self.bn = layers.BatchNormalization(name="bn")
        self.input_drop = layers.Dropout(self.dropout_rate)

        # Build EXPLÍCITO — ver nota em AASISTGraphAttentionLayer.build().
        self.proj_type1.build((None, None, in_dim))
        self.proj_type2.build((None, None, in_dim))
        self.att_proj.build((None, None, None, in_dim))  # tensor par-a-par
        self.att_projM.build((None, None, in_dim))
        self.proj_with_att.build((None, None, in_dim))
        self.proj_without_att.build((None, None, in_dim))
        self.proj_with_attM.build((None, None, in_dim))
        self.proj_without_attM.build((None, None, in_dim))
        self.bn.build((None, None, self.out_features))
        super().build(input_shape)

    def _derive_att_map(self, x, n1):
        pairwise = tf.expand_dims(x, 2) * tf.expand_dims(x, 1)   # (B,N,N,C)
        att = tf.tanh(self.att_proj(pairwise))                    # (B,N,N,out)

        # Blocos por tipo de aresta: 11 (tipo1↔tipo1), 22 (tipo2↔tipo2) e 12
        # (cruzado, compartilhado nas duas direções) — §2.3 do AASIST.
        a11 = tf.matmul(att[:, :n1, :n1, :], tf.cast(self.att_weight11, att.dtype))
        a12 = tf.matmul(att[:, :n1, n1:, :], tf.cast(self.att_weight12, att.dtype))
        a21 = tf.matmul(att[:, n1:, :n1, :], tf.cast(self.att_weight12, att.dtype))
        a22 = tf.matmul(att[:, n1:, n1:, :], tf.cast(self.att_weight22, att.dtype))

        top = tf.concat([a11, a12], axis=2)      # (B, n1, N, 1)
        bottom = tf.concat([a21, a22], axis=2)   # (B, n2, N, 1)
        att_map = tf.concat([top, bottom], axis=1) / tf.cast(
            self.temperature, att.dtype
        )
        return tf.nn.softmax(att_map, axis=-2)

    def _update_master(self, x, master):
        att = tf.tanh(self.att_projM(x * master))                 # (B,N,out)
        att = tf.matmul(att, tf.cast(self.att_weightM, att.dtype))  # (B,N,1)
        att = att / tf.cast(self.temperature, att.dtype)
        att = tf.nn.softmax(att, axis=-2)
        # (B,1,N) @ (B,N,C) -> (B,1,C)
        pooled = tf.matmul(tf.transpose(att, [0, 2, 1]), x)
        return self.proj_with_attM(pooled) + self.proj_without_attM(master)

    def call(self, inputs, training=None):
        if len(inputs) == 3:
            x1, x2, master = inputs
        else:
            x1, x2 = inputs
            master = None

        n1 = x1.shape[1] if x1.shape[1] is not None else tf.shape(x1)[1]

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = tf.concat([x1, x2], axis=1)

        if master is None:
            master = tf.reduce_mean(x, axis=1, keepdims=True)

        x = self.input_drop(x, training=training)
        att_map = tf.squeeze(self._derive_att_map(x, n1), axis=-1)  # (B,N,N)
        master = self._update_master(x, master)

        out = self.proj_with_att(tf.matmul(att_map, x)) + self.proj_without_att(x)
        out = self.bn(out, training=training)
        out = tf.nn.selu(out)

        return out[:, :n1, :], out[:, n1:, :], master

    def compute_output_shape(self, input_shape):
        s1, s2 = input_shape[0], input_shape[1]
        return (
            (s1[0], s1[1], self.out_features),
            (s2[0], s2[1], self.out_features),
            (s1[0], 1, self.out_features),
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_features": self.out_features,
            "temperature": self.temperature,
            "dropout_rate": self.dropout_rate,
        })
        return config


class GraphPoolLayer(layers.Layer):
    """Learnable graph pooling via top-k node selection.

    Computes a learned score per node, selects the top-k nodes (k = ratio * N),
    and gates the selected node features by their sigmoid scores.

    Reference: Graph U-Nets (Gao & Ji, 2019)
    """

    def __init__(self, ratio=0.5, target_nodes=None, **kwargs):
        super(GraphPoolLayer, self).__init__(**kwargs)
        self.ratio = ratio
        # `target_nodes` seleciona um número ABSOLUTO de nós (top-k), em vez de
        # uma fração. Serve para alinhar dois grafos antes de uma fusão
        # element-wise usando a MESMA primitiva de pooling dos artigos, sem
        # recorrer a uma projeção densa sobre o eixo de nós (que mistura nós
        # arbitrariamente e não é uma operação de grafo).
        self.target_nodes = None if target_nodes is None else int(target_nodes)

    def build(self, input_shape):
        in_features = input_shape[-1]

        self.score_proj = self.add_weight(
            name="score_proj",
            shape=(in_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super(GraphPoolLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, nodes, features)
        num_nodes = tf.shape(inputs)[1]
        if self.target_nodes is not None:
            k = tf.minimum(tf.constant(self.target_nodes, tf.int32), num_nodes)
            k = tf.maximum(k, 1)
        else:
            k = tf.maximum(
                tf.cast(tf.cast(num_nodes, tf.float32) * self.ratio, tf.int32), 1
            )

        # Compute scores: (batch, nodes, 1)
        scores = tf.matmul(inputs, self.score_proj)
        scores = tf.squeeze(scores, axis=-1)  # (batch, nodes)

        # Top-k selection
        _, top_indices = tf.math.top_k(scores, k=k, sorted=False)

        # Gather selected nodes
        batch_size = tf.shape(inputs)[0]

        # Create batch indices for gather_nd
        batch_indices = tf.repeat(
            tf.expand_dims(tf.range(batch_size), 1), k, axis=1
        )  # (batch, k)

        indices = tf.stack([batch_indices, top_indices], axis=-1)  # (batch, k, 2)
        selected_features = tf.gather_nd(inputs, indices)  # (batch, k, features)
        selected_scores = tf.gather_nd(scores, indices)    # (batch, k)

        # Gate by sigmoid of scores
        gate = tf.nn.sigmoid(selected_scores)  # (batch, k)
        gate = tf.expand_dims(gate, -1)        # (batch, k, 1)

        return selected_features * gate

    def compute_output_shape(self, input_shape):
        nodes = input_shape[1]
        if self.target_nodes is not None:
            pooled_nodes = (
                min(self.target_nodes, int(nodes)) if nodes is not None
                else self.target_nodes
            )
        else:
            pooled_nodes = (
                max(int(nodes * self.ratio), 1)
                if nodes is not None
                else None
            )
        return tf.TensorShape((input_shape[0], pooled_nodes, input_shape[-1]))

    def get_config(self):
        config = super(GraphPoolLayer, self).get_config()
        config.update({'ratio': self.ratio, 'target_nodes': self.target_nodes})
        return config


class GraphReadoutLayer(layers.Layer):
    """Graph readout combining max readout and attention-weighted readout.

    Produces a fixed-size graph-level representation from variable-size node features
    by concatenating max-pooled and attention-weighted node features.

    Output shape: (batch, 2 * in_features)

    Reference: AASIST (Jung et al., ICASSP 2022)
    """

    def __init__(self, **kwargs):
        super(GraphReadoutLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        in_features = input_shape[-1]

        self.att_w = self.add_weight(
            name="att_w",
            shape=(in_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super(GraphReadoutLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, nodes, features)

        # Max readout
        h_max = tf.reduce_max(inputs, axis=1)  # (batch, features)

        # Attention readout
        scores = tf.matmul(inputs, self.att_w)   # (batch, nodes, 1)
        alpha = tf.nn.softmax(scores, axis=1)    # (batch, nodes, 1)
        h_att = tf.reduce_sum(inputs * alpha, axis=1)  # (batch, features)

        return tf.concat([h_max, h_att], axis=-1)  # (batch, 2*features)

    def get_config(self):
        return super(GraphReadoutLayer, self).get_config()


class HSGALLayer(layers.Layer):
    """Heterogeneous Stacking Graph Attention Layer.

    The key contribution of AASIST: cross-domain attention between spectral
    and temporal graph nodes, enabling integrated spectro-temporal analysis.

    Takes (spectral_nodes, temporal_nodes) as input and produces updated
    representations for both via cross-attention followed by self-attention
    on the combined heterogeneous graph.

    Reference: Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
    Spectro-Temporal Graph Attention Networks", ICASSP 2022
    """

    def __init__(self, out_features, num_heads=2, dropout_rate=0.1, **kwargs):
        super(HSGALLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # input_shape is a list of two shapes: [spectral_shape, temporal_shape]
        spec_features = input_shape[0][-1]
        temp_features = input_shape[1][-1]

        # Cross-attention: spectral queries attend to temporal keys/values
        self.W_q_s2t = self.add_weight(
            name="W_q_s2t", shape=(spec_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_k_s2t = self.add_weight(
            name="W_k_s2t", shape=(temp_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_v_s2t = self.add_weight(
            name="W_v_s2t", shape=(temp_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)

        # Cross-attention: temporal queries attend to spectral keys/values
        self.W_q_t2s = self.add_weight(
            name="W_q_t2s", shape=(temp_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_k_t2s = self.add_weight(
            name="W_k_t2s", shape=(spec_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_v_t2s = self.add_weight(
            name="W_v_t2s", shape=(spec_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)

        # Attention vectors for additive GAT (per cross-attention direction)
        self.a_s2t = self.add_weight(
            name="a_s2t", shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform", trainable=True)
        self.a_t2s = self.add_weight(
            name="a_t2s", shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform", trainable=True)

        # Self-attention GAT on combined graph
        combined_features = self.out_features * self.num_heads
        self.gat_self = GATConvLayer(
            out_features=self.out_features, num_heads=self.num_heads,
            dropout_rate=self.dropout_rate, concat_heads=True,
            name=self.name + "_gat_self")
        self.gat_self.build(tf.TensorShape([None, None, combined_features]))

        # Layer norms
        self.ln_spec = layers.LayerNormalization(name=self.name + "_ln_spec")
        self.ln_temp = layers.LayerNormalization(name=self.name + "_ln_temp")

        super(HSGALLayer, self).build(input_shape)

    def _cross_attention(self, queries, keys_values, W_q, W_k, W_v, a_vec, training=None):
        """Additive cross-attention between two sets of nodes."""
        batch_size = tf.shape(queries)[0]
        n_q = tf.shape(queries)[1]
        n_kv = tf.shape(keys_values)[1]

        Q = tf.matmul(queries, W_q)       # (batch, n_q, out*heads)
        K = tf.matmul(keys_values, W_k)   # (batch, n_kv, out*heads)
        V = tf.matmul(keys_values, W_v)   # (batch, n_kv, out*heads)

        # Reshape to (batch, heads, nodes, out_features)
        Q = tf.reshape(Q, (batch_size, n_q, self.num_heads, self.out_features))
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.reshape(K, (batch_size, n_kv, self.num_heads, self.out_features))
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.reshape(V, (batch_size, n_kv, self.num_heads, self.out_features))
        V = tf.transpose(V, [0, 2, 1, 3])

        # Additive attention: e_ij = LeakyReLU(a^T * (Q_i + K_j))
        # a_vec: (heads, out_features, 1)
        e_q = tf.einsum('bhni,hio->bhno', Q, a_vec)   # (batch, heads, n_q, 1)
        e_k = tf.einsum('bhni,hio->bhno', K, a_vec)   # (batch, heads, n_kv, 1)

        e = e_q + tf.transpose(e_k, [0, 1, 3, 2])  # (batch, heads, n_q, n_kv)
        e = tf.nn.leaky_relu(e, alpha=0.2)

        alpha = tf.nn.softmax(e, axis=-1)

        if training:
            alpha = tf.nn.dropout(alpha, rate=self.dropout_rate)

        # Aggregate: (batch, heads, n_q, out_features)
        out = tf.matmul(alpha, V)
        # (batch, n_q, heads, out_features)
        out = tf.transpose(out, [0, 2, 1, 3])
        # (batch, n_q, heads * out_features)
        out = tf.reshape(out, (batch_size, n_q, self.num_heads * self.out_features))

        return out

    def call(self, inputs, training=None):
        spectral_nodes, temporal_nodes = inputs

        # Cross-attention: spectral attends to temporal
        spec_cross = self._cross_attention(
            spectral_nodes, temporal_nodes,
            self.W_q_s2t, self.W_k_s2t, self.W_v_s2t, self.a_s2t,
            training=training)

        # Cross-attention: temporal attends to spectral
        temp_cross = self._cross_attention(
            temporal_nodes, spectral_nodes,
            self.W_q_t2s, self.W_k_t2s, self.W_v_t2s, self.a_t2s,
            training=training)

        # Layer norm
        spec_cross = self.ln_spec(spec_cross)
        temp_cross = self.ln_temp(temp_cross)

        # Combine into heterogeneous graph for self-attention
        combined = tf.concat([spec_cross, temp_cross], axis=1)  # (batch, n_spec+n_temp, features)
        combined = self.gat_self(combined, training=training)

        # Split back
        n_spec = tf.shape(spec_cross)[1]
        spectral_out = combined[:, :n_spec, :]
        temporal_out = combined[:, n_spec:, :]

        return spectral_out, temporal_out

    def get_config(self):
        config = super(HSGALLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


# ====================== IMPROVED LAYERS FOR ACCURACY ======================


class AMSoftmaxLayer(layers.Layer):
    """Additive Margin Softmax (AM-Softmax / CosFace).

    Replaces Dense+Softmax for more discriminative embeddings in deepfake detection.
    Reference: Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face Recognition", CVPR 2018

    Applies angular margin penalty: cos(theta) - m, then scales by s.
    During inference (training=False), returns standard cosine similarity logits.

    ⚠️  OUTPUT TYPE: This layer emits RAW LOGITS (scaled cosine similarities),
    NOT probabilities. The loss function MUST use from_logits=True:
        - SparseCategoricalCrossentropy(from_logits=True)
        - categorical_crossentropy(..., from_logits=True)
    Using the string shortcut "sparse_categorical_crossentropy" (which defaults
    to from_logits=False) will compute log(negative_value) → NaN loss.

    scale=15.0 is the safe default (original paper uses 30-64, but smaller values
    prevent overflow in float16 and still provide effective margin training).
    """

    def __init__(self, num_classes, scale=15.0, margin=0.35, **kwargs):
        super(AMSoftmaxLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

    def build(self, input_shape):
        self.W = self.add_weight(
            name='am_softmax_weights',
            shape=(input_shape[-1], self.num_classes),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AMSoftmaxLayer, self).build(input_shape)

    def call(self, inputs, labels=None, training=None):
        # BUG FIX: tf.nn.l2_normalize(zero_vector) = 0/0 = NaN when the upstream
        # activations collapse to all-zeros (e.g. due to bad audio normalization or
        # corrupted input). Replace any NaN/Inf embeddings with a unit vector before
        # normalizing so the model degrades gracefully instead of poisoning gradients.
        inputs = tf.where(tf.math.is_finite(inputs), inputs, tf.zeros_like(inputs))

        # L2-normalize embeddings and weights (epsilon=1e-12 in TF prevents /0)
        x_norm = tf.nn.l2_normalize(inputs, axis=-1)
        w_norm = tf.nn.l2_normalize(self.W, axis=0)

        # Cosine similarity
        cosine = tf.matmul(x_norm, w_norm)
        # Clamp to [-1, 1] to guard against numerical noise from float16
        cosine = tf.clip_by_value(cosine, -1.0, 1.0)

        if training and labels is not None:
            # One-hot encode labels
            one_hot = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
            # Subtract margin from target class
            cosine = cosine - one_hot * self.margin

        # Scale logits
        logits = self.scale * cosine
        return logits

    def get_config(self):
        config = super(AMSoftmaxLayer, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'scale': self.scale,
            'margin': self.margin,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AMSoftmaxCrossEntropy(tf.keras.losses.Loss):
    """Entropia cruzada com margem aditiva (AM-Softmax/CosFace) NA LOSS.

    A ``AMSoftmaxLayer`` emite ``s·cos(θ)`` SEM margem: no grafo funcional os
    rótulos nunca chegam ao ``call`` da camada, então a margem lá é código
    morto. É aqui, com ``y_true`` disponível, que o AM-Softmax do paper
    acontece::

        logit_alvo ← s·(cos θ − m) = s·cos θ − s·m

    Mantenha ``scale``/``margin`` em sincronia com os da ``AMSoftmaxLayer``.

    Args:
        scale: fator ``s`` usado pela camada (default 15.0).
        margin: margem aditiva ``m`` do CosFace (default 0.35).
        label_smoothing: suavização de rótulo aplicada depois da margem.
    """

    def __init__(self, scale: float = 15.0, margin: float = 0.35,
                 label_smoothing: float = 0.1, name: str = "am_softmax_ce",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = float(scale)
        self.margin = float(margin)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.cast(tf.one_hot(y_true_int, num_classes), y_pred.dtype)
        # CosFace: subtrai s·m do logit da classe-alvo.
        y_pred = y_pred - one_hot * tf.cast(
            self.scale * self.margin, y_pred.dtype
        )
        k = tf.cast(num_classes, y_pred.dtype)
        ls = tf.cast(self.label_smoothing, y_pred.dtype)
        smoothed = one_hot * (1.0 - ls) + ls / k
        return tf.keras.losses.categorical_crossentropy(
            smoothed, y_pred, from_logits=True
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale,
            "margin": self.margin,
            "label_smoothing": self.label_smoothing,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class SparseLabelSmoothingCrossEntropy(tf.keras.losses.Loss):
    """Entropia cruzada com label smoothing para rótulos INTEIROS (esparsos).

    O Keras só oferece `label_smoothing` na versão categórica (one-hot). Esta
    classe faz a conversão internamente, e — por ser uma `Loss` registrada, e
    não uma closure definida dentro do builder — sobrevive a
    `load_model(..., compile=True)`.

    Args:
        label_smoothing: fator de suavização (0.0 desliga).
        from_logits: True se `y_pred` são logits; False para saída softmax.
    """

    def __init__(self, label_smoothing: float = 0.05, from_logits: bool = False,
                 name: str = "sparse_label_smoothing_ce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = float(label_smoothing)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        # reshape em vez de squeeze: squeeze total colapsaria batch=1 a escalar.
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.cast(tf.one_hot(y_true_int, num_classes), y_pred.dtype)
        k = tf.cast(num_classes, y_pred.dtype)
        ls = tf.cast(self.label_smoothing, y_pred.dtype)
        smoothed = one_hot * (1.0 - ls) + ls / k
        return tf.keras.losses.categorical_crossentropy(
            smoothed, y_pred, from_logits=self.from_logits
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits,
        })
        return config


class OCSoftmaxLayer(layers.Layer):
    """One-Class Softmax (OC-Softmax) — Zhang et al., 2021.

    Objetivo *one-class* para anti-spoofing: aprende UM vetor-centro e produz
    um score escalar de "bonafide-ness" `s = ŵ·x̂` (cosseno, maior = mais
    bonafide). Treinado com `oc_softmax_loss`, compacta os embeddings bonafide
    e afasta os spoof — generaliza melhor a **ataques não vistos** que a
    entropia cruzada binária.

    Saída: `(batch, 1)` (score em [-1, 1]). É uma alternativa OPCIONAL à cabeça
    de 2 unidades softmax (não é default — requer `oc_softmax_loss` e tratamento
    do score escalar na inferência).
    """

    def __init__(self, feat_dim=None, **kwargs):
        super(OCSoftmaxLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            name="oc_center",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(OCSoftmaxLayer, self).build(input_shape)

    def call(self, inputs):
        # NaN-safety (mesma proteção do AMSoftmax)
        inputs = tf.where(tf.math.is_finite(inputs), inputs, tf.zeros_like(inputs))
        x = tf.nn.l2_normalize(inputs, axis=-1)
        w = tf.nn.l2_normalize(self.w, axis=0)
        s = tf.matmul(x, w)  # (batch, 1) — similaridade cosseno com o centro
        return tf.clip_by_value(s, -1.0, 1.0)

    def get_config(self):
        config = super(OCSoftmaxLayer, self).get_config()
        config.update({"feat_dim": self.feat_dim})
        return config


def oc_softmax_loss(m0: float = 0.9, m1: float = 0.2, alpha: float = 20.0):
    """Loss do OC-Softmax (Zhang et al., 2021). `m0 > m1`.

    Convenção de rótulos do XFakeSong: **0 = real/bonafide** (target, deve ter
    score ≥ m0) e **1 = fake/spoof** (deve ter score ≤ m1).

    `y_pred`: score escalar (batch, 1) da `OCSoftmaxLayer`.
    Forma numericamente estável via `softplus` = log(1 + exp(z)).
    """
    def _loss(y_true, y_pred):
        y = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
        s = tf.reshape(tf.cast(y_pred, tf.float32), (-1,))
        margin = tf.where(tf.equal(y, 0.0), m0, m1)          # margem por classe
        sign = tf.where(tf.equal(y, 0.0), 1.0, -1.0)         # (-1)^y
        z = alpha * (margin - s) * sign
        return tf.reduce_mean(tf.math.softplus(z))
    return _loss


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


# Register all custom objects
tf.keras.utils.get_custom_objects().update({
    'AudioFeatureNormalization': AudioFeatureNormalization,
    'XFakeSong>AudioFeatureNormalization': AudioFeatureNormalization,
    'MagnitudeLayer': MagnitudeLayer,
    'XFakeSong>MagnitudeLayer': MagnitudeLayer,
    'ExpandDimsLayer': ExpandDimsLayer,
    'XFakeSong>ExpandDimsLayer': ExpandDimsLayer,
    'ResizeLayer': ResizeLayer,
    'XFakeSong>ResizeLayer': ResizeLayer,
    'RepeatChannelLayer': RepeatChannelLayer,
    'XFakeSong>RepeatChannelLayer': RepeatChannelLayer,
    'SafeEfficientNetInputLayer': SafeEfficientNetInputLayer,
    'XFakeSong>SafeEfficientNetInputLayer': SafeEfficientNetInputLayer,
    'AttentionLayer': AttentionLayer,
    'XFakeSong>AttentionLayer': AttentionLayer,
    'SincConvLayer': SincConvLayer,
    'XFakeSong>SincConvLayer': SincConvLayer,
    'ResidualBlock1D': ResidualBlock1D,
    'XFakeSong>ResidualBlock1D': ResidualBlock1D,
    'GATConvLayer': GATConvLayer,
    'XFakeSong>GATConvLayer': GATConvLayer,
    'GraphPoolLayer': GraphPoolLayer,
    'XFakeSong>GraphPoolLayer': GraphPoolLayer,
    'GraphReadoutLayer': GraphReadoutLayer,
    'XFakeSong>GraphReadoutLayer': GraphReadoutLayer,
    'HSGALLayer': HSGALLayer,
    'XFakeSong>HSGALLayer': HSGALLayer,
    'AMSoftmaxLayer': AMSoftmaxLayer,
    'XFakeSong>AMSoftmaxLayer': AMSoftmaxLayer,
    'OCSoftmaxLayer': OCSoftmaxLayer,
    'XFakeSong>OCSoftmaxLayer': OCSoftmaxLayer,
    'AttentionPoolingLayer': AttentionPoolingLayer,
    'XFakeSong>AttentionPoolingLayer': AttentionPoolingLayer,
    'ConvolutionStemLayer': ConvolutionStemLayer,
    'XFakeSong>ConvolutionStemLayer': ConvolutionStemLayer,
    'PreEmphasisLayer': PreEmphasisLayer,
    'XFakeSong>PreEmphasisLayer': PreEmphasisLayer,
    'DeltaFeatureLayer': DeltaFeatureLayer,
    'XFakeSong>DeltaFeatureLayer': DeltaFeatureLayer,
    'SqueezeExcitationBlock2D': SqueezeExcitationBlock2D,
    'XFakeSong>SqueezeExcitationBlock2D': SqueezeExcitationBlock2D,
    'CrossAttentionFusionLayer': CrossAttentionFusionLayer,
    'XFakeSong>CrossAttentionFusionLayer': CrossAttentionFusionLayer,
    'GatedFusionLayer': GatedFusionLayer,
    'XFakeSong>GatedFusionLayer': GatedFusionLayer,
})
@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ResidualBlock2D(layers.Layer):
    """Bloco residual 2D para os encoders AASIST/RawGAT-ST fiéis."""

    def __init__(self, out_channels, kernel_size=(3, 3), pool_size=(1, 3),
                 **kwargs):
        super().__init__(**kwargs)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(kernel_size)
        self.pool_size = tuple(pool_size)

    def build(self, input_shape):
        self.bn1 = layers.BatchNormalization(name=f"{self.name}_bn1")
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
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class SpectralPositionEmbedding(layers.Layer):
    """Embedding posicional treinável para os nós espectrais."""

    def build(self, input_shape):
        if input_shape[1] is None or input_shape[-1] is None:
            raise ValueError("SpectralPositionEmbedding exige dimensões estáticas")
        self.position = self.add_weight(
            name="position",
            shape=(1, int(input_shape[1]), int(input_shape[-1])),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + tf.cast(self.position, inputs.dtype)


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AdaptiveGraphResize(layers.Layer):
    """Projeção aprendível do eixo de nós para tamanho comum S/T.

    LEGADO (desde 2026-07-27) — nenhum builder a instancia. Uma projeção densa
    sobre o eixo de nós MISTURA nós arbitrariamente (não é operação de grafo) e
    não existe em Tak et al. (2021). O alinhamento de Gs/Gt antes da fusão
    element-wise passou a usar ``GraphPoolLayer(target_nodes=...)``, o mesmo
    top-k dos artigos. Mantida só para desserializar checkpoints antigos.
    """

    def __init__(self, target_nodes=12, **kwargs):
        super().__init__(**kwargs)
        self.target_nodes = int(target_nodes)

    def build(self, input_shape):
        if input_shape[1] is None:
            raise ValueError(
                "AdaptiveGraphResize exige número de nós estático"
            )
        self.projection = layers.Dense(
            self.target_nodes,
            use_bias=True,
            name=f"{self.name}_node_projection",
        )
        # Build explícito: como a camada define compute_output_shape(), o
        # Keras 3 pode montar o grafo funcional via inferência de shape sem
        # nunca tracejar call() — o que deixaria `projection` com built=False
        # (sem variáveis) até a primeira chamada real.
        self.projection.build((input_shape[0], input_shape[-1], input_shape[1]))
        super().build(input_shape)

    def call(self, inputs):
        transposed = tf.transpose(inputs, [0, 2, 1])
        projected = self.projection(transposed)
        return tf.transpose(projected, [0, 2, 1])

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            (input_shape[0], self.target_nodes, input_shape[-1])
        )

    def get_config(self):
        config = super().get_config()
        config.update({"target_nodes": self.target_nodes})
        return config


