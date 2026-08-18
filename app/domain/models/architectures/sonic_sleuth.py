"""Sonic Sleuth Architecture Implementation

Paper-faithful implementation based on:
Alshehri, A.; Almalki, D.; Alharbi, E.; Albaradei, S.
"Audio Deep Fake Detection with Sonic Sleuth Model"
MDPI Computers, 2024, 13(10), 256. DOI: 10.3390/computers13100256

Architecture (from Figure 3 of the paper):
- Feature extraction: LFCC, MFCC, and CQT spectrograms
- CNN: 3× Conv2D (32→64→128 filters, 3×3 kernel) + MaxPool2D(2×2) each
- Flatten → Dense(256) → Dense(128) → Dropout(0.1) → Dense(1, sigmoid)
- Binary classification (real vs fake)

Best result: LFCC achieves 98.27% accuracy, 0.016 EER on ASVspoof2019+In-the-Wild+FakeAVCeleb.
"""

# Third-party imports
import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from app.domain.models.architectures.layers import (
    ExpandDimsLayer,
    SqueezeExcitationBlock2D,
    cqt_triangular_filterbank,
    dct_matrix,
    ensure_flat_input,
    is_raw_audio,
    linear_triangular_filterbank,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction layers (LFCC, MFCC, CQT) as described in the paper
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class LFCCLayer(layers.Layer):
    """Linear Frequency Cepstral Coefficients (LFCC) extraction layer.

    Paper states LFCC achieved the best performance (98.27% accuracy, 0.016 EER).
    LFCC uses linearly-spaced filter banks instead of mel-scale, providing
    superior spectral resolution at high frequencies for capturing deepfake artifacts.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 n_filters=20, n_lfcc=20, **kwargs):
        super(LFCCLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_filters = n_filters
        self.n_lfcc = n_lfcc

    def build(self, input_shape):
        super().build(input_shape)
        # CORREÇÃO: o filterbank era montado com `tf.linspace(...).numpy()` e,
        # quando o tensor não era eager (build em graph mode), caía num
        # `np.linspace` de FALLBACK SILENCIOSO — produzindo um banco de filtros
        # DIFERENTE do pretendido, sem qualquer aviso. Agora usa os helpers
        # compartilhados em numpy puro (mesma construção do ensemble.py).
        self.filter_bank = tf.constant(
            linear_triangular_filterbank(
                self.n_fft, self.sample_rate, self.n_filters
            ),
            dtype=tf.float32,
        )
        self.dct_matrix = tf.constant(
            dct_matrix(self.n_filters, self.n_lfcc), dtype=tf.float32
        )

    def call(self, inputs):
        # STFT
        stft = tf.signal.stft(
            inputs, frame_length=self.n_fft,
            frame_step=self.hop_length, fft_length=self.n_fft
        )
        power_spectrum = tf.square(tf.abs(stft))
        # Apply linear filter bank
        filtered = tf.matmul(power_spectrum, self.filter_bank)
        # Log energy
        log_filtered = tf.math.log(filtered + 1e-6)
        # DCT to get cepstral coefficients
        lfcc = tf.matmul(log_filtered, self.dct_matrix)
        return lfcc

    def get_config(self):
        config = super().get_config()
        config.update({
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_filters': self.n_filters,
            'n_lfcc': self.n_lfcc,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class MFCCLayer(layers.Layer):
    """Mel-Frequency Cepstral Coefficients (MFCC) extraction layer.

    Paper reports MFCC achieved 98.04% accuracy, 0.0185 EER.
    Uses mel-scale filter banks based on human auditory perception.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 n_mels=40, n_mfcc=20, **kwargs):
        super(MFCCLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def build(self, input_shape):
        super().build(input_shape)
        # DCT matrix (helper compartilhado — ver layers.py)
        self.dct_matrix = tf.constant(
            dct_matrix(self.n_mels, self.n_mfcc), dtype=tf.float32
        )

    def call(self, inputs):
        stft = tf.signal.stft(
            inputs, frame_length=self.n_fft,
            frame_step=self.hop_length, fft_length=self.n_fft
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
        mfcc = tf.matmul(log_mel, self.dct_matrix)
        return mfcc

    def get_config(self):
        config = super().get_config()
        config.update({
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'n_mfcc': self.n_mfcc,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class CQTLayer(layers.Layer):
    """Constant-Q Transform (CQT) feature extraction layer.

    Paper reports CQT achieved 94.15% accuracy, 0.0757 EER alone,
    but CQT+LFCC ensemble achieves 84.92% on external data (complementary features).
    CQT provides logarithmic frequency resolution: higher resolution at low frequencies,
    better time resolution at high frequencies.

    Approximated via STFT with log-spaced frequency binning for TF graph compatibility.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 n_bins=84, bins_per_octave=12, **kwargs):
        super(CQTLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave

    def build(self, input_shape):
        super().build(input_shape)
        self.cqt_filter_bank = tf.constant(
            cqt_triangular_filterbank(
                self.n_fft, self.sample_rate, self.n_bins, self.bins_per_octave
            ),
            dtype=tf.float32,
        )

    def call(self, inputs):
        stft = tf.signal.stft(
            inputs, frame_length=self.n_fft,
            frame_step=self.hop_length, fft_length=self.n_fft
        )
        magnitude = tf.abs(stft)
        cqt = tf.matmul(magnitude, self.cqt_filter_bank)
        log_cqt = tf.math.log(cqt + 1e-6)
        return log_cqt

    def get_config(self):
        config = super().get_config()
        config.update({
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_bins': self.n_bins,
            'bins_per_octave': self.bins_per_octave,
        })
        return config


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class MelSpectrogramLayer(layers.Layer):
    """Custom layer to convert audio to mel spectrogram (legacy compatibility)."""

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256,
                 n_mels=128, **kwargs):
        super(MelSpectrogramLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def call(self, inputs):
        stft = tf.signal.stft(
            inputs, frame_length=self.n_fft,
            frame_step=self.hop_length, fft_length=self.n_fft
        )
        magnitude = tf.abs(stft)
        mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=(self.n_fft // 2) + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=self.sample_rate / 2.0
        )
        mel_spectrogram = tf.tensordot(magnitude, mel_weight_matrix, 1)
        mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=-1)
        return mel_spectrogram

    def get_config(self):
        config = super().get_config()
        config.update({
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ConvBlock(layers.Layer):
    """Conv2D [+ BatchNorm] + ReLU + MaxPool2D + Dropout block.

    ``use_batch_norm`` existe para permitir a configuração LITERAL da Figura 3
    do paper (Conv2D + MaxPool2D, sem BN). Default True preserva os modelos já
    treinados e desserializa configs antigas sem a chave.
    """

    def __init__(self, filters, kernel_size=(3, 3), dropout_rate=0.3,
                 use_batch_norm=True, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = bool(use_batch_norm)
        self.conv = layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            padding='same', use_bias=not self.use_batch_norm
        )
        self.bn = layers.BatchNormalization() if self.use_batch_norm else None
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        self.conv.build(input_shape)
        conv_shape = self.conv.compute_output_shape(input_shape)
        if self.bn is not None:
            self.bn.build(conv_shape)
        self.relu.build(conv_shape)
        pool_shape = self.maxpool.compute_output_shape(conv_shape)
        self.dropout.build(pool_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        if self.bn is not None:
            x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
        })
        return config


@tf.keras.utils.register_keras_serializable(
    package="XFakeSong", name="sonic_sleuth_preprocess"
)
def preprocess(x):
    """Global preprocessing function for Sonic Sleuth compatibility.

    Registrada com nome QUALIFICADO: várias arquiteturas exportavam funções
    diferentes sob a chave global 'preprocess' e a última importação vencia.

    Vetorizado: `tf.signal.stft` opera direto no batch (B, T) — o loop
    Python anterior (`for i in range(tf.shape(x)[0])`) quebrava em graph
    mode (range sobre tensor simbólico).
    """
    if len(x.shape) == 2 and x.shape[-1] == 1:
        x = tf.squeeze(x, axis=-1)
    if len(x.shape) == 2:
        stft = tf.signal.stft(
            x, frame_length=1024, frame_step=256, fft_length=1024
        )
        magnitude = tf.abs(stft)
        mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=128, num_spectrogram_bins=513,
            sample_rate=16000, lower_edge_hertz=0.0,
            upper_edge_hertz=8000.0
        )
        x = tf.tensordot(magnitude, mel_weight_matrix, 1)
    if len(x.shape) == 3:
        x = tf.expand_dims(x, axis=-1)
    x = tf.nn.sigmoid(x)
    return x


# ---------------------------------------------------------------------------
# Paper-faithful Sonic Sleuth model (Alshehri et al., 2024)
# ---------------------------------------------------------------------------

#: Progressão de filtros dos blocos convolucionais. Os três primeiros
#: (32 → 64 → 128) são exatamente os da Figura 3 do paper; 256/512 pertencem à
#: extensão usada por este projeto (`num_conv_blocks=5`).
_SONIC_SLEUTH_FILTERS = [32, 64, 128, 256, 512]


def _create_sonic_sleuth_paper(input_shape, num_classes=1, feature_type='lfcc',
                               sample_rate=16000, num_conv_blocks=5,
                               dropout_rate=0.3, classifier_dropout=None,
                               use_batch_norm=True, use_residual=True,
                               use_se_blocks=True, use_gap_gmp=True,
                               learning_rate=1e-3,
                               architecture='sonic_sleuth'):
    """Create Sonic Sleuth model (Alshehri et al., 2024).

    Configuração do paper (Figura 3), disponível via ``num_conv_blocks=3``,
    ``use_residual=False``, ``use_se_blocks=False``, ``use_gap_gmp=False``,
    ``use_batch_norm=False``, ``dropout_rate=0.0``, ``classifier_dropout=0.1``
    — é o que a variante ``sonic_sleuth_paper`` monta:
    - Feature extraction: LFCC (best), MFCC, or CQT
    - 3× Conv2D(filters, 3×3, relu, same) + MaxPool2D(2×2) each (32 → 64 → 128)
    - Flatten → Dense(256, relu) → Dense(128, relu) → Dropout(0.1)
    - Dense(1, sigmoid), Adam(lr=0.001), binary_crossentropy

    O DEFAULT deste projeto é a versão estendida (5 blocos + SE + residual +
    GAP/GMP), que é a que está treinada e promovida no benchmark.

    CORREÇÃO: estes parâmetros existiam no ``registry.default_params`` mas o
    builder os IGNORAVA (só lia ``sample_rate``) — a topologia era fixa no
    código e o dropout hardcoded em 0.3. Agora todos têm efeito real; os
    defaults abaixo reproduzem exatamente o comportamento anterior.

    Args:
        input_shape: (samples,) for raw audio or (time, features) for pre-extracted
        num_classes: 1 for binary (paper default)
        feature_type: 'lfcc' (best per paper), 'mfcc', 'cqt', or 'lfcc_cqt' (ensemble)
        sample_rate: Audio sample rate (default 16000)
        num_conv_blocks: 3 (paper) a 5 (extensão deste projeto)
        dropout_rate: dropout dentro de cada bloco convolucional
        classifier_dropout: dropout da cabeça (default: ``dropout_rate``)
        use_batch_norm: BatchNorm nos blocos (fora da Figura 3)
        use_residual: atalhos residuais a partir do 3º bloco
        use_se_blocks: Squeeze-and-Excitation após cada bloco
        use_gap_gmp: GAP+GMP concatenados no lugar de Flatten
        learning_rate: LR do Adam (paper: 1e-3)
    """
    num_conv_blocks = int(num_conv_blocks)
    if not 1 <= num_conv_blocks <= len(_SONIC_SLEUTH_FILTERS):
        raise ValueError(
            f"num_conv_blocks deve estar entre 1 e {len(_SONIC_SLEUTH_FILTERS)}"
        )
    if classifier_dropout is None:
        classifier_dropout = dropout_rate
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # ---------- Feature extraction ----------
    if is_raw_audio(input_shape):
        audio = ensure_flat_input(inputs)
        # Squeeze to (batch, time) if needed for STFT-based layers
        if len(input_shape) == 2 and input_shape[-1] == 1:
            audio = layers.Reshape((input_shape[0],), name='squeeze_channel')(audio)

        if feature_type == 'lfcc':
            # LFCC — best performance per paper (98.27% accuracy)
            features = LFCCLayer(
                sample_rate=sample_rate, n_fft=512, hop_length=160,
                n_filters=20, n_lfcc=20, name='lfcc_extraction'
            )(audio)
        elif feature_type == 'mfcc':
            # MFCC — 98.04% accuracy per paper
            features = MFCCLayer(
                sample_rate=sample_rate, n_fft=512, hop_length=160,
                n_mels=40, n_mfcc=20, name='mfcc_extraction'
            )(audio)
        elif feature_type == 'cqt':
            # CQT — 94.15% accuracy per paper
            features = CQTLayer(
                sample_rate=sample_rate, n_fft=512, hop_length=160,
                n_bins=84, bins_per_octave=12, name='cqt_extraction'
            )(audio)
        elif feature_type == 'lfcc_cqt':
            # LFCC + CQT ensemble (paper's best for external data: 84.92%)
            lfcc = LFCCLayer(
                sample_rate=sample_rate, n_fft=512, hop_length=160,
                n_filters=20, n_lfcc=20, name='lfcc_extraction'
            )(audio)
            cqt = CQTLayer(
                sample_rate=sample_rate, n_fft=512, hop_length=160,
                n_bins=84, bins_per_octave=12, name='cqt_extraction'
            )(audio)
            # Align time dimensions and concatenate features
            min_time = tf.minimum(tf.shape(lfcc)[1], tf.shape(cqt)[1])
            lfcc = lfcc[:, :min_time, :]
            cqt = cqt[:, :min_time, :]
            features = layers.Concatenate(axis=-1, name='lfcc_cqt_concat')([lfcc, cqt])
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}. Use 'lfcc', 'mfcc', 'cqt', or 'lfcc_cqt'.")

        # Add channel dimension for Conv2D: (batch, time, features) → (batch, time, features, 1)
        # ExpandDimsLayer (camada registrada) no lugar do `layers.Lambda`:
        # Lambda com lambda Python não é recarregável em safe_mode (Keras 3).
        if features.shape[1] is not None:
            x = layers.Reshape(
                (features.shape[1], features.shape[-1], 1), name='add_channel'
            )(features)
        else:
            x = ExpandDimsLayer(axis=-1, name='add_channel')(features)
    else:
        # Pre-extracted features (spectrogram input)
        x = inputs
        if len(input_shape) == 2:
            x = layers.Reshape((*input_shape, 1), name='add_channel')(x)

    # ---------- CNN ----------
    # Blocos Conv2D+[BN]+ReLU+MaxPool+Dropout com a progressão de filtros do
    # paper (32 → 64 → 128) estendida por 256 → 512. Residual (a partir do 3º
    # bloco) e SE são OPCIONAIS: desligados, o grafo é o da Figura 3.
    for index in range(num_conv_blocks):
        filters = _SONIC_SLEUTH_FILTERS[index]
        block_id = index + 1
        # Residual só a partir do 3º bloco: nos dois primeiros o número de
        # canais da entrada não bate com o do bloco.
        add_residual = use_residual and index >= 2
        if add_residual:
            shortcut = layers.Conv2D(
                filters, (1, 1), padding='same', use_bias=False,
                name=f'res_proj_{block_id}',
            )(x)
            shortcut = layers.MaxPooling2D((2, 2), name=f'res_pool_{block_id}')(shortcut)
        x = ConvBlock(
            filters=filters, kernel_size=(3, 3), dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm, name=f'conv_block_{block_id}',
        )(x)
        if add_residual:
            x = layers.Add(name=f'res_add_{block_id}')([x, shortcut])
        if use_se_blocks:
            x = SqueezeExcitationBlock2D(reduction=16, name=f'se_block_{block_id}')(x)

    # ---------- Classification head ----------
    if use_gap_gmp:
        gap = layers.GlobalAveragePooling2D(name='gap')(x)
        gmp = layers.GlobalMaxPooling2D(name='gmp')(x)
        x = layers.Concatenate(name='gap_gmp')([gap, gmp])
    else:
        x = layers.Flatten(name='flatten')(x)  # Figura 3 do paper
    x = layers.Dense(256, activation='relu', name='dense_1')(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.Dropout(classifier_dropout, name='classifier_dropout')(x)

    # Cabeça de saída PADRONIZADA: num_classes>=2 → softmax N-unidades
    # (convenção única do projeto). Só num_classes==1 usa sigmoid 1-unidade.
    # dtype='float32' para não saturar sob mixed_float16.
    if num_classes == 1:
        outputs = layers.Dense(
            1, activation='sigmoid', name='output', dtype='float32'
        )(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(
            num_classes, activation='softmax', name='output', dtype='float32'
        )(x)
        loss = 'sparse_categorical_crossentropy'

    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    # Paper uses Adam optimizer with default lr (1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(
        "Sonic Sleuth criado: feature_type=%s, blocos=%d, residual=%s, SE=%s, "
        "GAP+GMP=%s, BN=%s, dropout=%s, params=%d",
        feature_type, num_conv_blocks, use_residual, use_se_blocks,
        use_gap_gmp, use_batch_norm, dropout_rate, model.count_params(),
    )
    return model


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

#: Parâmetros da configuração LITERAL da Figura 3 (Alshehri et al., 2024).
_SONIC_SLEUTH_PAPER_CONFIG = {
    'num_conv_blocks': 3,       # 32 → 64 → 128
    'use_residual': False,
    'use_se_blocks': False,
    'use_gap_gmp': False,       # Flatten, como na figura
    'use_batch_norm': False,    # a figura mostra Conv2D + MaxPool apenas
    'dropout_rate': 0.0,        # sem dropout nos blocos
    'classifier_dropout': 0.1,  # único Dropout(0.1), antes da saída
    'learning_rate': 1e-3,
}

#: Chaves de construção aceitas por `_create_sonic_sleuth_paper`. Servem também
#: de contrato: qualquer outra chave enviada pelo registry é ignorada e AVISADA
#: (antes tudo caía em **kwargs silenciosamente e virava config morto).
_SONIC_SLEUTH_MODEL_KEYS = {
    'sample_rate', 'num_conv_blocks', 'dropout_rate', 'classifier_dropout',
    'use_batch_norm', 'use_residual', 'use_se_blocks', 'use_gap_gmp',
    'learning_rate',
}


def create_model(input_shape: Tuple[int, ...], num_classes: int = 1,
                 architecture: str = 'sonic_sleuth', **kwargs) -> models.Model:
    """Factory function for Sonic Sleuth model variants.

    Variants:
        'sonic_sleuth': LFCC + versão estendida (5 blocos + SE + residual)
        'sonic_sleuth_mfcc' / 'sonic_sleuth_cqt' / 'sonic_sleuth_lfcc_cqt':
            mesma topologia com outra representação de entrada
        'sonic_sleuth_paper': configuração LITERAL da Figura 3 do artigo
            (3 blocos 32/64/128, Flatten, Dropout(0.1), sem SE/residual/BN)
    """
    feature_map = {
        'sonic_sleuth': 'lfcc',
        'sonic_sleuth_lfcc': 'lfcc',
        'sonic_sleuth_paper': 'lfcc',
        'sonic_sleuth_mfcc': 'mfcc',
        'sonic_sleuth_cqt': 'cqt',
        'sonic_sleuth_lfcc_cqt': 'lfcc_cqt',
    }
    if architecture == 'default':
        architecture = 'sonic_sleuth'

    feature_type = feature_map.get(architecture, 'lfcc')

    params = dict(_SONIC_SLEUTH_PAPER_CONFIG) if architecture == 'sonic_sleuth_paper' else {}
    ignored = sorted(set(kwargs) - _SONIC_SLEUTH_MODEL_KEYS)
    if ignored:
        logger.warning(
            "Sonic Sleuth: parâmetros ignorados (não fazem parte da "
            "construção do modelo): %s", ignored,
        )
    # A CONFIGURAÇÃO DA VARIANTE PREVALECE sobre kwargs. Os kwargs chegam aqui
    # tanto de um override explícito quanto do `registry.default_params` — que
    # descrevem a variante PADRÃO (5 blocos + SE + residual). Sem esta regra,
    # pedir 'sonic_sleuth_paper' pelo registry/factory devolvia silenciosamente
    # o modelo estendido com o nome do paper.
    params.update({
        k: v for k, v in kwargs.items()
        if k in _SONIC_SLEUTH_MODEL_KEYS and k not in params
    })

    return _create_sonic_sleuth_paper(
        input_shape=input_shape,
        num_classes=num_classes,
        feature_type=feature_type,
        architecture=architecture,
        **params,
    )


# Register custom objects for model save/load compatibility.
# `preprocess` é registrado com PREFIXO de módulo: rawnet2.py, multiscale_cnn.py
# e wavlm.py registravam funções DIFERENTES sob a mesma chave global
# 'preprocess', e a última importação vencia — um .keras podia ser recarregado
# com o pré-processamento de outra arquitetura. A chave curta some.
tf.keras.utils.get_custom_objects().update({
    'LFCCLayer': LFCCLayer,
    'MFCCLayer': MFCCLayer,
    'CQTLayer': CQTLayer,
    'MelSpectrogramLayer': MelSpectrogramLayer,
    'ConvBlock': ConvBlock,
    'XFakeSong>sonic_sleuth_preprocess': preprocess,
})
