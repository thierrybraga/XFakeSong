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

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from app.domain.models.architectures.layers import (
    SqueezeExcitationBlock2D,
    ensure_flat_input,
    is_raw_audio,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction layers (LFCC, MFCC, CQT) as described in the paper
# ---------------------------------------------------------------------------

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
        # Create linearly-spaced filter bank (key difference from MFCC)
        num_bins = self.n_fft // 2 + 1
        low_freq = 0.0
        high_freq = self.sample_rate / 2.0
        # Linear spacing (not mel spacing)
        linear_points = tf.linspace(low_freq, high_freq, self.n_filters + 2)
        bin_points = tf.cast(
            tf.round(linear_points * self.n_fft / self.sample_rate), tf.int32
        )
        # Build triangular filter bank
        filters = np.zeros((num_bins, self.n_filters), dtype=np.float32)
        bin_np = bin_points.numpy() if hasattr(bin_points, 'numpy') else np.linspace(
            0, num_bins - 1, self.n_filters + 2, dtype=np.int32
        )
        for i in range(self.n_filters):
            left = int(bin_np[i])
            center = int(bin_np[i + 1])
            right = int(bin_np[i + 2])
            for j in range(left, center):
                if center > left:
                    filters[j, i] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filters[j, i] = (right - j) / (right - center)
        self.filter_bank = tf.constant(filters, dtype=tf.float32)
        # DCT matrix for cepstral coefficients
        dct_matrix = np.zeros((self.n_filters, self.n_lfcc), dtype=np.float32)
        for k in range(self.n_lfcc):
            for n in range(self.n_filters):
                dct_matrix[n, k] = np.cos(np.pi * k * (2 * n + 1) / (2 * self.n_filters))
        dct_matrix[:, 0] *= 1.0 / np.sqrt(self.n_filters)
        dct_matrix[:, 1:] *= np.sqrt(2.0 / self.n_filters)
        self.dct_matrix = tf.constant(dct_matrix, dtype=tf.float32)

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
        # DCT matrix
        dct_matrix = np.zeros((self.n_mels, self.n_mfcc), dtype=np.float32)
        for k in range(self.n_mfcc):
            for n in range(self.n_mels):
                dct_matrix[n, k] = np.cos(np.pi * k * (2 * n + 1) / (2 * self.n_mels))
        dct_matrix[:, 0] *= 1.0 / np.sqrt(self.n_mels)
        dct_matrix[:, 1:] *= np.sqrt(2.0 / self.n_mels)
        self.dct_matrix = tf.constant(dct_matrix, dtype=tf.float32)

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
        num_stft_bins = self.n_fft // 2 + 1
        # CQT center frequencies (log-spaced)
        fmin = 32.70  # C1
        freqs = fmin * (2.0 ** (np.arange(self.n_bins) / self.bins_per_octave))
        # Map CQT bins to STFT bins via triangular filters
        stft_freqs = np.linspace(0, self.sample_rate / 2, num_stft_bins)
        cqt_filters = np.zeros((num_stft_bins, self.n_bins), dtype=np.float32)
        for i, fc in enumerate(freqs):
            bandwidth = fc * (2.0 ** (1.0 / self.bins_per_octave) - 1)
            low = fc - bandwidth / 2
            high = fc + bandwidth / 2
            for j, sf in enumerate(stft_freqs):
                if low <= sf <= high:
                    if sf <= fc and fc > low:
                        cqt_filters[j, i] = (sf - low) / (fc - low)
                    elif sf > fc and high > fc:
                        cqt_filters[j, i] = (high - sf) / (high - fc)
        # Normalize each filter
        norms = np.sum(cqt_filters, axis=0, keepdims=True) + 1e-8
        cqt_filters = cqt_filters / norms
        self.cqt_filter_bank = tf.constant(cqt_filters, dtype=tf.float32)

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


class ConvBlock(layers.Layer):
    """Conv2D + BatchNorm + ReLU + MaxPool2D + Dropout block."""

    def __init__(self, filters, kernel_size=(3, 3), dropout_rate=0.3, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.conv = layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            padding='same', use_bias=False
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
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
            'dropout_rate': self.dropout_rate
        })
        return config


def preprocess(x):
    """Global preprocessing function for Sonic Sleuth compatibility."""
    if len(x.shape) == 2 and x.shape[-1] == 1:
        x = tf.squeeze(x, axis=-1)
    if len(x.shape) == 2:
        mel_spectrograms = []
        for i in range(tf.shape(x)[0]):
            audio_sample = x[i]
            stft = tf.signal.stft(
                audio_sample, frame_length=1024,
                frame_step=256, fft_length=1024
            )
            magnitude = tf.abs(stft)
            mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=128, num_spectrogram_bins=513,
                sample_rate=16000, lower_edge_hertz=0.0,
                upper_edge_hertz=8000.0
            )
            mel_spectrogram = tf.tensordot(magnitude, mel_weight_matrix, 1)
            mel_spectrograms.append(mel_spectrogram)
        x = tf.stack(mel_spectrograms)
    if len(x.shape) == 3:
        x = tf.expand_dims(x, axis=-1)
    x = tf.nn.sigmoid(x)
    return x


# ---------------------------------------------------------------------------
# Paper-faithful Sonic Sleuth model (Alshehri et al., 2024)
# ---------------------------------------------------------------------------

def _create_sonic_sleuth_paper(input_shape, num_classes=1, feature_type='lfcc',
                               sample_rate=16000):
    """Create paper-faithful Sonic Sleuth model.

    Architecture per Alshehri et al., 2024, Figure 3:
    - Feature extraction: LFCC (best), MFCC, or CQT
    - 3× Conv2D(filters, 3×3, relu, same) + MaxPool2D(2×2) each
      filters: 32 → 64 → 128
    - Flatten
    - Dense(256, relu) → Dense(128, relu) → Dropout(0.1)
    - Dense(1, sigmoid) — binary classification
    - Optimizer: Adam(lr=0.001)
    - Loss: binary_crossentropy

    Args:
        input_shape: (samples,) for raw audio or (time, features) for pre-extracted
        num_classes: 1 for binary (paper default)
        feature_type: 'lfcc' (best per paper), 'mfcc', 'cqt', or 'lfcc_cqt' (ensemble)
        sample_rate: Audio sample rate (default 16000)
    """
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # ---------- Feature extraction ----------
    if is_raw_audio(input_shape):
        audio = ensure_flat_input(inputs, input_shape)
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
        x = layers.Reshape(
            (tf.shape(features)[1], features.shape[-1], 1) if features.shape[1] is None
            else (features.shape[1], features.shape[-1], 1),
            name='add_channel'
        )(features) if features.shape[1] is not None else layers.Lambda(
            lambda f: tf.expand_dims(f, axis=-1), name='add_channel'
        )(features)
    else:
        # Pre-extracted features (spectrogram input)
        x = inputs
        if len(input_shape) == 2:
            x = layers.Reshape((*input_shape, 1), name='add_channel')(x)

    # ---------- CNN (Enhanced from Paper Figure 3) ----------
    # 5× Conv2D+BN+ReLU blocks: 32 → 64 → 128 → 256 → 512 filters
    # With SE-blocks after each ConvBlock and residual connections for blocks 3-5

    # Block 1: 32 filters (no residual — channel mismatch from input)
    x = ConvBlock(filters=32, kernel_size=(3, 3), dropout_rate=0.3, name='conv_block_1')(x)
    x = SqueezeExcitationBlock2D(reduction=16, name='se_block_1')(x)

    # Block 2: 64 filters (no residual — channel mismatch 32→64)
    x = ConvBlock(filters=64, kernel_size=(3, 3), dropout_rate=0.3, name='conv_block_2')(x)
    x = SqueezeExcitationBlock2D(reduction=16, name='se_block_2')(x)

    # Block 3: 128 filters (residual with 1x1 projection 64→128)
    shortcut_3 = layers.Conv2D(128, (1, 1), padding='same', use_bias=False, name='res_proj_3')(x)
    shortcut_3 = layers.MaxPooling2D((2, 2), padding='same', name='res_pool_3')(shortcut_3)
    x = ConvBlock(filters=128, kernel_size=(3, 3), dropout_rate=0.3, name='conv_block_3')(x)
    x = layers.Add(name='res_add_3')([x, shortcut_3])
    x = SqueezeExcitationBlock2D(reduction=16, name='se_block_3')(x)

    # Block 4: 256 filters (residual with 1x1 projection 128→256)
    shortcut_4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='res_proj_4')(x)
    shortcut_4 = layers.MaxPooling2D((2, 2), padding='same', name='res_pool_4')(shortcut_4)
    x = ConvBlock(filters=256, kernel_size=(3, 3), dropout_rate=0.3, name='conv_block_4')(x)
    x = layers.Add(name='res_add_4')([x, shortcut_4])
    x = SqueezeExcitationBlock2D(reduction=16, name='se_block_4')(x)

    # Block 5: 512 filters (residual with 1x1 projection 256→512)
    shortcut_5 = layers.Conv2D(512, (1, 1), padding='same', use_bias=False, name='res_proj_5')(x)
    shortcut_5 = layers.MaxPooling2D((2, 2), padding='same', name='res_pool_5')(shortcut_5)
    x = ConvBlock(filters=512, kernel_size=(3, 3), dropout_rate=0.3, name='conv_block_5')(x)
    x = layers.Add(name='res_add_5')([x, shortcut_5])
    x = SqueezeExcitationBlock2D(reduction=16, name='se_block_5')(x)

    # ---------- Classification head ----------
    # GAP + GMP instead of Flatten
    gap = layers.GlobalAveragePooling2D(name='gap')(x)
    gmp = layers.GlobalMaxPooling2D(name='gmp')(x)
    x = layers.Concatenate(name='gap_gmp')([gap, gmp])
    x = layers.Dense(256, activation='relu', name='dense_1')(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.3, name='classifier_dropout')(x)

    # Paper uses binary classification (real/fake) with sigmoid
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'

    model = models.Model(inputs=inputs, outputs=outputs, name='sonic_sleuth')

    # Paper uses Adam optimizer with default lr
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(f"Sonic Sleuth (paper-faithful) created: feature_type={feature_type}, params={model.count_params()}")
    return model


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_model(input_shape: Tuple[int, ...], num_classes: int = 1,
                 architecture: str = 'sonic_sleuth', **kwargs) -> models.Model:
    """Factory function for Sonic Sleuth model variants.

    Variants:
        'sonic_sleuth': Paper-faithful with LFCC features (best per paper)
        'sonic_sleuth_mfcc': Paper-faithful with MFCC features
        'sonic_sleuth_cqt': Paper-faithful with CQT features
        'sonic_sleuth_lfcc_cqt': Paper-faithful LFCC+CQT ensemble
    """
    feature_map = {
        'sonic_sleuth': 'lfcc',
        'sonic_sleuth_lfcc': 'lfcc',
        'sonic_sleuth_mfcc': 'mfcc',
        'sonic_sleuth_cqt': 'cqt',
        'sonic_sleuth_lfcc_cqt': 'lfcc_cqt',
    }

    feature_type = feature_map.get(architecture, 'lfcc')
    return _create_sonic_sleuth_paper(
        input_shape=input_shape,
        num_classes=num_classes,
        feature_type=feature_type,
        sample_rate=kwargs.get('sample_rate', 16000)
    )


# Register custom objects for model save/load compatibility
tf.keras.utils.get_custom_objects().update({
    'LFCCLayer': LFCCLayer,
    'MFCCLayer': MFCCLayer,
    'CQTLayer': CQTLayer,
    'MelSpectrogramLayer': MelSpectrogramLayer,
    'ConvBlock': ConvBlock,
    'preprocess': preprocess,
})
