"""Ensemble Architecture — Multi-Spectrogram Feature Fusion

Literature-based implementation combining:

1. Pham et al. (2024) — "Deepfake Audio Detection Using Spectrogram-based Feature
   and Ensemble of Deep Learning Models" (arXiv:2407.01777)
   - Multiple spectrogram representations (STFT, CQT + Mel/Gammatone/Linear filters)
   - CNN classifiers per representation
   - Fusion of high-performing models achieves EER 0.03 on ASVspoof 2019

2. ResNeXt + MLP Fusion (Knowledge-Based Systems, 2025)
   - Three spectral features (LFCC, MFCC, CQCC) processed independently
   - MLP-based fusion of embeddings
   - EER 1.05% on ASVspoof 2019 LA

Architecture (multi-feature ensemble):
- Input: raw audio (16kHz)
- Branch 1: Mel spectrogram (128 mels) -> CNN+SE -> embedding
- Branch 2: LFCC (20 coefficients) -> CNN+SE -> embedding
- Branch 3: CQT (84 bins) -> CNN+SE -> embedding
- Branch 4: MFCC (20 coefficients) -> CNN+SE -> embedding
- Cross-attention + gated fusion -> Dense(512) -> Dense(256) -> Dense(128) -> classification
- Score-level fusion variant: learnable weighted average of branch predictions
- Optimizer: AdamW(lr=5e-5, weight_decay=1e-5)
"""

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from app.domain.models.architectures.layers import (
    CrossAttentionFusionLayer,
    GatedFusionLayer,
    SqueezeExcitationBlock2D,
    create_classification_head,
    ensure_flat_input,
    is_raw_audio,
)

logger = logging.getLogger(__name__)


# ============================ FEATURE EXTRACTION LAYERS ============================

class MelSpectrogramBranch(layers.Layer):
    """Mel spectrogram extraction per Pham et al. (2024).

    128 mel bins, 512 FFT, 160 hop (10ms at 16kHz), log-power.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 n_mels=128, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def call(self, inputs):
        stft = tf.signal.stft(
            inputs, frame_length=self.n_fft,
            frame_step=self.hop_length, fft_length=self.n_fft
        )
        power = tf.square(tf.abs(stft))
        mel_weight = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=self.sample_rate / 2.0
        )
        mel_spec = tf.matmul(power, mel_weight)
        log_mel = tf.math.log(mel_spec + 1e-6)
        # Add channel dim: (batch, time, mels, 1)
        return tf.expand_dims(log_mel, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
        })
        return config


class LFCCBranch(layers.Layer):
    """Linear Frequency Cepstral Coefficients extraction.

    LFCC provides superior spectral resolution at high frequencies
    for capturing deepfake artifacts. Used in ResNeXt fusion paper.
    20 linearly-spaced filters, 20 LFCC coefficients.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 n_filters=20, n_lfcc=20, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_filters = n_filters
        self.n_lfcc = n_lfcc

    def build(self, input_shape):
        super().build(input_shape)
        num_bins = self.n_fft // 2 + 1
        low_freq = 0.0
        high_freq = self.sample_rate / 2.0
        linear_points = np.linspace(low_freq, high_freq, self.n_filters + 2)
        bin_points = np.round(linear_points * self.n_fft / self.sample_rate).astype(np.int32)

        filters = np.zeros((num_bins, self.n_filters), dtype=np.float32)
        for i in range(self.n_filters):
            left, center, right = int(bin_points[i]), int(bin_points[i + 1]), int(bin_points[i + 2])
            for j in range(left, center):
                if center > left:
                    filters[j, i] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filters[j, i] = (right - j) / (right - center)
        self.filter_bank = tf.constant(filters, dtype=tf.float32)

        dct_matrix = np.zeros((self.n_filters, self.n_lfcc), dtype=np.float32)
        for k in range(self.n_lfcc):
            for n in range(self.n_filters):
                dct_matrix[n, k] = np.cos(np.pi * k * (2 * n + 1) / (2 * self.n_filters))
        dct_matrix[:, 0] *= 1.0 / np.sqrt(self.n_filters)
        dct_matrix[:, 1:] *= np.sqrt(2.0 / self.n_filters)
        self.dct_matrix = tf.constant(dct_matrix, dtype=tf.float32)

    def call(self, inputs):
        stft = tf.signal.stft(
            inputs, frame_length=self.n_fft,
            frame_step=self.hop_length, fft_length=self.n_fft
        )
        power = tf.square(tf.abs(stft))
        filtered = tf.matmul(power, self.filter_bank)
        log_filtered = tf.math.log(filtered + 1e-6)
        lfcc = tf.matmul(log_filtered, self.dct_matrix)
        return tf.expand_dims(lfcc, axis=-1)

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


class CQTBranch(layers.Layer):
    """Constant-Q Transform feature extraction.

    CQT provides logarithmic frequency resolution.
    84 bins, 12 bins per octave (7 octaves from C1).
    Approximated via STFT with log-spaced frequency binning.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 n_bins=84, bins_per_octave=12, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave

    def build(self, input_shape):
        super().build(input_shape)
        num_stft_bins = self.n_fft // 2 + 1
        fmin = 32.70  # C1
        freqs = fmin * (2.0 ** (np.arange(self.n_bins) / self.bins_per_octave))
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
        return tf.expand_dims(log_cqt, axis=-1)

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


class MFCCBranch(layers.Layer):
    """Mel-Frequency Cepstral Coefficients extraction.

    MFCC uses mel-scale filter banks + DCT. Provides complementary
    information to LFCC's linearly-spaced filters. 40 mel bins, 20 coefficients.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 n_mels=40, n_mfcc=20, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def build(self, input_shape):
        super().build(input_shape)
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
        power = tf.square(tf.abs(stft))
        mel_weight = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=self.sample_rate / 2.0
        )
        mel_spec = tf.matmul(power, mel_weight)
        log_mel = tf.math.log(mel_spec + 1e-6)
        mfcc = tf.matmul(log_mel, self.dct_matrix)
        # Add channel dim: (batch, time, coeffs, 1)
        return tf.expand_dims(mfcc, axis=-1)

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


# ============================ CNN CLASSIFIER BRANCH ============================

def _create_cnn_branch(x, branch_name, filters=None):
    """Create a CNN branch with SE-blocks for feature classification.

    Per Pham et al.: CNN classifier on spectrogram features.
    4x Conv2D + BN + ReLU + SE + MaxPool -> GAP -> Dense(256) embedding.
    """
    if filters is None:
        filters = [32, 64, 128, 256]

    for i, f in enumerate(filters):
        x = layers.Conv2D(
            f, (3, 3), padding='same', use_bias=False,
            name=f'{branch_name}_conv_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'{branch_name}_bn_{i}')(x)
        x = layers.ReLU(name=f'{branch_name}_relu_{i}')(x)
        x = SqueezeExcitationBlock2D(reduction=16, name=f'{branch_name}_se_{i}')(x)
        x = layers.MaxPooling2D((2, 2), name=f'{branch_name}_pool_{i}')(x)

    x = layers.GlobalAveragePooling2D(name=f'{branch_name}_gap')(x)
    x = layers.Dense(256, activation='relu', name=f'{branch_name}_embed')(x)
    x = layers.Dropout(0.3, name=f'{branch_name}_dropout')(x)
    return x


# ============================ ENSEMBLE FUSION LAYERS ============================

class ScoreFusionLayer(layers.Layer):
    """Learnable weighted score-level fusion.

    Per Pham et al.: fuse predictions from multiple branches
    with softmax-normalized learnable weights.
    """

    def __init__(self, num_branches, **kwargs):
        super().__init__(**kwargs)
        self.num_branches = num_branches

    def build(self, input_shape):
        super().build(input_shape)
        self.branch_weights = self.add_weight(
            name='branch_weights',
            shape=(self.num_branches,),
            initializer='ones',
            trainable=True
        )

    def call(self, inputs):
        # inputs: list of (batch, num_classes) tensors
        weights = tf.nn.softmax(self.branch_weights)
        stacked = tf.stack(inputs, axis=-1)  # (batch, classes, branches)
        weighted = stacked * weights[None, None, :]
        return tf.reduce_sum(weighted, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'num_branches': self.num_branches})
        return config


# ============================ MODEL BUILDERS ============================

def _create_ensemble_feature_fusion(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    dropout_rate: float = 0.3,
    architecture: str = 'ensemble'
) -> models.Model:
    """Create multi-feature ensemble with cross-attention + gated MLP fusion.

    Architecture per Pham et al. (2024) + ResNeXt fusion paper:
    1. Four parallel feature extraction branches (Mel, LFCC, CQT, MFCC)
    2. CNN+SE classifier per branch -> 256-dim embeddings
    3. Cross-attention fusion + gated fusion -> MLP fusion -> classification

    This is the "feature-level fusion" approach from the literature.
    """
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # Flatten to 1D audio if needed
    if is_raw_audio(input_shape):
        audio = ensure_flat_input(inputs, input_shape)
    else:
        # For pre-extracted features, use single-branch CNN
        x = inputs
        if len(input_shape) == 2:
            x = layers.Reshape((*input_shape, 1), name='add_channel')(x)
        x = _create_cnn_branch(x, 'single', filters=[32, 64, 128, 256])
        outputs, loss = create_classification_head(
            x, num_classes, dropout_rate=dropout_rate, hidden_dims=[256, 128]
        )
        model = models.Model(inputs=inputs, outputs=outputs, name=architecture)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-5),
            loss=loss, metrics=['accuracy']
        )
        return model

    # ---------- Branch 1: Mel Spectrogram (128 mels) ----------
    mel_features = MelSpectrogramBranch(
        sample_rate=16000, n_fft=512, hop_length=160, n_mels=128,
        name='mel_extraction'
    )(audio)
    mel_embedding = _create_cnn_branch(mel_features, 'mel', filters=[32, 64, 128, 256])

    # ---------- Branch 2: LFCC (20 coefficients) ----------
    lfcc_features = LFCCBranch(
        sample_rate=16000, n_fft=512, hop_length=160,
        n_filters=20, n_lfcc=20, name='lfcc_extraction'
    )(audio)
    lfcc_embedding = _create_cnn_branch(lfcc_features, 'lfcc', filters=[32, 64, 128, 256])

    # ---------- Branch 3: CQT (84 bins) ----------
    cqt_features = CQTBranch(
        sample_rate=16000, n_fft=512, hop_length=160,
        n_bins=84, bins_per_octave=12, name='cqt_extraction'
    )(audio)
    cqt_embedding = _create_cnn_branch(cqt_features, 'cqt', filters=[32, 64, 128, 256])

    # ---------- Branch 4: MFCC (20 coefficients) ----------
    mfcc_features = MFCCBranch(
        sample_rate=16000, n_fft=512, hop_length=160,
        n_mels=40, n_mfcc=20, name='mfcc_extraction'
    )(audio)
    mfcc_embedding = _create_cnn_branch(mfcc_features, 'mfcc', filters=[32, 64, 128, 256])

    # ---------- Cross-Attention + Gated Fusion ----------
    branch_outputs = [mel_embedding, lfcc_embedding, cqt_embedding, mfcc_embedding]

    # Cross-attention fusion
    fused = CrossAttentionFusionLayer(
        embed_dim=128, num_heads=4, name='cross_attn_fusion'
    )(branch_outputs)

    # Gated fusion
    gated = GatedFusionLayer(name='gated_fusion')(branch_outputs)
    fused = layers.Concatenate(name='fusion_concat')([fused, gated])

    # MLP fusion head
    fused = layers.Dense(512, activation='relu', name='fusion_dense_1')(fused)
    fused = layers.BatchNormalization(name='fusion_bn_1')(fused)
    fused = layers.Dropout(dropout_rate, name='fusion_dropout_1')(fused)
    fused = layers.Dense(256, activation='relu', name='fusion_dense_2')(fused)
    fused = layers.BatchNormalization(name='fusion_bn_2')(fused)
    fused = layers.Dropout(dropout_rate * 0.67, name='fusion_dropout_2')(fused)
    fused = layers.Dense(128, activation='relu', name='fusion_dense_3')(fused)
    fused = layers.Dropout(dropout_rate * 0.5, name='fusion_dropout_3')(fused)

    # Classification
    if num_classes == 1 or num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(fused)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(fused)
        loss = 'sparse_categorical_crossentropy'

    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=5e-5,
            weight_decay=1e-5
        ),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(f"Ensemble (feature fusion) created: 4 branches (Mel+LFCC+CQT+MFCC), params={model.count_params()}")
    return model


def _create_ensemble_score_fusion(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    dropout_rate: float = 0.3,
    architecture: str = 'ensemble_score'
) -> models.Model:
    """Create multi-feature ensemble with score-level fusion.

    Architecture per Pham et al. (2024):
    1. Three parallel branches (Mel, LFCC, CQT) each with own classifier
    2. Learnable weighted average of branch predictions

    Score-level fusion is simpler and often competitive with feature-level.
    """
    inputs = layers.Input(shape=input_shape, name='audio_input')

    if is_raw_audio(input_shape):
        audio = ensure_flat_input(inputs, input_shape)
    else:
        # Fallback for non-raw input
        return _create_ensemble_feature_fusion(
            input_shape, num_classes, dropout_rate, architecture
        )

    # Determine output activation and loss
    if num_classes == 1 or num_classes == 2:
        out_units = 1
        out_activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        out_units = num_classes
        out_activation = 'softmax'
        loss = 'sparse_categorical_crossentropy'

    # ---------- Branch 1: Mel ----------
    mel_features = MelSpectrogramBranch(
        sample_rate=16000, n_fft=512, hop_length=160, n_mels=128,
        name='mel_extraction'
    )(audio)
    mel_emb = _create_cnn_branch(mel_features, 'mel', filters=[32, 64, 128])
    mel_pred = layers.Dense(out_units, activation=out_activation, name='mel_output')(mel_emb)

    # ---------- Branch 2: LFCC ----------
    lfcc_features = LFCCBranch(
        sample_rate=16000, n_fft=512, hop_length=160,
        n_filters=20, n_lfcc=20, name='lfcc_extraction'
    )(audio)
    lfcc_emb = _create_cnn_branch(lfcc_features, 'lfcc', filters=[32, 64, 128])
    lfcc_pred = layers.Dense(out_units, activation=out_activation, name='lfcc_output')(lfcc_emb)

    # ---------- Branch 3: CQT ----------
    cqt_features = CQTBranch(
        sample_rate=16000, n_fft=512, hop_length=160,
        n_bins=84, bins_per_octave=12, name='cqt_extraction'
    )(audio)
    cqt_emb = _create_cnn_branch(cqt_features, 'cqt', filters=[32, 64, 128])
    cqt_pred = layers.Dense(out_units, activation=out_activation, name='cqt_output')(cqt_emb)

    # ---------- Score-level fusion ----------
    outputs = ScoreFusionLayer(
        num_branches=3, name='score_fusion'
    )([mel_pred, lfcc_pred, cqt_pred])

    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=5e-5,
            weight_decay=1e-5
        ),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(f"Ensemble (score fusion) created: 3 branches (Mel+LFCC+CQT), params={model.count_params()}")
    return model


def _create_ensemble_lite(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    architecture: str = 'ensemble_lite'
) -> models.Model:
    """Lightweight ensemble with 2 branches (Mel + LFCC) and smaller CNNs."""
    inputs = layers.Input(shape=input_shape, name='audio_input')

    if is_raw_audio(input_shape):
        audio = ensure_flat_input(inputs, input_shape)
    else:
        x = inputs
        if len(input_shape) == 2:
            x = layers.Reshape((*input_shape, 1), name='add_channel')(x)
        x = _create_cnn_branch(x, 'single', filters=[16, 32, 64])
        outputs, loss = create_classification_head(
            x, num_classes, dropout_rate=0.2, hidden_dims=[128]
        )
        model = models.Model(inputs=inputs, outputs=outputs, name=architecture)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=loss, metrics=['accuracy']
        )
        return model

    # 2 branches only (lighter)
    mel_features = MelSpectrogramBranch(
        sample_rate=16000, n_fft=512, hop_length=160, n_mels=64,
        name='mel_extraction'
    )(audio)
    mel_emb = _create_cnn_branch(mel_features, 'mel', filters=[16, 32, 64])

    lfcc_features = LFCCBranch(
        sample_rate=16000, n_fft=512, hop_length=160,
        n_filters=20, n_lfcc=20, name='lfcc_extraction'
    )(audio)
    lfcc_emb = _create_cnn_branch(lfcc_features, 'lfcc', filters=[16, 32, 64])

    fused = layers.Concatenate(name='feature_concat')([mel_emb, lfcc_emb])
    fused = layers.Dense(128, activation='relu', name='fusion_dense')(fused)
    fused = layers.Dropout(0.2, name='fusion_dropout')(fused)

    if num_classes == 1 or num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(fused)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(fused)
        loss = 'sparse_categorical_crossentropy'

    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss, metrics=['accuracy']
    )

    logger.info(f"Ensemble Lite created: 2 branches (Mel+LFCC), params={model.count_params()}")
    return model


# ============================ FACTORY ============================

def create_model(input_shape: Tuple[int, ...], num_classes: int = 1,
                 architecture: str = 'ensemble', **kwargs) -> models.Model:
    """Factory function for ensemble model variants.

    Variants:
        'ensemble': Feature-level fusion (Mel + LFCC + CQT -> MLP fusion)
        'ensemble_score': Score-level fusion (learnable weighted average)
        'ensemble_lite': Lightweight 2-branch (Mel + LFCC)
    """
    if architecture == 'ensemble_score':
        return _create_ensemble_score_fusion(
            input_shape, num_classes, architecture=architecture
        )
    elif architecture == 'ensemble_lite':
        return _create_ensemble_lite(
            input_shape, num_classes, architecture=architecture
        )
    else:
        return _create_ensemble_feature_fusion(
            input_shape, num_classes, architecture=architecture,
            **kwargs
        )


# Register custom layers
tf.keras.utils.get_custom_objects().update({
    'MelSpectrogramBranch': MelSpectrogramBranch,
    'LFCCBranch': LFCCBranch,
    'CQTBranch': CQTBranch,
    'MFCCBranch': MFCCBranch,
    'ScoreFusionLayer': ScoreFusionLayer,
})
