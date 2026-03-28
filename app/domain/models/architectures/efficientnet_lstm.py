"""EfficientNet-LSTM Architecture Implementation

Literature-based implementation combining:
1. EfficientNet-B0 spatial feature extraction on spectrograms
   Ref: "Deepfake Audio Detection Using Spectrogram-based Feature and
   Ensemble of Deep Learning Models" (arXiv:2407.01777, 2024)
   - EfficientNet-B0 as transfer learning backbone on spectrograms
   - Evaluated on ASVspoof 2019, EER 0.03 (ensemble)

2. LSTM temporal modeling for sequential audio analysis
   Ref: "Hybrid CNN-LSTM Architectures for Deepfake Audio Detection
   Using MFCC and Spectrogram Analysis" (Science Publishing Group, 2025)
   - BiLSTM [256, 128] units, Adam lr=0.0001, dropout=0.3
   - 94.7% accuracy on FoR, 93.2% on ASVspoof 2019

Architecture:
- Input: raw audio or mel spectrogram
- Mel spectrogram: n_fft=512, hop_length=160, n_mels=128 (16kHz, 25ms window)
- EfficientNet-B0 extracts spatial feature maps from spectrogram
- Feature maps pooled along frequency axis to preserve temporal structure
- Bidirectional LSTM [256, 128] models temporal dependencies
- Attention mechanism focuses on key temporal segments
- Dense(256) -> Dropout(0.3) -> classification output
- Optimizer: Adam(lr=0.0001)
"""

# Third-party imports
import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from app.domain.models.architectures.layers import (
    AttentionLayer,
    DeltaFeatureLayer,
    ExpandDimsLayer,
    RepeatChannelLayer,
    ResizeLayer,
    create_classification_head,
    ensure_flat_input,
    is_raw_audio,
)

logger = logging.getLogger(__name__)


# ============================ CUSTOM LAYERS ============================

class MelSpectrogramFrontEnd(layers.Layer):
    """Mel spectrogram extraction matching CNN-LSTM paper parameters.

    Params from literature:
    - n_fft=512 (512-point FFT)
    - hop_length=160 (10ms at 16kHz)
    - n_mels=128 (128 mel-scaled triangular filters)
    - Hamming window
    - Log-power scaling
    - Frequency range: 0 to Nyquist (sample_rate / 2)
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 n_mels=128, **kwargs):
        super(MelSpectrogramFrontEnd, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def call(self, inputs):
        if len(inputs.shape) == 3 and inputs.shape[-1] == 1:
            inputs = tf.squeeze(inputs, axis=-1)
        # STFT with Hamming window (paper specifies Hamming)
        window = tf.signal.hamming_window(self.n_fft)
        stft = tf.signal.stft(
            inputs,
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=lambda size, dtype: tf.cast(window, dtype)
        )
        # Power spectrum
        power = tf.square(tf.abs(stft))

        # Mel filter bank (0 to Nyquist)
        mel_weight = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=self.sample_rate / 2.0
        )
        mel_spec = tf.matmul(power, mel_weight)

        # Log-power scaling (paper uses log-power)
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


class TemporalPoolingLayer(layers.Layer):
    """Pool EfficientNet feature maps along frequency axis to get temporal sequence.

    EfficientNet output: (batch, H, W, C) where H=time, W=frequency, C=channels
    After pooling frequency: (batch, H, C) = temporal sequence of feature vectors
    """

    def __init__(self, **kwargs):
        super(TemporalPoolingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Average pool along the frequency (width) axis
        # Input: (batch, time_steps, freq_steps, channels)
        # Output: (batch, time_steps, channels)
        return tf.reduce_mean(inputs, axis=2)

    def get_config(self):
        return super().get_config()


# ============================ MODEL BUILDERS ============================

def _create_efficientnet_lstm_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    lstm_units: int = 256,
    dropout_rate: float = 0.3,
    architecture: str = 'efficientnet_lstm'
) -> models.Model:
    """Create EfficientNet-LSTM model based on literature.

    Architecture:
    1. Mel spectrogram front-end (128 mels, 0-Nyquist) + delta features (if raw audio)
    2. Resize to 224x224x3 for EfficientNet-B0
    3. EfficientNet-B0 extracts spatial feature maps (7x7xC), last 3 layers fine-tuned
    4. Pool frequency axis -> temporal sequence (7, C)
    5. Bidirectional LSTM [256, 128] for temporal modeling
    6. Attention mechanism
    7. Dense(256) -> Dropout(0.3) -> classification
    """
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # ---------- Front-end: audio -> spectrogram ----------
    if is_raw_audio(input_shape):
        audio = ensure_flat_input(inputs, input_shape)

        # Mel spectrogram (512 FFT, 160 hop, 128 mels, 0-Nyquist)
        x = MelSpectrogramFrontEnd(
            sample_rate=16000, n_fft=512, hop_length=160, n_mels=128,
            name='mel_spectrogram'
        )(audio)
        # (batch, time_frames, 128) -> 3 channels via delta features (static + delta + delta-delta)
        x = DeltaFeatureLayer(order=2, width=2, name='delta_features')(x)
        # Resize to EfficientNet input
        x = ResizeLayer(target_height=224, target_width=224, name='resize')(x)
    else:
        # 2D spectrogram input
        x = inputs
        if len(input_shape) == 2:
            # (batch, time, freq) -> 3 channels via delta features
            x = DeltaFeatureLayer(order=2, width=2, name='delta_features')(x)
            x = ResizeLayer(target_height=224, target_width=224, name='resize')(x)
        elif len(input_shape) == 3:
            x = ResizeLayer(target_height=224, target_width=224, name='resize')(x)
            if input_shape[-1] != 3:
                x = layers.Conv2D(3, (1, 1), name='channel_proj')(x)

    # ---------- EfficientNet-B0 feature extraction ----------
    # Per arXiv:2407.01777: EfficientNet-B0 as backbone on spectrograms
    efficientnet = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )
    # Unfreeze last 3 EfficientNet layers for fine-tuning
    for layer in efficientnet.layers[-3:]:
        layer.trainable = True
    # Feature maps: (batch, 7, 7, 1280)
    feature_maps = efficientnet(x)

    # ---------- Temporal sequence extraction ----------
    # Pool frequency axis to preserve temporal structure
    # (batch, 7, 7, 1280) -> (batch, 7, 1280) temporal sequence
    temporal_features = TemporalPoolingLayer(name='temporal_pool')(feature_maps)

    # ---------- Bidirectional LSTM ----------
    # BiLSTM with [256, 128] units
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate),
        name='bilstm_1'
    )(temporal_features)

    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True, dropout=dropout_rate),
        name='bilstm_2'
    )(lstm_out)

    # ---------- Attention ----------
    attended, _ = AttentionLayer(
        return_attention=True, name='attention'
    )(lstm_out)

    # ---------- Classification head ----------
    # Dense(256) -> Dropout(0.3) -> output
    outputs, loss = create_classification_head(
        attended, num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=[256]
    )

    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    # Adam optimizer, lr=0.0001 (per CNN-LSTM paper)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(f"EfficientNet-LSTM model created: lstm_units={lstm_units}, params={model.count_params()}")
    return model


def _create_efficientnet_lstm_lite(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    architecture: str = 'efficientnet_lstm_lite'
) -> models.Model:
    """Lightweight EfficientNet-LSTM variant.

    Smaller EfficientNet input (112x112), single LSTM layer, fewer units.
    """
    inputs = layers.Input(shape=input_shape, name='audio_input')

    if is_raw_audio(input_shape):
        audio = ensure_flat_input(inputs, input_shape)
        x = MelSpectrogramFrontEnd(
            sample_rate=16000, n_fft=512, hop_length=160, n_mels=40,
            name='mel_spectrogram'
        )(audio)
        x = ExpandDimsLayer(axis=-1, name='expand_dims')(x)
        x = ResizeLayer(target_height=112, target_width=112, name='resize')(x)
        x = RepeatChannelLayer(repeats=3, axis=-1, name='repeat_channels')(x)
    else:
        x = inputs
        if len(input_shape) == 2:
            x = ExpandDimsLayer(axis=-1, name='expand_dims')(x)
            x = ResizeLayer(target_height=112, target_width=112, name='resize')(x)
            x = RepeatChannelLayer(repeats=3, axis=-1, name='repeat_channels')(x)
        elif len(input_shape) == 3:
            x = ResizeLayer(target_height=112, target_width=112, name='resize')(x)
            if input_shape[-1] != 3:
                x = layers.Conv2D(3, (1, 1), name='channel_proj')(x)

    efficientnet = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(112, 112, 3)
    )
    feature_maps = efficientnet(x)

    temporal_features = TemporalPoolingLayer(name='temporal_pool')(feature_maps)

    # Single LSTM layer, 64 units
    lstm_out = layers.LSTM(64, return_sequences=False, dropout=0.2,
                           name='lstm')(temporal_features)

    outputs, loss = create_classification_head(
        lstm_out, num_classes,
        dropout_rate=0.2,
        hidden_dims=[128]
    )

    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(f"EfficientNet-LSTM Lite created: params={model.count_params()}")
    return model


# ============================ FACTORY ============================

def create_model(input_shape: Tuple[int, ...], num_classes: int = 1,
                 architecture: str = 'efficientnet_lstm', **kwargs) -> models.Model:
    """Factory function for EfficientNet-LSTM variants.

    Variants:
        'efficientnet_lstm': Full model (EfficientNet-B0 + BiLSTM[256,128] + Attention + Delta features)
        'efficientnet_lstm_lite': Lightweight (smaller input + single LSTM(64))
    """
    if architecture == 'efficientnet_lstm_lite':
        return _create_efficientnet_lstm_lite(input_shape, num_classes, architecture)
    else:
        return _create_efficientnet_lstm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            lstm_units=kwargs.get('lstm_units', 256),
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            architecture=architecture
        )


# Register custom layers
tf.keras.utils.get_custom_objects().update({
    'MelSpectrogramFrontEnd': MelSpectrogramFrontEnd,
    'TemporalPoolingLayer': TemporalPoolingLayer,
    'AttentionLayer': AttentionLayer,
    'ExpandDimsLayer': ExpandDimsLayer,
    'ResizeLayer': ResizeLayer,
    'RepeatChannelLayer': RepeatChannelLayer,
    'DeltaFeatureLayer': DeltaFeatureLayer,
})
