"""EfficientNet-LSTM Architecture Implementation"""

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from typing import Tuple, Optional, Any, Dict
import logging
from app.domain.models.architectures.layers import (
    STFTLayer, MagnitudeLayer, ExpandDimsLayer, ResizeLayer, RepeatChannelLayer,
    is_raw_audio, ensure_flat_input, SqueezeExciteBlock, AttentionLayer,
    SafeEfficientNetInputLayer, create_classification_head
)

logger = logging.getLogger(__name__)

# ============================ CUSTOM LAYERS ============================
# Layers moved to app/domain/models/architectures/layers.py


# ============================ FACTORY FUNCTIONS ============================


def create_efficientnet_lstm_model(
    input_shape: Tuple[int, ...],
    num_classes: int,
    lstm_units: int = 256,
    attention_units: int = 128,
    dropout_rate: float = 0.3,
    architecture: str = 'efficientnet_lstm'
) -> models.Model:
    """
    Create EfficientNet-LSTM model.
    """
    logger.info(
        f"Creating EfficientNet-LSTM model with input_shape={input_shape}, num_classes={num_classes}")

    # Input layer
    inputs = layers.Input(shape=input_shape, name='efficientnet_lstm_input')

    # Handle 1D input (raw audio) by converting to spectrogram
    if is_raw_audio(input_shape):
        # If (batch, samples, 1), reshape to (batch, samples) for STFT
        input_tensor = ensure_flat_input(inputs, input_shape)

        # Convert 1D audio to spectrogram using STFT
        x = STFTLayer(
            frame_length=512,
            frame_step=256,
            fft_length=512,
            name='stft')(input_tensor)
        x = MagnitudeLayer(name='magnitude')(x)
        # Expand dimensions for image-like processing
        x = ExpandDimsLayer(axis=-1, name='expand_dims')(x)
        # Resize to EfficientNet input size
        x = ResizeLayer(target_height=224, target_width=224, name='resize')(x)
        # Convert to 3 channels
        x = RepeatChannelLayer(repeats=3, axis=-1, name='repeat_channels')(x)
    else:
        # Preprocessing to convert audio features to spectrogram-like format
        x = SafeEfficientNetInputLayer(name='safe_efficientnet_input')(inputs)

    # EfficientNet-B0 backbone
    efficientnet = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Fine-tune the last few layers
    for layer in efficientnet.layers[:-20]:
        layer.trainable = False

    # Extract features using EfficientNet
    efficientnet_features = efficientnet(x)

    # Add Squeeze-and-Excitation block
    se_features = SqueezeExciteBlock(
        efficientnet_features.shape[-1])(efficientnet_features)

    # Global Average Pooling to get feature vector
    pooled_features = layers.GlobalAveragePooling2D(
        name='efficientnet_gap')(se_features)

    # Reshape for LSTM (create time dimension)
    feature_dim = pooled_features.shape[-1]
    time_steps = 16  # Create 16 time steps
    segment_size = feature_dim // time_steps

    # Reshape to (batch, time_steps, segment_size)
    lstm_input = layers.Reshape(
        (time_steps, segment_size), name='lstm_reshape')(pooled_features)

    # Bidirectional LSTM layers
    lstm1 = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate),
        name='bilstm_1'
    )(lstm_input)

    lstm2 = layers.Bidirectional(
        layers.LSTM(
            lstm_units // 2,
            return_sequences=True,
            dropout=dropout_rate),
        name='bilstm_2'
    )(lstm1)

    # Attention mechanism
    attended_features, attention_weights = AttentionLayer(
        return_attention=True,
        name='attention_layer'
    )(lstm2)

    # Additional dense layers for classification
    outputs, loss = create_classification_head(
        attended_features,
        num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=[512, 256, 128]
    )

    # Create model
    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name='efficientnet_lstm_model')

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=1e-5,
            beta_1=0.9,
            beta_2=0.999
        ),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(
        f"EfficientNet-LSTM model created successfully with {model.count_params()} parameters")
    return model


def create_lightweight_efficientnet_lstm(
    input_shape: Tuple[int, ...],
    num_classes: int,
    architecture: str = 'efficientnet_lstm_lite'
) -> models.Model:
    """
    Create a lightweight version of EfficientNet-LSTM.
    """
    logger.info(
        f"Creating Lightweight EfficientNet-LSTM model with input_shape={input_shape}")

    # Input layer
    inputs = layers.Input(
        shape=input_shape,
        name='lite_efficientnet_lstm_input')

    # Preprocessing
    if len(input_shape) == 2:
        x = ExpandDimsLayer(axis=-1)(inputs)
        x = ResizeLayer(target_height=112, target_width=112)(x)
        x = RepeatChannelLayer(repeats=3, axis=-1)(x)
    else:
        x = ResizeLayer(target_height=112, target_width=112)(inputs)
        if input_shape[-1] != 3:
            # Simple 1x1 conv to get 3 channels
            x = layers.Conv2D(3, 1, activation='relu')(x)

    # Use EfficientNet without pre-trained weights
    efficientnet = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(112, 112, 3)
    )

    # Extract features
    efficientnet_features = efficientnet(x)
    pooled_features = layers.GlobalAveragePooling2D()(efficientnet_features)

    # Simpler LSTM structure
    feature_dim = pooled_features.shape[-1]
    time_steps = 8
    segment_size = feature_dim // time_steps

    lstm_input = layers.Reshape((time_steps, segment_size))(pooled_features)

    # Single LSTM layer
    lstm_output = layers.LSTM(
        128,
        return_sequences=False,
        dropout=0.2)(lstm_input)

    # Simple classification head
    outputs, loss = create_classification_head(
        lstm_output,
        num_classes,
        dropout_rate=0.3,
        hidden_dims=[256, 128]
    )

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name='lite_efficientnet_lstm_model')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(
        f"Lightweight EfficientNet-LSTM model created with {model.count_params()} parameters")
    return model


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'efficientnet_lstm', **kwargs) -> models.Model:
    """Factory function."""
    if architecture == 'efficientnet_lstm':
        return create_efficientnet_lstm_model(
            input_shape,
            num_classes,
            lstm_units=kwargs.get('lstm_units', 256),
            attention_units=kwargs.get('attention_units', 128),
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            architecture=architecture
        )
    elif architecture == 'efficientnet_lstm_lite':
        return create_lightweight_efficientnet_lstm(
            input_shape, num_classes, architecture=architecture)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


# Register custom layers
tf.keras.utils.get_custom_objects().update({
    'AttentionLayer': AttentionLayer,
    'SqueezeExciteBlock': SqueezeExciteBlock,
    'STFTLayer': STFTLayer,
    'MagnitudeLayer': MagnitudeLayer,
    'ExpandDimsLayer': ExpandDimsLayer,
    'ResizeLayer': ResizeLayer,
    'RepeatChannelLayer': RepeatChannelLayer,
    'SafeEfficientNetInputLayer': SafeEfficientNetInputLayer
})
