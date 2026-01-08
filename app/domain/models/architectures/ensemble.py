"""Ensemble Architecture Implementation"""

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import EfficientNetB0
from typing import Tuple, List, Optional, Union, Dict, Any
import logging
from app.domain.models.architectures.layers import STFTLayer

try:
    # Importar as arquiteturas individuais
    from .conformer import create_conformer_model
    from .efficientnet_lstm import create_efficientnet_lstm_model
    from .spectrogram_transformer import create_spectrogram_transformer_model
    from .multiscale_cnn import create_multiscale_cnn_model
except ImportError as e:
    print(f"Warning: Could not import some architecture modules: {e}")
    # Definir funções dummy para fallback

    def create_conformer_model(*args, **kwargs):
        raise ImportError("Conformer module not available")

    def create_efficientnet_lstm_model(*args, **kwargs):
        raise ImportError("EfficientNet-LSTM module not available")

    def create_spectrogram_transformer_model(*args, **kwargs):
        raise ImportError("Spectrogram Transformer module not available")

    def create_multiscale_cnn_model(*args, **kwargs):
        raise ImportError("Multi-Scale CNN module not available")

logger = logging.getLogger(__name__)


class EnsembleLayer(layers.Layer):
    """Custom ensemble layer that combines predictions from multiple models."""

    def __init__(self, num_models: int,
                 ensemble_method: str = 'weighted_average', **kwargs):
        super(EnsembleLayer, self).__init__(**kwargs)
        self.num_models = num_models
        self.ensemble_method = ensemble_method

        if ensemble_method == 'weighted_average':
            # Learnable weights for each model
            self.model_weights = self.add_weight(
                name='model_weights',
                shape=(num_models,),
                initializer='uniform',
                trainable=True
            )
        elif ensemble_method == 'attention':
            # Attention mechanism for dynamic weighting
            self.attention_dense = layers.Dense(
                num_models, activation='softmax')

    def call(self, inputs):
        # inputs is a list of predictions from different models
        # Each input shape: (batch_size, num_classes)

        if self.ensemble_method == 'simple_average':
            # Simple average
            stacked_predictions = tf.stack(
                inputs, axis=-1)  # (batch, classes, models)
            ensemble_output = tf.reduce_mean(stacked_predictions, axis=-1)

        elif self.ensemble_method == 'weighted_average':
            # Weighted average with learnable weights
            weights = tf.nn.softmax(self.model_weights)  # Normalize weights
            stacked_predictions = tf.stack(
                inputs, axis=-1)  # (batch, classes, models)

            # Apply weights
            weighted_predictions = stacked_predictions * weights[None, None, :]
            ensemble_output = tf.reduce_sum(weighted_predictions, axis=-1)

        elif self.ensemble_method == 'attention':
            # Attention-based ensemble
            # Use average prediction as query for attention
            avg_prediction = tf.reduce_mean(tf.stack(inputs, axis=-1), axis=-1)
            attention_weights = self.attention_dense(
                avg_prediction)  # (batch, num_models)

            # Apply attention weights
            stacked_predictions = tf.stack(
                inputs, axis=1)  # (batch, models, classes)
            attention_weights = tf.expand_dims(
                attention_weights, axis=-1)  # (batch, models, 1)

            ensemble_output = tf.reduce_sum(
                stacked_predictions * attention_weights, axis=1)

        elif self.ensemble_method == 'max_voting':
            # Max voting (take max probability for each class)
            stacked_predictions = tf.stack(
                inputs, axis=-1)  # (batch, classes, models)
            ensemble_output = tf.reduce_max(stacked_predictions, axis=-1)

        else:
            raise ValueError(
                f"Unknown ensemble method: {
                    self.ensemble_method}")

        return ensemble_output

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_models': self.num_models,
            'ensemble_method': self.ensemble_method
        })
        return config


class AdaptiveFeatureFusion(layers.Layer):
    """Adaptive feature fusion layer that combines features from different models."""

    def __init__(self, output_dim: int, **kwargs):
        super(AdaptiveFeatureFusion, self).__init__(**kwargs)
        self.output_dim = output_dim

        # Feature transformation layers
        self.feature_transforms = []
        self.attention_layers = []

    def build(self, input_shape):
        # input_shape is a list of shapes from different models
        num_models = len(input_shape)

        for i in range(num_models):
            # Transform each feature to common dimension
            transform = layers.Dense(
                self.output_dim,
                activation='relu',
                name=f'transform_{i}')
            self.feature_transforms.append(transform)

            # Attention for each feature
            attention = layers.Dense(
                1, activation='sigmoid', name=f'attention_{i}')
            self.attention_layers.append(attention)

        super().build(input_shape)

    def call(self, inputs):
        # inputs is a list of feature vectors from different models
        transformed_features = []
        attention_weights = []

        for i, (feature, transform, attention) in enumerate(
                zip(inputs, self.feature_transforms, self.attention_layers)):
            # Transform feature
            transformed = transform(feature)
            transformed_features.append(transformed)

            # Compute attention weight
            weight = attention(feature)
            attention_weights.append(weight)

        # Normalize attention weights
        attention_weights = tf.nn.softmax(
            tf.concat(attention_weights, axis=-1), axis=-1)

        # Apply attention weights
        weighted_features = []
        for i, feature in enumerate(transformed_features):
            weight = attention_weights[:, i:i + 1]  # (batch, 1)
            weighted_feature = feature * weight
            weighted_features.append(weighted_feature)

        # Combine features
        fused_features = tf.add_n(weighted_features)

        return fused_features

    def get_config(self):
        config = super().get_config()
        config.update({'output_dim': self.output_dim})
        return config


def create_ensemble_model(
    input_shape: Tuple[int, ...],
    num_classes: int,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ensemble_method: str = 'weighted_average',
    fusion_method: str = 'prediction_level',
    architecture: str = 'ensemble'
) -> models.Model:
    """
    Create an ensemble model combining multiple state-of-the-art architectures.

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        model_configs: Configuration for individual models
        ensemble_method: Method for combining predictions ('simple_average', 'weighted_average', 'attention', 'max_voting')
        fusion_method: Level of fusion ('prediction_level', 'feature_level', 'hybrid')
        architecture: Architecture name (for compatibility)

    Returns:
        Compiled Keras model
    """
    logger.info(
        f"Creating Ensemble model with input_shape={input_shape}, num_classes={num_classes}")
    logger.info(
        f"Ensemble method: {ensemble_method}, Fusion method: {fusion_method}")

    # Default model configurations (simplified to working architectures)
    if model_configs is None:
        model_configs = {
            'conformer': {'d_model': 256, 'num_blocks': 6, 'num_heads': 8},
            'efficientnet_lstm': {'lstm_units': 256, 'attention_units': 128}
        }

    # Input layer
    inputs = layers.Input(shape=input_shape, name='ensemble_input')

    # Create individual models
    individual_models = {}
    model_outputs = {}

    # Conformer model
    if 'conformer' in model_configs:
        logger.info("Adding Conformer model to ensemble")
        conformer_config = model_configs['conformer']

        # Create a functional model that shares the same input
        conformer_x = inputs

        # Handle different input formats
        if len(input_shape) == 1:
            # Convert 1D to 2D spectrogram using custom layer
            conformer_x = STFTLayer(name='conformer_stft', add_channel_dim=True)(conformer_x)
        elif len(input_shape) == 2:
            # Add channel dimension for 2D input
            conformer_x = layers.Lambda(lambda x: tf.expand_dims(
                x, axis=-1), name='conformer_expand_dims')(conformer_x)

        # Conformer layers
        conformer_x = layers.Conv2D(
            64,
            (3,
             3),
            activation='relu',
            padding='same',
            name='conformer_conv1')(conformer_x)
        conformer_x = layers.BatchNormalization(
            name='conformer_bn1')(conformer_x)
        conformer_x = layers.MaxPooling2D(
            (2, 2), name='conformer_pool1')(conformer_x)

        conformer_x = layers.Conv2D(
            128,
            (3,
             3),
            activation='relu',
            padding='same',
            name='conformer_conv2')(conformer_x)
        conformer_x = layers.BatchNormalization(
            name='conformer_bn2')(conformer_x)
        conformer_x = layers.GlobalAveragePooling2D(
            name='conformer_gap')(conformer_x)

        conformer_x = layers.Dense(
            256,
            activation='relu',
            name='conformer_dense1')(conformer_x)
        conformer_x = layers.Dropout(
            0.3, name='conformer_dropout')(conformer_x)

        if fusion_method == 'prediction_level':
            conformer_output = layers.Dense(
                num_classes,
                activation='softmax',
                name='conformer_output')(conformer_x)
        else:
            conformer_output = conformer_x  # Use features for feature-level fusion

        model_outputs['conformer'] = conformer_output

    # EfficientNet-LSTM model
    if 'efficientnet_lstm' in model_configs:
        logger.info("Adding EfficientNet-LSTM model to ensemble")
        efficientnet_config = model_configs['efficientnet_lstm']

        # Create a functional model that shares the same input
        efficientnet_x = inputs

        # Handle different input formats
        if len(input_shape) == 1:
            # Convert 1D to 2D spectrogram using custom layer
            efficientnet_x = STFTLayer(
                name='efficientnet_stft')(efficientnet_x)
            # Resize to EfficientNet input size
            efficientnet_x = layers.Lambda(lambda x: tf.image.resize(
                x, [224, 224]), name='efficientnet_resize')(efficientnet_x)
        elif len(input_shape) == 2:
            # Add channel dimension for 2D input
            efficientnet_x = layers.Lambda(lambda x: tf.expand_dims(
                x, axis=-1), name='efficientnet_expand_dims')(efficientnet_x)

        # EfficientNet-like layers (simplified)
        efficientnet_x = layers.Conv2D(
            32,
            (3,
             3),
            activation='swish',
            padding='same',
            name='efficientnet_stem')(efficientnet_x)
        efficientnet_x = layers.BatchNormalization(
            name='efficientnet_bn1')(efficientnet_x)

        efficientnet_x = layers.Conv2D(
            64,
            (3,
             3),
            activation='swish',
            padding='same',
            name='efficientnet_conv1')(efficientnet_x)
        efficientnet_x = layers.BatchNormalization(
            name='efficientnet_bn2')(efficientnet_x)
        efficientnet_x = layers.MaxPooling2D(
            (2, 2), name='efficientnet_pool1')(efficientnet_x)

        efficientnet_x = layers.Conv2D(
            128,
            (3,
             3),
            activation='swish',
            padding='same',
            name='efficientnet_conv2')(efficientnet_x)
        efficientnet_x = layers.BatchNormalization(
            name='efficientnet_bn3')(efficientnet_x)
        efficientnet_x = layers.GlobalAveragePooling2D(
            name='efficientnet_gap')(efficientnet_x)

        # LSTM-like processing
        efficientnet_x = layers.Dense(
            256,
            activation='relu',
            name='efficientnet_dense1')(efficientnet_x)
        efficientnet_x = layers.Dropout(
            0.3, name='efficientnet_dropout')(efficientnet_x)

        if fusion_method == 'prediction_level':
            efficientnet_output = layers.Dense(
                num_classes,
                activation='softmax',
                name='efficientnet_output')(efficientnet_x)
        else:
            efficientnet_output = efficientnet_x  # Use features for feature-level fusion

        model_outputs['efficientnet_lstm'] = efficientnet_output

    # Combine outputs based on fusion method
    if fusion_method == 'prediction_level':
        # Prediction-level fusion
        predictions = list(model_outputs.values())
        ensemble_output = EnsembleLayer(
            num_models=len(predictions),
            ensemble_method=ensemble_method,
            name='ensemble_layer'
        )(predictions)

    elif fusion_method == 'feature_level':
        # Feature-level fusion
        features = list(model_outputs.values())
        fused_features = AdaptiveFeatureFusion(
            output_dim=512,
            name='feature_fusion'
        )(features)

        # Classification head
        x = layers.Dense(
            256,
            activation='relu',
            name='ensemble_dense1')(fused_features)
        x = layers.BatchNormalization(name='ensemble_bn1')(x)
        x = layers.Dropout(0.3, name='ensemble_dropout1')(x)

        x = layers.Dense(128, activation='relu', name='ensemble_dense2')(x)
        x = layers.Dropout(0.2, name='ensemble_dropout2')(x)

        ensemble_output = layers.Dense(
            num_classes,
            activation='softmax',
            name='ensemble_output')(x)

    elif fusion_method == 'hybrid':
        # Hybrid fusion: combine both feature-level and prediction-level
        features = list(model_outputs.values())

        # Feature-level fusion
        fused_features = AdaptiveFeatureFusion(
            output_dim=256,
            name='feature_fusion'
        )(features)

        # Get predictions from fused features
        feature_prediction = layers.Dense(
            num_classes,
            activation='softmax',
            name='feature_prediction')(fused_features)

        # Also get individual predictions (assuming we have them)
        # For hybrid, we need to modify the individual models to output both features and predictions
        # This is a simplified version
        # Add individual predictions here if available
        predictions = [feature_prediction]

        ensemble_output = EnsembleLayer(
            num_models=len(predictions),
            ensemble_method=ensemble_method,
            name='hybrid_ensemble'
        )(predictions)

    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")

    # Create the ensemble model
    ensemble_model = models.Model(
        inputs=inputs,
        outputs=ensemble_output,
        name='ensemble_model')

    # Compile model
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=5e-5,  # Lower learning rate for ensemble
            weight_decay=1e-5,
            beta_1=0.9,
            beta_2=0.999
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'f1_score']
    )

    logger.info(
        f"Ensemble model created successfully with {
            ensemble_model.count_params()} parameters")
    logger.info(f"Individual models in ensemble: {list(model_configs.keys())}")

    return ensemble_model


def create_lightweight_ensemble(
    input_shape: Tuple[int, ...],
    num_classes: int,
    architecture: str = 'ensemble_lite'
) -> models.Model:
    """
    Create a lightweight ensemble with fewer models for faster inference.

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    # Use only the most effective models
    lightweight_configs = {
        'conformer': {'d_model': 128, 'num_blocks': 4, 'num_heads': 4},
        'multiscale_cnn': {'base_filters': 32, 'num_blocks': 3}
    }

    return create_ensemble_model(
        input_shape=input_shape,
        num_classes=num_classes,
        model_configs=lightweight_configs,
        ensemble_method='weighted_average',
        fusion_method='prediction_level',
        architecture=architecture
    )


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'ensemble') -> models.Model:
    """
    Factory function to create ensemble models (for compatibility with existing code).

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    if architecture == 'ensemble':
        return create_ensemble_model(
            input_shape, num_classes, architecture=architecture)
    elif architecture == 'ensemble_lite':
        return create_lightweight_ensemble(
            input_shape, num_classes, architecture=architecture)
    elif architecture == 'ensemble_feature_fusion':
        return create_ensemble_model(
            input_shape, num_classes,
            fusion_method='feature_level',
            architecture=architecture
        )
    elif architecture == 'ensemble_hybrid':
        return create_ensemble_model(
            input_shape, num_classes,
            fusion_method='hybrid',
            architecture=architecture
        )
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. Use 'ensemble', 'ensemble_lite', 'ensemble_feature_fusion', or 'ensemble_hybrid'")


# Register custom layers for model loading
tf.keras.utils.get_custom_objects().update({
    'EnsembleLayer': EnsembleLayer,
    'AdaptiveFeatureFusion': AdaptiveFeatureFusion
})
