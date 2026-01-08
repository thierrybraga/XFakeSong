"""Implementação da arquitetura WavLM para detecção de deepfakes.

Esta implementação segue uma arquitetura de dois estágios:
1. Extrator de características: Modelo WavLM pré-treinado (congelado)
2. Classificador: MLP para classificação binária (real vs deepfake)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Dict, Any, Optional
from app.domain.models.architectures.layers import create_classification_head
import numpy as np
import logging

from app.core.utils.audio_utils import normalize_audio

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tentar importar Transformers
try:
    from transformers import TFWav2Vec2Model, Wav2Vec2Processor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TFWav2Vec2Model = None
    Wav2Vec2Processor = None
    logger.info("Transformers library not found. Using simplified WavLM implementation.")


class WavLMFeatureExtractor(layers.Layer):
    """Extrator de características usando modelo WavLM pré-treinado."""

    def __init__(self, model_name: str = "microsoft/wavlm-base",
                 freeze_weights: bool = True, **kwargs):
        super(WavLMFeatureExtractor, self).__init__(**kwargs)
        self.model_name = model_name
        self.freeze_weights = freeze_weights
        self.feature_dim = 768  # Dimensão padrão do WavLM base

        if HF_AVAILABLE:
            try:
                # Carregar modelo WavLM pré-treinado
                self.wavlm_model = TFWav2Vec2Model.from_pretrained(
                    model_name,
                    from_tf=True
                )
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)

                # Congelar pesos se especificado
                if freeze_weights:
                    self.wavlm_model.trainable = False

                logger.info(f"WavLM model {model_name} loaded successfully")

            except Exception as e:
                logger.warning(
                    f"Failed to load WavLM model: {e}. Using simplified implementation.")
                self._use_simplified = True
        else:
            self._use_simplified = True

        if hasattr(self, '_use_simplified'):
            # Implementação simplificada usando CNN 1D
            self._build_simplified_extractor()

    def _build_simplified_extractor(self):
        """Constrói um extrator simplificado usando CNN 1D."""
        self.conv_layers = [
            layers.Conv1D(64, 10, strides=5, activation='relu', name='conv1'),
            layers.BatchNormalization(name='bn1'),
            layers.Conv1D(128, 8, strides=4, activation='relu', name='conv2'),
            layers.BatchNormalization(name='bn2'),
            layers.Conv1D(256, 4, strides=2, activation='relu', name='conv3'),
            layers.BatchNormalization(name='bn3'),
            layers.Conv1D(512, 4, strides=2, activation='relu', name='conv4'),
            layers.BatchNormalization(name='bn4'),
            layers.Conv1D(
                self.feature_dim,
                4,
                strides=2,
                activation='relu',
                name='conv5'),
            layers.BatchNormalization(name='bn5')
        ]

    def call(self, inputs, training=None):
        """Forward pass do extrator de características."""
        if hasattr(self, '_use_simplified'):
            # Usar implementação simplificada
            x = inputs

            # Processar entrada baseado na dimensionalidade
            if len(x.shape) == 3:  # (batch, time, freq) - espectrograma
                # Achatar para (batch, time*freq) e depois expandir para Conv1D
                batch_size = tf.shape(x)[0]
                x = tf.reshape(x, [batch_size, -1])  # (batch, time*freq)
                x = tf.expand_dims(x, axis=-1)  # (batch, time*freq, 1)
            elif len(x.shape) == 2:  # (batch, features)
                x = tf.expand_dims(x, axis=-1)  # (batch, features, 1)

            # Aplicar camadas convolucionais
            for layer in self.conv_layers:
                x = layer(x, training=training)

            return x
        else:
            # Usar modelo WavLM pré-treinado
            # Normalizar entrada para o range esperado pelo WavLM
            normalized_inputs = tf.nn.l2_normalize(inputs, axis=-1)

            # Extrair características usando WavLM
            outputs = self.wavlm_model(normalized_inputs, training=False)

            # Retornar hidden states
            return outputs.last_hidden_state

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'freeze_weights': self.freeze_weights,
            'feature_dim': self.feature_dim
        })
        return config





def preprocess(audio_data: np.ndarray, target_sr: int = 16000) -> np.ndarray:
    """Pré-processamento de áudio para WavLM.

    Args:
        audio_data: Array de áudio
        target_sr: Taxa de amostragem alvo (16kHz para WavLM)

    Returns:
        Áudio pré-processado
    """
    # Normalização usando utilitário
    audio_data = normalize_audio(audio_data)

    # Clipping para evitar valores extremos
    audio_data = np.clip(audio_data, -3.0, 3.0)

    return audio_data


def create_wavlm_model(input_shape: Tuple[int, ...],
                       num_classes: int = 1,
                       architecture: str = 'wavlm',
                       wavlm_model: str = "microsoft/wavlm-base",
                       freeze_wavlm: bool = True,
                       classifier_units: list = [512, 256],
                       dropout_rate: float = 0.3) -> models.Model:
    """Cria modelo WavLM completo.

    Args:
        input_shape: Formato da entrada (comprimento_audio,)
        num_classes: Número de classes (1 para classificação binária)
        architecture: Nome da arquitetura
        wavlm_model: Nome do modelo WavLM pré-treinado
        freeze_wavlm: Se deve congelar pesos do WavLM
        classifier_units: Unidades das camadas do classificador
        dropout_rate: Taxa de dropout

    Returns:
        Modelo Keras compilado
    """
    logger.info(
        f"Creating WavLM model with input_shape={input_shape}, num_classes={num_classes}")

    # Entrada
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # Estágio 1: Extrator de características WavLM
    feature_extractor = WavLMFeatureExtractor(
        model_name=wavlm_model,
        freeze_weights=freeze_wavlm,
        name='wavlm_feature_extractor'
    )
    features = feature_extractor(inputs)

    # Classification head
    outputs, loss = create_classification_head(
        features,
        num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=classifier_units
    )

    # Create model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f'wavlm_{architecture}')

    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy', 'precision', 'recall']
    )

    # Log do número de parâmetros
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w)
                           for w in model.trainable_weights])
    logger.info(
        f"WavLM model created successfully with {total_params} total parameters ({trainable_params} trainable)")

    return model


def create_wavlm_lite(input_shape: Tuple[int, ...],
                      num_classes: int = 1,
                      architecture: str = 'wavlm_lite') -> models.Model:
    """Cria versão lite do modelo WavLM.

    Args:
        input_shape: Formato da entrada
        num_classes: Número de classes
        architecture: Nome da arquitetura

    Returns:
        Modelo Keras compilado
    """
    return create_wavlm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        architecture=architecture,
        classifier_units=[256, 128],
        dropout_rate=0.2
    )


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'wavlm', **kwargs) -> models.Model:
    """Função principal para criar modelos WavLM.

    Args:
        input_shape: Formato da entrada
        num_classes: Número de classes
        architecture: Tipo de arquitetura ('wavlm' ou 'wavlm_lite')
        **kwargs: Parâmetros adicionais (ignorados para compatibilidade)

    Returns:
        Modelo Keras compilado
    """
    if architecture == 'wavlm_lite':
        return create_wavlm_lite(input_shape, num_classes, architecture)
    else:
        return create_wavlm_model(input_shape, num_classes, architecture)


# Registrar objetos personalizados no Keras
tf.keras.utils.get_custom_objects().update({
    'WavLMFeatureExtractor': WavLMFeatureExtractor,
    'preprocess': preprocess
})
