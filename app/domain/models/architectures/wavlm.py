"""Implementação da arquitetura WavLM para detecção de deepfakes.

Esta implementação segue uma arquitetura de dois estágios:
1. Extrator de características: Modelo WavLM pré-treinado (congelado)
2. Classificador: MLP para classificação binária (real vs deepfake)
"""

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from app.core.utils.audio_utils import normalize_audio
from app.domain.models.architectures.layers import (
    AttentionPoolingLayer,
    create_classification_head,
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tentar importar Transformers
try:
    from transformers import TFWavLMModel, Wav2Vec2Processor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TFWavLMModel = None
    Wav2Vec2Processor = None
    logger.info("Transformers library not found. Using simplified WavLM implementation.")


class WavLMFeatureExtractor(layers.Layer):
    """Extrator de características usando modelo WavLM pré-treinado."""

    def __init__(self, model_name: str = "microsoft/wavlm-base",
                 freeze_weights: bool = True, n_trainable_layers: int = 0,
                 **kwargs):
        super(WavLMFeatureExtractor, self).__init__(**kwargs)
        self.model_name = model_name
        self.freeze_weights = freeze_weights
        # Fine-tuning parcial: nº de camadas do encoder a descongelar (do topo).
        # >0 ativa o fine-tune recomendado (Tak et al. 2022); 0 = congelado.
        self.n_trainable_layers = int(n_trainable_layers)
        self.feature_dim = 768 if "base" in model_name else 1024

        if HF_AVAILABLE:
            try:
                # Carregar modelo WavLM pré-treinado
                self.wavlm_model = TFWavLMModel.from_pretrained(
                    model_name,
                    from_tf=True
                )

                # Configurar para retornar todos os hidden states para weighted sum
                self.num_layers = self.wavlm_model.config.num_hidden_layers + 1 # +1 para embedding inicial
                from app.domain.models.architectures.layers import WeightedSumLayer
                self.weighted_sum = WeightedSumLayer(num_layers=self.num_layers)

                # Trainability do backbone (congelado | fine-tune parcial | total)
                from app.domain.models.architectures.ssl_utils import (
                    set_ssl_backbone_trainability,
                )
                _mode = set_ssl_backbone_trainability(
                    self.wavlm_model, freeze_weights, self.n_trainable_layers
                )

                logger.info(
                    f"WavLM model {model_name} carregado ({self.num_layers} "
                    f"camadas) — backbone: {_mode}"
                )

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
            if len(x.shape) == 3:  # (batch, time, freq)
                batch_size = tf.shape(x)[0]
                x = tf.reshape(x, [batch_size, -1])
                x = tf.expand_dims(x, axis=-1)
            elif len(x.shape) == 2:  # (batch, features)
                x = tf.expand_dims(x, axis=-1)

            # Aplicar camadas convolucionais
            for layer in self.conv_layers:
                x = layer(x, training=training)

            return x
        else:
            # WavLM espera (batch, sequence_length)
            # Garantir formato correto
            if len(inputs.shape) == 3:
                inputs = tf.squeeze(inputs, axis=-1)

            # Extrair características usando WavLM com output_hidden_states=True
            outputs = self.wavlm_model(inputs, output_hidden_states=True, training=False)

            # Weighted sum de todos os hidden states (Fidelidade ao paper para downstream tasks)
            hidden_states = outputs.hidden_states
            x = self.weighted_sum(list(hidden_states))

            return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'freeze_weights': self.freeze_weights,
            'n_trainable_layers': self.n_trainable_layers,
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


def _create_wavlm_model(input_shape: Tuple[int, ...],
                        num_classes: int = 1,
                        architecture: str = 'wavlm',
                        wavlm_model: str = "microsoft/wavlm-base",
                        freeze_wavlm: bool = True,
                        n_trainable_layers: int = 3,
                        backend: str = "conv",
                        classifier_units: list = None,
                        dropout_rate: float = 0.3) -> models.Model:
    """Cria modelo WavLM completo fiel ao paper.

    Args:
        input_shape: Formato da entrada (samples,)
        num_classes: Número de classes (1 para classificação binária)
        architecture: Nome da arquitetura
        wavlm_model: Nome do modelo WavLM pré-treinado
        freeze_wavlm: Se deve congelar pesos do WavLM
        classifier_units: Unidades das camadas do classificador
        dropout_rate: Taxa de dropout

    Returns:
        Modelo Keras compilado
    """
    if classifier_units is None:
        classifier_units = [1024, 512, 256]

    logger.info(
        f"Creating WavLM model with input_shape={input_shape}, num_classes={num_classes}")

    # 1. Entrada (Raw Audio)
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # 2. Extrator de características WavLM (Self-Supervised)
    feature_extractor = WavLMFeatureExtractor(
        model_name=wavlm_model,
        freeze_weights=freeze_wavlm,
        n_trainable_layers=n_trainable_layers,
        name='wavlm_feature_extractor'
    )
    # x shape: (batch, sequence_length, feature_dim)
    x = feature_extractor(inputs)

    if backend == "aasist":
        # Back-end de grafo AASIST (receita SOTA: WavLM → grafo espectro-temporal)
        from app.domain.models.architectures.ssl_utils import (
            build_ssl_aasist_backend,
        )
        pooled = build_ssl_aasist_backend(
            x, dropout_rate=dropout_rate, name="wavlm_aasist"
        )
    else:
        # Back-end raso (conv 1D + attention pooling)
        conv_out = layers.Conv1D(256, 3, padding='same', activation='relu', name='temporal_conv1')(x)
        conv_out = layers.BatchNormalization(name='temporal_bn1')(conv_out)
        conv_out2 = layers.Conv1D(256, 3, padding='same', activation='relu', name='temporal_conv2')(conv_out)
        conv_out2 = layers.BatchNormalization(name='temporal_bn2')(conv_out2)
        conv_out2 = conv_out2 + conv_out  # residual
        conv_out3 = layers.Conv1D(256, 3, padding='same', activation='relu', name='temporal_conv3')(conv_out2)
        conv_out3 = layers.BatchNormalization(name='temporal_bn3')(conv_out3)
        conv_out3 = conv_out3 + conv_out2  # residual
        pooled = AttentionPoolingLayer(name='attention_pool')(conv_out3)

    # Classification Head
    outputs, loss = create_classification_head(
        pooled,
        num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=classifier_units
    )

    # Criar modelo
    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f'wavlm_{architecture}')

    # Compilar modelo. Em fine-tuning parcial (backbone SSL descongelado) usa-se
    # LR baixo (1e-5) para evitar esquecimento catastrófico dos pesos pré-treinados;
    # com backbone congelado, LR padrão (1e-4) para o classificador.
    lr = 1e-5 if n_trainable_layers and n_trainable_layers > 0 else 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(
        f"WavLM model {architecture} criado (lr={lr}, "
        f"n_trainable_layers={n_trainable_layers})"
    )
    return model


def create_model(input_shape: Tuple[int, ...], num_classes: int = 1,
                 architecture: str = 'wavlm', **kwargs) -> models.Model:
    """Função principal para criar modelos WavLM.

    Args:
        input_shape: Formato da entrada
        num_classes: Número de classes
        architecture: Tipo de arquitetura ('wavlm' ou 'wavlm_lite')
        **kwargs: Parâmetros adicionais

    Returns:
        Modelo Keras compilado
    """
    # Variante 'wavlm_aasist' → back-end de grafo AASIST (receita SOTA).
    if "aasist" in architecture and "backend" not in kwargs:
        kwargs["backend"] = "aasist"

    if architecture == 'wavlm_lite':
        return _create_wavlm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            architecture=architecture,
            classifier_units=[256, 128],
            dropout_rate=0.2,
            **kwargs
        )
    else:
        return _create_wavlm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            architecture=architecture,
            **kwargs
        )


# Registrar objetos personalizados no Keras
tf.keras.utils.get_custom_objects().update({
    'WavLMFeatureExtractor': WavLMFeatureExtractor,
    'preprocess': preprocess
})
