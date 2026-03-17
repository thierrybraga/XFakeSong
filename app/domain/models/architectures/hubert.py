"""Implementação da arquitetura HuBERT para detecção de deepfakes.

Esta implementação segue uma abordagem de duas etapas:
1. Extrator de características usando modelo HuBERT pré-treinado
2. Classificador MLP para detecção de deepfakes

Autor: Sistema de Detecção de Deepfakes
"""

import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from app.core.utils.audio_utils import normalize_audio, pad_or_truncate
from app.domain.models.architectures.layers import (
    AttentionPoolingLayer,
    create_classification_head,
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificar disponibilidade do Transformers (compatível com Keras 3)
try:
    from transformers import TFHubertModel, Wav2Vec2Processor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TFHubertModel = None
    Wav2Vec2Processor = None
    logger.info("Transformers library not found. Using simplified HuBERT implementation.")


class HuBERTFeatureExtractor(layers.Layer):
    """Extrator de características baseado em HuBERT.

    Usa o modelo pré-treinado do HuggingFace com soma ponderada de estados ocultos
    ou uma implementação simplificada como fallback.
    """

    def __init__(self,
                 model_name: str = "facebook/hubert-base-ls960",
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 12,
                 freeze_weights: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.freeze_weights = freeze_weights

        if HF_AVAILABLE:
            try:
                # Carregar modelo HuBERT pré-treinado
                self.hubert_model = TFHubertModel.from_pretrained(
                    model_name,
                    from_tf=True
                )

                # Configurar para retornar todos os hidden states para weighted sum
                # HuBERT base tem 12 camadas + 1 embedding inicial
                self.actual_num_layers = self.hubert_model.config.num_hidden_layers + 1
                from app.domain.models.architectures.layers import WeightedSumLayer
                self.weighted_sum = WeightedSumLayer(num_layers=self.actual_num_layers)

                # Congelar pesos se especificado
                if freeze_weights:
                    self.hubert_model.trainable = False

                logger.info(f"HuBERT model {model_name} loaded successfully with {self.actual_num_layers} layers")

            except Exception as e:
                logger.warning(
                    f"Failed to load HuBERT model: {e}. Using simplified implementation.")
                self._use_simplified = True
        else:
            self._use_simplified = True

        if hasattr(self, '_use_simplified'):
            # Implementação simplificada usando CNN 1D + Transformers (Keras nativo)
            self._build_simplified_extractor()

    def _build_simplified_extractor(self):
        """Constrói um extrator simplificado usando CNN 1D + Transformers."""
        # Camadas convolucionais com strides para downsampling (simulando HuBERT original)
        self.conv_layers = [
            layers.Conv1D(512, 10, strides=5, padding='same', activation='gelu', name="conv_0"),
            layers.Conv1D(512, 3, strides=2, padding='same', activation='gelu', name="conv_1"),
            layers.Conv1D(512, 3, strides=2, padding='same', activation='gelu', name="conv_2"),
            layers.Conv1D(512, 3, strides=2, padding='same', activation='gelu', name="conv_3"),
            layers.Conv1D(512, 3, strides=2, padding='same', activation='gelu', name="conv_4"),
            layers.Conv1D(512, 2, strides=2, padding='same', activation='gelu', name="conv_5"),
            layers.Conv1D(self.hidden_size, 2, strides=2, padding='same', activation='gelu', name="conv_6")
        ]

        # Camadas de extração de características (projeção linear após convs)
        self.feature_projection = layers.Dense(self.hidden_size, name="feature_projection")
        self.layer_norm = layers.LayerNormalization(name="layer_norm")

        # Camadas de atenção multi-cabeça
        self.attention_layers = []
        self.attention_layer_norms = []
        self.ff_layers = []
        self.ff_layer_norms = []

        for i in range(self.num_hidden_layers):
            self.attention_layers.append(
                layers.MultiHeadAttention(
                    num_heads=self.num_attention_heads,
                    key_dim=self.hidden_size // self.num_attention_heads,
                    name=f"attention_{i}"
                )
            )
            self.attention_layer_norms.append(layers.LayerNormalization(name=f"attn_ln_{i}"))

            self.ff_layers.append(
                keras.Sequential([
                    layers.Dense(self.hidden_size * 4, activation='gelu'),
                    layers.Dropout(0.1),
                    layers.Dense(self.hidden_size)
                ], name=f"ff_{i}")
            )
            self.ff_layer_norms.append(layers.LayerNormalization(name=f"ff_ln_{i}"))

    def call(self, inputs, training=None):
        """Forward pass do extrator de características."""
        if hasattr(self, '_use_simplified'):
            x = inputs
            if len(x.shape) == 2:
                x = tf.expand_dims(x, axis=-1)

            # Aplicar camadas convolucionais
            for conv_layer in self.conv_layers:
                x = conv_layer(x)

            # Projeção das características
            x = self.feature_projection(x)
            x = self.layer_norm(x)

            # Aplicar camadas de atenção e feed-forward
            for i in range(self.num_hidden_layers):
                attn_out = self.attention_layers[i](x, x, training=training)
                x = self.attention_layer_norms[i](x + attn_out)

                ff_out = self.ff_layers[i](x, training=training)
                x = self.ff_layer_norms[i](x + ff_out)

            return x
        else:
            # Usar modelo HuBERT pré-treinado
            if len(inputs.shape) == 3:
                inputs = tf.squeeze(inputs, axis=-1)

            # Extrair com hidden states para weighted sum (Fidelidade ao paper)
            outputs = self.hubert_model(inputs, output_hidden_states=True, training=False)

            # Soma ponderada de todos os hidden states
            x = self.weighted_sum(list(outputs.hidden_states))

            return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'freeze_weights': self.freeze_weights
        })
        return config





def preprocess_audio(audio_data: np.ndarray,
                     target_sr: int = 16000,
                     max_length: Optional[int] = None) -> np.ndarray:
    """Pré-processamento de áudio para HuBERT.

    Args:
        audio_data: Dados de áudio bruto
        target_sr: Taxa de amostragem alvo (16kHz)
        max_length: Comprimento máximo em amostras

    Returns:
        Áudio pré-processado
    """
    # Normalização
    audio_data = normalize_audio(audio_data)

    # Truncar ou fazer padding se necessário
    if max_length is not None:
        audio_data = pad_or_truncate(audio_data, max_length)

    return audio_data


def _create_hubert_model(input_shape: Tuple[int, ...],
                         num_classes: int = 1,
                         architecture: str = 'hubert',
                         model_name: str = "facebook/hubert-base-ls960",
                         freeze_hubert: bool = True,
                         classifier_hidden_dim: int = 256,
                         dropout_rate: float = 0.3,
                         **kwargs) -> keras.Model:
    """Cria modelo HuBERT completo fiel ao paper.

    Args:
        input_shape: Formato da entrada (samples,)
        num_classes: Número de classes (1 para classificação binária)
        architecture: Nome da arquitetura
        model_name: Nome do modelo HuBERT pré-treinado
        freeze_hubert: Se deve congelar pesos do HuBERT
        classifier_hidden_dim: Unidades da camada oculta do classificador
        dropout_rate: Taxa de dropout
        **kwargs: Argumentos adicionais (hidden_size, num_attention_heads, etc.)

    Returns:
        Modelo Keras compilado
    """
    logger.info(
        f"Creating HuBERT model with input_shape={input_shape}, num_classes={num_classes}")

    # 1. Entrada (Raw Audio)
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # 2. Extrator de características HuBERT (Self-Supervised)
    feature_extractor = HuBERTFeatureExtractor(
        model_name=model_name,
        freeze_weights=freeze_hubert,
        hidden_size=kwargs.get('hidden_size', 768),
        num_attention_heads=kwargs.get('num_attention_heads', 12),
        num_hidden_layers=kwargs.get('num_hidden_layers', 12),
        name='hubert_feature_extractor'
    )
    # x shape: (batch, sequence_length, hidden_size)
    x = feature_extractor(inputs)

    # 3. Temporal convolution block for richer representations
    conv_out = layers.Conv1D(256, 3, padding='same', activation='relu', name='temporal_conv1')(x)
    conv_out = layers.BatchNormalization(name='temporal_bn1')(conv_out)
    conv_out2 = layers.Conv1D(256, 3, padding='same', activation='relu', name='temporal_conv2')(conv_out)
    conv_out2 = layers.BatchNormalization(name='temporal_bn2')(conv_out2)
    conv_out2 = conv_out2 + conv_out  # residual

    # 4. Attention Pooling
    x = AttentionPoolingLayer(name='attention_pool')(conv_out2)

    # 5. Classification Head
    outputs, loss = create_classification_head(
        x,
        num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=[512, 256, 128]
    )

    # Criar modelo
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"hubert_{architecture}")

    # Compilar modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(f"HuBERT model {architecture} created successfully")
    return model


def create_model(input_shape: Tuple[int, ...],
                 num_classes: int = 1,
                 architecture: str = 'hubert',
                 **kwargs) -> keras.Model:
    """Função principal para criar modelos HuBERT.

    Args:
        input_shape: Formato da entrada (samples,)
        num_classes: Número de classes
        architecture: Tipo de arquitetura ('hubert' ou 'hubert_lite')
        **kwargs: Parâmetros adicionais

    Returns:
        Modelo Keras compilado
    """
    if architecture == 'hubert_lite':
        return _create_hubert_model(
            input_shape=input_shape,
            num_classes=num_classes,
            architecture=architecture,
            num_hidden_layers=6,
            classifier_hidden_dim=128,
            dropout_rate=0.2,
            **kwargs
        )
    else:
        return _create_hubert_model(
            input_shape=input_shape,
            num_classes=num_classes,
            architecture=architecture,
            **kwargs
        )


def preprocess(audio_data: np.ndarray, target_sr: int = 16000) -> np.ndarray:
    """Interface de pré-processamento.

    Args:
        audio_data: Dados de áudio
        target_sr: Taxa de amostragem alvo

    Returns:
        Áudio pré-processado
    """
    return preprocess_audio(audio_data, target_sr)


# Registrar objetos personalizados para serialização
keras.utils.get_custom_objects().update({
    'HuBERTFeatureExtractor': HuBERTFeatureExtractor
})


if __name__ == "__main__":
    # Teste básico
    print("Testando criação do modelo HuBERT...")

    # Testar com diferentes tamanhos de entrada
    test_shapes = [
        (48000,),   # 3 segundos a 16kHz
        (80000,),   # 5 segundos a 16kHz
        (160000,),  # 10 segundos a 16kHz
    ]

    for shape in test_shapes:
        print(f"\nTestando com input_shape: {shape}")

        # Criar modelo
        model = create_model(input_shape=shape, num_classes=1)

        # Testar inferência
        dummy_input = np.random.randn(1, *shape).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)

        print("✓ Modelo criado com sucesso")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")

        # Limpar memória
        del model
        tf.keras.backend.clear_session()

    print("\n✅ Todos os testes passaram!")
