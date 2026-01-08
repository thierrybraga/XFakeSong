"""Implementação da arquitetura HuBERT para detecção de deepfakes.

Esta implementação segue uma abordagem de duas etapas:
1. Extrator de características usando modelo HuBERT pré-treinado
2. Classificador MLP para detecção de deepfakes

Autor: Sistema de Detecção de Deepfakes
"""

import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from app.domain.models.architectures.layers import create_classification_head
from typing import Tuple, Optional, Dict, Any
import librosa
from app.core.utils.audio_utils import normalize_audio, pad_or_truncate

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificar disponibilidade do Transformers (compatível com Keras 3)
try:
    # Para compatibilidade com Keras 3, usar implementação simplificada
    HF_AVAILABLE = False
    logger.info(
        "Using simplified HuBERT implementation for Keras 3 compatibility.")
except Exception as e:
    HF_AVAILABLE = False
    logger.warning(f"Transformers library not available: {e}")


class HuBERTFeatureExtractor(layers.Layer):
    """Extrator de características baseado em HuBERT simplificado.

    Esta implementação simula o comportamento do HuBERT usando
    camadas convolucionais e de atenção otimizadas para compatibilidade com Keras 3.
    """

    def __init__(self,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 12,
                 freeze_features: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        # Usar configuração completa para máxima acurácia
        self.num_hidden_layers = num_hidden_layers  # Sem limitação de camadas
        self.freeze_features = freeze_features

        # Camadas convolucionais com strides para downsampling (simulando
        # HuBERT original)
        self.conv_layers = [
            layers.Conv1D(
                512,
                10,
                strides=5,
                padding='same',
                activation='gelu',
                name="conv_0"),
            layers.Conv1D(
                512,
                3,
                strides=2,
                padding='same',
                activation='gelu',
                name="conv_1"),
            layers.Conv1D(
                512,
                3,
                strides=2,
                padding='same',
                activation='gelu',
                name="conv_2"),
            layers.Conv1D(
                512,
                3,
                strides=2,
                padding='same',
                activation='gelu',
                name="conv_3"),
            layers.Conv1D(
                512,
                3,
                strides=2,
                padding='same',
                activation='gelu',
                name="conv_4"),
            layers.Conv1D(
                512,
                2,
                strides=2,
                padding='same',
                activation='gelu',
                name="conv_5"),
            layers.Conv1D(
                hidden_size,
                2,
                strides=2,
                padding='same',
                activation='gelu',
                name="conv_6")
        ]

        # Camadas de extração de características (projeção linear após convs)
        self.feature_projection = layers.Dense(
            hidden_size, name="feature_projection")
        self.layer_norm = layers.LayerNormalization(name="layer_norm")

        # Camadas de atenção multi-cabeça completas para máxima acurácia
        self.attention_layers = []
        for i in range(self.num_hidden_layers):
            attention_layer = layers.MultiHeadAttention(
                num_heads=num_attention_heads,  # Usar todas as cabeças de atenção
                key_dim=hidden_size // num_attention_heads,
                name=f"attention_{i}"
            )
            self.attention_layers.append(attention_layer)

        # Camadas de feed-forward completas para máxima acurácia
        self.feed_forward_layers = []
        for i in range(self.num_hidden_layers):
            ff_layer = keras.Sequential([
                layers.Dense(
                    hidden_size * 4,
                    activation='gelu',
                    name=f"ff_dense1_{i}"),
                # Restaurar 4x para máxima capacidade
                layers.Dropout(0.1, name=f"ff_dropout_{i}"),
                layers.Dense(hidden_size, name=f"ff_dense2_{i}")
            ], name=f"feed_forward_{i}")
            self.feed_forward_layers.append(ff_layer)

        # Normalização adicional
        self.attention_layer_norms = []
        self.ff_layer_norms = []
        for i in range(self.num_hidden_layers):
            self.attention_layer_norms.append(
                layers.LayerNormalization(name=f"attention_ln_{i}")
            )
            self.ff_layer_norms.append(
                layers.LayerNormalization(name=f"ff_ln_{i}")
            )

    def call(self, inputs, training=None):
        x = inputs

        # Aplicar camadas convolucionais (feature extractor)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Projeção das características
        x = self.feature_projection(x)
        x = self.layer_norm(x)

        # Aplicar camadas de atenção e feed-forward (reduzidas)
        for i in range(self.num_hidden_layers):
            # Atenção multi-cabeça com conexão residual
            attention_output = self.attention_layers[i](
                x, x, training=training)
            x = self.attention_layer_norms[i](x + attention_output)

            # Feed-forward com conexão residual
            ff_output = self.feed_forward_layers[i](x, training=training)
            x = self.ff_layer_norms[i](x + ff_output)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'freeze_features': self.freeze_features
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


def create_hubert_model(input_shape: Tuple[int, ...],
                        num_classes: int = 1,
                        architecture: str = 'hubert',
                        **kwargs) -> keras.Model:
    """Criar modelo HuBERT completo.

    Args:
        input_shape: Formato de entrada (samples,)
        num_classes: Número de classes de saída
        architecture: Variante da arquitetura ('hubert', 'hubert_lite')
        **kwargs: Argumentos adicionais

    Returns:
        Modelo HuBERT compilado
    """
    logger.info(
        f"Creating HuBERT model with input_shape={input_shape}, num_classes={num_classes}")

    # Configurações baseadas na variante (focadas em acurácia máxima)
    if architecture == 'hubert_lite':
        config = {
            'hidden_size': 512,
            'num_attention_heads': 8,
            'num_hidden_layers': 6,
            'classifier_hidden_dim': 256,
            'dropout_rate': 0.2
        }
    else:  # hubert padrão
        config = {
            'hidden_size': 768,
            'num_attention_heads': 12,
            'num_hidden_layers': 12,
            'classifier_hidden_dim': 512,
            'dropout_rate': 0.3
        }

    # Atualizar com kwargs
    config.update(kwargs)

    # Entrada
    inputs = keras.Input(shape=input_shape, name="audio_input")

    # Pré-processamento: converter áudio 1D para features 2D
    # Simular extração de características espectrais (otimizado)
    x = layers.Reshape((-1, 1))(inputs)  # (batch, time, 1)

    # Camadas convolucionais otimizadas para extração de características
    # básicas
    x = layers.Conv1D(32, 5, strides=4, padding='same', activation='relu')(
        x)  # Reduzir filtros e aumentar stride
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, strides=2, padding='same',
                      activation='relu')(x)  # Reduzir kernel size
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(
        config['hidden_size'],
        3,
        padding='same',
        activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Adicionar pooling para reduzir dimensionalidade temporal
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    # Extrator de características HuBERT
    feature_extractor = HuBERTFeatureExtractor(
        hidden_size=config['hidden_size'],
        num_attention_heads=config['num_attention_heads'],
        num_hidden_layers=config['num_hidden_layers'],
        name="hubert_feature_extractor"
    )

    hidden_states = feature_extractor(x)

    # Classificador
    outputs, loss = create_classification_head(
        hidden_states,
        num_classes,
        dropout_rate=config['dropout_rate'],
        hidden_dims=[config['classifier_hidden_dim']]
    )

    # Criar modelo
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"hubert_{architecture}")

    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )

    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w)
                           for w in model.trainable_weights])

    logger.info(
        f"HuBERT model created successfully with {total_params} total parameters ({trainable_params} trainable)")

    return model


def create_model(input_shape: Tuple[int, ...],
                 num_classes: int = 1,
                 architecture: str = 'hubert',
                 **kwargs) -> keras.Model:
    """Interface principal para criação do modelo HuBERT.

    Args:
        input_shape: Formato de entrada
        num_classes: Número de classes
        architecture: Variante da arquitetura
        **kwargs: Argumentos adicionais

    Returns:
        Modelo HuBERT
    """
    return create_hubert_model(
        input_shape, num_classes, architecture, **kwargs)


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

        print(f"✓ Modelo criado com sucesso")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")

        # Limpar memória
        del model
        tf.keras.backend.clear_session()

    print("\n✅ Todos os testes passaram!")
