"""Transformer Architecture Implementation"""

from __future__ import annotations

# Standard library imports
import logging
import os
from datetime import datetime
from typing import List, Tuple, Optional, Any, Dict, Callable

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.domain.models.architectures.safe_normalization import SafeInstanceNormalization
from app.domain.models.architectures.layers import AttentionLayer

# Configure logger for Transformer
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [Transformer] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ============================ CAMADAS CUSTOMIZADAS ======================
# Estas camadas devem ser importadas em predictor.py também.


class SafeAudioNormalization(SafeInstanceNormalization):
    """
    Camada de normalização segura para features de áudio.
    Alias para SafeInstanceNormalization para manter compatibilidade.
    """

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(axis=axis, **kwargs)


class AudioFeatureNormalization(SafeInstanceNormalization):
    """
    DEPRECATED: Esta classe foi substituída por SafeAudioNormalization.
    Mantida apenas para compatibilidade com modelos existentes.
    Agora usa SafeInstanceNormalization internamente para garantir segurança.
    """

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(axis=axis, **kwargs)
        logger.warning(
            "AudioFeatureNormalization está DEPRECATED. "
            "Use SafeAudioNormalization ou SafeInstanceNormalization em vez disso."
        )




    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


# ============================ FUNÇÃO PARA CODIFICAÇÃO POSICIONAL ========


def get_positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """
    Gera codificação posicional sinusoidal para Transformer.

    Args:
        seq_len: Comprimento da sequência (número de frames).
        d_model: Dimensão das features (deve corresponder à dimensão do modelo).

    Returns:
        Tensor com codificação posicional de forma (1, seq_len, d_model).
    """

    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    positions = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angles = get_angles(positions, i, d_model)

    # Aplica sin para índices pares e cos para índices ímpares
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    pos_encoding = angles[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# ============================ BLOCO TRANSFORMER ============================


class TransformerEncoderBlock(layers.Layer):
    """
    Bloco Transformer com Multi-Head Self-Attention e Feed-Forward.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout_rate: float = 0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        # Multi-Head Self-Attention
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self) -> Dict[str, Any]:
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config

# ============================ FUNÇÕES DE CONSTRUÇÃO DE MODELOS ==========


def create_model(input_shape: Tuple[int, ...], num_classes: int = 2, architecture: str = "transformer",
                 dropout_rate: float = 0.3, l2_reg_strength: float = 0.001) -> models.Model:
    """
    Cria e compila um modelo Keras baseado na arquitetura especificada.

    Args:
        input_shape: A forma dos dados de entrada (e.g., (frames, features_dim) para Transformer).
        num_classes: Número de classes de saída (padrão é 2 para REAL/FAKE).
        architecture: O tipo de arquitetura do modelo a ser construído (atualizado apenas para 'transformer').
        dropout_rate: Taxa de dropout a ser aplicada em algumas camadas.
        l2_reg_strength: Força da regularização L2 para camadas densas.

    Returns:
        Um modelo Keras compilado.
    """
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(x)

    if architecture == "transformer":
        # Adaptação para Transformer: Entrada (batch, sequence_length,
        # features)
        if len(input_shape) == 4 and input_shape[-1] == 1:
            x = layers.Reshape(
                (input_shape[0],
                 input_shape[1]),
                name="flatten_channel_for_transformer")(x)
        elif len(input_shape) == 2:
            pass  # Already in correct shape
        elif len(input_shape) == 3 and input_shape[-1] != 1:
            logger.warning(f"Input shape {input_shape} for Transformer expects 3D or 4D with last dim 1. "
                           f"Using as is, assuming last dim is feature.")
            pass
        else:
            raise ValueError(
                f"Input shape {input_shape} not suitable for 'transformer' architecture.")

        # Definir parâmetros do Transformer
        seq_len = input_shape[0]  # Comprimento da sequência (frames)
        d_model = input_shape[1] if len(input_shape) == 2 else input_shape[1] * input_shape[2] if len(
            input_shape) == 3 else input_shape[1]
        num_heads = 8  # Aumentado para melhor captura de dependências
        ff_dim = 256  # Aumentado para maior capacidade
        num_layers = 4  # Número de blocos Transformer

        # Ajustar dimensões se necessário
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)  # (batch, 1, features_dim)
            seq_len = 1

        # Projeção inicial para ajustar d_model
        x = layers.Dense(
            d_model,
            activation="relu",
            name="input_projection")(x)

        # Adicionar codificação posicional
        pos_encoding = get_positional_encoding(seq_len, d_model)
        x = x + pos_encoding

        # Múltiplos blocos Transformer
        for i in range(num_layers):
            x = TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                name=f"transformer_block_{i + 1}"
            )(x)

        # Pooling global para classificação
        x = layers.GlobalAveragePooling1D(name="transformer_avg_pool")(x)

    else:
        raise ValueError(
            f"Arquitetura '{architecture}' não reconhecida. Apenas 'transformer' é suportado nesta versão.")

    # Camadas densas comuns
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg_strength),
        name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.5, name="dropout_final")(x)

    # Camada de saída
    output_tensor = layers.Dense(
        num_classes,
        activation='softmax',
        name="output_layer")(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model

# ModelTrainer removido - usar a implementação principal em src.core.trainer

# ============================ FUNÇÕES DE AUMENTO DE DADOS (PLACEHOLDER) =


def simple_audio_augmenter(X_train: np.ndarray,
                           y_train: np.ndarray) -> tf.data.Dataset:
    """
    Função de placeholder para aumento de dados de áudio.
    Adicione técnicas de aumento de dados reais aqui (e.g., ruído, pitch shift, time stretch).
    """

    def _augment(audio_features, label):
        # Exemplo: Adicionar ruído aleatório (muito simplificado)
        noise = tf.random.normal(
            shape=tf.shape(audio_features),
            mean=0.0,
            stddev=0.01,
            dtype=tf.float32)
        augmented_audio_features = audio_features + noise
        return augmented_audio_features, label

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Exemplo de uso (apenas para teste direto do arquivo)
# NOTA: Código de teste removido para evitar duplicação.
# Use os testes centralizados em src/tests/ ou src/core/trainer.py para
# funcionalidades de teste.
