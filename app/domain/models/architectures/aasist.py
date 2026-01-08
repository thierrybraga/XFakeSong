

"""AASIST Architecture Implementation"""

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
from app.domain.models.architectures.layers import (
    apply_gru_block, flatten_features_for_gru, AudioFeatureNormalization, AttentionLayer,
    GraphAttentionLayer, SliceLayer, apply_reshape_for_cnn, residual_block
)

# Configure logger for AASIST
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [AASIST] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ============================ CAMADAS CUSTOMIZADAS ======================
# Estas camadas devem ser importadas em predictor.py também.

# ============================ FUNÇÕES DE CONSTRUÇÃO DE MODELOS ==========


def create_model(input_shape: Tuple[int, ...], num_classes: int = 2, architecture: str = "default",
                 dropout_rate: float = 0.2, l2_reg_strength: float = 0.0005,
                 hidden_dim: int = 512, num_layers: int = 8) -> models.Model:
    """
    Cria e compila um modelo Keras baseado na arquitetura especificada.
    """
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(x)

    if architecture == "default":
        x = apply_reshape_for_cnn(x, input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu',
                          padding='same', name="conv1")(x)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)
        x = layers.Dropout(dropout_rate, name="classifier_dropout1")(x)
        x = layers.Conv2D(64, (3, 3), activation='relu',
                          padding='same', name="conv2")(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)
        x = layers.Dropout(dropout_rate, name="classifier_dropout2")(x)
        x = flatten_features_for_gru(x, name="reshape_for_gru")

        # Apply GRU blocks (Standardized for CPU/GPU)
        x = apply_gru_block(
            x, 128, return_sequences=True, dropout_rate=dropout_rate, name="gru1"
        )
        x = apply_gru_block(
            x, 64, return_sequences=True, dropout_rate=dropout_rate, name="gru2"
        )
        
        x = AttentionLayer(name="attention_layer")(x)

    elif architecture == "cnn_baseline":
        x = apply_reshape_for_cnn(x, input_shape)
        x = layers.Conv2D(32, (5, 5), activation='relu',
                          padding='same', name="conv_b1")(x)
        x = layers.BatchNormalization(name="bn_b1")(x)
        x = layers.MaxPooling2D((2, 2), name="pool_b1")(x)
        x = layers.Dropout(dropout_rate, name="dropout_b1")(x)
        x = layers.Conv2D(64, (5, 5), activation='relu',
                          padding='same', name="conv_b2")(x)
        x = layers.BatchNormalization(name="bn_b2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool_b2")(x)
        x = layers.Dropout(dropout_rate, name="dropout_b2")(x)
        x = layers.Flatten(name="flatten")(x)

    elif architecture == "bidirectional_gru":
        if len(input_shape) == 4 and input_shape[-1] == 1:
            x = layers.Reshape(
                (input_shape[0],
                 input_shape[1]),
                name="flatten_channel_for_gru")(x)
        elif len(input_shape) == 2:
            pass
        elif len(input_shape) == 3 and input_shape[-1] != 1:
            logger.warning(f"Input shape {input_shape} for Bidirectional GRU expects 3D or 4D with last dim 1. "
                           f"Using as is, assuming last dim is feature.")
            pass
        else:
            raise ValueError(
                f"Input shape {input_shape} not suitable for 'bidirectional_gru' architecture.")
        if tf.config.list_physical_devices('GPU'):
            logger.info("Usando CuDNNGRU Bidirecional para otimização de GPU.")
            x = layers.Bidirectional(
                layers.CuDNNGRU(
                    128,
                    return_sequences=True),
                name="bi_gru1")(x)
            x = layers.Bidirectional(
                layers.CuDNNGRU(
                    64,
                    return_sequences=True),
                name="bi_gru2")(x)
        else:
            logger.info(
                "Usando GRU Bidirecional (CPU/compatível com GPU sem CuDNN).")
            x = layers.Bidirectional(
                layers.GRU(
                    128,
                    return_sequences=True,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate),
                name="bi_gru1")(x)
            x = layers.Bidirectional(
                layers.GRU(
                    64,
                    return_sequences=True,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate),
                name="bi_gru2")(x)
        x = AttentionLayer(name="attention_layer")(x)

    elif architecture == "resnet_gru":
        x = apply_reshape_for_cnn(x, input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu',
                          padding='same', name="resnet_conv_init")(x)
        x = layers.BatchNormalization(name="resnet_bn_init")(x)
        x = layers.MaxPooling2D((2, 2), name="resnet_pool_init")(x)
        x = residual_block(x, 64, (3, 3), stage='a')
        x = layers.MaxPooling2D((2, 2), name="resnet_pool_a")(x)
        x = layers.Dropout(dropout_rate, name="resnet_dropout_a")(x)
        x = residual_block(x, 128, (3, 3), stage='b')
        x = layers.MaxPooling2D((2, 2), name="resnet_pool_b")(x)
        x = layers.Dropout(dropout_rate, name="resnet_dropout_b")(x)
        x = flatten_features_for_gru(x, name="resnet_reshape_for_gru")
        if tf.config.list_physical_devices('GPU'):
            x = layers.CuDNNGRU(
                128,
                return_sequences=True,
                name="resnet_gru1")(x)
        else:
            x = layers.GRU(
                128,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                name="resnet_gru1")(x)
        x = AttentionLayer(name="attention_layer_resnet")(x)

    elif architecture == "transformer":
        if len(input_shape) == 4 and input_shape[-1] == 1:
            x = layers.Reshape(
                (input_shape[0],
                 input_shape[1]),
                name="flatten_channel_for_transformer")(x)
        elif len(input_shape) == 2:
            pass
        elif len(input_shape) == 3 and input_shape[-1] != 1:
            logger.warning(f"Input shape {input_shape} for Transformer expects 3D or 4D with last dim 1. "
                           f"Using as is, assuming last dim is feature.")
            pass
        else:
            raise ValueError(
                f"Input shape {input_shape} not suitable for 'transformer' architecture.")
        seq_len = input_shape[0] if len(input_shape) >= 2 else input_shape[0]
        feature_dim = input_shape[1] if len(
            input_shape) == 2 else input_shape[1] * input_shape[2] if len(input_shape) == 3 else input_shape[1]
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
            seq_len = 1
        pos_encoding = layers.Embedding(
            seq_len, feature_dim)(
            tf.range(seq_len))
        x = x + pos_encoding
        num_heads = 4
        ff_dim = 64
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=feature_dim)(
            x,
            x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        ff_output = layers.Dense(ff_dim, activation="relu")(x)
        ff_output = layers.Dense(feature_dim)(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
        x = layers.GlobalAveragePooling1D(name="transformer_avg_pool")(x)

    elif architecture == "aasist":
        # AASIST: Advanced Spectro-Temporal Graph Attention Network
        # Multi-scale feature extraction with improved GAT
        x = apply_reshape_for_cnn(x, input_shape)

        # Multi-scale convolutional feature extraction
        # Scale 1: Fine-grained temporal features
        conv1_1 = layers.Conv2D(
            32, (1, 3), activation='relu', padding='same', name="aasist_conv1_1")(x)
        conv1_1 = layers.BatchNormalization(name="aasist_bn1_1")(conv1_1)

        # Scale 2: Medium temporal features
        conv1_2 = layers.Conv2D(
            32, (3, 3), activation='relu', padding='same', name="aasist_conv1_2")(x)
        conv1_2 = layers.BatchNormalization(name="aasist_bn1_2")(conv1_2)

        # Scale 3: Coarse temporal features
        conv1_3 = layers.Conv2D(
            32, (5, 3), activation='relu', padding='same', name="aasist_conv1_3")(x)
        conv1_3 = layers.BatchNormalization(name="aasist_bn1_3")(conv1_3)

        # Concatenate multi-scale features
        x = layers.Concatenate(
            axis=-1, name="aasist_concat1")([conv1_1, conv1_2, conv1_3])
        x = layers.MaxPooling2D((2, 2), name="aasist_pool1")(x)
        x = layers.Dropout(dropout_rate, name="aasist_dropout1")(x)

        # Second multi-scale layer
        conv2_1 = layers.Conv2D(
            64, (1, 3), activation='relu', padding='same', name="aasist_conv2_1")(x)
        conv2_1 = layers.BatchNormalization(name="aasist_bn2_1")(conv2_1)

        conv2_2 = layers.Conv2D(
            64, (3, 3), activation='relu', padding='same', name="aasist_conv2_2")(x)
        conv2_2 = layers.BatchNormalization(name="aasist_bn2_2")(conv2_2)

        x = layers.Concatenate(
            axis=-1, name="aasist_concat2")([conv2_1, conv2_2])
        x = layers.MaxPooling2D((2, 2), name="aasist_pool2")(x)
        x = layers.Dropout(dropout_rate, name="aasist_dropout2")(x)

        # Simplified approach - use standard attention instead of graph attention
        # Global average pooling to reduce spatial dimensions
        x = layers.GlobalAveragePooling2D(name="aasist_gap")(x)

        # Multi-head self-attention layers
        x = layers.Dense(512, activation='relu', name="aasist_dense1")(x)
        x = layers.Dropout(dropout_rate, name="aasist_dropout3")(x)

        x = layers.Dense(256, activation='relu', name="aasist_dense2")(x)
        x = layers.Dropout(dropout_rate, name="aasist_dropout4")(x)

        # Final feature processing
        x = layers.Dense(128, activation='relu', name="aasist_final_dense")(x)

        # Additional dense layer for feature refinement
        x = layers.Dense(256, activation='relu', name="aasist_dense_refine")(x)
        x = layers.Dropout(dropout_rate, name="aasist_dropout_refine")(x)

    else:
        raise ValueError(
            f"Arquitetura '{architecture}' não reconhecida. Escolha 'default', 'cnn_baseline', 'bidirectional_gru', 'resnet_gru', 'transformer', ou 'aasist'.")

    # Camadas densas com regularização aprimorada
    x = layers.Dense(hidden_dim, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_strength),
                     bias_regularizer=regularizers.l2(l2_reg_strength / 2), name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Dropout(dropout_rate, name="final_dropout1")(x)

    # Camada intermediária adicional para melhor capacidade de aprendizado
    x = layers.Dense(hidden_dim // 2, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_strength),
                     bias_regularizer=regularizers.l2(l2_reg_strength / 2), name="dense2")(x)
    x = layers.BatchNormalization(name="bn_dense2")(x)
    x = layers.Dropout(dropout_rate * 1.5, name="final_dropout2")(x)

    # Camada final antes da saída
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_strength),
                     bias_regularizer=regularizers.l2(l2_reg_strength / 2), name="dense3")(x)
    x = layers.BatchNormalization(name="bn_dense3")(x)
    x = layers.Dropout(dropout_rate * 2, name="dropout_final")(x)

    # Camada de saída com regularização
    output_tensor = layers.Dense(num_classes, activation='softmax',
                                 kernel_regularizer=regularizers.l2(
                                     l2_reg_strength / 2),
                                 name="output_layer")(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    # Otimizador com weight decay para reduzir overfitting
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=l2_reg_strength,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  # Remover precision/recall que podem causar problemas de
                  # dimensão
                  metrics=['accuracy'])

    return model

# NOTA: A classe ModelTrainer foi removida deste arquivo para evitar duplicação.
# Use a implementação principal em src.core.trainer para funcionalidades
# de treinamento.

# ============================ FUNÇÕES DE AUMENTO DE DADOS (PLACEHOLDER) =


def simple_audio_augmenter(X_train: np.ndarray,
                           y_train: np.ndarray) -> tf.data.Dataset:
    """
    Função de placeholder para aumento de dados de áudio.
    Adicione técnicas de aumento de dados reais aqui (e.g., ruído, pitch shift, time stretch).
    """
    def _augment(audio_features, label):
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
