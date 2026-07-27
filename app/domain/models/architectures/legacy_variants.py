"""Variantes LEGADAS compartilhadas por AASIST e RawGAT-ST.

Estas topologias (CNN+Bi-GRU, CNN baseline, Bi-GRU, ResNet+GRU e um bloco
Transformer simples) **não** correspondem a nenhum dos artigos implementados:
são versões anteriores do projeto, mantidas apenas para desserializar e
reavaliar checkpoints antigos. As variantes fiéis aos papers vivem em
``aasist.py`` (``aasist``) e ``rawgat_st.py`` (``rawgat_st``).

Motivo do módulo: o mesmo código existia DUPLICADO byte a byte nos dois
arquivos (~300 linhas cada), incluindo a cabeça densa final — qualquer
correção precisava ser feita em dois lugares e, na prática, divergia. Os nomes
de camada são preservados exatamente como eram, para que os checkpoints
existentes continuem carregando.
"""

from __future__ import annotations

import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from app.domain.models.architectures.layers import (
    AttentionLayer,
    apply_gru_block,
    apply_reshape_for_cnn,
    flatten_features_for_gru,
    residual_block,
)

logger = logging.getLogger(__name__)

#: Variantes legadas atendidas por este módulo.
LEGACY_VARIANTS = (
    "cnn_gru_simple",
    "cnn_baseline",
    "bidirectional_gru",
    "resnet_gru",
    "transformer",
)


def _reshape_sequence_input(x, input_shape, variant_label: str, name: str):
    """Normaliza a entrada para (batch, tempo, features) nas variantes de sequência."""
    if len(input_shape) == 4 and input_shape[-1] == 1:
        return layers.Reshape((input_shape[0], input_shape[1]), name=name)(x)
    if len(input_shape) == 2:
        return x
    if len(input_shape) == 3 and input_shape[-1] != 1:
        logger.warning(
            "Input shape %s para '%s' espera 3D/4D com última dim 1. "
            "Usando como está, assumindo que a última dim é feature.",
            input_shape, variant_label,
        )
        return x
    raise ValueError(
        f"Input shape {input_shape} not suitable for '{variant_label}' architecture."
    )


def build_legacy_backbone(x, input_shape: Tuple[int, ...], architecture: str,
                          dropout_rate: float):
    """Constrói o backbone de uma variante legada e devolve o tensor achatado.

    Returns:
        Tensor 2D (batch, features) pronto para :func:`build_legacy_head`.

    Raises:
        ValueError: se ``architecture`` não for uma variante legada conhecida.
    """
    if architecture == "cnn_gru_simple":
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
        # GRU padronizada CPU/GPU (usa o kernel cuDNN quando elegível)
        x = apply_gru_block(
            x, 128, return_sequences=True, dropout_rate=dropout_rate, name="gru1"
        )
        x = apply_gru_block(
            x, 64, return_sequences=True, dropout_rate=dropout_rate, name="gru2"
        )
        return AttentionLayer(name="attention_layer")(x)

    if architecture == "cnn_baseline":
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
        return layers.Flatten(name="flatten")(x)

    if architecture == "bidirectional_gru":
        x = _reshape_sequence_input(
            x, input_shape, "bidirectional_gru", "flatten_channel_for_gru"
        )
        x = layers.Bidirectional(
            layers.GRU(128, return_sequences=True, dropout=dropout_rate),
            name="bi_gru1")(x)
        x = layers.Bidirectional(
            layers.GRU(64, return_sequences=True, dropout=dropout_rate),
            name="bi_gru2")(x)
        return AttentionLayer(name="attention_layer")(x)

    if architecture == "resnet_gru":
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
        x = apply_gru_block(
            x, 128, return_sequences=True, dropout_rate=dropout_rate,
            name="resnet_gru1",
        )
        return AttentionLayer(name="attention_layer_resnet")(x)

    if architecture == "transformer":
        x = _reshape_sequence_input(
            x, input_shape, "transformer", "flatten_channel_for_transformer"
        )
        seq_len = input_shape[0]
        if len(input_shape) == 2:
            feature_dim = input_shape[1]
        elif len(input_shape) == 3:
            feature_dim = input_shape[1] * input_shape[2]
        else:
            feature_dim = input_shape[1]
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
            seq_len = 1
        pos_encoding = layers.Embedding(seq_len, feature_dim)(tf.range(seq_len))
        x = x + pos_encoding
        num_heads = 4
        ff_dim = 64
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=feature_dim)(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        ff_output = layers.Dense(ff_dim, activation="relu")(x)
        ff_output = layers.Dense(feature_dim)(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
        return layers.GlobalAveragePooling1D(name="transformer_avg_pool")(x)

    raise ValueError(
        f"Variante legada '{architecture}' não reconhecida. "
        f"Disponíveis: {list(LEGACY_VARIANTS)}"
    )


def build_legacy_head(x, num_classes: int, hidden_dim: int,
                      dropout_rate: float, l2_reg_strength: float):
    """Cabeça densa das variantes legadas (Dense 3× + BN + Dropout → softmax)."""
    x = layers.Dense(hidden_dim, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_strength),
                     bias_regularizer=regularizers.l2(l2_reg_strength / 2),
                     name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Dropout(dropout_rate, name="final_dropout1")(x)

    x = layers.Dense(hidden_dim // 2, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_strength),
                     bias_regularizer=regularizers.l2(l2_reg_strength / 2),
                     name="dense2")(x)
    x = layers.BatchNormalization(name="bn_dense2")(x)
    x = layers.Dropout(min(dropout_rate * 1.5, 0.9), name="final_dropout2")(x)

    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_strength),
                     bias_regularizer=regularizers.l2(l2_reg_strength / 2),
                     name="dense3")(x)
    x = layers.BatchNormalization(name="bn_dense3")(x)
    x = layers.Dropout(min(dropout_rate * 2, 0.9), name="dropout_final")(x)

    # dtype='float32' na saída: sob mixed_float16 softmax+crossentropy em fp16
    # satura/perde precisão (mesma correção aplicada nas variantes atuais).
    return layers.Dense(num_classes, activation='softmax',
                        kernel_regularizer=regularizers.l2(l2_reg_strength / 2),
                        name="output_layer", dtype='float32')(x)


def build_legacy_model(input_tensor, x, input_shape: Tuple[int, ...],
                       architecture: str, num_classes: int, hidden_dim: int,
                       dropout_rate: float, l2_reg_strength: float,
                       model_name: str = None) -> models.Model:
    """Monta e compila uma variante legada completa (backbone + cabeça)."""
    x = build_legacy_backbone(x, input_shape, architecture, dropout_rate)
    output_tensor = build_legacy_head(
        x, num_classes, hidden_dim, dropout_rate, l2_reg_strength
    )
    model = models.Model(inputs=input_tensor, outputs=output_tensor,
                         name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=l2_reg_strength,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    logger.info(
        "Variante LEGADA '%s' criada (não corresponde a nenhum paper — use "
        "'aasist'/'rawgat_st' para as implementações fiéis).", architecture,
    )
    return model
