"""Modelos dos perfis de voz: uma cabeça por família de arquitetura.

O perfil trabalha com um vetor fixo de features (``features.py``), não com
áudio: cada família do registry ganha um modelo pequeno inspirado nela
(atenção, transformer, Conv1D, recorrente, multi-ramo), com MLP de reserva.
"""

import logging

logger = logging.getLogger(__name__)


def build_profile_model(
    input_dim: int,
    architecture: str,
):
    """Constrói modelo para voice profile usando o architecture registry.

    Para arquiteturas DL do registry: constrói um adapter de features
    que mapeia o vetor de features fixo para a entrada esperada pela
    arquitetura, usa a arquitetura do registry como backbone, e adiciona
    um classificador binário.

    Fallback: modelo MLP compacto se a arquitetura do registry falhar.
    """

    # Tentar usar a arquitetura do registry
    try:
        from app.domain.models.architectures.registry import (
            ArchitectureRegistry,
            architecture_registry,
        )

        # Resolver display name
        display_name = ArchitectureRegistry.normalize_architecture_name(architecture)
        arch_info = architecture_registry.get_architecture(display_name)
        input_req = arch_info.input_requirements

        logger.info(
            f"Construindo modelo com arquitetura '{display_name}' "
            f"(input_type={input_req.get('type')}) para profile (feature_dim={input_dim})"
        )

        # Para arquiteturas baseadas em features/spectrogram, usar adapter
        # O perfil trabalha com features fixas (146-dim), então criamos
        # um modelo que projeta essas features para o formato da arquitetura
        # e usa camadas inspiradas na arquitetura escolhida

        # Buscar configuração da arquitetura
        arch_params = arch_info.default_params
        dropout = arch_params.get("dropout_rate", 0.3)

        # Construir modelo adaptado por tipo de arquitetura
        if display_name in ("AASIST", "RawGAT-ST"):
            # Modelos de atenção: usar self-attention layers
            model = _build_attention_profile_model(input_dim, dropout, arch_params)
        elif display_name in ("Conformer", "SpectrogramTransformer"):
            # Transformer-based: usar transformer encoder leve
            model = _build_transformer_profile_model(input_dim, dropout, arch_params)
        elif display_name in ("RawNet2", "Sonic Sleuth"):
            # Conv-based: usar 1D convolutions sobre features
            model = _build_conv1d_profile_model(input_dim, dropout, arch_params)
        elif display_name in ("EfficientNet-LSTM", "WavLM", "HuBERT"):
            # Recurrent: usar LSTM/GRU sobre features
            model = _build_recurrent_profile_model(input_dim, dropout, arch_params)
        elif display_name in ("Ensemble", "MultiscaleCNN", "Hybrid CNN-Transformer"):
            # Multi-branch: usar arquitetura multi-head
            model = _build_multihead_profile_model(input_dim, dropout, arch_params)
        else:
            # Fallback genérico
            model = _build_mlp_profile_model(input_dim, dropout)

        logger.info(
            f"Modelo '{display_name}' construído: " f"{model.count_params()} parâmetros"
        )
        return model

    except Exception as e:
        logger.warning(
            f"Falha ao construir modelo via registry para '{architecture}': {e}. "
            f"Usando MLP fallback."
        )
        return _build_mlp_profile_model(input_dim, 0.3)


def _build_mlp_profile_model(input_dim: int, dropout: float):
    """MLP compacto (fallback e baseline)."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(dropout * 0.7),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def _build_attention_profile_model(input_dim: int, dropout: float, params: dict):
    """Modelo com self-attention (inspirado em AASIST/RawGAT-ST)."""
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(input_dim,))
    # Reshape para sequência de pseudo-tokens
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Reshape((16, 16))(x)
    # Multi-Head Attention
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=dropout)(
        x, x
    )
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)
    # Feed-forward
    ff = tf.keras.layers.Dense(64, activation="relu")(x)
    ff = tf.keras.layers.Dense(16)(ff)
    x = tf.keras.layers.Add()([x, ff])
    x = tf.keras.layers.LayerNormalization()(x)
    # Pool & classify
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="attention_profile")


def _build_transformer_profile_model(input_dim: int, dropout: float, params: dict):
    """Modelo Transformer encoder leve (inspirado em Conformer/AST)."""
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.Reshape((8, 16))(x)

    # 2 Transformer blocks
    for _ in range(2):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16, dropout=dropout
        )(x, x)
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)
        ff = tf.keras.layers.Dense(32, activation="gelu")(x)
        ff = tf.keras.layers.Dense(16)(ff)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="transformer_profile")


def _build_conv1d_profile_model(input_dim: int, dropout: float, params: dict):
    """Modelo Conv1D (inspirado em RawNet2/Sonic Sleuth)."""
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)

    # Conv blocks
    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv1D(
            filters,
            3,
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="conv1d_profile")


def _build_recurrent_profile_model(input_dim: int, dropout: float, params: dict):
    """Modelo LSTM bidirecional (inspirado em EfficientNet-LSTM/WavLM/HuBERT)."""
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(input_dim,))
    # Reshape em sequência
    x = tf.keras.layers.Reshape((input_dim // 2, 2))(inputs)
    # Bi-LSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=dropout)
    )(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, dropout=dropout))(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="recurrent_profile")


def _build_multihead_profile_model(input_dim: int, dropout: float, params: dict):
    """Modelo multi-branch (inspirado em Ensemble/MultiscaleCNN/Hybrid)."""
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(input_dim,))

    # Branch 1: Dense
    b1 = tf.keras.layers.Dense(128, activation="relu")(inputs)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Dropout(dropout)(b1)
    b1 = tf.keras.layers.Dense(64, activation="relu")(b1)

    # Branch 2: Conv1D
    x2 = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    x2 = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x2)
    x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
    b2 = tf.keras.layers.Dense(64, activation="relu")(x2)

    # Branch 3: Feature selection via attention
    attn_weights = tf.keras.layers.Dense(input_dim, activation="softmax")(inputs)
    x3 = tf.keras.layers.Multiply()([inputs, attn_weights])
    b3 = tf.keras.layers.Dense(64, activation="relu")(x3)

    # Fusion
    merged = tf.keras.layers.Concatenate()([b1, b2, b3])
    x = tf.keras.layers.Dense(128, activation="relu")(merged)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="multihead_profile")
