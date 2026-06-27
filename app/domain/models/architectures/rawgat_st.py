
"""RawGAT-ST Architecture Implementation"""

from __future__ import annotations

# Standard library imports
import logging
from typing import Tuple

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from app.domain.models.architectures.layers import (
    AttentionLayer,
    AudioFeatureNormalization,
    GATConvLayer,
    GraphPoolLayer,
    GraphReadoutLayer,
    MagnitudeLayer,
    ResidualBlock1D,
    SincConvLayer,
    apply_gru_block,
    apply_reshape_for_cnn,
    flatten_features_for_gru,
    residual_block,
)

# Configure logger for RawGAT-ST
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [RawGAT-ST] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ============================ CAMADAS CUSTOMIZADAS ======================


# ============================ FUNÇÕES DE CONSTRUÇÃO DE MODELOS ==========


def create_model(input_shape: Tuple[int, ...], num_classes: int = 2, architecture: str = "rawgat_st",
                 dropout_rate: float = 0.2, l2_reg_strength: float = 0.0005,
                 attention_heads: int = 8, hidden_dim: int = 512, num_layers: int = 6) -> models.Model:
    """
    Cria e compila um modelo Keras baseado na arquitetura especificada.

    Args:
        input_shape: A forma dos dados de entrada (e.g., (frames, features_dim, 1) para CNN).
        num_classes: Número de classes de saída (padrão é 2 para REAL/FAKE).
        architecture: O tipo de arquitetura.

    Variantes suportadas:
        - "rawgat_st" (DEFAULT): Paper-faithful (SincNet + GAT spectro-temporal)
        - "cnn_gru_simple" / "default" (alias legado): CNN 2D + Bi-GRU + Attention
        - "cnn_baseline" | "bidirectional_gru" | "resnet_gru" | "transformer"

    Returns:
        Um modelo Keras compilado.
    """
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(x)

    # Alias "default" -> "rawgat_st" (paper-faithful) por consistência com aasist.
    # O comportamento antigo (CNN+Bi-GRU) está disponível via "cnn_gru_simple".
    if architecture == "default":
        architecture = "rawgat_st"

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
            logger.warning(
                f"Input shape {input_shape} for Bidirectional GRU expects 3D or 4D with last dim 1.")
            pass
        else:
            raise ValueError(
                f"Input shape {input_shape} not suitable for 'bidirectional_gru' architecture.")
        x = layers.Bidirectional(
            layers.GRU(128, return_sequences=True, dropout=dropout_rate),
            name="bi_gru1")(x)
        x = layers.Bidirectional(
            layers.GRU(64, return_sequences=True, dropout=dropout_rate),
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
        x = apply_gru_block(
            x,
            units=128,
            return_sequences=True,
            dropout_rate=dropout_rate,
            name="resnet_gru1"
        )
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
            logger.warning(
                f"Input shape {input_shape} for Transformer expects 3D or 4D with last dim 1.")
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

    elif architecture == "rawgat_st":
        # Cabeça softmax multi-classe: com num_classes=1, softmax de 1 unidade
        # emite constante 1.0 e a CCE é identicamente zero (não aprende).
        # Promove para 2 classes (real/fake).
        if num_classes < 2:
            logger.info(
                "RawGAT-ST: num_classes=1 promovido para 2 (softmax de 1 "
                "unidade degeneraria a saída)."
            )
            num_classes = 2
        # ================================================================
        # RawGAT-ST: End-to-End Spectro-Temporal Graph Attention Networks
        # (Tak et al., 2021) — IMPLEMENTAÇÃO FIEL AO PAPER.
        #
        # Antes: recebia ESPECTROGRAMA, simulava SincNet com Conv2D (kernels
        # 1×15/1×9/1×5) e aplicava GAT genérico — divergia do paper.
        #
        # Agora: opera sobre ÁUDIO BRUTO via SincConv (filtros passa-banda
        # aprendíveis), encoder residual, e DOIS grafos — espectral (Gs, canais
        # como nós) e temporal (Gt, tempo como nós) — fundidos por multiplicação
        # element-wise (a "graph combination" do paper).
        # ================================================================
        if len(input_shape) == 1:
            x = layers.Reshape((-1, 1), name="rawgat_reshape_raw")(x)
        elif len(input_shape) == 2 and input_shape[-1] == 1:
            pass  # já (batch, time, 1)
        else:
            x = layers.Reshape((-1, 1), name="rawgat_reshape_raw")(x)

        # --- 1. SincNet front-end (filtros passa-banda aprendíveis) ---
        x = SincConvLayer(
            n_filters=70, kernel_size=129, sample_rate=16000,
            name="rawgat_sinc")(x)
        x = MagnitudeLayer(name="rawgat_sinc_abs")(x)
        x = layers.BatchNormalization(name="rawgat_sinc_bn")(x)
        x = layers.LeakyReLU(negative_slope=0.3, name="rawgat_sinc_lrelu")(x)
        x = layers.MaxPooling1D(pool_size=3, name="rawgat_sinc_pool")(x)

        # --- 2. Encoder residual (estilo RawNet2) ---
        x = ResidualBlock1D(out_channels=64, kernel_size=3, name="rawgat_res1")(x)
        x = layers.MaxPooling1D(pool_size=3, name="rawgat_res_pool1")(x)
        x = ResidualBlock1D(out_channels=128, kernel_size=3, name="rawgat_res2")(x)
        x = layers.MaxPooling1D(pool_size=3, name="rawgat_res_pool2")(x)
        encoder_out = x  # (batch, T_reduced, 128)

        # --- 3. Dois grafos: espectral (canais como nós) e temporal (tempo) ---
        spectral_nodes = layers.Permute(
            (2, 1), name="rawgat_spectral_transpose")(encoder_out)  # (B, 128, T')
        temporal_nodes = encoder_out                                # (B, T', 128)
        # O GAT materializa atenção densa N x N. Em clips de 5s, T' ainda
        # passa de milhares de nós; reduzimos o grafo temporal antes da
        # atenção para manter o treino viável em GPUs de 12 GB.
        temporal_nodes = layers.AveragePooling1D(
            pool_size=8,
            strides=8,
            padding="same",
            name="rawgat_temporal_graph_downsample",
        )(temporal_nodes)

        # --- 4. Graph Attention em cada grafo ---
        spectral_nodes = GATConvLayer(
            out_features=32, num_heads=attention_heads // 2 or 4,
            dropout_rate=dropout_rate, concat_heads=True,
            name="rawgat_gat_spectral")(spectral_nodes)
        temporal_nodes = GATConvLayer(
            out_features=32, num_heads=attention_heads // 2 or 4,
            dropout_rate=dropout_rate, concat_heads=True,
            name="rawgat_gat_temporal")(temporal_nodes)

        # --- 5. Graph pooling (top-k) ---
        spectral_nodes = GraphPoolLayer(
            ratio=0.5, name="rawgat_pool_spectral")(spectral_nodes)
        temporal_nodes = GraphPoolLayer(
            ratio=0.5, name="rawgat_pool_temporal")(temporal_nodes)

        # --- 6. Readout (max + atenção) por grafo → vetor (B, 2*F) ---
        spec_readout = GraphReadoutLayer(
            name="rawgat_readout_spectral")(spectral_nodes)
        temp_readout = GraphReadoutLayer(
            name="rawgat_readout_temporal")(temporal_nodes)

        # --- 7. Fusão espectro-temporal: multiplicação element-wise ---
        # (assinatura do RawGAT-ST). Concatena também os readouts individuais
        # para estabilidade numérica (evita perda de info quando o produto ≈ 0).
        fused = layers.Multiply(name="rawgat_graph_fusion")(
            [spec_readout, temp_readout])
        x = layers.Concatenate(name="rawgat_fusion_concat")(
            [fused, spec_readout, temp_readout])

        # --- 8. Classificador ---
        x = layers.Dense(128, activation="relu", name="rawgat_fc")(x)
        x = layers.Dropout(dropout_rate, name="rawgat_fc_drop")(x)
        output_tensor = layers.Dense(
            num_classes, activation="softmax", dtype="float32",
            name="output_layer")(x)

        model = models.Model(inputs=input_tensor, outputs=output_tensor)
        # AJUSTE (retune): LR 1e-4->5e-5 e clipnorm 1.0->0.7 para conter a
        # divergencia (val_loss subia de 0.39->1.85). weight_decay vem do
        # l2_reg_strength (registry: 0.0005->0.001).
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.00005,
            weight_decay=l2_reg_strength,
            global_clipnorm=0.7,  # estabilidade (grafos + SincConv)
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(
            "RawGAT-ST model created (paper-faithful: SincNet + Gs/Gt GAT + "
            "fusão element-wise)"
        )
        return model

    else:
        raise ValueError(
            f"Arquitetura '{architecture}' não reconhecida. Escolha 'default', 'cnn_baseline', 'bidirectional_gru', 'resnet_gru', 'transformer', ou 'rawgat_st'.")

    # Camadas densas com regularização aprimorada
    x = layers.Dense(hidden_dim, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_strength),
                     bias_regularizer=regularizers.l2(l2_reg_strength / 2), name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Dropout(dropout_rate, name="final_dropout1")(x)

    # Camada intermediária adicional
    x = layers.Dense(hidden_dim // 2, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_strength),
                     bias_regularizer=regularizers.l2(l2_reg_strength / 2), name="dense2")(x)
    x = layers.BatchNormalization(name="bn_dense2")(x)
    x = layers.Dropout(min(dropout_rate * 1.5, 0.9), name="final_dropout2")(x)

    # Camada final antes da saída
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_strength),
                     bias_regularizer=regularizers.l2(l2_reg_strength / 2), name="dense3")(x)
    x = layers.BatchNormalization(name="bn_dense3")(x)
    x = layers.Dropout(min(dropout_rate * 2, 0.9), name="dropout_final")(x)

    # Camada de saída com regularização
    output_tensor = layers.Dense(num_classes, activation='softmax',
                                 kernel_regularizer=regularizers.l2(
                                     l2_reg_strength / 2),
                                 name="output_layer")(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    # Otimizador com weight decay
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

# ModelTrainer removido - usar a implementação principal em src.core.trainer


def simple_audio_augmenter(X_train: np.ndarray,
                           y_train: np.ndarray) -> tf.data.Dataset:
    """Augmentation RawBoost (Tak et al., 2022) para áudio bruto.

    Substitui o antigo placeholder de ruído gaussiano fixo (ver aasist.py).
    """
    from app.domain.models.training.rawboost import rawboost_tf

    def _augment(audio_features, label):
        rank = audio_features.shape.rank
        a = audio_features
        if rank == 2 and audio_features.shape[-1] == 1:
            a = tf.squeeze(a, axis=-1)
        a = rawboost_tf(a, sr=16000, algo=4, p=0.8)
        if rank == 2:
            a = tf.expand_dims(a, axis=-1)
        return a, label
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# NOTA: Código de teste removido para evitar duplicação.
# Use os testes centralizados em src/tests/ ou src/core/trainer.py para
# funcionalidades de teste.
