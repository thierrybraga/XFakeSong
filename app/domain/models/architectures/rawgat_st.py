
"""RawGAT-ST Architecture Implementation"""

from __future__ import annotations

# Standard library imports
import logging
from typing import Tuple

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from app.domain.models.architectures.layers import (
    AASISTGraphAttentionLayer,
    AudioFeatureNormalization,
    AxisMaxAbsLayer,
    GATConvLayer,
    GraphPoolLayer,
    GraphReadoutLayer,
    MagnitudeLayer,
    ResidualBlock1D,
    ResidualBlock2D,
    SincConvLayer,
)

from app.domain.models.architectures.legacy_variants import (
    LEGACY_VARIANTS,
    build_legacy_model,
)

# Convenção do projeto: logger de módulo sem handlers manuais (a configuração
# de handlers/formatters é responsabilidade da aplicação; handlers locais
# duplicavam linhas de log).
logger = logging.getLogger(__name__)

# Temperatura da atenção de grafo (Tak et al., 2021). O AASIST, derivado deste
# trabalho, usa 2.0 nos GATs espectral/temporal — mesma escala adotada aqui
# para os três GATs (Gs, Gt e o espectro-temporal da fusão).
GAT_TEMPERATURE = 2.0

# Nº de nós a que Gs e Gt são reduzidos (por top-k) antes da fusão
# element-wise. O artigo obtém grafos compatíveis pelos próprios ratios de
# pooling; aqui o alvo é explícito porque o comprimento da janela é
# configurável e os dois eixos (frequência e tempo) não encolhem juntos.
GRAPH_FUSION_NODES = 12

# ============================ CAMADAS CUSTOMIZADAS ======================


def _rawgat_frontend(x: tf.Tensor) -> tf.Tensor:
    """Converte áudio bruto em um mapa Sinc espectro-temporal 2D."""
    x = SincConvLayer(
        n_filters=70,
        kernel_size=129,
        sample_rate=16000,
        name="rawgat_sinc",
    )(x)
    x = MagnitudeLayer(name="rawgat_sinc_abs")(x)
    x = layers.Permute((2, 1), name="rawgat_sinc_to_spectrogram")(x)
    x = layers.Reshape(
        (int(x.shape[1]), int(x.shape[2]), 1),
        name="rawgat_sinc_map",
    )(x)
    x = layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(3, 3),
        padding="same",
        name="rawgat_front_pool",
    )(x)
    x = layers.BatchNormalization(name="rawgat_front_bn")(x)
    return layers.Activation("selu", name="rawgat_front_selu")(x)


def _rawgat_encoder_branch(
    x: tf.Tensor,
    prefix: str,
    dropout_rate: float,
) -> tf.Tensor:
    """Encoder RawNet2 independente para um dos dois grafos."""
    for index, channels in enumerate((32, 32, 64, 64, 64, 64), start=1):
        x = ResidualBlock2D(
            channels,
            pool_size=(1, 3),
            name=f"{prefix}_encoder_{index}",
        )(x)
        if index in {2, 4}:
            x = layers.Dropout(
                dropout_rate * 0.5,
                name=f"{prefix}_encoder_drop_{index}",
            )(x)
    return x


def _build_paper_rawgat(
    input_tensor: tf.Tensor,
    x: tf.Tensor,
    num_classes: int,
    dropout_rate: float,
    l2_reg_strength: float,
    learning_rate: float,
    min_learning_rate: float,
    decay_steps: int,
    global_clipnorm: float,
) -> models.Model:
    """RawGAT-ST: grafos S/T separados, fusão multiplicativa e terceiro GAT."""
    if num_classes < 2:
        num_classes = 2
    if len(x.shape) == 2:
        x = layers.Reshape((-1, 1), name="rawgat_reshape_raw")(x)
    elif len(x.shape) != 3 or x.shape[-1] != 1:
        x = layers.Reshape((-1, 1), name="rawgat_reshape_raw")(x)

    front = _rawgat_frontend(x)
    spectral_map = _rawgat_encoder_branch(front, "rawgat_spectral", dropout_rate)
    temporal_map = _rawgat_encoder_branch(front, "rawgat_temporal", dropout_rate)

    # AxisMaxAbsLayer (não layers.Lambda com lambda Python crua): Keras 3
    # recusa desserializar Lambda de função Python em safe_mode (default),
    # o que quebrava o load do modelo salvo. Mesma computação exata.
    spectral = AxisMaxAbsLayer(
        axis=2, name="rawgat_spectral_nodes",
    )(spectral_map)
    temporal = AxisMaxAbsLayer(
        axis=1, name="rawgat_temporal_nodes",
    )(temporal_map)
    # A fusão multiplicativa e os logits de atenção são numericamente
    # sensíveis em float16. Mantemos somente o encoder 2D em mixed precision
    # e promovemos o pipeline gráfico para float32.
    spectral = layers.Activation(
        "linear", dtype="float32", name="rawgat_spectral_graph_float32"
    )(spectral)
    temporal = layers.Activation(
        "linear", dtype="float32", name="rawgat_temporal_graph_float32"
    )(temporal)
    # Atenção de grafo DO PAPER (produto par-a-par + tanh + temperatura), a
    # mesma formulação que o AASIST herda deste trabalho — não o GAT aditivo
    # de Velickovic que era usado aqui antes.
    spectral = AASISTGraphAttentionLayer(
        out_features=64, temperature=GAT_TEMPERATURE, dropout_rate=dropout_rate,
        name="rawgat_gat_spectral", dtype="float32",
    )(spectral)
    temporal = AASISTGraphAttentionLayer(
        out_features=64, temperature=GAT_TEMPERATURE, dropout_rate=dropout_rate,
        name="rawgat_gat_temporal", dtype="float32",
    )(temporal)
    spectral = GraphPoolLayer(
        0.81, name="rawgat_pool_spectral", dtype="float32"
    )(spectral)
    temporal = GraphPoolLayer(
        0.64, name="rawgat_pool_temporal", dtype="float32"
    )(temporal)
    # Alinhamento dos dois grafos antes da fusão element-wise por TOP-K
    # pooling — a mesma primitiva de pooling de grafo usada pelo artigo.
    # Antes isto era `AdaptiveGraphResize`, uma projeção DENSA aprendível sobre
    # o eixo de nós: além de não existir no paper, uma combinação linear de nós
    # não é uma operação de grafo (mistura nós arbitrariamente) e adicionava
    # parâmetros sem contrapartida na referência.
    spectral = GraphPoolLayer(
        target_nodes=GRAPH_FUSION_NODES, name="rawgat_align_spectral",
        dtype="float32",
    )(spectral)
    temporal = GraphPoolLayer(
        target_nodes=GRAPH_FUSION_NODES, name="rawgat_align_temporal",
        dtype="float32",
    )(temporal)

    fused = layers.Multiply(
        name="rawgat_graph_fusion", dtype="float32"
    )([spectral, temporal])
    fused = AASISTGraphAttentionLayer(
        out_features=32, temperature=GAT_TEMPERATURE, dropout_rate=dropout_rate,
        name="rawgat_gat_spectro_temporal", dtype="float32",
    )(fused)
    fused = GraphPoolLayer(
        0.64, name="rawgat_pool_spectro_temporal", dtype="float32"
    )(fused)
    readout = GraphReadoutLayer(
        name="rawgat_readout", dtype="float32"
    )(fused)
    readout = layers.Dropout(
        dropout_rate, name="rawgat_readout_dropout", dtype="float32"
    )(readout)
    output = layers.Dense(
        num_classes, activation=None, dtype="float32", name="output_layer"
    )(readout)
    model = models.Model(input_tensor, output, name="RawGAT_ST")
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=float(learning_rate),
        decay_steps=max(1, int(decay_steps)),
        alpha=float(min_learning_rate) / max(float(learning_rate), 1e-12),
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=schedule,
            weight_decay=l2_reg_strength,
            # AJUSTE 2026-08-06: era o literal 0.7, enquanto
            # registry.py::default_params declarava `gradient_clip: 0.5` — o
            # valor do registry NUNCA chegava aqui (config morto, o mesmo
            # padrão já eliminado do AASIST e do Conformer). Agora é
            # parâmetro de verdade, promovido pelo runner.
            global_clipnorm=float(global_clipnorm),
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


# ============================ FUNÇÕES DE CONSTRUÇÃO DE MODELOS ==========


def create_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
    architecture: str = "rawgat_st",
    # AJUSTE 2026-08-06: dropout 0.35->0.5 e l2 1e-3->3e-3 (sobreajuste em
    # clean_benchmark_15k — treino 0,998 vs val 0,85). Sincronizado com
    # registry.py::default_params e planning.py::NEURAL_BENCHMARK_HPARAMS.
    #
    # REVERTIDO EM 2026-08-17 pelo fatorial que o ajuste acima nunca teve:
    # dropout 0,50 (braço (d)) trava a validação em 0,5000 por 25 épocas com
    # treino a 95,4%; L2 3e-3 (braço (l)) não move o teto. Volta a 0,35/1e-3
    # nas TRÊS fontes. Tabela medida em planning.py::NEURAL_BENCHMARK_HPARAMS.
    dropout_rate: float = 0.35,
    l2_reg_strength: float = 0.001,
    attention_heads: int = 8,
    hidden_dim: int = 512,
    # (num_layers REMOVIDO: declarado e nunca lido — a profundidade é fixa
    # tanto na variante do paper quanto nas legadas.)
    temporal_pool_stride: int = 4,
    fusion_mode: str = "multiply",
    learning_rate: float = 5e-5,
    min_learning_rate: float = 5e-6,
    # 152.100 = ceil(24.324/16) x 100 épocas, o orçamento real do benchmark.
    decay_steps: int = 152_100,
    # Antes era o literal 0.7 dentro do compile; 0.5 é o valor que o
    # registry já declarava em `gradient_clip` e nunca chegava ao modelo.
    global_clipnorm: float = 0.5,
) -> models.Model:
    """
    Cria e compila um modelo Keras baseado na arquitetura especificada.

    Args:
        input_shape: A forma dos dados de entrada (e.g., (frames, features_dim, 1) para CNN).
        num_classes: Número de classes de saída (padrão é 2 para REAL/FAKE).
        architecture: O tipo de arquitetura.

    NOTA: ``attention_heads``, ``hidden_dim``, ``temporal_pool_stride`` e
    ``fusion_mode`` valem SOMENTE para as variantes
    legadas. A variante paper-faithful ``rawgat_st`` segue a topologia do
    artigo (dois encoders 2D, GAT S/T, fusão element-wise, terceiro GAT) e os
    ignora — não os coloque no registry esperando efeito.

    Variantes suportadas:
        - "rawgat_st"/"rawgat_st_paper": encoder 2D duplo + três GATs
        - "rawgat_st_legacy": implementação 1D anterior para checkpoints antigos
        - "rawgat_st_fast"/"rawgat_st_stable": aliases do legado otimizado
        - "cnn_gru_simple" / "default" (alias legado): CNN 2D + Bi-GRU + Attention
        - "cnn_baseline" | "bidirectional_gru" | "resnet_gru" | "transformer"

    Returns:
        Um modelo Keras compilado.
    """
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(x)

    # Alias "default" -> "rawgat_st" por consistência com AASIST.
    # O comportamento antigo (CNN+Bi-GRU) está disponível via "cnn_gru_simple".
    if architecture == "default":
        architecture = "rawgat_st"
    elif architecture == "rawgat_st_paper":
        architecture = "rawgat_st"
    elif architecture in {"rawgat_st_fast", "rawgat_st_stable", "rawgat_st_optimized"}:
        architecture = "rawgat_st_legacy"
        temporal_pool_stride = 8
        fusion_mode = "concat"

    if architecture in LEGACY_VARIANTS:
        # Variantes LEGADAS — implementação compartilhada em
        # legacy_variants.py (o mesmo código existia duplicado byte a byte
        # aqui e em aasist.py). Nomes de camada preservados para que
        # checkpoints antigos continuem carregando.
        return build_legacy_model(
            input_tensor=input_tensor,
            x=x,
            input_shape=input_shape,
            architecture=architecture,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            l2_reg_strength=l2_reg_strength,
            model_name="RawGAT_ST_legacy_variant",
        )

    elif architecture == "rawgat_st":
        return _build_paper_rawgat(
            input_tensor=input_tensor,
            x=x,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            l2_reg_strength=l2_reg_strength,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            decay_steps=decay_steps,
            global_clipnorm=global_clipnorm,
        )

    elif architecture == "rawgat_st_legacy":
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
        # (Tak et al., 2021) — implementação Keras alinhada ao paper.
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
        # O GAT materializa atenção densa N x N. O default usa downsampling
        # moderado para não inviabilizar batch/GPU; a variante rawgat_st_paper
        # define stride=1 e remove esta aproximação.
        temporal_pool_stride = max(1, int(temporal_pool_stride))
        if temporal_pool_stride > 1:
            temporal_nodes = layers.AveragePooling1D(
                pool_size=temporal_pool_stride,
                strides=temporal_pool_stride,
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
        # (assinatura do RawGAT-ST). A variante *_fast concatena tambem os
        # readouts individuais para preservar a receita otimizada anterior.
        fused = layers.Multiply(name="rawgat_graph_fusion")(
            [spec_readout, temp_readout])
        fusion_mode = str(fusion_mode).lower()
        if fusion_mode in {"multiply", "paper", "elementwise"}:
            x = fused
        elif fusion_mode in {"concat", "stable", "optimized"}:
            x = layers.Concatenate(name="rawgat_fusion_concat")(
                [fused, spec_readout, temp_readout])
        else:
            raise ValueError(
                "fusion_mode deve ser 'multiply'/'paper' ou 'concat'/'stable'."
            )

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
        # CORREÇÃO: o LR estava HARDCODED aqui e ignorava o parâmetro recebido
        # do registry/planning — o mesmo drift já corrigido no AASIST.
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=float(learning_rate),
            weight_decay=l2_reg_strength,
            global_clipnorm=0.7,  # estabilidade (grafos + SincConv)
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(
            "RawGAT-ST model created (SincNet + Gs/Gt GAT, fusion=%s, "
            "temporal_pool_stride=%d)",
            fusion_mode,
            temporal_pool_stride,
        )
        return model

    raise ValueError(
        f"Arquitetura '{architecture}' não reconhecida. Escolha 'rawgat_st' "
        f"(paper), 'rawgat_st_legacy' ou uma das variantes legadas "
        f"{list(LEGACY_VARIANTS)}."
    )


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
