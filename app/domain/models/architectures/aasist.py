

"""AASIST Architecture Implementation"""

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
    AASISTHtrgGraphAttentionLayer,
    AMSoftmaxCrossEntropy,
    AMSoftmaxLayer,
    AudioFeatureNormalization,
    AxisMaxAbsLayer,
    GATConvLayer,
    GraphPoolLayer,
    GraphReadoutLayer,
    HSGALLayer,
    MagnitudeLayer,
    MasterNodeSeed,
    ResidualBlock1D,
    ResidualBlock2D,
    SincConvLayer,
    SpectralPositionEmbedding,
)
from app.domain.models.architectures.legacy_variants import (
    LEGACY_VARIANTS,
    build_legacy_model,
)

# Convenção do projeto: logger de módulo sem handlers manuais (a configuração
# de handlers/formatters é responsabilidade da aplicação; handlers locais
# duplicavam linhas de log).
logger = logging.getLogger(__name__)

# Escala/margem do AM-Softmax (CosFace). Ficam AQUI, num único lugar, porque
# precisam casar entre a camada (`AMSoftmaxLayer`, que emite s·cos θ) e a loss
# (`AMSoftmaxCrossEntropy`, que aplica a margem). scale=15 em vez dos 30-64 do
# paper original mantém os logits numericamente seguros em float16.
AM_SOFTMAX_SCALE = 15.0
AM_SOFTMAX_MARGIN = 0.35

# Temperaturas da atenção de grafo (Jung et al., ICASSP 2022 — config oficial
# `temperatures: [2.0, 2.0, 100.0, 100.0]`): 2.0 nos GATs espectral/temporal e
# 100.0 nas HS-GAL. Temperatura alta ≈ atenção quase uniforme, que é o regime
# em que os autores estabilizam a camada heterogênea.
GAT_TEMPERATURE = 2.0
HSGAL_TEMPERATURE = 100.0

# ============================ CAMADAS CUSTOMIZADAS ======================
# Estas camadas devem ser importadas em predictor.py também.


def _build_aasist_encoder(x: tf.Tensor, dropout_rate: float) -> tf.Tensor:
    """RawNet2 2D encoder que preserva os eixos espectral e temporal."""
    x = SincConvLayer(
        n_filters=70, kernel_size=129, sample_rate=16000,
        name="aasist_sinc",
    )(x)
    x = MagnitudeLayer(name="aasist_sinc_abs")(x)
    x = layers.Permute((2, 1), name="aasist_sinc_to_spectrogram")(x)
    x = layers.Reshape(
        (int(x.shape[1]), int(x.shape[2]), 1),
        name="aasist_sinc_map",
    )(x)
    x = layers.MaxPooling2D(
        pool_size=(3, 3), strides=(3, 3), padding="same",
        name="aasist_front_pool",
    )(x)
    x = layers.BatchNormalization(name="aasist_front_bn")(x)
    x = layers.Activation("selu", name="aasist_front_selu")(x)
    for index, channels in enumerate((32, 32, 64, 64, 64, 64), start=1):
        x = ResidualBlock2D(
            channels, pool_size=(1, 3), name=f"aasist_encoder_{index}"
        )(x)
        if index in {2, 4}:
            x = layers.Dropout(
                dropout_rate * 0.5, name=f"aasist_encoder_drop_{index}"
            )(x)
    return x


def _build_paper_aasist(
    input_tensor: tf.Tensor,
    x: tf.Tensor,
    num_classes: int,
    dropout_rate: float,
    l2_reg_strength: float,
    classifier_head: str,
    learning_rate: float,
    min_learning_rate: float,
    decay_steps: int,
) -> models.Model:
    """AASIST com mapa 2D, master node, quatro HS-GALs e MGO."""
    if num_classes < 2:
        num_classes = 2
    if len(x.shape) == 2:
        x = layers.Reshape((-1, 1), name="aasist_reshape_raw")(x)
    elif len(x.shape) != 3 or x.shape[-1] != 1:
        x = layers.Reshape((-1, 1), name="aasist_reshape_raw")(x)

    encoded = _build_aasist_encoder(x, dropout_rate)
    # AxisMaxAbsLayer (não layers.Lambda com lambda Python crua): Keras 3
    # recusa desserializar Lambda de função Python em safe_mode (default),
    # o que quebrava o load do modelo salvo. Mesma computação exata
    # (max(|x|, axis)), só a forma de serializar muda.
    spectral = AxisMaxAbsLayer(
        axis=2, name="aasist_spectral_nodes",
    )(encoded)
    temporal = AxisMaxAbsLayer(
        axis=1, name="aasist_temporal_nodes",
    )(encoded)
    spectral = SpectralPositionEmbedding(name="aasist_spectral_position")(spectral)
    # Atenção de grafo DO PAPER (produto par-a-par + tanh + temperatura),
    # não o GAT aditivo de Velickovic. Temperaturas do artigo: 2.0 nos GATs
    # espectral/temporal e 100.0 nas HS-GAL.
    spectral = AASISTGraphAttentionLayer(
        out_features=64, temperature=GAT_TEMPERATURE,
        dropout_rate=dropout_rate, name="aasist_gat_spectral",
    )(spectral)
    temporal = AASISTGraphAttentionLayer(
        out_features=64, temperature=GAT_TEMPERATURE,
        dropout_rate=dropout_rate, name="aasist_gat_temporal",
    )(temporal)
    spectral = GraphPoolLayer(0.5, name="aasist_pool_spectral")(spectral)
    temporal = GraphPoolLayer(0.7, name="aasist_pool_temporal")(temporal)

    # Master node treinável por ramo (o AASIST injeta `master1`/`master2` como
    # nn.Parameter; a média dos nós é o fallback da própria HS-GAL).
    master1 = MasterNodeSeed(64, name="aasist_master1")(spectral)
    master2 = MasterNodeSeed(64, name="aasist_master2")(spectral)

    s1, t1, m1 = AASISTHtrgGraphAttentionLayer(
        out_features=32, temperature=HSGAL_TEMPERATURE,
        dropout_rate=dropout_rate, name="aasist_hsgal_11",
    )([spectral, temporal, master1])
    s1 = GraphPoolLayer(0.5, name="aasist_hpool_s1")(s1)
    t1 = GraphPoolLayer(0.5, name="aasist_hpool_t1")(t1)
    s1_aug, t1_aug, m1_aug = AASISTHtrgGraphAttentionLayer(
        out_features=32, temperature=HSGAL_TEMPERATURE,
        dropout_rate=dropout_rate, name="aasist_hsgal_12",
    )([s1, t1, m1])
    s1 = layers.Add(name="aasist_residual_s1")([s1, s1_aug])
    t1 = layers.Add(name="aasist_residual_t1")([t1, t1_aug])
    m1 = layers.Add(name="aasist_residual_m1")([m1, m1_aug])

    s2, t2, m2 = AASISTHtrgGraphAttentionLayer(
        out_features=32, temperature=HSGAL_TEMPERATURE,
        dropout_rate=dropout_rate, name="aasist_hsgal_21",
    )([spectral, temporal, master2])
    s2 = GraphPoolLayer(0.5, name="aasist_hpool_s2")(s2)
    t2 = GraphPoolLayer(0.5, name="aasist_hpool_t2")(t2)
    s2_aug, t2_aug, m2_aug = AASISTHtrgGraphAttentionLayer(
        out_features=32, temperature=HSGAL_TEMPERATURE,
        dropout_rate=dropout_rate, name="aasist_hsgal_22",
    )([s2, t2, m2])
    s2 = layers.Add(name="aasist_residual_s2")([s2, s2_aug])
    t2 = layers.Add(name="aasist_residual_t2")([t2, t2_aug])
    m2 = layers.Add(name="aasist_residual_m2")([m2, m2_aug])

    spectral = layers.Maximum(name="aasist_mgo_spectral")([s1, s2])
    temporal = layers.Maximum(name="aasist_mgo_temporal")([t1, t2])
    master = layers.Maximum(name="aasist_mgo_master")([m1, m2])
    readout = layers.Concatenate(name="aasist_extended_readout")([
        layers.GlobalMaxPooling1D(name="aasist_temporal_max")(
            MagnitudeLayer(name="aasist_temporal_abs")(temporal)
        ),
        layers.GlobalAveragePooling1D(name="aasist_temporal_mean")(temporal),
        layers.GlobalMaxPooling1D(name="aasist_spectral_max")(
            MagnitudeLayer(name="aasist_spectral_abs")(spectral)
        ),
        layers.GlobalAveragePooling1D(name="aasist_spectral_mean")(spectral),
        layers.Flatten(name="aasist_master_readout")(master),
    ])
    readout = layers.Dropout(dropout_rate, name="aasist_readout_dropout")(readout)

    head = str(classifier_head).lower()
    if head in {"am_softmax", "amsoftmax", "cosface"}:
        output = AMSoftmaxLayer(
            num_classes, scale=AM_SOFTMAX_SCALE, margin=AM_SOFTMAX_MARGIN,
            name="output_layer",
        )(readout)
        # CORREÇÃO: a margem do AM-Softmax vive na LOSS, não na camada. No
        # grafo funcional os rótulos nunca chegam ao `call` da AMSoftmaxLayer,
        # então o `margin=0.35` dela é inerte e esta cabeça era "AM-Softmax"
        # só no nome — treinava com CE sobre s·cos(θ), sem margem alguma
        # (o caminho legado abaixo já fazia o CosFace na loss).
        loss = AMSoftmaxCrossEntropy(
            scale=AM_SOFTMAX_SCALE, margin=AM_SOFTMAX_MARGIN,
            label_smoothing=0.1,
        )
    elif head in {"cross_entropy", "ce", "dense"}:
        output = layers.Dense(
            num_classes, activation=None, dtype="float32", name="output_layer"
        )(readout)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        raise ValueError("classifier_head deve ser 'cross_entropy' ou 'am_softmax'")

    output = layers.Activation(
        "linear", dtype="float32", name="output_cast"
    )(output)
    model = models.Model(input_tensor, output, name="AASIST")
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=float(learning_rate),
        decay_steps=max(1, int(decay_steps)),
        alpha=float(min_learning_rate) / max(float(learning_rate), 1e-12),
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=schedule,
            weight_decay=l2_reg_strength,
            global_clipnorm=1.0,
        ),
        loss=loss,
        metrics=["accuracy"],
    )
    logger.info(
        "AASIST criado (encoder 2D + GAT S/T + master node + MGO; head=%s, "
        "lr=%s, weight_decay=%s)", head, learning_rate, l2_reg_strength,
    )
    return model


# ============================ FUNÇÕES DE CONSTRUÇÃO DE MODELOS ==========


def create_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
    architecture: str = "aasist",
    dropout_rate: float = 0.2,
    # Defaults ALINHADOS com registry.py::default_params e
    # benchmarks/planning.py (retune: l2 2e-4, LR 3e-4). Antes divergiam
    # (5e-4/1e-4), então quem construísse o modelo sem passar os parâmetros
    # — o caminho do app/Gradio — treinava com uma receita diferente da
    # documentada.
    l2_reg_strength: float = 0.0002,
    hidden_dim: int = 512,
    # (num_layers REMOVIDO: era declarado e nunca lido — nem pela variante do
    # paper, que tem profundidade fixa pelo artigo, nem pelas legadas, cuja
    # profundidade é fixa no builder. Aceitá-lo dava a impressão falsa de um
    # knob de profundidade.)
    classifier_head: str = "cross_entropy",
    learning_rate: float = 3e-4,
    min_learning_rate: float = 5e-6,
    decay_steps: int = 100_000,
) -> models.Model:
    """
    Cria e compila um modelo Keras baseado na arquitetura especificada.

    NOTA: ``hidden_dim`` vale SOMENTE para as variantes legadas
    (cnn_gru_simple/cnn_baseline/resnet_gru/transformer). A variante
    paper-faithful ``aasist`` tem topologia fixa pelo artigo e o ignora.

    Variantes suportadas:
        - "aasist" (DEFAULT): encoder 2D + GAT S/T + master node + MGO
        - "aasist_legacy": implementação 1D anterior, para checkpoints antigos
        - "cnn_gru_simple" / "default" (alias legado): CNN 2D + Bi-GRU + Attention
        - "cnn_baseline": CNN 2D simples + flatten
        - "bidirectional_gru": Bi-GRU + Attention (sem CNN)
        - "resnet_gru": ResNet blocks + GRU + Attention
        - "transformer": Multi-head Self-Attention + FF
    """
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(x)

    # Alias "default" -> "aasist" (paper-faithful) para que usuários recebam
    # a versão correta por padrão via factory/registry. O comportamento antigo
    # (CNN+Bi-GRU) ainda está disponível via "cnn_gru_simple".
    if architecture == "default":
        architecture = "aasist"

    if architecture in LEGACY_VARIANTS:
        # Variantes LEGADAS (CNN+Bi-GRU, baseline, Bi-GRU, ResNet+GRU,
        # transformer simples) — implementação compartilhada em
        # legacy_variants.py. O mesmo código existia duplicado byte a byte
        # aqui e em rawgat_st.py; os nomes de camada são preservados para que
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
            model_name="AASIST_legacy_variant",
        )

    elif architecture == "aasist":
        return _build_paper_aasist(
            input_tensor=input_tensor,
            x=x,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            l2_reg_strength=l2_reg_strength,
            classifier_head=classifier_head,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            decay_steps=decay_steps,
        )

    elif architecture == "aasist_legacy":
        # A cabeça AM-Softmax é inerentemente multi-classe: com num_classes=1
        # a CCE sobre 1 logit é identicamente zero (modelo não aprende).
        # Promove para 2 classes (real/fake) — o Predictor já entende ambas
        # as convenções de saída.
        if num_classes < 2:
            logger.info(
                "AASIST: num_classes=1 promovido para 2 (AM-Softmax é "
                "multi-classe; 1 classe degeneraria a loss)."
            )
            num_classes = 2
        # ================================================================
        # AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
        # Graph Attention Networks (Jung et al., ICASSP 2022)
        #
        # Paper-faithful implementation with:
        # - SincConv encoder (learnable bandpass filters on raw waveform)
        # - Residual blocks with pre-activation (BN -> LeakyReLU -> Conv1D)
        # - Dual-branch: spectral GAT + temporal GAT
        # - HS-GAL: Heterogeneous Stacking Graph Attention Layer
        # - Graph readout (max + attention)
        # ================================================================

        # Ensure raw audio input is (batch, time, 1)
        if len(input_shape) == 1:
            x = layers.Reshape((-1, 1), name="aasist_reshape_raw")(x)
        elif len(input_shape) == 2 and input_shape[-1] == 1:
            pass  # Already (batch, time, 1)
        else:
            x = layers.Reshape((-1, 1), name="aasist_reshape_raw")(x)

        # --- 1. SincConv Encoder ---
        # Learnable bandpass filters (128 filters, kernel=129)
        x = SincConvLayer(
            n_filters=128, kernel_size=129, sample_rate=16000,
            name="sinc_conv")(x)
        x = MagnitudeLayer(name="sinc_abs")(x)
        x = layers.BatchNormalization(name="sinc_bn")(x)
        x = layers.LeakyReLU(negative_slope=0.3, name="sinc_lrelu")(x)
        x = layers.MaxPooling1D(pool_size=3, name="sinc_pool")(x)

        # --- 2. Residual Blocks Group 1 ---
        x = ResidualBlock1D(out_channels=128, kernel_size=3, name="res_block_1a")(x)
        x = ResidualBlock1D(out_channels=128, kernel_size=3, name="res_block_1b")(x)
        x = layers.MaxPooling1D(pool_size=3, name="res_pool_1")(x)

        # --- 3. Residual Blocks Group 2 ---
        x = ResidualBlock1D(out_channels=256, kernel_size=3, name="res_block_2a")(x)
        x = ResidualBlock1D(out_channels=256, kernel_size=3, name="res_block_2b")(x)
        x = layers.MaxPooling1D(pool_size=3, name="res_pool_2")(x)

        # --- 4. Residual Blocks Group 3 (paridade com o paper) ---
        # O encoder do AASIST segue o RawNet2: 6 blocos residuais no total
        # ([128,128,256,256,256,256]). Antes havia só 4 — adicionamos o 3º
        # grupo (mais 2 blocos de 256) para casar com a profundidade do paper.
        x = ResidualBlock1D(out_channels=256, kernel_size=3, name="res_block_3a")(x)
        x = ResidualBlock1D(out_channels=256, kernel_size=3, name="res_block_3b")(x)
        x = layers.MaxPooling1D(pool_size=3, name="res_pool_3")(x)

        # encoder_out: (batch, T_reduced, 256)
        encoder_out = x

        # --- 4. Spectral Branch ---
        # Transpose: treat channels (256) as nodes, time steps as features
        # spectral_nodes: (batch, 256, T_reduced)
        spectral_nodes = layers.Permute((2, 1), name="spectral_transpose")(encoder_out)

        # --- 5. Temporal Branch ---
        # temporal_nodes: (batch, T_reduced, 256) - as is
        temporal_nodes = encoder_out

        # --- 6. GAT on each branch ---
        spectral_nodes = GATConvLayer(
            out_features=32, num_heads=4, dropout_rate=dropout_rate,
            concat_heads=True, name="gat_spectral")(spectral_nodes)
        temporal_nodes = GATConvLayer(
            out_features=32, num_heads=4, dropout_rate=dropout_rate,
            concat_heads=True, name="gat_temporal")(temporal_nodes)

        # --- 7. HS-GAL: Heterogeneous Stacking Graph Attention ---
        spectral_nodes, temporal_nodes = HSGALLayer(
            out_features=32, num_heads=2, dropout_rate=dropout_rate,
            name="hsgal_1")([spectral_nodes, temporal_nodes])
        spectral_nodes, temporal_nodes = HSGALLayer(
            out_features=32, num_heads=2, dropout_rate=dropout_rate,
            name="hsgal_2")([spectral_nodes, temporal_nodes])

        # --- 8. Graph Pooling ---
        spectral_nodes = GraphPoolLayer(
            ratio=0.5, name="graph_pool_spectral")(spectral_nodes)
        temporal_nodes = GraphPoolLayer(
            ratio=0.5, name="graph_pool_temporal")(temporal_nodes)

        # --- 9. Graph Readout (max + attention) ---
        spec_readout = GraphReadoutLayer(name="readout_spectral")(spectral_nodes)
        temp_readout = GraphReadoutLayer(name="readout_temporal")(temporal_nodes)

        # --- 10. Concatenate and classify ---
        x = layers.Concatenate(name="aasist_readout_concat")([spec_readout, temp_readout])
        # AMSoftmaxLayer output: scaled cosine logits (range ≈ [-15, 15]).
        # scale=15 (reduced from 30) keeps logit magnitudes within a numerically
        # safe range for float16/float32 while still providing discriminative margins.
        output_tensor = AMSoftmaxLayer(
            num_classes, scale=AM_SOFTMAX_SCALE, margin=AM_SOFTMAX_MARGIN,
            name="output_layer",
        )(x)
        # Explicit float32 cast so mixed_float16 policy doesn't produce float16 logits
        output_tensor = layers.Activation('linear', dtype='float32', name='output_cast')(output_tensor)

        # Build complete model with paper-faithful architecture
        model = models.Model(inputs=input_tensor, outputs=output_tensor)

        # BUG FIX: AMSoftmaxLayer emite logits brutos — NÃO probabilidades.
        # categorical_crossentropy(from_logits=False) faz log(y_pred) e
        # y_pred ∈ [-15, 15] → log(valor_negativo) = NaN → loss NaN na época 1.
        # `AMSoftmaxCrossEntropy` usa from_logits=True (log-sum-exp estável) e
        # aplica a margem CosFace no logit da classe-alvo — a margem da CAMADA
        # é inerte no grafo funcional (os rótulos não chegam ao call dela).
        # Antes isto era uma closure local, que impedia
        # `load_model(..., compile=True)`; agora é uma Loss registrada.
        loss = AMSoftmaxCrossEntropy(
            scale=AM_SOFTMAX_SCALE, margin=AM_SOFTMAX_MARGIN,
            label_smoothing=0.1,
        )

        # AJUSTE (retune): subajuste (val_acc travada ~0.92). LR 1e-4->3e-4
        # (regularizacao estava forte demais p/ o LR baixo).
        #
        # CORREÇÃO (fiação de hiperparâmetros): weight_decay e learning_rate
        # eram valores FIXOS aqui e os parâmetros recebidos do registry/planning
        # eram silenciosamente ignorados — o retune documentado não chegava ao
        # otimizador. Ambos agora são consumidos de fato (mesmo padrão da
        # variante paper-faithful e do RawGAT-ST).
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=float(learning_rate),
            weight_decay=l2_reg_strength,
            global_clipnorm=1.0,  # previne gradientes explosivos
        )

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

        logger.info(
            "AASIST legacy criado (SincConv 1D + GAT + HS-GAL; lr=%s, "
            "weight_decay=%s)", learning_rate, l2_reg_strength,
        )
        return model

    raise ValueError(
        f"Arquitetura '{architecture}' não reconhecida. Escolha 'aasist' "
        f"(paper), 'aasist_legacy' ou uma das variantes legadas "
        f"{list(LEGACY_VARIANTS)}."
    )


# NOTA: A classe ModelTrainer foi removida deste arquivo para evitar duplicação.
# Use a implementação principal em src.core.trainer para funcionalidades
# de treinamento.

# ============================ FUNÇÕES DE AUMENTO DE DADOS (PLACEHOLDER) =


def simple_audio_augmenter(X_train: np.ndarray,
                           y_train: np.ndarray) -> tf.data.Dataset:
    """Augmentation RawBoost (Tak et al., 2022) para áudio bruto.

    Substitui o antigo placeholder de ruído gaussiano fixo. Aplica as três
    famílias de distorção do RawBoost (convolutiva linear+não-linear,
    impulsiva dependente do sinal e estacionária colorida) — a augmentation
    que mais melhora a generalização para ataques de spoofing não vistos.
    """
    from app.domain.models.training.rawboost import rawboost_tf

    def _augment(audio_features, label):
        rank = audio_features.shape.rank
        a = audio_features
        if rank == 2 and audio_features.shape[-1] == 1:
            a = tf.squeeze(a, axis=-1)  # (T, 1) -> (T,)
        a = rawboost_tf(a, sr=16000, algo=4, p=0.8)
        if rank == 2:
            a = tf.expand_dims(a, axis=-1)  # de volta a (T, 1)
        return a, label

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Exemplo de uso (apenas para teste direto do arquivo)
# NOTA: Código de teste removido para evitar duplicação.
# Use os testes centralizados em src/tests/ ou src/core/trainer.py para
# funcionalidades de teste.
