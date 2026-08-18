"""Spectrogram Transformer Architecture Implementation"""

# Third-party imports
import logging
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from app.domain.models.architectures.layers import (
    ASTInputNormalization,
    ExpandDimsLayer,
    LogMelSpectrogramLayer,
    ResizeLayer,
    STFTLayer,
    ensure_flat_input,
    is_raw_audio,
)

logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class SafeSpectrogramReshapeLayer(layers.Layer):
    """Layer to safely reshape spectrogram inputs."""

    def __init__(self, input_shape_tuple, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_tuple = input_shape_tuple

    def call(self, x):
        input_shape = self.input_shape_tuple

        if len(input_shape) == 2:
            # (time, features) -> (time, features, 1)
            x = tf.expand_dims(x, axis=-1)

            # Ensure minimum size
            target_height = tf.maximum(64, input_shape[0])
            target_width = tf.maximum(64, input_shape[1])
            x = tf.image.resize(x, (target_height, target_width))

        elif len(input_shape) == 3:
            if input_shape[-1] == 1:
                # Already has channel dimension
                target_height = tf.maximum(64, input_shape[0])
                target_width = tf.maximum(64, input_shape[1])
                x = tf.image.resize(x, (target_height, target_width))
            else:
                # Multiple channels, keep as is but ensure minimum size
                target_height = tf.maximum(64, input_shape[0])
                target_width = tf.maximum(64, input_shape[1])
                x = tf.image.resize(x, (target_height, target_width))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'input_shape_tuple': self.input_shape_tuple})
        return config


def create_safe_spectrogram_layer(input_shape):
    """Creates a SafeSpectrogramReshapeLayer."""
    return SafeSpectrogramReshapeLayer(
        input_shape, name='safe_spectrogram_reshape')


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class ClassTokenLayer(layers.Layer):
    """Custom layer to add class token to patch embeddings."""

    def __init__(self, embed_dim, **kwargs):
        super(ClassTokenLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, self.embed_dim),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_tokens = tf.tile(self.class_token, [batch_size, 1, 1])
        return tf.concat([class_tokens, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class PatchEmbedding(layers.Layer):
    """
    Patch embedding layer for Spectrogram Transformer.
    Supports overlapping patches as per the AST paper.
    """

    def __init__(self, patch_size: Tuple[int, int], embed_dim: int, stride: Optional[Tuple[int, int]] = None, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride if stride is not None else patch_size

        # Convolutional layer to create patches (with optional overlap)
        self.conv = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=self.stride,
            padding='valid',
            name='patch_conv'
        )

    def build(self, input_shape):
        # A sub-camada é criada no __init__; sem um build() explícito o Keras 3
        # marca a camada como construída SEM construir a Conv2D interna
        # ("does not have a build() method ... may cause failures down the
        # line") e o peso pode não ser restaurado no load.
        self.conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, height, width, channels)
        x = self.conv(inputs)
        # Reshape to (batch, num_patches, embed_dim)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embed_dim])
        return x

    def compute_output_shape(self, input_shape):
        conv_shape = self.conv.compute_output_shape(input_shape)
        num_patches = None
        if conv_shape[1] is not None and conv_shape[2] is not None:
            num_patches = conv_shape[1] * conv_shape[2]
        return (input_shape[0], num_patches, self.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'stride': self.stride
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class PositionalEncoding(layers.Layer):
    """Learnable positional encoding for patches."""

    def __init__(self, max_patches: int, embed_dim: int, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_patches = max_patches
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # Learnable positional embeddings. Criados em build() e não em
        # __init__(): pesos criados no construtor escapam ao ciclo de vida da
        # camada (build/trainable_weights) e são uma fonte conhecida de
        # inconsistência de serialização no Keras 3.
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, self.max_patches, self.embed_dim),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]

        # Take only the needed positional embeddings
        pos_emb = self.pos_embedding[:, :seq_len, :]

        return inputs + tf.cast(pos_emb, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_patches': self.max_patches,
            'embed_dim': self.embed_dim
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class SpectrogramTransformerBlock(layers.Layer):
    """Transformer block optimized for spectrogram analysis.

    norm_style:
        'pre'  — pre-LN (ViT/AST real: LayerNorm ANTES da atenção/FFN, residual
                 puro). É o formato do paper e o único estável para 12 blocos
                 treinados do zero: em post-LN a magnitude dos residuais cresce
                 com a profundidade e o treino degrada lentamente até colapsar
                 (accuracy → chute aleatório), como observado no benchmark.
        'post' — comportamento legado (LayerNorm depois do residual). Mantido
                 como default APENAS para desserializar modelos .keras antigos
                 sem alterar sua saída; novos builds usam 'pre'.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout_rate: float = 0.1, norm_style: str = 'post', **kwargs):
        super(SpectrogramTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        if norm_style not in ('pre', 'post'):
            raise ValueError(f"norm_style inválido: {norm_style!r}")
        self.norm_style = norm_style

        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )

        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])

        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        # Sub-camadas criadas no __init__ precisam ser construídas aqui —
        # senão o Keras 3 marca o bloco como construído com estado pendente.
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        self.attention.build(input_shape, input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None):
        if self.norm_style == 'pre':
            # Pre-LN (ViT/AST): x = x + Attn(LN(x)); x = x + FFN(LN(x))
            attn_input = self.layernorm1(inputs)
            attn_output = self.attention(
                attn_input, attn_input, training=training)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = inputs + tf.cast(attn_output, inputs.dtype)

            ffn_output = self.ffn(self.layernorm2(out1), training=training)
            ffn_output = self.dropout2(ffn_output, training=training)
            return out1 + tf.cast(ffn_output, out1.dtype)

        # Post-LN legado (compat com modelos salvos)
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = tf.cast(attn_output, inputs.dtype)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = tf.cast(ffn_output, out1.dtype)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'norm_style': self.norm_style
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class SpectralAttentionPooling(layers.Layer):
    """Attention-based pooling specifically designed for spectral features."""

    def __init__(self, embed_dim: int, **kwargs):
        super(SpectralAttentionPooling, self).__init__(**kwargs)
        self.embed_dim = embed_dim

        # Attention mechanism for pooling
        self.attention_weights = layers.Dense(1, activation='tanh')

    def build(self, input_shape):
        self.attention_weights.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def call(self, inputs):
        # inputs shape: (batch, num_patches, embed_dim)

        # Calculate attention scores
        attention_scores = self.attention_weights(
            inputs)  # (batch, num_patches, 1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)

        # Apply attention weights
        attended_features = tf.reduce_sum(inputs * attention_weights, axis=1)

        return attended_features

    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim})
        return config


def create_spectrogram_transformer_model(
    input_shape: Tuple[int, ...],
    num_classes: int,
    patch_size: Tuple[int, int] = (16, 16),
    stride: Tuple[int, int] = (10, 10),
    embed_dim: int = 768,
    num_blocks: int = 12,
    num_heads: int = 12,
    ff_dim: int = 3072,
    # P1 — retreino obrigatório: o modelo colapsava val→teste (~25 pp) por
    # sobreajuste. Mais regularização (dropout 0.1→0.3, weight_decay 1e-5→1e-4)
    # e LR de pico menor (1e-4→5e-5) reduzem o gap de generalização. Combinado
    # com restauração obrigatória do melhor checkpoint e augmentation SNR.
    # AJUSTE 2026-07-14: mesmo após o fix de decay_steps o treino degradava
    # lentamente até chute aleatório (EER final ~51%). Dois ajustes:
    # (1) blocos agora são pre-LN (ViT/AST real; post-LN a 12 blocos do zero
    #     é instável) e (2) LR de pico 5e-5→1e-5 e weight_decay 1e-4→1e-5
    #     — 87M params do zero pedem passo menor. Sincronizado com registry.py
    #     e benchmarks/planning.py.
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-5,
    warmup_steps: int = 2000,
    decay_steps: int = 50000,
    weight_decay: float = 1e-5,
    alpha: float = 1e-7,
    clipnorm: float = 1.0,
    pretrained: bool = False,
    architecture: str = 'spectrogram_transformer'
) -> models.Model:
    """
    Create Audio Spectrogram Transformer (AST) model for audio deepfake detection.
    Topologically aligned with AST (Gong et al., 2021), trained from scratch.

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        patch_size: Size of patches for patch embedding (default: 16x16)
        stride: Stride for overlapping patches (default: 10x10)
        embed_dim: Embedding dimension (ViT-Base: 768)
        num_blocks: Number of transformer blocks (ViT-Base: 12)
        num_heads: Number of attention heads (ViT-Base: 12)
        ff_dim: Feed-forward dimension (ViT-Base: 3072)
        dropout_rate: Dropout rate
        learning_rate: Peak learning rate after warmup
        warmup_steps: Number of linear warmup optimizer steps
        decay_steps: Number of cosine decay optimizer steps
        weight_decay: AdamW weight decay
        alpha: Minimum learning rate used by the warmup/cosine schedule
        pretrained: Reserved metadata flag. This Keras implementation trains
            AST from scratch; passing True logs a warning and does not load
            AudioSet/ImageNet weights.
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    logger.info(
        f"Creating AST model with input_shape={input_shape}, num_classes={num_classes}")
    # `pretrained=True` transfere os pesos do AST refinado em AudioSet (ver
    # ast_pretrained.py). O caminho TF do `transformers` é inviável com Keras 3,
    # mas o checkpoint PyTorch é legível sem tocar em TF — lemos o state_dict e
    # escrevemos nas camadas Keras. A flag NUNCA vira no-op: se os pesos não
    # puderem ser obtidos, a construção FALHA em vez de rotular como
    # "pré-treinado" um modelo treinado do zero.

    # Input layer
    inputs = layers.Input(shape=input_shape, name='ast_input')

    # Preprocessing based on input type
    if is_raw_audio(input_shape):
        input_tensor = ensure_flat_input(inputs)

        # Front-end do paper: 128 bandas MEL, janela de 25 ms e hop de 10 ms
        # a 16 kHz (400 e 160 amostras), em escala log.
        # CORREÇÃO: antes usava `STFTLayer` com os DEFAULTS (janela 2048 =
        # 128 ms, hop 512 = 32 ms), magnitude LINEAR sem mel nem log, e
        # redimensionava para 128×128 — três desvios simultâneos do artigo,
        # incluindo a distorção da razão tempo×frequência pelo resize.
        x = LogMelSpectrogramLayer(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=128,
            name='log_mel_frontend',
        )(input_tensor)
        x = ExpandDimsLayer(axis=-1, name='add_channel')(x)
        # `LogMelSpectrogramLayer.compute_output_shape` devolve o nº de quadros
        # de forma estática, então a grade de patches abaixo é calculável.
        processed_height = x.shape[1]
        processed_width = 128
    else:
        # Preprocessing spectrogram input
        x = create_safe_spectrogram_layer(input_shape)(inputs)
        processed_height = max(64, input_shape[0])
        processed_width = max(64, input_shape[1])

    # Normalização de entrada do AST (média 0, desvio 0,5 — §2.1 do artigo),
    # calculada por amostra para não vazar estatística entre partições.
    x = ASTInputNormalization(name='ast_input_norm')(x)

    # Patches DIRETO no espectrograma, como no paper AST (Gong et al., 2021:
    # patches 16×16 com stride 10 sobre o espectrograma, SEM conv stem).
    # Antes havia um ConvolutionStemLayer (÷8 espacial) que reduzia (100, 80)
    # a ~12×10 → com patch 8×8/stride 6 sobravam 1×1 = **1 patch** e o
    # Transformer de 12 blocos atendia sobre 2 tokens (CLS+1) — degenerado.
    grid_h = x.shape[1] if x.shape[1] is not None else processed_height
    grid_w = x.shape[2] if x.shape[2] is not None else processed_width

    # Adapta patch/stride para entradas pequenas (garante uma grade real de
    # patches): encolhe à metade enquanto não couberem ≥2 patches por eixo.
    ph, pw = patch_size
    sh, sw = stride
    while ph > 2 and (grid_h - ph) // sh + 1 < 2:
        ph, sh = max(2, ph // 2), max(1, sh // 2)
    while pw > 2 and (grid_w - pw) // sw + 1 < 2:
        pw, sw = max(2, pw // 2), max(1, sw // 2)
    patch_size = (ph, pw)
    stride = (sh, sw)

    # num_patches = floor((input - patch) / stride) + 1 (padding='valid')
    num_patches_h = (grid_h - patch_size[0]) // stride[0] + 1
    num_patches_w = (grid_w - patch_size[1]) // stride[1] + 1
    num_patches = num_patches_h * num_patches_w
    if num_patches < 1:
        raise ValueError(
            f"AST: entrada {grid_h}x{grid_w} pequena demais para patches "
            f"{patch_size} com stride {stride}."
        )

    logger.info(
        f"Using {num_patches} patches ({num_patches_h}x{num_patches_w}) "
        f"patch={patch_size} stride={stride} (direto no espectrograma, sem stem)")

    # Patch embedding (with overlap) on stem output
    x = PatchEmbedding(patch_size, embed_dim, stride=stride, name='patch_embedding')(x)

    # Add class token (Standard ViT/AST)
    x = ClassTokenLayer(embed_dim, name='class_token_layer')(x)

    # Positional encoding (learned)
    x = PositionalEncoding(num_patches + 1, embed_dim, name='pos_encoding')(x)

    # Transformer blocks (Standard ViT-Base, pre-LN como no paper).
    for i in range(num_blocks):
        x = SpectrogramTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            norm_style='pre',
            name=f'ast_block_{i}'
        )(x)

    # Final LayerNorm before head (obrigatório com pre-LN)
    x = layers.LayerNormalization(epsilon=1e-6, name='final_norm')(x)

    # Extract class token for classification (Standard ViT/AST)
    class_token_output = x[:, 0, :]

    # Cabeça do paper: AST usa APENAS LayerNorm + camada linear sobre o CLS.
    # A cabeça anterior (2 blocos Dense 1024/256 com skips lineares, ~1M
    # params extras) não existe no paper e só ampliava o sobreajuste
    # (gap val→teste) sem ganho de representação.
    x = layers.Dropout(dropout_rate, name='ast_head_dropout')(class_token_output)

    # Output layer
    # dtype='float32': sob mixed_float16, softmax+crossentropy em float16
    # satura/perde precisão e pode colapsar o treino (rede "morta" após a
    # 1a epoca). Mesma correção já aplicada em AASIST/RawGAT-ST.
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output', dtype='float32')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(
            num_classes,
            activation='softmax',
            name='output',
            dtype='float32')(x)
        loss = 'sparse_categorical_crossentropy'

    # Create model. `architecture` nomeia o modelo (antes o parâmetro era
    # declarado e ignorado, e a variante lite saía com o mesmo nome da completa).
    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=architecture)

    if pretrained:
        # Transferência dos pesos AudioSet. Qualquer falha PROPAGA — a flag
        # não pode degradar silenciosamente para treino do zero.
        from app.domain.models.architectures.ast_pretrained import (
            load_ast_pretrained_weights,
        )

        transfer_info = load_ast_pretrained_weights(
            model,
            num_blocks=num_blocks,
            num_heads=num_heads,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            num_patches_grid=(num_patches_h, num_patches_w),
        )
        logger.info("AST pretrained: %s", transfer_info)

    # Sprint 2.2: WarmupCosineDecay default para Transformers.
    # Warmup linear estabiliza Self-Attention nas primeiras épocas
    # (gradientes grandes) e cosine decay melhora convergência final.
    # P1 — clipnorm=1.0 + warmup maior estabilizam a atenção e evitam o colapso
    # val→0.5 visto no benchmark (treino divergia após o pico do warmup).
    from app.domain.models.training.optimization import create_warmup_cosine_optimizer
    optimizer = create_warmup_cosine_optimizer(
        initial_learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        weight_decay=weight_decay,
        alpha=alpha,
        clipnorm=clipnorm,
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    logger.info(
        f"Spectrogram Transformer model created successfully with {model.count_params()} parameters "
        f"(WarmupCosineDecay: lr={learning_rate}, warmup={warmup_steps}, "
        f"decay={decay_steps}, weight_decay={weight_decay}, alpha={alpha}, "
        f"clipnorm={clipnorm})")
    return model


def create_lightweight_spectrogram_transformer(
    input_shape: Tuple[int, ...],
    num_classes: int,
    architecture: str = 'spectrogram_transformer_lite',
    **kwargs
) -> models.Model:
    """
    Create a lightweight version of Spectrogram Transformer for faster inference.

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    params = {
        "patch_size": (16, 16),  # Larger patches
        "embed_dim": 128,        # Smaller embedding
        "num_blocks": 4,         # Fewer blocks
        "num_heads": 4,          # Fewer heads
        "ff_dim": 256,           # Smaller FF dimension
        "dropout_rate": 0.1,
    }
    # Config da variante PREVALECE — ver nota em create_small_spectrogram_transformer.
    params.update({k: v for k, v in kwargs.items() if k not in params})
    return create_spectrogram_transformer_model(
        input_shape=input_shape,
        num_classes=num_classes,
        architecture=architecture,
        **params
    )


#: Configuração "small" (ViT-Small): 12 blocos, embed 384, 6 cabeças, FF 1536.
#: DESVIO DELIBERADO do ViT-Base do paper AST — existe porque o AST original é
#: inicializado com pesos ImageNet/AudioSet, e treinar 87M parâmetros do zero
#: sobre ~21k amostras foi justamente o que degradou o modelo até chute
#: aleatório no benchmark (EER ~51%). ~22M parâmetros é o regime compatível com
#: treino do zero. A variante padrão continua sendo o ViT-Base do artigo.
_AST_SMALL_PARAMS = {
    "patch_size": (16, 16),
    "stride": (10, 10),
    "embed_dim": 384,
    "num_blocks": 12,
    "num_heads": 6,
    "ff_dim": 1536,
    "dropout_rate": 0.2,
}


def create_small_spectrogram_transformer(
    input_shape: Tuple[int, ...],
    num_classes: int,
    architecture: str = 'spectrogram_transformer_small',
    **kwargs
) -> models.Model:
    """AST na escala ViT-Small, para treino do zero (sem pesos pré-treinados).

    A CONFIGURAÇÃO DA VARIANTE PREVALECE sobre kwargs: eles chegam aqui tanto de
    um override explícito quanto do `registry.default_params`, que descrevem o
    ViT-Base. Sem esta regra, pedir 'spectrogram_transformer_small' pelo
    registry/factory devolvia silenciosamente um ViT-Base de 85M parâmetros.
    """
    params = dict(_AST_SMALL_PARAMS)
    params.update({k: v for k, v in kwargs.items() if k not in params})
    return create_spectrogram_transformer_model(
        input_shape=input_shape,
        num_classes=num_classes,
        architecture=architecture,
        **params
    )


def create_model(input_shape: Tuple[int, ...], num_classes: int,
                 architecture: str = 'spectrogram_transformer',
                 **kwargs) -> models.Model:
    """
    Factory function to create Spectrogram Transformer models (for compatibility with existing code).

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture name

    Returns:
        Compiled Keras model
    """
    # Alias "default" -> AST topologicamente alinhado ao artigo, treinado do zero.
    if architecture == 'default':
        architecture = 'spectrogram_transformer'

    if architecture == 'spectrogram_transformer':
        return create_spectrogram_transformer_model(
            input_shape, num_classes, architecture=architecture, **kwargs)
    elif architecture == 'spectrogram_transformer_small':
        return create_small_spectrogram_transformer(
            input_shape, num_classes, architecture=architecture, **kwargs)
    elif architecture == 'spectrogram_transformer_lite':
        return create_lightweight_spectrogram_transformer(
            input_shape, num_classes, architecture=architecture, **kwargs)
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. Use "
            "'spectrogram_transformer' (ViT-Base do paper), "
            "'spectrogram_transformer_small' (ViT-Small, para treino do zero) "
            "ou 'spectrogram_transformer_lite'."
        )


# Register custom layers and functions for model loading
tf.keras.utils.get_custom_objects().update({
    'PatchEmbedding': PatchEmbedding,
    'ClassTokenLayer': ClassTokenLayer,
    'PositionalEncoding': PositionalEncoding,
    'SpectrogramTransformerBlock': SpectrogramTransformerBlock,
    'SpectralAttentionPooling': SpectralAttentionPooling,
    'ResizeLayer': ResizeLayer,
    'SafeSpectrogramReshapeLayer': SafeSpectrogramReshapeLayer,
    'STFTLayer': STFTLayer
})
