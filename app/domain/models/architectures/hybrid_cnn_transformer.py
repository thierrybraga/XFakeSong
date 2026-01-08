"""Hybrid CNN-Transformer Architecture Implementation

Este módulo implementa uma arquitetura híbrida que combina CNNs para extração
de features espaciais com Transformers para modelagem temporal, especificamente
desenhada para detecção de deepfakes em áudio.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Optional, Dict, Any
import logging
from app.domain.models.architectures.layers import STFTLayer, ExpandDimsLayer

logger = logging.getLogger(__name__)


class ResidualBlock(layers.Layer):
    """Bloco residual com Squeeze-and-Excitation."""

    def __init__(self, filters: int, dropout_rate: float = 0.1,
                 use_se: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.use_se = use_se

        # Camadas convolucionais
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()

        # Squeeze-and-Excitation
        if use_se:
            self.se_global_pool = layers.GlobalAveragePooling2D()
            self.se_dense1 = layers.Dense(filters // 16, activation='relu')
            self.se_dense2 = layers.Dense(filters, activation='sigmoid')
            self.se_reshape = layers.Reshape((1, 1, filters))

        # Dropout
        self.dropout = layers.Dropout(dropout_rate)

        # Shortcut connection
        self.shortcut = None
        self.shortcut_bn = None

    def build(self, input_shape):
        super().build(input_shape)

        # Criar shortcut se necessário
        if input_shape[-1] != self.filters:
            self.shortcut = layers.Conv2D(self.filters, 1, padding='same')
            self.shortcut_bn = layers.BatchNormalization()

    def call(self, inputs, training=None):
        # Primeira convolução
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.swish(x)

        # Segunda convolução
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Squeeze-and-Excitation
        if self.use_se:
            se = self.se_global_pool(x)
            se = self.se_dense1(se)
            se = self.se_dense2(se)
            se = self.se_reshape(se)
            x = layers.Multiply()([x, se])

        # Dropout
        x = self.dropout(x, training=training)

        # Shortcut connection
        shortcut = inputs
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)

        # Residual connection
        x = layers.Add()([x, shortcut])
        x = tf.nn.swish(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'dropout_rate': self.dropout_rate,
            'use_se': self.use_se
        })
        return config


class MultiScaleAttention(layers.Layer):
    """Atenção multi-escala para capturar features em diferentes resoluções."""

    def __init__(self, filters: int, scales: list = [1, 3, 5], **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.scales = scales

        # Convoluções para cada escala
        self.scale_convs = []
        for scale in scales:
            conv = layers.Conv2D(
                filters // len(scales),
                kernel_size=scale,
                padding='same',
                activation='relu'
            )
            self.scale_convs.append(conv)

        # Atenção global
        self.global_pool = layers.GlobalAveragePooling2D()
        self.attention_dense = layers.Dense(filters, activation='sigmoid')

        # Convolução final
        self.final_conv = layers.Conv2D(filters, 1, activation='relu')

    def call(self, inputs):
        # Processar cada escala
        scale_outputs = []
        for conv in self.scale_convs:
            scale_out = conv(inputs)
            scale_outputs.append(scale_out)

        # Concatenar saídas das escalas
        multi_scale = layers.Concatenate(axis=-1)(scale_outputs)

        # Atenção global
        attention_weights = self.global_pool(multi_scale)
        attention_weights = self.attention_dense(attention_weights)
        attention_weights = layers.Reshape(
            (1, 1, self.filters))(attention_weights)

        # Aplicar atenção
        multi_scale = self.final_conv(multi_scale)
        attended = layers.Multiply()([multi_scale, attention_weights])

        return attended

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'scales': self.scales
        })
        return config


class TemporalAttention(layers.Layer):
    """Atenção temporal para sequências de áudio."""

    def __init__(self, units: int, num_heads: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads

        # Multi-head attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units // num_heads,
            dropout=0.1
        )

        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Feed forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(units * 4, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(units)
        ])

        # Dropout
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, inputs, training=None):
        # Multi-head attention
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed forward network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads
        })
        return config


def create_hybrid_cnn_transformer_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    base_filters: int = 64,
    num_residual_blocks: int = 3,
    num_transformer_layers: int = 2,
    attention_heads: int = 8,
    dropout_rate: float = 0.1,
    architecture: str = 'hybrid_cnn_transformer'
) -> models.Model:
    """
    Cria um modelo híbrido CNN-Transformer do zero.

    Args:
        input_shape: Formato da entrada (height, width, channels)
        num_classes: Número de classes (1 para classificação binária)
        base_filters: Número base de filtros para CNN
        num_residual_blocks: Número de blocos residuais
        num_transformer_layers: Número de camadas Transformer
        attention_heads: Número de cabeças de atenção
        dropout_rate: Taxa de dropout
        architecture: Nome da arquitetura

    Returns:
        Modelo Keras compilado
    """
    logger.info(
        f"Criando modelo híbrido CNN-Transformer com input_shape={input_shape}"
    )

    # Entrada
    inputs = layers.Input(shape=input_shape, name='hybrid_input')

    # Tratar diferentes formatos de entrada
    if len(input_shape) == 1 or (
        len(input_shape) == 2 and input_shape[-1] == 1
    ):
        # Converter áudio raw para espectrograma usando STFT
        input_tensor = inputs
        if len(input_shape) == 2:
            # Se for (samples, 1), remover dimensão extra para STFT funcionar
            # corretamente
            input_tensor = layers.Reshape((input_shape[0],))(inputs)

        x = STFTLayer(
            frame_length=512, frame_step=256, fft_length=512,
            name='stft_layer'
        )(input_tensor)
        x = ExpandDimsLayer(axis=-1, name='expand_dims')(x)
    elif len(input_shape) == 2:
        # Entrada 2D (time, features) - adicionar dimensão de canal
        x = ExpandDimsLayer(axis=-1, name='expand_dims_2d')(inputs)
    else:
        # Entrada já é 3D (time, features, channels)
        x = inputs

    # Branch CNN
    cnn_branch = x
    logger.info(f"Shape after input processing: {cnn_branch.shape}")

    # Camadas convolucionais iniciais
    cnn_branch = layers.Conv2D(
        base_filters,
        3,
        activation='relu',
        padding='same')(cnn_branch)
    cnn_branch = layers.BatchNormalization()(cnn_branch)
    cnn_branch = layers.MaxPooling2D(2)(cnn_branch)
    logger.info(f"Shape after initial conv/pool: {cnn_branch.shape}")

    # Blocos residuais
    for i in range(num_residual_blocks):
        filters = base_filters * (2 ** i)
        cnn_branch = ResidualBlock(
            filters=filters,
            dropout_rate=dropout_rate
        )(cnn_branch)

        # MaxPooling entre blocos (exceto o último)
        if i < num_residual_blocks - 1:
            cnn_branch = layers.MaxPooling2D(2)(cnn_branch)

        logger.info(f"Shape after residual block {i}: {cnn_branch.shape}")

    # Atenção multi-escala
    cnn_branch = MultiScaleAttention(filters=cnn_branch.shape[-1])(cnn_branch)

    # Converter para sequência para o Transformer
    cnn_shape = cnn_branch.shape
    sequence_length = cnn_shape[1] * cnn_shape[2]
    feature_dim = cnn_shape[3]

    cnn_flattened = layers.Reshape((sequence_length, feature_dim))(cnn_branch)

    # Branch Transformer
    transformer_branch = cnn_flattened

    # Camadas de atenção temporal
    for i in range(num_transformer_layers):
        transformer_branch = TemporalAttention(
            units=feature_dim,
            num_heads=attention_heads
        )(transformer_branch)

    # Pooling e combinação
    cnn_pooled = layers.GlobalAveragePooling2D()(cnn_branch)
    transformer_pooled = layers.GlobalAveragePooling1D()(transformer_branch)

    # Combinar branches
    combined = layers.Concatenate()([cnn_pooled, transformer_pooled])

    # Camadas finais
    x = layers.Dense(512, activation='relu')(combined)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Camada de saída
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    else:
        outputs = layers.Dense(
            num_classes,
            activation='softmax',
            name='output')(x)

    # Criar modelo
    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    logger.info(f"Modelo híbrido criado com {model.count_params()} parâmetros")

    return model


def create_lightweight_hybrid_cnn_transformer(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    architecture: str = 'hybrid_cnn_transformer_lite'
) -> models.Model:
    """
    Cria uma versão leve do modelo híbrido para inferência rápida.

    Args:
        input_shape: Formato da entrada
        num_classes: Número de classes
        architecture: Nome da arquitetura

    Returns:
        Modelo Keras compilado
    """
    return create_hybrid_cnn_transformer_model(
        input_shape=input_shape,
        num_classes=num_classes,
        base_filters=32,
        num_residual_blocks=2,
        num_transformer_layers=1,
        attention_heads=4,
        dropout_rate=0.15,
        architecture=architecture
    )


def create_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    architecture: str = 'hybrid_cnn_transformer'
) -> models.Model:
    """
    Função factory para criar modelos híbridos (compatibilidade com código existente).

    Args:
        input_shape: Formato da entrada
        num_classes: Número de classes
        architecture: Nome da arquitetura

    Returns:
        Modelo Keras compilado
    """
    if architecture == 'hybrid_cnn_transformer':
        return create_hybrid_cnn_transformer_model(
            input_shape, num_classes, architecture=architecture)
    elif architecture == 'hybrid_cnn_transformer_lite':
        return create_lightweight_hybrid_cnn_transformer(
            input_shape, num_classes, architecture=architecture)
    else:
        raise ValueError(
            f"Arquitetura não suportada: {architecture}. Use 'hybrid_cnn_transformer' ou 'hybrid_cnn_transformer_lite'")


# Registrar camadas customizadas para carregamento de modelo
tf.keras.utils.get_custom_objects().update({
    'ResidualBlock': ResidualBlock,
    'MultiScaleAttention': MultiScaleAttention,
    'TemporalAttention': TemporalAttention,
    'STFTLayer': STFTLayer,
    'ExpandDimsLayer': ExpandDimsLayer
})

# Exportar principais funções
__all__ = [
    'create_hybrid_cnn_transformer_model',
    'create_lightweight_hybrid_cnn_transformer',
    'create_model',
    'ResidualBlock',
    'MultiScaleAttention',
    'TemporalAttention'
]
