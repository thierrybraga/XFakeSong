"""RawNet2 Architecture Implementation

Paper-faithful implementation of RawNet2 for audio deepfake detection.
Operates directly on raw audio waveforms.

Reference: Jung et al., "Improved RawNet with Feature Map Scaling for
Text-Independent Speaker Verification using Raw Waveforms", 2020

Architecture:
1. SincNet front-end (learnable bandpass filters)
2. Residual blocks with Feature Map Scaling (FMS)
3. GRU for temporal modeling
4. Dense classifier
"""

# Third-party imports
import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

# CAMADAS PRÓPRIAS DO RawNet2 (separadas de layers.py em 2026-08-20).
#
# O RawNet2 tem front-end sinc PRÓPRIO (`RawNet2SincConv`), diferente do que
# AASIST e RawGAT-ST usam — mais um motivo para os módulos serem separados: uma
# correção no sinc de uma família não pode alcançar a outra.
from app.domain.models.architectures.rawnet2_layers import (
    RawNet2FeatureMapScaling,
    RawNet2MultiScaleConv1D,
    RawNet2ResidualBlock1D,
    RawNet2SincConv,
)

# Pré-processamento de sinal genérico: continua compartilhado.
from app.domain.models.architectures.layers import (
    AudioNormalizationLayer,
    PreEmphasisLayer,
)

# APENAS para desserializar artefatos gravados antes da separação de
# 2026-08-20. Não use em código novo: o RawNet2 constrói com as classes de
# `rawnet2_layers`. Ficam aqui porque os `.keras` antigos referenciam estes
# nomes e, com o banco sinc agora fixo, as classes novas não aceitam os pesos
# treináveis que aqueles artefatos gravaram.
from app.domain.models.architectures.layers import (
    FeatureMapScalingLayer as _LegacyFeatureMapScalingLayer,
    MultiScaleConv1DBlock as _LegacyMultiScaleConv1DBlock,
    ResidualBlock1D as _LegacyResidualBlock1D,
    SincNetLayer as _LegacySincNetLayer,
)
from app.utils.audio_utils import preprocess_legacy as preprocess

logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AudioResamplingLayer(layers.Layer):
    """LEGADO — não faz parte do grafo do RawNet2 atual.

    Nenhum builder deste módulo instancia esta camada; ela é mantida
    exclusivamente porque `detection/model_loader.py` a injeta em
    `custom_objects` para desserializar modelos antigos que a continham. Não
    use em arquiteturas novas (a reamostragem correta acontece no
    pré-processamento, fora do grafo).
    """

    def __init__(
        self,
        source_sample_rate: int = 16000,
        target_sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_sample_rate = int(source_sample_rate)
        self.target_sample_rate = int(target_sample_rate)

    def call(self, inputs):
        x = inputs
        squeezed = False
        if x.shape.rank == 3 and x.shape[-1] == 1:
            x = tf.squeeze(x, axis=-1)
            squeezed = True

        if self.source_sample_rate == self.target_sample_rate:
            y = x
        else:
            # `tf.signal.resample` NÃO existe — o branch antigo quebraria com
            # AttributeError na 1ª execução com taxas distintas. Reamostragem
            # por interpolação linear via tf.image.resize (1D como imagem Nx1).
            in_len = tf.shape(x)[-1]
            ratio = tf.cast(self.target_sample_rate, tf.float32) / tf.cast(
                self.source_sample_rate, tf.float32
            )
            target_len = tf.cast(tf.round(tf.cast(in_len, tf.float32) * ratio), tf.int32)
            img = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=-1)  # (B,T,1,1)
            img = tf.image.resize(img, [target_len, 1], method="bilinear")
            y = tf.squeeze(img, axis=[-1, -2])

        if squeezed:
            y = tf.expand_dims(y, axis=-1)
        return y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "source_sample_rate": self.source_sample_rate,
                "target_sample_rate": self.target_sample_rate,
            }
        )
        return config


def _create_rawnet2_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    sinc_filters: int = 128,
    sinc_kernel_size: int = 1024,
    res_filters: list = None,
    gru_units: int = 1024,  # paridade com o paper (Improved RawNet usa GRU 1024)
    gru_layers: int = 1,
    dense_units: int = 1024,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    clipnorm: float = 1.0,
    architecture: str = 'rawnet2'
) -> models.Model:
    """Criar modelo RawNet2 fiel ao paper 'Improved RawNet'.

    Args:
        input_shape: Formato da entrada (samples,)
        num_classes: Número de classes (1 para detecção binária)
        sinc_filters: Filtros na camada SincNet
        sinc_kernel_size: Tamanho do kernel na SincNet
        res_filters: Lista com número de filtros para cada bloco residual
        gru_units: Número de unidades na camada GRU
        gru_layers: Nº de camadas GRU empilhadas (1 = Improved RawNet/SV;
            3 = baseline anti-spoofing do ASVspoof 2021)
        dense_units: Número de unidades na camada densa
        dropout_rate: Taxa de dropout
        learning_rate: LR do AdamW (baseline do paper: 1e-4)
        weight_decay: weight decay desacoplado (baseline do paper: 1e-4)
        clipnorm: clipping global de gradiente (0/None desliga)
        architecture: Nome da arquitetura

    Returns:
        Modelo Keras compilado
    """
    if res_filters is None:
        res_filters = [128, 128, 256, 256, 256, 256]

    # Input layer
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # 1. Pré-processamento
    x = PreEmphasisLayer(name='pre_emphasis')(inputs)
    x = AudioNormalizationLayer(name='audio_normalization')(x)
    if len(x.shape) == 2:
        x = layers.Reshape((-1, 1))(x)

    # 2. SincNet Front-end
    x = RawNet2SincConv(filters=sinc_filters, kernel_size=sinc_kernel_size, name='sincnet')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)

    # 3. Residual Blocks + FMS
    # Paridade com o paper/implementação oficial (baseline ASVspoof 2021):
    # MaxPool(3) após CADA bloco residual e FMS na forma mul+add (x*y + y).
    # AJUSTE 2026-07-14: antes o pooling só existia após os blocos 2 e 4
    # (i in [1, 3]) → a GRU recebia ~590 passos temporais em vez de ~7
    # (com recorte de 1 s), o que explica tanto o treino lento (~130 min)
    # quanto o EER de 32,7% limpo vs baseline 2,89%: GRU(1024) não aprende
    # dependências sobre sequências tão longas de features quase-cruas.
    for i, filters in enumerate(res_filters):
        x = RawNet2ResidualBlock1D(out_channels=filters, name=f'res_block_{i + 1}')(x)
        x = RawNet2FeatureMapScaling(
            scale_mode='mul_add', name=f'fms_{i + 1}')(x)
        x = layers.MaxPooling1D(pool_size=3, name=f'res_pool_{i + 1}')(x)

    # 4. Temporal Modeling (GRU)
    # `gru_layers` segue o paper de cada variante: 1 camada GRU(1024) no
    # "Improved RawNet" (verificação de locutor, Jung et al. 2020) e 3 camadas
    # no baseline anti-spoofing do ASVspoof 2021 (Tak et al.).
    # Em Keras 3 o layers.GRU usa cuDNN automaticamente quando as condições
    # são atendidas (recurrent_dropout=0, ativações padrão, sem mask).
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.3)(x)

    for layer_index in range(int(gru_layers)):
        is_last = layer_index == int(gru_layers) - 1
        x = layers.GRU(
            units=gru_units, return_sequences=not is_last,
            dropout=dropout_rate, recurrent_dropout=0.0,
            name=f'gru_{layer_index + 1}'
        )(x)

    # 5. Classification Head
    # Dense head with explicit float32 output to be safe under mixed precision.
    x = layers.Dense(dense_units, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate, name='fc1_drop')(x)
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='output')(x)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')(x)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # Criar modelo
    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    # Baseline oficial do RawNet2 anti-spoofing: Adam com lr=1e-4 e
    # weight decay=1e-4. Aqui usamos AdamW (decaimento DESACOPLADO, a forma
    # correta do mesmo hiperparâmetro) + clipnorm, como nas demais
    # arquiteturas do projeto. Antes o otimizador era Adam(1e-4) HARDCODED,
    # sem regularização de pesos e sem clip — e o `learning_rate` do plano de
    # benchmark não tinha como chegar até aqui.
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        global_clipnorm=float(clipnorm) if clipnorm else None,
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    logger.info(
        f"Modelo {architecture} criado: gru_units={gru_units}, "
        f"sinc_filters={sinc_filters}, lr={learning_rate}, "
        f"weight_decay={weight_decay}, params={model.count_params()}"
    )
    return model


def create_lightweight_rawnet2(
    input_shape: Tuple[int, ...],
    num_classes: int = 1,
    architecture: str = 'rawnet2_lite'
) -> models.Model:
    """Criar versão leve do RawNet2."""
    return _create_rawnet2_model(
        input_shape=input_shape,
        num_classes=num_classes,
        sinc_filters=16,
        sinc_kernel_size=512,
        res_filters=[16, 16, 64, 64],
        gru_units=64,
        dense_units=32,
        dropout_rate=0.3,
        architecture=architecture
    )


#: Baseline anti-spoofing do ASVspoof 2021 (Tak et al.): 20 filtros Sinc,
#: blocos residuais [20,20,128,128,128,128] e GRU(1024) de 3 camadas. É ESTE o
#: "RawNet2" com que a literatura de anti-spoofing compara EER — e é uma
#: configuração DIFERENTE do "Improved RawNet" de verificação de locutor
#: (Jung et al., 2020), que usa 128 filtros Sinc e canais 128/256 e é o que a
#: variante `rawnet2` implementa. Declare qual foi usada ao reportar resultados.
_RAWNET2_ANTISPOOFING_PARAMS = {
    "sinc_filters": 20,
    "sinc_kernel_size": 1024,
    "res_filters": [20, 20, 128, 128, 128, 128],
    "gru_units": 1024,
    "gru_layers": 3,
    "dense_units": 1024,
}


def create_model(input_shape: Tuple[int, ...], num_classes: int = 1,
                 architecture: str = 'rawnet2', **kwargs) -> models.Model:
    """Função principal para criar modelos RawNet2.

    Variantes:
        'rawnet2': "Improved RawNet" (Jung et al., 2020 — verificação de
            locutor): Sinc 128, blocos [128,128,256,256,256,256], 1×GRU(1024).
        'rawnet2_antispoofing': baseline do ASVspoof 2021 (Tak et al.):
            Sinc 20, blocos [20,20,128,128,128,128], 3×GRU(1024).
        'rawnet2_lite': versão reduzida para inferência rápida.

    Args:
        input_shape: Formato da entrada (samples,)
        num_classes: Número de classes
        architecture: Tipo de arquitetura
        **kwargs: Parâmetros adicionais para _create_rawnet2_model

    Returns:
        Modelo Keras compilado
    """
    if architecture == 'default':
        architecture = 'rawnet2'

    if architecture == 'rawnet2_lite':
        return create_lightweight_rawnet2(input_shape, num_classes, architecture)
    if architecture == 'rawnet2_antispoofing':
        # A CONFIGURAÇÃO DA VARIANTE PREVALECE sobre kwargs: eles chegam tanto
        # de um override explícito quanto do `registry.default_params`, que
        # descrevem a variante de VERIFICAÇÃO DE LOCUTOR (Sinc 128, 1×GRU). Sem
        # esta regra, pedir 'rawnet2_antispoofing' pelo registry/factory
        # devolvia silenciosamente o RawNet2 de SV com o nome do baseline.
        params = dict(_RAWNET2_ANTISPOOFING_PARAMS)
        params.update({
            k: v for k, v in kwargs.items() if k not in params
        })
        return _create_rawnet2_model(
            input_shape, num_classes, architecture=architecture, **params)
    return _create_rawnet2_model(
        input_shape, num_classes, architecture=architecture, **kwargs)


# Registrar objetos personalizados no Keras.
# `preprocess` (= audio_utils.preprocess_legacy) é registrado com chave
# QUALIFICADA: a chave curta 'preprocess' era usada também por sonic_sleuth.py
# e wavlm.py com funções DIFERENTES, e a última importação vencia — um modelo
# salvo podia ser recarregado com o pré-processamento de outra arquitetura.
# As chaves ANTIGAS continuam apontando para as classes novas: um artefato
# .keras gravado antes da separação de 2026-08-20 referencia `SincNetLayer` pelo
# nome, e sem a entrada ele não desserializa. As chaves próprias entram ao lado,
# para os artefatos gerados a partir de agora.
tf.keras.utils.get_custom_objects().update({
    'AudioResamplingLayer': AudioResamplingLayer,
    'AudioNormalizationLayer': AudioNormalizationLayer,
    'PreEmphasisLayer': PreEmphasisLayer,
    'XFakeSong>preprocess_legacy': preprocess,
    # CHAVES LEGADAS -> CLASSES LEGADAS, não às novas.
    #
    # Mapear a chave antiga para a classe nova parecia natural, e QUEBRA: o
    # `RawNet2SincConv` nasce com banco FIXO (`trainable_filters=False`) desde
    # 2026-08-20, enquanto o artefato antigo gravou as frequências de corte
    # como pesos TREINÁVEIS. Keras separa os dois grupos (`trainable_variables`
    # x `non_trainable_variables`), então a restauração não encontra onde pôr
    # os pesos salvos — medido em `bench_rawnet2.keras`: "A total of 32 objects
    # could not be loaded". Cada nome carrega a semântica da época em que o
    # artefato foi gravado.
    'MultiScaleConv1DBlock': _LegacyMultiScaleConv1DBlock,
    'SincNetLayer': _LegacySincNetLayer,
    'ResidualBlock1D': _LegacyResidualBlock1D,
    'FeatureMapScalingLayer': _LegacyFeatureMapScalingLayer,
    # nomes próprios do RawNet2
    'RawNet2MultiScaleConv1D': RawNet2MultiScaleConv1D,
    'RawNet2SincConv': RawNet2SincConv,
    'RawNet2ResidualBlock1D': RawNet2ResidualBlock1D,
    'RawNet2FeatureMapScaling': RawNet2FeatureMapScaling,
})
