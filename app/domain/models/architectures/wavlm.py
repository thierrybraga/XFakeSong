"""Implementação da arquitetura WavLM para detecção de deepfakes.

Esta implementação segue uma arquitetura de dois estágios:
1. Extrator de características: Modelo WavLM pré-treinado (congelado)
2. Classificador: MLP para classificação binária (real vs deepfake)
"""

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from app.utils.audio_utils import normalize_audio
from app.domain.models.architectures.layers import (
    AttentionPoolingLayer,
    create_classification_head,
)

# Convenção do projeto: logger de módulo, SEM logging.basicConfig — configurar
# o root logger no import contamina qualquer processo que importe este módulo
# (inclusive CLIs e o servidor da API).
logger = logging.getLogger(__name__)

# O `transformers` não fornece WavLM em TensorFlow (`TFWavLMModel` não existe;
# e seus modelos TF nem importam com Keras 3). A solução adotada NÃO é o
# fallback: `ssl_backbone.PretrainedSSLBackbone` lê o checkpoint **PyTorch** e
# reimplementa o forward em Keras, com os pesos congelados — incluindo o viés
# posicional relativo com gating, que é a contribuição do artigo. O extrator
# CNN-1D abaixo permanece só para quando o checkpoint não está acessível.
HF_AVAILABLE = False
TFWavLMModel = None


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class WavLMFeatureExtractor(layers.Layer):
    """Extrator de características usando modelo WavLM pré-treinado."""

    def __init__(self, model_name: str = "microsoft/wavlm-base",
                 freeze_weights: bool = True, n_trainable_layers: int = 0,
                 feature_dim: int | None = None, **kwargs):
        super(WavLMFeatureExtractor, self).__init__(**kwargs)
        self.model_name = model_name
        self.freeze_weights = freeze_weights
        # Fine-tuning parcial: nº de camadas do encoder a descongelar (do topo).
        # >0 ativa o fine-tune recomendado (Tak et al. 2022); 0 = congelado.
        self.n_trainable_layers = int(n_trainable_layers)
        # `model_name` NÃO carrega pesos aqui (não existe WavLM em TF): ele só
        # dimensiona o extrator de fallback. Explicitado no log para não passar
        # a impressão de que um checkpoint do HuggingFace foi baixado.
        self.feature_dim = (
            int(feature_dim) if feature_dim is not None
            else 768 if "base" in model_name else 1024
        )
        logger.info(
            "WavLMFeatureExtractor: '%s' NÃO é carregado (WavLM não existe em "
            "TensorFlow); a string só define a largura do fallback CNN-1D "
            "(feature_dim=%d).", model_name, self.feature_dim,
        )

        # Não existe TFWavLMModel: o caminho é SEMPRE o extrator simplificado.
        self._use_simplified = True
        # Fallback explícito: aborta em modo estrito (XFAKE_STRICT_SSL) para
        # não comprometer o benchmark com um backbone que não é o WavLM real.
        from app.domain.models.architectures.ssl_utils import strict_ssl_guard
        strict_ssl_guard("WavLM")
        # Implementação simplificada usando CNN 1D
        self._build_simplified_extractor()

    def _build_simplified_extractor(self):
        """Constrói um extrator simplificado usando CNN 1D."""
        self.conv_layers = [
            layers.Conv1D(64, 10, strides=5, activation='relu', name='conv1'),
            layers.BatchNormalization(name='bn1'),
            layers.Conv1D(128, 8, strides=4, activation='relu', name='conv2'),
            layers.BatchNormalization(name='bn2'),
            layers.Conv1D(256, 4, strides=2, activation='relu', name='conv3'),
            layers.BatchNormalization(name='bn3'),
            layers.Conv1D(512, 4, strides=2, activation='relu', name='conv4'),
            layers.BatchNormalization(name='bn4'),
            layers.Conv1D(
                self.feature_dim,
                4,
                strides=2,
                activation='relu',
                name='conv5'),
            layers.BatchNormalization(name='bn5')
        ]

    def build(self, input_shape):
        """Inicializa subcamadas para serialização/reload estáveis."""
        x_shape = tf.TensorShape(input_shape)
        if x_shape.rank == 3:
            current_shape = x_shape
        elif x_shape.rank == 2:
            current_shape = x_shape.concatenate([1])
        else:
            current_shape = tf.TensorShape([None, None, 1])

        for layer in self.conv_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass do extrator de características (CNN-1D simplificada)."""
        x = inputs

        # Processar entrada baseado na dimensionalidade
        if len(x.shape) == 3:  # (batch, time, freq)
            batch_size = tf.shape(x)[0]
            x = tf.reshape(x, [batch_size, -1])
            x = tf.expand_dims(x, axis=-1)
        elif len(x.shape) == 2:  # (batch, features)
            x = tf.expand_dims(x, axis=-1)

        # Aplicar camadas convolucionais
        for layer in self.conv_layers:
            x = layer(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'freeze_weights': self.freeze_weights,
            'n_trainable_layers': self.n_trainable_layers,
            'feature_dim': self.feature_dim
        })
        return config





def preprocess(audio_data: np.ndarray, target_sr: int = 16000) -> np.ndarray:
    """Pré-processamento de áudio para WavLM.

    Args:
        audio_data: Array de áudio
        target_sr: Taxa de amostragem alvo (16kHz para WavLM)

    Returns:
        Áudio pré-processado

    Nota: esta função NÃO reamostra — o WavLM exige 16 kHz e o áudio deve
    chegar já reamostrado pelo pipeline de features. ``target_sr`` diferente de
    16000 é sinalizado em vez de ser ignorado em silêncio, como acontecia antes.
    """
    if int(target_sr) != 16000:
        logger.warning(
            "WavLM.preprocess: target_sr=%s ignorado — o backbone exige 16 kHz "
            "e esta função não reamostra. Reamostre no pipeline de features "
            "(app/domain/features/).", target_sr,
        )

    # Normalização usando utilitário
    audio_data = normalize_audio(audio_data)

    # Clipping para evitar valores extremos
    audio_data = np.clip(audio_data, -3.0, 3.0)

    return audio_data


def _create_wavlm_model(input_shape: Tuple[int, ...],
                        num_classes: int = 1,
                        architecture: str = 'wavlm',
                        wavlm_model: str = "microsoft/wavlm-base",
                        freeze_wavlm: bool = True,
                        n_trainable_layers: int = 3,
                        backend: str = "conv",
                        classifier_units: list = None,
                        dropout_rate: float = 0.3) -> models.Model:
    """Cria modelo WavLM completo fiel ao paper.

    Args:
        input_shape: Formato da entrada (samples,)
        num_classes: Número de classes (1 para classificação binária)
        architecture: Nome da arquitetura
        wavlm_model: Nome do modelo WavLM pré-treinado
        freeze_wavlm: Se deve congelar pesos do WavLM
        classifier_units: Unidades das camadas do classificador
        dropout_rate: Taxa de dropout

    Returns:
        Modelo Keras compilado
    """
    if classifier_units is None:
        classifier_units = [1024, 512, 256]

    logger.info(
        f"Creating WavLM model with input_shape={input_shape}, num_classes={num_classes}")

    # 1. Entrada (Raw Audio)
    inputs = layers.Input(shape=input_shape, name='audio_input')

    # 2. Extrator de características WavLM (Self-Supervised).
    # BACKBONE PRÉ-TREINADO REAL (Chen et al., 2022), congelado: os pesos são
    # lidos do checkpoint PyTorch e portados para Keras — inclusive o viés
    # posicional relativo COM GATING, que é a contribuição do artigo. Só a soma
    # ponderada das camadas e a cabeça treinam. O extrator CNN-1D do zero
    # permanece apenas como fallback quando o checkpoint não está acessível.
    using_pretrained = False
    backbone_info = None
    try:
        from app.domain.models.architectures.ssl_utils import (
            build_pretrained_ssl_features,
        )

        x, backbone_info = build_pretrained_ssl_features(
            inputs, family="wavlm", checkpoint=wavlm_model, name="wavlm"
        )
        using_pretrained = True
    except Exception as exc:  # noqa: BLE001
        from app.domain.models.architectures.ssl_utils import strict_ssl_guard

        logger.warning(
            "WavLM: backbone pré-treinado indisponível (%s). Caindo no "
            "extrator simplificado.", exc,
        )
        strict_ssl_guard("WavLM")
        feature_extractor = WavLMFeatureExtractor(
            model_name=wavlm_model,
            freeze_weights=freeze_wavlm,
            n_trainable_layers=n_trainable_layers,
            name='wavlm_feature_extractor'
        )
        x = feature_extractor(inputs)

    if using_pretrained and n_trainable_layers:
        logger.warning(
            "WavLM: n_trainable_layers=%s ignorado — este port mantém o "
            "backbone INTEIRAMENTE congelado (só cabeça + pesos de camada "
            "treinam), que é a receita pedida para uso downstream.",
            n_trainable_layers,
        )

    if backend == "aasist":
        # Back-end de grafo AASIST (receita SOTA: WavLM → grafo espectro-temporal)
        from app.domain.models.architectures.ssl_utils import (
            build_ssl_aasist_backend,
        )
        pooled = build_ssl_aasist_backend(
            x, dropout_rate=dropout_rate, name="wavlm_aasist"
        )
    else:
        # Back-end raso com projeção temporal. Evita Conv1D grande sobre
        # features SSL/fallback, que no CUDA pode escolher kernels com workspace
        # de dezenas de GB mesmo com batch=1.
        projected = layers.Dense(
            256, activation='relu', name='temporal_projection'
        )(x)
        projected = layers.LayerNormalization(name='temporal_projection_norm')(
            projected
        )
        pooled = AttentionPoolingLayer(name='attention_pool')(projected)

    # Classification Head
    outputs, loss = create_classification_head(
        pooled,
        num_classes,
        dropout_rate=dropout_rate,
        hidden_dims=classifier_units
    )

    # Criar modelo
    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f'wavlm_{architecture}')

    # Backbone CONGELADO (pré-treinado ou fallback do zero) → só a cabeça
    # treina, então 1e-4 é o LR adequado. Não há ramo de fine-tuning parcial:
    # este port mantém o backbone inteiramente congelado por construção.
    lr = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    frozen = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    logger.info(
        "WavLM model %s criado (lr=%s, backbone=%s, params treináveis=%d, "
        "congelados=%d)", architecture, lr,
        backbone_info["checkpoint"] if using_pretrained else "fallback CNN-1D",
        trainable, frozen,
    )
    return model


def create_model(input_shape: Tuple[int, ...], num_classes: int = 1,
                 architecture: str = 'wavlm', **kwargs) -> models.Model:
    """Função principal para criar modelos WavLM.

    Args:
        input_shape: Formato da entrada
        num_classes: Número de classes
        architecture: Tipo de arquitetura ('wavlm' ou 'wavlm_lite')
        **kwargs: Parâmetros adicionais

    Returns:
        Modelo Keras compilado
    """
    # Variante 'wavlm_aasist' → back-end de grafo AASIST (receita SOTA).
    if "aasist" in architecture and "backend" not in kwargs:
        kwargs["backend"] = "aasist"

    if architecture == 'wavlm_lite':
        return _create_wavlm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            architecture=architecture,
            classifier_units=[256, 128],
            dropout_rate=0.2,
            **kwargs
        )
    else:
        return _create_wavlm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            architecture=architecture,
            **kwargs
        )


# Registrar objetos personalizados no Keras.
# `preprocess` com chave QUALIFICADA — ver nota em rawnet2.py sobre a colisão
# da chave global 'preprocess' entre arquiteturas.
tf.keras.utils.get_custom_objects().update({
    'WavLMFeatureExtractor': WavLMFeatureExtractor,
    'XFakeSong>WavLMFeatureExtractor': WavLMFeatureExtractor,
    'XFakeSong>wavlm_preprocess': preprocess,
})
