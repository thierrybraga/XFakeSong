"""Utilitários para controle de fine-tuning de backbones SSL (WavLM/HuBERT).

Implementa o **fine-tuning parcial** recomendado pela literatura recente de
anti-spoofing (Tak et al., 2022, "…using wav2vec 2.0 and data augmentation"):
descongelar apenas as últimas N camadas do encoder Transformer do front-end
SSL — o que gera o maior ganho de EER — mantendo o extrator convolucional e as
camadas iniciais congeladas (evita overfitting e reduz custo de memória).

Funções best-effort: como a estrutura interna dos modelos HuggingFace varia,
a localização das camadas do encoder é tolerante a falhas e degrada para um
modo seguro (congelar tudo) se a estrutura não for reconhecida.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "find_encoder_layers",
    "set_ssl_backbone_trainability",
    "build_ssl_aasist_backend",
    "strict_ssl_guard",
]


def strict_ssl_guard(model_name: str) -> None:
    """Falha alto quando um modelo SSL cairia no fallback simplificado.

    Em modo estrito (`XFAKE_STRICT_SSL=1`), levanta RuntimeError em vez de usar
    silenciosamente o backbone CNN-1D treinado do zero — protege a integridade do
    benchmark, que NÃO deve comparar números do fallback com os SSL reais da
    literatura. Sem a flag, apenas registra um aviso explícito (comportamento
    padrão preservado). Ver docs/08_ARQUITETURAS.md (caveat SSL).
    """
    msg = (
        f"[SSL] {model_name}: backbone real indisponível no caminho TensorFlow — "
        f"usaria fallback CNN-1D treinado do zero."
    )
    if os.getenv("XFAKE_STRICT_SSL", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            msg + " XFAKE_STRICT_SSL está ligado: abortando para não comprometer o "
            "benchmark. Remova o modelo do preset, use o caminho PyTorch "
            "(*_original.pt) ou desligue XFAKE_STRICT_SSL para permitir o fallback."
        )
    logger.warning(msg + " (fallback permitido; ligue XFAKE_STRICT_SSL p/ exigir SSL real)")


def build_ssl_aasist_backend(x, dropout_rate: float = 0.3, proj_dim: int = 128,
                             name: str = "ssl"):
    """Back-end de grafo estilo AASIST sobre features SSL.

    Receita SOTA atual (wav2vec2/WavLM → grafo AASIST): em vez do back-end raso
    (conv 1D + attention pooling), pluga a sequência de hidden-states do SSL num
    grafo espectro-temporal (GAT espectral + temporal + HS-GAL + readout).

    Args:
        x: tensor (B, T, C) — saída do extrator SSL (hidden-states).
        dropout_rate: dropout dos blocos GAT.
        proj_dim: projeção dos hidden-states antes do grafo (reduz custo).
        name: prefixo dos nomes das camadas.

    Returns:
        Vetor (B, 4*32) pronto para `create_classification_head`.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    from app.domain.models.architectures.layers import (
        GATConvLayer,
        GraphPoolLayer,
        GraphReadoutLayer,
        HSGALLayer,
    )

    # Comprimento temporal FIXO. Necessário porque o ramo espectral usa o tempo
    # como dimensão de features dos nós (canais=nós) — e os pesos do GAT exigem
    # feature-dim estática. A sequência SSL tem T dinâmico, então reamostramos
    # para T_FIXED frames antes de montar os grafos.
    t_fixed = 64

    # Projeção dos hidden-states (768/1024 → proj_dim) + normalização
    h = layers.Conv1D(proj_dim, 1, name=f"{name}_proj")(x)
    h = layers.LayerNormalization(name=f"{name}_proj_ln")(h)

    def _resize_time(z):
        z4 = tf.expand_dims(z, axis=1)          # (B, 1, T, C)
        z4 = tf.image.resize(z4, [1, t_fixed])  # (B, 1, T_FIXED, C)
        return tf.squeeze(z4, axis=1)           # (B, T_FIXED, C)

    h = layers.Lambda(
        _resize_time, output_shape=(t_fixed, proj_dim),
        name=f"{name}_time_resize",
    )(h)

    # Dois grafos: espectral (canais como nós) e temporal (tempo como nós)
    spectral = layers.Permute((2, 1), name=f"{name}_spec_transpose")(h)
    temporal = h

    spectral = GATConvLayer(
        out_features=32, num_heads=4, dropout_rate=dropout_rate,
        concat_heads=True, name=f"{name}_gat_spec")(spectral)
    temporal = GATConvLayer(
        out_features=32, num_heads=4, dropout_rate=dropout_rate,
        concat_heads=True, name=f"{name}_gat_temp")(temporal)

    # HS-GAL: atenção heterogênea cruzada (contribuição do AASIST)
    spectral, temporal = HSGALLayer(
        out_features=32, num_heads=2, dropout_rate=dropout_rate,
        name=f"{name}_hsgal")([spectral, temporal])

    spectral = GraphPoolLayer(ratio=0.5, name=f"{name}_pool_spec")(spectral)
    temporal = GraphPoolLayer(ratio=0.5, name=f"{name}_pool_temp")(temporal)

    s = GraphReadoutLayer(name=f"{name}_readout_spec")(spectral)
    t = GraphReadoutLayer(name=f"{name}_readout_temp")(temporal)
    return layers.Concatenate(name=f"{name}_readout_concat")([s, t])


def find_encoder_layers(backbone: Any) -> Optional[List[Any]]:
    """Localiza a lista de camadas do encoder Transformer de um backbone HF.

    Tenta os caminhos comuns (`.wavlm/.hubert/.wav2vec2 → .encoder → .layer(s)`)
    e um fallback direto (`.encoder.layer(s)`). Retorna a lista de camadas ou
    None se não reconhecer a estrutura.
    """
    candidates_base = ("wavlm", "hubert", "wav2vec2", "model")
    for base_attr in candidates_base:
        base = getattr(backbone, base_attr, None)
        if base is None:
            continue
        enc = getattr(base, "encoder", None)
        if enc is None:
            continue
        for layers_attr in ("layer", "layers"):
            layers_list = getattr(enc, layers_attr, None)
            if layers_list is not None and len(layers_list) > 0:
                return list(layers_list)

    # Fallback: encoder diretamente no backbone
    enc = getattr(backbone, "encoder", None)
    if enc is not None:
        for layers_attr in ("layer", "layers"):
            layers_list = getattr(enc, layers_attr, None)
            if layers_list is not None and len(layers_list) > 0:
                return list(layers_list)
    return None


def set_ssl_backbone_trainability(
    backbone: Any,
    freeze_weights: bool = True,
    n_trainable_layers: int = 0,
) -> str:
    """Configura a treinabilidade de um backbone SSL.

    Precedência:
      1. `n_trainable_layers > 0` → fine-tuning PARCIAL: congela tudo e
         descongela apenas as últimas N camadas do encoder (recomendado).
      2. `freeze_weights=True` (e N=0) → extrator totalmente congelado (legado).
      3. caso contrário → backbone totalmente treinável.

    Returns:
        String descrevendo o modo aplicado (para logging/telemetria).
    """
    if n_trainable_layers and n_trainable_layers > 0:
        enc = find_encoder_layers(backbone)
        if enc:
            n_total = len(enc)
            n = min(int(n_trainable_layers), n_total)
            # Congela TUDO (recursivo) e reabilita só as últimas N camadas.
            try:
                backbone.trainable = False
                for lyr in enc[-n:]:
                    lyr.trainable = True
                msg = f"fine-tune parcial: últimas {n}/{n_total} camadas do encoder"
                logger.info("SSL %s", msg)
                return msg
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Falha ao descongelar camadas SSL (%s); congelando tudo.", e
                )
                try:
                    backbone.trainable = False
                except Exception:
                    pass
                return "frozen (fallback: estrutura de encoder não editável)"
        else:
            logger.warning(
                "Camadas do encoder SSL não localizadas; aplicando trainable=%s "
                "global.", (not freeze_weights)
            )
            try:
                backbone.trainable = bool(not freeze_weights)
            except Exception:
                pass
            return "encoder não localizado (trainable global)"

    # Sem fine-tuning parcial
    try:
        backbone.trainable = bool(not freeze_weights)
    except Exception:
        pass
    return "frozen" if freeze_weights else "fully trainable"
