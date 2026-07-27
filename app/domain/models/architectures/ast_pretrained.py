"""Transferência dos pesos pré-treinados do AST (PyTorch) para o AST Keras.

O Audio Spectrogram Transformer (Gong et al., Interspeech 2021) **depende** de
inicialização pré-treinada: o artigo parte de um ViT/DeiT treinado em ImageNet
e refina em AudioSet. Treinar 85M parâmetros do zero sobre dezenas de milhares
de amostras é o regime que, neste projeto, degradou o modelo até chute
aleatório (EER ~51%).

Por que este módulo existe: o ``transformers`` **não publica AST em
TensorFlow**, e neste stack seus modelos TF sequer importam (exigem o pacote
``tf-keras`` quando o Keras instalado é 3.x). O checkpoint **PyTorch**, porém,
é acessível sem tocar em TF — então lemos o ``state_dict`` do
``ASTModel`` e escrevemos os tensores diretamente nas camadas Keras deste
projeto.

Escopo: só a configuração ViT-Base (``embed_dim=768``, 12 blocos, 12 cabeças,
``ff_dim=3072``), que é a do checkpoint. A cabeça de classificação NÃO é
transferida (o AST do checkpoint tem 527 classes do AudioSet; aqui são 2).

Interpolação posicional: o checkpoint foi treinado com uma grade de patches
maior (1024×128 → 101×12) do que a usada aqui (300×128 → 29×12). O próprio
artigo prescreve *cortar ou interpolar* o embedding posicional ao mudar a
resolução de entrada — fazemos interpolação bilinear sobre a grade 2-D.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Checkpoint padrão — AST refinado em AudioSet (o do artigo).
DEFAULT_AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"

#: Configuração suportada pela transferência (ViT-Base, a do checkpoint).
_SUPPORTED = {"embed_dim": 768, "num_blocks": 12, "num_heads": 12, "ff_dim": 3072}


class ASTPretrainedUnavailable(RuntimeError):
    """Pesos pré-treinados do AST não puderam ser obtidos."""


def _load_state_dict(checkpoint: str) -> Tuple[Dict[str, np.ndarray], Any]:
    """Lê o ``state_dict`` do AST PyTorch como arrays numpy.

    Nada de TensorFlow aqui: o caminho PyTorch do ``transformers`` funciona
    mesmo quando o caminho TF está indisponível.
    """
    try:
        from transformers import ASTModel
    except Exception as exc:  # noqa: BLE001
        raise ASTPretrainedUnavailable(
            f"transformers indisponível para ler o AST PyTorch: {exc}"
        ) from exc

    try:
        torch_model = ASTModel.from_pretrained(checkpoint)
    except Exception as exc:  # noqa: BLE001
        raise ASTPretrainedUnavailable(
            f"não foi possível baixar/abrir o checkpoint '{checkpoint}': {exc}"
        ) from exc

    state = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}
    config = torch_model.config
    del torch_model
    return state, config


def _interpolate_position_embeddings(
    pos: np.ndarray, src_grid: Tuple[int, int], dst_grid: Tuple[int, int],
) -> np.ndarray:
    """Reamostra o embedding posicional de patches entre grades 2-D.

    Args:
        pos: ``(n_src_patches, embed_dim)`` — SEM os tokens especiais.
        src_grid / dst_grid: ``(freq_patches, time_patches)`` de origem/destino.

    Returns:
        ``(n_dst_patches, embed_dim)``.
    """
    import tensorflow as tf

    src_f, src_t = src_grid
    dst_f, dst_t = dst_grid
    embed_dim = pos.shape[-1]

    grid = pos.reshape(1, src_f, src_t, embed_dim)
    resized = tf.image.resize(
        tf.convert_to_tensor(grid), [dst_f, dst_t], method="bilinear"
    ).numpy()
    return resized.reshape(dst_f * dst_t, embed_dim)


def _assign(layer, weights) -> None:
    layer.set_weights(weights)


def load_ast_pretrained_weights(
    model,
    *,
    num_blocks: int,
    num_heads: int,
    embed_dim: int,
    ff_dim: int,
    num_patches_grid: Tuple[int, int],
    checkpoint: str = DEFAULT_AST_CHECKPOINT,
) -> Dict[str, Any]:
    """Escreve os pesos do AST pré-treinado no modelo Keras já construído.

    Args:
        model: modelo Keras criado por ``create_spectrogram_transformer_model``.
        num_blocks/num_heads/embed_dim/ff_dim: config do modelo (validada).
        num_patches_grid: ``(patches_no_tempo, patches_na_frequência)`` do
            modelo — usado para reamostrar o embedding posicional.
        checkpoint: id do checkpoint HuggingFace (PyTorch).

    Returns:
        Dicionário com metadados da transferência (para o relatório do run).

    Raises:
        ASTPretrainedUnavailable: se a config não for a do checkpoint ou se os
            pesos não puderem ser obtidos.
    """
    cfg = {"embed_dim": embed_dim, "num_blocks": num_blocks,
           "num_heads": num_heads, "ff_dim": ff_dim}
    if cfg != _SUPPORTED:
        raise ASTPretrainedUnavailable(
            f"pesos pré-treinados existem apenas para a configuração ViT-Base "
            f"{_SUPPORTED}; recebido {cfg}. Use pretrained=False ou a variante "
            f"'spectrogram_transformer' (default)."
        )

    state, hf_config = _load_state_dict(checkpoint)

    def w(key: str) -> np.ndarray:
        if key not in state:
            raise ASTPretrainedUnavailable(f"chave ausente no checkpoint: {key}")
        return state[key]

    transferred: list[str] = []

    # ── Patch embedding: Conv2d(1, 768, k=16, s=10) ────────────────────────
    # PyTorch: (out, in, kh, kw) → Keras: (kh, kw, in, out)
    conv_w = np.transpose(w("embeddings.patch_embeddings.projection.weight"),
                          (2, 3, 1, 0))
    conv_b = w("embeddings.patch_embeddings.projection.bias")
    _assign(model.get_layer("patch_embedding").conv, [conv_w, conv_b])
    transferred.append("patch_embedding")

    # ── Token de classe ───────────────────────────────────────────────────
    # O AST usa DOIS tokens especiais (CLS + destilação, herdados do DeiT);
    # este modelo tem apenas o CLS, então o de destilação é descartado.
    _assign(model.get_layer("class_token_layer"), [w("embeddings.cls_token")])
    transferred.append("class_token")

    # ── Embedding posicional (interpolado para a grade deste modelo) ───────
    pos = w("embeddings.position_embeddings")[0]      # (2 + n_src, 768)
    cls_pos = pos[:1]                                  # posição do CLS
    patch_pos = pos[2:]                                # descarta a de destilação

    src_n = patch_pos.shape[0]
    src_f = hf_config.num_mel_bins  # 128
    # AST: grade = (freq_patches, time_patches); deriva a de frequência da
    # config e a temporal por divisão exata.
    src_fp = (src_f - hf_config.patch_size) // hf_config.frequency_stride + 1
    src_tp = src_n // max(src_fp, 1)

    dst_tp, dst_fp = num_patches_grid
    patch_pos = _interpolate_position_embeddings(
        patch_pos, (src_fp, src_tp), (dst_fp, dst_tp)
    )
    pos_layer = model.get_layer("pos_encoding")
    expected = pos_layer.pos_embedding.shape[1]
    new_pos = np.concatenate([cls_pos, patch_pos], axis=0)[None, ...]
    if new_pos.shape[1] != expected:
        raise ASTPretrainedUnavailable(
            f"embedding posicional reamostrado com {new_pos.shape[1]} posições, "
            f"mas o modelo espera {expected}."
        )
    _assign(pos_layer, [new_pos.astype("float32")])
    transferred.append(
        f"position_embeddings({src_fp}x{src_tp}→{dst_fp}x{dst_tp})"
    )

    # ── Blocos Transformer ────────────────────────────────────────────────
    key_dim = embed_dim // num_heads
    for i in range(num_blocks):
        p = f"encoder.layer.{i}."
        block = model.get_layer(f"ast_block_{i}")

        # Pre-LN: layernorm_before → antes da atenção; layernorm_after → antes do FFN
        _assign(block.layernorm1, [w(p + "layernorm_before.weight"),
                                   w(p + "layernorm_before.bias")])
        _assign(block.layernorm2, [w(p + "layernorm_after.weight"),
                                   w(p + "layernorm_after.bias")])

        # MultiHeadAttention do Keras: kernels (embed, heads, key_dim);
        # PyTorch guarda Linear como (out, in) → transpõe antes de fatiar.
        def _qkv(name: str):
            kernel = w(p + f"attention.attention.{name}.weight").T   # (in, out)
            bias = w(p + f"attention.attention.{name}.bias")
            return (kernel.reshape(embed_dim, num_heads, key_dim),
                    bias.reshape(num_heads, key_dim))

        q_k, q_b = _qkv("query")
        k_k, k_b = _qkv("key")
        v_k, v_b = _qkv("value")
        out_k = w(p + "attention.output.dense.weight").T             # (in, out)
        out_k = out_k.reshape(num_heads, key_dim, embed_dim)
        out_b = w(p + "attention.output.dense.bias")
        _assign(block.attention,
                [q_k, q_b, k_k, k_b, v_k, v_b, out_k, out_b])

        # FFN: Dense(ff_dim, gelu) → Dense(embed_dim)
        _assign(block.ffn.layers[0], [w(p + "intermediate.dense.weight").T,
                                      w(p + "intermediate.dense.bias")])
        _assign(block.ffn.layers[-1], [w(p + "output.dense.weight").T,
                                       w(p + "output.dense.bias")])
        transferred.append(f"ast_block_{i}")

    # ── LayerNorm final ───────────────────────────────────────────────────
    _assign(model.get_layer("final_norm"),
            [w("layernorm.weight"), w("layernorm.bias")])
    transferred.append("final_norm")

    info = {
        "checkpoint": checkpoint,
        "source_patch_grid": [int(src_fp), int(src_tp)],
        "target_patch_grid": [int(dst_fp), int(dst_tp)],
        "layers_transferred": len(transferred),
        "classifier_head": "não transferida (527 classes AudioSet → 2)",
    }
    logger.info(
        "AST: pesos pré-treinados carregados de '%s' — %d grupos de camadas, "
        "embedding posicional reamostrado %dx%d → %dx%d.",
        checkpoint, len(transferred), src_fp, src_tp, dst_fp, dst_tp,
    )
    return info
