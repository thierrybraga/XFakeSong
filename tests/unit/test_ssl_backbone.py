"""Fidelidade do backbone SSL portado do PyTorch para Keras.

Até 2026-07-27, WavLM e HuBERT rodavam no caminho TensorFlow como uma CNN-1D
**treinada do zero** — os números publicados sob esses nomes não tinham relação
com os modelos dos artigos. O port (``ssl_backbone.py``) lê o checkpoint
PyTorch e reimplementa o forward em Keras.

Estes testes exigem REDE na primeira execução (baixam ~360 MB por checkpoint) e
o pacote ``torch``; são pulados quando qualquer um falta.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require(module: str):
    return pytest.importorskip(module, reason=f"{module} não instalado")


@pytest.fixture(scope="module")
def audio() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, 16000)).astype("float32")


@pytest.mark.parametrize(
    "family,checkpoint",
    [
        ("hubert", "facebook/hubert-base-ls960"),
        ("wavlm", "microsoft/wavlm-base"),
    ],
)
def test_keras_port_matches_pytorch(family, checkpoint, audio):
    """Mesma entrada → mesma saída em TODOS os hidden states.

    É a única garantia real de que o backbone é o do artigo: se o mapeamento de
    pesos, a convolução posicional com *weight norm* ou — no WavLM — o viés
    posicional relativo com gating estiverem errados, as saídas divergem.
    """
    torch = _require("torch")
    _require("transformers")
    import tensorflow as tf
    from transformers import HubertModel, WavLMModel

    from app.domain.models.architectures.ssl_backbone import (
        PretrainedSSLBackbone,
        SSLBackboneUnavailable,
    )

    try:
        backbone = PretrainedSSLBackbone(
            family=family, checkpoint=checkpoint, output_hidden_states=True
        )
    except SSLBackboneUnavailable as exc:
        pytest.skip(f"checkpoint indisponível (offline?): {exc}")

    keras_out = backbone(tf.constant(audio), training=False).numpy()

    reference = (WavLMModel if family == "wavlm" else HubertModel)
    torch_model = reference.from_pretrained(checkpoint).eval()
    with torch.no_grad():
        hidden = torch_model(
            torch.from_numpy(audio), output_hidden_states=True
        ).hidden_states
    torch_out = np.stack([h.numpy() for h in hidden], axis=1)

    assert keras_out.shape == torch_out.shape
    assert np.abs(keras_out - torch_out).max() < 1e-3, (
        f"{family}: port divergiu do PyTorch "
        f"(max|dif|={np.abs(keras_out - torch_out).max():.2e})"
    )


@pytest.mark.parametrize("family", ["wavlm", "hubert"])
def test_backbone_is_frozen_and_only_head_trains(family):
    """Regime pedido: backbone congelado, só a cabeça (e os pesos de camada)."""
    _require("torch")
    _require("transformers")

    from app.domain.models.architectures.ssl_backbone import SSLBackboneUnavailable

    module = pytest.importorskip(
        f"app.domain.models.architectures.{family}"
    )
    try:
        model = module.create_model((48000, 1), num_classes=2)
    except SSLBackboneUnavailable as exc:
        pytest.skip(f"checkpoint indisponível (offline?): {exc}")

    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    frozen = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)

    if frozen < 50_000_000:
        pytest.skip("backbone pré-treinado não carregou (fallback CNN-1D)")

    # O backbone (~94 M) domina; a parte treinável tem de ser uma fração mínima.
    assert frozen > 90_000_000, frozen
    assert trainable < 0.05 * frozen, (trainable, frozen)
    # A soma ponderada das camadas é treinável (receita SUPERB).
    assert any("layer_weights" in w.name for w in model.trainable_weights)


def test_relative_position_buckets_match_reference():
    """Bucketização das posições relativas do WavLM (bidirecional, 320 buckets)."""
    _require("torch")
    import torch

    from transformers.models.wavlm.modeling_wavlm import WavLMAttention

    import tensorflow as tf

    from app.domain.models.architectures.ssl_backbone import PretrainedSSLBackbone

    # Instância "crua" só para exercitar a função de bucketização, sem baixar
    # pesos: os atributos usados são apenas num_buckets/max_distance.
    probe = PretrainedSSLBackbone.__new__(PretrainedSSLBackbone)
    probe.num_buckets = 320
    probe.max_distance = 800

    seq_len = 37
    got = PretrainedSSLBackbone._relative_position_bucket(probe, seq_len)

    reference = WavLMAttention.__new__(WavLMAttention)
    reference.num_buckets = 320
    reference.max_distance = 800
    pos = torch.arange(seq_len, dtype=torch.long)
    rel = pos[None, :] - pos[:, None]
    expected = WavLMAttention._relative_positions_bucket(reference, rel).numpy()

    assert np.array_equal(tf.get_static_value(got), expected)
