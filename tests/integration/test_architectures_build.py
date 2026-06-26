"""Cobertura de construção das arquiteturas pelo MESMO caminho dos notebooks.

Os notebooks de modelo (`notebooks/models/*.ipynb`) preparam dados sintéticos com
`BenchmarkData.synthetic(...).prepare_for_architecture(arch)` e instanciam o
modelo via `create_model_by_name(arch, input_shape=prepared.X.shape[1:])`. Antes,
no gate bloqueante do CI (suíte `not smoke`), só DUAS arquiteturas eram
realmente construídas — `MultiscaleCNN` (espectrograma) e `RawNet2` (raw-audio),
ambas via execução de notebook (nbclient, lento). Este teste constrói as **10
arquiteturas neurais não-SSL** de forma rápida (build + 1 forward pass, sem
treino), pegando drift de construção que o `compile()` dos notebooks não detecta.

Exclusões deliberadas:
- **WavLM / HuBERT**: usam `TFWavLMModel/TFHubertModel.from_pretrained(...)`, que
  baixam checkpoints SSL do HuggingFace (rede + `transformers`). Continuam
  cobertos só no smoke (`tests/smoke/test_all_architectures.py`, TF real, manual).
- **SVM / RandomForest**: clássicos (sklearn), não passam pelo factory Keras;
  cobertos pelos notebooks `models/13_svm` e `models/14_random_forest` (nbclient).

Pula automaticamente quando o TensorFlow não está instalado (mesmo padrão dos
testes de execução de notebook).
"""
from __future__ import annotations

import numpy as np
import pytest

# TF é pesado e opcional no ambiente de dev — pula em vez de falhar.
pytest.importorskip("tensorflow")

from app.domain.models.architectures.factory import create_model_by_name  # noqa: E402
from benchmarks.data import BenchmarkData  # noqa: E402

# 10 arquiteturas neurais sem dependência de download SSL.
# Nomes exatos do registry (os mesmos aceitos por create_model_by_name).
NON_SSL_NEURAL = [
    # raw-audio
    "RawNet2",
    "AASIST",
    "RawGAT-ST",
    # espectrograma
    "Sonic Sleuth",
    "Conformer",
    "Hybrid CNN-Transformer",
    "SpectrogramTransformer",
    "EfficientNet-LSTM",
    "MultiscaleCNN",
    "Ensemble",
]


@pytest.fixture(scope="module", autouse=True)
def _seed():
    """Determinismo (numpy + TF) para o build/forward ser reprodutível."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(0)


@pytest.mark.parametrize("architecture", NON_SSL_NEURAL)
def test_architecture_builds_and_forwards(architecture):
    """Constrói a arquitetura pelo caminho do notebook e faz 1 forward pass.

    Replica `notebooks/models/*`: dados sintéticos -> contrato da arquitetura ->
    `create_model_by_name(..., input_shape=prepared.X.shape[1:])`. Como o
    input_shape vem da PRÓPRIA forma preparada, build e forward são sempre
    consistentes; a única falha possível é a arquitetura não construir/rodar —
    exatamente o drift que queremos pegar.
    """
    data = BenchmarkData.synthetic(n=8, shape=(64, 40), seed=7)
    prepared = data.prepare_for_architecture(architecture)
    input_shape = tuple(prepared.X.shape[1:])

    # num_classes=1 -> sigmoid de 1 unidade = p(fake) (convenção do projeto,
    # idêntica à célula de inspeção dos notebooks de modelo).
    model = create_model_by_name(
        architecture, input_shape=input_shape, num_classes=1
    )

    y = model(prepared.X[:1], training=False)
    out = tuple(int(d) for d in y.shape)

    assert out[0] == 1, f"{architecture}: batch de saída inesperado: {out}"
    assert out[-1] in (1, 2), f"{architecture}: nº de classes inesperado: {out}"
    assert model.count_params() > 0, f"{architecture}: modelo sem parâmetros"
    assert np.all(np.isfinite(np.asarray(y))), f"{architecture}: saída não-finita"


# Arquiteturas de espectrograma — o adaptador do benchmark precisa AUMENTAR
# entradas minúsculas para um tamanho treinável.
SPECTROGRAM_ARCHS = [
    "Sonic Sleuth", "Conformer", "Hybrid CNN-Transformer",
    "SpectrogramTransformer", "EfficientNet-LSTM", "MultiscaleCNN", "Ensemble",
]


@pytest.mark.parametrize("architecture", SPECTROGRAM_ARCHS)
def test_spectrogram_arch_builds_from_tiny_synthetic(architecture):
    """Regressão (Sonic Sleuth): a `quick` do benchmark usa
    `synthetic_shape=(8, 8)`. O adaptador `prepare_for_architecture` deve
    redimensionar essa grade minúscula para um espectrograma grande o bastante
    para a arquitetura construir — antes, os 5 blocos de pooling do Sonic Sleuth
    colapsavam (8, 8) até `1×1` e quebravam com "Negative dimension".
    """
    data = BenchmarkData.synthetic(n=8, shape=(8, 8), seed=3)
    prepared = data.prepare_for_architecture(architecture)
    T, F = int(prepared.X.shape[1]), int(prepared.X.shape[2])
    assert T >= 32 and F >= 32, f"{architecture}: preparado {T}x{F} pequeno demais"

    model = create_model_by_name(
        architecture, input_shape=tuple(prepared.X.shape[1:]), num_classes=1
    )
    y = model(prepared.X[:1], training=False)
    assert tuple(int(d) for d in y.shape)[0] == 1, f"{architecture}: batch inesperado"
    assert np.all(np.isfinite(np.asarray(y))), f"{architecture}: saída não-finita"


def test_conformer_accepts_benchmark_hyperparameter_overrides(monkeypatch):
    """Regressão P0: benchmark passa dropout_rate em parameters.

    O wrapper da variante também define dropout_rate; a mesclagem deve
    sobrescrever o default antes da chamada ao construtor, sem duplicar kwargs.
    """
    from app.domain.models.architectures import conformer as conformer_module

    captured = {}

    def fake_create_conformer_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        conformer_module, "create_conformer_model", fake_create_conformer_model
    )

    create_model_by_name(
        "Conformer",
        input_shape=(100, 80),
        num_classes=1,
        dropout_rate=0.3,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_steps=1500,
        decay_steps=50000,
        alpha=1e-7,
        clipnorm=1.0,
        label_smoothing=0.05,
        attention_heads=8,
        hidden_units=256,
        l2_reg_strength=5e-4,
    )

    assert captured["dropout_rate"] == 0.3
    assert captured["learning_rate"] == 1e-4
    assert captured["weight_decay"] == 1e-4
    assert captured["num_heads"] == 8
    assert captured["d_model"] == 256
    assert "attention_heads" not in captured
    assert "hidden_units" not in captured
    assert "l2_reg_strength" not in captured


def test_conformer_accepts_training_service_parameters_contract(monkeypatch):
    """Regressão P0: create_model(..., parameters={...}) não deve falhar."""
    from app.domain.models.architectures import conformer as conformer_module

    captured = {}

    def fake_create_conformer_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        conformer_module, "create_conformer_model", fake_create_conformer_model
    )

    parameters = {
        "dropout_rate": 0.3,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1500,
        "decay_steps": 50000,
        "alpha": 1e-7,
        "clipnorm": 1.0,
        "label_smoothing": 0.05,
    }
    create_model_by_name(
        "Conformer",
        input_shape=(100, 80),
        num_classes=1,
        parameters=parameters,
    )

    assert captured["dropout_rate"] == 0.3
    assert captured["learning_rate"] == 1e-4
    assert captured["weight_decay"] == 1e-4
    assert captured["clipnorm"] == 1.0
    assert captured["label_smoothing"] == 0.05


# Arquiteturas cujos supported_variants NÃO incluem "default" — instanciá-las
# pelo caminho do notebook (sem passar variant) usava o sentinel "default" e
# emitia um WARNING "Variant default not supported" assustador para quem roda os
# notebooks 10/11/12. "default" é o fallback universal e nunca deve avisar.
ARCHS_WITHOUT_DEFAULT_VARIANT = ["EfficientNet-LSTM", "MultiscaleCNN", "Ensemble"]


@pytest.mark.parametrize("architecture", ARCHS_WITHOUT_DEFAULT_VARIANT)
def test_default_variant_does_not_warn(architecture, caplog):
    """Regressão: pedir o variant default (implícito) não deve logar
    "not supported", mesmo quando "default" não está em supported_variants."""
    import logging

    from app.domain.models.architectures.factory import get_architecture_info

    info = get_architecture_info(architecture)
    # Pré-condição: estas arquiteturas realmente não listam "default".
    assert "default" not in info.supported_variants, (
        f"{architecture}: teste obsoleto — 'default' agora está nos variants"
    )

    data = BenchmarkData.synthetic(n=8, shape=(64, 40), seed=7)
    prepared = data.prepare_for_architecture(architecture)
    with caplog.at_level(logging.WARNING,
                         logger="app.domain.models.architectures.factory"):
        create_model_by_name(
            architecture, input_shape=tuple(prepared.X.shape[1:]), num_classes=1
        )
    offending = [r.getMessage() for r in caplog.records
                 if "not supported" in r.getMessage()]
    assert not offending, f"{architecture}: avisos espúrios de variant: {offending}"
