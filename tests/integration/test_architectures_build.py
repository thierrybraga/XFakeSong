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
