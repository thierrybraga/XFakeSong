"""Paridade NUMÉRICA entre a porta Keras do backbone SSL e o forward do PyTorch.

SUJEITO: ``app/domain/models/architectures/ssl_backbone.py::PretrainedSSLBackbone``.

Por que este teste existe
-------------------------
O ``transformers`` não entrega WavLM em TensorFlow, e os modelos TF que ele
entrega nem importam sob Keras 3. Por isso o projeto LÊ o ``state_dict`` do
checkpoint PyTorch e **reimplementa o forward** com operações Keras — extrator
convolucional, projeção de features, positional conv embedding, e o encoder
inteiro, incluindo o viés posicional relativo com *gating* que é a assinatura do
WavLM (Chen et al., 2022).

Uma reimplementação assim é fácil de errar em silêncio: um eixo trocado no
reshape do gate, um bucket de posição relativa com fórmula ligeiramente
diferente, ou o layer norm no lado errado do bloco produzem um backbone que
RODA, treina e reporta métricas — só que não é o WavLM. Nenhum teste de
formato pega isso, e o resultado do TCC passaria a descrever um modelo que a
literatura não conhece.

A auditoria de 2026-08-20 verificou que o backbone fica congelado e que a carga
de pesos falha alto (``self._state[key]`` levanta ``KeyError`` em chave
ausente, sem deixar peso aleatório). O que faltava era a prova numérica: dar a
MESMA entrada às duas implementações e comparar as ativações.

O que se compara
----------------
1. ``last_hidden_state`` — a saída que alimenta a cabeça.
2. TODOS os hidden states intermediários, camada a camada. Comparar só a saída
   final esconde erros que se cancelam; camada a camada localiza onde diverge.
3. A forma temporal ``T'``, que depende do "same padding remove 1" do
   positional conv embedding — errar isso desloca a sequência inteira em uma
   amostra.

Tolerância — CALIBRADA POR MUTAÇÃO, não escolhida no chute
----------------------------------------------------------
Os dois caminhos usam float32 e ordens de acumulação diferentes (PyTorch e
oneDNN reassociam somas), então igualdade exata não é o critério certo. Usamos
o erro relativo mediano e o percentil 99,9 do erro absoluto, ambos normalizados
pela escala da própria camada.

Os limites foram fixados depois de medir o piso de ruído E o efeito de defeitos
reais, injetados de propósito na porta Keras (2026-08-20, ``wavlm-base``, 1 s de
ruído gaussiano):

===========================================  ==============  ==============
Implementação                                erro mediano    p99,9
===========================================  ==============  ==============
correta (piso de ruído float32)              1,89e-06        1,74e-05
gating do viés posicional REMOVIDO           3,63e-01        2,92e+00
``gate_a``/``gate_b`` trocados no reshape    5,93e-01        4,91e+00
``num_buckets`` 320 -> 256                   4,88e-01        3,90e+00
===========================================  ==============  ==============

O limite mediano de 2e-4 fica ~100x acima do ruído e ~1800x abaixo do menor
defeito detectado — larga o bastante para não piscar com reassociação de somas,
apertada o bastante para pegar erro de eixo, de gate ou de bucket. Se alguém
afrouxar estes números, o teste perde justamente o que ele existe para pegar.

Vive em ``tests/smoke/`` por CUSTO, não por fragilidade: carrega um checkpoint
de 94,5M parâmetros e roda dois backbones completos. O ``conftest`` marca a
pasta como ``smoke``, que o ``addopts`` exclui da suíte rápida — rode com
``pytest -m smoke tests/smoke/test_ssl_backbone_parity_torch.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

CHECKPOINT = "microsoft/wavlm-base"
#: Erro relativo mediano tolerado por camada. Ruído de float32 fica ~1e-6.
TOL_MEDIANA = 2e-4
#: Cauda: percentil 99,9 do erro absoluto normalizado pelo desvio da camada.
TOL_CAUDA = 5e-3


def _sem_backbone(exc: Exception) -> bool:
    """Distingue 'dependência/checkpoint ausente' de 'a porta está errada'.

    Só o primeiro caso justifica pular; um erro de implementação tem de
    REPROVAR, não sumir como skip.
    """
    texto = f"{type(exc).__name__}: {exc}".lower()
    marcas = (
        "no module named",
        "connectionerror",
        "couldn't connect",
        "offline",
        "not a local folder",
        "can't load",
        "no such file",
        "sslbackboneunavailable",
    )
    return any(m in texto for m in marcas)


@pytest.fixture(scope="module")
def referencia_torch():
    """Hidden states do WavLM oficial (PyTorch) para uma entrada fixa."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    try:
        modelo = transformers.WavLMModel.from_pretrained(CHECKPOINT)
    except Exception as exc:  # noqa: BLE001
        if _sem_backbone(exc):
            pytest.skip(f"checkpoint {CHECKPOINT} indisponível offline: {exc}")
        raise
    modelo.eval()

    # Entrada determinística, não silêncio: silêncio esconde erro de viés
    # posicional porque a atenção fica quase uniforme.
    rng = np.random.default_rng(20260820)
    audio = (rng.standard_normal((1, 16000)) * 0.1).astype("float32")

    with torch.no_grad():
        saida = modelo(torch.from_numpy(audio), output_hidden_states=True)
    estados = [h.numpy() for h in saida.hidden_states]
    return {
        "audio": audio,
        "hidden_states": estados,
        "last": saida.last_hidden_state.numpy(),
        "num_layers": modelo.config.num_hidden_layers,
    }


@pytest.fixture(scope="module")
def saida_keras(referencia_torch):
    """Hidden states da porta Keras para a MESMA entrada."""
    pytest.importorskip("tensorflow")
    try:
        from app.domain.models.architectures.ssl_backbone import (
            PretrainedSSLBackbone,
        )

        backbone = PretrainedSSLBackbone(
            family="wavlm", checkpoint=CHECKPOINT, output_hidden_states=True
        )
        saida = backbone(referencia_torch["audio"], training=False)
    except Exception as exc:  # noqa: BLE001
        if _sem_backbone(exc):
            pytest.skip(f"backbone Keras indisponível: {exc}")
        raise
    return np.asarray(saida)


def _erros(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(erro relativo mediano, percentil 99,9 normalizado pela escala)."""
    a = np.asarray(a, dtype="float64").ravel()
    b = np.asarray(b, dtype="float64").ravel()
    assert a.shape == b.shape, f"formas diferentes: {a.shape} vs {b.shape}"
    diff = np.abs(a - b)
    escala = float(np.std(a)) or 1.0
    mediana = float(np.median(diff) / escala)
    cauda = float(np.percentile(diff, 99.9) / escala)
    return mediana, cauda


def test_forma_temporal_bate_com_o_torch(referencia_torch, saida_keras):
    """T' idêntico: o 'same padding remove 1' do pos-conv desloca a sequência.

    Se o corte da última amostra não for feito (ou for feito duas vezes), o
    comprimento sai errado por 1 e TODA a sequência fica deslocada em relação
    ao que o modelo viu no pré-treino.
    """
    esperado = referencia_torch["last"].shape  # (B, T', H)
    # Keras devolve (B, L+1, T', H) com output_hidden_states=True.
    assert saida_keras.ndim == 4, f"esperado (B, L+1, T', H), veio {saida_keras.shape}"
    assert saida_keras.shape[0] == esperado[0]
    assert saida_keras.shape[2] == esperado[1], (
        f"comprimento temporal divergente: Keras T'={saida_keras.shape[2]} "
        f"contra PyTorch T'={esperado[1]} — suspeite do corte do positional "
        "conv embedding"
    )
    assert saida_keras.shape[3] == esperado[2]


def test_numero_de_hidden_states_bate(referencia_torch, saida_keras):
    """L+1 estados: a saída do extrator (camada 0) mais as L do encoder.

    A soma ponderada da receita SUPERB pondera TODOS eles; somar um a menos
    muda o que a cabeça recebe.
    """
    esperado = len(referencia_torch["hidden_states"])
    assert esperado == referencia_torch["num_layers"] + 1
    assert saida_keras.shape[1] == esperado, (
        f"Keras expõe {saida_keras.shape[1]} hidden states, PyTorch {esperado}"
    )


def test_last_hidden_state_bate_numericamente(referencia_torch, saida_keras):
    """A saída que alimenta a cabeça é a mesma nas duas implementações."""
    mediana, cauda = _erros(referencia_torch["last"], saida_keras[:, -1])
    assert mediana < TOL_MEDIANA, (
        f"last_hidden_state divergente: erro relativo mediano {mediana:.2e} "
        f"(limite {TOL_MEDIANA:.0e})"
    )
    assert cauda < TOL_CAUDA, (
        f"last_hidden_state com cauda pesada: p99,9 {cauda:.2e} "
        f"(limite {TOL_CAUDA:.0e}) — divergência localizada, não ruído"
    )


def test_todas_as_camadas_batem(referencia_torch, saida_keras):
    """Camada a camada: localiza ONDE diverge, se divergir.

    Comparar só a saída final esconde erros que se compensam entre blocos.
    """
    problemas = []
    for i, esperado in enumerate(referencia_torch["hidden_states"]):
        mediana, cauda = _erros(esperado, saida_keras[:, i])
        if mediana >= TOL_MEDIANA or cauda >= TOL_CAUDA:
            problemas.append(
                f"camada {i}: mediana={mediana:.2e} cauda={cauda:.2e}"
            )
    assert not problemas, (
        "divergência a partir da primeira camada listada (as seguintes "
        "herdam o erro):\n  " + "\n  ".join(problemas)
    )


def test_backbone_esta_congelado():
    """Nenhum peso do backbone treina — a receita SUPERB congela o upstream."""
    pytest.importorskip("tensorflow")
    try:
        from app.domain.models.architectures.ssl_backbone import (
            PretrainedSSLBackbone,
        )

        backbone = PretrainedSSLBackbone(
            family="wavlm", checkpoint=CHECKPOINT, output_hidden_states=True
        )
        backbone(np.zeros((1, 16000), dtype="float32"), training=False)
    except Exception as exc:  # noqa: BLE001
        if _sem_backbone(exc):
            pytest.skip(f"backbone Keras indisponível: {exc}")
        raise
    assert backbone.trainable is False
    assert not backbone.trainable_weights, (
        f"{len(backbone.trainable_weights)} pesos treináveis no backbone "
        "congelado"
    )
    assert backbone.weights, "backbone sem pesos: a carga do state_dict falhou"
