"""O escopo oficial constrói o RawNet2 ANTI-SPOOFING, e todos concordam sobre isso.

SUJEITO: a topologia do RawNet2 nas fontes que a definem —
``rawnet2.py::_RAWNET2_ANTISPOOFING_PARAMS`` (definição canônica),
``planning.NEURAL_BENCHMARK_HPARAMS["rawnet2"]`` (o que o benchmark treina E o
que a interface Gradio exibe) e ``registry.default_params`` (a variante do app).

Por que este teste existe
-------------------------
"RawNet2" nomeia DUAS arquiteturas diferentes:

- **Improved RawNet** (Jung et al., 2020), de VERIFICAÇÃO DE LOCUTOR: Sinc 128,
  canais 128/256, 1×GRU(1024) — ~7,0M parâmetros.
- **Baseline anti-spoofing** (Tak et al., 2021): Sinc 20, canais 20/128,
  3×GRU(1024) — ~17,6M parâmetros. É este que a literatura de anti-spoofing
  compara por EER, e é este que o escopo oficial passou a treinar em
  2026-08-20.

Até essa data o benchmark construía a de verificação de locutor, e a tabela do
TCC comparava o EER dela com baselines da literatura que medem a outra.

O modo de falha que este teste tranca é mais sutil que a troca em si: a
topologia foi colocada, num primeiro momento, apenas em ``benchmarks/runner.py``.
A interface Gradio resolve seus defaults por
``planning.effective_hyperparameters()``, que lê ``NEURAL_BENCHMARK_HPARAMS``
sobre ``registry.default_params`` — nenhum dos dois sabia da mudança. Resultado:
a interface exibia e treinava a variante de locutor enquanto o benchmark
treinava a de anti-spoofing. A `CLAUDE.md` chama isso de "quarta fonte de
verdade"; aqui ela produzia duas arquiteturas distintas sob um nome só.
"""

from __future__ import annotations

import pytest

#: Topologia do baseline anti-spoofing (Tak et al., 2021).
TAK_2021 = {
    "sinc_filters": 20,
    "sinc_kernel_size": 1024,
    "res_filters": [20, 20, 128, 128, 128, 128],
    "gru_units": 1024,
    "gru_layers": 3,
    "dense_units": 1024,
}


def test_plano_do_benchmark_declara_a_topologia_anti_spoofing():
    """`NEURAL_BENCHMARK_HPARAMS` é o que o benchmark treina."""
    from benchmarks.planning import NEURAL_BENCHMARK_HPARAMS

    plano = NEURAL_BENCHMARK_HPARAMS["rawnet2"]
    divergencias = {
        chave: (plano.get(chave), esperado)
        for chave, esperado in TAK_2021.items()
        if plano.get(chave) != esperado
    }
    assert not divergencias, (
        "o plano do benchmark não descreve o baseline anti-spoofing "
        f"(chave: (plano, esperado)): {divergencias}"
    )


def test_definicao_canonica_da_arquitetura_concorda_com_o_plano():
    """A cópia em `planning.py` não pode divergir da definição da arquitetura.

    Os valores são duplicados de propósito: `planning.py` não importa de
    `architectures/` para não puxar TensorFlow no import da interface. O preço
    da duplicação é este teste.
    """
    pytest.importorskip("tensorflow")
    from app.domain.models.architectures.rawnet2 import (
        _RAWNET2_ANTISPOOFING_PARAMS,
    )
    from benchmarks.planning import NEURAL_BENCHMARK_HPARAMS

    plano = NEURAL_BENCHMARK_HPARAMS["rawnet2"]
    divergencias = {
        chave: (plano.get(chave), valor)
        for chave, valor in _RAWNET2_ANTISPOOFING_PARAMS.items()
        if plano.get(chave) != valor
    }
    assert not divergencias, (
        "planning.py e rawnet2.py descrevem topologias diferentes "
        f"(chave: (plano, arquitetura)): {divergencias}"
    )


def test_interface_gradio_resolve_a_mesma_topologia_do_benchmark():
    """`effective_hyperparameters` é o que a interface exibe e treina.

    Esta é a asserção que teria pego a divergência: ela percorre o MESMO
    caminho que a aba de treino do Gradio, em vez de ler o dicionário direto.
    """
    from benchmarks.planning import effective_hyperparameters

    efetivo = effective_hyperparameters("RawNet2")
    divergencias = {
        chave: (efetivo.get(chave), esperado)
        for chave, esperado in TAK_2021.items()
        if efetivo.get(chave) != esperado
    }
    assert not divergencias, (
        "a interface resolveria uma topologia diferente da que o benchmark "
        f"treina (chave: (interface, benchmark)): {divergencias}"
    )


def test_manifesto_declara_a_variante_que_e_construida():
    """O rótulo de proveniência tem de dizer QUAL RawNet2 foi treinado.

    O `benchmarks/config.py` avisa que rótulos defasados já existiram — um
    `rawnet2_paper_like` que não dizia qual dos dois. O nome da variante viaja
    para `results.json` e daí para o TCC.
    """
    from benchmarks.config import OFFICIAL_TCC_MODEL_MANIFEST

    entrada = next(
        item for item in OFFICIAL_TCC_MODEL_MANIFEST
        if item.get("benchmark_name") == "RawNet2"
    )
    variante = str(entrada.get("variant", "")).lower()
    assert "antispoofing" in variante or "anti_spoofing" in variante, (
        f"o rótulo {variante!r} não declara a variante anti-spoofing"
    )
    assert "speakerverification" not in variante, (
        f"o rótulo {variante!r} ainda anuncia a variante de verificação de locutor"
    )

def test_o_runner_tem_UM_SO_ramo_para_rawnet2():
    """A cadeia if/elif de `_run_neural` não pode ter dois ramos para o mesmo valor.

    ESTE é o teste que faltava. A troca de variante nasceu como um
    ``elif compact == "rawnet2"`` colocado ABAIXO de um ``if compact ==
    "rawnet2"`` já existente — e portanto inalcançável. Python entra apenas no
    primeiro ramo que casa, então a promoção da topologia nunca executava: o
    benchmark seguia construindo a variante de verificação de locutor enquanto
    manifesto, plano e tabela de custo declaravam a anti-spoofing.

    O defeito passou por três testes verdes (`test_plano...`,
    `test_definicao_canonica...`, `test_interface_gradio...`) porque todos
    olhavam as FONTES de configuração, nenhum o CAMINHO que as consome. Um
    ramo morto não muda nenhum dicionário — só o modelo que sai do outro lado.
    """
    import inspect
    import re

    from benchmarks import runner

    fonte = inspect.getsource(runner._run_neural)
    # Conta só ramos de código, não menções em comentário.
    ramos = re.findall(r'^\s*(?:el)?if\s+compact\s*==\s*"rawnet2"', fonte, re.M)
    assert len(ramos) == 1, (
        f"{len(ramos)} ramos para compact == 'rawnet2' em _run_neural — o "
        "segundo é inalcançável e sua lógica nunca executa"
    )


def test_o_runner_promove_a_topologia_para_parameters():
    """`parameters` é o único canal até `create_model` — a topologia precisa cair lá.

    Reproduz a decisão do runner sobre o `train_config` real do plano e afirma
    que as seis chaves de topologia chegam a `parameters`. Sem isso, o
    `create_model` recebe apenas os `default_params` do registry, que descrevem
    a OUTRA arquitetura.
    """
    from benchmarks.planning import NEURAL_BENCHMARK_HPARAMS

    train_config = dict(NEURAL_BENCHMARK_HPARAMS["rawnet2"])
    model_params = train_config.setdefault("parameters", {})
    for chave in TAK_2021:
        if chave in train_config:
            model_params[chave] = train_config[chave]

    faltando = [c for c in TAK_2021 if c not in model_params]
    assert not faltando, (
        f"o plano não carrega as chaves de topologia {faltando} — o runner não "
        "tem o que promover e o create_model cai nos defaults do registry"
    )
    divergentes = {
        c: (model_params[c], v) for c, v in TAK_2021.items() if model_params[c] != v
    }
    assert not divergentes, f"topologia promovida diverge do baseline: {divergentes}"
