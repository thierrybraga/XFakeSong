"""O artefato SSL declara o protocolo sob o qual rodou.

SUJEITO: o bloco ``config`` do ``results.json`` que
``scripts/benchmark/run_wavlm_original_benchmark.py`` grava, comparado ao que
``benchmarks/runner.py`` grava para as 9 entradas Keras.

Por que este teste existe
-------------------------
As 11 entradas do escopo oficial saem de DOIS runners: nove pelo caminho Keras
e duas — WavLM Original e HuBERT Original — por um runner PyTorch próprio. Os
dois aplicam a correção de banda quando ``--band-correction-hz`` é passado (e o
compose passa por padrão), mas até 2026-08-21 só o Keras REGISTRAVA a política
no artefato.

O efeito não era um número errado: era uma dúvida que o artefato não permitia
resolver. Na bateria de 2026-08-20, ``wavlm_original/results.json`` e
``hubert_original/results.json`` traziam ``band_correction_hz`` ausente
enquanto ``hybrid_cnn_transformer/results.json`` trazia ``7500.0``. Lendo só os
artefatos, a conclusão natural é que as duas entradas SSL rodaram sem a
correção — uma assimetria de protocolo que tornaria os EER delas não
comparáveis com o resto da tabela. Só o log do container desfazia a suspeita
(``[protocolo] correção aplicada a 12162 amostras``), e log não acompanha o
artefato.

Num TCC onde proveniência é argumento, o artefato precisa dizer sob qual
protocolo rodou. Este teste trava as duas metades: o bloco existe, e ele é
campo a campo igual ao do caminho Keras.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_SSL = ROOT / "scripts" / "benchmark" / "run_wavlm_original_benchmark.py"


@pytest.fixture(scope="module")
def runner_ssl():
    """Carrega o runner SSL como módulo, sem executá-lo como script."""
    spec = importlib.util.spec_from_file_location("sslrun", RUNNER_SSL)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_politica_de_banda_e_igual_a_do_runner_keras(runner_ssl):
    """Campo a campo idêntica — só `origem` distingue quem aplicou.

    Se as duas divergirem, os artefatos das 11 entradas deixam de ser
    comparáveis por leitura direta, que é o ponto de gravar a política.
    """
    from benchmarks.runner import _band_correction_policy

    class _Cfg:
        band_correction_hz = 7500.0

    ssl = runner_ssl._band_correction_block(7500.0)
    keras = _band_correction_policy(_Cfg(), None)

    assert ssl is not None, "runner SSL não produziu bloco de política"
    assert keras is not None, "runner Keras não produziu bloco de política"

    sem_origem = lambda d: {k: v for k, v in d.items() if k != "origem"}  # noqa: E731
    assert sem_origem(ssl) == sem_origem(keras), (
        "a política gravada pelos dois runners divergiu:\n"
        f"  SSL   = {ssl}\n  Keras = {keras}"
    )
    # `origem` existe justamente para dizer QUAL runner aplicou.
    assert ssl["origem"] != keras["origem"]


def test_sem_a_flag_a_politica_e_nula(runner_ssl):
    """Sem `--band-correction-hz` o campo é `None`, não um bloco vazio.

    `None` diz "não aplicada"; um dicionário com zeros diria "aplicada com
    corte zero", que é outra coisa.
    """
    assert runner_ssl._band_correction_block(None) is None
    assert runner_ssl._band_correction_block(0) is None


def test_o_bloco_config_do_ssl_declara_os_campos_de_protocolo():
    """`band_correction_hz`, `band_correction` e `checkpoint_monitor` no fonte.

    Verificação por leitura do fonte porque montar o `results.json` completo
    exigiria treinar um modelo. O que importa aqui é que as chaves entrem no
    dicionário `config`, não os seus valores num run específico.
    """
    fonte = RUNNER_SSL.read_text(encoding="utf-8")
    inicio = fonte.index('    results = {')
    bloco = fonte[inicio:inicio + 3000]
    for chave in ('"band_correction_hz"', '"band_correction"', '"checkpoint_monitor"'):
        assert chave in bloco, f"{chave} ausente do bloco config do runner SSL"


def test_checkpoint_monitor_e_declarado_como_nao_aplicavel():
    """O runner SSL tem laço de treino próprio: o monitor do Keras não se aplica.

    Declarar `None` com a nota evita a leitura de que a opção foi esquecida —
    que é como um `checkpoint_monitor` ausente aparece quando as outras nove
    entradas trazem `val_eer`.
    """
    fonte = RUNNER_SSL.read_text(encoding="utf-8")
    assert "checkpoint_monitor_nota" in fonte
    assert "nao aplicavel" in fonte or "não aplicável" in fonte
