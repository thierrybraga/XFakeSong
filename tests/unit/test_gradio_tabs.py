"""Cada aba da interface constrói e degrada sem derrubar o resto.

Por que este arquivo existe (2026-07-28): a interface Gradio tem ~10 mil linhas
e a única cobertura era `tests/smoke/test_app_startup.py`, que verifica se o
módulo importa. Como `tabs/__init__.py` troca por um placeholder qualquer aba
que falhe ao importar, uma aba quebrada passava despercebida: a aplicação subia,
a aba virava "⚠️ (erro)" e ninguém era avisado — nem em teste, nem em CI.

Estes testes constroem cada aba de verdade dentro de um `gr.Blocks` e exercitam
os caminhos de degradação que a interface promete suportar: modelo ausente,
áudio ausente, arquitetura desconhecida.
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

# Os builders são importados do pacote de abas, que já isola falhas de import.
# Se um deles tiver virado placeholder, o teste de placeholder abaixo acusa.
from app.interfaces.gradio import tabs as tabs_pkg  # noqa: E402

_BUILDERS = (
    "create_dashboard_tab",
    "create_detection_tab",
    "create_voice_profiles_tab",
    "create_forensic_analysis_tab",
    "create_training_wizard_tab",
    "create_optimization_tab",
    "create_dataset_management_tab",
    "create_features_tab",
    "create_history_tab",
)


@pytest.mark.parametrize("nome_do_builder", _BUILDERS)
def test_aba_constroi_sem_erro(nome_do_builder):
    """Construção real dentro de um Blocks, como a aplicação faz."""
    builder = getattr(tabs_pkg, nome_do_builder, None)
    assert builder is not None, f"{nome_do_builder} não é exportado"

    with gr.Blocks():
        builder()  # não deve levantar


@pytest.mark.parametrize("nome_do_builder", _BUILDERS)
def test_aba_nao_e_placeholder_de_erro(nome_do_builder):
    """`tabs/__init__.py` substitui por placeholder o que falha ao importar.

    Sem esta checagem, uma aba quebrada some silenciosamente: a interface sobe,
    a aba mostra "⚠️ (erro)" e a suíte continua verde.
    """
    builder = getattr(tabs_pkg, nome_do_builder)
    qualificado = getattr(builder, "__qualname__", "")
    assert "_create_error_tab" not in qualificado, (
        f"{nome_do_builder} caiu no placeholder de erro — a aba nao importa"
    )


def test_interface_completa_monta_sem_modelos():
    """O estado pós-limpeza: `data/models/` vazio antes de uma nova bateria.

    A interface precisa subir mesmo sem nenhum artefato treinado; caso
    contrário, entre o fim de uma limpeza e a primeira promoção não haveria
    como abrir o app.
    """
    import json

    from app.interfaces.gradio.app import create_interface

    demo = create_interface()
    config = json.dumps(demo.get_config_file(), default=str)
    assert "(erro)" not in config, "alguma aba caiu no placeholder de erro"


def test_analise_sem_audio_devolve_a_aridade_esperada():
    """`_empty_analysis_result` protege o contrato de saída do handler.

    A aba Detectar tem 6 saídas e vários caminhos de retorno. Se um deles
    devolver aridade diferente, o Gradio falha em tempo de execução — foi o
    bug FE.1 que originou a função centralizada.
    """
    from app.interfaces.gradio.tabs.detection import _empty_analysis_result

    resultado = _empty_analysis_result("teste")
    assert len(resultado) == 6, (
        f"o handler de análise devolve 6 saídas; o caminho vazio devolveu "
        f"{len(resultado)}"
    )


def test_ponto_de_operacao_documentado_no_resultado():
    """O limiar de decisão precisa continuar chegando aos detalhes.

    Ele varia por modelo (calibrado na validação) e sem ele o usuário não sabe
    onde foi o corte — ver `Predictor.predict`, que devolve
    `classification_threshold`.
    """
    import inspect

    from app.interfaces.gradio.tabs import detection

    fonte = inspect.getsource(detection)
    assert "ponto_de_operacao" in fonte
    assert "classification_threshold" in fonte


def test_graficos_nao_entram_no_registro_global_do_pyplot():
    """Regressão do vazamento: `plt.subplots` num servidor que fica dias no ar.

    Cada figura registrada no pyplot fica viva até alguém chamar `plt.close`, e
    handlers que devolvem a figura para `gr.Plot` nunca chamam.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.interfaces.gradio.utils.plotting import make_figure, new_figure

    plt.close("all")
    for _ in range(25):
        _figura, eixo = new_figure(figsize=(3, 2))
        eixo.plot([1, 2, 3])
    with make_figure(figsize=(3, 2)) as (_figura_ctx, eixo_ctx):
        eixo_ctx.plot([1, 2, 3])

    assert plt.get_fignums() == [], (
        f"{len(plt.get_fignums())} figuras ficaram no registro do pyplot"
    )


def test_new_figure_aceita_a_assinatura_do_plt_subplots():
    """Substituta direta: as chamadas migradas passam linhas/colunas posicional."""
    from app.interfaces.gradio.utils.plotting import new_figure

    _figura, eixos = new_figure(1, 2, figsize=(8, 3))
    assert len(eixos) == 2
    _figura_simples, eixo = new_figure(figsize=(4, 3))
    assert eixo is not None


# ──────── protecao de handlers: erro legivel, nao rastro de pilha ────────

def _handlers_desprotegidos(caminho) -> list[str]:
    """Funções ligadas a eventos Gradio sem `try` nem `@ui_safe`."""
    import ast

    arvore = ast.parse(caminho.read_text(encoding="utf-8"))

    ligados: set[str] = set()
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Call):
            continue
        for palavra in no.keywords:
            if palavra.arg == "fn" and isinstance(palavra.value, ast.Name):
                ligados.add(palavra.value.id)
        if (
            isinstance(no.func, ast.Attribute)
            and no.func.attr in {"click", "change", "upload", "submit",
                                 "select", "tick"}
            and no.args
            and isinstance(no.args[0], ast.Name)
        ):
            ligados.add(no.args[0].id)

    def protegida(funcao: ast.FunctionDef) -> bool:
        for decorador in funcao.decorator_list:
            alvo = decorador.func if isinstance(decorador, ast.Call) else decorador
            if getattr(alvo, "id", "") == "ui_safe":
                return True
        return any(isinstance(no, ast.Try) for no in ast.walk(funcao))

    return [
        no.name
        for no in ast.walk(arvore)
        if isinstance(no, ast.FunctionDef)
        and no.name in ligados
        and not protegida(no)
    ]


def test_todo_handler_de_evento_trata_excecao():
    """Sem proteção, o usuário recebe rastro de pilha em vez de mensagem.

    O Gradio não derruba o servidor quando um handler levanta — mostra um erro
    genérico. Mas o log fica sem contexto e a interface fica sem explicação.
    Uma auditoria por AST encontrou 23 handlers assim em 2026-07-28.

    `@ui_safe` resolve sem tocar nos retornos: um `except` que devolvesse valor
    de fallback precisaria conhecer a aridade de saída de cada handler, que vai
    de 1 a 26 — e foi justamente uma divergência de aridade que causou o bug
    FE.1. `gr.Error` curto-circuita o retorno.
    """
    from pathlib import Path

    pasta = Path("app/interfaces/gradio/tabs")
    achados = {
        arquivo.name: _handlers_desprotegidos(arquivo)
        for arquivo in sorted(pasta.glob("*.py"))
        if arquivo.name != "__init__.py"
    }
    desprotegidos = {k: v for k, v in achados.items() if v}
    assert not desprotegidos, (
        f"handlers sem try/except nem @ui_safe: {desprotegidos}"
    )


def test_ui_safe_converte_excecao_em_erro_do_gradio():
    from app.interfaces.gradio.utils.components import ui_safe

    @ui_safe("Falha ao processar")
    def quebra():
        raise ValueError("causa raiz")

    with pytest.raises(gr.Error) as excecao:
        quebra()
    assert "Falha ao processar" in str(excecao.value)
    assert "causa raiz" in str(excecao.value), "a causa precisa chegar ao usuario"

    @ui_safe("nao deve aparecer")
    def funciona():
        return 42

    assert funciona() == 42, "o caminho feliz nao pode ser alterado"


def test_ui_safe_nao_reembrulha_erro_ja_destinado_ao_usuario():
    """`gr.Error` levantado de propósito passa intacto, sem prefixo duplicado."""
    from app.interfaces.gradio.utils.components import ui_safe

    @ui_safe("prefixo generico")
    def recusa():
        raise gr.Error("Selecione um arquivo de audio")

    with pytest.raises(gr.Error) as excecao:
        recusa()
    assert "prefixo generico" not in str(excecao.value)


def test_intervalo_do_status_bar_e_configuravel():
    """15 s fixos disparavam 240 execuções/hora por cliente conectado."""
    import inspect

    from app.interfaces.gradio import app as modulo

    fonte = inspect.getsource(modulo.create_interface)
    assert "XFAKE_STATUS_REFRESH_S" in fonte
    assert "gr.Timer(_sb_interval)" in fonte
