"""Organização e honestidade da interface: o que a tela promete é o que acontece.

Por que este arquivo existe (2026-11-02): a revisão da interface encontrou opções sem
efeito, um resultado que parecia contradizer o veredito e tempos negativos no Painel.
Nenhum teste via isso, porque os handlers funcionavam: o problema era o que a tela dizia.
"""

from __future__ import annotations

import inspect
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

gr = pytest.importorskip("gradio")


@pytest.fixture(scope="module")
def config():
    from app.interfaces.gradio.app import create_interface

    return create_interface().get_config_file()


def _abas_principais(config: dict) -> list[str]:
    componentes = {c["id"]: c for c in config["components"]}

    def achar(no):
        componente = componentes.get(no.get("id"), {})
        if componente.get("props", {}).get("elem_id") == "main_tabs":
            return no
        for filho in no.get("children") or []:
            encontrado = achar(filho)
            if encontrado:
                return encontrado
        return None

    return [
        componentes[filho["id"]]["props"]["label"]
        for filho in achar(config["layout"])["children"]
    ]


def _microfone(config: dict) -> dict:
    return next(
        componente
        for componente in config["components"]
        if componente["type"] == "audio"
        and componente["props"].get("sources") == ["microphone"]
    )


def test_gravacao_chega_inteira_na_analise(config):
    """O streaming analisava só o último pedaço de 0,5 s e falhava da segunda gravação em diante."""
    microfone = _microfone(config)["props"]
    assert microfone["streaming"] is False
    assert microfone["type"] == "filepath"
    eventos = [alvo[1] for dep in config["dependencies"] for alvo in dep["targets"]]
    assert "stream" not in eventos


@pytest.mark.parametrize(
    ("arquivo", "gravacao", "esperado", "do_microfone"),
    [
        ("a.wav", None, "a.wav", False),
        (None, "b.wav", "b.wav", True),
        ("a.wav", "b.wav", "a.wav", False),
    ],
)
def test_analise_usa_o_arquivo_ou_a_gravacao(
    monkeypatch, arquivo, gravacao, esperado, do_microfone
):
    from app.interfaces.gradio.tabs import detection

    with gr.Blocks() as demo:
        detection.create_detection_tab()
    handler = next(f.fn for f in demo.fns.values() if f.name == "handle_analysis")
    monkeypatch.setattr(
        detection,
        "analyze_audio",
        lambda caminho, modelo, janelas, gravado_no_microfone: (
            caminho,
            gravado_no_microfone,
        ),
    )
    assert handler(arquivo, gravacao, "modelo", False) == (esperado, do_microfone)


def test_cartao_do_resultado_repassa_a_origem_do_audio():
    from app.interfaces.gradio.tabs import detection

    servico = SimpleNamespace(loaded_models={})
    dados = SimpleNamespace(
        is_fake=False, confidence=0.9, model_name="bench_aasist", metadata={}
    )
    cartao = detection._cartao_do_resultado
    assert "Gravação de microfone" in cartao(servico, dados, 3.0, True)
    assert "Gravação de microfone" not in cartao(servico, dados, 3.0)


def test_analise_sem_audio_orienta_a_pessoa():
    from app.interfaces.gradio.tabs import detection

    with gr.Blocks() as demo:
        detection.create_detection_tab()
    handler = next(f.fn for f in demo.fns.values() if f.name == "handle_analysis")
    assert "Nenhum áudio enviado." in handler(None, None, "modelo", False)[0]


def test_navegacao_segue_a_ordem_de_uso(config):
    assert _abas_principais(config) == [
        "🏠 Painel",
        "🎯 Detectar",
        "🔬 Investigar",
        "🎤 Perfis de voz",
        "🎓 Treinar",
        "📊 Benchmark",
        "🗂️ Datasets",
    ]


@pytest.mark.parametrize(
    "texto",
    [
        "Sprint",  # nome interno de planejamento, sem sentido para quem usa
        "Habilitar Parâmetros Customizados",  # a caixa só liberava a escolha do modelo
        "Hiperparâmetros (JSON)",  # o conteúdo nunca era lido
        "Stochastic Weight Averaging",  # chegava ao handler e parava ali
        "Mixup",
        "Calibração de temperatura",  # caixa solta: o assistente não calibra
        "EER threshold adaptativo",
        "Confidence",
    ],
)
def test_opcoes_sem_efeito_nao_voltam_para_a_tela(config, texto):
    assert texto not in json.dumps(config, ensure_ascii=False, default=str)


def test_assistente_nao_recebe_opcoes_que_o_treino_ignora():
    from app.interfaces.gradio.tabs.training_wizard import _run_training

    parametros = inspect.signature(_run_training).parameters
    assert "use_swa" not in parametros and "use_mixup" not in parametros


# ──────── cartão do veredito ────────


def _veredito(**campos):
    from app.interfaces.gradio.presenters.verdict_card import render_verdict

    base = {"is_fake": True, "confianca": 0.97, "pontuacao": 0.97}
    return render_verdict(**{**base, **campos})


def test_veredito_compara_a_pontuacao_com_o_limite_de_50():
    fake = _veredito()
    assert "Provável voz sintética" in fake and "acima do limite de 50%" in fake
    assert "97,0%" in fake

    real = _veredito(is_fake=False, confianca=0.97, pontuacao=0.03)
    assert "Provável voz real" in real and "abaixo do limite de 50%" in real


def test_caso_relatado_no_cartao_nao_se_contradiz():
    """P(fake) 0,802 com corte em 0,941 mostrava "real" com 80,2% de sintética."""
    from app.domain.services.detection.operating_point import (
        fake_score,
        verdict_confidence,
    )

    p_fake, limiar = 0.802, 0.941
    cartao = _veredito(
        is_fake=p_fake >= limiar,
        confianca=verdict_confidence(p_fake, limiar),
        pontuacao=fake_score(p_fake, limiar),
    )
    assert "Provável voz real" in cartao and "42,6%" in cartao
    assert "abaixo do limite" in cartao and "57,4%" in cartao
    assert "80,2%" not in cartao and "19,8%" not in cartao
    assert "pouca margem" in cartao, "perto do limite, o cartão avisa"


def test_marca_do_medidor_fica_sempre_no_centro():
    assert 'class="verdict-meter-mark" style="left:50%"' in _veredito()


def test_veredito_com_pouca_margem_avisa():
    assert "pouca margem" not in _veredito()
    assert "pouca margem" in _veredito(is_fake=False, confianca=0.55, pontuacao=0.45)


def test_veredito_avisa_quando_o_audio_foge_do_treino():
    assert "fora do padrão" not in _veredito()
    assert "fora do padrão" in _veredito(fora_do_padrao=True)


@pytest.mark.parametrize("is_fake", [True, False])
def test_veredito_de_microfone_avisa_nos_dois_sentidos(is_fake):
    """Medido em 2026-11-03: o perfil de um microfone baixa a detecção do AASIST para 48%."""
    assert "Gravação de microfone" not in _veredito(is_fake=is_fake)
    cartao = _veredito(is_fake=is_fake, gravado_no_microfone=True)
    assert "Gravação de microfone" in cartao and "dois lados" in cartao


def test_veredito_sem_pontuacao_nao_desenha_medidor():
    cartao = _veredito(pontuacao=None)
    assert "limite de 50%" not in cartao and "verdict-meter-mark" not in cartao


def test_veredito_escapa_o_nome_do_modelo():
    cartao = _veredito(modelo="<img src=x onerror=alert(1)>")
    assert "<img" not in cartao and "&lt;img" in cartao


def test_erro_escapa_a_mensagem_e_e_anunciado():
    from app.interfaces.gradio.presenters.verdict_card import render_error

    cartao = render_error("falhou <b>", "tente <i>")
    assert "<b>" not in cartao and "<i>" not in cartao and 'role="alert"' in cartao


# ──────── seleção de modelo ────────


def _servico(modelos, padrao=None):
    return SimpleNamespace(get_available_models=lambda: modelos, default_model=padrao)


def test_automatico_usa_o_modelo_padrao():
    from app.interfaces.gradio.tabs.detection import AUTO_MODEL, _resolver_modelo

    assert _resolver_modelo(_servico(["a", "b"], "b"), AUTO_MODEL) == "b"
    assert _resolver_modelo(_servico(["a", "b"]), AUTO_MODEL) == "a"
    assert _resolver_modelo(_servico([]), AUTO_MODEL) is None


def test_modelo_escolhido_e_respeitado_e_nao_cai_no_padrao_em_silencio():
    from app.interfaces.gradio.tabs.detection import _resolver_modelo

    assert _resolver_modelo(_servico(["a", "b"], "a"), "b") == "b"
    assert _resolver_modelo(_servico(["a", "b"], "a"), "removido") is None


def test_nome_do_modelo_na_tela():
    from app.interfaces.gradio.tabs.detection import _nome_do_modelo

    servico = SimpleNamespace(
        loaded_models={
            "bench_aasist": SimpleNamespace(architecture="AASIST"),
            "meu_modelo": SimpleNamespace(architecture="AASIST"),
        }
    )
    assert _nome_do_modelo(servico, "bench_aasist") == "AASIST"
    assert _nome_do_modelo(servico, "meu_modelo") == "meu_modelo (AASIST)"
    assert _nome_do_modelo(servico, "desconhecido") == "desconhecido"
    # Modelos carregam sob demanda: o nome não pode depender de já estar na memória.
    assert (
        _nome_do_modelo(servico, "bench_hybrid_cnn_transformer")
        == "Hybrid CNN-Transformer"
    )
    assert _nome_do_modelo(servico, "bench_randomforest") == "RandomForest"
    assert _nome_do_modelo(servico, "bench_rawgat_st") == "RawGAT-ST"


# ──────── Painel e barra superior ────────


def test_tempo_relativo_compara_em_utc():
    from app.interfaces.gradio.tabs.dashboard import _tempo_relativo

    agora = datetime(2026, 11, 2, 12, 0, 0)
    assert _tempo_relativo(agora - timedelta(seconds=30), agora) == "agora"
    assert _tempo_relativo(agora - timedelta(minutes=5), agora) == "há 5 min"
    assert _tempo_relativo(agora - timedelta(hours=3), agora) == "há 3 h"
    assert _tempo_relativo(agora - timedelta(days=2), agora) == "há 2 d"
    # Com fuso (PostgreSQL), o mesmo instante é o mesmo tempo decorrido.
    fuso = timezone(timedelta(hours=-3))
    assert _tempo_relativo(datetime(2026, 11, 2, 8, 0, tzinfo=fuso), agora) == "há 1 h"


def test_data_do_historico_e_convertida_para_a_hora_local():
    from app.interfaces.gradio.tabs.history import _data_local

    momento = datetime(2026, 11, 2, 12, 0, 0)
    esperado = momento.replace(tzinfo=timezone.utc).astimezone()
    assert _data_local(momento) == esperado.strftime("%Y-%m-%d %H:%M:%S")


def test_barra_superior_reflete_o_estado_real(monkeypatch):
    from app.interfaces.gradio import app as modulo
    from app.interfaces.gradio.tabs import dashboard

    monkeypatch.setattr(dashboard, "_count_profiles", lambda: 0)
    monkeypatch.setattr(dashboard, "_gpu_status", lambda: "⚠ sem GPU <aviso>")

    monkeypatch.setattr(dashboard, "_count_models", lambda: 0)
    sem_modelos = modulo._render_status_bar()
    assert "Sem modelos" in sem_modelos and "Pronto" not in sem_modelos
    assert "CPU" in sem_modelos and "&lt;aviso&gt;" in sem_modelos

    monkeypatch.setattr(dashboard, "_count_models", lambda: 3)
    assert "Pronto" in modulo._render_status_bar()


def test_tema_e_painel_mudam_so_no_navegador():
    """Sem ida ao servidor para trocar o tema, e pela classe do próprio Gradio."""
    from app.interfaces.gradio import app as modulo

    assert "classList.toggle('dark'" in modulo._THEME_TOGGLE_JS
    assert "classList.toggle('open')" in modulo._NOTIF_TOGGLE_JS


def test_css_nao_usa_margem_automatica_no_conteudo_da_pagina():
    """`margin: auto` no filho flex fazia a página encolher para o conteúdo da aba."""
    from app.interfaces.gradio.app import _CUSTOM_CSS

    assert "data-theme" not in _CUSTOM_CSS
    assert ".tab-nav" not in _CUSTOM_CSS, "classe do Gradio 4, sem efeito no 6"
    assert "svelte-" not in _CUSTOM_CSS, "classe gerada muda a cada versão do Gradio"
    bloco = _CUSTOM_CSS.split(".gradio-container main.contain {", 1)[1].split("}", 1)[0]
    assert "margin: 0" in bloco and "width: 100%" in bloco


# ──────── notificações ────────


def _evento(titulo, nivel, categoria, instante=None):
    return SimpleNamespace(
        title=titulo, level=nivel, category=categoria, timestamp=instante or 1000.0
    )


@pytest.mark.parametrize(
    ("evento", "aparece"),
    [
        (_evento("Modelo salvo", "success", "event"), True),
        (_evento("Falha", "error", "event"), True),
        (_evento("Erro na inferência", "error", "log"), True),
        (_evento("request_completed", "warning", "log"), False),
        (_evento("request_completed", "error", "log"), False),
        (_evento("TensorFlow sem GPU", "warning", "log"), False),
        (_evento("Version check", "info", "log"), False),
    ],
)
def test_painel_mostra_o_que_e_para_o_usuario(evento, aparece):
    from app.interfaces.gradio.utils.notifications import _para_o_usuario

    assert _para_o_usuario(evento) is aparece


def test_contagem_do_sino_zera_ao_marcar_como_lidas(monkeypatch):
    from app.interfaces.gradio.utils import notifications as modulo

    eventos = [
        _evento("Falha", "error", "event", 2000.0),
        _evento("Aviso", "warning", "event", 2001.0),
        _evento("Salvo", "success", "event", 2002.0),
        _evento("request_completed", "warning", "log", 2003.0),
    ]
    monkeypatch.setattr(modulo, "get_feedback_events", lambda limit: eventos)
    monkeypatch.setattr(modulo, "_lido_ate", 0.0)
    monkeypatch.setattr(modulo, "mark_feedback_read", lambda: None)

    assert modulo.attention_count() == 2
    modulo.mark_all_read()
    assert modulo.attention_count() == 0


def test_cache_do_gradio_fica_no_disco_em_que_os_uploads_chegam():
    """Em outro disco o arquivo só aparece depois da resposta, e o evento seguinte falha."""
    from app.utils.file_utils import get_gradio_temp_directory

    assert get_gradio_temp_directory().parent == Path(tempfile.gettempdir())
    codigo = Path("app/interfaces/web/main_fastapi.py").read_text(encoding="utf-8")
    assert "gradio_temp_dir = get_gradio_temp_directory()" in codigo
