"""Cartão de resultado da aba Detectar: o veredito em linguagem clara.

O resultado era um `gr.Label` com a palavra "REAL" ou "DEEPFAKE" mais um campo numérico
com um parágrafo sobre temperature scaling. Quem usa a tela precisa saber o que o modelo
concluiu, com que margem e sobre qual modelo. A explicação técnica continua no JSON.

Funções puras: recebem valores e devolvem HTML, sem Gradio nem serviço.
"""

from __future__ import annotations

from html import escape
from typing import Optional

# Abaixo disto, o áudio fica perto do limiar: na validação, o acerto nessa faixa é o
# mais baixo, e cai para cerca de 60% quando P(fake) e o veredito discordam.
_POUCA_MARGEM = 0.6


def _compacto(texto: str) -> str:
    return "".join(ch for ch in texto.lower() if ch.isalnum())


def nome_do_modelo(nome: str, arquitetura: Optional[str], conhecidas=()) -> str:
    """Nome para a tela: `bench_aasist` vira `AASIST`; modelo próprio mantém o nome.

    Os modelos carregam sob demanda, então `arquitetura` costuma vir vazia. Para os
    artefatos `bench_*`, o nome da arquitetura se recupera comparando com `conhecidas`.
    """
    if nome.startswith("bench_"):
        alvo = _compacto(nome.removeprefix("bench_"))
        for candidata in (arquitetura, *conhecidas):
            if candidata and _compacto(candidata) == alvo:
                return candidata
        return arquitetura or nome
    return f"{nome} ({arquitetura})" if arquitetura else nome


def _pct(valor: float) -> str:
    """0,979 -> `97,9%`: separador decimal da interface, que é em português."""
    return f"{valor * 100:.1f}".replace(".", ",") + "%"


def render_idle() -> str:
    """Estado inicial: diz o que fazer, em vez de um "100% aguardando análise"."""
    return (
        '<div class="verdict verdict-idle">'
        "<div><strong>Nenhuma análise ainda</strong>"
        "Envie ou grave um áudio e clique em <em>Analisar áudio</em>.</div>"
        "</div>"
    )


def render_error(mensagem: str, dica: Optional[str] = None) -> str:
    """Falha com o que aconteceu e, quando há, o que fazer a seguir."""
    corpo = f"<p>{escape(dica)}</p>" if dica else ""
    return (
        '<div class="verdict verdict-error" role="alert">'
        f'<h3 class="verdict-title">{escape(mensagem)}</h3>{corpo}'
        "</div>"
    )


def render_verdict(
    *,
    is_fake: bool,
    confianca: float,
    pontuacao: Optional[float] = None,
    modelo: Optional[str] = None,
    duracao_s: Optional[float] = None,
    fora_do_padrao: bool = False,
    gravado_no_microfone: bool = False,
) -> str:
    """Veredito, pontuação centrada no limiar e o que foi usado.

    `pontuacao` é P(fake) reescalada para que 50% seja sempre o ponto de decisão do
    modelo (`detection/operating_point.py`). Assim o veredito e o número nunca apontam
    para lados opostos, e a marca do medidor fica no mesmo lugar em todos os modelos.
    Abaixo de `_POUCA_MARGEM`, o cartão avisa que a decisão foi apertada.

    `gravado_no_microfone` vem da origem do áudio, não da análise: o estudo não avaliou
    gravações de campo, e o alerta de `fora_do_padrao` não dispara nelas com segurança.
    """
    classe = "verdict-fake" if is_fake else "verdict-real"
    icone = "⚠" if is_fake else "✓"
    titulo = "Provável voz sintética" if is_fake else "Provável voz real"

    medidor = ""
    frase = ""
    if pontuacao is not None:
        preenchido = min(max(pontuacao, 0.0), 1.0) * 100
        posicao = "acima" if is_fake else "abaixo"
        frase = (
            '<p class="verdict-conf">Pontuação de voz sintética: '
            f"<strong>{_pct(pontuacao)}</strong>, {posicao} do limite de 50%.</p>"
        )
        medidor = (
            '<div class="verdict-meter" aria-hidden="true">'
            f'<div class="verdict-meter-fill" style="width:{preenchido:.1f}%"></div>'
            '<div class="verdict-meter-mark" style="left:50%"></div></div>'
        )

    itens = [("Confiança no veredito", _pct(confianca))]
    if modelo:
        itens.append(("Modelo", escape(modelo)))
    if duracao_s is not None:
        itens.append(("Duração", f"{duracao_s:.1f} s".replace(".", ",")))
    meta = (
        '<dl class="verdict-meta">'
        + "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in itens)
        + "</dl>"
    )

    avisos = ""
    if gravado_no_microfone:
        avisos += (
            '<p class="verdict-warn">⚠ Gravação de microfone: os modelos não foram '
            "validados com este tipo de áudio e podem errar para os dois lados. Não "
            "conclua nada só por este veredito.</p>"
        )
    if fora_do_padrao:
        avisos += (
            '<p class="verdict-warn">⚠ Este áudio está fora do padrão dos dados de '
            "treino. Trate o resultado com cautela.</p>"
        )
    if confianca < _POUCA_MARGEM:
        avisos += (
            '<p class="verdict-warn">⚠ Decisão com pouca margem: confirme com outro '
            "modelo ou com a aba Investigar.</p>"
        )

    return (
        f'<div class="verdict {classe}" role="status" aria-live="polite">'
        "<div>"
        f'<h3 class="verdict-title"><span aria-hidden="true">{icone}</span> {titulo}</h3>'
        f"{frase}"
        "</div>"
        f"{medidor}{avisos}{meta}"
        '<p class="verdict-note">A escala é ajustada a cada modelo: 50% é sempre o ponto '
        "de decisão. Resultado de um modelo estatístico: use como indício, não como "
        "prova.</p>"
        "</div>"
    )
