"""Componentes de UI compartilhados entre os tabs Gradio.

Centraliza padrões visuais repetidos para garantir consistência de
tipografia, espaçamento e hierarquia em TODAS as telas do sistema.

Uso típico no topo de cada `create_*_tab()`:

    from app.interfaces.gradio.utils import page_header

    page_header(
        "🎙️", "Análise de Áudio",
        "Faça upload ou grave áudio para verificar autenticidade.",
    )
"""

from __future__ import annotations

import functools
import html
import logging

import gradio as gr

logger = logging.getLogger(__name__)

__all__ = ["page_header", "section_divider", "info_callout"]


def page_header(icon: str, title: str, subtitle: str = "") -> gr.HTML:
    """Cabeçalho de página padronizado (ícone + título + subtítulo).

    Substitui o antigo padrão ad-hoc ``gr.Markdown("### Título\\ndescrição")``
    que cada aba definia por conta própria, sem consistência tipográfica.

    Args:
        icon: emoji ou caractere usado como marca visual à esquerda.
        title: título da tela (uma linha, sem markdown).
        subtitle: descrição curta opcional exibida abaixo do título.

    Returns:
        Componente ``gr.HTML`` com a classe CSS ``.page-header`` (estilizada
        em ``gradio_app.py``). Os estilos usam as variáveis ``--xf-*`` do tema,
        então respeitam automaticamente o modo claro/escuro.
    """
    safe_icon = html.escape(icon or "")
    safe_title = html.escape(title or "")
    safe_subtitle = html.escape(subtitle or "")

    subtitle_html = (
        f'<p class="ph-subtitle">{safe_subtitle}</p>' if safe_subtitle else ""
    )

    return gr.HTML(
        f"""
        <div class="page-header">
          <span class="ph-icon">{safe_icon}</span>
          <div class="ph-text">
            <h2 class="ph-title">{safe_title}</h2>
            {subtitle_html}
          </div>
        </div>
        """,
        elem_classes="page-header-wrap",
    )


def section_divider() -> gr.HTML:
    """Divisor de seção sutil — substitui ``gr.Markdown("---")`` solto.

    Usa a borda do tema (``--xf-border``) com margem vertical consistente.
    """
    return gr.HTML('<div class="xf-divider"></div>')


def info_callout(text: str, variant: str = "info") -> gr.HTML:
    """Caixa de destaque (callout) para dicas, CTAs e avisos contextuais.

    Args:
        text: conteúdo (pode conter HTML inline simples, ex.: <b>).
        variant: ``info`` | ``success`` | ``warning`` | ``accent``.

    Returns:
        ``gr.HTML`` com a classe ``.xf-callout .xf-callout-<variant>``.
    """
    variant = variant if variant in {"info", "success", "warning", "accent"} else "info"
    return gr.HTML(
        f'<div class="xf-callout xf-callout-{variant}">{text}</div>'
    )


def ui_safe(mensagem: str = "Falha ao processar a solicitação"):
    """Converte exceção não tratada de um handler em `gr.Error` legível.

    Uma auditoria por AST em 2026-07-28 encontrou 23 handlers ligados a
    `.click`/`.change` sem `try/except`. O Gradio não derruba o servidor nesse
    caso — ele mostra um erro genérico —, mas o usuário recebe rastro de pilha
    em vez de mensagem, e o log não registra contexto.

    Este decorador existe porque a alternativa óbvia não funciona: um `except`
    que devolve um valor de fallback precisaria conhecer a ARIDADE de saída de
    cada handler, que varia de 1 a 26. `gr.Error` curto-circuita o retorno, o
    Gradio o renderiza como aviso limpo, e a aridade deixa de importar — foi
    justamente uma divergência de aridade que originou o bug FE.1.

    Uso::

        @ui_safe("Não foi possível atualizar o painel")
        def refresh_dashboard():
            ...
    """
    def decorador(funcao):
        @functools.wraps(funcao)
        def envolvida(*args, **kwargs):
            try:
                return funcao(*args, **kwargs)
            except gr.Error:
                raise  # já é uma mensagem destinada ao usuário
            except Exception as exc:  # noqa: BLE001 — superfície de UI
                logger.exception("%s: %s", funcao.__qualname__, exc)
                raise gr.Error(f"{mensagem}: {exc}") from exc

        return envolvida

    return decorador
