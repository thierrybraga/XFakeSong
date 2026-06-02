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

import html

import gradio as gr

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
