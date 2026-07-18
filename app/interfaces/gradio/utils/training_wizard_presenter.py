"""Presenters do wizard de treinamento Gradio.

Este modulo concentra HTML e navegacao visual do wizard. Ele nao executa treino,
nao acessa datasets e nao persiste modelos; apenas transforma estado de UI em
componentes renderizaveis pelo Gradio.
"""

from __future__ import annotations

from typing import List


def get_model_catalog() -> List[dict]:
    """Catálogo de arquiteturas para o Step 2 com informação amigável."""
    try:
        from app.domain.models.architectures.registry import architecture_registry

        archs = architecture_registry.list_architectures()
    except Exception:
        archs = [
            "AASIST",
            "RawGAT-ST",
            "RawNet2",
            "Sonic Sleuth",
            "WavLM",
            "HuBERT",
            "Conformer",
            "Hybrid CNN-Transformer",
            "SpectrogramTransformer",
            "EfficientNet-LSTM",
            "MultiscaleCNN",
            "Ensemble",
        ]

    catalog = {
        "AASIST": (
            "🕸️",
            "Graph Attention",
            "Spectro-temporal GAT + HS-GAL. EER 0.83% em ASVspoof. Recomendado para máxima accuracy.",
        ),
        "RawGAT-ST": (
            "🕸️",
            "Graph Attention",
            "Variante do AASIST com foco temporal. Bom para áudios curtos.",
        ),
        "RawNet2": (
            "🌊",
            "Raw audio",
            "SincNet + ResBlocks + GRU. Trabalha direto na waveform.",
        ),
        "Sonic Sleuth": (
            "🎯",
            "Lightweight",
            "Modelo leve (~3M params). 98.27% accuracy. Ideal para edge.",
        ),
        "WavLM": (
            "🤖",
            "SSL Backbone",
            "Self-supervised. Robusto a ruído e canal. Requer mais GPU.",
        ),
        "HuBERT": (
            "🤖",
            "SSL Backbone",
            "Hidden-Unit BERT. Aprende fonemas auto-supervisionado.",
        ),
        "Conformer": (
            "⚡",
            "Transformer + Conv",
            "Conv local + Self-Attention global. Estado-da-arte em speech.",
        ),
        "Hybrid CNN-Transformer": (
            "⚡",
            "Transformer + Conv",
            "CCT. CNN tokenizer + Transformer. 91.47% accuracy.",
        ),
        "SpectrogramTransformer": (
            "🔭",
            "Vision Transformer",
            "ViT adaptado para espectrogramas (AST).",
        ),
        "EfficientNet-LSTM": (
            "📊",
            "Transfer Learning",
            "EfficientNet + Bi-LSTM. Bom baseline com transfer learning.",
        ),
        "MultiscaleCNN": (
            "🔍",
            "CNN multi-escala",
            "Res2Net-50. Multi-scale hierárquico dentro do bloco residual.",
        ),
        "Ensemble": (
            "🎼",
            "Fusão multi-feature",
            "4 branches (Mel+LFCC+CQT+MFCC) + fusão. EER 3%.",
        ),
    }

    out = []
    for arch in archs:
        icon, category, desc = catalog.get(arch, ("🔧", "Outro", f"Arquitetura {arch}"))
        out.append(
            {
                "name": arch,
                "icon": icon,
                "category": category,
                "description": desc,
            }
        )

    out.append(
        {
            "name": "SVM",
            "icon": "📐",
            "category": "Classical ML",
            "description": "Support Vector Machine. Baseline rápido com features tabulares.",
        }
    )
    out.append(
        {
            "name": "Random Forest",
            "icon": "🌳",
            "category": "Classical ML",
            "description": "Ensemble de árvores. Robusto, paraleliza em CPU multi-core.",
        }
    )
    return out


def render_model_cards_html(selected: str = "") -> str:
    """HTML grid de cards de modelos, com card selecionado destacado."""
    html = '<div class="model-grid">'
    for model in get_model_catalog():
        is_selected = "model-card-selected" if model["name"] == selected else ""
        html += f"""
        <div class="model-card {is_selected}" data-arch="{model["name"]}">
            <div class="model-icon">{model["icon"]}</div>
            <div class="model-name">{model["name"]}</div>
            <div class="model-category">{model["category"]}</div>
            <div class="model-desc">{model["description"]}</div>
        </div>
        """
    html += "</div>"
    return html


def step_visibility(current: int):
    """Retorna 4 visibilidades para os 4 gr.Groups dos steps."""
    import gradio as gr

    return [gr.update(visible=(index == current)) for index in range(1, 5)]


def stepper_html(current: int) -> str:
    """Indicador visual de progresso para os passos 1-4."""
    steps = [
        ("1", "Dataset"),
        ("2", "Modelo"),
        ("3", "Hiperparâmetros"),
        ("4", "Treinar"),
    ]
    html = '<div class="wizard-stepper">'
    for index, (number, label) in enumerate(steps, start=1):
        if index < current:
            state = "done"
            icon = "✓"
        elif index == current:
            state = "active"
            icon = number
        else:
            state = "pending"
            icon = number
        html += f"""
        <div class="step step-{state}">
            <div class="step-circle">{icon}</div>
            <div class="step-label">{label}</div>
        </div>
        """
        if index < len(steps):
            html += '<div class="step-connector"></div>'
    html += "</div>"
    return html
