"""
Aba de Gestão de Dataset — XFakeSong

Centraliza visualização, download, preprocessamento e análise de
compatibilidade dos datasets de áudio para detecção de deepfake.
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import gradio as gr

from app.core.dataset_catalog import (
    PRESET_SELECTIONS,
    dataset_reference_markdown,
    get_tier,
    prefix_to_dataset,
    source_type_map,
    tier_reference_markdown,
)

matplotlib.use("Agg")

logger = logging.getLogger("gradio_dataset_management")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
DATASETS_DIR = BASE_DIR / "app" / "datasets"
REAL_DIR = DATASETS_DIR / "real"
FAKE_DIR = DATASETS_DIR / "fake"
SPLITS_DIR = DATASETS_DIR / "splits"

# Adiciona raiz ao path para importar scripts/
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------------------------
# Dark theme tokens (idênticos a voice_profiles.py)
# ---------------------------------------------------------------------------
_BG = "#0f172a"
_FACE = "#1e293b"
_TEXT = "#f1f5f9"
_GRID = "#334155"
_ACCENT = "#3b82f6"
_ACCENT2 = "#06b6d4"
_DANGER = "#ef4444"
_SUCCESS = "#22c55e"
_WARNING = "#f59e0b"

_SOURCE_COLORS = {
    "BRSpeech-DF":   "#3b82f6",
    "Fake Voices":   "#8b5cf6",
    "FLEURS":        "#06b6d4",
    "Common Voice PT": "#f59e0b",
    "CETUC":         "#10b981",
    "MLAAD-PT":      "#e879f9",
    "ASVspoof 2019": "#f97316",
    "ASVspoof 5":    "#fb923c",
    "WaveFake":      "#a855f7",
    "In-the-Wild":   "#14b8a6",
    "Synthetic":     "#6b7280",
    "Outros":        "#94a3b8",
}

# ---------------------------------------------------------------------------
# Balance-aware download helpers
# ---------------------------------------------------------------------------

# Classifica cada fonte quanto ao tipo de áudio que produz
_SOURCE_TYPE: Dict[str, str] = source_type_map()

# Estimativa de amostras por falante para Fake Voices (max_per_speaker=100 padrão)
_FAKE_VOICES_SAMPLES_PER_SPEAKER = 80   # conservador para evitar overcount

# Combinações pré-definidas para facilitar o uso
_PRESET_SELECTIONS: Dict[str, List[str]] = PRESET_SELECTIONS


def _compute_download_plan(
    selected: List[str],
    current_real: int,
    current_fake: int,
    target: int,
) -> Tuple[Dict[str, Dict], int, int]:
    """Calcula alocação balanceada entre as fontes selecionadas.

    Algoritmo:
    - Calcula gap = max(0, target - atual) para cada classe
    - Distribui o gap proporcionalmente entre as fontes que produzem essa classe
    - Fontes "both" participam de ambos os pools
    - Retorna plano {source: {max_samples, real_alloc, fake_alloc, type}}

    Returns:
        (plan, real_needed, fake_needed)
    """
    real_needed = max(0, target - current_real)
    fake_needed = max(0, target - current_fake)

    real_providers = [s for s in selected if _SOURCE_TYPE.get(s) in ("real", "both")]
    fake_providers = [s for s in selected if _SOURCE_TYPE.get(s) in ("fake", "both")]

    plan: Dict[str, Dict] = {}
    for source in selected:
        src_type = _SOURCE_TYPE.get(source, "both")
        real_alloc = 0
        fake_alloc = 0

        if src_type in ("real", "both") and real_providers and real_needed > 0:
            real_alloc = max(1, real_needed // len(real_providers))

        if src_type in ("fake", "both") and fake_providers and fake_needed > 0:
            fake_alloc = max(1, fake_needed // len(fake_providers))

        # max_samples a passar para cada função de download:
        # • fontes "real":  max_samples = real_alloc  (total real a baixar)
        # • fontes "fake":  max_samples = fake_alloc  (total fake a baixar)
        # • fontes "both":  usam max_samples // 2 por classe internamente,
        #                   então passamos 2 * max(alloc_real, alloc_fake)
        if src_type == "real":
            max_s = real_alloc
        elif src_type == "fake":
            max_s = fake_alloc
        else:
            # garante que ambas as classes atinjam suas alocações
            max_s = max(real_alloc, fake_alloc) * 2

        plan[source] = {
            "max_samples": max_s,
            "real_alloc":  real_alloc,
            "fake_alloc":  fake_alloc,
            "type":        src_type,
        }

    return plan, real_needed, fake_needed


def _format_plan_md(
    plan: Dict,
    real_needed: int,
    fake_needed: int,
    current_real: int,
    current_fake: int,
    target: int,
) -> str:
    """Formata o plano de download como markdown com tabela e estimativa final."""
    lines = [
        "#### Plano de Download Balanceado",
        "",
        f"**Estado atual:** {current_real:,} real · {current_fake:,} fake  ",
        f"**Alvo por classe:** {target:,}  ",
        f"**Gap a preencher:** +{real_needed:,} real · +{fake_needed:,} fake",
        "",
        "| Fonte | Tipo | Real (+) | Fake (+) | max_samples |",
        "|-------|:----:|:--------:|:--------:|:-----------:|",
    ]

    total_r = 0
    total_f = 0
    for source, info in plan.items():
        r = info["real_alloc"]
        f = info["fake_alloc"]
        total_r += r
        total_f += f
        if info["max_samples"] <= 0:
            lines.append(f"| {source} | {info['type']} | ⏭ 0 | ⏭ 0 | 0 |")
        else:
            lines.append(
                f"| **{source}** | `{info['type']}` | +{r:,} | +{f:,} | {info['max_samples']:,} |"
            )

    exp_real  = current_real  + total_r
    exp_fake  = current_fake  + total_f
    exp_ratio = exp_real / max(exp_fake, 1)

    if 0.8 <= exp_ratio <= 1.25:
        bal_tag = "✅ Balanceado"
    elif 0.5 <= exp_ratio <= 2.0:
        bal_tag = "⚠️ Aceitável (class_weight compensará)"
    else:
        bal_tag = "❌ Desbalanceado — considere adicionar mais fontes"

    lines += [
        "",
        f"**Resultado esperado:** ~{exp_real:,} real · ~{exp_fake:,} fake  ",
        f"**Ratio estimado:** {exp_ratio:.2f} — {bal_tag}",
        "",
        "> ℹ️ Fontes `both` baixam real **e** fake. Eventual excesso não prejudica o treino "
        "— class_weight balanceia automaticamente.",
    ]
    return "\n".join(lines)


def _balance_bar_html(real_count: int, fake_count: int) -> str:
    """Gera HTML com barra de balanço real/fake e KPIs."""
    total = max(real_count + fake_count, 1)
    real_pct = real_count / total * 100
    fake_pct = 100.0 - real_pct
    ratio = real_count / max(fake_count, 1)

    if real_count == 0 and fake_count == 0:
        badge, bcolor = "📭 Dataset vazio", "#94a3b8"
    elif 0.8 <= ratio <= 1.25:
        badge, bcolor = "✅ Balanceado", _SUCCESS
    elif 0.5 <= ratio <= 2.0:
        badge, bcolor = "⚠️ Aceitável", _WARNING
    else:
        badge, bcolor = "❌ Desbalanceado", _DANGER

    return (
        f'<div style="background:{_FACE};border:1px solid {_GRID};border-radius:8px;'
        f'padding:12px 14px;font-family:monospace;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
        f'<span style="color:{_TEXT};font-weight:bold;font-size:13px;">Estado do Dataset</span>'
        f'<span style="color:{bcolor};font-weight:bold;font-size:13px;">{badge}</span>'
        f'</div>'
        f'<div style="background:{_BG};border-radius:4px;height:16px;overflow:hidden;'
        f'display:flex;margin-bottom:8px;">'
        f'<div style="width:{real_pct:.1f}%;background:{_SUCCESS};transition:width .4s;"></div>'
        f'<div style="width:{fake_pct:.1f}%;background:{_DANGER};transition:width .4s;"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'color:{_TEXT};font-size:12px;">'
        f'<span>🟢 Real: <b>{real_count:,}</b> ({real_pct:.0f}%)</span>'
        f'<span style="color:{_GRID};">Ratio: <b style="color:{bcolor};">{ratio:.2f}</b></span>'
        f'<span>🔴 Fake: <b>{fake_count:,}</b> ({fake_pct:.0f}%)</span>'
        f'</div>'
        f'</div>'
    )


def _assess_training_readiness(real_count: int, fake_count: int) -> str:
    """Avalia prontidão do dataset para cada tier de modelo e retorna markdown.

    Thresholds empíricos baseados em literatura anti-spoofing (ASVspoof):
    - ML clássico:  300+ por classe
    - CNN leve:    1 000+ por classe
    - CNN/RNN:     2 000+ por classe
    - Transformer: 4 000+ por classe (underfitting severo abaixo disso)
    - Ensemble:    6 000+ por classe (4 branches)
    """
    per_class = min(real_count, fake_count)
    total     = real_count + fake_count
    ratio     = real_count / max(fake_count, 1)

    lines: List[str] = ["#### Prontidão para Treinamento", ""]

    if per_class == 0:
        return "\n".join(lines + [
            "❌ **Dataset vazio.** Inicie um download para continuar.",
        ])

    # ── Diagnóstico de balanceamento ──────────────────────────────────
    if real_count == 0 or fake_count == 0:
        lines.append("❌ **Uma das classes está vazia** — impossível treinar.")
    elif ratio < 0.5 or ratio > 2.0:
        deficit_cls = "real" if real_count < fake_count else "fake"
        diff = abs(real_count - fake_count)
        lines.append(
            f"❌ **Desbalanceamento severo** (ratio={ratio:.2f}) — faltam {diff:,} "
            f"amostras `{deficit_cls}`. Corrija antes de usar modelos DL."
        )
    elif 0.8 <= ratio <= 1.25:
        lines.append(f"✅ **Balanceamento ideal** (ratio={ratio:.2f}) — pronto para qualquer modelo.")
    else:
        lines.append(
            f"⚠️ **Balanceamento aceitável** (ratio={ratio:.2f}) — "
            "`class_weight=balanced` será aplicado automaticamente no treino."
        )

    lines.append("")
    lines.append("| Tier | Modelos | Mín/classe | Status |")
    lines.append("|------|---------|:----------:|:------:|")

    _TIERS = [
        ("Clássico",    "SVM, Random Forest",                              300),
        ("CNN Leve",    "RawNet2, Sonic Sleuth, MultiscaleCNN",            1_000),
        ("CNN/RNN",     "WavLM, HuBERT, EfficientNet-LSTM, RawGAT-ST",   2_000),
        ("Transformer", "Conformer, AASIST, SpectrogramTransformer",      4_000),
        ("Ensemble",    "Ensemble, Hybrid CNN-Transformer",               6_000),
    ]

    for tier_name, models, required in _TIERS:
        if per_class >= required:
            icon = "✅"
            note = f"Pronto ({per_class:,}/{required:,})"
        else:
            needed = required - per_class
            pct = per_class / required * 100
            icon = "🔶" if pct >= 50 else "❌"
            note = f"+{needed:,} necessários ({pct:.0f}%)"
        lines.append(f"| {icon} **{tier_name}** | {models} | {required:,} | {note} |")

    lines += [
        "",
        f"**Resumo:** {per_class:,} amostras/classe · {total:,} total "
        f"· ~{total * 3 / 3600:.1f}h estimado (avg 3s/amostra)",
    ]

    # ── Próximo passo recomendado ─────────────────────────────────────
    if per_class < 300:
        rec = "⚡ Baixe pelo menos 300/classe antes de treinar qualquer modelo."
    elif per_class < 1_000:
        rec = "📈 Você pode treinar SVM/RF agora. Baixe mais 1 000/classe para CNNs."
    elif per_class < 2_000:
        rec = "📈 CNNs leves estão prontas. +1 000/classe desbloqueia WavLM/HuBERT."
    elif per_class < 4_000:
        rec = "📈 CNN/RNN prontos. +2 000/classe para performance máxima em Transformers."
    elif per_class < 6_000:
        rec = "📈 Transformers habilitados. +2 000/classe para Ensemble completo."
    else:
        rec = "🚀 Dataset completo — todos os 14 modelos habilitados."
    lines += ["", f"> {rec}"]

    return "\n".join(lines)


def _style_ax(ax):
    """Aplica tema dark a um eixo matplotlib."""
    ax.set_facecolor(_FACE)
    ax.figure.set_facecolor(_BG)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(True, color=_GRID, alpha=0.3, linewidth=0.5)


# ---------------------------------------------------------------------------
# Dataset Scanner
# ---------------------------------------------------------------------------
def _scan_dataset(light: bool = False) -> Dict[str, Any]:
    """Escaneia o dataset e retorna estatísticas.

    Args:
        light: Se True, pula amostragem de durações (sf.info em 150 arquivos
               por classe). Use durante loops de download/progresso onde só os
               contadores importam — evita centenas de leituras de disco.
    """
    import soundfile as sf

    data: Dict[str, Any] = {
        "real_count": 0,
        "fake_count": 0,
        "real_duration_h": 0.0,
        "fake_duration_h": 0.0,
        "sources": {},
        "splits_exist": False,
        "splits_metadata": None,
        "durations_real": [],
        "durations_fake": [],
    }

    # Prefixos conhecidos -> nome de fonte, mantidos no catalogo central.
    prefix_map = prefix_to_dataset()
    prefix_map["synthetic"] = "Synthetic"

    for label, directory in [("real", REAL_DIR), ("fake", FAKE_DIR)]:
        if not directory.exists():
            continue
        wav_files = sorted(directory.glob("*.wav"))
        count = len(wav_files)
        data[f"{label}_count"] = count

        # Contar por fonte
        for wf in wav_files:
            name = wf.stem
            matched = False
            for prefix, source in prefix_map.items():
                if name.startswith(prefix):
                    key = f"{source} ({label})"
                    data["sources"][key] = data["sources"].get(key, 0) + 1
                    matched = True
                    break
            if not matched:
                key = f"Outros ({label})"
                data["sources"][key] = data["sources"].get(key, 0) + 1

        # Amostrar durações (max 150 arquivos) — pulado no modo light
        if light:
            continue
        sample_files = wav_files[:150]
        durations = []
        for wf in sample_files:
            try:
                info = sf.info(str(wf))
                durations.append(info.duration)
            except Exception as e:
                # FE.7: log em debug para não inundar console em datasets grandes
                logger.debug(f"sf.info falhou em {wf.name}: {e}")

        data[f"durations_{label}"] = durations
        if durations:
            avg_dur = np.mean(durations)
            data[f"{label}_duration_h"] = (avg_dur * count) / 3600.0

    # Splits
    meta_path = SPLITS_DIR / "splits_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                data["splits_metadata"] = json.load(f)
            data["splits_exist"] = True
        except Exception as e:
            # FE.7: log do erro real ao invés de silenciar
            logger.warning(f"Falha ao ler splits_metadata.json: {e}")

    return data


# ---------------------------------------------------------------------------
# Plot Builders
# ---------------------------------------------------------------------------
def _build_class_distribution(data: Dict) -> plt.Figure:
    """Bar chart: real vs fake."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _style_ax(ax)

    real_c = data["real_count"]
    fake_c = data["fake_count"]
    bars = ax.bar(
        ["Real", "Fake"],
        [real_c, fake_c],
        color=[_SUCCESS, _DANGER],
        width=0.5,
        edgecolor=_GRID,
        linewidth=0.5,
    )
    for bar, val in zip(bars, [real_c, fake_c]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(real_c, fake_c) * 0.02,
            str(val),
            ha="center",
            va="bottom",
            color=_TEXT,
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Amostras")
    ax.set_title("Distribuição por Classe")
    fig.tight_layout()
    return fig


def _build_source_pie(data: Dict) -> plt.Figure:
    """Pie chart: fontes de dados."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.set_facecolor(_BG)
    fig.set_facecolor(_BG)

    sources = data["sources"]
    if not sources:
        ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", color=_TEXT, fontsize=14)
        return fig

    labels = list(sources.keys())
    sizes = list(sources.values())

    # Cores por fonte base
    colors = []
    for lbl in labels:
        base = lbl.split(" (")[0]
        colors.append(_SOURCE_COLORS.get(base, "#94a3b8"))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
        textprops={"color": _TEXT, "fontsize": 8},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("#ffffff")

    ax.set_title("Fontes de Dados", color=_TEXT, fontsize=11)
    fig.tight_layout()
    return fig


def _build_duration_histogram(data: Dict) -> plt.Figure:
    """Histograma de durações sobrepostas."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _style_ax(ax)

    dr = data["durations_real"]
    df = data["durations_fake"]

    if dr:
        ax.hist(dr, bins=30, alpha=0.6, color=_SUCCESS, label=f"Real (n={len(dr)})", edgecolor=_GRID, linewidth=0.3)
    if df:
        ax.hist(df, bins=30, alpha=0.6, color=_DANGER, label=f"Fake (n={len(df)})", edgecolor=_GRID, linewidth=0.3)

    ax.set_xlabel("Duração (s)")
    ax.set_ylabel("Frequencia")
    ax.set_title("Distribuição de Durações")
    ax.legend(facecolor=_FACE, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Log Capture Helper
# ---------------------------------------------------------------------------
class _LogCapture(logging.Handler):
    """Handler que captura logs em uma lista."""

    def __init__(self):
        super().__init__()
        self.records: List[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        self.records.append(self.format(record))

    def text(self) -> str:
        return "\n".join(self.records[-100:])


# ---------------------------------------------------------------------------
# Compatibility Analyzer
# ---------------------------------------------------------------------------
def _analyze_compatibility(data: Dict) -> Tuple[List[List], str]:
    """Analisa compatibilidade do dataset com cada arquitetura."""
    try:
        from app.domain.models.architectures.registry import ArchitectureRegistry
        registry = ArchitectureRegistry()
    except Exception as e:
        # FE.7: log do motivo da falha (útil para debug)
        logger.warning(f"ArchitectureRegistry indisponível: {e}")
        registry = None

    total = data["real_count"] + data["fake_count"]
    ratio = data["real_count"] / max(data["fake_count"], 1)
    has_splits = data["splits_exist"]

    rows = []
    details_parts = []

    # Arquiteturas DL
    dl_archs = [
        ("AASIST", "aasist", "Attention"),
        ("RawGAT-ST", "rawgat_st", "Graph Attention"),
        ("RawNet2", "rawnet2", "Conv1D"),
        ("Sonic Sleuth", "sonic_sleuth", "Conv1D"),
        ("EfficientNet-LSTM", "efficientnet_lstm", "Recurrent"),
        ("WavLM", "wavlm", "Recurrent"),
        ("HuBERT", "hubert", "Recurrent"),
        ("Conformer", "conformer", "Transformer"),
        ("SpectrogramTransformer", "spectrogram_transformer", "Transformer"),
        ("MultiscaleCNN", "multiscale_cnn", "Multi-head"),
        ("Ensemble", "ensemble", "Multi-head"),
        ("Hybrid CNN-Transformer", "hybrid_cnn_transformer", "Multi-head"),
    ]

    for display_name, snake_name, arch_type in dl_archs:
        issues = []
        recs = []

        # Quantidade mínima
        if total < 200:
            issues.append("Poucos dados")
            recs.append(f"Recomendado >500 amostras por classe. Atual: {data['real_count']} real + {data['fake_count']} fake")
        elif total < 1000:
            recs.append("Dataset pequeno — considere data augmentation ou mais downloads")

        # Balanceamento
        if ratio < 0.5:
            issues.append("Desbalanceado")
            recs.append(f"Ratio real/fake = {ratio:.2f}. Baixe mais amostras reais (BRSpeech-DF bonafide ou FLEURS)")
        elif ratio > 2.0:
            issues.append("Desbalanceado")
            recs.append(f"Ratio real/fake = {ratio:.2f}. Baixe mais amostras fake (Fake Voices)")

        # Splits
        if not has_splits:
            issues.append("Sem splits")
            recs.append("Execute 'Criar Splits' na aba Pré-processamento")

        # Sample rate (todos esperam 16kHz — já garantido pelo pipeline)
        # Duração (verificar se há amostras muito curtas)
        all_durs = data["durations_real"] + data["durations_fake"]
        if all_durs:
            short = sum(1 for d in all_durs if d < 2.0)
            if short > len(all_durs) * 0.1:
                recs.append(f"{short}/{len(all_durs)} amostras <2s — podem reduzir performance de modelos recorrentes")

        # Input requirements do registry
        if registry:
            try:
                arch_info = registry.get_architecture(display_name)
                if arch_info and arch_info.input_requirements:
                    req = arch_info.input_requirements
                    if "sample_rate" in req and req["sample_rate"] != 16000:
                        issues.append(f"SR esperado: {req['sample_rate']}")
                        recs.append(f"Arquitetura espera {req['sample_rate']}Hz, dataset esta em 16kHz")
            except Exception as e:
                # FE.7: log do nome da arquitetura para facilitar debug
                logger.debug(f"Falha ao ler input_requirements de {display_name}: {e}")

        status = "OK" if not issues else " | ".join(issues)
        status_icon = "✅" if not issues else ("⚠️" if "Desbalanceado" in status else "❌")
        rec_text = "; ".join(recs) if recs else "Pronto para treinamento"

        rows.append([f"{status_icon} {display_name}", arch_type, status, rec_text])

        if recs:
            details_parts.append(f"### {display_name} ({arch_type})\n" + "\n".join(f"- {r}" for r in recs))

    # Classical ML
    for name, ml_type in [("SVM", "RBF Kernel"), ("Random Forest", "Ensemble Trees")]:
        issues = []
        recs = []
        if total < 50:
            issues.append("Poucos dados")
            recs.append("ML classico precisa de pelo menos 50 amostras por classe")
        if ratio < 0.5 or ratio > 2.0:
            issues.append("Desbalanceado")
            recs.append(f"Ratio {ratio:.2f}. class_weight='balanced' sera aplicado automaticamente")
        if not has_splits:
            issues.append("Sem splits")
            recs.append("Execute 'Criar Splits'")

        status = "OK" if not issues else " | ".join(issues)
        status_icon = "✅" if not issues else "⚠️"
        rec_text = "; ".join(recs) if recs else "Pronto para treinamento"
        rows.append([f"{status_icon} {name}", f"Classical ML ({ml_type})", status, rec_text])

        if recs:
            details_parts.append(f"### {name} ({ml_type})\n" + "\n".join(f"- {r}" for r in recs))

    details_md = "\n\n".join(details_parts) if details_parts else "Todas as arquiteturas estao compativeis com o dataset atual."
    return rows, details_md


# ===================================================================
# MAIN TAB CREATION
# ===================================================================
def create_dataset_management_tab():
    """Cria a aba de Gestao de Dataset."""

    with gr.Tab("📁 Datasets", id="tab_dataset_mgmt"):
        from app.interfaces.gradio.utils.components import page_header

        page_header(
            "📁",
            "Gestão de Datasets",
            "Visualize, baixe, pré-processe e analise a compatibilidade dos "
            "datasets de áudio.",
        )

        with gr.Tabs():
            # ===========================================================
            # SUB-TAB 1: VISÃO GERAL
            # ===========================================================
            with gr.Tab("Visão Geral", id="tab_ds_overview"):

                with gr.Row():
                    kpi_md = gr.Markdown("*Clique em Atualizar para carregar...*")

                with gr.Row():
                    with gr.Column(scale=1):
                        plot_class = gr.Plot(label="Distribuição por Classe")
                    with gr.Column(scale=1):
                        plot_source = gr.Plot(label="Fontes de Dados")

                with gr.Row():
                    with gr.Column(scale=1):
                        plot_duration = gr.Plot(label="Distribuição de Durações")
                    with gr.Column(scale=1):
                        splits_df = gr.Dataframe(
                            headers=["Split", "Real", "Fake", "Total"],
                            label="Resumo dos Splits",
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        meta_json = gr.JSON(label="Metadata (splits_metadata.json)")
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("Atualizar Dashboard", variant="primary", size="lg")

                def handle_refresh():
                    data = _scan_dataset()

                    real_c = data["real_count"]
                    fake_c = data["fake_count"]
                    total = real_c + fake_c
                    ratio = real_c / max(fake_c, 1)
                    real_h = data["real_duration_h"]
                    fake_h = data["fake_duration_h"]
                    splits_ok = "Sim" if data["splits_exist"] else "Nao"

                    kpi = (
                        f"| Metrica | Valor |\n"
                        f"|---------|-------|\n"
                        f"| Total de Arquivos | **{total:,}** |\n"
                        f"| Amostras Reais | **{real_c:,}** |\n"
                        f"| Amostras Fake | **{fake_c:,}** |\n"
                        f"| Ratio Real/Fake | **{ratio:.2f}** |\n"
                        f"| Duração Real | **{real_h:.1f}h** |\n"
                        f"| Duração Fake | **{fake_h:.1f}h** |\n"
                        f"| Duração Total | **{real_h + fake_h:.1f}h** |\n"
                        f"| Splits Criados | **{splits_ok}** |\n"
                        f"| Sample Rate | **16 kHz** |\n"
                        f"| Formato | **WAV PCM 16-bit mono** |"
                    )

                    # Tier + protocolo de falante (quando o dataset foi montado
                    # por tier e/ou os splits guardam estatística de falante).
                    meta_overview = data.get("splits_metadata") or {}
                    spk = meta_overview.get("speakers") or {}
                    strategy = meta_overview.get("split_strategy")
                    extra = ""
                    if strategy:
                        extra += f"\n| Estratégia de Split | **{strategy}** |"
                    if spk:
                        extra += (
                            f"\n| Falantes (total) | **{spk.get('total', '—')}** |"
                            f"\n| Falantes identificados | **{spk.get('identified', '—')}** |"
                            f"\n| Falantes não vistos no teste | **{spk.get('unseen_in_test', '—')}** |"
                        )
                    kpi += extra

                    fig_class = _build_class_distribution(data)
                    fig_source = _build_source_pie(data)
                    fig_dur = _build_duration_histogram(data)

                    # Splits table
                    splits_rows = []
                    meta = data["splits_metadata"]
                    if meta:
                        for split_name in ["train", "val", "test"]:
                            split_dir = SPLITS_DIR / split_name
                            r_count = len(list((split_dir / "real").glob("*.wav"))) if (split_dir / "real").exists() else 0
                            f_count = len(list((split_dir / "fake").glob("*.wav"))) if (split_dir / "fake").exists() else 0
                            splits_rows.append([split_name.capitalize(), r_count, f_count, r_count + f_count])
                    else:
                        splits_rows = [["Nenhum split criado", 0, 0, 0]]

                    return kpi, fig_class, fig_source, fig_dur, splits_rows, meta or {}

                refresh_btn.click(
                    fn=handle_refresh,
                    outputs=[kpi_md, plot_class, plot_source, plot_duration, splits_df, meta_json],
                )

            # ===========================================================
            # SUB-TAB 2: DOWNLOAD (balance-aware)
            # ===========================================================
            with gr.Tab("Download", id="tab_ds_download"):

                gr.Markdown("### Download e Balanceamento de Datasets")

                # ── Barra de balanço em tempo real ─────────────────────
                with gr.Row():
                    with gr.Column(scale=4):
                        dl_balance_html = gr.HTML(value=_balance_bar_html(0, 0))
                    with gr.Column(scale=1, min_width=120):
                        dl_bal_refresh = gr.Button("↻ Atualizar", size="sm")

                with gr.Row():
                    # ── Painel de configuração (esquerda) ───────────────
                    with gr.Column(scale=1):

                        with gr.Group():
                            gr.Markdown("**1. Escolher Tier**")
                            dl_tier = gr.Radio(
                                choices=[
                                    ("Test — smoke (100/classe)", "test"),
                                    ("Small — rápido (1.000/classe)", "small"),
                                    ("Medium — completo (3.000/classe)", "medium"),
                                    ("Large — falantes não vistos (10.000/classe)", "large"),
                                ],
                                value=None,
                                label="Tier de dataset",
                                info="Pré-configura tamanho, fontes e protocolo de split (ou configure manualmente abaixo)",
                            )
                            dl_tier_desc = gr.Markdown(
                                "*Selecione um tier para pré-configurar tamanho e "
                                "fontes, ou ajuste manualmente nos passos abaixo.*"
                            )

                        with gr.Group():
                            gr.Markdown("**2. Selecionar Fontes**")
                            dl_preset = gr.Dropdown(
                                choices=["— preset —"] + list(_PRESET_SELECTIONS.keys()),
                                value="— preset —",
                                label="Preset",
                                container=False,
                            )
                            dl_sources = gr.CheckboxGroup(
                                choices=list(_SOURCE_TYPE.keys()),
                                value=["BRSpeech-DF", "Fake Voices"],
                                label="Fontes de dados",
                            )

                        with gr.Group():
                            gr.Markdown("**3. Configurar Alvo de Balanceamento**")
                            dl_target = gr.Slider(
                                minimum=100,
                                maximum=10_000,
                                value=1_000,
                                step=100,
                                label="Alvo por classe (real e fake)",
                                info="Mínimo de amostras reais E fakes desejado após o download",
                            )
                            dl_speakers_override = gr.Slider(
                                minimum=0,
                                maximum=50,
                                value=0,
                                step=5,
                                label="Max falantes Fake Voices (0 = auto calculado pelo alvo)",
                                visible=True,
                            )

                        gr.Markdown("---")
                        dl_plan_btn = gr.Button("📋 Calcular Plano", variant="secondary", size="sm")
                        dl_btn      = gr.Button("⬇️ Iniciar Download Balanceado", variant="primary")

                    # ── Plano + readiness + log (direita) ──────────────
                    with gr.Column(scale=2):
                        dl_plan_md = gr.Markdown(
                            "*Selecione as fontes e clique em **Calcular Plano** para ver a "
                            "distribuição balanceada antes de baixar.*"
                        )
                        dl_readiness_md = gr.Markdown(
                            "*Aguardando análise...*",
                            label="Prontidão para Treinamento",
                        )
                        dl_log = gr.Textbox(
                            label="Log de Download",
                            lines=10,
                            max_lines=18,
                            interactive=False,
                        )
                        dl_status = gr.Markdown("*Aguardando...*")

                with gr.Accordion("Tiers de Dataset (test/small/medium/large)", open=False):
                    gr.Markdown(
                        tier_reference_markdown()
                        + "\n\n"
                        "> Os tiers são a fonte única de verdade do tamanho/finalidade "
                        "do dataset (`app/core/dataset_catalog.py`), compartilhada com "
                        "`scripts/build_dataset.py --tier ...`, o benchmark e a "
                        "documentação. Detalhes em `docs/12_DATASETS.md`."
                    )

                with gr.Accordion("Referência de Datasets", open=False):
                    gr.Markdown(
                        dataset_reference_markdown()
                        + "\n\n"
                        "> **Dica de balanceamento:** combine sempre pelo menos 1 fonte `both` "
                        "(BRSpeech-DF, ASVspoof 2019) com fontes especializadas "
                        "(Fake Voices para fakes, CETUC para reais) para máxima diversidade."
                    )

                # ── Handlers ───────────────────────────────────────────

                def _dl_refresh_balance():
                    data = _scan_dataset()
                    return (
                        _balance_bar_html(data["real_count"], data["fake_count"]),
                        _assess_training_readiness(data["real_count"], data["fake_count"]),
                    )

                def _dl_apply_preset(preset_name: str):
                    if preset_name in _PRESET_SELECTIONS:
                        return gr.update(value=_PRESET_SELECTIONS[preset_name])
                    return gr.update()

                def _dl_apply_tier(tier_name: str):
                    """Aplica um tier: alvo por classe + fontes + descrição.

                    A mudança em dl_sources cascateia para recalcular o plano
                    automaticamente (mesmo padrão do preset).
                    """
                    tier = get_tier(tier_name) if tier_name else None
                    if tier is None:
                        return gr.update(), gr.update(), gr.update()
                    split_txt = (
                        "split **disjunto por falante** (usuários não vistos) + "
                        "protocolo cross-generator"
                        if tier.speaker_aware
                        else "split 70/15/15 estratificado"
                    )
                    note = ""
                    if tier.speaker_aware:
                        note = (
                            "\n\n> 🔬 **Protocolo de usuários não vistos:** o `large` "
                            "identifica falantes (`speaker_manifest.json`) e mantém "
                            "cada falante inteiramente em treino **ou** teste. No "
                            "benchmark use `--speaker-split` ou `--unseen-speaker`."
                        )
                    desc = (
                        f"**Tier `{tier.name}`** — {tier.purpose}\n\n"
                        f"{tier.description}\n\n"
                        f"**{tier.per_class:,}/classe** ({tier.total:,} total) · "
                        f"fontes: {', '.join(tier.sources)} · {split_txt}.\n\n"
                        f"_Habilita:_ {tier.models_enabled}{note}"
                    )
                    return (
                        gr.update(value=tier.per_class),
                        gr.update(value=list(tier.sources)),
                        gr.update(value=desc),
                    )

                def _dl_on_sources_change(selected_sources, target):
                    """Handler único disparado quando as fontes mudam (manual OU via preset).

                    Atualiza: visibilidade do slider de falantes + barra de balanço +
                    plano + prontidão. Assim o usuário vê o plano recalculado sem
                    precisar clicar em "Calcular Plano" manualmente.
                    """
                    show_speakers = gr.update(
                        visible="Fake Voices" in (selected_sources or [])
                    )
                    # light=True: só precisamos dos contadores, não das durações
                    data = _scan_dataset(light=True)
                    bar = _balance_bar_html(data["real_count"], data["fake_count"])

                    if not selected_sources:
                        return (
                            show_speakers,
                            bar,
                            "*Selecione fontes para ver o plano balanceado.*",
                            _assess_training_readiness(
                                data["real_count"], data["fake_count"]
                            ),
                        )

                    plan, rn, fn = _compute_download_plan(
                        selected_sources,
                        data["real_count"], data["fake_count"],
                        int(target),
                    )
                    plan_text = _format_plan_md(
                        plan, rn, fn,
                        data["real_count"], data["fake_count"], int(target),
                    )
                    exp_real = data["real_count"] + sum(v["real_alloc"] for v in plan.values())
                    exp_fake = data["fake_count"] + sum(v["fake_alloc"] for v in plan.values())
                    readiness = (
                        "**Após este download (estimado):**\n\n"
                        + _assess_training_readiness(exp_real, exp_fake)
                    )
                    return show_speakers, bar, plan_text, readiness

                def _dl_compute_plan(selected_sources, target):
                    data = _scan_dataset()
                    readiness = _assess_training_readiness(
                        data["real_count"], data["fake_count"]
                    )
                    bar = _balance_bar_html(data["real_count"], data["fake_count"])

                    if not selected_sources:
                        return (
                            bar,
                            "*⚠️ Selecione pelo menos uma fonte.*",
                            readiness,
                        )

                    plan, rn, fn = _compute_download_plan(
                        selected_sources,
                        data["real_count"],
                        data["fake_count"],
                        int(target),
                    )
                    plan_text = _format_plan_md(
                        plan, rn, fn,
                        data["real_count"], data["fake_count"],
                        int(target),
                    )
                    # Readiness pós-download estimado
                    exp_real = data["real_count"] + sum(
                        v["real_alloc"] for v in plan.values()
                    )
                    exp_fake = data["fake_count"] + sum(
                        v["fake_alloc"] for v in plan.values()
                    )
                    readiness_after = _assess_training_readiness(exp_real, exp_fake)
                    combined_readiness = (
                        "**Atual:**\n\n" + readiness +
                        "\n\n---\n\n**Após este download:**\n\n" + readiness_after
                    )
                    return bar, plan_text, combined_readiness

                def handle_download(selected_sources, target, speakers_override):
                    """Generator: download balanceado com streaming de log e barra de progresso."""
                    data = _scan_dataset()

                    if not selected_sources:
                        yield (
                            _balance_bar_html(data["real_count"], data["fake_count"]),
                            "*⚠️ Selecione pelo menos uma fonte.*",
                            _assess_training_readiness(data["real_count"], data["fake_count"]),
                            "",
                            "❌ Nenhuma fonte selecionada",
                        )
                        return

                    try:
                        import scripts.download_datasets as dl_mod
                    except ImportError as exc:
                        yield (
                            _balance_bar_html(data["real_count"], data["fake_count"]),
                            "",
                            _assess_training_readiness(data["real_count"], data["fake_count"]),
                            f"Erro de importação:\n{exc}",
                            "❌ Não foi possível importar scripts/download_datasets.py",
                        )
                        return

                    capture = _LogCapture()
                    dl_mod.logger.addHandler(capture)
                    dl_mod.setup_dirs()

                    try:
                        # ── Estado inicial e plano ──────────────────────
                        plan, real_needed, fake_needed = _compute_download_plan(
                            selected_sources,
                            data["real_count"],
                            data["fake_count"],
                            int(target),
                        )
                        plan_text = _format_plan_md(
                            plan, real_needed, fake_needed,
                            data["real_count"], data["fake_count"],
                            int(target),
                        )

                        yield (
                            _balance_bar_html(data["real_count"], data["fake_count"]),
                            plan_text,
                            _assess_training_readiness(data["real_count"], data["fake_count"]),
                            "Iniciando download balanceado...",
                            "⏳ Preparando...",
                        )

                        # ── Executar cada fonte com alocação calculada ──
                        for i, source in enumerate(selected_sources):
                            info    = plan.get(source, {})
                            max_s   = info.get("max_samples", 0)
                            f_alloc = info.get("fake_alloc", 0)

                            if max_s <= 0:
                                capture.records.append(
                                    f"[INFO] {source}: alvo já atingido — pulando."
                                )
                                yield (
                                    _balance_bar_html(data["real_count"], data["fake_count"]),
                                    plan_text,
                                    _assess_training_readiness(data["real_count"], data["fake_count"]),
                                    capture.text(),
                                    f"⏳ [{i+1}/{len(selected_sources)}] {source}: já OK",
                                )
                                continue

                            capture.records.append(
                                f"[INFO] ── [{i+1}/{len(selected_sources)}] {source} "
                                f"(max_samples={max_s}) ──"
                            )
                            yield (
                                _balance_bar_html(data["real_count"], data["fake_count"]),
                                plan_text,
                                _assess_training_readiness(data["real_count"], data["fake_count"]),
                                capture.text(),
                                f"⏳ [{i+1}/{len(selected_sources)}] Baixando {source}...",
                            )

                            # ── Dispatch ───────────────────────────────
                            if source == "BRSpeech-DF":
                                dl_mod.download_brspeech(max_s)

                            elif source == "Fake Voices":
                                spk_override = int(speakers_override or 0)
                                if spk_override > 0:
                                    max_spk = spk_override
                                else:
                                    max_spk = max(5, min(50, math.ceil(
                                        f_alloc / _FAKE_VOICES_SAMPLES_PER_SPEAKER
                                    )))
                                capture.records.append(
                                    f"[INFO]   Fake Voices: {max_spk} falantes "
                                    f"(~{max_spk * _FAKE_VOICES_SAMPLES_PER_SPEAKER} amostras)"
                                )
                                dl_mod.download_fake_voices(max_speakers=max_spk)

                            elif source == "FLEURS":
                                dl_mod.download_fleurs(max_s)

                            elif source == "CETUC":
                                dl_mod.download_cetuc(max_s)

                            elif source == "MLAAD-PT":
                                dl_mod.download_mlaad_pt(max_s)

                            elif source == "Common Voice PT":
                                dl_mod.download_common_voice_pt(max_s)

                            elif source == "ASVspoof 2019":
                                dl_mod.download_asvspoof2019(max_s)

                            elif source == "WaveFake":
                                dl_mod.download_wavefake(max_s)

                            elif source == "In-the-Wild":
                                dl_mod.download_in_the_wild(max_s)

                            elif source == "ASVspoof 5":
                                dl_mod.download_asvspoof5(max_s)

                            # ── Atualiza barra e readiness após cada fonte ─
                            # light=True: só contadores (evita 300 sf.info/scan)
                            data = _scan_dataset(light=True)
                            yield (
                                _balance_bar_html(data["real_count"], data["fake_count"]),
                                plan_text,
                                _assess_training_readiness(data["real_count"], data["fake_count"]),
                                capture.text(),
                                f"⏳ [{i+1}/{len(selected_sources)}] {source} concluído.",
                            )

                        # ── Relatório final (scan completo p/ durações) ─
                        data   = _scan_dataset()
                        rc, fc = data["real_count"], data["fake_count"]
                        ratio  = rc / max(fc, 1)
                        rh, fh = data["real_duration_h"], data["fake_duration_h"]

                        if 0.8 <= ratio <= 1.25:
                            bal_line = f"✅ Dataset balanceado (ratio={ratio:.2f})"
                        elif 0.5 <= ratio <= 2.0:
                            bal_line = (
                                f"⚠️ Aceitável (ratio={ratio:.2f}) — "
                                "class_weight=balanced será aplicado no treino"
                            )
                        else:
                            menor = "real" if rc < fc else "fake"
                            diff  = abs(rc - fc)
                            bal_line = (
                                f"❌ Desbalanceado (ratio={ratio:.2f}) — "
                                f"faltam ~{diff:,} amostras `{menor}`. "
                                f"Use preset **Só Reforçar {menor.title()}** para corrigir."
                            )

                        final_status = (
                            f"✅ **Download concluído!**\n\n"
                            f"| Classe | Amostras | Duração |\n"
                            f"|--------|:--------:|:-------:|\n"
                            f"| Real   | **{rc:,}** | ~{rh:.1f}h |\n"
                            f"| Fake   | **{fc:,}** | ~{fh:.1f}h |\n"
                            f"| Total  | **{rc+fc:,}** | ~{rh+fh:.1f}h |\n\n"
                            f"{bal_line}\n\n"
                            f"**Próximo passo:** Aba **Pré-processamento → Pipeline Completo**"
                        )

                        yield (
                            _balance_bar_html(rc, fc),
                            plan_text,
                            _assess_training_readiness(rc, fc),
                            capture.text(),
                            final_status,
                        )

                    except Exception as e:
                        data = _scan_dataset()
                        yield (
                            _balance_bar_html(data["real_count"], data["fake_count"]),
                            "",
                            _assess_training_readiness(data["real_count"], data["fake_count"]),
                            capture.text() + f"\n\nERRO: {e}",
                            f"❌ Erro: {e}",
                        )
                    finally:
                        dl_mod.logger.removeHandler(capture)

                # Wire events
                dl_bal_refresh.click(
                    fn=_dl_refresh_balance,
                    outputs=[dl_balance_html, dl_readiness_md],
                )
                # Tier → define alvo + sources + descrição; o .change de sources
                # cascateia para recalcular o plano automaticamente.
                dl_tier.change(
                    fn=_dl_apply_tier,
                    inputs=[dl_tier],
                    outputs=[dl_target, dl_sources, dl_tier_desc],
                )
                # Preset → atualiza sources; o .change de sources cascateia
                # para recalcular o plano automaticamente (sem clicar em Calcular).
                dl_preset.change(
                    fn=_dl_apply_preset,
                    inputs=[dl_preset],
                    outputs=[dl_sources],
                )
                # Handler único: toggle falantes + barra + plano + prontidão.
                dl_sources.change(
                    fn=_dl_on_sources_change,
                    inputs=[dl_sources, dl_target],
                    outputs=[dl_speakers_override, dl_balance_html,
                             dl_plan_md, dl_readiness_md],
                )
                # Mudar o alvo também recalcula o plano.
                dl_target.release(
                    fn=_dl_on_sources_change,
                    inputs=[dl_sources, dl_target],
                    outputs=[dl_speakers_override, dl_balance_html,
                             dl_plan_md, dl_readiness_md],
                )
                dl_plan_btn.click(
                    fn=_dl_compute_plan,
                    inputs=[dl_sources, dl_target],
                    outputs=[dl_balance_html, dl_plan_md, dl_readiness_md],
                )
                dl_btn.click(
                    fn=handle_download,
                    inputs=[dl_sources, dl_target, dl_speakers_override],
                    outputs=[dl_balance_html, dl_plan_md, dl_readiness_md, dl_log, dl_status],
                )

            # ===========================================================
            # SUB-TAB 3: PREPROCESSAMENTO
            # ===========================================================
            with gr.Tab("Pré-processamento", id="tab_ds_preprocess"):

                gr.Markdown(
                    "### Pipeline de Pré-processamento\n"
                    "Valide, normalize, remova duplicatas e crie splits train/val/test."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        pp_validate_btn = gr.Button("Validar Dataset", variant="secondary")
                        pp_normalize_btn = gr.Button("Normalizar Áudio", variant="secondary")
                        pp_dedup_btn = gr.Button("Remover Duplicatas", variant="secondary")
                        gr.Markdown("---")
                        pp_train_ratio = gr.Slider(0.6, 0.9, value=0.8, step=0.05, label="Train ratio")
                        pp_val_ratio = gr.Slider(0.05, 0.2, value=0.1, step=0.05, label="Val ratio")
                        pp_splits_btn = gr.Button("Criar Splits", variant="primary")
                        gr.Markdown("---")
                        pp_full_btn = gr.Button("Pipeline Completo", variant="primary")
                        pp_zip_btn = gr.Button("Criar ZIP para Upload", variant="secondary")

                    with gr.Column(scale=2):
                        pp_log = gr.Textbox(label="Log", lines=16, max_lines=25, interactive=False)
                        pp_summary = gr.Markdown("*Aguardando...*")
                        pp_issues_df = gr.Dataframe(
                            headers=["Tipo", "Quantidade", "Detalhes"],
                            label="Problemas Encontrados",
                            interactive=False,
                        )

                def _get_pp_module():
                    import scripts.preprocess_dataset as pp
                    return pp

                def handle_validate():
                    pp = _get_pp_module()
                    capture = _LogCapture()
                    pp.logger.addHandler(capture)
                    try:
                        stats, issues = pp.validate_dataset()

                        # Resumo
                        parts = []
                        for label in ["real", "fake"]:
                            s = stats[label]
                            if s["durations"]:
                                parts.append(
                                    f"**{label.upper()}**: {s['count']} validos, "
                                    f"{s['total_duration']/3600:.1f}h, "
                                    f"media {np.mean(s['durations']):.1f}s"
                                )
                            else:
                                parts.append(f"**{label.upper()}**: Nenhum arquivo valido")

                        total_issues = sum(len(v) for v in issues.values())
                        if total_issues == 0:
                            parts.append("\n✅ **Nenhum problema encontrado!**")
                        else:
                            parts.append(f"\n⚠️ **{total_issues} problemas encontrados**")

                        summary = "\n\n".join(parts)

                        # Issues table
                        issue_rows = []
                        for issue_type, items in issues.items():
                            if items:
                                detail = f"Primeiros: {str(items[:3])[:100]}..."
                                issue_rows.append([issue_type, len(items), detail])

                        if not issue_rows:
                            issue_rows = [["Nenhum", 0, "Dataset limpo"]]

                        return capture.text(), summary, issue_rows
                    finally:
                        pp.logger.removeHandler(capture)

                def handle_normalize():
                    pp = _get_pp_module()
                    capture = _LogCapture()
                    pp.logger.addHandler(capture)
                    try:
                        pp.normalize_all()
                        return capture.text(), "✅ Normalizacao concluida", [["OK", 0, "Todos normalizados"]]
                    except Exception as e:
                        return capture.text(), f"❌ Erro: {e}", []
                    finally:
                        pp.logger.removeHandler(capture)

                def handle_dedup():
                    pp = _get_pp_module()
                    capture = _LogCapture()
                    pp.logger.addHandler(capture)
                    try:
                        removed = pp.remove_duplicates()
                        return capture.text(), f"✅ {removed} duplicatas removidas", []
                    except Exception as e:
                        return capture.text(), f"❌ Erro: {e}", []
                    finally:
                        pp.logger.removeHandler(capture)

                def handle_splits(train_r, val_r):
                    pp = _get_pp_module()
                    capture = _LogCapture()
                    pp.logger.addHandler(capture)
                    try:
                        test_r = round(1.0 - train_r - val_r, 2)
                        if test_r <= 0:
                            return capture.text(), "❌ Ratios invalidos (train + val >= 1.0)", []
                        pp.create_splits(train_r, val_r, test_r)
                        return capture.text(), f"✅ Splits criados ({train_r}/{val_r}/{test_r})", []
                    except Exception as e:
                        return capture.text(), f"❌ Erro: {e}", []
                    finally:
                        pp.logger.removeHandler(capture)

                def handle_full_pipeline(train_r, val_r):
                    pp = _get_pp_module()
                    capture = _LogCapture()
                    pp.logger.addHandler(capture)
                    try:
                        yield capture.text() + "\n[1/4] Validando...", "⏳ Validando...", []
                        stats, issues = pp.validate_dataset()
                        yield capture.text(), "⏳ Normalizando...", []

                        pp.normalize_all()
                        yield capture.text(), "⏳ Removendo duplicatas...", []

                        pp.remove_duplicates()
                        yield capture.text(), "⏳ Criando splits...", []

                        test_r = round(1.0 - train_r - val_r, 2)
                        pp.create_splits(train_r, val_r, test_r)

                        total_issues = sum(len(v) for v in issues.values())
                        summary = (
                            f"✅ **Pipeline completo!**\n\n"
                            f"- Validacao: {total_issues} problemas\n"
                            f"- Splits: {train_r}/{val_r}/{test_r}"
                        )
                        issue_rows = []
                        for it, items in issues.items():
                            if items:
                                issue_rows.append([it, len(items), str(items[:2])[:80]])

                        yield capture.text(), summary, issue_rows or [["OK", 0, "Tudo limpo"]]
                    except Exception as e:
                        yield capture.text(), f"❌ Erro: {e}", []
                    finally:
                        pp.logger.removeHandler(capture)

                def handle_zip():
                    pp = _get_pp_module()
                    capture = _LogCapture()
                    pp.logger.addHandler(capture)
                    try:
                        path = pp.create_training_zip()
                        size_mb = path.stat().st_size / (1024 * 1024) if path else 0
                        return capture.text(), f"✅ ZIP criado: {path} ({size_mb:.1f} MB)", []
                    except Exception as e:
                        return capture.text(), f"❌ Erro: {e}", []
                    finally:
                        pp.logger.removeHandler(capture)

                pp_validate_btn.click(fn=handle_validate, outputs=[pp_log, pp_summary, pp_issues_df])
                pp_normalize_btn.click(fn=handle_normalize, outputs=[pp_log, pp_summary, pp_issues_df])
                pp_dedup_btn.click(fn=handle_dedup, outputs=[pp_log, pp_summary, pp_issues_df])
                pp_splits_btn.click(fn=handle_splits, inputs=[pp_train_ratio, pp_val_ratio], outputs=[pp_log, pp_summary, pp_issues_df])
                pp_full_btn.click(fn=handle_full_pipeline, inputs=[pp_train_ratio, pp_val_ratio], outputs=[pp_log, pp_summary, pp_issues_df])
                pp_zip_btn.click(fn=handle_zip, outputs=[pp_log, pp_summary, pp_issues_df])

            # ===========================================================
            # SUB-TAB 4: COMPATIBILIDADE
            # ===========================================================
            with gr.Tab("Compatibilidade", id="tab_ds_compat"):

                gr.Markdown(
                    "### Analise de Compatibilidade\n"
                    "Verifica se o dataset atual e compativel com cada uma das 14 arquiteturas de deteccao."
                )

                compat_btn = gr.Button("Analisar Compatibilidade", variant="primary")

                compat_df = gr.Dataframe(
                    headers=["Arquitetura", "Tipo", "Status", "Recomendacoes"],
                    label="Compatibilidade por Arquitetura",
                    interactive=False,
                    wrap=True,
                )

                with gr.Accordion("Recomendacoes Detalhadas", open=False):
                    compat_details = gr.Markdown("*Execute a análise para ver recomendações...*")

                # Resumo geral
                compat_summary = gr.Markdown("")

                def handle_compatibility():
                    data = _scan_dataset()
                    rows, details = _analyze_compatibility(data)

                    ok_count = sum(1 for r in rows if r[2] == "OK")
                    warn_count = len(rows) - ok_count

                    summary = (
                        f"**Resultado:** {ok_count}/14 arquiteturas totalmente compativeis"
                    )
                    if warn_count:
                        summary += f", {warn_count} com recomendacoes"

                    summary += (
                        f"\n\n**Dataset atual:** {data['real_count']} real + {data['fake_count']} fake = "
                        f"{data['real_count'] + data['fake_count']} total "
                        f"(~{data['real_duration_h'] + data['fake_duration_h']:.1f}h)"
                    )

                    return rows, details, summary

                compat_btn.click(
                    fn=handle_compatibility,
                    outputs=[compat_df, compat_details, compat_summary],
                )
