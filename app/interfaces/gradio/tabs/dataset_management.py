"""
Aba de Gestão de Dataset — XFakeSong

Centraliza visualização, download, preprocessamento e análise de
compatibilidade dos datasets de áudio para detecção de deepfake.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import gradio as gr

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
    "BRSpeech-DF": "#3b82f6",
    "Fake Voices": "#8b5cf6",
    "FLEURS": "#06b6d4",
    "Common Voice": "#f59e0b",
    "CETUC": "#10b981",
    "Synthetic": "#6b7280",
    "Outros": "#94a3b8",
}


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
def _scan_dataset() -> Dict[str, Any]:
    """Escaneia o dataset e retorna estatísticas completas."""
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

    # Prefixos conhecidos → nome de fonte
    prefix_map = {
        "brspeech": "BRSpeech-DF",
        "fkvoice": "Fake Voices",
        "fleurs": "FLEURS",
        "cv": "Common Voice",
        "cetuc": "CETUC",
        "synthetic": "Synthetic",
    }

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

        # Amostrar durações (max 150 arquivos)
        sample_files = wav_files[:150]
        durations = []
        for wf in sample_files:
            try:
                info = sf.info(str(wf))
                durations.append(info.duration)
            except Exception:
                pass

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
        except Exception:
            pass

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
    ax.set_title("Distribuicao por Classe")
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

    ax.set_xlabel("Duracao (s)")
    ax.set_ylabel("Frequencia")
    ax.set_title("Distribuicao de Duracoes")
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
    except Exception:
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
            recs.append("Execute 'Criar Splits' na aba Preprocessamento")

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
            except Exception:
                pass

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

    with gr.Tab("Dataset", id="tab_dataset_mgmt"):
        gr.Markdown(
            "### Gestao de Dataset\n"
            "Visualize, baixe, preprocesse e analise a compatibilidade dos datasets de audio."
        )

        with gr.Tabs():
            # ===========================================================
            # SUB-TAB 1: VISAO GERAL
            # ===========================================================
            with gr.Tab("Visao Geral", id="tab_ds_overview"):

                with gr.Row():
                    kpi_md = gr.Markdown("*Clique em Atualizar para carregar...*")

                with gr.Row():
                    with gr.Column(scale=1):
                        plot_class = gr.Plot(label="Distribuicao por Classe")
                    with gr.Column(scale=1):
                        plot_source = gr.Plot(label="Fontes de Dados")

                with gr.Row():
                    with gr.Column(scale=1):
                        plot_duration = gr.Plot(label="Distribuicao de Duracoes")
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
                        f"| Duracao Real | **{real_h:.1f}h** |\n"
                        f"| Duracao Fake | **{fake_h:.1f}h** |\n"
                        f"| Duracao Total | **{real_h + fake_h:.1f}h** |\n"
                        f"| Splits Criados | **{splits_ok}** |\n"
                        f"| Sample Rate | **16 kHz** |\n"
                        f"| Formato | **WAV PCM 16-bit mono** |"
                    )

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
            # SUB-TAB 2: DOWNLOAD
            # ===========================================================
            with gr.Tab("Download", id="tab_ds_download"):

                gr.Markdown(
                    "### Download de Datasets PT-BR\n\n"
                    "| Dataset | Tipo | Descricao |\n"
                    "|---------|------|-----------|\n"
                    "| **BRSpeech-DF** | Real + Fake | 459K arquivos, bonafide/spoof, 24kHz→16kHz |\n"
                    "| **Fake Voices** | Fake | ~140h XTTS, 101 falantes, MIT |\n"
                    "| **FLEURS** | Real | Google, PT-BR, acesso publico |\n"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        dl_source = gr.Dropdown(
                            choices=["Todos", "BRSpeech-DF", "Fake Voices", "FLEURS"],
                            value="Todos",
                            label="Fonte",
                        )
                        dl_max_samples = gr.Slider(
                            minimum=50, maximum=5000, value=500, step=50,
                            label="Max amostras por classe",
                        )
                        dl_max_speakers = gr.Slider(
                            minimum=5, maximum=50, value=15, step=5,
                            label="Max falantes (Fake Voices)",
                            visible=True,
                        )
                        dl_btn = gr.Button("Iniciar Download", variant="primary")

                    with gr.Column(scale=2):
                        dl_log = gr.Textbox(
                            label="Log de Download",
                            lines=18,
                            max_lines=25,
                            interactive=False,
                        )
                        dl_status = gr.Markdown("*Aguardando...*")

                def toggle_speakers(source):
                    visible = source in ("Fake Voices", "Todos")
                    return gr.update(visible=visible)

                dl_source.change(fn=toggle_speakers, inputs=[dl_source], outputs=[dl_max_speakers])

                def handle_download(source, max_samples, max_speakers):
                    """Generator que executa download e streama logs."""
                    try:
                        import scripts.download_pt_datasets_v2 as dl_mod
                    except ImportError:
                        yield "Erro: nao foi possivel importar scripts/download_pt_datasets_v2.py", "❌ Erro de importacao"
                        return

                    capture = _LogCapture()
                    dl_mod.logger.addHandler(capture)
                    dl_mod.setup_dirs()

                    max_s = int(max_samples)
                    max_sp = int(max_speakers)

                    try:
                        yield capture.text() + "\nIniciando download...", "⏳ Baixando..."

                        if source in ("BRSpeech-DF", "Todos"):
                            dl_mod.download_brspeech(max_s)
                            yield capture.text(), "⏳ BRSpeech-DF concluido..."

                        if source in ("Fake Voices", "Todos"):
                            dl_mod.download_fake_voices(max_sp)
                            yield capture.text(), "⏳ Fake Voices concluido..."

                        if source in ("FLEURS", "Todos"):
                            dl_mod.download_fleurs(max_s)
                            yield capture.text(), "⏳ FLEURS concluido..."

                        # Report final
                        data = _scan_dataset()
                        final = (
                            f"✅ **Download concluido!**\n\n"
                            f"- Real: {data['real_count']} amostras (~{data['real_duration_h']:.1f}h)\n"
                            f"- Fake: {data['fake_count']} amostras (~{data['fake_duration_h']:.1f}h)\n"
                            f"- Total: {data['real_count'] + data['fake_count']}"
                        )
                        yield capture.text(), final

                    except Exception as e:
                        yield capture.text() + f"\n\nERRO: {e}", f"❌ Erro: {e}"
                    finally:
                        dl_mod.logger.removeHandler(capture)

                dl_btn.click(
                    fn=handle_download,
                    inputs=[dl_source, dl_max_samples, dl_max_speakers],
                    outputs=[dl_log, dl_status],
                )

            # ===========================================================
            # SUB-TAB 3: PREPROCESSAMENTO
            # ===========================================================
            with gr.Tab("Preprocessamento", id="tab_ds_preprocess"):

                gr.Markdown(
                    "### Pipeline de Preprocessamento\n"
                    "Valide, normalize, remova duplicatas e crie splits train/val/test."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        pp_validate_btn = gr.Button("Validar Dataset", variant="secondary")
                        pp_normalize_btn = gr.Button("Normalizar Audio", variant="secondary")
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
                    compat_details = gr.Markdown("*Execute a analise para ver recomendacoes...*")

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
