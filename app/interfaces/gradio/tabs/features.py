import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import gradio as gr
from app.core.interfaces.audio import AudioData, FeatureType
from app.domain.services.feature_extraction_service import (
    AudioFeatureExtractionService,
    ExtractionConfig,
)

# ── Estilo dark para plots ─────────────────────────────────────
_PLOT_BG = "#0f172a"
_PLOT_FACE = "#1e293b"
_PLOT_TEXT = "#f1f5f9"
_PLOT_GRID = "#334155"


def _style_feat_ax(ax, fig, title=""):
    """Aplica estilo dark a um eixo matplotlib."""
    fig.patch.set_facecolor(_PLOT_BG)
    ax.set_facecolor(_PLOT_FACE)
    ax.set_title(title, color=_PLOT_TEXT, fontweight="600", fontsize=12, pad=10)
    ax.tick_params(colors=_PLOT_TEXT, labelsize=9)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(_PLOT_TEXT)
        lbl.set_fontsize(10)
    for spine in ax.spines.values():
        spine.set_color(_PLOT_GRID)
    ax.grid(True, color=_PLOT_GRID, alpha=0.3, linewidth=0.5)


def create_features_tab():
    with gr.Tab("Extração de Features"):
        gr.Markdown("### Visualizador de Features de Áudio")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    type="filepath", label="Arquivo de Áudio")
                feature_type_dropdown = gr.Dropdown(
                    choices=[ft.value for ft in FeatureType],
                    value=FeatureType.SPECTRAL.value,
                    label="Tipo de Feature"
                )
                normalize_chk = gr.Checkbox(value=True, label="Normalizar")
                extract_btn = gr.Button(
                    "Extrair e Visualizar", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Visualização Gráfica"):
                        gr.Markdown("#### Forma de Onda")
                        wave_output = gr.Plot(label="Waveform")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### Feature Heatmap")
                                plot_output = gr.Plot(
                                    label="Visualização da Feature")
                            with gr.Column():
                                gr.Markdown("#### Distribuição de Valores")
                                hist_output = gr.Plot(label="Histograma")

                    with gr.Tab("Informações Detalhadas"):
                        stats_output = gr.Markdown("### Estatísticas")
                        json_output = gr.JSON(label="Dados Brutos")

        def extract_and_plot(audio_path, feature_type_str, normalize):
            if not audio_path:
                return None, None, None, "Selecione um arquivo de áudio.", {
                    "error": "Nenhum arquivo selecionado"}

            try:
                # 1. Carregar Áudio
                audio_data = AudioData.from_file(audio_path)

                # 2. Configurar Extração
                service = AudioFeatureExtractionService()
                try:
                    ft_enum = FeatureType(feature_type_str)
                except ValueError:
                    # Fallback para string se não estiver no Enum
                    # (compatibilidade)
                    ft_enum = feature_type_str

                config = ExtractionConfig(
                    feature_types=[ft_enum],
                    normalize=normalize
                )

                # 3. Extrair
                result = service.extract_features(
                    audio_data, config
                )

                if result.status.name != "SUCCESS":
                    return (
                        None,
                        None,
                        None,
                        f"Erro na extração: {result.errors}",
                        {"error": result.errors}
                    )

                features = result.data.features
                feature_data = features.features

                # --- Geração dos Gráficos (Dark Theme) ---

                # 1. Plot Waveform
                fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
                _style_feat_ax(ax_wave, fig_wave, "Waveform (Time Domain)")
                librosa.display.waveshow(
                    audio_data.samples, sr=audio_data.sample_rate,
                    ax=ax_wave, color="#3b82f6", alpha=0.85)
                fig_wave.tight_layout()

                # 2. Plot Feature Heatmap
                fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
                _style_feat_ax(ax_feat, fig_feat, "Feature Heatmap")
                plotted = False
                first_key = list(feature_data.keys())[
                    0] if feature_data else None
                main_feature_array = None

                if feature_type_str == "spectral" and (
                        "spectrogram" in feature_data):
                    S_db = feature_data["spectrogram"]
                    main_feature_array = S_db
                    if S_db.ndim == 1:
                        ax_feat.plot(S_db, color="#3b82f6")
                        ax_feat.set_title("Spectral Feature (Mean)",
                                          color=_PLOT_TEXT, fontweight="600")
                    else:
                        img = librosa.display.specshow(
                            S_db, sr=audio_data.sample_rate,
                            x_axis='time', y_axis='hz', ax=ax_feat,
                            cmap='magma')
                        cb = fig_feat.colorbar(img, ax=ax_feat,
                                               format='%+2.0f dB', pad=0.02)
                        cb.ax.tick_params(colors=_PLOT_TEXT, labelsize=8)
                        cb.outline.set_edgecolor(_PLOT_GRID)
                        ax_feat.set_title("Espectrograma",
                                          color=_PLOT_TEXT, fontweight="600")
                    plotted = True

                elif feature_type_str == "cepstral" and "mfcc" in feature_data:
                    mfcc = feature_data["mfcc"]
                    main_feature_array = mfcc
                    if mfcc.ndim == 2:
                        img = librosa.display.specshow(mfcc, x_axis='time',
                                                       ax=ax_feat, cmap='viridis')
                        cb = fig_feat.colorbar(img, ax=ax_feat, pad=0.02)
                        cb.ax.tick_params(colors=_PLOT_TEXT, labelsize=8)
                        cb.outline.set_edgecolor(_PLOT_GRID)
                        ax_feat.set_title("MFCC",
                                          color=_PLOT_TEXT, fontweight="600")
                        plotted = True

                elif not plotted and first_key:
                    arr = feature_data[first_key]
                    if isinstance(arr, np.ndarray):
                        main_feature_array = arr
                        if arr.ndim == 2:
                            im = ax_feat.imshow(arr, aspect='auto',
                                                origin='lower', cmap='viridis')
                            cb = fig_feat.colorbar(im, ax=ax_feat, pad=0.02)
                            cb.ax.tick_params(colors=_PLOT_TEXT, labelsize=8)
                        else:
                            ax_feat.plot(arr, color="#06b6d4")
                        ax_feat.set_title(f"Feature: {first_key}",
                                          color=_PLOT_TEXT, fontweight="600")
                        plotted = True

                if not plotted:
                    ax_feat.text(0.5, 0.5, "Visualização não disponível",
                                 ha='center', va='center', color=_PLOT_TEXT)
                fig_feat.tight_layout()

                # 3. Plot Histogram
                fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                _style_feat_ax(ax_hist, fig_hist, "Distribuição de Valores")
                if main_feature_array is not None and isinstance(
                        main_feature_array, np.ndarray):
                    flat_data = main_feature_array.flatten()
                    ax_hist.hist(flat_data, bins=50, color='#3b82f6',
                                 edgecolor='#1e293b', alpha=0.8)
                    ax_hist.set_xlabel("Valor")
                    ax_hist.set_ylabel("Frequência")
                    mean_val = np.mean(flat_data)
                    std_val = np.std(flat_data)
                    ax_hist.axvline(mean_val, color='#ef4444', linestyle='dashed',
                                    linewidth=1.5, label=f'Média: {mean_val:.2f}')
                    ax_hist.axvline(mean_val + std_val, color='#10b981',
                                    linestyle='dashed', linewidth=1,
                                    label=f'+1σ: {mean_val + std_val:.2f}')
                    ax_hist.axvline(mean_val - std_val, color='#10b981',
                                    linestyle='dashed', linewidth=1,
                                    label=f'-1σ: {mean_val - std_val:.2f}')
                    ax_hist.legend(facecolor=_PLOT_FACE, edgecolor=_PLOT_GRID,
                                   labelcolor=_PLOT_TEXT, fontsize=9)
                else:
                    ax_hist.text(0.5, 0.5, "Dados não numéricos",
                                 ha='center', va='center', color=_PLOT_TEXT)
                fig_hist.tight_layout()

                # --- Estatísticas Detalhadas ---
                stats_md = "### 📊 Estatísticas das Features\n\n"
                stats_md += f"**Duração do Áudio:** {
                    audio_data.duration:.2f}s\n"
                stats_md += f"**Sample Rate:** {audio_data.sample_rate} Hz\n\n"

                stats_md += "| Feature | Shape | Min | Max | Mean | Std |\n"
                stats_md += "|---|---|---|---|---|---|\n"

                json_data = {}
                for k, v in feature_data.items():
                    if isinstance(v, np.ndarray):
                        # Add row to markdown table
                        try:
                            v_min = f"{np.min(v):.4f}"
                            v_max = f"{np.max(v):.4f}"
                            v_mean = f"{np.mean(v):.4f}"
                            v_std = f"{np.std(v):.4f}"
                            stats_md += (
                                f"| `{k}` | `{v.shape}` | "
                                f"{v_min} | {v_max} | {v_mean} | {v_std} |\n"
                            )

                            # Add to JSON
                            json_data[k] = {
                                "shape": str(v.shape),
                                "min": float(np.min(v)),
                                "max": float(np.max(v)),
                                "mean": float(np.mean(v)),
                                "std": float(np.std(v)),
                                # Amostra pequena
                                "sample_values": v.flatten()[:20].tolist()
                            }
                        except Exception:
                            stats_md += (
                                f"| `{k}` | `{v.shape}` | - | - | - | - |\n"
                            )
                            json_data[k] = str(v)
                    else:
                        json_data[k] = str(v)

                return fig_wave, fig_feat, fig_hist, stats_md, json_data

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, None, f"Erro Fatal: {str(e)}", {
                    "error": str(e)}

        extract_btn.click(
            extract_and_plot,
            inputs=[audio_input, feature_type_dropdown, normalize_chk],
            outputs=[
                wave_output,
                plot_output,
                hist_output,
                stats_output,
                json_output]
        )
