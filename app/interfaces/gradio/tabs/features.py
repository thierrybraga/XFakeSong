import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io
import json
from pathlib import Path
from app.domain.services.feature_extraction_service import AudioFeatureExtractionService, ExtractionConfig
from app.core.interfaces.audio import AudioData, FeatureType


def create_features_tab():
    with gr.Tab("Extração de Features"):
        gr.Markdown("### 🔬 Visualizador de Features de Áudio")

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
                result = service.extract_features(audio_data, config)

                if result.status.name != "SUCCESS":
                    return None, None, None, f"Erro na extração: {result.errors}", {
                        "error": result.errors}

                features = result.data.features
                feature_data = features.features

                # --- Geração dos Gráficos ---

                # 1. Plot Waveform
                fig_wave = plt.figure(figsize=(10, 3))
                librosa.display.waveshow(
                    audio_data.samples, sr=audio_data.sample_rate)
                plt.title("Waveform (Time Domain)")
                plt.tight_layout()

                # 2. Plot Feature Heatmap
                fig_feat = plt.figure(figsize=(10, 6))
                plotted = False
                first_key = list(feature_data.keys())[
                    0] if feature_data else None
                main_feature_array = None

                if feature_type_str == "spectral" and "spectrogram" in feature_data:
                    S_db = feature_data["spectrogram"]
                    main_feature_array = S_db
                    if S_db.ndim == 1:
                        plt.plot(S_db)
                        plt.title("Spectral Feature (Mean)")
                    else:
                        librosa.display.specshow(
                            S_db, sr=audio_data.sample_rate, x_axis='time', y_axis='hz')
                        plt.colorbar(format='%+2.0f dB')
                        plt.title('Espectrograma')
                    plotted = True

                elif feature_type_str == "cepstral" and "mfcc" in feature_data:
                    mfcc = feature_data["mfcc"]
                    main_feature_array = mfcc
                    if mfcc.ndim == 2:
                        librosa.display.specshow(mfcc, x_axis='time')
                        plt.colorbar()
                        plt.title('MFCC')
                        plotted = True

                elif not plotted and first_key:
                    arr = feature_data[first_key]
                    if isinstance(arr, np.ndarray):
                        main_feature_array = arr
                        if arr.ndim == 2:
                            plt.imshow(arr, aspect='auto', origin='lower')
                            plt.colorbar()
                        else:
                            plt.plot(arr)
                        plt.title(f"Feature: {first_key}")
                        plotted = True

                if not plotted:
                    plt.text(
                        0.5,
                        0.5,
                        "Visualização não disponível",
                        ha='center',
                        va='center')
                plt.tight_layout()

                # 3. Plot Histogram
                fig_hist = plt.figure(figsize=(6, 4))
                if main_feature_array is not None and isinstance(
                        main_feature_array, np.ndarray):
                    flat_data = main_feature_array.flatten()
                    plt.hist(
                        flat_data,
                        bins=50,
                        color='skyblue',
                        edgecolor='black',
                        alpha=0.7)
                    plt.title("Distribuição de Valores")
                    plt.xlabel("Valor")
                    plt.ylabel("Frequência")
                    # Adicionar linhas de média e std
                    mean_val = np.mean(flat_data)
                    std_val = np.std(flat_data)
                    plt.axvline(
                        mean_val,
                        color='r',
                        linestyle='dashed',
                        linewidth=1,
                        label=f'Média: {
                            mean_val:.2f}')
                    plt.axvline(mean_val + std_val,
                                color='g',
                                linestyle='dashed',
                                linewidth=1,
                                label=f'+1 Std: {mean_val + std_val:.2f}')
                    plt.axvline(mean_val - std_val,
                                color='g',
                                linestyle='dashed',
                                linewidth=1,
                                label=f'-1 Std: {mean_val - std_val:.2f}')
                    plt.legend()
                else:
                    plt.text(
                        0.5,
                        0.5,
                        "Dados não numéricos",
                        ha='center',
                        va='center')
                plt.tight_layout()

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
                            stats_md += f"| `{k}` | `{
                                v.shape}` | {v_min} | {v_max} | {v_mean} | {v_std} |\n"

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
                            stats_md += f"| `{k}` | `{v.shape}` | - | - | - | - |\n"
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
