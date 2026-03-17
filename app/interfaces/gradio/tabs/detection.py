import json
import logging
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import gradio as gr
from app.core.interfaces.audio import AudioData

# Configurar logging
logger = logging.getLogger("gradio_detection_tab")

# Singleton para o serviço de detecção
_detection_service_instance = None


def get_detection_service():
    global _detection_service_instance
    if _detection_service_instance is None:
        try:
            from app.domain.services import detection_service as ds
            # Inicializa com diretório padrão 'models'
            _detection_service_instance = ds.DetectionService()
        except Exception as e:
            logger.error(f"Failed to init detection service: {e}")
            return None
    return _detection_service_instance


# Tentar importar serviços
try:
    from app.domain.models.architectures.registry import (
        get_architecture_info,
        get_available_architectures,
    )
    from app.domain.services.detection_service import DetectionService  # noqa: F401
    MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"Aviso: Não foi possível importar serviços de detecção ({e}). "
        f"Usando modo demonstração."
    )
    MODELS_AVAILABLE = False


# ── Estilo dark para plots ───────────────────────────────────────
_PLOT_BG = "#0f172a"
_PLOT_FACE = "#1e293b"
_PLOT_TEXT = "#f1f5f9"
_PLOT_GRID = "#334155"
_PLOT_ACCENT = "#3b82f6"
_PLOT_ACCENT2 = "#06b6d4"
_PLOT_DANGER = "#ef4444"


def _style_ax(ax, fig, title=""):
    """Aplica estilo dark consistente a um eixo matplotlib."""
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


def get_waveform_plot(y, sr):
    """Gera plot da forma de onda em estilo dark."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _style_ax(ax, fig, "Forma de Onda")
    librosa.display.waveshow(y, sr=sr, alpha=0.85, color=_PLOT_ACCENT, ax=ax)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def get_prosody_plot(y, sr):
    """Gera plot de prosódia (F0 e Energia) em estilo dark."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _style_ax(ax, fig, "Análise Prosódica: Energia e Pitch")

    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms)
    ax.plot(times, rms, label='Energia (RMS)', color=_PLOT_DANGER, alpha=0.7, linewidth=1.5)

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        times_f0 = librosa.times_like(f0)
        if np.nanmax(f0) > 0:
            f0_norm = f0 / np.nanmax(f0)
            ax.plot(times_f0, f0_norm, label='Pitch (F0 Norm.)',
                    color=_PLOT_ACCENT2, alpha=0.7, linewidth=1.5)
    except Exception as e:
        logger.warning(f"Erro ao calcular Pitch: {e}")

    ax.legend(facecolor=_PLOT_FACE, edgecolor=_PLOT_GRID,
              labelcolor=_PLOT_TEXT, fontsize=9)
    ax.set_xlabel("Tempo (s)")
    fig.tight_layout()
    return fig


def get_spectrogram_plot(y, sr):
    """Gera espectrograma Mel em estilo dark."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _style_ax(ax, fig, "Espectrograma Mel")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000,
        ax=ax, cmap='magma')
    cb = fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
    cb.ax.tick_params(colors=_PLOT_TEXT, labelsize=8)
    cb.outline.set_edgecolor(_PLOT_GRID)
    fig.tight_layout()
    return fig


def analyze_audio(audio_path, architecture, variant,
                  advanced_enabled, hyperparams_json, segmented):
    if not audio_path:
        return "Erro: Nenhum áudio fornecido.", 0.0, None, None, {}

    try:
        # Carregar áudio para visualização
        y, sr = librosa.load(audio_path, sr=16000)

        # Gerar plots
        fig_waveform = get_waveform_plot(y, sr)
        fig_prosody = get_prosody_plot(y, sr)
        fig_spectrogram = get_spectrogram_plot(y, sr)

        # Detecção
        result_label = "DESCONHECIDO"
        confidence = 0.0
        details = {}

        if MODELS_AVAILABLE:
            try:
                service = get_detection_service()
                if not service:
                    raise Exception("Serviço de detecção não inicializado")

                model_name = None

                # Seleção de Modelo
                if advanced_enabled and architecture:
                    # Tentar encontrar modelo compatível com a
                    # arquitetura/variante
                    model_name = service.find_model(
                        architecture, variant=variant
                    )
                    if not model_name:
                        # Fallback: Tentar qualquer modelo dessa arquitetura
                        models = service.get_available_models()
                        for m in models:
                            if architecture in m:  # Heurística simples
                                model_name = m
                                break

                    if not model_name:
                        logger.warning(
                            f"Nenhum modelo encontrado para "
                            f"{architecture}/{variant}"
                        )
                        # Não falha aqui, deixa o service usar o default
                        # se passar None, ou retornará erro se não houver
                        # default.
                else:
                    # Modo simples: usa o default do service ou
                    # o primeiro disponível
                    model_name = service.default_model
                    if not model_name:
                        models = service.get_available_models()
                        if models:
                            model_name = models[0]

                if not model_name:
                    return (
                        "MODELO NÃO ENCONTRADO",
                        0.0,
                        fig_waveform,
                        fig_prosody,
                        fig_spectrogram,
                        {"error": "Nenhum modelo treinado disponível. "
                                  "Treine um modelo na aba de Treinamento."}
                    )

                # Executar Detecção
                result_proc = service.detect_from_file(
                    audio_path,
                    model_name=model_name,
                    segmented=bool(segmented)
                )

                if result_proc.status.name == "SUCCESS":
                    data = result_proc.data
                    result_label = "DEEPFAKE" if data.is_fake else "REAL"
                    confidence = float(data.confidence)
                    details = {
                        "model": data.model_name,
                        "probabilities": data.probabilities,
                        "metadata": data.metadata,
                        "features_used": data.features_used
                    }

                    # Persistir Resultado (usando o serviço)
                    filename = (
                        Path(audio_path).name if audio_path else "unknown.wav"
                    )
                    service.save_analysis_result(data, filename)

                else:
                    details = {
                        "error": (
                            result_proc.errors[0] if result_proc.errors
                            else "Erro na inferência"
                        )
                    }
                    logger.error(f"Erro na inferência: {details['error']}")

            except Exception as e:
                logger.error(f"Erro na inferência: {e}")
                details = {"erro_inferencia": str(e)}

        # Mock de fallback (apenas se realmente falhou tudo)
        if (result_label in ["MODELO NÃO ENCONTRADO", "DESCONHECIDO"] and
                not details.get("error")):
            result_label = "DEMO MODE (Sem Modelo)"
            confidence = 0.0

        # Detalhes técnicos adicionais
        details["audio_info"] = {
            "duration": float(len(y) / sr),
            "sample_rate": sr,
            "rms_mean": float(np.mean(librosa.feature.rms(y=y)))
        }

        return (
            result_label,
            confidence,
            fig_waveform,
            fig_prosody,
            fig_spectrogram,
            json.dumps(details, indent=2)
        )

    except Exception as e:
        return (
            f"Erro: {str(e)}",
            0.0,
            None,
            None,
            None,
            json.dumps({"error": str(e)}, indent=2),
        )


def process_stream(new_chunk, state):
    """Processamento em tempo real do stream de áudio com detecção contínua."""
    try:
        if new_chunk is None:
            return (
                state,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )

        sr, data = new_chunk

        # Inicializar estado
        if state is None:
            state = {
                "audio": np.array([], dtype=np.float32),
                "sr": sr,
                "last_update": 0
            }

        # Converter e normalizar
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0

        # Converter estéreo para mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Acumular
        state["audio"] = np.concatenate((state["audio"], data))

        # Otimização: Gerar plots apenas se tiver dados suficientes
        # e não for muito frequente
        y = state["audio"]

        if len(y) < sr * 0.1:  # Menos de 0.1s
            return (
                state,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )

        # Throttling: Atualizar visualização no máximo a cada 0.5s (2 FPS)
        # Isso evita sobrecarregar a fila e o navegador
        # (causa de AbortError)
        import time
        current_time = time.time()
        if current_time - state.get("last_update", 0) < 0.5:
            return (
                state,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )

        state["last_update"] = current_time

        # Gerar plots rápidos (dark theme)
        fig_wave = Figure(figsize=(10, 3))
        ax_wave = fig_wave.add_subplot(111)
        _style_ax(ax_wave, fig_wave, "Forma de Onda (Tempo Real)")

        # Limitar visualização aos últimos 5 segundos
        window_size = sr * 5
        if len(y) > window_size:
            y_plot = y[-window_size:]
            x_start = (len(y) - window_size) / sr
        else:
            y_plot = y
            x_start = 0

        # Downsample para plotagem rápida (máx 2000 pontos)
        step = max(1, len(y_plot) // 2000)
        times_plot = np.linspace(
            x_start,
            x_start + len(y_plot)/sr,
            len(y_plot)
        )[::step]
        y_plot_ds = y_plot[::step]

        ax_wave.plot(times_plot, y_plot_ds, alpha=0.85, color=_PLOT_ACCENT)
        ax_wave.set_ylim(-1.0, 1.0)
        fig_wave.tight_layout()

        # 1. Espectrograma Mel
        fig_spec = Figure(figsize=(10, 4))
        ax_spec = fig_spec.add_subplot(111)
        _style_ax(ax_spec, fig_spec, f"Espectrograma Mel — {len(y) / sr:.1f}s")

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, hop_length=1024)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000,
            ax=ax_spec, cmap='magma')
        cb = fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB', pad=0.02)
        cb.ax.tick_params(colors=_PLOT_TEXT, labelsize=8)
        cb.outline.set_edgecolor(_PLOT_GRID)
        fig_spec.tight_layout()

        # 2. Prosódia
        fig_pros = Figure(figsize=(10, 4))
        ax_pros = fig_pros.add_subplot(111)
        _style_ax(ax_pros, fig_pros, "Análise Prosódica (Tempo Real)")

        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=1024)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=1024)
        ax_pros.plot(times, rms, label='Energia (RMS)', color=_PLOT_DANGER, alpha=0.7, linewidth=1.5)

        # Pitch (Estimativa rápida via Autocorrelação para Real-time)
        try:
            # Calcular autocorrelação apenas num frame recente para velocidade
            frame_len = int(sr * 0.05)  # 50ms
            if len(y) > frame_len:
                y_frame = y[-frame_len:]

                # Autocorrelação normalizada
                result = np.correlate(y_frame, y_frame, mode='full')
                result = result[len(result)//2:]

                # Encontrar pico entre lags correspondentes a 50Hz e 1000Hz
                min_lag = int(sr / 1000)
                max_lag = int(sr / 50)

                if len(result) > max_lag:
                    relevant = result[min_lag:max_lag]
                    if len(relevant) > 0:
                        lag = np.argmax(relevant) + min_lag
                        # Threshold de periodicidade
                        if result[lag] > 0.1 * result[0]:
                            f0_est = sr / lag
                            # Plotar linha horizontal indicando F0
                            # estimado atual
                            ax_pros.axhline(
                                y=f0_est/1000,
                                color=_PLOT_ACCENT2,
                                linestyle='--',
                                alpha=0.6,
                                label=f'Pitch Est. ({int(f0_est)}Hz)'
                            )
        except Exception:
            pass

        ax_pros.legend(loc='upper right', facecolor=_PLOT_FACE,
                       edgecolor=_PLOT_GRID, labelcolor=_PLOT_TEXT, fontsize=9)
        ax_pros.set_xlabel("Tempo (s)")
        fig_pros.tight_layout()

        # 3. Detecção Real-time (Analise Acumulada)
        label_upd = gr.update()
        conf_upd = gr.update()

        # Executar detecção se tiver pelo menos 0.5s de áudio
        if len(y) > sr * 0.5:
            service = get_detection_service()
            if service:
                try:
                    # Cria AudioData com o buffer acumulado
                    audio_data = AudioData(
                        samples=y,
                        sample_rate=sr,
                        duration=float(len(y) / sr)
                    )

                    res = service.detect_single(audio_data)
                    if res.status.name == "SUCCESS":
                        data_res = res.data
                        lbl = "DEEPFAKE" if data_res.is_fake else "REAL"
                        conf = float(data_res.confidence)

                        # Formato para Label output
                        label_upd = {
                            lbl: conf,
                            ("REAL" if lbl == "DEEPFAKE" else "DEEPFAKE"):
                            1.0 - conf
                        }
                        conf_upd = conf
                except Exception:
                    # Log menos verboso em stream
                    pass

        return state, fig_wave, fig_pros, fig_spec, label_upd, conf_upd

    except Exception as e:
        logger.error(f"Erro no stream: {e}")
        return (
            state,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update()
        )


def create_detection_tab():
    with gr.Tab("🕵️ Detecção", id="tab_detection"):
        gr.Markdown(
            "### Análise de Integridade de Áudio\n"
            "Faça upload ou grave áudio em tempo real para verificar autenticidade. "
            "O sistema analisa padrões espectrais, prosódicos e temporais "
            "para identificar deepfakes."
        )

        with gr.Row():
            # Coluna de Entrada (Esquerda/Topo)
            with gr.Column(scale=1, min_width=500):
                with gr.Group():
                    gr.Markdown("#### Entrada e Configuração")
                    audio_input = gr.Audio(
                        type="numpy", label="Arquivo de Áudio", sources=[
                            "upload", "microphone"], streaming=True)
                    stream_state = gr.State()

                    with gr.Accordion(
                        "⚙️ Configurações Avançadas", open=False
                    ):

                        if MODELS_AVAILABLE:
                            arch_choices = get_available_architectures()
                        else:
                            arch_choices = []

                        arch_select = gr.Dropdown(
                            choices=arch_choices,
                            label="Arquitetura do Modelo",
                            value=arch_choices[0] if arch_choices else None,
                        )
                        variant_select = gr.Dropdown(
                            choices=[], label="Variante", value=None)
                        advanced_enabled = gr.Checkbox(
                            label="Habilitar Parâmetros Customizados",
                            value=False
                        )
                        hyperparams_json = gr.Code(
                            label="Hiperparâmetros (JSON)",
                            language="json",
                            value="{}",
                            interactive=True,
                            lines=3)
                        segmented_chk = gr.Checkbox(
                            label="Inferência Segmentada (Para áudios longos)",
                            value=False
                        )

                    analyze_btn = gr.Button(
                        "🔍 Analisar Áudio", variant="primary", size="lg")

            # Coluna de Saída (Direita/Baixo)
            with gr.Column(scale=1, min_width=500):
                with gr.Group():
                    gr.Markdown("#### Resultado da Análise")
                    with gr.Row():
                        label_output = gr.Label(
                            label="Classificação",
                            num_top_classes=2,
                            scale=2
                        )
                        confidence_output = gr.Number(
                            label="Confiança", scale=1)

        gr.Markdown("---")

        # Seção de Detalhes Visuais (Full Width)
        gr.Markdown("### Visualização Forense")

        plot_waveform = gr.Plot(label="Forma de Onda")

        with gr.Row():
            with gr.Column(min_width=400):
                plot_spectrogram = gr.Plot(label="Espectrograma Mel")
            with gr.Column(min_width=400):
                plot_prosody = gr.Plot(
                    label="Análise Prosódica (Pitch/Energia)")

        with gr.Accordion("Metadados Técnicos (JSON)", open=False):
            json_output = gr.JSON(label="Raw Output")

        def update_variants(arch_name):
            try:
                if MODELS_AVAILABLE and arch_name:
                    info = get_architecture_info(arch_name)

                    # Buscar default params do DB para exibir no JSON
                    from app.domain.models.architectures.registry import (
                        architecture_registry,
                    )
                    params = architecture_registry.get_active_config(
                        arch_name, variant="default"
                    )

                    default_var = (
                        info.supported_variants[0]
                        if info.supported_variants else None
                    )
                    return (
                        gr.update(
                            choices=info.supported_variants,
                            value=default_var
                        ),
                        json.dumps(params, indent=2)
                    )
            except Exception as e:
                logger.error(f"Erro ao atualizar variantes: {e}")
                pass
            return gr.update(choices=[], value=None), json.dumps({}, indent=2)

        arch_select.change(
            update_variants,
            inputs=[arch_select],
            outputs=[
                variant_select,
                hyperparams_json])

        # Limpar estado ao limpar áudio
        def clear_state():
            return None

        audio_input.clear(
            fn=clear_state,
            inputs=None,
            outputs=[stream_state]
        )

        # Wrapper para lidar com upload vs stream
        def handle_analysis(
            audio_path,
            stream_state,
            architecture,
            variant,
            advanced_enabled,
            hyperparams_json,
            segmented
        ):
            import tempfile

            import numpy as np
            import soundfile as sf

            final_path = None

            # Handle numpy input (from upload/mic since type="numpy")
            if isinstance(audio_path, tuple):
                sr, data = audio_path
                try:
                    # Save to temp file
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False)
                    temp_file.close()

                    # Convert to float32 if needed
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) / 32768.0
                    elif data.dtype == np.int32:
                        data = data.astype(np.float32) / 2147483648.0

                    sf.write(temp_file.name, data, sr)
                    final_path = temp_file.name
                    logger.info(
                        f"Áudio convertido de numpy para: {final_path}"
                    )
                except Exception as e:
                    return (
                        f"Erro ao processar áudio: {str(e)}",
                        0.0,
                        None,
                        None,
                        {"error": str(e)}
                    )
            elif isinstance(audio_path, str) and audio_path:
                final_path = audio_path

            # Se não há path (ex: microfone stream) mas tem estado acumulado
            if not final_path and stream_state is not None and len(
                    stream_state.get("audio", [])) > 0:
                try:
                    # Salvar áudio do estado em arquivo temporário
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False)
                    temp_file.close()

                    sr = stream_state.get("sr", 16000)
                    audio_data = stream_state["audio"]

                    sf.write(temp_file.name, audio_data, sr)
                    final_path = temp_file.name
                    logger.info(
                        f"Usando áudio do stream salvo em: {final_path}")
                except Exception as e:
                    logger.error(f"Erro ao salvar stream para análise: {e}")
                    return (
                        f"Erro ao processar gravação: {str(e)}",
                        0.0,
                        None,
                        None,
                        {"error": str(e)}
                    )

            return analyze_audio(final_path, architecture, variant,
                                 advanced_enabled, hyperparams_json, segmented)

        analyze_btn.click(
            handle_analysis,
            inputs=[
                audio_input,
                stream_state,
                arch_select,
                variant_select,
                advanced_enabled,
                hyperparams_json,
                segmented_chk],
            outputs=[
                label_output,
                confidence_output,
                plot_waveform,
                plot_prosody,
                plot_spectrogram,
                json_output]
        )

        # Eventos de Streaming (Real-time)
        audio_input.stream(
            fn=process_stream,
            inputs=[audio_input, stream_state],
            outputs=[stream_state, plot_waveform, plot_prosody,
                     plot_spectrogram, label_output, confidence_output],
            show_progress="hidden"
        )
