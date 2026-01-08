import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import librosa
import librosa.display
import json
import logging
from pathlib import Path
import sys

from app.core.interfaces.audio import AudioData

# Configurar logging
logger = logging.getLogger("gradio_detection_tab")

# Singleton para o serviço de detecção
_detection_service_instance = None

def get_detection_service():
    global _detection_service_instance
    if _detection_service_instance is None:
        try:
            from app.domain.services.detection_service import DetectionService
            # Inicializa com diretório padrão 'models'
            _detection_service_instance = DetectionService()
        except Exception as e:
            logger.error(f"Failed to init detection service: {e}")
            return None
    return _detection_service_instance

# Tentar importar serviços
try:
    from app.domain.services.detection_service import DetectionService
    from app.domain.models.architectures.registry import get_architecture_info, get_available_architectures
    MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"Aviso: Não foi possível importar serviços de detecção ({e}). Usando modo demonstração.")
    MODELS_AVAILABLE = False


def get_waveform_plot(y, sr):
    """Gera plot da forma de onda."""
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title("Forma de Onda")
    plt.tight_layout()
    return plt.gcf()


def get_prosody_plot(y, sr):
    """Gera plot de prosódia (F0 e Energia)."""
    plt.figure(figsize=(10, 4))

    # Energia
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms)
    plt.plot(times, rms, label='Energia (RMS)', color='r', alpha=0.6)

    # F0 (Pitch)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        times_f0 = librosa.times_like(f0)

        # Normalizar F0 para plotar junto
        if np.nanmax(f0) > 0:
            f0_norm = f0 / np.nanmax(f0)
            plt.plot(
                times_f0,
                f0_norm,
                label='Pitch (F0 Normalizado)',
                color='b',
                alpha=0.6)
    except Exception as e:
        logger.warning(f"Erro ao calcular Pitch: {e}")

    plt.legend()
    plt.title("Análise Prosódica: Energia e Pitch")
    plt.xlabel("Tempo (s)")
    plt.tight_layout()

    return plt.gcf()


def get_spectrogram_plot(y, sr):
    """Gera espectrograma Mel."""
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(
        S_dB,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Espectrograma Mel")
    plt.tight_layout()
    return plt.gcf()


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
                    # Tentar encontrar modelo compatível com a arquitetura/variante
                    model_name = service.find_model(architecture, variant=variant)
                    if not model_name:
                         # Fallback: Tentar qualquer modelo dessa arquitetura
                        models = service.get_available_models()
                        for m in models:
                            if architecture in m: # Heurística simples
                                model_name = m
                                break
                    
                    if not model_name:
                         logger.warning(f"Nenhum modelo encontrado para {architecture}/{variant}")
                         # Não falha aqui, deixa o service usar o default se passar None, 
                         # ou retornará erro se não houver default.
                else:
                    # Modo simples: usa o default do service ou o primeiro disponível
                    model_name = service.default_model
                    if not model_name:
                        models = service.get_available_models()
                        if models:
                            model_name = models[0]

                if not model_name:
                    return "MODELO NÃO ENCONTRADO", 0.0, fig_waveform, fig_prosody, fig_spectrogram, {"error": "Nenhum modelo treinado disponível. Treine um modelo na aba de Treinamento."}

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
                    filename = Path(audio_path).name if audio_path else "unknown.wav"
                    service.save_analysis_result(data, filename)
                    
                else:
                    details = {
                        "error": result_proc.errors[0] if result_proc.errors else "Erro na inferência"
                    }
                    logger.error(f"Erro na inferência: {details['error']}")

            except Exception as e:
                logger.error(f"Erro na inferência: {e}")
                details = {"erro_inferencia": str(e)}

        # Mock de fallback (apenas se realmente falhou tudo)
        if result_label in ["MODELO NÃO ENCONTRADO", "DESCONHECIDO"] and not details.get("error"):
            result_label = "DEMO MODE (Sem Modelo)"
            confidence = 0.0

        # Detalhes técnicos adicionais
        details["audio_info"] = {
            "duration": float(len(y) / sr),
            "sample_rate": sr,
            "rms_mean": float(np.mean(librosa.feature.rms(y=y)))
        }

        return result_label, confidence, fig_waveform, fig_prosody, fig_spectrogram, json.dumps(details, indent=2)

    except Exception as e:
        return f"Erro: {str(e)}", 0.0, None, None, None, {"error": str(e)}

    except Exception as e:
        return f"Erro: {str(e)}", 0.0, None, None, {"error": str(e)}


def process_stream(new_chunk, state):
    """Processamento em tempo real do stream de áudio com detecção contínua."""
    try:
        if new_chunk is None:
            return state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

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

        # Otimização: Gerar plots apenas se tiver dados suficientes e não for muito frequente
        y = state["audio"]

        if len(y) < sr * 0.1:  # Menos de 0.1s
            return state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
        # Throttling: Atualizar visualização no máximo a cada 0.5s (2 FPS)
        # Isso evita sobrecarregar a fila e o navegador (causa de AbortError)
        import time
        current_time = time.time()
        if current_time - state.get("last_update", 0) < 0.5:
            return state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
        state["last_update"] = current_time

        # Gerar plots rápidos
        # 0. Forma de Onda (Janela deslizante de 5s)
        fig_wave = Figure(figsize=(10, 3))
        ax_wave = fig_wave.add_subplot(111)
        
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
        times_plot = np.linspace(x_start, x_start + len(y_plot)/sr, len(y_plot))[::step]
        y_plot_ds = y_plot[::step]
        
        ax_wave.plot(times_plot, y_plot_ds, alpha=0.8)
        ax_wave.set_title(f"Forma de Onda (Tempo Real)")
        ax_wave.set_ylim(-1.0, 1.0)
        fig_wave.tight_layout()

        # 1. Espectrograma Mel (mais rápido que o completo)
        fig_spec = Figure(figsize=(10, 4))
        ax_spec = fig_spec.add_subplot(111)

        # Usar n_fft menor para rapidez no stream?
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, hop_length=1024)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_dB,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            fmax=8000,
            ax=ax_spec)
        fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
        ax_spec.set_title(
            f"Espectrograma Mel (Tempo Real) - {len(y) / sr:.1f}s")
        fig_spec.tight_layout()

        # 2. Prosódia (Energia + Pitch Simplificado)
        fig_pros = Figure(figsize=(10, 4))
        ax_pros = fig_pros.add_subplot(111)

        # Energia (RMS)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=1024)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=1024)
        ax_pros.plot(times, rms, label='Energia (RMS)', color='r', alpha=0.6)

        # Pitch (Estimativa rápida via Autocorrelação para Real-time)
        try:
            # Calcular autocorrelação apenas num frame recente para velocidade
            frame_len = int(sr * 0.05) # 50ms
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
                        if result[lag] > 0.1 * result[0]: # Threshold de periodicidade
                            f0_est = sr / lag
                            # Plotar linha horizontal indicando F0 estimado atual
                            ax_pros.axhline(y=f0_est/1000, color='b', linestyle='--', alpha=0.5, label=f'Pitch Est. ({int(f0_est)}Hz)')
        except Exception:
            pass

        ax_pros.legend(loc='upper right')
        ax_pros.set_title(f"Análise Prosódica: Energia (Tempo Real)")
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
                        label_upd = {lbl: conf, ("REAL" if lbl == "DEEPFAKE" else "DEEPFAKE"): 1.0 - conf}
                        conf_upd = conf
                except Exception as ex:
                    # Log menos verboso em stream
                    pass

        return state, fig_wave, fig_pros, fig_spec, label_upd, conf_upd

    except Exception as e:
        logger.error(f"Erro no stream: {e}")
        return state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def create_detection_tab():
    with gr.Tab("Detecção (Inference)", id="tab_detection"):
        gr.Markdown("""
        ### 🕵️ Análise de Integridade de Áudio
        Faça upload de um arquivo de áudio para verificar se ele é autêntico ou sintético (DeepFake).
        """)

        with gr.Row():
            # Coluna de Entrada (Esquerda/Topo)
            with gr.Column(scale=1, min_width=500):
                with gr.Group():
                    gr.Markdown("#### 📥 Entrada e Configuração")
                    audio_input = gr.Audio(
                        type="numpy", label="Arquivo de Áudio", sources=[
                            "upload", "microphone"], streaming=True)
                    stream_state = gr.State()

                    with gr.Accordion("⚙️ Configurações Avançadas", open=False):

                        arch_choices = get_available_architectures() if MODELS_AVAILABLE else []
                        arch_select = gr.Dropdown(
                            choices=arch_choices,
                            label="Arquitetura do Modelo",
                            value=arch_choices[0] if arch_choices else None,
                        )
                        variant_select = gr.Dropdown(
                            choices=[], label="Variante", value=None)
                        advanced_enabled = gr.Checkbox(
                            label="Habilitar Parâmetros Customizados", value=False)
                        hyperparams_json = gr.Code(
                            label="Hiperparâmetros (JSON)",
                            language="json",
                            value="{}",
                            interactive=True,
                            lines=3)
                        segmented_chk = gr.Checkbox(
                            label="Inferência Segmentada (Para áudios longos)", value=False)

                    analyze_btn = gr.Button(
                        "🔍 Analisar Áudio", variant="primary", size="lg")

            # Coluna de Saída (Direita/Baixo)
            with gr.Column(scale=1, min_width=500):
                with gr.Group():
                    gr.Markdown("#### 📊 Resultado da Análise")
                    with gr.Row():
                        label_output = gr.Label(
                            label="Classificação", num_top_classes=2, scale=2)
                        confidence_output = gr.Number(
                            label="Confiança", scale=1)

        gr.Markdown("---")

        # Seção de Detalhes Visuais (Full Width)
        gr.Markdown("### 📈 Detalhes Forenses")
        
        plot_waveform = gr.Plot(label="Forma de Onda")
        
        with gr.Row():
            with gr.Column(min_width=400):
                plot_spectrogram = gr.Plot(label="Espectrograma Mel")
            with gr.Column(min_width=400):
                plot_prosody = gr.Plot(
                    label="Análise Prosódica (Pitch/Energia)")

        with gr.Accordion("📝 Metadados Técnicos (JSON)", open=False):
            json_output = gr.JSON(label="Raw Output")

        def update_variants(arch_name):
            try:
                if MODELS_AVAILABLE and arch_name:
                    info = get_architecture_info(arch_name)

                    # Buscar default params do DB para exibir no JSON
                    from app.domain.models.architectures.registry import architecture_registry
                    params = architecture_registry.get_active_config(
                        arch_name, variant="default")

                    return gr.update(choices=info.supported_variants, value=(
                        info.supported_variants[0] if info.supported_variants else None)), json.dumps(params, indent=2)
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
        def handle_analysis(audio_path, stream_state, architecture,
                            variant, advanced_enabled, hyperparams_json, segmented):
            import tempfile
            import soundfile as sf
            import numpy as np

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
                    logger.info(f"Áudio convertido de numpy para: {final_path}")
                except Exception as e:
                    return f"Erro ao processar áudio: {str(e)}", 0.0, None, None, {"error": str(e)}
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
                    return f"Erro ao processar gravação: {str(e)}", 0.0, None, None, {
                        "error": str(e)}

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
