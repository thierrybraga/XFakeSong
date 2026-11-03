import json
import logging
from pathlib import Path

# Componentes de UI compartilhados (cabeçalho padronizado, callout, divisor)
from app.interfaces.gradio.presenters.verdict_card import (  # noqa: E402
    nome_do_modelo,
    render_error,
    render_idle,
    render_verdict,
)
from app.interfaces.gradio.utils.components import (  # noqa: E402
    info_callout,
    page_header,
)

# FE.2 + FE.8: importa helpers compartilhados ANTES de matplotlib.pyplot.
# `plotting` força backend Agg (necessário em ambiente Gradio threaded).
from app.interfaces.gradio.utils.plotting import (
    PLOT_ACCENT,
    PLOT_ACCENT2,
    PLOT_DANGER,
    PLOT_FACE,
    PLOT_GRID,
    PLOT_TEXT,
    get_service_lock,
    notify_error,
    safe_tight_layout,
    style_ax,
)

# Aliases legados para minimizar diff em código existente
_PLOT_BG = PLOT_FACE
_PLOT_FACE = PLOT_FACE
_PLOT_TEXT = PLOT_TEXT
_PLOT_GRID = PLOT_GRID
_PLOT_ACCENT = PLOT_ACCENT
_PLOT_ACCENT2 = PLOT_ACCENT2
_PLOT_DANGER = PLOT_DANGER
_style_ax = style_ax

import gradio as gr  # noqa: E402
import librosa  # noqa: E402
import librosa.display  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from app.core.contracts.audio import AudioData  # noqa: E402
from app.interfaces.gradio.utils.components import ui_safe  # noqa: E402
from app.interfaces.gradio.utils.plotting import new_figure  # noqa: E402

# Configurar logging
logger = logging.getLogger("gradio_detection_tab")

# FE.3: Singleton com lock thread-safe (Gradio queue tem 20 workers)
_detection_service_instance = None
_service_lock = get_service_lock("detection_service")

# FE.4: limite de buffer de streaming (máx 30s de áudio acumulado)
_STREAM_MAX_BUFFER_SECONDS = 30
# Áudios mais longos que isto fazem prosody/pitch ser pulado por performance (FE.5)
_PROSODY_PYIN_MAX_DURATION_S = 15
# Valor da opção "Automático" na lista de modelos.
AUTO_MODEL = "__auto__"
# Para nomear os artefatos `bench_*` antes de carregá-los. Os clássicos não estão no
# registro das arquiteturas neurais.
_ARQUITETURAS_COM_NOME = (
    "AASIST", "RawGAT-ST", "RawNet2", "Conformer", "Hybrid CNN-Transformer",
    "SpectrogramTransformer", "MultiscaleCNN", "SVM", "RandomForest",
    "WavLM", "HuBERT", "Sonic Sleuth", "EfficientNet-LSTM", "Ensemble",
)


def get_detection_service():
    """Retorna o singleton de DetectionService, criando-o sob lock."""
    global _detection_service_instance
    if _detection_service_instance is not None:
        return _detection_service_instance
    with _service_lock:
        # Double-checked locking — outro thread pode ter criado entre
        # o check e o acquire do lock.
        if _detection_service_instance is None:
            try:
                # Reusa o MESMO singleton (data/models) da API e do reload do wizard
                # de treino — assim um modelo recém-treinado aparece aqui sem
                # reiniciar. Antes criava um DetectionService() próprio lendo o dir
                # default "models" (raiz, vazio).
                from app.dependencies import get_detection_service as _shared
                _detection_service_instance = _shared()
            except Exception as e:
                logger.error(f"Failed to init detection service: {e}")
                return None
    return _detection_service_instance


# Tentar importar serviços
try:
    from app.domain.services.detection_service import DetectionService  # noqa: F401
    MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"Aviso: Não foi possível importar serviços de detecção ({e}). "
        f"Usando modo demonstração."
    )
    MODELS_AVAILABLE = False


def get_waveform_plot(y, sr):
    """Gera plot da forma de onda em estilo dark.

    Para áudios longos (>60s), faz downsample para evitar render lento.
    """
    fig, ax = new_figure(figsize=(10, 3))
    _style_ax(ax, fig, "Forma de Onda")

    # FE.5: downsample defensivo para áudios longos (>1 min)
    max_points = 200_000  # ~12s a 16kHz nas waveshow padrão
    if len(y) > max_points:
        step = len(y) // max_points
        y_plot = y[::step]
        # waveshow recalcula seu próprio tempo — passamos hop_length compatível
        librosa.display.waveshow(y_plot, sr=sr // step, alpha=0.85,
                                 color=_PLOT_ACCENT, ax=ax)
    else:
        librosa.display.waveshow(y, sr=sr, alpha=0.85, color=_PLOT_ACCENT, ax=ax)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    safe_tight_layout(fig)
    return fig


def get_prosody_plot(y, sr):
    """Gera plot de prosódia (F0 e Energia) em estilo dark.

    FE.5: librosa.pyin é MUITO lento (segundos por minuto de áudio). Pulamos
    para áudios > _PROSODY_PYIN_MAX_DURATION_S e usamos apenas RMS energy.
    """
    fig, ax = new_figure(figsize=(10, 4))
    _style_ax(ax, fig, "Análise Prosódica: Energia e Pitch")

    duration_s = len(y) / sr if sr > 0 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms)
    ax.plot(times, rms, label='Energia (RMS)', color=_PLOT_DANGER, alpha=0.7, linewidth=1.5)

    # F0 via pyin é o gargalo — só roda em áudios curtos
    if duration_s <= _PROSODY_PYIN_MAX_DURATION_S:
        try:
            f0, _voiced_flag, _voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
            )
            times_f0 = librosa.times_like(f0)
            max_f0 = float(np.nanmax(f0)) if np.any(~np.isnan(f0)) else 0.0
            if max_f0 > 0:
                f0_norm = f0 / max_f0
                ax.plot(times_f0, f0_norm, label='Pitch (F0 Norm.)',
                        color=_PLOT_ACCENT2, alpha=0.7, linewidth=1.5)
        except Exception as e:
            logger.warning(f"Erro ao calcular Pitch: {e}")
    else:
        logger.info(
            f"pyin pulado (duração={duration_s:.1f}s > "
            f"{_PROSODY_PYIN_MAX_DURATION_S}s)"
        )
        ax.text(
            0.02, 0.95,
            f"Pitch detection pulado (>{_PROSODY_PYIN_MAX_DURATION_S}s)",
            transform=ax.transAxes,
            color=_PLOT_TEXT, fontsize=9, alpha=0.6,
            verticalalignment='top',
        )

    ax.legend(facecolor=_PLOT_FACE, edgecolor=_PLOT_GRID,
              labelcolor=_PLOT_TEXT, fontsize=9)
    ax.set_xlabel("Tempo (s)")
    safe_tight_layout(fig)
    return fig


def get_spectrogram_plot(y, sr):
    """Gera espectrograma Mel em estilo dark."""
    fig, ax = new_figure(figsize=(10, 4))
    _style_ax(ax, fig, "Espectrograma Mel")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000,
        ax=ax, cmap='magma')
    cb = fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
    cb.ax.tick_params(colors=_PLOT_TEXT, labelsize=8)
    cb.outline.set_edgecolor(_PLOT_GRID)
    safe_tight_layout(fig)
    return fig


def _empty_analysis_result(
    error_msg: str = "Nenhum áudio fornecido.",
    *,
    hint: str = None,
    error_code: str = None,
):
    """Retorna a tupla de 5 elementos esperada por analyze_btn.click outputs.

    FE.1: Garante que TODOS os paths de retorno tenham o mesmo arity.
    FE.9 + Notify.5: emite notificação com hint acionável (quando aplicável).
    Ordem: (veredito, waveform, prosody, spectrogram, json)
    """
    # Notify.5: notificação estruturada com hint
    notify_error(
        error_msg,
        hint=hint or "Verifique o arquivo de áudio e tente novamente.",
        error_code=error_code or "ANALYSIS_FAILED",
    )
    payload = {"error": error_msg}
    if hint:
        payload["hint"] = hint
    if error_code:
        payload["error_code"] = error_code
    return (
        render_error(error_msg, hint),
        None,    # plot_waveform
        None,    # plot_prosody
        None,    # plot_spectrogram
        json.dumps(payload, indent=2),
    )


def _nome_do_modelo(service, nome: str) -> str:
    return nome_do_modelo(
        nome,
        getattr(service.loaded_models.get(nome), "architecture", None),
        _ARQUITETURAS_COM_NOME,
    )


def refresh_model_choices():
    """Só os modelos instalados: escolher um sem artefato caía no padrão, em silêncio."""
    escolhas = [("Automático (modelo padrão)", AUTO_MODEL)]
    try:
        service = get_detection_service() if MODELS_AVAILABLE else None
        if service:
            escolhas += sorted(
                ((_nome_do_modelo(service, n), n) for n in service.get_available_models()),
                key=lambda par: par[0].lower(),
            )
    except Exception as exc:  # noqa: BLE001 - a tela abre mesmo sem o serviço
        logger.warning("Lista de modelos indisponível: %s", exc)
    return escolhas


def _resolver_modelo(service, selecionado):
    """Nome do artefato a usar, ou `None` se a escolha não existe mais."""
    disponiveis = service.get_available_models()
    if selecionado and selecionado != AUTO_MODEL:
        return selecionado if selecionado in disponiveis else None
    return service.default_model or (disponiveis[0] if disponiveis else None)


def _cartao_do_resultado(service, data, duracao_s):
    """Monta o veredito a partir do resultado do serviço."""
    metadados = data.metadata or {}
    return render_verdict(
        is_fake=bool(data.is_fake),
        confianca=float(data.confidence),
        pontuacao=metadados.get("fake_score"),
        modelo=_nome_do_modelo(service, data.model_name),
        duracao_s=duracao_s,
        fora_do_padrao=bool(metadados.get("is_ood")),
    )


def analyze_audio(audio_path, model_name, segmented):
    if not audio_path:
        return _empty_analysis_result(
            "Nenhum áudio enviado.",
            hint="Envie um arquivo ou grave pelo microfone.",
            error_code="NO_AUDIO",
        )

    try:
        # Carregar áudio para visualização
        y, sr = librosa.load(audio_path, sr=16000)
        duracao_s = float(len(y) / sr)

        # Gerar plots
        fig_waveform = get_waveform_plot(y, sr)
        fig_prosody = get_prosody_plot(y, sr)
        fig_spectrogram = get_spectrogram_plot(y, sr)

        card = render_error(
            "Detecção indisponível",
            "Os serviços de detecção não foram carregados neste ambiente.",
        )
        details = {}

        if MODELS_AVAILABLE:
            try:
                service = get_detection_service()
                if not service:
                    raise Exception("Serviço de detecção não inicializado")

                modelo_usado = _resolver_modelo(service, model_name)
                if not modelo_usado:
                    if model_name and model_name != AUTO_MODEL:
                        mensagem = "O modelo escolhido não está mais disponível."
                        dica = "Escolha outro modelo na lista."
                    else:
                        mensagem = "Nenhum modelo treinado disponível."
                        dica = "Treine um modelo na aba Treinar."
                    return (
                        render_error(mensagem, dica),
                        fig_waveform,
                        fig_prosody,
                        fig_spectrogram,
                        {"error": mensagem},
                    )

                # Executar Detecção
                result_proc = service.detect_from_file(
                    audio_path,
                    model_name=modelo_usado,
                    segmented=bool(segmented)
                )

                if result_proc.status.name == "SUCCESS":
                    data = result_proc.data
                    result_label = "DEEPFAKE" if data.is_fake else "REAL"
                    card = _cartao_do_resultado(service, data, duracao_s)
                    # PONTO DE OPERACAO no topo dos detalhes, nao enterrado em
                    # `metadata`. O limiar NAO e 0,5: desde a calibracao por
                    # validacao ele varia por modelo (medidos 0,79 no AASIST e
                    # 0,49 no Conformer). Sem isso, "DEEPFAKE, confianca 0,72"
                    # nao diz onde foi o corte — informacao que uma ferramenta
                    # forense precisa expor.
                    _meta = data.metadata or {}
                    _limiar = _meta.get("classification_threshold")
                    details = {
                        "veredito": result_label,
                        "ponto_de_operacao": {
                            "pontuacao_centrada": _meta.get("fake_score"),
                            "limiar_de_decisao": _limiar,
                            "origem": (
                                "EER calibrado na validacao"
                                if _limiar is not None and abs(_limiar - 0.5) > 1e-9
                                else "limiar fixo 0,5"
                            ),
                            "temperatura_aplicada": _meta.get("temperature_applied"),
                        },
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
                    erro = (
                        result_proc.errors[0] if result_proc.errors
                        else "Erro na inferência"
                    )
                    details = {"error": erro}
                    logger.error(f"Erro na inferência: {erro}")
                    card = render_error("Não foi possível analisar o áudio.", str(erro))

            except Exception as e:
                logger.error(f"Erro na inferência: {e}")
                details = {"erro_inferencia": str(e)}
                card = render_error("Não foi possível analisar o áudio.", str(e))

        # Detalhes técnicos adicionais
        details["audio_info"] = {
            "duration": duracao_s,
            "sample_rate": sr,
            "rms_mean": float(np.mean(librosa.feature.rms(y=y)))
        }

        return (
            card,
            fig_waveform,
            fig_prosody,
            fig_spectrogram,
            json.dumps(details, indent=2)
        )

    except Exception as e:
        logger.error(f"Erro em analyze_audio: {e}", exc_info=True)
        return _empty_analysis_result(str(e))


def process_stream(new_chunk, state):
    """Processamento em tempo real do stream de áudio com detecção contínua."""
    try:
        if new_chunk is None:
            return (
                state,
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

        # Acumular — FE.4: cap em _STREAM_MAX_BUFFER_SECONDS para evitar
        # crescimento ilimitado (gravação infinita = vazamento RAM).
        state["audio"] = np.concatenate((state["audio"], data))
        max_samples = _STREAM_MAX_BUFFER_SECONDS * (state.get("sr", sr) or sr)
        if len(state["audio"]) > max_samples:
            # Descarta os samples mais antigos (rolling window)
            state["audio"] = state["audio"][-max_samples:]

        # Otimização: Gerar plots apenas se tiver dados suficientes
        # e não for muito frequente
        y = state["audio"]

        if len(y) < sr * 0.1:  # Menos de 0.1s
            return (
                state,
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
                                label=(
                                    "Frequência fundamental estimada "
                                    f"({int(f0_est)} Hz)"
                                )
                            )
        except Exception as e:
            # FE.7: pitch estimation é best-effort; log em debug
            logger.debug(f"Pitch estimation pulado: {e}")

        ax_pros.legend(loc='upper right', facecolor=_PLOT_FACE,
                       edgecolor=_PLOT_GRID, labelcolor=_PLOT_TEXT, fontsize=9)
        ax_pros.set_xlabel("Tempo (s)")
        fig_pros.tight_layout()

        # 3. Detecção Real-time (Analise Acumulada)
        card_upd = gr.update()

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
                        card_upd = _cartao_do_resultado(
                            service, res.data, float(len(y) / sr)
                        )
                except Exception as e:
                    # FE.7: stream detection é best-effort; log em debug
                    logger.debug(f"Stream detection pulada: {e}")

        return state, fig_wave, fig_pros, fig_spec, card_upd

    except Exception as e:
        logger.error(f"Erro no stream: {e}")
        return (
            state,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update()
        )


def create_detection_tab():
    """Constrói "Um áudio" e "Em lote"; devolve a lista de modelos para atualização."""
    with gr.Tab("🎙️ Um áudio", id="tab_detection"):
        page_header(
            "🎙️",
            "Analisar um áudio",
            "Envie um arquivo ou grave pelo microfone para saber se a voz parece real "
            "ou sintética.",
        )

        with gr.Row(elem_classes="detect-main-grid"):
            # Coluna de Entrada (Esquerda/Topo)
            with gr.Column(scale=1, min_width=320, elem_classes="detect-column"):
                with gr.Group(elem_classes="detect-card"):
                    gr.Markdown("#### 1. Escolha o áudio")
                    with gr.Tabs():
                        with gr.Tab("Enviar arquivo"):
                            audio_file = gr.Audio(
                                type="filepath",
                                label="Arquivo WAV, MP3, FLAC ou OGG",
                                sources=["upload"],
                                elem_classes="detect-audio-input",
                            )
                        with gr.Tab("Gravar"):
                            audio_mic = gr.Audio(
                                type="numpy",
                                label="Microfone",
                                sources=["microphone"],
                                streaming=True,
                                elem_classes="detect-audio-input",
                            )
                    stream_state = gr.State()

                with gr.Group(elem_classes="detect-card"):
                    gr.Markdown("#### 2. Escolha o modelo")
                    model_select = gr.Dropdown(
                        choices=refresh_model_choices(),
                        value=AUTO_MODEL,
                        label="Modelo",
                        info="O automático usa o modelo padrão do sistema.",
                    )
                    segmented_chk = gr.Checkbox(
                        label="Analisar em janelas",
                        value=False,
                        info=(
                            "Divide o áudio e combina os resultados. "
                            "Recomendado acima de 30 segundos."
                        ),
                    )

                analyze_btn = gr.Button(
                    "Analisar áudio", variant="primary", size="lg",
                    elem_classes="detect-primary-action")

            # Coluna de Saída (Direita/Baixo)
            with gr.Column(scale=1, min_width=320, elem_classes="detect-column"):
                with gr.Group(elem_classes="detect-card detect-result-card"):
                    gr.Markdown("#### Resultado")
                    verdict_output = gr.HTML(render_idle(), elem_id="detect_verdict")

                info_callout(
                    "Quer entender o porquê? A aba <b>🔬 Investigar</b> traz fase, "
                    "formantes, jitter/shimmer, HNR e a explicabilidade do modelo.",
                    variant="accent",
                )

        # Os componentes de Plot permanecem recolhidos; a aridade do handler
        # `handle_analysis` (5 saídas) é preservada (lição do bug FE.1).
        with gr.Accordion("Gráficos do áudio", open=False):
            plot_waveform = gr.Plot(label="Forma de onda")

            with gr.Row(elem_classes="responsive-grid plot-grid"):
                with gr.Column(min_width=320):
                    plot_spectrogram = gr.Plot(label="Espectrograma Mel")
                with gr.Column(min_width=320):
                    plot_prosody = gr.Plot(
                        label="Análise prosódica (pitch e energia)")

        with gr.Accordion("Dados técnicos (JSON)", open=False):
            json_output = gr.JSON(label="Saída técnica")

        # Limpar estado ao limpar áudio
        @ui_safe("Falha ao limpar o estado")
        def clear_state():
            return None

        audio_mic.clear(
            fn=clear_state,
            inputs=None,
            outputs=[stream_state]
        )
        # Uma origem substitui a outra: sem isso, um arquivo esquecido na outra aba
        # seria analisado no lugar da gravação recém-feita.
        audio_mic.start_recording(fn=clear_state, inputs=None, outputs=[audio_file])
        audio_file.upload(
            fn=lambda: (None, None), inputs=None, outputs=[audio_mic, stream_state]
        )

        # Wrapper para lidar com upload vs stream
        @ui_safe("Falha ao analisar o áudio")
        def handle_analysis(file_path, mic_chunk, stream_state, model_name, segmented):
            import tempfile

            import numpy as np
            import soundfile as sf

            final_path = file_path or None

            # Gravação: o componente entrega (taxa, amostras).
            if not final_path and isinstance(mic_chunk, tuple):
                sr, data = mic_chunk
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
                    logger.error(f"Erro convertendo áudio numpy: {e}")
                    # FE.1: retorna 5 valores (arity correta)
                    return _empty_analysis_result(f"Falha ao processar o áudio: {e}")

            # Se não há path (ex: microfone stream) mas tem estado acumulado
            if not final_path and stream_state is not None and len(
                    stream_state.get("audio", [])) > 0:
                try:
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
                    # FE.1: retorna 5 valores
                    return _empty_analysis_result(f"Falha ao processar a gravação: {e}")

            if not final_path:
                # FE.1: nada para analisar
                return _empty_analysis_result(
                    "Nenhum áudio enviado.",
                    hint="Envie um arquivo ou grave pelo microfone.",
                    error_code="NO_AUDIO",
                )

            return analyze_audio(final_path, model_name, segmented)

        analyze_btn.click(
            handle_analysis,
            inputs=[
                audio_file,
                audio_mic,
                stream_state,
                model_select,
                segmented_chk],
            outputs=[
                verdict_output,
                plot_waveform,
                plot_prosody,
                plot_spectrogram,
                json_output]
        )

        # Eventos de Streaming (Real-time)
        audio_mic.stream(
            fn=process_stream,
            inputs=[audio_mic, stream_state],
            outputs=[stream_state, plot_waveform, plot_prosody,
                     plot_spectrogram, verdict_output],
            show_progress="hidden"
        )

    # ── 📦 Análise em Lote (sub-aba irmã, movida da antiga aba Forense) ──
    # Reaproveita os handlers de lote definidos em forensic_analysis.py
    # (funções de módulo puras, sem ciclo de import). Import tardio mantém
    # detection.py desacoplado da ordem de carga dos tabs.
    with gr.Tab("📦 Em lote", id="tab_detection_batch"):
        from app.interfaces.gradio.tabs.forensic_analysis import (
            export_batch_report,
            run_batch_analysis,
        )

        page_header(
            "📦",
            "Analisar vários áudios",
            "Envie vários arquivos de uma vez e receba um resumo e um relatório em CSV.",
        )

        batch_files = gr.Files(
            label="Arquivos de áudio",
            file_types=["audio"],
        )
        batch_btn = gr.Button("Analisar arquivos", variant="primary", size="lg")

        batch_summary = gr.Markdown(
            value="*Os resultados aparecem aqui depois da análise.*")

        # Resultados só depois de haver resultado: gráficos vazios parecem erro, e o
        # relatório não tem o que exportar antes da análise.
        with gr.Column(visible=False) as batch_results:
            with gr.Row(elem_classes="responsive-grid plot-grid"):
                with gr.Column():
                    plot_pie = gr.Plot(label="Distribuição de resultados")
                with gr.Column():
                    plot_conf_dist = gr.Plot(label="Distribuição da pontuação")

            batch_table = gr.Dataframe(
                label="Resultados por arquivo",
                headers=["Arquivo", "Resultado", "Confiança", "Modelo", "Duração"],
                interactive=False,
            )

            export_btn = gr.Button("Exportar relatório CSV", variant="secondary")
            report_file = gr.File(label="Relatório", visible=False)

        batch_btn.click(
            fn=run_batch_analysis,
            inputs=[batch_files],
            outputs=[plot_pie, plot_conf_dist, batch_table, batch_summary],
        ).then(
            fn=lambda arquivos: gr.update(visible=bool(arquivos)),
            inputs=[batch_files],
            outputs=[batch_results],
            show_progress="hidden",
        )
        export_btn.click(
            fn=export_batch_report,
            inputs=[batch_files],
            outputs=[report_file],
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[report_file],
            show_progress="hidden",
        )

    return model_select
