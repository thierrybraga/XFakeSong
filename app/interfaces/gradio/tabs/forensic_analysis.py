"""Aba de Analise Forense Completa para o Gradio UI.

Integra as medidas de `domain/services/forensic_measures` e os graficos de
`gradio/presenters/forensic_plots` com a interface Gradio.
"""

import csv
import logging
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from app.core.contracts.audio import AudioData
from app.domain.services.forensic_measures import (
    RADAR_AXES,
    batch_report,
    radar_axes,
)
from app.interfaces.gradio.presenters.forensic_plots import (
    AudioForensicVisualizer,
    BatchAnalysisVisualizer,
)
from app.interfaces.gradio.utils.components import page_header
from app.interfaces.gradio.utils.plotting import get_service_lock
from app.utils.file_utils import get_gradio_exports_directory

logger = logging.getLogger("gradio_forensic_tab")

# Singleton para servico de deteccao (mesmo padrão thread-safe de detection.py)
_detection_service = None
_service_lock = get_service_lock("forensic_detection_service")


def _get_detection_service():
    global _detection_service
    if _detection_service is not None:
        return _detection_service
    with _service_lock:
        if _detection_service is None:
            try:
                # Mesmo singleton (data/models) da API/Detectar/wizard — ver detection.py.
                from app.dependencies import get_detection_service as _shared
                _detection_service = _shared()
            except Exception as e:
                logger.warning(f"Detection service unavailable: {e}")
                return None
    return _detection_service


def _limiar_de(result, padrao: float = 0.5) -> float:
    """Limiar que decidiu o veredito: é por modelo, não um 0,5 fixo."""
    limiar = (result.data.metadata or {}).get("classification_threshold")
    return padrao if limiar is None else float(limiar)


def _pontuacao_por_segmento(service, y, sr, segment_duration: float = 1.0):
    """``(tempos, p_fake, limiar)`` por trecho, com 50% de sobreposição.

    ``result.data.confidence`` é a confiança do VEREDITO: ``p_real`` quando ele é
    real. Plotada como probabilidade de fake, invertia a leitura: num áudio real
    com ``p_fake`` 0,02, a linha marcava 0,98 e pintava o trecho como fake.

    Trecho em que a detecção falha fica de fora. Antes entrava como 0,5, um número
    que ninguém mediu e que cai de um lado ou do outro do limiar do modelo.
    """
    segment_samples = int(sr * segment_duration)
    hop_samples = segment_samples // 2  # 50% overlap
    tempos, pontos = [], []
    limiar = 0.5
    for start in range(0, len(y) - segment_samples, hop_samples):
        segment = y[start:start + segment_samples]
        seg_audio = AudioData(
            samples=segment, sample_rate=sr, duration=segment_duration
        )
        try:
            result = service.detect_single(seg_audio)
            if result.status.name == "SUCCESS":
                pontos.append(float(result.data.probabilities["fake"]))
                tempos.append((start + segment_samples / 2) / sr)
                limiar = _limiar_de(result, limiar)
        except Exception as e:
            # FE.7: log do segmento que falhou (útil em loops grandes)
            logger.debug(
                f"Segmento [{start}:{start+segment_samples}] falhou: {e}"
            )
    return np.array(tempos), np.array(pontos), limiar


def run_forensic_analysis(audio_path):
    """Handler principal: executa todas as visualizacoes forenses."""
    if not audio_path:
        empty = (None,) * 13 + ("Nenhum arquivo selecionado.",)
        return empty

    try:
        import librosa

        # Carregar audio
        y, sr = librosa.load(audio_path, sr=16000)
        viz = AudioForensicVisualizer(sr=sr)

        # Gerar todas as visualizacoes
        fig_multi_spec = viz.plot_multi_spectrogram(y)
        fig_phase = viz.plot_phase_spectrum(y)
        fig_chroma = viz.plot_chromagram(y)
        fig_formants = viz.plot_spectral_envelope_formants(y)
        fig_zcr = viz.plot_zcr(y)
        fig_spectral = viz.plot_spectral_temporal(y)
        fig_energy = viz.plot_energy_contour(y)
        fig_hnr = viz.plot_hnr_temporal(y)
        fig_jitter = viz.plot_jitter_shimmer(y)
        fig_f0 = viz.plot_f0_detailed(y)

        # Analise de deteccao (requer modelo treinado)
        fig_radar = None
        fig_anomaly = None
        fig_confidence = None

        service = _get_detection_service()
        if service:
            try:
                # Analise segmentada: P(fake) e o limiar que decide
                seg_times, seg_scores, limiar = _pontuacao_por_segmento(
                    service, y, sr
                )
                if len(seg_scores):
                    fig_confidence = viz.plot_fake_score_timeline(
                        seg_times, seg_scores, limiar)

                    # Anomaly heatmap
                    fig_anomaly = viz.plot_anomaly_heatmap(
                        y, seg_scores, seg_times)

                # Radar de explicabilidade: oito medidas do sinal, cada uma
                # normalizada pela sua faixa plausível (ver forensic_measures).
                try:
                    fig_radar = viz.plot_feature_importance_radar(
                        list(RADAR_AXES), radar_axes(y, sr))
                except Exception as e:
                    logger.warning(f"Feature importance computation failed: {e}")

            except Exception as e:
                logger.warning(f"Detection-based analysis failed: {e}")

        # Info summary
        duration = len(y) / sr
        info = (
            f"**Áudio:** {Path(audio_path).name}\n"
            f"**Duração:** {duration:.2f}s | "
            f"**Sample Rate:** {sr} Hz | "
            f"**Amostras:** {len(y):,}\n"
            f"**RMS Médio:** {float(np.sqrt(np.mean(y**2))):.4f} | "
            f"**Max Amplitude:** {float(np.max(np.abs(y))):.4f}"
        )

        return (
            fig_multi_spec, fig_phase, fig_chroma,
            fig_formants, fig_zcr, fig_spectral,
            fig_energy, fig_hnr, fig_jitter, fig_f0,
            fig_radar, fig_anomaly, fig_confidence,
            info
        )

    except Exception as e:
        logger.error(f"Forensic analysis error: {e}")
        import traceback
        traceback.print_exc()
        empty = (None,) * 13 + (f"Erro: {str(e)}",)
        return empty


def run_batch_analysis(files):
    """Handler de analise em lote."""
    if not files:
        return None, None, None, "Nenhum arquivo enviado."

    try:
        import librosa
        service = _get_detection_service()

        results = []
        limiar = 0.5
        for file_obj in files:
            file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
            filename = Path(file_path).name

            try:
                y, sr = librosa.load(file_path, sr=16000)
                audio_data = AudioData(
                    samples=y, sample_rate=sr,
                    duration=float(len(y) / sr)
                )

                if service:
                    result = service.detect_single(audio_data)
                    if result.status.name == "SUCCESS":
                        limiar = _limiar_de(result, limiar)
                        results.append({
                            'filename': filename,
                            'is_fake': result.data.is_fake,
                            'confidence': float(result.data.confidence),
                            'p_fake': float(result.data.probabilities["fake"]),
                            'model_name': result.data.model_name,
                            'duration': float(len(y) / sr)
                        })
                    else:
                        results.append({
                            'filename': filename,
                            'is_fake': False,
                            'confidence': 0.0,
                            'model_name': 'error',
                            'duration': float(len(y) / sr)
                        })
                else:
                    results.append({
                        'filename': filename,
                        'is_fake': False,
                        'confidence': 0.0,
                        'model_name': 'no_model',
                        'duration': float(len(y) / sr)
                    })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'is_fake': False,
                    'confidence': 0.0,
                    'model_name': f'error: {e}',
                    'duration': 0.0
                })

        if not results:
            return None, None, None, "Nenhum resultado obtido."

        batch_viz = BatchAnalysisVisualizer()

        n_fake = sum(1 for r in results if r['is_fake'])
        n_real = len(results) - n_fake
        confidences = np.array([r['confidence'] for r in results])

        fig_pie = batch_viz.plot_batch_summary_pie(n_real, n_fake)
        # A tabela mostra a confiança do veredito; o histograma, P(fake) e só do
        # que foi medido: arquivo que falhou tem confiança 0, mas nenhum P(fake).
        p_fakes = np.array([r['p_fake'] for r in results if 'p_fake' in r])
        fig_conf = batch_viz.plot_fake_score_distribution(p_fakes, limiar)

        # Build results table
        table_data = []
        for r in results:
            table_data.append([
                r['filename'],
                'FAKE' if r['is_fake'] else 'REAL',
                f"{r['confidence']:.4f}",
                r['model_name'],
                f"{r['duration']:.2f}s"
            ])

        df = pd.DataFrame(table_data, columns=[
            'Arquivo', 'Resultado', 'Confiança', 'Modelo', 'Duração'
        ])

        summary = (
            f"**Total analisado:** {len(results)}\n"
            f"**Fake:** {n_fake} ({n_fake/len(results)*100:.1f}%) | "
            f"**Real:** {n_real} ({n_real/len(results)*100:.1f}%)\n"
            f"**Confiança média:** {float(np.mean(confidences)):.4f}"
        )

        return fig_pie, fig_conf, df, summary

    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return None, None, None, f"Erro: {str(e)}"


def export_batch_report(files):
    """Exporta relatorio CSV da analise em lote."""
    if not files:
        return None

    try:
        import librosa
        service = _get_detection_service()
        results = []
        for file_obj in files:
            file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
            filename = Path(file_path).name

            try:
                y, sr = librosa.load(file_path, sr=16000)
                audio_data = AudioData(
                    samples=y, sample_rate=sr,
                    duration=float(len(y) / sr)
                )
                if service:
                    result = service.detect_single(audio_data)
                    if result.status.name == "SUCCESS":
                        results.append({
                            'filename': filename,
                            'is_fake': result.data.is_fake,
                            'confidence': float(result.data.confidence),
                            'model_name': result.data.model_name,
                            'duration': float(len(y) / sr)
                        })
                        continue
            except Exception as e:
                # FE.7: log do nome do arquivo que falhou para debug
                logger.debug(f"Detection batch falhou em {filename}: {e}")
            results.append({
                'filename': filename,
                'is_fake': False,
                'confidence': 0.0,
                'model_name': 'error',
                'duration': 0.0
            })

        report = batch_report(results)

        # Write CSV
        tmp = tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False, mode='w', newline='',
            encoding='utf-8',
            dir=get_gradio_exports_directory())
        writer = csv.writer(tmp)
        writer.writerow([
            'Arquivo', 'Resultado', 'Confiança', 'Modelo', 'Duração(s)'
        ])
        for r in results:
            writer.writerow([
                r['filename'],
                'FAKE' if r['is_fake'] else 'REAL',
                f"{r['confidence']:.6f}",
                r['model_name'],
                f"{r['duration']:.2f}"
            ])

        writer.writerow([])
        writer.writerow(['--- Resumo ---'])
        writer.writerow(['Total', report['total_analyzed']])
        writer.writerow(['Fake', report['total_fake']])
        writer.writerow(['Real', report['total_real']])
        writer.writerow(['% Fake', f"{report['fake_percentage']:.2f}%"])
        writer.writerow(['Confiança Média', f"{report['confidence_mean']:.6f}"])
        writer.writerow(['Confiança Std', f"{report['confidence_std']:.6f}"])
        tmp.close()

        return tmp.name

    except Exception as e:
        logger.error(f"Report export error: {e}")
        return None


def create_forensic_analysis_tab():
    """Cria a aba de Analise Forense no Gradio."""

    with gr.Tab("🔬 Investigar", id="tab_forensic"):
        page_header(
            "🔬",
            "Investigar",
            "Análise forense completa para perícia em áudios digitais: "
            "espectrogramas, fase, formantes, jitter/shimmer, HNR e "
            "explicabilidade do modelo.",
        )

        # ===== Input Section =====
        with gr.Row(elem_classes="responsive-grid"):
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    type="filepath",
                    label="Arquivo de Áudio para Análise",
                    sources=["upload"]
                )
                analyze_btn = gr.Button(
                    "▶ Executar Análise Forense",
                    variant="primary", size="lg"
                )
            with gr.Column(scale=2):
                info_output = gr.Markdown(
                    label="Informações do Áudio",
                    value="*Faça upload de um arquivo de áudio para iniciar.*"
                )

        gr.Markdown("---")

        # ===== Forensic Visualizations (Sub-tabs) =====
        with gr.Tabs():

            # --- Tab: Espectrogramas ---
            with gr.Tab("📊 Espectrogramas"):
                gr.Markdown("#### Comparação Multi-Espectrograma")
                plot_multi_spec = gr.Plot(
                    label="Mel / STFT / CQT / LFCC")
                with gr.Row(elem_classes="responsive-grid plot-grid"):
                    with gr.Column():
                        gr.Markdown("#### Espectro de Fase")
                        plot_phase = gr.Plot(
                            label="Fase STFT (revela artefatos de sintese)")
                    with gr.Column():
                        gr.Markdown("#### Cromograma")
                        plot_chroma = gr.Plot(
                            label="Características cromáticas (12 classes de pitch)")

            # --- Tab: Análise Espectral ---
            with gr.Tab("🌈 Análise Espectral"):
                gr.Markdown("#### Envelope espectral e formantes")
                plot_formants = gr.Plot(
                    label="Espectrograma com trilhas de formantes F1-F4")
                with gr.Row(elem_classes="responsive-grid plot-grid"):
                    with gr.Column():
                        gr.Markdown("#### Taxa de cruzamento por zero")
                        plot_zcr = gr.Plot(
                            label="Taxa de cruzamento por zero temporal")
                    with gr.Column():
                        gr.Markdown("#### Características espectrais e temporais")
                        plot_spectral = gr.Plot(
                            label="Centroide / Rolloff / Bandwidth")

            # --- Tab: Análise Temporal ---
            with gr.Tab("⏱️ Análise Temporal"):
                gr.Markdown("#### Contorno de Energia")
                plot_energy = gr.Plot(
                    label="Energia RMS com detecção de silêncio")
                with gr.Row(elem_classes="responsive-grid plot-grid"):
                    with gr.Column():
                        gr.Markdown("#### HNR Temporal")
                        plot_hnr = gr.Plot(
                            label="Relação harmônico-ruído por quadro")
                    with gr.Column():
                        gr.Markdown("#### Jitter / Shimmer")
                        plot_jitter = gr.Plot(
                            label="Microperturbações vocais")

            # --- Tab: Análise de Pitch ---
            with gr.Tab("🎵 Análise de Pitch"):
                gr.Markdown("#### Análise Detalhada de F0")
                plot_f0 = gr.Plot(
                    label=(
                        "Contorno de frequência fundamental com confiança "
                        "e vibrato"
                    )
                )

            # --- Tab: Explicabilidade (XAI) ---
            # (renomeada de "Análise de Detecção" — são visualizações de
            # explicabilidade do modelo, não um detector duplicado)
            with gr.Tab("🧠 Explicabilidade (XAI)"):
                gr.Markdown(
                    "#### Visualizações baseadas no modelo de detecção\n"
                    "*Requer modelo treinado carregado.*"
                )
                with gr.Row(elem_classes="responsive-grid plot-grid"):
                    with gr.Column():
                        plot_radar = gr.Plot(
                            label="Importância relativa das características")
                    with gr.Column():
                        plot_anomaly = gr.Plot(
                            label="Mapa de calor de anomalias sobre o espectrograma"
                        )
                gr.Markdown("#### Probabilidade de fake por segmento")
                plot_confidence = gr.Plot(
                    label="P(fake) ao longo do áudio, com o limiar do modelo")

        # ===== Event Handlers =====
        # NOTA: a análise em lote foi movida para a aba 🎯 Detectar →
        # 📦 Análise em Lote. As funções run_batch_analysis/export_batch_report
        # permanecem neste módulo e são importadas lá.

        # Single file analysis
        analyze_btn.click(
            fn=run_forensic_analysis,
            inputs=[audio_input],
            outputs=[
                plot_multi_spec, plot_phase, plot_chroma,
                plot_formants, plot_zcr, plot_spectral,
                plot_energy, plot_hnr, plot_jitter, plot_f0,
                plot_radar, plot_anomaly, plot_confidence,
                info_output
            ]
        )
