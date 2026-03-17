"""Aba de Analise Forense Completa para o Gradio UI.

Integra o motor de visualizacao forense com a interface Gradio.
"""

import csv
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import gradio as gr
from app.core.interfaces.audio import AudioData
from app.domain.services.forensic_visualization import (
    AudioForensicVisualizer,
    BatchAnalysisVisualizer,
)

logger = logging.getLogger("gradio_forensic_tab")

# Singleton para servico de deteccao
_detection_service = None


def _get_detection_service():
    global _detection_service
    if _detection_service is None:
        try:
            from app.domain.services.detection_service import DetectionService
            _detection_service = DetectionService()
        except Exception as e:
            logger.warning(f"Detection service unavailable: {e}")
            return None
    return _detection_service


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
                # Analise segmentada para confianca temporal
                segment_duration = 1.0  # 1 segundo
                segment_samples = int(sr * segment_duration)
                hop_samples = segment_samples // 2  # 50% overlap

                segment_times = []
                segment_confidences = []

                audio_data_full = AudioData(
                    samples=y, sample_rate=sr,
                    duration=float(len(y) / sr)
                )

                # Deteccao por segmento
                for start in range(0, len(y) - segment_samples, hop_samples):
                    segment = y[start:start + segment_samples]
                    seg_audio = AudioData(
                        samples=segment, sample_rate=sr,
                        duration=segment_duration
                    )
                    try:
                        result = service.detect_single(seg_audio)
                        if result.status.name == "SUCCESS":
                            conf = float(result.data.confidence)
                            segment_confidences.append(conf)
                            segment_times.append(
                                (start + segment_samples / 2) / sr)
                    except Exception:
                        segment_confidences.append(0.5)
                        segment_times.append(
                            (start + segment_samples / 2) / sr)

                if segment_confidences:
                    seg_times = np.array(segment_times)
                    seg_confs = np.array(segment_confidences)

                    # Confidence timeline
                    fig_confidence = viz.plot_confidence_timeline(
                        seg_times, seg_confs)

                    # Anomaly heatmap
                    fig_anomaly = viz.plot_anomaly_heatmap(
                        y, seg_confs, seg_times)

                # Feature importance radar (mock baseado em features usadas)
                feature_names = [
                    'Spectral', 'Cepstral', 'Temporal', 'Prosodic',
                    'Phase', 'Formant', 'HNR', 'Jitter/Shimmer'
                ]

                # Compute simple importance from feature statistics
                importances = []
                try:
                    import librosa
                    # Spectral centroid variability
                    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                    importances.append(float(np.std(sc) / (np.mean(sc) + 1e-10)))

                    # MFCC variability
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    importances.append(float(np.mean(np.std(mfcc, axis=1))))

                    # ZCR variability
                    zcr = librosa.feature.zero_crossing_rate(y)[0]
                    importances.append(float(np.std(zcr)))

                    # RMS variability
                    rms = librosa.feature.rms(y=y)[0]
                    importances.append(float(np.std(rms) / (np.mean(rms) + 1e-10)))

                    # Phase discontinuity
                    D = librosa.stft(y)
                    phase = np.angle(D)
                    phase_diff = np.diff(phase, axis=1)
                    importances.append(float(np.mean(np.abs(phase_diff))))

                    # Formant stability (approximated via LPC)
                    importances.append(0.5)  # placeholder

                    # HNR (approximated)
                    importances.append(0.4)  # placeholder

                    # Jitter (approximated)
                    importances.append(0.3)  # placeholder

                    importances = np.array(importances)
                    fig_radar = viz.plot_feature_importance_radar(
                        feature_names, importances)
                except Exception as e:
                    logger.warning(f"Feature importance computation failed: {e}")

            except Exception as e:
                logger.warning(f"Detection-based analysis failed: {e}")

        # Info summary
        duration = len(y) / sr
        info = (
            f"**Audio:** {Path(audio_path).name}\n"
            f"**Duracao:** {duration:.2f}s | "
            f"**Sample Rate:** {sr} Hz | "
            f"**Amostras:** {len(y):,}\n"
            f"**RMS Medio:** {float(np.sqrt(np.mean(y**2))):.4f} | "
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
        fig_conf = batch_viz.plot_confidence_distribution(confidences)

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
            'Arquivo', 'Resultado', 'Confianca', 'Modelo', 'Duracao'
        ])

        summary = (
            f"**Total analisado:** {len(results)}\n"
            f"**Fake:** {n_fake} ({n_fake/len(results)*100:.1f}%) | "
            f"**Real:** {n_real} ({n_real/len(results)*100:.1f}%)\n"
            f"**Confianca media:** {float(np.mean(confidences)):.4f}"
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
        batch_viz = BatchAnalysisVisualizer()

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
            except Exception:
                pass
            results.append({
                'filename': filename,
                'is_fake': False,
                'confidence': 0.0,
                'model_name': 'error',
                'duration': 0.0
            })

        report = batch_viz.generate_forensic_report_data(results)

        # Write CSV
        tmp = tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False, mode='w', newline='',
            encoding='utf-8')
        writer = csv.writer(tmp)
        writer.writerow([
            'Arquivo', 'Resultado', 'Confianca', 'Modelo', 'Duracao(s)'
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
        writer.writerow(['Confianca Media', f"{report['confidence_mean']:.6f}"])
        writer.writerow(['Confianca Std', f"{report['confidence_std']:.6f}"])
        tmp.close()

        return tmp.name

    except Exception as e:
        logger.error(f"Report export error: {e}")
        return None


def create_forensic_analysis_tab():
    """Cria a aba de Analise Forense no Gradio."""

    with gr.Tab("🔬 Análise Forense", id="tab_forensic"):
        gr.Markdown(
            "### Análise Forense de Áudio\n"
            "Sistema completo de visualização gráfica para perícia em áudios digitais. "
            "Gera espectrogramas, análises de fase, formantes, jitter/shimmer, "
            "HNR e muito mais."
        )

        # ===== Input Section =====
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    type="filepath",
                    label="Arquivo de Audio para Analise",
                    sources=["upload"]
                )
                analyze_btn = gr.Button(
                    "▶ Executar Análise Forense",
                    variant="primary", size="lg"
                )
            with gr.Column(scale=2):
                info_output = gr.Markdown(
                    label="Informacoes do Audio",
                    value="*Faca upload de um arquivo de audio para iniciar.*"
                )

        gr.Markdown("---")

        # ===== Forensic Visualizations (Sub-tabs) =====
        with gr.Tabs():

            # --- Tab: Espectrogramas ---
            with gr.Tab("📊 Espectrogramas"):
                gr.Markdown("#### Comparação Multi-Espectrograma")
                plot_multi_spec = gr.Plot(
                    label="Mel / STFT / CQT / LFCC")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Espectro de Fase")
                        plot_phase = gr.Plot(
                            label="Fase STFT (revela artefatos de sintese)")
                    with gr.Column():
                        gr.Markdown("#### Cromograma")
                        plot_chroma = gr.Plot(
                            label="Chroma Features (12 classes de pitch)")

            # --- Tab: Análise Espectral ---
            with gr.Tab("🌈 Análise Espectral"):
                gr.Markdown("#### Envelope Espectral & Formantes")
                plot_formants = gr.Plot(
                    label="Espectrograma com trilhas de formantes F1-F4")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Zero-Crossing Rate")
                        plot_zcr = gr.Plot(
                            label="Taxa de cruzamento por zero temporal")
                    with gr.Column():
                        gr.Markdown("#### Features Espectrais Temporais")
                        plot_spectral = gr.Plot(
                            label="Centroide / Rolloff / Bandwidth")

            # --- Tab: Análise Temporal ---
            with gr.Tab("⏱️ Análise Temporal"):
                gr.Markdown("#### Contorno de Energia")
                plot_energy = gr.Plot(
                    label="RMS Energy com deteccao de silencio")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### HNR Temporal")
                        plot_hnr = gr.Plot(
                            label="Harmonic-to-Noise Ratio por frame")
                    with gr.Column():
                        gr.Markdown("#### Jitter / Shimmer")
                        plot_jitter = gr.Plot(
                            label="Micro-perturbacoes vocais")

            # --- Tab: Análise de Pitch ---
            with gr.Tab("🎵 Análise de Pitch"):
                gr.Markdown("#### Análise Detalhada de F0")
                plot_f0 = gr.Plot(
                    label="Contorno de pitch com confianca e vibrato")

            # --- Tab: Análise de Detecção ---
            with gr.Tab("🎯 Análise de Detecção"):
                gr.Markdown(
                    "#### Visualizações baseadas no modelo de detecção\n"
                    "*Requer modelo treinado carregado.*"
                )
                with gr.Row():
                    with gr.Column():
                        plot_radar = gr.Plot(
                            label="Importancia Relativa de Features")
                    with gr.Column():
                        plot_anomaly = gr.Plot(
                            label="Heatmap de Anomalias sobre Espectrograma")
                gr.Markdown("#### Confianca por Segmento Temporal")
                plot_confidence = gr.Plot(
                    label="Score de confianca ao longo do audio")

            # --- Tab: Análise em Lote ---
            with gr.Tab("📦 Análise em Lote"):
                gr.Markdown(
                    "#### Análise de Múltiplos Arquivos\n"
                    "Faça upload de vários arquivos de áudio para análise em lote "
                    "com relatório exportável em CSV."
                )

                batch_files = gr.Files(
                    label="Upload de Arquivos de Audio",
                    file_types=["audio"]
                )
                with gr.Row():
                    batch_btn = gr.Button(
                        "Analisar Lote", variant="primary")
                    export_btn = gr.Button(
                        "Exportar Relatorio CSV", variant="secondary")

                batch_summary = gr.Markdown(
                    value="*Faca upload de arquivos para analise em lote.*")

                with gr.Row():
                    with gr.Column():
                        plot_pie = gr.Plot(
                            label="Distribuicao de Resultados")
                    with gr.Column():
                        plot_conf_dist = gr.Plot(
                            label="Distribuicao de Confianca")

                batch_table = gr.Dataframe(
                    label="Resultados por Arquivo",
                    headers=['Arquivo', 'Resultado', 'Confianca',
                             'Modelo', 'Duracao'],
                    interactive=False
                )

                report_file = gr.File(
                    label="Download Relatorio", visible=True)

        # ===== Event Handlers =====

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

        # Batch analysis
        batch_btn.click(
            fn=run_batch_analysis,
            inputs=[batch_files],
            outputs=[plot_pie, plot_conf_dist, batch_table, batch_summary]
        )

        # Export report
        export_btn.click(
            fn=export_batch_report,
            inputs=[batch_files],
            outputs=[report_file]
        )
