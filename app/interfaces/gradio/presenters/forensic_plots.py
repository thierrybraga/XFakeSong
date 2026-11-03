"""Gráficos forenses da aba Investigar e da análise do assistente de treino.

Só apresentação: as medidas vêm de ``app.domain.services.forensic_measures``
(P1-7). Toda função devolve ``matplotlib.figure.Figure`` pela API OOP, sem
``pyplot`` — seguro entre as threads do Gradio.
"""

import logging
from typing import List, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from app.domain.services.forensic_measures import (
    det_curve,
    formant_tracks,
    hnr_contour,
    jitter_shimmer,
    linear_cepstrum,
    pitch_analysis,
    threshold_sweep,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Forensic Theme
# =============================================================================

class ForensicTheme:
    """Tema visual escuro consistente para todas as visualizacoes forenses.

    Paleta alinhada com o design system SpeechChecker (slate/blue dark theme).
    """

    BG_COLOR = '#0f172a'
    BG_COLOR_LIGHT = '#1e293b'
    PANEL_COLOR = '#334155'
    TEXT_COLOR = '#f1f5f9'
    GRID_COLOR = '#334155'
    ACCENT_CYAN = '#06b6d4'
    ACCENT_ORANGE = '#f59e0b'
    ACCENT_GREEN = '#10b981'
    ACCENT_RED = '#ef4444'
    ACCENT_PURPLE = '#a78bfa'
    ACCENT_YELLOW = '#eab308'
    ACCENT_PINK = '#ec4899'
    ACCENT_BLUE = '#3b82f6'

    PALETTE = [
        ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,
        ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_PINK, ACCENT_BLUE,
    ]

    @staticmethod
    def apply(fig: Figure, axes=None) -> None:
        """Aplica tema forense a uma figura e seus eixos."""
        fig.patch.set_facecolor(ForensicTheme.BG_COLOR)

        if axes is None:
            axes = fig.get_axes()
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax in axes:
            ax.set_facecolor(ForensicTheme.BG_COLOR_LIGHT)
            ax.tick_params(colors=ForensicTheme.TEXT_COLOR, which='both')
            ax.xaxis.label.set_color(ForensicTheme.TEXT_COLOR)
            ax.yaxis.label.set_color(ForensicTheme.TEXT_COLOR)
            ax.title.set_color(ForensicTheme.TEXT_COLOR)
            for spine in ax.spines.values():
                spine.set_color(ForensicTheme.GRID_COLOR)
            ax.grid(True, alpha=0.15, color=ForensicTheme.GRID_COLOR)

    @staticmethod
    def get_anomaly_cmap():
        """Colormap para heatmap de anomalias: verde -> amarelo -> vermelho."""
        colors = ['#00e676', '#ffd600', '#ff6b35', '#ff1744']
        return LinearSegmentedColormap.from_list('forensic_anomaly', colors, N=256)

    @staticmethod
    def get_spectrogram_cmap():
        """Colormap para espectrogramas forenses: escuro -> ciano -> branco."""
        colors = ['#0a0a1a', '#0f3460', '#00769e', '#00d4ff', '#ffffff']
        return LinearSegmentedColormap.from_list('forensic_spec', colors, N=256)


# =============================================================================
# Audio Forensic Visualizer
# =============================================================================

class AudioForensicVisualizer:
    """Todas as visualizacoes forenses de audio."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    # -----------------------------------------------------------------
    # 1. Multi-Spectrogram Comparison
    # -----------------------------------------------------------------
    def plot_multi_spectrogram(self, y: np.ndarray) -> Figure:
        """4 paineis: Mel, STFT magnitude, CQT, LFCC."""
        import librosa

        fig = Figure(figsize=(16, 10))
        cmap = ForensicTheme.get_spectrogram_cmap()

        # Mel Spectrogram
        ax1 = fig.add_subplot(2, 2, 1)
        S_mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
        img1 = ax1.imshow(S_mel_db, aspect='auto', origin='lower', cmap=cmap,
                          extent=[0, len(y) / self.sr, 0, self.sr / 2])
        ax1.set_title('Mel Spectrogram')
        ax1.set_ylabel('Frequência (Hz)')
        fig.colorbar(img1, ax=ax1, format='%+.0f dB', fraction=0.046)

        # STFT Magnitude
        ax2 = fig.add_subplot(2, 2, 2)
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img2 = ax2.imshow(S_db, aspect='auto', origin='lower', cmap=cmap,
                          extent=[0, len(y) / self.sr, 0, self.sr / 2])
        ax2.set_title('STFT Magnitude (dB)')
        ax2.set_ylabel('Frequência (Hz)')
        fig.colorbar(img2, ax=ax2, format='%+.0f dB', fraction=0.046)

        # CQT
        ax3 = fig.add_subplot(2, 2, 3)
        try:
            C = np.abs(librosa.cqt(y, sr=self.sr, hop_length=512))
            C_db = librosa.amplitude_to_db(C, ref=np.max)
            img3 = ax3.imshow(C_db, aspect='auto', origin='lower', cmap=cmap,
                              extent=[0, len(y) / self.sr, 0, C_db.shape[0]])
            ax3.set_title('Constant-Q Transform')
            ax3.set_ylabel('Bins CQT')
            fig.colorbar(img3, ax=ax3, format='%+.0f dB', fraction=0.046)
        except Exception as e:
            ax3.text(0.5, 0.5, f'CQT indisponivel:\n{e}',
                     ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                     transform=ax3.transAxes)
            ax3.set_title('Constant-Q Transform')

        # LFCC (Linear Frequency Cepstral Coefficients)
        ax4 = fig.add_subplot(2, 2, 4)
        try:
            lfcc = linear_cepstrum(y)
            img4 = ax4.imshow(lfcc, aspect='auto', origin='lower', cmap=cmap,
                              extent=[0, len(y) / self.sr, 0, 20])
            ax4.set_title('LFCC (Linear Frequency Cepstral Coefficients)')
            ax4.set_ylabel('Coeficiente')
            fig.colorbar(img4, ax=ax4, fraction=0.046)
        except Exception as e:
            ax4.text(0.5, 0.5, f'LFCC indisponivel:\n{e}',
                     ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                     transform=ax4.transAxes)
            ax4.set_title('LFCC')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel('Tempo (s)')

        fig.suptitle('Comparação de espectrogramas', fontsize=14,
                     color=ForensicTheme.TEXT_COLOR, fontweight='bold')
        ForensicTheme.apply(fig)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # -----------------------------------------------------------------
    # 2. Phase Spectrum
    # -----------------------------------------------------------------
    def plot_phase_spectrum(self, y: np.ndarray) -> Figure:
        """Espectro de fase do STFT — revela fronteiras de sintese."""
        import librosa

        fig = Figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        D = librosa.stft(y, n_fft=2048, hop_length=512)
        phase = np.angle(D)

        img = ax.imshow(phase, aspect='auto', origin='lower', cmap='twilight',
                        extent=[0, len(y) / self.sr, 0, self.sr / 2],
                        vmin=-np.pi, vmax=np.pi)
        ax.set_title('Espectro de Fase (STFT)')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Frequência (Hz)')
        fig.colorbar(img, ax=ax, label='Fase (rad)', fraction=0.046)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 3. Chromagram
    # -----------------------------------------------------------------
    def plot_chromagram(self, y: np.ndarray) -> Figure:
        """Chroma features (12 bins)."""
        import librosa

        fig = Figure(figsize=(12, 4))
        ax = fig.add_subplot(111)

        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, hop_length=512)
        cmap = ForensicTheme.get_spectrogram_cmap()
        img = ax.imshow(chroma, aspect='auto', origin='lower', cmap=cmap,
                        extent=[0, len(y) / self.sr, 0, 12])
        ax.set_title('Cromograma')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Classe de Pitch')
        ax.set_yticks(np.arange(0.5, 12.5))
        ax.set_yticklabels(['C', 'C#', 'D', 'D#', 'E', 'F',
                            'F#', 'G', 'G#', 'A', 'A#', 'B'])
        fig.colorbar(img, ax=ax, fraction=0.046)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 4. Spectral Envelope & Formants
    # -----------------------------------------------------------------
    def plot_spectral_envelope_formants(self, y: np.ndarray) -> Figure:
        """Espectrograma com trilhas de formantes F1-F4 sobrepostas."""
        import librosa

        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        # Base spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        cmap = ForensicTheme.get_spectrogram_cmap()
        ax.imshow(S_db, aspect='auto', origin='lower', cmap=cmap,
                  extent=[0, len(y) / self.sr, 0, self.sr / 2])

        try:
            times, tracks = formant_tracks(y, self.sr)
            colors = [ForensicTheme.ACCENT_CYAN, ForensicTheme.ACCENT_ORANGE,
                      ForensicTheme.ACCENT_GREEN, ForensicTheme.ACCENT_RED]
            for i, (name, track) in enumerate(tracks.items()):
                ax.plot(times, track, color=colors[i], linewidth=1.5,
                        alpha=0.8, label=name)

            ax.legend(loc='upper right', fontsize=9,
                      facecolor=ForensicTheme.BG_COLOR, labelcolor=ForensicTheme.TEXT_COLOR)
        except Exception as e:
            logger.warning(f"Formant extraction failed: {e}")

        ax.set_title('Envelope Espectral & Formantes')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Frequência (Hz)')

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 5. Zero-Crossing Rate
    # -----------------------------------------------------------------
    def plot_zcr(self, y: np.ndarray) -> Figure:
        """Taxa de cruzamento por zero ao longo do tempo."""
        import librosa

        fig = Figure(figsize=(12, 4))
        ax = fig.add_subplot(111)

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
        times = librosa.times_like(zcr, sr=self.sr, hop_length=512)

        ax.fill_between(times, zcr, alpha=0.3, color=ForensicTheme.ACCENT_CYAN)
        ax.plot(times, zcr, color=ForensicTheme.ACCENT_CYAN, linewidth=1)

        # Moving average
        if len(zcr) > 20:
            kernel = np.ones(20) / 20
            zcr_smooth = np.convolve(zcr, kernel, mode='same')
            ax.plot(times, zcr_smooth, color=ForensicTheme.ACCENT_ORANGE,
                    linewidth=2, label='Media Movel')
            ax.legend(facecolor=ForensicTheme.BG_COLOR,
                      labelcolor=ForensicTheme.TEXT_COLOR)

        ax.set_title('Taxa de Cruzamento por Zero (ZCR)')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('ZCR')

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 6. Spectral Features Over Time
    # -----------------------------------------------------------------
    def plot_spectral_temporal(self, y: np.ndarray) -> Figure:
        """3 subplots: centroide espectral, rolloff, bandwidth vs tempo."""
        import librosa

        fig = Figure(figsize=(14, 8))

        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        times = librosa.times_like(S[0], sr=self.sr, hop_length=512)

        # Spectral Centroid
        ax1 = fig.add_subplot(3, 1, 1)
        centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
        ax1.plot(times, centroid, color=ForensicTheme.ACCENT_CYAN, linewidth=1)
        ax1.fill_between(times, centroid, alpha=0.2, color=ForensicTheme.ACCENT_CYAN)
        ax1.set_title('Centroide Espectral')
        ax1.set_ylabel('Hz')

        # Spectral Rolloff
        ax2 = fig.add_subplot(3, 1, 2)
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sr, roll_percent=0.85)[0]
        rolloff_95 = librosa.feature.spectral_rolloff(S=S, sr=self.sr, roll_percent=0.95)[0]
        ax2.plot(times, rolloff, color=ForensicTheme.ACCENT_ORANGE, linewidth=1, label='85%')
        ax2.plot(times, rolloff_95, color=ForensicTheme.ACCENT_RED, linewidth=1,
                 alpha=0.7, label='95%')
        ax2.fill_between(times, rolloff, rolloff_95, alpha=0.15,
                         color=ForensicTheme.ACCENT_ORANGE)
        ax2.set_title('Rolloff Espectral')
        ax2.set_ylabel('Hz')
        ax2.legend(facecolor=ForensicTheme.BG_COLOR,
                   labelcolor=ForensicTheme.TEXT_COLOR)

        # Spectral Bandwidth
        ax3 = fig.add_subplot(3, 1, 3)
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=self.sr)[0]
        ax3.plot(times, bandwidth, color=ForensicTheme.ACCENT_GREEN, linewidth=1)
        ax3.fill_between(times, bandwidth, alpha=0.2, color=ForensicTheme.ACCENT_GREEN)
        ax3.set_title('Bandwidth Espectral')
        ax3.set_xlabel('Tempo (s)')
        ax3.set_ylabel('Hz')

        fig.suptitle('Features Espectrais Temporais', fontsize=13,
                     color=ForensicTheme.TEXT_COLOR, fontweight='bold')
        ForensicTheme.apply(fig)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # -----------------------------------------------------------------
    # 7. Energy Contour with Silence Detection
    # -----------------------------------------------------------------
    def plot_energy_contour(self, y: np.ndarray,
                            silence_threshold: float = 0.01) -> Figure:
        """RMS energy com regioes de silencio sombreadas."""
        import librosa

        fig = Figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        times = librosa.times_like(rms, sr=self.sr, hop_length=512)

        # Energy contour
        ax.plot(times, rms, color=ForensicTheme.ACCENT_CYAN, linewidth=1.2)
        ax.fill_between(times, rms, alpha=0.25, color=ForensicTheme.ACCENT_CYAN)

        # Silence regions
        silence_mask = rms < silence_threshold
        ax.fill_between(times, 0, np.max(rms),
                        where=silence_mask, alpha=0.2,
                        color=ForensicTheme.ACCENT_RED, label='Silencio')

        # Threshold line
        ax.axhline(y=silence_threshold, color=ForensicTheme.ACCENT_ORANGE,
                   linestyle='--', linewidth=1, alpha=0.7, label=f'Threshold ({silence_threshold})')

        # Statistics
        mean_rms = np.mean(rms)
        ax.axhline(y=mean_rms, color=ForensicTheme.ACCENT_GREEN,
                   linestyle=':', linewidth=1, alpha=0.7, label=f'Media ({mean_rms:.4f})')

        ax.set_title('Contorno de Energia (RMS)')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('RMS')
        ax.legend(loc='upper right', facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR, fontsize=9)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 8. HNR Over Time
    # -----------------------------------------------------------------
    def plot_hnr_temporal(self, y: np.ndarray) -> Figure:
        """Harmonic-to-Noise Ratio por frame."""
        fig = Figure(figsize=(12, 4))
        ax = fig.add_subplot(111)

        times, hnr_values = hnr_contour(y, self.sr)

        # Color by value
        ax.fill_between(times, hnr_values, alpha=0.3, color=ForensicTheme.ACCENT_GREEN)
        ax.plot(times, hnr_values, color=ForensicTheme.ACCENT_GREEN, linewidth=1)

        # Thresholds
        ax.axhline(y=20, color=ForensicTheme.ACCENT_CYAN, linestyle='--',
                   alpha=0.5, label='HNR=20 dB (voz limpa)')
        ax.axhline(y=7, color=ForensicTheme.ACCENT_ORANGE, linestyle='--',
                   alpha=0.5, label='HNR=7 dB (voz ruidosa)')

        ax.set_title('Harmonic-to-Noise Ratio (HNR) Temporal')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('HNR (dB)')
        ax.legend(loc='upper right', facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR, fontsize=9)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 9. Jitter/Shimmer Analysis
    # -----------------------------------------------------------------
    def plot_jitter_shimmer(self, y: np.ndarray) -> Figure:
        """Analise de jitter e shimmer — micro-perturbacoes vocais."""
        fig = Figure(figsize=(14, 5))

        try:
            medida = jitter_shimmer(y, self.sr)

            if medida is None:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'Insuficientes frames vocais para analise',
                        ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                        transform=ax.transAxes)
                ForensicTheme.apply(fig)
                return fig

            # Left: Jitter
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(medida.jitter_times, medida.jitter,
                     color=ForensicTheme.ACCENT_CYAN, linewidth=0.8, alpha=0.7)
            jitter_mean = medida.jitter_mean
            ax1.axhline(y=jitter_mean, color=ForensicTheme.ACCENT_ORANGE,
                        linestyle='--', label=f'Media: {jitter_mean:.2f}%')
            ax1.axhline(y=1.04, color=ForensicTheme.ACCENT_RED,
                        linestyle=':', alpha=0.5, label='Limiar normal (1.04%)')
            ax1.set_title(f'Jitter Local ({jitter_mean:.2f}%)')
            ax1.set_xlabel('Tempo (s)')
            ax1.set_ylabel('Jitter (%)')
            ax1.legend(facecolor=ForensicTheme.BG_COLOR,
                       labelcolor=ForensicTheme.TEXT_COLOR, fontsize=8)

            # Right: Shimmer
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(medida.shimmer_times, medida.shimmer,
                     color=ForensicTheme.ACCENT_GREEN, linewidth=0.8, alpha=0.7)
            shimmer_mean = medida.shimmer_mean
            ax2.axhline(y=shimmer_mean, color=ForensicTheme.ACCENT_ORANGE,
                        linestyle='--', label=f'Media: {shimmer_mean:.2f}%')
            ax2.axhline(y=3.81, color=ForensicTheme.ACCENT_RED,
                        linestyle=':', alpha=0.5, label='Limiar normal (3.81%)')
            ax2.set_title(f'Shimmer Local ({shimmer_mean:.2f}%)')
            ax2.set_xlabel('Tempo (s)')
            ax2.set_ylabel('Shimmer (%)')
            ax2.legend(facecolor=ForensicTheme.BG_COLOR,
                       labelcolor=ForensicTheme.TEXT_COLOR, fontsize=8)

        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Erro na analise jitter/shimmer:\n{e}',
                    ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                    transform=ax.transAxes)

        fig.suptitle('Análise de microperturbações vocais',
                     color=ForensicTheme.TEXT_COLOR, fontweight='bold')
        ForensicTheme.apply(fig)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    # -----------------------------------------------------------------
    # 10. F0 Detailed Analysis
    # -----------------------------------------------------------------
    def plot_f0_detailed(self, y: np.ndarray) -> Figure:
        """Pitch contour (pyin) com banda de confianca e vibrato."""
        fig = Figure(figsize=(14, 6))

        try:
            pitch = pitch_analysis(y, self.sr)
            times, f0, voiced_mask = pitch.times, pitch.f0, pitch.voiced_mask

            # Main pitch contour
            ax1 = fig.add_subplot(2, 1, 1)

            # Plot with confidence coloring
            ax1.scatter(times[voiced_mask], f0[voiced_mask],
                        c=pitch.voiced_probs[voiced_mask], cmap='RdYlGn',
                        s=3, alpha=0.8, vmin=0, vmax=1)
            ax1.plot(times, f0, color=ForensicTheme.ACCENT_CYAN,
                     linewidth=0.8, alpha=0.5)

            # Statistics
            if pitch.f0_mean is not None:
                f0_mean, f0_std = pitch.f0_mean, pitch.f0_std
                ax1.axhline(y=f0_mean, color=ForensicTheme.ACCENT_ORANGE,
                            linestyle='--', alpha=0.7,
                            label=f'F0 medio: {f0_mean:.1f} Hz')
                ax1.fill_between(times, f0_mean - f0_std, f0_mean + f0_std,
                                 alpha=0.1, color=ForensicTheme.ACCENT_ORANGE)

            ax1.set_title('Contorno de pitch (F0) com confiança')
            ax1.set_ylabel('Frequência (Hz)')
            ax1.legend(facecolor=ForensicTheme.BG_COLOR,
                       labelcolor=ForensicTheme.TEXT_COLOR, fontsize=9)

            # Vibrato analysis (bottom)
            ax2 = fig.add_subplot(2, 1, 2)

            if pitch.f0_detrended is not None:
                f0_detrended = pitch.f0_detrended
                ax2.plot(times, f0_detrended, color=ForensicTheme.ACCENT_GREEN,
                         linewidth=0.8, alpha=0.8)
                ax2.fill_between(times, f0_detrended, alpha=0.2,
                                 color=ForensicTheme.ACCENT_GREEN)

                # Vibrato extent
                vibrato_extent = pitch.vibrato_extent
                ax2.axhline(y=vibrato_extent, color=ForensicTheme.ACCENT_ORANGE,
                            linestyle='--', alpha=0.5,
                            label=f'Extent: +/-{vibrato_extent:.1f} Hz')
                ax2.axhline(y=-vibrato_extent, color=ForensicTheme.ACCENT_ORANGE,
                            linestyle='--', alpha=0.5)

            ax2.set_title('Análise de vibrato (flutuação de F0)')
            ax2.set_xlabel('Tempo (s)')
            ax2.set_ylabel('Delta F0 (Hz)')
            ax2.legend(facecolor=ForensicTheme.BG_COLOR,
                       labelcolor=ForensicTheme.TEXT_COLOR, fontsize=9)

        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Erro na analise de pitch:\n{e}',
                    ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                    transform=ax.transAxes)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 11. Feature Importance Radar Chart
    # -----------------------------------------------------------------
    def plot_feature_importance_radar(self, feature_names: List[str],
                                      importances: np.ndarray) -> Figure:
        """Grafico radar/spider de importancia de features na deteccao."""
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        N = len(feature_names)
        if N == 0:
            ax.text(0, 0, 'Sem dados de importancia',
                    ha='center', va='center', color=ForensicTheme.TEXT_COLOR)
            return fig

        # Normaliza SOMENTE se o chamador ainda nao o fez.
        #
        # Dividir pelo maximo global mistura grandezas de unidades diferentes:
        # um eixo em unidades cepstrais (dezenas) achata para perto de zero
        # todos os que vivem em [0, 1]. Quando os valores ja chegam dentro de
        # [0, 1] — normalizados por eixo contra a faixa plausivel de cada
        # grandeza —, reescalar de novo pelo maximo distorceria a leitura,
        # empurrando o maior eixo para 1,0 mesmo quando ele esta longe do
        # extremo da sua propria escala.
        importances = np.asarray(importances, dtype=float)
        maximo = float(np.max(importances)) if importances.size else 0.0
        if maximo > 1.0:
            importances = importances / maximo

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        importances_plot = importances.tolist()

        # Close the polygon
        angles += angles[:1]
        importances_plot += importances_plot[:1]

        ax.plot(angles, importances_plot, color=ForensicTheme.ACCENT_CYAN,
                linewidth=2)
        ax.fill(angles, importances_plot, color=ForensicTheme.ACCENT_CYAN,
                alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, fontsize=8,
                           color=ForensicTheme.TEXT_COLOR)
        ax.set_ylim(0, 1.1)

        # Style
        ax.set_facecolor(ForensicTheme.BG_COLOR_LIGHT)
        ax.spines['polar'].set_color(ForensicTheme.GRID_COLOR)
        ax.grid(color=ForensicTheme.GRID_COLOR, alpha=0.3)
        ax.tick_params(axis='y', colors=ForensicTheme.TEXT_COLOR)

        fig.patch.set_facecolor(ForensicTheme.BG_COLOR)
        ax.set_title('Importância relativa das características',
                     color=ForensicTheme.TEXT_COLOR, fontweight='bold', pad=20)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 12. Anomaly Heatmap
    # -----------------------------------------------------------------
    def plot_anomaly_heatmap(self, y: np.ndarray,
                             segment_scores: np.ndarray,
                             segment_times: np.ndarray) -> Figure:
        """Espectrograma com overlay de anomalias colorido."""
        import librosa

        fig = Figure(figsize=(14, 6))
        ax = fig.add_subplot(111)

        # Base spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        ax.imshow(S_db, aspect='auto', origin='lower',
                  cmap='gray', alpha=0.6,
                  extent=[0, len(y) / self.sr, 0, self.sr / 2])

        # Overlay anomaly scores as vertical bands
        anomaly_cmap = ForensicTheme.get_anomaly_cmap()
        for i in range(len(segment_scores)):
            t_start = segment_times[i]
            t_end = segment_times[i + 1] if i + 1 < len(segment_times) else len(y) / self.sr
            score = np.clip(segment_scores[i], 0, 1)
            color = anomaly_cmap(score)
            ax.axvspan(t_start, t_end, alpha=0.35, color=color)

        # Color bar reference
        import matplotlib.cm as cm
        try:
            from matplotlib.colors import Normalize
            sm = cm.ScalarMappable(cmap=anomaly_cmap, norm=Normalize(0, 1))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='Score de Anomalia', fraction=0.046)
        except Exception:
            pass

        ax.set_title('Heatmap de Anomalias sobre Espectrograma')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Frequência (Hz)')

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 13. Confidence Timeline
    # -----------------------------------------------------------------
    def plot_fake_score_timeline(self, segment_times: np.ndarray,
                                 segment_scores: np.ndarray,
                                 threshold: float = 0.5) -> Figure:
        """Pontuação por segmento, com o limite de decisão."""
        fig = Figure(figsize=(12, 4))
        ax = fig.add_subplot(111)

        ax.plot(segment_times, segment_scores,
                color=ForensicTheme.ACCENT_CYAN, linewidth=2, marker='o',
                markersize=4)
        ax.fill_between(segment_times, segment_scores,
                        alpha=0.2, color=ForensicTheme.ACCENT_CYAN)

        # Threshold line
        ax.axhline(y=threshold, color=ForensicTheme.ACCENT_RED, linestyle='--',
                   linewidth=1.5, alpha=0.8,
                   label=f'Limite de decisão ({threshold:.0%})')

        # Color regions (a decisão do Predictor é `p_fake >= limiar`)
        above = segment_scores >= threshold
        ax.fill_between(segment_times, threshold, segment_scores,
                        where=above, alpha=0.15, color=ForensicTheme.ACCENT_RED,
                        label='Sintética (>= limite)')
        ax.fill_between(segment_times, segment_scores, threshold,
                        where=~above, alpha=0.15, color=ForensicTheme.ACCENT_GREEN,
                        label='Real (< limite)')

        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Pontuação de voz sintética por segmento')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Pontuação de voz sintética')
        ax.legend(loc='upper right', facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR, fontsize=9)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig


# =============================================================================
# Training Analytics Visualizer
# =============================================================================

class TrainingAnalyticsVisualizer:
    """Visualizacoes avancadas para analise de treinamento."""

    # -----------------------------------------------------------------
    # 1. Learning Rate Schedule
    # -----------------------------------------------------------------
    def plot_lr_schedule(self, lr_history: List[float]) -> Figure:
        """Learning rate ao longo das epocas."""
        fig = Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)

        epochs = list(range(1, len(lr_history) + 1))
        ax.plot(epochs, lr_history, color=ForensicTheme.ACCENT_CYAN,
                linewidth=2, marker='o', markersize=3)
        ax.set_yscale('log')
        ax.set_title('Schedule de Learning Rate')
        ax.set_xlabel('Epoca')
        ax.set_ylabel('Learning Rate')

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 2. DET Curve + EER
    # -----------------------------------------------------------------
    def plot_det_curve_eer(self, y_true: np.ndarray,
                           y_scores: np.ndarray) -> Tuple[Figure, float]:
        """Curva DET (FNR vs FPR) com ponto EER. Retorna (Figure, eer_value)."""
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        fpr, fnr, eer = det_curve(y_true, y_scores)

        ax.plot(fpr, fnr, color=ForensicTheme.ACCENT_CYAN, linewidth=2,
                label=f'DET Curve (EER = {eer * 100:.2f}%)')
        ax.plot([0, 1], [0, 1], color=ForensicTheme.GRID_COLOR,
                linestyle='--', linewidth=1)
        ax.plot(eer, eer, 'o', color=ForensicTheme.ACCENT_RED,
                markersize=10, label='EER Point')

        ax.set_title('Detection Error Tradeoff (DET) Curve')
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('False Negative Rate (FNR)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig, eer

    # -----------------------------------------------------------------
    # 3. Threshold Optimization
    # -----------------------------------------------------------------
    def plot_threshold_optimization(self, y_true: np.ndarray,
                                    y_scores: np.ndarray) -> Figure:
        """Precision, Recall, F1 plotados vs threshold."""
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        varredura = threshold_sweep(y_true, y_scores)
        thresholds = varredura.thresholds

        ax.plot(thresholds, varredura.precision, color=ForensicTheme.ACCENT_CYAN,
                linewidth=2, label='Precisão')
        ax.plot(thresholds, varredura.recall, color=ForensicTheme.ACCENT_ORANGE,
                linewidth=2, label='Revocação')
        ax.plot(thresholds, varredura.f1, color=ForensicTheme.ACCENT_GREEN,
                linewidth=2.5, label='F1-Score')

        # Best F1 threshold
        best_threshold = varredura.best_threshold
        ax.axvline(x=best_threshold, color=ForensicTheme.ACCENT_RED,
                   linestyle='--', alpha=0.7,
                   label=f'Melhor F1 @ {best_threshold:.2f}')

        ax.set_title('Otimização do limiar de decisão')
        ax.set_xlabel('Limiar de decisão')
        ax.set_ylabel('Métrica')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig


# =============================================================================
# Batch Analysis Visualizer
# =============================================================================

class BatchAnalysisVisualizer:
    """Visualizacoes de resultados de analise em lote."""

    # -----------------------------------------------------------------
    # 1. Batch Summary Pie
    # -----------------------------------------------------------------
    def plot_batch_summary_pie(self, n_real: int, n_fake: int) -> Figure:
        """Grafico pizza real vs fake."""
        fig = Figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        total = n_real + n_fake
        if total == 0:
            ax.text(0.5, 0.5, 'Sem resultados',
                    ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                    transform=ax.transAxes)
            fig.patch.set_facecolor(ForensicTheme.BG_COLOR)
            return fig

        sizes = [n_real, n_fake]
        labels = [f'Real\n({n_real})', f'Fake\n({n_fake})']
        colors = [ForensicTheme.ACCENT_GREEN, ForensicTheme.ACCENT_RED]
        explode = (0, 0.05)

        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'color': ForensicTheme.TEXT_COLOR, 'fontsize': 12}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title(f'Distribuição de resultados (total: {total})',
                     color=ForensicTheme.TEXT_COLOR, fontweight='bold')
        fig.patch.set_facecolor(ForensicTheme.BG_COLOR)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 2. Confidence Distribution
    # -----------------------------------------------------------------
    def plot_fake_score_distribution(self, fake_scores: np.ndarray,
                                     threshold: float = 0.5) -> Figure:
        """Histograma da pontuação, com o limite de decisão."""
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        if len(fake_scores) == 0:
            ax.text(0.5, 0.5, 'Sem dados',
                    ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                    transform=ax.transAxes)
            ForensicTheme.apply(fig)
            return fig

        bins = np.linspace(0, 1, 30)
        n, bins_out, patches = ax.hist(fake_scores, bins=bins, edgecolor='black',
                                       alpha=0.8, linewidth=0.5)

        # Color bars by value
        anomaly_cmap = ForensicTheme.get_anomaly_cmap()
        for patch, left_edge in zip(patches, bins_out[:-1]):
            patch.set_facecolor(anomaly_cmap(left_edge))

        # Threshold line
        ax.axvline(x=threshold, color='white', linestyle='--', linewidth=2,
                   label=f'Limite de decisão ({threshold:.0%})')

        # Stats
        mean_score = np.mean(fake_scores)
        ax.axvline(x=mean_score, color=ForensicTheme.ACCENT_CYAN,
                   linestyle=':', linewidth=1.5,
                   label=f'Media: {mean_score:.3f}')

        ax.set_title('Distribuição da pontuação de voz sintética')
        ax.set_xlabel('Pontuação de voz sintética')
        ax.set_ylabel('Contagem')
        ax.legend(facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig
